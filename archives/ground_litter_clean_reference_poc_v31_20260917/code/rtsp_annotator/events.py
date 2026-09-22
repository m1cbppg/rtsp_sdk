from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class EventNotFoundError(KeyError):
    pass


@dataclass(frozen=True, slots=True)
class EventRuleOptions:
    person_dwell_seconds: float = 20.0
    vehicle_dwell_seconds: float = 20.0
    actor_leave_grace_seconds: float = 3.0
    actor_association_seconds: float = 60.0
    garbage_persistence_seconds: float = 15.0
    minimum_change_area: float = 0.002

    def validate(self) -> None:
        if self.person_dwell_seconds <= 0:
            raise ValueError("person_dwell_seconds必须大于0")
        if self.vehicle_dwell_seconds <= 0:
            raise ValueError("vehicle_dwell_seconds必须大于0")
        if not 0 <= self.actor_leave_grace_seconds <= 30:
            raise ValueError("actor_leave_grace_seconds必须在[0, 30]范围内")
        if not 1 <= self.actor_association_seconds <= 600:
            raise ValueError("actor_association_seconds必须在[1, 600]范围内")
        if not 1 <= self.garbage_persistence_seconds <= 300:
            raise ValueError("garbage_persistence_seconds必须在[1, 300]范围内")
        if not 0 < self.minimum_change_area <= 0.5:
            raise ValueError("minimum_change_area必须在(0, 0.5]范围内")


@dataclass(frozen=True, slots=True)
class EventRoiOptions:
    roi_id: str
    polygon: tuple[tuple[float, float], ...]
    dwell_enabled: bool = True
    garbage_enabled: bool = True
    rules: EventRuleOptions = EventRuleOptions()

    def validate(self) -> None:
        if not _SAFE_IDENTIFIER.fullmatch(self.roi_id):
            raise ValueError(
                "事件ROI id只能包含字母、数字、下划线和短横线"
            )
        if len(self.polygon) < 3:
            raise ValueError("事件ROI至少需要3个顶点")
        for x, y in self.polygon:
            if not 0 <= x <= 1 or not 0 <= y <= 1:
                raise ValueError("事件ROI坐标必须在[0, 1]范围内")
        self.rules.validate()


@dataclass(frozen=True, slots=True)
class GarbageAnalysisOptions:
    enabled: bool = False
    analysis_fps: float = 3.0
    minimum_confidence: float = 0.35
    detection_mode: str = "items"
    background_change_enabled: bool = True
    display_detections: bool = True
    display_hold_seconds: float = 1.5
    maximum_display_boxes: int = 20
    minimum_pile_detections: int = 2
    pile_merge_distance: float = 0.18
    pile_box_padding: float = 0.04
    prompts: tuple[str, ...] = (
        "plastic bottle",
        "garbage bag",
        "plastic bag",
        "cardboard box",
        "paper waste",
        "can",
        "trash pile",
        "waste",
    )

    def validate(self) -> None:
        if not 0.1 <= self.analysis_fps <= 10:
            raise ValueError("garbage.analysis_fps必须在[0.1, 10]范围内")
        if not 0 < self.minimum_confidence <= 1:
            raise ValueError("garbage.minimum_confidence必须在(0, 1]范围内")
        if self.detection_mode not in {"items", "pile"}:
            raise ValueError("garbage.detection_mode必须是items或pile")
        if not 0.3 <= self.display_hold_seconds <= 10:
            raise ValueError(
                "garbage.display_hold_seconds必须在[0.3, 10]范围内"
            )
        if not 1 <= self.maximum_display_boxes <= 100:
            raise ValueError(
                "garbage.maximum_display_boxes必须在[1, 100]范围内"
            )
        if not 1 <= self.minimum_pile_detections <= 20:
            raise ValueError(
                "garbage.minimum_pile_detections必须在[1, 20]范围内"
            )
        if not 0.01 <= self.pile_merge_distance <= 0.5:
            raise ValueError(
                "garbage.pile_merge_distance必须在[0.01, 0.5]范围内"
            )
        if not 0 <= self.pile_box_padding <= 0.25:
            raise ValueError(
                "garbage.pile_box_padding必须在[0, 0.25]范围内"
            )
        if not self.prompts or any(not item.strip() for item in self.prompts):
            raise ValueError("garbage.prompts不能为空")


@dataclass(frozen=True, slots=True)
class WebhookOptions:
    url: str | None = None
    timeout_seconds: float = 3.0

    def validate(self) -> None:
        if self.url is not None and not self.url.startswith(("http://", "https://")):
            raise ValueError("webhook.url必须以http://或https://开头")
        if not 0.1 <= self.timeout_seconds <= 30:
            raise ValueError("webhook.timeout_seconds必须在[0.1, 30]范围内")


@dataclass(frozen=True, slots=True)
class EventDetectionOptions:
    enabled: bool = False
    rois: tuple[EventRoiOptions, ...] = ()
    garbage: GarbageAnalysisOptions = GarbageAnalysisOptions()
    webhook: WebhookOptions = WebhookOptions()
    person_classes: tuple[int, ...] = (0,)
    vehicle_classes: tuple[int, ...] = (2, 3, 5, 7)

    def validate(self) -> None:
        if self.enabled and not self.rois:
            raise ValueError("启用事件识别时至少需要一个ROI")
        if self.garbage.enabled and not self.enabled:
            raise ValueError("启用垃圾分析前必须启用event_detection")
        roi_ids = [item.roi_id for item in self.rois]
        if len(set(roi_ids)) != len(roi_ids):
            raise ValueError("事件ROI id不能重复")
        for roi in self.rois:
            roi.validate()
        self.garbage.validate()
        self.webhook.validate()
        actor_classes = self.person_classes + self.vehicle_classes
        if not actor_classes or any(item < 0 for item in actor_classes):
            raise ValueError("事件人物和车辆类别必须是非负类别ID")
        if set(self.person_classes) & set(self.vehicle_classes):
            raise ValueError("person_classes与vehicle_classes不能重叠")

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "person_classes": list(self.person_classes),
            "vehicle_classes": list(self.vehicle_classes),
            "rois": [
                {
                    "id": roi.roi_id,
                    "polygon": [list(point) for point in roi.polygon],
                    "dwell_enabled": roi.dwell_enabled,
                    "garbage_enabled": roi.garbage_enabled,
                    "rules": asdict(roi.rules),
                }
                for roi in self.rois
            ],
            "garbage": {
                "enabled": self.garbage.enabled,
                "analysis_fps": self.garbage.analysis_fps,
                "minimum_confidence": self.garbage.minimum_confidence,
                "detection_mode": self.garbage.detection_mode,
                "background_change_enabled": (
                    self.garbage.background_change_enabled
                ),
                "display_detections": self.garbage.display_detections,
                "display_hold_seconds": self.garbage.display_hold_seconds,
                "maximum_display_boxes": self.garbage.maximum_display_boxes,
                "minimum_pile_detections": (
                    self.garbage.minimum_pile_detections
                ),
                "pile_merge_distance": self.garbage.pile_merge_distance,
                "pile_box_padding": self.garbage.pile_box_padding,
                "prompts": list(self.garbage.prompts),
            },
            "webhook": asdict(self.webhook),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "EventDetectionOptions":
        data = payload or {}
        rois = tuple(
            EventRoiOptions(
                roi_id=str(item["id"]),
                polygon=tuple(
                    (float(point[0]), float(point[1]))
                    for point in item["polygon"]
                ),
                dwell_enabled=bool(item.get("dwell_enabled", True)),
                garbage_enabled=bool(item.get("garbage_enabled", True)),
                rules=EventRuleOptions(**dict(item.get("rules") or {})),
            )
            for item in data.get("rois", [])
        )
        garbage_data = dict(data.get("garbage") or {})
        if "prompts" in garbage_data:
            garbage_data["prompts"] = tuple(garbage_data["prompts"])
        webhook_data = dict(data.get("webhook") or {})
        options = cls(
            enabled=bool(data.get("enabled", False)),
            rois=rois,
            garbage=GarbageAnalysisOptions(**garbage_data),
            webhook=WebhookOptions(**webhook_data),
            person_classes=tuple(data.get("person_classes", (0,))),
            vehicle_classes=tuple(
                data.get("vehicle_classes", (2, 3, 5, 7))
            ),
        )
        options.validate()
        return options


@dataclass(slots=True)
class EventRecord:
    event_id: str
    stream_id: str
    event_type: str
    roi_id: str
    occurred_at: str
    message: str
    status: str = "pending"
    actor_type: str | None = None
    actor_track_id: int | None = None
    object_type: str | None = None
    confidence: float | None = None
    snapshot_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        stream_id: str,
        event_type: str,
        roi_id: str,
        message: str,
        actor_type: str | None = None,
        actor_track_id: int | None = None,
        object_type: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EventRecord":
        return cls(
            event_id=uuid.uuid4().hex,
            stream_id=stream_id,
            event_type=event_type,
            roi_id=roi_id,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            message=message,
            actor_type=actor_type,
            actor_track_id=actor_track_id,
            object_type=object_type,
            confidence=confidence,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventRepository:
    """Atomic file-backed event store shared by API and worker processes."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def append(self, event: EventRecord) -> None:
        stream_dir = self._stream_dir(event.stream_id)
        stream_dir.mkdir(parents=True, exist_ok=True)
        target = stream_dir / f"{event.event_id}.json"
        self._atomic_write(target, event.to_dict())

    def list(
        self,
        *,
        stream_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1_000))
        if stream_id is not None:
            directories: Iterable[Path] = (self._stream_dir(stream_id),)
        else:
            directories = (
                item for item in self.root.iterdir() if item.is_dir()
            )
        records: list[dict[str, Any]] = []
        for directory in directories:
            if not directory.is_dir():
                continue
            for path in directory.glob("*.json"):
                payload = self._read(path)
                if payload is not None:
                    records.append(payload)
        records.sort(
            key=lambda item: str(item.get("occurred_at", "")),
            reverse=True,
        )
        return records[:limit]

    def get(self, event_id: str) -> dict[str, Any]:
        self._validate_identifier(event_id, "event_id")
        for stream_dir in self.root.iterdir():
            if not stream_dir.is_dir():
                continue
            path = stream_dir / f"{event_id}.json"
            payload = self._read(path)
            if payload is not None:
                return payload
        raise EventNotFoundError(event_id)

    def review(self, event_id: str, status: str) -> dict[str, Any]:
        if status not in {"confirmed", "rejected"}:
            raise ValueError("事件审核状态必须是confirmed或rejected")
        with self._lock:
            payload = self.get(event_id)
            payload["status"] = status
            payload["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            path = self._stream_dir(str(payload["stream_id"])) / f"{event_id}.json"
            self._atomic_write(path, payload)
        return payload

    def media_path(self, event_id: str, field_name: str) -> Path:
        if field_name != "snapshot_path":
            raise ValueError("不支持的事件媒体字段")
        payload = self.get(event_id)
        value = payload.get(field_name)
        if not value:
            raise EventNotFoundError(event_id)
        path = Path(str(value)).expanduser().resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise EventNotFoundError(event_id) from exc
        if not path.is_file():
            raise EventNotFoundError(event_id)
        return path

    def _stream_dir(self, stream_id: str) -> Path:
        self._validate_identifier(stream_id, "stream_id")
        return self.root / stream_id

    @staticmethod
    def _validate_identifier(value: str, name: str) -> None:
        if not _SAFE_IDENTIFIER.fullmatch(value):
            raise ValueError(f"{name}格式无效")

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
