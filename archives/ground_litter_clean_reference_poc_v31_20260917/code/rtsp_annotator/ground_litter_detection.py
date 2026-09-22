"""Per-stream ground-litter detection options for the DeepStream side path.

This module is the API-facing half of the 零散垃圾识别 scenario.  It keeps the
native-pixel tiling behaviour that the offline pilot proved useful (a 640 tile
taken from the original 2560x1440 frame, never a rescaled whole frame), and it
adds a display layer so a low-rate analysis does not flicker on the published
RTSP stream.

The detector only produces boxes.  Whether a box is really abandoned litter
remains a human-review question, so the OSD label defaults to 疑似垃圾 rather
than a material class such as Plastic.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np

from .event_engine import NormalizedRect
from .ground_litter_geometry import box_overlap_fraction, prepare

_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9._/-]+\.pt$")
DEFAULT_MODEL = "turhancan_yolov8m_seg_trash.pt"
DEFAULT_MODEL_SUBDIR = "litter"
PERSON_VEHICLE_CLASS_IDS = (0, 1, 2, 3, 5, 7)
# COCO context classes that can occlude the ground (umbrella, bench, chair,
# plant, dining table). They are kept separate from person/vehicle filtering:
# a candidate under one is considered occluded, not permanently non-litter.
GROUND_CONTEXT_CLASS_IDS = (13, 25, 56, 58, 60)
MAXIMUM_ZONES = 16
MAXIMUM_TILES = 128


def _validate_polygon(polygon: Any, field_name: str) -> None:
    if len(polygon) < 3:
        raise ValueError(f"{field_name}至少需要3个顶点")
    for point in polygon:
        if len(point) != 2:
            raise ValueError(f"{field_name}顶点必须是[x, y]")
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"{field_name}顶点必须是有限数值")
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"{field_name}顶点必须在[0, 1]范围内")


@dataclass(frozen=True, slots=True)
class GroundLitterZone:
    """One reviewed ground region (usually one shop frontage)."""

    region_id: str
    polygon: tuple[tuple[float, float], ...]
    name: str = ""
    exclude_zones: tuple[tuple[tuple[float, float], ...], ...] = ()
    minimum_short_side_px: int = 12
    minimum_box_area_px: int = 160
    confidence: float | None = None
    night_confidence: float | None = None

    def confidence_for(self, night: bool, default: float) -> float:
        """A missing override inherits the stream threshold."""
        value = self.night_confidence if night else None
        if value is None:
            value = self.confidence
        return float(default if value is None else value)

    def validate(self) -> None:
        if not self.region_id or not isinstance(self.region_id, str):
            raise ValueError("ground_litter.zones[].region_id不能为空")
        _validate_polygon(self.polygon, "ground_litter.zones[].polygon")
        for index, exclusion in enumerate(self.exclude_zones):
            _validate_polygon(
                exclusion,
                f"ground_litter.zones[].exclude_zones[{index}]",
            )
        if not 1 <= int(self.minimum_short_side_px) <= 4096:
            raise ValueError(
                "ground_litter最小短边像素必须在[1, 4096]范围内"
            )
        if not 1 <= int(self.minimum_box_area_px) <= 16_777_216:
            raise ValueError(
                "ground_litter最小框面积必须在[1, 16777216]范围内"
            )
        for name, value in (("confidence", self.confidence), ("night_confidence", self.night_confidence)):
            if value is not None and not 0 < float(value) <= 1:
                raise ValueError(f"ground_litter.zones[].{name}必须在(0, 1]范围内")

    def to_payload(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "name": self.name,
            "polygon": [list(point) for point in self.polygon],
            "exclude_zones": [
                [list(point) for point in polygon]
                for polygon in self.exclude_zones
            ],
            "minimum_short_side_px": self.minimum_short_side_px,
            "minimum_box_area_px": self.minimum_box_area_px,
            "confidence": self.confidence,
            "night_confidence": self.night_confidence,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GroundLitterZone":
        values = dict(payload or {})
        values["polygon"] = tuple(
            (float(point[0]), float(point[1]))
            for point in values.get("polygon", ())
        )
        values["exclude_zones"] = tuple(
            tuple(
                (float(point[0]), float(point[1])) for point in polygon
            )
            for polygon in values.get("exclude_zones", ())
        )
        zone = cls(**values)
        zone.validate()
        return zone


@dataclass(frozen=True, slots=True)
class GroundLitterDetectionOptions:
    """Side-path litter detection for one stream.

    ``model`` may be a bare ``.pt`` name (resolved inside the ``litter``
    sub-directory first, then the model root) or a relative path such as
    ``litter/turhancan_yolov8m_seg_trash.pt``.
    """

    enabled: bool = False
    model: str = DEFAULT_MODEL
    actor_model: str | None = None
    analysis_fps: float = 1.0
    confidence: float = 0.20
    night_confidence: float | None = None
    tile_size_px: int = 640
    inference_imgsz: int | None = None
    local_actor_max_crops: int = 0
    box_smoothing_alpha: float = 1.0
    tile_overlap: float = 0.20
    nms_iou: float = 0.50
    maximum_tiles: int = 64
    actor_imgsz: int = 1280
    actor_confidence: float = 0.20
    actor_class_ids: tuple[int, ...] = PERSON_VEHICLE_CLASS_IDS
    context_class_ids: tuple[int, ...] = GROUND_CONTEXT_CLASS_IDS
    actor_overlap_threshold: float = 0.20
    zones: tuple[GroundLitterZone, ...] = ()
    overlay_exclude_zones: tuple[tuple[tuple[float, float], ...], ...] = ()
    minimum_hits: int = 2
    hit_window: int = 3
    hold_seconds: float = 3.0
    maximum_age_seconds: float = 12.0
    match_size_ratio: float = 3.0
    match_distance_ratio: float = 0.5
    maximum_boxes: int = 8
    display_zones: bool = True
    display_class: bool = False
    display_confidence: bool = False
    label: str = "疑似垃圾"

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(zone.region_id for zone in self.zones)

    @property
    def effective_imgsz(self) -> int:
        return int(self.inference_imgsz or self.tile_size_px)

    def confidence_for(self, night: bool) -> float:
        if night and self.night_confidence is not None:
            return float(self.night_confidence)
        return float(self.confidence)

    def validate(self) -> None:
        if self.model is None or not _SAFE_MODEL_NAME.match(str(self.model)):
            raise ValueError("ground_litter.model必须是models目录中的.pt文件")
        if str(self.model).startswith("/") or ".." in str(self.model):
            raise ValueError("ground_litter.model不能是绝对路径或越界路径")
        if self.actor_model is not None and not _SAFE_MODEL_NAME.match(
            str(self.actor_model)
        ):
            raise ValueError(
                "ground_litter.actor_model必须是models目录中的.pt文件"
            )
        if not 0.1 <= float(self.analysis_fps) <= 5:
            raise ValueError(
                "ground_litter.analysis_fps必须在[0.1, 5]范围内"
            )
        if not 0 < float(self.confidence) <= 1:
            raise ValueError(
                "ground_litter.confidence必须在(0, 1]范围内"
            )
        if self.night_confidence is not None and not (
            0 < float(self.night_confidence) <= 1
        ):
            raise ValueError(
                "ground_litter.night_confidence必须在(0, 1]范围内"
            )
        if not 160 <= int(self.tile_size_px) <= 1920:
            raise ValueError(
                "ground_litter.tile_size_px必须在[160, 1920]范围内"
            )
        if not 0 <= float(self.tile_overlap) < 0.8:
            raise ValueError(
                "ground_litter.tile_overlap必须在[0, 0.8)范围内"
            )
        if self.inference_imgsz is not None and not 160 <= self.inference_imgsz <= 1920:
            raise ValueError("ground_litter.inference_imgsz必须在[160, 1920]范围内")
        if not 0 <= self.local_actor_max_crops <= 8:
            raise ValueError("ground_litter.local_actor_max_crops必须在[0, 8]范围内")
        if self.local_actor_max_crops and self.actor_model is None:
            raise ValueError("ground_litter.local_actor_max_crops需要actor_model")
        if not 0 < self.box_smoothing_alpha <= 1:
            raise ValueError("ground_litter.box_smoothing_alpha必须在(0, 1]范围内")
        if not 0 < float(self.nms_iou) <= 1:
            raise ValueError("ground_litter.nms_iou必须在(0, 1]范围内")
        if not 1 <= int(self.maximum_tiles) <= MAXIMUM_TILES:
            raise ValueError(
                f"ground_litter.maximum_tiles必须在[1, {MAXIMUM_TILES}]范围内"
            )
        if not 320 <= int(self.actor_imgsz) <= 1920:
            raise ValueError(
                "ground_litter.actor_imgsz必须在[320, 1920]范围内"
            )
        if not 0 < float(self.actor_confidence) <= 1:
            raise ValueError(
                "ground_litter.actor_confidence必须在(0, 1]范围内"
            )
        if any(int(item) < 0 for item in self.actor_class_ids):
            raise ValueError(
                "ground_litter.actor_class_ids必须是非负类别ID列表"
            )
        if any(int(item) < 0 for item in self.context_class_ids):
            raise ValueError("ground_litter.context_class_ids必须是非负类别ID列表")
        if not 0 <= float(self.actor_overlap_threshold) <= 1:
            raise ValueError(
                "ground_litter.actor_overlap_threshold必须在[0, 1]范围内"
            )
        if len(self.zones) > MAXIMUM_ZONES:
            raise ValueError(
                f"ground_litter.zones最多{MAXIMUM_ZONES}个区域"
            )
        if self.enabled and not self.zones:
            raise ValueError("启用ground_litter时至少需要一个地面区域")
        seen: set[str] = set()
        for zone in self.zones:
            zone.validate()
            if zone.region_id in seen:
                raise ValueError(
                    f"ground_litter.zones区域ID重复: {zone.region_id}"
                )
            seen.add(zone.region_id)
        for index, polygon in enumerate(self.overlay_exclude_zones):
            _validate_polygon(
                polygon,
                f"ground_litter.overlay_exclude_zones[{index}]",
            )
        if not 1 <= int(self.hit_window) <= 20:
            raise ValueError(
                "ground_litter.hit_window必须在[1, 20]范围内"
            )
        if not 1 <= int(self.minimum_hits) <= int(self.hit_window):
            raise ValueError(
                "ground_litter.minimum_hits必须在[1, hit_window]范围内"
            )
        if not 0.1 <= float(self.hold_seconds) <= 30:
            raise ValueError(
                "ground_litter.hold_seconds必须在[0.1, 30]范围内"
            )
        if not float(self.hold_seconds) <= float(
            self.maximum_age_seconds
        ) <= 120:
            raise ValueError(
                "ground_litter.maximum_age_seconds必须在"
                "[hold_seconds, 120]范围内"
            )
        if not 1 <= float(self.match_size_ratio) <= 10:
            raise ValueError(
                "ground_litter.match_size_ratio必须在[1, 10]范围内"
            )
        if not 0 < float(self.match_distance_ratio) <= 5:
            raise ValueError(
                "ground_litter.match_distance_ratio必须在(0, 5]范围内"
            )
        if not 1 <= int(self.maximum_boxes) <= 64:
            raise ValueError(
                "ground_litter.maximum_boxes必须在[1, 64]范围内"
            )
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("ground_litter.label不能为空")

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "actor_model": self.actor_model,
            "analysis_fps": self.analysis_fps,
            "confidence": self.confidence,
            "night_confidence": self.night_confidence,
            "tile_size_px": self.tile_size_px,
            "inference_imgsz": self.inference_imgsz,
            "local_actor_max_crops": self.local_actor_max_crops,
            "box_smoothing_alpha": self.box_smoothing_alpha,
            "tile_overlap": self.tile_overlap,
            "nms_iou": self.nms_iou,
            "maximum_tiles": self.maximum_tiles,
            "actor_imgsz": self.actor_imgsz,
            "actor_confidence": self.actor_confidence,
            "actor_class_ids": list(self.actor_class_ids),
            "context_class_ids": list(self.context_class_ids),
            "actor_overlap_threshold": self.actor_overlap_threshold,
            "zones": [zone.to_payload() for zone in self.zones],
            "overlay_exclude_zones": [
                [list(point) for point in polygon]
                for polygon in self.overlay_exclude_zones
            ],
            "minimum_hits": self.minimum_hits,
            "hit_window": self.hit_window,
            "hold_seconds": self.hold_seconds,
            "maximum_age_seconds": self.maximum_age_seconds,
            "match_size_ratio": self.match_size_ratio,
            "match_distance_ratio": self.match_distance_ratio,
            "maximum_boxes": self.maximum_boxes,
            "display_zones": self.display_zones,
            "display_class": self.display_class,
            "display_confidence": self.display_confidence,
            "label": self.label,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any] | None,
    ) -> "GroundLitterDetectionOptions":
        values = dict(payload or {})
        values["zones"] = tuple(
            GroundLitterZone.from_payload(item)
            for item in values.get("zones", ())
        )
        values["overlay_exclude_zones"] = tuple(
            tuple(
                (float(point[0]), float(point[1])) for point in polygon
            )
            for polygon in values.get("overlay_exclude_zones", ())
        )
        if "actor_class_ids" in values:
            values["actor_class_ids"] = tuple(
                int(item) for item in values["actor_class_ids"]
            )
        if "context_class_ids" in values:
            values["context_class_ids"] = tuple(
                int(item) for item in values["context_class_ids"]
            )
        options = cls(**values)
        options.validate()
        return options


@dataclass(frozen=True, slots=True)
class GroundLitterCandidate:
    rectangle: NormalizedRect
    confidence: float
    class_name: str = ""
    region_id: str = ""
    tile_index: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "rectangle": [
                self.rectangle.left,
                self.rectangle.top,
                self.rectangle.width,
                self.rectangle.height,
            ],
            "confidence": round(float(self.confidence), 4),
            "class_name": self.class_name,
            "region_id": self.region_id,
            "tile_index": self.tile_index,
        }


@dataclass(frozen=True, slots=True)
class GroundLitterDetection:
    object_id: int
    rectangle: NormalizedRect
    confidence: float
    class_name: str = ""
    region_id: str = ""
    hits: int = 1
    source: str = "model"

    def to_payload(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "rectangle": [
                self.rectangle.left,
                self.rectangle.top,
                self.rectangle.width,
                self.rectangle.height,
            ],
            "confidence": round(float(self.confidence), 4),
            "class_name": self.class_name,
            "region_id": self.region_id,
            "hits": self.hits,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class GroundLitterSnapshot:
    state: str = "disabled"
    detections: tuple[GroundLitterDetection, ...] = ()
    result_version: int = 0
    updated_at: float | None = None
    message: str = ""
    last_inference_ms: float = 0.0
    analyzed_frames: int = 0
    raw_candidates: int = 0
    rejected_roi: int = 0
    rejected_actor: int = 0
    tile_count: int = 0

    @property
    def count(self) -> int:
        return len(self.detections)


class GroundLitterResultCache:
    """Thread-safe snapshots written by the side process and drawn by OSD."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[int, GroundLitterSnapshot] = {}

    def snapshot(self, pad_index: int) -> GroundLitterSnapshot:
        with self._lock:
            return self._entries.get(pad_index, GroundLitterSnapshot())

    def store_snapshot(
        self,
        pad_index: int,
        snapshot: GroundLitterSnapshot,
    ) -> None:
        with self._lock:
            previous = self._entries.get(pad_index)
            if previous is not None and (
                snapshot.result_version < previous.result_version
            ):
                return
            self._entries[pad_index] = snapshot

    def mark_state(
        self,
        pad_index: int,
        state: str,
        message: str = "",
    ) -> None:
        with self._lock:
            previous = self._entries.get(pad_index, GroundLitterSnapshot())
            self._entries[pad_index] = GroundLitterSnapshot(
                state=state,
                detections=previous.detections,
                result_version=previous.result_version,
                updated_at=previous.updated_at,
                message=message,
                last_inference_ms=previous.last_inference_ms,
                analyzed_frames=previous.analyzed_frames,
                raw_candidates=previous.raw_candidates,
                rejected_roi=previous.rejected_roi,
                rejected_actor=previous.rejected_actor,
                tile_count=previous.tile_count,
            )


@dataclass(slots=True)
class _LitterTrack:
    object_id: int
    rectangle: NormalizedRect
    confidence: float
    class_name: str
    region_id: str
    hits: list[bool] = field(default_factory=list)
    last_seen: float = 0.0
    last_hit: float = 0.0
    total_hits: int = 0
    confirmed: bool = False
    display_rectangle: NormalizedRect | None = None


def _same_litter_location(
    left: NormalizedRect,
    right: NormalizedRect,
    *,
    size_ratio: float,
    distance_ratio: float,
) -> bool:
    if min(left.width, right.width) <= 0 or min(
        left.height, right.height
    ) <= 0:
        return False
    if max(
        left.width / right.width,
        right.width / left.width,
        left.height / right.height,
        right.height / left.height,
    ) > size_ratio:
        return False
    x, y = left.center
    u, v = right.center
    distance = ((x - u) / min(left.width, right.width)) ** 2
    distance += ((y - v) / min(left.height, right.height)) ** 2
    return distance <= distance_ratio**2


class GroundLitterDisplayTracker:
    """Turn sparse per-analysis candidates into a stable OSD picture.

    Analysis runs far below the stream frame rate (about 1 FPS), so a raw
    per-frame box would blink.  A detection is published once it was seen in
    ``minimum_hits`` of the last ``hit_window`` analyses and it stays visible
    for ``hold_seconds`` after the most recent hit.
    """

    def __init__(self, options: GroundLitterDetectionOptions) -> None:
        options.validate()
        self._options = options
        self._tracks: dict[int, _LitterTrack] = {}
        self._next_id = 1
        self._version = 0
        self._last_timestamp: float | None = None
        self._last_snapshot = GroundLitterSnapshot()

    @property
    def version(self) -> int:
        return self._version

    def update(
        self,
        candidates: Iterable[GroundLitterCandidate],
        *,
        timestamp: float,
        occluders: Iterable[NormalizedRect] = (),
    ) -> GroundLitterSnapshot:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp必须是有限数值")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            return self._last_snapshot
        self._last_timestamp = timestamp
        self._expire(timestamp)
        pending = list(candidates)
        blockers = list(occluders)
        def blocked(rect: NormalizedRect) -> bool:
            box = [rect.left, rect.top, rect.left + rect.width, rect.top + rect.height]
            return any(box_overlap_fraction(box, [a.left, a.top, a.left + a.width, a.top + a.height])
                       > self._options.actor_overlap_threshold for a in blockers)
        pending = [candidate for candidate in pending if not blocked(candidate.rectangle)]
        for object_id, track in list(self._tracks.items()):
            if blocked(track.rectangle):
                del self._tracks[object_id]
        options = self._options
        used_candidates: set[int] = set()
        used_tracks: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for object_id, track in self._tracks.items():
            for index, candidate in enumerate(pending):
                if not _same_litter_location(
                    track.rectangle,
                    candidate.rectangle,
                    size_ratio=options.match_size_ratio,
                    distance_ratio=options.match_distance_ratio,
                ):
                    continue
                x, y = track.rectangle.center
                u, v = candidate.rectangle.center
                distance = math.hypot(x - u, y - v)
                pairs.append((distance, object_id, index))
        for _distance, object_id, index in sorted(pairs):
            if object_id in used_tracks or index in used_candidates:
                continue
            used_tracks.add(object_id)
            used_candidates.add(index)
            track = self._tracks[object_id]
            candidate = pending[index]
            old = track.display_rectangle or track.rectangle
            alpha = options.box_smoothing_alpha
            new = candidate.rectangle
            track.display_rectangle = NormalizedRect(
                *(a + alpha * (b - a) for a, b in zip(
                    (old.left, old.top, old.width, old.height),
                    (new.left, new.top, new.width, new.height),
                ))
            )
            track.rectangle = candidate.rectangle
            track.confidence = candidate.confidence
            track.class_name = candidate.class_name or track.class_name
            track.region_id = candidate.region_id or track.region_id
            track.last_seen = timestamp
            track.last_hit = timestamp
            track.total_hits += 1
            track.hits.append(True)
            del track.hits[: -options.hit_window]
        for index, candidate in enumerate(pending):
            if index in used_candidates:
                continue
            track = _LitterTrack(
                object_id=self._next_id,
                rectangle=candidate.rectangle,
                confidence=candidate.confidence,
                class_name=candidate.class_name,
                region_id=candidate.region_id,
                hits=[True],
                last_seen=timestamp,
                last_hit=timestamp,
                total_hits=1,
            )
            self._tracks[self._next_id] = track
            # A brand-new track already has its first hit; without this the
            # miss loop below would immediately count a miss against it.
            used_tracks.add(self._next_id)
            self._next_id += 1
        for object_id, track in self._tracks.items():
            if object_id in used_tracks:
                continue
            # A miss inside the hit window is recorded but never moves
            # ``last_seen``, so the hold timer measures real evidence only.
            track.hits.append(False)
            del track.hits[: -options.hit_window]
        visible: list[GroundLitterDetection] = []
        for track in self._tracks.values():
            if timestamp - track.last_seen > options.hold_seconds:
                continue
            track.confirmed |= sum(track.hits) >= options.minimum_hits
            if not track.confirmed:
                continue
            visible.append(
                GroundLitterDetection(
                    object_id=track.object_id,
                    rectangle=track.display_rectangle or track.rectangle,
                    confidence=track.confidence,
                    class_name=track.class_name,
                    region_id=track.region_id,
                    hits=track.total_hits,
                )
            )
        visible.sort(key=lambda item: -item.confidence)
        self._version += 1
        self._last_snapshot = GroundLitterSnapshot(
            state="running",
            detections=tuple(visible[: options.maximum_boxes]),
            result_version=self._version,
            updated_at=timestamp,
        )
        return self._last_snapshot

    def reset(self) -> None:
        self._tracks.clear()
        self._last_timestamp = None
        self._last_snapshot = GroundLitterSnapshot()

    def _expire(self, timestamp: float) -> None:
        limit = self._options.maximum_age_seconds
        for object_id, track in list(self._tracks.items()):
            if timestamp - track.last_seen > self._options.hold_seconds:
                track.confirmed = False
                track.hits.clear()
                track.display_rectangle = None
            if timestamp - track.last_seen > limit:
                del self._tracks[object_id]


def ground_litter_camera_view(
    options: GroundLitterDetectionOptions,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Adapt the API options to the geometry helper's camera schema."""
    return {
        "reference_size": [int(width), int(height)],
        "overlay_exclude_zones": [
            [list(point) for point in polygon]
            for polygon in options.overlay_exclude_zones
        ],
        "zones": [
            {
                "region_id": zone.region_id,
                "polygon": [list(point) for point in zone.polygon],
                "exclude_zones": [
                    [list(point) for point in polygon]
                    for polygon in zone.exclude_zones
                ],
                "minimum_short_side_px": int(zone.minimum_short_side_px),
                "minimum_box_area_px": int(zone.minimum_box_area_px),
            }
            for zone in options.zones
        ],
    }


def build_ground_litter_tiles(
    options: GroundLitterDetectionOptions,
    width: int,
    height: int,
) -> tuple[dict[str, np.ndarray], list[tuple[int, int, int, int]]]:
    """Return the ground masks and the native-pixel tile plan."""
    camera = ground_litter_camera_view(options, width, height)
    frame = np.zeros((int(height), int(width), 3), np.uint8)
    masks, tiles, _coverage = prepare(
        camera,
        frame,
        int(options.tile_size_px),
        float(options.tile_overlap),
    )
    if len(tiles) > int(options.maximum_tiles):
        raise ValueError(
            "地面区域分块数超过ground_litter.maximum_tiles: "
            f"{len(tiles)} > {options.maximum_tiles}"
        )
    return masks, tiles


class UltralyticsGroundLitterDetector:
    """Ultralytics detector that keeps the audited native-pixel tiling."""

    def __init__(
        self,
        *,
        model_path: Any,
        device: str = "cpu",
        half: bool = False,
        actor_model_path: Any | None = None,
        model_factory: Any | None = None,
    ) -> None:
        self._device = device
        self.last_actor_boxes: list[list[float]] = []
        self._half = half
        if model_factory is None:  # pragma: no cover - requires ultralytics
            from ultralytics import YOLO

            model_factory = YOLO
        self._model = model_factory(str(model_path))
        self._actor = (
            model_factory(str(actor_model_path))
            if actor_model_path is not None
            else None
        )
        classifier = getattr(self._model, "names", {}) or {}
        self.class_names = {
            int(key): str(value) for key, value in dict(classifier).items()
        }

    @property
    def has_actor_model(self) -> bool:
        return self._actor is not None

    def actor_boxes(
        self,
        frame: np.ndarray,
        options: GroundLitterDetectionOptions,
    ) -> list[list[float]]:
        """Independent person/vehicle boxes in native pixels.

        Only the four box coordinates are returned: the raw prediction rows also
        carry confidence and class id, and the caller feeds these straight into
        ``box_overlap_fraction``, which unpacks exactly four values.
        """
        if self._actor is None:
            return []
        rows = self._predict(
            self._actor,
            frame,
            imgsz=int(options.actor_imgsz),
            confidence=float(options.actor_confidence),
            class_ids=tuple(dict.fromkeys((*options.actor_class_ids, *options.context_class_ids))),
        )
        return [
            [float(value) for value in row[:4]]
            for row in rows
            if len(row) >= 4
        ]

    def candidates(
        self,
        frame: np.ndarray,
        options: GroundLitterDetectionOptions,
        *,
        masks: dict[str, np.ndarray],
        tiles: list[tuple[int, int, int, int]],
        night: bool = False,
        actors: Iterable[Iterable[float]] = (),
    ) -> tuple[list[GroundLitterCandidate], dict[str, int]]:
        """Run the tiled detector and keep only in-region, unoccluded boxes."""
        height, width = frame.shape[:2]
        # Keep detector recall at the lowest configured zone threshold; apply
        # the selected zone threshold after ROI assignment below.
        zone_thresholds = {
            zone.region_id: zone.confidence_for(night, options.confidence_for(night))
            for zone in options.zones
        }
        threshold = min(zone_thresholds.values(), default=options.confidence_for(night))
        actor_boxes = [list(map(float, box)) for box in actors]
        stats = {
            "raw_candidates": 0,
            "rejected_roi": 0,
            "rejected_actor": 0,
            "rejected_confidence": 0,
            "local_actor_crops": 0,
        }
        detections: list[dict[str, Any]] = []
        for tile_index, (x, y, right, bottom) in enumerate(tiles):
            rows = self._predict(
                self._model,
                frame[y:bottom, x:right],
                imgsz=options.effective_imgsz,
                confidence=threshold,
                class_ids=None,
            )
            for row in rows:
                a, c, d, e = row[:4]
                confidence = float(row[4])
                class_id = int(row[5])
                box = [
                    max(0, int(round(a + x))),
                    max(0, int(round(c + y))),
                    min(width, int(round(d + x))),
                    min(height, int(round(e + y))),
                ]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                detections.append(
                    {
                        "box": box,
                        "confidence": confidence,
                        "class_id": class_id,
                        "tile_index": tile_index,
                    }
                )
        stats["raw_candidates"] = len(detections)
        # Filter before cross-tile NMS: a high-score box on a facility outside
        # the ground must not suppress a valid overlapping ground detection.
        eligible_detections = []
        for detection in detections:
            x, y, right, bottom = detection["box"]
            cx, cy = min(width - 1, (x + right) // 2), min(height - 1, (y + bottom) // 2)
            eligible = [
                zone for zone in options.zones
                if masks[zone.region_id][cy, cx]
                and min(right - x, bottom - y) >= zone.minimum_short_side_px
                and (right - x) * (bottom - y) >= zone.minimum_box_area_px
            ]
            if not eligible:
                stats["rejected_roi"] += 1
                continue
            eligible = [zone for zone in eligible
                        if detection["confidence"] >= zone_thresholds[zone.region_id]]
            if not eligible:
                stats["rejected_confidence"] += 1
                continue
            detection["eligible_zones"] = eligible
            eligible_detections.append(detection)
        detections = eligible_detections
        if detections:
            boxes = [
                [
                    item["box"][0],
                    item["box"][1],
                    item["box"][2] - item["box"][0],
                    item["box"][3] - item["box"][1],
                ]
                for item in detections
            ]
            keep = cv2.dnn.NMSBoxes(
                boxes,
                [item["confidence"] for item in detections],
                0.0,  # Confidence has already been checked, inclusively.
                float(options.nms_iou),
            )
            indices = (
                [int(index) for index in np.asarray(keep).reshape(-1)]
                if len(keep)
                else []
            )
            detections = [detections[index] for index in indices]
        candidates: list[GroundLitterCandidate] = []
        reviewed_crops: set[tuple[int, int, int, int]] = set()
        for detection in detections:
            x, y, right, bottom = detection["box"]
            center_x = min(width - 1, (x + right) // 2)
            center_y = min(height - 1, (y + bottom) // 2)
            eligible = detection["eligible_zones"]
            # Review only eligible candidates, at most N unique crops per frame.
            # Context extends beyond the candidate so baskets/seats can be
            # recognised as part of a vehicle. Never treats a model miss as proof.
            already_covered = max((box_overlap_fraction(detection["box"], a)
                                   for a in actor_boxes), default=0.0)
            if (self._actor is not None and options.local_actor_max_crops
                    and already_covered <= options.actor_overlap_threshold
                    and len(reviewed_crops) < options.local_actor_max_crops):
                edge = max(640, 3 * max(right - x, bottom - y))
                cw, ch = min(width, edge), min(height, edge)
                lx = max(0, min(width - cw, center_x - cw // 2))
                ly = max(0, min(height - ch, center_y - ch // 2))
                crop = (lx, ly, lx + cw, ly + ch)
                if crop not in reviewed_crops:
                    reviewed_crops.add(crop)
                    stats["local_actor_crops"] += 1
                    for row in self._predict(
                        self._actor, frame[ly:ly + ch, lx:lx + cw],
                        imgsz=640, confidence=options.actor_confidence,
                        class_ids=tuple(dict.fromkeys((*options.actor_class_ids, *options.context_class_ids))),
                    ):
                        actor_boxes.append([row[0] + lx, row[1] + ly,
                                            row[2] + lx, row[3] + ly])
            if max(
                (
                    box_overlap_fraction(detection["box"], actor)
                    for actor in actor_boxes
                ),
                default=0.0,
            ) > float(options.actor_overlap_threshold):
                stats["rejected_actor"] += 1
                continue
            candidates.append(
                GroundLitterCandidate(
                    rectangle=_normalized(
                        detection["box"],
                        width,
                        height,
                    ),
                    confidence=detection["confidence"],
                    class_name=self.class_names.get(
                        detection["class_id"],
                        "",
                    ),
                    # Ambiguous ownership stays explicitly unassigned.
                    region_id=(
                        eligible[0].region_id
                        if len(eligible) == 1
                        else ""
                    ),
                    tile_index=detection["tile_index"],
                )
            )
        # A later local crop can reveal a vehicle covering an earlier candidate.
        retained = []
        for candidate in candidates:
            r = candidate.rectangle
            box = [r.left * width, r.top * height,
                   (r.left + r.width) * width, (r.top + r.height) * height]
            if max((box_overlap_fraction(box, a) for a in actor_boxes), default=0.0) > options.actor_overlap_threshold:
                stats["rejected_actor"] += 1
            else:
                retained.append(candidate)
        self.last_actor_boxes = actor_boxes
        return retained, stats

    def _predict(
        self,
        model: Any,
        image: np.ndarray,
        *,
        imgsz: int,
        confidence: float,
        class_ids: tuple[int, ...] | None,
    ) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "imgsz": int(imgsz),
            "conf": float(confidence),
            "verbose": False,
            "device": self._device,
        }
        if class_ids is not None:
            kwargs["classes"] = list(class_ids)
        if self._half:
            # Ultralytics replaced the deprecated half flag with quantize=16.
            kwargs["quantize"] = 16
        result = model.predict(image, **kwargs)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "data", None) is None:
            return []
        return [
            [float(value) for value in row[:6]]
            for row in boxes.data.cpu().tolist()
        ]


def _normalized(
    box: Iterable[float],
    width: int,
    height: int,
) -> NormalizedRect:
    x, y, right, bottom = (float(value) for value in box)
    return NormalizedRect(
        x / width,
        y / height,
        (right - x) / width,
        (bottom - y) / height,
    )


def as_bgr(frame: Any) -> np.ndarray:
    """The DeepStream side branch hands over RGB; Ultralytics expects BGR."""
    value = np.asarray(frame)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if (
        value.ndim == 3
        and value.shape[0] in {3, 4}
        and value.shape[-1] not in {3, 4}
    ):
        # Only unambiguous CHW input is transposed; a tall HWC frame whose
        # height happens to be 3 or 4 must not be reinterpreted.
        value = np.moveaxis(value, 0, -1)
    if value.ndim != 3 or value.shape[-1] < 3:
        raise ValueError(f"零散垃圾检测输入帧形状无效: {value.shape}")
    return np.ascontiguousarray(value[..., :3][..., ::-1])


def snapshot_is_fresh(
    snapshot: GroundLitterSnapshot,
    options: GroundLitterDetectionOptions,
    *,
    now: float,
) -> bool:
    """OSD guard: never draw boxes from a stalled side process."""
    if snapshot.updated_at is None:
        return False
    return (now - snapshot.updated_at) <= (
        float(options.hold_seconds) + 1.0
    )
