from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .event_engine import NormalizedRect


NormalizedPoint = tuple[float, float]
NormalizedPolygon = tuple[NormalizedPoint, ...]
NormalizedRegion = tuple[float, float, float, float]
_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.pt$")


@dataclass(frozen=True, slots=True)
class VesselDetectionOptions:
    """Per-stream high-recall vessel detection options.

    The vessel detector runs in a lossy side process so a large inference
    input or several perspective crops can never back-pressure the main RTSP
    pipeline. ``model=None`` reuses the stream's primary ``.pt`` model.
    """

    enabled: bool = False
    model: str | None = None
    analysis_fps: float = 5.0
    confidence: float = 0.10
    iou: float = 0.45
    imgsz: int = 1280
    class_ids: tuple[int, ...] = (8,)
    input_width: int = 1920
    input_height: int = 1080
    inference_regions: tuple[NormalizedRegion, ...] = (
        (0.0, 0.0, 1.0, 1.0),
    )
    roi: NormalizedPolygon | None = None
    exclude_rois: tuple[NormalizedPolygon, ...] = ()
    minimum_hits: int = 2
    hold_seconds: float = 1.0
    match_iou: float = 0.10
    maximum_center_distance: float = 1.50
    maximum_detections: int = 100
    duplicate_containment_threshold: float = 0.80
    large_box_area_threshold: float = 0.20
    large_box_minimum_confidence: float = 0.25
    maximum_box_area: float = 1.0
    display_ids: bool = False
    display_roi: bool = True

    def validate(self) -> None:
        if self.model is not None and not _SAFE_MODEL_NAME.fullmatch(
            self.model
        ):
            raise ValueError(
                "vessel_detection.model必须是models目录中的.pt文件名"
            )
        if not 0.1 <= self.analysis_fps <= 15:
            raise ValueError(
                "vessel_detection.analysis_fps必须在[0.1, 15]范围内"
            )
        if not 0 < self.confidence <= 1:
            raise ValueError(
                "vessel_detection.confidence必须在(0, 1]范围内"
            )
        if not 0 < self.iou <= 1:
            raise ValueError("vessel_detection.iou必须在(0, 1]范围内")
        if not 320 <= self.imgsz <= 2048:
            raise ValueError(
                "vessel_detection.imgsz必须在[320, 2048]范围内"
            )
        if not self.class_ids or any(item < 0 for item in self.class_ids):
            raise ValueError(
                "vessel_detection.class_ids必须是非空的非负类别ID列表"
            )
        if len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("vessel_detection.class_ids不能重复")
        if not 320 <= self.input_width <= 3840:
            raise ValueError(
                "vessel_detection.input_width必须在[320, 3840]范围内"
            )
        if not 180 <= self.input_height <= 2160:
            raise ValueError(
                "vessel_detection.input_height必须在[180, 2160]范围内"
            )
        if not self.inference_regions:
            raise ValueError(
                "vessel_detection.inference_regions至少需要一个区域"
            )
        for region in self.inference_regions:
            _validate_region(region)
        if self.roi is not None:
            _validate_polygon(self.roi, "vessel_detection.roi")
        for polygon in self.exclude_rois:
            _validate_polygon(polygon, "vessel_detection.exclude_rois")
        if not 1 <= self.minimum_hits <= 10:
            raise ValueError(
                "vessel_detection.minimum_hits必须在[1, 10]范围内"
            )
        if not 0.1 <= self.hold_seconds <= 5:
            raise ValueError(
                "vessel_detection.hold_seconds必须在[0.1, 5]范围内"
            )
        if not 0 <= self.match_iou <= 1:
            raise ValueError(
                "vessel_detection.match_iou必须在[0, 1]范围内"
            )
        if not 0.1 <= self.maximum_center_distance <= 5:
            raise ValueError(
                "vessel_detection.maximum_center_distance必须在"
                "[0.1, 5]范围内"
            )
        if not 1 <= self.maximum_detections <= 500:
            raise ValueError(
                "vessel_detection.maximum_detections必须在[1, 500]范围内"
            )
        if not 0 < self.duplicate_containment_threshold <= 1:
            raise ValueError(
                "vessel_detection.duplicate_containment_threshold必须在"
                "(0, 1]范围内"
            )
        if not 0 < self.large_box_area_threshold <= 1:
            raise ValueError(
                "vessel_detection.large_box_area_threshold必须在"
                "(0, 1]范围内"
            )
        if not 0 < self.large_box_minimum_confidence <= 1:
            raise ValueError(
                "vessel_detection.large_box_minimum_confidence必须在"
                "(0, 1]范围内"
            )
        if not 0 < self.maximum_box_area <= 1:
            raise ValueError(
                "vessel_detection.maximum_box_area必须在(0, 1]范围内"
            )
        if self.maximum_box_area < self.large_box_area_threshold:
            raise ValueError(
                "vessel_detection.maximum_box_area不能小于"
                "large_box_area_threshold"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "analysis_fps": self.analysis_fps,
            "confidence": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "class_ids": list(self.class_ids),
            "input_width": self.input_width,
            "input_height": self.input_height,
            "inference_regions": [
                list(region) for region in self.inference_regions
            ],
            "roi": (
                [list(point) for point in self.roi]
                if self.roi is not None
                else None
            ),
            "exclude_rois": [
                [list(point) for point in polygon]
                for polygon in self.exclude_rois
            ],
            "minimum_hits": self.minimum_hits,
            "hold_seconds": self.hold_seconds,
            "match_iou": self.match_iou,
            "maximum_center_distance": self.maximum_center_distance,
            "maximum_detections": self.maximum_detections,
            "duplicate_containment_threshold": (
                self.duplicate_containment_threshold
            ),
            "large_box_area_threshold": self.large_box_area_threshold,
            "large_box_minimum_confidence": (
                self.large_box_minimum_confidence
            ),
            "maximum_box_area": self.maximum_box_area,
            "display_ids": self.display_ids,
            "display_roi": self.display_roi,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any] | None,
    ) -> "VesselDetectionOptions":
        values = dict(payload or {})
        if "class_ids" in values:
            values["class_ids"] = tuple(int(item) for item in values["class_ids"])
        if "inference_regions" in values:
            values["inference_regions"] = tuple(
                tuple(float(value) for value in region)
                for region in values["inference_regions"]
            )
        if values.get("roi") is not None:
            values["roi"] = tuple(
                (float(point[0]), float(point[1]))
                for point in values["roi"]
            )
        if "exclude_rois" in values:
            values["exclude_rois"] = tuple(
                tuple(
                    (float(point[0]), float(point[1]))
                    for point in polygon
                )
                for polygon in values["exclude_rois"]
            )
        options = cls(**values)
        options.validate()
        return options


@dataclass(frozen=True, slots=True)
class VesselCandidate:
    rectangle: NormalizedRect
    confidence: float
    class_id: int
    source_region: int = 0


@dataclass(frozen=True, slots=True)
class VesselDetection:
    object_id: int
    rectangle: NormalizedRect
    confidence: float
    class_id: int
    hits: int


@dataclass(frozen=True, slots=True)
class VesselSnapshot:
    state: str = "disabled"
    detections: tuple[VesselDetection, ...] = ()
    result_version: int = 0
    updated_at: float | None = None
    message: str = ""
    last_inference_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)


class VesselResultCache:
    """Thread-safe snapshots written by the side process and drawn by OSD."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[int, VesselSnapshot] = {}

    def snapshot(self, pad_index: int) -> VesselSnapshot:
        with self._lock:
            return self._entries.get(pad_index, VesselSnapshot())

    def store_snapshot(
        self,
        pad_index: int,
        snapshot: VesselSnapshot,
    ) -> None:
        with self._lock:
            self._entries[pad_index] = snapshot

    def mark_state(
        self,
        pad_index: int,
        state: str,
        message: str = "",
    ) -> None:
        with self._lock:
            previous = self._entries.get(pad_index, VesselSnapshot())
            self._entries[pad_index] = VesselSnapshot(
                state=state,
                detections=previous.detections,
                result_version=previous.result_version,
                updated_at=previous.updated_at,
                message=message,
                last_inference_ms=previous.last_inference_ms,
            )


@dataclass(slots=True)
class _VesselTrack:
    object_id: int
    rectangle: NormalizedRect
    confidence: float
    class_id: int
    hits: int
    last_seen: float


class VesselTrackManager:
    """Small time-based tracker for sparse high-resolution detections."""

    def __init__(self, options: VesselDetectionOptions) -> None:
        options.validate()
        self.options = options
        self._tracks: dict[int, _VesselTrack] = {}
        self._next_id = 1
        self._version = 0

    def update(
        self,
        candidates: list[VesselCandidate],
        *,
        timestamp: float,
        inference_ms: float,
    ) -> VesselSnapshot:
        self._expire(timestamp)
        unmatched_tracks = set(self._tracks)
        unmatched_candidates = set(range(len(candidates)))
        matches: list[tuple[float, int, int]] = []
        for object_id, track in self._tracks.items():
            for candidate_index, candidate in enumerate(candidates):
                overlap = rectangle_iou(track.rectangle, candidate.rectangle)
                distance = normalized_center_distance(
                    track.rectangle,
                    candidate.rectangle,
                )
                if (
                    overlap < self.options.match_iou
                    and distance > self.options.maximum_center_distance
                ):
                    continue
                score = overlap * 2.0 - distance * 0.25
                matches.append((score, object_id, candidate_index))
        for _score, object_id, candidate_index in sorted(
            matches,
            reverse=True,
        ):
            if (
                object_id not in unmatched_tracks
                or candidate_index not in unmatched_candidates
            ):
                continue
            candidate = candidates[candidate_index]
            track = self._tracks[object_id]
            track.rectangle = candidate.rectangle
            track.confidence = candidate.confidence
            track.class_id = candidate.class_id
            track.hits += 1
            track.last_seen = timestamp
            unmatched_tracks.remove(object_id)
            unmatched_candidates.remove(candidate_index)
        for candidate_index in sorted(unmatched_candidates):
            candidate = candidates[candidate_index]
            object_id = self._next_id
            self._next_id += 1
            self._tracks[object_id] = _VesselTrack(
                object_id=object_id,
                rectangle=candidate.rectangle,
                confidence=candidate.confidence,
                class_id=candidate.class_id,
                hits=1,
                last_seen=timestamp,
            )
        self._expire(timestamp)
        self._version += 1
        detections = tuple(
            VesselDetection(
                object_id=track.object_id,
                rectangle=track.rectangle,
                confidence=track.confidence,
                class_id=track.class_id,
                hits=track.hits,
            )
            for track in sorted(
                self._tracks.values(),
                key=lambda item: item.object_id,
            )
            if track.hits >= self.options.minimum_hits
        )
        return VesselSnapshot(
            state="running",
            detections=detections,
            result_version=self._version,
            updated_at=timestamp,
            last_inference_ms=max(float(inference_ms), 0.0),
        )

    def _expire(self, timestamp: float) -> None:
        for object_id in [
            object_id
            for object_id, track in self._tracks.items()
            if timestamp - track.last_seen > self.options.hold_seconds
        ]:
            self._tracks.pop(object_id, None)


class UltralyticsVesselDetector:
    """Run one pretrained YOLO model over one or more perspective crops."""

    def __init__(
        self,
        *,
        model_path: Path,
        device: str = "cuda:0",
        half: bool = True,
        model_factory: Any | None = None,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"船舶检测模型不存在: {model_path}")
        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO
        self.model_path = model_path
        self.device = device
        self.half = half
        self._model = model_factory(str(model_path))

    def detect(
        self,
        frame: np.ndarray,
        options: VesselDetectionOptions,
    ) -> list[VesselCandidate]:
        options.validate()
        bgr = _as_bgr(frame)
        height, width = bgr.shape[:2]
        crops: list[np.ndarray] = []
        pixel_regions: list[tuple[int, int, int, int]] = []
        for region in options.inference_regions:
            left = min(max(int(math.floor(region[0] * width)), 0), width - 1)
            top = min(max(int(math.floor(region[1] * height)), 0), height - 1)
            right = min(max(int(math.ceil(region[2] * width)), left + 1), width)
            bottom = min(max(int(math.ceil(region[3] * height)), top + 1), height)
            crops.append(np.ascontiguousarray(bgr[top:bottom, left:right]))
            pixel_regions.append((left, top, right, bottom))
        kwargs: dict[str, Any] = {
            "source": crops,
            "imgsz": options.imgsz,
            "conf": options.confidence,
            "iou": options.iou,
            "classes": list(options.class_ids),
            "max_det": options.maximum_detections,
            "agnostic_nms": True,
            "device": self.device,
            "verbose": False,
        }
        if self.half:
            kwargs["quantize"] = 16
        results = list(self._model.predict(**kwargs))
        if len(results) != len(crops):
            raise RuntimeError(
                "船舶检测结果数量与透视分区数量不一致: "
                f"输入={len(crops)} 输出={len(results)}"
            )
        candidates: list[VesselCandidate] = []
        for source_region, (result, region) in enumerate(
            zip(results, pixel_regions)
        ):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy()
            region_left, region_top, _right, _bottom = region
            for box, score, class_id in zip(xyxy, scores, classes):
                left = (region_left + float(box[0])) / width
                top = (region_top + float(box[1])) / height
                right = (region_left + float(box[2])) / width
                bottom = (region_top + float(box[3])) / height
                rectangle = NormalizedRect(
                    max(left, 0.0),
                    max(top, 0.0),
                    min(max(right - left, 0.0), 1.0),
                    min(max(bottom - top, 0.0), 1.0),
                )
                candidate = VesselCandidate(
                    rectangle=rectangle,
                    confidence=float(score),
                    class_id=int(class_id),
                    source_region=source_region,
                )
                if _candidate_allowed(candidate, options):
                    candidates.append(candidate)
        return deduplicate_candidates(
            candidates,
            iou_threshold=options.iou,
            containment_threshold=options.duplicate_containment_threshold,
            limit=options.maximum_detections,
        )


def deduplicate_candidates(
    candidates: list[VesselCandidate],
    *,
    iou_threshold: float,
    containment_threshold: float = 0.80,
    limit: int,
) -> list[VesselCandidate]:
    selected: list[VesselCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: item.confidence,
        reverse=True,
    ):
        if any(
            _candidates_are_duplicates(
                candidate,
                current,
                iou_threshold=iou_threshold,
                containment_threshold=containment_threshold,
            )
            for current in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _candidates_are_duplicates(
    left: VesselCandidate,
    right: VesselCandidate,
    *,
    iou_threshold: float,
    containment_threshold: float,
) -> bool:
    if left.class_id != right.class_id:
        return False
    if rectangle_iou(left.rectangle, right.rectangle) >= iou_threshold:
        return True
    if left.source_region == right.source_region:
        return False
    return (
        rectangle_intersection_over_smaller(left.rectangle, right.rectangle)
        >= containment_threshold
    )


def rectangle_iou(left: NormalizedRect, right: NormalizedRect) -> float:
    intersection = _rectangle_intersection_area(left, right)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def rectangle_intersection_over_smaller(
    left: NormalizedRect,
    right: NormalizedRect,
) -> float:
    intersection = _rectangle_intersection_area(left, right)
    smaller = min(left.width * left.height, right.width * right.height)
    return intersection / smaller if smaller > 0 else 0.0


def _rectangle_intersection_area(
    left: NormalizedRect,
    right: NormalizedRect,
) -> float:
    intersection_width = max(
        min(left.left + left.width, right.left + right.width)
        - max(left.left, right.left),
        0.0,
    )
    intersection_height = max(
        min(left.top + left.height, right.top + right.height)
        - max(left.top, right.top),
        0.0,
    )
    return intersection_width * intersection_height


def normalized_center_distance(
    left: NormalizedRect,
    right: NormalizedRect,
) -> float:
    left_x, left_y = left.center
    right_x, right_y = right.center
    dx = (left_x - right_x) / max(left.width, right.width, 1e-6)
    dy = (left_y - right_y) / max(left.height, right.height, 1e-6)
    return math.hypot(dx, dy)


def _candidate_allowed(
    candidate: VesselCandidate,
    options: VesselDetectionOptions,
) -> bool:
    area = candidate.rectangle.width * candidate.rectangle.height
    if area > options.maximum_box_area:
        return False
    if (
        area >= options.large_box_area_threshold
        and candidate.confidence < options.large_box_minimum_confidence
    ):
        return False
    center = candidate.rectangle.center
    if options.roi is not None and not _point_in_polygon(center, options.roi):
        return False
    return not any(
        _point_in_polygon(center, polygon) for polygon in options.exclude_rois
    )


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3 and value.shape[0] in {3, 4}:
        value = np.moveaxis(value, 0, -1)
    if value.ndim != 3 or value.shape[-1] < 3:
        raise ValueError(f"船舶检测输入帧形状无效: {value.shape}")
    return np.ascontiguousarray(value[..., :3][..., ::-1])


def _validate_region(region: NormalizedRegion) -> None:
    if len(region) != 4:
        raise ValueError("船舶推理区域必须是[left, top, right, bottom]")
    left, top, right, bottom = region
    if not all(0 <= value <= 1 for value in region):
        raise ValueError("船舶推理区域坐标必须在[0, 1]范围内")
    if right <= left or bottom <= top:
        raise ValueError("船舶推理区域必须具有正面积")


def _validate_polygon(polygon: NormalizedPolygon, field_name: str) -> None:
    if len(polygon) < 3 or len(set(polygon)) < 3:
        raise ValueError(f"{field_name}至少需要3个不同的点")
    if any(
        not 0 <= coordinate <= 1
        for point in polygon
        for coordinate in point
    ):
        raise ValueError(f"{field_name}坐标必须在[0, 1]范围内")
    doubled_area = abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                polygon,
                polygon[1:] + polygon[:1],
            )
        )
    )
    if doubled_area <= 1e-9:
        raise ValueError(f"{field_name}不能是零面积多边形")


def _point_in_polygon(
    point: NormalizedPoint,
    polygon: NormalizedPolygon,
) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        cross = (
            (x - previous_x) * (current_y - previous_y)
            - (y - previous_y) * (current_x - previous_x)
        )
        if abs(cross) <= 1e-9 and (
            min(previous_x, current_x) - 1e-9
            <= x
            <= max(previous_x, current_x) + 1e-9
            and min(previous_y, current_y) - 1e-9
            <= y
            <= max(previous_y, current_y) + 1e-9
        ):
            return True
        if (current_y > y) != (previous_y > y):
            x_at_y = previous_x + (
                (y - previous_y)
                * (current_x - previous_x)
                / (current_y - previous_y)
            )
            if x <= x_at_y:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside
