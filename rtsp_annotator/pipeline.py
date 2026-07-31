from __future__ import annotations

import logging
import os
import queue
import re
import select
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from types import FrameType
from typing import Any, Callable

from .config import RoiPolygon, Settings, redact_url
from .labels import load_label_map, resolve_chinese_font, translated_names


LOGGER = logging.getLogger(__name__)
CUDA_DEVICE_PATTERN = re.compile(r"^(?:cuda:)?(\d+)$")
STATS_SAMPLE_NAMES = {
    "inference_latency_samples",
    "frame_age_samples",
    "detection_age_samples",
}


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame: Any
    source_sequence: int
    captured_at: float


@dataclass(frozen=True, slots=True)
class DetectionBox:
    xyxy: tuple[float, float, float, float]
    class_id: int


@dataclass(frozen=True, slots=True)
class DetectionSnapshot:
    boxes: tuple[DetectionBox, ...]
    names: dict[int, str]
    frame_shape: tuple[int, int]
    source_sequence: int
    captured_at: float


class LatestFrameSlot:
    """A one-item handoff that overwrites stale frames instead of queueing them."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._version = 0
        self._packet: FramePacket | None = None

    def publish(
        self,
        frame: Any,
        captured_at: float,
        source_sequence: int | None = None,
    ) -> FramePacket:
        with self._condition:
            self._version += 1
            packet = FramePacket(
                frame=frame,
                source_sequence=(
                    self._version if source_sequence is None else source_sequence
                ),
                captured_at=captured_at,
            )
            self._packet = packet
            self._condition.notify_all()
            return packet

    def wait_after(
        self,
        version: int,
        timeout: float,
    ) -> tuple[int, FramePacket | None]:
        with self._condition:
            if self._version <= version:
                self._condition.wait(timeout)
            if self._version <= version:
                return version, None
            return self._version, self._packet

    def latest(self) -> tuple[int, FramePacket | None]:
        with self._condition:
            return self._version, self._packet

    def wake_all(self) -> None:
        with self._condition:
            self._condition.notify_all()


@dataclass(frozen=True, slots=True)
class SourceInfo:
    width: int
    height: int
    fps: float


class SourceState:
    def __init__(self, fallback_fps: float) -> None:
        self._lock = threading.Lock()
        self._info: SourceInfo | None = None
        self._fallback_fps = fallback_fps

    def update(self, width: int, height: int, fps: float) -> SourceInfo:
        sane_fps = fps if 0.1 <= fps <= 120.0 else self._fallback_fps
        info = SourceInfo(width=width, height=height, fps=sane_fps)
        with self._lock:
            self._info = info
        return info

    def get(self) -> SourceInfo | None:
        with self._lock:
            return self._info


@dataclass(frozen=True, slots=True)
class StatsSnapshot:
    captured: int = 0
    inferred: int = 0
    published: int = 0
    unique_published: int = 0
    tracked_frames: int = 0
    inference_skipped: int = 0
    capture_reconnects: int = 0
    publisher_restarts: int = 0
    detections: int = 0
    inference_seconds: float = 0.0
    published_frame_age_seconds: float = 0.0
    detection_age_seconds: float = 0.0
    inference_latency_samples: tuple[float, ...] = ()
    frame_age_samples: tuple[float, ...] = ()
    detection_age_samples: tuple[float, ...] = ()


class PipelineStats:
    _COUNTER_NAMES = tuple(
        field.name
        for field in fields(StatsSnapshot)
        if field.name not in STATS_SAMPLE_NAMES
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values = {
            name: 0.0
            for name in self._COUNTER_NAMES
        }
        self._samples = {
            name: deque(maxlen=512)
            for name in STATS_SAMPLE_NAMES
        }

    def add(self, **increments: float | int) -> None:
        unknown = set(increments) - set(self._COUNTER_NAMES)
        if unknown:
            raise KeyError(f"unknown stats: {sorted(unknown)}")
        with self._lock:
            for name, value in increments.items():
                self._values[name] += value

    def observe(self, **samples: float) -> None:
        unknown = set(samples) - STATS_SAMPLE_NAMES
        if unknown:
            raise KeyError(f"unknown samples: {sorted(unknown)}")
        with self._lock:
            for name, value in samples.items():
                self._samples[name].append(float(value))

    def snapshot(self) -> StatsSnapshot:
        with self._lock:
            values = dict(self._values)
            samples = {
                name: tuple(items)
                for name, items in self._samples.items()
            }
        return StatsSnapshot(
            captured=int(values["captured"]),
            inferred=int(values["inferred"]),
            published=int(values["published"]),
            unique_published=int(values["unique_published"]),
            tracked_frames=int(values["tracked_frames"]),
            inference_skipped=int(values["inference_skipped"]),
            capture_reconnects=int(values["capture_reconnects"]),
            publisher_restarts=int(values["publisher_restarts"]),
            detections=int(values["detections"]),
            inference_seconds=float(values["inference_seconds"]),
            published_frame_age_seconds=float(
                values["published_frame_age_seconds"]
            ),
            detection_age_seconds=float(values["detection_age_seconds"]),
            inference_latency_samples=samples["inference_latency_samples"],
            frame_age_samples=samples["frame_age_samples"],
            detection_age_samples=samples["detection_age_samples"],
        )


@dataclass(frozen=True, slots=True)
class ResolvedDevice:
    value: str
    backend: str
    name: str


def resolve_inference_device(
    requested: str | None,
    torch_module: Any | None = None,
) -> ResolvedDevice:
    """Resolve auto/cpu/mps/cuda requests and fail instead of silently falling back."""
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError(
                "缺少 PyTorch，请先安装 requirements.txt 或 CUDA 版 PyTorch"
            ) from exc

    value = (requested or "auto").strip().lower()
    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda is not None and cuda.is_available())
    backends = getattr(torch_module, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    mps_available = bool(
        mps_backend is not None
        and mps_backend.is_built()
        and mps_backend.is_available()
    )

    if value == "auto":
        if cuda_available:
            value = "cuda:0"
        elif mps_available:
            value = "mps"
        else:
            value = "cpu"

    if value == "cpu":
        return ResolvedDevice(value="cpu", backend="cpu", name="CPU")

    if value == "mps":
        if not mps_available:
            raise RuntimeError(
                "已指定 MPS，但当前 PyTorch/macOS/Apple GPU 不支持 MPS"
            )
        name = "Apple Metal GPU"
        mps_runtime = getattr(torch_module, "mps", None)
        get_name = getattr(mps_runtime, "get_name", None)
        if callable(get_name):
            try:
                name = str(get_name())
            except RuntimeError:
                pass
        return ResolvedDevice(value="mps", backend="mps", name=name)

    if value == "cuda":
        value = "cuda:0"
    match = CUDA_DEVICE_PATTERN.fullmatch(value)
    if match is not None:
        index = int(match.group(1))
        if not cuda_available:
            raise RuntimeError(
                f"已指定 CUDA 设备 {index}，但当前 PyTorch 检测不到可用 CUDA"
            )
        device_count = int(cuda.device_count())
        if index >= device_count:
            raise RuntimeError(
                f"CUDA 设备编号 {index} 不存在，当前只检测到 {device_count} 张 GPU"
            )
        return ResolvedDevice(
            value=f"cuda:{index}",
            backend="cuda",
            name=str(cuda.get_device_name(index)),
        )

    raise ValueError(
        "device 必须是 auto、cpu、mps、cuda、cuda:N 或数字 GPU 编号"
    )


def validate_inference_precision(
    half: bool,
    device: ResolvedDevice,
) -> None:
    if half and device.backend != "cuda":
        raise RuntimeError("--half 仅支持 CUDA；MPS 和 CPU 请不要启用")


def normalized_roi_to_pixels(
    roi: RoiPolygon,
    width: int,
    height: int,
) -> tuple[tuple[int, int], ...]:
    """Convert a normalized ROI into pixel coordinates for the current frame."""
    if width <= 0 or height <= 0:
        raise ValueError("画面宽高必须大于 0")
    return tuple(
        (
            round(x * (width - 1)),
            round(y * (height - 1)),
        )
        for x, y in roi
    )


def point_in_polygon(
    point: tuple[float, float],
    polygon: tuple[tuple[int, int], ...],
) -> bool:
    """Return True when a point is inside or on the boundary of a polygon."""
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        cross_product = (
            (x - previous_x) * (current_y - previous_y)
            - (y - previous_y) * (current_x - previous_x)
        )
        if abs(cross_product) <= 1e-7 and (
            min(previous_x, current_x) <= x <= max(previous_x, current_x)
            and min(previous_y, current_y) <= y <= max(previous_y, current_y)
        ):
            return True

        crosses_scanline = (current_y > y) != (previous_y > y)
        if crosses_scanline:
            intersection_x = (
                (previous_x - current_x)
                * (y - current_y)
                / (previous_y - current_y)
                + current_x
            )
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def box_indices_inside_roi(
    xyxy: Any,
    polygon: tuple[tuple[int, int], ...],
) -> list[int]:
    """Return box indices whose center point is inside the ROI."""
    rows = xyxy
    if hasattr(rows, "detach"):
        rows = rows.detach()
    if hasattr(rows, "cpu"):
        rows = rows.cpu()
    if hasattr(rows, "tolist"):
        rows = rows.tolist()

    inside: list[int] = []
    for index, row in enumerate(rows):
        x1, y1, x2, y2 = (float(value) for value in row[:4])
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if point_in_polygon(center, polygon):
            inside.append(index)
    return inside


def draw_roi_boundary(
    frame: Any,
    polygon: tuple[tuple[int, int], ...],
    line_width: int,
) -> Any:
    """Draw the configured ROI boundary on an annotated BGR frame."""
    import cv2
    import numpy as np

    points = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        frame,
        [points],
        isClosed=True,
        color=(0, 255, 255),
        thickness=line_width,
        lineType=cv2.LINE_AA,
    )
    return frame


def _tensor_to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


@lru_cache(maxsize=128)
def _render_text_mask(path: str, size: int, label: str) -> Any:
    """Render only a small cached text mask instead of copying the full frame."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(path, size=size)
    probe = Image.new("L", (1, 1))
    probe_draw = ImageDraw.Draw(probe)
    left, top, right, bottom = probe_draw.textbbox((0, 0), label, font=font)
    width = max(1, right - left)
    height = max(1, bottom - top)
    mask = Image.new("L", (width, height))
    ImageDraw.Draw(mask).text((-left, -top), label, font=font, fill=255)
    return np.asarray(mask).copy()


@dataclass(frozen=True, slots=True)
class _RenderableBoxes:
    xyxy: tuple[tuple[float, float, float, float], ...]
    cls: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.xyxy)


def draw_detection_boxes(
    frame: Any,
    boxes: Any,
    names: dict[int, str],
    *,
    show_labels: bool,
    font_path: Path | None,
    line_width: int | None,
) -> Any:
    """Draw Chinese class names without ever reading or displaying confidence."""
    import cv2

    annotated = frame.copy()
    if boxes is None or len(boxes) == 0:
        return annotated

    height, width = annotated.shape[:2]
    stroke_width = line_width or max(round((height + width) * 0.0015), 2)
    font_size = max(round((height + width) * 0.012), 16)
    if show_labels:
        if font_path is None:
            raise RuntimeError("显示中文标签时必须提供可用中文字体")

    coordinates = _tensor_to_list(boxes.xyxy)
    class_ids = _tensor_to_list(boxes.cls)
    palette = (
        (50, 205, 50),
        (0, 215, 255),
        (255, 144, 30),
        (147, 20, 255),
        (60, 20, 220),
        (238, 130, 238),
    )

    for row, raw_class_id in zip(coordinates, class_ids):
        class_id = int(raw_class_id)
        x1, y1, x2, y2 = (round(float(value)) for value in row[:4])
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        color = palette[class_id % len(palette)]
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            stroke_width,
            lineType=cv2.LINE_AA,
        )
        if not show_labels or font_path is None:
            continue

        label = names.get(class_id, f"类别{class_id}")
        mask = _render_text_mask(str(font_path), font_size, label)
        text_height, text_width = mask.shape[:2]
        padding = max(2, stroke_width)
        label_height = text_height + padding * 2
        if y1 >= label_height:
            label_top = y1 - label_height
            label_bottom = y1
        else:
            label_top = y1
            label_bottom = min(height - 1, y1 + label_height)
        label_right = min(width - 1, x1 + text_width + padding * 2)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (label_right, label_bottom),
            color,
            thickness=-1,
        )
        text_x = x1 + padding
        text_y = label_top + padding
        available_width = max(0, min(text_width, width - text_x))
        available_height = max(0, min(text_height, height - text_y))
        if available_width == 0 or available_height == 0:
            continue
        visible_mask = mask[:available_height, :available_width]
        region = annotated[
            text_y : text_y + available_height,
            text_x : text_x + available_width,
        ]
        region[visible_mask > 0] = (255, 255, 255)
    return annotated


class YoloDetector:
    def __init__(self, settings: Settings) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "缺少 ultralytics，请先安装 requirements.txt"
            ) from exc

        resolved_device = resolve_inference_device(settings.device)
        validate_inference_precision(settings.half, resolved_device)

        LOGGER.info(
            "推理设备: %s (%s)，FP16=%s",
            resolved_device.value,
            resolved_device.name,
            "开启" if settings.half else "关闭",
        )
        LOGGER.info("加载 YOLO 模型: %s", settings.model_path)
        self._model = YOLO(str(settings.model_path))
        self._settings = settings
        self._device = resolved_device.value
        self._half = settings.half
        self._label_map = load_label_map(settings.label_map_path)
        self._font_path = (
            resolve_chinese_font(settings.font_path)
            if settings.show_labels
            else None
        )
        if settings.show_labels:
            LOGGER.info("中文标签字体: %s", self._font_path)
        if settings.roi is not None:
            LOGGER.info("已启用多边形识别区域，共 %d 个点", len(settings.roi))

    def annotate(self, frame: Any) -> tuple[Any, int]:
        snapshot = self.detect(
            frame,
            captured_at=time.monotonic(),
            source_sequence=0,
        )
        return (
            render_detection_snapshot(
                frame,
                snapshot,
                settings=self._settings,
                font_path=getattr(self, "_font_path", None),
            ),
            len(snapshot.boxes),
        )

    def detect(
        self,
        frame: Any,
        *,
        captured_at: float,
        source_sequence: int,
    ) -> DetectionSnapshot:
        kwargs = build_predict_kwargs(
            self._settings,
            device=getattr(self, "_device", self._settings.device),
            half=getattr(self, "_half", self._settings.half),
        )
        kwargs["source"] = frame
        results = self._model.predict(**kwargs)
        result = results[0] if results else None
        return prediction_to_snapshot(
            frame,
            result,
            settings=self._settings,
            model_names=getattr(self._model, "names", None),
            label_map=getattr(self, "_label_map", {}),
            captured_at=captured_at,
            source_sequence=source_sequence,
        )


def build_predict_kwargs(
    settings: Settings,
    *,
    device: str,
    half: bool,
) -> dict[str, Any]:
    """Build prediction arguments shared by single-frame and batched inference."""
    kwargs: dict[str, Any] = {
        "conf": settings.conf,
        "iou": settings.iou,
        "imgsz": settings.imgsz,
        "verbose": False,
    }
    if half:
        # Ultralytics replaced the deprecated half flag with quantize=16.
        kwargs["quantize"] = 16
    if settings.classes is not None:
        kwargs["classes"] = list(settings.classes)
    if device and device != "auto":
        kwargs["device"] = device
    return kwargs


def annotate_prediction(
    frame: Any,
    result: Any | None,
    *,
    settings: Settings,
    model_names: Any,
    label_map: dict[str, str],
    font_path: Path | None,
) -> tuple[Any, int]:
    """Apply per-stream ROI filtering and Chinese rendering to a prediction."""
    snapshot = prediction_to_snapshot(
        frame,
        result,
        settings=settings,
        model_names=model_names,
        label_map=label_map,
        captured_at=time.monotonic(),
        source_sequence=0,
    )
    return (
        render_detection_snapshot(
            frame,
            snapshot,
            settings=settings,
            font_path=font_path,
        ),
        len(snapshot.boxes),
    )


def prediction_to_snapshot(
    frame: Any,
    result: Any | None,
    *,
    settings: Settings,
    model_names: Any,
    label_map: dict[str, str],
    captured_at: float,
    source_sequence: int,
) -> DetectionSnapshot:
    """Copy the small detection metadata out of an Ultralytics result."""
    height, width = frame.shape[:2]
    names = translated_names(
        getattr(result, "names", model_names) if result is not None else model_names,
        label_map,
    )
    raw_boxes = getattr(result, "boxes", None) if result is not None else None
    coordinates = (
        _tensor_to_list(raw_boxes.xyxy)
        if raw_boxes is not None and hasattr(raw_boxes, "xyxy")
        else []
    )
    class_ids = (
        _tensor_to_list(raw_boxes.cls)
        if raw_boxes is not None and hasattr(raw_boxes, "cls")
        else []
    )
    polygon: tuple[tuple[int, int], ...] | None = None
    if settings.roi is not None:
        polygon = normalized_roi_to_pixels(
            settings.roi,
            width,
            height,
        )
    copied: list[DetectionBox] = []
    for row, raw_class_id in zip(coordinates, class_ids):
        values = tuple(float(value) for value in row[:4])
        if len(values) != 4:
            continue
        if polygon is not None:
            center = (
                (values[0] + values[2]) / 2.0,
                (values[1] + values[3]) / 2.0,
            )
            if not point_in_polygon(center, polygon):
                continue
        copied.append(
            DetectionBox(
                xyxy=values,
                class_id=int(raw_class_id),
            )
        )
    return DetectionSnapshot(
        boxes=tuple(copied),
        names=names,
        frame_shape=(height, width),
        source_sequence=source_sequence,
        captured_at=captured_at,
    )


def render_detection_snapshot(
    frame: Any,
    snapshot: DetectionSnapshot | None,
    *,
    settings: Settings,
    font_path: Path | None,
) -> Any:
    """Render current detection metadata on the current live video frame."""
    height, width = frame.shape[:2]
    if snapshot is None:
        boxes = _RenderableBoxes((), ())
        names: dict[int, str] = {}
    else:
        source_height, source_width = snapshot.frame_shape
        scale_x = width / source_width if source_width else 1.0
        scale_y = height / source_height if source_height else 1.0
        boxes = _RenderableBoxes(
            xyxy=tuple(
                (
                    item.xyxy[0] * scale_x,
                    item.xyxy[1] * scale_y,
                    item.xyxy[2] * scale_x,
                    item.xyxy[3] * scale_y,
                )
                for item in snapshot.boxes
            ),
            cls=tuple(item.class_id for item in snapshot.boxes),
        )
        names = snapshot.names
    annotated = draw_detection_boxes(
        frame,
        boxes,
        names,
        show_labels=settings.show_labels,
        font_path=font_path,
        line_width=settings.line_width,
    )
    if settings.roi is not None:
        polygon = normalized_roi_to_pixels(settings.roi, width, height)
        draw_roi_boundary(
            annotated,
            polygon,
            settings.roi_line_width,
        )
    return annotated


def _intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0,
        first[3] - first[1],
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0,
        second[3] - second[1],
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(slots=True)
class _MotionTrack:
    box: DetectionBox
    velocity: tuple[float, float, float, float]
    updated_at: float


class DetectionOverlayStore:
    """Thread-safe latest detections with short constant-velocity tracking."""

    def __init__(
        self,
        *,
        maximum_age_seconds: float = 0.5,
        maximum_extrapolation_seconds: float = 0.2,
        matching_iou: float = 0.15,
    ) -> None:
        self._maximum_age_seconds = maximum_age_seconds
        self._maximum_extrapolation_seconds = maximum_extrapolation_seconds
        self._matching_iou = matching_iou
        self._lock = threading.Lock()
        self._tracks: list[_MotionTrack] = []
        self._snapshot: DetectionSnapshot | None = None

    def update(self, snapshot: DetectionSnapshot) -> None:
        with self._lock:
            previous = list(self._tracks)
            available = set(range(len(previous)))
            updated: list[_MotionTrack] = []
            for detected in snapshot.boxes:
                best_index: int | None = None
                best_iou = self._matching_iou
                for index in available:
                    candidate = previous[index]
                    if candidate.box.class_id != detected.class_id:
                        continue
                    score = _intersection_over_union(
                        candidate.box.xyxy,
                        detected.xyxy,
                    )
                    if score >= best_iou:
                        best_index = index
                        best_iou = score
                velocity = (0.0, 0.0, 0.0, 0.0)
                if best_index is not None:
                    available.remove(best_index)
                    candidate = previous[best_index]
                    elapsed = snapshot.captured_at - candidate.updated_at
                    if elapsed > 1e-6:
                        measured = tuple(
                            (new - old) / elapsed
                            for new, old in zip(
                                detected.xyxy,
                                candidate.box.xyxy,
                            )
                        )
                        velocity = tuple(
                            old * 0.35 + new * 0.65
                            for old, new in zip(
                                candidate.velocity,
                                measured,
                            )
                        )
                updated.append(
                    _MotionTrack(
                        box=detected,
                        velocity=velocity,
                        updated_at=snapshot.captured_at,
                    )
                )
            self._tracks = updated
            self._snapshot = snapshot

    def snapshot_for(self, captured_at: float) -> DetectionSnapshot | None:
        with self._lock:
            source = self._snapshot
            tracks = tuple(self._tracks)
        if source is None:
            return None
        age = max(0.0, captured_at - source.captured_at)
        if age > self._maximum_age_seconds:
            return DetectionSnapshot(
                boxes=(),
                names=source.names,
                frame_shape=source.frame_shape,
                source_sequence=source.source_sequence,
                captured_at=source.captured_at,
            )
        extrapolation = min(age, self._maximum_extrapolation_seconds)
        height, width = source.frame_shape
        projected: list[DetectionBox] = []
        for track in tracks:
            values = tuple(
                coordinate + speed * extrapolation
                for coordinate, speed in zip(
                    track.box.xyxy,
                    track.velocity,
                )
            )
            x1 = max(0.0, min(values[0], width - 1.0))
            y1 = max(0.0, min(values[1], height - 1.0))
            x2 = max(x1, min(values[2], width - 1.0))
            y2 = max(y1, min(values[3], height - 1.0))
            projected.append(
                DetectionBox(
                    xyxy=(x1, y1, x2, y2),
                    class_id=track.box.class_id,
                )
            )
        return DetectionSnapshot(
            boxes=tuple(projected),
            names=source.names,
            frame_shape=source.frame_shape,
            source_sequence=source.source_sequence,
            captured_at=source.captured_at,
        )


def build_av_options(settings: Settings) -> dict[str, str]:
    """Build low-buffer FFmpeg demuxer options used by PyAV."""
    return {
        "rtsp_transport": settings.input_rtsp_transport,
        "fflags": "nobuffer",
        "flags": "low_delay",
        "max_delay": "0",
        "reorder_queue_size": "0",
        "probesize": "32",
        "analyzeduration": "0",
    }


class CaptureWorker(threading.Thread):
    def __init__(
        self,
        settings: Settings,
        output: LatestFrameSlot,
        source_state: SourceState,
        stats: PipelineStats,
        stop_event: threading.Event,
        errors: queue.Queue[BaseException],
    ) -> None:
        super().__init__(name="capture", daemon=True)
        self._settings = settings
        self._output = output
        self._source_state = source_state
        self._stats = stats
        self._stop_event = stop_event
        self._errors = errors

    def run(self) -> None:
        try:
            import av
        except ImportError as exc:
            LOGGER.exception("缺少 PyAV，请先安装 requirements.txt")
            self._errors.put(exc)
            self._stop_event.set()
            self._output.wake_all()
            return

        while not self._stop_event.is_set():
            container = None
            try:
                container = av.open(
                    self._settings.input_url,
                    mode="r",
                    options=build_av_options(self._settings),
                    timeout=(
                        self._settings.open_timeout_seconds,
                        self._settings.read_timeout_seconds,
                    ),
                )
                video = next(
                    stream
                    for stream in container.streams
                    if stream.type == "video"
                )
                # Frame threading can retain several decoded frames. Slice
                # threading preserves low latency while still allowing codecs
                # that support it to parallelize work inside one frame.
                video.thread_type = "SLICE"
                video.codec_context.thread_count = 1
                width = int(video.width or video.codec_context.width)
                height = int(video.height or video.codec_context.height)
                fps = (
                    float(video.average_rate)
                    if video.average_rate is not None
                    else self._settings.fallback_fps
                )
                info = self._source_state.update(width, height, fps)
                LOGGER.info(
                    "输入流已连接: %s，%dx%d @ %.2f FPS",
                    redact_url(self._settings.input_url),
                    info.width,
                    info.height,
                    info.fps,
                )

                for decoded in container.decode(video=0):
                    if self._stop_event.is_set():
                        break
                    frame = decoded.to_ndarray(format="bgr24")
                    height, width = frame.shape[:2]
                    if width != info.width or height != info.height:
                        info = self._source_state.update(width, height, fps)
                        LOGGER.warning("输入分辨率已变更为 %dx%d", width, height)
                    self._output.publish(frame, captured_at=time.monotonic())
                    self._stats.add(captured=1)

                if not self._stop_event.is_set():
                    self._retry_wait("输入流读取中断")
            except StopIteration:
                LOGGER.error("输入 RTSP 中没有视频轨道")
                self._retry_wait(None)
            except Exception:
                LOGGER.exception("拉流线程异常，准备重连")
                self._retry_wait(None)
            finally:
                if container is not None:
                    container.close()

    def _retry_wait(self, message: str | None) -> None:
        self._stats.add(capture_reconnects=1)
        if message:
            LOGGER.warning(
                "%s，%.1f 秒后重连",
                message,
                self._settings.reconnect_delay_seconds,
            )
        self._stop_event.wait(self._settings.reconnect_delay_seconds)


class DetectionWorker(threading.Thread):
    def __init__(
        self,
        detector: YoloDetector,
        input_slot: LatestFrameSlot,
        output_slot: LatestFrameSlot | DetectionOverlayStore,
        stats: PipelineStats,
        stop_event: threading.Event,
        errors: queue.Queue[BaseException],
    ) -> None:
        super().__init__(name="inference", daemon=True)
        self._detector = detector
        self._input = input_slot
        self._output = output_slot
        self._stats = stats
        self._stop_event = stop_event
        self._errors = errors

    def run(self) -> None:
        version = 0
        last_source_sequence = 0
        try:
            while not self._stop_event.is_set():
                version, packet = self._input.wait_after(version, timeout=0.5)
                if packet is None:
                    continue

                skipped = max(0, packet.source_sequence - last_source_sequence - 1)
                last_source_sequence = packet.source_sequence
                started = time.monotonic()
                detect = getattr(self._detector, "detect", None)
                update = getattr(self._output, "update", None)
                if callable(detect) and callable(update):
                    snapshot = detect(
                        packet.frame,
                        captured_at=packet.captured_at,
                        source_sequence=packet.source_sequence,
                    )
                    update(snapshot)
                    detections = len(snapshot.boxes)
                else:
                    annotated, detections = self._detector.annotate(
                        packet.frame
                    )
                    self._output.publish(
                        annotated,
                        captured_at=packet.captured_at,
                        source_sequence=packet.source_sequence,
                    )
                elapsed = time.monotonic() - started
                self._stats.add(
                    inferred=1,
                    inference_skipped=skipped,
                    detections=detections,
                    inference_seconds=elapsed,
                )
                self._stats.observe(inference_latency_samples=elapsed)
        except BaseException as exc:
            self._errors.put(exc)
            self._stop_event.set()
            wake_all = getattr(self._output, "wake_all", None)
            if callable(wake_all):
                wake_all()


def build_ffmpeg_command(
    settings: Settings,
    width: int,
    height: int,
    fps: float,
) -> list[str]:
    gop = max(1, round(fps * settings.gop_seconds))
    command = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-use_wallclock_as_timestamps",
        "1",
        "-fflags",
        "nobuffer",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        settings.encoder,
    ]
    if settings.encoder == "h264_nvenc":
        command.extend(
            [
                "-preset",
                settings.preset,
                "-tune",
                "ll",
                "-rc",
                "cbr",
                "-rc-lookahead",
                "0",
                "-delay",
                "0",
                "-zerolatency",
                "1",
            ]
        )
    else:
        command.extend(
            [
                "-preset",
                settings.preset,
                "-tune",
                "zerolatency",
            ]
        )
    command.extend(
        [
        "-pix_fmt",
        "yuv420p",
        "-bf",
        "0",
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        ]
    )
    if settings.encoder != "h264_nvenc":
        command.extend(["-sc_threshold", "0"])
    command.extend(
        [
        "-b:v",
        settings.bitrate,
        "-maxrate",
        settings.bitrate,
        "-bufsize",
        settings.bitrate,
        "-flush_packets",
        "1",
        "-muxdelay",
        "0.1",
        "-f",
        "rtsp",
        "-rtsp_transport",
        settings.output_rtsp_transport,
        settings.output_url,
        ]
    )
    return command


class FFmpegPublisher:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None

    def start(self, width: int, height: int, fps: float) -> None:
        self.close()
        command = build_ffmpeg_command(self._settings, width, height, fps)
        LOGGER.info(
            "开始发布: %s，%dx%d @ %.2f FPS，编码器=%s",
            redact_url(self._settings.output_url),
            width,
            height,
            fps,
            self._settings.encoder,
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self._process.stdin is not None:
            try:
                os.set_blocking(self._process.stdin.fileno(), False)
            except (AttributeError, OSError):
                # This is an optimization for graceful cancellation. The fallback
                # remains functional on platforms that cannot make pipes nonblocking.
                pass
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process,),
            name="ffmpeg-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def write(
        self,
        frame: Any,
        stop_event: threading.Event,
        timeout: float,
    ) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise BrokenPipeError("FFmpeg 发布进程尚未启动")
        if process.poll() is not None:
            raise BrokenPipeError(f"FFmpeg 已退出，退出码 {process.returncode}")

        if getattr(frame, "dtype", None) is None or str(frame.dtype) != "uint8":
            raise ValueError("发布帧必须是 uint8 数组")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("发布帧必须是 HxWx3 BGR 图像")
        if not frame.flags.c_contiguous:
            import numpy as np

            frame = np.ascontiguousarray(frame)

        if stop_event.is_set():
            raise InterruptedError("发布已取消")
        view = memoryview(frame).cast("B")
        deadline = time.monotonic() + timeout
        file_descriptor = process.stdin.fileno()
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("向 FFmpeg 写帧超时")
            try:
                written = os.write(file_descriptor, view)
            except BlockingIOError:
                if os.name == "nt":
                    stop_event.wait(min(0.001, remaining))
                else:
                    select.select([], [file_descriptor], [], min(0.1, remaining))
                continue
            if not written:
                raise BrokenPipeError("FFmpeg stdin 已关闭")
            view = view[written:]

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return

        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def _drain_stderr(self, process: subprocess.Popen[bytes]) -> None:
        if process.stderr is None:
            return
        for raw_line in iter(process.stderr.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                LOGGER.warning(
                    "FFmpeg: %s",
                    line.replace(self._settings.output_url, "<output-rtsp>"),
                )


class PublishWorker(threading.Thread):
    def __init__(
        self,
        settings: Settings,
        input_slot: LatestFrameSlot,
        source_state: SourceState,
        stats: PipelineStats,
        stop_event: threading.Event,
        errors: queue.Queue[BaseException],
        overlay_store: DetectionOverlayStore | None = None,
    ) -> None:
        super().__init__(name="publisher", daemon=True)
        self._settings = settings
        self._input = input_slot
        self._source_state = source_state
        self._stats = stats
        self._stop_event = stop_event
        self._errors = errors
        self._overlay_store = overlay_store
        self._font_path = (
            resolve_chinese_font(settings.font_path)
            if overlay_store is not None and settings.show_labels
            else None
        )
        self._publisher = FFmpegPublisher(settings)

    def run(self) -> None:
        last_frame: Any | None = None
        last_captured_at: float | None = None
        active_shape: tuple[int, int] | None = None
        active_fps: float | None = None
        version = 0
        next_deadline = time.monotonic()

        try:
            while not self._stop_event.is_set():
                current_version, packet = self._input.latest()
                has_new_frame = current_version > version and packet is not None
                if current_version > version and packet is not None:
                    version = current_version
                    last_frame = packet.frame
                    last_captured_at = packet.captured_at
                if last_frame is None:
                    version, packet = self._input.wait_after(version, timeout=0.5)
                    if packet is not None:
                        last_frame = packet.frame
                        last_captured_at = packet.captured_at
                    continue

                height, width = last_frame.shape[:2]
                shape = (width, height)
                fps = self._resolve_fps()
                if (
                    not self._publisher.is_running()
                    or active_shape != shape
                    or active_fps != fps
                ):
                    try:
                        self._publisher.start(width, height, fps)
                    except OSError:
                        LOGGER.exception("无法启动 FFmpeg")
                        self._stats.add(publisher_restarts=1)
                        self._stop_event.wait(
                            self._settings.reconnect_delay_seconds
                        )
                        continue
                    active_shape = shape
                    active_fps = fps
                    next_deadline = time.monotonic()

                try:
                    frame_to_publish = last_frame
                    detection_age = 0.0
                    tracked = 0
                    if self._overlay_store is not None:
                        overlay = self._overlay_store.snapshot_for(
                            last_captured_at
                            if last_captured_at is not None
                            else time.monotonic()
                        )
                        frame_to_publish = render_detection_snapshot(
                            last_frame,
                            overlay,
                            settings=self._settings,
                            font_path=self._font_path,
                        )
                        if overlay is not None:
                            detection_age = max(
                                0.0,
                                (
                                    last_captured_at
                                    if last_captured_at is not None
                                    else time.monotonic()
                                )
                                - overlay.captured_at,
                            )
                            tracked = int(
                                bool(overlay.boxes)
                                and overlay.source_sequence
                                != (
                                    packet.source_sequence
                                    if packet is not None
                                    else -1
                                )
                            )
                    self._publisher.write(
                        frame_to_publish,
                        stop_event=self._stop_event,
                        timeout=self._settings.open_timeout_seconds,
                    )
                    frame_age = (
                        time.monotonic() - last_captured_at
                        if last_captured_at is not None
                        else 0.0
                    )
                    self._stats.add(
                        published=1,
                        unique_published=int(has_new_frame),
                        tracked_frames=tracked,
                        published_frame_age_seconds=frame_age,
                        detection_age_seconds=detection_age,
                    )
                    self._stats.observe(
                        frame_age_samples=frame_age,
                        detection_age_samples=detection_age,
                    )
                except InterruptedError:
                    break
                except (BrokenPipeError, OSError, TimeoutError):
                    LOGGER.warning(
                        "输出发布中断，%.1f 秒后重试",
                        self._settings.reconnect_delay_seconds,
                    )
                    self._stats.add(publisher_restarts=1)
                    self._publisher.close()
                    self._stop_event.wait(self._settings.reconnect_delay_seconds)
                    continue

                interval = 1.0 / fps
                next_deadline += interval
                remaining = next_deadline - time.monotonic()
                if remaining > 0:
                    self._stop_event.wait(remaining)
                elif remaining < -interval:
                    next_deadline = time.monotonic()
        except BaseException as exc:
            self._errors.put(exc)
            self._stop_event.set()
        finally:
            self._publisher.close()

    def _resolve_fps(self) -> float:
        if self._settings.output_fps is not None:
            return self._settings.output_fps
        info = self._source_state.get()
        return info.fps if info is not None else self._settings.fallback_fps


def _check_runtime(settings: Settings) -> None:
    ffmpeg = shutil.which(settings.ffmpeg_path)
    if ffmpeg is None and not Path(settings.ffmpeg_path).is_file():
        raise RuntimeError(f"找不到 FFmpeg: {settings.ffmpeg_path}")


def _log_stats(
    previous: StatsSnapshot,
    current: StatsSnapshot,
    elapsed: float,
    *,
    stream_id: str | None = None,
) -> dict[str, float | int]:
    report = _build_stats_report(previous, current, elapsed)
    LOGGER.info(
        (
            "状态: stream=%s capture=%.1f FPS, inference=%.1f FPS, "
            "publish=%.1f FPS, unique=%.1f FPS, track=%.1f FPS, "
            "平均推理=%.1f ms, P95推理=%.1f ms, 内部帧龄=%.1f ms, "
            "P95帧龄=%.1f ms, "
            "本周期推理跳过=%d, 累计推理跳过=%d, 累计检出=%d, "
            "拉流重连=%d, 推流重启=%d"
        ),
        stream_id or "-",
        report["capture_fps"],
        report["inference_fps"],
        report["publish_fps"],
        report["unique_publish_fps"],
        report["tracking_fps"],
        report["average_inference_ms"],
        report["p95_inference_ms"],
        report["average_frame_age_ms"],
        report["p95_frame_age_ms"],
        report["interval_inference_skipped"],
        report["total_inference_skipped"],
        report["total_detections"],
        report["capture_reconnects"],
        report["publisher_restarts"],
    )
    return report


def _build_stats_report(
    previous: StatsSnapshot,
    current: StatsSnapshot,
    elapsed: float,
) -> dict[str, float | int]:
    captured = current.captured - previous.captured
    inferred = current.inferred - previous.inferred
    published = current.published - previous.published
    unique_published = current.unique_published - previous.unique_published
    tracked_frames = current.tracked_frames - previous.tracked_frames
    skipped = current.inference_skipped - previous.inference_skipped
    inference_seconds = current.inference_seconds - previous.inference_seconds
    published_frame_age_seconds = (
        current.published_frame_age_seconds
        - previous.published_frame_age_seconds
    )
    detection_age_seconds = (
        current.detection_age_seconds - previous.detection_age_seconds
    )
    average_ms = inference_seconds / inferred * 1000 if inferred else 0.0
    average_frame_age_ms = (
        published_frame_age_seconds / published * 1000 if published else 0.0
    )
    average_detection_age_ms = (
        detection_age_seconds / published * 1000 if published else 0.0
    )
    return {
        "capture_fps": captured / elapsed,
        "inference_fps": inferred / elapsed,
        "publish_fps": published / elapsed,
        "unique_publish_fps": unique_published / elapsed,
        "duplicate_publish_fps": (
            max(0, published - unique_published) / elapsed
        ),
        "tracking_fps": tracked_frames / elapsed,
        "average_inference_ms": average_ms,
        "p95_inference_ms": _percentile_ms(
            current.inference_latency_samples,
            0.95,
        ),
        "average_frame_age_ms": average_frame_age_ms,
        "p95_frame_age_ms": _percentile_ms(
            current.frame_age_samples,
            0.95,
        ),
        "average_detection_age_ms": average_detection_age_ms,
        "p95_detection_age_ms": _percentile_ms(
            current.detection_age_samples,
            0.95,
        ),
        "interval_inference_skipped": skipped,
        "total_inference_skipped": current.inference_skipped,
        "total_detections": current.detections,
        "capture_reconnects": current.capture_reconnects,
        "publisher_restarts": current.publisher_restarts,
    }


def _percentile_ms(samples: tuple[float, ...], fraction: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index] * 1000.0


def run_pipeline(
    settings: Settings,
    detector_factory: Callable[[Settings], YoloDetector] = YoloDetector,
    *,
    external_stop_event: threading.Event | None = None,
    stream_id: str | None = None,
    stats_callback: (
        Callable[[dict[str, float | int]], None] | None
    ) = None,
) -> None:
    settings.validate()
    _check_runtime(settings)
    LOGGER.info(
        "流水线启动: %s -> %s",
        redact_url(settings.input_url),
        redact_url(settings.output_url),
    )

    detector = detector_factory(settings)
    stop_event = external_stop_event or threading.Event()
    errors: queue.Queue[BaseException] = queue.Queue()
    captured_frames = LatestFrameSlot()
    detection_overlays = DetectionOverlayStore()
    source_state = SourceState(settings.fallback_fps)
    stats = PipelineStats()

    workers: list[threading.Thread] = [
        CaptureWorker(
            settings,
            captured_frames,
            source_state,
            stats,
            stop_event,
            errors,
        ),
        DetectionWorker(
            detector,
            captured_frames,
            detection_overlays,
            stats,
            stop_event,
            errors,
        ),
        PublishWorker(
            settings,
            captured_frames,
            source_state,
            stats,
            stop_event,
            errors,
            overlay_store=detection_overlays,
        ),
    ]

    previous_handlers: dict[int, Any] = {}

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        LOGGER.info("收到信号 %s，正在停止", signum)
        stop_event.set()
        captured_frames.wake_all()

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)

    for worker in workers:
        worker.start()

    previous = stats.snapshot()
    last_report = time.monotonic()
    fatal_error: BaseException | None = None
    try:
        while not stop_event.wait(0.25):
            try:
                fatal_error = errors.get_nowait()
            except queue.Empty:
                fatal_error = None
            if fatal_error is not None:
                break

            now = time.monotonic()
            if now - last_report >= settings.stats_interval_seconds:
                current = stats.snapshot()
                report = _log_stats(
                    previous,
                    current,
                    now - last_report,
                    stream_id=stream_id,
                )
                detector_metrics = getattr(detector, "metrics", None)
                if callable(detector_metrics):
                    report.update(detector_metrics())
                if stats_callback is not None:
                    stats_callback(report)
                previous = current
                last_report = now
        if fatal_error is None:
            try:
                fatal_error = errors.get_nowait()
            except queue.Empty:
                pass
    finally:
        stop_event.set()
        captured_frames.wake_all()
        for worker in workers:
            worker.join(timeout=settings.read_timeout_seconds + 5.0)
            if worker.is_alive():
                LOGGER.warning("线程未在超时内结束: %s", worker.name)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        close_detector = getattr(detector, "close", None)
        if callable(close_detector):
            close_detector()
        LOGGER.info("流水线已停止")

    if fatal_error is not None:
        raise RuntimeError("流水线工作线程失败") from fatal_error
