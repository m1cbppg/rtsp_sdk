from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .background_change import BackgroundChangeDetector, VisualChange
from .event_engine import (
    EventEngine,
    GarbageDetection,
    GarbageOverlay,
    GarbageSnapshot,
    NormalizedRect,
    TrackedObject,
)
from .event_delivery import WebhookDispatcher
from .event_evidence import EventEvidenceWriter
from .events import (
    EventDetectionOptions,
    EventRepository,
    GarbageAnalysisOptions,
)
from .fishing_risk import (
    FishingRiskEngine,
    FishingRiskOptions,
    FishingRiskResultCache,
    FishingRiskSnapshot,
)
from .gas_cylinder import (
    GasCylinderCameraProfile,
    GasCylinderOptions,
    GasCylinderResultCache,
    GasCylinderSnapshot,
)
from .gas_cylinder_process import (
    GasCylinderProcessClient,
    GasCylinderProcessConfig,
)
from .ground_litter_detection import (
    HYBRID_LITTER_METRIC_FIELDS,
    GroundLitterDetectionOptions,
    GroundLitterResultCache,
    GroundLitterSnapshot,
    snapshot_is_fresh,
)
from .ground_litter_process import (
    GroundLitterProcessClient,
    GroundLitterProcessConfig,
)
from .labels import chinese_label
from .license_plate import (
    LICENSE_PLATE_DETECTOR_UID,
    LICENSE_PLATE_RECOGNIZER_UID,
    PRIMARY_DETECTOR_UID,
    PlateConsensus,
)
from .ptz_verification import (
    CameraControlClient,
    PtzVerificationCoordinator,
    PtzVerificationOptions,
    PtzVerificationRepository,
)
from .vessel_detection import (
    SMALL_TARGET_PROPOSAL_CLASS_ID,
    VesselDetection,
    VesselDetectionOptions,
    VesselResultCache,
    VesselSnapshot,
    rectangle_intersection_over_smaller,
)
from .vessel_detection_process import (
    VesselDetectionProcessClient,
    VesselDetectionProcessConfig,
)


LOGGER = logging.getLogger("rtsp_annotator.deepstream")
GARBAGE_DETECTOR_UID = 4
GST_BUFFER_FLAG_DISCONT = 1 << 6


class PtzControlCommandMonitor:
    """Deliver atomic API control files to in-process PTZ coordinators."""

    def __init__(
        self,
        control_dir: Path,
        coordinators: dict[str, PtzVerificationCoordinator],
    ) -> None:
        self._control_dir = control_dir
        self._coordinators = coordinators
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._control_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="ptz-control-command-monitor",
            daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            for path in sorted(self._control_dir.glob("ptz-home-*.json")):
                self._consume(path)
            self._stop_event.wait(0.1)

    def _consume(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            stream_id = str(payload.get("stream_id", ""))
            request_id = str(payload.get("request_id", ""))
            action = str(payload.get("action", ""))
            coordinator = self._coordinators.get(stream_id)
            if (
                coordinator is None
                or action != "return_home"
                or not request_id
            ):
                LOGGER.warning("忽略无效PTZ控制命令: path=%s", path)
                return
            coordinator.request_return_home(request_id)
            LOGGER.warning(
                "已接收紧急回HOME命令: stream=%s request=%s",
                stream_id,
                request_id,
            )
        except Exception:
            LOGGER.exception("处理PTZ控制命令失败: path=%s", path)
        finally:
            path.unlink(missing_ok=True)


GST_BUFFER_FLAG_CORRUPTED = 1 << 8


def _buffer_quality_flags(buffer: Any) -> tuple[bool, bool]:
    """Return (corrupted, discontinuous) for a Gst-like buffer."""
    try:
        flags = int(buffer.get_flags())
    except (AttributeError, TypeError, ValueError):
        return False, False
    return (
        bool(flags & GST_BUFFER_FLAG_CORRUPTED),
        bool(flags & GST_BUFFER_FLAG_DISCONT),
    )


def _point_in_polygon(
    point: tuple[float, float],
    polygon: list[list[float]],
) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (
            min(previous_y, current_y) <= y <= max(previous_y, current_y)
            and min(previous_x, current_x) <= x <= max(previous_x, current_x)
        ):
            cross = (
                (current_x - previous_x) * (y - previous_y)
                - (current_y - previous_y) * (x - previous_x)
            )
            if abs(cross) <= 1e-9:
                return True
        if (current_y > y) != (previous_y > y):
            x_at_y = (
                previous_x
                + (y - previous_y)
                * (current_x - previous_x)
                / (current_y - previous_y)
            )
            if x <= x_at_y:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _safe_config_value(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("DeepStream配置路径不能包含换行")
    return value


def build_inference_config(config: dict[str, Any], path: Path) -> None:
    thresholds = [float(item["conf"]) for item in config["streams"]]
    threshold = min(thresholds)
    night_vision = config.get("night_vision") or {}
    input_gain = (
        float(night_vision.get("input_gain", 1.0))
        if night_vision.get("enabled", False)
        else 1.0
    )
    if not 1 <= input_gain <= 1.5:
        raise ValueError("night_vision.input_gain必须在[1, 1.5]范围内")
    net_scale_factor = input_gain / 255.0
    values = {
        key: _safe_config_value(str(config[key]))
        for key in (
            "onnx_path",
            "engine_path",
            "labels_path",
            "parser_library",
        )
    }
    content = "\n".join(
        [
            "[property]",
            f"gpu-id={int(config['gpu_id'])}",
            f"net-scale-factor={net_scale_factor:.17g}",
            "model-color-format=0",
            f"onnx-file={values['onnx_path']}",
            f"model-engine-file={values['engine_path']}",
            f"labelfile-path={values['labels_path']}",
            f"batch-size={int(config['batch_size'])}",
            "network-mode=2",
            f"num-detected-classes={int(config['label_count'])}",
            "interval=0",
            "gie-unique-id=1",
            "process-mode=1",
            "network-type=0",
            "cluster-mode=4",
            "maintain-aspect-ratio=1",
            "symmetric-padding=1",
            "parse-bbox-func-name=NvDsInferParseYolo",
            f"custom-lib-path={values['parser_library']}",
            "engine-create-func-name=NvDsInferYoloCudaEngineGet",
            "",
            "[class-attrs-all]",
            f"pre-cluster-threshold={threshold:.8f}",
            "topk=300",
            "",
        ]
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_lpd_config(config: dict[str, Any], path: Path) -> None:
    lpr = config["license_plate"]
    stream_options = [
        item["license_plate"] for item in config["streams"]
    ]
    vehicle_classes = sorted(
        {
            int(class_id)
            for options in stream_options
            for class_id in options["vehicle_classes"]
        }
    )
    night_vision = config.get("night_vision") or {}
    detector_threshold = (
        float(night_vision.get("plate_detector_confidence", 0.30))
        if night_vision.get("enabled", False)
        else 0.30
    )
    content = "\n".join(
        [
            "[property]",
            f"gpu-id={int(config['gpu_id'])}",
            "net-scale-factor=0.0039215697906911373",
            "model-color-format=0",
            f"onnx-file={_safe_config_value(lpr['detector_onnx_path'])}",
            (
                "model-engine-file="
                f"{_safe_config_value(lpr['detector_engine_path'])}"
            ),
            f"batch-size={int(lpr['detector_batch_size'])}",
            "network-mode=2",
            "num-detected-classes=1",
            "process-mode=2",
            (
                "secondary-reinfer-interval="
                f"{min(int(item['detector_interval']) for item in stream_options)}"
            ),
            f"gie-unique-id={LICENSE_PLATE_DETECTOR_UID}",
            "network-type=0",
            f"operate-on-gie-id={PRIMARY_DETECTOR_UID}",
            (
                "operate-on-class-ids="
                + ";".join(str(item) for item in vehicle_classes)
            ),
            "cluster-mode=3",
            (
                "output-blob-names="
                "output_cov/Sigmoid:0;output_bbox/BiasAdd:0"
            ),
            "input-object-min-height=74",
            "input-object-min-width=46",
            "",
            "[class-attrs-all]",
            f"pre-cluster-threshold={detector_threshold:.8f}",
            "",
        ]
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_lpr_config(config: dict[str, Any], path: Path) -> None:
    lpr = config["license_plate"]
    stream_options = [
        item["license_plate"] for item in config["streams"]
    ]
    threshold = min(
        float(item["minimum_plate_confidence"])
        for item in stream_options
    )
    reinfer_interval = min(
        int(item["recognition_reinfer_interval"])
        for item in stream_options
    )
    content = "\n".join(
        [
            "[property]",
            f"gpu-id={int(config['gpu_id'])}",
            f"onnx-file={_safe_config_value(lpr['recognizer_onnx_path'])}",
            (
                "model-engine-file="
                f"{_safe_config_value(lpr['recognizer_engine_path'])}"
            ),
            f"batch-size={int(lpr['recognizer_batch_size'])}",
            "network-mode=2",
            f"gie-unique-id={LICENSE_PLATE_RECOGNIZER_UID}",
            "output-blob-names=tf_op_layer_ArgMax;tf_op_layer_Max",
            "network-type=1",
            "parse-classifier-func-name=NvDsInferParseCustomNVPlate",
            (
                "custom-lib-path="
                f"{_safe_config_value(lpr['parser_library'])}"
            ),
            "process-mode=2",
            f"operate-on-gie-id={LICENSE_PLATE_DETECTOR_UID}",
            "classifier-async-mode=1",
            f"secondary-reinfer-interval={reinfer_interval}",
            "net-scale-factor=0.00392156862745098",
            "model-color-format=0",
            "maintain-aspect-ratio=0",
            "",
            "[class-attrs-all]",
            f"threshold={threshold:.8f}",
            "",
        ]
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_garbage_config(config: dict[str, Any], path: Path) -> None:
    garbage = config["garbage"]
    enabled_streams = [
        item["event_detection"]["garbage"]
        for item in config["streams"]
        if item.get("event_detection", {}).get("garbage", {}).get(
            "enabled", False
        )
    ]
    if not enabled_streams:
        raise RuntimeError("垃圾模型已启用但没有流开启垃圾分析")
    threshold = min(
        float(item["minimum_confidence"]) for item in enabled_streams
    )
    night_vision = config.get("night_vision") or {}
    input_gain = (
        float(night_vision.get("input_gain", 1.0))
        if night_vision.get("enabled", False)
        else 1.0
    )
    content = "\n".join(
        [
            "[property]",
            f"gpu-id={int(config['gpu_id'])}",
            f"net-scale-factor={input_gain / 255.0:.17g}",
            "model-color-format=0",
            f"onnx-file={_safe_config_value(garbage['onnx_path'])}",
            (
                "model-engine-file="
                f"{_safe_config_value(garbage['engine_path'])}"
            ),
            f"labelfile-path={_safe_config_value(garbage['labels_path'])}",
            f"batch-size={int(config['batch_size'])}",
            "network-mode=2",
            f"num-detected-classes={int(garbage['label_count'])}",
            # Frames are sampled before this element by a branch-local
            # BufferOperator. nvinfer therefore runs on every frame it sees,
            # and an empty metadata result really means "no garbage".
            "interval=0",
            f"gie-unique-id={GARBAGE_DETECTOR_UID}",
            "process-mode=1",
            "network-type=0",
            # YOLO-World uses the dense YOLOv8 head, unlike the main YOLO26
            # one-to-one export. Run DeepStream NMS before Python receives
            # metadata, then de-duplicate overlapping cross-prompt boxes.
            "cluster-mode=2",
            "maintain-aspect-ratio=1",
            "symmetric-padding=1",
            "parse-bbox-func-name=NvDsInferParseYolo",
            (
                "custom-lib-path="
                f"{_safe_config_value(garbage['parser_library'])}"
            ),
            "engine-create-func-name=NvDsInferYoloCudaEngineGet",
            "",
            "[class-attrs-all]",
            f"pre-cluster-threshold={threshold:.8f}",
            "nms-iou-threshold=0.45",
            "topk=100",
            "",
        ]
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_tracker_config(
    source_path: Path,
    target_path: Path,
    *,
    max_shadow_tracking_age: int,
) -> None:
    if not 1 <= max_shadow_tracking_age <= 200:
        raise ValueError(
            "tracker_max_shadow_tracking_age必须在[1, 200]范围内"
        )
    content = source_path.read_text(encoding="utf-8")
    updated, replacements = re.subn(
        r"(?m)^(\s*maxShadowTrackingAge:\s*)\d+(\s*(?:#.*)?)$",
        rf"\g<1>{max_shadow_tracking_age}\g<2>",
        content,
    )
    if replacements != 1:
        raise RuntimeError(
            "NvDCF配置中没有唯一的maxShadowTrackingAge"
        )
    temporary = target_path.with_suffix(".tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(target_path)


@dataclass(slots=True)
class StreamPolicy:
    stream_id: str
    classes: frozenset[int] | None
    conf: float
    roi: list[list[float]] | None
    labels: dict[int, str]
    display_detections: bool = True
    license_plate_enabled: bool = False
    minimum_plate_confirmations: int = 2
    event_detection: EventDetectionOptions = EventDetectionOptions()
    gas_cylinder: GasCylinderOptions = GasCylinderOptions()
    vessel_detection: VesselDetectionOptions = VesselDetectionOptions()
    ground_litter: GroundLitterDetectionOptions = GroundLitterDetectionOptions()
    fishing_risk: FishingRiskOptions = FishingRiskOptions()
    ptz_verification: PtzVerificationOptions = PtzVerificationOptions()


@dataclass(frozen=True, slots=True)
class _GarbageDisplayEntry:
    updated_at: float
    roi_id: str
    detections: tuple[GarbageDetection, ...]


class GarbageOverlayCache:
    """Thread-safe bridge from the lossy analysis branch to the main OSD."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[int, str], _GarbageDisplayEntry] = {}

    def update(
        self,
        *,
        pad_index: int,
        roi_id: str,
        snapshot: GarbageSnapshot,
    ) -> None:
        detections = snapshot.detections
        if not detections and snapshot.regions:
            detections = tuple(
                GarbageDetection(
                    rectangle=rectangle,
                    object_type=snapshot.object_type,
                    confidence=snapshot.semantic_confidence,
                )
                for rectangle in snapshot.regions
            )
        with self._lock:
            self._entries[(pad_index, roi_id)] = _GarbageDisplayEntry(
                updated_at=snapshot.timestamp,
                roi_id=roi_id,
                detections=detections,
            )

    def overlays(
        self,
        *,
        pad_index: int,
        options: GarbageAnalysisOptions,
        timestamp: float,
    ) -> list[GarbageOverlay]:
        if not options.enabled or not options.display_detections:
            return []
        with self._lock:
            expired = [
                key
                for key, entry in self._entries.items()
                if timestamp - entry.updated_at > options.display_hold_seconds
            ]
            for key in expired:
                self._entries.pop(key, None)
            candidates = [
                (entry.roi_id, detection)
                for (entry_pad, _roi_id), entry in self._entries.items()
                if entry_pad == pad_index
                for detection in entry.detections
            ]
        selected: list[tuple[str, GarbageDetection]] = []
        for roi_id, detection in sorted(
            candidates,
            key=lambda item: item[1].confidence,
            reverse=True,
        ):
            if any(
                _rectangle_iou(
                    detection.rectangle,
                    existing.rectangle,
                ) >= 0.5
                for _existing_roi, existing in selected
            ):
                continue
            selected.append((roi_id, detection))
            if len(selected) >= options.maximum_display_boxes:
                break
        return [
            GarbageOverlay(
                roi_id=roi_id,
                rectangle=detection.rectangle,
                label=f"垃圾：{_garbage_chinese_label(detection.object_type)}",
                state="detected",
                elapsed_seconds=0.0,
            )
            for roi_id, detection in selected
        ]


def _frame_dimensions(
    frame_meta: Any,
    *,
    fallback_width: int | None = None,
    fallback_height: int | None = None,
) -> tuple[float, float]:
    """Resolve dimensions when ServiceMaker leaves pipeline size as zero."""
    width = float(getattr(frame_meta, "pipeline_width", 0) or 0)
    height = float(getattr(frame_meta, "pipeline_height", 0) or 0)
    if width <= 1 and fallback_width is not None:
        width = float(fallback_width)
    if height <= 1 and fallback_height is not None:
        height = float(fallback_height)
    return max(width, 1.0), max(height, 1.0)


class MetricsState:
    def __init__(
        self,
        *,
        stream_ids: list[str],
        metrics_path: Path,
        interval_seconds: float,
        minimum_healthy_fps: float,
        group_id: str,
        generation: int,
        license_plate_enabled: dict[str, bool] | None = None,
        night_vision_enabled: dict[str, bool] | None = None,
        gas_cylinder_enabled: dict[str, bool] | None = None,
        gas_cylinder_cache: GasCylinderResultCache | None = None,
        vessel_detection_enabled: dict[str, bool] | None = None,
        vessel_detection_cache: VesselResultCache | None = None,
        ground_litter_enabled: dict[str, bool] | None = None,
        ground_litter_cache: GroundLitterResultCache | None = None,
        fishing_risk_enabled: dict[str, bool] | None = None,
        fishing_risk_cache: FishingRiskResultCache | None = None,
        ptz_verification_coordinators: dict[
            str, PtzVerificationCoordinator
        ] | None = None,
        ptz_verification_errors: dict[str, str] | None = None,
    ) -> None:
        self._metrics_path = metrics_path
        self._interval_seconds = interval_seconds
        self._minimum_healthy_fps = minimum_healthy_fps
        self._group_id = group_id
        self._generation = generation
        self._license_plate_enabled = license_plate_enabled or {}
        self._night_vision_enabled = night_vision_enabled or {}
        self._gas_cylinder_enabled = gas_cylinder_enabled or {}
        self._gas_cylinder_cache = gas_cylinder_cache
        self._vessel_detection_enabled = vessel_detection_enabled or {}
        self._vessel_detection_cache = vessel_detection_cache
        self._ground_litter_enabled = ground_litter_enabled or {}
        self._ground_litter_cache = ground_litter_cache
        self._fishing_risk_enabled = fishing_risk_enabled or {}
        self._fishing_risk_cache = fishing_risk_cache
        self._ptz_verification_coordinators = (
            ptz_verification_coordinators or {}
        )
        self._ptz_verification_errors = ptz_verification_errors or {}
        self._stream_pad_index = {
            stream_id: index for index, stream_id in enumerate(stream_ids)
        }
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._last_report_at = self._started_at
        self._frames = {stream_id: 0 for stream_id in stream_ids}
        self._pre_encode = {stream_id: 0 for stream_id in stream_ids}
        self._published = {stream_id: 0 for stream_id in stream_ids}
        self._detections = {stream_id: 0 for stream_id in stream_ids}
        self._plate_detections = {
            stream_id: 0 for stream_id in stream_ids
        }
        self._plate_reads = {stream_id: 0 for stream_id in stream_ids}
        self._garbage_frames = {stream_id: 0 for stream_id in stream_ids}
        self._corrupt_frames = {stream_id: 0 for stream_id in stream_ids}
        self._discontinuities = {stream_id: 0 for stream_id in stream_ids}
        self._frame_gaps = {stream_id: 0 for stream_id in stream_ids}
        self._last_pre_encode_at: dict[str, float | None] = {
            stream_id: None for stream_id in stream_ids
        }
        self._maximum_gap_ms = {stream_id: 0.0 for stream_id in stream_ids}
        self._last_frames = dict(self._frames)
        self._last_pre_encode = dict(self._pre_encode)
        self._last_published = dict(self._published)
        self._last_detections = dict(self._detections)
        self._last_plate_detections = dict(self._plate_detections)
        self._last_plate_reads = dict(self._plate_reads)
        self._last_garbage_frames = dict(self._garbage_frames)
        self._last_corrupt_frames = dict(self._corrupt_frames)
        self._last_discontinuities = dict(self._discontinuities)
        self._last_frame_gaps = dict(self._frame_gaps)
        self._inference_ms: dict[str, list[float]] = {
            stream_id: [] for stream_id in stream_ids
        }
        self._garbage_analysis_ms: dict[str, list[float]] = {
            stream_id: [] for stream_id in stream_ids
        }
        self._ground_litter: dict[str, dict[str, Any]] = {
            stream_id: {} for stream_id in stream_ids
        }

    def observe_frame(
        self,
        stream_id: str,
        detections: int,
        inference_ms: float | None = None,
    ) -> None:
        with self._lock:
            self._frames[stream_id] += 1
            self._detections[stream_id] += detections
            if inference_ms is not None:
                samples = self._inference_ms[stream_id]
                samples.append(inference_ms)
                if len(samples) > 2_000:
                    del samples[: len(samples) - 2_000]
            self._maybe_write_locked()

    def observe_pre_encode(
        self,
        stream_id: str,
        buffer: Any | None = None,
    ) -> None:
        with self._lock:
            self._pre_encode[stream_id] += 1
            now = time.monotonic()
            previous = self._last_pre_encode_at[stream_id]
            if previous is not None:
                gap_ms = (now - previous) * 1000.0
                if gap_ms > 500.0:
                    self._frame_gaps[stream_id] += 1
                    self._maximum_gap_ms[stream_id] = max(
                        self._maximum_gap_ms[stream_id],
                        gap_ms,
                    )
            self._last_pre_encode_at[stream_id] = now
            if buffer is not None:
                corrupted, discontinuous = _buffer_quality_flags(buffer)
                if corrupted:
                    self._corrupt_frames[stream_id] += 1
                if discontinuous:
                    self._discontinuities[stream_id] += 1

    def observe_license_plates(
        self,
        stream_id: str,
        detections: int,
        reads: int,
    ) -> None:
        with self._lock:
            self._plate_detections[stream_id] += detections
            self._plate_reads[stream_id] += reads

    def observe_ground_litter(
        self,
        stream_id: str,
        *,
        state: str,
        count: int,
        result_version: int,
        updated_at: float | None,
        last_inference_ms: float,
        analyzed_frames: int,
        tile_count: int,
        raw_candidates: int = 0,
        active_events: int = 0,
        confirmed_events: int = 0,
        cleared_events: int = 0,
        environment_state: str = "",
        message: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Keep the newest side-process snapshot for `/v1/streams/{id}`.

        ``extra`` carries the hybrid_v33 dual-channel telemetry without growing
        this signature with twenty more keyword arguments.
        """
        with self._lock:
            entry = {
                "state": str(state),
                "count": int(count),
                "result_version": int(result_version),
                "updated_at": updated_at,
                "last_inference_ms": float(last_inference_ms),
                "analyzed_frames": int(analyzed_frames),
                "tile_count": int(tile_count),
                "raw_candidates": int(raw_candidates),
                "active_events": int(active_events),
                "confirmed_events": int(confirmed_events),
                "cleared_events": int(cleared_events),
                "environment_state": str(environment_state),
                "message": str(message),
            }
            if extra:
                entry.update(extra)
            self._ground_litter[stream_id] = entry

    def observe_publish(self, stream_id: str) -> None:
        with self._lock:
            self._published[stream_id] += 1

    def observe_garbage_analysis(
        self,
        stream_id: str,
        duration_ms: float,
    ) -> None:
        with self._lock:
            self._garbage_frames[stream_id] += 1
            samples = self._garbage_analysis_ms[stream_id]
            samples.append(max(float(duration_ms), 0.0))
            if len(samples) > 1_000:
                del samples[: len(samples) - 1_000]

    def _maybe_write_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_report_at
        if elapsed < self._interval_seconds:
            return
        streams: dict[str, dict[str, float | int | bool | str | None]] = {}
        for stream_id in self._frames:
            pipeline_fps = (
                self._frames[stream_id] - self._last_frames[stream_id]
            ) / elapsed
            pre_encode_fps = (
                self._pre_encode[stream_id]
                - self._last_pre_encode[stream_id]
            ) / elapsed
            publish_fps = (
                self._published[stream_id]
                - self._last_published[stream_id]
            ) / elapsed
            interval_detections = (
                self._detections[stream_id]
                - self._last_detections[stream_id]
            )
            interval_plate_detections = (
                self._plate_detections[stream_id]
                - self._last_plate_detections[stream_id]
            )
            interval_plate_reads = (
                self._plate_reads[stream_id]
                - self._last_plate_reads[stream_id]
            )
            garbage_analysis_fps = (
                self._garbage_frames[stream_id]
                - self._last_garbage_frames[stream_id]
            ) / elapsed
            garbage_samples = self._garbage_analysis_ms[stream_id]
            interval_corrupt_frames = (
                self._corrupt_frames[stream_id]
                - self._last_corrupt_frames[stream_id]
            )
            interval_discontinuities = (
                self._discontinuities[stream_id]
                - self._last_discontinuities[stream_id]
            )
            interval_frame_gaps = (
                self._frame_gaps[stream_id]
                - self._last_frame_gaps[stream_id]
            )
            gas_snapshot = (
                self._gas_cylinder_cache.snapshot(
                    self._stream_pad_index[stream_id]
                )
                if self._gas_cylinder_cache is not None
                else GasCylinderSnapshot()
            )
            vessel_snapshot = (
                self._vessel_detection_cache.snapshot(
                    self._stream_pad_index[stream_id]
                )
                if self._vessel_detection_cache is not None
                else VesselSnapshot()
            )
            litter_entry = self._ground_litter.get(stream_id, {})
            fishing_snapshot = (
                self._fishing_risk_cache.snapshot(
                    self._stream_pad_index[stream_id]
                )
                if self._fishing_risk_cache is not None
                else FishingRiskSnapshot()
            )
            effective_fps = min(pipeline_fps, publish_fps)
            latency_samples = self._inference_ms[stream_id]
            ordered_latency = sorted(latency_samples)
            average_inference_ms = (
                sum(latency_samples) / len(latency_samples)
                if latency_samples
                else 0.0
            )
            p95_inference_ms = (
                ordered_latency[
                    min(
                        len(ordered_latency) - 1,
                        int(len(ordered_latency) * 0.95),
                    )
                ]
                if ordered_latency
                else 0.0
            )
            streams[stream_id] = {
                "capture_fps": pipeline_fps,
                "inference_fps": pipeline_fps,
                "pre_encode_fps": pre_encode_fps,
                "publish_fps": publish_fps,
                "unique_publish_fps": publish_fps,
                "duplicate_publish_fps": 0.0,
                "tracking_fps": pipeline_fps,
                "interval_corrupt_frames": interval_corrupt_frames,
                "total_corrupt_frames": self._corrupt_frames[stream_id],
                "interval_discontinuities": interval_discontinuities,
                "total_discontinuities": self._discontinuities[stream_id],
                "interval_frame_gaps": interval_frame_gaps,
                "total_frame_gaps": self._frame_gaps[stream_id],
                "max_interframe_gap_ms": self._maximum_gap_ms[stream_id],
                "average_inference_ms": average_inference_ms,
                "p95_inference_ms": p95_inference_ms,
                "interval_detections": interval_detections,
                "total_detections": self._detections[stream_id],
                "interval_plate_detections": interval_plate_detections,
                "interval_plate_reads": interval_plate_reads,
                "garbage_analysis_fps": garbage_analysis_fps,
                "average_garbage_analysis_ms": (
                    sum(garbage_samples) / len(garbage_samples)
                    if garbage_samples
                    else 0.0
                ),
                "total_plate_detections": (
                    self._plate_detections[stream_id]
                ),
                "total_plate_reads": self._plate_reads[stream_id],
                "license_plate_enabled": bool(
                    self._license_plate_enabled.get(stream_id, False)
                ),
                "night_vision_enabled": bool(
                    self._night_vision_enabled.get(stream_id, False)
                ),
                "vision_profile": (
                    "night"
                    if self._night_vision_enabled.get(stream_id, False)
                    else "day"
                ),
                "gas_cylinder_enabled": bool(
                    self._gas_cylinder_enabled.get(stream_id, False)
                ),
                "gas_cylinder_state": gas_snapshot.state,
                "gas_cylinder_count": gas_snapshot.count,
                "gas_cylinder_result_version": (
                    gas_snapshot.result_version
                ),
                "gas_cylinder_updated_at_unix": (
                    gas_snapshot.updated_at_unix or 0.0
                ),
                "gas_cylinder_last_inference_ms": (
                    gas_snapshot.last_inference_ms
                ),
                "vessel_detection_enabled": bool(
                    self._vessel_detection_enabled.get(stream_id, False)
                ),
                "vessel_detection_state": vessel_snapshot.state,
                "vessel_detection_count": vessel_snapshot.count,
                "vessel_detection_result_version": (
                    vessel_snapshot.result_version
                ),
                "vessel_detection_last_inference_ms": (
                    vessel_snapshot.last_inference_ms
                ),
                "ground_litter_enabled": bool(
                    self._ground_litter_enabled.get(stream_id, False)
                ),
                "ground_litter_state": litter_entry.get(
                    "state",
                    "starting"
                    if self._ground_litter_enabled.get(stream_id, False)
                    else "disabled",
                ),
                "ground_litter_count": int(
                    litter_entry.get("count", 0)
                ),
                "ground_litter_message": str(
                    litter_entry.get("message", "")
                ),
                "ground_litter_result_version": int(
                    litter_entry.get("result_version", 0)
                ),
                "ground_litter_analyzed_frames": int(
                    litter_entry.get("analyzed_frames", 0)
                ),
                "ground_litter_tile_count": int(
                    litter_entry.get("tile_count", 0)
                ),
                "ground_litter_raw_candidates": int(
                    litter_entry.get("raw_candidates", 0)
                ),
                "ground_litter_active_events": int(
                    litter_entry.get("active_events", 0)
                ),
                "ground_litter_confirmed_events": int(
                    litter_entry.get("confirmed_events", 0)
                ),
                "ground_litter_cleared_events": int(
                    litter_entry.get("cleared_events", 0)
                ),
                "ground_litter_environment_state": str(
                    litter_entry.get("environment_state", "")
                ),
                "ground_litter_last_inference_ms": float(
                    litter_entry.get("last_inference_ms", 0.0)
                ),
                "ground_litter_hybrid": {
                    name: litter_entry[name]
                    for name in HYBRID_LITTER_METRIC_FIELDS
                    if name in litter_entry
                },
                "fishing_risk_enabled": bool(
                    self._fishing_risk_enabled.get(stream_id, False)
                ),
                "fishing_risk_state": fishing_snapshot.state,
                "fishing_risk_suspect_count": fishing_snapshot.count,
                "fishing_risk_maximum_score": (
                    fishing_snapshot.maximum_score
                ),
                "fishing_risk_total_events": fishing_snapshot.total_events,
                "fishing_risk_result_version": (
                    fishing_snapshot.result_version
                ),
                "ptz_verification_state": (
                    self._ptz_verification_coordinators[stream_id].state
                    if stream_id in self._ptz_verification_coordinators
                    else (
                        "error"
                        if stream_id in self._ptz_verification_errors
                        else "disabled"
                    )
                ),
                "ptz_verification_last_error": (
                    self._ptz_verification_coordinators[stream_id].last_error
                    if stream_id in self._ptz_verification_coordinators
                    else self._ptz_verification_errors.get(stream_id)
                ),
                **(
                    self._ptz_verification_coordinators[
                        stream_id
                    ].tracking_metrics
                    if stream_id in self._ptz_verification_coordinators
                    else {
                        "tracking_duration_seconds": 0.0,
                        "tracking_corrections": 0,
                        "tracking_target_width_ratio": 0.0,
                        "tracking_target_height_ratio": 0.0,
                        "tracking_last_end_reason": None,
                        "ptz_last_return_home_request_id": None,
                        "ptz_manual_hold": False,
                        "ptz_primary_candidate_count": 0,
                        "ptz_sidecar_candidate_count": 0,
                        "ptz_trigger_status": "disabled",
                        "ptz_trigger_observations": 0,
                        "ptz_trigger_required_observations": 0,
                        "ptz_trigger_cooldown_remaining_seconds": 0.0,
                    }
                ),
                "pipeline_healthy": (
                    effective_fps >= self._minimum_healthy_fps
                ),
                "minimum_healthy_fps": self._minimum_healthy_fps,
            }
        payload = {
            "group_id": self._group_id,
            "generation": self._generation,
            "uptime_seconds": now - self._started_at,
            "updated_at_unix": time.time(),
            "streams": streams,
        }
        self._atomic_write(payload)
        self._last_report_at = now
        self._last_frames = dict(self._frames)
        self._last_pre_encode = dict(self._pre_encode)
        self._last_published = dict(self._published)
        self._last_detections = dict(self._detections)
        self._last_plate_detections = dict(self._plate_detections)
        self._last_plate_reads = dict(self._plate_reads)
        self._last_garbage_frames = dict(self._garbage_frames)
        self._last_corrupt_frames = dict(self._corrupt_frames)
        self._last_discontinuities = dict(self._discontinuities)
        self._last_frame_gaps = dict(self._frame_gaps)
        self._maximum_gap_ms = {
            stream_id: 0.0 for stream_id in self._maximum_gap_ms
        }
        self._inference_ms = {
            stream_id: [] for stream_id in self._inference_ms
        }
        self._garbage_analysis_ms = {
            stream_id: [] for stream_id in self._garbage_analysis_ms
        }
        for stream_id, report in streams.items():
            LOGGER.info(
                "状态: stream=%s, pipeline=%.1f FPS, "
                "pre-encode=%.1f FPS, publish=%.1f FPS, "
                "infer=%.1f/P95 %.1f ms, garbage=%.1f FPS, "
                "gas=%s/%d, vessel=%s/%d, fishing=%s/%d, "
                "detections=%d, healthy=%s",
                stream_id[:8],
                report["inference_fps"],
                report["pre_encode_fps"],
                report["publish_fps"],
                report["average_inference_ms"],
                report["p95_inference_ms"],
                report["garbage_analysis_fps"],
                report["gas_cylinder_state"],
                report["gas_cylinder_count"],
                report["vessel_detection_state"],
                report["vessel_detection_count"],
                report["fishing_risk_state"],
                report["fishing_risk_suspect_count"],
                report["interval_detections"],
                report["pipeline_healthy"],
            )

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._metrics_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self._metrics_path)


class OverlayProcessor:
    """Duck-typed metadata processor, testable without CUDA libraries."""

    def __init__(
        self,
        policies: dict[int, StreamPolicy],
        metrics: MetricsState,
        latency_tracker: "InferenceLatencyTracker | None" = None,
        event_engines: dict[int, EventEngine] | None = None,
        garbage_overlay_cache: GarbageOverlayCache | None = None,
        gas_cylinder_cache: GasCylinderResultCache | None = None,
        vessel_detection_cache: VesselResultCache | None = None,
        ground_litter_cache: GroundLitterResultCache | None = None,
        fishing_risk_cache: FishingRiskResultCache | None = None,
        ptz_verification_coordinators: dict[
            int, PtzVerificationCoordinator
        ] | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None:
        self._policies = policies
        self._metrics = metrics
        self._latency_tracker = latency_tracker
        self._event_engines = event_engines or {}
        self._garbage_overlay_cache = garbage_overlay_cache
        self._gas_cylinder_cache = gas_cylinder_cache
        self._vessel_detection_cache = vessel_detection_cache
        self._ground_litter_cache = ground_litter_cache
        self._fishing_risk_cache = fishing_risk_cache
        self._ptz_verification_coordinators = (
            ptz_verification_coordinators or {}
        )
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._plate_consensus = {
            pad_index: PlateConsensus(
                minimum_confirmations=policy.minimum_plate_confirmations,
            )
            for pad_index, policy in policies.items()
            if policy.license_plate_enabled
        }

    def process(self, batch_meta: Any, osd: Any) -> None:
        for frame_meta in batch_meta.frame_items:
            pad_index = int(frame_meta.pad_index)
            policy = self._policies.get(pad_index)
            if policy is None:
                continue
            coordinator = self._ptz_verification_coordinators.get(pad_index)
            ptz_overlay = None
            overlay_state = getattr(coordinator, "overlay_state", None)
            if callable(overlay_state):
                ptz_overlay = overlay_state()
            active_target_rectangle = (
                ptz_overlay.target_rectangle
                if ptz_overlay is not None
                else None
            )
            active_vessel_number = (
                getattr(ptz_overlay, "vessel_number", None)
                if ptz_overlay is not None
                else None
            )
            ptz_busy = bool(
                coordinator is not None and coordinator.is_busy
            )
            detections = 0
            plate_detections = 0
            plate_reads = 0
            width, height = _frame_dimensions(
                frame_meta,
                fallback_width=self._frame_width,
                fallback_height=self._frame_height,
            )
            tracked_objects: list[TrackedObject] = []
            primary_rectangles: dict[int, NormalizedRect] = {}
            visible_vessel_rectangles: list[NormalizedRect] = []
            primary_vessel_detections: list[VesselDetection] = []
            tracking_target_assigned = False
            for object_meta in frame_meta.object_items:
                vessel_rectangle: NormalizedRect | None = None
                component_id = int(
                    getattr(
                        object_meta,
                        "unique_component_id",
                        PRIMARY_DETECTOR_UID,
                    )
                )
                if component_id == LICENSE_PLATE_DETECTOR_UID:
                    if not policy.license_plate_enabled:
                        self._hide_object(object_meta)
                        continue
                    plate_detections += 1
                    plate = self._plate_label(
                        object_meta,
                        pad_index=int(frame_meta.pad_index),
                        frame_number=int(
                            getattr(frame_meta, "frame_number", 0)
                        ),
                    )
                    if plate is not None:
                        plate_reads += 1
                    self._style_plate(object_meta, plate, osd)
                    continue
                if component_id != PRIMARY_DETECTOR_UID:
                    self._hide_object(object_meta)
                    continue
                track_id = int(getattr(object_meta, "object_id", -1))
                if track_id >= 0:
                    rectangle = object_meta.rect_params
                    normalized = NormalizedRect(
                        float(rectangle.left) / width,
                        float(rectangle.top) / height,
                        float(rectangle.width) / width,
                        float(rectangle.height) / height,
                    )
                    tracked_objects.append(
                        TrackedObject(
                            track_id=track_id,
                            class_id=int(object_meta.class_id),
                            rectangle=normalized,
                        )
                    )
                    primary_rectangles[track_id] = normalized
                if not self._object_allowed(
                    object_meta,
                    policy,
                    width,
                    height,
                    ignore_roi=ptz_busy,
                ):
                    self._hide_object(object_meta)
                    continue
                detections += 1
                if int(object_meta.class_id) in set(
                    policy.vessel_detection.class_ids
                ):
                    rectangle = object_meta.rect_params
                    vessel_rectangle = NormalizedRect(
                        float(rectangle.left) / width,
                        float(rectangle.top) / height,
                        float(rectangle.width) / width,
                        float(rectangle.height) / height,
                    )
                    visible_vessel_rectangles.append(vessel_rectangle)
                    if coordinator is not None:
                        tracker_confidence = getattr(object_meta, "tracker_confidence", None)
                        reliable_tracker_update = (
                            isinstance(tracker_confidence, (int, float))
                            and .5 <= tracker_confidence <= 1
                        )
                        # Keep IDs distinct from sidecar tracks and the -1
                        # proposal class while retaining primary tracker
                        # stability across frames.
                        primary_object_id = -(
                            max(track_id, int(frame_meta.frame_number)) + 2
                            if track_id < 0
                            else track_id + 2
                        )
                        primary_vessel_detections.append(
                            VesselDetection(
                                object_id=primary_object_id,
                                rectangle=vessel_rectangle,
                                confidence=max(
                                    float(object_meta.confidence),
                                    0.0,
                                ),
                                class_id=int(object_meta.class_id),
                                hits=1,
                                source="primary",
                                frame_id=int(frame_meta.frame_number),
                                position_updated_at=time.monotonic(),
                                observation_kind=(
                                    "image_tracker_update"
                                    if reliable_tracker_update
                                    else "legacy_unknown"
                                ),
                            )
                        )
                tracking_target = bool(
                    not tracking_target_assigned
                    and vessel_rectangle is not None
                    and active_target_rectangle is not None
                    and max(
                        _rectangle_iou(
                            vessel_rectangle,
                            active_target_rectangle,
                        ),
                        rectangle_intersection_over_smaller(
                            vessel_rectangle,
                            active_target_rectangle,
                        ),
                    )
                    >= 0.30
                )
                if policy.display_detections:
                    self._style_object(
                        object_meta,
                        policy,
                        osd,
                        tracking_target=tracking_target,
                        vessel_number=(
                            active_vessel_number if tracking_target else None
                        ),
                    )
                else:
                    # Display-only switch: metadata, tracking, metrics and the
                    # ground-litter actor filter all stay intact, only the
                    # ordinary detection box is not drawn.
                    self._hide_object(object_meta)
                tracking_target_assigned = (
                    tracking_target_assigned or tracking_target
                )
            if coordinator is not None:
                coordinator.publish_primary_detections(
                    tuple(primary_vessel_detections),
                    updated_at=time.monotonic(),
                    view_generation=(
                        getattr(coordinator, "view_generation", 0)
                        if coordinator is not None else 0
                    ),
                )
            event_engine = self._event_engines.get(int(frame_meta.pad_index))
            if event_engine is not None:
                timestamp = time.monotonic()
                event_result = event_engine.observe_tracks(
                    timestamp=timestamp,
                    objects=tracked_objects,
                )
                overlay_index = 0
                event_garbage_overlays = event_result.garbage_overlays
                if self._garbage_overlay_cache is not None:
                    detected_overlays = self._garbage_overlay_cache.overlays(
                        pad_index=int(frame_meta.pad_index),
                        options=policy.event_detection.garbage,
                        timestamp=timestamp,
                    )
                    for detected_overlay in detected_overlays:
                        if any(
                            _garbage_event_overrides_detection(
                                event_overlay,
                                detected_overlay,
                            )
                            for event_overlay in event_garbage_overlays
                        ):
                            continue
                        self._draw_event_overlay(
                            batch_meta,
                            frame_meta,
                            detected_overlay,
                            osd,
                            overlay_index,
                            width=width,
                            height=height,
                        )
                        overlay_index += 1
                for actor_overlay in event_result.actor_overlays:
                    rectangle = primary_rectangles.get(actor_overlay.track_id)
                    if rectangle is None:
                        continue
                    self._draw_event_overlay(
                        batch_meta,
                        frame_meta,
                        GarbageOverlay(
                            roi_id=actor_overlay.roi_id,
                            rectangle=rectangle,
                            label=actor_overlay.label,
                            state=actor_overlay.state,
                            elapsed_seconds=actor_overlay.elapsed_seconds,
                        ),
                        osd,
                        overlay_index,
                        width=width,
                        height=height,
                    )
                    overlay_index += 1
                for garbage_overlay in event_garbage_overlays:
                    self._draw_event_overlay(
                        batch_meta,
                        frame_meta,
                        garbage_overlay,
                        osd,
                        overlay_index,
                        width=width,
                        height=height,
                    )
                    overlay_index += 1
                for roi in policy.event_detection.rois:
                    self._draw_roi(
                        batch_meta,
                        frame_meta,
                        [list(point) for point in roi.polygon],
                        osd,
                        width=width,
                        height=height,
                    )
            if policy.roi and not ptz_busy:
                self._draw_roi(
                    batch_meta,
                    frame_meta,
                    policy.roi,
                    osd,
                    width=width,
                    height=height,
                )
            if (
                policy.ground_litter.enabled
                and self._ground_litter_cache is not None
            ):
                self._draw_ground_litter(
                    batch_meta,
                    frame_meta,
                    self._ground_litter_cache.snapshot(
                        int(frame_meta.pad_index)
                    ),
                    policy.ground_litter,
                    osd,
                    width=width,
                    height=height,
                    now=time.monotonic(),
                    metrics=self._metrics,
                    stream_id=policy.stream_id,
                )
            if (
                policy.gas_cylinder.enabled
                and self._gas_cylinder_cache is not None
            ):
                self._draw_gas_cylinders(
                    batch_meta,
                    frame_meta,
                    self._gas_cylinder_cache.snapshot(
                        int(frame_meta.pad_index)
                    ),
                    policy.gas_cylinder,
                    osd,
                    width=width,
                    height=height,
                )
            if (
                policy.vessel_detection.enabled
                and self._vessel_detection_cache is not None
            ):
                fishing_snapshot = (
                    self._fishing_risk_cache.snapshot(
                        int(frame_meta.pad_index)
                    )
                    if self._fishing_risk_cache is not None
                    else FishingRiskSnapshot()
                )
                if (
                    policy.fishing_risk.enabled
                    and policy.fishing_risk.display_risk
                    and not ptz_busy
                ):
                    for zone in policy.fishing_risk.zones:
                        self._draw_roi(
                            batch_meta,
                            frame_meta,
                            [list(point) for point in zone.polygon],
                            osd,
                            width=width,
                            height=height,
                        )
                self._draw_vessels(
                    batch_meta,
                    frame_meta,
                    self._vessel_detection_cache.snapshot(
                        int(frame_meta.pad_index)
                    ),
                    policy.vessel_detection,
                    osd,
                    width=width,
                    height=height,
                    existing_rectangles=tuple(
                        visible_vessel_rectangles
                    ),
                    fishing_options=policy.fishing_risk,
                    fishing_snapshot=fishing_snapshot,
                    monitoring_view=not ptz_busy,
                    active_target_rectangle=active_target_rectangle,
                    active_vessel_number=active_vessel_number,
                )
            if (
                ptz_overlay is not None
                and policy.ptz_verification.display_operation_log
            ):
                self._draw_ptz_operation_log(
                    batch_meta,
                    frame_meta,
                    ptz_overlay.state,
                    ptz_overlay.operation_lines,
                    active=active_target_rectangle is not None,
                    osd=osd,
                    width=width,
                )
            latency_ms = (
                self._latency_tracker.finish(
                    int(frame_meta.pad_index),
                    int(frame_meta.frame_number),
                )
                if self._latency_tracker is not None
                else None
            )
            self._metrics.observe_license_plates(
                policy.stream_id,
                plate_detections,
                plate_reads,
            )
            self._metrics.observe_frame(
                policy.stream_id,
                detections,
                latency_ms,
            )

    def _plate_label(
        self,
        object_meta: Any,
        *,
        pad_index: int,
        frame_number: int,
    ) -> str | None:
        consensus = self._plate_consensus.get(pad_index)
        if consensus is None:
            return None
        track_id = int(getattr(object_meta, "object_id", -1))
        for label in self._classifier_labels(object_meta):
            confirmed = consensus.observe(
                pad_index=pad_index,
                track_id=track_id,
                frame_number=frame_number,
                value=label,
            )
            if confirmed is not None:
                return confirmed
        return consensus.get(
            pad_index=pad_index,
            track_id=track_id,
            frame_number=frame_number,
        )

    @staticmethod
    def _classifier_labels(object_meta: Any) -> list[str | bytes]:
        result: list[str | bytes] = []
        classifiers = getattr(object_meta, "classifier_items", ())
        for classifier in classifiers:
            component_id = getattr(
                classifier,
                "unique_component_id",
                LICENSE_PLATE_RECOGNIZER_UID,
            )
            if callable(component_id):
                component_id = component_id()
            if int(component_id) != LICENSE_PLATE_RECOGNIZER_UID:
                continue
            label_items = getattr(classifier, "label_items", None)
            if label_items is not None:
                for label_info in label_items:
                    value = getattr(label_info, "result_label", "")
                    if value:
                        result.append(value)
                continue
            get_label = getattr(classifier, "get_n_label", None)
            if not callable(get_label):
                get_label = getattr(classifier, "get_label", None)
            if callable(get_label):
                count = getattr(classifier, "n_labels", 1)
                if callable(count):
                    count = count()
                for index in range(int(count)):
                    value = get_label(index)
                    if value:
                        result.append(value)
        return result

    @staticmethod
    def _object_allowed(
        object_meta: Any,
        policy: StreamPolicy,
        width: float,
        height: float,
        *,
        ignore_roi: bool = False,
    ) -> bool:
        class_id = int(object_meta.class_id)
        if policy.classes is not None and class_id not in policy.classes:
            return False
        confidence = float(object_meta.confidence)
        if confidence >= 0 and confidence < policy.conf:
            return False
        if policy.roi and not ignore_roi:
            rectangle = object_meta.rect_params
            center = (
                (float(rectangle.left) + float(rectangle.width) / 2) / width,
                (float(rectangle.top) + float(rectangle.height) / 2) / height,
            )
            if not _point_in_polygon(center, policy.roi):
                return False
        return True

    @staticmethod
    def _hide_object(object_meta: Any) -> None:
        object_meta.rect_params.border_width = 0
        object_meta.text_params.display_text = b""

    @staticmethod
    def _style_object(
        object_meta: Any,
        policy: StreamPolicy,
        osd: Any,
        *,
        tracking_target: bool = False,
        vessel_number: str | None = None,
    ) -> None:
        class_id = int(object_meta.class_id)
        label = policy.labels.get(class_id, f"类别{class_id}")
        rectangle = object_meta.rect_params
        rectangle.border_width = 5 if tracking_target else 3
        rectangle.border_color = (
            osd.Color(1.0, 0.08, 0.08, 1.0)
            if tracking_target
            else osd.Color(0.0, 1.0, 0.0, 1.0)
        )
        text = object_meta.text_params
        if tracking_target:
            label = f"追踪中｜{label}"
            if vessel_number:
                label += f"｜船号 {vessel_number}"
        text.display_text = label.encode("utf-8")
        text.x_offset = max(int(rectangle.left), 0)
        text.y_offset = max(int(rectangle.top) - 24, 0)
        # ObjectMetadata exposes the native NvOSD_TextParams binding. Its
        # names differ from the higher-level osd.Text helper used by NVIDIA's
        # display-meta examples.
        text.font_params.name = osd.FontFamily.Serif
        text.font_params.size = 18
        text.font_params.color = osd.Color(1.0, 1.0, 1.0, 1.0)
        text.set_bg_clr = True
        text.text_bg_clr = (
            osd.Color(0.65, 0.0, 0.0, 0.92)
            if tracking_target
            else osd.Color(0.0, 0.45, 0.0, 0.9)
        )

    @staticmethod
    def _draw_ptz_operation_log(
        batch_meta: Any,
        frame_meta: Any,
        state: str,
        operation_lines: tuple[str, ...],
        *,
        active: bool,
        osd: Any,
        width: float,
    ) -> None:
        if not operation_lines and not active:
            return
        state_labels = {
            "verifying": "目标复核",
            "tracking": "持续追踪",
            "reacquiring": "重新捕获",
            "returning_home": "返回HOME",
            "manual_hold": "已锁停",
            "recovery_required": "等待恢复",
            "degraded": "控制异常",
            "running": "等待目标",
            "stopped": "已停止",
        }
        display_meta = batch_meta.acquire_display_meta()
        panel_x = max(int(width) - 430, 12)
        color = (
            osd.Color(0.72, 0.0, 0.0, 0.90)
            if active
            else osd.Color(0.08, 0.08, 0.08, 0.82)
        )
        lines = (
            f"PTZ｜{state_labels.get(state, state)}",
            *operation_lines[-4:],
        )
        for index, value in enumerate(lines):
            text = osd.Text()
            text.display_text = value.encode("utf-8")
            text.x_offset = panel_x
            text.y_offset = 12 + index * 23
            text.font.name = osd.FontFamily.Serif
            text.font.size = 17 if index == 0 else 14
            text.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
            text.set_bg_color = True
            text.bg_color = color
            display_meta.add_text(text)
        frame_meta.append(display_meta)

    @staticmethod
    def _style_plate(
        object_meta: Any,
        plate: str | None,
        osd: Any,
    ) -> None:
        rectangle = object_meta.rect_params
        rectangle.border_width = 3
        rectangle.border_color = osd.Color(0.0, 0.65, 1.0, 1.0)
        text = object_meta.text_params
        label = f"车牌：{plate}" if plate else "车牌识别中"
        text.display_text = label.encode("utf-8")
        text.x_offset = max(int(rectangle.left), 0)
        text.y_offset = max(int(rectangle.top) - 24, 0)
        text.font_params.name = osd.FontFamily.Serif
        text.font_params.size = 18
        text.font_params.color = osd.Color(1.0, 1.0, 1.0, 1.0)
        text.set_bg_clr = True
        text.text_bg_clr = osd.Color(0.0, 0.25, 0.55, 0.9)

    @staticmethod
    def _draw_roi(
        batch_meta: Any,
        frame_meta: Any,
        roi: list[list[float]],
        osd: Any,
        *,
        width: float | None = None,
        height: float | None = None,
    ) -> None:
        if width is None or height is None:
            width, height = _frame_dimensions(frame_meta)
        pixel_width = max(int(width), 1)
        pixel_height = max(int(height), 1)

        def x_coordinate(value: float) -> int:
            return min(max(int(value * pixel_width), 0), pixel_width - 1)

        def y_coordinate(value: float) -> int:
            return min(max(int(value * pixel_height), 0), pixel_height - 1)

        display_meta = batch_meta.acquire_display_meta()
        for start, end in zip(roi, roi[1:] + roi[:1]):
            line = osd.Line()
            line.x1 = x_coordinate(start[0])
            line.y1 = y_coordinate(start[1])
            line.x2 = x_coordinate(end[0])
            line.y2 = y_coordinate(end[1])
            line.width = 3
            line.color = osd.Color(1.0, 0.75, 0.0, 1.0)
            display_meta.add_line(line)
        frame_meta.append(display_meta)

    @staticmethod
    def _draw_event_overlay(
        batch_meta: Any,
        frame_meta: Any,
        overlay: Any,
        osd: Any,
        overlay_index: int,
        *,
        width: float | None = None,
        height: float | None = None,
    ) -> None:
        if width is None or height is None:
            width, height = _frame_dimensions(frame_meta)
        pixel_width = max(int(width), 1)
        pixel_height = max(int(height), 1)
        if overlay.state == "confirmed":
            color = osd.Color(1.0, 0.1, 0.1, 1.0)
        elif overlay.state == "detected":
            color = osd.Color(0.0, 0.75, 0.2, 1.0)
        else:
            color = osd.Color(1.0, 0.65, 0.0, 1.0)
        display_meta = batch_meta.acquire_display_meta()
        rectangle = overlay.rectangle
        text_x = 12
        text_y = 42 + overlay_index * 34
        if rectangle is not None:
            left = min(
                max(int(rectangle.left * pixel_width), 0),
                pixel_width - 1,
            )
            top = min(
                max(int(rectangle.top * pixel_height), 0),
                pixel_height - 1,
            )
            right = min(
                max(int((rectangle.left + rectangle.width) * pixel_width), 0),
                pixel_width - 1,
            )
            bottom = min(
                max(int((rectangle.top + rectangle.height) * pixel_height), 0),
                pixel_height - 1,
            )
            points = (
                (left, top, right, top),
                (right, top, right, bottom),
                (right, bottom, left, bottom),
                (left, bottom, left, top),
            )
            for x1, y1, x2, y2 in points:
                line = osd.Line()
                line.x1 = x1
                line.y1 = y1
                line.x2 = x2
                line.y2 = y2
                line.width = 4
                line.color = color
                display_meta.add_line(line)
            text_x = max(left, 0)
            text_y = max(top - 28, 0)
        text = osd.Text()
        text.display_text = overlay.label.encode("utf-8")
        text.x_offset = text_x
        text.y_offset = text_y
        text.font.name = osd.FontFamily.Serif
        text.font.size = 18
        text.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
        text.set_bg_color = True
        text.bg_color = color
        display_meta.add_text(text)
        frame_meta.append(display_meta)

    @staticmethod
    def _draw_ground_litter(
        batch_meta: Any,
        frame_meta: Any,
        snapshot: GroundLitterSnapshot,
        options: GroundLitterDetectionOptions,
        osd: Any,
        *,
        width: float,
        height: float,
        now: float,
        metrics: Any = None,
        stream_id: str = "",
    ) -> None:
        """Draw reviewed ground regions plus cached litter candidates."""
        pixel_width = max(int(width), 1)
        pixel_height = max(int(height), 1)
        if options.display_zones:
            for zone in options.zones:
                OverlayProcessor._draw_roi(
                    batch_meta,
                    frame_meta,
                    [list(point) for point in zone.polygon],
                    osd,
                    width=width,
                    height=height,
                )
                for exclusion in zone.exclude_zones:
                    OverlayProcessor._draw_exclusion(
                        batch_meta,
                        frame_meta,
                        [list(point) for point in exclusion],
                        osd,
                        width=width,
                        height=height,
                    )
            for exclusion in options.overlay_exclude_zones:
                OverlayProcessor._draw_exclusion(
                    batch_meta,
                    frame_meta,
                    [list(point) for point in exclusion],
                    osd,
                    width=width,
                    height=height,
                )
        fresh = snapshot_is_fresh(snapshot, options, now=now)
        detections = list(snapshot.detections) if fresh else []
        if metrics is not None and stream_id:
            metrics.observe_ground_litter(
                stream_id,
                state=snapshot.state,
                count=len(detections),
                result_version=snapshot.result_version,
                updated_at=snapshot.updated_at,
                last_inference_ms=snapshot.last_inference_ms,
                analyzed_frames=snapshot.analyzed_frames,
                tile_count=snapshot.tile_count,
                raw_candidates=snapshot.raw_candidates,
                active_events=snapshot.active_events,
                confirmed_events=snapshot.confirmed_events,
                cleared_events=snapshot.cleared_events,
                environment_state=snapshot.environment_state,
                message=snapshot.message,
                extra={
                    "semantic_raw_candidates": snapshot.semantic_raw_candidates,
                    "semantic_retained_candidates": (
                        snapshot.semantic_retained_candidates
                    ),
                    "prior_raw_candidates": snapshot.prior_raw_candidates,
                    "prior_retained_candidates": (
                        snapshot.prior_retained_candidates
                    ),
                    "semantic_only_active": snapshot.semantic_only_active,
                    "semantic_only_confirmed": snapshot.semantic_only_confirmed,
                    "prior_only_active": snapshot.prior_only_active,
                    "prior_only_confirmed": snapshot.prior_only_confirmed,
                    "fused_active": snapshot.fused_active,
                    "fused_confirmed": snapshot.fused_confirmed,
                    "cross_source_merges": snapshot.cross_source_merges,
                    "prior_environment_state": snapshot.prior_environment_state,
                    "semantic_model_runs_full": snapshot.semantic_model_runs_full,
                    "semantic_model_runs_crop": snapshot.semantic_model_runs_crop,
                    "semantic_crop_raw_candidates": (
                        snapshot.semantic_crop_raw_candidates
                    ),
                    "semantic_crop_unmatched_candidates": (
                        snapshot.semantic_crop_unmatched_candidates
                    ),
                    "input_frame_age_ms": snapshot.input_frame_age_ms,
                    "dropped_analysis_frames": snapshot.dropped_analysis_frames,
                    "last_prior_ms": snapshot.last_prior_ms,
                    "last_full_scan_ms": snapshot.last_full_scan_ms,
                    "last_crop_batch_ms": snapshot.last_crop_batch_ms,
                    "last_total_ms": snapshot.last_total_ms,
                    "branch_state": snapshot.branch_state,
                    "branch_message": snapshot.branch_message,
                },
            )
        if snapshot.state == "error":
            color = osd.Color(1.0, 0.15, 0.1, 1.0)
            title = f"{options.label}识别异常"
        elif snapshot.state == "abstaining":
            color = osd.Color(1.0, 0.65, 0.0, 1.0)
            title = f"{options.label}：环境变化，暂停判断"
        elif snapshot.state in {"starting", "disabled"}:
            color = osd.Color(1.0, 0.65, 0.0, 1.0)
            title = f"{options.label}识别启动中"
        elif detections:
            color = osd.Color(1.0, 0.35, 0.0, 1.0)
            title = f"{options.label}：{len(detections)}"
        else:
            color = osd.Color(0.0, 0.85, 0.2, 1.0)
            title = f"{options.label}：未发现"
        chunks = [
            detections[index : index + 4]
            for index in range(0, len(detections), 4)
        ] or [[]]
        for chunk_index, chunk in enumerate(chunks):
            display_meta = batch_meta.acquire_display_meta()
            if chunk_index == 0:
                header = osd.Text()
                header.display_text = title.encode("utf-8")
                header.x_offset = 12
                header.y_offset = 12
                header.font.name = osd.FontFamily.Serif
                header.font.size = 22
                header.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
                header.set_bg_color = True
                header.bg_color = color
                display_meta.add_text(header)
            for detection in chunk:
                rectangle = detection.rectangle
                left = min(
                    max(int(rectangle.left * pixel_width), 0),
                    pixel_width - 1,
                )
                top = min(
                    max(int(rectangle.top * pixel_height), 0),
                    pixel_height - 1,
                )
                right = min(
                    max(
                        int(
                            (rectangle.left + rectangle.width)
                            * pixel_width
                        ),
                        0,
                    ),
                    pixel_width - 1,
                )
                bottom = min(
                    max(
                        int(
                            (rectangle.top + rectangle.height)
                            * pixel_height
                        ),
                        0,
                    ),
                    pixel_height - 1,
                )
                for x1, y1, x2, y2 in (
                    (left, top, right, top),
                    (right, top, right, bottom),
                    (right, bottom, left, bottom),
                    (left, bottom, left, top),
                ):
                    line = osd.Line()
                    line.x1 = x1
                    line.y1 = y1
                    line.x2 = x2
                    line.y2 = y2
                    line.width = 3
                    line.color = color
                    display_meta.add_line(line)
                if not (
                    options.display_class or options.display_confidence
                ):
                    continue
                label = options.label
                if options.display_class and detection.class_name:
                    label += f" {detection.class_name}"
                if options.display_confidence:
                    label += f" {detection.confidence:.2f}"
                text = osd.Text()
                text.display_text = label.encode("utf-8")
                text.x_offset = left
                text.y_offset = max(top - 24, 0)
                text.font.name = osd.FontFamily.Serif
                text.font.size = 18
                text.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
                text.set_bg_color = True
                text.bg_color = color
                display_meta.add_text(text)
            frame_meta.append(display_meta)

    @staticmethod
    def _draw_exclusion(
        batch_meta: Any,
        frame_meta: Any,
        polygon: list[list[float]],
        osd: Any,
        *,
        width: float,
        height: float,
    ) -> None:
        pixel_width = max(int(width), 1)
        pixel_height = max(int(height), 1)
        display_meta = batch_meta.acquire_display_meta()
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            line = osd.Line()
            line.x1 = min(max(int(start[0] * pixel_width), 0), pixel_width - 1)
            line.y1 = min(
                max(int(start[1] * pixel_height), 0),
                pixel_height - 1,
            )
            line.x2 = min(max(int(end[0] * pixel_width), 0), pixel_width - 1)
            line.y2 = min(
                max(int(end[1] * pixel_height), 0),
                pixel_height - 1,
            )
            line.width = 2
            line.color = osd.Color(1.0, 0.15, 0.15, 1.0)
            display_meta.add_line(line)
        frame_meta.append(display_meta)

    @staticmethod
    def _draw_gas_cylinders(
        batch_meta: Any,
        frame_meta: Any,
        snapshot: GasCylinderSnapshot,
        options: GasCylinderOptions,
        osd: Any,
        *,
        width: float,
        height: float,
    ) -> None:
        """Batch line metadata so dozens of cached boxes stay inexpensive."""
        alarm_active = snapshot.count > options.alarm_threshold
        if alarm_active:
            color = osd.Color(1.0, 0.0, 0.0, 1.0)
            title = f"燃气瓶数量：{snapshot.count}（超量告警）"
        elif snapshot.state == "stable":
            color = osd.Color(0.0, 0.85, 0.2, 1.0)
            title = f"燃气瓶数量：{snapshot.count}"
        elif snapshot.state == "dirty":
            color = osd.Color(1.0, 0.65, 0.0, 1.0)
            title = f"上次燃气瓶：{snapshot.count}（场景变化）"
        elif snapshot.state == "error":
            color = osd.Color(1.0, 0.15, 0.1, 1.0)
            title = f"燃气瓶识别异常，上次：{snapshot.count}"
        else:
            color = osd.Color(1.0, 0.65, 0.0, 1.0)
            title = (
                f"燃气瓶重新识别中，上次：{snapshot.count}"
                if snapshot.count
                else "燃气瓶识别中"
            )
        detections = list(snapshot.detections)
        chunks = [detections[index : index + 4] for index in range(0, len(detections), 4)]
        if not chunks:
            chunks = [[]]
        pixel_width = max(int(width), 1)
        pixel_height = max(int(height), 1)
        for chunk_index, chunk in enumerate(chunks):
            display_meta = batch_meta.acquire_display_meta()
            if chunk_index == 0:
                header = osd.Text()
                header.display_text = title.encode("utf-8")
                header.x_offset = 12
                header.y_offset = 12
                header.font.name = osd.FontFamily.Serif
                header.font.size = 22
                header.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
                header.set_bg_color = True
                header.bg_color = color
                display_meta.add_text(header)
            for detection in chunk:
                rectangle = detection.rectangle
                left = min(max(int(rectangle.left * pixel_width), 0), pixel_width - 1)
                top = min(max(int(rectangle.top * pixel_height), 0), pixel_height - 1)
                right = min(
                    max(int((rectangle.left + rectangle.width) * pixel_width), 0),
                    pixel_width - 1,
                )
                bottom = min(
                    max(int((rectangle.top + rectangle.height) * pixel_height), 0),
                    pixel_height - 1,
                )
                for x1, y1, x2, y2 in (
                    (left, top, right, top),
                    (right, top, right, bottom),
                    (right, bottom, left, bottom),
                    (left, bottom, left, top),
                ):
                    line = osd.Line()
                    line.x1 = x1
                    line.y1 = y1
                    line.x2 = x2
                    line.y2 = y2
                    line.width = 3
                    line.color = color
                    display_meta.add_line(line)
                if options.display_ids:
                    label = osd.Text()
                    label.display_text = f"燃气瓶 #{detection.object_id}".encode(
                        "utf-8"
                    )
                    label.x_offset = left
                    label.y_offset = max(top - 24, 0)
                    label.font.name = osd.FontFamily.Serif
                    label.font.size = 16
                    label.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
                    label.set_bg_color = True
                    label.bg_color = color
                    display_meta.add_text(label)
            frame_meta.append(display_meta)

    @staticmethod
    def _draw_vessels(
        batch_meta: Any,
        frame_meta: Any,
        snapshot: VesselSnapshot,
        options: VesselDetectionOptions,
        osd: Any,
        *,
        width: float,
        height: float,
        existing_rectangles: tuple[NormalizedRect, ...] = (),
        fishing_options: FishingRiskOptions | None = None,
        fishing_snapshot: FishingRiskSnapshot = FishingRiskSnapshot(),
        monitoring_view: bool = True,
        active_target_rectangle: NormalizedRect | None = None,
        active_vessel_number: str | None = None,
    ) -> None:
        """Draw temporally confirmed vessel boxes from the lossy sidecar."""
        if options.display_roi and monitoring_view:
            if options.roi is not None:
                OverlayProcessor._draw_roi(
                    batch_meta,
                    frame_meta,
                    [list(point) for point in options.roi],
                    osd,
                    width=width,
                    height=height,
                )
            for polygon in options.exclude_rois:
                OverlayProcessor._draw_roi(
                    batch_meta,
                    frame_meta,
                    [list(point) for point in polygon],
                    osd,
                    width=width,
                    height=height,
                )

        now = time.monotonic()
        stale_after = options.hold_seconds + max(
            2.0 / options.analysis_fps,
            1.0,
        )
        fresh = (
            snapshot.updated_at is not None
            and now - snapshot.updated_at <= stale_after
        )
        all_detections = list(snapshot.detections) if fresh else []
        detections = [
            detection
            for detection in all_detections
            if not any(
                max(
                    _rectangle_iou(detection.rectangle, existing),
                    rectangle_intersection_over_smaller(
                        detection.rectangle,
                        existing,
                    ),
                )
                >= 0.5
                for existing in existing_rectangles
            )
        ]
        if not options.display_proposals:
            detections = [
                detection
                for detection in detections
                if detection.class_id != SMALL_TARGET_PROPOSAL_CLASS_ID
            ]
        risk_enabled = bool(
            fishing_options is not None
            and fishing_options.enabled
            and fishing_options.display_risk
        )
        risk_fresh = bool(
            risk_enabled
            and fishing_snapshot.updated_at is not None
            and now - fishing_snapshot.updated_at
            <= max(
                fishing_options.rules.track_lost_seconds
                if fishing_options is not None
                else 0.0,
                2.0,
            )
        )
        risk_by_object_id = {
            suspect.vessel_object_id: suspect
            for suspect in fishing_snapshot.suspects
        } if risk_fresh else {}
        if snapshot.state == "error":
            color = osd.Color(1.0, 0.15, 0.1, 1.0)
            title = "船舶检测异常"
        elif not fresh:
            color = osd.Color(1.0, 0.65, 0.0, 1.0)
            title = "船舶检测等待结果"
        else:
            color = osd.Color(0.0, 0.85, 1.0, 1.0)
            proposal_count = (
                sum(
                    item.class_id == SMALL_TARGET_PROPOSAL_CLASS_ID
                    for item in all_detections
                )
                if options.display_proposals
                else 0
            )
            confirmed_count = len(existing_rectangles) + sum(
                item.class_id != SMALL_TARGET_PROPOSAL_CLASS_ID
                for item in detections
            )
            title = f"船舶：{confirmed_count}"
            if proposal_count:
                title += f"｜疑似目标：{proposal_count}"
            if risk_enabled:
                title += f"｜风险候选：{len(risk_by_object_id)}"
                if risk_by_object_id:
                    color = osd.Color(1.0, 0.25, 0.05, 1.0)

        chunks = [
            detections[index : index + 4]
            for index in range(0, len(detections), 4)
        ] or [[]]
        pixel_width = max(int(width), 1)
        pixel_height = max(int(height), 1)
        active_detection: VesselDetection | None = None
        if active_target_rectangle is not None and detections:
            candidate = max(
                detections,
                key=lambda item: max(
                    _rectangle_iou(item.rectangle, active_target_rectangle),
                    rectangle_intersection_over_smaller(
                        item.rectangle,
                        active_target_rectangle,
                    ),
                ),
            )
            if max(
                _rectangle_iou(candidate.rectangle, active_target_rectangle),
                rectangle_intersection_over_smaller(
                    candidate.rectangle,
                    active_target_rectangle,
                ),
            ) >= 0.30:
                active_detection = candidate
        for chunk_index, chunk in enumerate(chunks):
            display_meta = batch_meta.acquire_display_meta()
            if chunk_index == 0:
                header = osd.Text()
                header.display_text = title.encode("utf-8")
                header.x_offset = 12
                header.y_offset = 48
                header.font.name = osd.FontFamily.Serif
                header.font.size = 22
                header.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
                header.set_bg_color = True
                header.bg_color = color
                display_meta.add_text(header)
            for detection in chunk:
                tracking_target = detection is active_detection
                risk = risk_by_object_id.get(detection.object_id)
                detection_color = (
                    osd.Color(1.0, 0.08, 0.08, 1.0)
                    if tracking_target
                    else osd.Color(1.0, 0.15, 0.05, 1.0)
                    if risk is not None
                    and fishing_options is not None
                    and risk.risk_score >= fishing_options.alert_score
                    else (
                        osd.Color(1.0, 0.65, 0.0, 1.0)
                        if risk is not None
                        else color
                    )
                )
                rectangle = detection.rectangle
                left = min(
                    max(int(rectangle.left * pixel_width), 0),
                    pixel_width - 1,
                )
                top = min(
                    max(int(rectangle.top * pixel_height), 0),
                    pixel_height - 1,
                )
                right = min(
                    max(
                        int(
                            (rectangle.left + rectangle.width)
                            * pixel_width
                        ),
                        0,
                    ),
                    pixel_width - 1,
                )
                bottom = min(
                    max(
                        int(
                            (rectangle.top + rectangle.height)
                            * pixel_height
                        ),
                        0,
                    ),
                    pixel_height - 1,
                )
                for x1, y1, x2, y2 in (
                    (left, top, right, top),
                    (right, top, right, bottom),
                    (right, bottom, left, bottom),
                    (left, bottom, left, top),
                ):
                    line = osd.Line()
                    line.x1 = x1
                    line.y1 = y1
                    line.x2 = x2
                    line.y2 = y2
                    line.width = 5 if tracking_target else 4
                    line.color = detection_color
                    display_meta.add_line(line)
                label = osd.Text()
                if tracking_target:
                    label_value = "追踪中｜船"
                    if active_vessel_number:
                        label_value += f"｜船号 {active_vessel_number}"
                elif risk is not None:
                    label_value = f"疑似捕捞线索 {risk.risk_score}分"
                elif detection.class_id == SMALL_TARGET_PROPOSAL_CLASS_ID:
                    label_value = (
                        f"疑似目标 #{detection.object_id}"
                        if options.display_ids
                        else "疑似目标"
                    )
                else:
                    label_value = (
                        f"船 #{detection.object_id}"
                        if options.display_ids
                        else "船"
                    )
                label.display_text = label_value.encode("utf-8")
                label.x_offset = left
                label.y_offset = max(top - 24, 0)
                label.font.name = osd.FontFamily.Serif
                label.font.size = 18
                label.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
                label.set_bg_color = True
                label.bg_color = detection_color
                display_meta.add_text(label)
            frame_meta.append(display_meta)


class InferenceLatencyTracker:
    def __init__(self, max_pending: int = 4_096) -> None:
        self._max_pending = max_pending
        self._lock = threading.Lock()
        self._started: dict[tuple[int, int], float] = {}

    def start_batch(self, batch_meta: Any) -> None:
        now = time.perf_counter()
        with self._lock:
            for frame_meta in batch_meta.frame_items:
                key = (
                    int(frame_meta.pad_index),
                    int(frame_meta.frame_number),
                )
                self._started[key] = now
            overflow = len(self._started) - self._max_pending
            if overflow > 0:
                for key in list(self._started)[:overflow]:
                    self._started.pop(key, None)

    def finish(self, pad_index: int, frame_number: int) -> float | None:
        with self._lock:
            started = self._started.pop(
                (pad_index, frame_number),
                None,
            )
        if started is None:
            return None
        return (time.perf_counter() - started) * 1_000


class GarbageMetadataProcessor:
    """Translate low-rate garbage detector metadata into temporal snapshots."""

    def __init__(
        self,
        *,
        policies: dict[int, StreamPolicy],
        event_engines: dict[int, EventEngine],
        labels: list[str],
        interval: int,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None:
        self._policies = policies
        self._event_engines = event_engines
        self._labels = labels
        self._interval = max(int(interval), 0)
        self._frame_width = frame_width
        self._frame_height = frame_height

    def process(self, batch_meta: Any) -> None:
        snapshots = self.collect(batch_meta, timestamp=time.monotonic())
        for (pad_index, roi_id), snapshot in snapshots.items():
            engine = self._event_engines.get(pad_index)
            if engine is not None:
                engine.observe_garbage(roi_id=roi_id, snapshot=snapshot)

    def collect(
        self,
        batch_meta: Any,
        *,
        timestamp: float,
    ) -> dict[tuple[int, str], GarbageSnapshot]:
        snapshots: dict[tuple[int, str], GarbageSnapshot] = {}
        for frame_meta in batch_meta.frame_items:
            pad_index = int(frame_meta.pad_index)
            engine = self._event_engines.get(pad_index)
            policy = self._policies.get(pad_index)
            if engine is None or policy is None:
                continue
            frame_number = int(getattr(frame_meta, "frame_number", 0))
            if self._interval and (frame_number - 1) % (
                self._interval + 1
            ):
                continue
            frame_snapshots, _actors = _collect_frame_observations(
                frame_meta,
                policy,
                self._labels,
                timestamp=timestamp,
                frame_width=self._frame_width,
                frame_height=self._frame_height,
            )
            for roi_id, snapshot in frame_snapshots.items():
                snapshots[(pad_index, roi_id)] = snapshot
        return snapshots


def _collect_frame_observations(
    frame_meta: Any,
    policy: StreamPolicy,
    labels: list[str],
    *,
    timestamp: float,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> tuple[dict[str, GarbageSnapshot], list[TrackedObject]]:
    """Read ServiceMaker's one-shot object iterator exactly once."""
    pipeline_width, pipeline_height = _frame_dimensions(
        frame_meta,
        fallback_width=frame_width,
        fallback_height=frame_height,
    )
    # The garbage branch scales both pixels and NvDsObjectMeta rectangles to
    # its 640x360 caps, while frame_meta.pipeline_* still reports the upstream
    # mux size (typically 1920x1080).  Prefer explicit branch dimensions when
    # supplied or every online box is normalised to one third of its actual
    # position and is drawn in the upper-left of the published frame.
    width = float(frame_width) if frame_width is not None else pipeline_width
    height = (
        float(frame_height) if frame_height is not None else pipeline_height
    )
    width = max(width, 1.0)
    height = max(height, 1.0)
    frame_number = int(getattr(frame_meta, "frame_number", 0))
    actor_classes = set(policy.event_detection.person_classes)
    actor_classes.update(policy.event_detection.vehicle_classes)
    threshold = policy.event_detection.garbage.minimum_confidence
    detections: list[tuple[NormalizedRect, float, str]] = []
    actors: list[TrackedObject] = []
    for object_meta in frame_meta.object_items:
        component_id = int(
            getattr(
                object_meta,
                "unique_component_id",
                PRIMARY_DETECTOR_UID,
            )
        )
        class_id = int(getattr(object_meta, "class_id", -1))
        track_id = int(getattr(object_meta, "object_id", -1))
        rectangle = object_meta.rect_params
        normalized = NormalizedRect(
            float(rectangle.left) / width,
            float(rectangle.top) / height,
            float(rectangle.width) / width,
            float(rectangle.height) / height,
        )
        if component_id == PRIMARY_DETECTOR_UID and class_id in actor_classes:
            if 0 <= track_id < 2**63:
                actors.append(TrackedObject(track_id, class_id, normalized))
            continue
        if component_id != GARBAGE_DETECTOR_UID:
            continue
        confidence = float(getattr(object_meta, "confidence", 0.0))
        if confidence < threshold:
            continue
        label = labels[class_id] if 0 <= class_id < len(labels) else "垃圾"
        garbage_options = policy.event_detection.garbage
        if garbage_options.detection_mode == "pile":
            if label.strip().lower() != "garbage":
                continue
            label = "trash pile"
        elif label not in garbage_options.prompts:
            continue
        detections.append((normalized, confidence, label))

    snapshots: dict[str, GarbageSnapshot] = {}
    for roi in policy.event_detection.rois:
        if not roi.garbage_enabled:
            continue
        selected = _deduplicate_detections(
            [
                item
                for item in detections
                if _point_in_polygon(
                    item[0].center,
                    [list(point) for point in roi.polygon],
                )
            ]
        )
        if policy.event_detection.garbage.detection_mode == "pile":
            selected = _merge_pile_detections(
                selected,
                policy.event_detection.garbage,
            )
        snapshots[roi.roi_id] = GarbageSnapshot(
            timestamp=timestamp,
            frame_number=frame_number,
            area_ratio=min(
                sum(item[0].width * item[0].height for item in selected),
                1.0,
            ),
            regions=tuple(item[0] for item in selected),
            semantic_confidence=max(
                (item[1] for item in selected),
                default=0.0,
            ),
            object_type=(
                max(selected, key=lambda item: item[1])[2]
                if selected
                else "垃圾"
            ),
            detections=tuple(
                GarbageDetection(
                    rectangle=rectangle,
                    confidence=confidence,
                    object_type=label,
                )
                for rectangle, confidence, label in selected
            ),
        )
    return snapshots, actors


def _deduplicate_detections(
    detections: list[tuple[NormalizedRect, float, str]],
) -> list[tuple[NormalizedRect, float, str]]:
    """Suppress overlapping cross-prompt boxes for the same physical item."""
    selected: list[tuple[NormalizedRect, float, str]] = []
    for detection in sorted(
        detections,
        key=lambda item: item[1],
        reverse=True,
    ):
        if any(
            _rectangle_iou(detection[0], existing[0]) >= 0.5
            for existing in selected
        ):
            continue
        selected.append(detection)
    return selected


def _merge_pile_detections(
    detections: list[tuple[NormalizedRect, float, str]],
    options: GarbageAnalysisOptions,
) -> list[tuple[NormalizedRect, float, str]]:
    """Cluster nearby garbage parts into stable pile-level rectangles."""
    if not detections:
        return []
    remaining = set(range(len(detections)))
    clusters: list[list[int]] = []
    while remaining:
        seed = remaining.pop()
        cluster = [seed]
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbours = [
                index
                for index in remaining
                if _rectangle_gap(
                    detections[current][0],
                    detections[index][0],
                ) <= options.pile_merge_distance
            ]
            for index in neighbours:
                remaining.remove(index)
                cluster.append(index)
                frontier.append(index)
        clusters.append(cluster)

    pile_candidates: list[tuple[NormalizedRect, float, int]] = []
    for cluster in clusters:
        rectangles = [detections[index][0] for index in cluster]
        pile_candidates.append(
            (
                _union_normalized_rectangles(rectangles),
                max(detections[index][1] for index in cluster),
                len(cluster),
            )
        )

    # A long irregular pile can contain two dense groups with a sparse gap.
    # Agglomerate their union rectangles as well, instead of drawing several
    # small item boxes over what is visibly one continuous accumulation.
    changed = True
    while changed:
        changed = False
        for left_index in range(len(pile_candidates)):
            for right_index in range(left_index + 1, len(pile_candidates)):
                left_item = pile_candidates[left_index]
                right_item = pile_candidates[right_index]
                if _rectangle_gap(left_item[0], right_item[0]) > (
                    options.pile_merge_distance
                ):
                    continue
                pile_candidates[left_index] = (
                    _union_normalized_rectangles(
                        [left_item[0], right_item[0]]
                    ),
                    max(left_item[1], right_item[1]),
                    left_item[2] + right_item[2],
                )
                pile_candidates.pop(right_index)
                changed = True
                break
            if changed:
                break

    merged: list[tuple[NormalizedRect, float, str]] = []
    for rectangle, confidence, detection_count in pile_candidates:
        if detection_count < options.minimum_pile_detections:
            continue
        left = rectangle.left
        top = rectangle.top
        right = rectangle.left + rectangle.width
        bottom = rectangle.top + rectangle.height
        padding = options.pile_box_padding
        padded_left = max(left - padding, 0.0)
        padded_top = max(top - padding, 0.0)
        padded_right = min(right + padding, 1.0)
        padded_bottom = min(bottom + padding, 1.0)
        merged.append(
            (
                NormalizedRect(
                    padded_left,
                    padded_top,
                    padded_right - padded_left,
                    padded_bottom - padded_top,
                ),
                confidence,
                "trash pile",
            )
        )
    return sorted(merged, key=lambda item: item[1], reverse=True)


def _union_normalized_rectangles(
    rectangles: list[NormalizedRect],
) -> NormalizedRect:
    left = min(item.left for item in rectangles)
    top = min(item.top for item in rectangles)
    right = max(item.left + item.width for item in rectangles)
    bottom = max(item.top + item.height for item in rectangles)
    return NormalizedRect(left, top, right - left, bottom - top)


def _rectangle_gap(first: NormalizedRect, second: NormalizedRect) -> float:
    first_right = first.left + first.width
    second_right = second.left + second.width
    first_bottom = first.top + first.height
    second_bottom = second.top + second.height
    horizontal = max(
        first.left - second_right,
        second.left - first_right,
        0.0,
    )
    vertical = max(
        first.top - second_bottom,
        second.top - first_bottom,
        0.0,
    )
    return (horizontal * horizontal + vertical * vertical) ** 0.5


def _rectangle_iou(first: NormalizedRect, second: NormalizedRect) -> float:
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(
        first.left + first.width,
        second.left + second.width,
    )
    bottom = min(
        first.top + first.height,
        second.top + second.height,
    )
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    union = (
        first.width * first.height
        + second.width * second.height
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _garbage_chinese_label(value: str) -> str:
    return {
        "plastic bottle": "塑料瓶",
        "garbage bag": "垃圾袋",
        "plastic bag": "塑料袋",
        "cardboard box": "纸箱",
        "paper waste": "废纸",
        "can": "易拉罐",
        "trash pile": "垃圾堆",
        "garbage": "垃圾堆",
        "waste": "垃圾",
    }.get(value.strip().lower(), value if value.strip() else "垃圾")


def _rectangle_contains_point(
    rectangle: NormalizedRect,
    point: tuple[float, float],
) -> bool:
    return (
        rectangle.left <= point[0] <= rectangle.left + rectangle.width
        and rectangle.top <= point[1] <= rectangle.top + rectangle.height
    )


def _garbage_event_overrides_detection(
    event_overlay: GarbageOverlay,
    detected_overlay: GarbageOverlay,
) -> bool:
    event_rectangle = event_overlay.rectangle
    detected_rectangle = detected_overlay.rectangle
    if event_rectangle is None or detected_rectangle is None:
        return False
    return (
        _rectangle_iou(event_rectangle, detected_rectangle) > 0
        or _rectangle_contains_point(
            event_rectangle,
            detected_rectangle.center,
        )
        or _rectangle_contains_point(
            detected_rectangle,
            event_rectangle.center,
        )
    )


class GarbageFrameProcessor:
    """Fuse semantic boxes and background change off the streaming path."""

    def __init__(
        self,
        *,
        policies: dict[int, StreamPolicy],
        event_engines: dict[int, EventEngine],
        labels: list[str],
        evidence_writers: dict[int, EventEvidenceWriter] | None = None,
        metrics_observer: Any | None = None,
        overlay_cache: GarbageOverlayCache | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None:
        self._policies = policies
        self._event_engines = event_engines
        self._evidence_writers = evidence_writers or {}
        self._metrics_observer = metrics_observer
        self._overlay_cache = overlay_cache
        self._labels = labels
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._background = {
            (pad_index, roi.roi_id): BackgroundChangeDetector(roi.polygon)
            for pad_index, policy in policies.items()
            for roi in policy.event_detection.rois
            if policy.event_detection.garbage.enabled
            and policy.event_detection.garbage.background_change_enabled
            and roi.garbage_enabled
        }
        # Semantic detectors occasionally miss a single sampled frame. Keep
        # the previous non-empty result for a short gap so one flicker cannot
        # reset a persistence timer or create a false removal event.
        self._last_semantic: dict[tuple[int, str], GarbageSnapshot] = {}
        self._semantic_misses: dict[tuple[int, str], int] = {}
        self._semantic_hold_frames = 3

    def process(
        self,
        batch_meta: Any,
        frames: list[Any],
    ) -> None:
        timestamp = time.monotonic()
        for frame_index, frame_meta in enumerate(batch_meta.frame_items):
            pad_index = int(frame_meta.pad_index)
            policy = self._policies.get(pad_index)
            engine = self._event_engines.get(pad_index)
            if policy is None or engine is None or frame_index >= len(frames):
                continue
            frame_number = int(getattr(frame_meta, "frame_number", 0))
            started_at = time.perf_counter()
            frame = _frame_to_small_numpy(frames[frame_index])
            semantic, actors = _collect_frame_observations(
                frame_meta,
                policy,
                self._labels,
                timestamp=timestamp,
                frame_width=self._frame_width,
                frame_height=self._frame_height,
            )
            for roi in policy.event_detection.rois:
                if not roi.garbage_enabled:
                    continue
                detector = self._background.get((pad_index, roi.roi_id))
                actor_present = any(
                    _point_in_polygon(
                        actor.rectangle.bottom_center,
                        [list(point) for point in roi.polygon],
                    )
                    for actor in actors
                )
                visual = (
                    detector.observe(frame, actor_present=actor_present)
                    if detector is not None
                    else VisualChange()
                )
                semantic_key = (pad_index, roi.roi_id)
                snapshot = semantic.get(
                    roi.roi_id,
                    GarbageSnapshot(
                        timestamp=timestamp,
                        frame_number=frame_number,
                        area_ratio=0.0,
                    ),
                )
                snapshot = self._stabilize_semantic(semantic_key, snapshot)
                snapshot = GarbageSnapshot(
                    timestamp=snapshot.timestamp,
                    frame_number=snapshot.frame_number,
                    area_ratio=snapshot.area_ratio,
                    regions=snapshot.regions,
                    semantic_confidence=snapshot.semantic_confidence,
                    object_type=snapshot.object_type,
                    visual_change_ratio=visual.area_ratio,
                    visual_regions=visual.regions,
                    detections=snapshot.detections,
                )
                if self._overlay_cache is not None:
                    self._overlay_cache.update(
                        pad_index=pad_index,
                        roi_id=roi.roi_id,
                        snapshot=snapshot,
                    )
                result = engine.observe_garbage(
                    roi_id=roi.roi_id,
                    snapshot=snapshot,
                )
                if result.events:
                    writer = self._evidence_writers.get(pad_index)
                    if writer is not None:
                        writer.attach_snapshots(result.events, frame)
                    if detector is not None:
                        detector.commit(frame)
            if self._metrics_observer is not None:
                self._metrics_observer(
                    policy.stream_id,
                    (time.perf_counter() - started_at) * 1_000,
                )

    def _stabilize_semantic(
        self,
        key: tuple[int, str],
        snapshot: GarbageSnapshot,
    ) -> GarbageSnapshot:
        if snapshot.regions:
            self._last_semantic[key] = snapshot
            self._semantic_misses[key] = 0
            return snapshot
        misses = self._semantic_misses.get(key, 0) + 1
        self._semantic_misses[key] = misses
        previous = self._last_semantic.get(key)
        if previous is None or misses > self._semantic_hold_frames:
            self._last_semantic.pop(key, None)
            return snapshot
        return GarbageSnapshot(
            timestamp=snapshot.timestamp,
            frame_number=snapshot.frame_number,
            area_ratio=previous.area_ratio,
            regions=previous.regions,
            semantic_confidence=previous.semantic_confidence,
            object_type=previous.object_type,
            detections=previous.detections,
        )


class GasCylinderFrameProcessor:
    """Consume only sampled frames; failures never escape to the main chain."""

    def __init__(
        self,
        client: GasCylinderProcessClient,
    ) -> None:
        self._client = client

    def process(self, batch_meta: Any, frames: list[Any]) -> None:
        timestamp = time.monotonic()
        for frame_index, frame_meta in enumerate(batch_meta.frame_items):
            if frame_index >= len(frames):
                continue
            pad_index = int(frame_meta.pad_index)
            try:
                if not self._client.accepts(pad_index):
                    continue
                self._client.submit(
                    pad_index,
                    _frame_to_small_numpy(frames[frame_index]),
                    timestamp=timestamp,
                )
            except Exception:
                LOGGER.exception(
                    "燃气瓶旁路分析失败，主RTSP继续运行: pad=%d",
                    pad_index,
                )


class VesselFrameProcessor:
    """Feed sampled mux frames to the non-blocking vessel process."""

    def __init__(
        self,
        client: VesselDetectionProcessClient,
        *,
        vessel_cache: VesselResultCache | None = None,
        fishing_risk_engines: dict[int, FishingRiskEngine] | None = None,
        fishing_risk_cache: FishingRiskResultCache | None = None,
        evidence_writers: dict[int, EventEvidenceWriter] | None = None,
        ptz_verification_coordinators: dict[
            int, PtzVerificationCoordinator
        ] | None = None,
    ) -> None:
        self._client = client
        self._vessel_cache = vessel_cache
        self._fishing_risk_engines = fishing_risk_engines or {}
        self._fishing_risk_cache = fishing_risk_cache
        self._evidence_writers = evidence_writers or {}
        self._ptz_verification_coordinators = (
            ptz_verification_coordinators or {}
        )
        self._last_risk_version: dict[int, int] = {}
        self._ptz_was_busy: dict[int, bool] = {}

    def process(self, batch_meta: Any, frames: list[Any]) -> None:
        timestamp = time.monotonic()
        for frame_index, frame_meta in enumerate(batch_meta.frame_items):
            if frame_index >= len(frames):
                continue
            pad_index = int(frame_meta.pad_index)
            try:
                coordinator = self._ptz_verification_coordinators.get(
                    pad_index
                )
                ptz_busy = bool(
                    coordinator is not None and coordinator.is_busy
                )
                view_generation = (
                    getattr(coordinator, "view_generation", 0)
                    if coordinator is not None
                    else 0
                )
                accepted = self._client.accepts(
                    pad_index,
                    timestamp=timestamp,
                )
                converted_frame: Any | None = None
                engine = self._fishing_risk_engines.get(pad_index)
                if engine is not None and self._vessel_cache is not None:
                    snapshot = self._vessel_cache.snapshot(pad_index)
                    was_busy = self._ptz_was_busy.get(pad_index, False)
                    self._ptz_was_busy[pad_index] = ptz_busy
                    if ptz_busy:
                        if not was_busy:
                            engine.reset_tracking()
                            previous = (
                                self._fishing_risk_cache.snapshot(pad_index)
                                if self._fishing_risk_cache is not None
                                else FishingRiskSnapshot()
                            )
                            if self._fishing_risk_cache is not None:
                                self._fishing_risk_cache.store_snapshot(
                                    pad_index,
                                    FishingRiskSnapshot(
                                        state="paused",
                                        result_version=(
                                            previous.result_version + 1
                                        ),
                                        updated_at=timestamp,
                                        total_events=previous.total_events,
                                        message=(
                                            "PTZ近景复核中，行为分析暂停"
                                        ),
                                    ),
                                )
                        # Consume zoom-view versions without feeding them into
                        # the fixed-camera fishing-risk trajectory engine.
                        self._last_risk_version[pad_index] = (
                            snapshot.result_version
                        )
                    else:
                        previous_version = self._last_risk_version.get(
                            pad_index,
                            -1,
                        )
                        if (
                            snapshot.state == "running"
                            and snapshot.result_version != previous_version
                        ):
                            result = engine.observe(
                                timestamp=(snapshot.updated_at or timestamp),
                                observed_at=datetime.now(timezone.utc),
                                detections=tuple(
                                    item
                                    for item in snapshot.detections
                                    if item.class_id
                                    != SMALL_TARGET_PROPOSAL_CLASS_ID
                                ),
                            )
                            self._last_risk_version[pad_index] = (
                                snapshot.result_version
                            )
                            if self._fishing_risk_cache is not None:
                                self._fishing_risk_cache.store_snapshot(
                                    pad_index,
                                    result.snapshot,
                                )
                            if result.events:
                                writer = self._evidence_writers.get(pad_index)
                                if writer is not None:
                                    converted_frame = _frame_to_small_numpy(
                                        frames[frame_index]
                                    )
                                    writer.attach_snapshots(
                                        result.events,
                                        converted_frame,
                                    )
                if accepted:
                    if converted_frame is None:
                        converted_frame = _frame_to_small_numpy(
                            frames[frame_index]
                        )
                    self._client.submit(
                        pad_index,
                        converted_frame,
                        timestamp=timestamp,
                        verification_active=ptz_busy,
                        view_generation=view_generation,
                    )
            except Exception:
                LOGGER.exception(
                    "船舶旁路分析失败，主RTSP继续运行: pad=%d",
                    pad_index,
                )


def _ground_litter_occluder_classes(
    options: GroundLitterDetectionOptions,
) -> tuple[int, ...]:
    """Primary-metadata classes that can hide or mimic a litter target."""
    return tuple(dict.fromkeys((
        *options.actor_class_ids,
        *options.context_class_ids,
    )))


class GroundLitterFrameProcessor:
    """Feed sampled native-resolution frames to the litter side process.

    Actor (person/vehicle) boxes come from the primary detector metadata that
    is already tracked on the main chain, so occlusion filtering costs no extra
    inference.  ``actor_model`` remains available for streams whose primary
    model filters vehicles out of the metadata.
    """

    MAXIMUM_ACTOR_BOXES = 64

    def __init__(
        self,
        client: GroundLitterProcessClient,
        *,
        night_by_pad: dict[int, bool] | None = None,
        actor_classes_by_pad: dict[int, tuple[int, ...]] | None = None,
    ) -> None:
        self._client = client
        self._night_by_pad = dict(night_by_pad or {})
        self._actor_classes_by_pad = dict(actor_classes_by_pad or {})

    def process(
        self,
        batch_meta: Any,
        frames: list[Any],
        *,
        frame_width: int,
        frame_height: int,
    ) -> None:
        timestamp = time.monotonic()
        width = max(int(frame_width), 1)
        height = max(int(frame_height), 1)
        for frame_index, frame_meta in enumerate(batch_meta.frame_items):
            if frame_index >= len(frames):
                continue
            pad_index = int(frame_meta.pad_index)
            try:
                if not self._client.accepts(pad_index, timestamp=timestamp):
                    continue
                actors = self._actor_boxes(
                    frame_meta,
                    pad_index=pad_index,
                    width=width,
                    height=height,
                )
                converted = _frame_to_small_numpy(frames[frame_index])
                self._client.submit(
                    pad_index,
                    converted,
                    timestamp=timestamp,
                    night=self._night_by_pad.get(pad_index, False),
                    actors=actors,
                )
            except Exception:
                LOGGER.exception(
                    "零散垃圾旁路分析失败，主RTSP继续运行: pad=%d",
                    pad_index,
                )

    def _actor_boxes(
        self,
        frame_meta: Any,
        *,
        pad_index: int,
        width: int,
        height: int,
    ) -> list[tuple[float, float, float, float]]:
        wanted = self._actor_classes_by_pad.get(pad_index)
        if not wanted:
            return []
        allowed = set(int(item) for item in wanted)
        boxes: list[tuple[float, float, float, float]] = []
        for object_meta in getattr(frame_meta, "object_items", ()) or ():
            if int(getattr(object_meta, "class_id", -1)) not in allowed:
                continue
            rectangle = object_meta.rect_params
            left = float(rectangle.left) / width
            top = float(rectangle.top) / height
            box_width = float(rectangle.width) / width
            box_height = float(rectangle.height) / height
            if box_width <= 0 or box_height <= 0:
                continue
            boxes.append(
                (
                    min(max(left, 0.0), 1.0),
                    min(max(top, 0.0), 1.0),
                    min(max(box_width, 0.0), 1.0),
                    min(max(box_height, 0.0), 1.0),
                )
            )
            if len(boxes) >= self.MAXIMUM_ACTOR_BOXES:
                break
        return boxes


def _frame_to_small_numpy(value: Any) -> Any:
    """Copy a ServiceMaker analysis surface into a NumPy-compatible frame."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return value
    try:
        # CPU-compatible tensors can go directly through NumPy.
        return np.from_dlpack(value.clone())
    except (AttributeError, BufferError, RuntimeError, TypeError):
        # DeepStream 8's Triton image already includes CuPy. It consumes CUDA
        # DLPack directly and performs the explicit device-to-host copy,
        # avoiding a large PyTorch dependency for a 640x360/3 FPS side path.
        import cupy

        return cupy.asnumpy(cupy.from_dlpack(value.clone()))


@dataclass(slots=True)
class _PlateTrack:
    rectangle: tuple[float, float, float, float]
    last_frame: int


class PlateIdentityTracker:
    """Assign stable IDs to LPD objects without a second nvtracker."""

    def __init__(self, max_age_frames: int = 10) -> None:
        self._max_age_frames = max_age_frames
        self._next_id = 1
        self._tracks: dict[tuple[int, int], _PlateTrack] = {}

    def process(self, batch_meta: Any) -> dict[tuple[int, int], int]:
        components: dict[tuple[int, int], int] = {}
        for frame_meta in batch_meta.frame_items:
            pad_index = int(frame_meta.pad_index)
            frame_number = int(getattr(frame_meta, "frame_number", 0))
            used_track_ids: set[int] = set()
            for object_meta in frame_meta.object_items:
                component_id = int(
                    getattr(object_meta, "unique_component_id", -1)
                )
                class_id = int(getattr(object_meta, "class_id", -1))
                key = (component_id, class_id)
                components[key] = components.get(key, 0) + 1
                if component_id != LICENSE_PLATE_DETECTOR_UID:
                    continue
                rectangle = self._rectangle(object_meta)
                track_id = self._match(
                    pad_index,
                    frame_number,
                    rectangle,
                    used_track_ids,
                )
                if track_id is None:
                    track_id = self._next_id
                    self._next_id += 1
                used_track_ids.add(track_id)
                self._tracks[(pad_index, track_id)] = _PlateTrack(
                    rectangle=rectangle,
                    last_frame=frame_number,
                )
                object_meta.object_id = track_id
            self._expire(pad_index, frame_number)
        return components

    def _match(
        self,
        pad_index: int,
        frame_number: int,
        rectangle: tuple[float, float, float, float],
        used_track_ids: set[int],
    ) -> int | None:
        best_id: int | None = None
        best_score = float("-inf")
        for (track_pad, track_id), state in self._tracks.items():
            if (
                track_pad != pad_index
                or track_id in used_track_ids
                or frame_number - state.last_frame > self._max_age_frames
            ):
                continue
            overlap = self._intersection_over_union(
                rectangle,
                state.rectangle,
            )
            distance = self._normalized_center_distance(
                rectangle,
                state.rectangle,
            )
            if overlap < 0.05 and distance > 1.25:
                continue
            score = overlap * 2.0 - distance
            if score > best_score:
                best_id = track_id
                best_score = score
        return best_id

    def _expire(self, pad_index: int, frame_number: int) -> None:
        expired = [
            key
            for key, state in self._tracks.items()
            if (
                key[0] == pad_index
                and frame_number - state.last_frame > self._max_age_frames
            )
        ]
        for key in expired:
            self._tracks.pop(key, None)

    @staticmethod
    def _rectangle(
        object_meta: Any,
    ) -> tuple[float, float, float, float]:
        rectangle = object_meta.rect_params
        return (
            float(rectangle.left),
            float(rectangle.top),
            max(float(rectangle.width), 1.0),
            max(float(rectangle.height), 1.0),
        )

    @staticmethod
    def _intersection_over_union(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> float:
        left_x, left_y, left_w, left_h = left
        right_x, right_y, right_w, right_h = right
        intersection_w = max(
            min(left_x + left_w, right_x + right_w)
            - max(left_x, right_x),
            0.0,
        )
        intersection_h = max(
            min(left_y + left_h, right_y + right_h)
            - max(left_y, right_y),
            0.0,
        )
        intersection = intersection_w * intersection_h
        union = left_w * left_h + right_w * right_h - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _normalized_center_distance(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> float:
        left_x, left_y, left_w, left_h = left
        right_x, right_y, right_w, right_h = right
        dx = (
            left_x + left_w / 2 - right_x - right_w / 2
        ) / max(left_w, right_w, 1.0)
        dy = (
            left_y + left_h / 2 - right_y - right_h / 2
        ) / max(left_h, right_h, 1.0)
        return (dx * dx + dy * dy) ** 0.5


def _load_policies(config: dict[str, Any]) -> dict[int, StreamPolicy]:
    english_labels = [
        line.strip()
        for line in Path(config["labels_path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    mapping = dict(config.get("label_map") or {})
    translated = {
        class_id: chinese_label(class_id, name, mapping)
        for class_id, name in enumerate(english_labels)
    }
    return {
        pad_index: StreamPolicy(
            stream_id=stream["stream_id"],
            classes=(
                frozenset(int(item) for item in stream["classes"])
                if stream.get("classes")
                else None
            ),
            conf=float(stream["conf"]),
            roi=stream.get("roi"),
            labels=translated,
            display_detections=bool(
                stream.get("display_detections", True)
            ),
            license_plate_enabled=bool(
                stream.get("license_plate", {}).get("enabled", False)
            ),
            minimum_plate_confirmations=int(
                stream.get("license_plate", {}).get(
                    "minimum_confirmations",
                    2,
                )
            ),
            event_detection=EventDetectionOptions.from_payload(
                stream.get("event_detection")
            ),
            gas_cylinder=GasCylinderOptions.from_payload(
                stream.get("gas_cylinder")
            ),
            vessel_detection=VesselDetectionOptions.from_payload(
                stream.get("vessel_detection")
            ),
            ground_litter=GroundLitterDetectionOptions.from_payload(
                stream.get("ground_litter")
            ),
            fishing_risk=FishingRiskOptions.from_payload(
                stream.get("fishing_risk")
            ),
            ptz_verification=PtzVerificationOptions.from_payload(
                stream.get("ptz_verification")
            ),
        )
        for pad_index, stream in enumerate(config["streams"])
    }


def _add_pipeline_nodes(
    pipeline: Any,
    config: dict[str, Any],
    inference_config_path: Path,
    latency_probe: Any,
    overlay_probe_factory: Any,
    plate_metadata_probe: Any,
    pre_encode_probe_factory: Any,
    publish_probe_factory: Any,
    lpd_config_path: Path | None = None,
    lpr_config_path: Path | None = None,
    garbage_config_path: Path | None = None,
    garbage_receiver: Any | None = None,
    garbage_skip_probe: Any | None = None,
    gas_cylinder_receiver: Any | None = None,
    gas_cylinder_skip_probe: Any | None = None,
    vessel_receiver: Any | None = None,
    vessel_skip_probe: Any | None = None,
    ground_litter_receiver: Any | None = None,
    ground_litter_skip_probe: Any | None = None,
) -> None:
    gpu_id = int(config["gpu_id"])
    pipeline.add(
        "nvstreammux",
        "mux",
        {
            "batch-size": int(config["batch_size"]),
            "batched-push-timeout": int(
                config["batch_push_timeout_us"]
            ),
            "width": int(config["mux_width"]),
            "height": int(config["mux_height"]),
            "live-source": True,
            "sync-inputs": False,
            "buffer-pool-size": 8,
        },
    )
    for index, stream in enumerate(config["streams"]):
        source = f"source_{index}"
        pipeline.add(
            "nvurisrcbin",
            source,
            {
                "uri": stream["input_url"],
                "gpu-id": gpu_id,
                "latency": int(config["source_latency_ms"]),
                # Keep enough decoder surfaces available for inference,
                # tracking and optional lossy analysis branches.  The
                # nvurisrcbin default is only one extra surface; under branch
                # pressure that can stall the decoder while it is still
                # holding reference pictures for a long-GOP camera stream.
                "num-extra-surfaces": 8,
                # Never discard an H.264 reference picture merely because
                # the RTSP jitter buffer temporarily exceeds its target.
                # A late frame is recoverable; a missing reference frame can
                # leave macroblock trails until the camera sends another IDR.
                "drop-on-latency": False,
                "rtsp-reconnect-interval": 5,
                "rtsp-reconnect-attempts": -1,
                "select-rtp-protocol": 4,
            },
        )
        pipeline.link((source, "mux"), ("vsrc_%u", ""))

    pipeline.add(
        "queue",
        "pre_infer",
        {
            # Public RTSP sources can pause briefly and then deliver a burst
            # of correctly timestamped frames.  Dropping that burst here
            # turns recoverable network jitter into permanently missing
            # output frames (visible as rhythmic stutter).  The GPU is fast
            # enough to drain this bounded queue after the source recovers;
            # keeping the frames trades a short, bounded latency increase for
            # continuous video and tracker/overlay alignment.
            "max-size-buffers": 75,
            "max-size-bytes": 0,
            "max-size-time": 0,
            "leaky": 0,
        },
    )
    pipeline.add(
        "nvinfer",
        "primary_infer",
        {
            "config-file-path": str(inference_config_path),
            "batch-size": int(config["batch_size"]),
        },
    )
    license_plate_enabled = bool(
        config.get("license_plate", {}).get("enabled", False)
    )
    tracker_properties = {
        "ll-config-file": config["tracker_config"],
        "ll-lib-file": config["tracker_library"],
        "gpu-id": gpu_id,
        "display-tracking-id": False,
    }
    if license_plate_enabled:
        if lpd_config_path is None or lpr_config_path is None:
            raise RuntimeError("车牌识别已启用但缺少推理配置")
        # Give the vehicle crops stable IDs before LPDNet. This makes
        # secondary-reinfer-interval effective for the expensive detector.
        pipeline.add(
            "nvtracker",
            "vehicle_tracker",
            tracker_properties,
        )
        pipeline.add(
            "nvinfer",
            "license_plate_detector",
            {
                "config-file-path": str(lpd_config_path),
                "batch-size": int(
                    config["license_plate"]["detector_batch_size"]
                ),
            },
        )
    if not license_plate_enabled:
        pipeline.add(
            "nvtracker",
            "tracker",
            tracker_properties,
        )
    if license_plate_enabled:
        pipeline.add(
            "nvinfer",
            "license_plate_recognizer",
            {
                "config-file-path": str(lpr_config_path),
                "batch-size": int(
                    config["license_plate"]["recognizer_batch_size"]
                ),
            },
        )
    pipeline.add("nvvideoconvert", "osd_convert", {"gpu-id": gpu_id})
    pipeline.add(
        "capsfilter",
        "rgba_caps",
        {"caps": "video/x-raw(memory:NVMM), format=RGBA"},
    )
    pipeline.add("nvstreamdemux", "demux")
    garbage_enabled = bool(
        config.get("garbage", {}).get("enabled", False)
    )
    gas_cylinder_enabled = bool(
        config.get("gas_cylinder", {}).get("enabled", False)
    )
    vessel_enabled = bool(
        config.get("vessel_detection", {}).get("enabled", False)
    )
    ground_litter_enabled = bool(
        config.get("ground_litter", {}).get("enabled", False)
    )
    analytics_enabled = (
        garbage_enabled
        or gas_cylinder_enabled
        or vessel_enabled
        or ground_litter_enabled
    )
    if analytics_enabled:
        pipeline.add("tee", "analytics_tee")
        pipeline.add(
            "queue",
            "analytics_main_queue",
            {
                "max-size-buffers": 2,
                "max-size-bytes": 0,
                "max-size-time": 0,
                "leaky": 0,
            },
        )
    if garbage_enabled:
        if (
            garbage_config_path is None
            or garbage_receiver is None
            or garbage_skip_probe is None
        ):
            raise RuntimeError("垃圾识别已启用但缺少推理配置或旁路组件")
        pipeline.add(
            "queue",
            "garbage_queue",
            {
                "max-size-buffers": 1,
                "max-size-bytes": 0,
                "max-size-time": 0,
                "leaky": 2,
            },
        )
        pipeline.add(
            "nvinfer",
            "garbage_infer",
            {
                "config-file-path": str(garbage_config_path),
                "batch-size": int(config["batch_size"]),
            },
        )
        pipeline.add(
            "nvvideoconvert",
            "garbage_convert",
            {"gpu-id": gpu_id},
        )
        pipeline.add(
            "capsfilter",
            "garbage_rgba_caps",
            {
                # ServiceMaker Buffer.extract() requires NvBufSurface. Keep
                # the scaled thumbnail in NVMM and copy it through DLPack
                # only inside the lossy receiver thread.
                "caps": (
                    "video/x-raw(memory:NVMM), format=RGB, "
                    "width=640, height=360"
                )
            },
        )
        pipeline.add(
            "appsink",
            "garbage_sink",
            {
                "sync": False,
                "async": False,
                "max-buffers": 1,
                "drop": True,
                "emit-signals": True,
            },
        )
    if gas_cylinder_enabled:
        if gas_cylinder_receiver is None or gas_cylinder_skip_probe is None:
            raise RuntimeError("燃气瓶识别已启用但缺少旁路组件")
        gas_config = config["gas_cylinder"]
        pipeline.add(
            "queue",
            "gas_cylinder_queue",
            {
                # Never let YOLOE back-pressure decode, OSD or NVENC. The
                # newest sampled frame is more useful than queued old frames.
                "max-size-buffers": 1,
                "max-size-bytes": 0,
                "max-size-time": 0,
                "leaky": 2,
            },
        )
        pipeline.add(
            "nvvideoconvert",
            "gas_cylinder_convert",
            {"gpu-id": gpu_id},
        )
        pipeline.add(
            "capsfilter",
            "gas_cylinder_rgb_caps",
            {
                "caps": (
                    "video/x-raw(memory:NVMM), format=RGB, "
                    f"width={int(gas_config['input_width'])}, "
                    f"height={int(gas_config['input_height'])}"
                )
            },
        )
        pipeline.add(
            "appsink",
            "gas_cylinder_sink",
            {
                "sync": False,
                "async": False,
                "max-buffers": 1,
                "drop": True,
                "emit-signals": True,
            },
        )
    if ground_litter_enabled:
        if (
            ground_litter_receiver is None
            or ground_litter_skip_probe is None
        ):
            raise RuntimeError("零散垃圾识别已启用但缺少旁路组件")
        pipeline.add(
            "queue",
            "ground_litter_queue",
            {
                # Same lossy contract as the other analysis branches: a slow
                # 640-tiled litter pass must never stall decode or NVENC.
                "max-size-buffers": 1,
                "max-size-bytes": 0,
                "max-size-time": 0,
                "leaky": 2,
            },
        )
        pipeline.add(
            "nvvideoconvert",
            "ground_litter_convert",
            {"gpu-id": gpu_id},
        )
        pipeline.add(
            "capsfilter",
            "ground_litter_rgb_caps",
            {
                # Deliberately keep the mux/native resolution: native-pixel
                # 640 tiles are the whole point of this detector, so this
                # branch must not be downscaled like the vessel branch.
                "caps": "video/x-raw(memory:NVMM), format=RGB",
            },
        )
        pipeline.add(
            "appsink",
            "ground_litter_sink",
            {
                "sync": False,
                "async": False,
                "max-buffers": 1,
                "drop": True,
                "emit-signals": True,
            },
        )
    if vessel_enabled:
        if vessel_receiver is None or vessel_skip_probe is None:
            raise RuntimeError("船舶识别已启用但缺少旁路组件")
        vessel_config = config["vessel_detection"]
        pipeline.add(
            "queue",
            "vessel_queue",
            {
                "max-size-buffers": 1,
                "max-size-bytes": 0,
                "max-size-time": 0,
                "leaky": 2,
            },
        )
        pipeline.add(
            "nvvideoconvert",
            "vessel_convert",
            {"gpu-id": gpu_id},
        )
        pipeline.add(
            "capsfilter",
            "vessel_rgb_caps",
            {
                "caps": (
                    "video/x-raw(memory:NVMM), format=RGB, "
                    f"width={int(vessel_config['input_width'])}, "
                    f"height={int(vessel_config['input_height'])}"
                )
            },
        )
        pipeline.add(
            "appsink",
            "vessel_sink",
            {
                "sync": False,
                "async": False,
                "max-buffers": 1,
                "drop": True,
                "emit-signals": True,
            },
        )
    analytics_nodes = ["mux", "pre_infer", "primary_infer"]
    if license_plate_enabled:
        analytics_nodes.extend(
            ("vehicle_tracker", "license_plate_detector")
        )
    if not license_plate_enabled:
        analytics_nodes.append("tracker")
    if license_plate_enabled:
        analytics_nodes.append("license_plate_recognizer")
    if analytics_enabled:
        analytics_nodes.append("analytics_tee")
        pipeline.link(*analytics_nodes)
        pipeline.link(
            ("analytics_tee", "analytics_main_queue"),
            ("src_%u", ""),
        )
        pipeline.link(
            "analytics_main_queue",
            "osd_convert",
            "rgba_caps",
            "demux",
        )
    if garbage_enabled:
        pipeline.link(("analytics_tee", "garbage_queue"), ("src_%u", ""))
        pipeline.link(
            "garbage_queue",
            "garbage_infer",
            "garbage_convert",
            "garbage_rgba_caps",
            "garbage_sink",
        )
        pipeline.attach("garbage_queue", garbage_skip_probe)
        pipeline.attach(
            "garbage_sink",
            garbage_receiver,
            tips="new-sample",
        )
    if gas_cylinder_enabled:
        pipeline.link(
            ("analytics_tee", "gas_cylinder_queue"),
            ("src_%u", ""),
        )
        pipeline.link(
            "gas_cylinder_queue",
            "gas_cylinder_convert",
            "gas_cylinder_rgb_caps",
            "gas_cylinder_sink",
        )
        pipeline.attach("gas_cylinder_queue", gas_cylinder_skip_probe)
        pipeline.attach(
            "gas_cylinder_sink",
            gas_cylinder_receiver,
            tips="new-sample",
        )
    if vessel_enabled:
        pipeline.link(("analytics_tee", "vessel_queue"), ("src_%u", ""))
        pipeline.link(
            "vessel_queue",
            "vessel_convert",
            "vessel_rgb_caps",
            "vessel_sink",
        )
        pipeline.attach("vessel_queue", vessel_skip_probe)
        pipeline.attach(
            "vessel_sink",
            vessel_receiver,
            tips="new-sample",
        )
    if ground_litter_enabled:
        pipeline.link(
            ("analytics_tee", "ground_litter_queue"),
            ("src_%u", ""),
        )
        pipeline.link(
            "ground_litter_queue",
            "ground_litter_convert",
            "ground_litter_rgb_caps",
            "ground_litter_sink",
        )
        pipeline.attach("ground_litter_queue", ground_litter_skip_probe)
        pipeline.attach(
            "ground_litter_sink",
            ground_litter_receiver,
            tips="new-sample",
        )
    if not analytics_enabled:
        analytics_nodes.extend(("osd_convert", "rgba_caps", "demux"))
        pipeline.link(*analytics_nodes)
    if license_plate_enabled:
        pipeline.attach("license_plate_detector", plate_metadata_probe)
    pipeline.attach("pre_infer", latency_probe)

    for index, stream in enumerate(config["streams"]):
        queue = f"publish_queue_{index}"
        overlay_anchor = f"overlay_anchor_{index}"
        osd_element = f"osd_{index}"
        convert = f"publish_convert_{index}"
        caps = f"publish_caps_{index}"
        encoder = f"encoder_{index}"
        parser = f"parser_{index}"
        parser_caps = f"parser_caps_{index}"
        clock_sync = f"publish_clock_{index}"
        sink = f"rtsp_sink_{index}"
        pipeline.add(
            "identity",
            overlay_anchor,
            {"silent": True},
        )
        pipeline.add(
            "queue",
            queue,
            {
                # Do not mark NVENC input discontinuous during brief downstream
                # stalls. Low-latency frame dropping remains upstream of
                # inference, while this small queue preserves encoder
                # reference-frame continuity.
                "max-size-buffers": 25,
                "max-size-bytes": 0,
                "max-size-time": 0,
                "leaky": 0,
            },
        )
        pipeline.add(
            "nvdsosd",
            osd_element,
            {"gpu-id": gpu_id, "process-mode": 1},
        )
        pipeline.add("nvvideoconvert", convert, {"gpu-id": gpu_id})
        pipeline.add(
            "capsfilter",
            caps,
            {"caps": "video/x-raw(memory:NVMM), format=I420"},
        )
        pipeline.add(
            "nvv4l2h264enc",
            encoder,
            {
                "bitrate": int(stream["bitrate_bps"]),
                "iframeinterval": int(
                    config["encoder_iframe_interval"]
                ),
                "idrinterval": int(
                    config["encoder_iframe_interval"]
                ),
                "num-B-Frames": 0,
                # DeepStream 8 otherwise defaults to Baseline + P1 +
                # low-latency tuning.  That combination prioritises encoder
                # throughput and produces conspicuous blocks around moving
                # objects even when the requested bitrate is high.  P4 High
                # Profile with spatial/temporal AQ is a balanced real-time
                # quality preset for the deployment's NVENC hardware.
                "profile": 4,
                "preset-id": 4,
                "tuning-info-id": 1,
                "aq": 8,
                "temporalaq": True,
                "num-Ref-Frames": 2,
                "insert-aud": True,
                "insert-sps-pps": True,
                "insert-vui": True,
            },
        )
        pipeline.add(
            "h264parse",
            parser,
            {
                "config-interval": -1,
                "disable-passthrough": True,
            },
        )
        pipeline.add(
            "capsfilter",
            parser_caps,
            {
                "caps": (
                    "video/x-h264, "
                    "stream-format=byte-stream, alignment=au"
                )
            },
        )
        pipeline.add(
            "clocksync",
            clock_sync,
            {
                # Do not clock-throttle encoded buffers here. Public RTSP
                # sources sometimes repeat or jump timestamps; synchronising
                # after the encoder back-pressures the entire inference path
                # and permanently loses otherwise valid catch-up frames.
                # RTP timestamps remain intact, so players can pace display
                # using their own jitter buffer without server-side loss.
                "sync": False,
            },
        )
        pipeline.add(
            "rtspclientsink",
            sink,
            {
                "location": stream["output_url"],
                "protocols": 4,
                "latency": 0,
                "rtx-time": 0,
            },
        )
        # Service Maker expects the request-pad template here.  Passing the
        # concrete pad name (for example ``src_0``) makes its caps lookup fail
        # before the pipeline starts. Repeated requests are allocated as
        # src_0, src_1, ... in the same order as the streams above.
        pipeline.link(("demux", overlay_anchor), ("src_%u", ""))
        pipeline.link(
            overlay_anchor,
            osd_element,
            queue,
            convert,
            caps,
            encoder,
            parser,
            parser_caps,
            clock_sync,
        )
        # rtspclientsink accepts encoded H.264 and creates its RTP payloader
        # internally. Feeding an already-payloaded application/x-rtp stream is
        # incompatible, and its sink is an on-request pad.
        pipeline.link((clock_sync, sink), ("", "sink_%u"))
        pipeline.attach(
            overlay_anchor,
            overlay_probe_factory(index),
        )
        pipeline.attach(
            queue,
            pre_encode_probe_factory(index),
        )
        pipeline.attach(
            clock_sync,
            publish_probe_factory(index),
        )


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise RuntimeError("不支持的DeepStream worker配置版本")
    streams = config.get("streams")
    if not isinstance(streams, list) or not 1 <= len(streams) <= 2:
        raise RuntimeError("DeepStream每组必须包含1到2路流")
    night_vision_enabled = bool(
        config.get("night_vision", {}).get("enabled", False)
    )
    if any(
        bool(item.get("night_vision", {}).get("enabled", False))
        != night_vision_enabled
        for item in streams
    ):
        raise RuntimeError("DeepStream同一组不能混合白天和夜间流")

    from pyservicemaker import (
        BatchMetadataOperator,
        BufferRetriever,
        BufferOperator,
        Pipeline,
        Probe,
        Receiver,
        osd,
    )

    inference_config_path = config_path.with_name("nvinfer.txt")
    build_inference_config(config, inference_config_path)
    tracker_config_path = config_path.with_name("tracker.yml")
    build_tracker_config(
        Path(config["tracker_config"]),
        tracker_config_path,
        max_shadow_tracking_age=int(
            config.get("tracker_max_shadow_tracking_age", 15)
        ),
    )
    config["tracker_config"] = str(tracker_config_path)
    lpd_config_path: Path | None = None
    lpr_config_path: Path | None = None
    if config.get("license_plate", {}).get("enabled", False):
        lpd_config_path = config_path.with_name("lpd_nvinfer.txt")
        lpr_config_path = config_path.with_name("lpr_nvinfer.txt")
        build_lpd_config(config, lpd_config_path)
        build_lpr_config(config, lpr_config_path)
        os.environ["LPR_DICT_PATH"] = config["license_plate"][
            "dictionary_path"
        ]
    garbage_config_path: Path | None = None
    garbage_labels: list[str] = []
    if config.get("garbage", {}).get("enabled", False):
        garbage_config_path = config_path.with_name("garbage_nvinfer.txt")
        build_garbage_config(config, garbage_config_path)
        garbage_labels = [
            item.strip()
            for item in Path(config["garbage"]["labels_path"])
            .read_text(encoding="utf-8")
            .splitlines()
            if item.strip()
        ]
    policies = _load_policies(config)
    gas_cylinder_cache = GasCylinderResultCache()
    gas_cylinder_client: GasCylinderProcessClient | None = None
    gas_config = config.get("gas_cylinder", {})
    if bool(gas_config.get("enabled", False)):
        try:
            GasCylinderCameraProfile.load(
                Path(gas_config["profile_root"]),
                str(gas_config["profile_id"]),
            )
            gas_options_by_pad = {
                pad_index: policy.gas_cylinder
                for pad_index, policy in policies.items()
                if policy.gas_cylinder.enabled
            }
            gas_cylinder_client = GasCylinderProcessClient(
                GasCylinderProcessConfig(
                    model_path=Path(gas_config["model_path"]),
                    profile_root=Path(gas_config["profile_root"]),
                    profile_id=str(gas_config["profile_id"]),
                    device=f"cuda:{int(config['gpu_id'])}",
                    imgsz=int(gas_config["imgsz"]),
                    options_by_pad=gas_options_by_pad,
                ),
                gas_cylinder_cache,
            )
        except Exception as exc:
            LOGGER.exception(
                "燃气瓶组件初始化失败，主RTSP将不受影响"
            )
            for pad_index, policy in policies.items():
                if policy.gas_cylinder.enabled:
                    gas_cylinder_cache.mark_state(
                        pad_index,
                        "error",
                        f"燃气瓶组件初始化失败: {type(exc).__name__}",
                    )
    vessel_detection_cache = VesselResultCache()
    vessel_detection_client: VesselDetectionProcessClient | None = None
    vessel_config = config.get("vessel_detection", {})
    if bool(vessel_config.get("enabled", False)):
        try:
            vessel_options_by_pad = {
                pad_index: policy.vessel_detection
                for pad_index, policy in policies.items()
                if policy.vessel_detection.enabled
            }
            vessel_detection_client = VesselDetectionProcessClient(
                VesselDetectionProcessConfig(
                    model_path=Path(vessel_config["model_path"]),
                    device=f"cuda:{int(config['gpu_id'])}",
                    half=True,
                    options_by_pad=vessel_options_by_pad,
                ),
                vessel_detection_cache,
            )
        except Exception as exc:
            LOGGER.exception("船舶组件初始化失败，主RTSP将不受影响")
            for pad_index, policy in policies.items():
                if policy.vessel_detection.enabled:
                    vessel_detection_cache.mark_state(
                        pad_index,
                        "error",
                        f"船舶组件初始化失败: {type(exc).__name__}",
                    )
            # Keep the OSD error state, but do not build a branch without a
            # live receiver. The primary infer/encode path still starts.
            vessel_config["enabled"] = False
    ground_litter_cache = GroundLitterResultCache()
    ground_litter_client: GroundLitterProcessClient | None = None
    ground_litter_config = config.get("ground_litter", {})
    if bool(ground_litter_config.get("enabled", False)):
        try:
            ground_litter_options_by_pad = {
                pad_index: policy.ground_litter
                for pad_index, policy in policies.items()
                if policy.ground_litter.enabled
            }
            ground_litter_client = GroundLitterProcessClient(
                GroundLitterProcessConfig(
                    model_path=Path(ground_litter_config["model_path"]),
                    device=f"cuda:{int(config['gpu_id'])}",
                    half=True,
                    actor_model_path=(
                        Path(ground_litter_config["actor_model_path"])
                        if ground_litter_config.get("actor_model_path")
                        else None
                    ),
                    options_by_pad=ground_litter_options_by_pad,
                    profile_root=(
                        Path(ground_litter_config["profile_root"])
                        if ground_litter_config.get("profile_root")
                        else None
                    ),
                ),
                ground_litter_cache,
            )
        except Exception as exc:
            LOGGER.exception("零散垃圾组件初始化失败，主RTSP将不受影响")
            for pad_index, policy in policies.items():
                if policy.ground_litter.enabled:
                    ground_litter_cache.mark_state(
                        pad_index,
                        "error",
                        f"零散垃圾组件初始化失败: {type(exc).__name__}",
                    )
            # Keep the OSD error state, but never build the branch without a
            # live receiver. The primary infer/encode path still starts.
            ground_litter_config["enabled"] = False
    ptz_verification_coordinators: dict[
        str, PtzVerificationCoordinator
    ] = {}
    ptz_verification_by_pad: dict[int, PtzVerificationCoordinator] = {}
    ptz_verification_errors: dict[str, str] = {}
    for pad_index, stream in enumerate(streams):
        options = policies[pad_index].ptz_verification
        if not options.enabled:
            continue
        stream_id = str(stream["stream_id"])
        if vessel_detection_client is None:
            ptz_verification_errors[stream_id] = "船舶检测组件不可用"
            continue
        try:
            repository = PtzVerificationRepository(
                Path(stream["event_root"]) / "vessel-verifications"
            )
            coordinator = PtzVerificationCoordinator(
                stream_id=stream_id,
                options=options,
                snapshot_provider=(
                    lambda pad=pad_index: vessel_detection_cache.snapshot(pad)
                ),
                repository=repository,
                camera_client=CameraControlClient(options),
                evidence_validator=(
                    lambda content, timeout, pad=pad_index: (
                        vessel_detection_client.validate_evidence(
                            pad,
                            content,
                            timeout=timeout,
                        )
                    )
                ),
            )
            ptz_verification_coordinators[stream_id] = coordinator
            ptz_verification_by_pad[pad_index] = coordinator
            coordinator.start()
        except Exception as exc:
            LOGGER.exception(
                "PTZ近景复核初始化失败，主RTSP将不受影响: stream=%s",
                stream_id,
            )
            ptz_verification_errors[stream_id] = (
                f"{type(exc).__name__}: {exc}"
            )
    ptz_control_monitor: PtzControlCommandMonitor | None = None
    if ptz_verification_coordinators:
        ptz_control_monitor = PtzControlCommandMonitor(
            Path(config["control_dir"]),
            ptz_verification_coordinators,
        )
        ptz_control_monitor.start()
    fishing_risk_cache = FishingRiskResultCache()
    event_engines: dict[int, EventEngine] = {}
    fishing_risk_engines: dict[int, FishingRiskEngine] = {}
    evidence_writers: dict[int, EventEvidenceWriter] = {}
    webhook_dispatchers: list[WebhookDispatcher] = []
    for pad_index, stream in enumerate(streams):
        event_options = policies[pad_index].event_detection
        risk_options = policies[pad_index].fishing_risk
        if not event_options.enabled and not risk_options.enabled:
            continue
        repository = EventRepository(Path(stream["event_root"]))
        evidence_writers[pad_index] = EventEvidenceWriter(repository)
        if event_options.enabled:
            dispatcher = WebhookDispatcher(
                event_options.webhook,
                repository=repository,
            )
            webhook_dispatchers.append(dispatcher)
            event_engines[pad_index] = EventEngine(
                stream_id=stream["stream_id"],
                options=event_options,
                on_event=dispatcher.enqueue,
            )
        if risk_options.enabled:
            risk_dispatcher = WebhookDispatcher(
                risk_options.webhook,
                repository=repository,
            )
            webhook_dispatchers.append(risk_dispatcher)
            fishing_risk_engines[pad_index] = FishingRiskEngine(
                stream_id=stream["stream_id"],
                options=risk_options,
                on_event=risk_dispatcher.enqueue,
            )
            fishing_risk_cache.mark_state(
                pad_index,
                "starting" if vessel_detection_client is not None else "error",
                (
                    "等待船舶轨迹"
                    if vessel_detection_client is not None
                    else "船舶检测组件不可用"
                ),
            )
    garbage_overlay_cache = GarbageOverlayCache()
    metrics = MetricsState(
        stream_ids=[item["stream_id"] for item in streams],
        metrics_path=Path(config["metrics_path"]),
        interval_seconds=float(config["stats_interval_seconds"]),
        minimum_healthy_fps=float(config["minimum_healthy_fps"]),
        group_id=config["group_id"],
        generation=int(config["generation"]),
        license_plate_enabled={
            item["stream_id"]: bool(
                item.get("license_plate", {}).get("enabled", False)
            )
            for item in streams
        },
        night_vision_enabled={
            item["stream_id"]: bool(
                item.get("night_vision", {}).get("enabled", False)
            )
            for item in streams
        },
        gas_cylinder_enabled={
            item["stream_id"]: bool(
                item.get("gas_cylinder", {}).get("enabled", False)
            )
            for item in streams
        },
        gas_cylinder_cache=gas_cylinder_cache,
        vessel_detection_enabled={
            item["stream_id"]: bool(
                item.get("vessel_detection", {}).get("enabled", False)
            )
            for item in streams
        },
        vessel_detection_cache=vessel_detection_cache,
        ground_litter_enabled={
            item["stream_id"]: bool(
                item.get("ground_litter", {}).get("enabled", False)
            )
            for item in streams
        },
        ground_litter_cache=ground_litter_cache,
        fishing_risk_enabled={
            item["stream_id"]: bool(
                item.get("fishing_risk", {}).get("enabled", False)
            )
            for item in streams
        },
        fishing_risk_cache=fishing_risk_cache,
        ptz_verification_coordinators=ptz_verification_coordinators,
        ptz_verification_errors=ptz_verification_errors,
    )
    latency_tracker = InferenceLatencyTracker()

    class OverlayOperator(BatchMetadataOperator):
        def __init__(self) -> None:
            super().__init__()
            self._processor = OverlayProcessor(
                policies,
                metrics,
                latency_tracker,
                event_engines,
                garbage_overlay_cache,
                gas_cylinder_cache,
                vessel_detection_cache,
                ground_litter_cache,
                fishing_risk_cache,
                ptz_verification_by_pad,
                frame_width=int(config["mux_width"]),
                frame_height=int(config["mux_height"]),
            )

        def handle_metadata(self, batch_meta: Any) -> None:
            self._processor.process(batch_meta, osd)

    class BufferCounter(BufferOperator):
        def __init__(
            self,
            stream_id: str,
            observer: Any,
            include_buffer: bool = False,
        ) -> None:
            super().__init__()
            self._stream_id = stream_id
            self._observer = observer
            self._include_buffer = include_buffer

        def handle_buffer(self, _buffer: Any) -> bool:
            if self._include_buffer:
                self._observer(self._stream_id, _buffer)
            else:
                self._observer(self._stream_id)
            return True

    class LatencyStart(BatchMetadataOperator):
        def handle_metadata(self, batch_meta: Any) -> None:
            latency_tracker.start_batch(batch_meta)

    class LicensePlateMetadata(BatchMetadataOperator):
        def __init__(self) -> None:
            super().__init__()
            self._frames = 0
            self._tracker = PlateIdentityTracker()

        def handle_metadata(self, batch_meta: Any) -> None:
            self._frames += 1
            components = self._tracker.process(batch_meta)
            if (
                os.environ.get("RTSP_LPR_DEBUG") == "1"
                and self._frames % 30 == 0
            ):
                LOGGER.warning("LPD诊断: components=%s", components)

    class GarbageFrames(BufferRetriever):
        def __init__(self) -> None:
            super().__init__()
            self._processor = GarbageFrameProcessor(
                policies=policies,
                event_engines=event_engines,
                labels=garbage_labels,
                evidence_writers=evidence_writers,
                metrics_observer=metrics.observe_garbage_analysis,
                overlay_cache=garbage_overlay_cache,
                # garbage_rgba_caps scales this lossy side branch to 640x360,
                # and metadata rectangles are transformed to the same space.
                frame_width=640,
                frame_height=360,
            )

        def consume(self, buffer: Any) -> int:
            try:
                frames = [
                    buffer.extract(index)
                    for index in range(int(buffer.batch_size))
                ]
                self._processor.process(buffer.batch_meta, frames)
            except Exception:
                # This receiver owns a lossy side branch. A bad analysis frame
                # must never terminate or stall the primary RTSP pipeline.
                LOGGER.exception("垃圾背景分析失败，已跳过当前帧")
            # DeepStream 8's BufferRetriever binding expects an integer
            # return. Returning None raises a pybind11 cast_error even when a
            # lossy frame error was handled above.
            return 1

    class GroundLitterFrames(BufferRetriever):
        def __init__(self) -> None:
            super().__init__()
            assert ground_litter_client is not None
            self._processor = GroundLitterFrameProcessor(
                ground_litter_client,
                night_by_pad={
                    pad_index: bool(
                        stream.get("night_vision", {}).get(
                            "enabled",
                            False,
                        )
                    )
                    for pad_index, stream in enumerate(config["streams"])
                },
                actor_classes_by_pad={
                    pad_index: _ground_litter_occluder_classes(
                        policy.ground_litter
                    )
                    for pad_index, policy in policies.items()
                },
            )

        def consume(self, buffer: Any) -> int:
            try:
                frames = [
                    buffer.extract(index)
                    for index in range(int(buffer.batch_size))
                ]
                self._processor.process(
                    buffer.batch_meta,
                    frames,
                    frame_width=int(config["mux_width"]),
                    frame_height=int(config["mux_height"]),
                )
            except Exception:
                # A lossy side branch must never terminate the RTSP pipeline.
                LOGGER.exception("零散垃圾帧提取失败，已跳过且主RTSP继续运行")
            return 1

    class GarbageIntervalSkipper(BufferOperator):
        def __init__(self, analysis_fps: float) -> None:
            super().__init__()
            self._period = 1.0 / max(float(analysis_fps), 0.1)
            self._next_due = 0.0

        def handle_buffer(self, _buffer: Any) -> bool:
            now = time.monotonic()
            if now < self._next_due:
                return False
            self._next_due = now + self._period
            return True

    class GasCylinderFrames(BufferRetriever):
        def __init__(self) -> None:
            super().__init__()
            assert gas_cylinder_client is not None
            self._processor = GasCylinderFrameProcessor(
                gas_cylinder_client
            )

        def consume(self, buffer: Any) -> int:
            try:
                frames = [
                    buffer.extract(index)
                    for index in range(int(buffer.batch_size))
                ]
                self._processor.process(buffer.batch_meta, frames)
            except Exception:
                LOGGER.exception(
                    "燃气瓶帧提取失败，已跳过且主RTSP继续运行"
                )
            return 1

    class VesselFrames(BufferRetriever):
        def __init__(self) -> None:
            super().__init__()
            assert vessel_detection_client is not None
            self._processor = VesselFrameProcessor(
                vessel_detection_client,
                vessel_cache=vessel_detection_cache,
                fishing_risk_engines=fishing_risk_engines,
                fishing_risk_cache=fishing_risk_cache,
                evidence_writers=evidence_writers,
                ptz_verification_coordinators=ptz_verification_by_pad,
            )

        def consume(self, buffer: Any) -> int:
            try:
                frames = [
                    buffer.extract(index)
                    for index in range(int(buffer.batch_size))
                ]
                self._processor.process(buffer.batch_meta, frames)
            except Exception:
                LOGGER.exception(
                    "船舶帧提取失败，已跳过且主RTSP继续运行"
                )
            return 1

    pipeline = Pipeline(f"rtsp-yolo-{config['group_id'][:8]}")
    references: list[Any] = []
    latency_probe = Probe("latency_start", LatencyStart())
    plate_metadata_probe = Probe(
        "plate_metadata",
        LicensePlateMetadata(),
    )
    garbage_receiver = (
        Receiver("garbage_frames", GarbageFrames())
        if garbage_config_path is not None
        else None
    )
    garbage_skip_probe = (
        Probe(
            "garbage_interval",
            GarbageIntervalSkipper(
                float(config.get("garbage", {}).get("analysis_fps", 3.0))
            ),
        )
        if garbage_config_path is not None
        else None
    )
    gas_cylinder_receiver = (
        Receiver("gas_cylinder_frames", GasCylinderFrames())
        if bool(gas_config.get("enabled", False))
        else None
    )
    gas_cylinder_skip_probe = (
        Probe(
            "gas_cylinder_interval",
            GarbageIntervalSkipper(
                float(gas_config.get("analysis_fps", 1.0))
            ),
        )
        if bool(gas_config.get("enabled", False))
        else None
    )
    vessel_receiver = (
        Receiver("vessel_frames", VesselFrames())
        if vessel_detection_client is not None
        else None
    )
    vessel_skip_probe = (
        Probe(
            "vessel_interval",
            GarbageIntervalSkipper(
                float(vessel_config.get("analysis_fps", 5.0))
            ),
        )
        if vessel_detection_client is not None
        else None
    )
    ground_litter_receiver = (
        Receiver("ground_litter_frames", GroundLitterFrames())
        if ground_litter_client is not None
        else None
    )
    ground_litter_skip_probe = (
        Probe(
            "ground_litter_interval",
            GarbageIntervalSkipper(
                float(ground_litter_config.get("analysis_fps", 1.0))
            ),
        )
        if ground_litter_client is not None
        else None
    )
    references.append(latency_probe)
    references.append(plate_metadata_probe)
    if garbage_receiver is not None:
        references.append(garbage_receiver)
    if garbage_skip_probe is not None:
        references.append(garbage_skip_probe)
    if gas_cylinder_receiver is not None:
        references.append(gas_cylinder_receiver)
    if gas_cylinder_skip_probe is not None:
        references.append(gas_cylinder_skip_probe)
    if vessel_receiver is not None:
        references.append(vessel_receiver)
    if vessel_skip_probe is not None:
        references.append(vessel_skip_probe)
    if ground_litter_receiver is not None:
        references.append(ground_litter_receiver)
    if ground_litter_skip_probe is not None:
        references.append(ground_litter_skip_probe)

    def counter_probe(
        name: str,
        index: int,
        observer: Any,
        *,
        include_buffer: bool = False,
    ) -> Any:
        probe = Probe(
            f"{name}_{index}",
            BufferCounter(
                streams[index]["stream_id"],
                observer,
                include_buffer=include_buffer,
            ),
        )
        references.append(probe)
        return probe

    def pre_encode_probe_factory(index: int) -> Any:
        return counter_probe(
            "pre_encode_counter",
            index,
            metrics.observe_pre_encode,
            include_buffer=True,
        )

    def overlay_probe_factory(index: int) -> Any:
        probe = Probe(f"overlay_{index}", OverlayOperator())
        references.append(probe)
        return probe

    def publish_probe_factory(index: int) -> Any:
        return counter_probe(
            "publish_counter",
            index,
            metrics.observe_publish,
        )

    _add_pipeline_nodes(
        pipeline,
        config,
        inference_config_path,
        latency_probe,
        overlay_probe_factory,
        plate_metadata_probe,
        pre_encode_probe_factory,
        publish_probe_factory,
        lpd_config_path,
        lpr_config_path,
        garbage_config_path,
        garbage_receiver,
        garbage_skip_probe,
        gas_cylinder_receiver,
        gas_cylinder_skip_probe,
        vessel_receiver,
        vessel_skip_probe,
        ground_litter_receiver,
        ground_litter_skip_probe,
    )
    stopped = threading.Event()

    def stop_pipeline(_signum: int, _frame: Any) -> None:
        if stopped.is_set():
            return
        stopped.set()
        for coordinator in ptz_verification_coordinators.values():
            coordinator.request_shutdown()
        pipeline.stop()

    signal.signal(signal.SIGTERM, stop_pipeline)
    signal.signal(signal.SIGINT, stop_pipeline)
    engine_exists = Path(config["engine_path"]).is_file()
    LOGGER.info(
        "启动DeepStream组: group=%s streams=%d engine=%s profile=%s",
        config["group_id"][:8],
        len(streams),
        "cached" if engine_exists else "building",
        "night" if night_vision_enabled else "day",
    )
    try:
        pipeline.start().wait()
    finally:
        if ptz_control_monitor is not None:
            ptz_control_monitor.shutdown()
        for coordinator in ptz_verification_coordinators.values():
            coordinator.shutdown(
                timeout=coordinator.options.shutdown_timeout_seconds
            )
        if gas_cylinder_client is not None:
            gas_cylinder_client.shutdown()
        if vessel_detection_client is not None:
            vessel_detection_client.shutdown()
        if ground_litter_client is not None:
            ground_litter_client.shutdown()
        for dispatcher in webhook_dispatchers:
            dispatcher.shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DeepStream双路推理worker")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
    )
    run(args.config.expanduser().resolve())


if __name__ == "__main__":
    main()
