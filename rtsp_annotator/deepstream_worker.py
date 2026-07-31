from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .labels import chinese_label
from .license_plate import (
    LICENSE_PLATE_DETECTOR_UID,
    LICENSE_PLATE_RECOGNIZER_UID,
    PRIMARY_DETECTOR_UID,
    PlateConsensus,
)


LOGGER = logging.getLogger("rtsp_annotator.deepstream")


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


@dataclass(slots=True)
class StreamPolicy:
    stream_id: str
    classes: frozenset[int] | None
    conf: float
    roi: list[list[float]] | None
    labels: dict[int, str]
    license_plate_enabled: bool = False
    minimum_plate_confirmations: int = 2


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
    ) -> None:
        self._metrics_path = metrics_path
        self._interval_seconds = interval_seconds
        self._minimum_healthy_fps = minimum_healthy_fps
        self._group_id = group_id
        self._generation = generation
        self._license_plate_enabled = license_plate_enabled or {}
        self._night_vision_enabled = night_vision_enabled or {}
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
        self._last_frames = dict(self._frames)
        self._last_pre_encode = dict(self._pre_encode)
        self._last_published = dict(self._published)
        self._last_detections = dict(self._detections)
        self._last_plate_detections = dict(self._plate_detections)
        self._last_plate_reads = dict(self._plate_reads)
        self._inference_ms: dict[str, list[float]] = {
            stream_id: [] for stream_id in stream_ids
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

    def observe_pre_encode(self, stream_id: str) -> None:
        with self._lock:
            self._pre_encode[stream_id] += 1

    def observe_license_plates(
        self,
        stream_id: str,
        detections: int,
        reads: int,
    ) -> None:
        with self._lock:
            self._plate_detections[stream_id] += detections
            self._plate_reads[stream_id] += reads

    def observe_publish(self, stream_id: str) -> None:
        with self._lock:
            self._published[stream_id] += 1

    def _maybe_write_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_report_at
        if elapsed < self._interval_seconds:
            return
        streams: dict[str, dict[str, float | int | bool | str]] = {}
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
                "average_inference_ms": average_inference_ms,
                "p95_inference_ms": p95_inference_ms,
                "interval_detections": interval_detections,
                "total_detections": self._detections[stream_id],
                "interval_plate_detections": interval_plate_detections,
                "interval_plate_reads": interval_plate_reads,
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
        self._inference_ms = {
            stream_id: [] for stream_id in self._inference_ms
        }
        for stream_id, report in streams.items():
            LOGGER.info(
                "状态: stream=%s, pipeline=%.1f FPS, "
                "pre-encode=%.1f FPS, publish=%.1f FPS, "
                "infer=%.1f/P95 %.1f ms, detections=%d, healthy=%s",
                stream_id[:8],
                report["inference_fps"],
                report["pre_encode_fps"],
                report["publish_fps"],
                report["average_inference_ms"],
                report["p95_inference_ms"],
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
    ) -> None:
        self._policies = policies
        self._metrics = metrics
        self._latency_tracker = latency_tracker
        self._plate_consensus = {
            pad_index: PlateConsensus(
                minimum_confirmations=policy.minimum_plate_confirmations,
            )
            for pad_index, policy in policies.items()
            if policy.license_plate_enabled
        }

    def process(self, batch_meta: Any, osd: Any) -> None:
        for frame_meta in batch_meta.frame_items:
            policy = self._policies.get(int(frame_meta.pad_index))
            if policy is None:
                continue
            detections = 0
            plate_detections = 0
            plate_reads = 0
            width = max(float(frame_meta.pipeline_width), 1.0)
            height = max(float(frame_meta.pipeline_height), 1.0)
            for object_meta in frame_meta.object_items:
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
                if not self._object_allowed(
                    object_meta,
                    policy,
                    width,
                    height,
                ):
                    self._hide_object(object_meta)
                    continue
                detections += 1
                self._style_object(object_meta, policy, osd)
            if policy.roi:
                self._draw_roi(batch_meta, frame_meta, policy.roi, osd)
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
    ) -> bool:
        class_id = int(object_meta.class_id)
        if policy.classes is not None and class_id not in policy.classes:
            return False
        confidence = float(object_meta.confidence)
        if confidence >= 0 and confidence < policy.conf:
            return False
        if policy.roi:
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
    ) -> None:
        class_id = int(object_meta.class_id)
        label = policy.labels.get(class_id, f"类别{class_id}")
        rectangle = object_meta.rect_params
        rectangle.border_width = 3
        rectangle.border_color = osd.Color(0.0, 1.0, 0.0, 1.0)
        text = object_meta.text_params
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
        text.text_bg_clr = osd.Color(0.0, 0.45, 0.0, 0.9)

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
    ) -> None:
        width = int(frame_meta.pipeline_width)
        height = int(frame_meta.pipeline_height)
        display_meta = batch_meta.acquire_display_meta()
        for start, end in zip(roi, roi[1:] + roi[:1]):
            line = osd.Line()
            line.x1 = int(start[0] * width)
            line.y1 = int(start[1] * height)
            line.x2 = int(end[0] * width)
            line.y2 = int(end[1] * height)
            line.width = 3
            line.color = osd.Color(1.0, 0.75, 0.0, 1.0)
            display_meta.add_line(line)
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
            license_plate_enabled=bool(
                stream.get("license_plate", {}).get("enabled", False)
            ),
            minimum_plate_confirmations=int(
                stream.get("license_plate", {}).get(
                    "minimum_confirmations",
                    2,
                )
            ),
        )
        for pad_index, stream in enumerate(config["streams"])
    }


def _add_pipeline_nodes(
    pipeline: Any,
    config: dict[str, Any],
    inference_config_path: Path,
    latency_probe: Any,
    overlay_probe: Any,
    plate_metadata_probe: Any,
    pre_encode_probe_factory: Any,
    publish_probe_factory: Any,
    lpd_config_path: Path | None = None,
    lpr_config_path: Path | None = None,
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
                "rtsp-reconnect-interval": 5,
                "rtsp-reconnect-attempts": -1,
                "select-rtp-protocol": 4,
            },
        )
        pipeline.link((source, "mux"), ("vsrc_%u", ""))

    pipeline.add(
        "queue",
        "pre_infer",
        {"max-size-buffers": 2, "leaky": 2},
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
    pipeline.add(
        "nvdsosd",
        "osd",
        {"gpu-id": gpu_id, "process-mode": 1},
    )
    pipeline.add("nvstreamdemux", "demux")
    analytics_nodes = ["mux", "pre_infer", "primary_infer"]
    if license_plate_enabled:
        analytics_nodes.extend(
            ("vehicle_tracker", "license_plate_detector")
        )
    if not license_plate_enabled:
        analytics_nodes.append("tracker")
    if license_plate_enabled:
        analytics_nodes.append("license_plate_recognizer")
    analytics_nodes.extend(("osd_convert", "rgba_caps", "osd", "demux"))
    pipeline.link(*analytics_nodes)
    if license_plate_enabled:
        pipeline.attach("license_plate_detector", plate_metadata_probe)
    pipeline.attach(
        (
            "license_plate_recognizer"
            if license_plate_enabled
            else "tracker"
        ),
        overlay_probe,
    )
    pipeline.attach("pre_infer", latency_probe)

    for index, stream in enumerate(config["streams"]):
        queue = f"publish_queue_{index}"
        convert = f"publish_convert_{index}"
        caps = f"publish_caps_{index}"
        encoder = f"encoder_{index}"
        parser = f"parser_{index}"
        parser_caps = f"parser_caps_{index}"
        clock_sync = f"publish_clock_{index}"
        sink = f"rtsp_sink_{index}"
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
                "insert-aud": True,
                "insert-sps-pps": True,
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
                # rtspclientsink is a bin rather than a clock-synchronising
                # GstBaseSink. Pace encoded access units from their PTS so
                # inference batching cannot publish them in bursts.
                "sync": True,
                "sync-to-first": True,
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
        pipeline.link(("demux", queue), ("src_%u", ""))
        pipeline.link(
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
        BufferOperator,
        Pipeline,
        Probe,
        osd,
    )

    inference_config_path = config_path.with_name("nvinfer.txt")
    build_inference_config(config, inference_config_path)
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
    policies = _load_policies(config)
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
    )
    latency_tracker = InferenceLatencyTracker()

    class OverlayOperator(BatchMetadataOperator):
        def __init__(self) -> None:
            super().__init__()
            self._processor = OverlayProcessor(
                policies,
                metrics,
                latency_tracker,
            )

        def handle_metadata(self, batch_meta: Any) -> None:
            self._processor.process(batch_meta, osd)

    class BufferCounter(BufferOperator):
        def __init__(
            self,
            stream_id: str,
            observer: Any,
        ) -> None:
            super().__init__()
            self._stream_id = stream_id
            self._observer = observer

        def handle_buffer(self, _buffer: Any) -> bool:
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

    pipeline = Pipeline(f"rtsp-yolo-{config['group_id'][:8]}")
    references: list[Any] = []
    latency_probe = Probe("latency_start", LatencyStart())
    overlay_probe = Probe("overlay", OverlayOperator())
    plate_metadata_probe = Probe(
        "plate_metadata",
        LicensePlateMetadata(),
    )
    references.extend((latency_probe, overlay_probe))
    references.append(plate_metadata_probe)

    def counter_probe(
        name: str,
        index: int,
        observer: Any,
    ) -> Any:
        probe = Probe(
            f"{name}_{index}",
            BufferCounter(streams[index]["stream_id"], observer),
        )
        references.append(probe)
        return probe

    def pre_encode_probe_factory(index: int) -> Any:
        return counter_probe(
            "pre_encode_counter",
            index,
            metrics.observe_pre_encode,
        )

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
        overlay_probe,
        plate_metadata_probe,
        pre_encode_probe_factory,
        publish_probe_factory,
        lpd_config_path,
        lpr_config_path,
    )
    stopped = threading.Event()

    def stop_pipeline(_signum: int, _frame: Any) -> None:
        if stopped.is_set():
            return
        stopped.set()
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
    pipeline.start().wait()


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
