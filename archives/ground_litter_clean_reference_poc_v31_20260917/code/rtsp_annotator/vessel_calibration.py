from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np

from .config import redact_url
from .event_engine import NormalizedRect
from .vessel_detection import (
    UltralyticsVesselDetector,
    VesselCandidate,
    VesselDetectionOptions,
    rectangle_iou,
)


DEFAULT_SEGMENTATION_MODEL = os.environ.get(
    "VESSEL_SEGMENTATION_MODEL",
    "openmmlab/upernet-convnext-tiny",
)
WATER_LABELS = frozenset({"water", "sea", "river", "lake"})
DEFAULT_SEGMENTATION_REVISION = "876ffc5"


@dataclass(frozen=True, slots=True)
class ReviewExclusionCandidate:
    rectangle: NormalizedRect
    frame_coverage: float
    median_confidence: float
    center_motion: float

    def to_payload(self) -> dict[str, Any]:
        left = self.rectangle.left
        top = self.rectangle.top
        right = left + self.rectangle.width
        bottom = top + self.rectangle.height
        return {
            "decision": "manual_review_required",
            "polygon": [
                [_round(left), _round(top)],
                [_round(right), _round(top)],
                [_round(right), _round(bottom)],
                [_round(left), _round(bottom)],
            ],
            "frame_coverage": _round(self.frame_coverage),
            "median_confidence": _round(self.median_confidence),
            "center_motion": _round(self.center_motion),
        }


@dataclass(slots=True)
class _CandidateCluster:
    frame_indexes: set[int] = field(default_factory=set)
    rectangles: list[NormalizedRect] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    @property
    def representative(self) -> NormalizedRect:
        return _median_rectangle(self.rectangles)


class PretrainedWaterSegmenter:
    """Lazy optional semantic segmenter used only during calibration."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_SEGMENTATION_MODEL,
        *,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor
            from transformers import AutoModelForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "缺少标定依赖，请安装项目的 calibration 可选依赖"
            ) from exc

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self.device = device
        self.model_name_or_path = model_name_or_path
        remote_options: dict[str, Any] = {}
        if not Path(model_name_or_path).expanduser().exists():
            remote_options["revision"] = DEFAULT_SEGMENTATION_REVISION
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                model_name_or_path,
                local_files_only=True,
                use_fast=False,
                **remote_options,
            )
            self._model = AutoModelForSemanticSegmentation.from_pretrained(
                model_name_or_path,
                local_files_only=True,
                use_safetensors=True,
                **remote_options,
            ).to(device)
        except OSError:
            self._processor = AutoImageProcessor.from_pretrained(
                model_name_or_path,
                use_fast=False,
                **remote_options,
            )
            self._model = AutoModelForSemanticSegmentation.from_pretrained(
                model_name_or_path,
                use_safetensors=True,
                **remote_options,
            ).to(device)
        self._model.eval()
        labels = {
            int(label_id): str(label).strip().lower()
            for label_id, label in self._model.config.id2label.items()
        }
        self.water_label_ids = tuple(
            label_id
            for label_id, label in labels.items()
            if label in WATER_LABELS
        )
        if not self.water_label_ids:
            raise RuntimeError(
                "语义分割模型没有water/sea/river/lake类别，"
                "无法标定水域"
            )

    def water_mask(self, bgr_frame: np.ndarray) -> np.ndarray:
        torch = self._torch
        height, width = bgr_frame.shape[:2]
        rgb = np.ascontiguousarray(bgr_frame[..., ::-1])
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self._model(**inputs).logits
            labels = logits.argmax(dim=1)[0].cpu().numpy()
        del inputs, logits
        low_resolution_mask = np.isin(labels, self.water_label_ids).astype(
            np.uint8
        )
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("缺少OpenCV，无法缩放水域掩码") from exc
        return cv2.resize(
            low_resolution_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)


def sample_video_frames(
    source: str,
    *,
    sample_count: int,
    sample_interval_seconds: float,
    transport: str = "tcp",
    open_timeout: float = 10.0,
    read_timeout: float = 15.0,
) -> list[np.ndarray]:
    if sample_count < 1:
        raise ValueError("sample_count必须大于0")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds必须大于0")
    try:
        import av
    except ImportError:
        return _sample_video_frames_with_opencv(
            source,
            sample_count=sample_count,
            sample_interval_seconds=sample_interval_seconds,
            transport=transport,
            open_timeout=open_timeout,
            read_timeout=read_timeout,
        )

    is_rtsp = source.lower().startswith(("rtsp://", "rtsps://"))
    kwargs: dict[str, Any] = {}
    if is_rtsp:
        kwargs["options"] = {
            "rtsp_transport": transport,
            "fflags": "nobuffer",
            "flags": "low_delay",
        }
        kwargs["timeout"] = (open_timeout, read_timeout)
    container = av.open(source, mode="r", **kwargs)
    frames: list[np.ndarray] = []
    try:
        stream = next(
            item for item in container.streams if item.type == "video"
        )
        stream.thread_type = "AUTO"
        fallback_rate = float(stream.average_rate or 25)
        first_time: float | None = None
        next_sample_time = 0.0
        for frame_index, frame in enumerate(container.decode(video=0)):
            frame_time = (
                float(frame.time)
                if frame.time is not None
                else frame_index / fallback_rate
            )
            if first_time is None:
                first_time = frame_time
            relative_time = max(frame_time - first_time, 0.0)
            if relative_time + 1e-6 < next_sample_time:
                continue
            frames.append(frame.to_ndarray(format="bgr24"))
            if len(frames) >= sample_count:
                break
            next_sample_time += sample_interval_seconds
    except StopIteration as exc:
        raise RuntimeError("输入中没有视频流") from exc
    finally:
        container.close()
    if len(frames) < sample_count:
        raise RuntimeError(
            f"标定画面不足: 需要{sample_count}帧，"
            f"只读取到{len(frames)}帧"
        )
    return frames


def _sample_video_frames_with_opencv(
    source: str,
    *,
    sample_count: int,
    sample_interval_seconds: float,
    transport: str,
    open_timeout: float,
    read_timeout: float,
) -> list[np.ndarray]:
    """Read calibration frames when the lightweight runtime lacks PyAV."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "缺少PyAV和OpenCV，无法读取标定视频"
        ) from exc

    is_rtsp = source.lower().startswith(("rtsp://", "rtsps://"))
    parameters: list[int] = []
    for property_name, timeout in (
        ("CAP_PROP_OPEN_TIMEOUT_MSEC", open_timeout),
        ("CAP_PROP_READ_TIMEOUT_MSEC", read_timeout),
    ):
        property_id = getattr(cv2, property_name, None)
        if property_id is not None:
            parameters.extend((property_id, round(timeout * 1000)))

    capture_options_name = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    previous_capture_options = os.environ.get(capture_options_name)
    if is_rtsp:
        os.environ[capture_options_name] = f"rtsp_transport;{transport}"
    try:
        capture = cv2.VideoCapture(
            source,
            getattr(cv2, "CAP_FFMPEG", 0),
            parameters,
        )
    finally:
        if is_rtsp:
            if previous_capture_options is None:
                os.environ.pop(capture_options_name, None)
            else:
                os.environ[capture_options_name] = previous_capture_options

    if not capture.isOpened():
        capture.release()
        raise RuntimeError("无法打开标定视频或RTSP流")

    frames: list[np.ndarray] = []
    fallback_rate = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fallback_rate) or fallback_rate <= 0:
        fallback_rate = 25.0
    first_time: float | None = None
    next_sample_time = 0.0
    frame_index = 0
    try:
        while len(frames) < sample_count:
            success, frame = capture.read()
            if not success:
                break
            timestamp_milliseconds = float(
                capture.get(cv2.CAP_PROP_POS_MSEC)
            )
            frame_time = (
                timestamp_milliseconds / 1000.0
                if math.isfinite(timestamp_milliseconds)
                and timestamp_milliseconds > 0
                else frame_index / fallback_rate
            )
            frame_index += 1
            if first_time is None:
                first_time = frame_time
            relative_time = max(frame_time - first_time, 0.0)
            if relative_time + 1e-6 < next_sample_time:
                continue
            frames.append(np.asarray(frame))
            next_sample_time += sample_interval_seconds
    finally:
        capture.release()
    if len(frames) < sample_count:
        raise RuntimeError(
            f"标定画面不足: 需要{sample_count}帧，"
            f"只读取到{len(frames)}帧"
        )
    return frames


def build_consensus_water_mask(
    masks: Sequence[np.ndarray],
    *,
    minimum_ratio: float = 0.60,
    close_ratio: float = 0.02,
    dilation_ratio: float = 0.008,
    minimum_area_ratio: float = 0.02,
) -> np.ndarray:
    if not masks:
        raise ValueError("至少需要一张水域掩码")
    if not 0 < minimum_ratio <= 1:
        raise ValueError("minimum_ratio必须在(0, 1]范围内")
    shape = np.asarray(masks[0]).shape
    if len(shape) != 2 or any(np.asarray(mask).shape != shape for mask in masks):
        raise ValueError("所有水域掩码必须是相同宽高的二维数组")
    votes = np.zeros(shape, dtype=np.uint16)
    for mask in masks:
        votes += np.asarray(mask, dtype=bool)
    required_votes = math.ceil(len(masks) * minimum_ratio)
    consensus = (votes >= required_votes).astype(np.uint8) * 255

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("缺少OpenCV，无法生成水域ROI") from exc
    minimum_dimension = min(shape)
    close_size = _odd_kernel_size(minimum_dimension * close_ratio)
    if close_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        )
        consensus = cv2.morphologyEx(consensus, cv2.MORPH_CLOSE, kernel)
    dilation_size = _odd_kernel_size(minimum_dimension * dilation_ratio)
    if dilation_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_size, dilation_size),
        )
        consensus = cv2.dilate(consensus, kernel)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        consensus,
        connectivity=8,
    )
    if count <= 1:
        raise RuntimeError("没有找到稳定水域，请改用人工ROI")
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
    if largest_area / consensus.size < minimum_area_ratio:
        raise RuntimeError("稳定水域面积过小，请改用人工ROI")
    return labels == largest_label


def water_mask_to_polygon(
    mask: np.ndarray,
    *,
    approximation_ratio: float = 0.008,
    maximum_points: int = 24,
) -> tuple[tuple[float, float], ...]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("缺少OpenCV，无法生成水域ROI") from exc
    binary = np.asarray(mask, dtype=np.uint8) * 255
    contours, _hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise RuntimeError("水域掩码为空，请改用人工ROI")
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    ratio = approximation_ratio
    polygon = cv2.approxPolyDP(contour, perimeter * ratio, True)
    while len(polygon) > maximum_points and ratio < 0.1:
        ratio *= 1.4
        polygon = cv2.approxPolyDP(contour, perimeter * ratio, True)
    height, width = binary.shape
    points = tuple(
        (
            _round(float(point[0][0]) / max(width - 1, 1)),
            _round(float(point[0][1]) / max(height - 1, 1)),
        )
        for point in polygon
    )
    if len(points) < 3 or len(set(points)) < 3:
        raise RuntimeError(
            "水域轮廓无法形成有效多边形，请改用人工ROI"
        )
    return points


def expand_water_mask_to_lower_envelope(
    mask: np.ndarray,
    *,
    minimum_column_coverage: float = 0.20,
    smoothing_ratio: float = 0.05,
    upward_margin_ratio: float = 0.015,
) -> np.ndarray:
    """Turn water evidence into a conservative below-horizon ROI.

    Dense boats and piers can occlude or confuse semantic water labels. A
    literal segmentation polygon would therefore remove the exact areas in
    which recall matters most. Fixed harbor cameras normally view navigable
    water below a shoreline/horizon, so interpolate the upper water boundary
    and keep everything below it. This deliberately trades extra dock pixels
    for avoiding silent vessel misses.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("水域掩码必须是二维数组")
    height, width = binary.shape
    valid_columns = np.flatnonzero(binary.any(axis=0))
    if len(valid_columns) / max(width, 1) < minimum_column_coverage:
        raise RuntimeError("水域横向覆盖不足，请改用人工ROI")
    boundaries = np.full(width, np.nan, dtype=np.float32)
    for x in valid_columns:
        boundaries[x] = float(np.flatnonzero(binary[:, x])[0])
    boundaries = np.interp(
        np.arange(width),
        valid_columns,
        boundaries[valid_columns],
    )
    window = _odd_kernel_size(width * smoothing_ratio)
    if window > 1:
        padding = window // 2
        padded = np.pad(boundaries, padding, mode="edge")
        boundaries = np.median(
            np.lib.stride_tricks.sliding_window_view(padded, window),
            axis=1,
        )
    boundaries = np.clip(
        boundaries - height * upward_margin_ratio,
        0,
        height - 1,
    ).astype(np.int32)
    envelope = np.zeros_like(binary)
    for x, top in enumerate(boundaries):
        envelope[top:, x] = True
    return envelope


def propose_inference_regions(
    polygon: Sequence[tuple[float, float]],
    *,
    margin: float = 0.03,
    maximum_zoom_area: float = 0.85,
) -> tuple[tuple[float, float, float, float], ...]:
    left = max(min(point[0] for point in polygon) - margin, 0.0)
    top = max(min(point[1] for point in polygon) - margin, 0.0)
    right = min(max(point[0] for point in polygon) + margin, 1.0)
    bottom = min(max(point[1] for point in polygon) + margin, 1.0)
    regions = [(0.0, 0.0, 1.0, 1.0)]
    if (right - left) * (bottom - top) <= maximum_zoom_area:
        regions.append(
            (_round(left), _round(top), _round(right), _round(bottom))
        )
    return tuple(regions)


def find_review_exclusion_candidates(
    detections_by_frame: Sequence[Sequence[VesselCandidate]],
    *,
    minimum_frame_coverage: float = 0.70,
    maximum_center_motion: float = 0.015,
    match_iou: float = 0.20,
    margin: float = 0.008,
) -> list[ReviewExclusionCandidate]:
    frame_count = len(detections_by_frame)
    if frame_count == 0:
        return []
    clusters: list[_CandidateCluster] = []
    for frame_index, detections in enumerate(detections_by_frame):
        used_clusters: set[int] = set()
        for candidate in sorted(
            detections,
            key=lambda item: item.confidence,
            reverse=True,
        ):
            best_index: int | None = None
            best_overlap = match_iou
            for cluster_index, cluster in enumerate(clusters):
                if cluster_index in used_clusters:
                    continue
                overlap = rectangle_iou(
                    candidate.rectangle,
                    cluster.representative,
                )
                if overlap >= best_overlap:
                    best_index = cluster_index
                    best_overlap = overlap
            if best_index is None:
                cluster = _CandidateCluster()
                clusters.append(cluster)
                best_index = len(clusters) - 1
            cluster = clusters[best_index]
            cluster.frame_indexes.add(frame_index)
            cluster.rectangles.append(candidate.rectangle)
            cluster.confidences.append(candidate.confidence)
            used_clusters.add(best_index)

    review: list[ReviewExclusionCandidate] = []
    minimum_frames = max(3, math.ceil(frame_count * minimum_frame_coverage))
    for cluster in clusters:
        if len(cluster.frame_indexes) < minimum_frames:
            continue
        representative = cluster.representative
        center_x, center_y = representative.center
        center_motion = max(
            math.hypot(
                rectangle.center[0] - center_x,
                rectangle.center[1] - center_y,
            )
            for rectangle in cluster.rectangles
        )
        if center_motion > maximum_center_motion:
            continue
        review.append(
            ReviewExclusionCandidate(
                rectangle=_expand_rectangle(representative, margin),
                frame_coverage=(
                    len(cluster.frame_indexes) / max(frame_count, 1)
                ),
                median_confidence=float(median(cluster.confidences)),
                center_motion=center_motion,
            )
        )
    return sorted(
        review,
        key=lambda item: (item.frame_coverage, item.median_confidence),
        reverse=True,
    )


def render_calibration_preview(
    frame: np.ndarray,
    water_mask: np.ndarray,
    polygon: Sequence[tuple[float, float]],
    review_candidates: Sequence[ReviewExclusionCandidate],
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("缺少OpenCV，无法生成标定预览") from exc
    canvas = frame.copy()
    overlay = canvas.copy()
    overlay[np.asarray(water_mask, dtype=bool)] = (0, 150, 0)
    canvas = cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0)
    height, width = canvas.shape[:2]
    points = np.asarray(
        [
            [round(x * (width - 1)), round(y * (height - 1))]
            for x, y in polygon
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    cv2.polylines(canvas, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
    for candidate in review_candidates:
        rectangle = candidate.rectangle
        left = round(rectangle.left * width)
        top = round(rectangle.top * height)
        right = round((rectangle.left + rectangle.width) * width)
        bottom = round((rectangle.top + rectangle.height) * height)
        cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"REVIEW {candidate.frame_coverage:.0%}",
            (left, max(top - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def calibrate_source(
    source: str,
    *,
    output_path: Path,
    preview_path: Path,
    segmentation_model: str = DEFAULT_SEGMENTATION_MODEL,
    vessel_model_path: Path | None = None,
    device: str = "auto",
    sample_count: int = 12,
    sample_interval_seconds: float = 2.0,
    water_minimum_ratio: float = 0.60,
    transport: str = "tcp",
) -> dict[str, Any]:
    frames = sample_video_frames(
        source,
        sample_count=sample_count,
        sample_interval_seconds=sample_interval_seconds,
        transport=transport,
    )
    segmenter_device = device
    if segmenter_device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                segmenter_device = "cuda:0"
            elif torch.backends.mps.is_available():
                segmenter_device = "mps"
            else:
                segmenter_device = "cpu"
        except (AttributeError, ImportError):
            segmenter_device = "cpu"
    segmenter = PretrainedWaterSegmenter(
        segmentation_model,
        device=segmenter_device,
    )
    masks = [segmenter.water_mask(frame) for frame in frames]
    water_evidence_mask = build_consensus_water_mask(
        masks,
        minimum_ratio=water_minimum_ratio,
    )
    water_mask = expand_water_mask_to_lower_envelope(water_evidence_mask)
    polygon = water_mask_to_polygon(water_mask)
    inference_regions = propose_inference_regions(polygon)

    detections_by_frame: list[list[VesselCandidate]] = []
    if vessel_model_path is not None:
        detector_device = device
        if detector_device == "auto":
            detector_device = (
                "cuda:0" if segmenter.device.startswith("cuda") else "cpu"
            )
        if detector_device == "mps":
            detector_device = "cpu"
        detector = UltralyticsVesselDetector(
            model_path=vessel_model_path,
            device=detector_device,
            half=detector_device.startswith("cuda"),
        )
        options = VesselDetectionOptions(
            enabled=True,
            inference_regions=inference_regions,
            roi=polygon,
        )
        detections_by_frame = [
            detector.detect(
                np.ascontiguousarray(frame[..., ::-1]),
                options,
            )
            for frame in frames
        ]
    review = find_review_exclusion_candidates(detections_by_frame)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "source": redact_url(source),
        "sampled_frames": len(frames),
        "sample_interval_seconds": sample_interval_seconds,
        "segmentation_model": segmentation_model,
        "water_label_ids": list(segmenter.water_label_ids),
        "vessel_detection": {
            "roi": [list(point) for point in polygon],
            "inference_regions": [list(region) for region in inference_regions],
            "exclude_rois": [],
        },
        "stable_detection_review_candidates": [
            candidate.to_payload() for candidate in review
        ],
        "warning": (
            "红色框只表示检测位置长期稳定，"
            "可能是真船也可能是固定误报；"
            "工具不会生成exclude_rois。只有人工确认是固定结构后，"
            "才可手工复制对应polygon。"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    preview = render_calibration_preview(
        frames[len(frames) // 2],
        water_mask,
        polygon,
        review,
    )
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("缺少OpenCV，无法保存标定预览") from exc
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"无法保存标定预览: {preview_path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yolo-vessel-calibrate",
        description=(
            "用预训练语义分割模型为固定船舶监控视角生成水域ROI建议。"
        ),
    )
    parser.add_argument(
        "--input",
        default=os.environ.get("RTSP_INPUT_URL"),
        required=os.environ.get("RTSP_INPUT_URL") is None,
        help="RTSP地址或本地视频，也可使用RTSP_INPUT_URL",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument(
        "--segmentation-model",
        default=DEFAULT_SEGMENTATION_MODEL,
        help="Hugging Face模型ID或本地模型目录",
    )
    parser.add_argument(
        "--vessel-model",
        type=Path,
        help=(
            "可选YOLO .pt模型；用于生成固定候选复核区，不会自动排除"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--water-minimum-ratio", type=float, default=0.60)
    parser.add_argument("--transport", choices=("tcp", "udp"), default="tcp")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    preview = (
        args.preview.expanduser().resolve()
        if args.preview is not None
        else output.with_suffix(".preview.jpg")
    )
    vessel_model = (
        args.vessel_model.expanduser().resolve()
        if args.vessel_model is not None
        else None
    )
    try:
        payload = calibrate_source(
            args.input,
            output_path=output,
            preview_path=preview,
            segmentation_model=args.segmentation_model,
            vessel_model_path=vessel_model,
            device=args.device,
            sample_count=args.sample_count,
            sample_interval_seconds=args.sample_interval,
            water_minimum_ratio=args.water_minimum_ratio,
            transport=args.transport,
        )
    except Exception as exc:
        parser.error(str(exc))
    print(f"已生成配置建议: {output}")
    print(f"已生成可视化复核图: {preview}")
    print(
        "需人工复核的固定候选区: "
        f"{len(payload['stable_detection_review_candidates'])}"
    )
    return 0


def _median_rectangle(rectangles: Sequence[NormalizedRect]) -> NormalizedRect:
    return NormalizedRect(
        float(median(item.left for item in rectangles)),
        float(median(item.top for item in rectangles)),
        float(median(item.width for item in rectangles)),
        float(median(item.height for item in rectangles)),
    )


def _expand_rectangle(rectangle: NormalizedRect, margin: float) -> NormalizedRect:
    left = max(rectangle.left - margin, 0.0)
    top = max(rectangle.top - margin, 0.0)
    right = min(rectangle.left + rectangle.width + margin, 1.0)
    bottom = min(rectangle.top + rectangle.height + margin, 1.0)
    return NormalizedRect(left, top, right - left, bottom - top)


def _odd_kernel_size(value: float) -> int:
    size = max(int(round(value)), 1)
    return size if size % 2 == 1 else size + 1


def _round(value: float) -> float:
    return round(float(value), 6)


if __name__ == "__main__":
    raise SystemExit(main())
