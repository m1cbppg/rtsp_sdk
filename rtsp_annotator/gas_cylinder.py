from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .event_engine import NormalizedRect


_SAFE_PROFILE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True, slots=True)
class GasCylinderOptions:
    """Per-stream options for a fixed-camera gas-cylinder detector."""

    enabled: bool = False
    profile_id: str = "camera_01_ir"
    analysis_fps: float = 1.0
    sample_count: int = 11
    sample_interval_seconds: float = 3.0
    minimum_confirmations: int = 4
    scene_stable_seconds: float = 3.0
    change_confirm_seconds: float = 3.0
    forced_refresh_seconds: float = 300.0
    retry_seconds: float = 10.0
    scene_change_ratio: float = 0.006
    motion_ratio: float = 0.003
    mask_nms_iou: float = 0.50
    temporal_match_iou: float = 0.30
    large_count_change: int = 2
    alarm_threshold: int = 18
    display_ids: bool = False
    include_partial: bool = True

    def validate(self) -> None:
        if not _SAFE_PROFILE_ID.fullmatch(self.profile_id):
            raise ValueError(
                "gas_cylinder.profile_id只能包含字母、数字、下划线和短横线"
            )
        if not 0.1 <= self.analysis_fps <= 5:
            raise ValueError("gas_cylinder.analysis_fps必须在[0.1, 5]范围内")
        if not 3 <= self.sample_count <= 15:
            raise ValueError("gas_cylinder.sample_count必须在[3, 15]范围内")
        if not 0.2 <= self.sample_interval_seconds <= 10:
            raise ValueError(
                "gas_cylinder.sample_interval_seconds必须在[0.2, 10]范围内"
            )
        if not 2 <= self.minimum_confirmations <= self.sample_count:
            raise ValueError(
                "gas_cylinder.minimum_confirmations必须在"
                "[2, sample_count]范围内"
            )
        if not 0.5 <= self.scene_stable_seconds <= 30:
            raise ValueError(
                "gas_cylinder.scene_stable_seconds必须在[0.5, 30]范围内"
            )
        if not 0.5 <= self.change_confirm_seconds <= 30:
            raise ValueError(
                "gas_cylinder.change_confirm_seconds必须在[0.5, 30]范围内"
            )
        if not 30 <= self.forced_refresh_seconds <= 86_400:
            raise ValueError(
                "gas_cylinder.forced_refresh_seconds必须在[30, 86400]范围内"
            )
        if not 1 <= self.retry_seconds <= 300:
            raise ValueError("gas_cylinder.retry_seconds必须在[1, 300]范围内")
        if not 0 < self.scene_change_ratio <= 0.5:
            raise ValueError(
                "gas_cylinder.scene_change_ratio必须在(0, 0.5]范围内"
            )
        if not 0 < self.motion_ratio <= 0.5:
            raise ValueError("gas_cylinder.motion_ratio必须在(0, 0.5]范围内")
        if not 0 < self.mask_nms_iou <= 1:
            raise ValueError("gas_cylinder.mask_nms_iou必须在(0, 1]范围内")
        if not 0 < self.temporal_match_iou <= 1:
            raise ValueError(
                "gas_cylinder.temporal_match_iou必须在(0, 1]范围内"
            )
        if not 1 <= self.large_count_change <= 20:
            raise ValueError(
                "gas_cylinder.large_count_change必须在[1, 20]范围内"
            )
        if not 1 <= self.alarm_threshold <= 1_000:
            raise ValueError(
                "gas_cylinder.alarm_threshold必须在[1, 1000]范围内"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "profile_id": self.profile_id,
            "analysis_fps": self.analysis_fps,
            "sample_count": self.sample_count,
            "sample_interval_seconds": self.sample_interval_seconds,
            "minimum_confirmations": self.minimum_confirmations,
            "scene_stable_seconds": self.scene_stable_seconds,
            "change_confirm_seconds": self.change_confirm_seconds,
            "forced_refresh_seconds": self.forced_refresh_seconds,
            "retry_seconds": self.retry_seconds,
            "scene_change_ratio": self.scene_change_ratio,
            "motion_ratio": self.motion_ratio,
            "mask_nms_iou": self.mask_nms_iou,
            "temporal_match_iou": self.temporal_match_iou,
            "large_count_change": self.large_count_change,
            "alarm_threshold": self.alarm_threshold,
            "display_ids": self.display_ids,
            "include_partial": self.include_partial,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any] | None,
    ) -> "GasCylinderOptions":
        options = cls(**dict(payload or {}))
        options.validate()
        return options


@dataclass(frozen=True, slots=True)
class GasPromptProfile:
    prompt_id: str
    boxes: tuple[tuple[float, float, float, float], ...]
    minimum_confidence: float = 0.03
    detection_roi: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True, slots=True)
class GasCylinderCameraProfile:
    profile_id: str
    reference_image: Path
    reference_width: int
    reference_height: int
    roi: tuple[tuple[float, float], ...]
    exclude_rois: tuple[tuple[tuple[float, float], ...], ...]
    prompts: tuple[GasPromptProfile, ...]

    @classmethod
    def load(
        cls,
        profile_root: Path,
        profile_id: str,
    ) -> "GasCylinderCameraProfile":
        if not _SAFE_PROFILE_ID.fullmatch(profile_id):
            raise ValueError("燃气瓶Profile ID格式无效")
        root = profile_root.expanduser().resolve()
        path = (root / f"{profile_id}.json").resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("燃气瓶Profile路径越界") from exc
        if not path.is_file():
            raise FileNotFoundError(f"燃气瓶Profile不存在: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != 1:
            raise ValueError("不支持的燃气瓶Profile版本")
        if str(payload.get("profile_id")) != profile_id:
            raise ValueError("燃气瓶Profile文件名与profile_id不一致")
        image = (path.parent / str(payload["reference_image"])).resolve()
        try:
            image.relative_to(path.parent)
        except ValueError as exc:
            raise ValueError("燃气瓶参考图路径越界") from exc
        if not image.is_file():
            raise FileNotFoundError(f"燃气瓶参考图不存在: {image}")
        size = payload.get("reference_size") or []
        if len(size) != 2 or int(size[0]) <= 0 or int(size[1]) <= 0:
            raise ValueError("燃气瓶Profile reference_size无效")
        roi = _validated_polygon(
            payload.get("roi")
            or [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "roi",
        )
        exclude_rois = tuple(
            _validated_polygon(item, "exclude_rois")
            for item in payload.get("exclude_rois", [])
        )
        prompts: list[GasPromptProfile] = []
        for item in payload.get("prompts", []):
            prompt_id = str(item.get("id", "")).strip()
            if not _SAFE_PROFILE_ID.fullmatch(prompt_id):
                raise ValueError("燃气瓶视觉提示ID格式无效")
            boxes = tuple(
                _validated_box(box, f"prompts.{prompt_id}.boxes")
                for box in item.get("boxes", [])
            )
            if not boxes:
                raise ValueError(f"燃气瓶视觉提示{prompt_id}没有参考框")
            confidence = float(item.get("minimum_confidence", 0.03))
            if not 0 < confidence <= 1:
                raise ValueError("燃气瓶视觉提示置信度无效")
            detection_roi = (
                _validated_polygon(item["detection_roi"], "detection_roi")
                if item.get("detection_roi") is not None
                else None
            )
            prompts.append(
                GasPromptProfile(
                    prompt_id=prompt_id,
                    boxes=boxes,
                    minimum_confidence=confidence,
                    detection_roi=detection_roi,
                )
            )
        if not prompts:
            raise ValueError("燃气瓶Profile至少需要一组视觉提示")
        return cls(
            profile_id=profile_id,
            reference_image=image,
            reference_width=int(size[0]),
            reference_height=int(size[1]),
            roi=roi,
            exclude_rois=exclude_rois,
            prompts=tuple(prompts),
        )


@dataclass(slots=True)
class GasCylinderCandidate:
    rectangle: NormalizedRect
    confidence: float
    prompt_id: str = ""
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    support: int = 1


@dataclass(frozen=True, slots=True)
class GasCylinderDetection:
    object_id: int
    rectangle: NormalizedRect
    confidence: float
    support: int


@dataclass(frozen=True, slots=True)
class GasCylinderSnapshot:
    state: str = "disabled"
    detections: tuple[GasCylinderDetection, ...] = ()
    result_version: int = 0
    updated_at_unix: float | None = None
    message: str = ""
    last_inference_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)


class GasCylinderResultCache:
    """Thread-safe stable results consumed by every output video frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[int, GasCylinderSnapshot] = {}
        self._next_ids: dict[int, int] = {}

    def snapshot(self, pad_index: int) -> GasCylinderSnapshot:
        with self._lock:
            return self._entries.get(pad_index, GasCylinderSnapshot())

    def mark_state(
        self,
        pad_index: int,
        state: str,
        message: str = "",
    ) -> GasCylinderSnapshot:
        with self._lock:
            previous = self._entries.get(pad_index, GasCylinderSnapshot())
            updated = GasCylinderSnapshot(
                state=state,
                detections=previous.detections,
                result_version=previous.result_version,
                updated_at_unix=previous.updated_at_unix,
                message=message,
                last_inference_ms=previous.last_inference_ms,
            )
            self._entries[pad_index] = updated
            return updated

    def store_snapshot(
        self,
        pad_index: int,
        snapshot: GasCylinderSnapshot,
    ) -> None:
        """Replace an entry with a result produced by an isolated process."""
        with self._lock:
            self._entries[pad_index] = snapshot
            self._next_ids[pad_index] = max(
                (item.object_id for item in snapshot.detections),
                default=0,
            ) + 1

    def publish(
        self,
        pad_index: int,
        candidates: list[GasCylinderCandidate],
        *,
        timestamp: float,
        inference_ms: float,
    ) -> GasCylinderSnapshot:
        with self._lock:
            previous = self._entries.get(pad_index, GasCylinderSnapshot())
            available = {item.object_id: item for item in previous.detections}
            used: set[int] = set()
            next_id = self._next_ids.get(
                pad_index,
                max(available, default=0) + 1,
            )
            detections: list[GasCylinderDetection] = []
            for candidate in sorted(
                candidates,
                key=lambda item: (item.rectangle.top, item.rectangle.left),
            ):
                best_id: int | None = None
                best_iou = 0.0
                for object_id, old in available.items():
                    if object_id in used:
                        continue
                    overlap = rectangle_iou(candidate.rectangle, old.rectangle)
                    if overlap >= 0.25 and overlap > best_iou:
                        best_id = object_id
                        best_iou = overlap
                if best_id is None:
                    best_id = next_id
                    next_id += 1
                used.add(best_id)
                detections.append(
                    GasCylinderDetection(
                        object_id=best_id,
                        rectangle=candidate.rectangle,
                        confidence=candidate.confidence,
                        support=candidate.support,
                    )
                )
            self._next_ids[pad_index] = next_id
            snapshot = GasCylinderSnapshot(
                state="stable",
                detections=tuple(detections),
                result_version=previous.result_version + 1,
                updated_at_unix=timestamp,
                message="",
                last_inference_ms=max(float(inference_ms), 0.0),
            )
            self._entries[pad_index] = snapshot
            return snapshot


def deduplicate_candidates(
    candidates: list[GasCylinderCandidate],
    *,
    iou_threshold: float,
) -> list[GasCylinderCandidate]:
    selected: list[GasCylinderCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: item.confidence,
        reverse=True,
    ):
        if any(
            candidate_iou(candidate, existing) >= iou_threshold
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return selected


def build_temporal_consensus(
    samples: list[list[GasCylinderCandidate]],
    *,
    minimum_confirmations: int,
    match_iou: float,
    nms_iou: float,
) -> list[GasCylinderCandidate]:
    clusters: list[list[tuple[int, GasCylinderCandidate]]] = []
    for sample_index, raw in enumerate(samples):
        candidates = deduplicate_candidates(raw, iou_threshold=nms_iou)
        used_clusters: set[int] = set()
        for candidate in sorted(
            candidates,
            key=lambda item: item.confidence,
            reverse=True,
        ):
            best_index: int | None = None
            best_overlap = 0.0
            for cluster_index, cluster in enumerate(clusters):
                if cluster_index in used_clusters:
                    continue
                overlap = max(
                    candidate_iou(candidate, existing)
                    for _index, existing in cluster
                )
                if overlap >= match_iou and overlap > best_overlap:
                    best_index = cluster_index
                    best_overlap = overlap
            if best_index is None:
                clusters.append([(sample_index, candidate)])
                used_clusters.add(len(clusters) - 1)
            else:
                clusters[best_index].append((sample_index, candidate))
                used_clusters.add(best_index)

    stable: list[GasCylinderCandidate] = []
    for cluster in clusters:
        support = len({sample_index for sample_index, _item in cluster})
        if support < minimum_confirmations:
            continue
        items = [item for _sample_index, item in cluster]
        coordinates = np.asarray(
            [
                [
                    item.rectangle.left,
                    item.rectangle.top,
                    item.rectangle.width,
                    item.rectangle.height,
                ]
                for item in items
            ],
            dtype=np.float32,
        )
        median = np.median(coordinates, axis=0)
        representative = max(items, key=lambda item: item.confidence)
        stable.append(
            GasCylinderCandidate(
                rectangle=NormalizedRect(*(float(value) for value in median)),
                confidence=float(np.mean([item.confidence for item in items])),
                prompt_id=representative.prompt_id,
                mask=representative.mask,
                support=support,
            )
        )
    return deduplicate_candidates(stable, iou_threshold=nms_iou)


def candidate_iou(
    first: GasCylinderCandidate,
    second: GasCylinderCandidate,
) -> float:
    if (
        first.mask is not None
        and second.mask is not None
        and first.mask.shape == second.mask.shape
    ):
        intersection = int(np.count_nonzero(first.mask & second.mask))
        union = int(np.count_nonzero(first.mask | second.mask))
        if union:
            return intersection / union
    return rectangle_iou(first.rectangle, second.rectangle)


def rectangle_iou(first: NormalizedRect, second: NormalizedRect) -> float:
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


class GasCylinderDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[GasCylinderCandidate]: ...


@dataclass(frozen=True, slots=True)
class SceneObservation:
    change_ratio: float
    motion_ratio: float


class StaticSceneMonitor:
    """Detect persistent layout changes without treating exposure as motion."""

    def __init__(
        self,
        profile: GasCylinderCameraProfile,
        *,
        width: int = 160,
        height: int = 90,
        pixel_threshold: float = 0.10,
    ) -> None:
        self.width = width
        self.height = height
        self.pixel_threshold = pixel_threshold
        self._mask = _polygon_mask(profile.roi, width, height)
        for polygon in profile.exclude_rois:
            self._mask &= ~_polygon_mask(polygon, width, height)
        if not self._mask.any():
            raise ValueError("燃气瓶场景监控ROI不包含有效像素")
        self._baseline: np.ndarray | None = None
        self._previous: np.ndarray | None = None

    def commit(self, frame: np.ndarray) -> None:
        gray = _normalized_gray(frame, self.width, self.height)
        self._baseline = gray.copy()
        self._previous = gray.copy()

    def observe(self, frame: np.ndarray) -> SceneObservation:
        gray = _normalized_gray(frame, self.width, self.height)
        if self._baseline is None:
            self.commit(frame)
            return SceneObservation(0.0, 0.0)
        change = self._difference_ratio(gray, self._baseline)
        motion = (
            self._difference_ratio(gray, self._previous)
            if self._previous is not None
            else 0.0
        )
        self._previous = gray
        return SceneObservation(change, motion)

    def _difference_ratio(
        self,
        current: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        signed = current - reference
        illumination = float(np.median(signed[self._mask]))
        changed = (np.abs(signed - illumination) >= self.pixel_threshold) & self._mask
        changed = _remove_isolated_pixels(changed)
        return int(np.count_nonzero(changed)) / max(
            int(np.count_nonzero(self._mask)),
            1,
        )


class GasCylinderCoordinator:
    """State machine that keeps expensive inference off the video path."""

    def __init__(
        self,
        *,
        pad_index: int,
        options: GasCylinderOptions,
        profile: GasCylinderCameraProfile,
        detector: GasCylinderDetector,
        cache: GasCylinderResultCache,
    ) -> None:
        options.validate()
        self.pad_index = pad_index
        self.options = options
        self.profile = profile
        self.detector = detector
        self.cache = cache
        self.monitor = StaticSceneMonitor(profile)
        self._phase = "sampling"
        self._samples: list[list[GasCylinderCandidate]] = []
        self._next_sample_at = 0.0
        self._last_refresh_at = 0.0
        self._changed_since: float | None = None
        self._quiet_since: float | None = None
        self._retry_at = 0.0
        self._pending_large_change: list[GasCylinderCandidate] | None = None
        self._sampling_reason = "initial"
        self._last_frame: np.ndarray | None = None
        self.cache.mark_state(pad_index, "sampling", "燃气瓶识别中")

    def process(self, frame: np.ndarray, *, timestamp: float | None = None) -> None:
        now = time.monotonic() if timestamp is None else float(timestamp)
        self._last_frame = frame
        if self._phase == "error":
            if now < self._retry_at:
                return
            self._begin_sampling(now, "retry")
        if self._phase == "sampling":
            if now < self._next_sample_at:
                return
            started = time.perf_counter()
            try:
                detections = self.detector.detect(frame)
            except Exception as exc:
                self._phase = "error"
                self._retry_at = now + self.options.retry_seconds
                self.cache.mark_state(
                    self.pad_index,
                    "error",
                    f"燃气瓶识别失败: {type(exc).__name__}",
                )
                raise
            self._samples.append(detections)
            self._next_sample_at = now + self.options.sample_interval_seconds
            if len(self._samples) >= self.options.sample_count:
                duration_ms = (time.perf_counter() - started) * 1_000
                self._finish_sampling(now, duration_ms)
            return

        observation = self.monitor.observe(frame)
        if self._phase == "stable":
            if now - self._last_refresh_at >= self.options.forced_refresh_seconds:
                self._begin_sampling(now, "periodic")
                return
            if observation.change_ratio >= self.options.scene_change_ratio:
                if self._changed_since is None:
                    self._changed_since = now
                elif now - self._changed_since >= self.options.change_confirm_seconds:
                    self._phase = "dirty"
                    self._quiet_since = None
                    self.cache.mark_state(
                        self.pad_index,
                        "dirty",
                        "场景变化，等待画面稳定",
                    )
            else:
                self._changed_since = None
            return

        if self._phase == "dirty":
            if observation.motion_ratio <= self.options.motion_ratio:
                if self._quiet_since is None:
                    self._quiet_since = now
                elif now - self._quiet_since >= self.options.scene_stable_seconds:
                    self._begin_sampling(now, "scene_changed")
            else:
                self._quiet_since = None

    def _begin_sampling(self, timestamp: float, reason: str) -> None:
        self._phase = "sampling"
        self._samples = []
        self._next_sample_at = timestamp
        self._sampling_reason = reason
        self.cache.mark_state(self.pad_index, "sampling", "燃气瓶识别中")

    def _finish_sampling(self, timestamp: float, inference_ms: float) -> None:
        candidates = build_temporal_consensus(
            self._samples,
            minimum_confirmations=self.options.minimum_confirmations,
            match_iou=self.options.temporal_match_iou,
            nms_iou=self.options.mask_nms_iou,
        )
        previous = self.cache.snapshot(self.pad_index)
        if (
            previous.result_version > 0
            and abs(len(candidates) - previous.count)
            > self.options.large_count_change
        ):
            if self._pending_large_change is None:
                self._pending_large_change = candidates
                self._begin_sampling(timestamp, "large_change_confirmation")
                return
            if abs(len(candidates) - len(self._pending_large_change)) > 1:
                self._pending_large_change = candidates
                self._begin_sampling(timestamp, "large_change_confirmation")
                return
        self._pending_large_change = None
        self.cache.publish(
            self.pad_index,
            candidates,
            timestamp=timestamp,
            inference_ms=inference_ms,
        )
        self.monitor.commit(self._last_frame)  # type: ignore[arg-type]
        self._phase = "stable"
        self._last_refresh_at = timestamp
        self._changed_since = None
        self._quiet_since = None
        self._samples = []


class UltralyticsYoloeGasCylinderDetector:
    """One YOLOE model with cached visual embeddings for all prompt profiles."""

    def __init__(
        self,
        *,
        model_path: Path,
        profile: GasCylinderCameraProfile,
        device: str = "cuda:0",
        half: bool = True,
        imgsz: int = 1280,
    ) -> None:
        from PIL import Image
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        if not model_path.is_file():
            raise FileNotFoundError(f"YOLOE燃气瓶模型不存在: {model_path}")
        self.profile = profile
        self.device = device
        self.half = half
        self.imgsz = imgsz
        self._lock = threading.Lock()
        self._model = YOLOE(str(model_path), task="segment")
        reference_rgb = np.asarray(
            Image.open(profile.reference_image).convert("RGB")
        )
        if reference_rgb.shape[:2] != (
            profile.reference_height,
            profile.reference_width,
        ):
            raise ValueError("燃气瓶参考图尺寸与Profile不一致")
        self._reference_bgr = np.ascontiguousarray(reference_rgb[..., ::-1])
        self._embeddings: dict[str, Any] = {}
        for prompt in profile.prompts:
            boxes = np.asarray(
                [
                    [
                        box[0] * profile.reference_width,
                        box[1] * profile.reference_height,
                        box[2] * profile.reference_width,
                        box[3] * profile.reference_height,
                    ]
                    for box in prompt.boxes
                ],
                dtype=np.float32,
            )
            predictor = YOLOEVPSegPredictor(
                overrides={
                    "task": "segment",
                    "mode": "predict",
                    "save": False,
                    "verbose": False,
                    "batch": 1,
                    "device": device,
                    "imgsz": imgsz,
                    "half": half,
                }
            )
            predictor.set_prompts(
                {
                    "bboxes": boxes,
                    "cls": np.zeros(len(boxes), dtype=np.int64),
                }
            )
            predictor.setup_model(model=self._model.model, verbose=False)
            embedding = predictor.get_vpe(self._reference_bgr)
            self._embeddings[prompt.prompt_id] = embedding.detach().cpu()
            self._model.predictor = None

    def detect(self, frame: np.ndarray) -> list[GasCylinderCandidate]:
        bgr = _as_bgr(frame)
        height, width = bgr.shape[:2]
        candidates: list[GasCylinderCandidate] = []
        with self._lock:
            device = next(self._model.model.parameters()).device
            for prompt in self.profile.prompts:
                embedding = self._embeddings[prompt.prompt_id].to(device)
                self._model.model.set_classes(["gas cylinder"], embedding)
                results = self._model.predict(
                    source=bgr,
                    imgsz=self.imgsz,
                    conf=prompt.minimum_confidence,
                    iou=0.50,
                    max_det=100,
                    agnostic_nms=True,
                    retina_masks=True,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )
                if not results:
                    continue
                result = results[0]
                if result.boxes is None:
                    continue
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                masks = (
                    result.masks.data.detach().cpu().numpy() > 0.5
                    if result.masks is not None
                    else None
                )
                for index, (box, score) in enumerate(zip(boxes, scores)):
                    rectangle = NormalizedRect(
                        max(float(box[0]) / width, 0.0),
                        max(float(box[1]) / height, 0.0),
                        min(float(box[2] - box[0]) / width, 1.0),
                        min(float(box[3] - box[1]) / height, 1.0),
                    )
                    if not self._allowed(rectangle, prompt):
                        continue
                    mask = (
                        _compact_mask(masks[index])
                        if masks is not None and index < len(masks)
                        else None
                    )
                    candidates.append(
                        GasCylinderCandidate(
                            rectangle=rectangle,
                            confidence=float(score),
                            prompt_id=prompt.prompt_id,
                            mask=mask,
                        )
                    )
        return deduplicate_candidates(candidates, iou_threshold=0.50)

    def _allowed(
        self,
        rectangle: NormalizedRect,
        prompt: GasPromptProfile,
    ) -> bool:
        center = rectangle.center
        if not _point_in_polygon(center, self.profile.roi):
            return False
        if any(
            _point_in_polygon(center, polygon)
            for polygon in self.profile.exclude_rois
        ):
            return False
        return prompt.detection_roi is None or _point_in_polygon(
            center,
            prompt.detection_roi,
        )


def _validated_box(
    value: Any,
    name: str,
) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name}必须是[x1,y1,x2,y2]")
    x1, y1, x2, y2 = (float(item) for item in value)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"{name}坐标必须是递增的[0,1]归一化值")
    return x1, y1, x2, y2


def _validated_polygon(
    value: Any,
    name: str,
) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{name}至少需要3个点")
    points = tuple((float(item[0]), float(item[1])) for item in value)
    if any(not 0 <= x <= 1 or not 0 <= y <= 1 for x, y in points):
        raise ValueError(f"{name}坐标必须在[0,1]范围内")
    return points


def _point_in_polygon(
    point: tuple[float, float],
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
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


def _polygon_mask(
    polygon: tuple[tuple[float, float], ...],
    width: int,
    height: int,
) -> np.ndarray:
    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    grid_x, grid_y = np.meshgrid(x, y)
    inside = np.zeros((height, width), dtype=bool)
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        crossing = (current_y > grid_y) != (previous_y > grid_y)
        denominator = previous_y - current_y
        if abs(denominator) < 1e-12:
            denominator = 1e-12
        x_at_y = (
            (previous_x - current_x)
            * (grid_y - current_y)
            / denominator
            + current_x
        )
        inside ^= crossing & (grid_x < x_at_y)
        previous_x, previous_y = current_x, current_y
    return inside


def _normalized_gray(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3 and value.shape[0] in {1, 3, 4}:
        value = np.moveaxis(value, 0, -1)
    if value.ndim == 3:
        value = value[..., :3].astype(np.float32).mean(axis=-1)
    elif value.ndim != 2:
        raise ValueError(f"不支持的燃气瓶分析帧形状: {value.shape}")
    value = value.astype(np.float32, copy=False)
    if float(value.max(initial=0.0)) > 1.5:
        value = value / 255.0
    y_index = np.linspace(0, value.shape[0] - 1, height).astype(np.intp)
    x_index = np.linspace(0, value.shape[1] - 1, width).astype(np.intp)
    return value[np.ix_(y_index, x_index)]


def _remove_isolated_pixels(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1)
    neighbours = np.zeros_like(mask, dtype=np.uint8)
    for y_offset in range(3):
        for x_offset in range(3):
            neighbours += padded[
                y_offset : y_offset + mask.shape[0],
                x_offset : x_offset + mask.shape[1],
            ]
    return mask & (neighbours >= 4)


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3 and value.shape[0] in {3, 4}:
        value = np.moveaxis(value, 0, -1)
    if value.ndim != 3 or value.shape[-1] < 3:
        raise ValueError(f"YOLOE输入帧形状无效: {value.shape}")
    # The DeepStream side branch is explicitly RGB; Ultralytics NumPy input
    # follows OpenCV's BGR convention.
    return np.ascontiguousarray(value[..., :3][..., ::-1])


def _compact_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[-2:]
    stride = max(1, math.ceil(max(height / 180, width / 320)))
    return np.asarray(mask[::stride, ::stride], dtype=bool)
