from __future__ import annotations

import logging
import multiprocessing
import queue
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .vessel_detection import (
    EvidenceValidationResult,
    UltralyticsVesselDetector,
    TemporalSmallTargetProposer,
    VesselDetectionOptions,
    VesselDetection,
    VesselResultCache,
    VesselSnapshot,
    VesselTrackManager,
    deduplicate_candidates,
)


LOGGER = logging.getLogger("rtsp_annotator.vessel_process")


@dataclass(frozen=True, slots=True)
class VesselDetectionProcessConfig:
    model_path: Path
    device: str
    half: bool
    options_by_pad: dict[int, VesselDetectionOptions]


_CLOSEUP_PROPOSAL_ROI = (
    (0.05, 0.05),
    (0.95, 0.05),
    (0.95, 0.95),
    (0.05, 0.95),
)


def _verification_options(
    options: VesselDetectionOptions,
) -> VesselDetectionOptions:
    """Derive a PTZ close-up profile without mutating HOME-view policy."""
    return replace(
        options,
        inference_regions=((0.0, 0.0, 1.0, 1.0),),
        roi=None,
        exclude_rois=(),
        proposal_roi=_CLOSEUP_PROPOSAL_ROI,
        # Camera motion invalidates the HOME-view temporal background. In
        # close-up mode, allow compact appearance proposals to bridge the
        # first zoom round until the boat classifier becomes confident.
        proposal_minimum_motion_ratio=0.0,
        display_roi=False,
    )


def _put_latest(
    output_queue: Any,
    item: tuple[int, VesselSnapshot],
) -> None:
    try:
        output_queue.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        output_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        output_queue.put_nowait(item)
    except queue.Full:
        pass


def _validate_evidence_image(
    detector: UltralyticsVesselDetector,
    content: bytes,
    options: VesselDetectionOptions,
) -> EvidenceValidationResult:
    try:
        import cv2

        encoded = np.frombuffer(content, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return EvidenceValidationResult(
                state="error",
                message="证据JPEG无法解码",
            )
        candidates = detector.detect(frame, _verification_options(options))
        detections: list[VesselDetection] = []
        sharpness: list[tuple[int, float]] = []
        height, width = frame.shape[:2]
        for object_id, candidate in enumerate(candidates, start=1):
            rectangle = candidate.rectangle
            detections.append(
                VesselDetection(
                    object_id=object_id,
                    rectangle=rectangle,
                    confidence=candidate.confidence,
                    class_id=candidate.class_id,
                    hits=1,
                )
            )
            left = min(max(int(rectangle.left * width), 0), width - 1)
            top = min(max(int(rectangle.top * height), 0), height - 1)
            right = min(
                max(
                    int((rectangle.left + rectangle.width) * width),
                    left + 1,
                ),
                width,
            )
            bottom = min(
                max(
                    int((rectangle.top + rectangle.height) * height),
                    top + 1,
                ),
                height,
            )
            crop = frame[top:bottom, left:right]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness.append(
                (
                    object_id,
                    float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                )
            )
        return EvidenceValidationResult(
            state="running",
            detections=tuple(detections),
            sharpness_by_object_id=tuple(sharpness),
        )
    except Exception as exc:
        return EvidenceValidationResult(
            state="error",
            message=f"证据复检失败: {type(exc).__name__}: {exc}",
        )


def _run_vessel_detection_process(
    config: VesselDetectionProcessConfig,
    input_queue: Any,
    output_queue: Any,
    view_queue: Any,
    evidence_queue: Any,
    evidence_output_queue: Any,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    try:
        detector = UltralyticsVesselDetector(
            model_path=config.model_path,
            device=config.device,
            half=config.half,
        )
        trackers = {
            pad_index: VesselTrackManager(options)
            for pad_index, options in config.options_by_pad.items()
        }
        proposers = {
            pad_index: TemporalSmallTargetProposer()
            for pad_index, options in config.options_by_pad.items()
            if options.small_target_proposals
        }
        verification_modes = {
            pad_index: False for pad_index in config.options_by_pad
        }
        view_generations = {
            pad_index: 0 for pad_index in config.options_by_pad
        }
    except Exception as exc:
        LOGGER.exception("船舶独立推理进程初始化失败")
        for pad_index in config.options_by_pad:
            _put_latest(
                output_queue,
                (
                    pad_index,
                    VesselSnapshot(
                        state="error",
                        message=(
                            "船舶检测组件初始化失败: "
                            f"{type(exc).__name__}"
                        ),
                    ),
                ),
            )
        return

    while True:
        try:
            evidence_item = evidence_queue.get_nowait()
        except queue.Empty:
            evidence_item = None
        if evidence_item is not None:
            request_id, evidence_pad, content = evidence_item
            evidence_pad = int(evidence_pad)
            evidence_options = config.options_by_pad.get(evidence_pad)
            if evidence_options is None:
                validation = EvidenceValidationResult(
                    state="error",
                    message=f"未知船舶推理pad: {evidence_pad}",
                )
            else:
                validation = _validate_evidence_image(
                    detector,
                    bytes(content),
                    evidence_options,
                )
            evidence_output_queue.put((request_id, validation))
            continue
        try:
            item = input_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        if item is None:
            return
        while True:
            try:
                changed_pad, changed_generation = view_queue.get_nowait()
            except queue.Empty:
                break
            changed_pad = int(changed_pad)
            changed_generation = int(changed_generation)
            if (
                changed_pad in view_generations
                and changed_generation > view_generations[changed_pad]
            ):
                view_generations[changed_pad] = changed_generation
                trackers[changed_pad].reset_tracking()
                if config.options_by_pad[changed_pad].small_target_proposals:
                    proposers[changed_pad] = TemporalSmallTargetProposer()
        if len(item) == 5:
            (
                pad_index,
                frame,
                timestamp,
                verification_active,
                view_generation,
            ) = item
        elif len(item) == 4:
            pad_index, frame, timestamp, verification_active = item
            view_generation = 0
        else:
            # Compatibility with an already queued pre-update frame.
            pad_index, frame, timestamp = item
            verification_active = False
            view_generation = 0
        pad_index = int(pad_index)
        home_options = config.options_by_pad.get(pad_index)
        tracker = trackers.get(pad_index)
        if home_options is None or tracker is None:
            continue
        view_generation = int(view_generation)
        if view_generation < view_generations[pad_index]:
            # This frame was queued before the most recent physical camera
            # movement. Never infer or publish its view-relative tracks.
            continue
        if view_generation > view_generations[pad_index]:
            view_generations[pad_index] = view_generation
            tracker.reset_tracking()
            if home_options.small_target_proposals:
                proposers[pad_index] = TemporalSmallTargetProposer()
        verification_active = bool(verification_active)
        if verification_modes[pad_index] != verification_active:
            verification_modes[pad_index] = verification_active
            tracker.reset_tracking()
            if home_options.small_target_proposals:
                proposers[pad_index] = TemporalSmallTargetProposer()
        options = (
            _verification_options(home_options)
            if verification_active
            else home_options
        )
        started = time.perf_counter()
        try:
            candidates = detector.detect(np.asarray(frame), options)
            proposer = proposers.get(pad_index)
            if proposer is not None:
                candidates.extend(proposer.detect(np.asarray(frame), options))
                candidates = deduplicate_candidates(
                    candidates,
                    iou_threshold=options.iou,
                    containment_threshold=(
                        options.duplicate_containment_threshold
                    ),
                    limit=options.maximum_detections,
                )
            inference_ms = (time.perf_counter() - started) * 1_000.0
            snapshot = tracker.update(
                candidates,
                timestamp=float(timestamp),
                inference_ms=inference_ms,
            )
        except Exception as exc:
            LOGGER.exception("船舶独立推理失败: pad=%d", pad_index)
            snapshot = VesselSnapshot(
                state="error",
                updated_at=float(timestamp),
                message=f"船舶检测失败: {type(exc).__name__}",
                last_inference_ms=(
                    time.perf_counter() - started
                )
                * 1_000.0,
            )
        _put_latest(output_queue, (
            pad_index, replace(snapshot, view_generation=view_generation),
        ))


class VesselDetectionProcessClient:
    """Non-blocking bridge from DeepStream to vessel inference."""

    def __init__(
        self,
        config: VesselDetectionProcessConfig,
        cache: VesselResultCache,
    ) -> None:
        self._cache = cache
        self._options_by_pad = dict(config.options_by_pad)
        self._next_due = {pad_index: 0.0 for pad_index in self._options_by_pad}
        context = multiprocessing.get_context("spawn")
        queue_size = max(len(self._options_by_pad) * 2, 2)
        self._input_queue = context.Queue(maxsize=queue_size)
        self._output_queue = context.Queue(maxsize=max(queue_size * 2, 4))
        self._view_queue = context.Queue(maxsize=max(queue_size * 2, 4))
        self._evidence_queue = context.Queue(maxsize=max(queue_size, 2))
        self._evidence_output_queue = context.Queue(
            maxsize=max(queue_size * 2, 4)
        )
        self._last_view_generation = {
            pad_index: -1 for pad_index in self._options_by_pad
        }
        self._process = context.Process(
            target=_run_vessel_detection_process,
            args=(
                config,
                self._input_queue,
                self._output_queue,
                self._view_queue,
                self._evidence_queue,
                self._evidence_output_queue,
            ),
            name="vessel-inference",
            daemon=True,
        )
        for pad_index in self._options_by_pad:
            cache.mark_state(pad_index, "starting", "船舶检测启动中")
        self._process.start()
        self._validation_condition = threading.Condition()
        self._validation_results: dict[str, EvidenceValidationResult] = {}
        self._validation_stop = threading.Event()
        self._validation_thread = threading.Thread(
            target=self._drain_validation_results,
            name="vessel-evidence-results",
            daemon=True,
        )
        self._validation_thread.start()

    def _drain_validation_results(self) -> None:
        while not self._validation_stop.is_set():
            try:
                request_id, result = self._evidence_output_queue.get(
                    timeout=0.1
                )
            except queue.Empty:
                continue
            with self._validation_condition:
                self._validation_results[str(request_id)] = result
                self._validation_condition.notify_all()

    def validate_evidence(
        self,
        pad_index: int,
        content: bytes,
        *,
        timeout: float,
    ) -> EvidenceValidationResult:
        if pad_index not in self._options_by_pad:
            raise KeyError(pad_index)
        if not self._process.is_alive():
            raise RuntimeError("船舶推理进程已退出")
        request_id = uuid.uuid4().hex
        try:
            self._evidence_queue.put(
                (request_id, pad_index, bytes(content)),
                timeout=min(max(timeout, 0.1), 1.0),
            )
        except queue.Full as exc:
            raise RuntimeError("证据复检队列已满") from exc
        deadline = time.monotonic() + timeout
        with self._validation_condition:
            while request_id not in self._validation_results:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("证据图片复检超时")
                self._validation_condition.wait(timeout=remaining)
            return self._validation_results.pop(request_id)

    def accepts(
        self,
        pad_index: int,
        *,
        timestamp: float | None = None,
    ) -> bool:
        """Check rate/process state before the device-to-host frame copy."""
        self.drain_results()
        options = self._options_by_pad.get(pad_index)
        if options is None:
            return False
        if not self._process.is_alive():
            current = self._cache.snapshot(pad_index)
            if current.state != "error":
                self._cache.mark_state(
                    pad_index,
                    "error",
                    "船舶独立推理进程已退出",
                )
            return False
        now = time.monotonic() if timestamp is None else float(timestamp)
        if now < self._next_due[pad_index]:
            return False
        self._next_due[pad_index] = now + 1.0 / options.analysis_fps
        return True

    def submit(
        self,
        pad_index: int,
        frame: np.ndarray,
        *,
        timestamp: float,
        verification_active: bool = False,
        view_generation: int = 0,
    ) -> bool:
        if pad_index not in self._options_by_pad:
            return False
        view_generation = int(view_generation)
        if self._last_view_generation[pad_index] != view_generation:
            view_event_enqueued = False
            try:
                self._view_queue.put_nowait(
                    (pad_index, view_generation)
                )
                view_event_enqueued = True
            except queue.Full:
                try:
                    self._view_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._view_queue.put_nowait(
                        (pad_index, view_generation)
                    )
                    view_event_enqueued = True
                except queue.Full:
                    pass
            if not view_event_enqueued:
                # Retry the generation event on the next frame. Advancing the
                # local marker here would permanently lose the reset signal
                # and allow already queued frames from the previous camera
                # view to publish stale detections.
                return False
            self._last_view_generation[pad_index] = view_generation
        try:
            self._input_queue.put_nowait(
                (
                    pad_index,
                    frame,
                    timestamp,
                    bool(verification_active),
                    view_generation,
                )
            )
        except queue.Full:
            return False
        return True

    def drain_results(self) -> None:
        while True:
            try:
                pad_index, snapshot = self._output_queue.get_nowait()
            except queue.Empty:
                return
            pad_index = int(pad_index)
            if snapshot.view_generation != self._last_view_generation.get(pad_index, 0):
                continue
            self._cache.store_snapshot(pad_index, snapshot)

    def shutdown(self, timeout: float = 5.0) -> None:
        self.drain_results()
        if self._process.is_alive():
            try:
                self._input_queue.put_nowait(None)
            except queue.Full:
                try:
                    self._input_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._input_queue.put_nowait(None)
                except queue.Full:
                    pass
            self._process.join(timeout=max(timeout, 0.0))
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._validation_stop.set()
        self._validation_thread.join(timeout=0.5)
        self._input_queue.close()
        self._output_queue.close()
        self._view_queue.close()
        self._evidence_queue.close()
        self._evidence_output_queue.close()
