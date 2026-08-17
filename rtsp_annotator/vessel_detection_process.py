from __future__ import annotations

import logging
import multiprocessing
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .vessel_detection import (
    UltralyticsVesselDetector,
    VesselDetectionOptions,
    VesselResultCache,
    VesselSnapshot,
    VesselTrackManager,
)


LOGGER = logging.getLogger("rtsp_annotator.vessel_process")


@dataclass(frozen=True, slots=True)
class VesselDetectionProcessConfig:
    model_path: Path
    device: str
    half: bool
    options_by_pad: dict[int, VesselDetectionOptions]


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


def _run_vessel_detection_process(
    config: VesselDetectionProcessConfig,
    input_queue: Any,
    output_queue: Any,
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
        item = input_queue.get()
        if item is None:
            return
        pad_index, frame, timestamp = item
        pad_index = int(pad_index)
        options = config.options_by_pad.get(pad_index)
        tracker = trackers.get(pad_index)
        if options is None or tracker is None:
            continue
        started = time.perf_counter()
        try:
            candidates = detector.detect(np.asarray(frame), options)
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
        _put_latest(output_queue, (pad_index, snapshot))


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
        self._process = context.Process(
            target=_run_vessel_detection_process,
            args=(config, self._input_queue, self._output_queue),
            name="vessel-inference",
            daemon=True,
        )
        for pad_index in self._options_by_pad:
            cache.mark_state(pad_index, "starting", "船舶检测启动中")
        self._process.start()

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
    ) -> bool:
        if pad_index not in self._options_by_pad:
            return False
        try:
            self._input_queue.put_nowait((pad_index, frame, timestamp))
        except queue.Full:
            return False
        return True

    def drain_results(self) -> None:
        while True:
            try:
                pad_index, snapshot = self._output_queue.get_nowait()
            except queue.Empty:
                return
            self._cache.store_snapshot(int(pad_index), snapshot)

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
        self._input_queue.close()
        self._output_queue.close()
