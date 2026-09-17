"""Lossy side process that runs the ground-litter model off the main RTSP path.

The main DeepStream pipeline copies one native-resolution frame every
``1 / analysis_fps`` seconds into this process.  Everything expensive (tiled
inference, actor overlap checks, display smoothing) happens here, so a slow
litter model can never back-pressure decode, OSD or NVENC.
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .event_engine import NormalizedRect

from .ground_litter_detection import (
    GroundLitterDetectionOptions,
    GroundLitterDisplayTracker,
    GroundLitterResultCache,
    GroundLitterSnapshot,
    UltralyticsGroundLitterDetector,
    as_bgr,
    build_ground_litter_tiles,
)


LOGGER = logging.getLogger("rtsp_annotator.ground_litter_process")


@dataclass(frozen=True, slots=True)
class GroundLitterProcessConfig:
    model_path: Path
    device: str
    half: bool
    actor_model_path: Path | None
    options_by_pad: dict[int, GroundLitterDetectionOptions]


def _put_latest(output_queue: Any, item: Any) -> None:
    """Keep only the newest snapshot; an old box is worse than no box."""
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


def _run_ground_litter_process(
    config: GroundLitterProcessConfig,
    input_queue: Any,
    output_queue: Any,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    try:
        detector = UltralyticsGroundLitterDetector(
            model_path=config.model_path,
            device=config.device,
            half=config.half,
            actor_model_path=config.actor_model_path,
        )
        trackers = {
            pad_index: GroundLitterDisplayTracker(options)
            for pad_index, options in config.options_by_pad.items()
        }
        counters = {
            pad_index: {"analyzed_frames": 0}
            for pad_index in config.options_by_pad
        }
        views: dict[int, tuple[Any, Any, tuple[int, int]]] = {}
    except Exception as exc:
        LOGGER.exception("零散垃圾独立推理进程初始化失败")
        for pad_index in config.options_by_pad:
            _put_latest(
                output_queue,
                (
                    pad_index,
                    GroundLitterSnapshot(
                        state="error",
                        message=(
                            "零散垃圾检测组件初始化失败: "
                            f"{type(exc).__name__}"
                        ),
                    ),
                ),
            )
        return
    for pad_index, options in config.options_by_pad.items():
        _put_latest(
            output_queue,
            (
                pad_index,
                GroundLitterSnapshot(
                    state="starting",
                    message=(
                        "零散垃圾识别启动中"
                        if not options.enabled
                        else f"零散垃圾识别启动中({len(options.zones)}区)"
                    ),
                ),
            ),
        )
    while True:
        try:
            item = input_queue.get()
        except (EOFError, OSError):  # pragma: no cover - parent died
            return
        if item is None:
            return
        pad_index, frame, timestamp, night, actors = item
        pad_index = int(pad_index)
        options = config.options_by_pad.get(pad_index)
        if options is None:
            continue
        try:
            bgr = as_bgr(frame)
            height, width = bgr.shape[:2]
            view = views.get(pad_index)
            if view is None or view[2] != (width, height):
                masks, tiles = build_ground_litter_tiles(
                    options,
                    width,
                    height,
                )
                trackers[pad_index].reset()
                view = views[pad_index] = (masks, tiles, (width, height))
                LOGGER.info(
                    "零散垃圾分块就绪: pad=%d 分辨率=%dx%d 分块=%d 区域=%d",
                    pad_index,
                    width,
                    height,
                    len(tiles),
                    len(options.zones),
                )
            masks, tiles, _size = view
            # The main chain sends normalized [left, top, width, height]; the
            # detector compares pixel boxes, so convert against this frame.
            actor_boxes = [
                [
                    float(box[0]) * width,
                    float(box[1]) * height,
                    (float(box[0]) + float(box[2])) * width,
                    (float(box[1]) + float(box[3])) * height,
                ]
                for box in actors
                if len(box) >= 4
            ]
            started = time.perf_counter()
            if detector.has_actor_model:
                actor_boxes.extend(detector.actor_boxes(bgr, options))
            candidates, stats = detector.candidates(
                bgr,
                options,
                masks=masks,
                tiles=tiles,
                night=bool(night),
                actors=actor_boxes,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            counters[pad_index]["analyzed_frames"] += 1
            snapshot = trackers[pad_index].update(
                candidates,
                timestamp=float(timestamp),
                occluders=[
                    NormalizedRect(a / width, b / height, (c - a) / width, (d - b) / height)
                    for a, b, c, d in getattr(detector, "last_actor_boxes", actor_boxes)
                ],
            )
            _put_latest(
                output_queue,
                (
                    pad_index,
                    replace(
                        snapshot,
                        last_inference_ms=round(elapsed_ms, 2),
                        analyzed_frames=counters[pad_index][
                            "analyzed_frames"
                        ],
                        raw_candidates=int(stats["raw_candidates"]),
                        rejected_roi=int(stats["rejected_roi"]),
                        rejected_actor=int(stats["rejected_actor"]),
                        tile_count=len(tiles),
                    ),
                ),
            )
        except Exception as exc:
            LOGGER.exception(
                "零散垃圾旁路分析失败，主RTSP继续运行: pad=%d",
                pad_index,
            )
            _put_latest(
                output_queue,
                (
                    pad_index,
                    GroundLitterSnapshot(
                        state="error",
                        message=(
                            "零散垃圾分析异常: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    ),
                ),
            )


class GroundLitterProcessClient:
    """Frame submission and result collection for one DeepStream group."""

    def __init__(
        self,
        config: GroundLitterProcessConfig,
        cache: GroundLitterResultCache,
    ) -> None:
        self._cache = cache
        self._options_by_pad = dict(config.options_by_pad)
        self._next_due = {
            pad_index: 0.0 for pad_index in self._options_by_pad
        }
        context = multiprocessing.get_context("spawn")
        queue_size = max(len(self._options_by_pad) * 2, 2)
        self._input_queue = context.Queue(maxsize=queue_size)
        self._output_queue = context.Queue(maxsize=max(queue_size * 2, 4))
        self._process = context.Process(
            target=_run_ground_litter_process,
            args=(config, self._input_queue, self._output_queue),
            name="ground-litter-inference",
            daemon=True,
        )
        for pad_index in self._options_by_pad:
            cache.mark_state(
                pad_index,
                "starting",
                "零散垃圾识别启动中",
            )
        self._process.start()

    @property
    def enabled_pads(self) -> tuple[int, ...]:
        return tuple(
            pad_index
            for pad_index, options in self._options_by_pad.items()
            if options.enabled
        )

    def accepts(
        self,
        pad_index: int,
        *,
        timestamp: float | None = None,
    ) -> bool:
        """Rate limit and process health check before the device copy."""
        self.drain_results()
        options = self._options_by_pad.get(pad_index)
        if options is None or not options.enabled:
            return False
        if not self._process.is_alive():
            current = self._cache.snapshot(pad_index)
            if current.state != "error":
                self._cache.mark_state(
                    pad_index,
                    "error",
                    "零散垃圾独立推理进程已退出",
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
        night: bool = False,
        actors: Any = (),
    ) -> bool:
        if pad_index not in self._options_by_pad:
            return False
        try:
            self._input_queue.put_nowait(
                (
                    pad_index,
                    frame,
                    float(timestamp),
                    bool(night),
                    list(actors),
                )
            )
        except queue.Full:
            # A queued old frame is less useful than the next fresh one.
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
