from __future__ import annotations

import logging
import multiprocessing
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gas_cylinder import (
    GasCylinderCameraProfile,
    GasCylinderCoordinator,
    GasCylinderOptions,
    GasCylinderResultCache,
    GasCylinderSnapshot,
    UltralyticsYoloeGasCylinderDetector,
)


LOGGER = logging.getLogger("rtsp_annotator.gas_process")


@dataclass(frozen=True, slots=True)
class GasCylinderProcessConfig:
    model_path: Path
    profile_root: Path
    profile_id: str
    device: str
    imgsz: int
    options_by_pad: dict[int, GasCylinderOptions]


def _put_latest(output_queue: Any, item: tuple[int, GasCylinderSnapshot]) -> None:
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


def _run_gas_cylinder_process(
    config: GasCylinderProcessConfig,
    input_queue: Any,
    output_queue: Any,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    cache = GasCylinderResultCache()
    try:
        profile = GasCylinderCameraProfile.load(
            config.profile_root,
            config.profile_id,
        )
        detector = UltralyticsYoloeGasCylinderDetector(
            model_path=config.model_path,
            profile=profile,
            device=config.device,
            half=False,
            imgsz=config.imgsz,
        )
        coordinators = {
            pad_index: GasCylinderCoordinator(
                pad_index=pad_index,
                options=options,
                profile=profile,
                detector=detector,
                cache=cache,
            )
            for pad_index, options in config.options_by_pad.items()
        }
    except Exception as exc:
        LOGGER.exception("燃气瓶独立推理进程初始化失败")
        for pad_index in config.options_by_pad:
            snapshot = GasCylinderSnapshot(
                state="error",
                message=f"燃气瓶组件初始化失败: {type(exc).__name__}",
            )
            _put_latest(output_queue, (pad_index, snapshot))
        return

    completed_pads: set[int] = set()
    while True:
        item = input_queue.get()
        if item is None:
            return
        pad_index, frame, timestamp = item
        coordinator = coordinators.get(int(pad_index))
        if coordinator is None:
            continue
        try:
            coordinator.process(
                np.asarray(frame),
                timestamp=float(timestamp),
            )
        except Exception:
            LOGGER.exception(
                "燃气瓶独立推理失败: pad=%d",
                pad_index,
            )
        snapshot = cache.snapshot(int(pad_index))
        _put_latest(output_queue, (int(pad_index), snapshot))
        if snapshot.state == "stable" and snapshot.result_version > 0:
            completed_pads.add(int(pad_index))
            if completed_pads == set(coordinators):
                LOGGER.info("燃气瓶首次稳定结果已完成，释放YOLOE进程")
                return


class GasCylinderProcessClient:
    """Non-blocking bridge from DeepStream to an isolated YOLOE process."""

    def __init__(
        self,
        config: GasCylinderProcessConfig,
        cache: GasCylinderResultCache,
    ) -> None:
        self._cache = cache
        self._pads = tuple(config.options_by_pad)
        self._completed_pads: set[int] = set()
        context = multiprocessing.get_context("spawn")
        self._input_queue = context.Queue(maxsize=1)
        self._output_queue = context.Queue(maxsize=max(len(self._pads) * 4, 4))
        self._process = context.Process(
            target=_run_gas_cylinder_process,
            args=(config, self._input_queue, self._output_queue),
            name="gas-cylinder-inference",
            daemon=True,
        )
        for pad_index in self._pads:
            cache.mark_state(pad_index, "sampling", "燃气瓶识别中")
        self._process.start()

    def submit(
        self,
        pad_index: int,
        frame: np.ndarray,
        *,
        timestamp: float,
    ) -> bool:
        if not self.accepts(pad_index):
            return False
        try:
            self._input_queue.put_nowait((pad_index, frame, timestamp))
        except queue.Full:
            return False
        return True

    def accepts(self, pad_index: int) -> bool:
        """Return before callers perform a costly device-to-host frame copy."""
        self.drain_results()
        if pad_index in self._completed_pads:
            if self._process.exitcode is not None:
                self._process.join(timeout=0)
            return False
        if not self._process.is_alive():
            self._cache.mark_state(
                pad_index,
                "error",
                "燃气瓶独立推理进程已退出",
            )
            return False
        return True

    def drain_results(self) -> None:
        while True:
            try:
                pad_index, snapshot = self._output_queue.get_nowait()
            except queue.Empty:
                return
            self._cache.store_snapshot(int(pad_index), snapshot)
            if snapshot.state == "stable" and snapshot.result_version > 0:
                self._completed_pads.add(int(pad_index))

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
