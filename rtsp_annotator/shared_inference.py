from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .labels import load_label_map, resolve_chinese_font
from .pipeline import (
    DetectionSnapshot,
    build_predict_kwargs,
    prediction_to_snapshot,
    render_detection_snapshot,
    resolve_inference_device,
    validate_inference_precision,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InferenceProfile:
    """Parameters that must match for frames to share an inference batch."""

    imgsz: int
    conf: float
    iou: float
    classes: tuple[int, ...] | None
    frame_shape: tuple[int, ...]

    @classmethod
    def from_request(cls, settings: Settings, frame: Any) -> InferenceProfile:
        return cls(
            imgsz=settings.imgsz,
            conf=settings.conf,
            iou=settings.iou,
            classes=settings.classes,
            frame_shape=tuple(int(value) for value in frame.shape),
        )


@dataclass(slots=True)
class InferenceRequest:
    frame: Any
    settings: Settings
    profile: InferenceProfile
    submitted_at: float = field(default_factory=time.monotonic)
    completed: threading.Event = field(default_factory=threading.Event)
    prediction: Any | None = None
    error: BaseException | None = None


class SharedModelWorker:
    """Own one YOLO instance and dynamically batch compatible stream frames."""

    def __init__(
        self,
        model_path: Path,
        *,
        instance_id: int,
        device: str,
        half: bool,
        max_batch_size: int,
        batch_wait_ms: float,
        model_factory: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.instance_id = instance_id
        self.device = device
        self.half = half
        self.max_batch_size = max_batch_size
        self.batch_wait_seconds = batch_wait_ms / 1000.0
        self._model_factory = model_factory
        self._condition = threading.Condition()
        self._pending: list[InferenceRequest] = []
        self._stopping = False
        self._fatal_error: BaseException | None = None
        self._clients = 0
        self._metrics_lock = threading.Lock()
        self._total_batches = 0
        self._total_frames = 0
        self._total_batch_seconds = 0.0
        self._last_batch_size = 0
        self._max_batch_observed = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"shared-inference-{model_path.stem}-{instance_id}",
            daemon=True,
        )
        self._thread.start()

    def add_client(self) -> None:
        with self._condition:
            if self._stopping:
                raise RuntimeError("共享模型工作线程已经停止")
            self._clients += 1
            self._condition.notify_all()

    def remove_client(self) -> int:
        with self._condition:
            self._clients = max(0, self._clients - 1)
            self._condition.notify_all()
            return self._clients

    def client_count(self) -> int:
        with self._condition:
            return self._clients

    def predict(self, frame: Any, settings: Settings) -> Any | None:
        request = InferenceRequest(
            frame=frame,
            settings=settings,
            profile=InferenceProfile.from_request(settings, frame),
        )
        with self._condition:
            if self._fatal_error is not None:
                raise RuntimeError("共享YOLO模型不可用") from self._fatal_error
            if self._stopping:
                raise RuntimeError("共享YOLO模型已经停止")
            self._pending.append(request)
            self._condition.notify_all()

        request.completed.wait()
        if request.error is not None:
            raise RuntimeError("共享YOLO推理失败") from request.error
        return request.prediction

    def close(self) -> None:
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            error = RuntimeError("共享YOLO模型正在停止")
            for request in self._pending:
                request.error = error
                request.completed.set()
            self._pending.clear()
            self._condition.notify_all()
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            LOGGER.warning("共享模型线程未在超时内退出: %s", self.model_path)

    def metrics(self) -> dict[str, float | int]:
        with self._condition:
            clients = self._clients
        with self._metrics_lock:
            batches = self._total_batches
            frames = self._total_frames
            seconds = self._total_batch_seconds
            last_batch_size = self._last_batch_size
            max_batch_observed = self._max_batch_observed
        return {
            "shared_model_clients": clients,
            "model_instance_id": self.instance_id,
            "model_instance_clients": clients,
            "average_batch_size": frames / batches if batches else 0.0,
            "average_batch_inference_ms": (
                seconds / batches * 1000.0 if batches else 0.0
            ),
            "last_batch_size": last_batch_size,
            "max_batch_observed": max_batch_observed,
        }

    def _run(self) -> None:
        try:
            model, resolved_device = self._load_model()
            self._active_model = model
            cuda_stream = self._create_cuda_stream(resolved_device)
        except BaseException as exc:
            LOGGER.exception("共享YOLO模型加载失败: %s", self.model_path)
            with self._condition:
                self._fatal_error = exc
                for request in self._pending:
                    request.error = exc
                    request.completed.set()
                self._pending.clear()
            return

        try:
            while True:
                batch = self._next_batch()
                if batch is None:
                    return
                self._execute_batch(
                    model,
                    resolved_device,
                    batch,
                    cuda_stream,
                )
        finally:
            self._active_model = None
            del model
            self._empty_cuda_cache(resolved_device)

    def _load_model(self) -> tuple[Any, str]:
        resolved = resolve_inference_device(self.device)
        validate_inference_precision(self.half, resolved)
        if self._model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO
        else:
            model_factory = self._model_factory
        LOGGER.info(
            "共享模型实例加载: id=%d，model=%s，设备=%s (%s)，FP16=%s",
            self.instance_id,
            self.model_path,
            resolved.value,
            resolved.name,
            "开启" if self.half else "关闭",
        )
        return model_factory(str(self.model_path)), resolved.value

    def _next_batch(self) -> list[InferenceRequest] | None:
        with self._condition:
            while not self._pending and not self._stopping:
                self._condition.wait()
            if self._stopping:
                return None

            first = self._pending.pop(0)
            batch = [first]
            self._collect_compatible(batch, first.profile)

            should_micro_batch = (
                self._clients > 1
                and self.max_batch_size > 1
                and self.batch_wait_seconds > 0
                and len(batch) < self.max_batch_size
            )
            if not should_micro_batch:
                return batch

            deadline = time.monotonic() + self.batch_wait_seconds
            while len(batch) < self.max_batch_size and not self._stopping:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
                self._collect_compatible(batch, first.profile)
            return batch

    def _collect_compatible(
        self,
        batch: list[InferenceRequest],
        profile: InferenceProfile,
    ) -> None:
        index = 0
        while index < len(self._pending) and len(batch) < self.max_batch_size:
            if self._pending[index].profile == profile:
                batch.append(self._pending.pop(index))
            else:
                index += 1

    def _execute_batch(
        self,
        model: Any,
        resolved_device: str,
        batch: list[InferenceRequest],
        cuda_stream: Any | None,
    ) -> None:
        first = batch[0]
        kwargs = build_predict_kwargs(
            first.settings,
            device=resolved_device,
            half=self.half,
        )
        frames = [request.frame for request in batch]
        kwargs["source"] = frames[0] if len(frames) == 1 else frames
        started = time.monotonic()
        try:
            if cuda_stream is None:
                predictions = list(model.predict(**kwargs))
            else:
                import torch

                with torch.cuda.stream(cuda_stream):
                    predictions = list(model.predict(**kwargs))
                cuda_stream.synchronize()
            if len(predictions) != len(batch):
                raise RuntimeError(
                    "YOLO批量结果数量不匹配: "
                    f"输入={len(batch)}，输出={len(predictions)}"
                )
            elapsed_ms = (time.monotonic() - started) * 1000.0
            with self._metrics_lock:
                self._total_batches += 1
                self._total_frames += len(batch)
                self._total_batch_seconds += elapsed_ms / 1000.0
                self._last_batch_size = len(batch)
                self._max_batch_observed = max(
                    self._max_batch_observed,
                    len(batch),
                )
            LOGGER.debug(
                (
                    "共享推理批次: instance=%d model=%s batch=%d "
                    "total=%.1fms per_frame=%.1fms"
                ),
                self.instance_id,
                self.model_path.name,
                len(batch),
                elapsed_ms,
                elapsed_ms / len(batch),
            )
            for request, prediction in zip(batch, predictions):
                request.prediction = prediction
        except BaseException as exc:
            LOGGER.exception(
                "共享YOLO批量推理失败: instance=%d model=%s batch=%d",
                self.instance_id,
                self.model_path.name,
                len(batch),
            )
            for request in batch:
                request.error = exc
        finally:
            for request in batch:
                request.completed.set()

    @staticmethod
    def _create_cuda_stream(device: str) -> Any | None:
        if not device.startswith("cuda"):
            return None
        import torch

        return torch.cuda.Stream(device=device)

    @staticmethod
    def _empty_cuda_cache(device: str) -> None:
        if not device.startswith("cuda"):
            return
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            LOGGER.debug("释放CUDA缓存失败", exc_info=True)


class SharedDetector:
    """Per-stream rendering context backed by a shared model worker."""

    def __init__(
        self,
        registry: SharedModelRegistry,
        worker_key: tuple[str, str, bool],
        worker: SharedModelWorker,
        settings: Settings,
    ) -> None:
        self._registry = registry
        self._worker_key = worker_key
        self._worker = worker
        self._settings = settings
        self._closed = False
        self._label_map = load_label_map(settings.label_map_path)
        self._font_path = (
            resolve_chinese_font(settings.font_path)
            if settings.show_labels
            else None
        )

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
                font_path=self._font_path,
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
        prediction = self._worker.predict(frame, self._settings)
        model = getattr(self._worker, "_active_model", None)
        model_names = getattr(model, "names", None)
        if model_names is None and prediction is not None:
            model_names = getattr(prediction, "names", None)
        return prediction_to_snapshot(
            frame,
            prediction,
            settings=self._settings,
            model_names=model_names,
            label_map=self._label_map,
            captured_at=captured_at,
            source_sequence=source_sequence,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._registry.release(self._worker_key, self._worker)

    def metrics(self) -> dict[str, float | int]:
        return self._worker.metrics()


class SharedModelRegistry:
    """Pool YOLO instances, pinning at most N streams to each instance."""

    def __init__(
        self,
        *,
        max_batch_size: int = 2,
        batch_wait_ms: float = 2.0,
        streams_per_model_instance: int = 2,
        model_factory: Any | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size必须大于0")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms不能为负数")
        if streams_per_model_instance <= 0:
            raise ValueError("streams_per_model_instance必须大于0")
        self.streams_per_model_instance = streams_per_model_instance
        self.max_batch_size = min(
            max_batch_size,
            streams_per_model_instance,
        )
        self.batch_wait_ms = batch_wait_ms
        self._model_factory = model_factory
        self._workers: dict[
            tuple[str, str, bool],
            list[SharedModelWorker],
        ] = {}
        self._next_instance_ids: dict[tuple[str, str, bool], int] = {}
        self._lock = threading.Lock()

    def acquire(self, settings: Settings) -> SharedDetector:
        key = (
            str(settings.model_path.resolve()),
            settings.device,
            settings.half,
        )
        with self._lock:
            workers = self._workers.setdefault(key, [])
            available = [
                worker
                for worker in workers
                if worker.client_count() < self.streams_per_model_instance
            ]
            if available:
                worker = min(available, key=lambda item: item.client_count())
            else:
                instance_id = self._next_instance_ids.get(key, 1)
                self._next_instance_ids[key] = instance_id + 1
                worker = SharedModelWorker(
                    settings.model_path,
                    instance_id=instance_id,
                    device=settings.device,
                    half=settings.half,
                    max_batch_size=self.max_batch_size,
                    batch_wait_ms=self.batch_wait_ms,
                    model_factory=self._model_factory,
                )
                workers.append(worker)
            worker.add_client()
        return SharedDetector(self, key, worker, settings)

    def release(
        self,
        key: tuple[str, str, bool],
        worker: SharedModelWorker,
    ) -> None:
        should_close = False
        with self._lock:
            workers = self._workers.get(key)
            if workers is None or worker not in workers:
                return
            if worker.remove_client() == 0:
                workers.remove(worker)
                if not workers:
                    self._workers.pop(key, None)
                    self._next_instance_ids.pop(key, None)
                should_close = True
        if should_close:
            worker.close()

    def close(self) -> None:
        with self._lock:
            workers = [
                worker
                for group in self._workers.values()
                for worker in group
            ]
            self._workers.clear()
            self._next_instance_ids.clear()
        for worker in workers:
            worker.close()
