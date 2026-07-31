from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .pipeline import run_pipeline
from .shared_inference import SharedModelRegistry
from .stream_manager import (
    ManagerSettings,
    ModelNotFoundError,
    StreamCapacityError,
    StreamNotFoundError,
    StreamSpec,
    authenticated_rtsp_url,
)


class PipelineSession:
    """One stream pipeline running in threads around a shared model worker."""

    def __init__(
        self,
        *,
        stream_id: str,
        settings: Settings,
        registry: SharedModelRegistry,
    ) -> None:
        self.stream_id = stream_id
        self.settings = settings
        self._registry = registry
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._returncode: int | None = None
        self._error: str | None = None
        self._metrics: dict[str, float | int] | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"stream-{stream_id[:8]}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def poll(self) -> int | None:
        with self._state_lock:
            return self._returncode

    def metrics(self) -> dict[str, float | int] | None:
        with self._state_lock:
            return dict(self._metrics) if self._metrics is not None else None

    def error(self) -> str | None:
        with self._state_lock:
            return self._error

    def stop(self, timeout: float = 15.0) -> None:
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError(f"流任务未在{timeout:.0f}秒内停止")

    def _run(self) -> None:
        returncode = 0
        error: str | None = None
        detector: Any | None = None
        try:
            detector = self._registry.acquire(self.settings)
            run_pipeline(
                self.settings,
                detector_factory=lambda _settings: detector,
                external_stop_event=self._stop_event,
                stream_id=self.stream_id,
                stats_callback=self._record_metrics,
            )
        except BaseException as exc:
            returncode = 1
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if detector is not None:
                detector.close()
            with self._state_lock:
                self._returncode = returncode
                self._error = error

    def _record_metrics(self, report: dict[str, float | int]) -> None:
        with self._state_lock:
            self._metrics = dict(report)


@dataclass(slots=True)
class SharedStreamRecord:
    stream_id: str
    path: str
    model: str
    classes: tuple[int, ...] | None
    created_at: datetime
    started_monotonic: float
    public_url: str
    session: PipelineSession


class SharedStreamManager:
    """API stream manager with one model instance per active model file."""

    def __init__(
        self,
        settings: ManagerSettings,
        *,
        registry: SharedModelRegistry | None = None,
    ) -> None:
        settings.validate()
        self._settings = settings
        self._registry = registry or SharedModelRegistry(
            max_batch_size=settings.max_batch_size,
            batch_wait_ms=settings.batch_wait_ms,
            streams_per_model_instance=(
                settings.streams_per_model_instance
            ),
        )
        self._records: dict[str, SharedStreamRecord] = {}
        self._lock = threading.Lock()

    def list_models(self) -> list[str]:
        return sorted(
            path.name
            for path in self._settings.model_root.glob("*.pt")
            if path.is_file()
        )

    def create(self, spec: StreamSpec) -> dict[str, Any]:
        if spec.license_plate.enabled:
            raise ModelNotFoundError(
                "中国车牌识别仅支持DeepStream后端"
            )
        if spec.night_vision.enabled:
            raise ModelNotFoundError(
                "夜间增强仅支持DeepStream后端"
            )
        model_path = self._resolve_model(spec.model)
        stream_id = uuid.uuid4().hex
        path = f"detected/{stream_id}"
        internal_url = authenticated_rtsp_url(
            self._settings.internal_rtsp_base_url,
            self._settings.publish_user,
            self._settings.publish_password,
            path,
        )
        public_url = authenticated_rtsp_url(
            self._settings.public_rtsp_base_url,
            self._settings.read_user,
            self._settings.read_password,
            path,
        )
        pipeline_settings = Settings(
            input_url=spec.input_url,
            output_url=internal_url,
            model_path=model_path,
            conf=spec.conf,
            iou=spec.iou,
            imgsz=spec.imgsz,
            classes=spec.classes,
            roi=spec.roi,
            device=self._settings.device,
            half=self._settings.half,
            label_map_path=self._settings.label_map_path,
            font_path=self._settings.font_path,
            output_fps=spec.output_fps,
            bitrate=spec.bitrate,
            encoder=self._settings.encoder,
            preset=self._settings.encoder_preset,
        )
        session = PipelineSession(
            stream_id=stream_id,
            settings=pipeline_settings,
            registry=self._registry,
        )
        record = SharedStreamRecord(
            stream_id=stream_id,
            path=path,
            model=model_path.name,
            classes=spec.classes,
            created_at=datetime.now(timezone.utc),
            started_monotonic=time.monotonic(),
            public_url=public_url,
            session=session,
        )

        with self._lock:
            active = sum(
                item.session.poll() is None for item in self._records.values()
            )
            if active >= self._settings.max_streams:
                raise StreamCapacityError(
                    f"已达到并发上限MAX_STREAMS={self._settings.max_streams}"
                )
            self._records[stream_id] = record
            session.start()
        return self._serialize(record)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._records.values())
        return [self._serialize(record) for record in records]

    def get(self, stream_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(stream_id)
        if record is None:
            raise StreamNotFoundError(stream_id)
        return self._serialize(record)

    def stop(self, stream_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.pop(stream_id, None)
        if record is None:
            raise StreamNotFoundError(stream_id)
        record.session.stop()
        result = self._serialize(record)
        result["status"] = "stopped"
        return result

    def shutdown(self) -> None:
        with self._lock:
            records = list(self._records.values())
            self._records.clear()
        for record in records:
            try:
                record.session.stop()
            except TimeoutError:
                pass
        self._registry.close()

    def _resolve_model(self, model: str) -> Path:
        if not model or Path(model).name != model or not model.endswith(".pt"):
            raise ModelNotFoundError(
                "model必须是models目录中的.pt文件名，不能包含路径"
            )
        root = self._settings.model_root.resolve()
        candidate = (root / model).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ModelNotFoundError("模型路径越界") from exc
        if not candidate.is_file():
            raise ModelNotFoundError(f"模型不存在: {model}")
        return candidate

    def _serialize(self, record: SharedStreamRecord) -> dict[str, Any]:
        exit_code = record.session.poll()
        if exit_code is not None:
            status = "failed" if exit_code != 0 else "stopped"
        elif (
            time.monotonic() - record.started_monotonic
            < self._settings.startup_grace_seconds
        ):
            status = "starting"
        else:
            status = "running"
        return {
            "stream_id": record.stream_id,
            "status": status,
            "rtsp_url": record.public_url,
            "model": record.model,
            "classes": (
                list(record.classes) if record.classes is not None else None
            ),
            "created_at": record.created_at.isoformat(),
            "exit_code": exit_code,
            "metrics": record.session.metrics(),
        }
