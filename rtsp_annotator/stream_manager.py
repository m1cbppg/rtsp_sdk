from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlsplit, urlunsplit

from .config import RoiPolygon
from .events import EventDetectionOptions
from .license_plate import LicensePlateOptions


ProcessFactory = Callable[..., Any]


class StreamNotFoundError(KeyError):
    pass


class ModelNotFoundError(ValueError):
    pass


class StreamCapacityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class NightVisionOptions:
    """Inference-only low-light profile.

    The published video remains unchanged.  DeepStream uses these values only
    while preparing detector input and filtering detector metadata.
    """

    enabled: bool = False
    confidence: float = 0.18
    input_gain: float = 1.18
    plate_detector_confidence: float = 0.20

    def validate(self) -> None:
        if not 0 < self.confidence <= 1:
            raise ValueError("night_vision.confidence必须在(0, 1]范围内")
        if not 1 <= self.input_gain <= 1.5:
            raise ValueError("night_vision.input_gain必须在[1, 1.5]范围内")
        if not 0 < self.plate_detector_confidence <= 1:
            raise ValueError(
                "night_vision.plate_detector_confidence必须在(0, 1]范围内"
            )

    def group_signature(
        self,
        *,
        include_plate_detector: bool = False,
    ) -> tuple[bool, float, float]:
        if not self.enabled:
            return (False, 1.0, 0.30)
        return (
            self.enabled,
            self.input_gain,
            (
                self.plate_detector_confidence
                if include_plate_detector
                else 0.30
            ),
        )


@dataclass(frozen=True, slots=True)
class StreamSpec:
    input_url: str
    model: str = "yolo26s.pt"
    classes: tuple[int, ...] | None = None
    conf: float = 0.10
    iou: float = 0.45
    imgsz: int = 640
    roi: RoiPolygon | None = None
    output_fps: float | None = None
    bitrate: str = "2500k"
    license_plate: LicensePlateOptions = LicensePlateOptions()
    night_vision: NightVisionOptions = NightVisionOptions()
    event_detection: EventDetectionOptions = EventDetectionOptions()


@dataclass(frozen=True, slots=True)
class ManagerSettings:
    model_root: Path
    internal_rtsp_base_url: str
    public_rtsp_base_url: str
    publish_user: str
    publish_password: str
    read_user: str
    read_password: str
    device: str = "cuda:0"
    half: bool = True
    label_map_path: Path | None = None
    font_path: Path | None = None
    encoder: str = "libx264"
    encoder_preset: str = "ultrafast"
    max_batch_size: int = 2
    batch_wait_ms: float = 2.0
    streams_per_model_instance: int = 2
    max_streams: int = 1
    startup_grace_seconds: float = 3.0

    def validate(self) -> None:
        if not self.model_root.is_dir():
            raise RuntimeError(f"模型目录不存在: {self.model_root}")
        _validate_rtsp_base(self.internal_rtsp_base_url)
        _validate_rtsp_base(self.public_rtsp_base_url)
        if not self.publish_user or not self.publish_password:
            raise RuntimeError("RTSP发布账号和密码不能为空")
        if not self.read_user or not self.read_password:
            raise RuntimeError("RTSP读取账号和密码不能为空")
        if self.label_map_path is not None and not self.label_map_path.is_file():
            raise RuntimeError(
                f"中文标签映射文件不存在: {self.label_map_path}"
            )
        if self.font_path is not None and not self.font_path.is_file():
            raise RuntimeError(f"中文字体文件不存在: {self.font_path}")
        if self.max_streams <= 0:
            raise RuntimeError("MAX_STREAMS必须大于0")
        if self.encoder not in {"libx264", "h264_nvenc"}:
            raise RuntimeError("encoder必须是libx264或h264_nvenc")
        if not self.encoder_preset:
            raise RuntimeError("encoder_preset不能为空")
        if self.max_batch_size <= 0:
            raise RuntimeError("max_batch_size必须大于0")
        if not 0 <= self.batch_wait_ms <= 20:
            raise RuntimeError("batch_wait_ms必须在[0, 20]范围内")
        if self.streams_per_model_instance <= 0:
            raise RuntimeError("streams_per_model_instance必须大于0")
        if self.startup_grace_seconds < 0:
            raise RuntimeError("STREAM_STARTUP_GRACE_SECONDS不能为负数")


@dataclass(slots=True)
class StreamRecord:
    stream_id: str
    path: str
    model: str
    classes: tuple[int, ...] | None
    created_at: datetime
    started_monotonic: float
    public_url: str
    process: Any


class StreamManager:
    def __init__(
        self,
        settings: ManagerSettings,
        process_factory: ProcessFactory = subprocess.Popen,
    ) -> None:
        settings.validate()
        self._settings = settings
        self._process_factory = process_factory
        self._records: dict[str, StreamRecord] = {}
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
        if spec.event_detection.enabled:
            raise ModelNotFoundError(
                "事件识别仅支持DeepStream后端"
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
        command = self._build_command(spec, model_path, internal_url)

        with self._lock:
            active = sum(
                record.process.poll() is None
                for record in self._records.values()
            )
            if active >= self._settings.max_streams:
                raise StreamCapacityError(
                    f"已达到并发上限MAX_STREAMS={self._settings.max_streams}"
                )

            process = self._process_factory(
                command,
                stdin=subprocess.DEVNULL,
                stdout=None,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            record = StreamRecord(
                stream_id=stream_id,
                path=path,
                model=model_path.name,
                classes=spec.classes,
                created_at=datetime.now(timezone.utc),
                started_monotonic=time.monotonic(),
                public_url=public_url,
                process=process,
            )
            self._records[stream_id] = record
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
        self._stop_process(record.process)
        result = self._serialize(record)
        result["status"] = "stopped"
        return result

    def shutdown(self) -> None:
        with self._lock:
            records = list(self._records.values())
            self._records.clear()
        for record in records:
            self._stop_process(record.process)

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

    def _build_command(
        self,
        spec: StreamSpec,
        model_path: Path,
        output_url: str,
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "rtsp_annotator",
            "--input",
            spec.input_url,
            "--output",
            output_url,
            "--model",
            str(model_path),
            "--device",
            self._settings.device,
            "--conf",
            str(spec.conf),
            "--iou",
            str(spec.iou),
            "--imgsz",
            str(spec.imgsz),
            "--bitrate",
            spec.bitrate,
            "--encoder",
            self._settings.encoder,
            "--preset",
            self._settings.encoder_preset,
        ]
        if self._settings.half:
            command.append("--half")
        if self._settings.label_map_path is not None:
            command.extend(
                ["--label-map", str(self._settings.label_map_path)]
            )
        if self._settings.font_path is not None:
            command.extend(["--font", str(self._settings.font_path)])
        if spec.classes:
            command.extend(
                ["--classes", ",".join(str(item) for item in spec.classes)]
            )
        if spec.roi:
            command.extend(
                [
                    "--roi",
                    ";".join(f"{x},{y}" for x, y in spec.roi),
                ]
            )
        if spec.output_fps is not None:
            command.extend(["--output-fps", str(spec.output_fps)])
        return command

    def _serialize(self, record: StreamRecord) -> dict[str, Any]:
        exit_code = record.process.poll()
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
        }

    @staticmethod
    def _stop_process(process: Any) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=8)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=3)


def authenticated_rtsp_url(
    base_url: str,
    username: str,
    password: str,
    path: str,
) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
        raise ValueError(f"无效RTSP基础地址: {base_url}")
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    credentials = f"{quote(username, safe='')}:{quote(password, safe='')}"
    normalized_path = f"{parsed.path.rstrip('/')}/{path.lstrip('/')}"
    return urlunsplit(
        (
            parsed.scheme,
            f"{credentials}@{host}",
            normalized_path,
            "",
            "",
        )
    )


def _validate_rtsp_base(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
        raise RuntimeError(f"无效RTSP基础地址: {value}")
