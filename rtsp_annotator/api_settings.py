from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .deepstream_manager import DeepStreamManagerSettings
from .stream_manager import ManagerSettings


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ApiServerConfig(StrictConfigModel):
    key: SecretStr = Field(min_length=16)
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)


class ModelsConfig(StrictConfigModel):
    root: Path


class RtspConfig(StrictConfigModel):
    internal_base_url: str
    public_base_url: str
    publish_user: str = Field(min_length=1)
    publish_password: SecretStr = Field(min_length=8)
    read_user: str = Field(min_length=1)
    read_password: SecretStr = Field(min_length=8)


class InferenceConfig(StrictConfigModel):
    backend: str = Field(
        default="python",
        pattern=r"^(python|deepstream)$",
    )
    device: str = "cuda:0"
    half: bool = True
    shared_model: bool = True
    max_batch_size: int = Field(default=2, ge=1, le=32)
    batch_wait_ms: float = Field(default=2.0, ge=0, le=20)
    streams_per_model_instance: int = Field(default=2, ge=1, le=16)
    max_streams: int = Field(default=1, ge=1)
    startup_grace_seconds: float = Field(default=3.0, ge=0)


class DeepStreamConfig(StrictConfigModel):
    onnx_root: Path = Path("/app/models")
    engine_root: Path = Path("/app/engines")
    runtime_root: Path = Path("/app/runtime")
    parser_library: Path = Path(
        "/opt/nvidia/deepstream/deepstream/lib/"
        "libnvdsinfer_custom_impl_Yolo.so"
    )
    tracker_library: Path = Path(
        "/opt/nvidia/deepstream/deepstream/lib/"
        "libnvds_nvmultiobjecttracker.so"
    )
    tracker_config: Path = Path(
        "/opt/nvidia/deepstream/deepstream/samples/configs/"
        "deepstream-app/config_tracker_NvDCF_perf.yml"
    )
    tracker_max_shadow_tracking_age: int = Field(
        default=15,
        ge=1,
        le=200,
    )
    gpu_id: int = Field(default=0, ge=0)
    streams_per_group: int = Field(default=2, ge=1, le=2)
    model_input_size: int = Field(default=640, ge=32, le=2048)
    mux_width: int = Field(default=1920, ge=64, le=7680)
    mux_height: int = Field(default=1080, ge=64, le=4320)
    batch_push_timeout_us: int = Field(default=20000, ge=1000, le=1000000)
    source_latency_ms: int = Field(default=300, ge=0, le=5000)
    encoder_iframe_interval: int = Field(default=25, ge=1, le=600)
    stats_interval_seconds: float = Field(default=5.0, ge=1, le=60)
    minimum_healthy_fps: float = Field(default=20.0, ge=1, le=120)
    worker_stop_grace_seconds: float = Field(default=0.5, ge=0, le=5)
    lpr_model_root: Path = Path("/app/models/lpr")
    lpr_parser_library: Path = Path(
        "/opt/nvidia/deepstream/deepstream/lib/"
        "libnvdsinfer_custom_impl_lpr.so"
    )
    lpr_detector_batch_size: int = Field(default=16, ge=1, le=64)
    lpr_recognizer_batch_size: int = Field(default=16, ge=1, le=64)
    garbage_onnx_path: Path = Path(
        "/app/models/events/yolo_world_garbage.onnx"
    )
    garbage_labels_path: Path = Path(
        "/app/models/events/yolo_world_garbage.labels.txt"
    )
    garbage_pile_onnx_path: Path = Path(
        "/app/models/events/street_garbage_pile.onnx"
    )
    garbage_pile_labels_path: Path = Path(
        "/app/models/events/street_garbage_pile.labels.txt"
    )
    garbage_parser_library: Path = Path(
        "/opt/nvidia/deepstream/deepstream/lib/"
        "libnvdsinfer_custom_impl_Yolo.so"
    )
    garbage_input_size: int = Field(default=640, ge=320, le=1280)


class OutputConfig(StrictConfigModel):
    encoder: str = Field(default="libx264", pattern=r"^(libx264|h264_nvenc)$")
    preset: str = Field(default="ultrafast", min_length=1)


class LabelsConfig(StrictConfigModel):
    map_file: Path | None = None
    font_file: Path | None = None


class EventsConfig(StrictConfigModel):
    storage_root: Path = Path("data/events")


class AppConfig(StrictConfigModel):
    api: ApiServerConfig
    models: ModelsConfig
    rtsp: RtspConfig
    inference: InferenceConfig
    deepstream: DeepStreamConfig = Field(default_factory=DeepStreamConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)

    def to_manager_settings(self) -> ManagerSettings:
        return ManagerSettings(
            model_root=self.models.root.expanduser(),
            internal_rtsp_base_url=self.rtsp.internal_base_url,
            public_rtsp_base_url=self.rtsp.public_base_url,
            publish_user=self.rtsp.publish_user,
            publish_password=self.rtsp.publish_password.get_secret_value(),
            read_user=self.rtsp.read_user,
            read_password=self.rtsp.read_password.get_secret_value(),
            device=self.inference.device,
            half=self.inference.half,
            max_batch_size=self.inference.max_batch_size,
            batch_wait_ms=self.inference.batch_wait_ms,
            streams_per_model_instance=(
                self.inference.streams_per_model_instance
            ),
            encoder=self.output.encoder,
            encoder_preset=self.output.preset,
            label_map_path=(
                self.labels.map_file.expanduser()
                if self.labels.map_file is not None
                else None
            ),
            font_path=(
                self.labels.font_file.expanduser()
                if self.labels.font_file is not None
                else None
            ),
            max_streams=self.inference.max_streams,
            startup_grace_seconds=self.inference.startup_grace_seconds,
        )

    def to_deepstream_manager_settings(self) -> DeepStreamManagerSettings:
        return DeepStreamManagerSettings(
            manager=self.to_manager_settings(),
            onnx_root=self.deepstream.onnx_root.expanduser(),
            engine_root=self.deepstream.engine_root.expanduser(),
            runtime_root=self.deepstream.runtime_root.expanduser(),
            parser_library=self.deepstream.parser_library.expanduser(),
            tracker_library=self.deepstream.tracker_library.expanduser(),
            tracker_config=self.deepstream.tracker_config.expanduser(),
            tracker_max_shadow_tracking_age=(
                self.deepstream.tracker_max_shadow_tracking_age
            ),
            gpu_id=self.deepstream.gpu_id,
            streams_per_group=self.deepstream.streams_per_group,
            model_input_size=self.deepstream.model_input_size,
            mux_width=self.deepstream.mux_width,
            mux_height=self.deepstream.mux_height,
            batch_push_timeout_us=self.deepstream.batch_push_timeout_us,
            source_latency_ms=self.deepstream.source_latency_ms,
            encoder_iframe_interval=(
                self.deepstream.encoder_iframe_interval
            ),
            stats_interval_seconds=self.deepstream.stats_interval_seconds,
            minimum_healthy_fps=self.deepstream.minimum_healthy_fps,
            worker_stop_grace_seconds=(
                self.deepstream.worker_stop_grace_seconds
            ),
            lpr_model_root=self.deepstream.lpr_model_root.expanduser(),
            lpr_parser_library=(
                self.deepstream.lpr_parser_library.expanduser()
            ),
            lpr_detector_batch_size=(
                self.deepstream.lpr_detector_batch_size
            ),
            lpr_recognizer_batch_size=(
                self.deepstream.lpr_recognizer_batch_size
            ),
            event_root=self.events.storage_root.expanduser().resolve(),
            garbage_onnx_path=self.deepstream.garbage_onnx_path.expanduser(),
            garbage_labels_path=(
                self.deepstream.garbage_labels_path.expanduser()
            ),
            garbage_pile_onnx_path=(
                self.deepstream.garbage_pile_onnx_path.expanduser()
            ),
            garbage_pile_labels_path=(
                self.deepstream.garbage_pile_labels_path.expanduser()
            ),
            garbage_parser_library=(
                self.deepstream.garbage_parser_library.expanduser()
            ),
            garbage_input_size=self.deepstream.garbage_input_size,
        )


def load_app_config(path: Path) -> AppConfig:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"API配置文件不存在: {resolved}")
    try:
        return AppConfig.model_validate_json(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"API配置文件无效: {resolved}: {exc}") from exc
