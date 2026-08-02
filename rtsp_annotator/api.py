from __future__ import annotations

import argparse
import hmac
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import FileResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .api_settings import AppConfig, load_app_config
from .config import parse_roi
from .deepstream_manager import DeepStreamStreamManager
from .events import (
    EventDetectionOptions,
    EventNotFoundError,
    EventRepository,
    EventRoiOptions,
    EventRuleOptions,
    GarbageAnalysisOptions,
    WebhookOptions,
)
from .license_plate import DEFAULT_VEHICLE_CLASSES, LicensePlateOptions
from .shared_stream_manager import SharedStreamManager
from .stream_manager import (
    ModelNotFoundError,
    NightVisionOptions,
    StreamCapacityError,
    StreamManager,
    StreamNotFoundError,
    StreamSpec,
)


class LicensePlateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    detector_interval: int = Field(
        default=0,
        ge=0,
        le=30,
        description="车牌检测跳帧数；0表示每帧检测，确保车牌框连续",
    )
    recognition_reinfer_interval: int = Field(
        default=15,
        ge=1,
        le=300,
        description="同一车牌重新识别的帧间隔",
    )
    minimum_confirmations: int = Field(default=2, ge=1, le=5)
    minimum_plate_confidence: float = Field(default=0.5, gt=0, le=1)
    vehicle_classes: list[int] = Field(
        default_factory=lambda: list(DEFAULT_VEHICLE_CLASSES),
        description="YOLO车辆类别ID；COCO默认2/3/5/7",
    )

    @field_validator("vehicle_classes")
    @classmethod
    def validate_vehicle_classes(cls, value: list[int]) -> list[int]:
        if not value or any(item < 0 for item in value):
            raise ValueError("vehicle_classes必须是非空的非负类别ID列表")
        return list(dict.fromkeys(value))

    def to_options(self) -> LicensePlateOptions:
        return LicensePlateOptions(
            enabled=self.enabled,
            detector_interval=self.detector_interval,
            recognition_reinfer_interval=(
                self.recognition_reinfer_interval
            ),
            minimum_confirmations=self.minimum_confirmations,
            minimum_plate_confidence=self.minimum_plate_confidence,
            vehicle_classes=tuple(self.vehicle_classes),
        )


class NightVisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    confidence: float = Field(
        default=0.18,
        gt=0,
        le=1,
        description="夜间最终显示阈值；仅在enabled=true时使用",
    )
    input_gain: float = Field(
        default=1.18,
        ge=1,
        le=1.5,
        description=(
            "夜间推理输入线性增益；"
            "只作用于模型输入，不改变输出视频"
        ),
    )
    plate_detector_confidence: float = Field(
        default=0.20,
        gt=0,
        le=1,
        description="夜间车牌检测阈值；仅影响车牌支路",
    )

    def to_options(self) -> NightVisionOptions:
        return NightVisionOptions(
            enabled=self.enabled,
            confidence=self.confidence,
            input_gain=self.input_gain,
            plate_detector_confidence=self.plate_detector_confidence,
        )


class EventRuleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_dwell_seconds: float = Field(default=20, gt=0, le=86_400)
    vehicle_dwell_seconds: float = Field(default=20, gt=0, le=86_400)
    actor_leave_grace_seconds: float = Field(default=3, ge=0, le=30)
    actor_association_seconds: float = Field(default=60, ge=1, le=600)
    garbage_persistence_seconds: float = Field(default=15, ge=1, le=300)
    minimum_change_area: float = Field(default=0.002, gt=0, le=0.5)

    def to_options(self) -> EventRuleOptions:
        return EventRuleOptions(**self.model_dump())


class EventRoiRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    polygon: list[tuple[float, float]]
    dwell_enabled: bool = True
    garbage_enabled: bool = True
    rules: EventRuleRequest = Field(default_factory=EventRuleRequest)

    @field_validator("polygon")
    @classmethod
    def validate_polygon(
        cls,
        value: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        serialized = ";".join(f"{x},{y}" for x, y in value)
        parsed = parse_roi(serialized)
        assert parsed is not None
        return list(parsed)

    def to_options(self) -> EventRoiOptions:
        return EventRoiOptions(
            roi_id=self.id,
            polygon=tuple(self.polygon),
            dwell_enabled=self.dwell_enabled,
            garbage_enabled=self.garbage_enabled,
            rules=self.rules.to_options(),
        )


class GarbageAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    analysis_fps: float = Field(default=3, ge=0.1, le=10)
    minimum_confidence: float = Field(default=0.35, gt=0, le=1)
    detection_mode: Literal["items", "pile"] = Field(
        default="items",
        description="items识别零散垃圾；pile识别并合并街景垃圾堆",
    )
    background_change_enabled: bool = Field(
        default=True,
        description="是否使用背景变化参与新增/清理事件判断",
    )
    display_detections: bool = Field(
        default=True,
        description="在主视频中持续显示垃圾语义框",
    )
    display_hold_seconds: float = Field(
        default=1.5,
        ge=0.3,
        le=10,
        description="垃圾旁路暂时无结果时保留最近框的秒数",
    )
    maximum_display_boxes: int = Field(
        default=20,
        ge=1,
        le=100,
        description="每路画面最多显示的普通垃圾框数量",
    )
    minimum_pile_detections: int = Field(
        default=2,
        ge=1,
        le=20,
        description="合并为垃圾堆前至少需要的组成检测数量",
    )
    pile_merge_distance: float = Field(
        default=0.18,
        ge=0.01,
        le=0.5,
        description="垃圾组成框聚类的最大归一化间距",
    )
    pile_box_padding: float = Field(
        default=0.04,
        ge=0,
        le=0.25,
        description="合并后的垃圾堆框向外扩展比例",
    )
    prompts: list[str] = Field(
        default_factory=lambda: list(GarbageAnalysisOptions().prompts),
        min_length=1,
        max_length=64,
    )

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("garbage.prompts不能包含空字符串")
        return list(dict.fromkeys(normalized))

    def to_options(self) -> GarbageAnalysisOptions:
        return GarbageAnalysisOptions(
            enabled=self.enabled,
            analysis_fps=self.analysis_fps,
            minimum_confidence=self.minimum_confidence,
            detection_mode=self.detection_mode,
            background_change_enabled=self.background_change_enabled,
            display_detections=self.display_detections,
            display_hold_seconds=self.display_hold_seconds,
            maximum_display_boxes=self.maximum_display_boxes,
            minimum_pile_detections=self.minimum_pile_detections,
            pile_merge_distance=self.pile_merge_distance,
            pile_box_padding=self.pile_box_padding,
            prompts=tuple(self.prompts),
        )


class EventWebhookRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    timeout_seconds: float = Field(default=3, ge=0.1, le=30)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("webhook.url必须以http://或https://开头")
        return value

    def to_options(self) -> WebhookOptions:
        return WebhookOptions(
            url=self.url,
            timeout_seconds=self.timeout_seconds,
        )


class EventDetectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    rois: list[EventRoiRequest] = Field(default_factory=list, max_length=16)
    garbage: GarbageAnalysisRequest = Field(
        default_factory=GarbageAnalysisRequest
    )
    webhook: EventWebhookRequest = Field(default_factory=EventWebhookRequest)
    person_classes: list[int] = Field(default_factory=lambda: [0])
    vehicle_classes: list[int] = Field(
        default_factory=lambda: list(DEFAULT_VEHICLE_CLASSES)
    )

    @field_validator("person_classes", "vehicle_classes")
    @classmethod
    def validate_actor_classes(cls, value: list[int]) -> list[int]:
        if any(item < 0 for item in value):
            raise ValueError("事件人物和车辆类别必须是非负ID列表")
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_event_options(self) -> "EventDetectionRequest":
        self.to_options()
        return self

    def to_options(self) -> EventDetectionOptions:
        options = EventDetectionOptions(
            enabled=self.enabled,
            rois=tuple(item.to_options() for item in self.rois),
            garbage=self.garbage.to_options(),
            webhook=self.webhook.to_options(),
            person_classes=tuple(self.person_classes),
            vehicle_classes=tuple(self.vehicle_classes),
        )
        options.validate()
        return options


class StreamCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_url: str = Field(
        description="原始RTSP地址，例如rtsp://user:pass@camera/live"
    )
    model: str = Field(
        default="yolo26s.pt",
        description="镜像/app/models目录内的.pt文件名",
    )
    classes: list[int] | None = Field(
        default=None,
        description="类别ID列表；null表示模型支持的全部类别",
    )
    conf: float = Field(default=0.25, gt=0, le=1)
    iou: float = Field(default=0.45, gt=0, le=1)
    imgsz: int = Field(default=640, ge=32, le=2048)
    roi: list[tuple[float, float]] | None = Field(
        default=None,
        description="归一化多边形顶点，例如[[0.1,0.1],[0.9,0.1],[0.5,0.9]]",
    )
    output_fps: float | None = Field(default=None, ge=0.1, le=120)
    bitrate: str = Field(default="2500k", pattern=r"^\d+[kKmM]?$")
    license_plate: LicensePlateRequest = Field(
        default_factory=LicensePlateRequest,
        description="中国车牌检测、跟踪和字符识别配置",
    )
    night_vision: NightVisionRequest = Field(
        default_factory=NightVisionRequest,
        description="独立夜间推理配置；默认关闭且不改变白天逻辑",
    )
    event_detection: EventDetectionRequest = Field(
        default_factory=EventDetectionRequest,
        description="区域停留、垃圾变化和疑似乱丢垃圾事件配置",
    )

    @field_validator("input_url")
    @classmethod
    def validate_input_url(cls, value: str) -> str:
        if not value.lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError("input_url必须以rtsp://或rtsps://开头")
        return value

    @field_validator("model")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if not value or value != Path(value).name or not value.endswith(".pt"):
            raise ValueError("model必须是/app/models中的.pt文件名")
        return value

    @field_validator("classes")
    @classmethod
    def validate_classes(
        cls,
        value: list[int] | None,
    ) -> list[int] | None:
        if value is None:
            return None
        if any(item < 0 for item in value):
            raise ValueError("类别ID不能为负数")
        return list(dict.fromkeys(value)) or None

    @field_validator("roi")
    @classmethod
    def validate_roi(
        cls,
        value: list[tuple[float, float]] | None,
    ) -> list[tuple[float, float]] | None:
        if value is None:
            return None
        serialized = ";".join(f"{x},{y}" for x, y in value)
        parsed = parse_roi(serialized)
        assert parsed is not None
        return list(parsed)

    def to_spec(self) -> StreamSpec:
        roi = tuple(self.roi) if self.roi is not None else None
        classes = tuple(self.classes) if self.classes is not None else None
        return StreamSpec(
            input_url=self.input_url,
            model=self.model,
            classes=classes,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            roi=roi,
            output_fps=self.output_fps,
            bitrate=self.bitrate,
            license_plate=self.license_plate.to_options(),
            night_vision=self.night_vision.to_options(),
            event_detection=self.event_detection.to_options(),
        )


class StreamResponse(BaseModel):
    stream_id: str
    status: str
    rtsp_url: str
    model: str
    classes: list[int] | None
    created_at: str
    exit_code: int | None
    metrics: dict[str, float | int | str | bool] | None = None
    license_plate: dict[str, Any] | None = None
    night_vision: dict[str, Any] | None = None
    event_detection: dict[str, Any] | None = None


class MediaMtxAuthRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    user: str = ""
    password: str = ""
    action: str
    path: str = ""


def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    config: AppConfig = request.app.state.config
    expected = config.api.key.get_secret_value()
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效X-API-Key",
        )


def _manager(request: Request) -> Any:
    return request.app.state.manager


def _events(request: Request) -> EventRepository:
    return request.app.state.events


def create_app(config_path: Path = Path("config/api.json")) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        config = load_app_config(config_path)
        manager_settings = config.to_manager_settings()
        if config.inference.backend == "deepstream":
            manager = DeepStreamStreamManager(
                config.to_deepstream_manager_settings()
            )
        elif config.inference.shared_model:
            manager = SharedStreamManager(manager_settings)
        else:
            manager = StreamManager(manager_settings)
        application.state.config = config
        application.state.manager = manager
        application.state.events = EventRepository(
            config.events.storage_root.expanduser().resolve()
        )
        yield
        manager.shutdown()

    application = FastAPI(
        title="RTSP YOLO Stream API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post(
        "/internal/mediamtx/auth",
        include_in_schema=False,
    )
    def mediamtx_auth(
        payload: MediaMtxAuthRequest,
        request: Request,
    ) -> Response:
        config: AppConfig = request.app.state.config
        path_allowed = payload.path.startswith("detected/")
        publish_allowed = (
            payload.action == "publish"
            and path_allowed
            and hmac.compare_digest(payload.user, config.rtsp.publish_user)
            and hmac.compare_digest(
                payload.password,
                config.rtsp.publish_password.get_secret_value(),
            )
        )
        read_allowed = (
            payload.action == "read"
            and path_allowed
            and hmac.compare_digest(payload.user, config.rtsp.read_user)
            and hmac.compare_digest(
                payload.password,
                config.rtsp.read_password.get_secret_value(),
            )
        )
        if not publish_allowed and not read_allowed:
            raise HTTPException(status_code=401, detail="RTSP认证失败")
        return Response(status_code=200)

    @application.get(
        "/v1/models",
        dependencies=[Depends(require_api_key)],
    )
    def list_models(request: Request) -> dict[str, list[str]]:
        return {"models": _manager(request).list_models()}

    @application.post(
        "/v1/streams",
        response_model=StreamResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_api_key)],
    )
    def create_stream(
        payload: StreamCreateRequest,
        request: Request,
    ) -> dict[str, Any]:
        try:
            return _manager(request).create(payload.to_spec())
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except StreamCapacityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get(
        "/v1/streams",
        response_model=list[StreamResponse],
        dependencies=[Depends(require_api_key)],
    )
    def list_streams(request: Request) -> list[dict[str, Any]]:
        return _manager(request).list()

    @application.get(
        "/v1/streams/{stream_id}",
        response_model=StreamResponse,
        dependencies=[Depends(require_api_key)],
    )
    def get_stream(stream_id: str, request: Request) -> dict[str, Any]:
        try:
            return _manager(request).get(stream_id)
        except StreamNotFoundError as exc:
            raise HTTPException(status_code=404, detail="流任务不存在") from exc

    @application.delete(
        "/v1/streams/{stream_id}",
        response_model=StreamResponse,
        dependencies=[Depends(require_api_key)],
    )
    def delete_stream(stream_id: str, request: Request) -> dict[str, Any]:
        try:
            return _manager(request).stop(stream_id)
        except StreamNotFoundError as exc:
            raise HTTPException(status_code=404, detail="流任务不存在") from exc

    @application.get(
        "/v1/events",
        dependencies=[Depends(require_api_key)],
    )
    def list_events(
        request: Request,
        stream_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        try:
            return _events(request).list(stream_id=stream_id, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get(
        "/v1/streams/{stream_id}/events",
        dependencies=[Depends(require_api_key)],
    )
    def list_stream_events(
        stream_id: str,
        request: Request,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        try:
            return _events(request).list(stream_id=stream_id, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get(
        "/v1/events/{event_id}",
        dependencies=[Depends(require_api_key)],
    )
    def get_event(event_id: str, request: Request) -> dict[str, Any]:
        try:
            return _events(request).get(event_id)
        except (EventNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="事件不存在") from exc

    @application.post(
        "/v1/events/{event_id}/confirm",
        dependencies=[Depends(require_api_key)],
    )
    def confirm_event(event_id: str, request: Request) -> dict[str, Any]:
        try:
            return _events(request).review(event_id, "confirmed")
        except (EventNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="事件不存在") from exc

    @application.post(
        "/v1/events/{event_id}/reject",
        dependencies=[Depends(require_api_key)],
    )
    def reject_event(event_id: str, request: Request) -> dict[str, Any]:
        try:
            return _events(request).review(event_id, "rejected")
        except (EventNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="事件不存在") from exc

    @application.get(
        "/v1/events/{event_id}/snapshot",
        dependencies=[Depends(require_api_key)],
        response_class=FileResponse,
    )
    def event_snapshot(event_id: str, request: Request) -> FileResponse:
        try:
            path = _events(request).media_path(event_id, "snapshot_path")
        except (EventNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="事件截图不存在") from exc
        return FileResponse(path)

    return application


app = create_app()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="启动RTSP YOLO HTTP API")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/api.json"),
        help="JSON配置文件",
    )
    args = parser.parse_args(argv)
    config = load_app_config(args.config)

    import uvicorn

    uvicorn.run(
        create_app(args.config),
        host=config.api.host,
        port=config.api.port,
    )


if __name__ == "__main__":
    main()
