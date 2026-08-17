from __future__ import annotations

import argparse
import asyncio
import hmac
from contextlib import asynccontextmanager
from datetime import datetime
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
from fastapi.responses import FileResponse, StreamingResponse
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
from .gas_cylinder import GasCylinderOptions
from .fishing_risk import (
    FishingRiskOptions,
    FishingRiskRuleOptions,
    FishingRiskScheduleOptions,
    FishingRiskZoneOptions,
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
from .stream_observability import (
    ObservedStreamManager,
    StreamLogStore,
    encode_sse,
)
from .vessel_detection import VesselDetectionOptions


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


class GasCylinderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    profile_id: str = Field(
        default="camera_01_ir",
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
    )
    analysis_fps: float = Field(default=1, ge=0.1, le=5)
    sample_count: int = Field(default=11, ge=3, le=15)
    sample_interval_seconds: float = Field(default=3, ge=0.2, le=10)
    minimum_confirmations: int = Field(default=4, ge=2, le=15)
    scene_stable_seconds: float = Field(default=3, ge=0.5, le=30)
    change_confirm_seconds: float = Field(default=3, ge=0.5, le=30)
    forced_refresh_seconds: float = Field(default=300, ge=30, le=86_400)
    retry_seconds: float = Field(default=10, ge=1, le=300)
    scene_change_ratio: float = Field(default=0.006, gt=0, le=0.5)
    motion_ratio: float = Field(default=0.003, gt=0, le=0.5)
    mask_nms_iou: float = Field(default=0.5, gt=0, le=1)
    temporal_match_iou: float = Field(default=0.3, gt=0, le=1)
    large_count_change: int = Field(default=2, ge=1, le=20)
    alarm_threshold: int = Field(
        default=18,
        ge=1,
        le=1_000,
        description="识别数量超过该值时统计背景和燃气瓶框显示为红色",
    )
    display_ids: bool = False
    include_partial: bool = True

    @model_validator(mode="after")
    def validate_confirmations(self) -> "GasCylinderRequest":
        if self.minimum_confirmations > self.sample_count:
            raise ValueError("minimum_confirmations不能大于sample_count")
        return self

    def to_options(self) -> GasCylinderOptions:
        options = GasCylinderOptions(**self.model_dump())
        options.validate()
        return options


class VesselDetectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: str | None = Field(
        default=None,
        description="独立船舶.pt模型；null时复用主模型权重",
    )
    analysis_fps: float = Field(default=5, ge=0.1, le=15)
    confidence: float = Field(
        default=0.10,
        gt=0,
        le=1,
        description="旁路原始候选阈值；时序确认会抑制低分误报",
    )
    iou: float = Field(default=0.45, gt=0, le=1)
    imgsz: int = Field(default=1280, ge=320, le=2048)
    class_ids: list[int] = Field(default_factory=lambda: [8], min_length=1)
    input_width: int = Field(default=1920, ge=320, le=3840)
    input_height: int = Field(default=1080, ge=180, le=2160)
    inference_regions: list[tuple[float, float, float, float]] = Field(
        default_factory=lambda: [(0.0, 0.0, 1.0, 1.0)],
        min_length=1,
        max_length=8,
        description="可重叠的[left,top,right,bottom]归一化推理分区",
    )
    roi: list[tuple[float, float]] | None = None
    exclude_rois: list[list[tuple[float, float]]] = Field(
        default_factory=list,
        max_length=16,
        description="已知桥墩、塔架等固定误报区域",
    )
    minimum_hits: int = Field(default=2, ge=1, le=10)
    hold_seconds: float = Field(default=1, ge=0.1, le=5)
    match_iou: float = Field(default=0.10, ge=0, le=1)
    maximum_center_distance: float = Field(default=1.5, ge=0.1, le=5)
    maximum_detections: int = Field(default=100, ge=1, le=500)
    duplicate_containment_threshold: float = Field(
        default=0.80,
        gt=0,
        le=1,
        description=(
            "跨推理分区框的小框被包含比例，用于合并同一艘船"
        ),
    )
    large_box_area_threshold: float = Field(default=0.20, gt=0, le=1)
    large_box_minimum_confidence: float = Field(default=0.25, gt=0, le=1)
    maximum_box_area: float = Field(default=1.0, gt=0, le=1)
    display_ids: bool = False
    display_roi: bool = True

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value != Path(value).name or not value.endswith(".pt"):
            raise ValueError("vessel_detection.model必须是.pt文件名")
        return value

    @field_validator("class_ids")
    @classmethod
    def validate_class_ids(cls, value: list[int]) -> list[int]:
        if any(item < 0 for item in value):
            raise ValueError("vessel_detection.class_ids不能为负数")
        return list(dict.fromkeys(value))

    @field_validator("inference_regions")
    @classmethod
    def validate_inference_regions(
        cls,
        value: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        for left, top, right, bottom in value:
            if not all(
                0 <= coordinate <= 1
                for coordinate in (left, top, right, bottom)
            ):
                raise ValueError("船舶推理分区坐标必须在[0, 1]范围内")
            if right <= left or bottom <= top:
                raise ValueError("船舶推理分区必须具有正面积")
        return list(dict.fromkeys(value))

    @field_validator("roi")
    @classmethod
    def validate_vessel_roi(
        cls,
        value: list[tuple[float, float]] | None,
    ) -> list[tuple[float, float]] | None:
        if value is None:
            return None
        return cls._validate_polygon(value)

    @field_validator("exclude_rois")
    @classmethod
    def validate_exclude_rois(
        cls,
        value: list[list[tuple[float, float]]],
    ) -> list[list[tuple[float, float]]]:
        return [cls._validate_polygon(item) for item in value]

    @staticmethod
    def _validate_polygon(
        value: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        serialized = ";".join(f"{x},{y}" for x, y in value)
        parsed = parse_roi(serialized)
        assert parsed is not None
        return list(parsed)

    def to_options(self) -> VesselDetectionOptions:
        options = VesselDetectionOptions(
            enabled=self.enabled,
            model=self.model,
            analysis_fps=self.analysis_fps,
            confidence=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            class_ids=tuple(self.class_ids),
            input_width=self.input_width,
            input_height=self.input_height,
            inference_regions=tuple(self.inference_regions),
            roi=(tuple(self.roi) if self.roi is not None else None),
            exclude_rois=tuple(
                tuple(polygon) for polygon in self.exclude_rois
            ),
            minimum_hits=self.minimum_hits,
            hold_seconds=self.hold_seconds,
            match_iou=self.match_iou,
            maximum_center_distance=self.maximum_center_distance,
            maximum_detections=self.maximum_detections,
            duplicate_containment_threshold=(
                self.duplicate_containment_threshold
            ),
            large_box_area_threshold=self.large_box_area_threshold,
            large_box_minimum_confidence=(
                self.large_box_minimum_confidence
            ),
            maximum_box_area=self.maximum_box_area,
            display_ids=self.display_ids,
            display_roi=self.display_roi,
        )
        options.validate()
        return options


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


class FishingRiskZoneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    polygon: list[tuple[float, float]]

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

    def to_options(self) -> FishingRiskZoneOptions:
        return FishingRiskZoneOptions(
            zone_id=self.id,
            polygon=tuple(self.polygon),
        )


class FishingRiskScheduleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    start_at: datetime
    end_at: datetime

    def to_options(self) -> FishingRiskScheduleOptions:
        return FishingRiskScheduleOptions(
            schedule_id=self.id,
            start_at=self.start_at,
            end_at=self.end_at,
        )


class FishingRiskRuleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_presence_seconds: float = Field(default=30, ge=1, le=86_400)
    loitering_seconds: float = Field(default=180, ge=5, le=86_400)
    loitering_radius_box_lengths: float = Field(default=4, ge=0.5, le=50)
    reversal_window_seconds: float = Field(default=120, ge=10, le=3_600)
    minimum_reversals: int = Field(default=2, ge=1, le=20)
    reversal_angle_degrees: float = Field(default=120, ge=60, le=180)
    minimum_motion_box_lengths: float = Field(default=0.5, ge=0.05, le=10)
    minimum_reversal_interval_seconds: float = Field(
        default=8,
        ge=1,
        le=300,
    )
    track_lost_seconds: float = Field(default=15, ge=1, le=300)
    track_match_box_lengths: float = Field(default=3, ge=0.5, le=20)
    startup_grace_seconds: float = Field(default=60, ge=0, le=3_600)
    preexisting_activation_box_lengths: float = Field(
        default=2,
        ge=0.5,
        le=50,
    )

    def to_options(self) -> FishingRiskRuleOptions:
        return FishingRiskRuleOptions(**self.model_dump())


class FishingRiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    timezone: str = "Asia/Shanghai"
    zones: list[FishingRiskZoneRequest] = Field(
        default_factory=list,
        max_length=16,
    )
    schedules: list[FishingRiskScheduleRequest] = Field(
        default_factory=list,
        max_length=32,
        description="空列表表示所有时间均参与风险分析",
    )
    rules: FishingRiskRuleRequest = Field(
        default_factory=FishingRiskRuleRequest
    )
    restricted_presence_score: int = Field(default=40, ge=0, le=100)
    loitering_score: int = Field(default=20, ge=0, le=100)
    direction_reversal_score: int = Field(default=20, ge=0, le=100)
    alert_score: int = Field(default=60, ge=1, le=100)
    cooldown_seconds: float = Field(default=300, ge=0, le=86_400)
    display_risk: bool = True
    webhook: EventWebhookRequest = Field(default_factory=EventWebhookRequest)

    @model_validator(mode="after")
    def validate_options(self) -> "FishingRiskRequest":
        self.to_options()
        return self

    def to_options(self) -> FishingRiskOptions:
        options = FishingRiskOptions(
            enabled=self.enabled,
            timezone=self.timezone,
            zones=tuple(item.to_options() for item in self.zones),
            schedules=tuple(item.to_options() for item in self.schedules),
            rules=self.rules.to_options(),
            restricted_presence_score=self.restricted_presence_score,
            loitering_score=self.loitering_score,
            direction_reversal_score=self.direction_reversal_score,
            alert_score=self.alert_score,
            cooldown_seconds=self.cooldown_seconds,
            display_risk=self.display_risk,
            webhook=self.webhook.to_options(),
        )
        options.validate()
        return options


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
    gas_cylinder: GasCylinderRequest = Field(
        default_factory=GasCylinderRequest,
        description="固定机位燃气瓶逐个识别和稳定计数配置",
    )
    vessel_detection: VesselDetectionRequest = Field(
        default_factory=VesselDetectionRequest,
        description="高分辨率、低延迟解耦的船舶检测旁路",
    )
    fishing_risk: FishingRiskRequest = Field(
        default_factory=FishingRiskRequest,
        description="仅凭监控轨迹生成疑似非法捕捞人工复核线索",
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

    @model_validator(mode="after")
    def validate_feature_dependencies(self) -> "StreamCreateRequest":
        if self.fishing_risk.enabled and not self.vessel_detection.enabled:
            raise ValueError(
                "启用fishing_risk前必须启用vessel_detection"
            )
        return self

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
            gas_cylinder=self.gas_cylinder.to_options(),
            vessel_detection=self.vessel_detection.to_options(),
            fishing_risk=self.fishing_risk.to_options(),
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
    gas_cylinder: dict[str, Any] | None = None
    vessel_detection: dict[str, Any] | None = None
    fishing_risk: dict[str, Any] | None = None


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


def _ensure_stream_or_log_exists(request: Request, stream_id: str) -> None:
    try:
        _manager(request).get(stream_id)
        return
    except StreamNotFoundError:
        pass
    try:
        exists = request.app.state.stream_logs.exists(stream_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not exists:
        raise HTTPException(
            status_code=404,
            detail="流任务和历史日志均不存在",
        )


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
        log_store = StreamLogStore(
            config.observability.log_root.expanduser().resolve(),
            max_file_bytes=config.observability.max_file_mb * 1024 * 1024,
            backup_count=config.observability.backup_count,
        )
        if config.observability.enabled:
            manager = ObservedStreamManager(
                manager,
                log_store,
                monitor_interval_seconds=(
                    config.observability.monitor_interval_seconds
                ),
                stale_after_seconds=(
                    config.observability.metrics_stale_seconds
                ),
                stall_fps=config.observability.stall_fps,
                degraded_fps_ratio=(
                    config.observability.degraded_fps_ratio
                ),
            )
        application.state.config = config
        application.state.manager = manager
        application.state.stream_logs = log_store
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

    @application.patch(
        "/v1/streams/{stream_id}/fishing-risk",
        response_model=StreamResponse,
        dependencies=[Depends(require_api_key)],
    )
    def update_fishing_risk(
        stream_id: str,
        payload: FishingRiskRequest,
        request: Request,
    ) -> dict[str, Any]:
        try:
            return _manager(request).update_fishing_risk(
                stream_id,
                payload.to_options(),
            )
        except StreamNotFoundError as exc:
            raise HTTPException(status_code=404, detail="流任务不存在") from exc
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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
        "/v1/streams/{stream_id}/logs",
        dependencies=[Depends(require_api_key)],
    )
    def list_stream_logs(
        stream_id: str,
        request: Request,
        limit: int = 200,
        after_sequence: int | None = None,
        level: str | None = None,
        event: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            _ensure_stream_or_log_exists(request, stream_id)
            return request.app.state.stream_logs.read(
                stream_id,
                limit=limit,
                after_sequence=after_sequence,
                level=level,
                event=event,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get(
        "/v1/streams/{stream_id}/logs/live",
        dependencies=[Depends(require_api_key)],
        response_class=StreamingResponse,
    )
    async def follow_stream_logs(
        stream_id: str,
        request: Request,
        after_sequence: int = 0,
    ) -> StreamingResponse:
        try:
            _ensure_stream_or_log_exists(request, stream_id)
            request.app.state.stream_logs.path_for(stream_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        async def event_stream():
            cursor = after_sequence
            while not await request.is_disconnected():
                entries = request.app.state.stream_logs.read(
                    stream_id,
                    limit=5_000,
                    after_sequence=cursor,
                )
                if entries:
                    for entry in entries:
                        cursor = max(cursor, int(entry["sequence"]))
                        yield encode_sse(entry)
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

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
