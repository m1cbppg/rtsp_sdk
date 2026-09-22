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
from .ground_litter_detection import (
    DEFAULT_MODEL as DEFAULT_GROUND_LITTER_MODEL,
    GroundLitterDetectionOptions,
    GroundLitterZone,
    PERSON_VEHICLE_CLASS_IDS,
    GROUND_CONTEXT_CLASS_IDS,
)
from .license_plate import DEFAULT_VEHICLE_CLASSES, LicensePlateOptions
from .ptz_verification import (
    PtzVerificationOptions,
    PtzVerificationRepository,
)
from .shared_stream_manager import SharedStreamManager
from .stream_manager import (
    ModelNotFoundError,
    NightVisionOptions,
    PtzControlUnavailableError,
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
    display_proposals: bool = Field(
        default=False,
        description="是否在转发画面绘制内部疑似小目标；默认仅供PTZ使用",
    )
    small_target_proposals: bool = Field(
        default=False,
        description=(
            "使用水面运动和局部外观生成未分类小目标，"
            "近景仍需船舶模型确认"
        ),
    )
    proposal_roi: list[tuple[float, float]] | None = Field(
        default=None,
        description="仅供疑似小目标生成使用的水面子区域",
    )
    proposal_background_alpha: float = Field(default=0.02, gt=0, le=0.5)
    proposal_threshold: int = Field(default=60, ge=1, le=255)
    proposal_appearance_enabled: bool = True
    proposal_appearance_threshold: int = Field(default=18, ge=1, le=255)
    proposal_appearance_blur_pixels: int = Field(default=31, ge=5, le=101)
    proposal_border_margin: float = Field(default=0.01, ge=0, le=0.10)
    proposal_minimum_area_pixels: int = Field(default=20, ge=1, le=100_000)
    proposal_maximum_area_pixels: int = Field(default=1_000, ge=1, le=1_000_000)
    proposal_minimum_width_pixels: int = Field(default=4, ge=1, le=1_000)
    proposal_minimum_height_pixels: int = Field(default=3, ge=1, le=1_000)
    proposal_minimum_fill_ratio: float = Field(default=0.20, ge=0, le=1)
    proposal_minimum_motion_ratio: float = Field(default=0.0, ge=0, le=1)
    proposal_maximum_candidates: int = Field(default=8, ge=1, le=100)
    proposal_global_change_ratio: float = Field(default=0.15, ge=0.01, le=1)

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

    @field_validator("roi", "proposal_roi")
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
            display_proposals=self.display_proposals,
            small_target_proposals=self.small_target_proposals,
            proposal_roi=(
                tuple(self.proposal_roi)
                if self.proposal_roi is not None
                else None
            ),
            proposal_background_alpha=self.proposal_background_alpha,
            proposal_threshold=self.proposal_threshold,
            proposal_appearance_enabled=self.proposal_appearance_enabled,
            proposal_appearance_threshold=(
                self.proposal_appearance_threshold
            ),
            proposal_appearance_blur_pixels=(
                self.proposal_appearance_blur_pixels
            ),
            proposal_border_margin=self.proposal_border_margin,
            proposal_minimum_area_pixels=(
                self.proposal_minimum_area_pixels
            ),
            proposal_maximum_area_pixels=(
                self.proposal_maximum_area_pixels
            ),
            proposal_minimum_width_pixels=(
                self.proposal_minimum_width_pixels
            ),
            proposal_minimum_height_pixels=(
                self.proposal_minimum_height_pixels
            ),
            proposal_minimum_fill_ratio=self.proposal_minimum_fill_ratio,
            proposal_minimum_motion_ratio=(
                self.proposal_minimum_motion_ratio
            ),
            proposal_maximum_candidates=self.proposal_maximum_candidates,
            proposal_global_change_ratio=(
                self.proposal_global_change_ratio
            ),
        )
        options.validate()
        return options


class GroundLitterZoneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(
        min_length=1,
        max_length=64,
        description="区域ID，例如merchant_01或left_walkway",
    )
    name: str | None = Field(
        default=None,
        max_length=128,
        description="区域显示名，例如门店01门前人行道",
    )
    polygon: list[tuple[float, float]] = Field(
        min_length=3,
        description="归一化地面多边形，顶点顺序任意",
    )
    exclude_zones: list[list[tuple[float, float]]] = Field(
        default_factory=list,
        max_length=16,
        description="该区域内已知固定物/棚体的排除多边形",
    )
    minimum_short_side_px: int = Field(
        default=12,
        ge=1,
        le=4096,
        description="原生像素最小短边；小于该值的框不显示",
    )
    minimum_box_area_px: int = Field(
        default=160,
        ge=1,
        le=16_777_216,
        description="原生像素最小框面积",
    )
    confidence: float | None = Field(default=None, gt=0, le=1, description="该区域白天最低置信度")
    night_confidence: float | None = Field(default=None, gt=0, le=1, description="该区域夜间最低置信度")

    @field_validator("polygon")
    @classmethod
    def validate_polygon(
        cls,
        value: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        return GroundLitterRequest._validate_polygon(value)

    @field_validator("exclude_zones")
    @classmethod
    def validate_exclude_zones(
        cls,
        value: list[list[tuple[float, float]]],
    ) -> list[list[tuple[float, float]]]:
        return [
            GroundLitterRequest._validate_polygon(item) for item in value
        ]

    def to_options(self) -> GroundLitterZone:
        zone = GroundLitterZone(
            region_id=self.region_id,
            polygon=tuple(self.polygon),
            name=self.name or self.region_id,
            exclude_zones=tuple(
                tuple(polygon) for polygon in self.exclude_zones
            ),
            minimum_short_side_px=self.minimum_short_side_px,
            minimum_box_area_px=self.minimum_box_area_px,
            confidence=self.confidence,
            night_confidence=self.night_confidence,
        )
        zone.validate()
        return zone


class GroundLitterRequest(BaseModel):
    """地面零散垃圾识别：原生像素分块推理 + 地面区域框显示。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: str = Field(
        default=DEFAULT_GROUND_LITTER_MODEL,
        description=(
            "垃圾模型.pt；相对models目录，可写litter/子目录，"
            "默认litter/turhancan_yolov8m_seg_trash.pt"
        ),
    )
    actor_model: str | None = Field(
        default=None,
        description=(
            "可选的人车遮挡模型；默认复用主链已跟踪目标，"
            "主模型类别被过滤时可设为yolo26s.pt"
        ),
    )
    analysis_fps: float = Field(default=1.0, ge=0.1, le=5)
    confidence: float = Field(
        default=0.20,
        gt=0,
        le=1,
        description="垃圾模型原始候选阈值；显示层会再要求多次命中",
    )
    night_confidence: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description="夜间阈值；仅在流的night_vision.enabled为true时生效",
    )
    tile_size_px: int = Field(
        default=640,
        ge=160,
        le=1920,
        description="原生像素裁剪边长；推理输入由inference_imgsz控制，未设时等于裁剪边长",
    )
    tile_overlap: float = Field(default=0.20, ge=0, lt=0.8)
    inference_imgsz: int | None = Field(
        default=None, ge=160, le=1920,
        description="垃圾模型输入尺寸；null沿用分块边长，320裁剪+640输入可放大候选",
    )
    local_actor_max_crops: int = Field(
        default=0, ge=0, le=8,
        description="每次分析局部人车复核裁剪上限；0关闭，需actor_model",
    )
    box_smoothing_alpha: float = Field(
        default=1.0, gt=0, le=1,
        description="显示框平滑系数；1不平滑，不影响原始候选关联",
    )
    nms_iou: float = Field(default=0.50, gt=0, le=1)
    maximum_tiles: int = Field(default=64, ge=1, le=128)
    actor_imgsz: int = Field(
        default=1280,
        ge=320,
        le=1920,
        description="仅当设置actor_model时使用的人车推理尺寸",
    )
    actor_confidence: float = Field(default=0.20, gt=0, le=1)
    actor_class_ids: list[int] = Field(
        default_factory=lambda: list(PERSON_VEHICLE_CLASS_IDS),
        min_length=1,
    )
    context_class_ids: list[int] = Field(
        default_factory=lambda: list(GROUND_CONTEXT_CLASS_IDS),
        min_length=0,
        description="地面遮挡物COCO类别，如雨伞/长椅/花盆；仅作为当前遮挡，不永久排除垃圾",
    )
    actor_overlap_threshold: float = Field(default=0.20, ge=0, le=1)
    zones: list[GroundLitterZoneRequest] = Field(
        default_factory=list,
        max_length=16,
        description="地面识别区域；enabled为true时至少一个",
    )
    overlay_exclude_zones: list[list[tuple[float, float]]] = Field(
        default_factory=list,
        max_length=16,
        description="全画面排除区，例如摄像头水印、固定棚体",
    )
    minimum_hits: int = Field(
        default=2,
        ge=1,
        le=20,
        description="窗口内至少命中次数才显示",
    )
    hit_window: int = Field(default=3, ge=1, le=20)
    hold_seconds: float = Field(
        default=3.0,
        ge=0.1,
        le=30,
        description="最后一次命中后保持显示的时间，用于消除闪烁",
    )
    maximum_age_seconds: float = Field(default=12.0, ge=0.1, le=120)
    maximum_boxes: int = Field(default=8, ge=1, le=64)
    display_zones: bool = Field(
        default=True,
        description="是否在输出画面绘制地面识别区域轮廓",
    )
    display_class: bool = Field(
        default=False,
        description="是否在标签后附加模型材质类别(Glass/Metal/...)",
    )
    display_confidence: bool = False
    label: str = Field(default="疑似垃圾", min_length=1, max_length=24)

    @staticmethod
    def _validate_polygon(
        value: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        serialized = ";".join(f"{x},{y}" for x, y in value)
        parsed = parse_roi(serialized)
        assert parsed is not None
        return list(parsed)

    @field_validator("overlay_exclude_zones")
    @classmethod
    def validate_overlay_exclude_zones(
        cls,
        value: list[list[tuple[float, float]]],
    ) -> list[list[tuple[float, float]]]:
        return [cls._validate_polygon(item) for item in value]

    @field_validator("actor_class_ids")
    @classmethod
    def validate_actor_class_ids(cls, value: list[int]) -> list[int]:
        if any(item < 0 for item in value):
            raise ValueError(
                "ground_litter.actor_class_ids不能为负数"
            )
        return list(dict.fromkeys(value))

    @field_validator("context_class_ids")
    @classmethod
    def validate_context_class_ids(cls, value: list[int]) -> list[int]:
        if any(item < 0 for item in value):
            raise ValueError("ground_litter.context_class_ids不能为负数")
        return list(dict.fromkeys(value))

    @field_validator("model", "actor_model")
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or not value.endswith(".pt"):
            raise ValueError("ground_litter模型必须是.pt文件")
        if value.startswith("/") or ".." in Path(value).parts:
            raise ValueError("ground_litter模型不能是绝对路径或越界路径")
        return value

    @model_validator(mode="after")
    def validate_zones(self) -> "GroundLitterRequest":
        if self.local_actor_max_crops and self.actor_model is None:
            raise ValueError("local_actor_max_crops需要actor_model")
        if not self.enabled:
            return self
        if not self.zones:
            raise ValueError("启用ground_litter时至少需要一个地面区域")
        if self.minimum_hits > self.hit_window:
            raise ValueError(
                "ground_litter.minimum_hits不能大于hit_window"
            )
        if self.maximum_age_seconds < self.hold_seconds:
            raise ValueError(
                "ground_litter.maximum_age_seconds不能小于hold_seconds"
            )
        return self

    def to_options(self) -> GroundLitterDetectionOptions:
        options = GroundLitterDetectionOptions(
            enabled=self.enabled,
            model=self.model,
            actor_model=self.actor_model,
            analysis_fps=self.analysis_fps,
            confidence=self.confidence,
            night_confidence=self.night_confidence,
            tile_size_px=self.tile_size_px,
            inference_imgsz=self.inference_imgsz,
            local_actor_max_crops=self.local_actor_max_crops,
            box_smoothing_alpha=self.box_smoothing_alpha,
            tile_overlap=self.tile_overlap,
            nms_iou=self.nms_iou,
            maximum_tiles=self.maximum_tiles,
            actor_imgsz=self.actor_imgsz,
            actor_confidence=self.actor_confidence,
            actor_class_ids=tuple(self.actor_class_ids),
            context_class_ids=tuple(self.context_class_ids),
            actor_overlap_threshold=self.actor_overlap_threshold,
            zones=tuple(zone.to_options() for zone in self.zones),
            overlay_exclude_zones=tuple(
                tuple(polygon) for polygon in self.overlay_exclude_zones
            ),
            minimum_hits=self.minimum_hits,
            hit_window=self.hit_window,
            hold_seconds=self.hold_seconds,
            maximum_age_seconds=self.maximum_age_seconds,
            maximum_boxes=self.maximum_boxes,
            display_zones=self.display_zones,
            display_class=self.display_class,
            display_confidence=self.display_confidence,
            label=self.label,
        )
        options.validate()
        return options


class PtzVerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "是否把船舶候选接入camera_control；"
            "false时仅识别并框选船舶，不创建摄像头控制客户端"
        ),
    )
    camera_id: str = Field(default="", max_length=128)
    camera_control_url: str = "http://camera-control:8080"
    camera_control_key_env: str = "CAMERA_CONTROL_API_KEY"
    trace_logging_enabled: bool = Field(
        default=False,
        description=(
            "是否为每个PTZ复核task输出独立JSONL控制trace；"
            "默认关闭，日志写入vessel-verifications/task-traces"
        ),
    )
    display_operation_log: bool = Field(
        default=False,
        description=(
            "是否在输出视频右上角显示PTZ状态和最近4条摄像头操作；"
            "不影响当前追踪船只的区别框选"
        ),
    )
    vessel_number_recognition_enabled: bool = Field(
        default=False,
        description="近景船舶达到目标尺寸后是否识别并显示船号",
    )
    vessel_number_fallback: str = Field(
        default="",
        max_length=32,
        description="OCR未得到可靠数字时使用的演示船号；空字符串表示不回退",
    )
    zoom_strategy: Literal["adaptive", "fixed"] = Field(
        default="adaptive",
        description="adaptive按重捕获船框闭环调节；fixed使用zoom_steps",
    )
    zoom_steps: list[int] = Field(default_factory=lambda: [4, 4], min_length=1, max_length=3)
    adaptive_target_width_ratio: float = Field(default=0.33, ge=0.03, le=0.8)
    adaptive_target_height_ratio: float = Field(default=0.33, ge=0.03, le=0.8)
    adaptive_min_step: int = Field(default=1, ge=1, le=16)
    adaptive_max_step: int = Field(default=6, ge=1, le=16)
    adaptive_max_rounds: int = Field(default=3, ge=1, le=6)
    adaptive_max_total_zoom_delta: int = Field(default=12, ge=1, le=48)
    adaptive_min_scale_growth_ratio: float = Field(default=1.12, ge=1, le=3)
    confirmed_target_fallback_zoom_rounds: int = Field(
        default=1,
        ge=0,
        le=2,
        description=(
            "已确认船舶在首次转动后短暂丢检时允许的保底中心变焦轮数；"
            "疑似目标不使用该策略"
        ),
    )
    confirmed_target_fallback_zoom_step: int = Field(
        default=3,
        ge=1,
        le=8,
        description="每轮保底中心变焦的相对步长，仍受总变焦上限约束",
    )
    command_timeout_seconds: float = Field(default=12, ge=2, le=60)
    reacquire_timeout_seconds: float = Field(default=4, ge=1, le=30)
    settle_seconds: float = Field(default=0.5, ge=0, le=5)
    maximum_off_home_seconds: float = Field(default=25, ge=5, le=120)
    capture_quality: int = Field(default=1, ge=1, le=6)
    evidence_validation_required: bool = Field(
        default=True,
        description="保存证据前是否对SDK返回JPEG再次执行船舶与清晰度校验",
    )
    evidence_capture_attempts: int = Field(default=2, ge=1, le=3)
    evidence_minimum_sharpness: float = Field(default=12.0, ge=0, le=10_000)
    evidence_target_scale_ratio: float = Field(default=0.70, ge=0.30, le=1.0)
    monitoring_interval_seconds: float = Field(default=0.25, ge=0.05, le=5)
    home_frame_delay_seconds: float = Field(default=1.5, ge=0, le=10)
    home_stable_frames: int = Field(
        default=2,
        ge=1,
        le=10,
        description="回HOME后至少连续接收多少个新全景分析帧才恢复ROI",
    )
    minimum_target_observations: int = Field(default=3, ge=1, le=10)
    primary_target_minimum_observations: int = Field(
        default=1,
        ge=1,
        le=10,
        description=(
            "主DeepStream检测器绿色船框触发PTZ前的最少观测次数；"
            "1表示首个绿框立即触发"
        ),
    )
    proposal_merge_radius: float = Field(default=0.04, ge=0, le=0.10)
    proposal_minimum_interval_seconds: float = Field(
        default=30,
        ge=0,
        le=3_600,
        description="两个疑似目标PTZ复核之间的最短间隔；已确认船舶不受限",
    )
    proposal_maximum_verifications_per_hour: int = Field(
        default=12,
        ge=1,
        le=3_600,
        description="每小时最多复核多少个疑似目标；已确认船舶不受限",
    )
    recent_target_seconds: float = Field(default=1_200, gt=0, le=86_400)
    confirmed_cooldown_seconds: float = Field(default=1_200, gt=0, le=86_400)
    negative_cooldown_seconds: float = Field(default=3_600, gt=0, le=604_800)
    lost_retry_seconds: float = Field(default=180, gt=0, le=86_400)
    dedup_base_radius: float = Field(default=0.018, gt=0, le=0.5)
    dedup_uncertainty_per_second: float = Field(default=0.0015, ge=0, le=0.05)
    dedup_maximum_radius: float = Field(default=0.08, gt=0, le=0.5)
    reacquire_strict_center_radius: float = Field(
        default=0.22,
        ge=0.05,
        le=0.75,
        description="PTZ动作后优先搜索的中央严格锁定半径",
    )
    reacquire_center_radius: float = Field(default=0.45, ge=0.05, le=0.75)
    reacquire_cluster_radius: float = Field(
        default=0.18,
        ge=0.02,
        le=0.30,
        description="近景中相邻小船合并为同一控制目标簇的中心距离",
    )
    continuous_tracking: bool = Field(
        default=False,
        description=(
            "近景确认和首次截图后是否持续锁定船舶；false保持截图后回HOME"
        ),
    )
    tracking_profile: Literal["standard", "demo_continuous"] = Field(
        default="standard",
        description=(
            "PTZ策略；demo_continuous从首个合格观测立即跟随，"
            "证据任务异步执行且活动会话不自动HOME"
        ),
    )
    tracking_center_deadband: float = Field(
        default=0.10,
        ge=0.03,
        le=0.30,
        description="船框中心允许偏离画面中心的半径，超出后才纠偏",
    )
    tracking_command_interval_seconds: float = Field(
        default=0.5,
        ge=0.1,
        le=5,
        description="持续跟踪期间相邻PTZ控制指令的最短间隔",
    )
    tracking_settle_seconds: float = Field(
        default=0.25,
        ge=0,
        le=2,
        description=(
            "持续追踪指令后的独立等待时间；与初次放大使用的settle_seconds分离"
        ),
    )
    tracking_recovery_enabled: bool = Field(
        default=False,
        description=(
            "持续追踪短暂丢框时是否先在最后位置逐档缩小视野重捕获，"
            "重捕获失败后才回HOME"
        ),
    )
    tracking_recovery_interval_seconds: float = Field(
        default=2.0,
        ge=0.5,
        le=15,
        description="丢框后相邻两次扩大视野重捕获的间隔",
    )
    tracking_recovery_zoom_out_step: int = Field(
        default=1,
        ge=1,
        le=4,
        description="每次重捕获使用的相对缩小步长",
    )
    tracking_recovery_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="一次连续丢框期间最多执行多少次扩大视野重捕获",
    )
    tracking_lost_timeout_seconds: float = Field(
        default=4,
        ge=1,
        le=30,
        description="持续未重识别到锁定船舶多久后结束跟踪并回HOME",
    )
    tracking_max_duration_seconds: float = Field(
        default=300,
        ge=0,
        le=3_600,
        description=(
            "单艘船最长持续跟踪时间；设为0表示不设时长上限，"
            "直到丢失目标、停流或手动中断"
        ),
    )
    tracking_zoom_hysteresis_ratio: float = Field(
        default=0.20,
        ge=0.05,
        le=0.50,
        description="目标尺寸相对设定值的缩放滞回比例，防止反复变倍",
    )
    tracking_zoom_step: int = Field(
        default=1,
        ge=1,
        le=4,
        description="持续跟踪时每次放大或缩小的相对步长",
    )
    tracking_initial_extra_zoom_step: int = Field(
        default=0,
        ge=0,
        le=4,
        description=(
            "进入持续跟踪和首次抓图前额外补充的相对放大步长；"
            "0保持原行为，1表示额外放大一档"
        ),
    )
    tracking_edge_guard_enabled: bool = Field(
        default=False,
        description="持续跟踪阶段启用边缘风险保护：不按尺寸缩小，只在可靠出画风险时减一档；关闭丢框盲缩小",
    )
    tracking_edge_response_seconds: float = Field(default=1, ge=.1, le=3)
    tracking_edge_motion_seconds: float = Field(default=.5, ge=.1, le=3)
    tracking_edge_uncertainty_seconds: float = Field(default=.25, ge=0, le=2)
    tracking_edge_cooldown_seconds: float = Field(default=8, ge=1, le=60)
    tracking_edge_stable_seconds: float = Field(default=3, ge=.5, le=15)
    tracking_edge_maximum_age_seconds: float = Field(default=.5, ge=.1, le=1)

    @model_validator(mode="after")
    def validate_adaptive_zoom(self) -> "PtzVerificationRequest":
        self.to_options()
        return self

    def to_options(self) -> PtzVerificationOptions:
        values = self.model_dump()
        values["zoom_steps"] = tuple(values["zoom_steps"])
        options = PtzVerificationOptions(**values)
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
    display_detections: bool = Field(
        default=True,
        description=(
            "是否在输出画面绘制普通检测框（人/车等）。关闭后只保留业务叠加，"
            "例如地面区域轮廓与垃圾框；跟踪、事件和旁路人车遮挡判定不受影响"
        ),
    )
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
    ground_litter: GroundLitterRequest = Field(
        default_factory=GroundLitterRequest,
        description="地面零散垃圾识别：原生像素分块+地面区域框显示",
    )
    fishing_risk: FishingRiskRequest = Field(
        default_factory=FishingRiskRequest,
        description="仅凭监控轨迹生成疑似非法捕捞人工复核线索",
    )
    ptz_verification: PtzVerificationRequest = Field(
        default_factory=PtzVerificationRequest,
        description="疑似小目标的PTZ放大、近景确认、回位和去重",
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
        if self.ptz_verification.enabled and not self.vessel_detection.enabled:
            raise ValueError(
                "启用ptz_verification前必须启用vessel_detection"
            )
        if self.ptz_verification.enabled:
            conflicts = []
            if self.license_plate.enabled:
                conflicts.append("license_plate")
            if self.event_detection.enabled:
                conflicts.append("event_detection")
            if self.gas_cylinder.enabled:
                conflicts.append("gas_cylinder")
            if self.ground_litter.enabled:
                # A moving camera invalidates reviewed ground regions.
                conflicts.append("ground_litter")
            if conflicts:
                raise ValueError(
                    "ptz_verification不能与固定视角功能同时启用: "
                    + ", ".join(conflicts)
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
            display_detections=self.display_detections,
            license_plate=self.license_plate.to_options(),
            night_vision=self.night_vision.to_options(),
            event_detection=self.event_detection.to_options(),
            gas_cylinder=self.gas_cylinder.to_options(),
            vessel_detection=self.vessel_detection.to_options(),
            ground_litter=self.ground_litter.to_options(),
            fishing_risk=self.fishing_risk.to_options(),
            ptz_verification=self.ptz_verification.to_options(),
        )


class StreamResponse(BaseModel):
    stream_id: str
    status: str
    rtsp_url: str
    model: str
    classes: list[int] | None
    created_at: str
    exit_code: int | None
    metrics: dict[str, float | int | str | bool | None] | None = None
    license_plate: dict[str, Any] | None = None
    night_vision: dict[str, Any] | None = None
    event_detection: dict[str, Any] | None = None
    gas_cylinder: dict[str, Any] | None = None
    vessel_detection: dict[str, Any] | None = None
    ground_litter: dict[str, Any] | None = None
    fishing_risk: dict[str, Any] | None = None
    ptz_verification: dict[str, Any] | None = None


class PtzReturnHomeResponse(BaseModel):
    stream_id: str
    request_id: str
    action: Literal["return_home"]
    status: Literal["accepted"]


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


def _ptz_verifications(request: Request) -> PtzVerificationRepository:
    return request.app.state.ptz_verifications


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
        application.state.ptz_verifications = PtzVerificationRepository(
            config.events.storage_root.expanduser().resolve()
            / "vessel-verifications"
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
        # MediaMTX 对 RTSP 上报 path="detected/<id>"（无前导斜杠），对 HLS/WebRTC 上报
        # "/detected/<id>"（带前导斜杠）且读取动作可能是 "play"。这里统一规范化路径，
        # 并把 "play" 视同读取，使浏览器端 HLS/WebRTC 读取也能通过鉴权。
        path_allowed = payload.path.lstrip("/").startswith("detected/")
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
            payload.action in ("read", "play")
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

    @application.post(
        "/v1/streams/{stream_id}/ptz/return-home",
        response_model=PtzReturnHomeResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_key)],
    )
    def return_ptz_home(
        stream_id: str,
        request: Request,
    ) -> dict[str, Any]:
        try:
            return _manager(request).return_ptz_home(stream_id)
        except StreamNotFoundError as exc:
            raise HTTPException(status_code=404, detail="流任务不存在") from exc
        except PtzControlUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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

    @application.get(
        "/v1/vessel-verifications",
        dependencies=[Depends(require_api_key)],
    )
    def list_vessel_verifications(
        request: Request,
        stream_id: str | None = None,
        result: str | None = None,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        try:
            return _ptz_verifications(request).list_jobs(
                stream_id=stream_id,
                result=result,
                after_sequence=after_sequence,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get(
        "/v1/vessel-verifications/live",
        dependencies=[Depends(require_api_key)],
        response_class=StreamingResponse,
    )
    async def follow_vessel_verifications(
        request: Request,
        stream_id: str | None = None,
        result: str | None = None,
        after_sequence: int = 0,
    ) -> StreamingResponse:
        async def event_stream():
            cursor = after_sequence
            while not await request.is_disconnected():
                entries = _ptz_verifications(request).list_jobs(
                    stream_id=stream_id,
                    result=result,
                    after_sequence=cursor,
                    limit=1_000,
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
        "/v1/vessel-verifications/{job_id}",
        dependencies=[Depends(require_api_key)],
    )
    def get_vessel_verification(
        job_id: str,
        request: Request,
    ) -> dict[str, Any]:
        try:
            return _ptz_verifications(request).get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="复核任务不存在") from exc

    @application.get(
        "/v1/vessel-verifications/{job_id}/images/{image_id}",
        dependencies=[Depends(require_api_key)],
        response_class=FileResponse,
    )
    def vessel_verification_image(
        job_id: str,
        image_id: str,
        request: Request,
    ) -> FileResponse:
        try:
            path, mime_type = _ptz_verifications(request).media_path(
                job_id,
                image_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="复核图片不存在") from exc
        return FileResponse(path, media_type=mime_type)

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
