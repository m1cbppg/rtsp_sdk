from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .deepstream_engine_builder import engine_path_for
from .events import GarbageAnalysisOptions
from .fishing_risk import FishingRiskOptions
from .gas_cylinder import (
    GasCylinderCameraProfile,
    GasCylinderOptions,
)
from .labels import load_label_map
from .ptz_verification import PtzVerificationOptions
from .stream_manager import (
    ManagerSettings,
    ModelNotFoundError,
    PtzControlUnavailableError,
    StreamCapacityError,
    StreamNotFoundError,
    StreamSpec,
    authenticated_rtsp_url,
)
from .ground_litter_detection import (
    DEFAULT_MODEL_SUBDIR as DEFAULT_GROUND_LITTER_SUBDIR,
    GroundLitterDetectionOptions as GroundLitterOptionsForApi,
)
from .vessel_detection import VesselDetectionOptions


ProcessFactory = Callable[..., Any]
LOGGER = logging.getLogger(__name__)
_WORKER_SHUTDOWN_OVERHEAD_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class DeepStreamManagerSettings:
    manager: ManagerSettings
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
    tracker_max_shadow_tracking_age: int = 15
    gpu_id: int = 0
    streams_per_group: int = 2
    model_input_size: int = 640
    mux_width: int = 1920
    mux_height: int = 1080
    batch_push_timeout_us: int = 20_000
    source_latency_ms: int = 300
    encoder_iframe_interval: int = 25
    stats_interval_seconds: float = 5.0
    minimum_healthy_fps: float = 20.0
    worker_stop_grace_seconds: float = 0.5
    lpr_model_root: Path = Path("/app/models/lpr")
    lpr_parser_library: Path = Path(
        "/opt/nvidia/deepstream/deepstream/lib/"
        "libnvdsinfer_custom_impl_lpr.so"
    )
    lpr_detector_batch_size: int = 16
    lpr_recognizer_batch_size: int = 16
    event_root: Path | None = None
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
    garbage_input_size: int = 640
    gas_cylinder_model_path: Path = Path(
        "/app/models/gas/yoloe-26l-seg.pt"
    )
    gas_cylinder_profile_root: Path = Path("/app/models/gas/profiles")
    gas_cylinder_input_width: int = 1280
    gas_cylinder_input_height: int = 720
    gas_cylinder_imgsz: int = 1280

    def validate(self) -> None:
        self.manager.validate()
        if self.streams_per_group != 2:
            raise RuntimeError("DeepStream当前固定为每个模型实例两路")
        if not self.onnx_root.is_dir():
            raise RuntimeError(f"ONNX模型目录不存在: {self.onnx_root}")
        if not self.parser_library.is_file():
            raise RuntimeError(
                f"DeepStream YOLO解析库不存在: {self.parser_library}"
            )
        if not self.tracker_library.is_file():
            raise RuntimeError(
                f"DeepStream跟踪库不存在: {self.tracker_library}"
            )
        if not self.tracker_config.is_file():
            raise RuntimeError(
                f"DeepStream跟踪配置不存在: {self.tracker_config}"
            )
        if not 1 <= self.tracker_max_shadow_tracking_age <= 200:
            raise RuntimeError(
                "tracker_max_shadow_tracking_age必须在1到200之间"
            )
        if self.gpu_id < 0:
            raise RuntimeError("gpu_id不能为负数")
        if self.model_input_size <= 0:
            raise RuntimeError("model_input_size必须大于0")
        if self.mux_width <= 0 or self.mux_height <= 0:
            raise RuntimeError("mux尺寸必须大于0")
        if self.batch_push_timeout_us <= 0:
            raise RuntimeError("batch_push_timeout_us必须大于0")
        if self.source_latency_ms < 0:
            raise RuntimeError("source_latency_ms不能为负数")
        if self.encoder_iframe_interval <= 0:
            raise RuntimeError("encoder_iframe_interval必须大于0")
        if self.stats_interval_seconds <= 0:
            raise RuntimeError("stats_interval_seconds必须大于0")
        if self.minimum_healthy_fps <= 0:
            raise RuntimeError("minimum_healthy_fps必须大于0")
        if not 0 <= self.worker_stop_grace_seconds <= 5:
            raise RuntimeError(
                "worker_stop_grace_seconds必须在0到5秒之间"
            )
        if self.lpr_detector_batch_size <= 0:
            raise RuntimeError("lpr_detector_batch_size必须大于0")
        if self.lpr_recognizer_batch_size <= 0:
            raise RuntimeError("lpr_recognizer_batch_size必须大于0")
        if self.garbage_input_size <= 0:
            raise RuntimeError("garbage_input_size必须大于0")
        if self.gas_cylinder_input_width <= 0:
            raise RuntimeError("gas_cylinder_input_width必须大于0")
        if self.gas_cylinder_input_height <= 0:
            raise RuntimeError("gas_cylinder_input_height必须大于0")
        if self.gas_cylinder_imgsz <= 0:
            raise RuntimeError("gas_cylinder_imgsz必须大于0")


@dataclass(slots=True)
class DeepStreamRecord:
    stream_id: str
    group_id: str
    path: str
    model: str
    classes: tuple[int, ...] | None
    created_at: datetime
    public_url: str
    spec: StreamSpec
    internal_url: str


@dataclass(slots=True)
class DeepStreamGroup:
    group_id: str
    model: str
    imgsz: int
    stream_ids: list[str]
    process: Any | None = None
    generation: int = 0
    started_monotonic: float = 0.0
    config_path: Path | None = None
    metrics_path: Path | None = None
    license_plate_enabled: bool = False
    garbage_enabled: bool = False
    garbage_mode: str = "items"
    gas_cylinder_enabled: bool = False
    gas_cylinder_profile_id: str | None = None
    vessel_signature: tuple[bool, str, int, int, int] = (
        False,
        "",
        0,
        0,
        0,
    )
    ground_litter_signature: tuple[bool, str, int, int, int, int, float, str] = (
        False,
        "",
        0,
        0,
        0,
        0,
        1.0,
        "",
    )
    night_vision_signature: tuple[bool, float, float] = (
        False,
        1.0,
        0.30,
    )


class DeepStreamStreamManager:
    """Runs one zero-copy DeepStream/TensorRT pipeline for each stream pair."""

    def __init__(
        self,
        settings: DeepStreamManagerSettings,
        process_factory: ProcessFactory = subprocess.Popen,
    ) -> None:
        settings.validate()
        self._settings = settings
        self._event_root = (
            settings.event_root
            if settings.event_root is not None
            else settings.runtime_root.parent / "events"
        )
        self._process_factory = process_factory
        self._records: dict[str, DeepStreamRecord] = {}
        self._groups: dict[str, DeepStreamGroup] = {}
        self._lock = threading.RLock()
        settings.engine_root.mkdir(parents=True, exist_ok=True)
        settings.runtime_root.mkdir(parents=True, exist_ok=True)
        self._event_root.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[str]:
        return sorted(
            path.name
            for path in self._settings.manager.model_root.glob("*.pt")
            if path.is_file()
            and (self._settings.onnx_root / f"{path.stem}.onnx").is_file()
        )

    def create(self, spec: StreamSpec) -> dict[str, Any]:
        spec.night_vision.validate()
        spec.event_detection.validate()
        spec.gas_cylinder.validate()
        spec.vessel_detection.validate()
        spec.ground_litter.validate()
        spec.fishing_risk.validate()
        spec.ptz_verification.validate()
        if spec.fishing_risk.enabled and not spec.vessel_detection.enabled:
            raise ModelNotFoundError(
                "启用fishing_risk前必须启用vessel_detection"
            )
        if spec.ptz_verification.enabled and not spec.vessel_detection.enabled:
            raise ModelNotFoundError(
                "启用ptz_verification前必须启用vessel_detection"
            )
        if spec.ptz_verification.enabled:
            conflicts = []
            if spec.license_plate.enabled:
                conflicts.append("license_plate")
            if spec.event_detection.enabled:
                conflicts.append("event_detection")
            if spec.gas_cylinder.enabled:
                conflicts.append("gas_cylinder")
            if spec.ground_litter.enabled:
                conflicts.append("ground_litter")
            if conflicts:
                raise ModelNotFoundError(
                    "ptz_verification不能与固定视角功能同时启用: "
                    + ", ".join(conflicts)
                )
        model_path, onnx_path, labels_path = self._resolve_model(spec.model)
        del onnx_path
        if spec.imgsz != self._settings.model_input_size:
            raise ModelNotFoundError(
                "DeepStream模型固定输入尺寸为"
                f"{self._settings.model_input_size}，请求imgsz={spec.imgsz}"
            )
        if spec.output_fps is not None:
            raise ModelNotFoundError(
                "DeepStream后端固定跟随源流FPS，请不要传output_fps"
            )
        stream_id = uuid.uuid4().hex
        path = f"detected/{stream_id}"
        internal_url = authenticated_rtsp_url(
            self._settings.manager.internal_rtsp_base_url,
            self._settings.manager.publish_user,
            self._settings.manager.publish_password,
            path,
        )
        public_url = authenticated_rtsp_url(
            self._settings.manager.public_rtsp_base_url,
            self._settings.manager.read_user,
            self._settings.manager.read_password,
            path,
        )

        with self._lock:
            if len(self._records) >= self._settings.manager.max_streams:
                raise StreamCapacityError(
                    "已达到并发上限"
                    f"MAX_STREAMS={self._settings.manager.max_streams}"
                )
            if spec.ptz_verification.enabled and any(
                record.spec.ptz_verification.enabled
                and record.spec.ptz_verification.camera_id
                == spec.ptz_verification.camera_id
                for record in self._records.values()
            ):
                raise StreamCapacityError(
                    "同一camera_id只能绑定一个活动PTZ复核流"
                )
            if spec.license_plate.enabled:
                self._validate_lpr_assets()
            if spec.event_detection.enabled:
                self._validate_event_classes(spec, labels_path)
            if spec.event_detection.garbage.enabled:
                self._validate_garbage_assets(spec.event_detection.garbage)
            if spec.gas_cylinder.enabled:
                self._validate_gas_cylinder_assets(spec.gas_cylinder)
            if spec.vessel_detection.enabled:
                self._validate_vessel_assets(
                    spec.vessel_detection,
                    primary_model=model_path.name,
                    primary_labels_path=labels_path,
                )
            if spec.ground_litter.enabled:
                self._validate_ground_litter_assets(spec.ground_litter)
            vessel_signature = self._vessel_signature(
                spec,
                primary_model=model_path.name,
            )
            ground_litter_signature = self._ground_litter_signature(spec)
            group = self._find_group(
                model_path.name,
                spec.imgsz,
                spec.license_plate.enabled,
                spec.night_vision.group_signature(
                    include_plate_detector=spec.license_plate.enabled,
                ),
                spec.event_detection.garbage.enabled,
                spec.event_detection.garbage.detection_mode,
                spec.gas_cylinder.enabled,
                spec.gas_cylinder.profile_id,
                vessel_signature,
                ground_litter_signature,
            )
            if group is None:
                group = DeepStreamGroup(
                    group_id=uuid.uuid4().hex,
                    model=model_path.name,
                    imgsz=spec.imgsz,
                    stream_ids=[],
                    license_plate_enabled=spec.license_plate.enabled,
                    garbage_enabled=spec.event_detection.garbage.enabled,
                    garbage_mode=(
                        spec.event_detection.garbage.detection_mode
                    ),
                    gas_cylinder_enabled=spec.gas_cylinder.enabled,
                    gas_cylinder_profile_id=(
                        spec.gas_cylinder.profile_id
                        if spec.gas_cylinder.enabled
                        else None
                    ),
                    vessel_signature=vessel_signature,
                    ground_litter_signature=ground_litter_signature,
                    night_vision_signature=(
                        spec.night_vision.group_signature(
                            include_plate_detector=(
                                spec.license_plate.enabled
                            ),
                        )
                    ),
                )
                self._groups[group.group_id] = group
            record = DeepStreamRecord(
                stream_id=stream_id,
                group_id=group.group_id,
                path=path,
                model=model_path.name,
                classes=spec.classes,
                created_at=datetime.now(timezone.utc),
                public_url=public_url,
                spec=spec,
                internal_url=internal_url,
            )
            self._records[stream_id] = record
            group.stream_ids.append(stream_id)
            try:
                self._restart_group(group)
            except BaseException:
                group.stream_ids.remove(stream_id)
                self._records.pop(stream_id, None)
                if group.stream_ids:
                    try:
                        self._restart_group(group)
                    except BaseException:
                        LOGGER.exception(
                            "恢复DeepStream组失败: %s",
                            group.group_id,
                        )
                else:
                    self._cleanup_group_files(group)
                    self._groups.pop(group.group_id, None)
                raise
            return self._serialize(record)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._serialize(record)
                for record in list(self._records.values())
            ]

    def get(self, stream_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(stream_id)
            if record is None:
                raise StreamNotFoundError(stream_id)
            return self._serialize(record)

    def update_fishing_risk(
        self,
        stream_id: str,
        options: FishingRiskOptions,
    ) -> dict[str, Any]:
        options.validate()
        with self._lock:
            record = self._records.get(stream_id)
            if record is None:
                raise StreamNotFoundError(stream_id)
            if options.enabled and not record.spec.vessel_detection.enabled:
                raise ModelNotFoundError(
                    "启用fishing_risk前必须启用vessel_detection"
                )
            previous_spec = record.spec
            record.spec = replace(previous_spec, fishing_risk=options)
            group = self._groups[record.group_id]
            try:
                self._restart_group(group)
            except BaseException:
                record.spec = previous_spec
                try:
                    self._restart_group(group)
                except BaseException:
                    LOGGER.exception(
                        "恢复疑似捕捞配置失败: stream=%s",
                        stream_id,
                    )
                raise
            return self._serialize(record)

    def return_ptz_home(self, stream_id: str) -> dict[str, Any]:
        """Atomically enqueue an emergency HOME command for one worker stream."""
        with self._lock:
            record = self._records.get(stream_id)
            if record is None:
                raise StreamNotFoundError(stream_id)
            if not record.spec.ptz_verification.enabled:
                raise PtzControlUnavailableError(
                    "该流未启用ptz_verification"
                )
            group = self._groups[record.group_id]
            if group.process is None or group.process.poll() is not None:
                raise PtzControlUnavailableError("DeepStream worker未运行")
            request_id = self._enqueue_ptz_home(group, stream_id)
            return {
                "stream_id": stream_id,
                "request_id": request_id,
                "action": "return_home",
                "status": "accepted",
            }

    def _enqueue_ptz_home(
        self,
        group: DeepStreamGroup,
        stream_id: str,
    ) -> str:
        request_id = uuid.uuid4().hex
        control_dir = self._settings.runtime_root / group.group_id / "control"
        control_dir.mkdir(parents=True, exist_ok=True)
        command_path = control_dir / (
            f"ptz-home-{stream_id}-{request_id}.json"
        )
        temporary = control_dir / f".{request_id}.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "stream_id": stream_id,
                    "action": "return_home",
                    "created_at_unix": time.time(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(command_path)
        return request_id

    def _return_ptz_home_before_stop(
        self,
        record: DeepStreamRecord,
        group: DeepStreamGroup,
    ) -> bool:
        """Ask the lease-owning worker to HOME and wait for its acknowledgement."""
        process = group.process
        if process is None or process.poll() is not None:
            return False
        # A missing metrics file means the worker has not completed startup;
        # the regular SIGTERM shutdown path remains the fallback in that case.
        if self._read_metrics(group, record.stream_id) is None:
            return False
        request_id = self._enqueue_ptz_home(group, record.stream_id)
        options = record.spec.ptz_verification
        deadline = time.monotonic() + (
            options.command_timeout_seconds * 2.0
            + options.home_frame_delay_seconds
            + self._settings.stats_interval_seconds * 2.0
            + 2.0
        )
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False
            metrics = self._read_metrics(group, record.stream_id)
            if (
                metrics is not None
                and metrics.get("ptz_last_return_home_request_id")
                == request_id
                and bool(metrics.get("ptz_manual_hold"))
                and metrics.get("ptz_verification_state") == "manual_hold"
            ):
                return True
            time.sleep(0.1)
        LOGGER.error(
            "删除流前等待PTZ回HOME超时: stream=%s camera=%s request=%s",
            record.stream_id,
            options.camera_id,
            request_id,
        )
        return False

    def stop(self, stream_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(stream_id)
            if record is None:
                raise StreamNotFoundError(stream_id)
            group = self._groups[record.group_id]
            if record.spec.ptz_verification.enabled:
                self._return_ptz_home_before_stop(record, group)
            self._records.pop(stream_id, None)
            group.stream_ids.remove(stream_id)
            stopped_ptz_options = (
                (record.spec.ptz_verification,)
                if record.spec.ptz_verification.enabled
                else ()
            )
            if group.stream_ids:
                self._restart_group(
                    group,
                    stopping_ptz_options=stopped_ptz_options,
                )
            else:
                self._stop_group(
                    group,
                    stopping_ptz_options=stopped_ptz_options,
                )
                self._groups.pop(group.group_id, None)
            result = self._serialize(record, forced_status="stopped")
            result["exit_code"] = 0
            return result

    def shutdown(self) -> None:
        with self._lock:
            groups = [
                (group, self._group_stop_grace_seconds(group))
                for group in self._groups.values()
            ]
            self._groups.clear()
            self._records.clear()
        for group, grace_seconds in groups:
            self._stop_group(group, grace_seconds=grace_seconds)

    def _find_group(
        self,
        model: str,
        imgsz: int,
        license_plate_enabled: bool,
        night_vision_signature: tuple[bool, float, float],
        garbage_enabled: bool,
        garbage_mode: str,
        gas_cylinder_enabled: bool,
        gas_cylinder_profile_id: str,
        vessel_signature: tuple[bool, str, int, int, int],
        ground_litter_signature: tuple[bool, str, int, int, int, int, float, str],
    ) -> DeepStreamGroup | None:
        candidates = (
            group
            for group in self._groups.values()
            if group.model == model
            and group.imgsz == imgsz
            and group.license_plate_enabled == license_plate_enabled
            and group.night_vision_signature == night_vision_signature
            and group.garbage_enabled == garbage_enabled
            and group.garbage_mode == garbage_mode
            and group.gas_cylinder_enabled == gas_cylinder_enabled
            and group.gas_cylinder_profile_id
            == (
                gas_cylinder_profile_id
                if gas_cylinder_enabled
                else None
            )
            and group.vessel_signature == vessel_signature
            and group.ground_litter_signature == ground_litter_signature
            and len(group.stream_ids) < self._settings.streams_per_group
        )
        return min(candidates, key=lambda item: item.group_id, default=None)

    def _validate_lpr_assets(self) -> None:
        required = (
            self._settings.lpr_model_root
            / "LPDNet_CCPD_pruned_tao5.onnx",
            self._settings.lpr_model_root
            / "ch_lprnet_baseline18_deployable.onnx",
            self._settings.lpr_model_root / "ch_lp_characters.txt",
            self._settings.lpr_parser_library,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ModelNotFoundError(
                "中国车牌识别资源缺失: " + ", ".join(missing)
            )

    def _validate_garbage_assets(
        self,
        options: GarbageAnalysisOptions,
    ) -> None:
        if options.detection_mode == "pile":
            onnx_path = self._settings.garbage_pile_onnx_path
            labels_path = self._settings.garbage_pile_labels_path
        else:
            onnx_path = self._settings.garbage_onnx_path
            labels_path = self._settings.garbage_labels_path
        required = (
            onnx_path,
            labels_path,
            self._settings.garbage_parser_library,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ModelNotFoundError(
                "垃圾识别资源缺失: " + ", ".join(missing)
            )
        labels = {
            item.strip()
            for item in labels_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if item.strip()
        }
        if options.detection_mode == "pile":
            if "garbage" not in {item.lower() for item in labels}:
                raise ModelNotFoundError(
                    "垃圾堆模型标签中缺少garbage类别"
                )
            return
        unsupported = sorted(set(options.prompts) - labels)
        if unsupported:
            raise ModelNotFoundError(
                "垃圾模型不支持这些类别: " + ", ".join(unsupported)
            )

    def _validate_gas_cylinder_assets(
        self,
        options: GasCylinderOptions,
    ) -> None:
        if not self._settings.gas_cylinder_model_path.is_file():
            raise ModelNotFoundError(
                "燃气瓶YOLOE模型不存在: "
                f"{self._settings.gas_cylinder_model_path}"
            )
        try:
            GasCylinderCameraProfile.load(
                self._settings.gas_cylinder_profile_root,
                options.profile_id,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ModelNotFoundError(str(exc)) from exc

    def _validate_vessel_assets(
        self,
        options: VesselDetectionOptions,
        *,
        primary_model: str,
        primary_labels_path: Path,
    ) -> None:
        model_name = options.model or primary_model
        model_path = self._resolve_pt_model(model_name)
        if options.input_width > self._settings.mux_width or (
            options.input_height > self._settings.mux_height
        ):
            raise ModelNotFoundError(
                "船舶旁路输入尺寸不能超过DeepStream mux尺寸"
                f"{self._settings.mux_width}x{self._settings.mux_height}；"
                "提高摄像头分辨率时需同步提高DEEPSTREAM_MUX_WIDTH/HEIGHT"
            )
        if model_path.name != primary_model:
            return
        label_count = sum(
            1
            for item in primary_labels_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if item.strip()
        )
        invalid = sorted(
            class_id
            for class_id in options.class_ids
            if class_id >= label_count
        )
        if invalid:
            raise ModelNotFoundError(
                "船舶类别超出模型范围: "
                + ", ".join(str(item) for item in invalid)
            )

    def _resolve_pt_model(self, model: str) -> Path:
        if not model or Path(model).name != model or not model.endswith(".pt"):
            raise ModelNotFoundError(
                "船舶模型必须是models目录中的.pt文件名"
            )
        root = self._settings.manager.model_root.resolve()
        path = (root / model).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ModelNotFoundError("船舶模型路径越界") from exc
        if not path.is_file():
            raise ModelNotFoundError(f"船舶模型不存在: {model}")
        return path

    @staticmethod
    def _vessel_signature(
        spec: StreamSpec,
        *,
        primary_model: str,
    ) -> tuple[bool, str, int, int, int]:
        options = spec.vessel_detection
        if not options.enabled:
            return (False, "", 0, 0, 0)
        return (
            True,
            options.model or primary_model,
            options.imgsz,
            options.input_width,
            options.input_height,
        )

    @staticmethod
    def _ground_litter_signature(
        spec: StreamSpec,
    ) -> tuple[bool, str, int, int, int, int, float, str]:
        options = spec.ground_litter
        if not options.enabled:
            return (False, "", 0, 0, 0, 0, 1.0, "")
        return (
            True,
            str(options.model),
            int(options.tile_size_px),
            int(options.inference_imgsz or 0),
            int(options.local_actor_max_crops),
            float(options.box_smoothing_alpha),
            int(options.actor_imgsz),
            str(options.actor_model or ""),
        )

    def _resolve_ground_litter_model(self, model: str) -> Path:
        """Resolve a litter ``.pt`` including the optional ``litter/`` prefix."""
        root = self._settings.manager.model_root.resolve()
        requested = Path(str(model))
        if requested.is_absolute() or ".." in requested.parts:
            raise ModelNotFoundError("零散垃圾模型路径越界")
        candidates = (
            [root / requested]
            if requested.parent != Path(".")
            else [
                root / DEFAULT_GROUND_LITTER_SUBDIR / requested,
                root / requested,
            ]
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            if resolved.is_file():
                return resolved
        raise ModelNotFoundError(f"零散垃圾模型不存在: {model}")

    def _validate_ground_litter_assets(
        self,
        options: GroundLitterOptionsForApi,
    ) -> None:
        self._resolve_ground_litter_model(options.model)
        if options.actor_model is not None:
            self._resolve_ground_litter_model(options.actor_model)

    @staticmethod
    def _validate_event_classes(spec: StreamSpec, labels_path: Path) -> None:
        label_count = sum(
            1
            for item in labels_path.read_text(encoding="utf-8").splitlines()
            if item.strip()
        )
        class_ids = (
            spec.event_detection.person_classes
            + spec.event_detection.vehicle_classes
        )
        invalid = sorted({item for item in class_ids if item >= label_count})
        if invalid:
            raise ModelNotFoundError(
                "事件类别超出主模型范围: "
                + ", ".join(str(item) for item in invalid)
            )

    def _resolve_model(self, model: str) -> tuple[Path, Path, Path]:
        if not model or Path(model).name != model or not model.endswith(".pt"):
            raise ModelNotFoundError(
                "model必须是models目录中的.pt文件名，不能包含路径"
            )
        model_root = self._settings.manager.model_root.resolve()
        model_path = (model_root / model).resolve()
        try:
            model_path.relative_to(model_root)
        except ValueError as exc:
            raise ModelNotFoundError("模型路径越界") from exc
        if not model_path.is_file():
            raise ModelNotFoundError(f"模型不存在: {model}")
        onnx_path = self._settings.onnx_root / f"{model_path.stem}.onnx"
        labels_path = (
            self._settings.onnx_root / f"{model_path.stem}.labels.txt"
        )
        if not onnx_path.is_file():
            raise ModelNotFoundError(
                f"DeepStream模型未导出: {onnx_path.name}；"
                "请先运行scripts/export_deepstream_models.sh"
            )
        if not labels_path.is_file():
            raise ModelNotFoundError(
                f"DeepStream标签文件不存在: {labels_path.name}"
            )
        return model_path, onnx_path, labels_path

    def _restart_group(
        self,
        group: DeepStreamGroup,
        *,
        stopping_ptz_options: tuple[PtzVerificationOptions, ...] = (),
    ) -> None:
        previous_generation = group.generation
        group.generation = previous_generation + 1
        group_dir = self._settings.runtime_root / group.group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        group.config_path = group_dir / "worker.json"
        group.metrics_path = group_dir / "metrics.json"
        control_dir = group_dir / "control"
        control_dir.mkdir(parents=True, exist_ok=True)
        try:
            payload = self._worker_payload(group)
        except BaseException:
            group.generation = previous_generation
            raise
        self._stop_process(
            group.process,
            grace_seconds=self._group_stop_grace_seconds(
                group,
                extra_ptz_options=stopping_ptz_options,
            ),
        )
        group.process = None
        group.metrics_path.unlink(missing_ok=True)
        for command_path in control_dir.iterdir():
            if command_path.is_file():
                command_path.unlink(missing_ok=True)
        temporary = group_dir / "worker.json.tmp"
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(group.config_path)
        process = self._process_factory(
            [
                sys.executable,
                "-m",
                "rtsp_annotator.deepstream_worker",
                "--config",
                str(group.config_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        group.process = process
        group.started_monotonic = time.monotonic()

    def _worker_payload(self, group: DeepStreamGroup) -> dict[str, Any]:
        records = [self._records[item] for item in group.stream_ids]
        model_stem = Path(group.model).stem
        labels_path = self._settings.onnx_root / f"{model_stem}.labels.txt"
        label_count = len(
            [
                line
                for line in labels_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
        )
        if label_count <= 0:
            raise ModelNotFoundError(f"标签文件为空: {labels_path}")
        label_map = load_label_map(self._settings.manager.label_map_path)
        night_vision = records[0].spec.night_vision
        payload: dict[str, Any] = {
            "version": 1,
            "group_id": group.group_id,
            "generation": group.generation,
            "gpu_id": self._settings.gpu_id,
            # The TensorRT engine has a dynamic 1..2 profile. Capacity two
            # does not mean an underfilled single-stream mux should be forced
            # to wait for batch two.
            "batch_size": len(records),
            "mux_width": self._settings.mux_width,
            "mux_height": self._settings.mux_height,
            "batch_push_timeout_us": self._settings.batch_push_timeout_us,
            "source_latency_ms": self._settings.source_latency_ms,
            "encoder_iframe_interval": (
                self._settings.encoder_iframe_interval
            ),
            "stats_interval_seconds": self._settings.stats_interval_seconds,
            "minimum_healthy_fps": self._settings.minimum_healthy_fps,
            "onnx_path": str(
                self._settings.onnx_root / f"{model_stem}.onnx"
            ),
            "engine_path": str(
                engine_path_for(
                    self._settings.onnx_root / f"{model_stem}.onnx",
                    self._settings.engine_root,
                    imgsz=group.imgsz,
                    batch_size=self._settings.streams_per_group,
                    gpu_id=self._settings.gpu_id,
                )
            ),
            "labels_path": str(labels_path),
            "label_count": label_count,
            "parser_library": str(self._settings.parser_library),
            "tracker_library": str(self._settings.tracker_library),
            "tracker_config": str(self._settings.tracker_config),
            "tracker_max_shadow_tracking_age": (
                self._settings.tracker_max_shadow_tracking_age
            ),
            "imgsz": group.imgsz,
            "metrics_path": str(group.metrics_path),
            "control_dir": str(
                self._settings.runtime_root / group.group_id / "control"
            ),
            "label_map": label_map,
            "night_vision": {
                "enabled": night_vision.enabled,
                "input_gain": night_vision.input_gain,
                "plate_detector_confidence": (
                    night_vision.plate_detector_confidence
                ),
            },
            "streams": [
                {
                    "stream_id": record.stream_id,
                    "input_url": record.spec.input_url,
                    "output_url": record.internal_url,
                    "classes": (
                        list(record.spec.classes)
                        if record.spec.classes is not None
                        else None
                    ),
                    "conf": (
                        record.spec.night_vision.confidence
                        if record.spec.night_vision.enabled
                        else record.spec.conf
                    ),
                    "day_conf": record.spec.conf,
                    "night_vision": {
                        "enabled": record.spec.night_vision.enabled,
                        "confidence": (
                            record.spec.night_vision.confidence
                        ),
                        "input_gain": (
                            record.spec.night_vision.input_gain
                        ),
                        "plate_detector_confidence": (
                            record.spec.night_vision
                            .plate_detector_confidence
                        ),
                    },
                    "roi": (
                        [list(point) for point in record.spec.roi]
                        if record.spec.roi is not None
                        else None
                    ),
                    "display_detections": bool(
                        record.spec.display_detections
                    ),
                    "bitrate_bps": _parse_bitrate(record.spec.bitrate),
                    "license_plate": {
                        "enabled": record.spec.license_plate.enabled,
                        "detector_interval": (
                            record.spec.license_plate.detector_interval
                        ),
                        "recognition_reinfer_interval": (
                            record.spec.license_plate
                            .recognition_reinfer_interval
                        ),
                        "minimum_confirmations": (
                            record.spec.license_plate.minimum_confirmations
                        ),
                        "minimum_plate_confidence": (
                            record.spec.license_plate
                            .minimum_plate_confidence
                        ),
                        "vehicle_classes": list(
                            record.spec.license_plate.vehicle_classes
                        ),
                    },
                    "event_detection": record.spec.event_detection.to_payload(),
                    "gas_cylinder": record.spec.gas_cylinder.to_payload(),
                    "vessel_detection": (
                        record.spec.vessel_detection.to_payload()
                    ),
                    "ground_litter": record.spec.ground_litter.to_payload(),
                    "fishing_risk": record.spec.fishing_risk.to_payload(),
                    "ptz_verification": (
                        record.spec.ptz_verification.to_payload()
                    ),
                    "event_root": str(self._event_root),
                }
                for record in records
            ],
        }
        if group.license_plate_enabled:
            detector_model = (
                self._settings.lpr_model_root
                / "LPDNet_CCPD_pruned_tao5.onnx"
            )
            recognizer_model = (
                self._settings.lpr_model_root
                / "ch_lprnet_baseline18_deployable.onnx"
            )
            payload["license_plate"] = {
                "enabled": True,
                "model_root": str(self._settings.lpr_model_root),
                "parser_library": str(
                    self._settings.lpr_parser_library
                ),
                "detector_onnx_path": str(detector_model),
                "detector_engine_path": str(
                    self._settings.engine_root
                    / (
                        "lpdnet_ch_b"
                        f"{self._settings.lpr_detector_batch_size}"
                        f"_gpu{self._settings.gpu_id}_fp16.engine"
                    )
                ),
                "detector_batch_size": (
                    self._settings.lpr_detector_batch_size
                ),
                "recognizer_onnx_path": str(recognizer_model),
                "recognizer_engine_path": str(
                    self._settings.engine_root
                    / (
                        "lprnet_ch_b"
                        f"{self._settings.lpr_recognizer_batch_size}"
                        f"_gpu{self._settings.gpu_id}_fp16.engine"
                    )
                ),
                "recognizer_batch_size": (
                    self._settings.lpr_recognizer_batch_size
                ),
                "dictionary_path": str(
                    self._settings.lpr_model_root
                    / "ch_lp_characters.txt"
                ),
            }
        if group.garbage_enabled:
            if group.garbage_mode == "pile":
                garbage_onnx_path = self._settings.garbage_pile_onnx_path
                garbage_labels_path = (
                    self._settings.garbage_pile_labels_path
                )
            else:
                garbage_onnx_path = self._settings.garbage_onnx_path
                garbage_labels_path = self._settings.garbage_labels_path
            garbage_labels = [
                item.strip()
                for item in garbage_labels_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if item.strip()
            ]
            garbage_analysis_fps = max(
                record.spec.event_detection.garbage.analysis_fps
                for record in records
                if record.spec.event_detection.garbage.enabled
            )
            payload["garbage"] = {
                "enabled": True,
                "detection_mode": group.garbage_mode,
                "onnx_path": str(garbage_onnx_path),
                "engine_path": str(
                    engine_path_for(
                        garbage_onnx_path,
                        self._settings.engine_root,
                        imgsz=self._settings.garbage_input_size,
                        batch_size=self._settings.streams_per_group,
                        gpu_id=self._settings.gpu_id,
                    )
                ),
                "labels_path": str(garbage_labels_path),
                "label_count": len(garbage_labels),
                "parser_library": str(
                    self._settings.garbage_parser_library
                ),
                "imgsz": self._settings.garbage_input_size,
                # Sampling is monotonic-time based in the worker, so 15, 25,
                # and 30 FPS cameras all receive the requested analysis rate.
                "analysis_fps": garbage_analysis_fps,
            }
        if group.gas_cylinder_enabled:
            payload["gas_cylinder"] = {
                "enabled": True,
                "model_path": str(self._settings.gas_cylinder_model_path),
                "profile_root": str(
                    self._settings.gas_cylinder_profile_root
                ),
                "profile_id": group.gas_cylinder_profile_id,
                "input_width": self._settings.gas_cylinder_input_width,
                "input_height": self._settings.gas_cylinder_input_height,
                "imgsz": self._settings.gas_cylinder_imgsz,
                "analysis_fps": max(
                    record.spec.gas_cylinder.analysis_fps
                    for record in records
                    if record.spec.gas_cylinder.enabled
                ),
            }
        if group.vessel_signature[0]:
            vessel_model = group.vessel_signature[1]
            payload["vessel_detection"] = {
                "enabled": True,
                "model_path": str(self._resolve_pt_model(vessel_model)),
                "input_width": group.vessel_signature[3],
                "input_height": group.vessel_signature[4],
                "imgsz": group.vessel_signature[2],
                "analysis_fps": max(
                    record.spec.vessel_detection.analysis_fps
                    for record in records
                    if record.spec.vessel_detection.enabled
                ),
            }
        if group.ground_litter_signature[0]:
            litter_actor_model = group.ground_litter_signature[7] or None
            payload["ground_litter"] = {
                "enabled": True,
                "model_path": str(
                    self._resolve_ground_litter_model(
                        group.ground_litter_signature[1]
                    )
                ),
                "actor_model_path": (
                    str(
                        self._resolve_ground_litter_model(
                            litter_actor_model
                        )
                    )
                    if litter_actor_model
                    else None
                ),
                "tile_size_px": group.ground_litter_signature[2],
                "inference_imgsz": group.ground_litter_signature[3] or None,
                "local_actor_max_crops": group.ground_litter_signature[4],
                "box_smoothing_alpha": group.ground_litter_signature[5],
                "actor_imgsz": group.ground_litter_signature[6],
                "analysis_fps": max(
                    record.spec.ground_litter.analysis_fps
                    for record in records
                    if record.spec.ground_litter.enabled
                ),
            }
        return payload

    def _stop_group(
        self,
        group: DeepStreamGroup,
        *,
        stopping_ptz_options: tuple[PtzVerificationOptions, ...] = (),
        grace_seconds: float | None = None,
    ) -> None:
        if grace_seconds is None:
            grace_seconds = self._group_stop_grace_seconds(
                group,
                extra_ptz_options=stopping_ptz_options,
            )
        self._stop_process(group.process, grace_seconds=grace_seconds)
        group.process = None
        self._cleanup_group_files(group)

    def _group_stop_grace_seconds(
        self,
        group: DeepStreamGroup,
        *,
        extra_ptz_options: tuple[PtzVerificationOptions, ...] = (),
    ) -> float:
        ptz_options = [
            record.spec.ptz_verification
            for stream_id in group.stream_ids
            if (record := self._records.get(stream_id)) is not None
            and record.spec.ptz_verification.enabled
        ]
        ptz_options.extend(
            options for options in extra_ptz_options if options.enabled
        )
        if not ptz_options:
            return self._settings.worker_stop_grace_seconds
        return max(
            self._settings.worker_stop_grace_seconds,
            _WORKER_SHUTDOWN_OVERHEAD_SECONDS
            + sum(
                options.shutdown_timeout_seconds
                for options in ptz_options
            ),
        )

    def _cleanup_group_files(self, group: DeepStreamGroup) -> None:
        group_dir = self._settings.runtime_root / group.group_id
        for name in (
            "worker.json",
            "worker.json.tmp",
            "metrics.json",
            "metrics.tmp",
            "nvinfer.txt",
            "nvinfer.tmp",
            "lpd_nvinfer.txt",
            "lpd_nvinfer.tmp",
            "lpr_nvinfer.txt",
            "lpr_nvinfer.tmp",
            "tracker.yml",
            "tracker.tmp",
            "garbage_nvinfer.txt",
            "garbage_nvinfer.tmp",
        ):
            (group_dir / name).unlink(missing_ok=True)
        control_dir = group_dir / "control"
        if control_dir.is_dir():
            for path in control_dir.iterdir():
                if path.is_file():
                    path.unlink(missing_ok=True)
            try:
                control_dir.rmdir()
            except OSError:
                pass
        try:
            group_dir.rmdir()
        except OSError:
            pass

    def _stop_process(
        self,
        process: Any | None,
        *,
        grace_seconds: float | None = None,
    ) -> None:
        if process is None or process.poll() is not None:
            return
        timeout = (
            self._settings.worker_stop_grace_seconds
            if grace_seconds is None
            else max(grace_seconds, 0.0)
        )
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(
                timeout=timeout
            )
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            LOGGER.info(
                "DeepStream worker在%.2f秒内未退出，强制结束: pid=%s",
                timeout,
                process.pid,
            )
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=3)

    def _read_metrics(
        self,
        group: DeepStreamGroup | None,
        stream_id: str,
    ) -> dict[str, float | int | str | bool] | None:
        if group is None or group.metrics_path is None:
            return None
        try:
            payload = json.loads(
                group.metrics_path.read_text(encoding="utf-8")
            )
            if int(payload["generation"]) != group.generation:
                return None
            metrics = payload["streams"].get(stream_id)
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not isinstance(metrics, dict):
            return None
        return {
            **metrics,
            "metrics_updated_at_unix": float(payload["updated_at_unix"]),
        }

    def _serialize(
        self,
        record: DeepStreamRecord,
        *,
        forced_status: str | None = None,
    ) -> dict[str, Any]:
        group = self._groups.get(record.group_id)
        process = group.process if group is not None else None
        exit_code = process.poll() if process is not None else 0
        if forced_status is not None:
            status = forced_status
        elif exit_code is not None:
            status = "failed" if exit_code != 0 else "stopped"
        elif group is not None and (
            time.monotonic() - group.started_monotonic
            < self._settings.manager.startup_grace_seconds
        ):
            status = "starting"
        else:
            status = "running"
        metrics = self._read_metrics(group, record.stream_id)
        if metrics is not None and group is not None:
            metrics.update(
                {
                    "model_instance_id": group.group_id,
                    "model_instance_clients": len(group.stream_ids),
                    "backend": "deepstream",
                }
            )
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
            "metrics": metrics,
            "license_plate": {
                "enabled": record.spec.license_plate.enabled,
                "country": "CN",
            },
            "night_vision": {
                "enabled": record.spec.night_vision.enabled,
                "profile": (
                    "night" if record.spec.night_vision.enabled else "day"
                ),
                "confidence": (
                    record.spec.night_vision.confidence
                    if record.spec.night_vision.enabled
                    else record.spec.conf
                ),
                "input_gain": (
                    record.spec.night_vision.input_gain
                    if record.spec.night_vision.enabled
                    else 1.0
                ),
            },
            "event_detection": {
                "enabled": record.spec.event_detection.enabled,
                "events_url": f"/v1/streams/{record.stream_id}/events",
                "roi_ids": [
                    item.roi_id for item in record.spec.event_detection.rois
                ],
                "garbage_enabled": (
                    record.spec.event_detection.garbage.enabled
                ),
            },
            "gas_cylinder": {
                "enabled": record.spec.gas_cylinder.enabled,
                "profile_id": record.spec.gas_cylinder.profile_id,
                "alarm_threshold": (
                    record.spec.gas_cylinder.alarm_threshold
                ),
                "alarm_active": bool(
                    metrics is not None
                    and int(metrics.get("gas_cylinder_count", 0))
                    > record.spec.gas_cylinder.alarm_threshold
                ),
                "state": (
                    str(metrics.get("gas_cylinder_state", "starting"))
                    if metrics is not None
                    else "starting"
                ),
                "count": (
                    int(metrics.get("gas_cylinder_count", 0))
                    if metrics is not None
                    else 0
                ),
                "result_version": (
                    int(metrics.get("gas_cylinder_result_version", 0))
                    if metrics is not None
                    else 0
                ),
                "last_updated_at_unix": (
                    metrics.get("gas_cylinder_updated_at_unix")
                    if metrics is not None
                    else None
                ),
            },
            "vessel_detection": {
                "enabled": record.spec.vessel_detection.enabled,
                "model": (
                    record.spec.vessel_detection.model or record.model
                ),
                "state": (
                    str(metrics.get("vessel_detection_state", "starting"))
                    if metrics is not None
                    else "starting"
                ),
                "count": (
                    int(metrics.get("vessel_detection_count", 0))
                    if metrics is not None
                    else 0
                ),
                "result_version": (
                    int(metrics.get("vessel_detection_result_version", 0))
                    if metrics is not None
                    else 0
                ),
                "last_inference_ms": (
                    float(
                        metrics.get(
                            "vessel_detection_last_inference_ms",
                            0.0,
                        )
                    )
                    if metrics is not None
                    else 0.0
                ),
            },
            "ground_litter": {
                "enabled": record.spec.ground_litter.enabled,
                "model": record.spec.ground_litter.model,
                "options": record.spec.ground_litter.to_payload(),
                "effective_imgsz": record.spec.ground_litter.effective_imgsz,
                "region_count": len(record.spec.ground_litter.zones),
                "state": (
                    "disabled"
                    if not record.spec.ground_litter.enabled
                    else (
                        str(
                            metrics.get(
                                "ground_litter_state",
                                "starting",
                            )
                        )
                        if metrics is not None
                        else "starting"
                    )
                ),
                "count": (
                    int(metrics.get("ground_litter_count", 0))
                    if metrics is not None
                    else 0
                ),
                "message": (
                    str(metrics.get("ground_litter_message", ""))
                    if metrics is not None
                    else ""
                ),
                "result_version": (
                    int(metrics.get("ground_litter_result_version", 0))
                    if metrics is not None
                    else 0
                ),
                "analyzed_frames": (
                    int(metrics.get("ground_litter_analyzed_frames", 0))
                    if metrics is not None
                    else 0
                ),
                "tile_count": (
                    int(metrics.get("ground_litter_tile_count", 0))
                    if metrics is not None
                    else 0
                ),
                "last_inference_ms": (
                    float(
                        metrics.get(
                            "ground_litter_last_inference_ms",
                            0.0,
                        )
                    )
                    if metrics is not None
                    else 0.0
                ),
            },
            "fishing_risk": {
                "enabled": record.spec.fishing_risk.enabled,
                "state": (
                    "disabled"
                    if not record.spec.fishing_risk.enabled
                    else (
                        str(metrics.get("fishing_risk_state", "starting"))
                        if metrics is not None
                        else "starting"
                    )
                ),
                "suspect_count": (
                    int(metrics.get("fishing_risk_suspect_count", 0))
                    if metrics is not None
                    else 0
                ),
                "maximum_score": (
                    int(metrics.get("fishing_risk_maximum_score", 0))
                    if metrics is not None
                    else 0
                ),
                "total_events": (
                    int(metrics.get("fishing_risk_total_events", 0))
                    if metrics is not None
                    else 0
                ),
                "events_url": f"/v1/streams/{record.stream_id}/events",
            },
            "ptz_verification": {
                "enabled": record.spec.ptz_verification.enabled,
                "integration_mode": (
                    "camera_control"
                    if record.spec.ptz_verification.enabled
                    else "detection_only"
                ),
                "zoom_strategy": record.spec.ptz_verification.zoom_strategy,
                "continuous_tracking": (
                    record.spec.ptz_verification.continuous_tracking
                ),
                "tracking_profile": record.spec.ptz_verification.tracking_profile,
                "effective_policy": record.spec.ptz_verification.effective_policy(),
                "edge_guard": {
                    "enabled": record.spec.ptz_verification.tracking_edge_guard_enabled,
                    "response_seconds": record.spec.ptz_verification.tracking_edge_response_seconds,
                    "motion_seconds": record.spec.ptz_verification.tracking_edge_motion_seconds,
                    "uncertainty_seconds": record.spec.ptz_verification.tracking_edge_uncertainty_seconds,
                    "cooldown_seconds": record.spec.ptz_verification.tracking_edge_cooldown_seconds,
                    "stable_seconds": record.spec.ptz_verification.tracking_edge_stable_seconds,
                    "maximum_age_seconds": record.spec.ptz_verification.tracking_edge_maximum_age_seconds,
                    "latency_source": "configured_assumption",
                    "effective_policy": (
                        "position_first_risk_only_shrink"
                        if record.spec.ptz_verification.tracking_edge_guard_enabled
                        else "legacy_scale_tracking"
                    ),
                    "ignored_parameters": (
                        ["tracking_recovery_enabled", "tracking_recovery_zoom_out_step",
                         "tracking_recovery_interval_seconds", "tracking_recovery_max_attempts",
                         "tracking_zoom_step", "tracking_zoom_hysteresis_ratio"]
                        if record.spec.ptz_verification.tracking_edge_guard_enabled else []
                    ),
                },
                "camera_id": record.spec.ptz_verification.camera_id,
                "state": (
                    "disabled"
                    if not record.spec.ptz_verification.enabled
                    else (
                        str(metrics.get("ptz_verification_state", "starting"))
                        if metrics is not None
                        else "starting"
                    )
                ),
                "last_error": (
                    metrics.get("ptz_verification_last_error")
                    if metrics is not None
                    else None
                ),
                "events_url": "/v1/vessel-verifications",
            },
        }


def _parse_bitrate(value: str) -> int:
    normalized = value.strip().lower()
    multiplier = 1
    if normalized.endswith("k"):
        multiplier = 1_000
        normalized = normalized[:-1]
    elif normalized.endswith("m"):
        multiplier = 1_000_000
        normalized = normalized[:-1]
    parsed = int(normalized) * multiplier
    if parsed <= 0:
        raise ValueError("bitrate必须大于0")
    return parsed
