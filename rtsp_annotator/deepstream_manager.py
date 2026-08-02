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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .deepstream_engine_builder import engine_path_for
from .events import GarbageAnalysisOptions
from .labels import load_label_map
from .stream_manager import (
    ManagerSettings,
    ModelNotFoundError,
    StreamCapacityError,
    StreamNotFoundError,
    StreamSpec,
    authenticated_rtsp_url,
)


ProcessFactory = Callable[..., Any]
LOGGER = logging.getLogger(__name__)


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
            if spec.license_plate.enabled:
                self._validate_lpr_assets()
            if spec.event_detection.enabled:
                self._validate_event_classes(spec, labels_path)
            if spec.event_detection.garbage.enabled:
                self._validate_garbage_assets(spec.event_detection.garbage)
            group = self._find_group(
                model_path.name,
                spec.imgsz,
                spec.license_plate.enabled,
                spec.night_vision.group_signature(
                    include_plate_detector=spec.license_plate.enabled,
                ),
                spec.event_detection.garbage.enabled,
                spec.event_detection.garbage.detection_mode,
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

    def stop(self, stream_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.pop(stream_id, None)
            if record is None:
                raise StreamNotFoundError(stream_id)
            group = self._groups[record.group_id]
            group.stream_ids.remove(stream_id)
            if group.stream_ids:
                self._restart_group(group)
            else:
                self._stop_group(group)
                self._groups.pop(group.group_id, None)
            result = self._serialize(record, forced_status="stopped")
            result["exit_code"] = 0
            return result

    def shutdown(self) -> None:
        with self._lock:
            groups = list(self._groups.values())
            self._groups.clear()
            self._records.clear()
        for group in groups:
            self._stop_group(group)

    def _find_group(
        self,
        model: str,
        imgsz: int,
        license_plate_enabled: bool,
        night_vision_signature: tuple[bool, float, float],
        garbage_enabled: bool,
        garbage_mode: str,
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

    def _restart_group(self, group: DeepStreamGroup) -> None:
        previous_generation = group.generation
        group.generation = previous_generation + 1
        group_dir = self._settings.runtime_root / group.group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        group.config_path = group_dir / "worker.json"
        group.metrics_path = group_dir / "metrics.json"
        try:
            payload = self._worker_payload(group)
        except BaseException:
            group.generation = previous_generation
            raise
        self._stop_process(group.process)
        group.process = None
        group.metrics_path.unlink(missing_ok=True)
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
        return payload

    def _stop_group(self, group: DeepStreamGroup) -> None:
        self._stop_process(group.process)
        group.process = None
        self._cleanup_group_files(group)

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
        try:
            group_dir.rmdir()
        except OSError:
            pass

    def _stop_process(self, process: Any | None) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(
                timeout=self._settings.worker_stop_grace_seconds
            )
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            LOGGER.info(
                "DeepStream worker在%.2f秒内未退出，强制结束: pid=%s",
                self._settings.worker_stop_grace_seconds,
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
        return dict(metrics) if isinstance(metrics, dict) else None

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
