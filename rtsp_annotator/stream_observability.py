from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .config import redact_url


LOGGER = logging.getLogger(__name__)
STREAM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
URL_PATTERN = re.compile(r"(?:rtsp|rtsps)://[^\s,\]\[\"']+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_text(value: str) -> str:
    return URL_PATTERN.sub(lambda match: redact_url(match.group(0)), value)


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value))


class StreamLogStore:
    """Thread-safe, per-stream JSONL event store with bounded rotation."""

    def __init__(
        self,
        root: Path,
        *,
        max_file_bytes: int = 32 * 1024 * 1024,
        backup_count: int = 3,
    ) -> None:
        if max_file_bytes < 1024:
            raise ValueError("max_file_bytes不能小于1024")
        if backup_count < 0:
            raise ValueError("backup_count不能为负数")
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._max_file_bytes = max_file_bytes
        self._backup_count = backup_count
        self._lock = threading.RLock()
        self._sequences: dict[str, int] = {}

    def path_for(self, stream_id: str) -> Path:
        self._validate_stream_id(stream_id)
        return self.root / f"{stream_id}.jsonl"

    def exists(self, stream_id: str) -> bool:
        path = self.path_for(stream_id)
        with self._lock:
            return any(
                candidate.is_file()
                for candidate in self._paths_oldest_first(path)
            )

    def append(
        self,
        stream_id: str,
        *,
        level: str,
        event: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = self.path_for(stream_id)
        normalized_level = level.upper()
        if normalized_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(f"不支持的日志级别: {level}")
        with self._lock:
            sequence = self._next_sequence_locked(stream_id)
            entry = {
                "timestamp": _utc_now(),
                "timestamp_unix": time.time(),
                "sequence": sequence,
                "stream_id": stream_id,
                "level": normalized_level,
                "event": event,
                "message": _redact_text(message),
                "details": _safe_value(dict(details or {})),
            }
            encoded = (
                json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            self._rotate_if_needed_locked(path, len(encoded))
            descriptor = os.open(
                path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o640,
            )
            try:
                os.write(descriptor, encoded)
            finally:
                os.close(descriptor)
            return entry

    def read(
        self,
        stream_id: str,
        *,
        limit: int = 200,
        after_sequence: int | None = None,
        level: str | None = None,
        event: str | None = None,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 5_000:
            raise ValueError("limit必须在1到5000之间")
        path = self.path_for(stream_id)
        normalized_level = level.upper() if level is not None else None
        with self._lock:
            entries: list[dict[str, Any]] = []
            for candidate in self._paths_oldest_first(path):
                if not candidate.is_file():
                    continue
                try:
                    lines = candidate.read_text(encoding="utf-8").splitlines()
                except OSError:
                    continue
                for line in lines:
                    try:
                        item = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    sequence = int(item.get("sequence", 0))
                    if after_sequence is not None and sequence <= after_sequence:
                        continue
                    if (
                        normalized_level is not None
                        and item.get("level") != normalized_level
                    ):
                        continue
                    if event is not None and item.get("event") != event:
                        continue
                    entries.append(item)
            return entries[-limit:]

    def follow(
        self,
        stream_id: str,
        *,
        after_sequence: int = 0,
        poll_seconds: float = 0.5,
        stop_event: threading.Event | None = None,
    ) -> Iterator[dict[str, Any]]:
        cursor = after_sequence
        stopper = stop_event or threading.Event()
        while not stopper.is_set():
            entries = self.read(
                stream_id,
                limit=5_000,
                after_sequence=cursor,
            )
            for entry in entries:
                cursor = max(cursor, int(entry["sequence"]))
                yield entry
            stopper.wait(poll_seconds)

    def _next_sequence_locked(self, stream_id: str) -> int:
        current = self._sequences.get(stream_id)
        if current is None:
            current = 0
            entries = self.read(stream_id, limit=1)
            if entries:
                current = int(entries[-1].get("sequence", 0))
        current += 1
        self._sequences[stream_id] = current
        return current

    def _rotate_if_needed_locked(self, path: Path, incoming_bytes: int) -> None:
        try:
            current_size = path.stat().st_size
        except FileNotFoundError:
            return
        if current_size + incoming_bytes <= self._max_file_bytes:
            return
        if self._backup_count == 0:
            path.unlink(missing_ok=True)
            return
        oldest = path.with_suffix(f".jsonl.{self._backup_count}")
        oldest.unlink(missing_ok=True)
        for index in range(self._backup_count - 1, 0, -1):
            source = path.with_suffix(f".jsonl.{index}")
            if source.exists():
                source.replace(path.with_suffix(f".jsonl.{index + 1}"))
        path.replace(path.with_suffix(".jsonl.1"))

    def _paths_oldest_first(self, path: Path) -> list[Path]:
        return [
            path.with_suffix(f".jsonl.{index}")
            for index in range(self._backup_count, 0, -1)
        ] + [path]

    @staticmethod
    def _validate_stream_id(stream_id: str) -> None:
        if not STREAM_ID_PATTERN.fullmatch(stream_id):
            raise ValueError("无效stream_id")


@dataclass(frozen=True, slots=True)
class PlaybackDiagnosis:
    health: str
    level: str
    message: str
    evidence: tuple[str, ...]


class StreamHealthEvaluator:
    def __init__(
        self,
        *,
        stale_after_seconds: float = 12.0,
        stall_fps: float = 1.0,
        degraded_fps_ratio: float = 0.8,
    ) -> None:
        self.stale_after_seconds = stale_after_seconds
        self.stall_fps = stall_fps
        self.degraded_fps_ratio = degraded_fps_ratio

    def diagnose(
        self,
        record: Mapping[str, Any],
        *,
        now_unix: float | None = None,
    ) -> PlaybackDiagnosis:
        now = time.time() if now_unix is None else now_unix
        status = str(record.get("status", "unknown"))
        if status == "failed":
            return PlaybackDiagnosis(
                "failed",
                "ERROR",
                "流处理进程已失败，当前无法播放",
                (f"exit_code={record.get('exit_code')}",),
            )
        if status == "stopped":
            return PlaybackDiagnosis("stopped", "INFO", "流已停止", ())
        if status == "starting":
            return PlaybackDiagnosis("starting", "INFO", "流正在启动", ())

        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            return PlaybackDiagnosis(
                "unobservable",
                "WARNING",
                "任务仍在运行，但后端尚未提供逐流播放指标",
                ("metrics_missing",),
            )
        updated_at = float(metrics.get("metrics_updated_at_unix", now))
        age = max(0.0, now - updated_at)
        if age > self.stale_after_seconds:
            return PlaybackDiagnosis(
                "stalled",
                "ERROR",
                (
                    f"连续{age:.1f}秒没有收到新播放指标，"
                    "疑似卡死或断流"
                ),
                ("metrics_stale", f"metrics_age_seconds={age:.1f}"),
            )

        corrupt = int(metrics.get("interval_corrupt_frames", 0))
        discontinuities = int(metrics.get("interval_discontinuities", 0))
        reconnects = int(metrics.get("interval_capture_reconnects", 0))
        if corrupt > 0:
            return PlaybackDiagnosis(
                "mosaic_risk",
                "ERROR",
                "解码链路报告坏帧，画面存在花屏或马赛克风险",
                (f"corrupt_frames={corrupt}",),
            )
        if discontinuities > 0:
            return PlaybackDiagnosis(
                "mosaic_risk",
                "WARNING",
                "视频缓冲出现不连续，画面可能短时卡顿或花屏",
                (f"discontinuities={discontinuities}",),
            )

        capture_fps = float(metrics.get("capture_fps", 0.0))
        publish_fps = float(metrics.get("publish_fps", 0.0))
        effective_fps = min(capture_fps, publish_fps)
        gap_count = int(metrics.get("interval_frame_gaps", 0))
        max_gap_ms = float(metrics.get("max_interframe_gap_ms", 0.0))
        if effective_fps < self.stall_fps or gap_count > 0:
            evidence = [
                f"capture_fps={capture_fps:.1f}",
                f"publish_fps={publish_fps:.1f}",
            ]
            if gap_count:
                evidence.extend(
                    (f"frame_gaps={gap_count}", f"max_gap_ms={max_gap_ms:.1f}")
                )
            return PlaybackDiagnosis(
                "stalled",
                "ERROR",
                "播放帧率接近零或出现明显帧间断，已判定卡顿",
                tuple(evidence),
            )

        minimum_fps = float(metrics.get("minimum_healthy_fps", 0.0))
        low_fps = minimum_fps > 0 and effective_fps < (
            minimum_fps * self.degraded_fps_ratio
        )
        if low_fps or reconnects > 0 or metrics.get("pipeline_healthy") is False:
            evidence = [
                f"capture_fps={capture_fps:.1f}",
                f"publish_fps={publish_fps:.1f}",
            ]
            if reconnects:
                evidence.append(f"capture_reconnects={reconnects}")
            return PlaybackDiagnosis(
                "degraded",
                "WARNING",
                "视频仍在播放，但帧率偏低或拉流发生重连",
                tuple(evidence),
            )
        return PlaybackDiagnosis(
            "healthy",
            "INFO",
            "播放正常，未发现卡顿或解码坏帧信号",
            (
                f"capture_fps={capture_fps:.1f}",
                f"publish_fps={publish_fps:.1f}",
            ),
        )


class ObservedStreamManager:
    """Manager decorator that continuously emits per-stream health logs."""

    def __init__(
        self,
        manager: Any,
        store: StreamLogStore,
        *,
        monitor_interval_seconds: float = 1.0,
        stale_after_seconds: float = 12.0,
        stall_fps: float = 1.0,
        degraded_fps_ratio: float = 0.8,
    ) -> None:
        self._manager = manager
        self.log_store = store
        self._monitor_interval_seconds = monitor_interval_seconds
        self._evaluator = StreamHealthEvaluator(
            stale_after_seconds=stale_after_seconds,
            stall_fps=stall_fps,
            degraded_fps_ratio=degraded_fps_ratio,
        )
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._last_fingerprint: dict[str, str] = {}
        self._thread = threading.Thread(
            target=self._monitor,
            name="stream-health-monitor",
            daemon=True,
        )
        self._thread.start()

    def list_models(self) -> list[str]:
        return self._manager.list_models()

    def create(self, spec: Any) -> dict[str, Any]:
        record = self._manager.create(spec)
        stream_id = str(record["stream_id"])
        self.log_store.append(
            stream_id,
            level="INFO",
            event="stream.created",
            message="流任务已创建，正在等待首批播放指标",
            details={
                "status": record.get("status"),
                "model": record.get("model"),
                "input_url": getattr(spec, "input_url", None),
            },
        )
        self._observe_record(record, force=True)
        return record

    def list(self) -> list[dict[str, Any]]:
        return self._manager.list()

    def get(self, stream_id: str) -> dict[str, Any]:
        return self._manager.get(stream_id)

    def update_fishing_risk(
        self,
        stream_id: str,
        options: Any,
    ) -> dict[str, Any]:
        record = self._manager.update_fishing_risk(stream_id, options)
        self.log_store.append(
            stream_id,
            level="INFO",
            event="fishing_risk.updated",
            message=(
                "疑似捕捞分析已开启"
                if bool(getattr(options, "enabled", False))
                else "疑似捕捞分析已关闭，保留纯船舶识别"
            ),
            details={"enabled": bool(getattr(options, "enabled", False))},
        )
        self._observe_record(record, force=True)
        return record

    def return_ptz_home(self, stream_id: str) -> dict[str, Any]:
        result = self._manager.return_ptz_home(stream_id)
        self.log_store.append(
            stream_id,
            level="WARNING",
            event="ptz.return_home_requested",
            message="已请求中断PTZ任务并紧急回HOME",
            details={"request_id": result.get("request_id")},
        )
        return result

    def stop(self, stream_id: str) -> dict[str, Any]:
        record = self._manager.stop(stream_id)
        self.log_store.append(
            stream_id,
            level="INFO",
            event="stream.stopped",
            message="流任务已停止",
            details={"exit_code": record.get("exit_code")},
        )
        with self._state_lock:
            self._last_fingerprint.pop(stream_id, None)
        return record

    def shutdown(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=max(2.0, self._monitor_interval_seconds * 2))
        try:
            records = self._manager.list()
        except Exception:
            records = []
        self._manager.shutdown()
        for record in records:
            stream_id = str(record.get("stream_id", ""))
            if STREAM_ID_PATTERN.fullmatch(stream_id):
                self.log_store.append(
                    stream_id,
                    level="INFO",
                    event="stream.shutdown",
                    message="API服务关闭，流任务已清理",
                )

    def _monitor(self) -> None:
        while not self._stop_event.wait(self._monitor_interval_seconds):
            try:
                records = self._manager.list()
            except Exception:
                LOGGER.exception("读取逐流健康状态失败")
                continue
            for record in records:
                try:
                    self._observe_record(record)
                except Exception:
                    LOGGER.exception("写入逐流健康日志失败")

    def _observe_record(
        self,
        record: Mapping[str, Any],
        *,
        force: bool = False,
    ) -> None:
        stream_id = str(record.get("stream_id", ""))
        if not STREAM_ID_PATTERN.fullmatch(stream_id):
            return
        diagnosis = self._evaluator.diagnose(record)
        metrics = record.get("metrics")
        metrics_fingerprint = ""
        if isinstance(metrics, Mapping):
            metrics_fingerprint = json.dumps(
                _safe_value(dict(metrics)),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        fingerprint = f"{record.get('status')}|{diagnosis.health}|{metrics_fingerprint}"
        with self._state_lock:
            if not force and self._last_fingerprint.get(stream_id) == fingerprint:
                return
            self._last_fingerprint[stream_id] = fingerprint
        self.log_store.append(
            stream_id,
            level=diagnosis.level,
            event="playback.health",
            message=diagnosis.message,
            details={
                "health": diagnosis.health,
                "evidence": diagnosis.evidence,
                "metrics": metrics if isinstance(metrics, Mapping) else None,
            },
        )


def encode_sse(entry: Mapping[str, Any]) -> str:
    payload = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    return f"id: {entry['sequence']}\nevent: stream-log\ndata: {payload}\n\n"
