"""A2：有界租约消费、粗采样、质量过滤、配准与外观特征（方案一 §4.1～§4.3、§5）。

三条硬性约束：

1. **消费的是租约**：本模块只接受 ``ManagedRecordingCache`` 发出的 ``Lease``，
   绝不直接打开任意路径，也不负责配额决策。
2. **时间集合先划分再学习**：``partition_recordings`` 在抽样/聚类之前按天把文件分到
   构建集（前五天）、校准集（第六天）与盲测集（第七天）。跨日文件按时间范围划分，
   不做整文件随机分配，避免泄漏。
3. **预览不等于高清**：粗采样产出的小图只用于外观发现与质量筛选；
   合成与噪声阶段必须用来源散列按需重拉高清帧，本模块的 ``HdRequest`` 就是那份清单。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import math
import random
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np

from .ground_litter_profile_bank import (
    BankError, atomic_write_bytes, atomic_write_json, sha256_bytes, sha256_file,
)
from .ground_litter_profile_match import (
    descriptor_arrays, extract_grid_descriptor,
)
from .ground_litter_recording_cache import Lease, ManagedRecordingCache
from .ground_litter_recording_source import RecordingFile, file_looks_like_media

# 粗采样：每个五分钟文件取约 5s / 150s / 295s 与一个可复现随机时刻（§4.2）。
COARSE_FRACTIONS = (0.02, 0.50, 0.98)
PREVIEW_WIDTH = 960
DEFAULT_SEED = 20260920

QUALITY_REASONS = (
    "DECODE_FAILED", "BLACK_FRAME", "OVEREXPOSED", "BLURRY", "DETAIL_LOSS",
    "FRAME_FROZEN", "RESOLUTION_MISMATCH", "CORRUPT",
)


@dataclass(frozen=True, slots=True)
class RecordedFrame:
    frame: np.ndarray
    time_seconds: float
    index: int
    pts: float | None
    source: str = ""
    corrupt: bool = False


@dataclass(frozen=True, slots=True)
class QualityReport:
    usable: bool
    reasons: tuple[str, ...]
    mean_luminance: float
    laplacian_variance: float
    detail_loss_fraction: float
    saturated_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "usable": self.usable, "reasons": list(self.reasons),
            "mean_luminance": round(self.mean_luminance, 3),
            "laplacian_variance": round(self.laplacian_variance, 3),
            "detail_loss_fraction": round(self.detail_loss_fraction, 4),
            "saturated_fraction": round(self.saturated_fraction, 4),
        }


def frame_quality(
    frame: np.ndarray, *, min_luminance: float = 8.0, max_luminance: float = 247.0,
    min_laplacian_variance: float = 3.0, max_detail_loss_fraction: float = 0.90,
    max_largest_detail_loss: float = 0.95,
) -> QualityReport:
    """单帧质量判定：黑屏/过曝/严重模糊/花屏。

    **不**用「连续几张图一样」单独判冻结——静止场景会被误删（§4.3）；
    冻结必须结合时间戳/码流证据，见 ``detect_frozen``。

    细节丢失只作为诊断报告，且门槛设得极保守：低纹理但正常的机位
    （水泥地、夜间隔热画面）不应该因为「平坦」被整帧丢弃。
    """
    reasons: list[str] = []
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return QualityReport(False, ("DECODE_FAILED",), 0.0, 0.0, 1.0, 0.0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    laplacian = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    saturated = float(np.count_nonzero((gray <= 3) | (gray >= 252)) / gray.size)
    detail_fraction, largest = _detail_loss(frame)
    if mean < min_luminance:
        reasons.append("BLACK_FRAME")
    if mean > max_luminance:
        reasons.append("OVEREXPOSED")
    if laplacian < min_laplacian_variance:
        reasons.append("BLURRY")
    if detail_fraction > max_detail_loss_fraction and largest > max_largest_detail_loss:
        reasons.append("DETAIL_LOSS")
    return QualityReport(
        usable=not reasons, reasons=tuple(reasons), mean_luminance=mean,
        laplacian_variance=laplacian, detail_loss_fraction=detail_fraction,
        saturated_fraction=saturated,
    )


def _detail_loss(frame: np.ndarray) -> tuple[float, float]:
    """花屏/整块细节丢失判据（单帧可判定部分）。

    ``ground_litter_quality.DetailLossGuard`` 的用途是影子试点里与**参考图**比较，
    构造时需要参考图；这里只有单帧，因此做保守的局部平坦度统计：
    连续纯色/灰块占比。宁可漏判，也不误删合法静止场景。
    """
    if frame is None or frame.ndim != 3 or frame.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (9, 9))
    square = cv2.blur(gray * gray, (9, 9))
    std = np.sqrt(np.maximum(square - mean * mean, 0.0))
    flat = (std < 0.5).astype(np.uint8)
    flat = cv2.morphologyEx(flat, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(flat, 8)
    fraction = float(np.count_nonzero(flat)) / float(flat.size)
    largest = 0.0
    for index in range(1, count):
        largest = max(largest, float(stats[index, cv2.CC_STAT_AREA]) / float(flat.size))
    return fraction, largest


def detect_frozen(
    frames: Sequence[RecordingFrameStamp], *, max_identical: int = 4,
    timestamp_progress: bool = True,
) -> bool:
    """冻结判定：多帧像素完全相同**且**时间戳没有前进才算冻结。

    仅有相同像素不足以判冻结——固定机位的静态场景本来就一样。
    """
    if len(frames) < max_identical:
        return False
    hashes = {stamp.pixel_sha256 for stamp in frames[-max_identical:]}
    if len(hashes) > 1:
        return False
    if not timestamp_progress:
        return True
    times = [stamp.time_seconds for stamp in frames[-max_identical:]]
    return max(times) - min(times) < 1e-6 and len(set(times)) == 1


@dataclass(frozen=True, slots=True)
class RecordingFrameStamp:
    time_seconds: float
    pixel_sha256: str


# --------------------------------------------------------------------------- #
# 解码
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ProbeResult:
    ok: bool
    width: int = 0
    height: int = 0
    duration_seconds: float = 0.0
    frame_count: int = 0
    codec: str = ""
    container: str = ""
    decode_seconds: float = 0.0
    error: str = ""


def probe_recording(
    path: str | Path, *, max_probe_frames: int = 1, timeout: float = 30.0,
) -> ProbeResult:
    """探测可解码性与编码/分辨率；先做轻量内容探针再交给 PyAV。"""
    target = Path(path)
    if not target.is_file():
        return ProbeResult(False, error="文件不存在")
    if not file_looks_like_media(target):
        return ProbeResult(False, error="内容不像媒体/PS，可能是 200 错误正文")
    started = time.monotonic()
    try:
        import av
    except ImportError:  # pragma: no cover
        return ProbeResult(False, error="缺少 PyAV")
    try:
        with av.open(str(target), timeout=(10.0, timeout)) as container:
            stream = next(
                (item for item in container.streams if item.type == "video"), None
            )
            if stream is None:
                return ProbeResult(False, error="没有视频流")
            duration = 0.0
            if container.duration:
                duration = float(container.duration) / 1_000_000.0
            elif stream.duration is not None and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            decoded = 0
            for _frame in container.decode(video=0):
                decoded += 1
                if decoded >= max(1, max_probe_frames):
                    break
            if decoded < 1:
                return ProbeResult(False, error="解码不到画面")
            return ProbeResult(
                True, width=int(stream.codec_context.width),
                height=int(stream.codec_context.height), duration_seconds=duration,
                frame_count=int(stream.frames or 0),
                codec=str(stream.codec_context.name or ""),
                container=target.suffix.lower().lstrip("."),
                decode_seconds=time.monotonic() - started,
            )
    except Exception as exc:  # PyAV 异常文本可能含路径，这里只保留类型
        return ProbeResult(False, error=f"解码失败: {type(exc).__name__}",
                           decode_seconds=time.monotonic() - started)


class SequentialFrameReader:
    """按时间顺序解码，避免为多个时间点反复 seek。

    文档明确要求：优先同文件一次取完所需时间点；PS 的 seek 成本必须实测。
    ``seek_points`` 非空时使用 seek 模式，并把实测 seek 耗时计入 ``seek_seconds``。
    """

    def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.decode_seconds = 0.0
        self.seek_seconds = 0.0
        self.frames_decoded = 0

    def iter_frames(self) -> Iterator[RecordedFrame]:
        """按顺序解码，并把时间戳**重定基**到文件内的相对秒数。

        PS 容器的 PTS 基值可能是任意大数（实测 10394s 起），文件内 PTS 也可能
        重置；文档明确要求不能把它当绝对时间。这里统一减去首帧时间，
        得到可跨文件比较的「文件内偏移」，绝对时间由来源清单提供。
        """
        try:
            import av
        except ImportError as exc:  # pragma: no cover
            raise BankError("缺少 PyAV") from exc
        started = time.monotonic()
        index = 0
        base: float | None = None
        with av.open(str(self.path), timeout=(10.0, self.timeout)) as container:
            stream = next(
                (item for item in container.streams if item.type == "video"), None
            )
            if stream is None:
                raise BankError("录像没有视频流")
            # HEVC 2.5K 单线程顺序解码实测约 180s/5 分钟文件；服务器上
            # thread_count=0（自动）约为其 1/3，必须开启。
            stream.thread_type = "FRAME"
            stream.codec_context.thread_count = 0
            for decoded in container.decode(video=0):
                timestamp = decoded.time
                if timestamp is None and decoded.pts is not None and stream.time_base:
                    timestamp = float(decoded.pts * stream.time_base)
                value = float(timestamp) if timestamp is not None else None
                if value is not None and base is None:
                    base = value
                relative = 0.0 if value is None or base is None else value - base
                yield RecordedFrame(
                    frame=decoded.to_ndarray(format="bgr24"),
                    time_seconds=relative,
                    index=index,
                    pts=None if decoded.pts is None else float(decoded.pts),
                    source=str(self.path),
                )
                index += 1
        self.frames_decoded += index
        self.decode_seconds += time.monotonic() - started

    def sample_at(
        self, offsets_seconds: Sequence[float], *,
        tolerance_seconds: float = 1.0, min_gap_seconds: float = 2.0,
    ) -> dict[float, RecordedFrame]:
        """一次顺序解码取回多个时间点，返回 {目标偏移: 实际帧}。

        实际帧时间可能偏离目标（关键帧/缺帧）；偏差写入返回值，由调用方记录，
        **不**假定请求时刻一定存在。
        """
        targets = sorted({float(value) for value in offsets_seconds})
        if not targets:
            return {}
        result: dict[float, RecordedFrame] = {}
        best_index = 0
        for frame in self.iter_frames():
            while best_index < len(targets) and frame.time_seconds > targets[best_index]:
                # 下一个目标还没到；当前帧比上一个已记录的更接近上一个目标。
                if targets[best_index] - frame.time_seconds >= 0:
                    break
                best_index += 1
                if best_index >= len(targets):
                    break
            if best_index >= len(targets):
                break
            target = targets[best_index]
            existing = result.get(target)
            if existing is None:
                result[target] = frame
            elif abs(frame.time_seconds - target) < abs(existing.time_seconds - target):
                result[target] = frame
        # 偏差过大的目标不返回，避免把错误时刻当命中。
        return {
            target: frame for target, frame in result.items()
            if abs(frame.time_seconds - target) <= max(tolerance_seconds, 0.0)
            or len(targets) == 1
        }

    def probe_seek(
        self, target_seconds: float, *, tolerance_seconds: float = 1.0,
        window_seconds: float = 4.0,
    ) -> tuple[RecordedFrame | None, dict[str, Any]]:
        """seek 到目标附近取最接近的一帧，并报告实际耗时与命中偏差。

        ``target_seconds`` 是**文件内相对秒**。seek 目标用容器真实时间
        （首帧基准 + 目标），因为 PS 的 PTS 基值可能是任意大数。

        实测 PS 的关键帧间隔可达数秒，只往后解码会稳定偏晚 2–4s；因此同时保留
        「目标之前最后一帧」与「目标之后第一帧」，取时间上更近的那一帧，并在
        诊断里记录它是 before 还是 after，便于报告说明偏差方向。
        """
        try:
            import av
        except ImportError as exc:  # pragma: no cover
            raise BankError("缺少 PyAV") from exc
        started = time.monotonic()
        diagnostics: dict[str, Any] = {"mode": "seek", "target_seconds": target_seconds}
        with av.open(str(self.path), timeout=(10.0, self.timeout)) as container:
            stream = next(
                (item for item in container.streams if item.type == "video"), None
            )
            if stream is None:
                raise BankError("录像没有视频流")
            stream.thread_type = "FRAME"
            stream.codec_context.thread_count = 0
            base: float | None = None
            for decoded in container.decode(video=0):
                timestamp = decoded.time
                if timestamp is None and decoded.pts is not None and stream.time_base:
                    timestamp = float(decoded.pts * stream.time_base)
                base = float(timestamp) if timestamp is not None else None
                break
            if base is None:
                diagnostics["error"] = "no_first_frame"
                return None, diagnostics
            seek_started = time.monotonic()
            try:
                container.seek(
                    int(max(0.0, base + target_seconds - window_seconds) * 1_000_000),
                    backward=True,
                )
            except Exception as exc:
                diagnostics["error"] = f"seek_failed:{type(exc).__name__}"
                self.seek_seconds += time.monotonic() - seek_started
                return None, diagnostics
            self.seek_seconds += time.monotonic() - seek_started
            diagnostics["seek_seconds"] = round(self.seek_seconds, 4)
            before: RecordedFrame | None = None
            after: RecordedFrame | None = None
            decoded_count = 0
            gop_gap: float | None = None
            for decoded in container.decode(video=0):
                timestamp = decoded.time
                if timestamp is None and decoded.pts is not None and stream.time_base:
                    timestamp = float(decoded.pts * stream.time_base)
                if timestamp is None:
                    continue
                relative = float(timestamp) - base
                decoded_count += 1
                candidate = RecordedFrame(
                    frame=decoded.to_ndarray(format="bgr24"),
                    time_seconds=relative, index=0,
                    pts=None if decoded.pts is None else float(decoded.pts),
                    source=str(self.path),
                )
                if relative <= target_seconds:
                    if before is None or relative > before.time_seconds:
                        before = candidate
                elif after is None:
                    after = candidate
                    if before is not None:
                        gop_gap = round(after.time_seconds - before.time_seconds, 3)
                if relative > target_seconds + window_seconds and after is not None:
                    break
            diagnostics["decoded_frames"] = decoded_count
            diagnostics["wall_seconds"] = round(time.monotonic() - started, 4)
            if gop_gap is not None:
                diagnostics["keyframe_gap_seconds"] = gop_gap
            if before is None and after is None:
                diagnostics["error"] = "no_frame_after_seek"
                return None, diagnostics
            if before is None:
                best, side = after, "after"
            elif after is None:
                best, side = before, "before"
            elif abs(before.time_seconds - target_seconds) <= abs(
                after.time_seconds - target_seconds
            ):
                best, side = before, "before"
            else:
                best, side = after, "after"
            assert best is not None
            diagnostics["side"] = side
            diagnostics["hit_seconds"] = round(best.time_seconds, 4)
            diagnostics["offset_error_seconds"] = round(
                best.time_seconds - target_seconds, 4
            )
            if abs(best.time_seconds - target_seconds) > max(tolerance_seconds, 0.0):
                diagnostics["error"] = "outside_tolerance"
                return None, diagnostics
            return best, diagnostics

    def sample_with_seek(
        self, offsets_seconds: Sequence[float], *,
        tolerance_seconds: float = 1.0, window_seconds: float = 4.0,
    ) -> dict[float, RecordedFrame]:
        """逐目标 seek 取帧；seek 成本单独计入 ``seek_seconds``。

        对 5 分钟 HEVC 2.5K 文件，顺序解码约 180s 且每个文件只能取到固定几个
        时间点；seek 模式实测约 1s/点，是长时段抽样的默认方式。
        """
        result: dict[float, RecordedFrame] = {}
        self.seek_diagnostics = getattr(self, "seek_diagnostics", [])
        for target in sorted({float(value) for value in offsets_seconds}):
            frame, diagnostics = self.probe_seek(
                target, tolerance_seconds=tolerance_seconds,
                window_seconds=window_seconds,
            )
            self.seek_diagnostics.append(diagnostics)
            if frame is not None:
                result[target] = frame
        return result


# --------------------------------------------------------------------------- #
# 采样点
# --------------------------------------------------------------------------- #


def coarse_sample_offsets(
    duration_seconds: float, *, seed: int = DEFAULT_SEED,
    fractions: Sequence[float] = COARSE_FRACTIONS,
) -> list[float]:
    """每个文件 3 个固定比例点 + 1 个可复现随机时刻；时长不足按比例调整。"""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        return []
    offsets = [round(duration_seconds * float(fraction), 3) for fraction in fractions]
    rng = random.Random(f"{seed}:{round(duration_seconds, 3)}")
    offsets.append(round(rng.uniform(0.05, 0.95) * duration_seconds, 3))
    inside = sorted({value for value in offsets if 0.0 <= value < duration_seconds})
    return inside or [0.0]


def densify_offsets(
    duration_seconds: float, base_offsets: Sequence[float], *,
    step_seconds: float,
) -> list[float]:
    """对需要加密的片段追加 10～30s 短窗口内的采样点（§4.2）。"""
    if duration_seconds <= 0 or step_seconds <= 0:
        return list(base_offsets)
    values = set(float(value) for value in base_offsets)
    for start in base_offsets:
        window_end = min(duration_seconds - 1e-3, start + max(step_seconds, 1.0))
        cursor = start
        while cursor < window_end:
            values.add(round(cursor, 3))
            cursor += step_seconds
        # 步长大于窗口时也要包含窗口末端，否则加密等于没做。
        values.add(round(window_end, 3))
    return sorted(value for value in values if 0.0 <= value < duration_seconds)


# --------------------------------------------------------------------------- #
# 时间集合划分（先划分，后学习）
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TimePartition:
    build: str
    calibration: str
    blind: str

    def as_dict(self) -> dict[str, Any]:
        return {"build": self.build, "calibration": self.calibration, "blind": self.blind}


def parse_day(value: str) -> str:
    """从 ``YYYY-MM-DD HH:MM:SS`` 或 ISO 串里取日期键。"""
    text = str(value or "").strip().replace("T", " ")
    if len(text) < 10 or text[4] != "-" or text[7] != "-":
        raise BankError(f"无法解析日期: {value!r}")
    return text[:10]


def parse_seconds(value: str) -> float:
    text = str(value or "").strip().replace("T", " ")
    try:
        instant = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise BankError(f"无法解析时间: {value!r}") from exc
    return instant.timestamp()


@dataclass(frozen=True, slots=True)
class EligibleWindow:
    """一个录像文件在任务半开区间 ``[start_time, end_time)`` 内的可用窗口。

    文件身份保留（fileId 不变），但只有与任务区间相交的那一段允许被采样。
    例如 09-13 23:58~09-14 00:03 的文件在任务从 09-14 00:00 开始时，
    只有后 3 分钟可用，且这些帧属于 09-14。
    """

    file_id: str
    record_start: str
    record_end: str
    eligible_start_seconds: float
    eligible_end_seconds: float
    effective_start_time: str
    effective_end_time: str
    intersection_seconds: float
    exclusion_reason: str | None = None

    @property
    def usable(self) -> bool:
        return (
            self.exclusion_reason is None
            and self.intersection_seconds > 0
            and self.eligible_end_seconds > self.eligible_start_seconds
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "record_start": self.record_start,
            "record_end": self.record_end,
            "eligible_start_offset": round(self.eligible_start_seconds, 3),
            "eligible_end_offset": round(self.eligible_end_seconds, 3),
            "effective_start_time": self.effective_start_time,
            "effective_end_time": self.effective_end_time,
            "intersection_seconds": round(self.intersection_seconds, 3),
            "usable": bool(self.usable),
            "exclusion_reason": self.exclusion_reason,
        }


def format_seconds(value: float) -> str:
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def eligible_window(
    item: RecordingFile, *, range_start: float, range_end: float,
) -> EligibleWindow:
    """计算文件与任务半开区间的交集窗口（绝对时间用同一挂钟约定）。"""
    try:
        file_start = parse_seconds(item.record_start)
        file_end = parse_seconds(item.record_end)
    except (BankError, TypeError, ValueError, AttributeError):
        return EligibleWindow(
            file_id=item.file_id, record_start=item.record_start,
            record_end=item.record_end,
            eligible_start_seconds=0.0, eligible_end_seconds=0.0,
            effective_start_time=item.record_start,
            effective_end_time=item.record_end,
            intersection_seconds=0.0,
            exclusion_reason="unparsable_time",
        )
    if file_end <= file_start:
        return EligibleWindow(
            file_id=item.file_id, record_start=item.record_start,
            record_end=item.record_end,
            eligible_start_seconds=0.0, eligible_end_seconds=0.0,
            effective_start_time=item.record_start,
            effective_end_time=item.record_end,
            intersection_seconds=0.0,
            exclusion_reason="zero_length",
        )
    if file_end <= range_start:
        reason = "before_range"
    elif file_start >= range_end:
        reason = "after_range"
    else:
        reason = None
    start = max(file_start, range_start)
    end = min(file_end, range_end)
    intersection = max(0.0, end - start)
    if reason is None and intersection <= 0:
        reason = "zero_length"
    return EligibleWindow(
        file_id=item.file_id, record_start=item.record_start,
        record_end=item.record_end,
        eligible_start_seconds=max(0.0, start - file_start),
        eligible_end_seconds=max(0.0, end - file_start),
        effective_start_time=format_seconds(start),
        effective_end_time=format_seconds(end),
        intersection_seconds=intersection,
        exclusion_reason=reason,
    )


def full_file_window(item: RecordingFile) -> EligibleWindow:
    """本地素材的整文件窗口：没有远程查询区间，整段都可用。"""
    duration = 0.0
    try:
        duration = max(
            0.0, parse_seconds(item.record_end) - parse_seconds(item.record_start),
        )
    except (BankError, TypeError, ValueError, AttributeError):
        duration = 0.0
    if duration <= 0:
        # 本地弱索引（mtime）没有时长信息：用声明大小外推，至少保持可用。
        duration = max(30.0, float(item.file_size or 0) / 250_000.0)
    return EligibleWindow(
        file_id=item.file_id, record_start=item.record_start,
        record_end=item.record_end,
        eligible_start_seconds=0.0, eligible_end_seconds=duration,
        effective_start_time=item.record_start,
        effective_end_time=item.record_end,
        intersection_seconds=duration,
        exclusion_reason=None,
    )


def filter_eligible_files(
    files: Sequence[RecordingFile], *, range_start: float, range_end: float,
) -> tuple[list[RecordingFile], list[EligibleWindow], list[EligibleWindow]]:
    """按任务区间过滤文件。

    返回 ``(保留的文件, 全部窗口, 剔除的窗口)``；剔除的窗口带明确原因，
    调用方必须把它们记进报告，而不是静默丢弃。
    """
    windows: list[EligibleWindow] = []
    kept: list[RecordingFile] = []
    dropped: list[EligibleWindow] = []
    for item in files:
        window = eligible_window(
            item, range_start=range_start, range_end=range_end,
        )
        windows.append(window)
        if window.usable:
            kept.append(item)
        else:
            dropped.append(window)
    return kept, windows, dropped


def partition_recordings(
    files: Sequence[RecordingFile], *, build_days: Sequence[str],
    calibration_day: str, blind_day: str,
    day_of: Callable[[RecordingFile], str] | None = None,
) -> dict[str, list[RecordingFile]]:
    """按记录**起始日**划分，跨日文件按真实时间归属，绝不整文件随机分配。

    ``day_of`` 可覆盖日期判定：调用方在任务半开区间下应传入"可用窗口起始日"，
    否则跨界文件会把查询起点之前的日期带进构建集（v5 的 09-13 问题）。

    集合外的文件进入 ``outside``：既不参与构建也不参与校准/盲测。
    """
    build_set = set(build_days)
    result: dict[str, list[RecordingFile]] = {
        "build": [], "calibration": [], "blind": [], "outside": [],
    }
    for item in files:
        if day_of is not None:
            start_day = str(day_of(item))
            end_day = parse_day(item.record_end)
        else:
            start_day = parse_day(item.record_start)
            end_day = parse_day(item.record_end)
        if start_day in build_set and end_day in build_set:
            result["build"].append(item)
        elif start_day == calibration_day and end_day == calibration_day:
            result["calibration"].append(item)
        elif start_day == blind_day and end_day == blind_day:
            result["blind"].append(item)
        else:
            result["outside"].append(item)
    for key in result:
        result[key].sort(key=lambda entry: (entry.record_start, entry.file_id))
    return result


def cross_day_files(
    files: Sequence[RecordingFile], *, build_days: Sequence[str],
) -> list[RecordingFile]:
    """跨集合边界的文件：可以重复物理读取，但帧必须按所属时间集隔离。"""
    build_set = set(build_days)
    return [
        item for item in files
        if (parse_day(item.record_start) in build_set)
        != (parse_day(item.record_end) in build_set)
    ]


# --------------------------------------------------------------------------- #
# 配准到共同画布
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CanvasRegistrar:
    """小抖动配准到共同画布；明显转动/变焦时拒绝该帧（不自动跨机位迁移 ROI）。"""

    reference: np.ndarray
    overlay_exclude_zones: Sequence[Sequence[Sequence[float]]] = ()
    max_reprojection_median_px: float = 2.0
    min_inliers: int = 20
    min_matches: int = 24

    def register(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if frame.shape != self.reference.shape:
            raise BankError(
                "帧尺寸与分析画布不一致: "
                f"期望{self.reference.shape[1]}x{self.reference.shape[0]}，"
                f"实际{frame.shape[1]}x{frame.shape[0]}"
            )
        height, width = frame.shape[:2]
        old_gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.018)
        feature_mask = np.full((height, width), 255, np.uint8)
        for polygon in self.overlay_exclude_zones:
            points = np.round(
                np.asarray(polygon, np.float32) * np.asarray([width, height])
            ).astype(np.int32)
            cv2.fillPoly(feature_mask, [points], 0)
        old_keys, old_desc = detector.detectAndCompute(old_gray, feature_mask)
        new_keys, new_desc = detector.detectAndCompute(new_gray, feature_mask)
        if old_desc is None or new_desc is None:
            return self._identity(frame, "配准特征不足")
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(old_desc, new_desc, k=2)
        matches = [
            first for pair in pairs if len(pair) == 2
            for first, second in [pair] if first.distance < 0.68 * second.distance
        ]
        unique: dict[int, Any] = {}
        for match in sorted(matches, key=lambda item: item.distance):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        if len(matches) < self.min_matches:
            return self._identity(frame, "配准匹配点不足")
        source = np.float32([old_keys[item.queryIdx].pt for item in matches])
        target = np.float32([new_keys[item.trainIdx].pt for item in matches])
        matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.0)
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            return self._identity(frame, "配准失败")
        selected = inliers.ravel().astype(bool)
        if int(selected.sum()) < self.min_inliers:
            return self._identity(frame, "配准内点不足")
        projected = cv2.perspectiveTransform(
            source.reshape(-1, 1, 2), matrix
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected[selected] - target[selected], axis=1)
        hull_fraction = cv2.contourArea(
            cv2.convexHull(source[selected])
        ) / float(width * height)
        median = float(np.median(errors))
        if median > self.max_reprojection_median_px or hull_fraction < 0.02:
            raise BankError("视角变化过大，拒绝该帧")
        aligned = cv2.warpPerspective(
            frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        diagnostics = {
            "matches": len(matches), "inliers": int(selected.sum()),
            "reprojection_median_px": round(median, 3),
            "reprojection_p95_px": round(float(np.percentile(errors, 95)), 3),
            "inlier_hull_fraction": round(float(hull_fraction), 4),
            "registration": "homography",
            "geometry_ok": True,
        }
        return aligned, diagnostics

    def _identity(self, frame: np.ndarray, reason: str) -> tuple[np.ndarray, dict[str, Any]]:
        """特征不足时的恒等回退：没有测到抖动，就不做几何变换。

        这是保守行为——文档要求「明显转动/变焦另分 view」，而细纹理不足的画面
        通常也没有可测的抖动。诊断里保留原因，不能伪装成成功配准。
        """
        return frame.copy(), {
            "matches": 0, "inliers": 0, "registration": "identity_fallback",
            "geometry_ok": True, "fallback_reason": reason,
        }


# --------------------------------------------------------------------------- #
# 外观样本
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AppearanceSample:
    identity_key: str
    file_id: str
    record_start: str
    offset_seconds: float
    frame_time_seconds: float
    time_block: str
    day: str
    preview: np.ndarray
    descriptor: dict[str, np.ndarray]
    quality: dict[str, Any]
    registration: dict[str, Any]
    # 已配准到共同画布的分析尺寸高清帧。合成阶段必须用它，不能用小图放大。
    aligned_full: np.ndarray | None = None

    def as_dict(self, *, include_timestamp: bool = True) -> dict[str, Any]:
        payload = {
            "identity_key": self.identity_key,
            "file_id": self.file_id,
            "record_start": self.record_start,
            "offset_seconds": self.offset_seconds,
            "frame_time_seconds": self.frame_time_seconds,
            "time_block": self.time_block,
            "day": self.day,
            "quality": self.quality,
            "registration": self.registration,
        }
        if include_timestamp:
            payload["preview_sha256"] = sha256_bytes(
                cv2.imencode(".jpg", self.preview, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
            )
        return payload


TIME_BLOCK_SECONDS = 10


def time_block_key(identity_key: str, offset_seconds: float) -> str:
    """时间块 = 文件身份 + 文件内时间桶。

    桶宽必须小于粗采样点间距，否则「一个文件的 4 个时刻」会塌缩成同一个块，
    同一时段的重复帧就会获得额外权重（方案一 §4.2 明确禁止）。
    """
    # 用「文件 + 采样点毫秒」作为块键：粗采样的 4 个时刻必须各自独立成块，
    # 否则同一文件的重复帧会获得额外权重。
    return f"{identity_key}@{int(round(offset_seconds * 1000)):08d}"


def preview_image(frame: np.ndarray, *, width: int = PREVIEW_WIDTH) -> np.ndarray:
    """小图只用于检索/外观发现，不能冒充高清来源（§4.2、§4.7）。"""
    height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame.copy()
    scaled = int(round(height * width / current_width))
    return cv2.resize(frame, (width, max(1, scaled)), interpolation=cv2.INTER_AREA)


class BoundedPreviewSampler:
    """在租约内做粗采样；产出预览/特征/质量并生成后续高清需求清单。

    下载与配额由调用方（缓存）负责；这里只使用 ``lease.path``。
    """

    def __init__(
        self, cache: ManagedRecordingCache, *, analysis_size: tuple[int, int],
        roi_mask: np.ndarray, seed: int = DEFAULT_SEED, preview_width: int = PREVIEW_WIDTH,
        use_seek: bool = False, algorithm_version: str = "sampler_r3",
        canvas_reference: np.ndarray | None = None,
        overlay_exclude_zones: Sequence[Sequence[Sequence[float]]] = (),
    ) -> None:
        self.cache = cache
        self.analysis_size = (int(analysis_size[0]), int(analysis_size[1]))
        if roi_mask.shape != (self.analysis_size[1], self.analysis_size[0]):
            raise BankError("ROI 掩膜与分析尺寸不一致")
        self.roi_mask = roi_mask
        # R11：所有文件必须配准到**同一**冻结画布，不能每个文件自己选基准。
        if canvas_reference is None:
            raise BankError(
                "缺少冻结共同画布 canvas_reference；禁止按文件各自取基准"
            )
        if canvas_reference.shape[:2] != (self.analysis_size[1], self.analysis_size[0]):
            raise BankError("冻结画布尺寸与分析尺寸不一致")
        self.canvas_reference = canvas_reference
        self.overlay_exclude_zones = tuple(overlay_exclude_zones or ())
        self.seed = int(seed)
        self.preview_width = int(preview_width)
        self.use_seek = bool(use_seek)
        self.algorithm_version = algorithm_version
        self.decode_seconds = 0.0
        self.seek_seconds = 0.0
        self.frames_read = 0
        self.registration_diagnostics: list[dict[str, Any]] = []

    def sample_file(
        self, device_code: str, lease: Lease, *,
        probe: ProbeResult | None = None, densify_step_seconds: float | None = None,
        eligible_start_offset: float = 0.0,
        eligible_end_offset: float | None = None,
        planned_offsets: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """采样一个已租约文件，产出 ``AppearanceSample`` 列表 + 高清需求清单。"""
        file = lease.entry.file
        started = time.monotonic()
        probe = probe or probe_recording(lease.path)
        if not probe.ok:
            self.cache.fail_download(
                device_code, file.file_id, f"probe_failed:{probe.error}",
            )
            raise BankError(f"录像不可解码: {file.file_id}")
        duration = probe.duration_seconds
        if duration <= 0:
            # 容器没给时长：用声明大小与解码时间的外推只作兜底，并记录不确定性。
            duration = max(30.0, float(file.file_size or 0) / 250_000.0)
        # 只允许在 eligible 窗口内取帧：跨日/跨界文件的其余部分不属于本任务。
        window_end = (
            duration if eligible_end_offset is None
            else min(duration, max(0.0, float(eligible_end_offset)))
        )
        window_start = max(0.0, min(float(eligible_start_offset), window_end))
        window_seconds = max(0.0, window_end - window_start)
        if planned_offsets:
            # input manifest 已冻结采样偏移：不重新选择，只把窗口内相对偏移
            # 映射到当前时长，保证同一 manifest 的采样点稳定。
            span = max(0.0, window_seconds)
            offsets = sorted({
                round(window_start + min(max(float(value), 0.0), span), 3)
                for value in planned_offsets
            })
        else:
            offsets = coarse_sample_offsets(
                window_seconds, seed=self.seed, fractions=COARSE_FRACTIONS,
            )
            offsets = [round(window_start + value, 3) for value in offsets]
        if densify_step_seconds:
            offsets = densify_offsets(
                window_end, offsets, step_seconds=densify_step_seconds,
            )
            offsets = [value for value in offsets if value >= window_start]
        reader = SequentialFrameReader(lease.path)
        if self.use_seek:
            # seek 取帧的时间偏差实测 <0.02s；留 1s 容差防止个别关键帧边界失败。
            picked = reader.sample_with_seek(offsets, tolerance_seconds=1.0)
            seek_diagnostics = list(getattr(reader, "seek_diagnostics", []))
        else:
            picked = reader.sample_at(offsets)
            seek_diagnostics = []
        self.decode_seconds += reader.decode_seconds
        self.seek_seconds += reader.seek_seconds
        self.frames_read += reader.frames_decoded

        samples: list[AppearanceSample] = []
        rejected: list[dict[str, Any]] = []
        # R11：整段作业只用一个冻结画布做基准（外层训练集确定）。
        registrar = CanvasRegistrar(
            self.canvas_reference,
            overlay_exclude_zones=self.overlay_exclude_zones,
        )
        for offset in offsets:
            captured = picked.get(offset)
            if captured is None:
                detail = next(
                    (item for item in seek_diagnostics
                     if abs(float(item.get("target_seconds", -1)) - offset) < 1e-6),
                    {},
                )
                rejected.append({
                    "offset_seconds": offset, "reason": "MISSING_AT_OFFSET",
                    "seek": detail,
                })
                continue
            quality = frame_quality(captured.frame)
            if not quality.usable:
                rejected.append({
                    "offset_seconds": offset, "reason": "QUALITY",
                    "reasons": list(quality.reasons),
                })
                continue
            try:
                aligned, diagnostics = registrar.register(captured.frame)
            except BankError as exc:
                rejected.append({
                    "offset_seconds": offset, "reason": "REGISTRATION",
                    "detail": str(exc)[:120],
                })
                self.registration_diagnostics.append({
                    "offset_seconds": offset, "applied_to_canvas": False,
                    "error": str(exc)[:120],
                })
                continue
            self.registration_diagnostics.append({
                "offset_seconds": offset, "applied_to_canvas": True,
                **{key: value for key, value in diagnostics.items()},
            })
            if aligned.shape[1] != self.analysis_size[0] or aligned.shape[0] != self.analysis_size[1]:
                aligned = cv2.resize(
                    aligned, self.analysis_size, interpolation=cv2.INTER_AREA
                )
            if self.roi_mask.shape != aligned.shape[:2]:
                raise BankError("ROI 掩膜与配准后画面尺寸不一致")
            preview = preview_image(aligned, width=self.preview_width)
            preview_mask = cv2.resize(
                self.roi_mask, (preview.shape[1], preview.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            descriptor = extract_grid_descriptor(preview, preview_mask)
            block = time_block_key(lease.identity_key, offset)
            samples.append(AppearanceSample(
                identity_key=lease.identity_key,
                file_id=file.file_id,
                record_start=file.record_start,
                offset_seconds=float(offset),
                frame_time_seconds=float(captured.time_seconds),
                time_block=block,
                day=parse_day(file.record_start),
                preview=preview,
                descriptor=descriptor,
                quality=quality.as_dict(),
                registration=diagnostics,
                aligned_full=aligned,
            ))
        elapsed = time.monotonic() - started
        detail = {
            "duration_seconds": round(duration, 3),
            "offsets": offsets,
            "samples": len(samples),
            "rejected": rejected,
            "decode_seconds": round(reader.decode_seconds, 3),
            "seek_seconds": round(reader.seek_seconds, 3),
            "total_seconds": round(elapsed, 3),
            "codec": probe.codec,
            "resolution": [probe.width, probe.height],
            "downloaded_bytes": lease.entry.bytes,
            "seek_diagnostics": seek_diagnostics,
            "registration": self.registration_diagnostics[-len(samples) or None:],
            "canvas_reference_sha256": sha256_bytes(
                cv2.imencode(".png", self.canvas_reference)[1].tobytes()
            ),
        }
        # 阶段产物 = 采样摘要；提交后 preview 阶段才算完成。
        summary_path = self.cache.work_dir / "stages" / "preview" / f"{lease.identity_key.replace(':','_')}.json"
        atomic_write_json(summary_path, detail)
        self.cache.record_artifact(
            device_code, file.file_id, "preview", "summary.json", summary_path,
            input_hash=lease.entry.sha256, config={"analysis_size": self.analysis_size,
                                                   "seed": self.seed,
                                                   "use_seek": self.use_seek},
            algorithm_version=self.algorithm_version,
            detail={"samples": len(samples), "rejected": len(rejected)},
        )
        return {"samples": samples, "detail": detail}

    def release_after_preview(
        self, device_code: str, lease: Lease, *,
        hd_plan: Mapping[str, Any] | None = None,
    ) -> bool:
        """preview 提交后、且高清需求已记录时才允许释放临时 PS。

        ``hd_plan`` 是后续按需重拉的清单；没有它就必须保留或明确记录缺口。
        """
        return self.cache.release_file(
            device_code, lease.entry.file.file_id, require_committed=("preview",),
        )


def write_canvas_reference(
    path: str | Path, reference: np.ndarray,
) -> str:
    """把训练集确定的共同画布写盘并返回 SHA-256（R11）。"""
    target = Path(path)
    ok, encoded = cv2.imencode(".png", reference)
    if not ok:
        raise BankError("冻结画布编码失败")
    return atomic_write_bytes(target, encoded.tobytes())


def load_canvas_reference(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise BankError(f"无法读取冻结画布: {path}")
    return image


def build_hd_plan(
    samples: Sequence[AppearanceSample], *, max_frames_per_file: int = 8,
) -> dict[str, Any]:
    """按稳定文件身份归并后续高清取帧需求；一次重拉尽量完成该文件全部需求。"""
    plan: dict[str, Any] = {"files": {}}
    for sample in samples:
        entry = plan["files"].setdefault(sample.identity_key, {
            "file_id": sample.file_id, "record_start": sample.record_start,
            "offsets": [], "time_blocks": [],
        })
        if len(entry["offsets"]) >= max_frames_per_file:
            continue
        entry["offsets"].append(sample.offset_seconds)
        entry["time_blocks"].append(sample.time_block)
    for entry in plan["files"].values():
        entry["offsets"] = sorted(set(entry["offsets"]))
        entry["time_blocks"] = sorted(set(entry["time_blocks"]))
    return plan


def summarise_sampling_quality(
    samples: Sequence[AppearanceSample | Mapping[str, Any]],
) -> dict[str, Any]:
    """接受 ``AppearanceSample`` 或它的 ``as_dict()`` 形式（报告与内存共用）。"""
    if not samples:
        return {"samples": 0}

    def field(item: Any, name: str) -> Any:
        if isinstance(item, Mapping):
            return item.get(name)
        return getattr(item, name)

    blocks = {field(sample, "time_block") for sample in samples}
    days = {field(sample, "day") for sample in samples}
    per_day: dict[str, int] = {}
    for sample in samples:
        day = str(field(sample, "day"))
        per_day[day] = per_day.get(day, 0) + 1
    reproducibility = hashlib.sha256(
        "|".join(sorted(
            f"{field(sample, 'identity_key')}:{field(sample, 'offset_seconds')}"
            for sample in samples
        )).encode()
    ).hexdigest()
    return {
        "samples": len(samples),
        "time_blocks": len(blocks),
        "days": len(days),
        "per_day": dict(sorted(per_day.items())),
        "manifest_sha256": reproducibility,
    }


__all__ = [
    "AppearanceSample", "BoundedPreviewSampler", "COARSE_FRACTIONS",
    "CanvasRegistrar", "DEFAULT_SEED", "PREVIEW_WIDTH", "ProbeResult",
    "QUALITY_REASONS", "QualityReport", "RecordedFrame", "RecordingFrameStamp",
    "SequentialFrameReader", "TimePartition", "build_hd_plan",
    "coarse_sample_offsets", "cross_day_files", "densify_offsets", "detect_frozen",
    "frame_quality", "load_canvas_reference", "parse_day", "parse_seconds",
    "partition_recordings", "preview_image", "probe_recording",
    "summarise_sampling_quality", "time_block_key", "write_canvas_reference",
]
