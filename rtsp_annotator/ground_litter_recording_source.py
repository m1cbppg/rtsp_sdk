"""回放文件接口适配器 + 有界下载（方案一 §4.4～§4.6）。

职责边界（刻意收窄）：

* 只解析 ``ctseelink/playback/file-urls`` 的**列表**响应：稳定身份（fileId）、真实起止
  时间、字符串 fileSize、临时 url、urlExpireSeconds。旧 ``playback_source.py`` 里的
  ``/devices/rtsp`` 与 ``playback/rtsp/by-time`` 解析器**不能**套用到这里。
* 签名 URL 只在下载上下文中短期存在：不写日志、不写报告、不写 SQLite、不进 Bank。
* 到期刷新：用「请求发起的单调时间 + urlExpireSeconds」保守计时，剩余不足时刷新；
  排队不跨有效期。
* 下载写 ``.part``，边收边计数与算 SHA-256，完成后校验期望大小、探针内容类型，再原子改名。
  Range/断点续传只有在**实测**响应 206 且对象身份一致后才启用；否则丢弃本次 ``.part`` 重拉。

磁盘配额、租约与阶段事务在 ``ground_litter_recording_cache``；本模块只提供
「刷新 URL → 下载一个文件」的原子动作，并把字节数/耗时交给上层记账。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_FILE_URLS_ENDPOINT = (
    "https://qyzcapi.dgjx0769.com/ims-mainte-pc/p-api/v1/monitor/"
    "play/ctseelink/playback/file-urls"
)

# 文档实测值：120s 有效期；起步要求开始下载前至少剩余 30s。
DEFAULT_URL_EXPIRE_SECONDS = 120.0
DEFAULT_REFRESH_MARGIN_SECONDS = 30.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_RANGE_PROBE_BYTES = 1

_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}$")
_MAX_ERROR_BODY = 400


class RecordingSourceError(RuntimeError):
    """来源层失败：请求失败、列表非法、下载失败。消息内**绝不**包含 URL。"""


@dataclass(frozen=True, slots=True)
class RecordingFile:
    """一条稳定录像身份。URL 不在这里。"""

    file_id: str
    file_name: str
    record_start: str
    record_end: str
    file_size: int | None
    file_type: str = ""
    error_message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "file_name": self.file_name,
            "record_start": self.record_start,
            "record_end": self.record_end,
            "file_size": self.file_size,
            "file_type": self.file_type,
        }

    @staticmethod
    def identity_key(device_code: str, file_id: str) -> str:
        if not file_id:
            raise RecordingSourceError("fileId 为空，不能建立稳定身份")
        return f"{device_code}:{file_id}"


@dataclass(frozen=True, slots=True)
class RecordingListEntry:
    """列表查询返回的稳定身份 + 本次临时 URL（只在内存/短期上下文中使用）。"""

    file: RecordingFile
    url: str = field(repr=False, default="")
    url_expire_seconds: float = DEFAULT_URL_EXPIRE_SECONDS
    issued_monotonic: float = field(default=0.0, repr=False)

    def expires_at(self) -> float:
        return self.issued_monotonic + max(0.0, float(self.url_expire_seconds))

    def remaining_seconds(self, now: float | None = None) -> float:
        return self.expires_at() - (time.monotonic() if now is None else now)

    def usable(self, *, margin: float = DEFAULT_REFRESH_MARGIN_SECONDS,
               now: float | None = None) -> bool:
        return bool(self.url) and self.remaining_seconds(now) >= margin


@dataclass(frozen=True, slots=True)
class RecordingListPage:
    entries: tuple[RecordingListEntry, ...]
    response_code: Any
    query_start: str
    query_end: str
    pagination: dict[str, Any] = field(default_factory=dict)
    raw_item_count: int = 0
    requested_monotonic: float = field(default=0.0, repr=False)

    def files(self) -> tuple[RecordingFile, ...]:
        return tuple(entry.file for entry in self.entries)


@dataclass(frozen=True, slots=True)
class ListQuery:
    """规范化后的列表查询请求（时间字符串按 Asia/Shanghai 解释，不带时区）。"""

    device_code: str
    start_time: str
    end_time: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d{20}", str(self.device_code or "")):
            raise RecordingSourceError("deviceCode 必须是 20 位数字")
        for name in ("start_time", "end_time"):
            value = getattr(self, name)
            if not _TIME_RE.match(str(value or "")):
                raise RecordingSourceError(f"{name} 必须是 YYYY-MM-DD HH:MM:SS")
        if self.end_time <= self.start_time:
            raise RecordingSourceError("end_time 必须晚于 start_time")

    def payload(self) -> dict[str, Any]:
        return {
            "deviceCode": self.device_code,
            "startTime": self.start_time,
            "endTime": self.end_time,
        }


@dataclass(frozen=True, slots=True)
class DownloadResult:
    file_id: str
    path: Path
    size: int
    sha256: str
    elapsed_seconds: float
    response_status: int
    resumed: bool = False
    range_supported: bool | None = None
    attempts: int = 1
    content_type: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "size": self.size,
            "sha256": self.sha256,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "response_status": self.response_status,
            "resumed": self.resumed,
            "range_supported": self.range_supported,
            "attempts": self.attempts,
            "content_type": self.content_type,
        }


class UrlRefreshPolicy:
    """单调时间 + 声明有效期的保守刷新策略。"""

    def __init__(self, *, margin_seconds: float = DEFAULT_REFRESH_MARGIN_SECONDS) -> None:
        if not math.isfinite(margin_seconds) or margin_seconds < 0:
            raise RecordingSourceError("刷新余量必须是非负有限值")
        self.margin_seconds = float(margin_seconds)
        self.refresh_count = 0
        self.expired_events = 0

    def needs_refresh(self, entry: RecordingListEntry | None, *, now: float | None = None) -> bool:
        if entry is None or not entry.url:
            return True
        if not entry.usable(margin=self.margin_seconds, now=now):
            self.expired_events += 1
            return True
        return False

    def mark_refreshed(self) -> None:
        self.refresh_count += 1


def _parse_size(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise RecordingSourceError("fileSize 类型非法")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise RecordingSourceError("fileSize 超出范围")
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if not re.fullmatch(r"\d+", text):
        raise RecordingSourceError("fileSize 必须是数字字符串")
    return int(text)


def _parse_record_time(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not _TIME_RE.match(text):
        raise RecordingSourceError(f"{field_name} 缺失或格式非法")
    return text.replace("T", " ")


def parse_file_urls_page(
    payload: Any, query: ListQuery, *, requested_monotonic: float | None = None,
) -> RecordingListPage:
    """解析 file-urls 列表响应；业务 code 非 200 或结构非法即报错。"""
    if not isinstance(payload, Mapping):
        raise RecordingSourceError("file-urls 响应不是 JSON 对象")
    code = payload.get("code")
    if code != 200:
        raise RecordingSourceError(f"file-urls 业务码非 200: {code!r}")
    data = payload.get("data")
    items: list[Any]
    pagination: dict[str, Any] = {}
    if isinstance(data, list):
        items = data
    elif isinstance(data, Mapping):
        for key in ("records", "list", "items", "fileList", "rows"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                items = list(candidate)
                break
        else:
            raise RecordingSourceError("file-urls data 中没有文件数组")
        for key in ("total", "pages", "pageNum", "pageSize", "current", "size", "hasNext"):
            if key in data:
                pagination[key] = data[key]
    else:
        raise RecordingSourceError("file-urls data 结构非法")

    issued = time.monotonic() if requested_monotonic is None else float(requested_monotonic)
    entries: list[RecordingListEntry] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise RecordingSourceError("file-urls 单项不是对象")
        file_id = str(item.get("fileId") or "").strip()
        if not file_id:
            raise RecordingSourceError("file-urls 单项缺少 fileId")
        size = _parse_size(item.get("fileSize"))
        expire = item.get("urlExpireSeconds", DEFAULT_URL_EXPIRE_SECONDS)
        try:
            expire_seconds = float(expire)
        except (TypeError, ValueError):
            raise RecordingSourceError("urlExpireSeconds 非法") from None
        if not math.isfinite(expire_seconds) or expire_seconds <= 0:
            raise RecordingSourceError("urlExpireSeconds 必须为正")
        entries.append(RecordingListEntry(
            file=RecordingFile(
                file_id=file_id,
                file_name=str(item.get("fileName") or ""),
                record_start=_parse_record_time(item.get("recordStartTime"), "recordStartTime"),
                record_end=_parse_record_time(item.get("recordEndTime"), "recordEndTime"),
                file_size=size,
                file_type=str(item.get("fileType") or ""),
                error_message=str(item.get("errorMessage") or ""),
            ),
            url=str(item.get("url") or ""),
            url_expire_seconds=expire_seconds,
            issued_monotonic=issued,
        ))
    entries.sort(key=lambda entry: (entry.file.record_start, entry.file.file_id))
    return RecordingListPage(
        entries=tuple(entries), response_code=code,
        query_start=query.start_time, query_end=query.end_time,
        pagination=pagination, raw_item_count=len(items),
        requested_monotonic=issued,
    )


class RecordingListClient:
    """file-urls 列表查询客户端。``opener`` 可注入以便确定性测试。"""

    def __init__(
        self, endpoint: str = DEFAULT_FILE_URLS_ENDPOINT, *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        headers: Mapping[str, str] | None = None,
        opener: Callable[..., Any] = urlopen,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not str(endpoint).startswith(("http://", "https://")):
            raise RecordingSourceError("endpoint 必须是 HTTP(S)")
        if not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise RecordingSourceError("timeout 必须在 (0, 120]")
        self.endpoint = str(endpoint)
        self.timeout = float(timeout)
        # 鉴权头从运行配置注入；不回显、不落盘。
        self._headers = {str(k): str(v) for k, v in (headers or {}).items()}
        self._opener = opener
        self._clock = clock
        self.request_count = 0

    def _post(self, payload: Mapping[str, Any]) -> Any:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers.update(self._headers)
        request = Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with self._opener(request, timeout=self.timeout) as response:
                status = getattr(response, "status", 200) or 200
                raw = response.read()
        except HTTPError as exc:
            raise RecordingSourceError(f"file-urls HTTP 错误: {exc.code}") from None
        except URLError as exc:
            raise RecordingSourceError(f"file-urls 网络错误: {exc.reason!r}") from None
        except RecordingSourceError:
            raise
        except Exception:
            # 异常链可能包含签名 URL，不向上传播原始异常。
            raise RecordingSourceError("file-urls 请求失败") from None
        if status != 200:
            raise RecordingSourceError(f"file-urls HTTP 状态异常: {status}")
        try:
            decoded = json.loads(raw.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeError):
            snippet = raw[:_MAX_ERROR_BODY].decode("utf-8", errors="replace")
            raise RecordingSourceError(
                f"file-urls 200 但不是合法 JSON（错误正文预览: {snippet[:80]!r}）"
            ) from None
        return decoded

    def query(self, query: ListQuery) -> RecordingListPage:
        self.request_count += 1
        requested = self._clock()
        payload = self._post(query.payload())
        return parse_file_urls_page(payload, query, requested_monotonic=requested)

    def query_hour(self, device_code: str, hour_start: str, hour_end: str) -> RecordingListPage:
        return self.query(ListQuery(device_code, hour_start, hour_end))


def deduplicate_files(
    pages: Iterable[RecordingListPage], device_code: str,
) -> list[RecordingFile]:
    """按 ``deviceCode + fileId`` 去重，保留最早出现的元数据与真实时间排序。

    重复物理读取是允许的（跨窗口边界可重叠）；这里只保证**身份**不重复。
    """
    seen: dict[str, RecordingFile] = {}
    for page in pages:
        for entry in page.entries:
            key = RecordingFile.identity_key(device_code, entry.file.file_id)
            if key not in seen:
                seen[key] = entry.file
    return sorted(seen.values(), key=lambda item: (item.record_start, item.file_id))


def detect_truncation(
    one_hour: Sequence[RecordingFile], half_hours: Sequence[RecordingFile],
) -> dict[str, Any]:
    """比较一小时窗口与两个半小时窗口的 fileId 并集，检测列表截断嫌疑。

    只是嫌疑提示：服务端分页/边界舍入也可能造成差异，必须记录而不是静默接受。
    """
    hour_ids = {item.file_id for item in one_hour}
    half_ids = {item.file_id for item in half_hours}
    return {
        "hour_count": len(hour_ids),
        "half_hour_count": len(half_ids),
        "missing_from_hour": sorted(half_ids - hour_ids),
        "extra_in_hour": sorted(hour_ids - half_ids),
        "suspected_truncation": bool(half_ids - hour_ids),
    }


# --------------------------------------------------------------------------- #
# 下载
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RangeProbe:
    supported: bool
    status: int | None
    content_range: str = ""
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported, "status": self.status,
            "detail": self.detail,
        }


class RangeUnsupported(RecordingSourceError):
    """对端不支持 Range；调用方必须整文件重拉而不是拼接。"""


class RecordingDownloader:
    """有界下载器：把一条临时 URL 落成受管缓存中的完整文件。

    这里**不做**配额决策与租约；调用方（缓存）必须先预留空间、再把目标路径交进来。
    ``probe_range`` 的结果会被缓存，避免每个文件都重复探测。
    """

    def __init__(
        self, *, timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        opener: Callable[..., Any] = urlopen,
        clock: Callable[[], float] = time.monotonic,
        expected_content_types: Sequence[str] = (),
    ) -> None:
        if max_attempts < 1:
            raise RecordingSourceError("max_attempts 必须为正")
        self.timeout = float(timeout)
        self.max_attempts = int(max_attempts)
        self._opener = opener
        self._clock = clock
        self._expected = tuple(expected_content_types)
        self._range_cache: dict[str, RangeProbe] = {}
        self.downloaded_bytes = 0
        self.download_seconds = 0.0
        self.attempts_total = 0
        self.range_fallbacks = 0

    # -- Range 探测 ------------------------------------------------------- #

    def probe_range(self, url: str, *, refresh: bool = False) -> RangeProbe:
        key = urlsplit(url).hostname or ""
        if not refresh and key in self._range_cache:
            return self._range_cache[key]
        request = Request(url, headers={"Range": "bytes=0-0"})
        probe = RangeProbe(supported=False, status=None, detail="未知")
        try:
            with self._opener(request, timeout=self.timeout) as response:
                status = int(getattr(response, "status", 200) or 200)
                content_range = str(response.headers.get("Content-Range", "") or "")
                response.read(1)
            if status == 206 and content_range:
                probe = RangeProbe(True, status, content_range, "206 + Content-Range")
            else:
                probe = RangeProbe(False, status, content_range, f"status={status}")
        except HTTPError as exc:
            probe = RangeProbe(False, exc.code, "", f"HTTP {exc.code}")
        except Exception:
            probe = RangeProbe(False, None, "", "probe failed")
        self._range_cache[key] = probe
        return probe

    # -- 下载 ------------------------------------------------------------- #

    def download(
        self, url: str, target: str | Path, *,
        expected_size: int | None = None,
        allow_resume: bool = False,
        progress: Callable[[int], None] | None = None,
        budget_hook: Callable[[int], None] | None = None,
    ) -> DownloadResult:
        """下载到 ``target``（原子改名完成）。失败时清理 ``.part`` 并抛错。

        ``allow_resume=True`` 时，只有 ``probe_range`` 实测支持且 ``.part`` 已存在
        才续传；否则删掉 ``.part`` 整文件重拉——绝不拼接两个不同版本的录像。
        """
        started = self._clock()
        destination = Path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_name(destination.name + ".part")
        last_error: Exception | None = None
        resumed = False
        status = 0
        content_type = ""
        attempts = 0
        range_supported: bool | None = None

        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            self.attempts_total += 1
            offset = 0
            resumed = False
            if allow_resume and part.is_file():
                probe = self.probe_range(url)
                range_supported = probe.supported
                if probe.supported:
                    offset = part.stat().st_size
                    resumed = offset > 0
                else:
                    self.range_fallbacks += 1
                    part.unlink(missing_ok=True)
            elif part.exists():
                part.unlink()
            headers: dict[str, str] = {}
            if resumed:
                headers["Range"] = f"bytes={offset}-"
            request = Request(url, headers=headers)
            digest = hashlib.sha256()
            written = offset
            if resumed:
                with part.open("rb") as existing:
                    for block in iter(lambda: existing.read(1024 * 1024), b""):
                        digest.update(block)
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    status = int(getattr(response, "status", 200) or 200)
                    content_type = str(response.headers.get("Content-Type", "") or "")
                    if resumed and status != 206:
                        raise RangeUnsupported(
                            f"续传请求返回 {status}，不是 206；丢弃 .part 重拉"
                        )
                    mode = "ab" if resumed else "wb"
                    with part.open(mode) as stream:
                        while True:
                            block = response.read(1024 * 256)
                            if not block:
                                break
                            stream.write(block)
                            digest.update(block)
                            written += len(block)
                            if budget_hook is not None:
                                budget_hook(len(block))
                            if progress is not None:
                                progress(written)
                        stream.flush()
                        os.fsync(stream.fileno())
            except RangeUnsupported as exc:
                last_error = exc
                part.unlink(missing_ok=True)
                continue
            except HTTPError as exc:
                last_error = RecordingSourceError(f"下载 HTTP 错误: {exc.code}")
                # 401/403/404/410 基本是签名过期或远端已删；由调用方刷新重试。
                part.unlink(missing_ok=True)
                continue
            except Exception:
                last_error = RecordingSourceError("下载中断")
                continue

            if expected_size is not None and written != expected_size:
                last_error = RecordingSourceError(
                    f"下载字节数不符: 期望 {expected_size}，实际 {written}"
                )
                part.unlink(missing_ok=True)
                continue
            if self._expected and content_type:
                if not any(token in content_type for token in self._expected):
                    last_error = RecordingSourceError(
                        f"响应内容类型不符合期望: {content_type!r}"
                    )
                    part.unlink(missing_ok=True)
                    continue
            if written == 0:
                last_error = RecordingSourceError("下载得到空文件")
                part.unlink(missing_ok=True)
                continue

            digest_value = digest.hexdigest()
            os.replace(part, destination)
            elapsed = self._clock() - started
            self.downloaded_bytes += written
            self.download_seconds += elapsed
            return DownloadResult(
                file_id="", path=destination, size=written, sha256=digest_value,
                elapsed_seconds=elapsed, response_status=status, resumed=resumed,
                range_supported=range_supported, attempts=attempts,
                content_type=content_type,
            )
        part.unlink(missing_ok=True)
        self.download_seconds += self._clock() - started
        raise last_error or RecordingSourceError("下载失败")

    def fetch_url_for_file(
        self, client: RecordingListClient, query: ListQuery, file_id: str,
        *, policy: UrlRefreshPolicy | None = None, now: float | None = None,
    ) -> RecordingListEntry:
        """重新查询覆盖该文件的小时间窗，按 fileId 取回**新** URL。

        这是「临近下载时刷新」的唯一入口；不要在作业开始时缓存七天签名地址。
        """
        policy = policy or UrlRefreshPolicy()
        page = client.query(query)
        policy.mark_refreshed()
        for entry in page.entries:
            if entry.file.file_id == file_id:
                if not entry.url:
                    raise RecordingSourceError("刷新后该文件仍没有可用 URL")
                if not entry.usable(margin=policy.margin_seconds, now=now):
                    raise RecordingSourceError("刷新得到的 URL 剩余有效期不足")
                return entry
        raise RecordingSourceError("刷新窗口内找不到该 fileId")


def file_looks_like_media(path: str | Path) -> bool:
    """轻量内容探针：拒绝明显不是容器/PS 的响应正文（例如 200 + JSON 错误）。"""
    try:
        with Path(path).open("rb") as stream:
            head = stream.read(16)
    except OSError:
        return False
    if len(head) < 8:
        return False
    if head.startswith(b"\x00\x00\x00") and head[4:8] in (
        b"ftyp", b"moov", b"mdat", b"free", b"skip",
    ):
        return True
    if head[:4] in (b"\x1a\x45\xdf\xa3", b"RIFF", b"OggS", b"FLV\x01"):
        return True
    if head.startswith(b"\xff\xd8\xff"):
        return True
    # PS/TS 流常见 0x47 同步字节，或平台自有的私有头。
    if head[:1] == b"\x47":
        return True
    if head[:4] in (b"PS\x00\x00", b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"):
        return True
    # 允许未知容器（平台可能使用私有封装），但不能是纯文本错误正文。
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return True
    printable = sum(1 for byte in head if 32 <= byte < 127 or byte in (9, 10, 13))
    return printable < len(head)


__all__ = [
    "DEFAULT_FILE_URLS_ENDPOINT", "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_REFRESH_MARGIN_SECONDS", "DEFAULT_URL_EXPIRE_SECONDS",
    "DownloadResult", "ListQuery", "RangeProbe", "RangeUnsupported",
    "RecordingDownloader", "RecordingFile", "RecordingListClient",
    "RecordingListEntry", "RecordingListPage", "RecordingSourceError",
    "UrlRefreshPolicy", "deduplicate_files", "detect_truncation",
    "file_looks_like_media", "parse_file_urls_page",
]
