"""Secret-safe adapter for the documented playback URL API.

The monitor platform has used more than one request wrapper.  Callers supply
the request payload described by the current API document; this module only
handles HTTP, response validation, and URL redaction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Callable
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class PlaybackRequest:
    device_code: str
    start: str
    end: str
    payload: dict[str, Any] = field(repr=False)

    def validate(self) -> None:
        if not self.device_code.isdigit() or len(self.device_code) != 20:
            raise ValueError("device_code must be a 20-digit value")
        if not self.start or not self.end or not isinstance(self.payload, dict):
            raise ValueError("start, end, and payload are required")


def build_ctseelink_playback_payload(device_code: str, playback_time: str,
                                     *, mute: int = 1,
                                     net_type: int = 0) -> dict[str, Any]:
    """Body for CtseeLinkPlaybackByTimeRequest at playback/rtsp/by-time.

    devices/rtsp is live-only, despite the PC wrapper advertising playback
    fields. It must never be used to claim a historical recording was read.
    """
    if not device_code.isdigit() or len(device_code) != 20:
        raise ValueError("device_code must be a 20-digit value")
    if not playback_time:
        raise ValueError("playback_time is required")
    try:
        instant = datetime.fromisoformat(playback_time)
    except ValueError:
        raise ValueError("playback_time must be an ISO camera-local datetime") from None
    if instant.tzinfo is not None:
        instant = instant.astimezone(ZoneInfo('Asia/Shanghai')).replace(tzinfo=None)
    if mute not in (0, 1) or net_type not in (0, 1):
        raise ValueError("mute and net_type must be 0 or 1")
    return {
        "deviceCode": device_code,
        "mute": mute,
        "netType": net_type,
        "playbackTime": instant.strftime('%Y-%m-%d %H:%M:%S'),
    }


@dataclass(frozen=True, slots=True)
class PlaybackRecording:
    url: str = field(repr=False)
    record_start: str
    record_end: str
    offset_seconds: float

    def metadata(self) -> dict:
        return {'record_start':self.record_start,'record_end':self.record_end,
                'offset_seconds':self.offset_seconds,'content_time_verified':False}


def redact_url(value: str) -> str:
    """Return a constant marker; camera addresses are sensitive too."""
    return "<redacted>"


def _find_url(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"url", "rtspurl", "rtsp_url", "playurl", "play_url"}:
                if isinstance(item, str) and item.startswith(("rtsp://", "rtsps://")):
                    return item
            found = _find_url(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_url(item)
            if found:
                return found
    return None


class PlaybackUrlClient:
    def __init__(self, endpoint: str, *, timeout: float = 12,
                 opener: Callable[..., Any] = urlopen) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be HTTP or HTTPS")
        if not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError("timeout must be in (0, 60]")
        self.endpoint = endpoint
        self.timeout = timeout
        self.opener = opener

    def _request(self, request: PlaybackRequest) -> dict:
        request.validate()
        body = json.dumps(request.payload, ensure_ascii=False).encode("utf-8")
        http_request = Request(self.endpoint, data=body, headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        try:
            with self.opener(http_request, timeout=self.timeout) as response:
                payload = json.load(response)
        except Exception:
            # Exception chains from HTTP clients can include signed URLs.
            raise RuntimeError("playback URL request failed") from None
        if not isinstance(payload, dict) or payload.get("code") != 200:
            raise RuntimeError("playback URL request returned a non-success code")
        return payload

    def resolve(self, request: PlaybackRequest) -> str:
        """Legacy URL extraction only; does not verify historical content."""
        payload = self._request(request)
        url = _find_url(payload.get("data"))
        if not url:
            raise RuntimeError("playback response did not contain an RTSP URL")
        return url

    def resolve_recording(self, request: PlaybackRequest) -> PlaybackRecording:
        """Require recording metadata; reject live responses before decoding.

        The returned file may start before the requested time. Its offset is
        retained, never silently treated as a server-side seek. OSD/content
        verification is still required; metadata alone is not ground truth.
        """
        if self.endpoint.rstrip('/').endswith('/devices/rtsp'):
            raise ValueError('devices/rtsp is live-only; use playback/rtsp/by-time')
        payload = self._request(request)
        data = payload.get('data')
        if not isinstance(data,dict):
            raise RuntimeError('playback response missing recording metadata')
        try:
            start = datetime.strptime(data['recordStartTime'],'%Y-%m-%d %H:%M:%S')
            end = datetime.strptime(data['recordEndTime'],'%Y-%m-%d %H:%M:%S')
            wanted = datetime.strptime(request.payload['playbackTime'],'%Y-%m-%d %H:%M:%S')
            offset = float(data['offsetSeconds'])
        except (KeyError,TypeError,ValueError):
            raise RuntimeError('playback response missing valid recording metadata') from None
        if not start <= wanted <= end or end <= start or not math.isfinite(offset) or abs(offset-(wanted-start).total_seconds())>1:
            raise RuntimeError('playback recording time does not cover requested time')
        url = _find_url(data)
        if not url:
            raise RuntimeError('playback response did not contain an RTSP URL')
        return PlaybackRecording(url,start.isoformat(),end.isoformat(),offset)
