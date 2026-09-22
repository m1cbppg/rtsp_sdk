"""Validated replay windows and deterministic frame sampling utilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True, slots=True)
class ReplayWindow:
    device_code: str
    start: datetime
    end: datetime
    mode: str
    label: str = ""

    def validate(self, timezone_name: str = "Asia/Shanghai") -> None:
        try:
            timezone = ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"invalid timezone: {timezone_name}") from exc
        if not self.device_code.isdigit() or len(self.device_code) != 20:
            raise ValueError("device_code must be a 20-digit value")
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("window times must include a timezone")
        if self.start.astimezone(timezone) >= self.end.astimezone(timezone):
            raise ValueError("window end must be after start")
        if self.mode not in {"day", "night", "unknown"}:
            raise ValueError("mode must be day, night, or unknown")

    @property
    def duration_seconds(self) -> float:
        return (self.end - self.start).total_seconds()


def parse_time(value: str, timezone_name: str = "Asia/Shanghai") -> datetime:
    """Parse ISO time and attach the configured camera timezone if omitted."""
    try:
        timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"invalid timezone: {timezone_name}") from exc
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO time: {value}") from exc
    return parsed.replace(tzinfo=timezone) if parsed.tzinfo is None else parsed


def sample_times(window: ReplayWindow, sample_fps: float) -> Iterable[datetime]:
    """Yield deterministic sample timestamps, including the first frame."""
    window.validate()
    if not 0 < sample_fps <= 10:
        raise ValueError("sample_fps must be in (0, 10]")
    step = timedelta(seconds=1.0 / sample_fps)
    current = window.start
    while current < window.end:
        yield current
        current += step


def load_windows(path: str | Path, timezone_name: str = "Asia/Shanghai") -> list[ReplayWindow]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("windows"), list):
        raise ValueError("sampling config requires a windows list")
    windows: list[ReplayWindow] = []
    for row in payload["windows"]:
        if not isinstance(row, dict):
            raise ValueError("each sampling window must be an object")
        window = ReplayWindow(
            device_code=str(row.get("device_code", "")),
            start=parse_time(str(row.get("start", "")), timezone_name),
            end=parse_time(str(row.get("end", "")), timezone_name),
            mode=str(row.get("mode", "unknown")),
            label=str(row.get("label", "")),
        )
        window.validate(timezone_name)
        windows.append(window)
    return windows
