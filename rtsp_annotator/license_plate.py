from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass, field


PRIMARY_DETECTOR_UID = 1
LICENSE_PLATE_DETECTOR_UID = 2
LICENSE_PLATE_RECOGNIZER_UID = 3

DEFAULT_VEHICLE_CLASSES = (2, 3, 5, 7)

_CHINESE_PLATE = re.compile(
    r"^[\u4e00-\u9fff][A-HJ-NP-Z][A-HJ-NP-Z0-9]{5,6}$"
)


def normalize_chinese_plate(value: str | bytes) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    normalized = re.sub(r"[\s·•._-]+", "", value).upper()
    if not _CHINESE_PLATE.fullmatch(normalized):
        return None
    return normalized


@dataclass(frozen=True, slots=True)
class LicensePlateOptions:
    enabled: bool = False
    detector_interval: int = 0
    recognition_reinfer_interval: int = 15
    minimum_confirmations: int = 2
    minimum_plate_confidence: float = 0.5
    vehicle_classes: tuple[int, ...] = DEFAULT_VEHICLE_CLASSES


@dataclass(slots=True)
class _TrackVotes:
    values: deque[str]
    last_frame: int
    confirmed: str | None = None


@dataclass(slots=True)
class PlateConsensus:
    minimum_confirmations: int = 2
    window_size: int = 5
    expire_after_frames: int = 250
    _tracks: dict[tuple[int, int], _TrackVotes] = field(
        default_factory=dict
    )

    def observe(
        self,
        *,
        pad_index: int,
        track_id: int,
        frame_number: int,
        value: str | bytes,
    ) -> str | None:
        normalized = normalize_chinese_plate(value)
        if normalized is None or track_id < 0:
            return None
        key = (pad_index, track_id)
        state = self._tracks.get(key)
        if state is None:
            state = _TrackVotes(
                values=deque(maxlen=self.window_size),
                last_frame=frame_number,
            )
            self._tracks[key] = state
        state.last_frame = frame_number
        state.values.append(normalized)
        candidate, votes = Counter(state.values).most_common(1)[0]
        if votes >= self.minimum_confirmations:
            state.confirmed = candidate
        self._expire(frame_number)
        return state.confirmed

    def get(
        self,
        *,
        pad_index: int,
        track_id: int,
        frame_number: int,
    ) -> str | None:
        state = self._tracks.get((pad_index, track_id))
        if state is None:
            return None
        if frame_number - state.last_frame > self.expire_after_frames:
            self._tracks.pop((pad_index, track_id), None)
            return None
        return state.confirmed

    def _expire(self, frame_number: int) -> None:
        if len(self._tracks) < 256:
            return
        expired = [
            key
            for key, state in self._tracks.items()
            if frame_number - state.last_frame > self.expire_after_frames
        ]
        for key in expired:
            self._tracks.pop(key, None)
