"""Bounded temporal recovery for candidates rejected by an appearance head.

The appearance classifier contributes evidence but does not permanently delete
a Turhancan proposal.  This module is deliberately model-free and keeps only a
small per-camera summary; callers remain responsible for ROI/actor filtering.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot, isfinite
from typing import Iterable


BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    candidate_id: str
    bbox: BBox
    semantic_score: float
    classifier_score: float
    classifier_passed: bool

    def validate(self) -> None:
        x1, y1, x2, y2 = self.bbox
        if not all(isfinite(value) for value in (*self.bbox, self.semantic_score,
                                                  self.classifier_score)):
            raise ValueError("candidate evidence must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("candidate bbox must have positive area")
        if not 0 <= self.semantic_score <= 1:
            raise ValueError("semantic_score must be in [0, 1]")
        if not self.candidate_id:
            raise ValueError("candidate_id is required")


@dataclass(frozen=True, slots=True)
class RecoveryOptions:
    ttl_seconds: float = 3.0
    minimum_hits: int = 3
    maximum_tracks: int = 32
    history_size: int = 8
    minimum_iou: float = 0.20
    maximum_center_scale: float = 0.75
    minimum_area_ratio: float = 0.25
    semantic_score_threshold: float | None = 0.45

    def validate(self) -> None:
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if self.minimum_hits < 2:
            raise ValueError("minimum_hits must be at least 2")
        if self.maximum_tracks < 1 or self.history_size < 1:
            raise ValueError("track and history capacities must be positive")
        if not 0 < self.minimum_iou <= 1:
            raise ValueError("minimum_iou must be in (0, 1]")
        if self.maximum_center_scale <= 0:
            raise ValueError("maximum_center_scale must be positive")
        if not 0 < self.minimum_area_ratio <= 1:
            raise ValueError("minimum_area_ratio must be in (0, 1]")
        if self.semantic_score_threshold is not None and not (
            0 <= self.semantic_score_threshold <= 1
        ):
            raise ValueError("semantic_score_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RecoveryEvent:
    track_id: int
    timestamp: float
    bbox: BBox
    hits: int
    age_seconds: float
    maximum_semantic_score: float
    reason: str
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecoveryObservation:
    baseline_passed: tuple[CandidateEvidence, ...]
    overlap_recovered: tuple[RecoveryEvent, ...]
    temporal_recovered: tuple[RecoveryEvent, ...]
    active_recovered: tuple[RecoveryEvent, ...]
    active_tracks: int
    evicted_tracks: int


@dataclass(slots=True)
class _Track:
    track_id: int
    first_seen: float
    last_seen: float
    bbox: BBox
    hits: int
    maximum_semantic_score: float
    recent_candidate_ids: deque[str]
    emitted: bool = False

    def update(self, timestamp: float, item: CandidateEvidence) -> None:
        self.last_seen = timestamp
        self.bbox = item.bbox
        self.hits += 1
        self.maximum_semantic_score = max(
            self.maximum_semantic_score, item.semantic_score,
        )
        self.recent_candidate_ids.append(item.candidate_id)


class DeferredRecoveryTracker:
    """One bounded tracker for one camera and one recovery policy."""

    def __init__(self, options: RecoveryOptions) -> None:
        options.validate()
        self.options = options
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._last_timestamp: float | None = None
        self._evicted_total = 0

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def observe(
        self, timestamp: float, candidates: Iterable[CandidateEvidence],
    ) -> RecoveryObservation:
        if not isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        self._last_timestamp = timestamp

        rows = tuple(candidates)
        for row in rows:
            row.validate()
        ids = [row.candidate_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique within a tick")

        expired = [
            track_id for track_id, track in self._tracks.items()
            if timestamp - track.last_seen > self.options.ttl_seconds
        ]
        for track_id in expired:
            del self._tracks[track_id]

        passed = tuple(row for row in rows if row.classifier_passed)
        filtered = tuple(row for row in rows if not row.classifier_passed)
        overlap_events: list[RecoveryEvent] = []
        deferred: list[CandidateEvidence] = []
        for row in filtered:
            overlap = next((other for other in passed if same_location(
                row.bbox, other.bbox, self.options,
            )), None)
            if overlap is None:
                deferred.append(row)
                continue
            overlap_events.append(RecoveryEvent(
                track_id=0,
                timestamp=timestamp,
                bbox=_union(row.bbox, overlap.bbox),
                hits=1,
                age_seconds=0.0,
                maximum_semantic_score=max(row.semantic_score, overlap.semantic_score),
                reason="same_frame_pass_overlap",
                candidate_ids=(row.candidate_id, overlap.candidate_id),
            ))

        used_tracks: set[int] = set()
        temporal_events: list[RecoveryEvent] = []
        for row in sorted(deferred, key=lambda item: item.semantic_score, reverse=True):
            matches = [
                (track_id, track) for track_id, track in self._tracks.items()
                if track_id not in used_tracks
                and same_location(track.bbox, row.bbox, self.options)
            ]
            if matches:
                track_id, track = max(
                    matches,
                    key=lambda pair: (_iou(pair[1].bbox, row.bbox),
                                      -_center_distance(pair[1].bbox, row.bbox)),
                )
                track.update(timestamp, row)
            else:
                self._make_room()
                track_id = self._next_track_id
                self._next_track_id += 1
                track = _Track(
                    track_id=track_id,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    bbox=row.bbox,
                    hits=1,
                    maximum_semantic_score=row.semantic_score,
                    recent_candidate_ids=deque(
                        [row.candidate_id], maxlen=self.options.history_size,
                    ),
                )
                self._tracks[track_id] = track
            used_tracks.add(track_id)
            threshold = self.options.semantic_score_threshold
            semantic_ok = threshold is None or track.maximum_semantic_score >= threshold
            if not track.emitted and track.hits >= self.options.minimum_hits and semantic_ok:
                track.emitted = True
                temporal_events.append(RecoveryEvent(
                    track_id=track.track_id,
                    timestamp=timestamp,
                    bbox=track.bbox,
                    hits=track.hits,
                    age_seconds=max(0.0, timestamp - track.first_seen),
                    maximum_semantic_score=track.maximum_semantic_score,
                    reason=("temporal_repeat" if threshold is None
                            else "temporal_repeat_high_semantic"),
                    candidate_ids=tuple(track.recent_candidate_ids),
                ))

        active_recovered = tuple(
            RecoveryEvent(
                track_id=track.track_id,
                timestamp=timestamp,
                bbox=track.bbox,
                hits=track.hits,
                age_seconds=max(0.0, timestamp - track.first_seen),
                maximum_semantic_score=track.maximum_semantic_score,
                reason="active_recovered_track",
                candidate_ids=tuple(track.recent_candidate_ids),
            )
            for track_id, track in self._tracks.items()
            if track_id in used_tracks and track.emitted
        )
        return RecoveryObservation(
            baseline_passed=passed,
            overlap_recovered=tuple(overlap_events),
            temporal_recovered=tuple(temporal_events),
            active_recovered=active_recovered,
            active_tracks=len(self._tracks),
            evicted_tracks=self._evicted_total,
        )

    def _make_room(self) -> None:
        if len(self._tracks) < self.options.maximum_tracks:
            return
        victim = min(
            self._tracks.values(),
            key=lambda track: (track.last_seen, track.hits, track.track_id),
        )
        del self._tracks[victim.track_id]
        self._evicted_total += 1


def same_location(left: BBox, right: BBox, options: RecoveryOptions) -> bool:
    if _iou(left, right) >= options.minimum_iou:
        return True
    left_area, right_area = _area(left), _area(right)
    ratio = min(left_area, right_area) / max(left_area, right_area)
    if ratio < options.minimum_area_ratio:
        return False
    scale = max(_diagonal(left), _diagonal(right))
    return _center_distance(left, right) <= options.maximum_center_scale * scale


def box_overlap_fraction(candidate: BBox, actor: BBox) -> float:
    """Fraction of the candidate covered by an actor box."""
    x1, y1 = max(candidate[0], actor[0]), max(candidate[1], actor[1])
    x2, y2 = min(candidate[2], actor[2]), min(candidate[3], actor[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / max(_area(candidate), 1e-12)


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _diagonal(box: BBox) -> float:
    return hypot(box[2] - box[0], box[3] - box[1])


def _center_distance(left: BBox, right: BBox) -> float:
    return hypot(
        (left[0] + left[2] - right[0] - right[2]) / 2,
        (left[1] + left[3] - right[1] - right[3]) / 2,
    )


def _iou(left: BBox, right: BBox) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(left) + _area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _union(left: BBox, right: BBox) -> BBox:
    return (
        min(left[0], right[0]), min(left[1], right[1]),
        max(left[2], right[2]), max(left[3], right[3]),
    )


__all__ = [
    "BBox", "CandidateEvidence", "DeferredRecoveryTracker", "RecoveryEvent",
    "RecoveryObservation", "RecoveryOptions", "box_overlap_fraction",
    "same_location",
]
