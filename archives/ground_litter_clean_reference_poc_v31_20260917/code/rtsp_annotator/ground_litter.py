"""Independent temporal filter for ground-level litter candidates.

The detector feeding this module may be Turhancan, another model, or a
background-change proposal.  This module deliberately does not classify an
object as litter from one frame: it applies the per-camera ground ROI,
fixed-object exclusions, actor overlap filtering, and persistence rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite
from typing import Iterable

from .event_engine import NormalizedRect


@dataclass(frozen=True, slots=True)
class GroundLitterOptions:
    ground_roi: tuple[tuple[float, float], ...]
    exclude_zones: tuple[tuple[tuple[float, float], ...], ...] = ()
    minimum_confidence: float = 0.20
    persistence_seconds: float = 15.0
    confirm_hits: int = 3
    association_distance: float = 0.06
    max_motion: float = 0.035
    actor_overlap_threshold: float = 0.15
    lost_track_seconds: float = 3.0
    require_background_change: bool = False
    minimum_change_fraction: float = 0.02
    # Deprecated experiment parameter; never causes periodic confirmation.
    cooldown_seconds: float = 300.0

    def validate(self) -> None:
        if len(self.ground_roi) < 3:
            raise ValueError("ground_roi至少需要3个顶点")
        _validate_polygon(self.ground_roi, "ground_roi")
        for index, polygon in enumerate(self.exclude_zones):
            _validate_polygon(polygon, f"exclude_zones[{index}]")
        if not 0 < self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence必须在(0, 1]范围内")
        if self.persistence_seconds <= 0:
            raise ValueError("persistence_seconds必须大于0")
        if self.confirm_hits < 1:
            raise ValueError("confirm_hits必须大于0")
        if not 0 < self.association_distance <= 1:
            raise ValueError("association_distance必须在(0, 1]范围内")
        if not 0 < self.max_motion <= 1:
            raise ValueError("max_motion必须在(0, 1]范围内")
        if not 0 <= self.actor_overlap_threshold <= 1:
            raise ValueError("actor_overlap_threshold必须在[0, 1]范围内")
        if self.lost_track_seconds <= 0 or self.cooldown_seconds < 0:
            raise ValueError("lost_track_seconds必须大于0且cooldown_seconds不能为负")
        if not 0 <= self.minimum_change_fraction <= 1:
            raise ValueError("minimum_change_fraction必须在[0, 1]范围内")


@dataclass(frozen=True, slots=True)
class GroundLitterDetection:
    rectangle: NormalizedRect
    confidence: float
    label: str = "litter"
    background_changed: bool | None = None
    change_fraction: float = 0.0


@dataclass(frozen=True, slots=True)
class GroundLitterActor:
    rectangle: NormalizedRect
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class GroundLitterCandidate:
    candidate_id: int
    rectangle: NormalizedRect
    label: str
    confidence: float
    hits: int
    age_seconds: float
    stationary: bool


@dataclass(frozen=True, slots=True)
class GroundLitterEvent:
    candidate_id: int
    timestamp: float
    rectangle: NormalizedRect
    label: str
    confidence: float
    evidence_hits: int
    persistence_seconds: float


@dataclass(slots=True)
class GroundLitterObservation:
    candidates: tuple[GroundLitterCandidate, ...] = ()
    events: tuple[GroundLitterEvent, ...] = ()


@dataclass(slots=True)
class _Track:
    candidate_id: int
    first_seen: float
    last_seen: float
    last_rect: NormalizedRect
    label: str
    confidence: float
    hits: int = 1
    centers: list[tuple[float, float]] = field(default_factory=list)
    confirmed_at: float | None = None
    last_event_at: float | None = None

    def update(self, timestamp: float, detection: GroundLitterDetection) -> None:
        self.last_seen = timestamp
        self.last_rect = detection.rectangle
        self.label = detection.label
        self.confidence = max(self.confidence, detection.confidence)
        self.hits += 1
        self.centers.append(detection.rectangle.center)
        if len(self.centers) > 30:
            del self.centers[:-30]

    @property
    def age(self) -> float:
        return max(self.last_seen - self.first_seen, 0.0)

    def motion(self) -> float:
        if len(self.centers) < 2:
            return 0.0
        x0, y0 = self.centers[0]
        return max(hypot(x - x0, y - y0) for x, y in self.centers)


class GroundLitterTracker:
    """Small deterministic state machine suitable for a per-camera worker."""

    def __init__(self, options: GroundLitterOptions) -> None:
        options.validate()
        self.options = options
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._last_timestamp: float | None = None

    def observe(
        self,
        timestamp: float,
        detections: Iterable[GroundLitterDetection],
        actors: Iterable[GroundLitterActor] = (),
    ) -> GroundLitterObservation:
        if not isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp不能为负")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamp必须单调递增；每路视频应使用独立tracker")
        if self._last_timestamp == timestamp:
            return GroundLitterObservation()
        self._last_timestamp = timestamp
        # Expire tracks before association so a long gap cannot resurrect an
        # old candidate merely because the next detection is nearby.
        for candidate_id, track in list(self._tracks.items()):
            if timestamp - track.last_seen > self.options.lost_track_seconds:
                del self._tracks[candidate_id]
        actors_tuple = tuple(actors)
        accepted = [
            item
            for item in detections
            if self._accept(item, actors_tuple)
        ]
        used: set[int] = set()
        for detection in accepted:
            match = self._match(detection, used)
            if match is None:
                track = _Track(
                    candidate_id=self._next_id,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    last_rect=detection.rectangle,
                    label=detection.label,
                    confidence=detection.confidence,
                    centers=[detection.rectangle.center],
                )
                self._tracks[self._next_id] = track
                used.add(self._next_id)
                self._next_id += 1
            else:
                self._tracks[match].update(timestamp, detection)
                used.add(match)

        candidates: list[GroundLitterCandidate] = []
        events: list[GroundLitterEvent] = []
        for track in self._tracks.values():
            stationary = track.motion() <= self.options.max_motion
            age = track.age
            candidate = GroundLitterCandidate(
                candidate_id=track.candidate_id,
                rectangle=track.last_rect,
                label=track.label,
                confidence=track.confidence,
                hits=track.hits,
                age_seconds=age,
                stationary=stationary,
            )
            candidates.append(candidate)
            if (
                track.candidate_id in used
                and track.confirmed_at is None
                and age >= self.options.persistence_seconds
                and track.hits >= self.options.confirm_hits
                and stationary
            ):
                track.confirmed_at = timestamp
                track.last_event_at = timestamp
                events.append(
                    GroundLitterEvent(
                        candidate_id=track.candidate_id,
                        timestamp=timestamp,
                        rectangle=track.last_rect,
                        label=track.label,
                        confidence=track.confidence,
                        evidence_hits=track.hits,
                        persistence_seconds=age,
                    )
                )
        return GroundLitterObservation(tuple(candidates), tuple(events))

    def _accept(
        self,
        detection: GroundLitterDetection,
        actors: tuple[GroundLitterActor, ...],
    ) -> bool:
        if detection.confidence < self.options.minimum_confidence:
            return False
        if self.options.require_background_change and (
            detection.background_changed is not True
            or detection.change_fraction < self.options.minimum_change_fraction
        ):
            return False
        point = detection.rectangle.bottom_center
        if not _point_in_polygon(point, self.options.ground_roi):
            return False
        if any(_point_in_polygon(point, zone) for zone in self.options.exclude_zones):
            return False
        return not any(
            _intersection(detection.rectangle, actor.rectangle) / max(
                detection.rectangle.width * detection.rectangle.height, 1e-12)
            >= self.options.actor_overlap_threshold
            for actor in actors
        )

    def _match(
        self,
        detection: GroundLitterDetection,
        used: set[int],
    ) -> int | None:
        center = detection.rectangle.center
        options = [
            (hypot(center[0] - track.last_rect.center[0], center[1] - track.last_rect.center[1]), candidate_id)
            for candidate_id, track in self._tracks.items()
            if candidate_id not in used
        ]
        if not options:
            return None
        distance, candidate_id = min(options)
        if distance <= self.options.association_distance:
            return candidate_id
        return None


def _validate_polygon(polygon: tuple[tuple[float, float], ...], name: str) -> None:
    if len(polygon) < 3 or any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in polygon):
        raise ValueError(f"{name}坐标必须是至少3个[0,1]点")


def _point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (current_y > y) != (previous_y > y):
            x_at_y = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x < x_at_y:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _intersection(left: NormalizedRect, right: NormalizedRect) -> float:
    x0 = max(left.left, right.left)
    y0 = max(left.top, right.top)
    x1 = min(left.left + left.width, right.left + right.width)
    y1 = min(left.top + left.height, right.top + right.height)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _iou(left: NormalizedRect, right: NormalizedRect) -> float:
    intersection = _intersection(left, right)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0
