from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Iterable

from .events import (
    EventDetectionOptions,
    EventRecord,
    EventRepository,
    EventRoiOptions,
)


@dataclass(frozen=True, slots=True)
class NormalizedRect:
    left: float
    top: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (
            self.left + self.width / 2,
            self.top + self.height / 2,
        )

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (self.left + self.width / 2, self.top + self.height)


@dataclass(frozen=True, slots=True)
class TrackedObject:
    track_id: int
    class_id: int
    rectangle: NormalizedRect


@dataclass(frozen=True, slots=True)
class ActorOverlay:
    track_id: int
    roi_id: str
    label: str
    state: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class GarbageDetection:
    rectangle: NormalizedRect
    object_type: str
    confidence: float


@dataclass(frozen=True, slots=True)
class GarbageSnapshot:
    timestamp: float
    frame_number: int
    area_ratio: float
    regions: tuple[NormalizedRect, ...] = ()
    semantic_confidence: float = 0.0
    object_type: str = "垃圾"
    visual_change_ratio: float = 0.0
    visual_regions: tuple[NormalizedRect, ...] = ()
    detections: tuple[GarbageDetection, ...] = ()


@dataclass(frozen=True, slots=True)
class GarbageOverlay:
    roi_id: str
    rectangle: NormalizedRect | None
    label: str
    state: str
    elapsed_seconds: float


@dataclass(slots=True)
class EventEngineResult:
    actor_overlays: list[ActorOverlay] = field(default_factory=list)
    garbage_overlays: list[GarbageOverlay] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)


@dataclass(slots=True)
class _DwellState:
    entered_at: float
    last_seen_at: float
    actor_type: str
    alerted: bool = False


@dataclass(slots=True)
class _ActorInteraction:
    actor_type: str
    trail: list[NormalizedRect] = field(default_factory=list)


@dataclass(slots=True)
class _GarbageRoiState:
    baseline: GarbageSnapshot | None = None
    baseline_started_at: float = 0.0
    actors_during_interaction: dict[int, _ActorInteraction] = field(
        default_factory=dict
    )
    last_actor_seen_at: float | None = None
    candidate_kind: str | None = None
    candidate_started_at: float = 0.0
    candidate_snapshot: GarbageSnapshot | None = None
    last_confirmed: GarbageOverlay | None = None
    last_confirmed_at: float = 0.0
    last_confirmed_persistent: bool = False


class EventEngine:
    """Deterministic temporal event engine independent from model runtime."""

    _BASELINE_WARMUP_SECONDS = 3.0

    def __init__(
        self,
        *,
        stream_id: str,
        options: EventDetectionOptions,
        repository: EventRepository | None = None,
        on_event: Callable[[EventRecord], None] | None = None,
    ) -> None:
        options.validate()
        self.stream_id = stream_id
        self.options = options
        self._repository = repository
        self._on_event = on_event
        self._lock = threading.RLock()
        self._dwell: dict[tuple[str, int], _DwellState] = {}
        self._garbage = {
            roi.roi_id: _GarbageRoiState() for roi in options.rois
        }

    def observe_tracks(
        self,
        *,
        timestamp: float,
        objects: Iterable[TrackedObject],
    ) -> EventEngineResult:
        result = EventEngineResult()
        if not self.options.enabled:
            return result
        actors = [item for item in objects if self._actor_type(item) is not None]
        with self._lock:
            seen: set[tuple[str, int]] = set()
            for actor in actors:
                actor_type = self._actor_type(actor)
                assert actor_type is not None
                for roi in self.options.rois:
                    if not _point_in_polygon(
                        actor.rectangle.bottom_center,
                        roi.polygon,
                    ):
                        continue
                    key = (roi.roi_id, actor.track_id)
                    seen.add(key)
                    state = self._dwell.get(key)
                    if state is None:
                        state = _DwellState(timestamp, timestamp, actor_type)
                        self._dwell[key] = state
                    state.last_seen_at = timestamp
                    garbage_state = self._garbage[roi.roi_id]
                    interaction = garbage_state.actors_during_interaction.get(
                        actor.track_id
                    )
                    if interaction is None:
                        interaction = _ActorInteraction(actor_type)
                        garbage_state.actors_during_interaction[
                            actor.track_id
                        ] = interaction
                    if (
                        not interaction.trail
                        or math.dist(
                            interaction.trail[-1].bottom_center,
                            actor.rectangle.bottom_center,
                        )
                        >= 0.01
                    ):
                        interaction.trail.append(actor.rectangle)
                        del interaction.trail[:-64]
                    garbage_state.last_actor_seen_at = timestamp
                    if not roi.dwell_enabled:
                        continue
                    elapsed = max(timestamp - state.entered_at, 0.0)
                    threshold = self._dwell_threshold(roi, actor_type)
                    if elapsed >= threshold:
                        overlay_state = "confirmed"
                        label = f"{_actor_label(actor_type)}区域停留 {int(elapsed)}秒"
                        if not state.alerted:
                            state.alerted = True
                            event = EventRecord.create(
                                stream_id=self.stream_id,
                                event_type="zone_dwell",
                                roi_id=roi.roi_id,
                                message=(
                                    f"{_actor_label(actor_type)}在区域内"
                                    f"停留超过{threshold:g}秒"
                                ),
                                actor_type=actor_type,
                                actor_track_id=actor.track_id,
                                metadata={"dwell_seconds": elapsed},
                            )
                            self._emit(event, result)
                    else:
                        overlay_state = "observing"
                        label = f"{_actor_label(actor_type)}停留 {int(elapsed)}秒"
                    result.actor_overlays.append(
                        ActorOverlay(
                            track_id=actor.track_id,
                            roi_id=roi.roi_id,
                            label=label,
                            state=overlay_state,
                            elapsed_seconds=elapsed,
                        )
                    )
            self._expire_tracks(timestamp, seen)
            for roi in self.options.rois:
                state = self._garbage[roi.roi_id]
                if (
                    state.candidate_kind is not None
                    and state.candidate_snapshot is not None
                ):
                    elapsed = max(
                        timestamp - state.candidate_started_at,
                        0.0,
                    )
                    result.garbage_overlays.append(
                        GarbageOverlay(
                            roi_id=roi.roi_id,
                            rectangle=_union_rect(
                                state.candidate_snapshot.regions
                                or state.candidate_snapshot.visual_regions
                            ),
                            label=_candidate_label(
                                state.candidate_kind,
                                elapsed,
                                roi.rules.garbage_persistence_seconds,
                            ),
                            state="candidate",
                            elapsed_seconds=elapsed,
                        )
                    )
                if state.last_confirmed is not None and (
                    state.last_confirmed_persistent
                    or timestamp - state.last_confirmed_at <= 10.0
                ):
                    result.garbage_overlays.append(state.last_confirmed)
        return result

    def observe_garbage(
        self,
        *,
        roi_id: str,
        snapshot: GarbageSnapshot,
    ) -> EventEngineResult:
        result = EventEngineResult()
        if not self.options.enabled or not self.options.garbage.enabled:
            return result
        roi = self._roi(roi_id)
        if not roi.garbage_enabled:
            return result
        with self._lock:
            state = self._garbage[roi_id]
            if state.baseline is None:
                state.baseline = snapshot
                state.baseline_started_at = snapshot.timestamp
                return result
            if (
                snapshot.timestamp - state.baseline_started_at
                < self._BASELINE_WARMUP_SECONDS
            ):
                # Startup detections are often incomplete while TensorRT and
                # the tracker settle. Preserve the largest observed semantic
                # footprint so a pre-existing item is not called newly added.
                if snapshot.area_ratio >= state.baseline.area_ratio:
                    state.baseline = snapshot
                return result
            if state.last_actor_seen_at is not None:
                quiet_for = snapshot.timestamp - state.last_actor_seen_at
                if quiet_for < roi.rules.actor_leave_grace_seconds:
                    result.garbage_overlays.append(
                        GarbageOverlay(
                            roi_id=roi_id,
                            rectangle=_union_rect(snapshot.regions),
                            label="疑似遗留物，等待人员离开",
                            state="observing",
                            elapsed_seconds=max(quiet_for, 0.0),
                        )
                    )
                    return result
                if quiet_for > roi.rules.actor_association_seconds:
                    state.actors_during_interaction.clear()
                    state.last_actor_seen_at = None
            change_kind = _classify_change(
                state.baseline,
                snapshot,
                minimum_change_area=roi.rules.minimum_change_area,
            )
            if change_kind == "unchanged":
                self._reset_candidate(state)
                if not state.actors_during_interaction:
                    state.baseline = snapshot
                return result
            if state.candidate_kind != change_kind:
                state.candidate_kind = change_kind
                state.candidate_started_at = snapshot.timestamp
                state.candidate_snapshot = snapshot
            else:
                state.candidate_snapshot = snapshot
            elapsed = max(snapshot.timestamp - state.candidate_started_at, 0.0)
            label = _candidate_label(change_kind, elapsed, roi.rules.garbage_persistence_seconds)
            result.garbage_overlays.append(
                GarbageOverlay(
                    roi_id=roi_id,
                    rectangle=_union_rect(
                        snapshot.regions or snapshot.visual_regions
                    ),
                    label=label,
                    state=(
                        "confirmed"
                        if elapsed >= roi.rules.garbage_persistence_seconds
                        else "candidate"
                    ),
                    elapsed_seconds=elapsed,
                )
            )
            if elapsed < roi.rules.garbage_persistence_seconds:
                return result
            event = self._garbage_event(roi, state, snapshot, change_kind)
            self._emit(event, result)
            confirmed = GarbageOverlay(
                roi_id=roi_id,
                rectangle=_union_rect(
                    snapshot.regions or snapshot.visual_regions
                ),
                label=event.message,
                state="confirmed",
                elapsed_seconds=elapsed,
            )
            state.last_confirmed = confirmed
            state.last_confirmed_at = snapshot.timestamp
            state.last_confirmed_persistent = change_kind == "added"
            state.baseline = snapshot
            state.actors_during_interaction.clear()
            state.last_actor_seen_at = None
            self._reset_candidate(state)
        return result

    def _garbage_event(
        self,
        roi: EventRoiOptions,
        state: _GarbageRoiState,
        snapshot: GarbageSnapshot,
        change_kind: str,
    ) -> EventRecord:
        actor_track_id: int | None = None
        actor_type: str | None = None
        associated = _associated_actor(state, snapshot)
        if associated is not None:
            actor_track_id, actor_type = associated
        if change_kind == "added" and actor_track_id is not None:
            event_type = "suspected_littering"
            message = "疑似乱丢垃圾"
        elif change_kind == "added":
            event_type = "unattended_garbage"
            message = "发现新增垃圾"
        elif change_kind == "removed":
            event_type = "garbage_removed"
            message = "垃圾已清理"
        elif change_kind == "moved":
            event_type = "garbage_moved"
            message = "垃圾位置发生移动"
        else:
            event_type = "garbage_interaction_uncertain"
            message = "垃圾变化待确认"
        return EventRecord.create(
            stream_id=self.stream_id,
            event_type=event_type,
            roi_id=roi.roi_id,
            message=message,
            actor_type=actor_type,
            actor_track_id=actor_track_id,
            object_type=snapshot.object_type,
            confidence=snapshot.semantic_confidence,
            metadata={
                "before_area_ratio": (
                    state.baseline.area_ratio if state.baseline else None
                ),
                "after_area_ratio": snapshot.area_ratio,
                "visual_change_ratio": snapshot.visual_change_ratio,
                "frame_number": snapshot.frame_number,
            },
        )

    def _expire_tracks(
        self,
        timestamp: float,
        seen: set[tuple[str, int]],
    ) -> None:
        for key, state in list(self._dwell.items()):
            if key in seen:
                continue
            roi = self._roi(key[0])
            if timestamp - state.last_seen_at > roi.rules.actor_leave_grace_seconds:
                self._dwell.pop(key, None)

    def _actor_type(self, actor: TrackedObject) -> str | None:
        if actor.class_id in self.options.person_classes:
            return "person"
        if actor.class_id in self.options.vehicle_classes:
            return "vehicle"
        return None

    @staticmethod
    def _dwell_threshold(roi: EventRoiOptions, actor_type: str) -> float:
        return (
            roi.rules.person_dwell_seconds
            if actor_type == "person"
            else roi.rules.vehicle_dwell_seconds
        )

    def _roi(self, roi_id: str) -> EventRoiOptions:
        for roi in self.options.rois:
            if roi.roi_id == roi_id:
                return roi
        raise KeyError(roi_id)

    def _emit(self, event: EventRecord, result: EventEngineResult) -> None:
        result.events.append(event)
        if self._repository is not None:
            self._repository.append(event)
        if self._on_event is not None:
            self._on_event(event)

    @staticmethod
    def _reset_candidate(state: _GarbageRoiState) -> None:
        state.candidate_kind = None
        state.candidate_started_at = 0.0
        state.candidate_snapshot = None


def _actor_label(actor_type: str) -> str:
    return "人员" if actor_type == "person" else "车辆"


def _candidate_label(kind: str, elapsed: float, persistence: float) -> str:
    if kind == "added":
        return f"疑似新增垃圾 {int(elapsed)}/{int(persistence)}秒"
    if kind == "removed":
        return f"疑似垃圾清理 {int(elapsed)}/{int(persistence)}秒"
    if kind == "moved":
        return f"疑似垃圾移动 {int(elapsed)}/{int(persistence)}秒"
    return "垃圾变化待确认"


def _associated_actor(
    state: _GarbageRoiState,
    snapshot: GarbageSnapshot,
) -> tuple[int, str] | None:
    target = _union_rect(snapshot.regions or snapshot.visual_regions)
    if target is None:
        return None
    margin = 0.12
    left = max(target.left - margin, 0.0)
    top = max(target.top - margin, 0.0)
    right = min(target.left + target.width + margin, 1.0)
    bottom = min(target.top + target.height + margin, 1.0)
    for track_id in reversed(state.actors_during_interaction):
        interaction = state.actors_during_interaction[track_id]
        if any(
            left <= rectangle.bottom_center[0] <= right
            and top <= rectangle.bottom_center[1] <= bottom
            for rectangle in interaction.trail
        ):
            return track_id, interaction.actor_type
    return None


def _classify_change(
    before: GarbageSnapshot,
    after: GarbageSnapshot,
    *,
    minimum_change_area: float,
) -> str:
    delta = after.area_ratio - before.area_ratio
    # Semantic prompt scores can flicker even when the pixels did not change.
    # Require a small independent background-change signal before declaring
    # add/remove/move. It is intentionally capped below the semantic area
    # threshold so a small bottle can still be confirmed.
    corroboration_area = min(minimum_change_area, 0.001)
    visually_corroborated = (
        after.visual_change_ratio >= corroboration_area
    )
    if delta >= minimum_change_area and visually_corroborated:
        return "added"
    if delta <= -minimum_change_area and visually_corroborated:
        return "removed"
    before_center = _regions_center(before.regions)
    after_center = _regions_center(after.regions)
    if before_center is not None and after_center is not None:
        distance = math.dist(before_center, after_center)
        if (
            distance >= 0.05
            and max(before.area_ratio, after.area_ratio) > 0
            and visually_corroborated
        ):
            return "moved"
    if after.visual_change_ratio >= minimum_change_area:
        return "uncertain"
    return "unchanged"


def _regions_center(
    regions: tuple[NormalizedRect, ...],
) -> tuple[float, float] | None:
    if not regions:
        return None
    centers = [item.center for item in regions]
    return (
        sum(item[0] for item in centers) / len(centers),
        sum(item[1] for item in centers) / len(centers),
    )


def _union_rect(
    regions: tuple[NormalizedRect, ...],
) -> NormalizedRect | None:
    if not regions:
        return None
    left = min(item.left for item in regions)
    top = min(item.top for item in regions)
    right = max(item.left + item.width for item in regions)
    bottom = max(item.top + item.height for item in regions)
    return NormalizedRect(left, top, right - left, bottom - top)


def _point_in_polygon(
    point: tuple[float, float],
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (
            min(previous_y, current_y) <= y <= max(previous_y, current_y)
            and min(previous_x, current_x) <= x <= max(previous_x, current_x)
        ):
            cross = (
                (current_x - previous_x) * (y - previous_y)
                - (current_y - previous_y) * (x - previous_x)
            )
            if abs(cross) <= 1e-9:
                return True
        if (current_y > y) != (previous_y > y):
            x_at_y = previous_x + (
                (y - previous_y)
                * (current_x - previous_x)
                / (current_y - previous_y)
            )
            if x <= x_at_y:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside
