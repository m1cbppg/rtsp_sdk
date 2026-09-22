#!/usr/bin/env python3
"""V3.2 lifecycle state for fixed-view ground anomaly events.

V3.2 keeps visible anomaly confirmation cumulative across occlusion, while
requiring clear evidence to be consecutive, timely, and supported by enough
valid ground pixels.  Diagnostic histories are bounded for long-running use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import numpy as np

from run_ground_litter_event_memory_v31 import (
    CONTEXT_OCCLUSION_EXTERNAL_AREA_PX,
    box_center,
    measure_context_support,
    same_fixed_view_anchor,
)
from run_ground_litter_online_replay_v31 import (
    ANALYSIS_FPS,
    ANCHOR_SUPPORT_PAD_PX,
    CLEAN_MAX_SUPPORT_PIXELS,
    CLEAR_CONFIRM_SECONDS,
    CONFIRM_VISIBLE_SECONDS,
    EVENT_MATCH_DISTANCE_PX,
    EVENT_MATCH_SIZE_RATIO,
    EVENT_MERGE_SIZE_RATIO,
    PENDING_EXPIRE_SECONDS,
    max_actor_overlap,
)


MIN_CLEAN_VALID_FRACTION = 0.8
MAX_SAMPLE_GAP_FACTOR = 1.5
EVENT_MERGE_DISTANCE_PX = EVENT_MATCH_DISTANCE_PX
MAX_BOX_HISTORY = 30
MAX_MERGED_BOX_HISTORY = 60
MAX_VISIBLE_TIMESTAMP_HISTORY = 3600
MAX_STATE_HISTORY = 256
MAX_CLOSED_EVENTS = 10_000


def anchor_observation(
    support: np.ndarray,
    valid: np.ndarray,
    box: Iterable[float],
    *,
    pad_px: int = ANCHOR_SUPPORT_PAD_PX,
) -> dict[str, float | int]:
    """Measure residual support and valid-ground coverage around an anchor."""
    if support.shape != valid.shape:
        raise ValueError("support and valid masks must have identical shapes")
    height, width = support.shape
    anchor_left, anchor_top, anchor_right, anchor_bottom = (
        int(round(value)) for value in box
    )
    anchor_left = max(0, anchor_left)
    anchor_top = max(0, anchor_top)
    anchor_right = min(width, anchor_right)
    anchor_bottom = min(height, anchor_bottom)
    if anchor_right <= anchor_left or anchor_bottom <= anchor_top:
        return {"region_pixels": 0, "valid_pixels": 0,
                "valid_fraction": 0.0, "support_pixels": 0}
    anchor_valid = valid[anchor_top:anchor_bottom, anchor_left:anchor_right] > 0
    region_pixels = int(anchor_valid.size)
    valid_pixels = int(np.count_nonzero(anchor_valid))
    left = max(0, anchor_left - pad_px)
    top = max(0, anchor_top - pad_px)
    right = min(width, anchor_right + pad_px)
    bottom = min(height, anchor_bottom + pad_px)
    valid_crop = valid[top:bottom, left:right] > 0
    support_pixels = int(np.count_nonzero(
        (support[top:bottom, left:right] > 0) & valid_crop
    ))
    return {
        "region_pixels": region_pixels,
        "valid_pixels": valid_pixels,
        "valid_fraction": valid_pixels / region_pixels,
        "support_pixels": support_pixels,
    }


@dataclass
class OnlineEvent:
    event_id: int
    first_seen: float
    anchor_box: list[float]
    state: str = "ANOMALY_PENDING"
    last_visible: float | None = None
    visible_timestamps: list[float] = field(default_factory=list)
    visible_observation_count: int = 0
    visible_evidence_seconds: float = 0.0
    visible_history_truncated: bool = False
    boxes: list[list[float]] = field(default_factory=list)
    state_history: list[dict[str, Any]] = field(default_factory=list)
    state_history_truncated: bool = False
    clear_observed_seconds: float = 0.0
    last_clean_observed_at: float | None = None
    pending_unmatched_seconds: float = 0.0
    occluded_samples: int = 0
    uncertain_samples: int = 0
    confirmed_at: float | None = None
    closed_at: float | None = None
    closed_reason: str | None = None

    def _append_state(self, timestamp: float, state: str, reason: str) -> None:
        self.state_history.append({
            "timestamp": round(float(timestamp), 3),
            "state": state,
            "reason": reason,
        })
        if len(self.state_history) > MAX_STATE_HISTORY:
            self.state_history = self.state_history[-MAX_STATE_HISTORY:]
            self.state_history_truncated = True

    def transition(self, timestamp: float, state: str, reason: str) -> None:
        if self.state != state or not self.state_history:
            self._append_state(timestamp, state, reason)
        self.state = state

    def reset_clear_evidence(self) -> None:
        self.clear_observed_seconds = 0.0
        self.last_clean_observed_at = None

    def observe(self, timestamp: float, box: Iterable[float], sample_period: float) -> None:
        timestamp = float(timestamp)
        values = [float(value) for value in box]
        self.boxes.append(values)
        self.boxes = self.boxes[-MAX_BOX_HISTORY:]
        self.anchor_box = np.median(np.asarray(self.boxes), axis=0).tolist()
        if self.last_visible != timestamp:
            self.visible_timestamps.append(timestamp)
            if len(self.visible_timestamps) > MAX_VISIBLE_TIMESTAMP_HISTORY:
                self.visible_timestamps = self.visible_timestamps[-MAX_VISIBLE_TIMESTAMP_HISTORY:]
                self.visible_history_truncated = True
            self.visible_observation_count += 1
            self.visible_evidence_seconds += sample_period
        self.last_visible = timestamp
        self.pending_unmatched_seconds = 0.0
        self.reset_clear_evidence()
        if (
            self.confirmed_at is None
            and self.visible_evidence_seconds >= CONFIRM_VISIBLE_SECONDS
        ):
            self.confirmed_at = timestamp
        self.transition(timestamp, "VISIBLE_ANOMALY", "candidate_visible")

    def observe_clean(
        self,
        timestamp: float,
        *,
        sample_period: float,
        max_sample_gap: float,
    ) -> None:
        timestamp = float(timestamp)
        previous = self.last_clean_observed_at
        if (
            previous is None
            or timestamp <= previous
            or timestamp - previous > max_sample_gap
        ):
            self.clear_observed_seconds = sample_period
        else:
            self.clear_observed_seconds += sample_period
        self.last_clean_observed_at = timestamp

    def absorb_same_frame_component(self, box: Iterable[float]) -> None:
        self.boxes.append([float(value) for value in box])
        self.boxes = self.boxes[-MAX_BOX_HISTORY:]
        self.anchor_box = np.median(np.asarray(self.boxes), axis=0).tolist()

    def absorb_event(self, other: "OnlineEvent", timestamp: float,
                     sample_period: float) -> None:
        self.first_seen = min(self.first_seen, other.first_seen)
        self.boxes.extend(other.boxes)
        self.boxes = self.boxes[-MAX_MERGED_BOX_HISTORY:]
        self.anchor_box = np.median(np.asarray(self.boxes), axis=0).tolist()
        retained_union = sorted(set(
            self.visible_timestamps + other.visible_timestamps
        ))
        self.visible_timestamps = retained_union[-MAX_VISIBLE_TIMESTAMP_HISTORY:]
        was_truncated = self.visible_history_truncated or other.visible_history_truncated
        self.visible_history_truncated = (
            was_truncated or len(retained_union) > MAX_VISIBLE_TIMESTAMP_HISTORY
        )
        if was_truncated:
            # Exact overlap is unknowable once either diagnostic history was
            # truncated.  Keep the larger total to avoid premature confirmation.
            self.visible_observation_count = max(
                self.visible_observation_count, other.visible_observation_count
            )
            self.visible_evidence_seconds = max(
                self.visible_evidence_seconds, other.visible_evidence_seconds
            )
        else:
            self.visible_observation_count = len(retained_union)
            self.visible_evidence_seconds = len(retained_union) * sample_period
        visible = [value for value in (self.last_visible, other.last_visible)
                   if value is not None]
        self.last_visible = max(visible) if visible else None
        confirmed = [value for value in (self.confirmed_at, other.confirmed_at)
                     if value is not None]
        self.confirmed_at = min(confirmed) if confirmed else None
        self.occluded_samples += other.occluded_samples
        self.uncertain_samples += other.uncertain_samples
        self.pending_unmatched_seconds = 0.0
        self.reset_clear_evidence()
        self._append_state(timestamp, self.state, f"merged_event_{other.event_id}")

    def as_dict(self, sample_period: float) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "first_seen": round(self.first_seen, 3),
            "last_visible": None if self.last_visible is None else round(self.last_visible, 3),
            "anchor_box": [round(value, 2) for value in self.anchor_box],
            "state": self.state,
            "visible_hits": self.visible_observation_count,
            "visible_duration_seconds": round(self.visible_evidence_seconds, 3),
            "confirmed_at": None if self.confirmed_at is None else round(self.confirmed_at, 3),
            "confirmation_delay_seconds": (
                None if self.confirmed_at is None
                else round(self.confirmed_at - self.first_seen, 3)
            ),
            "closed_at": None if self.closed_at is None else round(self.closed_at, 3),
            "closed_reason": self.closed_reason,
            "clear_observed_seconds": round(self.clear_observed_seconds, 3),
            "last_clean_observed_at": (
                None if self.last_clean_observed_at is None
                else round(self.last_clean_observed_at, 3)
            ),
            "pending_unmatched_seconds": round(self.pending_unmatched_seconds, 3),
            "occluded_samples": self.occluded_samples,
            "uncertain_samples": self.uncertain_samples,
            "state_history": self.state_history,
            "state_history_truncated": self.state_history_truncated,
            "visible_timestamps": [round(value, 3) for value in self.visible_timestamps],
            "visible_history_truncated": self.visible_history_truncated,
        }


class OnlineEventMemory:
    def __init__(
        self,
        *,
        sample_fps: float = ANALYSIS_FPS,
        actor_overlap_threshold: float = 0.2,
        min_clean_valid_fraction: float = MIN_CLEAN_VALID_FRACTION,
        max_sample_gap_factor: float = MAX_SAMPLE_GAP_FACTOR,
        max_closed_events: int = MAX_CLOSED_EVENTS,
    ) -> None:
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if not 0.0 <= min_clean_valid_fraction <= 1.0:
            raise ValueError("min_clean_valid_fraction must be between 0 and 1")
        if max_sample_gap_factor < 1.0:
            raise ValueError("max_sample_gap_factor must be at least 1")
        if max_closed_events < 0:
            raise ValueError("max_closed_events must not be negative")
        self.sample_period = 1.0 / sample_fps
        self.max_sample_gap = self.sample_period * max_sample_gap_factor
        self.actor_overlap_threshold = actor_overlap_threshold
        self.min_clean_valid_fraction = min_clean_valid_fraction
        self.max_closed_events = max_closed_events
        self.events: list[OnlineEvent] = []
        self.events_pruned = 0
        self.confirmed_events_pruned = 0
        self._next_id = 1
        self._last_update_timestamp: float | None = None

    @property
    def active(self) -> list[OnlineEvent]:
        return [event for event in self.events if event.closed_at is None]

    def update(
        self,
        *,
        timestamp: float,
        candidates: list[dict[str, Any]],
        support: np.ndarray,
        valid: np.ndarray,
        actors: list[list[float]],
        environment_state: str,
    ) -> dict[str, Any]:
        timestamp = float(timestamp)
        if (
            self._last_update_timestamp is not None
            and timestamp <= self._last_update_timestamp
        ):
            raise ValueError("timestamps must be strictly increasing")
        self._last_update_timestamp = timestamp

        if environment_state == "ENVIRONMENT_CHANGE":
            for event in self.active:
                event.reset_clear_evidence()
                event.transition(timestamp, "ENVIRONMENT_CHANGE", "frame_unavailable")
            return {"visible": 0, "context_blocked": 0,
                    "actor_blocked": 0, "ground_unavailable": len(self.active)}

        visible_candidates = []
        context_blocked = []
        actor_blocked = []
        for row in candidates:
            context = measure_context_support(support, valid, row["box"])
            if context["touching_external_support_px"] >= CONTEXT_OCCLUSION_EXTERNAL_AREA_PX:
                context_blocked.append(row)
                continue
            if max_actor_overlap(row["box"], actors) > self.actor_overlap_threshold:
                actor_blocked.append(row)
                continue
            visible_candidates.append(row)

        pairs = []
        active = self.active
        for event_index, event in enumerate(active):
            for candidate_index, candidate in enumerate(visible_candidates):
                if not same_fixed_view_anchor(
                    event.anchor_box,
                    candidate["box"],
                    center_distance_px=EVENT_MATCH_DISTANCE_PX,
                    size_ratio=EVENT_MATCH_SIZE_RATIO,
                ):
                    continue
                pairs.append((
                    math.dist(box_center(event.anchor_box), box_center(candidate["box"])),
                    event_index,
                    candidate_index,
                ))
        used_events: set[int] = set()
        used_candidates: set[int] = set()
        for _distance, event_index, candidate_index in sorted(pairs):
            if event_index in used_events or candidate_index in used_candidates:
                continue
            active[event_index].observe(
                timestamp, visible_candidates[candidate_index]["box"], self.sample_period
            )
            used_events.add(event_index)
            used_candidates.add(candidate_index)

        for candidate_index, candidate in enumerate(visible_candidates):
            if candidate_index in used_candidates:
                continue
            related = [
                (math.dist(box_center(event.anchor_box), box_center(candidate["box"])), event)
                for event in active
                if same_fixed_view_anchor(
                    event.anchor_box,
                    candidate["box"],
                    center_distance_px=EVENT_MATCH_DISTANCE_PX,
                    size_ratio=EVENT_MATCH_SIZE_RATIO,
                )
            ]
            if not related:
                continue
            _distance, event = min(related, key=lambda pair: pair[0])
            event.absorb_same_frame_component(candidate["box"])
            used_candidates.add(candidate_index)

        for candidate_index, candidate in enumerate(visible_candidates):
            if candidate_index in used_candidates:
                continue
            event = OnlineEvent(
                event_id=self._next_id,
                first_seen=timestamp,
                anchor_box=[float(value) for value in candidate["box"]],
            )
            event.observe(timestamp, candidate["box"], self.sample_period)
            self.events.append(event)
            self._next_id += 1

        ground_unavailable = 0
        for event_index, event in enumerate(active):
            if event_index in used_events:
                continue
            context = measure_context_support(support, valid, event.anchor_box)
            context_occluded = (
                context["touching_external_support_px"]
                >= CONTEXT_OCCLUSION_EXTERNAL_AREA_PX
            )
            actor_occluded = (
                max_actor_overlap(event.anchor_box, actors)
                > self.actor_overlap_threshold
            )
            if context_occluded or actor_occluded:
                event.occluded_samples += 1
                event.reset_clear_evidence()
                event.transition(
                    timestamp,
                    "OCCLUDED",
                    "context_occluded" if context_occluded else "actor_occluded",
                )
                continue
            observation = anchor_observation(support, valid, event.anchor_box)
            if observation["valid_fraction"] < self.min_clean_valid_fraction:
                ground_unavailable += 1
                event.uncertain_samples += 1
                event.reset_clear_evidence()
                event.transition(timestamp, "GROUND_UNAVAILABLE", "insufficient_valid_ground")
                continue
            if (
                event.confirmed_at is None
                and event.last_visible is not None
            ):
                event.pending_unmatched_seconds += self.sample_period
            if (
                event.confirmed_at is None
                and event.pending_unmatched_seconds >= PENDING_EXPIRE_SECONDS
            ):
                event.reset_clear_evidence()
                event.closed_at = timestamp
                event.closed_reason = "pending_timeout"
                event.transition(timestamp, "EXPIRED_PENDING", "pending_timeout")
                continue
            if observation["support_pixels"] <= CLEAN_MAX_SUPPORT_PIXELS:
                event.observe_clean(
                    timestamp,
                    sample_period=self.sample_period,
                    max_sample_gap=self.max_sample_gap,
                )
                event.transition(timestamp, "CLEAN_PENDING", "clean_reference_match")
                if event.clear_observed_seconds >= CLEAR_CONFIRM_SECONDS:
                    event.closed_at = timestamp
                    event.closed_reason = "clean_confirmed"
                    event.transition(timestamp, "CLEARED", "clean_confirmed")
                continue
            event.uncertain_samples += 1
            event.reset_clear_evidence()
            event.transition(timestamp, "ANOMALY_PENDING", "residual_without_candidate")

        self._merge_converged_events(timestamp)
        self._prune_closed_events()
        return {
            "visible": len(visible_candidates),
            "context_blocked": len(context_blocked),
            "actor_blocked": len(actor_blocked),
            "ground_unavailable": ground_unavailable,
        }

    def _merge_converged_events(self, timestamp: float) -> None:
        active = sorted(self.active, key=lambda event: event.event_id)
        consumed: set[int] = set()
        for index, primary in enumerate(active):
            if primary.event_id in consumed:
                continue
            for secondary in active[index + 1:]:
                if secondary.event_id in consumed:
                    continue
                if not same_fixed_view_anchor(
                    primary.anchor_box,
                    secondary.anchor_box,
                    center_distance_px=EVENT_MERGE_DISTANCE_PX,
                    size_ratio=EVENT_MERGE_SIZE_RATIO,
                ):
                    continue
                primary.absorb_event(secondary, timestamp, self.sample_period)
                if (
                    primary.confirmed_at is None
                    and primary.visible_evidence_seconds >= CONFIRM_VISIBLE_SECONDS
                ):
                    primary.confirmed_at = timestamp
                secondary.reset_clear_evidence()
                secondary.closed_at = timestamp
                secondary.closed_reason = f"merged_into:{primary.event_id}"
                secondary.transition(timestamp, "MERGED", f"merged_into:{primary.event_id}")
                consumed.add(secondary.event_id)

    def _prune_closed_events(self) -> None:
        closed = [event for event in self.events if event.closed_at is not None]
        excess = len(closed) - self.max_closed_events
        if excess <= 0:
            return
        removed = sorted(
            closed, key=lambda row: (row.closed_at or 0.0, row.event_id)
        )[:excess]
        remove_ids = {event.event_id for event in removed}
        self.events = [event for event in self.events if event.event_id not in remove_ids]
        self.events_pruned += len(remove_ids)
        self.confirmed_events_pruned += sum(
            event.confirmed_at is not None for event in removed
        )
