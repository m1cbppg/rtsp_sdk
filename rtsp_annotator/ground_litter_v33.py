"""Ground Litter V3.3 dual-channel recall: evidence model and event memory.

Two independent recall channels feed one event layer:

* the **semantic** channel runs ``turhancan_yolov8m_seg_trash.pt`` over the full
  ground ROI (and, as an auxiliary signal, over prior-guided crops);
* the **prior** channel runs Clean Reference change detection.

Either channel may confirm an event on its own. Neither may veto the other:
``prior_only`` does not need semantic confirmation, and ``semantic_only`` does not
need Clean Reference confirmation. Fusion is a union of two already-filtered
*event* streams, never a union of raw per-frame boxes.

This module is deliberately pure logic: it loads no model and manages no
process, so the whole state machine is unit-testable with deterministic fake
observations. The side process drives it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .event_engine import NormalizedRect
from .ground_litter_detection import GroundLitterDetection
from .ground_litter_geometry import box_overlap_fraction
from .ground_litter_v32 import _anchor_observation

EVIDENCE_SEMANTIC_ONLY = "semantic_only"
EVIDENCE_PRIOR_ONLY = "prior_only"
EVIDENCE_SEMANTIC_AND_PRIOR = "semantic_and_prior"

SOURCE_SEMANTIC = "hybrid_v33_semantic"
SOURCE_PRIOR = "hybrid_v33_prior"
SOURCE_FUSED = "hybrid_v33_fused"

# Same-position matching mirrors the V3.2 event constants so one camera cannot
# end up with two incompatible notions of "the same target".
MATCH_DISTANCE_PX = 30.0
MATCH_SIZE_RATIO = 5.0
CLEAN_MAX_SUPPORT_PIXELS = 2

# Displayable states. Clean/absence-pending states stay visible until the event
# actually closes, so a box never disappears a few seconds before its own close.
DISPLAY_STATES = frozenset({
    "SEMANTIC_VISIBLE",
    "PRIOR_VISIBLE",
    "FUSED_VISIBLE",
    "CLEAN_PENDING",
    "ABSENT_PENDING",
})

ALL_STATES = (
    "PENDING",
    "SEMANTIC_VISIBLE",
    "PRIOR_VISIBLE",
    "FUSED_VISIBLE",
    "OCCLUDED",
    "CLEAN_PENDING",
    "ABSENT_PENDING",
    "CLEARED",
    "EXPIRED_PENDING",
    "MERGED",
)

MAX_STATE_HISTORY = 128
MAX_BOX_HISTORY = 30
MERGE_CONFIRMATION_TIMESTAMPS = 2

Box = tuple[float, float, float, float]


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------


def _center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0)


def _short_side(box: Sequence[float]) -> float:
    return max(0.0, min(float(box[2]) - float(box[0]),
                        float(box[3]) - float(box[1])))


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1]))


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = _area(a) + _area(b) - intersection
    return float(intersection / union) if union > 0 else 0.0


def expand_box(box: Sequence[float], ratio: float) -> Box:
    half_width = (float(box[2]) - float(box[0])) * ratio / 2.0
    half_height = (float(box[3]) - float(box[1])) * ratio / 2.0
    center_x, center_y = _center(box)
    return (center_x - half_width, center_y - half_height,
            center_x + half_width, center_y + half_height)


def same_target(
    left: Sequence[float],
    right: Sequence[float],
    *,
    distance_px: float = MATCH_DISTANCE_PX,
    size_ratio: float = MATCH_SIZE_RATIO,
    pixel_scale: float = 1.0,
) -> bool:
    """Bounded centre distance plus bounded size ratio: one physical target."""
    if len(left) < 4 or len(right) < 4:
        return False
    if math.dist(_center(left), _center(right)) > distance_px * float(pixel_scale):
        return False
    larger = max(_short_side(left), _short_side(right), 1e-6)
    smaller = max(min(_short_side(left), _short_side(right)), 1e-6)
    return larger / smaller <= float(size_ratio)


def observations_match(
    semantic_box: Sequence[float],
    prior_box: Sequence[float],
    *,
    profile_scale: float = 1.0,
    iou_threshold: float = 0.10,
    area_ratio_limit: float = 8.0,
) -> bool:
    """Single-frame spatial association between one semantic and one prior box.

    Any one of three conditions suffices: IoU, the semantic centre inside the
    prior box expanded by 25 %, or a bounded centre distance with a bounded area
    ratio. Matching none of them means the boxes are unrelated and must not be
    merged merely because they appeared in the same tick.
    """
    if len(semantic_box) < 4 or len(prior_box) < 4:
        return False
    if box_iou(semantic_box, prior_box) >= iou_threshold:
        return True
    center_x, center_y = _center(semantic_box)
    left, top, right, bottom = expand_box(prior_box, 1.25)
    if left <= center_x <= right and top <= center_y <= bottom:
        return True
    distance = math.dist(_center(semantic_box), _center(prior_box))
    limit = max(24.0 * float(profile_scale),
                0.5 * max(_short_side(semantic_box), _short_side(prior_box)))
    if distance > limit:
        return False
    larger = max(_area(semantic_box), _area(prior_box), 1e-6)
    smaller = max(min(_area(semantic_box), _area(prior_box)), 1e-6)
    return larger / smaller <= area_ratio_limit


# ----------------------------------------------------------------------------
# Evidence structures
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SemanticObservation:
    """One litter-model detection inside the ground ROI, already filtered."""

    box_xyxy: Box
    confidence: float
    class_name: str
    region_id: str
    source: str  # "full_roi" | "prior_crop"
    observed_at: float


@dataclass(frozen=True, slots=True)
class PriorObservation:
    """A localised, persistent change against the Clean Reference."""

    box_xyxy: Box
    anomaly_score: float
    region_id: str
    support_pixels: int
    observed_at: float


@dataclass(frozen=True, slots=True)
class HybridObservation:
    """A per-tick observation carrying whichever channels fired."""

    anchor_box: Box
    display_box: Box
    region_id: str
    evidence_kind: str
    semantic: SemanticObservation | None
    prior: PriorObservation | None
    observed_at: float


def fuse_pair(
    semantic: SemanticObservation, prior: PriorObservation,
) -> HybridObservation:
    """Fuse one associated semantic/prior pair into a single observation."""
    anchor: Box = (  # type: ignore[assignment]
        (float(semantic.box_xyxy[0]) + float(prior.box_xyxy[0])) / 2.0,
        (float(semantic.box_xyxy[1]) + float(prior.box_xyxy[1])) / 2.0,
        (float(semantic.box_xyxy[2]) + float(prior.box_xyxy[2])) / 2.0,
        (float(semantic.box_xyxy[3]) + float(prior.box_xyxy[3])) / 2.0,
    )
    return HybridObservation(
        anchor_box=anchor,
        # The semantic box is class-attributed and tighter, so it wins display;
        # the anchor is the two-box compromise used for cross-tick matching.
        display_box=semantic.box_xyxy,
        region_id=semantic.region_id or prior.region_id,
        evidence_kind=EVIDENCE_SEMANTIC_AND_PRIOR,
        semantic=semantic,
        prior=prior,
        observed_at=float(semantic.observed_at),
    )


def associate_observations(
    semantic: Iterable[SemanticObservation],
    prior: Iterable[PriorObservation],
    *,
    tolerance_seconds: float,
    profile_scale: float = 1.0,
) -> list[HybridObservation]:
    """Fuse one tick's channel observations into hybrid observations.

    Greedy one-to-one matching ordered by IoU, then centre distance, then
    semantic confidence. Unmatched observations of either channel are returned as
    single-channel observations rather than being dropped.
    """
    semantic_list = list(semantic)
    prior_list = list(prior)
    scored: list[tuple[float, float, float, int, int]] = []
    for semantic_index, observation in enumerate(semantic_list):
        for prior_index, candidate in enumerate(prior_list):
            if abs(observation.observed_at - candidate.observed_at) > tolerance_seconds:
                continue
            if not observations_match(
                observation.box_xyxy, candidate.box_xyxy,
                profile_scale=profile_scale,
            ):
                continue
            scored.append((
                -box_iou(observation.box_xyxy, candidate.box_xyxy),
                math.dist(_center(observation.box_xyxy), _center(candidate.box_xyxy)),
                -float(observation.confidence),
                semantic_index,
                prior_index,
            ))
    scored.sort()

    used_semantic: set[int] = set()
    used_prior: set[int] = set()
    observations: list[HybridObservation] = []
    for _neg_iou, _distance, _neg_confidence, semantic_index, prior_index in scored:
        if semantic_index in used_semantic or prior_index in used_prior:
            continue
        used_semantic.add(semantic_index)
        used_prior.add(prior_index)
        observations.append(fuse_pair(
            semantic_list[semantic_index], prior_list[prior_index]
        ))
    for index, observation in enumerate(semantic_list):
        if index in used_semantic:
            continue
        observations.append(HybridObservation(
            anchor_box=observation.box_xyxy,
            display_box=observation.box_xyxy,
            region_id=observation.region_id,
            evidence_kind=EVIDENCE_SEMANTIC_ONLY,
            semantic=observation,
            prior=None,
            observed_at=float(observation.observed_at),
        ))
    for index, candidate in enumerate(prior_list):
        if index in used_prior:
            continue
        observations.append(HybridObservation(
            anchor_box=candidate.box_xyxy,
            display_box=candidate.box_xyxy,
            region_id=candidate.region_id,
            evidence_kind=EVIDENCE_PRIOR_ONLY,
            semantic=None,
            prior=candidate,
            observed_at=float(candidate.observed_at),
        ))
    return observations


def crop_semantic_observations(
    crop_rows: Sequence[Sequence[dict[str, Any]]],
    priors: Sequence[PriorObservation],
    *,
    timestamp: float,
    class_names: dict[int, str] | None = None,
    profile_scale: float = 1.0,
) -> tuple[list[SemanticObservation], int, int]:
    """Turn batched prior-crop results into semantic observations.

    Only detections that spatially associate with the prior that spawned the crop
    are kept. An extra detection inside a crop must not manufacture a
    ``semantic_only`` event: independent semantic recall comes from the full ROI
    scan, because a crop is a biased sample of the scene.

    Returns ``(matched, raw_count, unmatched_count)``.
    """
    names = class_names or {}
    matched: list[SemanticObservation] = []
    raw = 0
    unmatched = 0
    for index, rows in enumerate(crop_rows):
        if index >= len(priors):
            break
        prior = priors[index]
        for row in rows:
            raw += 1
            box: Box = tuple(  # type: ignore[assignment]
                float(value) for value in row["box"]
            )
            if not observations_match(box, prior.box_xyxy, profile_scale=profile_scale):
                unmatched += 1
                continue
            matched.append(SemanticObservation(
                box_xyxy=box,
                confidence=float(row.get("confidence", 0.0)),
                class_name=names.get(int(row.get("class_id", -1)), ""),
                region_id=prior.region_id,
                source="prior_crop",
                observed_at=float(timestamp),
            ))
    return matched, raw, unmatched


# ----------------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------------


@dataclass(slots=True)
class V33Event:
    """One tracked target. Never downgrades the evidence kind it reached."""

    event_id: int
    region_id: str
    anchor_box: list[float]
    display_box: list[float]
    first_seen_at: float
    last_seen_at: float
    last_semantic_at: float | None = None
    last_prior_at: float | None = None
    semantic_hits: int = 0
    prior_hits: int = 0
    semantic_class: str = ""
    maximum_semantic_confidence: float = 0.0
    evidence_kind: str = EVIDENCE_PRIOR_ONLY
    ever_semantic: bool = False
    ever_prior: bool = False
    current_support: set[str] = field(default_factory=set)
    confirmed_at: float | None = None
    closed_at: float | None = None
    closed_reason: str | None = None
    state: str = "PENDING"
    state_history: list[dict[str, Any]] = field(default_factory=list)
    box_history: deque = field(default_factory=lambda: deque(maxlen=MAX_BOX_HISTORY))
    semantic_window: deque = field(default_factory=deque)
    prior_window: deque = field(default_factory=deque)
    fused_window: deque = field(default_factory=deque)
    clean_seconds: float = 0.0
    last_clean_at: float | None = None
    absent_seconds: float = 0.0
    last_absent_at: float | None = None
    semantic_miss_count: int = 0
    suspended_since: float | None = None
    merges: int = 0

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    def transition(self, timestamp: float, state: str, reason: str) -> None:
        if self.state != state or not self.state_history:
            self.state_history.append({
                "timestamp": round(float(timestamp), 3),
                "state": state,
                "reason": reason,
            })
            self.state_history = self.state_history[-MAX_STATE_HISTORY:]
        self.state = state

    def reset_clean(self) -> None:
        self.clean_seconds = 0.0
        self.last_clean_at = None

    def reset_absent(self) -> None:
        self.absent_seconds = 0.0
        self.last_absent_at = None

    def reset_cross_reference_evidence(self) -> None:
        """普通切换参考时清空跨源证据窗口（方案二 §6.2）。

        保留事件身份、确认时间与首见时间；只丢弃"哪一帧命中过"的累计，
        防止跨两张参考拼凑确认/清走。
        """
        self.semantic_window.clear()
        self.prior_window.clear()
        self.fused_window.clear()
        self.current_support = set()
        self.semantic_miss_count = 0
        self.reset_clean()
        self.reset_absent()
        self.suspended_since = None

    def invalidate_previous_support(self) -> None:
        """参考代际变化：身份保留，但旧参考的 support 不再计入。"""
        self.ever_semantic = False
        self.ever_prior = False
        self.last_semantic_at = None
        self.last_prior_at = None
        self.semantic_hits = 0
        self.prior_hits = 0
        self.current_support = set()
        self.reset_cross_reference_evidence()


@dataclass(frozen=True, slots=True)
class V33TickResult:
    detections: tuple[GroundLitterDetection, ...]
    state: str
    message: str
    active_events: int
    confirmed_events: int
    cleared_events: int
    semantic_only_active: int
    semantic_only_confirmed: int
    prior_only_active: int
    prior_only_confirmed: int
    fused_active: int
    fused_confirmed: int
    cross_source_merges: int
    environment_state: str
    prior_available: bool
    semantic_ran: bool
    prior_raw: int
    prior_retained: int
    semantic_raw: int
    semantic_retained: int
    semantic_crop_raw: int = 0
    semantic_crop_unmatched: int = 0


class V33EventMemory:
    """Independent per-channel confirmation feeding one event lifecycle."""

    def __init__(self, options: Any, *, pixel_scale: float = 1.0) -> None:
        self.options = options
        self.pixel_scale = float(pixel_scale)
        self.sample_period = 1.0 / float(options.analysis_fps)
        scan_interval = float(options.semantic_scan_interval_seconds)
        # A gap longer than this restarts continuous evidence instead of
        # silently bridging a stalled interval.
        self.max_sample_gap = max(self.sample_period * 1.5, scan_interval * 1.5)
        # Freshness windows decide `current_support`. Without them a semantic
        # event would blink off on every tick where the (roughly 4 s) semantic
        # scan did not run.
        self.semantic_fresh_window = max(
            2.0 * scan_interval, float(options.semantic_confirm_span_seconds)
        )
        self.prior_fresh_window = max(
            2.0 * self.sample_period, float(options.prior_confirm_span_seconds)
        )
        self.events: list[V33Event] = []
        self._next_id = 1
        self._last_timestamp: float | None = None
        self._started_at: float | None = None
        self._last_result: V33TickResult | None = None
        self._cross_source_merges = 0
        self._merge_evidence: dict[tuple[int, int], deque[float]] = {}
        self._projection_source_cursor = 0
        self._reference_width = 2560
        self._reference_height = 1440
        # 参考代际：Bank 模式下由 Selector 提交时递增（方案二 §5.3）。
        self.reference_generation = 0
        self.reference_profile_id: str | None = None

    # -- setup -----------------------------------------------------------

    def set_reference_size(self, width: int, height: int) -> None:
        """Profile-space size used to normalise display boxes."""
        self._reference_width = max(1, int(width))
        self._reference_height = max(1, int(height))

    @property
    def active(self) -> list[V33Event]:
        return [event for event in self.events if event.is_open]

    # -- 参考代际（方案二 §5.3/§6.2）--------------------------------------

    def event_availability(
        self, event: V33Event, timestamp: float, *, prior_available: bool,
    ) -> dict[str, Any]:
        """区分「观察到了没有前景」与「无法判断」（方案二 §5.1）。

        ``support`` 为空且 ``available`` 为真 = 已判断且干净；
        ``available`` 为假 = 本 tick 根本没有可判断证据，不能当清走依据。
        """
        support, available, reason = self.per_event_support(
            event, timestamp, prior_available=prior_available
        )
        return {
            "event_id": event.event_id,
            "support": sorted(support),
            "available": available,
            "reason": reason,
            "observed_clean": available and not support,
            "clean_seconds": round(float(event.clean_seconds), 3),
            "semantic_miss_count": int(event.semantic_miss_count),
            "evidence_kind": event.evidence_kind,
            "state": event.state,
        }

    def per_event_support(
        self, event: V33Event, timestamp: float, *, prior_available: bool,
    ) -> tuple[set[str], bool, str]:
        """返回 (新鲜支持来源, 本帧是否可判断, 原因)。"""
        support = self._fresh_support(event, timestamp)
        if support:
            return support, True, "SUPPORT"
        if event.ever_prior and not prior_available:
            return support, False, "PROFILE_UNAVAILABLE"
        if event.state == "OCCLUDED":
            return support, False, "OCCLUDED"
        if event.state == "GROUND_UNAVAILABLE":
            return support, False, "GROUND_UNAVAILABLE"
        if not event.ever_prior and not event.ever_semantic:
            return support, False, "NO_EVIDENCE"
        return support, True, "OBSERVED_CLEAN"

    def notify_reference_switch(
        self, *, previous_profile_id: str, profile_id: str,
        generation: int, same_profile_id: bool = False,
    ) -> dict[str, Any]:
        """提交边界通知：保留身份，清空跨参考的待确认/清走证据。

        ``same_profile_id`` 表示 UNKNOWN 后恢复到同一个 profile_id——
        方案二 §5.3 要求它与换 ID 一样重置证据窗口，必须递增代际。
        """
        affected = 0
        for event in self.active:
            event.reset_cross_reference_evidence()
            event.invalidate_previous_support()
            affected += 1
        self._merge_evidence.clear()
        self.reference_generation = int(generation)
        self.reference_profile_id = str(profile_id)
        return {
            "previous_profile_id": previous_profile_id,
            "profile_id": profile_id,
            "generation": int(generation),
            "same_profile_id": bool(same_profile_id),
            "events_reset": affected,
        }

    def _assign_windows(self, event: V33Event) -> None:
        options = self.options
        event.semantic_window = deque(
            event.semantic_window, maxlen=int(options.semantic_hit_window)
        )
        event.prior_window = deque(
            event.prior_window, maxlen=int(options.prior_hit_window)
        )
        event.fused_window = deque(
            event.fused_window, maxlen=int(options.fused_hit_window)
        )

    @staticmethod
    def _window_confirmed(window: deque, hits: int, span: float) -> bool:
        timestamps = [float(stamp) for stamp, hit in window if hit]
        if len(timestamps) < int(hits):
            return False
        if span <= 0:
            return True
        return (max(timestamps) - min(timestamps)) >= float(span)

    def _occluded(
        self, box: Sequence[float], actors: Sequence[Sequence[float]],
    ) -> bool:
        if not actors:
            return False
        return max(
            (box_overlap_fraction(box, actor)
             for actor in actors if len(actor) >= 4),
            default=0.0,
        ) > float(self.options.actor_overlap_threshold)

    def _ground_state(
        self, box: Sequence[float],
        support: np.ndarray | None, valid: np.ndarray | None,
    ) -> tuple[float, int] | None:
        if support is None or valid is None:
            return None
        return _anchor_observation(
            support, valid, box, pixel_scale=self.pixel_scale
        )

    # -- main tick -------------------------------------------------------

    def update(
        self,
        *,
        timestamp: float,
        semantic: Sequence[SemanticObservation] | None,
        prior: Sequence[PriorObservation],
        prior_available: bool,
        environment_state: str,
        support: np.ndarray | None = None,
        valid: np.ndarray | None = None,
        actors: Sequence[Sequence[float]] = (),
        semantic_raw: int = 0,
        semantic_retained: int = 0,
        prior_raw: int = 0,
        prior_retained: int = 0,
        semantic_crop_raw: int = 0,
        semantic_crop_unmatched: int = 0,
        semantic_scan: bool = True,
    ) -> V33TickResult:
        timestamp = float(timestamp)
        if self._started_at is None:
            self._started_at = timestamp
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            # A repeated or rewound result must never double-count a hit.
            if self._last_result is not None:
                return self._last_result
            timestamp = float(self._last_timestamp) + 1e-6
        self._last_timestamp = timestamp

        semantic_ran = semantic is not None
        semantic_list = list(semantic or ())
        prior_list = list(prior)

        warmup_remaining = max(
            0.0,
            float(self.options.startup_suppress_seconds)
            - max(0.0, timestamp - self._started_at),
        )
        if warmup_remaining > 0:
            # Warm up, then start both channels from an empty memory so no
            # pre-warmup result can ever flash on screen.
            self.events = []
            result = self._result(
                detections=(), timestamp=timestamp, state="warming_up",
                message=(
                    f"hybrid_v33 profile={self.options.profile_id} "
                    f"startup_suppressed remaining={warmup_remaining:.1f}s "
                    f"env={environment_state}"
                ),
                environment_state=environment_state,
                prior_available=prior_available, semantic_ran=semantic_ran,
                semantic_raw=semantic_raw, semantic_retained=semantic_retained,
                prior_raw=prior_raw, prior_retained=prior_retained,
                semantic_crop_raw=semantic_crop_raw,
                semantic_crop_unmatched=semantic_crop_unmatched,
            )
            self._last_result = result
            return result

        observations = associate_observations(
            semantic_list, prior_list,
            tolerance_seconds=max(
                float(self.options.semantic_scan_interval_seconds),
                2.0 * self.sample_period,
            ),
            profile_scale=self.pixel_scale,
        )
        self._match_and_create(observations, timestamp)
        self._tick_windows(
            timestamp=timestamp, semantic_scan=semantic_ran and semantic_scan,
            prior_available=prior_available, actors=actors,
        )
        self._update_evidence(
            timestamp=timestamp, prior_available=prior_available,
            semantic_scan=semantic_ran and semantic_scan, actors=actors,
            support=support, valid=valid,
        )
        # Confirmation must run before the closing rules: those rules branch on
        # `confirmed_at`, which this step is what sets.
        self._evaluate_all(timestamp)
        self._close_events(timestamp=timestamp, prior_available=prior_available)
        self._prune_closed()

        result = self._result(
            detections=self._project(timestamp, prior_available=prior_available),
            timestamp=timestamp, state="running", message="",
            environment_state=environment_state,
            prior_available=prior_available, semantic_ran=semantic_ran,
            semantic_raw=semantic_raw, semantic_retained=semantic_retained,
            prior_raw=prior_raw, prior_retained=prior_retained,
            semantic_crop_raw=semantic_crop_raw,
            semantic_crop_unmatched=semantic_crop_unmatched,
        )
        self._last_result = result
        return result

    def _match_and_create(
        self, observations: list[HybridObservation], timestamp: float,
    ) -> None:
        active = self.active
        # `current_support` means "supported on THIS tick", so it is recomputed
        # for every open event before matching rather than carried over.
        for event in active:
            event.current_support = set()
        pairs: list[tuple[float, int, int]] = []
        for event_index, event in enumerate(active):
            for observation_index, observation in enumerate(observations):
                if same_target(
                    event.anchor_box, observation.anchor_box,
                    pixel_scale=self.pixel_scale,
                ):
                    pairs.append((
                        math.dist(_center(event.anchor_box),
                                  _center(observation.anchor_box)),
                        event_index, observation_index,
                    ))
        pairs.sort()
        used_events: set[int] = set()
        used_observations: set[int] = set()
        for _distance, event_index, observation_index in pairs:
            if event_index in used_events or observation_index in used_observations:
                continue
            used_events.add(event_index)
            used_observations.add(observation_index)
            observation = observations[observation_index]
            target = active[event_index]
            # A wide or unstable detector box can bridge two neighbouring pieces
            # of litter for one tick.  That single bridge is not proof that the
            # events are one physical target: the architecture requires the
            # association to persist at two different analysis timestamps.
            overlapping = [
                index for index, event in enumerate(active)
                if index != event_index and index not in used_events
                and event.is_open
                and same_target(
                    event.anchor_box, observation.anchor_box,
                    pixel_scale=self.pixel_scale,
                )
            ]
            merge_ready: list[int] = []
            for index in overlapping:
                pair = tuple(sorted((target.event_id, active[index].event_id)))
                history = self._merge_evidence.setdefault(pair, deque(maxlen=4))
                if not history or history[-1] != float(timestamp):
                    history.append(float(timestamp))
                maximum_gap = max(
                    2.0 * self.sample_period,
                    2.0 * float(self.options.semantic_scan_interval_seconds),
                )
                while history and timestamp - history[0] > maximum_gap:
                    history.popleft()
                if len(history) >= MERGE_CONFIRMATION_TIMESTAMPS:
                    merge_ready.append(index)
            if merge_ready:
                candidates = [target] + [active[index] for index in merge_ready]
                primary = min(candidates, key=lambda item: item.event_id)
                for secondary in candidates:
                    if secondary is not primary:
                        self._merge(primary, secondary, timestamp)
                for index in merge_ready:
                    used_events.add(index)
                target = primary
            self._apply_observation(target, observation, timestamp)

        for observation_index, observation in enumerate(observations):
            if observation_index in used_observations:
                continue
            self._create_event(observation, timestamp)

    def _apply_observation(
        self, event: V33Event, observation: HybridObservation, timestamp: float,
    ) -> None:
        event.last_seen_at = timestamp
        if observation.semantic is not None:
            event.last_semantic_at = timestamp
            event.semantic_hits += 1
            event.ever_semantic = True
            event.semantic_class = (
                observation.semantic.class_name or event.semantic_class
            )
            event.maximum_semantic_confidence = max(
                event.maximum_semantic_confidence,
                float(observation.semantic.confidence),
            )
            event.current_support.add("semantic")
        if observation.prior is not None:
            event.last_prior_at = timestamp
            event.prior_hits += 1
            event.ever_prior = True
            event.current_support.add("prior")
        if not event.region_id:
            event.region_id = observation.region_id
        event.box_history.append([float(value) for value in observation.anchor_box])
        stacked = np.asarray(event.box_history, dtype=float)
        event.anchor_box = np.median(stacked, axis=0).tolist()
        event.display_box = [float(value) for value in observation.display_box]
        event.reset_clean()
        event.reset_absent()
        event.suspended_since = None

    def _create_event(
        self, observation: HybridObservation, timestamp: float,
    ) -> V33Event:
        event = V33Event(
            event_id=self._next_id,
            region_id=observation.region_id,
            anchor_box=[float(value) for value in observation.anchor_box],
            display_box=[float(value) for value in observation.display_box],
            first_seen_at=timestamp,
            last_seen_at=timestamp,
        )
        event.box_history.append([float(value) for value in observation.anchor_box])
        self._apply_observation(event, observation, timestamp)
        self._assign_windows(event)
        self.events.append(event)
        self._next_id += 1
        return event

    # -- per-tick channels -----------------------------------------------

    def _tick_windows(
        self, *, timestamp: float, semantic_scan: bool,
        prior_available: bool, actors: Sequence[Sequence[float]],
    ) -> None:
        """Append exactly one entry per channel window for every open event.

        One tick counts at most once per channel, no matter how many tiles or
        crops reported the target. ``semantic_scan`` is True only for a full ROI
        scan: a prior-crop tick supplies auxiliary semantic evidence but must not
        advance the semantic confirmation window, whose cadence is defined in
        terms of full scans.
        """
        for event in self.active:
            if self._occluded(event.anchor_box, actors):
                event.current_support = set()
                if event.state != "OCCLUDED":
                    event.transition(
                        timestamp, "OCCLUDED", "actor_or_context_occluded"
                    )
            elif event.state == "OCCLUDED":
                event.transition(timestamp, "PENDING", "occlusion_cleared")
            if semantic_scan:
                event.semantic_window.append((
                    timestamp, event.last_semantic_at == timestamp,
                ))
            if prior_available:
                event.prior_window.append((
                    timestamp, event.last_prior_at == timestamp,
                ))
            event.fused_window.append((
                timestamp,
                event.last_semantic_at == timestamp
                and event.last_prior_at == timestamp,
            ))
            self._assign_windows(event)

    def _update_evidence(
        self, *, timestamp: float, prior_available: bool, semantic_scan: bool,
        actors: Sequence[Sequence[float]],
        support: np.ndarray | None, valid: np.ndarray | None,
    ) -> None:
        options = self.options
        for event in self.active:
            occluded = self._occluded(event.anchor_box, actors)
            if occluded:
                event.current_support = set()
            ground = self._ground_state(event.anchor_box, support, valid)

            # --- clean evidence: the prior channel's clear signal ---
            # Accumulated from real timestamps and *paused* (never reset) while
            # the spot cannot be judged: an occlusion or an environment
            # transition must not destroy a clean run that is already in
            # progress. `_elapsed` restarts the run when the pause exceeds the
            # allowed gap.
            if event.ever_prior and prior_available and ground is not None \
                    and not occluded:
                valid_fraction, support_pixels = ground
                clean_enough = (
                    valid_fraction >= float(options.min_clean_valid_fraction)
                    and support_pixels <= max(
                        1, round(CLEAN_MAX_SUPPORT_PIXELS * self.pixel_scale ** 2)
                    )
                )
                if clean_enough:
                    event.clean_seconds += self._elapsed(
                        event.last_clean_at, timestamp
                    )
                    event.last_clean_at = timestamp
                else:
                    # A residual is still present at the anchor: run ends.
                    event.reset_clean()

            # --- absence evidence: the semantic channel's clear signal ---
            # Only a full ROI scan can register a miss, and the run is paused
            # rather than cleared on ticks where no scan ran.
            if event.ever_semantic:
                if event.last_semantic_at == timestamp:
                    event.reset_absent()   # seen again: the absence run ends
                elif semantic_scan and not occluded and (
                    ground is None
                    or ground[0] >= float(options.min_clean_valid_fraction)
                ):
                    event.semantic_miss_count += 1
                    event.absent_seconds += self._elapsed(
                        event.last_absent_at, timestamp
                    )
                    event.last_absent_at = timestamp

            # --- prior suspension bookkeeping ---
            if prior_available:
                event.suspended_since = None
            elif event.suspended_since is None:
                event.suspended_since = timestamp

    def _elapsed(self, previous: float | None, timestamp: float) -> float:
        """Real elapsed time, restarting after an over-long gap.

        Evidence accumulates from actual timestamps rather than the nominal
        sample period, so an irregular tick cadence cannot silently under-count
        and a stalled channel cannot silently bridge the stall.
        """
        if previous is None:
            return 0.0
        gap = float(timestamp) - float(previous)
        if gap <= 0 or gap > self.max_sample_gap:
            return 0.0
        return gap

    # -- confirmation ----------------------------------------------------

    def _evaluate_all(self, timestamp: float) -> None:
        for event in self.active:
            self._evaluate(event, timestamp)

    def _evaluate(self, event: V33Event, timestamp: float) -> None:
        options = self.options
        semantic_confirmed = self._window_confirmed(
            event.semantic_window, int(options.semantic_confirm_hits),
            float(options.semantic_confirm_span_seconds),
        )
        prior_confirmed = self._window_confirmed(
            event.prior_window, int(options.prior_confirm_hits),
            float(options.prior_confirm_span_seconds),
        )
        fused_confirmed = self._window_confirmed(
            event.fused_window, int(options.fused_confirm_hits),
            float(options.fused_confirm_span_seconds),
        )
        # `evidence_kind` records the highest level ever reached and never
        # downgrades; `current_support` alone drives display and clearing.
        if event.ever_semantic and event.ever_prior:
            event.evidence_kind = EVIDENCE_SEMANTIC_AND_PRIOR
        elif event.ever_semantic:
            event.evidence_kind = EVIDENCE_SEMANTIC_ONLY
        else:
            event.evidence_kind = EVIDENCE_PRIOR_ONLY

        if event.state == "OCCLUDED":
            return
        if not (semantic_confirmed or prior_confirmed or fused_confirmed):
            if event.confirmed_at is None:
                event.transition(timestamp, "PENDING", "awaiting_confirmation")
            # A confirmed event keeps whatever state the closing rules set; a
            # tick without fresh support must not resurrect it as visible.
            return
        if event.confirmed_at is None:
            event.confirmed_at = timestamp
        if (semantic_confirmed and prior_confirmed) or fused_confirmed:
            event.transition(timestamp, "FUSED_VISIBLE", "fused_confirmed")
        elif semantic_confirmed:
            event.transition(timestamp, "SEMANTIC_VISIBLE", "semantic_confirmed")
        else:
            event.transition(timestamp, "PRIOR_VISIBLE", "prior_confirmed")

    # -- closing ---------------------------------------------------------

    def _close_events(self, *, timestamp: float, prior_available: bool) -> None:
        options = self.options
        for event in list(self.events):
            if not event.is_open:
                continue
            if event.confirmed_at is None:
                if timestamp - event.first_seen_at >= float(
                    options.pending_expire_seconds
                ):
                    event.closed_at = timestamp
                    event.closed_reason = "pending_timeout"
                    event.transition(timestamp, "EXPIRED_PENDING", "pending_timeout")
                continue

            # Occlusion pauses the clear timers and must not be overwritten by a
            # pending-clear state: the position is simply not judgeable now.
            if event.state == "OCCLUDED":
                continue

            if (
                not prior_available and event.suspended_since is not None
                and timestamp - event.suspended_since
                >= float(options.prior_suspend_expire_seconds)
            ):
                # Unjudgeable for too long. Close so memory stays bounded, but
                # never record this as "cleaned".
                event.closed_at = timestamp
                event.closed_reason = "profile_unavailable_timeout"
                event.transition(
                    timestamp, "EXPIRED_PENDING", "profile_unavailable_timeout"
                )
                continue

            clean_ready = event.clean_seconds >= float(options.clear_confirm_seconds)
            absent_ready = (
                event.semantic_miss_count >= int(options.semantic_clear_min_misses)
                and event.absent_seconds >= float(options.semantic_clear_seconds)
            )
            # Only an event with no support on this tick can be starting to
            # clear; an event still being seen keeps its visible state.
            clearing = not event.current_support

            if event.evidence_kind == EVIDENCE_SEMANTIC_ONLY:
                if not absent_ready:
                    if clearing:
                        event.transition(
                            timestamp, "ABSENT_PENDING", "semantic_absent"
                        )
                    continue
                if event.ever_prior and prior_available and not clean_ready:
                    # Clean Reference can also see this spot, so the later of the
                    # two conditions governs: one missed scan is not proof.
                    event.transition(
                        timestamp, "CLEAN_PENDING", "await_clean_evidence"
                    )
                    continue
                reason = "semantic_absent_confirmed"
            else:
                if not clean_ready:
                    if clearing:
                        event.transition(
                            timestamp, "CLEAN_PENDING", "clean_reference_match"
                        )
                    continue
                reason = "clean_confirmed"
            event.closed_at = timestamp
            event.closed_reason = reason
            event.transition(timestamp, "CLEARED", reason)

    def _merge(self, primary: V33Event, secondary: V33Event, timestamp: float) -> None:
        cross_source = (
            (primary.ever_semantic and secondary.ever_prior)
            or (primary.ever_prior and secondary.ever_semantic)
        )
        primary.first_seen_at = min(primary.first_seen_at, secondary.first_seen_at)
        primary.last_seen_at = max(primary.last_seen_at, secondary.last_seen_at)
        for window_name in ("semantic_window", "prior_window", "fused_window"):
            combined = sorted(
                set(getattr(primary, window_name))
                | set(getattr(secondary, window_name))
            )
            setattr(primary, window_name, deque(combined))
        primary.semantic_hits += secondary.semantic_hits
        primary.prior_hits += secondary.prior_hits
        primary.ever_semantic = primary.ever_semantic or secondary.ever_semantic
        primary.ever_prior = primary.ever_prior or secondary.ever_prior
        primary.maximum_semantic_confidence = max(
            primary.maximum_semantic_confidence,
            secondary.maximum_semantic_confidence,
        )
        if not primary.semantic_class:
            primary.semantic_class = secondary.semantic_class
        if not primary.region_id:
            primary.region_id = secondary.region_id
        primary.box_history.extend(secondary.box_history)
        primary.merges += 1 + secondary.merges
        if secondary.confirmed_at is not None:
            primary.confirmed_at = (
                secondary.confirmed_at if primary.confirmed_at is None
                else min(primary.confirmed_at, secondary.confirmed_at)
            )
        secondary.closed_at = timestamp
        secondary.closed_reason = f"merged_into:{primary.event_id}"
        secondary.transition(timestamp, "MERGED", secondary.closed_reason)
        if cross_source:
            self._cross_source_merges += 1
        for pair in list(self._merge_evidence):
            if secondary.event_id in pair:
                self._merge_evidence.pop(pair, None)

    def _prune_closed(self) -> None:
        closed = sorted(
            (event for event in self.events if not event.is_open),
            key=lambda item: (item.closed_at or 0.0, item.event_id),
        )
        excess = len(closed) - int(self.options.maximum_closed_events)
        if excess > 0:
            removed = {event.event_id for event in closed[:excess]}
            self.events = [
                event for event in self.events if event.event_id not in removed
            ]
        active_ids = {event.event_id for event in self.active}
        for pair in list(self._merge_evidence):
            if pair[0] not in active_ids or pair[1] not in active_ids:
                self._merge_evidence.pop(pair, None)

    # -- projection ------------------------------------------------------

    def _fresh_support(self, event: V33Event, timestamp: float) -> set[str]:
        support: set[str] = set()
        if (
            event.ever_semantic and event.last_semantic_at is not None
            and timestamp - event.last_semantic_at <= self.semantic_fresh_window
        ):
            support.add("semantic")
        if (
            event.ever_prior and event.last_prior_at is not None
            and timestamp - event.last_prior_at <= self.prior_fresh_window
        ):
            support.add("prior")
        return support

    def _project(
        self, timestamp: float, *, prior_available: bool,
    ) -> tuple[GroundLitterDetection, ...]:
        by_source: dict[str, list[tuple[V33Event, GroundLitterDetection]]] = {
            SOURCE_SEMANTIC: [],
            SOURCE_PRIOR: [],
            SOURCE_FUSED: [],
        }
        for event in self.active:
            if event.state not in DISPLAY_STATES:
                continue
            support = self._fresh_support(event, timestamp)
            if "semantic" in support:
                source = SOURCE_FUSED if event.ever_prior else SOURCE_SEMANTIC
            elif prior_available and "prior" in support:
                source = SOURCE_FUSED if event.ever_semantic else SOURCE_PRIOR
            else:
                # The only supporting channel is currently unusable, so the
                # event is hidden rather than shown with stale evidence.
                continue
            left, top, right, bottom = event.display_box
            detection = GroundLitterDetection(
                object_id=event.event_id,
                rectangle=NormalizedRect(
                    left / self._reference_width,
                    top / self._reference_height,
                    max(0.0, right - left) / self._reference_width,
                    max(0.0, bottom - top) / self._reference_height,
                ),
                confidence=float(event.maximum_semantic_confidence),
                class_name=event.semantic_class,
                region_id=event.region_id,
                hits=max(event.semantic_hits, event.prior_hits),
                source=source,
            )
            by_source[source].append((event, detection))

        # Confidence and anomaly scores do not have a shared scale.  Sort only
        # inside one source, then interleave non-empty sources round-robin.  The
        # cursor advances between projections so a tight maximum_boxes limit
        # cannot permanently hide prior-only evidence behind semantic boxes (or
        # vice versa).
        for source, rows in by_source.items():
            rows.sort(key=lambda row: (
                max(
                    0.0,
                    timestamp - max(
                        value for value in (
                            row[0].last_semantic_at,
                            row[0].last_prior_at,
                        )
                        if value is not None
                    ),
                ),
                row[0].confirmed_at if row[0].confirmed_at is not None else float("inf"),
                -row[1].confidence if source != SOURCE_PRIOR else 0.0,
                row[0].event_id,
            ))

        source_order = (SOURCE_FUSED, SOURCE_PRIOR, SOURCE_SEMANTIC)
        available_sources = tuple(
            source for source in source_order if by_source[source]
        )
        if not available_sources:
            return ()
        start = self._projection_source_cursor % len(available_sources)
        ordered_sources = available_sources[start:] + available_sources[:start]
        selected: list[GroundLitterDetection] = []
        maximum = int(self.options.maximum_boxes)
        while len(selected) < maximum:
            added = False
            for source in ordered_sources:
                rows = by_source[source]
                if not rows:
                    continue
                _event, detection = rows.pop(0)
                selected.append(detection)
                added = True
                if len(selected) >= maximum:
                    break
            if not added:
                break
        self._projection_source_cursor = (start + 1) % len(available_sources)
        return tuple(selected)

    def _result(
        self, *, detections, timestamp: float, state: str, message: str,
        environment_state: str, prior_available: bool, semantic_ran: bool,
        semantic_raw: int, semantic_retained: int,
        prior_raw: int, prior_retained: int,
        semantic_crop_raw: int, semantic_crop_unmatched: int,
    ) -> V33TickResult:
        active = self.active
        confirmed = [event for event in active if event.confirmed_at is not None]
        cleared = [
            event for event in self.events
            if event.closed_reason in ("clean_confirmed", "semantic_absent_confirmed")
        ]
        semantic_only = [e for e in active if e.evidence_kind == EVIDENCE_SEMANTIC_ONLY]
        prior_only = [e for e in active if e.evidence_kind == EVIDENCE_PRIOR_ONLY]
        fused = [e for e in active if e.evidence_kind == EVIDENCE_SEMANTIC_AND_PRIOR]
        if not message:
            if prior_available or semantic_ran:
                state = "running"
            else:
                state = "prior_abstaining"
            message = (
                f"hybrid_v33 profile={self.options.profile_id} "
                f"env={environment_state} "
                f"prior={'on' if prior_available else 'paused'} "
                f"events={len(active)} confirmed={len(confirmed)} "
                f"displayed={len(detections)} "
                f"semantic_only={len(semantic_only)} prior_only={len(prior_only)} "
                f"fused={len(fused)}"
            )
        return V33TickResult(
            detections=tuple(detections),
            state=state,
            message=message,
            active_events=len(active),
            confirmed_events=len(confirmed),
            cleared_events=len(cleared),
            semantic_only_active=len(semantic_only),
            semantic_only_confirmed=len(
                [e for e in semantic_only if e.confirmed_at is not None]
            ),
            prior_only_active=len(prior_only),
            prior_only_confirmed=len(
                [e for e in prior_only if e.confirmed_at is not None]
            ),
            fused_active=len(fused),
            fused_confirmed=len([e for e in fused if e.confirmed_at is not None]),
            cross_source_merges=self._cross_source_merges,
            environment_state=environment_state,
            prior_available=prior_available,
            semantic_ran=semantic_ran,
            prior_raw=prior_raw,
            prior_retained=prior_retained,
            semantic_raw=semantic_raw,
            semantic_retained=semantic_retained,
            semantic_crop_raw=semantic_crop_raw,
            semantic_crop_unmatched=semantic_crop_unmatched,
        )
