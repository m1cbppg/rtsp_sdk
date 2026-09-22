from __future__ import annotations

import math
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import median
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .event_engine import NormalizedRect
from .events import EventRecord, WebhookOptions
from .vessel_detection import VesselDetection, rectangle_iou


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True, slots=True)
class FishingRiskZoneOptions:
    zone_id: str
    polygon: tuple[tuple[float, float], ...]

    def validate(self) -> None:
        if not _SAFE_IDENTIFIER.fullmatch(self.zone_id):
            raise ValueError(
                "fishing_risk区域id只能包含字母、数字、下划线和短横线"
            )
        _validate_polygon(self.polygon, "fishing_risk.zones.polygon")


@dataclass(frozen=True, slots=True)
class FishingRiskScheduleOptions:
    schedule_id: str
    start_at: datetime
    end_at: datetime

    def validate(self) -> None:
        if not _SAFE_IDENTIFIER.fullmatch(self.schedule_id):
            raise ValueError(
                "fishing_risk时间表id只能包含字母、数字、下划线和短横线"
            )
        if self.start_at.utcoffset() is None or self.end_at.utcoffset() is None:
            raise ValueError("fishing_risk时间表必须包含UTC偏移量")
        if self.end_at <= self.start_at:
            raise ValueError("fishing_risk时间表end_at必须晚于start_at")

    def active(self, observed_at: datetime) -> bool:
        return self.start_at <= observed_at <= self.end_at


@dataclass(frozen=True, slots=True)
class FishingRiskRuleOptions:
    minimum_presence_seconds: float = 30.0
    loitering_seconds: float = 180.0
    loitering_radius_box_lengths: float = 4.0
    reversal_window_seconds: float = 120.0
    minimum_reversals: int = 2
    reversal_angle_degrees: float = 120.0
    minimum_motion_box_lengths: float = 0.50
    minimum_reversal_interval_seconds: float = 8.0
    track_lost_seconds: float = 15.0
    track_match_box_lengths: float = 3.0
    startup_grace_seconds: float = 60.0
    preexisting_activation_box_lengths: float = 2.0

    def validate(self) -> None:
        if not 1 <= self.minimum_presence_seconds <= 86_400:
            raise ValueError(
                "minimum_presence_seconds必须在[1, 86400]范围内"
            )
        if not 5 <= self.loitering_seconds <= 86_400:
            raise ValueError("loitering_seconds必须在[5, 86400]范围内")
        if not 0.5 <= self.loitering_radius_box_lengths <= 50:
            raise ValueError(
                "loitering_radius_box_lengths必须在[0.5, 50]范围内"
            )
        if not 10 <= self.reversal_window_seconds <= 3_600:
            raise ValueError(
                "reversal_window_seconds必须在[10, 3600]范围内"
            )
        if not 1 <= self.minimum_reversals <= 20:
            raise ValueError("minimum_reversals必须在[1, 20]范围内")
        if not 60 <= self.reversal_angle_degrees <= 180:
            raise ValueError(
                "reversal_angle_degrees必须在[60, 180]范围内"
            )
        if not 0.05 <= self.minimum_motion_box_lengths <= 10:
            raise ValueError(
                "minimum_motion_box_lengths必须在[0.05, 10]范围内"
            )
        if not 1 <= self.minimum_reversal_interval_seconds <= 300:
            raise ValueError(
                "minimum_reversal_interval_seconds必须在[1, 300]范围内"
            )
        if not 1 <= self.track_lost_seconds <= 300:
            raise ValueError("track_lost_seconds必须在[1, 300]范围内")
        if not 0.5 <= self.track_match_box_lengths <= 20:
            raise ValueError(
                "track_match_box_lengths必须在[0.5, 20]范围内"
            )
        if not 0 <= self.startup_grace_seconds <= 3_600:
            raise ValueError(
                "startup_grace_seconds必须在[0, 3600]范围内"
            )
        if not 0.5 <= self.preexisting_activation_box_lengths <= 50:
            raise ValueError(
                "preexisting_activation_box_lengths必须在[0.5, 50]范围内"
            )


@dataclass(frozen=True, slots=True)
class FishingRiskOptions:
    enabled: bool = False
    timezone: str = "Asia/Shanghai"
    zones: tuple[FishingRiskZoneOptions, ...] = ()
    schedules: tuple[FishingRiskScheduleOptions, ...] = ()
    rules: FishingRiskRuleOptions = FishingRiskRuleOptions()
    restricted_presence_score: int = 40
    loitering_score: int = 20
    direction_reversal_score: int = 20
    alert_score: int = 60
    cooldown_seconds: float = 300.0
    display_risk: bool = True
    webhook: WebhookOptions = WebhookOptions()

    def validate(self) -> None:
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"未知时区: {self.timezone}") from exc
        if self.enabled and not self.zones:
            raise ValueError("启用fishing_risk时至少需要一个zones区域")
        zone_ids = [item.zone_id for item in self.zones]
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("fishing_risk区域id不能重复")
        for zone in self.zones:
            zone.validate()
        schedule_ids = [item.schedule_id for item in self.schedules]
        if len(schedule_ids) != len(set(schedule_ids)):
            raise ValueError("fishing_risk时间表id不能重复")
        for schedule in self.schedules:
            schedule.validate()
        self.rules.validate()
        scores = (
            self.restricted_presence_score,
            self.loitering_score,
            self.direction_reversal_score,
        )
        if any(not 0 <= item <= 100 for item in scores):
            raise ValueError("fishing_risk各规则分值必须在[0, 100]范围内")
        if not 1 <= self.alert_score <= 100:
            raise ValueError("fishing_risk.alert_score必须在[1, 100]范围内")
        if self.alert_score > sum(scores):
            raise ValueError("fishing_risk.alert_score不能高于所有规则分值之和")
        if not 0 <= self.cooldown_seconds <= 86_400:
            raise ValueError(
                "fishing_risk.cooldown_seconds必须在[0, 86400]范围内"
            )
        self.webhook.validate()

    def active_schedule_ids(self, observed_at: datetime) -> tuple[str, ...]:
        if observed_at.utcoffset() is None:
            raise ValueError("observed_at必须包含UTC偏移量")
        if not self.schedules:
            return ("always",)
        return tuple(
            schedule.schedule_id
            for schedule in self.schedules
            if schedule.active(observed_at)
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "timezone": self.timezone,
            "zones": [
                {
                    "id": zone.zone_id,
                    "polygon": [list(point) for point in zone.polygon],
                }
                for zone in self.zones
            ],
            "schedules": [
                {
                    "id": schedule.schedule_id,
                    "start_at": schedule.start_at.isoformat(),
                    "end_at": schedule.end_at.isoformat(),
                }
                for schedule in self.schedules
            ],
            "rules": asdict(self.rules),
            "restricted_presence_score": self.restricted_presence_score,
            "loitering_score": self.loitering_score,
            "direction_reversal_score": self.direction_reversal_score,
            "alert_score": self.alert_score,
            "cooldown_seconds": self.cooldown_seconds,
            "display_risk": self.display_risk,
            "webhook": asdict(self.webhook),
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any] | None,
    ) -> "FishingRiskOptions":
        data = dict(payload or {})
        zones = tuple(
            FishingRiskZoneOptions(
                zone_id=str(item["id"]),
                polygon=tuple(
                    (float(point[0]), float(point[1]))
                    for point in item["polygon"]
                ),
            )
            for item in data.pop("zones", [])
        )
        schedules = tuple(
            FishingRiskScheduleOptions(
                schedule_id=str(item["id"]),
                start_at=_parse_datetime(item["start_at"]),
                end_at=_parse_datetime(item["end_at"]),
            )
            for item in data.pop("schedules", [])
        )
        rules = FishingRiskRuleOptions(**dict(data.pop("rules", {}) or {}))
        webhook = WebhookOptions(**dict(data.pop("webhook", {}) or {}))
        options = cls(
            zones=zones,
            schedules=schedules,
            rules=rules,
            webhook=webhook,
            **data,
        )
        options.validate()
        return options


@dataclass(frozen=True, slots=True)
class FishingRiskSuspect:
    risk_track_id: int
    vessel_object_id: int
    rectangle: NormalizedRect
    zone_id: str
    risk_score: int
    reasons: tuple[str, ...]
    dwell_seconds: float
    reversal_count: int
    compact_span_box_lengths: float | None = None


@dataclass(frozen=True, slots=True)
class FishingRiskSnapshot:
    state: str = "disabled"
    suspects: tuple[FishingRiskSuspect, ...] = ()
    result_version: int = 0
    updated_at: float | None = None
    total_events: int = 0
    message: str = ""

    @property
    def count(self) -> int:
        return len(self.suspects)

    @property
    def maximum_score(self) -> int:
        return max(
            (item.risk_score for item in self.suspects),
            default=0,
        )


@dataclass(slots=True)
class FishingRiskResult:
    snapshot: FishingRiskSnapshot
    events: list[EventRecord] = field(default_factory=list)


class FishingRiskResultCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[int, FishingRiskSnapshot] = {}

    def snapshot(self, pad_index: int) -> FishingRiskSnapshot:
        with self._lock:
            return self._entries.get(pad_index, FishingRiskSnapshot())

    def store_snapshot(
        self,
        pad_index: int,
        snapshot: FishingRiskSnapshot,
    ) -> None:
        with self._lock:
            self._entries[pad_index] = snapshot

    def mark_state(
        self,
        pad_index: int,
        state: str,
        message: str = "",
    ) -> None:
        with self._lock:
            previous = self._entries.get(
                pad_index,
                FishingRiskSnapshot(),
            )
            self._entries[pad_index] = FishingRiskSnapshot(
                state=state,
                suspects=previous.suspects,
                result_version=previous.result_version,
                updated_at=previous.updated_at,
                total_events=previous.total_events,
                message=message,
            )


@dataclass(frozen=True, slots=True)
class _TrailPoint:
    timestamp: float
    center: tuple[float, float]
    scale: float


@dataclass(slots=True)
class _ZoneState:
    entered_at: float
    last_seen_at: float
    trail: list[_TrailPoint] = field(default_factory=list)
    heading_anchor: tuple[float, float] | None = None
    last_motion_center: tuple[float, float] | None = None
    reversal_times: list[float] = field(default_factory=list)
    last_reversal_at: float | None = None
    alerted: bool = False


@dataclass(slots=True)
class _RiskTrack:
    risk_track_id: int
    vessel_object_id: int
    rectangle: NormalizedRect
    first_seen_at: float
    last_seen_at: float
    initial_center: tuple[float, float]
    initial_scale: float
    preexisting: bool
    zone_states: dict[str, _ZoneState] = field(default_factory=dict)
    last_alert_by_zone: dict[str, float] = field(default_factory=dict)


class FishingRiskEngine:
    """Deterministic, camera-only fishing-risk clue engine.

    It emits reviewable clues, never a legal conclusion. Long-lived risk IDs
    are associated independently from the short display hold used by the
    vessel detector, so a brief missed detection does not reset dwell time.
    """

    def __init__(
        self,
        *,
        stream_id: str,
        options: FishingRiskOptions,
        on_event: Callable[[EventRecord], None] | None = None,
    ) -> None:
        options.validate()
        self.stream_id = stream_id
        self.options = options
        self._on_event = on_event
        self._tracks: dict[int, _RiskTrack] = {}
        self._next_track_id = 1
        self._started_at: float | None = None
        self._version = 0
        self._total_events = 0

    def reset_tracking(self) -> None:
        """Forget spatial history after the camera viewpoint changes."""
        self._tracks.clear()
        self._started_at = None

    def observe(
        self,
        *,
        timestamp: float,
        observed_at: datetime,
        detections: Iterable[VesselDetection],
    ) -> FishingRiskResult:
        if not self.options.enabled:
            return FishingRiskResult(FishingRiskSnapshot())
        if observed_at.utcoffset() is None:
            raise ValueError("observed_at必须包含UTC偏移量")
        if self._started_at is None:
            self._started_at = timestamp
        self._expire(timestamp)
        matched = self._associate(list(detections), timestamp)
        active_schedules = self.options.active_schedule_ids(observed_at)
        suspects: list[FishingRiskSuspect] = []
        events: list[EventRecord] = []
        for track in matched:
            self._activate_preexisting_if_moved(track, timestamp)
            for zone in self.options.zones:
                inside = _point_in_polygon(
                    track.rectangle.center,
                    zone.polygon,
                )
                if not inside or not active_schedules:
                    track.zone_states.pop(zone.zone_id, None)
                    continue
                zone_state = track.zone_states.get(zone.zone_id)
                if zone_state is None:
                    zone_state = _ZoneState(timestamp, timestamp)
                    track.zone_states[zone.zone_id] = zone_state
                zone_state.last_seen_at = timestamp
                self._append_trail(zone_state, track, timestamp)
                if track.preexisting:
                    continue
                suspect = self._evaluate_zone(track, zone, zone_state, timestamp)
                if suspect.risk_score > 0:
                    suspects.append(suspect)
                if suspect.risk_score < self.options.alert_score:
                    continue
                if zone_state.alerted:
                    continue
                previous_alert = track.last_alert_by_zone.get(zone.zone_id)
                if (
                    previous_alert is not None
                    and timestamp - previous_alert < self.options.cooldown_seconds
                ):
                    continue
                event = self._create_event(
                    track,
                    suspect,
                    observed_at,
                    active_schedules,
                )
                zone_state.alerted = True
                track.last_alert_by_zone[zone.zone_id] = timestamp
                self._total_events += 1
                events.append(event)
                if self._on_event is not None:
                    self._on_event(event)
        self._version += 1
        snapshot = FishingRiskSnapshot(
            state="running",
            suspects=tuple(
                sorted(
                    suspects,
                    key=lambda item: (
                        -item.risk_score,
                        item.risk_track_id,
                        item.zone_id,
                    ),
                )
            ),
            result_version=self._version,
            updated_at=timestamp,
            total_events=self._total_events,
        )
        return FishingRiskResult(snapshot, events)

    def _associate(
        self,
        detections: list[VesselDetection],
        timestamp: float,
    ) -> list[_RiskTrack]:
        unmatched_tracks = set(self._tracks)
        unmatched_detections = set(range(len(detections)))
        candidates: list[tuple[float, int, int]] = []
        for risk_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                overlap = rectangle_iou(track.rectangle, detection.rectangle)
                distance = math.dist(
                    track.rectangle.center,
                    detection.rectangle.center,
                )
                scale = max(
                    _rectangle_scale(track.rectangle),
                    _rectangle_scale(detection.rectangle),
                    0.005,
                )
                distance_in_boxes = distance / scale
                same_source = track.vessel_object_id == detection.object_id
                if (
                    not same_source
                    and overlap <= 0
                    and distance_in_boxes
                    > self.options.rules.track_match_box_lengths
                ):
                    continue
                score = (
                    (10.0 if same_source else 0.0)
                    + overlap * 2.0
                    - distance_in_boxes * 0.1
                )
                candidates.append((score, risk_id, detection_index))
        matched: list[_RiskTrack] = []
        for _score, risk_id, detection_index in sorted(
            candidates,
            reverse=True,
        ):
            if (
                risk_id not in unmatched_tracks
                or detection_index not in unmatched_detections
            ):
                continue
            detection = detections[detection_index]
            track = self._tracks[risk_id]
            track.vessel_object_id = detection.object_id
            track.rectangle = detection.rectangle
            track.last_seen_at = timestamp
            unmatched_tracks.remove(risk_id)
            unmatched_detections.remove(detection_index)
            matched.append(track)
        assert self._started_at is not None
        for detection_index in sorted(unmatched_detections):
            detection = detections[detection_index]
            center = detection.rectangle.center
            scale = _rectangle_scale(detection.rectangle)
            track = _RiskTrack(
                risk_track_id=self._next_track_id,
                vessel_object_id=detection.object_id,
                rectangle=detection.rectangle,
                first_seen_at=timestamp,
                last_seen_at=timestamp,
                initial_center=center,
                initial_scale=scale,
                preexisting=(
                    self.options.rules.startup_grace_seconds > 0
                    and timestamp - self._started_at
                    <= self.options.rules.startup_grace_seconds
                ),
            )
            self._tracks[track.risk_track_id] = track
            self._next_track_id += 1
            matched.append(track)
        return matched

    def _activate_preexisting_if_moved(
        self,
        track: _RiskTrack,
        timestamp: float,
    ) -> None:
        if not track.preexisting:
            return
        moved = math.dist(track.initial_center, track.rectangle.center)
        if (
            moved / max(track.initial_scale, 0.005)
            < self.options.rules.preexisting_activation_box_lengths
        ):
            return
        track.preexisting = False
        track.first_seen_at = timestamp
        track.zone_states.clear()

    def _append_trail(
        self,
        state: _ZoneState,
        track: _RiskTrack,
        timestamp: float,
    ) -> None:
        point = _TrailPoint(
            timestamp,
            track.rectangle.center,
            _rectangle_scale(track.rectangle),
        )
        state.trail.append(point)
        history_seconds = max(
            self.options.rules.loitering_seconds,
            self.options.rules.reversal_window_seconds,
        ) + self.options.rules.track_lost_seconds
        cutoff = timestamp - history_seconds
        state.trail[:] = [item for item in state.trail if item.timestamp >= cutoff]
        del state.trail[:-2_000]

        previous = state.last_motion_center
        if previous is None:
            state.last_motion_center = point.center
            return
        distance = math.dist(previous, point.center)
        if distance < (
            self.options.rules.minimum_motion_box_lengths
            * max(point.scale, 0.005)
        ):
            return
        direction = (
            point.center[0] - previous[0],
            point.center[1] - previous[1],
        )
        state.last_motion_center = point.center
        if state.heading_anchor is None:
            state.heading_anchor = direction
            return
        angle = _angle_degrees(state.heading_anchor, direction)
        enough_time = (
            state.last_reversal_at is None
            or timestamp - state.last_reversal_at
            >= self.options.rules.minimum_reversal_interval_seconds
        )
        if (
            angle >= self.options.rules.reversal_angle_degrees
            and enough_time
        ):
            state.reversal_times.append(timestamp)
            state.last_reversal_at = timestamp
            state.heading_anchor = direction
        elif angle <= 45:
            state.heading_anchor = direction
        cutoff = timestamp - self.options.rules.reversal_window_seconds
        state.reversal_times[:] = [
            item for item in state.reversal_times if item >= cutoff
        ]

    def _evaluate_zone(
        self,
        track: _RiskTrack,
        zone: FishingRiskZoneOptions,
        state: _ZoneState,
        timestamp: float,
    ) -> FishingRiskSuspect:
        dwell_seconds = max(timestamp - state.entered_at, 0.0)
        reasons: list[str] = []
        score = 0
        if dwell_seconds >= self.options.rules.minimum_presence_seconds:
            reasons.append("restricted_period_presence")
            score += self.options.restricted_presence_score
        compact_span = self._compact_span(state, timestamp)
        if (
            dwell_seconds >= self.options.rules.loitering_seconds
            and compact_span is not None
            and compact_span
            <= self.options.rules.loitering_radius_box_lengths
        ):
            reasons.append("loitering")
            score += self.options.loitering_score
        reversal_count = len(state.reversal_times)
        if reversal_count >= self.options.rules.minimum_reversals:
            reasons.append("direction_reversal")
            score += self.options.direction_reversal_score
        return FishingRiskSuspect(
            risk_track_id=track.risk_track_id,
            vessel_object_id=track.vessel_object_id,
            rectangle=track.rectangle,
            zone_id=zone.zone_id,
            risk_score=min(score, 100),
            reasons=tuple(reasons),
            dwell_seconds=dwell_seconds,
            reversal_count=reversal_count,
            compact_span_box_lengths=compact_span,
        )

    def _compact_span(
        self,
        state: _ZoneState,
        timestamp: float,
    ) -> float | None:
        cutoff = timestamp - self.options.rules.loitering_seconds
        points = [item for item in state.trail if item.timestamp >= cutoff]
        if not points or points[0].timestamp > cutoff + 1.0:
            return None
        width = max(item.center[0] for item in points) - min(
            item.center[0] for item in points
        )
        height = max(item.center[1] for item in points) - min(
            item.center[1] for item in points
        )
        scale = max(median(item.scale for item in points), 0.005)
        return math.hypot(width, height) / scale

    def _create_event(
        self,
        track: _RiskTrack,
        suspect: FishingRiskSuspect,
        observed_at: datetime,
        active_schedules: tuple[str, ...],
    ) -> EventRecord:
        local_time = observed_at.astimezone(ZoneInfo(self.options.timezone))
        reason_labels = {
            "restricted_period_presence": "禁渔时空内持续出现",
            "loitering": "小范围长时间停留",
            "direction_reversal": "多次折返",
        }
        description = "、".join(reason_labels[item] for item in suspect.reasons)
        risk_level = "critical" if suspect.risk_score >= 80 else "high"
        return EventRecord.create(
            stream_id=self.stream_id,
            event_type="suspected_illegal_fishing",
            roi_id=suspect.zone_id,
            message=(
                f"船舶出现疑似捕捞线索：{description}；"
                f"风险分{suspect.risk_score}，需人工复核"
            ),
            actor_type="vessel",
            actor_track_id=track.risk_track_id,
            confidence=suspect.risk_score / 100.0,
            metadata={
                "legal_conclusion": False,
                "review_required": True,
                "risk_score": suspect.risk_score,
                "risk_level": risk_level,
                "reasons": list(suspect.reasons),
                "dwell_seconds": round(suspect.dwell_seconds, 3),
                "reversal_count": suspect.reversal_count,
                "compact_span_box_lengths": (
                    round(suspect.compact_span_box_lengths, 3)
                    if suspect.compact_span_box_lengths is not None
                    else None
                ),
                "zone_id": suspect.zone_id,
                "risk_track_id": suspect.risk_track_id,
                "vessel_object_id": suspect.vessel_object_id,
                "active_schedule_ids": list(active_schedules),
                "observed_at": observed_at.isoformat(),
                "observed_at_local": local_time.isoformat(),
            },
        )

    def _expire(self, timestamp: float) -> None:
        expired = [
            risk_id
            for risk_id, track in self._tracks.items()
            if timestamp - track.last_seen_at
            > self.options.rules.track_lost_seconds
        ]
        for risk_id in expired:
            self._tracks.pop(risk_id, None)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _validate_polygon(
    polygon: tuple[tuple[float, float], ...],
    field_name: str,
) -> None:
    if len(polygon) < 3:
        raise ValueError(f"{field_name}至少需要3个顶点")
    for x, y in polygon:
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError(f"{field_name}坐标必须在[0, 1]范围内")


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
            x_at_y = (
                previous_x
                + (y - previous_y)
                * (current_x - previous_x)
                / (current_y - previous_y)
            )
            if x <= x_at_y:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _rectangle_scale(rectangle: NormalizedRect) -> float:
    return math.hypot(rectangle.width, rectangle.height)


def _angle_degrees(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    left_length = math.hypot(*left)
    right_length = math.hypot(*right)
    if left_length <= 1e-12 or right_length <= 1e-12:
        return 0.0
    cosine = (
        left[0] * right[0] + left[1] * right[1]
    ) / (left_length * right_length)
    return math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))
