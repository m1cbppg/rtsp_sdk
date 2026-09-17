"""Conservative screen-space protection, not a calibrated vessel-speed model."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass

from .vessel_detection import VesselDetection


@dataclass(frozen=True, slots=True)
class EdgeDecision:
    zoom_delta: int = 0
    reason: str = "collecting_motion"
    x: float | None = None
    y: float | None = None
    velocity_reliable: bool = False
    time_to_edge_seconds: float | None = None
    predicted_margin: float | None = None


class TrackingEdgeGuard:
    """One-step escape protection with explicit action acknowledgement.

    Coordinates and velocities use frame widths/heights, not angles or world
    speed. Discard history after any camera action: no camera compensation is
    claimed. Call action_completed after confirmed physical completion only.
    """

    def __init__(
        self, *, response_seconds: float = 1.0, uncertainty_seconds: float = .25,
        motion_seconds: float = .5,
        cooldown_seconds: float = 8.0, stable_seconds: float = 3.0,
        maximum_age_seconds: float = .5, target_width: float = .33,
        target_height: float = .33, deadband: float = .05,
    ) -> None:
        self.response_seconds = response_seconds
        self.motion_seconds = motion_seconds
        self.uncertainty_seconds = uncertainty_seconds
        self.cooldown_seconds = cooldown_seconds
        self.stable_seconds = stable_seconds
        self.maximum_age_seconds = maximum_age_seconds
        self.target_width = target_width
        self.target_height = target_height
        self.deadband = deadband
        self._samples: deque[tuple[float, float, float, float]] = deque(maxlen=8)
        self._identity: tuple[str, int] | None = None
        self._last_stamp = -math.inf
        self._last_frame: tuple[str, int, int] | None = None
        self._not_before = -math.inf
        self._last_shrink = -math.inf
        self._stable_since: float | None = None
        self._risk_hits = 0
        self._reached_scale = False
        self._acceleration_until = -math.inf

    def action_completed(self, now: float, *, zoom_delta: int) -> None:
        self._samples.clear()
        self._risk_hits = 0
        if zoom_delta != 0:
            self._stable_since = None
        self._acceleration_until = -math.inf
        # Unknown upstream delay is an assumption, not a measured camera time.
        self._not_before = now + self.uncertainty_seconds
        if zoom_delta < 0:
            self._last_shrink = now
            self._reached_scale = False

    def update(self, detection: VesselDetection, *, now: float) -> EdgeDecision:
        stamp = detection.position_updated_at
        rect = detection.rectangle
        cx, cy = rect.center
        values = (cx, cy, rect.width, rect.height, detection.confidence, now)
        if (
            stamp is None or not math.isfinite(stamp)
            or not all(math.isfinite(v) for v in values)
            or not 0 <= now-stamp <= self.maximum_age_seconds
            or detection.observation_kind not in ("detector_measurement", "image_tracker_update")
            or (detection.observation_kind == "detector_measurement" and detection.confidence <= 0)
            or rect.width <= 0 or rect.height <= 0
        ):
            self._samples.clear()
            self._stable_since = None
            self._risk_hits = 0
            return EdgeDecision(reason="unreliable_observation")
        if stamp < self._not_before:
            return EdgeDecision(reason="camera_settling")
        frame_key = (
            (detection.source, detection.object_id, detection.frame_id)
            if detection.frame_id is not None else None
        )
        if stamp <= self._last_stamp or (frame_key is not None and frame_key == self._last_frame):
            return EdgeDecision(reason="duplicate_observation")
        self._last_stamp = stamp
        self._last_frame = frame_key
        identity = (detection.source, detection.object_id)
        score = max(rect.width/self.target_width, rect.height/self.target_height)
        discontinuity = bool(self._samples and (
            stamp-self._samples[-1][0] > self.maximum_age_seconds
            or not .75 <= score/max(self._samples[-1][3], 1e-6) <= 1.33
        ))
        if identity != self._identity or discontinuity:
            self._samples.clear()
            self._risk_hits = 0
            self._stable_since = None
            self._acceleration_until = -math.inf
        self._identity = identity
        self._samples.append((stamp, cx, cy, score))
        if len(self._samples) < 3:
            if max(abs(cx-.5), abs(cy-.5)) > .35:
                return EdgeDecision(reason="unmodelled_edge_recenter", x=cx, y=cy)
            return EdgeDecision()
        speeds = [((b[1]-a[1])/(b[0]-a[0]), (b[2]-a[2])/(b[0]-a[0]))
                  for a, b in zip(self._samples, list(self._samples)[1:])]
        # Two consecutive increments reject isolated box jitter while reacting
        # promptly to acceleration. Older samples provide the speed baseline.
        recent = speeds[-2:]
        vx = sum(v[0] for v in recent)/2
        vy = sum(v[1] for v in recent)/2
        reliable = all(
            abs(a-b) <= max(.04, max(abs(a), abs(b))*.5)
            for a, b in zip(recent[0], recent[1])
        )
        if not reliable:
            self._risk_hits = 0
            self._stable_since = None
            return EdgeDecision(reason="inconsistent_motion", x=cx, y=cy)
        horizon = now-stamp + self.response_seconds + self.motion_seconds + self.uncertainty_seconds
        dx, dy = vx*horizon, vy*horizon
        margins = (rect.left, 1-rect.left-rect.width, rect.top, 1-rect.top-rect.height)
        outward = (-vx, vx, -vy, vy)
        edge_times = [max(m, 0)/v for m, v in zip(margins, outward) if v > .01]
        edge_time = min(edge_times, default=math.inf)
        predicted_margin = min(m-v*horizon for m, v in zip(margins, outward))
        baseline = max((max(abs(x), abs(y)) for x, y in speeds[:-2]), default=0)
        speed = max(abs(vx), abs(vy))
        accelerating = bool(speeds[:-2] and speed > max(.04, baseline*1.7+.02))
        if accelerating:
            self._acceleration_until = now + min(horizon, 1.0)
        # Acceleration alone is not a reason to shrink. Allow a last-resort
        # escape for imminent clipping even without an acceleration baseline.
        risk = predicted_margin < .06 and (
            now < self._acceleration_until or edge_time < horizon
        )
        self._risk_hits = self._risk_hits+1 if risk else 0
        stable = (not risk and predicted_margin > .12
                  and abs(cx-.5) <= self.deadband and abs(cy-.5) <= self.deadband)
        if not stable:
            self._stable_since = None
        elif self._stable_since is None:
            self._stable_since = now
        delta = 0
        reason = "position_priority" if predicted_margin < .12 else "following"
        cooling = now-self._last_shrink < self.cooldown_seconds
        # Reliability already requires two consistent measured displacements.
        # Waiting another frame after confirming imminent clipping can consume
        # the remaining actuator margin without adding independent evidence.
        if self._risk_hits >= 1:
            reason = "escape_cooldown" if cooling else "edge_escape"
            delta = 0 if cooling else -1
        elif stable and not cooling:
            filtered_score = statistics.median(s[3] for s in list(self._samples)[-3:])
            if filtered_score >= 1:
                self._reached_scale = True
            if (not self._reached_scale and filtered_score < .95
                    and self._stable_since is not None
                    and now-self._stable_since >= self.stable_seconds):
                delta, reason = 1, "stable_zoom_in"
        # Bound prediction rather than sending an arbitrarily distant point.
        x = min(.95, max(.05, cx+max(-.30, min(.30, dx))))
        y = min(.95, max(.05, cy+max(-.30, min(.30, dy))))
        return EdgeDecision(delta, reason, x, y, True,
                            edge_time if math.isfinite(edge_time) else None,
                            predicted_margin)
