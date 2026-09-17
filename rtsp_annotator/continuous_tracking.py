"""Deterministic, bounded primitives for the continuous PTZ demo.

The production coordinator uses these small objects instead of treating a
blocking camera request as the tracking loop.  They are intentionally free of
DeepStream and HTTP dependencies so delayed closed-loop tests can exercise the
same policy on a laptop.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Any


class ObservationKind(str, Enum):
    MEASUREMENT = "detector_measurement"
    TRACKER = "image_tracker_update"
    PREDICTION = "prediction"
    HELD = "held_display"


@dataclass(frozen=True, slots=True)
class TrackingObservation:
    target_id: str
    x: float
    y: float
    width: float
    height: float
    source: str
    frame_id: int | None
    view_epoch: int
    source_timestamp: float
    received_at: float
    kind: ObservationKind = ObservationKind.MEASUREMENT
    confidence: float = 1.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def scale(self) -> float:
        return max(self.width, self.height)

    def usable_for_control(
        self,
        *,
        now: float,
        current_view_epoch: int,
        maximum_age: float,
    ) -> bool:
        if self.view_epoch != current_view_epoch:
            return False
        if self.kind is ObservationKind.HELD:
            return False
        if self.source_timestamp > now + 0.05:
            return False
        if now - self.source_timestamp > maximum_age:
            return False
        return self.confidence > 0.0


@dataclass(frozen=True, slots=True)
class TrackingIntent:
    sequence: int
    target_id: str
    x: float
    y: float
    zoom_delta: int
    created_at: float
    expires_at: float
    reason: str
    view_epoch: int


class LatestIntentDispatcher:
    """One in-flight action plus one replaceable latest intent.

    A slow/non-replaceable device never receives a stale queue of relative
    moves.  The executor callback runs outside the lock; a newly published
    intent supersedes the pending one and expired intents are discarded.
    """

    def __init__(
        self,
        execute: Callable[[TrackingIntent], Any],
        *,
        clock=time.monotonic,
        on_error: Callable[[TrackingIntent, BaseException], Any] | None = None,
        on_complete: Callable[[TrackingIntent, Any], Any] | None = None,
    ):
        self._execute = execute
        self._clock = clock
        self._condition = threading.Condition()
        self._pending: TrackingIntent | None = None
        self._in_flight = False
        self._closed = False
        self._error: BaseException | None = None
        self._on_error = on_error
        self._on_complete = on_complete
        self._sequence = 0
        self._thread = threading.Thread(target=self._run, name="ptz-latest-intent", daemon=True)
        self._thread.start()

    @property
    def pending(self) -> TrackingIntent | None:
        with self._condition:
            return self._pending

    @property
    def in_flight(self) -> bool:
        with self._condition:
            return self._in_flight

    @property
    def error(self) -> BaseException | None:
        with self._condition:
            return self._error

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def closed(self) -> bool:
        """Whether the dispatcher has been closed.

        ``failed`` and ``closed`` are intentionally separate so callers can
        distinguish an explicit STOP/HOME cancellation from a camera-control
        fault.  The worker thread may still be winding down after ``close``
        returns when an SDK call is non-interruptible.
        """
        with self._condition:
            return self._closed

    @property
    def status(self) -> str:
        """Return a small, race-safe lifecycle state for observability."""
        with self._condition:
            if self._closed:
                return "closed"
            if self._error is not None:
                return "faulted"
            if self._in_flight:
                return "executing"
            if self._pending is not None:
                return "pending"
            return "idle"

    def submit(
        self,
        *,
        target_id: str,
        x: float,
        y: float,
        zoom_delta: int,
        reason: str,
        view_epoch: int,
        ttl_seconds: float = 1.5,
    ) -> TrackingIntent:
        now = self._clock()
        with self._condition:
            if self._closed or self._error is not None:
                return TrackingIntent(
                    self._sequence, target_id, x, y, int(zoom_delta), now,
                    now, reason, view_epoch,
                )
            self._sequence += 1
            intent = TrackingIntent(
                self._sequence, target_id, x, y, int(zoom_delta), now,
                now + max(ttl_seconds, 0.05), reason, view_epoch,
            )
            if self._pending is None or intent.sequence > self._pending.sequence:
                self._pending = intent
            self._condition.notify()
            return intent

    def cancel(self) -> None:
        with self._condition:
            self._pending = None
            self._sequence += 1
            self._condition.notify_all()

    def close(self, timeout: float = 2.0) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=max(timeout, 0.0))

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed and self._pending is None:
                    self._condition.wait(0.2)
                if self._closed:
                    return
                intent = self._pending
                self._pending = None
                self._in_flight = True
            assert intent is not None
            try:
                if intent.expires_at >= self._clock():
                    result = self._execute(intent)
                    if self._on_complete is not None:
                        self._on_complete(intent, result)
            except Exception as exc:
                # A camera SDK failure must be visible to the coordinator and
                # must stop the dispatcher from issuing any newer relative
                # moves.  Keep the callback outside the condition lock so an
                # observer can safely call ``cancel``/``close``.
                with self._condition:
                    self._error = exc
                    self._pending = None
                    self._condition.notify_all()
                if self._on_error is not None:
                    try:
                        self._on_error(intent, exc)
                    except Exception:
                        # Error reporting is best effort; it must never kill
                        # the worker before ``_in_flight`` is cleared below.
                        pass
            finally:
                with self._condition:
                    self._in_flight = False
                    self._condition.notify_all()


@dataclass(slots=True)
class DemoTrackingController:
    """Pure policy helper shared by coordinator and offline simulation."""

    target_width: float = 0.33
    target_height: float = 0.33
    center_deadband: float = 0.10
    maximum_age: float = 0.75
    prediction_horizon: float = 1.0
    max_zoom_delta: int = 1

    def decide(
        self,
        observation: TrackingObservation,
        *,
        now: float,
        view_epoch: int,
        stable_for: float = 0.0,
    ) -> tuple[float, float, int, str] | None:
        if not observation.usable_for_control(
            now=now, current_view_epoch=view_epoch, maximum_age=self.maximum_age
        ):
            return None
        cx, cy = observation.center
        projected_x = cx + observation.velocity_x * self.prediction_horizon
        projected_y = cy + observation.velocity_y * self.prediction_horizon
        x = min(max(projected_x, 0.0), 1.0)
        y = min(max(projected_y, 0.0), 1.0)
        error = math.hypot(x - 0.5, y - 0.5)
        position = error > self.center_deadband
        scale = max(
            observation.width / max(self.target_width, 1e-6),
            observation.height / max(self.target_height, 1e-6),
        )
        # Position remains the primary objective; a small zoom step is allowed
        # while the target is inside a safe central corridor so long motion
        # does not postpone close-up indefinitely.
        zoom = self.max_zoom_delta if scale < 0.95 and error < 0.35 else 0
        if (x < 0.16 or x > 0.84) and (observation.velocity_x * (x - 0.5) > 0):
            zoom = -1
            position = True
        if zoom and stable_for < 0.0:
            zoom = 0
        if not position and zoom == 0:
            return None
        reason = "edge_escape" if zoom < 0 else ("position_correction" if position else "progressive_zoom")
        return (x, y, zoom, reason)


@dataclass(frozen=True, slots=True)
class ClosedLoopCase:
    name: str
    response_seconds: float
    video_delay_seconds: float
    motion_seconds: float
    speed: float
    acceleration: float = 0.0
    duration_seconds: float = 60.0
    initial_world_x: float = 0.42
    direction: int = 1


def simulate_closed_loop(case: ClosedLoopCase) -> dict[str, Any]:
    """Run a deterministic detector-stub PTZ loop with nonreplaceable moves."""
    dt = 0.05
    world_x = case.initial_world_x
    pan = 0.5
    zoom = 1.0
    target_world_width = 0.10
    action: dict[str, float | int] | None = None
    frames: list[tuple[float, float, float, float]] = []
    commands: list[dict[str, Any]] = []
    clipped = 0
    observed = 0
    center_errors: list[float] = []
    reached_at: float | None = None
    dropped_frames = {round(12.0 / dt), round(12.05 / dt)}
    controller = DemoTrackingController(
        target_width=0.33,
        target_height=0.33,
        center_deadband=0.07,
        maximum_age=1.2,
        prediction_horizon=case.response_seconds + case.video_delay_seconds,
    )
    previous: TrackingObservation | None = None
    for tick in range(round(case.duration_seconds / dt) + 1):
        t = tick * dt
        speed = case.speed + (case.acceleration * max(t - 20.0, 0.0) if 20 <= t < 22 else 0.0)
        world_x += case.direction * speed * dt
        if action is not None:
            start = float(action["start"])
            if t >= start:
                fraction = min(1.0, (t - start) / max(case.motion_seconds, dt))
                pan = float(action["old_pan"]) + fraction * (float(action["goal_pan"]) - float(action["old_pan"]))
                zoom = math.exp(
                    math.log(float(action["old_zoom"]))
                    + fraction * math.log(float(action["goal_zoom"]) / float(action["old_zoom"]))
                )
                if fraction >= 1.0:
                    action = None
        screen_x = 0.5 + (world_x - pan) * zoom
        screen_width = target_world_width * zoom
        visible = screen_x - screen_width / 2 >= 0 and screen_x + screen_width / 2 <= 1
        clipped += int(not visible)
        if visible:
            observed += 1
            frames.append((t, screen_x, screen_width, 0.16))
            if max(screen_width, 0.16) >= 0.33 and reached_at is None:
                reached_at = t
        center_errors.append(abs(screen_x - 0.5))
        delivered = None
        cutoff = t - case.video_delay_seconds
        while frames and frames[0][0] <= cutoff + 1e-9:
            delivered = frames.pop(0)
        if delivered is None or action is not None or tick in dropped_frames:
            continue
        stamp, observed_x, observed_width, observed_height = delivered
        observation = TrackingObservation(
            target_id="boat-1", x=observed_x - observed_width / 2,
            y=0.42, width=observed_width, height=observed_height,
            source="detector_stub", frame_id=round(stamp / dt), view_epoch=0,
            source_timestamp=t, received_at=t,
        )
        if previous is not None:
            d = max(t - previous.received_at, dt)
            observation = replace(
                observation,
                velocity_x=(observation.center[0] - previous.center[0]) / d,
            )
        previous = observation
        decision = controller.decide(observation, now=t, view_epoch=0)
        if decision is None:
            continue
        x, y, delta, reason = decision
        # Relative locate is an incremental correction.  Use a conservative
        # half-error step because the physical action cannot be replaced while
        # the video feedback still contains pre-action frames.
        correction_gain = 1.0 if delta > 0 else 0.5
        goal_pan = pan + (x - 0.5) / max(zoom, 1e-6) * correction_gain
        goal_zoom = max(1.0, zoom * (1.25 ** delta))
        action = {
            "start": t + case.response_seconds,
            "old_pan": pan, "goal_pan": goal_pan,
            "old_zoom": zoom, "goal_zoom": goal_zoom,
        }
        commands.append({"time": round(t, 3), "x": round(x, 4), "delta": delta, "reason": reason})
    return {
        "scenario": case.name,
        "response_seconds": case.response_seconds,
        "video_delay_seconds": case.video_delay_seconds,
        "motion_seconds": case.motion_seconds,
        "duration_seconds": case.duration_seconds,
        "clipped_seconds": round(clipped * dt, 3),
        "visible_fraction": round(observed / max(round(case.duration_seconds / dt) + 1, 1), 4),
        "maximum_center_error": round(max(center_errors or [0.0]), 4),
        "reached_scale_seconds": reached_at,
        "zoom_in_commands": sum(int(c["delta"] > 0) for c in commands),
        "zoom_out_commands": sum(int(c["delta"] < 0) for c in commands),
        "command_count": len(commands),
        "commands": commands,
        "quality_pass": bool(reached_at is not None and clipped == 0 and commands),
    }
