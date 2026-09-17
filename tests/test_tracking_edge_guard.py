import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.tracking_edge_guard import TrackingEdgeGuard
from rtsp_annotator.vessel_detection import VesselDetection
from rtsp_annotator.vessel_detection import VesselSnapshot
from rtsp_annotator.ptz_verification import (
    PtzVerificationCoordinator, PtzVerificationOptions, _TrackingObservation,
)


def boat(t, x=.5, width=.33, kind="detector_measurement"):
    return VesselDetection(
        1, NormalizedRect(x-width/2, .4, width, .15), .8, 8, 3,
        observation_kind=kind, position_updated_at=t, source="sidecar",
        frame_id=round(t*1000),
    )


class TrackingEdgeGuardTests(unittest.TestCase):
    def test_delayed_normal_cruise_matrix_has_no_escape_or_clipping(self):
        from scripts.evaluate_tracking_edge_guard import simulate

        for response in (.5, .75, 1.):
            for video in (.1, .3):
                for speed in (.01, .03):
                    for direction in (-1, 1):
                        with self.subTest(response=response, video=video,
                                          speed=speed, direction=direction):
                            result = simulate(response=response, video_delay=video,
                                              cruise=speed, burst=0, direction=direction)
                            self.assertEqual(result["zoom_out_commands"], 0)
                            self.assertEqual(result["clipped_seconds"], 0)
                            self.assertGreater(result["command_count"], 0)

    def test_guard_disables_blind_zoom_out_after_loss(self):
        camera = Mock()
        clock = Mock(return_value=0.)
        coordinator = PtzVerificationCoordinator(
            stream_id="test", options=PtzVerificationOptions(
                enabled=True, camera_id="test", continuous_tracking=True,
                evidence_validation_required=False, tracking_edge_guard_enabled=True,
                tracking_recovery_enabled=True, tracking_recovery_interval_seconds=.5,
                tracking_lost_timeout_seconds=1,
            ), snapshot_provider=lambda: VesselSnapshot(state="running"),
            repository=Mock(), camera_client=camera,
        )
        def advance(seconds):
            clock.return_value += seconds
        with patch("rtsp_annotator.ptz_verification.time.monotonic", clock), patch.object(
            coordinator._stop_event, "wait", side_effect=advance,
        ):
            result = coordinator._wait_for_tracking_detection(after_version=0, reference=boat(0))
        self.assertIsNone(result.detection)
        camera.locate.assert_not_called()

    def test_options_roundtrip_and_invalid_combinations(self):
        from rtsp_annotator.api import PtzVerificationRequest

        request = PtzVerificationRequest(
            enabled=True, camera_id="test", continuous_tracking=True,
            tracking_edge_guard_enabled=True, tracking_edge_response_seconds=.75,
            tracking_edge_cooldown_seconds=10,
        )
        options = request.to_options()
        self.assertEqual(PtzVerificationOptions.from_payload(options.to_payload()), options)
        for changes in ({"enabled": False}, {"continuous_tracking": False},
                        {"tracking_edge_response_seconds": float("nan")}):
            with self.assertRaises(ValueError):
                replace(options, **changes).validate()

    def test_coordinator_dispatches_guard_step_and_keeps_legacy_path_opt_in(self):
        camera = Mock()
        clock = Mock(return_value=0)
        coordinator = PtzVerificationCoordinator(
            stream_id="test",
            options=PtzVerificationOptions(
                enabled=True, camera_id="test", continuous_tracking=True,
                evidence_validation_required=False, tracking_edge_guard_enabled=True,
                tracking_center_deadband=.3, tracking_settle_seconds=0,
                tracking_max_duration_seconds=0,
            ),
            snapshot_provider=lambda: VesselSnapshot(), repository=Mock(),
            camera_client=camera,
        )
        positions = iter([.50, .502, .504, .534, .564, .594])
        version = 0

        def next_observation(**kwargs):
            nonlocal version
            x = next(positions, None)
            if x is None:
                return _TrackingObservation(None, version, "shutdown")
            version += 1
            clock.return_value = version*.1
            return _TrackingObservation(boat(clock.return_value, x), version)

        with patch("rtsp_annotator.ptz_verification.time.monotonic", clock), patch.object(
            coordinator, "_wait_for_tracking_detection", side_effect=next_observation,
        ), patch.object(coordinator, "_renew_lease"):
            coordinator._track_target(boat(0))
        self.assertTrue(camera.locate.called)
        self.assertEqual(camera.locate.call_args.args[2], -1)
        camera.home.assert_not_called()
        self.assertFalse(PtzVerificationOptions().tracking_edge_guard_enabled)

    def test_normal_centered_cruise_never_zooms_out_even_if_oversized(self):
        guard = TrackingEdgeGuard()
        for i in range(601):
            t = i/10
            result = guard.update(boat(t, .5 + .002*(i % 10), .45), now=t)
            self.assertGreaterEqual(result.zoom_delta, 0)

    def test_acceleration_needs_edge_risk_not_just_speed_change(self):
        guard = TrackingEdgeGuard()
        for i, x in enumerate([.40, .402, .404, .416, .428]):
            result = guard.update(boat(i*.1, x, .15), now=i*.1)
            self.assertGreaterEqual(result.zoom_delta, 0)

    def test_confirmed_outward_risk_requests_only_one_step(self):
        guard = TrackingEdgeGuard()
        results = [guard.update(boat(i*.1, x), now=i*.1)
                   for i, x in enumerate([.60, .602, .604, .634, .664, .694])]
        self.assertTrue(any(r.zoom_delta == -1 for r in results))
        self.assertTrue(all(r.zoom_delta >= -1 for r in results))

    def test_held_stale_and_duplicate_observations_cannot_trigger(self):
        guard = TrackingEdgeGuard()
        for i in range(10):
            result = guard.update(boat(0, .8, kind="held_display"), now=i/10)
            self.assertEqual(result.zoom_delta, 0)
        result = guard.update(boat(1, .8), now=3)
        self.assertEqual(result.reason, "unreliable_observation")
        guard.update(boat(4, .5), now=4)
        result = guard.update(boat(4, .8), now=4)
        self.assertEqual(result.reason, "duplicate_observation")

    def test_camera_action_invalidates_screen_velocity_and_delayed_frames(self):
        guard = TrackingEdgeGuard()
        guard.update(boat(0, .5), now=0)
        guard.update(boat(.1, .51), now=.1)
        guard.action_completed(1, zoom_delta=0)
        result = guard.update(boat(1.1, .85), now=1.1)
        self.assertEqual(result.reason, "camera_settling")
        result = guard.update(boat(2.5, .5), now=2.5)
        self.assertEqual(result.zoom_delta, 0)
        self.assertFalse(result.velocity_reliable)

    def test_shrink_cooldown_and_stable_recovery(self):
        guard = TrackingEdgeGuard(cooldown_seconds=5, stable_seconds=1)
        guard.action_completed(0, zoom_delta=-1)
        for i in range(15, 50):
            result = guard.update(boat(i/10, .5, .15), now=i/10)
            self.assertEqual(result.zoom_delta, 0)
        result = guard.update(boat(5, .5, .15), now=5)
        self.assertEqual(result.zoom_delta, 1)

    def test_successful_position_corrections_do_not_postpone_zoom_forever(self):
        guard = TrackingEdgeGuard(cooldown_seconds=1, stable_seconds=3,
                                  uncertainty_seconds=.1)
        guard.action_completed(0, zoom_delta=-1)
        result = None
        for second in range(1, 6):
            for offset in (.2, .3, .4):
                t = second+offset
                result = guard.update(boat(t, .5, .15), now=t)
            guard.action_completed(second+.5, zoom_delta=0)
        self.assertEqual(result.zoom_delta, 1)

    def test_identity_change_and_size_jump_reset_velocity(self):
        guard = TrackingEdgeGuard()
        guard.update(boat(0, .5), now=0)
        guard.update(boat(.1, .51), now=.1)
        result = guard.update(replace(boat(.2, .8), object_id=2), now=.2)
        self.assertFalse(result.velocity_reliable)
        self.assertEqual(result.zoom_delta, 0)


if __name__ == "__main__":
    unittest.main()
