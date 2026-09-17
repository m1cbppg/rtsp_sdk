import unittest
from unittest.mock import Mock, patch

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.ptz_verification import (
    PtzVerificationCoordinator, PtzVerificationOptions,
)
from rtsp_annotator.vessel_detection import (
    VesselCandidate, VesselDetection, VesselDetectionOptions,
    VesselSnapshot, VesselTrackManager,
    VesselResultCache,
)


class TrackingObservationContractTests(unittest.TestCase):
    def test_cache_rejects_old_view_and_out_of_order_source_time(self):
        cache = VesselResultCache()
        current = VesselSnapshot(state="running", updated_at=20, view_generation=2)
        cache.store_snapshot(0, current)
        cache.store_snapshot(0, VesselSnapshot(
            state="running", updated_at=21, view_generation=1,
        ))
        self.assertEqual(cache.snapshot(0), current)
        cache.store_snapshot(0, VesselSnapshot(
            state="running", updated_at=19, view_generation=2,
        ))
        self.assertEqual(cache.snapshot(0), current)

    def coordinator(self, snapshot):
        return PtzVerificationCoordinator(
            stream_id="observation-test",
            options=PtzVerificationOptions(evidence_validation_required=False),
            snapshot_provider=lambda: snapshot,
            repository=Mock(), camera_client=Mock(),
        )

    def test_new_empty_primary_does_not_refresh_twenty_second_old_sidecar(self):
        box = VesselDetection(1, NormalizedRect(.4, .4, .1, .1), .8, 8, 3)
        coordinator = self.coordinator(VesselSnapshot(
            state="running", detections=(box,), result_version=1, updated_at=80,
        ))
        coordinator.publish_primary_detections((), updated_at=100)
        with patch("rtsp_annotator.ptz_verification.time.monotonic", return_value=100):
            self.assertEqual(coordinator._snapshot().detections, ())

    def test_static_held_track_preserves_last_position_time(self):
        tracker = VesselTrackManager(VesselDetectionOptions(minimum_hits=1))
        candidate = VesselCandidate(NormalizedRect(.4, .4, .1, .1), .8, 8)
        measured = tracker.update([candidate], timestamp=10, inference_ms=2)
        held = tracker.update([], timestamp=10.2, inference_ms=2)
        self.assertEqual(measured.detections[0].observation_kind, "detector_measurement")
        self.assertEqual(held.detections[0].observation_kind, "held_display")
        self.assertEqual(held.detections[0].position_updated_at, 10)

    def test_held_track_cannot_enter_control_snapshot(self):
        tracker = VesselTrackManager(VesselDetectionOptions(minimum_hits=1))
        tracker.update([VesselCandidate(NormalizedRect(.4, .4, .1, .1), .8, 8)],
                       timestamp=10, inference_ms=2)
        held = tracker.update([], timestamp=10.2, inference_ms=2)
        with patch("rtsp_annotator.ptz_verification.time.monotonic", return_value=10.2):
            self.assertEqual(self.coordinator(held)._snapshot().detections, ())

    def test_demo_snapshot_rejects_sidecar_from_previous_view_generation(self):
        box = VesselDetection(
            1, NormalizedRect(.4, .4, .1, .1), .8, 8, 3,
            observation_kind="detector_measurement",
            position_updated_at=10.0,
            source="sidecar",
            frame_id=1,
        )
        coordinator = PtzVerificationCoordinator(
            stream_id="observation-test",
            options=PtzVerificationOptions(
            enabled=True, camera_id="camera-01", tracking_profile="demo_continuous",
            continuous_tracking=True, tracking_max_duration_seconds=0,
            evidence_validation_required=False,
            ),
            snapshot_provider=lambda: VesselSnapshot(
                state="running", detections=(box,), result_version=1,
                updated_at=10.0, view_generation=0,
            ),
            repository=Mock(), camera_client=Mock(),
        )
        coordinator._view_generation = 1
        with patch("rtsp_annotator.ptz_verification.time.monotonic", return_value=10.1):
            self.assertEqual(coordinator._snapshot().detections, ())


if __name__ == "__main__":
    unittest.main()
