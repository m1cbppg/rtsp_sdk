import unittest

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.ground_litter import (
    GroundLitterActor,
    GroundLitterCandidate,
    GroundLitterDetection,
    GroundLitterEvent,
    GroundLitterOptions,
    GroundLitterTracker,
)


ROI = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
BOX = NormalizedRect(0.40, 0.60, 0.06, 0.04)


def detection(rectangle=BOX, confidence=0.8):
    return GroundLitterDetection(rectangle, confidence, "plastic")


class GroundLitterTrackerTests(unittest.TestCase):
    def make_tracker(self, **kwargs):
        return GroundLitterTracker(GroundLitterOptions(ground_roi=ROI, **kwargs))

    def test_requires_persistence_and_multiple_hits(self):
        tracker = self.make_tracker(persistence_seconds=10, confirm_hits=3, lost_track_seconds=10)
        self.assertFalse(tracker.observe(0, [detection()]).events)
        self.assertFalse(tracker.observe(5, [detection()]).events)
        result = tracker.observe(10, [detection()])
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].evidence_hits, 3)

    def test_actor_overlap_is_suppressed(self):
        tracker = self.make_tracker(persistence_seconds=1, confirm_hits=1)
        actor = GroundLitterActor(NormalizedRect(0.39, 0.58, 0.10, 0.10), 4)
        result = tracker.observe(2, [detection()], [actor])
        self.assertFalse(result.candidates)
        self.assertFalse(result.events)

    def test_exclusion_zone_is_applied_to_bottom_center(self):
        tracker = self.make_tracker(
            exclude_zones=(((0.35, 0.55), (0.50, 0.55), (0.50, 0.75), (0.35, 0.75)),),
            persistence_seconds=1,
            confirm_hits=1,
        )
        self.assertFalse(tracker.observe(1, [detection()]).candidates)

    def test_motion_prevents_confirmation(self):
        tracker = self.make_tracker(persistence_seconds=2, confirm_hits=2, max_motion=0.01)
        tracker.observe(0, [detection(NormalizedRect(0.20, 0.60, 0.06, 0.04))])
        result = tracker.observe(2, [detection(NormalizedRect(0.22, 0.60, 0.06, 0.04))])
        self.assertFalse(result.events)
        self.assertTrue(any(not item.stationary for item in result.candidates))

    def test_lost_track_can_be_recreated_after_gap(self):
        tracker = self.make_tracker(persistence_seconds=1, confirm_hits=1, lost_track_seconds=2)
        tracker.observe(0, [detection()])
        first = tracker.observe(1, [detection()]).events[0]
        self.assertEqual(first.candidate_id, 1)
        tracker.observe(4, [])
        tracker.observe(5, [detection()])
        second = tracker.observe(6, [detection()]).events[0]
        self.assertEqual(second.candidate_id, 2)

    def test_timestamps_must_be_monotonic_per_camera(self):
        tracker = self.make_tracker()
        tracker.observe(2, [detection()])
        with self.assertRaises(ValueError):
            tracker.observe(1, [detection()])

    def test_small_bag_inside_person_box_is_suppressed(self):
        tracker = self.make_tracker()
        tiny = detection(NormalizedRect(.45, .65, .006, .009))
        actor = GroundLitterActor(NormalizedRect(.40, .30, .15, .50))
        self.assertFalse(tracker.observe(0, [tiny], [actor]).candidates)

    def test_same_timestamp_cannot_count_as_extra_hit(self):
        tracker = self.make_tracker()
        tracker.observe(0, [detection()])
        self.assertFalse(tracker.observe(0, [detection()]).events)
        self.assertEqual(tracker.observe(1, [detection()]).candidates[0].hits, 2)

    def test_persistent_item_does_not_periodically_reconfirm(self):
        tracker = self.make_tracker(persistence_seconds=1, confirm_hits=2,
                                    cooldown_seconds=0)
        tracker.observe(0, [detection()])
        self.assertEqual(len(tracker.observe(1, [detection()]).events), 1)
        for t in range(2, 610):
            self.assertFalse(tracker.observe(t, [detection()]).events)



if __name__ == "__main__":
    unittest.main()
