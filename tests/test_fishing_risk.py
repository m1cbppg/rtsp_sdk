from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.fishing_risk import (
    FishingRiskEngine,
    FishingRiskOptions,
    FishingRiskRuleOptions,
    FishingRiskScheduleOptions,
    FishingRiskZoneOptions,
)
from rtsp_annotator.vessel_detection import VesselDetection


ZONE = FishingRiskZoneOptions(
    "protected_water",
    ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
)
NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone(timedelta(hours=8)))


def vessel(
    x: float,
    y: float = 0.5,
    *,
    object_id: int = 7,
) -> VesselDetection:
    return VesselDetection(
        object_id=object_id,
        rectangle=NormalizedRect(x, y, 0.04, 0.03),
        confidence=0.8,
        class_id=8,
        hits=3,
    )


class FishingRiskTests(unittest.TestCase):
    def test_disabled_engine_is_a_strict_noop(self) -> None:
        engine = FishingRiskEngine(
            stream_id="stream-1",
            options=FishingRiskOptions(enabled=False),
        )

        result = engine.observe(
            timestamp=0,
            observed_at=NOW,
            detections=[vessel(0.5)],
        )

        self.assertEqual(result.snapshot.state, "disabled")
        self.assertEqual(result.snapshot.suspects, ())
        self.assertEqual(result.events, [])

    def test_presence_and_compact_loitering_emit_review_event(self) -> None:
        emitted = []
        options = FishingRiskOptions(
            enabled=True,
            zones=(ZONE,),
            rules=FishingRiskRuleOptions(
                minimum_presence_seconds=10,
                loitering_seconds=20,
                startup_grace_seconds=0,
            ),
        )
        engine = FishingRiskEngine(
            stream_id="stream-1",
            options=options,
            on_event=emitted.append,
        )

        result = None
        for second in range(0, 25, 5):
            result = engine.observe(
                timestamp=float(second),
                observed_at=NOW + timedelta(seconds=second),
                detections=[vessel(0.5 + second * 0.0001)],
            )

        assert result is not None
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.snapshot.maximum_score, 60)
        self.assertEqual(len(emitted), 1)
        event = result.events[0]
        self.assertEqual(event.event_type, "suspected_illegal_fishing")
        self.assertFalse(event.metadata["legal_conclusion"])
        self.assertTrue(event.metadata["review_required"])
        self.assertEqual(
            event.metadata["reasons"],
            ["restricted_period_presence", "loitering"],
        )

    def test_schedule_outside_window_suppresses_all_risk(self) -> None:
        options = FishingRiskOptions(
            enabled=True,
            zones=(ZONE,),
            schedules=(
                FishingRiskScheduleOptions(
                    "closure",
                    NOW + timedelta(days=1),
                    NOW + timedelta(days=2),
                ),
            ),
            rules=FishingRiskRuleOptions(
                minimum_presence_seconds=1,
                loitering_seconds=5,
                startup_grace_seconds=0,
            ),
        )
        engine = FishingRiskEngine(stream_id="stream-1", options=options)

        for second in range(10):
            result = engine.observe(
                timestamp=float(second),
                observed_at=NOW + timedelta(seconds=second),
                detections=[vessel(0.5)],
            )

        self.assertEqual(result.snapshot.suspects, ())
        self.assertEqual(result.events, [])

    def test_preexisting_stationary_vessel_is_ignored_until_it_moves(self) -> None:
        options = FishingRiskOptions(
            enabled=True,
            zones=(ZONE,),
            rules=FishingRiskRuleOptions(
                minimum_presence_seconds=5,
                loitering_seconds=10,
                startup_grace_seconds=60,
                preexisting_activation_box_lengths=2,
            ),
        )
        engine = FishingRiskEngine(stream_id="stream-1", options=options)

        for second in range(0, 70, 5):
            result = engine.observe(
                timestamp=float(second),
                observed_at=NOW + timedelta(seconds=second),
                detections=[vessel(0.5)],
            )
        self.assertEqual(result.snapshot.suspects, ())

        engine.observe(
            timestamp=75,
            observed_at=NOW + timedelta(seconds=75),
            detections=[vessel(0.65)],
        )
        events = []
        for second in (80, 85, 90):
            result = engine.observe(
                timestamp=float(second),
                observed_at=NOW + timedelta(seconds=second),
                detections=[vessel(0.65)],
            )
            events.extend(result.events)

        self.assertEqual(len(events), 1)
        self.assertEqual(result.snapshot.maximum_score, 60)

    def test_detector_id_change_does_not_reset_long_term_risk_track(self) -> None:
        options = FishingRiskOptions(
            enabled=True,
            zones=(ZONE,),
            rules=FishingRiskRuleOptions(
                minimum_presence_seconds=5,
                loitering_seconds=10,
                startup_grace_seconds=0,
            ),
        )
        engine = FishingRiskEngine(stream_id="stream-1", options=options)

        result = None
        for second in range(0, 15, 5):
            result = engine.observe(
                timestamp=float(second),
                observed_at=NOW + timedelta(seconds=second),
                detections=[
                    vessel(
                        0.5 + second * 0.0001,
                        object_id=7 if second == 0 else 99,
                    )
                ],
            )

        assert result is not None
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].actor_track_id, 1)

    def test_repeated_direction_reversal_adds_risk_reason(self) -> None:
        options = FishingRiskOptions(
            enabled=True,
            zones=(ZONE,),
            rules=FishingRiskRuleOptions(
                minimum_presence_seconds=1,
                loitering_seconds=300,
                reversal_window_seconds=60,
                minimum_reversals=2,
                minimum_motion_box_lengths=0.1,
                minimum_reversal_interval_seconds=1,
                startup_grace_seconds=0,
            ),
        )
        engine = FishingRiskEngine(stream_id="stream-1", options=options)

        result = None
        events = []
        for second, x in enumerate((0.2, 0.3, 0.2, 0.3, 0.2)):
            result = engine.observe(
                timestamp=float(second * 2),
                observed_at=NOW + timedelta(seconds=second * 2),
                detections=[vessel(x)],
            )
            events.extend(result.events)

        assert result is not None
        self.assertEqual(len(events), 1)
        self.assertIn(
            "direction_reversal",
            events[0].metadata["reasons"],
        )

    def test_payload_round_trip_preserves_schedule_and_rules(self) -> None:
        original = FishingRiskOptions(
            enabled=True,
            zones=(ZONE,),
            schedules=(
                FishingRiskScheduleOptions(
                    "closure",
                    NOW,
                    NOW + timedelta(days=1),
                ),
            ),
            rules=FishingRiskRuleOptions(minimum_presence_seconds=45),
        )

        restored = FishingRiskOptions.from_payload(original.to_payload())

        self.assertEqual(restored, original)


if __name__ == "__main__":
    unittest.main()
