"""Ground Litter V3.3 dual-channel recall tests.

Covers the two-channel contract from the dual-recall architecture spec:
semantic-only and prior-only must each confirm and display on their own, and a
target seen by both must produce exactly one fused event. Everything here uses
deterministic fake observations, so no model, GPU or camera is required.
"""

from __future__ import annotations

import unittest

import numpy as np

from rtsp_annotator.ground_litter_detection import (
    GROUND_LITTER_MODES,
    GroundLitterDetectionOptions,
    GroundLitterZone,
    prior_crop_rect,
)
from rtsp_annotator.ground_litter_v33 import (
    EVIDENCE_PRIOR_ONLY,
    EVIDENCE_SEMANTIC_AND_PRIOR,
    EVIDENCE_SEMANTIC_ONLY,
    SOURCE_FUSED,
    SOURCE_PRIOR,
    SOURCE_SEMANTIC,
    PriorObservation,
    SemanticObservation,
    V33EventMemory,
    associate_observations,
    box_iou,
    crop_semantic_observations,
    observations_match,
)

BOX_A = (100.0, 100.0, 130.0, 130.0)
BOX_B = (500.0, 400.0, 540.0, 440.0)


def options(**overrides):
    values = dict(
        mode="hybrid_v33",
        enabled=False,
        analysis_fps=0.5,
        startup_suppress_seconds=0.0,
        profile_id="camera_test",
        model="litter/turhancan_yolov8m_seg_trash.pt",
        semantic_scan_interval_seconds=4.0,
        semantic_confirm_hits=2,
        semantic_hit_window=3,
        semantic_confirm_span_seconds=4.0,
        prior_confirm_hits=4,
        prior_hit_window=6,
        prior_confirm_span_seconds=6.0,
        fused_confirm_hits=2,
        fused_hit_window=4,
        fused_confirm_span_seconds=2.0,
        pending_expire_seconds=20.0,
        clear_confirm_seconds=6.0,
        semantic_clear_seconds=8.0,
        semantic_clear_min_misses=2,
        maximum_closed_events=100,
        actor_overlap_threshold=0.2,
    )
    values.update(overrides)
    return GroundLitterDetectionOptions(**values)


def semantic(box, timestamp, confidence=0.5, region_id="r1"):
    return SemanticObservation(
        box_xyxy=box, confidence=confidence, class_name="Plastic",
        region_id=region_id, source="full_roi", observed_at=timestamp,
    )


def prior(box, timestamp, score=1.0, region_id="r1"):
    return PriorObservation(
        box_xyxy=box, anomaly_score=score, region_id=region_id,
        support_pixels=40, observed_at=timestamp,
    )


def clean_arrays(shape=(400, 400)):
    """Empty support (clean ground) and a fully valid mask.

    The array must cover the test boxes: ``_anchor_observation`` returns
    ``(0.0, 0)`` for a box outside the frame, which would silently read as
    "ground not available" instead of "clean ground".
    """
    return np.zeros(shape, np.uint8), np.ones(shape, np.uint8)


def dirty_arrays(box, shape=(400, 400)):
    support = np.zeros(shape, np.uint8)
    valid = np.ones(shape, np.uint8)
    x1, y1, x2, y2 = (int(v) for v in box)
    support[max(0, y1):y2, max(0, x1):x2] = 255
    return support, valid


class ConfigTests(unittest.TestCase):
    def test_hybrid_mode_is_registered(self):
        self.assertIn("hybrid_v33", GROUND_LITTER_MODES)

    def test_hybrid_requires_model_and_profile(self):
        zone = GroundLitterZone(
            region_id="r1", name="z",
            polygon=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
        )
        base = dict(mode="hybrid_v33", enabled=True, zones=(zone,))
        with self.assertRaises(ValueError):
            GroundLitterDetectionOptions(**base, model="m.pt").validate()
        with self.assertRaises(ValueError):
            GroundLitterDetectionOptions(
                **base, profile_id="p", model=None
            ).validate()

    def test_hybrid_bounds_are_validated(self):
        base = dict(mode="hybrid_v33", enabled=False)
        for override in (
            {"semantic_confirm_hits": 5, "semantic_hit_window": 3},
            {"prior_confirm_hits": 9, "prior_hit_window": 6},
            {"fused_confirm_hits": 9, "fused_hit_window": 4},
            {"prior_crop_maximum": 9},
            {"prior_crop_expand_ratio": 0.5},
            {"prior_crop_imgsz": 100},
            {"semantic_scan_interval_seconds": -1.0},
            {"prior_suspend_expire_seconds": 0.0},
            {"semantic_clear_min_misses": 0},
        ):
            with self.assertRaises(ValueError, msg=override):
                options(**base, **override).validate()

    def test_payload_round_trip_keeps_hybrid_fields(self):
        original = options()
        payload = original.to_payload()
        restored = GroundLitterDetectionOptions.from_payload(payload)
        self.assertEqual(payload, restored.to_payload())
        self.assertEqual(restored.semantic_scan_interval_seconds, 4.0)
        self.assertEqual(restored.prior_confirm_hits, 4)


class GeometryTests(unittest.TestCase):
    def test_box_iou_identity_and_disjoint(self):
        self.assertAlmostEqual(box_iou(BOX_A, BOX_A), 1.0)
        self.assertEqual(box_iou(BOX_A, BOX_B), 0.0)

    def test_observations_match_by_iou_and_by_containment(self):
        self.assertTrue(observations_match(BOX_A, BOX_A))
        # Semantic centre inside the prior box expanded by 25 %.
        inner = (105.0, 105.0, 110.0, 110.0)
        self.assertTrue(observations_match(inner, BOX_A))
        self.assertFalse(observations_match(BOX_A, BOX_B))

    def test_prior_crop_rect_geometry(self):
        # 15x8 target: 4x expansion is below the 160 px floor.
        self.assertEqual(
            prior_crop_rect([803, 565, 818, 573], expand_ratio=4.0,
                            maximum_source_px=480, width=2560, height=1440),
            (730, 489, 890, 649),
        )
        # Large target: capped by maximum_source_px.
        rect = prior_crop_rect([837, 597, 925, 668], expand_ratio=4.0,
                               maximum_source_px=352, width=2560, height=1440)
        self.assertEqual(rect[2] - rect[0], 352)
        # Corner clamping keeps the window inside the frame.
        self.assertEqual(
            prior_crop_rect([2, 2, 10, 10], expand_ratio=4.0,
                            maximum_source_px=480, width=100, height=100),
            (0, 0, 100, 100),
        )


class AssociationTests(unittest.TestCase):
    def test_same_target_fuses_into_one_observation(self):
        fused = associate_observations(
            [semantic(BOX_A, 10.0)], [prior(BOX_A, 10.0)], tolerance_seconds=4.0,
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].evidence_kind, EVIDENCE_SEMANTIC_AND_PRIOR)
        self.assertEqual(fused[0].display_box, BOX_A)

    def test_unrelated_boxes_stay_separate(self):
        fused = associate_observations(
            [semantic(BOX_A, 10.0)], [prior(BOX_B, 10.0)], tolerance_seconds=4.0,
        )
        kinds = sorted(item.evidence_kind for item in fused)
        self.assertEqual(kinds, [EVIDENCE_PRIOR_ONLY, EVIDENCE_SEMANTIC_ONLY])

    def test_time_gap_prevents_association(self):
        fused = associate_observations(
            [semantic(BOX_A, 10.0)], [prior(BOX_A, 30.0)], tolerance_seconds=4.0,
        )
        self.assertEqual(len(fused), 2)

    def test_crop_extra_detection_is_not_promoted(self):
        priors = [prior(BOX_A, 10.0)]
        rows = [[
            {"box": [105.0, 105.0, 125.0, 125.0], "confidence": 0.6, "class_id": 3},
            {"box": [900.0, 900.0, 950.0, 950.0], "confidence": 0.9, "class_id": 3},
        ]]
        matched, raw, unmatched = crop_semantic_observations(
            rows, priors, timestamp=10.0, class_names={3: "Plastic"},
        )
        self.assertEqual(raw, 2)
        self.assertEqual(unmatched, 1)
        self.assertEqual(len(matched), 1)
        self.assertTrue(observations_match(matched[0].box_xyxy, BOX_A))


class SemanticChannelTests(unittest.TestCase):
    def test_semantic_only_confirms_and_displays_without_prior(self):
        memory = V33EventMemory(options(), pixel_scale=1.0)
        for timestamp in (0.0, 2.0):
            memory.update(timestamp=timestamp, semantic=[], prior=[],
                          prior_available=True, environment_state="NORMAL")
        first = memory.update(timestamp=4.0, semantic=[semantic(BOX_A, 4.0)],
                              prior=[], prior_available=True,
                              environment_state="NORMAL")
        self.assertEqual(first.confirmed_events, 0)
        self.assertEqual(len(first.detections), 0)
        second = memory.update(timestamp=8.0, semantic=[semantic(BOX_A, 8.0)],
                               prior=[], prior_available=True,
                               environment_state="NORMAL")
        self.assertEqual(second.confirmed_events, 1)
        self.assertEqual(len(second.detections), 1)
        self.assertEqual(second.detections[0].source, SOURCE_SEMANTIC)
        self.assertEqual(second.semantic_only_confirmed, 1)

    def test_single_frame_semantic_never_confirms(self):
        memory = V33EventMemory(options(), pixel_scale=1.0)
        result = memory.update(timestamp=4.0, semantic=[semantic(BOX_A, 4.0)],
                               prior=[], prior_available=True,
                               environment_state="NORMAL")
        self.assertEqual(result.confirmed_events, 0)
        self.assertEqual(len(result.detections), 0)
        self.assertEqual(memory.active[0].state, "PENDING")

    def test_semantic_survives_a_tick_without_a_scan(self):
        memory = V33EventMemory(options(), pixel_scale=1.0)
        for timestamp in (0.0, 4.0):
            memory.update(timestamp=timestamp, semantic=[], prior=[],
                          prior_available=True, environment_state="NORMAL")
        memory.update(timestamp=5.0, semantic=[semantic(BOX_A, 5.0)], prior=[],
                      prior_available=True, environment_state="NORMAL")
        confirmed = memory.update(timestamp=9.0, semantic=[semantic(BOX_A, 9.0)],
                                  prior=[], prior_available=True,
                                  environment_state="NORMAL")
        self.assertEqual(confirmed.confirmed_events, 1)
        skipped = memory.update(timestamp=11.0, semantic=None, prior=[],
                                prior_available=True, environment_state="NORMAL")
        self.assertEqual(len(skipped.detections), 1)


class PriorChannelTests(unittest.TestCase):
    def test_prior_only_confirms_and_displays_without_semantic(self):
        support, valid = clean_arrays()
        memory = V33EventMemory(options(), pixel_scale=1.0)
        progress = []
        for step in range(8):
            timestamp = float(step * 2)
            candidate = [prior(BOX_A, timestamp)] if timestamp >= 6.0 else []
            result = memory.update(
                timestamp=timestamp, semantic=[], prior=candidate,
                prior_available=True, environment_state="NORMAL",
                support=support, valid=valid,
            )
            progress.append((timestamp, result.confirmed_events,
                             len(result.detections)))
        self.assertEqual(progress[0][1], 0)
        self.assertEqual(progress[3][1], 0)  # one hit
        self.assertEqual(progress[5][1], 0)  # three hits are still not enough
        self.assertEqual(progress[6][1], 1)  # four hits spanning six seconds
        self.assertEqual(progress[6][2], 1)
        final = memory.update(
            timestamp=16.0, semantic=[], prior=[prior(BOX_A, 16.0)],
            prior_available=True, environment_state="NORMAL",
            support=support, valid=valid,
        )
        self.assertEqual(final.detections[0].source, SOURCE_PRIOR)
        self.assertEqual(final.prior_only_confirmed, 1)

    def test_independent_confirmation_cadences_differ(self):
        """prior needs 4/6 while semantic needs 2/3: neither shares a rule."""
        support, valid = clean_arrays()
        memory = V33EventMemory(options(), pixel_scale=1.0)
        # Two semantic scans 4 s apart confirm; the prior channel has only two
        # hits at that point and must still be pending.
        memory.update(timestamp=0.0, semantic=[], prior=[],
                      prior_available=True, environment_state="NORMAL",
                      support=support, valid=valid)
        memory.update(timestamp=4.0, semantic=[semantic(BOX_A, 4.0)],
                      prior=[prior(BOX_A, 4.0)], prior_available=True,
                      environment_state="NORMAL", support=support, valid=valid)
        result = memory.update(timestamp=8.0, semantic=[semantic(BOX_A, 8.0)],
                               prior=[prior(BOX_A, 8.0)], prior_available=True,
                               environment_state="NORMAL",
                               support=support, valid=valid)
        self.assertEqual(result.confirmed_events, 1)
        self.assertEqual(result.fused_confirmed, 1)


class FusionTests(unittest.TestCase):
    def test_both_channels_on_one_target_emit_one_fused_event(self):
        memory = V33EventMemory(options(), pixel_scale=1.0)
        memory.update(timestamp=0.0, semantic=[], prior=[],
                      prior_available=True, environment_state="NORMAL")
        first = memory.update(timestamp=4.0, semantic=[semantic(BOX_A, 4.0)],
                              prior=[prior(BOX_A, 4.0)], prior_available=True,
                              environment_state="NORMAL")
        self.assertEqual(first.active_events, 1)
        self.assertEqual(first.fused_active, 1)
        second = memory.update(timestamp=6.0, semantic=[semantic(BOX_A, 6.0)],
                               prior=[prior(BOX_A, 6.0)], prior_available=True,
                               environment_state="NORMAL")
        self.assertEqual(second.active_events, 1)
        self.assertEqual(len(second.detections), 1)
        self.assertEqual(second.detections[0].source, SOURCE_FUSED)
        self.assertEqual(second.fused_confirmed, 1)

    def test_different_positions_stay_independent_events(self):
        memory = V33EventMemory(options(), pixel_scale=1.0)
        result = None
        for step in range(5):
            timestamp = float(10 + step * 2)
            result = memory.update(
                timestamp=timestamp,
                semantic=[semantic(BOX_A, timestamp)],
                prior=[prior(BOX_B, timestamp)],
                prior_available=True, environment_state="NORMAL",
            )
        self.assertEqual(result.active_events, 2)
        self.assertEqual(result.semantic_only_active, 1)
        self.assertEqual(result.prior_only_active, 1)
        self.assertEqual(len(result.detections), 2)
        self.assertEqual(
            sorted(item.source for item in result.detections),
            [SOURCE_PRIOR, SOURCE_SEMANTIC],
        )

    def test_evidence_upgrade_keeps_event_id(self):
        memory = V33EventMemory(options(), pixel_scale=1.0)
        memory.update(timestamp=0.0, semantic=[], prior=[],
                      prior_available=True, environment_state="NORMAL")
        memory.update(timestamp=4.0, semantic=[], prior=[prior(BOX_A, 4.0)],
                      prior_available=True, environment_state="NORMAL")
        memory.update(timestamp=6.0, semantic=[], prior=[prior(BOX_A, 6.0)],
                      prior_available=True, environment_state="NORMAL")
        before = memory.active[0].event_id
        self.assertEqual(memory.active[0].evidence_kind, EVIDENCE_PRIOR_ONLY)
        memory.update(timestamp=8.0, semantic=[semantic(BOX_A, 8.0)],
                      prior=[prior(BOX_A, 8.0)], prior_available=True,
                      environment_state="NORMAL")
        after = memory.active[0].event_id
        self.assertEqual(before, after)
        self.assertEqual(
            memory.active[0].evidence_kind, EVIDENCE_SEMANTIC_AND_PRIOR
        )

    def test_late_meeting_requires_two_timestamps_and_keeps_earlier_id(self):
        memory = V33EventMemory(options(), pixel_scale=1.0)
        memory.update(timestamp=0.0, semantic=[], prior=[],
                      prior_available=True, environment_state="NORMAL")
        # Two pending events 50 px apart: beyond the 30 px match distance.
        memory.update(timestamp=2.0, semantic=[semantic((100.0, 100.0, 120.0, 120.0), 2.0)],
                      prior=[], prior_available=True, environment_state="NORMAL")
        memory.update(timestamp=4.0, semantic=None,
                      prior=[prior((150.0, 100.0, 170.0, 120.0), 4.0)],
                      prior_available=True, environment_state="NORMAL")
        self.assertEqual(memory.active_events if hasattr(memory, "active_events")
                         else len(memory.active), 2)
        earlier = min(event.event_id for event in memory.active)
        # One bridge can be a wide detector box and must not merge two targets.
        memory.update(timestamp=6.0,
                      semantic=[semantic((125.0, 100.0, 145.0, 120.0), 6.0)],
                      prior=[], prior_available=True, environment_state="NORMAL")
        self.assertEqual(len(memory.active), 2)
        # A second distinct analysis timestamp proves the association persists.
        memory.update(timestamp=8.0,
                      semantic=[semantic((125.0, 100.0, 145.0, 120.0), 8.0)],
                      prior=[], prior_available=True, environment_state="NORMAL")
        open_events = memory.active
        self.assertEqual(len(open_events), 1)
        self.assertEqual(open_events[0].event_id, earlier)
        self.assertGreaterEqual(memory._cross_source_merges, 1)

    def test_maximum_boxes_bounds_display(self):
        memory = V33EventMemory(options(maximum_boxes=1), pixel_scale=1.0)
        result = None
        for step in range(5):
            timestamp = float(10 + step * 2)
            result = memory.update(
                timestamp=timestamp,
                semantic=[semantic(BOX_A, timestamp, confidence=0.9),
                          semantic(BOX_B, timestamp, confidence=0.8)],
                prior=[], prior_available=True, environment_state="NORMAL",
            )
        self.assertEqual(result.active_events, 2)
        self.assertEqual(len(result.detections), 1)

    def test_maximum_boxes_rotates_between_semantic_and_prior_sources(self):
        support, valid = dirty_arrays((100, 100, 130, 130))
        memory = V33EventMemory(options(maximum_boxes=1), pixel_scale=1.0)
        observed_sources = []
        for step in range(8):
            timestamp = float(step * 2)
            result = memory.update(
                timestamp=timestamp,
                semantic=(
                    [semantic(BOX_B, timestamp, confidence=0.1)]
                    if step % 2 == 0 else None
                ),
                prior=[prior(BOX_A, timestamp)],
                prior_available=True,
                environment_state="NORMAL",
                support=support,
                valid=valid,
                semantic_scan=step % 2 == 0,
            )
            if result.detections:
                observed_sources.append(result.detections[0].source)
        self.assertIn(SOURCE_SEMANTIC, observed_sources)
        self.assertIn(SOURCE_PRIOR, observed_sources)

    def test_one_bridge_does_not_merge_two_confirmed_neighbours(self):
        memory = V33EventMemory(options(), pixel_scale=1.0)
        left = (100.0, 100.0, 120.0, 120.0)
        right = (150.0, 100.0, 170.0, 120.0)
        bridge = (125.0, 100.0, 145.0, 120.0)
        for timestamp in (0.0, 4.0):
            memory.update(
                timestamp=timestamp,
                semantic=[semantic(left, timestamp), semantic(right, timestamp)],
                prior=[], prior_available=True, environment_state="NORMAL",
            )
        self.assertEqual(len(memory.active), 2)
        self.assertTrue(all(event.confirmed_at is not None for event in memory.active))
        memory.update(
            timestamp=8.0, semantic=[semantic(bridge, 8.0)], prior=[],
            prior_available=True, environment_state="NORMAL",
        )
        self.assertEqual(len(memory.active), 2)


class EnvironmentTests(unittest.TestCase):
    def test_prior_pause_does_not_stop_semantic(self):
        memory = V33EventMemory(options(), pixel_scale=1.0)
        memory.update(timestamp=0.0, semantic=[semantic(BOX_A, 0.0)], prior=[],
                      prior_available=False, environment_state="GLOBAL_LIGHT_CHANGE")
        result = memory.update(
            timestamp=4.0, semantic=[semantic(BOX_A, 4.0)], prior=[],
            prior_available=False, environment_state="GLOBAL_LIGHT_CHANGE",
        )
        self.assertEqual(result.prior_available, False)
        self.assertEqual(result.semantic_only_confirmed, 1)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.state, "running")

    def test_prior_only_is_hidden_while_prior_is_paused(self):
        support, valid = clean_arrays()
        memory = V33EventMemory(options(), pixel_scale=1.0)
        for step in range(6):
            timestamp = float(step * 2)
            memory.update(
                timestamp=timestamp, semantic=[], prior=[prior(BOX_A, timestamp)],
                prior_available=True, environment_state="NORMAL",
                support=support, valid=valid,
            )
        self.assertEqual(len(memory._project(10.0, prior_available=True)), 1)
        self.assertEqual(len(memory._project(10.0, prior_available=False)), 0)

    def test_environment_pause_never_clears_the_event(self):
        support, valid = clean_arrays()
        memory = V33EventMemory(options(), pixel_scale=1.0)
        for step in range(6):
            timestamp = float(step * 2)
            memory.update(
                timestamp=timestamp, semantic=[], prior=[prior(BOX_A, timestamp)],
                prior_available=True, environment_state="NORMAL",
                support=support, valid=valid,
            )
        paused = memory.update(
            timestamp=12.0, semantic=[], prior=[], prior_available=False,
            environment_state="GLOBAL_LIGHT_CHANGE",
            support=support, valid=valid,
        )
        self.assertEqual(paused.confirmed_events, 1)
        self.assertIsNone(memory.active[0].closed_at)

    def test_actor_occlusion_hides_and_pauses_timers(self):
        support, valid = dirty_arrays((100, 100, 130, 130))
        memory = V33EventMemory(options(), pixel_scale=1.0)
        for step in range(6):
            timestamp = float(step * 2)
            memory.update(
                timestamp=timestamp, semantic=[], prior=[prior(BOX_A, timestamp)],
                prior_available=True, environment_state="NORMAL",
                support=support, valid=valid,
            )
        self.assertEqual(len(memory._project(10.0, prior_available=True)), 1)
        occluded = memory.update(
            timestamp=12.0, semantic=[], prior=[], prior_available=True,
            environment_state="NORMAL", support=support, valid=valid,
            actors=[(100.0, 100.0, 130.0, 130.0)],
        )
        self.assertEqual(len(occluded.detections), 0)
        self.assertEqual(memory.active[0].state, "OCCLUDED")
        self.assertEqual(memory.active[0].clean_seconds, 0.0)

    def test_prior_suspend_timeout_closes_without_calling_it_cleaned(self):
        memory = V33EventMemory(options(prior_suspend_expire_seconds=10.0),
                                pixel_scale=1.0)
        for step in range(6):
            timestamp = float(step * 2)
            memory.update(
                timestamp=timestamp, semantic=[], prior=[prior(BOX_A, timestamp)],
                prior_available=True, environment_state="NORMAL",
            )
        closed = None
        for timestamp in (12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0):
            closed = memory.update(
                timestamp=timestamp, semantic=[], prior=[],
                prior_available=False, environment_state="ENVIRONMENT_CHANGE",
            )
        self.assertEqual(closed.confirmed_events, 0)
        self.assertEqual(memory.events[-1].closed_reason,
                         "profile_unavailable_timeout")


class LifecycleTests(unittest.TestCase):
    def _confirmed_prior_event(self, memory, support, valid, start=0.0):
        for step in range(6):
            timestamp = start + step * 2
            memory.update(
                timestamp=timestamp, semantic=[], prior=[prior(BOX_A, timestamp)],
                prior_available=True, environment_state="NORMAL",
                support=support, valid=valid,
            )

    def test_prior_only_clears_on_clean_reference_evidence(self):
        support, valid = dirty_arrays((100, 100, 130, 130))
        memory = V33EventMemory(options(), pixel_scale=1.0)
        self._confirmed_prior_event(memory, support, valid)
        self.assertEqual(len(memory.active), 1)
        clean_support, clean_valid = clean_arrays()
        closed = None
        for step in range(4):
            timestamp = 12.0 + step * 2
            closed = memory.update(
                timestamp=timestamp, semantic=[], prior=[],
                prior_available=True, environment_state="NORMAL",
                support=clean_support, valid=clean_valid,
            )
        self.assertEqual(len(memory.active), 0)
        self.assertEqual(memory.events[-1].closed_reason, "clean_confirmed")
        self.assertEqual(closed.cleared_events, 1)

    def test_semantic_only_clears_on_absence_not_a_single_miss(self):
        memory = V33EventMemory(options(), pixel_scale=1.0)
        support, valid = clean_arrays()
        for step in range(3):
            timestamp = float(step * 4)
            memory.update(timestamp=timestamp, semantic=[semantic(BOX_A, timestamp)],
                          prior=[], prior_available=True,
                          environment_state="NORMAL",
                          support=support, valid=valid)
        self.assertEqual(len(memory.active), 1)
        # One missed scan is not proof: still open.
        memory.update(timestamp=12.0, semantic=[], prior=[], prior_available=True,
                      environment_state="NORMAL", support=support, valid=valid)
        self.assertEqual(len(memory.active), 1)
        for timestamp in (16.0, 20.0, 24.0):
            memory.update(timestamp=timestamp, semantic=[], prior=[],
                          prior_available=True, environment_state="NORMAL",
                          support=support, valid=valid)
        self.assertEqual(len(memory.active), 0)
        self.assertEqual(memory.events[-1].closed_reason,
                         "semantic_absent_confirmed")

    def test_reappearance_after_clear_gets_a_new_event_id(self):
        support, valid = dirty_arrays((100, 100, 130, 130))
        memory = V33EventMemory(options(), pixel_scale=1.0)
        self._confirmed_prior_event(memory, support, valid)
        first_id = memory.active[0].event_id
        clean_support, clean_valid = clean_arrays()
        for step in range(4):
            memory.update(
                timestamp=12.0 + step * 2, semantic=[], prior=[],
                prior_available=True, environment_state="NORMAL",
                support=clean_support, valid=clean_valid,
            )
        self.assertEqual(len(memory.active), 0)
        for step in range(6):
            timestamp = 24.0 + step * 2
            memory.update(
                timestamp=timestamp, semantic=[], prior=[prior(BOX_A, timestamp)],
                prior_available=True, environment_state="NORMAL",
                support=support, valid=valid,
            )
        self.assertEqual(len(memory.active), 1)
        self.assertNotEqual(memory.active[0].event_id, first_id)

    def test_rewound_or_repeated_timestamp_does_not_double_count(self):
        memory = V33EventMemory(options(), pixel_scale=1.0)
        support, valid = clean_arrays()
        seen = []
        for step in range(8):
            timestamp = float(step * 2)
            result = memory.update(
                timestamp=timestamp, semantic=[], prior=[prior(BOX_A, timestamp)],
                prior_available=True, environment_state="NORMAL",
                support=support, valid=valid,
            )
            seen.append(result.confirmed_events)
        hits = memory.active[0].prior_hits
        memory.update(timestamp=6.0, semantic=[], prior=[prior(BOX_A, 6.0)],
                      prior_available=True, environment_state="NORMAL",
                      support=support, valid=valid)
        self.assertEqual(memory.active[0].prior_hits, hits)

    def test_startup_suppression_blocks_everything_then_clears_memory(self):
        memory = V33EventMemory(options(startup_suppress_seconds=15.0),
                                pixel_scale=1.0)
        warm = memory.update(timestamp=0.0, semantic=[semantic(BOX_A, 0.0)],
                             prior=[prior(BOX_A, 0.0)], prior_available=True,
                             environment_state="NORMAL")
        self.assertEqual(warm.state, "warming_up")
        self.assertEqual(len(warm.detections), 0)
        self.assertEqual(len(memory.events), 0)
        for step in range(8):
            timestamp = 16.0 + step * 2
            result = memory.update(
                timestamp=timestamp, semantic=[semantic(BOX_A, timestamp)],
                prior=[prior(BOX_A, timestamp)], prior_available=True,
                environment_state="NORMAL",
            )
        self.assertGreaterEqual(result.confirmed_events, 1)

    def test_semantic_absence_survives_ticks_without_a_scan(self) -> None:
        """Absence must pause, not reset, on ticks where no full scan ran.

        Resetting the run there is exactly what kept V3.2's ``cleared_events``
        pinned at zero, so this is a regression test for that failure mode.
        """
        memory = V33EventMemory(options(), pixel_scale=1.0)
        support, valid = clean_arrays()
        for step in range(3):
            timestamp = float(step * 4)
            memory.update(
                timestamp=timestamp, semantic=[semantic(BOX_A, timestamp)],
                prior=[], prior_available=True, environment_state="NORMAL",
                support=support, valid=valid, semantic_scan=True,
            )
        self.assertIsNotNone(memory.active[0].confirmed_at)
        for step in range(4):
            timestamp = 12.0 + step * 4
            memory.update(
                timestamp=timestamp, semantic=[], prior=[],
                prior_available=True, environment_state="NORMAL",
                support=support, valid=valid, semantic_scan=True,
            )
            memory.update(  # interleaved non-scan tick
                timestamp=timestamp + 2.0, semantic=None, prior=[],
                prior_available=True, environment_state="NORMAL",
                support=support, valid=valid, semantic_scan=False,
            )
        self.assertEqual(len(memory.active), 0)
        self.assertEqual(
            memory.events[-1].closed_reason, "semantic_absent_confirmed"
        )

    def test_clean_run_is_paused_by_environment_change_not_reset(self) -> None:
        support, valid = dirty_arrays((100, 100, 130, 130))
        memory = V33EventMemory(options(), pixel_scale=1.0)
        for step in range(6):
            timestamp = float(step * 2)
            memory.update(
                timestamp=timestamp, semantic=[], prior=[prior(BOX_A, timestamp)],
                prior_available=True, environment_state="NORMAL",
                support=support, valid=valid,
            )
        clean_support, clean_valid = clean_arrays()
        for step in range(2):
            timestamp = 12.0 + step * 2
            memory.update(
                timestamp=timestamp, semantic=[], prior=[],
                prior_available=True, environment_state="NORMAL",
                support=clean_support, valid=clean_valid,
            )
        accumulated = memory.active[0].clean_seconds
        self.assertGreater(accumulated, 0.0)
        memory.update(
            timestamp=16.0, semantic=[], prior=[], prior_available=False,
            environment_state="GLOBAL_LIGHT_CHANGE",
            support=clean_support, valid=clean_valid,
        )
        self.assertEqual(memory.active[0].clean_seconds, accumulated)

    def test_closed_event_memory_is_bounded(self):
        memory = V33EventMemory(options(maximum_closed_events=2), pixel_scale=1.0)
        support, valid = clean_arrays()
        for round_index in range(4):
            base = round_index * 100.0
            box = (100.0 + round_index * 200.0, 100.0,
                   130.0 + round_index * 200.0, 130.0)
            for step in range(2):
                memory.update(
                    timestamp=base + step * 20.0,
                    semantic=[semantic(box, base + step * 20.0)],
                    prior=[], prior_available=True, environment_state="NORMAL",
                    support=support, valid=valid,
                )
            memory.update(
                timestamp=base + 50.0, semantic=[], prior=[], prior_available=True,
                environment_state="NORMAL", support=support, valid=valid,
            )
            memory.update(
                timestamp=base + 60.0, semantic=[], prior=[], prior_available=True,
                environment_state="NORMAL", support=support, valid=valid,
            )
            memory.update(
                timestamp=base + 70.0, semantic=[], prior=[], prior_available=True,
                environment_state="NORMAL", support=support, valid=valid,
            )
        self.assertLessEqual(
            len([e for e in memory.events if not e.is_open]), 2
        )


if __name__ == "__main__":
    unittest.main()
