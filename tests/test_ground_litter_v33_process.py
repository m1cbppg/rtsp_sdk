"""Integration tests for the hybrid_v33 side-process scheduler and isolation.

These exercise ``_analyse_hybrid`` — the shipped tick function — with a fake
detector and a stubbed prior analysis, so the *scheduling* contract and the
cross-branch error isolation are covered deterministically. Real model batching
is proven separately by ``scripts/smoke_ground_litter_v33_real_model.py``.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rtsp_annotator.ground_litter_detection import (
    GroundLitterCandidate,
    GroundLitterDetectionOptions,
    GroundLitterZone,
)
from rtsp_annotator.ground_litter_process import _HybridPadState, _analyse_hybrid
from rtsp_annotator.ground_litter_v32 import (
    CleanReferenceFrameAnalysis,
    CleanReferenceProfileV32,
)
from rtsp_annotator.event_engine import NormalizedRect

FRAME = np.zeros((240, 320, 3), np.uint8)
BOX = (100.0, 100.0, 140.0, 140.0)


def options(**overrides):
    zone = GroundLitterZone(
        region_id="r1", name="z",
        polygon=((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)),
    )
    values = dict(
        mode="hybrid_v33",
        enabled=True,
        zones=(zone,),
        analysis_fps=0.5,
        startup_suppress_seconds=0.0,
        profile_id="p",
        model="litter/turhancan_yolov8m_seg_trash.pt",
        semantic_scan_interval_seconds=4.0,
        prior_crop_maximum=4,
        normal_stability_samples=1,
        maximum_boxes=4,
    )
    values.update(overrides)
    return GroundLitterDetectionOptions(**values)


def profile() -> CleanReferenceProfileV32:
    reference = np.zeros((240, 320, 3), np.uint8)
    valid = np.full((240, 320), 255, np.uint8)
    tolerance = np.zeros((240, 320), np.uint8)
    return CleanReferenceProfileV32(
        "p", Path("."), reference, valid, tolerance, {},
    )


class FakeDetector:
    """Records batch calls so scheduling and batching can be asserted."""

    class_names = {3: "Plastic"}

    def __init__(self, *, tiles=2, crops=1, fail_tiles=False, fail_crops=False):
        self.tile_calls = 0
        self.crop_calls = 0
        self.tile_batch_sizes: list[int] = []
        self.crop_batch_sizes: list[int] = []
        self._tiles = tiles
        self._crops = crops
        self._fail_tiles = fail_tiles
        self._fail_crops = fail_crops

    def tile_candidates_batch(self, frame, opts, *, masks, tiles, night, actors):
        self.tile_calls += 1
        self.tile_batch_sizes.append(len(tiles))
        if self._fail_tiles:
            raise RuntimeError("semantic boom")
        candidates = [
            GroundLitterCandidate(
                rectangle=NormalizedRect(0.3, 0.4, 0.1, 0.1),
                confidence=0.7, class_name="Plastic", region_id="r1",
                tile_index=0,
            )
            for _ in range(self._tiles)
        ]
        stats = {"raw_candidates": self._tiles + 1, "rejected_roi": 1,
                 "rejected_actor": 0, "model_batches": 1}
        return candidates, stats

    def crop_candidates_batch(
        self, frame, boxes, opts, *, night,
    ):
        self.crop_calls += 1
        boxes = list(boxes)
        self.crop_batch_sizes.append(len(boxes))
        if self._fail_crops:
            raise RuntimeError("crop boom")
        rects = [(0, 0, 160, 160) for _ in boxes]
        rows = [
            [{"box": [100.0, 100.0, 140.0, 140.0], "confidence": 0.6,
              "class_id": 3}]
            for _ in boxes
        ]
        return rows, rects, 1


def fake_analysis(environment_state: str = "NORMAL", actor_rows=None):
    return CleanReferenceFrameAnalysis(
        analysis_frame=FRAME,
        normalized=FRAME,
        valid=np.ones((240, 320), np.uint8),
        tolerance=np.zeros((240, 320), np.uint8),
        actor_rows=list(actor_rows or []),
        environment_state=environment_state,
        environment={"state": environment_state},
        alignment={},
        width=320,
        height=240,
        pixel_scale=0.125,
    )


def prior_candidate(count: int):
    return [
        {
            "box": [100.0 + index, 100.0, 140.0 + index, 140.0],
            "anomaly_score": 1.0 - index * 0.1,
            "region_id": "r1",
            "support_pixels": 300,
        }
        for index in range(count)
    ]


class _Patched:
    """Patch the prior channel so the scheduler can be driven deterministically."""

    def __init__(
        self, *, environment_state="NORMAL", candidates=1, actor_rows=None,
    ):
        self.environment_state = environment_state
        self.candidates = candidates
        self.actor_rows = actor_rows
        self.analysis = patch(
            "rtsp_annotator.ground_litter_process.analyze_prior_frame"
        )
        self.propose = patch(
            "rtsp_annotator.ground_litter_process.propose_prior_candidates"
        )

    def __enter__(self):
        analysis_mock = self.analysis.start()
        analysis_mock.return_value = (
            fake_analysis(self.environment_state, self.actor_rows), "aligned",
        )
        propose_mock = self.propose.start()
        propose_mock.return_value = (
            prior_candidate(self.candidates),
            np.zeros((240, 320), np.uint8),
            self.candidates,
        )
        return analysis_mock, propose_mock

    def __exit__(self, *exc):
        self.propose.stop()
        self.analysis.stop()
        return False


class SchedulerTests(unittest.TestCase):
    def test_full_scan_runs_on_its_interval_and_crops_fill_the_other_ticks(self):
        from rtsp_annotator.ground_litter_detection import (
            build_ground_litter_tiles,
        )

        options_value = options()
        state = _HybridPadState(options_value, profile())
        detector = FakeDetector()
        expected_tiles = len(
            build_ground_litter_tiles(options_value, 320, 240)[1]
        )
        self.assertGreaterEqual(expected_tiles, 1)
        with _Patched(candidates=3):
            for step in range(6):
                _analyse_hybrid(
                    state=state, detector=detector, bgr=FRAME,
                    timestamp=float(step * 2), night=False, actor_boxes=[],
                )
        # scan interval 4 s at a 2 s tick: ticks 0, 4, 8 scan; 2, 6, 10 crop.
        self.assertEqual(detector.tile_calls, 3)
        self.assertEqual(detector.crop_calls, 3)
        # One batched model call per scan, covering every planned tile.
        self.assertEqual(detector.tile_batch_sizes, [expected_tiles] * 3)
        self.assertTrue(all(size == 3 for size in detector.crop_batch_sizes))

    def test_full_scan_tick_does_not_also_run_crops(self):
        state = _HybridPadState(options(), profile())
        detector = FakeDetector()
        with _Patched(candidates=2):
            _analyse_hybrid(
                state=state, detector=detector, bgr=FRAME, timestamp=0.0,
                night=False, actor_boxes=[],
            )
        self.assertEqual(detector.tile_calls, 1)
        self.assertEqual(detector.crop_calls, 0)

    def test_crop_selection_respects_the_configured_maximum(self):
        state = _HybridPadState(
            options(prior_crop_maximum=2, semantic_scan_interval_seconds=60.0),
            profile(),
        )
        detector = FakeDetector()
        with _Patched(candidates=5):
            _analyse_hybrid(
                state=state, detector=detector, bgr=FRAME, timestamp=0.0,
                night=False, actor_boxes=[],
            )
            self.assertEqual(detector.tile_calls, 1)
            _analyse_hybrid(
                state=state, detector=detector, bgr=FRAME, timestamp=2.0,
                night=False, actor_boxes=[],
            )
        self.assertEqual(detector.crop_batch_sizes, [2])

    def test_no_crops_when_crop_maximum_is_zero(self):
        state = _HybridPadState(
            options(prior_crop_maximum=0, semantic_scan_interval_seconds=60.0),
            profile(),
        )
        detector = FakeDetector()
        with _Patched(candidates=4):
            _analyse_hybrid(state=state, detector=detector, bgr=FRAME,
                            timestamp=0.0, night=False, actor_boxes=[])
            _analyse_hybrid(state=state, detector=detector, bgr=FRAME,
                            timestamp=2.0, night=False, actor_boxes=[])
        self.assertEqual(detector.tile_calls, 1)
        self.assertEqual(detector.crop_calls, 0)


class IsolationTests(unittest.TestCase):
    def test_semantic_failure_leaves_prior_running(self):
        state = _HybridPadState(options(), profile())
        detector = FakeDetector(fail_tiles=True)
        with _Patched(candidates=2):
            snapshot = _analyse_hybrid(
                state=state, detector=detector, bgr=FRAME, timestamp=0.0,
                night=False, actor_boxes=[],
            )
        self.assertEqual(snapshot.branch_state, "semantic_degraded")
        self.assertNotEqual(snapshot.state, "error")
        self.assertGreaterEqual(snapshot.prior_raw_candidates, 1)

    def test_prior_failure_leaves_semantic_running(self):
        state = _HybridPadState(options(), profile())
        detector = FakeDetector()
        with patch(
            "rtsp_annotator.ground_litter_process.analyze_prior_frame",
            side_effect=RuntimeError("prior boom"),
        ):
            snapshot = _analyse_hybrid(
                state=state, detector=detector, bgr=FRAME, timestamp=0.0,
                night=False, actor_boxes=[],
            )
        self.assertEqual(snapshot.branch_state, "prior_degraded")
        self.assertGreaterEqual(snapshot.semantic_retained_candidates, 1)
        self.assertEqual(detector.tile_calls, 1)

    def test_environment_change_pauses_prior_but_keeps_semantic(self):
        state = _HybridPadState(options(), profile())
        detector = FakeDetector()
        with _Patched(environment_state="GLOBAL_LIGHT_CHANGE", candidates=2):
            snapshot = _analyse_hybrid(
                state=state, detector=detector, bgr=FRAME, timestamp=0.0,
                night=False, actor_boxes=[],
            )
        self.assertEqual(snapshot.prior_raw_candidates, 0)
        self.assertEqual(detector.tile_calls, 1)
        self.assertNotEqual(snapshot.state, "abstaining")

    def test_stability_gate_delays_prior_availability(self):
        state = _HybridPadState(options(normal_stability_samples=3), profile())
        detector = FakeDetector()
        with _Patched(candidates=1):
            first = _analyse_hybrid(state=state, detector=detector, bgr=FRAME,
                                    timestamp=0.0, night=False, actor_boxes=[])
            second = _analyse_hybrid(state=state, detector=detector, bgr=FRAME,
                                     timestamp=2.0, night=False, actor_boxes=[])
            third = _analyse_hybrid(state=state, detector=detector, bgr=FRAME,
                                    timestamp=4.0, night=False, actor_boxes=[])
        self.assertEqual(first.prior_raw_candidates, 0)
        self.assertEqual(second.prior_raw_candidates, 0)
        self.assertGreaterEqual(third.prior_raw_candidates, 1)

    def test_telemetry_is_populated_for_both_channels(self):
        state = _HybridPadState(options(), profile())
        detector = FakeDetector()
        with _Patched(candidates=3):
            snapshot = _analyse_hybrid(
                state=state, detector=detector, bgr=FRAME, timestamp=0.0,
                night=False, actor_boxes=[],
            )
        self.assertGreater(snapshot.semantic_raw_candidates, 0)
        self.assertGreater(snapshot.semantic_retained_candidates, 0)
        self.assertEqual(snapshot.prior_raw_candidates, 3)
        self.assertEqual(snapshot.prior_retained_candidates, 3)
        self.assertEqual(snapshot.semantic_model_runs_full, 1)
        self.assertEqual(snapshot.semantic_model_runs_crop, 0)
        self.assertGreaterEqual(snapshot.last_total_ms, 0.0)

    def test_result_version_increments_for_each_published_tick(self):
        state = _HybridPadState(options(), profile())
        with _Patched(candidates=0):
            first = _analyse_hybrid(
                state=state, detector=None, bgr=FRAME, timestamp=0.0,
                night=False, actor_boxes=[],
            )
            second = _analyse_hybrid(
                state=state, detector=None, bgr=FRAME, timestamp=2.0,
                night=False, actor_boxes=[],
            )
        self.assertEqual(first.result_version, 1)
        self.assertEqual(second.result_version, 2)

    def test_profile_space_actor_rows_occlude_resized_input(self):
        cases = (
            ((120, 160), [50.0, 50.0, 70.0, 70.0]),
            ((240, 320), [100.0, 100.0, 140.0, 140.0]),
            ((480, 640), [200.0, 200.0, 280.0, 280.0]),
        )
        for (height, width), native_actor in cases:
            with self.subTest(size=(width, height)):
                state = _HybridPadState(options(), profile())
                frame = np.zeros((height, width, 3), np.uint8)
                with _Patched(
                    candidates=1,
                    actor_rows=[[100.0, 100.0, 140.0, 140.0]],
                ):
                    snapshot = None
                    for step in range(6):
                        snapshot = _analyse_hybrid(
                            state=state, detector=None, bgr=frame,
                            timestamp=float(step * 2), night=False,
                            actor_boxes=[native_actor],
                        )
                self.assertIsNotNone(snapshot)
                self.assertEqual(snapshot.confirmed_events, 0)
                self.assertEqual(snapshot.detections, ())
                self.assertEqual(state.memory.active[0].state, "OCCLUDED")


if __name__ == "__main__":
    unittest.main()
