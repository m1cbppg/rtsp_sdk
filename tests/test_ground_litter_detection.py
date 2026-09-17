"""Unit tests for the API-facing ground-litter detection module."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.ground_litter_detection import (
    GroundLitterCandidate,
    GroundLitterDetection,
    GroundLitterDetectionOptions,
    GroundLitterDisplayTracker,
    GroundLitterResultCache,
    GroundLitterSnapshot,
    GroundLitterZone,
    UltralyticsGroundLitterDetector,
    as_bgr,
    build_ground_litter_tiles,
    snapshot_is_fresh,
)


def zone(
    region_id: str = "z1",
    polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    *,
    short_side: int = 4,
    area: int = 16,
) -> GroundLitterZone:
    return GroundLitterZone(
        region_id=region_id,
        polygon=polygon,
        minimum_short_side_px=short_side,
        minimum_box_area_px=area,
    )


def options(**overrides) -> GroundLitterDetectionOptions:
    values = {
        "enabled": True,
        "zones": (zone(),),
        "tile_size_px": 160,
        "maximum_tiles": 64,
    }
    values.update(overrides)
    return GroundLitterDetectionOptions(**values)


class FakeTensor:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[list[float]]:
        return self._rows


class FakeBoxes:
    """Mirrors the real ``result.boxes``: a tensor-like ``.data`` field."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.data = FakeTensor(rows)


class FakeResult:
    def __init__(self, rows: list[list[float]] | None) -> None:
        self.boxes = FakeBoxes(rows) if rows is not None else None


class FakeModel:
    """Minimal Ultralytics stand-in returning one fixed row set per call."""

    def __init__(self, rows: list[list[float]] | None, names=None) -> None:
        self.rows = rows
        self.names = names or {0: "Plastic", 1: "Paper"}
        self.kwargs: list[dict] = []

    def predict(self, image, **kwargs):
        self.kwargs.append(kwargs)
        return [FakeResult(self.rows)]


class GroundLitterOptionsTests(unittest.TestCase):
    def test_new_controls_round_trip_and_bounds(self):
        value = options(inference_imgsz=640, actor_model="yolo26s.pt",
                        local_actor_max_crops=4, box_smoothing_alpha=0.5,
                        context_class_ids=(25, 56))
        self.assertEqual(GroundLitterDetectionOptions.from_payload(value.to_payload()), value)
        for kwargs in ({"inference_imgsz": 128}, {"local_actor_max_crops": 9},
                       {"local_actor_max_crops": 1}, {"box_smoothing_alpha": 0}):
            with self.assertRaises(ValueError):
                options(**kwargs).validate()

    def test_crop_size_and_input_size_are_independent(self):
        model = FakeModel([[10, 20, 30, 40, 0.8, 0]])
        detector = UltralyticsGroundLitterDetector(model_path="trash.pt", model_factory=lambda _: model)
        value = options(tile_size_px=320, inference_imgsz=640)
        masks, tiles = build_ground_litter_tiles(value, 320, 320)
        candidates, _ = detector.candidates(np.zeros((320, 320, 3), np.uint8), value, masks=masks, tiles=tiles)
        self.assertEqual(model.kwargs[0]["imgsz"], 640)
        self.assertAlmostEqual(candidates[0].rectangle.left, 10 / 320)
        self.assertEqual(options(tile_size_px=320).effective_imgsz, 320)

    def test_local_actor_budget_and_rejection(self):
        litter = FakeModel([[200, 200, 220, 220, 0.8, 0], [800, 200, 820, 220, 0.7, 0]])
        actor = FakeModel([[180, 180, 250, 250, 0.9, 3]])
        detector = UltralyticsGroundLitterDetector(model_path="trash.pt", actor_model_path="actor.pt",
            model_factory=lambda path: actor if path == "actor.pt" else litter)
        value = options(tile_size_px=1024, actor_model="actor.pt", local_actor_max_crops=1)
        masks, tiles = build_ground_litter_tiles(value, 1024, 1024)
        candidates, stats = detector.candidates(np.zeros((1024, 1024, 3), np.uint8), value, masks=masks, tiles=tiles)
        self.assertEqual(len(actor.kwargs), 1)
        self.assertEqual(stats["local_actor_crops"], 1)
        self.assertEqual(stats["rejected_actor"], 1)
        self.assertEqual(len(candidates), 1)
        self.assertGreater(candidates[0].rectangle.left, 0.7)

    def test_enabled_requires_at_least_one_zone(self) -> None:
        with self.assertRaisesRegex(ValueError, "至少需要一个地面区域"):
            GroundLitterDetectionOptions(enabled=True).validate()

    def test_disabled_options_need_no_zone(self) -> None:
        GroundLitterDetectionOptions().validate()

    def test_minimum_hits_cannot_exceed_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum_hits"):
            options(minimum_hits=4, hit_window=3).validate()

    def test_hold_cannot_exceed_maximum_age(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum_age_seconds"):
            options(hold_seconds=10.0, maximum_age_seconds=5.0).validate()

    def test_unsafe_model_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "model"):
            options(model="../models/evil.pt").validate()
        with self.assertRaisesRegex(ValueError, "model"):
            options(model="/etc/passwd.pt").validate()

    def test_duplicate_region_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "区域ID重复"):
            options(zones=(zone("same"), zone("same"))).validate()

    def test_polygon_outside_unit_square_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "0, 1"):
            zone(polygon=((0.0, 0.0), (1.2, 0.0), (1.0, 1.0))).validate()

    def test_night_confidence_overrides_day_confidence(self) -> None:
        options_ = options(confidence=0.2, night_confidence=0.35)
        self.assertEqual(options_.confidence_for(False), 0.2)
        self.assertEqual(options_.confidence_for(True), 0.35)
        self.assertEqual(options(confidence=0.2).confidence_for(True), 0.2)

    def test_payload_round_trip_preserves_options(self) -> None:
        original = options(
            actor_model="yolo26s.pt",
            zones=(
                zone("merchant_01", short_side=8, area=64),
                zone(
                    "merchant_02",
                    polygon=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0)),
                    short_side=12,
                    area=160,
                ),
            ),
            overlay_exclude_zones=(
                ((0.0, 0.0), (0.4, 0.0), (0.4, 0.1), (0.0, 0.1)),
            ),
        )
        restored = GroundLitterDetectionOptions.from_payload(
            original.to_payload()
        )
        self.assertEqual(original, restored)

    def test_region_ids_are_exposed_in_order(self) -> None:
        options_ = options(zones=(zone("a"), zone("b")))
        self.assertEqual(options_.region_ids, ("a", "b"))


class GroundLitterGeometryTests(unittest.TestCase):
    def test_tiles_cover_the_ground_region(self) -> None:
        options_ = options(tile_size_px=640)
        masks, tiles = build_ground_litter_tiles(options_, 2560, 1440)
        self.assertTrue(tiles)
        mask = masks["z1"]
        for x, y, right, bottom in tiles:
            self.assertTrue(mask[y:bottom, x:right].any())

    def test_tile_plan_respects_maximum_tiles(self) -> None:
        options_ = options(tile_size_px=320, maximum_tiles=1)
        with self.assertRaisesRegex(ValueError, "maximum_tiles"):
            build_ground_litter_tiles(options_, 2560, 1440)

    def test_empty_exclusion_can_remove_all_ground(self) -> None:
        options_ = options(
            zones=(
                GroundLitterZone(
                    region_id="z1",
                    polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                    exclude_zones=(
                        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                    ),
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "visible ground"):
            build_ground_litter_tiles(options_, 320, 320)


class GroundLitterDetectorTests(unittest.TestCase):
    def _detector(self, rows, names=None):
        return UltralyticsGroundLitterDetector(
            model_path="model.pt",
            device="cpu",
            half=False,
            model_factory=lambda _path: FakeModel(rows, names),
        )

    def test_outside_high_score_box_cannot_suppress_ground_target(self):
        detector = self._detector([[30, 10, 70, 50, .8, 0],
                                   [36, 10, 76, 50, .9, 0]])
        value = options(zones=(zone(polygon=((0, 0), (.52, 0), (.52, 1), (0, 1))),))
        masks, _ = build_ground_litter_tiles(value, 100, 100)
        candidates, stats = detector.candidates(np.zeros((100, 100, 3), np.uint8),
                                               value, masks=masks, tiles=[(0, 0, 100, 100)])
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].rectangle.left, .3)
        self.assertEqual(stats["rejected_roi"], 1)

    def test_overlapping_inherited_zone_is_not_overridden_by_stricter_zone(self):
        detector = self._detector([[10, 10, 40, 40, .3, 0]])
        value = options(confidence=.2, zones=(zone("inherited"),
                                              replace(zone("strict"), confidence=.8)))
        masks, tiles = build_ground_litter_tiles(value, 100, 100)
        candidates, stats = detector.candidates(np.zeros((100, 100, 3), np.uint8),
                                               value, masks=masks, tiles=tiles)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].region_id, "inherited")
        self.assertEqual(stats["rejected_confidence"], 0)

    def test_low_zone_threshold_reaches_predict_and_other_zone_stays_strict(self):
        model = FakeModel([[10, 10, 30, 30, .12, 0], [70, 10, 90, 30, .2, 0]])
        detector = UltralyticsGroundLitterDetector(model_path="model.pt", model_factory=lambda _: model)
        value = options(zones=(
            replace(zone("far", ((0, 0), (.5, 0), (.5, 1), (0, 1))), confidence=.1),
            replace(zone("near", ((.5, 0), (1, 0), (1, 1), (.5, 1))), confidence=.3)))
        masks, tiles = build_ground_litter_tiles(value, 100, 100)
        candidates, stats = detector.candidates(np.zeros((100, 100, 3), np.uint8),
                                               value, masks=masks, tiles=tiles)
        self.assertEqual(model.kwargs[0]["conf"], .1)
        self.assertEqual([c.region_id for c in candidates], ["far"])
        self.assertEqual(stats["rejected_confidence"], 1)

    def test_threshold_boundary_survives_nms(self):
        detector = self._detector([[10, 10, 40, 40, .25, 0]])
        value = options(confidence=.25)
        masks, tiles = build_ground_litter_tiles(value, 100, 100)
        candidates, _ = detector.candidates(np.zeros((100, 100, 3), np.uint8),
                                            value, masks=masks, tiles=tiles)
        self.assertEqual(len(candidates), 1)

    def test_keeps_only_in_zone_boxes_above_the_size_gate(self) -> None:
        detector = self._detector(
            [
                [10, 10, 40, 40, 0.9, 0],  # inside, big enough
                [10, 60, 12, 62, 0.9, 0],  # inside but too small
                [60, 10, 90, 40, 0.9, 1],  # outside the left-half zone
            ]
        )
        options_ = options(
            zones=(
                zone(
                    polygon=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)),
                ),
            )
        )
        masks, tiles = build_ground_litter_tiles(options_, 100, 100)
        candidates, stats = detector.candidates(
            np.zeros((100, 100, 3), np.uint8),
            options_,
            masks=masks,
            tiles=tiles,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].class_name, "Plastic")
        self.assertEqual(candidates[0].region_id, "z1")
        self.assertAlmostEqual(candidates[0].rectangle.left, 0.10, places=3)
        self.assertEqual(stats["rejected_roi"], 2)
        self.assertEqual(stats["raw_candidates"], 3)

    def test_actor_overlap_rejects_a_ground_object(self) -> None:
        detector = self._detector([[10, 10, 40, 40, 0.9, 0]])
        options_ = options()
        masks, tiles = build_ground_litter_tiles(options_, 100, 100)
        candidates, stats = detector.candidates(
            np.zeros((100, 100, 3), np.uint8),
            options_,
            masks=masks,
            tiles=tiles,
            actors=[(8.0, 8.0, 42.0, 42.0)],
        )
        self.assertEqual(candidates, [])
        self.assertEqual(stats["rejected_actor"], 1)

    def test_actor_overlap_below_threshold_keeps_the_box(self) -> None:
        detector = self._detector([[60, 60, 90, 90, 0.9, 0]])
        options_ = options()
        masks, tiles = build_ground_litter_tiles(options_, 100, 100)
        candidates, _stats = detector.candidates(
            np.zeros((100, 100, 3), np.uint8),
            options_,
            masks=masks,
            tiles=tiles,
            actors=[(0.0, 0.0, 20.0, 20.0)],
        )
        self.assertEqual(len(candidates), 1)

    def test_cross_tile_duplicates_are_merged_by_nms(self) -> None:
        detector = self._detector([[10, 10, 60, 60, 0.8, 0]])
        options_ = options()
        masks, _tiles = build_ground_litter_tiles(options_, 100, 100)
        candidates, stats = detector.candidates(
            np.zeros((100, 100, 3), np.uint8),
            options_,
            masks=masks,
            tiles=[(0, 0, 100, 100), (0, 0, 100, 100)],
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(stats["raw_candidates"], 2)

    def test_ambiguous_region_ownership_stays_unassigned(self) -> None:
        detector = self._detector([[40, 10, 60, 40, 0.9, 0]])
        options_ = options(
            zones=(
                zone(
                    "left",
                    ((0.0, 0.0), (0.6, 0.0), (0.6, 1.0), (0.0, 1.0)),
                ),
                zone(
                    "right",
                    ((0.4, 0.0), (1.0, 0.0), (1.0, 1.0), (0.4, 1.0)),
                ),
            )
        )
        masks, tiles = build_ground_litter_tiles(options_, 100, 100)
        candidates, _stats = detector.candidates(
            np.zeros((100, 100, 3), np.uint8),
            options_,
            masks=masks,
            tiles=tiles,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].region_id, "")

    def test_tile_inference_uses_native_size_and_no_rescale(self) -> None:
        model = FakeModel([[10, 10, 40, 40, 0.9, 0]])
        detector = UltralyticsGroundLitterDetector(
            model_path="model.pt",
            device="cpu",
            half=False,
            model_factory=lambda _path: model,
        )
        options_ = options(tile_size_px=160)
        masks, tiles = build_ground_litter_tiles(options_, 320, 320)
        detector.candidates(
            np.zeros((320, 320, 3), np.uint8),
            options_,
            masks=masks,
            tiles=tiles,
        )
        self.assertTrue(model.kwargs)
        for kwargs in model.kwargs:
            self.assertEqual(kwargs["imgsz"], 160)
            self.assertNotIn("half", kwargs)
            self.assertNotIn("quantize", kwargs)

    def test_half_precision_uses_quantize_flag(self) -> None:
        model = FakeModel([[10, 10, 40, 40, 0.9, 0]])
        detector = UltralyticsGroundLitterDetector(
            model_path="model.pt",
            device="cuda:0",
            half=True,
            model_factory=lambda _path: model,
        )
        options_ = options()
        masks, tiles = build_ground_litter_tiles(options_, 100, 100)
        detector.candidates(
            np.zeros((100, 100, 3), np.uint8),
            options_,
            masks=masks,
            tiles=tiles,
        )
        self.assertEqual(model.kwargs[0]["quantize"], 16)

    def test_confident_night_threshold_is_applied(self) -> None:
        model = FakeModel([[10, 10, 40, 40, 0.9, 0]])
        detector = UltralyticsGroundLitterDetector(
            model_path="model.pt",
            device="cpu",
            half=False,
            model_factory=lambda _path: model,
        )
        options_ = options(confidence=0.2, night_confidence=0.4)
        masks, tiles = build_ground_litter_tiles(options_, 100, 100)
        detector.candidates(
            np.zeros((100, 100, 3), np.uint8),
            options_,
            masks=masks,
            tiles=tiles,
            night=True,
        )
        self.assertEqual(model.kwargs[0]["conf"], 0.4)

    def test_actor_model_is_optional_and_filtered_by_class(self) -> None:
        created: list[FakeModel] = []

        def factory(path: str) -> FakeModel:
            model = FakeModel([[0, 0, 10, 10, 0.5, 0]])
            created.append(model)
            return model

        detector = UltralyticsGroundLitterDetector(
            model_path="model.pt",
            actor_model_path="yolo26s.pt",
            device="cpu",
            half=False,
            model_factory=factory,
        )
        self.assertTrue(detector.has_actor_model)
        boxes = detector.actor_boxes(
            np.zeros((100, 100, 3), np.uint8),
            options(actor_imgsz=640, actor_class_ids=(0, 3)),
        )
        self.assertEqual(created[1].kwargs[0]["classes"], [0, 3, 13, 25, 56, 58, 60])
        self.assertEqual(created[1].kwargs[0]["imgsz"], 640)
        # Only coordinates may be returned: the caller unpacks exactly four
        # values in box_overlap_fraction, so passing the raw six-column rows
        # would raise on every analysed frame.
        self.assertEqual(boxes, [[0.0, 0.0, 10.0, 10.0]])

    def test_actor_boxes_feed_the_overlap_filter(self) -> None:
        def factory(_path: str) -> FakeModel:
            return FakeModel([[0, 0, 40, 40, 0.9, 0]])

        detector = UltralyticsGroundLitterDetector(
            model_path="model.pt",
            actor_model_path="yolo26s.pt",
            device="cpu",
            half=False,
            model_factory=factory,
        )
        options_ = options()
        masks, tiles = build_ground_litter_tiles(options_, 100, 100)
        frame = np.zeros((100, 100, 3), np.uint8)
        actors = detector.actor_boxes(frame, options_)
        candidates, stats = detector.candidates(
            frame,
            options_,
            masks=masks,
            tiles=tiles,
            actors=actors,
        )
        self.assertEqual(candidates, [])
        self.assertEqual(stats["rejected_actor"], 1)

    def test_empty_prediction_returns_no_candidates(self) -> None:
        detector = UltralyticsGroundLitterDetector(
            model_path="model.pt",
            device="cpu",
            half=False,
            model_factory=lambda _path: FakeModel(None),
        )
        options_ = options()
        masks, tiles = build_ground_litter_tiles(options_, 100, 100)
        candidates, stats = detector.candidates(
            np.zeros((100, 100, 3), np.uint8),
            options_,
            masks=masks,
            tiles=tiles,
        )
        self.assertEqual(candidates, [])
        self.assertEqual(stats["raw_candidates"], 0)

    def test_as_bgr_swaps_channels_into_ultralytics_order(self) -> None:
        frame = np.zeros((4, 4, 3), np.uint8)
        frame[..., 0] = 10
        frame[..., 2] = 30
        converted = as_bgr(frame)
        self.assertEqual(converted.shape, (4, 4, 3))
        self.assertEqual(converted[0, 0].tolist(), [30, 0, 10])

    def test_as_bgr_accepts_chw_and_single_batch_frames(self) -> None:
        chw = np.zeros((3, 8, 8), np.uint8)
        chw[0] = 10
        chw[2] = 30
        converted = as_bgr(chw)
        self.assertEqual(converted.shape, (8, 8, 3))
        self.assertEqual(converted[0, 0].tolist(), [30, 0, 10])
        self.assertEqual(as_bgr(chw[None, ...]).shape, (8, 8, 3))

    def test_as_bgr_rejects_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "形状无效"):
            as_bgr(np.zeros((4, 4), np.uint8))


class GroundLitterDisplayTrackerTests(unittest.TestCase):
    def test_occlusion_cancels_hold_and_requires_confirmation_again(self):
        tracker = self._tracker()
        tracker.update([self._candidate()], timestamp=1)
        self.assertEqual(tracker.update([self._candidate()], timestamp=2).count, 1)
        self.assertEqual(tracker.update([], timestamp=3,
            occluders=[NormalizedRect(0.1, 0.2, 0.4, 0.4)]).count, 0)
        self.assertEqual(tracker.update([self._candidate()], timestamp=4).count, 0)

    def test_confirmed_box_survives_window_misses_until_hold_expires(self):
        tracker = self._tracker(hold_seconds=3)
        tracker.update([self._candidate(confidence=0.9)], timestamp=1)
        tracker.update([self._candidate(confidence=0.2)], timestamp=1.5)
        for timestamp in (2, 2.5, 3, 3.5, 4):
            result = tracker.update([], timestamp=timestamp)
            self.assertEqual(result.count, 1)
            self.assertAlmostEqual(result.detections[0].confidence, 0.2)
        self.assertEqual(tracker.update([], timestamp=4.6).count, 0)
        self.assertEqual(tracker.update([self._candidate()], timestamp=5).count, 0)
        self.assertEqual(tracker.update([self._candidate()], timestamp=5.5).count, 1)

    def test_duplicate_or_old_timestamp_cannot_confirm(self):
        tracker = self._tracker()
        for timestamp in (10, 10, 9):
            self.assertEqual(tracker.update([self._candidate()], timestamp=timestamp).count, 0)
        self.assertEqual(tracker.update([self._candidate()], timestamp=11).count, 1)

    def test_display_smoothing_does_not_change_matching_geometry(self):
        tracker = self._tracker(minimum_hits=1, box_smoothing_alpha=0.5)
        tracker.update([self._candidate()], timestamp=1)
        result = tracker.update([self._candidate(left=0.204)], timestamp=2)
        self.assertAlmostEqual(result.detections[0].rectangle.left, 0.202)
        self.assertAlmostEqual(next(iter(tracker._tracks.values())).rectangle.left, 0.204)

    def _tracker(self, **overrides) -> GroundLitterDisplayTracker:
        values = {"minimum_hits": 2, "hit_window": 3}
        values.update(overrides)
        return GroundLitterDisplayTracker(options(**values))

    @staticmethod
    def _candidate(
        left: float = 0.2,
        top: float = 0.3,
        width: float = 0.02,
        confidence: float = 0.4,
    ) -> GroundLitterCandidate:
        return GroundLitterCandidate(
            rectangle=NormalizedRect(left, top, width, width),
            confidence=confidence,
            class_name="Plastic",
        )

    def test_box_needs_two_hits_inside_the_window(self) -> None:
        tracker = self._tracker()
        first = tracker.update([self._candidate()], timestamp=0.0)
        self.assertEqual(first.count, 0)
        missed = tracker.update([], timestamp=1.0)
        self.assertEqual(missed.count, 0)
        second = tracker.update([self._candidate()], timestamp=2.0)
        self.assertEqual(second.count, 1)
        self.assertEqual(second.result_version, 3)

    def test_box_is_held_after_the_model_stops_seeing_it(self) -> None:
        tracker = self._tracker(hold_seconds=3.0)
        tracker.update([self._candidate()], timestamp=0.0)
        tracker.update([self._candidate()], timestamp=1.0)
        held = tracker.update([], timestamp=3.5)
        self.assertEqual(held.count, 1)
        gone = tracker.update([], timestamp=5.0)
        self.assertEqual(gone.count, 0)

    def test_track_expires_after_maximum_age(self) -> None:
        tracker = self._tracker(hold_seconds=1.0, maximum_age_seconds=2.0)
        tracker.update([self._candidate()], timestamp=0.0)
        tracker.update([self._candidate()], timestamp=1.0)
        tracker.update([], timestamp=10.0)
        self.assertEqual(tracker._tracks, {})
        self.assertEqual(
            tracker.update([self._candidate()], timestamp=11.0).count,
            0,
        )

    def test_matching_is_one_to_one_per_candidate(self) -> None:
        tracker = self._tracker()
        tracker.update([self._candidate()], timestamp=0.0)
        result = tracker.update(
            [self._candidate(), self._candidate(left=0.21)],
            timestamp=1.0,
        )
        self.assertEqual(len(tracker._tracks), 2)
        self.assertEqual(result.count, 1)

    def test_distant_candidate_creates_a_new_track(self) -> None:
        tracker = self._tracker()
        tracker.update([self._candidate()], timestamp=0.0)
        tracker.update(
            [self._candidate(left=0.8, top=0.8)],
            timestamp=1.0,
        )
        self.assertEqual(len(tracker._tracks), 2)

    def test_size_ratio_guard_blocks_a_much_larger_box(self) -> None:
        tracker = self._tracker(minimum_hits=1)
        tracker.update([self._candidate()], timestamp=0.0)
        tracker.update(
            [self._candidate(width=0.40)],
            timestamp=1.0,
        )
        self.assertEqual(len(tracker._tracks), 2)

    def test_maximum_boxes_sorted_by_confidence(self) -> None:
        tracker = self._tracker(minimum_hits=1, maximum_boxes=2)
        result = tracker.update(
            [
                self._candidate(left=0.1, confidence=0.3),
                self._candidate(left=0.4, confidence=0.9),
                self._candidate(left=0.7, confidence=0.6),
            ],
            timestamp=0.0,
        )
        self.assertEqual(result.count, 2)
        self.assertAlmostEqual(result.detections[0].confidence, 0.9)
        self.assertAlmostEqual(result.detections[1].confidence, 0.6)

    def test_tracker_reports_the_highest_confidence_seen(self) -> None:
        tracker = self._tracker(minimum_hits=1)
        tracker.update([self._candidate(confidence=0.3)], timestamp=0.0)
        result = tracker.update(
            [self._candidate(confidence=0.7)],
            timestamp=1.0,
        )
        self.assertAlmostEqual(result.detections[0].confidence, 0.7)

    def test_reset_forgets_every_track(self) -> None:
        tracker = self._tracker(minimum_hits=1)
        tracker.update([self._candidate()], timestamp=0.0)
        tracker.reset()
        self.assertEqual(tracker._tracks, {})


class GroundLitterCacheTests(unittest.TestCase):
    def test_newer_snapshot_replaces_older_one(self) -> None:
        cache = GroundLitterResultCache()
        cache.store_snapshot(
            0,
            GroundLitterSnapshot(state="running", result_version=2),
        )
        cache.store_snapshot(
            0,
            GroundLitterSnapshot(state="starting", result_version=1),
        )
        self.assertEqual(cache.snapshot(0).result_version, 2)
        self.assertEqual(cache.snapshot(0).state, "running")

    def test_mark_state_keeps_last_detections(self) -> None:
        cache = GroundLitterResultCache()
        cache.store_snapshot(
            0,
            GroundLitterSnapshot(
                state="running",
                detections=(
                    GroundLitterDetection(
                        object_id=1,
                        rectangle=NormalizedRect(0.1, 0.1, 0.1, 0.1),
                        confidence=0.5,
                    ),
                ),
                result_version=4,
                updated_at=1.0,
            ),
        )
        cache.mark_state(0, "error", "进程已退出")
        snapshot = cache.snapshot(0)
        self.assertEqual(snapshot.state, "error")
        self.assertEqual(snapshot.count, 1)
        self.assertEqual(snapshot.result_version, 4)

    def test_unknown_pad_returns_disabled_snapshot(self) -> None:
        cache = GroundLitterResultCache()
        self.assertEqual(cache.snapshot(3).state, "disabled")

    def test_freshness_guard_drops_stale_or_unset_snapshots(self) -> None:
        options_ = options(hold_seconds=3.0)
        self.assertFalse(
            snapshot_is_fresh(
                GroundLitterSnapshot(state="running"),
                options_,
                now=10.0,
            )
        )
        self.assertTrue(
            snapshot_is_fresh(
                GroundLitterSnapshot(state="running", updated_at=9.0),
                options_,
                now=12.0,
            )
        )
        self.assertFalse(
            snapshot_is_fresh(
                GroundLitterSnapshot(state="running", updated_at=1.0),
                options_,
                now=12.0,
            )
        )


class GroundLitterZoneTests(unittest.TestCase):
    def test_zone_confidence_inheritance(self):
        self.assertEqual(zone().confidence_for(True, .4), .4)
        self.assertEqual(replace(zone(), confidence=.3).confidence_for(True, .4), .3)
        both = replace(zone(), confidence=.3, night_confidence=.1)
        self.assertEqual(both.confidence_for(False, .4), .3)
        self.assertEqual(both.confidence_for(True, .4), .1)

    def test_zone_payload_round_trip(self) -> None:
        original = GroundLitterZone(
            region_id="merchant_01",
            name="门店01",
            polygon=((0.1, 0.1), (0.4, 0.1), (0.4, 0.5)),
            exclude_zones=(((0.0, 0.0), (0.1, 0.0), (0.1, 0.1)),),
            minimum_short_side_px=8,
            minimum_box_area_px=64,
            confidence=0.30,
            night_confidence=0.22,
        )
        restored = GroundLitterZone.from_payload(original.to_payload())
        self.assertEqual(original, restored)

    def test_region_id_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "region_id"):
            GroundLitterZone(region_id="", polygon=((0, 0), (1, 0), (1, 1))).validate()

    def test_size_gate_bounds_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "短边"):
            zone(short_side=0).validate()
        with self.assertRaisesRegex(ValueError, "面积"):
            zone(area=0).validate()

    def test_candidate_payload_is_finite(self) -> None:
        payload = GroundLitterCandidate(
            rectangle=NormalizedRect(0.1, 0.2, 0.3, 0.4),
            confidence=0.5,
        ).to_payload()
        self.assertTrue(all(math.isfinite(v) for v in payload["rectangle"]))


if __name__ == "__main__":
    unittest.main()
