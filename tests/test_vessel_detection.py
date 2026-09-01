from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.vessel_detection import (
    SMALL_TARGET_PROPOSAL_CLASS_ID,
    TemporalSmallTargetProposer,
    UltralyticsVesselDetector,
    VesselCandidate,
    VesselDetectionOptions,
    VesselTrackManager,
    deduplicate_candidates,
)


class _Tensor:
    def __init__(self, value: object) -> None:
        self._value = np.asarray(value)

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._value


def _result(
    boxes: list[list[float]],
    scores: list[float],
    classes: list[int],
) -> SimpleNamespace:
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=_Tensor(boxes),
            conf=_Tensor(scores),
            cls=_Tensor(classes),
        )
    )


class _FakeModel:
    def __init__(self, _path: str) -> None:
        self.kwargs: dict[str, object] = {}

    def predict(self, **kwargs: object) -> list[SimpleNamespace]:
        self.kwargs = kwargs
        return [
            _result([[100, 20, 140, 50]], [0.55], [8]),
            _result([[0, 20, 40, 50]], [0.75], [8]),
        ]


class VesselDetectionTests(unittest.TestCase):
    def test_perspective_crops_map_back_and_deduplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "yolo26s.pt"
            model_path.touch()
            detector = UltralyticsVesselDetector(
                model_path=model_path,
                device="cpu",
                half=True,
                model_factory=_FakeModel,
            )
            options = VesselDetectionOptions(
                enabled=True,
                inference_regions=(
                    (0.0, 0.0, 1.0, 1.0),
                    (0.5, 0.0, 1.0, 1.0),
                ),
            )

            candidates = detector.detect(
                np.zeros((100, 200, 3), dtype=np.uint8),
                options,
            )

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].rectangle.left, 0.5)
        self.assertAlmostEqual(candidates[0].rectangle.top, 0.2)
        self.assertAlmostEqual(candidates[0].rectangle.width, 0.2)
        self.assertAlmostEqual(candidates[0].rectangle.height, 0.3)
        self.assertAlmostEqual(candidates[0].confidence, 0.75)
        self.assertEqual(detector._model.kwargs["classes"], [8])
        self.assertEqual(detector._model.kwargs["quantize"], 16)

    def test_roi_and_exclusion_filter_candidate_centers(self) -> None:
        options = VesselDetectionOptions(
            enabled=True,
            roi=((0.0, 0.0), (0.8, 0.0), (0.8, 1.0), (0.0, 1.0)),
            exclude_rois=(
                ((0.4, 0.0), (0.7, 0.0), (0.7, 1.0), (0.4, 1.0)),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "yolo26s.pt"
            model_path.touch()

            class FilterModel(_FakeModel):
                def predict(self, **kwargs: object) -> list[SimpleNamespace]:
                    self.kwargs = kwargs
                    return [
                        _result(
                            [
                                [20, 20, 40, 40],
                                [100, 20, 120, 40],
                                [170, 20, 190, 40],
                            ],
                            [0.8, 0.9, 0.95],
                            [8, 8, 8],
                        )
                    ]

            detector = UltralyticsVesselDetector(
                model_path=model_path,
                device="cpu",
                half=False,
                model_factory=FilterModel,
            )
            candidates = detector.detect(
                np.zeros((100, 200, 3), dtype=np.uint8),
                options,
            )

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].rectangle.left, 0.1)

    def test_temporal_tracker_confirms_and_holds_short_miss(self) -> None:
        options = VesselDetectionOptions(
            enabled=True,
            minimum_hits=2,
            hold_seconds=1.0,
        )
        tracker = VesselTrackManager(options)
        first = VesselCandidate(
            NormalizedRect(0.2, 0.3, 0.1, 0.1),
            0.2,
            8,
        )
        second = VesselCandidate(
            NormalizedRect(0.205, 0.3, 0.1, 0.1),
            0.3,
            8,
        )

        initial = tracker.update([first], timestamp=10.0, inference_ms=5)
        confirmed = tracker.update(
            [second],
            timestamp=10.2,
            inference_ms=6,
        )
        held = tracker.update([], timestamp=11.0, inference_ms=4)
        expired = tracker.update([], timestamp=11.3, inference_ms=4)

        self.assertEqual(initial.count, 0)
        self.assertEqual(confirmed.count, 1)
        self.assertEqual(confirmed.detections[0].object_id, 1)
        self.assertEqual(held.count, 1)
        self.assertEqual(expired.count, 0)

    def test_cross_region_containment_deduplicates_nested_boxes(self) -> None:
        large = VesselCandidate(
            NormalizedRect(0.10, 0.10, 0.20, 0.20),
            0.70,
            8,
            source_region=0,
        )
        nested = VesselCandidate(
            NormalizedRect(0.12, 0.12, 0.10, 0.10),
            0.80,
            8,
            source_region=1,
        )

        cross_region = deduplicate_candidates(
            [large, nested],
            iou_threshold=0.45,
            containment_threshold=0.80,
            limit=10,
        )
        same_region = deduplicate_candidates(
            [large, VesselCandidate(
                nested.rectangle,
                nested.confidence,
                nested.class_id,
                source_region=0,
            )],
            iou_threshold=0.45,
            containment_threshold=0.80,
            limit=10,
        )

        self.assertEqual(cross_region, [nested])
        self.assertEqual(len(same_region), 2)

    def test_large_low_confidence_box_is_filtered(self) -> None:
        class LargeBoxModel(_FakeModel):
            def predict(self, **kwargs: object) -> list[SimpleNamespace]:
                self.kwargs = kwargs
                return [
                    _result(
                        [
                            [0, 0, 120, 60],
                            [20, 20, 120, 70],
                            [20, 20, 180, 90],
                        ],
                        [0.24, 0.26, 0.95],
                        [8, 8, 8],
                    )
                ]

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "yolo26s.pt"
            model_path.touch()
            detector = UltralyticsVesselDetector(
                model_path=model_path,
                device="cpu",
                half=False,
                model_factory=LargeBoxModel,
            )
            candidates = detector.detect(
                np.zeros((100, 200, 3), dtype=np.uint8),
                VesselDetectionOptions(maximum_box_area=0.50),
            )

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].confidence, 0.26)

    def test_new_filter_options_round_trip_through_payload(self) -> None:
        original = VesselDetectionOptions(
            duplicate_containment_threshold=0.75,
            large_box_area_threshold=0.30,
            large_box_minimum_confidence=0.35,
            maximum_box_area=0.80,
            proposal_appearance_threshold=22,
            proposal_appearance_blur_pixels=41,
            proposal_border_margin=0.02,
            display_proposals=True,
            proposal_roi=((0.0, 0.4), (1.0, 0.4), (1.0, 0.8), (0.0, 0.8)),
            proposal_minimum_fill_ratio=0.3,
            proposal_minimum_motion_ratio=0.1,
            proposal_maximum_candidates=4,
        )

        restored = VesselDetectionOptions.from_payload(original.to_payload())

        self.assertEqual(restored, original)

    def test_tracker_reset_preserves_monotonic_result_version(self) -> None:
        options = VesselDetectionOptions(minimum_hits=1)
        tracker = VesselTrackManager(options)
        candidate = VesselCandidate(
            NormalizedRect(0.4, 0.4, 0.1, 0.1),
            confidence=0.5,
            class_id=8,
            source_region=0,
        )
        before = tracker.update(
            [candidate],
            timestamp=1.0,
            inference_ms=1.0,
        )

        tracker.reset_tracking()
        after = tracker.update(
            [candidate],
            timestamp=2.0,
            inference_ms=1.0,
        )

        self.assertGreater(after.result_version, before.result_version)
        self.assertEqual(after.detections[0].object_id, 2)

    def test_proposal_roi_limits_candidates_without_shrinking_vessel_roi(self) -> None:
        proposer = TemporalSmallTargetProposer()
        options = VesselDetectionOptions(
            small_target_proposals=True,
            roi=((0.0, 0.2), (1.0, 0.2), (1.0, 1.0), (0.0, 1.0)),
            proposal_roi=((0.0, 0.4), (1.0, 0.4), (1.0, 0.7), (0.0, 0.7)),
            proposal_threshold=255,
            proposal_appearance_threshold=10,
            proposal_minimum_area_pixels=6,
            proposal_maximum_area_pixels=500,
        )
        frame = np.full((100, 160, 3), 128, dtype=np.uint8)
        frame[50:58, 40:52] = 20
        frame[82:90, 100:112] = 20

        candidates = proposer.detect(frame, options)

        self.assertEqual(len(candidates), 1)
        self.assertLess(candidates[0].rectangle.center[1], 0.7)

    def test_motion_ratio_rejects_static_appearance_but_keeps_new_object(self) -> None:
        options = VesselDetectionOptions(
            small_target_proposals=True,
            proposal_threshold=15,
            proposal_appearance_threshold=10,
            proposal_minimum_area_pixels=6,
            proposal_maximum_area_pixels=500,
            proposal_minimum_motion_ratio=0.1,
        )
        plain = np.full((100, 160, 3), 128, dtype=np.uint8)
        object_frame = plain.copy()
        object_frame[50:58, 80:92] = 20

        static_proposer = TemporalSmallTargetProposer()
        self.assertEqual(static_proposer.detect(object_frame, options), [])

        moving_proposer = TemporalSmallTargetProposer()
        self.assertEqual(moving_proposer.detect(plain, options), [])
        self.assertEqual(len(moving_proposer.detect(object_frame, options)), 1)

    def test_temporal_small_target_proposer_finds_motion_in_water_roi(self) -> None:
        proposer = TemporalSmallTargetProposer()
        options = VesselDetectionOptions(
            small_target_proposals=True,
            roi=((0.0, 0.4), (1.0, 0.4), (1.0, 1.0), (0.0, 1.0)),
            proposal_threshold=15,
            proposal_minimum_area_pixels=6,
            proposal_maximum_area_pixels=500,
        )
        background = np.full((100, 160, 3), 128, dtype=np.uint8)
        changed = background.copy()
        changed[60:68, 80:92] = 20

        self.assertEqual(proposer.detect(background, options), [])
        candidates = proposer.detect(changed, options)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0].class_id,
            SMALL_TARGET_PROPOSAL_CLASS_ID,
        )
        self.assertGreater(candidates[0].rectangle.center[1], 0.4)

    def test_small_target_appearance_survives_background_adaptation(self) -> None:
        proposer = TemporalSmallTargetProposer()
        options = VesselDetectionOptions(
            small_target_proposals=True,
            roi=((0.0, 0.4), (1.0, 0.4), (1.0, 1.0), (0.0, 1.0)),
            proposal_threshold=255,
            proposal_appearance_enabled=True,
            proposal_appearance_threshold=10,
            proposal_appearance_blur_pixels=21,
            proposal_minimum_area_pixels=6,
            proposal_maximum_area_pixels=500,
        )
        frame = np.full((100, 160, 3), 128, dtype=np.uint8)
        frame[60:68, 80:92] = 20

        initial = proposer.detect(frame, options)
        latest = initial
        for _ in range(100):
            latest = proposer.detect(frame, options)

        self.assertEqual(len(initial), 1)
        self.assertEqual(len(latest), 1)
        self.assertEqual(
            latest[0].class_id,
            SMALL_TARGET_PROPOSAL_CLASS_ID,
        )

    def test_small_target_border_margin_rejects_edge_noise(self) -> None:
        proposer = TemporalSmallTargetProposer()
        options = VesselDetectionOptions(
            small_target_proposals=True,
            proposal_threshold=255,
            proposal_appearance_threshold=10,
            proposal_minimum_area_pixels=6,
            proposal_maximum_area_pixels=500,
            proposal_border_margin=0.05,
        )
        frame = np.full((100, 160, 3), 128, dtype=np.uint8)
        frame[50:58, 0:8] = 20

        self.assertEqual(proposer.detect(frame, options), [])

    def test_confirmed_yolo_box_wins_over_overlapping_motion_proposal(self) -> None:
        rectangle = NormalizedRect(0.4, 0.4, 0.1, 0.1)
        proposal = VesselCandidate(
            rectangle,
            confidence=0.9,
            class_id=SMALL_TARGET_PROPOSAL_CLASS_ID,
            source_region=-1,
        )
        confirmed = VesselCandidate(
            rectangle,
            confidence=0.2,
            class_id=8,
            source_region=0,
        )

        selected = deduplicate_candidates(
            [proposal, confirmed],
            iou_threshold=0.45,
            containment_threshold=0.8,
            limit=10,
        )

        self.assertEqual(selected, [confirmed])

    def test_invalid_regions_and_model_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "推理区域"):
            VesselDetectionOptions(
                inference_regions=((0.5, 0.1, 0.4, 0.8),)
            ).validate()
        with self.assertRaisesRegex(ValueError, "pt文件名"):
            VesselDetectionOptions(model="../model.pt").validate()
        with self.assertRaisesRegex(ValueError, "maximum_box_area"):
            VesselDetectionOptions(
                large_box_area_threshold=0.5,
                maximum_box_area=0.4,
            ).validate()
        with self.assertRaisesRegex(ValueError, "奇数"):
            VesselDetectionOptions(
                proposal_appearance_blur_pixels=20,
            ).validate()


if __name__ == "__main__":
    unittest.main()
