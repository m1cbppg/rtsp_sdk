from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.vessel_calibration import (
    build_consensus_water_mask,
    expand_water_mask_to_lower_envelope,
    find_review_exclusion_candidates,
    propose_inference_regions,
    _sample_video_frames_with_opencv,
    water_mask_to_polygon,
)
from rtsp_annotator.vessel_detection import VesselCandidate


class VesselCalibrationTests(unittest.TestCase):
    def test_opencv_fallback_samples_by_elapsed_time(self) -> None:
        class FakeCapture:
            def __init__(self, *_args: object) -> None:
                self.index = 0
                self.released = False

            def isOpened(self) -> bool:
                return True

            def get(self, property_id: int) -> float:
                if property_id == 1:
                    return 2.0
                if property_id == 2:
                    return self.index * 500.0
                return 0.0

            def read(self) -> tuple[bool, np.ndarray | None]:
                if self.index >= 8:
                    return False, None
                frame = np.full((2, 3, 3), self.index, dtype=np.uint8)
                self.index += 1
                return True, frame

            def release(self) -> None:
                self.released = True

        fake_cv2 = SimpleNamespace(
            CAP_FFMPEG=0,
            CAP_PROP_FPS=1,
            CAP_PROP_POS_MSEC=2,
            CAP_PROP_OPEN_TIMEOUT_MSEC=3,
            CAP_PROP_READ_TIMEOUT_MSEC=4,
            VideoCapture=FakeCapture,
        )
        with patch.dict("sys.modules", {"cv2": fake_cv2}):
            frames = _sample_video_frames_with_opencv(
                "sample.mp4",
                sample_count=3,
                sample_interval_seconds=1.0,
                transport="tcp",
                open_timeout=10.0,
                read_timeout=15.0,
            )

        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [0, 2, 4])

    def test_consensus_water_mask_produces_normalized_roi_and_zoom(self) -> None:
        masks = []
        for top in (40, 41, 39, 40, 42):
            mask = np.zeros((100, 160), dtype=bool)
            mask[top:, 5:155] = True
            masks.append(mask)

        consensus = build_consensus_water_mask(
            masks,
            minimum_ratio=0.60,
            close_ratio=0,
            dilation_ratio=0,
        )
        polygon = water_mask_to_polygon(consensus)
        regions = propose_inference_regions(polygon)

        self.assertGreater(consensus.mean(), 0.50)
        self.assertTrue(all(0 <= value <= 1 for point in polygon for value in point))
        self.assertGreaterEqual(len(polygon), 4)
        self.assertEqual(regions[0], (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(len(regions), 2)
        self.assertLess(regions[1][1], 0.45)

    def test_stable_boxes_are_review_only_but_moving_ship_is_not(self) -> None:
        frames: list[list[VesselCandidate]] = []
        for index in range(5):
            frames.append(
                [
                    VesselCandidate(
                        NormalizedRect(
                            0.20 + index * 0.001,
                            0.30,
                            0.10,
                            0.08,
                        ),
                        0.40 + index * 0.01,
                        8,
                    ),
                    VesselCandidate(
                        NormalizedRect(
                            0.50 + index * 0.05,
                            0.60,
                            0.08,
                            0.06,
                        ),
                        0.70,
                        8,
                    ),
                ]
            )

        review = find_review_exclusion_candidates(frames)

        self.assertEqual(len(review), 1)
        self.assertEqual(review[0].frame_coverage, 1.0)
        self.assertAlmostEqual(review[0].median_confidence, 0.42)
        self.assertLess(review[0].center_motion, 0.015)

    def test_water_envelope_fills_dense_marina_occlusions(self) -> None:
        water = np.zeros((100, 200), dtype=bool)
        water[35:, :110] = True
        water[45:, 110:140] = True
        water[35:, 140:160] = True
        # The right side and a central marina are unlabeled because boats and
        # docks obscure the water; a literal mask would miss those vessels.
        water[:, 65:85] = False
        water[:, 160:] = False

        envelope = expand_water_mask_to_lower_envelope(
            water,
            smoothing_ratio=0,
            upward_margin_ratio=0,
        )

        self.assertTrue(envelope[50, 75])
        self.assertTrue(envelope[50, 190])
        self.assertFalse(envelope[20, 190])

    def test_short_lived_box_is_not_an_exclusion_candidate(self) -> None:
        frames = [
            [VesselCandidate(NormalizedRect(0.1, 0.2, 0.1, 0.1), 0.8, 8)],
            [],
            [],
            [],
        ]

        self.assertEqual(find_review_exclusion_candidates(frames), [])


if __name__ == "__main__":
    unittest.main()
