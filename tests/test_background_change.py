from __future__ import annotations

import unittest

import numpy as np

from rtsp_annotator.background_change import BackgroundChangeDetector


ROI = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))


class BackgroundChangeDetectorTests(unittest.TestCase):
    def test_local_persistent_change_has_ratio_and_box(self) -> None:
        detector = BackgroundChangeDetector(ROI, width=80, height=40)
        baseline = np.zeros((80, 160, 3), dtype=np.uint8)
        detector.observe(baseline, actor_present=False)
        changed = baseline.copy()
        changed[30:60, 60:100] = 255

        result = detector.observe(changed, actor_present=False)

        self.assertGreater(result.area_ratio, 0.05)
        self.assertEqual(len(result.regions), 1)
        self.assertGreater(result.regions[0].width, 0.1)

    def test_global_brightness_shift_is_compensated(self) -> None:
        detector = BackgroundChangeDetector(ROI, width=80, height=40)
        baseline = np.full((80, 160, 3), 30, dtype=np.uint8)
        detector.observe(baseline, actor_present=False)

        result = detector.observe(
            np.full_like(baseline, 90),
            actor_present=False,
        )

        self.assertEqual(result.area_ratio, 0)

    def test_actor_freezes_baseline_until_change_is_evaluated(self) -> None:
        detector = BackgroundChangeDetector(ROI, width=80, height=40)
        baseline = np.zeros((80, 160, 3), dtype=np.uint8)
        changed = baseline.copy()
        changed[25:60, 50:110] = 255
        detector.observe(baseline, actor_present=False)
        detector.observe(changed, actor_present=True)

        after_leave = detector.observe(changed, actor_present=False)

        self.assertGreater(after_leave.area_ratio, 0.05)

    def test_commit_accepts_confirmed_scene(self) -> None:
        detector = BackgroundChangeDetector(ROI, width=80, height=40)
        baseline = np.zeros((80, 160, 3), dtype=np.uint8)
        changed = baseline.copy()
        changed[25:60, 50:110] = 255
        detector.observe(baseline, actor_present=False)
        detector.commit(changed)

        result = detector.observe(changed, actor_present=False)

        self.assertEqual(result.area_ratio, 0)


if __name__ == "__main__":
    unittest.main()
