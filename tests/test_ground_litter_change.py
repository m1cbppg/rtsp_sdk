import unittest

import cv2
import numpy as np

from rtsp_annotator.ground_litter_change import ChangeConfig, compare_patch


class GroundLitterChangeTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.full((120, 160, 3), 100, dtype=np.uint8)

    def test_local_object_change_is_detected(self):
        current = self.reference.copy()
        current[45:75, 65:95] = 220
        evidence = compare_patch(self.reference, current, (40, 20, 120, 100),
                                  ChangeConfig(minimum_component_area=20))
        self.assertTrue(evidence.changed)
        self.assertGreater(evidence.change_fraction, 0.02)

    def test_global_brightness_change_is_removed(self):
        current = np.full_like(self.reference, 125)
        evidence = compare_patch(self.reference, current, (40, 20, 120, 100))
        self.assertFalse(evidence.changed)
        self.assertLess(evidence.change_fraction, 0.02)

    def test_invalid_box_and_image_are_rejected(self):
        with self.assertRaises(ValueError):
            compare_patch(self.reference, self.reference, (20, 20, 10, 30))
        with self.assertRaises(ValueError):
            compare_patch(self.reference, np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5))

    def test_config_rejects_even_kernel(self):
        with self.assertRaises(ValueError):
            ChangeConfig(blur_kernel=2).validate()


if __name__ == "__main__":
    unittest.main()
