import unittest
import numpy as np

from scripts.probe_ground_litter_masks import support_fraction


class SupportFractionTests(unittest.TestCase):
    def test_fraction_is_intersection_over_instance_area(self):
        instance = np.zeros((4, 4), np.uint8); instance[1:3, 1:3] = 1
        ground = np.zeros((4, 4), np.uint8); ground[2:, 1:] = 1
        self.assertAlmostEqual(support_fraction(instance, ground), .5)

    def test_empty_mask_is_unknown(self):
        self.assertIsNone(support_fraction(np.zeros((2, 2)), np.ones((2, 2))))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            support_fraction(np.zeros((2, 2)), np.zeros((2, 3)))
