import tempfile
import unittest
from pathlib import Path

from rtsp_annotator.ground_litter_profile import BackgroundProfile


class GroundLitterProfileTests(unittest.TestCase):
    def test_background_change_is_disabled_by_default(self):
        profile = BackgroundProfile.from_camera({}, "day")
        self.assertFalse(profile.enabled)
        self.assertIsNone(profile.reference_image)

    def test_enabled_background_requires_reference(self):
        with self.assertRaises(ValueError):
            BackgroundProfile.from_camera({"day": {"background_change": {"enabled": True}}}, "day")

    def test_background_reads_explicit_reference_and_thresholds(self):
        profile = BackgroundProfile.from_camera({"day": {"background_change": {
            "enabled": True,
            "reference_image": "/tmp/clean.jpg",
            "delta_threshold": 30,
            "minimum_change_fraction": .05,
        }}}, "day")
        self.assertTrue(profile.enabled)
        self.assertEqual(profile.reference_image, "/tmp/clean.jpg")
        self.assertEqual(profile.change.delta_threshold, 30)
        self.assertEqual(profile.change.minimum_change_fraction, .05)


if __name__ == "__main__":
    unittest.main()
