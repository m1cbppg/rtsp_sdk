from __future__ import annotations

import unittest

from rtsp_annotator.roi_selector import format_normalized_roi


class RoiSelectorTests(unittest.TestCase):
    def test_pixel_points_are_formatted_as_normalized_roi(self) -> None:
        value = format_normalized_roi(
            [(0, 0), (199, 0), (199, 99), (0, 99)],
            width=200,
            height=100,
        )
        self.assertEqual(value, "0,0;1,0;1,1;0,1")

    def test_selector_requires_three_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "至少需要 3 个点"):
            format_normalized_roi(
                [(0, 0), (10, 10)],
                width=100,
                height=100,
            )


if __name__ == "__main__":
    unittest.main()
