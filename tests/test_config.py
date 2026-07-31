from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rtsp_annotator.config import Settings, parse_classes, parse_roi, redact_url


class ConfigTests(unittest.TestCase):
    def test_parse_classes(self) -> None:
        self.assertEqual(parse_classes("0, 2,0, 5"), (0, 2, 5))
        self.assertIsNone(parse_classes(""))
        self.assertIsNone(parse_classes(None))

    def test_parse_classes_rejects_negative_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            parse_classes("0,-1")

    def test_parse_roi(self) -> None:
        self.assertEqual(
            parse_roi("0.1,0.2; 0.9,0.2; 0.9,0.8; 0.1,0.8"),
            ((0.1, 0.2), (0.9, 0.2), (0.9, 0.8), (0.1, 0.8)),
        )
        self.assertIsNone(parse_roi(""))
        self.assertIsNone(parse_roi(None))

    def test_parse_roi_rejects_invalid_polygon(self) -> None:
        with self.assertRaisesRegex(ValueError, "至少需要 3 个点"):
            parse_roi("0.1,0.1;0.9,0.9")
        with self.assertRaisesRegex(ValueError, "0 到 1"):
            parse_roi("0,0;1.1,0;0,1")
        with self.assertRaisesRegex(ValueError, "零面积"):
            parse_roi("0,0;0.5,0.5;1,1")

    def test_redact_url_hides_credentials(self) -> None:
        value = redact_url("rtsp://camera-user:secret@10.0.0.8:554/live")
        self.assertEqual(value, "rtsp://***:***@10.0.0.8:554/live")
        self.assertNotIn("secret", value)
        self.assertNotIn("camera-user", value)

    def test_settings_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory, "model.pt")
            model.touch()
            settings = Settings(
                input_url="rtsp://camera/live",
                output_url="rtsp://localhost:8554/detected",
                model_path=model,
            )
            settings.validate()

    def test_settings_reject_same_input_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory, "model.pt")
            model.touch()
            settings = Settings(
                input_url="rtsp://localhost:8554/same",
                output_url="rtsp://localhost:8554/same",
                model_path=model,
            )
            with self.assertRaisesRegex(ValueError, "不能相同"):
                settings.validate()


if __name__ == "__main__":
    unittest.main()
