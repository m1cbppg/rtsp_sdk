from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

from rtsp_annotator.ground_litter_sampling import (
    ReplayWindow, load_windows, parse_time, sample_times,
)


class GroundLitterSamplingTests(unittest.TestCase):
    def test_parse_time_attaches_camera_timezone(self):
        value = parse_time("2026-09-11T08:00:00")
        self.assertEqual(value.tzinfo.key, "Asia/Shanghai")

    def test_sample_times_are_deterministic(self):
        window = ReplayWindow(
            "44180209031322001021",
            parse_time("2026-09-11T08:00:00"),
            parse_time("2026-09-11T08:00:05"),
            "day",
        )
        values = list(sample_times(window, 1))
        self.assertEqual(len(values), 5)
        self.assertEqual(values[0].second, 0)
        self.assertEqual(values[-1].second, 4)

    def test_invalid_window_is_rejected(self):
        with self.assertRaises(ValueError):
            ReplayWindow("bad", parse_time("2026-09-11T08:00:00"),
                         parse_time("2026-09-11T08:00:01"), "day").validate()
        with self.assertRaises(ValueError):
            ReplayWindow("44180209031322001021", parse_time("2026-09-11T08:00:01"),
                         parse_time("2026-09-11T08:00:00"), "day").validate()

    def test_load_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sampling.json"
            path.write_text(json.dumps({"windows": [{
                "device_code": "44180209031322001021",
                "start": "2026-09-11T08:00:00",
                "end": "2026-09-11T08:01:00",
                "mode": "day",
                "label": "quiet",
            }]}), encoding="utf-8")
            windows = load_windows(path)
            self.assertEqual(windows[0].label, "quiet")


if __name__ == "__main__":
    unittest.main()
