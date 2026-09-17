from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from rtsp_annotator.ground_litter_batch import (
    BatchOptions,
    parse_inputs,
    _review_page,
    iter_video_frames,
)


class GroundLitterBatchTests(unittest.TestCase):
    def test_input_parser_requires_device_equals_video(self):
        with self.assertRaises(ValueError):
            parse_inputs(["bad"])
        with self.assertRaises(ValueError):
            parse_inputs(["123=video.mp4"])
        parsed = parse_inputs(["44180209031322001030=video.ps"])
        self.assertEqual(parsed["44180209031322001030"], Path("video.ps"))

    def test_video_iterator_is_bounded_and_time_based(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
            self.assertTrue(writer.isOpened())
            for index in range(20):
                writer.write(np.full((24, 32, 3), index, np.uint8))
            writer.release()
            frames = list(iter_video_frames(
                source, BatchOptions(sample_fps=2, max_frames=3)))
            self.assertEqual([item.index for item in frames], [0, 2, 4])
            self.assertEqual([round(item.timestamp, 2) for item in frames],
                             [0.0, 0.4, 0.8])

    def test_batch_options_reject_unbounded_values(self):
        with self.assertRaises(ValueError):
            BatchOptions(sample_fps=0).validate()
        with self.assertRaises(ValueError):
            BatchOptions(max_candidate_frames=0).validate()

    def test_review_page_contains_only_local_event_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _review_page(output, [{
                "item_id": "abc123",
                "camera_id": "camera",
                "region_id": "zone",
                "first_seen_seconds": 1.0,
                "confirmed_image": "evidence/abc/confirmed.jpg",
            }], [{
                "review_id": "frame-00000001",
                "source_time_seconds": 1.0,
                "image": "candidate_frames/frame.jpg",
                "candidates": [{"label": "Paper"}],
            }])
            page = (output / "review.html").read_text(encoding="utf-8")
            self.assertIn("labels.json", page)
            self.assertIn("evidence/abc/confirmed.jpg", page)
            self.assertIn("candidate_frames/frame.jpg", page)
            self.assertIn("reviewItems", page)
            self.assertIn('input:checked', page)
            self.assertNotIn('name=""+n+""', page)
            self.assertNotIn("rtsp://", page)


if __name__ == "__main__":
    unittest.main()
