import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from rtsp_annotator.ground_litter_replay import sample_video, sample_stream


class GroundLitterReplayTests(unittest.TestCase):
    def make_video(self, directory: str) -> Path:
        path = Path(directory) / "sample.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(10):
            writer.write(np.full((48, 64, 3), index * 10, np.uint8))
        writer.release()
        return path

    def test_local_video_sampling_writes_bounded_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_video(directory)
            output = Path(directory) / "frames"
            result = sample_video(source, output, sample_fps=2, max_frames=3)
            self.assertEqual(result.frames, 3)
            self.assertEqual(len(list(output.glob("frame-*.jpg"))), 3)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertNotIn("token", (output / "manifest.json").read_text())

    def test_stream_sampler_rejects_non_rtsp_before_opening(self):
        with self.assertRaises(ValueError):
            sample_stream("https://example.test/live", tempfile.mkdtemp())

    def test_stream_sampler_rejects_negative_start_offset(self):
        with self.assertRaises(ValueError):
            sample_stream("rtsp://example.test/live", tempfile.mkdtemp(), start_offset_seconds=-1)


if __name__ == "__main__":
    unittest.main()
