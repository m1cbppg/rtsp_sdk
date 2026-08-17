from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from rtsp_annotator.stream_observability import (
    StreamHealthEvaluator,
    StreamLogStore,
    encode_sse,
)


class StreamLogStoreTests(unittest.TestCase):
    def test_entries_are_per_stream_ordered_and_credentials_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StreamLogStore(Path(directory), max_file_bytes=4096)
            first = store.append(
                "stream_1",
                level="INFO",
                event="stream.created",
                message="连接 rtsp://admin:secret@camera/live",
                details={"url": "rtsp://viewer:password@example/live"},
            )
            second = store.append(
                "stream_1",
                level="WARNING",
                event="playback.health",
                message="帧率偏低",
            )
            store.append(
                "stream_2",
                level="INFO",
                event="stream.created",
                message="other",
            )

            entries = store.read("stream_1", after_sequence=first["sequence"])

        self.assertEqual([item["sequence"] for item in entries], [second["sequence"]])
        self.assertEqual(entries[0]["stream_id"], "stream_1")
        serialized = str(first)
        self.assertNotIn("admin", serialized)
        self.assertNotIn("secret", serialized)
        self.assertNotIn("password", serialized)
        self.assertIn("***:***@camera", serialized)

    def test_sse_contains_resumable_sequence_and_json_payload(self) -> None:
        entry = {
            "sequence": 7,
            "stream_id": "abc",
            "message": "播放正常",
        }

        encoded = encode_sse(entry)

        self.assertIn("id: 7\n", encoded)
        self.assertIn("event: stream-log\n", encoded)
        self.assertIn('"stream_id":"abc"', encoded)


class StreamHealthEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = StreamHealthEvaluator(stale_after_seconds=10)

    def test_corrupt_frame_is_reported_as_mosaic_risk(self) -> None:
        diagnosis = self.evaluator.diagnose(
            {
                "status": "running",
                "metrics": {
                    "metrics_updated_at_unix": 100,
                    "capture_fps": 25,
                    "publish_fps": 25,
                    "interval_corrupt_frames": 1,
                },
            },
            now_unix=101,
        )

        self.assertEqual(diagnosis.health, "mosaic_risk")
        self.assertEqual(diagnosis.level, "ERROR")

    def test_stale_metrics_are_reported_as_stalled(self) -> None:
        diagnosis = self.evaluator.diagnose(
            {
                "status": "running",
                "metrics": {
                    "metrics_updated_at_unix": 80,
                    "capture_fps": 25,
                    "publish_fps": 25,
                },
            },
            now_unix=100,
        )

        self.assertEqual(diagnosis.health, "stalled")
        self.assertIn("20.0秒", diagnosis.message)

    def test_low_fps_is_degraded_but_not_mosaic(self) -> None:
        diagnosis = self.evaluator.diagnose(
            {
                "status": "running",
                "metrics": {
                    "metrics_updated_at_unix": time.time(),
                    "capture_fps": 12,
                    "publish_fps": 12,
                    "minimum_healthy_fps": 20,
                },
            }
        )

        self.assertEqual(diagnosis.health, "degraded")


if __name__ == "__main__":
    unittest.main()
