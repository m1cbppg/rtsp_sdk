from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from rtsp_annotator.shared_stream_manager import SharedStreamManager
from rtsp_annotator.stream_manager import ManagerSettings, StreamSpec


class FakeDetector:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeRegistry:
    def __init__(self) -> None:
        self.acquired: list[Any] = []
        self.detectors: list[FakeDetector] = []
        self.closed = False

    def acquire(self, settings: Any) -> FakeDetector:
        self.acquired.append(settings)
        detector = FakeDetector()
        self.detectors.append(detector)
        return detector

    def close(self) -> None:
        self.closed = True


def make_manager_settings(model_root: Path) -> ManagerSettings:
    return ManagerSettings(
        model_root=model_root,
        internal_rtsp_base_url="rtsp://mediamtx:8554",
        public_rtsp_base_url="rtsp://example.com:38554",
        publish_user="publisher",
        publish_password="publish-password",
        read_user="viewer",
        read_password="read-password",
        device="cuda:0",
        half=True,
        encoder="h264_nvenc",
        encoder_preset="p4",
        max_batch_size=4,
        batch_wait_ms=2,
        max_streams=2,
        startup_grace_seconds=0,
    )


class SharedStreamManagerTests(unittest.TestCase):
    def test_sessions_share_registry_and_expose_per_stream_metrics(self) -> None:
        registry = FakeRegistry()
        started = threading.Event()
        starts = 0
        starts_lock = threading.Lock()

        def fake_run_pipeline(
            settings: Any,
            detector_factory: Any,
            *,
            external_stop_event: threading.Event,
            stream_id: str,
            stats_callback: Any,
        ) -> None:
            nonlocal starts
            detector_factory(settings)
            stats_callback(
                {
                    "capture_fps": 25.0,
                    "inference_fps": 24.0,
                    "publish_fps": 25.0,
                    "average_inference_ms": 27.0,
                    "average_frame_age_ms": 60.0,
                    "interval_inference_skipped": 10,
                    "total_inference_skipped": 10,
                    "total_detections": 5,
                    "capture_reconnects": 0,
                    "publisher_restarts": 0,
                }
            )
            with starts_lock:
                starts += 1
                if starts == 2:
                    started.set()
            external_stop_event.wait(2)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Path(root, "model.pt").touch()
            manager = SharedStreamManager(
                make_manager_settings(root),
                registry=registry,  # type: ignore[arg-type]
            )
            with patch(
                "rtsp_annotator.shared_stream_manager.run_pipeline",
                side_effect=fake_run_pipeline,
            ):
                first = manager.create(
                    StreamSpec("rtsp://camera/one", model="model.pt")
                )
                second = manager.create(
                    StreamSpec("rtsp://camera/two", model="model.pt")
                )
                self.assertTrue(started.wait(1))
                first_state = manager.get(first["stream_id"])
                second_state = manager.get(second["stream_id"])
                stopped = manager.stop(first["stream_id"])
                manager.stop(second["stream_id"])
                manager.shutdown()

        self.assertEqual(len(registry.acquired), 2)
        self.assertTrue(all(item.model_path.name == "model.pt" for item in registry.acquired))
        self.assertTrue(all(item.encoder == "h264_nvenc" for item in registry.acquired))
        self.assertEqual(first_state["metrics"]["average_inference_ms"], 27.0)
        self.assertEqual(second_state["metrics"]["inference_fps"], 24.0)
        self.assertEqual(stopped["status"], "stopped")
        self.assertTrue(all(detector.closed for detector in registry.detectors))
        self.assertTrue(registry.closed)


if __name__ == "__main__":
    unittest.main()
