from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from rtsp_annotator.config import Settings
from rtsp_annotator.shared_inference import SharedModelRegistry


class FakeResult:
    boxes = None
    names = {0: "person"}


class RecordingModel:
    def __init__(self, _path: str, calls: list[dict[str, Any]]) -> None:
        self.names = {0: "person"}
        self._calls = calls

    def predict(self, **kwargs: Any) -> list[FakeResult]:
        source = kwargs["source"]
        batch_size = len(source) if isinstance(source, list) else 1
        self._calls.append(
            {
                "batch_size": batch_size,
                "conf": kwargs["conf"],
                "iou": kwargs["iou"],
                "imgsz": kwargs["imgsz"],
                "classes": kwargs.get("classes"),
            }
        )
        return [FakeResult() for _ in range(batch_size)]


def make_settings(model: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "input_url": "rtsp://camera/live",
        "output_url": "rtsp://server/detected/test",
        "model_path": model,
        "device": "cpu",
        "half": False,
        "show_labels": False,
        "conf": 0.25,
        "iou": 0.45,
        "imgsz": 640,
        "classes": (0,),
    }
    values.update(overrides)
    return Settings(**values)


class SharedInferenceTests(unittest.TestCase):
    def test_same_model_is_loaded_once_and_compatible_frames_are_batched(
        self,
    ) -> None:
        calls: list[dict[str, Any]] = []
        loads: list[str] = []

        def factory(path: str) -> RecordingModel:
            loads.append(path)
            return RecordingModel(path, calls)

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory, "model.pt")
            model.touch()
            settings = make_settings(model)
            registry = SharedModelRegistry(
                max_batch_size=4,
                batch_wait_ms=20,
                model_factory=factory,
            )
            first = registry.acquire(settings)
            second = registry.acquire(settings)
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            errors: list[BaseException] = []

            def run(detector: Any) -> None:
                try:
                    detector.annotate(frame)
                except BaseException as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=run, args=(first,)),
                threading.Thread(target=run, args=(second,)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)
            metrics = first.metrics()
            first.close()
            second.close()
            registry.close()

        self.assertFalse(errors)
        self.assertEqual(len(loads), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["batch_size"], 2)
        self.assertEqual(calls[0]["classes"], [0])
        self.assertEqual(metrics["shared_model_clients"], 2)
        self.assertEqual(metrics["model_instance_id"], 1)
        self.assertEqual(metrics["model_instance_clients"], 2)
        self.assertEqual(metrics["average_batch_size"], 2.0)
        self.assertEqual(metrics["max_batch_observed"], 2)

    def test_four_streams_are_pinned_two_per_model_instance(self) -> None:
        calls: list[dict[str, Any]] = []
        loads: list[str] = []

        def factory(path: str) -> RecordingModel:
            loads.append(path)
            return RecordingModel(path, calls)

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory, "model.pt")
            model.touch()
            settings = make_settings(model)
            registry = SharedModelRegistry(
                max_batch_size=4,
                batch_wait_ms=2,
                streams_per_model_instance=2,
                model_factory=factory,
            )
            detectors = [registry.acquire(settings) for _ in range(4)]
            metrics = [detector.metrics() for detector in detectors]

            deadline = time.monotonic() + 1
            while len(loads) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)

            detectors[0].close()
            replacement = registry.acquire(settings)
            replacement_metrics = replacement.metrics()
            for detector in detectors[1:]:
                detector.close()
            replacement.close()
            registry.close()

        self.assertEqual(len(loads), 2)
        self.assertEqual(
            [item["model_instance_id"] for item in metrics],
            [1, 1, 2, 2],
        )
        self.assertTrue(
            all(item["model_instance_clients"] == 2 for item in metrics)
        )
        self.assertEqual(replacement_metrics["model_instance_id"], 1)
        self.assertEqual(
            replacement_metrics["model_instance_clients"],
            2,
        )

    def test_single_stream_does_not_wait_for_micro_batch_window(self) -> None:
        calls: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory, "model.pt")
            model.touch()
            settings = make_settings(model)
            registry = SharedModelRegistry(
                max_batch_size=4,
                batch_wait_ms=100,
                model_factory=lambda path: RecordingModel(path, calls),
            )
            detector = registry.acquire(settings)
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            detector.annotate(frame)
            calls.clear()
            started = time.monotonic()
            detector.annotate(frame)
            elapsed = time.monotonic() - started
            detector.close()

        self.assertEqual(calls[0]["batch_size"], 1)
        self.assertLess(elapsed, 0.08)

    def test_different_thresholds_are_not_put_in_the_same_batch(self) -> None:
        calls: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory, "model.pt")
            model.touch()
            registry = SharedModelRegistry(
                max_batch_size=4,
                batch_wait_ms=10,
                model_factory=lambda path: RecordingModel(path, calls),
            )
            first = registry.acquire(make_settings(model, conf=0.25))
            second = registry.acquire(make_settings(model, conf=0.50))
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            threads = [
                threading.Thread(target=first.annotate, args=(frame,)),
                threading.Thread(target=second.annotate, args=(frame,)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)
            first.close()
            second.close()

        self.assertEqual(len(calls), 2)
        self.assertEqual([call["batch_size"] for call in calls], [1, 1])
        self.assertEqual({call["conf"] for call in calls}, {0.25, 0.50})


if __name__ == "__main__":
    unittest.main()
