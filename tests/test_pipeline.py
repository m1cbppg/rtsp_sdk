from __future__ import annotations

import tempfile
import queue
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from rtsp_annotator.config import Settings
from rtsp_annotator.pipeline import (
    DetectionBox,
    DetectionOverlayStore,
    DetectionSnapshot,
    LatestFrameSlot,
    DetectionWorker,
    PipelineStats,
    PublishWorker,
    ResolvedDevice,
    SourceState,
    YoloDetector,
    box_indices_inside_roi,
    build_av_options,
    build_ffmpeg_command,
    draw_detection_boxes,
    normalized_roi_to_pixels,
    point_in_polygon,
    resolve_inference_device,
    validate_inference_precision,
)
from rtsp_annotator.labels import (
    chinese_label,
    load_label_map,
    translated_names,
)


def make_settings(model: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "input_url": "rtsp://camera/live",
        "output_url": "rtsp://localhost:8554/detected",
        "model_path": model,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


class LatestFrameSlotTests(unittest.TestCase):
    def test_only_latest_frame_is_returned(self) -> None:
        slot = LatestFrameSlot()
        first = slot.publish("frame-1", captured_at=1.0)
        second = slot.publish("frame-2", captured_at=2.0)

        version, packet = slot.wait_after(0, timeout=0)

        self.assertEqual(version, 2)
        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.frame, "frame-2")
        self.assertEqual(first.source_sequence, 1)
        self.assertEqual(second.source_sequence, 2)

    def test_annotated_frame_preserves_source_sequence(self) -> None:
        slot = LatestFrameSlot()
        packet = slot.publish(
            "annotated",
            captured_at=3.0,
            source_sequence=99,
        )
        self.assertEqual(packet.source_sequence, 99)


class SourceStateTests(unittest.TestCase):
    def test_invalid_source_fps_uses_fallback(self) -> None:
        state = SourceState(fallback_fps=25.0)
        info = state.update(width=1920, height=1080, fps=0.0)
        self.assertEqual(info.fps, 25.0)

    def test_pyav_options_disable_rtsp_buffering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory, "model.pt")
            model.touch()
            settings = make_settings(model, input_rtsp_transport="udp")
            options = build_av_options(settings)
        self.assertEqual(options["rtsp_transport"], "udp")
        self.assertEqual(options["fflags"], "nobuffer")
        self.assertEqual(options["flags"], "low_delay")
        self.assertEqual(options["max_delay"], "0")
        self.assertEqual(options["reorder_queue_size"], "0")


class DeviceResolutionTests(unittest.TestCase):
    class FakeCuda:
        def __init__(self, available: bool, names: list[str]) -> None:
            self._available = available
            self._names = names

        def is_available(self) -> bool:
            return self._available

        def device_count(self) -> int:
            return len(self._names)

        def get_device_name(self, index: int) -> str:
            return self._names[index]

    class FakeMpsBackend:
        def __init__(self, available: bool) -> None:
            self._available = available

        def is_built(self) -> bool:
            return self._available

        def is_available(self) -> bool:
            return self._available

    class FakeMpsRuntime:
        @staticmethod
        def get_name() -> str:
            return "Apple MPS"

    @classmethod
    def fake_torch(
        cls,
        *,
        cuda_available: bool,
        cuda_names: list[str] | None = None,
        mps_available: bool,
    ) -> object:
        class Backends:
            mps = cls.FakeMpsBackend(mps_available)

        class Torch:
            cuda = cls.FakeCuda(cuda_available, cuda_names or [])
            backends = Backends()
            mps = cls.FakeMpsRuntime()

        return Torch()

    def test_auto_prefers_cuda_over_mps(self) -> None:
        device = resolve_inference_device(
            "auto",
            self.fake_torch(
                cuda_available=True,
                cuda_names=["NVIDIA RTX"],
                mps_available=True,
            ),
        )
        self.assertEqual(
            device,
            ResolvedDevice("cuda:0", "cuda", "NVIDIA RTX"),
        )

    def test_auto_uses_mps_when_cuda_is_unavailable(self) -> None:
        device = resolve_inference_device(
            "auto",
            self.fake_torch(cuda_available=False, mps_available=True),
        )
        self.assertEqual(device, ResolvedDevice("mps", "mps", "Apple MPS"))

    def test_cuda_alias_and_gpu_index_are_resolved(self) -> None:
        torch = self.fake_torch(
            cuda_available=True,
            cuda_names=["GPU 0", "GPU 1"],
            mps_available=False,
        )
        self.assertEqual(resolve_inference_device("cuda", torch).value, "cuda:0")
        self.assertEqual(resolve_inference_device("1", torch).value, "cuda:1")

    def test_unavailable_explicit_backend_fails(self) -> None:
        torch = self.fake_torch(cuda_available=False, mps_available=False)
        with self.assertRaisesRegex(RuntimeError, "检测不到可用 CUDA"):
            resolve_inference_device("cuda:0", torch)
        with self.assertRaisesRegex(RuntimeError, "不支持 MPS"):
            resolve_inference_device("mps", torch)

    def test_half_is_only_allowed_on_cuda(self) -> None:
        validate_inference_precision(
            True,
            ResolvedDevice("cuda:0", "cuda", "GPU"),
        )
        with self.assertRaisesRegex(RuntimeError, "仅支持 CUDA"):
            validate_inference_precision(
                True,
                ResolvedDevice("mps", "mps", "Apple MPS"),
            )


class StatsTests(unittest.TestCase):
    def test_stats_accumulate(self) -> None:
        stats = PipelineStats()
        stats.add(
            captured=3,
            inferred=1,
            inference_seconds=0.2,
            published_frame_age_seconds=0.1,
        )
        stats.add(captured=2)
        snapshot = stats.snapshot()
        self.assertEqual(snapshot.captured, 5)
        self.assertEqual(snapshot.inferred, 1)
        self.assertAlmostEqual(snapshot.inference_seconds, 0.2)
        self.assertAlmostEqual(snapshot.published_frame_age_seconds, 0.1)

    def test_latency_samples_are_bounded_and_preserved(self) -> None:
        stats = PipelineStats()
        for index in range(600):
            stats.observe(
                inference_latency_samples=index / 1000,
                frame_age_samples=index / 2000,
                detection_age_samples=index / 3000,
            )
        snapshot = stats.snapshot()
        self.assertEqual(len(snapshot.inference_latency_samples), 512)
        self.assertAlmostEqual(snapshot.inference_latency_samples[-1], 0.599)


class DetectionOverlayStoreTests(unittest.TestCase):
    @staticmethod
    def snapshot(
        box: tuple[float, float, float, float],
        *,
        captured_at: float,
        sequence: int,
    ) -> DetectionSnapshot:
        return DetectionSnapshot(
            boxes=(DetectionBox(box, 0),),
            names={0: "人员"},
            frame_shape=(100, 200),
            source_sequence=sequence,
            captured_at=captured_at,
        )

    def test_boxes_are_extrapolated_between_yolo_results(self) -> None:
        store = DetectionOverlayStore(
            maximum_age_seconds=1,
            maximum_extrapolation_seconds=0.2,
        )
        store.update(
            self.snapshot((10, 10, 30, 50), captured_at=1.0, sequence=1)
        )
        store.update(
            self.snapshot((20, 10, 40, 50), captured_at=1.1, sequence=2)
        )

        projected = store.snapshot_for(1.2)

        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertAlmostEqual(projected.boxes[0].xyxy[0], 26.5)
        self.assertEqual(projected.source_sequence, 2)

    def test_stale_boxes_expire_without_blocking_video(self) -> None:
        store = DetectionOverlayStore(maximum_age_seconds=0.25)
        store.update(
            self.snapshot((10, 10, 30, 50), captured_at=1.0, sequence=1)
        )

        expired = store.snapshot_for(1.5)

        self.assertIsNotNone(expired)
        assert expired is not None
        self.assertEqual(expired.boxes, ())


class RoiTests(unittest.TestCase):
    def test_normalized_roi_is_scaled_to_frame(self) -> None:
        polygon = normalized_roi_to_pixels(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            width=1920,
            height=1080,
        )
        self.assertEqual(
            polygon,
            ((0, 0), (1919, 0), (1919, 1079), (0, 1079)),
        )

    def test_point_inside_outside_and_on_boundary(self) -> None:
        polygon = ((10, 10), (90, 10), (90, 90), (10, 90))
        self.assertTrue(point_in_polygon((50, 50), polygon))
        self.assertTrue(point_in_polygon((10, 50), polygon))
        self.assertFalse(point_in_polygon((5, 50), polygon))

    def test_box_filter_uses_box_center(self) -> None:
        polygon = ((20, 20), (80, 20), (80, 80), (20, 80))
        boxes = [
            [30, 30, 50, 70],
            [0, 0, 10, 10],
            [70, 70, 100, 100],
        ]
        self.assertEqual(box_indices_inside_roi(boxes, polygon), [0])


class WorkerFailureTests(unittest.TestCase):
    def test_detector_failure_is_reported_and_stops_pipeline(self) -> None:
        class FailingDetector:
            def annotate(self, frame: object) -> tuple[object, int]:
                raise RuntimeError("inference failed")

        input_slot = LatestFrameSlot()
        output_slot = LatestFrameSlot()
        stats = PipelineStats()
        stop_event = threading.Event()
        errors: queue.Queue[BaseException] = queue.Queue()
        worker = DetectionWorker(
            FailingDetector(),  # type: ignore[arg-type]
            input_slot,
            output_slot,
            stats,
            stop_event,
            errors,
        )
        input_slot.publish("frame", captured_at=1.0)

        worker.start()
        worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertTrue(stop_event.is_set())
        self.assertIsInstance(errors.get_nowait(), RuntimeError)

    def test_publisher_failure_is_reported_and_stops_pipeline(self) -> None:
        class FakeFrame:
            shape = (10, 10, 3)

        class FailingPublisher:
            def is_running(self) -> bool:
                return True

            def start(self, width: int, height: int, fps: float) -> None:
                pass

            def write(self, *args: object, **kwargs: object) -> None:
                raise ValueError("invalid frame")

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory, "model.pt")
            model.touch()
            settings = make_settings(model)
            input_slot = LatestFrameSlot()
            source_state = SourceState(fallback_fps=25.0)
            stats = PipelineStats()
            stop_event = threading.Event()
            errors: queue.Queue[BaseException] = queue.Queue()
            worker = PublishWorker(
                settings,
                input_slot,
                source_state,
                stats,
                stop_event,
                errors,
            )
            worker._publisher = FailingPublisher()  # type: ignore[assignment]
            input_slot.publish(FakeFrame(), captured_at=1.0)

            worker.start()
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertTrue(stop_event.is_set())
        self.assertIsInstance(errors.get_nowait(), ValueError)


class DetectorArgumentsTests(unittest.TestCase):
    def test_fp16_quantization_is_not_passed_when_disabled(self) -> None:
        import numpy as np

        class FakeResult:
            boxes: list[object] = []

            def plot(self, **kwargs: object) -> str:
                return "annotated"

        class FakeModel:
            kwargs: dict[str, object] = {}

            def predict(self, **kwargs: object) -> list[FakeResult]:
                self.kwargs = kwargs
                return [FakeResult()]

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory, "model.pt")
            model_path.touch()
            detector = YoloDetector.__new__(YoloDetector)
            detector._model = FakeModel()
            detector._settings = make_settings(model_path, half=False)

            with patch(
                "rtsp_annotator.pipeline.draw_detection_boxes",
                return_value="annotated",
            ):
                annotated, count = detector.annotate(
                    np.zeros((10, 10, 3), dtype=np.uint8)
                )

        self.assertEqual(annotated, "annotated")
        self.assertEqual(count, 0)
        self.assertNotIn("quantize", detector._model.kwargs)

    def test_cuda_device_and_fp16_are_passed_to_ultralytics(self) -> None:
        import numpy as np

        class FakeResult:
            boxes: list[object] = []

            def plot(self, **kwargs: object) -> str:
                return "annotated"

        class FakeModel:
            kwargs: dict[str, object] = {}

            def predict(self, **kwargs: object) -> list[FakeResult]:
                self.kwargs = kwargs
                return [FakeResult()]

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory, "model.pt")
            model_path.touch()
            detector = YoloDetector.__new__(YoloDetector)
            detector._model = FakeModel()
            detector._settings = make_settings(model_path)
            detector._device = "cuda:0"
            detector._half = True

            with patch(
                "rtsp_annotator.pipeline.draw_detection_boxes",
                return_value="annotated",
            ):
                detector.annotate(
                    np.zeros((10, 10, 3), dtype=np.uint8)
                )

        self.assertEqual(detector._model.kwargs["device"], "cuda:0")
        self.assertEqual(detector._model.kwargs["quantize"], 16)

    def test_roi_filters_boxes_before_plotting(self) -> None:
        class FakeFrame:
            shape = (100, 200, 3)

        class FakeBoxes:
            def __init__(self, rows: list[list[float]]) -> None:
                self.xyxy = rows
                self.cls = [0 for _ in rows]

            def __len__(self) -> int:
                return len(self.xyxy)

            def __getitem__(self, indices: list[int]) -> "FakeBoxes":
                return FakeBoxes([self.xyxy[index] for index in indices])

        class FakeResult:
            def __init__(self) -> None:
                self.boxes = FakeBoxes(
                    [
                        [60, 30, 100, 70],
                        [0, 0, 20, 20],
                    ]
                )

            def plot(self, **kwargs: object) -> str:
                return "annotated"

        class FakeModel:
            def predict(self, **kwargs: object) -> list[FakeResult]:
                return [FakeResult()]

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory, "model.pt")
            model_path.touch()
            detector = YoloDetector.__new__(YoloDetector)
            detector._model = FakeModel()
            detector._settings = make_settings(
                model_path,
                roi=((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)),
            )

            with patch(
                "rtsp_annotator.pipeline.draw_roi_boundary",
                side_effect=lambda frame, polygon, line_width: frame,
            ) as draw, patch(
                "rtsp_annotator.pipeline.draw_detection_boxes",
                return_value="annotated",
            ):
                annotated, count = detector.annotate(FakeFrame())

        self.assertEqual(annotated, "annotated")
        self.assertEqual(count, 1)
        draw.assert_called_once()


class ChineseLabelTests(unittest.TestCase):
    def test_coco_names_are_translated_and_unknown_names_stay_chinese(self) -> None:
        self.assertEqual(chinese_label(0, "person"), "人员")
        self.assertEqual(chinese_label(2, "car"), "汽车")
        self.assertEqual(chinese_label(7, "custom-object"), "类别7")
        self.assertEqual(chinese_label(3, "安全帽"), "安全帽")

    def test_custom_mapping_accepts_original_name_and_class_id(self) -> None:
        translated = translated_names(
            {0: "worker", 1: "helmet"},
            {"worker": "工人", "1": "安全帽"},
        )
        self.assertEqual(translated, {0: "工人", 1: "安全帽"})

    def test_label_map_requires_chinese_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory, "valid.json")
            valid.write_text('{"helmet": "安全帽"}', encoding="utf-8")
            self.assertEqual(load_label_map(valid)["helmet"], "安全帽")

            invalid = Path(directory, "invalid.json")
            invalid.write_text('{"helmet": "helmet"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "必须包含中文"):
                load_label_map(invalid)

    def test_box_renderer_never_reads_confidence(self) -> None:
        import numpy as np

        class FakeBoxes:
            xyxy = [[5, 5, 30, 30]]
            cls = [0]

            def __len__(self) -> int:
                return 1

            @property
            def conf(self) -> object:
                raise AssertionError("绘制链路不应读取置信度")

        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        annotated = draw_detection_boxes(
            frame,
            FakeBoxes(),
            {0: "人员"},
            show_labels=False,
            font_path=None,
            line_width=2,
        )
        self.assertEqual(annotated.shape, frame.shape)
        self.assertTrue(np.any(annotated != frame))


class FFmpegCommandTests(unittest.TestCase):
    def test_command_contains_low_latency_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory, "model.pt")
            model.touch()
            settings = make_settings(
                model,
                bitrate="4M",
                gop_seconds=1.0,
                output_rtsp_transport="tcp",
            )

            command = build_ffmpeg_command(settings, 1920, 1080, 25.0)

        self.assertIn("zerolatency", command)
        self.assertEqual(
            command[command.index("-use_wallclock_as_timestamps") + 1],
            "1",
        )
        self.assertIn("nobuffer", command)
        self.assertIn("ultrafast", command)
        self.assertIn("yuv420p", command)
        self.assertEqual(command[command.index("-g") + 1], "25")
        self.assertEqual(command[command.index("-bf") + 1], "0")
        self.assertEqual(command[command.index("-rtsp_transport") + 1], "tcp")
        self.assertEqual(command[-1], settings.output_url)

    def test_nvenc_command_uses_low_latency_hardware_encoder_settings(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory, "model.pt")
            model.touch()
            settings = make_settings(
                model,
                encoder="h264_nvenc",
                preset="p4",
                bitrate="4M",
            )

            command = build_ffmpeg_command(settings, 1920, 1080, 25.0)

        self.assertEqual(command[command.index("-c:v") + 1], "h264_nvenc")
        self.assertEqual(command[command.index("-preset") + 1], "p4")
        self.assertEqual(command[command.index("-tune") + 1], "ll")
        self.assertEqual(command[command.index("-rc") + 1], "cbr")
        self.assertEqual(command[command.index("-rc-lookahead") + 1], "0")
        self.assertEqual(command[command.index("-delay") + 1], "0")
        self.assertEqual(command[command.index("-zerolatency") + 1], "1")
        self.assertNotIn("-sc_threshold", command)


if __name__ == "__main__":
    unittest.main()
