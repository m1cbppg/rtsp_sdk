from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rtsp_annotator.deepstream_worker import (
    MetricsState,
    OverlayProcessor,
    PlateIdentityTracker,
    StreamPolicy,
    InferenceLatencyTracker,
    _add_pipeline_nodes,
    _point_in_polygon,
    build_inference_config,
    build_lpd_config,
    build_lpr_config,
)


class FakeColor:
    def __new__(
        cls,
        red: float,
        green: float,
        blue: float,
        alpha: float,
    ) -> tuple[float, float, float, float]:
        return red, green, blue, alpha


class FakeLine:
    pass


class FakeDisplayMeta:
    def __init__(self) -> None:
        self.lines: list[FakeLine] = []

    def add_line(self, line: FakeLine) -> None:
        self.lines.append(line)


class FakeBatch:
    def __init__(self, frames: list[object]) -> None:
        self.frame_items = frames
        self.display_meta: list[FakeDisplayMeta] = []

    def acquire_display_meta(self) -> FakeDisplayMeta:
        item = FakeDisplayMeta()
        self.display_meta.append(item)
        return item


class FakeFrame:
    def __init__(self, objects: list[object]) -> None:
        self.pad_index = 0
        self.frame_number = 1
        self.pipeline_width = 1000
        self.pipeline_height = 500
        self.object_items = objects
        self.display_meta: list[FakeDisplayMeta] = []

    def append(self, item: FakeDisplayMeta) -> None:
        self.display_meta.append(item)


def fake_object(
    class_id: int,
    confidence: float,
    left: float = 100,
    top: float = 100,
) -> SimpleNamespace:
    font = SimpleNamespace(name=None, size=0, color=None)
    return SimpleNamespace(
        class_id=class_id,
        confidence=confidence,
        rect_params=SimpleNamespace(
            left=left,
            top=top,
            width=100,
            height=100,
            border_width=1,
            border_color=None,
        ),
        text_params=SimpleNamespace(
            display_text=b"old 0.99",
            x_offset=0,
            y_offset=0,
            font_params=font,
            set_bg_clr=False,
            text_bg_clr=None,
        ),
    )


class DeepStreamWorkerTests(unittest.TestCase):
    def test_metrics_expose_active_night_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            metrics = MetricsState(
                stream_ids=["night-stream"],
                metrics_path=path,
                interval_seconds=0,
                minimum_healthy_fps=20,
                group_id="group",
                generation=1,
                night_vision_enabled={"night-stream": True},
            )
            metrics.observe_frame("night-stream", detections=1)
            report = json.loads(path.read_text(encoding="utf-8"))[
                "streams"
            ]["night-stream"]

        self.assertTrue(report["night_vision_enabled"])
        self.assertEqual(report["vision_profile"], "night")

    def test_night_profile_uses_independent_gain_and_thresholds(self) -> None:
        config = {
            "gpu_id": 0,
            "onnx_path": "/models/model.onnx",
            "engine_path": "/engines/model_b2.engine",
            "labels_path": "/models/model.labels.txt",
            "parser_library": "/lib/parser.so",
            "batch_size": 1,
            "label_count": 80,
            "night_vision": {
                "enabled": True,
                "input_gain": 1.2,
                "plate_detector_confidence": 0.19,
            },
            "license_plate": {
                "detector_onnx_path": "/models/lpr/lpd.onnx",
                "detector_engine_path": "/engines/lpd.engine",
                "detector_batch_size": 16,
            },
            "streams": [
                {
                    "conf": 0.17,
                    "license_plate": {
                        "detector_interval": 0,
                        "vehicle_classes": [2, 3, 5, 7],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inference_path = root / "nvinfer.txt"
            lpd_path = root / "lpd.txt"
            build_inference_config(config, inference_path)
            build_lpd_config(config, lpd_path)
            inference_text = inference_path.read_text(encoding="utf-8")
            lpd_text = lpd_path.read_text(encoding="utf-8")

        self.assertIn(
            "net-scale-factor=0.0047058823529411761",
            inference_text,
        )
        self.assertIn(
            "pre-cluster-threshold=0.17000000",
            inference_text,
        )
        self.assertIn(
            "pre-cluster-threshold=0.19000000",
            lpd_text,
        )

    def test_license_plate_configs_enable_low_rate_async_sgies(self) -> None:
        config = {
            "gpu_id": 0,
            "license_plate": {
                "detector_onnx_path": "/models/lpr/lpd.onnx",
                "detector_engine_path": "/engines/lpd.engine",
                "detector_batch_size": 16,
                "recognizer_onnx_path": "/models/lpr/lpr.onnx",
                "recognizer_engine_path": "/engines/lpr.engine",
                "recognizer_batch_size": 16,
                "parser_library": "/lib/lpr.so",
            },
            "streams": [
                {
                    "license_plate": {
                        "detector_interval": 2,
                        "recognition_reinfer_interval": 15,
                        "minimum_plate_confidence": 0.5,
                        "vehicle_classes": [2, 3, 5, 7],
                    }
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lpd = root / "lpd.txt"
            lpr = root / "lpr.txt"
            build_lpd_config(config, lpd)
            build_lpr_config(config, lpr)
            lpd_text = lpd.read_text(encoding="utf-8")
            lpr_text = lpr.read_text(encoding="utf-8")

        self.assertIn("process-mode=2", lpd_text)
        self.assertIn("secondary-reinfer-interval=2", lpd_text)
        self.assertNotIn("\ninterval=", lpd_text)
        self.assertIn("operate-on-class-ids=2;3;5;7", lpd_text)
        self.assertIn("classifier-async-mode=1", lpr_text)
        self.assertIn("secondary-reinfer-interval=15", lpr_text)

    def test_pipeline_is_zero_copy_gpu_osd_and_nvenc_to_rtsp(self) -> None:
        class FakePipeline:
            def __init__(self) -> None:
                self.nodes: dict[str, tuple[str, dict[str, object]]] = {}
                self.links: list[tuple[object, ...]] = []

            def add(
                self,
                type_name: str,
                name: str,
                properties: dict[str, object] | None = None,
            ) -> "FakePipeline":
                self.nodes[name] = (type_name, properties or {})
                return self

            def link(self, *args: object) -> "FakePipeline":
                self.links.append(args)
                return self

            def attach(self, *_args: object) -> "FakePipeline":
                return self

        pipeline = FakePipeline()
        config = {
            "gpu_id": 0,
            "batch_size": 2,
            "batch_push_timeout_us": 20_000,
            "mux_width": 1920,
            "mux_height": 1080,
            "source_latency_ms": 100,
            "encoder_iframe_interval": 25,
            "tracker_config": "/tracker.yml",
            "tracker_library": "/tracker.so",
            "streams": [
                {
                    "input_url": "rtsp://camera/one",
                    "output_url": "rtsp://mediamtx/detected/one",
                    "bitrate_bps": 2_500_000,
                }
            ],
        }
        _add_pipeline_nodes(
            pipeline,
            config,
            Path("/runtime/nvinfer.txt"),
            object(),
            object(),
            object(),
            lambda _index: object(),
            lambda _index: object(),
        )

        self.assertEqual(pipeline.nodes["mux"][1]["batch-size"], 2)
        self.assertEqual(
            pipeline.nodes["osd"],
            ("nvdsosd", {"gpu-id": 0, "process-mode": 1}),
        )
        self.assertEqual(
            pipeline.nodes["encoder_0"][0],
            "nvv4l2h264enc",
        )
        self.assertEqual(
            pipeline.nodes["encoder_0"][1]["idrinterval"],
            25,
        )
        self.assertEqual(
            pipeline.nodes["encoder_0"][1]["num-B-Frames"],
            0,
        )
        self.assertTrue(
            pipeline.nodes["encoder_0"][1]["insert-aud"],
        )
        self.assertTrue(
            pipeline.nodes["parser_0"][1]["disable-passthrough"],
        )
        self.assertEqual(
            pipeline.nodes["publish_queue_0"][1]["leaky"],
            0,
        )
        self.assertEqual(
            pipeline.nodes["publish_queue_0"][1]["max-size-buffers"],
            25,
        )
        self.assertEqual(
            pipeline.nodes["parser_caps_0"][1]["caps"],
            "video/x-h264, stream-format=byte-stream, alignment=au",
        )
        self.assertEqual(
            pipeline.nodes["publish_clock_0"],
            (
                "clocksync",
                {"sync": True, "sync-to-first": True},
            ),
        )
        self.assertEqual(
            pipeline.nodes["rtsp_sink_0"][0],
            "rtspclientsink",
        )
        self.assertEqual(pipeline.nodes["rtsp_sink_0"][1]["protocols"], 4)
        self.assertEqual(pipeline.nodes["rtsp_sink_0"][1]["rtx-time"], 0)
        self.assertIn(
            (("demux", "publish_queue_0"), ("src_%u", "")),
            pipeline.links,
        )
        self.assertNotIn("payloader_0", pipeline.nodes)
        self.assertIn(
            (("publish_clock_0", "rtsp_sink_0"), ("", "sink_%u")),
            pipeline.links,
        )

        lpr_pipeline = FakePipeline()
        lpr_config = dict(config)
        lpr_config["license_plate"] = {
            "enabled": True,
            "detector_batch_size": 16,
            "recognizer_batch_size": 16,
        }
        _add_pipeline_nodes(
            lpr_pipeline,
            lpr_config,
            Path("/runtime/nvinfer.txt"),
            object(),
            object(),
            object(),
            lambda _index: object(),
            lambda _index: object(),
            Path("/runtime/lpd.txt"),
            Path("/runtime/lpr.txt"),
        )
        self.assertEqual(
            lpr_pipeline.nodes["license_plate_detector"][0],
            "nvinfer",
        )
        self.assertNotIn(
            "classifier-async-mode",
            lpr_pipeline.nodes["license_plate_recognizer"][1],
        )
        self.assertNotIn("tracker", lpr_pipeline.nodes)
        self.assertIn(
            (
                "mux",
                "pre_infer",
                "primary_infer",
                "vehicle_tracker",
                "license_plate_detector",
                "license_plate_recognizer",
                "osd_convert",
                "rgba_caps",
                "osd",
                "demux",
            ),
            lpr_pipeline.links,
        )
    def test_inference_config_uses_pair_engine_and_minimum_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nvinfer.txt"
            build_inference_config(
                {
                    "gpu_id": 0,
                    "onnx_path": "/models/model.onnx",
                    "engine_path": "/engines/model_b2.engine",
                    "labels_path": "/models/model.labels.txt",
                    "parser_library": "/lib/parser.so",
                    "batch_size": 2,
                    "label_count": 80,
                    "streams": [{"conf": 0.35}, {"conf": 0.20}],
                },
                path,
            )
            content = path.read_text(encoding="utf-8")

        self.assertIn("batch-size=2", content)
        self.assertIn("network-mode=2", content)
        self.assertIn("interval=0", content)
        self.assertIn("pre-cluster-threshold=0.20000000", content)
        self.assertIn("model-engine-file=/engines/model_b2.engine", content)

    def test_inference_latency_is_correlated_by_pad_and_frame(self) -> None:
        tracker = InferenceLatencyTracker()
        batch = SimpleNamespace(
            frame_items=[
                SimpleNamespace(pad_index=0, frame_number=10),
                SimpleNamespace(pad_index=1, frame_number=20),
            ]
        )
        tracker.start_batch(batch)
        first = tracker.finish(0, 10)
        second = tracker.finish(1, 20)
        missing = tracker.finish(0, 10)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertGreaterEqual(first or -1, 0)
        self.assertIsNone(missing)

    def test_plate_identity_tracker_keeps_id_for_moving_plate(self) -> None:
        tracker = PlateIdentityTracker()
        first = fake_object(0, 0.9, left=100, top=100)
        first.unique_component_id = 2
        first.object_id = -1
        first_frame = FakeFrame([first])
        tracker.process(FakeBatch([first_frame]))

        moved = fake_object(0, 0.9, left=118, top=104)
        moved.unique_component_id = 2
        moved.object_id = -1
        moved_frame = FakeFrame([moved])
        moved_frame.frame_number = 2
        tracker.process(FakeBatch([moved_frame]))

        far = fake_object(0, 0.9, left=800, top=400)
        far.unique_component_id = 2
        far.object_id = -1
        far_frame = FakeFrame([far])
        far_frame.frame_number = 3
        tracker.process(FakeBatch([far_frame]))

        self.assertGreaterEqual(first.object_id, 0)
        self.assertEqual(moved.object_id, first.object_id)
        self.assertNotEqual(far.object_id, first.object_id)

    def test_overlay_filters_classes_draws_roi_and_hides_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = MetricsState(
                stream_ids=["stream-1"],
                metrics_path=Path(directory) / "metrics.json",
                interval_seconds=60,
                minimum_healthy_fps=20,
                group_id="group",
                generation=1,
            )
            policy = StreamPolicy(
                stream_id="stream-1",
                classes=frozenset({0}),
                conf=0.25,
                roi=[
                    [0.05, 0.05],
                    [0.4, 0.05],
                    [0.4, 0.6],
                    [0.05, 0.6],
                ],
                labels={0: "人员"},
            )
            kept = fake_object(0, 0.9)
            hidden_class = fake_object(2, 0.9)
            hidden_conf = fake_object(0, 0.1)
            frame = FakeFrame([kept, hidden_class, hidden_conf])
            batch = FakeBatch([frame])
            fake_osd = SimpleNamespace(
                Color=FakeColor,
                Line=FakeLine,
                FontFamily=SimpleNamespace(Serif="serif"),
            )

            OverlayProcessor({0: policy}, metrics).process(batch, fake_osd)

        self.assertEqual(kept.rect_params.border_width, 3)
        self.assertEqual(kept.text_params.display_text, "人员".encode())
        self.assertNotIn(b"0.9", kept.text_params.display_text)
        self.assertEqual(kept.text_params.font_params.size, 18)
        self.assertTrue(kept.text_params.set_bg_clr)
        self.assertEqual(hidden_class.rect_params.border_width, 0)
        self.assertEqual(hidden_class.text_params.display_text, b"")
        self.assertEqual(hidden_conf.rect_params.border_width, 0)
        self.assertEqual(len(frame.display_meta), 1)
        self.assertEqual(len(frame.display_meta[0].lines), 4)

    def test_plate_overlay_is_chinese_and_stable_by_track_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = MetricsState(
                stream_ids=["stream-1"],
                metrics_path=Path(directory) / "metrics.json",
                interval_seconds=60,
                minimum_healthy_fps=20,
                group_id="group",
                generation=1,
            )
            policy = StreamPolicy(
                stream_id="stream-1",
                classes=frozenset({0}),
                conf=0.25,
                roi=None,
                labels={0: "人员"},
                license_plate_enabled=True,
                minimum_plate_confirmations=2,
            )
            plate = fake_object(0, 0.9)
            plate.unique_component_id = 2
            plate.object_id = 88
            plate.classifier_items = [
                SimpleNamespace(
                    unique_component_id=3,
                    label_items=[
                        SimpleNamespace(result_label="粤B12345")
                    ],
                )
            ]
            frame = FakeFrame([plate])
            batch = FakeBatch([frame])
            fake_osd = SimpleNamespace(
                Color=FakeColor,
                Line=FakeLine,
                FontFamily=SimpleNamespace(Serif="serif"),
            )
            processor = OverlayProcessor({0: policy}, metrics)
            processor.process(batch, fake_osd)
            first = plate.text_params.display_text
            frame.frame_number = 2
            processor.process(batch, fake_osd)
            second = plate.text_params.display_text

        self.assertEqual(first, "车牌识别中".encode())
        self.assertEqual(second, "车牌：粤B12345".encode())
        self.assertNotIn(b"0.9", second)

    def test_classifier_labels_support_deepstream8_get_n_label(self) -> None:
        classifier = SimpleNamespace(
            unique_component_id=3,
            n_labels=1,
            get_n_label=lambda index: "赣A195K9" if index == 0 else "",
        )
        object_meta = SimpleNamespace(classifier_items=[classifier])

        labels = OverlayProcessor._classifier_labels(object_meta)

        self.assertEqual(labels, ["赣A195K9"])

    def test_polygon_includes_boundary_and_rejects_outside(self) -> None:
        polygon = [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]
        self.assertTrue(_point_in_polygon((0.5, 0.5), polygon))
        self.assertTrue(_point_in_polygon((0.1, 0.1), polygon))
        self.assertFalse(_point_in_polygon((0.95, 0.5), polygon))


if __name__ == "__main__":
    unittest.main()
