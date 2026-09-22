from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rtsp_annotator.event_engine import (
    ActorOverlay,
    EventEngineResult,
    GarbageDetection,
    GarbageOverlay,
    GarbageSnapshot,
    NormalizedRect,
)
from rtsp_annotator.events import (
    EventDetectionOptions,
    EventRoiOptions,
    GarbageAnalysisOptions,
)
from rtsp_annotator.gas_cylinder import (
    GasCylinderCandidate,
    GasCylinderOptions,
    GasCylinderResultCache,
)
from rtsp_annotator.fishing_risk import (
    FishingRiskOptions,
    FishingRiskResultCache,
    FishingRiskSnapshot,
    FishingRiskSuspect,
    FishingRiskZoneOptions,
)
from rtsp_annotator.deepstream_worker import (
    GarbageOverlayCache,
    GroundLitterFrameProcessor,
    MetricsState,
    GarbageFrameProcessor,
    GarbageMetadataProcessor,
    OverlayProcessor,
    PlateIdentityTracker,
    PtzControlCommandMonitor,
    StreamPolicy,
    InferenceLatencyTracker,
    VesselFrameProcessor,
    _add_pipeline_nodes,
    _buffer_quality_flags,
    _ground_litter_occluder_classes,
    _merge_pile_detections,
    _load_policies,
    _point_in_polygon,
    build_inference_config,
    build_garbage_config,
    build_lpd_config,
    build_lpr_config,
    build_tracker_config,
)
from rtsp_annotator.ptz_verification import PtzVerificationOptions
from rtsp_annotator.ground_litter_detection import (
    GroundLitterDetection,
    GroundLitterDetectionOptions,
    GroundLitterResultCache,
    GroundLitterSnapshot,
    GroundLitterZone,
)
from rtsp_annotator.vessel_detection import (
    SMALL_TARGET_PROPOSAL_CLASS_ID,
    VesselDetection,
    VesselDetectionOptions,
    VesselResultCache,
    VesselSnapshot,
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


class FakeText:
    def __init__(self) -> None:
        self.display_text = b""
        self.x_offset = 0
        self.y_offset = 0
        self.font = SimpleNamespace(name=None, size=0, color=None)
        self.set_bg_color = False
        self.bg_color = None


class FakeDisplayMeta:
    def __init__(self) -> None:
        self.lines: list[FakeLine] = []
        self.texts: list[FakeText] = []

    def add_line(self, line: FakeLine) -> None:
        self.lines.append(line)

    def add_text(self, text: FakeText) -> None:
        self.texts.append(text)


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


class FakePipeline:
    """Minimal ServiceMaker pipeline stand-in for graph-shape assertions."""

    def __init__(self) -> None:
        self.nodes: dict[str, tuple[str, dict[str, object]]] = {}
        self.links: list[tuple[object, ...]] = []
        self.attachments: list[tuple[object, ...]] = []

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

    def attach(self, *args: object, **_kwargs: object) -> "FakePipeline":
        self.attachments.append(tuple(args))
        return self


class DeepStreamWorkerTests(unittest.TestCase):
    def test_ptz_control_monitor_delivers_and_removes_home_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            control_dir = Path(directory)
            command_path = control_dir / "ptz-home-stream-1-request-1.json"
            command_path.write_text(
                json.dumps(
                    {
                        "stream_id": "stream-1",
                        "request_id": "request-1",
                        "action": "return_home",
                    }
                ),
                encoding="utf-8",
            )
            coordinator = SimpleNamespace(
                request_return_home=lambda request_id: setattr(
                    coordinator,
                    "request_id",
                    request_id,
                )
            )
            monitor = PtzControlCommandMonitor(
                control_dir,
                {"stream-1": coordinator},  # type: ignore[dict-item]
            )

            monitor._consume(command_path)

            self.assertEqual(coordinator.request_id, "request-1")
            self.assertFalse(command_path.exists())

    def test_gstreamer_buffer_flags_expose_corruption_and_discontinuity(self) -> None:
        buffer = SimpleNamespace(get_flags=lambda: (1 << 8) | (1 << 6))

        corrupted, discontinuous = _buffer_quality_flags(buffer)

        self.assertTrue(corrupted)
        self.assertTrue(discontinuous)

    def test_metrics_include_bad_buffer_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            metrics = MetricsState(
                stream_ids=["stream-1"],
                metrics_path=path,
                interval_seconds=0,
                minimum_healthy_fps=20,
                group_id="group",
                generation=1,
            )
            buffer = SimpleNamespace(get_flags=lambda: (1 << 8) | (1 << 6))
            metrics.observe_pre_encode("stream-1", buffer)
            metrics.observe_frame("stream-1", detections=0)
            payload = json.loads(path.read_text(encoding="utf-8"))

        report = payload["streams"]["stream-1"]
        self.assertEqual(report["interval_corrupt_frames"], 1)
        self.assertEqual(report["interval_discontinuities"], 1)
        self.assertIn("updated_at_unix", payload)

    def test_pile_mode_clusters_parts_and_ignores_isolated_false_box(self) -> None:
        options = GarbageAnalysisOptions(
            enabled=True,
            detection_mode="pile",
            minimum_pile_detections=2,
            pile_merge_distance=0.18,
            pile_box_padding=0.04,
        )
        detections = [
            (NormalizedRect(0.40, 0.25, 0.05, 0.05), 0.55, "trash pile"),
            (NormalizedRect(0.50, 0.34, 0.06, 0.06), 0.45, "trash pile"),
            (NormalizedRect(0.62, 0.48, 0.07, 0.08), 0.40, "trash pile"),
            (NormalizedRect(0.90, 0.10, 0.03, 0.03), 0.35, "trash pile"),
        ]

        merged = _merge_pile_detections(detections, options)

        self.assertEqual(len(merged), 1)
        rectangle, confidence, label = merged[0]
        self.assertAlmostEqual(rectangle.left, 0.36)
        self.assertAlmostEqual(rectangle.top, 0.21)
        self.assertAlmostEqual(rectangle.width, 0.37)
        self.assertAlmostEqual(rectangle.height, 0.39)
        self.assertEqual(confidence, 0.55)
        self.assertEqual(label, "trash pile")

    def test_garbage_inference_config_is_low_rate(self) -> None:
        config = {
            "gpu_id": 0,
            "batch_size": 1,
            "night_vision": {"enabled": False},
            "garbage": {
                "onnx_path": "/models/events/garbage.onnx",
                "engine_path": "/engines/garbage.engine",
                "labels_path": "/models/events/garbage.labels.txt",
                "parser_library": "/lib/yolo.so",
                "label_count": 8,
            },
            "streams": [
                {
                    "event_detection": {
                        "garbage": {
                            "enabled": True,
                            "analysis_fps": 3,
                            "minimum_confidence": 0.35,
                        }
                    }
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "garbage.txt"
            build_garbage_config(config, path)
            content = path.read_text(encoding="utf-8")

        self.assertIn("interval=0", content)
        self.assertIn("gie-unique-id=4", content)
        self.assertIn("cluster-mode=2", content)
        self.assertIn("nms-iou-threshold=0.45", content)
        self.assertIn("pre-cluster-threshold=0.35000000", content)

    def test_garbage_semantic_result_survives_short_detector_flicker(self) -> None:
        processor = GarbageFrameProcessor(
            policies={},
            event_engines={},
            labels=[],
        )
        key = (0, "entrance")
        detected = GarbageSnapshot(
            timestamp=1,
            frame_number=1,
            area_ratio=0.02,
            regions=(NormalizedRect(0.2, 0.3, 0.1, 0.2),),
            semantic_confidence=0.8,
            object_type="plastic bottle",
            detections=(
                GarbageDetection(
                    rectangle=NormalizedRect(0.2, 0.3, 0.1, 0.2),
                    object_type="plastic bottle",
                    confidence=0.8,
                ),
            ),
        )
        empty = GarbageSnapshot(
            timestamp=2,
            frame_number=2,
            area_ratio=0.0,
        )

        processor._stabilize_semantic(key, detected)
        held = processor._stabilize_semantic(key, empty)
        processor._stabilize_semantic(key, empty)
        processor._stabilize_semantic(key, empty)
        expired = processor._stabilize_semantic(key, empty)

        self.assertEqual(held.regions, detected.regions)
        self.assertEqual(held.detections, detected.detections)
        self.assertEqual(held.timestamp, empty.timestamp)
        self.assertEqual(expired.regions, ())

    def test_garbage_overlay_cache_translates_limits_and_expires(self) -> None:
        cache = GarbageOverlayCache()
        cache.update(
            pad_index=0,
            roi_id="entrance",
            snapshot=GarbageSnapshot(
                timestamp=10,
                frame_number=1,
                area_ratio=0.03,
                detections=(
                    GarbageDetection(
                        NormalizedRect(0.1, 0.2, 0.1, 0.1),
                        "trash pile",
                        0.9,
                    ),
                    GarbageDetection(
                        NormalizedRect(0.6, 0.2, 0.1, 0.1),
                        "plastic bottle",
                        0.8,
                    ),
                ),
            ),
        )
        options = GarbageAnalysisOptions(
            enabled=True,
            display_hold_seconds=1,
            maximum_display_boxes=1,
        )

        active = cache.overlays(
            pad_index=0,
            options=options,
            timestamp=10.5,
        )
        expired = cache.overlays(
            pad_index=0,
            options=options,
            timestamp=11.1,
        )

        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].label, "垃圾：垃圾堆")
        self.assertEqual(active[0].state, "detected")
        self.assertEqual(expired, [])

    def test_garbage_analysis_accepts_primary_actor_inside_roi(self) -> None:
        options = EventDetectionOptions(
            enabled=True,
            rois=(
                EventRoiOptions(
                    roi_id="entrance",
                    polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                ),
            ),
            garbage=GarbageAnalysisOptions(enabled=True),
        )
        policy = StreamPolicy(
            stream_id="stream-1",
            classes=None,
            conf=0.25,
            roi=None,
            labels={0: "人员"},
            event_detection=options,
        )
        actor = fake_object(0, 0.9)
        actor.object_id = 7
        observed: list[object] = []
        def observe_garbage(**kwargs: object) -> EventEngineResult:
            observed.append(kwargs["snapshot"])
            return EventEngineResult()

        engine = SimpleNamespace(observe_garbage=observe_garbage)
        frame = FakeFrame([actor])
        frame.pipeline_width = 0
        frame.pipeline_height = 0
        frame.object_items = iter(frame.object_items)

        processor = GarbageFrameProcessor(
            policies={0: policy},
            event_engines={0: engine},
            labels=[],
            frame_width=1000,
            frame_height=500,
        )
        processor.process(
            FakeBatch([frame]),
            [np.zeros((90, 160, 3), dtype=np.uint8)],
        )

        self.assertEqual(len(observed), 1)
        self.assertIsNone(processor._background[(0, "entrance")]._baseline)

    def test_event_overlay_is_applied_per_tracked_actor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = MetricsState(
                stream_ids=["stream-1"],
                metrics_path=Path(directory) / "metrics.json",
                interval_seconds=60,
                minimum_healthy_fps=20,
                group_id="group",
                generation=1,
            )
            event_options = EventDetectionOptions(
                enabled=True,
                rois=(
                    EventRoiOptions(
                        roi_id="entrance",
                        polygon=(
                            (0.05, 0.05),
                            (0.5, 0.05),
                            (0.5, 0.8),
                            (0.05, 0.8),
                        ),
                    ),
                ),
            )
            policy = StreamPolicy(
                stream_id="stream-1",
                classes=frozenset({0}),
                conf=0.25,
                roi=None,
                labels={0: "人员"},
                event_detection=event_options,
            )
            tracked = fake_object(0, 0.9)
            tracked.object_id = 91
            engine = SimpleNamespace(
                observe_tracks=lambda **_kwargs: EventEngineResult(
                    actor_overlays=[
                        ActorOverlay(
                            track_id=91,
                            roi_id="entrance",
                            label="人员区域停留 20秒",
                            state="confirmed",
                            elapsed_seconds=20,
                        )
                    ]
                )
            )
            frame = FakeFrame([tracked])
            fake_osd = SimpleNamespace(
                Color=FakeColor,
                Line=FakeLine,
                Text=FakeText,
                FontFamily=SimpleNamespace(Serif="serif"),
            )

            OverlayProcessor(
                {0: policy},
                metrics,
                event_engines={0: engine},
            ).process(FakeBatch([frame]), fake_osd)

        self.assertEqual(
            tracked.text_params.display_text,
            "人员".encode(),
        )
        self.assertEqual(tracked.rect_params.border_width, 3)
        self.assertEqual(len(frame.display_meta[0].lines), 4)
        self.assertEqual(
            frame.display_meta[0].texts[0].display_text,
            "人员区域停留 20秒".encode(),
        )

    def test_confirmed_garbage_is_drawn_in_red_without_confidence(self) -> None:
        frame = FakeFrame([])
        batch = FakeBatch([frame])
        overlay = GarbageOverlay(
            roi_id="entrance",
            rectangle=NormalizedRect(0.2, 0.3, 0.1, 0.2),
            label="疑似乱丢垃圾",
            state="confirmed",
            elapsed_seconds=15,
        )
        fake_osd = SimpleNamespace(
            Color=FakeColor,
            Line=FakeLine,
            Text=FakeText,
            FontFamily=SimpleNamespace(Serif="serif"),
        )

        OverlayProcessor._draw_event_overlay(
            batch,
            frame,
            overlay,
            fake_osd,
            0,
        )

        self.assertEqual(len(frame.display_meta[0].lines), 4)
        self.assertEqual(
            frame.display_meta[0].texts[0].display_text,
            "疑似乱丢垃圾".encode(),
        )
        self.assertNotIn(b"0.", frame.display_meta[0].texts[0].display_text)

    def test_detected_garbage_is_drawn_in_green(self) -> None:
        frame = FakeFrame([])
        overlay = GarbageOverlay(
            roi_id="entrance",
            rectangle=NormalizedRect(0.2, 0.3, 0.1, 0.2),
            label="垃圾：垃圾堆",
            state="detected",
            elapsed_seconds=0,
        )
        fake_osd = SimpleNamespace(
            Color=FakeColor,
            Line=FakeLine,
            Text=FakeText,
            FontFamily=SimpleNamespace(Serif="serif"),
        )

        OverlayProcessor._draw_event_overlay(
            FakeBatch([frame]),
            frame,
            overlay,
            fake_osd,
            0,
        )

        self.assertEqual(
            frame.display_meta[0].lines[0].color,
            (0.0, 0.75, 0.2, 1.0),
        )
        self.assertEqual(
            frame.display_meta[0].texts[0].display_text,
            "垃圾：垃圾堆".encode(),
        )

    def test_event_garbage_overlay_replaces_normal_detection_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = MetricsState(
                stream_ids=["stream-1"],
                metrics_path=Path(directory) / "metrics.json",
                interval_seconds=60,
                minimum_healthy_fps=20,
                group_id="group",
                generation=1,
            )
            options = EventDetectionOptions(
                enabled=True,
                rois=(
                    EventRoiOptions(
                        roi_id="entrance",
                        polygon=((0, 0), (1, 0), (1, 1), (0, 1)),
                    ),
                ),
                garbage=GarbageAnalysisOptions(enabled=True),
            )
            policy = StreamPolicy(
                stream_id="stream-1",
                classes=None,
                conf=0.25,
                roi=None,
                labels={},
                event_detection=options,
            )
            rectangle = NormalizedRect(0.2, 0.3, 0.1, 0.2)
            cache = GarbageOverlayCache()
            cache.update(
                pad_index=0,
                roi_id="entrance",
                snapshot=GarbageSnapshot(
                    timestamp=time.monotonic(),
                    frame_number=1,
                    area_ratio=0.02,
                    detections=(
                        GarbageDetection(
                            rectangle,
                            "trash pile",
                            0.9,
                        ),
                    ),
                ),
            )
            engine = SimpleNamespace(
                observe_tracks=lambda **_kwargs: EventEngineResult(
                    garbage_overlays=[
                        GarbageOverlay(
                            roi_id="entrance",
                            rectangle=rectangle,
                            label="疑似新增垃圾 5/15秒",
                            state="candidate",
                            elapsed_seconds=5,
                        )
                    ]
                )
            )
            frame = FakeFrame([])
            fake_osd = SimpleNamespace(
                Color=FakeColor,
                Line=FakeLine,
                Text=FakeText,
                FontFamily=SimpleNamespace(Serif="serif"),
            )

            OverlayProcessor(
                {0: policy},
                metrics,
                event_engines={0: engine},
                garbage_overlay_cache=cache,
            ).process(FakeBatch([frame]), fake_osd)

        labels = [
            text.display_text
            for display_meta in frame.display_meta
            for text in display_meta.texts
        ]
        self.assertIn("疑似新增垃圾 5/15秒".encode(), labels)
        self.assertNotIn("垃圾：垃圾堆".encode(), labels)

    def test_garbage_metadata_is_filtered_by_roi_and_component(self) -> None:
        options = EventDetectionOptions(
            enabled=True,
            rois=(
                EventRoiOptions(
                    roi_id="entrance",
                    polygon=(
                        (0.0, 0.0),
                        (0.5, 0.0),
                        (0.5, 1.0),
                        (0.0, 1.0),
                    ),
                ),
            ),
            garbage=GarbageAnalysisOptions(
                enabled=True,
                minimum_confidence=0.35,
            ),
        )
        policy = StreamPolicy(
            stream_id="stream-1",
            classes=None,
            conf=0.25,
            roi=None,
            labels={0: "人员"},
            event_detection=options,
        )
        inside = fake_object(1, 0.8, left=100, top=100)
        inside.unique_component_id = 4
        overlapping_prompt = fake_object(0, 0.7, left=100, top=100)
        overlapping_prompt.unique_component_id = 4
        outside = fake_object(0, 0.9, left=800, top=100)
        outside.unique_component_id = 4
        primary = fake_object(0, 0.9, left=100, top=100)
        primary.unique_component_id = 1
        snapshots: list[object] = []
        engine = SimpleNamespace(
            observe_garbage=lambda **kwargs: snapshots.append(
                kwargs["snapshot"]
            )
        )

        GarbageMetadataProcessor(
            policies={0: policy},
            event_engines={0: engine},
            labels=["plastic bottle", "garbage bag"],
            interval=0,
        ).process(
            FakeBatch(
                [FakeFrame([inside, overlapping_prompt, outside, primary])]
            )
        )

        self.assertEqual(len(snapshots), 1)
        snapshot = snapshots[0]
        self.assertEqual(snapshot.object_type, "garbage bag")
        self.assertAlmostEqual(snapshot.area_ratio, 0.02)
        self.assertEqual(len(snapshot.regions), 1)
        self.assertEqual(len(snapshot.detections), 1)
        self.assertEqual(snapshot.detections[0].object_type, "garbage bag")

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

            def attach(
                self,
                *_args: object,
                **_kwargs: object,
            ) -> "FakePipeline":
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
            lambda _index: object(),
            object(),
            lambda _index: object(),
            lambda _index: object(),
        )

        self.assertEqual(pipeline.nodes["mux"][1]["batch-size"], 2)
        self.assertEqual(
            pipeline.nodes["source_0"][1]["num-extra-surfaces"],
            8,
        )
        self.assertFalse(
            pipeline.nodes["source_0"][1]["drop-on-latency"],
        )
        self.assertEqual(
            pipeline.nodes["osd_0"],
            ("nvdsosd", {"gpu-id": 0, "process-mode": 1}),
        )
        self.assertEqual(
            pipeline.nodes["overlay_anchor_0"],
            ("identity", {"silent": True}),
        )
        self.assertNotIn("osd", pipeline.nodes)
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
        self.assertEqual(pipeline.nodes["encoder_0"][1]["profile"], 4)
        self.assertEqual(pipeline.nodes["encoder_0"][1]["preset-id"], 4)
        self.assertEqual(
            pipeline.nodes["encoder_0"][1]["tuning-info-id"],
            1,
        )
        self.assertEqual(pipeline.nodes["encoder_0"][1]["aq"], 8)
        self.assertTrue(pipeline.nodes["encoder_0"][1]["temporalaq"])
        self.assertEqual(
            pipeline.nodes["encoder_0"][1]["num-Ref-Frames"],
            2,
        )
        self.assertTrue(
            pipeline.nodes["encoder_0"][1]["insert-aud"],
        )
        self.assertTrue(
            pipeline.nodes["encoder_0"][1]["insert-vui"],
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
                {"sync": False},
            ),
        )
        self.assertEqual(
            pipeline.nodes["rtsp_sink_0"][0],
            "rtspclientsink",
        )
        self.assertEqual(pipeline.nodes["rtsp_sink_0"][1]["protocols"], 4)
        self.assertEqual(pipeline.nodes["rtsp_sink_0"][1]["rtx-time"], 0)
        self.assertIn(
            (("demux", "overlay_anchor_0"), ("src_%u", "")),
            pipeline.links,
        )
        self.assertIn(
            (
                "overlay_anchor_0",
                "osd_0",
                "publish_queue_0",
                "publish_convert_0",
                "publish_caps_0",
                "encoder_0",
                "parser_0",
                "parser_caps_0",
                "publish_clock_0",
            ),
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
            lambda _index: object(),
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
                "demux",
            ),
            lpr_pipeline.links,
        )

        garbage_pipeline = FakePipeline()
        garbage_config = dict(config)
        garbage_config["garbage"] = {"enabled": True}
        garbage_config["streams"] = [
            {
                **config["streams"][0],
                "event_detection": {
                    "garbage": {"enabled": True}
                },
            }
        ]
        garbage_receiver = object()
        garbage_skip_probe = object()
        _add_pipeline_nodes(
            garbage_pipeline,
            garbage_config,
            Path("/runtime/nvinfer.txt"),
            object(),
            lambda _index: object(),
            object(),
            lambda _index: object(),
            lambda _index: object(),
            garbage_config_path=Path("/runtime/garbage.txt"),
            garbage_receiver=garbage_receiver,
            garbage_skip_probe=garbage_skip_probe,
        )

        self.assertEqual(
            garbage_pipeline.nodes["garbage_queue"][1]["leaky"],
            2,
        )
        self.assertEqual(
            garbage_pipeline.nodes["garbage_queue"][1][
                "max-size-buffers"
            ],
            1,
        )
        self.assertIn(
            (("analytics_tee", "garbage_queue"), ("src_%u", "")),
            garbage_pipeline.links,
        )
        self.assertIn(
            (
                "garbage_queue",
                "garbage_infer",
                "garbage_convert",
                "garbage_rgba_caps",
                "garbage_sink",
            ),
            garbage_pipeline.links,
        )
        self.assertEqual(
            garbage_pipeline.nodes["garbage_sink"][1]["max-buffers"],
            1,
        )
        self.assertEqual(
            garbage_pipeline.nodes["garbage_sink"][1]["drop"],
            True,
        )
        self.assertEqual(
            garbage_pipeline.nodes["garbage_rgba_caps"][1]["caps"],
            (
                "video/x-raw(memory:NVMM), format=RGB, "
                "width=640, height=360"
            ),
        )

        gas_pipeline = FakePipeline()
        gas_config = dict(config)
        gas_config["gas_cylinder"] = {
            "enabled": True,
            "input_width": 1280,
            "input_height": 720,
        }
        gas_receiver = object()
        gas_skip_probe = object()
        _add_pipeline_nodes(
            gas_pipeline,
            gas_config,
            Path("/runtime/nvinfer.txt"),
            object(),
            lambda _index: object(),
            object(),
            lambda _index: object(),
            lambda _index: object(),
            gas_cylinder_receiver=gas_receiver,
            gas_cylinder_skip_probe=gas_skip_probe,
        )
        self.assertEqual(
            gas_pipeline.nodes["gas_cylinder_queue"][1]["leaky"],
            2,
        )
        self.assertEqual(
            gas_pipeline.nodes["gas_cylinder_queue"][1][
                "max-size-buffers"
            ],
            1,
        )
        self.assertEqual(
            gas_pipeline.nodes["gas_cylinder_rgb_caps"][1]["caps"],
            (
                "video/x-raw(memory:NVMM), format=RGB, "
                "width=1280, height=720"
            ),
        )
        self.assertIn(
            (
                "gas_cylinder_queue",
                "gas_cylinder_convert",
                "gas_cylinder_rgb_caps",
                "gas_cylinder_sink",
            ),
            gas_pipeline.links,
        )
        self.assertEqual(
            gas_pipeline.nodes["publish_queue_0"][1]["leaky"],
            0,
        )

        vessel_pipeline = FakePipeline()
        vessel_config = dict(config)
        vessel_config["vessel_detection"] = {
            "enabled": True,
            "input_width": 1920,
            "input_height": 1080,
        }
        vessel_receiver = object()
        vessel_skip_probe = object()
        _add_pipeline_nodes(
            vessel_pipeline,
            vessel_config,
            Path("/runtime/nvinfer.txt"),
            object(),
            lambda _index: object(),
            object(),
            lambda _index: object(),
            lambda _index: object(),
            vessel_receiver=vessel_receiver,
            vessel_skip_probe=vessel_skip_probe,
        )
        self.assertEqual(
            vessel_pipeline.nodes["vessel_queue"][1]["leaky"],
            2,
        )
        self.assertEqual(
            vessel_pipeline.nodes["vessel_rgb_caps"][1]["caps"],
            (
                "video/x-raw(memory:NVMM), format=RGB, "
                "width=1920, height=1080"
            ),
        )
        self.assertIn(
            (
                "vessel_queue",
                "vessel_convert",
                "vessel_rgb_caps",
                "vessel_sink",
            ),
            vessel_pipeline.links,
        )
        self.assertEqual(
            vessel_pipeline.nodes["vessel_sink"][1]["drop"],
            True,
        )

    def test_gas_cylinder_osd_batches_four_boxes_per_display_meta(self) -> None:
        frame = FakeFrame([])
        batch = FakeBatch([frame])
        cache = GasCylinderResultCache()
        cache.publish(
            0,
            [
                GasCylinderCandidate(
                    NormalizedRect(0.05 + index * 0.1, 0.2, 0.07, 0.2),
                    0.8,
                )
                for index in range(5)
            ],
            timestamp=1,
            inference_ms=10,
        )
        fake_osd = SimpleNamespace(
            Color=FakeColor,
            Line=FakeLine,
            Text=FakeText,
            FontFamily=SimpleNamespace(Serif="serif"),
        )

        OverlayProcessor._draw_gas_cylinders(
            batch,
            frame,
            cache.snapshot(0),
            GasCylinderOptions(enabled=True),
            fake_osd,
            width=1000,
            height=500,
        )

        self.assertEqual(len(frame.display_meta), 2)
        self.assertEqual(len(frame.display_meta[0].lines), 16)
        self.assertEqual(len(frame.display_meta[1].lines), 4)
        self.assertEqual(
            frame.display_meta[0].texts[0].display_text,
            "燃气瓶数量：5".encode(),
        )

    def test_vessel_osd_draws_only_confirmed_cached_boxes(self) -> None:
        frame = FakeFrame([])
        batch = FakeBatch([frame])
        snapshot = VesselSnapshot(
            state="running",
            detections=tuple(
                VesselDetection(
                    object_id=index + 1,
                    rectangle=NormalizedRect(
                        0.05 + index * 0.1,
                        0.2,
                        0.07,
                        0.15,
                    ),
                    confidence=0.2,
                    class_id=8,
                    hits=2,
                )
                for index in range(5)
            ),
            result_version=2,
            updated_at=time.monotonic(),
        )
        fake_osd = SimpleNamespace(
            Color=FakeColor,
            Line=FakeLine,
            Text=FakeText,
            FontFamily=SimpleNamespace(Serif="serif"),
        )

        OverlayProcessor._draw_vessels(
            batch,
            frame,
            snapshot,
            VesselDetectionOptions(enabled=True, display_ids=True),
            fake_osd,
            width=1000,
            height=500,
        )

        self.assertEqual(len(frame.display_meta), 2)
        self.assertEqual(len(frame.display_meta[0].lines), 16)
        self.assertEqual(len(frame.display_meta[1].lines), 4)
        self.assertEqual(
            frame.display_meta[0].texts[0].display_text,
            "船舶：5".encode(),
        )
        self.assertEqual(
            frame.display_meta[0].texts[1].display_text,
            "船 #1".encode(),
        )

    def test_vessel_osd_hides_internal_ptz_proposals_by_default(self) -> None:
        frame = FakeFrame([])
        batch = FakeBatch([frame])
        snapshot = VesselSnapshot(
            state="running",
            detections=(
                VesselDetection(
                    object_id=1,
                    rectangle=NormalizedRect(0.2, 0.3, 0.1, 0.1),
                    confidence=0.3,
                    class_id=8,
                    hits=3,
                ),
                VesselDetection(
                    object_id=2,
                    rectangle=NormalizedRect(0.5, 0.5, 0.03, 0.02),
                    confidence=0.2,
                    class_id=SMALL_TARGET_PROPOSAL_CLASS_ID,
                    hits=5,
                ),
            ),
            result_version=2,
            updated_at=time.monotonic(),
        )
        fake_osd = SimpleNamespace(
            Color=FakeColor,
            Line=FakeLine,
            Text=FakeText,
            FontFamily=SimpleNamespace(Serif="serif"),
        )

        OverlayProcessor._draw_vessels(
            batch,
            frame,
            snapshot,
            VesselDetectionOptions(enabled=True),
            fake_osd,
            width=1000,
            height=500,
        )

        texts = [
            item.display_text.decode()
            for meta in frame.display_meta
            for item in meta.texts
        ]
        self.assertEqual(texts, ["船舶：1", "船"])

    def test_vessel_osd_suppresses_contained_sidecar_box(self) -> None:
        frame = FakeFrame([])
        batch = FakeBatch([frame])
        snapshot = VesselSnapshot(
            state="running",
            detections=(
                VesselDetection(
                    object_id=2,
                    rectangle=NormalizedRect(0.25, 0.35, 0.25, 0.15),
                    confidence=0.8,
                    class_id=8,
                    hits=3,
                ),
            ),
            result_version=2,
            updated_at=time.monotonic(),
        )
        fake_osd = SimpleNamespace(
            Color=FakeColor,
            Line=FakeLine,
            Text=FakeText,
            FontFamily=SimpleNamespace(Serif="serif"),
        )

        OverlayProcessor._draw_vessels(
            batch,
            frame,
            snapshot,
            VesselDetectionOptions(enabled=True),
            fake_osd,
            width=1000,
            height=500,
            existing_rectangles=(
                NormalizedRect(0.1, 0.2, 0.7, 0.5),
            ),
        )

        self.assertEqual(len(frame.display_meta[0].lines), 0)
        self.assertEqual(
            frame.display_meta[0].texts[0].display_text,
            "船舶：1".encode(),
        )

    def test_vessel_osd_marks_only_matching_risk_candidate(self) -> None:
        frame = FakeFrame([])
        batch = FakeBatch([frame])
        detection = VesselDetection(
            object_id=12,
            rectangle=NormalizedRect(0.2, 0.3, 0.1, 0.1),
            confidence=0.4,
            class_id=8,
            hits=3,
        )
        vessel_snapshot = VesselSnapshot(
            state="running",
            detections=(detection,),
            result_version=3,
            updated_at=time.monotonic(),
        )
        risk_snapshot = FishingRiskSnapshot(
            state="running",
            suspects=(
                FishingRiskSuspect(
                    risk_track_id=4,
                    vessel_object_id=12,
                    rectangle=detection.rectangle,
                    zone_id="protected_water",
                    risk_score=60,
                    reasons=("restricted_period_presence", "loitering"),
                    dwell_seconds=180,
                    reversal_count=0,
                ),
            ),
            result_version=2,
            updated_at=time.monotonic(),
        )
        options = FishingRiskOptions(
            enabled=True,
            zones=(
                FishingRiskZoneOptions(
                    "protected_water",
                    ((0, 0), (1, 0), (1, 1), (0, 1)),
                ),
            ),
        )
        fake_osd = SimpleNamespace(
            Color=FakeColor,
            Line=FakeLine,
            Text=FakeText,
            FontFamily=SimpleNamespace(Serif="serif"),
        )

        OverlayProcessor._draw_vessels(
            batch,
            frame,
            vessel_snapshot,
            VesselDetectionOptions(enabled=True),
            fake_osd,
            width=1000,
            height=500,
            fishing_options=options,
            fishing_snapshot=risk_snapshot,
        )

        self.assertEqual(
            frame.display_meta[0].texts[0].display_text,
            "船舶：1｜风险候选：1".encode(),
        )
        self.assertEqual(
            frame.display_meta[0].texts[1].display_text,
            "疑似捕捞线索 60分".encode(),
        )

    def test_vessel_frame_processor_consumes_each_risk_version_once(self) -> None:
        class FakeClient:
            def accepts(self, _pad_index: int, *, timestamp: float) -> bool:
                del timestamp
                return False

        class FakeRiskEngine:
            def __init__(self) -> None:
                self.calls = 0

            def observe(self, **_kwargs: object) -> object:
                self.calls += 1
                return SimpleNamespace(
                    snapshot=FishingRiskSnapshot(
                        state="running",
                        result_version=self.calls,
                        updated_at=time.monotonic(),
                    ),
                    events=[],
                )

        vessel_cache = VesselResultCache()
        vessel_cache.store_snapshot(
            0,
            VesselSnapshot(
                state="running",
                detections=(),
                result_version=5,
                updated_at=time.monotonic(),
            ),
        )
        risk_cache = FishingRiskResultCache()
        engine = FakeRiskEngine()
        processor = VesselFrameProcessor(
            FakeClient(),  # type: ignore[arg-type]
            vessel_cache=vessel_cache,
            fishing_risk_engines={0: engine},  # type: ignore[dict-item]
            fishing_risk_cache=risk_cache,
        )
        batch = FakeBatch([FakeFrame([])])
        frames = [np.zeros((8, 8, 3), dtype=np.uint8)]

        processor.process(batch, frames)
        processor.process(batch, frames)

        self.assertEqual(engine.calls, 1)
        self.assertEqual(risk_cache.snapshot(0).state, "running")

    def test_vessel_frame_processor_pauses_risk_but_keeps_closeup_sampling(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.submits = 0
                self.verification_modes: list[bool] = []
                self.view_generations: list[int] = []

            def accepts(self, _pad_index: int, *, timestamp: float) -> bool:
                del timestamp
                return True

            def submit(self, *_args: object, **kwargs: object) -> None:
                self.submits += 1
                self.verification_modes.append(
                    bool(kwargs.get("verification_active"))
                )
                self.view_generations.append(
                    int(kwargs.get("view_generation", -1))
                )

        class FakeRiskEngine:
            def __init__(self) -> None:
                self.resets = 0
                self.calls = 0

            def reset_tracking(self) -> None:
                self.resets += 1

            def observe(self, **_kwargs: object) -> object:
                self.calls += 1
                return SimpleNamespace(
                    snapshot=FishingRiskSnapshot(state="running"),
                    events=[],
                )

        vessel_cache = VesselResultCache()
        vessel_cache.store_snapshot(
            0,
            VesselSnapshot(
                state="running",
                detections=(),
                result_version=7,
                updated_at=time.monotonic(),
            ),
        )
        risk_cache = FishingRiskResultCache()
        engine = FakeRiskEngine()
        client = FakeClient()
        coordinator = SimpleNamespace(is_busy=True, view_generation=7)
        processor = VesselFrameProcessor(
            client,  # type: ignore[arg-type]
            vessel_cache=vessel_cache,
            fishing_risk_engines={0: engine},  # type: ignore[dict-item]
            fishing_risk_cache=risk_cache,
            ptz_verification_coordinators={0: coordinator},  # type: ignore[dict-item]
        )
        batch = FakeBatch([FakeFrame([])])
        frames = [np.zeros((8, 8, 3), dtype=np.uint8)]

        processor.process(batch, frames)
        processor.process(batch, frames)

        self.assertEqual(engine.resets, 1)
        self.assertEqual(engine.calls, 0)
        self.assertEqual(client.submits, 2)
        self.assertEqual(client.verification_modes, [True, True])
        self.assertEqual(client.view_generations, [7, 7])
        self.assertEqual(risk_cache.snapshot(0).state, "paused")

    def test_gas_cylinder_alarm_starts_only_above_threshold(self) -> None:
        fake_osd = SimpleNamespace(
            Color=FakeColor,
            Line=FakeLine,
            Text=FakeText,
            FontFamily=SimpleNamespace(Serif="serif"),
        )

        def draw(count: int) -> FakeFrame:
            frame = FakeFrame([])
            batch = FakeBatch([frame])
            cache = GasCylinderResultCache()
            cache.publish(
                0,
                [
                    GasCylinderCandidate(
                        NormalizedRect(
                            0.02 + (index % 10) * 0.095,
                            0.15 + (index // 10) * 0.35,
                            0.05,
                            0.20,
                        ),
                        0.8,
                    )
                    for index in range(count)
                ],
                timestamp=1,
                inference_ms=10,
            )
            OverlayProcessor._draw_gas_cylinders(
                batch,
                frame,
                cache.snapshot(0),
                GasCylinderOptions(enabled=True, alarm_threshold=18),
                fake_osd,
                width=1000,
                height=500,
            )
            return frame

        at_threshold = draw(18)
        self.assertEqual(
            at_threshold.display_meta[0].texts[0].display_text,
            "燃气瓶数量：18".encode(),
        )
        self.assertEqual(
            at_threshold.display_meta[0].texts[0].bg_color,
            (0.0, 0.85, 0.2, 1.0),
        )
        self.assertEqual(
            at_threshold.display_meta[0].lines[0].color,
            (0.0, 0.85, 0.2, 1.0),
        )

        above_threshold = draw(19)
        self.assertEqual(
            above_threshold.display_meta[0].texts[0].display_text,
            "燃气瓶数量：19（超量告警）".encode(),
        )
        self.assertEqual(
            above_threshold.display_meta[0].texts[0].bg_color,
            (1.0, 0.0, 0.0, 1.0),
        )
        self.assertEqual(
            above_threshold.display_meta[0].lines[0].color,
            (1.0, 0.0, 0.0, 1.0),
        )

    def test_tracker_config_reduces_shadow_tracking_age(self) -> None:
        source = (
            "[BaseConfig]\n"
            "maxShadowTrackingAge: 51    # original\n"
            "probationAge: 2\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.yml"
            target_path = root / "tracker.yml"
            source_path.write_text(source, encoding="utf-8")

            build_tracker_config(
                source_path,
                target_path,
                max_shadow_tracking_age=15,
            )

            content = target_path.read_text(encoding="utf-8")

        self.assertIn("maxShadowTrackingAge: 15    # original", content)
        self.assertIn("probationAge: 2", content)
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

    def test_overlay_hides_all_detection_boxes_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = MetricsState(
                stream_ids=["stream-1"],
                metrics_path=Path(directory) / "metrics.json",
                interval_seconds=0,
                minimum_healthy_fps=20,
                group_id="group",
                generation=1,
            )
            policy = StreamPolicy(
                stream_id="stream-1",
                # No class filter, so both objects pass the ordinary gating and
                # only the display switch can remove their boxes.
                classes=None,
                conf=0.25,
                roi=None,
                labels={0: "人员", 2: "车辆"},
                display_detections=False,
                ground_litter=GroundLitterDetectionOptions(
                    enabled=True,
                    zones=(
                        GroundLitterZone(
                            region_id="z1",
                            polygon=(
                                (0.05, 0.05),
                                (0.4, 0.05),
                                (0.4, 0.6),
                                (0.05, 0.6),
                            ),
                        ),
                    ),
                ),
            )
            person = fake_object(0, 0.9)
            vehicle = fake_object(2, 0.9)
            frame = FakeFrame([person, vehicle])
            batch = FakeBatch([frame])
            cache = GroundLitterResultCache()
            cache.store_snapshot(
                0,
                GroundLitterSnapshot(
                    state="running",
                    detections=(
                        GroundLitterDetection(
                            object_id=1,
                            rectangle=NormalizedRect(0.1, 0.2, 0.05, 0.05),
                            confidence=0.5,
                        ),
                    ),
                    result_version=1,
                    updated_at=time.monotonic(),
                ),
            )
            fake_osd = SimpleNamespace(
                Color=FakeColor,
                Line=FakeLine,
                Text=FakeText,
                FontFamily=SimpleNamespace(Serif="serif"),
            )

            OverlayProcessor(
                {0: policy},
                metrics,
                ground_litter_cache=cache,
            ).process(batch, fake_osd)
            report = json.loads(
                (Path(directory) / "metrics.json").read_text(encoding="utf-8")
            )

        # Ordinary boxes are gone, business overlays remain.
        self.assertEqual(person.rect_params.border_width, 0)
        self.assertEqual(person.text_params.display_text, b"")
        self.assertEqual(vehicle.rect_params.border_width, 0)
        self.assertTrue(frame.display_meta)
        line_total = sum(len(item.lines) for item in frame.display_meta)
        self.assertGreater(line_total, 0)
        titles = [
            text.display_text
            for item in frame.display_meta
            for text in item.texts
        ]
        self.assertIn("疑似垃圾：1".encode(), titles)
        # Detection accounting still reflects the real detections.
        self.assertEqual(
            report["streams"]["stream-1"]["interval_detections"],
            2,
        )

    def test_overlay_draws_detections_by_default(self) -> None:
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
            )
            person = fake_object(0, 0.9)
            fake_osd = SimpleNamespace(
                Color=FakeColor,
                Line=FakeLine,
                FontFamily=SimpleNamespace(Serif="serif"),
            )

            OverlayProcessor({0: policy}, metrics).process(
                FakeBatch([FakeFrame([person])]),
                fake_osd,
            )

        self.assertTrue(person.rect_params.border_width > 0)
        self.assertEqual(person.text_params.display_text, "人员".encode())

    def test_ptz_closeup_hides_home_rois_and_uses_full_frame(self) -> None:
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
                classes=frozenset({8}),
                conf=0.1,
                roi=[
                    [0.0, 0.0],
                    [0.5, 0.0],
                    [0.5, 1.0],
                    [0.0, 1.0],
                ],
                labels={8: "船舶"},
                vessel_detection=VesselDetectionOptions(
                    enabled=True,
                    roi=(
                        (0.0, 0.5),
                        (1.0, 0.5),
                        (1.0, 1.0),
                        (0.0, 1.0),
                    ),
                    display_roi=True,
                ),
            )
            vessel_cache = VesselResultCache()
            vessel_cache.store_snapshot(
                0,
                VesselSnapshot(
                    state="running",
                    detections=(),
                    result_version=1,
                    updated_at=time.monotonic(),
                ),
            )
            outside_home_roi = fake_object(8, 0.9, left=800, top=100)
            frame = FakeFrame([outside_home_roi])
            fake_osd = SimpleNamespace(
                Color=FakeColor,
                Line=FakeLine,
                Text=FakeText,
                FontFamily=SimpleNamespace(Serif="serif"),
            )

            OverlayProcessor(
                {0: policy},
                metrics,
                vessel_detection_cache=vessel_cache,
                ptz_verification_coordinators={
                    0: SimpleNamespace(
                        is_busy=True,
                        publish_primary_detections=lambda *_args, **_kwargs: None,
                    )
                },
            ).process(FakeBatch([frame]), fake_osd)

        self.assertEqual(outside_home_roi.rect_params.border_width, 3)
        self.assertEqual(
            outside_home_roi.text_params.display_text,
            "船舶".encode(),
        )
        self.assertEqual(
            sum(len(item.lines) for item in frame.display_meta),
            0,
        )

    def test_green_primary_vessel_box_is_published_to_ptz(self) -> None:
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
                classes=frozenset({8}),
                conf=0.1,
                roi=None,
                labels={8: "船舶"},
                vessel_detection=VesselDetectionOptions(enabled=True),
            )
            published: list[tuple[VesselDetection, ...]] = []
            coordinator = SimpleNamespace(
                is_busy=False,
                publish_primary_detections=lambda detections, **_kwargs: (
                    published.append(detections)
                ),
            )
            vessel = fake_object(8, 0.8, left=400, top=150)
            vessel.object_id = 77
            fake_osd = SimpleNamespace(
                Color=FakeColor,
                Line=FakeLine,
                Text=FakeText,
                FontFamily=SimpleNamespace(Serif="serif"),
            )

            OverlayProcessor(
                {0: policy},
                metrics,
                ptz_verification_coordinators={0: coordinator},
            ).process(FakeBatch([FakeFrame([vessel])]), fake_osd)

        self.assertEqual(len(published), 1)
        self.assertEqual(len(published[0]), 1)
        self.assertEqual(published[0][0].class_id, 8)
        self.assertEqual(published[0][0].object_id, -79)

    def test_active_ptz_target_is_red_and_operation_log_is_top_right(self) -> None:
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
                classes=frozenset({8}),
                conf=0.1,
                roi=None,
                labels={8: "船舶"},
                vessel_detection=VesselDetectionOptions(enabled=True),
                ptz_verification=PtzVerificationOptions(
                    display_operation_log=True,
                ),
            )
            target = fake_object(8, 0.9, left=100, top=100)
            target.object_id = 10
            other = fake_object(8, 0.9, left=500, top=100)
            other.object_id = 11
            coordinator = SimpleNamespace(
                is_busy=True,
                publish_primary_detections=lambda *_args, **_kwargs: None,
                overlay_state=lambda: SimpleNamespace(
                    state="tracking",
                    target_rectangle=NormalizedRect(0.1, 0.2, 0.1, 0.2),
                    vessel_number="10032",
                    operation_lines=(
                        "16:30:01 锁定追踪船只",
                        "16:30:02 追踪纠偏 变焦+1",
                    ),
                ),
            )
            frame = FakeFrame([target, other])
            batch = FakeBatch([frame])
            fake_osd = SimpleNamespace(
                Color=FakeColor,
                Line=FakeLine,
                Text=FakeText,
                FontFamily=SimpleNamespace(Serif="serif"),
            )

            OverlayProcessor(
                {0: policy},
                metrics,
                ptz_verification_coordinators={0: coordinator},
            ).process(batch, fake_osd)

        self.assertEqual(target.rect_params.border_width, 5)
        self.assertEqual(
            target.rect_params.border_color,
            (1.0, 0.08, 0.08, 1.0),
        )
        self.assertEqual(
            target.text_params.display_text,
            "追踪中｜船舶｜船号 10032".encode(),
        )
        self.assertEqual(other.rect_params.border_width, 3)
        self.assertEqual(
            other.rect_params.border_color,
            (0.0, 1.0, 0.0, 1.0),
        )
        self.assertEqual(other.text_params.display_text, "船舶".encode())
        log_texts = [
            text.display_text.decode()
            for meta in frame.display_meta
            for text in meta.texts
            if text.x_offset >= 570
        ]
        self.assertEqual(
            log_texts,
            [
                "PTZ｜持续追踪",
                "16:30:01 锁定追踪船只",
                "16:30:02 追踪纠偏 变焦+1",
            ],
        )

    def test_ptz_operation_log_is_hidden_when_display_switch_is_false(self) -> None:
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
                classes=frozenset({8}),
                conf=0.1,
                roi=None,
                labels={8: "船舶"},
                vessel_detection=VesselDetectionOptions(enabled=True),
                ptz_verification=PtzVerificationOptions(
                    display_operation_log=False,
                ),
            )
            vessel = fake_object(8, 0.9)
            coordinator = SimpleNamespace(
                is_busy=True,
                publish_primary_detections=lambda *_args, **_kwargs: None,
                overlay_state=lambda: SimpleNamespace(
                    state="tracking",
                    target_rectangle=NormalizedRect(0.1, 0.2, 0.1, 0.2),
                    operation_lines=("16:30:01 定位目标",),
                ),
            )
            frame = FakeFrame([vessel])
            batch = FakeBatch([frame])
            fake_osd = SimpleNamespace(
                Color=FakeColor,
                Line=FakeLine,
                Text=FakeText,
                FontFamily=SimpleNamespace(Serif="serif"),
            )

            OverlayProcessor(
                {0: policy},
                metrics,
                ptz_verification_coordinators={0: coordinator},
            ).process(batch, fake_osd)

        self.assertEqual(vessel.rect_params.border_width, 5)
        self.assertEqual(frame.display_meta, [])

    def test_cached_active_vessel_target_uses_tracking_style(self) -> None:
        frame = FakeFrame([])
        batch = FakeBatch([frame])
        target_rectangle = NormalizedRect(0.2, 0.3, 0.1, 0.1)
        snapshot = VesselSnapshot(
            state="running",
            detections=(
                VesselDetection(
                    object_id=7,
                    rectangle=target_rectangle,
                    confidence=0.4,
                    class_id=8,
                    hits=3,
                ),
            ),
            result_version=2,
            updated_at=time.monotonic(),
        )
        fake_osd = SimpleNamespace(
            Color=FakeColor,
            Line=FakeLine,
            Text=FakeText,
            FontFamily=SimpleNamespace(Serif="serif"),
        )

        OverlayProcessor._draw_vessels(
            batch,
            frame,
            snapshot,
            VesselDetectionOptions(enabled=True),
            fake_osd,
            width=1000,
            height=500,
            active_target_rectangle=target_rectangle,
        )

        self.assertTrue(
            all(line.width == 5 for line in frame.display_meta[0].lines)
        )
        self.assertEqual(
            frame.display_meta[0].texts[1].display_text,
            "追踪中｜船".encode(),
        )

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


    def test_ground_litter_branch_keeps_native_resolution(self) -> None:
        pipeline = FakePipeline()
        config = {
            "gpu_id": 0,
            "batch_size": 2,
            "batch_push_timeout_us": 20_000,
            "mux_width": 2560,
            "mux_height": 1440,
            "source_latency_ms": 100,
            "encoder_iframe_interval": 25,
            "tracker_config": "/tracker.yml",
            "tracker_library": "/tracker.so",
            "ground_litter": {"enabled": True, "analysis_fps": 1.0},
            "streams": [
                {
                    "input_url": "rtsp://camera/walkway",
                    "output_url": "rtsp://mediamtx/detected/litter",
                    "bitrate_bps": 2_500_000,
                }
            ],
        }
        receiver = object()
        skip_probe = object()
        pipeline = FakePipeline()
        _add_pipeline_nodes(
            pipeline,
            config,
            Path("/runtime/nvinfer.txt"),
            object(),
            lambda _index: object(),
            object(),
            lambda _index: object(),
            lambda _index: object(),
            ground_litter_receiver=receiver,
            ground_litter_skip_probe=skip_probe,
        )

        self.assertEqual(
            pipeline.nodes["ground_litter_queue"][1]["leaky"],
            2,
        )
        self.assertEqual(
            pipeline.nodes["ground_litter_queue"][1]["max-size-buffers"],
            1,
        )
        self.assertEqual(
            pipeline.nodes["ground_litter_rgb_caps"][1]["caps"],
            "video/x-raw(memory:NVMM), format=RGB",
        )
        self.assertEqual(
            pipeline.nodes["ground_litter_sink"][1]["drop"],
            True,
        )
        self.assertIn(
            (
                "ground_litter_queue",
                "ground_litter_convert",
                "ground_litter_rgb_caps",
                "ground_litter_sink",
            ),
            pipeline.links,
        )
        self.assertIn(
            ("ground_litter_queue", skip_probe),
            pipeline.attachments,
        )
        self.assertIn(
            ("ground_litter_sink", receiver),
            pipeline.attachments,
        )

    def test_ground_litter_branch_requires_its_receiver(self) -> None:
        pipeline = FakePipeline()
        config = {
            "gpu_id": 0,
            "batch_size": 1,
            "batch_push_timeout_us": 20_000,
            "mux_width": 1280,
            "mux_height": 720,
            "source_latency_ms": 100,
            "encoder_iframe_interval": 25,
            "tracker_config": "/tracker.yml",
            "tracker_library": "/tracker.so",
            "ground_litter": {"enabled": True},
            "streams": [
                {
                    "input_url": "rtsp://camera/one",
                    "output_url": "rtsp://mediamtx/detected/one",
                    "bitrate_bps": 2_000_000,
                }
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "缺少旁路组件"):
            _add_pipeline_nodes(
                pipeline,
                config,
                Path("/runtime/nvinfer.txt"),
                object(),
                lambda _index: object(),
                object(),
                lambda _index: object(),
                lambda _index: object(),
            )

    def test_ground_litter_osd_draws_zones_and_cached_boxes(self) -> None:
        frame = FakeFrame([])
        batch = FakeBatch([frame])
        cache = GroundLitterResultCache()
        cache.store_snapshot(
            0,
            GroundLitterSnapshot(
                state="running",
                detections=tuple(
                    GroundLitterDetection(
                        object_id=index + 1,
                        rectangle=NormalizedRect(
                            0.05 + index * 0.1,
                            0.2,
                            0.07,
                            0.15,
                        ),
                        confidence=0.5,
                        class_name="Plastic",
                    )
                    for index in range(5)
                ),
                result_version=3,
                updated_at=time.monotonic(),
            ),
        )
        fake_osd = SimpleNamespace(
            Color=FakeColor,
            Line=FakeLine,
            Text=FakeText,
            FontFamily=SimpleNamespace(Serif="serif"),
        )
        options = GroundLitterDetectionOptions(
            enabled=True,
            display_class=True,
            zones=(
                GroundLitterZone(
                    region_id="merchant_01",
                    polygon=(
                        (0.1, 0.1),
                        (0.5, 0.1),
                        (0.5, 0.5),
                        (0.1, 0.5),
                    ),
                ),
            ),
        )

        OverlayProcessor._draw_ground_litter(
            batch,
            frame,
            cache.snapshot(0),
            options,
            fake_osd,
            width=1000,
            height=500,
            now=time.monotonic(),
        )

        # One display meta for the zone outline plus two box batches, and the
        # zone outline is 4 lines while 5 boxes become 16 + 4 line segments.
        line_counts = [len(item.lines) for item in frame.display_meta]
        self.assertEqual(line_counts, [4, 16, 4])
        self.assertEqual(
            frame.display_meta[1].texts[0].display_text,
            "疑似垃圾：5".encode(),
        )
        self.assertEqual(
            frame.display_meta[1].texts[1].display_text,
            "疑似垃圾 Plastic".encode(),
        )

    def test_ground_litter_osd_hides_stale_snapshots(self) -> None:
        frame = FakeFrame([])
        batch = FakeBatch([frame])
        cache = GroundLitterResultCache()
        cache.store_snapshot(
            0,
            GroundLitterSnapshot(
                state="running",
                detections=(
                    GroundLitterDetection(
                        object_id=1,
                        rectangle=NormalizedRect(0.1, 0.1, 0.05, 0.05),
                        confidence=0.5,
                    ),
                ),
                result_version=1,
                updated_at=1.0,
            ),
        )
        fake_osd = SimpleNamespace(
            Color=FakeColor,
            Line=FakeLine,
            Text=FakeText,
            FontFamily=SimpleNamespace(Serif="serif"),
        )
        options = GroundLitterDetectionOptions(
            enabled=True,
            zones=(
                GroundLitterZone(
                    region_id="z1",
                    polygon=(
                        (0.1, 0.1),
                        (0.5, 0.1),
                        (0.5, 0.5),
                        (0.1, 0.5),
                    ),
                ),
            ),
        )

        OverlayProcessor._draw_ground_litter(
            batch,
            frame,
            cache.snapshot(0),
            options,
            fake_osd,
            width=1000,
            height=500,
            now=1_000.0,
        )

        self.assertEqual(
            frame.display_meta[1].texts[0].display_text,
            "疑似垃圾：未发现".encode(),
        )
        self.assertEqual(len(frame.display_meta), 2)
        self.assertEqual(len(frame.display_meta[1].lines), 0)

    def test_ground_litter_frame_processor_sends_native_frames(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.accepted: list[int] = []
                self.submitted: list[dict] = []

            def accepts(self, pad_index: int, *, timestamp=None) -> bool:
                self.accepted.append(pad_index)
                return pad_index == 0

            def submit(
                self,
                pad_index: int,
                frame: object,
                *,
                timestamp: float,
                night: bool = False,
                actors: object = (),
            ) -> bool:
                self.submitted.append(
                    {
                        "pad": pad_index,
                        "frame": frame,
                        "night": night,
                        "actors": list(actors),
                    }
                )
                return True

        client = FakeClient()
        processor = GroundLitterFrameProcessor(
            client,
            night_by_pad={0: True},
            actor_classes_by_pad={0: (0, 2), 1: (0,)},
        )
        actor = SimpleNamespace(
            class_id=0,
            rect_params=SimpleNamespace(
                left=100.0,
                top=50.0,
                width=200.0,
                height=100.0,
            ),
        )
        ignored = SimpleNamespace(
            class_id=8,
            rect_params=SimpleNamespace(
                left=0.0,
                top=0.0,
                width=10.0,
                height=10.0,
            ),
        )
        frame = FakeFrame([actor, ignored])
        frame.pad_index = 0
        batch = FakeBatch([frame])
        frames = [np.zeros((500, 1000, 3), np.uint8)]

        processor.process(
            batch,
            frames,
            frame_width=1000,
            frame_height=500,
        )

        self.assertEqual(client.accepted, [0])
        self.assertEqual(len(client.submitted), 1)
        submitted = client.submitted[0]
        self.assertEqual(submitted["pad"], 0)
        self.assertTrue(submitted["night"])
        self.assertEqual(submitted["actors"], [(0.1, 0.1, 0.2, 0.2)])

    def test_ground_litter_occluders_include_actor_and_context_classes(self) -> None:
        options = GroundLitterDetectionOptions(
            actor_class_ids=(0, 2, 7),
            context_class_ids=(7, 13, 56),
        )
        self.assertEqual(
            _ground_litter_occluder_classes(options),
            (0, 2, 7, 13, 56),
        )

    def test_ground_litter_frame_processor_skips_disabled_pads(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.submitted: list[int] = []

            def accepts(self, pad_index: int, *, timestamp=None) -> bool:
                return False

            def submit(self, pad_index: int, *_args, **_kwargs) -> bool:
                self.submitted.append(pad_index)
                return True

        client = FakeClient()
        processor = GroundLitterFrameProcessor(
            client,
            actor_classes_by_pad={0: (0,)},
        )
        frame = FakeFrame([])
        processor.process(
            FakeBatch([frame]),
            [np.zeros((10, 10, 3), np.uint8)],
            frame_width=10,
            frame_height=10,
        )
        self.assertEqual(client.submitted, [])

    def test_ground_litter_metrics_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            metrics = MetricsState(
                stream_ids=["stream-1"],
                metrics_path=path,
                interval_seconds=0,
                minimum_healthy_fps=20,
                group_id="group",
                generation=1,
                ground_litter_enabled={"stream-1": True},
            )
            metrics.observe_ground_litter(
                "stream-1",
                state="running",
                count=2,
                result_version=7,
                updated_at=time.monotonic(),
                last_inference_ms=420.5,
                analyzed_frames=11,
                tile_count=9,
                message="",
            )
            # The periodic writer is driven by the per-frame observer, exactly
            # like the OSD path where both are called for the same frame.
            metrics.observe_frame("stream-1", detections=0)
            report = json.loads(path.read_text(encoding="utf-8"))
            stream = report["streams"]["stream-1"]

        self.assertTrue(stream["ground_litter_enabled"])
        self.assertEqual(stream["ground_litter_state"], "running")
        self.assertEqual(stream["ground_litter_count"], 2)
        self.assertEqual(stream["ground_litter_result_version"], 7)
        self.assertEqual(stream["ground_litter_tile_count"], 9)
        self.assertEqual(stream["ground_litter_analyzed_frames"], 11)
        self.assertAlmostEqual(
            stream["ground_litter_last_inference_ms"],
            420.5,
        )

    def test_load_policies_rebuilds_ground_litter_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            labels.write_text("person\nbicycle\n", encoding="utf-8")
            config = {
                "labels_path": str(labels),
                "streams": [
                    {
                        "stream_id": "stream-1",
                        "classes": [0],
                        "conf": 0.25,
                        "ground_litter": {
                            "enabled": True,
                            "analysis_fps": 1.5,
                            "confidence": 0.18,
                            "hold_seconds": 4,
                            "zones": [
                                {
                                    "region_id": "merchant_01",
                                    "polygon": [
                                        [0.3, 0.1],
                                        [0.4, 0.1],
                                        [0.35, 0.4],
                                    ],
                                    "minimum_short_side_px": 8,
                                    "minimum_box_area_px": 64,
                                }
                            ],
                        },
                    },
                    {"stream_id": "stream-2", "classes": None, "conf": 0.25},
                ],
            }

            policies = _load_policies(config)

        self.assertTrue(policies[0].ground_litter.enabled)
        self.assertEqual(policies[0].ground_litter.analysis_fps, 1.5)
        self.assertEqual(
            policies[0].ground_litter.region_ids,
            ("merchant_01",),
        )
        self.assertEqual(
            policies[0].ground_litter.zones[0].minimum_short_side_px,
            8,
        )
        self.assertFalse(policies[1].ground_litter.enabled)
        self.assertEqual(policies[1].ground_litter.zones, ())


if __name__ == "__main__":
    unittest.main()
