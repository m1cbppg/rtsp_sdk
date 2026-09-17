from __future__ import annotations

import json
import signal
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, call, patch

from rtsp_annotator.deepstream_manager import (
    DeepStreamManagerSettings,
    DeepStreamStreamManager,
)
from rtsp_annotator.license_plate import LicensePlateOptions
from rtsp_annotator.events import (
    EventDetectionOptions,
    EventRoiOptions,
    GarbageAnalysisOptions,
)
from rtsp_annotator.gas_cylinder import GasCylinderOptions
from rtsp_annotator.ground_litter_detection import (
    GroundLitterDetectionOptions,
    GroundLitterZone,
)
from rtsp_annotator.fishing_risk import (
    FishingRiskOptions,
    FishingRiskRuleOptions,
    FishingRiskZoneOptions,
)
from rtsp_annotator.ptz_verification import PtzVerificationOptions
from rtsp_annotator.stream_manager import (
    ManagerSettings,
    ModelNotFoundError,
    NightVisionOptions,
    StreamSpec,
)
from rtsp_annotator.vessel_detection import VesselDetectionOptions


class FakeProcess:
    next_pid = 50_000

    def __init__(self, command: list[str], **_kwargs: object) -> None:
        self.command = command
        self.returncode: int | None = None
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = -9


def make_settings(root: Path) -> DeepStreamManagerSettings:
    models = root / "models"
    engines = root / "engines"
    runtime = root / "runtime"
    models.mkdir()
    parser = root / "parser.so"
    tracker = root / "tracker.so"
    tracker_config = root / "tracker.yml"
    parser.touch()
    tracker.touch()
    tracker_config.touch()
    return DeepStreamManagerSettings(
        manager=ManagerSettings(
            model_root=models,
            internal_rtsp_base_url="rtsp://mediamtx:8554",
            public_rtsp_base_url="rtsp://example.com:38554",
            publish_user="publisher",
            publish_password="publish-password",
            read_user="viewer",
            read_password="read-password",
            device="cuda:0",
            half=True,
            encoder="h264_nvenc",
            max_streams=4,
            startup_grace_seconds=0,
        ),
        onnx_root=models,
        engine_root=engines,
        runtime_root=runtime,
        parser_library=parser,
        tracker_library=tracker,
        tracker_config=tracker_config,
        lpr_model_root=models / "lpr",
        lpr_parser_library=parser,
    )


class DeepStreamManagerTests(unittest.TestCase):
    def test_emergency_home_command_is_enqueued_for_ptz_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "\n".join(f"class_{index}" for index in range(9)),
                encoding="utf-8",
            )
            processes: list[FakeProcess] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            manager = DeepStreamStreamManager(settings, factory)
            with patch("os.killpg"):
                created = manager.create(
                    StreamSpec(
                        "rtsp://camera/harbor",
                        model="model.pt",
                        vessel_detection=VesselDetectionOptions(enabled=True),
                        ptz_verification=PtzVerificationOptions(
                            enabled=True,
                            camera_id="camera-01",
                        ),
                    )
                )
                response = manager.return_ptz_home(created["stream_id"])
                command_paths = list(
                    settings.runtime_root.glob("*/control/ptz-home-*.json")
                )
                self.assertEqual(len(command_paths), 1)
                command = json.loads(
                    command_paths[0].read_text(encoding="utf-8")
                )
                manager.shutdown()

        self.assertEqual(response["status"], "accepted")
        self.assertEqual(command["action"], "return_home")
        self.assertEqual(command["stream_id"], created["stream_id"])
        self.assertEqual(command["request_id"], response["request_id"])

    def test_vessel_only_mode_reports_camera_control_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "\n".join(f"class_{index}" for index in range(9)),
                encoding="utf-8",
            )
            processes: list[FakeProcess] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            manager = DeepStreamStreamManager(settings, factory)
            with patch("os.killpg"):
                result = manager.create(
                    StreamSpec(
                        "rtsp://camera/harbor",
                        model="model.pt",
                        vessel_detection=VesselDetectionOptions(enabled=True),
                    )
                )
                payload = json.loads(
                    Path(processes[-1].command[-1]).read_text(encoding="utf-8")
                )
                manager.shutdown()

        self.assertTrue(payload["streams"][0]["vessel_detection"]["enabled"])
        self.assertFalse(payload["streams"][0]["ptz_verification"]["enabled"])
        self.assertEqual(result["ptz_verification"]["state"], "disabled")
        self.assertEqual(
            result["ptz_verification"]["integration_mode"],
            "detection_only",
        )
        self.assertFalse(result["ptz_verification"]["continuous_tracking"])

    def test_ptz_verification_rejects_fixed_view_feature_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            manager = DeepStreamStreamManager(settings, Mock())

            with self.assertRaisesRegex(
                ModelNotFoundError,
                "不能与固定视角功能同时启用",
            ):
                manager.create(
                    StreamSpec(
                        "rtsp://camera/harbor",
                        vessel_detection=VesselDetectionOptions(enabled=True),
                        ptz_verification=PtzVerificationOptions(
                            enabled=True,
                            camera_id="camera-01",
                        ),
                        license_plate=LicensePlateOptions(enabled=True),
                    )
                )

    def test_vessel_stream_gets_high_resolution_lossy_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "\n".join(f"class_{index}" for index in range(9)),
                encoding="utf-8",
            )
            processes: list[FakeProcess] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            manager = DeepStreamStreamManager(settings, factory)
            with patch("os.killpg"):
                result = manager.create(
                    StreamSpec(
                        "rtsp://camera/harbor",
                        model="model.pt",
                        vessel_detection=VesselDetectionOptions(
                            enabled=True,
                            confidence=0.1,
                            imgsz=1280,
                            class_ids=(8,),
                            inference_regions=(
                                (0.0, 0.0, 1.0, 1.0),
                                (0.1, 0.2, 0.9, 0.75),
                            ),
                        ),
                        fishing_risk=FishingRiskOptions(
                            enabled=True,
                            zones=(
                                FishingRiskZoneOptions(
                                    "protected_water",
                                    (
                                        (0.0, 0.4),
                                        (1.0, 0.4),
                                        (1.0, 1.0),
                                        (0.0, 1.0),
                                    ),
                                ),
                            ),
                            rules=FishingRiskRuleOptions(
                                minimum_presence_seconds=45,
                            ),
                        ),
                        ptz_verification=PtzVerificationOptions(
                            enabled=True,
                            camera_id="camera-01",
                            zoom_strategy="adaptive",
                            adaptive_max_step=5,
                        ),
                    )
                )
                payload = json.loads(
                    Path(processes[-1].command[-1]).read_text(
                        encoding="utf-8"
                    )
                )
                manager.shutdown()

        self.assertTrue(payload["vessel_detection"]["enabled"])
        self.assertEqual(payload["vessel_detection"]["imgsz"], 1280)
        self.assertEqual(
            payload["vessel_detection"]["model_path"],
            str((models / "model.pt").resolve()),
        )
        stream_options = payload["streams"][0]["vessel_detection"]
        self.assertEqual(stream_options["class_ids"], [8])
        self.assertEqual(len(stream_options["inference_regions"]), 2)
        self.assertTrue(result["vessel_detection"]["enabled"])
        self.assertEqual(result["vessel_detection"]["model"], "model.pt")
        self.assertTrue(payload["streams"][0]["fishing_risk"]["enabled"])
        self.assertEqual(
            payload["streams"][0]["fishing_risk"]["zones"][0]["id"],
            "protected_water",
        )
        self.assertEqual(
            payload["streams"][0]["fishing_risk"]["rules"][
                "minimum_presence_seconds"
            ],
            45,
        )
        self.assertTrue(result["fishing_risk"]["enabled"])
        self.assertTrue(payload["streams"][0]["ptz_verification"]["enabled"])
        self.assertEqual(
            payload["streams"][0]["ptz_verification"]["zoom_strategy"],
            "adaptive",
        )
        self.assertEqual(
            payload["streams"][0]["ptz_verification"]["adaptive_max_step"],
            5,
        )
        self.assertTrue(result["ptz_verification"]["enabled"])
        self.assertEqual(result["ptz_verification"]["zoom_strategy"], "adaptive")

    def test_vessel_input_cannot_claim_resolution_lost_by_mux(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "\n".join(f"class_{index}" for index in range(9)),
                encoding="utf-8",
            )
            manager = DeepStreamStreamManager(settings, FakeProcess)

            with self.assertRaisesRegex(Exception, "mux尺寸"):
                manager.create(
                    StreamSpec(
                        "rtsp://camera/4k",
                        model="model.pt",
                        vessel_detection=VesselDetectionOptions(
                            enabled=True,
                            input_width=2560,
                            input_height=1440,
                        ),
                    )
                )

    def test_fishing_risk_can_be_disabled_without_changing_stream_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "\n".join(f"class_{index}" for index in range(9)),
                encoding="utf-8",
            )
            processes: list[FakeProcess] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            manager = DeepStreamStreamManager(settings, factory)
            risk = FishingRiskOptions(
                enabled=True,
                zones=(
                    FishingRiskZoneOptions(
                        "protected_water",
                        ((0, 0), (1, 0), (1, 1), (0, 1)),
                    ),
                ),
            )
            with patch("os.killpg"):
                created = manager.create(
                    StreamSpec(
                        "rtsp://camera/harbor",
                        model="model.pt",
                        vessel_detection=VesselDetectionOptions(enabled=True),
                        fishing_risk=risk,
                    )
                )
                updated = manager.update_fishing_risk(
                    created["stream_id"],
                    FishingRiskOptions(enabled=False),
                )
                payload = json.loads(
                    Path(processes[-1].command[-1]).read_text(
                        encoding="utf-8"
                    )
                )
                manager.shutdown()

        self.assertEqual(updated["stream_id"], created["stream_id"])
        self.assertEqual(updated["rtsp_url"], created["rtsp_url"])
        self.assertFalse(updated["fishing_risk"]["enabled"])
        self.assertTrue(updated["vessel_detection"]["enabled"])
        self.assertFalse(payload["streams"][0]["fishing_risk"]["enabled"])
        self.assertTrue(payload["streams"][0]["vessel_detection"]["enabled"])

    def test_gas_cylinder_stream_gets_lossy_sidecar_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "person\n",
                encoding="utf-8",
            )
            gas_root = models / "gas"
            profiles = gas_root / "profiles"
            profiles.mkdir(parents=True)
            gas_model = gas_root / "yoloe-26l-seg.pt"
            gas_model.touch()
            (profiles / "reference.jpg").touch()
            (profiles / "camera_01_ir.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "profile_id": "camera_01_ir",
                        "reference_image": "reference.jpg",
                        "reference_size": [1280, 720],
                        "prompts": [
                            {
                                "id": "regular",
                                "boxes": [[0.1, 0.1, 0.2, 0.3]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            settings = replace(
                settings,
                gas_cylinder_model_path=gas_model,
                gas_cylinder_profile_root=profiles,
            )
            processes: list[FakeProcess] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            manager = DeepStreamStreamManager(settings, factory)
            with patch("os.killpg"):
                result = manager.create(
                    StreamSpec(
                        "rtsp://camera/gas",
                        model="model.pt",
                        gas_cylinder=GasCylinderOptions(
                            enabled=True,
                            profile_id="camera_01_ir",
                            sample_count=11,
                        ),
                    )
                )
                payload = json.loads(
                    Path(processes[-1].command[-1]).read_text(
                        encoding="utf-8"
                    )
                )
                manager.shutdown()

        self.assertTrue(payload["gas_cylinder"]["enabled"])
        self.assertEqual(payload["gas_cylinder"]["input_width"], 1280)
        self.assertEqual(payload["gas_cylinder"]["imgsz"], 1280)
        self.assertEqual(
            payload["streams"][0]["gas_cylinder"]["sample_count"],
            11,
        )
        self.assertTrue(result["gas_cylinder"]["enabled"])

    def test_garbage_stream_gets_isolated_branch_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "person\ncar\n",
                encoding="utf-8",
            )
            garbage_root = models / "events"
            garbage_root.mkdir()
            garbage_onnx = garbage_root / "garbage.onnx"
            garbage_labels = garbage_root / "garbage.labels.txt"
            garbage_onnx.touch()
            garbage_labels.write_text(
                "plastic bottle\ngarbage bag\n",
                encoding="utf-8",
            )
            settings = replace(
                settings,
                garbage_onnx_path=garbage_onnx,
                garbage_labels_path=garbage_labels,
                garbage_parser_library=settings.parser_library,
            )
            processes: list[FakeProcess] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            options = EventDetectionOptions(
                enabled=True,
                person_classes=(0,),
                vehicle_classes=(1,),
                rois=(
                    EventRoiOptions(
                        roi_id="entrance",
                        polygon=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
                    ),
                ),
                garbage=GarbageAnalysisOptions(
                    enabled=True,
                    analysis_fps=3,
                    display_detections=True,
                    display_hold_seconds=2,
                    maximum_display_boxes=12,
                    prompts=("plastic bottle", "garbage bag"),
                ),
            )
            manager = DeepStreamStreamManager(settings, factory)
            with patch("os.killpg"):
                manager.create(
                    StreamSpec(
                        "rtsp://camera/events",
                        model="model.pt",
                        event_detection=options,
                    )
                )
                payload = json.loads(
                    Path(processes[-1].command[-1]).read_text(
                        encoding="utf-8"
                    )
                )
                manager.shutdown()

        self.assertTrue(payload["garbage"]["enabled"])
        self.assertEqual(payload["garbage"]["detection_mode"], "items")
        self.assertEqual(payload["garbage"]["analysis_fps"], 3)
        self.assertNotIn("interval", payload["garbage"])
        self.assertTrue(
            payload["streams"][0]["event_detection"]["enabled"]
        )
        stream_garbage = payload["streams"][0]["event_detection"][
            "garbage"
        ]
        self.assertTrue(stream_garbage["display_detections"])
        self.assertEqual(stream_garbage["display_hold_seconds"], 2)
        self.assertEqual(stream_garbage["maximum_display_boxes"], 12)

    def test_pile_mode_selects_street_garbage_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "person\ncar\n",
                encoding="utf-8",
            )
            garbage_root = models / "events"
            garbage_root.mkdir()
            pile_onnx = garbage_root / "street_garbage_pile.onnx"
            pile_labels = garbage_root / "street_garbage_pile.labels.txt"
            pile_onnx.touch()
            pile_labels.write_text(
                "pothole\nroad_damage\ngarbage\n",
                encoding="utf-8",
            )
            settings = replace(
                settings,
                garbage_pile_onnx_path=pile_onnx,
                garbage_pile_labels_path=pile_labels,
                garbage_parser_library=settings.parser_library,
            )
            processes: list[FakeProcess] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            options = EventDetectionOptions(
                enabled=True,
                person_classes=(0,),
                vehicle_classes=(1,),
                rois=(
                    EventRoiOptions(
                        roi_id="roadside",
                        polygon=((0, 0), (1, 0), (1, 1), (0, 1)),
                    ),
                ),
                garbage=GarbageAnalysisOptions(
                    enabled=True,
                    analysis_fps=1,
                    detection_mode="pile",
                    background_change_enabled=False,
                    minimum_confidence=0.15,
                ),
            )
            manager = DeepStreamStreamManager(settings, factory)
            with patch("os.killpg"):
                manager.create(
                    StreamSpec(
                        "rtsp://camera/pile",
                        model="model.pt",
                        event_detection=options,
                    )
                )
                payload = json.loads(
                    Path(processes[-1].command[-1]).read_text(
                        encoding="utf-8"
                    )
                )
                manager.shutdown()

        self.assertEqual(payload["garbage"]["detection_mode"], "pile")
        self.assertEqual(payload["garbage"]["onnx_path"], str(pile_onnx))
        self.assertEqual(payload["garbage"]["labels_path"], str(pile_labels))
        self.assertEqual(payload["garbage"]["label_count"], 3)

    def test_night_streams_with_different_confidence_share_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "person\ncar\n",
                encoding="utf-8",
            )
            processes: list[FakeProcess] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            manager = DeepStreamStreamManager(settings, factory)
            with patch("os.killpg"):
                for index, confidence in enumerate((0.17, 0.20)):
                    manager.create(
                        StreamSpec(
                            f"rtsp://camera/night-{index}",
                            model="model.pt",
                            night_vision=NightVisionOptions(
                                enabled=True,
                                confidence=confidence,
                            ),
                        )
                    )
                payload = json.loads(
                    Path(processes[-1].command[-1]).read_text(
                        encoding="utf-8"
                    )
                )
                manager.shutdown()

        self.assertEqual(len(processes), 2)
        self.assertEqual(len(payload["streams"]), 2)
        self.assertEqual(
            [item["conf"] for item in payload["streams"]],
            [0.17, 0.20],
        )

    def test_day_and_night_streams_use_separate_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "person\ncar\n",
                encoding="utf-8",
            )
            processes: list[FakeProcess] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            manager = DeepStreamStreamManager(settings, factory)
            day = manager.create(
                StreamSpec(
                    "rtsp://camera/day",
                    model="model.pt",
                    conf=0.30,
                )
            )
            night = manager.create(
                StreamSpec(
                    "rtsp://camera/night",
                    model="model.pt",
                    conf=0.30,
                    night_vision=NightVisionOptions(
                        enabled=True,
                        confidence=0.17,
                        input_gain=1.2,
                        plate_detector_confidence=0.19,
                    ),
                )
            )
            payloads = [
                json.loads(
                    Path(process.command[-1]).read_text(encoding="utf-8")
                )
                for process in processes
            ]
            with patch("os.killpg"):
                manager.shutdown()

        self.assertNotEqual(
            day["night_vision"]["profile"],
            night["night_vision"]["profile"],
        )
        self.assertEqual(len(processes), 2)
        self.assertFalse(payloads[0]["night_vision"]["enabled"])
        self.assertEqual(payloads[0]["streams"][0]["conf"], 0.30)
        self.assertTrue(payloads[1]["night_vision"]["enabled"])
        self.assertEqual(payloads[1]["night_vision"]["input_gain"], 1.2)
        self.assertEqual(payloads[1]["streams"][0]["conf"], 0.17)

    def test_lpr_stream_gets_its_own_group_and_worker_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "person\ncar\n",
                encoding="utf-8",
            )
            settings.lpr_model_root.mkdir()
            for name in (
                "LPDNet_CCPD_pruned_tao5.onnx",
                "ch_lprnet_baseline18_deployable.onnx",
                "ch_lp_characters.txt",
            ):
                (settings.lpr_model_root / name).touch()
            processes: list[FakeProcess] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            manager = DeepStreamStreamManager(settings, factory)
            without_lpr = manager.create(
                StreamSpec("rtsp://camera/one", model="model.pt")
            )
            with_lpr = manager.create(
                StreamSpec(
                    "rtsp://camera/two",
                    model="model.pt",
                    license_plate=LicensePlateOptions(enabled=True),
                )
            )
            payload = json.loads(
                Path(processes[-1].command[-1]).read_text(encoding="utf-8")
            )
            with patch("os.killpg"):
                manager.shutdown()

        self.assertNotEqual(
            processes[0].command[-1],
            processes[1].command[-1],
        )
        self.assertFalse(without_lpr["license_plate"]["enabled"])
        self.assertTrue(with_lpr["license_plate"]["enabled"])
        self.assertTrue(payload["license_plate"]["enabled"])
        self.assertEqual(
            payload["streams"][0]["license_plate"]["vehicle_classes"],
            [2, 3, 5, 7],
        )

    def test_four_streams_are_assigned_to_two_fixed_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            model = settings.manager.model_root / "model.pt"
            model.touch()
            (settings.onnx_root / "model.onnx").touch()
            (settings.onnx_root / "model.labels.txt").write_text(
                "person\ncar\n",
                encoding="utf-8",
            )
            processes: list[FakeProcess] = []
            launched_batch_sizes: list[int] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                payload = json.loads(
                    Path(command[-1]).read_text(encoding="utf-8")
                )
                launched_batch_sizes.append(payload["batch_size"])
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            manager = DeepStreamStreamManager(
                settings,
                process_factory=factory,
            )
            with patch("os.killpg"):
                records = [
                    manager.create(
                        StreamSpec(
                            f"rtsp://camera/{index}",
                            model="model.pt",
                            classes=(0,),
                            conf=0.3,
                            roi=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
                        )
                    )
                    for index in range(4)
                ]
                states = [
                    manager.get(record["stream_id"]) for record in records
                ]
                payloads = [
                    json.loads(
                        Path(process.command[-1]).read_text(encoding="utf-8")
                    )
                    for process in (processes[1], processes[3])
                ]
                manager.shutdown()

        groups = [state["metrics"] for state in states]
        self.assertEqual(groups, [None, None, None, None])
        self.assertEqual(len(processes), 4)
        self.assertEqual(launched_batch_sizes, [1, 2, 1, 2])
        self.assertEqual([len(item["streams"]) for item in payloads], [2, 2])
        self.assertTrue(all(item["batch_size"] == 2 for item in payloads))
        self.assertNotEqual(
            payloads[0]["group_id"],
            payloads[1]["group_id"],
        )
        self.assertTrue(
            all(
                item["engine_path"].endswith(
                    "model_640_b2_gpu0_fp16.engine"
                )
                for item in payloads
            )
        )

    def test_create_requires_onnx_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            (settings.manager.model_root / "model.pt").touch()
            manager = DeepStreamStreamManager(
                settings,
                process_factory=FakeProcess,
            )
            self.assertEqual(manager.list_models(), [])
            (settings.onnx_root / "model.onnx").touch()
            self.assertEqual(manager.list_models(), ["model.pt"])
            with self.assertRaisesRegex(ValueError, "标签文件不存在"):
                manager.create(
                    StreamSpec("rtsp://camera/live", model="model.pt")
                )

    def test_slow_worker_is_force_killed_after_short_grace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            manager = DeepStreamStreamManager(
                settings,
                process_factory=FakeProcess,
            )
            process = Mock()
            process.pid = 61_234
            process.poll.return_value = None
            process.wait.side_effect = [
                subprocess.TimeoutExpired("worker", 0.5),
                0,
            ]

            with patch("os.killpg") as killpg:
                manager._stop_process(process)

        self.assertEqual(
            killpg.call_args_list,
            [
                call(process.pid, signal.SIGTERM),
                call(process.pid, signal.SIGKILL),
            ],
        )
        self.assertEqual(
            process.wait.call_args_list,
            [call(timeout=0.5), call(timeout=3)],
        )

    def test_stopping_ptz_stream_waits_for_worker_home_and_job_finish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "\n".join(f"class_{index}" for index in range(9)),
                encoding="utf-8",
            )
            processes: list[FakeProcess] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            manager = DeepStreamStreamManager(settings, factory)
            with patch("os.killpg"):
                created = manager.create(
                    StreamSpec(
                        "rtsp://camera/harbor",
                        model="model.pt",
                        vessel_detection=VesselDetectionOptions(enabled=True),
                        ptz_verification=PtzVerificationOptions(
                            enabled=True,
                            camera_id="camera-01",
                            command_timeout_seconds=2,
                            maximum_off_home_seconds=5,
                        ),
                    )
                )
                manager.stop(created["stream_id"])

        # 3 command timeouts + 10 seconds coordinator budget + 2 seconds
        # worker shutdown overhead.
        self.assertEqual(processes[0].wait_timeouts, [18.0])

    def test_stopping_ptz_stream_requests_home_before_signalling_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "\n".join(f"class_{index}" for index in range(9)),
                encoding="utf-8",
            )
            manager = DeepStreamStreamManager(settings, FakeProcess)
            with patch("os.killpg"):
                created = manager.create(
                    StreamSpec(
                        "rtsp://camera/harbor",
                        model="model.pt",
                        vessel_detection=VesselDetectionOptions(enabled=True),
                        ptz_verification=PtzVerificationOptions(
                            enabled=True,
                            camera_id="camera-01",
                        ),
                    )
                )
                record = manager._records[created["stream_id"]]
                group = manager._groups[record.group_id]
                with patch.object(
                    manager,
                    "_read_metrics",
                    return_value={"ptz_verification_state": "running"},
                ), patch.object(
                    manager,
                    "_return_ptz_home_before_stop",
                    return_value=True,
                ) as return_home:
                    manager.stop(created["stream_id"])

            return_home.assert_called_once_with(record, group)

    def test_delete_home_handshake_waits_for_matching_worker_ack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "\n".join(f"class_{index}" for index in range(9)),
                encoding="utf-8",
            )
            manager = DeepStreamStreamManager(settings, FakeProcess)
            created = manager.create(
                StreamSpec(
                    "rtsp://camera/harbor",
                    model="model.pt",
                    vessel_detection=VesselDetectionOptions(enabled=True),
                    ptz_verification=PtzVerificationOptions(
                        enabled=True,
                        camera_id="camera-01",
                    ),
                )
            )
            record = manager._records[created["stream_id"]]
            group = manager._groups[record.group_id]
            running = {"ptz_verification_state": "running"}
            acknowledged = {
                "ptz_verification_state": "manual_hold",
                "ptz_last_return_home_request_id": "home-request",
                "ptz_manual_hold": True,
            }
            with patch.object(
                manager,
                "_enqueue_ptz_home",
                return_value="home-request",
            ), patch.object(
                manager,
                "_read_metrics",
                side_effect=[running, running, acknowledged],
            ), patch("time.sleep"):
                completed = manager._return_ptz_home_before_stop(
                    record,
                    group,
                )

        self.assertTrue(completed)


    def test_ground_litter_stream_gets_a_native_resolution_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "\n".join(f"class_{index}" for index in range(9)),
                encoding="utf-8",
            )
            litter_root = models / "litter"
            litter_root.mkdir()
            litter_model = litter_root / "turhancan_yolov8m_seg_trash.pt"
            litter_model.touch()
            processes: list[FakeProcess] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            manager = DeepStreamStreamManager(settings, factory)
            with patch("os.killpg"):
                result = manager.create(
                    StreamSpec(
                        "rtsp://camera/walkway",
                        model="model.pt",
                        classes=(0,),
                        ground_litter=GroundLitterDetectionOptions(
                            enabled=True,
                            confidence=0.18,
                            zones=(
                                GroundLitterZone(
                                    region_id="merchant_01",
                                    polygon=(
                                        (0.32, 0.15),
                                        (0.367, 0.15),
                                        (0.35, 0.3),
                                        (0.27, 0.3),
                                    ),
                                    minimum_short_side_px=8,
                                    minimum_box_area_px=64,
                                ),
                            ),
                        ),
                    )
                )
                payload = json.loads(
                    Path(processes[-1].command[-1]).read_text(
                        encoding="utf-8"
                    )
                )
                manager.shutdown()

        self.assertTrue(payload["ground_litter"]["enabled"])
        self.assertEqual(
            payload["ground_litter"]["model_path"],
            str(litter_model.resolve()),
        )
        self.assertEqual(payload["ground_litter"]["tile_size_px"], 640)
        self.assertEqual(payload["ground_litter"]["analysis_fps"], 1.0)
        stream_options = payload["streams"][0]["ground_litter"]
        self.assertTrue(stream_options["enabled"])
        self.assertEqual(
            stream_options["zones"][0]["region_id"],
            "merchant_01",
        )
        self.assertEqual(
            stream_options["zones"][0]["minimum_short_side_px"],
            8,
        )
        self.assertTrue(result["ground_litter"]["enabled"])
        self.assertEqual(result["ground_litter"]["region_count"], 1)
        self.assertEqual(
            result["ground_litter"]["model"],
            "turhancan_yolov8m_seg_trash.pt",
        )

    def test_display_detections_is_forwarded_to_the_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "class_0\n", encoding="utf-8"
            )
            processes: list[FakeProcess] = []

            def factory(command: list[str], **kwargs: object) -> FakeProcess:
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process

            manager = DeepStreamStreamManager(settings, factory)
            with patch("os.killpg"):
                manager.create(
                    StreamSpec(
                        "rtsp://camera/walkway",
                        model="model.pt",
                        display_detections=False,
                    )
                )
                payload = json.loads(
                    Path(processes[-1].command[-1]).read_text(
                        encoding="utf-8"
                    )
                )
                manager.shutdown()

        self.assertFalse(payload["streams"][0]["display_detections"])

    def test_ground_litter_model_may_be_addressed_with_subdirectory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "class_0\n", encoding="utf-8"
            )
            litter_root = models / "litter"
            litter_root.mkdir()
            (litter_root / "custom.pt").touch()
            manager = DeepStreamStreamManager(settings, FakeProcess)
            resolved = manager._resolve_ground_litter_model(
                "litter/custom.pt"
            )
            bare = manager._resolve_ground_litter_model("custom.pt")
        self.assertEqual(resolved.name, "custom.pt")
        self.assertEqual(resolved, bare)

    def test_missing_ground_litter_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "class_0\n", encoding="utf-8"
            )
            manager = DeepStreamStreamManager(settings, FakeProcess)
            with self.assertRaisesRegex(
                ModelNotFoundError,
                "零散垃圾模型不存在",
            ):
                manager.create(
                    StreamSpec(
                        "rtsp://camera/walkway",
                        model="model.pt",
                        ground_litter=GroundLitterDetectionOptions(
                            enabled=True,
                            zones=(
                                GroundLitterZone(
                                    region_id="z1",
                                    polygon=(
                                        (0.0, 0.0),
                                        (1.0, 0.0),
                                        (1.0, 1.0),
                                    ),
                                ),
                            ),
                        ),
                    )
                )

    def test_incompatible_ground_litter_tile_sizes_use_separate_groups(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            models = settings.manager.model_root
            (models / "model.pt").touch()
            (models / "model.onnx").touch()
            (models / "model.labels.txt").write_text(
                "class_0\n", encoding="utf-8"
            )
            (models / "litter").mkdir()
            (models / "litter" / "litter.pt").touch()
            manager = DeepStreamStreamManager(settings, FakeProcess)
            zone = GroundLitterZone(
                region_id="z1",
                polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            )
            with patch("os.killpg"):
                first = manager.create(
                    StreamSpec(
                        "rtsp://camera/a",
                        model="model.pt",
                        ground_litter=GroundLitterDetectionOptions(
                            enabled=True,
                            model="litter/litter.pt",
                            tile_size_px=640,
                            zones=(zone,),
                        ),
                    )
                )
                second = manager.create(
                    StreamSpec(
                        "rtsp://camera/b",
                        model="model.pt",
                        ground_litter=GroundLitterDetectionOptions(
                            enabled=True,
                            model="litter/litter.pt",
                            tile_size_px=320,
                            zones=(zone,),
                        ),
                    )
                )
                third = manager.create(
                    StreamSpec(
                        "rtsp://camera/c",
                        model="model.pt",
                        ground_litter=GroundLitterDetectionOptions(
                            enabled=True,
                            model="litter/litter.pt",
                            tile_size_px=640,
                            zones=(zone,),
                        ),
                    )
                )
                first_group = manager._records[
                    first["stream_id"]
                ].group_id
                second_group = manager._records[
                    second["stream_id"]
                ].group_id
                third_group = manager._records[
                    third["stream_id"]
                ].group_id
                manager.shutdown()

        self.assertNotEqual(first_group, second_group)
        self.assertEqual(first_group, third_group)


if __name__ == "__main__":
    unittest.main()
