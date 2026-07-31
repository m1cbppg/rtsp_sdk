from __future__ import annotations

import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from rtsp_annotator.deepstream_manager import (
    DeepStreamManagerSettings,
    DeepStreamStreamManager,
)
from rtsp_annotator.license_plate import LicensePlateOptions
from rtsp_annotator.stream_manager import (
    ManagerSettings,
    NightVisionOptions,
    StreamSpec,
)


class FakeProcess:
    next_pid = 50_000

    def __init__(self, command: list[str], **_kwargs: object) -> None:
        self.command = command
        self.returncode: int | None = None
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
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


if __name__ == "__main__":
    unittest.main()
