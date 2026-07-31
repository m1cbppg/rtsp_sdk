from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtsp_annotator.stream_manager import (
    ManagerSettings,
    ModelNotFoundError,
    NightVisionOptions,
    StreamCapacityError,
    StreamManager,
    StreamSpec,
    authenticated_rtsp_url,
)


class FakeProcess:
    next_pid = 1000

    def __init__(self, command: list[str], **kwargs: object) -> None:
        self.command = command
        self.kwargs = kwargs
        self.returncode: int | None = None
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9


def make_settings(model_root: Path, **overrides: object) -> ManagerSettings:
    values: dict[str, object] = {
        "model_root": model_root,
        "internal_rtsp_base_url": "rtsp://mediamtx:8554",
        "public_rtsp_base_url": "rtsp://14.21.88.97:28554",
        "publish_user": "publisher",
        "publish_password": "publish-password",
        "read_user": "viewer",
        "read_password": "read-password",
        "device": "cuda:0",
        "half": True,
        "max_streams": 1,
        "startup_grace_seconds": 0.0,
    }
    values.update(overrides)
    return ManagerSettings(**values)  # type: ignore[arg-type]


class AuthenticatedUrlTests(unittest.TestCase):
    def test_credentials_and_path_are_added(self) -> None:
        url = authenticated_rtsp_url(
            "rtsp://example.com:28554/base",
            "view user",
            "p@ss",
            "detected/abc",
        )
        self.assertEqual(
            url,
            "rtsp://view%20user:p%40ss@example.com:28554/base/detected/abc",
        )


class StreamManagerTests(unittest.TestCase):
    def test_python_backend_rejects_night_vision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Path(root, "model.pt").touch()
            manager = StreamManager(make_settings(root), FakeProcess)

            with self.assertRaisesRegex(
                ModelNotFoundError,
                "夜间增强仅支持DeepStream后端",
            ):
                manager.create(
                    StreamSpec(
                        input_url="rtsp://camera/night",
                        model="model.pt",
                        night_vision=NightVisionOptions(enabled=True),
                    )
                )

    def test_create_returns_public_url_and_safe_command(self) -> None:
        created_processes: list[FakeProcess] = []

        def factory(command: list[str], **kwargs: object) -> FakeProcess:
            process = FakeProcess(command, **kwargs)
            created_processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Path(root, "model.pt").touch()
            label_map = Path(root, "labels.json")
            label_map.write_text('{"person":"人员"}', encoding="utf-8")
            manager = StreamManager(
                make_settings(root, label_map_path=label_map),
                factory,
            )
            result = manager.create(
                StreamSpec(
                    input_url="rtsp://camera/live",
                    model="model.pt",
                    classes=(0, 2),
                    roi=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
                )
            )

        self.assertEqual(result["status"], "running")
        self.assertIn("@14.21.88.97:28554/detected/", result["rtsp_url"])
        command = created_processes[0].command
        self.assertIn("rtsp://camera/live", command)
        self.assertIn("rtsp://publisher:publish-password@mediamtx:8554/", " ".join(command))
        self.assertEqual(command[command.index("--classes") + 1], "0,2")
        self.assertIn("--half", command)
        self.assertEqual(
            command[command.index("--label-map") + 1],
            str(label_map),
        )
        self.assertNotIn("rtsp://camera/live", str(result))

    def test_model_must_be_inside_model_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Path(root, "model.pt").touch()
            manager = StreamManager(make_settings(root), FakeProcess)

            with self.assertRaises(ModelNotFoundError):
                manager.create(
                    StreamSpec(
                        input_url="rtsp://camera/live",
                        model="../model.pt",
                    )
                )

    def test_capacity_prevents_second_active_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Path(root, "model.pt").touch()
            manager = StreamManager(make_settings(root), FakeProcess)
            manager.create(
                StreamSpec(
                    input_url="rtsp://camera/one",
                    model="model.pt",
                )
            )

            with self.assertRaises(StreamCapacityError):
                manager.create(
                    StreamSpec(
                        input_url="rtsp://camera/two",
                        model="model.pt",
                    )
                )

    def test_stop_terminates_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Path(root, "model.pt").touch()
            manager = StreamManager(make_settings(root), FakeProcess)
            created = manager.create(
                StreamSpec(
                    input_url="rtsp://camera/live",
                    model="model.pt",
                )
            )

            with patch("rtsp_annotator.stream_manager.os.killpg") as killpg:
                stopped = manager.stop(created["stream_id"])

        self.assertEqual(stopped["status"], "stopped")
        killpg.assert_called_once()

    def test_list_models_only_returns_pt_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Path(root, "b.pt").touch()
            Path(root, "a.pt").touch()
            Path(root, "notes.txt").touch()
            manager = StreamManager(make_settings(root), FakeProcess)

            self.assertEqual(manager.list_models(), ["a.pt", "b.pt"])
