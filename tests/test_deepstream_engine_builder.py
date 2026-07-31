from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtsp_annotator.deepstream_engine_builder import (
    build_auxiliary_engine,
    build_engine,
    engine_path_for,
)


class DeepStreamEngineBuilderTests(unittest.TestCase):
    def test_lpr_engine_uses_dynamic_object_batch_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "lpr.onnx"
            engine = root / "lpr.engine"
            model.touch()

            def fake_run(command: list[str], check: bool) -> None:
                self.assertTrue(check)
                Path(
                    next(
                        item.split("=", 1)[1]
                        for item in command
                        if item.startswith("--saveEngine=")
                    )
                ).write_bytes(b"engine")

            with patch(
                "rtsp_annotator.deepstream_engine_builder.subprocess.run",
                side_effect=fake_run,
            ) as run:
                build_auxiliary_engine(
                    model,
                    engine,
                    input_name="image_input",
                    input_shape=(3, 48, 96),
                    batch_size=16,
                )

        command = run.call_args.args[0]
        self.assertIn("--minShapes=image_input:1x3x48x96", command)
        self.assertIn("--optShapes=image_input:4x3x48x96", command)
        self.assertIn("--maxShapes=image_input:16x3x48x96", command)

    def test_tensorflow_style_input_name_is_quoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "lpd.onnx"
            engine = root / "lpd.engine"
            model.touch()

            def fake_run(command: list[str], check: bool) -> None:
                self.assertTrue(check)
                Path(
                    next(
                        item.split("=", 1)[1]
                        for item in command
                        if item.startswith("--saveEngine=")
                    )
                ).write_bytes(b"engine")

            with patch(
                "rtsp_annotator.deepstream_engine_builder.subprocess.run",
                side_effect=fake_run,
            ) as run:
                build_auxiliary_engine(
                    model,
                    engine,
                    input_name="input_1:0",
                    input_shape=(3, 1168, 720),
                    batch_size=16,
                )

        command = run.call_args.args[0]
        self.assertIn(
            "--minShapes='input_1:0':1x3x1168x720",
            command,
        )
        self.assertIn(
            "--optShapes='input_1:0':4x3x1168x720",
            command,
        )
        self.assertIn(
            "--maxShapes='input_1:0':16x3x1168x720",
            command,
        )

    def test_engine_name_matches_manager_contract(self) -> None:
        path = engine_path_for(
            Path("/models/yolo26s.onnx"),
            Path("/engines"),
            imgsz=640,
            batch_size=2,
            gpu_id=0,
        )
        self.assertEqual(
            path,
            Path("/engines/yolo26s_640_b2_gpu0_fp16.engine"),
        )

    def test_build_is_atomic_and_uses_dynamic_batch_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx_path = root / "model.onnx"
            engine_path = root / "model.engine"
            onnx_path.touch()

            def fake_run(command: list[str], check: bool) -> None:
                self.assertTrue(check)
                target = next(
                    item.split("=", 1)[1]
                    for item in command
                    if item.startswith("--saveEngine=")
                )
                Path(target).write_bytes(b"engine")

            with patch(
                "rtsp_annotator.deepstream_engine_builder.subprocess.run",
                side_effect=fake_run,
            ) as run_mock:
                build_engine(
                    onnx_path,
                    engine_path,
                    imgsz=640,
                    batch_size=2,
                )
                build_engine(
                    onnx_path,
                    engine_path,
                    imgsz=640,
                    batch_size=2,
                )

        command = run_mock.call_args.args[0]
        self.assertIn("--minShapes=input:1x3x640x640", command)
        self.assertIn("--optShapes=input:2x3x640x640", command)
        self.assertIn("--maxShapes=input:2x3x640x640", command)
        self.assertEqual(run_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
