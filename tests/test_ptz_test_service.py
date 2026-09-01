from __future__ import annotations

import math
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from rtsp_annotator.ptz_test_service import (
    TestServiceSettings as PtzTestSettings,
    VirtualPtzTestController,
    VirtualViewport,
    create_test_app,
)


class FakePublisher:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def stop(self) -> None:
        return None


class VirtualViewportTests(unittest.TestCase):
    def test_locate_maps_screen_coordinate_inside_current_view(self) -> None:
        viewport = VirtualViewport(math.log(2.0) / 2.0, 64.0)
        viewport.locate(0.75, 0.5, 4)
        self.assertAlmostEqual(viewport.center_x, 0.75)
        self.assertAlmostEqual(viewport.zoom, 4.0)

        viewport.locate(0.75, 0.5, 0)
        self.assertAlmostEqual(viewport.center_x, 0.8125)
        self.assertAlmostEqual(viewport.zoom, 4.0)

    def test_home_resets_view(self) -> None:
        viewport = VirtualViewport(0.2, 64.0)
        viewport.locate(0.2, 0.8, 4)
        viewport.home()
        self.assertEqual((viewport.center_x, viewport.center_y, viewport.zoom), (0.5, 0.5, 1.0))

    def test_render_crops_and_resizes(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, 100:] = 255
        viewport = VirtualViewport(0.2, 64.0, center_x=0.75, zoom=2.0)
        rendered = viewport.render(frame, width=100, height=50)
        self.assertEqual(rendered.shape, (50, 100, 3))
        self.assertGreater(float(rendered.mean()), 245.0)


class PtzTestApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        source = root / "source.mp4"
        source.touch()
        self.settings = PtzTestSettings(
            source_path=source,
            publish_url="rtsp://publisher:password@mediamtx:8554/detected/test",
            api_key="0123456789abcdef",
            output_width=320,
            output_height=180,
            artifact_root=root / "artifacts",
        )
        self.controller = VirtualPtzTestController(
            self.settings,
            publisher=FakePublisher(),  # type: ignore[arg-type]
        )
        self.controller._latest_frame = np.full((180, 320, 3), 127, dtype=np.uint8)
        self.client = TestClient(create_test_app(self.settings, self.controller))
        self.client.__enter__()
        self.headers = {"X-Camera-Control-Key": self.settings.api_key}

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def wait_command(self, command_id: str) -> dict:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            response = self.client.get(f"/v1/commands/{command_id}", headers=self.headers)
            payload = response.json()
            if payload["state"] in {"completed", "failed"}:
                return payload
            time.sleep(0.01)
        self.fail("command did not finish")

    def test_camera_control_contract_locate_capture_and_home(self) -> None:
        cameras = self.client.get("/v1/cameras", headers=self.headers)
        self.assertEqual(cameras.status_code, 200)
        self.assertEqual(cameras.json()[0]["camera_id"], "virtual-river-01")

        submitted = self.client.post(
            "/v1/cameras/virtual-river-01/commands/locate",
            headers=self.headers,
            json={"x": 0.7, "y": 0.5, "zoom_delta": 4, "autofocus": False},
        ).json()
        located = self.wait_command(submitted["command_id"])
        self.assertEqual(located["state"], "completed")
        self.assertGreater(located["result"]["virtual_status"]["zoom_level"], 1)

        submitted = self.client.post(
            "/v1/cameras/virtual-river-01/commands/capture",
            headers=self.headers,
            json={"timeout_seconds": 5, "quality": 1},
        ).json()
        captured = self.wait_command(submitted["command_id"])
        self.assertEqual(captured["state"], "completed")
        image = self.client.get(captured["result"]["download_url"], headers=self.headers)
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.headers["content-type"], "image/jpeg")

        submitted = self.client.post(
            "/v1/cameras/virtual-river-01/commands/home",
            headers=self.headers,
            json={"timeout_seconds": 5},
        ).json()
        home = self.wait_command(submitted["command_id"])
        self.assertEqual(home["result"]["virtual_status"]["preset_id"], 1)

        report = self.client.get("/v1/test/report", headers=self.headers).json()
        self.assertTrue(report["checks"]["at_least_one_locate"])
        self.assertTrue(report["checks"]["capture_returned"])
        self.assertTrue(report["checks"]["home_returned"])

    def test_rejects_wrong_key_and_unknown_camera(self) -> None:
        self.assertEqual(self.client.get("/v1/cameras").status_code, 401)
        response = self.client.get("/v1/cameras/not-found/status", headers=self.headers)
        self.assertEqual(response.status_code, 404)

    def test_control_lease_blocks_other_clients_and_can_be_released(self) -> None:
        acquired = self.client.post(
            "/v1/cameras/virtual-river-01/lease",
            headers=self.headers,
            json={"owner": "rtsp:test-stream", "ttl_seconds": 60},
        )
        self.assertEqual(acquired.status_code, 200)
        token = acquired.json()["token"]

        blocked = self.client.post(
            "/v1/cameras/virtual-river-01/commands/locate",
            headers=self.headers,
            json={"x": 0.7, "y": 0.5, "zoom_delta": 4},
        )
        self.assertEqual(blocked.status_code, 409)

        lease_headers = {
            **self.headers,
            "X-Camera-Control-Lease": token,
        }
        submitted = self.client.post(
            "/v1/cameras/virtual-river-01/commands/locate",
            headers=lease_headers,
            json={"x": 0.7, "y": 0.5, "zoom_delta": 4},
        )
        self.assertEqual(submitted.status_code, 202)
        self.assertEqual(
            self.wait_command(submitted.json()["command_id"])["state"],
            "completed",
        )

        released = self.client.delete(
            "/v1/cameras/virtual-river-01/lease",
            headers=lease_headers,
        )
        self.assertEqual(released.status_code, 204)
        accepted = self.client.post(
            "/v1/cameras/virtual-river-01/commands/home",
            headers=self.headers,
            json={"timeout_seconds": 5},
        )
        self.assertEqual(accepted.status_code, 202)


if __name__ == "__main__":
    unittest.main()
