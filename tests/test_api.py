from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from rtsp_annotator.api import StreamCreateRequest, create_app
from rtsp_annotator.events import EventRecord


class FakeManager:
    def list_models(self) -> list[str]:
        return ["yolo26s.pt"]

    def create(self, spec: object) -> dict[str, object]:
        return {
            "stream_id": "abc123",
            "status": "starting",
            "rtsp_url": "rtsp://viewer:password@example.com:28554/detected/abc123",
            "model": "yolo26s.pt",
            "classes": [0],
            "created_at": "2026-07-28T11:00:00+00:00",
            "exit_code": None,
        }

    def list(self) -> list[dict[str, object]]:
        return []

    def shutdown(self) -> None:
        pass


class ApiTests(unittest.TestCase):
    def test_event_request_maps_to_stream_spec(self) -> None:
        request = StreamCreateRequest.model_validate(
            {
                "input_url": "rtsp://camera/live",
                "event_detection": {
                    "enabled": True,
                    "rois": [
                        {
                            "id": "shop_entrance",
                            "polygon": [
                                [0.1, 0.1],
                                [0.9, 0.1],
                                [0.9, 0.9],
                                [0.1, 0.9],
                            ],
                            "rules": {
                                "person_dwell_seconds": 20,
                                "garbage_persistence_seconds": 15,
                            },
                        }
                    ],
                    "garbage": {
                        "enabled": True,
                        "analysis_fps": 1,
                        "detection_mode": "pile",
                        "background_change_enabled": False,
                        "display_detections": True,
                        "display_hold_seconds": 2.0,
                        "maximum_display_boxes": 12,
                        "minimum_pile_detections": 3,
                        "pile_merge_distance": 0.2,
                        "pile_box_padding": 0.05,
                    },
                },
            }
        )

        options = request.to_spec().event_detection

        self.assertTrue(options.enabled)
        self.assertEqual(options.rois[0].roi_id, "shop_entrance")
        self.assertEqual(options.rois[0].rules.person_dwell_seconds, 20)
        self.assertTrue(options.garbage.enabled)
        self.assertEqual(options.garbage.analysis_fps, 1)
        self.assertEqual(options.garbage.detection_mode, "pile")
        self.assertFalse(options.garbage.background_change_enabled)
        self.assertTrue(options.garbage.display_detections)
        self.assertEqual(options.garbage.display_hold_seconds, 2.0)
        self.assertEqual(options.garbage.maximum_display_boxes, 12)
        self.assertEqual(options.garbage.minimum_pile_detections, 3)
        self.assertEqual(options.garbage.pile_merge_distance, 0.2)
        self.assertEqual(options.garbage.pile_box_padding, 0.05)

    def test_event_request_can_disable_vehicle_classes(self) -> None:
        request = StreamCreateRequest.model_validate(
            {
                "input_url": "rtsp://camera/live",
                "event_detection": {
                    "enabled": True,
                    "person_classes": [0],
                    "vehicle_classes": [],
                    "rois": [
                        {
                            "id": "door",
                            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        }
                    ],
                },
            }
        )

        self.assertEqual(
            request.to_spec().event_detection.vehicle_classes,
            (),
        )

    def test_night_vision_request_is_independent_from_day_confidence(self) -> None:
        request = StreamCreateRequest.model_validate(
            {
                "input_url": "rtsp://camera/live",
                "conf": 0.30,
                "night_vision": {
                    "enabled": True,
                    "confidence": 0.17,
                    "input_gain": 1.2,
                    "plate_detector_confidence": 0.19,
                },
            }
        )
        spec = request.to_spec()

        self.assertEqual(spec.conf, 0.30)
        self.assertTrue(spec.night_vision.enabled)
        self.assertEqual(spec.night_vision.confidence, 0.17)
        self.assertEqual(spec.night_vision.input_gain, 1.2)
        self.assertEqual(
            spec.night_vision.plate_detector_confidence,
            0.19,
        )

    def test_night_vision_is_disabled_by_default(self) -> None:
        request = StreamCreateRequest.model_validate(
            {"input_url": "rtsp://camera/live", "conf": 0.27}
        )
        spec = request.to_spec()

        self.assertFalse(spec.night_vision.enabled)
        self.assertEqual(spec.conf, 0.27)

    def test_chinese_license_plate_request_maps_to_stream_spec(self) -> None:
        request = StreamCreateRequest.model_validate(
            {
                "input_url": "rtsp://camera/live",
                "license_plate": {
                    "enabled": True,
                    "detector_interval": 3,
                    "minimum_confirmations": 3,
                },
            }
        )
        spec = request.to_spec()
        self.assertTrue(spec.license_plate.enabled)
        self.assertEqual(spec.license_plate.detector_interval, 3)
        self.assertEqual(spec.license_plate.minimum_confirmations, 3)
        self.assertEqual(spec.license_plate.vehicle_classes, (2, 3, 5, 7))

    def test_auth_validation_and_create_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Path(root, "yolo26s.pt").touch()
            config_path = Path(root, "api.json")
            config = {
                "api": {
                    "key": "test-api-key-1234",
                    "host": "127.0.0.1",
                    "port": 8080,
                },
                "models": {"root": str(root)},
                "rtsp": {
                    "internal_base_url": "rtsp://mediamtx:8554",
                    "public_base_url": "rtsp://example.com:28554",
                    "publish_user": "publisher",
                    "publish_password": "publish-password",
                    "read_user": "viewer",
                    "read_password": "read-password",
                },
                "inference": {
                    "device": "cpu",
                    "half": False,
                    "max_streams": 1,
                    "startup_grace_seconds": 0,
                },
                "events": {"storage_root": str(root / "events")},
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            application = create_app(config_path)
            with TestClient(application) as client:
                client.app.state.manager = FakeManager()

                health = client.get("/health")
                unauthorized = client.get(
                    "/v1/models",
                    headers={"X-API-Key": "wrong"},
                )
                models = client.get(
                    "/v1/models",
                    headers={"X-API-Key": "test-api-key-1234"},
                )
                invalid = client.post(
                    "/v1/streams",
                    headers={"X-API-Key": "test-api-key-1234"},
                    json={"input_url": "http://not-rtsp"},
                )
                created = client.post(
                    "/v1/streams",
                    headers={"X-API-Key": "test-api-key-1234"},
                    json={
                        "input_url": "rtsp://camera-user:secret@camera/live",
                        "model": "yolo26s.pt",
                        "classes": [0],
                    },
                )
                publish_auth = client.post(
                    "/internal/mediamtx/auth",
                    json={
                        "user": "publisher",
                        "password": "publish-password",
                        "action": "publish",
                        "path": "detected/abc123",
                    },
                )
                rejected_read = client.post(
                    "/internal/mediamtx/auth",
                    json={
                        "user": "publisher",
                        "password": "publish-password",
                        "action": "read",
                        "path": "detected/abc123",
                    },
                )
                event = EventRecord.create(
                    stream_id="abc123",
                    event_type="zone_dwell",
                    roi_id="door",
                    message="人员区域停留",
                )
                snapshot = root / "events" / "abc123" / "media" / "x.jpg"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(b"jpeg")
                event.snapshot_path = str(snapshot)
                client.app.state.events.append(event)
                event_list = client.get(
                    "/v1/streams/abc123/events",
                    headers={"X-API-Key": "test-api-key-1234"},
                )
                confirmed = client.post(
                    f"/v1/events/{event.event_id}/confirm",
                    headers={"X-API-Key": "test-api-key-1234"},
                )
                event_snapshot = client.get(
                    f"/v1/events/{event.event_id}/snapshot",
                    headers={"X-API-Key": "test-api-key-1234"},
                )

        self.assertEqual(health.status_code, 200)
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(models.json(), {"models": ["yolo26s.pt"]})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json()["stream_id"], "abc123")
        self.assertNotIn("camera-user", created.text)
        self.assertEqual(publish_auth.status_code, 200)
        self.assertEqual(rejected_read.status_code, 401)
        self.assertEqual(event_list.status_code, 200)
        self.assertEqual(event_list.json()[0]["event_type"], "zone_dwell")
        self.assertEqual(confirmed.json()["status"], "confirmed")
        self.assertEqual(event_snapshot.content, b"jpeg")
