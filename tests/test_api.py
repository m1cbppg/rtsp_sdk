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

    def get(self, stream_id: str) -> dict[str, object]:
        if stream_id != "abc123":
            raise KeyError(stream_id)
        return self.create(object())

    def update_fishing_risk(
        self,
        stream_id: str,
        options: object,
    ) -> dict[str, object]:
        if stream_id != "abc123":
            raise KeyError(stream_id)
        result = self.create(object())
        result["fishing_risk"] = {
            "enabled": bool(getattr(options, "enabled", False)),
            "state": "disabled",
            "suspect_count": 0,
            "maximum_score": 0,
            "total_events": 0,
            "events_url": "/v1/streams/abc123/events",
        }
        return result

    def shutdown(self) -> None:
        pass


class ApiTests(unittest.TestCase):
    def test_fishing_risk_is_disabled_by_default(self) -> None:
        request = StreamCreateRequest.model_validate(
            {
                "input_url": "rtsp://camera/harbor",
                "vessel_detection": {"enabled": True},
            }
        )

        self.assertFalse(request.to_spec().fishing_risk.enabled)

    def test_fishing_risk_maps_to_camera_only_rule_options(self) -> None:
        request = StreamCreateRequest.model_validate(
            {
                "input_url": "rtsp://camera/harbor",
                "vessel_detection": {"enabled": True},
                "fishing_risk": {
                    "enabled": True,
                    "timezone": "Asia/Shanghai",
                    "zones": [
                        {
                            "id": "protected_water",
                            "polygon": [
                                [0, 0.4],
                                [1, 0.4],
                                [1, 1],
                                [0, 1],
                            ],
                        }
                    ],
                    "schedules": [
                        {
                            "id": "closure_2026",
                            "start_at": "2026-05-01T12:00:00+08:00",
                            "end_at": "2026-09-16T12:00:00+08:00",
                        }
                    ],
                    "rules": {
                        "minimum_presence_seconds": 45,
                        "loitering_seconds": 240,
                        "minimum_reversals": 3,
                    },
                    "alert_score": 60,
                },
            }
        )

        options = request.to_spec().fishing_risk

        self.assertTrue(options.enabled)
        self.assertEqual(options.zones[0].zone_id, "protected_water")
        self.assertEqual(options.schedules[0].schedule_id, "closure_2026")
        self.assertEqual(options.rules.minimum_presence_seconds, 45)
        self.assertEqual(options.rules.loitering_seconds, 240)
        self.assertEqual(options.rules.minimum_reversals, 3)
        self.assertEqual(options.alert_score, 60)

    def test_fishing_risk_requires_vessel_detection(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "必须启用vessel_detection",
        ):
            StreamCreateRequest.model_validate(
                {
                    "input_url": "rtsp://camera/harbor",
                    "fishing_risk": {
                        "enabled": True,
                        "zones": [
                            {
                                "id": "protected_water",
                                "polygon": [
                                    [0, 0],
                                    [1, 0],
                                    [1, 1],
                                    [0, 1],
                                ],
                            }
                        ],
                    },
                }
            )

    def test_vessel_request_maps_to_high_recall_sidecar_options(self) -> None:
        request = StreamCreateRequest.model_validate(
            {
                "input_url": "rtsp://camera/harbor",
                "vessel_detection": {
                    "enabled": True,
                    "confidence": 0.10,
                    "imgsz": 1280,
                    "class_ids": [8],
                    "inference_regions": [
                        [0, 0, 1, 1],
                        [0.1, 0.2, 0.9, 0.75],
                    ],
                    "roi": [[0, 0.2], [1, 0.2], [1, 1], [0, 1]],
                    "exclude_rois": [
                        [[0, 0], [0.1, 0], [0.1, 1], [0, 1]]
                    ],
                    "duplicate_containment_threshold": 0.75,
                    "large_box_area_threshold": 0.30,
                    "large_box_minimum_confidence": 0.35,
                    "maximum_box_area": 0.80,
                },
            }
        )

        options = request.to_spec().vessel_detection

        self.assertTrue(options.enabled)
        self.assertEqual(options.confidence, 0.10)
        self.assertEqual(options.imgsz, 1280)
        self.assertEqual(options.class_ids, (8,))
        self.assertEqual(len(options.inference_regions), 2)
        self.assertEqual(len(options.exclude_rois), 1)
        self.assertEqual(options.minimum_hits, 2)
        self.assertEqual(options.duplicate_containment_threshold, 0.75)
        self.assertEqual(options.large_box_area_threshold, 0.30)
        self.assertEqual(options.large_box_minimum_confidence, 0.35)
        self.assertEqual(options.maximum_box_area, 0.80)

    def test_gas_cylinder_defaults_cover_ir_exposure_cycle(self) -> None:
        request = StreamCreateRequest.model_validate(
            {
                "input_url": "rtsp://camera/live",
                "gas_cylinder": {"enabled": True},
            }
        )

        options = request.to_spec().gas_cylinder

        self.assertEqual(options.sample_interval_seconds, 3)
        self.assertEqual(options.minimum_confirmations, 4)
        self.assertEqual(options.alarm_threshold, 18)

    def test_gas_cylinder_request_maps_to_stream_spec(self) -> None:
        request = StreamCreateRequest.model_validate(
            {
                "input_url": "rtsp://camera/live",
                "gas_cylinder": {
                    "enabled": True,
                    "profile_id": "camera_01_ir",
                    "sample_count": 11,
                    "minimum_confirmations": 3,
                    "alarm_threshold": 20,
                    "display_ids": True,
                },
            }
        )

        options = request.to_spec().gas_cylinder

        self.assertTrue(options.enabled)
        self.assertEqual(options.profile_id, "camera_01_ir")
        self.assertEqual(options.sample_count, 11)
        self.assertEqual(options.minimum_confirmations, 3)
        self.assertEqual(options.alarm_threshold, 20)
        self.assertTrue(options.display_ids)

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
                "observability": {
                    "log_root": str(root / "stream-logs"),
                    "monitor_interval_seconds": 1,
                    "metrics_stale_seconds": 12,
                },
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
                fishing_disabled = client.patch(
                    "/v1/streams/abc123/fishing-risk",
                    headers={"X-API-Key": "test-api-key-1234"},
                    json={"enabled": False},
                )
                client.app.state.stream_logs.append(
                    "abc123",
                    level="WARNING",
                    event="playback.health",
                    message="视频出现卡顿",
                    details={"health": "stalled"},
                )
                stream_logs = client.get(
                    "/v1/streams/abc123/logs",
                    headers={"X-API-Key": "test-api-key-1234"},
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
        self.assertEqual(fishing_disabled.status_code, 200)
        self.assertFalse(fishing_disabled.json()["fishing_risk"]["enabled"])
        self.assertEqual(stream_logs.status_code, 200)
        self.assertEqual(stream_logs.json()[0]["details"]["health"], "stalled")
        self.assertEqual(publish_auth.status_code, 200)
        self.assertEqual(rejected_read.status_code, 401)
        self.assertEqual(event_list.status_code, 200)
        self.assertEqual(event_list.json()[0]["event_type"], "zone_dwell")
        self.assertEqual(confirmed.json()["status"], "confirmed")
        self.assertEqual(event_snapshot.content, b"jpeg")
