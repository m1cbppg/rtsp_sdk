from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from rtsp_annotator.api import StreamCreateRequest, create_app
from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.events import EventRecord
from rtsp_annotator.ground_litter_detection import GroundLitterDetectionOptions
from rtsp_annotator.ptz_verification import PtzVerificationOptions
from rtsp_annotator.vessel_detection import VesselDetection


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
        result = self.create(object())
        result["metrics"] = {
            "publish_fps": 25.0,
            "ground_litter_hybrid": {
                "branch_state": "ok",
                "semantic_model_runs_full": 3,
            },
        }
        return result

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

    def return_ptz_home(self, stream_id: str) -> dict[str, object]:
        if stream_id != "abc123":
            raise KeyError(stream_id)
        return {
            "stream_id": stream_id,
            "request_id": "request-123",
            "action": "return_home",
            "status": "accepted",
        }

    def shutdown(self) -> None:
        pass


class ApiTests(unittest.TestCase):
    def test_vessel_detection_does_not_enable_camera_control_by_default(self) -> None:
        request = StreamCreateRequest.model_validate(
            {
                "input_url": "rtsp://camera/harbor",
                "vessel_detection": {"enabled": True},
            }
        )

        spec = request.to_spec()
        self.assertTrue(spec.vessel_detection.enabled)
        self.assertFalse(spec.ptz_verification.enabled)
        self.assertEqual(spec.ptz_verification.camera_id, "")
        self.assertFalse(spec.ptz_verification.continuous_tracking)
        self.assertFalse(spec.ptz_verification.display_operation_log)
        self.assertEqual(spec.ptz_verification.adaptive_target_width_ratio, 0.33)
        self.assertEqual(spec.ptz_verification.adaptive_target_height_ratio, 0.33)
        self.assertEqual(spec.ptz_verification.tracking_initial_extra_zoom_step, 0)
        self.assertEqual(
            spec.ptz_verification.primary_target_minimum_observations,
            1,
        )

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
                    "small_target_proposals": True,
                    "display_proposals": False,
                    "proposal_roi": [
                        [0, 0.4], [1, 0.4], [1, 0.75], [0, 0.75]
                    ],
                    "proposal_appearance_threshold": 20,
                    "proposal_appearance_blur_pixels": 41,
                    "proposal_border_margin": 0.02,
                    "proposal_minimum_fill_ratio": 0.3,
                    "proposal_minimum_motion_ratio": 0.1,
                    "proposal_maximum_candidates": 4,
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
        self.assertTrue(options.small_target_proposals)
        self.assertFalse(options.display_proposals)
        self.assertEqual(len(options.proposal_roi or ()), 4)
        self.assertEqual(options.proposal_threshold, 60)
        self.assertTrue(options.proposal_appearance_enabled)
        self.assertEqual(options.proposal_appearance_threshold, 20)
        self.assertEqual(options.proposal_appearance_blur_pixels, 41)
        self.assertEqual(options.proposal_border_margin, 0.02)
        self.assertEqual(options.proposal_minimum_fill_ratio, 0.3)
        self.assertEqual(options.proposal_minimum_motion_ratio, 0.1)
        self.assertEqual(options.proposal_maximum_candidates, 4)

    def test_ptz_verification_maps_to_stream_spec(self) -> None:
        request = StreamCreateRequest.model_validate(
            {
                "input_url": "rtsp://camera/harbor",
                "vessel_detection": {"enabled": True},
                "ptz_verification": {
                    "enabled": True,
                    "camera_id": "camera-01",
                    "camera_control_url": "http://camera-control:8080",
                    "trace_logging_enabled": True,
                    "display_operation_log": True,
                    "vessel_number_recognition_enabled": True,
                    "vessel_number_fallback": "10032",
                    "zoom_strategy": "adaptive",
                    "adaptive_target_width_ratio": 0.30,
                    "adaptive_max_step": 5,
                    "adaptive_max_rounds": 4,
                    "adaptive_max_total_zoom_delta": 14,
                    "confirmed_target_fallback_zoom_rounds": 2,
                    "confirmed_target_fallback_zoom_step": 4,
                    "home_stable_frames": 3,
                    "evidence_capture_attempts": 3,
                    "evidence_minimum_sharpness": 18.0,
                    "evidence_target_scale_ratio": 0.8,
                    "reacquire_strict_center_radius": 0.20,
                    "reacquire_center_radius": 0.40,
                    "reacquire_cluster_radius": 0.16,
                    "proposal_minimum_interval_seconds": 45,
                    "proposal_maximum_verifications_per_hour": 6,
                    "confirmed_cooldown_seconds": 900,
                    "primary_target_minimum_observations": 1,
                    "continuous_tracking": True,
                    "tracking_center_deadband": 0.08,
                    "tracking_command_interval_seconds": 0.4,
                    "tracking_settle_seconds": 0.1,
                    "tracking_recovery_enabled": True,
                    "tracking_recovery_interval_seconds": 1.5,
                    "tracking_recovery_zoom_out_step": 2,
                    "tracking_recovery_max_attempts": 4,
                    "tracking_lost_timeout_seconds": 5,
                    "tracking_max_duration_seconds": 0,
                    "tracking_zoom_hysteresis_ratio": 0.15,
                    "tracking_zoom_step": 2,
                    "tracking_initial_extra_zoom_step": 1,
                },
            }
        )

        options = request.to_spec().ptz_verification
        self.assertTrue(options.enabled)
        self.assertEqual(options.camera_id, "camera-01")
        self.assertTrue(options.trace_logging_enabled)
        self.assertTrue(options.display_operation_log)
        self.assertTrue(options.vessel_number_recognition_enabled)
        self.assertEqual(options.vessel_number_fallback, "10032")
        self.assertEqual(options.zoom_strategy, "adaptive")
        self.assertEqual(options.adaptive_target_width_ratio, 0.30)
        self.assertEqual(options.adaptive_max_step, 5)
        self.assertEqual(options.adaptive_max_rounds, 4)
        self.assertEqual(options.adaptive_max_total_zoom_delta, 14)
        self.assertEqual(options.confirmed_target_fallback_zoom_rounds, 2)
        self.assertEqual(options.confirmed_target_fallback_zoom_step, 4)
        self.assertEqual(options.home_stable_frames, 3)
        self.assertEqual(options.evidence_capture_attempts, 3)
        self.assertEqual(options.evidence_minimum_sharpness, 18.0)
        self.assertEqual(options.evidence_target_scale_ratio, 0.8)
        self.assertEqual(options.reacquire_strict_center_radius, 0.20)
        self.assertEqual(options.reacquire_center_radius, 0.40)
        self.assertEqual(options.reacquire_cluster_radius, 0.16)
        self.assertEqual(options.proposal_minimum_interval_seconds, 45)
        self.assertEqual(options.proposal_maximum_verifications_per_hour, 6)
        self.assertEqual(options.confirmed_cooldown_seconds, 900)
        self.assertEqual(options.primary_target_minimum_observations, 1)
        self.assertTrue(options.continuous_tracking)
        self.assertEqual(options.tracking_center_deadband, 0.08)
        self.assertEqual(options.tracking_command_interval_seconds, 0.4)
        self.assertEqual(options.tracking_settle_seconds, 0.1)
        self.assertTrue(options.tracking_recovery_enabled)
        self.assertEqual(options.tracking_recovery_interval_seconds, 1.5)
        self.assertEqual(options.tracking_recovery_zoom_out_step, 2)
        self.assertEqual(options.tracking_recovery_max_attempts, 4)
        self.assertEqual(options.tracking_lost_timeout_seconds, 5)
        self.assertEqual(options.tracking_max_duration_seconds, 0)
        self.assertEqual(options.tracking_zoom_hysteresis_ratio, 0.15)
        self.assertEqual(options.tracking_zoom_step, 2)
        self.assertEqual(options.tracking_initial_extra_zoom_step, 1)

    def test_ptz_verification_requires_vessel_detection(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须启用vessel_detection"):
            StreamCreateRequest.model_validate(
                {
                    "input_url": "rtsp://camera/harbor",
                    "ptz_verification": {
                        "enabled": True,
                        "camera_id": "camera-01",
                    },
                }
            )

    def test_display_detections_defaults_on_and_can_be_disabled(self) -> None:
        default = StreamCreateRequest.model_validate(
            {"input_url": "rtsp://camera/walkway"}
        ).to_spec()
        hidden = StreamCreateRequest.model_validate(
            {
                "input_url": "rtsp://camera/walkway",
                "classes": [0],
                "display_detections": False,
            }
        ).to_spec()
        self.assertTrue(default.display_detections)
        self.assertFalse(hidden.display_detections)
        # The class filter is a separate concern and stays intact.
        self.assertEqual(hidden.classes, (0,))

    def test_ground_litter_request_maps_to_options(self) -> None:
        request = StreamCreateRequest.model_validate(
            {
                "input_url": "rtsp://camera/walkway",
                "classes": [0],
                "ground_litter": {
                    "enabled": True,
                    "confidence": 0.18,
                    "night_confidence": 0.25,
                    "analysis_fps": 1.5,
                    "inference_imgsz": 960,
                    "actor_model": "yolo26s.pt",
                    "local_actor_max_crops": 4,
                    "box_smoothing_alpha": 0.5,
                    "tile_size_px": 640,
                    "tile_overlap": 0.2,
                    "minimum_hits": 2,
                    "hit_window": 3,
                    "hold_seconds": 4,
                    "maximum_boxes": 6,
                    "display_class": True,
                    "zones": [
                        {
                            "region_id": "merchant_01",
                            "name": "门店01门前人行道",
                            "polygon": [
                                [0.32, 0.15],
                                [0.367, 0.15],
                                [0.35, 0.3],
                                [0.27, 0.3],
                            ],
                            "exclude_zones": [
                                [[0.3, 0.2], [0.34, 0.2], [0.34, 0.25]]
                            ],
                            "minimum_short_side_px": 8,
                            "minimum_box_area_px": 64,
                        }
                    ],
                    "overlay_exclude_zones": [
                        [[0.0, 0.035], [0.4, 0.035], [0.4, 0.105]]
                    ],
                },
            }
        )

        options = request.to_spec().ground_litter

        self.assertTrue(options.enabled)
        self.assertEqual(options.confidence, 0.18)
        self.assertEqual(options.confidence_for(True), 0.25)
        self.assertEqual(options.analysis_fps, 1.5)
        self.assertEqual(options.effective_imgsz, 960)
        self.assertEqual(options.local_actor_max_crops, 4)
        self.assertEqual(options.box_smoothing_alpha, 0.5)
        self.assertEqual(options.region_ids, ("merchant_01",))
        self.assertEqual(options.zones[0].minimum_short_side_px, 8)
        self.assertEqual(options.zones[0].minimum_box_area_px, 64)
        self.assertEqual(len(options.zones[0].exclude_zones), 1)
        self.assertEqual(len(options.overlay_exclude_zones), 1)
        self.assertEqual(options.hold_seconds, 4)
        self.assertEqual(options.maximum_boxes, 6)
        self.assertTrue(options.display_class)
        self.assertEqual(options.label, "疑似垃圾")
        self.assertEqual(options.model, "turhancan_yolov8m_seg_trash.pt")

    def test_ground_litter_defaults_stay_disabled(self) -> None:
        request = StreamCreateRequest.model_validate(
            {"input_url": "rtsp://camera/walkway"}
        )
        options = request.to_spec().ground_litter
        self.assertFalse(options.enabled)
        self.assertEqual(options.zones, ())

    def test_clean_reference_v32_request_maps_lifecycle_options(self) -> None:
        request = StreamCreateRequest.model_validate({
            "input_url": "rtsp://camera/walkway",
            "ground_litter": {
                "enabled": True,
                "mode": "clean_reference_v32",
                "profile_id": "camera_01_v32",
                "confirm_visible_seconds": 6,
                "clear_confirm_seconds": 7,
                "pending_expire_seconds": 18,
                "min_clean_valid_fraction": 0.85,
                "startup_suppress_seconds": 20,
                "normal_stability_samples": 4,
                "zones": [{
                    "region_id": "walkway",
                    "polygon": [[0.45, 0.23], [0.72, 0.23], [0.81, 1.0], [0.45, 1.0]],
                }],
            },
        })
        options = request.to_spec().ground_litter
        self.assertEqual(options.mode, "clean_reference_v32")
        self.assertEqual(options.profile_id, "camera_01_v32")
        self.assertEqual(options.confirm_visible_seconds, 6)
        self.assertEqual(options.clear_confirm_seconds, 7)
        self.assertEqual(options.pending_expire_seconds, 18)
        self.assertEqual(options.min_clean_valid_fraction, 0.85)
        self.assertEqual(options.startup_suppress_seconds, 20)
        self.assertEqual(options.normal_stability_samples, 4)
        self.assertEqual(
            GroundLitterDetectionOptions.from_payload(options.to_payload()),
            options,
        )

    def test_clean_reference_v32_requires_profile_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "profile_id"):
            StreamCreateRequest.model_validate({
                "input_url": "rtsp://camera/walkway",
                "ground_litter": {
                    "enabled": True,
                    "mode": "clean_reference_v32",
                    "zones": [{
                        "region_id": "walkway",
                        "polygon": [[0, 0], [1, 0], [1, 1]],
                    }],
                },
            })

    def test_ground_litter_requires_a_zone_when_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "至少需要一个地面区域"):
            StreamCreateRequest.model_validate(
                {
                    "input_url": "rtsp://camera/walkway",
                    "ground_litter": {"enabled": True},
                }
            )

    def test_ground_litter_rejects_escaping_model_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "越界"):
            StreamCreateRequest.model_validate(
                {
                    "input_url": "rtsp://camera/walkway",
                    "ground_litter": {
                        "enabled": True,
                        "model": "../../etc/passwd.pt",
                        "zones": [
                            {
                                "region_id": "z1",
                                "polygon": [[0, 0], [1, 0], [1, 1]],
                            }
                        ],
                    },
                }
            )

    def test_ground_litter_conflicts_with_ptz_verification(self) -> None:
        with self.assertRaisesRegex(ValueError, "不能与固定视角功能同时启用"):
            StreamCreateRequest.model_validate(
                {
                    "input_url": "rtsp://camera/walkway",
                    "vessel_detection": {"enabled": True},
                    "ground_litter": {
                        "enabled": True,
                        "zones": [
                            {
                                "region_id": "z1",
                                "polygon": [[0, 0], [1, 0], [1, 1]],
                            }
                        ],
                    },
                    "ptz_verification": {
                        "enabled": True,
                        "camera_id": "camera-01",
                    },
                }
            )

    def test_ptz_verification_rejects_invalid_adaptive_step_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "min<=max"):
            StreamCreateRequest.model_validate(
                {
                    "input_url": "rtsp://camera/harbor",
                    "vessel_detection": {"enabled": True},
                    "ptz_verification": {
                        "enabled": True,
                        "camera_id": "camera-01",
                        "adaptive_min_step": 8,
                        "adaptive_max_step": 4,
                    },
                }
            )

    def test_ptz_verification_rejects_inverted_reacquire_radii(self) -> None:
        with self.assertRaisesRegex(ValueError, "strict<=maximum"):
            StreamCreateRequest.model_validate(
                {
                    "input_url": "rtsp://camera/harbor",
                    "vessel_detection": {"enabled": True},
                    "ptz_verification": {
                        "enabled": True,
                        "camera_id": "camera-01",
                        "reacquire_strict_center_radius": 0.5,
                        "reacquire_center_radius": 0.4,
                    },
                }
            )

    def test_ptz_verification_rejects_fixed_view_features(self) -> None:
        features = {
            "license_plate": {"enabled": True},
            "event_detection": {
                "enabled": True,
                "rois": [
                    {
                        "id": "waterfront",
                        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    }
                ],
            },
            "gas_cylinder": {"enabled": True},
        }
        for feature, payload in features.items():
            with self.subTest(feature=feature):
                with self.assertRaisesRegex(
                    ValueError,
                    "不能与固定视角功能同时启用",
                ):
                    StreamCreateRequest.model_validate(
                        {
                            "input_url": "rtsp://camera/harbor",
                            "vessel_detection": {"enabled": True},
                            "ptz_verification": {
                                "enabled": True,
                                "camera_id": "camera-01",
                            },
                            feature: payload,
                        }
                    )

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
                stream_detail = client.get(
                    "/v1/streams/abc123",
                    headers={"X-API-Key": "test-api-key-1234"},
                )
                fishing_disabled = client.patch(
                    "/v1/streams/abc123/fishing-risk",
                    headers={"X-API-Key": "test-api-key-1234"},
                    json={"enabled": False},
                )
                return_home = client.post(
                    "/v1/streams/abc123/ptz/return-home",
                    headers={"X-API-Key": "test-api-key-1234"},
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
                # HLS/WebRTC 读取：MediaMTX 可能带前导斜杠路径，或用 "play" 动作，都应放行
                hls_play = client.post(
                    "/internal/mediamtx/auth",
                    json={
                        "user": "viewer",
                        "password": "read-password",
                        "action": "play",
                        "path": "detected/abc123",
                    },
                )
                hls_slash = client.post(
                    "/internal/mediamtx/auth",
                    json={
                        "user": "viewer",
                        "password": "read-password",
                        "action": "read",
                        "path": "/detected/abc123",
                    },
                )
                hls_slash_playlist = client.post(
                    "/internal/mediamtx/auth",
                    json={
                        "user": "viewer",
                        "password": "read-password",
                        "action": "play",
                        "path": "/detected/abc123/index.m3u8",
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
                verification_options = PtzVerificationOptions(
                    enabled=True,
                    camera_id="camera-01",
                )
                now = time.time()
                target = client.app.state.ptz_verifications.observe(
                    stream_id="abc123",
                    camera_id="camera-01",
                    detection=VesselDetection(
                        object_id=7,
                        rectangle=NormalizedRect(0.6, 0.5, 0.03, 0.02),
                        confidence=0.2,
                        class_id=8,
                        hits=4,
                    ),
                    now=now,
                    options=verification_options,
                )
                job = client.app.state.ptz_verifications.claim(
                    stream_id="abc123",
                    camera_id="camera-01",
                    target=target,
                    now=now,
                )
                assert job is not None
                client.app.state.ptz_verifications.mark_running(job.job_id, now)
                image_id = client.app.state.ptz_verifications.store_evidence(
                    job_id=job.job_id,
                    content=b"boat-jpeg",
                    mime_type="image/jpeg",
                    captured_at=now,
                )
                client.app.state.ptz_verifications.finish(
                    job_id=job.job_id,
                    result="boat_confirmed",
                    error=None,
                    home_returned=True,
                    now=now,
                    options=verification_options,
                )
                verification_list = client.get(
                    "/v1/vessel-verifications?stream_id=abc123",
                    headers={"X-API-Key": "test-api-key-1234"},
                )
                verification_image = client.get(
                    f"/v1/vessel-verifications/{job.job_id}/images/{image_id}",
                    headers={"X-API-Key": "test-api-key-1234"},
                )

        self.assertEqual(health.status_code, 200)
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(models.json(), {"models": ["yolo26s.pt"]})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json()["stream_id"], "abc123")
        self.assertNotIn("camera-user", created.text)
        self.assertEqual(stream_detail.status_code, 200)
        self.assertEqual(
            stream_detail.json()["metrics"]["ground_litter_hybrid"]["branch_state"],
            "ok",
        )
        self.assertEqual(fishing_disabled.status_code, 200)
        self.assertFalse(fishing_disabled.json()["fishing_risk"]["enabled"])
        self.assertEqual(return_home.status_code, 202)
        self.assertEqual(return_home.json()["action"], "return_home")
        self.assertEqual(return_home.json()["request_id"], "request-123")
        self.assertEqual(stream_logs.status_code, 200)
        self.assertEqual(stream_logs.json()[0]["details"]["health"], "stalled")
        self.assertEqual(publish_auth.status_code, 200)
        self.assertEqual(rejected_read.status_code, 401)
        self.assertEqual(hls_play.status_code, 200)
        self.assertEqual(hls_slash.status_code, 200)
        self.assertEqual(hls_slash_playlist.status_code, 200)
        self.assertEqual(event_list.status_code, 200)
        self.assertEqual(event_list.json()[0]["event_type"], "zone_dwell")
        self.assertEqual(confirmed.json()["status"], "confirmed")
        self.assertEqual(event_snapshot.content, b"jpeg")
        self.assertEqual(verification_list.status_code, 200)
        self.assertEqual(
            verification_list.json()[0]["result"],
            "boat_confirmed",
        )
        self.assertEqual(verification_image.content, b"boat-jpeg")
