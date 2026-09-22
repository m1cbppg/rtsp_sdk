"""API-contract tests for ``mode=hybrid_v33``.

The dual-recall spec requires the API to expose the two channels' tunables while
deliberately *not* offering any switch that could turn either channel into a
one-way gate, and to keep the legacy modes' behaviour unchanged.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from rtsp_annotator.api import StreamCreateRequest

ZONE = {
    "region_id": "walkway",
    "name": "人行道",
    "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
}


def request(**ground_litter):
    payload = {
        "input_url": "rtsp://camera/walkway",
        "classes": [0],
        "ground_litter": dict(ground_litter),
    }
    return StreamCreateRequest.model_validate(payload)


def hybrid_request(**overrides):
    values = dict(
        enabled=True,
        mode="hybrid_v33",
        profile_id="camera_01_v32_afternoon_1080p",
        model="litter/turhancan_yolov8m_seg_trash.pt",
        zones=[ZONE],
    )
    values.update(overrides)
    return request(**values)


class HybridModeContractTests(unittest.TestCase):
    def test_hybrid_mode_is_accepted(self):
        options = hybrid_request().to_spec().ground_litter
        self.assertEqual(options.mode, "hybrid_v33")

    def test_hybrid_requires_both_model_and_profile(self):
        with self.assertRaises(ValidationError):
            hybrid_request(profile_id=None)
        with self.assertRaises(ValidationError):
            hybrid_request(model="")

    def test_hybrid_defaults_are_independent_per_channel(self):
        options = hybrid_request().to_spec().ground_litter
        self.assertEqual(options.semantic_confirm_hits, 2)
        self.assertEqual(options.semantic_hit_window, 3)
        self.assertEqual(options.prior_confirm_hits, 4)
        self.assertEqual(options.prior_hit_window, 6)
        self.assertEqual(options.fused_confirm_hits, 2)
        self.assertEqual(options.fused_hit_window, 4)
        # The two channels must not share one threshold or one cadence.
        self.assertNotEqual(
            options.semantic_confirm_hits, options.prior_confirm_hits
        )
        self.assertNotEqual(
            options.semantic_confirm_span_seconds,
            options.prior_confirm_span_seconds,
        )

    def test_hybrid_tunables_round_trip(self):
        options = hybrid_request(
            semantic_scan_interval_seconds=6.0,
            semantic_confirm_hits=3,
            semantic_hit_window=5,
            semantic_confirm_span_seconds=8.0,
            semantic_clear_seconds=12.0,
            semantic_clear_min_misses=3,
            prior_confirm_hits=5,
            prior_hit_window=8,
            prior_confirm_span_seconds=10.0,
            prior_suspend_expire_seconds=60.0,
            fused_confirm_hits=3,
            fused_hit_window=6,
            fused_confirm_span_seconds=4.0,
            prior_crop_maximum=2,
            prior_crop_expand_ratio=3.0,
            prior_crop_maximum_source_px=320,
            prior_crop_imgsz=960,
        ).to_spec().ground_litter
        self.assertEqual(options.semantic_scan_interval_seconds, 6.0)
        self.assertEqual(options.semantic_confirm_hits, 3)
        self.assertEqual(options.semantic_hit_window, 5)
        self.assertEqual(options.semantic_clear_seconds, 12.0)
        self.assertEqual(options.semantic_clear_min_misses, 3)
        self.assertEqual(options.prior_confirm_hits, 5)
        self.assertEqual(options.prior_hit_window, 8)
        self.assertEqual(options.prior_suspend_expire_seconds, 60.0)
        self.assertEqual(options.fused_confirm_hits, 3)
        self.assertEqual(options.fused_hit_window, 6)
        self.assertEqual(options.prior_crop_maximum, 2)
        self.assertEqual(options.prior_crop_expand_ratio, 3.0)
        self.assertEqual(options.prior_crop_maximum_source_px, 320)
        self.assertEqual(options.prior_crop_imgsz, 960)
        # Round-tripping through the options payload must be lossless.
        self.assertEqual(
            options.to_payload(),
            type(options).from_payload(options.to_payload()).to_payload(),
        )

    def test_hits_must_fit_their_own_window(self):
        for override in (
            {"semantic_confirm_hits": 5, "semantic_hit_window": 3},
            {"prior_confirm_hits": 9, "prior_hit_window": 6},
            {"fused_confirm_hits": 9, "fused_hit_window": 4},
        ):
            with self.assertRaises(ValidationError, msg=override):
                hybrid_request(**override)

    def test_crop_and_interval_bounds(self):
        for override in (
            {"prior_crop_maximum": 9},
            {"prior_crop_expand_ratio": 1.0},
            {"prior_crop_maximum_source_px": 100},
            {"prior_crop_imgsz": 128},
            {"semantic_scan_interval_seconds": 0.1},
            {"prior_suspend_expire_seconds": 0},
            {"semantic_confirm_span_seconds": 0},
        ):
            with self.assertRaises(ValidationError, msg=override):
                hybrid_request(**override)

    def test_no_cross_channel_gate_switch_exists(self):
        """No field may require the other channel before showing a box."""
        with self.assertRaises(ValidationError):
            hybrid_request(require_semantic_for_prior=True)
        with self.assertRaises(ValidationError):
            hybrid_request(require_prior_for_semantic=True)
        with self.assertRaises(ValidationError):
            hybrid_request(prior_only_requires_semantic=True)
        with self.assertRaises(ValidationError):
            hybrid_request(semantic_only_requires_prior=True)


class LegacyModeRegressionTests(unittest.TestCase):
    def test_yolo_mode_keeps_its_defaults(self):
        options = request(enabled=True, zones=[ZONE]).to_spec().ground_litter
        self.assertEqual(options.mode, "yolo")
        self.assertEqual(options.analysis_fps, 1.0)
        self.assertEqual(options.confidence, 0.20)

    def test_clean_reference_v32_still_only_requires_a_profile(self):
        options = request(
            enabled=True, mode="clean_reference_v32",
            profile_id="camera_01_v32_afternoon_1080p", zones=[ZONE],
        ).to_spec().ground_litter
        self.assertEqual(options.mode, "clean_reference_v32")
        # The hybrid tunables exist but must not change the legacy defaults.
        self.assertEqual(options.prior_crop_maximum, 4)

    def test_legacy_request_without_hybrid_fields_still_parses(self):
        legacy = {
            "input_url": "rtsp://camera/walkway",
            "classes": [0],
            "ground_litter": {
                "enabled": True,
                "mode": "clean_reference_v32",
                "profile_id": "camera_01_v32_afternoon_1080p",
                "analysis_fps": 0.5,
                "zones": [ZONE],
            },
        }
        options = StreamCreateRequest.model_validate(
            legacy
        ).to_spec().ground_litter
        self.assertEqual(options.mode, "clean_reference_v32")
        self.assertEqual(options.semantic_scan_interval_seconds, 4.0)

    def test_disabled_mode_default_is_unchanged(self):
        options = StreamCreateRequest.model_validate(
            {"input_url": "rtsp://camera/walkway", "classes": [0]}
        ).to_spec().ground_litter
        self.assertFalse(options.enabled)
        self.assertEqual(options.mode, "yolo")


# The dual-recall tunables that a hybrid_v33 stream must be able to set. Kept as
# an explicit list so a renamed or deleted field breaks this test instead of
# silently making the shipped example a partial configuration.
HYBRID_TUNABLES = (
    "semantic_scan_interval_seconds",
    "semantic_confirm_hits",
    "semantic_hit_window",
    "semantic_confirm_span_seconds",
    "semantic_clear_seconds",
    "semantic_clear_min_misses",
    "prior_confirm_hits",
    "prior_hit_window",
    "prior_confirm_span_seconds",
    "prior_suspend_expire_seconds",
    "fused_confirm_hits",
    "fused_hit_window",
    "fused_confirm_span_seconds",
    "prior_crop_maximum",
    "prior_crop_expand_ratio",
    "prior_crop_maximum_source_px",
    "prior_crop_imgsz",
)

EXAMPLE_REQUEST = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "ground_litter_v33_hybrid_stream_request.example.json"
)


class HybridExampleRequestTests(unittest.TestCase):
    """The shipped Postman/example body must stay a valid hybrid_v33 request."""

    def setUp(self):
        self.payload = json.loads(EXAMPLE_REQUEST.read_text(encoding="utf-8"))

    def test_example_validates_against_the_live_request_model(self):
        request = StreamCreateRequest.model_validate(self.payload)
        options = request.to_spec().ground_litter
        self.assertEqual(options.mode, "hybrid_v33")
        self.assertTrue(options.enabled)
        self.assertEqual(options.profile_id, "camera_01_v32_afternoon_1080p")
        self.assertEqual(options.model, "litter/turhancan_yolov8m_seg_trash.pt")

    def test_example_sets_every_hybrid_tunable(self):
        block = self.payload["ground_litter"]
        missing = [name for name in HYBRID_TUNABLES if name not in block]
        self.assertEqual(missing, [], "示例请求缺少双通道字段")

    def test_example_tunables_survive_the_spec_conversion(self):
        options = StreamCreateRequest.model_validate(
            self.payload
        ).to_spec().ground_litter
        block = self.payload["ground_litter"]
        for name in HYBRID_TUNABLES:
            self.assertEqual(
                getattr(options, name), block[name],
                msg=f"{name} 在请求→Options 转换中丢失",
            )

    def test_example_zone_gate_matches_the_v32_pixel_equivalence(self):
        # 1021 mux is 1920x1080 while the camera is 2560x1440, so the reviewed
        # 8px/64px^2 native gate becomes 6px/36px^2; the example must not ship a
        # gate that would filter the confirmed 8x8px litter.
        zone = self.payload["ground_litter"]["zones"][0]
        self.assertEqual(zone["minimum_short_side_px"], 6)
        self.assertEqual(zone["minimum_box_area_px"], 36)

    def test_example_has_no_cross_channel_gate_switch(self):
        block = self.payload["ground_litter"]
        forbidden = (
            "require_semantic_for_prior",
            "prior_requires_semantic",
            "semantic_gate",
            "prior_gate",
        )
        self.assertEqual([name for name in forbidden if name in block], [])


if __name__ == "__main__":
    unittest.main()
