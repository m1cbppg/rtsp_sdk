"""Unit tests for the Ground Litter Rapid Eval v1 contract.

These tests are deliberately free of cv2/torch so they run in the plain repo venv.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from rtsp_annotator.ground_litter_rapid import (  # noqa: E402
    CAMERAS,
    EVAL_SPLIT,
    SEED,
    TRAIN_SPLIT,
    RapidError,
    assert_development_asset,
    assert_trainable,
    assert_writable_root,
    bonus_frame_id,
    box_contains_point,
    build_frame_manifest,
    build_split,
    canonical_sha256,
    decide_verdict,
    frame_id,
    frame_index_for,
    match_points,
    point_inside_roi,
    select_threshold,
    selection_hash,
    size_bucket,
    summarise_metrics,
    tile_starts,
    verify_split,
)

ROI = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def synthetic_files() -> list[dict]:
    rows = []
    for camera in CAMERAS:
        for index in range(13):
            file_id = f"{camera}-ps{index:02d}"
            rows.append({
                "camera_id": camera,
                "file_id": file_id,
                "file_name": f"{file_id}.ps",
                "ps_path": f"/home/sf01/development/{camera}/raw/{file_id}.ps",
                "sha256": f"{index:064d}",
                "canvas_size": [2560, 1440],
                "roi": ROI,
                "roi_geometry_version": "final-roi-20260922-user-reviewed",
                "record_start": "2026-09-22 15:55:33",
                "record_end": "2026-09-22 16:00:37",
                "duration_seconds": 304.0,
            })
    return rows


PRIOR = ["01021-ps00", "01022-ps01", "01022-ps02", "01022-ps03", "01027-ps00",
         "01030-ps00"]


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.files = synthetic_files()
        self.split = build_split(self.files, PRIOR)

    def test_counts_and_per_camera(self):
        self.assertEqual(self.split["counts"][TRAIN_SPLIT], 40)
        self.assertEqual(self.split["counts"][EVAL_SPLIT], 12)
        for camera in CAMERAS:
            per = self.split["per_camera"][camera]
            self.assertEqual(per["rapid_train"], 10)
            self.assertEqual(per["rapid_eval"], 3)

    def test_ps_never_spans_both_splits(self):
        seen: dict[str, str] = {}
        for row in self.split["rows"]:
            if row["file_id"] in seen:
                self.assertEqual(seen[row["file_id"]], row["split"])
            seen[row["file_id"]] = row["split"]
        self.assertEqual(len(seen), 52)

    def test_prior_inference_forced_into_train(self):
        for row in self.split["rows"]:
            if row["file_id"] in PRIOR:
                self.assertEqual(row["split"], TRAIN_SPLIT)
                self.assertTrue(row["excluded_from_holdout_due_prior_inference"])
                self.assertEqual(row["selection_reason"],
                                 "prior_inference_forced_rapid_train")
            else:
                self.assertFalse(row["excluded_from_holdout_due_prior_inference"])

    def test_deterministic_across_runs(self):
        again = build_split(self.files, PRIOR)
        self.assertEqual(again["split_sha256"], self.split["split_sha256"])
        self.assertEqual([r["file_id"] for r in again["rows"]],
                         [r["file_id"] for r in self.split["rows"]])

    def test_selection_hash_matches_split(self):
        for row in self.split["rows"]:
            self.assertEqual(row["selection_hash"],
                             selection_hash(SEED, row["split"], row["file_id"]))

    def test_verify_split_accepts_frozen_split(self):
        verify_split(self.split)

    def test_verify_split_rejects_tampering(self):
        tampered = json.loads(json.dumps(self.split))
        tampered["rows"][0]["split"] = (EVAL_SPLIT if tampered["rows"][0]["split"] == TRAIN_SPLIT
                                        else TRAIN_SPLIT)
        with self.assertRaises(RapidError):
            verify_split(tampered)

    def test_verify_split_rejects_hash_drift(self):
        tampered = json.loads(json.dumps(self.split))
        tampered["rows"][0]["selection_hash"] = "0" * 64
        with self.assertRaises(RapidError):
            verify_split(tampered)

    def test_prior_inference_must_exist(self):
        with self.assertRaises(RapidError):
            build_split(self.files, PRIOR + ["01021-does-not-exist"])

    def test_wrong_ps_count_is_rejected(self):
        with self.assertRaises(RapidError):
            build_split(self.files[:-1], PRIOR)


class FrameManifestTests(unittest.TestCase):
    def setUp(self):
        self.split = build_split(synthetic_files(), PRIOR)

    def test_fixed_grid_counts(self):
        manifest = build_frame_manifest(self.split, [])
        self.assertEqual(manifest["counts"]["fixed_train"], 200)
        self.assertEqual(manifest["counts"]["fixed_eval"], 60)
        self.assertEqual(len(manifest["frames"]), 260)
        for row in manifest["frames"]:
            self.assertIn(row["offset_seconds"], (30, 90, 150, 210, 270))
            if row["kind"] == "fixed":
                self.assertEqual(row["nominal_frame_index"],
                                 frame_index_for(row["offset_seconds"]))

    def test_bonus_frame_only_on_train_and_deduplicated(self):
        bonus = [{"file_id": "01021-ps00", "frame_index": 187, "decoded_timestamp": 7.476,
                  "truth_ids": ["t-0001"]},
                 {"file_id": "01021-ps00", "frame_index": 187, "decoded_timestamp": 7.476,
                  "truth_ids": ["t-0002"]}]
        manifest = build_frame_manifest(self.split, bonus)
        bonus_rows = [r for r in manifest["frames"] if r["kind"] == "bonus_train"]
        self.assertEqual(len(bonus_rows), 1)
        self.assertEqual(manifest["counts"]["bonus_train"], 1)
        self.assertEqual(manifest["counts"]["fixed_train"], 200)
        self.assertEqual(bonus_rows[0]["split"], TRAIN_SPLIT)
        self.assertEqual(bonus_rows[0]["frame_id"],
                         bonus_frame_id("01021", "01021-ps00", 187))

    def test_bonus_frame_on_eval_ps_is_refused(self):
        eval_row = next(r for r in self.split["rows"] if r["split"] == EVAL_SPLIT)
        with self.assertRaises(RapidError):
            build_frame_manifest(self.split, [{"file_id": eval_row["file_id"],
                                               "frame_index": 10,
                                               "decoded_timestamp": 0.4}])

    def test_frame_id_is_stable(self):
        self.assertEqual(frame_id("01021", "01021-ps00", 30), "01021_01021-ps00_t030")


class GuardTests(unittest.TestCase):
    def test_train_export_guard_refuses_eval(self):
        with self.assertRaises(RapidError):
            assert_trainable(EVAL_SPLIT, sample_id="f1")

    def test_train_export_guard_allows_train(self):
        assert_trainable(TRAIN_SPLIT, sample_id="f1")

    def test_sealed_path_is_refused(self):
        with self.assertRaises(RapidError):
            assert_development_asset("/home/sf01/feasibility/sealed_test/a.ps")
        with self.assertRaises(RapidError):
            assert_development_asset("/home/sf01/step2c1-blind-truth/artifact/truth.jsonl")

    def test_official_roots_are_not_writable(self):
        with self.assertRaises(RapidError):
            assert_writable_root("/home/sf01/step2c1-blind-truth/artifact")
        with self.assertRaises(RapidError):
            assert_writable_root("/home/sf01/step2b-20260923/out")

    def test_rapid_root_is_writable(self):
        assert_writable_root("/home/sf01/ground-litter-rapid-v1/artifact")


class GeometryTests(unittest.TestCase):
    def test_tile_starts_cover_full_axis(self):
        starts = tile_starts(2560)
        self.assertEqual(starts, [0, 512, 1024, 1536, 1920])
        self.assertEqual(starts[-1] + 640, 2560)
        self.assertEqual(tile_starts(1440), [0, 512, 800])

    def test_point_inside_roi(self):
        self.assertTrue(point_inside_roi(0.5 * 2560, 0.5 * 1440, ROI))
        self.assertFalse(point_inside_roi(10, 10, ROI))

    def test_box_contains_point(self):
        self.assertTrue(box_contains_point([0, 0, 10, 10], 5, 5))
        self.assertFalse(box_contains_point([0, 0, 10, 10], 11, 5))

    def test_size_buckets(self):
        self.assertEqual(size_bucket(9), "<10")
        self.assertEqual(size_bucket(10), "10-19")
        self.assertEqual(size_bucket(19.9), "10-19")
        self.assertEqual(size_bucket(20), "20-39")
        self.assertEqual(size_bucket(200), "80+")


class MatchingTests(unittest.TestCase):
    def points(self):
        return [
            {"truth_id": "t-1", "truth_class": "REQUIRED_LITTER", "source_xy": [100, 100]},
            {"truth_id": "t-2", "truth_class": "REQUIRED_LITTER", "source_xy": [300, 300]},
            {"truth_id": "t-3", "truth_class": "IGNORE_SMALL", "source_xy": [500, 500]},
        ]

    def predictions(self):
        return [
            {"prediction_id": "p1", "xyxy": [90, 90, 120, 120], "confidence": 0.5},
            {"prediction_id": "p2", "xyxy": [290, 290, 320, 320], "confidence": 0.4},
            {"prediction_id": "p3", "xyxy": [490, 490, 520, 520], "confidence": 0.3},
            {"prediction_id": "p4", "xyxy": [700, 700, 760, 760], "confidence": 0.2},
        ]

    def test_hits_miss_fp_and_x_suppression(self):
        result = match_points(self.points(), self.predictions(),
                              {"p1": "Y", "p2": "Y", "p3": "X", "p4": "N"})
        self.assertEqual(result["required_total"], 2)
        self.assertEqual(result["required_hit"], 2)
        self.assertEqual(result["required_miss"], 0)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["xfp_suppressed"], 1)

    def test_one_to_one_matching(self):
        points = [{"truth_id": "t-1", "truth_class": "REQUIRED_LITTER", "source_xy": [100, 100]},
                  {"truth_id": "t-2", "truth_class": "REQUIRED_LITTER", "source_xy": [105, 105]}]
        predictions = [{"prediction_id": "p1", "xyxy": [90, 90, 120, 120],
                        "confidence": 0.5}]
        result = match_points(points, predictions, {"p1": "Y"})
        self.assertEqual(result["required_hit"], 1)
        self.assertEqual(result["required_miss"], 1)

    def test_unjudged_prediction_is_not_fp(self):
        result = match_points(self.points(), self.predictions(), {"p1": "Y"})
        self.assertEqual(result["fp"], 0)

    def test_m_verdict_is_not_fp(self):
        result = match_points(self.points(), self.predictions(), {"p1": "M"})
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["required_hit"], 0)
        self.assertEqual(result["m_prediction_ids"], ["p1"])

    def test_iou_zero_confidence_does_not_create_hit(self):
        predictions = [{"prediction_id": "p9", "xyxy": [95, 95, 130, 130],
                        "confidence": 0.5}]
        result = match_points(self.points()[:1], predictions, {})
        self.assertEqual(result["required_hit"], 0)


class MetricsTests(unittest.TestCase):
    def test_summary_and_camera_breakdown(self):
        per_frame = [
            {"camera_id": "01021", "required_total": 2, "required_hit": 2, "required_miss": 0,
             "fp": 1, "prediction_count": 5},
            {"camera_id": "01021", "required_total": 0, "required_hit": 0, "required_miss": 0,
             "fp": 2, "prediction_count": 4},
            {"camera_id": "01022", "required_total": 1, "required_hit": 0, "required_miss": 1,
             "fp": 0, "prediction_count": 2},
        ]
        summary = summarise_metrics(per_frame)
        self.assertEqual(summary["frames"], 3)
        self.assertEqual(summary["required_points"], 3)
        self.assertEqual(summary["required_hit"], 2)
        self.assertAlmostEqual(summary["required_hit_rate"], 2 / 3)
        self.assertEqual(summary["fp"], 3)
        self.assertAlmostEqual(summary["fp_per_100_frames"], 100.0)
        self.assertEqual(summary["positive_frames"], 2)
        self.assertEqual(summary["positive_frame_hit"], 1)
        self.assertEqual(summary["per_camera"]["01021"]["fp"], 3)
        self.assertEqual(summary["per_camera"]["01022"]["required_hit_rate"], 0.0)

    def test_empty_summary_is_none_not_zero(self):
        summary = summarise_metrics([{"camera_id": "01021", "required_total": 0,
                                      "required_hit": 0, "required_miss": 0, "fp": 0,
                                      "prediction_count": 0}])
        self.assertIsNone(summary["required_hit_rate"])
        self.assertIsNone(summary["positive_frame_hit_rate"])


class ThresholdSelectionTests(unittest.TestCase):
    @staticmethod
    def metric(hit_rate, fp):
        return {"required_hit_rate": hit_rate, "fp_per_100_frames": fp}

    def test_prefers_highest_hit_rate(self):
        chosen = select_threshold({
            "0.01": self.metric(0.80, 5.0),
            "0.10": self.metric(0.90, 40.0),
            "0.30": self.metric(0.70, 1.0),
        })
        self.assertEqual(chosen["selected_threshold"], 0.10)

    def test_within_two_points_prefers_lower_fp(self):
        chosen = select_threshold({
            "0.05": self.metric(0.900, 60.0),
            "0.10": self.metric(0.895, 12.0),
            "0.20": self.metric(0.70, 1.0),
        })
        self.assertEqual(chosen["selected_threshold"], 0.10)

    def test_full_tie_prefers_higher_threshold(self):
        chosen = select_threshold({
            "0.10": self.metric(0.90, 10.0),
            "0.20": self.metric(0.90, 10.0),
        })
        self.assertEqual(chosen["selected_threshold"], 0.20)

    def test_requires_at_least_one_threshold(self):
        with self.assertRaises(RapidError):
            select_threshold({})


class VerdictTests(unittest.TestCase):
    def test_strong_improvement(self):
        verdict = decide_verdict({"required_hit_rate": 0.20, "fp_per_100_frames": 10.0},
                                 {"required_hit_rate": 0.35, "fp_per_100_frames": 14.0})
        self.assertEqual(verdict["verdict"], "RAPID_ITERATION_POSITIVE")
        self.assertAlmostEqual(verdict["hit_rate_delta_pp"], 15.0)

    def test_no_improvement_is_bottleneck(self):
        verdict = decide_verdict({"required_hit_rate": 0.20, "fp_per_100_frames": 10.0},
                                 {"required_hit_rate": 0.22, "fp_per_100_frames": 9.0})
        self.assertEqual(verdict["verdict"], "DATA_OR_IMAGING_BOTTLENECK")

    def test_regression(self):
        verdict = decide_verdict({"required_hit_rate": 0.50, "fp_per_100_frames": 10.0},
                                 {"required_hit_rate": 0.40, "fp_per_100_frames": 12.0})
        self.assertEqual(verdict["verdict"], "RAPID_V2_REGRESSION")

    def test_verdict_never_claims_production(self):
        verdict = decide_verdict({"required_hit_rate": 0.9, "fp_per_100_frames": 1.0},
                                 {"required_hit_rate": 0.99, "fp_per_100_frames": 1.0})
        self.assertNotIn(verdict["verdict"], ("Production PASS", "Formal GO", "Sealed PASS"))
        self.assertIn("temporal event layer", verdict["note"])


class ReviewStoreTests(unittest.TestCase):
    """Blindness, gating and resume on a synthetic but structurally real artifact."""

    @classmethod
    def setUpClass(cls):
        import serve_ground_litter_rapid_review as server

        cls.server = server

    def build_artifact(self, root: Path) -> dict:
        files = synthetic_files()
        split = build_split(files, PRIOR)
        train = next(r for r in split["rows"] if r["split"] == TRAIN_SPLIT)
        eval_row = next(r for r in split["rows"] if r["split"] == EVAL_SPLIT)
        frames = []
        for row in (train, eval_row):
            frames.append({
                "frame_id": frame_id(row["camera_id"], row["file_id"], 30),
                "camera_id": row["camera_id"], "file_id": row["file_id"],
                "split": row["split"], "kind": "fixed", "offset_seconds": 30,
                "requested_relative_seconds": 30.0, "nominal_frame_index": 750,
            })
        root.mkdir(parents=True, exist_ok=True)
        (root / "split.json").write_text(json.dumps(split), encoding="utf-8")
        with open(root / "frame_manifest.jsonl", "w", encoding="utf-8") as handle:
            for row in frames:
                handle.write(json.dumps(row) + "\n")
        records = []
        (root / "frames").mkdir(exist_ok=True)
        for row in frames:
            name = f'{row["frame_id"]}.png'
            (root / "frames" / name).write_bytes(b"\x89PNG\r\n\x1a\n")
            records.append({"frame_id": row["frame_id"], "image": name, "split": row["split"],
                            "camera_id": row["camera_id"], "file_id": row["file_id"],
                            "kind": row["kind"], "delta_ms": 0.0, "frame_index": 750,
                            "roi": ROI, "roi_geometry_version": "test", "image_sha256": "0" * 64,
                            "source_sha256": "0" * 64, "width": 2560, "height": 1440,
                            "canvas_size": [2560, 1440], "offset_seconds": 30,
                            "requested_relative_seconds": 30.0, "nominal_frame_index": 750,
                            "decoded_relative_seconds": 30.0, "is_bonus": False, "sought": True,
                            "image_bytes": 8})
        (root / "extraction_manifest.json").write_text(json.dumps({
            "records": records, "decode_failure_count": 0, "missing": [],
            "frames_extracted": len(records), "frames_requested": len(records),
            "split_sha256": split["split_sha256"], "counts": {}}), encoding="utf-8")
        (root / "baseline").mkdir(exist_ok=True)
        with open(root / "baseline" / "predictions.jsonl", "w", encoding="utf-8") as handle:
            for row in frames:
                handle.write(json.dumps({
                    "frame_id": row["frame_id"], "camera_id": row["camera_id"],
                    "file_id": row["file_id"], "split": row["split"],
                    "predictions": [{"prediction_id": f'{row["frame_id"]}_p000',
                                     "xyxy": [1480.0, 620.0, 1520.0, 660.0],
                                     "confidence": 0.42, "class_id": 0,
                                     "class_name": "ground_litter", "tile_xy": [1024, 512]}],
                }) + "\n")
        return {"split": split, "train": train, "eval": eval_row, "frames": frames}

    def test_stage_a_is_blind_and_stage_b_is_gated(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            fixture = self.build_artifact(artifact)
            store = self.server.RapidReviewStore(artifact)
            store.load_predictions()
            train_frame = fixture["frames"][0]["frame_id"]

            payload = store.frame_payload(train_frame)
            self.assertIsNone(payload["predictions"])
            self.assertIsNone(payload["prediction_reviews"])
            self.assertNotIn("prediction_count", payload)
            text = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("prediction_id", text)
            self.assertNotIn("confidence", text)
            self.assertEqual(payload["stage"], "truth")

            with self.assertRaises(RapidError):
                store.predictions_for(train_frame)
            with self.assertRaises(RapidError):
                store.review_prediction(train_frame, f"{train_frame}_p000", "Y")

            store.add_point(train_frame, "REQUIRED_LITTER", 1500.0, 640.0)
            payload = store.frame_payload(train_frame)
            self.assertIsNone(payload["predictions"])

            store.complete_truth(train_frame)
            payload = store.frame_payload(train_frame)
            self.assertEqual(len(payload["predictions"]), 1)
            self.assertEqual(payload["prediction_count"], 1)
            self.assertNotIn("confidence", json.dumps(payload["predictions"]))
            self.assertEqual(payload["stage"], "prediction")

            store.review_prediction(train_frame, f"{train_frame}_p000", "Y")
            self.assertEqual(store.stage_of(train_frame), "localization")

    def test_m_verdict_reopens_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            fixture = self.build_artifact(artifact)
            store = self.server.RapidReviewStore(artifact)
            store.load_predictions()
            train_frame = fixture["frames"][0]["frame_id"]
            store.complete_truth(train_frame)
            store.review_prediction(train_frame, f"{train_frame}_p000", "M")
            self.assertFalse(store.frame_state(train_frame)["truth_complete"])
            self.assertEqual(store.stage_of(train_frame), "truth")

    def test_eval_frame_export_guard_fires_over_the_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            fixture = self.build_artifact(artifact)
            store = self.server.RapidReviewStore(artifact)
            store.load_predictions()
            eval_frame = fixture["frames"][1]["frame_id"]
            self.assertEqual(fixture["frames"][1]["split"], EVAL_SPLIT)
            with self.assertRaises(RapidError):
                store.train_export_probe(eval_frame)
            train_frame = fixture["frames"][0]["frame_id"]
            self.assertTrue(store.train_export_probe(train_frame)["exported"])

    def test_eval_frame_reaches_done_without_localization(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            fixture = self.build_artifact(artifact)
            store = self.server.RapidReviewStore(artifact)
            store.load_predictions()
            eval_frame = fixture["frames"][1]["frame_id"]
            store.add_point(eval_frame, "REQUIRED_LITTER", 1500.0, 640.0)
            store.complete_truth(eval_frame)
            store.review_prediction(eval_frame, f"{eval_frame}_p000", "N")
            self.assertEqual(store.stage_of(eval_frame), "done")

    def test_resume_preserves_points_and_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            fixture = self.build_artifact(artifact)
            train_frame = fixture["frames"][0]["frame_id"]
            store = self.server.RapidReviewStore(artifact)
            store.load_predictions()
            store.add_point(train_frame, "UNCERTAIN", 100.0, 200.0)
            store.add_point(train_frame, "REQUIRED_LITTER", 1500.0, 640.0)
            store.complete_truth(train_frame)
            store.review_prediction(train_frame, f"{train_frame}_p000", "Y")

            resumed = self.server.RapidReviewStore(artifact)
            resumed.load_predictions()
            points = resumed.points_for(train_frame)
            self.assertEqual(len(points), 2)
            self.assertTrue(resumed.frame_state(train_frame)["truth_complete"])
            self.assertEqual(resumed.reviews_for(train_frame),
                             {f"{train_frame}_p000": "Y"})
            self.assertIn("truth_points.jsonl",
                          [p.name for p in (artifact / "review").iterdir()])

    def test_progress_shape_matches_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            fixture = self.build_artifact(artifact)
            store = self.server.RapidReviewStore(artifact)
            store.load_predictions()
            progress = store.progress()
            self.assertEqual(progress["truth_review"]["total"], 2)
            self.assertEqual(progress["truth_review"]["fixed_total"], 2)
            self.assertEqual(progress["rapid_eval"]["total"], 1)
            for key in ("required", "ignore", "uncertain"):
                self.assertIn(key, progress["truth_points"])
            self.assertIn("available", progress["prediction_review"])
            self.assertIn("required_on_train_complete", progress["localization"])

    def test_selftest_passes_on_clean_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            self.build_artifact(artifact)
            exit_code = self.server.main(["--artifact", str(artifact), "selftest"])
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
