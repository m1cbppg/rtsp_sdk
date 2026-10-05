"""Unit tests for the Ground Litter Rapid Eval v1 contract.

These tests are deliberately free of cv2/torch so they run in the plain repo venv.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from rtsp_annotator.ground_litter_rapid import (  # noqa: E402
    CAMERAS,
    COPY_CONFIRMED,
    COPY_PENDING,
    EVAL_SPLIT,
    ORIGIN_COPY,
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
    copy_eligibility,
    decide_verdict,
    dedupe_near_duplicates,
    frame_absolute_seconds,
    frame_id,
    frame_index_for,
    match_points,
    plan_training_export,
    point_inside_roi,
    read_jsonl,
    select_threshold,
    selection_hash,
    size_bucket,
    summarise_metrics,
    tile_starts,
    verify_split,
)

ROI = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def synthetic_files() -> list[dict]:
    """Synthetic Development inventory with realistic, gapless PS record times."""
    rows = []
    for camera in CAMERAS:
        base = datetime(2026, 9, 22, 15, 0, 0)
        for index in range(13):
            start = base + timedelta(seconds=304 * index)
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
                "record_start": start.strftime("%Y-%m-%d %H:%M:%S"),
                "record_end": (start + timedelta(seconds=304)).strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 304.0,
            })
    return rows


PRIOR = ["01021-ps00", "01022-ps01", "01022-ps02", "01022-ps03", "01027-ps00",
         "01030-ps00"]


def adjacent_train_then_eval(split: dict):
    """Find a rapid_train PS immediately followed by a rapid_eval PS on one camera.

    Those two are the real-world near-duplicate case across a PS boundary (gap = one PS
    length), which is what makes copy-previous eligible for an eval frame.
    """
    for camera in CAMERAS:
        rows = sorted((r for r in split["rows"] if r["camera_id"] == camera),
                      key=lambda r: r["record_start"])
        for previous, current in zip(rows, rows[1:]):
            if previous["split"] == TRAIN_SPLIT and current["split"] == EVAL_SPLIT:
                return previous, current
    raise AssertionError("synthetic split has no adjacent rapid_train -> rapid_eval PS pair")


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

    def build_artifact(self, root: Path, plans=None) -> dict:
        """Build a two-or-more frame artifact.  ``plans`` is a list of (file_id, offset)."""
        files = synthetic_files()
        split = build_split(files, PRIOR)
        by_file = {r["file_id"]: r for r in split["rows"]}
        train = next(r for r in split["rows"] if r["split"] == TRAIN_SPLIT)
        eval_row = next(r for r in split["rows"] if r["split"] == EVAL_SPLIT)
        if plans is None:
            plans = [(train["file_id"], 30), (eval_row["file_id"], 30)]
        frames = []
        for file_id_value, offset in plans:
            row = by_file[file_id_value]
            frames.append({
                "frame_id": frame_id(row["camera_id"], row["file_id"], offset),
                "camera_id": row["camera_id"], "file_id": row["file_id"],
                "split": row["split"], "kind": "fixed", "offset_seconds": offset,
                "requested_relative_seconds": float(offset),
                "nominal_frame_index": frame_index_for(offset),
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
                            "kind": row["kind"], "delta_ms": 0.0,
                            "frame_index": row["nominal_frame_index"],
                            "roi": ROI, "roi_geometry_version": "test", "image_sha256": "0" * 64,
                            "source_sha256": "0" * 64, "width": 2560, "height": 1440,
                            "canvas_size": [2560, 1440],
                            "offset_seconds": row["offset_seconds"],
                            "requested_relative_seconds": row["requested_relative_seconds"],
                            "nominal_frame_index": row["nominal_frame_index"],
                            "decoded_relative_seconds": row["requested_relative_seconds"],
                            "is_bonus": False, "sought": True, "image_bytes": 8})
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
        return {"split": split, "train": train, "eval": eval_row, "frames": frames,
                "by_file": by_file}

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


def sample(sample_id, camera, time_value, center, short_side=20.0, split=None):
    return {"sample_id": sample_id, "camera_id": camera, "time": float(time_value),
            "center_xy": list(center), "short_side": short_side, "split": split or TRAIN_SPLIT,
            "frame_id": sample_id.split("#")[0]}


class CopyEligibilityTests(unittest.TestCase):
    def test_same_ps_is_allowed(self):
        verdict = copy_eligibility({"camera_id": "01021", "file_id": "a", "absolute_seconds": 0},
                                   {"camera_id": "01021", "file_id": "a",
                                    "absolute_seconds": 500})
        self.assertTrue(verdict["allowed"])
        self.assertEqual(verdict["reason"], "same_ps")

    def test_time_continuous_across_ps_is_allowed(self):
        verdict = copy_eligibility({"camera_id": "01021", "file_id": "b", "absolute_seconds": 334},
                                   {"camera_id": "01021", "file_id": "a",
                                    "absolute_seconds": 30})
        self.assertTrue(verdict["allowed"])
        self.assertEqual(verdict["reason"], "temporally_continuous")

    def test_different_camera_is_refused(self):
        verdict = copy_eligibility({"camera_id": "01022", "file_id": "a", "absolute_seconds": 30},
                                   {"camera_id": "01021", "file_id": "a",
                                    "absolute_seconds": 30})
        self.assertFalse(verdict["allowed"])
        self.assertEqual(verdict["reason"], "different_camera")

    def test_large_time_gap_is_refused(self):
        verdict = copy_eligibility({"camera_id": "01021", "file_id": "c",
                                    "absolute_seconds": 3000},
                                   {"camera_id": "01021", "file_id": "a",
                                    "absolute_seconds": 30})
        self.assertFalse(verdict["allowed"])
        self.assertEqual(verdict["reason"], "time_gap_too_large")

    def test_missing_record_start_is_refused(self):
        verdict = copy_eligibility({"camera_id": "01021", "file_id": "b", "absolute_seconds": None},
                                   {"camera_id": "01021", "file_id": "a", "absolute_seconds": 30})
        self.assertFalse(verdict["allowed"])
        self.assertEqual(verdict["reason"], "no_record_start")

    def test_frame_absolute_seconds_uses_record_start_plus_offset(self):
        split_row = {"record_start": "2026-09-22 15:00:00"}
        first = frame_absolute_seconds({"requested_relative_seconds": 30}, split_row)
        second = frame_absolute_seconds({"requested_relative_seconds": 90}, split_row)
        self.assertAlmostEqual(second - first, 60.0)


class NearDuplicateTests(unittest.TestCase):
    def test_repeated_stationary_litter_is_capped(self):
        samples = [sample(f"f{i}", "01021", 1000 + 60 * i, [1000.0, 700.0], 20.0 + i)
                   for i in range(5)]
        result = dedupe_near_duplicates(samples, kind="positive")
        self.assertEqual(result["inputs_count"], 5)
        self.assertLessEqual(result["kept_count"], 2)
        self.assertEqual(result["cluster_count"], 1)
        self.assertEqual(result["dropped_count"], 5 - result["kept_count"])

    def test_distinct_positions_are_not_merged(self):
        samples = [sample("a", "01021", 1000, [400.0, 400.0]),
                   sample("b", "01021", 1000, [1800.0, 900.0])]
        result = dedupe_near_duplicates(samples, kind="positive")
        self.assertEqual(result["kept_count"], 2)
        self.assertEqual(result["cluster_count"], 2)

    def test_distant_times_are_not_merged(self):
        samples = [sample("a", "01021", 0, [1000.0, 700.0]),
                   sample("b", "01021", 5000, [1000.0, 700.0])]
        result = dedupe_near_duplicates(samples, kind="positive")
        self.assertEqual(result["kept_count"], 2)

    def test_different_cameras_are_never_merged(self):
        samples = [sample("a", "01021", 1000, [1000.0, 700.0]),
                   sample("b", "01022", 1000, [1000.0, 700.0])]
        result = dedupe_near_duplicates(samples, kind="positive")
        self.assertEqual(result["kept_count"], 2)

    def test_positive_representatives_are_deterministic(self):
        samples = [sample(f"f{i}", "01021", 1000 + 60 * i, [1000.0, 700.0], 10.0 + i)
                   for i in range(4)]
        first = dedupe_near_duplicates(samples, kind="positive")["kept"]
        shuffled = list(reversed(samples))
        second = dedupe_near_duplicates(shuffled, kind="positive")["kept"]
        self.assertEqual([s["sample_id"] for s in first],
                         [s["sample_id"] for s in second])

    def test_hard_negatives_keep_earliest_and_latest(self):
        samples = [sample(f"n{i}", "01021", 1000 + 60 * i, [500.0, 500.0])
                   for i in range(4)]
        kept = dedupe_near_duplicates(samples, kind="hard_negative")["kept"]
        self.assertEqual([s["sample_id"] for s in kept], ["n0", "n3"])

    def test_export_plan_refuses_eval_samples(self):
        with self.assertRaises(RapidError):
            plan_training_export([sample("x", "01021", 0, [1.0, 1.0], split=EVAL_SPLIT)], [])

    def test_export_plan_summary_shape(self):
        positives = [sample(f"p{i}", "01021", 0 + 30 * i, [800.0, 600.0]) for i in range(3)]
        negatives = [sample(f"n{i}", "01021", 0 + 30 * i, [2000.0, 900.0]) for i in range(3)]
        plan = plan_training_export(positives, negatives)
        self.assertEqual(plan["summary"]["positives_in"], 3)
        self.assertLessEqual(plan["summary"]["positives_exported"], 2)
        self.assertLessEqual(plan["summary"]["hard_negatives_exported"], 2)
        self.assertIn("Rapid-Eval", plan["note"])


class CopyPreviousStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import serve_ground_litter_rapid_review as server

        cls.server = server

    def make(self, tmp, plans=None):
        artifact = Path(tmp) / "artifact"
        fixture = ReviewStoreTests().build_artifact(artifact, plans)
        store = self.server.RapidReviewStore(artifact)
        store.load_predictions()
        return artifact, fixture, store

    @staticmethod
    def same_ps_plans():
        """Three fixed frames on one Rapid-Train PS: 30 s / 90 s / 150 s."""
        split = build_split(synthetic_files(), PRIOR)
        train = next(r for r in split["rows"] if r["split"] == TRAIN_SPLIT)
        file_id_value = train["file_id"]
        return [(file_id_value, 30), (file_id_value, 90), (file_id_value, 150)]

    def test_copy_source_detected_for_same_ps(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store = self.make(tmp, self.same_ps_plans())
            first, second = fixture["frames"][0]["frame_id"], fixture["frames"][1]["frame_id"]
            store.complete_truth(first)
            source = store.copy_source(second)
            self.assertTrue(source["allowed"])
            self.assertEqual(source["source_frame_id"], first)
            self.assertEqual(source["reason"], "same_ps")

    def test_copy_requires_confirm_and_keeps_stage_a_blind(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store = self.make(tmp, self.same_ps_plans())
            first, second = fixture["frames"][0]["frame_id"], fixture["frames"][1]["frame_id"]

            store.add_point(first, "REQUIRED_LITTER", 1500.0, 640.0)
            store.add_point(first, "IGNORE_SMALL", 300.0, 300.0)
            store.complete_truth(first)

            result = store.copy_previous(second)
            self.assertEqual(result["copy_state"], COPY_PENDING)
            self.assertFalse(result["truth_complete"])
            entry = store.frame_state(second)
            self.assertEqual(entry["copy_state"], COPY_PENDING)
            self.assertFalse(entry["truth_complete"])
            self.assertEqual(entry["copied_point_count"], 2)

            # Stage A stays blind even though the points are copied.
            payload = store.frame_payload(second)
            self.assertIsNone(payload["predictions"])
            self.assertNotIn("prediction_count", payload)
            self.assertTrue(payload["copy"]["pending"])
            self.assertEqual(payload["copy"]["state"], COPY_PENDING)
            self.assertEqual(len(payload["truth_points"]), 2)

            # Enter is what turns the copy into truth.
            store.complete_truth(second)
            entry = store.frame_state(second)
            self.assertTrue(entry["truth_complete"])
            self.assertEqual(entry["copy_state"], COPY_CONFIRMED)
            self.assertIsNotNone(store.frame_payload(second)["predictions"])

    def test_copy_preserves_source_native_coordinates_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store = self.make(tmp, self.same_ps_plans())
            first, second = fixture["frames"][0]["frame_id"], fixture["frames"][1]["frame_id"]
            store.add_point(first, "REQUIRED_LITTER", 1234.56, 789.01)
            store.complete_truth(first)
            store.copy_previous(second)
            copied = store.points_for(second)[0]
            self.assertEqual(copied["source_xy"], [1234.56, 789.01])
            self.assertEqual(copied["origin"], ORIGIN_COPY)
            self.assertEqual(copied["copied_from_frame_id"], first)

    def test_copy_is_refused_across_cameras(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = synthetic_files()
            split = build_split(files, PRIOR)
            train_rows = [r for r in split["rows"] if r["split"] == TRAIN_SPLIT]
            first_camera = train_rows[0]["camera_id"]
            other = next(r for r in train_rows if r["camera_id"] != first_camera)
            plans = [(train_rows[0]["file_id"], 30), (other["file_id"], 30)]
            artifact, fixture, store = self.make(tmp, plans)
            first, second = fixture["frames"][0]["frame_id"], fixture["frames"][1]["frame_id"]
            store.add_point(first, "REQUIRED_LITTER", 1500.0, 640.0)
            store.complete_truth(first)
            source = store.copy_source(second)
            self.assertFalse(source["allowed"])
            self.assertEqual(source["reason"], "different_camera")
            with self.assertRaises(RapidError):
                store.copy_previous(second)

    def test_copy_refused_without_a_completed_previous_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store = self.make(tmp, self.same_ps_plans())
            second = fixture["frames"][1]["frame_id"]
            with self.assertRaises(RapidError):
                store.copy_previous(second)

    def test_copy_only_replaces_this_frames_own_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store = self.make(tmp, self.same_ps_plans())
            first, second = fixture["frames"][0]["frame_id"], fixture["frames"][1]["frame_id"]
            store.add_point(first, "REQUIRED_LITTER", 1500.0, 640.0)
            store.complete_truth(first)
            store.add_point(second, "UNCERTAIN", 900.0, 400.0)
            store.copy_previous(second)
            classes = [p["truth_class"] for p in store.points_for(second)]
            self.assertEqual(classes, ["REQUIRED_LITTER"])
            self.assertEqual(len(store.points_for(first)), 1)

    def test_editing_after_copy_is_recorded_but_stays_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store = self.make(tmp, self.same_ps_plans())
            first, second = fixture["frames"][0]["frame_id"], fixture["frames"][1]["frame_id"]
            store.add_point(first, "REQUIRED_LITTER", 1500.0, 640.0)
            store.complete_truth(first)
            store.copy_previous(second)
            store.add_point(second, "REQUIRED_LITTER", 400.0, 400.0)
            entry = store.frame_state(second)
            self.assertEqual(entry["copy_state"], COPY_PENDING)
            self.assertFalse(entry["truth_complete"])
            self.assertTrue(entry["copy_edited"])

    def test_copied_localization_becomes_a_candidate_not_a_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store = self.make(tmp, self.same_ps_plans())
            first, second = fixture["frames"][0]["frame_id"], fixture["frames"][1]["frame_id"]
            point = store.add_point(first, "REQUIRED_LITTER", 1490.0, 630.0)
            store.complete_truth(first)
            prediction_id = f"{first}_p000"
            store.review_prediction(first, prediction_id, "Y")
            store.select_localization(point["truth_id"], 1,
                                      resolver=self.server.Localizer(artifact, store,
                                                                     enable_semantic=False)
                                      .candidates)

            store.copy_previous(second)
            copied = store.points_for(second)[0]
            self.assertIsNotNone(copied["copied_localization"])
            # Not selected yet on the new frame:
            self.assertEqual(store.localization_for(second), {})
            self.assertEqual(store.stage_of(second), "truth")

            store.complete_truth(second)
            self.assertEqual(store.stage_of(second), "prediction")
            # Stage B first: every baseline prediction must be judged before Stage C opens.
            store.review_prediction(second, f"{second}_p000", "Y")
            self.assertEqual(store.stage_of(second), "localization")
            localizer = self.server.Localizer(artifact, store, enable_semantic=False)
            payload = localizer.candidates(copied["truth_id"])
            sources = [c["proposal_source"] for c in payload["candidates"]]
            self.assertEqual(sources[0], ORIGIN_COPY)
            self.assertTrue(payload["candidates"][0]["contains_point"])

            store.select_localization(copied["truth_id"], 1, resolver=localizer.candidates)
            selection = store.localization_for(second)[copied["truth_id"]]
            self.assertEqual(selection["status"], "LOCALIZED")
            self.assertEqual(selection["proposal_source"], ORIGIN_COPY)

    def test_eval_frame_copy_still_requires_enter_and_stays_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = build_split(synthetic_files(), PRIOR)
            previous, target = adjacent_train_then_eval(split)
            plans = [(previous["file_id"], 30), (target["file_id"], 30)]
            artifact, fixture, store = self.make(tmp, plans)
            first, second = fixture["frames"][0]["frame_id"], fixture["frames"][1]["frame_id"]
            self.assertEqual(fixture["frames"][0]["split"], TRAIN_SPLIT)
            self.assertEqual(fixture["frames"][1]["split"], EVAL_SPLIT)
            store.add_point(first, "REQUIRED_LITTER", 1500.0, 640.0)
            store.complete_truth(first)

            # Even a near-duplicate eval frame must be copied, inspected and confirmed by hand.
            store.copy_previous(second)
            self.assertFalse(store.frame_state(second)["truth_complete"])
            progress = store.progress()
            self.assertEqual(progress["rapid_eval"]["total"], 1)
            self.assertEqual(progress["rapid_eval"]["complete"], 0)

            store.complete_truth(second)
            progress = store.progress()
            self.assertEqual(progress["rapid_eval"]["total"], 1)
            self.assertEqual(progress["rapid_eval"]["complete"], 1)

    def test_copy_previous_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store = self.make(tmp, self.same_ps_plans())
            first, second = fixture["frames"][0]["frame_id"], fixture["frames"][1]["frame_id"]
            store.add_point(first, "REQUIRED_LITTER", 1500.0, 640.0)
            store.complete_truth(first)
            store.copy_previous(second)

            resumed = self.server.RapidReviewStore(artifact)
            resumed.load_predictions()
            entry = resumed.frame_state(second)
            self.assertEqual(entry["copy_state"], COPY_PENDING)
            self.assertFalse(entry["truth_complete"])
            self.assertEqual(len(resumed.points_for(second)), 1)


class ExportPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import serve_ground_litter_rapid_review as server

        cls.server = server

    def build(self, tmp):
        artifact = Path(tmp) / "artifact"
        fixture = ReviewStoreTests().build_artifact(artifact)
        store = self.server.RapidReviewStore(artifact)
        store.load_predictions()
        localizer = self.server.Localizer(artifact, store, enable_semantic=False)
        return artifact, fixture, store, localizer

    def test_plan_excludes_eval_and_caps_near_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.build(tmp)
            train_frame = fixture["frames"][0]["frame_id"]
            point = store.add_point(train_frame, "REQUIRED_LITTER", 1490.0, 630.0)
            store.complete_truth(train_frame)
            store.review_prediction(train_frame, f"{train_frame}_p000", "Y")
            store.select_localization(point["truth_id"], 1, resolver=localizer.candidates)

            plan = store.export_plan()
            self.assertGreaterEqual(plan["summary"]["positives_in"], 1)
            self.assertEqual(plan["rapid_eval_denominator"]["frames"], 1)
            self.assertTrue(plan["rapid_eval_denominator"]["excluded_from_training"])
            for sample_row in plan["positives"]["kept"]:
                self.assertEqual(sample_row["split"], TRAIN_SPLIT)
            self.assertEqual(plan["summary"]["hard_negatives_in"], 0)

    def test_plan_does_not_touch_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.build(tmp)
            train_frame = fixture["frames"][0]["frame_id"]
            point = store.add_point(train_frame, "REQUIRED_LITTER", 1490.0, 630.0)
            store.complete_truth(train_frame)
            store.review_prediction(train_frame, f"{train_frame}_p000", "Y")
            store.select_localization(point["truth_id"], 1, resolver=localizer.candidates)

            before = {name: (artifact / name).read_bytes()
                      for name in ("split.json", "frame_manifest.jsonl")}
            truth_before = (artifact / "review" / "truth_points.jsonl").read_bytes()
            store.export_plan()
            for name, payload in before.items():
                self.assertEqual((artifact / name).read_bytes(), payload)
            self.assertEqual((artifact / "review" / "truth_points.jsonl").read_bytes(),
                             truth_before)


class LocalizationSubmitTests(unittest.TestCase):
    """Regression cover for the Stage C submit/advance bug (A-H) and its two root causes."""

    @classmethod
    def setUpClass(cls):
        import serve_ground_litter_rapid_review as server

        cls.server = server

    def make(self, tmp, plans=None):
        artifact = Path(tmp) / "artifact"
        fixture = ReviewStoreTests().build_artifact(artifact, plans)
        store = self.server.RapidReviewStore(artifact)
        store.load_predictions()
        localizer = self.server.Localizer(artifact, store, enable_semantic=False)
        return artifact, fixture, store, localizer

    @staticmethod
    def same_ps_plans():
        split = build_split(synthetic_files(), PRIOR)
        train = next(r for r in split["rows"] if r["split"] == TRAIN_SPLIT)
        return [(train["file_id"], 30), (train["file_id"], 90)]

    def three_points(self, store, frame_id):
        return [store.add_point(frame_id, "REQUIRED_LITTER", 1400.0 + 40 * i, 600.0 + 30 * i)
                for i in range(3)]

    # --- A: a plain manual point saves and advances ----------------------- #
    def test_A_manual_point_submit_saves_and_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.make(tmp, self.same_ps_plans())
            frame = fixture["frames"][0]["frame_id"]
            points = self.three_points(store, frame)
            store.complete_truth(frame)

            first = store.select_localization(points[0]["truth_id"], 0, frame_id=frame,
                                              resolver=localizer.candidates)
            self.assertEqual(first["status"], "UNLOCALIZED_SKIP")
            self.assertEqual(first["frame_id"], frame)
            self.assertEqual(first["remaining"], 2)
            self.assertEqual(first["next_truth_id"], points[1]["truth_id"])
            self.assertFalse(first["localization_complete"])
            self.assertEqual(store.localization_for(frame)[points[0]["truth_id"]]["status"],
                             "UNLOCALIZED_SKIP")

    # --- B: a copied point saves and advances ---------------------------- #
    def test_B_copied_point_submit_saves_and_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.make(tmp, self.same_ps_plans())
            source, target = fixture["frames"][0]["frame_id"], fixture["frames"][1]["frame_id"]
            store.add_point(source, "REQUIRED_LITTER", 1500.0, 640.0)
            store.complete_truth(source)
            store.copy_previous(target)
            copied = store.points_for(target)[0]
            self.assertEqual(copied["origin"], ORIGIN_COPY)
            store.complete_truth(target)

            result = store.select_localization(copied["truth_id"], 0, frame_id=target,
                                              resolver=localizer.candidates)
            self.assertEqual(result["frame_id"], target)
            self.assertEqual(store.localization_for(target)[copied["truth_id"]]["status"],
                             "UNLOCALIZED_SKIP")
            self.assertTrue(result["localization_complete"])

    # --- C/D: clicking candidate A is exactly choice 1, B is choice 2 ----- #
    def test_CD_candidate_choice_matches_the_offered_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.make(tmp, self.same_ps_plans())
            source, target = fixture["frames"][0]["frame_id"], fixture["frames"][1]["frame_id"]
            point = store.add_point(source, "REQUIRED_LITTER", 1490.0, 630.0)
            store.complete_truth(source)
            store.review_prediction(source, f"{source}_p000", "Y")
            store.select_localization(point["truth_id"], 1, frame_id=source,
                                      resolver=localizer.candidates)
            store.copy_previous(target)
            copied = store.points_for(target)[0]
            store.complete_truth(target)

            offered = localizer.candidates(copied["truth_id"], target)["candidates"]
            self.assertEqual(offered[0]["label"], "A")
            self.assertEqual(offered[0]["idx"], 0)
            self.assertEqual(offered[0]["proposal_source"], ORIGIN_COPY)

            # "click card A" and "press 1" both mean choice = idx + 1 = 1.
            result = store.select_localization(copied["truth_id"], 1, frame_id=target,
                                              resolver=localizer.candidates)
            self.assertEqual(result["proposal_source"], offered[0]["proposal_source"])
            self.assertEqual(result["selected_bbox_source_xyxy"], offered[0]["bbox_xyxy"])
            self.assertEqual(result["choice"], 1)

    # --- E: 0 = skip, and it also advances ------------------------------- #
    def test_E_zero_skips_and_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.make(tmp, self.same_ps_plans())
            frame = fixture["frames"][0]["frame_id"]
            points = self.three_points(store, frame)
            store.complete_truth(frame)
            result = store.select_localization(points[0]["truth_id"], 0, frame_id=frame)
            self.assertEqual(result["status"], "UNLOCALIZED_SKIP")
            self.assertIsNone(result["selected_bbox_source_xyxy"])
            self.assertEqual(result["next_truth_id"], points[1]["truth_id"])

    # --- F: duplicate submit is idempotent ------------------------------- #
    def test_F_duplicate_submit_does_not_append_a_second_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.make(tmp, self.same_ps_plans())
            frame = fixture["frames"][0]["frame_id"]
            points = self.three_points(store, frame)
            store.complete_truth(frame)

            first = store.select_localization(points[0]["truth_id"], 0, frame_id=frame)
            self.assertFalse(first["already_recorded"])
            rows_before = len(store.localization_reviews)
            created_before = first["created_at"]

            for _ in range(3):
                again = store.select_localization(points[0]["truth_id"], 0, frame_id=frame)
                self.assertTrue(again["already_recorded"])
                self.assertEqual(again["created_at"], created_before)
            self.assertEqual(len(store.localization_reviews), rows_before)
            on_disk = [r for r in read_jsonl(artifact / "review"
                                             / "localization_reviews.jsonl")
                       if r["truth_id"] == points[0]["truth_id"]]
            self.assertEqual(len(on_disk), 1)

    def test_F2_changing_the_choice_overwrites_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.make(tmp, self.same_ps_plans())
            source, target = fixture["frames"][0]["frame_id"], fixture["frames"][1]["frame_id"]
            point = store.add_point(source, "REQUIRED_LITTER", 1490.0, 630.0)
            store.complete_truth(source)
            store.review_prediction(source, f"{source}_p000", "Y")
            store.select_localization(point["truth_id"], 1, frame_id=source,
                                      resolver=localizer.candidates)
            store.copy_previous(target)
            copied = store.points_for(target)[0]
            store.complete_truth(target)

            first = store.select_localization(copied["truth_id"], 1, frame_id=target,
                                              resolver=localizer.candidates)
            self.assertEqual(first["status"], "LOCALIZED")
            # changing your mind replaces the decision, it never adds a second row
            second = store.select_localization(copied["truth_id"], 0, frame_id=target,
                                               resolver=localizer.candidates)
            self.assertFalse(second["already_recorded"])
            self.assertEqual(second["status"], "UNLOCALIZED_SKIP")
            rows = [r for r in store.localization_reviews
                    if r["truth_id"] == copied["truth_id"]]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "UNLOCALIZED_SKIP")

    # --- G: the last point completes Stage C ----------------------------- #
    def test_G_last_point_completes_stage_c(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.make(tmp, self.same_ps_plans())
            frame = fixture["frames"][0]["frame_id"]
            points = self.three_points(store, frame)
            store.complete_truth(frame)
            store.review_prediction(frame, f"{frame}_p000", "Y")

            for index, point in enumerate(points):
                result = store.select_localization(point["truth_id"], 0, frame_id=frame)
                self.assertEqual(result["remaining"], len(points) - index - 1)
                self.assertEqual(result["localization_complete"], index == len(points) - 1)
            self.assertIsNone(result["next_truth_id"])
            self.assertEqual(store.pending_localization(frame), [])
            self.assertEqual(store.stage_of(frame), "done")

    # --- H: refresh / restart keeps completed localizations -------------- #
    def test_H_completed_localization_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.make(tmp, self.same_ps_plans())
            frame = fixture["frames"][0]["frame_id"]
            points = self.three_points(store, frame)
            store.complete_truth(frame)
            store.select_localization(points[0]["truth_id"], 0, frame_id=frame)
            store.select_localization(points[1]["truth_id"], 0, frame_id=frame)

            resumed = self.server.RapidReviewStore(artifact)
            resumed.load_predictions()
            done = resumed.localization_for(frame)
            self.assertEqual(len(done), 2)
            pending = [p["truth_id"] for p in resumed.pending_localization(frame)]
            self.assertEqual(pending, [points[2]["truth_id"]])

    def test_H2_localization_reviews_are_not_truncated_by_a_new_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.make(tmp, self.same_ps_plans())
            frame = fixture["frames"][0]["frame_id"]
            points = self.three_points(store, frame)
            store.complete_truth(frame)
            for point in points[:2]:
                store.select_localization(point["truth_id"], 0, frame_id=frame)
            path = artifact / "review" / "localization_reviews.jsonl"
            self.assertEqual(len(read_jsonl(path)), 2)

            # A fresh process (as after a service restart) must load them, and its next write
            # must keep them instead of rewriting the file with a single row.
            resumed = self.server.RapidReviewStore(artifact)
            resumed.load_predictions()
            resumed.select_localization(points[2]["truth_id"], 0, frame_id=frame)
            self.assertEqual(len(read_jsonl(path)), 3)

    # --- the actual root cause: duplicate truth_id must not cross-write --- #
    def test_duplicate_truth_id_is_frame_scoped_and_never_cross_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.make(tmp, self.same_ps_plans())
            frame_a, frame_b = (fixture["frames"][0]["frame_id"],
                                fixture["frames"][1]["frame_id"])
            store.add_point(frame_a, "REQUIRED_LITTER", 100.0, 100.0)
            store.add_point(frame_b, "REQUIRED_LITTER", 200.0, 200.0)
            # Reproduce the legacy collision observed in the live artifact.
            for row in store.points:
                row["truth_id"] = "t-00039"
            store._save_points()

            reloaded = self.server.RapidReviewStore(artifact)
            reloaded.load_predictions()
            localizer = self.server.Localizer(artifact, reloaded, enable_semantic=False)
            self.assertEqual(len([p for p in reloaded.points
                                  if p["truth_id"] == "t-00039"]), 2)
            with self.assertRaises(RapidError):
                reloaded.find_point("t-00039")
            self.assertEqual(reloaded.find_point("t-00039", frame_a)["source_xy"], [100.0, 100.0])
            self.assertEqual(reloaded.find_point("t-00039", frame_b)["source_xy"], [200.0, 200.0])

            reloaded.complete_truth(frame_b)
            entry = reloaded.select_localization("t-00039", 0, frame_id=frame_b)
            self.assertEqual(entry["frame_id"], frame_b)
            self.assertEqual(entry["camera_id"], "01021")
            self.assertEqual(reloaded.localization_for(frame_a), {})
            self.assertNotIn("t-00039", reloaded.localization_for(frame_a))
            on_disk = [r for r in read_jsonl(artifact / "review"
                                             / "localization_reviews.jsonl")
                       if r["truth_id"] == "t-00039"]
            self.assertEqual(len(on_disk), 1)
            self.assertEqual(on_disk[0]["frame_id"], frame_b)

    def test_inherited_seed_runs_at_most_once(self):
        """A restart must never resurrect points the operator deleted or replaced."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            fixture = ReviewStoreTests().build_artifact(artifact, self.same_ps_plans())
            frame = fixture["frames"][1]["frame_id"]
            # mark that frame as a bonus train frame and give it official inherited marks
            manifest_path = artifact / "frame_manifest.jsonl"
            rows = read_jsonl(manifest_path)
            for row in rows:
                if row["frame_id"] == frame:
                    row["kind"] = "bonus_train"
            manifest_path.write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            inherited_dir = artifact / "inherited_truth"
            inherited_dir.mkdir(exist_ok=True)
            (inherited_dir / f"{frame}.jsonl").write_text("".join(json.dumps({
                "truth_id": f"t-000{i:02d}", "source_xy": [700.0 + 10 * i, 500.0 + 10 * i],
                "truth_class": "REQUIRED_LITTER", "in_roi": True,
            }) + "\n" for i in (1, 2)), encoding="utf-8")

            store = self.server.RapidReviewStore(artifact)
            store.load_predictions()
            seeded = store.points_for(frame)
            self.assertEqual(len(seeded), 2)
            self.assertTrue(all(p["origin"] == "official_discovery_inherited" for p in seeded))
            self.assertTrue(store.frame_state(frame)["inherited_seeded"])

            # the operator deletes one and replaces the rest via a copy
            store.delete_point(seeded[0]["truth_id"], frame)
            resumed = self.server.RapidReviewStore(artifact)
            resumed.load_predictions()
            self.assertEqual(len(resumed.points_for(frame)), 1)

            # and an emptied but completed frame must stay empty
            for point in list(resumed.points_for(frame)):
                resumed.delete_point(point["truth_id"], frame)
            resumed.complete_truth(frame)
            again = self.server.RapidReviewStore(artifact)
            again.load_predictions()
            self.assertEqual(again.points_for(frame), [])

    def test_add_point_never_reuses_a_deleted_truth_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.make(tmp, self.same_ps_plans())
            frame = fixture["frames"][0]["frame_id"]
            first = store.add_point(frame, "REQUIRED_LITTER", 100.0, 100.0)
            second = store.add_point(frame, "REQUIRED_LITTER", 200.0, 200.0)
            store.delete_point(second["truth_id"], frame)
            third = store.add_point(frame, "REQUIRED_LITTER", 300.0, 300.0)
            live = [p["truth_id"] for p in store.points]
            self.assertEqual(len(live), len(set(live)))
            self.assertNotEqual(third["truth_id"], second["truth_id"])
            self.assertNotEqual(third["truth_id"], first["truth_id"])

    def test_delete_point_only_affects_the_given_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, fixture, store, localizer = self.make(tmp, self.same_ps_plans())
            frame_a, frame_b = (fixture["frames"][0]["frame_id"],
                                fixture["frames"][1]["frame_id"])
            store.add_point(frame_a, "REQUIRED_LITTER", 100.0, 100.0)
            store.add_point(frame_b, "REQUIRED_LITTER", 200.0, 200.0)
            for row in store.points:
                row["truth_id"] = "t-00039"
            store._save_points()
            store.delete_point("t-00039", frame_a)
            self.assertEqual(store.points_for(frame_a), [])
            self.assertEqual(len(store.points_for(frame_b)), 1)


if __name__ == "__main__":
    unittest.main()
