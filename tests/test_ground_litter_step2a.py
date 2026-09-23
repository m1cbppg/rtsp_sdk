"""Step 2A: YOLO26s loader sanity + tiny overfit (§39).

Pure-stdlib logic is exercised with synthetic fixtures; real frozen Step 1C-2M / 1D
artifacts and the recorded Step 2A evidence are checked only when they are present on
disk (they are read-only inputs/evidence, not test fixtures).  No torch / ultralytics /
cv2 import is required for the suite: the one live-loader test skips when the real
Ultralytics stack is unavailable here.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rtsp_annotator.ground_litter_localization_review import (
    SEALED_MARKERS,
    SealedAssetError,
    assert_not_sealed,
    sha256_file,
)
from rtsp_annotator.ground_litter_positive_tiles import png_bytes
from rtsp_annotator.ground_litter_step2a import (
    CONF_LEVELS,
    DATA_YAML_NOTE,
    EASY_NEGATIVE_HARDNESS,
    IOU_NORMAL,
    IOU_SMALL,
    MAX_EASY_NEGATIVE_FRACTION,
    NEGATIVE_EXPECTED,
    POSITIVE_EXPECTED,
    SCHEMA_VERSION,
    SEED,
    SMALL_AREA_RATIO_MAX,
    SMALL_AREA_RATIO_MIN,
    SMALL_EXPAND,
    SMALL_GT_SHORT_SIDE_PX,
    TINY_MAX,
    TINY_MIN,
    TINY_NEGATIVE_TARGET,
    TINY_POSITIVE_TARGET,
    TRAIN_EQUALS_VAL,
    Step2AError,
    boundary_flags,
    box_iou,
    build_manifest,
    build_summary,
    center_inside,
    evaluate_predictions,
    expand_box,
    is_eligible_match,
    load_pools,
    match_predictions,
    overfit_verdict,
    select_subsets,
    stage_dataset,
    subset_manifest,
    verify_preflight,
    verify_staging,
    write_reports,
)

ROOT = Path(__file__).resolve().parents[1]
POSITIVE_ROOT = ROOT / "output" / "ground_litter_positive_tile_completion_20260923"
NEGATIVE_ROOT = ROOT / "output" / "ground_litter_hard_negatives_20260923"
WEIGHT = ROOT / "models" / "yolo26s.pt"
STEP2A_OUT = ROOT / "output" / "ground_litter_step2a_20260923"
CLI = ROOT / "scripts" / "run_ground_litter_step2a.py"
MODULE = ROOT / "rtsp_annotator" / "ground_litter_step2a.py"

#: recorded 2026-09-23 evidence for the pretrained yolo26s weight
WEIGHT_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
WEIGHT_BYTES = 20_422_725

CAMERAS = ("01021", "01022", "01028", "01030", "01027")
HARDNESS = "historical_false_positive"


def _has_real_pools() -> bool:
    return ((POSITIVE_ROOT / "positive_training_manifest_v2.jsonl").is_file()
            and (NEGATIVE_ROOT / "hard_negative_training_manifest.jsonl").is_file()
            and (POSITIVE_ROOT / "accepted_v2" / "images").is_dir()
            and (NEGATIVE_ROOT / "accepted" / "images").is_dir())


REAL_POOLS = _has_real_pools()
requires_real_pools = unittest.skipUnless(
    REAL_POOLS, "frozen Step 1C-2M / 1D training pools are not present on disk")


def _has_ultralytics() -> bool:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return True


def _box(x1: int, y1: int, side: int, tall: int | None = None):
    return [float(x1), float(y1), float(x1 + side), float(y1 + (tall or side))]


def _png(path: Path, seed: int) -> None:
    array = np.zeros((64, 64, 3), dtype="uint8")
    array[..., 0] = (seed * 37) % 256
    array[..., 1] = (seed * 71) % 256
    array[..., 2] = (seed * 13) % 256
    path.write_bytes(png_bytes(array))


def _yolo_lines(boxes) -> str:
    lines = []
    for x1, y1, x2, y2 in boxes:
        lines.append(f"0 {(x1 + x2) / 128.0:.6f} {(y1 + y2) / 128.0:.6f} "
                     f"{(x2 - x1) / 64.0:.6f} {(y2 - y1) / 64.0:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


POSITIVE_SPEC = [
    ("pt-a01", "01021", [_box(4, 4, 8)]),
    ("pt-a02", "01022", [_box(4, 4, 12)]),
    ("pt-a03", "01028", [_box(2, 2, 25), _box(34, 34, 14)]),
    ("pt-a04", "01030", [_box(2, 2, 30), _box(36, 36, 28)]),
    ("pt-a05", "01027", [_box(4, 4, 50)]),
    ("pt-a06", "01021", [_box(2, 2, 22), _box(30, 30, 9)]),
    ("pt-a07", "01022", [_box(2, 2, 24), _box(28, 28, 26), _box(2, 30, 11)]),
    ("pt-a08", "01028", [_box(6, 6, 35)]),
    ("pt-a09", "01030", [_box(2, 2, 23), _box(26, 26, 27), _box(2, 28, 8)]),
    ("pt-a10", "01027", [_box(8, 8, 40)]),
    ("pt-a11", "01021", [_box(10, 10, 33)]),
    ("pt-a12", "01022", [_box(12, 12, 34)]),
    ("pt-a13", "01028", [_box(14, 14, 36)]),
    ("pt-a14", "01030", [_box(16, 16, 37)]),
    ("pt-a15", "01027", [_box(18, 18, 38)]),
    ("pt-a16", "01021", [_box(20, 20, 15)]),
]

NEGATIVE_HARD = [(f"nt-b{index:02d}", CAMERAS[index % len(CAMERAS)])
                 for index in range(1, 13)]
NEGATIVE_EASY = [("nt-c01", "01021"), ("nt-c02", "01028")]


def build_synthetic_pools(root: Path) -> tuple[Path, Path]:
    """A tiny self-consistent Step 1C-2M / 1D-shaped pool for logic tests."""
    positive_root = root / "positive"
    negative_root = root / "negative"
    positive_images = positive_root / "accepted_v2" / "images"
    positive_labels = positive_root / "accepted_v2" / "labels"
    negative_images = negative_root / "accepted" / "images"
    negative_labels = negative_root / "accepted" / "labels"
    for path in (positive_images, positive_labels, negative_images, negative_labels):
        path.mkdir(parents=True, exist_ok=True)

    positive_rows = []
    for index, (tile_id, camera, boxes) in enumerate(POSITIVE_SPEC):
        image = positive_images / f"{tile_id}.png"
        label = positive_labels / f"{tile_id}.txt"
        _png(image, index + 1)
        label.write_text(_yolo_lines(boxes), encoding="utf-8")
        positive_rows.append({
            "tile_id": tile_id, "camera_id": camera,
            "image_path": str(image), "label_path": str(label),
            "image_sha256": sha256_file(image),
            "labels": [{"tile_xyxy": box} for box in boxes],
            "origin": "synthetic", "source_file_id": f"file-{camera}",
            "source_crop_xyxy": [0, 0, 640, 640],
        })
    negative_rows = []
    for index, (tile_id, camera) in enumerate(NEGATIVE_HARD + NEGATIVE_EASY):
        image = negative_images / f"{tile_id}.png"
        label = negative_labels / f"{tile_id}.txt"
        _png(image, 100 + index)
        label.write_bytes(b"")                       # the real 0-byte contract
        negative_rows.append({
            "negative_tile_id": tile_id, "camera_id": camera,
            "image_path": str(image), "label_path": str(label),
            "image_sha256": sha256_file(image),
            "origin": "synthetic",
            "hardness_source": (HARDNESS if (tile_id, camera) in NEGATIVE_HARD
                                else EASY_NEGATIVE_HARDNESS),
            "anchor_tile_xyxy": [0, 0, 640, 640],
            "source_file_id": f"file-{camera}",
            "source_crop_xyxy": [0, 0, 640, 640],
        })
    _write_jsonl(positive_root / "positive_training_manifest_v2.jsonl", positive_rows)
    _write_jsonl(negative_root / "hard_negative_training_manifest.jsonl", negative_rows)
    for path in (positive_root / "SUMMARY.json", positive_root / "MANIFEST.json",
                 negative_root / "SUMMARY.json", negative_root / "MANIFEST.json"):
        path.write_text("{}\n", encoding="utf-8")
    return positive_root, negative_root


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class SyntheticPoolCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        positive_root, negative_root = build_synthetic_pools(self.tmp / "pools")
        self.pools = load_pools(positive_root, negative_root)
        self.subsets = select_subsets(self.pools)
        self.manifest = subset_manifest(self.pools, self.subsets)


class TestFrozenSubsetDeterminism(SyntheticPoolCase):
    def test_selection_is_deterministic(self):
        again = select_subsets(load_pools(self.pools["positive_root"],
                                          self.pools["negative_root"]))
        self.assertEqual([row["row"]["tile_id"] for row in self.subsets["positives"]],
                         [row["row"]["tile_id"] for row in again["positives"]])
        self.assertEqual([row["row"]["tile_id"] for row in self.subsets["negatives"]],
                         [row["row"]["tile_id"] for row in again["negatives"]])
        self.assertEqual(json.dumps(subset_manifest(self.pools, self.subsets),
                                    sort_keys=True),
                         json.dumps(subset_manifest(self.pools, again), sort_keys=True))

    def test_subset_size_within_bounds(self):
        self.assertEqual(len(self.manifest["positive"]), TINY_POSITIVE_TARGET)
        self.assertEqual(len(self.manifest["negative"]), TINY_NEGATIVE_TARGET)
        self.assertGreaterEqual(len(self.manifest["positive"]), TINY_MIN)
        self.assertLessEqual(len(self.manifest["positive"]), TINY_MAX)
        self.assertGreaterEqual(len(self.manifest["negative"]), TINY_MIN)
        self.assertLessEqual(len(self.manifest["negative"]), TINY_MAX)

    def test_subset_covers_every_camera_and_easy_negative_fraction(self):
        coverage = self.manifest["coverage"]
        self.assertEqual(sorted(coverage["positive_cameras"]), sorted(CAMERAS))
        self.assertEqual(sorted(coverage["negative_cameras"]), sorted(CAMERAS))
        easy = coverage["easy_negative_count"]
        self.assertLessEqual(easy, int(TINY_NEGATIVE_TARGET
                                       * MAX_EASY_NEGATIVE_FRACTION))
        self.assertEqual(coverage["positive_bboxes"],
                         sum(entry["box_count"] for entry in self.manifest["positive"]))

    def test_subset_is_a_prefix_free_selection_without_duplicates(self):
        positives = self.manifest["positive_tile_ids"]
        negatives = self.manifest["negative_tile_ids"]
        self.assertEqual(len(positives), len(set(positives)))
        self.assertEqual(len(negatives), len(set(negatives)))
        self.assertEqual(len(set(positives) & set(negatives)), 0)

    def test_subset_manifest_records_seed_and_schema(self):
        self.assertEqual(self.manifest["seed"], SEED)
        self.assertEqual(self.manifest["schema_version"], SCHEMA_VERSION)
        self.assertIn(DATA_YAML_NOTE, self.manifest["note"])


class TestStagingIsReadOnly(SyntheticPoolCase):
    def test_staging_preserves_bytes_and_zero_byte_negative_labels(self):
        report = stage_dataset(self.manifest, self.tmp / "dataset", mode="hardlink")
        integrity = verify_staging(report)
        self.assertTrue(integrity["ok"], integrity)
        self.assertTrue(integrity["image_bytes_identical"])
        self.assertTrue(integrity["label_bytes_identical"])
        self.assertTrue(integrity["negative_labels_empty"])
        for row in report["files"]:
            if row["kind"] == "negative":
                self.assertEqual(Path(row["label_path"]).stat().st_size, 0)

    def test_staging_never_mutates_the_frozen_source(self):
        before = {entry["tile_id"]: (entry["image_sha256"], entry["label_sha256"])
                  for entry in self.manifest["positive"] + self.manifest["negative"]}
        source_paths = [entry["image_path"]
                        for entry in self.manifest["positive"] + self.manifest["negative"]]
        source_paths += [entry["label_path"]
                         for entry in self.manifest["positive"] + self.manifest["negative"]]
        snapshot = {path: sha256_file(path) for path in source_paths}
        stage_dataset(self.manifest, self.tmp / "dataset", mode="hardlink")
        self.assertEqual({path: sha256_file(path) for path in source_paths}, snapshot)
        after = {entry["tile_id"]: (entry["image_sha256"], entry["label_sha256"])
                 for entry in self.manifest["positive"] + self.manifest["negative"]}
        self.assertEqual(before, after)

    def test_staging_falls_back_to_copy_and_still_matches(self):
        report = stage_dataset(self.manifest, self.tmp / "copy_dataset", mode="copy")
        self.assertTrue(verify_staging(report)["ok"])

    def test_unknown_staging_mode_is_refused(self):
        with self.assertRaises(Step2AError):
            stage_dataset(self.manifest, self.tmp / "bad", mode="move")

    def test_data_yaml_is_overfit_only_and_train_equals_val(self):
        dataset = self.tmp / "dataset"
        report = stage_dataset(self.manifest, dataset, mode="hardlink")
        text = Path(report["data_yaml"]).read_text(encoding="utf-8")
        self.assertIn(DATA_YAML_NOTE, text)
        self.assertIn(f"TRAIN_EQUALS_VAL = {str(TRAIN_EQUALS_VAL).lower()}", text)
        self.assertIn(f"path: {dataset.resolve()}", text)
        self.assertIn("train: images/train", text)
        self.assertIn("val: images/train", text)
        self.assertIn("0: ground_litter", text)

    def test_positive_labels_use_class_zero_only(self):
        report = stage_dataset(self.manifest, self.tmp / "dataset", mode="hardlink")
        for row in report["files"]:
            if row["kind"] != "positive":
                continue
            lines = Path(row["label_path"]).read_text(encoding="utf-8").strip().splitlines()
            self.assertTrue(lines, row["tile_id"])
            for line in lines:
                fields = line.split()
                self.assertEqual(len(fields), 5, line)
                self.assertEqual(int(fields[0]), 0, line)


class TestStep0BMatchingIsFrozen(unittest.TestCase):
    def test_matching_constants_are_the_frozen_step0b_values(self):
        self.assertEqual(IOU_NORMAL, 0.30)
        self.assertEqual(IOU_SMALL, 0.20)
        self.assertEqual(SMALL_GT_SHORT_SIDE_PX, 20.0)
        self.assertEqual(SMALL_EXPAND, 0.50)
        self.assertEqual(SMALL_AREA_RATIO_MIN, 0.25)
        self.assertEqual(SMALL_AREA_RATIO_MAX, 4.00)
        self.assertEqual(tuple(CONF_LEVELS), (0.01, 0.05, 0.10))

    def test_evaluate_predictions_publishes_the_same_constants(self):
        manifest = {"positive": [], "negative": []}
        metrics = evaluate_predictions(manifest, {})["matching"]
        self.assertEqual(metrics["iou_normal"], IOU_NORMAL)
        self.assertEqual(metrics["iou_small"], IOU_SMALL)
        self.assertEqual(metrics["small_gt_short_side_px"], SMALL_GT_SHORT_SIDE_PX)
        self.assertEqual(metrics["small_expand"], SMALL_EXPAND)
        self.assertEqual(metrics["small_area_ratio"],
                         [SMALL_AREA_RATIO_MIN, SMALL_AREA_RATIO_MAX])
        self.assertTrue(metrics["one_to_one"])

    def test_small_target_gets_the_relaxed_rule(self):
        gt = _box(100, 100, 8)
        shifted = [104.0, 104.0, 112.0, 112.0]                 # IoU 16/112 = 0.143
        self.assertLess(box_iou(shifted, gt), IOU_SMALL)
        self.assertTrue(is_eligible_match(shifted, gt))        # center + area ratio

    def test_normal_target_at_small_iou_is_rejected(self):
        gt = _box(0, 0, 40)
        partial = [0.0, 0.0, 10.0, 40.0]
        self.assertLess(box_iou(partial, gt), IOU_NORMAL)
        self.assertFalse(is_eligible_match(partial, gt))

    def test_small_target_with_area_ratio_outside_range_is_rejected(self):
        gt = _box(100, 100, 8)                                  # 64 px^2
        huge = [50.0, 50.0, 158.0, 158.0]                       # 108x108 = 11664 px^2
        self.assertLess(box_iou(huge, gt), IOU_SMALL)
        self.assertTrue(center_inside(huge, expand_box(gt, SMALL_EXPAND)))
        self.assertFalse(is_eligible_match(huge, gt))

    def test_matching_is_one_to_one(self):
        gt = [_box(0, 0, 40), _box(100, 100, 40)]
        predictions = [{"bbox": _box(0, 0, 40), "score": 0.9},
                       {"bbox": _box(100, 100, 40), "score": 0.8}]
        result = match_predictions(gt, predictions)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fn"], 0)
        self.assertEqual(len(set(result["matched_gt"].values())), 2)

    def test_ties_break_on_higher_score(self):
        gt = [_box(0, 0, 40)]
        predictions = [{"bbox": _box(0, 0, 40), "score": 0.2},
                       {"bbox": _box(0, 0, 40), "score": 0.7}]
        result = match_predictions(gt, predictions)
        self.assertEqual(result["matched_gt"][0], 1)
        self.assertEqual(result["unmatched_predictions"], [0])


class TestEvaluationAndVerdict(unittest.TestCase):
    def _manifest(self):
        positive = [{"tile_id": "pt-1", "boxes": [_box(0, 0, 40)],
                     "image_path": "x", "label_path": "y"}]
        negative = [{"tile_id": "nt-1", "image_path": "x", "label_path": "y"}]
        return {"positive": positive, "negative": negative}

    def test_negative_predictions_are_counted_as_false_positives(self):
        predictions = {"pt-1": [{"bbox": _box(0, 0, 40), "score": 0.42}],
                       "nt-1": [{"bbox": _box(4, 4, 20), "score": 0.06}]}
        metrics = evaluate_predictions(self._manifest(), predictions)
        self.assertEqual(metrics["per_conf"]["0.01"]["negative_fp_images"], 1)
        self.assertEqual(metrics["per_conf"]["0.01"]["negative_total_predictions"], 1)
        self.assertEqual(metrics["per_conf"]["0.05"]["negative_fp_images"], 1)
        self.assertEqual(metrics["per_conf"]["0.10"]["negative_fp_images"], 0)
        self.assertEqual(metrics["per_conf"]["0.01"]["positive_gt_proposal_recall"], 1.0)
        self.assertEqual(metrics["per_conf"]["0.01"]["positive_image_hit_rate"], 1.0)

    def test_conf_levels_are_reported_verbatim(self):
        metrics = evaluate_predictions(self._manifest(), {})
        self.assertEqual(list(metrics["per_conf"]), ["0.01", "0.05", "0.10"])
        self.assertEqual(metrics["conf_levels"], list(CONF_LEVELS))

    def test_overfit_verdict_thresholds_and_preconditions(self):
        def post(recall, hit):
            return {"per_conf": {"0.01": {"positive_gt_proposal_recall": recall,
                                          "positive_image_hit_rate": hit}}}

        self.assertEqual(overfit_verdict(post(0.95, 0.95), loader_ok=True,
                                         training_ok=True, nan_free=True,
                                         loss_decreased=True)["verdict"],
                         "TRAINING_PIPELINE_PASS")
        self.assertEqual(overfit_verdict(post(0.75, 0.5), loader_ok=True,
                                         training_ok=True, nan_free=True,
                                         loss_decreased=True)["verdict"],
                         "PARTIAL_OVERFIT")
        self.assertEqual(overfit_verdict(post(0.4, 0.2), loader_ok=True,
                                         training_ok=True, nan_free=True,
                                         loss_decreased=True)["verdict"],
                         "OVERFIT_FAIL")
        self.assertEqual(overfit_verdict(post(0.95, 0.95), loader_ok=False,
                                         training_ok=True, nan_free=True,
                                         loss_decreased=True)["verdict"],
                         "OVERFIT_FAIL")

    def test_boundary_flags_are_all_false(self):
        flags = boundary_flags()
        self.assertTrue(flags)
        self.assertTrue(all(value is False for value in flags.values()), flags)
        for key in ("development_accessed", "sealed_accessed",
                    "upstream_dataset_bytes_modified", "pretrained_weight_modified",
                    "threshold_tuned", "step2b_started"):
            self.assertIn(key, flags)


class TestSealedAndDevelopmentAreRefused(unittest.TestCase):
    def test_sealed_markers_raise(self):
        for marker in SEALED_MARKERS:
            with self.assertRaises(SealedAssetError):
                assert_not_sealed(Path("/tmp") / marker / "images")

    def test_load_pools_refuses_a_sealed_root(self):
        with self.assertRaises(SealedAssetError):
            load_pools(Path("/tmp") / SEALED_MARKERS[0], Path("/tmp") / SEALED_MARKERS[1])

    def test_step2a_module_is_stdlib_only(self):
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])
        forbidden = {"numpy", "cv2", "torch", "ultralytics", "yaml", "PIL",
                     "pandas", "matplotlib"}
        self.assertEqual(imported & forbidden, set())

    def _string_paths(self, payload, found=None):
        """Collect filesystem-looking strings (prose notes are not paths)."""
        if found is None:
            found = []
        if isinstance(payload, dict):
            for value in payload.values():
                self._string_paths(value, found)
        elif isinstance(payload, list):
            for value in payload:
                self._string_paths(value, found)
        elif isinstance(payload, str) and (payload.startswith("/")
                                           or payload.startswith("output/")):
            found.append(payload)
        return found

    def test_recorded_artifacts_never_point_at_sealed_or_development_assets(self):
        for name in ("preflight.json", "environment.json", "tiny_overfit_manifest.json",
                     "staging_integrity.json", "loader_sanity.json", "SUMMARY.json",
                     "MANIFEST.json"):
            path = STEP2A_OUT / name
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for value in self._string_paths(payload):
                lowered = value.lower()
                for marker in SEALED_MARKERS:
                    self.assertNotIn(marker.lower(), lowered, f"{name}: {value}")
                self.assertNotIn("/sealed", lowered, f"{name}: {value}")
                self.assertNotIn("development", lowered, f"{name}: {value}")


class TestBaselineAndPostShareOneSubset(unittest.TestCase):
    def test_cli_routes_baseline_and_post_through_one_frozen_manifest(self):
        source = CLI.read_text(encoding="utf-8")
        self.assertIn('def _tiny_manifest(args: argparse.Namespace) -> dict:', source)
        self.assertIn('manifest_path = args.output / MANIFEST_NAME', source)
        predict = source.split("def _predict(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("manifest = _tiny_manifest(args)", predict)
        baseline = source.split("def cmd_baseline(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_predict(args, args.weight, \"baseline\"", baseline)
        post = source.split("def cmd_post(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_predict(args, best, \"post\"", post)
        # the frozen manifest is reused if it exists, never re-selected per command
        self.assertIn("if manifest_path.is_file():", source)
        self.assertIn("manifest_path = args.output / MANIFEST_NAME", source)

    def test_frozen_manifest_is_checked_before_the_pools(self):
        """The CUDA host holds only the tiny dataset; it must not need the full pools."""
        source = CLI.read_text(encoding="utf-8")
        tiny = source.split("def _tiny_manifest(", 1)[1].split("\ndef ", 1)[0]
        self.assertLess(tiny.index("manifest_path.is_file()"), tiny.index("_pools(args)"))
        self.assertIn("def _manifest_provenance(", source)
        for section in ("cmd_train(", "_predict("):
            body = source.split(f"def {section}", 1)[1].split("\ndef ", 1)[0]
            self.assertIn("_manifest_provenance(args)", body, section)
        loader = source.split("def cmd_loader_sanity(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('"manifest_provenance": _manifest_provenance(args)', loader)

    def test_recorded_baseline_and_post_cover_the_same_tiles(self):
        manifest_path = STEP2A_OUT / "tiny_overfit_manifest.json"
        baseline_path = STEP2A_OUT / "baseline_metrics.json"
        if not (manifest_path.is_file() and baseline_path.is_file()):
            self.skipTest("Step 2A baseline evidence is not present")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = set(manifest["positive_tile_ids"] + manifest["negative_tile_ids"])
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        self.assertEqual(set(baseline["predictions"]), expected)
        for conf in CONF_LEVELS:
            key = f"{conf:.2f}"
            detail = baseline["metrics"]["per_conf"][key]
            self.assertEqual({row["tile_id"] for row in detail["per_positive_image"]},
                             set(manifest["positive_tile_ids"]))
            self.assertEqual({row["tile_id"] for row in detail["per_negative_image"]},
                             set(manifest["negative_tile_ids"]))
        post_path = STEP2A_OUT / "post_metrics.json"
        if post_path.is_file():
            post = json.loads(post_path.read_text(encoding="utf-8"))
            self.assertEqual(set(post["predictions"]), expected)


class TestTrainingNeverOverwritesThePretrainedWeight(unittest.TestCase):
    def test_train_writes_only_under_the_output_runs_directory(self):
        source = CLI.read_text(encoding="utf-8")
        train = source.split("def cmd_train(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('runs = args.output / "runs"', train)
        self.assertIn('"project": str(runs)', train)
        self.assertIn('"name": "tiny_overfit"', train)
        self.assertIn('"exist_ok": True', train)
        self.assertNotIn("args.weight.write", train)
        self.assertNotIn("shutil.copy", train)
        self.assertNotIn("unlink", train)
        # the frozen weight is only ever handed to the YOLO constructor
        self.assertEqual(train.count("args.weight"), 1)

    def test_train_refuses_to_silently_use_a_non_cuda_device(self):
        source = CLI.read_text(encoding="utf-8")
        self.assertIn("def _required_device(", source)
        require = source.split("def _required_device(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("--allow-non-cuda", require)
        self.assertIn("no CUDA GPU", source)
        self.assertIn('raise Step2AError', require)

    def test_minimal_augmentation_and_determinism(self):
        source = CLI.read_text(encoding="utf-8")
        train = source.split("def cmd_train(", 1)[1].split("\ndef ", 1)[0]
        for key in ('"mosaic": 0.0', '"mixup": 0.0', '"copy_paste": 0.0',
                    '"close_mosaic": 0', '"degrees": 0.0', '"fliplr": 0.0',
                    '"hsv_h": 0.0', '"erasing": 0.0', f'"seed": SEED',
                    '"deterministic": True'):
            self.assertIn(key, train, key)
        loader = source.split("def _build_loader(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("cfg.augment = False", loader)
        self.assertIn("cfg.mosaic = 0.0", loader)

    def test_train_arg_filter_uses_the_ultralytics_config_not_the_wrapper_signature(self):
        """YOLO.train is (self, trainer, **kwargs); its signature cannot filter args."""
        source = CLI.read_text(encoding="utf-8")
        train = source.split("def cmd_train(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("from ultralytics.cfg import get_cfg", train)
        self.assertIn("set(vars(get_cfg()).keys())", train)
        self.assertIn("missing_required", train)
        self.assertIn("would silently drop frozen", train)
        self.assertNotIn("supported = set(inspect.signature(model.train).parameters)",
                         train)

    def test_loader_sanity_uses_the_real_ultralytics_dataset_and_dataloader(self):
        source = CLI.read_text(encoding="utf-8")
        loader = source.split("def _build_loader(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("from ultralytics.data.build import build_yolo_dataset, build_dataloader",
                      loader)
        self.assertNotIn("class ", loader)

    @unittest.skipUnless(WEIGHT.is_file(), "pretrained yolo26s.pt is not present")
    def test_pretrained_weight_is_untouched(self):
        self.assertEqual(WEIGHT.stat().st_size, WEIGHT_BYTES)
        self.assertEqual(sha256_file(WEIGHT), WEIGHT_SHA256)

    def test_recorded_preflight_weight_hash_matches_the_file(self):
        path = STEP2A_OUT / "preflight.json"
        if not (path.is_file() and WEIGHT.is_file()):
            self.skipTest("Step 2A preflight evidence is not present")
        preflight = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(preflight["weight"]["sha256"], sha256_file(WEIGHT))


@requires_real_pools
class TestRealFrozenPools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pools = load_pools(POSITIVE_ROOT, NEGATIVE_ROOT)
        cls.preflight = verify_preflight(cls.pools, WEIGHT)

    def test_counts_match_the_frozen_expectations(self):
        counts = self.preflight["counts"]
        self.assertEqual(counts["positive_images"], POSITIVE_EXPECTED["images"])
        self.assertEqual(counts["positive_boxes"], POSITIVE_EXPECTED["boxes"])
        self.assertEqual(counts["negative_images"], NEGATIVE_EXPECTED["images"])
        self.assertEqual(counts["negative_boxes"], NEGATIVE_EXPECTED["boxes"])
        self.assertTrue(self.preflight["counts_match"])
        self.assertEqual(self.preflight["problem_count"], 0)
        self.assertEqual(self.preflight["image_sha_conflict_count"], 0)
        self.assertEqual(self.preflight["source_crop_conflict_count"], 0)

    def test_every_negative_label_is_a_zero_byte_file(self):
        self.assertTrue(self.preflight["negative_label_all_empty"])
        self.assertEqual(self.preflight["negative_labels_not_empty"], [])
        for row in self.pools["negatives"]:
            self.assertEqual(row["label_bytes"], 0)
            self.assertEqual(row["box_count"], 0)
            self.assertEqual(Path(row["label_path"]).read_bytes(), b"")

    def test_positive_labels_are_non_empty_class_zero(self):
        for row in self.pools["positives"]:
            self.assertGreater(row["box_count"], 0)
            lines = Path(row["label_path"]).read_text(encoding="utf-8").split()
            self.assertEqual(set(lines[0::5]), {"0"})

    def test_tiny_subset_is_reproducible_from_the_frozen_pools(self):
        manifest = subset_manifest(self.pools, select_subsets(self.pools))
        recorded = STEP2A_OUT / "tiny_overfit_manifest.json"
        if not recorded.is_file():
            self.skipTest("recorded Step 2A tiny manifest is not present")
        self.assertEqual(manifest["positive_tile_ids"],
                         json.loads(recorded.read_text(encoding="utf-8"))["positive_tile_ids"])
        self.assertEqual(manifest["negative_tile_ids"],
                         json.loads(recorded.read_text(encoding="utf-8"))["negative_tile_ids"])
        self.assertEqual(len(manifest["positive"]), TINY_POSITIVE_TARGET)
        self.assertEqual(len(manifest["negative"]), TINY_NEGATIVE_TARGET)

    def test_real_subset_keeps_the_small_target_buckets_and_five_cameras(self):
        manifest = subset_manifest(self.pools, select_subsets(self.pools))
        buckets = manifest["coverage"]["short_side_buckets"]
        self.assertTrue(any(key == "<10" for key in buckets), buckets)
        self.assertTrue(any(key == "10-19" for key in buckets), buckets)
        self.assertTrue(any(key == "20-39" for key in buckets), buckets)
        self.assertTrue(manifest["coverage"]["multi_label_tiles"] >= 1)
        self.assertEqual(len(manifest["coverage"]["positive_cameras"]), 5)
        self.assertEqual(len(manifest["coverage"]["negative_cameras"]), 5)

    def test_staging_the_real_subset_changes_no_frozen_bytes(self):
        manifest = subset_manifest(self.pools, select_subsets(self.pools))
        tracked = [entry["image_path"] for entry in manifest["positive"] + manifest["negative"]]
        tracked += [entry["label_path"] for entry in manifest["positive"]
                    + manifest["negative"]]
        tracked += [POSITIVE_ROOT / "positive_training_manifest_v2.jsonl",
                    POSITIVE_ROOT / "SUMMARY.json", POSITIVE_ROOT / "MANIFEST.json",
                    NEGATIVE_ROOT / "hard_negative_training_manifest.jsonl",
                    NEGATIVE_ROOT / "SUMMARY.json", NEGATIVE_ROOT / "MANIFEST.json"]
        snapshot = {path: sha256_file(path) for path in tracked}
        with tempfile.TemporaryDirectory() as tmp:
            report = stage_dataset(manifest, Path(tmp) / "dataset", mode="hardlink")
            self.assertTrue(verify_staging(report)["ok"])
        self.assertEqual({path: sha256_file(path) for path in tracked}, snapshot)

    def test_recorded_loader_sanity_is_consistent_with_the_subset(self):
        path = STEP2A_OUT / "loader_sanity.json"
        if not path.is_file():
            self.skipTest("recorded Step 2A loader sanity is not present")
        loader = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((STEP2A_OUT / "tiny_overfit_manifest.json")
                              .read_text(encoding="utf-8"))
        self.assertTrue(loader["ok"], loader.get("positive_with_zero_boxes"))
        self.assertEqual(loader["dataset_length"],
                         len(manifest["positive"]) + len(manifest["negative"]))
        self.assertEqual(loader["positive_samples"], len(manifest["positive"]))
        self.assertEqual(loader["negative_samples"], len(manifest["negative"]))
        self.assertEqual(loader["unknown_samples"], 0)
        self.assertEqual(loader["positive_with_zero_boxes"], [])
        self.assertEqual(loader["negative_with_boxes"], [])
        self.assertEqual(loader["positive_box_total"],
                         sum(entry["box_count"] for entry in manifest["positive"]))
        self.assertEqual(loader["negative_box_total"], 0)
        self.assertTrue(loader["negative_label_bytes_zero"])
        self.assertGreaterEqual(loader["positive_min_boxes"], 1)
        self.assertTrue(all(batch["class_ids"] == [0] for batch in loader["mixed_batches"]
                            if batch["total_boxes"] > 0))
        for row in loader["per_sample"]:
            self.assertEqual(row["raw_cls_shape"][0], row["raw_label_boxes"])
        self.assertGreaterEqual(len(loader["mixed_batches"]), 2)

    def test_recorded_manifest_records_frozen_matching_and_boundaries(self):
        path = STEP2A_OUT / "MANIFEST.json"
        if not path.is_file():
            self.skipTest("recorded Step 2A MANIFEST is not present")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        matching = manifest["matching_parameters"]
        self.assertEqual(matching["iou_normal"], IOU_NORMAL)
        self.assertEqual(matching["iou_small"], IOU_SMALL)
        self.assertEqual(matching["small_gt_short_side_px"], SMALL_GT_SHORT_SIDE_PX)
        self.assertEqual(matching["small_expand"], SMALL_EXPAND)
        self.assertEqual(matching["small_area_ratio"],
                         [SMALL_AREA_RATIO_MIN, SMALL_AREA_RATIO_MAX])
        self.assertTrue(all(value is False for value in manifest["boundaries"].values()))
        self.assertEqual(manifest["counts"]["positive_images"],
                         POSITIVE_EXPECTED["images"])
        self.assertEqual(manifest["counts"]["negative_images"],
                         NEGATIVE_EXPECTED["images"])

    def test_recorded_summary_reports_the_environment_honestly(self):
        path = STEP2A_OUT / "SUMMARY.json"
        environment_path = STEP2A_OUT / "environment.json"
        if not (path.is_file() and environment_path.is_file()):
            self.skipTest("recorded Step 2A summary is not present")
        summary = json.loads(path.read_text(encoding="utf-8"))
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["seed"], SEED)
        self.assertTrue(all(value is False for value in summary["boundaries"].values()))
        verdict = summary["verdict"]["verdict"]
        if environment.get("cuda_available"):
            self.assertEqual(environment["blockers"], [])
            self.assertTrue(environment["cuda_device_name"])
            self.assertTrue(environment["cuda_vram_bytes"])
            self.assertIn(verdict, ("TRAINING_PIPELINE_PASS", "PARTIAL_OVERFIT",
                                    "OVERFIT_FAIL"))
        else:
            self.assertTrue(environment["blockers"])
            self.assertIn("no CUDA GPU", environment["blockers"][0])
            self.assertEqual(verdict, "NOT_EVALUATED")


class TestRecordedCudaRun(unittest.TestCase):
    """The authorised CUDA tiny overfit, checked against the recorded evidence."""

    @classmethod
    def setUpClass(cls):
        path = STEP2A_OUT / "SUMMARY.json"
        if not path.is_file():
            raise unittest.SkipTest("recorded Step 2A summary is not present")
        cls.summary = json.loads(path.read_text(encoding="utf-8"))
        cls.training = cls.summary.get("training") or {}
        cls.post = cls.summary.get("post") or {}
        cls.server = cls.summary.get("server_run") or {}
        cls.environment = cls.summary.get("environment") or {}
        if not cls.training:
            raise unittest.SkipTest("the recorded Step 2A summary has no tiny overfit run")
        manifest_path = STEP2A_OUT / "tiny_overfit_manifest.json"
        cls.manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                        if manifest_path.is_file() else {"positive": [], "negative": []})

    def test_training_ran_on_cuda_with_the_frozen_manifest(self):
        self.assertTrue(self.environment.get("cuda_available"))
        self.assertEqual(self.environment["blockers"], [])
        provenance = self.training["manifest_provenance"]
        self.assertEqual(provenance["source"], "frozen_file")
        self.assertTrue(provenance["reused_frozen_manifest"])
        manifest_path = STEP2A_OUT / "tiny_overfit_manifest.json"
        if manifest_path.is_file():
            self.assertEqual(provenance["sha256"], sha256_file(manifest_path))

    def test_training_parameters_are_the_frozen_ones(self):
        applied = self.training["applied_args"]
        self.assertEqual(applied["imgsz"], 640)
        self.assertEqual(applied["seed"], SEED)
        self.assertEqual(applied["device"], "cuda")
        self.assertIs(applied["deterministic"], True)
        for key in ("mosaic", "mixup", "copy_paste", "degrees", "translate", "scale",
                    "shear", "perspective", "flipud", "fliplr", "hsv_h", "hsv_s",
                    "hsv_v", "erasing"):
            self.assertEqual(applied[key], 0.0, key)
        self.assertEqual(applied["close_mosaic"], 0)
        self.assertIn(self.training["batch"], (8, 4, 2))
        self.assertLessEqual(self.training["epochs_requested"], 100)
        self.assertEqual(self.training["unsupported_args_skipped"], [])
        for key in ("data", "epochs", "batch", "optimizer", "project", "name"):
            self.assertIn(key, applied, key)

    def test_training_actually_iterated_was_finite_and_improved(self):
        self.assertGreaterEqual(self.training["epochs_run"], 1)
        self.assertLessEqual(self.training["epochs_run"],
                             self.training["epochs_requested"])
        self.assertIs(self.training["nan_or_inf"], False)
        self.assertIs(self.training["loss_decreased"], True)
        self.assertLess(self.training["final_box_loss"], self.training["initial_box_loss"])
        rows = self.training["epoch_metrics"]
        self.assertEqual(len(rows), self.training["epochs_run"])
        losses = [float(row["train/box_loss"]) for row in rows]
        self.assertTrue(all(value == value and abs(value) != float("inf")
                            for value in losses))

    def test_checkpoints_are_new_artifacts_not_the_pretrained_weight(self):
        checkpoints = self.server.get("checkpoint_hashes")
        if not checkpoints:
            self.skipTest("checkpoint hashes are not recorded")
        shas = {name: checkpoints[name]["sha256"] for name in ("best.pt", "last.pt")}
        for name, digest in shas.items():
            self.assertNotEqual(digest, WEIGHT_SHA256, name)
            self.assertGreater(checkpoints[name]["bytes"], 0)
        self.assertNotEqual(shas["best.pt"], shas["last.pt"])

    def test_frozen_input_verification_is_recorded_ok(self):
        verification = self.server.get("frozen_input_verification")
        if not verification:
            self.skipTest("frozen input verification is not recorded")
        self.assertTrue(verification["ok"], verification)
        self.assertTrue(verification["weight_matches_local_frozen"])
        self.assertEqual(verification["weight_sha256"], WEIGHT_SHA256)
        self.assertTrue(verification["all_negative_labels_zero_bytes"])
        self.assertTrue(verification["positive_labels_class_zero_only"])
        self.assertEqual(verification["frozen_inputs_mismatch"], [])
        self.assertEqual(verification["positive_tiles"], TINY_POSITIVE_TARGET)
        self.assertEqual(verification["negative_tiles"], TINY_NEGATIVE_TARGET)
        self.assertEqual(verification["positive_boxes"], 41)

    def test_uploaded_files_are_sha256_recorded(self):
        upload = self.server.get("upload_sha256sums")
        if not upload:
            self.skipTest("upload manifest is not recorded")
        files = upload["files"]
        self.assertEqual(upload["file_count"], len(files))
        self.assertEqual(files["assets/yolo26s.pt"], WEIGHT_SHA256)
        self.assertIn("code/scripts/run_ground_litter_step2a.py", files)
        self.assertIn("out/tiny_overfit_manifest.json", files)
        self.assertEqual(upload["archive_sha256"], upload["archive_received_sha256"])

    def test_derived_data_yaml_only_changed_the_path_line(self):
        record = self.server.get("data_yaml_record")
        if not record:
            self.skipTest("data.yaml record is not recorded")
        self.assertTrue(record["only_path_line_changed"], record["diff_lines"])
        old, new = record["diff_lines"]
        self.assertTrue(old.startswith("path: "))
        self.assertTrue(new.startswith("path: "))
        self.assertNotEqual(old, new)

    def test_post_uses_the_same_subset_matching_and_confidence_levels(self):
        self.assertEqual(self.post["conf_floor"], min(CONF_LEVELS))
        self.assertEqual(self.post["imgsz"], 640)
        metrics = self.post["metrics"]
        self.assertEqual(metrics["conf_levels"], list(CONF_LEVELS))
        self.assertEqual(list(metrics["per_conf"]), ["0.01", "0.05", "0.10"])
        matching = metrics["matching"]
        self.assertEqual(matching["iou_normal"], IOU_NORMAL)
        self.assertEqual(matching["iou_small"], IOU_SMALL)
        self.assertEqual(matching["small_gt_short_side_px"], SMALL_GT_SHORT_SIDE_PX)
        self.assertEqual(matching["small_expand"], SMALL_EXPAND)
        self.assertEqual(matching["small_area_ratio"],
                         [SMALL_AREA_RATIO_MIN, SMALL_AREA_RATIO_MAX])
        positive_ids = set(self.manifest["positive_tile_ids"])
        negative_ids = set(self.manifest["negative_tile_ids"])
        for conf in ("0.01", "0.05", "0.10"):
            detail = metrics["per_conf"][conf]
            self.assertEqual({row["tile_id"] for row in detail["per_positive_image"]},
                             positive_ids)
            self.assertEqual({row["tile_id"] for row in detail["per_negative_image"]},
                             negative_ids)
            buckets = detail["by_size_bucket"]
            self.assertEqual(sum(value["gt"] for value in buckets.values()), 41)
            groups = detail["by_label_count"]
            self.assertIn("single", groups)
            self.assertIn("multi", groups)
            self.assertEqual(sum(value["gt"] for value in groups.values()), 41)
            for value in list(buckets.values()) + list(groups.values()):
                self.assertLessEqual(value["tp"], value["gt"])
        self.assertEqual(self.post["verdict"]["verdict"],
                         self.summary["verdict"]["verdict"])
        self.assertEqual(set(self.post["baseline_comparison"]), {"0.01", "0.05", "0.10"})


class TestRealLoaderEndToEnd(unittest.TestCase):
    @unittest.skipUnless(REAL_POOLS and _has_ultralytics(),
                         "real Ultralytics stack is not available in this interpreter")
    def test_real_loader_sanity_runs_on_a_fresh_staged_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "step2a"
            for command in ("stage", "loader-sanity"):
                proc = subprocess.run(
                    [sys.executable, str(CLI), "--output", str(out), command],
                    capture_output=True, text=True, timeout=900)
                self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            report = json.loads((out / "loader_sanity.json").read_text(encoding="utf-8"))
            self.assertTrue(report["ok"], report.get("positive_with_zero_boxes"))
            self.assertEqual(report["negative_box_total"], 0)
            self.assertTrue(report["negative_label_bytes_zero"])


class TestReportWriting(SyntheticPoolCase):
    def test_write_reports_links_the_manifest_to_the_written_files(self):
        preflight = verify_preflight(self.pools, None)
        environment = {"platform": "test", "cuda_available": False,
                       "blockers": ["no CUDA GPU"]}
        staging = stage_dataset(self.manifest, self.tmp / "dataset", mode="hardlink")
        loader = {"ok": True}
        summary = build_summary(self.pools, preflight, environment, self.manifest,
                                staging, loader)
        payload = build_manifest(self.pools, summary, code_commit="0" * 40,
                                 generated_at="2026-09-23T00:00:00Z",
                                 config={"seed": SEED}, artifact_root=self.tmp,
                                 provenance={"step1c2m_evidence_commit": "8457ad4"})
        paths = write_reports(self.tmp, tiny_manifest=self.manifest, loader=loader,
                              summary=summary, manifest=payload)
        written = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(written["summary_sha256"], sha256_file(paths["summary"]))
        self.assertEqual(written["tiny_overfit_manifest_sha256"],
                         sha256_file(paths["tiny_overfit_manifest"]))
        self.assertEqual(written["matching_parameters"]["iou_normal"], IOU_NORMAL)
        self.assertEqual(written["boundaries"], boundary_flags())
        self.assertEqual(summary["tiny_dataset"]["positive_tiles"],
                         len(self.manifest["positive"]))
        self.assertEqual(summary["tiny_dataset"]["negative_tiles"],
                         len(self.manifest["negative"]))

    def test_summary_without_a_run_reports_not_evaluated(self):
        preflight = verify_preflight(self.pools, None)
        staging = stage_dataset(self.manifest, self.tmp / "dataset", mode="hardlink")
        summary = build_summary(self.pools, preflight, {"cuda_available": False},
                                self.manifest, staging, None)
        self.assertEqual(summary["loader_sanity"], {"status": "not_run"})
        self.assertEqual(summary["verdict"], {"verdict": "NOT_EVALUATED"})


if __name__ == "__main__":
    unittest.main()
