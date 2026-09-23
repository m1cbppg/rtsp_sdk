"""Step 2B: full fine-tune on the frozen pool + train-set sanity + checkpoint freeze.

Pure-stdlib logic plus the real frozen pool when it is present; no torch/ultralytics is
needed here.  The recorded CUDA evidence is re-checked only when those artifacts exist.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rtsp_annotator.ground_litter_localization_review import (
    SEALED_MARKERS,
    sha256_file,
)
from rtsp_annotator.ground_litter_step2a import (
    CONF_LEVELS,
    IOU_NORMAL,
    IOU_SMALL,
    SEED,
    SMALL_AREA_RATIO_MAX,
    SMALL_AREA_RATIO_MIN,
    SMALL_EXPAND,
    SMALL_GT_SHORT_SIDE_PX,
    load_pools,
    verify_preflight,
)
from rtsp_annotator.ground_litter_step2b import (
    DATA_YAML_NOTE,
    DISABLED_AUGMENTATION,
    EXPECTED,
    FULL_TRAIN_DATASET,
    FULL_TRAIN_MANIFEST,
    LIGHT_AUGMENTATION,
    MAX_EASY_NEGATIVE_FRACTION,
    REQUIRED_ARGS,
    TRAIN_RECIPE,
    TRAIN_SET_NOTE,
    TRAIN_SET_RECALL_BAR,
    VERDICT_COMPLETE,
    VERDICT_FAILED,
    VERDICT_INCOMPLETE,
    Step2BError,
    boundary_flags,
    checkpoint_freeze_record,
    training_completion_verdict,
    full_pool_manifest,
    size_bucket_names,
    verify_full_pool,
)

ROOT = Path(__file__).resolve().parents[1]
POSITIVE_ROOT = ROOT / "output" / "ground_litter_positive_tile_completion_20260923"
NEGATIVE_ROOT = ROOT / "output" / "ground_litter_hard_negatives_20260923"
WEIGHT = ROOT / "models" / "yolo26s.pt"
STEP2B_OUT = ROOT / "output" / "ground_litter_step2b_20260923"
CLI = ROOT / "scripts" / "run_ground_litter_step2b.py"
STEP2A_CLI = ROOT / "scripts" / "run_ground_litter_step2a.py"
MODULE = ROOT / "rtsp_annotator" / "ground_litter_step2b.py"

WEIGHT_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
#: the frozen Step 2C handoff is the last epoch; best.pt is diagnostic only
PRIMARY_SHA256 = "4852392aeae9a68669a50752eb1f7466fbcc9a426524faba86fb20a3b6351a94"
DIAGNOSTIC_SHA256 = "e3e1192bcf1dece87d31f2e282ff59630c93359f2b4539bf05f269d810e6012c"
SUPERSEDED_VERDICT = "FULL_FINETUNE_SANITY_PASS"

#: The operator-frozen Step 2B recipe, spelled out independently of the module.
FROZEN_RECIPE = {
    "hsv_h": 0.015, "hsv_s": 0.30, "hsv_v": 0.20,
    "translate": 0.02, "scale": 0.05, "fliplr": 0.5,
    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "close_mosaic": 0,
    "perspective": 0.0, "degrees": 0.0, "shear": 0.0, "flipud": 0.0, "erasing": 0.0,
}


def _has_real_pools() -> bool:
    return ((POSITIVE_ROOT / "positive_training_manifest_v2.jsonl").is_file()
            and (NEGATIVE_ROOT / "hard_negative_training_manifest.jsonl").is_file()
            and (POSITIVE_ROOT / "accepted_v2" / "images").is_dir()
            and (NEGATIVE_ROOT / "accepted" / "images").is_dir())


REAL_POOLS = _has_real_pools()
requires_real_pools = unittest.skipUnless(
    REAL_POOLS, "frozen Step 1C-2M / 1D training pools are not present on disk")


class TestFrozenRecipe(unittest.TestCase):
    def test_recipe_matches_the_operator_values_exactly(self):
        for key, value in FROZEN_RECIPE.items():
            self.assertEqual(TRAIN_RECIPE[key], value, key)

    def test_fixed_training_parameters(self):
        self.assertEqual(TRAIN_RECIPE["imgsz"], 640)
        self.assertEqual(TRAIN_RECIPE["epochs"], 100)
        self.assertEqual(TRAIN_RECIPE["batch"], 8)
        self.assertEqual(TRAIN_RECIPE["seed"], SEED)
        self.assertIs(TRAIN_RECIPE["deterministic"], True)
        self.assertIs(TRAIN_RECIPE["val"], True)
        self.assertIs(TRAIN_RECIPE["pretrained"], True)
        self.assertEqual(TRAIN_RECIPE["optimizer"], "auto")

    def test_light_and_disabled_augmentation_sets_are_disjoint_and_covered(self):
        self.assertEqual(set(LIGHT_AUGMENTATION) & set(DISABLED_AUGMENTATION), set())
        for key in LIGHT_AUGMENTATION + DISABLED_AUGMENTATION:
            self.assertIn(key, TRAIN_RECIPE, key)
        # no strong mosaic / perspective anywhere
        self.assertEqual(TRAIN_RECIPE["mosaic"], 0.0)
        self.assertEqual(TRAIN_RECIPE["perspective"], 0.0)

    def test_train_set_recall_bar_is_preregistered(self):
        self.assertEqual(TRAIN_SET_RECALL_BAR, 0.90)
        self.assertIn("train", TRAIN_SET_NOTE.lower())
        self.assertEqual(DATA_YAML_NOTE, "FULL_TRAIN_SET_SANITY_ONLY")


@requires_real_pools
class TestFullPoolManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pools = load_pools(POSITIVE_ROOT, NEGATIVE_ROOT)
        cls.preflight = verify_preflight(cls.pools, WEIGHT)
        cls.manifest = full_pool_manifest(cls.pools)

    def test_manifest_is_the_whole_frozen_pool(self):
        self.assertEqual(len(self.manifest["positive"]), EXPECTED["positive_images"])
        self.assertEqual(len(self.manifest["negative"]), EXPECTED["negative_images"])
        self.assertEqual(sum(entry["box_count"] for entry in self.manifest["positive"]),
                         EXPECTED["positive_boxes"])
        self.assertEqual(self.manifest["positive_tile_ids"],
                         [entry["tile_id"] for entry in self.manifest["positive"]])
        self.assertEqual(self.manifest["negative_tile_ids"],
                         [entry["tile_id"] for entry in self.manifest["negative"]])
        self.assertEqual(len(set(self.manifest["positive_tile_ids"]
                                 + self.manifest["negative_tile_ids"])), 107)
        self.assertTrue(self.manifest["selection"].startswith("none"))

    def test_coverage_reports_cameras_hardness_and_recordings(self):
        coverage = self.manifest["coverage"]
        self.assertEqual(len(coverage["positive_cameras"]), 5)
        self.assertEqual(len(coverage["negative_cameras"]), 5)
        self.assertEqual(coverage["easy_negative_count"], EXPECTED["easy_negatives"])
        self.assertEqual(coverage["hard_negative_count"], EXPECTED["hard_negatives"])
        self.assertEqual(coverage["positive_source_recordings"], 33)
        self.assertEqual(coverage["negative_source_recordings"], 30)
        self.assertEqual(sum(coverage["per_camera_positive"].values()),
                         EXPECTED["positive_images"])
        self.assertEqual(sum(coverage["per_camera_negative"].values()),
                         EXPECTED["negative_images"])
        self.assertTrue(coverage["multi_label_tiles"] >= 1)
        self.assertTrue(coverage["short_side_buckets"])

    def test_every_entry_has_the_fields_the_shared_engine_needs(self):
        for entry in self.manifest["positive"]:
            for key in ("tile_id", "camera_id", "image_path", "label_path",
                        "image_sha256", "label_sha256", "box_count", "boxes"):
                self.assertIn(key, entry, key)
            self.assertGreaterEqual(entry["box_count"], 1)
            self.assertEqual(len(entry["boxes"]), entry["box_count"])
            self.assertEqual(len(entry["short_sides"]), entry["box_count"])
        for entry in self.manifest["negative"]:
            self.assertEqual(entry["box_count"], 0)
            self.assertEqual(entry["label_bytes"], 0)

    def test_manifest_hashes_are_the_frozen_pool_hashes(self):
        self.assertEqual(self.manifest["source_hashes"]["positive_training_manifest_sha256"],
                         self.pools["positive_manifest_sha256"])
        self.assertEqual(
            self.manifest["source_hashes"]["hard_negative_training_manifest_sha256"],
            self.pools["negative_manifest_sha256"])

    def test_full_pool_check_passes_on_the_frozen_pool(self):
        report = verify_full_pool(self.pools, self.manifest, self.preflight)
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["counts"], {key: EXPECTED[key] for key in report["counts"]})
        self.assertEqual(report["duplicate_tiles"], [])
        self.assertEqual(report["pool_tiles_missing"], [])
        self.assertEqual(report["negative_label_not_empty"], [])
        self.assertEqual(report["positive_without_boxes"], [])
        self.assertEqual(report["pool_tiles_total"], 107)
        self.assertLessEqual(report["easy_negative_fraction"],
                             MAX_EASY_NEGATIVE_FRACTION)

    def test_full_pool_check_detects_tampering(self):
        tampered = json.loads(json.dumps(self.manifest))
        tampered["negative"][0]["label_bytes"] = 12
        report = verify_full_pool(self.pools, tampered, self.preflight)
        self.assertFalse(report["ok"])
        self.assertIn("negative_label_not_empty",
                      [problem["field"] for problem in report["problems"]])

        duplicated = json.loads(json.dumps(self.manifest))
        duplicated["negative"][0]["tile_id"] = duplicated["positive"][0]["tile_id"]
        report = verify_full_pool(self.pools, duplicated, self.preflight)
        self.assertIn("duplicate_tiles", [p["field"] for p in report["problems"]])

        missing = json.loads(json.dumps(self.manifest))
        missing["negative"].pop()
        report = verify_full_pool(self.pools, missing, self.preflight)
        self.assertIn("pool_tiles_missing", [p["field"] for p in report["problems"]])

        bad_counts = json.loads(json.dumps(self.manifest))
        bad_counts["positive"].pop()
        report = verify_full_pool(self.pools, bad_counts, self.preflight)
        self.assertIn("count_mismatch", [p["field"] for p in report["problems"]])

        easy_heavy = json.loads(json.dumps(self.manifest))
        easy_heavy["coverage"]["easy_negative_count"] = len(easy_heavy["negative"])
        report = verify_full_pool(self.pools, easy_heavy, self.preflight)
        self.assertIn("easy_fraction_exceeded", [p["field"] for p in report["problems"]])

    def test_size_bucket_names_cover_every_positive_box(self):
        names = [bucket for entry in self.manifest["positive"]
                 for bucket in size_bucket_names(entry["boxes"])]
        self.assertEqual(len(names), EXPECTED["positive_boxes"])
        self.assertTrue(set(names) <= {"<10", "10-19", "20-39", "40-79", "80+"})


class TestTrainingCompletionVerdict(unittest.TestCase):
    """The verdict only says whether the full run completed; metrics never grade it."""

    def _training(self, **overrides):
        applied = {"data": "d", "imgsz": 640, "epochs": 100, "batch": 8, "device": "cuda",
                   "seed": SEED, "deterministic": True, "optimizer": "auto",
                   "project": "p", "name": "full_finetune"}
        applied.update({key: TRAIN_RECIPE[key]
                        for key in LIGHT_AUGMENTATION + DISABLED_AUGMENTATION})
        applied["close_mosaic"] = TRAIN_RECIPE["close_mosaic"]
        training = {"epochs_requested": 100, "epochs_run": 100,
                    "unsupported_args_skipped": [], "applied_args": applied,
                    "nan_or_inf": False, "loss_decreased": True,
                    "initial_box_loss": 1.6, "final_box_loss": 0.3, "batch": 8,
                    "imgsz": 640, "device": "cuda", "seed": SEED, "save_dir": "/tmp/run"}
        training.update(overrides)
        return training

    def _post(self, recall, hit):
        return {"metrics": {"per_conf": {"0.01": {
            "positive_gt_proposal_recall": recall, "positive_image_hit_rate": hit}}}}

    def test_completed_run_is_complete(self):
        verdict = training_completion_verdict(self._training(), self._post(0.95, 0.9),
                                              loader_ok=True, full_pool_ok=True)
        self.assertEqual(verdict["verdict"], VERDICT_COMPLETE)
        self.assertEqual(verdict["failed_preconditions"], [])
        self.assertEqual(verdict["hard_failures"], [])
        self.assertEqual(verdict["completion_failures"], [])

    def test_train_set_metrics_never_change_the_verdict(self):
        """A low train-set recall must not turn a completed run into a failure label."""
        for recall, hit in ((0.0, 0.0), (0.5, 0.4), (1.0, 1.0)):
            verdict = training_completion_verdict(self._training(),
                                                  self._post(recall, hit),
                                                  loader_ok=True, full_pool_ok=True)
            self.assertEqual(verdict["verdict"], VERDICT_COMPLETE, (recall, hit))
            self.assertEqual(verdict["train_set_gt_proposal_recall_at_0.01"], recall)
        self.assertIn("informational", verdict["train_set_recall_bar_role"])
        self.assertIn("completion", verdict["verdict_basis"])

    def test_hard_failures_are_failed(self):
        cases = {
            "loader_ok": dict(loader_ok=False),
            "full_pool_ok": dict(full_pool_ok=False),
            "frozen_args_applied": dict(),
            "augmentation_matches_recipe": dict(),
            "nan_free": dict(),
        }
        for name, kwargs in cases.items():
            training = self._training()
            if name == "frozen_args_applied":
                training["unsupported_args_skipped"] = ["some_arg"]
            elif name == "augmentation_matches_recipe":
                training["applied_args"]["mosaic"] = 0.5
            elif name == "nan_free":
                training["nan_or_inf"] = True
            call = {"loader_ok": True, "full_pool_ok": True}
            call.update(kwargs)
            verdict = training_completion_verdict(training, self._post(1.0, 1.0), **call)
            self.assertEqual(verdict["verdict"], VERDICT_FAILED, name)
            self.assertIn(name, verdict["hard_failures"], name)
            self.assertIn(name, verdict["failed_preconditions"], name)

    def test_short_or_non_decreasing_run_is_incomplete_not_failed(self):
        short = self._training(epochs_run=50)
        verdict = training_completion_verdict(short, self._post(1.0, 1.0),
                                              loader_ok=True, full_pool_ok=True)
        self.assertEqual(verdict["verdict"], VERDICT_INCOMPLETE)
        self.assertEqual(verdict["hard_failures"], [])
        self.assertIn("epochs_completed", verdict["completion_failures"])

        flat = self._training(loss_decreased=False)
        verdict = training_completion_verdict(flat, self._post(1.0, 1.0),
                                              loader_ok=True, full_pool_ok=True)
        self.assertEqual(verdict["verdict"], VERDICT_INCOMPLETE)
        self.assertIn("loss_decreased", verdict["completion_failures"])

    def test_missing_required_arg_is_a_hard_failure(self):
        training = self._training()
        training["applied_args"].pop("mosaic")
        verdict = training_completion_verdict(training, self._post(1.0, 1.0),
                                              loader_ok=True, full_pool_ok=True)
        self.assertEqual(verdict["verdict"], VERDICT_FAILED)

    def test_verdict_never_claims_generalisation_or_quality(self):
        verdict = training_completion_verdict(self._training(), self._post(1.0, 1.0),
                                              loader_ok=True, full_pool_ok=True)
        self.assertIn("train", verdict["note"].lower())
        payload = json.dumps(verdict).lower()
        for forbidden in ("sanity_pass", "passed", "development", "sealed"):
            self.assertNotIn(forbidden, payload)
        # the disclaimer must be present, not the claim
        self.assertIn("no generalisation is claimed", payload)


class TestCheckpointFreeze(unittest.TestCase):
    def _training(self):
        return {"epochs_run": 100, "epochs_requested": 100, "batch": 8, "imgsz": 640,
                "device": "cuda", "seed": SEED, "save_dir": "/tmp/run",
                "initial_box_loss": 1.6, "final_box_loss": 0.3, "nan_or_inf": False}

    def _checkpoints(self):
        return {
            "best.pt": {"path": "/tmp/run/weights/best.pt", "bytes": 123,
                        "sha256": "a" * 64},
            "last.pt": {"path": "/tmp/run/weights/last.pt", "bytes": 124,
                        "sha256": "b" * 64},
            "best_equals_last": False,
        }

    def test_primary_is_the_last_epoch_and_best_is_forbidden(self):
        record = checkpoint_freeze_record(self._training(), self._checkpoints(),
                                          frozen_at="2026-09-23T00:00:00Z",
                                          manifest_sha256="c" * 64)
        self.assertEqual(record["primary"], "last.pt")
        self.assertEqual(record["primary_epoch"], 100)
        self.assertEqual(record["primary_sha256"], "b" * 64)
        self.assertEqual(record["primary_bytes"], 124)
        self.assertEqual(record["frozen_at"], "2026-09-23T00:00:00Z")
        self.assertEqual(record["train_manifest_sha256"], "c" * 64)
        self.assertEqual(sorted(record["checkpoints"]), ["best.pt", "last.pt"])
        self.assertFalse(record["best_equals_last"])
        self.assertIn("last epoch", record["selection_rule"])
        self.assertIn("best.pt is excluded", record["selection_rule"])

    def test_best_checkpoint_is_diagnostic_only(self):
        record = checkpoint_freeze_record(self._training(), self._checkpoints())
        diagnostic = record["diagnostic"]
        self.assertEqual(diagnostic["name"], "best.pt")
        self.assertEqual(diagnostic["sha256"], "a" * 64)
        self.assertEqual(diagnostic["role"], "diagnostic only")
        self.assertEqual(diagnostic["forbidden_as"],
                         "Step 2C first-round official checkpoint")
        self.assertIn("training-set validation metric", diagnostic["reason"])
        handoff = record["step2c_handoff"]
        self.assertIn("best.pt", handoff["forbidden_checkpoints"])
        forbidden = handoff["forbidden_checkpoints"]["best.pt"]
        self.assertEqual(forbidden["sha256"], "a" * 64)
        self.assertEqual(forbidden["role"], "diagnostic only")
        self.assertIn("training-set validation metric", forbidden["reason"])

    def test_step2c_handoff_points_only_at_the_primary(self):
        record = checkpoint_freeze_record(self._training(), self._checkpoints())
        handoff = record["step2c_handoff"]
        self.assertEqual(handoff["weights"], "/tmp/run/weights/last.pt")
        self.assertEqual(handoff["weights_sha256"], "b" * 64)
        self.assertEqual(handoff["weights_epoch"], 100)
        self.assertEqual(handoff["conf_levels"], list(CONF_LEVELS))
        self.assertEqual(handoff["matching"]["iou_normal"], IOU_NORMAL)
        self.assertEqual(handoff["matching"]["iou_small"], IOU_SMALL)
        self.assertEqual(handoff["matching"]["small_gt_short_side_px"],
                         SMALL_GT_SHORT_SIDE_PX)
        self.assertEqual(handoff["matching"]["small_expand"], SMALL_EXPAND)
        self.assertEqual(handoff["matching"]["small_area_ratio"],
                         [SMALL_AREA_RATIO_MIN, SMALL_AREA_RATIO_MAX])
        self.assertFalse(handoff["development_accessed"])
        self.assertFalse(handoff["sealed_accessed"])
        self.assertIn("SHA-256", handoff["rule"])
        self.assertNotIn("best.pt", [handoff["weights"]])

    def test_missing_primary_is_refused(self):
        checkpoints = self._checkpoints()
        checkpoints.pop("last.pt")
        with self.assertRaises(Step2BError):
            checkpoint_freeze_record(self._training(), checkpoints)

    def test_recipe_and_training_are_recorded_with_the_freeze(self):
        record = checkpoint_freeze_record(self._training(), self._checkpoints())
        self.assertEqual(record["recipe"], TRAIN_RECIPE)
        self.assertEqual(record["training"]["epochs_run"], 100)
        self.assertIs(record["training"]["nan_or_inf"], False)


class TestBoundariesAndStdlib(unittest.TestCase):
    def test_boundary_flags_are_all_false(self):
        flags = boundary_flags()
        self.assertTrue(all(value is False for value in flags.values()), flags)
        for key in ("development_accessed", "sealed_accessed", "step2c_started",
                    "hyperparameters_tuned", "tiny_subset_resampled"):
            self.assertIn(key, flags)

    def test_module_is_stdlib_only(self):
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])
        forbidden = {"numpy", "cv2", "torch", "ultralytics", "yaml", "PIL"}
        self.assertEqual(imported & forbidden, set())


class TestStep2bCli(unittest.TestCase):
    def test_cli_reuses_the_step2a_engine_with_only_two_name_knobs(self):
        source = CLI.read_text(encoding="utf-8")
        self.assertIn('"run_ground_litter_step2a"', source)
        self.assertIn("module.MANIFEST_NAME = FULL_TRAIN_MANIFEST", source)
        self.assertIn("module.DATASET_DIR_NAME = FULL_TRAIN_DATASET", source)
        for reused in ("cmd_loader_sanity", "cmd_baseline", "_predict", "_environment",
                       "_required_device", "_manifest_provenance"):
            self.assertIn(f"STEP2A.{reused}", source, reused)
        # its own names, so Step 2A files are never written by this CLI
        self.assertNotIn('"tiny_overfit_manifest.json"', source)
        self.assertNotIn('"tiny_overfit_dataset"', source)

    def test_cli_names_are_distinct_from_step2a(self):
        self.assertEqual(FULL_TRAIN_DATASET, "full_train_dataset")
        self.assertEqual(FULL_TRAIN_MANIFEST, "full_train_manifest.json")
        self.assertNotEqual(FULL_TRAIN_DATASET, "tiny_overfit_dataset")
        self.assertNotEqual(FULL_TRAIN_MANIFEST, "tiny_overfit_manifest.json")

    def test_train_command_enforces_the_frozen_recipe_and_the_device_gate(self):
        source = CLI.read_text(encoding="utf-8")
        train = source.split("def cmd_train(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("train_args.update(TRAIN_RECIPE)", train)
        self.assertIn("from ultralytics.cfg import get_cfg", train)
        self.assertIn("missing_required", train)
        self.assertIn("would silently drop frozen", train)
        self.assertIn("_required_device(args, environment)", train)
        self.assertIn('"name": "full_finetune"', train)
        self.assertNotIn("args.weight.write", train)

    def test_freeze_command_refuses_the_pretrained_weight_and_freezes_read_only(self):
        source = CLI.read_text(encoding="utf-8")
        freeze = source.split("def cmd_freeze(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("byte-identical to the pretrained weight", freeze)
        self.assertIn("checkpoint_hashes.json", freeze)
        self.assertIn("checkpoint_freeze.json", freeze)
        self.assertIn("_chmod_read_only(run_dir)", freeze)
        self.assertIn("_pretrained_weight_sha(args, training)", freeze)
        self.assertIn("pretrained_weight_sha_source", freeze)
        self.assertIn("PRIMARY_CHECKPOINT", freeze)
        self.assertIn("DIAGNOSTIC_CHECKPOINT", freeze)
        self.assertIn("never trains and never runs", freeze)

    def test_report_rederives_the_verdict_without_rerunning_inference(self):
        """Corrected handoff labels must not require a new training or inference run."""
        source = CLI.read_text(encoding="utf-8")
        report = source.split("def cmd_report(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("training_completion_verdict(", report)
        self.assertIn('_write_json(args, "verdict.json", verdict)', report)
        self.assertIn("superseded_verdict", report)
        self.assertIn("authoritative_source", report)
        self.assertNotIn("cmd_post(", report)
        self.assertNotIn("model.predict", report)

    def test_pretrained_weight_sha_falls_back_to_recorded_upload_evidence(self):
        """A host holding only the training set has no preflight.json."""
        source = CLI.read_text(encoding="utf-8")
        helper = source.split("def _pretrained_weight_sha(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('"preflight.json"', helper)
        self.assertIn('"training.json"', helper)
        self.assertIn('"frozen_input_verification.json"', helper)
        self.assertIn('"upload_sha256sums.json"', helper)
        self.assertIn("recorded pretrained weight SHA-256", helper)
        train = source.split("def cmd_train(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('"pretrained_weight"', train)

    def test_post_runs_on_the_training_set_only(self):
        source = CLI.read_text(encoding="utf-8")
        post = source.split("def cmd_post(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('_predict(args, best, "post"', post)
        self.assertIn("training_completion_verdict", post)
        self.assertIn("loader_ok", post)
        self.assertIn("full_pool_ok", post)


@requires_real_pools
class TestRealPreflightAndStage(unittest.TestCase):
    def test_preflight_and_stage_on_a_fresh_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "step2b"
            for command in ("preflight", "stage"):
                proc = subprocess.run(
                    [sys.executable, str(CLI), "--output", str(out), command],
                    capture_output=True, text=True, timeout=900)
                self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            manifest = json.loads((out / FULL_TRAIN_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["positive"]), EXPECTED["positive_images"])
            self.assertEqual(len(manifest["negative"]), EXPECTED["negative_images"])
            staging = json.loads((out / "staging_integrity.json").read_text(encoding="utf-8"))
            self.assertTrue(staging["integrity"]["ok"], staging["integrity"])
            self.assertEqual(staging["integrity"]["files"], 107)
            dataset = out / FULL_TRAIN_DATASET
            self.assertEqual(len(list((dataset / "images" / "train").glob("*.png"))), 107)
            self.assertEqual(len(list((dataset / "labels" / "train").glob("*.txt"))), 107)
            text = (dataset / "data.yaml").read_text(encoding="utf-8")
            self.assertIn(DATA_YAML_NOTE, text)
            self.assertIn("train: images/train", text)
            self.assertIn("val: images/train", text)
            self.assertIn("0: ground_litter", text)
            full_pool = json.loads((out / "full_pool_check.json").read_text(encoding="utf-8"))
            self.assertTrue(full_pool["ok"], full_pool["problems"])

    def test_staging_the_full_pool_changes_no_frozen_bytes(self):
        pools = load_pools(POSITIVE_ROOT, NEGATIVE_ROOT)
        manifest = full_pool_manifest(pools)
        tracked = [entry["image_path"] for entry in manifest["positive"] + manifest["negative"]]
        tracked += [entry["label_path"] for entry in manifest["positive"]
                    + manifest["negative"]]
        tracked += [POSITIVE_ROOT / "positive_training_manifest_v2.jsonl",
                    NEGATIVE_ROOT / "hard_negative_training_manifest.jsonl"]
        before = {path: sha256_file(path) for path in tracked}
        with tempfile.TemporaryDirectory() as tmp:
            from rtsp_annotator.ground_litter_step2a import stage_dataset, verify_staging
            report = stage_dataset(manifest, Path(tmp) / "dataset", mode="hardlink")
            self.assertTrue(verify_staging(report)["ok"])
        self.assertEqual({path: sha256_file(path) for path in tracked}, before)


class TestRecordedStep2bEvidence(unittest.TestCase):
    """The authorised CUDA run; skipped while the artifacts are absent."""

    @classmethod
    def setUpClass(cls):
        path = STEP2B_OUT / "SUMMARY.json"
        if not path.is_file():
            raise unittest.SkipTest("recorded Step 2B summary is not present")
        cls.summary = json.loads(path.read_text(encoding="utf-8"))
        cls.training = cls.summary.get("training") or {}
        cls.post = cls.summary.get("post") or {}
        cls.freeze = cls.summary.get("checkpoint_freeze") or {}
        cls.server = cls.summary.get("server_run") or {}
        cls.environment = cls.summary.get("environment") or {}

    def test_recipe_and_train_set_are_recorded(self):
        self.assertEqual(self.summary["recipe"], TRAIN_RECIPE)
        train_set = self.summary["train_set"]
        self.assertEqual(train_set["positive_tiles"], EXPECTED["positive_images"])
        self.assertEqual(train_set["negative_tiles"], EXPECTED["negative_images"])
        self.assertEqual(train_set["positive_bboxes"], EXPECTED["positive_boxes"])
        self.assertIs(train_set["train_equals_val"], True)
        self.assertIn("FULL_TRAIN_SET_SANITY_ONLY", train_set["note"])

    def test_boundaries_and_no_development_access(self):
        self.assertTrue(all(value is False
                            for value in self.summary["boundaries"].values()))
        self.assertFalse(self.freeze["step2c_handoff"]["development_accessed"])
        self.assertFalse(self.freeze["step2c_handoff"]["sealed_accessed"])

    def test_training_used_the_frozen_recipe_and_the_frozen_pool_manifest(self):
        if not self.training:
            self.skipTest("the full fine-tune was not run")
        applied = self.training["applied_args"]
        for key, value in FROZEN_RECIPE.items():
            self.assertEqual(applied[key], value, key)
        self.assertEqual(applied["imgsz"], 640)
        self.assertEqual(self.training["unsupported_args_skipped"], [])
        self.assertEqual(self.training["manifest_provenance"]["source"], "frozen_file")
        manifest_path = STEP2B_OUT / FULL_TRAIN_MANIFEST
        if manifest_path.is_file():
            self.assertEqual(self.training["manifest_provenance"]["sha256"],
                             sha256_file(manifest_path))
        self.assertIs(self.training["nan_or_inf"], False)
        self.assertIs(self.training["loss_decreased"], True)
        self.assertGreaterEqual(self.training["epochs_run"], 1)
        self.assertLessEqual(self.training["epochs_run"],
                             self.training["epochs_requested"])

    def test_checkpoint_was_frozen_by_hash(self):
        if not self.freeze:
            self.skipTest("no checkpoint freeze is recorded")
        self.assertEqual(self.freeze["primary"], "last.pt")
        self.assertEqual(self.freeze["primary_epoch"], 100)
        self.assertEqual(self.freeze["primary_sha256"], PRIMARY_SHA256)
        self.assertNotEqual(self.freeze["primary_sha256"], WEIGHT_SHA256)
        self.assertGreater(self.freeze["primary_bytes"], 0)
        self.assertEqual(self.freeze["step2c_handoff"]["weights_sha256"],
                         self.freeze["primary_sha256"])
        self.assertEqual(self.freeze["step2c_handoff"]["conf_levels"], list(CONF_LEVELS))
        read_only = self.freeze.get("read_only") or {}
        for name, info in read_only.items():
            if "mode_after" in info:
                self.assertEqual(int(info["mode_after"], 8) & 0o222, 0, name)

    def test_best_checkpoint_is_diagnostic_only_in_the_recorded_evidence(self):
        if not self.freeze:
            self.skipTest("no checkpoint freeze is recorded")
        diagnostic = self.freeze["diagnostic"]
        self.assertEqual(diagnostic["name"], "best.pt")
        self.assertEqual(diagnostic["sha256"], DIAGNOSTIC_SHA256)
        self.assertEqual(diagnostic["role"], "diagnostic only")
        self.assertEqual(diagnostic["forbidden_as"],
                         "Step 2C first-round official checkpoint")
        self.assertIn("training-set validation metric", diagnostic["reason"])
        forbidden = self.freeze["step2c_handoff"]["forbidden_checkpoints"]
        self.assertEqual(forbidden["best.pt"]["sha256"], DIAGNOSTIC_SHA256)
        self.assertNotEqual(self.freeze["step2c_handoff"]["weights_sha256"],
                            DIAGNOSTIC_SHA256)
        self.assertNotIn("best.pt", self.freeze["step2c_handoff"]["weights"])

    def test_verdict_is_full_training_complete(self):
        if not self.post:
            self.skipTest("no post-train metrics are recorded")
        verdict = self.summary["verdict"]
        self.assertEqual(verdict["verdict"], VERDICT_COMPLETE)
        self.assertEqual(verdict["verdict"], "FULL_TRAINING_COMPLETE")
        self.assertNotEqual(verdict["verdict"], SUPERSEDED_VERDICT)
        # the embedded raw post record deliberately keeps its original label
        self.assertEqual(self.post["verdict"]["verdict"], SUPERSEDED_VERDICT)
        self.assertEqual(verdict.get("superseded_verdict"),
                         self.post["verdict"]["verdict"])
        self.assertEqual(verdict["hard_failures"], [])
        self.assertEqual(verdict["completion_failures"], [])
        self.assertEqual(verdict["failed_preconditions"], [])
        for key, value in verdict["preconditions"].items():
            self.assertIsInstance(value, bool, key)
        self.assertIn("train", verdict["note"].lower())
        # the raw inference record keeps its original label; the normalisation is explicit
        self.assertEqual(verdict.get("superseded_verdict"), SUPERSEDED_VERDICT)
        self.assertIn("no training or inference was re-run", verdict["superseded_note"])
        verdict_path = STEP2B_OUT / "verdict.json"
        if verdict_path.is_file():
            standalone = json.loads(verdict_path.read_text(encoding="utf-8"))
            self.assertEqual(standalone["verdict"], verdict["verdict"])

    def test_post_uses_the_frozen_matching_and_confidence_levels(self):
        if not self.post:
            self.skipTest("no post-train metrics are recorded")
        metrics = self.post["metrics"]
        self.assertEqual(metrics["conf_levels"], list(CONF_LEVELS))
        matching = metrics["matching"]
        self.assertEqual(matching["iou_normal"], IOU_NORMAL)
        self.assertEqual(matching["iou_small"], IOU_SMALL)
        self.assertEqual(matching["small_gt_short_side_px"], SMALL_GT_SHORT_SIDE_PX)
        self.assertEqual(matching["small_expand"], SMALL_EXPAND)
        self.assertEqual(matching["small_area_ratio"],
                         [SMALL_AREA_RATIO_MIN, SMALL_AREA_RATIO_MAX])
        manifest_path = STEP2B_OUT / FULL_TRAIN_MANIFEST
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            positive_ids = set(manifest["positive_tile_ids"])
            negative_ids = set(manifest["negative_tile_ids"])
            for conf in ("0.01", "0.05", "0.10"):
                detail = metrics["per_conf"][conf]
                self.assertEqual({row["tile_id"] for row in detail["per_positive_image"]},
                                 positive_ids)
                self.assertEqual({row["tile_id"] for row in detail["per_negative_image"]},
                                 negative_ids)

    def test_environment_and_upload_evidence_are_honest(self):
        if not self.training:
            self.skipTest("the full fine-tune was not run")
        self.assertTrue(self.environment.get("cuda_available"))
        self.assertEqual(self.environment["blockers"], [])
        verification = self.server.get("frozen_input_verification")
        if verification:
            self.assertTrue(verification["ok"], verification)
            self.assertTrue(verification["weight_matches_local_frozen"])
            self.assertEqual(verification["weight_sha256"], WEIGHT_SHA256)
            self.assertTrue(verification["all_negative_labels_zero_bytes"])
            self.assertEqual(verification["positive_tiles"], EXPECTED["positive_images"])
            self.assertEqual(verification["negative_tiles"], EXPECTED["negative_images"])
        upload = self.server.get("upload_sha256sums")
        if upload:
            self.assertEqual(upload["file_count"], len(upload["files"]))
            self.assertEqual(upload["files"]["assets/yolo26s.pt"], WEIGHT_SHA256)

    def test_recorded_artifacts_never_point_at_sealed_or_development_assets(self):
        def paths(payload, found=None):
            if found is None:
                found = []
            if isinstance(payload, dict):
                for value in payload.values():
                    paths(value, found)
            elif isinstance(payload, list):
                for value in payload:
                    paths(value, found)
            elif isinstance(payload, str) and (payload.startswith("/")
                                               or payload.startswith("output/")):
                found.append(payload)
            return found

        for name in ("preflight.json", "full_pool_check.json", "full_train_manifest.json",
                     "staging_integrity.json", "loader_sanity.json", "training.json",
                     "post_metrics.json", "checkpoint_freeze.json", "SUMMARY.json",
                     "MANIFEST.json"):
            path = STEP2B_OUT / name
            if not path.is_file():
                continue
            for value in paths(json.loads(path.read_text(encoding="utf-8"))):
                lowered = value.lower()
                for marker in SEALED_MARKERS:
                    self.assertNotIn(marker.lower(), lowered, f"{name}: {value}")
                self.assertNotIn("/sealed", lowered, f"{name}: {value}")
                self.assertNotIn("development", lowered, f"{name}: {value}")


if __name__ == "__main__":
    unittest.main()
