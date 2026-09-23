"""Step 2B: full fine-tune on the frozen training pool + train-set sanity + freeze.

Step 2A proved the training *chain* works on a 12+10 tile subset.  Step 2B trains a
real model on the **whole** frozen pool (44 positive tiles / 96 boxes + 63 negative
tiles / 0 boxes) with the light augmentation recipe frozen by the operator, reports the
train-set metrics, freezes exactly one checkpoint by SHA-256, and stops.  It makes **no**
generalisation claim and no model-quality claim: everything here is measured on the
training tiles, and Development is Step 2C.

Two decisions are pre-registered here rather than taken from the results:

* the Step 2B verdict is ``FULL_TRAINING_COMPLETE`` -- a statement that the requested
  schedule ran to the end on the frozen pool with the frozen recipe and no NaN/Inf.  The
  train-set metrics are reported for the record and never grade the model.
* the single Step 2C handoff checkpoint is ``last.pt`` (epoch 100).  Because train == val,
  ``best.pt`` is chosen by the training-set validation metric, so using it would be
  checkpoint selection on the very data the sanity metrics come from; it is retained as a
  diagnostic only and is explicitly forbidden as the Step 2C first-round checkpoint.

The logic is stdlib-only; the Ultralytics loader/trainer/predictor are injected by
``scripts/run_ground_litter_step2b.py``.  Step 0B matching, staging and evaluation come
from :mod:`rtsp_annotator.ground_litter_step2a` unchanged -- they are never re-invented
here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ground_litter_localization_review import (  # reuse, do not duplicate
    atomic_write_json,
    sha256_file,
)
from .ground_litter_step2a import (  # reuse, do not duplicate
    CONF_LEVELS,
    IOU_NORMAL,
    IOU_SMALL,
    SEED,
    SMALL_AREA_RATIO_MAX,
    SMALL_AREA_RATIO_MIN,
    SMALL_EXPAND,
    SMALL_GT_SHORT_SIDE_PX,
    Step2AError,
    size_bucket,
)

SCHEMA_VERSION = "ground_litter_step2b_v1"
GENERATOR_VERSION = "step2b-1.0.0"

#: Step 2B stages its own dataset view; the names never collide with Step 2A.
FULL_TRAIN_DATASET = "full_train_dataset"
FULL_TRAIN_MANIFEST = "full_train_manifest.json"

TRAIN_SET_NOTE = ("FULL_TRAIN_SET_SANITY_ONLY: every metric in this step is measured on "
                  "the training tiles; train == val, so no generalisation is claimed")
#: Written into data.yaml as the first comment line (Step 2A wrote its own note).
DATA_YAML_NOTE = "FULL_TRAIN_SET_SANITY_ONLY"

#: The operator-frozen Step 2B recipe.  Light, native Ultralytics augmentation only:
#: §14.1 asks for light brightness/blur/compression/translation/scale, but 8.4.150 has
#: no native blur or JPEG-compression augmentation, so those two were deliberately left
#: out rather than approximated by a new parameter.
TRAIN_RECIPE: dict[str, Any] = {
    "imgsz": 640,
    "epochs": 100,
    "batch": 8,
    "seed": SEED,
    "deterministic": True,
    "optimizer": "auto",
    "pretrained": True,
    "val": True,
    "plots": False,
    # light augmentation (operator-specified values, verbatim)
    "hsv_h": 0.015,
    "hsv_s": 0.30,
    "hsv_v": 0.20,
    "translate": 0.02,
    "scale": 0.05,
    "fliplr": 0.5,
    # explicitly disabled
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "close_mosaic": 0,
    "perspective": 0.0,
    "degrees": 0.0,
    "shear": 0.0,
    "flipud": 0.0,
    "erasing": 0.0,
}
LIGHT_AUGMENTATION = ("hsv_h", "hsv_s", "hsv_v", "translate", "scale", "fliplr")
DISABLED_AUGMENTATION = ("mosaic", "mixup", "copy_paste", "perspective", "degrees",
                         "shear", "flipud", "erasing")
REQUIRED_ARGS = ("data", "imgsz", "epochs", "batch", "device", "seed", "deterministic",
                 "mosaic", "mixup", "copy_paste", "optimizer", "project", "name")

#: §16.3: the fine-tuned model must be able to fit the training samples.  Pre-registered
#: before the run; this is a train-set bar, not a generalisation criterion.
TRAIN_SET_RECALL_BAR = 0.90
#: Same loss rule Step 2A used: the final box loss must be at most 80% of the first.
LOSS_DECREASE_FACTOR = 0.80

#: Step 1C-2M / 1D frozen pool expectations (whole pool, no selection).
EXPECTED = {
    "positive_images": 44, "positive_boxes": 96,
    "negative_images": 63, "negative_boxes": 0,
    "easy_negatives": 15, "hard_negatives": 48,
}
#: Step 1D bounded the easy-background share of the negative pool.
MAX_EASY_NEGATIVE_FRACTION = 0.30
EASY_NEGATIVE_HARDNESS = "random_grid_background"

#: Step 2B verdict: does the full training run itself complete?  It says nothing about
#: model quality, and the train-set metrics never select or grade the model.  The earlier
#: "FULL_FINETUNE_SANITY_PASS" label is superseded: with train == val a "pass" would imply
#: a judgement the data cannot support.
VERDICT_COMPLETE = "FULL_TRAINING_COMPLETE"
VERDICT_INCOMPLETE = "FULL_TRAINING_INCOMPLETE"
VERDICT_FAILED = "FULL_TRAINING_FAILED"

#: A hard failure means the run must not be treated as a finished training artifact.
HARD_PRECONDITIONS = ("loader_ok", "full_pool_ok", "frozen_args_applied",
                      "augmentation_matches_recipe", "nan_free")
#: These only say whether the requested schedule actually ran to the end.
COMPLETION_PRECONDITIONS = ("epochs_completed", "loss_decreased")

#: The single Step 2C handoff checkpoint.  train == val, so best.pt was chosen on the
#: training-set validation metric; using it would be checkpoint selection on the very data
#: the sanity metrics come from, so the last epoch is the pre-registered artifact.
PRIMARY_CHECKPOINT = "last.pt"
DIAGNOSTIC_CHECKPOINT = "best.pt"
DIAGNOSTIC_REASON = ("best.pt was selected by the training-set validation metric while "
                     "train == val; it must not be used as the Step 2C first-round "
                     "official checkpoint")


class Step2BError(Step2AError):
    """Step 2B could not proceed."""


def _positive_entry(row: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "tile_id": row["tile_id"], "camera_id": row["camera_id"],
        "image_path": row["image_path"], "label_path": row["label_path"],
        "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"],
        "box_count": row["box_count"], "boxes": [list(box) for box in row["boxes"]],
        "short_sides": list(row["short_sides"]), "buckets": list(row["buckets"]),
        "origin": row.get("origin"), "source_file_id": row.get("source_file_id"),
        "reason": reason,
    }


def _negative_entry(row: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "tile_id": row["tile_id"], "camera_id": row["camera_id"],
        "image_path": row["image_path"], "label_path": row["label_path"],
        "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"],
        "label_bytes": row["label_bytes"], "box_count": row["box_count"],
        "hardness_source": row.get("hardness_source"),
        "origin": row.get("origin"), "source_file_id": row.get("source_file_id"),
        "reason": reason,
    }


def full_pool_manifest(pools: Mapping[str, Any]) -> dict[str, Any]:
    """The whole frozen pool as the Step 2B training set (no selection, no re-sampling)."""
    reason = "full frozen pool: no selection, no re-sampling"
    positive_rows = list(pools["positives"])
    negative_rows = list(pools["negatives"])
    buckets: dict[str, int] = {}
    for row in positive_rows:
        for bucket in row["buckets"]:
            buckets[bucket] = buckets.get(bucket, 0) + 1
    easy = sum(1 for row in negative_rows
               if row.get("hardness_source") == EASY_NEGATIVE_HARDNESS)
    per_camera_positive: dict[str, int] = {}
    per_camera_negative: dict[str, int] = {}
    for row in positive_rows:
        per_camera_positive[row["camera_id"]] = per_camera_positive.get(row["camera_id"], 0) + 1
    for row in negative_rows:
        per_camera_negative[row["camera_id"]] = per_camera_negative.get(row["camera_id"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "note": TRAIN_SET_NOTE,
        "seed": SEED,
        "selection": "none: the training set is the entire frozen Step 1C-2M / 1D pool",
        "positive_tile_ids": [row["tile_id"] for row in positive_rows],
        "negative_tile_ids": [row["tile_id"] for row in negative_rows],
        "positive": [_positive_entry(row, reason) for row in positive_rows],
        "negative": [_negative_entry(row, reason) for row in negative_rows],
        "coverage": {
            "positive_tiles": len(positive_rows),
            "negative_tiles": len(negative_rows),
            "positive_bboxes": sum(row["box_count"] for row in positive_rows),
            "multi_label_tiles": sum(1 for row in positive_rows if row["box_count"] >= 2),
            "three_plus_label_tiles": sum(1 for row in positive_rows
                                          if row["box_count"] >= 3),
            "short_side_buckets": dict(sorted(buckets.items())),
            "positive_cameras": sorted(per_camera_positive),
            "negative_cameras": sorted(per_camera_negative),
            "per_camera_positive": dict(sorted(per_camera_positive.items())),
            "per_camera_negative": dict(sorted(per_camera_negative.items())),
            "easy_negative_count": easy,
            "hard_negative_count": len(negative_rows) - easy,
            "positive_source_recordings": len({row.get("source_file_id")
                                               for row in positive_rows}),
            "negative_source_recordings": len({row.get("source_file_id")
                                               for row in negative_rows}),
        },
        "source_hashes": {
            "positive_training_manifest_sha256": pools["positive_manifest_sha256"],
            "hard_negative_training_manifest_sha256": pools["negative_manifest_sha256"],
        },
    }


def verify_full_pool(pools: Mapping[str, Any], manifest: Mapping[str, Any],
                     preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Independent checks on the Step 2B training set before anything is trained."""
    coverage = manifest["coverage"]
    counts = {
        "positive_images": len(manifest["positive"]),
        "positive_boxes": sum(entry["box_count"] for entry in manifest["positive"]),
        "negative_images": len(manifest["negative"]),
        "negative_boxes": sum(entry["box_count"] for entry in manifest["negative"]),
    }
    positive_ids = [entry["tile_id"] for entry in manifest["positive"]]
    negative_ids = [entry["tile_id"] for entry in manifest["negative"]]
    duplicates = sorted({tile for tile in positive_ids + negative_ids
                         if (positive_ids + negative_ids).count(tile) > 1})
    pool_ids = {row["tile_id"] for row in pools["positives"]} | \
        {row["tile_id"] for row in pools["negatives"]}
    missing = sorted(pool_ids - set(positive_ids) - set(negative_ids))
    expected_counts = {key: EXPECTED[key] for key in counts}
    negatives = manifest["negative"]
    easy = coverage["easy_negative_count"]
    easy_fraction = round(easy / len(negatives), 4) if negatives else 0.0
    non_empty = [entry["tile_id"] for entry in negatives
                 if int(entry.get("label_bytes") or 0) != 0]
    positive_without_boxes = [entry["tile_id"] for entry in manifest["positive"]
                              if entry["box_count"] < 1]
    problems: list[dict[str, Any]] = []
    for label, bad in (("count_mismatch", counts != expected_counts),
                       ("duplicate_tiles", bool(duplicates)),
                       ("pool_tiles_missing", bool(missing)),
                       ("negative_label_not_empty", bool(non_empty)),
                       ("positive_without_boxes", bool(positive_without_boxes)),
                       ("easy_fraction_exceeded",
                        easy_fraction > MAX_EASY_NEGATIVE_FRACTION),
                       ("preflight_not_ok",
                        bool(preflight is not None and not preflight.get("counts_match")))):
        if bad:
            problems.append({"field": label})
    return {
        "counts": counts, "expected": expected_counts, "counts_match": counts == expected_counts,
        "duplicate_tiles": duplicates, "pool_tiles_missing": missing,
        "pool_tiles_total": len(pool_ids), "negative_label_not_empty": non_empty,
        "positive_without_boxes": positive_without_boxes,
        "easy_negative_count": easy, "easy_negative_fraction": easy_fraction,
        "max_easy_negative_fraction": MAX_EASY_NEGATIVE_FRACTION,
        "coverage": coverage, "problems": problems, "ok": not problems,
    }


def training_completion_verdict(training: Mapping[str, Any], post: Mapping[str, Any], *,
                                loader_ok: bool, full_pool_ok: bool) -> dict[str, Any]:
    """Did the full training run complete?  Model quality is not judged here.

    ``FULL_TRAINING_COMPLETE`` only asserts that the requested schedule ran to the end on
    the frozen pool with the frozen recipe and no NaN/Inf.  The train-set metrics are
    reported for the record and never select or grade the model.
    """
    conf01 = post["metrics"]["per_conf"]["0.01"]
    recall = float(conf01["positive_gt_proposal_recall"])
    hit = float(conf01["positive_image_hit_rate"])
    applied = dict(training.get("applied_args") or {})
    preconditions = {
        "loader_ok": bool(loader_ok),
        "full_pool_ok": bool(full_pool_ok),
        "epochs_completed": (int(training.get("epochs_run") or 0) >= 1
                             and training.get("epochs_run") == training.get("epochs_requested")),
        "frozen_args_applied": (not training.get("unsupported_args_skipped")
                                and all(key in applied for key in REQUIRED_ARGS)),
        "augmentation_matches_recipe": all(
            applied.get(key) == TRAIN_RECIPE[key]
            for key in LIGHT_AUGMENTATION + DISABLED_AUGMENTATION + ("close_mosaic",)),
        "nan_free": not training.get("nan_or_inf"),
        "loss_decreased": bool(training.get("loss_decreased")),
    }
    result: dict[str, Any] = {
        "preconditions": preconditions,
        "failed_preconditions": sorted(key for key, value in preconditions.items()
                                       if not value),
        "hard_failures": sorted(key for key in HARD_PRECONDITIONS
                                if not preconditions[key]),
        "completion_failures": sorted(key for key in COMPLETION_PRECONDITIONS
                                      if not preconditions[key]),
        "train_set_gt_proposal_recall_at_0.01": recall,
        "train_set_image_hit_rate_at_0.01": hit,
        "train_set_recall_bar": TRAIN_SET_RECALL_BAR,
        "train_set_recall_bar_role": "informational reference only; never selects the "
                                     "checkpoint and never grades the model",
        "verdict_basis": "run completion only: schedule, frozen recipe, no NaN/Inf",
        "note": TRAIN_SET_NOTE,
    }
    if result["hard_failures"]:
        result["verdict"] = VERDICT_FAILED
    elif result["completion_failures"]:
        result["verdict"] = VERDICT_INCOMPLETE
    else:
        result["verdict"] = VERDICT_COMPLETE
    return result


def checkpoint_freeze_record(training: Mapping[str, Any], checkpoints: Mapping[str, Any],
                             *, primary: str = PRIMARY_CHECKPOINT,
                             diagnostic: str = DIAGNOSTIC_CHECKPOINT,
                             frozen_at: str | None = None,
                             manifest_sha256: str | None = None) -> dict[str, Any]:
    """Exactly one checkpoint is handed to Step 2C, identified by SHA-256."""
    entries = {name: dict(info) for name, info in checkpoints.items()
               if isinstance(info, Mapping) and "sha256" in info}
    if primary not in entries:
        raise Step2BError(f"primary checkpoint {primary!r} has no recorded hash")
    primary_info = entries[primary]
    diagnostic_info = entries.get(diagnostic)
    stamp = frozen_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema_version": SCHEMA_VERSION,
        "frozen_at": stamp,
        "primary": primary,
        "primary_epoch": training.get("epochs_run"),
        "primary_path": primary_info["path"],
        "primary_sha256": primary_info["sha256"],
        "primary_bytes": primary_info["bytes"],
        "selection_rule": (f"last epoch (epoch {training.get('epochs_run')}); best.pt is "
                           f"excluded because train == val would make it a selection on "
                           f"the training-set validation metric"),
        "checkpoints": entries,
        "best_equals_last": checkpoints.get("best_equals_last"),
        "diagnostic": ({
            "name": diagnostic,
            "path": diagnostic_info["path"],
            "sha256": diagnostic_info["sha256"],
            "bytes": diagnostic_info["bytes"],
            "role": "diagnostic only",
            "forbidden_as": "Step 2C first-round official checkpoint",
            "reason": DIAGNOSTIC_REASON,
        } if diagnostic_info else None),
        "training": {
            "epochs_run": training.get("epochs_run"),
            "epochs_requested": training.get("epochs_requested"),
            "batch": training.get("batch"), "imgsz": training.get("imgsz"),
            "device": training.get("device"), "seed": training.get("seed"),
            "save_dir": training.get("save_dir"),
            "initial_box_loss": training.get("initial_box_loss"),
            "final_box_loss": training.get("final_box_loss"),
            "nan_or_inf": training.get("nan_or_inf"),
        },
        "recipe": dict(TRAIN_RECIPE),
        "train_manifest_sha256": manifest_sha256,
        "step2c_handoff": {
            "weights": primary_info["path"],
            "weights_sha256": primary_info["sha256"],
            "weights_epoch": training.get("epochs_run"),
            "rule": (f"Step 2C must load exactly this SHA-256 ({primary}, last epoch); the "
                     f"checkpoint was fixed before any Development asset is looked at, and "
                     f"no re-selection on Development is allowed"),
            "forbidden_checkpoints": ({
                diagnostic: {
                    "sha256": diagnostic_info["sha256"],
                    "role": "diagnostic only",
                    "reason": DIAGNOSTIC_REASON,
                },
            } if diagnostic_info else {}),
            "matching": {
                "iou_normal": IOU_NORMAL, "iou_small": IOU_SMALL,
                "small_gt_short_side_px": SMALL_GT_SHORT_SIDE_PX,
                "small_expand": SMALL_EXPAND,
                "small_area_ratio": [SMALL_AREA_RATIO_MIN, SMALL_AREA_RATIO_MAX],
                "one_to_one": True,
            },
            "conf_levels": list(CONF_LEVELS),
            "train_val_split": "train == val (train-set sanity only)",
            "development_accessed": False,
            "sealed_accessed": False,
        },
    }


def boundary_flags() -> dict[str, bool]:
    return {
        "development_accessed": False,
        "sealed_accessed": False,
        "upstream_dataset_bytes_modified": False,
        "pretrained_weight_modified": False,
        "threshold_tuned": False,
        "hyperparameters_tuned": False,
        "tiny_subset_resampled": False,
        "step2c_started": False,
    }


def build_summary(pools: Mapping[str, Any], preflight: Mapping[str, Any],
                  environment: Mapping[str, Any], manifest: Mapping[str, Any],
                  full_pool: Mapping[str, Any], staging: Mapping[str, Any],
                  loader: Mapping[str, Any] | None,
                  baseline: Mapping[str, Any] | None = None,
                  post: Mapping[str, Any] | None = None,
                  training: Mapping[str, Any] | None = None,
                  verdict: Mapping[str, Any] | None = None,
                  freeze: Mapping[str, Any] | None = None,
                  manifest_sha256: str | None = None,
                  server_run: Mapping[str, Any] | None = None) -> dict[str, Any]:
    coverage = manifest["coverage"]
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "environment": dict(environment),
        "preflight": dict(preflight),
        "train_set": {
            "positive_tiles": coverage["positive_tiles"],
            "negative_tiles": coverage["negative_tiles"],
            "positive_bboxes": coverage["positive_bboxes"],
            "multi_label_tiles": coverage["multi_label_tiles"],
            "three_plus_label_tiles": coverage["three_plus_label_tiles"],
            "short_side_buckets": coverage["short_side_buckets"],
            "positive_cameras": coverage["positive_cameras"],
            "negative_cameras": coverage["negative_cameras"],
            "per_camera_positive": coverage["per_camera_positive"],
            "per_camera_negative": coverage["per_camera_negative"],
            "easy_negative_count": coverage["easy_negative_count"],
            "hard_negative_count": coverage["hard_negative_count"],
            "positive_source_recordings": coverage["positive_source_recordings"],
            "negative_source_recordings": coverage["negative_source_recordings"],
            "manifest_sha256": manifest_sha256,
            "train_equals_val": True,
            "note": TRAIN_SET_NOTE,
        },
        "recipe": dict(TRAIN_RECIPE),
        "full_pool_check": dict(full_pool),
        "staging_integrity": dict(staging.get("integrity") or staging),
        "loader_sanity": dict(loader or {"status": "not_run"}),
        "baseline": baseline, "post": post, "training": training,
        "verdict": dict(verdict or {"verdict": "NOT_EVALUATED"}),
        "checkpoint_freeze": freeze,
        "server_run": dict(server_run) if server_run else None,
        "boundaries": boundary_flags(),
        "input_context": {
            "positive_root": str(pools["positive_root"]),
            "negative_root": str(pools["negative_root"]),
        },
    }


def build_manifest(pools: Mapping[str, Any], summary: Mapping[str, Any], *,
                   code_commit: str, generated_at: str, config: Mapping[str, Any],
                   artifact_root: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_at": generated_at,
        "code_commit": code_commit,
        "seed": SEED,
        "input_sha256": dict(summary["preflight"]["hashes"]),
        "weight": summary["preflight"].get("weight"),
        "artifact_root": str(artifact_root),
        "config": dict(config),
        "recipe": dict(TRAIN_RECIPE),
        "counts": {
            "positive_images": summary["preflight"]["counts"]["positive_images"],
            "positive_boxes": summary["preflight"]["counts"]["positive_boxes"],
            "negative_images": summary["preflight"]["counts"]["negative_images"],
            "negative_labels_empty": summary["preflight"]["counts"]["negative_images"],
        },
        "matching_parameters": {
            "iou_normal": IOU_NORMAL, "iou_small": IOU_SMALL,
            "small_gt_short_side_px": SMALL_GT_SHORT_SIDE_PX,
            "small_expand": SMALL_EXPAND,
            "small_area_ratio": [SMALL_AREA_RATIO_MIN, SMALL_AREA_RATIO_MAX],
        },
        "upstream_provenance": dict(provenance),
        "boundaries": dict(summary["boundaries"]),
        "note": (f"{TRAIN_SET_NOTE}; no Development / Sealed access, no hyperparameter or "
                 f"threshold tuning, verdict = full-run completion only, and "
                 f"{PRIMARY_CHECKPOINT} (last epoch) is the single pre-registered "
                 f"checkpoint frozen for Step 2C while {DIAGNOSTIC_CHECKPOINT} is "
                 f"diagnostic only"),
    }


def write_reports(output_dir: Path | str, *, manifest: Mapping[str, Any],
                  loader: Mapping[str, Any] | None, summary: Mapping[str, Any],
                  manifest_json: Mapping[str, Any]) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / FULL_TRAIN_MANIFEST
    atomic_write_json(manifest_path, manifest)
    if loader is not None:
        atomic_write_json(output_dir / "loader_sanity.json", loader)
    summary_path = output_dir / "SUMMARY.json"
    atomic_write_json(summary_path, summary)
    payload = dict(manifest_json)
    payload["train_manifest_sha256"] = sha256_file(manifest_path)
    payload["summary_sha256"] = sha256_file(summary_path)
    manifest_out = output_dir / "MANIFEST.json"
    atomic_write_json(manifest_out, payload)
    return {"train_manifest": str(manifest_path), "summary": str(summary_path),
            "manifest": str(manifest_out)}


def size_bucket_names(boxes: Sequence[Sequence[float]]) -> list[str]:
    """Convenience for reports/tests: the size bucket of each GT box."""
    return [size_bucket(min(float(box[2]) - float(box[0]), float(box[3]) - float(box[1])))
            for box in boxes]
