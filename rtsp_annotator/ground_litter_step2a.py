"""Step 2A: YOLO26s loader sanity + tiny overfit orchestration.

This step does **not** fine-tune for real.  Its only job is to prove that the frozen
training data can actually be trained on: the real Ultralytics dataset/dataloader reads
the positive labels and the 0-byte negative labels correctly, a tiny deterministic subset
can be overfit, and the whole chain (weights, GPU, loss, checkpoint, inference) works.

The logic here is stdlib-only: the environment probe, the Ultralytics loader, the trainer
and the predictor are injected by ``scripts/run_ground_litter_step2a.py``.

Guarantees encoded here:

* every input is read-only; staging copies/hardlinks bytes and then re-verifies SHA-256,
* the tiny subset is deterministic (requirement coverage first, then stable tie-break),
* the Step 0B frozen matching rules are used verbatim, never re-invented,
* Development / Sealed assets are never opened (marker guard + recorded boundary flags).
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from .ground_litter_localization_review import (  # reuse, do not duplicate
    SealedAssetError,
    assert_not_sealed,
    atomic_write_json,
    read_jsonl,
    sha256_file,
    write_jsonl,
)

SCHEMA_VERSION = "ground_litter_step2a_v1"
GENERATOR_VERSION = "step2a-1.0.0"

#: Step 0B §9 frozen matching parameters.  Do not tune these here.
SMALL_GT_SHORT_SIDE_PX = 20.0
IOU_NORMAL = 0.30
IOU_SMALL = 0.20
SMALL_EXPAND = 0.50
SMALL_AREA_RATIO_MIN = 0.25
SMALL_AREA_RATIO_MAX = 4.00
CONF_LEVELS = (0.01, 0.05, 0.10)

SEED = 42
#: §7 allows 8-12 per side.  12 positives is chosen so that every §8 requirement *and*
#: camera coverage fit without dropping any of them.
TINY_POSITIVE_TARGET = 12
TINY_NEGATIVE_TARGET = 10
TINY_MIN = 8
TINY_MAX = 12
MAX_EASY_NEGATIVE_FRACTION = 0.30
EASY_NEGATIVE_HARDNESS = "random_grid_background"

DATA_YAML_NOTE = "OVERFIT_SANITY_ONLY"
TRAIN_EQUALS_VAL = True

DATASET_LOADER_SANITY = "dataset_loader_sanity"
TINY_DATASET = "tiny_overfit_dataset"

POSITIVE_EXPECTED = {"images": 44, "boxes": 96}
NEGATIVE_EXPECTED = {"images": 63, "boxes": 0}


class Step2AError(RuntimeError):
    """Step 2A could not proceed."""


# --------------------------------------------------------------------------- #
# geometry + Step 0B matching
# --------------------------------------------------------------------------- #


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def expand_box(box: Sequence[float], fraction: float) -> list[float]:
    width = float(box[2]) - float(box[0])
    height = float(box[3]) - float(box[1])
    dx = width * fraction / 2.0
    dy = height * fraction / 2.0
    return [float(box[0]) - dx, float(box[1]) - dy, float(box[2]) + dx, float(box[3]) + dy]


def center_inside(inner: Sequence[float], outer: Sequence[float]) -> bool:
    cx = (float(inner[0]) + float(inner[2])) / 2.0
    cy = (float(inner[1]) + float(inner[3])) / 2.0
    return (float(outer[0]) <= cx <= float(outer[2])
            and float(outer[1]) <= cy <= float(outer[3]))


def is_eligible_match(pred: Sequence[float], gt: Sequence[float]) -> bool:
    """Step 0B §9.1/§9.2: normal IoU>=0.30, small targets get a relaxed rule."""
    iou = box_iou(pred, gt)
    if iou >= IOU_NORMAL:
        return True
    short_side = min(float(gt[2]) - float(gt[0]), float(gt[3]) - float(gt[1]))
    if short_side > SMALL_GT_SHORT_SIDE_PX:
        return False
    if iou >= IOU_SMALL:
        return True
    if not center_inside(pred, expand_box(gt, SMALL_EXPAND)):
        return False
    gt_area = box_area(gt)
    if gt_area <= 0:
        return False
    ratio = box_area(pred) / gt_area
    return SMALL_AREA_RATIO_MIN <= ratio <= SMALL_AREA_RATIO_MAX


def match_predictions(gt_boxes: Sequence[Sequence[float]],
                      predictions: Sequence[Mapping[str, Any]]
                      ) -> dict[str, Any]:
    """One-to-one greedy matching (§9.3): best IoU first, ties by higher score."""
    edges: list[tuple[float, float, int, int]] = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(predictions):
            if not is_eligible_match(pred["bbox"], gt):
                continue
            edges.append((box_iou(pred["bbox"], gt), float(pred.get("score") or 0.0),
                          gi, pi))
    edges.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
    matched_gt: dict[int, int] = {}
    matched_pred: set[int] = set()
    for iou, _score, gi, pi in edges:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt[gi] = pi
        matched_pred.add(pi)
    return {
        "matched_gt": matched_gt,
        "matched_pred": sorted(matched_pred),
        "tp": len(matched_gt),
        "fn": len(gt_boxes) - len(matched_gt),
        "unmatched_predictions": [index for index in range(len(predictions))
                                  if index not in matched_pred],
        "iou": {gi: round(box_iou(predictions[pi]["bbox"], gt_boxes[gi]), 4)
                for gi, pi in matched_gt.items()},
    }


def size_bucket(short_side: float) -> str:
    if short_side < 10:
        return "<10"
    if short_side < 20:
        return "10-19"
    if short_side < 40:
        return "20-39"
    if short_side < 80:
        return "40-79"
    return "80+"


# --------------------------------------------------------------------------- #
# input (§1/§2)
# --------------------------------------------------------------------------- #


def load_pools(positive_root: Path | str, negative_root: Path | str) -> dict[str, Any]:
    positive_root = Path(positive_root)
    negative_root = Path(negative_root)
    assert_not_sealed(positive_root, negative_root)
    positive_manifest = positive_root / "positive_training_manifest_v2.jsonl"
    negative_manifest = negative_root / "hard_negative_training_manifest.jsonl"
    for path in (positive_manifest, negative_manifest,
                 positive_root / "SUMMARY.json", positive_root / "MANIFEST.json",
                 negative_root / "SUMMARY.json", negative_root / "MANIFEST.json"):
        if not path.is_file():
            raise FileNotFoundError(f"frozen Step 1C-2M / 1D artifact missing: {path}")

    positives: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    for row in read_jsonl(positive_manifest):
        image = Path(str(row["image_path"]))
        label = Path(str(row["label_path"]))
        boxes = [[float(v) for v in line["tile_xyxy"]] for line in row["labels"]]
        short = [round(min(b[2] - b[0], b[3] - b[1]), 3) for b in boxes]
        if not image.is_file() or not label.is_file():
            problems.append({"tile_id": row["tile_id"], "field": "missing_file"})
            continue
        if sha256_file(image) != row["image_sha256"]:
            problems.append({"tile_id": row["tile_id"], "field": "image_hash_drift"})
        labels_text = label.read_text(encoding="utf-8").strip().splitlines()
        if len(labels_text) != len(boxes):
            problems.append({"tile_id": row["tile_id"], "field": "label_line_mismatch",
                             "value": len(labels_text)})
        positives.append({
            "tile_id": str(row["tile_id"]), "kind": "positive",
            "camera_id": str(row["camera_id"]), "image_path": str(image),
            "label_path": str(label), "image_sha256": str(row["image_sha256"]),
            "label_sha256": sha256_file(label), "label_bytes": label.stat().st_size,
            "label_lines": len(labels_text),
            "boxes": boxes, "box_count": len(boxes),
            "short_sides": short,
            "buckets": [size_bucket(value) for value in short],
            "min_short_side": min(short) if short else None,
            "origin": row.get("origin"), "hardness_source": None,
            "anchor_bbox": None, "source_file_id": row.get("source_file_id"),
            "source_crop_xyxy": row.get("source_crop_xyxy"),
        })

    negatives: list[dict[str, Any]] = []
    for row in read_jsonl(negative_manifest):
        image = Path(str(row["image_path"]))
        label = Path(str(row["label_path"]))
        if not image.is_file() or not label.is_file():
            problems.append({"tile_id": row["negative_tile_id"], "field": "missing_file"})
            continue
        if sha256_file(image) != row["image_sha256"]:
            problems.append({"tile_id": row["negative_tile_id"],
                             "field": "image_hash_drift"})
        if label.stat().st_size != 0:
            problems.append({"tile_id": row["negative_tile_id"],
                             "field": "negative_label_not_empty",
                             "value": label.stat().st_size})
        negatives.append({
            "tile_id": str(row["negative_tile_id"]), "kind": "negative",
            "camera_id": str(row["camera_id"]), "image_path": str(image),
            "label_path": str(label), "image_sha256": str(row["image_sha256"]),
            "label_sha256": sha256_file(label), "label_bytes": label.stat().st_size,
            "label_lines": 0, "boxes": [], "box_count": 0, "short_sides": [],
            "buckets": [], "min_short_side": None, "origin": row.get("origin"),
            "hardness_source": row.get("hardness_source"),
            "anchor_bbox": row.get("anchor_tile_xyxy"),
            "source_file_id": row.get("source_file_id"),
            "source_crop_xyxy": row.get("source_crop_xyxy"),
        })

    positives.sort(key=lambda row: (row["camera_id"], row["tile_id"]))
    negatives.sort(key=lambda row: (row["camera_id"], row["tile_id"]))
    return {
        "positive_root": positive_root, "negative_root": negative_root,
        "positive_manifest_path": positive_manifest,
        "negative_manifest_path": negative_manifest,
        "positive_manifest_sha256": sha256_file(positive_manifest),
        "negative_manifest_sha256": sha256_file(negative_manifest),
        "positive_summary_sha256": sha256_file(positive_root / "SUMMARY.json"),
        "positive_manifest_json_sha256": sha256_file(positive_root / "MANIFEST.json"),
        "negative_summary_sha256": sha256_file(negative_root / "SUMMARY.json"),
        "negative_manifest_json_sha256": sha256_file(negative_root / "MANIFEST.json"),
        "positives": positives, "negatives": negatives, "problems": problems,
    }


def verify_preflight(pools: Mapping[str, Any], weight: Path | str | None = None
                     ) -> dict[str, Any]:
    boxes = sum(row["box_count"] for row in pools["positives"])
    counts = {
        "positive_images": len(pools["positives"]),
        "positive_boxes": boxes,
        "negative_images": len(pools["negatives"]),
        "negative_labels": len(pools["negatives"]),
        "negative_boxes": sum(row["box_count"] for row in pools["negatives"]),
    }
    expected = {
        "positive_images": POSITIVE_EXPECTED["images"],
        "positive_boxes": POSITIVE_EXPECTED["boxes"],
        "negative_images": NEGATIVE_EXPECTED["images"],
        "negative_labels": NEGATIVE_EXPECTED["images"],
        "negative_boxes": NEGATIVE_EXPECTED["boxes"],
    }
    positive_sha = {row["image_sha256"] for row in pools["positives"]}
    negative_sha = {row["image_sha256"] for row in pools["negatives"]}
    positive_crop = {(row["source_file_id"], tuple(row["source_crop_xyxy"] or []))
                     for row in pools["positives"]}
    negative_crop = {(row["source_file_id"], tuple(row["source_crop_xyxy"] or []))
                     for row in pools["negatives"]}
    weight_info = None
    if weight is not None:
        path = Path(weight)
        weight_info = {"path": str(path), "exists": path.is_file(),
                       "size_bytes": path.stat().st_size if path.is_file() else None,
                       "sha256": sha256_file(path) if path.is_file() else None}
    empty_labels = [row["tile_id"] for row in pools["negatives"]
                    if row["label_bytes"] != 0]
    return {
        "counts": counts, "expected": expected, "counts_match": counts == expected,
        "image_sha_conflict_count": len(positive_sha & negative_sha),
        "source_crop_conflict_count": len(positive_crop & negative_crop),
        "negative_label_all_empty": not empty_labels,
        "negative_labels_not_empty": empty_labels,
        "problem_count": len(pools["problems"]), "problems": list(pools["problems"]),
        "weight": weight_info,
        "hashes": {
            "positive_summary_sha256": pools["positive_summary_sha256"],
            "positive_manifest_sha256": pools["positive_manifest_json_sha256"],
            "positive_training_manifest_sha256": pools["positive_manifest_sha256"],
            "negative_summary_sha256": pools["negative_summary_sha256"],
            "negative_manifest_sha256": pools["negative_manifest_json_sha256"],
            "hard_negative_training_manifest_sha256": pools["negative_manifest_sha256"],
        },
    }


# --------------------------------------------------------------------------- #
# deterministic tiny subset (§7-§10)
# --------------------------------------------------------------------------- #


def _positive_requirements(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def first(predicate, reason, used):
        for row in rows:
            if row["tile_id"] in used:
                continue
            if predicate(row):
                return {"reason": reason, "row": row}
        return None

    used: set[str] = set()
    chosen: list[dict[str, Any]] = []
    multi = [row for row in rows if row["box_count"] >= 2]
    dense = [row for row in rows if row["box_count"] >= 3]
    plan = [
        (lambda row: any(value < 10 for value in row["short_sides"]),
         "small target <10px"),
        (lambda row: any(value < 20 for value in row["short_sides"]),
         "small target 10-19px"),
        (lambda row: sum(1 for value in row["short_sides"] if 20 <= value < 40) >= 1,
         "target 20-39px"),
        (lambda row: sum(1 for value in row["short_sides"] if 20 <= value < 40) >= 2,
         "second target 20-39px"),
        (lambda row: any(value >= 40 for value in row["short_sides"]),
         "target >=40px"),
        (lambda row: row["box_count"] >= 2, "multi-label tile"),
        (lambda row: row["box_count"] >= 3, "tile with >=3 labels"),
        (lambda row: row["box_count"] == 1, "simple single-litter tile"),
        (lambda row: row["box_count"] >= 3, "dense multi-litter tile"),
    ]
    for predicate, reason in plan:
        found = first(predicate, reason, used)
        if found:
            used.add(found["row"]["tile_id"])
            chosen.append(found)
    # camera coverage: prefer the four dominant cameras, never drop 01027 by quota
    for camera in ("01021", "01022", "01028", "01030", "01027"):
        if any(row["row"]["camera_id"] == camera for row in chosen):
            continue
        found = first(lambda row, camera=camera: row["camera_id"] == camera,
                      f"camera coverage {camera}", used)
        if found:
            used.add(found["row"]["tile_id"])
            chosen.append(found)
    # fill deterministically, preferring multi-label variety; never trim the picks above
    pool = sorted(rows, key=lambda row: (-row["box_count"], row["camera_id"],
                                         row["tile_id"]))
    for row in pool:
        if len(chosen) >= TINY_POSITIVE_TARGET:
            break
        if row["tile_id"] in used:
            continue
        used.add(row["tile_id"])
        chosen.append({"reason": "fill (multi-label first)", "row": row})
    return chosen


def _negative_requirements(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    hard = [row for row in rows
            if row["hardness_source"] != EASY_NEGATIVE_HARDNESS]
    easy = [row for row in rows
            if row["hardness_source"] == EASY_NEGATIVE_HARDNESS]
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for camera in ("01021", "01022", "01028", "01030", "01027"):
        for row in hard:
            if row["tile_id"] in used or row["camera_id"] != camera:
                continue
            used.add(row["tile_id"])
            chosen.append({"reason": f"hard negative, camera {camera}", "row": row})
            break
    for row in hard:
        if len(chosen) >= TINY_NEGATIVE_TARGET:
            break
        if row["tile_id"] in used:
            continue
        used.add(row["tile_id"])
        chosen.append({"reason": "hard negative fill", "row": row})
    max_easy = int(TINY_NEGATIVE_TARGET * MAX_EASY_NEGATIVE_FRACTION)
    for row in easy:
        if len(chosen) >= TINY_NEGATIVE_TARGET or max_easy <= 0:
            break
        if row["tile_id"] in used:
            continue
        used.add(row["tile_id"])
        chosen.append({"reason": "easy background (<=30%)", "row": row})
        max_easy -= 1
    return chosen


def select_subsets(pools: Mapping[str, Any]) -> dict[str, Any]:
    positives = _positive_requirements(pools["positives"])
    negatives = _negative_requirements(pools["negatives"])
    if not (TINY_MIN <= len(positives) <= TINY_MAX):
        raise Step2AError(f"positive subset has {len(positives)} tiles, outside "
                          f"{TINY_MIN}-{TINY_MAX}")
    if not (TINY_MIN <= len(negatives) <= TINY_MAX):
        raise Step2AError(f"negative subset has {len(negatives)} tiles, outside "
                          f"{TINY_MIN}-{TINY_MAX}")
    return {
        "seed": SEED,
        "selection": "deterministic: requirement coverage first, stable tie-break",
        "positives": positives, "negatives": negatives,
    }


def subset_manifest(pools: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    positive_rows = [entry["row"] for entry in subsets["positives"]]
    negative_rows = [entry["row"] for entry in subsets["negatives"]]
    buckets: dict[str, int] = {}
    for row in positive_rows:
        for bucket in row["buckets"]:
            buckets[bucket] = buckets.get(bucket, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "note": (f"{DATA_YAML_NOTE}: the tiny subset is used for training and for the "
                 f"overfit check only; it is not a generalisation measurement"),
        "seed": SEED,
        "selection": subsets["selection"],
        "positive_tile_ids": [row["tile_id"] for row in positive_rows],
        "negative_tile_ids": [row["tile_id"] for row in negative_rows],
        "positive": [{
            "tile_id": row["tile_id"], "camera_id": row["camera_id"],
            "image_path": row["image_path"], "label_path": row["label_path"],
            "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"],
            "box_count": row["box_count"], "boxes": row["boxes"],
            "short_sides": row["short_sides"], "buckets": row["buckets"],
            "reason": entry["reason"],
        } for entry, row in zip(subsets["positives"], positive_rows)],
        "negative": [{
            "tile_id": row["tile_id"], "camera_id": row["camera_id"],
            "image_path": row["image_path"], "label_path": row["label_path"],
            "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"],
            "label_bytes": row["label_bytes"], "hardness_source": row["hardness_source"],
            "reason": entry["reason"],
        } for entry, row in zip(subsets["negatives"], negative_rows)],
        "coverage": {
            "positive_cameras": sorted({row["camera_id"] for row in positive_rows}),
            "negative_cameras": sorted({row["camera_id"] for row in negative_rows}),
            "positive_tiles": len(positive_rows),
            "negative_tiles": len(negative_rows),
            "positive_bboxes": sum(row["box_count"] for row in positive_rows),
            "multi_label_tiles": sum(1 for row in positive_rows if row["box_count"] >= 2),
            "three_plus_label_tiles": sum(1 for row in positive_rows
                                          if row["box_count"] >= 3),
            "short_side_buckets": dict(sorted(buckets.items())),
            "easy_negative_count": sum(1 for row in negative_rows
                                       if row["hardness_source"]
                                       == EASY_NEGATIVE_HARDNESS),
        },
        "source_hashes": {
            "positive_training_manifest_sha256": pools["positive_manifest_sha256"],
            "hard_negative_training_manifest_sha256": pools["negative_manifest_sha256"],
        },
    }


# --------------------------------------------------------------------------- #
# staging (§6/§11)
# --------------------------------------------------------------------------- #


def stage_dataset(manifest: Mapping[str, Any], dataset_dir: Path | str, *,
                  mode: str = "hardlink") -> dict[str, Any]:
    """Build the YOLO dataset view without touching the frozen bytes."""
    dataset = Path(dataset_dir)
    images = dataset / "images" / "train"
    labels = dataset / "labels" / "train"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for kind in ("positive", "negative"):
        for entry in manifest[kind]:
            source_image = Path(entry["image_path"])
            source_label = Path(entry["label_path"])
            target_image = images / f"{entry['tile_id']}.png"
            target_label = labels / f"{entry['tile_id']}.txt"
            for source, target in ((source_image, target_image),
                                   (source_label, target_label)):
                if target.exists():
                    target.unlink()
                if mode == "hardlink":
                    try:
                        os.link(source, target)
                    except OSError:                     # different filesystem
                        shutil.copy2(source, target)
                elif mode == "copy":
                    shutil.copy2(source, target)
                elif mode == "symlink":
                    target.symlink_to(source)
                else:
                    raise Step2AError(f"unknown staging mode {mode!r}")
            rows.append({
                "tile_id": entry["tile_id"], "kind": kind,
                "image_path": str(target_image), "label_path": str(target_label),
                "image_sha256_source": entry["image_sha256"],
                "label_sha256_source": entry["label_sha256"],
                "image_sha256_staged": sha256_file(target_image),
                "label_sha256_staged": sha256_file(target_label),
                "label_bytes_staged": target_label.stat().st_size,
            })
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        f"# {DATA_YAML_NOTE}\n"
        f"# TRAIN_EQUALS_VAL = {str(TRAIN_EQUALS_VAL).lower()} -- do not use as a metric\n"
        f"path: {dataset.resolve()}\n"
        f"train: images/train\n"
        f"val: images/train\n"
        f"names:\n  0: ground_litter\n", encoding="utf-8")
    return {"dataset_dir": str(dataset), "data_yaml": str(data_yaml),
            "mode": mode, "files": rows}


def verify_staging(report: Mapping[str, Any]) -> dict[str, Any]:
    image_mismatch, label_mismatch, negative_not_empty = [], [], []
    for row in report["files"]:
        if row["image_sha256_source"] != row["image_sha256_staged"]:
            image_mismatch.append(row["tile_id"])
        if row["label_sha256_source"] != row["label_sha256_staged"]:
            label_mismatch.append(row["tile_id"])
        if row["kind"] == "negative" and row["label_bytes_staged"] != 0:
            negative_not_empty.append(row["tile_id"])
    return {"files": len(report["files"]),
            "image_mismatch": image_mismatch, "label_mismatch": label_mismatch,
            "negative_label_not_empty": negative_not_empty,
            "image_bytes_identical": not image_mismatch,
            "label_bytes_identical": not label_mismatch,
            "negative_labels_empty": not negative_not_empty,
            "ok": not (image_mismatch or label_mismatch or negative_not_empty)}


# --------------------------------------------------------------------------- #
# evaluation (§22/§25/§33-§35)
# --------------------------------------------------------------------------- #


def evaluate_predictions(manifest: Mapping[str, Any],
                         predictions_by_image: Mapping[str, Any], *,
                         conf_levels: Sequence[float] = CONF_LEVELS
                         ) -> dict[str, Any]:
    """Positive GT proposal recall, image-level hits and negative FP counts."""
    per_conf: dict[str, Any] = {}
    for conf in conf_levels:
        key = f"{conf:.2f}"
        tp = total_gt = 0
        hit_images = 0
        buckets: dict[str, dict[str, int]] = {}
        label_counts: dict[str, dict[str, int]] = {}
        matched_detail: list[dict[str, Any]] = []
        for entry in manifest["positive"]:
            gt_boxes = [list(box) for box in (entry.get("boxes") or [])]
            predictions = [p for p in (predictions_by_image.get(entry["tile_id"]) or [])
                           if float(p.get("score") or 0.0) >= conf]
            result = match_predictions(gt_boxes, predictions)
            tp += result["tp"]
            total_gt += len(gt_boxes)
            if result["tp"] > 0:
                hit_images += 1
            for index, box in enumerate(gt_boxes):
                short = min(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))
                bucket = buckets.setdefault(size_bucket(short), {"gt": 0, "tp": 0})
                bucket["gt"] += 1
                if index in result["matched_gt"]:
                    bucket["tp"] += 1
            group = ("single" if len(gt_boxes) == 1 else
                     ("multi" if len(gt_boxes) >= 2 else "none"))
            slot = label_counts.setdefault(group, {"images": 0, "hit_images": 0,
                                                   "gt": 0, "tp": 0})
            slot["images"] += 1
            slot["hit_images"] += 1 if result["tp"] > 0 else 0
            slot["gt"] += len(gt_boxes)
            slot["tp"] += result["tp"]
            matched_detail.append({"tile_id": entry["tile_id"], "gt": len(gt_boxes),
                                   "tp": result["tp"], "fn": result["fn"],
                                   "predictions": len(predictions),
                                   "iou": result["iou"]})
        negative_predictions = 0
        fp_images = 0
        per_negative: list[dict[str, Any]] = []
        for entry in manifest["negative"]:
            predictions = [p for p in (predictions_by_image.get(entry["tile_id"]) or [])
                           if float(p.get("score") or 0.0) >= conf]
            negative_predictions += len(predictions)
            if predictions:
                fp_images += 1
            per_negative.append({"tile_id": entry["tile_id"],
                                 "predictions": len(predictions)})
        per_conf[key] = {
            "conf": conf,
            "positive_gt_proposal_recall":
                round(tp / total_gt, 4) if total_gt else 0.0,
            "positive_gt_total": total_gt, "positive_gt_tp": tp,
            "positive_image_hit_rate":
                round(hit_images / len(manifest["positive"]), 4)
                if manifest["positive"] else 0.0,
            "positive_image_hits": hit_images,
            "positive_image_total": len(manifest["positive"]),
            "negative_total_predictions": negative_predictions,
            "negative_fp_images": fp_images,
            "negative_fp_image_rate":
                round(fp_images / len(manifest["negative"]), 4)
                if manifest["negative"] else 0.0,
            "negative_fp_per_image":
                round(negative_predictions / len(manifest["negative"]), 3)
                if manifest["negative"] else 0.0,
            "by_size_bucket": {bucket: {"gt": value["gt"], "tp": value["tp"],
                                        "recall": round(value["tp"] / value["gt"], 4)
                                        if value["gt"] else 0.0}
                               for bucket, value in sorted(buckets.items())},
            "by_label_count": {group: {
                **value,
                "proposal_recall": round(value["tp"] / value["gt"], 4)
                if value["gt"] else 0.0} for group, value in sorted(label_counts.items())},
            "per_positive_image": matched_detail,
            "per_negative_image": per_negative,
        }
    return {"conf_levels": list(conf_levels), "per_conf": per_conf,
            "matching": {"iou_normal": IOU_NORMAL, "iou_small": IOU_SMALL,
                         "small_gt_short_side_px": SMALL_GT_SHORT_SIDE_PX,
                         "small_expand": SMALL_EXPAND,
                         "small_area_ratio": [SMALL_AREA_RATIO_MIN,
                                              SMALL_AREA_RATIO_MAX],
                         "one_to_one": True}}


def overfit_verdict(post: Mapping[str, Any], *, loader_ok: bool,
                    training_ok: bool, nan_free: bool,
                    loss_decreased: bool) -> dict[str, Any]:
    """§23: training sanity only -- never a Step 0B pass/fail."""
    recall = float(post["per_conf"]["0.01"]["positive_gt_proposal_recall"])
    hit = float(post["per_conf"]["0.01"]["positive_image_hit_rate"])
    if not (loader_ok and training_ok and nan_free and loss_decreased):
        return {"verdict": "OVERFIT_FAIL",
                "reason": "pipeline precondition failed",
                "loader_ok": loader_ok, "training_ok": training_ok,
                "nan_free": nan_free, "loss_decreased": loss_decreased}
    if recall >= 0.90 and hit >= 0.90:
        verdict = "TRAINING_PIPELINE_PASS"
    elif recall >= 0.70:
        verdict = "PARTIAL_OVERFIT"
    else:
        verdict = "OVERFIT_FAIL"
    return {"verdict": verdict, "positive_gt_proposal_recall_at_0.01": recall,
            "positive_image_hit_rate_at_0.01": hit, "loader_ok": loader_ok,
            "training_ok": training_ok, "nan_free": nan_free,
            "loss_decreased": loss_decreased}


def boundary_flags() -> dict[str, bool]:
    return {
        "development_accessed": False,
        "sealed_accessed": False,
        "upstream_dataset_bytes_modified": False,
        "pretrained_weight_modified": False,
        "labels_rewritten": False,
        "negative_label_changed": False,
        "augmentation_added_to_data": False,
        "threshold_tuned": False,
        "step2b_started": False,
    }


def build_summary(pools: Mapping[str, Any], preflight: Mapping[str, Any],
                  environment: Mapping[str, Any], manifest: Mapping[str, Any],
                  staging: Mapping[str, Any], loader: Mapping[str, Any] | None,
                  baseline: Mapping[str, Any] | None = None,
                  post: Mapping[str, Any] | None = None,
                  training: Mapping[str, Any] | None = None,
                  verdict: Mapping[str, Any] | None = None) -> dict[str, Any]:
    coverage = manifest["coverage"]
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "environment": dict(environment),
        "preflight": dict(preflight),
        "tiny_dataset": {
            "positive_tiles": coverage["positive_tiles"],
            "negative_tiles": coverage["negative_tiles"],
            "positive_bboxes": coverage["positive_bboxes"],
            "positive_cameras": coverage["positive_cameras"],
            "negative_cameras": coverage["negative_cameras"],
            "short_side_buckets": coverage["short_side_buckets"],
            "multi_label_tiles": coverage["multi_label_tiles"],
            "three_plus_label_tiles": coverage["three_plus_label_tiles"],
            "easy_negative_count": coverage["easy_negative_count"],
            "data_yaml": staging.get("data_yaml"),
            "train_equals_val": TRAIN_EQUALS_VAL,
            "note": DATA_YAML_NOTE,
        },
        "loader_sanity": dict(loader or {"status": "not_run"}),
        "baseline": baseline, "post": post, "training": training,
        "verdict": dict(verdict or {"verdict": "NOT_EVALUATED"}),
        "boundaries": boundary_flags(),
        "input_context": {
            "positive_root": str(pools["positive_root"]),
            "negative_root": str(pools["negative_root"]),
        },
    }


def build_manifest(pools: Mapping[str, Any], summary: Mapping[str, Any], *,
                   code_commit: str, generated_at: str, config: Mapping[str, Any],
                   artifact_root: Path, provenance: Mapping[str, Any]
                   ) -> dict[str, Any]:
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
        "counts": {
            "positive_images": summary["preflight"]["counts"]["positive_images"],
            "negative_images": summary["preflight"]["counts"]["negative_images"],
            "tiny_positive_tiles": summary["tiny_dataset"]["positive_tiles"],
            "tiny_negative_tiles": summary["tiny_dataset"]["negative_tiles"],
        },
        "matching_parameters": {
            "iou_normal": IOU_NORMAL, "iou_small": IOU_SMALL,
            "small_gt_short_side_px": SMALL_GT_SHORT_SIDE_PX,
            "small_expand": SMALL_EXPAND,
            "small_area_ratio": [SMALL_AREA_RATIO_MIN, SMALL_AREA_RATIO_MAX],
        },
        "upstream_provenance": dict(provenance),
        "boundaries": dict(summary["boundaries"]),
        "note": (f"{DATA_YAML_NOTE}: no generalisation claim, no Development / Sealed "
                 f"access, no threshold tuning"),
    }


def write_reports(output_dir: Path, *, tiny_manifest: Mapping[str, Any],
                  loader: Mapping[str, Any] | None, summary: Mapping[str, Any],
                  manifest: Mapping[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tiny_path = output_dir / "tiny_overfit_manifest.json"
    atomic_write_json(tiny_path, tiny_manifest)
    if loader is not None:
        atomic_write_json(output_dir / "loader_sanity.json", loader)
    summary_path = output_dir / "SUMMARY.json"
    atomic_write_json(summary_path, summary)
    payload = dict(manifest)
    payload["tiny_overfit_manifest_sha256"] = sha256_file(tiny_path)
    payload["summary_sha256"] = sha256_file(summary_path)
    manifest_path = output_dir / "MANIFEST.json"
    atomic_write_json(manifest_path, payload)
    return {"tiny_overfit_manifest": str(tiny_path), "summary": str(summary_path),
            "manifest": str(manifest_path)}
