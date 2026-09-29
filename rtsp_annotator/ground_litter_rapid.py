"""Ground Litter Rapid Eval + Active Learning v1 — pure logic.

This module is deliberately free of torch/ultralytics/cv2 imports.  It holds the frozen
contract shared by the split builder, the frame extractor, the baseline predictor, the
review UI server and the Phase-2 evaluator:

* the deterministic Rapid-Train / Rapid-Eval Holdout split over the four currently usable
  Development cameras (01021/01022/01027/01030, 13 PS each = 52 PS),
* the fixed 5-offset frame grid (30/90/150/210/270 s) and bonus train frames,
* point-truth based matching (Required point hit / miss / FP),
* the threshold selection rule, and
* the hard guard that refuses to export any Rapid-Eval Holdout sample into training.

Nothing here may touch Sealed, the official 8801 artifact or the Step 2B checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Frozen contract
# --------------------------------------------------------------------------- #

SCHEMA_VERSION = "ground-litter-rapid-v1"

#: Fixed split seed.  Changing it invalidates the frozen split and every comparison.
SEED = "ground-litter-rapid-v1-20260929"

#: Only the four cameras that are usable in this round.  01028 stays deferred (occlusion).
CAMERAS = ("01021", "01022", "01027", "01030")

#: Fixed per-PS frame offsets relative to the start of the PS recording.
FRAME_OFFSETS = (30, 90, 150, 210, 270)

#: Each camera contributes this many PS to Rapid-Train and to Rapid-Eval Holdout.
TRAIN_PS_PER_CAMERA = 10
EVAL_PS_PER_CAMERA = 3

#: Rapid v1 fixed threshold sweep (baseline and V2 share it).
THRESHOLD_GRID = (0.01, 0.03, 0.05, 0.10, 0.20, 0.30)

#: Inference chain constants (must stay identical for baseline V1 and V2).
TILE = 640
STRIDE = 512
PROPOSAL_FLOOR = 0.01
TOP_K_PER_FRAME = 100
NMS_IOU = 0.50
SOURCE_WIDTH = 2560
SOURCE_HEIGHT = 1440
FPS = 25.0

#: Frozen Step 2B checkpoint — the Rapid Eval v1 baseline model.  `best.pt` is forbidden.
STEP2B_LAST_SHA256 = "4852392aeae9a68669a50752eb1f7466fbcc9a426524faba86fb20a3b6351a94"
STEP2B_LAST_PATH = "/home/sf01/step2b-20260923/out/runs/full_finetune/weights/last.pt"

#: Turhancan semantic model, located by SHA only.  Never downloaded, never used to grade.
SEMANTIC_SHA256 = "a2f8de0c7f714e2ab8b70c62490e2a41fd4a6681ca4a8dd442797c809a140278"

#: Paths that must never be written by Rapid v1 code.
FORBIDDEN_WRITE_ROOTS = (
    "/home/sf01/step2c1-blind-truth",
    "/home/sf01/step2b-20260923",
)

#: Any Development-relative path containing one of these tokens is out of bounds.
FORBIDDEN_PATH_TOKENS = ("sealed", "sealed_test")

TRUTH_CLASSES = ("REQUIRED_LITTER", "IGNORE_SMALL", "UNCERTAIN")

#: Substring that marks every Rapid-Eval Holdout sample.
EVAL_SPLIT = "rapid_eval"
TRAIN_SPLIT = "rapid_train"

SIZE_BUCKETS = ((0, 10, "<10"), (10, 20, "10-19"), (20, 40, "20-39"),
                (40, 80, "40-79"), (80, 10**9, "80+"))


class RapidError(RuntimeError):
    """A Rapid v1 contract violation.  Always hard-fails, never degrades silently."""


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload) -> str:
    """Stable JSON for hashing: sorted keys, no insignificant whitespace."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(payload) -> str:
    return sha256_text(canonical_json(payload))


def selection_hash(seed: str, split: str, file_id: str) -> str:
    """Per-row deterministic ranking hash.  The split is frozen by this value alone."""
    return sha256_text(f"{seed}|{split}|{file_id}")


# --------------------------------------------------------------------------- #
# Safety guards
# --------------------------------------------------------------------------- #


def assert_development_asset(path: str | Path) -> Path:
    """Refuse any path that is not a plain Development asset.

    Sealed material and the official Step 2C artifact are out of bounds for Rapid v1.
    """
    resolved = Path(path).expanduser()
    lowered = str(resolved).lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in lowered:
            raise RapidError(f"refusing sealed/forbidden path: {resolved}")
    if "/step2c1-blind-truth/artifact" in str(resolved):
        raise RapidError(f"refusing official artifact path: {resolved}")
    return resolved


def assert_writable_root(path: str | Path) -> Path:
    """Every Rapid v1 write must land under this root.

    Both the literal (``normpath``) and the symlink-resolved spelling are checked: the
    policy is about the path the operator wrote, and ``resolve()`` alone would let a
    symlinked ``/home`` on a development machine hide a forbidden prefix.
    """
    resolved = Path(path).expanduser().resolve()
    spellings = {os.path.normpath(str(Path(path).expanduser())),
                 os.path.normpath(str(resolved))}
    for forbidden in FORBIDDEN_WRITE_ROOTS:
        for spelling in spellings:
            if spelling == forbidden or spelling.startswith(forbidden + os.sep):
                raise RapidError(f"refusing to write inside {forbidden}: {spelling}")
    return resolved


def assert_trainable(split: str, *, sample_id: str = "") -> None:
    """Hard guard: Rapid-Eval Holdout frames may never enter any training export."""
    if split != TRAIN_SPLIT:
        raise RapidError(
            f"refuse training export for split={split!r} sample={sample_id!r}: "
            f"only {TRAIN_SPLIT} may be exported"
        )


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #


def rank_key(seed: str, file_id: str) -> str:
    return sha256_text(f"{seed}|{file_id}")


def build_split(files: list[dict], prior_inference_file_ids, seed: str = SEED) -> dict:
    """Deterministically freeze the Rapid-Train / Rapid-Eval Holdout split.

    ``files`` are Development manifest rows for :data:`CAMERAS`.  Any PS already used by
    the earlier exploratory detector probe is forced into Rapid-Train, because a
    previously-inferred PS is no longer a clean holdout.
    """
    prior = {str(x) for x in prior_inference_file_ids}
    by_camera: dict[str, list[dict]] = {camera: [] for camera in CAMERAS}
    seen: set[str] = set()
    for row in files:
        camera = row["camera_id"]
        if camera not in by_camera:
            continue
        file_id = row["file_id"]
        if file_id in seen:
            raise RapidError(f"duplicate file_id in manifest: {file_id}")
        seen.add(file_id)
        by_camera[camera].append(row)
    for camera in CAMERAS:
        rows = by_camera[camera]
        if len(rows) != TRAIN_PS_PER_CAMERA + EVAL_PS_PER_CAMERA:
            raise RapidError(
                f"camera {camera}: expected {TRAIN_PS_PER_CAMERA + EVAL_PS_PER_CAMERA} PS, "
                f"got {len(rows)}"
            )
        rows.sort(key=lambda r: r["file_id"])

    unknown = prior - seen
    if unknown:
        raise RapidError(f"prior-inference PS not present in manifest: {sorted(unknown)}")

    out_rows: list[dict] = []
    per_camera: dict[str, dict] = {}
    for camera in CAMERAS:
        rows = by_camera[camera]
        forced = [r for r in rows if r["file_id"] in prior]
        remaining = [r for r in rows if r["file_id"] not in prior]
        remaining.sort(key=lambda r: (rank_key(seed, r["file_id"]), r["file_id"]))
        if len(remaining) < EVAL_PS_PER_CAMERA:
            raise RapidError(f"camera {camera}: not enough unbiased PS for a holdout")
        eval_rows = remaining[:EVAL_PS_PER_CAMERA]
        train_rows = forced + remaining[EVAL_PS_PER_CAMERA:]
        train_rows.sort(key=lambda r: r["file_id"])
        if len(train_rows) != TRAIN_PS_PER_CAMERA:
            raise RapidError(f"camera {camera}: train={len(train_rows)}")
        for row in train_rows + eval_rows:
            split = TRAIN_SPLIT if row in train_rows else EVAL_SPLIT
            forced_flag = row["file_id"] in prior
            out_rows.append({
                "camera_id": camera,
                "file_id": row["file_id"],
                "file_name": row.get("file_name"),
                "ps_path": row.get("ps_path"),
                "source_sha256": row.get("sha256"),
                "canvas_size": list(row.get("canvas_size") or (SOURCE_WIDTH, SOURCE_HEIGHT)),
                "roi": row.get("roi"),
                "roi_geometry_version": row.get("roi_geometry_version"),
                "scene_version": row.get("scene_version"),
                "record_start": row.get("record_start"),
                "record_end": row.get("record_end"),
                "duration_seconds": row.get("duration_seconds"),
                "split": split,
                "selection_hash": selection_hash(seed, split, row["file_id"]),
                "rank_key": rank_key(seed, row["file_id"]),
                "excluded_from_holdout_due_prior_inference": forced_flag,
                "selection_reason": (
                    "prior_inference_forced_rapid_train" if forced_flag
                    else ("rapid_train_deterministic" if split == TRAIN_SPLIT
                          else "rapid_eval_holdout_deterministic")
                ),
            })

    out_rows.sort(key=lambda r: (r["camera_id"], r["split"], r["file_id"]))
    counts = {TRAIN_SPLIT: sum(r["split"] == TRAIN_SPLIT for r in out_rows),
              EVAL_SPLIT: sum(r["split"] == EVAL_SPLIT for r in out_rows)}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "cameras": list(CAMERAS),
        "scope_note": "Only the four currently usable cameras 01021/01022/01027/01030. "
                      "01028 is deferred (severe occlusion) and is never scored here.",
        "frame_offsets_seconds": list(FRAME_OFFSETS),
        "fps": FPS,
        "canvas_size": [SOURCE_WIDTH, SOURCE_HEIGHT],
        "prior_inference_source": "output/fourcam-exploratory-20260929/selection.json",
        "excluded_from_holdout_due_prior_inference": sorted(prior),
        "split_algorithm": (
            "per camera: PS already used by the exploratory detector probe are forced into "
            "rapid_train; the remaining PS are ranked by sha256(seed|file_id) ascending and "
            "the first 3 become rapid_eval holdout; the rest complete rapid_train"
        ),
        "split_frozen": True,
        "rows": out_rows,
        "counts": counts,
        "per_camera": {
            camera: {
                "rapid_train": sum(1 for r in out_rows
                                   if r["camera_id"] == camera and r["split"] == TRAIN_SPLIT),
                "rapid_eval": sum(1 for r in out_rows
                                  if r["camera_id"] == camera and r["split"] == EVAL_SPLIT),
                "forced_to_train": sum(1 for r in out_rows
                                       if r["camera_id"] == camera
                                       and r["excluded_from_holdout_due_prior_inference"]),
            }
            for camera in CAMERAS
        },
    }
    payload["split_sha256"] = canonical_sha256(payload["rows"])
    return payload


def verify_split(split: dict) -> None:
    """Re-derive and re-check a frozen split.  Any drift is a hard failure."""
    rows = split.get("rows") or []
    if canonical_sha256(rows) != split.get("split_sha256"):
        raise RapidError("split_sha256 mismatch: split.json was modified after freezing")
    counts = split.get("counts") or {}
    if counts.get(TRAIN_SPLIT) != TRAIN_PS_PER_CAMERA * len(CAMERAS):
        raise RapidError(f"unexpected rapid_train PS count: {counts.get(TRAIN_SPLIT)}")
    if counts.get(EVAL_SPLIT) != EVAL_PS_PER_CAMERA * len(CAMERAS):
        raise RapidError(f"unexpected rapid_eval PS count: {counts.get(EVAL_SPLIT)}")
    for camera in CAMERAS:
        per = split["per_camera"][camera]
        if per["rapid_train"] != TRAIN_PS_PER_CAMERA or per["rapid_eval"] != EVAL_PS_PER_CAMERA:
            raise RapidError(f"camera {camera} split imbalance: {per}")
    seen: set[str] = set()
    for row in rows:
        if row["file_id"] in seen:
            raise RapidError(f"PS appears twice in split: {row['file_id']}")
        seen.add(row["file_id"])
        if row["excluded_from_holdout_due_prior_inference"] and row["split"] != TRAIN_SPLIT:
            raise RapidError(f"prior-inference PS not in rapid_train: {row['file_id']}")
        expected = selection_hash(split["seed"], row["split"], row["file_id"])
        if row["selection_hash"] != expected:
            raise RapidError(f"selection hash drift for {row['file_id']}")


def split_by_file_id(split: dict) -> dict[str, dict]:
    return {row["file_id"]: row for row in split["rows"]}


# --------------------------------------------------------------------------- #
# Frame grid
# --------------------------------------------------------------------------- #


def frame_id(camera_id: str, file_id: str, offset_seconds: int) -> str:
    return f"{camera_id}_{file_id}_t{int(offset_seconds):03d}"


def bonus_frame_id(camera_id: str, file_id: str, frame_index: int) -> str:
    return f"{camera_id}_{file_id}_bonus_f{int(frame_index)}"


def frame_index_for(decoded_relative_seconds: float, fps: float = FPS) -> int:
    """Official-discovery frame index convention: round(seconds * fps)."""
    return int(round(float(decoded_relative_seconds) * float(fps)))


def build_frame_manifest(split: dict, bonus_frames: list[dict] | None = None) -> dict:
    """The fixed review grid: 5 frames per PS plus de-duplicated bonus train frames."""
    by_file = split_by_file_id(split)
    rows: list[dict] = []
    for row in split["rows"]:
        for offset in FRAME_OFFSETS:
            rows.append({
                "frame_id": frame_id(row["camera_id"], row["file_id"], offset),
                "camera_id": row["camera_id"],
                "file_id": row["file_id"],
                "split": row["split"],
                "kind": "fixed",
                "offset_seconds": int(offset),
                "requested_relative_seconds": float(offset),
                "nominal_frame_index": frame_index_for(offset),
            })
    fixed_ids = {r["frame_id"] for r in rows}
    bonus_rows: list[dict] = []
    bonus_by_id: dict[str, dict] = {}
    for bonus in bonus_frames or []:
        file_id = bonus["file_id"]
        if file_id not in by_file:
            raise RapidError(f"bonus frame on unknown PS: {file_id}")
        row = by_file[file_id]
        if row["split"] != TRAIN_SPLIT:
            raise RapidError(f"bonus frame must be rapid_train: {file_id}")
        index = int(bonus["frame_index"])
        identifier = bonus_frame_id(row["camera_id"], file_id, index)
        if identifier in fixed_ids:
            continue
        if identifier in bonus_by_id:
            # Same source frame reached from several truth marks: merge, never duplicate.
            existing = bonus_by_id[identifier]
            for truth_id in bonus.get("truth_ids") or []:
                if truth_id not in existing["inherited_truth_ids"]:
                    existing["inherited_truth_ids"].append(truth_id)
            continue
        seconds = float(bonus["decoded_timestamp"])
        row_payload = {
            "frame_id": identifier,
            "camera_id": row["camera_id"],
            "file_id": file_id,
            "split": TRAIN_SPLIT,
            "kind": "bonus_train",
            "offset_seconds": None,
            "requested_relative_seconds": seconds,
            "nominal_frame_index": index,
            "bonus_reason": bonus.get("reason", "existing_human_required_point"),
            "inherited_truth_ids": list(bonus.get("truth_ids") or []),
        }
        bonus_by_id[identifier] = row_payload
        bonus_rows.append(row_payload)
    rows.extend(bonus_rows)
    rows.sort(key=lambda r: (r["camera_id"], r["file_id"],
                             0 if r["kind"] == "fixed" else 1,
                             r["requested_relative_seconds"]))
    counts = {
        "fixed_train": sum(r["split"] == TRAIN_SPLIT and r["kind"] == "fixed" for r in rows),
        "fixed_eval": sum(r["split"] == EVAL_SPLIT and r["kind"] == "fixed" for r in rows),
        "bonus_train": sum(r["kind"] == "bonus_train" for r in rows),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": split["seed"],
        "split_sha256": split["split_sha256"],
        "offsets_seconds": list(FRAME_OFFSETS),
        "frames": rows,
        "counts": counts,
    }
    payload["manifest_sha256"] = canonical_sha256(rows)
    return payload


def target_seconds_for(manifest_row: dict) -> float:
    """Decoded-relative target time for a fixed frame; bonus frames carry their own."""
    if manifest_row["kind"] == "fixed":
        return float(manifest_row["offset_seconds"])
    return float(manifest_row["requested_relative_seconds"])


# --------------------------------------------------------------------------- #
# Inference geometry
# --------------------------------------------------------------------------- #


def tile_starts(length: int, tile: int = TILE, stride: int = STRIDE) -> list[int]:
    """Tile origins with full right/bottom coverage (identical to the frozen chain)."""
    if length < tile:
        raise RapidError(f"source axis {length} smaller than tile {tile}")
    return sorted(set(list(range(0, length - tile + 1, stride)) + [length - tile]))


def roi_mask_polygon(roi, width: int = SOURCE_WIDTH, height: int = SOURCE_HEIGHT):
    return [(round(float(x) * width), round(float(y) * height)) for x, y in roi]


def point_inside_roi(x: float, y: float, roi, width: int = SOURCE_WIDTH,
                     height: int = SOURCE_HEIGHT) -> bool:
    polygon = roi_mask_polygon(roi, width, height)
    inside = False
    count = len(polygon)
    for index in range(count):
        x0, y0 = polygon[index]
        x1, y1 = polygon[(index + 1) % count]
        if (y0 > y) != (y1 > y):
            x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < x_cross:
                inside = not inside
    return inside


def box_iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_contains_point(box, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def box_short_side(box) -> float:
    return min(abs(box[2] - box[0]), abs(box[3] - box[1]))


def size_bucket(short_side: float) -> str:
    for low, high, label in SIZE_BUCKETS:
        if low <= short_side < high:
            return label
    return SIZE_BUCKETS[-1][2]


def nms(rows: list[dict], iou: float = NMS_IOU) -> list[dict]:
    """Class-wise greedy NMS, highest confidence first (frozen chain behaviour)."""
    keep: list[dict] = []
    for row in sorted(rows, key=lambda r: r["confidence"], reverse=True):
        ok = True
        for old in keep:
            if row.get("class_id", 0) != old.get("class_id", 0):
                continue
            if box_iou(row["xyxy"], old["xyxy"]) > iou:
                ok = False
                break
        if ok:
            keep.append(row)
    return keep


# --------------------------------------------------------------------------- #
# Matching and metrics (point truth, not full GT boxes)
# --------------------------------------------------------------------------- #


def match_points(points: list[dict], predictions: list[dict],
                 reviews: dict[str, str]) -> dict:
    """One-to-one matching of Required point truth against user-judged predictions.

    ``reviews`` maps ``prediction_id`` -> verdict in {Y, N, X, M}.  A Required point is a
    hit when some prediction was judged ``Y`` and its box contains the point, and no other
    Required point claimed that prediction first (one-to-one).  ``X`` predictions cover
    IGNORE_SMALL / UNCERTAIN truth and count neither as hit nor as FP.
    """
    required = [p for p in points if p["truth_class"] == "REQUIRED_LITTER"]
    ignore = [p for p in points if p["truth_class"] in ("IGNORE_SMALL", "UNCERTAIN")]
    judged_y = [p for p in predictions if reviews.get(p["prediction_id"]) == "Y"]
    judged_n = [p for p in predictions if reviews.get(p["prediction_id"]) in ("N", "F")]
    judged_x = [p for p in predictions if reviews.get(p["prediction_id"]) == "X"]
    judged_m = [p for p in predictions if reviews.get(p["prediction_id"]) == "M"]

    candidates: list[tuple[float, str, str]] = []
    for point in required:
        for prediction in judged_y:
            if box_contains_point(prediction["xyxy"], point["source_xy"][0],
                                  point["source_xy"][1]):
                candidates.append((-float(prediction["confidence"]),
                                   point["truth_id"], prediction["prediction_id"]))
    candidates.sort()
    claimed_points: set[str] = set()
    claimed_predictions: set[str] = set()
    matches: list[dict] = []
    for _, truth_id, prediction_id in candidates:
        if truth_id in claimed_points or prediction_id in claimed_predictions:
            continue
        claimed_points.add(truth_id)
        claimed_predictions.add(prediction_id)
        matches.append({"truth_id": truth_id, "prediction_id": prediction_id})
    misses = [p["truth_id"] for p in required if p["truth_id"] not in claimed_points]
    return {
        "required_total": len(required),
        "required_hit": len(claimed_points),
        "required_miss": len(misses),
        "miss_truth_ids": misses,
        "ignore_total": len(ignore),
        "fp": len(judged_n),
        "fp_prediction_ids": [p["prediction_id"] for p in judged_n],
        "xfp_suppressed": len(judged_x),
        "m_prediction_ids": [p["prediction_id"] for p in judged_m],
        "matches": matches,
    }


def require_review_complete(review_state: dict, frame_ids: list[str]) -> None:
    """Any metric may only be computed on frames the operator explicitly completed."""
    missing = [fid for fid in frame_ids
               if not (review_state.get("frames", {}).get(fid, {}) or {}).get("truth_complete")]
    if missing:
        raise RapidError(
            f"refusing to compute metrics: {len(missing)} frame(s) lack truth_complete, "
            f"first={missing[0]}"
        )


def select_threshold(metrics_by_threshold: dict[str, dict]) -> dict:
    """Frozen selection rule (see README).

    1. highest Required hit rate;
    2. ties within 2 percentage points -> lower FP/100 frames;
    3. still tied -> higher threshold.
    """
    rows = []
    for key, payload in metrics_by_threshold.items():
        rows.append((float(key), payload))
    if not rows:
        raise RapidError("no threshold metrics to select from")
    rows.sort(key=lambda item: item[0])
    hits = {key: payload["required_hit_rate"] for key, payload in rows}
    best = max(hits.values())
    near = [(key, payload) for key, payload in rows if best - payload["required_hit_rate"] <= 0.02]
    near.sort(key=lambda item: (item[1]["fp_per_100_frames"], -item[0]))
    chosen = near[0]
    return {
        "selected_threshold": chosen[0],
        "rule": "max required_hit_rate; within 2pp prefer lower fp_per_100_frames; "
                "then higher threshold",
        "considered": {f"{k:.2f}": v for k, v in rows},
        "best_hit_rate": best,
        "tie_set": [k for k, _ in near],
    }


def summarise_metrics(per_frame: list[dict]) -> dict:
    """Aggregate per-frame match records into the Rapid v1 core indicators."""
    frames = len(per_frame)
    required = sum(r["required_total"] for r in per_frame)
    hit = sum(r["required_hit"] for r in per_frame)
    miss = sum(r["required_miss"] for r in per_frame)
    fp = sum(r["fp"] for r in per_frame)
    positive_frames = [r for r in per_frame if r["required_total"] > 0]
    positive_hit_frames = [r for r in positive_frames if r["required_hit"] > 0]
    predictions = sum(r.get("prediction_count", 0) for r in per_frame)
    return {
        "frames": frames,
        "required_points": required,
        "required_hit": hit,
        "required_miss": miss,
        "required_hit_rate": (hit / required) if required else None,
        "fp": fp,
        "fp_per_100_frames": (fp / frames * 100.0) if frames else None,
        "positive_frames": len(positive_frames),
        "positive_frame_hit": len(positive_hit_frames),
        "positive_frame_hit_rate": (len(positive_hit_frames) / len(positive_frames))
        if positive_frames else None,
        "predictions_total": predictions,
        "predictions_per_frame": (predictions / frames) if frames else None,
        "per_camera": _per_camera(per_frame),
    }


def _per_camera(per_frame: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in per_frame:
        grouped.setdefault(row.get("camera_id", "?"), []).append(row)
    out = {}
    for camera, rows in sorted(grouped.items()):
        required = sum(r["required_total"] for r in rows)
        hit = sum(r["required_hit"] for r in rows)
        positive = [r for r in rows if r["required_total"] > 0]
        out[camera] = {
            "frames": len(rows),
            "required_points": required,
            "required_hit": hit,
            "required_miss": sum(r["required_miss"] for r in rows),
            "required_hit_rate": (hit / required) if required else None,
            "fp": sum(r["fp"] for r in rows),
            "fp_per_100_frames": sum(r["fp"] for r in rows) / len(rows) * 100.0 if rows else None,
            "positive_frames": len(positive),
            "positive_frame_hit": sum(1 for r in positive if r["required_hit"] > 0),
        }
    return out


def decide_verdict(baseline: dict, v2: dict) -> dict:
    """Rapid decision rule.  Never a production / formal / Sealed result."""
    base_rate = baseline.get("required_hit_rate") or 0.0
    v2_rate = v2.get("required_hit_rate") or 0.0
    delta_pp = (v2_rate - base_rate) * 100.0
    base_fp = baseline.get("fp_per_100_frames") or 0.0
    v2_fp = v2.get("fp_per_100_frames") or 0.0
    if delta_pp < -3.0 or (base_fp > 0 and v2_fp > base_fp * 2.0 and v2_fp - base_fp > 20):
        verdict = "RAPID_V2_REGRESSION"
    elif delta_pp >= 10.0 and v2_fp <= max(base_fp + 20.0, base_fp * 2.0):
        verdict = "RAPID_ITERATION_POSITIVE"
    elif delta_pp < 5.0:
        verdict = "DATA_OR_IMAGING_BOTTLENECK"
    else:
        verdict = "INCONCLUSIVE_BETWEEN_5_AND_10PP"
    return {
        "verdict": verdict,
        "hit_rate_delta_pp": delta_pp,
        "baseline_hit_rate": base_rate,
        "v2_hit_rate": v2_rate,
        "baseline_fp_per_100_frames": base_fp,
        "v2_fp_per_100_frames": v2_fp,
        "note": "frame-level FP is NOT a production alert/day rate: Rapid v1 has no "
                "temporal event layer.",
    }


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def distance_to_box(box, x: float, y: float) -> float:
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)


# --------------------------------------------------------------------------- #
# Copy-previous-frame (near-duplicate review acceleration)
# --------------------------------------------------------------------------- #
#
# Many fixed grid frames are the same camera looking at the same motionless litter.
# "Copy previous" removes the re-clicking, but it must never remove the human check:
# the copied frame lands in COPIED_PENDING_CONFIRM and only an explicit confirm turns it
# into truth.  Rapid-Eval keeps every fixed frame in the denominator, exactly as before.

COPY_PENDING = "COPIED_PENDING_CONFIRM"
COPY_CONFIRMED = "CONFIRMED_COPY"

#: Two frames on the same camera may copy from each other when their recordings are the same
#: PS, or when their decoded times are this close (the review grid is 60 s apart inside a PS
#: and ~64 s across a PS boundary, so one PS length plus slack covers the continuous case).
CONTINUITY_MAX_GAP_SECONDS = 400.0

#: How far back in the review queue we are willing to look for a completed source frame.
COPY_SOURCE_LOOKBACK = 20

#: Candidate verdicts that mean "the operator confirmed this Required object is really there".
COPYABLE_TRUTH_CLASSES = TRUTH_CLASSES

#: Origins that record where a point came from.
ORIGIN_REVIEW = "rapid_v1_review"
ORIGIN_COPY = "copied_from_previous_frame"


def frame_absolute_seconds(frame_row: dict, split_row: dict) -> float | None:
    """Decoded wall-clock seconds of a frame, from the PS record start plus its offset.

    Only differences between frames matter, so the host timezone cancels out.
    """
    stamp = split_row.get("record_start")
    if not stamp:
        return None
    try:
        base = datetime.strptime(str(stamp), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    offset = frame_row.get("requested_relative_seconds")
    if offset is None:
        offset = frame_row.get("offset_seconds") or 0
    return base.timestamp() + float(offset)


def copy_eligibility(current: dict, previous: dict) -> dict:
    """May ``current`` copy truth from ``previous``?

    ``current``/``previous`` are ``{camera_id, file_id, absolute_seconds}``.
    Same camera is mandatory; then either the same PS, or a small time gap.
    """
    if current.get("camera_id") != previous.get("camera_id"):
        return {"allowed": False, "reason": "different_camera"}
    if current.get("file_id") == previous.get("file_id"):
        return {"allowed": True, "reason": "same_ps"}
    current_time = current.get("absolute_seconds")
    previous_time = previous.get("absolute_seconds")
    if current_time is None or previous_time is None:
        return {"allowed": False, "reason": "no_record_start"}
    gap = abs(float(current_time) - float(previous_time))
    if gap <= CONTINUITY_MAX_GAP_SECONDS:
        return {"allowed": True, "reason": "temporally_continuous",
                "gap_seconds": round(gap, 1)}
    return {"allowed": False, "reason": "time_gap_too_large", "gap_seconds": round(gap, 1)}


# --------------------------------------------------------------------------- #
# Near-duplicate cap (training export only)
# --------------------------------------------------------------------------- #
#
# This never changes the Rapid-Eval denominator, the frozen frame manifest, split.json or any
# recorded truth.  It only decides which already-reviewed Rapid-Train samples are worth
# exporting, so one motionless piece of litter observed in five near-identical frames does not
# become five training tiles.

NEAR_DUP_POSITION_RADIUS_PX = 96.0
NEAR_DUP_TIME_WINDOW_S = 600.0
NEAR_DUP_MAX_PER_CLUSTER = 2


def _cluster_key_ok(cluster: dict, sample: dict, radius: float, window: float) -> bool:
    if cluster["camera_id"] != sample["camera_id"]:
        return False
    if abs(cluster["time"] - sample["time"]) > window:
        return False
    dx = cluster["center"][0] - sample["center_xy"][0]
    dy = cluster["center"][1] - sample["center_xy"][1]
    return math.hypot(dx, dy) <= radius


def _representatives(members: list[dict], kind: str) -> list[dict]:
    """Pick at most two representatives from one near-duplicate cluster, deterministically."""
    ordered = sorted(members, key=lambda s: (s["time"], s["sample_id"]))
    if kind == "positive":
        # keep the most informative scale plus the earliest sighting, deduplicated
        largest = max(ordered, key=lambda s: (s.get("short_side") or 0.0,
                                              -s["time"], s["sample_id"]))
        keep = [ordered[0], largest]
    else:
        # keep the earliest and the latest so the negative still spans the recording
        keep = [ordered[0], ordered[-1]]
    seen: set[str] = set()
    result: list[dict] = []
    for sample in keep:
        if sample["sample_id"] in seen:
            continue
        seen.add(sample["sample_id"])
        result.append(sample)
    return result


def dedupe_near_duplicates(samples: list[dict], *, kind: str,
                           radius_px: float = NEAR_DUP_POSITION_RADIUS_PX,
                           time_window_s: float = NEAR_DUP_TIME_WINDOW_S,
                           max_per_cluster: int = NEAR_DUP_MAX_PER_CLUSTER) -> dict:
    """Greedy near-duplicate clustering over one camera's samples.

    A sample joins an existing cluster when it is on the same camera, within ``time_window_s``
    of the cluster's first sighting and within ``radius_px`` of the cluster centre — i.e. the
    same physical litter at nearly the same place in a short time span.
    """
    ordered = sorted(samples, key=lambda s: (s["camera_id"], s["time"], s["sample_id"]))
    clusters: list[dict] = []
    for sample in ordered:
        if not sample.get("center_xy"):
            raise RapidError(f'sample {sample.get("sample_id")} has no center_xy')
        sample = dict(sample)
        sample["time"] = float(sample["time"])
        target = None
        for cluster in clusters:
            if _cluster_key_ok(cluster, sample, radius_px, time_window_s):
                target = cluster
                break
        if target is None:
            clusters.append({"camera_id": sample["camera_id"], "time": sample["time"],
                             "center": list(sample["center_xy"]), "members": [sample]})
        else:
            target["members"].append(sample)

    kept: list[dict] = []
    dropped: list[dict] = []
    report: list[dict] = []
    for cluster in clusters:
        members = sorted(cluster["members"], key=lambda s: (s["time"], s["sample_id"]))
        chosen = _representatives(members, kind)
        if max_per_cluster < len(chosen):
            chosen = chosen[:max_per_cluster]
        chosen_ids = {s["sample_id"] for s in chosen}
        kept.extend(chosen)
        for sample in members:
            if sample["sample_id"] not in chosen_ids:
                dropped.append({**sample, "cluster_id": f'{cluster["camera_id"]}:'
                                                  f'{round(cluster["center"][0])}:'
                                                  f'{round(cluster["center"][1])}'})
        report.append({
            "cluster_id": f'{cluster["camera_id"]}:{round(cluster["center"][0])}:'
                          f'{round(cluster["center"][1])}',
            "camera_id": cluster["camera_id"],
            "center_xy": [round(v, 1) for v in cluster["center"]],
            "members": len(members),
            "kept": [s["sample_id"] for s in chosen],
            "dropped": [s["sample_id"] for s in members if s["sample_id"] not in chosen_ids],
        })
    kept.sort(key=lambda s: (s["camera_id"], s["time"], s["sample_id"]))
    return {
        "kind": kind,
        "inputs": len(samples),
        "kept": kept,
        "dropped": dropped,
        "clusters": sorted(report, key=lambda c: (c["camera_id"], c["cluster_id"])),
        "inputs_count": len(samples),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "cluster_count": len(clusters),
        "parameters": {"radius_px": radius_px, "time_window_s": time_window_s,
                       "max_per_cluster": max_per_cluster},
    }


def plan_training_export(positive_samples: list[dict], negative_samples: list[dict], *,
                         max_per_cluster: int = NEAR_DUP_MAX_PER_CLUSTER) -> dict:
    """Near-duplicate-capped Rapid-Train export plan.

    Refuses outright if any sample carries a Rapid-Eval Holdout frame: that split may never be
    exported, capped or otherwise.
    """
    for sample in list(positive_samples) + list(negative_samples):
        frame_id = sample.get("frame_id") or sample.get("sample_id") or "?"
        if sample.get("split") == EVAL_SPLIT:
            raise RapidError(f"refusing to export Rapid-Eval Holdout sample {frame_id}")
        assert_trainable(sample.get("split") or TRAIN_SPLIT, sample_id=frame_id)
    positives = dedupe_near_duplicates(positive_samples, kind="positive",
                                       max_per_cluster=max_per_cluster)
    negatives = dedupe_near_duplicates(negative_samples, kind="hard_negative",
                                       max_per_cluster=max_per_cluster)
    return {
        "schema_version": SCHEMA_VERSION,
        "note": "training-export view only; the Rapid-Eval Holdout denominator, the frozen "
                "frame manifest, split.json and all recorded truth are untouched",
        "max_per_cluster": max_per_cluster,
        "positives": positives,
        "hard_negatives": negatives,
        "summary": {
            "positives_in": positives["inputs_count"],
            "positives_exported": positives["kept_count"],
            "positives_dropped": positives["dropped_count"],
            "hard_negatives_in": negatives["inputs_count"],
            "hard_negatives_exported": negatives["kept_count"],
            "hard_negatives_dropped": negatives["dropped_count"],
            "clusters": positives["cluster_count"] + negatives["cluster_count"],
        },
    }
