#!/usr/bin/env python3
"""Score a third-day candidate pool and select the frozen active-learning budget."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import evaluate_ground_litter_candidate_classifier as classifier


SEMANTIC_QUOTA = {"01022": 80, "01021": 40, "01027": 40, "01028": 40, "01030": 40}
RANDOM_QUOTA = {"01022": 20, "01021": 10, "01027": 10, "01028": 10, "01030": 10}


def parse_named(values: list[str]) -> dict[str, str]:
    output = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("named arguments require NAME=PATH")
        name, path = value.split("=", 1)
        output[name] = path
    return output


def hour_bin(row: dict) -> str:
    hour = int(row["timestamp"][11:13])
    start = hour // 6 * 6
    return f"{start:02d}-{start + 5:02d}"


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-6)


def oof_and_pool_scores(
    labeled: list[dict], labeled_features: np.ndarray,
    pool_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, list[dict]]:
    labels = np.asarray([int(row["label"]) for row in labeled], dtype=np.int64)
    devices = sorted({row["device_code"] for row in labeled})
    oof = np.full(len(labeled), np.nan, dtype=np.float32)
    pool_predictions = []
    folds = []
    for device in devices:
        fit = np.asarray([i for i, row in enumerate(labeled) if row["device_code"] != device])
        validation = np.asarray([i for i, row in enumerate(labeled) if row["device_code"] == device])
        weight, bias, transform = classifier.train_linear_head(
            labeled_features[fit], labels[fit],
        )
        oof[validation] = classifier.linear_scores(
            labeled_features[validation], weight, bias, transform,
        )
        pool_predictions.append(classifier.linear_scores(
            pool_features, weight, bias, transform,
        ))
        folds.append({
            "held_out_camera": device[-5:], "fit": int(len(fit)),
            "validation": int(len(validation)),
            "validation_positive": int(labels[validation].sum()),
        })
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF predictions are incomplete")
    threshold = classifier.recall_threshold(oof, labels, 0.90)
    return oof, np.mean(np.stack(pool_predictions), axis=0), threshold, folds


def max_similarity_by_camera(
    labeled: list[dict], labeled_features: np.ndarray,
    pool: list[dict], pool_features: np.ndarray,
) -> np.ndarray:
    old = normalize(labeled_features)
    new = normalize(pool_features)
    output = np.zeros(len(pool), dtype=np.float32)
    for device in sorted({row["device_code"] for row in pool}):
        old_indices = [i for i, row in enumerate(labeled) if row["device_code"] == device]
        new_indices = [i for i, row in enumerate(pool) if row["device_code"] == device]
        if not old_indices or not new_indices:
            continue
        similarities = new[new_indices] @ old[old_indices].T
        output[new_indices] = similarities.max(axis=1)
    return output


def overlaps_reviewed_position(row: dict, labeled: list[dict]) -> bool:
    return any(
        old["device_code"] == row["device_code"]
        and classifier.spatially_related(row, old)
        for old in labeled
    )


def stable_tie(row: dict) -> str:
    return hashlib.sha256(("active-v1:" + row["proposal_id"]).encode()).hexdigest()


def take_rows(
    candidates: list[dict], selected: list[dict], count: int,
    reason: str, key,
) -> None:
    selected_ids = {row["proposal_id"] for row in selected}
    ordered = sorted(
        (row for row in candidates if row["proposal_id"] not in selected_ids),
        key=lambda row: (row["reviewed_position_overlap"], key(row), stable_tie(row)),
    )
    for row in ordered[:count]:
        copied = dict(row)
        copied["selection_reason"] = reason
        selected.append(copied)


def select_semantic(candidates: list[dict], quota: int) -> list[dict]:
    selected: list[dict] = []
    budgets = {
        "disagreement": round(quota * 0.35),
        "boundary": round(quota * 0.35),
        "likely_positive": round(quota * 0.15),
    }
    budgets["novel"] = quota - sum(budgets.values())
    disagreement = [row for row in candidates if row["turhancan_pass"] != row["yolo26s_pass"]]
    take_rows(
        disagreement, selected, budgets["disagreement"], "classifier_disagreement",
        key=lambda row: -abs(row["turhancan_margin"] - row["yolo26s_margin"]),
    )
    take_rows(
        candidates, selected, budgets["boundary"], "decision_boundary",
        key=lambda row: min(abs(row["turhancan_margin"]), abs(row["yolo26s_margin"])),
    )
    take_rows(
        candidates, selected, budgets["likely_positive"], "both_likely_positive",
        key=lambda row: -min(row["turhancan_margin"], row["yolo26s_margin"]),
    )
    take_rows(
        candidates, selected, budgets["novel"], "novel_appearance",
        key=lambda row: -row["novelty"],
    )
    if len(selected) < quota:
        take_rows(
            candidates, selected, quota - len(selected), "budget_fill",
            key=lambda row: min(abs(row["turhancan_margin"]), abs(row["yolo26s_margin"])),
        )
    return selected[:quota]


def select_random(candidates: list[dict], semantic: list[dict], quota: int) -> list[dict]:
    eligible = []
    for row in candidates:
        if any(
            old["frame_id"] == row["frame_id"]
            and classifier.box_iou(old["bbox"], row["bbox"]) >= 0.20
            for old in semantic
        ):
            continue
        eligible.append(row)
    selected = []
    bins = ("00-05", "06-11", "12-17", "18-23")
    base, extra = divmod(quota, len(bins))
    for index, name in enumerate(bins):
        count = base + (1 if index < extra else 0)
        group = sorted(
            (row for row in eligible if hour_bin(row) == name), key=stable_tie,
        )
        for row in group[:count]:
            copied = dict(row)
            copied["selection_reason"] = "random_audit"
            selected.append(copied)
    if len(selected) < quota:
        take_rows(
            eligible, selected, quota - len(selected), "random_audit_fill",
            key=lambda row: stable_tie(row),
        )
    return selected[:quota]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-manifest", type=Path, required=True)
    parser.add_argument("--labeled-embedding", action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    labeled = json.loads(args.labeled_manifest.read_text())["rows"]
    labeled_embedding_paths = parse_named(args.labeled_embedding)
    model_paths = parse_named(args.model)
    if set(labeled_embedding_paths) != {"turhancan", "yolo26s"} or set(model_paths) != set(labeled_embedding_paths):
        raise SystemExit("turhancan and yolo26s embeddings/models are both required")
    semantic, random_rows = [], []
    for path in args.pool:
        payload = json.loads(path.read_text())
        for row in payload["semantic"]:
            copied = dict(row)
            image_path = Path(copied["image"])
            copied["image"] = str(image_path if image_path.is_absolute() else path.parent / image_path)
            semantic.append(copied)
        random_rows.extend(payload["random"])
    pool_for_embedding = [{**row, "sample_id": row["proposal_id"]} for row in semantic]
    args.output.mkdir(parents=True, exist_ok=True)

    diagnostics = {}
    pool_features = {}
    old_features = {}
    for name in ("turhancan", "yolo26s"):
        old_cache = np.load(labeled_embedding_paths[name], allow_pickle=False)
        expected = [row["sample_id"] for row in labeled]
        if old_cache["sample_ids"].astype(str).tolist() != expected:
            raise RuntimeError(f"labeled embedding identity mismatch for {name}")
        old_features[name] = old_cache["embeddings"].astype(np.float32)
        pool_features[name] = classifier.extract_embeddings(
            pool_for_embedding, model_paths[name], args.output / f"pool_{name}.npz",
            args.imgsz, args.batch,
        )
        oof, scored, threshold, folds = oof_and_pool_scores(
            labeled, old_features[name], pool_features[name],
        )
        scale = max(1e-3, float(np.quantile(oof, 0.75) - np.quantile(oof, 0.25)))
        diagnostics[name] = {
            "threshold": float(threshold), "scale": scale, "folds": folds,
            "pool_scores": scored,
            "max_similarity": max_similarity_by_camera(
                labeled, old_features[name], semantic, pool_features[name],
            ),
        }

    scored_rows = []
    for index, row in enumerate(semantic):
        copied = dict(row)
        similarities = []
        for name in ("turhancan", "yolo26s"):
            details = diagnostics[name]
            score = float(details["pool_scores"][index])
            margin = (score - details["threshold"]) / details["scale"]
            copied[f"{name}_classifier_score"] = round(score, 8)
            copied[f"{name}_margin"] = round(margin, 8)
            copied[f"{name}_pass"] = bool(score >= details["threshold"])
            similarities.append(float(details["max_similarity"][index]))
        copied["novelty"] = round(1.0 - sum(similarities) / len(similarities), 8)
        copied["reviewed_position_overlap"] = overlaps_reviewed_position(copied, labeled)
        scored_rows.append(copied)

    selected = []
    per_camera = {}
    for suffix in sorted(SEMANTIC_QUOTA):
        camera_semantic = [row for row in scored_rows if row["device_code"].endswith(suffix)]
        camera_random = [row for row in random_rows if row["device_code"].endswith(suffix)]
        chosen_semantic = select_semantic(camera_semantic, SEMANTIC_QUOTA[suffix])
        chosen_random = select_random(camera_random, chosen_semantic, RANDOM_QUOTA[suffix])
        selected.extend(chosen_semantic + chosen_random)
        per_camera[suffix] = {
            "semantic_pool": len(camera_semantic), "random_pool": len(camera_random),
            "semantic_selected": len(chosen_semantic), "random_selected": len(chosen_random),
            "reasons": {reason: sum(row["selection_reason"] == reason for row in chosen_semantic + chosen_random)
                        for reason in sorted({row["selection_reason"] for row in chosen_semantic + chosen_random})},
            "hour_bins": {name: sum(hour_bin(row) == name for row in chosen_semantic + chosen_random)
                          for name in ("00-05", "06-11", "12-17", "18-23")},
        }
    payload = {
        "schema": "ground_litter_active_selection_v1", "count": len(selected),
        "semantic_count": sum(row["source"].startswith("semantic") for row in selected),
        "random_count": sum(row["source"] == "random_grid" for row in selected),
        "thresholds_frozen_from_days_1_2": {
            name: {key: value for key, value in details.items() if key not in ("pool_scores", "max_similarity")}
            for name, details in diagnostics.items()
        },
        "per_camera": per_camera, "selected": selected,
    }
    (args.output / "selection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({key: payload[key] for key in ("count", "semantic_count", "random_count", "per_camera")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
