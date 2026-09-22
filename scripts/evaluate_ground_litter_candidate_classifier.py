#!/usr/bin/env python3
"""Frozen-feature feasibility test for filtering ground-litter candidates.

The script deliberately avoids random train/test splits.  It evaluates both
day-held-out directions and leave-one-camera-out splits.  Test rows that share
the same fixed-camera spatial location with the opposite day are removed from
day-held-out scoring to reduce persistent-object leakage.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", action="append", required=True,
        help="NAME=PATH; may be supplied more than once",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--target-recall", type=float, default=0.90)
    return parser.parse_args()


def chunks(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for offset in range(0, len(rows), size):
        yield rows[offset:offset + size]


def extract_embeddings(
    rows: list[dict], model_path: str, cache: Path, imgsz: int, batch: int,
) -> np.ndarray:
    sample_ids = [row["sample_id"] for row in rows]
    if cache.is_file():
        loaded = np.load(cache, allow_pickle=False)
        cached_ids = loaded["sample_ids"].astype(str).tolist()
        if cached_ids == sample_ids:
            return loaded["embeddings"].astype(np.float32)
    model = YOLO(model_path)
    output: list[np.ndarray] = []
    done = 0
    for group in chunks(rows, batch):
        embedded = model.embed(
            source=[row["image"] for row in group], imgsz=imgsz,
            batch=batch, device=0, verbose=False,
        )
        if len(embedded) != len(group):
            raise RuntimeError(f"embedding count mismatch: {len(embedded)} != {len(group)}")
        output.extend(item.detach().float().cpu().numpy() for item in embedded)
        done += len(group)
        print(json.dumps({"embedded": done, "total": len(rows)}), flush=True)
    matrix = np.stack(output).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, sample_ids=np.asarray(sample_ids, dtype=np.str_), embeddings=matrix,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return matrix


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = a
    X1, Y1, X2, Y2 = b
    intersection = max(0.0, min(x2, X2) - max(x1, X1)) * max(
        0.0, min(y2, Y2) - max(y1, Y1)
    )
    union = (x2 - x1) * (y2 - y1) + (X2 - X1) * (Y2 - Y1) - intersection
    return intersection / max(1.0, union)


def spatially_related(a: dict, b: dict) -> bool:
    if a["device_code"] != b["device_code"]:
        return False
    if box_iou(a["bbox"], b["bbox"]) >= 0.20:
        return True
    x1, y1, x2, y2 = a["bbox"]
    X1, Y1, X2, Y2 = b["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    Cx, Cy = (X1 + X2) / 2, (Y1 + Y2) / 2
    diagonal = max(math.hypot(x2 - x1, y2 - y1), math.hypot(X2 - X1, Y2 - Y1))
    return math.hypot(cx - Cx, cy - Cy) <= max(24.0, 0.42 * diagonal)


def remove_cross_day_spatial_overlap(
    rows: list[dict], train_indices: np.ndarray, test_indices: np.ndarray,
) -> tuple[np.ndarray, dict]:
    train_by_device: dict[str, list[dict]] = {}
    for index in train_indices:
        row = rows[int(index)]
        train_by_device.setdefault(row["device_code"], []).append(row)
    kept, removed = [], []
    for index in test_indices:
        row = rows[int(index)]
        if any(spatially_related(row, old) for old in train_by_device.get(row["device_code"], [])):
            removed.append(int(index))
        else:
            kept.append(int(index))
    labels = [int(rows[index]["label"]) for index in removed]
    return np.asarray(kept, dtype=np.int64), {
        "removed": len(removed),
        "removed_positive": sum(labels),
        "removed_negative": len(labels) - sum(labels),
    }


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    truth = labels[order]
    positives = int(truth.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(truth) / np.arange(1, len(truth) + 1)
    return float((precision * truth).sum() / positives)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size)


def recall_threshold(scores: np.ndarray, labels: np.ndarray, target: float) -> float:
    positive = np.sort(scores[labels == 1])[::-1]
    if not len(positive):
        raise ValueError("threshold calibration has no positive rows")
    return float(positive[min(len(positive) - 1, math.ceil(target * len(positive)) - 1)])


def score_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    passed = scores >= threshold
    positives = labels == 1
    negatives = ~positives
    tp = int((passed & positives).sum())
    fp = int((passed & negatives).sum())
    return {
        "rows": int(len(labels)),
        "positive": int(positives.sum()),
        "negative": int(negatives.sum()),
        "average_precision": round(average_precision(scores, labels), 6),
        "roc_auc": round(roc_auc(scores, labels), 6),
        "threshold": round(float(threshold), 8),
        "recall": round(tp / max(1, int(positives.sum())), 6),
        "precision": round(tp / max(1, tp + fp), 6),
        "negative_reduction": round(1.0 - fp / max(1, int(negatives.sum())), 6),
        "candidate_reduction": round(1.0 - int(passed.sum()) / max(1, len(labels)), 6),
    }


def train_linear_head(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float, dict]:
    torch.manual_seed(20260922)
    matrix = features.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-6)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    matrix = (matrix - mean) / np.maximum(std, 1e-3)
    x = torch.from_numpy(matrix)
    y = torch.from_numpy(labels.astype(np.float32))
    layer = torch.nn.Linear(matrix.shape[1], 1)
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    positives = max(1, int(labels.sum()))
    negatives = max(1, len(labels) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives))
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.015, weight_decay=0.05)
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(layer(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
    return (
        layer.weight.detach().numpy().reshape(-1),
        float(layer.bias.detach().numpy()[0]),
        {"mean": mean.reshape(-1), "std": std.reshape(-1)},
    )


def linear_scores(features: np.ndarray, weight: np.ndarray, bias: float, transform: dict) -> np.ndarray:
    matrix = features.astype(np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-6)
    matrix = (matrix - transform["mean"]) / np.maximum(transform["std"], 1e-3)
    return matrix @ weight + bias


def camera_oof_ensemble_scores(
    rows: list[dict], features: np.ndarray, labels: np.ndarray,
    train: np.ndarray, test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Calibrate on predictions made by models that never saw that camera.

    A threshold fitted to in-sample logits is invalid because the small linear
    head can give much larger margins to its fitting rows.  Camera-wise OOF
    predictions have the same unseen-camera semantics as the outer test, and
    averaging the fold models gives a stable score at inference time.
    """
    devices = sorted({rows[int(index)]["device_code"] for index in train})
    oof = np.full(len(train), np.nan, dtype=np.float32)
    test_predictions: list[np.ndarray] = []
    folds = []
    for device in devices:
        fit_positions = np.asarray([
            position for position, index in enumerate(train)
            if rows[int(index)]["device_code"] != device
        ])
        validation_positions = np.asarray([
            position for position, index in enumerate(train)
            if rows[int(index)]["device_code"] == device
        ])
        if not len(fit_positions) or not len(validation_positions):
            continue
        fit_indices = train[fit_positions]
        validation_indices = train[validation_positions]
        if labels[fit_indices].sum() == 0 or labels[validation_indices].sum() == 0:
            continue
        weight, bias, transform = train_linear_head(features[fit_indices], labels[fit_indices])
        oof[validation_positions] = linear_scores(
            features[validation_indices], weight, bias, transform,
        )
        test_predictions.append(linear_scores(features[test], weight, bias, transform))
        folds.append({
            "held_out_camera": device[-5:], "fit_rows": int(len(fit_indices)),
            "validation_rows": int(len(validation_indices)),
            "validation_positive": int(labels[validation_indices].sum()),
        })
    valid = np.isfinite(oof)
    if not valid.all() or not test_predictions:
        raise RuntimeError("camera OOF calibration did not cover every training row")
    return oof, np.mean(np.stack(test_predictions), axis=0), folds


def evaluate_split(
    rows: list[dict], features: np.ndarray, train: np.ndarray, test: np.ndarray,
    name: str, target_recall: float, overlap: dict | None = None,
) -> dict:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    base = np.asarray([float(row["score"]) for row in rows], dtype=np.float32)
    oof_visual, test_visual, folds = camera_oof_ensemble_scores(
        rows, features, labels, train, test,
    )
    visual_threshold = recall_threshold(oof_visual, labels[train], target_recall)
    base_threshold = recall_threshold(base[train], labels[train], target_recall)
    oracle_visual_threshold = recall_threshold(test_visual, labels[test], target_recall)
    return {
        "name": name,
        "train": {"rows": int(len(train)), "positive": int(labels[train].sum())},
        "test": {"rows": int(len(test)), "positive": int(labels[test].sum())},
        "cross_day_overlap_filter": overlap,
        "calibration_folds": folds,
        "visual_oof_calibrated": score_metrics(test_visual, labels[test], visual_threshold),
        "visual_oof_calibration_metrics": score_metrics(
            oof_visual, labels[train], visual_threshold,
        ),
        "visual_test_oracle_90_recall": score_metrics(test_visual, labels[test], oracle_visual_threshold),
        "model_score_train_calibrated": score_metrics(base[test], labels[test], base_threshold),
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["rows"]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    args.output.mkdir(parents=True, exist_ok=True)
    model_specs = []
    for text in args.model:
        if "=" not in text:
            raise SystemExit("--model must be NAME=PATH")
        model_specs.append(text.split("=", 1))
    all_results = []
    for model_name, model_path in model_specs:
        cache = args.output / f"embeddings_{model_name}.npz"
        features = extract_embeddings(rows, model_path, cache, args.imgsz, args.batch)
        splits = []
        for train_day, test_day in (("day1", "day2"), ("day2", "day1")):
            train = np.asarray([i for i, row in enumerate(rows) if row["day"] == train_day])
            original_test = np.asarray([i for i, row in enumerate(rows) if row["day"] == test_day])
            test, overlap = remove_cross_day_spatial_overlap(rows, train, original_test)
            splits.append(evaluate_split(
                rows, features, train, test, f"{train_day}_to_{test_day}",
                args.target_recall, overlap,
            ))
        for device in sorted({row["device_code"] for row in rows}):
            train = np.asarray([i for i, row in enumerate(rows) if row["device_code"] != device])
            test = np.asarray([i for i, row in enumerate(rows) if row["device_code"] == device])
            splits.append(evaluate_split(
                rows, features, train, test, f"leave_camera_{device[-5:]}",
                args.target_recall,
            ))
        all_results.append({
            "model": model_name, "model_path": model_path,
            "embedding_shape": list(features.shape), "splits": splits,
        })
    payload = {
        "schema": "ground_litter_candidate_classifier_eval_v1",
        "manifest": str(args.manifest), "samples": len(rows),
        "positive": int(labels.sum()), "negative": int(len(labels) - labels.sum()),
        "target_recall": args.target_recall,
        "results": all_results,
        "limitations": [
            "Metrics apply to human-reviewed candidate cards, not all objects in video.",
            "Card labels are not corrected detection boxes; uncertain and box-wrong rows are excluded upstream.",
            "The test-oracle threshold measures separability only and is not a deployable threshold.",
        ],
    }
    (args.output / "RESULTS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
