#!/usr/bin/env python3
"""Fit and evaluate frozen-feature ground-litter candidate filters.

The fit command cannot accept holdout labels.  The evaluate command consumes a
previously frozen artifact and the separately released holdout reviews.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np


VALID_LABELS = {"LITTER": 1, "NON_LITTER": 0}
TRAINING_SEED = 20260922
TRAINING_STEPS = 500
LEARNING_RATE = 0.015
WEIGHT_DECAY = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_named(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if name in output:
            raise ValueError(f"duplicate named path: {name}")
        output[name] = Path(path)
    return output


def load_embedding_map(path: Path) -> tuple[dict[str, int], np.ndarray]:
    cache = np.load(path, allow_pickle=False)
    ids = cache["sample_ids"].astype(str).tolist()
    matrix = cache["embeddings"].astype(np.float32)
    if len(ids) != len(matrix) or len(ids) != len(set(ids)):
        raise ValueError(f"invalid embedding identity table: {path}")
    return {sample_id: index for index, sample_id in enumerate(ids)}, matrix


def build_training_rows(
    base_manifest: dict, active_selection: dict, active_reviews: dict,
) -> tuple[list[dict], dict]:
    rows = []
    for source in base_manifest["rows"]:
        row = dict(source)
        if int(row["label"]) not in (0, 1):
            raise ValueError("base manifest contains a non-binary label")
        row["embedding_id"] = row["sample_id"]
        row["origin"] = row.get("day", "base")
        rows.append(row)

    review_map = active_reviews["reviews"]
    active_counts = {"positive": 0, "negative": 0, "excluded": 0}
    for source in active_selection["selected"]:
        if not str(source["source"]).startswith("semantic"):
            continue
        review = review_map.get(source["proposal_id"])
        if review is None:
            raise ValueError(f"missing active review: {source['proposal_id']}")
        if review["label"] not in VALID_LABELS:
            active_counts["excluded"] += 1
            continue
        label = VALID_LABELS[review["label"]]
        active_counts["positive" if label else "negative"] += 1
        rows.append({
            "sample_id": f"active_day3:{source['proposal_id']}",
            "embedding_id": source["proposal_id"],
            "day": "active_day3",
            "origin": "active_day3",
            "device_code": source["device_code"],
            "timestamp": source["timestamp"],
            "label": label,
            "score": float(source["score"]),
            "source": source["source"],
            "bbox": source["bbox"],
        })
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate training sample_id")
    summary = {
        "base_rows": len(base_manifest["rows"]),
        "active": active_counts,
        "rows": len(rows),
        "positive": sum(int(row["label"]) for row in rows),
        "negative": sum(not int(row["label"]) for row in rows),
    }
    return rows, summary


def assemble_features(
    rows: list[dict], base_path: Path, pool_path: Path,
) -> np.ndarray:
    base_index, base = load_embedding_map(base_path)
    pool_index, pool = load_embedding_map(pool_path)
    output = []
    for row in rows:
        embedding_id = row["embedding_id"]
        if row["origin"] == "active_day3":
            if embedding_id not in pool_index:
                raise ValueError(f"active embedding missing: {embedding_id}")
            output.append(pool[pool_index[embedding_id]])
        else:
            if embedding_id not in base_index:
                raise ValueError(f"base embedding missing: {embedding_id}")
            output.append(base[base_index[embedding_id]])
    matrix = np.stack(output).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("training features contain non-finite values")
    return matrix


def train_linear_head_on_device(
    features: np.ndarray, labels: np.ndarray, device: str,
) -> tuple[np.ndarray, float, dict, float, float]:
    """Train the frozen-feature linear head on the explicitly requested device."""
    import torch

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(TRAINING_SEED)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        torch.cuda.manual_seed_all(TRAINING_SEED)
    matrix = features.astype(np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-6)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    matrix = (matrix - mean) / np.maximum(std, 1e-3)
    x = torch.from_numpy(matrix).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).to(device)
    layer = torch.nn.Linear(matrix.shape[1], 1).to(device)
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    positives = max(1, int(labels.sum()))
    negatives = max(1, len(labels) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=device),
    )
    optimizer = torch.optim.AdamW(
        layer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(TRAINING_STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(layer(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = (
        layer.weight.detach().cpu().numpy().reshape(-1),
        float(layer.bias.detach().cpu().numpy()[0]),
        {"mean": mean.reshape(-1), "std": std.reshape(-1)},
        elapsed,
        float(loss.detach().cpu()),
    )
    del layer, optimizer, loss_fn, x, y
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return output


def unweighted_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float,
) -> dict:
    return weighted_metrics(scores, labels, threshold, np.ones(len(labels)))


def weighted_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float, weights: np.ndarray,
) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (len(scores) == len(labels) == len(weights)) or not len(labels):
        raise ValueError("metric arrays must have the same non-zero length")
    if not np.isfinite(scores).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("metric scores/weights must be finite and weights positive")
    passed = scores >= threshold
    positive = labels == 1
    negative = labels == 0
    tp = float(weights[passed & positive].sum())
    fp = float(weights[passed & negative].sum())
    positive_weight = float(weights[positive].sum())
    negative_weight = float(weights[negative].sum())
    return {
        "rows": int(len(labels)),
        "positive_rows": int(positive.sum()),
        "negative_rows": int(negative.sum()),
        "positive_weight": positive_weight,
        "negative_weight": negative_weight,
        "threshold": float(threshold),
        "recall": tp / positive_weight if positive_weight else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "negative_reduction": 1.0 - fp / negative_weight if negative_weight else None,
        "candidate_reduction": 1.0 - float(weights[passed].sum()) / float(weights.sum()),
    }


def ensemble_scores(features: np.ndarray, artifact: dict[str, np.ndarray]) -> np.ndarray:
    predictions = []
    for weight, bias, mean, std in zip(
        artifact["weights"], artifact["biases"], artifact["means"], artifact["stds"],
    ):
        matrix = features.astype(np.float32)
        matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-6)
        matrix = (matrix - mean) / np.maximum(std, 1e-3)
        predictions.append(matrix @ weight + float(bias))
    return np.mean(np.stack(predictions), axis=0)


def camera_excluded_scores(
    features: np.ndarray, devices: list[str], artifact: dict[str, np.ndarray],
) -> np.ndarray:
    held_out = artifact["held_out_devices"].astype(str).tolist()
    lookup = {device: index for index, device in enumerate(held_out)}
    output = np.empty(len(features), dtype=np.float32)
    for row_index, device in enumerate(devices):
        if device not in lookup:
            raise ValueError(f"artifact has no camera-excluded head for {device}")
        head = lookup[device]
        vector = features[row_index].astype(np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-6)
        vector = (vector - artifact["means"][head]) / np.maximum(artifact["stds"][head], 1e-3)
        output[row_index] = vector @ artifact["weights"][head] + artifact["biases"][head]
    return output


def device_report(device_name: str) -> dict:
    """Describe the exact torch device and fail if requested CUDA is unavailable."""
    import torch

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested, but CUDA is unavailable in this container")
    device = torch.device(device_name)
    report = {
        "requested": device_name,
        "resolved_type": device.type,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": None,
        "gpu_total_memory_bytes": None,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report["gpu_name"] = properties.name
        report["gpu_total_memory_bytes"] = int(properties.total_memory)
    return report


def probe_command(args: argparse.Namespace) -> int:
    print(json.dumps(device_report(args.device), ensure_ascii=False, indent=2))
    return 0


def fit_command(args: argparse.Namespace) -> int:
    # Importing this module loads torch; keep it out of manifest-only/test paths.
    import evaluate_ground_litter_candidate_classifier as classifier

    runtime = device_report(args.device)

    base_paths = parse_named(args.base_embedding)
    pool_paths = parse_named(args.pool_embedding)
    if set(base_paths) != set(pool_paths):
        raise ValueError("base and pool embedding model names differ")
    manifest = json.loads(args.base_manifest.read_text())
    selection = json.loads(args.active_selection.read_text())
    reviews = json.loads(args.active_reviews.read_text())
    rows, summary = build_training_rows(manifest, selection, reviews)
    if summary["positive"] != 147 or summary["negative"] != 1005:
        raise ValueError(f"unexpected frozen training counts: {summary}")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    devices = np.asarray([row["device_code"] for row in rows], dtype=np.str_)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite training artifact: {args.output}")
    model_reports = {}
    with tempfile.TemporaryDirectory(
        dir=args.output.parent, prefix=f".{args.output.name}.tmp-",
    ) as temporary:
        staging = Path(temporary)
        for model_name in sorted(base_paths):
            features = assemble_features(rows, base_paths[model_name], pool_paths[model_name])
            oof = np.full(len(rows), np.nan, dtype=np.float32)
            heads = []
            for device in sorted(set(devices.tolist())):
                fit_indices = np.flatnonzero(devices != device)
                validation_indices = np.flatnonzero(devices == device)
                weight, bias, transform, elapsed, final_loss = train_linear_head_on_device(
                    features[fit_indices], labels[fit_indices], args.device,
                )
                oof[validation_indices] = classifier.linear_scores(
                    features[validation_indices], weight, bias, transform,
                )
                heads.append((
                    device, weight, bias, transform["mean"], transform["std"], elapsed,
                    final_loss, int(len(fit_indices)), int(labels[fit_indices].sum()),
                ))
            if not np.isfinite(oof).all():
                raise RuntimeError("camera-wise OOF predictions are incomplete")
            threshold = classifier.recall_threshold(oof, labels, args.target_recall)
            artifact_path = staging / f"{model_name}.npz"
            np.savez_compressed(
                artifact_path,
                weights=np.stack([head[1] for head in heads]).astype(np.float32),
                biases=np.asarray([head[2] for head in heads], dtype=np.float32),
                means=np.stack([head[3] for head in heads]).astype(np.float32),
                stds=np.stack([head[4] for head in heads]).astype(np.float32),
                held_out_devices=np.asarray([head[0] for head in heads], dtype=np.str_),
                threshold=np.asarray([threshold], dtype=np.float32),
            )
            model_reports[model_name] = {
                "artifact": artifact_path.name,
                "artifact_sha256": sha256(artifact_path),
                "embedding_dimension": int(features.shape[1]),
                "heads": len(heads),
                "head_training": {
                    head[0][-5:]: {
                        "excluded_camera": head[0],
                        "rows": head[7],
                        "positive": head[8],
                        "negative": head[7] - head[8],
                        "positive_weight": (head[7] - head[8]) / max(1, head[8]),
                        "steps": TRAINING_STEPS,
                        "final_training_loss": head[6],
                        "seconds": round(float(head[5]), 6),
                    }
                    for head in heads
                },
                "total_head_training_seconds": round(
                    sum(float(head[5]) for head in heads), 6
                ),
                "threshold": float(threshold),
                "camera_oof_metrics": unweighted_metrics(oof, labels, threshold),
            }
        report = {
            "schema": "ground_litter_candidate_filter_fit_v1",
            "compute": runtime,
            "training_config": {
                "seed": TRAINING_SEED,
                "steps_per_head": TRAINING_STEPS,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "loss": "BCEWithLogitsLoss",
                "class_weighting": "negative_count / positive_count per camera fold",
                "deterministic_algorithms": True,
            },
            "target_recall": args.target_recall,
            "training": summary,
            "input_sha256": {
                "base_manifest": sha256(args.base_manifest),
                "active_selection": sha256(args.active_selection),
                "active_reviews": sha256(args.active_reviews),
                "base_embeddings": {name: sha256(path) for name, path in base_paths.items()},
                "pool_embeddings": {name: sha256(path) for name, path in pool_paths.items()},
            },
            "models": model_reports,
            "holdout_labels_read": False,
            "notes": [
                "Threshold is calibrated on camera-wise out-of-fold training predictions.",
                "The separately frozen holdout labels were not accepted by or read by fit.",
            ],
        }
        (staging / "FIT_REPORT.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        staging.rename(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_holdout_rows(selection: dict, reviews: dict) -> tuple[list[dict], dict]:
    expected_identity = hashlib.sha256("\n".join(sorted(
        row["proposal_id"] for row in selection["selected"]
    )).encode()).hexdigest()
    if expected_identity != selection["identity_sha256"]:
        raise ValueError("holdout selection identity mismatch")
    review_map = reviews["reviews"]
    rows, excluded = [], {"UNCERTAIN": 0, "BOX_WRONG": 0}
    for source in selection["selected"]:
        review = review_map.get(source["proposal_id"])
        if review is None:
            raise ValueError(f"missing holdout review: {source['proposal_id']}")
        label_name = review["label"]
        if label_name not in VALID_LABELS:
            excluded[label_name] = excluded.get(label_name, 0) + 1
            continue
        rows.append({
            **source,
            "embedding_id": source["proposal_id"],
            "label": VALID_LABELS[label_name],
        })
    return rows, excluded


def evaluate_command(args: argparse.Namespace) -> int:
    pool_paths = parse_named(args.pool_embedding)
    fit_report = json.loads((args.artifact_dir / "FIT_REPORT.json").read_text())
    selection = json.loads(args.holdout_selection.read_text())
    reviews = json.loads(args.holdout_reviews.read_text())
    rows, excluded = build_holdout_rows(selection, reviews)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    weights = np.asarray([row["holdout_weight"] for row in rows], dtype=np.float64)
    devices = [row["device_code"] for row in rows]
    reports = {}
    for model_name, model_report in fit_report["models"].items():
        if model_name not in pool_paths:
            raise ValueError(f"missing holdout embedding path for {model_name}")
        index, matrix = load_embedding_map(pool_paths[model_name])
        features = np.stack([matrix[index[row["embedding_id"]]] for row in rows])
        artifact_path = args.artifact_dir / model_report["artifact"]
        if sha256(artifact_path) != model_report["artifact_sha256"]:
            raise ValueError(f"artifact hash mismatch for {model_name}")
        loaded = np.load(artifact_path, allow_pickle=False)
        artifact = {key: loaded[key] for key in loaded.files}
        threshold = float(artifact["threshold"][0])
        cross_camera = camera_excluded_scores(features, devices, artifact)
        ensemble = ensemble_scores(features, artifact)
        reports[model_name] = {
            "primary_camera_excluded_weighted": weighted_metrics(
                cross_camera, labels, threshold, weights,
            ),
            "primary_camera_excluded_unweighted": unweighted_metrics(
                cross_camera, labels, threshold,
            ),
            "exploratory_all_head_ensemble_weighted": weighted_metrics(
                ensemble, labels, threshold, weights,
            ),
            "by_camera": {
                device[-5:]: unweighted_metrics(
                    cross_camera[np.asarray(devices) == device],
                    labels[np.asarray(devices) == device], threshold,
                )
                for device in sorted(set(devices))
            },
        }
    output = {
        "schema": "ground_litter_candidate_filter_holdout_eval_v1",
        "fit_report_sha256": sha256(args.artifact_dir / "FIT_REPORT.json"),
        "holdout_selection_sha256": sha256(args.holdout_selection),
        "holdout_reviews_sha256": sha256(args.holdout_reviews),
        "holdout_identity_sha256": selection["identity_sha256"],
        "reviewed": len(reviews["reviews"]),
        "scored": len(rows),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "excluded": excluded,
        "models": reports,
        "limitations": [
            "This is a score-blind stratified holdout from the same date/pool as active-day3.",
            "Weighted metrics estimate the remaining candidate pool after active selection.",
            "A new-date shadow run remains required before production use.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--device", default="cuda:0")
    probe.set_defaults(handler=probe_command)
    fit = sub.add_parser("fit")
    fit.add_argument("--base-manifest", type=Path, required=True)
    fit.add_argument("--active-selection", type=Path, required=True)
    fit.add_argument("--active-reviews", type=Path, required=True)
    fit.add_argument("--base-embedding", action="append", required=True)
    fit.add_argument("--pool-embedding", action="append", required=True)
    fit.add_argument("--target-recall", type=float, default=0.90)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--output", type=Path, required=True)
    fit.set_defaults(handler=fit_command)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--artifact-dir", type=Path, required=True)
    evaluate.add_argument("--holdout-selection", type=Path, required=True)
    evaluate.add_argument("--holdout-reviews", type=Path, required=True)
    evaluate.add_argument("--pool-embedding", action="append", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(handler=evaluate_command)
    return root


def main() -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = parser().parse_args()
    if getattr(args, "target_recall", 0.90) <= 0 or getattr(args, "target_recall", 0.90) > 1:
        raise SystemExit("--target-recall must be in (0, 1]")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
