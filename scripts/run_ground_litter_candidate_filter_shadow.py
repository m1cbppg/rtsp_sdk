#!/usr/bin/env python3
"""Run a frozen candidate filter over new-date semantic pools.

This is an offline/read-only shadow tool.  It never creates a stream, displays a
box, sends an alert, trains a head, or changes the frozen threshold.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import train_ground_litter_candidate_filter as candidate_filter


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def simulate_latest_wins(service_seconds: list[float], interval_seconds: float) -> dict:
    """Simulate one worker plus a one-slot queue which replaces stale work."""
    if interval_seconds <= 0:
        raise ValueError("arrival interval must be positive")
    if any(value < 0 or not math.isfinite(value) for value in service_seconds):
        raise ValueError("service times must be finite and non-negative")
    if not service_seconds:
        return {"submitted": 0, "completed": 0, "dropped": 0,
                "maximum_queue_depth": 0, "age_p95_seconds": None,
                "age_max_seconds": None}
    arrivals = [index * interval_seconds for index in range(len(service_seconds))]
    next_arrival = 0
    current: tuple[int, float, float] | None = None
    queued: tuple[int, float] | None = None
    now = 0.0
    dropped = 0
    ages: list[float] = []
    maximum_depth = 0
    while next_arrival < len(arrivals) or current is not None or queued is not None:
        arrival_time = arrivals[next_arrival] if next_arrival < len(arrivals) else math.inf
        finish_time = current[2] if current is not None else math.inf
        if arrival_time <= finish_time:
            now = arrival_time
            work = (next_arrival, arrival_time)
            next_arrival += 1
            if current is None:
                index, captured = work
                current = (index, captured, now + service_seconds[index])
            else:
                if queued is not None:
                    dropped += 1
                queued = work
                maximum_depth = 1
        else:
            now = finish_time
            assert current is not None
            _, captured, _ = current
            ages.append(now - captured)
            current = None
            if queued is not None:
                index, captured = queued
                queued = None
                current = (index, captured, now + service_seconds[index])
    return {
        "submitted": len(service_seconds), "completed": len(ages),
        "dropped": dropped, "maximum_queue_depth": maximum_depth,
        "age_p95_seconds": percentile(ages, 95),
        "age_max_seconds": max(ages) if ages else None,
    }


def load_rows(pool_paths: list[Path]) -> tuple[list[dict], list[dict], dict]:
    semantic: list[dict] = []
    random_rows: list[dict] = []
    identities = {}
    for path in pool_paths:
        payload = json.loads(path.read_text())
        identities[str(path)] = sha256(path)
        for source in payload["semantic"]:
            row = dict(source)
            if "image" in row:
                image = Path(row["image"])
                row["image"] = str(image if image.is_absolute() else path.parent / image)
            elif "frame_image" not in row:
                raise ValueError("semantic row has neither crop image nor frame image")
            row["sample_id"] = row["proposal_id"]
            semantic.append(row)
        random_rows.extend(dict(row) for row in payload["random"])
    ids = [row["proposal_id"] for row in semantic]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate semantic proposal_id across pools")
    return semantic, random_rows, identities


def extract_by_tick(
    rows: list[dict], model_path: str, output: Path, imgsz: int, batch: int,
) -> tuple[np.ndarray, list[dict]]:
    import torch
    from ultralytics import YOLO

    import cv2

    model = YOLO(model_path)
    by_tick: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_tick[(row["device_code"], row["frame_id"])].append(index)
    matrix: list[np.ndarray | None] = [None] * len(rows)
    timings = []
    for tick_index, ((device, frame_id), indices) in enumerate(sorted(by_tick.items()), 1):
        sources = []
        if "frame_image" in rows[indices[0]]:
            frame = cv2.imread(rows[indices[0]]["frame_image"])
            if frame is None:
                raise RuntimeError(f"cannot read source frame for {device}:{frame_id}")
            h, w = frame.shape[:2]
            for index in indices:
                x1, y1, x2, y2 = rows[index]["bbox"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                side = max(32.0, (x2 - x1) * 1.8, (y2 - y1) * 1.8)
                patch = frame[
                    max(0, round(cy - side / 2)):min(h, round(cy + side / 2)),
                    max(0, round(cx - side / 2)):min(w, round(cx + side / 2)),
                ]
                if patch.size == 0:
                    raise RuntimeError(f"empty crop for {rows[index]['proposal_id']}")
                sources.append(patch)
        else:
            sources = [rows[index]["image"] for index in indices]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        for offset in range(0, len(indices), batch):
            selected = indices[offset:offset + batch]
            embedded = model.embed(
                source=sources[offset:offset + len(selected)],
                imgsz=imgsz, batch=batch, device=0, verbose=False,
            )
            if len(embedded) != len(selected):
                raise RuntimeError("embedding count mismatch")
            for index, value in zip(selected, embedded):
                matrix[index] = value.detach().float().cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        timings.append({
            "device_code": device, "frame_id": frame_id,
            "candidate_count": len(indices), "seconds": elapsed,
        })
        print(json.dumps({
            "tick": tick_index, "ticks": len(by_tick),
            "candidates": len(indices), "seconds": round(elapsed, 4),
        }), flush=True)
    if any(value is None for value in matrix):
        raise RuntimeError("embedding extraction left empty rows")
    features = np.stack(matrix).astype(np.float32)  # type: ignore[arg-type]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_ids=np.asarray([row["proposal_id"] for row in rows], dtype=np.str_),
        embeddings=features,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features, timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--analysis-fps", type=float, default=2.0)
    args = parser.parse_args()
    if args.analysis_fps <= 0:
        raise SystemExit("--analysis-fps must be positive")

    rows, random_rows, pool_hashes = load_rows(args.pool)
    if not rows:
        raise RuntimeError("shadow input has no semantic candidates")
    fit_path = args.artifact_dir / "FIT_REPORT.json"
    fit = json.loads(fit_path.read_text())
    model_report = fit["models"]["yolo26s"]
    artifact_path = args.artifact_dir / model_report["artifact"]
    if sha256(artifact_path) != model_report["artifact_sha256"]:
        raise RuntimeError("frozen yolo26s artifact SHA-256 mismatch")
    loaded = np.load(artifact_path, allow_pickle=False)
    artifact = {name: loaded[name] for name in loaded.files}
    threshold = float(artifact["threshold"][0])
    expected_devices = set(artifact["held_out_devices"].astype(str).tolist())
    actual_devices = {row["device_code"] for row in rows}
    if not actual_devices <= expected_devices:
        raise RuntimeError(f"artifact lacks camera heads: {sorted(actual_devices - expected_devices)}")

    args.output.mkdir(parents=True, exist_ok=False)
    features, tick_timings = extract_by_tick(
        rows, args.model, args.output / "yolo26s_embeddings.npz",
        args.imgsz, args.batch,
    )
    scores = candidate_filter.camera_excluded_scores(
        features, [row["device_code"] for row in rows], artifact,
    )
    if not np.isfinite(scores).all():
        raise RuntimeError("shadow scores contain non-finite values")
    decisions = []
    for row, score in zip(rows, scores):
        decisions.append({
            **row, "classifier": "yolo26s_camera_excluded_head_v1",
            "classifier_score": float(score), "threshold": threshold,
            "margin": float(score - threshold), "passed": bool(score >= threshold),
        })
    seconds = [float(row["seconds"]) for row in tick_timings]
    queue = simulate_latest_wins(seconds, 1.0 / args.analysis_fps)
    devices = sorted({row["device_code"] for row in tick_timings})
    by_camera_queue = {
        device[-5:]: simulate_latest_wins(
            [float(row["seconds"]) for row in tick_timings
             if row["device_code"] == device],
            1.0 / args.analysis_fps,
        )
        for device in devices
    }
    aggregate_queue = simulate_latest_wins(
        seconds, 1.0 / (args.analysis_fps * len(devices)),
    )
    performance = {
        "schema": "ground_litter_candidate_filter_shadow_performance_v1",
        "ticks": len(seconds), "candidates": len(rows),
        "candidate_count_p95": percentile(
            [float(row["candidate_count"]) for row in tick_timings], 95,
        ),
        "tick_seconds_p50": percentile(seconds, 50),
        "tick_seconds_p95": percentile(seconds, 95),
        "tick_seconds_p99": percentile(seconds, 99),
        "tick_seconds_max": max(seconds),
        "simulated_latest_wins_at_fps": args.analysis_fps,
        "latest_wins": queue,
        "per_camera_latest_wins": by_camera_queue,
        "aggregate_shared_worker_latest_wins": aggregate_queue,
        "aggregate_arrival_fps": args.analysis_fps * len(devices),
        "per_tick": tick_timings,
        "note": "Measures only YOLO26s crop embedding/filter stage on extracted frames.",
    }
    payload = {
        "schema": "ground_litter_candidate_filter_shadow_v1",
        "fit_report_sha256": sha256(fit_path),
        "artifact_sha256": sha256(artifact_path),
        "model_path": args.model, "threshold": threshold,
        "input_pool_sha256": pool_hashes,
        "semantic_count": len(decisions), "random_count": len(random_rows),
        "passed": sum(row["passed"] for row in decisions),
        "filtered": sum(not row["passed"] for row in decisions),
        "decisions": decisions,
    }
    (args.output / "SHADOW_DECISIONS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )
    (args.output / "PERFORMANCE.json").write_text(
        json.dumps(performance, ensure_ascii=False, indent=2)
    )
    print(json.dumps({
        "semantic": len(decisions), "passed": payload["passed"],
        "filtered": payload["filtered"],
        "tick_p95_seconds": performance["tick_seconds_p95"],
        "latest_wins": queue,
        "aggregate_shared_worker_latest_wins": aggregate_queue,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
