#!/usr/bin/env python3
"""Evaluate human-reviewed, new-date candidate-filter shadow decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


VALID = {"LITTER": 1, "NON_LITTER": 0}


def metrics(rows: list[dict]) -> dict:
    positive = sum(row["weight"] for row in rows if row["label"] == 1)
    negative = sum(row["weight"] for row in rows if row["label"] == 0)
    true_positive = sum(
        row["weight"] for row in rows if row["label"] == 1 and row["passed"]
    )
    false_positive = sum(
        row["weight"] for row in rows if row["label"] == 0 and row["passed"]
    )
    total = positive + negative
    passed = true_positive + false_positive
    return {
        "rows": len(rows),
        "positive_rows": sum(row["label"] == 1 for row in rows),
        "negative_rows": sum(row["label"] == 0 for row in rows),
        "positive_weight": positive, "negative_weight": negative,
        "recall": true_positive / positive if positive else None,
        "precision": true_positive / passed if passed else None,
        "negative_reduction": 1.0 - false_positive / negative if negative else None,
        "candidate_reduction": 1.0 - passed / total if total else None,
    }


def decide(overall: dict, by_camera: dict, positive_rows: int) -> tuple[str, list[str]]:
    reasons = []
    if positive_rows < 20:
        reasons.append("FEWER_THAN_20_POSITIVES")
    recall = overall["recall"]
    reduction = overall["negative_reduction"]
    camera_recalls = [
        value["recall"] for value in by_camera.values()
        if value["positive_rows"] >= 5 and value["recall"] is not None
    ]
    if recall is not None and recall < 0.90:
        reasons.append("OVERALL_RECALL_BELOW_0_90")
    if any(value < 0.80 for value in camera_recalls):
        reasons.append("CAMERA_RECALL_BELOW_0_80")
    if any(reason.endswith(("0_90", "0_80")) for reason in reasons):
        return "NO_GO", reasons
    if positive_rows < 20:
        return "INSUFFICIENT_POSITIVES", reasons
    if recall is not None and recall >= 0.95 and all(
        value >= 0.90 for value in camera_recalls
    ) and reduction is not None and reduction >= 0.25:
        return "GO_TO_TEMPORAL_SHADOW", reasons
    if reduction is not None and reduction < 0.20:
        reasons.append("NEGATIVE_REDUCTION_BELOW_0_20")
    else:
        reasons.append("METRIC_IN_GRAY_ZONE")
    return "REVIEW_GRAY_ZONE", reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    reviews = json.loads(args.reviews.read_text())
    identity = hashlib.sha256("\n".join(sorted(
        row["proposal_id"] for row in selection["selected"]
    )).encode()).hexdigest()
    if identity != selection["identity_sha256"]:
        raise ValueError("selection identity mismatch")
    review_map = reviews["reviews"]
    missing = [
        row["proposal_id"] for row in selection["selected"]
        if row["proposal_id"] not in review_map
    ]
    if missing:
        raise ValueError(f"{len(missing)} shadow cards are not reviewed")
    semantic, random_rows, excluded = [], [], {"UNCERTAIN": 0, "BOX_WRONG": 0}
    filtered_litter = []
    for source in selection["selected"]:
        label_name = review_map[source["proposal_id"]]["label"]
        if label_name not in VALID:
            excluded[label_name] = excluded.get(label_name, 0) + 1
            continue
        row = {
            "proposal_id": source["proposal_id"],
            "device_code": source["device_code"],
            "stratum": source["shadow_stratum"],
            "weight": float(source["shadow_weight"]),
            "label": VALID[label_name],
            "passed": bool(source.get("passed", False)),
            "bbox": source["bbox"],
            "timestamp": source["timestamp"],
        }
        if source["shadow_stratum"] == "random_grid":
            random_rows.append(row)
        else:
            semantic.append(row)
            if row["label"] == 1 and not row["passed"]:
                filtered_litter.append(row)
    overall = metrics(semantic)
    by_camera = {
        device[-5:]: metrics([row for row in semantic if row["device_code"] == device])
        for device in sorted({row["device_code"] for row in semantic})
    }
    decision, reasons = decide(overall, by_camera, overall["positive_rows"])
    random_positive = sum(row["label"] == 1 for row in random_rows)
    performance = json.loads(args.performance.read_text())
    output = {
        "schema": "ground_litter_candidate_filter_new_date_shadow_eval_v1",
        "decision": decision, "decision_reasons": reasons,
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "reviews_sha256": hashlib.sha256(args.reviews.read_bytes()).hexdigest(),
        "performance_sha256": hashlib.sha256(args.performance.read_bytes()).hexdigest(),
        "reviewed": len(review_map), "semantic_scored": len(semantic),
        "excluded": excluded, "overall_weighted": overall,
        "by_camera_weighted": by_camera,
        "filtered_litter_count": len(filtered_litter),
        "filtered_litter": filtered_litter,
        "random_grid": {
            "scored": len(random_rows), "litter_rows": random_positive,
            "litter_fraction_unweighted": (
                random_positive / len(random_rows) if random_rows else None
            ),
            "note": "Generator miss diagnostic; not part of candidate-filter recall.",
        },
        "performance": {
            key: performance.get(key) for key in (
                "ticks", "candidates", "tick_seconds_p95", "tick_seconds_p99",
                "tick_seconds_max", "latest_wins",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
