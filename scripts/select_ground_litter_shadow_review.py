#!/usr/bin/env python3
"""Select a deterministic, blinded, stratified review set for new-date shadow."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PER_LAYER = 12


def stable_key(row: dict, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{row['proposal_id']}".encode()).hexdigest()


def evenly(rows: list[dict], count: int, salt: str) -> list[dict]:
    if len(rows) <= count:
        return sorted(rows, key=lambda row: stable_key(row, salt))
    ordered = sorted(rows, key=lambda row: (float(row.get("margin", 0)), stable_key(row, salt)))
    indices = []
    for index in range(count):
        position = round(index * (len(ordered) - 1) / max(1, count - 1))
        if position not in indices:
            indices.append(position)
    if len(indices) < count:
        indices.extend(i for i in range(len(ordered)) if i not in indices)
    return [ordered[index] for index in indices[:count]]


def semantic_strata(rows: list[dict]) -> dict[str, list[dict]]:
    output = {}
    for passed, prefix in ((False, "filtered"), (True, "passed")):
        group = sorted(
            (row for row in rows if bool(row["passed"]) is passed),
            key=lambda row: (abs(float(row["margin"])), stable_key(row, prefix)),
        )
        split = (len(group) + 1) // 2
        output[f"{prefix}_near"] = group[:split]
        output[f"{prefix}_spread"] = group[split:]
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-layer", type=int, default=PER_LAYER)
    args = parser.parse_args()
    payload = json.loads(args.decisions.read_text())
    decisions = payload["decisions"]
    random_rows = []
    for path in args.pool:
        random_rows.extend(json.loads(path.read_text())["random"])
    devices = sorted({row["device_code"] for row in decisions})
    selected, report = [], {}
    for device in devices:
        camera = [row for row in decisions if row["device_code"] == device]
        strata = semantic_strata(camera)
        strata["random_grid"] = [row for row in random_rows if row["device_code"] == device]
        report[device[-5:]] = {}
        for name in ("filtered_near", "filtered_spread", "passed_near",
                     "passed_spread", "random_grid"):
            population = strata[name]
            chosen = evenly(population, args.per_layer, f"shadow-v1:{device}:{name}")
            weight = len(population) / len(chosen) if chosen else None
            for source in chosen:
                row = dict(source)
                row["shadow_stratum"] = name
                row["shadow_weight"] = weight
                # Do not add decision/score to review-data.json; build script only
                # publishes its explicit safe field list.
                selected.append(row)
            report[device[-5:]][name] = {
                "population": len(population), "selected": len(chosen),
                "weight": weight,
            }
    identity = hashlib.sha256("\n".join(sorted(
        row["proposal_id"] for row in selected
    )).encode()).hexdigest()
    output = {
        "schema": "ground_litter_candidate_filter_shadow_selection_v1",
        "decision_sha256": hashlib.sha256(args.decisions.read_bytes()).hexdigest(),
        "identity_sha256": identity, "count": len(selected),
        "per_camera": report, "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({"count": len(selected), "per_camera": report},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
