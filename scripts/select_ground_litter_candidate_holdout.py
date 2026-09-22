#!/usr/bin/env python3
"""Freeze a score-blind, stratified holdout from unseen semantic candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


RANK_BANDS = ((1, 10), (11, 20), (21, 40), (41, 60))
SOURCES = ("semantic_full", "semantic_tile")


def rank_band(rank: int) -> str:
    for start, end in RANK_BANDS:
        if start <= rank <= end:
            return f"{start:02d}-{end:02d}"
    raise ValueError(f"rank outside frozen 1..60 candidate range: {rank}")


def stable_key(seed: str, proposal_id: str) -> str:
    return hashlib.sha256(f"{seed}:{proposal_id}".encode()).hexdigest()


def select_holdout(
    pools: list[dict], excluded_ids: set[str], per_stratum: int, seed: str,
) -> dict:
    selected: list[dict] = []
    strata: list[dict] = []
    seen_ids: set[str] = set()
    for pool in sorted(pools, key=lambda item: item["device_code"]):
        device = pool["device_code"]
        suffix = device[-5:]
        rows = []
        for source_row in pool["semantic"]:
            row = dict(source_row)
            proposal_id = row["proposal_id"]
            if proposal_id in seen_ids:
                raise ValueError(f"duplicate proposal_id across pools: {proposal_id}")
            seen_ids.add(proposal_id)
            if proposal_id not in excluded_ids:
                rows.append(row)
        for start, end in RANK_BANDS:
            band = f"{start:02d}-{end:02d}"
            for source in SOURCES:
                eligible = [
                    row for row in rows
                    if row["source"] == source and rank_band(int(row["rank"])) == band
                ]
                ordered = sorted(
                    eligible,
                    key=lambda row: stable_key(seed, row["proposal_id"]),
                )
                if len(ordered) < per_stratum:
                    raise ValueError(
                        f"insufficient candidates for {suffix}/{band}/{source}: "
                        f"{len(ordered)} < {per_stratum}"
                    )
                population = len(ordered)
                weight = population / per_stratum
                chosen = []
                for row in ordered[:per_stratum]:
                    clean = dict(row)
                    # The holdout selection contract forbids classifier-derived fields.
                    forbidden = {
                        key for key in clean
                        if key.startswith(("turhancan_", "yolo26s_"))
                        or key in {"novelty", "selection_reason", "reviewed_position_overlap"}
                    }
                    if forbidden:
                        raise ValueError(
                            f"classifier-derived fields found in pool row: {sorted(forbidden)}"
                        )
                    clean.update({
                        "holdout_stratum": f"{suffix}/{band}/{source}",
                        "holdout_population": population,
                        "holdout_sample_count": per_stratum,
                        "holdout_weight": weight,
                    })
                    selected.append(clean)
                    chosen.append(clean["proposal_id"])
                strata.append({
                    "device_code": device,
                    "rank_band": band,
                    "source": source,
                    "population": population,
                    "sample_count": per_stratum,
                    "sample_weight": weight,
                    "selected_ids_sha256": hashlib.sha256(
                        "\n".join(chosen).encode()
                    ).hexdigest(),
                })
    identities = sorted(row["proposal_id"] for row in selected)
    return {
        "schema": "ground_litter_candidate_holdout_v1",
        "selection_rule": "score_blind_camera_rank_source_stratified_sha256",
        "seed": seed,
        "count": len(selected),
        "per_stratum": per_stratum,
        "excluded_previously_reviewed": len(excluded_ids),
        "identity_sha256": hashlib.sha256("\n".join(identities).encode()).hexdigest(),
        "strata": strata,
        "selected": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--exclude-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=5)
    parser.add_argument("--seed", default="ground-litter-holdout-v1-20260922")
    args = parser.parse_args()
    if args.per_stratum <= 0:
        raise SystemExit("--per-stratum must be positive")
    pools = [json.loads(path.read_text()) for path in args.pool]
    prior = json.loads(args.exclude_selection.read_text())
    excluded = {row["proposal_id"] for row in prior["selected"]}
    payload = select_holdout(pools, excluded, args.per_stratum, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    by_device = Counter(row["device_code"][-5:] for row in payload["selected"])
    print(json.dumps({
        "count": payload["count"],
        "identity_sha256": payload["identity_sha256"],
        "per_device": dict(sorted(by_device.items())),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
