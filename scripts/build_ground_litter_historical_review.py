#!/usr/bin/env python3
"""Rapid Dataset v2 review builder: episodes -> review groups -> diversity queue.

Turns the coarse-pass candidate observations into *unique-instance* review units:

* ``episode_id``  — one continuous litter presence, associated by camera / time /
  source position / size (spec §9.1);
* ``review_group_id`` — far-apart episodes suspected to be the same physical
  object, which a human resolves with SAME / NEW / UNCERTAIN (spec §9.3);
* one NORMAL representative frame per group, plus a hard-state frame only when a
  real appearance change justifies it (spec §11);
* a greedy multi-dimension diversity order so the first batches maximise new
  coverage instead of repeating the same fixed litter (spec §12);
* the fixed blind ROI set (spec §17).

    python scripts/build_ground_litter_historical_review.py \
        --artifact /home/sf01/ground-litter-historical-v2/artifact
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_historical import (  # noqa: E402
    BLIND_FRAMES_PER_CAMERA, CAMERAS, DEFAULT_SEED, FIRST_BATCH_SIZE, FULL_QUEUE_SIZE,
    HistoricalError, Observation, QueueEntry, REVIEW_MIN_CONFIDENCE, SOURCE_HEIGHT,
    SOURCE_WIDTH, bbox_center, blind_frame_selection, candidate_tier,
    choose_representatives, cluster_episodes, greedy_diversity_select,
    link_review_groups, observation_buckets, read_jsonl, size_class, write_json,
    write_jsonl,
)
from rtsp_annotator.ground_litter_classical_proposal import (  # noqa: E402
    classic_cv_proposals, dedupe_proposals,
)

TIER_ORDER = ("P1", "P2", "P3", "P3b", "LOW")
BLIND_BATCH = 4
RESERVE_BATCH = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--first-batch", type=int, default=FIRST_BATCH_SIZE)
    parser.add_argument("--queue-size", type=int, default=FULL_QUEUE_SIZE)
    parser.add_argument("--batch-split", default="50,50")
    parser.add_argument("--blind-per-camera", type=int, default=BLIND_FRAMES_PER_CAMERA)
    parser.add_argument("--splits", default="TRAIN",
                        help="day splits allowed into the training review queue")
    parser.add_argument("--no-classical", action="store_true",
                        help="skip candidate C when cv2 is unavailable")
    return parser.parse_args(argv)


def mark_duplicate_background(rows: list[dict]) -> dict[str, Any]:
    """Flag repeated appearances of the same fixed object.

    Spec §18: 100 false positives on the same floor tile are still 100 FP at
    runtime, but only one frame is worth reviewing.  The first observation of a
    (camera, location cell, size class, coarse colour) cluster is primary; the
    rest are duplicates.
    """
    seen: dict[tuple, str] = {}
    duplicates = 0
    for row in sorted(rows, key=lambda item: (item["camera_id"], item["timestamp"],
                                              item["observation_id"])):
        appearance = row.get("appearance") or [0, 0, 0]
        cx, cy = bbox_center(row["bbox_xyxy"])
        cell = (int(cx / SOURCE_WIDTH * 4), int(cy / SOURCE_HEIGHT * 3))
        colour = tuple(int(float(v) // 48) for v in appearance[:3])
        key = (row["camera_id"], cell, size_class(row["bbox_xyxy"]), colour)
        if key in seen:
            row["duplicate_background"] = True
            row["duplicate_of_window"] = seen[key]
            duplicates += 1
        else:
            row["duplicate_background"] = False
            seen[key] = row["window_id"]
    return {"clusters": len(seen), "duplicates": duplicates}


def build_candidates(artifact: Path, observation_row: dict, point: tuple[float, float],
                     *, allow_classical: bool) -> list[dict]:
    options: list[dict] = []
    boxes = observation_row.get("bbox_by_source") or {}
    labels = {"turhancan": "A — Turhancan 语义", "yolo": "B — Step 2B last.pt"}
    sources = {"turhancan": "semantic", "yolo": "yolo"}
    for name in ("turhancan", "yolo"):
        box = boxes.get(name)
        if not box:
            continue
        options.append({
            "candidate_id": "A" if name == "turhancan" else "B",
            "source": sources[name],
            "label": labels[name],
            "bbox_xyxy": [float(v) for v in box],
            "count": 1,
            "area": abs(float(box[2]) - float(box[0])) * abs(float(box[3]) - float(box[1])),
        })
    if allow_classical:
        frame_row = observation_row.get("_frame")
        if frame_row:
            try:
                import cv2
                image = cv2.imread(str(artifact / frame_row["image"]))
                if image is not None:
                    proposals = classic_cv_proposals(image, point[0], point[1])
                    if proposals:
                        best = max(proposals, key=lambda item: (item["contains_point"], item["area"]))
                        options.append({
                            "candidate_id": "C",
                            "source": "classical",
                            "label": f"C — classical {best['label']}",
                            "bbox_xyxy": best["bbox_xyxy"],
                            "count": 1,
                            "area": best["area"],
                        })
            except Exception:
                pass
    return dedupe_proposals(options)


def resolve_representatives(artifact: Path, rows: list[dict], frame_by_id: dict,
                            obs_by_id: dict, *, allow_classical: bool = True) -> list[dict]:
    observations = [Observation.from_dict(row) for row in rows]
    episodes = cluster_episodes(observations)
    groups = link_review_groups(episodes, observations)
    episode_by_id = {episode.episode_id: episode for episode in episodes}

    units: list[dict] = []
    for group in groups:
        member_ids: list[str] = []
        for episode_id in group.episode_ids:
            episode = episode_by_id[episode_id]
            member_ids.extend(episode.observation_ids)
        member_rows = [obs_by_id[oid] for oid in member_ids if oid in obs_by_id]
        if not member_rows:
            continue
        member_obs = [Observation.from_dict(row) for row in member_rows]
        representatives = choose_representatives(member_obs)
        rep_row = obs_by_id[representatives[0]["observation_id"]]
        tiers = [candidate_tier(obs) for obs in member_obs]
        tier = min(tiers, key=lambda name: TIER_ORDER.index(name) if name in TIER_ORDER else 99)
        point = bbox_center(rep_row["bbox_xyxy"])
        candidates = build_candidates(artifact, rep_row, point,
                                      allow_classical=allow_classical)

        # Spec §11: one NORMAL representative, plus at most two justified hard
        # states.  Everything else stays catalogued in episodes.jsonl and is
        # deliberately not sent to the human.
        unit_observations = []
        for representative in representatives:
            observation_id = representative["observation_id"]
            observation = next(item for item in member_obs
                               if item.observation_id == observation_id)
            row = obs_by_id[observation_id]
            frame = frame_by_id.get(row["frame_id"], {})
            unit_observations.append({
                "observation_id": row["observation_id"],
                "frame_id": row["frame_id"],
                "role": representative["role"],
                "offset_seconds": row["offset_seconds"],
                "record_start": frame.get("record_start"),
                "date": row["date"],
                "bbox_xyxy": [float(v) for v in row["bbox_xyxy"]],
                "source_label": observation.source_label,
                "confidence_by_source": row.get("confidence_by_source", {}),
                "contains_point": bool(
                    row["bbox_xyxy"][0] <= point[0] <= row["bbox_xyxy"][2]
                    and row["bbox_xyxy"][1] <= point[1] <= row["bbox_xyxy"][3]),
                "sampling": row.get("sampling", "coarse"),
                "duplicate_background": bool(row.get("duplicate_background")),
                "timestamp": row["timestamp"],
                "selection_bucket": row.get("selection_bucket"),
                "image": frame.get("image"),
            })
        unit_observations.sort(key=lambda item: (item["role"] != "normal", item["timestamp"]))
        units.append({
            "unit_id": group.review_group_id,
            "review_group_id": group.review_group_id,
            "kind": "candidate",
            "camera_id": group.camera_id,
            "day_split": rep_row.get("day_split"),
            "date": rep_row["date"],
            "timestamp": _stamp(frame_by_id.get(rep_row["frame_id"], {}).get("record_start"),
                                rep_row["offset_seconds"]),
            "tier": tier,
            "rank_score": round(max((max(row.get("confidence_by_source", {}).values(), default=0.0)
                                     for row in member_rows), default=0.0), 5),
            "blind": False,
            "suspected_same_object": group.suspected_same_object,
            "episode_count": len(group.episode_ids),
            "episode_ids": list(group.episode_ids),
            "link_evidence": group.link_evidence,
            "representative_observation_id": representatives[0]["observation_id"],
            "observations": unit_observations,
            "observation_count": len(member_rows),
            "candidates": candidates,
            "buckets": _unit_buckets(rep_row, frame_by_id.get(rep_row["frame_id"], {})),
            "selection_bucket": rep_row.get("selection_bucket"),
            "duplicate_share": round(
                sum(1 for row in member_rows if row.get("duplicate_background"))
                / max(len(member_rows), 1), 3),
        })
    return units, episodes, groups


def _stamp(record_start: str | None, offset: float) -> str:
    if not record_start:
        return ""
    from datetime import timedelta
    base = datetime.strptime(record_start, "%Y-%m-%d %H:%M:%S")
    return (base + timedelta(seconds=float(offset))).strftime("%Y-%m-%d %H:%M:%S")


def _unit_buckets(row: dict, frame: dict) -> list[str]:
    observation = Observation.from_dict(row)
    time_bucket = row.get("selection_bucket") or "unknown"
    return sorted(observation_buckets(observation, roi=frame.get("roi"),
                                      time_bucket=time_bucket))


def build_blind_units(frames: list[dict], *, per_camera: int, seed: str) -> list[dict]:
    chosen = blind_frame_selection(frames, per_camera=per_camera, seed=seed)
    units: list[dict] = []
    for index, row in enumerate(chosen, start=1):
        roi = row.get("roi") or []
        xs = [float(p[0]) for p in roi] or [0.0, 1.0]
        ys = [float(p[1]) for p in roi] or [0.0, 1.0]
        box = [min(xs) * SOURCE_WIDTH, min(ys) * SOURCE_HEIGHT,
               max(xs) * SOURCE_WIDTH, max(ys) * SOURCE_HEIGHT]
        units.append({
            "unit_id": f"bg-{index:05d}",
            "review_group_id": f"bg-{index:05d}",
            "kind": "blind",
            "camera_id": row["camera_id"],
            "day_split": row.get("day_split"),
            "date": row.get("date"),
            "timestamp": _stamp(row.get("record_start"), row.get("offset_seconds", 0.0)),
            "tier": "BLIND",
            "blind": True,
            "blind_kind": row.get("blind_kind"),
            "suspected_same_object": False,
            "episode_count": 0,
            "episode_ids": [],
            "link_evidence": {},
            "representative_observation_id": f"{row['frame_id']}_roi",
            "observations": [{
                "observation_id": f"{row['frame_id']}_roi",
                "frame_id": row["frame_id"],
                "role": "normal",
                "offset_seconds": row.get("offset_seconds", 0.0),
                "record_start": row.get("record_start"),
                "date": row.get("date"),
                "bbox_xyxy": box,
                "source_label": "unknown",
                "confidence_by_source": {},
                "contains_point": False,
                "sampling": row.get("sampling", "coarse"),
                "duplicate_background": False,
                "timestamp": 0.0,
                "selection_bucket": row.get("selection_bucket"),
                "image": row.get("image"),
            }],
            "candidates": [],
            "buckets": [f"camera:{row['camera_id']}", f"time:{row.get('selection_bucket')}"],
            "selection_bucket": row.get("selection_bucket"),
            "duplicate_share": 0.0,
        })
    return units


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = args.artifact
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    all_observations = read_jsonl(artifact / "candidate_observations.jsonl")
    all_frames = read_jsonl(artifact / "coarse_frames.jsonl")
    # DEV/FINAL must never reach the training review queue (spec §22 leakage).
    observation_rows = [row for row in all_observations if row.get("day_split") in splits]
    frame_rows = [row for row in all_frames if row.get("day_split") in splits]
    if not observation_rows:
        raise HistoricalError(
            f"no candidate observations for split(s) {sorted(splits)}; "
            "run the coarse pass first")
    # Both models are stored loosely at 0.01; only observations at or above the
    # review floor become review units.  Spec §6 ranks "Turhancan 极低分" last,
    # and the dry-run window showed p50 confidence 0.03 with 44/245 at >= 0.15.
    admitted = [row for row in observation_rows
                if max(row.get("confidence_by_source", {}).values(), default=0.0)
                >= REVIEW_MIN_CONFIDENCE]
    excluded_below_floor = len(observation_rows) - len(admitted)
    if not admitted:
        raise HistoricalError(
            f"every observation is below the {REVIEW_MIN_CONFIDENCE} review floor; "
            "nothing worth asking a human")
    observation_rows = admitted
    frame_by_id = {row["frame_id"]: row for row in frame_rows}
    obs_by_id = {row["observation_id"]: row for row in observation_rows}
    for row in observation_rows:
        row["_frame"] = frame_by_id.get(row["frame_id"], {})

    duplicate_stats = mark_duplicate_background(observation_rows)
    units, episodes, groups = resolve_representatives(
        artifact, observation_rows, frame_by_id, obs_by_id,
        allow_classical=not args.no_classical)

    entries = [QueueEntry(unit["unit_id"], unit["tier"], list(unit["buckets"]),
                          rank_score=float(unit.get("rank_score") or 0.0))
               for unit in units]
    ordered = greedy_diversity_select(entries, limit=min(args.queue_size, len(entries)),
                                      seed=args.seed)
    rank = {entry.review_group_id: index for index, entry in enumerate(ordered)}
    for unit in units:
        unit["queue_index"] = rank.get(unit["unit_id"], len(rank) + 1000)
        unit["batch"] = 0

    split = [int(value) for value in args.batch_split.split(",") if value.strip()]
    boundaries = [args.first_batch] + split
    ranked = sorted(units, key=lambda item: item["queue_index"])
    for index, unit in enumerate(ranked):
        if unit["queue_index"] >= len(rank):
            unit["batch"] = RESERVE_BATCH
            continue
        consumed = 0
        unit["batch"] = len(boundaries) + 1
        for step, size in enumerate(boundaries):
            consumed += size
            if index < consumed:
                unit["batch"] = step + 1
                break

    blind_units = build_blind_units(frame_rows, per_camera=args.blind_per_camera,
                                    seed=args.seed)
    for unit in blind_units:
        unit["queue_index"] = len(units) + blind_units.index(unit)
        unit["batch"] = BLIND_BATCH

    all_units = units + blind_units
    write_jsonl(artifact / "review_units.jsonl", all_units)
    write_jsonl(artifact / "episodes.jsonl", [episode.as_dict() for episode in episodes])
    write_jsonl(artifact / "review_groups.jsonl", [group.as_dict() for group in groups])
    write_jsonl(artifact / "blind_frames.jsonl", blind_units)

    batches: dict[str, list[str]] = defaultdict(list)
    for unit in all_units:
        batches[str(unit["batch"])].append(unit["unit_id"])
    queue = {
        "schema_version": "ground-litter-historical-v2",
        "seed": args.seed,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "total_units": len(all_units),
        "candidate_units": len(units),
        "blind_units": len(blind_units),
        "first_batch_size": args.first_batch,
        "queue_size": len(rank),
        "batch_plan": {"1": f"first {args.first_batch} candidate groups",
                       "2": f"next {split[0] if split else 0}",
                       "3": f"next {split[1] if len(split) > 1 else 0}",
                       str(BLIND_BATCH): "blind ROI frames (model hidden)",
                       str(RESERVE_BATCH): "reserve, re-ranked after batch 1"},
        "batches": {key: value for key, value in sorted(batches.items(), key=lambda kv: int(kv[0]))},
        "order": [unit["unit_id"] for unit in sorted(units, key=lambda u: u["queue_index"])],
    }
    write_json(artifact / "queue.json", queue)

    tier_counts = defaultdict(int)
    per_camera = defaultdict(lambda: defaultdict(int))
    for unit in units:
        tier_counts[unit["tier"]] += 1
        per_camera[unit["camera_id"]][unit["tier"]] += 1
    source_counts = defaultdict(int)
    for row in observation_rows:
        sources = tuple(sorted(row.get("confidence_by_source", {})))
        source_counts["both" if len(sources) > 1 else (sources[0] if sources else "none")] += 1
    summary = {
        "observations": len(observation_rows),
        "observations_before_floor": len(observation_rows) + excluded_below_floor,
        "excluded_below_review_floor": excluded_below_floor,
        "review_min_confidence": REVIEW_MIN_CONFIDENCE,
        "raw_source_support": dict(source_counts),
        "duplicate_background": duplicate_stats,
        "episodes": len(episodes),
        "review_groups": len(groups),
        "candidate_units": len(units),
        "blind_units": len(blind_units),
        "tier_counts": dict(tier_counts),
        "per_camera": {camera: dict(counts) for camera, counts in sorted(per_camera.items())},
        "first_batch": len(batches["1"]),
        "first_batch_per_camera": _per_camera_counts(
            [unit for unit in units if unit["batch"] == 1]),
        "suspected_same_object_units": sum(1 for unit in units if unit["suspected_same_object"]),
        "multi_frame_units": sum(1 for unit in units if len(unit["observations"]) > 1),
    }
    write_json(artifact / "review_build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def _per_camera_counts(units: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        counts[unit["camera_id"]] = counts.get(unit["camera_id"], 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
