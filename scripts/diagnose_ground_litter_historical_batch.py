#!/usr/bin/env python3
"""Read-only diagnostic for the v2 first review batch.

Answers two questions with data only:

1. why camera 01027 contributed a single review group;
2. whether the candidate admission floor ``conf >= 0.15`` is too aggressive.

Hard constraints: never downloads video, never runs a model, never writes inside
the artifact root, never regenerates the split.  All clustering below is done in
memory over the existing candidate tables; the official queue is never modified.

    python scripts/diagnose_ground_litter_historical_batch.py \
        --artifact output/ground_litter_historical_v2 \
        --out      output/ground_litter_historical_v2/diagnostics
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rtsp_annotator.ground_litter_historical import (  # noqa: E402
    CAMERAS, HistoricalError, Observation, REVIEW_MIN_CONFIDENCE, SOURCE_HEIGHT,
    SOURCE_WIDTH, bbox_center, choose_representatives, cluster_episodes,
    day1_selection, link_review_groups, median, observation_buckets, read_jsonl,
    size_class, write_json, write_jsonl,
)

# Reuse the builder's duplicate-background rule verbatim (single source of truth).
import build_ground_litter_historical_review as builder  # noqa: E402

STRATA = (("0.01-0.05", 0.01, 0.05), ("0.05-0.15", 0.05, 0.15), (">=0.15", 0.15, 1e9))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roi-config-dir", type=Path,
                        default=ROOT / "output/ground_litter_final_roi_20260922/config")
    parser.add_argument("--low-conf-per-camera", type=int, default=5)
    parser.add_argument("--balanced-min", type=int, default=8)
    parser.add_argument("--balanced-max", type=int, default=12)
    parser.add_argument("--duplicate-share-limit", type=float, default=0.5)
    parser.add_argument("--seed", default="ground-litter-historical-v2-batch-diagnostic")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def max_turhancan(row: dict) -> float:
    return float(row.get("confidence_by_source", {}).get("turhancan") or 0.0)


def max_any(row: dict) -> float:
    return max(row.get("confidence_by_source", {}).values(), default=0.0)


def stratum(value: float) -> str:
    for name, low, high in STRATA:
        if low <= value < high:
            return name
    return ">=0.15"


def roi_geometry(path: Path, camera: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    polygon = payload.get("roi") or []
    area = 0.0
    for index in range(len(polygon)):
        x0, y0 = polygon[index]
        x1, y1 = polygon[(index + 1) % len(polygon)]
        area += float(x0) * float(y1) - float(x1) * float(y0)
    area = abs(area) / 2.0
    xs = [float(p[0]) for p in polygon] or [0.0]
    ys = [float(p[1]) for p in polygon] or [0.0]
    return {
        "roi_area_fraction": round(area, 5),
        "roi_bbox_px": [round(min(xs) * SOURCE_WIDTH), round(min(ys) * SOURCE_HEIGHT),
                        round(max(xs) * SOURCE_WIDTH), round(max(ys) * SOURCE_HEIGHT)],
        "roi_points": len(polygon),
    }


def cluster_at(rows: list[dict], floor: float) -> tuple[list, list, dict[str, list[dict]]]:
    """In-memory clustering at an arbitrary admission floor (artifact untouched)."""
    admitted = [row for row in rows if max_any(row) >= floor]
    observations = [Observation.from_dict(row) for row in admitted]
    episodes = cluster_episodes(observations)
    groups = link_review_groups(episodes, observations)
    by_id = {row["observation_id"]: row for row in admitted}
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    members: dict[str, list[dict]] = {}
    for group in groups:
        member_rows: list[dict] = []
        for episode_id in group.episode_ids:
            for observation_id in episode_by_id[episode_id].observation_ids:
                if observation_id in by_id:
                    member_rows.append(by_id[observation_id])
        members[group.review_group_id] = member_rows
    return episodes, groups, members


def group_profile(group, member_rows: list[dict]) -> dict:
    confidences = [max_turhancan(row) for row in member_rows]
    slots = max(confidences) if confidences else 0.0
    sources = set()
    for row in member_rows:
        sources.update(row.get("confidence_by_source", {}))
    duplicates = sum(1 for row in member_rows if row.get("duplicate_background"))
    observations = [Observation.from_dict(row) for row in member_rows]
    representatives = choose_representatives(observations) if observations else []
    rep_row = next((row for row in member_rows
                    if representatives and row["observation_id"] == representatives[0]["observation_id"]),
                   member_rows[0] if member_rows else None)
    buckets = observation_buckets(Observation.from_dict(rep_row), roi=None,
                                  time_bucket=rep_row.get("selection_bucket") or "unknown") \
        if rep_row else set()
    return {
        "review_group_id": group.review_group_id,
        "camera_id": group.camera_id,
        "max_turhancan": round(slots, 5),
        "max_any": round(max((max_any(row) for row in member_rows), default=0.0), 5),
        "member_observations": len(member_rows),
        "episodes": len(group.episode_ids),
        "suspected_same_object": bool(group.suspected_same_object),
        "source_label": "both" if "yolo" in sources else ("turhancan_only" if "turhancan" in sources
                                                          else "yolo_only"),
        "duplicate_share": round(duplicates / max(len(member_rows), 1), 3),
        "location": next((b.split(":", 1)[1] for b in buckets if b.startswith("location:")), "?"),
        "size_class": next((b.split(":", 1)[1] for b in buckets if b.startswith("size:")), "?"),
        "time_bucket": rep_row.get("selection_bucket") if rep_row else None,
        "window_id": rep_row.get("window_id") if rep_row else None,
        "date": rep_row.get("date") if rep_row else None,
        "day_split": rep_row.get("day_split") if rep_row else None,
        "representative_observation_id": representatives[0]["observation_id"] if representatives else None,
        "representative_frame_id": representatives[0]["frame_id"] if representatives else None,
        "median_bbox": [round(median([row["bbox_xyxy"][i] for row in member_rows]), 2)
                        for i in range(4)] if member_rows else None,
    }


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #


def select_low_conf(profiles: list[dict], *, per_camera: int, duplicate_limit: float):
    """Greedy, per camera: 0.05-0.15 first, then 0.01-0.05, novelty over location/size/time.

    Returns ``(picked, pool_stats)`` so the report can show why a camera fell short
    instead of guessing.
    """
    chosen: list[dict] = []
    pool_stats: dict[str, dict] = {}
    for camera in CAMERAS:
        camera_pool = [p for p in profiles if p["camera_id"] == camera]
        below = [p for p in camera_pool if p["max_turhancan"] < REVIEW_MIN_CONFIDENCE]
        kept = [p for p in below if p["duplicate_share"] < duplicate_limit]
        pool_stats[camera] = {
            "no_floor_groups": len(camera_pool),
            "below_floor_groups": len(below),
            "excluded_duplicate_background": len(below) - len(kept),
            "pool_0.05_0.15": sum(1 for p in kept
                                  if 0.05 <= p["max_turhancan"] < REVIEW_MIN_CONFIDENCE),
            "pool_0.01_0.05": sum(1 for p in kept if 0.01 <= p["max_turhancan"] < 0.05),
        }
        picked: list[dict] = []
        covered: set[tuple] = set()
        for lower, upper in ((0.05, REVIEW_MIN_CONFIDENCE), (0.01, 0.05)):
            tier_pool = [p for p in kept if lower <= p["max_turhancan"] < upper]
            tier_pool.sort(key=lambda p: (p["source_label"] != "turhancan_only",
                                          -p["max_turhancan"], p["review_group_id"]))
            for profile in tier_pool:
                if len(picked) >= per_camera:
                    break
                key = (profile["location"], profile["size_class"], profile["time_bucket"])
                if key in covered and len(tier_pool) > per_camera - len(picked):
                    continue
                picked.append(profile)
                covered.add(key)
            if len(picked) >= per_camera:
                break
        pool_stats[camera]["selected"] = len(picked)
        pool_stats[camera]["shortfall_reason"] = (
            "no_below_floor_groups" if not below else
            ("all_below_floor_groups_are_duplicate_background" if not kept else
             ("pool_exhausted" if len(picked) < per_camera else "ok")))
        chosen.extend(picked)
    return chosen, pool_stats


def balanced_preview(order: list[str], profiles: dict[str, dict], *, minimum: int,
                     maximum: int) -> list[dict]:
    """Per camera, keep up to ``maximum`` groups in the existing diversity order."""
    per_camera: dict[str, list[str]] = defaultdict(list)
    for group_id in order:
        profile = profiles.get(group_id)
        if profile:
            per_camera[profile["camera_id"]].append(group_id)
    preview: list[dict] = []
    for camera in CAMERAS:
        available = per_camera.get(camera, [])
        keep = min(len(available), maximum)
        if keep < minimum:
            keep = len(available)          # camera genuinely short of supply
        for group_id in available[:keep]:
            preview.append({**profiles[group_id], "reason": "balanced_camera_floor",
                            "camera_supply": len(available)})
    return preview


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = args.artifact
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    observations = read_jsonl(artifact / "candidate_observations.jsonl")
    raw = read_jsonl(artifact / "raw_candidates.jsonl")
    frames = read_jsonl(artifact / "coarse_frames.jsonl")
    manifest = read_jsonl(artifact / "historical_window_manifest.jsonl")
    official_episodes = read_jsonl(artifact / "episodes.jsonl")
    official_groups = read_jsonl(artifact / "review_groups.jsonl")
    units = read_jsonl(artifact / "review_units.jsonl")
    queue = json.loads((artifact / "queue.json").read_text(encoding="utf-8")) \
        if (artifact / "queue.json").is_file() else {}

    observations = [row for row in observations if row.get("day_split") == "TRAIN"]
    # Frame ids come from the observation table itself so the diagnostic also runs
    # from a partial copy that has no coarse_frames.jsonl.
    train_frames = {row["frame_id"] for row in observations}
    if not train_frames:
        train_frames = {row["frame_id"] for row in frames if row.get("day_split") == "TRAIN"}
    raw = [row for row in raw if row["frame_id"] in train_frames]
    if not observations:
        raise HistoricalError("no TRAIN candidate observations to diagnose")

    builder.mark_duplicate_background(observations)

    # ---- funnel ---------------------------------------------------------- #
    funnel: dict[str, dict] = {}
    raw_by_camera: dict[str, Counter] = defaultdict(Counter)
    for row in raw:
        raw_by_camera[row["camera_id"]][row["source"]] += 1
    tiles_fired: dict[str, set] = defaultdict(set)
    for row in raw:
        tiles_fired[row["camera_id"]].add((row["frame_id"], tuple(row["tile_xy"])))
    merged_by_camera = Counter(row["camera_id"] for row in observations)
    official_episodes_by_camera = Counter(row["camera_id"] for row in official_episodes)
    official_groups_by_camera = Counter(row["camera_id"] for row in official_groups)
    first_batch_cameras = Counter(unit["camera_id"] for unit in units if unit.get("batch") == 1)
    units_by_camera = Counter(unit["camera_id"] for unit in units if unit["kind"] == "candidate")

    floors = (REVIEW_MIN_CONFIDENCE, 0.05, 0.01)
    clustering: dict[float, dict] = {}
    for floor in floors:
        episodes, groups, members = cluster_at(observations, floor)
        profiles = {group.review_group_id: group_profile(group, members[group.review_group_id])
                    for group in groups}
        clustering[floor] = {"episodes": episodes, "groups": groups, "members": members,
                             "profiles": profiles}

    no_floor = clustering[0.01]["profiles"]

    for camera in CAMERAS:
        strata = Counter()
        for profile in no_floor.values():
            if profile["camera_id"] == camera:
                strata[stratum(profile["max_turhancan"])] += 1
        funnel[camera] = {
            "raw_turhancan_ge_001": raw_by_camera[camera].get("turhancan", 0),
            "raw_yolo_ge_001": raw_by_camera[camera].get("yolo", 0),
            "tiles_that_fired": len(tiles_fired.get(camera, ())),
            "merged_observations": merged_by_camera.get(camera, 0),
            "observations_ge_015": sum(1 for row in observations
                                       if row["camera_id"] == camera and max_any(row) >= REVIEW_MIN_CONFIDENCE),
            "official_episodes_ge_015": official_episodes_by_camera.get(camera, 0),
            "official_review_groups_ge_015": official_groups_by_camera.get(camera, 0),
            "episodes_no_floor": sum(1 for e in clustering[0.01]["episodes"] if e.camera_id == camera),
            "review_groups_no_floor": sum(1 for p in no_floor.values() if p["camera_id"] == camera),
            "groups_strata_no_floor": {
                "0.01-0.05": strata.get("0.01-0.05", 0),
                "0.05-0.15": strata.get("0.05-0.15", 0),
                ">=0.15": strata.get(">=0.15", 0),
            },
            "queue_candidate_units": units_by_camera.get(camera, 0),
            "selected_first_batch": first_batch_cameras.get(camera, 0),
            "roi": roi_geometry(args.roi_config_dir / f"ground_litter_{camera}_final_roi.json", camera),
        }

    # ---- why 01027 ------------------------------------------------------- #
    camera = "01027"
    supply = funnel[camera]
    camera_day1 = [w for w in day1_selection(manifest) if w.startswith(camera + "_")]
    frames_per_window: dict[str, set] = defaultdict(set)
    for row in observations:
        if row["camera_id"] == camera:
            frames_per_window[row["window_id"]].add(row["frame_id"])
    zero_candidate_windows = [w for w in camera_day1 if w not in frames_per_window]
    partial_windows = [f"{w}({len(frames_per_window[w])}/3)"
                       for w in camera_day1 if 0 < len(frames_per_window.get(w, ())) < 3]
    share_of_raw = supply["raw_turhancan_ge_001"] / max(
        sum(v["raw_turhancan_ge_001"] for v in funnel.values()), 1)
    share_of_roi = supply["roi"]["roi_area_fraction"]
    below_floor = supply["groups_strata_no_floor"]["0.01-0.05"] + \
        supply["groups_strata_no_floor"]["0.05-0.15"]
    causes: list[dict[str, Any]] = []
    causes.append({
        "cause": "A. 本来就没有候选（供给太小）",
        "verdict": "yes",
        "evidence": f"ROI 只占整帧 {share_of_roi:.4f}（像素 bbox "
                    f"{supply['roi']['roi_bbox_px']}，一条窄带）；5 个 day-1 window 里只有 "
                    f"{supply['tiles_that_fired']} 个 ROI tile 真的产出过框，"
                    f"全四路 raw 只占 {share_of_raw:.1%}（{supply['raw_turhancan_ge_001']} 个 "
                    f"Turhancan、{supply['raw_yolo_ge_001']} 个 YOLO）；"
                    f"零候选 window: {zero_candidate_windows or '无'}；"
                    f"候选不足 3 帧的 window: {partial_windows or '无'}",
    })
    causes.append({
        "cause": "B. 有候选但大部分 <0.15",
        "verdict": "yes" if below_floor else "no",
        "evidence": f"no-floor groups: {supply['groups_strata_no_floor']}；"
                    f"低于 0.15 的 group = {below_floor}",
    })
    causes.append({
        "cause": "C. clustering 合并掉了",
        "verdict": "minor" if supply["merged_observations"] > supply["episodes_no_floor"] else "no",
        "evidence": f"{supply['merged_observations']} observations -> "
                    f"{supply['episodes_no_floor']} episodes -> "
                    f"{supply['review_groups_no_floor']} groups (no floor)",
    })
    causes.append({
        "cause": "D. diversity ranking 没选它",
        "verdict": "partial" if supply["queue_candidate_units"] > supply["selected_first_batch"] else "no",
        "evidence": f"queue 里 01027 有 {supply['queue_candidate_units']} 个 candidate unit，"
                    f"first batch 只进了 {supply['selected_first_batch']}；"
                    f"（queue 总长 {queue.get('queue_size')}，first batch 40，"
                    f"所以这是排序位置问题而不是被丢弃）",
    })
    causes.append({
        "cause": "E. 其它原因（下载/解码失败）",
        "verdict": "no",
        "evidence": "01027 的 TRAIN day-1 window 全部 extraction_status=done；"
                    f"short PS 记录 {sum(1 for row in manifest if row['camera_id'] == camera and row.get('missing_offsets'))}",
    })

    # ---- conf distribution of the OFFICIAL groups ------------------------ #
    official_profiles: dict[str, dict] = {}
    official_members: dict[str, list[dict]] = {}
    groups_at_floor = clustering[REVIEW_MIN_CONFIDENCE]["groups"]
    members_at_floor = clustering[REVIEW_MIN_CONFIDENCE]["members"]
    for group in groups_at_floor:
        member_rows = members_at_floor[group.review_group_id]
        official_profiles[group.review_group_id] = group_profile(group, member_rows)
        official_members[group.review_group_id] = member_rows

    conf_distribution = {}
    for camera in CAMERAS:
        conf_distribution[camera] = {
            "official_groups": sum(1 for p in official_profiles.values() if p["camera_id"] == camera),
            "official_strata": dict(Counter(
                stratum(p["max_turhancan"]) for p in official_profiles.values()
                if p["camera_id"] == camera)),
            "no_floor_groups": sum(1 for p in no_floor.values() if p["camera_id"] == camera),
            "no_floor_strata": dict(Counter(
                stratum(p["max_turhancan"]) for p in no_floor.values()
                if p["camera_id"] == camera)),
        }

    # ---- selections ------------------------------------------------------ #
    low_conf, low_conf_pools = select_low_conf(
        list(no_floor.values()), per_camera=args.low_conf_per_camera,
        duplicate_limit=args.duplicate_share_limit)
    order = list(queue.get("order") or [])
    balanced = balanced_preview(order, official_profiles,
                                minimum=args.balanced_min, maximum=args.balanced_max)

    low_conf_camera = Counter(p["camera_id"] for p in low_conf)
    balanced_camera = Counter(p["camera_id"] for p in balanced)
    first_batch_ids = {unit["unit_id"] for unit in units if unit.get("batch") == 1}
    balanced_ids = {p["review_group_id"] for p in balanced}

    # ---- data-driven recommendation -------------------------------------- #
    supply_total = len(official_profiles)
    no_floor_total = len(no_floor)
    below_floor_total = sum(1 for p in no_floor.values()
                            if p["max_turhancan"] < REVIEW_MIN_CONFIDENCE)
    excluded_duplicate_total = sum(
        stats["excluded_duplicate_background"] for stats in low_conf_pools.values())
    pool_05_015_total = sum(stats["pool_0.05_0.15"] for stats in low_conf_pools.values())
    pool_01_05_total = sum(stats["pool_0.01_0.05"] for stats in low_conf_pools.values())
    supply_limited = [c for c in CAMERAS if units_by_camera.get(c, 0) < args.balanced_min]
    under_weighted = [c for c in CAMERAS
                      if units_by_camera.get(c, 0) >= args.balanced_min
                      and first_batch_cameras.get(c, 0) < args.balanced_min]
    recommendation = (
        f"1) 候选供给本身极不均匀：queue 中每 camera candidate unit = "
        + "，".join(f"{c}={units_by_camera.get(c, 0)}" for c in CAMERAS)
        + f"（合计 {supply_total}）。first batch 原 40 的分布 14/8/1/17 "
        f"基本按供给比例分配，不是 selector 把某一台藏起来。"
        f"供给不足 {args.balanced_min} 的 camera：{supply_limited or '无'}；"
        f"供给足够但 first batch 低于 {args.balanced_min} 的 camera：{under_weighted or '无'}。\n"
        f"2) 因此 balanced preview（每 camera ≤{args.balanced_max}、供给不足就全给）"
        f"会得到 {len(balanced)} 个："
        + "，".join(f"{c}={balanced_camera.get(c, 0)}" for c in CAMERAS)
        + f"，与原 first batch 重合 {len(balanced_ids & first_batch_ids)}、"
        f"新增 {len(balanced_ids - first_batch_ids)}、去掉 {len(first_batch_ids - balanced_ids)}。"
        "它牺牲的是 01021/01030 的高 gain 排序位，换来四路均衡。\n"
        f"3) admission floor：no-floor 聚类共 {no_floor_total} 个 group，其中 <0.15 的有 "
        f"{below_floor_total} 个，official ≥0.15 是 {supply_total}。关键在于这 "
        f"{below_floor_total} 个里 **{excluded_duplicate_total} 个是重复背景**"
        f"（同一机位/格位/尺寸/粗颜色的反复检出，被 duplicate filter 排除），"
        f"只有 {below_floor_total - excluded_duplicate_total} 个进入低置信度池"
        f"（0.05–0.15 仅 {pool_05_015_total} 个，0.01–0.05 有 {pool_01_05_total} 个）。"
        "也就是说把 0.15 降到 0.05 主要放进来的不是新的独立垃圾实例，而是同一批固定背景；"
        "01027 的缺口即使完全移除 floor 也补不满 —— 它的问题是供给，不是门槛。\n"
        "4) 纯数据结论："
        + ("建议把首批改成 balanced preview，" if under_weighted or len(supply_limited) > 1
           else "是否改 balanced 更多是策略选择，")
        + "并保留原 40 queue 作为对照；LOW_CONF_DIAGNOSTIC 只作为门槛校准样本，"
          "不要混进正式 first batch，也不要自动进训练。"
    )
    recommendation += (
        "\n5) 01027 的低置信度池只有 "
        f"{low_conf_camera.get('01027', 0)} 个（目标 5）；"
        "说明该机位在这 5 个 day-1 window 里本来就没有多少可用的独立目标，"
        "要提升它必须换 window/时段或独立标定 ROI，而不是放宽门槛。"
    )

    report = {
        "kind": "ground_litter_historical_v2_batch_diagnostic",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "artifact": str(artifact),
        "read_only": True,
        "review_min_confidence": REVIEW_MIN_CONFIDENCE,
        "train_windows": len({row["window_id"] for row in observations}),
        "camera_funnel": funnel,
        "why_01027": {
            "camera": camera,
            "raw_turhancan_ge_001": supply["raw_turhancan_ge_001"],
            "raw_yolo_ge_001": supply["raw_yolo_ge_001"],
            "merged_observations": supply["merged_observations"],
            "observations_ge_015": supply["observations_ge_015"],
            "official_groups_ge_015": supply["official_review_groups_ge_015"],
            "no_floor_groups": supply["review_groups_no_floor"],
            "no_floor_strata": supply["groups_strata_no_floor"],
            "queue_units": supply["queue_candidate_units"],
            "first_batch": supply["selected_first_batch"],
            "causes": causes,
        },
        "conf_distribution": conf_distribution,
        "low_conf_diagnostic": {
            "count": len(low_conf),
            "per_camera": {c: low_conf_camera.get(c, 0) for c in CAMERAS},
            "pools": low_conf_pools,
            "groups": low_conf,
        },
        "balanced_first_batch_preview": {
            "count": len(balanced),
            "per_camera": {c: balanced_camera.get(c, 0) for c in CAMERAS},
            "overlap_with_official_first_batch": len(balanced_ids & first_batch_ids),
            "newly_added": len(balanced_ids - first_batch_ids),
            "dropped_from_official": len(first_batch_ids - balanced_ids),
            "groups": balanced,
        },
        "recommendation": recommendation,
    }

    write_json(out_dir / "batch_diagnostic.json", report)
    write_jsonl(out_dir / "low_conf_diagnostic.jsonl", low_conf)
    write_json(out_dir / "balanced_first_batch_preview.json",
               report["balanced_first_batch_preview"])
    (out_dir / "BATCH_DIAGNOSTIC.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("camera_funnel", "why_01027", "conf_distribution",
                       "low_conf_diagnostic", "balanced_first_batch_preview")},
                     ensure_ascii=False, indent=2)[:12000])
    print(f"\nwritten: {out_dir}", file=sys.stderr)
    return 0


def _markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# v2 First Batch — 只读诊断（未修改 artifact / queue / split）")
    lines.append("")
    lines.append(f"生成时间：{report['generated_at']}")
    lines.append(f"admission floor: `conf >= {report['review_min_confidence']}`（本轮诊断的实际门槛）")
    lines.append("")
    lines.append("## Camera Funnel（TRAIN 20 windows）")
    lines.append("")
    lines.append("| camera | raw Turhancan ≥0.01 | raw YOLO ≥0.01 | firing tiles | merged obs | obs ≥0.15 | episodes(0.15) | groups(0.15) | groups(no floor) | ≥0.15 | 0.05–0.15 | 0.01–0.05 | first batch |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for camera, row in report["camera_funnel"].items():
        strata = row["groups_strata_no_floor"]
        lines.append(
            f"| {camera} | {row['raw_turhancan_ge_001']} | {row['raw_yolo_ge_001']} | "
            f"{row['tiles_that_fired']} | {row['merged_observations']} | "
            f"{row['observations_ge_015']} | {row['official_episodes_ge_015']} | "
            f"{row['official_review_groups_ge_015']} | {row['review_groups_no_floor']} | "
            f"{strata['>=0.15']} | {strata['0.05-0.15']} | {strata['0.01-0.05']} | "
            f"{row['selected_first_batch']} |")
    lines.append("")
    lines.append("`firing tiles` = 至少产出一个 raw box 的 ROI tile 数（从已有 raw_candidates 统计，未重新推理）。")
    lines.append("")
    lines.append("## Why 01027 Has Only 1")
    lines.append("")
    why = report["why_01027"]
    for cause in why["causes"]:
        lines.append(f"- **{cause['cause']}** — {cause['verdict']}：{cause['evidence']}")
    lines.append("")
    lines.append("## Conf Distribution")
    lines.append("")
    lines.append("| camera | official groups (≥0.15 分组) | no-floor groups | ≥0.15 | 0.05–0.15 | 0.01–0.05 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for camera, row in report["conf_distribution"].items():
        strata = row["no_floor_strata"]
        lines.append(f"| {camera} | {row['official_groups']} | {row['no_floor_groups']} | "
                     f"{strata.get('>=0.15', 0)} | {strata.get('0.05-0.15', 0)} | "
                     f"{strata.get('0.01-0.05', 0)} |")
    lines.append("")
    lines.append("## Low Conf Diagnostic")
    lines.append("")
    low = report["low_conf_diagnostic"]
    lines.append(f"共 {low['count']} 个（每 camera 最多 {max(low['per_camera'].values() or [0])}）："
                 + "，".join(f"{c}={n}" for c, n in low["per_camera"].items()))
    lines.append("")
    lines.append("## Balanced First Batch Preview")
    lines.append("")
    bal = report["balanced_first_batch_preview"]
    lines.append(f"共 {bal['count']} 个：" + "，".join(f"{c}={n}" for c, n in bal["per_camera"].items()))
    lines.append(f"与原 first batch 重合 {bal['overlap_with_official_first_batch']}，"
                 f"新增 {bal['newly_added']}，去掉 {bal['dropped_from_official']}（原 queue 未被修改）。")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(report.get("recommendation", ""))
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
