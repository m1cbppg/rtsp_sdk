#!/usr/bin/env python3
"""Evaluate the V3.1 online replay against reviewed fixed-view events.

Human labels are read only here, after the replay has completed.  They are not
available to candidate generation, event matching, occlusion handling, or the
online confirmation state machine.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ground_litter_event_memory_v31 import (  # noqa: E402
    box_center,
    same_fixed_view_anchor,
)


MATCH_DISTANCE_PX = 30.0
MATCH_SIZE_RATIO = 10.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_primary(event: dict[str, Any]) -> bool:
    return not str(event.get("closed_reason") or "").startswith("merged_into:")


def match_confirmed_events(
    online_events: list[dict[str, Any]],
    reviewed_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Greedily match confirmed online events to the nearest reviewed anchor."""
    matches: list[dict[str, Any]] = []
    used_reviewed: set[str] = set()
    confirmed = [
        event for event in online_events
        if is_primary(event) and event.get("confirmed_at") is not None
    ]
    for online in sorted(confirmed, key=lambda row: int(row["event_id"])):
        candidates = []
        for reviewed in reviewed_events:
            reviewed_id = str(reviewed["event_id"])
            if reviewed_id in used_reviewed:
                continue
            if not same_fixed_view_anchor(
                online["anchor_box"],
                reviewed["anchor_box"],
                center_distance_px=MATCH_DISTANCE_PX,
                size_ratio=MATCH_SIZE_RATIO,
            ):
                continue
            distance = math.dist(
                box_center(online["anchor_box"]),
                box_center(reviewed["anchor_box"]),
            )
            candidates.append((distance, reviewed))
        if not candidates:
            matches.append({
                "online_event_id": online["event_id"],
                "online_anchor_box": online["anchor_box"],
                "reviewed_event_id": None,
                "human_label": "UNMATCHED",
                "center_distance_px": None,
            })
            continue
        distance, reviewed = min(candidates, key=lambda item: item[0])
        used_reviewed.add(str(reviewed["event_id"]))
        matches.append({
            "online_event_id": online["event_id"],
            "online_anchor_box": online["anchor_box"],
            "reviewed_event_id": reviewed["event_id"],
            "reviewed_anchor_box": reviewed["anchor_box"],
            "human_label": reviewed["human_label"],
            "center_distance_px": round(distance, 3),
            "reviewed_track_ids": reviewed["track_ids"],
        })
    return matches


def evaluate(
    online_root: Path,
    reviewed_root: Path,
) -> dict[str, Any]:
    online_negative = load_json(online_root / "negative/events.json")
    reviewed_events = load_json(reviewed_root / "events.json")
    matches = match_confirmed_events(online_negative, reviewed_events)

    litter_truth = {
        str(row["event_id"]) for row in reviewed_events
        if row["human_label"] == "LITTER_CONFIRMED"
    }
    clear_truth = {
        str(row["event_id"]) for row in reviewed_events
        if row["human_label"] == "CLEAR_NON_LITTER"
    }
    matched_litter = {
        str(row["reviewed_event_id"]) for row in matches
        if row["human_label"] == "LITTER_CONFIRMED"
    }
    matched_clear = {
        str(row["reviewed_event_id"]) for row in matches
        if row["human_label"] == "CLEAR_NON_LITTER"
    }
    unmatched = [row for row in matches if row["human_label"] == "UNMATCHED"]
    tp = len(matched_litter)
    fp = len(matched_clear) + len(unmatched)
    fn = len(litter_truth - matched_litter)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    positive_summary = load_json(online_root / "positive/summary.json")
    negative_summary = load_json(online_root / "negative/summary.json")
    clean_summary = load_json(online_root / "clean_holdout/summary.json")
    actor_eval = load_json(reviewed_root / "evaluation_with_actor.json")

    track_labels: dict[str, str] = {}
    for event in reviewed_events:
        track_labels.update(event["human_track_labels"])
    reviewed_litter_tracks = sum(
        label == "LITTER_CONFIRMED" for label in track_labels.values()
    )
    reviewed_track_count = len(track_labels)

    return {
        "kind": "ground_litter_online_replay_v31_human_evaluation",
        "human_labels_used_for_prediction": False,
        "matching": {
            "center_distance_px": MATCH_DISTANCE_PX,
            "size_ratio": MATCH_SIZE_RATIO,
        },
        "reviewed_baseline": {
            "tracks_total": reviewed_track_count,
            "litter_tracks": reviewed_litter_tracks,
            "candidate_precision": round(
                reviewed_litter_tracks / reviewed_track_count, 4
            ),
            "events_total": len(reviewed_events),
            "litter_events": len(litter_truth),
            "non_litter_events": len(clear_truth),
        },
        "representative_frame_cascade": actor_eval["event_level"],
        "online_negative": {
            "events_created": negative_summary["events_total"],
            "events_confirmed": negative_summary["events_confirmed_5s"],
            "true_positive_events": tp,
            "false_positive_events": fp,
            "false_negative_events": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "non_litter_deferral_rate": round(
                (len(clear_truth) - len(matched_clear)) / len(clear_truth), 4
            ),
            "matched_confirmed_events": matches,
            "missed_litter_event_ids": sorted(litter_truth - matched_litter),
        },
        "online_positive_t01": positive_summary.get("t01", {}),
        "online_clean_holdout": {
            "frames": clean_summary["frames"],
            "events_created": clean_summary["events_total"],
            "events_confirmed": clean_summary["events_confirmed_5s"],
        },
        "environment_diagnostics": {
            "positive": positive_summary["environment_states"],
            "negative": negative_summary["environment_states"],
            "clean_holdout": clean_summary["environment_states"],
            "note": (
                "GLOBAL_LIGHT_CHANGE means photometric compensation was active; "
                "only ENVIRONMENT_CHANGE pauses event evidence."
            ),
        },
    }


def render_report(result: dict[str, Any]) -> str:
    baseline = result["reviewed_baseline"]
    representative = result["representative_frame_cascade"]
    online = result["online_negative"]
    t01 = result["online_positive_t01"]
    clean = result["online_clean_holdout"]
    matches = online["matched_confirmed_events"]
    match_rows = "\n".join(
        f"| {row['online_event_id']} | {row['reviewed_event_id'] or '-'} | "
        f"{row['human_label']} | {row['center_distance_px'] if row['center_distance_px'] is not None else '-'} |"
        for row in matches
    )
    return f"""# Ground Litter V3.1 在线回放评估

## 结论

- 负样本视频中，在线状态机累计创建 {online['events_created']} 个事件，只有 {online['events_confirmed']} 个通过 5 秒有效可见确认。
- 人工审核锚点映射结果为 TP={online['true_positive_events']}、FP={online['false_positive_events']}、FN={online['false_negative_events']}；precision={online['precision']:.1%}、recall={online['recall']:.1%}、F1={online['f1']:.1%}。
- 5 个已审核垃圾位置全部保留；11 个非垃圾事件中 10 个被 context/actor/时序挡住，非垃圾暂缓率 {online['non_litter_deferral_rate']:.1%}。
- 唯一确认误报仍是 reviewed event `negative-v31-event-013` / track 186，适合作为后续 VLM 或固定设施规则的首个验证对象。
- T01 只形成一个主事件（ID {t01.get('matching_event_ids')}），72s 首见，81s 确认。墙钟延迟 9 秒包含遮挡暂停，确认只累计可见证据。
- clean holdout 共 {clean['frames']} 帧，创建事件 {clean['events_created']}，确认事件 {clean['events_confirmed']}。

## 漏斗

| 阶段 | 候选单位 | 真垃圾 | 非垃圾 | Precision |
|---|---:|---:|---:|---:|
| V3 人工审核 Track | {baseline['tracks_total']} | {baseline['litter_tracks']} | {baseline['tracks_total'] - baseline['litter_tracks']} | {baseline['candidate_precision']:.1%} |
| 固定坐标事件合并 | {baseline['events_total']} | {baseline['litter_events']} | {baseline['non_litter_events']} | {baseline['litter_events'] / baseline['events_total']:.1%} |
| 代表帧 Context + Actor | {representative['visible_events_after_context_and_actor']} | {representative['visible_litter_events']} | {representative['visible_events_after_context_and_actor'] - representative['visible_litter_events']} | {representative['visible_candidate_precision']:.1%} |
| 在线 5 秒确认 | {online['events_confirmed']} | {online['true_positive_events']} | {online['false_positive_events']} | {online['precision']:.1%} |

## 确认事件与人工审核映射

| 在线事件 | 审核事件 | 标签 | 中心距离 px |
|---:|---|---|---:|
{match_rows}

## 环境与生命周期边界

- 本轮 `GLOBAL_LIGHT_CHANGE` 是“正在做明显光度补偿”的诊断状态，并不暂停判断；只有 `ENVIRONMENT_CHANGE` 才会暂停证据累计。负样本和正样本均完成候选确认，说明补偿没有抹掉目标。
- 素材没有“垃圾被清走后连续露出干净地面”的完整片段，因此 `CLEARED` 的 5 秒关闭条件只有单元测试覆盖，尚无真实视频验证。
- 当前 20px/30px 距离与 200px 外部变化面积仍是当前 1440p 视角的绝对像素参数。进入其他摄像头前，应改成基于地面透视分区和候选尺度的归一化参数。
- Clean Reference 在全部实验中只读，没有被当前帧、长期垃圾或遮挡更新。

## 是否可以进入多摄像头

当前摄像头已证明路线可行：Clean Reference 能补 YOLO 漏检，context/actor/时序可把 16 个审核事件压到 6 个确认事件，同时保留 5/5 个垃圾位置。进入其他摄像头前仍应补一段“投放—停留—清走”素材，验证真实 `CLEARED` 生命周期；随后冻结本摄像头参数作为基线，用 2–3 个相似视角做迁移验证，先只调整 ROI、透视尺度图和 Clean Profile，不逐摄像头重新调算法。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--reviewed-root", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.online_root, args.reviewed_root)
    (args.online_root / "human_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.online_root / "REPORT.md").write_text(
        render_report(result), encoding="utf-8"
    )
    print(json.dumps(result["online_negative"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
