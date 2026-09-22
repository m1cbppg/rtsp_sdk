#!/usr/bin/env python3
"""Apply the existing actor/context model to visible V3.1 anomaly events.

The structural context gate runs first.  Actor inference is intentionally
limited to one representative frame for each still-visible event, matching the
future cascade where expensive semantic work follows cheap geometric gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rtsp_annotator.ground_litter_detection import (  # noqa: E402
    UltralyticsGroundLitterDetector,
)
from rtsp_annotator.ground_litter_geometry import box_overlap_fraction  # noqa: E402
from run_ground_litter_clean_temporal_poc import (  # noqa: E402
    read_frame,
    yolo_options,
)


def max_actor_overlap(
    candidate: Iterable[float], actor_boxes: Iterable[Iterable[float]]
) -> float:
    return max(
        (box_overlap_fraction(candidate, actor) for actor in actor_boxes),
        default=0.0,
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(events: list[dict[str, Any]]) -> dict[str, Any]:
    litter = [row for row in events if row["human_label"] == "LITTER_CONFIRMED"]
    non_litter = [
        row for row in events if row["human_label"] == "CLEAR_NON_LITTER"
    ]
    visible = [row for row in events if row["final_predicted_state"] != "OCCLUDED"]
    visible_litter = [row for row in visible if row["human_label"] == "LITTER_CONFIRMED"]
    deferred_non_litter = [
        row for row in non_litter if row["final_predicted_state"] == "OCCLUDED"
    ]
    return {
        "events_total": len(events),
        "litter_events": len(litter),
        "non_litter_events": len(non_litter),
        "visible_events_after_context_and_actor": len(visible),
        "visible_litter_events": len(visible_litter),
        "deferred_non_litter_events": len(deferred_non_litter),
        "litter_event_recall": round(len(visible_litter) / max(len(litter), 1), 4),
        "visible_candidate_precision": round(
            len(visible_litter) / max(len(visible), 1), 4
        ),
        "non_litter_event_deferral_rate": round(
            len(deferred_non_litter) / max(len(non_litter), 1), 4
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--v3-output", type=Path, required=True)
    parser.add_argument("--v31-output", type=Path, required=True)
    parser.add_argument("--litter-model", type=Path, required=True)
    parser.add_argument("--actor-model", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    events = load_json(args.v31_output / "events.json")
    manifest = [
        json.loads(line)
        for line in (
            args.v3_output / "vlm_review/negative_v3_manifest.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_by_review = {row["review_id"]: row for row in manifest}
    options = yolo_options(args.litter_model, args.actor_model)
    detector = UltralyticsGroundLitterDetector(
        model_path=args.litter_model,
        actor_model_path=args.actor_model,
        device=args.device,
    )
    video = args.input_dir / "正常负样本.mp4"
    actor_features = []
    updated_events = []
    for event in events:
        updated = dict(event)
        updated["context_predicted_state"] = event["predicted_state"]
        updated["actor_overlap"] = None
        updated["final_predicted_state"] = event["predicted_state"]
        if event["predicted_state"] == "OCCLUDED":
            updated_events.append(updated)
            continue
        item = manifest_by_review[event["representative_review_id"]]
        frame = read_frame(video, float(item["timestamp"]))
        actors = detector.actor_boxes(frame, options)
        overlap = max_actor_overlap(item["anomaly_bbox"], actors)
        matching = [
            [round(float(value), 3) for value in actor]
            for actor in actors
            if box_overlap_fraction(item["anomaly_bbox"], actor) > 0
        ]
        actor_features.append({
            "event_id": event["event_id"],
            "track_id": int(item["track"]["track_id"]),
            "timestamp": float(item["timestamp"]),
            "candidate_box": item["anomaly_bbox"],
            "actor_count": len(actors),
            "max_actor_overlap": round(overlap, 6),
            "matching_actor_boxes": matching,
        })
        updated["actor_overlap"] = round(overlap, 6)
        if overlap > float(options.actor_overlap_threshold):
            updated["final_predicted_state"] = "OCCLUDED"
        updated_events.append(updated)
        print(
            f"[{event['event_id']}] actors={len(actors)} overlap={overlap:.3f} "
            f"state={updated['final_predicted_state']}",
            flush=True,
        )

    result = {
        "kind": "ground_litter_event_actor_v31_offline_probe",
        "parameters": {
            "actor_model": args.actor_model.name,
            "actor_imgsz": int(options.actor_imgsz),
            "actor_confidence": float(options.actor_confidence),
            "actor_overlap_threshold": float(options.actor_overlap_threshold),
            "cascade_order": ["context_unavailable", "actor_overlap"],
            "human_labels_used_for_prediction": False,
        },
        "event_level": evaluate(updated_events),
    }
    (args.v31_output / "actor_features.json").write_text(
        json.dumps(actor_features, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.v31_output / "events_with_actor.json").write_text(
        json.dumps(updated_events, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.v31_output / "evaluation_with_actor.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    remaining = [
        row for row in updated_events
        if row["final_predicted_state"] != "OCCLUDED"
        and row["human_label"] == "CLEAR_NON_LITTER"
    ]
    metrics = result["event_level"]
    context_metrics = load_json(args.v31_output / "evaluation.json")["event_level"]
    v3_report = load_json(args.v3_output / "report.json")
    t01_summary_path = args.v31_output / "t01_context_guard_summary.json"
    t01_summary = load_json(t01_summary_path) if t01_summary_path.exists() else None
    report = f"""# Ground Litter V3.1：Actor Cascade 补充验证

- 结构上下文门后只对仍可见事件运行现有 actor/context 模型。
- 最终保留 {metrics['visible_events_after_context_and_actor']} 个待 YOLO/VLM 判断事件，其中垃圾 {metrics['visible_litter_events']} 个。
- 当前样本事件级候选 precision 为 {metrics['visible_candidate_precision']:.1%}，垃圾事件保留率为 {metrics['litter_event_recall']:.1%}。
- 非垃圾事件进入 `OCCLUDED` 的比例为 {metrics['non_litter_event_deferral_rate']:.1%}。
- 剩余非垃圾事件：{', '.join(row['event_id'] + '=' + '/'.join(map(str, row['track_ids'])) for row in remaining) or '无'}。

`OCCLUDED` 只暂停地面判断。Actor 结果不会写入 Clean Reference，也不会永久学习为非垃圾。
"""
    (args.v31_output / "ACTOR_REPORT.md").write_text(report, encoding="utf-8")
    final_report = f"""# Ground Litter V3.1：单摄像头收敛实验

## 最终结果

- V3 的 38 条已审核轨迹按固定地面坐标合并为 {metrics['events_total']} 个事件：{metrics['litter_events']} 个垃圾位置、{metrics['non_litter_events']} 个非垃圾事件。
- 结构上下文门将明显的大范围变化转为 `OCCLUDED`；现有 actor/context 模型继续处理人员和车辆。
- Cascade 后仅 {metrics['visible_events_after_context_and_actor']} 个事件进入后续 YOLO/VLM，其中 {metrics['visible_litter_events']} 个为垃圾，候选 precision 为 {metrics['visible_candidate_precision']:.1%}。
- 垃圾事件保留率为 {metrics['litter_event_recall']:.1%}；非垃圾事件暂缓率为 {metrics['non_litter_event_deferral_rate']:.1%}。
- T01 上下文保护：{t01_summary['retained'] if t01_summary else '未运行'}/{t01_summary['samples'] if t01_summary else '未运行'} 个 72–80 秒目标采样保留。
- V3 clean holdout 的 ≥10 秒轨迹仍为 {v3_report['clean_holdout']['v3']['tracks_at_least_seconds']['10']}；V3.1 不生成新候选，因此不会降低该门槛。
- 剩余需要语义判断的非垃圾事件：{', '.join(row['event_id'] + '=' + '/'.join(map(str, row['track_ids'])) for row in remaining) or '无'}。

## 状态语义

- `OCCLUDED` 表示当前位置的地面暂时不可可靠判断；不会输出垃圾，也不会更新 Clean Reference。
- `ANOMALY_PENDING` 表示局部变化可见，等待累计可见时间、YOLO 或 VLM 证据。
- 同一地面坐标的断裂 Track 归入同一事件，Persistence 使用累计可见时间，不把隐藏间隔计算为有效证据。

## 当前阶段边界

- 上下文特征和 actor 语义使用每个事件的代表帧验证，尚未完成逐秒在线状态回放。
- 当前事件合并没有“确认地面恢复干净后关闭事件”的生命周期；相同位置在很久以后出现新物体仍可能被合并。
- 200px 外部变化阈值仅适用于当前素材尺度。跨摄像头前必须归一化为局部透视尺度或候选面积比例。
- 人工审核标签用于最终评估，不参与事件合并、上下文门或 actor 预测。

## 进入多摄像头前的剩余工作

1. 在三段视频上做逐秒在线 replay：实现 `VISIBLE → OCCLUDED → REAPPEARED → CLEARED` 生命周期。
2. 增加明确的 clean-observation 条件，只有连续可见且与 Clean Reference 一致时才能关闭事件。
3. 将 200px 外部变化面积改成透视归一化指标，并在当前近/中/远区域分别验证。
4. 把最终 6 个事件组织成 VLM 离线输入；VLM 只处理 cascade 后的事件，不重新扫描整图。
5. 上述 replay 通过后冻结参数，再引入其他摄像头做迁移验证。

## 对比

| 阶段 | 待处理单位 | 垃圾 | 非垃圾 | 候选 precision |
|---|---:|---:|---:|---:|
| V3 人工审核 Track | 38 | 16 | 22 | 42.1% |
| V3.1 事件合并 | {metrics['events_total']} | {metrics['litter_events']} | {metrics['non_litter_events']} | {metrics['litter_events'] / metrics['events_total']:.1%} |
| 结构上下文门后 | {context_metrics['visible_events_after_context_gate']} | {context_metrics['visible_litter_events']} | {context_metrics['visible_events_after_context_gate'] - context_metrics['visible_litter_events']} | {context_metrics['visible_candidate_precision']:.1%} |
| Context + Actor 后 | {metrics['visible_events_after_context_and_actor']} | {metrics['visible_litter_events']} | {metrics['visible_events_after_context_and_actor'] - metrics['visible_litter_events']} | {metrics['visible_candidate_precision']:.1%} |
"""
    (args.v31_output / "FINAL_REPORT.md").write_text(final_report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
