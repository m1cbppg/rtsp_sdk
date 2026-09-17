"""Summarize a ground-litter pilot run without claiming model accuracy.

The optional labels file is a small human review export:
{"item_id":"...", "label":"true_litter|false_positive|uncertain"}
or a list of such objects. Labels are never inferred from detector output.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def log_rows(run_dir: Path):
    """Read numeric rotations oldest first; fail clearly on incomplete evidence."""
    rotations = [p for p in run_dir.glob("observations.jsonl.*")
                 if p.is_file() and p.suffix[1:].isdigit()]
    paths = sorted(rotations, key=lambda p: int(p.suffix[1:]), reverse=True)
    for path in [*paths, run_dir / "observations.jsonl"]:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("not an object")
                except ValueError:
                    raise ValueError(f"Invalid JSON event at {path.name}:{number}; "
                                     "retry on a complete log snapshot") from None
                yield row


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return round(values[index], 3)


def load_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("labels", [])
    if not isinstance(payload, list):
        raise ValueError("labels must be a list or {labels: [...]}")
    labels: dict[str, str] = {}
    allowed = {"true_litter", "false_positive", "uncertain"}
    for row in payload:
        if not isinstance(row, dict) or not row.get("item_id") or row.get("label") not in allowed:
            raise ValueError("each label requires item_id and a supported label")
        labels[str(row["item_id"])] = str(row["label"])
    return labels


def analyze(run_dir: Path, labels_path: Path | None = None) -> dict:
    observations = run_dir / "observations.jsonl"
    if not observations.is_file():
        raise ValueError(f"missing {observations}")
    labels = load_labels(labels_path)
    rows = defaultdict(lambda: {
        "observations": 0, "candidates": 0, "candidate_labels": Counter(),
        "created_ids": [], "cleared_ids": [], "inference_seconds": [],
        "result_age_seconds": [], "capture_generation": Counter(),
        "rejections": Counter(), "source_states": Counter(),
        "candidate_frames": 0, "confirmation_blocks": Counter(), "filter_reasons": Counter(),
        "alignment_displacement": [], "max_candidate_span_seconds": 0,
        "attempt_inference_seconds": [], "attempt_age_seconds": [],
        "stage_seconds": defaultdict(list),
    })
    event_counts = Counter()
    rejection_counts = Counter()
    source_state_counts = Counter()
    storage_samples = []
    for row in log_rows(run_dir):
        event = str(row.get("event", "unknown"))
        event_counts[event] += 1
        if event in {'observation','rejected'} and row.get('camera_id'):
            attempt = rows[str(row['camera_id'])]
            for field, target in [('inference_seconds','attempt_inference_seconds'),
                                  ('result_age_seconds','attempt_age_seconds')]:
                if row.get(field) is not None:
                    attempt[target].append(float(row[field]))
            for stage,value in (row.get('diagnostics') or {}).get('stage_seconds',{}).items():
                attempt['stage_seconds'][stage].append(float(value))
        if event == "rejected":
            reason = str(row.get("reason", "unknown"))
            rejection_counts[reason] += 1
            if row.get("camera_id"):
                rows[str(row["camera_id"])]["rejections"][reason] += 1
            continue
        if event == "source_state":
            state = str(row.get("state", "unknown"))
            source_state_counts[state] += 1
            if row.get("camera_id"):
                rows[str(row["camera_id"])]["source_states"][state] += 1
            continue
        if event == "storage":
            if row.get("bytes") is not None:
                storage_samples.append(int(row["bytes"]))
            continue
        # Only observation events represent detector candidates. Lifecycle,
        # sample and stop records intentionally have no camera payload.
        if event != "observation" or not row.get("camera_id"):
            continue
        camera = str(row["camera_id"])
        stat = rows[camera]
        stat["observations"] += 1
        candidates = row.get("candidates") or []
        stat["candidates"] += len(candidates)
        stat["candidate_frames"] += bool(candidates)
        diagnostics = row.get("diagnostics") or {}
        for track in diagnostics.get("confirmation", []):
            if not track.get("confirmed"):
                stat["confirmation_blocks"].update(track.get("reasons", []))
            stat["max_candidate_span_seconds"] = max(
                stat["max_candidate_span_seconds"], track.get("span_seconds", 0))
        stat["filter_reasons"].update(
            item.get("reason", "unknown") for item in diagnostics.get("rejected_candidates", []))
        displacement = diagnostics.get("alignment", {}).get("displacement_native_px")
        if displacement is not None:
            stat["alignment_displacement"].append(displacement)
        stat["candidate_labels"].update(str(c.get("label", "unknown")) for c in candidates)
        stat["created_ids"].extend(row.get("created_item_ids") or [])
        stat["cleared_ids"].extend(row.get("cleared_item_ids") or [])
        if row.get("inference_seconds") is not None:
            stat["inference_seconds"].append(float(row["inference_seconds"]))
        if row.get("result_age_seconds") is not None:
            stat["result_age_seconds"].append(float(row["result_age_seconds"]))
        stat["capture_generation"][str(row.get("generation", "unknown"))] += 1
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").is_file() else {}
    camera_report = {}
    all_created: list[str] = []
    for camera, stat in sorted(rows.items()):
        created = list(stat["created_ids"])
        all_created.extend(created)
        camera_report[camera] = {
            "observations": stat["observations"],
            "candidates": stat["candidates"],
            "candidate_frames": stat["candidate_frames"],
            "confirmation_blocked_track_observations": dict(stat["confirmation_blocks"]),
            "model_candidate_filter_reasons": dict(stat["filter_reasons"]),
            "max_candidate_span_seconds": stat["max_candidate_span_seconds"],
            "alignment_displacement_px_p95": percentile(stat["alignment_displacement"], .95),
            "candidate_labels": dict(stat["candidate_labels"]),
            "rejection_counts": dict(stat["rejections"]),
            "source_state_counts": dict(stat["source_states"]),
            "created_item_ids": created,
            "cleared_item_ids": list(stat["cleared_ids"]),
            "duplicate_created_ids": sorted(k for k, v in Counter(created).items() if v > 1),
            "capture_generations": dict(stat["capture_generation"]),
            "inference_seconds_p50": percentile(stat["inference_seconds"], .50),
            "inference_seconds_p95": percentile(stat["inference_seconds"], .95),
            "result_age_seconds_p95": percentile(stat["result_age_seconds"], .95),
            "all_attempts_with_inference_timing": len(stat['attempt_inference_seconds']),
            "all_attempt_inference_seconds_p95": percentile(stat['attempt_inference_seconds'], .95),
            "all_attempts_with_result_age": len(stat['attempt_age_seconds']),
            "all_attempt_result_age_seconds_p95": percentile(stat['attempt_age_seconds'], .95),
            "timing_scope": "local post-decoding time; legacy rejected events may lack timings",
            "stage_seconds": {stage:{'count':len(values),'p50':percentile(values,.5),
                                     'p95':percentile(values,.95)}
                              for stage,values in stat['stage_seconds'].items()},
            "review": {item: labels[item] for item in created if item in labels},
        }
    review_counts = Counter(labels[item] for item in all_created if item in labels)
    return {
        "status": "pilot_log_summary_not_accuracy",
        "run_directory": str(run_dir),
        "summary": summary,
        "event_counts": dict(event_counts),
        "rejection_counts": dict(rejection_counts),
        "source_state_counts": dict(source_state_counts),
        "storage_bytes_max": max(storage_samples) if storage_samples else None,
        "cameras": camera_report,
        "created_id_count": len(all_created),
        "unique_created_id_count": len(set(all_created)),
        "duplicate_created_ids": sorted(k for k, v in Counter(all_created).items() if v > 1),
        "duplicate_id_scope": "repeated ID writes only; distinct IDs for the same physical object require image review",
        "review_counts": dict(review_counts),
        "unreviewed_created_ids": [item for item in all_created if item not in labels],
        "optimization_queue": [
            "review every created item before enabling notifications",
            "inspect rejected/stale/view-change counters from summary.json",
            "compare candidate labels and ROI boundaries before changing confidence",
            "run a fixed day/night replay after each profile or threshold change",
        ],
    }


def markdown(report: dict) -> str:
    lines = ["# Ground-litter pilot log", "", "状态：仅日志分析，不是准确率报告。", ""]
    lines.append(f"创建记录：{report['created_id_count']}，唯一 ID：{report['unique_created_id_count']}，重复创建：{len(report['duplicate_created_ids'])}")
    lines.append("重复 ID 指同一 ID 被重复写入；同一实物分配不同 ID 仍需对照截图核查。")
    lines.append("事件统计：" + json.dumps(report["event_counts"], ensure_ascii=False))
    lines.append("拒绝原因：" + json.dumps(report["rejection_counts"], ensure_ascii=False))
    lines.append("源状态变化：" + json.dumps(report["source_state_counts"], ensure_ascii=False))
    if report["review_counts"]:
        lines.append("人工复核：" + "，".join(f"{k}={v}" for k, v in report["review_counts"].items()))
    lines += ["", "| 摄像头 | 观测 | 候选 | 创建ID | 推理P50/P95(s) | 结果年龄P95(s) |", "|---|---:|---:|---:|---:|---:|"]
    for camera, row in report["cameras"].items():
        lines.append(f"| {camera} | {row['observations']} | {row['candidates']} | {len(row['created_item_ids'])} | {row['inference_seconds_p50']}/{row['inference_seconds_p95']} | {row['result_age_seconds_p95']} |")
        lines.append("")
        lines.append("未确认原因（按候选轨迹观测计次，同一物体可重复出现）：" + json.dumps(row["confirmation_blocked_track_observations"], ensure_ascii=False))
        lines.append("模型候选过滤原因：" + json.dumps(row["model_candidate_filter_reasons"], ensure_ascii=False))
    lines += ["", "优化顺序："]
    lines += [f"- {item}" for item in report["optimization_queue"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(args.output), "markdown": str(args.output.with_suffix('.md')),
                      "created_id_count": report["created_id_count"],
                      "duplicate_created_ids": report["duplicate_created_ids"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
