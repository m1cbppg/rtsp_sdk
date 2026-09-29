#!/usr/bin/env python3
"""Emit the Rapid Dataset v2 Phase-1 report (spec §33).

Only the required sections are reported.  Model accuracy is deliberately absent:
no human truth exists yet, so any accuracy number would be fabricated.

    python scripts/report_ground_litter_historical_phase1.py \
        --artifact /home/sf01/ground-litter-historical-v2/artifact \
        --review-url http://127.0.0.1:18812/ --out artifact/PHASE1_REPORT.md
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_historical import read_jsonl  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--review-url", default="(not started)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tunnel-command", default="")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = args.artifact
    manifest = read_jsonl(artifact / "historical_window_manifest.jsonl")
    observations = read_jsonl(artifact / "candidate_observations.jsonl")
    units = read_jsonl(artifact / "review_units.jsonl")
    progress = _json(artifact / "coarse_progress.json", {})
    build = _json(artifact / "review_build_summary.json", {})
    queue = _json(artifact / "queue.json", {})

    downloaded = [row for row in manifest if row.get("download_status") == "downloaded"]
    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in downloaded:
        by_split[row["day_split"]].append(row)
    total_bytes = sum(int(row.get("source_file_size") or 0) for row in downloaded)
    seconds = sum(float(row.get("nominal_duration_seconds") or 0) for row in downloaded)

    source_counts = defaultdict(int)
    for row in observations:
        sources = tuple(sorted(row.get("confidence_by_source", {})))
        source_counts["both" if len(sources) > 1 else (sources[0] if sources else "none")] += 1

    from rtsp_annotator.ground_litter_historical import candidate_tier, Observation  # noqa: E402
    student_fp = 0
    for row in observations:
        tier = candidate_tier(Observation.from_dict(row))
        if tier == "P2":
            student_fp += 1

    first_batch = [unit for unit in units if unit.get("batch") == 1 and unit["kind"] == "candidate"]
    blind = [unit for unit in units if unit["kind"] == "blind"]
    per_camera_first = defaultdict(int)
    for unit in first_batch:
        per_camera_first[unit["camera_id"]] += 1

    lines: list[str] = []
    lines.append("# Ground Litter Historical Active Mining + Rapid Dataset v2 — 第一阶段报告")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append("范围：仅 01021/01022/01027/01030 四路；01028 不参与。")
    lines.append("")
    lines.append("## Historical Windows")
    lines.append("")
    lines.append(f"- train downloaded: **{len(by_split.get('TRAIN', []))}**"
                 f"（第一批目标 20）")
    lines.append(f"- dev cached: **{len(by_split.get('DEV', []))}**")
    lines.append(f"- final cached: **{len(by_split.get('FINAL', []))}**")
    lines.append(f"- windows done: **{progress.get('windows_done', len(downloaded))}**")
    lines.append(f"- total bytes: **{total_bytes / 1e9:.2f} GB**")
    lines.append(f"- 冻结 split：{json.dumps(_json(artifact / 'historical_window_manifest.meta.json', {}).get('day_split', {}), ensure_ascii=False)}")
    lines.append(f"- manifest SHA256: `{_json(artifact / 'historical_window_manifest.meta.json', {}).get('manifest_sha256', 'n/a')}`")
    lines.append("")
    lines.append("## Candidates")
    lines.append("")
    lines.append(f"- Turhancan-only: **{source_counts.get('turhancan', 0)}**")
    lines.append(f"- YOLO-only: **{source_counts.get('yolo', 0)}**")
    lines.append(f"- both: **{source_counts.get('both', 0)}**")
    lines.append(f"- 观测总数: **{len(observations)}**")
    lines.append(f"- student FP candidates (P2, Step2B 工作阈值附近): **{student_fp}**")
    lines.append(f"- 重复背景观测（同一固定目标重复出现）: **{build.get('duplicate_background', {}).get('duplicates', 0)}**")
    lines.append("")
    lines.append("## Groups")
    lines.append("")
    lines.append(f"- episodes: **{build.get('episodes', 0)}**")
    lines.append(f"- review groups: **{build.get('review_groups', 0)}**")
    lines.append(f"- selected first batch: **{len(first_batch)}**"
                 f"（queue 总长 {build.get('candidate_units', 0)}，分批 {queue.get('batch_plan', {})}）")
    lines.append(f"- blind ROI units: **{len(blind)}**（每 camera {len(blind) // 4 if blind else 0}）")
    lines.append(f"- suspected same-object groups: **{build.get('suspected_same_object_units', 0)}**")
    lines.append(f"- 多帧 group（有状态变化才 >1）: **{build.get('multi_frame_units', 0)}**")
    lines.append("")
    lines.append("| camera | first batch |")
    lines.append("| --- | ---: |")
    for camera in ("01021", "01022", "01027", "01030"):
        lines.append(f"| {camera} | {per_camera_first.get(camera, 0)} |")
    lines.append("")
    lines.append("## Review UI")
    lines.append("")
    lines.append(f"- URL: `{args.review_url}`")
    if args.tunnel_command:
        lines.append(f"- tunnel: `{args.tunnel_command}`")
    lines.append("")
    lines.append("| 操作 | 说明 |")
    lines.append("| --- | --- |")
    lines.append("| `R / I / U / N` | 必需垃圾 / 微小忽略 / 不确定 / 非垃圾背景 |")
    lines.append("| `1 / 2 / 3 / 0` | bbox 阶段：候选 A/B/C，`0` = UNLOCALIZED_REQUIRED |")
    lines.append("| `S / W / D` | suspected same-object：SAME / NEW / UNCERTAIN |")
    lines.append("| `← / →` | 上/下一个 review group |")
    lines.append("| blind unit | 判定前隐藏全部模型输出，判定后才揭示候选 |")
    lines.append("")
    lines.append("审核单位是 **review group**（一个独立垃圾实例 / 一个独立 hard-negative 簇），不是 frame。")
    lines.append("同一固定垃圾连续出现的近重复帧默认只保留 1 个 NORMAL 代表帧，最多 3 帧。")
    lines.append("")
    lines.append("## Safety")
    lines.append("")
    lines.append("- Sealed = **0**")
    lines.append("- official Step2C write = **0**")
    lines.append("- 8801 automated access = **0**")
    lines.append("- 旧 Rapid v1 artifact 未 reset、未修改；本阶段 split 与 eval 独立定义")
    lines.append("- 未训练、未导出 dataset、未代替用户审核")
    lines.append("")
    lines.append("## Efficiency")
    lines.append("")
    hours = seconds / 3600.0
    lines.append(f"- downloaded video hours: **{hours:.2f} h**")
    if hours > 0:
        lines.append(f"- candidate groups / hour video: **{build.get('review_groups', 0) / hours:.1f}**")
        lines.append(f"- blind units / hour video: **{len(blind) / hours:.1f}**")
    lines.append(f"- 抽取帧: **{len(read_jsonl(artifact / 'coarse_frames.jsonl'))}**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("不报告模型 accuracy：目前没有人工真值，任何准确率都会是伪造的。")
    lines.append("")
    lines.append("STOP：请只审核第一批 30~50 个 review group，不要继续旧的 265-frame Rapid Review。")
    lines.append("审核完成后回报，再据实算 Required yield / HN yield / 重复噪声比，"
                 "然后决定下一批 50 的排序。")
    lines.append("")

    text = "\n".join(lines)
    out = args.out or artifact / "PHASE1_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwritten: {out}", file=sys.stderr)
    return 0


def _json(path: Path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return default


if __name__ == "__main__":
    raise SystemExit(main())
