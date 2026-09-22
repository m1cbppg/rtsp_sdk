"""Generate the V3.3 false-positive and performance reports from artifacts.

Reads the real-weight smoke JSON and every replay summary produced by
``scripts/replay_ground_litter_v33.py``, so the numbers in the reports are
derived from files rather than transcribed by hand.

Usage:
  .venv/bin/python scripts/report_ground_litter_v33.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = "output/ground_litter_v33_replay_20260918"
DEFAULT_SMOKE = "output/ground_litter_v33_smoke_20260918/smoke.json"


def load_replays(directory: Path) -> list[dict[str, Any]]:
    runs = []
    for summary_path in sorted(directory.glob("*.summary.json")):
        label = summary_path.name[: -len(".summary.json")]
        if label.startswith("diag_"):
            continue
        raw_path = directory / f"{label}.json"
        if not raw_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stage_path = directory / f"{label}.stage.json"
        if "stage_ms" not in summary and stage_path.is_file():
            # Older runs wrote the stage table to its own file.
            summary["stage_ms"] = json.loads(
                stage_path.read_text(encoding="utf-8")
            )
        runs.append({
            "label": label,
            "summary": summary,
            "raw": json.loads(raw_path.read_text(encoding="utf-8")),
        })
    return runs


def fp_report(runs: list[dict[str, Any]]) -> str:
    lines = ["# V3.3 双通道 — 分来源误报报告（离线清洁回放）", ""]
    lines.append(
        "口径：直接驱动侧进程真实 tick 函数 `_analyse_hybrid`，"
        "在录制视频上以 `analysis_fps` 采样；每源分别统计**已显示事件**（按 event_id）。"
    )
    lines.append("")
    lines.append(
        "| 回放 | 置信度 | zone最小短边 | 最小面积 | 有效时长 | semantic-only | prior-only | fused | 合计 | 单帧闪框 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        totals = run["summary"].get("totals")
        if not totals:
            continue
        lines.append(
            f"| `{run['label']}` | {run['raw'].get('confidence')} | "
            f"{run['raw'].get('zone_minimum_short_side_px')} | "
            f"{run['raw'].get('zone_minimum_box_area_px')} | "
            f"{totals['span_seconds']}s | "
            f"{totals['semantic_only_displayed_events']} "
            f"({totals['semantic_only_events_per_hour']}/h) | "
            f"{totals['prior_only_displayed_events']} | "
            f"{totals['fused_displayed_events']} | "
            f"{totals['merged_displayed_events']} "
            f"({totals['merged_events_per_hour']}/h) | "
            f"{totals['single_frame_flashes']} |"
        )
    lines.append("")
    lines.append("## 过滤与拒绝统计")
    lines.append("")
    lines.append(
        "| 回放 | 先验提议 | 先验保留 | 几何拒绝 | semantic原始 | semantic保留 | 裁剪原始 | 裁剪未关联拒绝 | 环境非NORMAL tick | 降级 tick |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        totals = run["summary"].get("totals")
        if not totals:
            continue
        lines.append(
            f"| `{run['label']}` | {totals['prior_candidates_proposed']} | "
            f"{totals['prior_candidates_retained']} | "
            f"{totals['prior_rejected_by_geometry']} | "
            f"{totals['semantic_raw_candidates']} | "
            f"{totals['semantic_retained_candidates']} | "
            f"{totals['semantic_crop_raw']} | "
            f"{totals['semantic_crop_unmatched_rejected']} | "
            f"{totals['environment_not_normal_ticks']} | "
            f"{totals['degraded_ticks']} |"
        )
    lines.append("")
    lines.append("## 判定与边界")
    lines.append("")

    baseline = next(
        (r for r in runs if r["label"] == "clean_negative"), None
    )
    strict = next(
        (r for r in runs if r["label"] == "clean_negative_strict"), None
    )
    if baseline and baseline["summary"].get("totals"):
        totals = baseline["summary"]["totals"]
        lines.append(
            f"- **semantic-only 误报密度**：在用户标注为“正常负样本”的录像上，"
            f"示例请求配置（confidence={baseline['raw'].get('confidence')}）"
            f"得到 {totals['semantic_only_displayed_events']} 个已显示事件 / "
            f"{totals['span_seconds']}s，即 **{totals['semantic_only_events_per_hour']} 事件/小时**。"
            f"这远高于规格 §15.2 的候选门槛（每摄像头每 4 小时 ≤1 个）。"
        )
        lines.append(
            f"- **单帧闪框 = {totals['single_frame_flashes']}**：确认规则要求多次命中，"
            f"没有任何事件只显示一帧，符合 §18 第 4 条。"
        )
        lines.append(
            f"- **prior / fused 两源为 0**，但原因是环境状态非 NORMAL 的 tick 占 "
            f"{totals['environment_not_normal_ticks']}/{totals['ticks']}："
            f"现有 profile 采自 2026-09-18 14:40，而本录像为 09-17 素材，"
            f"先验通道按设计 abstain。**因此先验通道的误报率在本次可用素材下无法测量**，"
            f"这正是规格 §2.2 明确排除的“全天多环境 Profile”问题，不得据此声称 prior 无误报。"
        )
    if strict and strict["summary"].get("totals") and baseline and baseline["summary"].get("totals"):
        base_totals = baseline["summary"]["totals"]
        strict_totals = strict["summary"]["totals"]
        lines.append(
            f"- **收紧阈值的效果**：confidence {baseline['raw'].get('confidence')} → "
            f"{strict['raw'].get('confidence')}、zone 最小短边 "
            f"{baseline['raw'].get('zone_minimum_short_side_px')} → "
            f"{strict['raw'].get('zone_minimum_short_side_px')} 后，"
            f"semantic-only 误报由 {base_totals['semantic_only_displayed_events']} 个降至 "
            f"{strict_totals['semantic_only_displayed_events']} 个"
            f"（{base_totals['semantic_only_events_per_hour']}/h → "
            f"{strict_totals['semantic_only_events_per_hour']}/h）；"
            f"semantic 保留候选由 {base_totals['semantic_retained_candidates']} 降至 "
            f"{strict_totals['semantic_retained_candidates']}。"
            f"达到候选门槛所需的阈值标定属现场验收内容，本报告只给出量级。"
        )
    lines.append(
        "- **不得**用本报告的 prior-only = 0 推断先验通道无漏报：该通道在本次素材上"
        "几乎没有运行机会。"
    )
    lines.append("")
    lines.append("原始数据：同目录 `*.json`（逐 tick）与 `*.summary.json`（汇总）。")
    return "\n".join(lines) + "\n"


def perf_report(runs: list[dict[str, Any]], smoke: dict[str, Any] | None) -> str:
    lines = ["# V3.3 双通道 — 性能报告（本机 CPU，非目标 GPU）", ""]
    lines.append(
        "> **重要边界**：本机为 macOS/CPU，规格 §10.2 的预算（普通 tick <2000ms、"
        "全扫描 <1500ms、crop 批 <800ms、显存 ≤1.5GiB）是针对 RTX 3060 Ti 的。"
        "本报告的数字**不能用于判定是否达标**，只能用于观察阶段占比与相对量级；"
        "达标判定必须在服务器 GPU 上重测。"
    )
    lines.append("")
    if smoke:
        lines.append("## 真实权重 smoke（单帧，本机 CPU）")
        lines.append("")
        lines.append(f"- 权重：{smoke.get('weight_bytes', 0)} 字节，本地加载，无下载")
        lines.append(f"- 模型加载：{smoke.get('model_load_seconds')}s")
        lines.append(
            f"- 全 ROI 扫描：{smoke.get('full_scan_ms')}ms，"
            f"**{smoke.get('full_scan_model_calls')} 次批量调用**，"
            f"batch={smoke.get('full_scan_batch_size')}（tile 数 {smoke.get('tile_count')}）"
        )
        lines.append(
            f"- prior crop：{smoke.get('crop_batch_ms')}ms，"
            f"**{smoke.get('crop_model_calls')} 次批量调用**，"
            f"batch={smoke.get('crop_batch_size')}"
        )
        lines.append(f"- 判定：passed={smoke.get('passed')}")
        lines.append("")
    lines.append("## 回放分阶段耗时（P50 / P95 / max，ms）")
    lines.append("")
    lines.append("| 回放 | 阶段 | P50 | P95 | max |")
    lines.append("|---|---|---|---|---|")
    for run in runs:
        stage = run["summary"].get("stage_ms")
        if not stage:
            continue
        for name in ("prior_ms", "full_scan_ms", "crop_ms", "total_ms", "wall_ms"):
            values = stage.get(name) or {}
            if not values:
                continue
            lines.append(
                f"| `{run['label']}` | {name} | {values.get('p50')} | "
                f"{values.get('p95')} | {values.get('max')} |"
            )
    lines.append("")
    lines.append("## 调度与批量化")
    lines.append("")
    for run in runs:
        totals = run["summary"].get("totals")
        if not totals:
            continue
        lines.append(
            f"- `{run['label']}`：模型调用 full={totals['model_runs_full']}、"
            f"crop={totals['model_runs_crop']}；"
            f"crop 未关联拒绝={totals['semantic_crop_unmatched_rejected']}；"
            f"降级 tick={totals['degraded_ticks']}/{totals['ticks']}"
        )
    lines.append("")
    lines.append("## 主链影响")
    lines.append("")
    lines.append(
        "离线回放不构建 DeepStream 主管线，因此**本报告不含主链 FPS、duplicate FPS、"
        "输入帧年龄与显存**。这些必须在服务器候选镜像上实测后补充；"
        "在主链 FPS ≥20、duplicate=0 得到实测证据之前，不得声称性能达标。"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", default=DEFAULT_REPLAY)
    parser.add_argument("--smoke", default=DEFAULT_SMOKE)
    args = parser.parse_args()

    replay_dir = REPO / args.replay
    runs = load_replays(replay_dir)
    smoke_path = REPO / args.smoke
    smoke = (
        json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke_path.is_file() else None
    )
    (replay_dir / "REPORT.md").write_text(fp_report(runs), encoding="utf-8")

    perf_dir = REPO / "output/ground_litter_v33_perf_20260918"
    perf_dir.mkdir(parents=True, exist_ok=True)
    (perf_dir / "REPORT.md").write_text(perf_report(runs, smoke), encoding="utf-8")

    print(f"wrote {replay_dir}/REPORT.md")
    print(f"wrote {perf_dir}/REPORT.md")
    print(f"replays included: {[run['label'] for run in runs]}")


if __name__ == "__main__":
    main()
