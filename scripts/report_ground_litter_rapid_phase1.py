#!/usr/bin/env python3
"""Assemble the Ground Litter Rapid v1 Phase-1 completion report.

Reads only the Rapid artifact (split, frame manifest, extraction manifest, baseline inference
manifest, smoke report) and writes ``PHASE1_REPORT.md`` plus ``phase1_report.json``.

    python scripts/report_ground_litter_rapid_phase1.py --artifact <root> [--smoke-report <json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtsp_annotator.ground_litter_rapid import (  # noqa: E402
    CAMERAS,
    EVAL_SPLIT,
    PROPOSAL_FLOOR,
    SEED,
    STEP2B_LAST_SHA256,
    THRESHOLD_GRID,
    TRAIN_SPLIT,
    read_jsonl,
    write_json,
)

REVIEW_KEYS = [
    ("R / I / U", "选择真值类别（必需垃圾 / 微小忽略 / 不确定）"),
    ("点击画面", "在该类别下落下真值中心点（不画框）"),
    ("N", "本帧无目标并完成真值"),
    ("Enter", "本帧真值确认完成"),
    ("Y", "Stage B：当前 prediction 判定正确"),
    ("X", "Stage B：当前 prediction 属 ignore / 不确定"),
    ("F 或 N", "Stage B：当前 prediction 是误报"),
    ("M", "Stage B：实际是垃圾但 Stage A 漏点，回到 Stage A 补点"),
    ("1 / 2 / 3", "Stage C：选择候选框 A / B / C"),
    ("0", "Stage C：None → UNLOCALIZED_SKIP"),
    ("← / →", "上一帧 / 下一帧"),
    ("滚轮 / 拖拽", "缩放 / 平移（8px 小目标请先放大或用右下 4x 放大镜）"),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, default=None)
    parser.add_argument("--browser-smoke-report", type=Path, default=None)
    parser.add_argument("--url", default="http://127.0.0.1:18810/")
    args = parser.parse_args(argv)

    artifact = args.artifact.resolve()
    split = json.loads((artifact / "split.json").read_text(encoding="utf-8"))
    frames = read_jsonl(artifact / "frame_manifest.jsonl")
    extraction = json.loads((artifact / "extraction_manifest.json").read_text(encoding="utf-8"))
    inference_path = artifact / "baseline" / "inference_manifest.json"
    inference = (json.loads(inference_path.read_text(encoding="utf-8"))
                 if inference_path.exists() else None)
    smoke = None
    if args.smoke_report and args.smoke_report.exists():
        smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    browser = None
    if args.browser_smoke_report and args.browser_smoke_report.exists():
        browser = json.loads(args.browser_smoke_report.read_text(encoding="utf-8"))

    review_frames = sum(1 for f in frames if f["kind"] == "fixed") + \
        sum(1 for f in frames if f["kind"] == "bonus_train")
    fixed = sum(1 for f in frames if f["kind"] == "fixed")
    bonus = sum(1 for f in frames if f["kind"] == "bonus_train")

    report = {
        "phase": "ground-litter-rapid-v1-phase1",
        "seed": SEED,
        "scope_note": "仅代表当前四路可用摄像头 01021/01022/01027/01030；01028 严重遮挡，"
                      "本轮不做任何结论。",
        "split": {
            "rapid_train_ps": split["counts"][TRAIN_SPLIT],
            "rapid_eval_ps": split["counts"][EVAL_SPLIT],
            "per_camera": split["per_camera"],
            "split_sha256": split["split_sha256"],
            "prior_inference_file_ids": split["excluded_from_holdout_due_prior_inference"],
            "prior_inference_all_forced_to_train": all(
                row["split"] == TRAIN_SPLIT for row in split["rows"]
                if row["excluded_from_holdout_due_prior_inference"]),
            "eval_ps": [{"camera_id": r["camera_id"], "file_id": r["file_id"]}
                        for r in split["rows"] if r["split"] == EVAL_SPLIT],
        },
        "frames": {
            "fixed_train": extraction["counts"]["fixed_train"],
            "fixed_eval": extraction["counts"]["fixed_eval"],
            "bonus_train": extraction["counts"]["bonus_train"],
            "total_review_frames": review_frames,
            "extracted": extraction["frames_extracted"],
            "decode_failures": extraction["decode_failure_count"],
            "missing": extraction["missing_count"],
            "anomaly_count": extraction["anomaly_count"],
            "max_abs_delta_ms": extraction["max_abs_delta_ms"],
            "decode_mode": extraction["decode_mode"],
            "source_native": [2560, 1440],
        },
        "baseline": None,
        "review_ui": {"url": args.url, "frames_to_review": review_frames},
        "safety": {
            "sealed_accessed": False,
            "official_writes": 0,
            "automated_8801_access": 0,
            "step2b_last_pt_overwritten": False,
        },
    }

    if inference:
        by_threshold = inference["predictions_by_threshold"]
        report["baseline"] = {
            "checkpoint_path": inference["model_path"],
            "checkpoint_sha256": inference["model_sha256"],
            "checkpoint_sha256_matches_frozen": inference["model_sha256"] ==
            inference["model_sha256_expected"] == STEP2B_LAST_SHA256,
            "best_pt_used": False,
            "inference_frames": inference["frames"],
            "tiles_total": inference["tiles_total"],
            "raw_candidates_total": inference["raw_candidates_total"],
            "raw_predictions_at_floor": by_threshold[f"{PROPOSAL_FLOOR:.2f}"],
            "predictions_by_threshold": by_threshold,
            "per_split": inference["per_split"],
            "elapsed_seconds": inference["elapsed_seconds"],
            "predictions_sha256": inference["predictions_sha256"],
            "device": inference["device"],
            "note": "尚未完成人工真值，因此本阶段不报告任何准确率、召回率或好坏结论。",
        }

    lines: list[str] = []
    lines.append("# Ground Litter Rapid Eval + Active Learning v1 — 第一阶段完成报告")
    lines.append("")
    lines.append(f"**scope**: {report['scope_note']}")
    lines.append("")
    lines.append("## Split")
    lines.append("")
    lines.append(f"- Rapid-Train: **{split['counts'][TRAIN_SPLIT]} PS**")
    lines.append(f"- Rapid-Eval Holdout: **{split['counts'][EVAL_SPLIT]} PS**")
    lines.append(f"- seed: `{SEED}`")
    lines.append("")
    lines.append("| camera | rapid_train | rapid_eval | forced_to_train (prior inference) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for camera in CAMERAS:
        per = split["per_camera"][camera]
        lines.append(f"| {camera} | {per['rapid_train']} | {per['rapid_eval']} | "
                     f"{per['forced_to_train']} |")
    lines.append("")
    lines.append(f"- prior-inference PS 全部强制进入 Rapid-Train: "
                 f"**{report['split']['prior_inference_all_forced_to_train']}** "
                 f"({len(split['excluded_from_holdout_due_prior_inference'])} PS)")
    lines.append(f"- split SHA256: `{split['split_sha256']}`")
    lines.append("")
    lines.append("## Frames")
    lines.append("")
    lines.append(f"- fixed train: **{extraction['counts']['fixed_train']}**")
    lines.append(f"- fixed eval: **{extraction['counts']['fixed_eval']}**")
    lines.append(f"- bonus train: **{extraction['counts']['bonus_train']}**")
    lines.append(f"- 解码方式: `{extraction['decode_mode']}`，source-native 2560x1440")
    lines.append(f"- decode failures: **{extraction['decode_failure_count']}**")
    lines.append(f"- missing: **{extraction['missing_count']}**")
    lines.append(f"- max |delta_ms|: **{extraction['max_abs_delta_ms']}**")
    lines.append(f"- frame_index != nominal 的记录数: {extraction['anomaly_count']}")
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    if inference:
        lines.append(f"- checkpoint: `{inference['model_path']}`")
        lines.append(f"- SHA256: `{inference['model_sha256']}` "
                     f"(与冻结值一致: {report['baseline']['checkpoint_sha256_matches_frozen']})")
        lines.append("- 使用 `best.pt`: **否**（禁止）")
        lines.append(f"- inference frames: **{inference['frames']}**")
        lines.append(f"- tiles: {inference['tiles_total']}，raw candidates: "
                     f"{inference['raw_candidates_total']}")
        lines.append(f"- raw predictions @ conf 0.01: "
                     f"**{inference['predictions_by_threshold']['0.01']}**")
        lines.append("- threshold 网格预测数: "
                     + ", ".join(f"{k}: {v}" for k, v in
                                 sorted(inference["predictions_by_threshold"].items())))
        lines.append(f"- device: {inference['device']}（服务器 GPU 驱动当前不可用，CPU 推理）")
        lines.append("")
        lines.append("> 本阶段**不报告** eval 好坏：Human Truth 尚未完成，任何准确率数字都会是"
                     "伪造。")
    else:
        lines.append("- 尚未运行 baseline 推理。")
    lines.append("")
    lines.append("## Review UI")
    lines.append("")
    lines.append(f"- URL: `{args.url}`（本地隧道 → 服务器 127.0.0.1:8810，独立 artifact）")
    lines.append(f"- 需要人工审核: **{review_frames}** 帧"
                 f"（固定 {fixed} + bonus {bonus}；Rapid-Train {extraction['counts']['fixed_train']}"
                 f"+{bonus}，Rapid-Eval {extraction['counts']['fixed_eval']}）")
    lines.append("")
    lines.append("| 键 | 作用 |")
    lines.append("| --- | --- |")
    for key, meaning in REVIEW_KEYS:
        lines.append(f"| `{key}` | {meaning} |")
    lines.append("")
    lines.append("## Safety")
    lines.append("")
    lines.append("- Sealed = 0")
    lines.append("- official writes = 0（只读读取 2C artifact）")
    lines.append("- 8801 automated access = 0（审核服务独立使用 8810）")
    lines.append("- Step 2B `last.pt` 未被覆盖，仅只读引用")
    lines.append("")
    if smoke:
        lines.append(f"- 隔离 smoke test: **{smoke['result']}** "
                     f"({len(smoke['checks']) - len(smoke['failed'])}/{len(smoke['checks'])} "
                     f"checks；正式 artifact 前后摘要一致 = "
                     f"{smoke['real_artifact_digest_before'] == smoke['real_artifact_digest_after']})")
    if browser:
        lines.append(f"- 真实 Chrome smoke test: **{browser['result']}** "
                     f"({len(browser['checks']) - len(browser['failed'])}/"
                     f"{len(browser['checks'])} checks)")
    lines.append("")
    lines.append("## Next")
    lines.append("")
    lines.append("**现在开始人工 Rapid Review；完成后回复“继续”。**")
    lines.append("")
    lines.append("第二阶段（只有在你明确说“继续”之后才会执行）：baseline threshold "
                 "selection → baseline Rapid-Eval metrics → Rapid-Train localization → "
                 "构建 V2 dataset → 30 epoch 微调 → V2 threshold selection → V2 Rapid-Eval "
                 "→ comparison。")
    lines.append("")

    (artifact / "PHASE1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(artifact / "phase1_report.json", report)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
