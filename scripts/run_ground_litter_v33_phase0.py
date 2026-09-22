"""Ground Litter V3.3 Phase 0 — real model feasibility probe.

Answers the DS4.1 plan's Phase 0 question with evidence instead of assumption:

  Can `turhancan_yolov8m_seg_trash.pt` see the audited ground targets when the
  Clean Reference prior hands it an enlarged context crop, and does enlargement
  separate a real small litter object from the known hard negatives?

This script does NOT touch the production API, containers, or network. It reads
only local review artifacts and the reviewed model weight.

Outputs (under --out):
  phase0_corpus.json     audited targets, per-frame labels, label conflicts
  phase0_cropgrid.json   full per-(target, expansion, imgsz, frame) results
  REPORT.md              human-readable, per-object (never only averages)

Usage:
  .venv/bin/python scripts/run_ground_litter_v33_phase0.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INFERENCE = "output/litter_source_20260914/inference_part*/results.jsonl"
DEFAULT_DEDUP = "output/litter_source_20260914/deduplicated_labels.json"
DEFAULT_MODEL = "models/litter/turhancan_yolov8m_seg_trash.pt"
DEFAULT_OUT = "output/ground_litter_v33_phase0_20260918"

# Label semantics taken from the review record itself.
NEGATIVE_STATES = {
    "non_litter_tool_provisional",
    "non_litter_container_provisional",
}
POSITIVE_STATES = {"visible_small_ground_object_probable_litter_unverified"}

CONF_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25)
EXPANSIONS = (2.0, 3.0, 4.0)
IMGSZ = (640, 1280)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def contains(outer, inner) -> bool:
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def plan_crop(frame, box, expansion, max_source=480, min_side=160):
    """Mirror the plan's 6.2 crop rule so Phase 0 tests what V3.3 would do."""
    h, w = frame.shape[:2]
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    long_side = max(box[2] - box[0], box[3] - box[1])
    side = int(min(max(min_side, long_side * expansion), max_source))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad = max(0, -x0, -y0, (x0 + side) - w, (y0 + side) - h)
    if pad > 0:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0 += pad
        y0 += pad
    crop = frame[y0:y0 + side, x0:x0 + side]
    # crop origin in ORIGINAL frame coordinates
    origin = (origin_x, origin_y) = (x0 - pad, y0 - pad)
    return crop, origin, side


def load_corpus(inference_glob: str, dedup_path: str) -> dict:
    records = []
    for path in sorted(glob.glob(str(REPO / inference_glob))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit("no inference records found; check --inference")

    per_target: dict[str, dict] = {}
    target_frames: dict[str, list] = defaultdict(list)
    for record in records:
        frame_id = record.get("sample_id")
        sample_path = record.get("sample_path")
        for target in (record.get("targets") or []):
            tid = target.get("target_id")
            box = tuple(float(v) for v in target.get("box") or ())
            if not tid or len(box) != 4:
                continue
            entry = per_target.setdefault(tid, {
                "target_id": tid,
                "box": box,
                "state": target.get("state"),
                "inside_roi": bool(target.get("inside_roi")),
                "frames_labeled": 0,
                "full_scan_hits": 0,
                "full_scan_confs": [],
            })
            entry["frames_labeled"] += 1
            raw = (target.get("stages") or {}).get("raw") or []
            if raw:
                entry["full_scan_hits"] += 1
                entry["full_scan_confs"].append(
                    round(max(float(r.get("confidence", 0.0)) for r in raw), 4)
                )
            target_frames[tid].append({
                "frame_id": frame_id,
                "sample_path": sample_path,
                "box": box,
                "raw": [
                    {"box": [float(v) for v in r.get("box", [])],
                     "confidence": float(r.get("confidence", 0.0)),
                     "label": r.get("label")}
                    for r in raw
                ],
            })

    for entry in per_target.values():
        confs = sorted(entry["full_scan_confs"])
        entry["full_scan_conf_range"] = (
            [confs[0], confs[-1]] if confs else None
        )
        entry["polarity"] = (
            "positive" if entry["state"] in POSITIVE_STATES
            else "negative" if entry["state"] in NEGATIVE_STATES
            else "unresolved"
        )

    # Label-conflict audit against the dedup file (the historical "true_litter" set).
    conflicts = []
    dedup = json.loads((REPO / dedup_path).read_text(encoding="utf-8"))
    for item in dedup.get("items", []):
        box = tuple(float(v) for v in item["observations"][0]["box"])
        for tid, entry in per_target.items():
            if contains(entry["box"], box):
                conflicts.append({
                    "dedup_item": item["item_id"],
                    "dedup_box": list(box),
                    "dedup_label_set": item.get("label_set"),
                    "dedup_frames": item.get("frame_count"),
                    "contained_in_target": tid,
                    "target_state": entry["state"],
                    "target_polarity": entry["polarity"],
                    "conflict": ("true_litter" in (item.get("label_set") or [])
                                 and entry["polarity"] != "positive"),
                })
    for addition in dedup.get("user_confirmed_additions", []):
        box = tuple(float(v) for v in addition["location_box_native_px"])
        for tid, entry in per_target.items():
            if contains(entry["box"], box):
                conflicts.append({
                    "dedup_item": addition["item_id"],
                    "dedup_box": list(box),
                    "dedup_label_set": ["user_confirmed_litter"],
                    "dedup_frames": addition.get("raw_candidate_frames"),
                    "contained_in_target": tid,
                    "target_state": entry["state"],
                    "target_polarity": entry["polarity"],
                    "conflict": entry["polarity"] != "positive",
                })

    return {
        "frame_count": len(records),
        "targets": per_target,
        "target_frames": target_frames,
        "label_conflicts": conflicts,
    }


def resolve_frame_path(sample_path: str, frame_id: str) -> Path | None:
    if sample_path and Path(sample_path).is_file():
        return Path(sample_path)
    fallback = REPO / f"output/litter_source_20260914/review/{frame_id}.jpg"
    return fallback if fallback.is_file() else None


def run_crop_grid(corpus: dict, model_path: Path, device: str, limit_frames: int | None):
    from ultralytics import YOLO

    started = time.perf_counter()
    model = YOLO(str(model_path))
    load_seconds = round(time.perf_counter() - started, 1)
    class_names = {int(k): str(v) for k, v in dict(model.names).items()}

    results = []
    cache: dict[str, np.ndarray] = {}
    for tid, entries in corpus["target_frames"].items():
        for item in entries:
            if limit_frames is not None and results:
                pass
            frame_id = item["frame_id"]
            path = resolve_frame_path(item.get("sample_path"), frame_id)
            if path is None:
                continue
            key = str(path)
            if key not in cache:
                cache[key] = cv2.imread(key)
            frame = cache[key]
            if frame is None:
                continue
            for expansion in EXPANSIONS:
                crop, origin, side = plan_crop(frame, item["box"], expansion)
                for imgsz in IMGSZ:
                    t0 = time.perf_counter()
                    prediction = model.predict(
                        crop, imgsz=imgsz, conf=min(CONF_THRESHOLDS),
                        verbose=False, device=device,
                    )[0]
                    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    detections = []
                    for box in prediction.boxes:
                        xyxy = [float(v) for v in box.xyxy[0]]
                        # map crop coords -> original frame coords
                        mapped = [
                            xyxy[0] + origin[0], xyxy[1] + origin[1],
                            xyxy[2] + origin[0], xyxy[3] + origin[1],
                        ]
                        detections.append({
                            "box": [round(v, 1) for v in mapped],
                            "confidence": round(float(box.conf), 4),
                            "class": class_names.get(int(box.cls), str(int(box.cls))),
                            "iou_with_target": round(iou(mapped, item["box"]), 4),
                            "center_in_target": bool(
                                item["box"][0] <= (mapped[0] + mapped[2]) / 2 <= item["box"][2]
                                and item["box"][1] <= (mapped[1] + mapped[3]) / 2 <= item["box"][3]
                            ),
                        })
                    detections.sort(key=lambda d: -d["confidence"])
                    results.append({
                        "target_id": tid,
                        "frame_id": frame_id,
                        "expansion": expansion,
                        "crop_side_px": side,
                        "imgsz": imgsz,
                        "inference_ms": elapsed_ms,
                        "target_box": list(item["box"]),
                        "detections": detections,
                    })
    return {
        "model": str(model_path),
        "device": device,
        "model_load_seconds": load_seconds,
        "class_names": class_names,
        "rows": results,
        "note": (
            "conf sweep is applied post-hoc on a conf=0.05 run; equivalent for "
            "per-class NMS, and no crop is ever run once per threshold."
        ),
    }


def summarise(corpus: dict, grid: dict) -> dict:
    by_key: dict[tuple, list] = defaultdict(list)
    for row in grid["rows"]:
        by_key[(row["target_id"], row["expansion"], row["imgsz"])].append(row)

    summary = {}
    for tid, entry in corpus["targets"].items():
        frames_labeled = entry["frames_labeled"]
        per_config = {}
        for (key_tid, expansion, imgsz), rows in sorted(by_key.items()):
            if key_tid != tid:
                continue
            hits_by_conf = {}
            for threshold in CONF_THRESHOLDS:
                hits = 0
                confs = []
                for row in rows:
                    eligible = [
                        d for d in row["detections"]
                        if d["confidence"] >= threshold and d["center_in_target"]
                    ]
                    if eligible:
                        hits += 1
                        confs.append(max(d["confidence"] for d in eligible))
                hits_by_conf[f"{threshold:.2f}"] = {
                    "frames_with_hit": hits,
                    "frames_tested": len(rows),
                    "frame_hit_rate": round(hits / len(rows), 4) if rows else None,
                    "max_conf_median": (
                        round(float(np.median(confs)), 4) if confs else None
                    ),
                }
            # best-IoU view, independent of the centre rule
            best_iou = max(
                (max((d["iou_with_target"] for d in row["detections"]), default=0.0)
                 for row in rows), default=0.0,
            )
            per_config[f"exp{expansion}_imgsz{imgsz}"] = {
                "hits_by_conf": hits_by_conf,
                "best_iou_any_detection": round(float(best_iou), 4),
                "inference_ms_median": round(float(np.median(
                    [r["inference_ms"] for r in rows])), 1) if rows else None,
                "inference_ms_p95": round(float(np.percentile(
                    [r["inference_ms"] for r in rows], 95)), 1) if rows else None,
            }
        summary[tid] = {
            "state": entry["state"],
            "polarity": entry["polarity"],
            "box": list(entry["box"]),
            "box_size_px": [entry["box"][2] - entry["box"][0],
                            entry["box"][3] - entry["box"][1]],
            "inside_roi": entry["inside_roi"],
            "frames_labeled": frames_labeled,
            "full_scan_hits": entry["full_scan_hits"],
            "full_scan_conf_range": entry["full_scan_conf_range"],
            "full_scan_hit_rate": round(entry["full_scan_hits"] / frames_labeled, 4)
            if frames_labeled else None,
            "crop_configs": per_config,
        }
    return summary


def write_report(out_dir: Path, corpus: dict, grid: dict, summary: dict) -> None:
    lines = []
    lines.append("# Ground Litter V3.3 — Phase 0 模型可行性实测\n")
    lines.append(f"- 权重：`{grid['model']}`")
    lines.append(f"- 设备：`{grid['device']}`，模型加载 {grid['model_load_seconds']}s")
    lines.append(f"- 类别：{grid['class_names']}")
    lines.append(f"- 审核帧数：{corpus['frame_count']}\n")

    lines.append("## 1. 语料真实盘点（含标注冲突）\n")
    lines.append("| target_id | 状态 | 极性 | 框(原生) | 尺寸 | in ROI | 全扫描命中 | 全扫描conf |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for tid, s in summary.items():
        lines.append(
            f"| `{tid}` | {s['state']} | {s['polarity']} | {s['box']} | "
            f"{s['box_size_px'][0]}×{s['box_size_px'][1]} | {s['inside_roi']} | "
            f"{s['full_scan_hits']}/{s['frames_labeled']} | {s['full_scan_conf_range']} |"
        )
    lines.append("")
    conflicts = corpus["label_conflicts"]
    if conflicts:
        lines.append("### 标注冲突（会让闸门失效，必须先解决）\n")
        lines.append("| dedup 条目 | dedup 标签 | 落在哪个 target | 该 target 的真实标注 | 冲突 |")
        lines.append("|---|---|---|---|---|")
        for c in conflicts:
            lines.append(
                f"| {c['dedup_item']} | {c['dedup_label_set']} | `{c['contained_in_target']}` "
                f"| {c['target_state']} | {'**是**' if c['conflict'] else '否'} |"
            )
        lines.append("")

    lines.append("## 2. 放大裁剪后的检测结果（逐目标，不取平均）\n")
    for tid, s in summary.items():
        lines.append(f"### `{tid}`（{s['polarity']}，{s['box_size_px'][0]}×{s['box_size_px'][1]}px）\n")
        lines.append("| 配置 | conf阈值 | 命中帧/测试帧 | 命中率 | 中位命中conf | 推理P95(ms) |")
        lines.append("|---|---|---|---|---|---|")
        for cfg, data in s["crop_configs"].items():
            for threshold, h in data["hits_by_conf"].items():
                lines.append(
                    f"| {cfg} | {threshold} | {h['frames_with_hit']}/{h['frames_tested']} | "
                    f"{h['frame_hit_rate']} | {h['max_conf_median']} | {data['inference_ms_p95']} |"
                )
        lines.append("")

    lines.append("## 3. 判读\n")
    lines.append("- 命中定义：裁剪内任一检测框的**中心落在人工目标框内**；")
    lines.append("  同时单独记录 `best_iou_any_detection` 作为不受中心规则影响的对照。")
    lines.append("- 全扫描命中来自既有审核记录的 `stages.raw`，与本次裁剪实验同源不同管线。")
    lines.append("- **本报告不给出生产召回率**：语料量级与标注一致性都不足以支持。\n")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", default=DEFAULT_INFERENCE)
    parser.add_argument("--dedup", default=DEFAULT_DEDUP)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-frames", type=int, default=None,
                        help="debug: cap frames per target")
    args = parser.parse_args()

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(args.inference, args.dedup)
    (out_dir / "phase0_corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.limit_frames:
        for tid in list(corpus["target_frames"]):
            corpus["target_frames"][tid] = corpus["target_frames"][tid][:args.limit_frames]

    grid = run_crop_grid(corpus, REPO / args.model, args.device, args.limit_frames)
    (out_dir / "phase0_cropgrid.json").write_text(
        json.dumps(grid, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = summarise(corpus, grid)
    (out_dir / "phase0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(out_dir, corpus, grid, summary)
    print(f"wrote {out_dir}/REPORT.md and 3 JSON files", flush=True)
    for tid, s in summary.items():
        print(f"  {tid:<24} {s['polarity']:<11} full_scan {s['full_scan_hits']}/{s['frames_labeled']}",
              flush=True)


if __name__ == "__main__":
    main()
