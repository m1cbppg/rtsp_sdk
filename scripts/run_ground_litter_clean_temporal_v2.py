#!/usr/bin/env python3
"""Development V2: seed/support Clean Reference proposals plus frozen adaptive mask.

The V2 acceptance target deliberately ignores truly tiny isolated changes.  A
proposal needs at least four high-threshold seed pixels, but is displayed only
when weaker surrounding evidence reconstructs a normal-sized footprint.
Adaptive masks are calibrated from a declared clean prelude and frozen before
evaluation; they never modify the long-term Clean Reference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ground-litter-v2")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ground_litter_clean_temporal_poc import (  # noqa: E402
    ANALYSIS_FPS,
    LUMINANCE_THRESHOLD,
    SIGNATURE_THRESHOLD,
    T01_APPEARANCE_SECONDS,
    T01_REGION,
    boxes_related,
    center,
    draw_tracks,
    iter_sampled_frames,
    load_json,
    plot_timeline,
    raw_jsonl,
    read_frame,
    read_raw_jsonl,
    robust_normalize,
    save_image,
    t01_metrics,
    track_funnel,
    track_rows,
    transform_reference,
    video_metadata,
)


SUPPORT_SIGNATURE_THRESHOLD = 1.2
SUPPORT_LUMINANCE_THRESHOLD = 25.0
MINIMUM_SEED_PIXELS = 4
MINIMUM_SUPPORT_AREA = 30
MINIMUM_SUPPORT_SHORT_SIDE = 6
MAXIMUM_SUPPORT_SIDE = 80
MAXIMUM_SUPPORT_BOX_AREA = 2500
ADAPTIVE_CALIBRATION_SECONDS = 60.0
ADAPTIVE_HIGH_FREQUENCY = 0.40
ADAPTIVE_SUPPORT_FREQUENCY = 0.80


def residual_maps(reference: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    old = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    new = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def signature(gray: np.ndarray) -> np.ndarray:
        mean = cv2.GaussianBlur(gray, (0, 0), 12)
        variance = np.maximum(cv2.GaussianBlur(gray * gray, (0, 0), 12) - mean * mean, 0)
        return (gray - mean) / (np.sqrt(variance) + 5)

    signature_residual = np.abs(signature(new) - signature(old))
    delta = new - old
    luma_residual = np.abs(delta - cv2.GaussianBlur(delta, (0, 0), 12))
    return signature_residual, luma_residual


def candidate_maps(
    signature: np.ndarray,
    luminance: np.ndarray,
    valid: np.ndarray,
    adaptive_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = (valid > 0) & (adaptive_mask == 0)
    seed = (
        (signature >= SIGNATURE_THRESHOLD)
        & (luminance >= LUMINANCE_THRESHOLD)
        & eligible
    ).astype(np.uint8)
    support = (
        (signature >= SUPPORT_SIGNATURE_THRESHOLD)
        & (luminance >= SUPPORT_LUMINANCE_THRESHOLD)
        & eligible
    ).astype(np.uint8)
    # Large high-threshold regions are temporary unavailable, not litter.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    broad = np.zeros_like(seed)
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area >= 900 or (area >= 350 and max(width, height) >= 80):
            broad[labels == index] = 255
    if broad.any():
        broad = cv2.dilate(broad, np.ones((15, 15), np.uint8))
        seed[broad > 0] = 0
        support[broad > 0] = 0
    marker = cv2.dilate(seed, np.ones((9, 9), np.uint8))
    grown = cv2.bitwise_and(support, marker)
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return seed, grown, broad


def proposals(
    normalized_reference: np.ndarray,
    current: np.ndarray,
    valid: np.ndarray,
    adaptive_mask: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    signature, luminance = residual_maps(normalized_reference, current)
    seed, grown, broad = candidate_maps(signature, luminance, valid, adaptive_mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grown, 8)
    rows = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        component = labels == index
        seed_pixels = int(seed[component].sum())
        if seed_pixels < MINIMUM_SEED_PIXELS:
            continue
        if area < MINIMUM_SUPPORT_AREA or min(width, height) < MINIMUM_SUPPORT_SHORT_SIDE:
            continue
        if max(width, height) > MAXIMUM_SUPPORT_SIDE or width * height > MAXIMUM_SUPPORT_BOX_AREA:
            continue
        rows.append({
            "box": [x, y, x + width, y + height],
            "source": "clean_reference_seed_support_v2",
            "seed_pixels": seed_pixels,
            "support_area_px": area,
            "support_short_side_px": min(width, height),
            "support_fill_ratio": round(area / max(width * height, 1), 4),
            "p90_signature_residual": round(float(np.percentile(signature[component], 90)), 3),
            "p90_local_luminance_residual": round(float(np.percentile(luminance[component], 90)), 3),
            "anomaly_score": round(float(min(1.0,
                0.4 * seed_pixels / 12.0
                + 0.3 * np.percentile(signature[component], 90) / 4.0
                + 0.3 * np.percentile(luminance[component], 90) / 150.0)), 4),
        })
    rows.sort(key=lambda row: (-row["anomaly_score"], -row["support_area_px"]))
    return rows, grown, broad


def build_adaptive_mask(
    video: Path,
    aligned_reference: np.ndarray,
    valid: np.ndarray,
    output: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    high_count = np.zeros(valid.shape, np.uint16)
    support_count = np.zeros(valid.shape, np.uint16)
    samples = 0
    empty = np.zeros(valid.shape, np.uint8)
    for timestamp, frame in iter_sampled_frames(
        video, sample_fps=ANALYSIS_FPS, start=0.0, end=ADAPTIVE_CALIBRATION_SECONDS
    ):
        normalized, _ = robust_normalize(aligned_reference, frame, valid)
        signature, luminance = residual_maps(normalized, frame)
        seed, grown, _broad = candidate_maps(signature, luminance, valid, empty)
        high_count += (seed > 0).astype(np.uint16)
        support_count += (grown > 0).astype(np.uint16)
        samples += 1
    high_frequency = high_count.astype(np.float32) / max(samples, 1)
    support_frequency = support_count.astype(np.float32) / max(samples, 1)
    adaptive = (
        (high_frequency >= ADAPTIVE_HIGH_FREQUENCY)
        | (support_frequency >= ADAPTIVE_SUPPORT_FREQUENCY)
    ).astype(np.uint8) * 255
    adaptive = cv2.dilate(adaptive, np.ones((3, 3), np.uint8))
    adaptive[valid == 0] = 0
    save_image(output / "adaptive_noise_mask.png", adaptive)
    heat = cv2.applyColorMap(np.clip(high_frequency * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[valid == 0] = (20, 20, 20)
    save_image(output / "adaptive_high_frequency_heatmap.png", heat)
    diagnostics = {
        "calibration_seconds": ADAPTIVE_CALIBRATION_SECONDS,
        "sample_count": samples,
        "frozen_after_seconds": ADAPTIVE_CALIBRATION_SECONDS,
        "high_frequency_threshold": ADAPTIVE_HIGH_FREQUENCY,
        "support_frequency_threshold": ADAPTIVE_SUPPORT_FREQUENCY,
        "masked_valid_fraction": float(np.count_nonzero(adaptive) / max(np.count_nonzero(valid), 1)),
        "updates_after_calibration": 0,
    }
    (output / "adaptive_build.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return adaptive, diagnostics


def scan_video(
    name: str,
    video: Path,
    output: Path,
    reference: np.ndarray,
    valid: np.ndarray,
    metadata: dict[str, Any],
    *,
    adaptive: bool,
) -> dict[str, Any]:
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    representative = read_frame(video, 0.0)
    aligned_reference, aligned_valid, alignment = transform_reference(reference, valid, representative)
    if adaptive:
        adaptive_mask, adaptive_info = build_adaptive_mask(
            video, aligned_reference, aligned_valid, directory)
        start = ADAPTIVE_CALIBRATION_SECONDS
    else:
        adaptive_mask = np.zeros(aligned_valid.shape, np.uint8)
        adaptive_info = {"enabled": False}
        start = 0.0
    rows_by_time: dict[float, list[dict[str, Any]]] = {}
    heat = np.zeros(aligned_valid.shape, np.float32)
    last_frame = representative
    started = time.perf_counter()
    for index, (timestamp, frame) in enumerate(iter_sampled_frames(
        video, sample_fps=ANALYSIS_FPS, start=start
    )):
        normalized, _illumination = robust_normalize(aligned_reference, frame, aligned_valid)
        rows, _grown, _broad = proposals(
            normalized, frame, aligned_valid, adaptive_mask)
        rows_by_time[timestamp] = rows
        for row in rows:
            x, y, right, bottom = row["box"]
            heat[y:bottom, x:right] += 1
        last_frame = frame
        if index and index % 60 == 0:
            print(f"[{name}] {timestamp:.0f}s raw={sum(map(len, rows_by_time.values()))}", flush=True)
    tracks = track_rows(rows_by_time, source="clean_reference_seed_support_v2",
                        sample_fps=ANALYSIS_FPS)
    raw_jsonl(directory / "raw_candidates.jsonl", rows_by_time)
    (directory / "tracks.json").write_text(json.dumps(tracks, indent=2) + "\n")
    plot_timeline(directory / "timeline.png", tracks, f"{name}: V2 seed/support tracks",
                  metadata["duration_seconds"])
    save_image(directory / "stable_tracks_overlay.jpg",
               draw_tracks(last_frame, tracks, f"{name}: V2 tracks >=5s"))
    colored = cv2.applyColorMap(np.clip(heat / max(float(heat.max()), 1) * 255, 0, 255).astype(np.uint8),
                                cv2.COLORMAP_TURBO)
    colored[aligned_valid == 0] = (20, 20, 20)
    save_image(directory / "candidate_heatmap.png", colored)
    result = {
        "name": name,
        "evaluation_start_seconds": start,
        "sample_fps": ANALYSIS_FPS,
        "alignment": alignment,
        "adaptive": adaptive_info,
        "metrics": track_funnel(tracks, sum(map(len, rows_by_time.values())),
                                 metadata["duration_seconds"] - start),
        "t01": t01_metrics(rows_by_time, tracks, ANALYSIS_FPS) if "positive" in name else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def fusion_from_cached_yolo(
    v1_output: Path,
    v2_output: Path,
    positive_video: Path,
    duration: float,
) -> dict[str, Any]:
    directory = v2_output / "positive_fusion"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = load_json(v1_output / "positive_yolo_fusion" / "checkpoint_rows.json")
    reference_rows = read_raw_jsonl(v2_output / "positive_reference" / "raw_candidates.jsonl")
    yolo_by_time: dict[float, list[dict[str, Any]]] = {}
    fusion_by_time: dict[float, list[dict[str, Any]]] = {}
    for key, payload in checkpoint.items():
        timestamp = float(key)
        if timestamp < ADAPTIVE_CALIBRATION_SECONDS:
            continue
        yolo_rows = payload["retained_yolo"]
        reference = reference_rows.get(timestamp, [])
        yolo_by_time[timestamp] = yolo_rows
        fused = []
        used: set[int] = set()
        for row in yolo_rows:
            matches = [index for index, anomaly in enumerate(reference)
                       if boxes_related(row["box"], anomaly["box"])]
            used.update(matches)
            fused.append({**row, "source": "fusion_v2",
                          "sources": ["yolo", "clean_reference_v2"] if matches else ["yolo"]})
        for index, row in enumerate(reference):
            if index not in used:
                fused.append({**row, "source": "fusion_v2", "sources": ["clean_reference_v2"]})
        fusion_by_time[timestamp] = fused
    yolo_tracks = track_rows(yolo_by_time, source="yolo", sample_fps=.5)
    fusion_tracks = track_rows(fusion_by_time, source="fusion_v2", sample_fps=.5)
    raw_jsonl(directory / "fusion_raw_candidates.jsonl", fusion_by_time)
    (directory / "fusion_tracks.json").write_text(json.dumps(fusion_tracks, indent=2) + "\n")
    plot_timeline(directory / "fusion_timeline.png", fusion_tracks,
                  "positive: V2 reference + cached YOLO", duration)
    last = read_frame(positive_video, duration - 1)
    save_image(directory / "fusion_stable_overlay.jpg",
               draw_tracks(last, fusion_tracks, "V2 fusion tracks >=5s"))
    result = {
        "evaluation_start_seconds": ADAPTIVE_CALIBRATION_SECONDS,
        "yolo": track_funnel(yolo_tracks, sum(map(len, yolo_by_time.values())),
                             duration - ADAPTIVE_CALIBRATION_SECONDS),
        "fusion": track_funnel(fusion_tracks, sum(map(len, fusion_by_time.values())),
                               duration - ADAPTIVE_CALIBRATION_SECONDS),
        "t01_yolo": t01_metrics(yolo_by_time, yolo_tracks, .5),
        "t01_fusion": t01_metrics(fusion_by_time, fusion_tracks, .5),
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def comparable_v1_metrics(v1_output: Path, output: Path,
                          positive_duration: float, negative_duration: float) -> dict[str, Any]:
    results = {}
    for name, duration in (("positive_reference", positive_duration),
                           ("negative_reference", negative_duration)):
        rows = read_raw_jsonl(v1_output / name / "raw_candidates.jsonl")
        rows = {timestamp: values for timestamp, values in rows.items()
                if timestamp >= ADAPTIVE_CALIBRATION_SECONDS}
        tracks = track_rows(rows, source="v1_comparable", sample_fps=1.0)
        results[name] = track_funnel(
            tracks, sum(map(len, rows.values())), duration - ADAPTIVE_CALIBRATION_SECONDS)
    (output / "v1_comparable_60s.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def create_t01_evidence(output: Path, video: Path) -> None:
    rows_by_time = read_raw_jsonl(output / "positive_reference" / "raw_candidates.jsonl")
    panels = []
    for timestamp in (70.0, 72.0, 74.0, 76.0, 80.0, 82.0):
        frame = read_frame(video, timestamp)
        x, y, right, bottom = T01_REGION
        margin = 100
        a, b = max(0, x-margin), max(0, y-margin)
        c, d = min(frame.shape[1], right+margin), min(frame.shape[0], bottom+margin)
        crop = frame[b:d, a:c].copy()
        for row in rows_by_time.get(timestamp, []):
            if not boxes_related(row["box"], T01_REGION):
                continue
            p, q, r, s = row["box"]
            cv2.rectangle(crop, (p-a, q-b), (r-a, s-b), (0, 255, 255), 2)
            cv2.putText(crop, f"seed {row['seed_pixels']} area {row['support_area_px']}",
                        (p-a, max(18, q-b-4)), cv2.FONT_HERSHEY_SIMPLEX, .42,
                        (0, 255, 255), 1, cv2.LINE_AA)
        crop = cv2.resize(crop, (350, 310), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(crop, (0, 0), (349, 34), (20, 20, 20), -1)
        cv2.putText(crop, f"V2 Reference t={timestamp:.0f}s", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(crop)
    save_image(output / "positive_reference" / "t01_v2_evidence.jpg", np.hstack(panels))


def create_vlm_manifest(
    output: Path,
    video: Path,
    reference: np.ndarray,
    tracks: list[dict[str, Any]],
    *,
    prefix: str,
    limit: int = 40,
) -> int:
    directory = output / "vlm_review" / prefix
    directory.mkdir(parents=True, exist_ok=True)
    selected = sorted((row for row in tracks if row["lifetime_seconds"] >= 3),
                      key=lambda row: -row["lifetime_seconds"])[:limit]
    manifest = []
    for row in selected:
        timestamp = float(row["timestamps"][len(row["timestamps"]) // 2])
        current = read_frame(video, timestamp)
        x, y, right, bottom = (int(round(value)) for value in row["median_box"])
        object_margin = max(24, max(right-x, bottom-y))
        oa, ob = max(0, x-object_margin), max(0, y-object_margin)
        oc, od = min(current.shape[1], right+object_margin), min(current.shape[0], bottom+object_margin)
        context_margin = max(80, 3 * max(right-x, bottom-y))
        a, b = max(0, x-context_margin), max(0, y-context_margin)
        c, d = min(current.shape[1], right+context_margin), min(current.shape[0], bottom+context_margin)
        reference_path = directory / f"track-{row['track_id']:04d}-reference.jpg"
        current_path = directory / f"track-{row['track_id']:04d}-current.jpg"
        context_path = directory / f"track-{row['track_id']:04d}-context.jpg"
        save_image(reference_path, reference[ob:od, oa:oc])
        save_image(current_path, current[ob:od, oa:oc])
        context = current[b:d, a:c].copy()
        cv2.rectangle(context, (x-a, y-b), (right-a, bottom-b), (0, 255, 255), 2)
        save_image(context_path, context)
        manifest.append({
            "review_id": f"{prefix}-track-{row['track_id']}",
            "clean_reference_crop": str(reference_path.resolve()),
            "current_crop": str(current_path.resolve()),
            "context_crop": str(context_path.resolve()),
            "anomaly_bbox": row["median_box"],
            "object_crop_origin": [oa, ob],
            "timestamp": timestamp,
            "track": {key: row[key] for key in (
                "track_id", "source", "first_seen", "last_seen", "hits",
                "lifetime_seconds", "visible_duration_seconds", "continuity",
            )},
            "suggested_state": (
                "ANOMALY_CONFIRMED" if row["lifetime_seconds"] >= 5
                else "ANOMALY_PENDING"
            ),
            "allowed_labels": [
                "NEW_GROUND_OBJECT", "OCCLUDED", "FIXED_FACILITY",
                "ENVIRONMENT_CHANGE", "CLEAR_NON_LITTER", "UNCERTAIN",
            ],
        })
    with (output / "vlm_review" / f"{prefix}_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in manifest:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(manifest)


def write_report(output: Path, positive: dict[str, Any], negative: dict[str, Any],
                 fusion: dict[str, Any], manifest_counts: dict[str, int],
                 v1: dict[str, Any]) -> None:
    t01 = positive["t01"]
    negative_counts = negative["metrics"]["tracks_at_least_seconds"]
    gate_target = t01["stable_5s_delay"] is not None and t01["stable_5s_delay"] <= 5
    report = f"""# Clean Reference V2：正常尺寸优先实验

## 结论

- V2 忽略真正只有 3–4px、没有周边支持的变化；候选必须有 ≥4 个强 seed，并恢复为面积 ≥30px、短边 ≥6px 的物体区域。
- T01 首次 raw={t01['first_raw_seconds']}，稳定 1/3/5 秒={t01['first_stable_1s']} / {t01['first_stable_3s']} / {t01['first_stable_5s']}，5 秒延迟={t01['stable_5s_delay']}。目标门槛：**{'PASS' if gate_target else 'FAIL'}**。
- 负样本评估从 60 秒开始，raw={negative['metrics']['raw_candidates']}，≥5/10/30 秒轨迹={negative_counts['5']} / {negative_counts['10']} / {negative_counts['30']}。
- 正样本 Fusion ≥5 秒轨迹：YOLO={fusion['yolo']['tracks_at_least_seconds']['5']}，V2 Fusion={fusion['fusion']['tracks_at_least_seconds']['5']}。
- 同一 60 秒后区间，正样本 V1→V2：raw {v1['positive_reference']['raw_candidates']}→{positive['metrics']['raw_candidates']}，≥10 秒轨迹 {v1['positive_reference']['tracks_at_least_seconds']['10']}→{positive['metrics']['tracks_at_least_seconds']['10']}。
- 同一 60 秒后区间，负样本 V1→V2：raw {v1['negative_reference']['raw_candidates']}→{negative['metrics']['raw_candidates']}，≥10 秒轨迹 {v1['negative_reference']['tracks_at_least_seconds']['10']}→{negative['metrics']['tracks_at_least_seconds']['10']}。
- T01 区域在目标出现前 64–66 秒也有环境残差轨迹；V2 解决了尺寸和召回问题，但语义精度仍需 actor/context 与 VLM。
- Adaptive mask 只使用已声明无目标的 0–60 秒，60 秒后冻结；Clean Reference 始终只读。这是开发集方法，生产中必须由人工 clean confirmation 和 actor/OCCLUDED mask 保护初始化。

## 固定参数

- 强 seed：signature≥{SIGNATURE_THRESHOLD} 且 luminance≥{LUMINANCE_THRESHOLD}
- 周边支持：signature≥{SUPPORT_SIGNATURE_THRESHOLD} 且 luminance≥{SUPPORT_LUMINANCE_THRESHOLD}
- seed≥{MINIMUM_SEED_PIXELS}px，support area≥{MINIMUM_SUPPORT_AREA}px，support short side≥{MINIMUM_SUPPORT_SHORT_SIDE}px
- Adaptive 高频出现率≥{ADAPTIVE_HIGH_FREQUENCY:.0%} 或 support 出现率≥{ADAPTIVE_SUPPORT_FREQUENCY:.0%}

## VLM 接口

- 已生成 positive {manifest_counts['positive']} 条、negative {manifest_counts['negative']} 条 review manifest。
- 每条包含 clean reference crop、current crop、context crop、bbox、时间戳和 persistence。
- 本轮没有调用 VLM；manifest 只验证输入契约。

## 边界

- V2 参数在 T01 开发样本上确定，不能作为独立召回结论。
- Negative 视频没有逐框语义真值；长轨迹可能包含真实新增地面物、车辆边缘或设施变化。
- 下一轮必须使用新的 YOLO 漏检样本做盲测，并固定 V2 参数不再调节。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    payload = {
        "kind": "development_v2_seed_support_adaptive",
        "parameters": {
            "seed_signature": SIGNATURE_THRESHOLD,
            "seed_luminance": LUMINANCE_THRESHOLD,
            "support_signature": SUPPORT_SIGNATURE_THRESHOLD,
            "support_luminance": SUPPORT_LUMINANCE_THRESHOLD,
            "minimum_seed_pixels": MINIMUM_SEED_PIXELS,
            "minimum_support_area": MINIMUM_SUPPORT_AREA,
            "minimum_support_short_side": MINIMUM_SUPPORT_SHORT_SIDE,
            "adaptive_calibration_seconds": ADAPTIVE_CALIBRATION_SECONDS,
        },
        "positive": positive,
        "negative": negative,
        "fusion": fusion,
        "v1_comparable_from_60s": v1,
        "vlm_manifest_counts": manifest_counts,
        "gate_target_pass": gate_target,
    }
    (output / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    contract = """# VLM review contract

VLM 只复核候选，不从整图主动寻找垃圾。每条 JSONL 输入包含：

- `clean_reference_crop`：冻结 Clean Reference 的局部图；
- `current_crop`：同位置当前局部图；
- `context_crop`：包含周边语义的当前图；
- `anomaly_bbox`、`timestamp`、`track`：位置和 persistence；
- `suggested_state`：`ANOMALY_PENDING` 或 `ANOMALY_CONFIRMED`。

期望输出：`label`、`confidence`、`reason`。`label` 只能是：
`NEW_GROUND_OBJECT`、`OCCLUDED`、`FIXED_FACILITY`、
`ENVIRONMENT_CHANGE`、`CLEAR_NON_LITTER`、`UNCERTAIN`。

只有 `NEW_GROUND_OBJECT` 且通过业务垃圾类别规则时进入
`LITTER_CONFIRMED`；`UNCERTAIN` 保留为疑似新增地面异物。
"""
    (output / "VLM_REVIEW_CONTRACT.md").write_text(contract, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--v1-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    clean = args.input_dir / "clean_reference.mp4"
    positive_video = args.input_dir / "小垃圾正样本.mp4"
    negative_video = args.input_dir / "正常负样本.mp4"
    arrays = np.load(args.v1_output / "reference_arrays.npz")
    reference, valid = arrays["reference"], arrays["valid"]
    positive_metadata = video_metadata(positive_video)
    negative_metadata = video_metadata(negative_video)
    positive_path = args.output / "positive_reference" / "summary.json"
    if positive_path.exists():
        positive = load_json(positive_path)
    else:
        positive = scan_video("positive_reference", positive_video, args.output,
                              reference, valid, positive_metadata, adaptive=True)
    negative_path = args.output / "negative_reference" / "summary.json"
    if negative_path.exists():
        negative = load_json(negative_path)
    else:
        negative = scan_video("negative_reference", negative_video, args.output,
                              reference, valid, negative_metadata, adaptive=True)
    fusion_path = args.output / "positive_fusion" / "summary.json"
    if fusion_path.exists():
        fusion = load_json(fusion_path)
    else:
        fusion = fusion_from_cached_yolo(args.v1_output, args.output, positive_video,
                                         positive_metadata["duration_seconds"])
    v1 = comparable_v1_metrics(args.v1_output, args.output,
                               positive_metadata["duration_seconds"],
                               negative_metadata["duration_seconds"])
    create_t01_evidence(args.output, positive_video)
    positive_tracks = load_json(args.output / "positive_fusion" / "fusion_tracks.json")
    negative_tracks = load_json(args.output / "negative_reference" / "tracks.json")
    manifest_counts = {
        "positive": create_vlm_manifest(args.output, positive_video, reference, positive_tracks,
                                        prefix="positive_fusion"),
        "negative": create_vlm_manifest(args.output, negative_video, reference, negative_tracks,
                                        prefix="negative"),
    }
    write_report(args.output, positive, negative, fusion, manifest_counts, v1)
    print(json.dumps({"output": str(args.output), "positive": positive["metrics"],
                      "negative": negative["metrics"], "t01": positive["t01"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
