#!/usr/bin/env python3
"""Build an event-level V3.1 PoC from reviewed V3 anomaly tracks.

This is deliberately an offline evaluation layer.  It does not change the
production ground-litter path and it never updates the Clean Reference.  It
adds two pieces that the V3 review showed were missing:

* label-agnostic fixed-view event consolidation for fragmented tracks; and
* conservative context-unavailable evidence when a small candidate touches a
  much larger current-vs-reference change component.

Human labels are used only after predictions are produced, for evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ground_litter_clean_temporal_poc import read_frame  # noqa: E402
from run_ground_litter_clean_temporal_v2 import (  # noqa: E402
    SUPPORT_LUMINANCE_THRESHOLD,
    SUPPORT_SIGNATURE_THRESHOLD,
    residual_maps,
)
from run_ground_litter_prior_region_v3 import protected_normalize  # noqa: E402


EVENT_CENTER_DISTANCE_PX = 20.0
EVENT_SIZE_RATIO = 3.0
CONTEXT_MARGIN_PX = 80
CONTEXT_INNER_PAD_PX = 12
CONTEXT_TOUCH_GAP_PX = 20
CONTEXT_CANDIDATE_GUARD_PX = 4
CONTEXT_OCCLUSION_EXTERNAL_AREA_PX = 200


def box_center(box: Iterable[float]) -> tuple[float, float]:
    left, top, right, bottom = box
    return (left + right) / 2.0, (top + bottom) / 2.0


def box_area(box: Iterable[float]) -> float:
    left, top, right, bottom = box
    return max(0.0, right - left) * max(0.0, bottom - top)


def same_fixed_view_anchor(
    left: Iterable[float],
    right: Iterable[float],
    *,
    center_distance_px: float = EVENT_CENTER_DISTANCE_PX,
    size_ratio: float = EVENT_SIZE_RATIO,
) -> bool:
    left = list(left)
    right = list(right)
    areas = (max(box_area(left), 1.0), max(box_area(right), 1.0))
    if max(areas) / min(areas) > size_ratio:
        return False
    return math.dist(box_center(left), box_center(right)) <= center_distance_px


def group_tracks_into_events(
    tracks: list[dict[str, Any]],
    *,
    center_distance_px: float = EVENT_CENTER_DISTANCE_PX,
    size_ratio: float = EVENT_SIZE_RATIO,
) -> list[list[dict[str, Any]]]:
    """Group tracks by a fixed-view ground anchor without using review labels."""
    parent = list(range(len(tracks)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, track in enumerate(tracks):
        for other_index in range(index):
            if same_fixed_view_anchor(
                track["median_box"],
                tracks[other_index]["median_box"],
                center_distance_px=center_distance_px,
                size_ratio=size_ratio,
            ):
                union(index, other_index)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, track in enumerate(tracks):
        grouped.setdefault(find(index), []).append(track)
    return sorted(
        (sorted(group, key=lambda row: int(row["track_id"])) for group in grouped.values()),
        key=lambda group: min(int(row["track_id"]) for row in group),
    )


def measure_context_support(
    support: np.ndarray,
    valid: np.ndarray,
    box: Iterable[float],
    *,
    margin_px: int = CONTEXT_MARGIN_PX,
    inner_pad_px: int = CONTEXT_INNER_PAD_PX,
    touch_gap_px: int = CONTEXT_TOUCH_GAP_PX,
    candidate_guard_px: int = CONTEXT_CANDIDATE_GUARD_PX,
) -> dict[str, Any]:
    """Measure broad residual structure around a candidate.

    The candidate's padded core is excluded from the ring fraction.  A large
    support component touching the expanded candidate is evidence that the
    ground is temporarily unavailable, not evidence that the candidate is
    permanently non-litter.
    """
    height, width = support.shape
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    left = max(0, center_x - margin_px)
    top = max(0, center_y - margin_px)
    right = min(width, center_x + margin_px + 1)
    bottom = min(height, center_y + margin_px + 1)

    local_support = (support[top:bottom, left:right] > 0).astype(np.uint8)
    local_valid = valid[top:bottom, left:right] > 0
    ring = np.ones(local_support.shape, bool)
    inner_left = max(left, x1 - inner_pad_px) - left
    inner_top = max(top, y1 - inner_pad_px) - top
    inner_right = min(right, x2 + inner_pad_px) - left
    inner_bottom = min(bottom, y2 + inner_pad_px) - top
    ring[inner_top:inner_bottom, inner_left:inner_right] = False
    ring_valid = ring & local_valid

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        local_support, 8
    )
    largest_component = 0
    touching_component = 0
    touching_external_component = 0
    guard_left = max(left, x1 - candidate_guard_px) - left
    guard_top = max(top, y1 - candidate_guard_px) - top
    guard_right = min(right, x2 + candidate_guard_px) - left
    guard_bottom = min(bottom, y2 + candidate_guard_px) - top
    outside_candidate = np.ones(local_support.shape, bool)
    outside_candidate[guard_top:guard_bottom, guard_left:guard_right] = False
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        largest_component = max(largest_component, area)
        component_left = left + int(stats[component, cv2.CC_STAT_LEFT])
        component_top = top + int(stats[component, cv2.CC_STAT_TOP])
        component_right = component_left + int(stats[component, cv2.CC_STAT_WIDTH])
        component_bottom = component_top + int(stats[component, cv2.CC_STAT_HEIGHT])
        touches = (
            component_left < x2 + touch_gap_px
            and component_right > x1 - touch_gap_px
            and component_top < y2 + touch_gap_px
            and component_bottom > y1 - touch_gap_px
        )
        if touches:
            touching_component = max(touching_component, area)
            external_area = int(
                np.count_nonzero((labels == component) & outside_candidate)
            )
            touching_external_component = max(
                touching_external_component, external_area
            )

    ring_pixels = max(int(np.count_nonzero(ring_valid)), 1)
    return {
        "context_box": [left, top, right, bottom],
        "ring_support_fraction": round(
            float(np.count_nonzero((local_support > 0) & ring_valid) / ring_pixels), 6
        ),
        "largest_support_component_px": largest_component,
        "touching_support_component_px": touching_component,
        "touching_external_support_px": touching_external_component,
    }


def union_visible_seconds(timestamps: Iterable[float], sample_period: float = 1.0) -> float:
    ordered = sorted(set(float(value) for value in timestamps))
    if not ordered:
        return 0.0
    # At 1 FPS each unique hit is one second of visible evidence.  Keeping the
    # helper explicit avoids counting hidden gaps as persistence.
    return len(ordered) * sample_period


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_context_features(
    manifest: list[dict[str, Any]],
    *,
    video: Path,
    reference: np.ndarray,
    aligned_valid: np.ndarray,
    alignment_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    height, width = aligned_valid.shape
    aligned_reference = cv2.warpPerspective(
        reference,
        alignment_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    aligned_valid = cv2.warpPerspective(
        aligned_valid,
        alignment_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    cache: dict[float, tuple[np.ndarray, dict[str, Any]]] = {}
    features = []
    for item in manifest:
        timestamp = float(item["timestamp"])
        if timestamp not in cache:
            frame = read_frame(video, timestamp)
            normalized, usable, diagnostics = protected_normalize(
                aligned_reference, frame, aligned_valid
            )
            signature, luminance = residual_maps(normalized, frame)
            support = (
                (signature >= SUPPORT_SIGNATURE_THRESHOLD)
                & (luminance >= SUPPORT_LUMINANCE_THRESHOLD)
                & (usable > 0)
            ).astype(np.uint8)
            cache[timestamp] = support, diagnostics
        support, diagnostics = cache[timestamp]
        measurement = measure_context_support(
            support, aligned_valid, item["anomaly_bbox"]
        )
        external = int(measurement["touching_external_support_px"])
        features.append({
            "review_id": item["review_id"],
            "track_id": int(item["track"]["track_id"]),
            "timestamp": timestamp,
            **measurement,
            "environment_state": diagnostics["state"],
            "predicted_visibility": (
                "OCCLUDED_CONTEXT"
                if external >= CONTEXT_OCCLUSION_EXTERNAL_AREA_PX
                else "VISIBLE"
            ),
        })
    return features


def event_rows(
    groups: list[list[dict[str, Any]]],
    *,
    manifest_by_track: dict[int, dict[str, Any]],
    reviews_by_track: dict[int, dict[str, Any]],
    context_by_track: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    events = []
    for number, group in enumerate(groups, 1):
        track_ids = [int(track["track_id"]) for track in group]
        labels = sorted({reviews_by_track[track_id]["label"] for track_id in track_ids})
        timestamps = [
            timestamp
            for track in group
            for timestamp in track.get("timestamps", [])
        ]
        occluded_tracks = [
            track_id for track_id in track_ids
            if context_by_track[track_id]["predicted_visibility"] == "OCCLUDED_CONTEXT"
        ]
        occluded_fraction = len(occluded_tracks) / len(track_ids)
        state = "OCCLUDED" if occluded_fraction >= 0.5 else "ANOMALY_PENDING"
        boxes = np.asarray([track["median_box"] for track in group], np.float64)
        events.append({
            "event_id": f"negative-v31-event-{number:03d}",
            "track_ids": track_ids,
            "track_count": len(track_ids),
            "anchor_box": np.median(boxes, axis=0).round(2).tolist(),
            "first_seen": min(float(track["first_seen"]) for track in group),
            "last_seen": max(float(track["last_seen"]) for track in group),
            "visible_duration_seconds": union_visible_seconds(timestamps),
            "maximum_track_lifetime_seconds": max(
                float(track["lifetime_seconds"]) for track in group
            ),
            "predicted_state": state,
            "context_occluded_track_ids": occluded_tracks,
            "context_occluded_track_fraction": round(occluded_fraction, 4),
            "representative_review_id": manifest_by_track[track_ids[0]]["review_id"],
            "human_label": labels[0] if len(labels) == 1 else "LABEL_CONFLICT",
            "human_track_labels": {
                str(track_id): reviews_by_track[track_id]["label"]
                for track_id in track_ids
            },
        })
    return events


def evaluation_payload(
    events: list[dict[str, Any]],
    context_features: list[dict[str, Any]],
    *,
    fingerprint: str,
) -> dict[str, Any]:
    visible_events = [row for row in events if row["predicted_state"] != "OCCLUDED"]
    litter_events = [row for row in events if row["human_label"] == "LITTER_CONFIRMED"]
    visible_litter = [
        row for row in visible_events if row["human_label"] == "LITTER_CONFIRMED"
    ]
    non_litter_events = [
        row for row in events if row["human_label"] == "CLEAR_NON_LITTER"
    ]
    occluded_non_litter = [
        row for row in non_litter_events if row["predicted_state"] == "OCCLUDED"
    ]
    track_labels = Counter(
        row["human_label"]
        for event in events
        for row in ({"human_label": label} for label in event["human_track_labels"].values())
    )
    return {
        "kind": "ground_litter_event_memory_v31_offline_poc",
        "source_fingerprint": fingerprint,
        "parameters": {
            "event_center_distance_px": EVENT_CENTER_DISTANCE_PX,
            "event_size_ratio": EVENT_SIZE_RATIO,
            "context_margin_px": CONTEXT_MARGIN_PX,
            "context_touch_gap_px": CONTEXT_TOUCH_GAP_PX,
            "context_candidate_guard_px": CONTEXT_CANDIDATE_GUARD_PX,
            "context_occlusion_external_area_px": CONTEXT_OCCLUSION_EXTERNAL_AREA_PX,
            "human_labels_used_for_prediction": False,
            "clean_reference_updated": False,
        },
        "track_level": {
            "reviewed_tracks": len(context_features),
            "label_counts": dict(track_labels),
            "context_occluded_tracks": sum(
                row["predicted_visibility"] == "OCCLUDED_CONTEXT"
                for row in context_features
            ),
        },
        "event_level": {
            "events_total": len(events),
            "litter_events": len(litter_events),
            "non_litter_events": len(non_litter_events),
            "visible_events_after_context_gate": len(visible_events),
            "visible_litter_events": len(visible_litter),
            "occluded_non_litter_events": len(occluded_non_litter),
            "litter_event_recall_after_context_gate": round(
                len(visible_litter) / max(len(litter_events), 1), 4
            ),
            "visible_candidate_precision": round(
                len(visible_litter) / max(len(visible_events), 1), 4
            ),
            "non_litter_event_deferral_rate": round(
                len(occluded_non_litter) / max(len(non_litter_events), 1), 4
            ),
        },
    }


def write_report(output: Path, evaluation: dict[str, Any], events: list[dict[str, Any]]) -> None:
    track = evaluation["track_level"]
    event = evaluation["event_level"]
    duplicate_events = [row for row in events if row["track_count"] > 1]
    visible_false = [
        row for row in events
        if row["predicted_state"] != "OCCLUDED"
        and row["human_label"] == "CLEAR_NON_LITTER"
    ]
    report = f"""# Ground Litter V3.1：事件记忆与上下文不可用 PoC

## 结论

- 输入 {track['reviewed_tracks']} 条已审核 V3 轨迹，经不使用人工标签的固定视角空间合并后得到 {event['events_total']} 个异常事件。
- 人工真值约为 {event['litter_events']} 个垃圾位置和 {event['non_litter_events']} 个非垃圾事件。
- 保守的上下文不可用规则将 {event['occluded_non_litter_events']}/{event['non_litter_events']} 个非垃圾事件转为 `OCCLUDED`，垃圾事件保留 {event['visible_litter_events']}/{event['litter_events']}。
- 进入后续 YOLO/VLM 判断的可见事件为 {event['visible_events_after_context_gate']} 个，当前样本上的事件级候选 precision 为 {event['visible_candidate_precision']:.1%}。
- 这条规则只表示地面视觉被较大变化占用；它不输出 `non_litter`，不更新 Clean Reference，也不会把遮挡物学习成干净地面。

## 解决的问题

- 同一垃圾因遮挡和检测闪烁形成多条 Track：事件层按固定地面坐标合并。
- 人、手推车、摊位边缘形成小连通域：当候选与更大的上下文变化连通域相接时，暂停判断。
- Persistence 改用事件累计可见时间；隐藏间隔不算可见证据，也不因为一次遮挡立即创建新事件。

## 仍需处理

- 当前仍有 {len(visible_false)} 个非垃圾事件未被纯视觉上下文规则拦截：{', '.join(row['event_id'] + '=' + '/'.join(map(str, row['track_ids'])) for row in visible_false) or '无'}。
- 下一步应对这部分运行 actor/context 语义检测；墙脚或 ROI 边界应使用显式边界保护，不应继续降低全局连通域阈值。
- 当前事件合并在整个约五分钟片段内按固定坐标聚类，尚未加入“确认地面恢复干净后关闭事件”的在线生命周期判断。
- 候选框外 200px 是当前 1440p 素材上的保守 PoC 阈值。它经过 T01 保护检查；跨摄像头前仍应改为透视尺度归一化，并在新摄像头上冻结验证，不能直接复制绝对像素值。

## 重复事件

"""
    for row in duplicate_events:
        report += (
            f"- `{row['event_id']}`：tracks={row['track_ids']}，"
            f"human={row['human_label']}，state={row['predicted_state']}\n"
        )
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--v1-output", type=Path, required=True)
    parser.add_argument("--v3-output", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest_path = args.v3_output / "vlm_review/negative_v3_manifest.jsonl"
    manifest_raw = manifest_path.read_bytes()
    fingerprint = hashlib.sha256(manifest_raw).hexdigest()[:16]
    manifest = [
        json.loads(line)
        for line in manifest_raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    review = load_json(args.review)
    if review.get("fingerprint") != fingerprint:
        raise ValueError(
            f"review fingerprint {review.get('fingerprint')} does not match {fingerprint}"
        )
    reviews_by_track = {
        int(row["track_id"]): row for row in review.get("reviewed", [])
    }
    manifest_by_track = {
        int(row["track"]["track_id"]): row for row in manifest
    }
    if set(reviews_by_track) != set(manifest_by_track):
        raise ValueError("reviewed track IDs do not exactly match the V3 review manifest")

    all_tracks = load_json(args.v3_output / "negative/v3_tracks.json")
    reviewed_tracks = [
        row for row in all_tracks if int(row["track_id"]) in reviews_by_track
    ]
    if len(reviewed_tracks) != len(reviews_by_track):
        raise ValueError("one or more reviewed tracks are missing from V3 tracks")

    arrays = np.load(args.v1_output / "reference_arrays.npz")
    reference = arrays["reference"]
    aligned_valid = cv2.imread(
        str(args.v3_output / "reference_valid_mask_v3.png"), cv2.IMREAD_GRAYSCALE
    )
    if aligned_valid is None:
        raise FileNotFoundError(args.v3_output / "reference_valid_mask_v3.png")
    negative_summary = load_json(args.v3_output / "negative/summary.json")
    alignment_matrix = np.asarray(
        negative_summary["alignment"]["matrix"], np.float64
    )
    context = build_context_features(
        manifest,
        video=args.input_dir / "正常负样本.mp4",
        reference=reference,
        aligned_valid=aligned_valid,
        alignment_matrix=alignment_matrix,
    )
    context_by_track = {int(row["track_id"]): row for row in context}
    groups = group_tracks_into_events(reviewed_tracks)
    events = event_rows(
        groups,
        manifest_by_track=manifest_by_track,
        reviews_by_track=reviews_by_track,
        context_by_track=context_by_track,
    )
    evaluation = evaluation_payload(events, context, fingerprint=fingerprint)

    (args.output / "context_features.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "events.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output, evaluation, events)
    print(json.dumps({"output": str(args.output), **evaluation}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
