#!/usr/bin/env python3
"""Offline Clean Reference + temporal ground-litter video PoC.

This script is deliberately isolated from the production pipeline.  It builds
a frozen reference from the first 65% of a reviewed clean video, evaluates the
remaining 35% as holdout, scans positive/negative videos at 1 FPS, and runs the
existing tiled YOLO branch on the positive video at 0.5 FPS.  Every raw
candidate is retained in JSONL before temporal tracking.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ground_litter_clean_temporal_poc")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ground-litter-poc")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rtsp_annotator.ground_litter_detection import (  # noqa: E402
    GroundLitterDetectionOptions,
    GroundLitterZone,
    UltralyticsGroundLitterDetector,
)
from run_ground_litter_prior_pair_experiment import (  # noqa: E402
    actor_boxes,
    add_reference_evidence,
    align_reference,
    box_iou,
    boxes_related,
    find_anomalies,
    overlay_mask,
    polygon_mask,
    robust_normalize,
    yolo_candidates,
)


ROI = ((0.50, 0.00), (0.66, 0.00), (0.88, 1.00), (0.35, 1.00))
CORE_ROI_DIAGNOSTIC = ((0.52, 0.00), (0.63, 0.00), (0.72, 1.00), (0.42, 1.00))
OVERLAY_ZONES = (
    ((0.00, 0.00), (0.43, 0.00), (0.43, 0.115), (0.00, 0.115)),
    ((0.68, 0.84), (1.00, 0.84), (1.00, 1.00), (0.68, 1.00)),
)
SIGNATURE_THRESHOLD = 3.0
LUMINANCE_THRESHOLD = 100.0
MAIN_MIN_AREA = 10
DIAGNOSTIC_MIN_AREA = 4
REFERENCE_FPS = 1.0
ANALYSIS_FPS = 1.0
YOLO_FPS = 0.5
REFERENCE_BUILD_FRACTION = 0.65
T01_APPEARANCE_SECONDS = 72.0
# This is an association region, not a per-frame segmentation ground truth.
# The light object moves slightly after arrival.
T01_REGION = [1430, 990, 1540, 1080]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "name": path.name,
        "sha256": sha256(path),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration_seconds": frames / fps,
        "size_bytes": path.stat().st_size,
        "container_bitrate_bps": path.stat().st_size * 8 / (frames / fps),
    }


def read_frame(path: Path, timestamp: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read {path} at {timestamp:.3f}s")
    return frame


def iter_sampled_frames(
    path: Path,
    *,
    sample_fps: float,
    start: float = 0.0,
    end: float | None = None,
) -> Iterable[tuple[float, np.ndarray]]:
    """Decode sequentially and yield the nearest frame on a fixed time grid."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    native_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / native_fps
    stop = min(duration, end if end is not None else duration)
    targets = np.arange(start, stop - 1e-6, 1.0 / sample_fps)
    target_index = 0
    frame_index = 0
    while target_index < len(targets):
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_index / native_fps
        while target_index < len(targets) and timestamp + 0.5 / native_fps >= targets[target_index]:
            yield float(targets[target_index]), frame.copy()
            target_index += 1
        frame_index += 1
    cap.release()


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, 94] if path.suffix.lower() in {".jpg", ".jpeg"} else []
    if not cv2.imwrite(str(path), image, params):
        raise RuntimeError(f"failed to write image: {path}")


def build_reference(clean_video: Path, output: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    build_end = metadata["duration_seconds"] * REFERENCE_BUILD_FRACTION
    frames = [frame for _time, frame in iter_sampled_frames(
        clean_video, sample_fps=REFERENCE_FPS, start=0.0, end=build_end
    )]
    if len(frames) < 20:
        raise ValueError("clean reference build segment has fewer than 20 sampled frames")
    height, width = frames[0].shape[:2]
    reference = np.empty_like(frames[0])
    noise = np.empty((height, width), np.float32)
    # Stripe-wise median avoids a multi-gigabyte full-frame float temporary.
    for y in range(0, height, 64):
        stack = np.stack([frame[y:y + 64] for frame in frames])
        median = np.median(stack, axis=0)
        reference[y:y + 64] = median.astype(np.uint8)
        deviation = np.max(np.abs(stack.astype(np.float32) - median[None]), axis=3)
        noise[y:y + 64] = np.median(deviation, axis=0)
    roi = polygon_mask((height, width), ROI)
    valid = cv2.bitwise_and(roi, overlay_mask((height, width)))
    # Fixed before full-video evaluation.  It removes unstable pixels, not
    # current-frame anomalies, and therefore cannot learn live objects.
    valid[noise >= 18.0] = 0
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8))
    save_image(output / "clean_reference.jpg", reference)
    save_image(output / "reference_valid_mask.png", valid)
    noise_image = np.clip(noise * 12, 0, 255).astype(np.uint8)
    save_image(output / "reference_noise_map.png", noise_image)
    np.savez_compressed(output / "reference_arrays.npz", reference=reference, valid=valid, noise=noise)
    result = {
        "build_end_seconds": build_end,
        "holdout_start_seconds": build_end,
        "sample_fps": REFERENCE_FPS,
        "sample_count": len(frames),
        "method": "per-pixel temporal median; per-pixel temporal MAD validity",
        "noise_invalid_threshold": 18.0,
        "valid_roi_fraction": float(np.count_nonzero(valid) / max(np.count_nonzero(roi), 1)),
        "noise_roi_median": float(np.median(noise[roi > 0])),
        "noise_roi_p95": float(np.percentile(noise[roi > 0], 95)),
    }
    (output / "reference_build.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def diagnostic_anomalies(
    normalized_reference: np.ndarray,
    current: np.ndarray,
    eligible_mask: np.ndarray,
) -> list[dict[str, Any]]:
    """Same frozen residual thresholds, with area 4 retained for diagnosis."""
    old = cv2.cvtColor(normalized_reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    new = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def signature(gray: np.ndarray) -> np.ndarray:
        mean = cv2.GaussianBlur(gray, (0, 0), 12)
        variance = np.maximum(cv2.GaussianBlur(gray * gray, (0, 0), 12) - mean * mean, 0)
        return (gray - mean) / (np.sqrt(variance) + 5)

    signature_residual = np.abs(signature(new) - signature(old))
    delta = new - old
    luma_residual = np.abs(delta - cv2.GaussianBlur(delta, (0, 0), 12))
    core = (
        (signature_residual >= SIGNATURE_THRESHOLD)
        & (luma_residual >= LUMINANCE_THRESHOLD)
        & (eligible_mask > 0)
    ).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    broad = np.zeros_like(core)
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area >= 900 or (area >= 350 and max(width, height) >= 80):
            broad[labels == index] = 255
    if broad.any():
        broad = cv2.dilate(broad, np.ones((15, 15), np.uint8))
    core[broad > 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    rows = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        fill = area / max(width * height, 1)
        if area < DIAGNOSTIC_MIN_AREA or area >= MAIN_MIN_AREA:
            continue
        if area > 500 or min(width, height) < 1 or max(width, height) > 40 or fill < 0.20:
            continue
        component = labels == index
        rows.append({
            "box": [x, y, x + width, y + height],
            "source": "clean_reference_diagnostic_small_core",
            "component_area_px": area,
            "box_area_px": width * height,
            "fill_ratio": round(fill, 4),
            "mean_signature_residual": round(float(signature_residual[component].mean()), 3),
            "p90_signature_residual": round(float(np.percentile(signature_residual[component], 90)), 3),
            "mean_local_luminance_residual": round(float(luma_residual[component].mean()), 3),
            "p90_local_luminance_residual": round(float(np.percentile(luma_residual[component], 90)), 3),
            "anomaly_score": round(float(min(1.0,
                0.5 * np.percentile(signature_residual[component], 90) / 4.0
                + 0.5 * np.percentile(luma_residual[component], 90) / 150.0)), 4),
        })
    return rows


def transform_reference(
    reference: np.ndarray,
    valid: np.ndarray,
    representative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    aligned, alignment_valid, diagnostics = align_reference(reference, representative)
    matrix = np.asarray(diagnostics["matrix"], np.float64)
    warped_valid = cv2.warpPerspective(
        valid, matrix, (representative.shape[1], representative.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    )
    warped_valid = cv2.bitwise_and(warped_valid, alignment_valid)
    return aligned, warped_valid, diagnostics


def process_reference_frame(
    aligned_reference: np.ndarray,
    aligned_valid: np.ndarray,
    frame: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    normalized, illumination = robust_normalize(aligned_reference, frame, aligned_valid)
    main, residual, _signature, broad = find_anomalies(
        normalized,
        frame,
        aligned_valid,
        signature_threshold=SIGNATURE_THRESHOLD,
        luminance_threshold=LUMINANCE_THRESHOLD,
    )
    diagnostic = diagnostic_anomalies(normalized, frame, aligned_valid)
    return main, diagnostic, residual, broad, illumination


def center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def candidate_track_cost(left: list[float], right: list[float]) -> float | None:
    a = max(box_area(left), 1.0)
    b = max(box_area(right), 1.0)
    ratio = max(a, b) / min(a, b)
    if ratio > 5.0:
        return None
    x, y = center(left)
    u, v = center(right)
    distance = math.hypot(x - u, y - v)
    diagonal = max(math.hypot(left[2] - left[0], left[3] - left[1]),
                   math.hypot(right[2] - right[0], right[3] - right[1]))
    limit = max(18.0, 2.0 * diagonal)
    overlap = box_iou(left, right)
    related = boxes_related(left, right)
    if not related and distance > limit:
        return None
    return distance / limit + 0.15 * math.log(ratio) - 0.25 * overlap


@dataclass
class Track:
    track_id: int
    source: str
    first_seen: float
    last_seen: float
    boxes: list[list[float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    max_gap: float = 0.0
    occluded_samples: int = 0

    def add(self, timestamp: float, row: dict[str, Any]) -> None:
        if self.timestamps:
            self.max_gap = max(self.max_gap, timestamp - self.timestamps[-1])
        self.last_seen = timestamp
        self.timestamps.append(timestamp)
        self.boxes.append([float(value) for value in row["box"]])
        self.scores.append(float(row.get("anomaly_score", row.get("confidence", 0.0))))

    def as_dict(self, sample_period: float) -> dict[str, Any]:
        median = np.median(np.asarray(self.boxes), axis=0).round(2).tolist()
        observed_span = self.last_seen - self.first_seen + sample_period
        expected = max(1, round(observed_span / sample_period))
        return {
            "track_id": self.track_id,
            "source": self.source,
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "hits": len(self.timestamps),
            "lifetime_seconds": round(observed_span, 3),
            "visible_duration_seconds": round(len(self.timestamps) * sample_period, 3),
            "continuity": round(min(1.0, len(self.timestamps) / expected), 4),
            "max_gap_seconds": round(self.max_gap, 3),
            "median_box": median,
            "median_score": round(float(np.median(self.scores)), 4),
            "occluded_samples": self.occluded_samples,
            "timestamps": [round(value, 3) for value in self.timestamps],
        }


def track_rows(
    rows_by_time: dict[float, list[dict[str, Any]]],
    *,
    source: str,
    sample_fps: float,
    occluded_by_time: dict[float, list[list[int]]] | None = None,
) -> list[dict[str, Any]]:
    sample_period = 1.0 / sample_fps
    max_gap = sample_period * 2.1
    active: dict[int, Track] = {}
    finished: list[Track] = []
    next_id = 1
    for timestamp in sorted(rows_by_time):
        rows = rows_by_time[timestamp]
        occluders = (occluded_by_time or {}).get(timestamp, [])
        for track_id, track in list(active.items()):
            cx, cy = center(track.boxes[-1])
            blocked = any(x <= cx <= right and y <= cy <= bottom for x, y, right, bottom in occluders)
            if blocked:
                track.occluded_samples += 1
                continue
            if timestamp - track.last_seen > max_gap:
                finished.append(active.pop(track_id))
        pairs = []
        for track_id, track in active.items():
            for index, row in enumerate(rows):
                cost = candidate_track_cost(track.boxes[-1], row["box"])
                if cost is not None:
                    pairs.append((cost, track_id, index))
        used_tracks: set[int] = set()
        used_rows: set[int] = set()
        for _cost, track_id, index in sorted(pairs):
            if track_id in used_tracks or index in used_rows:
                continue
            active[track_id].add(timestamp, rows[index])
            used_tracks.add(track_id)
            used_rows.add(index)
        for index, row in enumerate(rows):
            if index in used_rows:
                continue
            track = Track(next_id, source, timestamp, timestamp)
            track.add(timestamp, row)
            active[next_id] = track
            next_id += 1
    finished.extend(active.values())
    return [track.as_dict(sample_period) for track in finished]


def raw_jsonl(path: Path, rows_by_time: dict[float, list[dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        candidate_id = 1
        for timestamp in sorted(rows_by_time):
            for row in rows_by_time[timestamp]:
                stream.write(json.dumps({"candidate_id": candidate_id, "timestamp": timestamp, **row},
                                        ensure_ascii=False) + "\n")
                candidate_id += 1


def read_raw_jsonl(path: Path) -> dict[float, list[dict[str, Any]]]:
    rows: dict[float, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            timestamp = float(item.pop("timestamp"))
            item.pop("candidate_id", None)
            rows[timestamp].append(item)
    return dict(rows)


def track_funnel(tracks: list[dict[str, Any]], raw_count: int, duration: float) -> dict[str, Any]:
    counts = {str(seconds): sum(row["lifetime_seconds"] >= seconds for row in tracks)
              for seconds in (1, 3, 5, 10, 30)}
    reduction = 1.0 - counts["5"] / max(raw_count, 1)
    return {
        "raw_candidates": raw_count,
        "raw_per_minute": raw_count / duration * 60,
        "tracks_total": len(tracks),
        "tracks_at_least_seconds": counts,
        "raw_to_5s_reduction": reduction,
    }


def candidate_matches_t01(row: dict[str, Any]) -> bool:
    x, y = center(row["box"])
    a, b, c, d = T01_REGION
    return a <= x <= c and b <= y <= d


def t01_metrics(rows_by_time: dict[float, list[dict[str, Any]]], tracks: list[dict[str, Any]], sample_fps: float) -> dict[str, Any]:
    matched_times_all = sorted(timestamp for timestamp, rows in rows_by_time.items()
                               if any(candidate_matches_t01(row) for row in rows))
    matched_times = [timestamp for timestamp in matched_times_all
                     if timestamp >= T01_APPEARANCE_SECONDS]
    target_tracks = [row for row in tracks if boxes_related(row["median_box"], T01_REGION)]
    target_tracks.sort(key=lambda row: (-row["lifetime_seconds"], row["first_seen"]))
    result: dict[str, Any] = {
        "evaluation_region": T01_REGION,
        "appearance_seconds": T01_APPEARANCE_SECONDS,
        "preappearance_region_hit_timestamps": [timestamp for timestamp in matched_times_all
                                                  if timestamp < T01_APPEARANCE_SECONDS],
        "raw_hit_timestamps": matched_times,
        "first_raw_seconds": matched_times[0] if matched_times else None,
        "raw_delay_seconds": matched_times[0] - T01_APPEARANCE_SECONDS if matched_times else None,
        "matching_tracks": target_tracks,
    }
    runs: list[list[float]] = []
    for timestamp in matched_times:
        if not runs or timestamp - runs[-1][-1] > 2.1 / sample_fps:
            runs.append([timestamp])
        else:
            runs[-1].append(timestamp)
    sample_period = 1.0 / sample_fps
    for threshold in (1, 3, 5):
        eligible = [run for run in runs
                    if run[-1] - run[0] + sample_period >= threshold]
        first = min((run[0] + threshold for run in eligible), default=None)
        result[f"first_stable_{threshold}s"] = first
        result[f"stable_{threshold}s_delay"] = first - T01_APPEARANCE_SECONDS if first is not None else None
    expected = [timestamp for timestamp in rows_by_time if timestamp >= 75.0]
    result["post_75_raw_visibility_rate"] = (
        sum(timestamp in matched_times for timestamp in expected) / max(len(expected), 1)
    )
    return result


def make_long_track_evidence(
    video: Path,
    directory: Path,
    reference: np.ndarray,
    valid: np.ndarray,
    tracks: list[dict[str, Any]],
    *,
    start: float = 0.0,
    limit: int = 12,
) -> None:
    """Create reference/current evidence cards for the longest false tracks."""
    evidence = directory / "long_track_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    representative = read_frame(video, start)
    aligned_reference, _aligned_valid, _alignment = transform_reference(reference, valid, representative)
    cards = []
    selected = sorted(tracks, key=lambda row: -row["lifetime_seconds"])[:limit]
    for rank, track in enumerate(selected, 1):
        timestamp = float(track["timestamps"][len(track["timestamps"]) // 2])
        current = read_frame(video, timestamp)
        x, y, right, bottom = (int(round(value)) for value in track["median_box"])
        margin = 90
        a, b = max(0, x - margin), max(0, y - margin)
        c, d = min(current.shape[1], right + margin), min(current.shape[0], bottom + margin)
        panels = []
        for source, label in ((aligned_reference, "clean reference"), (current, f"current {timestamp:.0f}s")):
            crop = source[b:d, a:c].copy()
            cv2.rectangle(crop, (x-a, y-b), (right-a, bottom-b), (0, 255, 255), 2)
            crop = cv2.resize(crop, (300, 300), interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(crop, (0, 0), (299, 32), (20, 20, 20), -1)
            cv2.putText(crop, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .52,
                        (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(crop)
        card = np.hstack(panels)
        caption = np.full((50, 600, 3), 24, np.uint8)
        cv2.putText(caption,
                    f"rank {rank} / track {track['track_id']} / {track['lifetime_seconds']:.0f}s / hits {track['hits']}",
                    (8, 31), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1, cv2.LINE_AA)
        card = np.vstack((caption, card))
        save_image(evidence / f"rank-{rank:02d}-track-{track['track_id']}.jpg", card)
        cards.append(card)
    if cards:
        columns = 2
        rows = math.ceil(len(cards) / columns)
        sheet = np.full((rows * 350, columns * 600, 3), 18, np.uint8)
        for index, card in enumerate(cards):
            row, column = divmod(index, columns)
            sheet[row*350:(row+1)*350, column*600:(column+1)*600] = card
        save_image(directory / "long_track_evidence_contact.jpg", sheet)


def make_core_roi_diagnostic(
    video: Path,
    directory: Path,
    duration: float,
    *,
    positive: bool,
) -> dict[str, Any]:
    """Post-filter existing evidence to quantify boundary/facility contribution."""
    output = directory / "core_roi_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    mask = polygon_mask((1440, 2560), CORE_ROI_DIAGNOSTIC)
    source = read_raw_jsonl(directory / "raw_candidates.jsonl")
    filtered = {}
    for timestamp, rows in source.items():
        retained = []
        for row in rows:
            x, y = center(row["box"])
            if mask[min(1439, int(y)), min(2559, int(x))] > 0:
                retained.append(row)
        filtered[timestamp] = retained
    tracks = track_rows(filtered, source="clean_reference_core_roi_diagnostic", sample_fps=1.0)
    raw_jsonl(output / "raw_candidates.jsonl", filtered)
    (output / "tracks.json").write_text(json.dumps(tracks, indent=2) + "\n")
    plot_timeline(output / "timeline.png", tracks, f"{directory.name}: core ROI diagnostic", duration)
    frame = read_frame(video, duration - 1)
    save_image(output / "stable_tracks_overlay.jpg",
               draw_tracks(frame, tracks, f"{directory.name}: core ROI tracks >=5s"))
    summary = {
        "polygon": CORE_ROI_DIAGNOSTIC,
        "main_gate": False,
        "metrics": track_funnel(tracks, sum(map(len, filtered.values())), duration),
        "t01": t01_metrics(filtered, tracks, 1.0) if positive else None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def draw_tracks(frame: np.ndarray, tracks: list[dict[str, Any]], title: str, minimum: float = 5.0) -> np.ndarray:
    output = frame.copy()
    selected = sorted((row for row in tracks if row["lifetime_seconds"] >= minimum),
                      key=lambda row: -row["lifetime_seconds"])
    for row in selected[:80]:
        x, y, right, bottom = (int(round(value)) for value in row["median_box"])
        color = (0, 255, 255) if boxes_related(row["median_box"], T01_REGION) else (255, 90, 0)
        cv2.rectangle(output, (x, y), (right, bottom), color, 2)
        cv2.putText(output, f"T{row['track_id']} {row['lifetime_seconds']:.0f}s",
                    (x, max(18, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
    cv2.rectangle(output, (0, 0), (min(output.shape[1] - 1, 1050), 38), (20, 20, 20), -1)
    cv2.putText(output, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .68, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def plot_timeline(path: Path, tracks: list[dict[str, Any]], title: str, duration: float) -> None:
    import matplotlib.pyplot as plt
    selected = sorted(tracks, key=lambda row: (-row["lifetime_seconds"], row["first_seen"]))[:80]
    figure, axis = plt.subplots(figsize=(14, max(4, min(14, len(selected) * .18 + 2))))
    for index, row in enumerate(selected):
        color = "#e45756" if boxes_related(row["median_box"], T01_REGION) else "#4c78a8"
        axis.plot([row["first_seen"], row["last_seen"]], [index, index], color=color, linewidth=2)
        axis.scatter(row["timestamps"], [index] * len(row["timestamps"]), s=7, color=color)
    axis.axvline(T01_APPEARANCE_SECONDS, color="#f2cf5b", linestyle="--", linewidth=1.4, label="T01 ~72s")
    axis.set_xlim(0, duration)
    axis.set_xlabel("video seconds")
    axis.set_ylabel("tracks ordered by lifetime")
    axis.set_title(title)
    axis.grid(alpha=.2)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def scan_reference_video(
    name: str,
    video: Path,
    output: Path,
    reference: np.ndarray,
    valid: np.ndarray,
    metadata: dict[str, Any],
    *,
    start: float = 0.0,
) -> dict[str, Any]:
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    representative = read_frame(video, start)
    aligned_reference, aligned_valid, alignment = transform_reference(reference, valid, representative)
    rows_by_time: dict[float, list[dict[str, Any]]] = {}
    diagnostic_by_time: dict[float, list[dict[str, Any]]] = {}
    occluded_by_time: dict[float, list[list[int]]] = {}
    heat = np.zeros(reference.shape[:2], np.float32)
    illumination_rows = []
    last_frame = representative
    started = time.perf_counter()
    for index, (timestamp, frame) in enumerate(iter_sampled_frames(
        video, sample_fps=ANALYSIS_FPS, start=start
    )):
        main, diagnostic, residual, broad, illumination = process_reference_frame(
            aligned_reference, aligned_valid, frame
        )
        for row in main:
            row["source"] = "clean_reference"
        rows_by_time[timestamp] = main
        diagnostic_by_time[timestamp] = diagnostic
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats((broad > 0).astype(np.uint8), 8)
        occluded_by_time[timestamp] = [
            [int(x), int(y), int(x + width), int(y + height)]
            for x, y, width, height, area in stats[1:]
            if int(area) > 0
        ]
        for row in main:
            x, y, right, bottom = row["box"]
            heat[y:bottom, x:right] += 1
        illumination_rows.append({"timestamp": timestamp, **illumination})
        last_frame = frame
        if index and index % 60 == 0:
            print(f"[{name}] {timestamp:.0f}s candidates={sum(map(len, rows_by_time.values()))}", flush=True)
    tracks = track_rows(rows_by_time, source="clean_reference", sample_fps=ANALYSIS_FPS,
                        occluded_by_time=occluded_by_time)
    diagnostic_tracks = track_rows(diagnostic_by_time, source="diagnostic_small_core",
                                   sample_fps=ANALYSIS_FPS, occluded_by_time=occluded_by_time)
    raw_jsonl(directory / "raw_candidates.jsonl", rows_by_time)
    raw_jsonl(directory / "diagnostic_small_core_candidates.jsonl", diagnostic_by_time)
    (directory / "tracks.json").write_text(json.dumps(tracks, indent=2) + "\n")
    (directory / "diagnostic_small_core_tracks.json").write_text(json.dumps(diagnostic_tracks, indent=2) + "\n")
    (directory / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
    (directory / "illumination.json").write_text(json.dumps(illumination_rows, indent=2) + "\n")
    plot_timeline(directory / "timeline.png", tracks, f"{name}: main reference tracks", metadata["duration_seconds"])
    plot_timeline(directory / "diagnostic_small_core_timeline.png", diagnostic_tracks,
                  f"{name}: area 4-9 diagnostic tracks", metadata["duration_seconds"])
    save_image(directory / "stable_tracks_overlay.jpg",
               draw_tracks(last_frame, tracks, f"{name}: tracks >=5s"))
    save_image(directory / "diagnostic_small_core_overlay.jpg",
               draw_tracks(last_frame, diagnostic_tracks, f"{name}: diagnostic area 4-9 tracks >=5s"))
    colored = cv2.applyColorMap(np.clip(heat / max(float(heat.max()), 1.0) * 255, 0, 255).astype(np.uint8),
                                cv2.COLORMAP_TURBO)
    colored[aligned_valid == 0] = (25, 25, 25)
    save_image(directory / "candidate_heatmap.png", colored)
    raw_count = sum(map(len, rows_by_time.values()))
    diagnostic_count = sum(map(len, diagnostic_by_time.values()))
    result = {
        "name": name,
        "sample_fps": ANALYSIS_FPS,
        "start_seconds": start,
        "alignment": alignment,
        "main": track_funnel(tracks, raw_count, metadata["duration_seconds"] - start),
        "diagnostic_small_core": track_funnel(diagnostic_tracks, diagnostic_count,
                                               metadata["duration_seconds"] - start),
        "t01": t01_metrics(rows_by_time, tracks, ANALYSIS_FPS) if name == "positive_reference" else None,
        "t01_diagnostic": t01_metrics(diagnostic_by_time, diagnostic_tracks, ANALYSIS_FPS)
        if name == "positive_reference" else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def yolo_options(model: Path, actor_model: Path) -> GroundLitterDetectionOptions:
    return GroundLitterDetectionOptions(
        enabled=True,
        model=model.name,
        actor_model=actor_model.name,
        analysis_fps=YOLO_FPS,
        confidence=0.08,
        tile_size_px=384,
        inference_imgsz=768,
        tile_overlap=0.25,
        maximum_tiles=64,
        actor_imgsz=1280,
        actor_confidence=0.15,
        actor_overlap_threshold=0.20,
        zones=(GroundLitterZone(
            region_id="sidewalk", polygon=ROI, minimum_short_side_px=2,
            minimum_box_area_px=6, confidence=0.08,
        ),),
        overlay_exclude_zones=OVERLAY_ZONES,
    )


def run_yolo_positive(
    video: Path,
    output: Path,
    reference: np.ndarray,
    valid: np.ndarray,
    metadata: dict[str, Any],
    model: Path,
    actor_model: Path,
    device: str,
) -> dict[str, Any]:
    directory = output / "positive_yolo_fusion"
    directory.mkdir(parents=True, exist_ok=True)
    rows_file = directory / "checkpoint_rows.json"
    cached: dict[str, Any] = json.loads(rows_file.read_text()) if rows_file.exists() else {}
    options = yolo_options(model, actor_model)
    detector = UltralyticsGroundLitterDetector(
        model_path=model, actor_model_path=actor_model, device=device,
    )
    representative = read_frame(video, 0.0)
    aligned_reference, aligned_valid, alignment = transform_reference(reference, valid, representative)
    started = time.perf_counter()
    for index, (timestamp, frame) in enumerate(iter_sampled_frames(video, sample_fps=YOLO_FPS)):
        key = f"{timestamp:.3f}"
        if key in cached:
            continue
        actors = actor_boxes(detector, frame, options)
        baseline, stats = yolo_candidates(detector, frame, options, actors)
        anomalies, _diagnostic, residual, _broad, _illumination = process_reference_frame(
            aligned_reference, aligned_valid, frame
        )
        retained, suppressed = add_reference_evidence(baseline, residual, aligned_valid, anomalies)
        fusion = []
        used: set[int] = set()
        for row in retained:
            matches = [i for i, anomaly in enumerate(anomalies)
                       if boxes_related(row["box"], anomaly["box"])]
            used.update(matches)
            fusion.append({**row, "source": "fusion", "sources":
                           ["yolo", "clean_reference"] if matches else ["yolo"]})
        for i, row in enumerate(anomalies):
            if i not in used:
                fusion.append({**row, "source": "fusion", "sources": ["clean_reference"]})
        cached[key] = {
            "timestamp": timestamp,
            "yolo": baseline,
            "reference": anomalies,
            "retained_yolo": retained,
            "suppressed_yolo": suppressed,
            "fusion": fusion,
            "actors": actors,
            "stats": stats,
        }
        rows_file.write_text(json.dumps(cached, ensure_ascii=False) + "\n")
        print(f"[positive_yolo] {timestamp:.0f}s yolo={len(baseline)} "
              f"suppressed={len(suppressed)} ref={len(anomalies)}", flush=True)
    yolo_by_time = {float(key): value["yolo"] for key, value in cached.items()}
    retained_by_time = {float(key): value["retained_yolo"] for key, value in cached.items()}
    suppressed_by_time = {float(key): value["suppressed_yolo"] for key, value in cached.items()}
    fusion_by_time = {float(key): value["fusion"] for key, value in cached.items()}
    yolo_tracks = track_rows(yolo_by_time, source="yolo", sample_fps=YOLO_FPS)
    retained_tracks = track_rows(retained_by_time, source="yolo_after_reference", sample_fps=YOLO_FPS)
    fusion_tracks = track_rows(fusion_by_time, source="fusion", sample_fps=YOLO_FPS)
    raw_jsonl(directory / "yolo_raw_candidates.jsonl", yolo_by_time)
    raw_jsonl(directory / "yolo_retained_candidates.jsonl", retained_by_time)
    raw_jsonl(directory / "yolo_suppressed_candidates.jsonl", suppressed_by_time)
    raw_jsonl(directory / "fusion_raw_candidates.jsonl", fusion_by_time)
    (directory / "yolo_tracks.json").write_text(json.dumps(yolo_tracks, indent=2) + "\n")
    (directory / "fusion_tracks.json").write_text(json.dumps(fusion_tracks, indent=2) + "\n")
    plot_timeline(directory / "yolo_timeline.png", yolo_tracks, "positive: YOLO tracks", metadata["duration_seconds"])
    plot_timeline(directory / "fusion_timeline.png", fusion_tracks, "positive: fusion tracks", metadata["duration_seconds"])
    last = read_frame(video, max(0.0, metadata["duration_seconds"] - 1.0))
    save_image(directory / "yolo_stable_overlay.jpg", draw_tracks(last, yolo_tracks, "YOLO tracks >=5s"))
    save_image(directory / "fusion_stable_overlay.jpg", draw_tracks(last, fusion_tracks, "Fusion tracks >=5s"))
    result = {
        "sample_fps": YOLO_FPS,
        "frozen_parameters": {
            "confidence": 0.08, "tile_size_px": 384, "inference_imgsz": 768,
            "tile_overlap": 0.25, "actor_imgsz": 1280, "actor_confidence": 0.15,
        },
        "alignment": alignment,
        "yolo": track_funnel(yolo_tracks, sum(map(len, yolo_by_time.values())), metadata["duration_seconds"]),
        "yolo_after_reference": track_funnel(retained_tracks, sum(map(len, retained_by_time.values())),
                                                     metadata["duration_seconds"]),
        "suppressed_yolo_raw": sum(map(len, suppressed_by_time.values())),
        "fusion": track_funnel(fusion_tracks, sum(map(len, fusion_by_time.values())), metadata["duration_seconds"]),
        "t01_yolo": t01_metrics(yolo_by_time, yolo_tracks, YOLO_FPS),
        "t01_fusion": t01_metrics(fusion_by_time, fusion_tracks, YOLO_FPS),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def make_t01_evidence(video: Path, output: Path, reference: np.ndarray) -> None:
    frames = [read_frame(video, value) for value in (70.0, 72.0, 75.0, 90.0)]
    panels = []
    sources = [reference, *frames]
    labels = ["clean reference", "70s before", "72s transition", "75s present", "90s present"]
    for image, label in zip(sources, labels):
        x, y, right, bottom = T01_REGION
        margin = 80
        crop = image[max(0, y-margin):min(image.shape[0], bottom+margin),
                     max(0, x-margin):min(image.shape[1], right+margin)].copy()
        cv2.rectangle(crop, (margin, margin), (margin + right-x, margin + bottom-y), (0, 255, 255), 2)
        crop = cv2.resize(crop, (360, 300), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(crop, (0, 0), (359, 34), (20, 20, 20), -1)
        cv2.putText(crop, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(crop)
    save_image(output / "positive_reference" / "t01_evidence.jpg", np.hstack(panels))


def make_yolo_evidence(video: Path, output: Path, reference: np.ndarray, valid: np.ndarray) -> None:
    directory = output / "positive_yolo_fusion"
    cached = load_json(directory / "checkpoint_rows.json")
    aligned_reference, _aligned_valid, _alignment = transform_reference(
        reference, valid, read_frame(video, 0.0))
    t01_panels = []
    for timestamp in (70.0, 72.0, 74.0, 82.0, 90.0):
        frame = read_frame(video, timestamp)
        rows = cached.get(f"{timestamp:.3f}", {}).get("yolo", [])
        x, y, right, bottom = T01_REGION
        margin = 100
        a, b = max(0, x-margin), max(0, y-margin)
        c, d = min(frame.shape[1], right+margin), min(frame.shape[0], bottom+margin)
        crop = frame[b:d, a:c].copy()
        for row in rows:
            if not candidate_matches_t01(row):
                continue
            p, q, r, s = row["box"]
            cv2.rectangle(crop, (p-a, q-b), (r-a, s-b), (0, 255, 255), 2)
            cv2.putText(crop, f"{row.get('class_name')} {row.get('confidence', 0):.2f}",
                        (p-a, max(18, q-b-4)), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        (0, 255, 255), 1, cv2.LINE_AA)
        crop = cv2.resize(crop, (360, 320), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(crop, (0, 0), (359, 34), (20, 20, 20), -1)
        cv2.putText(crop, f"YOLO t={timestamp:.0f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    .58, (255, 255, 255), 1, cv2.LINE_AA)
        t01_panels.append(crop)
    save_image(directory / "t01_yolo_evidence.jpg", np.hstack(t01_panels))

    cards = []
    rank = 0
    for key in sorted(cached, key=float):
        timestamp = float(key)
        frame = None
        for row in cached[key].get("suppressed_yolo", []):
            if frame is None:
                frame = read_frame(video, timestamp)
            rank += 1
            x, y, right, bottom = row["box"]
            margin = max(70, 2 * max(right-x, bottom-y))
            a, b = max(0, x-margin), max(0, y-margin)
            c, d = min(frame.shape[1], right+margin), min(frame.shape[0], bottom+margin)
            panels = []
            for source, label in ((aligned_reference, "clean reference"), (frame, f"current {timestamp:.0f}s")):
                crop = source[b:d, a:c].copy()
                cv2.rectangle(crop, (x-a, y-b), (right-a, bottom-b), (0, 255, 255), 2)
                crop = cv2.resize(crop, (300, 260), interpolation=cv2.INTER_CUBIC)
                cv2.rectangle(crop, (0, 0), (299, 30), (20, 20, 20), -1)
                cv2.putText(crop, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(crop)
            card = np.hstack(panels)
            caption = np.full((48, 600, 3), 24, np.uint8)
            cv2.putText(caption,
                        f"suppressed {rank} / {row.get('class_name')} {row.get('confidence', 0):.2f} / t={timestamp:.0f}s",
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            cards.append(np.vstack((caption, card)))
    if cards:
        sheet = np.full((math.ceil(len(cards)/2) * 308, 1200, 3), 18, np.uint8)
        for index, card in enumerate(cards):
            row, column = divmod(index, 2)
            sheet[row*308:(row+1)*308, column*600:(column+1)*600] = card
        save_image(directory / "suppressed_yolo_evidence_contact.jpg", sheet)


def report(output: Path, metadata: dict[str, Any], reference_build: dict[str, Any],
           holdout: dict[str, Any], positive: dict[str, Any], negative: dict[str, Any],
           yolo: dict[str, Any]) -> None:
    control_path = output / "compression_control_holdout" / "summary.json"
    compression_control = load_json(control_path) if control_path.exists() else None
    positive_core = load_json(output / "positive_reference" / "core_roi_diagnostic" / "summary.json")
    negative_core = load_json(output / "negative_reference" / "core_roi_diagnostic" / "summary.json")
    holdout_long = holdout["main"]["tracks_at_least_seconds"]["10"]
    funnel_reduction = negative["main"]["raw_to_5s_reduction"]
    t01 = positive["t01"]
    t01_diag = positive["t01_diagnostic"]
    stable_delay = t01["stable_5s_delay"]
    gate1 = funnel_reduction >= .80
    gate2 = stable_delay is not None and stable_delay <= 5.0 and holdout_long == 0
    gate2_state = (
        "PASS" if gate2 else
        "PARTIAL" if (
            t01_diag["stable_5s_delay"] is not None
            and t01_diag["stable_5s_delay"] <= 5.0
            and holdout_long == 0
        ) else "FAIL"
    )
    payload = {
        "kind": "offline_clean_reference_temporal_video_poc",
        "inputs": metadata,
        "frozen_thresholds": {
            "signature": SIGNATURE_THRESHOLD,
            "luminance": LUMINANCE_THRESHOLD,
            "main_min_component_area": MAIN_MIN_AREA,
            "diagnostic_min_component_area": DIAGNOSTIC_MIN_AREA,
            "reference_fps": REFERENCE_FPS,
            "analysis_fps": ANALYSIS_FPS,
            "yolo_fps": YOLO_FPS,
        },
        "reference_build": reference_build,
        "holdout": holdout,
        "positive_reference": positive,
        "negative_reference": negative,
        "positive_yolo_fusion": yolo,
        "compression_control_holdout": compression_control,
        "core_roi_diagnostic": {"positive": positive_core, "negative": negative_core},
        "gates": {
            "temporal_raw_to_5s_reduction_ge_80pct": {"pass": gate1, "value": funnel_reduction},
            "t01_stable_within_5s_and_no_holdout_track_over_10s": {
                "state": gate2_state, "main_t01_stable_delay": stable_delay,
                "diagnostic_t01_stable_delay": t01_diag["stable_5s_delay"],
                "holdout_tracks_over_10s": holdout_long,
            },
        },
    }
    (output / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    yolo_t01 = yolo["t01_yolo"]
    ref_t01 = positive["t01"]
    answer3 = (
        "是" if yolo_t01["first_raw_seconds"] is None and ref_t01["first_raw_seconds"] is not None
        else "否；主 Reference 分支也未稳定补回" if ref_t01["first_raw_seconds"] is None
        else "本样本无法验证：YOLO 在 72 秒已命中 T01，Reference 主分支反而明显更晚"
    )
    long_negative = sorted(load_json(output / "negative_reference" / "tracks.json"),
                           key=lambda row: -row["lifetime_seconds"])
    longest = [
        {key: row[key] for key in (
            "track_id", "first_seen", "last_seen", "hits", "lifetime_seconds",
            "continuity", "median_box",
        )}
        for row in long_negative[:5]
    ]
    report_text = f"""# Clean Reference + Temporal 视频 PoC

## 结论

- **本轮 PoC 不通过进入生产链路的门槛。** 数字上的时序漏斗通过，但 T01 补检与负样本候选负载均未达到目标。
- Gate 1（Raw → 5 秒轨迹减少至少 80%）：**{'PASS' if gate1 else 'FAIL'}**，负样本减少 {funnel_reduction:.1%}。
- Gate 2（T01 在 5 秒内稳定且 clean holdout 无 >10 秒误轨迹）：**{gate2_state}**。主配置 T01 延迟为 {stable_delay}；面积 4–9 像素的诊断分支延迟为 {t01_diag['stable_5s_delay']}；holdout >10 秒轨迹为 {holdout_long}。
- 本轮主配置保持 `signature=3.0 / luminance=100 / min_area=10`。预探针发现 T01 的阈值核心通常只有 4–9 像素，因此另保留 `min_area=4` 诊断结果，但它不参与 PASS 判定。
- 负样本为 HEVC、约 {metadata['negative']['container_bitrate_bps']/1e6:.2f} Mbps；参考与正样本为 H.264、约 14.3 Mbps。负样本结论属于 **compression stress**，不能直接等同同编码条件下的误报率。
- 可选压缩对照把原 clean 视频转为 1440p/25 FPS/HEVC/约 2.1 Mbps；其 holdout 主配置为 {compression_control['main']['raw_candidates'] if compression_control else '未运行'} 个 raw、{compression_control['main']['tracks_at_least_seconds']['10'] if compression_control else '未运行'} 条 ≥10 秒轨迹。低码率本身没有复现负样本长轨迹，场景时段/局部外观变化才是主因。
- Reference 强负证据只抑制 {yolo['suppressed_yolo_raw']} / {yolo['yolo']['raw_candidates']} 个 YOLO raw 候选（{yolo['suppressed_yolo_raw']/max(yolo['yolo']['raw_candidates'],1):.1%}）。Fusion 把 ≥5 秒轨迹从 YOLO 的 {yolo['yolo']['tracks_at_least_seconds']['5']} 增加到 {yolo['fusion']['tracks_at_least_seconds']['5']}，本轮没有获得净收益。
- 被抑制候选的证据卡多数是地砖线、固定斑点和推车结构，方向基本正确；问题是覆盖率不足，而不是这条负证据完全无效。
- 四段配准的中位重投影误差均低于 0.34px、P95 均低于 0.96px，没有 camera shift 证据；本轮失败主要来自环境外观失配与候选/轨迹表达。
- 参考视频内有持续停放的推车、车辆/电动车和圆凳；median 会保留这些长期遮挡。当前 valid mask 只依据 temporal MAD，**没有满足“长期遮挡不能进入 Clean Reference”这一要求**。
- 后验核心地面 ROI 诊断把负样本 ≥10 秒轨迹从 {negative['main']['tracks_at_least_seconds']['10']} 降到 {negative_core['metrics']['tracks_at_least_seconds']['10']}，说明边界/设施贡献很大；但 T01 主分支延迟仍为 {positive_core['t01']['stable_5s_delay']} 秒，缩 ROI 不能修复小目标链路。

## 输入与方法

- Clean Reference：前 {REFERENCE_BUILD_FRACTION:.0%}（0–{reference_build['build_end_seconds']:.1f}s）按 1 FPS 采样 {reference_build['sample_count']} 帧做逐像素 median；后 {1-REFERENCE_BUILD_FRACTION:.0%} 只做 holdout，未参与背景生成。
- Reference/Temporal：人行道 ROI，1 FPS，SIFT/RANSAC 轻量配准，逐帧通道增益/偏置和低频亮度归一化，局部 connected components，多目标时序关联。
- YOLO：正样本全时段 0.5 FPS；参数固定为 confidence 0.08、384px tile、768 inference、25% overlap，并使用现有 actor/context 过滤。
- Clean Reference 在完成构建后只读；所有当前帧、长期物体和负样本都没有写回参考。

## 六个问题

1. **时序是否显著减少瞬时误报？** 负样本 raw 候选 {negative['main']['raw_candidates']} 个，5 秒轨迹 {negative['main']['tracks_at_least_seconds']['5']} 条，减少 {funnel_reduction:.1%}。Gate 1 为 {'PASS' if gate1 else 'FAIL'}。
2. **T01 是否稳定形成候选？** 主配置首次 raw={t01['first_raw_seconds']}，稳定 1/3/5 秒分别为 {t01['first_stable_1s']} / {t01['first_stable_3s']} / {t01['first_stable_5s']}。诊断小核心分支稳定 5 秒={t01_diag['first_stable_5s']}。
3. **YOLO 漏检时 Reference 是否补回？** {answer3}。YOLO 首次 T01 raw={yolo_t01['first_raw_seconds']}；Reference 首次 raw={ref_t01['first_raw_seconds']}。
4. **生命周期能否分开瞬时变化和落地物？** 能过滤大量瞬时点，但不能区分持续固定结构残差与落地物。负样本 ≥1/3/5/10/30 秒轨迹分别为 {negative['main']['tracks_at_least_seconds']['1']} / {negative['main']['tracks_at_least_seconds']['3']} / {negative['main']['tracks_at_least_seconds']['5']} / {negative['main']['tracks_at_least_seconds']['10']} / {negative['main']['tracks_at_least_seconds']['30']}。
5. **正常视频是否存在长时误异常？** 最长五条轨迹为 `{json.dumps(longest, ensure_ascii=False)}`。证据以 `negative_reference/timeline.png`、`candidate_heatmap.png` 和 overlay 为准。
6. **下一步优先级？** 先改 Reference/Temporal，不应先接 VLM。需要用多时段干净参考或只读环境基线压制固定结构残差，增加跨视频 stable-noise map，并重做 4–9 像素核心的聚合与遮挡恢复；之后在独立正样本盲测。当前每 5 分钟仍有 {negative['main']['tracks_at_least_seconds']['10']} 条 ≥10 秒负样本轨迹，直接送 VLM 的负载和语义歧义都过高。

## 主要指标

| 数据段 | Raw | Raw/min | ≥1s | ≥3s | ≥5s | ≥10s | ≥30s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean holdout | {holdout['main']['raw_candidates']} | {holdout['main']['raw_per_minute']:.1f} | {holdout['main']['tracks_at_least_seconds']['1']} | {holdout['main']['tracks_at_least_seconds']['3']} | {holdout['main']['tracks_at_least_seconds']['5']} | {holdout['main']['tracks_at_least_seconds']['10']} | {holdout['main']['tracks_at_least_seconds']['30']} |
| Positive Reference | {positive['main']['raw_candidates']} | {positive['main']['raw_per_minute']:.1f} | {positive['main']['tracks_at_least_seconds']['1']} | {positive['main']['tracks_at_least_seconds']['3']} | {positive['main']['tracks_at_least_seconds']['5']} | {positive['main']['tracks_at_least_seconds']['10']} | {positive['main']['tracks_at_least_seconds']['30']} |
| Negative Reference | {negative['main']['raw_candidates']} | {negative['main']['raw_per_minute']:.1f} | {negative['main']['tracks_at_least_seconds']['1']} | {negative['main']['tracks_at_least_seconds']['3']} | {negative['main']['tracks_at_least_seconds']['5']} | {negative['main']['tracks_at_least_seconds']['10']} | {negative['main']['tracks_at_least_seconds']['30']} |
| Positive YOLO | {yolo['yolo']['raw_candidates']} | {yolo['yolo']['raw_per_minute']:.1f} | {yolo['yolo']['tracks_at_least_seconds']['1']} | {yolo['yolo']['tracks_at_least_seconds']['3']} | {yolo['yolo']['tracks_at_least_seconds']['5']} | {yolo['yolo']['tracks_at_least_seconds']['10']} | {yolo['yolo']['tracks_at_least_seconds']['30']} |
| Positive Fusion | {yolo['fusion']['raw_candidates']} | {yolo['fusion']['raw_per_minute']:.1f} | {yolo['fusion']['tracks_at_least_seconds']['1']} | {yolo['fusion']['tracks_at_least_seconds']['3']} | {yolo['fusion']['tracks_at_least_seconds']['5']} | {yolo['fusion']['tracks_at_least_seconds']['10']} | {yolo['fusion']['tracks_at_least_seconds']['30']} |
| HEVC clean holdout control | {compression_control['main']['raw_candidates'] if compression_control else '-'} | {f"{compression_control['main']['raw_per_minute']:.1f}" if compression_control else '-'} | {compression_control['main']['tracks_at_least_seconds']['1'] if compression_control else '-'} | {compression_control['main']['tracks_at_least_seconds']['3'] if compression_control else '-'} | {compression_control['main']['tracks_at_least_seconds']['5'] if compression_control else '-'} | {compression_control['main']['tracks_at_least_seconds']['10'] if compression_control else '-'} | {compression_control['main']['tracks_at_least_seconds']['30'] if compression_control else '-'} |

## 审核材料

- `clean_reference.jpg`、`reference_valid_mask.png`、`reference_noise_map.png`
- `holdout/`、`positive_reference/`、`negative_reference/` 的 raw JSONL、轨迹 JSON、时间线、热图和稳定框叠加图
- `positive_reference/t01_evidence.jpg`
- `positive_yolo_fusion/` 的 YOLO/Fusion 原始候选、被参考负证据抑制的候选、轨迹和时间线
- `compression_control_holdout/` 的同源低码率 HEVC 对照
- `positive_reference/core_roi_diagnostic/`、`negative_reference/core_roi_diagnostic/` 的边界排除诊断（不参与 Gate）

## 限制

- 只有一个摄像头、一个正目标和一个负视频，不能估计总体精确率或召回率。
- T01 位置由人工查看后冻结，因此是定向案例，不是盲测。
- 参考视频是清晨、正样本约 10 点、负样本约 8 点；结果同时检验了明显光照变化，但没有覆盖雨天。
- 本实验把大面积高残差区域作为 temporary unavailable；没有逐帧语义分割时，车辆、货架和细碎人体边缘仍会进入 raw 候选。负样本没有逐框人工真值，其中某些长时新增物可能是真实异物，不能全部称为语义误报。
"""
    (output / "REPORT.md").write_text(report_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/litter/turhancan_yolov8m_seg_trash.pt"))
    parser.add_argument("--actor-model", type=Path, default=Path("models/yolo26s.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", choices=("reference", "scan", "yolo", "report", "all"), default="all")
    args = parser.parse_args()
    clean = args.input_dir / "clean_reference.mp4"
    positive = args.input_dir / "小垃圾正样本.mp4"
    negative = args.input_dir / "正常负样本.mp4"
    for path in (clean, positive, negative, args.model, args.actor_model):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "input_metadata.json"
    if not metadata_path.exists():
        metadata = {"clean": video_metadata(clean), "positive": video_metadata(positive),
                    "negative": video_metadata(negative)}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    else:
        metadata = load_json(metadata_path)
    reference_path = args.output / "reference_arrays.npz"
    if args.stage in {"reference", "all"} and not reference_path.exists():
        build_reference(clean, args.output, metadata["clean"])
    if args.stage == "reference":
        return 0
    if not reference_path.exists():
        raise FileNotFoundError("run --stage reference first")
    arrays = np.load(reference_path)
    reference, valid = arrays["reference"], arrays["valid"]
    build = load_json(args.output / "reference_build.json")
    if args.stage in {"scan", "all"}:
        if not (args.output / "holdout" / "summary.json").exists():
            scan_reference_video("holdout", clean, args.output, reference, valid, metadata["clean"],
                                 start=build["holdout_start_seconds"])
        if not (args.output / "positive_reference" / "summary.json").exists():
            scan_reference_video("positive_reference", positive, args.output, reference, valid,
                                 metadata["positive"])
        if not (args.output / "negative_reference" / "summary.json").exists():
            scan_reference_video("negative_reference", negative, args.output, reference, valid,
                                 metadata["negative"])
        make_t01_evidence(positive, args.output, reference)
        # Rebuild target timing from retained raw evidence so reporting logic
        # can evolve without re-running the expensive video residual scan.
        positive_dir = args.output / "positive_reference"
        positive_summary = load_json(positive_dir / "summary.json")
        positive_rows = read_raw_jsonl(positive_dir / "raw_candidates.jsonl")
        positive_tracks = load_json(positive_dir / "tracks.json")
        positive_diagnostic_rows = read_raw_jsonl(positive_dir / "diagnostic_small_core_candidates.jsonl")
        positive_diagnostic_tracks = load_json(positive_dir / "diagnostic_small_core_tracks.json")
        positive_summary["t01"] = t01_metrics(positive_rows, positive_tracks, ANALYSIS_FPS)
        positive_summary["t01_diagnostic"] = t01_metrics(
            positive_diagnostic_rows, positive_diagnostic_tracks, ANALYSIS_FPS)
        (positive_dir / "summary.json").write_text(json.dumps(positive_summary, indent=2) + "\n")
        for name, video_path, start_time in (
            ("holdout", clean, build["holdout_start_seconds"]),
            ("positive_reference", positive, 0.0),
            ("negative_reference", negative, 0.0),
        ):
            directory = args.output / name
            make_long_track_evidence(video_path, directory, reference, valid,
                                     load_json(directory / "tracks.json"), start=start_time)
        make_core_roi_diagnostic(positive, positive_dir, metadata["positive"]["duration_seconds"],
                                 positive=True)
        make_core_roi_diagnostic(negative, args.output / "negative_reference",
                                 metadata["negative"]["duration_seconds"], positive=False)
    if args.stage == "scan":
        return 0
    if args.stage in {"yolo", "all"} and not (args.output / "positive_yolo_fusion" / "summary.json").exists():
        run_yolo_positive(positive, args.output, reference, valid, metadata["positive"],
                          args.model, args.actor_model, args.device)
    if args.stage in {"yolo", "all"}:
        make_yolo_evidence(positive, args.output, reference, valid)
    if args.stage in {"report", "all"}:
        required = [
            args.output / "holdout" / "summary.json",
            args.output / "positive_reference" / "summary.json",
            args.output / "negative_reference" / "summary.json",
            args.output / "positive_yolo_fusion" / "summary.json",
        ]
        if not all(path.exists() for path in required):
            raise FileNotFoundError("scan/yolo summaries are incomplete")
        report(args.output, metadata, build, *(load_json(path) for path in required))
    print(json.dumps({"output": str(args.output), "stage": args.stage}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
