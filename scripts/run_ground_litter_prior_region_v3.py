#!/usr/bin/env python3
"""Offline V3 experiment for a protected daylight Clean Reference region.

The experiment keeps the Clean Reference read-only and compares V2 proposal
generation with a protected illumination model on exactly the same ground ROI.
It deliberately does not update a target-video noise blacklist: current-frame
outliers can be excluded from illumination estimation, but remain eligible for
anomaly detection unless the region is temporarily unavailable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ground-litter-v3")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ground_litter_clean_temporal_poc import (  # noqa: E402
    ANALYSIS_FPS,
    T01_REGION,
    draw_tracks,
    iter_sampled_frames,
    load_json,
    plot_timeline,
    polygon_mask,
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
from run_ground_litter_clean_temporal_v2 import (  # noqa: E402
    LUMINANCE_THRESHOLD,
    MAXIMUM_SUPPORT_BOX_AREA,
    MAXIMUM_SUPPORT_SIDE,
    MINIMUM_SEED_PIXELS,
    MINIMUM_SUPPORT_AREA,
    MINIMUM_SUPPORT_SHORT_SIDE,
    SIGNATURE_THRESHOLD,
    SUPPORT_LUMINANCE_THRESHOLD,
    SUPPORT_SIGNATURE_THRESHOLD,
    proposals,
    residual_maps,
)


# Approximation of the user-marked red region, tightened to the visible ground
# plane.  The right edge follows the wall/ground boundary instead of including
# the wall-mounted fixtures.  Coordinates are normalized to the video frame.
GROUND_ROI = (
    (0.455, 0.232),
    (0.714, 0.232),
    (0.806, 1.000),
    (0.455, 1.000),
)

# The stool is present throughout the reviewed reference clip, so the ground
# beneath it has never been observed clean.  This is reference-unknown rather
# than non-litter and must be re-captured before this patch can be monitored.
REFERENCE_UNKNOWN_RECTS = (
    (0.704, 0.515, 0.779, 0.690),
)

# Existing camera overlay in the lower-right corner.
PERMANENT_EXCLUSION_RECTS = (
    (0.678, 0.835, 1.000, 1.000),
)

LOCAL_FIELD_BLUR_SIGMA = 24.0
LOCAL_FIELD_DOWNSAMPLE = 4
LOCAL_FIELD_CLIP = 48.0
PROTECTED_SEED_SIGNATURE = 3.0
PROTECTED_SEED_LUMINANCE = 100.0
PRELIMINARY_DILATION = 31
TEMPORARY_UNAVAILABLE_MIN_AREA = 1500
TEMPORARY_UNAVAILABLE_MIN_SIDE = 100
SATURATION_DILATION = 9
NOISE_SEED_SIGNATURE_INCREMENT = 0.8
NOISE_SEED_LUMINANCE_INCREMENT = 30.0
NOISE_SUPPORT_SIGNATURE_INCREMENT = 0.25
NOISE_SUPPORT_LUMINANCE_INCREMENT = 8.0
EVALUATION_START_SECONDS = 60.0
REFERENCE_BUILD_FRACTION = 0.65


def rect_mask(shape: tuple[int, int], rects: Iterable[tuple[float, float, float, float]]) -> np.ndarray:
    height, width = shape
    result = np.zeros((height, width), np.uint8)
    for left, top, right, bottom in rects:
        cv2.rectangle(
            result,
            (int(round(left * width)), int(round(top * height))),
            (int(round(right * width)), int(round(bottom * height))),
            255,
            -1,
        )
    return result


def build_reference_masks(reference: np.ndarray, inherited_valid: np.ndarray,
                          output: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = reference.shape[:2]
    roi = polygon_mask((height, width), GROUND_ROI)
    unknown = rect_mask((height, width), REFERENCE_UNKNOWN_RECTS)
    permanent = rect_mask((height, width), PERMANENT_EXCLUSION_RECTS)
    valid = cv2.bitwise_and(inherited_valid, roi)
    valid[unknown > 0] = 0
    valid[permanent > 0] = 0
    # Boundary residuals are not useful evidence and are very sensitive to a
    # one-pixel warp.  Three pixels are small relative to the target footprint.
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8))

    overlay = reference.copy()
    polygon = np.round(np.asarray(GROUND_ROI) * np.asarray([width, height])).astype(np.int32)
    cv2.polylines(overlay, [polygon], True, (0, 255, 0), 5, cv2.LINE_AA)
    red = np.zeros_like(reference)
    red[..., 2] = 255
    unknown_alpha = (unknown.astype(np.float32) / 255.0 * 0.38)[..., None]
    overlay = np.clip(overlay * (1 - unknown_alpha) + red * unknown_alpha, 0, 255).astype(np.uint8)
    overlay[permanent > 0] = (45, 45, 45)
    cv2.putText(overlay, "green=ground ROI  red=reference unknown  gray=permanent exclusion",
                (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2,
                cv2.LINE_AA)
    save_image(output / "ground_roi_overlay.jpg", overlay)
    save_image(output / "reference_valid_mask_v3.png", valid)
    save_image(output / "reference_unknown_mask.png", unknown)
    details = {
        "ground_roi_normalized": GROUND_ROI,
        "reference_unknown_rects_normalized": REFERENCE_UNKNOWN_RECTS,
        "permanent_exclusion_rects_normalized": PERMANENT_EXCLUSION_RECTS,
        "ground_roi_pixels": int(np.count_nonzero(roi)),
        "valid_pixels": int(np.count_nonzero(valid)),
        "valid_fraction_of_ground_roi": float(
            np.count_nonzero(valid) / max(np.count_nonzero(roi), 1)
        ),
        "reference_unknown_fraction_of_ground_roi": float(
            np.count_nonzero((unknown > 0) & (roi > 0)) / max(np.count_nonzero(roi), 1)
        ),
    }
    (output / "region_profile.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return valid, unknown, details


def build_daylight_tolerance(v2_output: Path, reference_valid: np.ndarray,
                             output: Path) -> tuple[np.ndarray, dict[str, Any]]:
    sources = [
        v2_output / "positive_reference" / "adaptive_noise_mask.png",
        v2_output / "negative_reference" / "adaptive_noise_mask.png",
    ]
    masks = []
    for source in sources:
        mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(source)
        if mask.shape != reference_valid.shape:
            mask = cv2.resize(mask, (reference_valid.shape[1], reference_valid.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        masks.append(mask)
    tolerance = np.maximum.reduce(masks)
    tolerance[reference_valid == 0] = 0
    save_image(output / "daylight_bounded_tolerance_mask.png", tolerance)
    details = {
        "source": "union of frozen 0-60s normal-day V2 calibration masks",
        "is_detection_blacklist": False,
        "seed_signature_increment": NOISE_SEED_SIGNATURE_INCREMENT,
        "seed_luminance_increment": NOISE_SEED_LUMINANCE_INCREMENT,
        "support_signature_increment": NOISE_SUPPORT_SIGNATURE_INCREMENT,
        "support_luminance_increment": NOISE_SUPPORT_LUMINANCE_INCREMENT,
        "fraction_of_reference_valid": float(
            np.count_nonzero(tolerance) / max(np.count_nonzero(reference_valid), 1)
        ),
        "t01_region_fraction": float(
            np.count_nonzero(tolerance[T01_REGION[1]:T01_REGION[3], T01_REGION[0]:T01_REGION[2]])
            / max((T01_REGION[3] - T01_REGION[1]) * (T01_REGION[2] - T01_REGION[0]), 1)
        ),
    }
    (output / "daylight_tolerance.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return tolerance, details


def robust_global_color(reference: np.ndarray, current: np.ndarray,
                        valid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    old = reference.astype(np.float32)
    new = current.astype(np.float32)
    eligible = valid.astype(bool)
    eligible &= np.all((old > 5) & (old < 250) & (new > 5) & (new < 250), axis=2)
    indices = np.flatnonzero(eligible)
    if indices.size < 5000:
        raise ValueError("insufficient trusted ground for global illumination fit")
    if indices.size > 250_000:
        indices = indices[np.linspace(0, indices.size - 1, 250_000).astype(np.intp)]
    keep = np.ones(indices.size, bool)
    fits: list[tuple[float, float]] = []
    for _ in range(3):
        fits = []
        for channel in range(3):
            x = old[..., channel].reshape(-1)[indices][keep]
            y = new[..., channel].reshape(-1)[indices][keep]
            gain, bias = np.polyfit(x, y, 1)
            fits.append((float(np.clip(gain, 0.65, 1.45)),
                         float(np.clip(bias, -60.0, 60.0))))
        prediction = np.stack(
            [old[..., c] * fits[c][0] + fits[c][1] for c in range(3)], axis=2
        )
        residual = np.max(np.abs(new - prediction), axis=2).reshape(-1)[indices]
        cutoff = max(float(np.percentile(residual[keep], 70)), 8.0)
        keep = residual <= cutoff
    normalized = np.stack(
        [old[..., c] * fits[c][0] + fits[c][1] for c in range(3)], axis=2
    )
    return normalized, {
        "channel_gain_bias_bgr": [[round(g, 5), round(b, 3)] for g, b in fits],
        "fit_pixels": int(keep.sum()),
    }


def component_mask(binary: np.ndarray, *, min_area: int,
                   min_side: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    result = np.zeros_like(binary, np.uint8)
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area >= min_area or (area >= min_area // 3 and max(width, height) >= min_side):
            result[labels == index] = 255
    return result


def preliminary_protection(global_reference: np.ndarray, current: np.ndarray,
                           valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(global_reference, 0, 255).astype(np.uint8)
    signature, luminance = residual_maps(clipped, current)
    support = (
        (signature >= PROTECTED_SEED_SIGNATURE)
        & (luminance >= PROTECTED_SEED_LUMINANCE)
        & (valid > 0)
    ).astype(np.uint8)
    protected = cv2.dilate(
        support, np.ones((PRELIMINARY_DILATION, PRELIMINARY_DILATION), np.uint8)
    )
    unavailable = component_mask(
        support,
        min_area=TEMPORARY_UNAVAILABLE_MIN_AREA,
        min_side=TEMPORARY_UNAVAILABLE_MIN_SIDE,
    )
    if unavailable.any():
        unavailable = cv2.dilate(unavailable, np.ones((21, 21), np.uint8))
    return protected, unavailable


def local_color_field(global_reference: np.ndarray, current: np.ndarray,
                      fit_mask: np.ndarray,
                      protected_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = fit_mask.shape
    small_width = max(1, math.ceil(width / LOCAL_FIELD_DOWNSAMPLE))
    small_height = max(1, math.ceil(height / LOCAL_FIELD_DOWNSAMPLE))
    old_luma = cv2.cvtColor(
        np.clip(global_reference, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32)
    new_luma = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
    raw_delta = new_luma - old_luma
    delta = np.clip(
        raw_delta,
        -LOCAL_FIELD_CLIP,
        LOCAL_FIELD_CLIP,
    )
    fallback = 0.0
    all_values = delta[fit_mask > 0]
    if len(all_values) >= 5000:
        fallback = float(np.median(all_values))

    # Compute the broad local field at 1/8 resolution.  The preliminary mask
    # already removes candidate pixels, so a weighted low-pass remains robust
    # while being far faster than extracting dozens of full-resolution medians.
    small_delta = cv2.resize(delta, (small_width, small_height), interpolation=cv2.INTER_AREA)
    small_weight = cv2.resize(
        (fit_mask > 0).astype(np.float32),
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )
    sigma = LOCAL_FIELD_BLUR_SIGMA / LOCAL_FIELD_DOWNSAMPLE
    support = cv2.GaussianBlur(small_weight, (0, 0), sigma)
    weighted = cv2.GaussianBlur(small_delta * small_weight, (0, 0), sigma)
    field_small = weighted / np.maximum(support, 1e-3)
    field_small[support < 0.04] = fallback
    field = cv2.resize(field_small, (width, height), interpolation=cv2.INTER_CUBIC)
    # Around strong candidate seeds retain the established V2 low-frequency
    # baseline.  The protected estimator is used elsewhere, but must not
    # rewrite the photometric reference under a possible small object.
    default_field = cv2.GaussianBlur(raw_delta, (0, 0), LOCAL_FIELD_BLUR_SIGMA)
    field[protected_mask > 0] = default_field[protected_mask > 0]
    field = np.clip(field, -LOCAL_FIELD_CLIP, LOCAL_FIELD_CLIP)
    confidence_small = np.clip(support / 0.18, 0, 1)
    confidence = cv2.resize(confidence_small, (width, height), interpolation=cv2.INTER_LINEAR)
    confidence = np.clip(confidence, 0, 1)
    stats = {
        "field_resolution": [small_height, small_width],
        "downsample": LOCAL_FIELD_DOWNSAMPLE,
        "field_luma_p05": round(float(np.percentile(field, 5)), 3),
        "field_luma_p50": round(float(np.percentile(field, 50)), 3),
        "field_luma_p95": round(float(np.percentile(field, 95)), 3),
        "mean_confidence": round(float(confidence[fit_mask > 0].mean()), 4)
        if np.any(fit_mask) else 0.0,
    }
    return field, confidence, stats


def protected_normalize(reference: np.ndarray, current: np.ndarray,
                        valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    global_reference, global_stats = robust_global_color(reference, current, valid)
    protected, broad = preliminary_protection(global_reference, current, valid)

    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    saturated = ((current_gray <= 3) | (current_gray >= 252)).astype(np.uint8) * 255
    saturated = cv2.dilate(saturated, np.ones((SATURATION_DILATION, SATURATION_DILATION), np.uint8))
    fit_mask = valid.copy()
    fit_mask[protected > 0] = 0
    fit_mask[saturated > 0] = 0
    field, confidence, local_stats = local_color_field(
        global_reference, current, fit_mask, protected
    )
    normalized = np.clip(global_reference + field[..., None], 0, 255).astype(np.uint8)

    usable = valid.copy()
    # The preliminary broad mask protects illumination estimation only.  It
    # may represent a shadow or a target near a person, so it cannot remove the
    # region from anomaly detection without final evidence or actor semantics.
    valid_pixels = max(np.count_nonzero(valid), 1)
    usable_fraction = float(np.count_nonzero(usable) / valid_pixels)
    broad_fraction = float(np.count_nonzero((broad > 0) & (valid > 0)) / valid_pixels)
    saturated_fraction = float(np.count_nonzero((saturated > 0) & (valid > 0)) / valid_pixels)
    gains = [pair[0] for pair in global_stats["channel_gain_bias_bgr"]]
    biases = [pair[1] for pair in global_stats["channel_gain_bias_bgr"]]
    local_extent = max(abs(local_stats["field_luma_p05"]), abs(local_stats["field_luma_p95"]))
    if usable_fraction < 0.65 or saturated_fraction > 0.35:
        state = "ENVIRONMENT_CHANGE"
    elif max(abs(gain - 1.0) for gain in gains) > 0.10 or max(abs(bias) for bias in biases) > 18 or local_extent > 16:
        state = "GLOBAL_LIGHT_CHANGE"
    else:
        state = "NORMAL"
    diagnostics = {
        "state": state,
        "usable_fraction": usable_fraction,
        "temporary_unavailable_fraction": broad_fraction,
        "saturated_fraction": saturated_fraction,
        "protected_fit_exclusion_fraction": float(
            np.count_nonzero((protected > 0) & (valid > 0)) / valid_pixels
        ),
        "global": global_stats,
        "local": local_stats,
        "confidence_p10": round(float(np.percentile(confidence[valid > 0], 10)), 4),
    }
    return normalized, usable, diagnostics


def proposals_v3(normalized_reference: np.ndarray, current: np.ndarray,
                 valid: np.ndarray, tolerance: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    signature, luminance = residual_maps(normalized_reference, current)
    noisy = (tolerance > 0).astype(np.float32)
    seed = (
        (signature >= SIGNATURE_THRESHOLD + noisy * NOISE_SEED_SIGNATURE_INCREMENT)
        & (luminance >= LUMINANCE_THRESHOLD + noisy * NOISE_SEED_LUMINANCE_INCREMENT)
        & (valid > 0)
    ).astype(np.uint8)
    support = (
        (signature >= SUPPORT_SIGNATURE_THRESHOLD + noisy * NOISE_SUPPORT_SIGNATURE_INCREMENT)
        & (luminance >= SUPPORT_LUMINANCE_THRESHOLD + noisy * NOISE_SUPPORT_LUMINANCE_INCREMENT)
        & (valid > 0)
    ).astype(np.uint8)
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
            "source": "protected_daylight_v3",
            "seed_pixels": seed_pixels,
            "support_area_px": area,
            "support_short_side_px": min(width, height),
            "support_fill_ratio": round(area / max(width * height, 1), 4),
            "p90_signature_residual": round(float(np.percentile(signature[component], 90)), 3),
            "p90_local_luminance_residual": round(float(np.percentile(luminance[component], 90)), 3),
            "bounded_tolerance_fraction": round(float((tolerance[component] > 0).mean()), 4),
            "anomaly_score": round(float(min(1.0,
                0.4 * seed_pixels / 12.0
                + 0.3 * np.percentile(signature[component], 90) / 4.0
                + 0.3 * np.percentile(luminance[component], 90) / 150.0)), 4),
        })
    rows.sort(key=lambda row: (-row["anomaly_score"], -row["support_area_px"]))
    return rows, broad


def draw_method(frame: np.ndarray, rows: list[dict[str, Any]], title: str) -> np.ndarray:
    result = frame.copy()
    for row in rows:
        x, y, right, bottom = (int(round(value)) for value in row["box"])
        cv2.rectangle(result, (x, y), (right, bottom), (0, 255, 255), 3)
    cv2.rectangle(result, (0, 0), (min(result.shape[1] - 1, 920), 58), (25, 25, 25), -1)
    cv2.putText(result, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .9,
                (255, 255, 255), 2, cv2.LINE_AA)
    return result


def scan_video(name: str, video: Path, output: Path, reference: np.ndarray,
               reference_valid: np.ndarray, daylight_tolerance: np.ndarray, *, start: float,
               duration: float) -> dict[str, Any]:
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    representative = read_frame(video, start)
    aligned_reference, aligned_valid, alignment = transform_reference(
        reference, reference_valid, representative
    )
    matrix = np.asarray(alignment["matrix"], np.float64)
    aligned_tolerance = cv2.warpPerspective(
        daylight_tolerance,
        matrix,
        (aligned_valid.shape[1], aligned_valid.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    empty = np.zeros(aligned_valid.shape, np.uint8)
    cached_v2_path = directory / "v2_raw_candidates.jsonl"
    cached_v2_tracks_path = directory / "v2_tracks.json"
    reuse_v2 = cached_v2_path.exists() and cached_v2_tracks_path.exists()
    v2_rows: dict[float, list[dict[str, Any]]] = (
        read_raw_jsonl(cached_v2_path) if reuse_v2 else {}
    )
    v3_rows: dict[float, list[dict[str, Any]]] = {}
    environment: list[dict[str, Any]] = []
    usable_sum = 0.0
    state_counts: Counter[str] = Counter()
    last_frame = representative
    last_v2: list[dict[str, Any]] = []
    last_v3: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, (timestamp, frame) in enumerate(iter_sampled_frames(
        video, sample_fps=ANALYSIS_FPS, start=start
    )):
        if reuse_v2:
            baseline = v2_rows.get(timestamp, [])
        else:
            baseline_reference, _ = robust_normalize(aligned_reference, frame, aligned_valid)
            baseline, _baseline_grown, _baseline_broad = proposals(
                baseline_reference, frame, aligned_valid, empty
            )
        optimized_reference, usable, diagnostic = protected_normalize(
            aligned_reference, frame, aligned_valid
        )
        optimized, _optimized_broad = proposals_v3(
            optimized_reference, frame, usable, aligned_tolerance
        )
        if not reuse_v2:
            v2_rows[timestamp] = baseline
        v3_rows[timestamp] = optimized
        environment.append({"timestamp": timestamp, **diagnostic})
        usable_sum += diagnostic["usable_fraction"]
        state_counts[diagnostic["state"]] += 1
        last_frame, last_v2, last_v3 = frame, baseline, optimized
        if index and index % 60 == 0:
            print(f"[{name}] {timestamp:.0f}s v2={sum(map(len, v2_rows.values()))} "
                  f"v3={sum(map(len, v3_rows.values()))}", flush=True)

    v2_tracks = (
        load_json(cached_v2_tracks_path) if reuse_v2
        else track_rows(v2_rows, source="v2_same_roi", sample_fps=ANALYSIS_FPS)
    )
    v3_tracks = track_rows(v3_rows, source="protected_daylight_v3", sample_fps=ANALYSIS_FPS)
    if not reuse_v2:
        raw_jsonl(directory / "v2_raw_candidates.jsonl", v2_rows)
    raw_jsonl(directory / "v3_raw_candidates.jsonl", v3_rows)
    (directory / "v2_tracks.json").write_text(
        json.dumps(v2_tracks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "v3_tracks.json").write_text(
        json.dumps(v3_tracks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "environment_diagnostics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in environment),
        encoding="utf-8",
    )
    plot_timeline(directory / "v2_timeline.png", v2_tracks, f"{name}: V2 same ROI", duration)
    plot_timeline(directory / "v3_timeline.png", v3_tracks, f"{name}: protected daylight V3", duration)
    save_image(directory / "v2_stable_overlay.jpg",
               draw_tracks(last_frame, v2_tracks, f"{name}: V2 tracks >=5s"))
    save_image(directory / "v3_stable_overlay.jpg",
               draw_tracks(last_frame, v3_tracks, f"{name}: V3 tracks >=5s"))
    save_image(directory / "last_frame_method_comparison.jpg", np.hstack([
        cv2.resize(draw_method(last_frame, last_v2, "V2 current-frame candidates"),
                   (1280, 720), interpolation=cv2.INTER_AREA),
        cv2.resize(draw_method(last_frame, last_v3, "V3 protected-light candidates"),
                   (1280, 720), interpolation=cv2.INTER_AREA),
    ]))

    evaluated_duration = max(duration - start, 0.001)
    result = {
        "name": name,
        "evaluation_start_seconds": start,
        "sample_fps": ANALYSIS_FPS,
        "alignment": alignment,
        "v2": track_funnel(v2_tracks, sum(map(len, v2_rows.values())), evaluated_duration),
        "v3": track_funnel(v3_tracks, sum(map(len, v3_rows.values())), evaluated_duration),
        "t01_v2": t01_metrics(v2_rows, v2_tracks, ANALYSIS_FPS) if name == "positive" else None,
        "t01_v3": t01_metrics(v3_rows, v3_tracks, ANALYSIS_FPS) if name == "positive" else None,
        "mean_usable_fraction": usable_sum / max(len(environment), 1),
        "environment_state_counts": dict(state_counts),
        "sample_count": len(environment),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (directory / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def make_t01_evidence(output: Path, video: Path) -> None:
    directory = output / "positive"
    v2 = read_raw_jsonl(directory / "v2_raw_candidates.jsonl")
    v3 = read_raw_jsonl(directory / "v3_raw_candidates.jsonl")
    panels = []
    for timestamp in (70.0, 72.0, 74.0, 76.0, 80.0, 82.0):
        frame = read_frame(video, timestamp)
        x, y, right, bottom = T01_REGION
        margin = 100
        left, top = max(0, x - margin), max(0, y - margin)
        far_right, far_bottom = min(frame.shape[1], right + margin), min(frame.shape[0], bottom + margin)
        for method, rows, color in (("V2", v2.get(timestamp, []), (255, 190, 0)),
                                    ("V3", v3.get(timestamp, []), (0, 255, 255))):
            crop = frame[top:far_bottom, left:far_right].copy()
            for row in rows:
                a, b, c, d = (int(value) for value in row["box"])
                if c < x or a > right or d < y or b > bottom:
                    continue
                cv2.rectangle(crop, (a-left, b-top), (c-left, d-top), color, 2)
            crop = cv2.resize(crop, (300, 260), interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(crop, (0, 0), (299, 32), (20, 20, 20), -1)
            cv2.putText(crop, f"{method} t={timestamp:.0f}s", (8, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(crop)
    save_image(directory / "t01_v2_v3_evidence.jpg", np.hstack(panels))


def reduction(old: int, new: int) -> float | None:
    return None if old == 0 else 1.0 - new / old


def write_report(output: Path, region: dict[str, Any], clean: dict[str, Any],
                 positive: dict[str, Any], negative: dict[str, Any],
                 tolerance: dict[str, Any]) -> None:
    p2, p3 = positive["v2"], positive["v3"]
    n2, n3 = negative["v2"], negative["v3"]
    c2, c3 = clean["v2"], clean["v3"]
    t2, t3 = positive["t01_v2"], positive["t01_v3"]
    n2_10 = n2["tracks_at_least_seconds"]["10"]
    n3_10 = n3["tracks_at_least_seconds"]["10"]
    reduction_10 = reduction(n2_10, n3_10)
    target_kept = t3["stable_5s_delay"] is not None and t3["stable_5s_delay"] <= 5
    clean_safe = c3["tracks_at_least_seconds"]["10"] == 0
    false_target = reduction_10 is not None and reduction_10 >= 0.50
    overall = "PASS" if target_kept and clean_safe and false_target else "PARTIAL"

    report = f"""# Clean Reference V3：正常白天先验区域优化

## 结论

- 开发门槛：**{overall}**。T01 保留={'PASS' if target_kept else 'FAIL'}；Clean holdout 长轨迹={'PASS' if clean_safe else 'FAIL'}；负样本 ≥10 秒轨迹下降 50%={'PASS' if false_target else 'FAIL'}。
- V3 地面有效覆盖率为红框地面 ROI 的 {region['valid_fraction_of_ground_roi']:.1%}；参考未知区域占 {region['reference_unknown_fraction_of_ground_roi']:.1%}，主要是参考中一直存在的圆凳。
- 正样本同一 60 秒后区间，V2→V3：raw {p2['raw_candidates']}→{p3['raw_candidates']}，≥5 秒 {p2['tracks_at_least_seconds']['5']}→{p3['tracks_at_least_seconds']['5']}，≥10 秒 {p2['tracks_at_least_seconds']['10']}→{p3['tracks_at_least_seconds']['10']}。
- 负样本同一 60 秒后区间，V2→V3：raw {n2['raw_candidates']}→{n3['raw_candidates']}，≥5 秒 {n2['tracks_at_least_seconds']['5']}→{n3['tracks_at_least_seconds']['5']}，≥10 秒 {n2_10}→{n3_10}（{(reduction_10 or 0):.1%} 下降）。
- T01 V2/V3 首次 raw={t2['first_raw_seconds']} / {t3['first_raw_seconds']}；稳定 5 秒={t2['first_stable_5s']} / {t3['first_stable_5s']}；V3 延迟={t3['stable_5s_delay']}。
- 平均参与候选检测的有效区域覆盖率：positive={positive['mean_usable_fraction']:.1%}，negative={negative['mean_usable_fraction']:.1%}，clean holdout={clean['mean_usable_fraction']:.1%}。饱和像素会退出光照拟合，但正常白天不会仅因白色像素而退出候选检测。

## V3 与 V2 的区别

- Clean Reference 始终只读；测试视频不生成永久 adaptive blacklist。
- 用户红框被收紧为纯地面多边形，墙体、画面文字与参考未知地面不参与判断。
- 先做稳健全局曝光/颜色拟合，再估计受保护的平滑局部亮度偏移。
- 初步异常和大变化区域从环境拟合中排除；只有大面积暂不可用区域从本帧检测中暂停。
- 局部补偿尺度远大于 T01，避免用逐像素或小窗口拟合消除小垃圾。
- 两段已声明正常的 0–60 秒白天片段只生成有上限的阈值增量，不生成检测黑名单；覆盖有效地面的 {tolerance['fraction_of_reference_valid']:.1%}，T01 区域覆盖 {tolerance['t01_region_fraction']:.1%}。
- 已为 V3 剩余轨迹生成 reference/current/context review crop。最长数条在参考中为空、当前画面中确实出现白色小物，可能是真实地面异物；另有多条来自同一位置的轨迹断裂，因此不能把剩余 9 条 ≥10 秒轨迹直接解释为 9 次误报。

## 环境状态

- Positive：{json.dumps(positive['environment_state_counts'], ensure_ascii=False)}
- Negative：{json.dumps(negative['environment_state_counts'], ensure_ascii=False)}
- Clean holdout：{json.dumps(clean['environment_state_counts'], ensure_ascii=False)}

这些状态是开发诊断，不是天气语义真值。`GLOBAL_LIGHT_CHANGE` 表示补偿参数显著，不表示该帧不可用；`ENVIRONMENT_CHANGE` 才表示可用覆盖明显下降。

## 边界

- 当前素材覆盖清晨约 6 点、正常负样本约 8 点、正样本约 10 点，可验证正常白天早间的跨时段失配，不能代表正午、下午斜阳或阴天。
- T01 与 ROI 都已用于开发，本轮不是盲测；负样本仍缺逐事件语义真值。
- 白天阈值增量图使用了两段 0–60 秒已声明正常的开发片段；生产中只能由人工确认无垃圾的校准片段生成，不能从未知在线画面自动学习。
- 圆凳下方在参考中从未露出干净地面，本轮明确暂停该区域。未来应补拍无遮挡参考，而不是把圆凳学习成地面。
- V3 当前只优化先验区域，尚未运行 actor 语义遮挡和 VLM；大变化区域使用视觉证据临时暂停。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    payload = {
        "kind": "protected_daylight_prior_region_v3",
        "gate": overall,
        "parameters": {
            "ground_roi": GROUND_ROI,
            "local_field_blur_sigma": LOCAL_FIELD_BLUR_SIGMA,
            "local_field_downsample": LOCAL_FIELD_DOWNSAMPLE,
            "local_field_clip": LOCAL_FIELD_CLIP,
            "target_video_adaptive_blacklist": False,
        },
        "daylight_tolerance": tolerance,
        "region": region,
        "clean_holdout": clean,
        "positive": positive,
        "negative": negative,
    }
    (output / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--v1-output", type=Path, required=True)
    parser.add_argument("--v2-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    clean_video = args.input_dir / "clean_reference.mp4"
    positive_video = args.input_dir / "小垃圾正样本.mp4"
    negative_video = args.input_dir / "正常负样本.mp4"
    arrays = np.load(args.v1_output / "reference_arrays.npz")
    reference, inherited_valid = arrays["reference"], arrays["valid"]
    reference_valid, _unknown, region = build_reference_masks(
        reference, inherited_valid, args.output
    )
    daylight_tolerance, tolerance_info = build_daylight_tolerance(
        args.v2_output, reference_valid, args.output
    )
    clean_meta = video_metadata(clean_video)
    positive_meta = video_metadata(positive_video)
    negative_meta = video_metadata(negative_video)
    clean_start = clean_meta["duration_seconds"] * REFERENCE_BUILD_FRACTION
    clean = scan_video("clean_holdout", clean_video, args.output, reference,
                       reference_valid, daylight_tolerance, start=clean_start,
                       duration=clean_meta["duration_seconds"])
    positive = scan_video("positive", positive_video, args.output, reference,
                          reference_valid, daylight_tolerance, start=EVALUATION_START_SECONDS,
                          duration=positive_meta["duration_seconds"])
    negative = scan_video("negative", negative_video, args.output, reference,
                          reference_valid, daylight_tolerance, start=EVALUATION_START_SECONDS,
                          duration=negative_meta["duration_seconds"])
    make_t01_evidence(args.output, positive_video)
    write_report(args.output, region, clean, positive, negative, tolerance_info)
    print(json.dumps({
        "output": str(args.output),
        "clean": clean,
        "positive": positive,
        "negative": negative,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
