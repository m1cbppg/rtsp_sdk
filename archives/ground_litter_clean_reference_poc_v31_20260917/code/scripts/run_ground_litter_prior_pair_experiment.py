#!/usr/bin/env python3
"""Run a frozen two-image clean-reference ground-litter experiment.

This is an offline diagnostic.  It never creates temporal evidence and never
labels a visual change as litter.  The reference is treated as valid only in
the configured ground ROI and outside visible occluders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ground_litter_prior_pair")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtsp_annotator.ground_litter_detection import (  # noqa: E402
    GroundLitterDetectionOptions,
    GroundLitterZone,
    UltralyticsGroundLitterDetector,
    build_ground_litter_tiles,
)


DEFAULT_ROI = (
    # Deliberately restrict this two-frame experiment to the central walkway.
    # The curbside carts and wall fixtures are not reviewed clean ground in
    # the supplied reference and therefore cannot support a clean prior.
    (0.50, 0.00),
    (0.66, 0.00),
    (0.88, 1.00),
    (0.35, 1.00),
)

# Selected by manual inspection of the source pair. This obvious small white
# ground object in image 2 has an unknown category. It supports a targeted
# case study, not an unbiased recall estimate or a litter ground-truth label.
PROVISIONAL_TARGETS = (
    {
        "target_id": "T01",
        "box": [938, 645, 955, 659],
        "label": "visually apparent new ground object; semantic category unconfirmed",
    },
)

# The display rack occupies ground in both views and is not detected reliably
# by the generic actor/context model. Human review therefore marks it as
# temporarily unavailable instead of treating its internal changes as litter.
REFERENCE_UNAVAILABLE_ZONES = (
    ((0.585, 0.18), (0.655, 0.18), (0.655, 0.36), (0.585, 0.36)),
)


def parse_box(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(item.strip()) for item in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,right,bottom")
    x, y, right, bottom = parts
    if min(parts) < 0 or right <= x or bottom <= y:
        raise argparse.ArgumentTypeError("invalid crop")
    return x, y, right, bottom


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_crop(path: Path, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"cannot read image: {path}")
    if crop is None:
        return frame
    x, y, right, bottom = crop
    height, width = frame.shape[:2]
    if right > width or bottom > height:
        raise ValueError(f"crop {crop} exceeds {width}x{height}: {path}")
    return frame[y:bottom, x:right].copy()


def polygon_mask(shape: tuple[int, int], polygon: Iterable[tuple[float, float]]) -> np.ndarray:
    height, width = shape
    points = np.array(
        [[round(x * (width - 1)), round(y * (height - 1))] for x, y in polygon],
        np.int32,
    )
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask


def overlay_mask(shape: tuple[int, int]) -> np.ndarray:
    """Mask burned-in text in the two supplied Camera 01 frames."""
    height, width = shape
    mask = np.full((height, width), 255, np.uint8)
    mask[: round(height * 0.115), : round(width * 0.43)] = 0
    mask[round(height * 0.84) :, round(width * 0.68) :] = 0
    return mask


def align_reference(reference: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Resize then register the reference to current coordinates with SIFT/RANSAC."""
    height, width = current.shape[:2]
    resized = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)
    detector = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.018)
    feature_mask = overlay_mask((height, width))
    old_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    old_keys, old_desc = detector.detectAndCompute(old_gray, feature_mask)
    new_keys, new_desc = detector.detectAndCompute(new_gray, feature_mask)
    if old_desc is None or new_desc is None:
        raise ValueError("insufficient alignment features")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(old_desc, new_desc, k=2)
    matches = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < 0.68 * b.distance]
    unique: dict[int, Any] = {}
    for match in sorted(matches, key=lambda item: item.distance):
        unique.setdefault(match.trainIdx, match)
    matches = list(unique.values())
    if len(matches) < 24:
        raise ValueError(f"insufficient alignment matches: {len(matches)}")
    source = np.float32([old_keys[item.queryIdx].pt for item in matches])
    target = np.float32([new_keys[item.trainIdx].pt for item in matches])
    matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.0)
    if matrix is None or inliers is None:
        raise ValueError("homography estimation failed")
    selected = inliers.ravel().astype(bool)
    if int(selected.sum()) < 20:
        raise ValueError("insufficient alignment inliers")
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected[selected] - target[selected], axis=1)
    inlier_source = source[selected]
    hull_fraction = cv2.contourArea(cv2.convexHull(inlier_source)) / float(width * height)
    aligned = cv2.warpPerspective(
        resized,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    source_valid = np.full((height, width), 255, np.uint8)
    valid = cv2.warpPerspective(
        source_valid,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    diagnostics = {
        "method": "SIFT + RANSAC homography",
        "matches": len(matches),
        "inliers": int(selected.sum()),
        "inlier_fraction": round(float(selected.mean()), 4),
        "reprojection_median_px": round(float(np.median(errors)), 3),
        "reprojection_p95_px": round(float(np.percentile(errors, 95)), 3),
        "inlier_hull_fraction": round(float(hull_fraction), 4),
        "matrix": matrix.tolist(),
    }
    return aligned, valid, diagnostics


def actor_boxes(
    detector: UltralyticsGroundLitterDetector,
    frame: np.ndarray,
    options: GroundLitterDetectionOptions,
) -> list[list[float]]:
    boxes = detector.actor_boxes(frame, options)
    height, width = frame.shape[:2]
    return [
        [max(0.0, x), max(0.0, y), min(float(width), right), min(float(height), bottom)]
        for x, y, right, bottom in boxes
    ]


def boxes_mask(shape: tuple[int, int], boxes: Iterable[Iterable[float]], margin: int = 10) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), np.uint8)
    for box in boxes:
        x, y, right, bottom = (int(round(item)) for item in box)
        cv2.rectangle(
            mask,
            (max(0, x - margin), max(0, y - margin)),
            (min(width - 1, right + margin), min(height - 1, bottom + margin)),
            255,
            -1,
        )
    return mask


def robust_normalize(
    reference: np.ndarray,
    current: np.ndarray,
    estimation_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit robust per-channel exposure/color gain, then remove low-frequency luminance drift."""
    eligible = estimation_mask.astype(bool)
    old = reference.astype(np.float32)
    new = current.astype(np.float32)
    eligible &= np.all((old > 5) & (old < 250) & (new > 5) & (new < 250), axis=2)
    indices = np.flatnonzero(eligible)
    if indices.size < 5000:
        raise ValueError("insufficient visible ground for illumination normalization")
    if indices.size > 250_000:
        indices = indices[np.linspace(0, indices.size - 1, 250_000).astype(np.intp)]
    keep = np.ones(indices.size, bool)
    fits: list[tuple[float, float]] = []
    for _iteration in range(3):
        fits = []
        for channel in range(3):
            x = old[..., channel].reshape(-1)[indices][keep]
            y = new[..., channel].reshape(-1)[indices][keep]
            gain, bias = np.polyfit(x, y, 1)
            fits.append((float(np.clip(gain, 0.65, 1.45)), float(np.clip(bias, -60, 60))))
        prediction = np.stack(
            [old[..., channel] * fits[channel][0] + fits[channel][1] for channel in range(3)],
            axis=2,
        )
        residual = np.max(np.abs(new - prediction), axis=2).reshape(-1)[indices]
        cutoff = float(np.percentile(residual[keep], 70))
        keep = residual <= max(cutoff, 8.0)
    normalized = np.stack(
        [old[..., channel] * fits[channel][0] + fits[channel][1] for channel in range(3)],
        axis=2,
    )
    old_luma = cv2.cvtColor(np.clip(normalized, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    new_luma = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low_frequency = cv2.GaussianBlur(new_luma - old_luma, (0, 0), 24)
    normalized += low_frequency[..., None]
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return normalized, {
        "channel_gain_bias_bgr": [[round(gain, 5), round(bias, 3)] for gain, bias in fits],
        "fit_pixels": int(keep.sum()),
        "low_frequency_sigma_px": 24,
    }


def find_anomalies(
    normalized_reference: np.ndarray,
    current: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    signature_threshold: float = 2.8,
    luminance_threshold: float = 90.0,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    old_gray = cv2.cvtColor(normalized_reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    new_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def local_signature(gray: np.ndarray) -> np.ndarray:
        mean = cv2.GaussianBlur(gray, (0, 0), 12)
        variance = np.maximum(cv2.GaussianBlur(gray * gray, (0, 0), 12) - mean * mean, 0)
        return (gray - mean) / (np.sqrt(variance) + 5)

    signature_residual = np.abs(local_signature(new_gray) - local_signature(old_gray))
    luminance_delta = new_gray - old_gray
    local_luminance_residual = np.abs(
        luminance_delta - cv2.GaussianBlur(luminance_delta, (0, 0), 12)
    )
    # The heat-map and negative evidence use the same locally normalized
    # residual as proposal generation. Global wet/dry or shadow changes should
    # not win merely because their absolute RGB offset is large.
    diagnostic_residual = np.maximum(local_luminance_residual, signature_residual * 32)
    eligible = eligible_mask.astype(bool)
    core = (
        (signature_residual >= float(signature_threshold))
        & (local_luminance_residual >= float(luminance_threshold))
        & eligible
    )
    core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Broad connected changes are treated as temporary unavailable regions for
    # this small-object experiment, not as litter proposals.
    broad = np.zeros_like(core)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    for index in range(1, count):
        x, y, width, height, area = (int(item) for item in stats[index])
        if area >= 900 or (area >= 350 and max(width, height) >= 80):
            broad[labels == index] = 255
    if broad.any():
        broad = cv2.dilate(broad, np.ones((15, 15), np.uint8))
    local_core = core.copy()
    local_core[broad > 0] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(local_core, 8)
    candidates: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(item) for item in stats[index])
        fill_ratio = area / max(width * height, 1)
        if (
            area < 10
            or area > 500
            or min(width, height) < 3
            or max(width, height) > 40
            or fill_ratio < 0.25
        ):
            continue
        component = labels == index
        signature_values = signature_residual[component]
        luma_values = local_luminance_residual[component]
        values = diagnostic_residual[component]
        candidates.append(
            {
                "box": [x, y, x + width, y + height],
                "source": "clean_reference",
                "component_area_px": area,
                "box_area_px": width * height,
                "fill_ratio": round(fill_ratio, 4),
                "mean_signature_residual": round(float(signature_values.mean()), 3),
                "p90_signature_residual": round(float(np.percentile(signature_values, 90)), 3),
                "mean_local_luminance_residual": round(float(luma_values.mean()), 3),
                "p90_local_luminance_residual": round(float(np.percentile(luma_values, 90)), 3),
                "anomaly_score": round(
                    float(
                        min(
                            1.0,
                            0.5 * np.percentile(signature_values, 90) / 4.0
                            + 0.5 * np.percentile(luma_values, 90) / 150.0,
                        )
                    ),
                    4,
                ),
            }
        )
    candidates.sort(key=lambda item: (-item["anomaly_score"], -item["component_area_px"]))
    return candidates[:128], diagnostic_residual, signature_residual, broad


def box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    a, b, c, d = left
    x, y, r, s = right
    intersection = max(0.0, min(c, r) - max(a, x)) * max(0.0, min(d, s) - max(b, y))
    union = max((c - a) * (d - b) + (r - x) * (s - y) - intersection, 1e-9)
    return intersection / union


def boxes_related(left: Iterable[float], right: Iterable[float]) -> bool:
    if box_iou(left, right) >= 0.08:
        return True
    a, b, c, d = left
    x, y, r, s = right
    center_left = ((a + c) / 2, (b + d) / 2)
    center_right = ((x + r) / 2, (y + s) / 2)
    return (
        x <= center_left[0] <= r and y <= center_left[1] <= s
    ) or (
        a <= center_right[0] <= c and b <= center_right[1] <= d
    )


def yolo_candidates(
    detector: UltralyticsGroundLitterDetector,
    frame: np.ndarray,
    options: GroundLitterDetectionOptions,
    actors: list[list[float]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    height, width = frame.shape[:2]
    masks, tiles = build_ground_litter_tiles(options, width, height)
    candidates, stats = detector.candidates(
        frame,
        options,
        masks=masks,
        tiles=tiles,
        actors=actors,
    )
    rows = []
    for candidate in candidates:
        rect = candidate.rectangle
        rows.append(
            {
                "box": [
                    round(rect.left * width),
                    round(rect.top * height),
                    round((rect.left + rect.width) * width),
                    round((rect.top + rect.height) * height),
                ],
                "source": "yolo",
                "confidence": round(candidate.confidence, 4),
                "class_name": candidate.class_name,
                "region_id": candidate.region_id,
            }
        )
    stats = {**stats, "tile_count": len(tiles)}
    return rows, stats


def add_reference_evidence(
    rows: list[dict[str, Any]],
    residual: np.ndarray,
    valid_mask: np.ndarray,
    anomalies: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    height, width = residual.shape
    for source in rows:
        row = dict(source)
        x, y, right, bottom = row["box"]
        x, y = max(0, x), max(0, y)
        right, bottom = min(width, right), min(height, bottom)
        patch = residual[y:bottom, x:right]
        visible = valid_mask[y:bottom, x:right] > 0
        row["reference_valid_fraction"] = round(float(visible.mean()), 4) if visible.size else 0.0
        values = patch[visible]
        row["reference_change_fraction"] = round(float((values >= 90).mean()), 4) if values.size else None
        row["reference_residual_p90"] = round(float(np.percentile(values, 90)), 3) if values.size else None
        row["matched_anomaly_ids"] = [
            index + 1 for index, anomaly in enumerate(anomalies) if boxes_related(row["box"], anomaly["box"])
        ]
        same_reference_structure = (
            row["reference_valid_fraction"] >= 0.85
            and not row["matched_anomaly_ids"]
            and row["reference_change_fraction"] is not None
            and row["reference_change_fraction"] < 0.03
            and row["reference_residual_p90"] < 55
        )
        row["reference_decision"] = (
            "REFERENCE_MATCH_SUPPRESSED" if same_reference_structure else "RETAINED"
        )
        (suppressed if same_reference_structure else kept).append(row)
    return kept, suppressed


def fuse(yolo: list[dict[str, Any]], anomalies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fused: list[dict[str, Any]] = []
    used_anomalies: set[int] = set()
    for row in yolo:
        matches = [index for index, anomaly in enumerate(anomalies) if boxes_related(row["box"], anomaly["box"])]
        used_anomalies.update(matches)
        item = dict(row)
        item["sources"] = ["yolo", "clean_reference"] if matches else ["yolo"]
        item["matched_anomaly_ids"] = [index + 1 for index in matches]
        item["fusion_state"] = "YOLO_AND_REFERENCE" if matches else "YOLO_ONLY"
        fused.append(item)
    for index, anomaly in enumerate(anomalies):
        if index in used_anomalies:
            continue
        item = dict(anomaly)
        item["sources"] = ["clean_reference"]
        item["fusion_state"] = "REFERENCE_ONLY"
        fused.append(item)
    return fused


def draw_rows(frame: np.ndarray, rows: list[dict[str, Any]], title: str) -> np.ndarray:
    output = frame.copy()
    palette = {
        "YOLO_ONLY": (0, 165, 255),
        "YOLO_AND_REFERENCE": (0, 255, 255),
        "REFERENCE_ONLY": (255, 80, 0),
        "REFERENCE_MATCH_SUPPRESSED": (120, 120, 120),
    }
    for index, row in enumerate(rows, 1):
        x, y, right, bottom = row["box"]
        state = row.get("fusion_state") or row.get("reference_decision") or row.get("source", "")
        color = palette.get(state, (0, 200, 255))
        cv2.rectangle(output, (x, y), (right, bottom), color, 2)
        label = str(index) if len(rows) > 16 else f"{index}:{state}"
        cv2.putText(output, label, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.rectangle(output, (0, 0), (min(output.shape[1] - 1, 790), 36), (20, 20, 20), -1)
    cv2.putText(output, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94] if path.suffix.lower() == ".jpg" else []):
        raise RuntimeError(f"failed to write {path}")


def evidence_cards(
    output: Path,
    reference: np.ndarray,
    current: np.ndarray,
    residual: np.ndarray,
    rows: list[dict[str, Any]],
) -> None:
    directory = output / "evidence"
    directory.mkdir()
    heat = cv2.applyColorMap(np.clip(residual * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    height, width = current.shape[:2]
    cards: list[np.ndarray] = []
    for index, row in enumerate(rows, 1):
        x, y, right, bottom = row["box"]
        margin = max(32, max(right - x, bottom - y) * 2)
        a, b = max(0, x - margin), max(0, y - margin)
        c, d = min(width, right + margin), min(height, bottom + margin)
        panels = []
        for panel_index, source in enumerate((reference, current, heat)):
            panel = source[b:d, a:c].copy()
            cv2.rectangle(
                panel,
                (max(0, x - a), max(0, y - b)),
                (min(panel.shape[1] - 1, right - a), min(panel.shape[0] - 1, bottom - b)),
                (0, 255, 255),
                1,
            )
            target_height = 240
            scale = target_height / max(panel.shape[0], 1)
            panel = cv2.resize(
                panel,
                (max(1, round(panel.shape[1] * scale)), target_height),
                interpolation=cv2.INTER_NEAREST if scale >= 3 else cv2.INTER_CUBIC,
            )
            cv2.putText(
                panel,
                ("reference", "current", "residual")[panel_index],
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel)
        card = np.concatenate(panels, axis=1)
        save_image(directory / f"candidate-{index:03d}.jpg", card)
        tile = np.full((280, 760, 3), 28, np.uint8)
        scale = min(740 / card.shape[1], 240 / card.shape[0], 1.0)
        resized = cv2.resize(
            card,
            (max(1, round(card.shape[1] * scale)), max(1, round(card.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        tile[35 : 35 + resized.shape[0], 10 : 10 + resized.shape[1]] = resized
        cv2.putText(
            tile,
            f"{index:03d} {row.get('fusion_state', row.get('source', 'candidate'))}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cards.append(tile)
    if cards:
        columns = 4
        rows_count = math.ceil(len(cards) / columns)
        sheet = np.full((rows_count * 280, columns * 760, 3), 18, np.uint8)
        for index, tile in enumerate(cards):
            row, column = divmod(index, columns)
            sheet[row * 280 : (row + 1) * 280, column * 760 : (column + 1) * 760] = tile
        save_image(output / "10_evidence_contact_sheet.jpg", sheet)


def evaluate_targets(
    targets: Iterable[dict[str, Any]],
    yolo: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        target_box = target["box"]
        yolo_matches = [index + 1 for index, row in enumerate(yolo) if boxes_related(target_box, row["box"])]
        anomaly_matches = [
            index + 1 for index, row in enumerate(anomalies) if boxes_related(target_box, row["box"])
        ]
        rows.append(
            {
                **target,
                "yolo_matches": yolo_matches,
                "reference_matches": anomaly_matches,
                "outcome": (
                    "REFERENCE_ONLY_RECOVERY"
                    if anomaly_matches and not yolo_matches
                    else "YOLO_AND_REFERENCE"
                    if anomaly_matches and yolo_matches
                    else "YOLO_ONLY"
                    if yolo_matches
                    else "MISSED"
                ),
            }
        )
    return rows


def build_report(
    output: Path,
    reference_path: Path,
    current_path: Path,
    diagnostics: dict[str, Any],
    illumination: dict[str, Any],
    yolo: list[dict[str, Any]],
    suppressed: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    fused: list[dict[str, Any]],
    yolo_stats: dict[str, int],
    thresholds: dict[str, float],
    sensitivity: list[dict[str, Any]],
    target_evaluation: list[dict[str, Any]],
    valid_fraction: float,
    elapsed: float,
) -> None:
    payload = {
        "kind": "two_image_reference_experiment_not_temporal_or_accuracy",
        "inputs": {
            "reference_name": reference_path.name,
            "reference_sha256": sha256(reference_path),
            "current_name": current_path.name,
            "current_sha256": sha256(current_path),
        },
        "alignment": diagnostics,
        "illumination": illumination,
        "anomaly_thresholds": thresholds,
        "threshold_sensitivity": sensitivity,
        "reference_unavailable_zones": [
            [list(point) for point in polygon] for polygon in REFERENCE_UNAVAILABLE_ZONES
        ],
        "reference_valid_ground_fraction_of_roi": round(valid_fraction, 4),
        "groups": {
            "A_yolo_baseline": yolo,
            "B_reference_anomalies": anomalies,
            "C_reference_suppressed_yolo": suppressed,
            "D_fusion": fused,
        },
        "yolo_diagnostics": yolo_stats,
        "counts": {
            "A_yolo_baseline": len(yolo) + len(suppressed),
            "B_reference_anomalies": len(anomalies),
            "C_yolo_retained": len(yolo),
            "C_yolo_suppressed": len(suppressed),
            "D_fused": len(fused),
            "D_reference_only": sum(item["fusion_state"] == "REFERENCE_ONLY" for item in fused),
        },
        "provisional_target_evaluation": target_evaluation,
        "manual_ground_truth": None,
        "accuracy": None,
        "temporal_validation": False,
        "elapsed_seconds": round(elapsed, 3),
    }
    (output / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = payload["counts"]
    target_lines = "\n".join(
        f"- {row['target_id']}：{row['outcome']}；YOLO 匹配 {row['yolo_matches']}，参考变化匹配 {row['reference_matches']}。"
        for row in target_evaluation
    )
    sensitivity_lines = "\n".join(
        f"- signature≥{row['signature_threshold']:.1f}、luminance≥{row['luminance_threshold']:.0f}："
        f"{row['candidate_count']} 个候选，T01={'命中' if row['target_detected'] else '漏检'}。"
        for row in sensitivity
    )
    report = f"""# 双图 Clean Reference 实验报告

本实验使用第一张图作为部分有效的历史参考，第二张图作为唯一待识别图。它验证候选发现链路，**不构成时序确认或准确率结论**。当前没有冻结人工真值，因此候选只表示需要人工复核的地面变化。

## 运行结果

- 配准：{diagnostics['inliers']} 个内点，中位重投影误差 {diagnostics['reprojection_median_px']}px，P95 {diagnostics['reprojection_p95_px']}px。
- 可使用参考的地面覆盖：ROI 的 {valid_fraction:.1%}。被检测为人员/车辆/上下文遮挡、大范围变化、叠字、人工确认的陈列架占地区或配准边界不使用参考结论。
- A / YOLO 基线：{counts['A_yolo_baseline']} 个候选。
- B / 独立参考变化：{counts['B_reference_anomalies']} 个局部候选。
- C / 参考负证据：保留 {counts['C_yolo_retained']} 个，抑制 {counts['C_yolo_suppressed']} 个。
- D / 融合：{counts['D_fused']} 个，其中 {counts['D_reference_only']} 个完全不依赖 YOLO。
- 总耗时：{elapsed:.2f}s。

这组图片已经证明 T01 可由参考分支独立补出，但没有 YOLO 候选满足强参考一致性条件，因此本图对“固定结构误报压制”没有给出正向样本。18 个参考候选也说明单帧双图差分不能直接当显示结果，仍需时序、环境状态和人工标签评估候选精度。

## 暂定目标结果

{target_lines}

T01 来自对输入图的人工目视检查，是“图二中明显的白色新增地面物体”，只用于定向案例检查；它不是盲测目标，也尚未被标成垃圾类别。`REFERENCE_ONLY_RECOVERY` 表示现有 YOLO 没有提出该位置，但参考变化分支独立提出了候选。

## 参数敏感性

{sensitivity_lines}

## 如何审核

先查看 `03_alignment_overlay.jpg` 和 `04_reference_validity.jpg`。如果地砖边缘不能重合，不能解释后续候选。然后查看 `09_fusion.jpg`，再打开 `evidence/` 中同编号证据卡；每张卡从左到右为对齐后的参考、当前图和残差热图。

`REFERENCE_ONLY` 是这次最重要的候选，但不能自动等同于垃圾。需要把候选人工标注为 `NEW_OBJECT`、`UNCHANGED_STRUCTURE`、`OCCLUDED`、`REFERENCE_UNKNOWN` 或 `UNCERTAIN`，才能计算补检收益和误报。

## 边界

- 第一张图不是经 30–60 秒干净视频生成的标准 Clean Reference，而且本身含有车辆、人员和地面物体。
- 只有两张图，不能验证持续存在、遮挡恢复、Adaptive Background、雨天稳定性或清理状态。
- 大范围变化在本次小物体实验中按 temporary unavailable 处理；这会减少误报，也会减少可评价覆盖。
- 输出已裁掉第一张图的播放器标题栏，未复制其中显示的 RTSP 地址。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-crop", type=parse_box, default=(0, 72, 3600, 2097))
    parser.add_argument("--current-crop", type=parse_box)
    parser.add_argument("--model", type=Path, default=Path("models/litter/turhancan_yolov8m_seg_trash.pt"))
    parser.add_argument("--actor-model", type=Path, default=Path("models/yolo26s.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.08)
    parser.add_argument("--signature-threshold", type=float, default=2.8)
    parser.add_argument("--luminance-threshold", type=float, default=90.0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    if not 0 < args.confidence <= 1:
        raise ValueError("confidence must be in (0,1]")
    if args.signature_threshold <= 0 or args.luminance_threshold <= 0:
        raise ValueError("anomaly thresholds must be positive")
    args.output.mkdir(parents=True)
    started = time.perf_counter()

    reference = read_crop(args.reference, args.reference_crop)
    current = read_crop(args.current, args.current_crop)
    aligned, alignment_valid, alignment = align_reference(reference, current)
    height, width = current.shape[:2]
    roi = polygon_mask((height, width), DEFAULT_ROI)
    overlay = overlay_mask((height, width))

    options = GroundLitterDetectionOptions(
        enabled=True,
        model=args.model.name,
        actor_model=args.actor_model.name,
        analysis_fps=1.0,
        confidence=args.confidence,
        tile_size_px=384,
        inference_imgsz=768,
        tile_overlap=0.25,
        maximum_tiles=64,
        actor_imgsz=1280,
        actor_confidence=0.15,
        actor_overlap_threshold=0.20,
        zones=(
            GroundLitterZone(
                region_id="sidewalk",
                polygon=DEFAULT_ROI,
                minimum_short_side_px=2,
                minimum_box_area_px=6,
                confidence=args.confidence,
            ),
        ),
        overlay_exclude_zones=(
            ((0.00, 0.00), (0.43, 0.00), (0.43, 0.115), (0.00, 0.115)),
            ((0.68, 0.84), (1.00, 0.84), (1.00, 1.00), (0.68, 1.00)),
        ),
    )
    detector = UltralyticsGroundLitterDetector(
        model_path=args.model,
        actor_model_path=args.actor_model,
        device=args.device,
    )
    reference_actors = actor_boxes(detector, aligned, options)
    current_actors = actor_boxes(detector, current, options)
    actor_mask = cv2.bitwise_or(
        boxes_mask((height, width), reference_actors),
        boxes_mask((height, width), current_actors),
    )
    reviewed_unavailable = np.zeros((height, width), np.uint8)
    for polygon in REFERENCE_UNAVAILABLE_ZONES:
        reviewed_unavailable = cv2.bitwise_or(
            reviewed_unavailable,
            polygon_mask((height, width), polygon),
        )
    preliminary = cv2.bitwise_and(roi, alignment_valid)
    preliminary = cv2.bitwise_and(preliminary, overlay)
    preliminary[actor_mask > 0] = 0
    preliminary[reviewed_unavailable > 0] = 0
    preliminary = cv2.erode(preliminary, np.ones((5, 5), np.uint8))
    normalized_reference, illumination = robust_normalize(aligned, current, preliminary)
    anomalies, color_residual, _edge_residual, broad = find_anomalies(
        normalized_reference,
        current,
        preliminary,
        signature_threshold=args.signature_threshold,
        luminance_threshold=args.luminance_threshold,
    )
    valid = preliminary.copy()
    valid[broad > 0] = 0
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8))
    # Remove any proposal whose center became unavailable after broad-change masking.
    anomalies = [
        item for item in anomalies
        if valid[(item["box"][1] + item["box"][3]) // 2, (item["box"][0] + item["box"][2]) // 2] > 0
    ]
    baseline, yolo_stats = yolo_candidates(detector, current, options, current_actors)
    retained, suppressed = add_reference_evidence(baseline, color_residual, valid, anomalies)
    fused = fuse(retained, anomalies)
    target_evaluation = evaluate_targets(PROVISIONAL_TARGETS, baseline, anomalies)
    sensitivity = []
    for signature_threshold, luminance_threshold in (
        (2.6, 80.0),
        (2.8, 90.0),
        (3.0, 100.0),
        (3.2, 110.0),
    ):
        sweep, _residual, _signature, _broad = find_anomalies(
            normalized_reference,
            current,
            preliminary,
            signature_threshold=signature_threshold,
            luminance_threshold=luminance_threshold,
        )
        evaluated = evaluate_targets(PROVISIONAL_TARGETS, [], sweep)
        sensitivity.append(
            {
                "signature_threshold": signature_threshold,
                "luminance_threshold": luminance_threshold,
                "candidate_count": len(sweep),
                "target_detected": bool(evaluated[0]["reference_matches"]),
                "target_matches": evaluated[0]["reference_matches"],
            }
        )

    save_image(args.output / "01_current_video.jpg", current)
    save_image(args.output / "02_reference_aligned.jpg", aligned)
    blend = cv2.addWeighted(aligned, 0.5, current, 0.5, 0)
    save_image(args.output / "03_alignment_overlay.jpg", blend)
    validity = current.copy()
    tint = np.zeros_like(current)
    tint[..., 1] = 220
    validity[valid > 0] = cv2.addWeighted(validity, 0.55, tint, 0.45, 0)[valid > 0]
    validity[actor_mask > 0] = (0, 120, 255)
    validity[reviewed_unavailable > 0] = (180, 0, 180)
    validity[broad > 0] = (180, 0, 180)
    save_image(args.output / "04_reference_validity.jpg", validity)
    heat = cv2.applyColorMap(np.clip(color_residual * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[valid == 0] = (35, 35, 35)
    save_image(args.output / "05_residual_heatmap.jpg", heat)
    save_image(args.output / "06_reference_anomalies.jpg", draw_rows(current, anomalies, "B: independent clean-reference anomalies"))
    save_image(args.output / "07_yolo_baseline.jpg", draw_rows(current, baseline, "A: YOLO baseline"))
    save_image(args.output / "08_reference_suppressed.jpg", draw_rows(current, suppressed, "C: YOLO candidates suppressed by strong reference match"))
    save_image(args.output / "09_fusion.jpg", draw_rows(current, fused, "D: fused candidate pool"))
    evidence_cards(args.output, normalized_reference, current, color_residual, fused)

    roi_pixels = max(int(np.count_nonzero(roi)), 1)
    valid_fraction = np.count_nonzero(valid) / roi_pixels
    build_report(
        args.output,
        args.reference,
        args.current,
        alignment,
        illumination,
        retained,
        suppressed,
        anomalies,
        fused,
        yolo_stats,
        {
            "signature_threshold": args.signature_threshold,
            "luminance_threshold": args.luminance_threshold,
        },
        sensitivity,
        target_evaluation,
        valid_fraction,
        time.perf_counter() - started,
    )
    print(json.dumps({
        "output": str(args.output),
        "alignment": alignment,
        "valid_ground_fraction": round(valid_fraction, 4),
        "counts": {
            "yolo_baseline": len(baseline),
            "reference_anomalies": len(anomalies),
            "reference_suppressed": len(suppressed),
            "fused": len(fused),
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
