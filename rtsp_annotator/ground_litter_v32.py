"""Production Clean Reference V3.2 ground-litter analysis.

The reference is immutable. Live frames may be aligned to it, but they never
replace or update it. Raw change components remain internal diagnostics; only
confirmed lifecycle events are converted to OSD detections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .event_engine import NormalizedRect
from .ground_litter_detection import (
    GroundLitterDetection,
    GroundLitterDetectionOptions,
    GroundLitterSnapshot,
)
from .ground_litter_geometry import box_overlap_fraction


PROFILE_KIND = "ground_litter_clean_reference_v32"
PROFILE_FILES = ("reference.png", "valid_mask.png", "daylight_tolerance.png")

SIGNATURE_THRESHOLD = 3.0
LUMINANCE_THRESHOLD = 100.0
SUPPORT_SIGNATURE_THRESHOLD = 1.2
SUPPORT_LUMINANCE_THRESHOLD = 25.0
MINIMUM_SEED_PIXELS = 4
MINIMUM_SUPPORT_AREA = 30
MINIMUM_SUPPORT_SHORT_SIDE = 6
MAXIMUM_SUPPORT_SIDE = 80
MAXIMUM_SUPPORT_BOX_AREA = 2500

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

EVENT_MATCH_DISTANCE_PX = 30.0
EVENT_MATCH_SIZE_RATIO = 5.0
EVENT_MERGE_DISTANCE_PX = EVENT_MATCH_DISTANCE_PX
EVENT_MERGE_SIZE_RATIO = 10.0
CONTEXT_MARGIN_PX = 80
CONTEXT_INNER_PAD_PX = 12
CONTEXT_TOUCH_GAP_PX = 20
CONTEXT_CANDIDATE_GUARD_PX = 4
CONTEXT_OCCLUSION_EXTERNAL_AREA_PX = 200
ANCHOR_SUPPORT_PAD_PX = 4
CLEAN_MAX_SUPPORT_PIXELS = 2
MAX_SAMPLE_GAP_FACTOR = 1.5
MAX_BOX_HISTORY = 30
MAX_MERGED_BOX_HISTORY = 60
MAX_VISIBLE_TIMESTAMP_HISTORY = 3600
MAX_STATE_HISTORY = 256


def _scaled_odd(value: int, scale: float, *, minimum: int = 3) -> int:
    result = max(minimum, int(round(value * scale)))
    return result if result % 2 else result + 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CleanReferenceProfileV32:
    profile_id: str
    root: Path
    reference: np.ndarray
    valid: np.ndarray
    tolerance: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def load(cls, root: Path, profile_id: str) -> "CleanReferenceProfileV32":
        directory = (Path(root).resolve() / profile_id).resolve()
        try:
            directory.relative_to(Path(root).resolve())
        except ValueError as exc:
            raise ValueError("ground_litter.profile_id路径越界") from exc
        metadata_path = directory / "profile.json"
        if not metadata_path.is_file():
            raise ValueError(f"Clean Reference profile不存在: {profile_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("kind") != PROFILE_KIND:
            raise ValueError("Clean Reference profile kind无效")
        if metadata.get("profile_id") != profile_id:
            raise ValueError("Clean Reference profile_id与目录不一致")
        checksums = metadata.get("sha256")
        if not isinstance(checksums, dict):
            raise ValueError("Clean Reference profile缺少SHA-256")
        for filename in PROFILE_FILES:
            path = directory / filename
            expected = checksums.get(filename)
            if not path.is_file() or not isinstance(expected, str):
                raise ValueError(f"Clean Reference profile缺少文件: {filename}")
            if _sha256(path) != expected.lower():
                raise ValueError(f"Clean Reference profile校验失败: {filename}")
        reference = cv2.imread(str(directory / "reference.png"), cv2.IMREAD_COLOR)
        valid = cv2.imread(str(directory / "valid_mask.png"), cv2.IMREAD_GRAYSCALE)
        tolerance = cv2.imread(
            str(directory / "daylight_tolerance.png"), cv2.IMREAD_GRAYSCALE
        )
        if reference is None or valid is None or tolerance is None:
            raise ValueError("Clean Reference profile图像无法读取")
        height, width = reference.shape[:2]
        if valid.shape != (height, width) or tolerance.shape != (height, width):
            raise ValueError("Clean Reference profile图像尺寸不一致")
        if metadata.get("reference_size") != [width, height]:
            raise ValueError("Clean Reference profile参考尺寸不一致")
        if np.count_nonzero(valid) < 5000:
            raise ValueError("Clean Reference profile有效地面不足")
        tolerance = tolerance.copy()
        tolerance[valid == 0] = 0
        return cls(profile_id, directory, reference, valid, tolerance, metadata)


def align_profile(
    profile: CleanReferenceProfileV32,
    frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Register the reviewed reference once to the live fixed-camera view."""
    if frame.shape != profile.reference.shape:
        raise ValueError(
            "Clean Reference分辨率不匹配: "
            f"期望{profile.reference.shape[1]}x{profile.reference.shape[0]}，"
            f"实际{frame.shape[1]}x{frame.shape[0]}"
        )
    height, width = frame.shape[:2]
    old_gray = cv2.cvtColor(profile.reference, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.018)
    feature_mask = np.full((height, width), 255, np.uint8)
    for polygon in profile.metadata.get("overlay_exclude_zones", []):
        points = np.round(
            np.asarray(polygon, np.float32) * np.asarray([width, height])
        ).astype(np.int32)
        cv2.fillPoly(feature_mask, [points], 0)
    old_keys, old_desc = detector.detectAndCompute(old_gray, feature_mask)
    new_keys, new_desc = detector.detectAndCompute(new_gray, feature_mask)
    if old_desc is None or new_desc is None:
        raise ValueError("Clean Reference视角匹配特征不足")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(old_desc, new_desc, k=2)
    matches = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.68 * second.distance
    ]
    unique: dict[int, Any] = {}
    for match in sorted(matches, key=lambda item: item.distance):
        unique.setdefault(match.trainIdx, match)
    matches = list(unique.values())
    if len(matches) < 24:
        raise ValueError("Clean Reference视角匹配点不足")
    source = np.float32([old_keys[item.queryIdx].pt for item in matches])
    target = np.float32([new_keys[item.trainIdx].pt for item in matches])
    matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.0)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        raise ValueError("Clean Reference视角匹配失败")
    selected = inliers.ravel().astype(bool)
    if int(selected.sum()) < 20:
        raise ValueError("Clean Reference视角匹配内点不足")
    projected = cv2.perspectiveTransform(
        source.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected[selected] - target[selected], axis=1)
    hull_fraction = cv2.contourArea(
        cv2.convexHull(source[selected])
    ) / float(width * height)
    if float(np.median(errors)) > 2.0 or hull_fraction < 0.02:
        raise ValueError("Clean Reference视角变化过大，需要重新标定")
    aligned_reference = cv2.warpPerspective(
        profile.reference, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    aligned_valid = cv2.warpPerspective(
        profile.valid, matrix, (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    aligned_tolerance = cv2.warpPerspective(
        profile.tolerance, matrix, (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    aligned_tolerance[aligned_valid == 0] = 0
    diagnostics = {
        "matches": len(matches),
        "inliers": int(selected.sum()),
        "reprojection_median_px": round(float(np.median(errors)), 3),
        "reprojection_p95_px": round(float(np.percentile(errors, 95)), 3),
        "inlier_hull_fraction": round(float(hull_fraction), 4),
    }
    return aligned_reference, aligned_valid, aligned_tolerance, diagnostics


def residual_maps(reference: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    old = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    new = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def signature(gray: np.ndarray) -> np.ndarray:
        mean = cv2.GaussianBlur(gray, (0, 0), 12)
        variance = np.maximum(
            cv2.GaussianBlur(gray * gray, (0, 0), 12) - mean * mean, 0
        )
        return (gray - mean) / (np.sqrt(variance) + 5)

    signature_residual = np.abs(signature(new) - signature(old))
    delta = new - old
    luminance_residual = np.abs(delta - cv2.GaussianBlur(delta, (0, 0), 12))
    return signature_residual, luminance_residual


def _robust_global_color(
    reference: np.ndarray, current: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    old = reference.astype(np.float32)
    new = current.astype(np.float32)
    eligible = valid.astype(bool)
    eligible &= np.all((old > 5) & (old < 250) & (new > 5) & (new < 250), axis=2)
    indices = np.flatnonzero(eligible)
    if indices.size < 5000:
        raise ValueError("Clean Reference可信地面不足")
    if indices.size > 250_000:
        indices = indices[
            np.linspace(0, indices.size - 1, 250_000).astype(np.intp)
        ]
    # The robust fit is intentionally based on at most 250k pixels.  Keep the
    # iterative prediction and residual calculation on that same sample; the
    # previous implementation rebuilt a full-resolution three-channel
    # prediction on every iteration only to index it back down immediately.
    # The final normalised reference is still produced at full resolution.
    old_sample = old.reshape(-1, 3)[indices]
    new_sample = new.reshape(-1, 3)[indices]
    keep = np.ones(indices.size, bool)
    fits: list[tuple[float, float]] = []
    for _ in range(3):
        fits = []
        for channel in range(3):
            x = old_sample[keep, channel]
            y = new_sample[keep, channel]
            gain, bias = np.polyfit(x, y, 1)
            fits.append((float(np.clip(gain, 0.65, 1.45)),
                         float(np.clip(bias, -60.0, 60.0))))
        prediction = np.stack(
            [old_sample[:, channel] * fits[channel][0] + fits[channel][1]
             for channel in range(3)], axis=1,
        )
        residual = np.max(np.abs(new_sample - prediction), axis=1)
        cutoff = max(float(np.percentile(residual[keep], 70)), 8.0)
        keep = residual <= cutoff
    normalized = np.stack(
        [old[..., channel] * fits[channel][0] + fits[channel][1]
         for channel in range(3)], axis=2,
    )
    return normalized, fits


def _component_mask(binary: np.ndarray, *, min_area: int, min_side: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), 8
    )
    result = np.zeros_like(binary, np.uint8)
    for index in range(1, count):
        _x, _y, width, height, area = (int(value) for value in stats[index])
        if area >= min_area or (area >= min_area // 3 and max(width, height) >= min_side):
            result[labels == index] = 255
    return result


def protected_normalize(
    reference: np.ndarray, current: np.ndarray, valid: np.ndarray,
    *, pixel_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    masks = compute_compensation_masks(
        reference, current, valid, pixel_scale=pixel_scale,
    )
    field = local_luminance_field(
        masks["compensated"], current, masks["protected"],
        masks["fit_mask"], pixel_scale=pixel_scale,
    )
    normalized = np.clip(
        masks["global_reference"] + field[..., None], 0, 255,
    ).astype(np.uint8)
    return normalized, valid.copy(), _classify_environment(masks, field, valid)


def compute_compensation_masks(
    reference: np.ndarray, current: np.ndarray, valid: np.ndarray,
    *, pixel_scale: float = 1.0,
) -> dict[str, Any]:
    """纯函数：全局补偿、保护区域与三个拟合/饱和度掩膜。

    与 ``protected_normalize`` 共用同一段实现，供 Bank adapter 与工厂离线阶段
    复用；它**只**计算拟合/评分所需的掩膜，绝不产生可用性结论（方案二 §5.1）。
    返回的 ``protected``/``broad`` 是拟合排除区域，不是检测不可用区域。
    """
    global_reference, fits = _robust_global_color(reference, current, valid)
    clipped = np.clip(global_reference, 0, 255).astype(np.uint8)
    signature, luminance = residual_maps(clipped, current)
    preliminary = (
        (signature >= PROTECTED_SEED_SIGNATURE)
        & (luminance >= PROTECTED_SEED_LUMINANCE)
        & (valid > 0)
    ).astype(np.uint8)
    protected = cv2.dilate(
        preliminary,
        np.ones((
            _scaled_odd(PRELIMINARY_DILATION, pixel_scale),
            _scaled_odd(PRELIMINARY_DILATION, pixel_scale),
        ), np.uint8),
    )
    broad = _component_mask(
        preliminary,
        min_area=max(1, round(TEMPORARY_UNAVAILABLE_MIN_AREA * pixel_scale**2)),
        min_side=max(1, round(TEMPORARY_UNAVAILABLE_MIN_SIDE * pixel_scale)),
    )
    if broad.any():
        broad_kernel = _scaled_odd(21, pixel_scale)
        broad = cv2.dilate(
            broad, np.ones((broad_kernel, broad_kernel), np.uint8)
        )

    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    saturated = ((current_gray <= 3) | (current_gray >= 252)).astype(np.uint8) * 255
    saturated_kernel = _scaled_odd(SATURATION_DILATION, pixel_scale)
    saturated = cv2.dilate(
        saturated, np.ones((saturated_kernel, saturated_kernel), np.uint8)
    )
    fit_mask = valid.copy()
    fit_mask[protected > 0] = 0
    fit_mask[saturated > 0] = 0
    return {
        "global_reference": global_reference,
        "compensated": clipped,
        "fits": fits,
        "preliminary": preliminary,
        "protected": protected,
        "broad": broad,
        "saturated": saturated,
        "fit_mask": fit_mask,
        "signature": signature,
        "luminance": luminance,
    }


def local_luminance_field(
    compensated: np.ndarray, current: np.ndarray, protected: np.ndarray,
    fit_mask: np.ndarray, *, pixel_scale: float = 1.0,
) -> np.ndarray:
    """受限局部亮度场：拟合掩膜内估计，保护区域回落到直接高斯场。"""
    height, width = fit_mask.shape
    small_width = max(1, math.ceil(width / LOCAL_FIELD_DOWNSAMPLE))
    small_height = max(1, math.ceil(height / LOCAL_FIELD_DOWNSAMPLE))
    old_luma = cv2.cvtColor(compensated, cv2.COLOR_BGR2GRAY).astype(np.float32)
    new_luma = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
    raw_delta = new_luma - old_luma
    delta = np.clip(raw_delta, -LOCAL_FIELD_CLIP, LOCAL_FIELD_CLIP)
    values = delta[fit_mask > 0]
    fallback = float(np.median(values)) if len(values) >= 5000 else 0.0
    size = (small_width, small_height)
    small_delta = cv2.resize(delta, size, interpolation=cv2.INTER_AREA)
    small_weight = cv2.resize(
        (fit_mask > 0).astype(np.float32), size, interpolation=cv2.INTER_AREA
    )
    sigma = LOCAL_FIELD_BLUR_SIGMA * pixel_scale / LOCAL_FIELD_DOWNSAMPLE
    weight = cv2.GaussianBlur(small_weight, (0, 0), sigma)
    weighted = cv2.GaussianBlur(small_delta * small_weight, (0, 0), sigma)
    field_small = weighted / np.maximum(weight, 1e-3)
    field_small[weight < 0.04] = fallback
    field = cv2.resize(field_small, (width, height), interpolation=cv2.INTER_CUBIC)
    default_field = cv2.GaussianBlur(
        raw_delta, (0, 0), LOCAL_FIELD_BLUR_SIGMA * pixel_scale
    )
    field[protected > 0] = default_field[protected > 0]
    return np.clip(field, -LOCAL_FIELD_CLIP, LOCAL_FIELD_CLIP)


def _classify_environment(
    masks: dict[str, Any], field: np.ndarray, valid: np.ndarray,
) -> dict[str, Any]:
    """环境状态分类（与旧实现的阈值完全一致）。"""
    valid_mask = valid
    valid_pixels = max(int(np.count_nonzero(valid_mask)), 1)
    saturated = masks["saturated"]
    saturated_fraction = float(
        np.count_nonzero((saturated > 0) & (valid_mask > 0)) / valid_pixels
    )
    fits = masks["fits"]
    gains = [pair[0] for pair in fits]
    biases = [pair[1] for pair in fits]
    valid_field = field[valid_mask > 0]
    if valid_field.size == 0:
        local_extent = 0.0
    else:
        local_extent = max(
            abs(float(np.percentile(valid_field, 5))),
            abs(float(np.percentile(valid_field, 95))),
        )
    if saturated_fraction > 0.35:
        state = "ENVIRONMENT_CHANGE"
    elif (max(abs(gain - 1.0) for gain in gains) > 0.10
          or max(abs(bias) for bias in biases) > 18
          or local_extent > 16):
        state = "GLOBAL_LIGHT_CHANGE"
    else:
        state = "NORMAL"
    return {
        "state": state,
        "saturated_fraction": round(saturated_fraction, 4),
        "gains": [round(float(gain), 5) for gain in gains],
        "biases": [round(float(bias), 5) for bias in biases],
        "local_extent": round(float(local_extent), 4),
        "max_gain_delta": round(max(abs(gain - 1.0) for gain in gains), 5),
        "max_abs_bias": round(max(abs(bias) for bias in biases), 5),
    }



def propose_v32(
    normalized_reference: np.ndarray,
    current: np.ndarray,
    valid: np.ndarray,
    tolerance: np.ndarray,
    *,
    pixel_scale: float = 1.0,
) -> tuple[list[dict[str, Any]], np.ndarray]:
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
        _x, _y, width, height, area = (int(value) for value in stats[index])
        if (
            area >= round(900 * pixel_scale**2)
            or (
                area >= round(350 * pixel_scale**2)
                and max(width, height) >= round(80 * pixel_scale)
            )
        ):
            broad[labels == index] = 255
    if broad.any():
        broad_kernel = _scaled_odd(15, pixel_scale)
        broad = cv2.dilate(
            broad, np.ones((broad_kernel, broad_kernel), np.uint8)
        )
        seed[broad > 0] = 0
        support[broad > 0] = 0
    marker_kernel = _scaled_odd(9, pixel_scale)
    marker = cv2.dilate(
        seed, np.ones((marker_kernel, marker_kernel), np.uint8)
    )
    grown = cv2.bitwise_and(support, marker)
    close_kernel = _scaled_odd(5, pixel_scale)
    grown = cv2.morphologyEx(
        grown, cv2.MORPH_CLOSE,
        np.ones((close_kernel, close_kernel), np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grown, 8)
    rows: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        component = labels == index
        seed_pixels = int(seed[component].sum())
        if seed_pixels < max(1, round(MINIMUM_SEED_PIXELS * pixel_scale**2)):
            continue
        if (
            area < max(1, round(MINIMUM_SUPPORT_AREA * pixel_scale**2))
            or min(width, height) < max(1, round(MINIMUM_SUPPORT_SHORT_SIDE * pixel_scale))
        ):
            continue
        if (
            max(width, height) > round(MAXIMUM_SUPPORT_SIDE * pixel_scale)
            or width * height > round(MAXIMUM_SUPPORT_BOX_AREA * pixel_scale**2)
        ):
            continue
        rows.append({
            "box": [x, y, x + width, y + height],
            "anomaly_score": round(float(min(
                1.0,
                0.4 * seed_pixels / 12.0
                + 0.3 * np.percentile(signature[component], 90) / 4.0
                + 0.3 * np.percentile(luminance[component], 90) / 150.0,
            )), 4),
        })
    rows.sort(key=lambda row: -float(row["anomaly_score"]))
    return rows, support


def _box_center(box: Iterable[float]) -> tuple[float, float]:
    left, top, right, bottom = box
    return (left + right) / 2.0, (top + bottom) / 2.0


def _box_area(box: Iterable[float]) -> float:
    left, top, right, bottom = box
    return max(0.0, right - left) * max(0.0, bottom - top)


def _same_anchor(
    left: Iterable[float], right: Iterable[float], *, distance: float, size_ratio: float
) -> bool:
    left_values, right_values = list(left), list(right)
    areas = max(_box_area(left_values), 1.0), max(_box_area(right_values), 1.0)
    return (
        max(areas) / min(areas) <= size_ratio
        and math.dist(_box_center(left_values), _box_center(right_values)) <= distance
    )


def _max_actor_overlap(box: Iterable[float], actors: Iterable[Iterable[float]]) -> float:
    return max((box_overlap_fraction(box, actor) for actor in actors), default=0.0)


def _context_external_support(
    support: np.ndarray, valid: np.ndarray, box: Iterable[float],
    *, pixel_scale: float = 1.0,
) -> int:
    height, width = support.shape
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    margin = max(1, round(CONTEXT_MARGIN_PX * pixel_scale))
    touch_gap = max(1, round(CONTEXT_TOUCH_GAP_PX * pixel_scale))
    guard = max(1, round(CONTEXT_CANDIDATE_GUARD_PX * pixel_scale))
    left = max(0, center_x - margin)
    top = max(0, center_y - margin)
    right = min(width, center_x + margin + 1)
    bottom = min(height, center_y + margin + 1)
    local = ((support[top:bottom, left:right] > 0)
             & (valid[top:bottom, left:right] > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(local, 8)
    guard_left = max(left, x1 - guard) - left
    guard_top = max(top, y1 - guard) - top
    guard_right = min(right, x2 + guard) - left
    guard_bottom = min(bottom, y2 + guard) - top
    outside = np.ones(local.shape, bool)
    outside[guard_top:guard_bottom, guard_left:guard_right] = False
    maximum = 0
    for component in range(1, count):
        component_left = left + int(stats[component, cv2.CC_STAT_LEFT])
        component_top = top + int(stats[component, cv2.CC_STAT_TOP])
        component_right = component_left + int(stats[component, cv2.CC_STAT_WIDTH])
        component_bottom = component_top + int(stats[component, cv2.CC_STAT_HEIGHT])
        touches = (
            component_left < x2 + touch_gap
            and component_right > x1 - touch_gap
            and component_top < y2 + touch_gap
            and component_bottom > y1 - touch_gap
        )
        if touches:
            maximum = max(maximum, int(np.count_nonzero(
                (labels == component) & outside
            )))
    return maximum


def _anchor_observation(
    support: np.ndarray, valid: np.ndarray, box: Iterable[float],
    *, pixel_scale: float = 1.0,
) -> tuple[float, int]:
    height, width = support.shape
    left, top, right, bottom = (int(round(value)) for value in box)
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if right <= left or bottom <= top:
        return 0.0, 0
    anchor_valid = valid[top:bottom, left:right] > 0
    valid_fraction = float(np.count_nonzero(anchor_valid) / anchor_valid.size)
    pad = max(1, round(ANCHOR_SUPPORT_PAD_PX * pixel_scale))
    x1, y1 = max(0, left - pad), max(0, top - pad)
    x2, y2 = min(width, right + pad), min(height, bottom + pad)
    support_pixels = int(np.count_nonzero(
        (support[y1:y2, x1:x2] > 0) & (valid[y1:y2, x1:x2] > 0)
    ))
    return valid_fraction, support_pixels


@dataclass(slots=True)
class V32Event:
    event_id: int
    first_seen: float
    anchor_box: list[float]
    state: str = "ANOMALY_PENDING"
    last_visible: float | None = None
    visible_timestamps: list[float] = field(default_factory=list)
    visible_observation_count: int = 0
    visible_evidence_seconds: float = 0.0
    boxes: list[list[float]] = field(default_factory=list)
    state_history: list[dict[str, Any]] = field(default_factory=list)
    clear_observed_seconds: float = 0.0
    last_clean_observed_at: float | None = None
    pending_unmatched_seconds: float = 0.0
    confirmed_at: float | None = None
    closed_at: float | None = None
    closed_reason: str | None = None
    confidence: float = 0.0
    region_id: str = ""

    def transition(self, timestamp: float, state: str, reason: str) -> None:
        if self.state != state or not self.state_history:
            self.state_history.append({
                "timestamp": round(float(timestamp), 3),
                "state": state,
                "reason": reason,
            })
            self.state_history = self.state_history[-MAX_STATE_HISTORY:]
        self.state = state

    def reset_clear(self) -> None:
        self.clear_observed_seconds = 0.0
        self.last_clean_observed_at = None

    def observe(
        self, timestamp: float, candidate: dict[str, Any], sample_period: float,
        confirm_seconds: float,
    ) -> None:
        box = [float(value) for value in candidate["box"]]
        self.boxes.append(box)
        self.boxes = self.boxes[-MAX_BOX_HISTORY:]
        self.anchor_box = np.median(np.asarray(self.boxes), axis=0).tolist()
        if self.last_visible != timestamp:
            self.visible_timestamps.append(timestamp)
            self.visible_timestamps = self.visible_timestamps[-MAX_VISIBLE_TIMESTAMP_HISTORY:]
            self.visible_observation_count += 1
            self.visible_evidence_seconds += sample_period
        self.last_visible = timestamp
        self.pending_unmatched_seconds = 0.0
        self.confidence = float(candidate.get("anomaly_score", self.confidence))
        self.region_id = str(candidate.get("region_id", self.region_id))
        self.reset_clear()
        if self.confirmed_at is None and self.visible_evidence_seconds >= confirm_seconds:
            self.confirmed_at = timestamp
        self.transition(timestamp, "VISIBLE_ANOMALY", "candidate_visible")

    def observe_clean(self, timestamp: float, sample_period: float, max_gap: float) -> None:
        previous = self.last_clean_observed_at
        if previous is None or timestamp <= previous or timestamp - previous > max_gap:
            self.clear_observed_seconds = sample_period
        else:
            self.clear_observed_seconds += sample_period
        self.last_clean_observed_at = timestamp

    def absorb(self, other: "V32Event", timestamp: float, sample_period: float) -> None:
        self.first_seen = min(self.first_seen, other.first_seen)
        self.boxes.extend(other.boxes)
        self.boxes = self.boxes[-MAX_MERGED_BOX_HISTORY:]
        self.anchor_box = np.median(np.asarray(self.boxes), axis=0).tolist()
        union = sorted(set(self.visible_timestamps + other.visible_timestamps))
        self.visible_timestamps = union[-MAX_VISIBLE_TIMESTAMP_HISTORY:]
        self.visible_observation_count = len(union)
        self.visible_evidence_seconds = len(union) * sample_period
        values = [value for value in (self.last_visible, other.last_visible) if value is not None]
        self.last_visible = max(values) if values else None
        confirmed = [value for value in (self.confirmed_at, other.confirmed_at) if value is not None]
        self.confirmed_at = min(confirmed) if confirmed else None
        self.confidence = max(self.confidence, other.confidence)
        self.reset_clear()
        self.transition(timestamp, self.state, f"merged_event_{other.event_id}")


class V32EventMemory:
    def __init__(
        self, options: GroundLitterDetectionOptions, *, pixel_scale: float = 1.0
    ) -> None:
        self.options = options
        self.pixel_scale = float(pixel_scale)
        self.sample_period = 1.0 / float(options.analysis_fps)
        self.max_sample_gap = self.sample_period * MAX_SAMPLE_GAP_FACTOR
        self.events: list[V32Event] = []
        self._next_id = 1
        self._last_timestamp: float | None = None

    @property
    def active(self) -> list[V32Event]:
        return [event for event in self.events if event.closed_at is None]

    def update(
        self, *, timestamp: float, candidates: list[dict[str, Any]],
        support: np.ndarray, valid: np.ndarray, actors: list[list[float]],
        environment_state: str,
    ) -> None:
        timestamp = float(timestamp)
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("V3.2时间戳必须严格递增")
        self._last_timestamp = timestamp
        if environment_state != "NORMAL":
            for event in self.active:
                event.reset_clear()
                event.transition(timestamp, "ENVIRONMENT_CHANGE", "frame_unavailable")
            return

        visible: list[dict[str, Any]] = []
        for candidate in candidates:
            if _context_external_support(
                support, valid, candidate["box"], pixel_scale=self.pixel_scale
            ) >= max(1, round(CONTEXT_OCCLUSION_EXTERNAL_AREA_PX * self.pixel_scale**2)):
                continue
            if _max_actor_overlap(candidate["box"], actors) > self.options.actor_overlap_threshold:
                continue
            visible.append(candidate)

        active = self.active
        pairs: list[tuple[float, int, int]] = []
        for event_index, event in enumerate(active):
            for candidate_index, candidate in enumerate(visible):
                if _same_anchor(event.anchor_box, candidate["box"],
                                distance=EVENT_MATCH_DISTANCE_PX,
                                size_ratio=EVENT_MATCH_SIZE_RATIO):
                    pairs.append((math.dist(_box_center(event.anchor_box),
                                            _box_center(candidate["box"])),
                                  event_index, candidate_index))
        used_events: set[int] = set()
        used_candidates: set[int] = set()
        for _distance, event_index, candidate_index in sorted(pairs):
            if event_index in used_events or candidate_index in used_candidates:
                continue
            active[event_index].observe(
                timestamp, visible[candidate_index], self.sample_period,
                self.options.confirm_visible_seconds,
            )
            used_events.add(event_index)
            used_candidates.add(candidate_index)
        for candidate_index, candidate in enumerate(visible):
            if candidate_index in used_candidates:
                continue
            related = [
                event for event in active
                if _same_anchor(event.anchor_box, candidate["box"],
                                distance=EVENT_MATCH_DISTANCE_PX,
                                size_ratio=EVENT_MATCH_SIZE_RATIO)
            ]
            if related:
                event = min(related, key=lambda item: math.dist(
                    _box_center(item.anchor_box), _box_center(candidate["box"])
                ))
                event.boxes.append([float(value) for value in candidate["box"]])
                event.boxes = event.boxes[-MAX_BOX_HISTORY:]
                event.anchor_box = np.median(np.asarray(event.boxes), axis=0).tolist()
                used_candidates.add(candidate_index)
        for candidate_index, candidate in enumerate(visible):
            if candidate_index in used_candidates:
                continue
            event = V32Event(
                event_id=self._next_id,
                first_seen=timestamp,
                anchor_box=[float(value) for value in candidate["box"]],
            )
            event.observe(
                timestamp, candidate, self.sample_period,
                self.options.confirm_visible_seconds,
            )
            self.events.append(event)
            self._next_id += 1

        for event_index, event in enumerate(active):
            if event_index in used_events:
                continue
            context_occluded = (
                _context_external_support(
                    support, valid, event.anchor_box,
                    pixel_scale=self.pixel_scale,
                )
                >= max(1, round(
                    CONTEXT_OCCLUSION_EXTERNAL_AREA_PX * self.pixel_scale**2
                ))
            )
            actor_occluded = (
                _max_actor_overlap(event.anchor_box, actors)
                > self.options.actor_overlap_threshold
            )
            if context_occluded or actor_occluded:
                event.reset_clear()
                event.transition(timestamp, "OCCLUDED",
                                 "context_occluded" if context_occluded else "actor_occluded")
                continue
            valid_fraction, support_pixels = _anchor_observation(
                support, valid, event.anchor_box,
                pixel_scale=self.pixel_scale,
            )
            if valid_fraction < self.options.min_clean_valid_fraction:
                event.reset_clear()
                event.transition(timestamp, "GROUND_UNAVAILABLE", "insufficient_valid_ground")
                continue
            if event.confirmed_at is None and event.last_visible is not None:
                event.pending_unmatched_seconds += self.sample_period
            if (event.confirmed_at is None
                    and event.pending_unmatched_seconds >= self.options.pending_expire_seconds):
                event.reset_clear()
                event.closed_at = timestamp
                event.closed_reason = "pending_timeout"
                event.transition(timestamp, "EXPIRED_PENDING", "pending_timeout")
                continue
            if support_pixels <= max(
                1, round(CLEAN_MAX_SUPPORT_PIXELS * self.pixel_scale**2)
            ):
                event.observe_clean(timestamp, self.sample_period, self.max_sample_gap)
                event.transition(timestamp, "CLEAN_PENDING", "clean_reference_match")
                if event.clear_observed_seconds >= self.options.clear_confirm_seconds:
                    event.closed_at = timestamp
                    event.closed_reason = "clean_confirmed"
                    event.transition(timestamp, "CLEARED", "clean_confirmed")
                continue
            event.reset_clear()
            event.transition(timestamp, "ANOMALY_PENDING", "residual_without_candidate")
        self._merge(timestamp)
        closed = sorted(
            (event for event in self.events if event.closed_at is not None),
            key=lambda event: (event.closed_at or 0.0, event.event_id),
        )
        excess = len(closed) - self.options.maximum_closed_events
        if excess > 0:
            removed = {event.event_id for event in closed[:excess]}
            self.events = [event for event in self.events if event.event_id not in removed]

    def _merge(self, timestamp: float) -> None:
        active = sorted(self.active, key=lambda event: event.event_id)
        consumed: set[int] = set()
        for index, primary in enumerate(active):
            if primary.event_id in consumed:
                continue
            for secondary in active[index + 1:]:
                if secondary.event_id in consumed:
                    continue
                if not _same_anchor(primary.anchor_box, secondary.anchor_box,
                                    distance=EVENT_MERGE_DISTANCE_PX,
                                    size_ratio=EVENT_MERGE_SIZE_RATIO):
                    continue
                primary.absorb(secondary, timestamp, self.sample_period)
                if (primary.confirmed_at is None
                        and primary.visible_evidence_seconds >= self.options.confirm_visible_seconds):
                    primary.confirmed_at = timestamp
                secondary.closed_at = timestamp
                secondary.closed_reason = f"merged_into:{primary.event_id}"
                secondary.transition(timestamp, "MERGED", secondary.closed_reason)
                consumed.add(secondary.event_id)


def assign_prior_regions(
    candidates: list[dict[str, Any]],
    zones: Iterable[Any],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Keep prior candidates whose centre is inside a zone and outside exclusions."""
    zones = tuple(zones)
    if not zones:
        return candidates
    polygons = [
        (
            zone,
            np.round(np.asarray(zone.polygon) * np.asarray([width, height])).astype(np.int32),
            [
                np.round(
                    np.asarray(exclusion) * np.asarray([width, height])
                ).astype(np.int32)
                for exclusion in zone.exclude_zones
            ],
        )
        for zone in zones
    ]
    retained: list[dict[str, Any]] = []
    for candidate in candidates:
        left, top, right, bottom = candidate["box"]
        center = ((left + right) / 2.0, (top + bottom) / 2.0)
        eligible = [
            zone for zone, polygon, exclusions in polygons
            if cv2.pointPolygonTest(polygon, center, False) >= 0
            and not any(
                cv2.pointPolygonTest(exclusion, center, False) >= 0
                for exclusion in exclusions
            )
            and min(right - left, bottom - top) >= zone.minimum_short_side_px
            and (right - left) * (bottom - top) >= zone.minimum_box_area_px
        ]
        if not eligible:
            continue
        item = dict(candidate)
        item["region_id"] = eligible[0].region_id if len(eligible) == 1 else ""
        retained.append(item)
    return retained


@dataclass(frozen=True, slots=True)
class CleanReferenceFrameAnalysis:
    """One frame of Clean Reference analysis with no event-memory side effects.

    V3.2's processor and the V3.3 prior channel both consume this so the two can
    never drift into two different normalization implementations. Candidate
    proposal is a separate, explicit step (``propose_prior_candidates``) because
    V3.2 deliberately skips it during warmup and while the stability gate holds.
    """

    analysis_frame: np.ndarray
    normalized: np.ndarray
    valid: np.ndarray
    tolerance: np.ndarray
    actor_rows: list[list[float]]
    environment_state: str
    environment: dict[str, Any]
    alignment: dict[str, Any]
    width: int
    height: int
    pixel_scale: float


def analyze_prior_frame(
    profile: CleanReferenceProfileV32,
    options: Any,
    frame: np.ndarray,
    *,
    aligned: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    actors: Iterable[Iterable[float]] = (),
) -> tuple[CleanReferenceFrameAnalysis, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Align and normalize a single frame; no candidates, no event memory.

    Returns the analysis plus the aligned ``(reference, valid, tolerance)`` triple
    so the caller can cache it: alignment is the expensive SIFT step and is stable
    for a fixed camera.
    """
    pixel_scale = profile.reference.shape[1] / 2560.0
    input_height, input_width = frame.shape[:2]
    reference_height, reference_width = profile.reference.shape[:2]
    input_ratio = input_width / max(input_height, 1)
    reference_ratio = reference_width / max(reference_height, 1)
    if abs(input_ratio - reference_ratio) > 0.002:
        raise ValueError(
            "Clean Reference画面比例不匹配: "
            f"期望{reference_width}x{reference_height}，"
            f"实际{input_width}x{input_height}"
        )
    if (input_width, input_height) == (reference_width, reference_height):
        analysis_frame = frame
        actor_rows = [list(map(float, box)) for box in actors]
    else:
        analysis_frame = cv2.resize(
            frame,
            (reference_width, reference_height),
            interpolation=(
                cv2.INTER_CUBIC
                if input_width < reference_width
                else cv2.INTER_AREA
            ),
        )
        scale_x = reference_width / input_width
        scale_y = reference_height / input_height
        actor_rows = [
            [
                float(box[0]) * scale_x,
                float(box[1]) * scale_y,
                float(box[2]) * scale_x,
                float(box[3]) * scale_y,
            ]
            for box in actors
            if len(box) >= 4
        ]
    alignment: dict[str, Any] = {}
    if aligned is None:
        reference, valid, tolerance, alignment = align_profile(profile, analysis_frame)
        aligned = (reference, valid, tolerance)
    reference, valid, tolerance = aligned
    normalized, usable, environment = protected_normalize(
        reference, analysis_frame, valid, pixel_scale=pixel_scale
    )
    height, width = analysis_frame.shape[:2]
    return (
        CleanReferenceFrameAnalysis(
            analysis_frame=analysis_frame,
            normalized=normalized,
            valid=usable,
            tolerance=tolerance,
            actor_rows=actor_rows,
            environment_state=str(environment.get("state", "")),
            environment=dict(environment),
            alignment=alignment,
            width=width,
            height=height,
            pixel_scale=pixel_scale,
        ),
        aligned,
    )


def propose_prior_candidates(
    options: Any,
    analysis: CleanReferenceFrameAnalysis,
) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    """Propose and region-assign prior candidates for an analysed frame.

    Returns ``(candidates, support, proposed)`` where ``proposed`` is the count
    *before* the ROI/exclusion/min-size filter, so observability can report how
    many proposals the geometry suppressed. Returns empty candidates when the
    environment is not ``NORMAL``: neither V3.2 nor the V3.3 prior channel uses
    candidates in that state.
    """
    if analysis.environment_state != "NORMAL":
        return [], np.zeros((analysis.height, analysis.width), np.uint8), 0
    proposed, support = propose_v32(
        analysis.normalized,
        analysis.analysis_frame,
        analysis.valid,
        analysis.tolerance,
        pixel_scale=analysis.pixel_scale,
    )
    retained = assign_prior_regions(
        proposed, getattr(options, "zones", ()),
        analysis.width, analysis.height,
    )
    return retained, support, len(proposed)


class CleanReferenceV32Processor:
    """One fixed camera's V3.2 analysis and confirmed-event projection."""

    def __init__(
        self,
        options: GroundLitterDetectionOptions,
        profile: CleanReferenceProfileV32,
    ) -> None:
        self.options = options
        self.profile = profile
        self.pixel_scale = profile.reference.shape[1] / 2560.0
        self.memory = V32EventMemory(options, pixel_scale=self.pixel_scale)
        self._aligned: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._alignment: dict[str, Any] = {}
        self._version = 0
        self._started_at: float | None = None
        self._normal_streak = 0

    @property
    def alignment(self) -> dict[str, Any]:
        return dict(self._alignment)

    def update(
        self, frame: np.ndarray, *, timestamp: float,
        actors: Iterable[Iterable[float]] = (),
    ) -> GroundLitterSnapshot:
        timestamp = float(timestamp)
        if self._started_at is None:
            self._started_at = timestamp
        input_height, input_width = frame.shape[:2]
        analysis, aligned = analyze_prior_frame(
            self.profile,
            self.options,
            frame,
            aligned=self._aligned,
            actors=actors,
        )
        self._aligned = aligned
        if analysis.alignment:
            self._alignment = analysis.alignment
        analysis_frame = analysis.analysis_frame
        actor_rows = analysis.actor_rows
        usable = analysis.valid
        environment_state = analysis.environment_state
        warmup_elapsed = max(0.0, timestamp - self._started_at)
        warmup_remaining = max(
            0.0,
            float(self.options.startup_suppress_seconds) - warmup_elapsed,
        )
        if warmup_remaining > 0:
            self._normal_streak = 0
            self._version += 1
            return GroundLitterSnapshot(
                state="warming_up",
                result_version=self._version,
                updated_at=timestamp,
                message=(
                    f"clean_reference_v32 profile={self.profile.profile_id} "
                    f"startup_suppressed remaining={warmup_remaining:.1f}s "
                    f"environment={environment_state}"
                ),
                environment_state=environment_state,
            )
        if environment_state != "NORMAL":
            self._normal_streak = 0
            self.memory = V32EventMemory(
                self.options, pixel_scale=self.pixel_scale
            )
            self._version += 1
            return GroundLitterSnapshot(
                state="abstaining",
                result_version=self._version,
                updated_at=timestamp,
                message=(
                    f"clean_reference_v32 profile={self.profile.profile_id} "
                    f"environment={environment_state} memory_reset=true"
                ),
                environment_state=environment_state,
            )
        self._normal_streak += 1
        if self._normal_streak < int(self.options.normal_stability_samples):
            self._version += 1
            return GroundLitterSnapshot(
                state="abstaining",
                result_version=self._version,
                updated_at=timestamp,
                message=(
                    f"clean_reference_v32 profile={self.profile.profile_id} "
                    f"environment=NORMAL stability={self._normal_streak}/"
                    f"{self.options.normal_stability_samples}"
                ),
                environment_state=environment_state,
            )
        candidates, support, _proposed = propose_prior_candidates(
            self.options, analysis
        )
        self.memory.update(
            timestamp=timestamp, candidates=candidates, support=support,
            valid=usable, actors=actor_rows,
            environment_state=environment_state,
        )
        height, width = analysis.height, analysis.width
        detections: list[GroundLitterDetection] = []
        for event in self.memory.active:
            if event.confirmed_at is None:
                continue
            if event.state not in {"VISIBLE_ANOMALY", "ANOMALY_PENDING"}:
                continue
            left, top, right, bottom = event.anchor_box
            detections.append(GroundLitterDetection(
                object_id=event.event_id,
                rectangle=NormalizedRect(
                    left / width, top / height,
                    max(0.0, right - left) / width,
                    max(0.0, bottom - top) / height,
                ),
                confidence=event.confidence,
                class_name="",
                region_id=event.region_id,
                hits=event.visible_observation_count,
                source="clean_reference_v32",
            ))
        detections.sort(key=lambda item: (-item.confidence, item.object_id))
        self._version += 1
        confirmed = sum(event.confirmed_at is not None for event in self.memory.events)
        cleared = sum(event.closed_reason == "clean_confirmed" for event in self.memory.events)
        message = (
            f"clean_reference_v32 profile={self.profile.profile_id} "
            f"environment={environment_state} active={len(self.memory.active)} "
            f"confirmed={confirmed} raw={len(candidates)}"
        )
        return GroundLitterSnapshot(
            state="running",
            detections=tuple(detections[: self.options.maximum_boxes]),
            result_version=self._version,
            updated_at=float(timestamp),
            message=message,
            raw_candidates=len(candidates),
            tile_count=0,
            active_events=len(self.memory.active),
            confirmed_events=confirmed,
            cleared_events=cleared,
            environment_state=environment_state,
        )

    def _assign_regions(
        self, candidates: list[dict[str, Any]], width: int, height: int
    ) -> list[dict[str, Any]]:
        return assign_prior_regions(
            candidates, self.options.zones, width, height
        )
