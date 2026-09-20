"""A4/B2：Bank prior adapter、三类掩膜与静态覆盖（方案一 §7、方案二 §3.4/§5.1）。

本模块解决设计复核 F2：``fit_mask`` / ``foreground_support`` / ``availability_mask``
必须是**三种不同含义**的数据，不能复用一个「异常区域」掩膜。

* ``fit_mask``：稳定锚点与共同可见性，仅用于背景拟合/评分。拟合排除**不等于**检测不可用。
* ``foreground_support``：在可判断区域内按**冻结的离线噪声阈值**产生的残差证据。
* ``availability_mask``：ROI ∩ 离线有效资产 ∩ 配准后有效视野，再扣除独立证据
  （actor 遮挡、坏帧、配准失败、大范围成像损失）。原因必须可枚举；
  **没有** ``LIVE_RESIDUAL_TOO_HIGH`` 这种逐像素失效规则。

候选的可用性由 ``prior_available`` 与本帧是否可判断共同决定，二者不能合并。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from . import ground_litter_v32 as v32
from .ground_litter_profile_bank import BankError, ProfileRecord
from .ground_litter_profile_match import (
    ColorCompensation, apply_compensation, evaluate_match, residual_maps,
    robust_color_compensation, score_profile,
)

AVAILABILITY_REASONS = (
    "OK",
    "OUTSIDE_ROI",
    "PROFILE_ASSET_UNSUPPORTED",
    "GEOMETRY_INVALID",
    "FRAME_CORRUPT",
    "ACTOR_OCCLUDED",
    "LARGE_EXPOSURE_LOSS",
    "NO_COMMON_VISIBILITY",
)

# 「大范围」判据：空间尺度必须显著超过目标尺度（方案二 §5.1）。
LARGE_LOSS_MIN_SIDE_FACTOR = 2.0
LARGE_LOSS_MIN_AREA_FACTOR = 4.0
MAX_PRIOR_TARGET_SIDE_PX = v32.MAXIMUM_SUPPORT_SIDE
MAX_PRIOR_TARGET_AREA_PX = v32.MAXIMUM_SUPPORT_BOX_AREA


@dataclass(frozen=True, slots=True)
class BankPriorContext:
    """一个候选参考的全尺寸上下文（坐标系：Bank 共同画布）。"""

    profile_id: str
    reference: np.ndarray
    valid: np.ndarray
    noise: dict[str, np.ndarray]
    metadata: dict[str, Any]
    geometry_diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def reference_size(self) -> tuple[int, int]:
        return int(self.reference.shape[1]), int(self.reference.shape[0])

    def threshold(self, name: str, shape: tuple[int, int]) -> np.ndarray:
        """读取冻结阈值图；缺失或尺寸不符按低支持回退基础阈值。"""
        array = self.noise.get(name)
        if array is None or tuple(array.shape) != shape:
            base = _FALLBACK_BASE.get(name)
            if base is None:
                raise BankError(f"noise.npz 缺少 {name} 且没有基础阈值")
            return np.full(shape, base, np.float32)
        return np.asarray(array, np.float32)

    def foreground_support(self, current: np.ndarray) -> np.ndarray:
        """按冻结的离线阈值产生前景证据掩膜（0/1 uint8）。

        直接用 candidate 的 signature/luminance 残差与 noise 阈值图比较；
        **不**在直播中抬高阈值，也不因残差大而删除该像素。
        """
        signature, luminance = residual_maps(self.reference, current)
        shape = signature.shape
        seed_sig = self.threshold("seed_signature_threshold", shape)
        seed_lum = self.threshold("seed_luminance_threshold", shape)
        support_sig = self.threshold("support_signature_threshold", shape)
        support_lum = self.threshold("support_luminance_threshold", shape)
        valid = self.valid > 0
        seed = (
            (signature >= seed_sig) & (luminance >= seed_lum) & valid
        ).astype(np.uint8)
        support = (
            (signature >= support_sig) & (luminance >= support_lum) & valid
        ).astype(np.uint8)
        marker = cv2.dilate(
            seed, np.ones((_scaled_odd(9, 1.0), _scaled_odd(9, 1.0)), np.uint8)
        )
        grown = cv2.bitwise_and(support, marker)
        return cv2.morphologyEx(
            grown, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )


_FALLBACK_BASE = {
    "seed_signature_threshold": v32.SIGNATURE_THRESHOLD,
    "seed_luminance_threshold": v32.LUMINANCE_THRESHOLD,
    "support_signature_threshold": v32.SUPPORT_SIGNATURE_THRESHOLD,
    "support_luminance_threshold": v32.SUPPORT_LUMINANCE_THRESHOLD,
}


def _scaled_odd(value: int, scale: float, *, minimum: int = 3) -> int:
    result = max(minimum, int(round(value * scale)))
    return result if result % 2 else result + 1


def build_prior_context(
    bank: Any, profile_id: str, *,
    geometry_diagnostics: Mapping[str, Any] | None = None,
) -> BankPriorContext:
    record: ProfileRecord = bank.profile(profile_id)
    return BankPriorContext(
        profile_id=profile_id,
        reference=bank.load_reference(profile_id),
        valid=bank.load_valid(profile_id),
        noise=bank.load_noise(profile_id),
        metadata=dict(record.metadata),
        geometry_diagnostics=dict(geometry_diagnostics or {}),
    )


def roi_mask_from_geometry(
    geometry: Mapping[str, Any], width: int, height: int,
) -> np.ndarray:
    """ROI ∩ 非排除区，作为 availability 的上限。"""
    mask = np.zeros((height, width), np.uint8)
    roi = geometry.get("roi") or []
    if roi:
        points = np.round(
            np.asarray(roi, np.float32) * np.asarray([width, height])
        ).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    else:
        mask[:] = 255
    for key in ("exclude_zones", "overlay_exclude_zones"):
        for polygon in geometry.get(key, []) or []:
            points = np.round(
                np.asarray(polygon, np.float32) * np.asarray([width, height])
            ).astype(np.int32)
            cv2.fillPoly(mask, [points], 0)
    return mask


@dataclass(frozen=True, slots=True)
class MaskSet:
    fit_mask: np.ndarray
    foreground_support: np.ndarray
    availability_mask: np.ndarray
    availability_reason: np.ndarray
    diagnostics: dict[str, Any]

    def reason_histogram(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for index, name in enumerate(AVAILABILITY_REASONS):
            total = int(np.count_nonzero(self.availability_reason == index))
            if total:
                counts[name] = total
        return counts


def compute_mask_set(
    context: BankPriorContext, current: np.ndarray, *,
    roi_mask: np.ndarray, actors: Sequence[Sequence[float]] = (),
    common_protection: np.ndarray | None = None,
    corruption: str | None = None, geometry_valid: bool = True,
    config: Mapping[str, Any] | None = None,
) -> MaskSet:
    """生成三类掩膜，并把每个像素的不可用原因编码进 ``availability_reason``。

    关键约束：目标尺度内的残差**不会**使像素不可用；只有独立证据
    （ROI 之外、资产不支持、配准失败、坏帧、actor 遮挡、大范围成像损失）才会。
    """
    config = config or {}
    height, width = current.shape[:2]
    if current.shape != context.reference.shape:
        raise BankError("当前帧与候选参考尺寸不一致")
    reasons = np.zeros((height, width), np.uint8)

    masks = v32.compute_compensation_masks(
        context.reference, current, context.valid, pixel_scale=1.0,
    )
    fit_mask = masks["fit_mask"].copy()
    if common_protection is not None:
        fit_mask[common_protection == 0] = 0

    available = np.zeros((height, width), bool)
    reasons[~((context.valid > 0) & (roi_mask > 0))] = (
        AVAILABILITY_REASONS.index("OUTSIDE_ROI")
    )
    asset_supported = (context.valid > 0) & (roi_mask > 0)
    reasons[asset_supported] = 0
    available = asset_supported.copy()

    if not geometry_valid:
        reasons[available] = AVAILABILITY_REASONS.index("GEOMETRY_INVALID")
        available[:] = False
    if corruption:
        reasons[available] = AVAILABILITY_REASONS.index("FRAME_CORRUPT")
        available[:] = False

    actor_pixels = np.zeros((height, width), np.uint8)
    for box in actors:
        left, top, right, bottom = (int(round(float(value))) for value in box)
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
        if right > left and bottom > top:
            actor_pixels[top:bottom, left:right] = 255
    if actor_pixels.any():
        occluded = (actor_pixels > 0) & available
        reasons[occluded] = AVAILABILITY_REASONS.index("ACTOR_OCCLUDED")
        available[occluded] = False

    # 大范围成像损失：必须同时满足「超出目标尺度的空间范围」与「像素严重不可用」，
    # 单个高亮小区域本身不构成失效依据。
    loss = (context.valid > 0) & (roi_mask > 0) & (
        (current[..., 0] <= 2) & (current[..., 1] <= 2) & (current[..., 2] <= 2)
    )
    if loss.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            loss.astype(np.uint8), 8
        )
        min_side = round(LARGE_LOSS_MIN_SIDE_FACTOR * MAX_PRIOR_TARGET_SIDE_PX)
        min_area = round(LARGE_LOSS_MIN_AREA_FACTOR * MAX_PRIOR_TARGET_AREA_PX)
        for index in range(1, count):
            _x, _y, w, h, area = (int(value) for value in stats[index])
            if area >= min_area and max(w, h) >= min_side:
                big = (labels == index) & available
                reasons[big] = AVAILABILITY_REASONS.index("LARGE_EXPOSURE_LOSS")
                available[big] = False

    if not available.any():
        diagnostics = {
            "available_fraction": 0.0,
            "fit_fraction": 0.0,
            "common_visibility": False,
            "reason_histogram": {
                name: int(np.count_nonzero(reasons == index))
                for index, name in enumerate(AVAILABILITY_REASONS)
                if np.count_nonzero(reasons == index)
            },
            "compensation": {
                "gains": [round(float(pair[0]), 5) for pair in masks["fits"]],
                "biases": [round(float(pair[1]), 5) for pair in masks["fits"]],
            },
        }
        return MaskSet(
            fit_mask=np.zeros_like(fit_mask),
            foreground_support=np.zeros((height, width), np.uint8),
            availability_mask=np.zeros((height, width), np.uint8),
            availability_reason=np.full(
                (height, width), AVAILABILITY_REASONS.index("NO_COMMON_VISIBILITY"),
                np.uint8,
            ),
            diagnostics=diagnostics,
        )

    availability_mask = available.astype(np.uint8) * 255
    fit_mask[~available] = 0
    support = context.foreground_support(current)
    support[~available] = 0

    valid_total = max(int(np.count_nonzero(roi_mask > 0)), 1)
    diagnostics = {
        "available_fraction": round(float(available.sum()) / valid_total, 5),
        "fit_fraction": round(
            float(np.count_nonzero(fit_mask)) / valid_total, 5
        ),
        "common_visibility": True,
        "reason_histogram": {
            name: int(np.count_nonzero(reasons == index))
            for index, name in enumerate(AVAILABILITY_REASONS)
            if np.count_nonzero(reasons == index)
        },
        "compensation": {
            "gains": [round(float(pair[0]), 5) for pair in masks["fits"]],
            "biases": [round(float(pair[1]), 5) for pair in masks["fits"]],
        },
    }
    return MaskSet(
        fit_mask=fit_mask, foreground_support=support,
        availability_mask=availability_mask, availability_reason=reasons,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True, slots=True)
class BankSelectionEvaluation:
    """单帧对单个候选的完整评估结果（含三类掩膜与包络结论）。"""

    profile_id: str
    score: dict[str, Any] | None
    outcome: dict[str, Any]
    mask_diagnostics: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    support_pixels: int
    availability_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "score": self.score,
            "outcome": self.outcome,
            "mask": self.mask_diagnostics,
            "candidate_count": len(self.candidates),
            "support_pixels": self.support_pixels,
            "availability_fraction": self.availability_fraction,
        }


def evaluate_bank_frame(
    context: BankPriorContext, frame: np.ndarray, *,
    envelope: Any, config: Mapping[str, Any] | None = None,
    roi_mask: np.ndarray, actors: Sequence[Sequence[float]] = (),
    common_protection: np.ndarray | None = None,
    geometry_diagnostics: Mapping[str, Any] | None = None,
    corruption: str | None = None, geometry_valid: bool = True,
) -> BankSelectionEvaluation:
    """全尺寸 Bank prior 分析：评分 → 三类掩膜 → 前景证据（**不**改事件 memory）。"""
    config = config or {}
    merged_diagnostics = dict(context.geometry_diagnostics)
    merged_diagnostics.update(geometry_diagnostics or {})
    try:
        score = score_profile(
            frame, context.reference, context.valid,
            common_protection if common_protection is not None
            else np.full(frame.shape[:2], 255, np.uint8),
            profile_id=context.profile_id, config=config,
            geometry_diagnostics=merged_diagnostics,
        )
    except BankError as exc:
        outcome = evaluate_match(
            None, envelope, config, geometry_diagnostics=merged_diagnostics,
            missing=True,
        )
        return BankSelectionEvaluation(
            profile_id=context.profile_id, score=None, outcome=outcome.as_dict(),
            mask_diagnostics={"error": str(exc)[:160]}, candidates=(),
            support_pixels=0, availability_fraction=0.0,
        )
    masks = compute_mask_set(
        context, frame, roi_mask=roi_mask, actors=actors,
        common_protection=common_protection, corruption=corruption,
        geometry_valid=geometry_valid, config=config,
    )
    outcome = evaluate_match(
        score, envelope, config, geometry_diagnostics=merged_diagnostics,
    )
    rows = _support_candidates(masks.foreground_support, masks.availability_mask)
    return BankSelectionEvaluation(
        profile_id=context.profile_id,
        score=score.as_dict(),
        outcome=outcome.as_dict(),
        mask_diagnostics=masks.diagnostics,
        candidates=tuple(rows),
        support_pixels=int(np.count_nonzero(masks.foreground_support)),
        availability_fraction=float(masks.diagnostics.get("available_fraction", 0.0)),
    )


def _support_candidates(
    support: np.ndarray, availability: np.ndarray,
) -> list[dict[str, Any]]:
    """把前景支持连通域转成候选框，但**不**按可用性/噪声门槛删除小目标。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), 8)
    rows: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        component = labels == index
        if int(np.count_nonzero(availability[component])) == 0:
            # 完全落在不可判断区域的连通域不产生证据，但保留原因供报告。
            continue
        rows.append({
            "box": [x, y, x + width, y + height],
            "support_pixels": int(area),
            "anomaly_score": round(
                float(min(1.0, area / max(1.0, float(v32.MINIMUM_SUPPORT_AREA)) / 4.0)), 4
            ),
        })
    rows.sort(key=lambda row: -row["support_pixels"])
    return rows


# --------------------------------------------------------------------------- #
# 静态潜在覆盖
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StaticCoverage:
    frames: int
    covered_frames: int
    per_profile: dict[str, int]
    unique_assignment: dict[str, int]

    @property
    def fraction(self) -> float:
        return 0.0 if self.frames == 0 else self.covered_frames / self.frames

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "potential_coverage",
            "frames": self.frames,
            "covered_frames": self.covered_frames,
            "fraction": round(self.fraction, 5),
            "per_profile": dict(sorted(self.per_profile.items())),
            "unique_assignment": dict(sorted(self.unique_assignment.items())),
            "note": "静态潜在覆盖只表示库的能力，不能冒充 Selector 动态可用覆盖",
        }


def static_potential_coverage(
    descriptor: Mapping[str, np.ndarray], bank: Any,
    descriptors: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    *, config: Mapping[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """单帧唯一分配：只把该帧分配给一个 Profile（静态潜在覆盖）。

    返回 ``(best_profile_id, diagnostics)``；这**不是**动态可用性结论。
    """
    from .ground_litter_profile_match import (
        descriptor_coarse_distance, global_descriptor_scale,
    )

    config = config or {}
    payloads = descriptors or {pid: bank.load_descriptor(pid) for pid in bank.ids()}
    scale = global_descriptor_scale(list(payloads.values()))
    rows: list[tuple[str, float]] = []
    for profile_id, payload in payloads.items():
        try:
            rows.append((profile_id, descriptor_coarse_distance(
                descriptor, payload, scale=scale,
            )))
        except BankError:
            continue
    if not rows:
        return None, {"candidates_considered": 0}
    rows.sort(key=lambda item: (item[1], item[0]))
    return rows[0][0], {
        "candidates_considered": len(rows),
        "best_distance": round(rows[0][1], 5),
        "scale": {key: round(value, 5) for key, value in scale.items()},
    }


__all__ = [
    "AVAILABILITY_REASONS", "BankPriorContext", "BankSelectionEvaluation",
    "MaskSet", "StaticCoverage", "build_prior_context", "compute_mask_set",
    "evaluate_bank_frame", "roi_mask_from_geometry", "static_potential_coverage",
]
