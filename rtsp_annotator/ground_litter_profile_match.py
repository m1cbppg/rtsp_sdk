"""Shared single-frame matcher for Profile Banks (方案一 §5/§7，方案二 §3）。

离线工厂与未来在线 Selector **必须**共用这一份实现：分块描述子、公共稳健尺度、
受限颜色补偿、残差、评分公式与进入/保持包络都在这里定义一次。

本模块只依赖 numpy/cv2 与 Bank 数据契约，不依赖生产 API、不读盘（除传入的数组）、
不持有任何跨帧状态，因此可以在离线回放和在线 tick 中逐字复用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .ground_litter_profile_bank import BankError, ProfileBank

DESCRIPTOR_KEYS = ("grid_luminance", "grid_chroma", "grid_structure", "grid_weight")

# 与 V3.2 相同的残差量纲（signature / luminance），保证旧噪声资产仍可解释。
RESIDUAL_SIGNATURE_SIGMA = 12.0
RESIDUAL_SIGNATURE_EPS = 5.0
RESIDUAL_LUMINANCE_SIGMA = 12.0


# --------------------------------------------------------------------------- #
# 网格与描述子
# --------------------------------------------------------------------------- #


def grid_bounds(width: int, height: int, cols: int, rows: int
                ) -> list[tuple[int, int, int, int]]:
    """按实际画布尺寸给出 (left, top, right, bottom) 像素边界列表。"""
    if cols < 1 or rows < 1:
        raise BankError("网格列数与行数必须为正")
    xs = np.rint(np.linspace(0, width, cols + 1)).astype(int)
    ys = np.rint(np.linspace(0, height, rows + 1)).astype(int)
    bounds: list[tuple[int, int, int, int]] = []
    for row in range(rows):
        for col in range(cols):
            left, right = int(xs[col]), int(xs[col + 1])
            top, bottom = int(ys[row]), int(ys[row + 1])
            bounds.append((left, top, max(left + 1, right), max(top + 1, bottom)))
    return bounds


def _scaled_odd(value: int, scale: float, *, minimum: int = 3) -> int:
    result = max(minimum, int(round(value * scale)))
    return result if result % 2 else result + 1


def _structure_map(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def extract_grid_descriptor(
    frame: np.ndarray, mask: np.ndarray, *, cols: int = 16, rows: int = 9,
    min_weight: float = 0.02,
) -> dict[str, np.ndarray]:
    """在掩膜内按网格计算亮度/色度/结构摘要。

    ``frame`` 为 BGR uint8；``mask`` 为 0/255 的单通道可评分区域。返回的四个
    数组都是 ``(rows, cols)`` float32，缺失网格权重为 0。**不做逐图归一化**——
    文档要求保留绝对亮度/色度，不能让每张图各自消除明暗差异。
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise BankError("描述子输入必须是 BGR 三通道图")
    height, width = frame.shape[:2]
    if mask.shape != (height, width):
        raise BankError("描述子掩膜尺寸不一致")
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    structure = _structure_map(gray)
    eligible = mask > 0

    luminance = np.zeros((rows, cols), np.float32)
    chroma = np.zeros((rows, cols), np.float32)
    struct = np.zeros((rows, cols), np.float32)
    weight = np.zeros((rows, cols), np.float32)
    for index, (left, top, right, bottom) in enumerate(
        grid_bounds(width, height, cols, rows)
    ):
        row, col = divmod(index, cols)
        cell = eligible[top:bottom, left:right]
        total = int(cell.size)
        if total == 0:
            continue
        selected = int(np.count_nonzero(cell))
        fraction = selected / total
        if fraction < min_weight:
            continue
        lum = lab[top:bottom, left:right, 0][cell]
        a = lab[top:bottom, left:right, 1][cell]
        b = lab[top:bottom, left:right, 2][cell]
        edge = structure[top:bottom, left:right][cell]
        luminance[row, col] = float(np.median(lum))
        chroma[row, col] = float(
            math.hypot(float(np.median(a)) - 128.0, float(np.median(b)) - 128.0)
        )
        struct[row, col] = float(np.median(edge))
        weight[row, col] = float(fraction)
    return {
        "grid_luminance": luminance,
        "grid_chroma": chroma,
        "grid_structure": struct,
        "grid_weight": weight,
    }


def descriptor_arrays(payload: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = [key for key in DESCRIPTOR_KEYS if key not in payload]
    if missing:
        raise BankError(f"descriptor.npz 缺少: {', '.join(missing)}")
    return {key: np.asarray(payload[key], np.float32) for key in DESCRIPTOR_KEYS}


def descriptor_coarse_distance(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray],
    *, scale: Mapping[str, float] | None = None,
) -> float:
    """小图粗检索距离：亮度/色度/结构三通道加权中位数绝对差。

    只用于排序（Top-K 与扩展游标），不用于合格判定；合格判定必须走全尺寸评分。
    """
    a = descriptor_arrays(left)
    b = descriptor_arrays(right)
    if a["grid_luminance"].shape != b["grid_luminance"].shape:
        raise BankError("描述子网格尺寸不一致")
    weight = np.minimum(a["grid_weight"], b["grid_weight"])
    visible = weight > 0
    if not np.any(visible):
        return float("inf")
    scales = dict(scale or {})
    lum_scale = max(float(scales.get("luminance", 12.0)), 1e-6)
    chroma_scale = max(float(scales.get("chroma", 10.0)), 1e-6)
    struct_scale = max(float(scales.get("structure", 0.06)), 1e-6)
    lum = np.abs(a["grid_luminance"] - b["grid_luminance"]) / lum_scale
    chroma = np.abs(a["grid_chroma"] - b["grid_chroma"]) / chroma_scale
    struct = np.abs(a["grid_structure"] - b["grid_structure"]) / struct_scale
    combined = 0.5 * lum + 0.2 * chroma + 0.3 * struct
    return float(np.median(combined[visible]))


def global_descriptor_scale(descriptors: Sequence[Mapping[str, np.ndarray]]) -> dict[str, float]:
    """从整个构建集估计公共稳健尺度（MAD 的 1.4826 倍），并设下限。

    公共尺度必须来自构建集整体：不能让噪声大的 Profile 用自己的巨大标准差把
    残差除小（方案二 §3.3）。
    """
    if not descriptors:
        raise BankError("需要至少一个描述子来估计公共尺度")
    result: dict[str, float] = {}
    for key, floor in (("grid_luminance", 12.0), ("grid_chroma", 10.0),
                       ("grid_structure", 0.06)):
        values = np.concatenate([
            np.asarray(item[key], np.float32).ravel()
            for item in descriptors
            if key in item and np.asarray(item[key]).size
        ]) if any(key in item for item in descriptors) else np.array([0.0], np.float32)
        if values.size == 0:
            result[key] = float(floor)
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust = 1.4826 * mad
        result[key] = float(max(robust, floor))
    return result


# --------------------------------------------------------------------------- #
# 残差与补偿
# --------------------------------------------------------------------------- #


def residual_maps(reference: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """与 V3.2 完全相同的 signature/luminance 残差量纲。

    工厂在共享补偿实现冻结后必须用这一函数生成正式 noise，避免离线与在线量纲漂移。
    """
    old = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    new = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def signature(gray: np.ndarray) -> np.ndarray:
        mean = cv2.GaussianBlur(gray, (0, 0), RESIDUAL_SIGNATURE_SIGMA)
        variance = np.maximum(
            cv2.GaussianBlur(gray * gray, (0, 0), RESIDUAL_SIGNATURE_SIGMA) - mean * mean,
            0,
        )
        return (gray - mean) / (np.sqrt(variance) + RESIDUAL_SIGNATURE_EPS)

    signature_residual = np.abs(signature(new) - signature(old))
    delta = new - old
    luminance_residual = np.abs(
        delta - cv2.GaussianBlur(delta, (0, 0), RESIDUAL_LUMINANCE_SIGMA)
    )
    return signature_residual, luminance_residual


@dataclass(frozen=True, slots=True)
class ColorCompensation:
    """受限的低自由度全局增益/偏移，同时保留**原始**拟合需求。"""

    gains: tuple[float, float, float]
    biases: tuple[float, float, float]
    raw_gains: tuple[float, float, float]
    raw_biases: tuple[float, float, float]
    sample_pixels: int

    @property
    def clipped(self) -> bool:
        return any(
            abs(clipped - raw) > 1e-6
            for clipped, raw in zip(
                self.gains + self.biases, self.raw_gains + self.raw_biases
            )
        )

    @property
    def max_gain_delta(self) -> float:
        return max(abs(gain - 1.0) for gain in self.raw_gains)

    @property
    def max_abs_bias(self) -> float:
        return max(abs(bias) for bias in self.raw_biases)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gains": [round(value, 5) for value in self.gains],
            "biases": [round(value, 5) for value in self.biases],
            "raw_gains": [round(value, 5) for value in self.raw_gains],
            "raw_biases": [round(value, 5) for value in self.raw_biases],
            "clipped": self.clipped,
            "sample_pixels": self.sample_pixels,
        }


def robust_color_compensation(
    reference: np.ndarray, current: np.ndarray, fit_mask: np.ndarray,
    *, gain_range: Sequence[float] = (0.65, 1.45),
    bias_range: Sequence[float] = (-60.0, 60.0),
    iterations: int = 3, max_sample: int = 250_000,
) -> ColorCompensation:
    """受限全局线性补偿；返回裁剪后的系数与未裁剪的原始需求。

    原始需求必须暴露出来：方案二 §3.4 要求过大的原始需求不能被参数裁剪掩盖。
    """
    old = reference.astype(np.float32)
    new = current.astype(np.float32)
    eligible = (fit_mask > 0)
    eligible &= np.all((old > 5) & (old < 250) & (new > 5) & (new < 250), axis=2)
    indices = np.flatnonzero(eligible)
    if indices.size < 5000:
        raise BankError("Profile 匹配可信拟合区域不足")
    if indices.size > max_sample:
        indices = indices[np.linspace(0, indices.size - 1, max_sample).astype(np.intp)]
    old_sample = old.reshape(-1, 3)[indices]
    new_sample = new.reshape(-1, 3)[indices]
    keep = np.ones(indices.size, bool)
    raw_fits: list[tuple[float, float]] = [(1.0, 0.0)] * 3
    for _ in range(max(1, iterations)):
        raw_fits = []
        for channel in range(3):
            x = old_sample[keep, channel]
            y = new_sample[keep, channel]
            if x.size < 64 or float(np.ptp(x)) < 1e-3:
                raw_fits.append((1.0, 0.0))
                continue
            gain, bias = np.polyfit(x, y, 1)
            raw_fits.append((float(gain), float(bias)))
        prediction = np.stack([
            old_sample[:, channel] * raw_fits[channel][0] + raw_fits[channel][1]
            for channel in range(3)
        ], axis=1)
        residual = np.max(np.abs(new_sample - prediction), axis=1)
        cutoff = max(float(np.percentile(residual[keep], 70)), 8.0)
        keep = residual <= cutoff
    clipped = [
        (float(np.clip(gain, gain_range[0], gain_range[1])),
         float(np.clip(bias, bias_range[0], bias_range[1])))
        for gain, bias in raw_fits
    ]
    return ColorCompensation(
        gains=tuple(pair[0] for pair in clipped),
        biases=tuple(pair[1] for pair in clipped),
        raw_gains=tuple(pair[0] for pair in raw_fits),
        raw_biases=tuple(pair[1] for pair in raw_fits),
        sample_pixels=int(indices.size),
    )


def apply_compensation(reference: np.ndarray, compensation: ColorCompensation) -> np.ndarray:
    old = reference.astype(np.float32)
    normalized = np.stack([
        old[..., channel] * compensation.gains[channel] + compensation.biases[channel]
        for channel in range(3)
    ], axis=2)
    return np.clip(normalized, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 全尺寸评分
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ProfileScore:
    profile_id: str
    score: float
    cell_q50: float
    cell_q90: float
    compensation_cost: float
    missing_fraction: float
    visible_cells: int
    total_cells: int
    anchor_fraction: float
    distinct_regions: int
    compensation: dict[str, Any] | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # 同一 tick 的补偿结果必须被评分与前景提取共用：否则匹配通过而 prior 残差
    # 仍是未补偿的原始差异（R10）。这两个字段不参与 as_dict 以保持报告兼容。
    compensated_reference: Any = field(default=None, repr=False, compare=False)
    fit_mask: Any = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "score": round(self.score, 5),
            "cell_q50": round(self.cell_q50, 5),
            "cell_q90": round(self.cell_q90, 5),
            "compensation_cost": round(self.compensation_cost, 5),
            "missing_fraction": round(self.missing_fraction, 5),
            "visible_cells": self.visible_cells,
            "total_cells": self.total_cells,
            "anchor_fraction": round(self.anchor_fraction, 5),
            "distinct_regions": self.distinct_regions,
            "compensation": self.compensation,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True, slots=True)
class MatchEnvelope:
    """进入/保持门槛包络。样本足够时用高分位 + 有限余量，否则走公共回退。"""

    enter: float
    hold: float
    calibrated: bool
    samples: int
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "enter": round(self.enter, 5), "hold": round(self.hold, 5),
            "calibrated": self.calibrated, "samples": self.samples,
            "source": self.source,
        }


def envelope_from_samples(
    scores: Iterable[float], config: Mapping[str, Any],
) -> MatchEnvelope:
    """从参考来源之外的连续块统计进入/保持门槛（方案二 §3.4）。"""
    values = np.asarray([float(value) for value in scores], np.float32)
    envelope = dict(config.get("envelope", {}))
    quantile = float(envelope.get("quantile", 0.95))
    enter_margin = float(envelope.get("enter_margin", 0.08))
    hold_margin = float(envelope.get("hold_margin", 0.20))
    minimum = int(envelope.get("min_calibration_samples", 8))
    fallback_enter = float(envelope.get("fallback_enter", 1.6))
    fallback_hold = float(envelope.get("fallback_hold", 2.2))
    if values.size >= minimum:
        base = float(np.quantile(values, quantile))
        return MatchEnvelope(
            enter=base + enter_margin, hold=base + hold_margin,
            calibrated=True, samples=int(values.size), source="calibration_quantile",
        )
    return MatchEnvelope(
        enter=fallback_enter, hold=fallback_hold,
        calibrated=False, samples=int(values.size), source="fallback",
    )


def _distinct_regions(cells: np.ndarray, cols: int) -> int:
    """统计权重非零网格落在几个分散区域（粗 3×3 分区）。"""
    rows = cells.shape[0]
    region_rows = max(1, min(3, rows))
    region_cols = max(1, min(3, cols))
    occupied = 0
    for row in range(region_rows):
        top = int(round(row * rows / region_rows))
        bottom = int(round((row + 1) * rows / region_rows))
        for col in range(region_cols):
            left = int(round(col * cols / region_cols))
            right = int(round((col + 1) * cols / region_cols))
            if np.any(cells[top:bottom, left:right]):
                occupied += 1
    return occupied


def score_profile(
    frame: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    common_mask: np.ndarray,
    *,
    profile_id: str = "",
    config: Mapping[str, Any] | None = None,
    geometry_diagnostics: Mapping[str, Any] | None = None,
) -> ProfileScore:
    """在共同画布上对一个候选参考做全尺寸评分。

    ``common_mask`` 是本 tick 对所有候选**一致**的共同保护区域：actor/OSD/已知目标
    在进入本函数前已被移除。每个候选不允许删除自己的高残差位置来提高得分。
    """
    config = config or {}
    if frame.shape != reference.shape:
        raise BankError("候选参考尺寸与当前画布不一致")
    grid = dict(config.get("grid", {}))
    cols = int(grid.get("cols", 16))
    rows = int(grid.get("rows", 9))
    min_weight = float(grid.get("min_weight", 0.02))
    weights = dict(config.get("score_weights", {}))
    w_lum = float(weights.get("luminance", 0.4))
    w_chroma = float(weights.get("chroma", 0.2))
    w_struct = float(weights.get("structure", 0.4))
    w_q50 = float(weights.get("q50", 0.5))
    w_q90 = float(weights.get("q90", 0.5))
    w_comp = float(weights.get("compensation", 0.2))
    w_missing = float(weights.get("missing", 1.0))
    floors = dict(config.get("scale_floor", {}))
    comp_config = dict(config.get("compensation", {}))

    fit_mask = np.zeros(frame.shape[:2], np.uint8)
    fit_mask[(common_mask > 0) & (valid > 0)] = 255
    if int(np.count_nonzero(fit_mask)) < 5000:
        raise BankError("共同拟合区域不足，无法评分")

    compensation = robust_color_compensation(
        reference, frame, fit_mask,
        gain_range=comp_config.get("gain_range", (0.65, 1.45)),
        bias_range=comp_config.get("bias_range", (-60.0, 60.0)),
    )
    normalized = apply_compensation(reference, compensation)
    signature, luminance = residual_maps(normalized, frame)

    bounds = grid_bounds(frame.shape[1], frame.shape[0], cols, rows)
    comp_cost = (
        compensation.max_gain_delta / max(float(comp_config.get("max_abs_gain_delta", 0.18)), 1e-6)
        + compensation.max_abs_bias / max(float(comp_config.get("max_abs_bias", 30.0)), 1e-6)
    ) / 2.0
    cell_values: list[float] = []
    cell_weights: list[float] = []
    cell_mask = np.zeros((rows, cols), bool)
    for index, (left, top, right, bottom) in enumerate(bounds):
        row, col = divmod(index, cols)
        cell = fit_mask[top:bottom, left:right] > 0
        total = int(cell.size)
        if total == 0:
            continue
        fraction = int(np.count_nonzero(cell)) / total
        if fraction < min_weight:
            continue
        lum_values = luminance[top:bottom, left:right][cell]
        sig_values = signature[top:bottom, left:right][cell]
        lum_scale = max(float(floors.get("luminance", 12.0)), 1e-6)
        sig_scale = max(float(floors.get("structure", 0.06)), 1e-6)
        # 亮度残差取中位数（对目标尺度的小连通域稳健），结构残差取高分位。
        cell_value = (
            w_lum * float(np.median(lum_values)) / lum_scale
            + w_struct * float(np.percentile(sig_values, 90)) / sig_scale
        )
        cell_values.append(cell_value)
        cell_weights.append(fraction)
        cell_mask[row, col] = True

    if not cell_values:
        raise BankError("没有任何可用网格，无法评分")
    values = np.asarray(cell_values, np.float32)
    weights_array = np.asarray(cell_weights, np.float32)
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights_array[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= max(float(cumulative[-1]), 1e-6)
    q50 = float(np.interp(0.50, cumulative, sorted_values))
    q90 = float(np.interp(0.90, cumulative, sorted_values))
    visible = int(np.count_nonzero(cell_mask))
    total_cells = rows * cols
    missing_fraction = 1.0 - visible / total_cells
    anchor_fraction = float(np.count_nonzero(fit_mask)) / float(
        max(int(np.count_nonzero(valid > 0)), 1)
    )
    score = (
        w_q50 * q50 + w_q90 * q90 + w_comp * comp_cost + w_missing * missing_fraction
    )
    diagnostics: dict[str, Any] = {}
    if geometry_diagnostics:
        diagnostics.update(geometry_diagnostics)
    diagnostics["compensation_clipped"] = compensation.clipped
    diagnostics["raw_gain_delta"] = round(compensation.max_gain_delta, 5)
    diagnostics["raw_bias"] = round(compensation.max_abs_bias, 5)
    return ProfileScore(
        compensated_reference=normalized,
        fit_mask=fit_mask,
        profile_id=profile_id,
        score=float(score),
        cell_q50=q50,
        cell_q90=q90,
        compensation_cost=float(comp_cost),
        missing_fraction=float(missing_fraction),
        visible_cells=visible,
        total_cells=total_cells,
        anchor_fraction=anchor_fraction,
        distinct_regions=_distinct_regions(cell_mask, cols),
        compensation=compensation.as_dict(),
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# 进入/保持判定
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    profile_id: str
    score: ProfileScore | None
    enter_eligible: bool
    hold_eligible: bool
    geometry_ok: bool
    anchors_ok: bool
    envelope_ok: bool
    compensation_ok: bool
    reason: str
    missing: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "score": None if self.score is None else round(self.score.score, 5),
            "enter_eligible": self.enter_eligible,
            "hold_eligible": self.hold_eligible,
            "geometry_ok": self.geometry_ok,
            "anchors_ok": self.anchors_ok,
            "envelope_ok": self.envelope_ok,
            "compensation_ok": self.compensation_ok,
            "reason": self.reason,
            "missing": self.missing,
            "details": None if self.score is None else self.score.as_dict(),
        }


def coerce_envelope(value: Any) -> MatchEnvelope:
    """把冻结在 Bank 里的包络（dict 或 MatchEnvelope）统一成对象。

    冻结资产以 JSON 存储，运行时/评估读回来是 dict；两者必须走同一判定逻辑。
    """
    if isinstance(value, MatchEnvelope):
        return value
    if isinstance(value, Mapping):
        return MatchEnvelope(
            enter=float(value.get("enter", value.get("fallback_enter", 1.6))),
            hold=float(value.get("hold", value.get("fallback_hold", 2.2))),
            calibrated=bool(value.get("calibrated", False)),
            samples=int(value.get("samples", 0) or 0),
            source=str(value.get("source", "frozen")),
        )
    raise BankError("包络必须是 MatchEnvelope 或包含 enter/hold 的映射")


def evaluate_match(
    score: ProfileScore | None,
    envelope: MatchEnvelope | Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    geometry_diagnostics: Mapping[str, Any] | None = None,
    missing: bool = False,
) -> MatchOutcome:
    """把评分转成进入/保持资格，并把每个失败原因显式命名。"""
    envelope = coerce_envelope(envelope)
    geometry_config = dict(config.get("geometry", {}))
    compensation_config = dict(config.get("compensation", {}))
    if score is None:
        return MatchOutcome(
            profile_id="", score=None, enter_eligible=False, hold_eligible=False,
            geometry_ok=False, anchors_ok=False, envelope_ok=False,
            compensation_ok=False, reason="SCORE_UNAVAILABLE", missing=missing,
        )
    diagnostics = dict(geometry_diagnostics or {})
    merged = {**diagnostics, **dict(score.diagnostics or {})}
    geometry_ok = bool(merged.get("geometry_ok", True))
    if geometry_ok and "reprojection_p95_px" in merged:
        geometry_ok = (
            float(merged["reprojection_p95_px"])
            <= float(geometry_config.get("max_reprojection_p95_px", 2.0))
            and float(merged.get("inlier_hull_fraction", 1.0))
            >= float(geometry_config.get("min_inlier_hull_fraction", 0.02))
        )
    anchors_ok = (
        score.anchor_fraction >= float(geometry_config.get("min_anchor_fraction", 0.35))
        and score.distinct_regions >= int(geometry_config.get("min_distinct_regions", 3))
    )
    compensation_ok = not bool(merged.get("compensation_clipped", False)) and (
        float(merged.get("raw_gain_delta", 0.0))
        <= float(compensation_config.get("max_abs_gain_delta", 0.18)) * 2.0
        and float(merged.get("raw_bias", 0.0))
        <= float(compensation_config.get("max_abs_bias", 30.0)) * 2.0
    )
    hold_ok = score.score <= envelope.hold
    enter_ok = score.score <= envelope.enter
    if not geometry_ok:
        reason = "GEOMETRY_INVALID"
    elif not anchors_ok:
        reason = "ANCHORS_INSUFFICIENT"
    elif not compensation_ok:
        reason = "COMPENSATION_OUT_OF_RANGE"
    elif not hold_ok:
        reason = "SCORE_ABOVE_HOLD"
    elif not enter_ok:
        reason = "SCORE_ABOVE_ENTER"
    else:
        reason = "MATCHED"
    return MatchOutcome(
        profile_id=score.profile_id,
        score=score,
        enter_eligible=bool(enter_ok and geometry_ok and anchors_ok and compensation_ok),
        hold_eligible=bool(hold_ok and geometry_ok and anchors_ok and compensation_ok),
        geometry_ok=geometry_ok,
        anchors_ok=anchors_ok,
        envelope_ok=bool(enter_ok),
        compensation_ok=compensation_ok,
        reason=reason,
        missing=missing,
    )


@dataclass(frozen=True, slots=True)
class ResidualSupport:
    """共享的「补偿后残差 → 前景支持」结果。

    评分（score_profile）与 prior 前景提取都调用这里，保证两者使用**同一份**
    受限补偿与同一套噪声阈值，不会出现「匹配通过但 prior 用未补偿差异」。
    """

    compensated_reference: np.ndarray
    signature: np.ndarray
    luminance: np.ndarray
    seed: np.ndarray
    support: np.ndarray
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.diagnostics)


def analyze_residual_support(
    frame: np.ndarray, reference: np.ndarray, valid: np.ndarray,
    fit_mask: np.ndarray, thresholds: Mapping[str, np.ndarray], *,
    pixel_scale: float = 1.0, min_availability: np.ndarray | None = None,
) -> ResidualSupport:
    """用已冻结阈值从**补偿后**参考生成 seed/support 前景掩膜。

    ``thresholds`` 必须包含 seed/support 的 signature 与 luminance 阈值图，
    与 ``BankPriorContext.threshold()`` 的键一致。
    """
    height, width = frame.shape[:2]
    signature, luminance = residual_maps(reference, frame)
    usable = (valid > 0)
    if min_availability is not None:
        usable &= (min_availability > 0)
    seed = (
        (signature >= thresholds["seed_signature_threshold"])
        & (luminance >= thresholds["seed_luminance_threshold"])
        & usable
    ).astype(np.uint8)
    support = (
        (signature >= thresholds["support_signature_threshold"])
        & (luminance >= thresholds["support_luminance_threshold"])
        & usable
    ).astype(np.uint8)
    marker = cv2.dilate(
        seed, np.ones((_scaled_odd(9, pixel_scale), _scaled_odd(9, pixel_scale)), np.uint8)
    )
    grown = cv2.bitwise_and(support, marker)
    grown = cv2.morphologyEx(
        grown, cv2.MORPH_CLOSE,
        np.ones((_scaled_odd(5, pixel_scale), _scaled_odd(5, pixel_scale)), np.uint8),
    )
    grown[fit_mask == 0] = 0
    return ResidualSupport(
        compensated_reference=reference, signature=signature, luminance=luminance,
        seed=seed, support=grown,
        diagnostics={
            "seed_pixels": int(np.count_nonzero(seed)),
            "support_pixels": int(np.count_nonzero(grown)),
            "pixel_scale": round(float(pixel_scale), 5),
            "compensated": True,
        },
    )


def rank_by_coarse_distance(
    descriptor: Mapping[str, np.ndarray],
    bank: ProfileBank,
    *,
    descriptors: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    scale: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """全库小向量粗检索，返回按距离升序的 (profile_id, distance)。

    文档要求比较**全部** N 个小向量，不按时间标签筛库；排序只决定检查顺序。
    """
    if descriptors is None:
        descriptors = {pid: bank.load_descriptor(pid) for pid in bank.ids()}
    rows: list[tuple[str, float]] = []
    for profile_id in bank.ids():
        payload = descriptors.get(profile_id)
        if payload is None:
            continue
        try:
            distance = descriptor_coarse_distance(descriptor, payload, scale=scale)
        except BankError:
            continue
        if math.isfinite(distance):
            rows.append((profile_id, distance))
    rows.sort(key=lambda item: (item[1], item[0]))
    return rows


__all__ = [
    "ColorCompensation", "DESCRIPTOR_KEYS", "MatchEnvelope", "MatchOutcome",
    "ProfileScore", "apply_compensation", "coerce_envelope", "descriptor_arrays",
    "descriptor_coarse_distance", "envelope_from_samples", "evaluate_match",
    "extract_grid_descriptor", "global_descriptor_scale", "grid_bounds",
    "rank_by_coarse_distance", "residual_maps", "robust_color_compensation",
    "score_profile",
]
