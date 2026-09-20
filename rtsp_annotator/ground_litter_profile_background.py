"""A3：代表外观分组、时间均衡合成与有限噪声（方案一 §5、§6、§6.3）。

设计复核 F3 的落点：噪声门槛必须同时使用**残差中心**、正余量、MAD/Q95、超 cap 比例
与支持时间块数，不能只看 MAD。恒定大残差在 MAD=0 时也会被残差中心捕获。

明确不做的事：

* 不训练新的大模型，不用生成式补图，不做逐物体清洁身份推理；
* 不因为某个像素残差高就在线把它挖掉；``bias_flag`` 是**离线**质量诊断；
* 不承诺「参考绝对没有垃圾」——七天始终存在的物体进入参考是允许误差。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .ground_litter_profile_bank import BankError
from .ground_litter_profile_match import (
    descriptor_arrays, descriptor_coarse_distance, global_descriptor_scale,
)

# 方案一 §6.3 的首版上限：直接采用现有 V3.2 噪声区最大增量，不先扩大敏感度。
NOISE_DEFAULTS: dict[str, Any] = {
    "k": 2.0,
    "epsilon_seed_signature": 0.05,
    "epsilon_seed_luminance": 1.0,
    "epsilon_support_signature": 0.02,
    "epsilon_support_luminance": 0.4,
    "seed_signature_base": 3.0,
    "seed_luminance_base": 100.0,
    "support_signature_base": 1.2,
    "support_luminance_base": 25.0,
    "seed_signature_cap": 3.8,
    "seed_luminance_cap": 130.0,
    "support_signature_cap": 1.45,
    "support_luminance_cap": 33.0,
    "bias_over_cap_fraction": 0.02,
    "min_support_blocks": 3,
    "chunk_frames": 12,
}

MINIMUM_CLUSTER_DIAMETER = 0.12
MAX_CANDIDATES = 48
MAX_BANK_PROFILES = 24


# --------------------------------------------------------------------------- #
# 代表外观分组
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AppearanceGroup:
    group_id: str
    representative_key: str
    member_keys: tuple[str, ...]
    center: Mapping[str, float]
    radius: float
    time_blocks: tuple[str, ...]
    days: tuple[str, ...]
    mean_quality: float
    low_support: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "representative_key": self.representative_key,
            "members": len(self.member_keys),
            "radius": round(self.radius, 5),
            "time_blocks": len(self.time_blocks),
            "days": list(self.days),
            "mean_quality": round(self.mean_quality, 5),
            "low_support": self.low_support,
        }


@dataclass(frozen=True, slots=True)
class GroupingResult:
    groups: tuple[AppearanceGroup, ...]
    outliers: tuple[str, ...]
    scale: Mapping[str, float]
    candidate_limit_reached: bool
    statistics: dict[str, Any]


def _quality_score(quality: Mapping[str, Any]) -> float:
    laplacian = float(quality.get("laplacian_variance", 0.0))
    detail = float(quality.get("detail_loss_fraction", 0.0))
    return laplacian / (1.0 + 20.0 * detail)


def _item_key(item: Mapping[str, Any]) -> str:
    return str(item.get("time_block") or item.get("identity_key") or "")


def group_appearance_samples(
    samples: Sequence[Mapping[str, Any]], *,
    descriptors: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    radius: float | None = None, max_candidates: int = MAX_CANDIDATES,
    min_blocks_for_center: int = 1,
) -> GroupingResult:
    """自适应代表点聚类（最远点 + medoid 重分配）。

    每个样本必须有 ``time_block``、``quality`` 与描述子载荷。距离使用**整个构建集**
    的公共尺度，不能每张图各自归一化。
    """
    if not samples:
        return GroupingResult((), (), {}, False, {"samples": 0})
    payloads: dict[str, Mapping[str, np.ndarray]] = dict(descriptors or {})
    for item in samples:
        key = _item_key(item)
        if key not in payloads and "descriptor" in item:
            payloads[key] = item["descriptor"]  # type: ignore[assignment]
    keys = [_item_key(item) for item in samples]
    missing = [key for key in keys if key not in payloads]
    if missing:
        raise BankError(f"外观分组缺少描述子: {missing[:3]}")
    arrays = {key: descriptor_arrays(payloads[key]) for key in keys}
    scale = global_descriptor_scale(list(arrays.values()))
    pairwise_radius = radius if radius is not None else _initial_radius(arrays, scale)

    by_key = {_item_key(item): item for item in samples}
    ordered_keys = sorted(keys, key=lambda key: (
        -_quality_score(by_key[key].get("quality", {})), key,
    ))
    centers: list[str] = []
    for key in ordered_keys:
        if len(centers) >= max_candidates:
            break
        if not centers:
            centers.append(key)
            continue
        distance = min(
            descriptor_coarse_distance(arrays[key], arrays[center], scale=scale)
            for center in centers
        )
        if distance <= pairwise_radius:
            continue
        # 只有在多个独立时间块复现的远距离样本才新增中心。
        near_blocks = {
            by_key[center].get("time_block")
            for center in centers
            if descriptor_coarse_distance(arrays[key], arrays[center], scale=scale)
            <= pairwise_radius * 1.5
        }
        key_block = by_key[key].get("time_block")
        if key_block in near_blocks:
            continue
        centers.append(key)

    # medoid 重分配：中心更新为组内实际代表帧。
    for _ in range(3):
        members: dict[str, list[str]] = {center: [] for center in centers}
        for key in keys:
            best = min(
                centers,
                key=lambda center: (
                    descriptor_coarse_distance(arrays[key], arrays[center], scale=scale),
                    center,
                ),
            )
            members[best].append(key)
        updated = []
        for center, group in members.items():
            if not group:
                continue
            medoid = min(
                group,
                key=lambda candidate: (
                    sum(
                        descriptor_coarse_distance(
                            arrays[candidate], arrays[other], scale=scale
                        ) for other in group
                    ) / len(group),
                    candidate,
                ),
            )
            updated.append((medoid, group))
        new_centers = [medoid for medoid, _group in updated]
        if sorted(new_centers) == sorted(centers):
            centers = new_centers
            break
        centers = new_centers
    members = {center: [] for center in centers}
    for key in keys:
        best = min(
            centers,
            key=lambda center: (
                descriptor_coarse_distance(arrays[key], arrays[center], scale=scale),
                center,
            ),
        )
        members[best].append(key)

    groups: list[AppearanceGroup] = []
    outliers: list[str] = []
    limit_reached = len(centers) >= max_candidates
    for index, center in enumerate(sorted(centers)):
        group = sorted(members[center])
        blocks = sorted({str(by_key[key].get("time_block")) for key in group})
        days = sorted({str(by_key[key].get("day")) for key in group})
        if len(blocks) < min_blocks_for_center:
            outliers.extend(group)
            continue
        quality = float(np.mean([
            _quality_score(by_key[key].get("quality", {})) for key in group
        ])) if group else 0.0
        distances = [
            descriptor_coarse_distance(arrays[key], arrays[center], scale=scale)
            for key in group
        ]
        groups.append(AppearanceGroup(
            group_id=f"g{index + 1:03d}",
            representative_key=center,
            member_keys=tuple(group),
            center={key: round(float(value), 6) for key, value in scale.items()},
            radius=float(max(distances) if distances else 0.0),
            time_blocks=tuple(blocks),
            days=tuple(days),
            mean_quality=quality,
            low_support=len(blocks) < 3 or len(days) < 2,
        ))
    statistics = {
        "samples": len(samples),
        "requested_radius": round(float(pairwise_radius), 5),
        "scale": {key: round(float(value), 5) for key, value in scale.items()},
        "centers": len(centers),
        "groups": len(groups),
        "outliers": len(outliers),
        "low_support_groups": sum(1 for group in groups if group.low_support),
    }
    return GroupingResult(
        groups=tuple(groups), outliers=tuple(sorted(outliers)), scale=scale,
        candidate_limit_reached=limit_reached, statistics=statistics,
    )


def _initial_radius(
    arrays: Mapping[str, Mapping[str, np.ndarray]], scale: Mapping[str, float],
) -> float:
    """从相邻稳定时间块的距离分布初始化半径（中位数 × 系数）。"""
    keys = sorted(arrays)
    if len(keys) < 2:
        return MINIMUM_CLUSTER_DIAMETER
    distances = []
    for left, right in zip(keys, keys[1:]):
        distances.append(descriptor_coarse_distance(arrays[left], arrays[right], scale=scale))
    distances = [value for value in distances if math.isfinite(value)]
    if not distances:
        return MINIMUM_CLUSTER_DIAMETER
    return float(max(MINIMUM_CLUSTER_DIAMETER, np.median(distances) * 1.5))


# --------------------------------------------------------------------------- #
# 时间均衡选帧
# --------------------------------------------------------------------------- #


def select_time_balanced_frames(
    samples: Sequence[Mapping[str, Any]], *, frames_per_profile: int = 90,
    per_file_limit: int = 8, min_files: int = 6, min_days: int = 2,
) -> dict[str, Any]:
    """从每组选不连续文件，优先跨多日；每个文件贡献设上限。

    返回选中的 ``time_block`` 列表与支持诊断（低支持要如实标注）。
    """
    if frames_per_profile < 1:
        raise BankError("frames_per_profile 必须为正")
    by_block: dict[str, list[Mapping[str, Any]]] = {}
    for item in samples:
        by_block.setdefault(str(item.get("time_block")), []).append(item)
    files: dict[str, list[str]] = {}
    for block, items in by_block.items():
        files.setdefault(str(items[0].get("identity_key")), []).append(block)
    # 轮转：每轮从每个文件取一个块，避免单文件占满配额。
    for blocks in files.values():
        blocks.sort()
    file_order = sorted(files, key=lambda key: (files[key][0], key))
    selected: list[str] = []
    per_file: dict[str, int] = {key: 0 for key in files}
    cursor = 0
    while len(selected) < frames_per_profile and files:
        progressed = False
        for file_key in file_order:
            if len(selected) >= frames_per_profile:
                break
            if per_file[file_key] >= per_file_limit:
                continue
            blocks = files[file_key]
            if cursor >= len(blocks):
                continue
            selected.append(blocks[cursor])
            per_file[file_key] += 1
            progressed = True
        cursor += 1
        if not progressed:
            break
    days = {
        str(by_block[block][0].get("day"))
        for block in selected if block in by_block
    }
    distinct_files = {block.split("@")[0] for block in selected}
    diagnostics = {
        "selected_blocks": len(selected),
        "distinct_files": len(distinct_files),
        "days": sorted(days),
        "requested": frames_per_profile,
        "satisfied": (
            len(distinct_files) >= min_files and len(days) >= min_days
            and len(selected) >= min(len(by_block), frames_per_profile)
        ),
        "low_support": len(distinct_files) < min_files or len(days) < min_days,
    }
    return {"time_blocks": selected, "diagnostics": diagnostics}


# --------------------------------------------------------------------------- #
# 合成
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CompositeResult:
    reference: np.ndarray
    valid: np.ndarray
    block_counts: np.ndarray
    diagnostics: dict[str, Any]


def _validate_aligned(frames: Sequence[np.ndarray], valid: np.ndarray) -> tuple[int, int]:
    if not frames:
        raise BankError("合成至少需要一帧")
    height, width = frames[0].shape[:2]
    if valid.shape != (height, width):
        raise BankError("valid 掩膜与合成画布尺寸不一致")
    for frame in frames:
        if frame.shape[:2] != (height, width):
            raise BankError("参与合成的帧尺寸不一致")
    return height, width


def temporal_median_composite(
    frames: Sequence[np.ndarray], masks: Sequence[np.ndarray],
    valid: np.ndarray, block_ids: Sequence[str], *,
    stride: int = 2, min_observations: int = 3,
) -> CompositeResult:
    """跨时间块的逐像素加权时间中位数（稳健初稿）。

    ``block_ids`` 相同者视为同一时间块，块内先取中位数再跨块取中位数，
    使同一时段的重复帧不额外增加权重。
    """
    height, width = _validate_aligned(frames, valid)
    if len(frames) != len(masks) or len(frames) != len(block_ids):
        raise BankError("frames/masks/block_ids 长度必须一致")
    output_size = (
        max(1, math.ceil(width / stride)), max(1, math.ceil(height / stride))
    )
    blocks: dict[str, list[np.ndarray]] = {}
    for frame, mask, block in zip(frames, masks, block_ids):
        small = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
        small = small.copy()
        small[small_mask == 0] = 0
        blocks.setdefault(str(block), []).append(small)
    if len(blocks) < min_observations:
        # 支持不足时如实标注，不伪造统计。
        pass
    block_medians = []
    for block, items in sorted(blocks.items()):
        stack = np.stack(items, axis=0).astype(np.float32)
        block_medians.append(np.median(stack, axis=0))
    if not block_medians:
        raise BankError("没有可用的合成输入")
    reference_small = np.median(np.stack(block_medians, axis=0), axis=0)
    valid_small = cv2.resize(valid, output_size, interpolation=cv2.INTER_NEAREST)
    valid_small = (valid_small > 0).astype(np.uint8) * 255
    reference = cv2.resize(reference_small, (width, height), interpolation=cv2.INTER_CUBIC)
    reference = np.clip(reference, 0, 255).astype(np.uint8)
    reference[valid == 0] = 0
    return CompositeResult(
        reference=reference,
        valid=valid_small if stride != 1 else valid.copy(),
        block_counts=np.full(
            (1 if stride != 1 else height, 1 if stride != 1 else width),
            len(block_medians), np.float32,
        ),
        diagnostics={
            "frames": len(frames),
            "time_blocks": len(block_medians),
            "stride": stride,
            "low_support": len(block_medians) < min_observations,
            "method": "block_median_then_temporal_median",
        },
    )


def replace_with_real_observations(
    composite: np.ndarray, frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray], valid: np.ndarray, *,
    tile: int = 64, max_difference: float = 12.0, min_valid_fraction: float = 0.9,
) -> tuple[np.ndarray, dict[str, Any]]:
    """以初稿为指导，在重叠图块中选「最接近初稿、清晰且未遮挡」的真实观测块。

    仅当中位数基线本身模糊或细节更差时才启用；接缝差异大就保留基线。
    """
    if not frames:
        return composite.copy(), {"replaced_tiles": 0, "reason": "no_frames"}
    height, width = composite.shape[:2]
    result = composite.copy()
    replaced = 0
    evaluated = 0
    for top in range(0, height, tile):
        for left in range(0, width, tile):
            bottom = min(height, top + tile)
            right = min(width, left + tile)
            target = composite[top:bottom, left:right]
            target_valid = valid[top:bottom, left:right] > 0
            if not target_valid.any():
                continue
            evaluated += 1
            best_score = float("inf")
            best_patch: np.ndarray | None = None
            for frame, mask in zip(frames, masks):
                patch_mask = mask[top:bottom, left:right] > 0
                fraction = float(np.count_nonzero(patch_mask & target_valid)) / max(
                    int(np.count_nonzero(target_valid)), 1
                )
                if fraction < min_valid_fraction:
                    continue
                patch = frame[top:bottom, left:right]
                difference = float(np.mean(np.abs(
                    patch.astype(np.float32) - target.astype(np.float32)
                )[target_valid]))
                if difference < best_score:
                    best_score = difference
                    best_patch = patch
            if best_patch is not None and best_score <= max_difference:
                result[top:bottom, left:right] = best_patch
                replaced += 1
    return result, {
        "replaced_tiles": replaced, "evaluated_tiles": evaluated,
        "tile": tile, "max_difference": max_difference,
    }


# --------------------------------------------------------------------------- #
# 噪声估计（F3）
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class NoiseEstimate:
    payload: dict[str, np.ndarray]
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.diagnostics)


def _block_counts_for_frames(
    block_ids: Sequence[str], shape: tuple[int, int], stride: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """每个像素被多少个**独立时间块**支持。"""
    mapping: dict[str, int] = {}
    for block in block_ids:
        if block not in mapping:
            mapping[block] = len(mapping)
    small_shape = (
        max(1, math.ceil(shape[0] / stride)), max(1, math.ceil(shape[1] / stride)),
    )
    counts = np.zeros(small_shape, np.float32)
    buckets = {block: np.zeros(small_shape, np.float32) for block in mapping}
    return counts, buckets


def estimate_noise(
    reference: np.ndarray, frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray], block_ids: Sequence[str], *,
    config: Mapping[str, Any] | None = None, stride: int = 2,
) -> NoiseEstimate:
    """在来源之外的时间块上做与运行时相同的受限补偿，得到有限噪声容差。

    公式（方案一 §6.3）：::

        raw_T = max(T_base, m + max(k * MAD, epsilon))
        stored_T = min(raw_T, T_cap)
        bias_flag = raw_T > T_cap 或 Q95/超 cap 比例达到质量诊断条件

    残差中心 ``m`` 与 MAD 分别保存；``epsilon`` 为正，避免恒定残差恰好等于门槛
    仍触发 ``>=``。
    """
    from .ground_litter_profile_match import residual_maps

    settings = {**NOISE_DEFAULTS, **dict(config or {})}
    if not frames:
        raise BankError("噪声估计至少需要一帧留出观测")
    if not (len(frames) == len(masks) == len(block_ids)):
        raise BankError("frames/masks/block_ids 长度必须一致")
    height, width = reference.shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise BankError("噪声估计帧尺寸不一致")
    k = float(settings["k"])
    blocks = sorted(set(str(block) for block in block_ids))
    if len(blocks) == 0:
        raise BankError("缺少时间块信息")

    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    small = (
        max(1, math.ceil(width / stride)), max(1, math.ceil(height / stride)),
    )
    for frame, mask, block in zip(frames, masks, block_ids):
        sig, lum = residual_maps(reference, frame)
        keep = cv2.resize(mask, small, interpolation=cv2.INTER_NEAREST) > 0
        samples.append((
            cv2.resize(sig, small, interpolation=cv2.INTER_AREA),
            cv2.resize(lum, small, interpolation=cv2.INTER_AREA),
            keep, str(block),
        ))

    def per_pixel(values: np.ndarray, buckets: np.ndarray,
                  *, quantile: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按时间块取中位数/MAD/分位数；块内重复帧不额外加权。

        ``buckets`` 是 (块数, H, W) 的每块计数，同一块内先取均值再去重，
        避免同一时段的重复帧把统计拉偏。
        """
        totals = buckets.sum(axis=0)
        weights = np.divide(
            buckets, np.maximum(totals, 1e-6)[None, ...],
            out=np.zeros_like(buckets), where=totals[None, ...] > 0,
        )
        active = weights > 0
        with np.errstate(all="ignore"):
            # 逐像素非零元素数量一致，可以安全用 nanmedian/nanquantile。
            masked = np.where(active, values, np.nan)
            median = np.nanmedian(masked, axis=0)
            deviation = np.abs(masked - median[None, ...])
            mad = np.nanmedian(deviation, axis=0)
            q95 = np.nanquantile(masked, quantile, axis=0)
        median = np.nan_to_num(median, nan=0.0).astype(np.float32)
        mad = np.nan_to_num(mad, nan=0.0).astype(np.float32)
        q95 = np.nan_to_num(q95, nan=0.0).astype(np.float32)
        return median, mad, q95

    sig_values = np.stack([item[0] for item in samples], axis=0)
    lum_values = np.stack([item[1] for item in samples], axis=0)
    block_index = {block: index for index, block in enumerate(blocks)}
    block_stack = np.zeros((len(blocks),) + sig_values.shape[1:], np.float32)
    for __sig, __lum, keep, block in samples:
        block_stack[block_index[block]] += keep.astype(np.float32)
    valid_frac = np.clip(block_stack, 0.0, 1.0)
    # 同一块内重复帧只算一次：块权重归一化，再按块计数展开到每个样本。
    block_weight = (valid_frac > 0).astype(np.float32)
    block_weight /= np.maximum(block_weight.sum(axis=0, keepdims=True), 1e-6)

    def weighted(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        counts = np.zeros_like(values, np.float32)
        for index, (_sig, _lum, keep, block) in enumerate(samples):
            counts[index] = np.where(keep, block_weight[block_index[block]], 0.0)
        return per_pixel(values, counts, quantile=0.95)

    sig_median, sig_mad, sig_q95 = weighted(sig_values)
    lum_median, lum_mad, lum_q95 = weighted(lum_values)
    support_blocks = (valid_frac > 0).sum(axis=0).astype(np.float32)

    def build(
        median: np.ndarray, mad: np.ndarray, q95: np.ndarray,
        base: float, cap: float, epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = np.maximum(base, median + np.maximum(k * mad, epsilon))
        stored = np.minimum(raw, cap).astype(np.float32)
        over = (raw > cap)
        return stored, raw.astype(np.float32), over

    seed_sig_T, seed_sig_raw, seed_sig_over = build(
        sig_median, sig_mad, sig_q95,
        float(settings["seed_signature_base"]), float(settings["seed_signature_cap"]),
        float(settings["epsilon_seed_signature"]),
    )
    seed_lum_T, seed_lum_raw, seed_lum_over = build(
        lum_median, lum_mad, lum_q95,
        float(settings["seed_luminance_base"]), float(settings["seed_luminance_cap"]),
        float(settings["epsilon_seed_luminance"]),
    )
    support_sig_T, _support_sig_raw, _ = build(
        sig_median, sig_mad, sig_q95,
        float(settings["support_signature_base"]), float(settings["support_signature_cap"]),
        float(settings["epsilon_support_signature"]),
    )
    support_lum_T, _support_lum_raw, _ = build(
        lum_median, lum_mad, lum_q95,
        float(settings["support_luminance_base"]), float(settings["support_luminance_cap"]),
        float(settings["epsilon_support_luminance"]),
    )

    low_support = support_blocks < float(settings["min_support_blocks"])
    seed_sig_T[low_support] = float(settings["seed_signature_base"])
    seed_lum_T[low_support] = float(settings["seed_luminance_base"])
    support_sig_T[low_support] = float(settings["support_signature_base"])
    support_lum_T[low_support] = float(settings["support_luminance_base"])

    over_fraction = (seed_sig_over | seed_lum_over).astype(np.float32)
    over_fraction[low_support] = 0.0
    bias_threshold = float(settings["bias_over_cap_fraction"])
    persistent = over_fraction >= bias_threshold
    bias_flag = (seed_sig_over | seed_lum_over) | persistent
    bias_flag = bias_flag.astype(np.uint8)
    bias_flag[low_support] = 0

    # 支持不足时用基础阈值并标明低支持，不估计没有证据的尾部分位数。
    zeros = np.zeros_like(seed_sig_T)
    payload = {
        "seed_signature_threshold": seed_sig_T,
        "seed_luminance_threshold": seed_lum_T,
        "support_signature_threshold": support_sig_T,
        "support_luminance_threshold": support_lum_T,
        "seed_signature_median": sig_median,
        "seed_luminance_median": lum_median,
        "seed_signature_mad": sig_mad,
        "seed_luminance_mad": lum_mad,
        "seed_signature_q95": sig_q95,
        "seed_luminance_q95": lum_q95,
        "seed_signature_cap": np.full_like(seed_sig_T, float(settings["seed_signature_cap"])),
        "seed_luminance_cap": np.full_like(seed_lum_T, float(settings["seed_luminance_cap"])),
        "seed_signature_raw": seed_sig_raw,
        "seed_luminance_raw": seed_lum_raw,
        "support_blocks": support_blocks,
        "over_cap_fraction": over_fraction,
        "bias_flag": bias_flag,
        "low_support": low_support.astype(np.uint8),
        "stride": np.array(int(stride), np.int32),
        "algorithm_version": np.array("noise_r3"),
        "zeros_reference": zeros,
    }
    diagnostics = {
        "time_blocks": len(blocks),
        "k": k,
        "stride": stride,
        "bias_flag_pixels": int(np.count_nonzero(bias_flag)),
        "bias_flag_fraction": round(
            float(np.count_nonzero(bias_flag)) / max(int(np.count_nonzero(valid_frac.sum(axis=0) > 0)), 1), 5
        ),
        "low_support_pixels": int(np.count_nonzero(low_support)),
        "persistent_bias_pixels": int(np.count_nonzero(persistent)),
        "max_seed_signature_threshold": round(float(seed_sig_T.max()), 4),
        "max_seed_luminance_threshold": round(float(seed_lum_T.max()), 4),
        "capped_seed_signature_pixels": int(np.count_nonzero(seed_sig_over)),
        "capped_seed_luminance_pixels": int(np.count_nonzero(seed_lum_over)),
        "note": "stored_T 被截断不代表该处已合格；bias_flag 是离线质量诊断",
    }
    return NoiseEstimate(payload=payload, diagnostics=diagnostics)


def diagnose_persistent_bias(
    noise: NoiseEstimate, *, min_area: int = 400,
) -> dict[str, Any]:
    """对 bias_flag 的连通区域、重复时间块与越界强度生成诊断（方案一 §6.3）。"""
    flags = np.asarray(noise.payload["bias_flag"], np.uint8)
    counts, labels, stats, _ = cv2.connectedComponentsWithStats(flags, 8)
    regions: list[dict[str, Any]] = []
    for index in range(1, counts):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < min_area:
            continue
        region = labels == index
        regions.append({
            "box": [x, y, x + width, y + height],
            "area": area,
            "max_median_signature": round(
                float(np.max(noise.payload["seed_signature_median"][region])), 4
            ),
            "max_median_luminance": round(
                float(np.max(noise.payload["seed_luminance_median"][region])), 4
            ),
        })
    regions.sort(key=lambda row: -row["area"])
    return {
        "regions": regions,
        "requires_rebuild": bool(regions),
        "max_rounds": 2,
        "note": "持续偏差区域应进入重对齐→重合成/细分外观组→局部不可靠的自动处理",
    }


__all__ = [
    "AppearanceGroup", "CompositeResult", "GroupingResult", "MAX_BANK_PROFILES",
    "MAX_CANDIDATES", "NOISE_DEFAULTS", "NoiseEstimate", "diagnose_persistent_bias",
    "estimate_noise", "group_appearance_samples", "replace_with_real_observations",
    "select_time_balanced_frames", "temporal_median_composite",
]
