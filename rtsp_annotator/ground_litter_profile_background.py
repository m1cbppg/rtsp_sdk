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
    # 逐像素 95 分位是 O(帧数) 排序：必须限制参与估计的帧数，
    # 否则 5 分钟级 2.5K 帧数会把内存与耗时都放大到不可接受。
    "max_estimate_frames": 24,
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


def observation_mask_from_frames(
    frames: Sequence[np.ndarray], roi: np.ndarray, *,
    block_ids: Sequence[str] | None = None,
    deviant_fraction: float = 0.35, motion_threshold: float = 12.0,
    min_area: int = 64, dilate: int = 5,
) -> tuple[np.ndarray, dict[str, Any], list[np.ndarray]]:
    """由真实观测导出「可用于合成」的逐帧掩膜（R7）。

    两类像素被排除：

    * **移动物体**：该帧与其它帧（同组）差异很大，说明这一帧在该处看到的是
      人/车/短时运动，而不是稳定背景；
    * **持续运动区域**：帧间最大差异高的像素（例如一直有人经过的地带）。

    这是纯观测统计，不需要模型：对一小组帧逐像素比较即可。排除只影响合成，
    **不**影响发布的 ROI；用户允许长期静止物体进入背景。
    """
    if not frames:
        raise BankError("observation_mask_from_frames 需要至少一帧")
    height, width = frames[0].shape[:2]
    if roi.shape != (height, width):
        raise BankError("ROI 与帧尺寸不一致")
    stack = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) for frame in frames],
        axis=0,
    )
    count = stack.shape[0]
    masks: list[np.ndarray] = []
    deviant_counts = np.zeros((height, width), np.float32)
    for index in range(count):
        others = np.delete(stack, index, axis=0)
        if others.shape[0] == 0:
            deviation = np.zeros((height, width), np.float32)
        else:
            median = np.median(others, axis=0)
            deviation = np.abs(stack[index] - median)
        moving = (deviation > motion_threshold) & (roi > 0)
        moving = cv2.morphologyEx(
            moving.astype(np.uint8), cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )
        moving = cv2.dilate(moving, np.ones((dilate, dilate), np.uint8))
        mask = np.zeros((height, width), np.uint8)
        mask[(roi > 0) & (moving == 0)] = 255
        masks.append(mask)
        deviant_counts += (deviation > motion_threshold).astype(np.float32)
    # 持续被判定为运动的像素（跨多数帧）视为长期遮挡地带，标为局部不可用。
    persistent = (deviant_counts >= max(1.0, count * deviant_fraction))
    persistent = cv2.morphologyEx(
        persistent.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8),
    )
    persistent = cv2.dilate(persistent, np.ones((dilate, dilate), np.uint8))
    availability = np.zeros((height, width), np.uint8)
    availability[(roi > 0) & (persistent == 0)] = 255
    if min_area > 0 and persistent.any():
        count_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
            persistent.astype(np.uint8), 8,
        )
        for cc in range(1, count_cc):
            if int(stats[cc, cv2.CC_STAT_AREA]) < min_area:
                area_mask = (labels == cc)
                availability[area_mask & (roi > 0)] = 255
    available_fraction = float(np.count_nonzero(availability)) / max(
        int(np.count_nonzero(roi)), 1
    )
    return availability, {
        "frames": count,
        "roi_pixels": int(np.count_nonzero(roi)),
        "available_pixels": int(np.count_nonzero(availability)),
        "available_fraction_of_roi": round(available_fraction, 5),
        "persistent_deviant_pixels": int(np.count_nonzero(persistent)),
        "method": "per-frame deviation vs median-of-others",
        "blocks": list(block_ids or []),
    }, masks


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


def _weighted_quantile_counts(
    values: np.ndarray, weights: np.ndarray, quantile: float,
    totals: np.ndarray, *, iterations: int = 32,
) -> np.ndarray:
    """逐像素加权分位数：二分阈值而不是排序。

    ``values`` 形状 (frames, n)，``weights`` 同形且非负。返回长度 n 的分位数。
    对 24 帧、数百万像素的场景，这比 ``np.nanquantile`` 快一个数量级。
    """
    low = np.nanmin(values, axis=0).astype(np.float64)
    high = np.nanmax(values, axis=0).astype(np.float64)
    target = quantile * np.maximum(totals, 0.0)
    for _ in range(int(iterations)):
        mid = 0.5 * (low + high)
        below = ((values <= mid[None, :]) * np.where(np.isnan(values), 0.0, weights)).sum(axis=0)
        move_up = below < target
        low = np.where(move_up, mid, low)
        high = np.where(move_up, high, mid)
    return (0.5 * (low + high)).astype(np.float32)


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


def select_calibration_observations(
    groups: Sequence[Mapping[str, Any]],
    calibration_samples: Sequence[Mapping[str, Any]], *,
    descriptors: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    group_members: Mapping[str, Sequence[str]] | None = None,
    radius: float | None = None, max_distance: float | None = None,
    tolerance: float = 0.6,
) -> dict[str, Any]:
    """把独立校准观测按**外观匹配**分配到各候选组（C3）。

    接线缺陷的根源是：校准日的样本从未进入合成阶段，``calibration_blocks``
    恒为空，于是所有 Profile 的噪声都退化成"用参考自己的观测"，并以
    ``low_support_used_all_blocks`` 发布。这里显式做两件事：

    1. 独立校准观测按自身外观聚类（不参与参考合成与分组）；
    2. 每个候选组只接受**外观接近自己**的校准观测。接近程度用"该组自身成员的
       外观离散度"作标尺（``group_members`` 提供），而不是一个与场景无关的
       绝对阈值；显式传 ``max_distance`` 时才改用绝对阈值。

    匹配不上时宁可标注 ``no_appearance_match`` 并退化，也不允许拿别的场景
    冒充独立校准；同时记录每个组的匹配距离与近失距离，便于事后复核。
    """
    payloads: dict[str, Mapping[str, np.ndarray]] = dict(descriptors or {})
    for item in calibration_samples:
        key = _item_key(item)
        if key not in payloads and "descriptor" in item:
            payloads[key] = item["descriptor"]  # type: ignore[assignment]
    empty = {
        "per_group": {
            str(group.get("group_id")): {
                "blocks": [], "distance": None, "reason": "no_calibration_samples",
            } for group in groups
        },
        "clusters": 0,
        "assigned_observations": 0,
        "unassigned_observations": len(calibration_samples),
        "scale": {},
    }
    if not groups or not calibration_samples:
        return empty
    missing = [key for key in (_item_key(item) for item in calibration_samples)
               if key not in payloads]
    if missing:
        empty["unassigned_reason"] = f"missing_descriptor:{missing[:3]}"
        return empty

    clustering = group_appearance_samples(
        list(calibration_samples), descriptors=payloads, radius=radius,
    )
    scale = clustering.scale or global_descriptor_scale(
        [descriptor_arrays(payloads[_item_key(item)]) for item in calibration_samples]
    )
    # 标尺下限：候选组自身的离散度只能描述"同一段素材内部"的波动；校准日与
    # 构建日之间的真实光照/色调差通常大于它（实测 group_diameter 只有十几，
    # 跨日校准距离二十几，结果 13/13 组都匹配不上）。因此允许放宽到校准观测
    # **整体**的公共离散尺度（`clustering.statistics["requested_radius"]`），
    # 但绝不放开到"任意观测都算独立校准"：匹配距离、限值与近失都逐组留证。
    global_radius = float(clustering.statistics.get("requested_radius") or 0.0)
    fallback_scale = max(global_radius, 1.0)
    # 每个组的"自身离散度"标尺：组内成员两两距离最大值（至少取一个下限，
    # 避免只有一个成员时退化成 0 导致任何观测都被拒）。
    members_map = dict(group_members or {})
    group_arrays: dict[str, Mapping[str, np.ndarray]] = {}
    group_limits: dict[str, float] = {}
    for group in groups:
        gid = str(group.get("group_id"))
        member_keys = [
            str(key) for key in (members_map.get(gid) or [])
        ] or [str(group.get("representative_key") or "")]
        arrays = [descriptor_arrays(payloads[key]) for key in member_keys
                  if key in payloads]
        if not arrays:
            continue
        group_arrays[gid] = arrays[0]
        diameter = 0.0
        for left in range(len(arrays)):
            for right in range(left + 1, len(arrays)):
                diameter = max(diameter, descriptor_coarse_distance(
                    arrays[left], arrays[right], scale=scale,
                ))
        group_limits[gid] = max(
            diameter * (1.0 + tolerance),
            fallback_scale * max(tolerance, 0.5),
            1e-6,
        )
    absolute = float(max_distance) if max_distance is not None else None
    per_group: dict[str, dict[str, Any]] = {
        str(group.get("group_id")): {
            "blocks": [], "distances": [], "near_misses": [], "reason": None,
            "limit": (
                round(absolute, 5) if absolute is not None
                else round(group_limits.get(str(group.get("group_id")), 0.0), 5)
            ),
        }
        for group in groups
    }
    assigned = 0
    unassigned = 0
    for item in calibration_samples:
        key = _item_key(item)
        arrays = descriptor_arrays(payloads[key])
        candidates = [
            (descriptor_coarse_distance(arrays, group_arrays[gid], scale=scale), gid)
            for gid in group_arrays
        ]
        if not candidates:
            unassigned += 1
            continue
        distance, gid = min(candidates)
        limit = absolute if absolute is not None else group_limits.get(gid, 0.0)
        if limit > 0 and distance > limit:
            # 外观不匹配：宁可不分配，也不串用别的场景当独立校准。
            unassigned += 1
            per_group[gid]["near_misses"].append({
                "time_block": str(item.get("time_block")),
                "distance": round(float(distance), 5),
                "limit": round(float(limit), 5),
            })
            continue
        block = str(item.get("time_block") or key)
        if block not in per_group[gid]["blocks"]:
            per_group[gid]["blocks"].append(block)
        per_group[gid]["distances"].append(round(float(distance), 5))
        assigned += 1
    for gid, payload in per_group.items():
        if not payload["blocks"]:
            payload["reason"] = "no_appearance_match"
        payload["distance"] = (
            round(max(payload["distances"]), 5) if payload["distances"] else None
        )
        payload["blocks"].sort()
        payload["near_misses"] = sorted(
            payload["near_misses"], key=lambda row: row["distance"],
        )[:8]
    return {
        "per_group": per_group,
        "clusters": len(clustering.groups),
        "assigned_observations": assigned,
        "unassigned_observations": unassigned,
        "match_threshold_mode": "absolute" if absolute is not None else "group_diameter",
        "tolerance": tolerance,
        "scale": {key: round(float(value), 6) for key, value in scale.items()},
        "statistics": dict(clustering.statistics),
    }


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
    limit = int(settings.get("max_estimate_frames", 24))
    if limit > 0 and len(frames) > limit:
        # 按时间均匀抽稀，保持时间块覆盖，同时限制逐像素分位的代价。
        indexes = sorted({
            round(index * (len(frames) - 1) / (limit - 1)) for index in range(limit)
        }) if limit > 1 else [0]
        frames = [frames[index] for index in indexes]
        masks = [masks[index] for index in indexes]
        block_ids = [block_ids[index] for index in indexes]
    height, width = reference.shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise BankError("噪声估计帧尺寸不一致")
    k = float(settings["k"])
    blocks = sorted(set(str(block) for block in block_ids))
    if len(blocks) == 0:
        raise BankError("缺少时间块信息")

    small_h = max(1, math.ceil(height / stride))
    small_w = max(1, math.ceil(width / stride))
    cv_size = (small_w, small_h)          # cv2.resize 需要 (宽, 高)
    small = (small_h, small_w)            # numpy 数组是 (行, 列)
    # R8：每个观测只在自己**实际有效**的像素上计权。之前用全组并集当权重，
    # 会把某块被遮挡的像素也算成有支持，支持计数和分位都被污染。
    blocks = sorted(set(str(block) for block in block_ids))
    block_index = {block: index for index, block in enumerate(blocks)}
    resized: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    support = np.zeros(small, bool)
    for frame, mask, block in zip(frames, masks, block_ids):
        sig, lum = residual_maps(reference, frame)
        keep = cv2.resize(mask, cv_size, interpolation=cv2.INTER_NEAREST) > 0
        sig_s = cv2.resize(sig, cv_size, interpolation=cv2.INTER_AREA)
        lum_s = cv2.resize(lum, cv_size, interpolation=cv2.INTER_AREA)
        support |= keep
        resized.append((sig_s, lum_s, keep, str(block)))
    rows, cols = np.nonzero(support)
    n = int(rows.size)
    zeros = np.zeros(small, np.float32)
    if n == 0:
        raise BankError("噪声估计没有可用的有效像素")

    # 每个 (时间块, 像素) 的权重：该块在该像素有效则记 1；同一块出现多次
    # （重复观测）只聚合一次，不额外加权。
    sig_values = np.zeros((len(blocks), n), np.float32)
    lum_values = np.zeros((len(blocks), n), np.float32)
    weights = np.zeros((len(blocks), n), np.float32)
    for sig_s, lum_s, keep, block in resized:
        index = block_index[block]
        sig_values[index, :] = np.maximum(sig_values[index, :], sig_s[rows, cols])
        lum_values[index, :] = np.maximum(lum_values[index, :], lum_s[rows, cols])
        weights[index, :] = np.maximum(
            weights[index, :], keep[rows, cols].astype(np.float32),
        )

    totals = weights.sum(axis=0)
    support_blocks = (weights > 0).sum(axis=0).astype(np.float32)

    def weighted_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        median = _weighted_quantile_counts(values, weights, 0.5, totals)
        deviation = np.abs(values - median[None, :])
        mad = _weighted_quantile_counts(deviation, weights, 0.5, totals)
        q95 = _weighted_quantile_counts(values, weights, 0.95, totals)
        return median, mad, q95

    sig_median_sel, sig_mad_sel, sig_q95_sel = weighted_stats(sig_values)
    lum_median_sel, lum_mad_sel, lum_q95_sel = weighted_stats(lum_values)

    def scatter(values: np.ndarray) -> np.ndarray:
        out = zeros.copy()
        out[rows, cols] = values
        return out

    sig_median = scatter(sig_median_sel)
    sig_mad = scatter(sig_mad_sel)
    sig_q95 = scatter(sig_q95_sel)
    lum_median = scatter(lum_median_sel)
    lum_mad = scatter(lum_mad_sel)
    lum_q95 = scatter(lum_q95_sel)
    support_out = scatter(support_blocks)
    # 逐块有效像素数（供报告核对：某块只覆盖半幅时不应计满）
    per_block_valid_pixels = {
        block: int(np.count_nonzero(weights[block_index[block]]))
        for block in blocks
    }
    del sig_values, lum_values, weights

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

    low_support = support_out < float(settings["min_support_blocks"])
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
        "support_blocks": support_out,
        "over_cap_fraction": over_fraction,
        "bias_flag": bias_flag,
        "low_support": low_support.astype(np.uint8),
        "stride": np.array(int(stride), np.int32),
        "algorithm_version": np.array("noise_r3"),
        "zeros_reference": zeros,
    }
    diagnostics = {
        "time_blocks": len(blocks),
        "per_block_valid_pixels": per_block_valid_pixels,
        "calibration_blocks": int(settings.get("_calibration_block_count", 0) or 0),
        "independent_of_reference": bool(settings.get("_independent_calibration", False)),
        "k": k,
        "stride": stride,
        "bias_flag_pixels": int(np.count_nonzero(bias_flag)),
        "bias_flag_fraction": round(
            float(np.count_nonzero(bias_flag)) / max(int(np.count_nonzero(support_out > 0)), 1), 5
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
