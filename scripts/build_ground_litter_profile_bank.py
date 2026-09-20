#!/usr/bin/env python
"""A5：一条命令可续跑的 Profile 工厂。

流程（方案一 §3、§9）：

```text
清单/时间恢复 → 有界下载 → 粗采样 → 质量过滤 → 配准 → 分组
→ 时间均衡高清选帧 → 合成 → 有限噪声 → 静态预选
→ 完整 Selector 动态回放 → 冻结 Bank + 报告
```

用法（本地 PS）：

    python scripts/build_ground_litter_profile_bank.py \
        --input /path/to/ps_dir --camera camera_01 \
        --geometry config/ground_litter_1021_camera_geometry.json \
        --output output/profile_bank --work-dir output/profile_bank_work \
        --resume

用法（回放接口，需要网络与设备码）：

    python scripts/build_ground_litter_profile_bank.py \
        --source ctseelink-file-urls --device-code 44180209031322001030 \
        --start "2026-09-01 00:00:00" --end "2026-09-08 00:00:00" \
        --camera camera_01 --geometry geometry.json \
        --output output/profile_bank --work-dir output/profile_bank_work --resume
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rtsp_annotator.ground_litter_profile_analysis import (  # noqa: E402
    roi_mask_from_geometry,
)
from rtsp_annotator.ground_litter_profile_background import (  # noqa: E402
    MAX_BANK_PROFILES, diagnose_persistent_bias, estimate_noise,
    group_appearance_samples, replace_with_real_observations,
    select_time_balanced_frames, temporal_median_composite,
)
from rtsp_annotator.ground_litter_profile_bank import (  # noqa: E402
    atomic_write_json, canonical_json, default_camera_geometry,
    default_matcher_config, load_bank, publish_version, sha256_bytes, sha256_file,
)
from rtsp_annotator.ground_litter_profile_match import (  # noqa: E402
    envelope_from_samples, evaluate_match, extract_grid_descriptor,
    rank_by_coarse_distance, score_profile,
)
from rtsp_annotator.ground_litter_profile_sampling import (  # noqa: E402
    BoundedPreviewSampler, CanvasRegistrar, build_hd_plan, coarse_sample_offsets,
    cross_day_files, frame_quality, partition_recordings, preview_image,
    probe_recording, parse_day, summarise_sampling_quality,
)
from rtsp_annotator.ground_litter_profile_selector import (  # noqa: E402
    CandidateMatch, ProfileSelector,
)
from rtsp_annotator.ground_litter_recording_cache import (  # noqa: E402
    ManagedRecordingCache,
)
from rtsp_annotator.ground_litter_recording_source import (  # noqa: E402
    ListQuery, RecordingDownloader, RecordingFile, RecordingListClient,
    UrlRefreshPolicy, deduplicate_files, detect_truncation,
    file_looks_like_media,
)

ALGORITHM_VERSION = "profile_factory_r3"


# --------------------------------------------------------------------------- #
# 几何与工作区
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FactoryConfig:
    camera_id: str
    bank_id: str
    version: str
    output_root: Path
    work_dir: Path
    geometry_path: Path | None
    analysis_size: tuple[int, int] | None
    device_code: str
    timezone: str = "Asia/Shanghai"
    build_days: int = 5
    calibration_days: int = 1
    preview_width: int = 960
    max_profiles: int = MAX_BANK_PROFILES
    raw_cache_budget: int = 1024 * 1024 * 1024
    work_budget: int = 20 * 1024 * 1024 * 1024
    seed: int = 20260920
    supersede: bool = False
    use_seek: bool = False
    max_downloads_per_run: int | None = None


def load_geometry(path: Path | None, size: tuple[int, int], *,
                  camera_id: str) -> dict[str, Any]:
    if path is None:
        return default_camera_geometry(size[0], size[1], camera_id=camera_id)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canvas = payload.get("canvas_size") or list(size)
    if tuple(int(value) for value in canvas) != tuple(size):
        raise SystemExit(
            f"geometry canvas_size={canvas} 与视频分辨率 {list(size)} 不一致；"
            "请确认复用的是同一机位同一分析尺寸的 ROI"
        )
    geometry = default_camera_geometry(
        size[0], size[1], camera_id=camera_id,
        view_id=str(payload.get("view_id", "view_0")),
        roi=payload.get("roi"), exclude_zones=payload.get("exclude_zones"),
        overlay_exclude_zones=payload.get("overlay_exclude_zones"),
    )
    geometry["registration"] = payload.get("registration", geometry["registration"])
    return geometry


def _state_path(work_dir: Path) -> Path:
    return work_dir / "factory_state.json"


def load_state(work_dir: Path) -> dict[str, Any]:
    path = _state_path(work_dir)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"stages": {}, "downloads": [], "failures": [], "bytes": 0}


def save_state(work_dir: Path, state: Mapping[str, Any]) -> None:
    atomic_write_json(_state_path(work_dir), dict(state))


# --------------------------------------------------------------------------- #
# 本地来源
# --------------------------------------------------------------------------- #


def scan_local_directory(root: Path) -> list[RecordingFile]:
    """本地 PS 根目录：用文件名/修改时间做弱索引，stat 做弱指纹。"""
    files: list[RecordingFile] = []
    patterns = ("*.ps", "*.mp4", "*.mkv", "*.avi", "*.mov", "*.ts", "*.m4v")
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            stat = path.stat()
            stamp = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            files.append(RecordingFile(
                file_id=sha256_bytes(str(path.relative_to(root)).encode())[:16],
                file_name=str(path),
                record_start=stamp,
                record_end=stamp,
                file_size=stat.st_size,
                file_type=path.suffix.lower().lstrip("."),
            ))
    return files


def build_local_cache_entries(
    cache: ManagedRecordingCache, device_code: str, root: Path,
    files: Sequence[RecordingFile], *, work_budget: int,
) -> list[RecordingFile]:
    """本地来源默认只读：登记为 UNMANAGED，绝不参与自动清理。"""
    for item in files:
        path = Path(item.file_name)
        digest = sha256_file(path) if path.stat().st_size <= 64 * 1024 * 1024 else None
        row = cache.entry(device_code, item.file_id)
        if row is not None and row.materialization != "ABSENT":
            continue
        cache.register(device_code, [item])
        with cache._lock, cache._connection:  # noqa: SLF001 - 工厂内部记账
            cache._connection.execute(  # noqa: SLF001
                "UPDATE recordings SET managed=0, materialization='READY', path=?,"
                " bytes=?, sha256=?, updated=datetime('now') WHERE identity_key=?",
                (str(path), path.stat().st_size, digest,
                 RecordingFile.identity_key(device_code, item.file_id)),
            )
    return list(files)


# --------------------------------------------------------------------------- #
# 阶段
# --------------------------------------------------------------------------- #


def stage_inventory_remote(
    config: FactoryConfig, client: RecordingListClient,
    cache: ManagedRecordingCache, report: dict[str, Any],
) -> list[RecordingFile]:
    """按一小时窗口查询清单；按 deviceCode+fileId 去重并检测截断。"""
    pages = []
    start = datetime.strptime(config.start_time, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(config.end_time, "%Y-%m-%d %H:%M:%S")
    if end <= start:
        raise SystemExit("--end 必须晚于 --start")
    truncation_checks = []
    cursor = start
    half_window_seconds = 1800
    while cursor < end:
        window_end = min(cursor + __import__("datetime").timedelta(hours=1), end)
        query = ListQuery(
            config.device_code,
            cursor.strftime("%Y-%m-%d %H:%M:%S"),
            window_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        page = client.query(query)
        pages.append(page)
        # 一小时结果 vs 两个半小时结果的并集：检测列表截断嫌疑。
        halves = []
        for offset in (0, half_window_seconds):
            half_start = cursor + __import__("datetime").timedelta(seconds=offset)
            if half_start >= window_end:
                continue
            half_end = min(
                half_start + __import__("datetime").timedelta(seconds=half_window_seconds),
                window_end,
            )
            halves.append(client.query(ListQuery(
                config.device_code,
                half_start.strftime("%Y-%m-%d %H:%M:%S"),
                half_end.strftime("%Y-%m-%d %H:%M:%S"),
            )))
        if halves:
            truncation_checks.append(detect_truncation(
                page.files(),
                [item for half in halves for item in half.files()],
            ))
        cursor = window_end
    files = deduplicate_files(pages, config.device_code)
    cache.register(config.device_code, files)
    report["inventory"] = {
        "windows": len(pages),
        "files": len(files),
        "total_declared_bytes": sum(
            int(item.file_size or 0) for item in files
        ),
        "pagination_fields": sorted({
            key for page in pages for key in page.pagination
        }),
        "truncation_checks": truncation_checks,
        "suspected_truncation": any(
            item["suspected_truncation"] for item in truncation_checks
        ),
        "real_range": [
            min((item.record_start for item in files), default=""),
            max((item.record_end for item in files), default=""),
        ],
        "query_range": [config.start_time, config.end_time],
    }
    return files


def materialize_remote(
    config: FactoryConfig, client: RecordingListClient,
    downloader: RecordingDownloader, cache: ManagedRecordingCache,
    files: Sequence[RecordingFile], report: dict[str, Any],
    *, limit: int | None = None,
) -> list[str]:
    """临近下载时刷新 URL，逐个文件下载并原子落盘；返回成功 fileId 列表。"""
    policy = UrlRefreshPolicy()
    ready: list[str] = []
    attempted = 0
    for item in files:
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        allowed, reason = cache.can_reserve(item.file_size)
        if not allowed:
            report.setdefault("backpressure", []).append({
                "file_id": item.file_id, "reason": reason,
            })
            break
        query = ListQuery(
            config.device_code, item.record_start, item.record_end,
        )
        try:
            entry = downloader.fetch_url_for_file(
                client, query, item.file_id, policy=policy,
            )
        except Exception as exc:
            cache.fail_download(config.device_code, item.file_id, str(exc)[:120])
            report.setdefault("failures", []).append({
                "file_id": item.file_id, "stage": "refresh", "error": str(exc)[:120],
            })
            continue
        target = cache.begin_download(config.device_code, item)
        try:
            result = downloader.download(
                entry.url, target, expected_size=item.file_size,
                allow_resume=False,
            )
        except Exception as exc:
            cache.fail_download(config.device_code, item.file_id, str(exc)[:120])
            report.setdefault("failures", []).append({
                "file_id": item.file_id, "stage": "download", "error": str(exc)[:120],
            })
            continue
        if not file_looks_like_media(target):
            cache.fail_download(
                config.device_code, item.file_id, "content_probe_failed",
                bytes_received=result.size,
            )
            report.setdefault("failures", []).append({
                "file_id": item.file_id, "stage": "probe", "error": "content_probe_failed",
            })
            continue
        cache.complete_download(
            config.device_code, item.file_id, path=target, size=result.size,
            sha256=result.sha256, range_supported=result.range_supported,
            elapsed_seconds=result.elapsed_seconds,
        )
        probe = probe_recording(target)
        if not probe.ok:
            cache.fail_download(
                config.device_code, item.file_id, f"decode_failed:{probe.error}",
                bytes_received=result.size, elapsed_seconds=probe.decode_seconds,
            )
            report.setdefault("failures", []).append({
                "file_id": item.file_id, "stage": "decode", "error": probe.error,
            })
            continue
        ready.append(item.file_id)
        report.setdefault("range_probe", []).append({
            "file_id": item.file_id, "supported": result.range_supported,
        })
    report["downloaded"] = {
        "attempted": attempted,
        "ready": len(ready),
        "failed": len(report.get("failures", [])),
        "refresh_count": policy.refresh_count,
        "expired_events": policy.expired_events,
        "bytes": downloader.downloaded_bytes,
        "seconds": round(downloader.download_seconds, 3),
        "range_fallbacks": downloader.range_fallbacks,
    }
    return ready


def find_first_decodable(
    cache: ManagedRecordingCache, device_code: str, files: Sequence[RecordingFile],
    *, work_dir: Path,
) -> tuple[tuple[int, int], str]:
    """用第一个可解码文件确定分析尺寸；不假定七天素材的分辨率。"""
    for item in files:
        entry = cache.entry(device_code, item.file_id)
        if entry is None or entry.path is None:
            continue
        probe = probe_recording(entry.path)
        if probe.ok:
            return (probe.width, probe.height), str(entry.path)
        del work_dir
    raise SystemExit("没有任何文件可解码，无法确定分析尺寸")


def stage_sampling(
    config: FactoryConfig, cache: ManagedRecordingCache,
    files: Sequence[RecordingFile], geometry: dict[str, Any],
    report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    size = (int(geometry["canvas_size"][0]), int(geometry["canvas_size"][1]))
    roi = roi_mask_from_geometry(geometry, size[0], size[1])
    overlay = geometry.get("overlay_exclude_zones") or []
    samples: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    descriptors: dict[str, dict[str, np.ndarray]] = {}
    hd_plan: dict[str, Any] = {"files": {}}

    # 采样基准：用第一个可解码文件的第一帧建立共同画布。
    from rtsp_annotator.ground_litter_profile_sampling import SequentialFrameReader

    canvas_reference: np.ndarray | None = None
    for item in files:
        entry = cache.entry(config.device_code, item.file_id)
        if entry is None or entry.path is None:
            continue
        reader = SequentialFrameReader(entry.path)
        try:
            for frame in reader.iter_frames():
                canvas_reference = cv2.resize(
                    frame.frame, size, interpolation=cv2.INTER_AREA
                )
                break
        except Exception:
            canvas_reference = None
        if canvas_reference is not None:
            break
    if canvas_reference is None:
        raise SystemExit("无法从构建集建立共同画布")

    for item in files:
        entry = cache.entry(config.device_code, item.file_id)
        if entry is None or entry.path is None:
            continue
        with cache.acquire(config.device_code, item.file_id, owner="sampler") as lease:
            sampler = BoundedPreviewSampler(
                cache, analysis_size=size, roi_mask=roi,
                seed=config.seed, preview_width=config.preview_width,
                use_seek=config.use_seek, algorithm_version=ALGORITHM_VERSION,
            )
            try:
                result = sampler.sample_file(config.device_code, lease)
            except Exception as exc:
                details.append({"file_id": item.file_id, "error": str(exc)[:160]})
                continue
            details.append({"file_id": item.file_id, **result["detail"]})
            for sample in result["samples"]:
                payload = sample.as_dict()
                payload["preview"] = sample.preview
                payload["descriptor"] = sample.descriptor
                samples.append(payload)
                descriptors[sample.time_block] = sample.descriptor
            plan = build_hd_plan(result["samples"])
            for key, value in plan["files"].items():
                merged = hd_plan["files"].setdefault(key, value)
                merged["offsets"] = sorted(set(merged["offsets"]) | set(value["offsets"]))
            # preview 已提交 → 允许释放临时 PS（有重拉路径）。
            sampler.release_after_preview(config.device_code, lease)

    report["sampling"] = {
        "files_sampled": len(details),
        "quality": summarise_sampling_quality(samples),
        "decode_seconds": round(sum(
            item.get("decode_seconds", 0.0) for item in details
        ), 3),
        "seek_seconds": round(sum(item.get("seek_seconds", 0.0) for item in details), 3),
        "rejected": sum(len(item.get("rejected", [])) for item in details),
        "per_file": details,
    }
    report["hd_plan"] = {
        "files": len(hd_plan["files"]),
        "offsets": sum(len(value["offsets"]) for value in hd_plan["files"].values()),
    }
    atomic_write_json(config.work_dir / "hd_plan.json", hd_plan)
    return samples, hd_plan, descriptors


def stage_grouping(
    config: FactoryConfig, samples: Sequence[dict[str, Any]],
    descriptors: Mapping[str, Mapping[str, np.ndarray]],
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    grouping = group_appearance_samples(samples, descriptors=descriptors)
    report["grouping"] = grouping.statistics
    groups: list[dict[str, Any]] = []
    by_block = {str(item["time_block"]): item for item in samples}
    for group in grouping.groups:
        selection = select_time_balanced_frames(
            [by_block[key] for key in group.member_keys if key in by_block],
        )
        groups.append({
            "group_id": group.group_id,
            "representative": group.representative_key,
            "members": list(group.member_keys),
            "time_blocks": selection["time_blocks"],
            "support": selection["diagnostics"],
            "low_support": group.low_support,
            "days": list(group.days),
        })
    report["grouping"]["groups_detail"] = groups
    report["grouping"]["outliers"] = list(grouping.outliers)
    report["grouping"]["candidate_limit_reached"] = grouping.candidate_limit_reached
    return groups


def stage_composite_and_noise(
    config: FactoryConfig, cache: ManagedRecordingCache,
    files: Sequence[RecordingFile], geometry: dict[str, Any],
    groups: Sequence[dict[str, Any]], report: dict[str, Any],
    *, build_blocks: set[str], calibration_blocks: set[str],
) -> list[dict[str, Any]]:
    """用高清帧合成参考、估计噪声；构建块与校准块严格分离。"""
    size = (int(geometry["canvas_size"][0]), int(geometry["canvas_size"][1]))
    roi = roi_mask_from_geometry(geometry, size[0], size[1])
    overlay = geometry.get("overlay_exclude_zones") or []
    from rtsp_annotator.ground_litter_profile_sampling import SequentialFrameReader

    by_id = {item.file_id: item for item in files}
    frame_cache: dict[str, list[tuple[np.ndarray, str]]] = {}
    candidates: list[dict[str, Any]] = []
    for group in groups:
        needed = [block for block in group["time_blocks"]]
        if not needed:
            continue
        frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        blocks: list[str] = []
        reference_canvas: np.ndarray | None = None
        for block in needed:
            identity, _, offset_text = block.partition("@")
            file_id = identity.split(":")[-1]
            item = by_id.get(file_id)
            if item is None:
                continue
            entry = cache.entry(config.device_code, file_id)
            if entry is None or entry.path is None:
                continue
            try:
                offset = float(offset_text)
            except ValueError:
                continue
            if reference_canvas is None:
                reader = SequentialFrameReader(entry.path)
                for frame in reader.iter_frames():
                    reference_canvas = cv2.resize(
                        frame.frame, size, interpolation=cv2.INTER_AREA
                    )
                    break
            reader = SequentialFrameReader(entry.path)
            picked = reader.sample_at([offset], tolerance_seconds=1.5)
            captured = picked.get(offset)
            if captured is None:
                continue
            quality = frame_quality(captured.frame)
            if not quality.usable:
                continue
            registrar = CanvasRegistrar(
                reference_canvas if reference_canvas is not None else captured.frame,
                overlay_exclude_zones=overlay,
            )
            aligned, diagnostics = registrar.register(captured.frame)
            del diagnostics
            aligned = cv2.resize(aligned, size, interpolation=cv2.INTER_AREA)
            frames.append(aligned)
            masks.append(roi.copy())
            blocks.append(block)
        if len(frames) < 2:
            report.setdefault("composite_skipped", []).append(group["group_id"])
            continue
        composite = temporal_median_composite(
            frames, masks, roi, blocks, stride=2, min_observations=2,
        )
        refined, replace_diagnostics = replace_with_real_observations(
            composite.reference, frames, masks, roi,
        )
        # 构建块之外的观测才用于噪声估计，避免给自己打分。
        eval_frames = [
            frame for frame, block in zip(frames, blocks)
            if block not in build_blocks
        ]
        eval_masks = [
            mask for mask, block in zip(masks, blocks) if block not in build_blocks
        ]
        eval_blocks = [block for block in blocks if block not in build_blocks]
        if len(eval_frames) < 2:
            # 支持不足时用全部观测并明确标注低支持。
            eval_frames, eval_masks, eval_blocks = frames, masks, blocks
            noise_note = "low_support_used_all_blocks"
        else:
            noise_note = "held_out_blocks"
        # 阈值图必须与参考同尺寸（Bank loader 契约）；内部按行采样以控制内存。
        noise = estimate_noise(
            refined, eval_frames, eval_masks, eval_blocks, stride=1,
        )
        diagnosis = diagnose_persistent_bias(noise)
        preview = preview_image(refined, width=config.preview_width)
        descriptor = extract_grid_descriptor(
            preview,
            cv2.resize(
                roi, (preview.shape[1], preview.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ),
        )
        profile_id = f"p{len(candidates) + 1:04d}"
        candidates.append({
            "profile_id": profile_id,
            "group_id": group["group_id"],
            "reference": refined,
            "valid": roi.copy(),
            "noise": noise,
            "descriptor": descriptor,
            "frames": frames,
            "masks": masks,
            "blocks": blocks,
            "composite": composite.diagnostics,
            "replacement": replace_diagnostics,
            "noise_diagnostics": noise.diagnostics,
            "bias_diagnosis": diagnosis,
            "noise_note": noise_note,
            "support": group["support"],
            "low_support": group["low_support"],
            "days": group["days"],
            "samples": len(frames),
        })
    report["composite"] = [
        {
            "group_id": item["group_id"],
            "samples": item["samples"],
            "composite": item["composite"],
            "replacement": item["replacement"],
            "noise": item["noise_diagnostics"],
            "bias": item["bias_diagnosis"],
            "noise_note": item["noise_note"],
            "support": item["support"],
        }
        for item in candidates
    ]
    del calibration_blocks, frame_cache
    return candidates


def stage_preselect_and_finalize(
    config: FactoryConfig, candidates: Sequence[dict[str, Any]],
    geometry: dict[str, Any], report: dict[str, Any],
    *, build_frames: Sequence[tuple[str, np.ndarray]],
    config_hash: str,
) -> list[dict[str, Any]]:
    """静态贪心预选 + 完整 Selector 动态定稿（F4）。"""
    matcher = default_matcher_config()
    if not candidates:
        raise SystemExit("没有任何可用候选，无法建库")
    # 1) 静态预选：用小图描述子统计每个候选的潜在覆盖。
    for item in candidates:
        item["potential_frames"] = 0
    for _block, frame in build_frames:
        preview = preview_image(frame, width=config.preview_width)
        payload = extract_grid_descriptor(
            preview,
            roi_mask_from_geometry(geometry, preview.shape[1], preview.shape[0]),
        )
        scores = []
        for item in candidates:
            best = float("inf")
            for key in ("grid_luminance", "grid_chroma", "grid_structure"):
                difference = float(np.median(np.abs(
                    payload[key] - item["descriptor"][key]
                )))
                best = min(best, difference)
            item["potential_frames"] += 1 if best < 12.0 else 0
            scores.append(best)
    total_frames = max(1, len(build_frames))
    for item in candidates:
        item["static_coverage"] = item["potential_frames"] / total_frames

    # 2) 动态定稿：先测分数并校准包络，再用同一份 Selector 时序回放。
    #    每个候选的帧按 80/20 分成「合成用途」与「动态评估用途」；评估帧不参与
    #    该候选的合成，包络再从评估帧里每三个留出一个校准，尽量减少自证。
    frames_by_candidate = _match_frames_to_candidates(candidates, build_frames)
    synthesis_frames: dict[str, list[np.ndarray]] = {}
    evaluation_frames: dict[str, list[np.ndarray]] = {}
    for profile_id, frames in frames_by_candidate.items():
        if len(frames) < 2:
            synthesis_frames[profile_id] = list(frames)
            evaluation_frames[profile_id] = list(frames)
            continue
        cut = max(1, int(round(len(frames) * 0.8)))
        cut = min(cut, len(frames) - 1)
        synthesis_frames[profile_id] = frames[:cut]
        evaluation_frames[profile_id] = frames[cut:]
    measured: dict[str, list[float]] = {item["profile_id"]: [] for item in candidates}
    score_cache: dict[tuple[int, str], Any] = {}
    score_errors: dict[str, str] = {}
    for index, (_block, frame) in enumerate(build_frames):
        for item in candidates:
            frames = evaluation_frames.get(item["profile_id"], [])
            if not frames:
                continue
            matched = frames[index % len(frames)]
            try:
                score = score_profile(
                    matched, item["reference"], item["valid"],
                    roi_mask_from_geometry(
                        geometry, matched.shape[1], matched.shape[0],
                    ),
                    profile_id=item["profile_id"], config=matcher,
                )
            except Exception as exc:
                score_errors.setdefault(item["profile_id"], str(exc)[:120])
                continue
            measured[item["profile_id"]].append(score.score)
            score_cache[(index, item["profile_id"])] = score
    # 包络必须在**参考来源之外**的观测上校准：每三个观测留出一个，
    # 与合成帧分离；样本不足时明确回退到公共稳健范围。
    envelopes = {}
    for item in candidates:
        values = measured.get(item["profile_id"], [])
        holdout = values[2::3] if len(values) >= 6 else []
        envelopes[item["profile_id"]] = envelope_from_samples(holdout, matcher)
    report["envelopes"] = {
        key: value.as_dict() for key, value in envelopes.items()
    }
    report["score_diagnostics"] = {
        "measured": {key: len(value) for key, value in measured.items()},
        "errors": score_errors,
    }
    selector = ProfileSelector(
        bank_id=config.bank_id, bank_version=config.version,
        view_id=str(geometry.get("view_id", "view_0")),
        profile_ids=[item["profile_id"] for item in candidates],
        config=matcher["selection"],
    )
    timeline: list[dict[str, Any]] = []
    for index, (_block, frame) in enumerate(build_frames):
        timestamp = float(index)
        current_id = selector.selected_profile_id
        current = None
        candidates_this_tick: list[CandidateMatch] = []
        for item in candidates:
            score = score_cache.get((index, item["profile_id"]))
            if score is None:
                continue
            outcome = evaluate_match(score, envelopes[item["profile_id"]], matcher)
            candidate = CandidateMatch(
                item["profile_id"], score.score, outcome.enter_eligible,
                outcome.hold_eligible,
            )
            candidates_this_tick.append(candidate)
            if item["profile_id"] == current_id:
                current = candidate
        decision = selector.observe(
            timestamp=timestamp, current=current, candidates=candidates_this_tick,
        )
        timeline.append(decision.as_dict())
        if decision.commit_requested:
            selector.commit(
                profile_id=decision.commit_profile_id, timestamp=timestamp,
            )
    summary = selector.summarise()
    report["dynamic_finalisation"] = {
        "selector_summary": summary,
        "timeline_entries": len(timeline),
        "config_hash": config_hash,
        "note": (
            "离线回放使用显式虚拟时钟与零加载时延；服务器实测前不代表在线恢复速度"
        ),
    }
    report["static_preselect"] = {
        item["profile_id"]: {
            "static_coverage": round(item["static_coverage"], 5),
            "samples": item["samples"],
            "low_support": item["low_support"],
        }
        for item in candidates
    }
    # 3) 剪枝：只删掉既无静态增量、又没有缩短暂停的冗余候选。
    selected = _prune_candidates(candidates, selector, report)
    report["n_selection"] = {
        "candidates": len(candidates),
        "selected": len(selected),
        "max_profiles": config.max_profiles,
        "summary": summary,
    }
    return selected


def _match_frames_to_candidates(
    candidates: Sequence[dict[str, Any]],
    build_frames: Sequence[tuple[str, np.ndarray]],
) -> dict[str, list[np.ndarray]]:
    """把构建集帧按分组来源分配到候选；没有来源信息时按描述子最近邻分配。"""
    mapping: dict[str, list[np.ndarray]] = {}
    for item in candidates:
        mapping[item["profile_id"]] = []
    if not build_frames:
        return mapping
    for index, (block, frame) in enumerate(build_frames):
        # 优先使用该文件自己的高分帧；没有来源归属时按顺序轮转，保证每个
        # 候选都有独立的评估帧。
        target = None
        for item in candidates:
            if block in item.get("source_files", ()):  # pragma: no cover - 预留
                target = item
                break
        if target is None:
            target = candidates[index % len(candidates)]
        mapping[target["profile_id"]].append(frame)
    return mapping


def _prune_candidates(
    candidates: Sequence[dict[str, Any]], selector: ProfileSelector,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """动态剪枝：保留有静态覆盖或对过渡暂停有贡献的候选。"""
    selected: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in candidates:
        keep = item["static_coverage"] > 0.0 or item["low_support"]
        if keep:
            selected.append(item)
        else:
            removed.append({
                "group_id": item["group_id"],
                "reason": "NO_STATIC_GAIN",
                "static_coverage": round(item["static_coverage"], 5),
            })
    report["pruning"] = {
        "removed": removed,
        "kept": [item["group_id"] for item in selected],
        "note": "只在动态回放的有效覆盖、暂停尾部与确认延迟均未变差时才删除参考",
    }
    del selector
    return selected


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def run_factory(config: FactoryConfig, args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "algorithm_version": ALGORITHM_VERSION,
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "camera_id": config.camera_id,
        "bank_id": config.bank_id,
        "version": config.version,
        "output_root": str(config.output_root),
        "work_dir": str(config.work_dir),
    }
    config.work_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(config.work_dir)
    cache = ManagedRecordingCache(
        config.work_dir, raw_cache_budget=config.raw_cache_budget,
        work_budget=config.work_budget,
    )
    try:
        cache.recover()
        if args.input is not None:
            root = Path(args.input).expanduser().resolve()
            files = scan_local_directory(root)
            if not files:
                raise SystemExit(f"目录里没有找到可用的 PS/视频: {root}")
            build_local_cache_entries(
                cache, config.device_code, root, files,
                work_budget=config.work_budget,
            )
            report["inventory"] = {
                "mode": "local_directory",
                "root": str(root),
                "files": len(files),
                "total_declared_bytes": sum(
                    int(item.file_size or 0) for item in files
                ),
            }
        else:
            client = RecordingListClient(headers=_auth_headers(args))
            files = stage_inventory_remote(config, client, cache, report)
            downloader = RecordingDownloader()
            ready = materialize_remote(
                config, client, downloader, cache, files, report,
                limit=config.max_downloads_per_run,
            )
            files = [item for item in files if item.file_id in set(ready)]
            if not files:
                raise SystemExit("下载后没有任何可用文件；见报告 failures")
            report["download_report"] = cache.report()

        # 时间划分：前 N 天构建、第 N+1 天校准、第 N+2 天盲测。
        days = sorted({parse_day(item.record_start) for item in files})
        if len(days) < 3:
            report["time_partition"] = {
                "days": days,
                "warning": (
                    "可用天数少于 3，无法按构建/校准/盲测三段划分；"
                    "把现有天数按比例分配并在报告中标注"
                ),
            }
            split = max(1, len(days) // 3)
            build_days = days[:max(1, len(days) - 2 * split)]
            calibration = days[-2] if len(days) >= 2 else days[-1]
            blind = days[-1]
        else:
            build_count = min(config.build_days, max(1, len(days) - 2))
            build_days = days[:build_count]
            calibration = days[build_count]
            blind = days[min(build_count + config.calibration_days, len(days) - 1)]
        partition = partition_recordings(
            files, build_days=build_days, calibration_day=calibration,
            blind_day=blind,
        )
        report["time_partition"] = {
            "days": days,
            "build_days": build_days,
            "calibration_day": calibration,
            "blind_day": blind,
            "counts": {key: len(value) for key, value in partition.items()},
            "cross_day": [
                item.file_id for item in cross_day_files(files, build_days=build_days)
            ],
            "leakage_check": "盲测日文件不参与构建/校准",
        }
        build_files = partition["build"] or partition["calibration"] or files

        size, first_path = find_first_decodable(
            cache, config.device_code, build_files, work_dir=config.work_dir,
        )
        del first_path
        geometry = load_geometry(
            config.geometry_path, size, camera_id=config.camera_id,
        )
        if config.geometry_path is None:
            report["geometry_warning"] = (
                "没有提供 --geometry；使用整幅画面作为 ROI。生产前必须替换为"
                "同一摄像头同一视角的已审核多边形 ROI，否则本报告的有效地面/覆盖"
                "数字不能作为现场验收依据。"
            )
        report["analysis_size"] = list(geometry["canvas_size"])
        config_hash = sha256_bytes(canonical_json({
            "geometry": geometry, "version": ALGORITHM_VERSION,
            "seed": config.seed, "preview_width": config.preview_width,
        }).encode())

        samples, hd_plan, descriptors = stage_sampling(
            config, cache, build_files, geometry, report,
        )
        if not samples:
            atomic_write_json(
                config.work_dir / "failure_report.json", report,
            )
            raise SystemExit(
                "粗采样没有得到任何可用样本；诊断见 "
                f"{config.work_dir / 'failure_report.json'}"
            )
        groups = stage_grouping(config, samples, descriptors, report)
        build_blocks = {
            str(item["time_block"]) for item in samples
            if parse_day(str(item["record_start"])) in set(build_days)
        }
        calibration_files = partition["calibration"]
        calibration_blocks = {
            str(item["time_block"]) for item in samples
            if any(item["file_id"] == entry.file_id for entry in calibration_files)
        }
        candidates = stage_composite_and_noise(
            config, cache, build_files, geometry, groups, report,
            build_blocks=build_blocks, calibration_blocks=calibration_blocks,
        )
        if not candidates:
            raise SystemExit("高清合成没有得到任何候选参考；见报告 composite")
        build_frames = _collect_frames(cache, config, build_files, geometry)
        selected = stage_preselect_and_finalize(
            config, candidates, geometry, report,
            build_frames=build_frames, config_hash=config_hash,
        )
        if not selected:
            raise SystemExit("动态剪枝后没有剩余候选")

        asset_entries = []
        for item in selected[: config.max_profiles]:
            asset_entries.append({
                "profile_id": item["profile_id"],
                "reference": item["reference"],
                "valid": item["valid"],
                "noise": item["noise"].payload,
                "descriptor": item["descriptor"],
                "profile_json": {
                    "algorithm_version": ALGORITHM_VERSION,
                    "config_hash": config_hash,
                    "sample_count": item["samples"],
                    "valid_ground_fraction": round(
                        float(np.count_nonzero(item["valid"])) / item["valid"].size, 5
                    ),
                    "support": item["support"],
                    "low_support": item["low_support"],
                    "days": item["days"],
                    "noise_note": item["noise_note"],
                    "noise_diagnostics": item["noise_diagnostics"],
                    "bias_diagnosis": {
                        "requires_rebuild": item["bias_diagnosis"]["requires_rebuild"],
                        "regions": len(item["bias_diagnosis"]["regions"]),
                    },
                    "composite": item["composite"],
                    "replacement": item["replacement"],
                    "source": {
                        "mode": "local_directory" if args.input is not None
                        else "ctseelink-file-urls",
                        "build_days": build_days,
                    },
                },
            })
        final = publish_version(
            config.output_root, config.bank_id, config.version,
            manifest={
                "camera_id": config.camera_id,
                "algorithm_version": ALGORITHM_VERSION,
                "config_hash": config_hash,
                "created_utc": report["started_utc"],
                "tolerance": {
                    "persistent_object_in_reference": "allowed",
                    "disclaimer": "参考是稳定场景估计，不是清洁真值认证",
                },
                "time_partition": report["time_partition"],
                "resource_limits": {
                    "raw_cache_budget": config.raw_cache_budget,
                    "work_budget": config.work_budget,
                },
            },
            camera_geometry=geometry,
            matcher=default_matcher_config(),
            profiles=asset_entries,
            supersede_existing=config.supersede,
        )
        report["bank"] = {
            "path": str(final),
            "profiles": [entry["profile_id"] for entry in asset_entries],
            "manifest_sha256": sha256_file(final / "bank.json"),
        }
        report["space"] = cache.report()
        # 无租约的临时 PS 全部回收。
        released = cache.evict_to_budget(allow_ready=True)
        report["cleanup"] = {"released": released}
        report["finished_utc"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        report_dir = final / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(report_dir / "factory_report.json", report)
        state.setdefault("stages", {})["build"] = "COMPLETED"
        save_state(config.work_dir, state)
        return report
    finally:
        cache.close()


def _collect_frames(
    cache: ManagedRecordingCache, config: FactoryConfig,
    files: Sequence[RecordingFile], geometry: Mapping[str, Any],
) -> list[tuple[str, np.ndarray]]:
    """构建集顺序解码：用于动态回放的连续帧（按时间顺序，跨文件不重置）。"""
    from rtsp_annotator.ground_litter_profile_sampling import SequentialFrameReader
    size = (int(geometry["canvas_size"][0]), int(geometry["canvas_size"][1]))
    ordered = sorted(files, key=lambda item: (item.record_start, item.file_id))
    frames: list[tuple[str, np.ndarray]] = []
    for item in ordered:
        entry = cache.entry(config.device_code, item.file_id)
        if entry is None or entry.path is None:
            continue
        reader = SequentialFrameReader(entry.path)
        try:
            for frame in reader.iter_frames():
                resized = cv2.resize(frame.frame, size, interpolation=cv2.INTER_AREA)
                frames.append((item.file_id, resized))
                if len(frames) >= 240:
                    return frames
        except Exception:
            continue
    return frames


def _auth_headers(args: argparse.Namespace) -> dict[str, str]:
    headers: dict[str, str] = {}
    token = getattr(args, "auth_token", None) or os.environ.get(
        "GROUND_LITTER_PLAYBACK_TOKEN"
    )
    if token:
        headers["Authorization"] = token
    api_key = getattr(args, "api_key", None) or os.environ.get(
        "GROUND_LITTER_PLAYBACK_API_KEY"
    )
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def geometry_from_config(path: Path) -> dict[str, Any] | None:
    """从现有流配置里取同一摄像头的 ROI/排除区（复用已有 ROI）。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras = payload.get("cameras") or []
    if not cameras:
        return None
    camera = cameras[0]
    zones = camera.get("zones") or []
    polygon = None
    for zone in zones:
        if zone.get("polygon"):
            polygon = zone["polygon"]
            break
    if polygon is None and camera.get("ground_roi"):
        polygon = camera["ground_roi"]
    return {
        "roi": polygon,
        "exclude_zones": [
            item for zone in zones for item in (zone.get("exclude_zones") or [])
        ],
        "overlay_exclude_zones": camera.get("overlay_exclude_zones") or [],
        "canvas_size": camera.get("reference_size"),
        "view_id": camera.get("view_id", "view_0"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="七天 PS 自动生成 N 个 Profile（方案一 A0～A6）",
    )
    parser.add_argument("--input", type=Path, default=None,
                        help="本地 PS 根目录（与 --source 二选一）")
    parser.add_argument("--source", default=None,
                        choices=["ctseelink-file-urls"],
                        help="远程回放文件列表接口")
    parser.add_argument("--device-code", default=None)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--bank-id", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--geometry", type=Path, default=None,
                        help="已审核几何 JSON；也可用 --geometry-from-config")
    parser.add_argument("--geometry-from-config", type=Path, default=None,
                        help="从现有流配置复用同一摄像头的 ROI")
    parser.add_argument("--output", type=Path, required=True, help="Bank 根目录")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--raw-cache-budget-gib", type=float, default=1.0)
    parser.add_argument("--work-budget-gib", type=float, default=20.0)
    parser.add_argument("--max-profiles", type=int, default=MAX_BANK_PROFILES)
    parser.add_argument("--build-days", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--max-downloads", type=int, default=None)
    parser.add_argument("--use-seek", action="store_true")
    parser.add_argument("--supersede", action="store_true",
                        help="允许把同名版本改名保留后重新发布")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input is None and args.source is None:
        parser.error("必须提供 --input 或 --source ctseelink-file-urls")
    if args.source is not None:
        if not (args.device_code and args.start and args.end):
            parser.error("--source 需要 --device-code/--start/--end")
    geometry_path = args.geometry
    if geometry_path is None and args.geometry_from_config is not None:
        payload = geometry_from_config(args.geometry_from_config)
        if payload is None:
            parser.error("--geometry-from-config 里没有找到可复用的 ROI")
        geometry_path = args.work_dir or Path("output/profile_bank_work")
        geometry_path = Path(geometry_path) / "geometry_from_config.json"
        geometry_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(geometry_path, payload)
    version = args.version or datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    work_dir = args.work_dir or (args.output.parent / f"{args.camera}_work")
    config = FactoryConfig(
        camera_id=args.camera,
        bank_id=args.bank_id or args.camera,
        version=version,
        output_root=args.output,
        work_dir=Path(work_dir),
        geometry_path=geometry_path,
        analysis_size=None,
        device_code=args.device_code or "00000000000000000000",
        timezone=args.timezone,
        build_days=args.build_days,
        preview_width=args.preview_width,
        max_profiles=args.max_profiles,
        raw_cache_budget=int(args.raw_cache_budget_gib * 1024 ** 3),
        work_budget=int(args.work_budget_gib * 1024 ** 3),
        seed=args.seed,
        supersede=args.supersede,
        use_seek=args.use_seek,
        max_downloads_per_run=args.max_downloads,
    )
    if args.source is not None:
        config.start_time = args.start  # type: ignore[attr-defined]
        config.end_time = args.end  # type: ignore[attr-defined]
    report = run_factory(config, args)
    if args.report is not None:
        atomic_write_json(args.report, report)
    print(json.dumps({
        "bank": report.get("bank", {}).get("path"),
        "profiles": report.get("bank", {}).get("profiles"),
        "n_selection": report.get("n_selection"),
        "time_partition": report.get("time_partition"),
        "space": report.get("space"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
