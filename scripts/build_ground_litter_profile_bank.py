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
    BankPriorContext, compute_mask_set, roi_mask_from_geometry,
)
from rtsp_annotator.ground_litter_profile_background import (  # noqa: E402
    MAX_BANK_PROFILES, diagnose_persistent_bias, estimate_noise,
    observation_mask_from_frames,
    group_appearance_samples, replace_with_real_observations,
    select_time_balanced_frames, temporal_median_composite,
)
from rtsp_annotator.ground_litter_profile_bank import (  # noqa: E402
    BankError, atomic_write_json, canonical_json, default_camera_geometry,
    default_matcher_config, load_bank, publish_version, sha256_bytes,
    sha256_file,
)
from rtsp_annotator.ground_litter_profile_match import (  # noqa: E402
    envelope_from_samples, evaluate_match, extract_grid_descriptor,
    rank_by_coarse_distance, score_profile,
)

from rtsp_annotator.ground_litter_profile_sampling import (  # noqa: E402
    BoundedPreviewSampler, CanvasRegistrar, SequentialFrameReader,
    build_hd_plan, coarse_sample_offsets, cross_day_files, frame_quality,
    partition_recordings, preview_image, probe_recording, parse_day,
    summarise_sampling_quality, write_canvas_reference,
)
from rtsp_annotator.ground_litter_profile_selector import (  # noqa: E402
    CandidateMatch, ProfileSelector,
)
from rtsp_annotator.ground_litter_recording_cache import (  # noqa: E402
    ManagedRecordingCache,
)
from rtsp_annotator.ground_litter_recording_source import (  # noqa: E402
    ListQuery, RecordingDownloader, RecordingFile, RecordingListClient,
    RecordingSourceError, UrlRefreshPolicy, deduplicate_files,
    detect_truncation, file_looks_like_media,
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
    per_day_hours: int = 0
    max_files: int = 400
    start_time: str = ""
    end_time: str = ""


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


def inventory_cache_path(work_dir: Path, start: str, end: str) -> Path:
    key = sha256_bytes(f"{start}|{end}".encode())[:16]
    return work_dir / f"inventory_{key}.json"


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


def _hour_slot(record_start: str) -> str:
    text = str(record_start or "").replace("T", " ")
    return text[:13] if len(text) >= 13 else text


def robust_query(
    client: RecordingListClient, query: ListQuery, *,
    attempts: int = 3, base_delay: float = 1.0,
) -> Any:
    """清单查询的有限退避重试。

    网络抖动、限流或单次超时不应让整段七天作业失败（方案一 §4.5）。超过尝试
    次数后抛错，由调用方决定是跳过该窗口还是终止。
    """
    last: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return client.query(query)
        except RecordingSourceError as exc:
            last = exc
            if attempt >= attempts:
                break
            time.sleep(base_delay * (2 ** (attempt - 1)))
    assert last is not None
    raise last


def select_files_by_day_hours(
    files: Sequence[RecordingFile], *, per_day_hours: int,
    min_gap_minutes: int = 5, max_files: int = 400,
) -> list[RecordingFile]:
    """按「天 × 小时」均匀抽样，保证覆盖一天内的光照变化（含夜间）。

    同一小时内只取一个文件，避免大段重复录像占满预算。选择是确定性的：
    按 (record_start, file_id) 排序后每小时内取最早的一个。
    """
    grouped: dict[str, dict[str, RecordingFile]] = {}
    for item in sorted(files, key=lambda entry: (entry.record_start, entry.file_id)):
        day = parse_day(item.record_start)
        hour = _hour_slot(item.record_start)
        grouped.setdefault(day, {}).setdefault(hour, item)
    chosen: list[RecordingFile] = []
    for day in sorted(grouped):
        hours = sorted(grouped[day])
        if per_day_hours and per_day_hours < len(hours):
            # 均匀取 per_day_hours 个小时槽，覆盖凌晨/白天/夜晚。
            indexes = [
                round(index * (len(hours) - 1) / (per_day_hours - 1))
                for index in range(per_day_hours)
            ] if per_day_hours > 1 else [0]
            hours = [hours[index] for index in sorted(set(indexes))]
        for hour in hours:
            chosen.append(grouped[day][hour])
    del min_gap_minutes
    return chosen[:max_files]


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
        try:
            page = robust_query(client, query)
        except RecordingSourceError as exc:
            report.setdefault("inventory_failures", []).append({
                "start": query.start_time, "error": str(exc)[:120],
            })
            cursor = window_end
            continue
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
            try:
                halves.append(robust_query(client, ListQuery(
                    config.device_code,
                    half_start.strftime("%Y-%m-%d %H:%M:%S"),
                    half_end.strftime("%Y-%m-%d %H:%M:%S"),
                )))
            except RecordingSourceError:
                continue
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


def stream_materialize_and_sample(
    config: FactoryConfig, client: RecordingListClient,
    downloader: RecordingDownloader, cache: ManagedRecordingCache,
    files: Sequence[RecordingFile], geometry: dict[str, Any],
    report: dict[str, Any], *,
    prefetch_slots: int = 2, wait_timeout: float = 900.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """有界「拉取→处理→提交→释放」流水线（R5）。

    与旧实现的三点差别：

    1. 不再先整批下载：预取窗口固定为 ``prefetch_slots`` 个文件，处理完一个
       才补下一个；
    2. 预算不足时 ``wait_for_capacity`` **等待消费**，不 ``break`` 截断计划，
       因此分区不会被静默缩小；
    3. 释放发生在**退出 lease 之后**（旧代码在 lease 上下文中调用 release，
       必然被 ``LEASED`` 拒绝）。
    """
    size = (int(geometry["canvas_size"][0]), int(geometry["canvas_size"][1]))
    roi = roi_mask_from_geometry(geometry, size[0], size[1])
    overlay = geometry.get("overlay_exclude_zones") or []
    canvas_path = config.work_dir / "canvas_reference.png"

    samples: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    descriptors: dict[str, dict[str, np.ndarray]] = {}
    hd_plan: dict[str, Any] = {"files": {}}
    final_states: list[dict[str, Any]] = []
    policy = UrlRefreshPolicy()
    pipeline: dict[str, Any] = {
        "planned": len(files), "prefetch_slots": prefetch_slots,
        "waited_seconds": 0.0, "wait_events": [], "truncated_plan": False,
        "downloaded_files": 0, "released_files": 0,
    }

    # 第一帧用于冻结画布：优先用已就绪的文件；冷缓存时先取一个源。
    canvas_reference: np.ndarray | None = None
    canvas_source: Path | None = None
    for item in files:
        entry = cache.entry(config.device_code, item.file_id)
        if entry is not None and entry.path is not None and entry.path.is_file():
            canvas_source = entry.path
            break
        fallback = Path(item.file_name)
        if fallback.is_file():
            canvas_source = fallback
            break
    if canvas_source is not None:
        try:
            for frame in SequentialFrameReader(canvas_source).iter_frames():
                canvas_reference = cv2.resize(
                    frame.frame, size, interpolation=cv2.INTER_AREA,
                )
                break
        except Exception:
            canvas_reference = None
    if canvas_reference is None:
        raise SystemExit("无法从构建集建立共同画布")
    canvas_sha = write_canvas_reference(canvas_path, canvas_reference)
    report["frozen_canvas"] = {
        "path": str(canvas_path), "sha256": canvas_sha,
        "size": [int(canvas_reference.shape[1]), int(canvas_reference.shape[0])],
        "derived_from": "first decodable training frame",
        "policy": "one frozen canvas for every file; no per-file re-basing",
    }
    print(f"[canvas] frozen {canvas_sha[:16]}", flush=True)

    def ensure_ready(item: RecordingFile) -> tuple[bool, str]:
        entry = cache.entry(config.device_code, item.file_id)
        if entry is not None and entry.path is not None and entry.path.is_file():
            return True, ""
        ok, reason = cache.wait_for_capacity(
            item.file_size, timeout=wait_timeout,
        )
        if not ok:
            pipeline["truncated_plan"] = True
            return False, reason
        window = ListQuery(config.device_code, item.record_start, item.record_end)
        try:
            fresh = downloader.fetch_url_for_file(
                client, window, item.file_id, policy=policy,
            )
            target = cache.begin_download(config.device_code, item)
            result = downloader.download(
                fresh.url, target, expected_size=item.file_size,
                allow_resume=False,
            )
        except Exception as exc:
            cache.fail_download(config.device_code, item.file_id, str(exc)[:120])
            return False, f"download_error:{type(exc).__name__}"
        if not file_looks_like_media(target):
            cache.fail_download(config.device_code, item.file_id,
                                "content_probe_failed")
            return False, "content_probe_failed"
        probe = probe_recording(target)
        if not probe.ok:
            cache.fail_download(config.device_code, item.file_id,
                                f"decode_failed:{probe.error}")
            return False, "decode_failed"
        cache.complete_download(
            config.device_code, item.file_id, path=target, size=result.size,
            sha256=result.sha256, range_supported=result.range_supported,
            elapsed_seconds=result.elapsed_seconds,
        )
        pipeline["downloaded_files"] += 1
        return True, ""

    inflight: list[RecordingFile] = []
    pending = list(files)

    def fill_prefetch() -> None:
        while pending and len(inflight) < prefetch_slots:
            inflight.append(pending.pop(0))

    for index, item in enumerate(files, start=1):
        fill_prefetch()
        if index % 10 == 1 or index == len(files):
            print(f"[sampling] {index}/{len(files)} {item.record_start} "
                  f"samples={len(samples)}", flush=True)
        wait_started = time.monotonic()
        ok, reason = ensure_ready(item)
        waited = time.monotonic() - wait_started
        pipeline["waited_seconds"] += waited
        if not ok:
            pipeline["wait_events"].append({
                "file_id": item.file_id, "reason": reason,
                "waited_seconds": round(waited, 3),
            })
            final_states.append({
                "file_id": item.file_id, "state": f"failed:{reason}",
                "record_start": item.record_start,
            })
            if item in inflight:
                inflight.remove(item)
            continue
        if item in inflight:
            inflight.remove(item)
        entry = cache.entry(config.device_code, item.file_id)
        if entry is None or entry.path is None:
            final_states.append({"file_id": item.file_id, "state": "not_ready"})
            continue
        try:
            with cache.acquire(config.device_code, item.file_id,
                               owner="sampler") as lease:
                sampler = BoundedPreviewSampler(
                    cache, analysis_size=size, roi_mask=roi,
                    seed=config.seed, preview_width=config.preview_width,
                    use_seek=config.use_seek,
                    algorithm_version=ALGORITHM_VERSION,
                    canvas_reference=canvas_reference,
                    overlay_exclude_zones=overlay,
                )
                result = sampler.sample_file(config.device_code, lease)
                file_id = item.file_id
        except Exception as exc:
            details.append({"file_id": item.file_id, "error": str(exc)[:160]})
            final_states.append({
                "file_id": item.file_id, "state": "sample_error",
                "error": type(exc).__name__,
            })
            continue
        details.append({"file_id": item.file_id, **result["detail"]})
        for sample in result["samples"]:
            payload = sample.as_dict()
            payload["preview"] = sample.preview
            payload["descriptor"] = sample.descriptor
            payload["hd_frame"] = (
                cv2.resize(sample.aligned_full, size, interpolation=cv2.INTER_AREA)
                if sample.aligned_full is not None else None
            )
            samples.append(payload)
            descriptors[sample.time_block] = sample.descriptor
        plan = build_hd_plan(result["samples"])
        for key, value in plan["files"].items():
            merged = hd_plan["files"].setdefault(key, value)
            merged["offsets"] = sorted(set(merged["offsets"]) | set(value["offsets"]))
        # 退出 lease 之后再释放：旧代码在 lease 内调用会被 LEASED 拒绝。
        released = cache.release_file(
            config.device_code, file_id, require_committed=("preview",),
        )
        if released:
            pipeline["released_files"] += 1
        final_states.append({
            "file_id": file_id, "state": "consumed",
            "record_start": item.record_start, "released": bool(released),
            "samples": len(result["samples"]),
        })

    pipeline["final_states"] = final_states
    pipeline["consumed"] = sum(
        1 for row in final_states if row["state"] == "consumed"
    )
    pipeline["failed"] = sum(
        1 for row in final_states if row["state"].startswith("failed")
    )
    pipeline["waited_seconds"] = round(pipeline["waited_seconds"], 3)
    pipeline["refresh_count"] = policy.refresh_count
    report["pipeline"] = pipeline
    report["sampling"] = {
        "files_sampled": len(details),
        "quality": summarise_sampling_quality(samples),
        "decode_seconds": round(sum(
            item.get("decode_seconds", 0.0) for item in details
        ), 3),
        "seek_seconds": round(sum(item.get("seek_seconds", 0.0) for item in details), 3),
        "rejected": sum(len(item.get("rejected", [])) for item in details),
        "mode": "seek" if config.use_seek else "sequential",
        "bytes_downloaded": sum(
            int(item.get("downloaded_bytes") or 0) for item in details
        ),
        "per_file": details,
    }
    registration = [
        row for detail in details for row in detail.get("registration", []) or []
    ]
    report["registration"] = {
        "applied_to_canvas": sum(
            1 for row in registration if row.get("applied_to_canvas")
        ),
        "rejected": sum(
            1 for row in registration if not row.get("applied_to_canvas")
        ),
        "canvas_sha256": canvas_sha,
    }
    report["hd_plan"] = {
        "files": len(hd_plan["files"]),
        "offsets": sum(len(value["offsets"]) for value in hd_plan["files"].values()),
    }
    atomic_write_json(config.work_dir / "hd_plan.json", hd_plan)
    return samples, details, hd_plan, descriptors


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
    groups: Sequence[dict[str, Any]], samples: Sequence[dict[str, Any]],
    report: dict[str, Any],
    *, build_blocks: set[str], calibration_blocks: set[str],
) -> list[dict[str, Any]]:
    """用高清帧合成参考、估计噪声；构建块与校准块严格分离。"""
    size = (int(geometry["canvas_size"][0]), int(geometry["canvas_size"][1]))
    roi = roi_mask_from_geometry(geometry, size[0], size[1])
    from rtsp_annotator.ground_litter_profile_sampling import SequentialFrameReader

    by_id = {item.file_id: item for item in files}
    candidates: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        _t0 = time.monotonic()
        cached_hits = sum(
            1 for sample in samples
            if sample.get("hd_frame") is not None
            and str(sample["time_block"]) in set(group["time_blocks"])
        )
        print(f"[composite] group {group_index}/{len(groups)} {group['group_id']} "
              f"blocks={len(group['time_blocks'])} cached_hd={cached_hits}", flush=True)
        needed = [block for block in group["time_blocks"]]
        if not needed:
            continue
        frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        blocks: list[str] = []
        cached_frames = {
            str(item["time_block"]): item.get("hd_frame")
            for item in samples if item.get("hd_frame") is not None
        }
        for block in needed:
            cached = cached_frames.get(block)
            if cached is not None:
                frames.append(cached)
                masks.append(roi.copy())
                blocks.append(block)
                continue
            # 缓存未命中（例如本条目的采样被质量过滤）：按需重解码一次。
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
            reader = SequentialFrameReader(entry.path)
            picked = reader.sample_at([offset], tolerance_seconds=1.5)
            captured = picked.get(offset)
            if captured is None:
                continue
            if not frame_quality(captured.frame).usable:
                continue
            frames.append(cv2.resize(
                captured.frame, size, interpolation=cv2.INTER_AREA,
            ))
            masks.append(roi.copy())
            blocks.append(block)
        if len(frames) < 2:
            report.setdefault("composite_skipped", []).append(group["group_id"])
            print(f"[composite]   skipped (frames={len(frames)})", flush=True)
            continue
        # R7：用真实观测导出遮挡/运动掩膜；每帧一个 mask，而不是整块 ROI。
        availability_roi, observation_diag, observation_masks = (
            observation_mask_from_frames(frames, roi, block_ids=blocks)
        )
        # 逐帧掩膜用于合成：既排除本帧的运动物体，也排除长期遮挡地带。
        composite_masks = [
            np.minimum(mask, availability_roi)
            for mask in observation_masks
        ]
        composite = temporal_median_composite(
            frames, composite_masks, availability_roi, blocks,
            stride=2, min_observations=2,
        )
        refined, replace_diagnostics = replace_with_real_observations(
            composite.reference, frames, composite_masks, availability_roi,
        )
        # R7：优先用**独立校准块**估噪声；退化时明确标注，而不是悄悄用构建块。
        eval_frames = [
            frame for frame, block in zip(frames, blocks)
            if block in calibration_blocks
        ]
        eval_masks = [
            mask for mask, block in zip(composite_masks, blocks)
            if block in calibration_blocks
        ]
        eval_blocks = [block for block in blocks if block in calibration_blocks]
        independent = len(eval_frames) >= 2
        if not independent:
            eval_frames = [
                frame for frame, block in zip(frames, blocks)
                if block not in build_blocks
            ]
            eval_masks = [
                mask for mask, block in zip(composite_masks, blocks)
                if block not in build_blocks
            ]
            eval_blocks = [block for block in blocks if block not in build_blocks]
            noise_note = "no_independent_calibration_blocks"
        else:
            noise_note = "independent_calibration_blocks"
        if len(eval_frames) < 2:
            eval_frames, eval_masks, eval_blocks = frames, composite_masks, blocks
            noise_note = "low_support_used_all_blocks"
        # 阈值图必须与参考同尺寸（Bank loader 契约）。逐像素分位代价随帧数线性
        # 增长，这里按时间均匀抽稀到 24 帧（块覆盖保持不变）。
        noise = estimate_noise(
            refined, eval_frames, eval_masks, eval_blocks, stride=1,
            config={
                "max_estimate_frames": 24,
                "_calibration_block_count": len(eval_blocks),
                "_independent_calibration": independent,
            },
        )
        diagnosis = diagnose_persistent_bias(noise)
        # R7：发布的 valid 反映局部可用性：ROI ∩ 观测可用 ∩ 非持续偏差。
        bias_flag = np.asarray(noise.payload["bias_flag"], np.uint8)
        low_support_flag = np.asarray(noise.payload["low_support"], np.uint8)
        published_valid = np.zeros_like(availability_roi)
        published_valid[availability_roi > 0] = 255
        published_valid[bias_flag > 0] = 0
        published_valid[low_support_flag > 0] = 0
        published_valid = cv2.morphologyEx(
            published_valid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
        )
        if int(np.count_nonzero(published_valid)) < 5000:
            published_valid = availability_roi.copy()
            noise_note += "|valid_fallback_to_observation_mask"
        print(f"[composite]   done frames={len(frames)} eval={len(eval_frames)} "
              f"bias_px={diagnosis and noise.diagnostics.get('bias_flag_pixels')} "
              f"seconds={round(time.monotonic() - _t0, 1)}", flush=True)
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
            "valid": published_valid,
            "noise": noise,
            "descriptor": descriptor,
            "composite": composite.diagnostics,
            "replacement": replace_diagnostics,
            "noise_diagnostics": noise.diagnostics,
            "bias_diagnosis": diagnosis,
            "noise_note": noise_note,
            "support": group["support"],
            "low_support": group["low_support"],
            "days": group["days"],
            "samples": len(frames),
            "observation": observation_diag,
            "valid_fraction_of_roi": round(
                float(np.count_nonzero(published_valid))
                / max(int(np.count_nonzero(roi)), 1), 5,
            ),
            "context": BankPriorContext(
                profile_id=profile_id,
                reference=refined,
                valid=published_valid,
                noise=dict(noise.payload),
                metadata={"group_id": group["group_id"], "low_support": group["low_support"]},
            ),
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
            "observation": item["observation"],
            "valid_fraction_of_roi": item["valid_fraction_of_roi"],
        }
        for item in candidates
    ]
    del calibration_blocks
    return candidates


def _canvas_scaled(candidates: Sequence[dict[str, Any]], size: tuple[int, int],
                   replay_w: int, replay_h: int) -> dict[str, dict[str, Any]]:
    """把每个候选的参考/有效图/阈值图缩放到回放画布（几何是归一化的）。"""
    del size
    contexts: dict[str, dict[str, Any]] = {}
    for item in candidates:
        context = item["context"]
        contexts[item["profile_id"]] = {
            "reference": cv2.resize(context.reference, (replay_w, replay_h),
                                    interpolation=cv2.INTER_AREA),
            "valid": cv2.resize(context.valid, (replay_w, replay_h),
                                interpolation=cv2.INTER_NEAREST),
        }
    return contexts


def fit_frozen_envelopes(
    config: "FactoryConfig", cache: ManagedRecordingCache,
    candidates: Sequence[dict[str, Any]], calibration_files: Sequence[RecordingFile],
    geometry: Mapping[str, Any], matcher: Mapping[str, Any], report: dict[str, Any],
    *, max_frames_per_file: int = 6, replay_scale: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """只在**校准集**上拟合进入/保持包络，并记录来源（R1）。

    校准集与构建集、盲测集按天隔离；这里只读校准文件，绝不使用待评数据。
    """
    size = (int(geometry["canvas_size"][0]), int(geometry["canvas_size"][1]))
    roi = roi_mask_from_geometry(geometry, size[0], size[1])
    overlay = geometry.get("overlay_exclude_zones") or []
    by_id = {item.file_id: item for item in calibration_files}
    contexts = {
        item["profile_id"]: item["context"] for item in candidates
    }
    scores: dict[str, list[float]] = {pid: [] for pid in contexts}
    frames_used: list[dict[str, Any]] = []
    registrar: CanvasRegistrar | None = None
    for item in calibration_files:
        entry = cache.entry(config.device_code, item.file_id)
        if entry is None or entry.path is None:
            continue
        probe = probe_recording(entry.path)
        if not probe.ok:
            continue
        reader = SequentialFrameReader(entry.path)
        offsets = coarse_sample_offsets(probe.duration_seconds,
                                        seed=config.seed)
        picked = (
            reader.sample_with_seek(offsets, tolerance_seconds=1.0)
            if config.use_seek else reader.sample_at(offsets)
        )
        taken = 0
        for offset in offsets:
            captured = picked.get(offset)
            if captured is None:
                continue
            frame = cv2.resize(captured.frame, size, interpolation=cv2.INTER_AREA)
            if not frame_quality(frame).usable:
                continue
            if registrar is None:
                registrar = CanvasRegistrar(frame, overlay_exclude_zones=overlay)
            try:
                aligned, diagnostics = registrar.register(frame)
            except BankError:
                continue
            del diagnostics
            aligned = cv2.resize(aligned, size, interpolation=cv2.INTER_AREA)
            for pid, context in contexts.items():
                context_frame = aligned
                try:
                    score = score_profile(
                        context_frame, context.reference, context.valid, roi,
                        profile_id=pid, config=matcher,
                    )
                except BankError:
                    continue
                scores[pid].append(score.score)
            taken += 1
            frames_used.append({
                "file_id": item.file_id,
                "record_start": item.record_start,
                "offset_seconds": round(float(offset), 3),
            })
            if taken >= max_frames_per_file:
                break
    envelopes: dict[str, dict[str, Any]] = {}
    for pid, values in scores.items():
        envelope = envelope_from_samples(values, matcher)
        envelopes[pid] = {
            "envelope": envelope.as_dict(),
            "calibration": {
                "source": "calibration_day",
                "samples": len(values),
            },
        }
    low = [pid for pid, values in scores.items() if len(values) < 6]
    report["calibration"] = {
        "files": len(calibration_files),
        "frames_used": frames_used,
        "profiles_with_samples": {pid: len(v) for pid, v in scores.items()},
        "profiles_without_enough_samples": low,
        "source": "calibration_day_only",
        "note": "包络只在校准集上拟合；盲测与评估只读该冻结结果，不得重新拟合",
    }
    del by_id, replay_scale
    return envelopes


def replay_all_candidates(
    config: "FactoryConfig", candidates: Sequence[dict[str, Any]],
    replay_frames: Sequence[dict[str, Any]], geometry: Mapping[str, Any],
    matcher: Mapping[str, Any], report: dict[str, Any], *,
    replay_w: int = 960, replay_h: int = 540,
    leave_one_out: bool = True, loo_stride: int = 4,
) -> dict[str, Any]:
    """共同回放：所有候选面对**同一** ``(source_time, frame)``（R2）。

    返回每个候选的逐 tick 记录与留一法动态对照；不在这里做静态前 K 截断。
    """
    if not candidates:
        raise SystemExit("没有任何可用候选，无法建库")
    if not replay_frames:
        raise SystemExit("没有可用的连续回放帧")
    # 回放画布不超过实际帧尺寸；全尺寸评分更容易打爆内存，等比缩小到上限。
    frame_h, frame_w = replay_frames[0]["frame"].shape[:2]
    scale = min(1.0, replay_w / max(frame_w, 1), replay_h / max(frame_h, 1))
    replay_w = max(64, int(round(frame_w * scale)))
    replay_h = max(64, int(round(frame_h * scale)))
    frozen = {item["profile_id"]: item["envelope"] for item in candidates}
    missing = [pid for pid, env in frozen.items() if not env]
    if missing:
        raise SystemExit(f"候选缺少冻结包络: {missing[:4]}")
    roi = roi_mask_from_geometry(geometry, replay_w, replay_h)
    contexts = _canvas_scaled(candidates, (replay_w, replay_h), replay_w, replay_h)
    replay_config = dict(matcher.get("selection", {}))
    nominal = float(
        replay_frames[1]["source_time"] - replay_frames[0]["source_time"]
    ) if len(replay_frames) > 1 else 2.0
    nominal = max(0.05, min(nominal, 3600.0))
    replay_config["tick_interval_seconds"] = nominal
    replay_config.setdefault("join_gap_seconds", max(300.0, 4.0 * nominal))

    def run(subset: Sequence[str], stride: int = 1) -> dict[str, Any]:
        selector = ProfileSelector(
            bank_id=config.bank_id, bank_version=config.version,
            view_id=str(geometry.get("view_id", "view_0")),
            profile_ids=subset, config=replay_config,
        )
        frames = list(replay_frames)[:: max(1, stride)]
        for row in frames:
            frame = row["frame"]
            current_id = selector.selected_profile_id
            current = None
            matches: list[CandidateMatch] = []
            for pid in subset:
                context = contexts[pid]
                try:
                    score = score_profile(
                        frame, context["reference"], context["valid"], roi,
                        profile_id=pid, config=matcher,
                    )
                except BankError:
                    continue
                outcome = evaluate_match(score, frozen[pid], matcher)
                candidate = CandidateMatch(
                    pid, score.score, outcome.enter_eligible,
                    outcome.hold_eligible,
                )
                matches.append(candidate)
                if pid == current_id:
                    current = candidate
            decision = selector.observe(
                timestamp=float(row["source_time"]), current=current,
                candidates=matches, tested_profile_ids=subset,
            )
            if decision.commit_requested:
                selector.commit(profile_id=decision.commit_profile_id,
                                timestamp=float(row["source_time"]))
        return selector.summarise(join_gap_seconds=replay_config["join_gap_seconds"])

    baseline = run([item["profile_id"] for item in candidates])
    report["replay"] = {
        "scored_ticks": len(replay_frames),
        "canvas": [replay_w, replay_h],
        "nominal_tick_seconds": round(nominal, 4),
        "frame_reuse": "all candidates share the same frame and source_time",
    }
    result = {
        "baseline": baseline,
        "envelopes": {pid: env for pid, env in frozen.items()},
    }
    if not leave_one_out or len(candidates) < 2:
        result["leave_one_out"] = {}
        return result
    loo: dict[str, Any] = {}
    for item in candidates:
        subset = [other["profile_id"] for other in candidates
                  if other["profile_id"] != item["profile_id"]]
        summary = run(subset, stride=loo_stride)
        loo[item["profile_id"]] = {
            "effective_fraction_without": summary["effective_fraction"],
            "effective_fraction_delta": round(
                baseline["effective_fraction"] - summary["effective_fraction"], 5
            ),
            "pause_max_without": summary["pause_max"],
            "pause_max_delta": round(
                summary["pause_max"] - baseline["pause_max"], 3
            ),
            "switch_count_without": summary["switch_count"],
        }
    result["leave_one_out"] = loo
    return result


def select_profiles_dynamically(
    config: "FactoryConfig", candidates: Sequence[dict[str, Any]],
    replay: Mapping[str, Any], report: dict[str, Any], *,
    min_coverage_delta: float = 0.005, max_pause_delta: float = 5.0,
) -> list[dict[str, Any]]:
    """按留一法动态对照决定保留哪些候选（R2）。

    删除规则（必须同时满足才删）：
    * 去掉它不会让有效覆盖下降超过 ``min_coverage_delta``；
    * 去掉它不会让最长暂停增加超过 ``max_pause_delta``。

    这仍然是“过渡参考可保留”的实现：只要它对暂停尾部或覆盖有贡献就保留。
    """
    loo = dict(replay.get("leave_one_out") or {})
    keep: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in candidates:
        stats = loo.get(item["profile_id"])
        if stats is None:
            keep.append(item)
            continue
        coverage_lost = float(stats.get("effective_fraction_delta", 0.0))
        pause_added = float(stats.get("pause_max_delta", 0.0))
        if coverage_lost > min_coverage_delta or pause_added > max_pause_delta:
            keep.append(item)
        else:
            removed.append({
                "profile_id": item["profile_id"],
                "group_id": item["group_id"],
                "reason": "DYNAMICALLY_REDUNDANT",
                "leave_one_out": stats,
            })
    if len(keep) > config.max_profiles:
        # 资源上限不是预设 N：只在超过上限时按动态贡献排序裁剪。
        ordered = sorted(
            keep,
            key=lambda entry: (
                -float((loo.get(entry["profile_id"]) or {}).get(
                    "effective_fraction_delta", 0.0)),
                entry["profile_id"],
            ),
        )
        for item in ordered[config.max_profiles:]:
            removed.append({
                "profile_id": item["profile_id"], "group_id": item["group_id"],
                "reason": "EXCEEDS_RESOURCE_LIMIT",
            })
        keep = ordered[: config.max_profiles]
    report["pruning"] = {
        "removed": removed,
        "kept": [item["profile_id"] for item in keep],
        "method": "leave_one_out_dynamic_replay",
        "note": ("删除必须同时满足：有效覆盖下降 ≤ 阈值，且最长暂停增加 ≤ 阈值；"
                 "静态覆盖不再是删除依据"),
    }
    return keep


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def _install_stack_dumper() -> None:
    """``kill -USR1 <pid>`` 把当前 Python 栈写到 stderr，用于诊断卡住位置。

    长时间离线作业必须能在不重启、不丢状态的前提下抓栈，否则只能靠猜。
    """
    import faulthandler
    import signal

    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except (AttributeError, ValueError):  # pragma: no cover - 平台不支持
        pass


def run_factory(config: FactoryConfig, args: argparse.Namespace) -> dict[str, Any]:
    _install_stack_dumper()
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
        local_paths: dict[str, Path] = {}
        if args.input is not None:
            root = Path(args.input).expanduser().resolve()
            files = scan_local_directory(root)
            if not files:
                raise SystemExit(f"目录里没有找到可用的 PS/视频: {root}")
            build_local_cache_entries(
                cache, config.device_code, root, files,
                work_budget=config.work_budget,
            )
            local_paths = {item.file_id: Path(item.file_name) for item in files}
            report["inventory"] = {
                "mode": "local_directory",
                "root": str(root),
                "files": len(files),
                "total_declared_bytes": sum(
                    int(item.file_size or 0) for item in files
                ),
            }
        else:
            cache_path = inventory_cache_path(
                config.work_dir, config.start_time, config.end_time,
            )
            cached_inventory = None
            if args.resume and cache_path.is_file():
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    cached_inventory = [
                        RecordingFile(
                            file_id=row["file_id"], file_name=row.get("file_name", ""),
                            record_start=row["record_start"],
                            record_end=row["record_end"],
                            file_size=row.get("file_size"),
                            file_type=row.get("file_type", ""),
                        )
                        for row in cached.get("files", [])
                    ]
                    report["inventory"] = {**cached.get("summary", {}),
                                           "source": "resume_cache"}
                except (OSError, ValueError, KeyError):
                    cached_inventory = None
            # client 无论走不走清单缓存都必须存在：下载与高清重拉都要用它。
            client = RecordingListClient(headers=_auth_headers(args))
            if cached_inventory:
                files = cached_inventory
                print(f"[inventory] reused cache with {len(files)} files", flush=True)
            else:
                files = stage_inventory_remote(config, client, cache, report)
                atomic_write_json(cache_path, {
                    "summary": {k: v for k, v in report["inventory"].items()},
                    "files": [item.as_dict() for item in files],
                })
            all_files = list(files)
            files = select_files_by_day_hours(
                files, per_day_hours=config.per_day_hours,
                max_files=config.max_files,
            )
            report["sampling_plan"] = {
                "inventory_files": len(all_files),
                "selected_files": len(files),
                "per_day_hours": config.per_day_hours,
                "per_day_selected": {
                    day: sum(1 for item in files if parse_day(item.record_start) == day)
                    for day in sorted({parse_day(item.record_start) for item in files})
                },
            }
            # R5：不在这里整批下载；分区冻结后由有界流水线按需拉取。
            downloader = RecordingDownloader()

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

        # R5：拉取→处理→提交→释放，全程有界；分区已在上一步冻结。
        if args.input is not None:
            samples, sampling_details, hd_plan, descriptors = (
                stream_materialize_and_sample(
                    config, None, RecordingDownloader(), cache, build_files,
                    geometry, report,
                    prefetch_slots=int(getattr(args, "prefetch_slots", 2) or 2),
                )
            )
        else:
            samples, sampling_details, hd_plan, descriptors = (
                stream_materialize_and_sample(
                    config, client, downloader, cache, build_files,
                    geometry, report,
                    prefetch_slots=int(getattr(args, "prefetch_slots", 2) or 2),
                )
            )
        del sampling_details
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
        # 高清阶段按稳定身份重拉：preview 释放过的临时 PS 必须重新取得，
        # 不能用小图放大冒充高清素材。
        repull = _ensure_remote_entries(
            config, args, cache, files, report,
            local_paths=local_paths if args.input is not None else {},
        )
        report["hd_repull"] = repull
        candidates = stage_composite_and_noise(
            config, cache, build_files, geometry, groups, samples, report,
            build_blocks=build_blocks, calibration_blocks=calibration_blocks,
        )
        if not candidates:
            raise SystemExit("高清合成没有得到任何候选参考；见报告 composite")
        print("[replay] collecting continuous frames", flush=True)
        _rt0 = time.monotonic()
        # 合成已完成，采样阶段的 HD 帧缓存不再需要；释放后再收集回放帧，
        # 避免两者叠加把内存推高（实测曾到 9.9GB RSS）。
        for sample in samples:
            sample.pop("hd_frame", None)
        import gc as _gc
        _gc.collect()
        replay_frames = _collect_replay_frames(
            cache, config, build_files, geometry,
            frame_budget=int(getattr(args, "max_replay_frames", 180) or 180),
        )
        print(f"[replay] collected {len(replay_frames)} frames "
              f"in {round(time.monotonic() - _rt0, 1)}s", flush=True)
        report["replay_frames"] = {
            "frames": len(replay_frames),
            "files": len(build_files),
            "source_span_seconds": round(
                float(replay_frames[-1]["source_time"])
                - float(replay_frames[0]["source_time"]), 3,
            ) if len(replay_frames) > 1 else 0.0,
            "note": "动态定稿只回放有限连续片段；在线恢复速度必须由服务器实测",
        }

        # R1：包络只在**校准集**（独立日期）上拟合，并随 Bank 冻结发布。
        print("[calibration] fitting frozen envelopes on the calibration day", flush=True)
        matcher = default_matcher_config()
        envelope_records = fit_frozen_envelopes(
            config, cache, candidates, partition["calibration"], geometry,
            matcher, report,
        )
        for item in candidates:
            item["envelope"] = (envelope_records.get(item["profile_id"]) or {}).get(
                "envelope"
            )
        missing_envelopes = [
            item["profile_id"] for item in candidates if not item["envelope"]
        ]
        if missing_envelopes:
            raise SystemExit(
                "校准集没有覆盖到这些候选的包络，拒绝发布未校准 Bank："
                f"{missing_envelopes[:6]}"
            )
        metrics = matcher.setdefault("metrics", {})
        metrics["score_scale"] = "per-profile cell-normalised S(p)"
        matcher["calibration"] = {
            "source": "calibration_day",
            "calibrated_utc": report["started_utc"],
            "method": "envelope_from_samples",
            "quantile": matcher["envelope"]["quantile"],
            "enter_margin": matcher["envelope"]["enter_margin"],
            "hold_margin": matcher["envelope"]["hold_margin"],
            "config_hash": config_hash,
            "canvas_sha256": report["frozen_canvas"]["sha256"],
            "inputs": {
                "days": [report["time_partition"]["calibration_day"]],
                "files": len(partition["calibration"]),
                "samples": report["calibration"]["frames_used"],
            },
        }
        matcher["profiles"] = envelope_records

        # R2：所有候选面对同一帧、同一源时间的共同回放；再做留一法动态对照。
        print("[select] joint replay over shared frames", flush=True)
        replay = replay_all_candidates(
            config, candidates, replay_frames, geometry, matcher, report,
            leave_one_out=True,
            loo_stride=int(getattr(args, "loo_stride", 4) or 4),
        )
        selected = select_profiles_dynamically(config, candidates, replay, report)
        if not selected:
            raise SystemExit("动态选择后没有剩余候选")
        report["n_selection"] = {
            "candidates": len(candidates),
            "selected": len(selected),
            "max_profiles": config.max_profiles,
            "baseline": replay["baseline"],
            "leave_one_out": replay["leave_one_out"],
            "summary": replay["baseline"],
        }
        # 记录每 tick 的逐候选分数与决策，便于复核“同一帧”与选 N 过程。
        atomic_write_json(
            config.work_dir / "dynamic_selection.json",
            {"baseline": replay["baseline"], "leave_one_out": replay["leave_one_out"],
             "pruning": report.get("pruning", {})},
        )

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
                        "calibration_day": report["time_partition"]["calibration_day"],
                        "blind_day": report["time_partition"]["blind_day"],
                    },
                    "envelope": item.get("envelope"),
                    "observation": item.get("observation"),
                    "valid_fraction_of_roi": item.get("valid_fraction_of_roi"),
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
                "frozen_canvas": report.get("frozen_canvas", {}),
                "calibration": matcher.get("calibration", {}),
                "resource_limits": {
                    "raw_cache_budget": config.raw_cache_budget,
                    "work_budget": config.work_budget,
                },
            },
            camera_geometry=geometry,
            matcher=matcher,
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


def _ensure_remote_entries(
    config: FactoryConfig, args: argparse.Namespace,
    cache: ManagedRecordingCache, files: Sequence[RecordingFile],
    report: dict[str, Any], *, local_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """确保高清阶段所需文件在盘上；本地源直接复用，远程源即时刷新重拉。"""
    summary: dict[str, Any] = {
        "checked": len(files), "reused_local": 0, "repulled": 0,
        "failed": [], "bytes": 0, "seconds": 0.0, "refreshes": 0,
    }
    client: RecordingListClient | None = None
    downloader: RecordingDownloader | None = None
    policy = UrlRefreshPolicy()
    for item in files:
        entry = cache.entry(config.device_code, item.file_id)
        if entry is not None and entry.path is not None and entry.path.is_file():
            continue
        local = local_paths.get(item.file_id)
        if local is not None and local.is_file():
            cache.register(config.device_code, [item])
            with cache._lock, cache._connection:  # noqa: SLF001 - 工厂内部记账
                cache._connection.execute(  # noqa: SLF001
                    "UPDATE recordings SET managed=0, materialization='READY',"
                    " path=?, bytes=?, updated=datetime('now') WHERE identity_key=?",
                    (str(local), local.stat().st_size,
                     RecordingFile.identity_key(config.device_code, item.file_id)),
                )
            summary["reused_local"] += 1
            continue
        if args.source is None:
            summary["failed"].append({
                "file_id": item.file_id, "reason": "NO_SOURCE_TO_REPULL",
            })
            continue
        if client is None:
            client = RecordingListClient(headers=_auth_headers(args))
            downloader = RecordingDownloader()
        allowed, reason = cache.can_reserve(item.file_size)
        if not allowed:
            summary["failed"].append({"file_id": item.file_id, "reason": reason})
            continue
        window = ListQuery(config.device_code, item.record_start, item.record_end)
        try:
            fresh = downloader.fetch_url_for_file(
                client, window, item.file_id, policy=policy,
            )
            target = cache.begin_download(config.device_code, item)
            result = downloader.download(
                fresh.url, target, expected_size=item.file_size, allow_resume=False,
            )
        except Exception as exc:
            cache.fail_download(config.device_code, item.file_id, str(exc)[:120])
            summary["failed"].append({
                "file_id": item.file_id, "reason": str(exc)[:120],
            })
            continue
        cache.complete_download(
            config.device_code, item.file_id, path=target, size=result.size,
            sha256=result.sha256, range_supported=result.range_supported,
            elapsed_seconds=result.elapsed_seconds,
        )
        summary["repulled"] += 1
        summary["bytes"] += result.size
        summary["seconds"] += result.elapsed_seconds
    summary["refreshes"] = policy.refresh_count
    summary["seconds"] = round(summary["seconds"], 3)
    return summary


def _collect_replay_frames(
    cache: ManagedRecordingCache, config: FactoryConfig,
    files: Sequence[RecordingFile], geometry: Mapping[str, Any],
    *, frame_budget: int = 180,
) -> list[dict[str, Any]]:
    """按时间顺序收集连续回放帧，并保留**真实源时间**与帧指纹（R2）。

    每帧记录 ``source_time``（录像起始时刻 + 文件内偏移）与帧内容 SHA-256，
    使“同一 tick 的所有候选用同一帧”与“时间线真实”都可被复核。
    """
    import gc as _gc

    size = (int(geometry["canvas_size"][0]), int(geometry["canvas_size"][1]))
    ordered = sorted(files, key=lambda item: (item.record_start, item.file_id))
    rows: list[dict[str, Any]] = []
    per_file = max(1, frame_budget // max(1, len(ordered)))
    for item in ordered:
        if len(rows) >= frame_budget:
            break
        entry = cache.entry(config.device_code, item.file_id)
        if entry is None or entry.path is None:
            continue
        try:
            base = parse_seconds(item.record_start)
        except Exception:
            base = 0.0
        reader = SequentialFrameReader(entry.path)
        taken = 0
        try:
            for frame in reader.iter_frames():
                resized = cv2.resize(frame.frame, size, interpolation=cv2.INTER_AREA)
                offset = float(frame.time_seconds)
                del frame
                rows.append({
                    "file_id": item.file_id,
                    "record_start": item.record_start,
                    "offset_seconds": round(offset, 3),
                    "source_time": base + offset,
                    "frame": resized,
                    "frame_sha256": sha256_bytes(
                        cv2.imencode(".jpg", resized,
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                    ),
                })
                taken += 1
                if taken >= per_file or len(rows) >= frame_budget:
                    break
        except Exception:
            continue
        finally:
            _gc.collect()
    return rows


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
    parser.add_argument("--prefetch-slots", type=int, default=2,
                        help="有界流水线的预取槽数（下载中+等待处理的文件数）")
    parser.add_argument("--per-day-hours", type=int, default=0,
                        help="每天最多取多少个小时槽的文件；0=全部")
    parser.add_argument("--max-files", type=int, default=400,
                        help="抽样后最多处理多少个文件")
    parser.add_argument("--max-replay-frames", type=int, default=180,
                        help="动态定稿连续回放的帧预算")
    parser.add_argument("--use-seek", action="store_true",
                        help="用 seek 逐点取帧（服务器实测 1s/点，推荐）")
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
        per_day_hours=getattr(args, "per_day_hours", 0),
        max_files=getattr(args, "max_files", 400),
    )
    if args.source is not None:
        config.start_time = args.start
        config.end_time = args.end
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
