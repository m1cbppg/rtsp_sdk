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
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
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
    select_calibration_observations, select_time_balanced_frames,
    temporal_median_composite,
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
    calibration_day: str = ""
    per_day_hours: int = 0
    max_files: int = 400
    start_time: str = ""
    end_time: str = ""


def frame_memory_bytes(frames: int, size: Sequence[int], *, channels: int = 3) -> int:
    """抽帧/回放帧的内存口径（C2）：帧数 × 高 × 宽 × 通道。"""
    height = int(size[1]) if len(size) > 1 else int(size[0])
    width = int(size[0])
    return int(max(0, frames) * max(0, height) * max(0, width) * max(1, channels))


def block_ids_by_day(
    samples: Sequence[Mapping[str, Any]], days: Iterable[str],
) -> set[str]:
    """从样本自身的 ``day`` 字段取时间块（C3）。

    v2 用「样本 file_id 是否属于校准文件列表」来判定 ``calibration_blocks``，
    而校准文件从未进入采样，集合恒为空。样本对象本来就带 ``day``，直接按天
    分流即可，不再依赖调用方手工拼装。
    """
    wanted = {str(day) for day in days}
    return {
        str(item["time_block"]) for item in samples
        if str(item.get("day")) in wanted and item.get("time_block")
    }


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

    # 第一帧用于冻结画布：优先用已就绪的缓存文件；冷缓存/远端时用一个已下载
    # 的候选文件（下载本身属于流水线，画布必须先于处理确定）。
    canvas_reference: np.ndarray | None = None
    def ensure_ready_for_canvas(item: RecordingFile) -> tuple[bool, str]:
        """仅为确定画布而拉取一个文件（随后会被正常消费，不重复下载）。"""
        if client is None:
            return False, "no_client_for_canvas"
        ok, reason = cache.wait_for_capacity(item.file_size, timeout=wait_timeout)
        if not ok:
            return False, reason
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
            return False, f"download_error:{type(exc).__name__}"
        cache.complete_download(
            config.device_code, item.file_id, path=target, size=result.size,
            sha256=result.sha256, range_supported=result.range_supported,
            elapsed_seconds=result.elapsed_seconds,
        )
        pipeline["downloaded_files"] += 1
        return True, ""

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
    if canvas_source is None and files:
        # 远端场景：为了确定画布需要先取一个文件。逐个尝试，避免因个别素材
        # 已过期/删除而中止整段作业；尝试次数有界。
        attempts = 0
        for candidate in files:
            attempts += 1
            if attempts > 5:
                break
            ok, reason = ensure_ready_for_canvas(candidate)
            if not ok:
                print(f"[canvas] 候选不可用({candidate.record_start}): {reason}",
                      flush=True)
                continue
            entry = cache.entry(config.device_code, candidate.file_id)
            if entry is not None and entry.path is not None:
                canvas_source = entry.path
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
    *, work_dir: Path, allow_source_paths: bool = False,
) -> tuple[tuple[int, int], str]:
    """用第一个可解码文件确定分析尺寸；不假定七天素材的分辨率。

    ``allow_source_paths`` 为本机素材目录场景：冷缓存时可以直接探测源文件，
    不必先把它下载（它是本地只读输入）。
    """
    del work_dir
    for item in files:
        entry = cache.entry(device_code, item.file_id)
        candidate = None
        if entry is not None and entry.path is not None and entry.path.is_file():
            candidate = entry.path
        elif allow_source_paths:
            source = Path(item.file_name)
            if source.is_file():
                candidate = source
        if candidate is None:
            continue
        probe = probe_recording(candidate)
        if probe.ok:
            return (probe.width, probe.height), str(candidate)
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


def _decode_calibration_frame(
    cache: ManagedRecordingCache, config: FactoryConfig,
    geometry: Mapping[str, Any], observation: Mapping[str, Any],
    size: tuple[int, int],
) -> np.ndarray | None:
    """按需解码一个独立校准观测的高清帧（有界：一次一帧）。"""
    from rtsp_annotator.ground_litter_profile_sampling import SequentialFrameReader

    block = str(observation.get("time_block") or "")
    identity, _, offset_text = block.partition("@")
    file_id = identity.split(":")[-1] or str(observation.get("file_id") or "")
    if not file_id:
        return None
    entry = cache.entry(config.device_code, file_id)
    if entry is None or entry.path is None:
        return None
    try:
        offset = float(offset_text or observation.get("offset_seconds") or 0.0)
    except (TypeError, ValueError):
        return None
    reader = SequentialFrameReader(entry.path)
    picked = reader.sample_at([offset], tolerance_seconds=1.5)
    captured = picked.get(offset)
    if captured is None:
        return None
    if not frame_quality(captured.frame).usable:
        return None
    return cv2.resize(captured.frame, size, interpolation=cv2.INTER_AREA)


def stage_composite_and_noise(
    config: FactoryConfig, cache: ManagedRecordingCache,
    files: Sequence[RecordingFile], geometry: dict[str, Any],
    groups: Sequence[dict[str, Any]], samples: Sequence[dict[str, Any]],
    report: dict[str, Any],
    *, build_blocks: set[str], calibration_blocks: set[str],
    calibration_observations: Sequence[dict[str, Any]] = (),
    calibration_assignment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """用高清帧合成参考、估计噪声；构建块与校准块严格分离（C3）。

    ``calibration_observations`` 是**独立校准日**的采样（不参与参考合成）。
    它们按外观匹配分配给各候选组，分配给某组的观测才允许进入该组的噪声估计。
    """
    size = (int(geometry["canvas_size"][0]), int(geometry["canvas_size"][1]))
    roi = roi_mask_from_geometry(geometry, size[0], size[1])
    from rtsp_annotator.ground_litter_profile_sampling import SequentialFrameReader

    by_id = {item.file_id: item for item in files}
    candidates: list[dict[str, Any]] = []
    assignment = dict(calibration_assignment or {})
    per_group_blocks: dict[str, set[str]] = {
        gid: set(payload.get("blocks") or [])
        for gid, payload in (assignment.get("per_group") or {}).items()
    }
    calibration_by_block: dict[str, dict[str, Any]] = {
        str(item["time_block"]): item for item in calibration_observations
        if item.get("time_block")
    }
    # 独立校准块与构建块不得重叠：重叠就说明"独立"是假的。
    overlap = set(calibration_blocks) & set(build_blocks)
    if overlap:
        raise SystemExit(
            "校准块与构建块重叠，拒绝用构建素材冒充独立校准："
            f"{sorted(overlap)[:4]}"
        )
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
        # C2：合成中间产物（逐帧高清 + 参考/掩膜/阈值图）单独记账。
        cache.commit_stage_bytes(
            "composite",
            frame_memory_bytes(len(frames), size)
            + frame_memory_bytes(4, size)
            + int(availability_roi.nbytes)
            + int(sum(mask.nbytes for mask in composite_masks)),
        )
        # R7 + C3：噪声估计必须用**独立校准日**的观测，而且只接受外观匹配到
        # 本组的那些块。参考自身的观测绝不能进噪声估计。
        group_id = str(group["group_id"])
        matched_blocks = per_group_blocks.get(group_id, set())
        assignment_payload = (assignment.get("per_group") or {}).get(group_id) or {}
        eval_frames: list[np.ndarray] = []
        eval_masks: list[np.ndarray] = []
        eval_blocks: list[str] = []

        def _collect(block_ids: Iterable[str]) -> tuple[list, list, list]:
            collected_frames: list[np.ndarray] = []
            collected_masks: list[np.ndarray] = []
            collected_blocks: list[str] = []
            for block_id in block_ids:
                # 校准块与构建块本来就不相交；这里再挡一次，防止未来改动
                # 让参考自身的观测混进噪声估计。
                if block_id in build_blocks:
                    continue
                observation = calibration_by_block.get(block_id)
                if observation is None:
                    continue
                frame = observation.get("hd_frame")
                if frame is None:
                    frame = _decode_calibration_frame(
                        cache, config, geometry, observation, size,
                    )
                if frame is None:
                    continue
                collected_frames.append(frame)
                collected_masks.append(roi.copy())
                collected_blocks.append(block_id)
            return collected_frames, collected_masks, collected_blocks

        # 主路径：本组**外观匹配**到的校准观测（与 group["time_blocks"] 不重叠，
        # 校准日样本从设计上就不会出现在构建组的 blocks 里）。
        eval_frames, eval_masks, eval_blocks = _collect(sorted(matched_blocks))
        degradation_reason: str | None = None
        if len(eval_frames) >= 2:
            noise_note = "independent_calibration_blocks"
        else:
            # 该校准素材不可用/外观不匹配：允许"跨组但仍是校准日"的独立观测，
            # 但必须标注原因；实在没有独立素材才退化为参考自身观测。
            cross_frames, cross_masks, cross_blocks = _collect(
                sorted(calibration_by_block)
            )
            if len(cross_frames) >= 2:
                eval_frames, eval_masks, eval_blocks = (
                    cross_frames, cross_masks, cross_blocks,
                )
                noise_note = "independent_calibration_day_cross_group"
                degradation_reason = (
                    str(assignment_payload.get("reason"))
                    or "no_appearance_match_for_group"
                )
            else:
                eval_frames, eval_masks, eval_blocks = frames, composite_masks, blocks
                noise_note = "low_support_used_all_blocks"
                degradation_reason = (
                    "no_independent_calibration_material"
                    if not calibration_by_block else "calibration_frames_undecodable"
                )
        # 独立性口径：来源必须是校准日（不是构建语料）；"外观是否匹配本组"
        # 单独记账，因为它决定这组噪声标定代表的是不是同一画面条件。
        noise_source = (
            "calibration_day" if noise_note.startswith("independent_calibration")
            else "reference_self"
        )
        independent = noise_source == "calibration_day" and len(set(eval_blocks)) >= 2
        appearance_matched = bool(matched_blocks) and all(
            block in matched_blocks for block in eval_blocks
        )
        # 阈值图必须与参考同尺寸（Bank loader 契约）。逐像素分位代价随帧数线性
        # 增长，这里按时间均匀抽稀到 24 帧（块覆盖保持不变）。
        noise = estimate_noise(
            refined, eval_frames, eval_masks, eval_blocks, stride=1,
            config={
                "max_estimate_frames": 24,
                "_calibration_block_count": len(eval_blocks),
                "_independent_calibration": independent,
                "_appearance_matched": appearance_matched,
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
        # 合成中间帧在本组评分/描述子提取后即可丢弃，释放记账。
        cache.release_stage_bytes("composite", frame_memory_bytes(len(frames), size))
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
            "noise_calibration": {
                "source": noise_source,
                "note": noise_note,
                "independent_blocks": len(set(eval_blocks)),
                "appearance_match_distance": assignment_payload.get("distance"),
                "appearance_match_reason": assignment_payload.get("reason"),
            "appearance_match_limit": assignment_payload.get("limit"),
            "matched_blocks": sorted(matched_blocks),
                "degradation_reason": degradation_reason,
                "independent_of_reference": bool(independent),
                "appearance_matched": bool(appearance_matched),
            },
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
            "noise_calibration": item["noise_calibration"],
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


@dataclass(slots=True)
class ReplayScoreMatrix:
    """逐 ``(frame, candidate)`` 只算一次的评分缓存（C1）。

    基线、留一法、保守删除复核与终选复核都从同一矩阵取分，因此
    "删掉某候选" 与 "删掉另一个候选" 之间的差异只来自候选集合本身，
    不再混入采样密度差（v2 的 ``loo_stride=4`` 问题）。
    """

    profile_ids: tuple[str, ...]
    frames: tuple[dict[str, Any], ...]
    scores: list[dict[str, dict[str, Any]]]
    frame_digests: tuple[str, ...]
    frame_count: int
    replay_w: int = 0
    replay_h: int = 0
    stride_supported: bool = True

    def subset_index(self, subset: Sequence[str]) -> list[int]:
        wanted = list(dict.fromkeys(str(pid) for pid in subset))
        return [
            index for index, pid in enumerate(self.profile_ids) if pid in set(wanted)
        ]

    def subset_digests(self, subset: Sequence[str]) -> tuple[str, ...]:
        """子集实际参与评分的帧摘要（用于证明基线/LOO 帧完全一致）。"""
        del subset
        return self.frame_digests


def build_replay_score_matrix(
    candidates: Sequence[dict[str, Any]],
    replay_frames: Sequence[dict[str, Any]], geometry: Mapping[str, Any],
    matcher: Mapping[str, Any], *, replay_w: int = 960, replay_h: int = 540,
) -> ReplayScoreMatrix:
    """把回放帧缩放到固定画布并对**每个候选**评一次分。

    历史缺陷：v2 的 baseline 用 stride=1、留一法用 stride=4，两者看到的帧不同，
    ``effective_fraction_delta`` 里混进了采样密度差。这里把评分与时间线推进分开：
    评分只做一次，之后所有对照都复用同一矩阵。
    """
    if not candidates:
        raise SystemExit("没有任何可用候选，无法建库")
    if not replay_frames:
        raise SystemExit("没有可用的连续回放帧")
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
    profile_ids = tuple(item["profile_id"] for item in candidates)
    scores: list[dict[str, dict[str, Any]]] = []
    for row in replay_frames:
        frame = row["frame"]
        # 回放画布与缩放后的参考必须同尺寸，否则评分会因尺寸不一致被跳过
        # （历史上这会让所有 tick 都变成 NO_ELIGIBLE_PROFILE）。
        if frame.shape[0] != replay_h or frame.shape[1] != replay_w:
            frame = cv2.resize(
                frame, (replay_w, replay_h), interpolation=cv2.INTER_AREA,
            )
        per_frame: dict[str, dict[str, Any]] = {}
        for pid in profile_ids:
            context = contexts[pid]
            try:
                score = score_profile(
                    frame, context["reference"], context["valid"], roi,
                    profile_id=pid, config=matcher,
                )
            except BankError:
                continue
            outcome = evaluate_match(score, frozen[pid], matcher)
            per_frame[pid] = {
                "score": float(score.score),
                "enter_eligible": bool(outcome.enter_eligible),
                "hold_eligible": bool(outcome.hold_eligible),
            }
        scores.append(per_frame)
    digests = tuple(
        str(row.get("frame_sha256") or row.get("source_time"))
        for row in replay_frames
    )
    # C2：评分完成后回放帧的**像素**不再被任何对照步骤使用（时间线只读时间戳
    # 与分数），保留它们只会让内存峰值随帧数线性增长。这里立即丢弃像素，
    # 只留时间轴、帧指纹与帧尺寸等轻量记录。
    light_frames = tuple(
        {
            key: row.get(key)
            for key in ("file_id", "record_start", "offset_seconds",
                        "source_time", "replay_time", "tick_interval_seconds",
                        "frame_sha256")
        }
        for row in replay_frames
    )
    return ReplayScoreMatrix(
        profile_ids=profile_ids, frames=light_frames, scores=scores,
        frame_digests=digests, frame_count=len(replay_frames),
        replay_w=replay_w, replay_h=replay_h,
    )


def project_selection_timeline(
    matrix: ReplayScoreMatrix, subset: Sequence[str], *, selector_config: Mapping[str, Any],
    bank_id: str, bank_version: str, view_id: str, nominal_tick: float,
    stride: int = 1, join_gap_seconds: float | None = None,
    stop_below_fraction: float | None = None,
) -> dict[str, Any]:
    """在评分矩阵上推进 Selector 状态机（不重新评分）。

    ``stop_below_fraction`` 用于保守删除的早期退出：一旦有效覆盖已经低于
    "明显不如基线" 的界限，就停止本轮（该候选必然要保留），避免无谓计算。
    """
    subset_ids = [str(pid) for pid in dict.fromkeys(subset)]
    if not subset_ids:
        raise SystemExit("回放子集不能为空")
    step = max(1, int(stride))
    indexes = list(range(0, matrix.frame_count, step))
    join_gap = (
        float(selector_config.get("join_gap_seconds", 300.0))
        if join_gap_seconds is None else float(join_gap_seconds)
    )
    # 子集自己的真实节拍：抽样 stride>1 时相邻观测间隔变大，如果仍沿用原
    # nominal，`_Evidence` 的连续证据窗口会把每次观测都当成断层清空，
    # 于是"抽样变稀"被误读成"删候选导致覆盖归零"。这里按实际观测时刻重算。
    times = [
        float(matrix.frames[index].get("replay_time")
              or matrix.frames[index]["source_time"])
        for index in indexes
    ]
    per_tick_intervals: list[float] = []
    for position, timestamp in enumerate(times):
        if position + 1 < len(times):
            per_tick_intervals.append(max(1e-6, times[position + 1] - timestamp))
        elif per_tick_intervals:
            per_tick_intervals.append(per_tick_intervals[-1])
        else:
            per_tick_intervals.append(float(nominal_tick))
    derived_nominal = (
        sorted(per_tick_intervals)[len(per_tick_intervals) // 2]
        if per_tick_intervals else float(nominal_tick)
    )
    run_config = dict(selector_config)
    run_config["tick_interval_seconds"] = float(derived_nominal)
    run_config["result_validity_seconds"] = max(
        float(run_config.get("result_validity_seconds", derived_nominal)),
        float(derived_nominal),
    )
    run_config["max_observation_gap_seconds"] = max(
        float(run_config.get("max_observation_gap_seconds", 4.0)),
        2.0 * float(derived_nominal),
    )
    selector = ProfileSelector(
        bank_id=bank_id, bank_version=bank_version, view_id=view_id,
        profile_ids=subset_ids, config=run_config,
    )
    for index, per_tick in zip(indexes, per_tick_intervals):
        row = matrix.frames[index]
        per_frame = matrix.scores[index]
        current_id = selector.selected_profile_id
        current = None
        matches: list[CandidateMatch] = []
        for pid in subset_ids:
            entry = per_frame.get(pid)
            if entry is None:
                continue
            candidate = CandidateMatch(
                pid, entry["score"], entry["enter_eligible"],
                entry["hold_eligible"], diagnostics={"fresh": True},
            )
            matches.append(candidate)
            if pid == current_id:
                current = candidate
        timestamp = float(row.get("replay_time") or row["source_time"])
        decision = selector.observe(
            timestamp=timestamp, current=current, candidates=matches,
            tested_profile_ids=subset_ids,
            tick_interval_seconds=float(per_tick),
        )
        if decision.commit_requested:
            selector.commit(profile_id=decision.commit_profile_id,
                            timestamp=timestamp)
        if stop_below_fraction is not None:
            summary = selector.summarise(join_gap_seconds=join_gap)
            if summary["effective_fraction"] < stop_below_fraction:
                summary["early_stop_index"] = index
                summary["stride"] = step
                return summary
    summary = selector.summarise(join_gap_seconds=join_gap)
    summary["stride"] = step
    summary["scored_frames"] = len(indexes)
    return summary


def replay_all_candidates(
    config: "FactoryConfig", candidates: Sequence[dict[str, Any]],
    replay_frames: Sequence[dict[str, Any]], geometry: Mapping[str, Any],
    matcher: Mapping[str, Any], report: dict[str, Any], *,
    replay_w: int = 960, replay_h: int = 540,
    leave_one_out: bool = True, loo_stride: int = 1,
) -> dict[str, Any]:
    """共同回放：所有候选面对**同一** ``(source_time, frame)``（R2）。

    C1 修复：评分矩阵只构建一次，基线、留一法与终选复核都从同一矩阵取分，
    因此三者的帧、时间轴与观测条件**完全相同**。

    注意 ``loo_stride``：v2 用 ``stride=4`` 跑留一法，既改变了参与评分的帧，
    也改变了观测节拍（相邻观测间隔变大→连续证据窗口不同），这是 C1 认定
    "对照不公平" 的直接来源。参数仍接受以兼容历史脚本，但**一律按 1 执行**，
    实际取值记录在 ``report['replay']['loo_stride']`` 与 ``loo_stride_ignored``。
    """
    matrix = build_replay_score_matrix(
        candidates, replay_frames, geometry, matcher,
        replay_w=replay_w, replay_h=replay_h,
    )
    requested_stride = max(1, int(loo_stride))
    loo_stride = 1
    canvas = [matrix.replay_w, matrix.replay_h]
    replay_config = dict(matcher.get("selection", {}))
    nominal = float(replay_frames[0].get("tick_interval_seconds") or 2.0)
    nominal = max(0.05, min(nominal, 3600.0))
    replay_config["tick_interval_seconds"] = nominal
    replay_config["result_validity_seconds"] = max(nominal, 4.0)
    # 观测间隔上限必须随实际节拍缩放；否则每个 tick 都会清空连续证据，
    # 动态覆盖恒为 0（历史上这正是“静态 1.0 / 动态 0.0”的成因之一）。
    replay_config["max_observation_gap_seconds"] = max(4.0, 2.0 * nominal)
    replay_config.setdefault("join_gap_seconds", max(300.0, 4.0 * nominal))
    bank_id = str(getattr(config, "bank_id", "bank"))
    bank_version = str(getattr(config, "version", "v1"))
    view_id = str(geometry.get("view_id", "view_0"))

    def run(subset: Sequence[str], stride: int = 1) -> dict[str, Any]:
        return project_selection_timeline(
            matrix, subset, selector_config=replay_config, bank_id=bank_id,
            bank_version=bank_version, view_id=view_id,
            nominal_tick=nominal, stride=stride,
            join_gap_seconds=replay_config["join_gap_seconds"],
        )

    all_ids = list(matrix.profile_ids)
    baseline = run(all_ids, stride=1)
    report["replay"] = {
        "scored_ticks": matrix.frame_count,
        "canvas": canvas,
        "nominal_tick_seconds": round(nominal, 4),
        "frame_reuse": "all candidates share the same frame and source_time",
        "score_matrix": {
            "method": "cache_per_frame_per_candidate",
            "candidates": len(matrix.profile_ids),
            "frames": matrix.frame_count,
            "entries": matrix.frame_count * len(matrix.profile_ids),
            "frame_digests": list(matrix.frame_digests[:8]),
        },
        "loo_stride": loo_stride,
        "loo_stride_requested": requested_stride,
        "loo_stride_ignored": requested_stride != loo_stride,
        "loo_frames_identical_to_baseline": True,
    }
    result: dict[str, Any] = {
        "baseline": baseline,
        "envelopes": {
            item["profile_id"]: item["envelope"] for item in candidates
        },
        "score_matrix": matrix,
        "selector_config": dict(replay_config),
        "nominal_tick_seconds": nominal,
        "view_id": view_id,
        "baseline_frame_digests": list(matrix.frame_digests),
        "report": report,
    }
    if not leave_one_out or len(candidates) < 2:
        result["leave_one_out"] = {}
        return result
    loo: dict[str, Any] = {}
    for item in candidates:
        subset = [other["profile_id"] for other in candidates
                  if other["profile_id"] != item["profile_id"]]
        summary = run(subset, stride=max(1, int(loo_stride)))
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
            "scored_frames": summary.get("scored_frames"),
        }
    result["leave_one_out"] = loo
    return result


def compare_selection_metrics(
    config: "FactoryConfig", candidates: Sequence[dict[str, Any]],
    replay_frames: Sequence[dict[str, Any]], geometry: Mapping[str, Any],
    matcher: Mapping[str, Any], *, replay_w: int = 960, replay_h: int = 540,
    subsets: Sequence[Sequence[str]] = (),
    replay: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """候选增删的显式动态对照（R2 验收证据）。

    对给定候选子集各跑一次**真实共同回放**，返回指标；用于回答“删掉这些候选
    后有效覆盖/暂停是否变差”，而不是只看静态排名。

    C1 修复：若传入 ``replay``（含评分矩阵），子集直接复用矩阵取分，
    不再重新评分，因此与基线/留一法逐帧可比。
    """
    matrix = None
    selector_config: Mapping[str, Any] = dict(matcher.get("selection", {}))
    nominal = 2.0
    if replay is not None and replay.get("score_matrix") is not None:
        matrix = replay["score_matrix"]
        selector_config = replay.get("selector_config") or selector_config
        nominal = float(replay.get("nominal_tick_seconds") or 2.0)
    else:
        matrix = build_replay_score_matrix(
            candidates, replay_frames, geometry, matcher,
            replay_w=replay_w, replay_h=replay_h,
        )
        nominal = float(replay_frames[0].get("tick_interval_seconds") or 2.0)
        nominal = max(0.05, min(nominal, 3600.0))
        selector_config = dict(selector_config)
        selector_config["tick_interval_seconds"] = nominal
        selector_config["result_validity_seconds"] = max(nominal, 4.0)
        selector_config["max_observation_gap_seconds"] = max(4.0, 2.0 * nominal)
        selector_config.setdefault("join_gap_seconds", max(300.0, 4.0 * nominal))
    rows: list[dict[str, Any]] = []
    for subset in subsets:
        chosen = [item for item in candidates if item["profile_id"] in set(subset)]
        if not chosen:
            continue
        summary = project_selection_timeline(
            matrix, [item["profile_id"] for item in chosen],
            selector_config=selector_config,
            bank_id=str(getattr(config, "bank_id", "bank")),
            bank_version=str(getattr(config, "version", "v1")),
            view_id=str(geometry.get("view_id", "view_0")),
            nominal_tick=nominal, stride=1,
            join_gap_seconds=float(selector_config.get("join_gap_seconds", 300.0)),
        )
        rows.append({
            "subset": list(subset),
            "size": len(chosen),
            "summary": summary,
            "frames": matrix.frame_count if matrix is not None else len(replay_frames),
            "envelopes_used": sorted(
                item["profile_id"] for item in chosen if item.get("envelope")
            ),
        })
    return rows


def _one_shot_prune_from_metrics(
    config: "FactoryConfig", candidates: Sequence[dict[str, Any]],
    replay: Mapping[str, Any], report: dict[str, Any], *,
    min_coverage_delta: float, max_pause_delta: float,
) -> list[dict[str, Any]]:
    """只拿到预计算留一法指标时的旧语义回退（非工厂主路径）。

    该方法**不是** C1 的定稿路径（它无法做逐次复核），保留它只为不静默改变
    以「precomputed LOO」调用历史脚本的行为；报告里会显式标注 ``legacy``。
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
                "profile_id": item["profile_id"], "group_id": item.get("group_id"),
                "reason": "DYNAMICALLY_REDUNDANT", "leave_one_out": stats,
            })
    if not keep:
        ranked = sorted(
            candidates,
            key=lambda entry: (
                -float((loo.get(entry["profile_id"]) or {}).get(
                    "effective_fraction_delta", 0.0)),
                entry["profile_id"],
            ),
        )
        keep = [ranked[0]]
        removed = [row for row in removed
                   if row["profile_id"] != ranked[0]["profile_id"]]
        report.setdefault("pruning_warnings", []).append(
            "ALL_CANDIDATES_REDUNDANT: 留一法显示每个候选都可删除；已回退到"
            "保留贡献最大的候选。请检查冻结包络是否过松或候选区分度是否过低。"
        )
    if len(keep) > config.max_profiles:
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
                "profile_id": item["profile_id"], "group_id": item.get("group_id"),
                "reason": "EXCEEDS_RESOURCE_LIMIT",
            })
        keep = ordered[: config.max_profiles]
    report["pruning"] = {
        "removed": removed,
        "kept": [item["profile_id"] for item in keep],
        "baseline": dict(replay.get("baseline") or {}),
        "leave_one_out": loo,
        "min_coverage_delta": min_coverage_delta,
        "max_pause_delta": max_pause_delta,
        "method": "leave_one_out_dynamic_replay",
        "implementation_path": "legacy_one_shot_from_precomputed_loo",
        "note": ("兼容路径：只有预计算 LOO，没有逐次复核；工厂主流程不使用该路径"),
    }
    return keep


def prune_profiles_conservatively(
    config: "FactoryConfig", candidates: Sequence[dict[str, Any]],
    replay: Mapping[str, Any], report: dict[str, Any], *,
    min_coverage_delta: float = 0.005, max_pause_delta: float = 5.0,
) -> list[dict[str, Any]]:
    """保守**逐次**删除：每轮只删一个，删后重新复核剩余集合（C1）。

    v2 的缺陷是"一次性删除"：所有 LOO 代价都相对全库计算，删掉一个候选后
    另一个原本冗余的候选可能不再冗余（两个完全相同的参考就是最典型的反例）。
    这里每轮只删除当前 LOO 中代价最小的一个冗余候选，然后**在剩余集合上重新
    计算 LOO**，保证删除集合内部不互相掩盖。

    删除规则（必须同时满足）：
    * 去掉它不会让有效覆盖下降超过 ``min_coverage_delta``；
    * 去掉它不会让最长暂停增加超过 ``max_pause_delta``。
    """
    matrix = replay.get("score_matrix")
    if matrix is None:
        # 兼容路径：调用方只给了留一法指标（没有原始帧/矩阵）。
        # 工厂主流程永远走矩阵路径；这里保留旧语义以免历史脚本与验收脚本
        # 直接以「precomputed LOO」调用时静默改变行为。
        return _one_shot_prune_from_metrics(
            config, candidates, replay, report,
            min_coverage_delta=min_coverage_delta, max_pause_delta=max_pause_delta,
        )
    selector_config = dict(replay.get("selector_config") or {})
    nominal = float(replay.get("nominal_tick_seconds") or 2.0)
    join_gap = float(selector_config.get("join_gap_seconds", 300.0))
    bank_id = str(getattr(config, "bank_id", "bank"))
    bank_version = str(getattr(config, "version", "v1"))
    view_id = str(replay.get("view_id", "view_0"))

    def timeline(subset: Sequence[str], *, stop_below_fraction: float | None = None):
        return project_selection_timeline(
            matrix, subset, selector_config=selector_config, bank_id=bank_id,
            bank_version=bank_version, view_id=view_id, nominal_tick=nominal,
            stride=1, join_gap_seconds=join_gap,
            stop_below_fraction=stop_below_fraction,
        )

    baseline = dict(replay.get("baseline") or {})
    if not baseline:
        baseline = timeline(list(matrix.profile_ids))
    base_coverage = float(baseline.get("effective_fraction", 0.0))
    base_pause = float(baseline.get("pause_max", 0.0))
    remaining: list[dict[str, Any]] = list(candidates)
    removed: list[dict[str, Any]] = []
    deletion_order: list[dict[str, Any]] = []
    loo_history: dict[str, dict[str, Any]] = {}
    # 每轮的 LOO 都在**当前剩余集合**上重算；历史记录保留最后一次观测值。
    while len(remaining) > 1:
        current_ids = [item["profile_id"] for item in remaining]
        evaluations: list[dict[str, Any]] = []
        current_summary = baseline if len(remaining) == len(candidates) else timeline(current_ids)
        for item in remaining:
            subset = [pid for pid in current_ids if pid != item["profile_id"]]
            summary = timeline(
                subset, stop_below_fraction=min_coverage_delta,
            )
            coverage_delta = float(current_summary.get("effective_fraction", 0.0)) - float(
                summary["effective_fraction"]
            )
            pause_delta = float(summary["pause_max"]) - float(
                current_summary.get("pause_max", 0.0)
            )
            evaluations.append({
                "profile_id": item["profile_id"],
                "effective_fraction_delta": round(coverage_delta, 5),
                "pause_max_delta": round(pause_delta, 3),
                "pause_max_without": summary["pause_max"],
                "switch_count_without": summary["switch_count"],
                "early_stopped": "early_stop_index" in summary,
            })
        evaluations.sort(key=lambda row: (
            row["effective_fraction_delta"], row["pause_max_delta"], row["profile_id"],
        ))
        best = evaluations[0]
        if (best["effective_fraction_delta"] > min_coverage_delta
                or best["pause_max_delta"] > max_pause_delta
                or best["early_stopped"]):
            break
        victim = next(item for item in remaining
                      if item["profile_id"] == best["profile_id"])
        # 删除后必须重新复核剩余集合：既写入本轮证据，也作为下一轮的基线。
        after = timeline([item["profile_id"] for item in remaining
                          if item["profile_id"] != victim["profile_id"]])
        remaining = [item for item in remaining
                     if item["profile_id"] != victim["profile_id"]]
        loo_history[victim["profile_id"]] = best
        removal = {
            "profile_id": victim["profile_id"],
            "group_id": victim["group_id"],
            "reason": "DYNAMICALLY_REDUNDANT",
            "round": len(deletion_order) + 1,
            "leave_one_out": best,
            "set_size_after": len(remaining),
            "summary_after": {
                "effective_fraction": after["effective_fraction"],
                "pause_max": after["pause_max"],
                "switch_count": after["switch_count"],
            },
        }
        removed.append(removal)
        deletion_order.append({
            "round": removal["round"],
            "removed": victim["profile_id"],
            "candidates_evaluated": [row["profile_id"] for row in evaluations],
            "chosen_reason": "min_loo_cost_below_threshold",
            "loo_cost": best,
        })
        baseline = after
    if not remaining:
        # 理论不可达（循环条件保证至少留 1），保留保底逻辑以防未来改动。
        ranked = sorted(candidates, key=lambda entry: entry["profile_id"])
        remaining = [ranked[0]]
        removed = [row for row in removed
                   if row["profile_id"] != ranked[0]["profile_id"]]
        report.setdefault("pruning_warnings", []).append(
            "ALL_CANDIDATES_REDUNDANT: 已回退到保留排序第一个候选"
        )
    keep = remaining
    if len(keep) > config.max_profiles:
        # 资源上限不是预设 N：超限时仍然逐次删除，且每次删除都做完整复核。
        keep = _trim_to_resource_limit(
            config, keep, timeline, removed, report,
            budget_baseline=baseline, min_coverage_delta=min_coverage_delta,
            max_pause_delta=max_pause_delta,
        )
    report["pruning"] = {
        "removed": removed,
        "kept": [item["profile_id"] for item in keep],
        "baseline": dict(replay.get("baseline") or {}),
        "leave_one_out": loo_history,
        "min_coverage_delta": min_coverage_delta,
        "max_pause_delta": max_pause_delta,
        "method": "conservative_one_at_a_time_dynamic_replay",
        "deletion_order": deletion_order,
        "note": ("逐次删除：每轮只在当前剩余集合上重算 LOO，删除后重新复核；"
                 "静态覆盖不是删除依据"),
    }
    return keep


def _trim_to_resource_limit(
    config: "FactoryConfig", keep: Sequence[dict[str, Any]], timeline: Any,
    removed: list[dict[str, Any]], report: dict[str, Any], *,
    budget_baseline: Mapping[str, Any], min_coverage_delta: float,
    max_pause_delta: float,
) -> list[dict[str, Any]]:
    """超过 ``max_profiles`` 时继续逐次删除，并保留终选复核证据。"""
    remaining = list(keep)
    current = dict(budget_baseline)
    while len(remaining) > config.max_profiles:
        evaluations: list[dict[str, Any]] = []
        for item in remaining:
            subset = [entry["profile_id"] for entry in remaining
                      if entry["profile_id"] != item["profile_id"]]
            summary = timeline(subset)
            evaluations.append({
                "profile_id": item["profile_id"],
                "effective_fraction_delta": round(
                    float(current.get("effective_fraction", 0.0))
                    - float(summary["effective_fraction"]), 5,
                ),
                "pause_max_delta": round(
                    float(summary["pause_max"])
                    - float(current.get("pause_max", 0.0)), 3,
                ),
                "pause_max_without": summary["pause_max"],
            })
        evaluations.sort(key=lambda row: (
            row["effective_fraction_delta"], row["pause_max_delta"], row["profile_id"],
        ))
        victim_row = evaluations[0]
        victim = next(item for item in remaining
                      if item["profile_id"] == victim_row["profile_id"])
        current = timeline([entry["profile_id"] for entry in remaining
                            if entry["profile_id"] != victim["profile_id"]])
        remaining = [entry for entry in remaining
                     if entry["profile_id"] != victim["profile_id"]]
        removed.append({
            "profile_id": victim["profile_id"], "group_id": victim["group_id"],
            "reason": "EXCEEDS_RESOURCE_LIMIT",
            "round": len(removed) + 1,
            "leave_one_out": victim_row,
            "summary_after": {
                "effective_fraction": current["effective_fraction"],
                "pause_max": current["pause_max"],
                "switch_count": current["switch_count"],
            },
        })
    report.setdefault("resource_trim", {})["final"] = {
        "effective_fraction": current.get("effective_fraction"),
        "pause_max": current.get("pause_max"),
        "switch_count": current.get("switch_count"),
    }
    return remaining


def select_profiles_dynamically(
    config: "FactoryConfig", candidates: Sequence[dict[str, Any]],
    replay: Mapping[str, Any], report: dict[str, Any], *,
    min_coverage_delta: float = 0.005, max_pause_delta: float = 5.0,
) -> list[dict[str, Any]]:
    """兼容入口：动态定稿一律走保守逐次删除（R2 + C1）。

    保留旧名字，因为历史报告与验收脚本按此调用；行为已改为
    :func:`prune_profiles_conservatively`，不再是一次性删除。
    """
    return prune_profiles_conservatively(
        config, candidates, replay, report,
        min_coverage_delta=min_coverage_delta, max_pause_delta=max_pause_delta,
    )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def _verify_selected_subsets(
    config: "FactoryConfig", candidates: Sequence[dict[str, Any]],
    selected: Sequence[dict[str, Any]], replay: Mapping[str, Any],
    geometry: Mapping[str, Any], matcher: Mapping[str, Any],
) -> dict[str, Any]:
    """终选集合与资源裁剪集合都要用**同一评分矩阵**重新复核（C1）。

    一次性删除的旧实现只给出"从全库删掉它"的代价；这里对最终要发布的集合
    重跑完整时间线，并逐帧核对帧摘要与基线一致，避免"选完之后没人验过"。
    """
    matrix = replay["score_matrix"]
    selector_config = dict(replay.get("selector_config") or {})
    nominal = float(replay.get("nominal_tick_seconds") or 2.0)
    join_gap = float(selector_config.get("join_gap_seconds", 300.0))
    baseline_digests = list(replay.get("baseline_frame_digests") or [])
    baseline = dict(replay.get("baseline") or {})

    def verify(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        ids = [item["profile_id"] for item in items]
        summary = project_selection_timeline(
            matrix, ids, selector_config=selector_config,
            bank_id=str(getattr(config, "bank_id", "bank")),
            bank_version=str(getattr(config, "version", "v1")),
            view_id=str(replay.get("view_id", geometry.get("view_id", "view_0"))),
            nominal_tick=nominal, stride=1, join_gap_seconds=join_gap,
        )
        digests = list(matrix.subset_digests(ids))
        return {
            "profile_ids": ids,
            "size": len(ids),
            "summary": {
                key: summary.get(key) for key in (
                    "effective_fraction", "pause_max", "pause_p95",
                    "switch_count", "observed_seconds", "effective_seconds",
                    "off_air_seconds", "non_observable_seconds",
                )
            },
            "frames": matrix.frame_count,
            "frames_identical_to_baseline": digests == baseline_digests,
            "coverage_delta_vs_baseline": round(
                float(baseline.get("effective_fraction", 0.0))
                - float(summary["effective_fraction"]), 5,
            ),
            "pause_max_delta_vs_baseline": round(
                float(summary["pause_max"]) - float(baseline.get("pause_max", 0.0)), 3,
            ),
            "verified_on_shared_score_matrix": True,
        }

    final_ids = [item["profile_id"] for item in selected]
    trimmed = list(selected)[: config.max_profiles]
    trimmed_ids = [item["profile_id"] for item in trimmed]
    return {
        "method": "replay_selected_set_on_shared_score_matrix",
        "baseline": baseline,
        "final_set": verify(selected),
        "resource_trimmed": verify(trimmed),
        "resource_trimmed_ids": trimmed_ids,
        "trimmed_equals_final": trimmed_ids == final_ids,
        "note": ("终选与资源裁剪后的集合都用与基线相同的帧、时间轴和观测条件复核；"
                 "不能只用删除前的旧指标发布"),
    }


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
        elif config.calibration_day:
            # 显式指定校准日：构建日 = 早于校准日的全部日期，盲测日 = 最晚日期。
            calibration = str(config.calibration_day)
            if calibration not in days:
                raise SystemExit(
                    f"--calibration-day {calibration} 不在可用日期 {days}"
                )
            build_days = [day for day in days if day < calibration]
            later = [day for day in days if day > calibration]
            blind = later[-1] if later else calibration
            if not build_days:
                raise SystemExit("指定的校准日之前没有构建日")
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

        # 分析尺寸先由几何配置或源文件探测确定；**不**要求文件已经下载完毕，
        # 因为拉取本身属于有界流水线（R5）。
        geometry_size = None
        if config.geometry_path is not None:
            probe_geometry = json.loads(
                Path(config.geometry_path).read_text(encoding="utf-8")
            )
            canvas = probe_geometry.get("canvas_size")
            if canvas:
                geometry_size = (int(canvas[0]), int(canvas[1]))
        if geometry_size is None:
            geometry_size, _ = find_first_decodable(
                cache, config.device_code, build_files,
                work_dir=config.work_dir,
                allow_source_paths=args.input is not None,
            )
        size = geometry_size
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
        build_blocks = block_ids_by_day(samples, build_days)
        calibration_files = partition["calibration"]
        # C3：校准日的素材必须**真的被采样**，否则 calibration_blocks 恒为空，
        # 噪声只能退化成"用参考自己的观测"（v2 的 13/13
        # low_support_used_all_blocks / independent_of_reference=false）。
        calibration_observations: list[dict[str, Any]] = []
        if calibration_files:
            print(f"[calibration] sampling {len(calibration_files)} calibration files",
                  flush=True)
            # 校准采样会覆盖 report["pipeline"]/["sampling"]/["hd_plan"]，
            # 先备份构建侧的记录，避免校准阶段把构建阶段的证据冲掉。
            preserved = {
                key: report.get(key)
                for key in ("pipeline", "sampling", "hd_plan", "registration",
                            "frozen_canvas")
            }
            with contextlib.redirect_stdout(io.StringIO()):
                cal_samples, cal_details, _cal_hd, cal_descriptors = (
                    stream_materialize_and_sample(
                        config, None if args.input is not None else client,
                        RecordingDownloader(), cache, calibration_files,
                        geometry, report,
                        prefetch_slots=int(getattr(args, "prefetch_slots", 2) or 2),
                    )
                )
            calibration_pipeline = report.get("pipeline")
            calibration_sampling = report.get("sampling")
            for key, value in preserved.items():
                if value is not None:
                    report[key] = value
            report["calibration_sampling"] = {
                "pipeline": calibration_pipeline,
                "sampling": calibration_sampling,
            }
            del cal_details, _cal_hd
            for sample in cal_samples:
                calibration_observations.append(sample)
            descriptors.update({
                item["time_block"]: item["descriptor"]
                for item in calibration_observations if item.get("time_block")
            })
            del cal_descriptors
        calibration_blocks = {
            str(item["time_block"]) for item in calibration_observations
            if item.get("time_block")
        }
        report["noise_calibration_plan"] = {
            "calibration_files": len(calibration_files),
            "calibration_observations": len(calibration_observations),
            "calibration_blocks": len(calibration_blocks),
            "build_blocks": len(build_blocks),
            "overlap_blocks": sorted(calibration_blocks & build_blocks),
            "blind_files": len(partition.get("blind") or []),
            "note": ("独立校准观测来自校准日，不参与参考合成；"
                     "构建块与校准块必须无交集"),
        }
        # 外观匹配：只有与某候选组外观接近的校准观测才能给该组估噪声。
        calibration_assignment = select_calibration_observations(
            groups, calibration_observations, descriptors=descriptors,
            group_members={row["group_id"]: row.get("members") or [] for row in groups},
        )
        report["noise_calibration_assignment"] = calibration_assignment
        # 高清阶段按稳定身份重拉：preview 释放过的临时 PS 必须重新取得，
        # 不能用小图放大冒充高清素材。
        # C2：只重拉**本阶段真正需要**的文件（有可用构建样本的构建文件 +
        # 校准文件），盲测日素材既不下拉也不计入阶段字节。
        needed_hd_files = sorted(
            {str(sample["file_id"]) for sample in samples
             if sample.get("file_id")}
            | {str(item.file_id) for item in calibration_files}
        )
        report["material_plan"] = {
            "build_files": len(build_files),
            "calibration_files": len(calibration_files),
            "blind_files": len(partition.get("blind") or []),
            "needed_file_ids": needed_hd_files,
            "blind_files_excluded_from_hd": [
                str(item.file_id) for item in (partition.get("blind") or [])
            ],
        }
        repull = _ensure_remote_entries(
            config, args, cache, files, report,
            local_paths=local_paths if args.input is not None else {},
            needed_file_ids=needed_hd_files,
        )
        report["hd_repull"] = repull
        candidates = stage_composite_and_noise(
            config, cache, build_files, geometry, groups, samples, report,
            build_blocks=build_blocks, calibration_blocks=calibration_blocks,
            calibration_observations=calibration_observations,
            calibration_assignment=calibration_assignment,
        )
        if not candidates:
            raise SystemExit("高清合成没有得到任何候选参考；见报告 composite")
        # C2：合成消费完成后立刻归还构建 PS，不把临时盘占用留到全流程结束。
        _release_materialized(
            cache, config, build_files, report, stage="after_composite",
            require_committed=("preview",),
        )
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
            "frame_bytes_estimate": frame_memory_bytes(
                len(replay_frames), size,
            ),
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
        # C2：包络拟合已消费完校准素材，立即归还临时 PS。
        _release_materialized(
            cache, config, partition["calibration"], report,
            stage="after_envelope_fit",
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
        # C1：baseline 与 LOO 共用同一评分矩阵，loo_stride 默认 1（与基线同帧）。
        print("[select] joint replay over shared frames", flush=True)
        replay = replay_all_candidates(
            config, candidates, replay_frames, geometry, matcher, report,
            leave_one_out=True,
            loo_stride=int(getattr(args, "loo_stride", 1) or 1),
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
            "method": "conservative_one_at_a_time_dynamic_replay",
            "loo_stride": report["replay"]["loo_stride"],
            "loo_frames_identical_to_baseline":
                report["replay"]["loo_frames_identical_to_baseline"],
        }
        # C1 终选复核：删除后的集合必须重新跑完整回放，并与基线逐帧对照。
        final_verification = _verify_selected_subsets(
            config, candidates, selected, replay, geometry, matcher,
        )
        report["final_set_verification"] = final_verification
        # 显式候选增删对照（验收证据）：全库、终选、资源裁剪后、以及每个候选单独。
        trimmed_ids = final_verification["resource_trimmed_ids"]
        subset_compare = compare_selection_metrics(
            config, candidates, replay_frames, geometry, matcher,
            replay=replay,
            subsets=[
                [item["profile_id"] for item in candidates],
                [item["profile_id"] for item in selected],
                trimmed_ids,
            ] + [[item["profile_id"]] for item in candidates[: max(1, len(candidates))]],
        )
        report["selection_comparison"] = subset_compare
        fair = {
            "method": "cached_score_matrix_conservative_pruning",
            "frames_identical_for_all_subsets": True,
            "frame_count": replay["score_matrix"].frame_count,
            "frame_digests": list(replay["baseline_frame_digests"]),
            "loo_stride": report["replay"]["loo_stride"],
            "candidates": [item["profile_id"] for item in candidates],
            "baseline": replay["baseline"],
            "deletion_order": report.get("pruning", {}).get("deletion_order", []),
            "kept": report.get("pruning", {}).get("kept", []),
            "final_set_verification": final_verification,
            "subset_comparison": subset_compare,
            "note": ("所有子集共用同一评分矩阵与同一时间轴；删除代价在删除当时的"
                     "剩余集合上重算，不是一次性从全库估算。"),
        }
        report["n_selection_fair_comparison"] = fair
        # 记录每 tick 的逐候选分数与决策，便于复核“同一帧”与选 N 过程。
        atomic_write_json(
            config.work_dir / "dynamic_selection.json",
            {"baseline": replay["baseline"], "leave_one_out": replay["leave_one_out"],
             "pruning": report.get("pruning", {}),
             "final_set_verification": final_verification,
             "selection_comparison": subset_compare},
        )
        atomic_write_json(
            config.work_dir / "n_selection_fair_comparison.json", fair,
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
                    "noise_calibration": item.get("noise_calibration"),
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
        # C2：全链路资源包络——磁盘（原始 PS / 中间产物）与内存（抽帧、回放帧）。
        stage_report = cache.stage_report()
        planned_final = [
            {
                "file_id": row.get("file_id"),
                "state": row.get("state"),
                "record_start": row.get("record_start"),
                "released": row.get("released"),
            }
            for row in (report.get("pipeline", {}).get("final_states") or [])
        ]
        report["resource_envelope"] = {
            "raw_cache_budget": config.raw_cache_budget,
            "work_budget": config.work_budget,
            "peak_raw_bytes": stage_report["peak_raw_bytes"],
            "peak_work_bytes": stage_report["peak_work_bytes"],
            "final_raw_bytes": stage_report["raw_bytes"],
            "final_work_bytes": stage_report["work_bytes"],
            "backpressure_observed": stage_report["backpressure"],
            "backpressure_reason": stage_report["backpressure_reason"],
            "stage_bytes": stage_report["stages"],
            "stage_events": stage_report["events"][-200:],
            "frame_memory_estimate": {
                "preview_frames": frame_memory_bytes(
                    len(samples), (config.preview_width,
                                   int(config.preview_width * size[1] / max(size[0], 1))),
                ),
                "replay_frames": frame_memory_bytes(len(replay_frames), size),
                "note": ("回放帧在评分矩阵构建后立即丢弃像素，因此峰值不等于"
                         "帧数×画布×3 的长期驻留；这是估算上限口径"),
            },
            "planned_final_states": planned_final,
            "planned_final_summary": {
                state: sum(1 for row in planned_final if row["state"] == state)
                for state in sorted({str(row["state"]) for row in planned_final})
            },
            "budget_shrinks": report.get("budget_shrinks", []),
            "material_failures": report.get("hd_repull", {}).get("failed", []),
            "note": ("原始 PS、抽帧/回放帧与合成中间产物分别记账；任何超预算都"
                     "记录为 backpressure/预算类失败，不伪装成素材失败"),
        }
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
    needed_file_ids: Sequence[str] | set[str] | None = None,
) -> dict[str, Any]:
    """确保**本阶段真正需要**的高清文件在盘上（C2）。

    v2 的缺陷：这里对 ``files``（分区后的全部文件，含盲测日）逐个重拉，
    盲测素材被下载却从不参与合成/校准，1.66GB 峰值里有相当部分是它。
    现在只处理调用方声明的 ``needed_file_ids``；未声明的文件既不下载也不
    计入阶段字节。
    """
    needed = (
        None if needed_file_ids is None
        else {str(value) for value in needed_file_ids}
    )
    targets = [
        item for item in files if needed is None or str(item.file_id) in needed
    ]
    skipped_not_needed = len(files) - len(targets)
    summary: dict[str, Any] = {
        "checked": len(targets), "considered": len(files), "reused_local": 0,
        "repulled": 0, "failed": [], "bytes": 0, "seconds": 0.0, "refreshes": 0,
        "skipped_not_needed": skipped_not_needed,
        "skipped_not_needed_ids": [
            item.file_id for item in files
            if needed is not None and str(item.file_id) not in needed
        ][:24],
        "failure_kinds": {},
        "needed_ids": sorted(needed) if needed is not None else None,
    }

    def note_failure(file_id: str, kind: str, reason: str) -> None:
        """失败必须分类：网络/预算受限 ≠ 素材质量失败。"""
        summary["failed"].append({
            "file_id": file_id, "kind": kind, "reason": reason[:160],
        })
        kinds = summary["failure_kinds"]
        kinds[kind] = kinds.get(kind, 0) + 1
        cache.log_stage_event(
            "hd_materialize", file_id=file_id, kind=kind, reason=reason[:160],
        )

    client: RecordingListClient | None = None
    downloader: RecordingDownloader | None = None
    policy = UrlRefreshPolicy()
    for item in targets:
        entry = cache.entry(config.device_code, item.file_id)
        if entry is not None and entry.path is not None and entry.path.is_file():
            existing_bytes = int(
                getattr(entry, "bytes", 0) or item.file_size or 0
            )
            if existing_bytes:
                cache.commit_stage_bytes("hd_materialize", existing_bytes)
            cache.log_stage_event(
                "hd_materialize", file_id=item.file_id, kind="reused_cache",
                bytes=existing_bytes,
            )
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
            cache.log_stage_event(
                "hd_materialize", file_id=item.file_id, kind="reused_local",
                bytes=int(local.stat().st_size),
            )
            continue
        if args.source is None:
            note_failure(item.file_id, "NO_SOURCE_TO_REPULL", "no --source configured")
            continue
        if client is None:
            client = RecordingListClient(headers=_auth_headers(args))
            downloader = RecordingDownloader()
        allowed, reason = cache.can_reserve(item.file_size)
        if not allowed:
            note_failure(item.file_id, "BUDGET_REJECTED", reason)
            continue
        window = ListQuery(config.device_code, item.record_start, item.record_end)
        try:
            fresh = downloader.fetch_url_for_file(
                client, window, item.file_id, policy=policy,
            )
        except Exception as exc:
            note_failure(item.file_id, "URL_REFRESH_FAILED",
                         f"{type(exc).__name__}: {exc}")
            continue
        try:
            target = cache.begin_download(config.device_code, item)
            result = downloader.download(
                fresh.url, target, expected_size=item.file_size, allow_resume=False,
            )
        except Exception as exc:
            cache.fail_download(config.device_code, item.file_id, str(exc)[:120])
            note_failure(item.file_id, "NETWORK_FAILURE",
                         f"{type(exc).__name__}: {exc}")
            continue
        if not file_looks_like_media(target):
            cache.fail_download(config.device_code, item.file_id,
                                "content_probe_failed")
            note_failure(item.file_id, "MATERIAL_INVALID", "content_probe_failed")
            continue
        cache.complete_download(
            config.device_code, item.file_id, path=target, size=result.size,
            sha256=result.sha256, range_supported=result.range_supported,
            elapsed_seconds=result.elapsed_seconds,
        )
        summary["repulled"] += 1
        summary["bytes"] += result.size
        summary["seconds"] += result.elapsed_seconds
        cache.commit_stage_bytes("hd_materialize", result.size)
        cache.log_stage_event(
            "hd_materialize", file_id=item.file_id, kind="downloaded",
            bytes=int(result.size),
        )
    summary["refreshes"] = policy.refresh_count
    summary["seconds"] = round(summary["seconds"], 3)
    summary["stage"] = cache.stage_report()
    return summary


def _release_materialized(
    cache: ManagedRecordingCache, config: FactoryConfig,
    files: Sequence[RecordingFile], report: dict[str, Any], *,
    stage: str, require_committed: Sequence[str] = (),
) -> dict[str, Any]:
    """消费→提交→释放：阶段结束后立刻归还临时 PS（C2）。

    ``require_committed`` 里只保留**已经真正提交**的阶段名；否则 release 会被
    安全策略拒绝，临时盘占用就一直留着到全流程结束。
    """
    released: list[str] = []
    denied: list[dict[str, str]] = []
    for item in files:
        entry = cache.entry(config.device_code, item.file_id)
        if entry is None or entry.path is None:
            continue
        guard = tuple(
            name for name in require_committed
            if cache.stage_committed(config.device_code, item.file_id, name)
        )
        if guard:
            ok = cache.release_file(
                config.device_code, item.file_id, require_committed=guard,
            )
        else:
            ok = cache.release_file(config.device_code, item.file_id)
        if ok:
            released.append(item.file_id)
        else:
            denied.append({
                "file_id": item.file_id,
                "reason": "release_denied",
            })
    summary = {
        "stage": stage, "released": released, "release_count": len(released),
        "denied": denied, "denied_count": len(denied),
        "space": cache.stage_report(),
    }
    report.setdefault("stage_releases", {})[stage] = summary
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
    per_file = max(1, frame_budget // max(1, len(ordered)))
    # 每个文件的抽帧要覆盖整段录像，而不是只解开头几帧：否则回放时间轴会
    # 退化成「同一秒内 105 个 tick」，覆盖率与暂停统计都失去意义。
    per_file_rows: list[list[dict[str, Any]]] = []
    for item in ordered:
        if sum(len(rows) for rows in per_file_rows) >= frame_budget:
            break
        entry = cache.entry(config.device_code, item.file_id)
        if entry is None or entry.path is None:
            continue
        try:
            base = parse_seconds(item.record_start)
        except Exception:
            base = 0.0
        probe = probe_recording(entry.path)
        duration = float(probe.duration_seconds or 0.0)
        if duration <= 0:
            try:
                duration = max(
                    0.0, parse_seconds(item.record_end) - base,
                )
            except Exception:
                duration = 0.0
        if duration <= 0:
            duration = float(per_file)
        offsets = [
            round(duration * (index + 0.5) / per_file, 3)
            for index in range(per_file)
        ]
        reader = SequentialFrameReader(entry.path)
        captured_rows: list[dict[str, Any]] = []
        try:
            if config.use_seek:
                picked = reader.sample_with_seek(offsets, tolerance_seconds=3.0)
            else:
                picked = reader.sample_at(offsets, tolerance_seconds=3.0)
        except Exception:
            picked = {}
        for offset in offsets:
            captured = picked.get(offset)
            if captured is None:
                continue
            resized = cv2.resize(
                captured.frame, size, interpolation=cv2.INTER_AREA,
            )
            captured_rows.append({
                "file_id": item.file_id,
                "record_start": item.record_start,
                "offset_seconds": round(float(captured.time_seconds), 3),
                "source_time": base + float(captured.time_seconds),
                "replay_time": base + float(offset),
                "frame": resized,
                "frame_sha256": sha256_bytes(
                    cv2.imencode(".jpg", resized,
                                 [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                ),
            })
        del reader
        _gc.collect()
        captured_rows.sort(key=lambda row: row["replay_time"])
        per_file_rows.append(captured_rows)
    rows = [row for group in per_file_rows for row in group]
    # 计划节拍 = 实际相邻回放点的间隔中位数（真实时间轴，不是文件内偏移）。
    if len(rows) > 1:
        deltas = sorted(
            rows[index + 1]["replay_time"] - rows[index]["replay_time"]
            for index in range(len(rows) - 1)
        )
        nominal = deltas[len(deltas) // 2]
        for row in rows:
            row["tick_interval_seconds"] = round(float(nominal), 4)
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
    parser.add_argument("--calibration-day", default="",
                        help="显式指定校准日（YYYY-MM-DD）；之前为构建日，之后为盲测日")
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
        calibration_day=getattr(args, "calibration_day", "") or "",
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
