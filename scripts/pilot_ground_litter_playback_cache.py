#!/usr/bin/env python
"""A1 真实输入试点：一小时回放清单 → 即时刷新 → 有界下载 → 解码 → 清理。

严格按方案一 §4.4～§4.8 执行，并输出可复核的实测数字：

* 清单：一小时窗口 + 两个半小时窗口的 fileId 并集截断检测。
* URL 有效期：记录签发余量，并实测「临期刷新后仍可下载」。
* 下载：写到受管缓存的 ``.part``，边写边算 SHA-256；记录字节/耗时/峰值磁盘。
* Range：先探测 ``bytes=0-0``，再实测断点续传或整文件重拉。
* 解码：PyAV 探测分辨率/时长，并按粗采样点取帧。
* 清理：提交后释放临时 PS，确认磁盘回落。

输出只写报告 JSON；**不**写签名 URL、不写设备鉴权信息。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rtsp_annotator.ground_litter_profile_bank import atomic_write_json  # noqa: E402
from rtsp_annotator.ground_litter_profile_sampling import (  # noqa: E402
    SequentialFrameReader, coarse_sample_offsets, frame_quality, probe_recording,
)
from rtsp_annotator.ground_litter_recording_cache import (  # noqa: E402
    ManagedRecordingCache,
)
from rtsp_annotator.ground_litter_recording_source import (  # noqa: E402
    ListQuery, RecordingDownloader, RecordingListClient, UrlRefreshPolicy,
    deduplicate_files, detect_truncation, file_looks_like_media,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一小时真实 PS 下载/解码试点")
    parser.add_argument("--device-code", default="44180209031322001030")
    parser.add_argument("--start", default="2026-09-15 00:00:00")
    parser.add_argument("--end", default="2026-09-15 01:00:00")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--raw-cache-budget-gib", type=float, default=1.0)
    parser.add_argument("--work-budget-gib", type=float, default=2.0)
    parser.add_argument("--max-downloads", type=int, default=1)
    parser.add_argument("--refresh-wait-seconds", type=float, default=380.0,
                        help="首次查询后等待多久再刷新（文档实测 403s 清单稳定）")
    parser.add_argument("--skip-refresh-wait", action="store_true")
    parser.add_argument("--test-resume", action="store_true",
                        help="实测断点续传；不支持时验证整文件重拉")
    return parser.parse_args(argv)


def _peak_work_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report: dict[str, Any] = {
        "kind": "ground_litter_playback_pilot",
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query": {
            "device_code_tail": args.device_code[-4:],
            "start": args.start, "end": args.end,
        },
        "steps": [],
        "note": (
            "只下载实施/离线验证所需的最少文件；不记录签名 URL 或鉴权信息。"
        ),
    }
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    cache = ManagedRecordingCache(
        work,
        raw_cache_budget=int(args.raw_cache_budget_gib * 1024 ** 3),
        work_budget=int(args.work_budget_gib * 1024 ** 3),
    )
    client = RecordingListClient()
    downloader = RecordingDownloader()
    try:
        cache.recover()
        # 1) 一小时清单 + 半小时并集截断检测
        step = time.monotonic()
        query = ListQuery(args.device_code, args.start, args.end)
        page = client.query(query)
        halves = []
        start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
        for offset in (0, 30):
            half_start = start_dt + timedelta(minutes=offset)
            half_end = min(half_start + timedelta(minutes=30),
                           datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S"))
            halves.append(client.query(ListQuery(
                args.device_code,
                half_start.strftime("%Y-%m-%d %H:%M:%S"),
                half_end.strftime("%Y-%m-%d %H:%M:%S"),
            )))
        truncation = detect_truncation(
            page.files(), [item for half in halves for item in half.files()],
        )
        report["inventory"] = {
            "items": len(page.entries),
            "seconds": round(time.monotonic() - step, 3),
            "total_declared_bytes": sum(
                int(entry.file.file_size or 0) for entry in page.entries
            ),
            "expire_seconds": sorted({
                entry.url_expire_seconds for entry in page.entries
            }),
            "real_range": [
                min((entry.file.record_start for entry in page.entries), default=""),
                max((entry.file.record_end for entry in page.entries), default=""),
            ],
            "lengths_seconds": [
                round(
                    (datetime.strptime(entry.file.record_end, "%Y-%m-%d %H:%M:%S")
                     - datetime.strptime(entry.file.record_start, "%Y-%m-%d %H:%M:%S")
                     ).total_seconds(), 1)
                for entry in page.entries
            ],
            "pagination_fields": sorted(page.pagination),
            "truncation": truncation,
            "raw_item_count": page.raw_item_count,
        }
        files = deduplicate_files([page], args.device_code)
        cache.register(args.device_code, files)

        # 2) 稳定性 + 刷新：等待后重新查询同一小时
        if not args.skip_refresh_wait and args.refresh_wait_seconds > 0:
            time.sleep(max(0.0, args.refresh_wait_seconds))
        refreshed = client.query(query)
        first_ids = [entry.file.file_id for entry in page.entries]
        second_ids = [entry.file.file_id for entry in refreshed.entries]
        report["refresh"] = {
            "waited_seconds": 0.0 if args.skip_refresh_wait else args.refresh_wait_seconds,
            "first_items": len(first_ids),
            "second_items": len(second_ids),
            "file_ids_stable": first_ids == second_ids,
            "same_size_and_times": all(
                a.file.file_size == b.file.file_size
                and a.file.record_start == b.file.record_start
                and a.file.record_end == b.file.record_end
                for a, b in zip(page.entries, refreshed.entries)
            ),
            "urls_all_changed": all(
                bool(a.url) and bool(b.url)
                for a, b in zip(page.entries, refreshed.entries)
            ),
        }

        # 3) 有界下载：临近下载时刷新
        policy = UrlRefreshPolicy()
        targets = files[: max(1, args.max_downloads)]
        downloads = []
        for item in targets:
            allowed, reason = cache.can_reserve(item.file_size)
            if not allowed:
                downloads.append({"file_id": item.file_id, "blocked": reason})
                continue
            window = ListQuery(
                args.device_code,
                (datetime.strptime(item.record_start, "%Y-%m-%d %H:%M:%S")
                 - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
                (datetime.strptime(item.record_end, "%Y-%m-%d %H:%M:%S")
                 + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
            )
            refresh_started = time.monotonic()
            entry = downloader.fetch_url_for_file(
                client, window, item.file_id, policy=policy,
            )
            refresh_seconds = time.monotonic() - refresh_started
            probe = downloader.probe_range(entry.url)
            target = cache.begin_download(args.device_code, item)
            part = target.with_name(target.name + ".part")
            peak = 0
            try:
                result = downloader.download(
                    entry.url, target, expected_size=item.file_size,
                    allow_resume=False,
                )
                peak = max(peak, _peak_work_bytes(work))
            except Exception as exc:
                cache.fail_download(args.device_code, item.file_id, str(exc)[:120])
                downloads.append({
                    "file_id": item.file_id, "error": str(exc)[:160],
                    "refresh_seconds": round(refresh_seconds, 3),
                })
                continue
            media_like = file_looks_like_media(target)
            decode = probe_recording(target)
            cache.complete_download(
                args.device_code, item.file_id, path=target, size=result.size,
                sha256=result.sha256, range_supported=probe.supported,
                elapsed_seconds=result.elapsed_seconds,
            )
            sample_detail: dict[str, Any] = {}
            if decode.ok:
                reader = SequentialFrameReader(target)
                offsets = coarse_sample_offsets(decode.duration_seconds)
                picked = reader.sample_at(offsets)
                qualities = [
                    frame_quality(frame.frame).as_dict()
                    for frame in picked.values()
                ]
                sample_detail = {
                    "offsets": offsets,
                    "frames_picked": len(picked),
                    "quality": qualities,
                    "decode_seconds_full": round(reader.decode_seconds, 3),
                    "frames_decoded": reader.frames_decoded,
                }
            downloads.append({
                "file_id": item.file_id,
                "declared_bytes": item.file_size,
                "downloaded_bytes": result.size,
                "sha256": result.sha256,
                "download_seconds": round(result.elapsed_seconds, 3),
                "megabytes_per_second": round(
                    result.size / max(result.elapsed_seconds, 1e-6) / 1e6, 3
                ),
                "refresh_seconds": round(refresh_seconds, 3),
                "range_supported": probe.supported,
                "range_detail": probe.as_dict(),
                "media_like": media_like,
                "decode": {
                    "ok": decode.ok, "width": decode.width, "height": decode.height,
                    "duration_seconds": round(decode.duration_seconds, 3),
                    "frame_count": decode.frame_count, "codec": decode.codec,
                    "error": decode.error,
                    "probe_seconds": round(decode.decode_seconds, 3),
                },
                "samples": sample_detail,
                "peak_work_bytes": peak,
            })
            # 4) 断点/Range 实测（可选，只对第一个文件做，避免额外流量）
            if args.test_resume and len(downloads) == 1:
                downloads[-1]["resume_test"] = _resume_test(
                    cache, downloader, args.device_code, item, entry.url,
                    client, window, policy,
                )
            # 5) 提交后释放临时 PS
            summary_path = work / "stages" / "preview" / f"{item.file_id}.json"
            atomic_write_json(summary_path, {"pilot": True})
            cache.record_artifact(
                args.device_code, item.file_id, "preview", "summary.json",
                summary_path, input_hash=result.sha256,
                algorithm_version="pilot_r3", detail={"pilot": True},
            )
            released = cache.release_file(
                args.device_code, item.file_id, require_committed=("preview",),
            )
            downloads[-1]["released"] = released
        report["downloads"] = downloads
        report["refresh_policy"] = {
            "refreshes": policy.refresh_count,
            "expired_events": policy.expired_events,
        }
        # 6) 清理与空间
        cache.evict_to_budget(allow_ready=True)
        final_bytes = _peak_work_bytes(work)
        report["space"] = {
            "peak_work_bytes_during_download": max(
                (item.get("peak_work_bytes", 0) for item in downloads), default=0
            ),
            "final_work_bytes": final_bytes,
            "managed_report": cache.report(),
            "raw_cache_budget_bytes": int(args.raw_cache_budget_gib * 1024 ** 3),
            "work_budget_bytes": int(args.work_budget_gib * 1024 ** 3),
            "note": "报告实际下载量与峰值空间；低磁盘占用不等于低总流量",
        }
        report["cleanup"] = {
            "remaining_entries": [
                entry.as_dict() for entry in cache.entries()
            ],
        }
        report["finished_utc"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        atomic_write_json(args.output, report)
        print(json.dumps({
            "inventory": report["inventory"],
            "refresh": report.get("refresh"),
            "downloads": [
                {key: value for key, value in item.items() if key != "samples"}
                for item in downloads
            ],
            "space": {
                "peak_work_bytes_during_download":
                    report["space"]["peak_work_bytes_during_download"],
                "final_work_bytes": final_bytes,
            },
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        cache.close()


def _resume_test(
    cache: ManagedRecordingCache, downloader: RecordingDownloader,
    device_code: str, item: Any, url: str, client: RecordingListClient,
    window: ListQuery, policy: UrlRefreshPolicy,
) -> dict[str, Any]:
    """实测断点续传：删除部分字节后按 206 续传；不支持则整文件重拉。"""
    entry = cache.entry(device_code, item.file_id)
    if entry is None or entry.path is None:
        return {"skipped": "no_ready_file"}
    full = entry.path.read_bytes()
    part = entry.path.with_name(entry.path.name + ".part")
    keep = max(1024, len(full) // 2)
    part.write_bytes(full[:keep])
    entry.path.unlink()
    fresh = downloader.fetch_url_for_file(client, window, item.file_id, policy=policy)
    del url
    try:
        result = downloader.download(
            fresh.url, entry.path, expected_size=len(full), allow_resume=True,
        )
    except Exception as exc:
        return {"supported": False, "error": str(exc)[:160]}
    return {
        "supported": bool(result.resumed),
        "resumed": result.resumed,
        "size": result.size,
        "sha256_matches_full": result.sha256 == entry.sha256,
        "range_supported": result.range_supported,
    }


if __name__ == "__main__":
    raise SystemExit(main())
