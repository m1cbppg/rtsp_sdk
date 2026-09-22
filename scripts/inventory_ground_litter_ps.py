#!/usr/bin/env python
"""A1：回放文件清单索引（远程 ctseelink 或本地目录）。

只做索引与元数据落盘：**不**下载媒体，也**不**把签名 URL 写入任何文件。
远程模式支持按小时窗口查询、按 deviceCode+fileId 去重、一小时/半小时并集截断检测。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rtsp_annotator.ground_litter_profile_bank import atomic_write_json  # noqa: E402
from rtsp_annotator.ground_litter_recording_source import (  # noqa: E402
    DEFAULT_FILE_URLS_ENDPOINT, ListQuery, RecordingListClient,
    RecordingSourceError, deduplicate_files, detect_truncation,
)

SENSITIVE_KEYS = {
    "url", "urlexpireseconds", "token", "signature", "timestamp",
    "accesstoken", "authorization", "password",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="索引回放文件清单（不下载媒体）")
    parser.add_argument("--source", choices=["ctseelink-file-urls", "local"],
                        default="ctseelink-file-urls")
    parser.add_argument("--endpoint", default=DEFAULT_FILE_URLS_ENDPOINT)
    parser.add_argument("--device-code", default=None)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--input", type=Path, default=None,
                        help="本地目录（--source local 时使用）")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-minutes", type=int, default=60)
    parser.add_argument("--check-truncation", action="store_true",
                        help="额外查询两个半小时窗口并比较 fileId 并集")
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--api-key", default=None)
    return parser.parse_args(argv)


def index_remote(args: argparse.Namespace) -> dict[str, Any]:
    if not (args.device_code and args.start and args.end):
        raise SystemExit("--source ctseelink-file-urls 需要 --device-code/--start/--end")
    headers: dict[str, str] = {}
    if args.auth_token:
        headers["Authorization"] = args.auth_token
    if args.api_key:
        headers["X-API-Key"] = args.api_key
    client = RecordingListClient(args.endpoint, headers=headers)
    # ISO ``T`` avoids fragile shell quoting while the historical space form
    # remains accepted by ``fromisoformat``.
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    if end <= start:
        raise SystemExit("--end 必须晚于 --start")
    window = timedelta(minutes=max(1, args.window_minutes))
    pages = []
    truncation = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + window, end)
        query = ListQuery(
            args.device_code,
            cursor.strftime("%Y-%m-%d %H:%M:%S"),
            window_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        try:
            page = client.query(query)
        except RecordingSourceError as exc:
            pages.append(None)
            truncation.append({
                "query": [query.start_time, query.end_time],
                "error": str(exc)[:160],
            })
            cursor = window_end
            continue
        pages.append(page)
        if args.check_truncation:
            halves = []
            for offset in (0, max(1, args.window_minutes // 2)):
                half_start = cursor + timedelta(minutes=offset)
                if half_start >= window_end:
                    continue
                half_end = min(half_start + window, window_end)
                halves.append(client.query(ListQuery(
                    args.device_code,
                    half_start.strftime("%Y-%m-%d %H:%M:%S"),
                    half_end.strftime("%Y-%m-%d %H:%M:%S"),
                )))
            if halves:
                truncation.append(detect_truncation(
                    page.files(),
                    [item for half in halves for item in half.files()],
                ))
        cursor = window_end

    good_pages = [page for page in pages if page is not None]
    files = deduplicate_files(good_pages, args.device_code)
    return {
        "kind": "ground_litter_recording_index",
        "source": "ctseelink-file-urls",
        "device_code": args.device_code,
        "query_range": [args.start, args.end],
        "windows": len(pages),
        "failed_windows": sum(1 for page in pages if page is None),
        "files": [item.as_dict() for item in files],
        "file_count": len(files),
        "total_declared_bytes": sum(int(item.file_size or 0) for item in files),
        "real_range": [
            min((item.record_start for item in files), default=""),
            max((item.record_end for item in files), default=""),
        ],
        "pagination_fields": sorted({
            key for page in good_pages for key in page.pagination
        }),
        "truncation_checks": truncation,
        "suspected_truncation": any(
            item.get("suspected_truncation") for item in truncation
        ),
        "note": (
            "只索引元数据；临时签名 URL 不入库。远端保留期限与可重拉能力必须由"
            "A1 实际下载试点确认，本索引不能证明。"
        ),
    }


def index_local(args: argparse.Namespace) -> dict[str, Any]:
    if args.input is None:
        raise SystemExit("--source local 需要 --input")
    root = Path(args.input).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")
    rows = []
    for pattern in ("*.ps", "*.mp4", "*.mkv", "*.avi", "*.mov", "*.ts", "*.m4v"):
        for path in sorted(root.rglob(pattern)):
            if not path.is_file():
                continue
            stat = path.stat()
            rows.append({
                "file_name": str(path.relative_to(root)),
                "bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            })
    return {
        "kind": "ground_litter_recording_index",
        "source": "local",
        "root": str(root),
        "files": rows,
        "file_count": len(rows),
        "total_declared_bytes": sum(row["bytes"] for row in rows),
        "note": "用户本地源默认只读；不参与自动清理，也不计算完整 SHA-256（弱索引）。",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = index_remote(args) if args.source == "ctseelink-file-urls" else index_local(args)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    for key in SENSITIVE_KEYS:
        if f'"{key}"' in text.lower():
            raise SystemExit(f"索引包含敏感字段 {key}，已拒绝写出")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "files": payload["file_count"],
        "total_declared_bytes": payload["total_declared_bytes"],
        "suspected_truncation": payload.get("suspected_truncation"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
