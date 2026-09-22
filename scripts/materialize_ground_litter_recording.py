#!/usr/bin/env python3
"""Refresh and download one frozen playback file identity without storing its URL."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_recording_source import (  # noqa: E402
    ListQuery, RecordingDownloader, RecordingListClient,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".part").exists():
        raise SystemExit("output or partial file already exists")
    payload = json.loads(args.manifest.read_text())
    device = payload["device_code"]
    source = next(
        (row for row in payload["selected_files"] if row["file_id"] == args.file_id),
        None,
    )
    if source is None:
        raise SystemExit("file ID is not present in the frozen manifest")
    start = datetime.strptime(source["record_start"], "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(source["record_end"], "%Y-%m-%d %H:%M:%S")
    query = ListQuery(
        device,
        (start - timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S"),
        (end + timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S"),
    )
    page = RecordingListClient().query(query)
    entry = next(
        (row for row in page.entries if row.file.file_id == args.file_id), None,
    )
    if entry is None:
        raise RuntimeError("refreshed playback response omitted the frozen file ID")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    downloader = RecordingDownloader(timeout=45, max_attempts=3)
    downloader.download(entry.url, args.output, expected_size=source.get("file_size"))
    identity = {
        "schema": "ground_litter_materialized_recording_v1",
        "device_code": device,
        "file_id": source["file_id"],
        "file_name": source["file_name"],
        "record_start": source["record_start"],
        "record_end": source["record_end"],
        "declared_bytes": int(source["file_size"]),
        "actual_bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "signed_url_persisted": False,
    }
    sidecar = args.output.with_suffix(args.output.suffix + ".identity.json")
    sidecar.write_text(json.dumps(identity, ensure_ascii=False, indent=2))
    print(json.dumps(identity, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
