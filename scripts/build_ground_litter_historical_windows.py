#!/usr/bin/env python3
"""Build the frozen historical window manifest for Rapid Dataset v2 (spec §2-§4).

Reads the replay listing endpoint only (metadata), one query per camera / day /
daylight hour, and selects the 92 windows deterministically.  Signed URLs are
never logged, stored or returned; the cache keeps stable file identity only.

    python scripts/build_ground_litter_historical_windows.py \
        --output output/ground_litter_historical_v2/historical_window_manifest.jsonl \
        --cache  output/ground_litter_historical_v2/listing_cache

The output manifest records the frozen split and time use.  Download state is
appended later by ``run_ground_litter_historical_coarse.py``; ``manifest_sha256``
covers only the frozen fields so a re-download cannot move the split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_historical import (  # noqa: E402
    CAMERAS, DAY_SPLIT, DAYTIME_END_HOUR, DAYTIME_START_HOUR, DEFAULT_SEED,
    DEVICE_CODE_PREFIX, EXPECTED_WINDOW_TOTAL, FINAL_DAYS, SCHEMA_VERSION,
    HistoricalError, RecordingSlot, manifest_sha256, select_windows, write_json,
    write_jsonl,
)
from rtsp_annotator.ground_litter_recording_source import (  # noqa: E402
    ListQuery, RecordingListClient, RecordingSourceError,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--now", default=None,
                        help="ISO camera-local override for a reproducible D7 boundary")
    parser.add_argument("--final-day-margin-minutes", type=float, default=10.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=15.0)
    return parser.parse_args(argv)


def auth_headers(args) -> dict[str, str]:
    headers: dict[str, str] = {}
    token = args.auth_token or os.environ.get("GROUND_LITTER_PLAYBACK_TOKEN")
    if token:
        headers["Authorization"] = token
    api_key = args.api_key or os.environ.get("GROUND_LITTER_PLAYBACK_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def hour_windows(date: str, *, last_hour: int) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    for hour in range(DAYTIME_START_HOUR, last_hour):
        start = f"{date} {hour:02d}:00:00"
        end = f"{date} {(hour + 1) % 24:02d}:00:00" if hour < 23 else f"{date} 23:59:59"
        windows.append((start, end))
    return windows


def load_cached(cache_path: Path, *, refresh: bool) -> dict | None:
    if refresh or not cache_path.is_file():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def query_day(client: RecordingListClient, cache_dir: Path, camera: str, date: str, *,
              last_hour: int, refresh: bool) -> tuple[list[RecordingSlot], int]:
    device_code = DEVICE_CODE_PREFIX + camera
    slots: dict[str, RecordingSlot] = {}
    requests = 0
    for start, end in hour_windows(date, last_hour=last_hour):
        cache_path = cache_dir / f"{camera}_{date.replace('-', '')}_{start[11:13]}.json"
        cached = load_cached(cache_path, refresh=refresh)
        if cached is None:
            page = client.query(ListQuery(device_code, start, end))
            requests += 1
            cached = {
                "camera_id": camera,
                "date": date,
                "query_start": start,
                "query_end": end,
                "files": [entry.file.as_dict() for entry in page.entries],
            }
            write_json(cache_path, cached)
        for row in cached.get("files", []):
            slot = RecordingSlot(
                file_id=row["file_id"],
                record_start=row["record_start"],
                record_end=row["record_end"],
                file_size=row.get("file_size"),
                file_name=row.get("file_name", ""),
            )
            if not (DAYTIME_START_HOUR <= slot.start_hour < DAYTIME_END_HOUR):
                continue
            slots.setdefault(slot.file_id, slot)
    return sorted(slots.values(), key=lambda s: (s.record_start, s.file_id)), requests


def main(argv=None) -> int:
    args = parse_args(argv)
    now = (datetime.fromisoformat(args.now) if args.now else datetime.now())
    args.cache.mkdir(parents=True, exist_ok=True)
    headers = auth_headers(args)
    client = RecordingListClient(timeout=args.timeout, headers=headers)
    if headers:
        print(f"auth headers: {sorted(headers)} (values not printed)", flush=True)

    final_day = FINAL_DAYS[0]
    final_margin = timedelta(minutes=args.final_day_margin_minutes)
    boundary = now - final_margin
    day7_usable_end = None
    if now.strftime("%Y-%m-%d") == final_day:
        if boundary.hour < DAYTIME_START_HOUR:
            raise HistoricalError(
                "final day has no fully elapsed daylight window yet; re-run later")
        day7_usable_end = boundary.strftime("%H:%M:%S")

    slots_by_camera_day: dict[tuple[str, str], list[RecordingSlot]] = {}
    total_requests = 0
    for camera in CAMERAS:
        for date, split in DAY_SPLIT.items():
            last_hour = DAYTIME_END_HOUR
            if date == final_day and day7_usable_end:
                last_hour = min(DAYTIME_END_HOUR, now.hour + 1)
            slots, requests = query_day(client, args.cache, camera, date,
                                        last_hour=last_hour, refresh=args.refresh)
            total_requests += requests
            slots_by_camera_day[(camera, date)] = slots
            print(f"{camera} {date} {split}: {len(slots)} daytime PS "
                  f"({requests} live queries)", flush=True)

    rows = select_windows(slots_by_camera_day, seed=args.seed, day7_usable_end=day7_usable_end)
    if len(rows) != EXPECTED_WINDOW_TOTAL:
        raise HistoricalError(f"expected {EXPECTED_WINDOW_TOTAL} windows, got {len(rows)}")

    write_jsonl(args.output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["day_split"]] = counts.get(row["day_split"], 0) + 1
    meta = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "day_split": DAY_SPLIT,
        "daytime_hours": [DAYTIME_START_HOUR, DAYTIME_END_HOUR],
        "final_day_usable_end": day7_usable_end,
        "live_query_count": total_requests,
        "window_total": len(rows),
        "counts_by_split": counts,
        "manifest_sha256": manifest_sha256(rows),
        "note": "split and time use are frozen here; download state is appended later",
    }
    write_json(args.output.with_suffix(".meta.json"), meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecordingSourceError as exc:
        print(f"listing failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
