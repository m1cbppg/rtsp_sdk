#!/usr/bin/env python3
"""Stream selected PS recordings, extract native-resolution frame triplets, delete PS."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_audit_dataset import select_files_evenly
from rtsp_annotator.ground_litter_profile_sampling import SequentialFrameReader
from rtsp_annotator.ground_litter_recording_source import ListQuery, RecordingDownloader, RecordingListClient


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--file-count", type=int, default=12)
    p.add_argument("--samples-per-file", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inventory = json.loads(args.inventory.read_text())
    device = inventory["device_code"]
    selected = select_files_evenly(inventory["files"], args.file_count)
    raw_dir, frame_dir = args.output / "raw", args.output / "frames"
    raw_dir.mkdir(parents=True, exist_ok=True); frame_dir.mkdir(parents=True, exist_ok=True)
    client, downloader = RecordingListClient(), RecordingDownloader(timeout=45, max_attempts=3)
    rows, failures = [], []
    for file_index, item in enumerate(selected):
        start = datetime.strptime(item["record_start"], "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(item["record_end"], "%Y-%m-%d %H:%M:%S")
        query = ListQuery(device, (start-timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S"),
                          (end+timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S"))
        path = raw_dir / f"{item['file_id']}.ps"
        try:
            page = client.query(query)
            entry = next(entry for entry in page.entries if entry.file.file_id == item["file_id"])
            downloader.download(entry.url, path, expected_size=item.get("file_size"))
            duration = max(20.0, (end-start).total_seconds())
            centers = [(i+1)*duration/(args.samples_per_file+1) for i in range(args.samples_per_file)]
            offsets = sorted({max(0.0, min(duration-0.2, center+delta)) for center in centers for delta in (-2.0, 0.0, 2.0)})
            reader = SequentialFrameReader(path)
            frames = reader.sample_with_seek(offsets, tolerance_seconds=1.2, window_seconds=5.0)
            for sample_index, center in enumerate(centers):
                triplet = [min(frames, key=lambda actual: abs(actual-(center+delta))) for delta in (-2.0, 0.0, 2.0)] if frames else []
                if len(set(triplet)) < 3:
                    failures.append({"file_id": item["file_id"], "center": center, "error": "missing_triplet"}); continue
                paths = {}
                for role, actual in zip(("before", "current", "after"), triplet):
                    frame_id = f"f{file_index:02d}s{sample_index:02d}"
                    target = frame_dir / f"{frame_id}_{role}.jpg"
                    cv2.imwrite(str(target), frames[actual].frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    paths[role] = str(target.relative_to(args.output))
                actual_time = start + timedelta(seconds=center)
                rows.append({"frame_id": f"f{file_index:02d}s{sample_index:02d}", "device_code": device,
                             "timestamp": actual_time.isoformat(sep=" "), "file_id": item["file_id"],
                             "offset_seconds": round(center, 3), "paths": paths,
                             "shape": list(frames[triplet[1]].frame.shape[:2])})
        except Exception as exc:
            failures.append({"file_id": item["file_id"], "error": f"{type(exc).__name__}:{str(exc)[:160]}"})
        finally:
            path.unlink(missing_ok=True)
            path.with_name(path.name+".part").unlink(missing_ok=True)
        print(json.dumps({"processed": file_index+1, "selected": len(selected), "frames": len(rows), "failures": len(failures)}), flush=True)
    payload = {"kind": "ground_litter_audit_frames", "device_code": device, "selected_files": selected,
               "frames": rows, "failures": failures, "temporary_ps_remaining": len(list(raw_dir.glob("*.ps")))}
    (args.output / "frames_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if rows else 2


if __name__ == "__main__": raise SystemExit(main())
