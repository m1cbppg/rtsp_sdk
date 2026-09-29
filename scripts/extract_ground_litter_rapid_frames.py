#!/usr/bin/env python3
"""Extract the fixed Ground Litter Rapid v1 review frames from source-native PS files.

Why sequential and not ``-ss``: these PS files are HEVC and a random seek is *not*
frame-accurate.  Measured on the 01021 Development PS, seeking to 30.000 s returned frame
751 (true PTS 30.040 s) while reporting a bogus ``POS_MSEC`` of 30026 — a full frame late and
an untrustworthy delta.  This extractor therefore decodes each PS from frame 0 and captures
exactly ``round(offset_seconds * fps)``, so ``delta_ms`` is 0 by construction and the frame
index follows the same convention the official Discovery pass used.

Frames are written losslessly at the source 2560x1440 canvas — never resized, never taken
from a Proxy B screenshot.  One worker per PS, bounded parallelism.

    python scripts/extract_ground_litter_rapid_frames.py \
        --artifact ~/ground-litter-rapid-v1/artifact \
        --development-root <development split root> --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtsp_annotator.ground_litter_rapid import (  # noqa: E402
    FPS,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    RapidError,
    assert_development_asset,
    read_jsonl,
    sha256_file,
    verify_split,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="debug: only N frames")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-format", default="png", choices=("png", "jpg"))
    parser.add_argument("--jpg-quality", type=int, default=97)
    return parser.parse_args(argv)


def _prepare_cv2():
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
    os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
    import cv2
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:                                             # pragma: no cover
        pass
    return cv2


def extract_one_ps(payload: dict) -> dict:
    """Sequentially decode one PS and capture every requested target frame index."""
    cv2 = _prepare_cv2()
    file_id = payload["file_id"]
    source = payload["source"]
    out_dir = Path(payload["out_dir"])
    extension = payload["extension"]
    quality = int(payload["jpg_quality"])
    resume = payload["resume"]
    targets = sorted(payload["targets"], key=lambda t: t["target_index"])
    wanted = {t["target_index"]: t for t in targets}
    max_index = max(wanted)
    records: list[dict] = []
    failures: list[dict] = []
    anomalies: list[dict] = []

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        return {"file_id": file_id, "records": [],
                "failures": [{"file_id": file_id, "error": f"cannot decode {source}"}],
                "anomalies": [], "source_frame_count": 0, "fps": 0.0}
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            target = wanted[index]
            path = out_dir / f'{target["frame_id"]}.{extension}'
            meta_path = out_dir / f'{target["frame_id"]}.json'
            if resume and path.exists() and meta_path.exists():
                records.append(json.loads(meta_path.read_text(encoding="utf-8")))
            else:
                height, width = frame.shape[:2]
                if (width, height) != (SOURCE_WIDTH, SOURCE_HEIGHT):
                    failures.append({"frame_id": target["frame_id"], "file_id": file_id,
                                     "error": f"decoded {width}x{height}, expected "
                                              f"{SOURCE_WIDTH}x{SOURCE_HEIGHT}"})
                else:
                    if extension == "png":
                        cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    else:
                        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    record = {
                        "frame_id": target["frame_id"],
                        "camera_id": target["camera_id"],
                        "file_id": file_id,
                        "split": target["split"],
                        "kind": target["kind"],
                        "offset_seconds": target["offset_seconds"],
                        "requested_relative_seconds": target["requested_relative_seconds"],
                        "decoded_relative_seconds": round(index / FPS, 4),
                        "frame_index": index,
                        "nominal_frame_index": target["nominal_frame_index"],
                        "delta_ms": round(
                            (index / FPS - target["requested_relative_seconds"]) * 1000.0, 3),
                        "decode_mode": "sequential_frame_index",
                        "width": width,
                        "height": height,
                        "image": path.name,
                        "image_bytes": path.stat().st_size,
                        "image_sha256": sha256_file(path),
                        "source_sha256": target["source_sha256"],
                        "roi": target["roi"],
                        "roi_geometry_version": target["roi_geometry_version"],
                        "canvas_size": [SOURCE_WIDTH, SOURCE_HEIGHT],
                        "is_bonus": target["kind"] == "bonus_train",
                    }
                    if record["kind"] == "bonus_train":
                        record["official_decoded_timestamp"] = \
                            target.get("official_decoded_timestamp")
                    if index != target["nominal_frame_index"]:
                        anomalies.append({
                            "frame_id": record["frame_id"],
                            "nominal_frame_index": target["nominal_frame_index"],
                            "frame_index": index, "reason": "frame_index_offset"})
                    if abs(record["delta_ms"]) >= 1000.0 / FPS:
                        anomalies.append({"frame_id": record["frame_id"],
                                          "delta_ms": record["delta_ms"],
                                          "reason": "delta_exceeds_one_frame"})
                    write_json(meta_path, record)
                    records.append(record)
                    print(f'{record["frame_id"]} f={index} delta_ms={record["delta_ms"]} '
                          f'{record["image_bytes"] // 1024}KiB', flush=True)
        if index >= max_index:
            break
        index += 1
    source_frames = index + 1
    capture.release()
    got = {r["frame_id"] for r in records}
    for target in targets:
        if target["frame_id"] not in got:
            failures.append({"frame_id": target["frame_id"], "file_id": file_id,
                             "error": f'target frame index {target["target_index"]} beyond '
                                      f"end of stream ({source_frames} frames)"})
    return {"file_id": file_id, "records": records, "failures": failures,
            "anomalies": anomalies, "source_frame_count": source_frames, "fps": fps}


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = args.artifact.resolve()
    dev_root = args.development_root.resolve()
    split = json.loads((artifact / "split.json").read_text(encoding="utf-8"))
    verify_split(split)
    frames = read_jsonl(artifact / "frame_manifest.jsonl")
    if not frames:
        raise RapidError("frame_manifest.jsonl is empty")
    by_file = {row["file_id"]: row for row in split["rows"]}
    if args.limit:
        frames = frames[: args.limit]

    out_dir = artifact / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    extension = "png" if args.save_format == "png" else "jpg"

    grouped: dict[str, list[dict]] = {}
    for row in frames:
        grouped.setdefault(row["file_id"], []).append(row)

    payloads = []
    for file_id, targets in grouped.items():
        row = by_file[file_id]
        source = Path(row["ps_path"]).resolve()
        assert_development_asset(source)
        if dev_root not in source.parents:
            raise RapidError(f"source outside Development root: {source}")
        if not source.exists():
            raise RapidError(f"missing source PS: {source}")
        payloads.append({
            "file_id": file_id,
            "source": str(source),
            "out_dir": str(out_dir),
            "extension": extension,
            "jpg_quality": args.jpg_quality,
            "resume": not args.no_resume,
            "targets": [{
                "frame_id": t["frame_id"],
                "camera_id": t["camera_id"],
                "split": t["split"],
                "kind": t["kind"],
                "offset_seconds": t["offset_seconds"],
                "requested_relative_seconds": t["requested_relative_seconds"],
                "nominal_frame_index": t["nominal_frame_index"],
                "target_index": int(t["nominal_frame_index"]),
                "source_sha256": row["source_sha256"],
                "roi": row["roi"],
                "roi_geometry_version": row["roi_geometry_version"],
                "official_decoded_timestamp": (t.get("requested_relative_seconds")
                                               if t["kind"] == "bonus_train" else None),
            } for t in targets],
        })

    started = time.time()
    records: list[dict] = []
    failures: list[dict] = []
    anomalies: list[dict] = []
    source_frames: dict[str, int] = {}
    workers = max(1, min(args.workers, len(payloads)))
    print(f"extracting {len(frames)} frames from {len(payloads)} PS "
          f"(sequential decode, {workers} workers, {extension})", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_one_ps, payload): payload["file_id"]
                   for payload in payloads}
        for future in as_completed(futures):
            result = future.result()
            records.extend(result["records"])
            failures.extend(result["failures"])
            anomalies.extend(result["anomalies"])
            source_frames[result["file_id"]] = result["source_frame_count"]
            print(f'== done {result["file_id"]} '
                  f'({len(result["records"])}/{len(grouped[result["file_id"]])} frames, '
                  f'source has {result["source_frame_count"]} frames, '
                  f'{result["fps"]:.3f} fps)', flush=True)

    order = {row["frame_id"]: position for position, row in enumerate(frames)}
    records.sort(key=lambda r: order.get(r["frame_id"], 1 << 30))
    found = {row["frame_id"] for row in records}
    missing = [row["frame_id"] for row in frames if row["frame_id"] not in found]
    manifest = {
        "schema_version": split["schema_version"],
        "split_sha256": split["split_sha256"],
        "decode_mode": "sequential_frame_index",
        "decode_note": "each PS is decoded from frame 0; targets are exact frame indices "
                       "round(offset*fps) at 25 fps, so delta_ms is 0 by construction",
        "frames_requested": len(frames),
        "frames_extracted": len(records),
        "decode_failures": failures,
        "decode_failure_count": len(failures),
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "missing": missing,
        "missing_count": len(missing),
        "save_format": args.save_format,
        "fps": FPS,
        "max_abs_delta_ms": max((abs(r["delta_ms"]) for r in records), default=None),
        "source_frame_counts": source_frames,
        "elapsed_seconds": round(time.time() - started, 2),
        "workers": workers,
        "counts": {
            "fixed_train": sum(r["kind"] == "fixed" and r["split"] == "rapid_train"
                               for r in records),
            "fixed_eval": sum(r["kind"] == "fixed" and r["split"] == "rapid_eval"
                              for r in records),
            "bonus_train": sum(r["kind"] == "bonus_train" for r in records),
        },
        "records": records,
        "sealed_accessed": False,
    }
    write_json(artifact / "extraction_manifest.json", manifest)
    print(f"\nextracted={len(records)} failures={len(failures)} anomalies={len(anomalies)} "
          f"missing={len(missing)} max_abs_delta_ms={manifest['max_abs_delta_ms']} "
          f"elapsed={manifest['elapsed_seconds']}s")
    if failures or missing:
        print(f"HARD FAIL: {len(failures)} decode failure(s), {len(missing)} missing frame(s)")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
