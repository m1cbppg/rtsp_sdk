#!/usr/bin/env python3
"""Rapid Dataset v2 coarse pass: bounded download -> sequential frames -> two models.

For every selected 5-minute window this script:

1. refreshes the signed replay URL for the frozen ``source_file_id``;
2. downloads the PS into a bounded cache (never a full 7-day pull);
3. sequentially decodes the source-native frames at ~30 s / 150 s / 270 s
   (or a 15 s dense grid for windows that earned it) and writes 2560x1440 PNGs;
4. runs Turhancan and the Step 2B ``last.pt`` with production-scale tiling and
   records raw per-model boxes plus merged candidate observations;
5. releases the PS so peak disk stays bounded, and updates the window manifest.

    python scripts/run_ground_litter_historical_coarse.py \
        --artifact /home/sf01/ground-litter-historical-v2/artifact \
        --selection day1 --device cpu
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_historical import (  # noqa: E402
    COARSE_OFFSETS, DENSE_OFFSETS, FPS, HistoricalError, SOURCE_HEIGHT,
    SOURCE_WIDTH, day1_selection, read_jsonl, sha256_file,
    write_json, write_jsonl,
)
from rtsp_annotator.ground_litter_production_scale import (  # noqa: E402
    merge_cross_model_candidates, predict_frame, roi_mask,
)

ROI_TEMPLATE = "ground_litter_{camera}_final_roi.json"
DEFAULT_LAST_MODEL = "/home/sf01/step2b-20260923/out/runs/full_finetune/weights/last.pt"
SEMANTIC_CANDIDATES = (
    "/home/sf01/step2c1-blind-truth/exploratory-fourcam-20260929/turhancan_yolov8m_seg_trash.pt",
    "/home/sf01/ground_litter_train/models/turhancan_yolov8m_seg_trash.pt",
    "/home/sf01/ground-litter-pilot-20260911/ground_litter_shadow_20260913/models/litter/"
    "turhancan_yolov8m_seg_trash.pt",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--roi-config-dir", type=Path,
                        default=ROOT / "output/ground_litter_final_roi_20260922/config")
    parser.add_argument("--selection", default="day1",
                        choices=("day1", "train", "dev", "final", "all", "list"))
    parser.add_argument("--window-id", action="append", default=[])
    parser.add_argument("--camera", action="append", default=[],
                        help="restrict to these cameras (parallel workers)")
    parser.add_argument("--tag", default="",
                        help="suffix for coarse_frames/raw_candidates/candidate_observations; "
                             "keeps parallel workers from interleaving their JSONL output")
    parser.add_argument("--dense-window", action="append", default=[])
    parser.add_argument("--last-model", type=Path, default=Path(DEFAULT_LAST_MODEL))
    parser.add_argument("--semantic-model", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-download", action="store_true",
                        help="assume the PS is already at <artifact>/raw/<file_id>.ps")
    parser.add_argument("--keep-ps", action="store_true")
    parser.add_argument("--skip-inference", action="store_true",
                        help="cache frames only (used for the untouched DEV/FINAL sets)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-format", default="png", choices=("png",))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--download-timeout", type=float, default=60.0)
    return parser.parse_args(argv)


def camera_roi(args, camera: str) -> dict:
    path = args.roi_config_dir / ROI_TEMPLATE.format(camera=camera)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("camera_id") != camera:
        raise HistoricalError(f"ROI config camera mismatch: {path}")
    if list(payload.get("canvas_size", [])) != [SOURCE_WIDTH, SOURCE_HEIGHT]:
        raise HistoricalError(f"ROI canvas mismatch: {path}")
    return payload


def resolve_semantic(path: Path | None) -> Path:
    if path is not None:
        if not path.is_file():
            raise HistoricalError(f"semantic model not found: {path}")
        return path
    for candidate in SEMANTIC_CANDIDATES:
        if Path(candidate).is_file():
            return Path(candidate)
    raise HistoricalError("Turhancan semantic model not found in the known search paths")


def prepare_cv2():
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
    os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
    import cv2
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:  # pragma: no cover - OpenCV build without the python logging shim
        pass
    return cv2


def decode_target_frames(path: Path, offsets: list[float], *, timeout: float = 30.0,
                         allow_short: bool = True):
    """Sequentially decode to the requested offsets.

    Returns ``(frames, fps, targets, missing)``.  Some PS files are shorter than
    270 s (recorder restarts, truncated archive segments); rather than failing the
    whole window this records the missing offsets so the manifest shows a real
    shortfall.  ``allow_short=False`` restores the hard error.

    OpenCV is used deliberately: these HEVC MPEG-PS files are not frame-accurate
    under random seek (measured one frame late with a bogus POS_MSEC), and the
    production server's train venv has cv2 but not PyAV.  Reading from frame 0
    makes ``round(offset * fps)`` exact by construction.
    """
    cv2 = prepare_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise HistoricalError(f"cannot open source: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or FPS
        if abs(fps - FPS) > 0.1:
            raise HistoricalError(f"unexpected source fps {fps} for {path}")
        targets = {int(round(offset * fps)): offset for offset in offsets}
        wanted = set(targets)
        highest = max(wanted)
        frames: dict[int, Any] = {}
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = frame
                wanted.discard(index)
            if index >= highest:
                break
            index += 1
        missing = sorted(wanted)
        if missing and not allow_short:
            raise HistoricalError(f"source ended before frames {missing}: {path}")
    finally:
        capture.release()
    return frames, fps, targets, missing


def appearance_of(frame, box) -> tuple[list[float], str]:
    import numpy as np
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], max(x2, x1 + 1)), min(frame.shape[0], max(y2, y1 + 1))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return [0.0, 0.0, 0.0], "B0T0"
    mean = crop.reshape(-1, 3).mean(axis=0)
    # background context: a 3x expanded box, for brightness / texture bucketing
    pad_x, pad_y = (x2 - x1), (y2 - y1)
    bx1, by1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    bx2, by2 = min(frame.shape[1], x2 + pad_x), min(frame.shape[0], y2 + pad_y)
    back = frame[by1:by2, bx1:bx2]
    gray = back.mean(axis=2) if back.size else np.zeros((1, 1))
    brightness = 1 if float(gray.mean()) >= 110.0 else 0
    texture = 1 if float(gray.std()) >= 18.0 else 0
    return ([float(v) for v in mean][::-1], f"B{brightness}T{texture}")


def process_window(args, window: dict, models: dict, roi: dict, client, downloader) -> dict:
    device_code = window["device_code"]
    target = args.artifact / "raw" / f"{window['source_file_id']}.ps"
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.is_file() and not args.skip_download:
        entry = None
        record_start = window["start_time"]
        for shift in (0, -1, 1):
            hour = int(record_start[11:13]) + shift
            if not 0 <= hour <= 23:
                continue
            start = f"{record_start[:11]}{hour:02d}:00:00"
            end = f"{record_start[:11]}{hour + 1:02d}:00:00" if hour < 23 else f"{record_start[:11]}23:59:59"
            from rtsp_annotator.ground_litter_recording_source import ListQuery
            page = client.query(ListQuery(device_code, start, end))
            for candidate in page.entries:
                if candidate.file.file_id == window["source_file_id"] and candidate.usable():
                    entry = candidate
                    break
            if entry is not None:
                break
        if entry is None:
            raise HistoricalError(
                f"replay listing has no usable URL for {window['window_id']} "
                f"({window['source_file_id']})")
        downloader.download(entry.url, target, expected_size=window.get("source_file_size") or None)

    source_sha = sha256_file(target)
    offsets = list(DENSE_OFFSETS) if window["window_id"] in set(args.dense_window) \
        else list(COARSE_OFFSETS)
    started = time.monotonic()
    frames, fps, targets, missing = decode_target_frames(target, offsets, timeout=args.timeout)
    decode_seconds = time.monotonic() - started
    if not frames:
        raise HistoricalError(f"no frame decoded from {target}")
    missing_offsets = [float(targets[index]) for index in missing]
    if missing_offsets:
        print(f"    short source: missing offsets {missing_offsets} in {target.name}", flush=True)

    import cv2
    frame_rows, raw_rows, observation_rows = [], [], []
    mask = roi_mask(roi["roi"], SOURCE_WIDTH, SOURCE_HEIGHT)
    stamp_base = datetime.strptime(window["start_time"], "%Y-%m-%d %H:%M:%S").timestamp()
    for index in sorted(frames):
        frame = frames[index]
        if frame.shape[1] != SOURCE_WIDTH or frame.shape[0] != SOURCE_HEIGHT:
            raise HistoricalError(f"unexpected frame size {frame.shape} for {window['window_id']}")
        offset = float(targets[index])
        frame_id = f"{window['window_id']}_t{int(round(offset)):03d}"
        image_rel = f"frames/{window['window_id']}/t{int(round(offset)):03d}.png"
        image_path = args.artifact / image_rel
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise HistoricalError(f"failed to write {image_path}")
        frame_rows.append({
            "frame_id": frame_id,
            "camera_id": window["camera_id"],
            "window_id": window["window_id"],
            "date": window["date"],
            "day_split": window["day_split"],
            "selection_bucket": window["selection_bucket"],
            "offset_seconds": offset,
            "frame_index": int(index),
            "width": SOURCE_WIDTH,
            "height": SOURCE_HEIGHT,
            "record_start": window["start_time"],
            "record_end": window["end_time"],
            "image": image_rel,
            "image_sha256": sha256_file(image_path),
            "roi": roi["roi"],
            "roi_geometry_version": roi["geometry_version"],
            "source_file_id": window["source_file_id"],
            "source_sha256": source_sha,
            "sampling": "dense" if len(offsets) > len(COARSE_OFFSETS) else "coarse",
        })

        per_model = {}
        if args.skip_inference:
            del frame
            continue
        for name, model in models.items():
            result = predict_frame(model, frame, mask, roi["roi"], device=args.device)
            per_model[name] = result
            for box in result["boxes"]:
                raw_rows.append({
                    "frame_id": frame_id,
                    "camera_id": window["camera_id"],
                    "source": name,
                    "bbox_xyxy": box["xyxy"],
                    "confidence": box["confidence"],
                    "class_id": box["class_id"],
                    "class_name": box["class_name"],
                    "tile_xy": box["tile_xy"],
                    "tile_truncated": box["tile_truncated"],
                })
        merged_input = []
        for name, result in per_model.items():
            for box in result["boxes"]:
                merged_input.append({**box, "source": name})
        merged = merge_cross_model_candidates(merged_input)
        for order, observation in enumerate(merged):
            mean_rgb, background = appearance_of(frame, observation["bbox_xyxy"])
            observation_rows.append({
                "observation_id": f"{frame_id}_o{order:02d}",
                "frame_id": frame_id,
                "camera_id": window["camera_id"],
                "window_id": window["window_id"],
                "date": window["date"],
                "day_split": window["day_split"],
                "selection_bucket": window["selection_bucket"],
                "offset_seconds": offset,
                "timestamp": stamp_base + offset,
                "bbox_xyxy": observation["bbox_xyxy"],
                "bbox_by_source": observation.get("bbox_by_source", {}),
                "sampling": "dense" if len(offsets) > len(COARSE_OFFSETS) else "coarse",
                "confidence_by_source": observation["confidence_by_source"],
                "class_name_by_source": observation["class_name_by_source"],
                "tile_truncated": any(bool(box.get("tile_truncated"))
                                      for box in merged_input
                                      if box["xyxy"] == observation["bbox_xyxy"]),
                "appearance": mean_rgb,
                "background_key": background,
            })
        del frame

    if not args.keep_ps:
        target.unlink(missing_ok=True)
    return {
        "frames": frame_rows,
        "raw": raw_rows,
        "observations": observation_rows,
        "source_sha256": source_sha,
        "decode_seconds": round(decode_seconds, 2),
        "fps": fps,
        "offsets": offsets,
        "missing_offsets": missing_offsets,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    args.artifact.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.artifact / "historical_window_manifest.jsonl"
    rows = read_jsonl(manifest_path)
    if not rows:
        raise HistoricalError(f"manifest not found or empty: {manifest_path}")

    if args.window_id:
        wanted = set(args.window_id)
    elif args.selection == "day1":
        wanted = set(day1_selection(rows))
    elif args.selection == "list":
        for row in rows:
            print(f"{row['window_id']}  {row['camera_id']} {row['start_time']}  "
                  f"{row['day_split']}/{row['selection_bucket']}")
        return 0
    elif args.selection == "all":
        wanted = {row["window_id"] for row in rows}
    else:
        wanted = {row["window_id"] for row in rows if row["purpose"] == args.selection}
    selected = [row for row in rows if row["window_id"] in wanted]
    if args.camera:
        allowed = set(args.camera)
        selected = [row for row in selected if row["camera_id"] in allowed]
    selected.sort(key=lambda row: (row["date"], row["camera_id"], row["start_time"]))
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise HistoricalError("no windows selected")

    from rtsp_annotator.ground_litter_recording_source import (
        RecordingDownloader, RecordingListClient,
    )

    if args.skip_inference:
        models: dict = {}
        print("frames-only mode: model inference disabled", flush=True)
    else:
        from ultralytics import YOLO
        semantic = resolve_semantic(args.semantic_model)
        if not args.last_model.is_file():
            raise HistoricalError(f"Step 2B last.pt not found: {args.last_model}")
        print(f"semantic: {semantic}", flush=True)
        print(f"baseline : {args.last_model}", flush=True)
        models = {"turhancan": YOLO(str(semantic)), "yolo": YOLO(str(args.last_model))}
    client = RecordingListClient(timeout=args.timeout)
    downloader = RecordingDownloader(timeout=args.download_timeout)

    by_id = {row["window_id"]: row for row in rows}
    totals = {"windows_done": 0, "frames": 0, "raw": 0, "observations": 0, "bytes": 0}
    started = time.monotonic()
    for order, window in enumerate(selected, start=1):
        row = by_id[window["window_id"]]
        if not args.no_resume and row.get("extraction_status") == "done":
            print(f"[{order}/{len(selected)}] {window['window_id']} already done", flush=True)
            continue
        roi = camera_roi(args, window["camera_id"])
        try:
            result = process_window(args, window, models, roi, client, downloader)
        except Exception as exc:  # keep the batch going; record the failure
            row["extraction_status"] = "failed"
            row["download_status"] = row.get("download_status", "pending")
            write_jsonl(manifest_path, rows)
            print(f"[{order}/{len(selected)}] {window['window_id']} FAILED: {exc}", flush=True)
            continue
        row["download_status"] = "downloaded"
        row["local_path"] = None if not args.keep_ps else str(
            args.artifact / "raw" / f"{window['source_file_id']}.ps")
        row["source_sha256"] = result["source_sha256"]
        row["reason_downloaded"] = f"coarse:{args.selection}"
        row["extraction_status"] = "done"
        row["frame_count"] = len(result["frames"])
        row["missing_offsets"] = result["missing_offsets"]
        suffix = f".{args.tag}" if args.tag else ""
        _append(args.artifact / f"coarse_frames{suffix}.jsonl", result["frames"])
        _append(args.artifact / f"raw_candidates{suffix}.jsonl", result["raw"])
        _append(args.artifact / f"candidate_observations{suffix}.jsonl", result["observations"])
        write_jsonl(manifest_path, rows)
        totals["windows_done"] += 1
        totals["frames"] += len(result["frames"])
        totals["raw"] += len(result["raw"])
        totals["observations"] += len(result["observations"])
        if result["missing_offsets"]:
            totals["short_windows"] = totals.get("short_windows", 0) + 1
        print(f"[{order}/{len(selected)}] {window['window_id']} "
              f"frames={len(result['frames'])} raw={len(result['raw'])} "
              f"obs={len(result['observations'])} decode={result['decode_seconds']}s",
              flush=True)
        write_json(args.artifact / "coarse_progress.json", {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "selected": len(selected),
            **totals,
            "elapsed_seconds": round(time.monotonic() - started, 1),
        })

    print("done: " + json.dumps(totals, ensure_ascii=False), flush=True)
    return 0


def _append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
