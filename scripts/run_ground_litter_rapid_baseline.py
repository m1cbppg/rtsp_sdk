#!/usr/bin/env python3
"""Run the frozen Step 2B ``last.pt`` baseline over the Rapid v1 review frames.

The inference chain is deliberately the production-like one and is shared verbatim with the
Phase-2 V2 evaluation:

    source-native 2560x1440 -> ROI -> 640x640 tiles (stride 512)
      -> YOLO -> tile->source coordinates -> keep prediction centre inside ROI
      -> class-wise NMS at IoU 0.50 -> confidence floor 0.01 -> top 100 per frame

Every raw proposal (bbox, confidence, class, tile) is persisted once so the later threshold
sweep never re-runs inference.

    python scripts/run_ground_litter_rapid_baseline.py \
        --artifact ~/ground-litter-rapid-v1/artifact \
        --model <last.pt> --model-sha256 <sha>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtsp_annotator.ground_litter_rapid import (  # noqa: E402
    NMS_IOU,
    PROPOSAL_FLOOR,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    STEP2B_LAST_SHA256,
    THRESHOLD_GRID,
    TILE,
    TOP_K_PER_FRAME,
    RapidError,
    box_iou,
    read_jsonl,
    sha256_file,
    tile_starts,
    verify_split,
    write_json,
)

#: Per-tile detector cap.  Ultralytics defaults to 300, which at conf 0.01 can silently
#: truncate the raw candidate pool before our own NMS and top-100/frame selection.
MAX_DET_PER_TILE = 1000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", default=STEP2B_LAST_SHA256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="baseline/predictions.jsonl")
    parser.add_argument("--manifest-name", default="baseline/inference_manifest.json")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def roi_window_indices(roi, width: int, height: int, mask):
    import numpy as np
    import cv2
    polygon = np.array([(round(float(x) * width), round(float(y) * height))
                        for x, y in roi], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    windows = []
    for y in tile_starts(height):
        for x in tile_starts(width):
            if np.any(mask[y:y + TILE, x:x + TILE]):
                windows.append((x, y))
    return windows


def predict_frame(model, frame, roi, names, *, device: str, batch: int) -> dict:
    import cv2
    import numpy as np
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    windows = roi_window_indices(roi, width, height, mask)
    raw: list[dict] = []
    for base in range(0, len(windows), max(1, batch)):
        part = windows[base:base + max(1, batch)]
        results = model.predict([frame[y:y + TILE, x:x + TILE] for x, y in part],
                                imgsz=TILE, conf=PROPOSAL_FLOOR, iou=0.7,
                                max_det=MAX_DET_PER_TILE, agnostic_nms=False,
                                device=device, verbose=False)
        for (x, y), result in zip(part, results):
            for box in result.boxes:
                local = box.xyxy[0].cpu().tolist()
                coords = [round(local[0] + x, 3), round(local[1] + y, 3),
                          round(local[2] + x, 3), round(local[3] + y, 3)]
                cx = int(round((coords[0] + coords[2]) / 2.0))
                cy = int(round((coords[1] + coords[3]) / 2.0))
                if not (0 <= cx < width and 0 <= cy < height and mask[cy, cx]):
                    continue
                class_id = int(box.cls[0])
                raw.append({"xyxy": coords, "confidence": round(float(box.conf[0]), 6),
                            "class_id": class_id,
                            "class_name": str(names[class_id]) if names else str(class_id),
                            "tile_xy": [int(x), int(y)]})
    kept: list[dict] = []
    for row in sorted(raw, key=lambda r: r["confidence"], reverse=True):
        ok = True
        for old in kept:
            if row["class_id"] != old["class_id"]:
                continue
            if box_iou(row["xyxy"], old["xyxy"]) > NMS_IOU:
                ok = False
                break
        if ok:
            kept.append(row)
    kept = kept[:TOP_K_PER_FRAME]
    return {"tiles": len(windows), "raw_count": len(raw), "boxes": kept}


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = args.artifact.resolve()
    split = json.loads((artifact / "split.json").read_text(encoding="utf-8"))
    verify_split(split)
    extraction = json.loads((artifact / "extraction_manifest.json").read_text(encoding="utf-8"))
    if extraction.get("decode_failure_count") or extraction.get("missing"):
        raise RapidError("extraction is incomplete; refusing to infer")
    records = extraction["records"]
    if args.limit:
        records = records[: args.limit]

    if not args.no_resume:
        actual = sha256_file(args.model)
        if actual != args.model_sha256:
            raise RapidError(f"model SHA mismatch: {actual} != {args.model_sha256}")
    else:
        actual = sha256_file(args.model)

    out_path = artifact / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    done: dict[str, dict] = {}
    if not args.no_resume and partial.exists():
        for row in read_jsonl(partial):
            done[row["frame_id"]] = row

    from ultralytics import YOLO
    import cv2
    model = YOLO(str(args.model))
    names = model.names

    started = time.time()
    results_rows: list[dict] = []
    mode = "a" if done else "w"
    handle = open(partial, mode, encoding="utf-8")
    try:
        for index, record in enumerate(records, 1):
            if record["frame_id"] in done:
                results_rows.append(done[record["frame_id"]])
                continue
            image_path = artifact / "frames" / record["image"]
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RapidError(f"cannot read extracted frame {image_path}")
            height, width = frame.shape[:2]
            if (width, height) != (SOURCE_WIDTH, SOURCE_HEIGHT):
                raise RapidError(f"{record['frame_id']}: {width}x{height} is not source-native")
            row = split_file_row(split, record["file_id"])
            prediction = predict_frame(model, frame, row["roi"], names,
                                       device=args.device, batch=args.batch)
            boxes = [{"prediction_id": f'{record["frame_id"]}_p{position:03d}', **box}
                     for position, box in enumerate(prediction["boxes"])]
            out = {
                "frame_id": record["frame_id"],
                "camera_id": record["camera_id"],
                "file_id": record["file_id"],
                "split": record["split"],
                "kind": record["kind"],
                "offset_seconds": record["offset_seconds"],
                "source_width": width,
                "source_height": height,
                "roi_geometry_version": record["roi_geometry_version"],
                "tiles": prediction["tiles"],
                "raw_candidates": prediction["raw_count"],
                "prediction_count": len(boxes),
                "predictions": boxes,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            results_rows.append(out)
            print(f'[{index}/{len(records)}] {out["frame_id"]} tiles={out["tiles"]} '
                  f'raw={out["raw_candidates"]} kept={out["prediction_count"]}', flush=True)
    finally:
        handle.close()
    partial.replace(out_path)

    counts_by_threshold = {
        f"{threshold:.2f}": sum(
            1 for row in results_rows
            for box in row["predictions"] if box["confidence"] >= threshold)
        for threshold in THRESHOLD_GRID
    }
    manifest = {
        "schema_version": split["schema_version"],
        "split_sha256": split["split_sha256"],
        "model_path": str(args.model),
        "model_sha256": actual,
        "model_sha256_expected": args.model_sha256,
        "device": args.device,
        "imgsz": TILE,
        "stride": 512,
        "proposal_floor": PROPOSAL_FLOOR,
        "nms_iou": NMS_IOU,
        "top_k_per_frame": TOP_K_PER_FRAME,
        "frames": len(results_rows),
        "tiles_total": sum(r["tiles"] for r in results_rows),
        "raw_candidates_total": sum(r["raw_candidates"] for r in results_rows),
        "predictions_total": sum(r["prediction_count"] for r in results_rows),
        "predictions_by_threshold": counts_by_threshold,
        "per_split": {
            split_name: {
                "frames": sum(1 for r in results_rows if r["split"] == split_name),
                "predictions": sum(r["prediction_count"] for r in results_rows
                                   if r["split"] == split_name),
            }
            for split_name in ("rapid_train", "rapid_eval")
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "predictions_file": out_path.name,
        "predictions_sha256": sha256_file(out_path),
        "sealed_accessed": False,
    }
    write_json(artifact / args.manifest_name, manifest)
    print(f"\nframes={manifest['frames']} predictions={manifest['predictions_total']} "
          f"elapsed={manifest['elapsed_seconds']}s")
    print("by threshold:", json.dumps(counts_by_threshold))
    return 0


def split_file_row(split: dict, file_id: str) -> dict:
    for row in split["rows"]:
        if row["file_id"] == file_id:
            return row
    raise RapidError(f"unknown file_id in split: {file_id}")


if __name__ == "__main__":
    raise SystemExit(main())
