#!/usr/bin/env python3
"""Offline mask/ground diagnostics. No network, training, or production changes.

Use unannotated source images to evaluate candidates. Annotated output images
are allowed only with --annotated-input and are explicitly non-accuracy evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtsp_annotator.ground_litter_detection import (
    GroundLitterDetectionOptions, build_ground_litter_tiles,
)


def support_fraction(instance: np.ndarray, ground: np.ndarray) -> float | None:
    """Unknown is distinct from zero; mask and ground must share native pixels."""
    if instance.ndim != 2 or instance.shape != ground.shape:
        raise ValueError("Instance and ground masks must share a 2-D native shape")
    if not np.isfinite(instance).all() or not np.isfinite(ground).all():
        raise ValueError("Non-finite mask")
    foreground = instance > 0.5
    area = int(foreground.sum())
    return float(np.count_nonzero(foreground & (ground != 0)) / area) if area else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--images", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--night", action="store_true")
    parser.add_argument("--annotated-input", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("Local model file is required; no automatic downloads")
    if args.output.exists():
        parser.error("Use a new output directory to preserve earlier evidence")
    payload = json.loads(args.request.read_text())
    options = GroundLitterDetectionOptions.from_payload(payload["ground_litter"])
    options.validate()
    args.output.mkdir(parents=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output / "ultralytics"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model = YOLO(str(args.model))
    report = {
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "options": options.to_payload(), "night": args.night,
        "annotated_input": args.annotated_input,
        "scope": "Geometry diagnostics only; no ground truth or accuracy claim",
        "frames": [],
    }
    threshold = min(z.confidence_for(args.night, options.confidence_for(args.night))
                    for z in options.zones)
    for number, path in enumerate(args.images):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Unreadable input image: {path.name}")
        height, width = frame.shape[:2]
        masks, tiles = build_ground_litter_tiles(options, width, height)
        union = np.maximum.reduce(list(masks.values()))
        rows = []
        started = time.monotonic()
        annotated = frame.copy()
        for tile_id, (x, y, right, bottom) in enumerate(tiles):
            crop = frame[y:bottom, x:right]
            result = model.predict(crop, imgsz=options.effective_imgsz,
                                   conf=threshold, device="cpu", verbose=False,
                                   retina_masks=True, max_det=100)[0]
            if result.boxes is None:
                continue
            segments = None if result.masks is None else result.masks.data.cpu().numpy()
            for index, row in enumerate(result.boxes.data.cpu().tolist()):
                a, b, c, d, score, label = row[:6]
                box = [max(0, round(a+x)), max(0, round(b+y)),
                       min(width, round(c+x)), min(height, round(d+y))]
                cx, cy = min(width-1, (box[0]+box[2])//2), min(height-1, (box[1]+box[3])//2)
                region_ids = [z.region_id for z in options.zones if masks[z.region_id][cy, cx]]
                fraction = None
                if segments is not None:
                    fraction = support_fraction(segments[index], union[y:bottom, x:right])
                rows.append({"box": box, "confidence": score,
                             "class_name": result.names[int(label)],
                             "tile_id": tile_id, "center_regions": region_ids,
                             "ground_mask_fraction": fraction})
                if region_ids:
                    color = (0, 200, 0) if fraction is not None and fraction >= .5 else (0, 0, 255)
                    cv2.rectangle(annotated, tuple(box[:2]), tuple(box[2:]), color, 2)
                    value = "unknown" if fraction is None else f"{fraction:.2f}"
                    cv2.putText(annotated, f"{len(rows)-1} g={value}",
                                (box[0], max(16, box[1]-3)), cv2.FONT_HERSHEY_SIMPLEX,
                                .45, color, 1)
        elapsed = time.monotonic() - started
        for z in options.zones:
            pts = np.rint(np.array(z.polygon)*[width,height]).astype(np.int32)
            cv2.polylines(annotated, [pts], True, (0,255,255), 1)
        cv2.imwrite(str(args.output/f"frame_{number:03d}.jpg"), annotated)
        report["frames"].append({"image_name": path.name,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "size": [width,height], "tiles": len(tiles),
                                 "seconds": elapsed, "candidates": rows})
        print(json.dumps({"image": path.name, "candidates": len(rows),
                          "center_in_ground": sum(bool(r["center_regions"]) for r in rows),
                          "seconds": round(elapsed,2)}, ensure_ascii=False), flush=True)
    (args.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
