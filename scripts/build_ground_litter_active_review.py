#!/usr/bin/env python3
"""Render a blinded review dataset from a frozen active-learning selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2


def crop(image, box, scale):
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(32.0, (x2 - x1) * scale, (y2 - y1) * scale)
    return image[
        max(0, round(cy - side / 2)):min(h, round(cy + side / 2)),
        max(0, round(cx - side / 2)):min(w, round(cx + side / 2)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.output / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    frame_maps = {}
    for row in selection["selected"]:
        device = row["device_code"]
        if device not in frame_maps:
            manifest = json.loads((args.root / device / "frames_manifest.json").read_text())
            frame_maps[device] = {item["frame_id"]: item for item in manifest["frames"]}

    items = []
    for row in selection["selected"]:
        device, frame_id = row["device_code"], row["frame_id"]
        frame = frame_maps[device][frame_id]
        triplet = {
            role: cv2.imread(str(args.root / device / frame["paths"][role]))
            for role in ("before", "current", "after")
        }
        if any(image is None for image in triplet.values()):
            raise RuntimeError(f"missing frame triplet for {device}:{frame_id}")
        image = triplet["current"]
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in row["bbox"]]
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        review_id = row["proposal_id"]
        marked = image.copy()
        cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 0, 255), max(2, w // 900))
        paths = {}
        rendered = {
            "context": (crop(marked, (x1, y1, x2, y2), 4.0), 90),
            "crop": (crop(image, (x1, y1, x2, y2), 1.8), 94),
            "before": (crop(triplet["before"], (x1, y1, x2, y2), 4.0), 86),
            "after": (crop(triplet["after"], (x1, y1, x2, y2), 4.0), 86),
        }
        for role, (asset, quality) in rendered.items():
            path = assets / f"{review_id}-{role}.jpg"
            cv2.imwrite(str(path), asset, [cv2.IMWRITE_JPEG_QUALITY, quality])
            paths[role] = str(path.relative_to(args.output))
        items.append({
            "review_id": review_id, "device_code": device,
            "timestamp": row["timestamp"], "frame_id": frame_id,
            "bbox": [x1, y1, x2, y2], "source": row["source"],
            "context_image": paths["context"], "crop_image": paths["crop"],
            "before_image": paths["before"], "after_image": paths["after"],
        })

    items.sort(key=lambda row: hashlib.sha256(
        ("active-review-order-v1:" + row["review_id"]).encode()
    ).hexdigest())
    stable = [{
        "review_id": row["review_id"], "bbox": row["bbox"], "source": row["source"],
    } for row in items]
    fingerprint = hashlib.sha256(json.dumps(
        stable, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    payload = {
        "dataset": args.dataset, "fingerprint": fingerprint,
        "count": len(items),
        "labels": ["LITTER", "NON_LITTER", "BOX_WRONG", "UNCERTAIN"],
        "items": items,
    }
    (args.output / "review-data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({
        "dataset": args.dataset, "count": len(items), "fingerprint": fingerprint,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
