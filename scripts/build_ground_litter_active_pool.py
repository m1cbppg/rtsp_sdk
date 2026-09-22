#!/usr/bin/env python3
"""Build a deduplicated semantic pool plus independent random audit proposals."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_audit_dataset import (  # noqa: E402
    Proposal, dedupe_persistent_proposals, expanded_mask, polygon_mask,
    random_grid_proposals,
)


def crop(image, box, scale=1.8):
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(32.0, (x2 - x1) * scale, (y2 - y1) * scale)
    return image[
        max(0, round(cy - side / 2)):min(h, round(cy + side / 2)),
        max(0, round(cx - side / 2)):min(w, round(cx + side / 2)),
    ]


def proposal_id(device: str, frame_id: str, source: str, bbox) -> str:
    raw = f"{device}:{frame_id}:{source}:{','.join(map(str, bbox))}"
    return f"{device[-5:]}-{frame_id}-{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-per-frame", type=int, default=5)
    parser.add_argument(
        "--keep-all-semantic", action="store_true",
        help="keep every per-frame semantic proposal (shadow/performance mode)",
    )
    parser.add_argument(
        "--no-crop-assets", action="store_true",
        help="record source-frame+bbox for in-memory embedding instead of JPEG crops",
    )
    args = parser.parse_args()

    frames = json.loads((args.root / "frames_manifest.json").read_text())
    semantic = json.loads((args.root / "semantic_proposals.json").read_text())
    geometry = json.loads(args.geometry.read_text())
    frame_by_id = {row["frame_id"]: row for row in frames["frames"]}
    semantic_by_id = {row["frame_id"]: row for row in semantic["frames"]}
    device = frames["device_code"]

    proposal_rows: list[Proposal] = []
    metadata: dict[tuple, dict] = {}
    for frame_id, group in semantic_by_id.items():
        for rank, item in enumerate(group["proposals"], 1):
            bbox = tuple(int(value) for value in item["bbox"])
            row = Proposal(frame_id, bbox, item["source"], float(item["score"]))
            proposal_rows.append(row)
            metadata[(frame_id, bbox, item["source"], float(item["score"]))] = {
                "rank": rank, "class_id": item["class_id"],
                "class_name": item["class_name"],
                "raw_kept_count": group.get("raw_kept_count"),
            }
    selected_proposals = (
        proposal_rows if args.keep_all_semantic
        else dedupe_persistent_proposals(proposal_rows)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output / "crops"
    if not args.no_crop_assets:
        crop_dir.mkdir(parents=True, exist_ok=True)
    semantic_items, random_items = [], []

    audit_masks = {}
    for frame_id, frame in frame_by_id.items():
        current = cv2.imread(str(args.root / frame["paths"]["current"]))
        if current is None:
            raise RuntimeError(f"cannot read frame {frame_id}")
        shape = current.shape[:2]
        if shape not in audit_masks:
            core = polygon_mask(shape, geometry["roi"])
            audit_masks[shape] = expanded_mask(core, 0.08)
        audit = audit_masks[shape]
        for proposal in random_grid_proposals(
            frame_id, current, audit, count=args.random_per_frame, seed=20260922,
        ):
            pid = proposal_id(device, frame_id, proposal.source, proposal.bbox)
            random_items.append({
                "proposal_id": pid, "device_code": device,
                "timestamp": frame["timestamp"], "frame_id": frame_id,
                "bbox": list(proposal.bbox), "source": proposal.source,
                "score": round(float(proposal.score), 8),
            })

    cached_frame_id = None
    current = None
    for proposal in selected_proposals:
        frame = frame_by_id[proposal.frame_id]
        if proposal.frame_id != cached_frame_id:
            current = cv2.imread(str(args.root / frame["paths"]["current"]))
            if current is None:
                raise RuntimeError(f"cannot read frame {proposal.frame_id}")
            cached_frame_id = proposal.frame_id
        pid = proposal_id(device, proposal.frame_id, proposal.source, proposal.bbox)
        assert current is not None
        extra = metadata[(
            proposal.frame_id, proposal.bbox, proposal.source, float(proposal.score),
        )]
        output_row = {
            "proposal_id": pid, "device_code": device,
            "timestamp": frame["timestamp"], "frame_id": proposal.frame_id,
            "bbox": list(proposal.bbox), "source": proposal.source,
            "score": round(float(proposal.score), 8),
            "rank": extra["rank"], "class_id": extra["class_id"],
            "class_name": extra["class_name"],
            "raw_kept_count": extra["raw_kept_count"],
        }
        if args.no_crop_assets:
            output_row["frame_image"] = str(args.root / frame["paths"]["current"])
        else:
            patch = crop(current, proposal.bbox)
            if patch.size == 0:
                continue
            path = crop_dir / f"{pid}.jpg"
            cv2.imwrite(str(path), patch, [cv2.IMWRITE_JPEG_QUALITY, 94])
            output_row["image"] = str(path.relative_to(args.output))
        semantic_items.append(output_row)

    payload = {
        "schema": "ground_litter_active_pool_v1", "device_code": device,
        "semantic_deduplicated_across_frames": not args.keep_all_semantic,
        "crop_assets_materialized": not args.no_crop_assets,
        "semantic_count": len(semantic_items), "random_count": len(random_items),
        "semantic": semantic_items, "random": random_items,
    }
    (args.output / "pool.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({
        "device_code": device, "semantic": len(semantic_items),
        "random": len(random_items), "crop_dir": str(crop_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
