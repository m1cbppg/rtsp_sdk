#!/usr/bin/env python3
"""Replay V3 candidates with the hardened V3.2 event lifecycle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ground_litter_event_state_v32 import (
    MAX_CLOSED_EVENTS,
    MAX_SAMPLE_GAP_FACTOR,
    MAX_STATE_HISTORY,
    MAX_VISIBLE_TIMESTAMP_HISTORY,
    EVENT_MERGE_DISTANCE_PX,
    MIN_CLEAN_VALID_FRACTION,
    OnlineEvent,
    OnlineEventMemory,
    anchor_observation,
)
from rtsp_annotator.ground_litter_detection import UltralyticsGroundLitterDetector
from run_ground_litter_clean_temporal_poc import yolo_options
from run_ground_litter_online_replay_v31 import (
    CLEAN_MAX_SUPPORT_PIXELS,
    CLEAR_CONFIRM_SECONDS,
    CONFIRM_VISIBLE_SECONDS,
    CONTEXT_OCCLUSION_EXTERNAL_AREA_PX,
    EVENT_MATCH_DISTANCE_PX,
    EVENT_MATCH_SIZE_RATIO,
    EVENT_MERGE_SIZE_RATIO,
    PENDING_EXPIRE_SECONDS,
    load_json,
    replay_video as replay_video_v31,
)


def replay_video(**kwargs):
    return replay_video_v31(**kwargs, memory_class=OnlineEventMemory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--v1-output", type=Path, required=True)
    parser.add_argument("--v3-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--litter-model", type=Path, required=True)
    parser.add_argument("--actor-model", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--videos",
        nargs="+",
        choices=("negative", "positive", "clean_holdout"),
        default=("negative", "positive", "clean_holdout"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.v1_output / "reference_arrays.npz")
    reference = arrays["reference"]
    reference_valid = cv2.imread(
        str(args.v3_output / "reference_valid_mask_v3.png"), cv2.IMREAD_GRAYSCALE
    )
    if reference_valid is None:
        raise FileNotFoundError(args.v3_output / "reference_valid_mask_v3.png")
    options = yolo_options(args.litter_model, args.actor_model)
    detector = UltralyticsGroundLitterDetector(
        model_path=args.litter_model,
        actor_model_path=args.actor_model,
        device=args.device,
    )
    video_config = {
        "negative": ("正常负样本.mp4", "negative"),
        "positive": ("小垃圾正样本.mp4", "positive"),
        "clean_holdout": ("clean_reference.mp4", "clean_holdout"),
    }
    results = {}
    for name in args.videos:
        filename, directory = video_config[name]
        v3_summary = load_json(args.v3_output / directory / "summary.json")
        results[name] = replay_video(
            name=name,
            video=args.input_dir / filename,
            start=float(v3_summary["evaluation_start_seconds"]),
            v3_directory=args.v3_output / directory,
            output=args.output / directory,
            reference=reference,
            reference_valid=reference_valid,
            detector=detector,
            detector_options=options,
        )
    payload = {
        "kind": "ground_litter_online_event_replay_v32",
        "parameters": {
            "event_match_distance_px": EVENT_MATCH_DISTANCE_PX,
            "event_match_size_ratio": EVENT_MATCH_SIZE_RATIO,
            "event_merge_distance_px": EVENT_MERGE_DISTANCE_PX,
            "event_merge_size_ratio": EVENT_MERGE_SIZE_RATIO,
            "context_external_area_px": CONTEXT_OCCLUSION_EXTERNAL_AREA_PX,
            "clear_confirm_seconds": CLEAR_CONFIRM_SECONDS,
            "clean_max_support_pixels": CLEAN_MAX_SUPPORT_PIXELS,
            "min_clean_valid_fraction": MIN_CLEAN_VALID_FRACTION,
            "max_sample_gap_factor": MAX_SAMPLE_GAP_FACTOR,
            "confirm_visible_seconds": CONFIRM_VISIBLE_SECONDS,
            "pending_expire_seconds": PENDING_EXPIRE_SECONDS,
            "max_closed_events": MAX_CLOSED_EVENTS,
            "actor_model": args.actor_model.name,
            "actor_overlap_threshold": float(options.actor_overlap_threshold),
            "clean_reference_updated": False,
        },
        "videos": results,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
