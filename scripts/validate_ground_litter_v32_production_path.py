#!/usr/bin/env python3
"""Replay a video through the packaged Clean Reference V3.2 processor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtsp_annotator.ground_litter_detection import (  # noqa: E402
    GroundLitterDetectionOptions,
    GroundLitterZone,
    UltralyticsGroundLitterDetector,
)
from rtsp_annotator.ground_litter_v32 import (  # noqa: E402
    CleanReferenceProfileV32,
    CleanReferenceV32Processor,
)


# The bottom 6% produced the single reviewed V3 false event (edge/compression
# instability). Keep the first production acceptance area inside y=0.94.
GROUND_ROI = ((0.455, 0.232), (0.714, 0.232), (0.798, 0.94), (0.455, 0.94))
REVIEWED_FIXED_DIFFERENCE = (
    (0.655, 0.415), (0.685, 0.415), (0.685, 0.455), (0.655, 0.455)
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--profile-id", default="camera_01_v32_1080p")
    parser.add_argument("--litter-model", type=Path, required=True)
    parser.add_argument("--actor-model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--analysis-fps", type=float, default=1.0)
    parser.add_argument("--input-width", type=int, default=1920)
    parser.add_argument("--input-height", type=int, default=1080)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    options = GroundLitterDetectionOptions(
        enabled=True,
        mode="clean_reference_v32",
        profile_id=args.profile_id,
        analysis_fps=args.analysis_fps,
        actor_model=args.actor_model.name,
        actor_confidence=0.3,
        actor_imgsz=1280,
        zones=(GroundLitterZone(
            region_id="walkway",
            polygon=GROUND_ROI,
            exclude_zones=(REVIEWED_FIXED_DIFFERENCE,),
            minimum_short_side_px=6,
            minimum_box_area_px=36,
        ),),
        display_zones=True,
        display_class=False,
        display_confidence=False,
    )
    options.validate()
    profile = CleanReferenceProfileV32.load(args.profile_root, args.profile_id)
    processor = CleanReferenceV32Processor(options, profile)
    detector = UltralyticsGroundLitterDetector(
        model_path=args.litter_model,
        actor_model_path=args.actor_model,
        device=args.device,
    )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"无法读取视频: {args.video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        raise ValueError("视频FPS无效")
    next_sample = 0.0
    timeline = []
    started = time.perf_counter()
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = frame_index / source_fps
        frame_index += 1
        if timestamp + 1e-6 < next_sample:
            continue
        next_sample += 1.0 / options.analysis_fps
        if args.input_width > 0 and args.input_height > 0:
            frame = cv2.resize(
                frame,
                (args.input_width, args.input_height),
                interpolation=cv2.INTER_AREA,
            )
        actors = detector.actor_boxes(frame, options)
        snapshot = processor.update(frame, timestamp=timestamp, actors=actors)
        timeline.append({
            "timestamp": round(timestamp, 3),
            "state": snapshot.state,
            "raw_candidates": snapshot.raw_candidates,
            "displayed_event_ids": [item.object_id for item in snapshot.detections],
            "actors": len(actors),
        })
    capture.release()

    events = [{
        "event_id": event.event_id,
        "first_seen": event.first_seen,
        "confirmed_at": event.confirmed_at,
        "closed_at": event.closed_at,
        "closed_reason": event.closed_reason,
        "state": event.state,
        "anchor_box": [round(value, 2) for value in event.anchor_box],
        "visible_observation_count": event.visible_observation_count,
        "state_history": event.state_history,
    } for event in processor.memory.events]
    primary = [
        event for event in events
        if event["confirmed_at"] is not None
        and not str(event["closed_reason"] or "").startswith("merged_into:")
    ]
    cleared = [event for event in primary if event["closed_reason"] == "clean_confirmed"]
    open_events = [event for event in primary if event["closed_at"] is None]
    passed = (
        len(primary) == 2
        and len(cleared) == 1
        and len(open_events) == 1
        and open_events[0]["event_id"] != cleared[0]["event_id"]
        and open_events[0]["first_seen"] > cleared[0]["closed_at"]
        and any(
            row["state"] == "OCCLUDED"
            for row in cleared[0]["state_history"]
        )
    )
    result = {
        "kind": "ground_litter_v32_production_path_validation",
        "passed": passed,
        "profile_id": args.profile_id,
        "input_size": [args.input_width, args.input_height],
        "analysis_fps": args.analysis_fps,
        "alignment": processor.alignment,
        "frames": len(timeline),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "events": events,
        "timeline": timeline,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "passed": passed,
        "frames": len(timeline),
        "elapsed_seconds": result["elapsed_seconds"],
        "primary_events": primary,
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
