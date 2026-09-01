#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rtsp_annotator.config import parse_roi
from rtsp_annotator.vessel_detection import (
    TemporalSmallTargetProposer,
    VesselDetectionOptions,
    VesselTrackManager,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用录像离线评估水面小目标候选数量和稳定轨迹",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--roi", required=True, help="x,y;x,y;...归一化水面区域")
    parser.add_argument("--exclude-roi", action="append", default=[])
    parser.add_argument("--analysis-fps", type=float, default=5.0)
    parser.add_argument("--maximum-seconds", type=float, default=300.0)
    parser.add_argument("--minimum-hits", type=int, default=4)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--disable-appearance", action="store_true")
    parser.add_argument("--appearance-threshold", type=int, default=18)
    parser.add_argument("--appearance-blur", type=int, default=31)
    parser.add_argument("--border-margin", type=float, default=0.01)
    parser.add_argument("--minimum-area", type=int, default=20)
    parser.add_argument("--maximum-area", type=int, default=1_000)
    parser.add_argument("--minimum-width", type=int, default=4)
    parser.add_argument("--minimum-height", type=int, default=3)
    args = parser.parse_args()

    import cv2

    roi = parse_roi(args.roi)
    if roi is None:
        raise ValueError("--roi不能为空")
    exclude_rois = tuple(
        parsed
        for item in args.exclude_roi
        if (parsed := parse_roi(item)) is not None
    )
    options = VesselDetectionOptions(
        enabled=True,
        analysis_fps=args.analysis_fps,
        roi=roi,
        exclude_rois=exclude_rois,
        minimum_hits=args.minimum_hits,
        hold_seconds=1.5,
        small_target_proposals=True,
        proposal_threshold=args.threshold,
        proposal_appearance_enabled=not args.disable_appearance,
        proposal_appearance_threshold=args.appearance_threshold,
        proposal_appearance_blur_pixels=args.appearance_blur,
        proposal_border_margin=args.border_margin,
        proposal_minimum_area_pixels=args.minimum_area,
        proposal_maximum_area_pixels=args.maximum_area,
        proposal_minimum_width_pixels=args.minimum_width,
        proposal_minimum_height_pixels=args.minimum_height,
    )
    options.validate()
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开录像: {args.input}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    sample_every = max(int(round(source_fps / args.analysis_fps)), 1)
    maximum_frames = int(max(args.maximum_seconds, 0.0) * source_fps)
    proposer = TemporalSmallTargetProposer()
    tracker = VesselTrackManager(options)
    raw_counts: list[int] = []
    stable_counts: list[int] = []
    observations: list[dict[str, object]] = []
    frame_number = 0
    while frame_number < maximum_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_number % sample_every == 0:
            timestamp = frame_number / source_fps
            candidates = proposer.detect(frame, options)
            snapshot = tracker.update(
                candidates,
                timestamp=timestamp,
                inference_ms=0,
            )
            raw_counts.append(len(candidates))
            stable_counts.append(snapshot.count)
            if snapshot.detections:
                observations.append(
                    {
                        "timestamp": round(timestamp, 3),
                        "detections": [
                            {
                                "id": item.object_id,
                                "box": [
                                    round(item.rectangle.left, 6),
                                    round(item.rectangle.top, 6),
                                    round(item.rectangle.width, 6),
                                    round(item.rectangle.height, 6),
                                ],
                                "hits": item.hits,
                            }
                            for item in snapshot.detections
                        ],
                    }
                )
        frame_number += 1
    capture.release()
    summary = {
        "input": str(args.input),
        "source_fps": source_fps,
        "analysis_fps": args.analysis_fps,
        "sample_count": len(raw_counts),
        "raw_candidate_median": (
            statistics.median(raw_counts) if raw_counts else 0
        ),
        "raw_candidate_maximum": max(raw_counts, default=0),
        "stable_candidate_median": (
            statistics.median(stable_counts) if stable_counts else 0
        ),
        "stable_candidate_maximum": max(stable_counts, default=0),
        "appearance_enabled": not args.disable_appearance,
        "appearance_threshold": args.appearance_threshold,
        "observations": observations,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
