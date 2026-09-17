#!/usr/bin/env python3
"""Replay litter detector JSON through the independent ground-litter filter.

This is a temporal stability experiment, not an accuracy benchmark: the
screening reports do not contain exhaustive ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.ground_litter import (
    GroundLitterDetection,
    GroundLitterOptions,
    GroundLitterTracker,
)


def replay(report: dict, *, association_distance: float, persistence: float,
           confidence: float, confirm_hits: int, max_motion: float) -> dict:
    roi = tuple(tuple(point) for point in report["roi"])
    output: dict[str, dict[str, int]] = {}
    for segment in ("day", "night"):
        tracker = GroundLitterTracker(
            GroundLitterOptions(
                ground_roi=roi,
                association_distance=association_distance,
                persistence_seconds=persistence,
                minimum_confidence=confidence,
                confirm_hits=confirm_hits,
                max_motion=max_motion,
                # The screening JSON is sampled every ~10 seconds.  Bridge
                # that offline gap; production analysis runs at 1–2 FPS and
                # should use a shorter per-camera value (usually 3 seconds).
                lost_track_seconds=20.0,
            )
        )
        events = 0
        candidates = 0
        for frame in report["frames"]:
            if not frame["image"].startswith(segment):
                continue
            detections = []
            for item in frame["detections"]:
                left, top, right, bottom = item["box"]
                detections.append(
                    GroundLitterDetection(
                        rectangle=NormalizedRect(
                            left / 2560,
                            top / 1440,
                            (right - left) / 2560,
                            (bottom - top) / 1440,
                        ),
                        confidence=float(item["confidence"]),
                        label=str(item["label"]),
                    )
                )
            result = tracker.observe(float(frame["nominal_time_s"]), detections)
            events += len(result.events)
            candidates += len(result.candidates)
        output[segment] = {"events": events, "candidate_observations": candidates}
    output["total"] = {
        "events": output["day"]["events"] + output["night"]["events"],
        "candidate_observations": (
            output["day"]["candidate_observations"]
            + output["night"]["candidate_observations"]
        ),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--association-distance", type=float, default=0.15)
    parser.add_argument("--persistence", type=float, default=30.0)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--confirm-hits", type=int, default=4)
    parser.add_argument("--max-motion", type=float, default=0.035)
    args = parser.parse_args()
    result = {
        "report": str(args.report),
        "parameters": {
            "association_distance": args.association_distance,
            "persistence": args.persistence,
            "confidence": args.confidence,
            "confirm_hits": args.confirm_hits,
            "max_motion": args.max_motion,
        },
        "result": replay(
            json.loads(args.report.read_text(encoding="utf-8")),
            association_distance=args.association_distance,
            persistence=args.persistence,
            confidence=args.confidence,
            confirm_hits=args.confirm_hits,
            max_motion=args.max_motion,
        ),
        "limitation": "筛查报告没有穷尽真值，events不能解释为准确率或误报率",
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
