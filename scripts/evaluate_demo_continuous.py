"""Run the deterministic discovery-to-close-up demo closed-loop matrix.

This is a detector-stub and PTZ physics test. It is deliberately separate
from DeepStream, real image accuracy, and camera acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtsp_annotator.continuous_tracking import ClosedLoopCase, simulate_closed_loop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for response in (0.5, 0.75, 1.0):
        for video in (0.1, 0.3, 0.7):
            for direction in (-1, 1):
                cases.append(simulate_closed_loop(ClosedLoopCase(
                    name=(f"normal_r{response:g}_v{video:g}_{direction:+d}"
                          if video == 0.1 else
                          f"pressure_video_delay_r{response:g}_v{video:g}_{direction:+d}"),
                    response_seconds=response,
                    video_delay_seconds=video,
                    motion_seconds=0.2,
                    speed=0.001,
                    direction=direction,
                )))
    # Predeclared boundary cases: they must retain invariants, but are not
    # silently relabelled as normal if the device model cannot keep up.
    for response in (0.5, 0.75, 1.0):
        cases.append(simulate_closed_loop(ClosedLoopCase(
            name=f"pressure_acceleration_r{response:g}",
            response_seconds=response, video_delay_seconds=0.7,
            motion_seconds=1.0, speed=0.01, acceleration=0.04,
            initial_world_x=0.68,
        )))
    normal = [case for case in cases if case["scenario"].startswith("normal_")]
    report = {
        "scenario_version": "demo-continuous-closed-loop-v1",
        "scope": "detector stub + nonreplaceable PTZ physics; no HTTP/SDK/GPU/OSD/device",
        "normal_passed": sum(int(c["quality_pass"]) for c in normal),
        "normal_total": len(normal),
        "pressure_passed": sum(int(c["quality_pass"]) for c in cases if c not in normal),
        "pressure_total": len(cases) - len(normal),
        "assumptions": {
            "response_seconds": [0.5, 0.75, 1.0],
            "video_delay_seconds": [0.1, 0.3, 0.7],
            "motion_seconds": [0.2, 1.0],
            "target_world_width": 0.1,
            "zoom_gain_per_step": 1.25,
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False))
    return 0 if report["normal_passed"] == report["normal_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
