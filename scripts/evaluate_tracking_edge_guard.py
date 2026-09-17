"""Policy-only delayed PTZ simulation; not a DeepStream/device acceptance test."""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.tracking_edge_guard import TrackingEdgeGuard
from rtsp_annotator.vessel_detection import VesselDetection


def simulate(*, response: float, video_delay: float, cruise: float,
             burst: float, direction: int) -> dict:
    dt, duration, motion_duration = .05, 60., .2
    zoom, pan, world = 4., 0., 0.
    world_width = .33/zoom
    guard = TrackingEdgeGuard(response_seconds=response,
                              uncertainty_seconds=video_delay+.1)
    frames = deque()
    action = None
    commands = []
    clipped = 0
    count = 0
    max_error = 0.
    for tick in range(round(duration/dt)+1):
        t = tick*dt
        speed = burst if burst and 20 <= t < 22 else cruise
        if tick:
            world += direction*speed*dt
        if action is not None:
            start, old_pan, goal_pan, old_zoom, goal_zoom, delta = action
            fraction = min(1., max(0., (t-start)/motion_duration))
            pan = old_pan + fraction*(goal_pan-old_pan)
            zoom = math.exp(math.log(old_zoom)+fraction*math.log(goal_zoom/old_zoom))
            if fraction >= 1:
                guard.action_completed(t, zoom_delta=delta)
                action = None
        x = .5+(world-pan)*zoom
        width = world_width*zoom
        visible = x-width/2 >= 0 and x+width/2 <= 1
        if tick:
            clipped += int(not visible)
            count += 1
        max_error = max(max_error, abs(x-.5))
        frames.append((t, x, width, visible))
        delivered = None
        while frames and frames[0][0] <= t-video_delay+1e-9:
            delivered = frames.popleft()
        if delivered is None:
            continue
        stamp, observed_x, observed_width, observed_visible = delivered
        if not observed_visible:
            # No synthetic perfect detection after the physical target clips.
            continue
        if action is not None:
            # Match the current synchronous coordinator: frames keep arriving
            # but no new control decision is consumed during a physical action.
            continue
        detection = VesselDetection(
            1, NormalizedRect(observed_x-observed_width/2, .42, observed_width, .16),
            # Match the live adapter's frame-receive monotonic timestamp.
            # Upstream video age is separately budgeted in uncertainty_seconds.
            .9, 8, 3, "detector_measurement", t, "simulated", round(stamp/dt),
        )
        decision = guard.update(detection, now=t)
        if decision.x is None:
            continue
        if decision.zoom_delta == 0 and abs(decision.x-.5) <= .05:
            continue
        delta = decision.zoom_delta
        # One nonreplaceable action. Gain is a simulation assumption, not a
        # mapping of any real camera's relative zoom step to optical ratio.
        goal_pan = pan + (decision.x-.5)/zoom
        goal_zoom = max(1., zoom*1.2**delta)
        action = (t+response, pan, goal_pan, zoom, goal_zoom, delta)
        commands.append({"time": round(t, 3), "delta": delta,
                         "reason": decision.reason})
    shrinks = [c for c in commands if c["delta"] < 0]
    return {
        "response_seconds": response, "video_delay_seconds": video_delay,
        "cruise_home_widths_per_second": cruise,
        "burst_home_widths_per_second": burst, "direction": direction,
        "duration_seconds": duration, "motion_seconds": motion_duration,
        "clipped_seconds": round(clipped*dt, 3),
        "visible_fraction": round(1-clipped/count, 4),
        "maximum_center_error": round(max_error, 4),
        "zoom_out_commands": len(shrinks),
        "zoom_in_commands": sum(c["delta"] > 0 for c in commands),
        "command_count": len(commands), "final_zoom": round(zoom, 3),
        "commands": commands,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for response in (.5, .75, 1.):
        for video in (.1, .3):
            for direction in (-1, 1):
                for label, cruise, burst in (("cruise_slow", .01, 0),
                                              ("cruise_normal", .03, 0),
                                              ("burst", .01, .06),
                                              ("stress_burst", .01, .12)):
                    result = simulate(response=response, video_delay=video,
                                      cruise=cruise, burst=burst, direction=direction)
                    result["scenario"] = label
                    result["quality_pass"] = result["clipped_seconds"] == 0 and (
                        burst != 0 or result["zoom_out_commands"] == 0
                    )
                    cases.append(result)
    report = {
        "scenario_version": "edge-guard-policy-v1",
        "scope": "policy-only; detector stub; no HTTP, SDK, GPU, identity or OSD validation",
        "assumptions": "unbounded pan; start at zoom 4; 1.2x per step; .2s physical movement with .5s guard motion budget; receive timestamps; separate mechanical and video delay",
        "passed": sum(c["quality_pass"] for c in cases), "total": len(cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))
    # Stress is intentionally beyond the nominal speed; normal/burst failures
    # remain failures and are never relabelled after running the matrix.
    return int(any(not c["quality_pass"] for c in cases if c["scenario"] != "stress_burst"))


if __name__ == "__main__":
    raise SystemExit(main())
