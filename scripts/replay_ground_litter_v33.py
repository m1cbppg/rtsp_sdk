"""Offline clean-replay harness for the V3.3 dual-channel pipeline.

Drives the *real* hybrid tick function from the side process
(``_analyse_hybrid``) over a recorded video, so the measured false-positive
behaviour comes from the shipped code path rather than a re-implementation.

Reports per evidence source, as the spec requires:
  semantic-only / prior-only / fused displayed events per hour, the merged
  total, single-frame flashes, and how many candidates each filter rejected.

Usage:
  .venv/bin/python scripts/replay_ground_litter_v33.py \
      --video "/Users/mlcbppg/Desktop/9月17日/正常负样本.mp4" \
      --label clean_negative --max-seconds 300
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

REPO = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHT = "models/litter/turhancan_yolov8m_seg_trash.pt"
DEFAULT_OUT = "output/ground_litter_v33_replay_20260918"
REQUEST = "config/ground_litter_v33_hybrid_stream_request.example.json"

SOURCE_LABELS = {
    "hybrid_v33_semantic": "semantic_only",
    "hybrid_v33_prior": "prior_only",
    "hybrid_v33_fused": "fused",
}


def load_options(**overrides):
    """Reviewed example request, optionally narrowed for a stricter run."""
    from dataclasses import replace

    from rtsp_annotator.ground_litter_detection import (
        GroundLitterDetectionOptions,
    )

    payload = json.loads((REPO / REQUEST).read_text(encoding="utf-8"))
    options = GroundLitterDetectionOptions.from_payload(
        dict(payload["ground_litter"])
    )
    if not overrides:
        return options
    zones = options.zones
    if "minimum_short_side_px" in overrides or "minimum_box_area_px" in overrides:
        zones = tuple(
            replace(
                zone,
                minimum_short_side_px=overrides.get(
                    "minimum_short_side_px", zone.minimum_short_side_px
                ),
                minimum_box_area_px=overrides.get(
                    "minimum_box_area_px", zone.minimum_box_area_px
                ),
            )
            for zone in zones
        )
    option_values = {
        "zones": zones,
        "confidence": overrides.get("confidence", options.confidence),
    }
    for name in (
        "semantic_confirm_hits",
        "semantic_hit_window",
        "semantic_confirm_span_seconds",
    ):
        if name in overrides:
            option_values[name] = overrides[name]
    return replace(options, **option_values)


def replay(
    *, video: Path, label: str, weight: Path, profile_root: Path,
    device: str, max_seconds: float | None, option_overrides: dict | None = None,
) -> dict[str, Any]:
    """Run the real hybrid tick over sampled frames and return raw records."""
    from rtsp_annotator.ground_litter_detection import (
        UltralyticsGroundLitterDetector,
    )
    from rtsp_annotator.ground_litter_process import (
        _HybridPadState,
        _analyse_hybrid,
    )
    from rtsp_annotator.ground_litter_v32 import CleanReferenceProfileV32

    options = load_options(**(option_overrides or {}))
    if not options.profile_id:
        raise SystemExit("example request has no profile_id")
    profile = CleanReferenceProfileV32.load(
        profile_root, options.profile_id
    )
    detector = UltralyticsGroundLitterDetector(
        model_path=weight, device=device, half=False,
    )
    state = _HybridPadState(options, profile)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    native_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    analysis_period = 1.0 / float(options.analysis_fps)
    step = max(1, int(round(native_fps * analysis_period)))

    records: list[dict[str, Any]] = []
    index = 0
    timestamp = 0.0
    started = time.perf_counter()
    while True:
        if not capture.grab():
            break
        if index % step == 0:
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                break
            if max_seconds is not None and timestamp > float(max_seconds):
                break
            tick_started = time.perf_counter()
            snapshot = _analyse_hybrid(
                state=state, detector=detector, bgr=frame,
                timestamp=timestamp, night=False, actor_boxes=[],
            )
            records.append({
                "timestamp": round(timestamp, 2),
                "state": snapshot.state,
                "branch_state": snapshot.branch_state,
                "branch_message": snapshot.branch_message,
                "environment_state": snapshot.environment_state,
                "prior_available": snapshot.prior_environment_state == "NORMAL"
                and snapshot.state != "prior_abstaining",
                "displayed": [
                    {
                        "event_id": detection.object_id,
                        "source": SOURCE_LABELS.get(
                            detection.source, detection.source
                        ),
                        "confidence": round(float(detection.confidence), 4),
                        "class_name": detection.class_name,
                        "region_id": detection.region_id,
                        "rectangle": {
                            "left": detection.rectangle.left,
                            "top": detection.rectangle.top,
                            "width": detection.rectangle.width,
                            "height": detection.rectangle.height,
                        },
                    }
                    for detection in snapshot.detections
                ],
                "active_events": snapshot.active_events,
                "confirmed_events": snapshot.confirmed_events,
                "cleared_events": snapshot.cleared_events,
                "semantic_only_active": snapshot.semantic_only_active,
                "prior_only_active": snapshot.prior_only_active,
                "fused_active": snapshot.fused_active,
                "cross_source_merges": snapshot.cross_source_merges,
                "semantic_raw": snapshot.semantic_raw_candidates,
                "semantic_retained": snapshot.semantic_retained_candidates,
                "prior_raw": snapshot.prior_raw_candidates,
                "prior_retained": snapshot.prior_retained_candidates,
                "semantic_crop_raw": snapshot.semantic_crop_raw_candidates,
                "semantic_crop_unmatched": (
                    snapshot.semantic_crop_unmatched_candidates
                ),
                "model_runs_full": snapshot.semantic_model_runs_full,
                "model_runs_crop": snapshot.semantic_model_runs_crop,
                "prior_ms": snapshot.last_prior_ms,
                "full_scan_ms": snapshot.last_full_scan_ms,
                "crop_ms": snapshot.last_crop_batch_ms,
                "total_ms": snapshot.last_total_ms,
                "wall_ms": round((time.perf_counter() - tick_started) * 1000.0, 1),
            })
            timestamp += analysis_period
        index += 1
    capture.release()

    return {
        "label": label,
        "confidence": options.confidence,
        "zone_minimum_short_side_px": (
            options.zones[0].minimum_short_side_px if options.zones else None
        ),
        "zone_minimum_box_area_px": (
            options.zones[0].minimum_box_area_px if options.zones else None
        ),
        "video": str(video),
        "native_fps": native_fps,
        "total_frames": total_frames,
        "analysis_fps": options.analysis_fps,
        "startup_suppress_seconds": options.startup_suppress_seconds,
        "weight": str(weight),
        "device": device,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "ticks": len(records),
        "records": records,
    }


def summarise(run: dict[str, Any]) -> dict[str, Any]:
    records = run["records"]
    if not records:
        return {"label": run["label"], "error": "no ticks"}
    warmup = float(run["startup_suppress_seconds"])
    observed = [
        record for record in records if record["timestamp"] >= warmup
    ]
    if not observed:
        return {"label": run["label"], "error": "no post-warmup ticks"}
    span = max(observed[-1]["timestamp"] - warmup, 1e-6)
    hours = span / 3600.0

    # A displayed "event" is one event_id; a flash is an event_id seen in
    # exactly one tick.
    per_source: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for record in observed:
        for item in record["displayed"]:
            per_source[item["source"]][item["event_id"]] += 1

    def rate(events: dict[int, int]) -> float:
        return round(len(events) / hours, 3) if hours > 0 else 0.0

    all_ids: set[int] = set()
    flashes: dict[str, int] = {}
    for source, events in per_source.items():
        flashes[source] = sum(1 for count in events.values() if count == 1)
        all_ids |= set(events)

    merged_flash_ids = set()
    display_counts: dict[int, int] = defaultdict(int)
    for record in observed:
        for item in record["displayed"]:
            display_counts[item["event_id"]] += 1
    merged_flash_ids = {k for k, v in display_counts.items() if v == 1}

    totals = {
        "ticks": len(observed),
        "span_seconds": round(span, 1),
        "semantic_only_displayed_events": len(per_source.get("semantic_only", {})),
        "prior_only_displayed_events": len(per_source.get("prior_only", {})),
        "fused_displayed_events": len(per_source.get("fused", {})),
        "merged_displayed_events": len(all_ids),
        "semantic_only_events_per_hour": rate(per_source.get("semantic_only", {})),
        "prior_only_events_per_hour": rate(per_source.get("prior_only", {})),
        "fused_events_per_hour": rate(per_source.get("fused", {})),
        "merged_events_per_hour": round(len(all_ids) / hours, 3) if hours else 0.0,
        "single_frame_flashes": len(merged_flash_ids),
        "flashes_by_source": flashes,
        "prior_candidates_proposed": sum(r["prior_raw"] for r in observed),
        "prior_candidates_retained": sum(r["prior_retained"] for r in observed),
        "prior_rejected_by_geometry": sum(
            max(0, r["prior_raw"] - r["prior_retained"]) for r in observed
        ),
        "semantic_raw_candidates": sum(r["semantic_raw"] for r in observed),
        "semantic_retained_candidates": sum(
            r["semantic_retained"] for r in observed
        ),
        "semantic_crop_raw": sum(r["semantic_crop_raw"] for r in observed),
        "semantic_crop_unmatched_rejected": sum(
            r["semantic_crop_unmatched"] for r in observed
        ),
        "environment_not_normal_ticks": sum(
            1 for r in observed if r["environment_state"] != "NORMAL"
        ),
        "occluded_ticks": sum(1 for r in observed if r["state"] == "OCCLUDED"),
        "degraded_ticks": sum(
            1 for r in observed if r["branch_state"] != "ok"
        ),
        "model_runs_full": observed[-1]["model_runs_full"],
        "model_runs_crop": observed[-1]["model_runs_crop"],
        "cross_source_merges": observed[-1]["cross_source_merges"],
    }
    return {"label": run["label"], "totals": totals}


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    def pick(q: float) -> float:
        index = min(len(ordered) - 1, int(len(ordered) * q))
        return round(ordered[index], 1)
    return {"p50": pick(0.50), "p95": pick(0.95), "max": round(ordered[-1], 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--weight", default=DEFAULT_WEIGHT)
    parser.add_argument(
        "--profile-root", default="models/litter/profiles"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--confidence", type=float, default=None,
                        help="override the semantic channel confidence")
    parser.add_argument("--minimum-short-side-px", type=int, default=None)
    parser.add_argument("--minimum-box-area-px", type=int, default=None)
    parser.add_argument("--semantic-confirm-hits", type=int, default=None)
    parser.add_argument("--semantic-hit-window", type=int, default=None)
    parser.add_argument("--semantic-confirm-span-seconds", type=float, default=None)
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.confidence is not None:
        overrides["confidence"] = args.confidence
    if args.minimum_short_side_px is not None:
        overrides["minimum_short_side_px"] = args.minimum_short_side_px
    if args.minimum_box_area_px is not None:
        overrides["minimum_box_area_px"] = args.minimum_box_area_px
    if args.semantic_confirm_hits is not None:
        overrides["semantic_confirm_hits"] = args.semantic_confirm_hits
    if args.semantic_hit_window is not None:
        overrides["semantic_hit_window"] = args.semantic_hit_window
    if args.semantic_confirm_span_seconds is not None:
        overrides["semantic_confirm_span_seconds"] = (
            args.semantic_confirm_span_seconds
        )

    run = replay(
        video=Path(args.video),
        label=args.label,
        weight=(REPO / args.weight).resolve(),
        profile_root=(REPO / args.profile_root).resolve(),
        device=args.device,
        max_seconds=args.max_seconds,
        option_overrides=overrides,
    )
    summary = summarise(run)
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if "totals" in summary:
        observed = [
            record for record in run["records"]
            if record["timestamp"] >= run["startup_suppress_seconds"]
        ]
        stage = {
            "prior_ms": percentiles([r["prior_ms"] for r in observed]),
            "full_scan_ms": percentiles(
                [r["full_scan_ms"] for r in observed if r["full_scan_ms"] > 0]
            ),
            "crop_ms": percentiles(
                [r["crop_ms"] for r in observed if r["crop_ms"] > 0]
            ),
            "total_ms": percentiles([r["total_ms"] for r in observed]),
            "wall_ms": percentiles([r["wall_ms"] for r in observed]),
        }
        summary["stage_ms"] = stage
        (out_dir / f"{args.label}.stage.json").write_text(
            json.dumps(stage, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (out_dir / f"{args.label}.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"{args.label}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {out_dir}/{args.label}.json")


if __name__ == "__main__":
    main()
