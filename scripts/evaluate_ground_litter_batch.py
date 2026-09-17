#!/usr/bin/env python3
"""Run the independent ground-litter detector on finite local replays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_batch import BatchOptions, parse_inputs, run_video
from rtsp_annotator.ground_litter_runtime import LitterModel, validate_profiles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--input", action="append", required=True,
                        help="DEVICE=local.ps; repeat for multiple cameras")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("day", "night"), required=True)
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=10000)
    parser.add_argument("--max-duration-seconds", type=float)
    parser.add_argument("--max-candidate-frames", type=int, default=24)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.profiles.read_text(encoding="utf-8"))
    validate_profiles(payload, allow_draft=args.allow_draft)
    inputs = parse_inputs(args.input)
    cameras = {c["device_code"]: c for c in payload["cameras"]}
    missing = sorted(set(inputs) - set(cameras))
    if missing:
        parser.error("camera missing from profiles: " + ",".join(missing))
    for source in inputs.values():
        if not source.is_file():
            parser.error(f"video missing: {source}")
    import os
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/litter_batch")
    import torch
    torch.set_num_threads(4)
    model = LitterModel(payload["model"]["path"], device=args.device)
    options = BatchOptions(sample_fps=args.sample_fps,
                           max_frames=args.max_frames,
                           max_duration_seconds=args.max_duration_seconds,
                           max_candidate_frames=args.max_candidate_frames,
                           jpeg_quality=args.jpeg_quality)
    reports = []
    for device, source in inputs.items():
        reports.append(run_video(camera=cameras[device], source=source,
                                 output=args.output / device,
                                 model_config=payload["model"], mode=args.mode,
                                 model=model, options=options,
                                 allow_draft=args.allow_draft))
    aggregate = {
        "status": "offline_shadow_batch_not_accuracy",
        "mode": args.mode,
        "cameras": reports,
        "total_sampled_frames": sum(r["sampled_frames"] for r in reports),
        "total_confirmed_records": sum(r["confirmed_records"] for r in reports),
        "total_candidate_review_frames": sum(r["candidate_review_count"] for r in reports),
        "notifications_sent": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
