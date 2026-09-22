"""Real-weight smoke test for the V3.3 dual-channel pipeline.

Proves something unit tests with a fake model cannot: that the actual
``turhancan_yolov8m_seg_trash.pt`` weight runs for BOTH channels and that each
tick issues exactly one batched model call per group rather than one call per
tile or per crop.

No network access is used or needed: the weight is loaded from the repository
path, and the script fails loudly if that file is missing rather than letting
Ultralytics attempt a download.

Usage:
  .venv/bin/python scripts/smoke_ground_litter_v33_real_model.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHT = "models/litter/turhancan_yolov8m_seg_trash.pt"
DEFAULT_FRAME = "output/ground_litter_v33_replay_20260918/positive_span8_event6.jpg"
DEFAULT_OUT = "output/ground_litter_v33_smoke_20260918/smoke.json"


class CountingModel:
    """Proxy that records how many model calls and how large each batch was."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.last_ms: float = 0.0

    def predict(self, images: Any, **kwargs: Any) -> Any:
        batch = len(images) if isinstance(images, list) else 1
        self.calls += 1
        self.batch_sizes.append(batch)
        started = time.perf_counter()
        result = self._inner.predict(images, **kwargs)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_options():
    from rtsp_annotator.ground_litter_detection import (
        GroundLitterDetectionOptions,
    )

    request = json.loads(
        (REPO / "config/ground_litter_v33_hybrid_stream_request.example.json")
        .read_text(encoding="utf-8")
    )
    payload = dict(request["ground_litter"])
    # The smoke run uses the frame's own resolution; zone thresholds stay as the
    # reviewed request sets them.
    options = GroundLitterDetectionOptions.from_payload(payload)
    return options


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default=DEFAULT_WEIGHT)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    weight_path = (REPO / args.weight).resolve()
    frame_path = (REPO / args.frame).resolve()
    report: dict[str, Any] = {
        "weight": str(weight_path),
        "weight_bytes": weight_path.stat().st_size if weight_path.is_file() else 0,
        "weight_present": weight_path.is_file(),
        "frame": str(frame_path),
        "device": args.device,
        "network_used": False,
    }
    if not weight_path.is_file():
        raise SystemExit(
            f"weight not found: {weight_path}; refusing to let Ultralytics download"
        )
    if not frame_path.is_file():
        raise SystemExit(f"frame not found: {frame_path}")

    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise SystemExit(f"frame unreadable: {frame_path}")
    options = build_options()
    report["options"] = {
        "mode": options.mode,
        "analysis_fps": options.analysis_fps,
        "semantic_scan_interval_seconds": options.semantic_scan_interval_seconds,
        "prior_crop_maximum": options.prior_crop_maximum,
        "prior_crop_imgsz": options.prior_crop_imgsz,
        "maximum_tiles": options.maximum_tiles,
        "zones": len(options.zones),
    }

    from rtsp_annotator.ground_litter_detection import (
        UltralyticsGroundLitterDetector,
        build_ground_litter_tiles,
    )

    started = time.perf_counter()
    detector = UltralyticsGroundLitterDetector(
        model_path=weight_path, device=args.device, half=False,
    )
    report["model_load_seconds"] = round(time.perf_counter() - started, 2)
    counter = CountingModel(detector._model)
    detector._model = counter
    report["class_names"] = {int(k): str(v) for k, v in detector.class_names.items()}

    height, width = frame.shape[:2]
    report["frame_size"] = [width, height]

    # ---------------------------- full ROI scan ----------------------------
    masks, tiles = build_ground_litter_tiles(options, width, height)
    report["tile_count"] = len(tiles)
    before = counter.calls
    scan_started = time.perf_counter()
    candidates, stats = detector.tile_candidates_batch(
        frame, options, masks=masks, tiles=tiles, night=False, actors=[],
    )
    report["full_scan_ms"] = round((time.perf_counter() - scan_started) * 1000.0, 1)
    report["full_scan_model_calls"] = counter.calls - before
    report["full_scan_batch_size"] = counter.batch_sizes[-1] if counter.batch_sizes else 0
    report["full_scan_stats"] = {k: int(v) for k, v in stats.items()}
    report["full_scan_candidates"] = len(candidates)
    report["full_scan_classes"] = sorted({c.class_name for c in candidates})

    # --------------------------- prior crop batch --------------------------
    # A small synthetic prior box over a textured patch so the crop path runs
    # regardless of whether the full scan happened to fire.
    boxes = [(1200.0, 700.0, 1260.0, 760.0), (400.0, 900.0, 440.0, 950.0)]
    before = counter.calls
    crop_started = time.perf_counter()
    rows_per_crop, rects, batches = detector.crop_candidates_batch(
        frame, boxes, options, night=False,
    )
    report["crop_batch_ms"] = round((time.perf_counter() - crop_started) * 1000.0, 1)
    report["crop_model_calls"] = counter.calls - before
    report["crop_batch_size"] = counter.batch_sizes[-1] if counter.batch_sizes else 0
    report["crop_rects"] = [list(rect) for rect in rects]
    report["crop_raw_detections"] = sum(len(rows) for rows in rows_per_crop)

    # ------------------------------- verdict -------------------------------
    checks = {
        "weight_loaded_from_disk": report["weight_present"],
        "full_scan_used_one_batched_call": (
            report["full_scan_model_calls"] == 1
            and report["full_scan_batch_size"] == report["tile_count"]
            and report["tile_count"] > 1
        ),
        "crop_batch_used_one_call": (
            report["crop_model_calls"] == 1
            and report["crop_batch_size"] == len(boxes)
        ),
        "model_produced_rows": (
            report["full_scan_stats"].get("raw_candidates", 0) > 0
            or report["crop_raw_detections"] > 0
        ),
        "crop_rects_inside_frame": all(
            0 <= rect[0] and 0 <= rect[1]
            and rect[2] <= width and rect[3] <= height
            for rect in rects
        ),
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {out_path}")
    if not report["passed"]:
        raise SystemExit("real-model smoke FAILED")


if __name__ == "__main__":
    main()
