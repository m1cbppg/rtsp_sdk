from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from rtsp_annotator.gas_cylinder import (
    GasCylinderCameraProfile,
    UltralyticsYoloeGasCylinderDetector,
    build_temporal_consensus,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    profile = GasCylinderCameraProfile.load(
        Path("/app/models/gas/profiles"),
        "camera_01_ir",
    )
    detector = UltralyticsYoloeGasCylinderDetector(
        model_path=Path("/app/models/gas/yoloe-26l-seg.pt"),
        profile=profile,
        device="cuda:0",
        half=False,
        imgsz=1280,
    )
    samples = []
    timings = []
    if args.video.is_dir():
        image_paths = sorted(args.video.glob("frame_*.jpg"))[:11]
        if len(image_paths) != 11:
            raise RuntimeError("frame directory must contain 11 JPEG files")
        frames = [cv2.imread(str(path)) for path in image_paths]
        duration = 11.0
    else:
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {args.video}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        frames = []
        try:
            for offset in range(11):
                capture.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"cannot read sample {offset}")
                frames.append(bgr)
        finally:
            capture.release()
    for offset, bgr in enumerate(frames):
        if bgr is None:
            raise RuntimeError(f"cannot decode sample {offset}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        samples.append(detector.detect(rgb))
        timings.append((time.perf_counter() - started) * 1000)
    consensus_by_support = {
        minimum: build_temporal_consensus(
            samples,
            minimum_confirmations=minimum,
            match_iou=0.30,
            nms_iou=0.50,
        )
        for minimum in range(2, 7)
    }
    stable = consensus_by_support[3]
    print(
        json.dumps(
            {
                "duration_seconds": round(duration, 2),
                "sample_candidate_counts": [len(item) for item in samples],
                "stable_count": len(stable),
                "stable_counts_by_minimum_support": {
                    minimum: len(items)
                    for minimum, items in consensus_by_support.items()
                },
                "minimum_support": min(
                    (item.support for item in stable),
                    default=0,
                ),
                "mean_inference_ms": round(sum(timings) / len(timings), 1),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
