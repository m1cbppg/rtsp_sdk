from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import ultralytics
from PIL import Image

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from rtsp_annotator.gas_cylinder import (  # noqa: E402
    GasCylinderCameraProfile,
    UltralyticsYoloeGasCylinderDetector,
)
import rtsp_annotator.deepstream_worker  # noqa: E402,F401


def main() -> None:
    model = Path("/app/models/gas/yoloe-26l-seg.pt")
    profiles = Path("/app/models/gas/profiles")
    profile = GasCylinderCameraProfile.load(profiles, "camera_01_ir")
    started = time.perf_counter()
    detector = UltralyticsYoloeGasCylinderDetector(
        model_path=model,
        profile=profile,
        device="cuda:0",
        half=False,
        imgsz=1280,
    )
    load_ms = (time.perf_counter() - started) * 1000
    reference_rgb = np.asarray(Image.open(profile.reference_image).convert("RGB"))
    started = time.perf_counter()
    detections = detector.detect(reference_rgb)
    inference_ms = (time.perf_counter() - started) * 1000
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0),
                "ultralytics": ultralytics.__version__,
                "gstreamer": Gst.version_string(),
                "profile": profile.profile_id,
                "model_load_ms": round(load_ms, 1),
                "reference_candidates": len(detections),
                "inference_ms": round(inference_ms, 1),
                "max_cuda_memory_mib": round(
                    torch.cuda.max_memory_allocated() / 1024 / 1024,
                    1,
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
