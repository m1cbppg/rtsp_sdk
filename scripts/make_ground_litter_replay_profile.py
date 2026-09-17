"""Create a disposable profile whose reference image is a local replay frame.

This is for replay calibration only. It does not change the source profile and
does not infer which physical camera produced a video file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def first_frame(source: Path):
    import av

    av.logging.set_level(av.logging.PANIC)
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.is_corrupt:
                continue
            return frame.to_ndarray(format="bgr24")
    raise ValueError("Replay has no decodable video frame")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error("Replay video missing")
    payload = json.loads(args.profiles.read_text(encoding="utf-8"))
    cameras = [c for c in payload.get("cameras", []) if c.get("device_code") == args.device]
    if len(cameras) != 1:
        parser.error("Device is not uniquely present in profile")
    frame = first_frame(args.video)
    height, width = frame.shape[:2]
    args.output.mkdir(parents=True, exist_ok=True)
    image = args.output / f"{args.device}-replay-reference.jpg"
    if not cv2.imwrite(str(image), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError("Could not write replay reference")
    camera = cameras[0]
    camera["reference_image"] = str(image)
    camera["reference_size"] = [width, height]
    camera["enabled"] = False
    camera["calibration_status"] = "draft"
    result = args.output / "profiles.json"
    result.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"profiles": str(result), "reference_image": str(image),
                      "device": args.device, "reference_size": [width, height]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
