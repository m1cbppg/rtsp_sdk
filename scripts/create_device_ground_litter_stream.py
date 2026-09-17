#!/usr/bin/env python3
"""Resolve a short-lived device RTSP URL and create a litter stream immediately."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

SOURCE_ENDPOINT = "https://qyzcapi.dgjx0769.com/ims-mainte-pc/p-api/v1/monitor/play/ctseelink/devices/rtsp"


def resolve(device_code: str) -> str:
    req = urllib.request.Request(
        SOURCE_ENDPOINT,
        data=json.dumps({"deviceCode": device_code}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        payload = json.load(response)
    url = (payload.get("data") or {}).get("url", "")
    if payload.get("code") != 200 or not url.startswith(("rtsp://", "rtsps://")):
        raise RuntimeError("设备接口未返回有效 RTSP")
    return url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("device_code")
    parser.add_argument("--api", default="http://127.0.0.1:38080")
    parser.add_argument("--api-key", default=os.environ.get("RTSP_API_KEY"))
    args = parser.parse_args()
    if not args.api_key:
        parser.error("需要 --api-key 或 RTSP_API_KEY")
    source = resolve(args.device_code)
    config = {
        "input_url": source,
        "model": "yolo26s.pt",
        "classes": [0],
        "conf": 0.35,
        "bitrate": "2000k",
        "display_detections": False,
        "ground_litter": {
            "enabled": True,
            "model": "turhancan_yolov8m_seg_trash.pt",
            "actor_model": "yolo26s.pt",
            "analysis_fps": 2.0,
            "confidence": 0.24,
            "night_confidence": 0.18,
            "tile_size_px": 384,
            "inference_imgsz": 768,
            "tile_overlap": 0.25,
            "nms_iou": 0.5,
            "maximum_tiles": 64,
            "minimum_hits": 3,
            "hit_window": 5,
            "hold_seconds": 3.0,
            "maximum_boxes": 6,
            "local_actor_max_crops": 3,
            "box_smoothing_alpha": 0.5,
            "actor_imgsz": 1280,
            "actor_confidence": 0.30,
            "actor_class_ids": [0, 1, 2, 3, 5, 7],
            "context_class_ids": [13, 25, 56, 58, 60],
            "actor_overlap_threshold": 0.15,
            "zones": [
                {"region_id": "near_sidewalk_1", "polygon": [[0.50, 0.45], [0.82, 0.45], [0.95, 0.72], [0.42, 0.72]], "minimum_short_side_px": 10, "minimum_box_area_px": 100, "confidence": 0.24, "night_confidence": 0.18},
                {"region_id": "near_sidewalk_2", "polygon": [[0.42, 0.72], [0.95, 0.72], [0.98, 0.98], [0.32, 0.98]], "minimum_short_side_px": 12, "minimum_box_area_px": 160, "confidence": 0.30, "night_confidence": 0.24},
            ],
            "overlay_exclude_zones": [[[0.0, 0.035], [0.4, 0.035], [0.4, 0.105], [0.0, 0.105]], [[0.72, 0.882], [0.855, 0.882], [0.855, 0.938], [0.72, 0.938]]],
            "display_zones": True,
            "display_class": True,
            "display_confidence": True,
            "label": "疑似垃圾",
        },
    }
    req = urllib.request.Request(
        args.api.rstrip("/") + "/v1/streams",
        data=json.dumps(config, ensure_ascii=False).encode(),
        headers={"X-API-Key": args.api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.load(response)
    # The response contains the authenticated output URL by design; the source
    # URL is never printed or persisted.
    print(json.dumps({k: result.get(k) for k in ("stream_id", "status", "rtsp_url")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
