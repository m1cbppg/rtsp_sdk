#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


WATER_ROI = [
    [0.000, 0.580],
    [0.120, 0.575],
    [0.250, 0.555],
    [0.380, 0.530],
    [0.460, 0.495],
    [0.550, 0.490],
    [0.620, 0.475],
    [0.660, 0.460],
    [0.720, 0.460],
    [0.760, 0.480],
    [0.860, 0.485],
    [0.930, 0.470],
    [1.000, 0.460],
    [1.000, 1.000],
    [0.000, 1.000],
]

EXCLUDE_ROIS = [
    [[0.015, 0.900], [0.240, 0.900], [0.240, 0.990], [0.015, 0.990]],
    [[0.955, 0.770], [1.000, 0.750], [1.000, 1.000], [0.955, 1.000]],
]


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"无效环境变量行: {raw}")
        values[key] = value
    return values


def request_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict | None = None,
) -> dict | list:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url,
        data=data,
        method=method,
        headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(f"API返回HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise SystemExit(f"无法连接隔离测试API: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="注册隔离PTZ船舶识别流")
    parser.add_argument("--env-file", type=Path, default=Path(".env.ptz-test"))
    parser.add_argument("--api-url", default="http://127.0.0.1:18081")
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--small-target-proposals", action="store_true")
    parser.add_argument("--show-proposals", action="store_true")
    parser.add_argument(
        "--continuous-tracking",
        action="store_true",
        help="近景确认和首张截图后持续跟踪同一艘船",
    )
    parser.add_argument("--wait-seconds", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = load_env(args.env_file)
    required = [
        "PTZ_TEST_API_KEY",
        "PTZ_TEST_CONTROL_KEY",
        "PTZ_TEST_RTSP_READ_USER",
        "PTZ_TEST_RTSP_READ_PASSWORD",
    ]
    missing = [name for name in required if not env.get(name)]
    if missing:
        raise SystemExit("缺少配置: " + ", ".join(missing))

    existing = request_json("GET", f"{args.api_url}/v1/streams", env["PTZ_TEST_API_KEY"])
    if existing:
        raise SystemExit("隔离API已有流；请先停止旧流，避免测试结果混在一起")

    read_user = quote(env["PTZ_TEST_RTSP_READ_USER"], safe="")
    read_password = quote(env["PTZ_TEST_RTSP_READ_PASSWORD"], safe="")
    payload = {
        "input_url": (
            f"rtsp://{read_user}:{read_password}"
            "@ptz-test-mediamtx:8554/detected/virtual-input"
        ),
        "model": args.model,
        "classes": [8],
        "conf": 0.10,
        "imgsz": 640,
        "bitrate": "4000k",
        "vessel_detection": {
            "enabled": True,
            "analysis_fps": 5,
            "confidence": 0.10,
            "imgsz": 1280,
            "class_ids": [8],
            "input_width": 1920,
            "input_height": 1080,
            "roi": WATER_ROI,
            "exclude_rois": EXCLUDE_ROIS,
            "minimum_hits": 2,
            "hold_seconds": 1.0,
            "display_proposals": args.show_proposals,
            "small_target_proposals": args.small_target_proposals,
            "proposal_threshold": 60,
            "proposal_appearance_enabled": True,
            "proposal_appearance_threshold": 18,
            "proposal_minimum_motion_ratio": 0.10,
            "proposal_maximum_candidates": 4,
        },
        "ptz_verification": {
            "enabled": True,
            "camera_id": "virtual-river-01",
            "camera_control_url": "http://ptz-test-camera-control:8080",
            "camera_control_key_env": "CAMERA_CONTROL_API_KEY",
            "zoom_strategy": "adaptive",
            "adaptive_target_width_ratio": 0.33,
            "adaptive_target_height_ratio": 0.33,
            "adaptive_min_step": 1,
            # Keep the demo conservative: several small optical moves are
            # easier to associate across buffered RTSP frames than one large
            # jump that can crop out a moving vessel.
            "adaptive_max_step": 2,
            "adaptive_max_rounds": 6,
            "adaptive_max_total_zoom_delta": 12,
            "confirmed_target_fallback_zoom_rounds": 0,
            "adaptive_min_scale_growth_ratio": 1.08,
            "continuous_tracking": args.continuous_tracking,
            "tracking_center_deadband": 0.05,
            "tracking_command_interval_seconds": 0.2,
            "tracking_settle_seconds": 0.1,
            "tracking_recovery_enabled": True,
            "tracking_recovery_interval_seconds": 2.0,
            "tracking_recovery_zoom_out_step": 1,
            "tracking_recovery_max_attempts": 3,
            "tracking_lost_timeout_seconds": 30,
            "tracking_max_duration_seconds": 0,
            "tracking_zoom_hysteresis_ratio": 0.20,
            "tracking_zoom_step": 1,
            "tracking_initial_extra_zoom_step": (
                1 if args.continuous_tracking else 0
            ),
            "vessel_number_recognition_enabled": args.continuous_tracking,
            "vessel_number_fallback": (
                "10032" if args.continuous_tracking else ""
            ),
            "command_timeout_seconds": 12,
            "reacquire_timeout_seconds": 8,
            # Small zoom steps limit overshoot while a short wait keeps pace
            # with the moving vessel in this prerecorded demonstration.
            "settle_seconds": 0.5,
            "maximum_off_home_seconds": 45,
            "minimum_target_observations": 2,
            "home_frame_delay_seconds": 0.5,
            "proposal_minimum_interval_seconds": 5,
            "proposal_maximum_verifications_per_hour": 60,
            "lost_retry_seconds": 15,
        },
    }
    created = request_json(
        "POST",
        f"{args.api_url}/v1/streams",
        env["PTZ_TEST_API_KEY"],
        payload,
    )
    session_path = Path("ptz-test-runtime/session.json")
    session_path.write_text(
        json.dumps(created, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    stream_id = str(created["stream_id"])
    deadline = time.monotonic() + args.wait_seconds
    current = created
    while time.monotonic() < deadline:
        time.sleep(1)
        current = request_json(
            "GET",
            f"{args.api_url}/v1/streams/{stream_id}",
            env["PTZ_TEST_API_KEY"],
        )
        if current.get("status") in {"running", "failed", "stopped"}:
            break

    print(json.dumps(current, ensure_ascii=False, indent=2))
    print("\n肉眼观察地址:")
    print(current.get("rtsp_url"))
    print("\n虚拟摄像机原始/变焦视图:")
    print(
        f"rtsp://{read_user}:{read_password}@127.0.0.1:18554/"
        "detected/virtual-input"
    )


if __name__ == "__main__":
    main()
