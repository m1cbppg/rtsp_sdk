"""Read-only local/LAN checks for a continuous PTZ demo environment."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from urllib.request import Request, urlopen


def _get(url: str, api_key: str | None = None) -> dict:
    headers = {"X-API-Key": api_key} if api_key else {}
    with urlopen(Request(url, headers=headers), timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="optional API URL; no network is used without it")
    parser.add_argument("--api-key-env", default="RTSP_API_KEY")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "module": "rtsp_annotator.continuous_tracking",
        "camera_control_key_present": bool(os.getenv("CAMERA_CONTROL_API_KEY")),
        "config": str(args.config) if args.config else None,
    }
    if args.config:
        result["config_exists"] = args.config.is_file()
    if args.base_url:
        try:
            result["health"] = _get(args.base_url.rstrip("/") + "/health")
        except Exception as exc:
            result["health_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if "health_error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
