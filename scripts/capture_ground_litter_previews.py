"""Capture one local calibration image per explicitly selected camera.

No URL, token or response body is persisted or logged. No camera control or
project stream is created. Run this script only for authorized cameras.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

ENDPOINT = (
    "https://qyzcapi.dgjx0769.com/ims-mainte-pc/p-api/v1/monitor/"
    "play/ctseelink/devices/rtsp"
)


def capture(code: str, output: str) -> None:
    import av
    av.logging.set_level(av.logging.PANIC)
    target = Path(output)
    status = {"device_code": code, "captured_at": datetime.now(timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(
            ENDPOINT, data=json.dumps({"deviceCode": code}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=12) as response:
            payload = json.load(response)
        if payload.get("code") != 200:
            raise ValueError("upstream rejected request")
        url = payload["data"]["url"]
        if not url.startswith(("rtsp://", "rtsps://")):
            raise ValueError("invalid scheme")
        with av.open(url, options={"rtsp_transport": "tcp"}, timeout=(12, 12)) as video:
            for index, frame in enumerate(video.decode(video=0)):
                if index == 10:
                    frame.to_image().save(target / f"{code}.jpg", quality=95)
                    status.update(status="ok", width=frame.width, height=frame.height)
                    break
            else:
                raise ValueError("insufficient frames")
    except Exception as exc:
        status.update(status="error", error_type=type(exc).__name__)
    (target / f"{code}.json").write_text(json.dumps(status, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not code.isdigit() or len(code) != 20 for code in args.devices):
        parser.error("Expected 20 digit device codes")
    args.output.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    for code in args.devices:
        # Refuse to overwrite calibration evidence from another capture.
        if (args.output / f"{code}.json").exists() or (args.output / f"{code}.jpg").exists():
            parser.error("Output contains prior capture; use a new directory")
        proc = ctx.Process(target=capture, args=(code, str(args.output)))
        proc.start()
        proc.join(35)
        if proc.is_alive():
            proc.terminate()
            proc.join(3)
            if proc.is_alive():
                proc.kill()
                proc.join()
        meta = args.output / f"{code}.json"
        if meta.exists():
            print(meta.read_text(), flush=True)
        else:
            meta.write_text(json.dumps({"device_code": code, "status": "capture_failed_or_timeout"}) + "\n")
            print(f"{code}: capture_failed_or_timeout", flush=True)


if __name__ == "__main__":
    main()
