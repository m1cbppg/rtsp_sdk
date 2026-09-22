#!/usr/bin/env python3
"""Serve the five-camera final-ROI editor and persist normalized polygons."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output" / "ground_litter_final_roi_20260922"
REVIEWS_PATH = OUTPUT_ROOT / "roi_reviews.json"
DEVICES = {
    "01021": "44180209031322001021",
    "01022": "44180209031322001022",
    "01027": "44180209031322001027",
    "01028": "44180209031322001028",
    "01030": "44180209031322001030",
}


def frame_identity(camera_id: str) -> dict:
    path = OUTPUT_ROOT / "frames" / f"{camera_id}.jpg"
    if not path.is_file():
        raise ValueError(f"missing representative frame: {camera_id}")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canvas_size": [2560, 1440],
    }


def load_reviews() -> dict:
    if REVIEWS_PATH.is_file():
        return json.loads(REVIEWS_PATH.read_text())
    return {
        "schema": "ground_litter_final_roi_review_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cameras": {},
    }


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(path)


def validate_submission(payload: dict) -> tuple[str, list[list[float]]]:
    camera_id = str(payload.get("camera_id", ""))
    if camera_id not in DEVICES:
        raise ValueError("unknown camera_id")
    raw = payload.get("points")
    if not isinstance(raw, list) or len(raw) < 3 or len(raw) > 64:
        raise ValueError("ROI needs 3 to 64 points")
    points = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("each point must contain x and y")
        x, y = float(item[0]), float(item[1])
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("normalized points must be in [0, 1]")
        points.append([round(x, 6), round(y, 6)])
    # Reject degenerate clicks on one line or one tiny spot.
    area = abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2
    if area < .001:
        raise ValueError("ROI polygon area is too small")
    return camera_id, points


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def do_GET(self):  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/roi":
            reviews = load_reviews()
            payload = {
                **reviews,
                "available": {
                    camera_id: {
                        "device_code": device,
                        "image_url": f"/output/ground_litter_final_roi_20260922/frames/{camera_id}.jpg",
                        **frame_identity(camera_id),
                    }
                    for camera_id, device in DEVICES.items()
                },
            }
            return self._json(200, payload)
        if route == "/":
            self.path = "/tools/ground_litter_roi_ui/index.html"
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/api/roi":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024:
                raise ValueError("invalid request size")
            submission = json.loads(self.rfile.read(length))
            camera_id, points = validate_submission(submission)
            identity = frame_identity(camera_id)
            reviews = load_reviews()
            reviews["updated_at"] = datetime.now(timezone.utc).isoformat()
            reviews["cameras"][camera_id] = {
                "camera_id": camera_id,
                "device_code": DEVICES[camera_id],
                "roi": points,
                "canvas_size": identity["canvas_size"],
                "frame_path": identity["path"],
                "frame_sha256": identity["sha256"],
                "saved_at": reviews["updated_at"],
                "purpose": "final_ground_litter_detection_roi",
            }
            atomic_write(REVIEWS_PATH, reviews)
            config = {
                "kind": "ground_litter_camera_geometry",
                "schema_version": 1,
                "geometry_version": "final-roi-20260922-user-reviewed",
                **reviews["cameras"][camera_id],
                "exclude_zones": [],
            }
            atomic_write(
                OUTPUT_ROOT / "config" / f"ground_litter_{camera_id}_final_roi.json",
                config,
            )
            return self._json(200, {
                "ok": True, "camera_id": camera_id,
                "saved": len(reviews["cameras"]), "total": len(DEVICES),
            })
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)})

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if not str(args[0] if args else "").startswith("GET /api/roi"):
            super().log_message(format, *args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    for camera_id in DEVICES:
        frame_identity(camera_id)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"ROI editor: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
