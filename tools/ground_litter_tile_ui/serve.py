#!/usr/bin/env python3
"""Minimal annotation-completeness review server for Step 1C-2.

The page answers exactly one question per tile: "does this 640x640 tile show every
Required Litter with a usable box?"  It therefore offers **no** way to add, move,
resize or re-propose a box: MISSING_REQUIRED just excludes the tile (§21).

Pure stdlib.  Overlays are drawn in the browser; the PNG served here is always the
untouched source-native tile.

    .venv/bin/python scripts/build_ground_litter_positive_tiles.py serve
"""
from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from urllib.parse import parse_qs, quote, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_positive_tiles import (  # noqa: E402
    CLASS_NAME,
    REVIEW_DECISIONS,
    REVIEW_SCHEMA_VERSION,
    STATUS_READY,
    TILE_SIZE,
    PositiveTileError,
    ReviewError,
    TileReviewState,
    apply_review,
)

HERE = Path(__file__).resolve().parent


class ReviewHandler(SimpleHTTPRequestHandler):
    candidates: list = []
    reviewable: list = []
    by_id: dict = {}
    state_path: Path = None                 # type: ignore[assignment]
    images_dir: Path = None                 # type: ignore[assignment]

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- plumbing ---------------------------------------------------------- #

    def _json(self, status: int, payload) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _bytes(self, status: int, data: bytes, content_type: str,
               cache: str = "max-age=3600") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def _state(self) -> TileReviewState:
        return TileReviewState.load(self.state_path, candidate_count=len(self.candidates))

    def _image_path(self, tile_id: str) -> Path | None:
        candidate = self.by_id.get(tile_id)
        if candidate is None or candidate.get("candidate_generation_status") != STATUS_READY:
            return None
        recorded = candidate.get("image_path")
        if not recorded:
            return None
        path = Path(str(recorded)).resolve()
        root = Path(self.images_dir).resolve()
        if root not in path.parents:
            return None
        if path.name != f"{tile_id}.png":
            return None
        return path if path.is_file() else None

    # -- payloads ---------------------------------------------------------- #

    def meta_payload(self) -> dict:
        state = self._state()
        rows = apply_review(self.reviewable, state)
        queue = [{
            "tile_id": row["tile_id"],
            "primary_episode_id": row["primary_episode_id"],
            "camera_id": row["camera_id"],
            "label_count": row.get("label_count"),
            "source_timestamp": row.get("step1c0_decoded_timestamp"),
            "review_status": row.get("annotation_review_status"),
            "positive_training_ready": row.get("positive_training_ready"),
            "known_unlocalized_required_present":
                row.get("known_unlocalized_required_present"),
        } for row in rows]
        return {
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "tile_size": TILE_SIZE,
            "class_name": CLASS_NAME,
            "reviewable_count": len(self.reviewable),
            "candidate_count": len(self.candidates),
            "excluded_by_generation_count": len(self.candidates) - len(self.reviewable),
            "progress": state.progress(self.candidates),
            "options": list(REVIEW_DECISIONS),
            "queue": queue,
        }

    def tile_payload(self, tile_id: str) -> dict:
        candidate = self.by_id.get(tile_id)
        if candidate is None or candidate.get("candidate_generation_status") != STATUS_READY:
            return {}
        state = self._state()
        row = apply_review([candidate], state)[0]
        return {
            "tile_id": row["tile_id"],
            "primary_episode_id": row["primary_episode_id"],
            "primary_episode_ids": row.get("primary_episode_ids"),
            "merged_from_tile_ids": row.get("merged_from_tile_ids"),
            "camera_id": row["camera_id"],
            "scene_version": row.get("scene_version"),
            "source_file_id": row["source_file_id"],
            "requested_timestamp": row.get("requested_timestamp"),
            "step1c0_decoded_timestamp": row.get("step1c0_decoded_timestamp"),
            "step1c2_decoded_timestamp": row.get("step1c2_decoded_timestamp"),
            "timestamp_delta_ms": row.get("timestamp_delta_ms"),
            "frame_interval_ms": row.get("frame_interval_ms"),
            "source_width": row.get("source_width"),
            "source_height": row.get("source_height"),
            "crop_size": row.get("crop_size"),
            "source_crop_xyxy": row.get("source_crop_xyxy"),
            "primary_source_bbox": row.get("primary_source_bbox"),
            "min_label_margin_px": row.get("min_label_margin_px"),
            "labels": [{
                "letter": chr(ord("A") + index),
                "episode_ids": label["episode_ids"],
                "class_id": label["class_id"],
                "class_name": label["class_name"],
                "source_xyxy": label["source_xyxy"],
                "tile_xyxy": label["tile_xyxy"],
                "yolo_xywh_norm": label["yolo_xywh_norm"],
                "source_short_side_px": label.get("source_short_side_px"),
                "size_bucket": label.get("size_bucket"),
            } for index, label in enumerate(row.get("labels") or [])],
            "label_count": row.get("label_count"),
            "all_known_label_episode_ids": row.get("all_known_label_episode_ids"),
            "known_required_same_frame_count": row.get("known_required_same_frame_count"),
            "known_unlocalized_required_present":
                row.get("known_unlocalized_required_present"),
            "known_unlocalized_required_episode_ids":
                row.get("known_unlocalized_required_episode_ids"),
            "known_unlocalized_required_same_frame_ids":
                row.get("known_unlocalized_required_same_frame_ids"),
            "known_unlocalized_required_in_crop_ids":
                row.get("known_unlocalized_required_in_crop_ids"),
            "risk_geometry_source": row.get("risk_geometry_source"),
            "risk_flags": row.get("risk_flags") or [],
            "ignore_small_in_crop_ids": row.get("ignore_small_in_crop_ids"),
            "non_litter_same_frame_ids": row.get("non_litter_same_frame_ids"),
            "uncertain_truth_in_crop_ids": row.get("uncertain_truth_in_crop_ids"),
            "other_truth_same_frame_ids": row.get("other_truth_same_frame_ids"),
            "image_sha256": row.get("image_sha256"),
            "size_bytes": row.get("size_bytes"),
            "image_url": f"/api/image?tile_id={quote(tile_id)}",
            "annotation_review_status": row.get("annotation_review_status"),
            "annotation_review_note": row.get("annotation_review_note"),
            "annotation_reviewed_at": row.get("annotation_reviewed_at"),
            "positive_training_ready": row.get("positive_training_ready"),
            "options": list(REVIEW_DECISIONS),
        }

    # -- HTTP -------------------------------------------------------------- #

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            return self._serve_file(HERE / "index.html")
        if path in ("/app.js", "/styles.css"):
            return self._serve_file(HERE / path.lstrip("/"))
        if path == "/api/meta":
            return self._json(200, self.meta_payload())
        if path == "/api/tile":
            payload = self.tile_payload((query.get("id") or [""])[0])
            if not payload:
                return self._json(404, {"error": "unknown or non-reviewable tile"})
            return self._json(200, payload)
        if path == "/api/image":
            image = self._image_path((query.get("tile_id") or [""])[0])
            if image is None:
                return self._json(404, {"error": "image not found"})
            return self._bytes(200, image.read_bytes(), "image/png")
        return self._json(404, {"error": "not found"})

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            return self._json(404, {"error": "missing static file"})
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return self._bytes(200, path.read_bytes(), content_type, cache="no-store")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "invalid JSON body"})
        tile_id = str(body.get("tile_id") or "")
        candidate = self.by_id.get(tile_id)
        if candidate is None:
            return self._json(404, {"error": "unknown tile"})
        state = self._state()
        try:
            if parsed.path == "/api/review":
                state.decide(candidate, str(body.get("decision") or ""),
                             note=str(body.get("note") or ""),
                             confirmed_no_unlabeled_required=bool(
                                 body.get("confirmed_no_unlabeled_required")))
            elif parsed.path == "/api/reset":
                state.reset(tile_id)
            elif parsed.path == "/api/skip":
                state.skip(tile_id)
            else:
                return self._json(404, {"error": "not found"})
        except (ReviewError, PositiveTileError) as exc:
            return self._json(400, {"error": str(exc)})
        record = state.get(tile_id) or {}
        return self._json(200, {
            "ok": True,
            "progress": state.progress(self.candidates),
            "annotation_review_status": record.get("annotation_review_status"),
            "positive_training_ready": record.get("positive_training_ready"),
        })


def create_server(handler_class, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), handler_class)


def configure(args: argparse.Namespace, data: dict, candidates: list) -> None:
    ReviewHandler.candidates = list(candidates)
    ReviewHandler.reviewable = [c for c in candidates
                                if c.get("candidate_generation_status") == STATUS_READY]
    ReviewHandler.by_id = {str(c["tile_id"]): c for c in candidates}
    ReviewHandler.images_dir = Path(getattr(args, "output")) / "candidate_tiles" / "images"
    ReviewHandler.state_path = Path(args.state)
    ReviewHandler.state_path.parent.mkdir(parents=True, exist_ok=True)
    TileReviewState.load(ReviewHandler.state_path,
                         candidate_count=len(candidates)).save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8774)
    args = parser.parse_args()
    from rtsp_annotator.ground_litter_positive_tiles import read_jsonl

    candidates = read_jsonl(Path(args.output) / "tile_candidates.jsonl")
    configure(args, {}, candidates)
    server = create_server(ReviewHandler, args.host, args.port)
    print(f"Step 1C-2 annotation-complete review: http://{args.host}:{args.port}/",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
