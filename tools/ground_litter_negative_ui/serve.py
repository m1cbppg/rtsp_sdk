#!/usr/bin/env python3
"""Minimal negative-completeness review server for Step 1D.

The only question per tile is "does this 640x640 tile really contain no REQUIRED
Litter?".  The page therefore never offers a box tool, and the historical non-litter
anchor box is drawn only as context (it is not a training label).

Pure stdlib; the PNG served here is always the untouched candidate tile.

    .venv/bin/python scripts/build_ground_litter_hard_negatives.py serve
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

from rtsp_annotator.ground_litter_hard_negatives import (  # noqa: E402
    REVIEW_DECISIONS,
    REVIEW_SCHEMA_VERSION,
    STATUS_READY,
    TILE_SIZE,
    HardNegativeError,
    NegativeReviewState,
    ReviewError,
    apply_review,
)

HERE = Path(__file__).resolve().parent


class NegativeHandler(SimpleHTTPRequestHandler):
    candidates: list = []
    reviewable: list = []
    by_id: dict = {}
    state_path: Path = None                 # type: ignore[assignment]
    images_dir: Path = None                 # type: ignore[assignment]
    input_fingerprint: str = ""

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

    def _state(self) -> NegativeReviewState:
        return NegativeReviewState.load(self.state_path,
                                        candidate_count=len(self.candidates))

    def _image_path(self, candidate_id: str) -> Path | None:
        row = self.by_id.get(candidate_id)
        if row is None or row.get("candidate_generation_status") != STATUS_READY:
            return None
        recorded = row.get("image_path")
        if not recorded:
            return None
        path = Path(str(recorded)).resolve()
        root = Path(self.images_dir).resolve()
        if root not in path.parents or path.name != f"{candidate_id}.png":
            return None
        return path if path.is_file() else None

    # -- payloads ---------------------------------------------------------- #

    def meta_payload(self) -> dict:
        state = self._state()
        rows = apply_review(self.reviewable, state)
        queue = [{
            "negative_tile_id": row["negative_tile_id"],
            "camera_id": row["camera_id"],
            "timestamp": row.get("timestamp"),
            "origin": row.get("origin"),
            "hardness_source": row.get("hardness_source"),
            "risk_flags": row.get("risk_flags") or [],
            "review_status": row.get("review_status"),
            "hard_negative_ready": row.get("hard_negative_ready"),
        } for row in rows]
        return {
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "tile_size": TILE_SIZE,
            "reviewable_count": len(self.reviewable),
            "candidate_count": len(self.candidates),
            "excluded_by_generation_count": len(self.candidates) - len(self.reviewable),
            "progress": state.progress(self.candidates),
            "options": list(REVIEW_DECISIONS),
            "queue": queue,
        }

    def candidate_payload(self, candidate_id: str) -> dict:
        row = self.by_id.get(candidate_id)
        if row is None or row.get("candidate_generation_status") != STATUS_READY:
            return {}
        state = self._state()
        item = apply_review([row], state)[0]
        return {
            "negative_tile_id": item["negative_tile_id"],
            "camera_id": item["camera_id"],
            "scene_version": item.get("scene_version"),
            "source_file_id": item.get("source_file_id"),
            "timestamp": item.get("timestamp"),
            "decoded_timestamp": item.get("decoded_timestamp"),
            "timestamp_delta_ms": item.get("timestamp_delta_ms"),
            "source_width": item.get("source_width"),
            "source_height": item.get("source_height"),
            "source_crop_xyxy": item.get("source_crop_xyxy"),
            "anchor_type": item.get("anchor_type"),
            "anchor_source_xyxy": item.get("anchor_source_xyxy"),
            "anchor_tile_xyxy": item.get("anchor_tile_xyxy"),
            "anchor_min_margin_px": item.get("anchor_min_margin_px"),
            "origin": item.get("origin"),
            "historical_label": item.get("historical_label"),
            "historical_source": item.get("historical_source"),
            "hardness_source": item.get("hardness_source"),
            "source_card_id": item.get("source_card_id"),
            "source_review_batch": item.get("source_review_batch"),
            "note": item.get("note"),
            "risk_flags": item.get("risk_flags") or [],
            "known_required_boxes": item.get("known_required_boxes") or [],
            "uncertain_truth_ids": item.get("uncertain_truth_ids") or [],
            "ignore_small_ids": item.get("ignore_small_ids") or [],
            "image_sha256": item.get("image_sha256"),
            "size_bytes": item.get("size_bytes"),
            "image_url": f"/api/image?candidate_id={quote(candidate_id)}",
            "review_status": item.get("review_status"),
            "review_reason": item.get("review_reason"),
            "reviewed_at": item.get("reviewed_at"),
            "hard_negative_ready": item.get("hard_negative_ready"),
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
        if path == "/api/candidate":
            payload = self.candidate_payload((query.get("id") or [""])[0])
            if not payload:
                return self._json(404, {"error": "unknown or non-reviewable candidate"})
            return self._json(200, payload)
        if path == "/api/image":
            image = self._image_path((query.get("candidate_id") or [""])[0])
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
        candidate_id = str(body.get("candidate_id") or "")
        candidate = self.by_id.get(candidate_id)
        if candidate is None:
            return self._json(404, {"error": "unknown candidate"})
        state = self._state()
        try:
            if parsed.path == "/api/review":
                state.decide(candidate, str(body.get("decision") or ""),
                             reason=str(body.get("reason") or ""))
            elif parsed.path == "/api/reset":
                state.reset(candidate_id)
            elif parsed.path == "/api/skip":
                state.skip(candidate_id)
            else:
                return self._json(404, {"error": "not found"})
        except (ReviewError, HardNegativeError) as exc:
            return self._json(400, {"error": str(exc)})
        record = state.get(candidate_id) or {}
        return self._json(200, {
            "ok": True,
            "progress": state.progress(self.candidates),
            "review_status": record.get("review_status"),
            "hard_negative_ready": record.get("hard_negative_ready"),
        })


def create_server(handler_class, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), handler_class)


def configure(args: argparse.Namespace, *, plan: dict, candidates: list,
              positive_rows: list = ()) -> None:
    NegativeHandler.candidates = list(candidates)
    NegativeHandler.reviewable = [row for row in candidates
                                  if row.get("candidate_generation_status") == STATUS_READY]
    NegativeHandler.by_id = {str(row["negative_tile_id"]): row for row in candidates}
    NegativeHandler.images_dir = (Path(str(candidates[0]["image_path"])).parent
                                  if candidates else Path(args.output))
    NegativeHandler.state_path = Path(args.state)
    NegativeHandler.state_path.parent.mkdir(parents=True, exist_ok=True)
    NegativeHandler.input_fingerprint = str(getattr(args, "input_fingerprint", ""))
    NegativeReviewState.load(NegativeHandler.state_path,
                             candidate_count=len(candidates),
                             input_fingerprint=NegativeHandler.input_fingerprint).save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8776)
    args = parser.parse_args()
    from rtsp_annotator.ground_litter_hard_negatives import read_jsonl

    candidates = read_jsonl(Path(args.output) / "negative_candidates.jsonl")
    configure(args, plan={}, candidates=candidates)
    server = create_server(NegativeHandler, args.host, args.port)
    print(f"Step 1D hard negative review: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
