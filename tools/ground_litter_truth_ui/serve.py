#!/usr/bin/env python3
"""Minimal local review server for Step 1C-1R truth reconciliation.

Only truth is being decided here, so the page deliberately offers **no** localization
controls: no box verdict buttons, no candidate selection, no bbox pick, no point click.
Localization was frozen by Step 1C-1 and is read-only context.

Pure stdlib (this step needs no image processing):

    .venv/bin/python scripts/reconcile_ground_litter_truth.py serve
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

from rtsp_annotator.ground_litter_truth_reconciliation import (  # noqa: E402
    DECISIONS,
    DECISION_TO_TRUTH_CLASS,
    REVIEW_SCHEMA_VERSION,
    ReconciliationInput,
    ReconciliationError,
    ReviewError,
    ReviewState,
)

HERE = Path(__file__).resolve().parent


class ReviewHandler(SimpleHTTPRequestHandler):
    data: ReconciliationInput = None      # type: ignore[assignment]
    state_path: Path = None               # type: ignore[assignment]
    recovery_root: Path = None            # type: ignore[assignment]
    queue: list = []

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, status: int, payload) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _state(self) -> ReviewState:
        return ReviewState.load(self.state_path, data=self.data)

    def _frame_path(self, episode_id: str) -> Path | None:
        episode = self.data.by_id.get(episode_id)
        # Only the 60 in-scope episodes are part of this review surface: the 109 VERIFIED
        # and 11 UNRESOLVED are frozen and must not be served here at all (§11/§12/§21).
        if episode is None or not episode.in_scope or not episode.verification_frame_path:
            return None
        frame = Path(episode.verification_frame_path).resolve()
        root = Path(self.recovery_root).resolve()
        if root != frame and root not in frame.parents:
            return None
        return frame if frame.is_file() else None

    def _episode_payload(self, episode_id: str, state: ReviewState) -> dict:
        episode = self.data.by_id.get(episode_id)
        if episode is None or not episode.in_scope:
            return {}
        decision = state.get(episode_id) or {}
        width = episode.source_width or 2560
        height = episode.source_height or 1440

        def norm(box):
            if not box:
                return None
            return [round(box[0] / width, 6), round(box[1] / height, 6),
                    round(box[2] / width, 6), round(box[3] / height, 6)]

        # A point-only episode (ADD_MISSING_TARGET) has no historical bbox.  Step 1C-1
        # already mapped the click into source-frame pixels; expose that mapping instead
        # of guessing from the raw crop-relative coordinates.
        point_source = dict(episode.original_point_source or {})
        frame_point = None
        if point_source.get("ok") and point_source.get("x") is not None:
            try:
                frame_point = [float(point_source["x"]), float(point_source["y"])]
            except (TypeError, ValueError):
                frame_point = None

        return {
            "episode_id": episode.episode_id,
            "truth_target_id": episode.episode_id,
            "camera_id": episode.camera_id,
            "scene_version": episode.scene_version,
            "origin": episode.origin,
            "source_timestamp": episode.source_timestamp,
            "source_width": width,
            "source_height": height,
            "frame_url": f"/api/frame?episode={quote(episode_id)}",
            "original_bbox": episode.original_bbox,
            "original_bbox_norm": norm(episode.original_bbox),
            "original_point": dict(episode.original_point or {}) or None,
            "original_point_source": point_source or None,
            "original_point_frame": frame_point,
            "original_crop_box": (list(point_source["crop_box"])
                                  if point_source.get("crop_box") else None),
            "location_type": "BBOX" if episode.original_bbox else "POINT",
            "localization_status": episode.localization_status,
            "localization_decision": episode.localization_decision,
            "localization_verified_bbox": episode.localization_verified_bbox,
            # the Step 1C-1 free-text reason, shown for context only
            "prior_truth_review_reason": (episode.prior_truth_review_reason
                                          or "NO_PRIOR_REASON"),
            "reconciliation_decision": decision.get("reconciliation_decision"),
            "reconciled_truth_class": decision.get("reconciled_truth_class"),
            "review_reason_optional": decision.get("review_reason_optional") or "",
            "reviewed_at": decision.get("reviewed_at"),
            "revision": decision.get("revision"),
            "options": list(DECISIONS),
            "option_truth_class": dict(DECISION_TO_TRUTH_CLASS),
        }

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        state = self._state()
        if path in ("/", "/index.html"):
            return self._serve_file(HERE / "index.html")
        if path in ("/app.js", "/styles.css"):
            return self._serve_file(HERE / path.lstrip("/"))
        if path == "/api/meta":
            return self._json(200, self.meta_payload())
        if path == "/api/episode":
            payload = self._episode_payload((query.get("id") or [""])[0], state)
            if not payload:
                return self._json(404, {"error": "unknown or out-of-scope episode"})
            return self._json(200, payload)
        if path == "/api/frame":
            frame = self._frame_path((query.get("episode") or [""])[0])
            if frame is None:
                return self._json(404, {"error": "frame not found"})
            return self._bytes(200, frame.read_bytes(), "image/jpeg")
        return self._json(404, {"error": "not found"})

    def meta_payload(self) -> dict:
        state = self._state()
        statuses = self.data.counts()
        queue = []
        for row in self.queue:
            entry = dict(row)
            record = state.get(row["episode_id"]) or {}
            entry["reconciliation_decision"] = record.get("reconciliation_decision")
            entry["reconciled_truth_class"] = record.get("reconciled_truth_class")
            entry["reviewed"] = bool(record.get("review_status") == "human_reviewed")
            queue.append(entry)
        return {
            "in_scope_count": len(self.queue),
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "progress": state.progress(self.data),
            # frozen upstream counts, so the page never hardcodes them
            "frozen": {
                "verified_bbox": statuses.get("VERIFIED_BBOX", 0),
                "localization_unresolved": statuses.get("LOCALIZATION_UNRESOLVED", 0),
                "needs_relocalization": statuses.get("NEEDS_RELOCALIZATION", 0),
            },
            "queue": queue,
        }

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            return self._json(404, {"error": "missing static file"})
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return self._bytes(200, path.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "invalid JSON body"})
        episode_id = str(body.get("episode_id") or "")
        episode = self.data.by_id.get(episode_id)
        if episode is None:
            return self._json(404, {"error": "unknown episode"})
        state = self._state()
        try:
            if parsed.path == "/api/decision":
                state.decide(episode, str(body.get("decision") or ""),
                             reason=str(body.get("reason") or ""))
            elif parsed.path == "/api/reset":
                state.reset(episode_id)
            else:
                return self._json(404, {"error": "not found"})
        except (ReviewError, ReconciliationError) as exc:
            return self._json(400, {"error": str(exc)})
        record = state.get(episode_id) or {}
        return self._json(200, {
            "ok": True,
            "progress": state.progress(self.data),
            "reconciliation_decision": record.get("reconciliation_decision"),
            "reconciled_truth_class": record.get("reconciled_truth_class"),
        })


def create_server(handler_class, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), handler_class)


def configure(args: argparse.Namespace, data: ReconciliationInput) -> None:
    ReviewHandler.data = data
    ReviewHandler.recovery_root = Path(args.recovery_root)
    ReviewHandler.state_path = Path(args.state)
    ReviewHandler.state_path.parent.mkdir(parents=True, exist_ok=True)
    ReviewHandler.queue = [
        {
            "episode_id": e.episode_id,
            "camera_id": e.camera_id,
            "origin": e.origin,
            "source_timestamp": e.source_timestamp,
            "prior_truth_review_reason": e.prior_truth_review_reason or "NO_PRIOR_REASON",
            "localization_status": e.localization_status,
            "has_original_bbox": e.original_bbox is not None,
        }
        for e in data.in_scope
    ]
    ReviewState.load(ReviewHandler.state_path, data=data).save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--gold-manifest", default=None)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--localization-root", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8773)
    args = parser.parse_args()
    from rtsp_annotator.ground_litter_truth_reconciliation import (
        load_reconciliation_input,
    )
    data = load_reconciliation_input(args.gold, args.recovery_root,
                                     args.localization_root,
                                     gold_manifest=args.gold_manifest)
    configure(args, data)
    server = create_server(ReviewHandler, args.host, args.port)
    print(f"Step 1C-1R truth reconciliation: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
