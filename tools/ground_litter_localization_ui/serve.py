#!/usr/bin/env python3
"""Local review server for Step 1C-1 localization confirmation/repair.

Same conventions as the Step 1B / audit UIs: stdlib ``http.server``, vanilla JS, atomic
JSON writes, no framework.  All review logic lives in
``rtsp_annotator.ground_litter_localization_review``; this file is transport only.

Needs cv2 for proposal generation, so run it with the profile virtualenv:

    .venv-profile/bin/python scripts/review_ground_litter_localizations.py serve
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

from rtsp_annotator.ground_litter_localization_review import (  # noqa: E402
    REVIEW_SCHEMA_VERSION,
    LocalizationInput,
    LocalizationError,
    ReviewState,
    ReviewError,
    Reviewer,
    validate_bbox,
)

HERE = Path(__file__).resolve().parent


class ReviewHandler(SimpleHTTPRequestHandler):
    data: LocalizationInput = None            # type: ignore[assignment]
    state_path: Path = None                   # type: ignore[assignment]
    recovery_root: Path = None                # type: ignore[assignment]
    queue: list = []

    # -- helpers ------------------------------------------------------------ #

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

    def _reviewer(self) -> Reviewer:
        from rtsp_annotator.ground_litter_localization_proposal import (
            ClassicProposalEngine,
        )
        return Reviewer(data=self.data, state=self._state(),
                        engine=ClassicProposalEngine())

    def _frame_path(self, episode_id: str) -> Path | None:
        episode = self.data.by_id.get(episode_id)
        if episode is None or not episode.verification_frame_path:
            return None
        frame = Path(episode.verification_frame_path).resolve()
        root = Path(self.recovery_root).resolve()
        if root != frame and root not in frame.parents:
            return None
        return frame if frame.is_file() else None

    def _episode_payload(self, episode_id: str, state: ReviewState) -> dict:
        episode = self.data.by_id.get(episode_id)
        if episode is None:
            return {}
        decision = state.get(episode_id) or {}
        width = episode.source_width or 2560
        height = episode.source_height or 1440

        def norm(box):
            if not box:
                return None
            return [round(box[0] / width, 6), round(box[1] / height, 6),
                    round(box[2] / width, 6), round(box[3] / height, 6)]

        return {
            "episode_id": episode.episode_id,
            "truth_target_id": episode.episode_id,
            "camera_id": episode.camera_id,
            "scene_version": episode.scene_version,
            "origin": episode.origin,
            "origin_flags": list(episode.origin_flags),
            "source_timestamp": episode.source_timestamp,
            "source_file_id": episode.source_file_id,
            "source_width": width,
            "source_height": height,
            "frame_url": f"/api/frame?episode={quote(episode_id)}",
            "original_location_type": ("POINT" if episode.origin == "manual_missing_target"
                                       else ("BBOX" if episode.original_bbox else "UNKNOWN")),
            "original_bbox": list(episode.original_bbox) if episode.original_bbox else None,
            "original_bbox_norm": norm(episode.original_bbox),
            "original_point": dict(episode.original_point or {}) or None,
            "original_point_source": dict(episode.point_source or {}) or None,
            "point_mapping_verified": bool(episode.point_source
                                           and episode.point_source.get("ok")),
            "member_card_ids": list(episode.member_card_ids),
            "member_labels": list(episode.member_labels),
            "screen": dict(episode.screen),
            "upstream_localization_status": episode.upstream_localization_status,
            "localization_status": decision.get("localization_status",
                                                "NEEDS_RELOCALIZATION"),
            "localization_decision": decision.get("localization_decision"),
            "verified_bbox": decision.get("verified_bbox"),
            "verified_bbox_norm": decision.get("verified_bbox_norm"),
            "proposals": decision.get("proposals") or [],
            "proposal_revision": int(decision.get("proposal_revision") or 0),
            "selected_proposal_id": decision.get("selected_proposal_id"),
            "note": decision.get("note") or "",
            "truth_review_reason": decision.get("truth_review_reason") or "",
        }

    # -- routing ------------------------------------------------------------ #

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
            queue = []
            for row in self.queue:
                entry = dict(row)
                record = state.get(row["episode_id"]) or {}
                entry["localization_status"] = record.get(
                    "localization_status", "NEEDS_RELOCALIZATION")
                entry["localization_decision"] = record.get("localization_decision")
                entry["reviewed"] = bool(record.get("review_status") == "human_reviewed")
                queue.append(entry)
            return self._json(200, {
                "required_episode_count": self.data.required_count,
                "gold_sha256": self.data.gold_sha256,
                "review_schema_version": REVIEW_SCHEMA_VERSION,
                "progress": state.progress(self.data),
                "queue": queue,
            })
        if path == "/api/episode":
            payload = self._episode_payload((query.get("id") or [""])[0], state)
            if not payload:
                return self._json(404, {"error": "unknown episode"})
            return self._json(200, payload)
        if path == "/api/frame":
            frame = self._frame_path((query.get("episode") or [""])[0])
            if frame is None:
                return self._json(404, {"error": "frame not found"})
            return self._bytes(200, frame.read_bytes(), "image/jpeg")
        return self._json(404, {"error": "not found"})

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
                decision = str(body.get("decision") or "")
                if decision == "BOX_OK":
                    state.decide_box_ok(episode, note=str(body.get("note") or ""),
                                        confirm_oversized=bool(body.get("confirm_oversized")))
                elif decision == "PROPOSAL_SELECTED":
                    state.select_proposal(episode, str(body.get("proposal_id") or ""),
                                          note=str(body.get("note") or ""))
                elif decision == "UNRESOLVED":
                    state.decide_unresolved(episode, note=str(body.get("note") or ""))
                elif decision == "TRUTH_REVIEW_REQUIRED":
                    state.decide_truth_review_required(
                        episode, reason=str(body.get("reason") or ""),
                        note=str(body.get("note") or ""))
                else:
                    return self._json(400, {"error": f"unknown decision {decision!r}"})
            elif parsed.path == "/api/proposals":
                from rtsp_annotator.ground_litter_localization_proposal import (
                    ClassicProposalEngine,
                )
                reviewer = Reviewer(data=self.data, state=state,
                                    engine=ClassicProposalEngine())
                reviewer.generate_proposals(episode_id)
            elif parsed.path == "/api/reset":
                state.reset(episode_id)
            else:
                return self._json(404, {"error": "not found"})
        except ReviewError as exc:
            return self._json(400, {"error": str(exc)})
        except LocalizationError as exc:
            return self._json(400, {"error": str(exc)})
        record = state.get(episode_id) or {}
        return self._json(200, {
            "ok": True,
            "progress": state.progress(self.data),
            "localization_status": record.get("localization_status"),
            "proposals": record.get("proposals") or [],
        })


def create_server(handler_class, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), handler_class)


def configure(args: argparse.Namespace, data: LocalizationInput) -> None:
    ReviewHandler.data = data
    ReviewHandler.recovery_root = Path(args.recovery_root)
    ReviewHandler.state_path = Path(args.state)
    ReviewHandler.state_path.parent.mkdir(parents=True, exist_ok=True)
    ReviewHandler.queue = [
        {
            "episode_id": e.episode_id,
            "camera_id": e.camera_id,
            "origin": e.origin,
            "origin_flags": list(e.origin_flags),
            "source_timestamp": e.source_timestamp,
            "has_original_bbox": e.original_bbox is not None,
            "likely_box_bad": bool(e.screen.get("likely_box_bad")),
            "screen_reasons": list(e.screen.get("screen_reasons") or []),
            "source_width": e.source_width,
            "source_height": e.source_height,
        }
        for e in data.required
    ]
    ReviewState.load(ReviewHandler.state_path, data=data).save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--gold-manifest", default=None)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--step1a-artifact", default=None)
    parser.add_argument("--state", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()
    from rtsp_annotator.ground_litter_localization_review import load_localization_input
    data = load_localization_input(args.gold, args.recovery_root,
                                   step1a_artifact=args.step1a_artifact,
                                   gold_manifest=args.gold_manifest)
    configure(args, data)
    server = create_server(ReviewHandler, args.host, args.port)
    print(f"Step 1C-1 localization review: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
