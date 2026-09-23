#!/usr/bin/env python3
"""Minimal supplemental-target completion server for Step 1C-2M.

Only the 62 MISSING_REQUIRED tiles are served.  The reviewer clicks the centre of a
missed Required Litter, gets at most three machine proposals (A/B/C), and picks one;
nothing is ever auto-selected and no box can be drawn by hand.

Overlays are drawn in the browser; the PNG served here is always the untouched
Step 1C-2 candidate tile.

    .venv-profile/bin/python scripts/complete_ground_litter_positive_tiles.py serve
"""
from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import time
from urllib.parse import parse_qs, quote, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_point_proposal import (  # noqa: E402
    PointProposalEngine,
    size_prior_from_labels,
)
from rtsp_annotator.ground_litter_tile_completion import (  # noqa: E402
    FINAL_TILE_STATUSES,
    LOCALIZATION_STATUSES,
    LOCALIZATION_VERIFIED,
    MAX_PROPOSAL_REVISIONS,
    MAX_SUPPLEMENTAL_PER_TILE,
    STATE_SCHEMA_VERSION,
    TILE_SIZE,
    TILE_STATUSES,
    TILE_STATUS_COMPLETE,
    TILE_STATUS_NEEDS,
    CompletionError,
    CompletionState,
    ReviewError,
    apply_state,
    merged_labels,
    queue_rows,
)

HERE = Path(__file__).resolve().parent


class CompletionHandler(SimpleHTTPRequestHandler):
    tiles: dict = {}
    raw_missing: list = []
    decisions: dict = {}
    queue: list = []
    state_path: Path = None                 # type: ignore[assignment]
    images_dir: Path = None                 # type: ignore[assignment]
    input_fingerprint: str = ""
    engine: PointProposalEngine = None      # type: ignore[assignment]
    size_prior: dict = {}

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- plumbing ---------------------------------------------------------- #

    def _json(self, status: int, payload) -> None:
        body = dict(payload)
        body.setdefault("ok", status < 400)
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, status: int, code: str, message: str, **extra) -> None:
        self._json(status, {"ok": False, "error_code": code, "message": message, **extra})

    def _bytes(self, status: int, data: bytes, content_type: str,
               cache: str = "max-age=3600") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def _state(self) -> CompletionState:
        return CompletionState.load(self.state_path,
                                    input_fingerprint=self.input_fingerprint)

    def _queue_rows(self) -> list:
        """The 62 raw MISSING_REQUIRED rows, re-derived with the live review state."""
        return queue_rows({"missing_required": self.raw_missing,
                           "decisions": self.decisions, "by_id": self.tiles})

    def _body(self) -> dict:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(size) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return {}

    def _image_path(self, tile_id: str) -> Path | None:
        row = self.tiles.get(tile_id)
        if row is None:
            return None
        recorded = row.get("image_path")
        if not recorded:
            return None
        path = Path(str(recorded)).resolve()
        root = Path(self.images_dir).resolve()
        if root not in path.parents or path.name != f"{tile_id}.png":
            return None
        return path if path.is_file() else None

    # -- payloads ---------------------------------------------------------- #

    def meta_payload(self) -> dict:
        state = self._state()
        rows = self._queue_rows()
        applied = apply_state(rows, state)
        queue = []
        for row in applied:
            entry = state.tiles.get(row["tile_id"]) or {}
            queue.append({
                "tile_id": row["tile_id"],
                "primary_episode_id": row["primary_episode_id"],
                "camera_id": row["camera_id"],
                "source_timestamp": row["step1c0_decoded_timestamp"],
                "existing_label_count": row["existing_label_count"],
                "known_unlocalized_required_present":
                    row["known_unlocalized_required_present"],
                "completion_status": row["completion_status"],
                "supplemental_count": len(entry.get("supplemental_targets") or []),
                "supplemental_verified": sum(
                    1 for t in (entry.get("supplemental_targets") or [])
                    if t.get("localization_status") == LOCALIZATION_VERIFIED),
                "final_label_count": row["final_label_count"],
                "positive_training_ready": row["positive_training_ready"],
            })
        return {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "tile_size": TILE_SIZE,
            "queue_count": len(self.queue),
            "progress": state.progress(rows),
            "status_options": list(TILE_STATUSES),
            "final_status_options": list(FINAL_TILE_STATUSES),
            "localization_statuses": list(LOCALIZATION_STATUSES),
            "max_supplemental_per_tile": MAX_SUPPLEMENTAL_PER_TILE,
            "max_proposal_revisions": MAX_PROPOSAL_REVISIONS,
            "size_prior": dict(self.size_prior),
            "queue": queue,
        }

    def tile_payload(self, tile_id: str) -> dict:
        row = self.tiles.get(tile_id)
        if row is None:
            return {}
        state = self._state()
        state.tile(tile_id)
        applied = apply_state([row], state)[0]
        entry = state.tiles.get(tile_id) or {}
        return {
            "tile_id": tile_id,
            "primary_episode_id": row["primary_episode_id"],
            "camera_id": row["camera_id"],
            "source_file_id": row["source_file_id"],
            "step1c0_decoded_timestamp": row["step1c0_decoded_timestamp"],
            "source_width": row["source_width"],
            "source_height": row["source_height"],
            "source_crop_xyxy": row["source_crop_xyxy"],
            "image_sha256": row["image_sha256"],
            "image_url": f"/api/image?tile_id={quote(tile_id)}",
            "existing_labels": [{
                "label_id": f"existing-{index}",
                "letter": chr(ord("A") + index),
                "episode_ids": label.get("episode_ids"),
                "tile_xyxy": label["tile_xyxy"],
                "source_xyxy": label["source_xyxy"],
                "yolo_xywh_norm": label["yolo_xywh_norm"],
                "source_short_side_px": label.get("source_short_side_px"),
                "size_bucket": label.get("size_bucket"),
            } for index, label in enumerate(row.get("labels") or [])],
            "existing_label_count": row["existing_label_count"],
            "known_unlocalized_required_present":
                row["known_unlocalized_required_present"],
            "risk_flags": row.get("risk_flags"),
            "review_note": row.get("review_note"),
            "supplemental_targets": entry.get("supplemental_targets") or [],
            "merged_labels": applied["merged_labels"],
            "final_label_count": applied["final_label_count"],
            "completion_status": applied["completion_status"],
            "recheck": entry.get("recheck"),
            "completion_note": entry.get("note") or "",
            "positive_training_ready": applied["positive_training_ready"],
            "localization_statuses": list(LOCALIZATION_STATUSES),
            "max_proposal_revisions": MAX_PROPOSAL_REVISIONS,
            "max_supplemental_per_tile": MAX_SUPPLEMENTAL_PER_TILE,
            "size_prior": dict(self.size_prior),
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
                return self._error(404, "UNKNOWN_TILE", "unknown tile id")
            return self._json(200, payload)
        if path == "/api/image":
            image = self._image_path((query.get("tile_id") or [""])[0])
            if image is None:
                return self._error(404, "IMAGE_NOT_FOUND", "tile image not found")
            return self._bytes(200, image.read_bytes(), "image/png")
        return self._error(404, "NOT_FOUND", "not found")

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            return self._error(404, "MISSING_STATIC", "missing static file")
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return self._bytes(200, path.read_bytes(), content_type, cache="no-store")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        body = self._body()
        tile_id = str(body.get("tile_id") or "")
        row = self.tiles.get(tile_id)
        if row is None:
            return self._error(404, "UNKNOWN_TILE", "unknown tile id")
        state = self._state()
        try:
            if parsed.path == "/api/point":
                return self._point(state, row, body)
            if parsed.path == "/api/select":
                return self._select(state, row, body)
            if parsed.path == "/api/reject":
                state.reject_target(tile_id, str(body.get("target_id") or ""),
                                    reason=str(body.get("reason") or ""))
                return self._json(200, self._state_echo(state, tile_id))
            if parsed.path == "/api/delete":
                state.delete_target(tile_id, str(body.get("target_id") or ""))
                return self._json(200, self._state_echo(state, tile_id))
            if parsed.path == "/api/clear":
                state.clear_targets(tile_id)
                return self._json(200, self._state_echo(state, tile_id))
            if parsed.path == "/api/recheck":
                state.recheck(tile_id, str(body.get("decision") or ""),
                              note=str(body.get("note") or ""))
                return self._json(200, self._state_echo(state, tile_id))
            if parsed.path == "/api/skip":
                state.skip(tile_id)
                return self._json(200, {"progress": state.progress(self._queue_rows())})
        except ReviewError as exc:
            return self._error(400, "REVIEW_REJECTED", str(exc))
        except CompletionError as exc:                      # pragma: no cover
            return self._error(400, "COMPLETION_ERROR", str(exc))
        return self._error(404, "NOT_FOUND", "not found")

    # -- operations -------------------------------------------------------- #

    def _state_echo(self, state: CompletionState, tile_id: str) -> dict:
        entry = state.tiles.get(tile_id) or {}
        return {
            "tile_id": tile_id,
            "completion_status": entry.get("status") or TILE_STATUS_NEEDS,
            "supplemental_targets": entry.get("supplemental_targets") or [],
            "recheck": entry.get("recheck"),
            "progress": state.progress(self._queue_rows()),
            "positive_training_ready": (entry.get("status") == TILE_STATUS_COMPLETE),
        }

    def _point(self, state: CompletionState, row: dict, body: dict) -> None:
        try:
            tile_x = float(body.get("tile_x"))
            tile_y = float(body.get("tile_y"))
        except (TypeError, ValueError):
            return self._error(400, "POINT_INVALID", "tile_x / tile_y must be numbers")
        target_id = str(body.get("target_id") or "")
        started = time.monotonic()
        result = self.engine.propose(tile_path=Path(str(row["image_path"])),
                                     point_tile_xy=[tile_x, tile_y], size=TILE_SIZE,
                                     size_prior=self.size_prior, revision=1)
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 1)
        if not result.get("ok"):
            return self._json(200, {
                "ok": False, "error_code": result.get("error_code") or "PROPOSAL_FAILED",
                "message": result.get("message") or "no proposal could be generated",
                "elapsed_ms": elapsed_ms, "candidates": []})
        if target_id:
            revision = int(next(
                (t.get("proposal_revision") for t in state.targets(row["tile_id"])
                 if t["supplemental_target_id"] == target_id), 1)) + 1
            result = dict(result)
            result["revision"] = revision
            target = state.repoint(row, target_id, point_tile=[tile_x, tile_y],
                                   proposal_result=result)
        else:
            target = state.add_target(row, point_tile=[tile_x, tile_y],
                                      proposal_result=result)
        return self._json(200, {
            "ok": True, "elapsed_ms": elapsed_ms, "size_prior": dict(self.size_prior),
            "supplemental_target_id": target["supplemental_target_id"],
            "proposal_revision": target["proposal_revision"],
            "original_click": target["original_click"],
            "click_history": target["click_history"],
            "candidates": target["proposal_candidates"],
            "progress": state.progress(self._queue_rows())})

    def _select(self, state: CompletionState, row: dict, body: dict) -> None:
        target_id = str(body.get("target_id") or "")
        letter = str(body.get("letter") or "")
        if not target_id or not letter:
            return self._error(400, "SELECT_INVALID", "target_id and letter are required")
        target = state.target(row["tile_id"], target_id)
        others = [{"label_id": f"existing-{index}", "tile_xyxy": label["tile_xyxy"]}
                  for index, label in enumerate(row.get("labels") or [])]
        others += [{"label_id": item["supplemental_target_id"],
                    "tile_xyxy": item["verified_tile_xyxy"]}
                   for item in state.targets(row["tile_id"])
                   if item["supplemental_target_id"] != target_id
                   and item.get("verified_tile_xyxy")]
        try:
            outcome = state.select_proposal(
                row["tile_id"], target_id, letter, existing_boxes=others,
                confirm_independent=bool(body.get("confirm_independent")))
        except ReviewError as exc:
            return self._error(400, "SELECT_REJECTED", str(exc))
        if outcome.get("warning"):
            return self._json(200, {
                "ok": True, "warning": outcome["warning"],
                "matches": outcome.get("matches", []),
                "reason": outcome.get("reason"), "sides": outcome.get("sides", []),
                "supplemental_target_id": target_id,
                "progress": state.progress(self._queue_rows())})
        return self._json(200, {
            "ok": True, "warning": None, "target": outcome["target"],
            "supplemental_target_ids": [target_id],
            "progress": state.progress(self._queue_rows())})


def create_server(handler_class, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), handler_class)


def configure(args: argparse.Namespace, data: dict) -> None:
    rows = queue_rows(data)
    CompletionHandler.queue = rows
    CompletionHandler.raw_missing = list(data["missing_required"])
    CompletionHandler.decisions = dict(data["decisions"])
    # the handlers work on the queue rows: they carry the original labels, the crop and
    # the image path in the exact shape the UI needs
    CompletionHandler.tiles = {str(row["tile_id"]): row for row in rows}
    CompletionHandler.images_dir = Path(str(rows[0]["image_path"])).parent
    CompletionHandler.state_path = Path(args.state)
    CompletionHandler.state_path.parent.mkdir(parents=True, exist_ok=True)
    CompletionHandler.input_fingerprint = str(getattr(args, "input_fingerprint", ""))
    CompletionHandler.engine = PointProposalEngine()
    prior_labels = [label for row in data["accepted_rows"]
                    for label in (row.get("labels") or [])]
    CompletionHandler.size_prior = size_prior_from_labels(prior_labels)
    CompletionState.load(CompletionHandler.state_path,
                         input_fingerprint=CompletionHandler.input_fingerprint).save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step1c2-root", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    args = parser.parse_args()
    from rtsp_annotator.ground_litter_tile_completion import load_completion_input

    data = load_completion_input(args.step1c2_root)
    configure(args, data)
    server = create_server(CompletionHandler, args.host, args.port)
    print(f"Step 1C-2M supplemental completion: http://{args.host}:{args.port}/",
          flush=True)
    print(f"queue={len(CompletionHandler.queue)} state={args.state}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
