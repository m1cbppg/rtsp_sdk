#!/usr/bin/env python3
"""Local, offline review server for Step 1B gold-episode confirmation.

Same conventions as ``tools/ground_litter_audit_ui``: stdlib ``http.server``,
vanilla JS, atomic JSON writes, no framework and no network access.  All review
logic lives in ``rtsp_annotator.ground_litter_gold_episode_review`` so it stays
unit-testable; this file is only transport.

Run:
    python tools/ground_litter_gold_review_ui/serve.py \
        --artifact output/ground_litter_episode_candidates_20260923/episode_candidates.jsonl \
        --manifest output/ground_litter_episode_candidates_20260923/MANIFEST.json \
        --state    output/ground_litter_gold_episode_review_20260923/review_state.json \
        --repo-root .
"""
from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_gold_episode_review import (  # noqa: E402
    MANUAL_TARGET_TRUTH_CLASSES,
    REVIEW_SCHEMA_VERSION,
    NearDuplicateTarget,
    ReviewError,
    ReviewState,
    build_queue,
    load_step1a,
    merge_suggestions,
)

HERE = Path(__file__).resolve().parent
ASSET_SLOTS = ("context", "crop", "before", "current", "after")

#: Only geometry/timing/scale facts may reach the review page.  Anything that
#: names a predictor, a score or a model is dropped: Step 0B requires Blind review.
SAFE_EVIDENCE_KEYS = (
    "grouping_method",
    "time_span_seconds",
    "bbox_center_spread_px",
    "bbox_diagonal_spread_px",
    "center_spread_limit_px",
    "binding_pair_center_distance_px",
    "binding_pair_threshold_pressure",
    "complete_linkage_pressure_within_limits",
    "thresholds",
    "pairwise_summary",
)


def _safe_grouping_evidence(evidence) -> dict:
    if not isinstance(evidence, dict):
        return {}
    return {key: evidence[key] for key in SAFE_EVIDENCE_KEYS if key in evidence}



class ReviewHandler(SimpleHTTPRequestHandler):
    repo_root: Path
    artifact: Path
    manifest: Path
    state_path: Path
    step1a = None
    queue: list = []

    # -- helpers ------------------------------------------------------------ #

    def log_message(self, fmt, *args):  # quieter, single line
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, status: int, payload) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _state(self) -> ReviewState:
        return ReviewState.load(self.state_path, step1a=self.step1a)

    def _resolve_asset(self, batch_key: str, relative: str) -> Path | None:
        base = self.step1a.batch_dirs.get(batch_key)
        if not base:
            return None
        base_path = Path(base).resolve()
        candidate = (base_path / relative).resolve()
        if base_path != candidate and base_path not in candidate.parents:
            return None
        return candidate if candidate.is_file() else None

    def _image_url(self, batch_key: str, relative: str | None) -> str | None:
        if not relative:
            return None
        return "/api/image?batch=%s&path=%s" % (quote(batch_key), quote(relative, safe=""))

    def _member_payload(self, member: dict) -> dict:
        batch_key = str(member.get("batch_key") or "")
        assets = dict(member.get("assets") or {})
        tiles: dict[str, dict] = {}
        for slot in ASSET_SLOTS:
            if slot == "current":
                source = assets.get("context_image")
                tiles[slot] = {
                    "url": self._image_url(batch_key, source),
                    "asset_path": source,
                    "missing": not bool(source),
                    "derived_from": "context_image",
                    "note": "no current_image field exists; context_image is the current-frame crop",
                }
                continue
            relative = assets.get("%s_image" % slot)
            resolved = self._resolve_asset(batch_key, str(relative)) if relative else None
            tiles[slot] = {
                "url": self._image_url(batch_key, str(relative)) if relative else None,
                "asset_path": relative,
                "missing": resolved is None,
                "derived_from": None,
                "note": "" if resolved is not None else "MISSING",
            }
        return {
            "card_id": member.get("card_id"),
            "raw_review_id": member.get("raw_review_id"),
            "batch_key": batch_key,
            "original_label": member.get("label"),
            "timestamp": member.get("timestamp"),
            "frame_id": member.get("frame_id"),
            "bbox": member.get("bbox"),
            "source_file_id": member.get("source_file_id"),
            "tiles": tiles,
        }

    def _suggestion_with_image(self, suggestion: dict) -> dict:
        """Attach a resolvable preview URL for a *neighbouring* candidate.

        The neighbour's representative card is not in this candidate's member list,
        so its image must be resolved from the neighbour's own lineage.
        """
        out = dict(suggestion)
        url = None
        other = self.step1a.candidate_by_id.get(suggestion.get("candidate_id"))
        if other:
            members = {m.get("card_id"): m
                       for m in (other.get("lineage") or {}).get("review_cards") or []}
            member = members.get(suggestion.get("representative_card_id"))
            if member:
                assets = dict(member.get("assets") or {})
                url = self._image_url(str(member.get("batch_key") or ""),
                                      assets.get("context_image"))
        out["representative_url"] = url
        return out

    def _candidate_payload(self, candidate_id: str, state: ReviewState) -> dict:
        candidate = self.step1a.candidate_by_id.get(candidate_id)
        if candidate is None:
            return {}
        queue_row = next((q for q in self.queue if q["episode_candidate_id"] == candidate_id), {})
        return {
            "episode_candidate_id": candidate_id,
            "camera_id": candidate.get("camera_id"),
            "scene_version": candidate.get("scene_version") or "UNKNOWN_HISTORICAL",
            "start_timestamp": candidate.get("start_timestamp"),
            "end_timestamp": candidate.get("end_timestamp"),
            "member_count": candidate.get("member_count"),
            "original_labels_summary": candidate.get("original_labels_summary"),
            "grouping_evidence": _safe_grouping_evidence(
                candidate.get("candidate_grouping_evidence")),
            "risk": {
                "queue_priority": queue_row.get("queue_priority"),
                "grouping_risk": queue_row.get("grouping_risk"),
                "lineage_only": queue_row.get("lineage_only"),
                "grouping_risk_reasons": queue_row.get("grouping_risk_reasons", []),
                "lineage_reasons": queue_row.get("lineage_reasons", []),
                "same_frame_neighbour_candidate_ids":
                    queue_row.get("same_frame_neighbour_candidate_ids", []),
            },
            "representative_card_id": candidate.get("representative_card_id"),
            "members": [self._member_payload(m)
                        for m in (candidate.get("lineage") or {}).get("review_cards") or []],
            "merge_suggestions": [self._suggestion_with_image(s)
                                  for s in merge_suggestions(self.step1a, candidate_id)],
            "decision": state.get(candidate_id),
            "manual_targets": [
                {"manual_target_id": row["manual_target_id"],
                 "truth_class": row.get("truth_class"),
                 "point": dict(row.get("point") or {}),
                 "source_member_card_id": row.get("source_member_card_id"),
                 "localization_status": row.get("localization_status"),
                 "origin": row.get("origin"),
                 "note": row.get("note"),
                 "created_at": row.get("created_at"),
                 "updated_at": row.get("updated_at"),
                 "revision": row.get("revision")}
                for row in state.manual_targets_for(candidate_id)
            ],
            "manual_target_truth_classes": list(MANUAL_TARGET_TRUTH_CLASSES),
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
                record = state.get(row["episode_candidate_id"]) or {}
                entry["status"] = record.get("status", "pending")
                entry["decision"] = record.get("decision")
                queue.append(entry)
            return self._json(200, {
                "artifact": str(self.artifact),
                "artifact_sha256": self.step1a.artifact_sha256,
                "candidate_count": self.step1a.candidate_count,
                "member_card_count": self.step1a.member_card_count,
                "step1a_code_commit": self.step1a.step1a_code_commit,
                "review_schema_version": REVIEW_SCHEMA_VERSION,
                "progress": state.progress(self.step1a),
                "queue": queue,
            })
        if path == "/api/candidate":
            candidate_id = (query.get("id") or [""])[0]
            payload = self._candidate_payload(candidate_id, state)
            if not payload:
                return self._json(404, {"error": "unknown candidate"})
            return self._json(200, payload)
        if path == "/api/image":
            batch_key = (query.get("batch") or [""])[0]
            relative = (query.get("path") or [""])[0]
            resolved = self._resolve_asset(batch_key, relative)
            if resolved is None:
                return self._json(404, {"error": "image not found"})
            content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
            return self._bytes(200, resolved.read_bytes(), content_type)
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
        state = self._state()
        try:
            if parsed.path == "/api/decision":
                candidate_id = str(body["candidate_id"])
                state.decide(
                    self.step1a, candidate_id, str(body["decision"]),
                    note=str(body.get("note") or ""),
                    episodes=body.get("episodes"),
                    merge_targets=body.get("merge_targets"),
                    localization_status=str(body.get("localization_status") or "OK"),
                )
            elif parsed.path == "/api/skip":
                state.skip(str(body["candidate_id"]), note=str(body.get("note") or ""))
            elif parsed.path == "/api/reset":
                state.reset(str(body["candidate_id"]))
            elif parsed.path == "/api/manual-target":
                record, warnings = state.add_manual_target(
                    self.step1a, str(body["candidate_id"]),
                    card_id=str(body["card_id"]),
                    truth_class=str(body.get("truth_class") or "REQUIRED_LITTER"),
                    clicked_asset_type=str(body["clicked_asset_type"]),
                    clicked_asset_path=str(body["clicked_asset_path"]),
                    x=body["x"], y=body["y"],
                    image_width=body["image_width"], image_height=body["image_height"],
                    note=str(body.get("note") or ""),
                    allow_near_duplicate=bool(body.get("allow_near_duplicate")),
                )
                return self._json(200, {
                    "ok": True,
                    "manual_target": record,
                    "near_duplicate_warnings": warnings,
                    "progress": state.progress(self.step1a),
                })
            elif parsed.path == "/api/manual-target/update":
                record = state.update_manual_target(
                    str(body["manual_target_id"]),
                    truth_class=(str(body["truth_class"]) if body.get("truth_class") else None),
                    note=(str(body["note"]) if "note" in body else None),
                )
                return self._json(200, {"ok": True, "manual_target": record})
            elif parsed.path == "/api/manual-target/repoint":
                record = state.repoint_manual_target(
                    self.step1a, str(body["manual_target_id"]),
                    clicked_asset_type=str(body["clicked_asset_type"]),
                    clicked_asset_path=str(body["clicked_asset_path"]),
                    x=body["x"], y=body["y"],
                    image_width=body["image_width"], image_height=body["image_height"],
                )
                return self._json(200, {"ok": True, "manual_target": record})
            elif parsed.path == "/api/manual-target/delete":
                record = state.delete_manual_target(str(body["manual_target_id"]))
                return self._json(200, {"ok": True, "deleted": record})
            else:
                return self._json(404, {"error": "not found"})
        except NearDuplicateTarget as exc:
            # Ask, never silently refuse: two real pieces of litter can be adjacent.
            return self._json(409, {
                "error": str(exc),
                "requires_confirmation": True,
                "near_duplicates": exc.near_duplicates,
            })
        except ReviewError as exc:
            return self._json(400, {"error": str(exc)})
        except KeyError as exc:
            return self._json(400, {"error": f"missing field {exc}"})
        return self._json(200, {
            "ok": True,
            "progress": state.progress(self.step1a),
            "decision": state.get(str(body.get("candidate_id"))),
        })


def create_server(handler_class, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), handler_class)


def configure(args: argparse.Namespace) -> None:
    ReviewHandler.repo_root = Path(args.repo_root).resolve()
    ReviewHandler.artifact = Path(args.artifact)
    ReviewHandler.manifest = Path(args.manifest)
    ReviewHandler.state_path = Path(args.state)
    ReviewHandler.step1a = load_step1a(
        ReviewHandler.artifact, ReviewHandler.manifest, repo_root=ReviewHandler.repo_root)
    ReviewHandler.queue = build_queue(ReviewHandler.step1a)
    ReviewHandler.state_path.parent.mkdir(parents=True, exist_ok=True)
    # touch the state file so a first run is resumable from the start
    ReviewState.load(ReviewHandler.state_path, step1a=ReviewHandler.step1a).save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    configure(args)
    server = create_server(ReviewHandler, args.host, args.port)
    print(f"Step 1B gold-episode review: http://{args.host}:{args.port}/", flush=True)
    print(f"queue={len(ReviewHandler.queue)} candidates", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
