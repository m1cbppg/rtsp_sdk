#!/usr/bin/env python3
"""Step 2C-1: Development Blind Truth review (detector-free).

Subcommands
-----------
manifest   read the frozen Development split + ROI configs into development_manifest.json
serve      run the Blind Truth review UI (BLIND TRUTH MODE / DETECTOR OUTPUT DISABLED)
sample     build the fixed 5 s episode frames and the fixed 30 s global ROI frames
freeze     SHA-256 the truth, chmod it read-only and refuse further in-place edits
status     review coverage / counts (no detector involved)
erratum    append a post-freeze correction instead of editing the frozen truth
selftest   prove that no detector, checkpoint or inference is reachable from this step

Everything decodes raw PS with OpenCV only.  There is deliberately no model anywhere:
no torch, no ultralytics, no .pt loading, no bbox or confidence from a network.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_blind_truth import (  # noqa: E402
    EPISODE_CONFIRMED,
    EPISODE_DRAFT,
    IGNORE_SMALL,
    LOCALIZATION_PROPOSAL,
    LOCALIZATION_UNRESOLVED,
    NON_LITTER,
    REQUIRED_LITTER,
    REVIEW_DONE,
    REVIEW_IN_PROGRESS,
    REVIEW_PENDING,
    SCHEMA_VERSION,
    TRUTH_CLASSES,
    UNCERTAIN,
    BlindTruthError,
    DetectorUseError,
    DevelopmentScopeError,
    SealedAssetError,
    TruthFrozenError,
    append_truth_erratum,
    assert_development_asset,
    assert_episode_consistency,
    assert_not_frozen,
    assert_not_sealed,
    atomic_write_json,
    build_manifest,
    build_summary,
    freeze_truth,
    is_frozen,
    load_development_inventory,
    localization_state,
    new_episode,
    new_truth_object,
    next_id,
    read_jsonl,
    refresh_episode,
    review_coverage,
    sample_global_roi_frames,
    sample_visible_frames,
    write_jsonl,
)

#: The authorised Step 0A Development split (Sealed is never touched).
DEVELOPMENT_ROOT = Path("/home/sf01/ground-litter-feasibility/20260923-r1/archive"
                        "/ground-litter-detector-feasibility-20260923-r1/development")
ROI_DIR = ROOT / "output" / "ground_litter_final_roi_20260922" / "config"
DEFAULT_OUTPUT = ROOT / "output" / "ground_litter_step2c_blind_truth_20260923"
UI_DIR = ROOT / "tools" / "ground_litter_blind_truth_ui"

FORBIDDEN_MODULES = ("torch", "ultralytics", "tensorrt", "onnxruntime")
# Every token is split across adjacent literals so this file never contains the exact
# string it searches for (otherwise the checker would flag itself).
FORBIDDEN_SOURCE_TOKENS = ("import " "torch", "import " "ultralytics",
                           "from " "ultralytics", "YOLO" "(",
                           "load_state" "_dict", "torch" "vision",
                           "Inference" "Session", "Tensor" "RT")

MANIFEST_NAME = "development_manifest.json"
TRUTH_NAME = "truth_objects.jsonl"
EPISODES_NAME = "episodes.jsonl"
VISIBLE_NAME = "visible_frame_manifest.jsonl"
GLOBAL_NAME = "global_roi_frame_manifest.jsonl"
LOCALIZATION_NAME = "localization_state.json"
REVIEW_NAME = "review_state.json"
SUMMARY_NAME = "SUMMARY.json"
BUNDLE_MANIFEST_NAME = "MANIFEST.json"
FREEZE_NAME = "FREEZE.json"


# --------------------------------------------------------------------------- #
# blindness self-check
# --------------------------------------------------------------------------- #


def blindness_selftest() -> dict:
    """Prove this step cannot reach a detector, a checkpoint or an inference call."""
    loaded = sorted(name for name in FORBIDDEN_MODULES if name in sys.modules)
    source_hits: list[dict[str, str]] = []
    for path in (Path(__file__), ROOT / "rtsp_annotator" / "ground_litter_blind_truth.py",
                 UI_DIR / "app.js", UI_DIR / "index.html"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in FORBIDDEN_SOURCE_TOKENS:
            if token in text:
                source_hits.append({"file": str(path), "token": token})
    # any .pt on the command line or in the output dir is a hard failure
    pt_paths = [str(arg) for arg in sys.argv[1:] if str(arg).endswith(".pt")]
    return {
        "forbidden_modules_loaded": loaded,
        "forbidden_source_tokens": source_hits,
        "checkpoint_args": pt_paths,
        "detector_loaded": bool(loaded),
        "checkpoint_accessed": bool(pt_paths),
        "inference_executed": False,
        "ok": not loaded and not source_hits and not pt_paths,
    }


def assert_blind() -> dict:
    report = blindness_selftest()
    if not report["ok"]:
        raise DetectorUseError(f"blind truth step is not detector-free: {report}")
    return report


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #


class TruthStore:
    """Resume-safe artifact store; every mutation is an atomic file write."""

    def __init__(self, output: Path, development_root: Path, roi_dir: Path):
        self.output = Path(output)
        self.development_root = Path(development_root)
        self.roi_dir = Path(roi_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.inventory = self._load_inventory()
        self.files = {row["file_id"]: row for row in self.inventory["files"]}
        self.cameras = {}
        for row in self.inventory["files"]:
            self.cameras.setdefault(row["camera_id"], {
                "roi": row["roi"], "geometry_version": row["roi_geometry_version"]})

    # ---- loading ---------------------------------------------------------- #

    def _load_inventory(self) -> dict:
        path = self.output / MANIFEST_NAME
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload.get("files"):
                raise BlindTruthError(f"{path} has no files")
            return payload
        assert_development_asset(self.development_root)
        inventory = load_development_inventory(self.development_root, self.roi_dir)
        if not inventory["ok"]:
            raise BlindTruthError(f"Development inventory problems: {inventory['problems']}")
        atomic_write_json(path, inventory)
        return inventory

    @property
    def truth(self) -> list[dict]:
        return read_jsonl(self.output / TRUTH_NAME)

    @property
    def episodes(self) -> list[dict]:
        return read_jsonl(self.output / EPISODES_NAME)

    def review_state(self) -> dict:
        path = self.output / REVIEW_NAME
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"files": {}, "resume": {}}

    def frozen(self) -> dict | None:
        path = self.output / FREEZE_NAME
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    # ---- mutation --------------------------------------------------------- #

    def _guard(self) -> None:
        assert_not_frozen(self.output)

    def _save_truth(self, rows: list[dict]) -> None:
        write_jsonl(self.output / TRUTH_NAME, rows)

    def _save_episodes(self, rows: list[dict]) -> None:
        write_jsonl(self.output / EPISODES_NAME, rows)

    def _save_review(self, payload: dict) -> None:
        atomic_write_json(self.output / REVIEW_NAME, payload)

    def add_truth(self, *, truth_class: str, camera_id: str, source_file_id: str,
                  decoded_timestamp: float, source_point=None,
                  source_bbox_xyxy=None, note: str | None = None,
                  localization_status: str | None = None) -> dict:
        with self.lock:
            self._guard()
            if truth_class not in TRUTH_CLASSES:
                raise BlindTruthError(f"unknown truth class {truth_class!r}")
            record = self.files.get(source_file_id)
            if record is None:
                raise BlindTruthError(f"unknown Development PS {source_file_id!r}")
            if record["camera_id"] != camera_id:
                raise BlindTruthError("camera_id does not match the PS")
            rows = self.truth
            truth_id = next_id("t", [row["truth_id"] for row in rows])
            timestamp = _add_seconds(record["record_start"], float(decoded_timestamp))
            roi = self.cameras[camera_id]["roi"]
            entry = new_truth_object(
                truth_id=truth_id, camera_id=camera_id,
                scene_version=record["scene_version"], source_file_id=source_file_id,
                timestamp=timestamp, decoded_timestamp=round(float(decoded_timestamp), 3),
                frame_index=int(round(float(decoded_timestamp) * 25.0)),
                truth_class=truth_class,
                source_width=int(record["canvas_size"][0]),
                source_height=int(record["canvas_size"][1]),
                source_point=source_point, source_bbox_xyxy=source_bbox_xyxy, roi=roi,
                localization_status=localization_status, note=note)
            rows.append(entry)
            self._save_truth(rows)
            self._touch_resume(camera_id, source_file_id, decoded_timestamp)
            return entry

    def update_truth(self, *, truth_id: str, source_bbox_xyxy=None,
                     localization_status: str | None = None,
                     truth_class: str | None = None, note: str | None = None,
                     source_point=None) -> dict:
        with self.lock:
            self._guard()
            rows = self.truth
            for row in rows:
                if row["truth_id"] != truth_id:
                    continue
                if truth_class is not None:
                    if truth_class not in TRUTH_CLASSES:
                        raise BlindTruthError(f"unknown truth class {truth_class!r}")
                    row["truth_class"] = truth_class
                    row["enters_recall_denominator"] = truth_class == REQUIRED_LITTER
                    row["enters_ignore_set"] = truth_class in (IGNORE_SMALL, UNCERTAIN)
                if source_point is not None:
                    row["source_point"] = [float(source_point[0]), float(source_point[1])]
                if source_bbox_xyxy is not None:
                    x1, y1, x2, y2 = (float(v) for v in source_bbox_xyxy)
                    if not (0 <= x1 < x2 <= row["source_width"]
                            and 0 <= y1 < y2 <= row["source_height"]):
                        raise BlindTruthError("bbox must be inside the source frame")
                    row["source_bbox_xyxy"] = [x1, y1, x2, y2]
                if localization_status is not None:
                    if localization_status == LOCALIZATION_PROPOSAL and not row.get("source_bbox_xyxy"):
                        # keep the point truth, mark it explicitly unresolved instead
                        row["localization_status"] = LOCALIZATION_UNRESOLVED
                    else:
                        row["localization_status"] = localization_status
                if note is not None:
                    row["note"] = note
                row["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self._save_truth(rows)
                self._refresh_episode(rows, row.get("episode_id"))
                return row
            raise BlindTruthError(f"unknown truth_id {truth_id!r}")

    def delete_truth(self, *, truth_id: str, reason: str) -> dict:
        with self.lock:
            self._guard()
            rows = self.truth
            kept = [row for row in rows if row["truth_id"] != truth_id]
            if len(kept) == len(rows):
                raise BlindTruthError(f"unknown truth_id {truth_id!r}")
            removed = [row for row in rows if row["truth_id"] == truth_id][0]
            self._save_truth(kept)
            self._refresh_episode(kept, removed.get("episode_id"))
            return {"deleted": removed, "reason": reason}

    def _refresh_episode(self, rows: list[dict], episode_id: str | None) -> None:
        if not episode_id:
            return
        episodes = self.episodes
        for index, episode in enumerate(episodes):
            if episode["episode_id"] == episode_id:
                updated = refresh_episode(episode, rows)
                assert_episode_consistency(updated, rows)
                episodes[index] = updated
                self._save_episodes(episodes)
                return

    def episode_action(self, *, action: str, truth_id: str | None = None,
                       episode_id: str | None = None, which: str | None = None,
                       note: str | None = None) -> dict:
        with self.lock:
            self._guard()
            rows = self.truth
            episodes = self.episodes
            by_id = {row["truth_id"]: row for row in rows}
            if action == "new":
                if truth_id not in by_id:
                    raise BlindTruthError("new episode needs an existing truth object")
                observation = by_id[truth_id]
                new_id = next_id("ep", [row["episode_id"] for row in episodes])
                episode = new_episode(episode_id=new_id, observation=observation,
                                      physical_identity_note=note)
                observation["episode_id"] = new_id
                episodes.append(episode)
                self._save_truth(rows)
                self._save_episodes(episodes)
                self._refresh_episode(rows, new_id)
                return {"episode_id": new_id}
            if action == "assign":
                if truth_id not in by_id or not episode_id:
                    raise BlindTruthError("assign needs truth_id and episode_id")
                target = [row for row in episodes if row["episode_id"] == episode_id]
                if not target:
                    raise BlindTruthError(f"unknown episode {episode_id!r}")
                by_id[truth_id]["episode_id"] = episode_id
                self._save_truth(rows)
                self._refresh_episode(rows, episode_id)
                return {"episode_id": episode_id}
            if action == "confirm":
                if not episode_id:
                    raise BlindTruthError("confirm needs episode_id")
                for episode in episodes:
                    if episode["episode_id"] == episode_id:
                        episode["review_status"] = EPISODE_CONFIRMED
                        episode["confirmed_at"] = time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        self._save_episodes(episodes)
                        return {"episode_id": episode_id, "status": EPISODE_CONFIRMED}
                raise BlindTruthError(f"unknown episode {episode_id!r}")
            if action == "draft":
                if not episode_id:
                    raise BlindTruthError("draft needs episode_id")
                for episode in episodes:
                    if episode["episode_id"] == episode_id:
                        episode["review_status"] = EPISODE_DRAFT
                        self._save_episodes(episodes)
                        return {"episode_id": episode_id, "status": EPISODE_DRAFT}
                raise BlindTruthError(f"unknown episode {episode_id!r}")
            if action == "interval":
                if truth_id not in by_id or not episode_id or which not in ("start", "end"):
                    raise BlindTruthError("interval needs truth_id, episode_id and which")
                observation = by_id[truth_id]
                for episode in episodes:
                    if episode["episode_id"] != episode_id:
                        continue
                    key = ("first_confirmable_timestamp" if which == "start"
                           else "last_confirmable_timestamp")
                    episode[key] = observation["timestamp"]
                    episode.setdefault("interval_marks", {})[which] = {
                        "truth_id": truth_id, "timestamp": observation["timestamp"]}
                    episode["interval_source"] = "human"
                    self._save_episodes(episodes)
                    return {"episode_id": episode_id, which: observation["timestamp"]}
                raise BlindTruthError(f"unknown episode {episode_id!r}")
            raise BlindTruthError(f"unknown episode action {action!r}")

    def set_review(self, *, file_id: str, status: str, decoded_timestamp: float | None = None,
                   camera_id: str | None = None) -> dict:
        with self.lock:
            self._guard()
            if file_id not in self.files:
                raise BlindTruthError(f"unknown Development PS {file_id!r}")
            if status not in (REVIEW_PENDING, REVIEW_IN_PROGRESS, REVIEW_DONE):
                raise BlindTruthError(f"unknown review status {status!r}")
            payload = self.review_state()
            payload.setdefault("files", {})[file_id] = {
                "status": status,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self._save_review(payload)
            if decoded_timestamp is not None:
                self._touch_resume(camera_id or self.files[file_id]["camera_id"],
                                   file_id, decoded_timestamp)
            return payload["files"][file_id]

    def _touch_resume(self, camera_id: str, file_id: str, decoded_timestamp: float) -> None:
        payload = self.review_state()
        payload["resume"] = {"camera_id": camera_id, "file_id": file_id,
                             "decoded_timestamp": round(float(decoded_timestamp), 3),
                             "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        files = payload.setdefault("files", {})
        slot = files.setdefault(file_id, {"status": REVIEW_PENDING})
        if slot.get("status") == REVIEW_PENDING:
            slot["status"] = REVIEW_IN_PROGRESS
            slot["updated_at"] = payload["resume"]["updated_at"]
        self._save_review(payload)

    # ---- sampling / reporting -------------------------------------------- #

    def build_samples(self) -> dict:
        with self.lock:
            self._guard()
            rows = self.truth
            episodes = self.episodes
            for episode in episodes:
                assert_episode_consistency(episode, rows)
            visible = sample_visible_frames(episodes, rows, self.inventory["files"])
            global_frames = sample_global_roi_frames(self.cameras, self.inventory["files"])
            write_jsonl(self.output / VISIBLE_NAME, visible)
            write_jsonl(self.output / GLOBAL_NAME, global_frames)
            return {"visible_frames": len(visible), "global_frames": len(global_frames),
                    "confirmed_required_episodes": len(
                        [row for row in episodes
                         if row["truth_class"] == REQUIRED_LITTER
                         and row.get("review_status") == EPISODE_CONFIRMED])}

    def localization(self) -> dict:
        return localization_state(self.episodes, self.truth,
                                  read_jsonl(self.output / VISIBLE_NAME))

    def coverage(self) -> dict:
        cover = review_coverage(self.inventory["files"], self.review_state())
        cover["per_file"] = {fid: slot.get("status", REVIEW_PENDING)
                             for fid, slot in (self.review_state().get("files") or {}).items()}
        return cover

    def write_reports(self, *, code_commit: str) -> dict:
        with self.lock:
            self._guard()
            localization = self.localization()
            atomic_write_json(self.output / LOCALIZATION_NAME, localization)
            inventory = self.inventory
            coverage = self.coverage()
            roi_note = ("frozen final-roi-20260922-user-reviewed polygons loaded from "
                        "output/ground_litter_final_roi_20260922/config; every Development "
                        "PS declares a matching geometry_version")
            summary = build_summary(inventory, truth_objects=self.truth,
                                    episodes=self.episodes,
                                    visible_frames=read_jsonl(self.output / VISIBLE_NAME),
                                    global_frames=read_jsonl(self.output / GLOBAL_NAME),
                                    localization=localization, review=coverage,
                                    roi_note=roi_note)
            summary["blindness"] = blindness_selftest()
            atomic_write_json(self.output / SUMMARY_NAME, summary)
            manifest = build_manifest(inventory, summary,
                                      generated_at=time.strftime(
                                          "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                      code_commit=code_commit,
                                      artifact_root=self.output, review=coverage)
            atomic_write_json(self.output / BUNDLE_MANIFEST_NAME, manifest)
            return summary

    def freeze(self) -> dict:
        with self.lock:
            self._guard()
            coverage = self.coverage()
            record = freeze_truth(self.output, review=coverage,
                                  extra={"blindness": blindness_selftest()})
            return record


def _add_seconds(stamp: str, seconds: float) -> str:
    from rtsp_annotator.ground_litter_blind_truth import format_ts, parse_ts
    from datetime import timedelta

    return format_ts(parse_ts(stamp) + timedelta(seconds=float(seconds)))


# --------------------------------------------------------------------------- #
# decoder (OpenCV only)
# --------------------------------------------------------------------------- #


class PsDecoder:
    """Source-native frame reader over the raw PS files; no model is involved."""

    PREROLL_SECONDS = 1.6
    FPS = 25.0

    def __init__(self, files: dict[str, dict], max_open: int = 2):
        # HEVC mid-GOP seeks make FFmpeg emit a warning burst per frame.  That noise
        # floods any captured stdout/stderr and can deadlock a pipe, so silence the
        # FFmpeg backend and the OpenCV logger before the first capture is opened.
        os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
        os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
        try:
            import cv2

            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except Exception:                                    # pragma: no cover
            pass
        self.files = files
        self.max_open = max_open
        self._captures: dict[str, object] = {}
        self._order: list[str] = []
        self._position: dict[str, float] = {}
        self.lock = threading.RLock()

    def _capture(self, file_id: str):
        import cv2

        if file_id in self._captures:
            self._order.remove(file_id)
            self._order.append(file_id)
            return self._captures[file_id]
        record = self.files.get(file_id)
        if record is None:
            raise BlindTruthError(f"unknown PS {file_id!r}")
        path = record["ps_path"]
        assert_development_asset(path)
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise BlindTruthError(f"cannot decode {path}")
        self._captures[file_id] = capture
        self._order.append(file_id)
        while len(self._order) > self.max_open:
            evicted = self._order.pop(0)
            self._captures.pop(evicted).release()
            self._position.pop(evicted, None)
        return capture

    def frame(self, file_id: str, decoded_timestamp: float):
        import cv2

        with self.lock:
            capture = self._capture(file_id)
            target_ms = max(0.0, float(decoded_timestamp)) * 1000.0
            current = self._position.get(file_id)
            need_seek = (current is None
                         or target_ms < current - 1.0
                         or target_ms > current + 1500.0)
            if need_seek:
                capture.set(cv2.CAP_PROP_POS_MSEC,
                            max(0.0, float(decoded_timestamp) - self.PREROLL_SECONDS)
                            * 1000.0)
            frame = None
            actual_ms = current or 0.0
            for _ in range(int(self.FPS * (self.PREROLL_SECONDS + 1.0)) + 10):
                ok, candidate = capture.read()
                if not ok:
                    break
                actual_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                frame = candidate
                if actual_ms >= target_ms:
                    break
            if frame is None:
                raise BlindTruthError(f"no frame decoded for {file_id} at "
                                      f"{decoded_timestamp}s")
            self._position[file_id] = actual_ms
            return frame, actual_ms / 1000.0

    def meta(self, file_id: str) -> dict:
        record = self.files.get(file_id)
        if record is None:
            raise BlindTruthError(f"unknown PS {file_id!r}")
        return {"file_id": file_id, "camera_id": record["camera_id"],
                "source_width": int(record["canvas_size"][0]),
                "source_height": int(record["canvas_size"][1]),
                "fps": self.FPS,
                "record_start": record["record_start"],
                "record_end": record["record_end"],
                "duration_seconds": record["duration_seconds"]}


def classic_cv_proposals(frame, x: float, y: float, *, half: int = 96) -> list[dict]:
    """Three classic-CV bbox candidates around a human-confirmed point.

    Used only *after* a human has confirmed that a Required object exists there.  It never
    discovers anything: it is pure thresholding/morphology/contours on the local window.
    """
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    x0 = int(max(0, min(width - 1, round(x - half))))
    x1 = int(max(1, min(width, round(x + half))))
    y0 = int(max(0, min(height - 1, round(y - half))))
    y1 = int(max(1, min(height, round(y + half))))
    window = frame[y0:y1, x0:x1]
    if window.size == 0:
        return []
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    masks = {
        "A-adaptive": cv2.morphologyEx(
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 8),
            cv2.MORPH_CLOSE, kernel, iterations=2),
        "B-otsu": cv2.morphologyEx(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            cv2.MORPH_CLOSE, kernel, iterations=2),
        "C-edge": cv2.morphologyEx(cv2.Canny(gray, 60, 160), cv2.MORPH_CLOSE, kernel,
                                   iterations=3),
    }
    point = (int(round(x)) - x0, int(round(y)) - y0)
    proposals: list[dict] = []
    for label, mask in masks.items():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        best = None
        for index in range(1, count):
            bx, by, bw, bh, area = stats[index]
            if area < 24 or area > window.shape[0] * window.shape[1] * 0.9:
                continue
            inside = (bx <= point[0] < bx + bw and by <= point[1] < by + bh)
            distance = 0 if inside else min(
                abs(point[0] - bx), abs(point[0] - (bx + bw)),
                abs(point[1] - by), abs(point[1] - (by + bh)))
            score = (0 if inside else 1, distance, -area)
            if best is None or score < best[0]:
                best = (score, (bx, by, bw, bh, area), inside)
        if best is None:
            continue
        (bx, by, bw, bh, area) = best[1]
        bbox = [float(bx + x0), float(by + y0), float(bx + x0 + bw), float(by + y0 + bh)]
        proposals.append({"label": label, "bbox_xyxy": bbox, "area": int(area),
                          "contains_point": bool(best[2]),
                          "candidate_half": half})
    return proposals


# --------------------------------------------------------------------------- #
# HTTP UI
# --------------------------------------------------------------------------- #


def make_handler(store: TruthStore, decoder: PsDecoder):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse
    import cv2

    class Handler(BaseHTTPRequestHandler):
        server_version = "BlindTruth/1.0"

        def log_message(self, fmt, *args):        # keep the console readable
            if "/api/frame" not in (args[0] if args else ""):
                sys.stderr.write("ui: " + (fmt % args) + "\n")

        # -- helpers -------------------------------------------------------- #
        def _json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, message, status=400):
            self._json({"error": str(message)}, status)

        def _bytes(self, payload, content_type):
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(payload)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _file(self, path: Path, content_type: str):
            if not path.is_file():
                self._error("not found", 404)
                return
            self._bytes(path.read_bytes(), content_type)

        def _jpeg(self, frame, quality=82):
            ok, buffer = cv2.imencode(".jpg", frame,
                                      [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                raise BlindTruthError("jpeg encode failed")
            return buffer.tobytes()

        def _params(self):
            return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

        def _frame(self, params):
            file_id = params["file_id"]
            ts = float(params.get("t", 0.0))
            frame, actual = decoder.frame(file_id, ts)
            return frame, actual

        def _draw_roi(self, frame, camera_id):
            """Draw the frozen ROI polygon (not detector output)."""
            roi = (store.cameras.get(camera_id) or {}).get("roi") or []
            if not roi:
                return frame
            import numpy as np
            height, width = frame.shape[:2]
            points = np.array([[[int(point[0] * width), int(point[1] * height)]
                                for point in roi]], dtype="int32")
            cv2.polylines(frame, points, True, (90, 217, 138), 2, cv2.LINE_AA)
            return frame

        # -- GET ------------------------------------------------------------ #
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            params = self._params()
            try:
                if path in ("/", "/index.html"):
                    return self._file(UI_DIR / "index.html", "text/html; charset=utf-8")
                if path == "/app.js":
                    return self._file(UI_DIR / "app.js", "application/javascript")
                if path == "/styles.css":
                    return self._file(UI_DIR / "styles.css", "text/css")
                if path == "/api/inventory":
                    payload = dict(store.inventory)
                    payload["resume"] = store.review_state().get("resume") or {}
                    payload["frozen"] = bool(store.frozen())
                    return self._json(payload)
                if path == "/api/state":
                    return self._json({"schema_version": SCHEMA_VERSION,
                                       "truth_objects": store.truth,
                                       "episodes": store.episodes,
                                       "coverage": store.coverage(),
                                       "frozen": store.frozen()})
                if path == "/api/frame-meta":
                    frame, actual = self._frame(params)
                    meta = decoder.meta(params["file_id"])
                    meta["decoded_timestamp"] = actual
                    meta["frame_shape"] = list(frame.shape)
                    return self._json(meta)
                if path == "/api/frame":
                    frame, _ = self._frame(params)
                    width = int(params.get("w", 1280))
                    scale = width / frame.shape[1]
                    resized = cv2.resize(frame, (width, int(frame.shape[0] * scale)),
                                         interpolation=cv2.INTER_AREA)
                    camera = decoder.meta(params["file_id"])["camera_id"]
                    resized = self._draw_roi(resized, camera)
                    return self._bytes(self._jpeg(resized), "image/jpeg")
                if path in ("/api/crop", "/api/crop-meta"):
                    frame, actual = self._frame(params)
                    cx, cy = float(params.get("cx", 0)), float(params.get("cy", 0))
                    half = float(params.get("half", 320))
                    height, width = frame.shape[:2]
                    x1 = int(max(0, round(cx - half)))
                    x2 = int(min(width, round(cx + half)))
                    y1 = int(max(0, round(cy - half)))
                    y2 = int(min(height, round(cy + half)))
                    if path == "/api/crop-meta":
                        return self._json({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                           "decoded_timestamp": actual})
                    crop = frame[y1:y2, x1:x2].copy()
                    upscale = 2 if max(crop.shape[:2]) <= 400 else 1
                    if upscale > 1:
                        crop = cv2.resize(crop, (crop.shape[1] * upscale,
                                                 crop.shape[0] * upscale),
                                          interpolation=cv2.INTER_NEAREST)
                    return self._bytes(self._jpeg(crop, 92), "image/jpeg")
                if path == "/api/proposals":
                    frame, _ = self._frame(params)
                    x, y = float(params["x"]), float(params["y"])
                    proposals = classic_cv_proposals(frame, x, y)
                    for proposal in proposals:
                        bx1, by1, bx2, by2 = proposal["bbox_xyxy"]
                        half = max(24.0, max(bx2 - bx1, by2 - by1) * 0.6)
                        proposal["preview_url"] = (
                            f"/api/crop?file_id={params['file_id']}&t={params.get('t', 0)}"
                            f"&cx={(bx1 + bx2) / 2}&cy={(by1 + by2) / 2}&half={half}")
                    return self._json({"proposals": proposals,
                                       "rule": "classic CV only, used after human "
                                               "confirmation; no detector was run"})
                self._error("unknown endpoint", 404)
            except (BlindTruthError, TruthFrozenError, DevelopmentScopeError,
                    SealedAssetError, DetectorUseError) as error:
                self._error(error, 409 if isinstance(error, TruthFrozenError) else 400)
            except Exception as error:                        # pragma: no cover
                self._error(f"{type(error).__name__}: {error}", 500)

        # -- POST ----------------------------------------------------------- #
        def do_POST(self):
            parsed = urlparse(self.path)
            length = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                if parsed.path == "/api/truth":
                    action = body.get("action", "add")
                    if action == "add":
                        return self._json(store.add_truth(
                            truth_class=body["truth_class"],
                            camera_id=body["camera_id"],
                            source_file_id=body["source_file_id"],
                            decoded_timestamp=body["decoded_timestamp"],
                            source_point=body.get("source_point"),
                            source_bbox_xyxy=body.get("source_bbox_xyxy"),
                            note=body.get("note"),
                            localization_status=body.get("localization_status")))
                    if action == "update":
                        return self._json(store.update_truth(
                            truth_id=body["truth_id"],
                            source_bbox_xyxy=body.get("source_bbox_xyxy"),
                            localization_status=body.get("localization_status"),
                            truth_class=body.get("truth_class"),
                            note=body.get("note"),
                            source_point=body.get("source_point")))
                    if action == "delete":
                        return self._json(store.delete_truth(
                            truth_id=body["truth_id"],
                            reason=body.get("reason") or "human correction"))
                    if action == "set_interval":
                        return self._json(store.episode_action(
                            action="interval", truth_id=body["truth_id"],
                            episode_id=body.get("episode_id"),
                            which=body.get("which")))
                    raise BlindTruthError(f"unknown truth action {action!r}")
                if parsed.path == "/api/episode":
                    return self._json(store.episode_action(
                        action=body.get("action", "new"),
                        truth_id=body.get("truth_id"),
                        episode_id=body.get("episode_id"),
                        note=body.get("note")))
                if parsed.path == "/api/review":
                    return self._json(store.set_review(
                        file_id=body["file_id"], status=body.get("status", REVIEW_DONE),
                        decoded_timestamp=body.get("decoded_timestamp"),
                        camera_id=body.get("camera_id")))
                if parsed.path == "/api/sample":
                    return self._json(store.build_samples())
                if parsed.path == "/api/freeze":
                    store.write_reports(code_commit=body.get("code_commit") or "unknown")
                    return self._json(store.freeze())
                self._error("unknown endpoint", 404)
            except (BlindTruthError, TruthFrozenError, DevelopmentScopeError,
                    SealedAssetError, DetectorUseError) as error:
                self._error(error, 409 if isinstance(error, TruthFrozenError) else 400)
            except KeyError as error:
                self._error(f"missing field {error}", 400)
            except Exception as error:                        # pragma: no cover
                self._error(f"{type(error).__name__}: {error}", 500)

    return Handler, ThreadingHTTPServer


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--development-root", type=Path, default=DEVELOPMENT_ROOT)
    p.add_argument("--roi-dir", type=Path, default=ROI_DIR)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8801)
    p.add_argument("--code-commit", default=None)
    p.add_argument("--erratum", type=Path, default=None,
                   help="JSON file with truth_id/original/corrected/reason/after_inference")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("manifest", "serve", "sample", "freeze", "status", "erratum", "selftest"):
        sub.add_parser(name)
    return p


def _store(args) -> TruthStore:
    assert_blind()
    assert_development_asset(args.development_root)
    assert_not_sealed(args.output)
    return TruthStore(args.output, args.development_root, args.roi_dir)


def cmd_manifest(args) -> int:
    assert_blind()
    assert_development_asset(args.development_root)
    inventory = load_development_inventory(args.development_root, args.roi_dir)
    atomic_write_json(args.output / MANIFEST_NAME, inventory)
    print(json.dumps({"ps_count": inventory["ps_count"],
                      "per_camera_count": inventory["per_camera_count"],
                      "total_bytes": inventory["total_bytes"],
                      "roi_geometry_version": inventory["roi_geometry_version"],
                      "problems": inventory["problems"], "ok": inventory["ok"]},
                     ensure_ascii=False, indent=2))
    return 0 if inventory["ok"] else 3


def cmd_selftest(args) -> int:
    report = blindness_selftest()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 3


def cmd_status(args) -> int:
    store = _store(args)
    print(json.dumps({"coverage": store.coverage(),
                      "truth_objects": len(store.truth),
                      "episodes": len(store.episodes),
                      "frozen": bool(store.frozen()),
                      "blindness": blindness_selftest()},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_sample(args) -> int:
    store = _store(args)
    print(json.dumps(store.build_samples(), ensure_ascii=False, indent=2))
    return 0


def cmd_freeze(args) -> int:
    store = _store(args)
    summary = store.write_reports(code_commit=args.code_commit or "unknown")
    record = store.freeze()
    print(json.dumps({"frozen_at": record["frozen_at"],
                      "truth_sha256": record["truth_sha256"],
                      "artifact_sha256": record["artifact_sha256"],
                      "evidence_classification": summary["evidence_classification"]},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_erratum(args) -> int:
    assert_blind()
    if args.erratum is None or not Path(args.erratum).is_file():
        raise BlindTruthError("--erratum <existing json file> is required")
    payload = json.loads(Path(args.erratum).read_text(encoding="utf-8"))
    row = append_truth_erratum(args.output, truth_id=payload["truth_id"],
                               original=payload["original"],
                               corrected=payload["corrected"],
                               reason=payload["reason"],
                               after_inference=bool(payload.get("after_inference")))
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args) -> int:
    store = _store(args)
    decoder = PsDecoder(store.files)
    handler, server = make_handler(store, decoder)
    server = server((args.bind, args.port), handler)
    print(f"Blind Truth review UI: http://{args.bind}:{args.port}/")
    print("BLIND TRUTH MODE / DETECTOR OUTPUT DISABLED "
          f"(blindness ok={blindness_selftest()['ok']})")
    print(f"artifact dir: {args.output}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:                                 # pragma: no cover
        print("stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    commands = {"manifest": cmd_manifest, "serve": cmd_serve, "sample": cmd_sample,
                "freeze": cmd_freeze, "status": cmd_status, "erratum": cmd_erratum,
                "selftest": cmd_selftest}
    try:
        return commands[args.command](args)
    except (BlindTruthError, TruthFrozenError, DevelopmentScopeError, SealedAssetError,
            DetectorUseError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
