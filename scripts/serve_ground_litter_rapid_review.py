#!/usr/bin/env python3
"""Ground Litter Rapid Eval v1 — isolated human review UI (Stage A + Stage B + localization).

Isolated by construction:

* own artifact root (default ``~/ground-litter-rapid-v1/artifact``), never the official
  ``/home/sf01/step2c1-blind-truth/artifact``;
* own port (default 8810) so the official 8801 review environment is never touched;
* Stage A (human truth discovery) is blind: the server refuses to serialise any model
  output for a frame whose ``truth_complete`` is false, not even a candidate count;
* every Rapid-Eval Holdout frame is refused by the training-export guard.

    python scripts/serve_ground_litter_rapid_review.py --artifact <root> serve --port 8810

Subcommands: ``serve``, ``status``, ``selftest``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtsp_annotator.ground_litter_rapid import (  # noqa: E402
    EVAL_SPLIT,
    FPS,
    PROPOSAL_FLOOR,
    SEMANTIC_SHA256,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    THRESHOLD_GRID,
    TILE,
    TRAIN_SPLIT,
    TRUTH_CLASSES,
    RapidError,
    assert_development_asset,
    assert_trainable,
    assert_writable_root,
    point_inside_roi,
    read_jsonl,
    sha256_file,
    tile_starts,
    verify_split,
    write_json,
)

#: Bounded search list for the Turhancan semantic model.  Located by SHA-256 only.
SEMANTIC_SEARCH_PATHS = (
    "/home/sf01/step2c1-blind-truth/exploratory-fourcam-20260929/turhancan_yolov8m_seg_trash.pt",
    "/home/sf01/ground_litter_train/models/turhancan_yolov8m_seg_trash.pt",
    "/home/sf01/ground-litter-pilot-20260911/ground_litter_shadow_20260913/models/litter/"
    "turhancan_yolov8m_seg_trash.pt",
    "/home/sf01/rtsp-deepstream/releases/ground-litter-20260915/models/litter/"
    "turhancan_yolov8m_seg_trash.pt",
)

VERDICTS = ("Y", "N", "X", "M")
STAGES = ("truth", "prediction", "localization", "done")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


class RapidReviewStore:
    """Atomic, resumable review state.  Every write replaces a file via a temp file."""

    def __init__(self, artifact: Path):
        self.artifact = assert_writable_root(artifact)
        self.review_dir = self.artifact / "review"
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.split = json.loads((self.artifact / "split.json").read_text(encoding="utf-8"))
        verify_split(self.split)
        self.frames = read_jsonl(self.artifact / "frame_manifest.jsonl")
        if not self.frames:
            raise RapidError("frame_manifest.jsonl is empty")
        self.frame_by_id = {row["frame_id"]: row for row in self.frames}
        self.file_row = {row["file_id"]: row for row in self.split["rows"]}
        extraction_path = self.artifact / "extraction_manifest.json"
        self.extraction = (json.loads(extraction_path.read_text(encoding="utf-8"))
                           if extraction_path.exists() else None)
        self.state: dict = {}
        self.points: list[dict] = []
        self.prediction_reviews: list[dict] = []
        self.localization_reviews: list[dict] = []
        self.localization_candidates: dict = {}
        self._predictions_raw: dict[str, list[dict]] = {}
        self.predictions_path: Path | None = None
        self._load()

    # -- files ------------------------------------------------------------- #
    def _path(self, name: str) -> Path:
        return self.review_dir / name

    def _load(self) -> None:
        with self.lock:
            state_path = self._path("review_state.json")
            self.state = (json.loads(state_path.read_text(encoding="utf-8"))
                          if state_path.exists() else {"frames": {}})
            self.state.setdefault("frames", {})
            self.points = read_jsonl(self._path("truth_points.jsonl"))
            self.prediction_reviews = read_jsonl(self._path("prediction_reviews.jsonl"))
            candidate_path = self._path("localization_candidates.json")
            self.localization_candidates = (
                json.loads(candidate_path.read_text(encoding="utf-8"))
                if candidate_path.exists() else {})
            self._materialise_inherited_points()

    def _materialise_inherited_points(self) -> None:
        """Seed bonus train frames with the human points already recorded officially."""
        existing = {(p["frame_id"], tuple(p["source_xy"])) for p in self.points}
        added = False
        inherited_dir = self.artifact / "inherited_truth"
        for row in self.frames:
            if row["kind"] != "bonus_train":
                continue
            source = inherited_dir / f'{row["frame_id"]}.jsonl'
            if not source.exists():
                continue
            for mark in read_jsonl(source):
                key = (row["frame_id"], tuple(mark["source_xy"]))
                if key in existing:
                    continue
                self.points.append({
                    "truth_id": mark["truth_id"],
                    "frame_id": row["frame_id"],
                    "camera_id": row["camera_id"],
                    "file_id": row["file_id"],
                    "source_xy": list(mark["source_xy"]),
                    "truth_class": mark["truth_class"],
                    "in_roi": bool(mark.get("in_roi", True)),
                    "created_at": mark.get("created_at") or now_iso(),
                    "origin": "official_discovery_inherited",
                    "note": mark.get("note"),
                })
                existing.add(key)
                added = True
        if added:
            self._save_points()

    def _save_json(self, name: str, payload) -> None:
        write_json(self._path(name), payload)

    def _save_jsonl(self, name: str, rows: list[dict]) -> None:
        path = self._path(name)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)

    def _save_points(self) -> None:
        self._save_jsonl("truth_points.jsonl", self.points)

    def _save_prediction_reviews(self) -> None:
        self._save_jsonl("prediction_reviews.jsonl", self.prediction_reviews)

    def _save_localization_reviews(self) -> None:
        self._save_jsonl("localization_reviews.jsonl", self.localization_reviews)

    def _save_state(self) -> None:
        self.state["updated_at"] = now_iso()
        self._save_json("review_state.json", self.state)

    # -- frame helpers ----------------------------------------------------- #
    def frame_state(self, frame_id: str) -> dict:
        entry = self.state["frames"].setdefault(frame_id, {})
        entry.setdefault("truth_complete", False)
        return entry

    def points_for(self, frame_id: str) -> list[dict]:
        return [p for p in self.points if p["frame_id"] == frame_id]

    def reviews_for(self, frame_id: str) -> dict[str, str]:
        return {r["prediction_id"]: r["verdict"] for r in self.prediction_reviews
                if r["frame_id"] == frame_id}

    def localization_for(self, frame_id: str) -> dict[str, dict]:
        return {r["truth_id"]: r for r in self.localization_reviews
                if r["frame_id"] == frame_id}

    def predictions_for(self, frame_id: str) -> list[dict]:
        """Model output — only ever returned for an explicitly completed truth frame."""
        entry = self.state["frames"].get(frame_id) or {}
        if not entry.get("truth_complete"):
            raise RapidError(
                f"blindness violation: predictions requested for {frame_id} before "
                f"truth_complete"
            )
        return self._predictions_raw.get(frame_id, [])

    def load_predictions(self, path: Path | None = None) -> None:
        path = path or (self.artifact / "baseline" / "predictions.jsonl")
        self._predictions_raw = {}
        if path.exists():
            for row in read_jsonl(path):
                self._predictions_raw[row["frame_id"]] = row["predictions"]
        self.predictions_path = path

    # -- mutations --------------------------------------------------------- #
    def add_point(self, frame_id: str, truth_class: str, x: float, y: float,
                  note: str | None = None) -> dict:
        with self.lock:
            frame = self.frame_by_id.get(frame_id)
            if frame is None:
                raise RapidError(f"unknown frame {frame_id}")
            if truth_class not in TRUTH_CLASSES:
                raise RapidError(f"unknown truth class {truth_class}")
            row = self.file_row[frame["file_id"]]
            x, y = float(x), float(y)
            if not (0 <= x < SOURCE_WIDTH and 0 <= y < SOURCE_HEIGHT):
                raise RapidError(f"point outside the source canvas: ({x}, {y})")
            point = {
                "truth_id": f't-{len(self.points) + 1:05d}',
                "frame_id": frame_id,
                "camera_id": frame["camera_id"],
                "file_id": frame["file_id"],
                "source_xy": [round(x, 2), round(y, 2)],
                "truth_class": truth_class,
                "in_roi": point_inside_roi(x, y, row["roi"]),
                "created_at": now_iso(),
                "origin": "rapid_v1_review",
                "note": note,
            }
            self.points.append(point)
            self._save_points()
            entry = self.frame_state(frame_id)
            entry["truth_complete"] = False
            entry["truth_updated_at"] = now_iso()
            self._save_state()
            return point

    def delete_point(self, truth_id: str) -> dict:
        with self.lock:
            found = [p for p in self.points if p["truth_id"] == truth_id]
            if not found:
                raise RapidError(f"unknown truth_id {truth_id}")
            self.points = [p for p in self.points if p["truth_id"] != truth_id]
            self.localization_reviews = [r for r in self.localization_reviews
                                         if r["truth_id"] != truth_id]
            self._save_points()
            self._save_localization_reviews()
            self.localization_candidates.pop(truth_id, None)
            self._save_json("localization_candidates.json", self.localization_candidates)
            return found[0]

    def complete_truth(self, frame_id: str, complete: bool = True) -> dict:
        with self.lock:
            if frame_id not in self.frame_by_id:
                raise RapidError(f"unknown frame {frame_id}")
            entry = self.frame_state(frame_id)
            entry["truth_complete"] = bool(complete)
            entry["truth_completed_at"] = now_iso() if complete else None
            if not complete:
                entry["prediction_review_complete"] = False
            self._save_state()
            return entry

    def review_prediction(self, frame_id: str, prediction_id: str, verdict: str) -> dict:
        with self.lock:
            if verdict not in VERDICTS:
                raise RapidError(f"unknown verdict {verdict}")
            entry = self.frame_state(frame_id)
            if not entry.get("truth_complete"):
                raise RapidError(f"refusing to review predictions for {frame_id}: "
                                 f"truth is not complete")
            frame = self.frame_by_id[frame_id]
            self.prediction_reviews = [r for r in self.prediction_reviews
                                       if r["prediction_id"] != prediction_id]
            row = {"frame_id": frame_id, "prediction_id": prediction_id, "verdict": verdict,
                   "camera_id": frame["camera_id"], "file_id": frame["file_id"],
                   "split": frame["split"], "created_at": now_iso()}
            self.prediction_reviews.append(row)
            if verdict == "M":
                entry["truth_complete"] = False
                entry["m_reopened_at"] = now_iso()
                self._save_state()
            self._save_prediction_reviews()
            return row

    def select_localization(self, truth_id: str, choice: int) -> dict:
        with self.lock:
            candidates = self.localization_candidates.get(truth_id) or {}
            options = candidates.get("candidates") or []
            entry = {"truth_id": truth_id, "choice": int(choice), "created_at": now_iso()}
            if int(choice) == 0:
                entry.update({"status": "UNLOCALIZED_SKIP",
                              "selected_bbox_source_xyxy": None,
                              "proposal_source": None, "proposal_id": None})
            else:
                match = next((c for c in options if int(c["idx"]) == int(choice) - 1), None)
                if match is None:
                    raise RapidError(f"candidate {choice} unavailable for {truth_id}")
                entry.update({"status": "LOCALIZED",
                              "selected_bbox_source_xyxy": match["bbox_xyxy"],
                              "proposal_source": match["proposal_source"],
                              "proposal_id": match["proposal_id"]})
            point = next((p for p in self.points if p["truth_id"] == truth_id), None)
            if point:
                entry.update({"frame_id": point["frame_id"],
                              "camera_id": point["camera_id"],
                              "file_id": point["file_id"],
                              "split": self.frame_by_id[point["frame_id"]]["split"]})
            self.localization_reviews = [r for r in self.localization_reviews
                                         if r["truth_id"] != truth_id]
            self.localization_reviews.append(entry)
            self._save_localization_reviews()
            return entry

    # -- progress ---------------------------------------------------------- #
    def progress(self) -> dict:
        fixed = [f for f in self.frames if f["kind"] == "fixed"]
        bonus = [f for f in self.frames if f["kind"] == "bonus_train"]
        complete = [f for f in self.frames if self.frame_state(f["frame_id"])["truth_complete"]]
        required = [p for p in self.points if p["truth_class"] == "REQUIRED_LITTER"]
        ignore = [p for p in self.points if p["truth_class"] == "IGNORE_SMALL"]
        uncertain = [p for p in self.points if p["truth_class"] == "UNCERTAIN"]
        train_frames = [f for f in self.frames if f["split"] == TRAIN_SPLIT]
        eval_frames = [f for f in self.frames if f["split"] == EVAL_SPLIT]
        total_predictions = 0
        reviewed_predictions = 0
        for frame in complete:
            total_predictions += len(self._predictions_raw.get(frame["frame_id"], []))
        reviewed_predictions = len({r["prediction_id"] for r in self.prediction_reviews})
        localizable = [p for p in required
                       if self.frame_by_id[p["frame_id"]]["split"] == TRAIN_SPLIT
                       and self.frame_state(p["frame_id"])["truth_complete"]]
        localized = [r for r in self.localization_reviews
                     if r.get("status") in ("LOCALIZED", "UNLOCALIZED_SKIP")]
        return {
            "truth_review": {
                "complete": len(complete),
                "total": len(self.frames),
                "fixed_total": len(fixed),
                "bonus_total": len(bonus),
            },
            "rapid_train": {
                "complete": sum(1 for f in train_frames
                                if self.frame_state(f["frame_id"])["truth_complete"]),
                "fixed_total": sum(1 for f in train_frames if f["kind"] == "fixed"),
                "bonus_total": sum(1 for f in train_frames if f["kind"] == "bonus_train"),
            },
            "rapid_eval": {
                "complete": sum(1 for f in eval_frames
                                if self.frame_state(f["frame_id"])["truth_complete"]),
                "total": len(eval_frames),
            },
            "truth_points": {
                "required": len(required),
                "ignore": len(ignore),
                "uncertain": len(uncertain),
                "total": len(self.points),
            },
            "prediction_review": {
                "reviewed": reviewed_predictions,
                "available": total_predictions,
                "note": "prediction counts are only included for frames whose truth is complete",
            },
            "localization": {
                "selected": len(localized),
                "required_on_train_complete": len(localizable),
                "localized": sum(1 for r in localized if r.get("status") == "LOCALIZED"),
                "skipped": sum(1 for r in localized if r.get("status") == "UNLOCALIZED_SKIP"),
            },
            "inference": {
                "predictions_loaded": bool(self._predictions_raw),
                "predictions_file": str(getattr(self, "predictions_path", "")),
            },
        }

    def frame_payload(self, frame_id: str, *, localizer=None) -> dict:
        frame = self.frame_by_id.get(frame_id)
        if frame is None:
            raise RapidError(f"unknown frame {frame_id}")
        entry = self.frame_state(frame_id)
        row = self.file_row[frame["file_id"]]
        payload: dict = {
            "frame": {
                "frame_id": frame_id,
                "camera_id": frame["camera_id"],
                "file_id": frame["file_id"],
                "split": frame["split"],
                "kind": frame["kind"],
                "offset_seconds": frame["offset_seconds"],
                "requested_relative_seconds": frame["requested_relative_seconds"],
                "nominal_frame_index": frame["nominal_frame_index"],
                "width": SOURCE_WIDTH,
                "height": SOURCE_HEIGHT,
                "roi": row["roi"],
                "roi_geometry_version": row["roi_geometry_version"],
                "record_start": row["record_start"],
                "inherited_truth_ids": frame.get("inherited_truth_ids") or [],
            },
            "stage": self.stage_of(frame_id),
            "truth_complete": bool(entry.get("truth_complete")),
            "truth_points": redact_points(self.points_for(frame_id)),
            "localization_required": row["split"] == TRAIN_SPLIT,
        }
        if not entry.get("truth_complete"):
            # Stage A blindness: no model output, not even a count, leaves the server.
            payload["predictions"] = None
            payload["prediction_reviews"] = None
            payload["localization_candidates"] = None
        else:
            predictions = self._predictions_raw.get(frame_id, [])
            payload["predictions"] = [
                {"prediction_id": p["prediction_id"], "xyxy": p["xyxy"],
                 "class_name": p.get("class_name")}
                for p in predictions
            ]
            payload["prediction_reviews"] = self.reviews_for(frame_id)
            payload["localization_candidates"] = {
                p["truth_id"]: self.localization_candidates.get(p["truth_id"])
                for p in self.points_for(frame_id)
                if p["truth_class"] == "REQUIRED_LITTER"
            }
            payload["localization_reviews"] = self.localization_for(frame_id)
            payload["prediction_count"] = len(predictions)
        return payload

    def stage_of(self, frame_id: str) -> str:
        entry = self.frame_state(frame_id)
        if not entry.get("truth_complete"):
            return "truth"
        predictions = self._predictions_raw.get(frame_id, [])
        reviews = self.reviews_for(frame_id)
        if any(p["prediction_id"] not in reviews for p in predictions):
            return "prediction"
        frame = self.frame_by_id[frame_id]
        if frame["split"] == TRAIN_SPLIT:
            required = [p for p in self.points_for(frame_id)
                        if p["truth_class"] == "REQUIRED_LITTER"]
            done = self.localization_for(frame_id)
            if any(p["truth_id"] not in done for p in required):
                return "localization"
        return "done"

    # -- export guard ------------------------------------------------------ #
    def train_export_probe(self, frame_id: str) -> dict:
        """Exercises the real training-export guard end to end."""
        frame = self.frame_by_id.get(frame_id)
        if frame is None:
            raise RapidError(f"unknown frame {frame_id}")
        assert_trainable(frame["split"], sample_id=frame_id)
        return {"exported": True, "frame_id": frame_id, "split": frame["split"]}


def redact_points(points: list[dict]) -> list[dict]:
    return [{"truth_id": p["truth_id"], "source_xy": p["source_xy"],
             "truth_class": p["truth_class"], "in_roi": p["in_roi"],
             "origin": p.get("origin"), "created_at": p.get("created_at")}
            for p in points]


# --------------------------------------------------------------------------- #
# Localization candidates (A: baseline Y box, B: semantic, C: classical CV)
# --------------------------------------------------------------------------- #


class Localizer:
    def __init__(self, artifact: Path, store: "RapidReviewStore", *, enable_semantic=True):
        self.artifact = artifact
        self.store = store
        self._semantic = None
        self._semantic_path = None
        self.semantic_note = None
        if enable_semantic:
            self._locate_semantic()
        else:
            self.semantic_note = "semantic_disabled"

    def _locate_semantic(self) -> None:
        for path in SEMANTIC_SEARCH_PATHS:
            candidate = Path(path)
            if not candidate.exists():
                continue
            try:
                if sha256_file(candidate) == SEMANTIC_SHA256:
                    self._semantic_path = candidate
                    return
            except OSError:
                continue
        self.semantic_note = "semantic_model_not_found_by_sha256_no_download"

    def _semantic_model(self):
        if self._semantic is None:
            if self._semantic_path is None:
                return None
            from ultralytics import YOLO
            self._semantic = YOLO(str(self._semantic_path))
        return self._semantic

    def candidates(self, truth_id: str, *, force: bool = False) -> dict:
        if not force:
            cached = self.store.localization_candidates.get(truth_id)
            if cached:
                return cached
        point = next((p for p in self.store.points if p["truth_id"] == truth_id), None)
        if point is None:
            raise RapidError(f"unknown truth_id {truth_id}")
        frame_id = point["frame_id"]
        frame_row = self.store.frame_by_id[frame_id]
        file_row = self.store.file_row[frame_row["file_id"]]
        x, y = point["source_xy"]
        options: list[dict] = []

        # A — a baseline prediction the operator judged Y that contains the point.
        reviews = self.store.reviews_for(frame_id)
        for prediction in self.store._predictions_raw.get(frame_id, []):
            if reviews.get(prediction["prediction_id"]) != "Y":
                continue
            box = prediction["xyxy"]
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                options.append({"proposal_source": "baseline_judged_Y",
                                "proposal_id": prediction["prediction_id"],
                                "bbox_xyxy": box,
                                "contains_point": True,
                                "confidence": prediction.get("confidence")})
                break

        # B — Turhancan semantic proposal near the point (Rapid-Train frames only).
        if frame_row["split"] == TRAIN_SPLIT:
            semantic = self._semantic_candidate(frame_id, file_row, x, y)
            if semantic:
                options.append(semantic)

        # C — classical point-seeded proposal (Step 1C logic, unchanged).
        classical = self._classical_candidate(frame_id, file_row, x, y)
        if classical:
            options.append(classical)

        options = options[:3]
        labels = "ABC"
        for index, option in enumerate(options):
            option["idx"] = index
            option["label"] = labels[index]
        payload = {
            "truth_id": truth_id,
            "frame_id": frame_id,
            "split": frame_row["split"],
            "point": [x, y],
            "candidates": options,
            "semantic_model": str(self._semantic_path) if self._semantic_path else None,
            "semantic_note": self.semantic_note,
            "computed_at": now_iso(),
        }
        self.store.localization_candidates[truth_id] = payload
        self.store._save_json("localization_candidates.json",
                              self.store.localization_candidates)
        return payload

    def _frame_image(self, frame_id: str):
        import cv2
        record = next((r for r in (self.store.extraction or {}).get("records", [])
                       if r["frame_id"] == frame_id), None)
        if record is None:
            return None
        path = self.artifact / "frames" / record["image"]
        return cv2.imread(str(path), cv2.IMREAD_COLOR) if path.exists() else None

    def _semantic_candidate(self, frame_id: str, file_row: dict, x: float, y: float):
        model = self._semantic_model()
        if model is None:
            return None
        import cv2
        import numpy as np
        frame = self._frame_image(frame_id)
        if frame is None:
            return None
        roi = file_row["roi"]
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        polygon = np.array([(round(float(u) * SOURCE_WIDTH), round(float(v) * SOURCE_HEIGHT))
                            for u, v in roi], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        boxes: list[dict] = []
        for tile_y in tile_starts(SOURCE_HEIGHT):
            for tile_x in tile_starts(SOURCE_WIDTH):
                if not (tile_x <= x < tile_x + TILE and tile_y <= y < tile_y + TILE):
                    continue
                if not np.any(mask[tile_y:tile_y + TILE, tile_x:tile_x + TILE]):
                    continue
                result = model.predict(frame[tile_y:tile_y + TILE, tile_x:tile_x + TILE],
                                       imgsz=TILE, conf=0.10, iou=0.7, device="cpu",
                                       verbose=False)[0]
                for box in result.boxes:
                    local = box.xyxy[0].cpu().tolist()
                    coords = [round(local[0] + tile_x, 2), round(local[1] + tile_y, 2),
                              round(local[2] + tile_x, 2), round(local[3] + tile_y, 2)]
                    boxes.append({"xyxy": coords, "confidence": round(float(box.conf[0]), 5)})
        if not boxes:
            return None
        containing = [b for b in boxes if b["xyxy"][0] <= x <= b["xyxy"][2]
                      and b["xyxy"][1] <= y <= b["xyxy"][3]]
        pool = containing or boxes
        best = max(pool, key=lambda b: b["confidence"])
        return {"proposal_source": "semantic_turhancan",
                "proposal_id": f'semantic_{best["confidence"]:.3f}',
                "bbox_xyxy": best["xyxy"],
                "contains_point": bool(containing),
                "confidence": best["confidence"]}

    def _classical_candidate(self, frame_id: str, file_row: dict, x: float, y: float):
        frame = self._frame_image(frame_id)
        if frame is None:
            return None
        proposals = classic_cv_proposals(frame, x, y)
        if not proposals:
            return None
        best = next((p for p in proposals if p["contains_point"]), None) or proposals[0]
        return {"proposal_source": "classical_point_seeded",
                "proposal_id": best["label"],
                "bbox_xyxy": best["bbox_xyxy"],
                "contains_point": bool(best["contains_point"]),
                "area": best["area"]}

    def crop_jpeg(self, truth_id: str, idx: int, *, scale: int = 4,
                  half: int = 96, quality: int = 88):
        import cv2
        import numpy as np
        payload = self.store.localization_candidates.get(truth_id)
        if payload is None:
            payload = self.candidates(truth_id, force=True)
        options = payload["candidates"]
        if idx < 0 or idx >= len(options):
            raise RapidError(f"candidate {idx} unavailable for {truth_id}")
        option = options[idx]
        frame = self._frame_image(payload["frame_id"])
        if frame is None:
            raise RapidError("frame image unavailable")
        x, y = payload["point"]
        box = option["bbox_xyxy"]
        span = max(box[2] - box[0], box[3] - box[1], 24.0)
        half = int(max(24, min(180, span * 0.9 + 16)))
        x0 = int(max(0, min(SOURCE_WIDTH - 1, round(x - half))))
        x1 = int(max(1, min(SOURCE_WIDTH, round(x + half))))
        y0 = int(max(0, min(SOURCE_HEIGHT - 1, round(y - half))))
        y1 = int(max(1, min(SOURCE_HEIGHT, round(y + half))))
        crop = frame[y0:y1, x0:x1].copy()
        if crop.size == 0:
            raise RapidError("empty crop")
        bx0, by0 = int(round(box[0])) - x0, int(round(box[1])) - y0
        bx1, by1 = int(round(box[2])) - x0, int(round(box[3])) - y0
        cv2.rectangle(crop, (bx0, by0), (bx1, by1), (0, 255, 0), 1)
        centre = (int(round(x)) - x0, int(round(y)) - y0)
        cv2.drawMarker(crop, centre, (0, 0, 255), cv2.MARKER_CROSS, 9, 1)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RapidError("cannot encode candidate crop")
        return buffer.tobytes()


def classic_cv_proposals(frame, x: float, y: float, *, half: int = 96) -> list[dict]:
    """Step 1C classical point-seeded proposals — reused verbatim, never discovers."""
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
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
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
        bx, by, bw, bh, area = best[1]
        proposals.append({
            "label": label,
            "bbox_xyxy": [float(bx + x0), float(by + y0),
                          float(bx + x0 + bw), float(by + y0 + bh)],
            "area": int(area), "contains_point": bool(best[2]), "candidate_half": half,
        })
    return proposals


# --------------------------------------------------------------------------- #
# Frame image service
# --------------------------------------------------------------------------- #


class FrameImages:
    def __init__(self, artifact: Path, store: "RapidReviewStore"):
        self.artifact = artifact
        self.store = store
        self.cache = artifact / "ui_cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.records = {r["frame_id"]: r for r in (store.extraction or {}).get("records", [])}

    def display_jpeg(self, frame_id: str, quality: int = 92) -> bytes:
        import cv2
        record = self.records.get(frame_id)
        if record is None:
            raise RapidError(f"frame not extracted: {frame_id}")
        cached = self.cache / f"{frame_id}.q{quality}.jpg"
        if cached.exists():
            return cached.read_bytes()
        source = self.artifact / "frames" / record["image"]
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            raise RapidError(f"cannot read {source}")
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RapidError("cannot encode display frame")
        data = buffer.tobytes()
        # The cache directory may have been removed while the service was running; never let
        # that turn into a 500 on an otherwise healthy frame request.
        self.cache.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(cached.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(cached)
        return data

    def source_crop_png(self, frame_id: str, cx: float, cy: float, *, half: int = 28,
                        scale: int = 4) -> bytes:
        """Lossless source-native crop for the magnifier.

        The overview image is a JPEG, and upscaling a JPEG region 4x is exactly where an 8 px
        target would be misjudged.  The magnifier therefore reads the source PNG instead.
        """
        import cv2
        record = self.records.get(frame_id)
        if record is None:
            raise RapidError(f"frame not extracted: {frame_id}")
        source = self.artifact / "frames" / record["image"]
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RapidError(f"cannot read {source}")
        height, width = image.shape[:2]
        half = max(4, min(200, int(half)))
        scale = max(1, min(8, int(scale)))
        x0 = int(max(0, min(width - 1, round(cx) - half)))
        x1 = int(max(1, min(width, round(cx) + half)))
        y0 = int(max(0, min(height - 1, round(cy) - half)))
        y1 = int(max(1, min(height, round(cy) + half)))
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            raise RapidError("empty source crop")
        centre = (int(round(cx)) - x0, int(round(cy)) - y0)
        cv2.drawMarker(crop, centre, (0, 0, 255), cv2.MARKER_CROSS, 9, 1)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ok, buffer = cv2.imencode(".png", crop)
        if not ok:
            raise RapidError("cannot encode source crop")
        return buffer.tobytes()


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def make_handler(store: RapidReviewStore, images: FrameImages, localizer: Localizer,
                 *, repo_root: Path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    class Handler(BaseHTTPRequestHandler):
        server_version = "RapidReview/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            sys.stderr.write("rapid-ui: " + (fmt % args) + "\n")

        def _send(self, status, body: bytes, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status=200):
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        def _error(self, message, status=400):
            self._json({"error": str(message)}, status)

        def _params(self):
            parsed = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            params["__path"] = parsed.path
            return params

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

        def do_GET(self):                                        # noqa: N802
            params = self._params()
            path = params["__path"]
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                if path == "/api/frames":
                    return self._json({
                        "frames": [{"frame_id": f["frame_id"], "camera_id": f["camera_id"],
                                    "split": f["split"], "kind": f["kind"],
                                    "offset_seconds": f["offset_seconds"]}
                                   for f in store.frames],
                        "seed": store.split["seed"],
                        "split_sha256": store.split["split_sha256"],
                    })
                if path == "/api/state":
                    payload = store.frame_payload(params.get("frame_id", ""),
                                                  localizer=localizer)
                    return self._json(payload)
                if path == "/api/progress":
                    return self._json(store.progress())
                if path == "/api/frame_image":
                    data = images.display_jpeg(params.get("frame_id", ""),
                                               int(params.get("q", 92)))
                    return self._send(200, data, "image/jpeg")
                if path == "/api/source_crop":
                    data = images.source_crop_png(
                        params.get("frame_id", ""), float(params.get("cx", 0)),
                        float(params.get("cy", 0)),
                        half=int(params.get("half", 28)),
                        scale=int(params.get("scale", 4)))
                    return self._send(200, data, "image/png")
                if path == "/api/localize":
                    return self._json(localizer.candidates(params.get("truth_id", "")))
                if path == "/api/candidate_image":
                    data = localizer.crop_jpeg(params.get("truth_id", ""),
                                               int(params.get("idx", 0)),
                                               scale=int(params.get("scale", 4)))
                    return self._send(200, data, "image/jpeg")
                if path == "/api/health":
                    return self._json({"ok": True, "artifact": str(store.artifact),
                                       "frames": len(store.frames),
                                       "predictions_loaded": bool(store._predictions_raw)})
                return self._error(f"unknown path {path}", 404)
            except RapidError as error:
                return self._error(error, 409)
            except Exception as error:                            # noqa: BLE001
                return self._error(f"{type(error).__name__}: {error}", 500)

        def do_POST(self):                                       # noqa: N802
            params = self._params()
            path = params["__path"]
            try:
                body = self._body()
                if path == "/api/truth":
                    point = store.add_point(body["frame_id"], body["truth_class"],
                                            body["x"], body["y"], body.get("note"))
                    return self._json({"point": point, "progress": store.progress()})
                if path == "/api/truth_delete":
                    removed = store.delete_point(body["truth_id"])
                    return self._json({"removed": removed, "progress": store.progress()})
                if path == "/api/truth_complete":
                    entry = store.complete_truth(body["frame_id"], body.get("complete", True))
                    return self._json({"entry": entry, "stage": store.stage_of(body["frame_id"]),
                                       "progress": store.progress()})
                if path == "/api/prediction_review":
                    row = store.review_prediction(body["frame_id"], body["prediction_id"],
                                                  body["verdict"])
                    return self._json({"review": row, "stage": store.stage_of(body["frame_id"]),
                                       "progress": store.progress()})
                if path == "/api/localize":
                    return self._json(localizer.candidates(body["truth_id"]))
                if path == "/api/localize_select":
                    entry = store.select_localization(body["truth_id"], body["choice"])
                    return self._json({"selection": entry,
                                       "stage": store.stage_of(entry["frame_id"]),
                                       "progress": store.progress()})
                if path == "/api/train_export_probe":
                    return self._json(store.train_export_probe(body["frame_id"]))
                return self._error(f"unknown path {path}", 404)
            except RapidError as error:
                return self._error(error, 409)
            except KeyError as error:
                return self._error(f"missing field {error}", 400)
            except Exception as error:                            # noqa: BLE001
                return self._error(f"{type(error).__name__}: {error}", 500)

    return Handler


# --------------------------------------------------------------------------- #
# UI page
# --------------------------------------------------------------------------- #

PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>Ground Litter Rapid Eval v1 — Review</title>
<style>
 :root{--bg:#14161a;--panel:#1d2027;--line:#2c313c;--fg:#e8eaee;--dim:#98a0ae;
       --req:#ff4d4d;--ign:#4da3ff;--unc:#ffb84d;--ok:#3ddc84;--box:#00e0a0;}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.45 -apple-system,
      "Helvetica Neue",Arial,"PingFang SC",sans-serif;overflow:hidden}
 #top{display:flex;align-items:center;gap:14px;padding:7px 12px;background:var(--panel);
      border-bottom:1px solid var(--line)}
 .badge{padding:2px 9px;border-radius:10px;font-weight:700;font-size:12px}
 .train{background:#1d3b2a;color:#7ff0ab;border:1px solid #2f6b46}
 .eval{background:#3b2a1d;color:#ffc98a;border:1px solid #6b4a2f}
 .stage{padding:2px 9px;border-radius:10px;background:#2a2f3a;color:#cfd6e2}
 .stage.on{background:#3a2f5c;color:#cbb8ff;border:1px solid #5c4a8f}
 #top .grow{flex:1}
 #top b{color:#fff}
 #wrap{display:flex;height:calc(100vh - 40px)}
 #left{flex:1;display:flex;flex-direction:column;min-width:0}
 #canvasHost{flex:1;position:relative;background:#0b0c0f;overflow:hidden}
 canvas#view{position:absolute;inset:0;cursor:crosshair}
 #right{width:320px;border-left:1px solid var(--line);background:var(--panel);
        display:flex;flex-direction:column;overflow-y:auto}
 .sec{padding:9px 11px;border-bottom:1px solid var(--line)}
 .sec h3{margin:0 0 7px;font-size:11px;letter-spacing:.09em;color:var(--dim);
         text-transform:uppercase}
 button{background:#2b3140;color:var(--fg);border:1px solid var(--line);border-radius:6px;
        padding:5px 9px;cursor:pointer;font-size:12px;font-family:inherit}
 button:hover{background:#384152}
 button.act{outline:2px solid #7a6cff;background:#3a3363}
 button.r{border-color:#7a2b2b}button.r.act{background:#5c2222}
 button.i{border-color:#2b4a7a}button.i.act{background:#1f3b63}
 button.u{border-color:#7a5c2b}button.u.act{background:#5c4520}
 .row{display:flex;gap:6px;flex-wrap:wrap}
 .muted{color:var(--dim)}
 .pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;
       background:#2b3140;margin:1px 3px 1px 0}
 .pill.r{background:#5c2222}.pill.i{background:#1f3b63}.pill.u{background:#5c4520}
 #loupe{position:absolute;right:10px;bottom:10px;width:216px;height:216px;
        border:2px solid #6a7385;background:#000;border-radius:4px;z-index:5}
 #loupeLabel{position:absolute;right:12px;bottom:236px;font-size:11px;color:#cfd6e2;
             background:#0009;padding:1px 5px;border-radius:3px}
 #help{font-size:11px;color:var(--dim);white-space:pre-line}
 #cands{display:flex;flex-direction:column;gap:6px}
 .cand{border:2px solid var(--line);border-radius:6px;padding:4px;background:#12141a;
       cursor:pointer;display:flex;gap:8px;align-items:center}
 .cand:hover{border-color:#5c6a8f}
 .cand.act{border-color:#7a6cff}
 .cand img{display:block;width:148px;height:148px;flex:0 0 148px;
           image-rendering:pixelated;object-fit:contain;background:#000;border-radius:3px}
 .candmeta{font-size:11px;line-height:1.5;min-width:0}
 .candmeta b{color:#fff;font-size:13px}
 .candmeta small{display:block;color:var(--dim);font-size:10px;word-break:break-all}
 #toast{position:fixed;left:50%;top:52px;transform:translateX(-50%);background:#000c;
        border:1px solid var(--line);padding:6px 13px;border-radius:6px;display:none;
        font-size:12px;z-index:50}
 .bar{height:7px;background:#22262f;border-radius:4px;overflow:hidden;margin:3px 0}
 .bar>i{display:block;height:100%;background:#5a6cff}
 #bottom{display:flex;gap:8px;align-items:center;padding:6px 11px;border-top:1px solid var(--line);
         background:var(--panel)}
 #bottom .grow{flex:1}
 table.k{width:100%;border-collapse:collapse;font-size:11px}
 table.k td{padding:1px 0}
 table.k td:last-child{text-align:right;color:#fff}
</style></head>
<body>
<div id="top">
  <span id="splitBadge" class="badge train">RAPID-TRAIN</span>
  <span class="stage" id="stageBadge">STAGE A — TRUTH</span>
  <span id="frameLabel" class="muted"></span>
  <span class="grow"></span>
  <span class="muted" id="progressLabel"></span>
</div>
<div id="wrap">
  <div id="left">
    <div id="canvasHost">
      <canvas id="view"></canvas>
      <canvas id="loupe" width="216" height="216"></canvas>
      <div id="loupeLabel">4x source-native</div>
    </div>
    <div id="bottom">
      <button id="btnPrev">← 上一帧</button>
      <button id="btnNext">下一帧 →</button>
      <button id="btnUnfinished">下一个未完成</button>
      <span class="grow"></span>
      <span class="muted">ROI 内所有正常可见、具有清理意义的垃圾都要点 (R)</span>
      <button id="btnZoomOut">−</button><span id="zoomLabel" class="muted">fit</span>
      <button id="btnZoomIn">+</button><button id="btnFit">fit</button>
    </div>
  </div>
  <div id="right">
    <div class="sec">
      <h3>Stage A — 人工真值</h3>
      <div class="row">
        <button class="r" id="mR">R 必需垃圾</button>
        <button class="i" id="mI">I 微小忽略</button>
        <button class="u" id="mU">U 不确定</button>
      </div>
      <div class="row" style="margin-top:7px">
        <button id="btnDone">本帧真值确认完成 (Enter)</button>
        <button id="btnNoTarget">无目标 (N)</button>
      </div>
      <div id="pointsList" style="margin-top:7px"></div>
    </div>
    <div class="sec" id="secPred" style="display:none">
      <h3>Stage B — Prediction Review</h3>
      <div id="predInfo" class="muted"></div>
      <div class="row" style="margin-top:7px">
        <button id="pY">Y 正确 (Y)</button>
        <button id="pX">X ignore (X)</button>
        <button id="pN">F/N 误报</button>
        <button id="pM">M 漏标→回A</button>
      </div>
      <div id="predList" style="margin-top:7px"></div>
    </div>
    <div class="sec" id="secLoc" style="display:none">
      <h3>Stage C — Localization (候选框)</h3>
      <div id="locInfo" class="muted">只选 1/2/3/0，不要画框</div>
      <div id="cands" style="margin-top:7px"></div>
      <div class="row" style="margin-top:7px">
        <button id="locNone">0 = None / UNLOCALIZED_SKIP</button>
      </div>
    </div>
    <div class="sec">
      <h3>进度</h3>
      <table class="k" id="progTable"></table>
    </div>
    <div class="sec">
      <h3>快捷键</h3>
      <div id="help">R/I/U 选择真值类别，点击=落点
N 无目标；Enter 本帧真值完成
Stage B: Y 正确 · X ignore · F 或 N 误报 · M 回 Stage A
Stage C: 1 / 2 / 3 选候选 · 0 = None
← → 上一帧 / 下一帧 · 滚轮缩放 · 拖拽平移
Shift+点击 = 强制删除最近的点</div>
    </div>
  </div>
</div>
<div id="toast"></div>
<script>
const S = {frames:[], idx:0, state:null, image:null, mode:'REQUIRED_LITTER',
           view:{scale:1,tx:0,ty:0}, dragging:false, lastMouse:[0,0],
           currentPrediction:0, currentCandidate:0, loadedFrame:null};
const el = id => document.getElementById(id);
const COLORS = {REQUIRED_LITTER:'#ff4d4d', IGNORE_SMALL:'#4da3ff', UNCERTAIN:'#ffb84d'};

function toast(msg, ms=1700){ el('toast').textContent = msg; el('toast').style.display='block';
  clearTimeout(window._t); window._t = setTimeout(()=>el('toast').style.display='none', ms); }

async function api(path, opts){
  const r = await fetch(path, opts);
  const text = await r.text();
  let data; try { data = JSON.parse(text); } catch(e){ data = {raw:text}; }
  if(!r.ok) throw new Error((data && data.error) || ('HTTP '+r.status));
  return data;
}
const get = (p,q)=>api(p + (q?('?'+new URLSearchParams(q)):''));
const post = (p,body)=>api(p,{method:'POST',headers:{'Content-Type':'application/json'},
                              body:JSON.stringify(body)});

/* ---------- view transform ---------- */
function fitView(){
  const host = el('canvasHost');
  const cw = host.clientWidth, ch = host.clientHeight;
  const scale = Math.min(cw/2560, ch/1440);
  S.view = {scale, tx:(cw-2560*scale)/2, ty:(ch-1440*scale)/2};
}
function resizeCanvas(){
  const host = el('canvasHost');
  const c = el('view');
  c.width = host.clientWidth; c.height = host.clientHeight;
  if(S.image) fitView();
  draw();
}
function toSource(px,py){ return [(px-S.view.tx)/S.view.scale, (py-S.view.ty)/S.view.scale]; }
function toCanvas(sx,sy){ return [sx*S.view.scale+S.view.tx, sy*S.view.scale+S.view.ty]; }

/* ---------- drawing ---------- */
function draw(){
  const c = el('view'), g = c.getContext('2d');
  g.setTransform(1,0,0,1,0,0);
  g.fillStyle = '#0b0c0f'; g.fillRect(0,0,c.width,c.height);
  if(!S.image) return;
  g.imageSmoothingEnabled = S.view.scale < 1;
  g.drawImage(S.image, S.view.tx, S.view.ty, 2560*S.view.scale, 1440*S.view.scale);
  const st = S.state; if(!st) return;
  /* ROI */
  const roi = st.frame.roi;
  if(roi && roi.length){
    g.save(); g.beginPath();
    roi.forEach((p,i)=>{ const [x,y]=toCanvas(p[0]*2560,p[1]*1440);
      i?g.lineTo(x,y):g.moveTo(x,y); });
    g.closePath(); g.strokeStyle='rgba(255,220,0,.85)'; g.lineWidth=2; g.stroke();
    g.restore();
  }
  /* predictions (Stage B onward only) */
  if(st.truth_complete && st.predictions){
    st.predictions.forEach((p,i)=>{
      const [x0,y0]=toCanvas(p.xyxy[0],p.xyxy[1]);
      const [x1,y1]=toCanvas(p.xyxy[2],p.xyxy[3]);
      const verdict = (st.prediction_reviews||{})[p.prediction_id];
      const cur = (st.stage==='prediction' || st.stage==='done') && i===S.currentPrediction;
      g.lineWidth = cur?3:1.5;
      g.strokeStyle = verdict==='Y'?'#3ddc84':verdict==='N'?'#ff4d4d':
                      verdict==='X'?'#4da3ff':verdict==='M'?'#ffb84d':
                      (cur?'#ffffff':'#00e0a0');
      g.strokeRect(x0,y0,x1-x0,y1-y0);
      g.font='bold 16px sans-serif'; g.fillStyle=g.strokeStyle;
      g.fillText(String(i+1), x0+3, Math.max(14,y0-4));
    });
  }
  /* truth points */
  (st.truth_points||[]).forEach(p=>{
    const [x,y]=toCanvas(p.source_xy[0],p.source_xy[1]);
    const col = COLORS[p.truth_class]||'#fff';
    g.strokeStyle=col; g.lineWidth=2;
    g.beginPath(); g.moveTo(x-9,y); g.lineTo(x+9,y); g.moveTo(x,y-9); g.lineTo(x,y+9); g.stroke();
    g.beginPath(); g.arc(x,y,6,0,6.2832); g.stroke();
  });
  drawLoupe();
}
/* Lossless magnifier: the overview canvas is JPEG, so an 8 px target would be misjudged if
   the loupe just upscaled it. The loupe fetches a lossless source-native PNG crop instead,
   and falls back to the canvas copy until that arrives. */
const LO = {half:27, scale:4, img:null, key:'', inflight:false, pending:null};
function loupeKey(sx,sy){
  return (S.loadedFrame||'')+'|'+Math.round(sx/2)+'|'+Math.round(sy/2);
}
function requestLoupe(sx,sy){
  const key = loupeKey(sx,sy);
  if(key===LO.key){ return; }
  if(LO.inflight){ LO.pending={sx:sx,sy:sy}; return; }
  LO.inflight=true; LO.key=key;
  const img=new Image();
  img.onload=()=>{ LO.img=img; LO.inflight=false; drawLoupe();
    if(LO.pending){ const p=LO.pending; LO.pending=null; requestLoupe(p.sx,p.sy); } };
  img.onerror=()=>{ LO.inflight=false; };
  img.src='/api/source_crop?frame_id='+encodeURIComponent(S.loadedFrame)+
          '&cx='+sx.toFixed(1)+'&cy='+sy.toFixed(1)+'&half='+LO.half+'&scale='+LO.scale;
}
function drawLoupe(){
  const l = el('loupe'), g = l.getContext('2d');
  const [sx,sy] = S.lastMouse || [0,0];
  const side = 2*LO.half*LO.scale;
  g.fillStyle='#000'; g.fillRect(0,0,l.width,l.height);
  if(!S.image) return;
  if(LO.img && LO.key===loupeKey(sx,sy)){
    g.imageSmoothingEnabled=false;
    g.drawImage(LO.img, 0, 0, side, side);
    return;
  }
  const half = LO.half, scale = LO.scale;
  const x0 = Math.max(0, Math.min(2560-2*half, sx-half));
  const y0 = Math.max(0, Math.min(1440-2*half, sy-half));
  g.imageSmoothingEnabled=false;
  g.drawImage(S.image, x0, y0, 2*half, 2*half, 0,0, side, side);
  const cx=(sx-x0)*scale, cy=(sy-y0)*scale;
  g.strokeStyle='#ff4d4d'; g.lineWidth=1;
  g.beginPath(); g.moveTo(cx-12,cy); g.lineTo(cx+12,cy); g.moveTo(cx,cy-12); g.lineTo(cx,cy+12);
  g.stroke();
}
function zoomAt(px,py,factor){
  const before = toSource(px,py);
  S.view.scale = Math.max(0.08, Math.min(12, S.view.scale*factor));
  S.view.tx = px - before[0]*S.view.scale;
  S.view.ty = py - before[1]*S.view.scale;
  el('zoomLabel').textContent = S.view.scale.toFixed(2)+'x';
  draw();
}

/* ---------- navigation ---------- */
async function loadFrame(i){
  S.idx = Math.max(0, Math.min(S.frames.length-1, i));
  const f = S.frames[S.idx];
  S.currentPrediction = 0; S.currentCandidate = 0;
  const st = await get('/api/state', {frame_id:f.frame_id});
  S.state = st;
  el('loadedFrame') && (el('loadedFrame').textContent = f.frame_id);
  renderHeader(); renderSidebar();
  el('progressLabel').textContent = '';
  await loadImage(f.frame_id);
  refreshProgress();
  preloadNext();
}
function loadImage(frameId){
  return new Promise((resolve,reject)=>{
    const img = new Image();
    img.onload = ()=>{ S.image = img; S.loadedFrame = frameId; fitView();
      el('zoomLabel').textContent='fit'; draw(); resolve(); };
    img.onerror = ()=>reject(new Error('image load failed'));
    img.src = '/api/frame_image?frame_id='+encodeURIComponent(frameId);
  });
}
function preloadNext(){
  const n = S.frames[S.idx+1];
  if(n){ const i = new Image(); i.src='/api/frame_image?frame_id='+encodeURIComponent(n.frame_id); }
}
function renderHeader(){
  const f = S.state.frame;
  const badge = el('splitBadge');
  const train = f.split==='rapid_train';
  badge.className = 'badge ' + (train?'train':'eval');
  badge.textContent = train ? 'RAPID-TRAIN' : 'RAPID-EVAL HOLDOUT';
  const names = {truth:'STAGE A — TRUTH (model hidden)', prediction:'STAGE B — PREDICTION',
                 localization:'STAGE C — LOCALIZATION', done:'FRAME DONE'};
  el('stageBadge').textContent = names[S.state.stage] || S.state.stage;
  el('stageBadge').className = 'stage' + (S.state.stage==='truth'?'':' on');
  el('frameLabel').textContent = `${S.idx+1}/${S.frames.length} · ${f.camera_id} · `+
     `${f.kind==='bonus_train'?'BONUS ':(f.offset_seconds+'s')} · ${f.file_id.slice(-13)} · `+
     `truth_complete=${S.state.truth_complete}`;
  el('secPred').style.display = S.state.truth_complete ? 'block':'none';
  el('secLoc').style.display = (S.state.truth_complete && S.state.localization_required &&
                                S.state.stage!=='prediction') ? 'block':'none';
}
function renderSidebar(){
  /* Stage A points */
  const list = el('pointsList'); list.innerHTML='';
  (S.state.truth_points||[]).forEach(p=>{
    const d=document.createElement('div');
    d.innerHTML = `<span class="pill ${p.truth_class==='REQUIRED_LITTER'?'r':
      p.truth_class==='IGNORE_SMALL'?'i':'u'}">${p.truth_class.replace('_LITTER','').replace('IGNORE_SMALL','IGNORE')}</span>`+
      `<span class="muted">${p.source_xy[0].toFixed(0)},${p.source_xy[1].toFixed(0)}</span>`+
      `${p.in_roi?'':' <span class="muted">(ROI外)</span>'}`;
    const b=document.createElement('button'); b.textContent='×'; b.style.marginLeft='5px';
    b.onclick=async()=>{ await post('/api/truth_delete',{truth_id:p.truth_id}); await reload(); };
    d.appendChild(b); list.appendChild(d);
  });
  if(!(S.state.truth_points||[]).length) list.innerHTML='<span class="muted">本帧暂无点位</span>';
  /* Stage B */
  if(S.state.truth_complete){
    const preds = S.state.predictions||[];
    const rev = S.state.prediction_reviews||{};
    const remaining = preds.filter(p=>!rev[p.prediction_id]).length;
    el('predInfo').textContent = `共 ${preds.length} 个框 · 未判 ${remaining} · `+
      `已判 ${preds.length-remaining}`;
    const pl = el('predList'); pl.innerHTML='';
    preds.forEach((p,i)=>{
      const v = rev[p.prediction_id]||'';
      const d=document.createElement('span');
      d.className='pill'; d.style.cursor='pointer';
      d.style.outline = i===S.currentPrediction ? '2px solid #7a6cff':'';
      d.textContent = (i+1)+':'+(v||'?');
      if(v==='Y')d.style.background='#1d5c39'; if(v==='N')d.style.background='#5c2222';
      if(v==='X')d.style.background='#1f3b63'; if(v==='M')d.style.background='#5c4520';
      d.onclick=()=>{ S.currentPrediction=i; draw(); renderSidebar(); };
      pl.appendChild(d);
    });
    renderCandidates();
  }
}
async function renderCandidates(){
  const box = el('cands'); box.innerHTML='';
  if(S.state.stage!=='localization'){ return; }
  const required = (S.state.truth_points||[]).filter(p=>p.truth_class==='REQUIRED_LITTER');
  const done = S.state.localization_reviews||{};
  const pending = required.filter(p=>!done[p.truth_id]);
  if(!pending.length){ el('locInfo').textContent='本帧所有 Required 已定位'; return; }
  /* Pick the first pending point; the operator can switch with the number keys. */
  if(!S.locTruth || !pending.some(p=>p.truth_id===S.locTruth)){
    S.locTruth = pending[0].truth_id;
  }
  el('locInfo').textContent = `点位 ${S.locTruth} (${pending.length} 个待定位) · `+
    `正在计算候选框…（首次会加载 semantic 模型，稍等）`;
  let data;
  try { data = await post('/api/localize', {truth_id:S.locTruth}); }
  catch(err){ el('locInfo').textContent = '候选框计算失败: '+err.message+' → 可按 0 跳过';
              return; }
  el('locInfo').textContent = `点位 ${S.locTruth} (${pending.length} 个待定位) · 只选 1/2/3/0`;
  box.innerHTML='';
  data.candidates.forEach((c,i)=>{
    const d=document.createElement('div');
    d.className='cand'+(i===S.currentCandidate?' act':'');
    d.innerHTML = `<img src="/api/candidate_image?truth_id=${encodeURIComponent(S.locTruth)}&idx=${i}">`+
      `<div class="candmeta"><b>${c.label} — 按 ${i+1}</b>`+
      `<small>${c.proposal_source}</small>`+
      `<small>包含该点: ${c.contains_point?'是':'否'}</small></div>`;
    d.onclick=()=>selectCandidate(i+1);
    box.appendChild(d);
  });
  if(!data.candidates.length){
    box.innerHTML='<span class="muted">无候选框（semantic/classical 均未propose）→ 按 0</span>';
  }
  if(data.semantic_note) box.innerHTML += `<div class="muted" style="width:100%">note: ${data.semantic_note}</div>`;
}
async function selectCandidate(choice){
  if(!S.locTruth) return;
  await post('/api/localize_select',{truth_id:S.locTruth, choice:choice});
  S.locTruth = null; S.currentCandidate=0;
  await reload();
}
async function refreshProgress(){
  const p = await get('/api/progress');
  const t = el('progTable');
  const pct = (a,b)=> b? Math.round(100*a/b)+'%':'—';
  t.innerHTML = `
   <tr><td>Truth Review</td><td>${p.truth_review.complete} / ${p.truth_review.total}
     <span class="muted">(固定${p.truth_review.fixed_total}+bonus${p.truth_review.bonus_total})</span></td></tr>
   <tr><td>Rapid-Train</td><td>${p.rapid_train.complete} / ${p.rapid_train.fixed_total}+${p.rapid_train.bonus_total}</td></tr>
   <tr><td>Rapid-Eval</td><td>${p.rapid_eval.complete} / ${p.rapid_eval.total}</td></tr>
   <tr><td>Required points</td><td>${p.truth_points.required}</td></tr>
   <tr><td>Ignore / Uncertain</td><td>${p.truth_points.ignore} / ${p.truth_points.uncertain}</td></tr>
   <tr><td>Prediction review</td><td>${p.prediction_review.reviewed} / ${p.prediction_review.available}</td></tr>
   <tr><td>Localization</td><td>${p.localization.selected} / ${p.localization.required_on_train_complete}
     <span class="muted">(${p.localization.localized}框/${p.localization.skipped}skip)</span></td></tr>`;
}
async function reload(){ await loadFrame(S.idx); }

/* ---------- events ---------- */
function setMode(m){ S.mode=m;
  el('mR').classList.toggle('act',m==='REQUIRED_LITTER');
  el('mI').classList.toggle('act',m==='IGNORE_SMALL');
  el('mU').classList.toggle('act',m==='UNCERTAIN');
}
el('mR').onclick=()=>setMode('REQUIRED_LITTER');
el('mI').onclick=()=>setMode('IGNORE_SMALL');
el('mU').onclick=()=>setMode('UNCERTAIN');
el('btnDone').onclick=()=>completeTruth(true);
el('btnNoTarget').onclick=()=>completeTruth(true);
el('btnPrev').onclick=()=>loadFrame(S.idx-1);
el('btnNext').onclick=()=>loadFrame(S.idx+1);
el('btnUnfinished').onclick=()=>jumpUnfinished();
el('btnZoomIn').onclick=()=>{ const c=el('view'); zoomAt(c.width/2,c.height/2,1.25); };
el('btnZoomOut').onclick=()=>{ const c=el('view'); zoomAt(c.width/2,c.height/2,1/1.25); };
el('btnFit').onclick=()=>{ fitView(); el('zoomLabel').textContent='fit'; draw(); };
el('pY').onclick=()=>judge('Y'); el('pX').onclick=()=>judge('X');
el('pN').onclick=()=>judge('N'); el('pM').onclick=()=>judge('M');
el('locNone').onclick=()=>selectCandidate(0);

async function completeTruth(){
  await post('/api/truth_complete',{frame_id:S.frames[S.idx].frame_id, complete:true});
  await reload();
}
async function judge(verdict){
  if(!S.state.truth_complete) return;
  const preds = S.state.predictions||[];
  const rev = S.state.prediction_reviews||{};
  let i = S.currentPrediction;
  while(i<preds.length && rev[preds[i].prediction_id]) i++;
  if(i>=preds.length){ toast('本帧 prediction 已全部判完'); return; }
  const p = preds[i];
  await post('/api/prediction_review',{frame_id:S.frames[S.idx].frame_id,
                                       prediction_id:p.prediction_id, verdict});
  if(verdict==='M'){ setMode('REQUIRED_LITTER'); toast('已回到 Stage A：请补点后重新确认'); }
  await reload();
}
async function jumpUnfinished(){
  for(let k=S.idx+1;k<S.frames.length;k++){
    const st = await get('/api/state',{frame_id:S.frames[k].frame_id});
    if(!st.truth_complete){ await loadFrame(k); return; }
  }
  for(let k=0;k<=S.idx;k++){
    const st = await get('/api/state',{frame_id:S.frames[k].frame_id});
    if(!st.truth_complete){ await loadFrame(k); return; }
  }
  toast('所有帧真值已完成');
}

const c = el('view');
c.addEventListener('wheel', e=>{ e.preventDefault();
  const r=c.getBoundingClientRect(); zoomAt(e.clientX-r.left, e.clientY-r.top,
    e.deltaY<0?1.15:1/1.15); }, {passive:false});
let dragMoved=false;
c.addEventListener('mousedown', e=>{ S.dragging=true; dragMoved=false;
  S.dragStart=[e.clientX,e.clientY]; });
window.addEventListener('mouseup', ()=>{ S.dragging=false; });
c.addEventListener('mousemove', e=>{
  const r=c.getBoundingClientRect();
  const [sx,sy]=toSource(e.clientX-r.left, e.clientY-r.top);
  S.lastMouse=[sx,sy];
  if(sx>=0&&sy>=0&&sx<2560&&sy<1440) requestLoupe(sx,sy);
  if(S.dragging && S.dragStart){
    if(Math.abs(e.clientX-S.dragStart[0])+Math.abs(e.clientY-S.dragStart[1])>3)
      dragMoved=true;
    S.view.tx += e.clientX-S.dragStart[0]; S.view.ty += e.clientY-S.dragStart[1];
    S.dragStart=[e.clientX,e.clientY];
  }
  draw();
});
c.addEventListener('mouseleave', ()=>{ S.dragging=false; });
c.addEventListener('click', async e=>{
  if(dragMoved){ dragMoved=false; return; }
  const r=c.getBoundingClientRect();
  const [sx,sy]=toSource(e.clientX-r.left, e.clientY-r.top);
  if(sx<0||sy<0||sx>=2560||sy>=1440) return;
  if(e.shiftKey){
    const pts = (S.state.truth_points||[]);
    let best=null;
    pts.forEach(p=>{ const dx=p.source_xy[0]-sx, dy=p.source_xy[1]-sy;
      const d=Math.hypot(dx,dy); if(d<16 && (!best||d<best.d)) best={d,p}; });
    if(best){ await post('/api/truth_delete',{truth_id:best.p.truth_id}); await reload(); }
    else toast('附近 16px 内没有可删除的点位');
    return;
  }
  if(S.state.stage!=='truth'){ toast('当前不是 Stage A：请先按 M 回到真值阶段'); return; }
  await post('/api/truth',{frame_id:S.frames[S.idx].frame_id, truth_class:S.mode,
                           x:sx, y:sy});
  await reload();
});
window.addEventListener('keydown', async e=>{
  const k = e.key;
  if(k==='r'||k==='R') return setMode('REQUIRED_LITTER');
  if(k==='i'||k==='I') return setMode('IGNORE_SMALL');
  if(k==='u'||k==='U') return setMode('UNCERTAIN');
  if(k==='Enter'){ e.preventDefault(); return completeTruth(); }
  if(k==='ArrowLeft'){ return loadFrame(S.idx-1); }
  if(k==='ArrowRight'){ return loadFrame(S.idx+1); }
  if(S.state && S.state.truth_complete && S.state.stage==='prediction'){
    if(k==='y'||k==='Y') return judge('Y');
    if(k==='x'||k==='X') return judge('X');
    if(k==='f'||k==='F') return judge('N');
    if(k==='n'||k==='N') return judge('N');
    if(k==='m'||k==='M') return judge('M');
  }
  if(S.state && S.state.truth_complete && S.state.stage==='truth'){
    if(k==='n'||k==='N') return completeTruth();
  }
  if(S.state && S.state.stage==='localization'){
    if(k==='1'){ return selectCandidate(1); }
    if(k==='2'){ return selectCandidate(2); }
    if(k==='3'){ return selectCandidate(3); }
    if(k==='0'){ return selectCandidate(0); }
  }
});
window.addEventListener('resize', resizeCanvas);

(async function boot(){
  const data = await get('/api/frames');
  S.frames = data.frames;
  setMode('REQUIRED_LITTER');
  resizeCanvas();
  const wanted = new URLSearchParams(location.search).get('frame');
  if(wanted){
    const j = S.frames.findIndex(f=>f.frame_id===wanted);
    if(j>=0){ await loadFrame(j); toast('deep link: '+wanted, 2600); return; }
    toast('未知 frame: '+wanted, 3000);
  }
  /* resume at the first frame whose truth is not complete */
  let start = 0;
  for(let i=0;i<S.frames.length;i++){
    const st = await get('/api/state',{frame_id:S.frames[i].frame_id});
    if(!st.truth_complete){ start=i; break; }
    start = i;
  }
  await loadFrame(start);
  toast('已恢复：从第 '+(start+1)+' 帧继续 ('+S.frames[start].split+')', 2600);
})();
</script></body></html>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--no-semantic", action="store_true",
                   help="disable the Turhancan candidate source (candidate B)")
    sub = p.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--bind", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8810)
    sub.add_parser("status")
    sub.add_parser("selftest")
    return p


def build(artifact: Path, *, enable_semantic=True):
    store = RapidReviewStore(artifact)
    store.load_predictions()
    images = FrameImages(artifact, store)
    localizer = Localizer(artifact, store, enable_semantic=enable_semantic)
    return store, images, localizer


def cmd_serve(args) -> int:
    from http.server import ThreadingHTTPServer
    store, images, localizer = build(args.artifact, enable_semantic=not args.no_semantic)
    handler = make_handler(store, images, localizer, repo_root=Path(__file__).resolve().parents[1])
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Rapid v1 review UI on http://{args.bind}:{args.port}/")
    print(f"artifact={store.artifact}")
    print(f"frames={len(store.frames)} predictions_loaded={bool(store._predictions_raw)}")
    print(f"semantic={localizer._semantic_path or localizer.semantic_note}")
    print("official 8801 is never contacted; no writes leave the artifact root")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_status(args) -> int:
    store = RapidReviewStore(args.artifact)
    store.load_predictions()
    print(json.dumps(store.progress(), ensure_ascii=False, indent=2))
    return 0


def cmd_selftest(args) -> int:
    """Blindness + guard self-test that never needs a network or a model."""
    store = RapidReviewStore(args.artifact)
    store.load_predictions()
    failures: list[str] = []
    stage_a_blind = True
    for frame in store.frames:
        entry = store.state["frames"].get(frame["frame_id"]) or {}
        if entry.get("truth_complete"):
            continue
        payload = store.frame_payload(frame["frame_id"])
        if payload.get("predictions") is not None:
            stage_a_blind = False
            failures.append(f'blindness leak: predictions for {frame["frame_id"]}')
        text = json.dumps(payload, ensure_ascii=False)
        if store._predictions_raw.get(frame["frame_id"]) and '"prediction_id"' in text:
            stage_a_blind = False
            failures.append(f'blindness leak: prediction ids in {frame["frame_id"]}')
        break
    if not stage_a_blind:
        pass
    eval_frames = [f for f in store.frames if f["split"] == EVAL_SPLIT]
    if not eval_frames:
        failures.append("no rapid_eval frame in the manifest")
    for frame in eval_frames[:1]:
        try:
            store.train_export_probe(frame["frame_id"])
            failures.append(f'training export guard did not fire for {frame["frame_id"]}')
        except RapidError:
            pass
    result = {
        "stage_a_blind": stage_a_blind,
        "eval_export_refused": not any("guard did not fire" in f for f in failures),
        "frames": len(store.frames),
        "rapid_eval_frames": len(eval_frames),
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "selftest":
        return cmd_selftest(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
