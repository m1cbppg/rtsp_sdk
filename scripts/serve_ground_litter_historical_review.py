#!/usr/bin/env python3
"""Ground Litter Historical Active Mining + Rapid Dataset v2 — human review service.

The review unit is a **review group** (one unique litter instance, or one independent
hard-negative cluster), never a single frame.  This service replaces the older
frame-based review UI.

Isolated by construction:

* it only ever reads/writes inside its own ``--artifact`` root, and refuses any
  traversal (``..``) or absolute escape with HTTP 400/403;
* it never touches ``/home/sf01/step2c1-blind-truth``, anything matching ``sealed``,
  or the official 8801 review service;
* it binds ``127.0.0.1:8812`` by default and never emits signed video URLs or host
  secrets;
* it is Python-stdlib only.  ``cv2``/``numpy`` are imported lazily inside the media
  renderer, and every cv2-dependent feature degrades to a clear "unavailable"
  payload when they are missing (the repo venv currently has no importable cv2).

Usage::

    python scripts/serve_ground_litter_historical_review.py --artifact <root> serve --port 8812

Subcommands: ``serve``, ``status``, ``selftest``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

# --------------------------------------------------------------------------- #
# Constants and contract
# --------------------------------------------------------------------------- #

DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 8812

#: Verdict codes.  ``R`` must be followed by a bbox decision; I/U/N complete at once.
VERDICTS = ("R", "I", "U", "N")
VERDICT_LABELS = {
    "R": "必需垃圾",
    "I": "微小忽略",
    "U": "不确定",
    "N": "非垃圾/背景",
}
BBOX_CHOICES = (0, 1, 2, 3)
LINK_DECISIONS = ("SAME", "NEW", "UNCERTAIN")

#: Data-contract file names.
COARSE_FRAMES_NAME = "coarse_frames.jsonl"
REVIEW_UNITS_NAME = "review_units.jsonl"
REVIEW_DIR_NAME = "review"
DECISIONS_NAME = "decisions.jsonl"
BBOX_DECISIONS_NAME = "bbox_decisions.jsonl"
LINK_DECISIONS_NAME = "link_decisions.jsonl"

#: Paths this service must never write into.
FORBIDDEN_WRITE_ROOTS = (
    "/home/sf01/step2c1-blind-truth",
)
#: Any artifact path containing one of these tokens is out of bounds.
FORBIDDEN_PATH_TOKENS = ("sealed",)

CONTEXT_MAX_WIDTH = 1600
CROP_MIN_SIDE = 160.0
CROP_TARGET_LONG_SIDE = 360.0
JPEG_QUALITY = 90
MEDIA_CACHE_ENTRIES = 256

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")


class ReviewError(RuntimeError):
    """A review-service contract violation."""


class ClientError(ReviewError):
    """An error that maps directly to an HTTP status for the operator."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = int(status)


class MediaUnavailable(ReviewError):
    """Raised when a cv2-dependent rendering feature cannot run."""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Isolation helpers
# --------------------------------------------------------------------------- #


def assert_artifact_root(path: str | Path) -> Path:
    """Resolve the artifact root and refuse forbidden locations.

    Both the literal and the symlink-resolved spelling are checked: the operator's
    written path is what the policy is about, and ``resolve()`` alone would let a
    symlinked ``/home`` hide a forbidden prefix.
    """
    given = Path(path).expanduser()
    resolved = given.resolve()
    spellings = {os.path.normpath(str(given)), os.path.normpath(str(resolved))}
    for spelling in spellings:
        lowered = spelling.lower()
        for token in FORBIDDEN_PATH_TOKENS:
            if token in lowered:
                raise ReviewError(f"refusing sealed/forbidden path: {spelling}")
        for forbidden in FORBIDDEN_WRITE_ROOTS:
            if spelling == forbidden or spelling.startswith(forbidden + os.sep):
                raise ReviewError(f"refusing forbidden root: {spelling}")
    return resolved


def resolve_under(root: Path, relative: str | Path) -> Path:
    """Resolve ``relative`` under ``root``; refuse absolute paths and ``..`` escapes."""
    rel = Path(str(relative))
    if rel.is_absolute():
        raise ClientError(f"absolute path refused: {relative}", 403)
    if any(part == ".." for part in rel.parts):
        raise ClientError(f"path traversal refused: {relative}", 403)
    candidate = (root / rel).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ClientError(f"path escapes artifact root: {relative}", 403)
    return candidate


# --------------------------------------------------------------------------- #
# JSON / JSONL IO
# --------------------------------------------------------------------------- #


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write JSONL atomically (tmp + fsync + rename) and deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class ReviewStore:
    """Reader/writer for the review-group artifact.  Every write is atomic."""

    def __init__(self, artifact: str | Path):
        self.artifact = assert_artifact_root(artifact)
        if not self.artifact.exists():
            raise ReviewError(f"artifact root does not exist: {self.artifact}")
        self.review_dir = self.artifact / REVIEW_DIR_NAME
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

        self.frames: dict[str, dict] = {}
        self.frame_order: list[str] = []
        self._load_frames()

        self.units: list[dict] = []
        self.unit_by_id: dict[str, dict] = {}
        self._load_units()

        self.decisions: dict[str, dict] = {}
        self.bbox: dict[str, dict] = {}
        self.links: dict[str, dict] = {}
        self._load_decisions()

    # -- loading ----------------------------------------------------------- #
    def _load_frames(self) -> None:
        rows = read_jsonl(self.artifact / COARSE_FRAMES_NAME)
        for index, row in enumerate(rows):
            frame_id = str(row.get("frame_id") or "")
            if not frame_id:
                continue
            self.frames[frame_id] = row
            self.frame_order.append(frame_id)
        if not self.frames:
            raise ReviewError(f"{COARSE_FRAMES_NAME} is empty or missing")

    def _load_units(self) -> None:
        rows = read_jsonl(self.artifact / REVIEW_UNITS_NAME)
        if not rows:
            raise ReviewError(f"{REVIEW_UNITS_NAME} is empty or missing")
        seen: set[str] = set()
        for row in rows:
            unit_id = str(row.get("unit_id") or "")
            if not unit_id or unit_id in seen:
                continue
            seen.add(unit_id)
            self.units.append(row)
        self.units.sort(key=lambda u: (int(u.get("queue_index") or 0), str(u.get("unit_id"))))
        self.unit_by_id = {str(u["unit_id"]): u for u in self.units}

    def _load_decisions(self) -> None:
        # Last row wins, so a legacy duplicate never double-counts and the next write
        # rewrites exactly one row per unit.
        for row in read_jsonl(self._path(DECISIONS_NAME)):
            uid = str(row.get("unit_id") or "")
            if uid:
                self.decisions[uid] = row
        for row in read_jsonl(self._path(BBOX_DECISIONS_NAME)):
            uid = str(row.get("unit_id") or "")
            if uid:
                self.bbox[uid] = row
        for row in read_jsonl(self._path(LINK_DECISIONS_NAME)):
            uid = str(row.get("unit_id") or "")
            if uid:
                self.links[uid] = row

    def _path(self, name: str) -> Path:
        return self.review_dir / name

    # -- persistence ------------------------------------------------------- #
    def _ordered(self, mapping: Mapping[str, dict]) -> list[dict]:
        order = {str(u["unit_id"]): i for i, u in enumerate(self.units)}
        return [mapping[key] for key in sorted(
            mapping, key=lambda k: (order.get(k, 10 ** 9), str(k)))]

    def _persist_decisions(self) -> None:
        write_jsonl_atomic(self._path(DECISIONS_NAME), self._ordered(self.decisions))

    def _persist_bbox(self) -> None:
        write_jsonl_atomic(self._path(BBOX_DECISIONS_NAME), self._ordered(self.bbox))

    def _persist_links(self) -> None:
        write_jsonl_atomic(self._path(LINK_DECISIONS_NAME), self._ordered(self.links))

    # -- lookups ----------------------------------------------------------- #
    def unit(self, unit_id: str) -> dict:
        unit = self.unit_by_id.get(str(unit_id))
        if unit is None:
            raise ClientError(f"unknown unit_id: {unit_id}", 404)
        return unit

    def frame_for_unit(self, unit: dict) -> dict | None:
        observations = unit.get("observations") or []
        if not observations:
            return None
        rep_id = unit.get("representative_observation_id")
        rep = next((o for o in observations if o.get("observation_id") == rep_id), None)
        if rep is None:
            rep = observations[0]
        return self.frames.get(str(rep.get("frame_id") or ""))

    def representative_observation(self, unit: dict) -> dict | None:
        observations = unit.get("observations") or []
        if not observations:
            return None
        rep_id = unit.get("representative_observation_id")
        rep = next((o for o in observations if o.get("observation_id") == rep_id), None)
        return rep or observations[0]

    def image_path(self, frame_row: Mapping[str, Any]) -> Path:
        rel = frame_row.get("image")
        if not rel:
            raise ClientError("frame row has no image path", 404)
        return resolve_under(self.artifact, rel)

    # -- decisions --------------------------------------------------------- #
    def decision_of(self, unit_id: str) -> str | None:
        row = self.decisions.get(str(unit_id))
        return str(row.get("verdict")) if row else None

    def bbox_choice_of(self, unit_id: str) -> int | None:
        row = self.bbox.get(str(unit_id))
        if not row:
            return None
        try:
            return int(row.get("choice"))
        except (TypeError, ValueError):
            return None

    def link_decision_of(self, unit_id: str) -> str | None:
        row = self.links.get(str(unit_id))
        return str(row.get("decision")) if row else None

    def model_revealed(self, unit: Mapping[str, Any]) -> bool:
        """Blind units hide their model output until a verdict is recorded."""
        if not unit.get("blind"):
            return True
        return str(unit.get("unit_id")) in self.decisions

    def is_complete(self, unit_id: str) -> bool:
        verdict = self.decision_of(unit_id)
        if verdict is None:
            return False
        if verdict != "R":
            return True
        return str(unit_id) in self.bbox

    def stage_of(self, unit_id: str) -> str:
        verdict = self.decision_of(unit_id)
        if verdict is None:
            return "classify"
        if verdict == "R" and str(unit_id) not in self.bbox:
            return "bbox"
        return "done"

    def decide(self, unit_id: str, verdict: Any) -> dict:
        with self.lock:
            unit_id = str(unit_id or "")
            self.unit(unit_id)
            code = str(verdict or "").strip().upper()
            if code not in VERDICTS:
                raise ClientError(f"invalid verdict {verdict!r}; expected one of {VERDICTS}", 400)
            existing = self.decisions.get(unit_id)
            if existing and str(existing.get("verdict")) == code:
                # Idempotent re-decide: keep the original timestamp, do not churn the file.
                return self.unit_payload(unit_id)
            created = existing.get("created_at") if existing else None
            self.decisions[unit_id] = {
                "unit_id": unit_id,
                "verdict": code,
                "created_at": created or now_iso(),
            }
            self._persist_decisions()
            return self.unit_payload(unit_id)

    def set_bbox(self, unit_id: str, choice: Any) -> dict:
        with self.lock:
            unit_id = str(unit_id or "")
            unit = self.unit(unit_id)
            if self.decision_of(unit_id) != "R":
                raise ClientError("bbox decision requires a recorded REQUIRED (R) verdict", 409)
            try:
                value = int(choice)
            except (TypeError, ValueError):
                raise ClientError(f"invalid bbox choice {choice!r}", 400)
            if value not in BBOX_CHOICES:
                raise ClientError(f"bbox choice must be one of {BBOX_CHOICES}", 400)
            candidate_id = None
            if value >= 1:
                candidates = unit.get("candidates") or []
                if value > len(candidates):
                    raise ClientError(
                        f"no candidate for choice {value}; unit has {len(candidates)}", 409)
                candidate_id = candidates[value - 1].get("candidate_id")
            existing = self.bbox.get(unit_id)
            if existing and int(existing.get("choice", -1)) == value:
                return self.unit_payload(unit_id)
            created = existing.get("created_at") if existing else None
            self.bbox[unit_id] = {
                "unit_id": unit_id,
                "choice": value,
                "candidate_id": candidate_id,
                "created_at": created or now_iso(),
            }
            self._persist_bbox()
            return self.unit_payload(unit_id)

    def set_link(self, unit_id: str, decision: Any) -> dict:
        with self.lock:
            unit_id = str(unit_id or "")
            unit = self.unit(unit_id)
            if not unit.get("suspected_same_object"):
                raise ClientError(
                    "link decision is only valid for suspected_same_object units", 409)
            code = str(decision or "").strip().upper()
            if code not in LINK_DECISIONS:
                raise ClientError(
                    f"invalid link decision {decision!r}; expected {LINK_DECISIONS}", 400)
            existing = self.links.get(unit_id)
            if existing and str(existing.get("decision")) == code:
                return self.unit_payload(unit_id)
            created = existing.get("created_at") if existing else None
            self.links[unit_id] = {
                "unit_id": unit_id,
                "decision": code,
                "created_at": created or now_iso(),
            }
            self._persist_links()
            return self.unit_payload(unit_id)

    # -- aggregates -------------------------------------------------------- #
    def counts_by_verdict(self) -> dict:
        counts = {code: 0 for code in VERDICTS}
        undecided = 0
        for unit in self.units:
            verdict = self.decision_of(str(unit["unit_id"]))
            if verdict in counts:
                counts[verdict] += 1
            else:
                undecided += 1
        counts["undecided"] = undecided
        return counts

    def progress(self) -> dict:
        total = len(self.units)
        decided = bbox_done = complete = revealed = 0
        for unit in self.units:
            unit_id = str(unit["unit_id"])
            if self.decision_of(unit_id) is not None:
                decided += 1
            if unit_id in self.bbox:
                bbox_done += 1
            if self.is_complete(unit_id):
                complete += 1
            if unit.get("blind") and self.model_revealed(unit):
                revealed += 1
        return {
            "total": total,
            "decided": decided,
            "bbox_done": bbox_done,
            "complete": complete,
            "remaining": total - complete,
            "model_revealed": revealed,
        }

    def default_batch(self) -> int:
        batches = sorted({int(u.get("batch") or 0) for u in self.units})
        if not batches:
            return 1
        incomplete = [int(u.get("batch") or 0) for u in self.units
                      if not self.is_complete(str(u["unit_id"]))]
        if incomplete:
            return min(incomplete)
        return batches[-1]

    def queue(self, batch: int | None = None, offset: int = 0,
              limit: int = 50) -> dict:
        batch = self.default_batch() if batch is None else int(batch)
        rows = [u for u in self.units if int(u.get("batch") or 0) == batch]
        total = len(rows)
        offset = max(0, int(offset))
        limit = max(1, min(1000, int(limit)))
        window = rows[offset:offset + limit]
        units = []
        for unit in window:
            unit_id = str(unit["unit_id"])
            units.append({
                "unit_id": unit_id,
                "kind": unit.get("kind"),
                "camera_id": unit.get("camera_id"),
                "timestamp": unit.get("timestamp"),
                "tier": unit.get("tier"),
                "queue_index": unit.get("queue_index"),
                "complete": self.is_complete(unit_id),
                "verdict": self.decision_of(unit_id),
                "thumb_url": f"/media/{unit_id}/context.jpg",
            })
        return {"total": total, "batch": batch, "units": units}

    def state(self) -> dict:
        return {
            "artifact": str(self.artifact),
            "progress": self.progress(),
            "batch": self.default_batch(),
            "counts_by_verdict": self.counts_by_verdict(),
        }

    def unit_payload(self, unit_id: str) -> dict:
        unit = self.unit(unit_id)
        uid = str(unit["unit_id"])
        revealed = self.model_revealed(unit)
        payload = dict(unit)
        if unit.get("blind") and not revealed:
            # Hard blindness: no candidate rows and no candidate-derived URLs.
            payload["candidates"] = []
        observations = payload.get("observations") or []
        candidates = payload.get("candidates") or []
        observation_urls = {
            str(o.get("observation_id")): f"/media/{uid}/obs_{o.get('observation_id')}.jpg"
            for o in observations if o.get("observation_id")
        }
        candidate_urls = {}
        if revealed:
            candidate_urls = {
                str(c.get("candidate_id")): f"/media/{uid}/cand_{c.get('candidate_id')}.jpg"
                for c in candidates if c.get("candidate_id")
            }
        payload["decision"] = self.decision_of(uid)
        payload["bbox_choice"] = self.bbox_choice_of(uid)
        payload["link_decision"] = self.link_decision_of(uid)
        payload["model_revealed"] = revealed
        payload["complete"] = self.is_complete(uid)
        payload["stage"] = self.stage_of(uid)
        payload["images"] = {
            "context_url": f"/media/{uid}/context.jpg",
            "crop_url": f"/media/{uid}/crop.jpg",
            "observation_urls": observation_urls,
            "candidate_urls": candidate_urls,
        }
        return payload


# --------------------------------------------------------------------------- #
# Media rendering (cv2, lazily imported and fully optional)
# --------------------------------------------------------------------------- #


def import_cv2():
    try:  # pragma: no cover - exercised by presence/absence of cv2 in the env
        import cv2  # noqa: F401
        return cv2
    except Exception:  # noqa: BLE001
        return None


class MediaRenderer:
    """Renders context/crop/observation/candidate JPEGs, with a bounded cache."""

    def __init__(self, store: ReviewStore):
        self.store = store
        self.artifact = store.artifact
        self.cv2 = import_cv2()
        self._cache: "OrderedDict[str, bytes]" = OrderedDict()
        self._lock = threading.RLock()

    def available(self) -> bool:
        return self.cv2 is not None

    # -- public ------------------------------------------------------------ #
    def render(self, unit_id: str, name: str) -> bytes:
        unit = self.store.unit(unit_id)
        mode, target = self._classify(unit, name)
        key = f"{unit.get('unit_id')}__{name}"
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        data = self._render_uncached(unit, mode, target)
        with self._lock:
            self._cache[key] = data
            while len(self._cache) > MEDIA_CACHE_ENTRIES:
                self._cache.popitem(last=False)
        self._write_preview(unit, name, data)
        return data

    def _classify(self, unit: Mapping[str, Any], name: str) -> tuple[str, dict | None]:
        if name == "context":
            return "context", None
        if name == "crop":
            return "crop", None
        if name.startswith("obs_"):
            observation_id = name[len("obs_"):]
            match = next((o for o in (unit.get("observations") or [])
                          if str(o.get("observation_id")) == observation_id), None)
            if match is None:
                raise ClientError(f"unknown observation: {observation_id}", 404)
            return "obs", match
        if name.startswith("cand_"):
            candidate_id = name[len("cand_"):]
            if unit.get("blind") and not self.store.model_revealed(unit):
                # A blind unit's candidate images stay hidden until a verdict is recorded.
                raise ClientError("model output is hidden until a verdict is recorded", 404)
            match = next((c for c in (unit.get("candidates") or [])
                          if str(c.get("candidate_id")) == candidate_id), None)
            if match is None:
                raise ClientError(f"unknown candidate: {candidate_id}", 404)
            return "cand", match
        raise ClientError(f"unknown media name: {name}", 404)

    # -- preview disk cache ------------------------------------------------ #
    def _write_preview(self, unit: Mapping[str, Any], name: str, data: bytes) -> None:
        uid = str(unit.get("unit_id") or "")
        if not _SAFE_NAME.match(uid) or not _SAFE_NAME.match(name):
            return
        try:
            directory = self.artifact / "previews" / uid
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{name}.jpg"
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
        except OSError:
            # A preview cache is a convenience; never fail a review request over it.
            pass

    # -- geometry ---------------------------------------------------------- #
    @staticmethod
    def _to_pixels(box: Any, width: int, height: int) -> list[float] | None:
        if not box or len(box) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            return None
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            # Tolerate normalized boxes even though the contract is source pixels.
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        return [x1, y1, x2, y2]

    def _rep_box(self, unit: Mapping[str, Any], width: int, height: int) -> list[float] | None:
        rep = self.store.representative_observation(unit)
        if rep is None:
            return None
        return self._to_pixels(rep.get("bbox_xyxy"), width, height)

    @staticmethod
    def _draw_rect(cv2, image, box: list[float], color, thickness: int = 2) -> None:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # -- rendering --------------------------------------------------------- #
    def _render_uncached(self, unit: Mapping[str, Any], mode: str,
                         target: dict | None) -> bytes:
        cv2 = self.cv2
        if cv2 is None:
            raise MediaUnavailable(
                "cv2 不可用：无法渲染图像。请安装 opencv-python 后重启服务。")
        try:
            import numpy as np
        except Exception:  # noqa: BLE001
            np = None

        frame_row = self.store.frame_for_unit(unit)
        if frame_row is None:
            raise ClientError("unit has no resolvable coarse frame", 404)
        image_path = self.store.image_path(frame_row)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ClientError(f"cannot read source frame: {image_path.name}", 404)
        height, width = image.shape[:2]

        if mode == "context":
            return self._render_context(cv2, np, image, frame_row, unit, width, height)

        if mode == "crop":
            box = self._rep_box(unit, width, height)
            if box is None:
                raise ClientError("unit has no representative bbox", 404)
            return self._render_crop(cv2, image, box, width, height)

        if mode == "obs":
            box = self._to_pixels((target or {}).get("bbox_xyxy"), width, height)
            if box is None:
                raise ClientError("observation has no bbox", 404)
            return self._render_crop(cv2, image, box, width, height)

        if mode == "cand":
            box = self._to_pixels((target or {}).get("bbox_xyxy"), width, height)
            if box is None:
                raise ClientError("candidate has no bbox", 404)
            point = self._rep_box(unit, width, height)
            point_xy = ((point[0] + point[2]) / 2.0, (point[1] + point[3]) / 2.0) if point else None
            return self._render_crop(cv2, image, box, width, height, point=point_xy)

        raise ClientError(f"unsupported render mode: {mode}", 404)

    def _render_context(self, cv2, np, image, frame_row: Mapping[str, Any],
                        unit: Mapping[str, Any], width: int, height: int) -> bytes:
        canvas = image
        if width > CONTEXT_MAX_WIDTH:
            scale = CONTEXT_MAX_WIDTH / float(width)
            canvas = cv2.resize(image, (CONTEXT_MAX_WIDTH, int(round(height * scale))),
                                interpolation=cv2.INTER_AREA)
        ch, cw = canvas.shape[:2]
        roi = frame_row.get("roi") or []
        if np is not None and roi:
            points = self._roi_points(roi, cw, ch, np)
            if points is not None and len(points) >= 3:
                cv2.polylines(canvas, [points], True, (0, 215, 255), 2)
        box = self._rep_box(unit, cw, ch)
        if box is not None:
            self._draw_rect(cv2, canvas, box, (0, 255, 0), 2)
        return self._encode(cv2, canvas)

    @staticmethod
    def _roi_points(roi: Any, width: int, height: int, np):
        try:
            if roi and isinstance(roi[0], (list, tuple)) and len(roi[0]) == 2:
                pairs = [(float(x), float(y)) for x, y in roi]
            elif len(roi) == 4:
                x1, y1, x2, y2 = (float(v) for v in roi)
                pairs = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            else:
                return None
        except (TypeError, ValueError):
            return None
        points = []
        for x, y in pairs:
            if abs(x) <= 1.0 and abs(y) <= 1.0:
                points.append([int(round(x * width)), int(round(y * height))])
            else:
                points.append([int(round(x)), int(round(y))])
        return np.array(points, dtype=np.int32).reshape(-1, 1, 2)

    def _render_crop(self, cv2, image, box: list[float], width: int, height: int,
                     *, point: tuple[float, float] | None = None) -> bytes:
        x1, y1, x2, y2 = box
        side = max(CROP_MIN_SIDE, max(x2 - x1, y2 - y1) * 1.5)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = side / 2.0
        x0 = int(max(0, min(width - 1, round(cx - half))))
        y0 = int(max(0, min(height - 1, round(cy - half))))
        x1c = int(max(x0 + 1, min(width, round(cx + half))))
        y1c = int(max(y0 + 1, min(height, round(cy + half))))
        crop = image[y0:y1c, x0:x1c]
        if getattr(crop, "size", 0) == 0:
            raise ClientError("empty crop", 404)
        long_side = max(crop.shape[0], crop.shape[1])
        if long_side > 0 and long_side < CROP_TARGET_LONG_SIDE:
            scale = CROP_TARGET_LONG_SIDE / float(long_side)
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST)
        sx = crop.shape[1] / float(x1c - x0)
        sy = crop.shape[0] / float(y1c - y0)
        rect = [(x1 - x0) * sx, (y1 - y0) * sy, (x2 - x0) * sx, (y2 - y0) * sy]
        self._draw_rect(cv2, crop, rect, (0, 255, 0), 2)
        if point is not None:
            px = int(round((point[0] - x0) * sx))
            py = int(round((point[1] - y0) * sy))
            cv2.circle(crop, (px, py), 4, (0, 0, 255), -1)
        return self._encode(cv2, crop)

    @staticmethod
    def _encode(cv2, image) -> bytes:
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise MediaUnavailable("cv2 无法编码 JPEG")
        return buffer.tobytes()


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def make_handler(store: ReviewStore, renderer: MediaRenderer, *, page: str | None = None):
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import parse_qs, unquote, urlparse

    html = PAGE if page is None else page

    class Handler(BaseHTTPRequestHandler):
        server_version = "GroundLitterHistoricalReview/2.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
            sys.stderr.write("historical-review: " + (fmt % args) + "\n")

        # -- low level ----------------------------------------------------- #
        def _send(self, status: int, body: bytes, content_type: str = "application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _json(self, payload, status: int = 200):
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        def _error(self, message, status: int = 400, *, ok: bool = False):
            self._json({"ok": ok, "error": str(message)}, status)

        def _params(self) -> dict:
            parsed = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            params["__path"] = parsed.path
            return params

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            if not raw:
                return {}
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ClientError("invalid JSON body", 400)
            if not isinstance(data, dict):
                raise ClientError("JSON body must be an object", 400)
            return data

        # -- GET ----------------------------------------------------------- #
        def do_GET(self):  # noqa: N802 - stdlib signature
            params = self._params()
            path = params["__path"]
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                if path == "/favicon.ico":
                    return self._send(204, b"", "image/x-icon")
                if path == "/api/state":
                    return self._json(store.state())
                if path == "/api/queue":
                    batch = params.get("batch")
                    return self._json(store.queue(
                        int(batch) if batch not in (None, "") else None,
                        int(params.get("offset", 0) or 0),
                        int(params.get("limit", 50) or 50)))
                if path == "/api/unit":
                    return self._json(store.unit_payload(params.get("unit_id", "")))
                if path.startswith("/media/"):
                    return self._media(path)
                return self._error(f"unknown path: {path}", 404)
            except ClientError as error:
                return self._error(error, error.status)
            except MediaUnavailable as error:
                return self._json({
                    "ok": False,
                    "unavailable": True,
                    "cv2": False,
                    "error": str(error),
                }, 503)
            except ReviewError as error:
                return self._error(error, 500)
            except Exception as error:  # noqa: BLE001
                return self._error(f"{type(error).__name__}: {error}", 500)

        def _media(self, raw_path: str):
            # Refuse dot-segments before any decoding: the raw request path is the only
            # place a traversal attempt is still visible.
            if ".." in raw_path.split("/") or "\\" in raw_path:
                return self._error("path traversal refused", 403)
            rest = raw_path[len("/media/"):]
            parts = rest.split("/")
            if len(parts) != 2 or not parts[1]:
                return self._error("media path must be /media/<unit_id>/<name>.jpg", 404)
            unit_id = unquote(parts[0])
            filename = unquote(parts[1])
            if "/" in unit_id or "\\" in unit_id or unit_id in ("", ".", ".."):
                return self._error("invalid unit id", 403)
            if not filename.endswith(".jpg"):
                return self._error("media name must end with .jpg", 404)
            name = filename[:-len(".jpg")]
            try:
                data = renderer.render(unit_id, name)
            except ClientError as error:
                return self._error(error, error.status)
            except MediaUnavailable as error:
                return self._json({
                    "ok": False,
                    "unavailable": True,
                    "cv2": False,
                    "error": str(error),
                }, 503)
            return self._send(200, data, "image/jpeg")

        # -- POST ---------------------------------------------------------- #
        def do_POST(self):  # noqa: N802 - stdlib signature
            params = self._params()
            path = params["__path"]
            try:
                body = self._body()
                if path == "/api/decide":
                    payload = store.decide(body.get("unit_id", ""), body.get("verdict"))
                elif path == "/api/bbox":
                    payload = store.set_bbox(body.get("unit_id", ""), body.get("choice"))
                elif path == "/api/link":
                    payload = store.set_link(body.get("unit_id", ""), body.get("decision"))
                else:
                    return self._error(f"unknown path: {path}", 404)
                return self._json({
                    "ok": True,
                    "unit_id": payload["unit_id"],
                    "decision": payload["decision"],
                    "bbox_choice": payload["bbox_choice"],
                    "link_decision": payload["link_decision"],
                    "model_revealed": payload["model_revealed"],
                    "stage": payload["stage"],
                    "complete": payload["complete"],
                    "progress": store.progress(),
                })
            except ClientError as error:
                return self._json({"ok": False, "error": str(error)}, error.status)
            except ReviewError as error:
                return self._json({"ok": False, "error": str(error)}, 500)
            except Exception as error:  # noqa: BLE001
                return self._json({"ok": False, "error": f"{type(error).__name__}: {error}"}, 500)

    return Handler


# --------------------------------------------------------------------------- #
# Single page UI (vanilla JS + inline CSS, fully offline)
# --------------------------------------------------------------------------- #

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>地面零散垃圾历史主动挖掘 v2 · 人工复核</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { font: 15px/1.55 system-ui, "PingFang SC", "Microsoft YaHei", sans-serif;
         margin: 0; background: #eef1f5; color: #1d2733; }
  header { position: sticky; top: 0; z-index: 10; background: #14213d; color: #fff;
           padding: 10px 18px; display: flex; flex-wrap: wrap; gap: 14px; align-items: baseline; }
  header b { font-size: 17px; }
  header .meta { color: #c9d6ea; font-size: 13px; }
  #progress { margin-left: auto; font-variant-numeric: tabular-nums; }
  #banner { display: none; background: #b42318; color: #fff; padding: 10px 18px;
            font-weight: 600; white-space: pre-wrap; }
  main { display: grid; grid-template-columns: minmax(320px, 1.15fr) minmax(320px, 1fr);
         gap: 16px; padding: 16px; }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
  .card { background: #fff; border: 1px solid #d7dee8; border-radius: 10px; padding: 12px; }
  .card h3 { margin: 0 0 8px; font-size: 14px; color: #43536b; font-weight: 600; }
  .card img { width: 100%; background: #0d1117; border-radius: 6px; display: block; min-height: 80px; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .tabs { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
  .tabs button { border: 1px solid #c3cddb; background: #f7f9fc; border-radius: 999px;
                 padding: 3px 11px; cursor: pointer; font-size: 13px; }
  .tabs button.on { background: #14213d; color: #fff; border-color: #14213d; }
  .verdicts { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  .verdicts button { border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer;
                     font-size: 15px; font-weight: 700; color: #fff; }
  .verdicts button:disabled { opacity: .35; cursor: not-allowed; }
  .bR { background: #b42318; } .bI { background: #b54708; }
  .bU { background: #175cd3; } .bN { background: #475467; }
  .hint { margin-top: 8px; color: #566781; font-size: 13px; }
  .blind { background: #fff4e5; border: 1px solid #f5c07a; color: #93370d;
           padding: 10px; border-radius: 8px; font-weight: 600; margin-top: 8px; }
  .cands { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
           gap: 10px; margin-top: 10px; }
  .cand { border: 1px solid #d7dee8; border-radius: 8px; padding: 8px; background: #fbfcfe; }
  .cand img { width: 100%; border-radius: 6px; background: #0d1117; min-height: 60px; }
  .cand .lab { font-size: 13px; font-weight: 700; margin-bottom: 5px; }
  .cand .src { color: #667085; font-size: 12px; }
  .cand.pick { outline: 3px solid #12b76a; }
  .stage { margin-top: 10px; font-weight: 700; color: #175cd3; }
  .keyrow { color: #667085; font-size: 12px; margin-top: 6px; }
  code { background: #eef1f5; padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>
<div id="banner"></div>
<header>
  <b>地面零散垃圾历史主动挖掘 v2</b>
  <span class="meta" id="unitmeta">加载中…</span>
  <span id="progress"></span>
</header>
<main>
  <section class="card">
    <h3>上下文（source-native 原帧，含 ROI 与代表框）</h3>
    <img id="ctx" alt="context">
  </section>
  <section class="card">
    <h3>代表裁剪（可切换多观测）</h3>
    <div class="tabs" id="tabs"></div>
    <img id="crop" alt="crop">
    <div class="stage" id="stage"></div>
    <div class="verdicts" id="verdicts"></div>
    <div class="hint" id="hint"></div>
    <div class="keyrow">R 必需垃圾 · I 微小忽略 · U 不确定 · N 非垃圾/背景 ｜ 定位 1/2/3，0 无法定位 ｜ S 同一 / W 新物体 / D 不确定 ｜ ← → / Backspace 前后切换</div>
  </section>
</main>
<section class="card" style="margin:0 16px 20px">
  <h3>候选与来源（blind 单位分类后才显示）</h3>
  <div class="blind" id="blindnote" style="display:none">模型输出已隐藏：请先记录判断，随后自动显示候选。</div>
  <div class="cands" id="cands"></div>
</section>

<script>
const LABELS = {R:"必需垃圾", I:"微小忽略", U:"不确定", N:"非垃圾/背景"};
const ROLE_LABELS = {normal:"正常", shadow_change:"阴影变化", low_contrast:"低对比",
  pose_change:"姿态变化", mild_occlusion:"轻微遮挡", background_change:"背景变化"};
const state = { queue: [], index: 0, unit: null, batch: 1 };

function banner(msg) {
  const el = document.getElementById("banner");
  if (!msg) { el.style.display = "none"; el.textContent = ""; return; }
  el.textContent = "请求失败：" + msg;
  el.style.display = "block";
}

async function api(path, options) {
  let res, data = null;
  try {
    res = await fetch(path, options);
    const text = await res.text();
    data = text ? JSON.parse(text) : null;
  } catch (err) {
    banner(path + " — " + err);
    throw err;
  }
  if (!res.ok) {
    const msg = (data && (data.error || data.message)) || ("HTTP " + res.status);
    banner(path + " — " + msg);
    throw new Error(msg);
  }
  banner(null);
  return data;
}

function post(path, body) {
  return api(path, { method: "POST", headers: {"Content-Type": "application/json"},
                     body: JSON.stringify(body) });
}

function unitLabel(u) {
  const qi = (u.queue_index === null || u.queue_index === undefined) ? state.index : u.queue_index;
  return "机位 " + (u.camera_id || "?") + " · " + (u.timestamp || "?") +
         " · " + (u.tier || "?") + " · 组 " + qi +
         (u.blind ? " · blind" : "") + (u.suspected_same_object ? " · 疑似同一物体" : "");
}

async function loadQueue() {
  const st = await api("/api/state");
  state.batch = st.batch;
  const q = await api("/api/queue?batch=" + state.batch + "&offset=0&limit=1000");
  state.queue = q.units || [];
  document.getElementById("progress").textContent =
    "进度 " + st.progress.decided + " / " + st.progress.total +
    "（完整 " + st.progress.complete + "，剩余 " + st.progress.remaining + "）";
  const firstOpen = state.queue.findIndex(x => !x.complete);
  state.index = firstOpen >= 0 ? firstOpen : 0;
}

async function showUnit() {
  if (!state.queue.length) { banner("队列为空"); return; }
  state.index = Math.max(0, Math.min(state.queue.length - 1, state.index));
  const uid = state.queue[state.index].unit_id;
  const u = await api("/api/unit?unit_id=" + encodeURIComponent(uid));
  state.unit = u;
  render(u);
}

function render(u) {
  document.getElementById("unitmeta").textContent = unitLabel(u);
  document.getElementById("ctx").src = u.images.context_url + "?t=" + Date.now();
  const crop = document.getElementById("crop");
  const obsList = u.observations || [];
  let activeObs = null;
  const tabs = document.getElementById("tabs");
  tabs.innerHTML = "";
  if (obsList.length > 1) {
    obsList.forEach((o, i) => {
      const b = document.createElement("button");
      b.textContent = (ROLE_LABELS[o.role] || o.role || "观测") + " #" + (i + 1);
      if (i === 0) b.classList.add("on");
      b.onclick = () => {
        [...tabs.children].forEach(c => c.classList.remove("on"));
        b.classList.add("on");
        crop.src = (u.images.observation_urls || {})[o.observation_id] + "?t=" + Date.now();
      };
      tabs.appendChild(b);
    });
    activeObs = obsList[0];
  }
  const rep = u.representative_observation_id;
  if (activeObs) {
    crop.src = (u.images.observation_urls || {})[activeObs.observation_id] + "?t=" + Date.now();
  } else {
    crop.src = u.images.crop_url + "?t=" + Date.now();
  }

  const stage = document.getElementById("stage");
  if (u.stage === "classify") {
    stage.textContent = "阶段：分类判断";
  } else if (u.stage === "bbox") {
    stage.textContent = "阶段：定位选择（1/2/3 选候选，0 = 无法定位）";
  } else {
    stage.textContent = "阶段：完成（" + (LABELS[u.decision] || u.decision || "") + "）";
  }

  const vb = document.getElementById("verdicts");
  vb.innerHTML = "";
  ["R", "I", "U", "N"].forEach(code => {
    const b = document.createElement("button");
    b.className = "b" + code;
    b.textContent = code + " " + LABELS[code];
    if (u.decision === code) b.disabled = true;
    b.onclick = () => decide(code);
    vb.appendChild(b);
  });

  const hint = document.getElementById("hint");
  if (u.stage === "bbox") {
    const parts = [];
    (u.candidates || []).forEach((c, i) => {
      parts.push((i + 1) + "=" + (c.candidate_id || "?") +
                 (c.label ? " " + c.label : " " + (c.source || "")));
    });
    hint.textContent = "定位候选：" + (parts.join(" · ") || "无（只能按 0 无法定位）");
  } else if (u.suspected_same_object) {
    hint.textContent = "疑似同一物体：S 同一 / W 新物体 / D 不确定" +
      (u.link_decision ? "（已记录：" + u.link_decision + "）" : "");
  } else {
    hint.textContent = "";
  }

  const blind = document.getElementById("blindnote");
  blind.style.display = (u.blind && !u.model_revealed) ? "block" : "none";

  const cands = document.getElementById("cands");
  cands.innerHTML = "";
  const revealedCands = u.model_revealed ? (u.candidates || []) : [];
  if (!revealedCands.length) {
    const p = document.createElement("div");
    p.className = "src";
    p.textContent = u.blind && !u.model_revealed ? "（隐藏中）" : "（无候选）";
    cands.appendChild(p);
  }
  revealedCands.forEach(c => {
    const div = document.createElement("div");
    div.className = "cand" + (u.bbox_choice && u.candidates &&
      (u.candidates[u.bbox_choice - 1] || {}).candidate_id === c.candidate_id ? " pick" : "");
    const lab = document.createElement("div");
    lab.className = "lab";
    lab.textContent = c.label || ((c.candidate_id || "?") + " · " + (c.source || ""));
    const img = document.createElement("img");
    img.src = (u.images.candidate_urls || {})[c.candidate_id] + "?t=" + Date.now();
    const src = document.createElement("div");
    src.className = "src";
    src.textContent = "来源 " + (c.source || "?") + (c.contains_point ? " · 含代表点" : "");
    div.appendChild(lab); div.appendChild(img); div.appendChild(src);
    cands.appendChild(div);
  });
}

function next() {
  if (state.index < state.queue.length - 1) { state.index += 1; showUnit().catch(() => {}); }
  else { banner(null); document.getElementById("stage").textContent = "已是最后一组"; }
}
function move(delta) { state.index += delta; showUnit().catch(() => {}); }

async function decide(code) {
  const u = state.unit;
  if (!u) return;
  try {
    const res = await post("/api/decide", { unit_id: u.unit_id, verdict: code });
    if (res.stage === "bbox") {
      await refreshProgress();
      await showUnit();            // re-fetch: blind units reveal candidates now
    } else {
      state.queue[state.index].complete = !!res.complete;
      state.queue[state.index].verdict = res.decision;
      await refreshProgress();
      next();
    }
  } catch (err) { /* banner already shown; never advance silently */ }
}

async function chooseBbox(choice) {
  const u = state.unit;
  if (!u) return;
  try {
    const res = await post("/api/bbox", { unit_id: u.unit_id, choice: choice });
    state.queue[state.index].complete = !!res.complete;
    await refreshProgress();
    next();
  } catch (err) { /* banner already shown */ }
}

async function chooseLink(decision) {
  const u = state.unit;
  if (!u) return;
  try {
    await post("/api/link", { unit_id: u.unit_id, decision: decision });
    await showUnit();
  } catch (err) { /* banner already shown */ }
}

async function refreshProgress() {
  try {
    const st = await api("/api/state");
    document.getElementById("progress").textContent =
      "进度 " + st.progress.decided + " / " + st.progress.total +
      "（完整 " + st.progress.complete + "，剩余 " + st.progress.remaining + "）";
  } catch (err) { /* banner already shown */ }
}

document.addEventListener("keydown", ev => {
  if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA")) return;
  const u = state.unit;
  if (ev.key === "ArrowRight") { move(1); return; }
  if (ev.key === "ArrowLeft" || ev.key === "Backspace") { ev.preventDefault(); move(-1); return; }
  if (!u) return;
  const key = (ev.key || "").toLowerCase();
  if (u.stage === "classify") {
    const vmap = {r: "R", i: "I", u: "U", n: "N"};
    if (vmap[key]) { decide(vmap[key]); return; }
  }
  if (u.stage === "bbox" && "0123".includes(ev.key)) { chooseBbox(parseInt(ev.key, 10)); return; }
  if (u.suspected_same_object) {
    const lmap = {s: "SAME", w: "NEW", d: "UNCERTAIN"};
    if (lmap[key]) { chooseLink(lmap[key]); return; }
  }
});

(async function init() {
  try { await loadQueue(); await showUnit(); }
  catch (err) { /* banner already shown */ }
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", type=Path, required=True,
                   help="artifact root holding coarse_frames.jsonl / review_units.jsonl")
    # ``serve`` is the default: running with no subcommand starts the UI.
    sub = p.add_subparsers(dest="command", required=False)
    serve = sub.add_parser("serve", help="serve the review UI (default)")
    serve.add_argument("--bind", default=DEFAULT_BIND)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub.add_parser("status", help="print progress JSON")
    sub.add_parser("selftest", help="offline blindness + isolation self-test")
    return p


def build(artifact: Path):
    store = ReviewStore(artifact)
    renderer = MediaRenderer(store)
    return store, renderer


def cmd_serve(args) -> int:
    from http.server import ThreadingHTTPServer

    store, renderer = build(args.artifact)
    handler = make_handler(store, renderer)
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    server.daemon_threads = True
    print(f"Ground Litter Historical Review v2 on http://{args.bind}:{args.port}/")
    print(f"artifact={store.artifact}")
    print(f"units={len(store.units)} frames={len(store.frames)} batch={store.default_batch()}")
    print(f"cv2={'available' if renderer.available() else 'UNAVAILABLE (media endpoints return 503)'}")
    print("isolated: writes stay under the artifact root; official 8801 is never contacted")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_status(args) -> int:
    store, renderer = build(args.artifact)
    payload = store.state()
    payload["cv2_available"] = renderer.available()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_selftest(args) -> int:
    """Blindness, queue and isolation self-test.  Never needs the network or cv2."""
    store, renderer = build(args.artifact)
    failures: list[str] = []

    if not store.units:
        failures.append("no review units loaded")
    order = [int(u.get("queue_index") or 0) for u in store.units]
    if order != sorted(order):
        failures.append("review units are not ordered by queue_index")

    for unit in store.units:
        uid = str(unit["unit_id"])
        payload = store.unit_payload(uid)
        if payload["unit_id"] != uid:
            failures.append(f"payload unit_id mismatch for {uid}")
        if not str(payload["images"]["context_url"]).startswith("/media/"):
            failures.append(f"context_url not a local media path for {uid}")
        if unit.get("blind") and not store.model_revealed(unit):
            if payload["candidates"]:
                failures.append(f"blindness leak: candidates serialized for {uid}")
            if payload["images"]["candidate_urls"]:
                failures.append(f"blindness leak: candidate urls for {uid}")
            if payload["model_revealed"]:
                failures.append(f"blindness leak: model_revealed true for {uid}")

    # Every declared write target must resolve inside the artifact root.
    try:
        resolve_under(store.artifact, "../escape.jsonl")
        failures.append("traversal guard did not fire")
    except ClientError:
        pass
    try:
        resolve_under(store.artifact, "/etc/passwd")
        failures.append("absolute-path guard did not fire")
    except ClientError:
        pass

    for forbidden in ("/home/sf01/step2c1-blind-truth/artifact", "/tmp/sealed-thing"):
        try:
            assert_artifact_root(forbidden)
            failures.append(f"forbidden root accepted: {forbidden}")
        except ReviewError:
            pass

    result = {
        "ok": not failures,
        "artifact": str(store.artifact),
        "units": len(store.units),
        "frames": len(store.frames),
        "blind_units": sum(1 for u in store.units if u.get("blind")),
        "cv2_available": renderer.available(),
        "progress": store.progress(),
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    command = args.command or "serve"
    if command == "serve":
        if not getattr(args, "bind", None):
            args.bind = DEFAULT_BIND
        if not getattr(args, "port", None):
            args.port = DEFAULT_PORT
        return cmd_serve(args)
    if command == "status":
        return cmd_status(args)
    if command == "selftest":
        return cmd_selftest(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
