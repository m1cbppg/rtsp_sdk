"""Step 1C-2M: complete multi-target positive tiles (Missing-Required salvage).

Step 1C-2 showed that most failed tiles are not wrong primary boxes but *partial*
annotation: a second or third REQUIRED_LITTER is visible in the same 640x640 tile and
had no box.  This step keeps the tile exactly as reviewed and lets a human add the
missing targets, without ever drawing a box:

    point click -> machine A/B/C proposals -> human selects -> repeat -> recheck

Hard boundaries:

* only the MISSING_REQUIRED tiles are processed; the 29 accepted tiles and the 2
  BOX_PROBLEM tiles are never re-reviewed;
* the crop never moves, the PNG is never re-decoded, re-cropped, resized or re-encoded,
  and no original (verified) bbox is modified;
* a supplemental target is a training annotation, not a Gold episode: no cross-frame or
  temporal identity is inferred;
* the screening / historical geometry of Step 1C-1 is never reused as a label;
* nothing is auto-selected: a proposal only becomes a label after an explicit click, and
  a tile becomes training data only after an explicit final recheck.

The module is stdlib-only; the proposal engine and the image files are injected, so the
queue, coordinate, duplicate, truncation, state and build logic is fully testable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .ground_litter_localization_review import (  # reuse, do not duplicate
    ReviewError,
    SealedAssetError,
    assert_not_sealed,
    atomic_write_json,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from .ground_litter_positive_tiles import (  # reuse, do not duplicate
    CLASS_ID,
    CLASS_NAME,
    TILE_SIZE,
    PositiveTileError,
    canonical_tile_id,
    label_txt_lines,
    write_bytes_atomic,
)

SCHEMA_VERSION = "ground_litter_tile_completion_v1"
GENERATOR_VERSION = "step1c2m-1.0.0"
STATE_SCHEMA_VERSION = "tile_completion_state_v1"

MISSING = "MISSING_REQUIRED"
COMPLETE = "ANNOTATION_COMPLETE"
BOX_PROBLEM = "BOX_PROBLEM"

TILE_STATUS_NEEDS = "NEEDS_COMPLETION"
TILE_STATUS_COMPLETE = "ANNOTATION_COMPLETE_AFTER_SUPPLEMENT"
TILE_STATUS_STILL_MISSING = "STILL_MISSING_REQUIRED"
TILE_STATUS_UNRESOLVED = "SUPPLEMENTAL_LOCALIZATION_UNRESOLVED"
TILE_STATUS_TRUNCATED = "TARGET_TRUNCATED"
TILE_STATUS_UNCERTAIN = "UNCERTAIN_COMPLETENESS"
TILE_STATUSES = (TILE_STATUS_NEEDS, TILE_STATUS_COMPLETE, TILE_STATUS_STILL_MISSING,
                 TILE_STATUS_UNRESOLVED, TILE_STATUS_TRUNCATED, TILE_STATUS_UNCERTAIN)
FINAL_TILE_STATUSES = TILE_STATUSES[1:]

LOCALIZATION_VERIFIED = "VERIFIED_BBOX"
LOCALIZATION_UNRESOLVED = "SUPPLEMENTAL_LOCALIZATION_UNRESOLVED"
LOCALIZATION_TRUNCATED = "TARGET_TRUNCATED_BY_TILE"
LOCALIZATION_STATUSES = (LOCALIZATION_VERIFIED, LOCALIZATION_UNRESOLVED,
                         LOCALIZATION_TRUNCATED)

MAX_SUPPLEMENTAL_PER_TILE = 20
MAX_PROPOSAL_REVISIONS = 2
DUPLICATE_IOU = 0.80
LABEL_DEDUP_IOU = 0.95
REJECT_UNRESOLVED = "REJECT_UNRESOLVED"


class CompletionError(RuntimeError):
    """Step 1C-2M could not proceed."""


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
    area_b = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_contains_point(box: Sequence[float], point: Sequence[float]) -> bool:
    return (float(box[0]) <= float(point[0]) <= float(box[2])
            and float(box[1]) <= float(point[1]) <= float(box[3]))


def normalize_box(box: Sequence[float] | None) -> list[float] | None:
    if not box or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(v == v for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def tile_to_source(box_or_point: Sequence[float], crop_xyxy: Sequence[float]
                   ) -> list[float]:
    """tile coordinates -> source coordinates (crop origin is the only offset)."""
    x1, y1 = float(crop_xyxy[0]), float(crop_xyxy[1])
    return [round(float(v) + (x1 if index % 2 == 0 else y1), 3)
            for index, v in enumerate(box_or_point)]


def source_to_tile(box_or_point: Sequence[float], crop_xyxy: Sequence[float]
                   ) -> list[float]:
    x1, y1 = float(crop_xyxy[0]), float(crop_xyxy[1])
    return [round(float(v) - (x1 if index % 2 == 0 else y1), 3)
            for index, v in enumerate(box_or_point)]


def validate_tile_box(box: Sequence[float], *, size: int = TILE_SIZE
                      ) -> list[float] | None:
    """Return a clean tile box or None when it is degenerate / outside the tile."""
    clean = normalize_box(box)
    if clean is None:
        return None
    if not (0 <= clean[0] < clean[2] <= size and 0 <= clean[1] < clean[3] <= size):
        return None
    return [round(v, 3) for v in clean]


def validate_source_box(box: Sequence[float], source_width: int, source_height: int,
                        *, size: int = TILE_SIZE) -> list[float] | None:
    clean = normalize_box(box)
    if clean is None:
        return None
    if not (0 <= clean[0] and 0 <= clean[1]
            and clean[2] <= source_width and clean[3] <= source_height):
        return None
    if clean[2] - clean[0] > size or clean[3] - clean[1] > size:
        return None
    return [round(v, 3) for v in clean]


def truncation_for_box(box: Sequence[float], *, size: int = TILE_SIZE,
                       tolerance: float = 1.0) -> list[str]:
    """Which tile sides the box touches; a touched side means possible truncation."""
    sides = []
    if float(box[0]) <= tolerance:
        sides.append("left")
    if float(box[1]) <= tolerance:
        sides.append("top")
    if float(box[2]) >= size - tolerance:
        sides.append("right")
    if float(box[3]) >= size - tolerance:
        sides.append("bottom")
    return sides


def yolo_from_tile_box(box: Sequence[float], *, size: int = TILE_SIZE) -> list[float]:
    width = float(box[2]) - float(box[0])
    height = float(box[3]) - float(box[1])
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    values = [cx / size, cy / size, width / size, height / size]
    return [round(min(1.0, max(0.0, v)), 6) for v in values]


def size_bucket(short_side: float) -> str:
    if short_side < 10:
        return "<10"
    if short_side < 20:
        return "10-19"
    if short_side < 40:
        return "20-39"
    if short_side < 80:
        return "40-79"
    return "80+"


def supplemental_target_id(tile_id: str, ordinal: int) -> str:
    """Deterministic, stable supplemental id (never a random UUID)."""
    material = f"step1c2m|{tile_id}|{int(ordinal)}"
    return "st-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


# --------------------------------------------------------------------------- #
# input (§2)
# --------------------------------------------------------------------------- #


def load_completion_input(step1c2_root: Path | str) -> dict[str, Any]:
    root = Path(step1c2_root)
    assert_not_sealed(root)
    tiles_path = root / "tile_candidates.jsonl"
    summary_path = root / "SUMMARY.json"
    manifest_path = root / "MANIFEST.json"
    review_path = root / "review_state.json"
    for path in (tiles_path, summary_path, manifest_path, review_path):
        if not path.is_file():
            raise FileNotFoundError(f"required Step 1C-2 artifact missing: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    decisions = dict(review.get("decisions") or {})
    rows = read_jsonl(tiles_path)
    by_id = {str(row["tile_id"]): row for row in rows}

    missing: list[dict[str, Any]] = []
    complete: list[dict[str, Any]] = []
    box_problem: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    for tile_id, row in sorted(by_id.items()):
        record = decisions.get(tile_id) or {}
        status = str(record.get("annotation_review_status") or "")
        image_path = row.get("image_path")
        image_exists = bool(image_path) and Path(str(image_path)).is_file()
        if not image_exists:
            problems.append({"tile_id": tile_id, "field": "candidate_png_missing",
                             "value": image_path})
        elif sha256_file(Path(str(image_path))) != row.get("image_sha256"):
            problems.append({"tile_id": tile_id, "field": "candidate_png_hash_drift",
                             "value": image_path})
        if not row.get("source_crop_xyxy"):
            problems.append({"tile_id": tile_id, "field": "crop_missing", "value": None})
        if status == MISSING:
            missing.append(row)
        elif status == COMPLETE:
            complete.append(row)
        elif status == BOX_PROBLEM:
            box_problem.append(row)
        else:
            other.append(row)

    accepted_manifest_path = root / "positive_training_manifest.jsonl"
    accepted_rows = (read_jsonl(accepted_manifest_path)
                     if accepted_manifest_path.is_file() else [])
    if not accepted_rows:
        problems.append({"tile_id": None, "field": "positive_training_manifest_missing",
                         "value": str(accepted_manifest_path)})
    for row in accepted_rows:
        image = Path(str(row.get("image_path") or ""))
        if not image.is_file():
            problems.append({"tile_id": row.get("tile_id"),
                             "field": "accepted_png_missing", "value": str(image)})
        elif sha256_file(image) != row.get("image_sha256"):
            problems.append({"tile_id": row.get("tile_id"),
                             "field": "accepted_png_hash_drift", "value": str(image)})

    if manifest.get("tile_candidates_sha256") not in (None, sha256_file(tiles_path)):
        problems.append({"tile_id": None, "field": "tile_candidates_hash_mismatch",
                         "value": manifest.get("tile_candidates_sha256")})
    if manifest.get("summary_sha256") not in (None, sha256_file(summary_path)):
        problems.append({"tile_id": None, "field": "summary_hash_mismatch",
                         "value": manifest.get("summary_sha256")})

    return {
        "root": root,
        "tiles_path": tiles_path,
        "tiles_sha256": sha256_file(tiles_path),
        "summary_path": summary_path,
        "summary_sha256": sha256_file(summary_path),
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "review_state_path": review_path,
        "review_state_sha256": sha256_file(review_path),
        "accepted_manifest_path": accepted_manifest_path,
        "accepted_manifest_sha256": (sha256_file(accepted_manifest_path)
                                     if accepted_manifest_path.is_file() else None),
        "summary": summary,
        "manifest": manifest,
        "review": review,
        "decisions": decisions,
        "tiles": rows,
        "by_id": by_id,
        "missing_required": missing,
        "complete": complete,
        "box_problem": box_problem,
        "other": other,
        "accepted_rows": accepted_rows,
        "problems": problems,
    }


def verify_preflight(data: Mapping[str, Any]) -> dict[str, Any]:
    counts = {
        "step1c2_annotation_complete": len(data["complete"]),
        "step1c2_missing_required": len(data["missing_required"]),
        "step1c2_box_problem": len(data["box_problem"]),
        "step1c2_other": len(data["other"]),
        "step1c2_total": len(data["tiles"]),
        "old_accepted_positive_tile_count": len(data["accepted_rows"]),
    }
    expected = {
        "step1c2_annotation_complete": 29,
        "step1c2_missing_required": 62,
        "step1c2_box_problem": 2,
        "step1c2_other": 0,
        "step1c2_total": 93,
        "old_accepted_positive_tile_count": 29,
    }
    return {
        **counts,
        "expected": expected,
        "counts_match": counts == expected,
        "candidate_png_count": sum(1 for row in data["tiles"] if row.get("image_path")),
        "problem_count": len(data["problems"]),
        "problems": list(data["problems"])[:20],
        "hashes": {
            "step1c2_summary_sha256": data["summary_sha256"],
            "step1c2_manifest_sha256": data["manifest_sha256"],
            "tile_candidates_sha256": data["tiles_sha256"],
            "review_state_sha256": data["review_state_sha256"],
            "positive_training_manifest_sha256": data["accepted_manifest_sha256"],
        },
    }


def queue_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The 62 MISSING_REQUIRED tiles, ready for the completion UI."""
    rows = []
    for row in sorted(data["missing_required"],
                      key=lambda item: (item["camera_id"], item["primary_episode_id"])):
        labels = list(row.get("labels") or [])
        rows.append({
            "tile_id": row["tile_id"],
            "primary_episode_id": row["primary_episode_id"],
            "primary_episode_ids": row.get("primary_episode_ids") or [],
            "camera_id": row["camera_id"],
            "source_file_id": row.get("source_file_id"),
            "step1c0_decoded_timestamp": row.get("step1c0_decoded_timestamp"),
            "source_crop_xyxy": row.get("source_crop_xyxy"),
            "source_width": row.get("source_width"),
            "source_height": row.get("source_height"),
            "image_path": row.get("image_path"),
            "image_sha256": row.get("image_sha256"),
            "existing_label_count": len(labels),
            "existing_labels": labels,
            "labels": labels,          # same list: merged_labels() reads "labels"
            "known_unlocalized_required_present":
                bool(row.get("known_unlocalized_required_present")),
            "risk_flags": list(row.get("risk_flags") or []),
            "review_note": ((data["decisions"].get(row["tile_id"]) or {})
                            .get("annotation_review_note") or ""),
        })
    return rows


# --------------------------------------------------------------------------- #
# supplemental target records (§5/§15/§20)
# --------------------------------------------------------------------------- #


def build_supplemental_target(*, tile_id: str, ordinal: int, tile_row: Mapping[str, Any],
                              point_tile: Sequence[float], proposal_revision: int) -> dict[str, Any]:
    crop = tile_row["source_crop_xyxy"]
    point_tile_clean = [round(float(point_tile[0]), 3), round(float(point_tile[1]), 3)]
    point_source = tile_to_source(point_tile_clean, crop)
    return {
        "supplemental_target_id": supplemental_target_id(tile_id, ordinal),
        "ordinal": int(ordinal),
        "tile_id": tile_id,
        "source_file_id": tile_row.get("source_file_id"),
        "camera_id": tile_row.get("camera_id"),
        "timestamp": tile_row.get("step1c0_decoded_timestamp"),
        "crop_source_xyxy": list(crop),
        "source_width": int(tile_row.get("source_width") or 0),
        "source_height": int(tile_row.get("source_height") or 0),
        "original_click": {
            "tile_x": point_tile_clean[0], "tile_y": point_tile_clean[1],
            "source_x": point_source[0], "source_y": point_source[1],
        },
        "click_history": [{
            "revision": int(proposal_revision),
            "tile_x": point_tile_clean[0], "tile_y": point_tile_clean[1],
            "source_x": point_source[0], "source_y": point_source[1],
        }],
        "proposal_revision": int(proposal_revision),
        "proposal_candidates": [],
        "proposal_error": None,
        "selected_proposal": None,
        "verified_source_xyxy": None,
        "verified_tile_xyxy": None,
        "class_id": CLASS_ID,
        "class_name": CLASS_NAME,
        "localization_status": None,
        "truncated_sides": [],
        "duplicate_of": [],
        "duplicate_confirmed_independent": False,
        "created_at": None,
        "updated_at": None,
    }


PROPOSAL_LETTERS = ("A", "B", "C")


def set_proposals(target: dict[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the engine result to a target.  Proposals never become labels here.

    Accepts either the engine's normalised candidate shape (``bbox_tile_xyxy``) or a
    raw ``{"bbox": ...}`` row, so a test double and the real engine behave the same.
    """
    point = (target.get("current_click") or target["original_click"])
    point_tile = [float(point["tile_x"]), float(point["tile_y"])]
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(result.get("candidates") or []):
        box = row.get("bbox_tile_xyxy") or row.get("bbox")
        if not box:
            continue
        box = [round(float(v), 3) for v in box]
        source = row.get("bbox_source_xyxy")
        candidates.append({
            "letter": str(row.get("letter") or PROPOSAL_LETTERS[index]),
            "bbox_tile_xyxy": box,
            "bbox_source_xyxy": ([round(float(v), 3) for v in source] if source
                                 else tile_to_source(box, target["crop_source_xyxy"])),
            "method": str(row.get("method") or "unknown"),
            "contains_point": bool(row.get("contains_point")
                                   if row.get("contains_point") is not None
                                   else box_contains_point(box, point_tile)),
            "touches_border": row.get("touches_border"),
        })
    target["proposal_candidates"] = candidates
    target["proposal_ok"] = bool(result.get("ok"))
    target["proposal_error"] = (None if result.get("ok")
                                else {"error_code": result.get("error_code"),
                                      "message": result.get("message")})
    return target


def duplicate_report(bbox_tile: Sequence[float], point_tile: Sequence[float],
                     others: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """§17: warn when a new box duplicates or swallows an existing target."""
    matches: list[dict[str, Any]] = []
    for other in others:
        box = other.get("tile_xyxy")
        if not box:
            continue
        overlap = box_iou(bbox_tile, box)
        point_inside = box_contains_point(box, point_tile)
        if overlap >= DUPLICATE_IOU or point_inside:
            matches.append({
                "label_id": other.get("label_id"),
                "tile_xyxy": [round(float(v), 3) for v in box],
                "iou": round(overlap, 4),
                "point_inside": bool(point_inside),
            })
    matches.sort(key=lambda item: (-item["iou"], str(item["label_id"])))
    return {
        "possible_duplicate": bool(matches),
        "matches": matches,
        "reason": ("same_box" if matches and matches[0]["iou"] >= 0.95
                   else ("point_inside_existing" if matches and matches[0]["point_inside"]
                         else ("high_iou" if matches else None))),
    }


# --------------------------------------------------------------------------- #
# completion state (§21/§34)
# --------------------------------------------------------------------------- #


class CompletionState:
    """Per-tile completion progress with an append-only audit trail."""

    def __init__(self, path: Path, *, input_fingerprint: str = "") -> None:
        self.path = Path(path)
        self.input_fingerprint = input_fingerprint
        self.tiles: dict[str, dict[str, Any]] = {}
        self.audit_trail: list[dict[str, Any]] = []

    # -- persistence --------------------------------------------------------- #

    @classmethod
    def load(cls, path: Path | str, *, input_fingerprint: str = "") -> "CompletionState":
        target = Path(path)
        state = cls(target, input_fingerprint=input_fingerprint)
        if not target.is_file():
            return state
        payload = json.loads(target.read_text(encoding="utf-8"))
        state.tiles = dict(payload.get("tiles") or {})
        state.audit_trail = list(payload.get("audit_trail") or [])
        recorded = str(payload.get("input_fingerprint") or "")
        state.input_fingerprint = recorded or input_fingerprint
        if input_fingerprint and recorded and recorded != input_fingerprint:
            raise ReviewError("completion state belongs to a different Step 1C-2 input")
        return state

    def save(self) -> None:
        atomic_write_json(self.path, {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "input_fingerprint": self.input_fingerprint,
            "tiles": self.tiles,
            "audit_trail": self.audit_trail,
        })

    def _record(self, tile_id: str, action: str, payload: Mapping[str, Any]) -> None:
        from datetime import datetime

        self.audit_trail.append({
            "at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tile_id": tile_id, "action": action, "payload": dict(payload)})

    def _stamp(self) -> str:
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    # -- tile access --------------------------------------------------------- #

    def tile(self, tile_id: str) -> dict[str, Any]:
        entry = self.tiles.get(tile_id)
        if entry is None:
            entry = {"tile_id": tile_id, "status": TILE_STATUS_NEEDS,
                     "supplemental_targets": [], "next_ordinal": 1,
                     "recheck": None, "note": "", "updated_at": None}
            self.tiles[tile_id] = entry
        return entry

    def targets(self, tile_id: str) -> list[dict[str, Any]]:
        return list(self.tile(tile_id)["supplemental_targets"])

    def target(self, tile_id: str, target_id: str) -> dict[str, Any]:
        for item in self.tile(tile_id)["supplemental_targets"]:
            if item["supplemental_target_id"] == target_id:
                return item
        raise ReviewError(f"unknown supplemental target {target_id}")

    def _touch(self, tile_id: str) -> None:
        entry = self.tile(tile_id)
        entry["updated_at"] = self._stamp()

    # -- operations ---------------------------------------------------------- #

    def add_target(self, tile_row: Mapping[str, Any], *, point_tile: Sequence[float],
                   proposal_result: Mapping[str, Any],
                   revision: int = 1) -> dict[str, Any]:
        tile_id = str(tile_row["tile_id"])
        entry = self.tile(tile_id)
        if len(entry["supplemental_targets"]) >= MAX_SUPPLEMENTAL_PER_TILE:
            raise ReviewError(
                f"{tile_id} already has {MAX_SUPPLEMENTAL_PER_TILE} supplemental targets")
        if not (0 <= float(point_tile[0]) < TILE_SIZE
                and 0 <= float(point_tile[1]) < TILE_SIZE):
            raise ReviewError("the click must be inside the 640x640 tile")
        ordinal = int(entry.get("next_ordinal") or 1)
        target = build_supplemental_target(
            tile_id=tile_id, ordinal=ordinal, tile_row=tile_row,
            point_tile=point_tile, proposal_revision=int(revision))
        target["created_at"] = self._stamp()
        set_proposals(target, proposal_result)
        entry["next_ordinal"] = ordinal + 1
        entry["supplemental_targets"].append(target)
        entry["status"] = TILE_STATUS_NEEDS
        entry["recheck"] = None
        self._touch(tile_id)
        self._record(tile_id, "add_supplemental", {
            "supplemental_target_id": target["supplemental_target_id"],
            "point_tile": target["original_click"], "revision": target["proposal_revision"],
            "proposal_ok": target.get("proposal_ok"),
            "proposal_count": len(target["proposal_candidates"])})
        self.save()
        return target

    def repoint(self, tile_row: Mapping[str, Any], target_id: str, *,
                point_tile: Sequence[float],
                proposal_result: Mapping[str, Any]) -> dict[str, Any]:
        tile_id = str(tile_row["tile_id"])
        target = self.target(tile_id, target_id)
        revision = int(target["proposal_revision"]) + 1
        if revision > MAX_PROPOSAL_REVISIONS:
            raise ReviewError(
                f"{target_id} already used {MAX_PROPOSAL_REVISIONS} proposal revisions")
        crop = target["crop_source_xyxy"]
        point = [round(float(point_tile[0]), 3), round(float(point_tile[1]), 3)]
        target["proposal_revision"] = revision
        target["click_history"].append({
            "revision": revision, "tile_x": point[0], "tile_y": point[1],
            "source_x": tile_to_source(point, crop)[0],
            "source_y": tile_to_source(point, crop)[1]})
        target["original_click"] = target["click_history"][0]
        target["current_click"] = target["click_history"][-1]
        target["selected_proposal"] = None
        target["verified_source_xyxy"] = None
        target["verified_tile_xyxy"] = None
        target["localization_status"] = None
        target["truncated_sides"] = []
        target["duplicate_of"] = []
        target["duplicate_confirmed_independent"] = False
        set_proposals(target, proposal_result)
        self._touch(tile_id)
        self._record(tile_id, "repoint", {
            "supplemental_target_id": target_id, "revision": revision,
            "point_tile": point,
            "proposal_ok": target.get("proposal_ok"),
            "proposal_count": len(target["proposal_candidates"])})
        self.save()
        return target

    def select_proposal(self, tile_id: str, target_id: str, letter: str, *,
                        existing_boxes: Sequence[Mapping[str, Any]],
                        confirm_independent: bool = False) -> dict[str, Any]:
        target = self.target(tile_id, target_id)
        candidates = {str(row["letter"]): row for row in target["proposal_candidates"]}
        if letter not in candidates:
            raise ReviewError(f"{target_id} has no proposal {letter!r}")
        chosen = candidates[letter]
        tile_box = validate_tile_box(chosen["bbox_tile_xyxy"])
        if tile_box is None:
            target["localization_status"] = LOCALIZATION_UNRESOLVED
            self._touch(tile_id)
            self._record(tile_id, "select_proposal_rejected",
                         {"supplemental_target_id": target_id, "letter": letter,
                          "reason": "bbox_outside_tile"})
            self.save()
            raise ReviewError(f"proposal {letter} is not a usable tile box")
        width = int(target.get("source_width") or 0)
        height = int(target.get("source_height") or 0)
        if width <= 0 or height <= 0:      # defensive: never invent frame bounds
            raise ReviewError(f"{tile_id} has no source frame size recorded")
        source_box = validate_source_box(chosen["bbox_source_xyxy"], width, height)
        if source_box is None:
            raise ReviewError(f"proposal {letter} is not a usable source box")

        click = target.get("current_click") or target["original_click"]
        report = duplicate_report(tile_box, [click["tile_x"], click["tile_y"]],
                                  existing_boxes)
        if report["possible_duplicate"] and not confirm_independent:
            self._touch(tile_id)
            self._record(tile_id, "duplicate_warning",
                         {"supplemental_target_id": target_id, "letter": letter,
                          "matches": report["matches"], "reason": report["reason"]})
            self.save()
            return {"warning": "POSSIBLE_DUPLICATE_TARGET", **report,
                    "target": target}

        sides = truncation_for_box(tile_box)
        if sides:
            # the target would run into the tile edge: never silently accept it
            target["selected_proposal"] = letter
            target["verified_tile_xyxy"] = tile_box
            target["verified_source_xyxy"] = source_box
            target["truncated_sides"] = sides
            target["localization_status"] = LOCALIZATION_TRUNCATED
            target["duplicate_of"] = [match["label_id"] for match in report["matches"]]
            target["duplicate_confirmed_independent"] = bool(confirm_independent)
            self._touch(tile_id)
            self._record(tile_id, "select_proposal_truncated",
                         {"supplemental_target_id": target_id, "letter": letter,
                          "tile_xyxy": tile_box, "sides": sides})
            self.save()
            return {"warning": "TARGET_TRUNCATED_BY_TILE", "sides": sides,
                    "target": target}

        target["selected_proposal"] = letter
        target["verified_tile_xyxy"] = tile_box
        target["verified_source_xyxy"] = source_box
        target["localization_status"] = LOCALIZATION_VERIFIED
        target["truncated_sides"] = []
        target["duplicate_of"] = [match["label_id"] for match in report["matches"]]
        target["duplicate_confirmed_independent"] = bool(confirm_independent)
        self._touch(tile_id)
        self._record(tile_id, "select_proposal", {
            "supplemental_target_id": target_id, "letter": letter,
            "tile_xyxy": tile_box, "source_xyxy": source_box,
            "method": chosen["method"],
            "duplicate_of": target["duplicate_of"],
            "confirmed_independent": bool(confirm_independent)})
        self.save()
        return {"warning": None, "target": target}

    def reject_target(self, tile_id: str, target_id: str, *, reason: str = "") -> dict:
        target = self.target(tile_id, target_id)
        target["selected_proposal"] = None
        target["verified_tile_xyxy"] = None
        target["verified_source_xyxy"] = None
        target["localization_status"] = LOCALIZATION_UNRESOLVED
        self._touch(tile_id)
        self._record(tile_id, "reject_proposals",
                     {"supplemental_target_id": target_id, "reason": reason.strip()})
        self.save()
        return target

    def delete_target(self, tile_id: str, target_id: str) -> None:
        entry = self.tile(tile_id)
        before = len(entry["supplemental_targets"])
        entry["supplemental_targets"] = [
            item for item in entry["supplemental_targets"]
            if item["supplemental_target_id"] != target_id]
        if len(entry["supplemental_targets"]) == before:
            raise ReviewError(f"unknown supplemental target {target_id}")
        self._touch(tile_id)
        self._record(tile_id, "delete_supplemental",
                     {"supplemental_target_id": target_id})
        self.save()

    def clear_targets(self, tile_id: str) -> None:
        entry = self.tile(tile_id)
        entry["supplemental_targets"] = []
        entry["status"] = TILE_STATUS_NEEDS
        entry["recheck"] = None
        self._touch(tile_id)
        self._record(tile_id, "clear_supplemental", {})
        self.save()

    def recheck(self, tile_id: str, decision: str, *, note: str = "") -> dict[str, Any]:
        if decision not in FINAL_TILE_STATUSES:
            raise ReviewError(f"unknown completion decision {decision!r}")
        entry = self.tile(tile_id)
        targets = entry["supplemental_targets"]
        verified = [t for t in targets
                    if t.get("localization_status") == LOCALIZATION_VERIFIED]
        unresolved = [t for t in targets
                      if t.get("localization_status") in (None, LOCALIZATION_UNRESOLVED)]
        truncated = [t for t in targets
                     if t.get("localization_status") == LOCALIZATION_TRUNCATED]
        if decision == TILE_STATUS_COMPLETE:
            # report the most specific blocker first
            if truncated:
                raise ReviewError(
                    "cannot complete: a supplemental target is truncated by the tile")
            if unresolved:
                raise ReviewError(
                    "cannot complete: some supplemental targets have no verified bbox")
            if not verified:
                raise ReviewError("cannot complete: no supplemental target has a bbox")
        entry["recheck"] = {"decision": decision, "note": note.strip(),
                            "at": self._stamp(),
                            "supplemental_verified": len(verified),
                            "supplemental_unresolved": len(unresolved),
                            "supplemental_truncated": len(truncated)}
        entry["status"] = decision
        entry["note"] = note.strip()
        self._touch(tile_id)
        self._record(tile_id, f"recheck:{decision}",
                     {"note": note.strip(), "targets": len(targets),
                      "verified": len(verified), "unresolved": len(unresolved),
                      "truncated": len(truncated)})
        self.save()
        return entry

    def skip(self, tile_id: str) -> None:
        self._record(tile_id, "skip", {})
        self.save()

    def progress(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        reviewed = skipped = 0
        for row in rows:
            entry = self.tiles.get(str(row["tile_id"])) or {}
            status = str(entry.get("status") or TILE_STATUS_NEEDS)
            if status in FINAL_TILE_STATUSES:
                reviewed += 1
            if any(entry_item["action"] == "skip" and entry_item["tile_id"] == row["tile_id"]
                   for entry_item in self.audit_trail):
                skipped += 1
        return {"queue": len(rows), "reviewed": reviewed,
                "pending": len(rows) - reviewed, "skipped": skipped}


# --------------------------------------------------------------------------- #
# merged labels for a completed tile
# --------------------------------------------------------------------------- #


def merged_labels(tile_row: Mapping[str, Any], supplementals: Sequence[Mapping[str, Any]]
                  ) -> list[dict[str, Any]]:
    """existing verified labels UNION human-selected supplemental labels (§26)."""
    merged: list[dict[str, Any]] = []
    for index, label in enumerate(tile_row.get("labels") or []):
        box = [round(float(v), 3) for v in label["tile_xyxy"]]
        merged.append({
            "label_id": f"existing-{index}",
            "origin": "step1c2_verified",
            "episode_ids": list(label.get("episode_ids") or []),
            "supplemental_target_ids": [],
            "class_id": CLASS_ID,
            "class_name": CLASS_NAME,
            "tile_xyxy": box,
            "source_xyxy": [round(float(v), 3) for v in label["source_xyxy"]],
            "yolo_xywh_norm": list(label["yolo_xywh_norm"]),
            "source_short_side_px": label.get("source_short_side_px"),
            "size_bucket": label.get("size_bucket"),
        })
    for target in supplementals:
        if target.get("localization_status") != LOCALIZATION_VERIFIED:
            continue
        box = [round(float(v), 3) for v in target["verified_tile_xyxy"]]
        merged.append({
            "label_id": target["supplemental_target_id"],
            "origin": "step1c2m_supplemental",
            "episode_ids": [],
            "supplemental_target_ids": [target["supplemental_target_id"]],
            "class_id": CLASS_ID,
            "class_name": CLASS_NAME,
            "tile_xyxy": box,
            "source_xyxy": [round(float(v), 3) for v in target["verified_source_xyxy"]],
            "yolo_xywh_norm": yolo_from_tile_box(box),
            "source_short_side_px": round(min(box[2] - box[0], box[3] - box[1]), 3),
            "size_bucket": size_bucket(min(box[2] - box[0], box[3] - box[1])),
        })
    deduped: list[dict[str, Any]] = []
    for row in merged:
        for kept in deduped:
            if box_iou(kept["tile_xyxy"], row["tile_xyxy"]) >= LABEL_DEDUP_IOU:
                kept["episode_ids"] = sorted(set(kept["episode_ids"])
                                             | set(row["episode_ids"]))
                kept["supplemental_target_ids"] = sorted(
                    set(kept["supplemental_target_ids"])
                    | set(row["supplemental_target_ids"]))
                kept["merged_label_ids"] = sorted(
                    set(kept.get("merged_label_ids") or [kept["label_id"]])
                    | {row["label_id"]})
                break
        else:
            deduped.append(dict(row))
    deduped.sort(key=lambda item: (item["tile_xyxy"], item["label_id"]))
    for row in deduped:
        row["yolo_xywh_norm"] = yolo_from_tile_box(row["tile_xyxy"])
    return deduped


def label_lines(labels: Sequence[Mapping[str, Any]]) -> list[str]:
    return label_txt_lines(labels)


def apply_state(rows: Sequence[Mapping[str, Any]], state: CompletionState
                ) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        entry = state.tiles.get(str(row["tile_id"])) or {}
        item["completion_status"] = str(entry.get("status") or TILE_STATUS_NEEDS)
        item["supplemental_targets"] = list(entry.get("supplemental_targets") or [])
        item["recheck"] = entry.get("recheck")
        item["completion_note"] = entry.get("note") or ""
        item["merged_labels"] = merged_labels(row, item["supplemental_targets"])
        item["final_label_count"] = len(item["merged_labels"])
        item["positive_training_ready"] = item["completion_status"] == TILE_STATUS_COMPLETE
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# accepted_v2 (§23/§25/§28)
# --------------------------------------------------------------------------- #


def build_accepted_v2(data: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                      accepted_root: Path, *, state: CompletionState | None = None
                      ) -> dict[str, Any]:
    """29 frozen positives + the salvaged MISSING_REQUIRED tiles."""
    applied = apply_state(rows, state) if state is not None else list(rows)
    queue_ids = {str(row["tile_id"]) for row in rows}
    progress = (state.progress(rows) if state is not None else
                {"queue": len(rows), "reviewed": 0, "pending": len(rows), "skipped": 0})
    if progress["pending"] or progress["skipped"]:
        raise CompletionError(
            f"refusing to build: pending={progress['pending']} "
            f"skipped={progress['skipped']}")

    images_out = accepted_root / "images"
    labels_out = accepted_root / "labels"
    by_id = {str(row["tile_id"]): row for row in applied}
    rows_out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def emit(*, tile_id: str, image_source: Path, image_sha: str, camera_id: str,
             source_file_id: str, crop: Sequence[float], labels: Sequence[Mapping[str, Any]],
             primary_ids: Sequence[str], origin: str) -> None:
        if tile_id in seen:
            return
        seen.add(tile_id)
        if not image_source.is_file():
            raise CompletionError(f"{tile_id}: image missing at {image_source}")
        if sha256_file(image_source) != image_sha:
            raise CompletionError(f"{tile_id}: image bytes do not match the recorded sha")
        target_image = images_out / f"{tile_id}.png"
        write_bytes_atomic(target_image, image_source.read_bytes())
        if sha256_file(target_image) != image_sha:      # §28 immutable pixels
            raise CompletionError(f"{tile_id}: accepted_v2 image hash mismatch")
        lines = label_lines(labels)
        if not lines:
            raise CompletionError(f"{tile_id}: refuses to write an empty positive label")
        for label in labels:
            box = label["tile_xyxy"]
            if not (0 <= box[0] < box[2] <= TILE_SIZE
                    and 0 <= box[1] < box[3] <= TILE_SIZE):
                raise CompletionError(f"{tile_id}: label outside the tile")
            for value in label["yolo_xywh_norm"]:
                if not 0.0 <= value <= 1.0:
                    raise CompletionError(f"{tile_id}: YOLO value out of range")
        target_label = labels_out / f"{tile_id}.txt"
        write_bytes_atomic(target_label, ("\n".join(lines) + "\n").encode("utf-8"))
        rows_out.append({
            "tile_id": tile_id,
            "origin": origin,
            "primary_episode_id": (by_id.get(tile_id) or {}).get("primary_episode_id"),
            "primary_episode_ids": list(primary_ids),
            "camera_id": camera_id,
            "source_file_id": source_file_id,
            "source_crop_xyxy": list(crop),
            "image_path": str(target_image),
            "image_sha256": image_sha,
            "label_path": str(target_label),
            "label_count": len(lines),
            "labels": [dict(label) for label in labels],
            "annotation_complete": True,
            "positive_training_ready": True,
        })

    # 1) the 29 Step 1C-2 accepted tiles, copied byte for byte and never re-reviewed
    for row in data["accepted_rows"]:
        tile_id = str(row["tile_id"])
        candidate = data["by_id"].get(tile_id) or {}
        labels: list[dict[str, Any]] = []
        for index, label in enumerate(row.get("labels") or []):
            box = [round(float(v), 3) for v in label["tile_xyxy"]]
            labels.append({
                "label_id": f"existing-{index}", "origin": "step1c2_verified",
                "episode_ids": list(label.get("episode_ids") or []),
                "supplemental_target_ids": [], "class_id": CLASS_ID,
                "class_name": CLASS_NAME, "tile_xyxy": box,
                "source_xyxy": [round(float(v), 3) for v in label["source_xyxy"]],
                "yolo_xywh_norm": list(label["yolo_xywh_norm"]),
                "source_short_side_px": label.get("source_short_side_px"),
                "size_bucket": label.get("size_bucket")})
        emit(tile_id=tile_id, image_source=Path(str(row["image_path"])),
             image_sha=str(row["image_sha256"]), camera_id=str(row["camera_id"]),
             source_file_id=str(row.get("source_file_id") or candidate.get("source_file_id")),
             crop=candidate.get("source_crop_xyxy") or row.get("source_crop_xyxy"),
             labels=labels, primary_ids=list(row.get("primary_episode_ids") or []),
             origin="step1c2_accepted_frozen")

    # 2) the salvaged tiles
    salvaged = 0
    for row in applied:
        tile_id = str(row["tile_id"])
        if tile_id not in queue_ids:
            continue
        if row["completion_status"] != TILE_STATUS_COMPLETE:
            continue
        labels = row["merged_labels"]
        emit(tile_id=tile_id, image_source=Path(str(row["image_path"])),
             image_sha=str(row["image_sha256"]), camera_id=str(row["camera_id"]),
             source_file_id=str(row["source_file_id"]),
             crop=row["source_crop_xyxy"], labels=labels,
             primary_ids=list(row.get("primary_episode_ids") or []),
             origin="step1c2m_salvaged")
        salvaged += 1

    return {"accepted_v2_tile_count": len(rows_out), "salvaged_tile_count": salvaged,
            "frozen_tile_count": len(data["accepted_rows"]), "rows": rows_out}


# --------------------------------------------------------------------------- #
# SUMMARY / MANIFEST (§41-§43)
# --------------------------------------------------------------------------- #


def build_summary(data: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                  preflight: Mapping[str, Any], *, state: CompletionState | None = None,
                  accepted_v2: Mapping[str, Any] | None = None,
                  proposal_stats: Mapping[str, Any] | None = None,
                  image_immutability: Mapping[str, Any] | None = None) -> dict[str, Any]:
    applied = apply_state(rows, state) if state is not None else list(rows)
    status_counts = {status: 0 for status in TILE_STATUSES}
    per_camera: dict[str, dict[str, int]] = {}
    buckets: dict[str, int] = {}
    label_hist: dict[str, int] = {}
    targets_total = targets_verified = targets_unresolved = targets_truncated = 0
    duplicates_flagged = duplicates_confirmed = 0
    episodes: set[str] = set()
    for row in applied:
        camera = per_camera.setdefault(str(row["camera_id"]), {
            "missing_input": 0, "salvaged": 0, "final_accepted": 0,
            "still_missing": 0, "unresolved": 0, "truncated": 0, "uncertain": 0})
        camera["missing_input"] += 1
        status = str(row["completion_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == TILE_STATUS_COMPLETE:
            camera["salvaged"] += 1
        elif status == TILE_STATUS_STILL_MISSING:
            camera["still_missing"] += 1
        elif status == TILE_STATUS_UNRESOLVED:
            camera["unresolved"] += 1
        elif status == TILE_STATUS_TRUNCATED:
            camera["truncated"] += 1
        elif status == TILE_STATUS_UNCERTAIN:
            camera["uncertain"] += 1
        for target in row["supplemental_targets"]:
            targets_total += 1
            localization = target.get("localization_status")
            if localization == LOCALIZATION_VERIFIED:
                targets_verified += 1
            elif localization == LOCALIZATION_TRUNCATED:
                targets_truncated += 1
            elif localization == LOCALIZATION_UNRESOLVED:
                targets_unresolved += 1
            if target.get("duplicate_of"):
                duplicates_flagged += 1
                if target.get("duplicate_confirmed_independent"):
                    duplicates_confirmed += 1
        if row["completion_status"] == TILE_STATUS_COMPLETE:
            labels = row["merged_labels"]
            label_hist[str(len(labels))] = label_hist.get(str(len(labels)), 0) + 1
            episodes.update(e for label in labels for e in label.get("episode_ids") or [])

    accepted_rows = list((accepted_v2 or {}).get("rows") or [])
    final_camera: dict[str, int] = {}
    final_label_total = 0
    final_labels_hist: dict[str, int] = {}
    for row in accepted_rows:
        final_camera[str(row["camera_id"])] = final_camera.get(str(row["camera_id"]), 0) + 1
        final_label_total += int(row["label_count"])
        key = str(row["label_count"])
        final_labels_hist[key] = final_labels_hist.get(key, 0) + 1
        # §42: the size histogram covers the original labels *and* the supplemental ones
        for label in row.get("labels") or []:
            bucket = label.get("size_bucket") or "unknown"
            buckets[bucket] = buckets.get(bucket, 0) + 1
    for camera, count in final_camera.items():
        per_camera.setdefault(camera, {
            "missing_input": 0, "salvaged": 0, "final_accepted": 0, "still_missing": 0,
            "unresolved": 0, "truncated": 0, "uncertain": 0})["final_accepted"] = count
    # the 29 frozen tiles keep their own (already complete) episode provenance
    for row in data["accepted_rows"]:
        for label in row.get("labels") or []:
            episodes.update(label.get("episode_ids") or [])

    progress = (state.progress(rows) if state is not None else
                {"queue": len(rows), "reviewed": 0, "pending": len(rows), "skipped": 0})
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "tile_size": TILE_SIZE,
        "class_mapping": {str(CLASS_ID): CLASS_NAME},
        "input": {
            "step1c2_root": str(data["root"]),
            "step1c2_complete": preflight["step1c2_annotation_complete"],
            "step1c2_missing_required": preflight["step1c2_missing_required"],
            "step1c2_box_problem": preflight["step1c2_box_problem"],
            "step1c2_total": preflight["step1c2_total"],
            "old_accepted_positive_tile_count":
                preflight["old_accepted_positive_tile_count"],
            "sha256": dict(preflight["hashes"]),
        },
        "completion": {
            "queue": progress["queue"],
            "reviewed": progress["reviewed"],
            "pending": progress["pending"],
            "skipped": progress["skipped"],
            "status_counts": status_counts,
            "notes": [{"tile_id": row["tile_id"], "status": row["completion_status"],
                       "note": row["completion_note"]}
                      for row in applied if row["completion_note"]],
        },
        "supplemental": {
            "supplemental_target_count": targets_total,
            "verified_supplemental_bbox_count": targets_verified,
            "unresolved_supplemental_count": targets_unresolved,
            "truncated_supplemental_count": targets_truncated,
            "duplicate_warning_count": duplicates_flagged,
            "duplicate_confirmed_independent_count": duplicates_confirmed,
        },
        "proposal": dict(proposal_stats or {
            "first_pass_success": 0, "second_pass_success": 0, "failed": 0}),
        "final": {
            "old_accepted_positive_tile_count":
                preflight["old_accepted_positive_tile_count"],
            "salvaged_positive_tile_count": (accepted_v2 or {}).get(
                "salvaged_tile_count", 0),
            "accepted_v2_positive_tile_count":
                (accepted_v2 or {}).get("accepted_v2_tile_count", 0),
            "accepted_v2_label_count": final_label_total,
            "unique_primary_episode_count": len(episodes),
            "tiles_with_1_label": final_labels_hist.get("1", 0),
            "tiles_with_2_labels": final_labels_hist.get("2", 0),
            "tiles_with_3plus_labels": sum(
                count for key, count in final_labels_hist.items()
                if key.isdigit() and int(key) >= 3),
            "label_count_histogram": dict(sorted(final_labels_hist.items())),
        },
        "per_camera": dict(sorted(per_camera.items())),
        "target_size_histogram": dict(sorted(buckets.items())),
        "image_immutability": dict(image_immutability or {}),
        "boundaries": {
            "gold_modified": False,
            "recovery_evidence_modified": False,
            "localization_modified": False,
            "truth_reconciliation_modified": False,
            "step1c2_review_decision_modified": False,
            "step1c2_artifact_modified": False,
            "crop_moved": False,
            "alternative_crop_created": False,
            "image_resized": False,
            "image_reencoded": False,
            "primary_bbox_modified": False,
            "existing_label_modified": False,
            "hard_negatives_generated": False,
            "detector_run": False,
            "segmentation_run": False,
            "training_started": False,
            "sealed_accessed": False,
        },
    }


def build_manifest(data: Mapping[str, Any], summary: Mapping[str, Any], *,
                   code_commit: str, generated_at: str, config: Mapping[str, Any],
                   artifact_root: Path, provenance: Mapping[str, Any]
                   ) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_at": generated_at,
        "code_commit": code_commit,
        "step1c2_summary_sha256": data["summary_sha256"],
        "step1c2_manifest_sha256": data["manifest_sha256"],
        "tile_candidates_sha256": data["tiles_sha256"],
        "review_state_sha256": data["review_state_sha256"],
        "tile_size": TILE_SIZE,
        "image_pixels_immutable": True,
        "crop_moved": False,
        "resize": False,
        "class_mapping": {str(CLASS_ID): CLASS_NAME},
        "artifact_root": str(artifact_root),
        "config": dict(config),
        "counts": {
            "old_accepted_positive_tile_count":
                summary["final"]["old_accepted_positive_tile_count"],
            "completion_input_count": summary["input"]["step1c2_missing_required"],
            "salvaged_positive_tile_count": summary["final"]["salvaged_positive_tile_count"],
            "accepted_v2_positive_tile_count":
                summary["final"]["accepted_v2_positive_tile_count"],
            "accepted_v2_label_count": summary["final"]["accepted_v2_label_count"],
            "supplemental_target_count":
                summary["supplemental"]["supplemental_target_count"],
        },
        "upstream_provenance": dict(provenance),
        "boundaries": dict(summary["boundaries"]),
    }


def write_outputs(output_dir: Path, *, summary: Mapping[str, Any],
                  manifest: Mapping[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "SUMMARY.json"
    atomic_write_json(summary_path, summary)
    payload = dict(manifest)
    payload["summary_sha256"] = sha256_file(summary_path)
    manifest_path = output_dir / "MANIFEST.json"
    atomic_write_json(manifest_path, payload)
    return {"summary": str(summary_path), "manifest": str(manifest_path)}


def proposal_statistics(applied: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """§41: first-pass / second-pass / failed proposal outcomes."""
    first = second = failed = 0
    for row in applied:
        for target in row.get("supplemental_targets") or []:
            localization = target.get("localization_status")
            revision = int(target.get("proposal_revision") or 1)
            if localization == LOCALIZATION_VERIFIED:
                if revision <= 1:
                    first += 1
                else:
                    second += 1
            elif localization == LOCALIZATION_UNRESOLVED:
                failed += 1
    return {"first_pass_success": first, "second_pass_success": second,
            "failed": failed}
