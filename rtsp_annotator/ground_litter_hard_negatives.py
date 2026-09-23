"""Step 1D: hard negative pool from historically human-confirmed NON_LITTER cards.

The positive pool is frozen (44 accepted 640x640 tiles, 96 ground_litter boxes).  This
step builds the matching *negative* pool from objects a human already ruled out on the
training side, and only those:

* the candidate universe is the historical review batches (2,083 NON_LITTER cards, all
  with a human label and a source-frame bbox) plus the 8 targets that Step 1C-1R
  reconciled to NON_LITTER;
* a candidate is only usable when its (camera, timestamp) falls inside a raw PS that is
  already on disk from Step 1C-0, so nothing is downloaded twice;
* the 640x640 source-native crop is anchored on the historical non-litter object, so the
  confusing object stays inside the tile (that is what makes it a *hard* negative);
* a candidate whose crop provably contains a **verified** REQUIRED_LITTER box from the
  same frame is excluded at generation time (deterministic, §31); unverified historical
  or UNCERTAIN geometry only raises a warning for the human;
* every surviving candidate still has to pass a human "no REQUIRED_LITTER in this tile"
  review; nothing is auto-approved, no bbox is ever created, and the negative label is an
  empty file.

The module is stdlib-only; the decoder is injected and the images are files, so the
sampling, geometry, overlap, state and build logic is fully testable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ground_litter_localization_review import (  # reuse, do not duplicate
    ReviewError,
    assert_not_sealed,
    atomic_write_json,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from .ground_litter_positive_tiles import (  # reuse, do not duplicate
    CLASS_ID,
    CLASS_NAME,
    MIN_LABEL_MARGIN_PX,
    TILE_SIZE,
    _delta_ms,
    _seconds,
    _timestamp_plus,
    crop_tile,
    plan_crop,
    png_bytes,
    write_bytes_atomic,
)

SCHEMA_VERSION = "ground_litter_hard_negatives_v1"
GENERATOR_VERSION = "step1d-1.0.0"
REVIEW_SCHEMA_VERSION = "hard_negative_review_v1"

REVIEW_DECISIONS = ("NEGATIVE_OK", "REQUIRED_PRESENT", "UNCERTAIN", "BAD_CROP")
PENDING = "PENDING"
READY = "NEGATIVE_OK"

STATUS_READY = "READY_FOR_REVIEW"
STATUS_KNOWN_REQUIRED = "KNOWN_REQUIRED_OVERLAP_EXCLUDED"
STATUS_TOO_SMALL = "SOURCE_TOO_SMALL"
STATUS_ANCHOR_TOO_BIG = "ANCHOR_EXCEEDS_TILE"
STATUS_MISMATCH = "SOURCE_FRAME_MISMATCH"
STATUS_DECODE_FAILED = "DECODE_FAILED"
CANDIDATE_STATUSES = (STATUS_READY, STATUS_KNOWN_REQUIRED, STATUS_TOO_SMALL,
                      STATUS_ANCHOR_TOO_BIG, STATUS_MISMATCH, STATUS_DECODE_FAILED)

ORIGIN_HISTORICAL = "historical_non_litter"
ORIGIN_RECONCILED = "reconciled_non_litter"

RISK_KNOWN_REQUIRED = "KNOWN_REQUIRED_PRESENT"
RISK_UNCERTAIN = "UNCERTAIN_TRUTH_IN_CROP"
RISK_NEARBY_REQUIRED = "NEARBY_VERIFIED_REQUIRED_OTHER_FRAME"
RISK_IGNORE_SMALL = "IGNORE_SMALL_IN_CROP"
RISK_FLAGS = (RISK_KNOWN_REQUIRED, RISK_UNCERTAIN, RISK_NEARBY_REQUIRED,
              RISK_IGNORE_SMALL)

#: Deterministic coarse hardness label from the historical sampling source.  This is
#: metadata we actually have; no new taxonomy is invented (§20).
HARDNESS_BY_SOURCE_PREFIX = (
    ("semantic", "historical_false_positive"),
    ("texture", "surface_texture"),
    ("temporal", "temporal_candidate"),
    ("random_grid", "random_grid_background"),
)
EASY_HARDNESS = ("random_grid_background",)
#: Conservative duplicate unit (§6): one representative per (camera, PS, minute, 64 px
#: grid cell), and at most `MAX_PER_TEN_MINUTES` per (camera, PS, ten minutes).
DEDUP_GRID_PX = 64
DEDUP_BUCKET_SECONDS = 60
DEDUP_BURST_SECONDS = 600
MAX_PER_BURST = 2
#: §21: easy random-grid background may not dominate the pool.
MAX_EASY_FRACTION = 0.30
#: The historical card timestamps are second-granular, so "the same frame" is decided on
#: that grid; the verified-required exclusion is deliberately conservative.
SAME_FRAME_TOLERANCE_SECONDS = 1.0
NEARBY_WINDOW_SECONDS = 60.0
#: A decoded frame may not drift further than this from the card timestamp.
FRAME_MISMATCH_TOLERANCE_SECONDS = 1.0


class HardNegativeError(RuntimeError):
    """Step 1D could not proceed."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def camera_from_device_code(device_code: str | None) -> str:
    text = str(device_code or "")
    return text[-5:] if len(text) >= 5 else ""


def hardness_from_source(source: str | None) -> str:
    text = str(source or "")
    for prefix, label in HARDNESS_BY_SOURCE_PREFIX:
        if text.startswith(prefix):
            return label
    return "other"


def is_easy(hardness: str) -> bool:
    return hardness in EASY_HARDNESS


def box_intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (float(a[2]) <= float(b[0]) or float(a[0]) >= float(b[2])
                or float(a[3]) <= float(b[1]) or float(a[1]) >= float(b[3]))


def negative_tile_id(camera_id: str, source_file_id: str, decoded_timestamp: str,
                     crop_xyxy: Sequence[float], anchor_identity: str) -> str:
    """Deterministic candidate id (§23): a re-run must produce the same id."""
    material = "|".join([
        "step1d", str(camera_id), str(source_file_id), str(decoded_timestamp),
        ",".join(str(int(v)) for v in crop_xyxy), str(anchor_identity)])
    return "hn-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def _window(source_files: Mapping[str, Mapping[str, Any]], camera_id: str
            ) -> list[Mapping[str, Any]]:
    rows = [row for row in source_files.values()
            if str(row.get("camera_id")) == camera_id]
    rows.sort(key=lambda row: str(row.get("recording_start") or ""))
    return rows


def locate_local_source(source_files: Mapping[str, Mapping[str, Any]], camera_id: str,
                        timestamp: str) -> Mapping[str, Any] | None:
    """The local raw PS whose [recording_start, +duration] window contains the moment."""
    moment = _seconds(timestamp)
    if moment is None:
        return None
    for row in _window(source_files, camera_id):
        start = _seconds(row.get("recording_start"))
        if start is None:
            continue
        duration = float(row.get("duration") or 0.0)
        if start <= moment <= start + duration:
            return row
    return None


# --------------------------------------------------------------------------- #
# input (§4)
# --------------------------------------------------------------------------- #


def load_historical_input(batch_dirs: Sequence[Path | str], *, source_files_path: Path | str,
                          reconciled_overlay_path: Path | str | None = None,
                          localization_path: Path | str | None = None,
                          recovery_evidence_path: Path | str | None = None
                          ) -> dict[str, Any]:
    """Every human-labelled NON_LITTER card, with its local source lineage resolved."""
    paths = [Path(item) for item in batch_dirs]
    sources_path = Path(source_files_path)
    assert_not_sealed(*paths, sources_path, reconciled_overlay_path, localization_path,
                      recovery_evidence_path)
    source_rows = read_jsonl(sources_path)
    source_files = {str(row["source_file_id"]): row for row in source_rows}

    cards: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}
    for root in paths:
        data_path = root / "review-data.json"
        reviews_path = root / "reviews.json"
        if not data_path.is_file() or not reviews_path.is_file():
            raise FileNotFoundError(f"historical review batch missing: {root}")
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
        reviews = reviews.get("reviews", reviews) if isinstance(reviews, dict) else reviews
        items = {str(item.get("review_id")): item for item in payload.get("items") or []}
        counts: dict[str, int] = {}
        for review_id, review in (reviews.items() if isinstance(reviews, dict)
                                  else [(row["review_id"], row) for row in reviews]):
            label = str(review.get("label") or "")
            counts[label] = counts.get(label, 0) + 1
            label_counts[label] = label_counts.get(label, 0) + 1
            if label != "NON_LITTER":
                continue
            item = items.get(str(review_id)) or {}
            camera = str(review_id).split("-")[0]
            if len(camera) != 5 or not camera.isdigit():
                camera = camera_from_device_code(item.get("device_code"))
            bbox = item.get("bbox") or review.get("bbox")
            cards.append({
                "source_review_batch": root.name,
                "source_card_id": str(review_id),
                "camera_id": camera,
                "device_code": item.get("device_code"),
                "timestamp": item.get("timestamp"),
                "frame_id": item.get("frame_id"),
                "historical_label": label,
                "historical_source": item.get("source") or review.get("source"),
                "anchor_source_xyxy": list(bbox) if bbox else None,
                "note": review.get("note") or "",
                "reviewed_at": review.get("reviewed_at"),
                "origin": ORIGIN_HISTORICAL,
                "hardness_source": hardness_from_source(
                    item.get("source") or review.get("source")),
            })
        batches.append({"batch": root.name, "directory": str(root),
                        "review_data_sha256": sha256_file(data_path),
                        "reviews_sha256": sha256_file(reviews_path),
                        "label_counts": dict(sorted(counts.items())),
                        "fingerprint": payload.get("fingerprint")})

    # source B/C: the Step 1C-1R targets a human reconciled to NON_LITTER
    reconciled: list[dict[str, Any]] = []
    if reconciled_overlay_path and localization_path:
        overlay = read_jsonl(Path(reconciled_overlay_path))
        localization = {str(row.get("episode_id")): row
                        for row in read_jsonl(Path(localization_path))}
        evidence = ({str(row.get("episode_id")): row
                     for row in read_jsonl(Path(recovery_evidence_path))}
                    if recovery_evidence_path else {})
        for row in overlay:
            if str(row.get("effective_truth_bucket")) != "NON_LITTER":
                continue
            episode_id = str(row.get("episode_id"))
            loc = localization.get(episode_id) or {}
            ev = evidence.get(episode_id) or {}
            reconciled.append({
                "source_review_batch": "step1c1r_reconciliation",
                "source_card_id": episode_id,
                "camera_id": str(row.get("camera_id") or ""),
                "device_code": None,
                "timestamp": ev.get("decoded_timestamp"),
                "frame_id": None,
                "historical_label": "NON_LITTER",
                "historical_source": f"reconciled:{row.get('origin')}",
                "anchor_source_xyxy": (list(loc["original_bbox"])
                                       if loc.get("original_bbox") else None),
                "note": str(row.get("reason") or ""),
                "reviewed_at": row.get("reviewed_at"),
                "origin": ORIGIN_RECONCILED,
                "hardness_source": "historical_false_positive",
                "source_file_id_hint": ev.get("source_file_id"),
            })
        cards.extend(reconciled)

    for card in cards:
        if card["origin"] == ORIGIN_RECONCILED:
            origin_note = str(card.get("historical_source") or "")
            card["hardness_source"] = ("historical_false_positive"
                                       if "box_wrong" in origin_note
                                       else "reconciled_non_litter")
        else:
            card["hardness_source"] = hardness_from_source(
                card.get("historical_source"))
        source = (source_files.get(str(card.get("source_file_id_hint")))
                  if card.get("source_file_id_hint") else None)
        if source is None:
            source = locate_local_source(source_files, str(card["camera_id"]),
                                         str(card.get("timestamp") or ""))
        card["recoverable"] = source is not None
        card["source_file_id"] = str(source["source_file_id"]) if source else None
        card["local_ps_path"] = str(source["local_ps_path"]) if source else None
        card["recording_start"] = source.get("recording_start") if source else None
        card["source_width"] = int(source.get("source_width") or 0) if source else 0
        card["source_height"] = int(source.get("source_height") or 0) if source else 0
        card["ps_sha256"] = source.get("local_sha256") if source else None
        card["offset_seconds"] = (
            round((_seconds(card["timestamp"]) or 0.0)
                  - (_seconds(source["recording_start"]) or 0.0), 3) if source else None)
        card["anchor_identity"] = str(card["source_card_id"])

    by_camera: dict[str, int] = {}
    recoverable_by_camera: dict[str, int] = {}
    for card in cards:
        if card["recoverable"]:
            recoverable_by_camera[str(card["camera_id"])] = \
                recoverable_by_camera.get(str(card["camera_id"]), 0) + 1
        by_camera[str(card["camera_id"])] = by_camera.get(str(card["camera_id"]), 0) + 1

    return {
        "batches": batches,
        "batch_dirs": [str(path) for path in paths],
        "source_files_path": sources_path,
        "source_files_sha256": sha256_file(sources_path),
        "source_files": source_files,
        "local_ps_count": len(source_files),
        "cards": cards,
        "historical_non_litter_count": len(cards) - len(reconciled),
        "reconciled_non_litter_count": len(reconciled),
        "label_counts": dict(sorted(label_counts.items())),
        "recoverable_count": sum(1 for card in cards if card["recoverable"]),
        "per_camera_total": dict(sorted(by_camera.items())),
        "per_camera_recoverable": dict(sorted(recoverable_by_camera.items())),
        "reconciled_overlay_path": (Path(reconciled_overlay_path)
                                    if reconciled_overlay_path else None),
        "localization_path": Path(localization_path) if localization_path else None,
        "recovery_evidence_path": (Path(recovery_evidence_path)
                                   if recovery_evidence_path else None),
    }


def load_verified_required_boxes(*, tile_candidates_path: Path | str,
                                 completion_manifest_path: Path | str | None = None
                                 ) -> tuple[dict[tuple[str, str], list[dict[str, Any]]],
                                            dict[str, list[dict[str, Any]]]]:
    """Verified REQUIRED_LITTER boxes, indexed by exact frame and by source file.

    Only Step 1C-1-verified boxes (every Step 1C-2 tile label) and Step 1C-2M
    human-selected supplemental boxes are used; unverified historical geometry never
    excludes a candidate (§31).
    """
    by_frame: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_file: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(Path(tile_candidates_path)):
        source_file_id = str(row.get("source_file_id"))
        timestamp = str(row.get("step1c0_decoded_timestamp"))
        for index, label in enumerate(row.get("labels") or []):
            entry = {"box": [float(v) for v in label["source_xyxy"]],
                     "origin": f"step1c2:{row.get('tile_id')}",
                     "tile_id": row.get("tile_id"), "index": index}
            by_frame.setdefault((source_file_id, timestamp), []).append(entry)
            by_file.setdefault(source_file_id, []).append(entry)
    if completion_manifest_path and Path(completion_manifest_path).is_file():
        for row in read_jsonl(Path(completion_manifest_path)):
            source_file_id = str(row.get("source_file_id"))
            timestamp = str(row.get("step1c0_decoded_timestamp"))
            for target in row.get("supplemental_targets") or []:
                if target.get("localization_status") != "VERIFIED_BBOX":
                    continue
                entry = {"box": [float(v) for v in target["verified_source_xyxy"]],
                         "origin": f"step1c2m:{target['supplemental_target_id']}",
                         "tile_id": row.get("tile_id"), "index": None}
                by_frame.setdefault((source_file_id, timestamp), []).append(entry)
                by_file.setdefault(source_file_id, []).append(entry)
    return by_frame, by_file


def load_truth_context(*, overlay_path: Path | str, localization_path: Path | str
                       ) -> dict[str, list[dict[str, Any]]]:
    """Non-REQUIRED / unlocalized truth near a frame (warnings only, never exclusion)."""
    localization = {str(row.get("episode_id")): row
                    for row in read_jsonl(Path(localization_path))}
    context: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(Path(overlay_path)):
        bucket = str(row.get("effective_truth_bucket"))
        if bucket == "REQUIRED_LITTER":
            continue
        episode_id = str(row.get("episode_id"))
        loc = localization.get(episode_id) or {}
        box = loc.get("original_bbox")
        point = None
        point_source = loc.get("original_point_source") or {}
        if point_source.get("ok") and point_source.get("x") is not None:
            point = [float(point_source["x"]), float(point_source["y"])]
        context.setdefault(str(row.get("camera_id")), []).append({
            "episode_id": episode_id, "bucket": bucket,
            "tile_id": loc.get("tile_id"), "box": [float(v) for v in box] if box else None,
            "point": point,
            "timestamp": (loc.get("source_timestamp") or row.get("source_timestamp"))})
    return context


# --------------------------------------------------------------------------- #
# sampling + crop planning (§5/§6/§8/§31)
# --------------------------------------------------------------------------- #


def plan_candidates(data: Mapping[str, Any], *, required_by_frame: Mapping,
                    required_by_file: Mapping, truth_context: Mapping,
                    hard_target: int = 130, easy_target: int = 40,
                    per_camera_min: int = 12, size: int = TILE_SIZE,
                    margin: float = MIN_LABEL_MARGIN_PX,
                    frame_interval_by_file: Mapping[str, float] | None = None
                    ) -> dict[str, Any]:
    """Deterministic sampling, dedup, crop and overlap screening."""
    intervals = dict(frame_interval_by_file or {})
    cards = [card for card in data["cards"]
             if card["recoverable"] and card.get("anchor_source_xyxy")]
    for card in cards:
        card["_easy"] = is_easy(str(card["hardness_source"]))
    hard = [card for card in cards if not card["_easy"]]
    easy = [card for card in cards if card["_easy"]]
    for group in (hard, easy):
        group.sort(key=lambda card: (str(card["camera_id"]),
                                     str(card["timestamp"]),
                                     str(card["source_card_id"])))

    def pick(pool: Sequence[Mapping[str, Any]], target: int) -> list[Mapping[str, Any]]:
        if target <= 0:
            return []
        by_camera: dict[str, list[Mapping[str, Any]]] = {}
        for card in pool:
            by_camera.setdefault(str(card["camera_id"]), []).append(card)
        ordered = sorted(by_camera, key=lambda cam: (-len(by_camera[cam]), cam))
        chosen: list[Mapping[str, Any]] = []
        seen_keys: set[tuple] = set()
        bursts: dict[tuple[str, str, int], int] = {}
        # round robin so every camera is represented before any camera gets a second slot
        while len(chosen) < target and any(by_camera[cam] for cam in ordered):
            for camera in ordered:
                queue = by_camera[camera]
                while queue:
                    card = queue.pop(0)
                    box = card["anchor_source_xyxy"]
                    center = ((float(box[0]) + float(box[2])) / 2.0,
                              (float(box[1]) + float(box[3])) / 2.0)
                    moment = _seconds(card["timestamp"]) or 0.0
                    key = (camera, card["source_file_id"],
                           int(moment // DEDUP_BUCKET_SECONDS),
                           int(center[0] // DEDUP_GRID_PX),
                           int(center[1] // DEDUP_GRID_PX))
                    burst = (camera, str(card["source_file_id"]),
                             int(moment // DEDUP_BURST_SECONDS))
                    if key in seen_keys:
                        continue
                    if bursts.get(burst, 0) >= MAX_PER_BURST:
                        continue
                    seen_keys.add(key)
                    bursts[burst] = bursts.get(burst, 0) + 1
                    chosen.append(card)
                    break
                if len(chosen) >= target:
                    break
        return chosen

    sampled_hard = pick(hard, hard_target)
    # §21: easy background is only a bounded supplement (<= 30 % of the final pool)
    ratio = MAX_EASY_FRACTION / (1.0 - MAX_EASY_FRACTION)
    max_easy = min(easy_target, int(len(sampled_hard) * ratio))
    sampled_easy = pick(easy, max_easy)
    sampled = list(sampled_hard) + list(sampled_easy)

    candidates: list[dict[str, Any]] = []
    excluded_known_required: list[dict[str, Any]] = []
    per_camera: dict[str, int] = {}
    for card in sampled:
        anchor = [float(v) for v in card["anchor_source_xyxy"]]
        width = int(card["source_width"] or 0)
        height = int(card["source_height"] or 0)
        crop_plan = plan_crop(anchor, [], width, height, size=size, margin=margin)
        base = {
            "source_card_id": card["source_card_id"],
            "source_review_batch": card["source_review_batch"],
            "camera_id": card["camera_id"],
            "scene_version": "UNKNOWN_HISTORICAL",
            "source_file_id": card["source_file_id"],
            "source_ps_path": card["local_ps_path"],
            "timestamp": card["timestamp"],
            "recording_start": card["recording_start"],
            "offset_seconds": card["offset_seconds"],
            "source_width": width, "source_height": height,
            "anchor_type": "bbox",
            "anchor_source_xyxy": [round(v, 3) for v in anchor],
            "anchor_source_point": None,
            "origin": card["origin"],
            "historical_label": card["historical_label"],
            "historical_source": card["historical_source"],
            "hardness_source": card["hardness_source"],
            "note": card.get("note") or "",
            "reviewed_at": card.get("reviewed_at"),
            "risk_flags": [],
            "known_required_boxes": [],
            "uncertain_truth_ids": [],
            "ignore_small_ids": [],
            "crop_size": size,
        }
        if not crop_plan["ok"]:
            # plan_crop() calls the anchor the "target"; name the refusal for this step
            status = (STATUS_ANCHOR_TOO_BIG
                      if crop_plan["status"] == "TARGET_EXCEEDS_TILE"
                      else crop_plan["status"])
            base.update({"candidate_generation_status": status,
                         "candidate_generation_detail": crop_plan.get("detail"),
                         "source_crop_xyxy": None, "negative_tile_id": None})
            candidates.append(base)
            continue
        crop = crop_plan["crop_xyxy"]
        base["source_crop_xyxy"] = list(crop)
        base["anchor_tile_xyxy"] = [round(float(anchor[0]) - crop[0], 3),
                                   round(float(anchor[1]) - crop[1], 3),
                                   round(float(anchor[2]) - crop[0], 3),
                                   round(float(anchor[3]) - crop[1], 3)]
        base["anchor_min_margin_px"] = round(min(
            base["anchor_tile_xyxy"][0], base["anchor_tile_xyxy"][1],
            size - base["anchor_tile_xyxy"][2], size - base["anchor_tile_xyxy"][3]), 3)

        # --- §31: only verified, same-frame REQUIRED boxes may exclude ------------- #
        same_frame: list[dict[str, Any]] = []
        for (file_id, timestamp), entries in required_by_frame.items():
            if file_id != card["source_file_id"]:
                continue
            delta = (_seconds(timestamp) or 0.0) - (_seconds(card["timestamp"]) or 0.0)
            if abs(delta) <= SAME_FRAME_TOLERANCE_SECONDS:
                same_frame.extend(entries)
        hits = [entry for entry in same_frame if box_intersects(entry["box"], crop)]
        if hits:
            base["known_required_boxes"] = [entry["origin"] for entry in hits]
            base["risk_flags"].append(RISK_KNOWN_REQUIRED)
            base["candidate_generation_status"] = STATUS_KNOWN_REQUIRED
            base["candidate_generation_detail"] = (
                "a verified REQUIRED_LITTER box from the same frame intersects this crop")
            base["negative_tile_id"] = negative_tile_id(
                str(card["camera_id"]), str(card["source_file_id"]),
                str(card["timestamp"]), crop, str(card["anchor_identity"]))
            excluded_known_required.append(base)
            candidates.append(base)
            continue

        # --- warnings only: nearby verified boxes, UNCERTAIN truth, IGNORE_SMALL --- #
        for entry in required_by_file.get(str(card["source_file_id"]), []):
            if any(entry is same for same in same_frame):
                continue
            if not box_intersects(entry["box"], crop):
                continue
            origin_tile = entry.get("tile_id")
            entry_ts = None
            for (file_id, timestamp), entries in required_by_frame.items():
                if any(item is entry for item in entries):
                    entry_ts = timestamp
                    break
            delta = abs((_seconds(entry_ts or "") or 0.0)
                        - (_seconds(card["timestamp"]) or 0.0))
            if delta <= NEARBY_WINDOW_SECONDS:
                base["risk_flags"].append(RISK_NEARBY_REQUIRED)
                base["known_required_boxes"].append(f"nearby:{origin_tile}")
        for target in truth_context.get(str(card["camera_id"]), []):
            geometry = None
            if target.get("box"):
                geometry = "box"
                if not box_intersects(target["box"], crop):
                    continue
            elif target.get("point"):
                geometry = "point"
                point = target["point"]
                if not (crop[0] <= point[0] <= crop[2] and crop[1] <= point[1] <= crop[3]):
                    continue
            else:
                continue
            if target["bucket"] in ("UNCERTAIN", "IDENTITY_AMBIGUOUS"):
                base["risk_flags"].append(RISK_UNCERTAIN)
                base["uncertain_truth_ids"].append(target["episode_id"])
            elif target["bucket"] == "IGNORE_SMALL":
                base["risk_flags"].append(RISK_IGNORE_SMALL)
                base["ignore_small_ids"].append(target["episode_id"])
        base["risk_flags"] = sorted(set(base["risk_flags"]))
        base["candidate_generation_status"] = STATUS_READY
        base["candidate_generation_detail"] = None
        base["negative_tile_id"] = negative_tile_id(
            str(card["camera_id"]), str(card["source_file_id"]), str(card["timestamp"]),
            crop, str(card["anchor_identity"]))
        per_camera[str(card["camera_id"])] = per_camera.get(str(card["camera_id"]), 0) + 1
        candidates.append(base)

    status_counts = {status: 0 for status in CANDIDATE_STATUSES}
    hardness_counts: dict[str, int] = {}
    origin_counts: dict[str, int] = {}
    for row in candidates:
        status_counts[row["candidate_generation_status"]] = \
            status_counts.get(row["candidate_generation_status"], 0) + 1
        hardness_counts[row["hardness_source"]] = \
            hardness_counts.get(row["hardness_source"], 0) + 1
        origin_counts[row["origin"]] = origin_counts.get(row["origin"], 0) + 1
    return {
        "candidates": candidates,
        "excluded_known_required": excluded_known_required,
        "sampled_count": len(sampled),
        "sampled_hard_count": len(sampled_hard),
        "sampled_easy_count": len(sampled_easy),
        "hard_pool_available": len(hard),
        "easy_pool_available": len(easy),
        "status_counts": status_counts,
        "hardness_counts": dict(sorted(hardness_counts.items())),
        "origin_counts": dict(sorted(origin_counts.items())),
        "per_camera_candidate": dict(sorted(per_camera.items())),
        "unique_source_ps": len({row["source_file_id"] for row in candidates}),
        "unique_source_frames": len({(row["source_file_id"], row["timestamp"])
                                     for row in candidates}),
    }


def verify_preflight(data: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    candidates = plan["candidates"]
    return {
        "historical_non_litter_count": data["historical_non_litter_count"],
        "reconciled_non_litter_count": data["reconciled_non_litter_count"],
        "label_counts": dict(data["label_counts"]),
        "historical_recoverable_count": data["recoverable_count"],
        "local_ps_count": data["local_ps_count"],
        "per_camera_recoverable": dict(data["per_camera_recoverable"]),
        "sampled_candidate_count": plan["sampled_count"],
        "hardness_counts": dict(plan["hardness_counts"]),
        "origin_counts": dict(plan["origin_counts"]),
        "status_counts": dict(plan["status_counts"]),
        "candidate_tile_count": len(candidates),
        "ready_candidate_count": plan["status_counts"].get(STATUS_READY, 0),
        "exact_duplicate_removed": plan.get("exact_duplicate_removed", 0),
        "known_required_overlap_excluded":
            plan["status_counts"].get(STATUS_KNOWN_REQUIRED, 0),
        "per_camera_candidate": dict(plan["per_camera_candidate"]),
        "unique_source_ps": plan["unique_source_ps"],
        "unique_source_frames": plan["unique_source_frames"],
        "hashes": {
            "source_files_sha256": data["source_files_sha256"],
            "batches": [{"batch": batch["batch"],
                         "review_data_sha256": batch["review_data_sha256"],
                         "reviews_sha256": batch["reviews_sha256"]}
                        for batch in data["batches"]],
        },
    }


# --------------------------------------------------------------------------- #
# pixel generation (§8/§24/§25)
# --------------------------------------------------------------------------- #


def generate_candidates(data: Mapping[str, Any], plan: Mapping[str, Any], decoder: Any,
                        images_dir: Path, *, size: int = TILE_SIZE,
                        existing: Sequence[Mapping[str, Any]] = (),
                        input_fingerprint: str = "",
                        existing_fingerprint: str | None = None,
                        on_progress: Any = None) -> dict[str, Any]:
    """Decode each candidate frame once and write the source-native 640x640 PNG."""
    if existing_fingerprint and input_fingerprint and \
            existing_fingerprint != input_fingerprint:
        raise HardNegativeError(
            "existing artifact was generated from different inputs; refusing to "
            "overwrite (use a new --output directory)")
    existing_by_id = {str(row.get("negative_tile_id")): row for row in existing}
    stats = {"decoded_frames": 0, "reused_candidates": 0, "decode_failed": 0,
             "source_frame_mismatch": 0}
    rows: list[dict[str, Any]] = []

    for candidate in plan["candidates"]:
        row = dict(candidate)
        row.setdefault("image_path", None)
        row.setdefault("image_sha256", None)
        row.setdefault("size_bytes", None)
        row.setdefault("decoded_timestamp", None)
        row.setdefault("timestamp_delta_ms", None)
        row.setdefault("frame_identity_confirmed", None)
        row["annotation_review_status"] = (PENDING
                                           if row["candidate_generation_status"] == STATUS_READY
                                           else None)
        row["hard_negative_ready"] = False
        row["review_reason"] = None
        row["review_schema_version"] = REVIEW_SCHEMA_VERSION
        if row["candidate_generation_status"] != STATUS_READY:
            rows.append(row)
            if on_progress:
                on_progress(row)
            continue

        record = existing_by_id.get(str(row["negative_tile_id"]))
        if record and record.get("image_path") and record.get("source_crop_xyxy") == \
                row["source_crop_xyxy"] and record.get("anchor_source_xyxy") == \
                row["anchor_source_xyxy"]:
            path = Path(str(record["image_path"]))
            if path.is_file() and record.get("image_sha256") == sha256_file(path):
                reused = dict(row)
                for key in ("image_path", "image_sha256", "size_bytes",
                            "decoded_timestamp", "timestamp_delta_ms",
                            "frame_identity_confirmed"):
                    reused[key] = record.get(key)
                stats["reused_candidates"] += 1
                rows.append(reused)
                if on_progress:
                    on_progress(reused)
                continue

        result = decoder.decode_frame(Path(str(row["source_ps_path"])),
                                      float(row["offset_seconds"] or 0.0))
        if not result.get("ok"):
            stats["decode_failed"] += 1
            row["candidate_generation_status"] = STATUS_DECODE_FAILED
            row["candidate_generation_detail"] = str(result.get("error") or "decode_failed")
            rows.append(row)
            continue
        stats["decoded_frames"] += 1
        achieved = _timestamp_plus(row["recording_start"],
                                   float(result.get("decoded_offset_seconds") or 0.0))
        row["decoded_timestamp"] = achieved
        row["timestamp_delta_ms"] = _delta_ms(row["timestamp"], achieved)
        tolerance = FRAME_MISMATCH_TOLERANCE_SECONDS * 1000.0
        row["frame_identity_confirmed"] = (
            row["timestamp_delta_ms"] is not None
            and row["timestamp_delta_ms"] <= tolerance)
        if not row["frame_identity_confirmed"]:
            stats["source_frame_mismatch"] += 1
            row["candidate_generation_status"] = STATUS_MISMATCH
            row["candidate_generation_detail"] = (
                f"decoded frame {achieved} differs from the card timestamp "
                f"{row['timestamp']} by {row['timestamp_delta_ms']}ms")
            rows.append(row)
            continue
        try:
            tile = crop_tile(result["frame"], row["source_crop_xyxy"], size=size)
            payload = png_bytes(tile[:, :, ::-1])          # BGR -> RGB
        except Exception as exc:
            row["candidate_generation_status"] = STATUS_DECODE_FAILED
            row["candidate_generation_detail"] = f"crop/encode failed: {exc}"
            rows.append(row)
            continue
        image_path = Path(images_dir) / f"{row['negative_tile_id']}.png"
        write_bytes_atomic(image_path, payload)
        row["image_path"] = str(image_path)
        row["image_sha256"] = hashlib.sha256(payload).hexdigest()
        row["size_bytes"] = len(payload)
        rows.append(row)
        if on_progress:
            on_progress(row)

    deduped, removed = dedup_exact(rows)
    stats["exact_duplicate_removed"] = removed
    stats["pruned_orphan_image_count"] = _prune_orphan_images(images_dir, deduped)
    return {"candidates": deduped, "stats": stats,
            "input_fingerprint": input_fingerprint, "pre_dedup_count": len(rows)}


def dedup_exact(rows: Sequence[Mapping[str, Any]]
                ) -> tuple[list[dict[str, Any]], int]:
    """§26: only exact duplicates (same image bytes *and* same crop) are merged."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    removed = 0
    for row in rows:
        item = dict(row)
        image_sha = item.get("image_sha256")
        crop = item.get("source_crop_xyxy")
        if not image_sha or not crop:
            key = ("unique", str(item.get("negative_tile_id") or ""))
        else:
            key = (str(image_sha),
                   f"{item.get('source_file_id')}|{item.get('decoded_timestamp') or item.get('timestamp')}|"
                   + ",".join(str(int(v)) for v in crop))
        if key in seen:
            removed += 1
            continue
        seen[key] = item
        order.append(key)
    kept = [seen[key] for key in order]
    kept.sort(key=lambda row: (str(row.get("camera_id")),
                               str(row.get("timestamp")),
                               str(row.get("source_card_id"))))
    return kept, removed


def _prune_orphan_images(images_dir: Path, candidates: Sequence[Mapping[str, Any]]) -> int:
    directory = Path(images_dir)
    if not directory.is_dir():
        return 0
    kept = {f"{row['negative_tile_id']}.png" for row in candidates
            if row.get("negative_tile_id")}
    removed = 0
    for path in sorted(directory.glob("hn-*.png")):
        if path.name not in kept:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def candidate_fingerprint(data: Mapping[str, Any], *, size: int = TILE_SIZE,
                          extra_material: Sequence[str] = ()) -> str:
    """Stable fingerprint of every read-only input the candidate set depends on."""
    material = "|".join([GENERATOR_VERSION, str(data["source_files_sha256"]), str(size)]
                        + [f"{batch['batch']}:{batch['review_data_sha256']}:"
                           f"{batch['reviews_sha256']}" for batch in data["batches"]]
                        + [str(item) for item in extra_material])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# review state (§12/§15/§16)
# --------------------------------------------------------------------------- #


class NegativeReviewState:
    """Append-only audit trail + last decision per candidate, atomically persisted."""

    def __init__(self, path: Path, *, candidate_count: int = 0,
                 input_fingerprint: str = "") -> None:
        self.path = Path(path)
        self.candidate_count = candidate_count
        self.input_fingerprint = input_fingerprint
        self.decisions: dict[str, dict[str, Any]] = {}
        self.audit_trail: list[dict[str, Any]] = []

    @classmethod
    def load(cls, path: Path | str, *, candidate_count: int = 0,
             input_fingerprint: str = "") -> "NegativeReviewState":
        target = Path(path)
        state = cls(target, candidate_count=candidate_count,
                    input_fingerprint=input_fingerprint)
        if not target.is_file():
            return state
        payload = json.loads(target.read_text(encoding="utf-8"))
        state.candidate_count = int(payload.get("candidate_count") or candidate_count)
        state.input_fingerprint = str(payload.get("input_fingerprint")
                                      or input_fingerprint)
        state.decisions = dict(payload.get("decisions") or {})
        state.audit_trail = list(payload.get("audit_trail") or [])
        if input_fingerprint and state.input_fingerprint and \
                state.input_fingerprint != input_fingerprint:
            raise ReviewError("review state belongs to a different candidate set")
        return state

    def save(self) -> None:
        atomic_write_json(self.path, {
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "candidate_count": self.candidate_count,
            "input_fingerprint": self.input_fingerprint,
            "decisions": self.decisions,
            "audit_trail": self.audit_trail,
        })

    def get(self, candidate_id: str) -> dict[str, Any] | None:
        return self.decisions.get(candidate_id)

    def decide(self, candidate: Mapping[str, Any], decision: str, *,
               reason: str = "") -> dict[str, Any]:
        from datetime import datetime

        if candidate.get("candidate_generation_status") != STATUS_READY:
            raise ReviewError(
                f"{candidate.get('negative_tile_id')} is not reviewable "
                f"({candidate.get('candidate_generation_status')})")
        if decision not in REVIEW_DECISIONS:
            raise ReviewError(f"unknown negative decision {decision!r}")
        if decision != READY and not reason.strip():
            raise ReviewError(f"{decision} requires a short reason")
        candidate_id = str(candidate["negative_tile_id"])
        previous = self.decisions.get(candidate_id) or {}
        record = {
            "negative_tile_id": candidate_id,
            "review_status": decision,
            "reason": reason.strip(),
            "hard_negative_ready": decision == READY,
            "risk_flags": list(candidate.get("risk_flags") or []),
            "reviewed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "revision": int(previous.get("revision") or 0) + 1,
            "image_sha256": candidate.get("image_sha256"),
        }
        self.decisions[candidate_id] = record
        self._record(candidate_id, f"review:{decision}",
                     {"reason": reason.strip(), "revision": record["revision"]})
        self.save()
        return record

    def reset(self, candidate_id: str) -> None:
        self.decisions.pop(candidate_id, None)
        self._record(candidate_id, "reset", {})
        self.save()

    def skip(self, candidate_id: str) -> None:
        self._record(candidate_id, "skip", {})
        self.save()

    def _record(self, candidate_id: str, action: str, payload: Mapping[str, Any]) -> None:
        from datetime import datetime

        self.audit_trail.append({
            "at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "negative_tile_id": candidate_id, "action": action, "payload": dict(payload)})

    def progress(self, candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        reviewable = [row for row in candidates
                      if row.get("candidate_generation_status") == STATUS_READY]
        reviewed = sum(1 for row in reviewable
                       if (self.decisions.get(str(row["negative_tile_id"])) or {})
                       .get("review_status") in REVIEW_DECISIONS)
        return {"reviewable": len(reviewable), "reviewed": reviewed,
                "pending": len(reviewable) - reviewed,
                "skipped": sum(1 for entry in self.audit_trail
                               if entry["action"] == "skip")}


def apply_review(candidates: Sequence[Mapping[str, Any]], state: NegativeReviewState
                 ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        record = state.get(str(candidate.get("negative_tile_id"))) or {}
        status = record.get("review_status")
        if status:
            row["review_status"] = status
            row["review_reason"] = record.get("reason")
            row["reviewed_at"] = record.get("reviewed_at")
        elif row.get("candidate_generation_status") != STATUS_READY:
            row["review_status"] = None
        else:
            row["review_status"] = PENDING
        row["hard_negative_ready"] = bool(
            row.get("candidate_generation_status") == STATUS_READY
            and row["review_status"] == READY)
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get("camera_id")), str(row.get("timestamp")),
                               str(row.get("source_card_id"))))
    return rows


# --------------------------------------------------------------------------- #
# accepted pool (§29/§30)
# --------------------------------------------------------------------------- #


def check_positive_conflict(candidates: Sequence[Mapping[str, Any]],
                            positive_rows: Sequence[Mapping[str, Any]],
                            positive_tiles_path: Path | str | None = None
                            ) -> dict[str, Any]:
    """§30: a tile may never be both positive and negative."""
    positive_sha = {str(row.get("image_sha256")): str(row.get("tile_id"))
                    for row in positive_rows if row.get("image_sha256")}
    positive_crop: dict[tuple[str, str], str] = {}
    if positive_tiles_path and Path(positive_tiles_path).is_file():
        for row in read_jsonl(Path(positive_tiles_path)):
            crop = row.get("source_crop_xyxy")
            if crop:
                positive_crop[(str(row.get("source_file_id")),
                               ",".join(str(int(v)) for v in crop))] = \
                    str(row.get("tile_id"))
    sha_conflicts: list[dict[str, Any]] = []
    crop_conflicts: list[dict[str, Any]] = []
    for row in candidates:
        if not row.get("hard_negative_ready"):
            continue
        sha = row.get("image_sha256")
        if sha and sha in positive_sha:
            sha_conflicts.append({"negative_tile_id": row["negative_tile_id"],
                                  "positive_tile_id": positive_sha[sha],
                                  "image_sha256": sha})
        crop = row.get("source_crop_xyxy")
        if crop:
            key = (str(row.get("source_file_id")),
                   ",".join(str(int(v)) for v in crop))
            if key in positive_crop:
                crop_conflicts.append({"negative_tile_id": row["negative_tile_id"],
                                       "positive_tile_id": positive_crop[key],
                                       "crop": list(crop)})
    return {"sha_conflicts": sha_conflicts, "crop_conflicts": crop_conflicts,
            "positive_tile_count": len(positive_rows),
            "positive_sha_count": len(positive_sha),
            "sha_conflict_count": len(sha_conflicts),
            "crop_conflict_count": len(crop_conflicts),
            "ok": not sha_conflicts and not crop_conflicts}


def build_accepted(candidates: Sequence[Mapping[str, Any]], accepted_root: Path, *,
                   state: NegativeReviewState,
                   positive_rows: Sequence[Mapping[str, Any]] = (),
                   positive_tiles_path: Path | str | None = None
                   ) -> dict[str, Any]:
    """Copy the reviewed tile bytes verbatim and write an empty YOLO label (§29)."""
    rows = apply_review(candidates, state)
    progress = state.progress(candidates)
    if progress["pending"] or progress["skipped"]:
        raise HardNegativeError(
            f"refusing to build: pending={progress['pending']} "
            f"skipped={progress['skipped']}")
    conflict = check_positive_conflict(rows, positive_rows, positive_tiles_path)
    if not conflict["ok"]:
        raise HardNegativeError(
            f"positive/negative conflict: {conflict['sha_conflicts']} "
            f"{conflict['crop_conflicts']}")

    images_out = accepted_root / "images"
    labels_out = accepted_root / "labels"
    accepted_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("review_status") != READY:
            continue
        if row.get("candidate_generation_status") != STATUS_READY:
            raise HardNegativeError(f"{row['negative_tile_id']} is not reviewable")
        source = Path(str(row["image_path"]))
        if not source.is_file():
            raise HardNegativeError(f"{row['negative_tile_id']}: candidate image missing")
        if sha256_file(source) != row["image_sha256"]:
            raise HardNegativeError(f"{row['negative_tile_id']}: candidate image drifted")
        target_image = images_out / f"{row['negative_tile_id']}.png"
        write_bytes_atomic(target_image, source.read_bytes())
        if sha256_file(target_image) != row["image_sha256"]:
            raise HardNegativeError(f"{row['negative_tile_id']}: accepted image mismatch")
        target_label = labels_out / f"{row['negative_tile_id']}.txt"
        write_bytes_atomic(target_label, b"")                 # 0-byte empty label
        if target_label.stat().st_size != 0:
            raise HardNegativeError(f"{row['negative_tile_id']}: label must be empty")
        accepted_rows.append({
            "negative_tile_id": row["negative_tile_id"],
            "camera_id": row["camera_id"],
            "scene_version": row["scene_version"],
            "source_file_id": row["source_file_id"],
            "timestamp": row["timestamp"],
            "decoded_timestamp": row.get("decoded_timestamp"),
            "source_crop_xyxy": row["source_crop_xyxy"],
            "anchor_source_xyxy": row["anchor_source_xyxy"],
            "anchor_tile_xyxy": row.get("anchor_tile_xyxy"),
            "origin": row["origin"],
            "hardness_source": row["hardness_source"],
            "source_card_id": row["source_card_id"],
            "source_review_batch": row["source_review_batch"],
            "risk_flags": row.get("risk_flags") or [],
            "image_path": str(target_image),
            "image_sha256": row["image_sha256"],
            "label_path": str(target_label),
            "label_bytes": 0,
            "label_line_count": 0,
            "hard_negative_ready": True,
        })
    return {"accepted_tile_count": len(accepted_rows),
            "accepted_label_count": 0,
            "conflict": conflict, "rows": accepted_rows}


# --------------------------------------------------------------------------- #
# SUMMARY / MANIFEST (§40-§42)
# --------------------------------------------------------------------------- #


def build_summary(data: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
                  preflight: Mapping[str, Any], *, state: NegativeReviewState | None = None,
                  accepted: Mapping[str, Any] | None = None,
                  empty_label_check: Mapping[str, Any] | None = None) -> dict[str, Any]:
    rows = apply_review(candidates, state) if state is not None else list(candidates)
    review_counts = {decision: 0 for decision in REVIEW_DECISIONS}
    review_counts[PENDING] = 0
    per_camera: dict[str, dict[str, int]] = {}
    origin_counts: dict[str, int] = {}
    hardness_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    for row in rows:
        camera = per_camera.setdefault(str(row["camera_id"]), {
            "candidate": 0, "reviewable": 0, "accepted": 0, "required_present": 0,
            "uncertain": 0, "bad_crop": 0, "excluded": 0})
        camera["candidate"] += 1
        if row.get("candidate_generation_status") != STATUS_READY:
            camera["excluded"] += 1
        else:
            camera["reviewable"] += 1
            status = row.get("review_status") or PENDING
            review_counts[status] = review_counts.get(status, 0) + 1
            if status == READY:
                camera["accepted"] += 1
            elif status == "REQUIRED_PRESENT":
                camera["required_present"] += 1
            elif status == "UNCERTAIN":
                camera["uncertain"] += 1
            elif status == "BAD_CROP":
                camera["bad_crop"] += 1
        origin_counts[str(row["origin"])] = origin_counts.get(str(row["origin"]), 0) + 1
        hardness_counts[str(row["hardness_source"])] = \
            hardness_counts.get(str(row["hardness_source"]), 0) + 1
        for flag in row.get("risk_flags") or []:
            risk_counts[flag] = risk_counts.get(flag, 0) + 1
    accepted_rows = list((accepted or {}).get("rows") or [])
    accepted_camera: dict[str, int] = {}
    for row in accepted_rows:
        accepted_camera[str(row["camera_id"])] = \
            accepted_camera.get(str(row["camera_id"]), 0) + 1
    progress = (state.progress(candidates) if state is not None else
                {"reviewable": sum(1 for row in rows
                                   if row.get("candidate_generation_status") == STATUS_READY),
                 "reviewed": 0,
                 "pending": sum(1 for row in rows
                                if row.get("candidate_generation_status") == STATUS_READY),
                 "skipped": 0})
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "tile_size": TILE_SIZE,
        "class_mapping": {str(CLASS_ID): CLASS_NAME},
        "negative_label": "empty_file",
        "input": {
            "historical_non_litter_input_count":
                preflight["historical_non_litter_count"],
            "reconciled_non_litter_input_count":
                preflight["reconciled_non_litter_count"],
            "historical_label_counts": dict(preflight["label_counts"]),
            "historical_recoverable_count": preflight["historical_recoverable_count"],
            "sampled_candidate_count": preflight["sampled_candidate_count"],
            "local_ps_count": preflight["local_ps_count"],
            "per_camera_recoverable": dict(preflight["per_camera_recoverable"]),
            "hardness_counts": dict(preflight["hardness_counts"]),
            "origin_counts": dict(preflight["origin_counts"]),
            "sha256": dict(preflight["hashes"]),
        },
        "generate": {
            "candidate_tile_count": preflight["candidate_tile_count"],
            "ready_candidate_count": preflight["ready_candidate_count"],
            "status_counts": dict(preflight["status_counts"]),
            "known_required_overlap_excluded":
                preflight["known_required_overlap_excluded"],
            "exact_duplicate_removed": preflight.get("exact_duplicate_removed", 0),
            "per_camera_candidate": dict(preflight["per_camera_candidate"]),
            "unique_source_ps": preflight["unique_source_ps"],
            "unique_source_frames": preflight["unique_source_frames"],
            "risk_flag_counts": dict(sorted(risk_counts.items())),
        },
        "review": {
            **progress,
            "counts": review_counts,
            "decisions": {decision: review_counts.get(decision, 0)
                          for decision in REVIEW_DECISIONS},
        },
        "accepted": {
            "state": "built" if accepted_rows else "not_built",
            "accepted_hard_negative_tile_count": len(accepted_rows),
            "accepted_label_count": 0,
            "per_camera_accepted": dict(sorted(accepted_camera.items())),
            "positive_conflict_count":
                (accepted or {}).get("conflict", {}).get("sha_conflict_count", 0)
                + (accepted or {}).get("conflict", {}).get("crop_conflict_count", 0),
            "conflict_detail": (accepted or {}).get("conflict"),
            "empty_label_verified_count":
                sum(1 for row in accepted_rows if row.get("label_bytes") == 0),
        },
        "empty_label_loader_check": dict(empty_label_check or {}),
        "per_camera": dict(sorted(per_camera.items())),
        "boundaries": {
            "gold_modified": False,
            "recovery_evidence_modified": False,
            "localization_modified": False,
            "truth_reconciliation_modified": False,
            "positive_pool_modified": False,
            "step1c2_artifacts_modified": False,
            "bbox_created": False,
            "training_label_created": False,
            "crop_moved_from_anchor": False,
            "image_resized": False,
            "image_reencoded": False,
            "augmentation_applied": False,
            "train_val_split_made": False,
            "detector_run": False,
            "segmentation_run": False,
            "training_started": False,
            "sealed_accessed": False,
            "random_unaudited_frame_used": False,
        },
    }


def build_manifest(data: Mapping[str, Any], summary: Mapping[str, Any], *,
                   code_commit: str, generated_at: str, config: Mapping[str, Any],
                   artifact_root: Path, provenance: Mapping[str, Any],
                   positive_pool_manifest_sha256: str = "",
                   positive_pool_summary_sha256: str = ""
                   ) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_at": generated_at,
        "code_commit": code_commit,
        "tile_size": TILE_SIZE,
        "source_native": True,
        "resize": False,
        "class_mapping": {str(CLASS_ID): CLASS_NAME},
        "negative_label": {"format": "empty_txt", "bytes": 0},
        "positive_pool_manifest_sha256": positive_pool_manifest_sha256,
        "positive_pool_summary_sha256": positive_pool_summary_sha256,
        "historical_review_sha256": [
            {"batch": batch["batch"], "review_data_sha256": batch["review_data_sha256"],
             "reviews_sha256": batch["reviews_sha256"]}
            for batch in data["batches"]],
        "source_recovery_sha256": {
            "source_files_sha256": data["source_files_sha256"]},
        "artifact_root": str(artifact_root),
        "config": dict(config),
        "counts": {
            "historical_non_litter_input_count":
                summary["input"]["historical_non_litter_input_count"],
            "sampled_candidate_count": summary["input"]["sampled_candidate_count"],
            "ready_candidate_count": summary["generate"]["ready_candidate_count"],
            "reviewed": summary["review"]["reviewed"],
            "accepted_hard_negative_tile_count":
                summary["accepted"]["accepted_hard_negative_tile_count"],
        },
        "upstream_provenance": dict(provenance),
        "boundaries": dict(summary["boundaries"]),
    }


def write_outputs(output_dir: Path, *, candidates: Sequence[Mapping[str, Any]],
                  summary: Mapping[str, Any], manifest: Mapping[str, Any]
                  ) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "negative_candidates.jsonl"
    write_jsonl(candidate_path, candidates)
    summary_path = output_dir / "SUMMARY.json"
    atomic_write_json(summary_path, summary)
    payload = dict(manifest)
    payload["negative_candidates_sha256"] = sha256_file(candidate_path)
    payload["summary_sha256"] = sha256_file(summary_path)
    manifest_path = output_dir / "MANIFEST.json"
    atomic_write_json(manifest_path, payload)
    return {"negative_candidates": str(candidate_path),
            "summary": str(summary_path), "manifest": str(manifest_path)}
