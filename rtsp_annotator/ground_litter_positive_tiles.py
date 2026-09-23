"""Step 1C-2: source-native 640x640 positive training tiles + annotation completeness.

Input is the Step 1C-1R ``training_episode_manifest.jsonl``: this step does **not**
recompute eligibility.  Only the 109 ``training_eligible`` episodes are processed, one
primary source frame each.

The design is deliberately narrow:

* the only pixels that ever enter a tile come from a fresh decode of the raw PS at the
  Step 1C-0 confirmed ``decoded_timestamp`` — no screenshot, no UI overlay, no resize;
* a tile is a literal 640x640 slice of the source frame, so an object keeps the exact
  number of real pixels it has in the 2560x1440 source;
* every REQUIRED_LITTER that is verifiably in the same decoder frame and inside the
  crop must carry a label, otherwise the tile is not annotation-complete;
* a human must confirm completeness for every tile; the program never auto-approves;
* no bbox is created, moved, clipped or scaled, and nothing upstream is written.

numpy is imported lazily inside the two pixel helpers so that the geometry, eligibility
and review logic stay importable (and testable) without an image stack.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zlib

from .ground_litter_localization_review import (  # reuse, do not duplicate
    LocalizationError,
    ReviewError,
    SealedAssetError,
    assert_not_sealed,
    atomic_write_json,
    read_jsonl,
    sha256_file,
    write_jsonl,
)

SCHEMA_VERSION = "ground_litter_positive_tiles_v1"
GENERATOR_VERSION = "step1c2-1.0.0"
REVIEW_SCHEMA_VERSION = "positive_tile_review_v1"

TILE_SIZE = 640
MIN_LABEL_MARGIN_PX = 32
CLASS_ID = 0
CLASS_NAME = "ground_litter"

#: A candidate that produced a tile image and therefore can be reviewed by a human.
STATUS_READY = "READY_FOR_REVIEW"
#: Generation outcomes that can never become training data (§9/§10/§6/§17).
CANDIDATE_STATUSES = (
    STATUS_READY,
    "SOURCE_TOO_SMALL",
    "TARGET_EXCEEDS_TILE",
    "MULTI_TARGET_EXCEEDS_TILE",
    "SOURCE_FRAME_MISMATCH",
    "KNOWN_UNLOCALIZED_REQUIRED_PRESENT",
    "UNCERTAIN_TRUTH_IN_CROP",
    "DECODE_FAILED",
    "CROP_UNSTABLE",
)
REVIEW_DECISIONS = (
    "ANNOTATION_COMPLETE",
    "MISSING_REQUIRED",
    "BOX_PROBLEM",
    "UNCERTAIN_COMPLETENESS",
)
REVIEW_STATUSES = ("PENDING",) + REVIEW_DECISIONS

PENDING = "PENDING"
COMPLETE = "ANNOTATION_COMPLETE"

REQUIRED_TRUTH = "REQUIRED_LITTER"
REQUIRED_LOCALIZATION = "VERIFIED_BBOX"
REQUIRED_SOURCE = "RECOVERED_SOURCE_NATIVE"

#: §13: two episodes are the same decoder frame when their Step 1C-0 decoded timestamps
#: are within one real frame interval of each other in the same source file.  The
#: grouping is then confirmed by exact equality of the Step 1C-2 decode.
DEFAULT_FRAME_INTERVAL_SECONDS = 0.04       # only used when the stream cannot be probed
SAME_FRAME_CONFIRM_TOLERANCE_SECONDS = 1e-6

LABEL_IOU_DEDUP = 0.95                      # §35, same frame only


class PositiveTileError(RuntimeError):
    """Step 1C-2 could not proceed."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_tile_id(primary_episode_id: str, source_file_id: str,
                      decoded_timestamp: str,
                      crop_xyxy: Sequence[int] | None = None) -> str:
    """Deterministic tile id (§24) — identical inputs must give an identical id."""
    crop_text = (",".join(str(int(v)) for v in crop_xyxy) if crop_xyxy else "nocrop")
    material = "|".join(["step1c2", str(primary_episode_id), str(source_file_id),
                         str(decoded_timestamp), crop_text])
    return "pt-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def normalize_bbox(box: Sequence[float] | None) -> tuple[float, float, float, float] | None:
    if box is None or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(v == v for v in (x1, y1, x2, y2)):        # NaN
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def bbox_contains(outer: Sequence[float], inner: Sequence[float]) -> bool:
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def bbox_union(boxes: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def label_min_margin(crop: Sequence[float], box: Sequence[float]) -> float:
    """Smallest distance from the box to the crop edge (negative = clipped)."""
    return min(box[0] - crop[0], box[1] - crop[1],
               crop[2] - box[2], crop[3] - box[3])


# --------------------------------------------------------------------------- #
# tile geometry (§7-§11)
# --------------------------------------------------------------------------- #


def clamp_crop_origin(value: float, extent: int, size: int) -> int:
    return int(max(0, min(int(extent) - int(size), round(value))))


def _axis_crop_origin(boxes: Sequence[Sequence[float]], lo_index: int, hi_index: int,
                      extent: int, size: int, margin: float,
                      preferred: float) -> int | None:
    """Pick a crop origin that contains every box with >= margin where possible."""
    clamp_hi = extent - size
    if clamp_hi < 0:
        return None
    lower = 0.0
    upper = float(clamp_hi)
    for box in boxes:
        # origin <= box.lo - margin and origin >= box.hi + margin - size
        upper = min(upper, box[lo_index] - margin)
        lower = max(lower, box[hi_index] + margin - size)
    if lower > upper:
        return None
    return int(round(min(max(preferred, lower), upper)))


def plan_crop(primary_box: Sequence[float], other_boxes: Sequence[Sequence[float]],
              source_width: int, source_height: int, *,
              size: int = TILE_SIZE, margin: float = MIN_LABEL_MARGIN_PX
              ) -> dict[str, Any]:
    """Choose the 640x640 source crop for a primary bbox.

    Starts centred on the primary, then keeps every same-frame bbox that the crop
    touches fully inside with the requested margin, expanding to the union of those
    boxes when a single centred crop would clip one of them.  Never resizes anything.
    """
    if source_width < size or source_height < size:
        return {"ok": False, "status": "SOURCE_TOO_SMALL",
                "detail": f"source {source_width}x{source_height} < {size}"}
    primary = normalize_bbox(primary_box)
    if primary is None:
        return {"ok": False, "status": "DECODE_FAILED", "detail": "primary bbox invalid"}
    if primary[2] - primary[0] > size or primary[3] - primary[1] > size:
        return {"ok": False, "status": "TARGET_EXCEEDS_TILE",
                "detail": f"primary bbox {primary[2] - primary[0]:.0f}x"
                          f"{primary[3] - primary[1]:.0f} exceeds {size}"}

    others = [b for b in (normalize_bbox(b) for b in other_boxes) if b is not None]

    def crop_for(boxes: Sequence[Sequence[float]]) -> tuple[int, int] | None:
        preferred_x = (primary[0] + primary[2]) / 2.0 - size / 2.0 \
            if len(boxes) == 1 else (min(b[0] for b in boxes)
                                     + max(b[2] for b in boxes)) / 2.0 - size / 2.0
        preferred_y = (primary[1] + primary[3]) / 2.0 - size / 2.0 \
            if len(boxes) == 1 else (min(b[1] for b in boxes)
                                     + max(b[3] for b in boxes)) / 2.0 - size / 2.0
        ox = _axis_crop_origin(boxes, 0, 2, source_width, size, margin, preferred_x)
        oy = _axis_crop_origin(boxes, 1, 3, source_height, size, margin, preferred_y)
        if ox is None or oy is None:
            # fall back to "must not clip" (margin 0) before giving up
            ox = _axis_crop_origin(boxes, 0, 2, source_width, size, 0.0, preferred_x)
            oy = _axis_crop_origin(boxes, 1, 3, source_height, size, 0.0, preferred_y)
        if ox is None or oy is None:
            return None
        return (ox, oy)

    selected: list[tuple[float, float, float, float]] = [primary]
    crop: tuple[int, int] | None = None
    for _ in range(6):
        origin = crop_for(selected)
        if origin is None:
            status = ("TARGET_EXCEEDS_TILE" if len(selected) == 1
                      else "MULTI_TARGET_EXCEEDS_TILE")
            return {"ok": False, "status": status,
                    "detail": f"{len(selected)} target(s) cannot fit in {size}x{size}"}
        crop = (origin[0], origin[1], origin[0] + size, origin[1] + size)
        touching = [b for b in others if bbox_intersects(b, crop)]
        grown = list(selected)
        for box in touching:
            if not any(abs(box[0] - b[0]) < 1e-9 and abs(box[2] - b[2]) < 1e-9
                       and abs(box[1] - b[1]) < 1e-9 and abs(box[3] - b[3]) < 1e-9
                       for b in grown):
                grown.append(box)
        if len(grown) == len(selected):
            break
        selected = grown
    else:
        return {"ok": False, "status": "CROP_UNSTABLE",
                "detail": "crop did not stabilise while collecting same-frame targets"}

    origin = crop_for(selected)
    if origin is None:
        return {"ok": False, "status": "MULTI_TARGET_EXCEEDS_TILE",
                "detail": f"{len(selected)} target(s) cannot fit in {size}x{size}"}
    crop = (origin[0], origin[1], origin[0] + size, origin[1] + size)

    contained = [b for b in selected if bbox_contains(crop, b)]
    partial = [b for b in selected if bbox_intersects(b, crop)
               and not bbox_contains(crop, b)]
    if partial:
        return {"ok": False, "status": "MULTI_TARGET_EXCEEDS_TILE",
                "detail": f"{len(partial)} same-frame target(s) would be clipped"}
    return {
        "ok": True,
        "status": STATUS_READY,
        "crop_xyxy": list(crop),
        "contained_boxes": contained,
        "min_label_margin_px": (round(min(label_min_margin(crop, b) for b in contained), 3)
                                if contained else 0.0),
    }


def source_to_tile(box: Sequence[float], crop: Sequence[float]) -> list[float]:
    return [float(box[0]) - crop[0], float(box[1]) - crop[1],
            float(box[2]) - crop[0], float(box[3]) - crop[1]]


def tile_to_yolo(box: Sequence[float], size: int = TILE_SIZE) -> list[float]:
    """tile xyxy -> YOLO cx cy w h, all normalised and inside [0, 1] (§11)."""
    width = float(box[2]) - float(box[0])
    height = float(box[3]) - float(box[1])
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    values = [cx / size, cy / size, width / size, height / size]
    return [round(min(1.0, max(0.0, v)), 6) for v in values]


def short_side_px(box: Sequence[float]) -> float:
    return round(min(float(box[2]) - float(box[0]), float(box[3]) - float(box[1])), 3)


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


# --------------------------------------------------------------------------- #
# pixel helpers (lazy numpy; the only place that touches an image)
# --------------------------------------------------------------------------- #


def crop_tile(frame: Any, crop_xyxy: Sequence[int], *, size: int = TILE_SIZE) -> Any:
    """Pure source-pixel slice.  No resize, no interpolation, no letterbox."""
    import numpy as np

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise PositiveTileError(f"unexpected frame shape {array.shape}")
    x1, y1, x2, y2 = (int(v) for v in crop_xyxy)
    if x2 - x1 != size or y2 - y1 != size:
        raise PositiveTileError(f"crop is not {size}x{size}: {crop_xyxy}")
    height, width = array.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise PositiveTileError(f"crop {crop_xyxy} outside frame {width}x{height}")
    tile = array[y1:y2, x1:x2]
    if tile.shape[0] != size or tile.shape[1] != size:
        raise PositiveTileError(f"crop produced {tile.shape} instead of {size}x{size}")
    return np.ascontiguousarray(tile)


def png_bytes(array: Any) -> bytes:
    """Deterministic lossless PNG for an 8-bit HxWx3 uint8 array (§23/§25)."""
    import numpy as np

    data = np.ascontiguousarray(array)
    if data.dtype != np.uint8:
        raise PositiveTileError(f"PNG writer needs uint8, got {data.dtype}")
    if data.ndim != 3 or data.shape[2] != 3:
        raise PositiveTileError(f"PNG writer needs HxWx3, got {data.shape}")
    height, width = int(data.shape[0]), int(data.shape[1])
    raw = bytearray()
    for row in range(height):
        raw.append(0)                       # filter type 0 = None, fixed
        raw += data[row].tobytes()

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)   # 8-bit truecolour
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b""))


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                       prefix=path.name + ".", suffix=".part")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# input (§1/§2)
# --------------------------------------------------------------------------- #


def load_positive_tile_input(
    *,
    training_manifest_path: Path | str,
    truth_overlay_path: Path | str,
    localization_path: Path | str,
    recovery_evidence_path: Path | str,
    source_files_path: Path | str,
    gold_path: Path | str | None = None,
    gold_manifest: Path | str | None = None,
) -> dict[str, Any]:
    """Load the 109 training-eligible episodes plus screening context, read-only."""
    manifest_path = Path(training_manifest_path)
    overlay_path = Path(truth_overlay_path)
    loc_path = Path(localization_path)
    evidence_path = Path(recovery_evidence_path)
    sources_path = Path(source_files_path)
    assert_not_sealed(manifest_path, overlay_path, loc_path, evidence_path, sources_path,
                      gold_path, gold_manifest)
    for path in (manifest_path, overlay_path, loc_path, evidence_path, sources_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"required input missing: {path}")

    manifest_rows = read_jsonl(manifest_path)
    overlay_rows = {str(r.get("episode_id")): r for r in read_jsonl(overlay_path)}
    loc_rows = {str(r.get("episode_id")): r for r in read_jsonl(loc_path)}
    evidence = {str(r.get("episode_id")): r for r in read_jsonl(evidence_path)}
    sources = {str(r.get("source_file_id")): r for r in read_jsonl(sources_path)}

    eligible: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    for row in manifest_rows:
        episode_id = str(row.get("episode_id") or "")
        if not row.get("training_eligible"):
            continue
        truth = overlay_rows.get(episode_id) or {}
        loc = loc_rows.get(episode_id) or {}
        ev = evidence.get(episode_id) or {}
        bbox = normalize_bbox(row.get("verified_bbox") or loc.get("verified_bbox"))
        source_file_id = str(row.get("source_file_id") or "")
        source = sources.get(source_file_id) or {}
        record = {
            "episode_id": episode_id,
            "camera_id": str(row.get("camera_id") or ""),
            "scene_version": str(row.get("scene_version") or ""),
            "origin": str(row.get("origin") or ""),
            "source_file_id": source_file_id,
            "source_ps_path": str(source.get("local_ps_path") or ""),
            "ps_sha256": source.get("local_sha256"),
            "recording_start": source.get("recording_start"),
            "recording_duration_seconds": source.get("duration"),
            "requested_timestamp": (ev.get("requested_timestamp")
                                    or row.get("source_timestamp")),
            "step1c0_decoded_timestamp": ev.get("decoded_timestamp"),
            "step1c0_timestamp_delta_ms": ev.get("timestamp_delta_ms"),
            "source_width": int(row.get("source_width") or source.get("source_width") or 0),
            "source_height": int(row.get("source_height") or source.get("source_height") or 0),
            "verified_bbox": list(bbox) if bbox else None,
            "verification_frame_path": str(row.get("verification_frame_path") or ""),
            "effective_truth_class": row.get("effective_truth_class"),
            "effective_truth_bucket": row.get("effective_truth_bucket"),
            "localization_status": row.get("localization_status"),
            "source_recovery_status": row.get("source_recovery_status"),
            "truth_localization_status": truth.get("localization_status"),
        }
        for field in ("effective_truth_class", "source_recovery_status",
                      "localization_status", "verified_bbox"):
            if field == "verified_bbox":
                bad = record["verified_bbox"] is None
            elif field == "effective_truth_class":
                bad = record[field] != REQUIRED_TRUTH
            else:
                expected = (REQUIRED_SOURCE if field == "source_recovery_status"
                            else REQUIRED_LOCALIZATION)
                bad = record[field] != expected
            if bad:
                problems.append({"episode_id": episode_id, "field": field,
                                 "value": record[field]})
        if not record["source_file_id"] or not record["step1c0_decoded_timestamp"]:
            problems.append({"episode_id": episode_id, "field": "frame_identity",
                             "value": record["source_file_id"]})
        if not record["source_ps_path"] or not Path(record["source_ps_path"]).is_file():
            problems.append({"episode_id": episode_id, "field": "source_ps_missing",
                             "value": record["source_ps_path"]})
        elif not record["recording_start"]:
            problems.append({"episode_id": episode_id, "field": "recording_start_missing",
                             "value": None})
        if record["source_width"] <= 0 or record["source_height"] <= 0:
            problems.append({"episode_id": episode_id, "field": "source_size_unknown",
                             "value": None})
        eligible.append(record)

    # Screening context: every other REQUIRED_LITTER that is not a verified bbox, plus
    # the non-required truth buckets, with whatever location evidence exists.
    screening: list[dict[str, Any]] = []
    for episode_id, truth in overlay_rows.items():
        bucket = str(truth.get("effective_truth_bucket") or "")
        if str(truth.get("effective_truth_class")) == REQUIRED_TRUTH \
                and str(truth.get("localization_status")) == REQUIRED_LOCALIZATION:
            continue
        if bucket == REQUIRED_TRUTH:
            kind = "UNLOCALIZED_REQUIRED"
        elif bucket in ("IGNORE_SMALL", "NON_LITTER", "UNCERTAIN", "IDENTITY_AMBIGUOUS"):
            kind = bucket
        else:
            continue
        ev = evidence.get(episode_id) or {}
        loc = loc_rows.get(episode_id) or {}
        box = normalize_bbox(loc.get("original_bbox"))
        point_source = loc.get("original_point_source") or {}
        point = None
        if point_source.get("ok") and point_source.get("x") is not None:
            point = [float(point_source["x"]), float(point_source["y"])]
        screening.append({
            "episode_id": episode_id,
            "kind": kind,
            "camera_id": str(truth.get("camera_id") or ""),
            "source_file_id": str(ev.get("source_file_id") or ""),
            "step1c0_decoded_timestamp": ev.get("decoded_timestamp"),
            "screening_bbox": list(box) if box else None,
            "screening_point": point,
            "screening_geometry_source": ("historical_bbox" if box
                                          else ("mapped_point" if point else None)),
            "reconciliation_decision": truth.get("reconciliation_decision"),
        })

    return {
        "training_manifest_path": manifest_path,
        "training_manifest_sha256": sha256_file(manifest_path),
        "truth_overlay_path": overlay_path,
        "truth_overlay_sha256": sha256_file(overlay_path),
        "localization_path": loc_path,
        "localization_sha256": sha256_file(loc_path),
        "recovery_evidence_path": evidence_path,
        "recovery_evidence_sha256": sha256_file(evidence_path),
        "source_files_path": sources_path,
        "source_files_sha256": sha256_file(sources_path),
        "gold_path": Path(gold_path) if gold_path else None,
        "gold_sha256": sha256_file(gold_path) if gold_path else None,
        "episodes": eligible,
        "screening": screening,
        "eligibility_problems": problems,
    }


def verify_preflight(data: Mapping[str, Any]) -> dict[str, Any]:
    episodes = data["episodes"]
    per_camera: dict[str, int] = {}
    for episode in episodes:
        per_camera[episode["camera_id"]] = per_camera.get(episode["camera_id"], 0) + 1
    resolutions = sorted({f"{e['source_width']}x{e['source_height']}" for e in episodes})
    return {
        "training_eligible_episode_count": len(episodes),
        "per_camera": dict(sorted(per_camera.items())),
        "unique_source_ps_count": len({e["source_file_id"] for e in episodes}),
        "source_resolutions": resolutions,
        "resolved_source_resolution": resolutions[0] if len(resolutions) == 1 else None,
        "eligibility_problem_count": len(data["eligibility_problems"]),
        "eligibility_problems": list(data["eligibility_problems"]),
        "screening_target_count": len(data["screening"]),
        "screening_by_kind": _counter(t["kind"] for t in data["screening"]),
        "expected_count_ok": len(episodes) == 109,
        "hashes": {
            "gold_sha256": data["gold_sha256"],
            "recovery_evidence_sha256": data["recovery_evidence_sha256"],
            "localization_sha256": data["localization_sha256"],
            "truth_reconciliation_sha256": data["truth_overlay_sha256"],
            "training_episode_manifest_sha256": data["training_manifest_sha256"],
            "source_files_sha256": data["source_files_sha256"],
        },
    }


def _counter(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# same-frame grouping (§13)
# --------------------------------------------------------------------------- #


def cluster_same_frame(episodes: Sequence[Mapping[str, Any]],
                       frame_interval_by_file: Mapping[str, float] | None = None
                       ) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    """Group episodes whose Step 1C-0 decoded timestamps fall in one frame interval.

    The grouping anchors on the first timestamp of each cluster (no chaining), and is
    additionally confirmed against the Step 1C-2 decode before any label is shared.
    """
    intervals = dict(frame_interval_by_file or {})
    by_file: dict[str, list[Mapping[str, Any]]] = {}
    for episode in episodes:
        by_file.setdefault(str(episode["source_file_id"]), []).append(episode)

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for source_file_id, items in by_file.items():
        ordered = sorted(items, key=lambda e: (str(e["step1c0_decoded_timestamp"]),
                                               str(e["episode_id"])))
        interval = float(intervals.get(source_file_id, DEFAULT_FRAME_INTERVAL_SECONDS))
        current: list[Mapping[str, Any]] = []
        anchor = None
        for episode in ordered:
            moment = _seconds(episode["step1c0_decoded_timestamp"])
            if anchor is None or moment is None or (moment - anchor) <= interval:
                current.append(episode)
                if anchor is None and moment is not None:
                    anchor = moment
            else:
                _flush(groups, source_file_id, current)
                current = [episode]
                anchor = moment
        if current:
            _flush(groups, source_file_id, current)
    return groups


def _flush(groups: dict, source_file_id: str, items: Sequence[Mapping[str, Any]]) -> None:
    key = (source_file_id, str(items[0]["step1c0_decoded_timestamp"]))
    groups[key] = list(items)


def _seconds(timestamp: str | None) -> float | None:
    from datetime import datetime

    if not timestamp:
        return None
    text = str(timestamp).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


def confirm_same_frame(primary: Mapping[str, Any],
                       other: Mapping[str, Any]) -> bool:
    """Two episodes share a frame only when the re-decoded frame times are identical."""
    if primary["source_file_id"] != other["source_file_id"]:
        return False
    a = primary.get("step1c2_decoded_timestamp")
    b = other.get("step1c2_decoded_timestamp")
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= SAME_FRAME_CONFIRM_TOLERANCE_SECONDS


# --------------------------------------------------------------------------- #
# labels (§11/§12/§14/§35)
# --------------------------------------------------------------------------- #


def build_label(episode_ids: Sequence[str], source_box: Sequence[float],
                crop: Sequence[float], *, size: int = TILE_SIZE) -> dict[str, Any]:
    tile_box = source_to_tile(source_box, crop)
    return {
        "episode_ids": sorted(str(e) for e in episode_ids),
        "class_name": CLASS_NAME,
        "class_id": CLASS_ID,
        "source_xyxy": [round(float(v), 3) for v in source_box],
        "tile_xyxy": [round(float(v), 3) for v in tile_box],
        "yolo_xywh_norm": tile_to_yolo(tile_box, size),
        "source_short_side_px": short_side_px(source_box),
        "size_bucket": size_bucket(short_side_px(source_box)),
    }


def dedup_labels(labels: Sequence[Mapping[str, Any]], *,
                 iou_threshold: float = LABEL_IOU_DEDUP) -> list[dict[str, Any]]:
    """§35: same-frame boxes with IoU >= threshold collapse to one label + provenance."""
    merged: list[dict[str, Any]] = []
    for label in labels:
        candidate = dict(label)
        for kept in merged:
            if bbox_iou(kept["source_xyxy"], candidate["source_xyxy"]) >= iou_threshold:
                kept["episode_ids"] = sorted(set(kept["episode_ids"])
                                             | set(candidate["episode_ids"]))
                break
        else:
            merged.append(candidate)
    for label in merged:
        label["episode_ids"] = sorted(label["episode_ids"])
    return sorted(merged, key=lambda item: (item["source_xyxy"], item["episode_ids"]))


def label_txt_lines(labels: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = []
    for label in labels:
        cx, cy, width, height = label["yolo_xywh_norm"]
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
    return lines


# --------------------------------------------------------------------------- #
# candidate generation
# --------------------------------------------------------------------------- #


def _screening_index(data: Mapping[str, Any]
                     ) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    index: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for target in data["screening"]:
        key = (str(target["source_file_id"]),
               str(target["step1c0_decoded_timestamp"]))
        index.setdefault(key, []).append(target)
    return index


def screening_hit(target: Mapping[str, Any], crop: Sequence[float]
                  ) -> tuple[bool, bool]:
    """Return (intersects_crop, geometry_available)."""
    box = target.get("screening_bbox")
    if box:
        return bbox_intersects(box, crop), True
    point = target.get("screening_point")
    if point:
        inside = crop[0] <= point[0] <= crop[2] and crop[1] <= point[1] <= crop[3]
        return inside, True
    return False, False


def build_candidate_for_episode(episode: Mapping[str, Any],
                                frame_mates: Sequence[Mapping[str, Any]],
                                screening: Sequence[Mapping[str, Any]],
                                *,
                                size: int = TILE_SIZE,
                                margin: float = MIN_LABEL_MARGIN_PX) -> dict[str, Any]:
    """Geometry-only candidate plan (no pixels): deterministic from metadata."""
    crop_plan = plan_crop(episode["verified_bbox"] or [],
                          [mate["verified_bbox"] for mate in frame_mates
                           if mate["verified_bbox"]],
                          int(episode["source_width"]), int(episode["source_height"]),
                          size=size, margin=margin)
    base = {
        "primary_episode_id": episode["episode_id"],
        "primary_episode_ids": [episode["episode_id"]],
        "camera_id": episode["camera_id"],
        "scene_version": episode["scene_version"],
        "source_file_id": episode["source_file_id"],
        "source_ps_path": episode["source_ps_path"],
        "requested_timestamp": episode["requested_timestamp"],
        "step1c0_decoded_timestamp": episode["step1c0_decoded_timestamp"],
        "source_width": int(episode["source_width"]),
        "source_height": int(episode["source_height"]),
        "primary_source_bbox": [round(float(v), 3) for v in (episode["verified_bbox"] or [])],
        "crop_size": size,
        "labels": [],
        "label_count": 0,
        "min_label_margin_px": None,
        "known_required_same_frame_count": 0,
        "known_unlocalized_required_present": False,
        "known_unlocalized_required_in_crop_ids": [],
        "known_unlocalized_required_same_frame_ids": [],
        "known_unlocalized_required_without_geometry_ids": [],
        "ignore_small_in_crop_ids": [],
        "non_litter_same_frame_ids": [],
        "uncertain_truth_in_crop_ids": [],
        "other_truth_same_frame_ids": [],
        "frame_mate_episode_ids": sorted(m["episode_id"] for m in frame_mates
                                         if m["episode_id"] != episode["episode_id"]),
    }
    if not crop_plan["ok"]:
        base.update({"candidate_generation_status": crop_plan["status"],
                     "candidate_generation_detail": crop_plan.get("detail"),
                     "source_crop_xyxy": None,
                     "tile_id": canonical_tile_id(
                         episode["episode_id"], episode["source_file_id"],
                         str(episode["step1c0_decoded_timestamp"]), None)})
        return base

    crop = crop_plan["crop_xyxy"]
    base["source_crop_xyxy"] = list(crop)
    base["min_label_margin_px"] = crop_plan["min_label_margin_px"]

    contained = crop_plan["contained_boxes"]

    def _key(box: Sequence[float]) -> tuple[float, float, float, float]:
        return (round(float(box[0]), 6), round(float(box[1]), 6),
                round(float(box[2]), 6), round(float(box[3]), 6))

    # Every verified bbox the crop touches is fully inside it: label each distinct
    # geometry once and keep every episode id mapped onto it as provenance (§35).
    by_box: dict[tuple[float, float, float, float], set[str]] = {
        _key(box): set() for box in contained}
    for mate in frame_mates:
        box = mate["verified_bbox"]
        if not box:
            continue
        key = _key(box)
        if key in by_box:
            by_box[key].add(str(mate["episode_id"]))
    by_box.setdefault(_key(episode["verified_bbox"]), set()).add(
        str(episode["episode_id"]))
    labels = [build_label(sorted(ids), list(key), crop, size=size)
              for key, ids in by_box.items() if ids]
    labels = dedup_labels(labels)
    base["labels"] = labels
    base["label_count"] = len(labels)

    status = STATUS_READY
    detail = None
    for target in screening:
        if target["kind"] != "UNLOCALIZED_REQUIRED":
            hit, _available = screening_hit(target, crop)
            if target["kind"] == "IGNORE_SMALL" and hit:
                base["ignore_small_in_crop_ids"].append(target["episode_id"])
            elif target["kind"] == "NON_LITTER":
                base["non_litter_same_frame_ids"].append(target["episode_id"])
            elif target["kind"] in ("UNCERTAIN", "IDENTITY_AMBIGUOUS") and hit:
                base["uncertain_truth_in_crop_ids"].append(target["episode_id"])
            elif target["kind"] in ("UNCERTAIN", "IDENTITY_AMBIGUOUS"):
                base["other_truth_same_frame_ids"].append(target["episode_id"])
            continue
        base["known_required_same_frame_count"] += 1
        base["known_unlocalized_required_same_frame_ids"].append(target["episode_id"])
        hit, available = screening_hit(target, crop)
        if not available:
            base["known_unlocalized_required_without_geometry_ids"].append(
                target["episode_id"])
            base["known_unlocalized_required_present"] = True
        elif hit:
            base["known_unlocalized_required_present"] = True
            base["known_unlocalized_required_in_crop_ids"].append(target["episode_id"])
    for key in ("known_unlocalized_required_in_crop_ids",
                "known_unlocalized_required_same_frame_ids",
                "known_unlocalized_required_without_geometry_ids",
                "ignore_small_in_crop_ids", "non_litter_same_frame_ids",
                "uncertain_truth_in_crop_ids", "other_truth_same_frame_ids"):
        base[key] = sorted(set(base[key]))

    if base["known_unlocalized_required_present"]:
        status = "KNOWN_UNLOCALIZED_REQUIRED_PRESENT"
        detail = ("same frame contains REQUIRED_LITTER without a verified bbox whose "
                  "location touches this crop")
    elif base["uncertain_truth_in_crop_ids"]:
        status = "UNCERTAIN_TRUTH_IN_CROP"
        detail = ("same frame contains UNCERTAIN / IDENTITY_AMBIGUOUS truth inside the "
                  "crop (Step 0B §6.4)")
    elif not base["labels"]:
        status = "DECODE_FAILED"
        detail = "no label could be built for the primary target"
    base["candidate_generation_status"] = status
    base["candidate_generation_detail"] = detail
    base["tile_id"] = canonical_tile_id(episode["episode_id"],
                                        episode["source_file_id"],
                                        str(episode["step1c0_decoded_timestamp"]), crop)
    return base


def fill_review_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate.setdefault("annotation_review_status",
                         PENDING if candidate["candidate_generation_status"] == STATUS_READY
                         else None)
    candidate.setdefault("annotation_review_reason", None)
    candidate.setdefault("annotation_review_note", None)
    candidate.setdefault("annotation_reviewed_at", None)
    candidate["positive_training_ready"] = False
    candidate.setdefault("image_path", None)
    candidate.setdefault("image_sha256", None)
    candidate.setdefault("size_bytes", None)
    candidate.setdefault("annotation_review_schema_version", REVIEW_SCHEMA_VERSION)
    return candidate


def generate_candidates(data: Mapping[str, Any], decoder: Any, images_dir: Path, *,
                        size: int = TILE_SIZE, margin: float = MIN_LABEL_MARGIN_PX,
                        existing: Sequence[Mapping[str, Any]] = (),
                        input_fingerprint: str = "",
                        existing_fingerprint: str | None = None,
                        known_intervals: Mapping[str, float] | None = None,
                        on_progress: Any = None) -> dict[str, Any]:
    """Decode every eligible frame from the raw PS and write the 640x640 tiles.

    Same-frame grouping is three stage and evidence based (§13):

    1. candidates are pre-clustered from the Step 1C-0 decoded timestamps using the
       frame interval measured on the real stream;
    2. one frame is decoded per distinct requested offset;
    3. episodes are grouped by the *achieved* Step 1C-2 frame timestamp, so two episodes
       only ever share labels when their decoder frames are provably identical.

    Idempotent (§46): a candidate whose tile image already exists with the recorded
    SHA256 and whose crop is unchanged is reused without decoding or probing anything.
    """
    if existing_fingerprint and input_fingerprint and \
            existing_fingerprint != input_fingerprint:
        raise PositiveTileError(
            "existing artifact was generated from different inputs; refusing to "
            "overwrite (use a new --output directory)")
    existing_by_id: dict[str, Mapping[str, Any]] = {}
    for row in existing:
        existing_by_id[str(row.get("tile_id"))] = row
        for merged_id in row.get("merged_from_tile_ids") or []:
            existing_by_id[str(merged_id)] = row      # exact duplicate: same bytes
    episodes = list(data["episodes"])
    screening_by_frame = _screening_index(data)
    stats = {"decoded_frames": 0, "reused_candidates": 0, "decode_failed": 0,
             "source_frame_mismatch": 0, "probe_failed": 0}

    # Frame intervals already measured for these PS files are reused verbatim: the
    # grouping must not depend on whether a re-run happened to probe a file (§46).
    intervals: dict[str, float] = {
        str(key): float(value) for key, value in (known_intervals or {}).items()}

    def plan_for(member: Mapping[str, Any], frame_mates: Sequence[Mapping[str, Any]]
                 ) -> dict[str, Any]:
        frame_key = (str(member["source_file_id"]),
                     str(member["step1c0_decoded_timestamp"]))
        plan = build_candidate_for_episode(
            member, frame_mates, screening_by_frame.get(frame_key, []),
            size=size, margin=margin)
        plan["frame_interval_ms"] = round(
            intervals.get(str(member["source_file_id"]), DEFAULT_FRAME_INTERVAL_SECONDS)
            * 1000.0, 3)
        plan["step1c2_decoded_timestamp"] = None
        plan["timestamp_delta_ms"] = None
        plan["frame_identity_confirmed"] = None
        return plan

    def is_reusable(plan: Mapping[str, Any]) -> bool:
        """Reuse only when the recorded row still describes exactly this plan.

        Comparing the crop alone is not enough: a changed same-frame group changes the
        labels (and the primary set) while the crop can stay identical.
        """
        record = existing_by_id.get(str(plan.get("tile_id")))
        if not record or not record.get("image_path") or not plan.get("source_crop_xyxy"):
            return False
        # The crop fixes the pixels and the labels fix the YOLO lines.
        # primary_episode_ids is merged provenance that §34 dedup may have widened, so it
        # is deliberately not compared.  A recorded label set that is a strict superset of
        # the plan is accepted: it can only have grown by merging candidates whose tile
        # bytes were identical, i.e. the very same frame and crop.
        if record.get("source_crop_xyxy") != plan.get("source_crop_xyxy"):
            return False
        record_labels = list(record.get("labels") or [])
        plan_labels = list(plan.get("labels") or [])
        if plan_labels != record_labels:
            recorded = {(tuple(label["source_xyxy"]), tuple(label["episode_ids"]))
                        for label in record_labels}
            if not all((tuple(label["source_xyxy"]), tuple(label["episode_ids"]))
                       in recorded for label in plan_labels):
                return False
        path = Path(str(record["image_path"]))
        return path.is_file() and record.get("image_sha256") == sha256_file(path)

    def build_plans(grouping) -> list[dict[str, Any]]:
        built: list[dict[str, Any]] = []
        for key in sorted(grouping, key=lambda k: (k[0], k[1])):
            members = grouping[key]
            entries = [{"member": member, "plan": plan_for(member, members)}
                       for member in members]
            for entry in entries:
                entry["reusable"] = is_reusable(entry["plan"])
                # Every candidate that has a crop needs its frame identity confirmed from
                # the real decode, including the ones a later rule will exclude: their
                # status depends on the crop, and the crop depends on the same-frame
                # group, so guessing it from metadata alone would be circular.
                entry["needs_pixels"] = entry["plan"].get("source_crop_xyxy") is not None
            built.append({"key": key, "members": members, "entries": entries})
        return built

    groups = cluster_same_frame(episodes, intervals)
    plans = build_plans(groups)
    # §46: only files that still need pixels are probed, so a fully reused re-run
    # decodes and probes nothing at all.
    pending_files = sorted({str(entry["member"]["source_file_id"])
                            for plan_entry in plans for entry in plan_entry["entries"]
                            if entry["needs_pixels"] and not entry["reusable"]})
    if pending_files:
        for source_file_id in pending_files:
            if source_file_id in intervals:
                continue
            path = next(e["source_ps_path"] for e in episodes
                        if e["source_file_id"] == source_file_id)
            probe = decoder.probe_interval(Path(path))
            if probe.get("ok") and probe.get("frame_interval_seconds"):
                intervals[source_file_id] = float(probe["frame_interval_seconds"])
            else:
                stats["probe_failed"] += 1
        groups = cluster_same_frame(episodes, intervals)
        plans = build_plans(groups)

    # ---- stage 2: decode one frame per distinct requested offset ---------- #
    decoded: dict[tuple[str, float], dict[str, Any]] = {}
    for plan_entry in plans:
        for entry in plan_entry["entries"]:
            if entry["reusable"] or not entry["needs_pixels"]:
                continue
            member = entry["member"]
            offset = _offset_seconds(member)
            if offset is None:
                continue
            token = (str(member["source_file_id"]), round(offset, 3))
            if token in decoded:
                continue
            result = decoder.decode_frame(Path(member["source_ps_path"]), token[1])
            if result.get("ok"):
                stats["decoded_frames"] += 1
            else:
                stats["decode_failed"] += 1
            decoded[token] = result

    def achieved_timestamp(member: Mapping[str, Any]) -> str | None:
        offset = _offset_seconds(member)
        if offset is None:
            return None
        token = (str(member["source_file_id"]), round(offset, 3))
        result = decoded.get(token)
        if result is None:
            record = existing_by_id.get(
                str(plan_for(member, [member]).get("tile_id")))
            return (record or {}).get("step1c2_decoded_timestamp")
        if not result.get("ok"):
            return None
        return _timestamp_plus(member.get("recording_start"),
                               float(result.get("decoded_offset_seconds") or 0.0))

    # ---- stage 3: regroup by the achieved frame, then emit ---------------- #
    def confirmed_groups(plan_entry):
        confirmed: dict[str, list[Mapping[str, Any]]] = {}
        for member in plan_entry["members"]:
            moment = achieved_timestamp(member)
            confirmed.setdefault(str(moment), []).append(member)
        return [(moment, confirmed[moment]) for moment in sorted(confirmed)]

    def decode_token(member: Mapping[str, Any]):
        offset = _offset_seconds(member)
        return (str(member["source_file_id"]),
                round(offset, 3) if offset is not None else 0.0)

    # The achieved frame can split a provisional group, which changes the crop and the
    # labels.  Those members may not have been decoded in stage 2, so collect them and
    # decode them now (bounded: only groups that actually split).
    extra: set[tuple[str, float]] = set()
    for plan_entry in plans:
        for _moment, members in confirmed_groups(plan_entry):
            for member in members:
                plan = plan_for(member, members)
                if plan.get("source_crop_xyxy") and not is_reusable(plan):
                    token = decode_token(member)
                    if token not in decoded:
                        extra.add(token)
    for token in sorted(extra):
        member = next((e for plan_entry in plans for e in plan_entry["members"]
                       if decode_token(e) == token), None)
        if member is None:                                  # pragma: no cover
            continue
        result = decoder.decode_frame(Path(member["source_ps_path"]), token[1])
        if result.get("ok"):
            stats["decoded_frames"] += 1
        else:
            stats["decode_failed"] += 1
        decoded[token] = result
    stats["extra_decode_round_count"] = len(extra)

    rows: list[dict[str, Any]] = []
    emitted: dict[str, dict[str, Any]] = {}
    for plan_entry in plans:
        for moment, members in confirmed_groups(plan_entry):
            for member in members:
                plan = plan_for(member, members)
                record = existing_by_id.get(str(plan.get("tile_id")))
                if record and record.get("image_path") and is_reusable(plan):
                    tile_id = str(record["tile_id"])
                    stats["reused_candidates"] += 1
                    if tile_id in emitted:
                        # the same merged tile reuses one record for several members;
                        # keep the merge provenance so the index is run-stable
                        keeper = emitted[tile_id]
                        keeper["primary_episode_ids"] = sorted(
                            set(keeper["primary_episode_ids"])
                            | set(plan["primary_episode_ids"]))
                        own_id = str(plan.get("tile_id") or "")
                        if own_id and own_id != tile_id:
                            # never list the kept tile as merged into itself
                            keeper["merged_from_tile_ids"] = sorted(
                                set(keeper.get("merged_from_tile_ids") or []) | {own_id})
                        continue
                    row = fill_review_fields(dict(record))
                    emitted[tile_id] = row
                    rows.append(row)
                    if on_progress:
                        on_progress(row)
                    continue
                if not plan.get("source_crop_xyxy"):
                    rows.append(fill_review_fields(plan))
                    continue
                if plan["candidate_generation_status"] != STATUS_READY:
                    # decided from metadata alone (SOURCE_TOO_SMALL, §17, §6.4, ...)
                    rows.append(fill_review_fields(plan))
                    continue
                result = decoded.get(decode_token(member))
                plan["step1c2_decoded_timestamp"] = moment if moment != "None" else None
                plan["timestamp_delta_ms"] = _delta_ms(
                    member.get("step1c0_decoded_timestamp"), plan["step1c2_decoded_timestamp"])
                tolerance_ms = max(plan["frame_interval_ms"],
                                   DEFAULT_FRAME_INTERVAL_SECONDS * 1000.0)
                plan["frame_identity_confirmed"] = (
                    plan["timestamp_delta_ms"] is not None
                    and plan["timestamp_delta_ms"] <= tolerance_ms)
                if not result or not result.get("ok"):
                    plan["candidate_generation_status"] = "DECODE_FAILED"
                    plan["candidate_generation_detail"] = str(
                        (result or {}).get("error") or "frame not decoded")
                    rows.append(fill_review_fields(plan))
                    continue
                if not plan["frame_identity_confirmed"]:
                    stats["source_frame_mismatch"] += 1
                    plan["candidate_generation_status"] = "SOURCE_FRAME_MISMATCH"
                    plan["candidate_generation_detail"] = (
                        f"re-decoded frame {plan['step1c2_decoded_timestamp']} differs "
                        f"from the Step 1C-0 frame "
                        f"{member.get('step1c0_decoded_timestamp')} by "
                        f"{plan['timestamp_delta_ms']}ms (tolerance {tolerance_ms}ms)")
                    rows.append(fill_review_fields(plan))
                    continue
                try:
                    tile = crop_tile(result["frame"], plan["source_crop_xyxy"], size=size)
                    payload = png_bytes(tile[:, :, ::-1])          # BGR -> RGB
                except Exception as exc:
                    plan["candidate_generation_status"] = "DECODE_FAILED"
                    plan["candidate_generation_detail"] = f"crop/encode failed: {exc}"
                    rows.append(fill_review_fields(plan))
                    continue
                image_path = Path(images_dir) / f"{plan['tile_id']}.png"
                write_bytes_atomic(image_path, payload)
                plan["image_path"] = str(image_path)
                plan["image_sha256"] = sha256_bytes(payload)
                plan["size_bytes"] = len(payload)
                rows.append(fill_review_fields(plan))
                if on_progress:
                    on_progress(plan)

    deduped, merges = dedup_candidates(rows)
    stats["pruned_orphan_image_count"] = _prune_orphan_images(images_dir, deduped)
    stats["merged_superseded_count"] = max(0, len(episodes) - len(deduped))
    return {"candidates": deduped, "merges": merges, "stats": stats,
            "intervals": intervals, "input_fingerprint": input_fingerprint,
            "pre_dedup_count": len(rows)}


def _prune_orphan_images(images_dir: Path, candidates: Sequence[Mapping[str, Any]]) -> int:
    """§34 merging can leave an identical PNG behind under the merged tile id.

    The bytes are preserved under the kept tile id, so the redundant copy is removed to
    keep the artifact exactly equal to the candidate index.
    """
    directory = Path(images_dir)
    if not directory.is_dir():
        return 0
    kept = {f"{row['tile_id']}.png" for row in candidates if row.get("tile_id")}
    removed = 0
    for path in sorted(directory.glob("pt-*.png")):
        if path.name not in kept:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def _offset_seconds(episode: Mapping[str, Any]) -> float | None:
    start = _seconds(episode.get("recording_start"))
    moment = _seconds(episode.get("step1c0_decoded_timestamp"))
    if start is None or moment is None:
        return None
    return moment - start


def _timestamp_plus(recording_start: str | None, offset_seconds: float) -> str | None:
    """Millisecond precision: the Step 1C-0 frame is only known to the second."""
    from datetime import datetime, timedelta

    start = _seconds(recording_start)
    if start is None:
        return None
    moment = datetime.fromtimestamp(start) + timedelta(seconds=float(offset_seconds))
    return moment.strftime("%Y-%m-%d %H:%M:%S.") + f"{moment.microsecond // 1000:03d}"


def _delta_ms(reference: str | None, achieved: str | None) -> float | None:
    a = _seconds(reference)
    b = _seconds(achieved)
    if a is None or b is None:
        return None
    return round(abs(b - a) * 1000.0, 3)


def candidate_fingerprint(data: Mapping[str, Any], *, size: int = TILE_SIZE) -> str:
    """Stable fingerprint of the read-only inputs a candidate set depends on."""
    material = "|".join([
        str(data["training_manifest_sha256"]), str(data["truth_overlay_sha256"]),
        str(data["localization_sha256"]), str(data["recovery_evidence_sha256"]),
        str(data["source_files_sha256"]), str(size), GENERATOR_VERSION])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def dedup_candidates(candidates: Sequence[Mapping[str, Any]]
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """§34: merge only exact duplicates (same image SHA256 *and* same crop identity)."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    merges: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda c: (str(c.get("tile_id") or ""),
                                                       str(c["primary_episode_id"]))):
        image_sha = candidate.get("image_sha256")
        crop = candidate.get("source_crop_xyxy")
        identity = (str(candidate["source_file_id"]),
                    str(candidate.get("step1c2_decoded_timestamp")
                        or candidate["step1c0_decoded_timestamp"]),
                    ",".join(str(int(v)) for v in crop) if crop else "none")
        if not image_sha or not crop:
            unique_key = ("unique", str(candidate.get("tile_id") or ""))
            merged[unique_key] = dict(candidate)
            order.append(unique_key)
            continue
        key = (image_sha, ",".join([identity[0], identity[1], identity[2]]))
        if key not in merged:
            record = dict(candidate)
            # a reused row already carries the merge provenance of the previous run
            record.setdefault("merged_from_tile_ids", [])
            merged[key] = record
            order.append(key)
            continue
        keeper = merged[key]
        keeper["primary_episode_ids"] = sorted(set(keeper["primary_episode_ids"])
                                               | set(candidate["primary_episode_ids"]))
        keeper["merged_from_tile_ids"] = sorted(set(keeper["merged_from_tile_ids"])
                                                | {str(candidate.get("tile_id") or "")})
        merged_labels = dedup_labels(list(keeper["labels"]) + list(candidate["labels"]))
        keeper["labels"] = merged_labels
        keeper["label_count"] = len(merged_labels)
        keeper["all_known_label_episode_ids"] = sorted(
            {e for label in merged_labels for e in label["episode_ids"]})
        for field in ("known_unlocalized_required_in_crop_ids",
                      "known_unlocalized_required_same_frame_ids",
                      "ignore_small_in_crop_ids", "non_litter_same_frame_ids",
                      "uncertain_truth_in_crop_ids", "other_truth_same_frame_ids",
                      "frame_mate_episode_ids"):
            keeper[field] = sorted(set(keeper.get(field) or [])
                                   | set(candidate.get(field) or []))
        merges.append({"kept_tile_id": keeper["tile_id"],
                       "merged_tile_id": candidate.get("tile_id"),
                       "primary_episode_id": candidate["primary_episode_id"]})
    rows = []
    for key in order:
        record = merged[key]
        # invariant: a tile is never listed as merged into itself, even if an older
        # artifact recorded it that way
        record["merged_from_tile_ids"] = sorted(
            {str(value) for value in (record.get("merged_from_tile_ids") or [])
             if value and str(value) != str(record["tile_id"])})
        record["all_known_label_episode_ids"] = sorted(
            {e for label in record["labels"] for e in label["episode_ids"]})
        rows.append(record)
    rows.sort(key=lambda row: (row["camera_id"], row["primary_episode_id"]))
    return rows, merges


# --------------------------------------------------------------------------- #
# review state (§28/§29)
# --------------------------------------------------------------------------- #


class TileReviewState:
    """Append-only audit trail + last decision per tile, atomically persisted."""

    def __init__(self, path: Path, *, candidate_count: int = 0,
                 input_fingerprint: str = "") -> None:
        self.path = Path(path)
        self.candidate_count = candidate_count
        self.input_fingerprint = input_fingerprint
        self.decisions: dict[str, dict[str, Any]] = {}
        self.audit_trail: list[dict[str, Any]] = []

    @classmethod
    def load(cls, path: Path | str, *, candidate_count: int = 0,
             input_fingerprint: str = "") -> "TileReviewState":
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

    def get(self, tile_id: str) -> dict[str, Any] | None:
        return self.decisions.get(tile_id)

    def decide(self, candidate: Mapping[str, Any], decision: str, *,
               note: str = "") -> dict[str, Any]:
        from datetime import datetime

        if candidate.get("candidate_generation_status") != STATUS_READY:
            raise ReviewError(
                f"{candidate.get('tile_id')} is not reviewable "
                f"({candidate.get('candidate_generation_status')})")
        if decision not in REVIEW_DECISIONS:
            raise ReviewError(f"unknown review decision {decision!r}")
        tile_id = str(candidate["tile_id"])
        previous = self.decisions.get(tile_id) or {}
        record = {
            "tile_id": tile_id,
            "primary_episode_id": candidate["primary_episode_id"],
            "annotation_review_status": decision,
            "annotation_review_note": note.strip(),
            "annotation_review_reason": decision,
            "positive_training_ready": decision == COMPLETE,
            "reviewed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "revision": int(previous.get("revision") or 0) + 1,
            "label_count": candidate.get("label_count"),
            "image_sha256": candidate.get("image_sha256"),
        }
        self.decisions[tile_id] = record
        self._record(tile_id, f"review:{decision}",
                     {"note": note.strip(), "revision": record["revision"]})
        self.save()
        return record

    def reset(self, tile_id: str) -> None:
        self.decisions.pop(tile_id, None)
        self._record(tile_id, "reset", {})
        self.save()

    def skip(self, tile_id: str) -> None:
        self._record(tile_id, "skip", {})
        self.save()

    def _record(self, tile_id: str, action: str, payload: Mapping[str, Any]) -> None:
        from datetime import datetime

        self.audit_trail.append({
            "at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tile_id": tile_id, "action": action, "payload": dict(payload)})

    def progress(self, candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        reviewable = [c for c in candidates
                      if c.get("candidate_generation_status") == STATUS_READY]
        reviewed = sum(1 for c in reviewable
                       if (self.decisions.get(str(c["tile_id"])) or {})
                       .get("annotation_review_status") in REVIEW_DECISIONS)
        return {"reviewable": len(reviewable), "reviewed": reviewed,
                "pending": len(reviewable) - reviewed,
                "skipped": sum(1 for entry in self.audit_trail
                               if entry["action"] == "skip")}


def apply_review(candidates: Sequence[Mapping[str, Any]],
                 state: TileReviewState, *, reviewable_only: bool = False
                 ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        record = state.get(str(candidate.get("tile_id"))) or {}
        status = record.get("annotation_review_status")
        if status:
            row["annotation_review_status"] = status
            row["annotation_review_reason"] = record.get("annotation_review_reason")
            row["annotation_review_note"] = record.get("annotation_review_note")
            row["annotation_reviewed_at"] = record.get("reviewed_at")
        elif row.get("candidate_generation_status") != STATUS_READY:
            row["annotation_review_status"] = None
        else:
            row["annotation_review_status"] = PENDING
        row["positive_training_ready"] = bool(
            row.get("candidate_generation_status") == STATUS_READY
            and row["annotation_review_status"] == COMPLETE)
        if not reviewable_only or row.get("candidate_generation_status") == STATUS_READY:
            rows.append(row)
    rows.sort(key=lambda row: (row["camera_id"], row["primary_episode_id"]))
    return rows


# --------------------------------------------------------------------------- #
# accepted dataset (§30-§33)
# --------------------------------------------------------------------------- #


def build_accepted(candidates: Sequence[Mapping[str, Any]],
                   images_dir: Path, accepted_root: Path, *,
                   state: TileReviewState | None = None) -> dict[str, Any]:
    """Copy the reviewed tile bytes verbatim and emit YOLO labels.

    §31: the accepted image must be byte-identical to the reviewed candidate image.
    """
    rows = apply_review(candidates, state) if state is not None else list(candidates)
    reviewable = [r for r in rows
                  if r.get("candidate_generation_status") == STATUS_READY]
    pending = [r for r in reviewable if r.get("annotation_review_status") == PENDING]
    skipped = (state.progress(candidates)["skipped"] if state is not None else 0)
    if pending or skipped:
        raise PositiveTileError(
            f"refusing to build: pending={len(pending)} skipped={skipped}")

    images_out = accepted_root / "images"
    labels_out = accepted_root / "labels"
    manifest_rows: list[dict[str, Any]] = []
    accepted = 0
    label_total = 0
    for row in rows:
        if row.get("annotation_review_status") != COMPLETE:
            continue
        if row.get("candidate_generation_status") != STATUS_READY:
            raise PositiveTileError(f"{row['tile_id']} is complete but not reviewable")
        if not row.get("positive_training_ready"):
            raise PositiveTileError(f"{row['tile_id']} complete but not training-ready")
        labels = list(row.get("labels") or [])
        if not labels:
            raise PositiveTileError(f"{row['tile_id']} has no labels (§33)")
        label_episode_ids = {e for label in labels for e in label["episode_ids"]}
        if row["primary_episode_id"] not in label_episode_ids:
            raise PositiveTileError(f"{row['tile_id']} lost its primary bbox (§32)")
        for label in labels:
            tile_box = label["tile_xyxy"]
            if not (0 <= tile_box[0] < tile_box[2] <= TILE_SIZE
                    and 0 <= tile_box[1] < tile_box[3] <= TILE_SIZE):
                raise PositiveTileError(f"{row['tile_id']} label outside the tile")
            for value in label["yolo_xywh_norm"]:
                if not 0.0 <= value <= 1.0:
                    raise PositiveTileError(f"{row['tile_id']} yolo value out of range")
        source = Path(row["image_path"])
        if not source.is_file():
            raise PositiveTileError(f"{row['tile_id']} candidate image missing")
        if sha256_file(source) != row["image_sha256"]:
            raise PositiveTileError(f"{row['tile_id']} candidate image hash drifted")
        target_image = images_out / f"{row['tile_id']}.png"
        write_bytes_atomic(target_image, source.read_bytes())
        if sha256_file(target_image) != row["image_sha256"]:
            raise PositiveTileError(f"{row['tile_id']} accepted image hash mismatch (§31)")
        lines = label_txt_lines(labels)
        target_label = labels_out / f"{row['tile_id']}.txt"
        write_bytes_atomic(target_label, ("\n".join(lines) + "\n").encode("utf-8"))
        accepted += 1
        label_total += len(lines)
        manifest_rows.append({
            "tile_id": row["tile_id"],
            "primary_episode_id": row["primary_episode_id"],
            "primary_episode_ids": row["primary_episode_ids"],
            "camera_id": row["camera_id"],
            "scene_version": row["scene_version"],
            "source_file_id": row["source_file_id"],
            "step1c2_decoded_timestamp": row.get("step1c2_decoded_timestamp"),
            "source_crop_xyxy": row["source_crop_xyxy"],
            "source_width": row["source_width"],
            "source_height": row["source_height"],
            "image_path": str(target_image),
            "image_sha256": row["image_sha256"],
            "label_path": str(target_label),
            "label_count": len(lines),
            "labels": [{"episode_ids": label["episode_ids"],
                        "class_id": label["class_id"],
                        "class_name": label["class_name"],
                        "source_xyxy": label["source_xyxy"],
                        "tile_xyxy": label["tile_xyxy"],
                        "yolo_xywh_norm": label["yolo_xywh_norm"],
                        "source_short_side_px": label["source_short_side_px"],
                        "size_bucket": label["size_bucket"]} for label in labels],
            "annotation_complete": True,
            "positive_training_ready": True,
        })
    return {"accepted_tile_count": accepted, "accepted_label_count": label_total,
            "rows": manifest_rows}


# --------------------------------------------------------------------------- #
# summary + manifest (§47-§50)
# --------------------------------------------------------------------------- #


def build_summary(data: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
                  preflight: Mapping[str, Any], *, state: TileReviewState | None = None,
                  merges: Sequence[Mapping[str, Any]] = (), decoded: bool = False,
                  accepted: Mapping[str, Any] | None = None,
                  stats: Mapping[str, Any] | None = None) -> dict[str, Any]:
    rows = apply_review(candidates, state) if state is not None else list(candidates)
    episodes = data["episodes"]
    per_camera: dict[str, dict[str, int]] = {}
    for episode in episodes:
        entry = per_camera.setdefault(str(episode["camera_id"]), {
            "input_eligible": 0, "candidate": 0, "reviewable": 0, "annotation_complete": 0,
            "missing_required": 0, "box_problem": 0, "uncertain": 0,
            "excluded_by_generation": 0, "accepted": 0})
        entry["input_eligible"] += 1

    status_counts = {status: 0 for status in CANDIDATE_STATUSES}
    review_counts = {"PENDING": 0, "ANNOTATION_COMPLETE": 0, "MISSING_REQUIRED": 0,
                     "BOX_PROBLEM": 0, "UNCERTAIN_COMPLETENESS": 0}
    buckets: dict[str, int] = {}
    label_hist: dict[str, int] = {}
    for row in rows:
        camera = per_camera.setdefault(str(row["camera_id"]), {
            "input_eligible": 0, "candidate": 0, "reviewable": 0, "annotation_complete": 0,
            "missing_required": 0, "box_problem": 0, "uncertain": 0,
            "excluded_by_generation": 0, "accepted": 0})
        camera["candidate"] += 1
        status = str(row.get("candidate_generation_status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == STATUS_READY:
            camera["reviewable"] += 1
        else:
            camera["excluded_by_generation"] += 1
        review_status = row.get("annotation_review_status") or "PENDING"
        if status == STATUS_READY:
            review_counts[review_status] = review_counts.get(review_status, 0) + 1
            if review_status == COMPLETE:
                camera["annotation_complete"] += 1
            elif review_status == "MISSING_REQUIRED":
                camera["missing_required"] += 1
            elif review_status == "BOX_PROBLEM":
                camera["box_problem"] += 1
            elif review_status == "UNCERTAIN_COMPLETENESS":
                camera["uncertain"] += 1
            if row.get("positive_training_ready"):
                camera["accepted"] += 1
        for label in row.get("labels") or []:
            key = label.get("size_bucket") or "unknown"
            buckets[key] = buckets.get(key, 0) + 1
        label_hist[str(row.get("label_count") or 0)] = \
            label_hist.get(str(row.get("label_count") or 0), 0) + 1

    completed = [r for r in rows if r.get("annotation_review_status") == COMPLETE]
    accepted_ids = {str(r["tile_id"]) for r in completed}
    represented = {e for r in completed for label in (r.get("labels") or [])
                   for e in label["episode_ids"]}
    lost = sorted(set(str(e["episode_id"]) for e in episodes) - represented)
    progress = (state.progress(candidates) if state is not None
                else {"reviewable": review_counts["PENDING"] + len(completed),
                      "reviewed": len(completed), "pending": review_counts["PENDING"],
                      "skipped": 0})
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "tile_size": TILE_SIZE,
        "class_mapping": {str(CLASS_ID): CLASS_NAME},
        "input": {
            "training_manifest": str(data["training_manifest_path"]),
            "sha256": dict(preflight["hashes"]),
            "training_eligible_episode_count":
                preflight["training_eligible_episode_count"],
            "per_camera": preflight["per_camera"],
            "unique_source_ps_count": preflight["unique_source_ps_count"],
            "resolved_source_resolution": preflight["resolved_source_resolution"],
            "expected_count_ok": preflight["expected_count_ok"],
        },
        "generate": {
            "decoded": bool(decoded),
            "candidate_tile_count": int((stats or {}).get("pre_dedup_count", len(rows))),
            "deduplicated_tile_count": len(rows),
            "merged_duplicate_tile_count": sum(1 for r in rows
                                               if r.get("merged_from_tile_ids")),
            "merge_count": len(list(merges)),
            "decoded_frame_count": int((stats or {}).get("decoded_frames", 0)),
            "reused_candidate_count": int((stats or {}).get("reused_candidates", 0)),
            "probe_failed_source_file_count": int((stats or {}).get("probe_failed", 0)),
            "pruned_orphan_image_count":
                int((stats or {}).get("pruned_orphan_image_count", 0)),
            "extra_decode_round_count":
                int((stats or {}).get("extra_decode_round_count", 0)),
            "generation_failed_count": sum(
                status_counts.get(code, 0) for code in (
                    "DECODE_FAILED", "SOURCE_FRAME_MISMATCH", "CROP_UNSTABLE")),
            "excluded_by_generation_count": sum(1 for r in rows
                                                if r.get("candidate_generation_status")
                                                != STATUS_READY),
            "candidate_accounting": {
                "eligible_episode_count": len(episodes),
                "primary_candidate_count": len(episodes),
                "reviewable_tile_count": review_counts["PENDING"]
                + sum(review_counts.get(d, 0) for d in REVIEW_DECISIONS),
                "known_unlocalized_required_present_count":
                    status_counts.get("KNOWN_UNLOCALIZED_REQUIRED_PRESENT", 0)
                    + status_counts.get("UNCERTAIN_TRUTH_IN_CROP", 0),
                "technical_failure_count": sum(
                    status_counts.get(code, 0) for code in (
                        "DECODE_FAILED", "SOURCE_FRAME_MISMATCH", "CROP_UNSTABLE",
                        "SOURCE_TOO_SMALL", "TARGET_EXCEEDS_TILE",
                        "MULTI_TARGET_EXCEEDS_TILE")),
                "merged_into_another_tile_count":
                    int((stats or {}).get("merged_superseded_count", 0)),
                "distinct_tile_count": len(rows),
            },
            "status_counts": status_counts,
            "source_frame_mismatch_count": status_counts.get("SOURCE_FRAME_MISMATCH", 0),
            "target_exceeds_tile_count": status_counts.get("TARGET_EXCEEDS_TILE", 0),
            "source_too_small_count": status_counts.get("SOURCE_TOO_SMALL", 0),
            "multi_target_exceeds_tile_count":
                status_counts.get("MULTI_TARGET_EXCEEDS_TILE", 0),
            "known_unlocalized_required_present_count":
                status_counts.get("KNOWN_UNLOCALIZED_REQUIRED_PRESENT", 0),
            "uncertain_truth_in_crop_count":
                status_counts.get("UNCERTAIN_TRUTH_IN_CROP", 0),
            "multi_label_tile_count": sum(1 for r in rows
                                          if (r.get("label_count") or 0) >= 2),
            "tiles_with_1_label": sum(1 for r in rows if (r.get("label_count") or 0) == 1),
            "tiles_with_2plus_labels": sum(1 for r in rows
                                           if (r.get("label_count") or 0) >= 2),
            "label_count_histogram": dict(sorted(label_hist.items())),
            "bbox_short_side_buckets": dict(sorted(buckets.items())),
            "per_camera_candidate": {c: v["candidate"] for c, v in per_camera.items()},
        },
        "review": {
            **progress,
            "counts": review_counts,
            "decisions": {d: review_counts.get(d, 0) for d in REVIEW_DECISIONS},
        },
        "accepted": {
            "state": "built" if (accepted or {}).get("accepted_tile_count") else "not_built",
            "accepted_positive_tile_count": (accepted or {}).get("accepted_tile_count", 0),
            "accepted_label_count": (accepted or {}).get("accepted_label_count", 0),
            "unique_episode_ids_represented": len(represented),
            "episodes_lost_due_annotation_incomplete": len(lost),
            "episodes_lost_ids": lost,
            "accepted_tile_ids": sorted(accepted_ids),
        },
        "per_camera": dict(sorted(per_camera.items())),
        "boundaries": {
            "gold_modified": False,
            "recovery_evidence_modified": False,
            "localization_modified": False,
            "truth_reconciliation_modified": False,
            "training_manifest_modified": False,
            "bbox_created_or_modified": False,
            "image_resized": False,
            "hard_negatives_generated": False,
            "detector_run": False,
            "segmentation_run": False,
            "training_started": False,
            "sealed_accessed": False,
            "point_to_bbox": False,
            "augmentation_applied": False,
            "train_val_split_made": False,
        },
    }
    return summary


def build_manifest(data: Mapping[str, Any], summary: Mapping[str, Any], *,
                   code_commit: str, generated_at: str, config: Mapping[str, Any],
                   artifact_root: Path, candidate_index_path: Path,
                   review_state_path: Path, provenance: Mapping[str, Any]
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
        "input_sha256": dict(summary["input"]["sha256"]),
        "artifact_root": str(artifact_root),
        "config": dict(config),
        "outputs": {
            "tile_candidates": str(candidate_index_path),
            "images_dir": str(config.get("images_dir", "")),
            "review_state": str(review_state_path),
            "accepted_root": str(config.get("accepted_root", "")),
        },
        "counts": {
            "input_training_eligible_episode_count":
                summary["input"]["training_eligible_episode_count"],
            "candidate_tile_count": summary["generate"]["candidate_tile_count"],
            "reviewable_tile_count": summary["review"]["reviewable"],
            "reviewed": summary["review"]["reviewed"],
            "accepted_positive_tile_count":
                summary["accepted"]["accepted_positive_tile_count"],
        },
        "upstream_provenance": dict(provenance),
        "boundaries": dict(summary["boundaries"]),
    }


def write_outputs(output_dir: Path, *, candidates: Sequence[Mapping[str, Any]],
                  summary: Mapping[str, Any], manifest: Mapping[str, Any]
                  ) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "tile_candidates.jsonl"
    write_jsonl(candidate_path, candidates)
    summary_path = output_dir / "SUMMARY.json"
    atomic_write_json(summary_path, summary)
    payload = dict(manifest)
    payload["tile_candidates_sha256"] = sha256_file(candidate_path)
    payload["summary_sha256"] = sha256_file(summary_path)
    manifest_path = output_dir / "MANIFEST.json"
    atomic_write_json(manifest_path, payload)
    return {"tile_candidates": str(candidate_path), "summary": str(summary_path),
            "manifest": str(manifest_path)}
