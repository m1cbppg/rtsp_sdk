"""Step 2C-1: Development unlock + frozen Blind Truth (no detector, ever).

This module builds the *human* truth for the Step 0A Development split before any model
output exists.  It is stdlib-only on purpose: nothing here may import or instantiate a
detector, load a checkpoint, or run inference.

Boundaries encoded here:

* Development is allowed; Sealed is a hard failure.  ``assert_development_asset`` accepts
  only paths inside the frozen Development split and rejects anything that names the
  Sealed split, so the two can never be confused.
* Truth classes, episode semantics and the ROI boundary follow the frozen Step 0B
  protocol and are not re-defined.
* Sampling is deterministic: a fixed 5 s grid (at most 5 frames per episode) and a fixed
  30 s grid for the global ROI FP frames.  No content-based or model-based frame picking.
* Bounding boxes, when they exist at all, are in source-frame ``xyxy`` plus
  ``source_width`` / ``source_height`` -- never tile or resize coordinates.
* A point truth is a first-class result; an object that cannot be localized reliably stays
  in the truth as ``LOCALIZATION_UNRESOLVED`` and must go to manual adjudication instead
  of being counted as an automatic FN.
* Freezing makes the truth immutable; later corrections must be appended as a truth
  erratum that records the original value, the reason, the time and whether it happened
  after inference.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "ground_litter_step2c_blind_truth_v1"
GENERATOR_VERSION = "step2c1-1.0.0"
IDENTITY_SCHEMA = "ground_litter_feasibility_raw_ps_identity_v1"

# ---------------------------------------------------------------- split boundaries

#: The frozen Step 0A bundle root and the two splits under it.
BUNDLE_ROOT_MARKER = "ground-litter-detector-feasibility-20260923-r1"
DEVELOPMENT_SPLIT = "development"
SEALED_SPLIT = "sealed_test"
DEVELOPMENT_MARKER = f"{BUNDLE_ROOT_MARKER}/{DEVELOPMENT_SPLIT}"

#: Any of these means the path belongs to (or points into) the Sealed split.
SEALED_MARKERS = (
    SEALED_SPLIT,
    "SEALED_DO_NOT_TUNE",
    "sealed-test",
    "/sealed/",
    "sealed_ps",
)

#: Step 0A freeze markers kept for compatibility with the earlier steps.
LEGACY_SEALED_MARKERS = (
    "sealed_test",
    "SEALED_DO_NOT_TUNE",
    "ground-litter-detector-feasibility-20260923-r1",
    "ground-litter-feasibility/20260923-r1",
)


class BlindTruthError(RuntimeError):
    """Blind truth work could not proceed."""


class SealedAssetError(BlindTruthError):
    """A Step 0A Sealed asset was supplied; hard fail."""


class DevelopmentScopeError(BlindTruthError):
    """A path is outside the authorised Development split."""


class TruthFrozenError(BlindTruthError):
    """The truth is frozen; use a truth erratum instead of an in-place edit."""


class DetectorUseError(BlindTruthError):
    """Something tried to bring detector output into the blind truth step."""


def assert_not_sealed(*values: Any) -> None:
    """Hard-fail if any value names a Step 0A Sealed asset (legacy marker set)."""
    for value in values:
        text = str(value or "")
        if not text:
            continue
        lowered = text.lower()
        for marker in LEGACY_SEALED_MARKERS:
            if marker.lower() in lowered:
                raise SealedAssetError(
                    f"refusing to touch a Step 0A Sealed asset (matched {marker!r})")


def assert_development_asset(*values: Any) -> None:
    """Hard-fail unless every value is inside the authorised Development split.

    Raises :class:`SealedAssetError` for anything that names the Sealed split and
    :class:`DevelopmentScopeError` for anything outside Development, so a Development
    path can never be silently swapped for a Sealed one.
    """
    for value in values:
        text = str(value or "")
        if not text:
            raise DevelopmentScopeError("empty path is not a Development asset")
        lowered = text.lower()
        for marker in SEALED_MARKERS:
            if marker.lower() in lowered:
                raise SealedAssetError(
                    f"refusing to touch a Sealed asset (matched {marker!r})")
        if DEVELOPMENT_MARKER not in text:
            raise DevelopmentScopeError(
                f"not inside the authorised Development split: {text!r}")


# ---------------------------------------------------------------- small helpers


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path | str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def parse_ts(text: str) -> datetime:
    """Parse an absolute ``YYYY-MM-DD HH:MM:SS[.fff]`` timestamp (camera local time)."""
    text = str(text).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise BlindTruthError(f"unparsable timestamp: {text!r}")


def format_ts(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------- truth vocabulary

REQUIRED_LITTER = "REQUIRED_LITTER"
IGNORE_SMALL = "IGNORE_SMALL"
UNCERTAIN = "UNCERTAIN"
NON_LITTER = "NON_LITTER"
TRUTH_CLASSES = (REQUIRED_LITTER, IGNORE_SMALL, UNCERTAIN, NON_LITTER)
#: Only Required enters the recall denominator; the other two still need to be kept so the
#: evaluator can ignore a detection instead of counting it as FP.
DENOMINATOR_CLASSES = (REQUIRED_LITTER,)
IGNORE_CLASSES = (IGNORE_SMALL, UNCERTAIN)

LOCALIZATION_POINT = "POINT_TRUTH"
LOCALIZATION_BOX_OK = "BOX_OK"
LOCALIZATION_PROPOSAL = "PROPOSAL_SELECTED"
LOCALIZATION_UNRESOLVED = "LOCALIZATION_UNRESOLVED"
LOCALIZATION_STATUSES = (LOCALIZATION_POINT, LOCALIZATION_BOX_OK,
                         LOCALIZATION_PROPOSAL, LOCALIZATION_UNRESOLVED)

REVIEW_PENDING = "PENDING"
REVIEW_IN_PROGRESS = "IN_PROGRESS"
REVIEW_DONE = "REVIEWED"
REVIEW_STATUSES = (REVIEW_PENDING, REVIEW_IN_PROGRESS, REVIEW_DONE)

EPISODE_DRAFT = "DRAFT"
EPISODE_CONFIRMED = "CONFIRMED"
EPISODE_STATUSES = (EPISODE_DRAFT, EPISODE_CONFIRMED)

SAMPLE_REASON_VISIBLE = "fixed_5s_grid"
SAMPLE_REASON_GLOBAL = "fixed_30s_grid"
VISIBLE_GRID_SECONDS = 5.0
GLOBAL_GRID_SECONDS = 30.0
MAX_VISIBLE_FRAMES_PER_EPISODE = 5

#: §25 evidence levels for the Development split.
EVIDENCE_EXPLORATORY = "exploratory"
EVIDENCE_DIRECTIONAL = "directional"
EVIDENCE_STRONGER = "stronger_evidence"


def evidence_classification(required_episodes: int, camera_count: int) -> str:
    if required_episodes < 20:
        return EVIDENCE_EXPLORATORY
    if required_episodes < 50:
        return EVIDENCE_DIRECTIONAL
    if camera_count >= 3:
        return EVIDENCE_STRONGER
    return EVIDENCE_DIRECTIONAL


# ---------------------------------------------------------------- ROI geometry


def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray casting on normalized coordinates; the boundary counts as inside."""
    inside = False
    count = len(polygon)
    if count < 3:
        return False
    j = count - 1
    for i in range(count):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x <= x_cross:
                inside = not inside
        j = i
    return inside


def in_roi(point_xy: Sequence[float], roi: Sequence[Sequence[float]],
           source_width: int, source_height: int) -> bool:
    """Normalize a source-pixel point and test it against the frozen normalized ROI."""
    if not roi:
        return False
    x_norm = float(point_xy[0]) / float(source_width)
    y_norm = float(point_xy[1]) / float(source_height)
    return point_in_polygon(x_norm, y_norm, roi)


# ---------------------------------------------------------------- Development inventory


def load_roi_configs(roi_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Load the frozen per-camera ROI configs (normalized polygons)."""
    roi_dir = Path(roi_dir)
    configs: dict[str, dict[str, Any]] = {}
    for path in sorted(roi_dir.glob("ground_litter_*_final_roi.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        camera = str(payload["camera_id"])
        configs[camera] = {
            "camera_id": camera,
            "device_code": payload.get("device_code"),
            "geometry_version": payload.get("geometry_version"),
            "canvas_size": list(payload.get("canvas_size") or []),
            "roi": [[float(v) for v in point] for point in payload.get("roi") or []],
            "exclude_zones": [[[float(v) for v in point] for point in zone]
                              for zone in payload.get("exclude_zones") or []],
            "config_path": str(path),
            "config_sha256": sha256_file(path),
            "frame_sha256": payload.get("frame_sha256"),
        }
    return configs


def load_development_inventory(development_root: Path | str,
                               roi_dir: Path | str) -> dict[str, Any]:
    """Read the 65 frozen Development PS identities; Sealed is never touched."""
    development_root = Path(development_root)
    assert_development_asset(development_root)
    rois = load_roi_configs(roi_dir)
    records: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    for sidecar in sorted(development_root.glob("*/raw/*.ps.identity.json")):
        assert_development_asset(sidecar)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("schema") != IDENTITY_SCHEMA:
            problems.append({"file": str(sidecar), "field": "schema"})
            continue
        if payload.get("split") != DEVELOPMENT_SPLIT:
            # A Sealed identity inside the Development tree is a hard failure, not a skip.
            raise SealedAssetError(
                f"{sidecar} declares split={payload.get('split')!r}; only "
                f"{DEVELOPMENT_SPLIT!r} is authorised")
        ps_path = sidecar.with_name(sidecar.name.replace(".identity.json", ""))
        if not ps_path.is_file():
            problems.append({"file": str(ps_path), "field": "ps_missing"})
        camera = str(payload["camera_id"])
        roi = rois.get(camera)
        if roi is None:
            problems.append({"file": str(sidecar), "field": "roi_config_missing",
                             "camera_id": camera})
        elif roi["geometry_version"] != payload["roi"]["geometry_version"]:
            problems.append({"file": str(sidecar), "field": "roi_geometry_version_mismatch",
                             "sidecar": payload["roi"]["geometry_version"],
                             "config": roi["geometry_version"]})
        elif (roi.get("frame_sha256") or "") != (payload["roi"].get("reference_frame_sha256") or ""):
            problems.append({"file": str(sidecar), "field": "roi_reference_frame_mismatch"})
        if int(payload["actual_bytes"]) != int(payload["declared_bytes"]):
            problems.append({"file": str(sidecar), "field": "byte_count_mismatch"})
        records.append({
            "file_id": str(payload["file_id"]),
            "ps_path": str(ps_path),
            "identity_path": str(sidecar),
            "camera_id": camera,
            "device_code": payload.get("device_code"),
            "scene_version": str(payload["scene_version"]),
            "file_name": payload.get("file_name"),
            "record_start": str(payload["record_start"]),
            "record_end": str(payload["record_end"]),
            "bytes": int(payload["actual_bytes"]),
            "sha256": str(payload["sha256"]),
            "canvas_size": list((payload.get("roi") or {}).get("canvas_size") or []),
            "roi": list((roi or {}).get("roi") or []),
            "roi_geometry_version": (roi or {}).get("geometry_version"),
            "roi_config_path": (roi or {}).get("config_path"),
            "roi_available": bool(roi),
        })
    records.sort(key=lambda row: (row["camera_id"], row["record_start"], row["file_id"]))
    for row in records:
        row["duration_seconds"] = round(
            (parse_ts(row["record_end"]) - parse_ts(row["record_start"])).total_seconds(), 3)
    per_camera: dict[str, int] = {}
    for row in records:
        per_camera[row["camera_id"]] = per_camera.get(row["camera_id"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "split": DEVELOPMENT_SPLIT,
        "development_root": str(development_root),
        "sealed_accessed": False,
        "ps_count": len(records),
        "total_bytes": sum(row["bytes"] for row in records),
        "per_camera_count": dict(sorted(per_camera.items())),
        "cameras": sorted(per_camera),
        "roi_geometry_version": sorted({row["roi_geometry_version"] for row in records
                                        if row["roi_geometry_version"]}),
        "problems": problems,
        "ok": not problems,
        "files": records,
    }


# ---------------------------------------------------------------- truth objects


def new_truth_object(*, truth_id: str, camera_id: str, scene_version: str,
                     source_file_id: str, timestamp: str, decoded_timestamp: float | None,
                     frame_index: int | None, truth_class: str,
                     source_width: int, source_height: int,
                     source_point: Sequence[float] | None = None,
                     source_bbox_xyxy: Sequence[float] | None = None,
                     roi: Sequence[Sequence[float]] | None = None,
                     episode_id: str | None = None,
                     localization_status: str | None = None,
                     note: str | None = None) -> dict[str, Any]:
    """Build one human truth observation.

    Geometry is always source-frame: a point is ``[x, y]`` in source pixels and a box is
    ``[x1, y1, x2, y2]`` in source pixels, both accompanied by the source frame size.
    """
    if truth_class not in TRUTH_CLASSES:
        raise BlindTruthError(f"unknown truth class {truth_class!r}")
    status = localization_status
    if status is None:
        status = LOCALIZATION_BOX_OK if source_bbox_xyxy else (
            LOCALIZATION_POINT if source_point else LOCALIZATION_UNRESOLVED)
    if status not in LOCALIZATION_STATUSES:
        raise BlindTruthError(f"unknown localization status {status!r}")
    if status in (LOCALIZATION_BOX_OK, LOCALIZATION_PROPOSAL) and not source_bbox_xyxy:
        raise BlindTruthError(f"{status} requires a source bbox")
    if source_bbox_xyxy is not None:
        x1, y1, x2, y2 = (float(v) for v in source_bbox_xyxy)
        source_bbox_xyxy = [x1, y1, x2, y2]
        if x2 <= x1 or y2 <= y1:
            raise BlindTruthError("bbox must be x2>x1 and y2>y1")
        if x1 < 0 or y1 < 0 or x2 > source_width or y2 > source_height:
            raise BlindTruthError("bbox must lie inside the source frame")
    point = None
    if source_point is not None:
        point = [float(source_point[0]), float(source_point[1])]
        if not (0.0 <= point[0] <= source_width and 0.0 <= point[1] <= source_height):
            raise BlindTruthError("point must lie inside the source frame")
    inside = None
    if roi:
        probe = point or ([(source_bbox_xyxy[0] + source_bbox_xyxy[2]) / 2.0,
                            (source_bbox_xyxy[1] + source_bbox_xyxy[3]) / 2.0]
                          if source_bbox_xyxy else None)
        if probe:
            inside = in_roi(probe, roi, source_width, source_height)
    return {
        "truth_id": truth_id, "camera_id": camera_id, "scene_version": scene_version,
        "source_file_id": source_file_id, "timestamp": timestamp,
        "decoded_timestamp": decoded_timestamp, "frame_index": frame_index,
        "truth_class": truth_class,
        "source_width": int(source_width), "source_height": int(source_height),
        "source_point": point, "source_bbox_xyxy": source_bbox_xyxy,
        "localization_status": status, "in_roi": inside,
        "enters_recall_denominator": truth_class in DENOMINATOR_CLASSES,
        "enters_ignore_set": truth_class in IGNORE_CLASSES,
        "episode_id": episode_id, "note": note,
        "created_at": _now(), "updated_at": _now(),
    }


def next_id(prefix: str, existing: Iterable[str]) -> str:
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
    top = 0
    for value in existing:
        match = pattern.match(str(value))
        if match:
            top = max(top, int(match.group(1)))
    return f"{prefix}-{top + 1:04d}"


# ---------------------------------------------------------------- episodes


def episode_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    """Episode identity is camera + scene_version (Step 0B §4)."""
    return str(row["camera_id"]), str(row["scene_version"])


def same_episode_candidate(a: Mapping[str, Any], b: Mapping[str, Any], *,
                           max_gap_seconds: float = 300.0,
                           max_distance_px: float = 400.0) -> dict[str, Any]:
    """Suggest whether two observations belong to one physical litter item.

    This is deliberately a *suggestion* for the human: it only uses camera, scene_version,
    time continuity and point distance.  No detector output is involved.
    """
    same_identity = episode_identity(a) == episode_identity(b)
    gap = abs((parse_ts(a["timestamp"]) - parse_ts(b["timestamp"])).total_seconds())
    distance = None
    if a.get("source_point") and b.get("source_point"):
        dx = float(a["source_point"][0]) - float(b["source_point"][0])
        dy = float(a["source_point"][1]) - float(b["source_point"][1])
        distance = (dx * dx + dy * dy) ** 0.5
    continuous = gap <= max_gap_seconds
    close = distance is not None and distance <= max_distance_px
    return {"same_camera_scene": same_identity, "gap_seconds": round(gap, 3),
            "distance_px": None if distance is None else round(distance, 3),
            "suggest_same_episode": bool(same_identity and continuous and close),
            "criterion": "camera+scene_version, time continuity, point distance only"}


def new_episode(*, episode_id: str, observation: Mapping[str, Any],
                physical_identity_note: str | None = None,
                review_status: str = EPISODE_DRAFT) -> dict[str, Any]:
    if review_status not in EPISODE_STATUSES:
        raise BlindTruthError(f"unknown episode status {review_status!r}")
    camera, scene = episode_identity(observation)
    point = observation.get("source_point") or (
        [(observation["source_bbox_xyxy"][0] + observation["source_bbox_xyxy"][2]) / 2.0,
         (observation["source_bbox_xyxy"][1] + observation["source_bbox_xyxy"][3]) / 2.0]
        if observation.get("source_bbox_xyxy") else None)
    return {
        "episode_id": episode_id, "camera_id": camera, "scene_version": scene,
        "truth_class": observation["truth_class"],
        "first_confirmable_timestamp": observation["timestamp"],
        "last_confirmable_timestamp": observation["timestamp"],
        "source_file_ids": [observation["source_file_id"]],
        "representative_points": [point] if point else [],
        "observation_ids": [observation["truth_id"]],
        "physical_identity_note": physical_identity_note,
        "review_status": review_status,
        "created_at": _now(), "updated_at": _now(),
    }


def refresh_episode(episode: Mapping[str, Any],
                    observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute the visible interval, file list and class from the linked observations."""
    rows = [row for row in observations
            if row.get("episode_id") == episode["episode_id"]]
    updated = dict(episode)
    if rows:
        stamps = sorted(str(row["timestamp"]) for row in rows)
        updated["first_confirmable_timestamp"] = stamps[0]
        updated["last_confirmable_timestamp"] = stamps[-1]
        updated["source_file_ids"] = sorted({str(row["source_file_id"]) for row in rows})
        updated["observation_ids"] = [row["truth_id"] for row in rows]
        points = [row["source_point"] for row in rows if row.get("source_point")]
        if points:
            updated["representative_points"] = points
        required = [row for row in rows if row["truth_class"] == REQUIRED_LITTER]
        updated["truth_class"] = (required[0]["truth_class"] if required
                                  else rows[0]["truth_class"])
    updated["observation_count"] = len(rows)
    updated["updated_at"] = _now()
    return updated


def assert_episode_consistency(episode: Mapping[str, Any],
                               observations: Sequence[Mapping[str, Any]]) -> None:
    """Hard-fail if an episode links observations from another camera or scene."""
    identity = episode_identity(episode)
    for row in observations:
        if row.get("episode_id") != episode["episode_id"]:
            continue
        if episode_identity(row) != identity:
            raise BlindTruthError(
                f"episode {episode['episode_id']} mixes camera/scene: {identity} vs "
                f"{episode_identity(row)}")
    if episode["truth_class"] not in TRUTH_CLASSES:
        raise BlindTruthError(f"unknown episode class {episode['truth_class']!r}")
    first, last = episode["first_confirmable_timestamp"], episode["last_confirmable_timestamp"]
    if parse_ts(last) < parse_ts(first):
        raise BlindTruthError("episode last_confirmable_timestamp precedes the first")


# ---------------------------------------------------------------- deterministic sampling


def _map_timestamp(timestamp: str, files: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    moment = parse_ts(timestamp)
    for row in files:
        if parse_ts(row["record_start"]) <= moment < parse_ts(row["record_end"]):
            return row
    # tolerate an endpoint exactly at the end of the last chunk
    for row in files:
        if moment == parse_ts(row["record_end"]):
            return row
    raise BlindTruthError(f"timestamp {timestamp} is outside the Development PS range")


def sample_visible_frames(episodes: Sequence[Mapping[str, Any]],
                          truth_objects: Sequence[Mapping[str, Any]],
                          files: Sequence[Mapping[str, Any]], *,
                          grid_seconds: float = VISIBLE_GRID_SECONDS,
                          max_frames: int = MAX_VISIBLE_FRAMES_PER_EPISODE
                          ) -> list[dict[str, Any]]:
    """Fixed grid over each Required episode's visible interval, capped deterministically.

    The first frame is the first confirmable timestamp, then one frame every
    ``grid_seconds`` while the interval lasts; the first ``max_frames`` timestamps are
    kept.  Nothing here looks at image content.
    """
    if max_frames < 1:
        raise BlindTruthError("max_frames must be >= 1")
    by_id = {row["truth_id"]: row for row in truth_objects}
    out: list[dict[str, Any]] = []
    for episode in sorted(episodes, key=lambda row: row["episode_id"]):
        if episode["truth_class"] != REQUIRED_LITTER:
            continue
        if episode.get("review_status") != EPISODE_CONFIRMED:
            continue
        start = parse_ts(episode["first_confirmable_timestamp"])
        end = parse_ts(episode["last_confirmable_timestamp"])
        if end < start:
            raise BlindTruthError(f"{episode['episode_id']}: interval is inverted")
        timestamps: list[datetime] = []
        offset = 0.0
        while len(timestamps) < max_frames:
            moment = start + timedelta(seconds=offset)
            if moment > end:
                break
            timestamps.append(moment)
            offset += grid_seconds
        if not timestamps:
            timestamps = [start]
        rows = [by_id[oid] for oid in episode.get("observation_ids") or []
                if oid in by_id]
        anchor = rows[0] if rows else None
        for index, moment in enumerate(timestamps):
            stamp = format_ts(moment)
            record = _map_timestamp(stamp, files)
            out.append({
                "frame_id": f"vf-{episode['episode_id']}-{index + 1:02d}",
                "episode_id": episode["episode_id"],
                "camera_id": episode["camera_id"],
                "scene_version": episode["scene_version"],
                "source_file_id": record["file_id"],
                "timestamp": stamp,
                "decoded_timestamp": round(
                    (moment - parse_ts(record["record_start"])).total_seconds(), 3),
                "source_width": int(record["canvas_size"][0]),
                "source_height": int(record["canvas_size"][1]),
                "truth_class": REQUIRED_LITTER,
                "source_bbox_xyxy": (anchor or {}).get("source_bbox_xyxy"),
                "source_point": (anchor or {}).get("source_point"),
                "localization_status": (anchor or {}).get(
                    "localization_status", LOCALIZATION_UNRESOLVED),
                "sample_reason": SAMPLE_REASON_VISIBLE,
                "grid_seconds": grid_seconds,
                "grid_index": index,
            })
    return out


def sample_global_roi_frames(cameras: Mapping[str, Mapping[str, Any]],
                             files: Sequence[Mapping[str, Any]], *,
                             grid_seconds: float = GLOBAL_GRID_SECONDS
                             ) -> list[dict[str, Any]]:
    """Fixed 30 s grid over every Development PS, independent of content (Step 0B §7.2)."""
    out: list[dict[str, Any]] = []
    for record in sorted(files, key=lambda row: (row["camera_id"], row["record_start"])):
        camera = cameras.get(record["camera_id"]) or {}
        start = parse_ts(record["record_start"])
        end = parse_ts(record["record_end"])
        index = 0
        offset = 0.0
        while True:
            moment = start + timedelta(seconds=offset)
            if moment >= end:
                break
            index += 1
            out.append({
                "global_frame_id": f"gf-{record['camera_id']}-{record['file_id']}-{index:02d}",
                "camera_id": record["camera_id"],
                "scene_version": record["scene_version"],
                "source_file_id": record["file_id"],
                "timestamp": format_ts(moment),
                "decoded_timestamp": round(offset, 3),
                "source_width": int(record["canvas_size"][0]),
                "source_height": int(record["canvas_size"][1]),
                "roi": list(camera.get("roi") or []),
                "roi_geometry_version": camera.get("geometry_version"),
                "sample_reason": SAMPLE_REASON_GLOBAL,
                "grid_seconds": grid_seconds,
                "grid_index": index,
            })
            offset += grid_seconds
    return out


# ---------------------------------------------------------------- localization state


def localization_state(episodes: Sequence[Mapping[str, Any]],
                       truth_objects: Sequence[Mapping[str, Any]],
                       visible_frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise what can be auto-matched and what must go to manual adjudication."""
    per_object = []
    for row in sorted(truth_objects, key=lambda item: item["truth_id"]):
        if row["truth_class"] != REQUIRED_LITTER:
            continue
        per_object.append({
            "truth_id": row["truth_id"], "camera_id": row["camera_id"],
            "episode_id": row.get("episode_id"),
            "localization_status": row["localization_status"],
            "has_bbox": bool(row.get("source_bbox_xyxy")),
            "has_point": bool(row.get("source_point")),
            "needs_manual_adjudication":
                row["localization_status"] == LOCALIZATION_UNRESOLVED,
        })
    counts: dict[str, int] = {}
    for row in per_object:
        counts[row["localization_status"]] = counts.get(row["localization_status"], 0) + 1
    unresolved = [row["truth_id"] for row in per_object
                  if row["needs_manual_adjudication"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "required_objects": len(per_object),
        "per_status_counts": dict(sorted(counts.items())),
        "auto_matchable": sum(1 for row in per_object if row["has_bbox"]),
        "point_only": sum(1 for row in per_object
                          if row["has_point"] and not row["has_bbox"]),
        "unresolved": unresolved,
        "unresolved_count": len(unresolved),
        "rule": ("a Required object without a reliable bbox is kept in the truth and "
                 "routed to manual adjudication; it must never be scored as an automatic "
                 "false negative"),
        "objects": per_object,
        "visible_frames": len(visible_frames),
        "episodes": len(episodes),
    }


# ---------------------------------------------------------------- review coverage


def review_coverage(files: Sequence[Mapping[str, Any]],
                    review_state: Mapping[str, Any]) -> dict[str, Any]:
    """How much of the Development split the human has actually watched (§29)."""
    statuses = review_state.get("files") or {}
    per_camera: dict[str, dict[str, int]] = {}
    pending: list[str] = []
    for row in files:
        camera = row["camera_id"]
        slot = per_camera.setdefault(camera, {"total": 0, "reviewed": 0, "in_progress": 0,
                                             "pending": 0})
        slot["total"] += 1
        status = (statuses.get(row["file_id"]) or {}).get("status", REVIEW_PENDING)
        if status == REVIEW_DONE:
            slot["reviewed"] += 1
        elif status == REVIEW_IN_PROGRESS:
            slot["in_progress"] += 1
        else:
            slot["pending"] += 1
        if status != REVIEW_DONE:
            pending.append(row["file_id"])
    total = len(files)
    reviewed = sum(slot["reviewed"] for slot in per_camera.values())
    return {
        "total_files": total, "reviewed": reviewed, "pending": total - reviewed,
        "reviewed_fraction": round(reviewed / total, 4) if total else 0.0,
        "per_camera": dict(sorted(per_camera.items())),
        "pending_file_ids": pending,
        "complete": reviewed == total and total > 0,
    }


def assert_review_complete(coverage: Mapping[str, Any]) -> None:
    if not coverage.get("complete"):
        raise BlindTruthError(
            f"Blind Truth cannot freeze: {coverage.get('reviewed')}/"
            f"{coverage.get('total_files')} Development PS reviewed")


# ---------------------------------------------------------------- freeze


FROZEN_ARTIFACTS = ("truth_objects.jsonl", "episodes.jsonl",
                    "visible_frame_manifest.jsonl", "global_roi_frame_manifest.jsonl",
                    "development_manifest.json", "localization_state.json",
                    "SUMMARY.json", "MANIFEST.json")


def freeze_truth(artifact_dir: Path | str, *, review: Mapping[str, Any],
                 extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Freeze the artifact by SHA-256 and strip the write bits (§23/§24)."""
    artifact_dir = Path(artifact_dir)
    assert_review_complete(review)
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for name in FROZEN_ARTIFACTS:
        path = artifact_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        hashes[name] = sha256_file(path)
    if missing:
        raise BlindTruthError(f"cannot freeze, missing artifacts: {missing}")
    combined = sha256_file_lines(hashes)
    record = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "frozen_at": _now(),
        "artifact_sha256": hashes,
        "truth_sha256": combined,
        "artifacts": list(FROZEN_ARTIFACTS),
        "review_coverage": dict(review),
        "detector_loaded": False, "checkpoint_accessed": False, "inference_executed": False,
        "sealed_accessed": False,
        "extra": dict(extra or {}),
    }
    atomic_write_json(artifact_dir / "FREEZE.json", record)
    read_only: dict[str, Any] = {}
    for path in sorted(artifact_dir.glob("*")):
        if path.is_file():
            mode = path.stat().st_mode & 0o777
            os.chmod(path, mode & ~0o222)
            read_only[path.name] = {"mode_before": oct(mode),
                                    "mode_after": oct(path.stat().st_mode & 0o777)}
    record["read_only"] = read_only
    return record


def sha256_file_lines(hashes: Mapping[str, str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashes[name].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def is_frozen(artifact_dir: Path | str) -> bool:
    return (Path(artifact_dir) / "FREEZE.json").is_file()


def assert_not_frozen(artifact_dir: Path | str) -> None:
    if is_frozen(artifact_dir):
        raise TruthFrozenError(
            "Blind Truth is frozen; append a truth erratum instead of editing in place")


def append_truth_erratum(artifact_dir: Path | str, *, truth_id: str,
                         original: Mapping[str, Any], corrected: Mapping[str, Any],
                         reason: str, after_inference: bool) -> dict[str, Any]:
    """Post-freeze corrections go here, never into the frozen truth (§24)."""
    if not is_frozen(artifact_dir):
        raise BlindTruthError("truth erratum is only for a frozen Blind Truth")
    if not reason:
        raise BlindTruthError("a truth erratum must state a reason")
    row = {
        "schema_version": SCHEMA_VERSION,
        "erratum_at": _now(),
        "truth_id": truth_id,
        "original": dict(original),
        "corrected": dict(corrected),
        "reason": reason,
        "recorded_after_inference": bool(after_inference),
    }
    path = Path(artifact_dir) / "TRUTH_ERRATA.jsonl"
    rows = read_jsonl(path)
    rows.append(row)
    write_jsonl(path, rows)
    record = json.loads((Path(artifact_dir) / "FREEZE.json").read_text(encoding="utf-8"))
    record.setdefault("errata", []).append({
        "erratum_at": row["erratum_at"], "truth_id": truth_id,
        "recorded_after_inference": bool(after_inference),
    })
    atomic_write_json(Path(artifact_dir) / "FREEZE.json", record)
    return row


# ---------------------------------------------------------------- summary / manifest


def build_summary(inventory: Mapping[str, Any], *, truth_objects: Sequence[Mapping[str, Any]],
                  episodes: Sequence[Mapping[str, Any]],
                  visible_frames: Sequence[Mapping[str, Any]],
                  global_frames: Sequence[Mapping[str, Any]],
                  localization: Mapping[str, Any], review: Mapping[str, Any],
                  roi_note: str) -> dict[str, Any]:
    required = [row for row in truth_objects if row["truth_class"] == REQUIRED_LITTER]
    required_episodes = [row for row in episodes if row["truth_class"] == REQUIRED_LITTER]
    ignore = [row for row in truth_objects if row["truth_class"] == IGNORE_SMALL]
    uncertain = [row for row in truth_objects if row["truth_class"] == UNCERTAIN]
    non_litter = [row for row in truth_objects if row["truth_class"] == NON_LITTER]
    per_camera_required: dict[str, int] = {}
    for row in required_episodes:
        per_camera_required[row["camera_id"]] = per_camera_required.get(row["camera_id"], 0) + 1
    cameras_with_required = len([c for c, n in per_camera_required.items() if n > 0])
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_at": _now(),
        "development": {
            "ps_count": inventory["ps_count"],
            "total_bytes": inventory["total_bytes"],
            "per_camera_count": inventory["per_camera_count"],
            "cameras": inventory["cameras"],
            "roi_note": roi_note,
        },
        "truth": {
            "natural_required_episode_count": len(required_episodes),
            "required_object_count": len(required),
            "ignore_small_object_count": len(ignore),
            "ignore_small_episode_count": len(
                [row for row in episodes if row["truth_class"] == IGNORE_SMALL]),
            "uncertain_object_count": len(uncertain),
            "non_litter_object_count": len(non_litter),
            "per_camera_required_episode_count": dict(sorted(per_camera_required.items())),
            "cameras_with_required": cameras_with_required,
        },
        "evidence_classification": evidence_classification(len(required_episodes),
                                                           cameras_with_required),
        "evidence_rule": ("<20 Required episodes = exploratory; 20-49 = directional; "
                          ">=50 with >=3 cameras = stronger evidence; recorded only, no "
                          "model judgement is made in this step"),
        "sampling": {
            "visible_frames": len(visible_frames),
            "visible_reason": SAMPLE_REASON_VISIBLE,
            "visible_grid_seconds": VISIBLE_GRID_SECONDS,
            "max_frames_per_episode": MAX_VISIBLE_FRAMES_PER_EPISODE,
            "global_roi_frames": len(global_frames),
            "global_reason": SAMPLE_REASON_GLOBAL,
            "global_grid_seconds": GLOBAL_GRID_SECONDS,
        },
        "localization": {
            "per_status_counts": localization["per_status_counts"],
            "auto_matchable": localization["auto_matchable"],
            "point_only": localization["point_only"],
            "unresolved_count": localization["unresolved_count"],
        },
        "review_coverage": review,
        "blindness": {
            "detector_loaded": False, "checkpoint_accessed": False,
            "inference_executed": False, "proposal_recall_computed": False,
            "threshold_selection_performed": False,
        },
        "boundaries": {
            "development_accessed": True, "sealed_accessed": False,
            "training_pool_modified": False, "checkpoint_modified": False,
            "truth_modified_after_freeze": False,
        },
        "note": ("Blind Truth was built before any detector inference for this step; no "
                 "model output was loaded, displayed or used for frame selection"),
    }


def build_manifest(inventory: Mapping[str, Any], summary: Mapping[str, Any], *,
                   generated_at: str, code_commit: str, artifact_root: Path | str,
                   truth_sha256: str | None = None,
                   review: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_at": generated_at,
        "code_commit": code_commit,
        "artifact_root": str(artifact_root),
        "development_sha256": {row["file_id"]: row["sha256"] for row in inventory["files"]},
        "development_ps_count": inventory["ps_count"],
        "roi_geometry_version": inventory["roi_geometry_version"],
        "review_coverage": dict(review or {}),
        "truth_sha256": truth_sha256,
        "blindness": dict(summary["blindness"]),
        "boundaries": dict(summary["boundaries"]),
        "note": summary["note"],
    }
