"""Step 1B: human confirmation of historical Silver episode identity + trainability audit.

Scope (see ``docs/plans/2026-09-23-ground-litter-detector-feasibility-plan.md``
§10.1 and the frozen ``…step0b-evaluation-protocol.md``):

* Step 1A produced automatic *episode candidates*.  This module turns them into a
  human-reviewable workflow whose output is a human-confirmed ``episode_id``.
  ``episode_candidate_id`` and ``episode_id`` are separate identities and must
  never be conflated.
* It only ever *audits* whether a confirmed episode can be traced back to a
  source-native frame.  It does not build 640x640 tiles, does not train, and does
  not touch Sealed data or any detector output.
* The Step 1A artifact is treated as an immutable input; nothing here writes to it.
* Pure stdlib on purpose: the test virtualenv has no cv2, and this step needs no
  image processing beyond reading JPEG dimensions from file headers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid

SCHEMA_VERSION = "ground_litter_gold_episode_review_v1"
#: State-file schema.  Bumped to v2 when manual missing targets were added; the
#: loader stays tolerant so an existing v1 review_state keeps every decision.
REVIEW_SCHEMA_VERSION = "gold_episode_review_v2"
#: FROZEN hash namespace for ``ge-<camera>-<hash>`` episode ids.  Deliberately NOT
#: tied to REVIEW_SCHEMA_VERSION: bumping the state schema must never rename an
#: episode that a human already confirmed.
EPISODE_ID_NAMESPACE = "gold_episode_review_v1"

DECISIONS = ("CONFIRM", "SPLIT", "MERGE", "NON_LITTER", "IGNORE_SMALL", "UNCERTAIN")
EPISODE_PRODUCING_DECISIONS = ("CONFIRM", "SPLIT", "MERGE")
NON_EPISODE_DECISIONS = ("NON_LITTER", "IGNORE_SMALL", "UNCERTAIN")

TRUTH_CLASSES = ("REQUIRED_LITTER", "IGNORE_SMALL", "NON_LITTER", "UNCERTAIN")
LOCALIZATION_STATUSES = ("OK", "NEEDS_RELOCALIZATION")
TRAINABILITY_STATUSES = ("TRAINABLE_SOURCE_NATIVE", "REVIEW_ONLY", "LINEAGE_UNRESOLVED")

#: Historical Silver has no scene_version.  Never invent one (plan §13).
UNKNOWN_SCENE_VERSION = "UNKNOWN_HISTORICAL"

# --- human-discovered missing targets -------------------------------------- #
#: A reviewer saw a second (third, ...) independent piece of litter in the same
#: image that no historical candidate ever boxed.  Step 0B §3.5 forbids treating
#: "this frame has litter" as per-object truth, so each one becomes its own target.
MANUAL_TARGET_ORIGIN = "HUMAN_DISCOVERED_MISSING_TARGET"
#: NON_LITTER is deliberately absent: this action only ever *adds* a target.
MANUAL_TARGET_TRUTH_CLASSES = ("REQUIRED_LITTER", "IGNORE_SMALL", "UNCERTAIN")
MANUAL_TARGET_ASSET_TYPES = ("context", "current")
#: Two manual points closer than this trigger a UI prompt.  Compared in native
#: pixels *of the shared clicked asset* (see _manual_points_close): the assets are
#: crops, so a normalised crop coordinate is not a source-frame coordinate and
#: must never be compared across different assets.
MANUAL_TARGET_DUPLICATE_PX = 10.0
MANUAL_TARGET_DUPLICATE_FRACTION = 0.02
#: Silver review cards sample their before/after evidence frames at +/- this many
#: seconds (mirrors the Step 1A grouping default, which documents the sampling).
SILVER_BEFORE_AFTER_SECONDS = 2.0

GROUPING_RISK_REASONS = (
    "borderline_within_threshold",
    "near_threshold_pair_possible_false_split",
    "complete_linkage_split_possible_false_split",
    "same_frame_merge_risk",
    "same_frame_neighbour_candidate",
    "box_wrong_participates_in_grouping",
)
LINEAGE_ONLY_REASONS = ("lineage_source_file_id_unavailable",)

DEFAULT_RETENTION_DAYS = 7.0


# --------------------------------------------------------------------------- #
# small stdlib helpers
# --------------------------------------------------------------------------- #


def jpeg_dimensions(path: Path | str) -> tuple[int, int] | None:
    """Read (width, height) from a JPEG header without any image library."""
    try:
        with open(path, "rb") as handle:
            if handle.read(2) != b"\xff\xd8":
                return None
            while True:
                byte = handle.read(1)
                while byte and byte != b"\xff":
                    byte = handle.read(1)
                if not byte:
                    return None
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if not marker:
                    return None
                code = marker[0]
                if code in (0xD8, 0xD9) or 0xD0 <= code <= 0xD7:
                    continue
                raw_length = handle.read(2)
                if len(raw_length) != 2:
                    return None
                length = int.from_bytes(raw_length, "big")
                if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                    payload = handle.read(5)
                    if len(payload) < 5:
                        return None
                    height = int.from_bytes(payload[1:3], "big")
                    width = int.from_bytes(payload[3:5], "big")
                    return (width, height)
                handle.seek(length - 2, 1)
    except OSError:
        return None


def parse_timestamp(text: str | None) -> datetime | None:
    if not isinstance(text, str) or not text.strip():
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Step 1A input
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Step1AInput:
    artifact_path: Path
    artifact_sha256: str
    manifest_path: Path | None
    step1a_code_commit: str
    batch_dirs: dict[str, str]
    candidates: tuple[dict[str, Any], ...]
    canvas_by_camera: dict[str, tuple[int, int]]

    @property
    def candidate_by_id(self) -> dict[str, dict[str, Any]]:
        return {c["episode_candidate_id"]: c for c in self.candidates}

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def member_card_count(self) -> int:
        return sum(int(c["member_count"]) for c in self.candidates)

    def member_card_ids(self, candidate_id: str) -> list[str]:
        row = self.candidate_by_id.get(candidate_id)
        if row is None:
            raise KeyError(f"unknown candidate {candidate_id}")
        return list(row["member_review_card_ids"])

    def card_index(self) -> dict[str, tuple[str, dict[str, Any]]]:
        """card_id -> (candidate_id, member lineage row)."""
        index: dict[str, tuple[str, dict[str, Any]]] = {}
        for candidate in self.candidates:
            cid = candidate["episode_candidate_id"]
            for member in candidate["lineage"]["review_cards"]:
                index[member["card_id"]] = (cid, member)
        return index


def load_step1a(
    artifact_path: Path | str,
    manifest_path: Path | str | None = None,
    *,
    repo_root: Path | str | None = None,
) -> Step1AInput:
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Step 1A artifact not found: {artifact}")
    digest = sha256_file(artifact)

    candidates: list[dict[str, Any]] = []
    with artifact.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                candidates.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {artifact}") from exc

    batch_dirs: dict[str, str] = {}
    commit = ""
    canvas: dict[str, tuple[int, int]] = {}
    resolved_manifest: Path | None = None
    if manifest_path is not None:
        resolved_manifest = Path(manifest_path)
        if resolved_manifest.is_file():
            payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
            commit = str(payload.get("git_commit") or "")
            batch_dirs = dict((payload.get("inputs") or {}).get("batches") or {})
            for batch in (payload.get("input_manifest") or {}).get("batches") or []:
                batch_dirs.setdefault(str(batch["batch_key"]), str(batch["directory"]))

    if repo_root is not None:
        root = Path(repo_root)
        batch_dirs = {
            key: (value if Path(value).is_absolute() else str(root / value))
            for key, value in batch_dirs.items()
        }

    seen: set[str] = set()
    for candidate in candidates:
        cid = candidate.get("episode_candidate_id")
        if not cid:
            raise ValueError("candidate without episode_candidate_id")
        if cid in seen:
            raise ValueError(f"duplicate episode_candidate_id in Step 1A artifact: {cid}")
        seen.add(cid)
        camera = str(candidate.get("camera_id") or "")
        canvas.setdefault(camera, (2560, 1440))

    return Step1AInput(
        artifact_path=artifact,
        artifact_sha256=digest,
        manifest_path=resolved_manifest,
        step1a_code_commit=commit,
        batch_dirs=batch_dirs,
        candidates=tuple(candidates),
        canvas_by_camera=canvas,
    )


# --------------------------------------------------------------------------- #
# risk classification and queueing
# --------------------------------------------------------------------------- #


def _member_assets(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list((candidate.get("lineage") or {}).get("review_cards") or [])


def _bbox_center(bbox: Sequence[float] | None) -> tuple[float, float] | None:
    if not bbox or len(bbox) != 4:
        return None
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _center_distance(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    left, right = _bbox_center(a), _bbox_center(b)
    if left is None or right is None:
        return None
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


#: Mirrors the Step 1A spatial rule so "would these have merged?" stays consistent.
_MERGE_LIMIT_RATIO = 0.75
_MERGE_LIMIT_FLOOR_PX = 24.0


def _diagonal(bbox: Sequence[float] | None) -> float | None:
    if not bbox or len(bbox) != 4:
        return None
    return ((float(bbox[2]) - float(bbox[0])) ** 2
            + (float(bbox[3]) - float(bbox[1])) ** 2) ** 0.5


def _merge_plausible(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """True when Step 1A's own spatial rule would have considered merging these."""
    distance = _center_distance(a.get("bbox"), b.get("bbox"))
    if distance is None:
        return False
    left, right = _diagonal(a.get("bbox")), _diagonal(b.get("bbox"))
    if left is None or right is None:
        return False
    limit = max(_MERGE_LIMIT_FLOOR_PX, _MERGE_LIMIT_RATIO * (left + right) / 2.0)
    return distance <= limit


def classify_risk(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Split Step 1A ambiguity into GROUPING_RISK vs LINEAGE_ONLY.

    A missing ``file_id`` says nothing about whether the grouping is wrong, so it
    must not be queued as the highest human risk.  Equally, a BOX_WRONG card is
    only a *grouping* risk when it actually participated in a multi-card group;
    as a singleton its issue is localization, not episode identity.
    """
    reasons = [str(r) for r in candidate.get("ambiguity_reasons") or []]
    grouping: list[str] = list(reasons)
    lineage: list[str] = []
    for reason in reasons:
        if reason in LINEAGE_ONLY_REASONS:
            lineage.append(reason)
            grouping.remove(reason)

    members = _member_assets(candidate)
    member_count = int(candidate.get("member_count") or 0)
    if member_count > 1:
        seen_times: dict[str, int] = {}
        for member in members:
            key = str(member.get("timestamp") or member.get("frame_id") or "")
            if key:
                seen_times[key] = seen_times.get(key, 0) + 1
        if any(count > 1 for count in seen_times.values()):
            grouping.append("same_frame_merge_risk")
        if any(str(m.get("label")) == "BOX_WRONG" for m in members):
            grouping.append("box_wrong_participates_in_grouping")

    grouping = sorted(set(grouping))
    lineage = sorted(set(lineage))
    return {
        "grouping_risk": bool(grouping),
        "lineage_only": bool(lineage) and not grouping,
        "grouping_risk_reasons": grouping,
        "lineage_reasons": lineage,
        "ambiguity_reasons": reasons,
    }


def _queue_priority(candidate: Mapping[str, Any], risk: Mapping[str, Any]) -> int:
    """1 GROUPING_RISK, 2 multi-card, 3 BOX_WRONG, 4 other positive, 5 LINEAGE_ONLY."""
    members = _member_assets(candidate)
    if risk["grouping_risk"]:
        return 1
    if int(candidate.get("member_count") or 0) > 1:
        return 2
    if any(str(m.get("label")) == "BOX_WRONG" for m in members):
        return 3
    if risk["lineage_only"]:
        return 5
    return 4


def _candidate_sort_key(candidate: Mapping[str, Any], risk: Mapping[str, Any]) -> tuple:
    return (
        _queue_priority(candidate, risk),
        str(candidate.get("camera_id") or ""),
        str(candidate.get("scene_version") or UNKNOWN_SCENE_VERSION),
        str(candidate.get("start_timestamp") or ""),
        str(candidate.get("episode_candidate_id") or ""),
    )


def build_queue(step1a: Step1AInput) -> list[dict[str, Any]]:
    """Ordered review queue with risk classification, independent of processing order."""
    rows: list[dict[str, Any]] = []
    # same-frame neighbours across candidates are a real cross-candidate merge risk
    frame_owners: dict[tuple[str, str], list[str]] = {}
    for candidate in step1a.candidates:
        camera = str(candidate.get("camera_id") or "")
        for member in _member_assets(candidate):
            key = (camera, str(member.get("timestamp") or member.get("frame_id") or ""))
            frame_owners.setdefault(key, []).append(str(candidate["episode_candidate_id"]))

    for candidate in step1a.candidates:
        risk = classify_risk(candidate)
        camera = str(candidate.get("camera_id") or "")
        neighbours: set[str] = set()
        for member in _member_assets(candidate):
            key = (camera, str(member.get("timestamp") or member.get("frame_id") or ""))
            for other_id in frame_owners.get(key, []):
                if other_id == candidate["episode_candidate_id"]:
                    continue
                other = step1a.candidate_by_id.get(other_id)
                if other is None:
                    continue
                # Same frame alone is common; only a *merge-plausible* pair is a
                # real risk of the conservative Step 1A split.
                if any(_merge_plausible(member, peer) for peer in _member_assets(other)):
                    neighbours.add(other_id)
        if neighbours:
            risk["grouping_risk_reasons"] = sorted(
                set(risk["grouping_risk_reasons"]) | {"same_frame_neighbour_candidate"})
            risk["grouping_risk"] = True
            risk["lineage_only"] = False
        rows.append({
            "episode_candidate_id": candidate["episode_candidate_id"],
            "camera_id": camera,
            "scene_version": candidate.get("scene_version") or UNKNOWN_SCENE_VERSION,
            "start_timestamp": candidate.get("start_timestamp"),
            "end_timestamp": candidate.get("end_timestamp"),
            "member_count": int(candidate.get("member_count") or 0),
            "original_labels_summary": candidate.get("original_labels_summary"),
            "queue_priority": _queue_priority(candidate, risk),
            "same_frame_neighbour_candidate_ids": sorted(neighbours),
            **risk,
        })
    rows.sort(key=lambda row: _candidate_sort_key(
        step1a.candidate_by_id[row["episode_candidate_id"]], row))
    return rows


# --------------------------------------------------------------------------- #
# merge suggestions
# --------------------------------------------------------------------------- #


def _scene_compatible(left: str | None, right: str | None) -> bool:
    if left in (None, "", UNKNOWN_SCENE_VERSION) or right in (None, "", UNKNOWN_SCENE_VERSION):
        return True
    return left == right


def merge_suggestions(
    step1a: Step1AInput, candidate_id: str, *, limit: int = 5,
) -> list[dict[str, Any]]:
    """Top-N neighbouring candidates that might be the same physical episode.

    Suggestion only: it is never applied automatically.
    """
    source = step1a.candidate_by_id.get(candidate_id)
    if source is None:
        raise KeyError(f"unknown candidate {candidate_id}")
    source_start = parse_timestamp(source.get("start_timestamp"))
    source_end = parse_timestamp(source.get("end_timestamp"))
    source_members = _member_assets(source)
    scene = source.get("scene_version")

    scored: list[tuple[float, dict[str, Any]]] = []
    for other in step1a.candidates:
        if other["episode_candidate_id"] == candidate_id:
            continue
        if str(other.get("camera_id")) != str(source.get("camera_id")):
            continue
        if not _scene_compatible(scene, other.get("scene_version")):
            continue
        other_start = parse_timestamp(other.get("start_timestamp"))
        other_end = parse_timestamp(other.get("end_timestamp"))
        if source_end and other_start and other_start > source_end:
            time_gap = (other_start - source_end).total_seconds()
        elif other_end and source_start and other_end < source_start:
            time_gap = (source_start - other_end).total_seconds()
        else:
            time_gap = 0.0
        best_distance: float | None = None
        for left in source_members:
            for right in _member_assets(other):
                distance = _center_distance(left.get("bbox"), right.get("bbox"))
                if distance is not None and (best_distance is None or distance < best_distance):
                    best_distance = distance
        reasons: list[str] = []
        if time_gap <= 600.0:
            reasons.append("time_adjacent")
        if best_distance is not None and best_distance <= 96.0:
            reasons.append("spatially_adjacent")
        if any("false_split" in r or "near_threshold" in r
               for r in other.get("ambiguity_reasons") or []):
            reasons.append("step1a_split_risk_flag")
        if not reasons:
            continue
        score = time_gap + (best_distance if best_distance is not None else 1e6)
        scored.append((score, {
            "candidate_id": other["episode_candidate_id"],
            "camera_id": other.get("camera_id"),
            "scene_version": other.get("scene_version") or UNKNOWN_SCENE_VERSION,
            "start_timestamp": other.get("start_timestamp"),
            "end_timestamp": other.get("end_timestamp"),
            "member_count": int(other.get("member_count") or 0),
            "representative_card_id": other.get("representative_card_id"),
            "representative_frame": other.get("representative_frame"),
            "representative_bbox": other.get("representative_bbox"),
            "time_gap_seconds": round(time_gap, 3),
            "min_center_distance_px": None if best_distance is None else round(best_distance, 3),
            "risk_reason": ";".join(reasons),
        }))
    scored.sort(key=lambda item: (item[0], item[1]["candidate_id"]))
    return [row for _, row in scored[:limit]]


# --------------------------------------------------------------------------- #
# review state
# --------------------------------------------------------------------------- #


class ReviewError(ValueError):
    """Invalid human decision; the caller must surface it, never coerce it."""


class NearDuplicateTarget(ReviewError):
    """The clicked point is very close to an existing manual target.

    Raised so the reviewer is *asked* before a possibly-duplicate target is
    created.  It is never fatal: re-submitting with ``allow_near_duplicate=True``
    succeeds, because two real pieces of litter can genuinely sit close together.
    """

    def __init__(self, near_duplicates: Sequence[Mapping[str, Any]]) -> None:
        self.near_duplicates = [dict(row) for row in near_duplicates]
        super().__init__(
            f"{len(self.near_duplicates)} existing manual target(s) are very close")


def make_confirm_episode(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "draft_id": "e1",
        "member_card_ids": list(candidate["member_review_card_ids"]),
        "truth_class": "REQUIRED_LITTER",
        "localization_status": "OK",
    }


def validate_episode_partition(
    candidate: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """A split must cover every member exactly once; nothing may be dropped."""
    expected = list(candidate["member_review_card_ids"])
    expected_set = set(expected)
    if not episodes:
        raise ReviewError("split requires at least one episode assignment")
    seen: list[str] = []
    cleaned: list[dict[str, Any]] = []
    for index, raw in enumerate(episodes, 1):
        members = list(raw.get("member_card_ids") or [])
        if not members:
            raise ReviewError(f"episode {index} has no members")
        truth_class = str(raw.get("truth_class") or "REQUIRED_LITTER")
        if truth_class not in TRUTH_CLASSES:
            raise ReviewError(f"episode {index} has invalid truth_class {truth_class!r}")
        localization = str(raw.get("localization_status") or "OK")
        if localization not in LOCALIZATION_STATUSES:
            raise ReviewError(f"episode {index} has invalid localization_status")
        if truth_class == "REQUIRED_LITTER" and localization == "NEEDS_RELOCALIZATION":
            pass  # allowed: litter confirmed but box untrusted
        for member in members:
            if member not in expected_set:
                raise ReviewError(f"member {member} does not belong to this candidate")
            if member in seen:
                raise ReviewError(f"member {member} assigned to more than one episode")
            seen.append(member)
        cleaned.append({
            "draft_id": str(raw.get("draft_id") or f"e{index}"),
            "member_card_ids": members,
            "truth_class": truth_class,
            "localization_status": localization,
        })
    missing = [m for m in expected if m not in set(seen)]
    if missing:
        raise ReviewError(f"members left unassigned: {missing}")
    if len(cleaned) < 2:
        raise ReviewError("SPLIT requires at least two episode assignments")
    return cleaned


def validate_merge(
    step1a: Step1AInput, state: "ReviewState", candidate_id: str,
    targets: Sequence[str],
) -> list[str]:
    source = step1a.candidate_by_id.get(candidate_id)
    if source is None:
        raise ReviewError(f"unknown candidate {candidate_id}")
    if not targets:
        raise ReviewError("MERGE requires at least one target candidate")
    unique = sorted(set(targets))
    if candidate_id in unique:
        raise ReviewError("cannot merge a candidate with itself")
    for target in unique:
        row = step1a.candidate_by_id.get(target)
        if row is None:
            raise ReviewError(f"unknown merge target {target}")
        if str(row.get("camera_id")) != str(source.get("camera_id")):
            raise ReviewError(f"cannot merge across cameras: {candidate_id} / {target}")
        if not _scene_compatible(source.get("scene_version"), row.get("scene_version")):
            raise ReviewError(f"cannot merge across scene_version: {candidate_id} / {target}")
        for side, cid in (("source", candidate_id), ("target", target)):
            decision = (state.get(cid) or {}).get("decision")
            if decision in NON_EPISODE_DECISIONS:
                raise ReviewError(f"{side} {cid} is already decided {decision}")
            if decision == "SPLIT":
                raise ReviewError(f"{side} {cid} was split and cannot be merged")
    return unique


def validate_manual_point(
    *,
    x: float,
    y: float,
    image_width: int,
    image_height: int,
    clicked_asset_type: str,
    clicked_asset_path: str,
    member: Mapping[str, Any],
    batch_key: str,
) -> dict[str, Any]:
    """Validate a human click and return authoritative image-native coordinates.

    The browser may render the asset at any CSS size, so only *image-native* pixels
    are accepted, and the server recomputes the normalised value rather than
    trusting a client-supplied one.
    """
    if clicked_asset_type not in MANUAL_TARGET_ASSET_TYPES:
        raise ReviewError(
            f"clicked_asset_type must be one of {MANUAL_TARGET_ASSET_TYPES}")
    try:
        width = int(image_width)
        height = int(image_height)
    except (TypeError, ValueError):
        raise ReviewError("clicked image dimensions must be integers") from None
    if width <= 0 or height <= 0:
        raise ReviewError("clicked image dimensions must be positive")
    try:
        px = float(x)
        py = float(y)
    except (TypeError, ValueError):
        raise ReviewError("point x/y must be numeric") from None
    if not (0.0 <= px <= float(width)) or not (0.0 <= py <= float(height)):
        raise ReviewError(
            f"point ({px}, {py}) is outside the clicked image ({width}x{height})")

    # Both click targets resolve to the same underlying asset: the schema has no
    # separate current_image, and "current" is the context crop of the source frame.
    assets = dict(member.get("assets") or {})
    expected = assets.get("context_image")
    if not expected or str(clicked_asset_path) != str(expected):
        raise ReviewError(
            "clicked asset does not belong to the source member card")
    x_norm = px / float(width)
    y_norm = py / float(height)
    return {
        "x": round(px, 3),
        "y": round(py, 3),
        "x_norm": round(x_norm, 8),
        "y_norm": round(y_norm, 8),
        "clicked_image_width": width,
        "clicked_image_height": height,
        "clicked_asset_type": clicked_asset_type,
        "clicked_asset_path": str(clicked_asset_path),
        "clicked_asset_derived_from": (
            "context_image" if clicked_asset_type == "current" else None),
        "batch_key": batch_key,
    }


def _manual_points_close(
    left: Mapping[str, Any], right: Mapping[str, Any],
    threshold_px: float = MANUAL_TARGET_DUPLICATE_PX,
) -> bool:
    """Whether two clicks are effectively the same spot.

    Only meaningful when both clicks used the *same* asset: the assets are crops
    of the source frame, so their normalised coordinates live in different spaces.
    Returns False for different assets rather than inventing a comparison.
    """
    left_point = dict(left.get("point") or {})
    right_point = dict(right.get("point") or {})
    if not left_point or not right_point:
        return False
    left_card = left.get("source_member_card_id")
    right_card = right.get("source_member_card_id")
    if left_card and right_card and left_card != right_card:
        return False
    if left_point.get("clicked_asset_path") != right_point.get("clicked_asset_path"):
        return False
    try:
        lx, ly = float(left_point["x"]), float(left_point["y"])
        rx, ry = float(right_point["x"]), float(right_point["y"])
        width = float(left_point.get("clicked_image_width") or 0)
        height = float(left_point.get("clicked_image_height") or 0)
    except (KeyError, TypeError, ValueError):
        return False
    if width <= 0 or height <= 0:
        return False
    limit = max(float(threshold_px), MANUAL_TARGET_DUPLICATE_FRACTION * max(width, height))
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5 < limit


@dataclass
class ReviewState:
    """Crash-safe review state with an append-only audit trail.

    Every mutation is written immediately; a reload returns identical state.
    """

    path: Path
    schema_version: str = REVIEW_SCHEMA_VERSION
    input_artifact: str = ""
    input_sha256: str = ""
    candidate_count: int = 0
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    #: manual_target_id -> human-discovered missing target (schema v2)
    manual_targets: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str, *, step1a: Step1AInput | None = None) -> "ReviewState":
        target = Path(path)
        if not target.is_file():
            state = cls(path=target)
            if step1a is not None:
                state.input_artifact = str(step1a.artifact_path)
                state.input_sha256 = step1a.artifact_sha256
                state.candidate_count = step1a.candidate_count
            return state
        payload = json.loads(target.read_text(encoding="utf-8"))
        state = cls(
            path=target,
            # the version this file was actually authored with, for diagnostics
            schema_version=str(payload.get("review_schema_version") or REVIEW_SCHEMA_VERSION),
            input_artifact=str((payload.get("input") or {}).get("artifact") or ""),
            input_sha256=str((payload.get("input") or {}).get("sha256") or ""),
            candidate_count=int((payload.get("input") or {}).get("candidate_count") or 0),
            candidates=dict(payload.get("candidates") or {}),
            audit_trail=list(payload.get("audit_trail") or []),
            # v1 state has no manual_targets key; tolerate its absence rather than
            # resetting anything a reviewer already decided.
            manual_targets=dict(payload.get("manual_targets") or {}),
        )
        if step1a is not None and state.input_sha256 and \
                state.input_sha256 != step1a.artifact_sha256:
            raise ReviewError(
                "review state belongs to a different Step 1A artifact "
                f"({state.input_sha256[:12]} != {step1a.artifact_sha256[:12]})")
        return state

    def get(self, candidate_id: str, default: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return self.candidates.get(candidate_id, default)

    def status_of(self, candidate_id: str) -> str:
        row = self.candidates.get(candidate_id)
        if not row:
            return "pending"
        return str(row.get("status") or "pending")

    def reviewed_ids(self) -> list[str]:
        return sorted(cid for cid, row in self.candidates.items()
                      if row.get("status") == "reviewed")

    def _record(self, candidate_id: str, action: str, payload: Mapping[str, Any]) -> None:
        self.audit_trail.append({
            "at": utc_now_text(),
            "candidate_id": candidate_id,
            "action": action,
            "payload": dict(payload),
        })

    def save(self) -> None:
        # Always write the current schema: a loaded v1 file is migrated in place on
        # the next write, and the version it came from is preserved for audit.
        atomic_write_json(self.path, {
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "loaded_schema_version": self.schema_version,
            "input": {
                "artifact": self.input_artifact,
                "sha256": self.input_sha256,
                "candidate_count": self.candidate_count,
            },
            "candidates": self.candidates,
            "manual_targets": self.manual_targets,
            "audit_trail": self.audit_trail,
        })

    # -- human-discovered missing targets ----------------------------------- #

    def manual_targets_for(self, candidate_id: str) -> list[dict[str, Any]]:
        rows = [row for row in self.manual_targets.values()
                if row.get("source_episode_candidate_id") == candidate_id]
        rows.sort(key=lambda row: (str(row.get("created_at") or ""),
                                   str(row.get("manual_target_id") or "")))
        return rows

    def near_duplicate_manual_targets(
        self, step1a: Step1AInput, candidate_id: str, point: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            {"manual_target_id": row["manual_target_id"],
             "truth_class": row.get("truth_class"),
             "point": dict(row.get("point") or {})}
            for row in self.manual_targets_for(candidate_id)
            if _manual_points_close(
                row,
                {"point": point, "source_member_card_id": point.get("source_member_card_id")})
        ]

    def add_manual_target(
        self, step1a: Step1AInput, candidate_id: str, *,
        card_id: str,
        truth_class: str,
        clicked_asset_type: str,
        clicked_asset_path: str,
        x: float,
        y: float,
        image_width: int,
        image_height: int,
        note: str = "",
        allow_near_duplicate: bool = False,
        manual_target_id: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Record one reviewer-clicked target without ever expanding it into a bbox.

        Returns the persisted record plus any proximity warnings.  The candidate's
        own review decision is deliberately untouched: ADD_MISSING_TARGET and
        CONFIRM are orthogonal operations.
        """
        candidate = step1a.candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ReviewError(f"unknown candidate {candidate_id}")
        if truth_class not in MANUAL_TARGET_TRUTH_CLASSES:
            raise ReviewError(
                f"truth_class must be one of {MANUAL_TARGET_TRUTH_CLASSES}")
        members = {m.get("card_id"): m
                   for m in (candidate.get("lineage") or {}).get("review_cards") or []}
        member = members.get(card_id)
        if member is None:
            raise ReviewError(f"card {card_id} is not a member of {candidate_id}")
        batch_key = str(member.get("batch_key") or "")
        point = validate_manual_point(
            x=x, y=y, image_width=image_width, image_height=image_height,
            clicked_asset_type=clicked_asset_type, clicked_asset_path=clicked_asset_path,
            member=member, batch_key=batch_key)

        warnings = self.near_duplicate_manual_targets(
            step1a, candidate_id,
            dict(point, source_member_card_id=card_id))
        if warnings and not allow_near_duplicate:
            raise NearDuplicateTarget(warnings)

        target_id = manual_target_id or manual_target_id_for(
            str(candidate.get("camera_id") or ""))
        if target_id in self.manual_targets:
            raise ReviewError(f"manual target {target_id} already exists")
        now = utc_now_text()
        record = {
            "manual_target_id": target_id,
            "truth_target_id": target_id,
            "source_episode_candidate_id": candidate_id,
            "source_member_card_id": card_id,
            "camera_id": candidate.get("camera_id"),
            "scene_version": UNKNOWN_SCENE_VERSION,
            "timestamp": member.get("timestamp"),
            "truth_class": truth_class,
            "location_type": "POINT",
            "point": point,
            "localization_status": "NEEDS_RELOCALIZATION",
            "origin": MANUAL_TARGET_ORIGIN,
            "appearance_tag": "",
            "review_status": "human_reviewed",
            "note": note,
            "created_at": now,
            "updated_at": now,
            "revision": 1,
            "review_schema_version": REVIEW_SCHEMA_VERSION,
        }
        self.manual_targets[target_id] = record
        self._record(candidate_id, "manual_target:add", {
            "manual_target_id": target_id, "card_id": card_id,
            "truth_class": truth_class, "point": point,
            "clicked_asset_type": clicked_asset_type,
            "forced_past_near_duplicate": bool(warnings)})
        self.save()
        return record, warnings

    def update_manual_target(
        self, manual_target_id: str, *,
        truth_class: str | None = None, note: str | None = None,
    ) -> dict[str, Any]:
        """Change the classification only; the id is never regenerated."""
        row = self.manual_targets.get(manual_target_id)
        if row is None:
            raise ReviewError(f"unknown manual target {manual_target_id}")
        if truth_class is not None:
            if truth_class not in MANUAL_TARGET_TRUTH_CLASSES:
                raise ReviewError(
                    f"truth_class must be one of {MANUAL_TARGET_TRUTH_CLASSES}")
            row["truth_class"] = truth_class
        if note is not None:
            row["note"] = note
        row["updated_at"] = utc_now_text()
        row["revision"] = int(row.get("revision") or 1) + 1
        self._record(str(row.get("source_episode_candidate_id") or ""),
                     "manual_target:update", {
                         "manual_target_id": manual_target_id,
                         "truth_class": row.get("truth_class"), "note": row.get("note")})
        self.save()
        return row

    def repoint_manual_target(
        self, step1a: Step1AInput, manual_target_id: str, *,
        clicked_asset_type: str, clicked_asset_path: str,
        x: float, y: float, image_width: int, image_height: int,
    ) -> dict[str, Any]:
        """Move an existing target to a newly clicked point, keeping its id."""
        row = self.manual_targets.get(manual_target_id)
        if row is None:
            raise ReviewError(f"unknown manual target {manual_target_id}")
        candidate_id = str(row.get("source_episode_candidate_id") or "")
        candidate = step1a.candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ReviewError(f"unknown candidate {candidate_id}")
        members = {m.get("card_id"): m
                   for m in (candidate.get("lineage") or {}).get("review_cards") or []}
        member = members.get(str(row.get("source_member_card_id") or ""))
        if member is None:
            raise ReviewError("source member card is missing from the candidate")
        point = validate_manual_point(
            x=x, y=y, image_width=image_width, image_height=image_height,
            clicked_asset_type=clicked_asset_type, clicked_asset_path=clicked_asset_path,
            member=member, batch_key=str(member.get("batch_key") or ""))
        previous = dict(row.get("point") or {})
        row["point"] = point
        row["updated_at"] = utc_now_text()
        row["revision"] = int(row.get("revision") or 1) + 1
        self._record(candidate_id, "manual_target:repoint", {
            "manual_target_id": manual_target_id,
            "previous_point": previous, "point": point})
        self.save()
        return row

    def delete_manual_target(self, manual_target_id: str) -> dict[str, Any]:
        """Delete a target; the audit trail keeps the full history."""
        row = self.manual_targets.pop(manual_target_id, None)
        if row is None:
            raise ReviewError(f"unknown manual target {manual_target_id}")
        self._record(str(row.get("source_episode_candidate_id") or ""),
                     "manual_target:delete", {
                         "manual_target_id": manual_target_id,
                         "truth_class": row.get("truth_class"),
                         "point": row.get("point")})
        self.save()
        return row

    def manual_target_counts(self) -> dict[str, Any]:
        rows = list(self.manual_targets.values())
        by_class: dict[str, int] = {c: 0 for c in MANUAL_TARGET_TRUTH_CLASSES}
        by_camera: dict[str, int] = {}
        for row in rows:
            key = str(row.get("truth_class") or "UNKNOWN")
            by_class[key] = by_class.get(key, 0) + 1
            camera = str(row.get("camera_id") or "")
            by_camera[camera] = by_camera.get(camera, 0) + 1
        return {
            "manual_missing_target_count": len(rows),
            "manual_required_count": by_class.get("REQUIRED_LITTER", 0),
            "manual_ignore_small_count": by_class.get("IGNORE_SMALL", 0),
            "manual_uncertain_count": by_class.get("UNCERTAIN", 0),
            "manual_targets_by_camera": dict(sorted(by_camera.items())),
            "manual_targets_needing_relocalization": sum(
                1 for row in rows
                if row.get("localization_status") == "NEEDS_RELOCALIZATION"),
        }

    # -- mutations ---------------------------------------------------------- #

    def decide(
        self, step1a: Step1AInput, candidate_id: str, decision: str,
        *,
        note: str = "",
        episodes: Sequence[Mapping[str, Any]] | None = None,
        merge_targets: Sequence[str] | None = None,
        truth_class: str | None = None,
        localization_status: str = "OK",
    ) -> dict[str, Any]:
        if candidate_id not in step1a.candidate_by_id:
            raise ReviewError(f"unknown candidate {candidate_id}")
        if decision not in DECISIONS:
            raise ReviewError(f"unknown decision {decision!r}")
        candidate = step1a.candidate_by_id[candidate_id]

        if decision == "CONFIRM":
            payload_episodes = [make_confirm_episode(candidate)]
            payload_merge: list[str] = []
            payload_truth = "REQUIRED_LITTER"
        elif decision == "SPLIT":
            payload_episodes = validate_episode_partition(candidate, episodes or [])
            payload_merge = []
            payload_truth = None
        elif decision == "MERGE":
            payload_merge = validate_merge(step1a, self, candidate_id, merge_targets or [])
            payload_episodes = [make_confirm_episode(candidate)]
            payload_truth = "REQUIRED_LITTER"
        else:
            payload_episodes = []
            payload_merge = []
            payload_truth = decision
            if decision != truth_class and truth_class is not None:
                raise ReviewError(f"truth_class {truth_class!r} does not match {decision}")

        if localization_status not in LOCALIZATION_STATUSES:
            raise ReviewError(f"invalid localization_status {localization_status!r}")
        if payload_truth == "REQUIRED_LITTER" and localization_status != "OK":
            raise ReviewError("REQUIRED_LITTER confirm uses localization_status OK or "
                              "per-episode NEEDS_RELOCALIZATION")

        entry = {
            "status": "reviewed",
            "decision": decision,
            "truth_class": payload_truth,
            "localization_status": localization_status if payload_truth else None,
            "episodes": payload_episodes,
            "merge_targets": payload_merge,
            "note": note,
            "decided_at": utc_now_text(),
        }
        previous = self.candidates.get(candidate_id)
        if previous:
            entry["revision"] = int(previous.get("revision") or 1) + 1
        else:
            entry["revision"] = 1
        self.candidates[candidate_id] = entry
        self._record(candidate_id, f"decide:{decision}", {
            "note": note, "episodes": payload_episodes, "merge_targets": payload_merge})
        self.save()
        return entry

    def skip(self, candidate_id: str, *, note: str = "") -> dict[str, Any]:
        entry = {
            "status": "skipped",
            "decision": None,
            "truth_class": None,
            "localization_status": None,
            "episodes": [],
            "merge_targets": [],
            "note": note,
            "decided_at": utc_now_text(),
            "revision": int((self.candidates.get(candidate_id) or {}).get("revision") or 0) + 1,
        }
        self.candidates[candidate_id] = entry
        self._record(candidate_id, "skip", {"note": note})
        self.save()
        return entry

    def reset(self, candidate_id: str) -> None:
        self.candidates.pop(candidate_id, None)
        self._record(candidate_id, "reset", {})
        self.save()

    def progress(self, step1a: Step1AInput) -> dict[str, int]:
        reviewed = sum(1 for c in step1a.candidates
                       if self.status_of(c["episode_candidate_id"]) == "reviewed")
        skipped = sum(1 for c in step1a.candidates
                      if self.status_of(c["episode_candidate_id"]) == "skipped")
        return {
            "total": step1a.candidate_count,
            "reviewed": reviewed,
            "skipped": skipped,
            "pending": step1a.candidate_count - reviewed - skipped,
        }


# --------------------------------------------------------------------------- #
# episode resolution
# --------------------------------------------------------------------------- #


def episode_id_for(camera_id: str, member_card_ids: Iterable[str]) -> str:
    """Deterministic id derived only from the final member card set."""
    members = sorted(str(m) for m in member_card_ids)
    blob = "|".join([EPISODE_ID_NAMESPACE, str(camera_id), *members])
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    return f"ge-{camera_id}-{digest}"


def manual_target_id_for(camera_id: str, token: str | None = None) -> str:
    """Persistent identity for a human-discovered target.

    Two genuinely different pieces of litter can sit on the same pixel, so a
    content hash cannot identify them: the id must be minted once and persisted.
    """
    suffix = token or uuid.uuid4().hex[:12]
    return f"mt-{camera_id}-{suffix}"


def manual_episode_id_for(camera_id: str, manual_target_id: str) -> str:
    suffix = str(manual_target_id).rsplit("-", 1)[-1]
    return f"ge-{camera_id}-manual-{suffix}"



class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            # deterministic: smaller id wins as root
            if b < a:
                a, b = b, a
            self.parent[b] = a


def resolve_episodes(
    step1a: Step1AInput, state: ReviewState,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn review decisions into human-confirmed episodes plus conflict records.

    A MERGE absorbs its target candidates even when those targets have not been
    reviewed on their own: the human saw the neighbour and declared it the same
    physical episode, so the neighbour's members must join that episode.
    """
    episodes: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    merge_edges: list[tuple[str, str]] = []
    absorbed: set[str] = set()
    producers: set[str] = set()
    for candidate_id, row in sorted(state.candidates.items()):
        decision = row.get("decision")
        if decision in EPISODE_PRODUCING_DECISIONS:
            producers.add(candidate_id)
        if decision != "MERGE":
            continue
        for target in row.get("merge_targets") or []:
            if target not in step1a.candidate_by_id:
                conflicts.append({"type": "merge_target_missing",
                                  "candidate_id": candidate_id, "target": target})
                continue
            merge_edges.append((candidate_id, target))
            absorbed.add(target)

    union = _UnionFind(step1a.candidate_by_id)
    for left, right in merge_edges:
        union.union(left, right)

    involved = sorted(producers | absorbed)
    components: dict[str, list[str]] = {}
    for candidate_id in involved:
        components.setdefault(union.find(candidate_id), []).append(candidate_id)

    for root in sorted(components):
        members_of = sorted(components[root])
        decisions = {state.get(cid, {}).get("decision") for cid in members_of}

        conflicted = sorted(cid for cid in members_of
                            if state.get(cid, {}).get("decision") in NON_EPISODE_DECISIONS)
        if conflicted:
            conflicts.append({"type": "merge_component_contains_non_episode_decision",
                              "candidates": members_of, "conflicting": conflicted})
            continue
        if "SPLIT" in decisions and len(members_of) > 1:
            conflicts.append({"type": "merge_component_contains_split",
                              "candidates": members_of})
            continue

        if decisions == {"SPLIT"}:
            candidate = step1a.candidate_by_id[members_of[0]]
            for draft in state.get(members_of[0], {}).get("episodes") or []:
                episodes.append({
                    "camera_id": candidate["camera_id"],
                    "member_card_ids": sorted(draft["member_card_ids"]),
                    "truth_class": draft["truth_class"],
                    "localization_status": draft["localization_status"],
                    "source_episode_candidate_ids": members_of,
                    "grouping_review": {"confirmed": True, "split": True,
                                        "merged": False, "merge_source_candidates": []},
                    "decision": "SPLIT",
                })
            continue

        member_ids: list[str] = []
        for cid in members_of:
            member_ids.extend(step1a.member_card_ids(cid))
        localization = "OK"
        for cid in members_of:
            if (state.get(cid, {}).get("localization_status") or "OK") != "OK":
                localization = "NEEDS_RELOCALIZATION"
        camera = step1a.candidate_by_id[members_of[0]]["camera_id"]
        merged = len(members_of) > 1
        episodes.append({
            "camera_id": camera,
            "member_card_ids": sorted(member_ids),
            "truth_class": "REQUIRED_LITTER",
            "localization_status": localization,
            "source_episode_candidate_ids": members_of,
            "grouping_review": {
                "confirmed": True,
                "split": False,
                "merged": merged,
                "merge_source_candidates": members_of if merged else [],
            },
            "decision": "MERGE" if merged else "CONFIRM",
        })

    seen_members: dict[str, str] = {}
    for episode in episodes:
        for member in episode["member_card_ids"]:
            if member in seen_members:
                conflicts.append({"type": "member_in_multiple_episodes",
                                  "member": member,
                                  "episode_a": seen_members[member],
                                  "episode_b": episode["camera_id"]})
            seen_members[member] = episode["camera_id"]
    for episode in episodes:
        episode["episode_id"] = episode_id_for(episode["camera_id"], episode["member_card_ids"])
    ids = [e["episode_id"] for e in episodes]
    if len(ids) != len(set(ids)):
        conflicts.append({"type": "episode_id_collision", "episode_ids": sorted(ids)})
    episodes.sort(key=lambda e: (e["camera_id"], min(e["member_card_ids"]), e["episode_id"]))
    return episodes, conflicts


# --------------------------------------------------------------------------- #
# source-native trainability audit
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainabilityPolicy:
    retention_days: float | None = DEFAULT_RETENTION_DAYS
    reference_time: datetime | None = None
    require_assets_present: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "retention_days": self.retention_days,
            "reference_time": (self.reference_time or datetime.now()).strftime(
                "%Y-%m-%dT%H:%M:%S"),
            "require_assets_present": self.require_assets_present,
        }


def _resolve_asset(step1a: Step1AInput, batch_key: str, relative: str) -> Path | None:
    base = step1a.batch_dirs.get(batch_key)
    if not base:
        return None
    path = Path(base) / relative
    return path if path.is_file() else None


def audit_trainability(
    episode: Mapping[str, Any], step1a: Step1AInput, *,
    policy: TrainabilityPolicy | None = None,
    source_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether a confirmed episode can be traced to source-native pixels.

    Method A = a real source-resolution full-frame asset exists locally.
    Method B = a stable PS ``file_id`` plus a precise timestamp can be re-fetched
               and decoded.  Reachability is *not* verified here; it is reported.
    """
    policy = policy or TrainabilityPolicy()
    source_evidence = source_evidence or {}
    verified_ps = set(str(x) for x in source_evidence.get("verified_reachable_file_ids") or [])
    frame_assets = dict(source_evidence.get("source_frame_assets") or {})
    card_index = step1a.card_index()
    reference = policy.reference_time or datetime.now()

    per_member: list[dict[str, Any]] = []
    for card_id in episode["member_card_ids"]:
        entry = card_index.get(card_id)
        if entry is None:
            per_member.append({"card_id": card_id, "lineage_missing": True})
            continue
        _, member = entry
        assets = dict(member.get("assets") or {})
        asset_paths: dict[str, str] = {}
        asset_dims: dict[str, list[int] | None] = {}
        assets_present = True
        for slot in ("context_image", "crop_image", "before_image", "after_image"):
            relative = assets.get(slot)
            if not relative:
                asset_paths[slot] = ""
                asset_dims[slot] = None
                if slot in ("context_image", "crop_image"):
                    assets_present = False
                continue
            resolved = _resolve_asset(step1a, str(member.get("batch_key")), str(relative))
            asset_paths[slot] = str(resolved) if resolved else str(relative)
            if resolved is None:
                asset_dims[slot] = None
                if slot in ("context_image", "crop_image"):
                    assets_present = False
            else:
                dims = jpeg_dimensions(resolved)
                asset_dims[slot] = list(dims) if dims else None

        timestamp = parse_timestamp(member.get("timestamp"))
        file_id = member.get("source_file_id")
        full_frame = dict(frame_assets.get(card_id) or {})
        if full_frame:
            declared = full_frame.get("path")
            if not declared or not Path(str(declared)).is_file():
                # Never accept an unverifiable "source frame" claim.
                full_frame = {}
            else:
                dims = jpeg_dimensions(Path(str(declared)))
                full_frame["measured_dimensions"] = list(dims) if dims else None
        per_member.append({
            "card_id": card_id,
            "batch_key": member.get("batch_key"),
            "frame_id": member.get("frame_id"),
            "timestamp": member.get("timestamp"),
            "bbox": member.get("bbox"),
            "original_label": member.get("label"),
            "source_file_id": file_id,
            "assets_present": assets_present,
            "asset_paths": asset_paths,
            "asset_dimensions": asset_dims,
            "verified_full_frame_asset": full_frame or None,
            "ps_reachability_verified": bool(file_id) and str(file_id) in verified_ps,
            "file_id_timestamp_within_retention": (
                None if (timestamp is None or policy.retention_days is None)
                else (reference - timestamp) <= timedelta(days=policy.retention_days)),
        })

    def any_member(predicate) -> bool:
        return any(predicate(m) for m in per_member)

    method = ""
    reason = ""
    if any_member(lambda m: m.get("verified_full_frame_asset")):
        status = "TRAINABLE_SOURCE_NATIVE"
        method = "A_source_resolution_frame_asset"
    elif any_member(lambda m: m.get("source_file_id")
                    and m.get("file_id_timestamp_within_retention") is not False
                    and (m.get("assets_present") or not policy.require_assets_present)):
        status = "TRAINABLE_SOURCE_NATIVE"
        method = "B_ps_file_id_plus_timestamp"
        if not any_member(lambda m: m.get("ps_reachability_verified")):
            reason = "ps_reachability_unverified_requires_refetch"
    elif any_member(lambda m: m.get("source_file_id")
                    and m.get("file_id_timestamp_within_retention") is False):
        status = "REVIEW_ONLY"
        method = "none"
        reason = "recording_beyond_declared_retention_window"
    elif any_member(lambda m: not m.get("source_file_id") and not m.get("frame_id")
                    and m.get("assets_present")):
        # Derived crops only, and no frame/timestamp route back to the source.
        status = "REVIEW_ONLY"
        method = "none"
        reason = "only_derived_crop_or_context_no_source_route"
    elif any_member(lambda m: not m.get("source_file_id") and m.get("frame_id")
                    and m.get("timestamp")):
        status = "LINEAGE_UNRESOLVED"
        method = "none"
        reason = "source_file_id_missing_may_recover_via_frame_id_timestamp"
    elif any_member(lambda m: not m.get("assets_present")):
        status = "LINEAGE_UNRESOLVED"
        method = "none"
        reason = "referenced_asset_missing"
    else:
        status = "LINEAGE_UNRESOLVED"
        method = "none"
        reason = "insufficient_lineage_evidence"

    return {
        "trainability_status": status,
        "lineage_method": method,
        "lineage_reason": reason,
        "policy": policy.as_dict(),
        "members": per_member,
        "note": ("Audit only. No 640x640 source-native tile is produced in Step 1B; "
                 "a TRAINABLE_SOURCE_NATIVE result still requires the PS re-fetch to "
                 "succeed before the frame can actually be used."),
    }


def candidate_lineage_readiness(
    step1a: Step1AInput, *,
    policy: TrainabilityPolicy | None = None,
    source_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-candidate *input lineage* readiness, before any human decision.

    Explicitly NOT a Gold result: it only says what the trainability classification
    would be if the candidate were later confirmed as REQUIRED_LITTER.  It exists so
    Step 1C can plan around lineage gaps.
    """
    counts: dict[str, int] = {status: 0 for status in TRAINABILITY_STATUSES}
    reasons: dict[str, int] = {}
    for candidate in step1a.candidates:
        audit = audit_trainability(
            {"member_card_ids": list(candidate["member_review_card_ids"])},
            step1a, policy=policy, source_evidence=source_evidence)
        status = audit["trainability_status"]
        counts[status] = counts.get(status, 0) + 1
        reason = audit.get("lineage_reason") or audit.get("lineage_method") or "none"
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "hypothetical_if_confirmed_required_litter": True,
        "statement": ("Input-lineage readiness only. This is NOT Gold truth and does "
                      "not mean any candidate has been reviewed."),
        "counts": counts,
        "reason_histogram": dict(sorted(reasons.items())),
        "policy": (policy or TrainabilityPolicy()).as_dict(),
    }


# --------------------------------------------------------------------------- #
# Gold artifact construction
# --------------------------------------------------------------------------- #


def build_gold_records(
    step1a: Step1AInput, state: ReviewState, *,
    policy: TrainabilityPolicy | None = None,
    source_evidence: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes, conflicts = resolve_episodes(step1a, state)
    card_index = step1a.card_index()
    records: list[dict[str, Any]] = []
    for episode in episodes:
        camera = episode["camera_id"]
        member_ids = episode["member_card_ids"]
        stamps = []
        labels: dict[str, int] = {}
        for card_id in member_ids:
            entry = card_index.get(card_id)
            if entry is None:
                continue
            _, member = entry
            moment = parse_timestamp(member.get("timestamp"))
            if moment:
                stamps.append(moment)
            key = str(member.get("original_label") or "UNKNOWN")
            labels[key] = labels.get(key, 0) + 1
        candidate_ids = episode["source_episode_candidate_ids"]
        reviewed_at = max(
            (str((state.get(cid) or {}).get("decided_at") or "") for cid in candidate_ids),
            default="")
        trainability = audit_trainability(
            episode, step1a, policy=policy, source_evidence=source_evidence)
        record = {
            "episode_id": episode["episode_id"],
            "truth_class": episode["truth_class"],
            "camera_id": camera,
            "scene_version": UNKNOWN_SCENE_VERSION,
            "scene_version_status": "unknown_historical_no_metadata",
            "start_timestamp": min(stamps).strftime("%Y-%m-%d %H:%M:%S") if stamps else None,
            "end_timestamp": max(stamps).strftime("%Y-%m-%d %H:%M:%S") if stamps else None,
            "source_episode_candidate_ids": candidate_ids,
            "member_card_ids": member_ids,
            "member_count": len(member_ids),
            "original_label_summary": dict(sorted(labels.items())),
            "review_decision": episode["decision"],
            "review_status": "human_reviewed",
            "localization_status": episode["localization_status"],
            "trainability_status": trainability["trainability_status"],
            "trainability_evidence": trainability,
            "grouping_review": episode["grouping_review"],
            "notes": "; ".join(
                str((state.get(cid) or {}).get("note") or "") for cid in candidate_ids
            ).strip("; "),
            "reviewed_at": reviewed_at,
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "record_kind": "episode",
            "is_litter_episode": episode["truth_class"] == "REQUIRED_LITTER",
        }
        if episode["truth_class"] == "REQUIRED_LITTER":
            record["trainability_status"] = trainability["trainability_status"]
        else:
            # Non-litter / ignore / uncertain carry no training route.
            record["trainability_status"] = "REVIEW_ONLY"
            record["trainability_evidence"] = {
                "trainability_status": "REVIEW_ONLY",
                "lineage_method": "none",
                "lineage_reason": "not_a_required_litter_episode",
                "policy": (policy or TrainabilityPolicy()).as_dict(),
                "members": trainability["members"],
                "note": "Not a REQUIRED_LITTER episode; kept only as review evidence.",
            }
            record["localization_status"] = None
        records.append(record)

    # Whole-candidate non-episode classifications still need a lineage-complete
    # record, but they are explicitly NOT litter episodes.
    for candidate_id in sorted(state.candidates):
        row = state.get(candidate_id) or {}
        decision = row.get("decision")
        if decision not in NON_EPISODE_DECISIONS:
            continue
        candidate = step1a.candidate_by_id.get(candidate_id)
        if candidate is None:
            conflicts.append({"type": "outcome_candidate_missing", "candidate_id": candidate_id})
            continue
        member_ids = sorted(step1a.member_card_ids(candidate_id))
        stamps = []
        labels: dict[str, int] = {}
        for card_id in member_ids:
            entry = card_index.get(card_id)
            if entry is None:
                continue
            _, member = entry
            moment = parse_timestamp(member.get("timestamp"))
            if moment:
                stamps.append(moment)
            key = str(member.get("original_label") or "UNKNOWN")
            labels[key] = labels.get(key, 0) + 1
        records.append({
            "episode_id": episode_id_for(str(candidate["camera_id"]), member_ids),
            "truth_class": decision,
            "camera_id": candidate["camera_id"],
            "scene_version": UNKNOWN_SCENE_VERSION,
            "scene_version_status": "unknown_historical_no_metadata",
            "start_timestamp": min(stamps).strftime("%Y-%m-%d %H:%M:%S") if stamps else None,
            "end_timestamp": max(stamps).strftime("%Y-%m-%d %H:%M:%S") if stamps else None,
            "source_episode_candidate_ids": [candidate_id],
            "member_card_ids": member_ids,
            "member_count": len(member_ids),
            "original_label_summary": dict(sorted(labels.items())),
            "review_decision": decision,
            "review_status": "human_reviewed",
            "localization_status": None,
            "trainability_status": "REVIEW_ONLY",
            "trainability_evidence": {
                "trainability_status": "REVIEW_ONLY",
                "lineage_method": "none",
                "lineage_reason": "candidate_classified_not_required_litter",
                "policy": (policy or TrainabilityPolicy()).as_dict(),
                "members": [],
                "note": "Whole-candidate classification outcome; not a litter episode.",
            },
            "grouping_review": {"confirmed": False, "split": False, "merged": False,
                                "merge_source_candidates": []},
            "notes": str(row.get("note") or ""),
            "reviewed_at": str(row.get("decided_at") or ""),
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "record_kind": "classification_outcome",
            "is_litter_episode": False,
        })

    # Human-discovered missing targets: each becomes its own independent truth
    # target / provisional episode.  They are never merged into the candidate they
    # were found next to (plan §3.5 forbids collapsing per-object truth).
    for manual_target_id in sorted(state.manual_targets):
        target = state.manual_targets[manual_target_id]
        source_card = str(target.get("source_member_card_id") or "")
        source_candidate = str(target.get("source_episode_candidate_id") or "")
        entry = card_index.get(source_card)
        if entry is None:
            conflicts.append({"type": "manual_target_source_card_missing",
                              "manual_target_id": manual_target_id,
                              "source_member_card_id": source_card})
            continue
        _, member = entry
        camera = str(target.get("camera_id") or "")
        truth_class = str(target.get("truth_class") or "")
        if truth_class not in MANUAL_TARGET_TRUTH_CLASSES:
            conflicts.append({"type": "manual_target_invalid_truth_class",
                              "manual_target_id": manual_target_id,
                              "truth_class": truth_class})
            continue

        stamp = parse_timestamp(member.get("timestamp"))
        visible_intervals = []
        if stamp is not None:
            member_assets = dict(member.get("assets") or {})
            has_context = bool(member_assets.get("before_image")) and \
                bool(member_assets.get("after_image"))
            pad = timedelta(seconds=SILVER_BEFORE_AFTER_SECONDS) if has_context \
                else timedelta(0)
            left, right = stamp - pad, stamp + pad
            visible_intervals.append({
                "start": left.strftime("%Y-%m-%d %H:%M:%S"),
                "end": right.strftime("%Y-%m-%d %H:%M:%S"),
                "basis": ("source_review_card_before_after_window" if has_context
                          else "single_source_review_card_instant"),
            })

        inherited = audit_trainability(
            {"member_card_ids": [source_card]}, step1a,
            policy=policy, source_evidence=source_evidence)
        if truth_class == "REQUIRED_LITTER":
            trainability_status = inherited["trainability_status"]
            trainability_evidence = inherited
        else:
            trainability_status = "REVIEW_ONLY"
            trainability_evidence = {
                "trainability_status": "REVIEW_ONLY",
                "lineage_method": inherited.get("lineage_method", "none"),
                "lineage_reason": "manual_target_classified_not_required_litter",
                "policy": inherited["policy"],
                "members": inherited["members"],
                "note": ("Human-discovered target with a non-REQUIRED_LITTER class; "
                         "kept as review evidence only."),
            }

        records.append({
            "episode_id": manual_episode_id_for(camera, manual_target_id),
            "truth_target_id": target.get("truth_target_id") or manual_target_id,
            "truth_class": truth_class,
            "camera_id": camera,
            "scene_version": UNKNOWN_SCENE_VERSION,
            "scene_version_status": "unknown_historical_no_metadata",
            "start_timestamp": (stamp.strftime("%Y-%m-%d %H:%M:%S") if stamp else None),
            "end_timestamp": (stamp.strftime("%Y-%m-%d %H:%M:%S") if stamp else None),
            "visible_intervals": visible_intervals,
            "source_episode_candidate_ids": [source_candidate] if source_candidate else [],
            "source_member_card_id": source_card,
            "member_card_ids": [source_card],
            "member_count": 1,
            "original_label_summary": {str(member.get("label") or "UNKNOWN"): 1},
            "review_decision": "ADD_MISSING_TARGET",
            "review_status": "human_reviewed",
            "location_type": "POINT",
            "point": dict(target.get("point") or {}),
            "localization_status": "NEEDS_RELOCALIZATION",
            "origin": MANUAL_TARGET_ORIGIN,
            "appearance_tag": str(target.get("appearance_tag") or ""),
            "trainability_status": trainability_status,
            "trainability_evidence": trainability_evidence,
            "grouping_review": {"confirmed": False, "split": False, "merged": False,
                                "merge_source_candidates": []},
            "notes": str(target.get("note") or ""),
            "reviewed_at": str(target.get("updated_at") or target.get("created_at") or ""),
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "record_kind": "manual_missing_target",
            "is_litter_episode": truth_class == "REQUIRED_LITTER",
        })

    records.sort(key=lambda r: (r["camera_id"], r["episode_id"]))
    return records, conflicts


def build_summary(
    step1a: Step1AInput, state: ReviewState,
    records: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    *,
    queue: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    progress = state.progress(step1a)
    decisions: dict[str, int] = {d: 0 for d in DECISIONS}
    for row in state.candidates.values():
        if row.get("status") == "reviewed" and row.get("decision") in decisions:
            decisions[str(row["decision"])] += 1

    truth_counts: dict[str, int] = {t: 0 for t in TRUTH_CLASSES}
    for record in records:
        truth_counts[str(record["truth_class"])] = truth_counts.get(str(record["truth_class"]), 0) + 1
    litter_episodes = [r for r in records if r["truth_class"] == "REQUIRED_LITTER"]
    outcomes = [r for r in records if r.get("record_kind") == "classification_outcome"]

    trainability: dict[str, int] = {t: 0 for t in TRAINABILITY_STATUSES}
    per_camera: dict[str, dict[str, int]] = {}
    for record in records:
        if record["truth_class"] != "REQUIRED_LITTER":
            continue
        trainability[str(record["trainability_status"])] = \
            trainability.get(str(record["trainability_status"]), 0) + 1
        bucket = per_camera.setdefault(str(record["camera_id"]), {
            "required_litter_episodes": 0, "trainable_episodes": 0})
        bucket["required_litter_episodes"] += 1
        if record["trainability_status"] == "TRAINABLE_SOURCE_NATIVE":
            bucket["trainable_episodes"] += 1

    queue = list(queue or [])
    grouping_risk = [q for q in queue if q.get("grouping_risk")]
    lineage_only = [q for q in queue if q.get("lineage_only")]

    manual_records = [r for r in records if r.get("record_kind") == "manual_missing_target"]
    manual_block = state.manual_target_counts()
    manual_block["manual_targets_trainable_lineage"] = sum(
        1 for r in manual_records if r["trainability_status"] == "TRAINABLE_SOURCE_NATIVE")
    manual_block["manual_targets_lineage_unresolved"] = sum(
        1 for r in manual_records if r["trainability_status"] == "LINEAGE_UNRESOLVED")
    manual_block["manual_gold_episode_count"] = sum(
        1 for r in manual_records if r["is_litter_episode"])
    manual_block["manual_targets_are_independent_of_source_candidate"] = True
    manual_block["manual_targets_merged_automatically"] = False

    ops = {
        "confirm_operations": decisions["CONFIRM"],
        "split_operations": decisions["SPLIT"],
        "merge_operations": decisions["MERGE"],
        "non_litter_decisions": decisions["NON_LITTER"],
        "ignore_small_decisions": decisions["IGNORE_SMALL"],
        "uncertain_decisions": decisions["UNCERTAIN"],
        "add_missing_target_operations": len(manual_records),
    }
    reviewed_candidates = progress["reviewed"]
    return {
        "schema_version": SCHEMA_VERSION,
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "input": {
            "step1a_artifact": str(step1a.artifact_path),
            "step1a_artifact_sha256": step1a.artifact_sha256,
            "step1a_code_commit": step1a.step1a_code_commit,
            "step1a_candidate_count": step1a.candidate_count,
            "step1a_member_card_count": step1a.member_card_count,
            "step1a_summary": {
                "singleton_groups": sum(1 for c in step1a.candidates
                                        if int(c.get("member_count") or 0) == 1),
                "multi_card_groups": sum(1 for c in step1a.candidates
                                         if int(c.get("member_count") or 0) > 1),
                "ambiguous_groups": sum(1 for c in step1a.candidates
                                        if c.get("candidate_ambiguous")),
            },
        },
        "human_review": {
            "total": progress["total"],
            "reviewed": progress["reviewed"],
            "skipped": progress["skipped"],
            "pending": progress["pending"],
            "reviewed_fraction": round(progress["reviewed"] / progress["total"], 6)
            if progress["total"] else 0.0,
            "decisions": decisions,
            "operations": ops,
            "review_complete": progress["pending"] == 0,
        },
        "gold_episodes": {
            "gold_episode_count": len(litter_episodes),
            "manual_gold_episode_count": sum(
                1 for r in litter_episodes
                if r.get("record_kind") == "manual_missing_target"),
            "classification_outcome_count": len(outcomes),
            "by_truth_class": truth_counts,
            "merge_compression": sum(
                max(0, len(r["source_episode_candidate_ids"]) - 1) for r in records
                if r.get("grouping_review", {}).get("merged")),
            "split_expansion": sum(
                1 for r in records if r.get("grouping_review", {}).get("split")),
            "net_candidate_to_episode_change": (
                len(litter_episodes) - reviewed_candidates if reviewed_candidates else 0),
            "statement": ("Gold episodes exist only for human-reviewed candidates. "
                          "Empty counts mean no human review has happened yet."),
        },
        "source_native_trainability": {
            "counts": trainability,
            "per_camera": dict(sorted(per_camera.items())),
            "counts_apply_to": "REQUIRED_LITTER episodes only",
        },
        "manual_missing_targets": manual_block,
        "risk_coverage": {
            "queue_total": len(queue),
            "grouping_risk_total": len(grouping_risk),
            "grouping_risk_reviewed": sum(
                1 for q in grouping_risk
                if state.status_of(q["episode_candidate_id"]) == "reviewed"),
            "lineage_only_total": len(lineage_only),
            "lineage_only_reviewed": sum(
                1 for q in lineage_only
                if state.status_of(q["episode_candidate_id"]) == "reviewed"),
            "box_wrong_derived_gold": sum(
                1 for r in records
                if r["original_label_summary"].get("BOX_WRONG")),
        },
        "conflicts": list(conflicts),
        "boundaries": {
            "sealed_inference_accessed": False,
            "training_started": False,
            "step1a_artifact_modified": False,
            "annotation_complete_tiles_generated": False,
            "hard_negative_mining_started": False,
            "automatic_merge_applied_without_human": False,
            "scene_version_fabricated": False,
            "manual_targets_auto_merged": False,
            "manual_target_bbox_inferred": False,
        },
    }


def build_manifest(
    step1a: Step1AInput, state: ReviewState, *,
    step1b_code_commit: str,
    config: Mapping[str, Any],
    generated_at: str,
    review_state_path: Path,
    gold_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "generated_at": generated_at,
        "step1a": {
            "artifact": str(step1a.artifact_path),
            "artifact_sha256": step1a.artifact_sha256,
            "manifest": str(step1a.manifest_path) if step1a.manifest_path else None,
            "code_commit": step1a.step1a_code_commit,
            "candidate_count": step1a.candidate_count,
            "member_card_count": step1a.member_card_count,
        },
        "step1b": {
            "code_commit": step1b_code_commit,
            "config": dict(config),
            "review_dataset_fingerprint": review_fingerprint(state),
        },
        "outputs": {
            "gold_episodes": str(gold_path),
            "review_state": str(review_state_path),
            "summary": str(summary_path),
        },
        "boundaries": {
            "step1a_artifact_immutable": True,
            "sealed_inference_accessed": False,
            "training_started": False,
            "auto_gold_generated": False,
            "manual_target_bbox_inferred": False,
            "manual_targets_auto_merged": False,
        },
    }


def review_fingerprint(state: ReviewState) -> str:
    stable = {
        cid: {
            "status": row.get("status"),
            "decision": row.get("decision"),
            "member_card_ids": sorted(
                m for ep in (row.get("episodes") or []) for m in ep.get("member_card_ids") or []),
            "merge_targets": sorted(row.get("merge_targets") or []),
            "truth_class": row.get("truth_class"),
        }
        for cid, row in sorted(state.candidates.items())
    }
    blob = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manual = json.dumps({
        mid: {"truth_class": row.get("truth_class"),
              "point": row.get("point"),
              "card_id": row.get("source_member_card_id"),
              "candidate_id": row.get("source_episode_candidate_id")}
        for mid, row in sorted(state.manual_targets.items())
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256((blob + "||" + manual).encode("utf-8")).hexdigest()


def write_review_outputs(
    output_dir: Path, *,
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_path = output_dir / "gold_episodes.jsonl"
    with gold_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = output_dir / "SUMMARY.json"
    atomic_write_json(summary_path, summary)
    manifest_path = output_dir / "MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "gold_episodes": str(gold_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
    }
