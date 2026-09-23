"""Step 1C-1: localization review for the 180 REQUIRED_LITTER Gold episodes.

Step 1B's CONFIRM only proves "this litter is real"; it says nothing about whether the
historical box is usable as detector supervision.  This module decides, per episode,
between ``VERIFIED_BBOX`` / ``LOCALIZATION_UNRESOLVED`` / ``TRUTH_REVIEW_REQUIRED``,
using a machine proposal helper that a human must explicitly select from.

Boundaries: Gold truth and Step 1C-0 recovery evidence are immutable inputs.  No
detector is run, no training tile is produced, nothing is trained.  Pure stdlib so the
unit-test venv needs no cv2; only the proposal engine touches images and is injected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

SCHEMA_VERSION = "ground_litter_gold_localization_v1"
GENERATOR_VERSION = "step1c1-1.0.0"
REVIEW_SCHEMA_VERSION = "gold_localization_review_v1"

LOCALIZATION_STATUSES = (
    "VERIFIED_BBOX",
    "NEEDS_RELOCALIZATION",
    "LOCALIZATION_UNRESOLVED",
    "TRUTH_REVIEW_REQUIRED",
)
#: Only these may enter Step 1C-2 detector training tiles.
TRAINING_READY_STATUS = "VERIFIED_BBOX"

DECISIONS = ("BOX_OK", "PROPOSAL_SELECTED", "UNRESOLVED", "TRUTH_REVIEW_REQUIRED")

ORIGINS = ("manual_missing_target", "box_wrong", "split_derived", "historical_litter")

MAX_PROPOSALS = 3
PROPOSAL_IDS = ("A", "B", "C")

#: Below this the bbox cannot supervise a detector at all.
MIN_BBOX_SHORT_SIDE_PX = 3.0
#: A box covering more of the frame than this is treated as suspect and needs an
#: explicit human confirmation rather than silent acceptance (§14).
OVERSIZED_AREA_FRACTION = 0.05
#: Two Required episodes on one camera whose boxes overlap this much are a
#: "one box covers two targets" risk that must be reviewed, never auto-accepted (§11).
SHARED_BBOX_IOU = 0.70
#: Two Required episodes whose boxes coincide and whose timestamps are within this many
#: seconds may be one box covering two objects in the same frame (§11).
SAME_FRAME_WINDOW_SECONDS = 300.0

#: Exact reproduction of the historical context crop used by the Silver review UI
#: (scripts/build_ground_litter_active_review.py: max(32, w*scale, h*scale) side,
#: centred on the box, clipped to the frame).
CONTEXT_CROP_SCALE = 4.0
CONTEXT_CROP_MIN_SIDE = 32.0

SEALED_MARKERS = (
    "sealed_test",
    "SEALED_DO_NOT_TUNE",
    "ground-litter-detector-feasibility-20260923-r1",
    "ground-litter-feasibility/20260923-r1",
)


class LocalizationError(RuntimeError):
    """The localization review could not proceed."""


class SealedAssetError(LocalizationError):
    """A Step 0A Sealed asset was supplied; hard fail (§20)."""


def assert_not_sealed(*values: Any) -> None:
    for value in values:
        text = str(value or "")
        for marker in SEALED_MARKERS:
            if marker.lower() in text.lower():
                raise SealedAssetError(f"refusing to touch a Step 0A Sealed asset ({marker!r})")


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #


def parse_timestamp(text: str | None) -> datetime | None:
    if not isinstance(text, str) or not text.strip():
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


def normalise_bbox(bbox: Sequence[float] | None) -> tuple[float, float, float, float] | None:
    if bbox is None or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def validate_bbox(
    bbox: Sequence[float] | None, *, frame_width: int, frame_height: int,
    require_confirmation_over_fraction: float = OVERSIZED_AREA_FRACTION,
) -> dict[str, Any]:
    """Validate a candidate source-frame bbox; never silently accept a nonsense box."""
    issues: list[str] = []
    if bbox is None or len(bbox) != 4:
        return {"ok": False, "issues": ["bbox_missing"], "bbox": None,
                "requires_confirmation": False}
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return {"ok": False, "issues": ["bbox_not_numeric"], "bbox": None,
                "requires_confirmation": False}
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        issues.append("bbox_not_finite")
    if x2 <= x1 or y2 <= y1:
        issues.append("bbox_zero_or_negative_area")
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        issues.append("bbox_out_of_frame")
    area = bbox_area((x1, y1, x2, y2))
    requires_confirmation = False
    if frame_width > 0 and frame_height > 0 and area > require_confirmation_over_fraction * frame_width * frame_height:
        requires_confirmation = True
        issues.append("bbox_covers_large_fraction_of_frame")
    # An oversized box is still a *legal* box: it needs an explicit human confirmation
    # rather than being discarded, so its coordinates must survive.
    hard_issues = [i for i in issues if i != "bbox_covers_large_fraction_of_frame"]
    return {
        "ok": not issues and not requires_confirmation,
        "issues": issues,
        "hard_issues": hard_issues,
        "bbox": [x1, y1, x2, y2] if not hard_issues else None,
        "requires_confirmation": requires_confirmation,
        "area_px": area,
        "area_fraction": round(area / float(frame_width * frame_height), 6)
        if frame_width and frame_height else None,
    }


def context_crop_box(parent_bbox: Sequence[float], *, frame_width: int = 2560,
                     frame_height: int = 1440, scale: float = CONTEXT_CROP_SCALE,
                     min_side: float = CONTEXT_CROP_MIN_SIDE
                     ) -> tuple[int, int, int, int]:
    """Reproduce the historical context crop of a candidate box, exactly."""
    x1, y1, x2, y2 = (float(v) for v in parent_bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(float(min_side), (x2 - x1) * float(scale), (y2 - y1) * float(scale))
    return (max(0, round(cx - side / 2)), max(0, round(cy - side / 2)),
            min(frame_width, round(cx + side / 2)), min(frame_height, round(cy + side / 2)))


def map_point_to_source(
    parent_bbox: Sequence[float] | None, point: Mapping[str, Any] | None, *,
    frame_width: int = 2560, frame_height: int = 1440,
) -> dict[str, Any]:
    """Map a context-crop-normalised manual point back to source-frame pixels.

    The point recorded in Step 1B is normalised to the *crop*, not the source frame.
    The crop is reproducible from the parent card's box, so we can invert it — and we
    self-check by requiring the derived crop size to equal the clicked image size that
    was recorded at review time.  If it does not match, the mapping is NOT trusted.
    """
    if parent_bbox is None or not point:
        return {"ok": False, "reason": "missing_parent_bbox_or_point"}
    normalised = normalise_bbox(parent_bbox)
    if normalised is None:
        return {"ok": False, "reason": "invalid_parent_bbox"}
    try:
        x_norm = float(point.get("x_norm"))
        y_norm = float(point.get("y_norm"))
        clicked_w = int(point.get("clicked_image_width") or 0)
        clicked_h = int(point.get("clicked_image_height") or 0)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "invalid_point"}
    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        return {"ok": False, "reason": "point_outside_crop"}
    crop = context_crop_box(normalised, frame_width=frame_width, frame_height=frame_height)
    derived_w, derived_h = crop[2] - crop[0], crop[3] - crop[1]
    if clicked_w <= 0 or clicked_h <= 0:
        return {"ok": False, "reason": "clicked_size_missing",
                "derived_crop_size": [derived_w, derived_h]}
    if (derived_w, derived_h) != (clicked_w, clicked_h):
        return {"ok": False, "reason": "derived_crop_size_mismatch",
                "derived_crop_size": [derived_w, derived_h],
                "recorded_crop_size": [clicked_w, clicked_h]}
    return {
        "ok": True,
        "reason": "derived_crop_size_matches_recorded",
        "crop_box": list(crop),
        "x": round(crop[0] + x_norm * derived_w, 3),
        "y": round(crop[1] + y_norm * derived_h, 3),
        "point_source": [round(crop[0] + x_norm * derived_w, 3),
                         round(crop[1] + y_norm * derived_h, 3)],
        "derived_crop_size": [derived_w, derived_h],
        "recorded_crop_size": [clicked_w, clicked_h],
    }


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LocalizationEpisode:
    episode_id: str
    truth_class: str
    camera_id: str
    scene_version: str
    record_kind: str
    origin: str
    origin_flags: tuple[str, ...]
    source_timestamp: str | None
    member_card_ids: tuple[str, ...]
    member_labels: tuple[str, ...]
    original_bbox: tuple[float, float, float, float] | None
    original_bbox_source: str
    original_point: Mapping[str, Any] | None
    point_source: Mapping[str, Any] | None
    source_file_id: str | None
    source_width: int
    source_height: int
    verification_frame_path: str
    source_recovery_status: str
    shared_bbox_episode_ids: tuple[str, ...]
    screen: Mapping[str, Any]
    upstream_localization_status: str | None = None

    @property
    def origin_counts_as_box_wrong(self) -> bool:
        return "box_wrong" in self.origin_flags

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "truth_class": self.truth_class,
            "camera_id": self.camera_id,
            "scene_version": self.scene_version,
            "record_kind": self.record_kind,
            "origin": self.origin,
            "origin_flags": list(self.origin_flags),
            "source_timestamp": self.source_timestamp,
            "member_card_ids": list(self.member_card_ids),
            "member_labels": list(self.member_labels),
            "original_bbox": list(self.original_bbox) if self.original_bbox else None,
            "original_bbox_source": self.original_bbox_source,
            "original_point": dict(self.original_point or {}) or None,
            "point_source": dict(self.point_source or {}) or None,
            "source_file_id": self.source_file_id,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "verification_frame_path": self.verification_frame_path,
            "source_recovery_status": self.source_recovery_status,
            "shared_bbox_episode_ids": list(self.shared_bbox_episode_ids),
            "screen": dict(self.screen),
            "upstream_localization_status": self.upstream_localization_status,
        }


@dataclass(frozen=True)
class LocalizationInput:
    gold_path: Path
    gold_sha256: str
    recovery_summary_path: Path | None
    recovery_evidence_sha256: str
    required: tuple[LocalizationEpisode, ...]

    @property
    def required_count(self) -> int:
        return len(self.required)

    @property
    def by_id(self) -> dict[str, LocalizationEpisode]:
        return {e.episode_id: e for e in self.required}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


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


def load_step1a_card_labels(step1a_artifact: Path | str) -> dict[str, str]:
    """card_id -> historical human label (authoritative for BOX_WRONG origin).

    Gold's own ``original_label_summary`` is unusable for this: Step 1B built it from a
    field name ("original_label") that does not exist on the Step 1A lineage row, so it
    reads ``{"UNKNOWN": n}``.  The per-member ``original_label`` is correct, and so is
    the Step 1A lineage ``label``; we use both rather than the broken summary.
    """
    mapping: dict[str, str] = {}
    path = Path(step1a_artifact)
    if not path.is_file():
        return mapping
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            for card in (row.get("lineage") or {}).get("review_cards") or []:
                card_id = str(card.get("card_id") or "")
                label = card.get("label")
                if card_id and label:
                    mapping[card_id] = str(label)
    return mapping


def classify_origin(record: Mapping[str, Any],
                    member_labels: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    flags: list[str] = []
    if str(record.get("record_kind")) == "manual_missing_target":
        flags.append("manual_missing_target")
    if any(str(l) == "BOX_WRONG" for l in member_labels):
        flags.append("box_wrong")
    if (record.get("grouping_review") or {}).get("split"):
        flags.append("split_derived")
    if not flags:
        flags.append("historical_litter")
    priority = {"manual_missing_target": 0, "box_wrong": 1, "split_derived": 2,
                "historical_litter": 3}
    primary = min(flags, key=lambda f: priority.get(f, 9))
    return primary, tuple(sorted(flags))


def load_localization_input(
    gold_path: Path | str,
    recovery_root: Path | str,
    *,
    step1a_artifact: Path | str | None = None,
    gold_manifest: Path | str | None = None,
) -> LocalizationInput:
    gold = Path(gold_path)
    if not gold.is_file():
        raise FileNotFoundError(f"Gold artifact not found: {gold}")
    assert_not_sealed(gold, recovery_root, step1a_artifact, gold_manifest)
    gold_sha = sha256_file(gold)

    root = Path(recovery_root)
    evidence_rows = read_jsonl(root / "episode_source_evidence.jsonl")
    evidence = {str(r.get("episode_id")): r for r in evidence_rows}
    evidence_path = root / "episode_source_evidence.jsonl"
    evidence_sha = sha256_file(evidence_path) if evidence_path.is_file() else ""

    labels = load_step1a_card_labels(step1a_artifact) if step1a_artifact else {}

    records = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    required: list[LocalizationEpisode] = []
    seen: set[str] = set()
    for record in records:
        if str(record.get("truth_class")) != "REQUIRED_LITTER":
            continue
        episode_id = str(record.get("episode_id") or "")
        if not episode_id:
            raise LocalizationError("Required record without episode_id")
        if episode_id in seen:
            raise LocalizationError(f"duplicate episode_id in Gold: {episode_id}")
        seen.add(episode_id)

        members = (record.get("trainability_evidence") or {}).get("members") or []
        member_ids = tuple(str(m) for m in (record.get("member_card_ids") or []))
        member_labels: list[str] = []
        for member in members:
            label = member.get("original_label") or labels.get(str(member.get("card_id") or ""))
            if label:
                member_labels.append(str(label))
        if not member_labels:
            member_labels = [str(labels.get(m, "")) for m in member_ids if labels.get(m)]
        origin, origin_flags = classify_origin(record, member_labels)

        evidence_row = evidence.get(episode_id) or {}
        frame_path = str(evidence_row.get("verification_frame_path") or "")
        width = int(evidence_row.get("source_width") or 0)
        height = int(evidence_row.get("source_height") or 0)

        original_bbox: tuple[float, float, float, float] | None = None
        bbox_source = "none"
        if origin == "manual_missing_target":
            # The member box belongs to the PARENT card (a different object) and is not
            # the localisation of this newly discovered target.
            bbox_source = "none_manual_target_has_point_only"
        else:
            for member in members:
                candidate = normalise_bbox(member.get("bbox"))
                if candidate is not None:
                    original_bbox = candidate
                    bbox_source = "gold_trainability_evidence_member_bbox"
                    break

        original_point = record.get("point") if origin == "manual_missing_target" else None
        parent_bbox = normalise_bbox(members[0].get("bbox")) if members else None
        point_source = (map_point_to_source(parent_bbox, original_point,
                                            frame_width=width or 2560,
                                            frame_height=height or 1440)
                        if original_point else None)

        required.append(LocalizationEpisode(
            episode_id=episode_id,
            truth_class="REQUIRED_LITTER",
            camera_id=str(record.get("camera_id") or ""),
            scene_version=str(record.get("scene_version") or ""),
            record_kind=str(record.get("record_kind") or ""),
            origin=origin,
            origin_flags=origin_flags,
            source_timestamp=evidence_row.get("requested_timestamp")
            or record.get("start_timestamp"),
            member_card_ids=member_ids,
            member_labels=tuple(member_labels),
            original_bbox=original_bbox,
            original_bbox_source=bbox_source,
            original_point=original_point,
            point_source=point_source,
            source_file_id=evidence_row.get("source_file_id"),
            source_width=width,
            source_height=height,
            verification_frame_path=frame_path,
            source_recovery_status=str(evidence_row.get("source_recovery_status") or ""),
            shared_bbox_episode_ids=(),
            screen={},
            upstream_localization_status=record.get("localization_status"),
        ))

    required.sort(key=lambda e: (e.camera_id, e.episode_id))
    enriched = _annotate_screening(required)
    return LocalizationInput(
        gold_path=gold, gold_sha256=gold_sha, recovery_summary_path=root / "SUMMARY.json",
        recovery_evidence_sha256=evidence_sha, required=tuple(enriched))


def _annotate_screening(episodes: Sequence[LocalizationEpisode]
                        ) -> list[LocalizationEpisode]:
    """Screen for ordering only — never a decision (§21)."""
    by_camera: dict[str, list[LocalizationEpisode]] = {}
    for episode in episodes:
        by_camera.setdefault(episode.camera_id, []).append(episode)

    shared: dict[str, set[str]] = {e.episode_id: set() for e in episodes}
    same_frame: dict[str, set[str]] = {e.episode_id: set() for e in episodes}
    for rows in by_camera.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if not (a.original_bbox and b.original_bbox):
                    continue
                if bbox_iou(a.original_bbox, b.original_bbox) < SHARED_BBOX_IOU:
                    continue
                shared[a.episode_id].add(b.episode_id)
                shared[b.episode_id].add(a.episode_id)
                ta, tb = parse_timestamp(a.source_timestamp), parse_timestamp(b.source_timestamp)
                if ta and tb and abs((ta - tb).total_seconds()) <= SAME_FRAME_WINDOW_SECONDS:
                    # same box, near-simultaneous: one box may cover two objects
                    same_frame[a.episode_id].add(b.episode_id)
                    same_frame[b.episode_id].add(a.episode_id)

    out: list[LocalizationEpisode] = []
    for episode in episodes:
        peers = tuple(sorted(shared[episode.episode_id]))
        same_frame_peers = tuple(sorted(same_frame[episode.episode_id]))
        reasons: list[str] = []
        if episode.origin == "manual_missing_target":
            reasons.append("manual_point_without_bbox")
        if episode.origin == "box_wrong":
            reasons.append("historical_box_wrong")
        if episode.original_bbox is None and episode.origin != "manual_missing_target":
            reasons.append("missing_original_bbox")
        if episode.original_bbox is not None:
            width = episode.original_bbox[2] - episode.original_bbox[0]
            height = episode.original_bbox[3] - episode.original_bbox[1]
            frame_area = float(episode.source_width * episode.source_height) or 1.0
            if width < MIN_BBOX_SHORT_SIDE_PX or height < MIN_BBOX_SHORT_SIDE_PX:
                reasons.append("bbox_below_min_short_side")
            if bbox_area(episode.original_bbox) > OVERSIZED_AREA_FRACTION * frame_area:
                reasons.append("suspected_oversized_bbox")
        if peers:
            reasons.append("shared_bbox_with_other_episodes")
        if same_frame_peers:
            reasons.append("possible_multi_object_box")
        if episode.point_source is not None and not episode.point_source.get("ok"):
            reasons.append("manual_point_mapping_unverified")
        if episode.source_recovery_status != "RECOVERED_SOURCE_NATIVE":
            reasons.append("source_not_recovered")
        screen = {
            "likely_box_bad": bool(reasons),
            "screen_reasons": sorted(reasons),
            "screened_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        screen["same_frame_conflict_episode_ids"] = list(same_frame_peers)
        out.append(LocalizationEpisode(**{**episode.__dict__,
                                          "shared_bbox_episode_ids": peers,
                                          "screen": screen}))
    # Review the risky ones first: likely_box_bad (box_wrong, manual points, shared
    # boxes) ahead of the plain historical boxes, then camera/id for stability.
    out.sort(key=lambda e: (not e.screen["likely_box_bad"],
                            e.camera_id, e.episode_id))
    return out


# --------------------------------------------------------------------------- #
# proposals
# --------------------------------------------------------------------------- #


class ProposalEngine(Protocol):
    def propose(self, *, frame_path: Path, seed_bbox: Sequence[float] | None,
                point: Sequence[float] | None, frame_width: int, frame_height: int,
                revision: int) -> list[dict[str, Any]]:
        ...


def validate_proposals(proposals: Sequence[Mapping[str, Any]], *,
                       frame_width: int, frame_height: int) -> list[dict[str, Any]]:
    """Drop anything that is not a legal source-frame box; never a 0-area/NaN box.

    Letters are assigned by the engine's ranking order (A first), so the reviewer always
    sees a stable A/B/C regardless of what the engine called things.
    """
    cleaned: list[dict[str, Any]] = []
    for row in proposals:
        verdict = validate_bbox(row.get("bbox"), frame_width=frame_width,
                               frame_height=frame_height,
                               require_confirmation_over_fraction=1.1)
        if not verdict["ok"]:
            continue
        x1, y1, x2, y2 = verdict["bbox"]
        cleaned.append({
            "bbox": [x1, y1, x2, y2],
            "bbox_norm": [round(x1 / frame_width, 6), round(y1 / frame_height, 6),
                          round(x2 / frame_width, 6), round(y2 / frame_height, 6)],
            "method": str(row.get("method") or "unknown"),
            "revision": int(row.get("revision") or 1),
            "area_px": verdict["area_px"],
        })
        if len(cleaned) >= MAX_PROPOSALS:
            break
    for index, row in enumerate(cleaned):
        row["proposal_id"] = PROPOSAL_IDS[index]
    return cleaned


# --------------------------------------------------------------------------- #
# review state
# --------------------------------------------------------------------------- #


class ReviewError(ValueError):
    """Invalid reviewer action; surfaced, never coerced."""


@dataclass
class ReviewState:
    path: Path
    schema_version: str = REVIEW_SCHEMA_VERSION
    input_gold_sha256: str = ""
    required_count: int = 0
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    audit_trail: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str, *, data: LocalizationInput | None = None) -> "ReviewState":
        target = Path(path)
        if not target.is_file():
            state = cls(path=target)
            if data is not None:
                state.input_gold_sha256 = data.gold_sha256
                state.required_count = data.required_count
            return state
        payload = json.loads(target.read_text(encoding="utf-8"))
        state = cls(
            path=target,
            schema_version=str(payload.get("review_schema_version") or REVIEW_SCHEMA_VERSION),
            input_gold_sha256=str((payload.get("input") or {}).get("gold_sha256") or ""),
            required_count=int((payload.get("input") or {}).get("required_count") or 0),
            decisions=dict(payload.get("decisions") or {}),
            audit_trail=list(payload.get("audit_trail") or []),
        )
        if data is not None and state.input_gold_sha256 and \
                state.input_gold_sha256 != data.gold_sha256:
            raise ReviewError("review state belongs to a different Gold artifact")
        return state

    def get(self, episode_id: str) -> dict[str, Any] | None:
        return self.decisions.get(episode_id)

    def status_of(self, episode_id: str) -> str:
        row = self.decisions.get(episode_id)
        if not row:
            return "pending"
        return str(row.get("localization_status") or "pending")

    def save(self) -> None:
        atomic_write_json(self.path, {
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "loaded_schema_version": self.schema_version,
            "input": {"gold_sha256": self.input_gold_sha256,
                      "required_count": self.required_count},
            "decisions": self.decisions,
            "audit_trail": self.audit_trail,
        })

    def _record(self, episode_id: str, action: str, payload: Mapping[str, Any]) -> None:
        self.audit_trail.append({
            "at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "episode_id": episode_id, "action": action, "payload": dict(payload)})

    # -- mutations ---------------------------------------------------------- #

    def _base(self, episode: LocalizationEpisode) -> dict[str, Any]:
        return {
            "episode_id": episode.episode_id,
            "truth_target_id": episode.episode_id,
            "camera_id": episode.camera_id,
            "scene_version": episode.scene_version,
            "source_file_id": episode.source_file_id,
            "source_timestamp": episode.source_timestamp,
            "source_width": episode.source_width,
            "source_height": episode.source_height,
            "verification_frame_path": episode.verification_frame_path,
            "origin": episode.origin,
            "original_location_type": "POINT" if episode.origin == "manual_missing_target"
            else ("BBOX" if episode.original_bbox else "UNKNOWN"),
            "original_bbox": list(episode.original_bbox) if episode.original_bbox else None,
            "original_point": dict(episode.original_point or {}) or None,
            "original_point_source": dict(episode.point_source or {}) or None,
            "review_schema_version": REVIEW_SCHEMA_VERSION,
        }

    def decide_box_ok(self, episode: LocalizationEpisode, *, note: str = "",
                      confirm_oversized: bool = False) -> dict[str, Any]:
        if episode.original_bbox is None:
            raise ReviewError("this episode has no original bbox; use a proposal instead")
        verdict = validate_bbox(episode.original_bbox, frame_width=episode.source_width,
                                frame_height=episode.source_height)
        if not verdict["ok"]:
            if not (verdict["requires_confirmation"] and confirm_oversized):
                raise ReviewError(
                    f"original bbox is not usable: {', '.join(verdict['issues'])}"
                    + (" (pass confirm_oversized to accept anyway)"
                       if verdict["requires_confirmation"] else ""))
        row = {**self._base(episode),
               "localization_decision": "BOX_OK",
               "verified_bbox": list(verdict["bbox"]),
               "verified_bbox_norm": [
                   round(verdict["bbox"][0] / episode.source_width, 6),
                   round(verdict["bbox"][1] / episode.source_height, 6),
                   round(verdict["bbox"][2] / episode.source_width, 6),
                   round(verdict["bbox"][3] / episode.source_height, 6)],
               "location_type": "BBOX",
               "localization_status": "VERIFIED_BBOX",
               "proposal_source_type": "original_bbox_unchanged",
               "proposal_revision": 0,
               "oversized_confirmed_by_human": bool(verdict["requires_confirmation"]),
               "note": note,
               "reviewed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
               "review_status": "human_reviewed"}
        self.decisions[episode.episode_id] = row
        self._record(episode.episode_id, "decide:BOX_OK",
                     {"bbox": verdict["bbox"], "note": note})
        self.save()
        return row

    def store_proposals(self, episode: LocalizationEpisode,
                        proposals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        row = self.decisions.get(episode.episode_id) or {
            **self._base(episode), "review_status": "in_progress"}
        row["proposals"] = [dict(p) for p in proposals]
        row["proposal_revision"] = int(row.get("proposal_revision") or 0) + 1
        row.setdefault("localization_status", "NEEDS_RELOCALIZATION")
        row.setdefault("localization_decision", None)
        self.decisions[episode.episode_id] = row
        self._record(episode.episode_id, "proposals:generated",
                     {"count": len(proposals),
                      "revision": row["proposal_revision"],
                      "methods": [p.get("method") for p in proposals]})
        self.save()
        return row

    def select_proposal(self, episode: LocalizationEpisode, proposal_id: str,
                        *, note: str = "") -> dict[str, Any]:
        row = self.decisions.get(episode.episode_id)
        if not row or not row.get("proposals"):
            raise ReviewError("no proposals available to select from")
        chosen = next((p for p in row["proposals"]
                       if str(p.get("proposal_id")) == str(proposal_id)), None)
        if chosen is None:
            raise ReviewError(f"unknown proposal_id {proposal_id!r}")
        verdict = validate_bbox(chosen["bbox"], frame_width=episode.source_width,
                                frame_height=episode.source_height,
                                require_confirmation_over_fraction=1.1)
        if not verdict["ok"]:
            raise ReviewError(f"selected proposal is not a legal bbox: "
                              f"{', '.join(verdict['issues'])}")
        x1, y1, x2, y2 = verdict["bbox"]
        updated = {**row,
                   "localization_decision": "PROPOSAL_SELECTED",
                   "selected_proposal_id": str(proposal_id),
                   "verified_bbox": [x1, y1, x2, y2],
                   "verified_bbox_norm": [round(x1 / episode.source_width, 6),
                                          round(y1 / episode.source_height, 6),
                                          round(x2 / episode.source_width, 6),
                                          round(y2 / episode.source_height, 6)],
                   "location_type": "BBOX",
                   "localization_status": "VERIFIED_BBOX",
                   "proposal_source_type": str(chosen.get("method") or "unknown"),
                   "note": note,
                   "reviewed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "review_status": "human_reviewed"}
        self.decisions[episode.episode_id] = updated
        self._record(episode.episode_id, "decide:PROPOSAL_SELECTED",
                     {"proposal_id": proposal_id, "bbox": [x1, y1, x2, y2], "note": note})
        self.save()
        return updated

    def decide_unresolved(self, episode: LocalizationEpisode, *, note: str = "") -> dict[str, Any]:
        row = {**(self.decisions.get(episode.episode_id) or self._base(episode)),
               "localization_decision": "UNRESOLVED",
               "verified_bbox": None,
               "location_type": ("POINT" if episode.origin == "manual_missing_target"
                                 else "UNKNOWN"),
               "localization_status": "LOCALIZATION_UNRESOLVED",
               "note": note,
               "reviewed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
               "review_status": "human_reviewed"}
        self.decisions[episode.episode_id] = row
        self._record(episode.episode_id, "decide:UNRESOLVED", {"note": note})
        self.save()
        return row

    def decide_truth_review_required(self, episode: LocalizationEpisode, *,
                                     reason: str, note: str = "") -> dict[str, Any]:
        if not reason.strip():
            raise ReviewError("TRUTH_REVIEW_REQUIRED needs a reason")
        row = {**(self.decisions.get(episode.episode_id) or self._base(episode)),
               "localization_decision": "TRUTH_REVIEW_REQUIRED",
               "verified_bbox": None,
               "localization_status": "TRUTH_REVIEW_REQUIRED",
               "truth_review_reason": reason,
               "note": note,
               "reviewed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
               "review_status": "human_reviewed"}
        self.decisions[episode.episode_id] = row
        self._record(episode.episode_id, "decide:TRUTH_REVIEW_REQUIRED",
                     {"reason": reason, "note": note})
        self.save()
        return row

    def reset(self, episode_id: str) -> None:
        self.decisions.pop(episode_id, None)
        self._record(episode_id, "reset", {})
        self.save()

    def progress(self, data: LocalizationInput) -> dict[str, int]:
        reviewed = sum(1 for e in data.required
                       if self.status_of(e.episode_id) != "pending")
        return {"total": data.required_count, "reviewed": reviewed,
                "pending": data.required_count - reviewed}


@dataclass
class Reviewer:
    """Ties the pure state machine to an injected proposal engine."""

    data: LocalizationInput
    state: ReviewState
    engine: ProposalEngine | None = None

    def generate_proposals(self, episode_id: str) -> dict[str, Any]:
        episode = self.data.by_id.get(episode_id)
        if episode is None:
            raise ReviewError(f"unknown episode {episode_id}")
        if self.engine is None:
            raise ReviewError("no proposal engine configured")
        frame = Path(episode.verification_frame_path)
        if not frame.is_file():
            raise ReviewError("verification frame is missing")
        seed = episode.original_bbox
        point = None
        if episode.point_source and episode.point_source.get("ok"):
            point = episode.point_source.get("point_source")
        revision = int((self.state.get(episode_id) or {}).get("proposal_revision") or 0) + 1
        raw = self.engine.propose(frame_path=frame, seed_bbox=seed, point=point,
                                  frame_width=episode.source_width,
                                  frame_height=episode.source_height, revision=revision)
        cleaned = validate_proposals(raw, frame_width=episode.source_width,
                                     frame_height=episode.source_height)
        return self.state.store_proposals(episode, cleaned)

    def build_records(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for episode in self.data.required:
            decision = self.state.get(episode.episode_id)
            if decision is None:
                rows.append({
                    **{k: v for k, v in {
                        "episode_id": episode.episode_id,
                        "truth_target_id": episode.episode_id,
                        "camera_id": episode.camera_id,
                        "scene_version": episode.scene_version,
                        "source_file_id": episode.source_file_id,
                        "source_timestamp": episode.source_timestamp,
                        "source_width": episode.source_width,
                        "source_height": episode.source_height,
                        "verification_frame_path": episode.verification_frame_path,
                        "origin": episode.origin,
                        "original_bbox": list(episode.original_bbox)
                        if episode.original_bbox else None,
                        "original_point": dict(episode.original_point or {}) or None,
                    }.items()},
                    "original_location_type": "POINT"
                    if episode.origin == "manual_missing_target"
                    else ("BBOX" if episode.original_bbox else "UNKNOWN"),
                    "original_point_source": dict(episode.point_source or {}) or None,
                    "localization_decision": None,
                    "verified_bbox": None,
                    "location_type": None,
                    "localization_status": "NEEDS_RELOCALIZATION",
                    "proposal_source_type": None,
                    "proposal_revision": 0,
                    "training_localization_ready": False,
                    "review_status": "pending",
                    "review_schema_version": REVIEW_SCHEMA_VERSION,
                })
                continue
            record = {k: v for k, v in decision.items() if k != "proposals"}
            record["proposal_count"] = len(decision.get("proposals") or [])
            record.pop("selected_proposal_id", None)
            if decision.get("selected_proposal_id"):
                record["selected_proposal_id"] = decision["selected_proposal_id"]
            record["training_localization_ready"] = (
                record.get("localization_status") == TRAINING_READY_STATUS)
            rows.append(record)
        rows.sort(key=lambda r: (r["camera_id"], r["episode_id"]))
        return rows


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def build_summary(data: LocalizationInput, records: Sequence[Mapping[str, Any]],
                  state: ReviewState) -> dict[str, Any]:
    status_counts = {s: 0 for s in LOCALIZATION_STATUSES}
    decision_counts = {d: 0 for d in DECISIONS}
    origin_block: dict[str, dict[str, int]] = {
        o: {"total": 0, "verified": 0, "unresolved": 0} for o in ORIGINS}
    per_camera: dict[str, dict[str, int]] = {}
    proposals_required = 0
    first_pass = 0
    second_pass = 0
    failed = 0
    reviewed = 0

    for record in records:
        status = str(record.get("localization_status") or "NEEDS_RELOCALIZATION")
        status_counts[status] = status_counts.get(status, 0) + 1
        decision = record.get("localization_decision")
        if decision in decision_counts:
            decision_counts[decision] += 1
        if record.get("review_status") != "pending":
            reviewed += 1
        origin = str(record.get("origin") or "historical_litter")
        bucket = origin_block.setdefault(origin, {"total": 0, "verified": 0,
                                                  "unresolved": 0})
        bucket["total"] += 1
        if status == "VERIFIED_BBOX":
            bucket["verified"] += 1
        elif status in ("LOCALIZATION_UNRESOLVED", "TRUTH_REVIEW_REQUIRED"):
            bucket["unresolved"] += 1
        cam = per_camera.setdefault(str(record.get("camera_id")),
                                    {"required": 0, "verified": 0, "unresolved": 0})
        cam["required"] += 1
        if status == "VERIFIED_BBOX":
            cam["verified"] += 1
        elif status in ("LOCALIZATION_UNRESOLVED", "TRUTH_REVIEW_REQUIRED"):
            cam["unresolved"] += 1

        revision = int(record.get("proposal_revision") or 0)
        if revision >= 1:
            proposals_required += 1
            if decision == "PROPOSAL_SELECTED":
                if revision == 1:
                    first_pass += 1
                else:
                    second_pass += 1
            elif decision in ("UNRESOLVED", "TRUTH_REVIEW_REQUIRED"):
                failed += 1

    manual = [r for r in records if r.get("origin") == "manual_missing_target"]
    manual_verified = sum(1 for r in manual
                          if r.get("localization_status") == "VERIFIED_BBOX")
    ready = sum(1 for r in records if r.get("training_localization_ready"))
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "input": {
            "gold_artifact": str(data.gold_path),
            "gold_sha256": data.gold_sha256,
            "recovery_evidence_sha256": data.recovery_evidence_sha256,
            "required_episode_count": data.required_count,
        },
        "review": {"reviewed": reviewed, "pending": data.required_count - reviewed},
        "localization": {s: status_counts.get(s, 0) for s in LOCALIZATION_STATUSES},
        "decision": {d: decision_counts.get(d, 0) for d in DECISIONS},
        "origin": origin_block,
        "proposal": {
            "episodes_requiring_proposal": proposals_required,
            "proposal_first_pass_success": first_pass,
            "proposal_second_pass_success": second_pass,
            "proposal_failed": failed,
        },
        "manual_missing": {
            "manual_total": len(manual),
            "manual_verified": manual_verified,
            "manual_unresolved": len(manual) - manual_verified,
        },
        "per_camera": dict(sorted(per_camera.items())),
        "training_localization_ready": ready,
        "training_localization_ready_rule": (
            "truth_class=REQUIRED_LITTER AND source_recovery_status="
            "RECOVERED_SOURCE_NATIVE AND localization_status=VERIFIED_BBOX"),
        "unresolved_episode_ids": sorted(
            r["episode_id"] for r in records
            if r.get("localization_status") in ("LOCALIZATION_UNRESOLVED",
                                                "TRUTH_REVIEW_REQUIRED")),
        "boundaries": {
            "gold_modified": False,
            "recovery_evidence_modified": False,
            "truth_class_changed": False,
            "episodes_merged_or_split": False,
            "episode_id_changed": False,
            "training_tiles_generated": False,
            "annotation_complete_performed": False,
            "hard_negative_mining_started": False,
            "training_started": False,
            "detector_run": False,
            "sealed_accessed": False,
            "auto_verified_without_human": False,
        },
    }


def build_manifest(data: LocalizationInput, summary: Mapping[str, Any], *,
                   code_commit: str, generated_at: str, config: Mapping[str, Any],
                   artifact_root: Path, localization_path: Path,
                   review_state_path: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_at": generated_at,
        "code_commit": code_commit,
        "gold_input_sha256": data.gold_sha256,
        "recovery_evidence_sha256": data.recovery_evidence_sha256,
        "localizations_sha256": sha256_file(localization_path)
        if Path(localization_path).is_file() else "",
        "artifact_root": str(artifact_root),
        "config": dict(config),
        "outputs": {"localizations": str(localization_path),
                    "review_state": str(review_state_path)},
        "upstream_provenance": dict(provenance),
        "counts": {
            "required_episode_count": data.required_count,
            "verified_bbox": summary["localization"]["VERIFIED_BBOX"],
            "unresolved": summary["localization"]["LOCALIZATION_UNRESOLVED"],
            "truth_review_required": summary["localization"]["TRUTH_REVIEW_REQUIRED"],
            "training_localization_ready": summary["training_localization_ready"],
        },
        "boundaries": dict(summary["boundaries"]),
    }


def build_upstream_provenance(
    *, step1c0_reported_execution_commit: str, step1c0_manifest_commit: str,
    step1c0_evidence_commit: str, code_equivalence_verified: bool, note: str = "",
) -> dict[str, Any]:
    """Disambiguate the Step 1C-0 commit recorded in its MANIFEST (§27)."""
    return {
        "step1c0_reported_execution_commit": step1c0_reported_execution_commit,
        "step1c0_manifest_commit": step1c0_manifest_commit,
        "step1c0_evidence_commit": step1c0_evidence_commit,
        "step1c0_code_equivalence_verified": bool(code_equivalence_verified),
        "step1c0_provenance_mismatch_explained": (
            "Step 1C-0's MANIFEST records the git HEAD at the moment the final run wrote "
            "its evidence, which was the Step 1B Gold-evidence commit. The recovery code "
            "is byte-identical between that commit and the reported execution commit, so "
            "the recorded hash identifies the same code. Recorded here so no downstream "
            "step has to guess."),
        "note": note,
    }


def write_outputs(output_dir: Path, *, records: Sequence[Mapping[str, Any]],
                  summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    loc_path = output_dir / "localizations.jsonl"
    write_jsonl(loc_path, records)
    summary_path = output_dir / "SUMMARY.json"
    atomic_write_json(summary_path, summary)
    manifest_path = output_dir / "MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    return {"localizations": str(loc_path), "summary": str(summary_path),
            "manifest": str(manifest_path)}
