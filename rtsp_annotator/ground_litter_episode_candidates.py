"""Step 1A: rebuild historical Silver review cards into episode *candidates*.

Scope and hard boundaries (see
``docs/plans/2026-09-23-ground-litter-detector-feasibility-plan.md`` §10.1 and
``docs/plans/2026-09-23-ground-litter-step0b-evaluation-protocol.md`` §3):

* This module only *pre-clusters* existing human review cards to reduce repeated
  manual review.  ``episode_candidate_id`` is explicitly **not** an ``episode_id``.
* It is detector output blind: it never reads model predictions, and it never uses
  the candidate ``score`` field to rank or select anything.  Representative
  selection follows the auditable image/geometry/time priorities instead.
* It never writes to, or rewrites, the original Silver labels, images, timestamps,
  boxes or manifests.  Every artifact it produces is a new, deletable, rebuildable
  derivation.
* It is deliberately stdlib-only.  The repository virtualenv used for tests has no
  ``cv2``, and Step 1A performs no image processing, so importing the cv2-backed
  audit helpers would make the logic untestable for no benefit.  Visual embedding
  was intentionally not introduced for this first version (plan §10.1 allows
  time + space + box scale).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "ground_litter_episode_candidates_v1"
GENERATOR_VERSION = "step1a-1.0.0"

#: Labels treated as "this really is litter" (BOX_WRONG = litter, bad box).
POSITIVE_LABELS = ("LITTER", "BOX_WRONG")

#: Every grouping threshold lives here, never inline in the algorithm.
DEFAULT_CONFIG = {
    # Maximum real-world gap between two cards' observed intervals that still
    # allows them to be considered one continuous candidate appearance.
    # Measured data: intra-session review frames sit ~76 s apart, while the next
    # session on the same camera/day is ~2.9 h away, so 600 s sits in the empty
    # band between the two populations.
    "max_time_gap_seconds": 600.0,
    # Spatial compatibility is scale-relative: centres must be within
    # ratio * mean(box diagonal), with an absolute floor so that very small
    # targets still tolerate a few pixels of jitter.
    "min_center_distance_px": 24.0,
    "max_center_distance_ratio": 0.75,
    # Reject pairing a 6 px fragment with a 200 px object.
    "max_size_ratio": 3.0,
    # Near-threshold reporting band (does not affect grouping decisions).
    "near_threshold_factor": 1.25,
    # before/after evidence frames are sampled at centre -/+ this many seconds.
    "before_after_seconds": 2.0,
}

_TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GroupingConfig:
    """Centralised, serialisable grouping thresholds."""

    max_time_gap_seconds: float = 600.0
    min_center_distance_px: float = 24.0
    max_center_distance_ratio: float = 0.75
    max_size_ratio: float = 3.0
    near_threshold_factor: float = 1.25
    before_after_seconds: float = 2.0

    def __post_init__(self) -> None:
        for name in ("max_time_gap_seconds", "min_center_distance_px",
                     "max_center_distance_ratio", "max_size_ratio",
                     "near_threshold_factor", "before_after_seconds"):
            value = float(getattr(self, name))
            if not value > 0:
                raise ValueError(f"{name} must be positive")
        if self.max_size_ratio < 1.0:
            raise ValueError("max_size_ratio must be >= 1")
        if self.near_threshold_factor < 1.0:
            raise ValueError("near_threshold_factor must be >= 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_time_gap_seconds": self.max_time_gap_seconds,
            "min_center_distance_px": self.min_center_distance_px,
            "max_center_distance_ratio": self.max_center_distance_ratio,
            "max_size_ratio": self.max_size_ratio,
            "near_threshold_factor": self.near_threshold_factor,
            "before_after_seconds": self.before_after_seconds,
            "positive_labels": list(POSITIVE_LABELS),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GroupingConfig":
        known = set(DEFAULT_CONFIG)
        unknown = set(payload) - known - {"positive_labels"}
        if unknown:
            raise ValueError(f"unknown grouping config keys: {sorted(unknown)}")
        kwargs = {k: float(payload[k]) for k in known if k in payload}
        return cls(**kwargs)


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #


def parse_timestamp(text: str) -> datetime | None:
    """Tolerant parser for the two timestamp shapes present in the Silver data."""
    if not isinstance(text, str) or not text.strip():
        return None
    raw = text.strip()
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def box_center(bbox: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_diagonal(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def box_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class ReviewCard:
    """One historical human review card, normalised without altering its source."""

    card_id: str
    raw_review_id: str
    batch_key: str
    camera_id: str
    device_code: str
    label: str | None
    scene_version: str | None
    timestamp: datetime | None
    timestamp_text: str | None
    bbox: tuple[float, float, float, float] | None
    frame_id: str | None
    proposal_source: str | None
    source_file_id: str | None
    context_image: str | None
    crop_image: str | None
    before_image: str | None
    after_image: str | None
    context_image_exists: bool
    crop_image_exists: bool
    unresolved_reasons: tuple[str, ...] = ()

    @property
    def scene_key(self) -> str:
        return self.scene_version or "__scene_version_unavailable__"

    @property
    def is_positive(self) -> bool:
        return self.label in POSITIVE_LABELS

    @property
    def is_unresolved(self) -> bool:
        return bool(self.unresolved_reasons)

    @property
    def has_before_after(self) -> bool:
        return bool(self.before_image) and bool(self.after_image)

    def observed_interval(self, config: GroupingConfig) -> tuple[datetime, datetime] | None:
        """Card's observed time window; before/after frames widen it by +/- delta."""
        if self.timestamp is None:
            return None
        pad = timedelta(seconds=config.before_after_seconds) if self.has_before_after \
            else timedelta(0)
        return (self.timestamp - pad, self.timestamp + pad)

    def as_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "raw_review_id": self.raw_review_id,
            "batch_key": self.batch_key,
            "camera_id": self.camera_id,
            "device_code": self.device_code,
            "label": self.label,
            "scene_version": self.scene_version,
            "timestamp": self.timestamp_text,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "frame_id": self.frame_id,
            "proposal_source": self.proposal_source,
            "source_file_id": self.source_file_id,
            "context_image": self.context_image,
            "crop_image": self.crop_image,
            "before_image": self.before_image,
            "after_image": self.after_image,
            "has_before_after": self.has_before_after,
            "unresolved_reasons": list(self.unresolved_reasons),
        }


@dataclass(frozen=True)
class CleanGap:
    """Human/independent evidence that a position was observed empty for a while.

    Two cards separated by such an interval must not be merged: the object was
    gone in between, so the same position means two different physical events.
    """

    camera_id: str
    start: datetime
    end: datetime
    source: str = ""

    def overlaps(self, left: datetime, right: datetime) -> bool:
        return self.start <= right and self.end >= left


def load_clean_gaps(payload: Mapping[str, Any]) -> list[CleanGap]:
    rows = payload.get("clean_gaps") or []
    gaps: list[CleanGap] = []
    for index, row in enumerate(rows):
        camera_id = str(row.get("camera_id") or "").strip()
        if len(camera_id) != 5 or not camera_id.isdigit():
            raise ValueError(f"clean_gaps[{index}].camera_id must be 5 digits")
        start = parse_timestamp(str(row.get("start") or ""))
        end = parse_timestamp(str(row.get("end") or ""))
        if start is None or end is None or end <= start:
            raise ValueError(f"clean_gaps[{index}] needs a valid start < end")
        gaps.append(CleanGap(camera_id, start, end, str(row.get("source") or "")))
    return gaps


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_review_batch(
    batch_key: str,
    directory: Path | str,
    *,
    config: GroupingConfig | None = None,
    require_positive_labels: bool = True,
) -> tuple[list[ReviewCard], dict[str, Any]]:
    """Load one ``review-data.json`` + ``reviews.json`` pair, read-only.

    Card identity is ``batch_key:raw_review_id`` because ``review_id`` is *not*
    globally unique across the historical batches (25 ids collide between formal
    day1/day2 with different boxes, and 3 between active day3/shadow).
    """
    config = config or GroupingConfig()
    root = Path(directory)
    data = json.loads((root / "review-data.json").read_text(encoding="utf-8"))
    reviews = json.loads((root / "reviews.json").read_text(encoding="utf-8"))
    label_map = reviews.get("reviews") if isinstance(reviews, dict) else None
    if not isinstance(label_map, dict):
        raise ValueError(f"{root}/reviews.json has no 'reviews' mapping")

    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{root}/review-data.json has no 'items' list")

    cards: list[ReviewCard] = []
    for row in items:
        raw_id = row.get("review_id")
        if not raw_id:
            continue
        raw_id = str(raw_id)
        label_row = label_map.get(raw_id) or {}
        label = label_row.get("label")
        device_code = str(row.get("device_code") or "")
        camera_id = device_code[-5:] if len(device_code) >= 5 else ""
        bbox_raw = row.get("bbox")
        bbox: tuple[float, float, float, float] | None = None
        if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
            try:
                vals = tuple(float(v) for v in bbox_raw)
            except (TypeError, ValueError):
                vals = None
            if vals is not None and vals[2] > vals[0] and vals[3] > vals[1]:
                bbox = (vals[0], vals[1], vals[2], vals[3])

        timestamp_text = row.get("timestamp")
        timestamp = parse_timestamp(timestamp_text) if isinstance(timestamp_text, str) else None

        context_image = row.get("context_image")
        crop_image = row.get("crop_image")
        context_exists = bool(context_image) and (root / str(context_image)).is_file()
        crop_exists = bool(crop_image) and (root / str(crop_image)).is_file()

        reasons: list[str] = []
        if timestamp is None:
            reasons.append("missing_or_unparsable_timestamp")
        if bbox is None:
            reasons.append("missing_or_invalid_bbox")
        if not camera_id:
            reasons.append("missing_camera_id")

        cards.append(ReviewCard(
            card_id=f"{batch_key}:{raw_id}",
            raw_review_id=raw_id,
            batch_key=batch_key,
            camera_id=camera_id,
            device_code=device_code,
            label=str(label) if label is not None else None,
            scene_version=(str(row["scene_version"]) if row.get("scene_version") else None),
            timestamp=timestamp,
            timestamp_text=str(timestamp_text) if timestamp_text is not None else None,
            bbox=bbox,
            frame_id=str(row["frame_id"]) if row.get("frame_id") else None,
            proposal_source=str(row["source"]) if row.get("source") else None,
            source_file_id=str(row["file_id"]) if row.get("file_id") else None,
            context_image=str(context_image) if context_image else None,
            crop_image=str(crop_image) if crop_image else None,
            before_image=str(row["before_image"]) if row.get("before_image") else None,
            after_image=str(row["after_image"]) if row.get("after_image") else None,
            context_image_exists=context_exists,
            crop_image_exists=crop_exists,
            unresolved_reasons=tuple(reasons),
        ))

    diagnostics = {
        "batch_key": batch_key,
        "directory": str(root),
        "raw_card_count": len(cards),
        "dataset": data.get("dataset"),
        "dataset_fingerprint": data.get("fingerprint"),
    }
    return cards, diagnostics


def load_batches(
    batch_dirs: Mapping[str, Path | str], *, config: GroupingConfig | None = None,
) -> tuple[list[ReviewCard], list[dict[str, Any]]]:
    cards: list[ReviewCard] = []
    diagnostics: list[dict[str, Any]] = []
    for batch_key in sorted(batch_dirs):
        loaded, diag = load_review_batch(batch_key, batch_dirs[batch_key], config=config)
        cards.extend(loaded)
        diag["positive_card_count"] = sum(1 for c in loaded if c.is_positive)
        diagnostics.append(diag)
    return cards, diagnostics


def cards_fingerprint(cards: Iterable[ReviewCard]) -> str:
    """Canonical hash of the exact input evidence used by the grouping."""
    stable = [
        {
            "card_id": c.card_id,
            "device_code": c.device_code,
            "timestamp": c.timestamp_text,
            "bbox": list(c.bbox) if c.bbox is not None else None,
            "label": c.label,
        }
        for c in sorted(cards, key=lambda x: x.card_id)
    ]
    blob = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# grouping primitives
# --------------------------------------------------------------------------- #


def _interval_gap_seconds(
    a: tuple[datetime, datetime], b: tuple[datetime, datetime],
) -> float:
    """Seconds between two observed intervals; 0.0 when they overlap."""
    if a[1] >= b[0] and b[1] >= a[0]:
        return 0.0
    if a[1] < b[0]:
        return (b[0] - a[1]).total_seconds()
    return (a[0] - b[1]).total_seconds()


def _center_limit(a: ReviewCard, b: ReviewCard, config: GroupingConfig) -> float:
    assert a.bbox is not None and b.bbox is not None
    scale = (box_diagonal(a.bbox) + box_diagonal(b.bbox)) / 2.0
    return max(config.min_center_distance_px, config.max_center_distance_ratio * scale)


def _blocked_by_clean_gap(
    a: ReviewCard, b: ReviewCard, gaps: Sequence[CleanGap], config: GroupingConfig,
) -> bool:
    """A clean gap lying between the two cards forbids merging them."""
    if not gaps:
        return False
    interval_a = a.observed_interval(config)
    interval_b = b.observed_interval(config)
    if interval_a is None or interval_b is None:
        return False
    earlier, later = sorted((interval_a, interval_b), key=lambda iv: iv[1])
    gap_start, gap_end = earlier[1], later[0]
    if gap_start >= gap_end:
        return False  # observed intervals already touch/overlap
    return any(g.camera_id == a.camera_id and g.overlaps(gap_start, gap_end) for g in gaps)


def pair_compatible(
    a: ReviewCard, b: ReviewCard, config: GroupingConfig,
    clean_gaps: Sequence[CleanGap] = (),
) -> bool:
    """Whether two cards may belong to one automatic episode candidate."""
    if a.camera_id != b.camera_id:
        return False
    if a.scene_key != b.scene_key:
        return False
    if a.timestamp is None or b.timestamp is None:
        return False
    if a.bbox is None or b.bbox is None:
        return False
    interval_a = a.observed_interval(config)
    interval_b = b.observed_interval(config)
    if interval_a is None or interval_b is None:
        return False
    if _interval_gap_seconds(interval_a, interval_b) > config.max_time_gap_seconds:
        return False
    if _blocked_by_clean_gap(a, b, clean_gaps, config):
        return False
    center = _center_distance(a, b)
    if center > _center_limit(a, b, config):
        return False
    ratio = _size_ratio(a, b)
    if ratio > config.max_size_ratio:
        return False
    return True


def _center_distance(a: ReviewCard, b: ReviewCard) -> float:
    assert a.bbox is not None and b.bbox is not None
    ax, ay = box_center(a.bbox)
    bx, by = box_center(b.bbox)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _size_ratio(a: ReviewCard, b: ReviewCard) -> float:
    assert a.bbox is not None and b.bbox is not None
    da, db = box_diagonal(a.bbox), box_diagonal(b.bbox)
    lo, hi = (da, db) if da <= db else (db, da)
    return hi / lo if lo > 1e-9 else float("inf")


def _pair_pressure(a: ReviewCard, b: ReviewCard, config: GroupingConfig) -> float:
    """How close a pair is to the merge thresholds (1.0 == exactly at threshold)."""
    if a.bbox is None or b.bbox is None or a.timestamp is None or b.timestamp is None:
        return float("inf")
    interval_a = a.observed_interval(config)
    interval_b = b.observed_interval(config)
    if interval_a is None or interval_b is None:
        return float("inf")
    temporal = _interval_gap_seconds(interval_a, interval_b) / config.max_time_gap_seconds
    spatial = _center_distance(a, b) / _center_limit(a, b, config)
    size = _size_ratio(a, b) / config.max_size_ratio
    return max(temporal, spatial, size)


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #


@dataclass
class CandidateGroup:
    episode_candidate_id: str
    camera_id: str
    scene_version: str | None
    scene_key: str
    members: list[ReviewCard] = field(default_factory=list)
    ambiguity_reasons: list[str] = field(default_factory=list)
    grouping_method: str = "time_space_scale_complete_linkage_v1"

    @property
    def is_ambiguous(self) -> bool:
        return bool(self.ambiguity_reasons)


def _candidate_id(camera_id: str, scene_key: str, members: Sequence[ReviewCard]) -> str:
    ids = sorted(m.card_id for m in members)
    blob = "|".join([SCHEMA_VERSION, camera_id, scene_key, *ids])
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    safe = _SAFE_ID.sub("-", camera_id) or "cam"
    return f"ec-{safe}-{digest}"


def group_cards(
    cards: Sequence[ReviewCard],
    config: GroupingConfig | None = None,
    clean_gaps: Sequence[CleanGap] = (),
) -> tuple[list[CandidateGroup], list[ReviewCard]]:
    """Cluster positive cards into episode candidates.

    Uses **complete linkage** (a card must be compatible with *every* member of a
    group).  Single-linkage chaining is deliberately avoided: a chain of
    individually-close boxes can walk across a whole day and produce exactly the
    false merges this step is meant to prevent.

    Returns ``(groups, unresolved_cards)``.  Unresolved cards (missing timestamp or
    bbox) never join a group; each becomes its own flagged singleton so that no
    evidence is dropped.
    """
    config = config or GroupingConfig()
    positives = [c for c in cards if c.is_positive]
    groupable = [c for c in positives if not c.is_unresolved]
    unresolved = [c for c in positives if c.is_unresolved]

    ordered = sorted(
        groupable,
        key=lambda c: (c.camera_id, c.scene_key, c.timestamp, c.card_id),  # type: ignore[arg-type]
    )

    grouped: list[CandidateGroup] = []
    for card in ordered:
        placed = False
        for group in grouped:
            if group.camera_id != card.camera_id or group.scene_key != card.scene_key:
                continue
            if all(pair_compatible(card, member, config, clean_gaps) for member in group.members):
                group.members.append(card)
                placed = True
                break
        if not placed:
            grouped.append(CandidateGroup(
                episode_candidate_id="",
                camera_id=card.camera_id,
                scene_version=card.scene_version,
                scene_key=card.scene_key,
                members=[card],
            ))

    # Unresolved cards: explicit singletons, never merged, always flagged.
    for card in sorted(unresolved, key=lambda c: (c.camera_id, c.card_id)):
        grouped.append(CandidateGroup(
            episode_candidate_id="",
            camera_id=card.camera_id or "unknown",
            scene_version=card.scene_version,
            scene_key=card.scene_key,
            members=[card],
            ambiguity_reasons=["unresolved_member:" + ",".join(card.unresolved_reasons)],
            grouping_method="unresolved_fallback",
        ))

    for group in grouped:
        group.episode_candidate_id = _candidate_id(
            group.camera_id, group.scene_key, group.members)

    _annotate_ambiguity(grouped, config, clean_gaps)
    _assert_unique_ids(grouped)

    grouped.sort(key=lambda g: (_group_start(g), g.camera_id, g.episode_candidate_id))
    return grouped, sorted(unresolved, key=lambda c: (c.camera_id, c.card_id))


def _assert_unique_ids(groups: Sequence[CandidateGroup]) -> None:
    seen: dict[str, str] = {}
    for group in groups:
        key = group.episode_candidate_id
        signature = ",".join(sorted(m.card_id for m in group.members))
        if key in seen and seen[key] != signature:
            raise RuntimeError("episode_candidate_id collision across different members")
        seen[key] = signature
    # Identical member sets must not occur twice.
    signatures = [",".join(sorted(m.card_id for m in g.members)) for g in groups]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("duplicate candidate member sets produced")


def _annotate_ambiguity(
    groups: Sequence[CandidateGroup], config: GroupingConfig,
    clean_gaps: Sequence[CleanGap],
) -> None:
    by_bucket: dict[tuple[str, str], list[CandidateGroup]] = {}
    for group in groups:
        by_bucket.setdefault((group.camera_id, group.scene_key), []).append(group)

    for group in groups:
        reasons = list(group.ambiguity_reasons)
        if len(group.members) > 1:
            best_temporal = 0.0
            best_spatial = 0.0
            best_size = 1.0
            for i in range(len(group.members)):
                for j in range(i + 1, len(group.members)):
                    a, b = group.members[i], group.members[j]
                    ia, ib = a.observed_interval(config), b.observed_interval(config)
                    if ia and ib:
                        best_temporal = max(
                            best_temporal,
                            _interval_gap_seconds(ia, ib) / config.max_time_gap_seconds)
                    if a.bbox and b.bbox:
                        best_spatial = max(
                            best_spatial,
                            _center_distance(a, b) / _center_limit(a, b, config))
                        best_size = max(best_size, _size_ratio(a, b))
            if best_temporal > 0.7 or best_spatial > 0.7 or best_size > 0.7 * config.max_size_ratio:
                reasons.append("borderline_within_threshold")
        if any(m.source_file_id is None for m in group.members):
            reasons.append("lineage_source_file_id_unavailable")
        group.ambiguity_reasons = sorted(set(reasons))

    # Cross-group diagnostics: complete linkage may have split a pair that would
    # individually have been mergeable, or the pair may be just outside threshold.
    for bucket_groups in by_bucket.values():
        for i in range(len(bucket_groups)):
            for j in range(i + 1, len(bucket_groups)):
                left, right = bucket_groups[i], bucket_groups[j]
                best = float("inf")
                for a in left.members:
                    for b in right.members:
                        if a.is_unresolved or b.is_unresolved:
                            continue
                        if a.scene_key != b.scene_key or a.camera_id != b.camera_id:
                            continue
                        best = min(best, _pair_pressure(a, b, config))
                if best <= 1.0:
                    reason = "complete_linkage_split_possible_false_split"
                elif best <= config.near_threshold_factor:
                    reason = "near_threshold_pair_possible_false_split"
                else:
                    continue
                if reason not in left.ambiguity_reasons:
                    left.ambiguity_reasons = sorted(set(left.ambiguity_reasons + [reason]))
                if reason not in right.ambiguity_reasons:
                    right.ambiguity_reasons = sorted(set(right.ambiguity_reasons + [reason]))


def _group_start(group: CandidateGroup) -> datetime:
    stamps = [m.timestamp for m in group.members if m.timestamp is not None]
    return min(stamps) if stamps else datetime.min


def _group_end(group: CandidateGroup) -> datetime:
    stamps = [m.timestamp for m in group.members if m.timestamp is not None]
    return max(stamps) if stamps else datetime.min


# --------------------------------------------------------------------------- #
# representative selection
# --------------------------------------------------------------------------- #


def select_representative(group: CandidateGroup) -> ReviewCard:
    """Auditable priority order; predictor score is never consulted.

    1. valid context/crop images present
    2. complete bbox + crop evidence
    3. larger target area
    4. closest to the group's median time
    5. stable card_id tie-break
    """
    stamps = sorted(m.timestamp for m in group.members if m.timestamp is not None)
    median = stamps[len(stamps) // 2] if stamps else None

    def sort_key(card: ReviewCard) -> tuple:
        images_valid = card.context_image_exists and card.crop_image_exists
        geometry_complete = card.bbox is not None and card.crop_image is not None
        area = box_area(card.bbox) if card.bbox is not None else -1.0
        if median is not None and card.timestamp is not None:
            distance = abs((card.timestamp - median).total_seconds())
        else:
            distance = float("inf")
        return (
            0 if images_valid else 1,
            0 if geometry_complete else 1,
            -area,
            distance,
            card.card_id,
        )

    return sorted(group.members, key=sort_key)[0]


# --------------------------------------------------------------------------- #
# artifact construction
# --------------------------------------------------------------------------- #


def _relative_assets(card: ReviewCard) -> dict[str, Any]:
    return {
        "context_image": card.context_image,
        "crop_image": card.crop_image,
        "before_image": card.before_image,
        "after_image": card.after_image,
    }


def group_to_record(
    group: CandidateGroup, config: GroupingConfig,
    card_lookup: Mapping[str, ReviewCard],
) -> dict[str, Any]:
    representative = select_representative(group)
    members = sorted(group.members, key=lambda c: (c.timestamp or datetime.min, c.card_id))
    stamps = [m.timestamp for m in members if m.timestamp is not None]
    start = min(stamps) if stamps else None
    end = max(stamps) if stamps else None

    centers = [box_center(m.bbox) for m in members if m.bbox is not None]
    diagonals = [box_diagonal(m.bbox) for m in members if m.bbox is not None]
    center_spread = 0.0
    if len(centers) > 1:
        center_spread = max(
            ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            for i, a in enumerate(centers) for b in centers[i + 1:]
        )
    size_spread = (max(diagonals) - min(diagonals)) if diagonals else 0.0

    # Report the *binding* pair so that spread and limit are directly comparable:
    # complete linkage guarantees every pair is compatible, i.e. pressure <= 1.0.
    pairwise_pressure = 0.0
    binding_limit: float | None = None
    binding_distance: float | None = None
    if len(members) > 1:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                pressure = _pair_pressure(a, b, config)
                if a.bbox is not None and b.bbox is not None:
                    distance = _center_distance(a, b)
                    limit = _center_limit(a, b, config)
                else:
                    distance = limit = None
                if pressure >= pairwise_pressure:
                    pairwise_pressure = pressure
                    binding_limit = limit
                    binding_distance = distance

    time_span = (end - start).total_seconds() if start and end else 0.0
    label_counts: dict[str, int] = {}
    for member in members:
        key = member.label or "UNKNOWN"
        label_counts[key] = label_counts.get(key, 0) + 1

    source_files = sorted({m.source_file_id for m in members if m.source_file_id})
    frame_ids = sorted({m.frame_id for m in members if m.frame_id})

    predecessor: dict[str, Any] = {}
    if len(members) > 1:
        predecessor = {
            "time_gap_seconds": round(time_span, 3),
            "bbox_center_spread_px": round(center_spread, 3),
            "bbox_diagonal_spread_px": round(size_spread, 3),
            "max_time_gap_threshold_seconds": config.max_time_gap_seconds,
            "max_center_distance_ratio": config.max_center_distance_ratio,
            "max_size_ratio": config.max_size_ratio,
        }

    return {
        "episode_candidate_id": group.episode_candidate_id,
        "episode_id": None,
        "episode_id_status": "NOT_ASSIGNED_CANDIDATE_ONLY",
        "camera_id": group.camera_id,
        "scene_version": group.scene_version,
        "scene_version_status": (
            "resolved" if group.scene_version else "unavailable_in_silver_review_cards"),
        "start_timestamp": start.strftime("%Y-%m-%d %H:%M:%S") if start else None,
        "end_timestamp": end.strftime("%Y-%m-%d %H:%M:%S") if end else None,
        "member_review_card_ids": [m.card_id for m in members],
        "member_raw_review_ids": [m.raw_review_id for m in members],
        "member_batches": sorted({m.batch_key for m in members}),
        "member_count": len(members),
        "source_file_ids": source_files,
        "source_file_ids_status": (
            "resolved" if source_files else "unavailable_in_silver_review_cards"),
        "source_frame_ids": frame_ids,
        "representative_card_id": representative.card_id,
        "representative_raw_review_id": representative.raw_review_id,
        "representative_frame_id": representative.frame_id,
        "representative_frame": representative.context_image,
        "representative_crop_image": representative.crop_image,
        "representative_bbox": list(representative.bbox) if representative.bbox else None,
        "representative_selection_basis": [
            "valid_context_and_crop_image",
            "complete_bbox_and_crop",
            "larger_target_area",
            "closest_to_group_median_time",
            "stable_card_id_tiebreak",
        ],
        "representative_detector_score_used": False,
        "candidate_grouping_evidence": {
            "grouping_method": group.grouping_method,
            "time_span_seconds": round(time_span, 3),
            "bbox_center_spread_px": round(center_spread, 3),
            "bbox_diagonal_spread_px": round(size_spread, 3),
            # Limit of the *binding* pair, so spread and limit are comparable.
            "center_spread_limit_px": (
                round(binding_limit, 3) if binding_limit is not None else None),
            "binding_pair_center_distance_px": (
                round(binding_distance, 3) if binding_distance is not None else None),
            "binding_pair_threshold_pressure": (
                round(pairwise_pressure, 4) if len(members) > 1 else None),
            "complete_linkage_pressure_within_limits": (
                pairwise_pressure <= 1.0 if len(members) > 1 else True),
            "uses_visual_embedding": False,
            "uses_detector_confidence": False,
            "thresholds": config.as_dict(),
            "pairwise_summary": predecessor,
        },
        "original_labels_summary": {
            "LITTER": label_counts.get("LITTER", 0),
            "BOX_WRONG": label_counts.get("BOX_WRONG", 0),
            "other": {k: v for k, v in sorted(label_counts.items())
                      if k not in ("LITTER", "BOX_WRONG")},
            "total": len(members),
        },
        "candidate_ambiguous": group.is_ambiguous,
        "ambiguity_reasons": list(group.ambiguity_reasons),
        "review_state": "auto_candidate_pending_human_episode_identity",
        "lineage": {
            "review_cards": [
                {
                    "card_id": m.card_id,
                    "raw_review_id": m.raw_review_id,
                    "batch_key": m.batch_key,
                    "label": m.label,
                    "timestamp": m.timestamp_text,
                    "frame_id": m.frame_id,
                    "source_file_id": m.source_file_id,
                    "proposal_source": m.proposal_source,
                    "bbox": list(m.bbox) if m.bbox else None,
                    "assets": _relative_assets(m),
                }
                for m in members
            ],
        },
    }


def build_summary(
    groups: Sequence[CandidateGroup],
    records: Sequence[Mapping[str, Any]],
    cards: Sequence[ReviewCard],
    *,
    config: GroupingConfig,
    unresolved: Sequence[ReviewCard],
    batch_diagnostics: Sequence[Mapping[str, Any]],
    clean_gap_evidence_used: bool,
) -> dict[str, Any]:
    positives = [c for c in cards if c.is_positive]
    label_counts: dict[str, int] = {}
    for card in cards:
        key = card.label or "UNKNOWN"
        label_counts[key] = label_counts.get(key, 0) + 1

    group_sizes = sorted(len(g.members) for g in groups)
    median_size = 0.0
    if group_sizes:
        mid = len(group_sizes) // 2
        median_size = float(group_sizes[mid]) if len(group_sizes) % 2 \
            else (group_sizes[mid - 1] + group_sizes[mid]) / 2.0

    per_camera: dict[str, int] = {}
    per_camera_ambiguous: dict[str, int] = {}
    for group in groups:
        per_camera[group.camera_id] = per_camera.get(group.camera_id, 0) + 1
        if group.is_ambiguous:
            per_camera_ambiguous[group.camera_id] = \
                per_camera_ambiguous.get(group.camera_id, 0) + 1

    unresolved_ids = {c.card_id for c in unresolved}
    filtered = [c for c in positives if c.card_id in unresolved_ids]

    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "input": {
            "total_cards": len(cards),
            "label_counts": dict(sorted(label_counts.items())),
            "LITTER_cards": label_counts.get("LITTER", 0),
            "BOX_WRONG_cards": label_counts.get("BOX_WRONG", 0),
            "positive_cards_grouped_input": len(positives),
            "batches": list(batch_diagnostics),
        },
        "output": {
            "candidate_group_count": len(groups),
            "singleton_group_count": sum(1 for g in groups if len(g.members) == 1),
            "multi_card_group_count": sum(1 for g in groups if len(g.members) > 1),
            "max_group_size": max(group_sizes) if group_sizes else 0,
            "median_group_size": median_size,
            "candidate_count_per_camera": dict(sorted(per_camera.items())),
            "ambiguous_group_count": sum(1 for g in groups if g.is_ambiguous),
            "ambiguous_group_count_per_camera": dict(sorted(per_camera_ambiguous.items())),
            "filtered_or_unresolved_card_count": len(filtered),
            "unresolved_card_ids": [c.card_id for c in filtered],
            "cards_in_groups": sum(len(g.members) for g in groups),
            "members_in_multi_card_groups": sum(
                len(g.members) for g in groups if len(g.members) > 1),
            "duplicate_card_memberships": 0,
        },
        "interpretation": {
            "candidate_independent_events_estimate": len(groups),
            "statement": (
                "This is an automatic episode-candidate count, NOT the true Gold "
                "episode count.  Final physical episode identity requires human review; "
                "same location does not imply same physical object."),
            "gold_episode_count": None,
            "detector_outputs_consulted": False,
            "clean_gap_evidence_used": clean_gap_evidence_used,
        },
        "config": config.as_dict(),
    }


def build_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    cards: Sequence[ReviewCard],
    batch_diagnostics: Sequence[Mapping[str, Any]],
    config: GroupingConfig,
    git_commit: str,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "git_commit": git_commit,
        "generated_at": generated_at,
        "config": config.as_dict(),
        "input_manifest": {
            "batches": list(batch_diagnostics),
            "input_cards_sha256": cards_fingerprint(cards),
            "card_count": len(cards),
        },
        "outputs": {
            "episode_candidates_jsonl": "episode_candidates.jsonl",
            "summary_json": "SUMMARY.json",
            "candidate_count": len(records),
        },
        "boundaries": {
            "episode_candidate_id_is_not_episode_id": True,
            "gold_truth_claimed": False,
            "original_silver_modified": False,
            "sealed_test_accessed": False,
            "detector_outputs_consulted": False,
            "visual_embedding_used": False,
            "deletable_and_rebuildable": True,
        },
        "known_input_limitations": [
            "review_id is not globally unique across historical batches; card_id is "
            "batch-qualified to avoid silent collisions.",
            "800 of 2460 cards (active_day3, holdout, shadow_20260921) carry no "
            "source file_id; lineage for those stops at frame_id + timestamp.",
            "no historical card records a scene_version; grouping therefore cannot "
            "separate scene versions for this dataset.",
            "220 pilot_01030 cards have no before/after frames.",
        ],
    }


def git_commit_of(repo_root: Path) -> str:
    """Best-effort commit id without importing gitpython; returns '' when unknown."""
    try:
        head = (repo_root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if head.startswith("ref: "):
        ref = head[5:].strip()
        try:
            return (repo_root / ".git" / ref).read_text(encoding="utf-8").strip()
        except OSError:
            packed = repo_root / ".git" / "packed-refs"
            try:
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(" " + ref):
                        return line.split(" ", 1)[0]
            except OSError:
                return ""
        return ""
    return head


def write_artifacts(
    output_dir: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "episode_candidates.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = output_dir / "SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return {
        "episode_candidates_jsonl": str(jsonl_path),
        "summary_json": str(summary_path),
        "manifest_json": str(manifest_path),
    }
