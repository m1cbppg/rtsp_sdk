"""Ground Litter Historical Active Mining + Rapid Dataset v2 — pure logic core.

This module is deliberately free of cv2/torch/ultralytics and of any network or
filesystem side effect beyond JSON/JSONL helpers.  It implements the decision
logic that the rest of the v2 pipeline shares:

* the frozen 7-day time split (D1-D5 TRAIN / D6 DEV / D7 FINAL);
* deterministic, seeded 5-minute window selection with hour diversity;
* the two-layer identity model (``episode_id`` then ``review_group_id``);
* representative-frame selection (normal / extra hard states);
* multi-dimension diversity ordering of the first review batches.

Nothing here decides model quality.  It only decides *what a human is asked to
look at*, which is the point of Dataset v2: one review unit is one unique
Required litter instance or one independent hard-negative cluster, never a
near-duplicate frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "ground-litter-historical-v2"

#: Only these four cameras participate.  01028 is deferred (severe occlusion).
CAMERAS: tuple[str, ...] = ("01021", "01022", "01027", "01030")

#: Business device codes are the 15-digit site id + the 5-digit camera id.
DEVICE_CODE_PREFIX = "441802090313220"

#: The absolute, frozen time split.  Recorded so that no later step can move a
#: date after seeing a model result.
DAY_SPLIT: dict[str, str] = {
    "2026-09-23": "TRAIN",
    "2026-09-24": "TRAIN",
    "2026-09-25": "TRAIN",
    "2026-09-26": "TRAIN",
    "2026-09-27": "TRAIN",
    "2026-09-28": "DEV",
    "2026-09-29": "FINAL",
}

TRAIN_DAYS: tuple[str, ...] = tuple(d for d, s in DAY_SPLIT.items() if s == "TRAIN")
DEV_DAYS: tuple[str, ...] = tuple(d for d, s in DAY_SPLIT.items() if s == "DEV")
FINAL_DAYS: tuple[str, ...] = tuple(d for d, s in DAY_SPLIT.items() if s == "FINAL")

#: Day-time buckets, camera-local clock hours [start, end).
DAYTIME_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("early", 6, 10),
    ("mid", 10, 14),
    ("late", 14, 18),
)
DAYTIME_START_HOUR = 6
DAYTIME_END_HOUR = 18

#: Windows per camera / day for each split.
TRAIN_WINDOWS_PER_DAY = 3          # one per early/mid/late bucket
DEV_WINDOWS_PER_CAMERA = 4
FINAL_WINDOWS_PER_CAMERA = 4
EXPECTED_WINDOW_TOTAL = (
    len(CAMERAS) * len(TRAIN_DAYS) * TRAIN_WINDOWS_PER_DAY
    + len(CAMERAS) * DEV_WINDOWS_PER_CAMERA
    + len(CAMERAS) * FINAL_WINDOWS_PER_CAMERA
)  # 4*5*3 + 4*4 + 4*4 = 92

#: Day-1 download budget: one window per camera / train day.
DAY1_TRAIN_WINDOWS_PER_CAMERA_DAY = 1
#: Buckets rotate across train days so the first batch is not all the same hour.
DAY1_BUCKET_ROTATION: tuple[str, ...] = ("early", "mid", "late", "early", "mid")

#: Coarse scan: three sequential source-native frames per 5-minute window.
COARSE_OFFSETS: tuple[float, ...] = (30.0, 150.0, 270.0)
#: Dense scan for windows that earn it (one frame every 15 s).
DENSE_OFFSETS: tuple[float, ...] = tuple(float(v) for v in range(15, 300, 15))

FPS = 25.0
SOURCE_WIDTH = 2560
SOURCE_HEIGHT = 1440

#: Production-scale tiling contract.  Never resized.
TILE = 640
STRIDE = 512
TILE_OVERLAP = TILE - STRIDE           # 128
NMS_IOU = 0.50
CROSS_MODEL_MATCH_IOU = 0.30
#: Both models are stored loosely; the two confidences are never compared.
CANDIDATE_CONF_FLOOR = 0.01
#: "near the real working threshold" band used to tag hard-negative candidates.
YOLO_WORKING_THRESHOLD = 0.20
YOLO_WORKING_BAND = (0.12, 0.40)
#: Both models are stored loosely at 0.01, but spec §6 treats "Turhancan 极低分"
#: as low priority, so only observations at or above this floor are admitted to
#: the human review queue.  Nothing is discarded from the candidate table.
REVIEW_MIN_CONFIDENCE = 0.15
#: P1 (student miss) needs a plausible semantic score, not a 0.02 ghost box.
P1_MIN_CONFIDENCE = 0.25

#: Identity contract (spec §9.1).
MAX_GAP_DENSE_SECONDS = 90.0
MAX_GAP_COARSE_SECONDS = 360.0
RADIUS_FRACTION = 0.6
RADIUS_MIN_PX = 8.0
RADIUS_MAX_PX = 32.0
SIZE_RATIO_MIN = 1.0 / 3.0
SIZE_RATIO_MAX = 3.0
AMBIGUITY_DISTANCE_RATIO = 0.80
#: review_group linkage is deliberately looser than episode association.
REVIEW_GROUP_RADIUS_FACTOR = 2.0
REVIEW_GROUP_APPEARANCE_MAX_DISTANCE = 0.35

#: Review-batch sizing (spec §12 / §19).
FIRST_BATCH_SIZE = 40                 # inside the required 30..50 band
FULL_QUEUE_SIZE = 150
BATCH_SPLIT = (50, 50, 50)
RANDOM_RESERVE_FRACTION = 0.10

#: Blind mining (spec §17).
BLIND_FRAMES_PER_CAMERA = 8
BLIND_TRULY_RANDOM_FRACTION = 0.5

DEFAULT_SEED = "ground-litter-historical-v2-20260929"

SIZE_CLASSES: tuple[tuple[str, float, float], ...] = (
    ("tiny", 0.0, 16.0),
    ("small", 16.0, 32.0),
    ("large", 32.0, float("inf")),
)
LOCATION_GRID_COLS = 4
LOCATION_GRID_ROWS = 3
COLOR_BINS = 8


class HistoricalError(RuntimeError):
    """Any v2 pipeline contract violation."""


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(target)
    return count


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        return []
    rows: list[dict] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(*parts: Any) -> str:
    """Deterministic ordering key that is stable across platforms."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def seeded_rng(seed: str, *parts: Any) -> random.Random:
    return random.Random(stable_rank(seed, *parts))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bbox_short_side(box: Sequence[float]) -> float:
    return float(min(abs(box[2] - box[0]), abs(box[3] - box[1])))


def bbox_diagonal(box: Sequence[float]) -> float:
    return float(math.hypot(box[2] - box[0], box[3] - box[1]))


def bbox_center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def association_radius(box: Sequence[float]) -> float:
    """``r = clamp(0.6 * sqrt(w*h), 8, 32)`` in source pixels (spec §9.1)."""
    width = abs(float(box[2]) - float(box[0]))
    height = abs(float(box[3]) - float(box[1]))
    return clamp(RADIUS_FRACTION * math.sqrt(max(0.0, width * height)),
                 RADIUS_MIN_PX, RADIUS_MAX_PX)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    x0, y0 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x1, y1 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_union(boxes: Sequence[Sequence[float]]) -> list[float]:
    xs0 = [float(b[0]) for b in boxes]
    ys0 = [float(b[1]) for b in boxes]
    xs1 = [float(b[2]) for b in boxes]
    ys1 = [float(b[3]) for b in boxes]
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def median(values: Sequence[float]) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise HistoricalError("median of empty sequence")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def size_class(box: Sequence[float]) -> str:
    side = math.sqrt(max(0.0, abs(box[2] - box[0]) * abs(box[3] - box[1])))
    for name, low, high in SIZE_CLASSES:
        if low <= side < high:
            return name
    return SIZE_CLASSES[-1][0]


def buffer_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """0.0 identical .. 1.0 maximally different.  Used only for state flags."""
    if len(a) != len(b) or not a:
        return 1.0
    total = 0.0
    for left, right in zip(a, b):
        total += abs(float(left) - float(right))
    return total / (len(a) * 255.0)


# --------------------------------------------------------------------------- #
# 1. frozen time split and window selection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RecordingSlot:
    """One ~5-minute PS as offered by the replay listing endpoint."""

    file_id: str
    record_start: str          # "YYYY-MM-DD HH:MM:SS" camera-local
    record_end: str
    file_size: int | None = None
    file_name: str = ""

    @property
    def date(self) -> str:
        return self.record_start[:10]

    @property
    def start_hour(self) -> int:
        return int(self.record_start[11:13])

    @property
    def start_minute_of_day(self) -> int:
        return int(self.record_start[11:13]) * 60 + int(self.record_start[14:16])

    @property
    def duration_seconds(self) -> float:
        return _seconds_between(self.record_start, self.record_end)

    def as_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "file_name": self.file_name,
            "record_start": self.record_start,
            "record_end": self.record_end,
            "file_size": self.file_size,
        }


def _seconds_between(start: str, end: str) -> float:
    from datetime import datetime

    fmt = "%Y-%m-%d %H:%M:%S"
    return (datetime.strptime(end, fmt) - datetime.strptime(start, fmt)).total_seconds()


def bucket_of(hour: int) -> str:
    for name, low, high in DAYTIME_BUCKETS:
        if low <= hour < high:
            return name
    return "outside"


def select_windows(
    slots_by_camera_day: Mapping[tuple[str, str], Sequence[RecordingSlot]], *,
    seed: str = DEFAULT_SEED, day7_usable_end: str | None = None,
) -> list[dict[str, Any]]:
    """Pick the frozen 92 windows deterministically (spec §2/§3).

    ``slots_by_camera_day`` maps ``(camera_id, date)`` to the source PS listing.
    ``day7_usable_end`` is the camera-local ``HH:MM:SS`` boundary for a partial
    final day; windows starting at or after it are never selected.
    """
    selected: list[dict[str, Any]] = []
    used_hours: dict[str, dict[int, int]] = {}

    for camera in CAMERAS:
        for date in DAY_SPLIT:
            split = DAY_SPLIT[date]
            slots = [s for s in slots_by_camera_day.get((camera, date), [])
                     if DAYTIME_START_HOUR <= s.start_hour < DAYTIME_END_HOUR]
            if date in FINAL_DAYS and day7_usable_end:
                slots = [s for s in slots if s.record_start[11:19] < day7_usable_end]
            slots = sorted(slots, key=lambda s: (s.record_start, s.file_id))

            if split == "TRAIN":
                for bucket, _, _ in DAYTIME_BUCKETS:
                    pool = [s for s in slots if bucket_of(s.start_hour) == bucket]
                    if not pool:
                        continue
                    chosen = _choose(pool, seed, camera, date, bucket, used_hours)
                    selected.append(_window_row(camera, date, split, bucket, chosen,
                                                purpose="train", seed=seed))
            else:
                purpose = "dev" if split == "DEV" else "final"
                count = DEV_WINDOWS_PER_CAMERA if split == "DEV" else FINAL_WINDOWS_PER_CAMERA
                chosen_rows = _choose_many(slots, count, seed, camera, date, purpose, used_hours)
                for chosen in chosen_rows:
                    selected.append(_window_row(camera, date, split, bucket_of(chosen.start_hour),
                                                chosen, purpose=purpose, seed=seed))

    expected = EXPECTED_WINDOW_TOTAL
    if len(selected) != expected:
        # A short day is a real data limitation, not something to paper over.
        raise HistoricalError(
            f"window selection produced {len(selected)} rows, expected {expected}; "
            "the listing was incomplete for at least one camera/day"
        )
    selected.sort(key=lambda row: (row["date"], row["camera_id"], row["start_time"]))
    return selected


def _choose(pool: Sequence[RecordingSlot], seed: str, camera: str, date: str,
            bucket: str, used_hours: dict[str, dict[int, int]]) -> RecordingSlot:
    rng = seeded_rng(seed, camera, date, bucket)
    candidates = list(pool)
    rng.shuffle(candidates)
    history = used_hours.setdefault(f"{camera}|{bucket}", {})
    # Prefer the least-used hour-of-day so different dates do not always land on
    # the same clock slot; the seeded shuffle breaks ties.
    picked = min(candidates, key=lambda slot: history.get(slot.start_hour, 0))
    history[picked.start_hour] = history.get(picked.start_hour, 0) + 1
    return picked


def _choose_many(pool: Sequence[RecordingSlot], count: int, seed: str, camera: str,
                 date: str, purpose: str,
                 used_hours: dict[str, dict[int, int]]) -> list[RecordingSlot]:
    """Pick ``count`` windows spanning as many buckets as possible."""
    rng = seeded_rng(seed, camera, date, purpose)
    candidates = list(pool)
    rng.shuffle(candidates)
    history = used_hours.setdefault(f"{camera}|{purpose}", {})
    chosen_ids: set[str] = set()
    chosen: list[RecordingSlot] = []
    seen_buckets: set[str] = set()
    for slot in candidates:
        if len(chosen) >= count:
            break
        bucket = bucket_of(slot.start_hour)
        if bucket in seen_buckets and len(chosen) + (len(DAYTIME_BUCKETS) - len(seen_buckets)) <= count:
            continue
        chosen.append(slot)
        chosen_ids.add(slot.file_id)
        seen_buckets.add(bucket)
    for slot in candidates:
        if len(chosen) >= count:
            break
        if slot.file_id not in chosen_ids:
            chosen.append(slot)
            chosen_ids.add(slot.file_id)
    for slot in chosen:
        history[slot.start_hour] = history.get(slot.start_hour, 0) + 1
    return chosen


def _window_row(camera: str, date: str, split: str, bucket: str, slot: RecordingSlot,
                *, purpose: str, seed: str) -> dict[str, Any]:
    window_id = f"{camera}_{date}_{slot.record_start[11:16].replace(':', '')}"
    return {
        "schema_version": SCHEMA_VERSION,
        "window_id": window_id,
        "camera_id": camera,
        "device_code": DEVICE_CODE_PREFIX + camera,
        "date": date,
        "day_split": split,
        "purpose": purpose,
        "start_time": slot.record_start,
        "end_time": slot.record_end,
        "selection_bucket": bucket,
        "selection_seed": seed,
        "selection_reason": f"{purpose}:{bucket}:seeded-5min-window",
        "source_file_id": slot.file_id,
        "source_file_name": slot.file_name,
        "source_file_size": slot.file_size,
        "nominal_duration_seconds": slot.duration_seconds,
        "download_status": "pending",
        "local_path": None,
        "source_sha256": None,
        "reason_downloaded": None,
        "extraction_status": "pending",
        "frame_count": 0,
    }


def manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash only the frozen selection fields so download state cannot move it."""
    frozen = (
        "window_id", "camera_id", "date", "day_split", "purpose",
        "start_time", "end_time", "selection_bucket", "selection_seed",
        "source_file_id",
    )
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r["window_id"]):
        digest.update(json.dumps({k: row.get(k) for k in frozen},
                                 ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def day1_selection(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Window ids for the Day-1 download batch: one per camera / TRAIN day."""
    chosen: list[str] = []
    for date in TRAIN_DAYS:
        index = TRAIN_DAYS.index(date) % len(DAY1_BUCKET_ROTATION)
        bucket = DAY1_BUCKET_ROTATION[index]
        for camera in CAMERAS:
            match = [r for r in rows
                     if r["camera_id"] == camera and r["date"] == date
                     and r["selection_bucket"] == bucket and r["purpose"] == "train"]
            if match:
                chosen.append(match[0]["window_id"])
    return chosen


# --------------------------------------------------------------------------- #
# 2. two-layer identity
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Observation:
    """A model-agnostic litter observation inside one source-native frame."""

    observation_id: str
    camera_id: str
    frame_id: str
    window_id: str
    timestamp: float                      # epoch seconds, camera-local wall clock
    bbox_xyxy: list[float]
    sampling: str = "coarse"              # "coarse" | "dense"
    confidence_by_source: dict[str, float] = field(default_factory=dict)
    class_name_by_source: dict[str, str] = field(default_factory=dict)
    appearance: list[float] | None = None  # mean RGB of the crop, optional
    background_key: str | None = None
    tile_boundary_truncated: bool = False
    covered_location: bool = False
    covered_appearance: bool = False
    duplicate_background: bool = False

    @property
    def source_label(self) -> str:
        sources = set(self.confidence_by_source)
        if sources == {"turhancan", "yolo"}:
            return "both"
        if sources == {"turhancan"}:
            return "turhancan_only"
        if sources == {"yolo"}:
            return "yolo_only"
        return "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "window_id": self.window_id,
            "timestamp": self.timestamp,
            "bbox_xyxy": list(self.bbox_xyxy),
            "sampling": self.sampling,
            "confidence_by_source": dict(self.confidence_by_source),
            "class_name_by_source": dict(self.class_name_by_source),
            "appearance": list(self.appearance) if self.appearance else None,
            "background_key": self.background_key,
            "tile_boundary_truncated": self.tile_boundary_truncated,
            "source_label": self.source_label,
        }

    @staticmethod
    def from_dict(row: Mapping[str, Any]) -> "Observation":
        return Observation(
            observation_id=row["observation_id"],
            camera_id=row["camera_id"],
            frame_id=row["frame_id"],
            window_id=row.get("window_id", ""),
            timestamp=float(row["timestamp"]),
            bbox_xyxy=[float(v) for v in row["bbox_xyxy"]],
            sampling=row.get("sampling", "coarse"),
            confidence_by_source={k: float(v) for k, v in (row.get("confidence_by_source") or {}).items()},
            class_name_by_source=dict(row.get("class_name_by_source") or {}),
            appearance=list(row["appearance"]) if row.get("appearance") else None,
            background_key=row.get("background_key"),
            tile_boundary_truncated=bool(row.get("tile_boundary_truncated", False)),
            covered_location=bool(row.get("covered_location", False)),
            covered_appearance=bool(row.get("covered_appearance", False)),
            duplicate_background=bool(row.get("duplicate_background", False)),
        )


@dataclass(slots=True)
class Episode:
    episode_id: str
    camera_id: str
    observation_ids: list[str]
    start_timestamp: float
    end_timestamp: float
    median_bbox: list[float]
    ambiguous_parents: list[str] = field(default_factory=list)
    clean_gap_before: bool = False

    @property
    def duration_seconds(self) -> float:
        return self.end_timestamp - self.start_timestamp

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "camera_id": self.camera_id,
            "observation_ids": list(self.observation_ids),
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_seconds": self.duration_seconds,
            "observation_count": len(self.observation_ids),
            "median_bbox": list(self.median_bbox),
            "ambiguous_parents": list(self.ambiguous_parents),
            "clean_gap_before": self.clean_gap_before,
        }


def _median_bbox(observations: Sequence[Observation]) -> list[float]:
    return [
        median([o.bbox_xyxy[0] for o in observations]),
        median([o.bbox_xyxy[1] for o in observations]),
        median([o.bbox_xyxy[2] for o in observations]),
        median([o.bbox_xyxy[3] for o in observations]),
    ]


def _size_compatible(a: Sequence[float], b: Sequence[float]) -> bool:
    for index in ((0, 2), (1, 3)):
        left = abs(float(a[index[1]]) - float(a[index[0]]))
        right = abs(float(b[index[1]]) - float(b[index[0]]))
        if left <= 0 or right <= 0:
            continue
        ratio = left / right
        if ratio < SIZE_RATIO_MIN or ratio > SIZE_RATIO_MAX:
            return False
    return True


def cluster_episodes(observations: Sequence[Observation]) -> list[Episode]:
    """Associate observations into episodes (spec §9.1/§9.2/§10).

    Same camera, one-to-one matching, gap and size limits, spatial tolerance from
    the episode's running median box.  Small targets use centre distance rather
    than IoU so that a few pixels of jitter cannot split one litter item.
    """
    episodes: list[Episode] = []
    per_camera: dict[str, list[Observation]] = {}
    for observation in observations:
        per_camera.setdefault(observation.camera_id, []).append(observation)

    for camera in sorted(per_camera):
        ordered = sorted(per_camera[camera], key=lambda o: (o.timestamp, o.observation_id))
        open_episodes: list[Episode] = []
        for observation in ordered:
            matches: list[tuple[float, Episode]] = []
            for episode in open_episodes:
                gap = observation.timestamp - episode.end_timestamp
                allow = (MAX_GAP_DENSE_SECONDS
                         if episode.observation_ids and observation.sampling == "dense"
                         and _episode_is_dense(episode, observations)
                         else MAX_GAP_COARSE_SECONDS)
                if gap > allow:
                    continue
                if not _size_compatible(observation.bbox_xyxy, episode.median_bbox):
                    continue
                radius = association_radius(episode.median_bbox)
                distance = center_distance(observation.bbox_xyxy, episode.median_bbox)
                iou = bbox_iou(observation.bbox_xyxy, episode.median_bbox)
                if distance > radius and iou < 0.10:
                    continue
                matches.append((distance / max(radius, 1e-6), episode))
            matches.sort(key=lambda pair: pair[0])
            ambiguous: list[str] = []
            if len(matches) >= 2 and matches[0][0] > 0 and \
                    matches[1][0] / matches[0][0] < 1.0 / AMBIGUITY_DISTANCE_RATIO:
                ambiguous = [matches[0][1].episode_id, matches[1][1].episode_id]
            if matches and not ambiguous:
                episode = matches[0][1]
                episode.observation_ids.append(observation.observation_id)
                episode.end_timestamp = max(episode.end_timestamp, observation.timestamp)
                episode.start_timestamp = min(episode.start_timestamp, observation.timestamp)
                members = [o for o in observations if o.observation_id in set(episode.observation_ids)]
                episode.median_bbox = _median_bbox(members)
            else:
                episode = Episode(
                    episode_id=f"ep-{len(episodes) + 1:05d}",
                    camera_id=camera,
                    observation_ids=[observation.observation_id],
                    start_timestamp=observation.timestamp,
                    end_timestamp=observation.timestamp,
                    median_bbox=list(observation.bbox_xyxy),
                    ambiguous_parents=ambiguous,
                )
                episodes.append(episode)
                open_episodes.append(episode)
            # close episodes that can no longer accept anything
            open_episodes = [e for e in open_episodes
                             if observation.timestamp - e.end_timestamp <= MAX_GAP_COARSE_SECONDS]

    for episode in episodes:
        if episode.observation_ids:
            continue
    return episodes


def _episode_is_dense(episode: Episode, observations: Sequence[Observation]) -> bool:
    by_id = {o.observation_id: o for o in observations}
    members = [by_id[oid] for oid in episode.observation_ids if oid in by_id]
    return bool(members) and all(m.sampling == "dense" for m in members)


@dataclass(slots=True)
class ReviewGroup:
    review_group_id: str
    camera_id: str
    episode_ids: list[str]
    suspected_same_object: bool
    link_evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "review_group_id": self.review_group_id,
            "camera_id": self.camera_id,
            "episode_ids": list(self.episode_ids),
            "episode_count": len(self.episode_ids),
            "suspected_same_object": self.suspected_same_object,
            "link_evidence": dict(self.link_evidence),
        }


def link_review_groups(episodes: Sequence[Episode],
                       observations: Sequence[Observation]) -> list[ReviewGroup]:
    """Link far-apart episodes into ``suspected same object`` review groups.

    Two observations separated by tens of minutes, hours or days must never
    silently extend one episode (spec §9.3).  They are grouped for a human
    SAME / NEW / UNCERTAIN decision instead.
    """
    by_id = {o.observation_id: o for o in observations}
    parent: dict[str, str] = {e.episode_id: e.episode_id for e in episodes}
    evidence: dict[tuple[str, str], dict[str, Any]] = {}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    ordered = sorted(episodes, key=lambda e: (e.camera_id, e.start_timestamp))
    for index, left in enumerate(ordered):
        left_obs = [by_id[oid] for oid in left.observation_ids if oid in by_id]
        if not left_obs:
            continue
        for right in ordered[index + 1:]:
            if right.camera_id != left.camera_id:
                continue
            gap = right.start_timestamp - left.end_timestamp
            if gap <= MAX_GAP_COARSE_SECONDS:
                continue
            radius = association_radius(left.median_bbox) * REVIEW_GROUP_RADIUS_FACTOR
            distance = center_distance(left.median_bbox, right.median_bbox)
            if distance > radius:
                continue
            if not _size_compatible(left.median_bbox, right.median_bbox):
                continue
            right_obs = [by_id[oid] for oid in right.observation_ids if oid in by_id]
            appearance_distance = _appearance_distance(left_obs, right_obs)
            if appearance_distance is not None and \
                    appearance_distance > REVIEW_GROUP_APPEARANCE_MAX_DISTANCE:
                continue
            union(left.episode_id, right.episode_id)
            evidence[f"{find(left.episode_id)}|{right.episode_id}"] = {
                "left_episode_id": left.episode_id,
                "right_episode_id": right.episode_id,
                "gap_seconds": gap,
                "center_distance_px": distance,
                "radius_px": radius,
                "appearance_distance": appearance_distance,
            }

    buckets: dict[str, list[Episode]] = {}
    for episode in episodes:
        buckets.setdefault(find(episode.episode_id), []).append(episode)

    groups: list[ReviewGroup] = []
    for root in sorted(buckets, key=lambda key: min(e.start_timestamp for e in buckets[key])):
        members = sorted(buckets[root], key=lambda e: e.start_timestamp)
        related = {k: v for k, v in evidence.items() if k.split("|", 1)[0] == root}
        groups.append(ReviewGroup(
            review_group_id=f"rg-{len(groups) + 1:05d}",
            camera_id=members[0].camera_id,
            episode_ids=[e.episode_id for e in members],
            suspected_same_object=len(members) > 1,
            link_evidence={"pairs": related, "episode_gaps_seconds": [
                members[i + 1].start_timestamp - members[i].end_timestamp
                for i in range(len(members) - 1)
            ]},
        ))
    return groups


def _appearance_distance(left: Sequence[Observation],
                         right: Sequence[Observation]) -> float | None:
    left_vectors = [o.appearance for o in left if o.appearance]
    right_vectors = [o.appearance for o in right if o.appearance]
    if not left_vectors or not right_vectors:
        return None
    left_mean = [sum(v[i] for v in left_vectors) / len(left_vectors) for i in range(len(left_vectors[0]))]
    right_mean = [sum(v[i] for v in right_vectors) / len(right_vectors) for i in range(len(right_vectors[0]))]
    return buffer_similarity(left_mean, right_mean)


# --------------------------------------------------------------------------- #
# 3. representative frames
# --------------------------------------------------------------------------- #

EXTRA_STATE_FLAGS = ("shadow_change", "low_contrast", "pose_change",
                     "mild_occlusion", "background_change")


def state_flags(observations: Sequence[Observation]) -> list[str]:
    """Derive the spec §11 extra-frame triggers from observation features."""
    if len(observations) < 2:
        return []
    flags: list[str] = []
    appearances = [o.appearance for o in observations if o.appearance]
    if len(appearances) >= 2:
        first, last = appearances[0], appearances[-1]
        if buffer_similarity(first, last) > 0.18:
            flags.append("shadow_change")
        if buffer_similarity(first, last) > 0.30:
            flags.append("low_contrast")
    backgrounds = {o.background_key for o in observations if o.background_key}
    if len(backgrounds) > 1:
        flags.append("background_change")
    boxes = [o.bbox_xyxy for o in observations]
    sizes = [bbox_short_side(b) for b in boxes]
    if sizes and max(sizes) / max(min(sizes), 1e-6) > 1.6:
        flags.append("pose_change")
    return flags


def choose_representatives(observations: Sequence[Observation], *,
                           max_frames: int = 3) -> list[dict[str, Any]]:
    """Pick 1 normal representative, plus at most two justified hard states.

    Deliberately **not** the highest-confidence, largest-box or prettiest frame
    (spec §11): score reward closeness to the episode middle, the median box
    size, an untruncated tile and a typical appearance.
    """
    if not observations:
        raise HistoricalError("no observations to choose a representative from")
    ordered = sorted(observations, key=lambda o: (o.timestamp, o.observation_id))
    median_box = _median_bbox(ordered)
    median_side = math.sqrt(max(0.0, abs(median_box[2] - median_box[0]) *
                                abs(median_box[3] - median_box[1])))
    middle = (len(ordered) - 1) / 2.0
    appearances = [o.appearance for o in ordered if o.appearance]
    typical = None
    if appearances:
        typical = [sum(v[i] for v in appearances) / len(appearances) for i in range(len(appearances[0]))]

    scored: list[tuple[float, Observation]] = []
    for index, observation in enumerate(ordered):
        side = math.sqrt(max(0.0, abs(observation.bbox_xyxy[2] - observation.bbox_xyxy[0]) *
                            abs(observation.bbox_xyxy[3] - observation.bbox_xyxy[1])))
        score = abs(index - middle) / max(len(ordered), 1)
        score += abs(side - median_side) / max(median_side, 1e-6)
        if observation.tile_boundary_truncated:
            score += 1.0
        if typical and observation.appearance:
            score += buffer_similarity(observation.appearance, typical)
        scored.append((score, observation))
    scored.sort(key=lambda pair: pair[0])

    chosen: list[dict[str, Any]] = [{
        "role": "normal",
        "observation_id": scored[0][1].observation_id,
        "frame_id": scored[0][1].frame_id,
    }]
    flags = state_flags(ordered)
    if flags and max_frames > 1:
        # Prefer a frame that is *not* the representative and that carries a state change.
        remaining = [pair for pair in scored if pair[1].observation_id != chosen[0]["observation_id"]]
        for role, pair in zip(flags, remaining[:max_frames - 1]):
            chosen.append({"role": role, "observation_id": pair[1].observation_id,
                           "frame_id": pair[1].frame_id})
    return chosen[:max_frames]


# --------------------------------------------------------------------------- #
# 4. candidate priority and diversity ordering
# --------------------------------------------------------------------------- #


def candidate_tier(observation: Observation) -> str:
    """P1 / P2 / P3 / P3b / LOW review priority (spec §6)."""
    label = observation.source_label
    yolo = observation.confidence_by_source.get("yolo")
    turhancan = float(observation.confidence_by_source.get("turhancan") or 0.0)
    if label == "yolo_only":
        if yolo is not None and YOLO_WORKING_BAND[0] <= yolo <= YOLO_WORKING_BAND[1]:
            return "P2"
        return "P3"
    if label == "both":
        if not observation.covered_location or not observation.covered_appearance:
            return "P3"
        return "P3b"
    if label == "turhancan_only":
        if turhancan < REVIEW_MIN_CONFIDENCE:
            # Spec §6: very low Turhancan score + repeated background is low priority.
            return "LOW"
        if turhancan >= P1_MIN_CONFIDENCE and not observation.covered_location:
            return "P1"
        return "P3"
    return "LOW"


def above_review_floor(observation: Observation) -> bool:
    """True when any source scored at or above the human-review admission floor."""
    return max(observation.confidence_by_source.values(), default=0.0) >= REVIEW_MIN_CONFIDENCE


def observation_buckets(observation: Observation, *, roi: Sequence[Sequence[float]] | None,
                        time_bucket: str) -> set[str]:
    """Coverage buckets used by the greedy diversity selection (spec §12)."""
    buckets = {f"camera:{observation.camera_id}",
               f"source:{observation.source_label}",
               f"size:{size_class(observation.bbox_xyxy)}",
               f"time:{time_bucket}"}
    cx, cy = bbox_center(observation.bbox_xyxy)
    buckets.add(f"location:{_grid_cell(cx, cy, roi)}")
    if observation.appearance:
        buckets.add(f"appearance:{_color_bin(observation.appearance)}")
    if observation.background_key:
        buckets.add(f"background:{observation.background_key}")
    if observation.tile_boundary_truncated:
        buckets.add("disagreement:tile_truncated")
    return buckets


def _grid_cell(cx: float, cy: float, roi: Sequence[Sequence[float]] | None) -> str:
    """4x3 grid in normalized frame coordinates, so cameras are comparable."""
    nx = clamp(cx / SOURCE_WIDTH, 0.0, 0.999)
    ny = clamp(cy / SOURCE_HEIGHT, 0.0, 0.999)
    column = int(nx * LOCATION_GRID_COLS)
    row = int(ny * LOCATION_GRID_ROWS)
    return f"{row}{column}"


def _color_bin(appearance: Sequence[float]) -> str:
    if not appearance:
        return "na"
    total = sum(float(v) for v in appearance) + 1e-6
    red = int(float(appearance[0]) / total * COLOR_BINS) if len(appearance) > 0 else 0
    green = int(float(appearance[1]) / total * COLOR_BINS) if len(appearance) > 1 else 0
    blue = int(float(appearance[2]) / total * COLOR_BINS) if len(appearance) > 2 else 0
    return f"{min(red, COLOR_BINS - 1)}{min(green, COLOR_BINS - 1)}{min(blue, COLOR_BINS - 1)}"


@dataclass(slots=True)
class QueueEntry:
    review_group_id: str
    tier: str
    buckets: list[str]
    rank_score: float = 0.0
    covered_count: int = 0
    selected_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "review_group_id": self.review_group_id,
            "tier": self.tier,
            "buckets": list(self.buckets),
            "rank_score": self.rank_score,
            "covered_count": self.covered_count,
            "selected_reason": self.selected_reason,
        }


def greedy_diversity_select(
    entries: Sequence[QueueEntry], *, limit: int, seed: str = DEFAULT_SEED,
    random_reserve_fraction: float = RANDOM_RESERVE_FRACTION,
) -> list[QueueEntry]:
    """Order review groups so each one adds as many uncovered buckets as possible.

    No embeddings; a plain greedy set cover over camera / location / size / time /
    appearance / background / source / disagreement buckets, with a small
    seeded random reserve at the end (spec §12).
    """
    tier_rank = {"P1": 0, "P2": 1, "P3": 2, "P3b": 3, "LOW": 4}
    reserve = int(round(limit * random_reserve_fraction)) if limit else 0
    if random_reserve_fraction > 0 and limit:
        reserve = max(1, reserve)
    reserve = min(reserve, max(0, limit - 1))
    greedy_budget = max(0, limit - reserve)

    remaining = list(entries)
    order_index = {id(entry): index for index, entry in enumerate(entries)}
    covered: set[str] = set()
    ordered: list[QueueEntry] = []
    while remaining and len(ordered) < greedy_budget:
        best = None
        best_key = None
        for entry in remaining:
            gain = len(set(entry.buckets) - covered)
            # Insertion index is the final tie-break so the order is reproducible
            # from the same input order instead of depending on a hash.
            key = (-gain, tier_rank.get(entry.tier, 9), -float(entry.rank_score),
                   order_index[id(entry)])
            if best_key is None or key < best_key:
                best, best_key = entry, key
        assert best is not None
        best.covered_count = len(set(best.buckets) - covered)
        best.selected_reason = f"greedy:+{best.covered_count}"
        covered.update(best.buckets)
        ordered.append(best)
        remaining.remove(best)

    if len(ordered) < limit and remaining:
        rng = seeded_rng(seed, "random_reserve")
        reserve_pool = list(remaining)
        rng.shuffle(reserve_pool)
        for entry in reserve_pool[:limit - len(ordered)]:
            entry.covered_count = 0
            entry.selected_reason = "random_reserve"
            ordered.append(entry)
            remaining.remove(entry)

    return ordered


def batch_slices(total: int, *, batch_size: int = FIRST_BATCH_SIZE) -> list[list[int]]:
    """Split the ordered queue into review batches (spec §12: 50/50/50)."""
    if total <= 0:
        return []
    return [list(range(start, min(start + batch_size, total)))
            for start in range(0, total, batch_size)]


def blind_frame_selection(frames: Sequence[Mapping[str, Any]], *, per_camera: int,
                          seed: str = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Pick blind ROI frames: half truly random, half coverage-driven (spec §17)."""
    chosen: list[dict[str, Any]] = []
    true_random_count = int(round(per_camera * BLIND_TRULY_RANDOM_FRACTION))
    for camera in CAMERAS:
        pool = [dict(row) for row in frames if row.get("camera_id") == camera]
        if not pool:
            continue
        rng = seeded_rng(seed, "blind", camera)
        rng.shuffle(pool)
        picked: list[dict[str, Any]] = []
        for row in pool[:true_random_count]:
            row = dict(row)
            row["blind_kind"] = "truly_random"
            picked.append(row)
        remaining = pool[true_random_count:]
        remaining.sort(key=lambda row: (row.get("date", ""), row.get("window_id", ""),
                                        row.get("offset_seconds", 0.0)))
        coverage: set[tuple[str, str]] = set()
        for row in remaining:
            key = (row.get("date", ""), row.get("selection_bucket", ""))
            if key in coverage:
                continue
            item = dict(row)
            item["blind_kind"] = "coverage_gap"
            picked.append(item)
            coverage.add(key)
            if len(picked) >= per_camera:
                break
        for row in remaining:
            if len(picked) >= per_camera:
                break
            if any(p["frame_id"] == row.get("frame_id") for p in picked):
                continue
            item = dict(row)
            item["blind_kind"] = "coverage_gap"
            picked.append(item)
        chosen.extend(picked[:per_camera])
    return chosen
