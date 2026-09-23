"""Step 1C-0: recover and verify the original PS behind human-confirmed Gold episodes.

Scope (plan §13, frozen Step 0B protocol):

* Turns the *hypothetical* lineage readiness recorded by Step 1B into a status that
  comes from actually contacting the historical recording source, downloading the
  original PS, hashing it, decoding it and mapping the Gold timestamp onto a real
  source-resolution frame.
* Never touches Gold truth, never builds 640x640 tiles, never runs a detector and
  never reads anything from the Step 0A Sealed archive.
* Pure stdlib.  ``ground_litter_recording_source`` is stdlib-only, so the whole
  resolver/store/runner is unit-testable; only decoding needs PyAV/cv2 and is
  therefore injected as a ``DecodeAdapter``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .ground_litter_recording_source import (
    ListQuery,
    RecordingListClient,
    RecordingSourceError,
    UrlRefreshPolicy,
    file_looks_like_media,
)

SCHEMA_VERSION = "ground_litter_gold_source_recovery_v1"
GENERATOR_VERSION = "step1c0-1.0.0"

# ---------------------------------------------------------------- constants

RECOVERY_STATUSES = (
    "RECOVERED_SOURCE_NATIVE",
    "LINEAGE_UNRESOLVED",
    "LINEAGE_AMBIGUOUS",
    "SOURCE_NOT_FOUND",
    "SOURCE_EXPIRED",
    "DOWNLOAD_FAILED",
    "HASH_MISMATCH",
    "DECODE_FAILED",
    "TIMESTAMP_UNRESOLVED",
    "SOURCE_METADATA_INVALID",
)

#: Internal, non-final: the resolver found a unique source and the episode still has
#: to be downloaded/decoded/verified.  Never reported as an episode status.
RESOLVED_PENDING_FETCH = "RESOLVED_PENDING_FETCH"

#: |delta| <= this is a normal timestamp verification (§13).
TIMESTAMP_OK_MS = 500.0
#: Beyond this the mapping is not considered reliable and the episode fails (§13).
TIMESTAMP_HARD_MS = 2000.0

#: Default primary lookup window around a Gold timestamp.
PRIMARY_WINDOW_MINUTES = 30
#: Fallback window before declaring a file missing.
WIDE_WINDOW_HOURS = 3

#: Anything matching one of these must never be touched by this task (§20).
SEALED_MARKERS = (
    "sealed_test",
    "SEALED_DO_NOT_TUNE",
    "ground-litter-detector-feasibility-20260923-r1",
    "ground-litter-feasibility/20260923-r1",
)

_FILE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class RecoveryError(RuntimeError):
    """Recovery could not proceed; the caller must surface it."""


class SealedAssetError(RecoveryError):
    """A Step 0A Sealed asset was supplied to a Silver->Gold recovery run (§20)."""


class GoldChangedError(RecoveryError):
    """The Gold artifact changed underneath the run (§26 hard fail)."""


# ---------------------------------------------------------------- sealed guard


def assert_not_sealed(*values: Any) -> None:
    """Hard-fail if any provided path/manifest/metadata names a Step 0A Sealed asset."""
    for value in values:
        text = str(value or "")
        if not text:
            continue
        for marker in SEALED_MARKERS:
            if marker.lower() in text.lower():
                raise SealedAssetError(
                    f"refusing to touch a Step 0A Sealed asset (matched {marker!r})")


# ---------------------------------------------------------------- Gold input


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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def parse_timestamp(text: str | None) -> datetime | None:
    if not isinstance(text, str) or not text.strip():
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class EpisodeCard:
    card_id: str
    timestamp: str | None
    frame_id: str | None
    source_file_id: str | None
    batch_key: str | None

    @property
    def moment(self) -> datetime | None:
        return parse_timestamp(self.timestamp)


@dataclass(frozen=True)
class RequiredEpisode:
    episode_id: str
    camera_id: str
    scene_version: str
    record_kind: str
    is_manual_missing_target: bool
    localization_status: str | None
    requested_timestamp: str | None
    source_member_card_ids: tuple[str, ...]
    cards: tuple[EpisodeCard, ...]
    #: the file_ids Step 1B believed in; a hypothesis, not a verification
    hypothesised_file_ids: tuple[str, ...]
    #: authoritative 20-digit device code for this camera; never guessed from a prefix
    device_code: str = ""

    @property
    def requested_moment(self) -> datetime | None:
        return parse_timestamp(self.requested_timestamp)

    @property
    def has_known_file_id(self) -> bool:
        return bool(self.hypothesised_file_ids)


@dataclass(frozen=True)
class GoldInput:
    path: Path
    sha256: str
    schema_version: str
    review_schema_version: str
    code_commit: str
    required: tuple[RequiredEpisode, ...]
    manual_required: tuple[RequiredEpisode, ...]

    @property
    def required_count(self) -> int:
        return len(self.required)


def load_device_codes(*sources: Path | str) -> dict[str, str]:
    """camera_id -> 20-digit device_code from authoritative local metadata.

    Never synthesised from a prefix: a wrong device code makes every recording query
    fail silently, which would masquerade as "source expired".  Accepts either the
    per-camera final-ROI configs (camera_id + device_code) or a JSONL artifact whose
    rows carry ``camera_id``/``device_code``.
    """
    mapping: dict[str, str] = {}
    for source in sources:
        path = Path(source)
        if path.is_dir():
            candidates = sorted(path.glob("ground_litter_*_final_roi.json")) or \
                sorted(path.glob("*.json"))
        elif path.is_file():
            candidates = [path]
        else:
            continue
        for candidate in candidates:
            try:
                if candidate.suffix == ".jsonl":
                    with candidate.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            text = line.strip()
                            if text:
                                _absorb_device_row(json.loads(text), mapping)
                else:
                    _absorb_device_row(
                        json.loads(candidate.read_text(encoding="utf-8")), mapping)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
    return mapping


def _absorb_device_row(row: Any, mapping: dict[str, str]) -> None:
    if not isinstance(row, dict):
        return
    camera = str(row.get("camera_id") or "")
    device = str(row.get("device_code") or "")
    if camera and re.fullmatch(r"\d{20}", device):
        mapping.setdefault(camera, device)
    for cam in row.get("cameras") or []:
        if isinstance(cam, dict):
            _absorb_device_row(cam, mapping)


def load_gold(path: Path | str, *, manifest_path: Path | str | None = None,
              device_codes: Mapping[str, str] | None = None) -> GoldInput:
    device_codes = dict(device_codes or {})
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Gold artifact not found: {target}")
    assert_not_sealed(target)
    digest = sha256_file(target)

    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise RecoveryError(f"invalid JSON on line {number} of {target}") from exc

    schema = review_schema = commit = ""
    if manifest_path is not None and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert_not_sealed(str(manifest_path))
        schema = str(manifest.get("schema_version") or "")
        review_schema = str(manifest.get("review_schema_version") or "")
        commit = str((manifest.get("step1b") or {}).get("code_commit") or "")

    required: list[RequiredEpisode] = []
    seen: set[str] = set()
    for record in records:
        if str(record.get("truth_class")) != "REQUIRED_LITTER":
            continue
        episode_id = str(record.get("episode_id") or "")
        if not episode_id:
            raise RecoveryError("Required record without episode_id")
        if episode_id in seen:
            raise RecoveryError(f"duplicate episode_id in Gold: {episode_id}")
        seen.add(episode_id)

        cards: list[EpisodeCard] = []
        for member in (record.get("trainability_evidence") or {}).get("members") or []:
            assert_not_sealed(member.get("asset_paths"))
            cards.append(EpisodeCard(
                card_id=str(member.get("card_id") or ""),
                timestamp=member.get("timestamp"),
                frame_id=member.get("frame_id"),
                source_file_id=(str(member["source_file_id"])
                                if member.get("source_file_id") else None),
                batch_key=member.get("batch_key"),
            ))
        if not cards:
            # fall back to the declared member ids so the episode is never dropped
            cards = [EpisodeCard(card_id=str(cid), timestamp=record.get("start_timestamp"),
                                 frame_id=None, source_file_id=None, batch_key=None)
                     for cid in record.get("member_card_ids") or []]

        hypothesised = tuple(sorted({c.source_file_id for c in cards if c.source_file_id}))
        camera = str(record.get("camera_id") or "")
        required.append(RequiredEpisode(
            episode_id=episode_id,
            camera_id=camera,
            scene_version=str(record.get("scene_version") or ""),
            record_kind=str(record.get("record_kind") or ""),
            is_manual_missing_target=str(record.get("record_kind")) == "manual_missing_target",
            localization_status=record.get("localization_status"),
            requested_timestamp=record.get("start_timestamp"),
            source_member_card_ids=tuple(str(c) for c in record.get("member_card_ids") or []),
            cards=tuple(cards),
            device_code=str(device_codes.get(camera, "")),
            hypothesised_file_ids=hypothesised,
        ))

    required.sort(key=lambda e: (e.camera_id, e.episode_id))
    manual = tuple(e for e in required if e.is_manual_missing_target)
    return GoldInput(
        path=target, sha256=digest, schema_version=schema,
        review_schema_version=review_schema, code_commit=commit,
        required=tuple(required), manual_required=manual,
    )


# ---------------------------------------------------------------- resolution


@dataclass(frozen=True)
class Recording:
    file_id: str
    camera_id: str
    record_start: str
    record_end: str
    file_size: int | None
    file_name: str = ""

    @property
    def start(self) -> datetime | None:
        return parse_timestamp(self.record_start)

    @property
    def end(self) -> datetime | None:
        return parse_timestamp(self.record_end)

    def contains(self, moment: datetime) -> bool:
        start, end = self.start, self.end
        if start is None or end is None:
            return False
        return start <= moment <= end


@dataclass
class Resolution:
    status: str
    reason_code: str
    reason_detail: str
    method: str = ""
    recording: Recording | None = None
    candidates: tuple[Recording, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.status == RESOLVED_PENDING_FETCH and self.recording is not None


class ListingPort(Protocol):
    """Minimal listing surface so tests can inject a fake remote."""

    def list_recordings(self, device_code: str, start: str, end: str) -> list[Recording]:
        ...


def _episode_card_timestamp(episode: RequiredEpisode) -> datetime | None:
    for card in episode.cards:
        moment = card.moment
        if moment is not None:
            return moment
    return episode.requested_moment


def resolve_episode(
    episode: RequiredEpisode, listing: ListingPort, *,
    primary_window_minutes: int = PRIMARY_WINDOW_MINUTES,
    wide_window_hours: int = WIDE_WINDOW_HOURS,
) -> Resolution:
    """Resolve one episode to a unique original recording.

    Two modes, both fully determined by remote metadata (never by file names):

    A. a known ``file_id`` -> exact match required;
    B. no ``file_id`` -> the timestamp must fall inside exactly one recording.
    """
    moment = _episode_card_timestamp(episode)
    if moment is None:
        return Resolution("SOURCE_METADATA_INVALID", "gold_timestamp_missing",
                          "episode has no parsable timestamp or frame time")
    if not episode.camera_id:
        return Resolution("SOURCE_METADATA_INVALID", "gold_camera_missing",
                          "episode has no camera_id")

    if episode.hypothesised_file_ids:
        return _resolve_by_file_id(episode, listing, moment, primary_window_minutes,
                                   wide_window_hours)
    return _resolve_by_timestamp(episode, listing, moment, primary_window_minutes,
                                 wide_window_hours)


def _window(moment: datetime, minutes: int) -> tuple[str, str]:
    pad = timedelta(minutes=minutes)
    return ((moment - pad).strftime("%Y-%m-%d %H:%M:%S"),
            (moment + pad).strftime("%Y-%m-%d %H:%M:%S"))


def _resolve_by_file_id(
    episode: RequiredEpisode, listing: ListingPort, moment: datetime,
    primary_window_minutes: int, wide_window_hours: int,
) -> Resolution:
    device = _device_code(episode)
    if device is None:
        return Resolution("SOURCE_METADATA_INVALID", "device_code_unavailable",
                          f"no authoritative 20-digit device code for camera "
                          f"{episode.camera_id!r}")
    wanted = set(episode.hypothesised_file_ids)
    attempts = [(primary_window_minutes, 60), (wide_window_hours * 60, 60 * 24)]
    saw_any_recording = False
    last_error = ""
    for minutes, _ in attempts:
        start, end = _window(moment, minutes)
        listings, error = _safe_list(listing, device, start, end)
        if listings is None:
            last_error = error
            continue
        if listings:
            saw_any_recording = True
        hits = [r for r in listings if r.file_id in wanted]
        if hits:
            return _pick(episode, hits, moment, "file_id_exact_match", "A_known_file_id")
    if last_error:
        return Resolution("SOURCE_NOT_FOUND", "listing_api_error", last_error)
    if not saw_any_recording:
        return Resolution("SOURCE_EXPIRED", "no_recordings_in_window",
                          f"no recordings at all around {episode.requested_timestamp}")
    return Resolution("SOURCE_NOT_FOUND", "file_id_absent_from_listing",
                      f"file_id(s) {sorted(wanted)} not present anymore")


def _resolve_by_timestamp(
    episode: RequiredEpisode, listing: ListingPort, moment: datetime,
    primary_window_minutes: int, wide_window_hours: int,
) -> Resolution:
    device = _device_code(episode)
    if device is None:
        return Resolution("SOURCE_METADATA_INVALID", "device_code_unavailable",
                          f"no authoritative 20-digit device code for camera "
                          f"{episode.camera_id!r}")
    start, end = _window(moment, primary_window_minutes)
    listings, error = _safe_list(listing, device, start, end)
    if listings is None:
        return Resolution("SOURCE_NOT_FOUND", "listing_api_error", error)
    inside = sorted((r for r in listings if r.contains(moment)),
                    key=lambda r: (r.record_start, r.file_id))
    if not inside:
        # Widen once *before* concluding anything: an empty narrow window is not the
        # same as an expired recording, and only the wider view can tell them apart.
        wide_start, wide_end = _window(moment, wide_window_hours * 60)
        wide, wide_error = _safe_list(listing, device, wide_start, wide_end)
        if wide is None and not listings:
            return Resolution("SOURCE_NOT_FOUND", "listing_api_error", wide_error)
        if wide:
            listings = wide
            inside = sorted((r for r in wide if r.contains(moment)),
                            key=lambda r: (r.record_start, r.file_id))
    if not inside:
        if not listings:
            return Resolution("SOURCE_EXPIRED", "no_recordings_in_window",
                              f"no recordings at all around {episode.requested_timestamp}")
        return Resolution("SOURCE_NOT_FOUND", "no_recording_contains_timestamp",
                          "no recording window covers the Gold timestamp")
    if len(inside) > 1:
        return Resolution("LINEAGE_AMBIGUOUS", "multiple_recordings_contain_timestamp",
                          f"{len(inside)} recordings contain the timestamp",
                          candidates=tuple(inside))
    return Resolution(RESOLVED_PENDING_FETCH, "resolved_by_unique_timestamp_window",
                      "exactly one recording contains the Gold timestamp",
                      method="B_camera_plus_timestamp",
                      recording=inside[0], candidates=tuple(inside))


def _pick(episode: RequiredEpisode, hits: Sequence[Recording], moment: datetime,
          reason_code: str, method: str) -> Resolution:
    ordered = sorted(hits, key=lambda r: (r.record_start, r.file_id))
    containing = [r for r in ordered if r.contains(moment)]
    if len(containing) == 1:
        return Resolution(RESOLVED_PENDING_FETCH, reason_code,
                          "file_id matched and timestamp falls inside the recording",
                          method=method, recording=containing[0],
                          candidates=tuple(ordered))
    if len(containing) > 1:
        return Resolution("LINEAGE_AMBIGUOUS", "multiple_recordings_contain_timestamp",
                          f"{len(containing)} recordings contain the timestamp",
                          candidates=tuple(containing))
    if len(ordered) == 1:
        return Resolution("SOURCE_METADATA_INVALID", "timestamp_outside_recording",
                          "file_id matched but the Gold timestamp is outside it",
                          method=method, recording=ordered[0], candidates=tuple(ordered))
    return Resolution("SOURCE_METADATA_INVALID", "timestamp_outside_recording",
                      "file_id matched several recordings, none containing the timestamp",
                      method=method, candidates=tuple(ordered))


def _device_code(episode: RequiredEpisode) -> str | None:
    """Only ever the authoritative device code recorded with the episode."""
    code = str(episode.device_code or "")
    return code if re.fullmatch(r"\d{20}", code) else None


def _safe_list(listing: ListingPort, device: str, start: str, end: str) -> tuple[list[Recording] | None, str]:
    try:
        return list(listing.list_recordings(device, start, end)), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:120]}"


# ---------------------------------------------------------------- remote port


class HttpListing:
    """Real ``ListingPort`` on top of the repository recording client."""

    def __init__(self, endpoint: str | None = None,
                 client: RecordingListClient | None = None) -> None:
        self.client = client or (RecordingListClient(endpoint) if endpoint
                                 else RecordingListClient())

    def list_recordings(self, device_code: str, start: str, end: str) -> list[Recording]:
        page = self.client.query(ListQuery(device_code, start, end))
        return [
            Recording(file_id=entry.file.file_id, camera_id=device_code[-5:],
                      record_start=entry.file.record_start, record_end=entry.file.record_end,
                      file_size=entry.file.file_size, file_name=entry.file.file_name)
            for entry in page.entries
        ]


class HttpDownloader:
    """Real downloader that preserves PS into a permanent artifact directory.

    Deliberately does **not** use ``ManagedRecordingCache``: that cache may clean up
    its own files, and §19 requires the recovered PS to survive as an
    artifact-owned training-evidence bundle.
    """

    def __init__(self, listing: HttpListing, *, timeout: float = 60.0,
                 max_attempts: int = 3) -> None:
        from .ground_litter_recording_source import RecordingDownloader
        self.listing = listing
        self.downloader = RecordingDownloader(timeout=timeout, max_attempts=max_attempts)
        self._policy = UrlRefreshPolicy()

    def fetch(self, device_code: str, file_id: str, moment: datetime,
              target: Path, *, expected_size: int | None = None) -> Any:
        start, end = _window(moment, PRIMARY_WINDOW_MINUTES)
        entry = self.downloader.fetch_url_for_file(
            self.listing.client, ListQuery(device_code, start, end), file_id,
            policy=self._policy)
        return self.downloader.download(entry.url, target,
                                        expected_size=expected_size, allow_resume=True)


# ---------------------------------------------------------------- source store


@dataclass
class SourceFileRecord:
    source_file_id: str
    camera_id: str
    recording_start: str | None = None
    recording_end: str | None = None
    remote_reference: str = ""
    local_ps_path: str = ""
    size_bytes: int = 0
    local_sha256: str = ""
    remote_sha256: str | None = None
    remote_hash_verified: bool = False
    decode_status: str = "PENDING"
    source_width: int = 0
    source_height: int = 0
    duration: float | None = None
    codec: str = ""
    recovery_status: str = "PENDING"
    recovery_reason: str = ""
    downloaded_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_file_id": self.source_file_id,
            "camera_id": self.camera_id,
            "remote_reference": self.remote_reference,
            "recording_start": self.recording_start,
            "recording_end": self.recording_end,
            "local_ps_path": self.local_ps_path,
            "size_bytes": self.size_bytes,
            "local_sha256": self.local_sha256,
            "remote_sha256": self.remote_sha256,
            "remote_hash_verified": self.remote_hash_verified,
            "decode_status": self.decode_status,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "duration": self.duration,
            "codec": self.codec,
            "recovery_status": self.recovery_status,
            "recovery_reason": self.recovery_reason,
            "downloaded_at": self.downloaded_at,
        }

    @staticmethod
    def from_dict(row: Mapping[str, Any]) -> "SourceFileRecord":
        known = {f for f in SourceFileRecord.__dataclass_fields__}  # type: ignore[attr-defined]
        return SourceFileRecord(**{k: v for k, v in row.items() if k in known})


class DecodeAdapter(Protocol):
    """Decoding surface; the real one needs PyAV/cv2, so it is injected."""

    def probe(self, path: Path) -> dict[str, Any]:
        ...

    def sample_frame(self, path: Path, offset_seconds: float) -> dict[str, Any]:
        ...


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned[:180] or "source"


@dataclass
class RecoveryConfig:
    artifact_root: Path
    gold_path: Path
    gold_manifest: Path | None = None
    endpoint: str | None = None
    #: download cap; 0 or None means unlimited
    max_downloads: int | None = None
    verify_hashes_on_resume: bool = False
    frame_quality: int = 95

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_root": str(self.artifact_root),
            "gold_path": str(self.gold_path),
            "gold_manifest": str(self.gold_manifest) if self.gold_manifest else None,
            "endpoint": self.endpoint or "default",
            "max_downloads": self.max_downloads,
            "verify_hashes_on_resume": self.verify_hashes_on_resume,
            "frame_quality": self.frame_quality,
        }


@dataclass
class EpisodeEvidence:
    episode_id: str
    truth_class: str
    camera_id: str
    scene_version: str
    source_member_card_ids: list[str]
    source_file_id: str | None
    source_file_resolution_method: str
    requested_timestamp: str | None
    decoded_timestamp: str | None = None
    timestamp_delta_ms: float | None = None
    timestamp_match_quality: str = "UNRESOLVED"
    verification_frame_path: str | None = None
    source_width: int = 0
    source_height: int = 0
    source_recovery_status: str = "LINEAGE_UNRESOLVED"
    reason_code: str = ""
    reason_detail: str = ""
    manual_missing_target: bool = False
    localization_status: str | None = None
    all_source_file_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class GoldSourceRecovery:
    """Resolve -> download -> hash -> decode -> verification frame -> evidence."""

    def __init__(
        self,
        config: RecoveryConfig,
        listing: ListingPort,
        *,
        fetcher: Callable[..., Any] | None = None,
        decoder: DecodeAdapter | None = None,
        now: Callable[[], str] | None = None,
    ) -> None:
        assert_not_sealed(config.artifact_root, config.gold_path, config.gold_manifest)
        self.config = config
        self.listing = listing
        self.fetcher = fetcher
        self.decoder = decoder
        self._now = now or (lambda: datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.root = Path(config.artifact_root)
        self.raw_dir = self.root / "raw_ps"
        self.frames_dir = self.root / "verification_frames"
        self.source_files_path = self.root / "source_files.jsonl"
        self.evidence_path = self.root / "episode_source_evidence.jsonl"
        self.files: dict[str, SourceFileRecord] = {
            row["source_file_id"]: SourceFileRecord.from_dict(row)
            for row in read_jsonl(self.source_files_path)
            if row.get("source_file_id")
        }
        self.evidence: dict[str, EpisodeEvidence] = {
            row["episode_id"]: EpisodeEvidence(**{
                k: v for k, v in row.items()
                if k in EpisodeEvidence.__dataclass_fields__})  # type: ignore[attr-defined]
            for row in read_jsonl(self.evidence_path)
            if row.get("episode_id")
        }
        self.downloads_done = 0

    # -- persistence -------------------------------------------------------- #

    def save(self) -> None:
        write_jsonl(self.source_files_path,
                    [self.files[k].as_dict() for k in sorted(self.files)])
        write_jsonl(self.evidence_path,
                    [self.evidence[k].as_dict() for k in sorted(self.evidence)])

    # -- source files ------------------------------------------------------- #

    def local_path_for(self, recording: Recording) -> Path:
        return self.raw_dir / safe_name(recording.camera_id) / (safe_name(recording.file_id) + ".ps")

    def _reusable(self, record: SourceFileRecord | None, path: Path) -> bool:
        if record is None or not path.is_file():
            return False
        if int(record.size_bytes or 0) != path.stat().st_size:
            return False
        if record.decode_status != "OK":
            return False
        if self.config.verify_hashes_on_resume and record.local_sha256:
            return sha256_file(path) == record.local_sha256
        return bool(record.local_sha256)

    def ensure_source_file(self, recording: Recording, moment: datetime,
                           device_code: str) -> SourceFileRecord:
        """Download at most once per ``source_file_id``; resume-safe and artifact-owned."""
        existing = self.files.get(recording.file_id)
        path = self.local_path_for(recording)
        if self._reusable(existing, path):
            existing.recovery_status = "PRESERVED"
            existing.recovery_reason = "reused_verified_local_copy"
            return existing

        record = existing or SourceFileRecord(source_file_id=recording.file_id,
                                              camera_id=recording.camera_id)
        record.recording_start = recording.record_start
        record.recording_end = recording.record_end
        record.remote_reference = f"ctseelink/playback/file-urls::{recording.file_id}"
        record.local_ps_path = str(path)

        if self.fetcher is None:
            record.recovery_status = "DOWNLOAD_FAILED"
            record.recovery_reason = "no_downloader_configured"
            self.files[recording.file_id] = record
            return record
        if self.config.max_downloads is not None and \
                self.downloads_done >= self.config.max_downloads:
            record.recovery_status = "DOWNLOAD_FAILED"
            record.recovery_reason = "download_cap_reached"
            self.files[recording.file_id] = record
            return record

        # a stale .part is never a success artifact
        part = path.with_name(path.name + ".part")
        if part.exists():
            part.unlink(missing_ok=True)

        try:
            result = self.fetcher(device_code, recording.file_id,
                                  moment, path, expected_size=recording.file_size)
        except Exception as exc:
            record.recovery_status = "DOWNLOAD_FAILED"
            record.recovery_reason = f"download_error:{type(exc).__name__}"
            self.files[recording.file_id] = record
            self.save()
            return record
        self.downloads_done += 1

        size = int(getattr(result, "size", 0) or 0)
        digest = str(getattr(result, "sha256", "") or "")
        if size <= 0 or not path.is_file() or not file_looks_like_media(path):
            record.recovery_status = "DOWNLOAD_FAILED"
            record.recovery_reason = "downloaded_object_is_not_media"
            self.files[recording.file_id] = record
            self.save()
            return record
        if recording.file_size is not None and int(recording.file_size) != size:
            record.recovery_status = "DOWNLOAD_FAILED"
            record.recovery_reason = "size_mismatch_vs_remote_metadata"
            self.files[recording.file_id] = record
            self.save()
            return record

        record.size_bytes = size
        record.local_sha256 = digest or sha256_file(path)
        record.downloaded_at = self._now()
        record.recovery_status = "DOWNLOADED"
        record.recovery_reason = "downloaded"
        record.remote_hash_verified = False
        record.remote_sha256 = None
        self.files[recording.file_id] = record
        self.save()
        return record

    def decode_source_file(self, record: SourceFileRecord) -> dict[str, Any]:
        path = Path(record.local_ps_path)
        if self.decoder is None:
            record.decode_status = "NO_DECODER"
            record.recovery_status = "DECODE_FAILED"
            record.recovery_reason = "no_decoder_configured"
            return {"ok": False, "error": "no_decoder_configured"}
        if record.decode_status == "OK" and record.source_width > 0 and record.source_height > 0:
            # The PS on disk is immutable and already verified: do not probe it again
            # for every episode that shares it (48 files back 180 episodes).
            return {"ok": True, "width": record.source_width,
                    "height": record.source_height,
                    "duration_seconds": record.duration, "codec": record.codec,
                    "cached": True, "error": ""}
        try:
            probe = self.decoder.probe(path)
        except Exception as exc:
            probe = {"ok": False, "error": f"decoder_exception:{type(exc).__name__}"}
        if not probe.get("ok"):
            record.decode_status = "FAILED"
            record.source_width = int(probe.get("width") or 0)
            record.source_height = int(probe.get("height") or 0)
            record.recovery_status = "DECODE_FAILED"
            record.recovery_reason = str(probe.get("error") or "decode_failed")[:120]
            return probe
        record.decode_status = "OK"
        record.source_width = int(probe.get("width") or 0)
        record.source_height = int(probe.get("height") or 0)
        record.duration = probe.get("duration_seconds")
        record.codec = str(probe.get("codec") or "")
        if record.source_width <= 0 or record.source_height <= 0:
            record.decode_status = "FAILED"
            record.recovery_status = "DECODE_FAILED"
            record.recovery_reason = "resolution_unreadable"
        return probe

    # -- episodes ----------------------------------------------------------- #

    def recover_episode(self, episode: RequiredEpisode, *, force: bool = False) -> EpisodeEvidence:
        if not force:
            existing = self.evidence.get(episode.episode_id)
            if existing is not None and self._evidence_artifacts_intact(existing):
                # A verified episode is never re-derived: a transient remote error on a
                # later run must not downgrade evidence that is already backed by a
                # preserved PS and an on-disk verification frame.
                return existing
        resolution = resolve_episode(episode, self.listing)
        evidence = EpisodeEvidence(
            episode_id=episode.episode_id,
            truth_class="REQUIRED_LITTER",
            camera_id=episode.camera_id,
            scene_version=episode.scene_version,
            source_member_card_ids=list(episode.source_member_card_ids),
            source_file_id=None,
            source_file_resolution_method=resolution.method,
            requested_timestamp=episode.requested_timestamp,
            manual_missing_target=episode.is_manual_missing_target,
            localization_status=episode.localization_status,
            all_source_file_ids=sorted({r.file_id for r in resolution.candidates}),
        )

        if resolution.status != RESOLVED_PENDING_FETCH or resolution.recording is None:
            evidence.source_recovery_status = resolution.status
            evidence.reason_code = resolution.reason_code
            evidence.reason_detail = resolution.reason_detail
            self.evidence[episode.episode_id] = evidence
            self.save()
            return evidence

        recording = resolution.recording
        evidence.source_file_id = recording.file_id
        record = self.ensure_source_file(
            recording, episode.requested_moment or _episode_card_timestamp(episode),
            episode.device_code)
        if record.recovery_status == "DOWNLOAD_FAILED":
            evidence.source_recovery_status = "DOWNLOAD_FAILED"
            evidence.reason_code = record.recovery_reason
            evidence.reason_detail = "original PS could not be preserved"
            self.evidence[episode.episode_id] = evidence
            self.save()
            return evidence

        probe = self.decode_source_file(record)
        evidence.source_width = record.source_width
        evidence.source_height = record.source_height
        if record.decode_status != "OK":
            evidence.source_recovery_status = "DECODE_FAILED"
            evidence.reason_code = record.recovery_reason or "decode_failed"
            evidence.reason_detail = str(probe.get("error") or "")[:160]
            self.evidence[episode.episode_id] = evidence
            self.save()
            return evidence

        return self._verify_frame(episode, evidence, record)

    def _verify_frame(self, episode: RequiredEpisode, evidence: EpisodeEvidence,
                      record: SourceFileRecord) -> EpisodeEvidence:
        moment = _episode_card_timestamp(episode)
        start = parse_timestamp(record.recording_start)
        if moment is None or start is None:
            evidence.source_recovery_status = "TIMESTAMP_UNRESOLVED"
            evidence.reason_code = "timestamp_unavailable"
            evidence.reason_detail = "cannot compute a file-relative offset"
            self.evidence[episode.episode_id] = evidence
            self.save()
            return evidence
        offset = (moment - start).total_seconds()
        if offset < 0 or (record.duration and offset > record.duration + 5):
            evidence.source_recovery_status = "TIMESTAMP_UNRESOLVED"
            evidence.reason_code = "timestamp_outside_ps_duration"
            evidence.reason_detail = f"offset={offset:.1f}s duration={record.duration}"
            self.evidence[episode.episode_id] = evidence
            self.save()
            return evidence

        try:
            sampled = self.decoder.sample_frame(Path(record.local_ps_path), offset)
        except Exception as exc:
            sampled = {"ok": False, "error": f"sampler_exception:{type(exc).__name__}"}
        if not sampled.get("ok"):
            evidence.source_recovery_status = "TIMESTAMP_UNRESOLVED"
            evidence.reason_code = "source_frame_not_decoded"
            evidence.reason_detail = str(sampled.get("error") or "")[:160]
            self.evidence[episode.episode_id] = evidence
            self.save()
            return evidence

        actual_offset = float(sampled.get("decoded_offset_seconds", offset))
        delta_ms = abs(actual_offset - offset) * 1000.0
        evidence.timestamp_delta_ms = round(delta_ms, 3)
        decoded_moment = start + timedelta(seconds=actual_offset)
        evidence.decoded_timestamp = decoded_moment.strftime("%Y-%m-%d %H:%M:%S")

        frame_bytes = sampled.get("frame_jpeg") or b""
        width = int(sampled.get("width") or record.source_width)
        height = int(sampled.get("height") or record.source_height)
        if not frame_bytes or width <= 0 or height <= 0:
            evidence.source_recovery_status = "TIMESTAMP_UNRESOLVED"
            evidence.reason_code = "verification_frame_empty"
            evidence.reason_detail = "sampler returned no usable frame"
            self.evidence[episode.episode_id] = evidence
            self.save()
            return evidence

        hint = (moment.strftime("%Y%m%dT%H%M%S"))
        frame_path = self.frames_dir / safe_name(episode.camera_id) / \
            f"{safe_name(episode.episode_id)}__{hint}.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = frame_path.with_name(frame_path.name + ".part")
        tmp.write_bytes(frame_bytes)
        os.replace(tmp, frame_path)
        evidence.verification_frame_path = str(frame_path)
        evidence.source_width = width
        evidence.source_height = height

        if delta_ms <= TIMESTAMP_OK_MS:
            evidence.timestamp_match_quality = "OK"
            evidence.source_recovery_status = "RECOVERED_SOURCE_NATIVE"
            evidence.reason_code = "source_frame_verified"
            evidence.reason_detail = (
                f"delta={delta_ms:.0f}ms method={evidence.source_file_resolution_method}")
        elif delta_ms <= TIMESTAMP_HARD_MS:
            evidence.timestamp_match_quality = "WEAK"
            evidence.source_recovery_status = "RECOVERED_SOURCE_NATIVE"
            evidence.reason_code = "TIMESTAMP_MATCH_WEAK"
            evidence.reason_detail = (
                f"delta={delta_ms:.0f}ms exceeds {TIMESTAMP_OK_MS:.0f}ms but within "
                f"{TIMESTAMP_HARD_MS:.0f}ms")
        else:
            evidence.timestamp_match_quality = "UNRESOLVED"
            evidence.source_recovery_status = "TIMESTAMP_UNRESOLVED"
            evidence.reason_code = "timestamp_delta_exceeds_hard_limit"
            evidence.reason_detail = (
                f"delta={delta_ms:.0f}ms exceeds {TIMESTAMP_HARD_MS:.0f}ms; no recording "
                "metadata documents a different timestamp semantic")
        self.evidence[episode.episode_id] = evidence
        self.save()
        return evidence

    # -- batch -------------------------------------------------------------- #

    def run(self, episodes: Sequence[RequiredEpisode], *,
            on_episode: Callable[[RequiredEpisode, EpisodeEvidence], None] | None = None,
            force: bool = False) -> list[EpisodeEvidence]:
        out: list[EpisodeEvidence] = []
        for episode in episodes:
            outcome = self.recover_episode(episode, force=force)
            out.append(outcome)
            if on_episode is not None:
                on_episode(episode, outcome)
        return out

    def _evidence_artifacts_intact(self, evidence: EpisodeEvidence) -> bool:
        """True only when a prior success is still backed by files on disk."""
        if evidence.source_recovery_status != "RECOVERED_SOURCE_NATIVE":
            return False
        frame = evidence.verification_frame_path
        if not frame or not Path(frame).is_file():
            return False
        record = self.files.get(str(evidence.source_file_id or ""))
        if record is None or record.decode_status != "OK":
            return False
        return Path(record.local_ps_path).is_file()


# ---------------------------------------------------------------- reporting


def build_summary(gold: GoldInput, evidence: Sequence[EpisodeEvidence],
                  files: Mapping[str, SourceFileRecord],
                  *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    status_counts = {status: 0 for status in RECOVERY_STATUSES}
    per_camera: dict[str, dict[str, int]] = {}
    resolutions: dict[str, int] = {}
    total_bytes = 0
    downloaded = 0
    failed = 0
    for record in files.values():
        if record.size_bytes:
            total_bytes += int(record.size_bytes)
        if record.recovery_status in ("DOWNLOADED", "PRESERVED"):
            downloaded += 1
        elif record.recovery_status in ("DOWNLOAD_FAILED", "HASH_MISMATCH"):
            failed += 1

    episodes = list(evidence)
    for row in episodes:
        status = row.source_recovery_status
        status_counts[status] = status_counts.get(status, 0) + 1
        bucket = per_camera.setdefault(row.camera_id, {
            "required": 0, "recovered": 0, "unresolved": 0, "failed": 0})
        bucket["required"] += 1
        if status == "RECOVERED_SOURCE_NATIVE":
            bucket["recovered"] += 1
        elif status in ("LINEAGE_UNRESOLVED", "LINEAGE_AMBIGUOUS", "TIMESTAMP_UNRESOLVED"):
            bucket["unresolved"] += 1
        else:
            bucket["failed"] += 1
        if row.source_width and row.source_height:
            key = f"{row.source_width}x{row.source_height}"
            resolutions[key] = resolutions.get(key, 0) + 1

    manual = [r for r in episodes if r.manual_missing_target]
    manual_recovered = sum(1 for r in manual
                           if r.source_recovery_status == "RECOVERED_SOURCE_NATIVE")
    observed = {row.episode_id for row in episodes}
    missing = [e.episode_id for e in gold.required if e.episode_id not in observed]

    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "gold": {
            "artifact": str(gold.path),
            "sha256": gold.sha256,
            "schema_version": gold.schema_version,
            "review_schema_version": gold.review_schema_version,
            "code_commit": gold.code_commit,
            "required_episode_count": gold.required_count,
            "manual_missing_target_count": len(gold.manual_required),
        },
        "episodes": {
            "completed_episodes": len(episodes),
            "episodes_without_status": missing,
            **{status: status_counts.get(status, 0) for status in RECOVERY_STATUSES},
        },
        "files": {
            "unique_source_files_resolved": len(files),
            "unique_source_files_downloaded": downloaded,
            "unique_source_files_failed": failed,
            "total_ps_bytes_preserved": total_bytes,
        },
        "source_resolution": {
            "resolution_histogram": dict(sorted(resolutions.items())),
        },
        "per_camera": dict(sorted(per_camera.items())),
        "manual_missing": {
            "manual_required_total": len(manual),
            "manual_required_recovered": manual_recovered,
            "manual_required_unresolved": len(manual) - manual_recovered,
        },
        "boundaries": {
            "gold_modified": False,
            "sealed_accessed": False,
            "training_started": False,
            "training_tiles_generated": False,
            "bbox_modified": False,
            "episode_merged_or_split": False,
            "ps_auto_deleted": False,
        },
        "extra": dict(extra or {}),
    }


def build_manifest(gold: GoldInput, summary: Mapping[str, Any], *,
                   code_commit: str, generated_at: str, config: Mapping[str, Any],
                   artifact_root: Path, evidence_path: Path,
                   verification_frame_count: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_at": generated_at,
        "code_commit": code_commit,
        "gold_input_sha256": gold.sha256,
        "source_evidence_sha256": sha256_file(evidence_path)
        if Path(evidence_path).is_file() else "",
        "artifact_root": str(artifact_root),
        "verification_frame_count": verification_frame_count,
        "config": dict(config),
        "counts": {
            "required_episode_count": gold.required_count,
            "recovered": summary["episodes"]["RECOVERED_SOURCE_NATIVE"],
            "unique_source_files_downloaded":
                summary["files"]["unique_source_files_downloaded"],
            "total_ps_bytes_preserved": summary["files"]["total_ps_bytes_preserved"],
        },
        "boundaries": dict(summary["boundaries"]),
    }
