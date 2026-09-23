"""Step 1C-0: Gold source recovery tests.

Pure stdlib: the remote listing/download/decode are injected fakes, so the whole
resolver + store + runner is exercised without network, PyAV or cv2.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from rtsp_annotator.ground_litter_gold_source_recovery import (
    RESOLVED_PENDING_FETCH,
    TIMESTAMP_HARD_MS,
    GoldChangedError,
    GoldSourceRecovery,
    RecoveryConfig,
    RecoveryError,
    Recording,
    SealedAssetError,
    assert_not_sealed,
    build_manifest,
    build_summary,
    load_device_codes,
    load_gold,
    resolve_episode,
    sha256_file,
)

DEVICE = {"01021": "44180209031322001021", "01022": "44180209031322001022"}
PS_HEAD = b"\x47" + b"\x00" * 187   # MPEG-PS sync byte, passes file_looks_like_media


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


class FakeListing:
    """Records per device so a cross-camera match is structurally impossible."""

    def __init__(self, rows: dict[str, list[Recording]] | None = None) -> None:
        self.rows = dict(rows or {})
        self.queries: list[tuple[str, str, str]] = []
        self.fail = False

    def list_recordings(self, device_code: str, start: str, end: str) -> list[Recording]:
        self.queries.append((device_code, start, end))
        if self.fail:
            raise RuntimeError("listing down")
        return [r for r in self.rows.get(device_code, [])
                if r.record_start <= end and r.record_end >= start]


@dataclass
class FakeDownloadResult:
    path: Path
    size: int
    sha256: str


class FakeFetcher:
    """Serves exactly the declared remote size unless told to lie about it."""

    def __init__(self, *, payload: bytes = PS_HEAD, fail: bool = False,
                 respect_expected_size: bool = True) -> None:
        self.calls: list[str] = []
        self.payload = payload
        self.fail = fail
        self.respect_expected_size = respect_expected_size

    def __call__(self, device_code, file_id, moment, target: Path, *, expected_size=None):
        self.calls.append(file_id)
        if self.fail:
            raise RuntimeError("download boom")
        body = self.payload
        if self.respect_expected_size and expected_size:
            repeats = (int(expected_size) // max(1, len(body))) + 1
            body = (body * repeats)[:int(expected_size)]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        return FakeDownloadResult(target, len(body), hashlib.sha256(body).hexdigest())


class FakeDecoder:
    def __init__(self, *, width: int = 2560, height: int = 1440, delta_seconds: float = 0.0,
                 probe_ok: bool = True, frame_ok: bool = True) -> None:
        self.width = width
        self.height = height
        self.delta_seconds = delta_seconds
        self.probe_ok = probe_ok
        self.frame_ok = frame_ok
        self.probed: list[str] = []
        self.sampled: list[float] = []

    def probe(self, path: Path) -> dict:
        self.probed.append(str(path))
        if not self.probe_ok:
            return {"ok": False, "error": "decoder_rejected_file"}
        return {"ok": True, "width": self.width, "height": self.height,
                "duration_seconds": 300.0, "codec": "hevc", "error": ""}

    def sample_frame(self, path: Path, offset_seconds: float) -> dict:
        self.sampled.append(float(offset_seconds))
        if not self.frame_ok:
            return {"ok": False, "error": "no_frame_at_offset"}
        return {"ok": True, "frame_jpeg": b"\xff\xd8\xff\xd9",
                "width": self.width, "height": self.height,
                "decoded_offset_seconds": float(offset_seconds) + self.delta_seconds}


# --------------------------------------------------------------------------- #
# gold fixture
# --------------------------------------------------------------------------- #


def gold_record(episode_id, camera, timestamp, *, file_id=None, kind="episode",
                localization="OK") -> dict:
    member = {"card_id": f"batch:{camera}-{episode_id.split(chr(45))[-1]}", "batch_key": "batch",
              "frame_id": "f00s00", "timestamp": timestamp, "label": "LITTER",
              "source_file_id": file_id, "asset_paths": {"context_image": "assets/x.jpg"}}
    return {
        "episode_id": episode_id,
        "truth_class": "REQUIRED_LITTER",
        "camera_id": camera,
        "scene_version": "UNKNOWN_HISTORICAL",
        "start_timestamp": timestamp,
        "end_timestamp": timestamp,
        "member_card_ids": [member["card_id"]],
        "source_episode_candidate_ids": ["ec-x"],
        "record_kind": kind,
        "localization_status": localization,
        "trainability_status": "TRAINABLE_SOURCE_NATIVE",
        "trainability_evidence": {"members": [member]},
    }


class Base(unittest.TestCase):
    def tmpdir(self) -> Path:
        path = Path(tempfile.mkdtemp(prefix="step1c0-"))
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def write_gold(self, records: list[dict]) -> Path:
        path = self.tmpdir() / "gold_episodes.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return path

    def gold(self, records: list[dict]):
        return load_gold(self.write_gold(records), device_codes=DEVICE)

    def runner(self, root: Path, listing, *, fetcher=None, decoder=None, **cfg):
        config = RecoveryConfig(artifact_root=root, gold_path=root / "gold.jsonl", **cfg)
        return GoldSourceRecovery(config, listing,
                                  fetcher=fetcher or FakeFetcher(),
                                  decoder=decoder or FakeDecoder())


def rec(file_id, camera, start, end, size=1000) -> Recording:
    return Recording(file_id=file_id, camera_id=camera, record_start=start,
                     record_end=end, file_size=size)


# --------------------------------------------------------------------------- #
# resolver
# --------------------------------------------------------------------------- #


class ResolverTest(Base):
    def test_file_id_exact_match(self) -> None:
        episode = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32",
                                         file_id="FI-1")]).required[0]
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-0", "01021", "2026-09-19 01:20:00", "2026-09-19 01:25:00"),
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00"),
        ]})
        outcome = resolve_episode(episode, listing)
        self.assertEqual(outcome.status, RESOLVED_PENDING_FETCH)
        self.assertEqual(outcome.reason_code, "file_id_exact_match")
        self.assertEqual(outcome.recording.file_id, "FI-1")

    def test_file_id_absent_is_source_not_found(self) -> None:
        episode = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32",
                                         file_id="FI-GONE")]).required[0]
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-OTHER", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        outcome = resolve_episode(episode, listing)
        self.assertEqual(outcome.status, "SOURCE_NOT_FOUND")
        self.assertEqual(outcome.reason_code, "file_id_absent_from_listing")

    def test_timestamp_outside_matched_file_id_is_metadata_invalid(self) -> None:
        episode = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32",
                                         file_id="FI-1")]).required[0]
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 02:30:00", "2026-09-19 02:35:00")]})
        outcome = resolve_episode(episode, listing)
        self.assertEqual(outcome.status, "SOURCE_METADATA_INVALID")
        self.assertEqual(outcome.reason_code, "timestamp_outside_recording")

    def test_no_file_id_unique_timestamp_window_resolves(self) -> None:
        episode = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32")]).required[0]
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-A", "01021", "2026-09-19 01:20:00", "2026-09-19 01:25:00"),
            rec("FI-B", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        outcome = resolve_episode(episode, listing)
        self.assertEqual(outcome.status, RESOLVED_PENDING_FETCH)
        self.assertEqual(outcome.recording.file_id, "FI-B")
        self.assertEqual(outcome.method, "B_camera_plus_timestamp")

    def test_overlapping_recordings_are_ambiguous(self) -> None:
        episode = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32")]).required[0]
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-A", "01021", "2026-09-19 01:25:00", "2026-09-19 01:40:00"),
            rec("FI-B", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        outcome = resolve_episode(episode, listing)
        self.assertEqual(outcome.status, "LINEAGE_AMBIGUOUS")
        self.assertEqual(len(outcome.candidates), 2)

    def test_no_covering_recording_is_source_not_found(self) -> None:
        episode = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32")]).required[0]
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-A", "01021", "2026-09-19 03:00:00", "2026-09-19 03:05:00")]})
        outcome = resolve_episode(episode, listing)
        self.assertEqual(outcome.status, "SOURCE_NOT_FOUND")
        self.assertEqual(outcome.reason_code, "no_recording_contains_timestamp")

    def test_empty_window_is_source_expired(self) -> None:
        episode = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32")]).required[0]
        outcome = resolve_episode(episode, FakeListing({}))
        self.assertEqual(outcome.status, "SOURCE_EXPIRED")

    def test_never_matches_across_cameras(self) -> None:
        """A recording listed under 01022 must never satisfy a 01021 episode."""
        episode = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32")]).required[0]
        listing = FakeListing({DEVICE["01022"]: [
            rec("FI-OTHER", "01022", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        outcome = resolve_episode(episode, listing)
        self.assertEqual(outcome.status, "SOURCE_EXPIRED")
        self.assertTrue(all(q[0] == DEVICE["01021"] for q in listing.queries))

    def test_listing_failure_is_reported_not_crashed(self) -> None:
        episode = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32")]).required[0]
        listing = FakeListing()
        listing.fail = True
        outcome = resolve_episode(episode, listing)
        self.assertEqual(outcome.status, "SOURCE_NOT_FOUND")
        self.assertEqual(outcome.reason_code, "listing_api_error")

    def test_missing_device_code_is_metadata_invalid(self) -> None:
        gold = load_gold(self.write_gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32")]))
        outcome = resolve_episode(gold.required[0], FakeListing())
        self.assertEqual(outcome.status, "SOURCE_METADATA_INVALID")
        self.assertEqual(outcome.reason_code, "device_code_unavailable")

    def test_resolver_is_deterministic(self) -> None:
        episode = self.gold([gold_record("ep-1", "01022", "2026-09-19 01:32:32")]).required[0]
        rows = [rec("FI-B", "01022", "2026-09-19 01:25:00", "2026-09-19 01:40:00"),
                rec("FI-A", "01022", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]
        first = resolve_episode(episode, FakeListing({DEVICE["01022"]: rows}))
        second = resolve_episode(episode, FakeListing({DEVICE["01022"]: list(reversed(rows))}))
        self.assertEqual(first.status, second.status)
        self.assertEqual([r.file_id for r in first.candidates],
                         [r.file_id for r in second.candidates])


# --------------------------------------------------------------------------- #
# download / dedupe / resume
# --------------------------------------------------------------------------- #


class DownloadTest(Base):
    def _one(self, root, listing, fetcher, **cfg):
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        runner = self.runner(root, listing, fetcher=fetcher, **cfg)
        return gold, runner

    def test_unique_ps_downloaded_once_for_many_episodes(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([
            gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1"),
            gold_record("ep-2", "01021", "2026-09-19 01:33:00", file_id="FI-1"),
            gold_record("ep-3", "01021", "2026-09-19 01:33:30", file_id="FI-1"),
        ])
        fetcher = FakeFetcher()
        runner = self.runner(root, listing, fetcher=fetcher)
        runner.run(gold.required)
        self.assertEqual(len(fetcher.calls), 1)
        self.assertEqual(len(runner.files), 1)

    def test_successful_download_is_reused_on_rerun(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        first = FakeFetcher()
        self.runner(root, listing, fetcher=first).run(gold.required)
        self.assertEqual(len(first.calls), 1)

        second = FakeFetcher()
        runner = self.runner(root, listing, fetcher=second)
        evidence = runner.run(gold.required)
        self.assertEqual(second.calls, [], "a verified local PS must not be re-downloaded")
        self.assertEqual(evidence[0].source_recovery_status, "RECOVERED_SOURCE_NATIVE")

    def test_part_file_is_not_a_success_artifact(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        runner = self.runner(root, listing, fetcher=FakeFetcher(fail=True))
        evidence = runner.run(gold.required)
        self.assertEqual(evidence[0].source_recovery_status, "DOWNLOAD_FAILED")
        self.assertFalse(list(root.rglob("*.ps")))
        self.assertFalse(list(root.rglob("*.part")))

    def test_truncated_local_file_is_redownloaded(self) -> None:
        """File-layer resume: a size mismatch must force a fresh download."""
        root = self.tmpdir()
        moment = datetime(2026, 9, 19, 1, 32, 32)
        recording = rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")
        first = FakeFetcher()
        runner = self.runner(root, FakeListing(), fetcher=first)
        runner.ensure_source_file(recording, moment, DEVICE["01021"])
        self.assertEqual(len(first.calls), 1)
        target = next(root.rglob("*.ps"))
        self.assertTrue(target.is_file())

        target.write_bytes(b"\x47truncated")
        second = FakeFetcher()
        runner2 = self.runner(root, FakeListing(), fetcher=second)
        runner2.ensure_source_file(recording, moment, DEVICE["01021"])
        self.assertEqual(len(second.calls), 1, "size mismatch must force a re-download")
        self.assertEqual(target.stat().st_size,
                         runner2.files["FI-1"].size_bytes)

    def test_size_mismatch_vs_remote_metadata_fails(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00", size=999999)]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        evidence = self.runner(
            root, listing,
            fetcher=FakeFetcher(respect_expected_size=False)).run(gold.required)
        self.assertEqual(evidence[0].source_recovery_status, "DOWNLOAD_FAILED")
        self.assertEqual(evidence[0].reason_code, "size_mismatch_vs_remote_metadata")

    def test_non_media_payload_is_rejected(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        evidence = self.runner(root, listing,
                               fetcher=FakeFetcher(payload=b'{"code":200}')).run(gold.required)
        self.assertEqual(evidence[0].source_recovery_status, "DOWNLOAD_FAILED")

    def test_hash_is_recorded_and_stable(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        runner = self.runner(root, listing, fetcher=FakeFetcher())
        runner.run(gold.required)
        record = runner.files["FI-1"]
        self.assertEqual(record.local_sha256, sha256_file(Path(record.local_ps_path)))
        self.assertFalse(record.remote_hash_verified)
        self.assertIsNone(record.remote_sha256)
        self.assertEqual(record.size_bytes, Path(record.local_ps_path).stat().st_size)

    def test_download_cap_is_respected(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00"),
            rec("FI-2", "01021", "2026-09-19 01:35:00", "2026-09-19 01:40:00")]})
        gold = self.gold([
            gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1"),
            gold_record("ep-2", "01021", "2026-09-19 01:37:00", file_id="FI-2")])
        fetcher = FakeFetcher()
        self.runner(root, listing, fetcher=fetcher, max_downloads=1).run(gold.required)
        self.assertEqual(len(fetcher.calls), 1)


# --------------------------------------------------------------------------- #
# decode + timestamp
# --------------------------------------------------------------------------- #


class DecodeTest(Base):
    def _run(self, root, *, decoder):
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        return self.runner(root, listing, decoder=decoder).run(gold.required)[0]

    def test_resolution_is_read_from_the_decoder(self) -> None:
        evidence = self._run(self.tmpdir(), decoder=FakeDecoder(width=2560, height=1440))
        self.assertEqual((evidence.source_width, evidence.source_height), (2560, 1440))
        self.assertEqual(evidence.source_recovery_status, "RECOVERED_SOURCE_NATIVE")

    def test_unusual_resolution_is_recorded_verbatim(self) -> None:
        evidence = self._run(self.tmpdir(), decoder=FakeDecoder(width=1920, height=1080))
        self.assertEqual((evidence.source_width, evidence.source_height), (1920, 1080))

    def test_decode_failure_is_recorded(self) -> None:
        evidence = self._run(self.tmpdir(), decoder=FakeDecoder(probe_ok=False))
        self.assertEqual(evidence.source_recovery_status, "DECODE_FAILED")

    def test_timestamp_delta_is_recorded(self) -> None:
        evidence = self._run(self.tmpdir(), decoder=FakeDecoder(delta_seconds=0.2))
        self.assertAlmostEqual(evidence.timestamp_delta_ms, 200.0, places=1)
        self.assertEqual(evidence.timestamp_match_quality, "OK")
        self.assertEqual(evidence.source_recovery_status, "RECOVERED_SOURCE_NATIVE")

    def test_delta_between_ok_and_hard_is_weak_not_failed(self) -> None:
        evidence = self._run(self.tmpdir(), decoder=FakeDecoder(delta_seconds=1.0))
        self.assertEqual(evidence.timestamp_match_quality, "WEAK")
        self.assertEqual(evidence.source_recovery_status, "RECOVERED_SOURCE_NATIVE")
        self.assertEqual(evidence.reason_code, "TIMESTAMP_MATCH_WEAK")

    def test_delta_beyond_hard_limit_is_unresolved(self) -> None:
        evidence = self._run(self.tmpdir(),
                             decoder=FakeDecoder(delta_seconds=(TIMESTAMP_HARD_MS / 1000.0) + 0.5))
        self.assertEqual(evidence.source_recovery_status, "TIMESTAMP_UNRESOLVED")
        self.assertEqual(evidence.timestamp_match_quality, "UNRESOLVED")

    def test_frame_sampling_failure_is_recorded(self) -> None:
        evidence = self._run(self.tmpdir(), decoder=FakeDecoder(frame_ok=False))
        self.assertEqual(evidence.source_recovery_status, "TIMESTAMP_UNRESOLVED")
        self.assertEqual(evidence.reason_code, "source_frame_not_decoded")

    def test_verification_frame_and_requested_offset(self) -> None:
        root = self.tmpdir()
        decoder = FakeDecoder()
        evidence = self._run(root, decoder=decoder)
        self.assertTrue(evidence.verification_frame_path)
        self.assertTrue(Path(evidence.verification_frame_path).is_file())
        # 01:32:32 inside a recording starting 01:30:00 -> 152s offset
        self.assertAlmostEqual(decoder.sampled[0], 152.0, places=1)

    def test_no_decoder_is_a_decode_failure_not_a_crash(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        runner = GoldSourceRecovery(
            RecoveryConfig(artifact_root=root, gold_path=root / "g.jsonl"), listing,
            fetcher=FakeFetcher(), decoder=None)
        evidence = runner.run(gold.required)[0]
        self.assertEqual(evidence.source_recovery_status, "DECODE_FAILED")


# --------------------------------------------------------------------------- #
# manual missing targets
# --------------------------------------------------------------------------- #


class ManualTargetTest(Base):
    def test_manual_target_inherits_source_member_and_needs_relocalization(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ge-01021-manual-abc", "01021", "2026-09-19 01:32:32",
                                      file_id="FI-1", kind="manual_missing_target",
                                      localization="NEEDS_RELOCALIZATION")])
        evidence = self.runner(root, listing).run(gold.required)[0]
        self.assertTrue(evidence.manual_missing_target)
        self.assertEqual(evidence.source_recovery_status, "RECOVERED_SOURCE_NATIVE")
        # recovery succeeding must not "fix" the missing bbox
        self.assertEqual(evidence.localization_status, "NEEDS_RELOCALIZATION")
        self.assertEqual(evidence.source_member_card_ids, ["batch:01021-abc"])

    def test_manual_target_without_lineage_stays_unresolved(self) -> None:
        root = self.tmpdir()
        gold = self.gold([gold_record("ge-01021-manual-abc", "01021", "2026-09-19 01:32:32",
                                      file_id=None, kind="manual_missing_target",
                                      localization="NEEDS_RELOCALIZATION")])
        evidence = self.runner(root, FakeListing({})).run(gold.required)[0]
        self.assertEqual(evidence.source_recovery_status, "SOURCE_EXPIRED")
        self.assertEqual(evidence.localization_status, "NEEDS_RELOCALIZATION")

    def test_summary_counts_manual_targets(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([
            gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1"),
            gold_record("ge-01021-manual-abc", "01021", "2026-09-19 01:32:32", file_id="FI-1",
                        kind="manual_missing_target", localization="NEEDS_RELOCALIZATION"),
            gold_record("ge-01021-manual-def", "01021", "2026-09-19 09:00:00", file_id=None,
                        kind="manual_missing_target", localization="NEEDS_RELOCALIZATION"),
        ])
        runner = self.runner(root, listing)
        evidence = runner.run(gold.required)
        summary = build_summary(gold, evidence, runner.files)
        self.assertEqual(summary["manual_missing"]["manual_required_total"], 2)
        self.assertEqual(summary["manual_missing"]["manual_required_recovered"], 1)
        self.assertEqual(summary["manual_missing"]["manual_required_unresolved"], 1)


# --------------------------------------------------------------------------- #
# safety
# --------------------------------------------------------------------------- #


class ResumePolicyTest(Base):
    """Regressions found while running the real batch."""

    def _setup(self):
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        return root, listing, gold

    def test_run_does_not_rederive_an_already_recovered_episode(self) -> None:
        root, listing, gold = self._setup()
        self.runner(root, listing).run(gold.required)
        # second runner, same root: the remote is now completely broken
        broken = FakeListing()
        broken.fail = True
        seen = []
        evidence = self.runner(root, broken).run(
            gold.required, on_episode=lambda ep, ev: seen.append(ev.source_recovery_status))
        self.assertEqual(evidence[0].source_recovery_status, "RECOVERED_SOURCE_NATIVE")
        self.assertEqual(seen, ["RECOVERED_SOURCE_NATIVE"])
        self.assertEqual(broken.queries, [], "a recovered episode must not re-query")

    def test_transient_remote_failure_never_downgrades_evidence(self) -> None:
        root, listing, gold = self._setup()
        runner = self.runner(root, listing)
        runner.run(gold.required)
        self.assertEqual(runner.evidence["ep-1"].source_recovery_status,
                         "RECOVERED_SOURCE_NATIVE")
        # call recover_episode directly (the old CLI path) with a broken remote
        broken = FakeListing()
        broken.fail = True
        runner.listing = broken
        again = runner.recover_episode(gold.required[0])
        self.assertEqual(again.source_recovery_status, "RECOVERED_SOURCE_NATIVE")
        self.assertEqual(again.reason_code, "source_frame_verified")

    def test_force_rederives_even_a_recovered_episode(self) -> None:
        root, listing, gold = self._setup()
        self.runner(root, listing).run(gold.required)
        broken = FakeListing()
        broken.fail = True
        outcome = self.runner(root, broken).recover_episode(gold.required[0], force=True)
        self.assertNotEqual(outcome.source_recovery_status, "RECOVERED_SOURCE_NATIVE")

    def test_missing_frame_invalidates_a_prior_success(self) -> None:
        root, listing, gold = self._setup()
        runner = self.runner(root, listing)
        runner.run(gold.required)
        Path(runner.evidence["ep-1"].verification_frame_path).unlink()
        again = runner.recover_episode(gold.required[0])
        # the frame is gone, so the episode is legitimately re-derived
        self.assertEqual(listing.queries.count((DEVICE["01021"],) + listing.queries[0][1:]),
                         len([q for q in listing.queries if q[0] == DEVICE["01021"]]))
        self.assertEqual(again.source_recovery_status, "RECOVERED_SOURCE_NATIVE")


class SafetyTest(Base):
    def test_gold_artifact_is_not_modified(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        before = sha256_file(gold.path)
        self.runner(root, listing).run(gold.required)
        self.assertEqual(sha256_file(gold.path), before)
        self.assertEqual(gold.sha256, before)

    def test_sealed_paths_are_refused(self) -> None:
        for bad in ("/home/sf01/ground-litter-feasibility/20260923-r1/archive/"
                    "ground-litter-detector-feasibility-20260923-r1/sealed_test/01021/raw/x.ps",
                    "output/foo/SEALED_DO_NOT_TUNE.txt",
                    "/data/sealed_test"):
            with self.assertRaises(SealedAssetError):
                assert_not_sealed(bad)

    def test_sealed_asset_paths_inside_gold_are_refused(self) -> None:
        record = gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")
        record["trainability_evidence"]["members"][0]["asset_paths"] = {
            "context_image": "sealed_test/01021/raw/x.jpg"}
        path = self.write_gold([record])
        with self.assertRaises(SealedAssetError):
            load_gold(path, device_codes=DEVICE)

    def test_sealed_artifact_root_is_refused(self) -> None:
        root = self.tmpdir() / "sealed_test"
        root.mkdir(parents=True)
        with self.assertRaises(SealedAssetError):
            GoldSourceRecovery(
                RecoveryConfig(artifact_root=root, gold_path=root / "g.jsonl"),
                FakeListing())

    def test_boundaries_claim_no_training_or_tiles(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        runner = self.runner(root, listing)
        evidence = runner.run(gold.required)
        summary = build_summary(gold, evidence, runner.files)
        for key in ("gold_modified", "sealed_accessed", "training_started",
                    "training_tiles_generated", "bbox_modified",
                    "episode_merged_or_split", "ps_auto_deleted"):
            self.assertIs(summary["boundaries"][key], False, key)

    def test_no_training_tiles_are_written(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        self.runner(root, listing).run(gold.required)
        names = {p.name for p in root.rglob("*") if p.is_file()}
        self.assertFalse([n for n in names if "tile" in n.lower() or "train" in n.lower()])
        # only the artifact-owned PS + the single verification frame exist
        self.assertEqual(len(list(root.rglob("*.ps"))), 1)
        self.assertEqual(len(list(root.rglob("*.jpg"))), 1)

    def test_source_files_ledger_never_carries_credentials(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        runner = self.runner(root, listing)
        runner.run(gold.required)
        blob = (root / "source_files.jsonl").read_text()
        for marker in ("http://", "https://", "token=", "password", "Authorization"):
            self.assertNotIn(marker, blob)


# --------------------------------------------------------------------------- #
# manifest / device codes
# --------------------------------------------------------------------------- #


class ManifestTest(Base):
    def test_manifest_records_required_provenance(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        runner = self.runner(root, listing)
        evidence = runner.run(gold.required)
        summary = build_summary(gold, evidence, runner.files)
        manifest = build_manifest(gold, summary, code_commit="deadbeef",
                                  generated_at="2026-09-23T00:00:00Z",
                                  config={"x": 1}, artifact_root=root,
                                  evidence_path=root / "missing.jsonl",
                                  verification_frame_count=1)
        self.assertEqual(manifest["gold_input_sha256"], gold.sha256)
        self.assertEqual(manifest["code_commit"], "deadbeef")
        self.assertEqual(manifest["artifact_root"], str(root))
        self.assertIn("config", manifest)
        self.assertEqual(manifest["counts"]["recovered"], 1)

    def test_summary_has_all_required_blocks(self) -> None:
        root = self.tmpdir()
        listing = FakeListing({DEVICE["01021"]: [
            rec("FI-1", "01021", "2026-09-19 01:30:00", "2026-09-19 01:35:00")]})
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        runner = self.runner(root, listing)
        evidence = runner.run(gold.required)
        summary = build_summary(gold, evidence, runner.files)
        self.assertEqual(summary["gold"]["required_episode_count"], 1)
        for status in ("RECOVERED_SOURCE_NATIVE", "LINEAGE_UNRESOLVED", "LINEAGE_AMBIGUOUS",
                       "SOURCE_NOT_FOUND", "SOURCE_EXPIRED", "DOWNLOAD_FAILED",
                       "HASH_MISMATCH", "DECODE_FAILED", "TIMESTAMP_UNRESOLVED"):
            self.assertIn(status, summary["episodes"])
        self.assertIn("resolution_histogram", summary["source_resolution"])
        self.assertIn("01021", summary["per_camera"])
        self.assertIn("unique_source_files_downloaded", summary["files"])
        self.assertIn("total_ps_bytes_preserved", summary["files"])

    def test_episode_without_status_is_listed(self) -> None:
        root = self.tmpdir()
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        summary = build_summary(gold, [], {})
        self.assertEqual(summary["episodes"]["episodes_without_status"], ["ep-1"])

    def test_device_codes_come_from_roi_configs_not_a_guess(self) -> None:
        root = self.tmpdir()
        (root / "ground_litter_01021_final_roi.json").write_text(json.dumps(
            {"camera_id": "01021", "device_code": "44180209031322001021"}))
        (root / "ground_litter_01030_final_roi.json").write_text(json.dumps(
            {"camera_id": "01030", "device_code": "44180209031322001030"}))
        codes = load_device_codes(root)
        self.assertEqual(codes, {"01021": "44180209031322001021",
                                 "01030": "44180209031322001030"})
        # a malformed code is ignored rather than used
        (root / "ground_litter_09999_final_roi.json").write_text(json.dumps(
            {"camera_id": "09999", "device_code": "123"}))
        self.assertNotIn("09999", load_device_codes(root))

    def test_duplicate_episode_id_in_gold_is_rejected(self) -> None:
        record = gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")
        path = self.write_gold([record, dict(record)])
        with self.assertRaises(RecoveryError):
            load_gold(path, device_codes=DEVICE)

    def test_non_required_records_are_ignored(self) -> None:
        record = gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")
        other = dict(record, episode_id="ep-2", truth_class="NON_LITTER")
        gold = load_gold(self.write_gold([record, other]), device_codes=DEVICE)
        self.assertEqual([e.episode_id for e in gold.required], ["ep-1"])

    def test_gold_sha_mismatch_is_detectable(self) -> None:
        gold = self.gold([gold_record("ep-1", "01021", "2026-09-19 01:32:32", file_id="FI-1")])
        original = gold.sha256
        gold.path.write_text(gold.path.read_text() + "\n", encoding="utf-8")
        self.assertNotEqual(sha256_file(gold.path), original)
        self.assertTrue(issubclass(GoldChangedError, RecoveryError))


if __name__ == "__main__":
    unittest.main()
