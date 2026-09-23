"""Step 1C-1R: truth reconciliation tests (§23).

Pure stdlib, no cv2/network.  The fixture is a synthetic but fully shaped Step 1C-1
localization artifact: 109 VERIFIED_BBOX + 11 LOCALIZATION_UNRESOLVED (frozen) and
60 TRUTH_REVIEW_REQUIRED (the only in-scope set), i.e. 180 REQUIRED_LITTER episodes.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from rtsp_annotator.ground_litter_localization_review import (
    ReviewError,
    SealedAssetError,
    sha256_file,
)
from rtsp_annotator.ground_litter_truth_reconciliation import (
    DECISION_TO_TRUTH_CLASS,
    DECISIONS,
    EFFECTIVE_BUCKETS,
    GENERATOR_VERSION,
    REVIEW_SCHEMA_VERSION,
    SCHEMA_VERSION,
    EpisodeContext,
    ReconciliationError,
    ReviewState,
    build_manifest,
    build_overlay_rows,
    build_summary,
    build_upstream_provenance,
    eligibility,
    load_reconciliation_input,
    verify_preflight,
    write_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "ground_litter_truth_ui"
CLI = ROOT / "scripts" / "reconcile_ground_litter_truth.py"

W, H = 2560, 1440
CAMERAS = ("01021", "01022", "01027", "01028", "01030")
ORIGINS = ("historical_litter", "box_wrong", "manual_missing_target", "split_derived")
REASON_BY_ORIGIN = {
    "historical_litter": "垃圾太小",
    "box_wrong": "不是垃圾",
    "manual_missing_target": "无法判断",
    "split_derived": "框内目标太多",
}
N_VERIFIED, N_UNRESOLVED, N_REVIEW = 109, 11, 60


def _row(episode_id, camera, origin, status, *, reason="", bbox=None, point=None,
         point_source=None, source_recovery="RECOVERED_SOURCE_NATIVE", frame=None) -> dict:
    row = {
        "episode_id": episode_id, "truth_class": "REQUIRED_LITTER",
        "camera_id": camera, "scene_version": "UNKNOWN_HISTORICAL",
        "origin": origin, "source_timestamp": "2026-08-12 03:04:05",
        "source_file_id": "ps-1", "source_width": W, "source_height": H,
        "verification_frame_path": frame or "",
        "localization_status": status, "localization_decision": status,
        "verified_bbox": [100.0, 100.0, 140.0, 140.0] if status == "VERIFIED_BBOX" else None,
        "original_bbox": bbox, "original_point": point,
        "original_point_source": point_source,
        "truth_review_reason": reason,
    }
    return row


class Fixture:
    """Builds a real on-disk Step 1C-1-shaped localization artifact."""

    def __init__(self, root: Path, *, source_recovery_overrides=None,
                 outside_frame_for=None) -> None:
        self.root = root
        self.gold = root / "gold_episodes.jsonl"
        self.recovery = root / "recovery"
        self.localization = root / "localization"
        self.manifest = root / "gold" / "MANIFEST.json"
        self.out = root / "out"
        self.state = root / "state" / "review_state.json"
        self.outside_frame = root / "outside" / "other.jpg"
        overrides = source_recovery_overrides or {}

        self.recovery.mkdir(parents=True, exist_ok=True)
        frames = self.recovery / "verification_frames"
        frames.mkdir(parents=True, exist_ok=True)
        self.localization.mkdir(parents=True, exist_ok=True)
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.outside_frame.parent.mkdir(parents=True, exist_ok=True)
        self.outside_frame.write_bytes(b"\xff\xd8\xff\xd9")

        self.in_scope_ids: list[str] = []
        self.verified_ids: list[str] = []
        self.unresolved_ids: list[str] = []
        rows: list[dict] = []

        def frame_for(episode_id: str) -> str:
            if outside_frame_for and episode_id == outside_frame_for:
                return str(self.outside_frame)
            path = frames / f"{episode_id}.jpg"
            path.write_bytes(b"\xff\xd8\xff\xd9")
            return str(path)

        for i in range(N_VERIFIED):
            episode_id = f"ge-verified-{i:03d}"
            self.verified_ids.append(episode_id)
            rows.append(_row(episode_id, CAMERAS[i % len(CAMERAS)], "historical_litter",
                             "VERIFIED_BBOX", bbox=[100.0, 100.0, 140.0, 140.0],
                             frame=frame_for(episode_id)))
        for i in range(N_UNRESOLVED):
            episode_id = f"ge-unresolved-{i:03d}"
            self.unresolved_ids.append(episode_id)
            rows.append(_row(episode_id, CAMERAS[i % len(CAMERAS)], "historical_litter",
                             "LOCALIZATION_UNRESOLVED",
                             bbox=[200.0, 200.0, 240.0, 240.0],
                             frame=frame_for(episode_id)))
        for i in range(N_REVIEW):
            episode_id = f"ge-review-{i:03d}"
            origin = ORIGINS[i % len(ORIGINS)]
            self.in_scope_ids.append(episode_id)
            bbox = [300.0, 300.0, 340.0, 340.0] if origin != "manual_missing_target" else None
            point = ({"x_norm": 0.5, "y_norm": 0.5,
                      "clicked_image_width": 112, "clicked_image_height": 112}
                     if origin == "manual_missing_target" else None)
            point_source = ({
                "ok": True, "x": 1500.0, "y": 900.0, "point_source": [1500.0, 900.0],
                "crop_box": [1444.0, 844.0, 1556.0, 956.0],
                "derived_crop_size": [112, 112], "recorded_crop_size": [112, 112],
                "reason": "derived_crop_size_matches_recorded",
            } if origin == "manual_missing_target" else None)
            rows.append(_row(episode_id, CAMERAS[i % len(CAMERAS)], origin,
                             "TRUTH_REVIEW_REQUIRED",
                             reason=REASON_BY_ORIGIN[origin], bbox=bbox, point=point,
                             point_source=point_source,
                             frame=frame_for(episode_id)))
        # a decoy that is not REQUIRED_LITTER: it must never reach the overlay
        decoy = _row("ge-decoy", CAMERAS[0], "historical_litter", "VERIFIED_BBOX",
                     bbox=[1.0, 1.0, 2.0, 2.0], frame=frame_for("ge-decoy"))
        decoy["truth_class"] = "NON_LITTER"
        rows.append(decoy)

        self.gold.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.gold_manifest = self.manifest
        self.gold_manifest.write_text(json.dumps({"step": "1b"}), encoding="utf-8")

        evidence = []
        for row in rows:
            if row["truth_class"] != "REQUIRED_LITTER":
                continue
            evidence.append({
                "episode_id": row["episode_id"], "camera_id": row["camera_id"],
                "source_recovery_status": overrides.get(
                    row["episode_id"], "RECOVERED_SOURCE_NATIVE"),
                "source_file_id": "ps-1"})
        (self.recovery / "episode_source_evidence.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in evidence), encoding="utf-8")

        (self.localization / "localizations.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows
                    if r["truth_class"] == "REQUIRED_LITTER"), encoding="utf-8")
        (self.localization / "MANIFEST.json").write_text(
            json.dumps({"schema_version": "gold_localization_v1",
                        "code_commit": "809ce49196dadc639fe90438da3c27057baa34bd"}),
            encoding="utf-8")
        (self.localization / "SUMMARY.json").write_text(
            json.dumps({"review": {"reviewed": 180}}), encoding="utf-8")

    def load(self):
        return load_reconciliation_input(self.gold, self.recovery, self.localization,
                                         gold_manifest=self.gold_manifest)


def ctx(**overrides) -> EpisodeContext:
    base = {
        "episode_id": "ge-x", "camera_id": "01021", "scene_version": "UNKNOWN_HISTORICAL",
        "origin": "historical_litter", "source_timestamp": "2026-08-12 03:04:05",
        "source_file_id": "ps-1", "source_width": W, "source_height": H,
        "verification_frame_path": "/tmp/x.jpg",
        "source_recovery_status": "RECOVERED_SOURCE_NATIVE",
        "original_truth_class": "REQUIRED_LITTER",
        "localization_status": "TRUTH_REVIEW_REQUIRED",
        "localization_decision": "TRUTH_REVIEW_REQUIRED",
        "localization_verified_bbox": None, "original_bbox": [1.0, 2.0, 3.0, 4.0],
        "original_point": None, "original_point_source": None,
        "in_scope": True, "prior_truth_review_reason": "垃圾太小",
    }
    base.update(overrides)
    return EpisodeContext(**base)


def decision_row(decision: str, **overrides) -> dict:
    row = {"reconciled_truth_class": DECISION_TO_TRUTH_CLASS[decision],
           "reconciliation_decision": decision,
           "effective_truth_bucket": (decision if decision == "IDENTITY_AMBIGUOUS"
                                      else DECISION_TO_TRUTH_CLASS[decision]),
           "exclusion_reason": None if decision == "KEEP_REQUIRED" else decision,
           "review_reason_optional": "because"}
    row.update(overrides)
    return row


def decide_all(data, state: ReviewState) -> None:
    for index, episode in enumerate(data.in_scope):
        decision = DECISIONS[index % len(DECISIONS)]
        state.decide(episode, decision,
                     reason="one box, several objects" if decision == "IDENTITY_AMBIGUOUS"
                     else "")


class Base(unittest.TestCase):
    def tmpdir(self) -> Path:
        path = Path(tempfile.mkdtemp(prefix="step1c1r-"))
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def fixture(self, **kwargs) -> Fixture:
        return Fixture(self.tmpdir(), **kwargs)


# --------------------------------------------------------------------------- #
# §23 filtering: only the 60 TRUTH_REVIEW_REQUIRED are in scope
# --------------------------------------------------------------------------- #


class FilteringTest(Base):
    def test_only_the_60_truth_review_required_are_in_scope(self) -> None:
        f = self.fixture()
        data = f.load()
        self.assertEqual(data.required_before, 180)     # decoy NON_LITTER dropped
        self.assertEqual(len(data.episodes), 180)
        self.assertEqual(len(data.in_scope), N_REVIEW)
        self.assertEqual({e.episode_id for e in data.in_scope}, set(f.in_scope_ids))

    def test_verified_and_unresolved_are_frozen_out_of_the_queue(self) -> None:
        f = self.fixture()
        data = f.load()
        scope = {e.episode_id for e in data.in_scope}
        self.assertFalse(scope & set(f.verified_ids))
        self.assertFalse(scope & set(f.unresolved_ids))
        preflight = verify_preflight(data)
        self.assertEqual(preflight["verified_bbox_before"], N_VERIFIED)
        self.assertEqual(preflight["localization_unresolved"], N_UNRESOLVED)
        self.assertEqual(preflight["truth_review_required"], N_REVIEW)
        self.assertTrue(preflight["sum_matches"])
        self.assertEqual(N_VERIFIED + N_UNRESOLVED + N_REVIEW, 180)

    def test_deciding_a_frozen_episode_is_refused(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        frozen = next(e for e in data.episodes if e.episode_id in f.verified_ids)
        with self.assertRaises(ReviewError) as caught:
            state.decide(frozen, "NON_LITTER")
        self.assertIn("frozen", str(caught.exception))
        self.assertEqual(state.decisions, {})

    def test_the_60_all_carry_their_prior_reason_and_no_verified_bbox(self) -> None:
        f = self.fixture()
        data = f.load()
        for episode in data.in_scope:
            self.assertTrue(episode.prior_truth_review_reason)
            self.assertIsNone(episode.localization_verified_bbox)
        reasons = {e.prior_truth_review_reason for e in data.in_scope}
        self.assertEqual(reasons, set(REASON_BY_ORIGIN.values()))

    def test_duplicate_or_missing_input_is_refused(self) -> None:
        f = self.fixture()
        loc = f.localization / "localizations.jsonl"
        original = loc.read_text(encoding="utf-8")
        row = json.loads(original.splitlines()[0])
        loc.write_text(original + json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaises(ReconciliationError):
            f.load()
        loc.write_text("", encoding="utf-8")
        with self.assertRaises(ReconciliationError):
            f.load()
        loc.unlink()
        with self.assertRaises(FileNotFoundError):
            f.load()

    def test_sealed_assets_are_refused(self) -> None:
        f = self.fixture()
        with self.assertRaises(SealedAssetError):
            load_reconciliation_input(f.root / "sealed_test" / "gold.jsonl",
                                      f.recovery, f.localization)
        with self.assertRaises(SealedAssetError):
            load_reconciliation_input(f.gold, f.recovery,
                                      f.root / "SEALED_DO_NOT_TUNE" / "loc")


# --------------------------------------------------------------------------- #
# §23 all five decisions persist
# --------------------------------------------------------------------------- #


class DecisionTest(Base):
    def test_every_decision_persists_with_its_truth_mapping(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        episode = data.in_scope[0]
        for decision in DECISIONS:
            state.decide(episode, decision, reason="r" if decision == "IDENTITY_AMBIGUOUS" else "")
            row = state.get(episode.episode_id)
            self.assertEqual(row["reconciliation_decision"], decision)
            self.assertEqual(row["reconciled_truth_class"],
                             DECISION_TO_TRUTH_CLASS[decision])
            self.assertEqual(row["review_status"], "human_reviewed")
            self.assertEqual(row["previous_localization_status"],
                             "TRUTH_REVIEW_REQUIRED")
            self.assertEqual(row["previous_truth_review_reason"], "垃圾太小")
            self.assertEqual(row["review_schema_version"], REVIEW_SCHEMA_VERSION)
        self.assertEqual(len(state.decisions), 1)
        self.assertEqual(len(state.audit_trail), len(DECISIONS))

    def test_identity_ambiguous_is_not_uncertain_and_needs_a_reason(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        episode = data.in_scope[0]
        with self.assertRaises(ReviewError):
            state.decide(episode, "IDENTITY_AMBIGUOUS")
        with self.assertRaises(ReviewError):
            state.decide(episode, "IDENTITY_AMBIGUOUS", reason="   ")
        row = state.decide(episode, "IDENTITY_AMBIGUOUS", reason="one box, two objects")
        self.assertIsNone(row["reconciled_truth_class"])
        self.assertEqual(row["effective_truth_bucket"], "IDENTITY_AMBIGUOUS")
        self.assertNotEqual(row["effective_truth_bucket"], "UNCERTAIN")
        self.assertNotEqual(row["reconciled_truth_class"], "UNCERTAIN")

    def test_unknown_decision_is_refused(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        with self.assertRaises(ReviewError):
            state.decide(data.in_scope[0], "BOX_OK")
        with self.assertRaises(ReviewError):
            state.decide(data.in_scope[0], "KEEP")

    def test_keep_required_never_invents_a_bbox(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        row = state.decide(data.in_scope[0], "KEEP_REQUIRED")
        self.assertIsNone(row["verified_bbox"])
        overlay, manifest = build_overlay_rows(data, state)
        kept = next(r for r in overlay if r["episode_id"] == data.in_scope[0].episode_id)
        self.assertIsNone(kept["verified_bbox"])
        self.assertEqual(kept["reconciled_truth_class"], "REQUIRED_LITTER")
        self.assertIsNone(kept["reason"])

    def test_reset_clears_a_decision_and_is_audited(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        episode = data.in_scope[0]
        state.decide(episode, "UNCERTAIN")
        state.reset(episode.episode_id)
        self.assertIsNone(state.get(episode.episode_id))
        self.assertEqual(state.audit_trail[-1]["action"], "reset")
        self.assertEqual(state.progress(data)["reviewed"], 0)


# --------------------------------------------------------------------------- #
# §23 overlay: upstream artifacts are untouched, truth is correct
# --------------------------------------------------------------------------- #


class OverlayTest(Base):
    def test_overlay_covers_all_180_and_freezes_out_of_scope_truth(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        decide_all(data, state)
        overlay, manifest = build_overlay_rows(data, state)
        self.assertEqual(len(overlay), 180)
        self.assertEqual(len(manifest), 180)
        by_id = {r["episode_id"]: r for r in overlay}
        for episode_id in f.verified_ids + f.unresolved_ids:
            row = by_id[episode_id]
            self.assertFalse(row["in_scope"])
            self.assertIsNone(row["reconciliation_decision"])
            self.assertEqual(row["effective_truth_class"], "REQUIRED_LITTER")
            self.assertEqual(row["original_truth_class"], "REQUIRED_LITTER")
        for row in overlay:
            self.assertEqual(row["original_truth_class"], "REQUIRED_LITTER")
            self.assertIn(row["effective_truth_bucket"], EFFECTIVE_BUCKETS)

    def test_overlay_effective_truth_matches_each_decision(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        decide_all(data, state)
        overlay, _ = build_overlay_rows(data, state)
        by_id = {r["episode_id"]: r for r in overlay}
        for index, episode in enumerate(data.in_scope):
            decision = DECISIONS[index % len(DECISIONS)]
            row = by_id[episode.episode_id]
            self.assertEqual(row["reconciliation_decision"], decision)
            self.assertEqual(row["reconciled_truth_class"],
                             DECISION_TO_TRUTH_CLASS[decision])
            self.assertEqual(row["effective_truth_class"],
                             DECISION_TO_TRUTH_CLASS[decision])
            if decision == "IDENTITY_AMBIGUOUS":
                self.assertIsNone(row["effective_truth_class"])
                self.assertEqual(row["effective_truth_bucket"], "IDENTITY_AMBIGUOUS")
                self.assertEqual(
                    row["evaluation_semantics"],
                    "excluded_from_automated_bbox_scoring_pending_identity")
            else:
                self.assertEqual(row["effective_truth_bucket"],
                                 DECISION_TO_TRUTH_CLASS[decision])

    def test_pending_in_scope_rows_are_never_silently_required(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        overlay, _ = build_overlay_rows(data, state)
        pending = [r for r in overlay if r["in_scope"]]
        self.assertEqual(len(pending), N_REVIEW)
        for row in pending:
            self.assertEqual(row["effective_truth_bucket"], "PENDING_RECONCILIATION")
            self.assertIsNone(row["effective_truth_class"])
            self.assertNotEqual(row["effective_truth_class"], "REQUIRED_LITTER")
            self.assertEqual(row["effective_truth_class"], None)
            self.assertEqual(row["evaluation_semantics"],
                             "excluded_pending_human_reconciliation")
            self.assertTrue(row["training_excluded"])

    def test_build_never_modifies_gold_or_localization(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        decide_all(data, state)
        before = (sha256_file(f.gold), sha256_file(f.localization / "localizations.jsonl"),
                  sha256_file(f.recovery / "episode_source_evidence.jsonl"),
                  sha256_file(f.manifest))
        overlay, manifest_rows = build_overlay_rows(data, state)
        summary = build_summary(data, state, overlay, manifest_rows, verify_preflight(data))
        payload = build_manifest(
            data, summary, code_commit="0" * 40, generated_at="2026-09-23T00:00:00Z",
            config={}, artifact_root=f.out, overlay_path=f.out / "truth_reconciliation.jsonl",
            manifest_path=f.out / "training_episode_manifest.jsonl",
            review_state_path=f.state, provenance={})
        write_outputs(f.out, overlay=overlay, manifest_rows=manifest_rows,
                      summary=summary, manifest=payload)
        after = (sha256_file(f.gold), sha256_file(f.localization / "localizations.jsonl"),
                 sha256_file(f.recovery / "episode_source_evidence.jsonl"),
                 sha256_file(f.manifest))
        self.assertEqual(before, after)
        # the written manifest fingerprints the bytes actually on disk
        written = json.loads((f.out / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(written["gold_input_sha256"], data.gold_sha256)
        self.assertEqual(written["localization_input_sha256"], data.localization_sha256)
        self.assertEqual(written["truth_reconciliation_sha256"],
                         sha256_file(f.out / "truth_reconciliation.jsonl"))
        self.assertEqual(written["training_episode_manifest_sha256"],
                         sha256_file(f.out / "training_episode_manifest.jsonl"))
        self.assertEqual(written["summary_sha256"], sha256_file(f.out / "SUMMARY.json"))
        self.assertNotEqual(written["truth_reconciliation_sha256"], "")

    def test_summary_counts_are_consistent(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        decide_all(data, state)
        overlay, manifest_rows = build_overlay_rows(data, state)
        summary = build_summary(data, state, overlay, manifest_rows, verify_preflight(data))
        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["generator_version"], GENERATOR_VERSION)
        for decision in DECISIONS:
            self.assertEqual(summary["reconciliation"][decision], N_REVIEW // 5)
        self.assertEqual(sum(summary["reconciliation"].values()), N_REVIEW)
        # 109 verified + 11 unresolved (all REQUIRED_LITTER) + 12 KEEP_REQUIRED
        self.assertEqual(summary["effective_truth_totals"]["REQUIRED_LITTER"],
                         120 + N_REVIEW // 5)
        self.assertEqual(summary["effective_truth_totals"]["IDENTITY_AMBIGUOUS"],
                         N_REVIEW // 5)
        self.assertEqual(sum(summary["effective_truth_totals"].values()), 180)
        self.assertEqual(summary["pending_reconciliation_count"], 0)
        self.assertEqual(summary["total_episodes"], 180)
        self.assertEqual(summary["review"]["total_in_scope"], N_REVIEW)
        self.assertEqual(summary["review"]["reviewed"], N_REVIEW)
        self.assertEqual(summary["review"]["pending"], 0)
        self.assertTrue(summary["consistency"]["reconciliation_sum_equals_in_scope"])
        self.assertEqual(summary["consistency"]["reconciliation_sum"], N_REVIEW)
        self.assertTrue(summary["consistency"]["totals_180"])
        self.assertTrue(summary["consistency"]["buckets_plus_pending_equals_total"])
        self.assertEqual(summary["consistency"]["verified_plus_unresolved_plus_truth_review"],
                         180)
        self.assertEqual(set(summary["per_origin"]), set(ORIGINS))
        self.assertEqual(len(summary["per_camera"]), len(CAMERAS))
        for camera in summary["per_camera"].values():
            self.assertLessEqual(sum(camera[d.lower()] for d in DECISIONS),
                                 camera["original_required"])
        self.assertEqual(sum(camera["training_eligible"]
                             for camera in summary["per_camera"].values()), N_VERIFIED)
        for reason, value in summary["boundaries"].items():
            self.assertIs(value, False, reason)

    def test_build_with_zero_reviewed_is_a_valid_completion_state(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        overlay, manifest_rows = build_overlay_rows(data, state)
        summary = build_summary(data, state, overlay, manifest_rows, verify_preflight(data))
        self.assertEqual(summary["review"]["reviewed"], 0)
        self.assertEqual(summary["review"]["pending"], N_REVIEW)
        self.assertEqual(sum(summary["reconciliation"].values()), 0)
        # nothing is silently promoted to REQUIRED_LITTER while it is still undecided
        self.assertEqual(summary["pending_reconciliation_count"], N_REVIEW)
        self.assertEqual(summary["effective_truth_totals"]["REQUIRED_LITTER"], 120)
        self.assertEqual(sum(summary["effective_truth_totals"].values()), 120)
        self.assertTrue(summary["consistency"]["buckets_plus_pending_equals_total"])
        self.assertFalse(summary["consistency"]["reconciliation_sum_equals_in_scope"])
        self.assertTrue(all(r["effective_truth_class"] is None
                            for r in overlay if r["in_scope"]))
        self.assertEqual(len(manifest_rows), 180)


# --------------------------------------------------------------------------- #
# §23 training eligibility (§13)
# --------------------------------------------------------------------------- #


class EligibilityTest(unittest.TestCase):
    def test_keep_required_with_recovered_source_and_verified_bbox_is_eligible(self) -> None:
        episode = ctx(in_scope=True, localization_status="VERIFIED_BBOX",
                      source_recovery_status="RECOVERED_SOURCE_NATIVE")
        verdict = eligibility(episode, decision_row("KEEP_REQUIRED"))
        self.assertEqual(verdict["effective_truth_class"], "REQUIRED_LITTER")
        self.assertTrue(verdict["truth_ok"] and verdict["source_ok"]
                        and verdict["localization_ok"])
        self.assertTrue(verdict["training_eligible"])
        self.assertIsNone(verdict["training_exclusion_reason"])

    def test_keep_required_with_unresolved_localization_is_not_eligible(self) -> None:
        episode = ctx(in_scope=True, localization_status="LOCALIZATION_UNRESOLVED")
        verdict = eligibility(episode, decision_row("KEEP_REQUIRED"))
        self.assertEqual(verdict["effective_truth_class"], "REQUIRED_LITTER")
        self.assertFalse(verdict["training_eligible"])
        self.assertEqual(verdict["training_exclusion_reason"],
                         "localization_status_LOCALIZATION_UNRESOLVED")

    def test_keep_required_without_a_recovered_source_is_not_eligible(self) -> None:
        episode = ctx(in_scope=True, localization_status="VERIFIED_BBOX",
                      source_recovery_status="SOURCE_EXPIRED")
        verdict = eligibility(episode, decision_row("KEEP_REQUIRED"))
        self.assertFalse(verdict["training_eligible"])
        self.assertEqual(verdict["training_exclusion_reason"],
                         "source_recovery_status_SOURCE_EXPIRED")

    def test_every_non_keep_decision_is_ineligible_even_with_a_verified_bbox(self) -> None:
        for decision in ("IGNORE_SMALL", "NON_LITTER", "UNCERTAIN",
                         "IDENTITY_AMBIGUOUS"):
            episode = ctx(in_scope=True, localization_status="VERIFIED_BBOX")
            verdict = eligibility(episode, decision_row(decision))
            self.assertFalse(verdict["training_eligible"], decision)
            self.assertFalse(verdict["truth_ok"], decision)
            self.assertNotEqual(verdict["effective_truth_class"], "REQUIRED_LITTER")

    def test_identity_ambiguous_has_no_truth_class_at_all(self) -> None:
        verdict = eligibility(ctx(in_scope=True, localization_status="VERIFIED_BBOX"),
                              decision_row("IDENTITY_AMBIGUOUS"))
        self.assertIsNone(verdict["effective_truth_class"])
        self.assertEqual(verdict["effective_truth_bucket"], "IDENTITY_AMBIGUOUS")
        self.assertEqual(verdict["training_exclusion_reason"], "IDENTITY_AMBIGUOUS")

    def test_pending_reconciliation_is_not_eligible(self) -> None:
        verdict = eligibility(ctx(in_scope=True, localization_status="VERIFIED_BBOX"), None)
        self.assertIsNone(verdict["effective_truth_class"])
        self.assertEqual(verdict["effective_truth_bucket"], "PENDING_RECONCILIATION")
        self.assertFalse(verdict["training_eligible"])
        self.assertEqual(verdict["training_exclusion_reason"],
                         "truth_reconciliation_pending")

    def test_out_of_scope_truth_cannot_be_flipped_by_a_decision(self) -> None:
        episode = ctx(in_scope=False, localization_status="VERIFIED_BBOX")
        verdict = eligibility(episode, decision_row("NON_LITTER"))
        self.assertEqual(verdict["effective_truth_class"], "REQUIRED_LITTER")
        self.assertTrue(verdict["training_eligible"])

    def test_fixture_eligibility_excludes_the_decided_60(self) -> None:
        f = Fixture(Path(tempfile.mkdtemp(prefix="step1c1r-elig-")),
                    source_recovery_overrides={"ge-verified-000": "SOURCE_EXPIRED"})
        self.addCleanup(shutil.rmtree, f.root, True)
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        decide_all(data, state)
        _, manifest_rows = build_overlay_rows(data, state)
        by_id = {r["episode_id"]: r for r in manifest_rows}
        eligible = [r["episode_id"] for r in manifest_rows if r["training_eligible"]]
        self.assertEqual(sorted(eligible), sorted(f.verified_ids[1:]))
        self.assertEqual(len(eligible), N_VERIFIED - 1)
        self.assertFalse(by_id[f.unresolved_ids[0]]["training_eligible"])
        self.assertFalse(by_id[f.in_scope_ids[0]]["training_eligible"])
        self.assertEqual(by_id[f.in_scope_ids[0]]["training_exclusion_reason"],
                         "localization_status_TRUTH_REVIEW_REQUIRED")
        self.assertEqual(by_id[f.unresolved_ids[0]]["training_exclusion_reason"],
                         "localization_status_LOCALIZATION_UNRESOLVED")
        summary = build_summary(data, state, *build_overlay_rows(data, state),
                                preflight=verify_preflight(data))
        self.assertEqual(summary["training"]["training_eligible_episode_count"],
                         N_VERIFIED - 1)
        self.assertIn(f.verified_ids[0],
                      summary["training"]["training_excluded_localization_reason"])


# --------------------------------------------------------------------------- #
# §23 resume
# --------------------------------------------------------------------------- #


class ResumeTest(Base):
    def test_restart_keeps_decisions_and_a_further_edit_works(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        decide_all(data, state)
        first = state.get(data.in_scope[0].episode_id)
        self.assertEqual(len(state.decisions), N_REVIEW)
        self.assertEqual(len(state.audit_trail), N_REVIEW)

        restarted = ReviewState.load(f.state, data=data)
        self.assertEqual(restarted.decisions, state.decisions)
        self.assertEqual(len(restarted.audit_trail), N_REVIEW)
        self.assertEqual(restarted.get(data.in_scope[0].episode_id), first)
        self.assertEqual(restarted.progress(data),
                         {"total": N_REVIEW, "reviewed": N_REVIEW, "pending": 0})

        changed = restarted.decide(data.in_scope[0], "NON_LITTER")
        self.assertEqual(changed["revision"], first["revision"] + 1)
        self.assertEqual(changed["reconciliation_decision"], "NON_LITTER")
        self.assertEqual(len(restarted.audit_trail), N_REVIEW + 1)

        again = ReviewState.load(f.state, data=data)
        self.assertEqual(again.progress(data)["reviewed"], N_REVIEW)
        self.assertEqual(again.get(data.in_scope[0].episode_id)["reconciliation_decision"],
                         "NON_LITTER")
        self.assertEqual(len(again.audit_trail), N_REVIEW + 1)
        actions = [entry["action"] for entry in again.audit_trail]
        self.assertEqual(actions.count("reset"), 0)
        self.assertEqual(actions.count("reconcile:KEEP_REQUIRED"), N_REVIEW // 5)
        self.assertEqual(actions.count("reconcile:NON_LITTER"), N_REVIEW // 5 + 1)
        first_episode = [entry for entry in again.audit_trail
                         if entry["episode_id"] == data.in_scope[0].episode_id]
        self.assertEqual(len(first_episode), 2)
        self.assertEqual([entry["action"] for entry in first_episode],
                         ["reconcile:KEEP_REQUIRED", "reconcile:NON_LITTER"])

    def test_state_from_a_different_localization_artifact_is_refused(self) -> None:
        f = self.fixture()
        data = f.load()
        state = ReviewState.load(f.state, data=data)
        state.decide(data.in_scope[0], "KEEP_REQUIRED")
        other = Fixture(self.tmpdir())
        other_data = other.load()
        with self.assertRaises(ReviewError):
            ReviewState.load(f.state, data=other_data)
        # same artifact still loads
        self.assertEqual(ReviewState.load(f.state, data=data).progress(data)["reviewed"], 1)


# --------------------------------------------------------------------------- #
# §23 safety: no localization / proposal / detection / training surface
# --------------------------------------------------------------------------- #


class SafetyTest(unittest.TestCase):
    def test_logic_module_has_no_localization_or_detection_dependency(self) -> None:
        source = (ROOT / "rtsp_annotator"
                  / "ground_litter_truth_reconciliation.py").read_text(encoding="utf-8")
        for forbidden in ("import cv2", "ClassicProposalEngine", "BOX_OK", "BOX_BAD",
                          "ground_litter_localization_proposal", "propose(",
                          "torch", "ultralytics", "numpy", "train_tiles",
                          "build_ground_litter_tiles", "cv2"):
            self.assertNotIn(forbidden, source, forbidden)
        for line in source.splitlines():
            self.assertFalse(line.strip().startswith(("import cv2", "from cv2")), line)

    def test_cli_exposes_exactly_the_four_commands(self) -> None:
        source = CLI.read_text(encoding="utf-8")
        self.assertNotIn("/api/proposal", source)
        self.assertEqual(source.count("sub.add_parser("), 4)
        for command in ("plan", "serve", "status", "build"):
            self.assertIn(f'sub.add_parser("{command}")', source)

    def test_recorded_upstream_commits_all_resolve(self) -> None:
        """No placeholder may survive into the immutable provenance record."""
        if not (ROOT / ".git").exists():
            self.skipTest("not a git checkout")
        spec = importlib.util.spec_from_file_location("reconcile_cli_under_test", CLI)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        for name in ("STEP1C0_EXECUTION", "STEP1C0_MANIFEST", "STEP1C0_EVIDENCE",
                     "STEP1C1_EXECUTION", "STEP1C1_MANIFEST", "STEP1C1_EVIDENCE"):
            sha = getattr(module, name)
            self.assertGreaterEqual(len(sha), 7, name)
            self.assertTrue(all(c in "0123456789abcdef" for c in sha), name)
            result = subprocess.run(["git", "cat-file", "-t", sha], cwd=str(ROOT),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0,
                             f"{name}={sha} does not resolve: {result.stderr.strip()}")
            self.assertEqual(result.stdout.strip(), "commit", name)

    def test_review_ui_offers_only_the_five_truth_decisions(self) -> None:
        html = (TOOLS / "index.html").read_text(encoding="utf-8")
        app = (TOOLS / "app.js").read_text(encoding="utf-8")
        server = (TOOLS / "serve.py").read_text(encoding="utf-8")
        for decision in DECISIONS:
            self.assertIn(decision, html)
            self.assertIn(decision, app)
        for forbidden in ("BOX_OK", "BOX_BAD", "/api/proposal", "proposal",
                          "KEEP_REQUIRED_BBOX", "POINT_OK"):
            self.assertNotIn(forbidden, server, forbidden)
            self.assertNotIn(forbidden, html, forbidden)
            self.assertNotIn(forbidden, app, forbidden)
        # the review server routes are exactly meta / episode / frame (GET) and
        # decision / reset (POST)
        for route in ('path == "/api/meta"', 'path == "/api/episode"',
                      'path == "/api/frame"', 'parsed.path == "/api/decision"',
                      'parsed.path == "/api/reset"'):
            self.assertIn(route, server)
        self.assertEqual(server.count('parsed.path == "/api/'), 2)
        self.assertIn("NO_PRIOR_REASON", app)
        self.assertIn("prior", app)
        self.assertIn("IDENTITY_AMBIGUOUS", app)

    def test_app_js_references_only_ids_that_exist_in_the_page(self) -> None:
        """A typo in a getElementById id would only surface in a browser at runtime."""
        app = (TOOLS / "app.js").read_text(encoding="utf-8")
        html = (TOOLS / "index.html").read_text(encoding="utf-8")
        used = set(re.findall(r'\$\("([^"]+)"\)', app))
        used |= set(re.findall(r'paintView\("([^"]+)"', app))
        in_page = set(re.findall(r'id="([^"]+)"', html))
        rendered_by_app = set(re.findall(r'id="([^"]+)"', app))
        self.assertTrue(used)
        self.assertEqual(used - in_page - rendered_by_app, set())
        self.assertLessEqual({"f-status", "f-camera", "f-origin", "f-search", "queue",
                              "progress", "status-line", "meta", "detail"}, used)
        self.assertLessEqual({"view-full", "view-zoom", "reason"}, used)
        actions = set(re.findall(r'data-act="([^"]+)"', app))
        self.assertEqual(actions, set(DECISIONS) | {"SKIP", "RESET"})
        self.assertEqual(set(re.findall(r'data-act="\$\{[^}]*\}"', app)), set())

    def test_upstream_provenance_records_both_commit_pairs(self) -> None:
        provenance = build_upstream_provenance(
            step1c0_execution_commit="a" * 40, step1c0_manifest_commit="b" * 40,
            step1c0_evidence_commit="c", step1c1_execution_commit="d" * 40,
            step1c1_manifest_commit="e" * 40, step1c1_evidence_commit="f",
            code_equivalence_verified=True)
        self.assertEqual(provenance["step1c1_reported_execution_commit"], "d" * 40)
        self.assertEqual(provenance["step1c1_manifest_commit"], "e" * 40)
        self.assertEqual(provenance["step1c0_manifest_commit"], "b" * 40)
        self.assertEqual(provenance["step1c0_reported_execution_commit"], "a" * 40)
        self.assertTrue(provenance["step1c1_code_equivalence_verified"])
        self.assertTrue(provenance["step1c0_code_equivalence_verified"])
        self.assertTrue(provenance["mismatch_explained"])


# --------------------------------------------------------------------------- #
# the local review server (stdlib http.server, no sockets opened)
# --------------------------------------------------------------------------- #


def load_serve_module():
    spec = importlib.util.spec_from_file_location("ground_litter_truth_ui_serve",
                                                 TOOLS / "serve.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ServeTest(Base):
    def test_meta_payload_reports_progress_and_frozen_counts(self) -> None:
        import argparse
        f = self.fixture()
        data = f.load()
        ui = load_serve_module()
        ui.configure(argparse.Namespace(recovery_root=str(f.recovery), state=str(f.state)),
                     data)
        handler = ui.ReviewHandler.__new__(ui.ReviewHandler)
        payload = handler.meta_payload()
        self.assertEqual(payload["in_scope_count"], N_REVIEW)
        self.assertEqual(payload["frozen"], {"verified_bbox": N_VERIFIED,
                                             "localization_unresolved": N_UNRESOLVED,
                                             "needs_relocalization": 0})
        self.assertEqual(len(payload["queue"]), N_REVIEW)
        self.assertEqual(payload["progress"],
                         {"total": N_REVIEW, "reviewed": 0, "pending": N_REVIEW})
        state = ReviewState.load(f.state, data=data)
        state.decide(data.in_scope[0], "NON_LITTER")
        payload = handler.meta_payload()
        self.assertEqual(payload["progress"]["reviewed"], 1)
        by_id = {row["episode_id"]: row for row in payload["queue"]}
        self.assertEqual(by_id[data.in_scope[0].episode_id]["reconciliation_decision"],
                         "NON_LITTER")
        self.assertTrue(by_id[data.in_scope[0].episode_id]["reviewed"])

    def test_point_only_episode_is_mapped_into_source_pixels(self) -> None:
        """A POINT episode must never be drawn from crop-relative coordinates."""
        import argparse
        f = self.fixture()
        data = f.load()
        ui = load_serve_module()
        ui.configure(argparse.Namespace(recovery_root=str(f.recovery), state=str(f.state)),
                     data)
        handler = ui.ReviewHandler.__new__(ui.ReviewHandler)
        state = ReviewState.load(f.state, data=data)
        point_ids = [e.episode_id for e in data.in_scope
                     if e.origin == "manual_missing_target"]
        self.assertTrue(point_ids)
        for episode_id in point_ids:
            payload = handler._episode_payload(episode_id, state)
            self.assertIsNone(payload["original_bbox"])
            self.assertEqual(payload["location_type"], "POINT")
            self.assertEqual(payload["original_point_frame"], [1500.0, 900.0])
            self.assertEqual(payload["original_crop_box"],
                             [1444.0, 844.0, 1556.0, 956.0])
        bbox_episode = next(e for e in data.in_scope if e.original_bbox)
        payload = handler._episode_payload(bbox_episode.episode_id, state)
        self.assertEqual(payload["location_type"], "BBOX")
        self.assertIsNone(payload["original_point_frame"])
        self.assertIsNone(payload["original_crop_box"])

    def test_an_unmappable_point_is_not_fabricated(self) -> None:
        import argparse
        import dataclasses
        f = self.fixture()
        data = f.load()
        ui = load_serve_module()
        ui.configure(argparse.Namespace(recovery_root=str(f.recovery), state=str(f.state)),
                     data)
        broken = dataclasses.replace(
            data.in_scope[0], original_bbox=None,
            original_point={"x_norm": 0.5, "y_norm": 0.5},
            original_point_source={"ok": False, "reason": "derived_crop_size_mismatch"})
        handler = ui.ReviewHandler.__new__(ui.ReviewHandler)
        handler.data = type("_Data", (), {"by_id": {broken.episode_id: broken}})()
        payload = handler._episode_payload(broken.episode_id, ReviewState.load(f.state))
        self.assertIsNone(payload["original_point_frame"])
        self.assertEqual(payload["location_type"], "POINT")
        self.assertEqual(payload["original_point_source"]["ok"], False)

    def test_queue_is_only_the_60_and_payload_has_no_proposal(self) -> None:
        import argparse
        f = self.fixture()
        data = f.load()
        ui = load_serve_module()
        ui.configure(argparse.Namespace(recovery_root=str(f.recovery), state=str(f.state)),
                     data)
        self.assertEqual(len(ui.ReviewHandler.queue), N_REVIEW)
        self.assertEqual({row["episode_id"] for row in ui.ReviewHandler.queue},
                         set(f.in_scope_ids))
        for row in ui.ReviewHandler.queue:
            self.assertTrue(row["prior_truth_review_reason"])
            self.assertNotIn("proposal", row)

        state = ReviewState.load(f.state, data=data)
        handler = ui.ReviewHandler.__new__(ui.ReviewHandler)
        payload = handler._episode_payload(f.in_scope_ids[0], state)
        self.assertEqual(set(payload["options"]), set(DECISIONS))
        self.assertEqual(payload["option_truth_class"]["IDENTITY_AMBIGUOUS"], None)
        self.assertEqual(payload["prior_truth_review_reason"], "垃圾太小")
        self.assertTrue(payload["frame_url"].startswith("/api/frame?episode="))
        self.assertIsNone(payload["reconciled_truth_class"])
        for key in payload:
            self.assertNotIn("proposal", key)
        self.assertEqual(handler._episode_payload(f.verified_ids[0], state), {})
        self.assertEqual(handler._episode_payload(f.unresolved_ids[0], state), {})
        self.assertEqual(handler._episode_payload("does-not-exist", state), {})

    def test_frame_serving_stays_inside_the_recovery_root(self) -> None:
        import argparse
        f = self.fixture(outside_frame_for="ge-review-059")
        data = f.load()
        ui = load_serve_module()
        ui.configure(argparse.Namespace(recovery_root=str(f.recovery), state=str(f.state)),
                     data)
        handler = ui.ReviewHandler.__new__(ui.ReviewHandler)
        inside = handler._frame_path("ge-review-000")
        self.assertIsNotNone(inside)
        self.assertTrue(str(inside).startswith(str(f.recovery.resolve())))
        self.assertIsNone(handler._frame_path("ge-review-059"))
        self.assertIsNone(handler._frame_path("ge-verified-000"))
        self.assertIsNone(handler._frame_path("nope"))

    def test_review_state_is_created_next_to_the_requested_path(self) -> None:
        import argparse
        f = self.fixture()
        data = f.load()
        ui = load_serve_module()
        nested = f.root / "nested" / "deep" / "review_state.json"
        ui.configure(argparse.Namespace(recovery_root=str(f.recovery), state=str(nested)),
                     data)
        self.assertTrue(nested.is_file())
        self.assertEqual(json.loads(nested.read_text(encoding="utf-8"))[
            "review_schema_version"], REVIEW_SCHEMA_VERSION)


# --------------------------------------------------------------------------- #
# CLI end to end: plan -> status -> build
# --------------------------------------------------------------------------- #


class CliTest(Base):
    def run_cli(self, *argv) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(CLI), *argv],
                              cwd=str(ROOT), capture_output=True, text=True)

    def base_args(self, f: Fixture) -> list[str]:
        return ["--gold", str(f.gold), "--gold-manifest", str(f.gold_manifest),
                "--recovery-root", str(f.recovery),
                "--localization-root", str(f.localization),
                "--output", str(f.out), "--repo-root", str(ROOT)]

    def test_plan_status_build_with_zero_reviewed(self) -> None:
        f = self.fixture()
        plan = self.run_cli(*self.base_args(f), "plan")
        self.assertEqual(plan.returncode, 0, plan.stderr)
        report = json.loads(plan.stdout)
        self.assertEqual(report["preflight"]["truth_review_required"], N_REVIEW)
        self.assertEqual(report["in_scope_count"], N_REVIEW)
        self.assertEqual(report["prior_reason_present"], N_REVIEW)
        self.assertEqual(report["prior_reason_missing"], 0)
        self.assertEqual(sum(report["prior_reason_histogram"].values()), N_REVIEW)
        self.assertEqual(report["options"], list(DECISIONS))
        self.assertTrue((f.out / "plan.json").is_file())

        status = self.run_cli(*self.base_args(f), "status")
        self.assertEqual(status.returncode, 0, status.stderr)
        summary = json.loads(status.stdout)
        self.assertEqual(summary["total"], N_REVIEW)
        self.assertEqual(summary["reviewed"], 0)
        self.assertEqual(summary["training_eligible_episode_count"], N_VERIFIED)

        build = self.run_cli(*self.base_args(f), "build")
        self.assertEqual(build.returncode, 0, build.stderr)
        result = json.loads(build.stdout)
        self.assertEqual(result["reviewed"], 0)
        self.assertEqual(result["pending"], N_REVIEW)
        for name in ("truth_reconciliation.jsonl", "training_episode_manifest.jsonl",
                     "SUMMARY.json", "MANIFEST.json"):
            self.assertTrue((f.out / name).is_file(), name)
        overlay = [json.loads(line) for line in
                   (f.out / "truth_reconciliation.jsonl").read_text(
                       encoding="utf-8").splitlines()]
        self.assertEqual(len(overlay), 180)
        manifest_json = json.loads((f.out / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest_json["counts"]["in_scope"], N_REVIEW)
        self.assertEqual(manifest_json["counts"]["reviewed"], 0)
        self.assertEqual(manifest_json["localization_input_sha256"], f.load().localization_sha256)
        self.assertTrue(manifest_json["upstream_provenance"]
                        ["step1c1_code_equivalence_verified"])
        self.assertEqual(len(manifest_json["code_commit"]), 40)

    def test_serve_subcommand_is_available(self) -> None:
        f = self.fixture()
        source = CLI.read_text(encoding="utf-8")
        self.assertIn('sub.add_parser("serve")', source)
        self.assertIn("ui.configure(args, data)", source)


if __name__ == "__main__":
    unittest.main()
