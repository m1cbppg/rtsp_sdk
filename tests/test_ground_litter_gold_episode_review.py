"""Step 1B: gold-episode review workflow + trainability audit tests.

Pure-stdlib, no cv2/model dependency: the review logic must stay testable in the
repository virtualenv.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from rtsp_annotator.ground_litter_gold_episode_review import (
    REVIEW_SCHEMA_VERSION,
    NearDuplicateTarget,
    ReviewError,
    ReviewState,
    Step1AInput,
    TrainabilityPolicy,
    audit_trainability,
    build_gold_records,
    build_queue,
    build_summary,
    classify_risk,
    episode_id_for,
    jpeg_dimensions,
    load_step1a,
    merge_suggestions,
    resolve_episodes,
    review_fingerprint,
    sha256_file,
    validate_episode_partition,
)

BASE_TIME = "2026-09-20 10:00:00"


def make_card(
    card_id: str,
    *,
    camera: str = "01021",
    batch: str = "batchx",
    timestamp: str | None = BASE_TIME,
    bbox=(100.0, 100.0, 140.0, 140.0),
    label: str = "LITTER",
    file_id: str | None = "ps-1",
    frame_id: str | None = "f00s00",
    assets: dict | None = None,
) -> dict:
    return {
        "card_id": card_id,
        "raw_review_id": card_id.split(":", 1)[-1],
        "batch_key": batch,
        "label": label,
        "timestamp": timestamp,
        "frame_id": frame_id,
        "source_file_id": file_id,
        "proposal_source": "semantic_tile",   # must never be rendered as truth
        "bbox": list(bbox) if bbox else None,
        "assets": assets if assets is not None else {
            "context_image": "assets/c-context.jpg",
            "crop_image": "assets/c-crop.jpg",
            "before_image": "assets/c-before.jpg",
            "after_image": "assets/c-after.jpg",
        },
    }


def make_candidate(
    candidate_id: str,
    cards: list[dict],
    *,
    camera: str = "01021",
    scene_version: str | None = None,
    ambiguity: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    labels: dict | None = None,
) -> dict:
    member_ids = [c["card_id"] for c in cards]
    label_summary = labels or {
        l: sum(1 for c in cards if c["label"] == l) for l in {c["label"] for c in cards}
    }
    return {
        "episode_candidate_id": candidate_id,
        "camera_id": camera,
        "scene_version": scene_version,
        "start_timestamp": start or BASE_TIME,
        "end_timestamp": end or BASE_TIME,
        "member_review_card_ids": member_ids,
        "member_count": len(member_ids),
        "source_file_ids": sorted({c["source_file_id"] for c in cards if c["source_file_id"]}),
        "representative_card_id": member_ids[0],
        "representative_frame": "assets/c-context.jpg",
        "representative_bbox": cards[0]["bbox"],
        "candidate_ambiguous": bool(ambiguity),
        "ambiguity_reasons": list(ambiguity or []),
        "lineage": {"review_cards": cards},
        "original_labels_summary": label_summary,
        "candidate_grouping_evidence": {
            "grouping_method": "time_space_scale_complete_linkage_v1",
            "time_span_seconds": 0.0,
            "bbox_center_spread_px": 0.0,
            "bbox_diagonal_spread_px": 0.0,
        },
    }


def make_step1a(candidates: list[dict], batch_dirs: dict | None = None) -> Step1AInput:
    return Step1AInput(
        artifact_path=Path("/nonexistent/episode_candidates.jsonl"),
        artifact_sha256="0" * 64,
        manifest_path=None,
        step1a_code_commit="a" * 40,
        batch_dirs=batch_dirs or {},
        candidates=tuple(candidates),
        canvas_by_camera={},
    )


class TempMixin(unittest.TestCase):
    def tmpdir(self) -> Path:
        path = Path(tempfile.mkdtemp(prefix="step1b-test-"))
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def make_state(self, step1a: Step1AInput, name: str = "review_state.json") -> ReviewState:
        return ReviewState.load(self.tmpdir() / name, step1a=step1a)


# --------------------------------------------------------------------------- #
# review state persistence + resume
# --------------------------------------------------------------------------- #


class ReviewStateTest(TempMixin):
    def setUp(self) -> None:
        card = make_card("batchx:a")
        self.step1a = make_step1a([make_candidate("ec-1", [card])])

    def _round_trip(self, decision: str, expected_truth: str | None = None) -> ReviewState:
        state = self.make_state(self.step1a)
        state.decide(self.step1a, "ec-1", decision, note=f"note-{decision}")
        reloaded = ReviewState.load(state.path, step1a=self.step1a)
        record = reloaded.get("ec-1")
        self.assertEqual(record["status"], "reviewed")
        self.assertEqual(record["decision"], decision)
        # The decision is not the truth class: CONFIRM maps to REQUIRED_LITTER.
        self.assertEqual(record["truth_class"], expected_truth or decision)
        self.assertEqual(record["note"], f"note-{decision}")
        return reloaded

    def test_confirm_saves_and_restores(self) -> None:
        state = self._round_trip("CONFIRM", "REQUIRED_LITTER")
        self.assertEqual(state.get("ec-1")["episodes"][0]["truth_class"], "REQUIRED_LITTER")

    def test_non_litter_saves_and_restores(self) -> None:
        self._round_trip("NON_LITTER")

    def test_ignore_small_saves_and_restores(self) -> None:
        self._round_trip("IGNORE_SMALL")

    def test_uncertain_saves_and_restores(self) -> None:
        self._round_trip("UNCERTAIN")

    def test_progress_and_skip_and_reset(self) -> None:
        state = self.make_state(self.step1a)
        self.assertEqual(state.progress(self.step1a)["pending"], 1)
        state.skip("ec-1", note="later")
        self.assertEqual(state.progress(self.step1a)["skipped"], 1)
        state.decide(self.step1a, "ec-1", "CONFIRM")
        self.assertEqual(state.progress(self.step1a)["reviewed"], 1)
        state.reset("ec-1")
        self.assertEqual(state.progress(self.step1a)["pending"], 1)

    def test_resume_after_reload_is_identical(self) -> None:
        state = self.make_state(self.step1a)
        state.decide(self.step1a, "ec-1", "CONFIRM", note="x")
        first = json.loads(state.path.read_text())
        again = ReviewState.load(state.path, step1a=self.step1a)
        self.assertEqual(again.candidates, state.candidates)
        self.assertEqual(again.audit_trail, state.audit_trail)
        again.save()
        self.assertEqual(json.loads(state.path.read_text())["candidates"], first["candidates"])

    def test_audit_trail_records_every_change(self) -> None:
        state = self.make_state(self.step1a)
        state.decide(self.step1a, "ec-1", "CONFIRM")
        state.decide(self.step1a, "ec-1", "NON_LITTER")
        state.reset("ec-1")
        actions = [row["action"] for row in state.audit_trail]
        self.assertEqual(actions,
                         ["decide:CONFIRM", "decide:NON_LITTER", "reset"])
        self.assertEqual(state.get("ec-1"), None)

    def test_rejects_state_from_other_artifact(self) -> None:
        state = self.make_state(self.step1a)
        state.input_sha256 = "f" * 64
        state.save()
        with self.assertRaises(ReviewError):
            ReviewState.load(state.path, step1a=self.step1a)


# --------------------------------------------------------------------------- #
# SPLIT
# --------------------------------------------------------------------------- #


class SplitTest(TempMixin):
    def setUp(self) -> None:
        cards = [
            make_card("batchx:a", bbox=(0, 0, 20, 20)),
            make_card("batchx:b", bbox=(500, 500, 520, 520)),
            make_card("batchx:c", bbox=(2, 2, 22, 22)),
        ]
        self.candidate = make_candidate("ec-1", cards)
        self.step1a = make_step1a([self.candidate])

    def test_split_creates_multiple_episodes(self) -> None:
        state = self.make_state(self.step1a)
        state.decide(self.step1a, "ec-1", "SPLIT", episodes=[
            {"draft_id": "e1", "member_card_ids": ["batchx:a", "batchx:c"],
             "truth_class": "REQUIRED_LITTER", "localization_status": "OK"},
            {"draft_id": "e2", "member_card_ids": ["batchx:b"],
             "truth_class": "REQUIRED_LITTER", "localization_status": "NEEDS_RELOCALIZATION"},
        ])
        episodes, conflicts = resolve_episodes(self.step1a, state)
        self.assertEqual(conflicts, [])
        self.assertEqual(len(episodes), 2)
        self.assertEqual({m for e in episodes for m in e["member_card_ids"]},
                         {"batchx:a", "batchx:b", "batchx:c"})
        merged = [e for e in episodes if e["localization_status"] == "NEEDS_RELOCALIZATION"]
        self.assertEqual(len(merged), 1)

    def test_split_can_assign_non_litter_members(self) -> None:
        state = self.make_state(self.step1a)
        state.decide(self.step1a, "ec-1", "SPLIT", episodes=[
            {"draft_id": "e1", "member_card_ids": ["batchx:a", "batchx:c"],
             "truth_class": "REQUIRED_LITTER", "localization_status": "OK"},
            {"draft_id": "e2", "member_card_ids": ["batchx:b"],
             "truth_class": "NON_LITTER", "localization_status": "OK"},
        ])
        episodes, _ = resolve_episodes(self.step1a, state)
        classes = sorted(e["truth_class"] for e in episodes)
        self.assertEqual(classes, ["NON_LITTER", "REQUIRED_LITTER"])

    def test_split_rejects_duplicate_assignment(self) -> None:
        with self.assertRaises(ReviewError):
            validate_episode_partition(self.candidate, [
                {"member_card_ids": ["batchx:a", "batchx:b"]},
                {"member_card_ids": ["batchx:a"], "truth_class": "REQUIRED_LITTER"},
            ])

    def test_split_rejects_silent_member_loss(self) -> None:
        with self.assertRaises(ReviewError) as ctx:
            validate_episode_partition(self.candidate, [
                {"member_card_ids": ["batchx:a"]},
                {"member_card_ids": ["batchx:b"]},
            ])
        self.assertIn("unassigned", str(ctx.exception))

    def test_split_rejects_unknown_member(self) -> None:
        with self.assertRaises(ReviewError):
            validate_episode_partition(self.candidate, [
                {"member_card_ids": ["batchx:a", "batchx:b", "batchx:c"]},
                {"member_card_ids": ["batchx:zzz"]},
            ])

    def test_split_requires_at_least_two_groups(self) -> None:
        with self.assertRaises(ReviewError):
            validate_episode_partition(self.candidate, [
                {"member_card_ids": ["batchx:a", "batchx:b", "batchx:c"]},
            ])

    def test_split_rejects_bad_truth_class(self) -> None:
        with self.assertRaises(ReviewError):
            validate_episode_partition(self.candidate, [
                {"member_card_ids": ["batchx:a"], "truth_class": "LITTER"},
                {"member_card_ids": ["batchx:b", "batchx:c"]},
            ])


# --------------------------------------------------------------------------- #
# MERGE
# --------------------------------------------------------------------------- #


class MergeTest(TempMixin):
    def _setup(self, *, left_camera="01021", right_camera="01021",
               left_scene=None, right_scene=None):
        left = make_candidate("ec-A", [make_card("batchx:a", camera=left_camera)],
                              camera=left_camera, scene_version=left_scene)
        right = make_candidate("ec-B", [make_card("batchx:b", camera=right_camera)],
                               camera=right_camera, scene_version=right_scene)
        return make_step1a([left, right])

    def test_merge_combines_candidates_and_keeps_lineage(self) -> None:
        step1a = self._setup()
        state = self.make_state(step1a)
        state.decide(step1a, "ec-A", "MERGE", merge_targets=["ec-B"])
        episodes, conflicts = resolve_episodes(step1a, state)
        self.assertEqual(conflicts, [])
        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertEqual(sorted(episode["source_episode_candidate_ids"]), ["ec-A", "ec-B"])
        self.assertEqual(len(episode["member_card_ids"]), 2)
        self.assertTrue(episode["grouping_review"]["merged"])

    def test_merge_absorbs_unreviewed_target(self) -> None:
        step1a = self._setup()
        state = self.make_state(step1a)
        state.decide(step1a, "ec-A", "MERGE", merge_targets=["ec-B"])
        # ec-B itself was never decided, yet its members must join the episode.
        self.assertIsNone(state.get("ec-B"))
        episodes, _ = resolve_episodes(step1a, state)
        self.assertEqual(set(episodes[0]["member_card_ids"]), {"batchx:a", "batchx:b"})

    def test_merge_rejects_cross_camera(self) -> None:
        step1a = self._setup(right_camera="01022")
        state = self.make_state(step1a)
        with self.assertRaises(ReviewError) as ctx:
            state.decide(step1a, "ec-A", "MERGE", merge_targets=["ec-B"])
        self.assertIn("across cameras", str(ctx.exception))

    def test_merge_rejects_explicit_scene_version_mismatch(self) -> None:
        step1a = self._setup(left_scene="v1", right_scene="v2")
        state = self.make_state(step1a)
        with self.assertRaises(ReviewError) as ctx:
            state.decide(step1a, "ec-A", "MERGE", merge_targets=["ec-B"])
        self.assertIn("scene_version", str(ctx.exception))

    def test_merge_allows_unknown_historical_scene(self) -> None:
        step1a = self._setup(left_scene=None, right_scene=None)
        state = self.make_state(step1a)
        state.decide(step1a, "ec-A", "MERGE", merge_targets=["ec-B"])
        episodes, conflicts = resolve_episodes(step1a, state)
        self.assertEqual(conflicts, [])
        self.assertEqual(len(episodes), 1)

    def test_merge_rejects_already_decided_non_litter_target(self) -> None:
        step1a = self._setup()
        state = self.make_state(step1a)
        state.decide(step1a, "ec-B", "NON_LITTER")
        with self.assertRaises(ReviewError):
            state.decide(step1a, "ec-A", "MERGE", merge_targets=["ec-B"])

    def test_merge_conflict_is_reported_not_silently_dropped(self) -> None:
        step1a = self._setup()
        state = self.make_state(step1a)
        state.decide(step1a, "ec-A", "MERGE", merge_targets=["ec-B"])
        # the target is later re-classified, creating a conflict at build time
        state.candidates["ec-B"] = {"status": "reviewed", "decision": "UNCERTAIN",
                                    "truth_class": "UNCERTAIN", "episodes": [],
                                    "merge_targets": [], "note": ""}
        episodes, conflicts = resolve_episodes(step1a, state)
        self.assertEqual(episodes, [])
        self.assertTrue(any(c["type"] == "merge_component_contains_non_episode_decision"
                            for c in conflicts))

    def test_merge_suggestions_are_same_camera_and_bounded(self) -> None:
        cards = [make_card(f"batchx:s{i}", bbox=(100 + i * 5, 100, 140 + i * 5, 140),
                           timestamp=f"2026-09-20 10:0{i}:00") for i in range(8)]
        candidates = [make_candidate("ec-main", [cards[0]])]
        for i, card in enumerate(cards[1:], 1):
            candidates.append(make_candidate(f"ec-{i}", [card]))
        candidates.append(make_candidate("ec-other", [make_card("batchx:z", camera="01022")],
                                         camera="01022"))
        step1a = make_step1a(candidates)
        suggestions = merge_suggestions(step1a, "ec-main")
        self.assertLessEqual(len(suggestions), 5)
        self.assertNotIn("ec-other", [s["candidate_id"] for s in suggestions])


# --------------------------------------------------------------------------- #
# deterministic episode ids
# --------------------------------------------------------------------------- #


class EpisodeIdTest(unittest.TestCase):
    def test_episode_id_is_deterministic_and_order_independent(self) -> None:
        self.assertEqual(episode_id_for("01021", ["a", "b"]),
                         episode_id_for("01021", ["b", "a"]))

    def test_episode_id_differs_for_different_member_sets(self) -> None:
        self.assertNotEqual(episode_id_for("01021", ["a", "b"]),
                            episode_id_for("01021", ["a", "c"]))
        self.assertNotEqual(episode_id_for("01021", ["a"]),
                            episode_id_for("01022", ["a"]))

    def test_episode_id_is_not_the_step1a_candidate_id(self) -> None:
        episode = episode_id_for("01021", ["a"])
        self.assertTrue(episode.startswith("ge-01021-"))
        self.assertNotIn("ec-", episode)


# --------------------------------------------------------------------------- #
# risk classification + queue
# --------------------------------------------------------------------------- #


class RiskTest(unittest.TestCase):
    def test_missing_file_id_is_lineage_only_not_grouping_risk(self) -> None:
        candidate = make_candidate("ec-1", [make_card("batchx:a", file_id=None)],
                                   ambiguity=["lineage_source_file_id_unavailable"])
        risk = classify_risk(candidate)
        self.assertFalse(risk["grouping_risk"])
        self.assertTrue(risk["lineage_only"])

    def test_borderline_reason_is_grouping_risk(self) -> None:
        candidate = make_candidate("ec-1", [make_card("batchx:a")],
                                   ambiguity=["borderline_within_threshold"])
        risk = classify_risk(candidate)
        self.assertTrue(risk["grouping_risk"])
        self.assertFalse(risk["lineage_only"])

    def test_box_wrong_singleton_is_not_grouping_risk(self) -> None:
        candidate = make_candidate("ec-1", [make_card("batchx:a", label="BOX_WRONG")])
        risk = classify_risk(candidate)
        self.assertFalse(risk["grouping_risk"])

    def test_box_wrong_inside_group_is_grouping_risk(self) -> None:
        candidate = make_candidate("ec-1", [
            make_card("batchx:a", label="BOX_WRONG"),
            make_card("batchx:b", label="LITTER"),
        ])
        risk = classify_risk(candidate)
        self.assertIn("box_wrong_participates_in_grouping", risk["grouping_risk_reasons"])

    def test_queue_priority_order(self) -> None:
        # distinct timestamps and positions so no same-frame neighbour flag fires
        candidates = [
            make_candidate("ec-other", [make_card("batchx:p1", timestamp="2026-09-20 14:00:00",
                                                  bbox=(100, 900, 140, 940))]),
            make_candidate("ec-lineage",
                           [make_card("batchx:p5", timestamp="2026-09-20 15:00:00",
                                      bbox=(1000, 900, 1040, 940), file_id=None)],
                           ambiguity=["lineage_source_file_id_unavailable"]),
            make_candidate("ec-multi",
                           [make_card("batchx:p2a", timestamp="2026-09-20 11:00:00",
                                      bbox=(1000, 100, 1040, 140)),
                            make_card("batchx:p2b", timestamp="2026-09-20 12:00:00",
                                      bbox=(1500, 100, 1540, 140))]),
            make_candidate("ec-box", [make_card("batchx:p3", timestamp="2026-09-20 13:00:00",
                                                bbox=(2000, 100, 2040, 140),
                                                label="BOX_WRONG")]),
            make_candidate("ec-group", [make_card("batchx:p0", timestamp="2026-09-20 10:00:00",
                                                  bbox=(100, 100, 140, 140))],
                           ambiguity=["borderline_within_threshold"]),
        ]
        rows = {r["episode_candidate_id"]: r for r in build_queue(make_step1a(candidates))}
        self.assertEqual(rows["ec-group"]["queue_priority"], 1)
        self.assertEqual(rows["ec-multi"]["queue_priority"], 2)
        self.assertEqual(rows["ec-box"]["queue_priority"], 3)
        self.assertEqual(rows["ec-other"]["queue_priority"], 4)
        self.assertEqual(rows["ec-lineage"]["queue_priority"], 5)
        self.assertFalse(rows["ec-other"]["grouping_risk"])
        self.assertTrue(rows["ec-lineage"]["lineage_only"])

    def test_same_frame_neighbour_requires_merge_plausible_distance(self) -> None:
        near = [make_candidate("ec-n1", [make_card("batchx:n1", bbox=(100, 100, 140, 140))]),
                make_candidate("ec-n2", [make_card("batchx:n2", bbox=(110, 100, 150, 140))])]
        far = [make_candidate("ec-f1", [make_card("batchx:f1", bbox=(100, 100, 140, 140))]),
               make_candidate("ec-f2", [make_card("batchx:f2", bbox=(1200, 900, 1240, 940))])]
        near_rows = {r["episode_candidate_id"]: r for r in build_queue(make_step1a(near))}
        far_rows = {r["episode_candidate_id"]: r for r in build_queue(make_step1a(far))}
        self.assertIn("same_frame_neighbour_candidate",
                      near_rows["ec-n1"]["grouping_risk_reasons"])
        self.assertNotIn("same_frame_neighbour_candidate",
                         far_rows["ec-f1"]["grouping_risk_reasons"])


# --------------------------------------------------------------------------- #
# trainability audit
# --------------------------------------------------------------------------- #


class TrainabilityTest(TempMixin):
    def _step1a_with_assets(self, card: dict) -> Step1AInput:
        root = self.tmpdir()
        batch = root / "batchdir"
        (batch / "assets").mkdir(parents=True, exist_ok=True)
        for slot in ("context_image", "crop_image", "before_image", "after_image"):
            relative = (card.get("assets") or {}).get(slot)
            if not relative:
                continue
            target = batch / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\xff\xd8\xff\xd9")  # minimal file, dims unreadable
        candidate = make_candidate("ec-1", [card])
        return make_step1a([candidate], batch_dirs={"batchx": str(batch)})

    def _audit(self, card: dict, *, evidence=None, policy=None) -> dict:
        step1a = self._step1a_with_assets(card)
        return audit_trainability(
            {"member_card_ids": [card["card_id"]]}, step1a,
            policy=policy or TrainabilityPolicy(reference_time=datetime(2026, 9, 21, 0, 0, 0)),
            source_evidence=evidence or {})

    def test_source_frame_asset_gives_trainable(self) -> None:
        card = make_card("batchx:a", file_id=None)
        path = self.tmpdir() / "frame.jpg"
        path.write_bytes(b"\xff\xd8\xff\xd9")
        audit = self._audit(card, evidence={
            "source_frame_assets": {"batchx:a": {"path": str(path), "width": 2560, "height": 1440}}})
        self.assertEqual(audit["trainability_status"], "TRAINABLE_SOURCE_NATIVE")
        self.assertEqual(audit["lineage_method"], "A_source_resolution_frame_asset")

    def test_unverifiable_source_frame_claim_is_rejected(self) -> None:
        card = make_card("batchx:a", file_id=None)
        audit = self._audit(card, evidence={
            "source_frame_assets": {"batchx:a": {"path": "/does/not/exist.jpg"}}})
        self.assertNotEqual(audit["trainability_status"], "TRAINABLE_SOURCE_NATIVE")

    def test_ps_plus_timestamp_gives_trainable(self) -> None:
        card = make_card("batchx:a", file_id="ps-1", timestamp=BASE_TIME)
        audit = self._audit(card)
        self.assertEqual(audit["trainability_status"], "TRAINABLE_SOURCE_NATIVE")
        self.assertEqual(audit["lineage_method"], "B_ps_file_id_plus_timestamp")
        self.assertEqual(audit["lineage_reason"], "ps_reachability_unverified_requires_refetch")
        member = audit["members"][0]
        self.assertEqual(member["source_file_id"], "ps-1")
        self.assertEqual(member["frame_id"], "f00s00")

    def test_verified_reachable_ps_drops_the_refetch_warning(self) -> None:
        card = make_card("batchx:a", file_id="ps-1")
        audit = self._audit(card, evidence={"verified_reachable_file_ids": ["ps-1"]})
        self.assertTrue(audit["members"][0]["ps_reachability_verified"])
        self.assertEqual(audit["lineage_reason"], "")

    def test_retention_expired_gives_review_only(self) -> None:
        card = make_card("batchx:a", file_id="ps-1", timestamp="2026-09-01 10:00:00")
        audit = self._audit(card, policy=TrainabilityPolicy(
            retention_days=7.0, reference_time=datetime(2026, 9, 21, 0, 0, 0)))
        self.assertEqual(audit["trainability_status"], "REVIEW_ONLY")
        self.assertEqual(audit["lineage_reason"], "recording_beyond_declared_retention_window")

    def test_only_derived_crops_gives_review_only(self) -> None:
        card = make_card("batchx:a", file_id=None, frame_id=None)
        audit = self._audit(card)
        self.assertEqual(audit["trainability_status"], "REVIEW_ONLY")
        self.assertEqual(audit["lineage_reason"],
                         "only_derived_crop_or_context_no_source_route")

    def test_missing_file_id_but_recoverable_gives_lineage_unresolved(self) -> None:
        card = make_card("batchx:a", file_id=None, frame_id="f00s00", timestamp=BASE_TIME)
        audit = self._audit(card)
        self.assertEqual(audit["trainability_status"], "LINEAGE_UNRESOLVED")
        self.assertEqual(audit["lineage_reason"],
                         "source_file_id_missing_may_recover_via_frame_id_timestamp")

    def test_missing_asset_gives_lineage_unresolved(self) -> None:
        card = make_card("batchx:a", file_id=None, frame_id=None, assets={
            "context_image": "assets/missing-context.jpg", "crop_image": None})
        audit = self._audit(card)
        self.assertEqual(audit["trainability_status"], "LINEAGE_UNRESOLVED")

    def test_audit_never_claims_tiles_or_training(self) -> None:
        card = make_card("batchx:a")
        audit = self._audit(card)
        self.assertIn("Audit only", audit["note"])

    def test_jpeg_dimensions_reads_header(self) -> None:
        # minimal well-formed JPEG header chain: SOI + APP0 + SOF0 (1x1) + EOI
        blob = bytes.fromhex(
            "ffd8"                      # SOI
            "ffe00010" "4a46494600010100000100010000"   # APP0 / JFIF, len 16
            "ffc00011" "08" "0001" "0001" "03" "011100" "021100" "031100"  # SOF0 len 17
            "ffd9")
        path = self.tmpdir() / "one.jpg"
        path.write_bytes(blob)
        self.assertEqual(jpeg_dimensions(path), (1, 1))
        self.assertIsNone(jpeg_dimensions(self.tmpdir() / "nope.jpg"))
        not_jpeg = self.tmpdir() / "not.jpg"
        not_jpeg.write_bytes(b"not a jpeg at all")
        self.assertIsNone(jpeg_dimensions(not_jpeg))

    def test_jpeg_dimensions_matches_a_real_review_asset(self) -> None:
        # exercise the parser against genuine Silver assets when they are present
        asset = (Path(__file__).resolve().parents[1]
                 / "output/ground_litter_audit_formal_day1/devices/"
                   "44180209031322001021/review/assets")
        if not asset.is_dir():
            self.skipTest("historical Silver assets not present in this checkout")
        jpegs = sorted(asset.glob("*-context.jpg"))
        if not jpegs:
            self.skipTest("no context assets found")
        dims = jpeg_dimensions(jpegs[0])
        self.assertIsNotNone(dims)
        self.assertGreater(dims[0], 0)
        self.assertGreater(dims[1], 0)


# --------------------------------------------------------------------------- #
# gold output, summary, boundaries
# --------------------------------------------------------------------------- #


class GoldOutputTest(TempMixin):
    def _step1a(self) -> Step1AInput:
        return make_step1a([
            make_candidate("ec-A", [make_card("batchx:a")]),
            make_candidate("ec-B", [make_card("batchx:b", bbox=(600, 600, 640, 640))]),
        ])

    def test_empty_state_produces_zero_gold(self) -> None:
        step1a = self._step1a()
        state = self.make_state(step1a)
        records, conflicts = build_gold_records(step1a, state)
        self.assertEqual(records, [])
        self.assertEqual(conflicts, [])
        summary = build_summary(step1a, state, records, conflicts, queue=build_queue(step1a))
        self.assertEqual(summary["human_review"]["reviewed"], 0)
        self.assertEqual(summary["human_review"]["pending"], 2)
        self.assertEqual(summary["gold_episodes"]["gold_episode_count"], 0)
        self.assertFalse(summary["human_review"]["review_complete"])

    def test_confirm_produces_one_gold_episode_with_scene_unknown(self) -> None:
        step1a = self._step1a()
        state = self.make_state(step1a)
        state.decide(step1a, "ec-A", "CONFIRM")
        records, _ = build_gold_records(step1a, state)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["truth_class"], "REQUIRED_LITTER")
        self.assertEqual(record["scene_version"], "UNKNOWN_HISTORICAL")
        self.assertEqual(record["record_kind"], "episode")
        self.assertTrue(record["is_litter_episode"])
        self.assertEqual(record["source_episode_candidate_ids"], ["ec-A"])
        self.assertIn(record["trainability_status"],
                      ("TRAINABLE_SOURCE_NATIVE", "REVIEW_ONLY", "LINEAGE_UNRESOLVED"))

    def test_non_litter_candidate_becomes_classification_outcome(self) -> None:
        step1a = self._step1a()
        state = self.make_state(step1a)
        state.decide(step1a, "ec-A", "NON_LITTER")
        records, _ = build_gold_records(step1a, state)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["record_kind"], "classification_outcome")
        self.assertFalse(records[0]["is_litter_episode"])
        self.assertEqual(records[0]["trainability_status"], "REVIEW_ONLY")

    def test_summary_taxonomy_and_boundaries(self) -> None:
        step1a = self._step1a()
        state = self.make_state(step1a)
        state.decide(step1a, "ec-A", "CONFIRM")
        state.decide(step1a, "ec-B", "IGNORE_SMALL")
        records, conflicts = build_gold_records(step1a, state)
        summary = build_summary(step1a, state, records, conflicts, queue=build_queue(step1a))
        self.assertEqual(summary["gold_episodes"]["by_truth_class"]["REQUIRED_LITTER"], 1)
        self.assertEqual(summary["gold_episodes"]["by_truth_class"]["IGNORE_SMALL"], 1)
        self.assertEqual(summary["gold_episodes"]["gold_episode_count"], 1)
        self.assertEqual(summary["human_review"]["operations"]["confirm_operations"], 1)
        self.assertEqual(summary["human_review"]["operations"]["ignore_small_decisions"], 1)
        self.assertTrue(summary["human_review"]["review_complete"])
        boundaries = summary["boundaries"]
        self.assertFalse(boundaries["sealed_inference_accessed"])
        self.assertFalse(boundaries["training_started"])
        self.assertFalse(boundaries["annotation_complete_tiles_generated"])
        self.assertFalse(boundaries["hard_negative_mining_started"])
        self.assertFalse(boundaries["scene_version_fabricated"])

    def test_artifact_is_not_modified_by_a_full_run(self) -> None:
        root = self.tmpdir()
        artifact = root / "episode_candidates.jsonl"
        candidates = [
            make_candidate("ec-A", [make_card("batchx:a")]),
            make_candidate("ec-B", [make_card("batchx:b")]),
        ]
        artifact.write_text("".join(json.dumps(c) + "\n" for c in candidates), encoding="utf-8")
        step1a = load_step1a(artifact)
        before = sha256_file(artifact)
        self.assertEqual(step1a.artifact_sha256, before)
        state = ReviewState.load(root / "state.json", step1a=step1a)
        state.decide(step1a, "ec-A", "CONFIRM")
        build_gold_records(step1a, state)
        self.assertEqual(sha256_file(artifact), before)

    def test_load_step1a_rejects_duplicate_candidate_ids(self) -> None:
        root = self.tmpdir()
        artifact = root / "dupes.jsonl"
        candidate = make_candidate("ec-A", [make_card("batchx:a")])
        artifact.write_text(json.dumps(candidate) + "\n" + json.dumps(candidate) + "\n",
                            encoding="utf-8")
        with self.assertRaises(ValueError):
            load_step1a(artifact)

    def test_conflicting_split_and_merge_is_reported(self) -> None:
        step1a = make_step1a([
            make_candidate("ec-A", [make_card("batchx:a")]),
            make_candidate("ec-B", [make_card("batchx:b")]),
        ])
        state = self.make_state(step1a)
        state.decide(step1a, "ec-A", "MERGE", merge_targets=["ec-B"])
        state.candidates["ec-B"] = {
            "status": "reviewed", "decision": "SPLIT",
            "episodes": [{"draft_id": "e1", "member_card_ids": ["batchx:b"],
                          "truth_class": "REQUIRED_LITTER", "localization_status": "OK"}],
            "merge_targets": [], "truth_class": None, "note": ""}
        episodes, conflicts = resolve_episodes(step1a, state)
        self.assertEqual(episodes, [])
        self.assertTrue(any(c["type"] == "merge_component_contains_split" for c in conflicts))


# --------------------------------------------------------------------------- #
# ADD_MISSING_TARGET: human-discovered extra litter in the same image
# --------------------------------------------------------------------------- #

CTX = "assets/c-context.jpg"


def add_target(state, step1a, candidate_id="ec-A", *, card_id="batchx:a",
               truth_class="REQUIRED_LITTER", asset=CTX, x=50.0, y=25.0,
               width=200, height=100, asset_type="context", note="",
               force=False, target_id=None):
    return state.add_manual_target(
        step1a, candidate_id, card_id=card_id, truth_class=truth_class,
        clicked_asset_type=asset_type, clicked_asset_path=asset,
        x=x, y=y, image_width=width, image_height=height, note=note,
        allow_near_duplicate=force, manual_target_id=target_id)


class ManualTargetStateTest(TempMixin):
    def setUp(self) -> None:
        self.card = make_card("batchx:a")
        self.step1a = make_step1a([make_candidate("ec-A", [self.card])])

    def test_add_one_target_and_reload(self) -> None:
        state = self.make_state(self.step1a)
        record, warnings = add_target(state, self.step1a, target_id="mt-01021-0001")
        self.assertEqual(warnings, [])
        self.assertEqual(record["manual_target_id"], "mt-01021-0001")
        self.assertEqual(record["location_type"], "POINT")
        self.assertEqual(record["localization_status"], "NEEDS_RELOCALIZATION")
        self.assertEqual(record["origin"], "HUMAN_DISCOVERED_MISSING_TARGET")
        reloaded = ReviewState.load(state.path, step1a=self.step1a)
        self.assertIn("mt-01021-0001", reloaded.manual_targets)

    def test_add_multiple_targets_on_one_card(self) -> None:
        state = self.make_state(self.step1a)
        add_target(state, self.step1a, x=20, y=20, target_id="mt-01021-a")
        add_target(state, self.step1a, x=80, y=20, target_id="mt-01021-b")
        add_target(state, self.step1a, x=140, y=80, target_id="mt-01021-c")
        self.assertEqual(len(state.manual_targets_for("ec-A")), 3)
        self.assertEqual(len(state.manual_targets), 3)

    def test_truth_class_update_keeps_the_id(self) -> None:
        state = self.make_state(self.step1a)
        add_target(state, self.step1a, target_id="mt-01021-0001")
        updated = state.update_manual_target("mt-01021-0001", truth_class="IGNORE_SMALL")
        self.assertEqual(updated["manual_target_id"], "mt-01021-0001")
        self.assertEqual(updated["truth_class"], "IGNORE_SMALL")
        reloaded = ReviewState.load(state.path, step1a=self.step1a)
        self.assertEqual(reloaded.manual_targets["mt-01021-0001"]["truth_class"],
                         "IGNORE_SMALL")

    def test_delete_keeps_audit_history(self) -> None:
        state = self.make_state(self.step1a)
        add_target(state, self.step1a, target_id="mt-01021-0001")
        state.delete_manual_target("mt-01021-0001")
        self.assertNotIn("mt-01021-0001", state.manual_targets)
        actions = [row["action"] for row in state.audit_trail]
        self.assertIn("manual_target:add", actions)
        self.assertIn("manual_target:delete", actions)
        reloaded = ReviewState.load(state.path, step1a=self.step1a)
        self.assertEqual(reloaded.manual_targets, {})
        self.assertTrue(any(row["action"] == "manual_target:delete"
                            for row in reloaded.audit_trail))

    def test_repoint_changes_coordinates_but_not_id(self) -> None:
        state = self.make_state(self.step1a)
        add_target(state, self.step1a, x=10, y=10, target_id="mt-01021-0001")
        state.repoint_manual_target(
            self.step1a, "mt-01021-0001", clicked_asset_type="context",
            clicked_asset_path=CTX, x=150, y=80, image_width=200, image_height=100)
        row = state.manual_targets["mt-01021-0001"]
        self.assertEqual(row["manual_target_id"], "mt-01021-0001")
        self.assertEqual(row["point"]["x"], 150.0)
        self.assertEqual(row["point"]["x_norm"], 0.75)
        self.assertTrue(any(r["action"] == "manual_target:repoint"
                            for r in state.audit_trail))

    def test_every_operation_is_audited(self) -> None:
        state = self.make_state(self.step1a)
        add_target(state, self.step1a, target_id="mt-01021-0001")
        state.update_manual_target("mt-01021-0001", truth_class="UNCERTAIN")
        state.repoint_manual_target(
            self.step1a, "mt-01021-0001", clicked_asset_type="context",
            clicked_asset_path=CTX, x=11, y=11, image_width=200, image_height=100)
        state.delete_manual_target("mt-01021-0001")
        actions = [row["action"] for row in state.audit_trail]
        self.assertEqual(actions, ["manual_target:add", "manual_target:update",
                                   "manual_target:repoint", "manual_target:delete"])

    def test_id_is_not_an_array_index_and_is_stable(self) -> None:
        state = self.make_state(self.step1a)
        first, _ = add_target(state, self.step1a, x=20, y=20)
        state.delete_manual_target(first["manual_target_id"])
        second, _ = add_target(state, self.step1a, x=80, y=20)
        self.assertNotEqual(first["manual_target_id"], second["manual_target_id"])
        self.assertTrue(second["manual_target_id"].startswith("mt-01021-"))

    def test_unknown_truth_class_and_card_are_rejected(self) -> None:
        state = self.make_state(self.step1a)
        with self.assertRaises(ReviewError):
            add_target(state, self.step1a, truth_class="NON_LITTER")
        with self.assertRaises(ReviewError):
            add_target(state, self.step1a, card_id="batchx:zzz")
        with self.assertRaises(ReviewError):
            add_target(state, self.step1a, candidate_id="ec-unknown")

    def test_near_duplicate_asks_then_can_be_forced(self) -> None:
        state = self.make_state(self.step1a)
        add_target(state, self.step1a, x=50, y=25, target_id="mt-01021-0001")
        with self.assertRaises(NearDuplicateTarget) as ctx:
            add_target(state, self.step1a, x=52, y=26)
        self.assertEqual(len(ctx.exception.near_duplicates), 1)
        # nothing was persisted by the rejected attempt
        self.assertEqual(len(state.manual_targets), 1)
        forced, _ = add_target(state, self.step1a, x=52, y=26, force=True,
                               target_id="mt-01021-0002")
        self.assertEqual(len(state.manual_targets), 2)
        self.assertEqual(forced["manual_target_id"], "mt-01021-0002")

    def test_far_apart_points_do_not_trigger_the_prompt(self) -> None:
        state = self.make_state(self.step1a)
        add_target(state, self.step1a, x=10, y=10, target_id="mt-01021-0001")
        _, warnings = add_target(state, self.step1a, x=180, y=90,
                                 target_id="mt-01021-0002")
        self.assertEqual(warnings, [])

    def test_different_assets_are_never_compared(self) -> None:
        """Crops of different cards are different coordinate spaces: no prompt."""
        left = make_card("batchx:a", assets={
            "context_image": "assets/a-context.jpg", "crop_image": "assets/a-crop.jpg"})
        right = make_card("batchx:b", assets={
            "context_image": "assets/b-context.jpg", "crop_image": "assets/b-crop.jpg"})
        step1a = make_step1a([make_candidate("ec-A", [left, right])])
        state = ReviewState.load(self.tmpdir() / "two.json", step1a=step1a)
        state.add_manual_target(step1a, "ec-A", card_id="batchx:a",
            truth_class="REQUIRED_LITTER", clicked_asset_type="context",
            clicked_asset_path="assets/a-context.jpg", x=50, y=25,
            image_width=200, image_height=100, manual_target_id="mt-01021-0001")
        # identical coordinates, but a different card's crop -> must not be flagged
        _, warnings = state.add_manual_target(step1a, "ec-A", card_id="batchx:b",
            truth_class="REQUIRED_LITTER", clicked_asset_type="context",
            clicked_asset_path="assets/b-context.jpg", x=50, y=25,
            image_width=200, image_height=100, manual_target_id="mt-01021-0002")
        self.assertEqual(warnings, [])

    def test_mismatched_asset_path_is_rejected(self) -> None:
        state = self.make_state(self.step1a)
        with self.assertRaises(ReviewError):
            add_target(state, self.step1a, asset="assets/other-context.jpg")

    def test_adding_a_target_does_not_change_the_candidate_decision(self) -> None:
        state = self.make_state(self.step1a)
        state.decide(self.step1a, "ec-A", "CONFIRM")
        before = json.loads(state.path.read_text())["candidates"]["ec-A"]
        add_target(state, self.step1a, x=80, y=60, target_id="mt-01021-0001")
        after = json.loads(state.path.read_text())["candidates"]["ec-A"]
        self.assertEqual(before, after)
        self.assertEqual(state.status_of("ec-A"), "reviewed")

    def test_add_then_confirm_and_confirm_then_add_both_work(self) -> None:
        for order in ("add_first", "confirm_first"):
            state = self.make_state(self.step1a, name=f"state-{order}.json")
            if order == "add_first":
                add_target(state, self.step1a, x=80, y=60, target_id="mt-01021-0001")
                state.decide(self.step1a, "ec-A", "CONFIRM")
            else:
                state.decide(self.step1a, "ec-A", "CONFIRM")
                add_target(state, self.step1a, x=80, y=60, target_id="mt-01021-0001")
            records, _ = build_gold_records(self.step1a, state)
            kinds = sorted(r["record_kind"] for r in records)
            self.assertEqual(kinds, ["episode", "manual_missing_target"])


class ManualTargetCoordinateTest(TempMixin):
    def setUp(self) -> None:
        self.step1a = make_step1a([make_candidate("ec-A", [make_card("batchx:a")])])

    def test_css_scaled_click_maps_to_image_native(self) -> None:
        """A click at 25% of a CSS-scaled element must land at 25% natively."""
        state = self.make_state(self.step1a)
        # browser rendered a 1840x920 box for a 200x100 asset, click at 460,230
        scale = 1840 / 200
        native_x, native_y = 460 / scale, 230 / scale
        record, _ = add_target(state, self.step1a, x=native_x, y=native_y,
                               width=200, height=100, target_id="mt-01021-0001")
        self.assertEqual(record["point"]["x"], 50.0)
        self.assertEqual(record["point"]["y"], 25.0)
        self.assertEqual(record["point"]["x_norm"], 0.25)
        self.assertEqual(record["point"]["y_norm"], 0.25)

    def test_normalized_values_are_computed_by_the_server(self) -> None:
        state = self.make_state(self.step1a)
        record, _ = add_target(state, self.step1a, x=30, y=60, width=300, height=120,
                               target_id="mt-01021-0001")
        self.assertAlmostEqual(record["point"]["x_norm"], 0.1, places=8)
        self.assertAlmostEqual(record["point"]["y_norm"], 0.5, places=8)
        self.assertEqual(record["point"]["clicked_image_width"], 300)
        self.assertEqual(record["point"]["clicked_image_height"], 120)

    def test_out_of_bounds_clicks_are_rejected(self) -> None:
        state = self.make_state(self.step1a)
        for x, y in ((-1, 10), (10, -1), (201, 10), (10, 101)):
            with self.assertRaises(ReviewError):
                add_target(state, self.step1a, x=x, y=y, width=200, height=100)

    def test_zero_or_invalid_dimensions_are_rejected(self) -> None:
        state = self.make_state(self.step1a)
        with self.assertRaises(ReviewError):
            add_target(state, self.step1a, width=0, height=0)
        with self.assertRaises(ReviewError):
            add_target(state, self.step1a, width="abc", height=100)

    def test_asset_must_belong_to_the_source_card(self) -> None:
        state = self.make_state(self.step1a)
        with self.assertRaises(ReviewError):
            add_target(state, self.step1a, asset="assets/someone-elses.jpg")

    def test_invalid_asset_type_is_rejected(self) -> None:
        state = self.make_state(self.step1a)
        with self.assertRaises(ReviewError):
            add_target(state, self.step1a, asset_type="crop")

    def test_current_click_records_shared_base_asset(self) -> None:
        state = self.make_state(self.step1a)
        record, _ = add_target(state, self.step1a, asset_type="current",
                               target_id="mt-01021-0001")
        self.assertEqual(record["point"]["clicked_asset_type"], "current")
        self.assertEqual(record["point"]["clicked_asset_derived_from"], "context_image")


class ManualTargetBuildTest(TempMixin):
    def setUp(self) -> None:
        self.step1a = make_step1a([make_candidate("ec-A", [make_card("batchx:a")])])

    def _state_with(self, truth_class: str) -> ReviewState:
        state = self.make_state(self.step1a)
        add_target(state, self.step1a, truth_class=truth_class, x=80, y=60,
                   target_id="mt-01021-abcd1234ef56")
        return state

    def test_required_litter_becomes_its_own_gold_episode(self) -> None:
        records, conflicts = build_gold_records(
            self.step1a, self._state_with("REQUIRED_LITTER"))
        self.assertEqual(conflicts, [])
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["episode_id"], "ge-01021-manual-abcd1234ef56")
        self.assertTrue(record["is_litter_episode"])
        self.assertEqual(record["record_kind"], "manual_missing_target")
        self.assertEqual(record["origin"], "HUMAN_DISCOVERED_MISSING_TARGET")
        self.assertEqual(record["truth_target_id"], "mt-01021-abcd1234ef56")
        self.assertEqual(record["location_type"], "POINT")
        self.assertEqual(record["review_decision"], "ADD_MISSING_TARGET")

    def test_ignore_small_and_uncertain_are_emitted_too(self) -> None:
        for truth_class in ("IGNORE_SMALL", "UNCERTAIN"):
            records, _ = build_gold_records(self.step1a, self._state_with(truth_class))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["truth_class"], truth_class)
            self.assertFalse(records[0]["is_litter_episode"])
            self.assertEqual(records[0]["trainability_status"], "REVIEW_ONLY")

    def test_manual_target_never_merges_into_the_source_candidate(self) -> None:
        state = self.make_state(self.step1a)
        state.decide(self.step1a, "ec-A", "CONFIRM")
        add_target(state, self.step1a, x=80, y=60, target_id="mt-01021-0001")
        records, _ = build_gold_records(self.step1a, state)
        self.assertEqual(len(records), 2)
        episode = next(r for r in records if r["record_kind"] == "episode")
        manual = next(r for r in records if r["record_kind"] == "manual_missing_target")
        self.assertNotEqual(episode["episode_id"], manual["episode_id"])
        self.assertNotIn(manual["episode_id"], episode["source_episode_candidate_ids"])
        self.assertEqual(manual["source_episode_candidate_ids"], ["ec-A"])
        self.assertEqual(manual["member_card_ids"], ["batchx:a"])

    def test_identity_is_stable_across_rebuilds_and_reloads(self) -> None:
        state = self._state_with("REQUIRED_LITTER")
        first, _ = build_gold_records(self.step1a, state)
        reloaded = ReviewState.load(state.path, step1a=self.step1a)
        second, _ = build_gold_records(self.step1a, reloaded)
        self.assertEqual([r["episode_id"] for r in first],
                         [r["episode_id"] for r in second])
        # changing the class must not rename the target
        state.update_manual_target("mt-01021-abcd1234ef56", truth_class="UNCERTAIN")
        third, _ = build_gold_records(self.step1a, state)
        self.assertEqual(third[0]["episode_id"], first[0]["episode_id"])

    def test_multiple_manual_targets_yield_multiple_episodes(self) -> None:
        state = self.make_state(self.step1a)
        add_target(state, self.step1a, x=20, y=20, target_id="mt-01021-a")
        add_target(state, self.step1a, x=80, y=20, target_id="mt-01021-b")
        add_target(state, self.step1a, x=140, y=80, target_id="mt-01021-c")
        records, _ = build_gold_records(self.step1a, state)
        self.assertEqual(len(records), 3)
        self.assertEqual(len({r["episode_id"] for r in records}), 3)

    def test_localization_is_always_needs_relocalization(self) -> None:
        records, _ = build_gold_records(
            self.step1a, self._state_with("REQUIRED_LITTER"))
        self.assertEqual(records[0]["localization_status"], "NEEDS_RELOCALIZATION")
        self.assertEqual(records[0]["point"]["x"], 80.0)

    def test_visible_interval_reflects_before_after_availability(self) -> None:
        with_ba = make_card("batchx:a")
        without = make_card("batchx:b", assets={
            "context_image": "assets/c-context.jpg", "crop_image": "assets/c-crop.jpg"})
        step1a = make_step1a([make_candidate("ec-A", [with_ba, without])])
        state = ReviewState.load(self.tmpdir() / "s.json", step1a=step1a)
        state.add_manual_target(step1a, "ec-A", card_id="batchx:a",
            truth_class="REQUIRED_LITTER", clicked_asset_type="context",
            clicked_asset_path=CTX, x=10, y=10, image_width=100, image_height=100,
            manual_target_id="mt-01021-a")
        state.add_manual_target(step1a, "ec-A", card_id="batchx:b",
            truth_class="REQUIRED_LITTER", clicked_asset_type="context",
            clicked_asset_path=CTX, x=10, y=10, image_width=100, image_height=100,
            manual_target_id="mt-01021-b")
        records, _ = build_gold_records(step1a, state)
        by_card = {r["source_member_card_id"]: r for r in records}
        self.assertEqual(by_card["batchx:a"]["visible_intervals"][0]["start"],
                         "2026-09-20 09:59:58")
        self.assertEqual(by_card["batchx:b"]["visible_intervals"][0]["start"],
                         "2026-09-20 10:00:00")
        self.assertIn("before_after",
                      by_card["batchx:a"]["visible_intervals"][0]["basis"])

    def test_summary_reports_the_manual_block(self) -> None:
        state = self.make_state(self.step1a)
        state.decide(self.step1a, "ec-A", "CONFIRM")
        add_target(state, self.step1a, truth_class="REQUIRED_LITTER", x=20, y=20,
                   target_id="mt-01021-a")
        add_target(state, self.step1a, truth_class="IGNORE_SMALL", x=80, y=20,
                   target_id="mt-01021-b")
        records, conflicts = build_gold_records(self.step1a, state)
        summary = build_summary(self.step1a, state, records, conflicts,
                                queue=build_queue(self.step1a))
        block = summary["manual_missing_targets"]
        self.assertEqual(block["manual_missing_target_count"], 2)
        self.assertEqual(block["manual_required_count"], 1)
        self.assertEqual(block["manual_ignore_small_count"], 1)
        self.assertEqual(block["manual_uncertain_count"], 0)
        self.assertEqual(block["manual_targets_by_camera"], {"01021": 2})
        self.assertEqual(block["manual_targets_needing_relocalization"], 2)
        self.assertFalse(block["manual_targets_merged_automatically"])
        self.assertIs(block["manual_targets_are_independent_of_source_candidate"], True)
        self.assertGreaterEqual(block["manual_targets_trainable_lineage"]
                                + block["manual_targets_lineage_unresolved"], 1)
        self.assertEqual(summary["gold_episodes"]["manual_gold_episode_count"], 1)
        self.assertEqual(summary["human_review"]["operations"]
                         ["add_missing_target_operations"], 2)
        self.assertFalse(summary["boundaries"]["manual_target_bbox_inferred"])


class ManualTargetTrainabilityTest(TempMixin):
    def _record(self, *, file_id, frame_id="f00s00", retention_days=7.0,
                reference=datetime(2026, 9, 21, 0, 0, 0)):
        # Trainability inspects real asset presence, so materialise the crop files
        # and point the Step 1A input at the batch directory (as in production).
        root = self.tmpdir()
        batch = root / "batchdir"
        (batch / "assets").mkdir(parents=True, exist_ok=True)
        for name in ("c-context.jpg", "c-crop.jpg", "c-before.jpg", "c-after.jpg"):
            (batch / "assets" / name).write_bytes(b"\xff\xd8\xff\xd9")
        card = make_card("batchx:a", file_id=file_id, frame_id=frame_id)
        step1a = make_step1a([make_candidate("ec-A", [card])],
                             batch_dirs={"batchx": str(batch)})
        state = ReviewState.load(root / "s.json", step1a=step1a)
        state.add_manual_target(step1a, "ec-A", card_id="batchx:a",
            truth_class="REQUIRED_LITTER", clicked_asset_type="context",
            clicked_asset_path=CTX, x=10, y=10, image_width=100, image_height=100,
            manual_target_id="mt-01021-0001")
        records, _ = build_gold_records(
            step1a, state,
            policy=TrainabilityPolicy(retention_days=retention_days,
                                      reference_time=reference))
        return records[0]

    def test_source_with_ps_lineage_inherits_trainability(self) -> None:
        record = self._record(file_id="ps-1")
        self.assertEqual(record["trainability_status"], "TRAINABLE_SOURCE_NATIVE")
        self.assertEqual(record["trainability_evidence"]["lineage_method"],
                         "B_ps_file_id_plus_timestamp")

    def test_trainable_lineage_still_needs_relocalization(self) -> None:
        record = self._record(file_id="ps-1")
        self.assertEqual(record["trainability_status"], "TRAINABLE_SOURCE_NATIVE")
        self.assertEqual(record["localization_status"], "NEEDS_RELOCALIZATION")

    def test_source_without_lineage_cannot_be_promoted(self) -> None:
        record = self._record(file_id=None, frame_id="f00s00")
        self.assertEqual(record["trainability_status"], "LINEAGE_UNRESOLVED")
        self.assertEqual(record["localization_status"], "NEEDS_RELOCALIZATION")

    def test_source_with_no_recovery_route_is_review_only(self) -> None:
        record = self._record(file_id=None, frame_id=None)
        self.assertEqual(record["trainability_status"], "REVIEW_ONLY")
        self.assertEqual(record["localization_status"], "NEEDS_RELOCALIZATION")

    def test_expired_retention_downgrades_to_review_only(self) -> None:
        record = self._record(file_id="ps-1", reference=datetime(2026, 10, 20, 0, 0, 0))
        self.assertEqual(record["trainability_status"], "REVIEW_ONLY")
        self.assertEqual(record["localization_status"], "NEEDS_RELOCALIZATION")


class ManualTargetCompatibilityTest(TempMixin):
    """The patch must not disturb any pre-existing Step 1B operation."""

    def setUp(self) -> None:
        self.step1a = make_step1a([
            make_candidate("ec-A", [make_card("batchx:a")]),
            make_candidate("ec-B", [make_card("batchx:b", bbox=(600, 600, 640, 640))]),
        ])

    def test_all_previous_decisions_still_work(self) -> None:
        for decision in ("CONFIRM", "NON_LITTER", "IGNORE_SMALL", "UNCERTAIN"):
            state = self.make_state(self.step1a, name=f"s-{decision}.json")
            state.decide(self.step1a, "ec-A", decision)
            record = ReviewState.load(state.path, step1a=self.step1a).get("ec-A")
            self.assertEqual(record["decision"], decision)

    def test_split_and_merge_still_work_alongside_manual_targets(self) -> None:
        state = self.make_state(self.step1a)
        state.decide(self.step1a, "ec-A", "MERGE", merge_targets=["ec-B"])
        add_target(state, self.step1a, candidate_id="ec-A", x=80, y=60,
                   target_id="mt-01021-0001")
        records, conflicts = build_gold_records(self.step1a, state)
        self.assertEqual(conflicts, [])
        kinds = sorted(r["record_kind"] for r in records)
        self.assertEqual(kinds, ["episode", "manual_missing_target"])

    def test_schema_is_v2_and_episode_ids_are_unchanged(self) -> None:
        self.assertEqual(REVIEW_SCHEMA_VERSION, "gold_episode_review_v2")
        # frozen regression values: bumping the state schema must not rename them
        self.assertEqual(episode_id_for("01021", ["a"]), "ge-01021-cbe59f4c7d44")
        self.assertEqual(episode_id_for("01030", ["y", "x"]), "ge-01030-0c2f523e71ca")

    def test_v1_state_loads_without_losing_decisions(self) -> None:
        state = self.make_state(self.step1a)
        state.decide(self.step1a, "ec-A", "CONFIRM", note="legacy")
        legacy = json.loads(state.path.read_text())
        legacy["review_schema_version"] = "gold_episode_review_v1"
        legacy.pop("manual_targets", None)          # a genuine v1 file has no such key
        state.path.write_text(json.dumps(legacy), encoding="utf-8")
        reloaded = ReviewState.load(state.path, step1a=self.step1a)
        self.assertEqual(reloaded.get("ec-A")["decision"], "CONFIRM")
        self.assertEqual(reloaded.get("ec-A")["note"], "legacy")
        self.assertEqual(reloaded.manual_targets, {})
        self.assertEqual(reloaded.progress(self.step1a)["reviewed"], 1)

    def test_v1_then_manual_target_survives_save_and_reload(self) -> None:
        state = self.make_state(self.step1a)
        state.decide(self.step1a, "ec-A", "CONFIRM")
        legacy = json.loads(state.path.read_text())
        legacy.pop("manual_targets", None)
        state.path.write_text(json.dumps(legacy), encoding="utf-8")
        reloaded = ReviewState.load(state.path, step1a=self.step1a)
        add_target(reloaded, self.step1a, x=80, y=60, target_id="mt-01021-0001")
        again = ReviewState.load(state.path, step1a=self.step1a)
        self.assertEqual(again.get("ec-A")["decision"], "CONFIRM")
        self.assertIn("mt-01021-0001", again.manual_targets)

    def test_v1_state_migrates_to_v2_on_write(self) -> None:
        state = self.make_state(self.step1a)
        state.decide(self.step1a, "ec-A", "CONFIRM")
        legacy = json.loads(state.path.read_text())
        legacy["review_schema_version"] = "gold_episode_review_v1"
        legacy.pop("manual_targets", None)
        state.path.write_text(json.dumps(legacy), encoding="utf-8")

        reloaded = ReviewState.load(state.path, step1a=self.step1a)
        self.assertEqual(reloaded.schema_version, "gold_episode_review_v1")
        reloaded.save()
        written = json.loads(state.path.read_text())
        self.assertEqual(written["review_schema_version"], REVIEW_SCHEMA_VERSION)
        self.assertEqual(written["loaded_schema_version"], "gold_episode_review_v1")
        # migration must not lose progress
        self.assertEqual(written["candidates"]["ec-A"]["decision"], "CONFIRM")
        self.assertEqual(written["manual_targets"], {})

    def test_fingerprint_changes_when_a_manual_target_is_added(self) -> None:
        state = self.make_state(self.step1a)
        before = review_fingerprint(state)
        add_target(state, self.step1a, x=80, y=60, target_id="mt-01021-0001")
        self.assertNotEqual(before, review_fingerprint(state))


if __name__ == "__main__":
    unittest.main()
