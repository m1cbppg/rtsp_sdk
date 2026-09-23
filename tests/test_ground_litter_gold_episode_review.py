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


if __name__ == "__main__":
    unittest.main()
