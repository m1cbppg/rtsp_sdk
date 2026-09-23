"""Step 1C-1: localization review tests.

Pure stdlib: the proposal engine is injected as a fake, so no cv2/network is needed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from rtsp_annotator.ground_litter_localization_review import (
    MAX_PROPOSALS,
    SCHEMA_VERSION,
    LocalizationEpisode,
    LocalizationError,
    LocalizationInput,
    ReviewError,
    ReviewState,
    Reviewer,
    SealedAssetError,
    assert_not_sealed,
    bbox_iou,
    build_manifest,
    build_summary,
    build_upstream_provenance,
    classify_origin,
    context_crop_box,
    load_localization_input,
    map_point_to_source,
    normalise_bbox,
    sha256_file,
    validate_bbox,
    validate_proposals,
)

W, H = 2560, 1440


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def gold_record(episode_id, camera, timestamp, *, bbox=None, label="LITTER",
                kind="episode", point=None, split=False, file_id="ps-1",
                localization="OK") -> dict:
    member = {"card_id": f"batch:{camera}-{episode_id.split('-')[-1]}", "batch_key": "batch",
              "frame_id": "f00s00", "timestamp": timestamp, "bbox": bbox,
              "original_label": label, "source_file_id": file_id,
              "asset_paths": {"context_image": "assets/x.jpg"}}
    record = {
        "episode_id": episode_id, "truth_class": "REQUIRED_LITTER", "camera_id": camera,
        "scene_version": "UNKNOWN_HISTORICAL", "start_timestamp": timestamp,
        "end_timestamp": timestamp, "member_card_ids": [member["card_id"]],
        "source_episode_candidate_ids": ["ec-x"], "record_kind": kind,
        "localization_status": localization,
        "trainability_evidence": {"members": [member]},
        "grouping_review": {"confirmed": True, "split": split, "merged": False,
                            "merge_source_candidates": []},
        # deliberately the broken Step 1B value, to prove it is not relied upon
        "original_label_summary": {"UNKNOWN": 1},
    }
    if point is not None:
        record["point"] = point
    return record


class Base(unittest.TestCase):
    def tmpdir(self) -> Path:
        path = Path(tempfile.mkdtemp(prefix="step1c1-"))
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def build_input(self, records, *, labels=None, evidence_extra=None):
        root = self.tmpdir()
        gold = root / "gold_episodes.jsonl"
        gold.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        recovery = root / "recovery"
        (recovery / "verification_frames").mkdir(parents=True, exist_ok=True)
        rows = []
        for record in records:
            frame = recovery / "verification_frames" / f"{record['episode_id']}.jpg"
            frame.write_bytes(b"\xff\xd8\xff\xd9")
            row = {
                "episode_id": record["episode_id"],
                "camera_id": record["camera_id"],
                "source_file_id": "ps-1",
                "requested_timestamp": record["start_timestamp"],
                "source_width": W, "source_height": H,
                "verification_frame_path": str(frame),
                "source_recovery_status": "RECOVERED_SOURCE_NATIVE",
                "source_file_resolution_method": "A_known_file_id",
            }
            row.update(evidence_extra or {})
            rows.append(row)
        (recovery / "episode_source_evidence.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        step1a = None
        if labels:
            step1a = root / "step1a.jsonl"
            step1a.write_text(json.dumps({"lineage": {"review_cards": [
                {"card_id": cid, "label": lab} for cid, lab in labels.items()]}}) + "\n",
                encoding="utf-8")
        return load_localization_input(gold, recovery, step1a_artifact=step1a), root


class FakeEngine:
    """Deterministic stand-in for the cv2 engine."""

    def __init__(self, boxes=None, fail=False):
        self.boxes = boxes or [[100.0, 100.0, 140.0, 140.0], [200.0, 200.0, 260.0, 260.0],
                               [300.0, 300.0, 340.0, 340.0]]
        self.fail = fail
        self.calls = []

    def propose(self, *, frame_path, seed_bbox, point, frame_width, frame_height, revision):
        self.calls.append({"seed": seed_bbox, "point": point, "revision": revision})
        if self.fail:
            raise LocalizationError("engine exploded")
        return [{"bbox": list(b), "method": f"fake_{i}", "revision": revision}
                for i, b in enumerate(self.boxes)]


def reviewer(data, state, engine=None):
    return Reviewer(data=data, state=state, engine=engine)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


class GeometryTest(unittest.TestCase):
    def test_normalise_rejects_degenerate(self) -> None:
        self.assertIsNone(normalise_bbox([10, 10, 10, 20]))
        self.assertIsNone(normalise_bbox([10, 10, 20, 10]))
        self.assertIsNone(normalise_bbox([float("nan"), 0, 10, 10]))
        self.assertIsNone(normalise_bbox(None))
        self.assertEqual(normalise_bbox([1, 2, 3, 4]), (1.0, 2.0, 3.0, 4.0))

    def test_validate_bbox_rejects_out_of_frame_and_zero_area(self) -> None:
        self.assertFalse(validate_bbox([-1, 0, 10, 10], frame_width=W,
                                       frame_height=H)["ok"])
        self.assertFalse(validate_bbox([0, 0, W + 5, 10], frame_width=W,
                                       frame_height=H)["ok"])
        self.assertFalse(validate_bbox([10, 10, 10, 20], frame_width=W,
                                       frame_height=H)["ok"])
        self.assertTrue(validate_bbox([10, 10, 20, 20], frame_width=W,
                                      frame_height=H)["ok"])

    def test_validate_bbox_flags_oversized_for_confirmation(self) -> None:
        verdict = validate_bbox([0, 0, 2000, 1200], frame_width=W, frame_height=H)
        self.assertTrue(verdict["requires_confirmation"])
        self.assertIn("bbox_covers_large_fraction_of_frame", verdict["issues"])
        confirmed = validate_bbox([0, 0, 2000, 1200], frame_width=W, frame_height=H,
                                  require_confirmation_over_fraction=1.1)
        self.assertTrue(confirmed["ok"])

    def test_context_crop_box_reproduces_the_historical_crop(self) -> None:
        # values verified against the real Silver review data
        self.assertEqual(context_crop_box([1836, 349, 1864, 370]), (1794, 304, 1906, 416))
        self.assertEqual(context_crop_box([1214, 1320, 1248, 1351]), (1163, 1268, 1299, 1404))

    def test_map_point_to_source_verified(self) -> None:
        point = {"x_norm": 0.50666667, "y_norm": 0.89666667,
                 "clicked_image_width": 112, "clicked_image_height": 112}
        result = map_point_to_source([1836, 349, 1864, 370], point,
                                     frame_width=W, frame_height=H)
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["x"], 1850.747, places=2)
        self.assertAlmostEqual(result["y"], 404.427, places=2)

    def test_map_point_rejects_a_css_resized_recording(self) -> None:
        """A recorded size that does not match the derived crop must not be trusted."""
        point = {"x_norm": 0.5, "y_norm": 0.5,
                 "clicked_image_width": 640, "clicked_image_height": 640}
        result = map_point_to_source([1836, 349, 1864, 370], point,
                                     frame_width=W, frame_height=H)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "derived_crop_size_mismatch")

    def test_map_point_rejects_out_of_range_norm(self) -> None:
        point = {"x_norm": 1.4, "y_norm": 0.5, "clicked_image_width": 112,
                 "clicked_image_height": 112}
        self.assertFalse(map_point_to_source([1836, 349, 1864, 370], point,
                                             frame_width=W, frame_height=H)["ok"])

    def test_bbox_iou(self) -> None:
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(bbox_iou([0, 0, 10, 10], [50, 50, 60, 60]), 0.0)


# --------------------------------------------------------------------------- #
# loading / screening / origin
# --------------------------------------------------------------------------- #


class LoaderTest(Base):
    def test_origin_comes_from_member_labels_not_the_broken_summary(self) -> None:
        records = [
            gold_record("ep-litter", "01021", "2026-09-19 01:00:00", bbox=[10, 10, 40, 40]),
            gold_record("ep-wrong", "01021", "2026-09-19 01:01:00", bbox=[50, 50, 80, 80],
                        label="BOX_WRONG"),
            gold_record("ep-manual", "01021", "2026-09-19 01:02:00", label="LITTER",
                        kind="manual_missing_target",
                        point={"x_norm": 0.5, "y_norm": 0.5, "clicked_image_width": 112,
                               "clicked_image_height": 112}),
            gold_record("ep-split", "01021", "2026-09-19 01:03:00", bbox=[90, 90, 120, 120],
                        split=True),
        ]
        data, _ = self.build_input(records)
        origins = {e.episode_id: e.origin for e in data.required}
        self.assertEqual(origins["ep-litter"], "historical_litter")
        self.assertEqual(origins["ep-wrong"], "box_wrong")
        self.assertEqual(origins["ep-manual"], "manual_missing_target")
        self.assertEqual(origins["ep-split"], "split_derived")

    def test_manual_target_uses_point_not_the_parent_bbox(self) -> None:
        record = gold_record("ep-manual", "01021", "2026-09-19 01:02:00", bbox=[1836, 349, 1864, 370],
                             kind="manual_missing_target",
                             point={"x_norm": 0.5, "y_norm": 0.5,
                                    "clicked_image_width": 112, "clicked_image_height": 112})
        data, _ = self.build_input([record])
        episode = data.required[0]
        self.assertIsNone(episode.original_bbox)
        self.assertEqual(episode.original_bbox_source,
                         "none_manual_target_has_point_only")
        self.assertTrue(episode.point_source["ok"])

    def test_box_wrong_and_manual_are_flagged_likely_box_bad(self) -> None:
        records = [
            gold_record("ep-wrong", "01021", "2026-09-19 01:01:00", bbox=[50, 50, 80, 80],
                        label="BOX_WRONG"),
            gold_record("ep-manual", "01021", "2026-09-19 01:02:00", kind="manual_missing_target",
                        point={"x_norm": 0.5, "y_norm": 0.5, "clicked_image_width": 112,
                               "clicked_image_height": 112}),
            gold_record("ep-ok", "01021", "2026-09-19 01:03:00", bbox=[100, 100, 130, 130]),
        ]
        data, _ = self.build_input(records)
        screens = {e.episode_id: e.screen for e in data.required}
        self.assertTrue(screens["ep-wrong"]["likely_box_bad"])
        self.assertIn("historical_box_wrong", screens["ep-wrong"]["screen_reasons"])
        self.assertTrue(screens["ep-manual"]["likely_box_bad"])
        self.assertIn("manual_point_without_bbox", screens["ep-manual"]["screen_reasons"])
        self.assertFalse(screens["ep-ok"]["likely_box_bad"])

    def test_two_episodes_sharing_one_box_are_flagged(self) -> None:
        records = [
            gold_record("ep-a", "01022", "2026-09-19 01:00:00", bbox=[100, 100, 200, 200]),
            gold_record("ep-b", "01022", "2026-09-19 04:00:00", bbox=[102, 102, 202, 202]),
        ]
        data, _ = self.build_input(records)
        for episode in data.required:
            self.assertTrue(episode.shared_bbox_episode_ids)
            self.assertIn("shared_bbox_with_other_episodes",
                          episode.screen["screen_reasons"])
            # hours apart -> not the same-frame "one box, two objects" case
            self.assertNotIn("possible_multi_object_box",
                             episode.screen["screen_reasons"])

    def test_same_frame_shared_box_is_the_multi_object_case(self) -> None:
        records = [
            gold_record("ep-a", "01022", "2026-09-19 01:00:00", bbox=[100, 100, 200, 200]),
            gold_record("ep-b", "01022", "2026-09-19 01:00:30", bbox=[102, 102, 202, 202]),
        ]
        data, _ = self.build_input(records)
        for episode in data.required:
            self.assertIn("possible_multi_object_box", episode.screen["screen_reasons"])

    def test_screening_orders_risky_first(self) -> None:
        records = [
            gold_record("ep-ok", "01021", "2026-09-19 01:00:00", bbox=[100, 100, 130, 130]),
            gold_record("ep-wrong", "01021", "2026-09-19 01:01:00", bbox=[50, 50, 80, 80],
                        label="BOX_WRONG"),
        ]
        data, _ = self.build_input(records)
        self.assertEqual(data.required[0].episode_id, "ep-wrong")

    def test_non_required_records_are_ignored_and_duplicates_rejected(self) -> None:
        record = gold_record("ep-1", "01021", "2026-09-19 01:00:00", bbox=[10, 10, 40, 40])
        other = dict(record, episode_id="ep-2", truth_class="NON_LITTER")
        data, _ = self.build_input([record, other])
        self.assertEqual([e.episode_id for e in data.required], ["ep-1"])
        root = self.tmpdir()
        gold = root / "g.jsonl"
        gold.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
        (root / "rec").mkdir()
        (root / "rec" / "episode_source_evidence.jsonl").write_text("", encoding="utf-8")
        with self.assertRaises(LocalizationError):
            load_localization_input(gold, root / "rec")


# --------------------------------------------------------------------------- #
# proposals
# --------------------------------------------------------------------------- #


class ProposalTest(unittest.TestCase):
    def test_letters_are_assigned_by_rank(self) -> None:
        cleaned = validate_proposals(
            [{"bbox": [10, 10, 50, 50], "method": "m1"},
             {"bbox": [60, 60, 90, 90], "method": "m2"}],
            frame_width=W, frame_height=H)
        self.assertEqual([p["proposal_id"] for p in cleaned], ["A", "B"])

    def test_illegal_proposals_are_dropped(self) -> None:
        cleaned = validate_proposals(
            [{"bbox": [10, 10, 10, 50]}, {"bbox": [-5, 0, 10, 10]},
             {"bbox": [float("nan"), 0, 10, 10]}, {"bbox": [20, 20, 60, 60]}],
            frame_width=W, frame_height=H)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["bbox"], [20.0, 20.0, 60.0, 60.0])

    def test_proposals_capped_at_three(self) -> None:
        rows = [{"bbox": [i * 100, 0, i * 100 + 50, 50]} for i in range(1, 6)]
        self.assertEqual(len(validate_proposals(rows, frame_width=W, frame_height=H)),
                         MAX_PROPOSALS)

    def test_normalised_bbox_is_derived(self) -> None:
        cleaned = validate_proposals([{"bbox": [0, 0, 1280, 720]}], frame_width=W,
                                     frame_height=H)
        self.assertEqual(cleaned[0]["bbox_norm"], [0.0, 0.0, 0.5, 0.5])


# --------------------------------------------------------------------------- #
# decisions
# --------------------------------------------------------------------------- #


class DecisionTest(Base):
    def _data(self, records):
        data, root = self.build_input(records)
        state = ReviewState.load(root / "state.json", data=data)
        return data, state

    def test_box_ok_verifies_and_keeps_source_xyxy(self) -> None:
        data, state = self._data([gold_record("ep-1", "01021", "2026-09-19 01:00:00",
                                              bbox=[874, 774, 896, 787])])
        row = state.decide_box_ok(data.required[0])
        self.assertEqual(row["localization_status"], "VERIFIED_BBOX")
        self.assertEqual(row["localization_decision"], "BOX_OK")
        self.assertEqual(row["verified_bbox"], [874.0, 774.0, 896.0, 787.0])
        self.assertEqual(row["location_type"], "BBOX")
        self.assertEqual(row["proposal_source_type"], "original_bbox_unchanged")

    def test_box_ok_refuses_a_broken_bbox(self) -> None:
        data, state = self._data([gold_record("ep-1", "01021", "2026-09-19 01:00:00",
                                              bbox=[0, 0, 10, 10])])
        # force a degenerate box past the loader by editing the dataclass
        broken = LocalizationEpisode(**{**data.required[0].__dict__,
                                        "original_bbox": (10.0, 10.0, 10.0, 20.0)})
        with self.assertRaises(ReviewError):
            state.decide_box_ok(broken)

    def test_box_ok_requires_confirmation_for_an_oversized_box(self) -> None:
        data, state = self._data([gold_record("ep-1", "01021", "2026-09-19 01:00:00",
                                              bbox=[0, 0, 2000, 1200])])
        with self.assertRaises(ReviewError):
            state.decide_box_ok(data.required[0])
        row = state.decide_box_ok(data.required[0], confirm_oversized=True)
        self.assertEqual(row["localization_status"], "VERIFIED_BBOX")
        self.assertTrue(row["oversized_confirmed_by_human"])

    def test_proposal_flow_produces_verified_bbox(self) -> None:
        data, state = self._data([gold_record("ep-1", "01021", "2026-09-19 01:00:00",
                                              bbox=[50, 50, 80, 80], label="BOX_WRONG")])
        engine = FakeEngine()
        reviewer(data, state, engine).generate_proposals("ep-1")
        stored = state.get("ep-1")
        self.assertEqual([p["proposal_id"] for p in stored["proposals"]], ["A", "B", "C"])
        self.assertEqual(stored["proposal_revision"], 1)
        row = state.select_proposal(data.required[0], "B")
        self.assertEqual(row["localization_status"], "VERIFIED_BBOX")
        self.assertEqual(row["localization_decision"], "PROPOSAL_SELECTED")
        self.assertEqual(row["verified_bbox"], [200.0, 200.0, 260.0, 260.0])
        self.assertEqual(row["selected_proposal_id"], "B")

    def test_proposal_alone_is_not_truth(self) -> None:
        """Machine proposals must never become VERIFIED_BBOX without a human pick."""
        data, state = self._data([gold_record("ep-1", "01021", "2026-09-19 01:00:00",
                                              bbox=[50, 50, 80, 80], label="BOX_WRONG")])
        reviewer(data, state, FakeEngine()).generate_proposals("ep-1")
        stored = state.get("ep-1")
        self.assertIsNone(stored.get("verified_bbox"))
        self.assertNotEqual(stored.get("localization_status"), "VERIFIED_BBOX")
        records = reviewer(data, state).build_records()
        self.assertFalse(records[0]["training_localization_ready"])

    def test_unknown_proposal_id_is_rejected(self) -> None:
        data, state = self._data([gold_record("ep-1", "01021", "2026-09-19 01:00:00",
                                              bbox=[50, 50, 80, 80])])
        reviewer(data, state, FakeEngine()).generate_proposals("ep-1")
        with self.assertRaises(ReviewError):
            state.select_proposal(data.required[0], "Z")

    def test_selecting_without_proposals_is_rejected(self) -> None:
        data, state = self._data([gold_record("ep-1", "01021", "2026-09-19 01:00:00",
                                              bbox=[50, 50, 80, 80])])
        with self.assertRaises(ReviewError):
            state.select_proposal(data.required[0], "A")

    def test_unresolved_never_produces_a_bbox(self) -> None:
        data, state = self._data([gold_record("ep-1", "01021", "2026-09-19 01:00:00",
                                              bbox=[50, 50, 80, 80], label="BOX_WRONG")])
        reviewer(data, state, FakeEngine()).generate_proposals("ep-1")
        row = state.decide_unresolved(data.required[0])
        self.assertEqual(row["localization_status"], "LOCALIZATION_UNRESOLVED")
        self.assertIsNone(row["verified_bbox"])
        records = reviewer(data, state).build_records()
        self.assertFalse(records[0]["training_localization_ready"])

    def test_truth_review_required_needs_a_reason(self) -> None:
        data, state = self._data([gold_record("ep-1", "01021", "2026-09-19 01:00:00",
                                              bbox=[50, 50, 80, 80])])
        with self.assertRaises(ReviewError):
            state.decide_truth_review_required(data.required[0], reason="  ")
        row = state.decide_truth_review_required(
            data.required[0], reason="one box appears to hold two Gold targets")
        self.assertEqual(row["localization_status"], "TRUTH_REVIEW_REQUIRED")
        self.assertIsNone(row["verified_bbox"])

    def test_engine_failure_is_surfaced(self) -> None:
        data, state = self._data([gold_record("ep-1", "01021", "2026-09-19 01:00:00",
                                              bbox=[50, 50, 80, 80])])
        with self.assertRaises(LocalizationError):
            reviewer(data, state, FakeEngine(fail=True)).generate_proposals("ep-1")

    def test_missing_frame_is_surfaced(self) -> None:
        data, state = self._data([gold_record("ep-1", "01021", "2026-09-19 01:00:00",
                                              bbox=[50, 50, 80, 80])])
        Path(data.required[0].verification_frame_path).unlink()
        with self.assertRaises(ReviewError):
            reviewer(data, state, FakeEngine()).generate_proposals("ep-1")


# --------------------------------------------------------------------------- #
# manual point
# --------------------------------------------------------------------------- #


class ManualPointTest(Base):
    def test_point_becomes_a_bbox_and_the_original_point_is_kept(self) -> None:
        record = gold_record("ep-manual", "01021", "2026-09-19 01:00:00",
                             bbox=[1836, 349, 1864, 370], kind="manual_missing_target",
                             localization="NEEDS_RELOCALIZATION",
                             point={"x_norm": 0.50666667, "y_norm": 0.89666667,
                                    "clicked_image_width": 112, "clicked_image_height": 112})
        data, root = self.build_input([record])
        state = ReviewState.load(root / "s.json", data=data)
        episode = data.required[0]
        engine = FakeEngine(boxes=[[1748, 378, 1895, 435]])
        reviewer(data, state, engine).generate_proposals("ep-manual")
        # the engine must have received the source-frame point, not a crop coordinate
        self.assertAlmostEqual(engine.calls[0]["point"][0], 1850.747, places=2)
        self.assertAlmostEqual(engine.calls[0]["point"][1], 404.427, places=2)

        row = state.select_proposal(episode, "A")
        self.assertEqual(row["location_type"], "BBOX")
        self.assertEqual(row["localization_status"], "VERIFIED_BBOX")
        self.assertEqual(row["verified_bbox"], [1748.0, 378.0, 1895.0, 435.0])
        # the manual point evidence must survive
        self.assertIsNotNone(row["original_point"])
        self.assertTrue(row["original_point_source"]["ok"])

    def test_manual_target_without_a_usable_point_is_screened(self) -> None:
        record = gold_record("ep-manual", "01021", "2026-09-19 01:00:00",
                             kind="manual_missing_target",
                             point={"x_norm": 0.5, "y_norm": 0.5,
                                    "clicked_image_width": 999, "clicked_image_height": 999})
        data, _ = self.build_input([record])
        self.assertIn("manual_point_mapping_unverified",
                      data.required[0].screen["screen_reasons"])


# --------------------------------------------------------------------------- #
# summary / records / safety
# --------------------------------------------------------------------------- #


class SummaryTest(Base):
    def test_records_cover_all_episodes_and_ready_flag(self) -> None:
        records = [
            gold_record("ep-ok", "01021", "2026-09-19 01:00:00", bbox=[100, 100, 130, 130]),
            gold_record("ep-wrong", "01021", "2026-09-19 01:01:00", bbox=[50, 50, 80, 80],
                        label="BOX_WRONG"),
            gold_record("ep-manual", "01022", "2026-09-19 01:02:00", kind="manual_missing_target",
                        point={"x_norm": 0.5, "y_norm": 0.5, "clicked_image_width": 112,
                               "clicked_image_height": 112}),
        ]
        data, root = self.build_input(records)
        state = ReviewState.load(root / "s.json", data=data)
        state.decide_box_ok(data.required[0] if data.required[0].episode_id == "ep-ok"
                            else next(e for e in data.required if e.episode_id == "ep-ok"))
        rows = reviewer(data, state).build_records()
        self.assertEqual(len(rows), 3)
        by_id = {r["episode_id"]: r for r in rows}
        self.assertTrue(by_id["ep-ok"]["training_localization_ready"])
        self.assertFalse(by_id["ep-wrong"]["training_localization_ready"])
        self.assertFalse(by_id["ep-manual"]["training_localization_ready"])
        self.assertEqual(by_id["ep-wrong"]["localization_status"], "NEEDS_RELOCALIZATION")

        summary = build_summary(data, rows, state)
        self.assertEqual(summary["input"]["required_episode_count"], 3)
        self.assertEqual(summary["review"]["reviewed"], 1)
        self.assertEqual(summary["review"]["pending"], 2)
        self.assertEqual(summary["localization"]["VERIFIED_BBOX"], 1)
        self.assertEqual(summary["decision"]["BOX_OK"], 1)
        self.assertEqual(summary["origin"]["box_wrong"]["total"], 1)
        self.assertEqual(summary["manual_missing"]["manual_total"], 1)
        self.assertEqual(summary["training_localization_ready"], 1)
        for key in ("gold_modified", "recovery_evidence_modified", "truth_class_changed",
                    "episodes_merged_or_split", "episode_id_changed",
                    "training_tiles_generated", "annotation_complete_performed",
                    "hard_negative_mining_started", "training_started", "detector_run",
                    "sealed_accessed", "auto_verified_without_human"):
            self.assertIs(summary["boundaries"][key], False, key)

    def test_summary_proposal_accounting(self) -> None:
        records = [gold_record(f"ep-{i}", "01021", f"2026-09-19 01:0{i}:00",
                               bbox=[50 + i, 50, 80 + i, 80], label="BOX_WRONG")
                   for i in range(3)]
        data, root = self.build_input(records)
        state = ReviewState.load(root / "s.json", data=data)
        engine = FakeEngine()
        rid = reviewer(data, state, engine)
        rid.generate_proposals("ep-0")
        state.select_proposal(data.by_id["ep-0"], "A")
        rid.generate_proposals("ep-1")
        state.decide_unresolved(data.by_id["ep-1"])
        rows = reviewer(data, state).build_records()
        summary = build_summary(data, rows, state)
        self.assertEqual(summary["proposal"]["episodes_requiring_proposal"], 2)
        self.assertEqual(summary["proposal"]["proposal_first_pass_success"], 1)
        self.assertEqual(summary["proposal"]["proposal_failed"], 1)

    def test_manifest_records_provenance_and_hashes(self) -> None:
        records = [gold_record("ep-1", "01021", "2026-09-19 01:00:00", bbox=[10, 10, 40, 40])]
        data, root = self.build_input(records)
        state = ReviewState.load(root / "s.json", data=data)
        rows = reviewer(data, state).build_records()
        summary = build_summary(data, rows, state)
        loc = root / "localizations.jsonl"
        loc.write_text("{}\n", encoding="utf-8")
        manifest = build_manifest(
            data, summary, code_commit="cafe", generated_at="2026-09-23T00:00:00Z",
            config={"x": 1}, artifact_root=root, localization_path=loc,
            review_state_path=root / "s.json", provenance=build_upstream_provenance(
                step1c0_reported_execution_commit="2e5da41",
                step1c0_manifest_commit="81c3dde",
                step1c0_evidence_commit="bfdecd5", code_equivalence_verified=True))
        self.assertEqual(manifest["gold_input_sha256"], data.gold_sha256)
        self.assertEqual(manifest["code_commit"], "cafe")
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(manifest["upstream_provenance"]["step1c0_reported_execution_commit"],
                         "2e5da41")
        self.assertTrue(manifest["upstream_provenance"]["step1c0_code_equivalence_verified"])
        self.assertIn("source_evidence_sha256".replace("source_", "recovery_"), manifest)


class SafetyTest(Base):
    def test_gold_and_recovery_evidence_are_not_modified(self) -> None:
        records = [gold_record("ep-1", "01021", "2026-09-19 01:00:00", bbox=[10, 10, 40, 40])]
        data, root = self.build_input(records)
        gold_before = sha256_file(data.gold_path)
        evidence_before = sha256_file(root / "recovery" / "episode_source_evidence.jsonl")
        state = ReviewState.load(root / "s.json", data=data)
        rid = reviewer(data, state, FakeEngine())
        rid.generate_proposals("ep-1")
        state.select_proposal(data.required[0], "A")
        rid.build_records()
        self.assertEqual(sha256_file(data.gold_path), gold_before)
        self.assertEqual(sha256_file(root / "recovery" / "episode_source_evidence.jsonl"),
                         evidence_before)

    def test_sealed_paths_are_refused(self) -> None:
        for bad in ("/home/sf01/ground-litter-feasibility/20260923-r1/archive/"
                    "ground-litter-detector-feasibility-20260923-r1/sealed_test/x.jpg",
                    "output/foo/SEALED_DO_NOT_TUNE.txt", "/data/sealed_test"):
            with self.assertRaises(SealedAssetError):
                assert_not_sealed(bad)

    def test_sealed_recovery_root_is_refused(self) -> None:
        record = gold_record("ep-1", "01021", "2026-09-19 01:00:00", bbox=[10, 10, 40, 40])
        data, root = self.build_input([record])
        sealed = root / "sealed_test"
        sealed.mkdir()
        with self.assertRaises(SealedAssetError):
            load_localization_input(data.gold_path, sealed)

    def test_no_training_tiles_are_written(self) -> None:
        records = [gold_record("ep-1", "01021", "2026-09-19 01:00:00", bbox=[10, 10, 40, 40])]
        data, root = self.build_input(records)
        state = ReviewState.load(root / "s.json", data=data)
        state.decide_box_ok(data.required[0])
        reviewer(data, state).build_records()
        names = {p.name.lower() for p in root.rglob("*") if p.is_file()}
        self.assertFalse([n for n in names if "tile" in n or "train" in n])


class ResumeTest(Base):
    def test_state_reload_is_identical_and_ids_stable(self) -> None:
        records = [gold_record("ep-1", "01021", "2026-09-19 01:00:00",
                               bbox=[50, 50, 80, 80], label="BOX_WRONG")]
        data, root = self.build_input(records)
        state = ReviewState.load(root / "s.json", data=data)
        reviewer(data, state, FakeEngine()).generate_proposals("ep-1")
        state.select_proposal(data.required[0], "A")
        first = json.loads((root / "s.json").read_text())

        again = ReviewState.load(root / "s.json", data=data)
        self.assertEqual(again.decisions, state.decisions)
        self.assertEqual(again.audit_trail, state.audit_trail)
        self.assertEqual(again.get("ep-1")["verified_bbox"], [100.0, 100.0, 140.0, 140.0])
        again.save()
        self.assertEqual(json.loads((root / "s.json").read_text())["decisions"],
                         first["decisions"])
        # episode identity is the Gold episode_id and never regenerated
        self.assertEqual(again.get("ep-1")["episode_id"], "ep-1")
        self.assertEqual(again.get("ep-1")["truth_target_id"], "ep-1")

    def test_state_from_another_gold_is_rejected(self) -> None:
        records = [gold_record("ep-1", "01021", "2026-09-19 01:00:00", bbox=[10, 10, 40, 40])]
        data, root = self.build_input(records)
        state = ReviewState.load(root / "s.json", data=data)
        state.input_gold_sha256 = "f" * 64
        state.save()
        with self.assertRaises(ReviewError):
            ReviewState.load(root / "s.json", data=data)

    def test_audit_trail_records_every_action(self) -> None:
        records = [gold_record("ep-1", "01021", "2026-09-19 01:00:00",
                               bbox=[50, 50, 80, 80], label="BOX_WRONG")]
        data, root = self.build_input(records)
        state = ReviewState.load(root / "s.json", data=data)
        reviewer(data, state, FakeEngine()).generate_proposals("ep-1")
        state.select_proposal(data.required[0], "A")
        state.reset("ep-1")
        self.assertEqual([r["action"] for r in state.audit_trail],
                         ["proposals:generated", "decide:PROPOSAL_SELECTED", "reset"])
        self.assertIsNone(state.get("ep-1"))

    def test_progress_counts(self) -> None:
        records = [gold_record(f"ep-{i}", "01021", f"2026-09-19 01:0{i}:00",
                               bbox=[10 + i, 10, 40 + i, 40]) for i in range(3)]
        data, root = self.build_input(records)
        state = ReviewState.load(root / "s.json", data=data)
        self.assertEqual(state.progress(data), {"total": 3, "reviewed": 0, "pending": 3})
        state.decide_box_ok(data.required[0])
        self.assertEqual(state.progress(data), {"total": 3, "reviewed": 1, "pending": 2})


if __name__ == "__main__":
    unittest.main()
