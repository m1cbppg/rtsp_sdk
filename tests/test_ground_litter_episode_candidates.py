"""Step 1A: episode-candidate grouping tests.

These tests intentionally avoid cv2/model dependencies: the grouping logic is
pure-stdlib and must stay testable in the repository virtualenv.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import random
import unittest

from rtsp_annotator.ground_litter_episode_candidates import (
    CleanGap,
    GroupingConfig,
    ReviewCard,
    build_summary,
    cards_fingerprint,
    group_cards,
    group_to_record,
    load_review_batch,
    pair_compatible,
    parse_timestamp,
    select_representative,
)

BASE_TIME = datetime(2026, 9, 20, 10, 0, 0)


def make_card(
    card_id: str = "b:1",
    *,
    camera: str = "01021",
    scene_version: str | None = None,
    seconds: float = 0.0,
    bbox=(100.0, 100.0, 140.0, 140.0),
    label: str = "LITTER",
    batch: str = "b",
    frame_id: str | None = "f00s00",
    source_file_id: str | None = "file-1",
    context_exists: bool = True,
    crop_exists: bool = True,
    has_before_after: bool = True,
    unresolved: tuple[str, ...] = (),
    timestamp: datetime | None = None,
) -> ReviewCard:
    moment = timestamp if timestamp is not None else BASE_TIME + timedelta(seconds=seconds)
    return ReviewCard(
        card_id=card_id,
        raw_review_id=card_id.split(":", 1)[-1],
        batch_key=batch,
        camera_id=camera,
        device_code="44180209031322" + camera,
        label=label,
        scene_version=scene_version,
        timestamp=moment,
        timestamp_text=moment.strftime("%Y-%m-%d %H:%M:%S"),
        bbox=tuple(float(v) for v in bbox) if bbox is not None else None,
        frame_id=frame_id,
        proposal_source="semantic_tile",
        source_file_id=source_file_id,
        context_image="assets/x-context.jpg",
        crop_image="assets/x-crop.jpg",
        before_image="assets/x-before.jpg" if has_before_after else None,
        after_image="assets/x-after.jpg" if has_before_after else None,
        context_image_exists=context_exists,
        crop_image_exists=crop_exists,
        unresolved_reasons=unresolved,
    )


class GroupingTest(unittest.TestCase):
    # ---- required behaviour ------------------------------------------------ #

    def test_adjacent_time_and_near_bbox_merge(self):
        cards = [
            make_card("b:a", seconds=0.0),
            make_card("b:b", seconds=76.0, bbox=(104.0, 104.0, 144.0, 144.0)),
        ]
        groups, unresolved = group_cards(cards)
        self.assertEqual(unresolved, [])
        self.assertEqual(len(groups), 1)
        self.assertEqual({m.card_id for m in groups[0].members}, {"b:a", "b:b"})

    def test_long_time_gap_does_not_merge(self):
        cards = [
            make_card("b:a", seconds=0.0),
            make_card("b:b", seconds=60.0 * 60 * 3),  # 3 h later, same spot
        ]
        groups, _ = group_cards(cards)
        self.assertEqual(len(groups), 2)

    def test_different_camera_never_merges(self):
        cards = [
            make_card("b:a", camera="01021"),
            make_card("b:b", camera="01022"),
        ]
        groups, _ = group_cards(cards)
        self.assertEqual(len(groups), 2)
        self.assertEqual({g.camera_id for g in groups}, {"01021", "01022"})

    def test_different_scene_version_never_merges(self):
        cards = [
            make_card("b:a", scene_version="v1"),
            make_card("b:b", scene_version="v2"),
        ]
        groups, _ = group_cards(cards)
        self.assertEqual(len(groups), 2)

    def test_clean_gap_evidence_prevents_merge(self):
        cards = [
            make_card("b:a", seconds=0.0),
            make_card("b:b", seconds=200.0),  # well inside the 600 s window
        ]
        # Without independent clean-gap evidence these merge.
        self.assertEqual(len(group_cards(cards)[0]), 1)
        gap = CleanGap("01021", BASE_TIME + timedelta(seconds=50),
                       BASE_TIME + timedelta(seconds=150), "human_clean_confirmation")
        groups, _ = group_cards(cards, GroupingConfig(), [gap])
        self.assertEqual(len(groups), 2)

    def test_clean_gap_for_other_camera_does_not_block(self):
        cards = [
            make_card("b:a", seconds=0.0, camera="01021"),
            make_card("b:b", seconds=200.0, camera="01021"),
        ]
        gap = CleanGap("01022", BASE_TIME, BASE_TIME + timedelta(seconds=400), "x")
        self.assertEqual(len(group_cards(cards, GroupingConfig(), [gap])[0]), 1)

    def test_missing_timestamp_or_bbox_is_unresolved_not_a_crash(self):
        cards = [
            make_card("b:a", seconds=0.0),
            make_card("b:no_ts", seconds=100.0, timestamp=None, unresolved=("missing_timestamp",)),
            make_card("b:no_box", seconds=100.0, bbox=None, unresolved=("missing_bbox",)),
        ]
        groups, unresolved = group_cards(cards)
        self.assertEqual(len(unresolved), 2)
        by_id = {g.episode_candidate_id: g for g in groups}
        self.assertEqual(len(by_id), 3)
        fallback = [g for g in groups if g.grouping_method == "unresolved_fallback"]
        self.assertEqual(len(fallback), 2)
        for group in fallback:
            self.assertTrue(group.is_ambiguous)
            self.assertTrue(any(r.startswith("unresolved_member:")
                                for r in group.ambiguity_reasons))

    def test_repeated_runs_are_stable_even_with_shuffled_input(self):
        cards = [
            make_card("b:a", seconds=0.0),
            make_card("b:b", seconds=76.0, bbox=(104.0, 104.0, 144.0, 144.0)),
            make_card("b:c", seconds=60.0 * 60 * 5),
            make_card("b:d", camera="01030", seconds=0.0),
        ]
        first, _ = group_cards(cards)
        shuffled = list(cards)
        random.Random(7).shuffle(shuffled)
        second, _ = group_cards(shuffled)
        signature = lambda gs: [(g.episode_candidate_id, sorted(m.card_id for m in g.members))
                                for g in gs]
        self.assertEqual(signature(first), signature(second))

    # ---- algorithm safety -------------------------------------------------- #

    def test_complete_linkage_prevents_cross_day_chaining(self):
        # A-B and B-C are each compatible, A-C is not.  Single linkage would fuse
        # all three; complete linkage must stop it (false-merge protection).
        cards = [
            make_card("b:a", seconds=0.0, bbox=(0.0, 0.0, 28.0, 28.0)),
            make_card("b:b", seconds=100.0, bbox=(25.0, 0.0, 53.0, 28.0)),
            make_card("b:c", seconds=200.0, bbox=(50.0, 0.0, 78.0, 28.0)),
        ]
        config = GroupingConfig()
        self.assertTrue(pair_compatible(cards[0], cards[1], config))
        self.assertTrue(pair_compatible(cards[1], cards[2], config))
        self.assertFalse(pair_compatible(cards[0], cards[2], config))
        groups, _ = group_cards(cards, config)
        self.assertEqual(len(groups), 2)
        # The split is reported as a possible false split rather than hidden.
        flagged = [g for g in groups if any("false_split" in r for r in g.ambiguity_reasons)]
        self.assertEqual(len(flagged), 2)

    def test_size_mismatch_does_not_merge(self):
        cards = [
            make_card("b:small", seconds=0.0, bbox=(100.0, 100.0, 106.0, 106.0)),
            make_card("b:large", seconds=10.0, bbox=(98.0, 98.0, 298.0, 298.0)),
        ]
        self.assertEqual(len(group_cards(cards)[0]), 2)

    def test_only_positive_labels_are_grouped(self):
        cards = [
            make_card("b:litter", label="LITTER"),
            make_card("b:non", label="NON_LITTER"),
            make_card("b:unc", label="UNCERTAIN"),
            make_card("b:bad", label="BOX_WRONG", seconds=5.0, bbox=(101.0, 101.0, 141.0, 141.0)),
        ]
        groups, _ = group_cards(cards)
        members = {m.card_id for g in groups for m in g.members}
        self.assertEqual(members, {"b:litter", "b:bad"})
        self.assertEqual(len(groups), 1)
        self.assertEqual(members, {"b:litter", "b:bad"})

    def test_no_before_after_narrows_observed_interval(self):
        # Without before/after frames there is no +/-2 s context to bridge the gap.
        cards = [
            make_card("b:a", seconds=0.0, has_before_after=False),
            make_card("b:b", seconds=602.0, has_before_after=False),
        ]
        self.assertEqual(len(group_cards(cards)[0]), 2)
        bridged = [
            make_card("b:a", seconds=0.0),
            make_card("b:b", seconds=602.0),
        ]
        self.assertEqual(len(group_cards(bridged)[0]), 1)

    # ---- representative + evidence ----------------------------------------- #

    def test_representative_prefers_valid_images_then_area_then_median_time(self):
        group_cards_input = [
            make_card("b:broken", seconds=0.0, context_exists=False, crop_exists=False),
            make_card("b:small", seconds=76.0, bbox=(100.0, 100.0, 140.0, 140.0)),
            make_card("b:big", seconds=152.0, bbox=(100.0, 100.0, 180.0, 180.0)),
        ]
        groups, _ = group_cards(group_cards_input)
        self.assertEqual(len(groups), 1)
        representative = select_representative(groups[0])
        self.assertEqual(representative.card_id, "b:big")
        record = group_to_record(groups[0], GroupingConfig(),
                                 {c.card_id: c for c in group_cards_input})
        self.assertFalse(record["representative_detector_score_used"])
        self.assertIsNone(record["episode_id"])
        self.assertEqual(record["episode_id_status"], "NOT_ASSIGNED_CANDIDATE_ONLY")

    def test_group_record_preserves_full_lineage_and_label_summary(self):
        cards = [
            make_card("b:a", seconds=0.0, source_file_id="ps-1"),
            make_card("b:b", seconds=76.0, label="BOX_WRONG", source_file_id="ps-2",
                      bbox=(102.0, 102.0, 142.0, 142.0)),
        ]
        groups, _ = group_cards(cards)
        record = group_to_record(groups[0], GroupingConfig(), {c.card_id: c for c in cards})
        self.assertEqual(record["member_count"], 2)
        self.assertEqual(sorted(record["source_file_ids"]), ["ps-1", "ps-2"])
        self.assertEqual(record["original_labels_summary"]["LITTER"], 1)
        self.assertEqual(record["original_labels_summary"]["BOX_WRONG"], 1)
        self.assertEqual(record["original_labels_summary"]["total"], 2)
        lineage_cards = record["lineage"]["review_cards"]
        self.assertEqual(len(lineage_cards), 2)
        for entry in lineage_cards:
            self.assertIn("assets", entry)
            self.assertIn("card_id", entry)
        payload = json.dumps(record)
        # no signed download URL or credential may ever be persisted
        for marker in ("http://", "https://", "token=", "signature="):
            self.assertNotIn(marker, payload)
        self.assertNotIn("signed_url", payload)

    def test_reported_spread_and_limit_are_comparable(self):
        # Regression: the reported centre spread and the reported limit must come
        # from the *binding* pair, otherwise a reviewer sees spread > limit even
        # though complete linkage guarantees every pair is within its own limit.
        cards = [
            make_card("b:a", seconds=0.0, bbox=(0.0, 0.0, 28.0, 28.0)),
            make_card("b:b", seconds=100.0, bbox=(25.0, 0.0, 53.0, 28.0)),
        ]
        groups, _ = group_cards(cards)
        self.assertEqual(len(groups), 1)
        record = group_to_record(groups[0], GroupingConfig(), {c.card_id: c for c in cards})
        evidence = record["candidate_grouping_evidence"]
        self.assertTrue(evidence["complete_linkage_pressure_within_limits"])
        self.assertLessEqual(evidence["binding_pair_threshold_pressure"], 1.0)
        self.assertLessEqual(evidence["binding_pair_center_distance_px"],
                             evidence["center_spread_limit_px"])

    def test_lineage_absence_is_flagged_ambiguous(self):
        cards = [make_card("b:a", seconds=0.0, source_file_id=None)]
        groups, _ = group_cards(cards)
        record = group_to_record(groups[0], GroupingConfig(), {c.card_id: c for c in cards})
        self.assertTrue(record["candidate_ambiguous"])
        self.assertIn("lineage_source_file_id_unavailable", record["ambiguity_reasons"])
        self.assertEqual(record["source_file_ids_status"],
                         "unavailable_in_silver_review_cards")

    def test_deterministic_candidate_ids_and_zero_duplicate_membership(self):
        cards = [
            make_card("b:a", seconds=0.0),
            make_card("b:b", seconds=76.0, bbox=(104.0, 104.0, 144.0, 144.0)),
            make_card("b:c", seconds=76.0, bbox=(600.0, 600.0, 640.0, 640.0)),
        ]
        groups, _ = group_cards(cards)
        ids = [g.episode_candidate_id for g in groups]
        self.assertEqual(len(ids), len(set(ids)))
        members = [m.card_id for g in groups for m in g.members]
        self.assertEqual(sorted(members), ["b:a", "b:b", "b:c"])
        # a card belongs to exactly one candidate
        self.assertEqual(len(members), len(set(members)))

    def test_summary_reports_candidate_count_as_estimate_only(self):
        cards = [
            make_card("b:a", seconds=0.0),
            make_card("b:b", seconds=76.0, bbox=(104.0, 104.0, 144.0, 144.0)),
            make_card("b:c", seconds=60.0 * 60 * 4),
        ]
        groups, unresolved = group_cards(cards)
        summary = build_summary(groups, [], cards, config=GroupingConfig(),
                                unresolved=unresolved, batch_diagnostics=[],
                                clean_gap_evidence_used=False)
        self.assertEqual(summary["input"]["LITTER_cards"], 3)
        self.assertEqual(summary["output"]["candidate_group_count"], 2)
        self.assertEqual(summary["output"]["singleton_group_count"], 1)
        self.assertEqual(summary["output"]["multi_card_group_count"], 1)
        self.assertEqual(summary["output"]["max_group_size"], 2)
        self.assertIsNone(summary["interpretation"]["gold_episode_count"])
        self.assertEqual(summary["interpretation"]["candidate_independent_events_estimate"], 2)
        self.assertIn("NOT the true Gold", summary["interpretation"]["statement"])

    def test_cards_fingerprint_changes_with_input(self):
        a = [make_card("b:a", seconds=0.0)]
        b = [make_card("b:a", seconds=0.0), make_card("b:b", seconds=5.0)]
        self.assertEqual(cards_fingerprint(a), cards_fingerprint(a))
        self.assertNotEqual(cards_fingerprint(a), cards_fingerprint(b))


class ConfigTest(unittest.TestCase):
    def test_config_round_trip(self):
        config = GroupingConfig(max_time_gap_seconds=120.0)
        self.assertEqual(GroupingConfig.from_dict(config.as_dict()), config)

    def test_config_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            GroupingConfig.from_dict({"nope": 1})

    def test_config_rejects_non_positive_values(self):
        with self.assertRaises(ValueError):
            GroupingConfig(max_time_gap_seconds=0)
        with self.assertRaises(ValueError):
            GroupingConfig(max_size_ratio=0.5)


class LoaderTest(unittest.TestCase):
    def _write_batch(self, root: Path, items, labels) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "review-data.json").write_text(json.dumps({
            "dataset": "unit-test", "fingerprint": "fp", "count": len(items),
            "items": items}), encoding="utf-8")
        (root / "reviews.json").write_text(json.dumps({
            "dataset": "unit-test", "fingerprint": "fp",
            "reviews": {k: {"label": v} for k, v in labels.items()}}), encoding="utf-8")

    def test_loader_batch_qualifies_card_ids_and_maps_labels(self):
        root = Path(self._tmp())
        items = [
            {"review_id": "01021-f00s00-0001", "device_code": "44180209031322001021",
             "timestamp": "2026-09-20 10:00:00", "bbox": [10, 10, 50, 50],
             "frame_id": "f00s00", "context_image": "assets/a-context.jpg",
             "crop_image": "assets/a-crop.jpg", "file_id": "ps-1"},
            {"review_id": "01021-f00s00-0002", "device_code": "44180209031322001021",
             "timestamp": "2026-09-20 10:01:16", "bbox": [12, 12, 52, 52],
             "frame_id": "f00s00", "context_image": "assets/b-context.jpg",
             "crop_image": "assets/b-crop.jpg"},
        ]
        self._write_batch(root, items, {"01021-f00s00-0001": "LITTER",
                                        "01021-f00s00-0002": "BOX_WRONG"})
        cards, diag = load_review_batch("batchx", root)
        self.assertEqual(diag["raw_card_count"], 2)
        self.assertEqual([c.card_id for c in cards],
                         ["batchx:01021-f00s00-0001", "batchx:01021-f00s00-0002"])
        self.assertEqual([c.label for c in cards], ["LITTER", "BOX_WRONG"])
        self.assertEqual([c.camera_id for c in cards], ["01021", "01021"])
        self.assertEqual(cards[0].source_file_id, "ps-1")
        self.assertIsNone(cards[1].source_file_id)
        # image paths do not exist in the temp dir; the flag must be honest
        self.assertFalse(cards[0].context_image_exists)

    def test_same_review_id_in_two_batches_does_not_collide(self):
        left = Path(self._tmp()); right = Path(self._tmp())
        item = {"review_id": "01021-f00s00-0001", "device_code": "44180209031322001021",
                "timestamp": "2026-09-20 10:00:00", "bbox": [10, 10, 50, 50]}
        other = dict(item, bbox=[900, 900, 940, 940])
        self._write_batch(left, [item], {"01021-f00s00-0001": "LITTER"})
        self._write_batch(right, [other], {"01021-f00s00-0001": "LITTER"})
        a, _ = load_review_batch("day2", left)
        b, _ = load_review_batch("day1", right)
        cards = a + b
        self.assertEqual(len({c.card_id for c in cards}), 2)
        groups, _ = group_cards(cards)
        # identical ids, very different positions -> still two candidates
        self.assertEqual(len(groups), 2)

    def test_loader_rejects_broken_manifest(self):
        root = Path(self._tmp())
        root.mkdir(parents=True, exist_ok=True)
        (root / "review-data.json").write_text(json.dumps({"items": []}), encoding="utf-8")
        (root / "reviews.json").write_text(json.dumps({"reviews": []}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_review_batch("bad", root)

    def test_parse_timestamp_accepts_both_shapes_and_rejects_junk(self):
        self.assertEqual(parse_timestamp("2026-09-20 10:00:00"),
                         datetime(2026, 9, 20, 10, 0, 0))
        self.assertEqual(parse_timestamp("2026-09-20 10:00:00.500000"),
                         datetime(2026, 9, 20, 10, 0, 0, 500000))
        self.assertIsNone(parse_timestamp("not-a-time"))
        self.assertIsNone(parse_timestamp(""))

    def _tmp(self) -> str:
        import shutil
        import tempfile
        path = tempfile.mkdtemp(prefix="step1a-test-")
        self.addCleanup(shutil.rmtree, path, True)
        return path


if __name__ == "__main__":
    unittest.main()
