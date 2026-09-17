from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from rtsp_annotator.ground_litter_acceptance import (
    evaluate, make_truth_template, read_logs, render_report,
)
from scripts.evaluate_ground_litter_acceptance import main


CAMERA = "44180209031322001030"
DIGEST = "a" * 64
FINGERPRINTS = [{"sha256": "b" * 64, "bytes": 123}]


def observation(at, visible=(), created=(), cleared=(), *, run="one", generation=None):
    return {"event": "observation", "camera_id": CAMERA, "run_id": run,
            "generation": generation, "source_time_seconds": at,
            "candidates": [], "visible_item_ids": list(visible),
            "created_item_ids": list(created), "cleared_item_ids": list(cleared)}


def truth_fixture():
    truth = make_truth_template({"device_code": CAMERA, "view_id": "fixed-v1",
                                 "day": {"confirm_seconds": 15}}, "day")
    base = datetime(2026, 9, 12, 10, tzinfo=timezone(timedelta(hours=8)))
    truth.update(duration_seconds=240, reviewer="human-reviewer", reviewed_at=base.isoformat(),
                 coverage_complete=True, records_review_complete=True,
                 log_fingerprints=deepcopy(FINGERPRINTS),
                 timeline=[{"run_id": "one", "generation": None, "field": "source_time_seconds",
                            "offset_seconds": 0, "verified": True}])
    truth["source"].update(media_sha256=DIGEST, camera_verified=True, camera_evidence="camera.jpg",
                            anchors=[{"media_seconds": at, "expected_time": (base+timedelta(seconds=at)).isoformat(),
                                      "observed_time": (base+timedelta(seconds=at)).isoformat(),
                                      "evidence": f"anchor-{at}.jpg"} for at in (0, 239)])
    truth["episodes"] = [{"episode_id": "paper-1", "category": "paper", "region_id": "walkway",
                           "box": [.3, .4, .35, .45], "eligible": True,
                           "intervals": [{"start": 0, "end": 40, "state": "visible"},
                                         {"start": 40, "end": 60, "state": "occluded"},
                                         {"start": 60, "end": 100, "state": "visible"},
                                         {"start": 100, "end": 240, "state": "clean"}]}]
    truth["record_reviews"] = [{"item_id": "item-1", "label": "true_litter",
                                 "episode_ids": ["paper-1"], "evidence": "confirmation.jpg"}]
    return truth


class AcceptanceTests(unittest.TestCase):
    def evaluate(self, truth=None, rows=None):
        return evaluate(truth or truth_fixture(), rows if rows is not None else [
            observation(20, ["item-1"], ["item-1"]), observation(75, ["item-1"]),
            observation(170, cleared=["item-1"])], FINGERPRINTS, media_sha256=DIGEST)

    def test_template_never_invents_truth_or_passes(self):
        camera = {"device_code": CAMERA, "view_id": "fixed-v1", "day": {"confirm_seconds": 15}}
        truth = make_truth_template(camera, "day")
        report = self.evaluate(truth)
        self.assertTrue(report["blockers"])
        self.assertTrue(all(value is None for value in report["metrics"].values()))
        self.assertFalse(report["production_go"])
        self.assertIn("无法计算", render_report(report))
        self.assertEqual(truth["episodes"], [])

    def test_complete_appearance_occlusion_cleanup(self):
        report = self.evaluate()
        self.assertEqual(report["blockers"], [])
        self.assertEqual(report["metrics"]["record_precision"]["value"], 1)
        self.assertEqual(report["metrics"]["visible_episode_recall"]["denominator"], 1)
        self.assertEqual(report["metrics"]["clear_recall"]["value"], 1)
        self.assertEqual(report["metrics"]["erroneous_clears"]["count"], 0)
        self.assertIsNone(report["accuracy"])
        self.assertFalse(report["notifications_go"])
        self.assertLess(report["metrics"]["record_precision"]["wilson_95"][0], .5)

    def test_no_detection_of_visible_truth_is_miss_not_clean_scene(self):
        truth = truth_fixture()
        truth["record_reviews"] = []
        report = self.evaluate(truth, [observation(20), observation(80)])
        self.assertEqual(report["metrics"]["visible_episode_recall"]["value"], 0)
        self.assertIsNone(report["metrics"]["record_precision"]["value"])
        self.assertEqual(report["missed_episode_ids"], ["paper-1"])

    def test_candidate_boxes_do_not_count_as_confirmed_true_positives(self):
        truth = truth_fixture()
        truth["record_reviews"] = []
        row = observation(30)
        row["candidates"] = [{"box": [10, 20, 30, 40], "label": "Paper"}] * 20
        report = self.evaluate(truth, [row])
        self.assertEqual(report["descriptive"]["candidate_boxes"], 20)
        self.assertEqual(report["metrics"]["visible_episode_recall"]["value"], 0)

    def test_different_ids_for_same_physical_item_count_as_duplicate(self):
        truth = truth_fixture()
        truth["record_reviews"].append({**truth["record_reviews"][0], "item_id": "item-2"})
        report = self.evaluate(truth, [observation(20, ["item-1"], ["item-1"]),
                                       observation(75, ["item-2"], ["item-2"])])
        self.assertEqual(report["metrics"]["duplicate_records"], 1)
        self.assertEqual(report["metrics"]["visible_episode_recall"]["numerator"], 1)

    def test_restart_with_same_id_and_explicit_clock_mapping(self):
        truth = truth_fixture()
        truth["timeline"].append({"run_id": "two", "generation": 2, "field": "source_time_seconds",
                                   "offset_seconds": 70, "verified": True})
        report = self.evaluate(truth, [observation(20, ["item-1"], ["item-1"]),
                                       observation(5, ["item-1"], run="two", generation=2)])
        self.assertEqual(report["blockers"], [])
        self.assertEqual(report["metrics"]["duplicate_records"], 0)

    def test_unmapped_restart_and_non_increasing_time_are_blocked(self):
        for rows in ([observation(20), observation(1, run="two")],
                     [observation(20), observation(19)]):
            with self.subTest(rows=rows):
                self.assertTrue(self.evaluate(rows=rows)["blockers"])

    def test_occlusion_and_too_short_clean_interval_cannot_justify_clear(self):
        for at in (50, 110):
            with self.subTest(at=at):
                report = self.evaluate(rows=[observation(20, ["item-1"], ["item-1"]),
                                             observation(at, cleared=["item-1"])])
                self.assertEqual(report["metrics"]["erroneous_clears"]["count"], 1)
                self.assertEqual(report["metrics"]["clear_recall"]["value"], 0)

    def test_fully_occluded_truth_is_not_in_recall_denominator(self):
        truth = truth_fixture()
        truth["episodes"][0]["intervals"] = [{"start": 0, "end": 240, "state": "occluded"}]
        truth["record_reviews"] = []
        report = self.evaluate(truth, [observation(20)])
        self.assertEqual(report["metrics"]["visible_episode_recall"]["denominator"], 0)
        self.assertIsNone(report["metrics"]["visible_episode_recall"]["value"])

    def test_clean_then_reappearance_with_new_id_is_two_episodes(self):
        truth = truth_fixture()
        episode = deepcopy(truth["episodes"][0])
        episode.update(episode_id="paper-2", intervals=[{"start": 180, "end": 240, "state": "visible"}])
        truth["episodes"][0]["intervals"][-1]["end"] = 180
        truth["episodes"].append(episode)
        truth["record_reviews"].append({"item_id": "item-2", "label": "true_litter",
                                        "episode_ids": ["paper-2"], "evidence": "new-placement.jpg"})
        report = self.evaluate(truth, [observation(20, ["item-1"], ["item-1"]),
                                       observation(170, cleared=["item-1"]),
                                       observation(210, ["item-2"], ["item-2"])])
        self.assertEqual(report["metrics"]["visible_episode_recall"]["numerator"], 2)
        self.assertEqual(report["metrics"]["duplicate_records"], 0)
        self.assertEqual(report["metrics"]["merged_episodes"], 0)
        truth["record_reviews"] = [truth["record_reviews"][0]]
        truth["record_reviews"][0]["episode_ids"].append("paper-2")
        report = self.evaluate(truth, [observation(20, ["item-1"], ["item-1"]), observation(210, ["item-1"])])
        self.assertEqual(report["metrics"]["merged_episodes"], 1)

    def test_false_facility_record_is_not_a_litter_episode(self):
        truth = truth_fixture()
        truth["episodes"] = []
        truth["record_reviews"][0].update(label="false_positive", episode_ids=[])
        report = self.evaluate(truth)
        self.assertEqual(report["metrics"]["record_precision"]["value"], 0)
        self.assertEqual(report["metrics"]["false_records_per_camera_hour"], 15)
        self.assertIsNone(report["metrics"]["visible_episode_recall"]["value"])

    def test_uncertain_or_missing_reviews_block_metrics(self):
        for reviews in ([], [{"item_id": "item-1", "label": "uncertain", "episode_ids": [], "evidence": "x.jpg"}]):
            truth = truth_fixture()
            truth["record_reviews"] = reviews
            report = self.evaluate(truth)
            self.assertTrue(report["blockers"])
            self.assertIsNone(report["metrics"]["record_precision"])

    def test_wrong_camera_media_log_hash_and_content_clock_are_blocked(self):
        truth = truth_fixture()
        truth["source"]["anchors"][1]["observed_time"] = "2026-09-12T18:03:59+08:00"
        self.assertTrue(self.evaluate(truth)["blockers"])
        self.assertTrue(evaluate(truth_fixture(), [observation(20)], FINGERPRINTS, media_sha256="c"*64)["blockers"])
        self.assertTrue(evaluate(truth_fixture(), [observation(20)], [], media_sha256=DIGEST)["blockers"])
        row = observation(20)
        row["camera_id"] = "44180209031322001021"
        self.assertTrue(self.evaluate(rows=[row])["blockers"])

    def test_malformed_truth_and_conflicting_reviews_fail(self):
        for mutation in ("overlap", "nan", "duplicate", "dangling", "eligible"):
            truth = truth_fixture()
            if mutation == "overlap":
                truth["episodes"][0]["intervals"][1]["start"] = 39
            elif mutation == "nan":
                truth["duration_seconds"] = float("nan")
            elif mutation == "duplicate":
                truth["record_reviews"] *= 2
            elif mutation == "dangling":
                truth["record_reviews"][0]["episode_ids"] = ["absent"]
            else:
                truth["episodes"][0]["eligible"] = None
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.evaluate(truth)

    def test_cli_writes_blocked_report_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            truth = truth_fixture()
            truth["coverage_complete"] = False
            (root / "truth.json").write_text(json.dumps(truth))
            (root / "observations.jsonl").write_text(json.dumps(observation(20)) + "\n")
            args = ["--truth", str(root / "truth.json"), "--logs", str(root / "observations.jsonl"),
                    "--output", str(root / "result")]
            self.assertEqual(main(args), 2)
            self.assertIn("无法计算", (root / "result/report.md").read_text())
            with self.assertRaises(FileExistsError):
                main(args)

    def test_corrupt_json_log_does_not_disappear_from_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "observations.jsonl"
            path.write_text('{"event":')
            with self.assertRaisesRegex(ValueError, "line 1"):
                read_logs([path])


if __name__ == "__main__":
    unittest.main()
