"""Step 2C-1: Development unlock + frozen Blind Truth (detector-free).

Covers the §32 list: no detector import or inference, no checkpoint load, Sealed rejected,
Development allowed, truth-class semantics, episode continuity, deterministic 5 s / 30 s
sampling, source-frame geometry, point truth, unresolved localization preserved, review
resume, freeze immutability and post-freeze erratum.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from rtsp_annotator.ground_litter_blind_truth import (
    DEVELOPMENT_MARKER,
    EPISODE_CONFIRMED,
    EVIDENCE_DIRECTIONAL,
    EVIDENCE_EXPLORATORY,
    EVIDENCE_STRONGER,
    GLOBAL_GRID_SECONDS,
    IGNORE_SMALL,
    LOCALIZATION_BOX_OK,
    LOCALIZATION_POINT,
    LOCALIZATION_PROPOSAL,
    LOCALIZATION_UNRESOLVED,
    MAX_VISIBLE_FRAMES_PER_EPISODE,
    NON_LITTER,
    REQUIRED_LITTER,
    REVIEW_DONE,
    SAMPLE_REASON_GLOBAL,
    SAMPLE_REASON_VISIBLE,
    TRUTH_CLASSES,
    UNCERTAIN,
    VISIBLE_GRID_SECONDS,
    BlindTruthError,
    DevelopmentScopeError,
    SealedAssetError,
    TruthFrozenError,
    append_truth_erratum,
    assert_development_asset,
    assert_episode_consistency,
    assert_not_frozen,
    assert_not_sealed,
    assert_review_complete,
    build_summary,
    evidence_classification,
    format_ts,
    freeze_truth,
    in_roi,
    is_frozen,
    load_development_inventory,
    localization_state,
    new_episode,
    new_truth_object,
    next_id,
    parse_ts,
    point_in_polygon,
    refresh_episode,
    review_coverage,
    same_episode_candidate,
    sample_global_roi_frames,
    sample_visible_frames,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "serve_ground_litter_blind_truth.py"
MODULE = ROOT / "rtsp_annotator" / "ground_litter_blind_truth.py"
UI_DIR = ROOT / "tools" / "ground_litter_blind_truth_ui"
BUNDLE_DIRNAME = "ground-litter-detector-feasibility-20260923-r1"
CANVAS = [2560, 1440]
ROI = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
MANIFEST_NAME = "development_manifest.json"


def load_cli():
    spec = importlib.util.spec_from_file_location("serve_blind_truth", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI_MODULE = load_cli()


# --------------------------------------------------------------------------- #
# synthetic Development bundle
# --------------------------------------------------------------------------- #


def build_fixture(root: Path, *, cameras=("01021", "01022"), per_camera=2,
                  split="development") -> tuple[Path, Path]:
    """A miniature Step 0A-shaped Development bundle plus matching ROI configs."""
    dev = root / BUNDLE_DIRNAME / split
    roi_dir = root / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        config = {
            "kind": "ground_litter_camera_geometry", "schema_version": 1,
            "geometry_version": "final-roi-20260922-user-reviewed",
            "camera_id": camera, "device_code": f"44180209031322{camera}",
            "roi": ROI, "exclude_zones": [], "canvas_size": CANVAS,
            "frame_path": f"frames/{camera}.png", "frame_sha256": f"ref-{camera}",
            "purpose": "ground_litter_final_roi",
            "saved_at": "2026-09-22T00:00:00Z",
        }
        (roi_dir / f"ground_litter_{camera}_final_roi.json").write_text(
            json.dumps(config), encoding="utf-8")
        for index in range(per_camera):
            start = f"2026-09-22 15:{index * 10:02d}:00"
            end_minute = index * 10 + 5
            end = f"2026-09-22 15:{end_minute:02d}:00"
            file_id = f"fid-{camera}-{index}"
            ps = dev / camera / "raw" / f"{file_id}__chunk{index}.ps"
            ps.parent.mkdir(parents=True, exist_ok=True)
            ps.write_bytes(b"PSDATA" + bytes([index]) * 32)
            identity = {
                "schema": "ground_litter_feasibility_raw_ps_identity_v1",
                "experiment_id": BUNDLE_DIRNAME, "split": split, "camera_id": camera,
                "device_code": f"44180209031322{camera}",
                "scene_version": f"{camera}-final-roi-20260922-abc123",
                "roi": {"config_path": str(roi_dir / f"ground_litter_{camera}_final_roi.json"),
                        "config_blob_sha": "blob", "geometry_version":
                        "final-roi-20260922-user-reviewed",
                        "reference_frame_sha256": f"ref-{camera}",
                        "canvas_size": CANVAS},
                "file_id": file_id, "file_name": f"chunk{index}.ps",
                "record_start": start, "record_end": end,
                "declared_bytes": ps.stat().st_size, "actual_bytes": ps.stat().st_size,
                "sha256": sha256_file(ps), "signed_url_persisted": False,
                "archive_relative_path": f"{BUNDLE_DIRNAME}/{split}/{camera}/raw/{ps.name}",
                "materialized_at_utc": "2026-09-23T02:00:00Z",
            }
            (ps.parent / (ps.name + ".identity.json")).write_text(
                json.dumps(identity), encoding="utf-8")
    return dev, roi_dir


class FixtureCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.dev, self.roi_dir = build_fixture(self.root)
        self.inventory = load_development_inventory(self.dev, self.roi_dir)
        self.files = {row["file_id"]: row for row in self.inventory["files"]}

    def observation(self, *, camera="01021", index=0, truth_class=REQUIRED_LITTER,
                    point=(1280.0, 900.0), timestamp=None, truth_id="t-0001",
                    bbox=None, status=None):
        record = self.files[f"fid-{camera}-{index}"]
        return new_truth_object(
            truth_id=truth_id, camera_id=camera, scene_version=record["scene_version"],
            source_file_id=record["file_id"],
            timestamp=timestamp or "2026-09-22 15:01:00",
            decoded_timestamp=60.0, frame_index=1500, truth_class=truth_class,
            source_width=CANVAS[0], source_height=CANVAS[1],
            source_point=point, source_bbox_xyxy=bbox, roi=ROI,
            localization_status=status)


# --------------------------------------------------------------------------- #
# §32: blindness
# --------------------------------------------------------------------------- #


class TestBlindness(unittest.TestCase):
    def test_logic_module_imports_no_detector_stack(self):
        for name in ("torch", "ultralytics", "tensorrt", "cv2", "numpy"):
            self.assertNotIn(name, {
                line.split()[1].split(".")[0]
                for line in MODULE.read_text(encoding="utf-8").splitlines()
                if line.startswith("import ") or line.startswith("from ")
            }, name)

    def test_cli_selftest_reports_no_detector(self):
        report = CLI_MODULE.blindness_selftest()
        self.assertTrue(report["ok"], report)
        self.assertFalse(report["detector_loaded"])
        self.assertFalse(report["checkpoint_accessed"])
        self.assertFalse(report["inference_executed"])
        self.assertEqual(report["forbidden_source_tokens"], [])

    def test_no_checkpoint_is_reachable(self):
        source = CLI.read_text(encoding="utf-8")
        self.assertNotIn("YOLO" "(", source)
        self.assertNotIn("state" "_dict", source)
        self.assertNotIn("auto" "cast", source)
        # the only video access is raw decoding, and there is no weight-file argument
        self.assertIn("cv2.VideoCapture", source)
        self.assertNotIn('"--weight"', source)
        self.assertNotIn("load(", source.replace("download(", ""))
        # a .pt argument is a hard failure
        original = sys.argv
        try:
            sys.argv = ["x", "--weight", "last.pt"]
            report = CLI_MODULE.blindness_selftest()
        finally:
            sys.argv = original
        self.assertFalse(report["ok"])
        self.assertTrue(report["checkpoint_accessed"])

    def test_ui_contains_the_blind_banner_and_no_model_output(self):
        html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn("BLIND TRUTH MODE", html)
        self.assertIn("DETECTOR OUTPUT DISABLED", html)
        script = (UI_DIR / "app.js").read_text(encoding="utf-8")
        for forbidden in ("confidence", "score", "yolo", "detector(", "predict("):
            self.assertNotIn(forbidden, script.lower(), forbidden)
        self.assertIn("/api/crop", script)
        self.assertIn("/api/truth", script)


# --------------------------------------------------------------------------- #
# §32: Development allowed, Sealed rejected
# --------------------------------------------------------------------------- #


class TestSplitBoundary(FixtureCase):
    def test_development_path_is_accepted(self):
        assert_development_asset(self.dev)
        assert_development_asset(self.dev / "01021" / "raw" / "fid-01021-0__chunk0.ps")
        self.assertIn(DEVELOPMENT_MARKER, str(self.dev))

    def test_sealed_paths_hard_fail(self):
        for candidate in ("/data/sealed_test/raw/a.ps",
                          "/x/" + BUNDLE_DIRNAME + "/sealed_test/raw/a.ps",
                          "/x/SEALED_DO_NOT_TUNE/a.ps",
                          "/x/sealed-test/a.ps"):
            with self.assertRaises(SealedAssetError):
                assert_development_asset(candidate)

    def test_paths_outside_development_are_refused(self):
        with self.assertRaises(DevelopmentScopeError):
            assert_development_asset("/tmp/somewhere/else")
        with self.assertRaises(DevelopmentScopeError):
            assert_development_asset("")

    def test_legacy_sealed_guard_still_works(self):
        with self.assertRaises(SealedAssetError):
            assert_not_sealed("/bundle/ground-litter-feasibility/20260923-r1/x")

    def test_inventory_reads_only_development(self):
        self.assertTrue(self.inventory["ok"], self.inventory["problems"])
        self.assertEqual(self.inventory["split"], "development")
        self.assertFalse(self.inventory["sealed_accessed"])
        self.assertEqual(self.inventory["ps_count"], 4)
        self.assertEqual(self.inventory["per_camera_count"], {"01021": 2, "01022": 2})
        for row in self.inventory["files"]:
            self.assertEqual(row["canvas_size"], CANVAS)
            self.assertTrue(row["roi_available"])
            self.assertEqual(row["roi"], ROI)
            self.assertEqual(row["duration_seconds"], 300.0)

    def test_a_sealed_identity_inside_the_tree_is_a_hard_failure(self):
        sealed = self.root / BUNDLE_DIRNAME / "sealed_test" / "01021" / "raw"
        sealed.mkdir(parents=True)
        (sealed / "x.ps").write_bytes(b"x")
        (sealed / "x.ps.identity.json").write_text(json.dumps({
            "schema": "ground_litter_feasibility_raw_ps_identity_v1", "split": "sealed_test",
            "camera_id": "01021", "scene_version": "s", "file_id": "x",
            "record_start": "2026-09-22 15:00:00", "record_end": "2026-09-22 15:05:00",
            "declared_bytes": 1, "actual_bytes": 1, "sha256": "0" * 64,
            "roi": {"geometry_version": "g", "reference_frame_sha256": "r",
                    "canvas_size": CANVAS}}), encoding="utf-8")
        with self.assertRaises(SealedAssetError):
            load_development_inventory(self.root / BUNDLE_DIRNAME / "sealed_test",
                                       self.roi_dir)

    def test_roi_mismatch_is_reported(self):
        sidecar = sorted((self.dev / "01021" / "raw").glob("*.identity.json"))[0]
        payload = json.loads(sidecar.read_text())
        payload["roi"]["geometry_version"] = "something-else"
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        inventory = load_development_inventory(self.dev, self.roi_dir)
        self.assertFalse(inventory["ok"])
        self.assertIn("roi_geometry_version_mismatch",
                      [problem["field"] for problem in inventory["problems"]])


# --------------------------------------------------------------------------- #
# §32: truth classes and geometry
# --------------------------------------------------------------------------- #


class TestTruthObjects(FixtureCase):
    def test_point_truth_is_supported_and_in_source_coordinates(self):
        row = self.observation(point=(1280.0, 900.0))
        self.assertEqual(row["source_point"], [1280.0, 900.0])
        self.assertEqual(row["localization_status"], LOCALIZATION_POINT)
        self.assertEqual(row["source_width"], 2560)
        self.assertEqual(row["source_height"], 1440)
        self.assertIsNone(row["source_bbox_xyxy"])
        self.assertTrue(row["in_roi"])                     # 0.5, 0.625 inside the ROI

    def test_point_outside_the_roi_is_flagged_and_excluded_from_the_denominator(self):
        row = self.observation(point=(100.0, 100.0))
        self.assertFalse(row["in_roi"])

    def test_bbox_must_be_source_frame_xyxy(self):
        row = self.observation(bbox=[1000.0, 800.0, 1100.0, 900.0],
                               status=LOCALIZATION_BOX_OK)
        self.assertEqual(row["source_bbox_xyxy"], [1000.0, 800.0, 1100.0, 900.0])
        self.assertEqual(row["localization_status"], LOCALIZATION_BOX_OK)
        for bad in ([1100.0, 800.0, 1000.0, 900.0],      # x2 < x1
                    [-1.0, 0.0, 10.0, 10.0],             # outside the frame
                    [0.0, 0.0, 4000.0, 100.0]):
            with self.assertRaises(BlindTruthError):
                self.observation(bbox=bad, status=LOCALIZATION_BOX_OK)

    def test_proposal_status_requires_a_bbox(self):
        with self.assertRaises(BlindTruthError):
            self.observation(status=LOCALIZATION_PROPOSAL)
        row = self.observation(bbox=[10.0, 10.0, 60.0, 60.0],
                               status=LOCALIZATION_PROPOSAL)
        self.assertEqual(row["localization_status"], LOCALIZATION_PROPOSAL)

    def test_unknown_class_or_status_is_refused(self):
        with self.assertRaises(BlindTruthError):
            self.observation(truth_class="SOMETHING")
        with self.assertRaises(BlindTruthError):
            self.observation(status="WHATEVER")

    def test_only_required_enters_the_denominator(self):
        self.assertTrue(self.observation(truth_class=REQUIRED_LITTER)
                        ["enters_recall_denominator"])
        for truth_class in (IGNORE_SMALL, UNCERTAIN, NON_LITTER):
            row = self.observation(truth_class=truth_class)
            self.assertFalse(row["enters_recall_denominator"], truth_class)
        for truth_class in (IGNORE_SMALL, UNCERTAIN):
            self.assertTrue(self.observation(truth_class=truth_class)["enters_ignore_set"])
        self.assertFalse(self.observation(truth_class=NON_LITTER)["enters_ignore_set"])

    def test_all_step0b_classes_are_available(self):
        self.assertEqual(set(TRUTH_CLASSES),
                         {REQUIRED_LITTER, IGNORE_SMALL, UNCERTAIN, NON_LITTER})

    def test_next_id_is_sequential(self):
        self.assertEqual(next_id("t", []), "t-0001")
        self.assertEqual(next_id("t", ["t-0001", "t-0007", "x"]), "t-0008")

    def test_roi_geometry(self):
        self.assertTrue(point_in_polygon(0.5, 0.5, ROI))
        self.assertFalse(point_in_polygon(0.1, 0.5, ROI))
        self.assertTrue(in_roi([1280, 720], ROI, 2560, 1440))
        self.assertFalse(in_roi([10, 720], ROI, 2560, 1440))


# --------------------------------------------------------------------------- #
# §32: episodes
# --------------------------------------------------------------------------- #


class TestEpisodes(FixtureCase):
    def test_episode_min_max_and_files_come_from_observations(self):
        first = self.observation(timestamp="2026-09-22 15:01:00", truth_id="t-0001")
        second = self.observation(timestamp="2026-09-22 15:03:30", truth_id="t-0002",
                                  point=(1300.0, 905.0))
        episode = new_episode(episode_id="ep-0001", observation=first)
        first["episode_id"] = "ep-0001"
        second["episode_id"] = "ep-0001"
        updated = refresh_episode(episode, [first, second])
        self.assertEqual(updated["first_confirmable_timestamp"], "2026-09-22 15:01:00")
        self.assertEqual(updated["last_confirmable_timestamp"], "2026-09-22 15:03:30")
        self.assertEqual(updated["observation_count"], 2)
        self.assertEqual(updated["source_file_ids"], ["fid-01021-0"])
        assert_episode_consistency(updated, [first, second])

    def test_episode_cannot_mix_cameras(self):
        first = self.observation(camera="01021", truth_id="t-0001")
        other = self.observation(camera="01022", truth_id="t-0002")
        episode = new_episode(episode_id="ep-0001", observation=first)
        first["episode_id"] = "ep-0001"
        other["episode_id"] = "ep-0001"
        with self.assertRaises(BlindTruthError):
            assert_episode_consistency(episode, [first, other])

    def test_inverted_interval_is_refused(self):
        row = self.observation()
        episode = new_episode(episode_id="ep-0001", observation=row)
        episode["first_confirmable_timestamp"] = "2026-09-22 15:10:00"
        episode["last_confirmable_timestamp"] = "2026-09-22 15:00:00"
        with self.assertRaises(BlindTruthError):
            assert_episode_consistency(episode, [])

    def test_continuity_suggestion_uses_only_camera_scene_time_and_distance(self):
        a = self.observation(timestamp="2026-09-22 15:01:00", point=(1280.0, 900.0))
        near = self.observation(timestamp="2026-09-22 15:02:00", point=(1300.0, 910.0),
                                truth_id="t-0002")
        far = self.observation(timestamp="2026-09-22 15:02:00", point=(300.0, 200.0),
                               truth_id="t-0003")
        late = self.observation(timestamp="2026-09-22 18:00:00", point=(1280.0, 900.0),
                                truth_id="t-0004")
        other = self.observation(camera="01022", timestamp="2026-09-22 15:02:00",
                                 point=(1280.0, 900.0), truth_id="t-0005")
        self.assertTrue(same_episode_candidate(a, near)["suggest_same_episode"])
        self.assertFalse(same_episode_candidate(a, far)["suggest_same_episode"])
        self.assertFalse(same_episode_candidate(a, late)["suggest_same_episode"])
        self.assertFalse(same_episode_candidate(a, other)["suggest_same_episode"])
        report = same_episode_candidate(a, near)
        self.assertIn("camera+scene_version", report["criterion"])
        for forbidden in ("score", "confidence", "detector", "model"):
            self.assertNotIn(forbidden, report["criterion"].lower(), forbidden)


# --------------------------------------------------------------------------- #
# §32: deterministic sampling
# --------------------------------------------------------------------------- #


class TestSampling(FixtureCase):
    def _confirmed_episode(self):
        rows = []
        for index, (stamp, point) in enumerate((
                ("2026-09-22 15:00:10", (1280.0, 900.0)),
                ("2026-09-22 15:04:40", (1285.0, 902.0)))):
            rows.append(self.observation(timestamp=stamp, point=point,
                                         truth_id=f"t-{index + 1:04d}",
                                         index=0))
        episode = new_episode(episode_id="ep-0001", observation=rows[0],
                              review_status=EPISODE_CONFIRMED)
        for row in rows:
            row["episode_id"] = "ep-0001"
        return [refresh_episode(episode, rows)], rows

    def test_five_second_grid_is_deterministic_and_capped(self):
        episodes, rows = self._confirmed_episode()
        first = sample_visible_frames(episodes, rows, self.inventory["files"])
        second = sample_visible_frames(episodes, rows, self.inventory["files"])
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), MAX_VISIBLE_FRAMES_PER_EPISODE)
        self.assertEqual(len(first), 5)          # 15:00:10 .. 15:04:40 spans 5 grid points
        stamps = [row["timestamp"] for row in first]
        self.assertEqual(stamps[0], "2026-09-22 15:00:10.000")
        self.assertEqual(stamps[1], "2026-09-22 15:00:15.000")
        for row in first:
            self.assertEqual(row["sample_reason"], SAMPLE_REASON_VISIBLE)
            self.assertEqual(row["grid_seconds"], VISIBLE_GRID_SECONDS)
            self.assertEqual(row["truth_class"], REQUIRED_LITTER)
            self.assertEqual(row["source_width"], 2560)
            self.assertEqual(row["source_height"], 1440)
            self.assertTrue(row["source_file_id"] in self.files)

    def test_short_episode_still_gets_one_frame(self):
        row = self.observation(timestamp="2026-09-22 15:02:00")
        episode = new_episode(episode_id="ep-0009", observation=row,
                              review_status=EPISODE_CONFIRMED)
        row["episode_id"] = "ep-0009"
        rows = sample_visible_frames([episode], [row], self.inventory["files"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timestamp"], "2026-09-22 15:02:00.000")

    def test_only_confirmed_required_episodes_are_sampled(self):
        row = self.observation()
        draft = new_episode(episode_id="ep-0001", observation=row)
        row["episode_id"] = "ep-0001"
        self.assertEqual(sample_visible_frames([draft], [row],
                                               self.inventory["files"]), [])
        ignore_row = self.observation(truth_class=IGNORE_SMALL, truth_id="t-0002")
        ignore = new_episode(episode_id="ep-0002", observation=ignore_row,
                             review_status=EPISODE_CONFIRMED)
        ignore_row["episode_id"] = "ep-0002"
        self.assertEqual(sample_visible_frames([ignore], [ignore_row],
                                               self.inventory["files"]), [])

    def test_thirty_second_global_grid_is_content_independent(self):
        frames = sample_global_roi_frames({camera: {"roi": ROI, "geometry_version": "g"}
                                           for camera in self.inventory["cameras"]},
                                          self.inventory["files"])
        # two 5 minute chunks per camera, 30 s grid -> 10 frames each
        self.assertEqual(len(frames), 4 * 10)
        per_file: dict[str, int] = {}
        for row in frames:
            per_file[row["source_file_id"]] = per_file.get(row["source_file_id"], 0) + 1
        self.assertEqual(set(per_file.values()), {10})
        for row in frames:
            self.assertEqual(row["sample_reason"], SAMPLE_REASON_GLOBAL)
            self.assertEqual(row["grid_seconds"], GLOBAL_GRID_SECONDS)
            self.assertEqual(row["roi"], ROI)
            self.assertEqual(row["source_width"], 2560)
        self.assertEqual(frames, sample_global_roi_frames(
            {camera: {"roi": ROI, "geometry_version": "g"}
             for camera in self.inventory["cameras"]}, self.inventory["files"]))

    def test_global_grid_has_no_content_input_at_all(self):
        import inspect

        signature = inspect.signature(sample_global_roi_frames)
        self.assertEqual(list(signature.parameters),
                         ["cameras", "files", "grid_seconds"])
        source = inspect.getsource(sample_global_roi_frames)
        for forbidden in ("frame", "image", "detector", "score", "confidence"):
            self.assertNotIn(forbidden + " ", source.lower(), forbidden)


# --------------------------------------------------------------------------- #
# §32: localization
# --------------------------------------------------------------------------- #


class TestLocalization(FixtureCase):
    def test_point_only_and_unresolved_are_preserved_for_adjudication(self):
        point = self.observation(truth_id="t-0001")
        unresolved = self.observation(truth_id="t-0002",
                                      status=LOCALIZATION_UNRESOLVED,
                                      point=(1290.0, 905.0))
        boxed = self.observation(truth_id="t-0003", bbox=[1200.0, 850.0, 1300.0, 950.0],
                                 status=LOCALIZATION_BOX_OK)
        report = localization_state([], [point, unresolved, boxed], [])
        self.assertEqual(report["required_objects"], 3)
        self.assertEqual(report["auto_matchable"], 1)
        self.assertEqual(report["point_only"], 2)
        self.assertEqual(report["unresolved"], ["t-0002"])
        self.assertEqual(report["unresolved_count"], 1)
        flagged = {row["truth_id"]: row["needs_manual_adjudication"]
                   for row in report["objects"]}
        self.assertFalse(flagged["t-0001"])
        self.assertTrue(flagged["t-0002"])
        self.assertIn("never", report["rule"])
        self.assertIn("false negative", report["rule"])

    def test_ignore_and_uncertain_objects_are_not_in_the_localization_report(self):
        ignore = self.observation(truth_id="t-0001", truth_class=IGNORE_SMALL)
        uncertain = self.observation(truth_id="t-0002", truth_class=UNCERTAIN)
        report = localization_state([], [ignore, uncertain], [])
        self.assertEqual(report["required_objects"], 0)


# --------------------------------------------------------------------------- #
# §32: review coverage / resume
# --------------------------------------------------------------------------- #


class TestReviewCoverage(FixtureCase):
    def test_pending_until_reviewed(self):
        coverage = review_coverage(self.inventory["files"], {"files": {}})
        self.assertEqual(coverage["reviewed"], 0)
        self.assertEqual(coverage["pending"], 4)
        self.assertFalse(coverage["complete"])
        with self.assertRaises(BlindTruthError):
            assert_review_complete(coverage)

    def test_partial_and_complete_coverage(self):
        state = {"files": {"fid-01021-0": {"status": REVIEW_DONE}}}
        coverage = review_coverage(self.inventory["files"], state)
        self.assertEqual(coverage["reviewed"], 1)
        self.assertEqual(coverage["per_camera"]["01021"]["reviewed"], 1)
        self.assertIn("fid-01022-0", coverage["pending_file_ids"])
        full = {"files": {row["file_id"]: {"status": REVIEW_DONE}
                          for row in self.inventory["files"]}}
        coverage = review_coverage(self.inventory["files"], full)
        self.assertTrue(coverage["complete"])
        assert_review_complete(coverage)


# --------------------------------------------------------------------------- #
# §32: freeze, immutability, erratum
# --------------------------------------------------------------------------- #


class TestFreezeAndErratum(FixtureCase):
    def _artifact_dir(self) -> Path:
        out = self.root / "artifact"
        out.mkdir(exist_ok=True)
        complete = {"files": {row["file_id"]: {"status": REVIEW_DONE}
                              for row in self.inventory["files"]}}
        coverage = review_coverage(self.inventory["files"], complete)
        (out / "development_manifest.json").write_text(json.dumps(self.inventory))
        (out / "truth_objects.jsonl").write_text(
            json.dumps(self.observation()) + "\n", encoding="utf-8")
        (out / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
        (out / "visible_frame_manifest.jsonl").write_text("{}\n", encoding="utf-8")
        (out / "global_roi_frame_manifest.jsonl").write_text("{}\n", encoding="utf-8")
        (out / "localization_state.json").write_text("{}", encoding="utf-8")
        (out / "SUMMARY.json").write_text("{}", encoding="utf-8")
        (out / "MANIFEST.json").write_text("{}", encoding="utf-8")
        self.coverage = coverage
        return out

    def test_freeze_hashes_every_artifact_and_makes_it_read_only(self):
        out = self._artifact_dir()
        record = freeze_truth(out, review=self.coverage)
        self.assertTrue(is_frozen(out))
        for name in record["artifact_sha256"]:
            self.assertEqual(record["artifact_sha256"][name], sha256_file(out / name))
        self.assertEqual(len(record["truth_sha256"]), 64)
        self.assertFalse(record["detector_loaded"])
        self.assertFalse(record["checkpoint_accessed"])
        self.assertFalse(record["inference_executed"])
        self.assertFalse(record["sealed_accessed"])
        for path in out.glob("*"):
            if path.is_file():
                self.assertEqual(path.stat().st_mode & 0o222, 0, path.name)

    def test_freeze_requires_complete_review(self):
        out = self._artifact_dir()
        partial = review_coverage(self.inventory["files"], {"files": {}})
        with self.assertRaises(BlindTruthError):
            freeze_truth(out, review=partial)
        self.assertFalse(is_frozen(out))

    def test_freeze_requires_every_artifact(self):
        out = self._artifact_dir()
        (out / "visible_frame_manifest.jsonl").unlink()
        with self.assertRaises(BlindTruthError):
            freeze_truth(out, review=self.coverage)

    def test_post_freeze_edits_are_rejected(self):
        out = self._artifact_dir()
        freeze_truth(out, review=self.coverage)
        with self.assertRaises(TruthFrozenError):
            assert_not_frozen(out)

    def test_erratum_records_everything_required(self):
        out = self._artifact_dir()
        freeze_truth(out, review=self.coverage)
        row = append_truth_erratum(
            out, truth_id="t-0001", original={"truth_class": REQUIRED_LITTER},
            corrected={"truth_class": IGNORE_SMALL},
            reason="the item is a bottle cap, too small to clean", after_inference=True)
        for key in ("erratum_at", "original", "corrected", "reason",
                    "recorded_after_inference"):
            self.assertIn(key, row)
        self.assertTrue(row["recorded_after_inference"])
        self.assertEqual(row["original"]["truth_class"], REQUIRED_LITTER)
        lines = (out / "TRUTH_ERRATA.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads((out / "FREEZE.json").read_text(encoding="utf-8"))
        self.assertEqual(record["errata"][0]["truth_id"], "t-0001")

    def test_erratum_requires_a_reason_and_a_frozen_truth(self):
        out = self._artifact_dir()
        with self.assertRaises(BlindTruthError):
            append_truth_erratum(out, truth_id="t-0001", original={}, corrected={},
                                 reason="", after_inference=False)
        freeze_truth(out, review=self.coverage)
        with self.assertRaises(BlindTruthError):
            append_truth_erratum(out, truth_id="t-0001", original={}, corrected={},
                                 reason="", after_inference=False)


# --------------------------------------------------------------------------- #
# §32 store-level behaviour (resume, freeze guards)
# --------------------------------------------------------------------------- #


class TestTruthStore(FixtureCase):
    def _store(self):
        out = self.root / "artifact"
        return CLI_MODULE.TruthStore(out, self.dev, self.roi_dir)

    def test_add_update_delete_and_resume(self):
        store = self._store()
        row = store.add_truth(truth_class=REQUIRED_LITTER, camera_id="01021",
                              source_file_id="fid-01021-0", decoded_timestamp=61.0,
                              source_point=[1280.0, 900.0])
        self.assertEqual(row["truth_id"], "t-0001")
        self.assertTrue(row["in_roi"])
        self.assertEqual(row["timestamp"], "2026-09-22 15:01:01.000")
        resumed = store.review_state()["resume"]
        self.assertEqual(resumed["file_id"], "fid-01021-0")
        self.assertEqual(resumed["decoded_timestamp"], 61.0)
        self.assertEqual(store.review_state()["files"]["fid-01021-0"]["status"],
                         "IN_PROGRESS")

        episode = store.episode_action(action="new", truth_id="t-0001")
        self.assertEqual(episode["episode_id"], "ep-0001")
        store.episode_action(action="confirm", episode_id="ep-0001")
        self.assertEqual(store.episodes[0]["review_status"], EPISODE_CONFIRMED)
        store.episode_action(action="interval", truth_id="t-0001",
                             episode_id="ep-0001", which="end")
        self.assertEqual(store.episodes[0]["last_confirmable_timestamp"],
                         "2026-09-22 15:01:01.000")

        store.update_truth(truth_id="t-0001",
                           source_bbox_xyxy=[1200.0, 850.0, 1300.0, 950.0],
                           localization_status=LOCALIZATION_BOX_OK)
        self.assertEqual(store.truth[0]["source_bbox_xyxy"],
                         [1200.0, 850.0, 1300.0, 950.0])
        store.delete_truth(truth_id="t-0001", reason="misclick")
        self.assertEqual(store.truth, [])

    def test_proposal_without_bbox_degrades_to_unresolved(self):
        store = self._store()
        store.add_truth(truth_class=REQUIRED_LITTER, camera_id="01021",
                        source_file_id="fid-01021-0", decoded_timestamp=10.0,
                        source_point=[1280.0, 900.0])
        row = store.update_truth(truth_id="t-0001",
                                 localization_status=LOCALIZATION_PROPOSAL)
        self.assertEqual(row["localization_status"], LOCALIZATION_UNRESOLVED)

    def test_unknown_ps_and_camera_mismatch_are_refused(self):
        store = self._store()
        with self.assertRaises(BlindTruthError):
            store.add_truth(truth_class=REQUIRED_LITTER, camera_id="01021",
                            source_file_id="nope", decoded_timestamp=1.0,
                            source_point=[10.0, 10.0])
        with self.assertRaises(BlindTruthError):
            store.add_truth(truth_class=REQUIRED_LITTER, camera_id="01022",
                            source_file_id="fid-01021-0", decoded_timestamp=1.0,
                            source_point=[1280.0, 900.0])

    def test_cli_commands_run_through_argparse(self):
        """Catches wiring mistakes the direct TruthStore tests would miss."""
        out = self.root / "cli-artifact"
        common = ["--output", str(out), "--development-root", str(self.dev),
                  "--roi-dir", str(self.roi_dir)]
        quiet = contextlib.redirect_stdout(io.StringIO())
        with quiet:
            self.assertEqual(CLI_MODULE.main(common + ["manifest"]), 0)
            self.assertEqual(CLI_MODULE.main(common + ["status"]), 0)
            self.assertEqual(CLI_MODULE.main(common + ["sample"]), 0)
        inventory = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(inventory["ps_count"], 4)
        # freeze must be refused while the review is incomplete
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.assertEqual(CLI_MODULE.main(common + ["freeze"]), 3)
            self.assertEqual(CLI_MODULE.main(["--output", str(out),
                                              "--development-root", str(self.dev),
                                              "--roi-dir", str(self.roi_dir),
                                              "--erratum", str(self.root / "nope.json"),
                                              "erratum"]), 3)
        self.assertIn("REFUSED", errors.getvalue())

    def test_cli_refuses_a_sealed_development_root(self):
        sealed = self.root / BUNDLE_DIRNAME / "sealed_test"
        sealed.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stderr(io.StringIO()):
            code = CLI_MODULE.main(["--output", str(self.root / "x"),
                                    "--development-root", str(sealed),
                                    "--roi-dir", str(self.roi_dir), "manifest"])
        self.assertEqual(code, 3)

    def test_samples_and_reports_then_freeze_blocks_mutation(self):
        store = self._store()
        store.add_truth(truth_class=REQUIRED_LITTER, camera_id="01021",
                        source_file_id="fid-01021-0", decoded_timestamp=30.0,
                        source_point=[1280.0, 900.0])
        store.episode_action(action="new", truth_id="t-0001")
        store.episode_action(action="confirm", episode_id="ep-0001")
        result = store.build_samples()
        self.assertEqual(result["confirmed_required_episodes"], 1)
        self.assertEqual(result["visible_frames"], 1)
        self.assertEqual(result["global_frames"], 40)
        for row in self.inventory["files"]:
            store.set_review(file_id=row["file_id"], status=REVIEW_DONE)
        store.write_reports(code_commit="test")
        summary = json.loads((store.output / "SUMMARY.json").read_text())
        self.assertEqual(summary["evidence_classification"], EVIDENCE_EXPLORATORY)
        self.assertFalse(summary["blindness"]["detector_loaded"])
        record = store.freeze()
        self.assertEqual(len(record["truth_sha256"]), 64)
        with self.assertRaises(TruthFrozenError):
            store.add_truth(truth_class=REQUIRED_LITTER, camera_id="01021",
                            source_file_id="fid-01021-0", decoded_timestamp=40.0,
                            source_point=[1280.0, 900.0])
        with self.assertRaises(TruthFrozenError):
            store.set_review(file_id="fid-01021-0", status="PENDING")


# --------------------------------------------------------------------------- #
# summary / evidence level
# --------------------------------------------------------------------------- #


class TestSummary(FixtureCase):
    def test_evidence_classification_thresholds(self):
        self.assertEqual(evidence_classification(0, 0), EVIDENCE_EXPLORATORY)
        self.assertEqual(evidence_classification(19, 5), EVIDENCE_EXPLORATORY)
        self.assertEqual(evidence_classification(20, 2), EVIDENCE_DIRECTIONAL)
        self.assertEqual(evidence_classification(49, 5), EVIDENCE_DIRECTIONAL)
        self.assertEqual(evidence_classification(50, 3), EVIDENCE_STRONGER)
        self.assertEqual(evidence_classification(50, 2), EVIDENCE_DIRECTIONAL)

    def test_summary_counts_and_no_model_judgement(self):
        rows = [self.observation(truth_id="t-0001"),
                self.observation(truth_id="t-0002", truth_class=IGNORE_SMALL),
                self.observation(truth_id="t-0003", truth_class=UNCERTAIN),
                self.observation(truth_id="t-0004", truth_class=NON_LITTER)]
        episode = new_episode(episode_id="ep-0001", observation=rows[0],
                              review_status=EPISODE_CONFIRMED)
        rows[0]["episode_id"] = "ep-0001"
        summary = build_summary(self.inventory, truth_objects=rows, episodes=[episode],
                                visible_frames=[], global_frames=[],
                                localization=localization_state([episode], rows, []),
                                review=review_coverage(self.inventory["files"], {}),
                                roi_note="frozen roi")
        truth = summary["truth"]
        self.assertEqual(truth["natural_required_episode_count"], 1)
        self.assertEqual(truth["required_object_count"], 1)
        self.assertEqual(truth["ignore_small_object_count"], 1)
        self.assertEqual(truth["uncertain_object_count"], 1)
        self.assertEqual(truth["non_litter_object_count"], 1)
        self.assertEqual(truth["per_camera_required_episode_count"], {"01021": 1})
        self.assertEqual(summary["evidence_classification"], EVIDENCE_EXPLORATORY)
        self.assertFalse(summary["blindness"]["inference_executed"])
        self.assertFalse(summary["blindness"]["proposal_recall_computed"])
        self.assertFalse(summary["blindness"]["threshold_selection_performed"])
        self.assertTrue(summary["boundaries"]["development_accessed"])
        self.assertFalse(summary["boundaries"]["sealed_accessed"])
        self.assertFalse(summary["boundaries"]["truth_modified_after_freeze"])
        self.assertIn("no model output", summary["note"])


if __name__ == "__main__":
    unittest.main()
