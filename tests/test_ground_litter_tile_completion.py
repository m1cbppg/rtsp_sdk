"""Step 1C-2M: supplemental multi-target completion tests (§39).

Pure stdlib; the point-proposal engine and the tile images are injected, so no cv2 /
GPU / network is required.  The real proposal engine is exercised by the joint test.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest

from rtsp_annotator.ground_litter_localization_review import (
    ReviewError,
    sha256_file,
)
from rtsp_annotator.ground_litter_positive_tiles import (
    CLASS_ID,
    TILE_SIZE,
    png_bytes,
    write_jsonl,
)
from rtsp_annotator.ground_litter_tile_completion import (
    DUPLICATE_IOU,
    FINAL_TILE_STATUSES,
    GENERATOR_VERSION,
    LOCALIZATION_TRUNCATED,
    LOCALIZATION_UNRESOLVED,
    LOCALIZATION_VERIFIED,
    MAX_PROPOSAL_REVISIONS,
    MAX_SUPPLEMENTAL_PER_TILE,
    SCHEMA_VERSION,
    TILE_STATUS_COMPLETE,
    TILE_STATUS_NEEDS,
    TILE_STATUS_STILL_MISSING,
    TILE_STATUS_UNCERTAIN,
    TILE_STATUSES,
    CompletionError,
    CompletionState,
    apply_state,
    box_contains_point,
    box_iou,
    build_accepted_v2,
    build_manifest,
    build_summary,
    load_completion_input,
    merged_labels,
    proposal_statistics,
    queue_rows,
    source_to_tile,
    supplemental_target_id,
    tile_to_source,
    truncation_for_box,
    validate_source_box,
    validate_tile_box,
    verify_preflight,
    yolo_from_tile_box,
)

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "ground_litter_completion_ui"
CLI = ROOT / "scripts" / "complete_ground_litter_positive_tiles.py"


def synthetic_tile(width: int = TILE_SIZE, height: int = TILE_SIZE):
    import numpy as np

    ys, xs = np.mgrid[0:height, 0:width]
    return np.stack([(xs % 256), (ys % 256), ((xs + ys) % 256)], axis=2).astype("uint8")


class FakeProposalEngine:
    """Deterministic A/B/C generator: no cv2, no image content needed."""

    def __init__(self, *, count=3, fail=False, box=(400, 400, 440, 430)):
        self.count = count
        self.fail = fail
        self.box = list(box)
        self.calls: list[tuple[float, float, int]] = []

    def propose(self, *, tile_path, point_tile_xy, size, size_prior=None, revision=1):
        self.calls.append((float(point_tile_xy[0]), float(point_tile_xy[1]), int(revision)))
        if self.fail:
            return {"ok": False, "error_code": "NO_PROPOSAL_FOUND",
                    "message": "simulated proposal failure", "candidates": []}
        candidates = []
        for index, letter in enumerate(("A", "B", "C")[: self.count]):
            box = [self.box[0] + index * 60, self.box[1], self.box[2] + index * 60,
                   self.box[3]]
            candidates.append({
                "letter": letter, "bbox_tile_xyxy": box,
                "method": f"fake_{letter.lower()}",
                "contains_point": box_contains_point(box, point_tile_xy),
                "touches_border": None})
        return {"ok": True, "candidates": candidates, "point_tile": list(point_tile_xy),
                "revision": int(revision)}


# --------------------------------------------------------------------------- #
# fixture: a miniature but fully shaped Step 1C-2 artifact
# --------------------------------------------------------------------------- #


def tile_row(tile_id, *, camera="01021", episode=None, labels=None, crop=(800, 600, 1440, 1240),
             source=(2560, 1440), image=None, image_sha="", width=TILE_SIZE) -> dict:
    labels = labels or [{"episode_ids": [episode], "tile_xyxy": [100.0, 100.0, 140.0, 130.0],
                         "source_xyxy": [crop[0] + 100.0, crop[1] + 100.0,
                                         crop[0] + 140.0, crop[1] + 130.0],
                         "yolo_xywh_norm": yolo_from_tile_box([100, 100, 140, 130]),
                         "source_short_side_px": 30.0, "size_bucket": "20-39",
                         "class_id": CLASS_ID, "class_name": "ground_litter"}]
    return {
        "tile_id": tile_id, "primary_episode_id": episode or f"ge-{tile_id}",
        "primary_episode_ids": [episode or f"ge-{tile_id}"], "camera_id": camera,
        "scene_version": "UNKNOWN_HISTORICAL", "source_file_id": "ps-a",
        "source_ps_path": "/tmp/fake/ps-a.ps",
        "step1c0_decoded_timestamp": "2026-09-19 01:32:32",
        "step1c2_decoded_timestamp": "2026-09-19 01:32:32.006",
        "source_width": source[0], "source_height": source[1],
        "source_crop_xyxy": list(crop), "crop_size": width,
        "primary_source_bbox": [900.0, 700.0, 940.0, 730.0],
        "labels": labels, "label_count": len(labels),
        "image_path": image, "image_sha256": image_sha, "size_bytes": None,
        "candidate_generation_status": "READY_FOR_REVIEW",
        "annotation_review_status": None, "positive_training_ready": False,
        "known_unlocalized_required_present": False, "risk_flags": [],
        "all_known_label_episode_ids": [episode or f"ge-{tile_id}"],
    }


class Fixture:
    """Writes a Step 1C-2-shaped artifact plus a completion output directory."""

    def __init__(self, root: Path, *, n_complete=2, n_missing=3, n_box_problem=1) -> None:
        self.root = root
        self.step1c2 = root / "step1c2"
        self.images = self.step1c2 / "candidate_tiles" / "images"
        self.accepted = self.step1c2 / "accepted"
        self.images.mkdir(parents=True, exist_ok=True)
        (self.accepted / "images").mkdir(parents=True, exist_ok=True)
        (self.accepted / "labels").mkdir(parents=True, exist_ok=True)
        self.out = root / "out"
        self.state = root / "state" / "completion_state.json"
        self.rows: list[dict] = []
        self.decisions: dict[str, dict] = {}
        self.accepted_rows: list[dict] = []
        self.payload = png_bytes(synthetic_tile())

        for index in range(n_complete):
            self._add(f"pt-complete-{index}", "01021", "ANNOTATION_COMPLETE", accepted=True)
        for index in range(n_missing):
            self._add(f"pt-missing-{index}", "01022", "MISSING_REQUIRED", accepted=False)
        for index in range(n_box_problem):
            self._add(f"pt-box-{index}", "01027", "BOX_PROBLEM", accepted=False)
        self.write()

    def _add(self, tile_id: str, camera: str, decision: str, *, accepted: bool,
             labels=None) -> None:
        image = self.images / f"{tile_id}.png"
        image.write_bytes(self.payload)
        row = tile_row(tile_id, camera=camera,
                       episode=f"ge-{camera}-{tile_id.split('-')[-1]}", labels=labels,
                       image=str(image), image_sha=sha256_file(image))
        self.rows.append(row)
        self.decisions[tile_id] = {
            "tile_id": tile_id, "annotation_review_status": decision,
            "annotation_review_note": f"note-{decision}",
            "positive_training_ready": decision == "ANNOTATION_COMPLETE",
            "reviewed_at": "2026-09-23T00:00:00Z", "label_count": row["label_count"],
            "image_sha256": row["image_sha256"]}
        if accepted:
            accepted_image = self.accepted / "images" / f"{tile_id}.png"
            accepted_image.write_bytes(self.payload)
            accepted_label = self.accepted / "labels" / f"{tile_id}.txt"
            accepted_label.write_text("0 0.187500 0.179688 0.062500 0.046875\n",
                                      encoding="utf-8")
            self.accepted_rows.append({
                "tile_id": tile_id, "primary_episode_id": row["primary_episode_id"],
                "primary_episode_ids": row["primary_episode_ids"], "camera_id": camera,
                "source_file_id": "ps-a", "source_crop_xyxy": row["source_crop_xyxy"],
                "image_path": str(accepted_image), "image_sha256": row["image_sha256"],
                "label_path": str(accepted_label), "label_count": 1,
                "labels": row["labels"], "annotation_complete": True,
                "positive_training_ready": True})

    def write(self) -> None:
        write_jsonl(self.step1c2 / "tile_candidates.jsonl", self.rows)
        (self.step1c2 / "review_state.json").write_text(
            json.dumps({"review_schema_version": "positive_tile_review_v1",
                        "candidate_count": len(self.rows), "input_fingerprint": "x",
                        "decisions": self.decisions, "audit_trail": []}),
            encoding="utf-8")
        counts = {"ANNOTATION_COMPLETE": 0, "MISSING_REQUIRED": 0, "BOX_PROBLEM": 0,
                  "PENDING": 0, "UNCERTAIN_COMPLETENESS": 0}
        for decision in self.decisions.values():
            counts[decision["annotation_review_status"]] += 1
        summary = {
            "schema_version": "ground_litter_positive_tiles_v1",
            "generator_version": "step1c2-1.1.0", "tile_size": 640,
            "class_mapping": {"0": "ground_litter"},
            "input": {"training_eligible_episode_count": len(self.rows)},
            "generate": {"candidate_tile_count": len(self.rows),
                         "deduplicated_tile_count": len(self.rows)},
            "review": {"reviewable": len(self.rows), "reviewed": len(self.rows),
                       "pending": 0, "skipped": 0, "counts": counts,
                       "decisions": {key: value for key, value in counts.items()
                                     if key != "PENDING"}},
            "accepted": {"state": "built",
                         "accepted_positive_tile_count": len(self.accepted_rows),
                         "accepted_label_count": len(self.accepted_rows),
                         "unique_episode_ids_represented": len(self.accepted_rows)},
            "per_camera": {}, "boundaries": {},
        }
        (self.step1c2 / "SUMMARY.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        write_jsonl(self.step1c2 / "positive_training_manifest.jsonl", self.accepted_rows)
        manifest = {
            "schema_version": "ground_litter_positive_tiles_v1",
            "generator_version": "step1c2-1.1.0", "code_commit": "0" * 40,
            "tile_size": 640, "source_native": True, "resize": False,
            "class_mapping": {"0": "ground_litter"},
            "input_sha256": {}, "artifact_root": str(self.step1c2), "config": {},
            "outputs": {}, "counts": {
                "candidate_tile_count": len(self.rows),
                "reviewable_tile_count": len(self.rows),
                "reviewed": len(self.rows),
                "accepted_positive_tile_count": len(self.accepted_rows)},
            "upstream_provenance": {}, "boundaries": {},
            "tile_candidates_sha256": sha256_file(self.step1c2 / "tile_candidates.jsonl"),
            "summary_sha256": sha256_file(self.step1c2 / "SUMMARY.json"),
        }
        (self.step1c2 / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    def load(self) -> dict:
        return load_completion_input(self.step1c2)

    def missing_rows(self, data) -> list[dict]:
        return queue_rows(data)


class Base(unittest.TestCase):
    def tmpdir(self) -> Path:
        path = Path(tempfile.mkdtemp(prefix="step1c2m-"))
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def fixture(self, **kwargs) -> Fixture:
        return Fixture(self.tmpdir(), **kwargs)

    def setup_one(self, *, existing_labels=1):
        """One missing tile with a chosen number of existing verified labels."""
        labels = None
        if existing_labels > 1:
            labels = []
            for index in range(existing_labels):
                box = [100.0 + index * 120, 100.0, 140.0 + index * 120, 130.0]
                labels.append({
                    "episode_ids": [f"ge-x{index}"],
                    "tile_xyxy": box,
                    "source_xyxy": [box[0] + 800, box[1] + 600, box[2] + 800, box[3] + 600],
                    "yolo_xywh_norm": yolo_from_tile_box(box),
                    "source_short_side_px": 30.0, "size_bucket": "20-39"})
        f = self.fixture(n_complete=0, n_missing=1, n_box_problem=0)
        f.rows[0] = tile_row(f.rows[0]["tile_id"], labels=labels,
                             image=f.rows[0]["image_path"],
                             image_sha=f.rows[0]["image_sha256"])
        f.rows[0]["label_count"] = len(f.rows[0]["labels"])
        f.accepted_rows = []
        f.write()
        data = f.load()
        return f, data, f.missing_rows(data)


# --------------------------------------------------------------------------- #
# §39 Queue
# --------------------------------------------------------------------------- #


class QueueTest(Base):
    def test_queue_is_exactly_the_missing_required_tiles(self) -> None:
        f = self.fixture(n_complete=29, n_missing=62, n_box_problem=2)
        data = f.load()
        preflight = verify_preflight(data)
        self.assertEqual(preflight["step1c2_annotation_complete"], 29)
        self.assertEqual(preflight["step1c2_missing_required"], 62)
        self.assertEqual(preflight["step1c2_box_problem"], 2)
        self.assertEqual(preflight["old_accepted_positive_tile_count"], 29)
        rows = queue_rows(data)
        self.assertEqual(len(rows), 62)
        ids = {row["tile_id"] for row in rows}
        self.assertNotIn("pt-complete-0", ids)
        self.assertNotIn("pt-box-0", ids)
        self.assertTrue(all(row["tile_id"].startswith("pt-missing") for row in rows))

    def test_preflight_reports_a_count_mismatch_instead_of_proceeding(self) -> None:
        f = self.fixture(n_complete=2, n_missing=2, n_box_problem=1)
        data = f.load()
        preflight = verify_preflight(data)
        self.assertFalse(preflight["counts_match"])
        self.assertNotIn("counts_match", [])          # explicit, no silent pass
        self.assertEqual(preflight["problem_count"], 0)

    def test_preflight_detects_a_changed_candidate_png(self) -> None:
        f = self.fixture()
        Path(f.rows[0]["image_path"]).write_bytes(b"\x89PNG\r\n\x1a\nbroken")
        problems = {p["field"] for p in verify_preflight(f.load())["problems"]}
        self.assertIn("candidate_png_hash_drift", problems)

    def test_preflight_detects_a_changed_step1c2_manifest_hash(self) -> None:
        f = self.fixture()
        manifest = json.loads((f.step1c2 / "MANIFEST.json").read_text())
        manifest["tile_candidates_sha256"] = "0" * 64
        (f.step1c2 / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        problems = {p["field"] for p in verify_preflight(f.load())["problems"]}
        self.assertIn("tile_candidates_hash_mismatch", problems)

    def test_queue_rows_expose_existing_labels_and_counts(self) -> None:
        f = self.fixture(n_complete=0, n_missing=2, n_box_problem=0)
        rows = f.missing_rows(f.load())
        for row in rows:
            self.assertEqual(row["existing_label_count"], len(row["existing_labels"]))
            self.assertTrue(row["image_path"])
            self.assertEqual(len(row["source_crop_xyxy"]), 4)


# --------------------------------------------------------------------------- #
# §39 Point + coordinates
# --------------------------------------------------------------------------- #


class CoordinateTest(unittest.TestCase):
    def test_tile_to_source_and_back(self) -> None:
        crop = [800, 600, 1440, 1240]
        self.assertEqual(tile_to_source([100, 200], crop), [900, 800])
        self.assertEqual(tile_to_source([10, 20, 30, 40], crop), [810, 620, 830, 640])
        self.assertEqual(source_to_tile([900, 800], crop), [100, 200])
        for point in ([0, 0], [640, 640], [123.4, 567.8]):
            self.assertEqual(source_to_tile(tile_to_source(point, crop), crop), list(point))

    def test_validate_tile_box_rejects_out_of_tile_and_degenerate(self) -> None:
        self.assertIsNotNone(validate_tile_box([10, 20, 30, 40]))
        self.assertIsNone(validate_tile_box([-1, 20, 30, 40]))
        self.assertIsNone(validate_tile_box([10, 20, 30, 641]))
        self.assertIsNone(validate_tile_box([30, 20, 30, 40]))
        self.assertIsNone(validate_tile_box([10, 40, 30, 20]))

    def test_validate_source_box_uses_the_frame_and_tile_size(self) -> None:
        self.assertIsNotNone(validate_source_box([100, 100, 200, 200], 2560, 1440))
        self.assertIsNone(validate_source_box([-5, 100, 200, 200], 2560, 1440))
        self.assertIsNone(validate_source_box([2500, 100, 2600, 200], 2560, 1440))
        self.assertIsNone(validate_source_box([100, 100, 800, 200], 2560, 1440))  # > 640

    def test_truncation_and_yolo_geometry(self) -> None:
        self.assertEqual(truncation_for_box([100, 100, 200, 200]), [])
        self.assertEqual(truncation_for_box([0, 100, 200, 200]), ["left"])
        self.assertEqual(truncation_for_box([0, 0, 640, 640]), ["left", "top", "right", "bottom"])
        self.assertEqual(yolo_from_tile_box([0, 0, 640, 640]), [0.5, 0.5, 1.0, 1.0])
        for value in yolo_from_tile_box([10, 20, 30, 40]):
            self.assertTrue(0.0 <= value <= 1.0)

    def test_supplemental_target_id_is_stable_and_deterministic(self) -> None:
        one = supplemental_target_id("pt-abc", 1)
        again = supplemental_target_id("pt-abc", 1)
        other = supplemental_target_id("pt-abc", 2)
        self.assertEqual(one, again)
        self.assertNotEqual(one, other)
        self.assertTrue(one.startswith("st-"))
        self.assertEqual(len(one), len("st-") + 20)


# --------------------------------------------------------------------------- #
# §39 Proposal
# --------------------------------------------------------------------------- #


class ProposalTest(Base):
    def test_point_creates_a_target_with_at_most_three_proposals(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        engine = FakeProposalEngine(count=3)
        result = engine.propose(tile_path=Path(rows[0]["image_path"]),
                                point_tile_xy=[200, 200], size=TILE_SIZE)
        target = state.add_target(rows[0], point_tile=[200, 200], proposal_result=result)
        self.assertEqual([c["letter"] for c in target["proposal_candidates"]], ["A", "B", "C"])
        self.assertIsNone(target["selected_proposal"])
        self.assertIsNone(target["verified_tile_xyxy"])
        self.assertEqual(target["localization_status"], None)

    def test_proposals_never_become_labels_without_a_pick(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        target = state.add_target(rows[0], point_tile=[420, 415],
                                  proposal_result=FakeProposalEngine().propose(
                                      tile_path=Path(rows[0]["image_path"]),
                                      point_tile_xy=[420, 415], size=TILE_SIZE))
        self.assertIsNone(target["selected_proposal"])
        applied = apply_state([rows[0]], state)[0]
        self.assertEqual(applied["final_label_count"], 1)          # only the existing one
        self.assertFalse(applied["positive_training_ready"])

    def test_only_around_the_human_point(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        engine = FakeProposalEngine(box=(190, 190, 230, 220))
        engine.propose(tile_path=Path(rows[0]["image_path"]), point_tile_xy=[205, 200],
                       size=TILE_SIZE)
        self.assertEqual(engine.calls[-1][0:2], (205.0, 200.0))

    def test_select_stores_source_and_tile_coordinates(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        target = state.add_target(rows[0], point_tile=[420, 415],
                                  proposal_result=FakeProposalEngine().propose(
                                      tile_path=Path(rows[0]["image_path"]),
                                      point_tile_xy=[420, 415], size=TILE_SIZE))
        outcome = state.select_proposal(rows[0]["tile_id"],
                                       target["supplemental_target_id"], "A",
                                       existing_boxes=[])
        saved = outcome["target"]
        self.assertEqual(outcome["warning"], None)
        self.assertEqual(saved["localization_status"], LOCALIZATION_VERIFIED)
        self.assertEqual(saved["verified_tile_xyxy"], [400.0, 400.0, 440.0, 430.0])
        self.assertEqual(saved["verified_source_xyxy"], [1200.0, 1000.0, 1240.0, 1030.0])

    def test_second_revision_then_hard_stop(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        target = state.add_target(rows[0], point_tile=[200, 200],
                                  proposal_result=FakeProposalEngine().propose(
                                      tile_path=Path(rows[0]["image_path"]),
                                      point_tile_xy=[200, 200], size=TILE_SIZE))
        target_id = target["supplemental_target_id"]
        again = state.repoint(rows[0], target_id, point_tile=[260, 240],
                              proposal_result=FakeProposalEngine().propose(
                                  tile_path=Path(rows[0]["image_path"]),
                                  point_tile_xy=[260, 240], size=TILE_SIZE))
        self.assertEqual(again["proposal_revision"], 2)
        self.assertEqual(len(again["click_history"]), 2)
        self.assertEqual(again["original_click"]["tile_x"], 200)      # always kept
        with self.assertRaises(ReviewError):
            state.repoint(rows[0], target_id, point_tile=[300, 300],
                          proposal_result=FakeProposalEngine().propose(
                              tile_path=Path(rows[0]["image_path"]),
                              point_tile_xy=[300, 300], size=TILE_SIZE))
        self.assertEqual(MAX_PROPOSAL_REVISIONS, 2)

    def test_failed_proposals_become_unresolved(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        target = state.add_target(rows[0], point_tile=[200, 200],
                                  proposal_result=FakeProposalEngine(fail=True).propose(
                                      tile_path=Path(rows[0]["image_path"]),
                                      point_tile_xy=[200, 200], size=TILE_SIZE))
        self.assertEqual(target["proposal_candidates"], [])
        self.assertEqual(target["proposal_error"]["error_code"], "NO_PROPOSAL_FOUND")
        state.reject_target(rows[0]["tile_id"], target["supplemental_target_id"],
                            reason="two rounds failed")
        applied = apply_state([rows[0]], state)[0]
        self.assertEqual(applied["supplemental_targets"][0]["localization_status"],
                         LOCALIZATION_UNRESOLVED)
        self.assertFalse(applied["positive_training_ready"])

    def test_proposal_engine_returns_structured_errors(self) -> None:
        from rtsp_annotator.ground_litter_point_proposal import PointProposalEngine

        engine = PointProposalEngine()
        missing = engine.propose(tile_path=Path("/tmp/does-not-exist.png"),
                                point_tile_xy=[10, 10], size=TILE_SIZE)
        self.assertFalse(missing["ok"])
        self.assertEqual(missing["error_code"], "TILE_IMAGE_MISSING")
        self.assertTrue(missing["message"])
        outside = engine.propose(tile_path=Path("/tmp/does-not-exist.png"),
                                 point_tile_xy=[700, 10], size=TILE_SIZE)
        self.assertFalse(outside["ok"])
        self.assertEqual(outside["error_code"], "POINT_OUTSIDE_TILE")


# --------------------------------------------------------------------------- #
# §39 Multiple missing + existing label protection
# --------------------------------------------------------------------------- #


class MultiTargetTest(Base):
    def _add(self, state, rows, engine, point):
        result = engine.propose(tile_path=Path(rows[0]["image_path"]),
                                point_tile_xy=list(point), size=TILE_SIZE)
        return state.add_target(rows[0], point_tile=list(point), proposal_result=result)

    def test_a_tile_can_collect_five_supplemental_targets(self) -> None:
        f, data, rows = self.setup_one(existing_labels=1)
        state = CompletionState.load(f.state)
        engine = FakeProposalEngine()
        points = [(160, 300), (260, 300), (360, 300), (460, 300), (560, 300)]
        ids = []
        for index, point in enumerate(points):
            engine.box = [point[0] - 20, point[1] - 15, point[0] + 20, point[1] + 15]
            target = self._add(state, rows, engine, point)
            ids.append(target["supplemental_target_id"])
            state.select_proposal(rows[0]["tile_id"], target["supplemental_target_id"],
                                  "A", existing_boxes=[])
        applied = apply_state([rows[0]], state)[0]
        self.assertEqual(len(applied["supplemental_targets"]), 5)
        self.assertEqual(len(set(ids)), 5)
        # 1 existing + 5 supplemental, all distinct boxes
        self.assertEqual(applied["final_label_count"], 6)
        self.assertTrue(all(label["class_id"] == CLASS_ID
                            for label in applied["merged_labels"]))

    def test_existing_labels_are_never_modified(self) -> None:
        f, data, rows = self.setup_one(existing_labels=2)
        state = CompletionState.load(f.state)
        before = json.dumps(rows[0]["labels"], sort_keys=True)
        engine = FakeProposalEngine(box=(400, 400, 440, 430))
        target = self._add(state, rows, engine, (410, 410))
        state.select_proposal(rows[0]["tile_id"], target["supplemental_target_id"], "A",
                              existing_boxes=[])
        applied = apply_state([rows[0]], state)[0]
        self.assertEqual(json.dumps(rows[0]["labels"], sort_keys=True), before)
        origins = [label["origin"] for label in applied["merged_labels"]]
        self.assertEqual(origins.count("step1c2_verified"), 2)
        self.assertEqual(origins.count("step1c2m_supplemental"), 1)

    def test_delete_and_clear(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        engine = FakeProposalEngine()
        first = self._add(state, rows, engine, (150, 150))
        second = self._add(state, rows, engine, (350, 350))
        state.delete_target(rows[0]["tile_id"], first["supplemental_target_id"])
        self.assertEqual(len(state.targets(rows[0]["tile_id"])), 1)
        state.clear_targets(rows[0]["tile_id"])
        self.assertEqual(state.targets(rows[0]["tile_id"]), [])
        self.assertEqual(state.tile(rows[0]["tile_id"])["status"], TILE_STATUS_NEEDS)
        with self.assertRaises(ReviewError):
            state.delete_target(rows[0]["tile_id"], second["supplemental_target_id"])

    def test_max_supplemental_targets_is_enforced(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        engine = FakeProposalEngine()
        for index in range(MAX_SUPPLEMENTAL_PER_TILE):
            point = (20 + (index % 20) * 30, 20 + (index // 20) * 30)
            engine.box = [point[0], point[1], point[0] + 10, point[1] + 10]
            self._add(state, rows, engine, point)
        with self.assertRaises(ReviewError):
            self._add(state, rows, engine, (600, 600))
        self.assertEqual(len(state.targets(rows[0]["tile_id"])),
                         MAX_SUPPLEMENTAL_PER_TILE)


# --------------------------------------------------------------------------- #
# §39 Duplicate + boundary
# --------------------------------------------------------------------------- #


class GuardTest(Base):
    def test_click_inside_an_existing_box_warns_about_a_duplicate(self) -> None:
        f, data, rows = self.setup_one(existing_labels=1)
        state = CompletionState.load(f.state)
        target = state.add_target(rows[0], point_tile=[120, 115],
                                  proposal_result=FakeProposalEngine(
                                      box=(100, 100, 140, 130)).propose(
                                      tile_path=Path(rows[0]["image_path"]),
                                      point_tile_xy=[120, 115], size=TILE_SIZE))
        outcome = state.select_proposal(
            rows[0]["tile_id"], target["supplemental_target_id"], "A",
            existing_boxes=[{"label_id": "existing-0", "tile_xyxy": [100, 100, 140, 130]}])
        self.assertEqual(outcome["warning"], "POSSIBLE_DUPLICATE_TARGET")
        self.assertTrue(outcome["reason"] in ("same_box", "point_inside_existing"))
        # nothing was saved
        self.assertIsNone(outcome["target"]["selected_proposal"])
        self.assertIsNone(outcome["target"]["verified_tile_xyxy"])

    def test_high_iou_but_distinct_click_still_warns(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        target = state.add_target(rows[0], point_tile=[400, 400],
                                  proposal_result=FakeProposalEngine(
                                      box=(103, 100, 143, 130)).propose(
                                      tile_path=Path(rows[0]["image_path"]),
                                      point_tile_xy=[400, 400], size=TILE_SIZE))
        outcome = state.select_proposal(
            rows[0]["tile_id"], target["supplemental_target_id"], "A",
            existing_boxes=[{"label_id": "existing-0", "tile_xyxy": [100, 100, 140, 130]}])
        self.assertEqual(outcome["warning"], "POSSIBLE_DUPLICATE_TARGET")
        self.assertGreaterEqual(outcome["matches"][0]["iou"], DUPLICATE_IOU)

    def test_confirming_an_independent_target_saves_it(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        target = state.add_target(rows[0], point_tile=[400, 400],
                                  proposal_result=FakeProposalEngine(
                                      box=(100, 100, 140, 130)).propose(
                                      tile_path=Path(rows[0]["image_path"]),
                                      point_tile_xy=[400, 400], size=TILE_SIZE))
        state.select_proposal(
            rows[0]["tile_id"], target["supplemental_target_id"], "A",
            existing_boxes=[{"label_id": "existing-0", "tile_xyxy": [100, 100, 140, 130]}])
        outcome = state.select_proposal(
            rows[0]["tile_id"], target["supplemental_target_id"], "A",
            existing_boxes=[{"label_id": "existing-0", "tile_xyxy": [100, 100, 140, 130]}],
            confirm_independent=True)
        self.assertEqual(outcome["target"]["localization_status"], LOCALIZATION_VERIFIED)
        self.assertTrue(outcome["target"]["duplicate_confirmed_independent"])

    def test_a_proposal_touching_the_tile_edge_is_truncated_and_cannot_complete(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        target = state.add_target(rows[0], point_tile=[20, 400],
                                  proposal_result=FakeProposalEngine(
                                      box=(0, 380, 60, 430)).propose(
                                      tile_path=Path(rows[0]["image_path"]),
                                      point_tile_xy=[20, 400], size=TILE_SIZE))
        outcome = state.select_proposal(rows[0]["tile_id"],
                                       target["supplemental_target_id"], "A",
                                       existing_boxes=[])
        self.assertEqual(outcome["warning"], "TARGET_TRUNCATED_BY_TILE")
        self.assertEqual(outcome["sides"], ["left"])
        self.assertEqual(outcome["target"]["localization_status"], LOCALIZATION_TRUNCATED)
        with self.assertRaises(ReviewError) as caught:
            state.recheck(rows[0]["tile_id"], TILE_STATUS_COMPLETE)
        self.assertIn("truncated", str(caught.exception))

    def test_tile_cannot_complete_while_a_target_is_unresolved(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        good = state.add_target(rows[0], point_tile=[400, 400],
                                proposal_result=FakeProposalEngine(
                                    box=(380, 380, 420, 410)).propose(
                                    tile_path=Path(rows[0]["image_path"]),
                                    point_tile_xy=[400, 400], size=TILE_SIZE))
        state.select_proposal(rows[0]["tile_id"], good["supplemental_target_id"], "A",
                              existing_boxes=[])
        bad = state.add_target(rows[0], point_tile=[200, 200],
                               proposal_result=FakeProposalEngine(fail=True).propose(
                                   tile_path=Path(rows[0]["image_path"]),
                                   point_tile_xy=[200, 200], size=TILE_SIZE))
        state.reject_target(rows[0]["tile_id"], bad["supplemental_target_id"], reason="x")
        with self.assertRaises(ReviewError):
            state.recheck(rows[0]["tile_id"], TILE_STATUS_COMPLETE)


# --------------------------------------------------------------------------- #
# §39 Recheck + resume
# --------------------------------------------------------------------------- #


class RecheckTest(Base):
    def _complete_one(self, f, rows, state, point=(400, 400)):
        target = state.add_target(rows[0], point_tile=list(point),
                                  proposal_result=FakeProposalEngine(
                                      box=(point[0] - 20, point[1] - 15,
                                           point[0] + 20, point[1] + 15)).propose(
                                      tile_path=Path(rows[0]["image_path"]),
                                      point_tile_xy=list(point), size=TILE_SIZE))
        state.select_proposal(rows[0]["tile_id"], target["supplemental_target_id"], "A",
                              existing_boxes=[])
        return target

    def test_complete_requires_an_explicit_recheck(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        self._complete_one(f, rows, state)
        applied = apply_state([rows[0]], state)[0]
        self.assertEqual(applied["completion_status"], TILE_STATUS_NEEDS)
        self.assertFalse(applied["positive_training_ready"])
        entry = state.recheck(rows[0]["tile_id"], TILE_STATUS_COMPLETE, note="checked")
        self.assertEqual(entry["status"], TILE_STATUS_COMPLETE)
        applied = apply_state([rows[0]], state)[0]
        self.assertTrue(applied["positive_training_ready"])

    def test_other_final_statuses_stay_out_of_training(self) -> None:
        for status in (TILE_STATUS_STILL_MISSING, TILE_STATUS_UNCERTAIN):
            f, data, rows = self.setup_one()
            state = CompletionState.load(f.state)
            self._complete_one(f, rows, state)
            state.recheck(rows[0]["tile_id"], status, note="reviewer says so")
            applied = apply_state([rows[0]], state)[0]
            self.assertEqual(applied["completion_status"], status)
            self.assertFalse(applied["positive_training_ready"])

    def test_unknown_status_is_rejected(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        with self.assertRaises(ReviewError):
            state.recheck(rows[0]["tile_id"], "DONE")

    def test_resume_keeps_targets_and_audit_trail(self) -> None:
        f, data, rows = self.setup_one()
        state = CompletionState.load(f.state)
        target = self._complete_one(f, rows, state)
        state.recheck(rows[0]["tile_id"], TILE_STATUS_COMPLETE, note="ok")
        restarted = CompletionState.load(f.state, input_fingerprint="")
        self.assertEqual(len(restarted.targets(rows[0]["tile_id"])), 1)
        self.assertEqual(restarted.tile(rows[0]["tile_id"])["status"], TILE_STATUS_COMPLETE)
        self.assertTrue(restarted.audit_trail)
        self.assertEqual(restarted.targets(rows[0]["tile_id"])[0]
                         ["supplemental_target_id"], target["supplemental_target_id"])
        self.assertEqual(restarted.progress(rows)["reviewed"], 1)

    def test_state_bound_to_a_different_input_is_refused(self) -> None:
        f, data, rows = self.setup_one()
        CompletionState.load(f.state, input_fingerprint="fingerprint-a").save()
        with self.assertRaises(ReviewError):
            CompletionState.load(f.state, input_fingerprint="fingerprint-b")


# --------------------------------------------------------------------------- #
# §39 Build / labels / image immutability
# --------------------------------------------------------------------------- #


class BuildTest(Base):
    def _salvage(self, f, data, rows, index=0, point=(400, 400)):
        state = CompletionState.load(f.state)
        target = state.add_target(rows[index], point_tile=list(point),
                                  proposal_result=FakeProposalEngine(
                                      box=(point[0] - 20, point[1] - 15,
                                           point[0] + 20, point[1] + 15)).propose(
                                      tile_path=Path(rows[index]["image_path"]),
                                      point_tile_xy=list(point), size=TILE_SIZE))
        state.select_proposal(rows[index]["tile_id"],
                              target["supplemental_target_id"], "A", existing_boxes=[])
        state.recheck(rows[index]["tile_id"], TILE_STATUS_COMPLETE, note="ok")
        return state

    def test_merged_labels_union_and_dedup(self) -> None:
        labels = [
            {"episode_ids": ["ge-1"], "tile_xyxy": [100.0, 100.0, 140.0, 130.0],
             "source_xyxy": [900.0, 700.0, 940.0, 730.0],
             "yolo_xywh_norm": yolo_from_tile_box([100, 100, 140, 130]),
             "source_short_side_px": 30.0, "size_bucket": "20-39"},
            {"episode_ids": ["ge-2"], "tile_xyxy": [300.0, 300.0, 340.0, 330.0],
             "source_xyxy": [1100.0, 900.0, 1140.0, 930.0],
             "yolo_xywh_norm": yolo_from_tile_box([300, 300, 340, 330]),
             "source_short_side_px": 30.0, "size_bucket": "20-39"},
        ]
        row = tile_row("pt-x", labels=labels)
        supplementals = [
            {"supplemental_target_id": "st-1", "localization_status": LOCALIZATION_VERIFIED,
             "verified_tile_xyxy": [500.0, 500.0, 540.0, 530.0],
             "verified_source_xyxy": [1300.0, 1100.0, 1340.0, 1130.0]},
            # nearly identical to the first existing label: must dedup into it
            {"supplemental_target_id": "st-2", "localization_status": LOCALIZATION_VERIFIED,
             "verified_tile_xyxy": [101.0, 100.0, 141.0, 130.0],
             "verified_source_xyxy": [901.0, 700.0, 941.0, 730.0]},
            {"supplemental_target_id": "st-3", "localization_status": LOCALIZATION_UNRESOLVED,
             "verified_tile_xyxy": None, "verified_source_xyxy": None},
        ]
        merged = merged_labels(row, supplementals)
        self.assertEqual(len(merged), 3)                     # 2 existing + 1 new
        first = merged[0]
        self.assertEqual(first["episode_ids"], ["ge-1"])
        self.assertEqual(first["supplemental_target_ids"], ["st-2"])
        self.assertEqual(len(merged[0]["merged_label_ids"]), 2)
        self.assertTrue(all(label["class_id"] == CLASS_ID for label in merged))
        for label in merged:
            for value in label["yolo_xywh_norm"]:
                self.assertTrue(0.0 <= value <= 1.0)

    def test_build_refuses_while_anything_is_pending(self) -> None:
        f = self.fixture(n_complete=1, n_missing=2, n_box_problem=0)
        data = f.load()
        rows = f.missing_rows(data)
        state = CompletionState.load(f.state)
        with self.assertRaises(CompletionError) as caught:
            build_accepted_v2(data, rows, f.out / "accepted_v2", state=state)
        self.assertIn("pending", str(caught.exception))
        self.assertFalse((f.out / "accepted_v2").exists())

    def test_build_outputs_frozen_plus_salvaged_with_identical_images(self) -> None:
        f = self.fixture(n_complete=2, n_missing=2, n_box_problem=1)
        data = f.load()
        rows = f.missing_rows(data)
        state = self._salvage(f, data, rows, 0)
        state.add_target(rows[1], point_tile=[200, 200],
                         proposal_result=FakeProposalEngine(fail=True).propose(
                             tile_path=Path(rows[1]["image_path"]),
                             point_tile_xy=[200, 200], size=TILE_SIZE))
        state.recheck(rows[1]["tile_id"], TILE_STATUS_STILL_MISSING, note="left one")
        accepted = build_accepted_v2(data, rows, f.out / "accepted_v2", state=state)
        self.assertEqual(accepted["frozen_tile_count"], 2)
        self.assertEqual(accepted["salvaged_tile_count"], 1)
        self.assertEqual(accepted["accepted_v2_tile_count"], 3)
        by_id = {row["tile_id"]: row for row in data["by_id"].values()}
        for row in accepted["rows"]:
            image = Path(row["image_path"])
            self.assertTrue(image.is_file())
            self.assertEqual(sha256_file(image), by_id[row["tile_id"]]["image_sha256"])
            label = Path(row["label_path"])
            lines = label.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), row["label_count"])
            for line in lines:
                parts = line.split()
                self.assertEqual(parts[0], str(CLASS_ID))
                for value in parts[1:]:
                    self.assertTrue(0.0 <= float(value) <= 1.0)
        salvaged = [row for row in accepted["rows"] if row["origin"] == "step1c2m_salvaged"]
        self.assertEqual(len(salvaged), 1)
        self.assertEqual(salvaged[0]["label_count"], 2)       # existing + supplemental

    def test_build_hard_fails_when_the_candidate_image_changed(self) -> None:
        f = self.fixture(n_complete=1, n_missing=1, n_box_problem=0)
        data = f.load()
        rows = f.missing_rows(data)
        state = self._salvage(f, data, rows, 0)
        Path(rows[0]["image_path"]).write_bytes(b"\x89PNG\r\n\x1a\nchanged")
        with self.assertRaises(CompletionError):
            build_accepted_v2(data, rows, f.out / "accepted_v2", state=state)

    def test_summary_and_manifest(self) -> None:
        f = self.fixture(n_complete=2, n_missing=2, n_box_problem=1)
        data = f.load()
        rows = f.missing_rows(data)
        state = self._salvage(f, data, rows, 0)
        state.recheck(rows[1]["tile_id"], TILE_STATUS_UNCERTAIN, note="unsure")
        accepted = build_accepted_v2(data, rows, f.out / "accepted_v2", state=state)
        preflight = verify_preflight(data)
        applied = apply_state(rows, state)
        summary = build_summary(data, rows, preflight, state=state,
                                accepted_v2=accepted,
                                proposal_stats=proposal_statistics(applied))
        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["generator_version"], GENERATOR_VERSION)
        self.assertEqual(summary["input"]["step1c2_missing_required"], 2)
        self.assertEqual(summary["input"]["old_accepted_positive_tile_count"], 2)
        self.assertEqual(summary["completion"]["status_counts"][TILE_STATUS_COMPLETE], 1)
        self.assertEqual(summary["completion"]["status_counts"][TILE_STATUS_UNCERTAIN], 1)
        self.assertEqual(summary["completion"]["pending"], 0)
        self.assertEqual(summary["supplemental"]["supplemental_target_count"], 1)
        self.assertEqual(summary["supplemental"]["verified_supplemental_bbox_count"], 1)
        self.assertEqual(summary["proposal"]["first_pass_success"], 1)
        self.assertEqual(summary["final"]["old_accepted_positive_tile_count"], 2)
        self.assertEqual(summary["final"]["salvaged_positive_tile_count"], 1)
        self.assertEqual(summary["final"]["accepted_v2_positive_tile_count"], 3)
        self.assertEqual(summary["final"]["accepted_v2_label_count"], 4)
        self.assertEqual(summary["final"]["tiles_with_2_labels"], 1)
        # only cameras with completion input or accepted_v2 output appear here
        self.assertEqual(set(summary["per_camera"]), {"01021", "01022"})
        manifest = build_manifest(
            data, summary, code_commit="0" * 40, generated_at="2026-09-23T00:00:00Z",
            config={"x": "y"}, artifact_root=f.out,
            provenance={"note": "n"})
        self.assertEqual(manifest["tile_size"], 640)
        self.assertTrue(manifest["image_pixels_immutable"])
        self.assertFalse(manifest["crop_moved"])
        self.assertFalse(manifest["resize"])
        self.assertEqual(manifest["class_mapping"], {"0": "ground_litter"})
        self.assertEqual(manifest["counts"]["completion_input_count"], 2)
        self.assertEqual(manifest["counts"]["accepted_v2_positive_tile_count"], 3)
        for value in summary["boundaries"].values():
            self.assertIs(value, False)

    def test_target_size_histogram_covers_all_labels(self) -> None:
        f = self.fixture(n_complete=1, n_missing=1, n_box_problem=0)
        data = f.load()
        rows = f.missing_rows(data)
        state = self._salvage(f, data, rows, 0, point=(300, 300))
        accepted = build_accepted_v2(data, rows, f.out / "accepted_v2", state=state)
        summary = build_summary(data, rows, verify_preflight(data), state=state,
                                accepted_v2=accepted)
        self.assertTrue(summary["target_size_histogram"])
        self.assertEqual(sum(summary["target_size_histogram"].values()), 3)


# --------------------------------------------------------------------------- #
# §39 Safety
# --------------------------------------------------------------------------- #


class SafetyTest(unittest.TestCase):
    def test_no_image_processing_or_learning_dependency(self) -> None:
        import ast

        source = (ROOT / "rtsp_annotator"
                  / "ground_litter_tile_completion.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for module in ("cv2", "torch", "torchvision", "ultralytics", "numpy"):
            self.assertNotIn(module, imported, module)
        lowered = source.lower()
        for forbidden in ("cv2.", "imwrite", "crop_tile", "model.train(",
                          "segment_anything", "sam2", "nn.module", "dataloader"):
            self.assertFalse(forbidden in lowered, f"forbidden token: {forbidden}")
        self.assertIn('"detector_run": False', source)
        self.assertIn('"hard_negatives_generated": False', source)

    def test_crop_is_never_touched(self) -> None:
        source = (ROOT / "rtsp_annotator"
                  / "ground_litter_tile_completion.py").read_text(encoding="utf-8")
        self.assertFalse("plan_crop" in source, "plan_crop must not be called here")
        self.assertFalse("def crop" in source, "no cropping code may live here")
        self.assertTrue('"crop_moved": False' in source,
                        "the crop_moved boundary must be declared False")
        self.assertTrue('"alternative_crop_created": False' in source,
                        "the alternative-crop boundary must be declared False")

    def test_cli_exposes_exactly_the_four_commands(self) -> None:
        source = CLI.read_text(encoding="utf-8")
        self.assertEqual(source.count("sub.add_parser("), 4)
        for command in ("plan", "serve", "status", "build"):
            self.assertIn(f'sub.add_parser("{command}")', source)
        self.assertNotIn("/api/proposal", source)

    def test_ui_never_draws_into_the_image_and_has_no_box_drawing(self) -> None:
        html = (TOOLS / "index.html").read_text(encoding="utf-8")
        app = (TOOLS / "app.js").read_text(encoding="utf-8")
        server = (TOOLS / "serve.py").read_text(encoding="utf-8")
        for forbidden in ("canvas", "toDataURL", "drawImage", "drag", "mousedown",
                          "resize_handle", "BOX_OK", "proposal_endpoint"):
            self.assertNotIn(forbidden.lower(), app.lower(), forbidden)
        self.assertIn("tilePointFromEvent", app)
        self.assertIn("bbox_tile_xyxy", app)
        self.assertIn('"image/png"', server)
        self.assertIn("image.read_bytes()", server)
        self.assertNotIn("cv2.imwrite", server)
        for status in FINAL_TILE_STATUSES:
            self.assertIn(status, app)
        self.assertIn("RECHECK COMPLETE", app)
        self.assertIn("只补", app)               # the IGNORE_SMALL guidance hint
        self.assertIn("新增遗漏 REQUIRED", app)
        self.assertIn("TARGET_TRUNCATED_BY_TILE", app)
        self.assertIn("POSSIBLE_DUPLICATE_TARGET", app)

    def test_app_js_references_only_ids_that_exist_in_the_page(self) -> None:
        app = (TOOLS / "app.js").read_text(encoding="utf-8")
        html = (TOOLS / "index.html").read_text(encoding="utf-8")
        used = set(re.findall(r'\$\("([^"]+)"\)', app))
        in_page = set(re.findall(r'id="([^"]+)"', html))
        rendered = set(re.findall(r'id="([^"]+)"', app))
        self.assertTrue(used)
        self.assertEqual(used - in_page - rendered, set())
        self.assertLessEqual({"f-status", "f-camera", "f-labels", "f-search", "queue",
                              "progress", "status-line", "meta", "detail"}, used)
        self.assertLessEqual({"tilewrap", "note"}, used)

    def test_proposal_module_keeps_cv2_lazy(self) -> None:
        import ast

        source = (ROOT / "rtsp_annotator"
                  / "ground_litter_point_proposal.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top.add(node.module)
        self.assertNotIn("cv2", top)
        self.assertIn("import cv2", source)      # imported inside the methods
        lowered = source.lower()
        for forbidden in ("segment_anything", "mobile_sam", "sam2", "model("):
            self.assertFalse(forbidden in lowered, f"forbidden token: {forbidden}")
        # proposals carry geometry and a method name only: no confidence / score field
        from rtsp_annotator.ground_litter_point_proposal import PointProposalEngine

        result = PointProposalEngine().propose(
            tile_path=Path("/tmp/not-needed.png"), point_tile_xy=[700, 700],
            size=TILE_SIZE)
        self.assertFalse(result["ok"])
        for candidate in result.get("candidates") or []:
            self.assertEqual(set(candidate) & {"confidence", "score", "prob"}, set())

    def test_no_sealed_access_and_no_upstream_writes(self) -> None:
        source = (ROOT / "rtsp_annotator"
                  / "ground_litter_tile_completion.py").read_text(encoding="utf-8")
        self.assertIn("assert_not_sealed", source)
        for forbidden in ("os.remove", "rmtree", "shutil", "unlink"):
            self.assertNotIn(forbidden, source, forbidden)


# --------------------------------------------------------------------------- #
# CLI end to end
# --------------------------------------------------------------------------- #


class CliTest(Base):
    def _run(self, *argv):
        import subprocess

        return subprocess.run([sys.executable, str(CLI), *argv], cwd=str(ROOT),
                              capture_output=True, text=True)

    def _args(self, f: Fixture, out: Path) -> list[str]:
        return ["--step1c2-root", str(f.step1c2), "--output", str(out),
                "--repo-root", str(ROOT)]

    def test_plan_status_and_build_refusal(self) -> None:
        f = self.fixture(n_complete=2, n_missing=3, n_box_problem=1)
        out = f.root / "artifact"
        plan = self._run(*self._args(f, out), "plan")
        self.assertEqual(plan.returncode, 0, plan.stderr)
        report = json.loads(plan.stdout)
        self.assertEqual(report["missing_required_input"], 3)
        self.assertEqual(report["old_accepted"], 2)
        self.assertEqual(report["box_problem_excluded"], 1)
        # a fixture that does not match the real 29/62/2 counts is reported, not hidden
        self.assertFalse(report["preflight"]["counts_match"])
        self.assertEqual([blocker["kind"] for blocker in report["blockers"]],
                         ["preflight_counts_mismatch"])
        self.assertTrue((out / "plan.json").is_file())
        self.assertFalse((out / "accepted_v2").exists())

        status = self._run(*self._args(f, out), "status")
        self.assertEqual(status.returncode, 0, status.stderr)
        payload = json.loads(status.stdout)
        self.assertEqual(payload["queue"], 3)
        self.assertEqual(payload["progress"]["pending"], 3)
        self.assertFalse(payload["build_allowed"])

        build = self._run(*self._args(f, out), "build")
        self.assertEqual(build.returncode, 4)
        self.assertIn("REFUSED", build.stderr)
        self.assertFalse((out / "accepted_v2").exists())

    def test_build_after_completing_the_queue(self) -> None:
        f = self.fixture(n_complete=2, n_missing=2, n_box_problem=1)
        out = f.root / "artifact"
        self._run(*self._args(f, out), "plan")
        data = f.load()
        rows = f.missing_rows(data)
        state = CompletionState.load(out / "completion_state.json")
        for index, row in enumerate(rows):
            point = (300 + index * 60, 400)
            target = state.add_target(row, point_tile=list(point),
                                      proposal_result=FakeProposalEngine(
                                          box=(point[0] - 20, point[1] - 15,
                                               point[0] + 20, point[1] + 15)).propose(
                                          tile_path=Path(row["image_path"]),
                                          point_tile_xy=list(point), size=TILE_SIZE))
            state.select_proposal(row["tile_id"], target["supplemental_target_id"], "A",
                                  existing_boxes=[])
            state.recheck(row["tile_id"], TILE_STATUS_COMPLETE, note="ok")
        build = self._run(*self._args(f, out), "build")
        self.assertEqual(build.returncode, 0, build.stderr)
        payload = json.loads(build.stdout)
        self.assertEqual(payload["old_accepted_positive_tile_count"], 2)
        self.assertEqual(payload["salvaged_positive_tile_count"], 2)
        self.assertEqual(payload["accepted_v2_positive_tile_count"], 4)
        self.assertEqual(payload["image_immutability"]["mismatch"], [])
        for name in ("SUMMARY.json", "MANIFEST.json", "tile_completion_manifest.jsonl",
                     "positive_training_manifest_v2.jsonl"):
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(len(list((out / "accepted_v2" / "images").glob("*.png"))), 4)
        self.assertEqual(len(list((out / "accepted_v2" / "labels").glob("*.txt"))), 4)
        manifest = json.loads((out / "MANIFEST.json").read_text())
        self.assertTrue(manifest["image_pixels_immutable"])
        self.assertFalse(manifest["crop_moved"])
        v2 = (out / "positive_training_manifest_v2.jsonl").read_text(encoding="utf-8")
        for row in data["accepted_rows"]:
            self.assertIn(row["tile_id"], v2)


if __name__ == "__main__":
    unittest.main()
