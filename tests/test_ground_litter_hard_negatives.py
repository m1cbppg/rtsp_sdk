"""Step 1D: hard negative pool tests (§38).

Pure stdlib + numpy; the decoder is injected, so no cv2 / network / PS is required here.
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
    SealedAssetError,
    sha256_file,
)
from rtsp_annotator.ground_litter_positive_tiles import (
    TILE_SIZE,
    png_bytes,
    write_jsonl,
)
from rtsp_annotator.ground_litter_hard_negatives import (
    CANDIDATE_STATUSES,
    GENERATOR_VERSION,
    MAX_EASY_FRACTION,
    ORIGIN_HISTORICAL,
    ORIGIN_RECONCILED,
    PENDING,
    READY,
    REVIEW_DECISIONS,
    RISK_IGNORE_SMALL,
    RISK_KNOWN_REQUIRED,
    RISK_NEARBY_REQUIRED,
    RISK_UNCERTAIN,
    SCHEMA_VERSION,
    STATUS_KNOWN_REQUIRED,
    STATUS_READY,
    HardNegativeError,
    NegativeReviewState,
    apply_review,
    build_accepted,
    build_manifest,
    build_summary,
    camera_from_device_code,
    candidate_fingerprint,
    check_positive_conflict,
    dedup_exact,
    generate_candidates,
    hardness_from_source,
    load_historical_input,
    load_truth_context,
    load_verified_required_boxes,
    locate_local_source,
    negative_tile_id,
    plan_candidates,
    verify_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "ground_litter_negative_ui"
CLI = ROOT / "scripts/build_ground_litter_hard_negatives.py"

W, H = 2560, 1440
RECORDING_START = "2026-09-18 10:00:00"


def synthetic_frame(width: int = W, height: int = H):
    import numpy as np

    ys, xs = np.mgrid[0:height, 0:width]
    return np.stack([(xs % 256), (ys % 256), ((xs + ys) % 256)], axis=2).astype("uint8")


class FakeDecoder:
    def __init__(self, *, width=W, height=H, mismatch=(), fail=()):
        self.width, self.height = width, height
        self.frame = synthetic_frame(width, height)
        self.mismatch = set(mismatch)
        self.fail = set(fail)
        self.decodes: list[tuple[str, float]] = []

    def probe_interval(self, path):                      # pragma: no cover
        return {"ok": True, "frame_interval_seconds": 0.04}

    def decode_frame(self, path, offset_seconds):
        offset = round(float(offset_seconds), 3)
        self.decodes.append((str(path), offset))
        if offset in self.fail:
            return {"ok": False, "error": "simulated_decode_failure"}
        achieved = offset + (5.0 if offset in self.mismatch else 0.0)
        return {"ok": True, "frame": self.frame, "width": self.width,
                "height": self.height, "decoded_offset_seconds": achieved}


def timestamp(offset_seconds: int) -> str:
    minutes, seconds = divmod(36000 + offset_seconds, 60)
    return f"2026-09-18 {minutes // 60:02d}:{minutes % 60:02d}:{seconds:02d}"


class Fixture:
    """A miniature but fully shaped Step 1A batch + Step 1C-0 source manifest."""

    def __init__(self, root: Path, *, cards=None) -> None:
        self.root = root
        self.batch = root / "batch_review"
        self.ps_dir = root / "raw_ps"
        self.out = root / "out"
        self.state = root / "state" / "review_state.json"
        self.batch.mkdir(parents=True, exist_ok=True)
        self.ps_dir.mkdir(parents=True, exist_ok=True)

        self.cards = cards if cards is not None else [
            ("01021", 30, [1200, 600, 1240, 640], "NON_LITTER", "semantic_tile"),
            ("01021", 90, [400, 400, 460, 450], "NON_LITTER", "random_grid"),
            ("01022", 130, [1800, 900, 1860, 950], "NON_LITTER", "texture"),
            ("01022", 200, [100, 100, 200, 200], "LITTER", "semantic_tile"),
            ("01027", 260, [1000, 500, 1060, 560], "NON_LITTER", "semantic_full"),
            ("01027", 320, [600, 700, 640, 740], "UNCERTAIN", "temporal"),
            ("01028", 380, [1500, 1000, 1560, 1050], "NON_LITTER", "semantic_tile"),
            ("01030", 440, [700, 300, 760, 360], "BOX_WRONG", "semantic_tile"),
        ]
        items, reviews = [], {}
        for index, (camera, offset, bbox, label, source) in enumerate(self.cards):
            review_id = f"{camera}-f{index:02d}s00-{index:012x}"
            moment = timestamp(offset)
            items.append({"review_id": review_id, "device_code": f"4418020903132200{camera}",
                          "timestamp": moment, "frame_id": f"f{index:02d}s00",
                          "bbox": bbox, "source": source,
                          "context_image": "assets/x-context.jpg"})
            reviews[review_id] = {"review_id": review_id, "label": label, "note": "",
                                  "reviewed_at": "2026-09-22T01:00:00Z",
                                  "source": source, "frame_id": f"f{index:02d}s00",
                                  "bbox": bbox}
        (self.batch / "review-data.json").write_text(
            json.dumps({"dataset": "fixture", "fingerprint": "fp", "count": len(items),
                        "labels": sorted({c[3] for c in self.cards}), "items": items}),
            encoding="utf-8")
        (self.batch / "reviews.json").write_text(
            json.dumps({"dataset": "fixture", "fingerprint": "fp", "reviews": reviews}),
            encoding="utf-8")

        self.ps_path = self.ps_dir / "ps-a.ps"
        self.ps_path.write_bytes(b"PS" + b"\x00" * 64)
        self.source_files = root / "source_files.jsonl"   # also the CLI's recovery root
        write_jsonl(self.source_files, [{
            "source_file_id": "ps-a", "camera_id": "01021", "local_ps_path": str(self.ps_path),
            "local_sha256": sha256_file(self.ps_path), "recording_start": RECORDING_START,
            "duration": 7200.0, "source_width": W, "source_height": H,
            "recovery_status": "PRESERVED"}]
            + [{"source_file_id": f"ps-{camera[-2:]}", "camera_id": camera,
                "local_ps_path": str(self.ps_path),
                "local_sha256": sha256_file(self.ps_path),
                "recording_start": RECORDING_START, "duration": 7200.0,
                "source_width": W, "source_height": H, "recovery_status": "PRESERVED"}
               for camera in ("01022", "01027", "01028", "01030")])

    # -- optional upstream artifacts ---------------------------------------- #
    def write_required_boxes(self, rows) -> Path:
        path = self.root / "tile_candidates.jsonl"
        write_jsonl(path, rows)
        return path

    def write_completion_manifest(self, rows) -> Path:
        path = self.root / "tile_completion_manifest.jsonl"
        write_jsonl(path, rows)
        return path

    def write_truth_context(self, overlay, localization) -> tuple[Path, Path]:
        overlay_path = self.root / "truth_reconciliation.jsonl"
        localization_path = self.root / "localizations.jsonl"
        write_jsonl(overlay_path, overlay)
        write_jsonl(localization_path, localization)
        return overlay_path, localization_path

    def load(self, **kwargs) -> dict:
        return load_historical_input([self.batch], source_files_path=self.source_files,
                                     **kwargs)


def required_row(tile_id, camera, source_file_id, moment, labels) -> dict:
    return {"tile_id": tile_id, "camera_id": camera, "source_file_id": source_file_id,
            "step1c0_decoded_timestamp": moment,
            "labels": [{"episode_ids": [f"ge-{tile_id}"],
                        "source_xyxy": [float(v) for v in box],
                        "tile_xyxy": [float(v) - 300 for v in box]}
                       for box in labels]}


def plan(data, *, required_by_frame=None, required_by_file=None, truth_context=None,
         hard_target=10, easy_target=4) -> dict:
    return plan_candidates(data, required_by_frame=required_by_frame or {},
                           required_by_file=required_by_file or {},
                           truth_context=truth_context or {},
                           hard_target=hard_target, easy_target=easy_target)


class Base(unittest.TestCase):
    def tmpdir(self) -> Path:
        path = Path(tempfile.mkdtemp(prefix="step1d-"))
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def fixture(self, **kwargs) -> Fixture:
        return Fixture(self.tmpdir(), **kwargs)

    def generate(self, f: Fixture, data, planned, **kwargs) -> dict:
        return generate_candidates(data, planned, FakeDecoder(),
                                   f.out / "candidate_tiles" / "images", **kwargs)


# --------------------------------------------------------------------------- #
# §38 Eligibility
# --------------------------------------------------------------------------- #


class EligibilityTest(Base):
    def test_only_human_non_litter_cards_become_candidates(self) -> None:
        f = self.fixture()
        data = f.load()
        labels = {card["historical_label"] for card in data["cards"]}
        self.assertEqual(labels, {"NON_LITTER"})
        self.assertEqual(data["historical_non_litter_count"], 5)
        self.assertEqual(data["label_counts"]["LITTER"], 1)
        self.assertEqual(data["label_counts"]["UNCERTAIN"], 1)
        self.assertEqual(data["label_counts"]["BOX_WRONG"], 1)
        planned = plan(data)
        for row in planned["candidates"]:
            self.assertEqual(row["historical_label"], "NON_LITTER")

    def test_reconciled_non_litter_is_an_extra_source(self) -> None:
        f = self.fixture()
        overlay, localization = f.write_truth_context(
            [{"episode_id": "ge-x", "camera_id": "01021", "effective_truth_bucket":
              "NON_LITTER", "effective_truth_class": "REQUIRED_LITTER",
              "origin": "box_wrong", "reconciliation_decision": "NON_LITTER"}],
            [{"episode_id": "ge-x", "original_bbox": [500.0, 500.0, 560.0, 560.0]}])
        data = load_historical_input(
            [f.batch], source_files_path=f.source_files,
            reconciled_overlay_path=overlay, localization_path=localization)
        self.assertEqual(data["reconciled_non_litter_count"], 1)
        origins = {card["origin"] for card in data["cards"]}
        self.assertEqual(origins, {ORIGIN_HISTORICAL, ORIGIN_RECONCILED})

    def test_sealed_paths_are_refused(self) -> None:
        f = self.fixture()
        with self.assertRaises(SealedAssetError):
            load_historical_input([f.root / "SEALED_DO_NOT_TUNE"],
                                  source_files_path=f.source_files)

    def test_hardness_and_camera_helpers(self) -> None:
        self.assertEqual(hardness_from_source("semantic_tile_low"),
                         "historical_false_positive")
        self.assertEqual(hardness_from_source("random_grid"), "random_grid_background")
        self.assertEqual(hardness_from_source("texture"), "surface_texture")
        self.assertEqual(hardness_from_source("something_else"), "other")
        self.assertEqual(camera_from_device_code("44180209031322001030"), "01030")


# --------------------------------------------------------------------------- #
# §38 Source
# --------------------------------------------------------------------------- #


class SourceTest(Base):
    def test_camera_and_timestamp_resolves_to_a_local_ps(self) -> None:
        f = self.fixture()
        data = f.load()
        recoverable = [card for card in data["cards"] if card["recoverable"]]
        self.assertTrue(recoverable)
        for card in recoverable:
            self.assertTrue(Path(card["local_ps_path"]).is_file())
            self.assertEqual(card["source_file_id"], "ps-a" if card["camera_id"] == "01021"
                             else f"ps-{card['camera_id'][-2:]}")
            self.assertGreater(card["offset_seconds"], 0)

    def test_a_card_outside_every_window_is_not_recoverable(self) -> None:
        f = self.fixture(cards=[("01021", 100000, [10, 10, 40, 40], "NON_LITTER",
                                 "semantic_tile")])
        data = f.load()
        self.assertEqual(data["recoverable_count"], 0)
        planned = plan(data)
        self.assertEqual(planned["candidates"], [])

    def test_locate_local_source_matches_the_recording_window(self) -> None:
        f = self.fixture()
        sources = {row["source_file_id"]: row
                   for row in (json.loads(line) for line in
                               f.source_files.read_text().splitlines())}
        self.assertEqual(locate_local_source(sources, "01021", timestamp(30))["source_file_id"],
                         "ps-a")
        self.assertIsNone(locate_local_source(sources, "01021", "2026-09-18 23:00:00"))
        self.assertIsNone(locate_local_source(sources, "01099", timestamp(30)))

    def test_reuse_only_no_download_code_path(self) -> None:
        source = (ROOT / "rtsp_annotator"
                  / "ground_litter_hard_negatives.py").read_text(encoding="utf-8")
        for forbidden in ("import requests", "import urllib", "http.client", "ctseelink",
                          "urlopen", "http://", "https://"):
            self.assertFalse(forbidden in source.lower(), forbidden)
        self.assertIn("local_ps_path", source)
        self.assertIn("locate_local_source", source)


# --------------------------------------------------------------------------- #
# §38 Crop
# --------------------------------------------------------------------------- #


class CropTest(Base):
    def test_crop_is_640_source_native_and_keeps_the_anchor(self) -> None:
        f = self.fixture()
        data = f.load()
        planned = plan(data)
        generated = self.generate(f, data, planned)
        ready = [row for row in generated["candidates"]
                 if row["candidate_generation_status"] == STATUS_READY]
        self.assertTrue(ready)
        frame = synthetic_frame()
        for row in ready:
            crop = row["source_crop_xyxy"]
            self.assertEqual(crop[2] - crop[0], TILE_SIZE)
            self.assertEqual(crop[3] - crop[1], TILE_SIZE)
            anchor = row["anchor_source_xyxy"]
            self.assertLessEqual(crop[0], anchor[0])
            self.assertLessEqual(crop[1], anchor[1])
            self.assertGreaterEqual(crop[2], anchor[2])
            self.assertGreaterEqual(crop[3], anchor[3])
            # the tile is a pure pixel slice: no resize, no resample
            from rtsp_annotator.ground_litter_positive_tiles import png_bytes as encode

            payload = Path(row["image_path"]).read_bytes()
            import zlib

            scan = zlib.decompress(payload[payload.index(b"IDAT") + 4:
                                           payload.index(b"IEND") - 4])
            width = int.from_bytes(payload[16:20], "big")
            self.assertEqual(width, TILE_SIZE)
            row_bytes = scan[1:1 + TILE_SIZE * 3]
            expected = frame[crop[1], crop[0]:crop[0] + TILE_SIZE][:, ::-1].tobytes()
            self.assertEqual(row_bytes, expected, "first tile row must equal the source row")
            self.assertEqual(width, TILE_SIZE)

    def test_crop_clamps_at_the_frame_edges(self) -> None:
        f = self.fixture(cards=[
            ("01021", 30, [5, 5, 40, 40], "NON_LITTER", "semantic_tile"),
            ("01021", 60, [W - 40, 5, W - 5, 40], "NON_LITTER", "semantic_tile"),
            ("01021", 90, [5, H - 40, 40, H - 5], "NON_LITTER", "semantic_tile"),
            ("01021", 120, [W - 40, H - 40, W - 5, H - 5], "NON_LITTER",
             "semantic_tile")])
        data = f.load()
        planned = plan(data, hard_target=10, easy_target=0)
        generated = self.generate(f, data, planned)
        for row in generated["candidates"]:
            crop = row["source_crop_xyxy"]
            self.assertEqual(crop[2] - crop[0], TILE_SIZE)
            self.assertGreaterEqual(crop[0], 0)
            self.assertLessEqual(crop[2], W)
            self.assertGreaterEqual(crop[1], 0)
            self.assertLessEqual(crop[3], H)

    def test_anchor_larger_than_the_tile_is_excluded(self) -> None:
        f = self.fixture(cards=[("01021", 30, [100, 100, 100 + 700, 300], "NON_LITTER",
                                 "semantic_tile")])
        data = f.load()
        planned = plan(data)
        statuses = {row["candidate_generation_status"] for row in planned["candidates"]}
        self.assertIn("ANCHOR_EXCEEDS_TILE", statuses)
        self.assertEqual(planned["status_counts"]["ANCHOR_EXCEEDS_TILE"], 1)


# --------------------------------------------------------------------------- #
# §38 Negative completeness
# --------------------------------------------------------------------------- #


class CompletenessTest(Base):
    def _ready(self):
        f = self.fixture(cards=[
            ("01021", 30, [1200, 600, 1240, 640], "NON_LITTER", "semantic_tile"),
            ("01021", 90, [400, 400, 460, 450], "NON_LITTER", "semantic_tile")])
        data = f.load()
        planned = plan(data, hard_target=5, easy_target=0)
        generated = self.generate(f, data, planned)
        rows = [row for row in generated["candidates"]
                if row["candidate_generation_status"] == STATUS_READY]
        self.assertEqual(len(rows), 2)
        return f, data, rows

    def test_only_negative_ok_is_ready(self) -> None:
        f, data, rows = self._ready()
        state = NegativeReviewState.load(f.state, candidate_count=len(rows))
        for decision in REVIEW_DECISIONS:
            state.decide(rows[0], decision, reason="because")
            applied = [row for row in apply_review(rows, state)
                       if row["negative_tile_id"] == rows[0]["negative_tile_id"]][0]
            self.assertEqual(applied["review_status"], decision)
            self.assertEqual(applied["hard_negative_ready"], decision == READY)

    def test_unreviewed_stays_pending_and_not_ready(self) -> None:
        f, data, rows = self._ready()
        state = NegativeReviewState.load(f.state, candidate_count=len(rows))
        for row in apply_review(rows, state):
            self.assertEqual(row["review_status"], PENDING)
            self.assertFalse(row["hard_negative_ready"])
        self.assertEqual(state.progress(rows),
                         {"reviewable": 2, "reviewed": 0, "pending": 2, "skipped": 0})

    def test_reason_is_required_for_every_non_negative_decision(self) -> None:
        f, data, rows = self._ready()
        state = NegativeReviewState.load(f.state, candidate_count=len(rows))
        for decision in ("REQUIRED_PRESENT", "UNCERTAIN", "BAD_CROP"):
            with self.assertRaises(ReviewError):
                state.decide(rows[0], decision)
        state.decide(rows[0], READY)                    # no reason needed
        self.assertTrue(state.get(rows[0]["negative_tile_id"])["hard_negative_ready"])

    def test_unknown_decision_is_rejected(self) -> None:
        f, data, rows = self._ready()
        state = NegativeReviewState.load(f.state, candidate_count=len(rows))
        with self.assertRaises(ReviewError):
            state.decide(rows[0], "NEGATIVE")

    def test_resume_keeps_decisions_and_audit(self) -> None:
        f, data, rows = self._ready()
        state = NegativeReviewState.load(f.state, candidate_count=len(rows))
        state.decide(rows[0], READY)
        state.decide(rows[1], "REQUIRED_PRESENT", reason="a bag is visible")
        restarted = NegativeReviewState.load(f.state, candidate_count=len(rows))
        self.assertEqual(restarted.progress(rows)["reviewed"], 2)
        self.assertTrue(restarted.get(rows[0]["negative_tile_id"])["hard_negative_ready"])
        self.assertFalse(restarted.get(rows[1]["negative_tile_id"])["hard_negative_ready"])
        self.assertEqual(len(restarted.audit_trail), 2)
        restarted.reset(rows[0]["negative_tile_id"])
        self.assertEqual(restarted.audit_trail[-1]["action"], "reset")

    def test_state_from_a_different_candidate_set_is_refused(self) -> None:
        f, data, rows = self._ready()
        NegativeReviewState.load(f.state, candidate_count=len(rows),
                                 input_fingerprint="a").save()
        with self.assertRaises(ReviewError):
            NegativeReviewState.load(f.state, input_fingerprint="b")


# --------------------------------------------------------------------------- #
# §38 Known Required overlap / IGNORE_SMALL / UNCERTAIN
# --------------------------------------------------------------------------- #


class OverlapTest(Base):
    def test_verified_same_frame_required_intersecting_the_crop_excludes(self) -> None:
        f = self.fixture(cards=[("01021", 30, [1200, 600, 1240, 640], "NON_LITTER",
                                 "semantic_tile")])
        data = f.load()
        row = required_row("pt-x", "01021", "ps-a", timestamp(30),
                           [[1210.0, 610.0, 1250.0, 650.0]])
        by_frame, by_file = load_verified_required_boxes(
            tile_candidates_path=f.write_required_boxes([row]))
        planned = plan(data, required_by_frame=by_frame, required_by_file=by_file)
        candidate = planned["candidates"][0]
        self.assertEqual(candidate["candidate_generation_status"], STATUS_KNOWN_REQUIRED)
        self.assertIn(RISK_KNOWN_REQUIRED, candidate["risk_flags"])
        self.assertEqual(planned["status_counts"][STATUS_KNOWN_REQUIRED], 1)

    def test_verified_required_in_another_frame_does_not_exclude(self) -> None:
        f = self.fixture(cards=[("01021", 30, [1200, 600, 1240, 640], "NON_LITTER",
                                 "semantic_tile")])
        data = f.load()
        row = required_row("pt-x", "01021", "ps-a", timestamp(300),
                           [[1210.0, 610.0, 1250.0, 650.0]])
        by_frame, by_file = load_verified_required_boxes(
            tile_candidates_path=f.write_required_boxes([row]))
        candidate = plan(data, required_by_frame=by_frame,
                         required_by_file=by_file)["candidates"][0]
        self.assertEqual(candidate["candidate_generation_status"], STATUS_READY)
        self.assertNotIn(RISK_KNOWN_REQUIRED, candidate["risk_flags"])

    def test_nearby_other_frame_required_raises_a_warning_only(self) -> None:
        f = self.fixture(cards=[("01021", 30, [1200, 600, 1240, 640], "NON_LITTER",
                                 "semantic_tile")])
        data = f.load()
        row = required_row("pt-x", "01021", "ps-a", timestamp(45),
                           [[1210.0, 610.0, 1250.0, 650.0]])
        by_frame, by_file = load_verified_required_boxes(
            tile_candidates_path=f.write_required_boxes([row]))
        candidate = plan(data, required_by_frame=by_frame,
                         required_by_file=by_file)["candidates"][0]
        self.assertEqual(candidate["candidate_generation_status"], STATUS_READY)
        self.assertIn(RISK_NEARBY_REQUIRED, candidate["risk_flags"])
        self.assertFalse(candidate["known_required_boxes"] == [])

    def test_supplemental_verified_boxes_are_part_of_the_required_set(self) -> None:
        f = self.fixture(cards=[("01021", 30, [1200, 600, 1240, 640], "NON_LITTER",
                                 "semantic_tile")])
        f.write_required_boxes([])
        completion = f.write_completion_manifest([{
            "tile_id": "pt-y", "camera_id": "01021", "source_file_id": "ps-a",
            "step1c0_decoded_timestamp": timestamp(30),
            "supplemental_targets": [{"supplemental_target_id": "st-1",
                                      "localization_status": "VERIFIED_BBOX",
                                      "verified_source_xyxy": [1215.0, 615.0, 1255.0, 655.0]}]}])
        by_frame, by_file = load_verified_required_boxes(
            tile_candidates_path=f.root / "tile_candidates.jsonl",
            completion_manifest_path=completion)
        self.assertTrue(by_frame)

    def test_ignore_small_does_not_block_but_is_flagged(self) -> None:
        f = self.fixture(cards=[("01021", 30, [1200, 600, 1240, 640], "NON_LITTER",
                                 "semantic_tile")])
        overlay, localization = f.write_truth_context(
            [{"episode_id": "ge-small", "camera_id": "01021",
              "effective_truth_bucket": "IGNORE_SMALL"}],
            [{"episode_id": "ge-small", "original_bbox": [1210.0, 610.0, 1220.0, 620.0]}])
        context = load_truth_context(overlay_path=overlay, localization_path=localization)
        candidate = plan(f.load(), truth_context=context)["candidates"][0]
        self.assertEqual(candidate["candidate_generation_status"], STATUS_READY)
        self.assertIn(RISK_IGNORE_SMALL, candidate["risk_flags"])
        self.assertEqual(candidate["ignore_small_ids"], ["ge-small"])

    def test_non_litter_context_does_not_block(self) -> None:
        f = self.fixture(cards=[("01021", 30, [1200, 600, 1240, 640], "NON_LITTER",
                                 "semantic_tile")])
        overlay, localization = f.write_truth_context(
            [{"episode_id": "ge-nl", "camera_id": "01021",
              "effective_truth_bucket": "NON_LITTER"}],
            [{"episode_id": "ge-nl", "original_bbox": [1210.0, 610.0, 1220.0, 620.0]}])
        context = load_truth_context(overlay_path=overlay, localization_path=localization)
        candidate = plan(f.load(), truth_context=context)["candidates"][0]
        self.assertEqual(candidate["candidate_generation_status"], STATUS_READY)
        self.assertEqual(candidate["risk_flags"], [])

    def test_uncertain_context_is_a_warning_not_an_exclusion(self) -> None:
        f = self.fixture(cards=[("01021", 30, [1200, 600, 1240, 640], "NON_LITTER",
                                 "semantic_tile")])
        overlay, localization = f.write_truth_context(
            [{"episode_id": "ge-unc", "camera_id": "01021",
              "effective_truth_bucket": "UNCERTAIN"}],
            [{"episode_id": "ge-unc", "original_bbox": [1210.0, 610.0, 1220.0, 620.0]}])
        context = load_truth_context(overlay_path=overlay, localization_path=localization)
        candidate = plan(f.load(), truth_context=context)["candidates"][0]
        self.assertEqual(candidate["candidate_generation_status"], STATUS_READY)
        self.assertIn(RISK_UNCERTAIN, candidate["risk_flags"])
        self.assertEqual(candidate["uncertain_truth_ids"], ["ge-unc"])

    def test_mapped_point_context_is_used_when_there_is_no_box(self) -> None:
        f = self.fixture(cards=[("01021", 30, [1200, 600, 1240, 640], "NON_LITTER",
                                 "semantic_tile")])
        overlay, localization = f.write_truth_context(
            [{"episode_id": "ge-p", "camera_id": "01021",
              "effective_truth_bucket": "UNCERTAIN"}],
            [{"episode_id": "ge-p", "original_bbox": None,
              "original_point_source": {"ok": True, "x": 1220.0, "y": 620.0}}])
        context = load_truth_context(overlay_path=overlay, localization_path=localization)
        candidate = plan(f.load(), truth_context=context)["candidates"][0]
        self.assertIn(RISK_UNCERTAIN, candidate["risk_flags"])


# --------------------------------------------------------------------------- #
# §38 Duplicate / sampling
# --------------------------------------------------------------------------- #


class SamplingTest(Base):
    def test_conservative_spatial_temporal_dedup(self) -> None:
        # four cards a few seconds apart on the same object: only a couple survive
        cards = [("01021", 30 + index * 5, [1200, 600, 1240, 640], "NON_LITTER",
                  "semantic_tile") for index in range(4)]
        f = self.fixture(cards=cards)
        planned = plan(f.load(), hard_target=20, easy_target=0)
        self.assertLessEqual(len(planned["candidates"]), 2)
        self.assertEqual(planned["candidates"][0]["camera_id"], "01021")

    def test_distinct_objects_in_the_same_minute_are_kept(self) -> None:
        cards = [("01021", 30, [300, 300, 340, 340], "NON_LITTER", "semantic_tile"),
                 ("01021", 32, [1500, 900, 1540, 940], "NON_LITTER", "semantic_tile")]
        f = self.fixture(cards=cards)
        planned = plan(f.load(), hard_target=20, easy_target=0)
        self.assertEqual(len(planned["candidates"]), 2)

    def test_multiple_objects_in_a_burst_are_capped(self) -> None:
        cards = [("01021", 30, [200 + index * 400, 200, 240 + index * 400, 240],
                  "NON_LITTER", "semantic_tile") for index in range(5)]
        f = self.fixture(cards=cards)
        planned = plan(f.load(), hard_target=20, easy_target=0)
        self.assertLessEqual(len(planned["candidates"]), 2)

    def test_easy_background_is_a_bounded_supplement(self) -> None:
        cards = [("01021", 30 + index * 600, [200 + index * 300, 200,
                                              240 + index * 300, 240],
                  "NON_LITTER", "random_grid") for index in range(10)]
        f = self.fixture(cards=cards)
        planned = plan(f.load(), hard_target=20, easy_target=10)
        self.assertEqual(planned["sampled_hard_count"], 0)
        self.assertEqual(planned["sampled_easy_count"], 0)     # no hard pool -> no easy

    def test_easy_share_stays_under_the_cap(self) -> None:
        hard = [("01021", 30 + index * 600, [200 + index * 300, 200,
                                             240 + index * 300, 240],
                 "NON_LITTER", "semantic_tile") for index in range(10)]
        easy = [("01021", 60 + index * 600, [100 + index * 300, 800,
                                             140 + index * 300, 840],
                 "NON_LITTER", "random_grid") for index in range(10)]
        f = self.fixture(cards=hard + easy)
        planned = plan(f.load(), hard_target=10, easy_target=10)
        self.assertEqual(planned["sampled_hard_count"], 10)
        self.assertLessEqual(planned["sampled_easy_count"],
                             int(10 * MAX_EASY_FRACTION / (1 - MAX_EASY_FRACTION)))

    def test_candidate_id_is_deterministic(self) -> None:
        one = negative_tile_id("01021", "ps-a", timestamp(30), [900, 400, 1540, 1040], "c1")
        again = negative_tile_id("01021", "ps-a", timestamp(30), [900, 400, 1540, 1040], "c1")
        other = negative_tile_id("01021", "ps-a", timestamp(30), [0, 0, 640, 640], "c1")
        self.assertEqual(one, again)
        self.assertNotEqual(one, other)
        self.assertTrue(one.startswith("hn-"))

    def test_exact_duplicate_dedup_only(self) -> None:
        base = {"negative_tile_id": "hn-a", "image_sha256": "sha", "source_file_id": "ps",
                "timestamp": "t", "decoded_timestamp": "t", "source_crop_xyxy": [0, 0, 640, 640],
                "camera_id": "01021", "source_card_id": "c1"}
        same = dict(base, negative_tile_id="hn-b", source_card_id="c2")
        other_crop = dict(base, negative_tile_id="hn-c", source_crop_xyxy=[1, 1, 641, 641])
        kept, removed = dedup_exact([base, same, other_crop])
        self.assertEqual(len(kept), 2)
        self.assertEqual(removed, 1)

    def test_generate_is_idempotent(self) -> None:
        f = self.fixture()
        data = f.load()
        planned = plan(data)
        first = self.generate(f, data, planned)
        decoder = FakeDecoder()
        second = generate_candidates(
            data, planned, decoder, f.out / "candidate_tiles" / "images",
            existing=first["candidates"], input_fingerprint="fp",
            existing_fingerprint="fp")
        self.assertEqual(decoder.decodes, [])
        self.assertEqual([row["negative_tile_id"] for row in first["candidates"]],
                         [row["negative_tile_id"] for row in second["candidates"]])
        for before, after in zip(first["candidates"], second["candidates"]):
            self.assertEqual(before["image_sha256"], after["image_sha256"])

    def test_changed_inputs_refuse_to_overwrite(self) -> None:
        f = self.fixture()
        data = f.load()
        planned = plan(data)
        first = self.generate(f, data, planned)
        with self.assertRaises(HardNegativeError):
            generate_candidates(data, planned, FakeDecoder(),
                                f.out / "candidate_tiles" / "images",
                                existing=first["candidates"],
                                input_fingerprint="new", existing_fingerprint="old")


# --------------------------------------------------------------------------- #
# §38 Build / label / positive conflict
# --------------------------------------------------------------------------- #


class BuildTest(Base):
    def _reviewed(self, decisions):
        f = self.fixture(cards=[
            ("01021", 30, [1200, 600, 1240, 640], "NON_LITTER", "semantic_tile"),
            ("01021", 90, [400, 400, 460, 450], "NON_LITTER", "semantic_tile"),
            ("01022", 130, [1800, 900, 1860, 950], "NON_LITTER", "texture")])
        data = f.load()
        planned = plan(data, hard_target=5, easy_target=0)
        generated = self.generate(f, data, planned)
        rows = [row for row in generated["candidates"]
                if row["candidate_generation_status"] == STATUS_READY]
        state = NegativeReviewState.load(f.state, candidate_count=len(rows))
        for row, decision in zip(rows, decisions):
            state.decide(row, decision, reason="because" if decision != READY else "")
        return f, data, rows, state

    def test_build_refuses_while_anything_is_pending(self) -> None:
        f, data, rows, state = self._reviewed([READY, READY, READY])
        state.reset(rows[0]["negative_tile_id"])
        with self.assertRaises(HardNegativeError) as caught:
            build_accepted(rows, f.out / "accepted", state=state)
        self.assertIn("pending=1", str(caught.exception))
        self.assertFalse((f.out / "accepted").exists())

    def test_build_refuses_when_a_tile_was_skipped(self) -> None:
        f, data, rows, state = self._reviewed([READY, READY, READY])
        state.skip(rows[0]["negative_tile_id"])
        with self.assertRaises(HardNegativeError) as caught:
            build_accepted(rows, f.out / "accepted", state=state)
        self.assertIn("skipped=1", str(caught.exception))

    def test_build_writes_images_and_empty_labels(self) -> None:
        f, data, rows, state = self._reviewed([READY, "REQUIRED_PRESENT", READY])
        accepted = build_accepted(rows, f.out / "accepted", state=state)
        self.assertEqual(accepted["accepted_tile_count"], 2)
        self.assertEqual(accepted["accepted_label_count"], 0)
        by_id = {row["negative_tile_id"]: row for row in rows}
        for row in accepted["rows"]:
            image = Path(row["image_path"])
            label = Path(row["label_path"])
            self.assertEqual(sha256_file(image), by_id[row["negative_tile_id"]]["image_sha256"])
            self.assertEqual(label.read_bytes(), b"")
            self.assertEqual(label.stat().st_size, 0)
            self.assertEqual(row["label_bytes"], 0)
            self.assertEqual(row["label_line_count"], 0)

    def test_hard_fail_when_a_candidate_image_drifted(self) -> None:
        f, data, rows, state = self._reviewed([READY, READY, READY])
        Path(rows[0]["image_path"]).write_bytes(b"\x89PNG\r\n\x1a\nchanged")
        with self.assertRaises(HardNegativeError):
            build_accepted(rows, f.out / "accepted", state=state)

    def test_positive_sha_collision_is_a_hard_fail(self) -> None:
        f, data, rows, state = self._reviewed([READY, READY, READY])
        sha = rows[0]["image_sha256"]
        positive = [{"tile_id": "pt-positive", "image_sha256": sha}]
        with self.assertRaises(HardNegativeError) as caught:
            build_accepted(rows, f.out / "accepted", state=state,
                           positive_rows=positive)
        self.assertIn("conflict", str(caught.exception))

    def test_same_source_crop_conflict_is_a_hard_fail(self) -> None:
        f, data, rows, state = self._reviewed([READY, READY, READY])
        tile_candidates = f.write_required_boxes([{
            "tile_id": "pt-positive", "source_file_id": rows[0]["source_file_id"],
            "source_crop_xyxy": rows[0]["source_crop_xyxy"], "labels": []}])
        reviewed = apply_review(rows, state)
        conflict = check_positive_conflict(reviewed, [], tile_candidates)
        self.assertFalse(conflict["ok"])
        self.assertEqual(conflict["crop_conflict_count"], 1)
        with self.assertRaises(HardNegativeError):
            build_accepted(rows, f.out / "accepted", state=state,
                           positive_tiles_path=tile_candidates)

    def test_no_conflict_for_independent_tiles(self) -> None:
        f, data, rows, state = self._reviewed([READY, READY, READY])
        tile_candidates = f.write_required_boxes([{
            "tile_id": "pt-positive", "source_file_id": "ps-other",
            "source_crop_xyxy": [0, 0, 640, 640], "labels": []}])
        conflict = check_positive_conflict(apply_review(rows, state), [], tile_candidates)
        self.assertTrue(conflict["ok"], json.dumps(conflict))


# --------------------------------------------------------------------------- #
# §38 Summary / manifest / empty label
# --------------------------------------------------------------------------- #


class SummaryTest(Base):
    def test_summary_reports_generate_and_review_state(self) -> None:
        f = self.fixture()
        data = f.load()
        planned = plan(data, hard_target=5, easy_target=2)
        generated = self.generate(f, data, planned)
        preflight = verify_preflight(data, planned)
        preflight["exact_duplicate_removed"] = generated["stats"]["exact_duplicate_removed"]
        state = NegativeReviewState.load(f.state,
                                         candidate_count=len(generated["candidates"]))
        summary = build_summary(data, generated["candidates"], preflight, state=state)
        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["generator_version"], GENERATOR_VERSION)
        self.assertEqual(summary["tile_size"], TILE_SIZE)
        self.assertEqual(summary["class_mapping"], {"0": "ground_litter"})
        self.assertEqual(summary["negative_label"], "empty_file")
        self.assertEqual(summary["input"]["historical_non_litter_input_count"], 5)
        self.assertEqual(summary["review"]["reviewed"], 0)
        self.assertEqual(summary["review"]["pending"], summary["review"]["reviewable"])
        self.assertTrue(summary["review"]["reviewable"] > 0)
        self.assertEqual(summary["accepted"]["state"], "not_built")
        self.assertEqual(summary["accepted"]["accepted_hard_negative_tile_count"], 0)
        for value in summary["boundaries"].values():
            self.assertIs(value, False)

    def test_summary_after_build_reports_accepted_and_conflicts(self) -> None:
        f = self.fixture(cards=[
            ("01021", 30, [1200, 600, 1240, 640], "NON_LITTER", "semantic_tile"),
            ("01022", 90, [400, 400, 460, 450], "NON_LITTER", "semantic_tile")])
        data = f.load()
        planned = plan(data, hard_target=5, easy_target=0)
        generated = self.generate(f, data, planned)
        rows = [row for row in generated["candidates"]
                if row["candidate_generation_status"] == STATUS_READY]
        state = NegativeReviewState.load(f.state, candidate_count=len(rows))
        state.decide(rows[0], READY)
        state.decide(rows[1], "UNCERTAIN", reason="cannot tell")
        accepted = build_accepted(rows, f.out / "accepted", state=state)
        summary = build_summary(data, rows, verify_preflight(data, planned), state=state,
                                accepted=accepted)
        self.assertEqual(summary["review"]["counts"][READY], 1)
        self.assertEqual(summary["review"]["counts"]["UNCERTAIN"], 1)
        self.assertEqual(summary["accepted"]["accepted_hard_negative_tile_count"], 1)
        self.assertEqual(summary["accepted"]["positive_conflict_count"], 0)
        self.assertEqual(summary["accepted"]["empty_label_verified_count"], 1)

    def test_manifest_records_provenance_and_flags(self) -> None:
        f = self.fixture()
        data = f.load()
        planned = plan(data, hard_target=3, easy_target=0)
        generated = self.generate(f, data, planned)
        state = NegativeReviewState.load(f.state,
                                         candidate_count=len(generated["candidates"]))
        summary = build_summary(data, generated["candidates"],
                                verify_preflight(data, planned), state=state)
        manifest = build_manifest(
            data, summary, code_commit="0" * 40, generated_at="2026-09-23T00:00:00Z",
            config={}, artifact_root=f.out, provenance={"note": "n"},
            positive_pool_manifest_sha256="a", positive_pool_summary_sha256="b")
        self.assertEqual(manifest["tile_size"], TILE_SIZE)
        self.assertTrue(manifest["source_native"])
        self.assertFalse(manifest["resize"])
        self.assertEqual(manifest["class_mapping"], {"0": "ground_litter"})
        self.assertEqual(manifest["negative_label"], {"format": "empty_txt", "bytes": 0})
        self.assertEqual(manifest["positive_pool_manifest_sha256"], "a")
        self.assertTrue(manifest["historical_review_sha256"])
        self.assertTrue(manifest["source_recovery_sha256"]["source_files_sha256"])
        for value in manifest["boundaries"].values():
            self.assertIs(value, False)

    def test_fingerprint_covers_inputs_and_extra_material(self) -> None:
        f = self.fixture()
        data = f.load()
        self.assertNotEqual(candidate_fingerprint(data),
                            candidate_fingerprint(data, extra_material=["x"]))


class EmptyLabelTest(unittest.TestCase):
    def test_empty_label_is_a_valid_background_sample(self) -> None:
        check = _empty_label_check()
        self.assertEqual(check["empty_label_bytes"], 0)
        self.assertEqual(check["parsed_label_rows"], 0)
        self.assertFalse(check["validation_block_entered"])
        self.assertTrue(check["empty_label_is_background"])
        if check.get("available"):
            self.assertTrue(check["guard_present"], "ultralytics parser guard changed")

    def test_accepted_label_is_written_as_zero_bytes(self) -> None:
        f = Fixture(Path(tempfile.mkdtemp(prefix="step1d-lbl-")))
        self.addCleanup(shutil.rmtree, f.root, True)
        data = f.load()
        planned = plan(data, hard_target=1, easy_target=0)
        generated = generate_candidates(data, planned, FakeDecoder(),
                                        f.out / "candidate_tiles" / "images")
        rows = [row for row in generated["candidates"]
                if row["candidate_generation_status"] == STATUS_READY][:1]
        state = NegativeReviewState.load(f.state, candidate_count=len(rows))
        state.decide(rows[0], READY)
        accepted = build_accepted(rows, f.out / "accepted", state=state)
        label = Path(accepted["rows"][0]["label_path"])
        self.assertEqual(label.stat().st_size, 0)
        self.assertEqual(label.read_bytes(), b"")
        self.assertEqual(label.name, f"{rows[0]['negative_tile_id']}.txt")


def _empty_label_check() -> dict:
    """Reuse the CLI's own checker so the test and the artifact agree."""
    spec = importlib.util.spec_from_file_location("build_hard_negatives_cli", CLI)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._empty_label_check()


# --------------------------------------------------------------------------- #
# §38 Safety + UI + CLI
# --------------------------------------------------------------------------- #


class SafetyTest(unittest.TestCase):
    def test_no_model_or_image_stack_dependency(self) -> None:
        import ast

        source = (ROOT / "rtsp_annotator"
                  / "ground_litter_hard_negatives.py").read_text(encoding="utf-8")
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
        for forbidden in ("cv2.", "imwrite", "albumentations", "imgaug", "random_flip",
                          "mosaic(", "copy_paste", "model.train", "detector.infer",
                          "import requests"):
            self.assertFalse(forbidden in lowered, forbidden)
        self.assertIn("assert_not_sealed", source)
        self.assertFalse("def crop" in source)            # no cropping code of its own
        self.assertIn("plan_crop(anchor, []", source)     # only the anchor, never others

    def test_no_augmentation_and_no_split(self) -> None:
        source = (ROOT / "rtsp_annotator"
                  / "ground_litter_hard_negatives.py").read_text(encoding="utf-8")
        lowered = source.lower()
        for forbidden in ("def train_val", "def val_split", "split_dataset(", "hsv_",
                          "degrees=", "translate=", "shear="):
            self.assertFalse(forbidden in lowered, forbidden)
        for declared in ('"train_val_split_made": False', '"augmentation_applied": False',
                         '"development_accessed": False' if False else
                         '"sealed_accessed": False'):
            self.assertIn(declared, source)

    def test_no_truth_or_positive_write(self) -> None:
        source = (ROOT / "rtsp_annotator"
                  / "ground_litter_hard_negatives.py").read_text(encoding="utf-8")
        for forbidden in ("truth_reconciliation.jsonl", "localizations.jsonl",
                          "gold_episodes.jsonl"):
            self.assertNotIn(forbidden, source)
        for forbidden in ("os.remove", "shutil", "rmtree"):
            self.assertFalse(forbidden in source, forbidden)

    def test_cli_exposes_exactly_the_five_commands(self) -> None:
        source = CLI.read_text(encoding="utf-8")
        self.assertEqual(source.count("sub.add_parser("), 5)
        for command in ("plan", "generate", "serve", "status", "build"):
            self.assertIn(f'sub.add_parser("{command}")', source)
        self.assertNotIn("/api/proposal", source)

    def test_ui_has_the_four_decisions_and_no_box_tool(self) -> None:
        html = (TOOLS / "index.html").read_text(encoding="utf-8")
        app = (TOOLS / "app.js").read_text(encoding="utf-8")
        server = (TOOLS / "serve.py").read_text(encoding="utf-8")
        for decision in REVIEW_DECISIONS:
            self.assertIn(decision, html)
            self.assertIn(decision, app)
        for forbidden in ("BOX_OK", "BOX_BAD", "/api/proposal", "proposal",
                          "POINT_OK", "add_box", "new_bbox", "SAM", "canvas",
                          "toDataURL", "drawImage"):
            self.assertFalse(forbidden in server, forbidden)
            self.assertFalse(forbidden in html, forbidden)
            self.assertFalse(forbidden in app, forbidden)
        for route in ('path == "/api/meta"', 'path == "/api/candidate"',
                      'path == "/api/image"', 'parsed.path == "/api/review"',
                      'parsed.path == "/api/reset"', 'parsed.path == "/api/skip"'):
            self.assertIn(route, server)
        self.assertIn("IGNORE_SMALL", app)                # the reminder must be visible
        self.assertIn("anchor", app)
        self.assertIn("非训练标签", app)
        self.assertIn('"image/png"', server)
        self.assertIn("image.read_bytes()", server)

    def test_app_js_references_only_ids_that_exist_in_the_page(self) -> None:
        app = (TOOLS / "app.js").read_text(encoding="utf-8")
        html = (TOOLS / "index.html").read_text(encoding="utf-8")
        used = set(re.findall(r'\$\("([^"]+)"\)', app))
        in_page = set(re.findall(r'id="([^"]+)"', html))
        rendered = set(re.findall(r'id="([^"]+)"', app))
        self.assertTrue(used)
        self.assertEqual(used - in_page - rendered, set())
        self.assertLessEqual({"f-status", "f-camera", "f-origin", "f-search", "queue",
                              "progress", "status-line", "meta", "detail"}, used)
        actions = set(re.findall(r'data-act="([^"]+)"', app))
        self.assertEqual(actions, set(REVIEW_DECISIONS) | {"SKIP", "RESET"})


class CliTest(Base):
    def _run(self, *argv):
        import subprocess

        return subprocess.run([sys.executable, str(CLI), *argv], cwd=str(ROOT),
                              capture_output=True, text=True)

    def _args(self, f: Fixture, out: Path) -> list[str]:
        return ["--batch-dir", str(f.batch), "--recovery-root", str(f.root),
                "--output", str(out), "--repo-root", str(ROOT)]

    def test_plan_is_read_only_and_status_refuses_build(self) -> None:
        f = self.fixture()
        out = f.root / "artifact"
        plan = self._run(*self._args(f, out), "plan")
        self.assertEqual(plan.returncode, 0, plan.stderr)
        report = json.loads(plan.stdout)
        self.assertEqual(report["historical_non_litter_input_count"], 5)
        self.assertTrue(report["sampled_candidate_count"] >= 1)
        self.assertEqual(report["blockers"], [])
        self.assertTrue((out / "plan.json").is_file())
        self.assertFalse((out / "candidate_tiles").exists())

        # with no generated candidate index, status refuses instead of crashing
        status = self._run(*self._args(f, out), "status")
        self.assertEqual(status.returncode, 3)
        self.assertIn("REFUSED", status.stderr)

    def test_sealed_batch_is_refused(self) -> None:
        f = self.fixture()
        out = f.root / "artifact"
        result = self._run("--batch-dir", str(f.root / "SEALED_DO_NOT_TUNE"),
                           "--recovery-root", str(f.root), "--output", str(out),
                           "plan")
        self.assertEqual(result.returncode, 3)
        self.assertIn("REFUSED", result.stderr)


if __name__ == "__main__":
    unittest.main()
