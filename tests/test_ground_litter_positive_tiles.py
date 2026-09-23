"""Step 1C-2: source-native positive tiles + annotation-complete review tests (§43).

Pure stdlib + numpy (available in the repo .venv); the raw-PS decode is injected as a
deterministic fake, so no PyAV/cv2/GPU/network is required here.  The real decoder is
exercised separately by the small-scale joint test on the real PS files.
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
    CLASS_ID,
    CLASS_NAME,
    DEFAULT_FRAME_INTERVAL_SECONDS,
    MIN_LABEL_MARGIN_PX,
    REVIEW_DECISIONS,
    SCHEMA_VERSION,
    STATUS_READY,
    TILE_SIZE,
    PositiveTileError,
    TileReviewState,
    apply_review,
    bbox_iou,
    build_accepted,
    build_candidate_for_episode,
    build_label,
    build_manifest,
    build_summary,
    canonical_tile_id,
    candidate_fingerprint,
    cluster_same_frame,
    crop_tile,
    dedup_candidates,
    dedup_labels,
    generate_candidates,
    label_txt_lines,
    load_positive_tile_input,
    plan_crop,
    png_bytes,
    sha256_bytes,
    short_side_px,
    size_bucket,
    source_to_tile,
    tile_to_yolo,
    verify_preflight,
    write_bytes_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "ground_litter_tile_ui"
CLI = ROOT / "scripts" / "build_ground_litter_positive_tiles.py"

W, H = 2560, 1440
RECORDING_START = "2026-09-20 10:00:00"
PS = "/tmp/fake/01021-ps.ps"


def _timestamp(offset_seconds: int) -> str:
    base_minutes, seconds = divmod(36000 + offset_seconds, 60)
    return f"2026-09-20 {base_minutes // 60:02d}:{base_minutes % 60:02d}:{seconds:02d}"


def synthetic_frame(width: int = W, height: int = H):
    """A frame whose pixels encode their own coordinates, so a crop is self-proving."""
    import numpy as np

    ys, xs = np.mgrid[0:height, 0:width]
    return np.stack([(xs % 256), (ys % 256), ((xs + ys) % 256)], axis=2).astype("uint8")


class FakeDecoder:
    """Deterministic stand-in for the raw-PS decoder."""

    def __init__(self, *, width=W, height=H, interval=DEFAULT_FRAME_INTERVAL_SECONDS,
                 frame=None, fail_offsets=(), mismatch_offsets=(), offset_step=0.0,
                 snap=None):
        self.width, self.height = width, height
        self.interval = interval
        self.frame = synthetic_frame(width, height) if frame is None else frame
        self.fail_offsets = set(fail_offsets)
        self.mismatch_offsets = set(mismatch_offsets)
        self.offset_step = float(offset_step)
        self.snap = snap
        self.probes: list[str] = []
        self.decodes: list[tuple[str, float]] = []

    def probe_interval(self, path):
        self.probes.append(str(path))
        return {"ok": True, "frame_interval_seconds": self.interval,
                "width": self.width, "height": self.height}

    def decode_frame(self, path, offset_seconds):
        offset = round(float(offset_seconds), 3)
        self.decodes.append((str(path), offset))
        if offset in self.fail_offsets:
            return {"ok": False, "error": "simulated_decode_failure"}
        achieved = offset + (3.0 if offset in self.mismatch_offsets else 0.0)
        if self.snap:
            achieved = round(achieved / self.snap) * self.snap
        elif self.offset_step:
            achieved = offset + self.offset_step
        return {"ok": True, "frame": self.frame, "width": self.width,
                "height": self.height, "decoded_offset_seconds": achieved}


# --------------------------------------------------------------------------- #
# fixture
# --------------------------------------------------------------------------- #


def episode_row(episode_id, *, bbox, camera="01021", timestamp=None, source_file=PS,
                width=W, height=H, origin="historical_litter", eligible=True,
                truth="REQUIRED_LITTER", localization="VERIFIED_BBOX",
                recovery="RECOVERED_SOURCE_NATIVE", bucket=None) -> dict:
    return {
        "episode_id": episode_id,
        "camera_id": camera,
        "scene_version": "UNKNOWN_HISTORICAL",
        "origin": origin,
        "effective_truth_class": truth,
        "effective_truth_bucket": bucket or truth,
        "localization_status": localization,
        "source_recovery_status": recovery,
        "training_eligible": eligible,
        "training_exclusion_reason": None if eligible else "synthetic",
        "verified_bbox": list(bbox) if bbox else None,
        "source_file_id": source_file,
        "source_timestamp": timestamp or _timestamp(10),
        "source_width": width,
        "source_height": height,
        "verification_frame_path": "/tmp/fake/frame.jpg",
    }


class Fixture:
    """Writes a Step 1C-1R-shaped input set on disk."""

    def __init__(self, root: Path, *, width=W, height=H) -> None:
        self.root = root
        self.width, self.height = width, height
        self.ps_dir = root / "raw_ps"
        self.ps_dir.mkdir(parents=True, exist_ok=True)
        self.out = root / "out"
        self.state = root / "state" / "review_state.json"
        self.manifest_rows: list[dict] = []
        self.overlay_rows: list[dict] = []
        self.loc_rows: list[dict] = []
        self.evidence_rows: list[dict] = []
        self.source_rows: list[dict] = []
        self._files: dict[str, Path] = {}

    def ps_file(self, source_file_id: str) -> Path:
        if source_file_id not in self._files:
            path = self.ps_dir / f"{source_file_id}.ps"
            path.write_bytes(b"PS" + b"\x00" * 32)
            self._files[source_file_id] = path
            self.source_rows.append({
                "source_file_id": source_file_id, "camera_id": "01021",
                "local_ps_path": str(path), "local_sha256": sha256_file(path),
                "recording_start": RECORDING_START, "duration": 300.0,
                "source_width": self.width, "source_height": self.height,
                "recovery_status": "PRESERVED",
            })
        return self._files[source_file_id]

    def add_episode(self, episode_id, *, bbox=None, camera="01021", offset=10,
                    source_file="ps-a", origin="historical_litter", eligible=True,
                    truth="REQUIRED_LITTER", localization="VERIFIED_BBOX",
                    recovery="RECOVERED_SOURCE_NATIVE", bucket=None,
                    screening_bbox=None, screening_point=None,
                    decision=None, decoded_timestamp=None) -> None:
        timestamp = decoded_timestamp or _timestamp(offset)
        self.ps_file(source_file)
        row = episode_row(episode_id, bbox=bbox, camera=camera, timestamp=timestamp,
                          source_file=source_file, width=self.width, height=self.height,
                          origin=origin, eligible=eligible, truth=truth,
                          localization=localization, recovery=recovery, bucket=bucket)
        if eligible:
            self.manifest_rows.append(row)
        self.overlay_rows.append({
            "episode_id": episode_id, "camera_id": camera,
            "effective_truth_class": truth, "effective_truth_bucket": bucket or truth,
            "localization_status": localization,
            "reconciliation_decision": decision,
            "source_file_id": source_file, "step1c0_decoded_timestamp": timestamp,
        })
        self.loc_rows.append({
            "episode_id": episode_id, "camera_id": camera,
            "original_bbox": screening_bbox,
            "original_point_source": ({"ok": True, "x": screening_point[0],
                                       "y": screening_point[1]}
                                      if screening_point else None),
        })
        self.evidence_rows.append({
            "episode_id": episode_id, "camera_id": camera,
            "source_file_id": source_file, "requested_timestamp": timestamp,
            "decoded_timestamp": timestamp, "timestamp_delta_ms": 0.0,
            "source_width": self.width, "source_height": self.height,
        })

    def write(self) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        paths = {
            "training_manifest_path": self.root / "training_episode_manifest.jsonl",
            "truth_overlay_path": self.root / "truth_reconciliation.jsonl",
            "localization_path": self.root / "localizations.jsonl",
            "recovery_evidence_path": self.root / "episode_source_evidence.jsonl",
            "source_files_path": self.root / "source_files.jsonl",
        }
        for key, rows in (("training_manifest_path", self.manifest_rows),
                          ("truth_overlay_path", self.overlay_rows),
                          ("localization_path", self.loc_rows),
                          ("recovery_evidence_path", self.evidence_rows),
                          ("source_files_path", self.source_rows)):
            paths[key].write_text("".join(json.dumps(r) + "\n" for r in rows),
                                 encoding="utf-8")
        self.gold = self.root / "gold_episodes.jsonl"
        self.gold.write_text(json.dumps({"episode_id": "x"}) + "\n", encoding="utf-8")
        self.gold_manifest = self.root / "gold" / "MANIFEST.json"
        self.gold_manifest.parent.mkdir(parents=True, exist_ok=True)
        self.gold_manifest.write_text(json.dumps({"step": "1b"}), encoding="utf-8")
        return load_positive_tile_input(gold_path=self.gold,
                                        gold_manifest=self.gold_manifest, **paths)


class Base(unittest.TestCase):
    def tmpdir(self) -> Path:
        path = Path(tempfile.mkdtemp(prefix="step1c2-"))
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def fixture(self, **kwargs) -> Fixture:
        return Fixture(self.tmpdir(), **kwargs)

    def data_with(self, *episodes) -> tuple[dict, Fixture]:
        f = self.fixture()
        for episode in episodes:
            f.add_episode(**episode)
        return f.write(), f

    def generate(self, data, fixture, *, decoder=None, **kwargs) -> dict:
        decoder = decoder or FakeDecoder(width=fixture.width, height=fixture.height)
        return generate_candidates(
            data, decoder, fixture.out / "candidate_tiles" / "images",
            input_fingerprint=candidate_fingerprint(data), **kwargs)


# --------------------------------------------------------------------------- #
# §43 Eligibility
# --------------------------------------------------------------------------- #


class EligibilityTest(Base):
    def test_only_training_eligible_episodes_enter(self) -> None:
        data, f = self.data_with(
            dict(episode_id="ge-yes", bbox=[1000, 700, 1040, 740]),
            dict(episode_id="ge-ignore", bbox=[1000, 700, 1040, 740], eligible=False,
                 truth="IGNORE_SMALL", localization="VERIFIED_BBOX", bucket="IGNORE_SMALL"),
            dict(episode_id="ge-nonlitter", bbox=[1000, 700, 1040, 740], eligible=False,
                 truth="NON_LITTER", localization="VERIFIED_BBOX", bucket="NON_LITTER"),
            dict(episode_id="ge-uncertain", bbox=[1000, 700, 1040, 740], eligible=False,
                 truth="UNCERTAIN", localization="VERIFIED_BBOX", bucket="UNCERTAIN"),
            dict(episode_id="ge-identity", bbox=[1000, 700, 1040, 740], eligible=False,
                 truth="REQUIRED_LITTER", localization="VERIFIED_BBOX",
                 bucket="IDENTITY_AMBIGUOUS"),
            dict(episode_id="ge-unresolved", bbox=None, eligible=False,
                 truth="REQUIRED_LITTER", localization="LOCALIZATION_UNRESOLVED",
                 bucket="REQUIRED_LITTER"),
            dict(episode_id="ge-keep-nobbox", bbox=None, eligible=False,
                 truth="REQUIRED_LITTER", localization="TRUTH_REVIEW_REQUIRED",
                 bucket="REQUIRED_LITTER", decision="KEEP_REQUIRED"),
        )
        self.assertEqual([e["episode_id"] for e in data["episodes"]], ["ge-yes"])
        preflight = verify_preflight(data)
        self.assertEqual(preflight["training_eligible_episode_count"], 1)
        self.assertEqual(preflight["per_camera"], {"01021": 1})
        self.assertFalse(preflight["expected_count_ok"])
        self.assertEqual(preflight["eligibility_problem_count"], 0)

    def test_non_required_buckets_become_screening_targets_not_episodes(self) -> None:
        data, f = self.data_with(
            dict(episode_id="ge-yes", bbox=[1000, 700, 1040, 740]),
            dict(episode_id="ge-ignore", bbox=[1000, 700, 1040, 740], eligible=False,
                 truth="IGNORE_SMALL", bucket="IGNORE_SMALL",
                 screening_bbox=[1000, 700, 1040, 740]),
            dict(episode_id="ge-keep", bbox=None, eligible=False,
                 truth="REQUIRED_LITTER", localization="TRUTH_REVIEW_REQUIRED",
                 screening_bbox=[1005, 705, 1035, 735]),
        )
        kinds = {t["episode_id"]: t["kind"] for t in data["screening"]}
        self.assertEqual(kinds["ge-ignore"], "IGNORE_SMALL")
        self.assertEqual(kinds["ge-keep"], "UNLOCALIZED_REQUIRED")
        self.assertEqual(len(data["episodes"]), 1)

    def test_eligibility_problems_are_reported(self) -> None:
        f = self.fixture()
        f.add_episode("ge-bad", bbox=[1000, 700, 1040, 740], recovery="SOURCE_EXPIRED")
        data = f.write()
        preflight = verify_preflight(data)
        self.assertEqual(preflight["eligibility_problem_count"], 1)
        self.assertEqual(preflight["eligibility_problems"][0]["field"],
                         "source_recovery_status")

    def test_missing_source_ps_is_a_preflight_problem(self) -> None:
        f = self.fixture()
        f.add_episode("ge-missing", bbox=[1000, 700, 1040, 740])
        data = f.write()
        for path in f.ps_dir.glob("*.ps"):
            path.unlink()
        data = f.write()
        problems = {p["field"] for p in verify_preflight(data)["eligibility_problems"]}
        self.assertIn("source_ps_missing", problems)

    def test_sealed_paths_are_refused(self) -> None:
        f = self.fixture()
        f.add_episode("ge-1", bbox=[1000, 700, 1040, 740])
        with self.assertRaises(SealedAssetError):
            load_positive_tile_input(
                training_manifest_path=f.root / "SEALED_DO_NOT_TUNE" / "x.jsonl",
                truth_overlay_path=f.root / "truth_reconciliation.jsonl",
                localization_path=f.root / "localizations.jsonl",
                recovery_evidence_path=f.root / "episode_source_evidence.jsonl",
                source_files_path=f.root / "source_files.jsonl")


# --------------------------------------------------------------------------- #
# §43 Crop
# --------------------------------------------------------------------------- #


class CropTest(unittest.TestCase):
    def test_centered_crop_keeps_the_primary_fully_inside(self) -> None:
        plan = plan_crop([1240, 680, 1320, 760], [], W, H)
        self.assertTrue(plan["ok"], plan)
        crop = plan["crop_xyxy"]
        self.assertEqual(crop, [960, 400, 1600, 1040])
        self.assertGreaterEqual(plan["min_label_margin_px"], MIN_LABEL_MARGIN_PX)
        self.assertTrue(all(plan["contained_boxes"]))

    def test_left_edge_crop_is_clamped_and_still_contains_the_box(self) -> None:
        plan = plan_crop([10, 700, 60, 750], [], W, H)
        self.assertTrue(plan["ok"], plan)
        self.assertEqual(plan["crop_xyxy"], [0, 405, 640, 1045])
        self.assertTrue(plan["contained_boxes"])
        self.assertGreaterEqual(plan["min_label_margin_px"], 0)

    def test_right_edge_crop_is_clamped(self) -> None:
        # 5 px from the right frame edge: the crop must clamp and give up the margin
        plan = plan_crop([W - 30, 700, W - 5, 730], [], W, H)
        self.assertTrue(plan["ok"], plan)
        self.assertEqual(plan["crop_xyxy"], [W - 640, 395, W, 1035])
        self.assertEqual(plan["min_label_margin_px"], 5.0)

    def test_top_and_bottom_edges_are_clamped(self) -> None:
        top = plan_crop([1200, 5, 1260, 55], [], W, H)
        bottom = plan_crop([1200, H - 40, 1260, H - 3], [], W, H)
        self.assertEqual(top["crop_xyxy"], [910, 0, 1550, 640])
        self.assertEqual(bottom["crop_xyxy"], [910, H - 640, 1550, H])

    def test_crop_shifts_so_a_same_frame_target_is_not_clipped(self) -> None:
        # secondary box 40 px to the right of the primary: a naive centred crop is fine,
        # but a secondary 300 px away forces the crop to grow towards the union
        primary = [1240, 680, 1320, 760]
        secondary = [1500, 680, 1580, 760]
        plan = plan_crop(primary, [secondary], W, H)
        self.assertTrue(plan["ok"], plan)
        crop = plan["crop_xyxy"]
        for box in (primary, secondary):
            self.assertGreaterEqual(box[0], crop[0])
            self.assertGreaterEqual(box[1], crop[1])
            self.assertLessEqual(box[2], crop[2])
            self.assertLessEqual(box[3], crop[3])

    def test_source_smaller_than_the_tile_fails_without_upscaling(self) -> None:
        plan = plan_crop([10, 10, 50, 50], [], 320, 240)
        self.assertFalse(plan["ok"])
        self.assertEqual(plan["status"], "SOURCE_TOO_SMALL")

    def test_primary_larger_than_the_tile_fails(self) -> None:
        plan = plan_crop([100, 100, 100 + 641, 200], [], W, H)
        self.assertFalse(plan["ok"])
        self.assertEqual(plan["status"], "TARGET_EXCEEDS_TILE")

    def test_two_targets_that_cannot_share_a_tile_fail(self) -> None:
        # the second box sits inside the first crop, so it must be labeled too, but the
        # union is 900 px wide and cannot fit in one 640 tile
        plan = plan_crop([100, 100, 540, 540], [[560, 100, 1000, 540]], W, H)
        self.assertFalse(plan["ok"])
        self.assertEqual(plan["status"], "MULTI_TARGET_EXCEEDS_TILE")

    def test_a_target_outside_the_crop_is_not_forced_into_the_tile(self) -> None:
        plan = plan_crop([100, 100, 200, 200], [[1500, 100, 1600, 200]], W, H)
        self.assertTrue(plan["ok"], plan)
        self.assertEqual(plan["crop_xyxy"], [0, 0, 640, 640])
        self.assertEqual(len(plan["contained_boxes"]), 1)

    def test_margin_is_relaxed_but_never_clipped(self) -> None:
        # a 600 px wide box cannot keep a 32 px margin in a 640 crop, but must stay inside
        plan = plan_crop([0, 700, 600, 760], [], W, H)
        self.assertTrue(plan["ok"], plan)
        self.assertEqual(plan["min_label_margin_px"], 0.0)
        crop = plan["crop_xyxy"]
        self.assertLessEqual(600, crop[2])
        self.assertGreaterEqual(0, crop[0])


# --------------------------------------------------------------------------- #
# §43 Source-native pixels + §11 bbox transform
# --------------------------------------------------------------------------- #


class PixelTest(unittest.TestCase):
    def test_crop_is_a_pure_pixel_slice_and_640x640(self) -> None:
        frame = synthetic_frame(700, 700)
        tile = crop_tile(frame, [30, 40, 670, 680])
        self.assertEqual(tile.shape, (640, 640, 3))
        self.assertTrue((tile[0, 0] == frame[40, 30]).all())
        self.assertTrue((tile[639, 639] == frame[679, 669]).all())
        for y, x in ((0, 0), (1, 5), (100, 200), (639, 639), (321, 17)):
            self.assertTrue((tile[y, x] == frame[40 + y, 30 + x]).all())

    def test_png_is_lossless_and_deterministic(self) -> None:
        frame = synthetic_frame(80, 60)
        first = png_bytes(frame)
        second = png_bytes(frame.copy())
        self.assertEqual(first, second)
        self.assertEqual(sha256_bytes(first), sha256_bytes(second))
        self.assertTrue(first.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(int.from_bytes(first[16:20], "big"), 80)
        self.assertEqual(int.from_bytes(first[20:24], "big"), 60)

    def test_png_rejects_wrong_shape(self) -> None:
        import numpy as np

        with self.assertRaises(PositiveTileError):
            png_bytes(np.zeros((10, 10), dtype="uint8"))
        with self.assertRaises(PositiveTileError):
            png_bytes(np.zeros((10, 10, 3), dtype="float32"))

    def test_crop_rejects_a_wrong_size_crop(self) -> None:
        frame = synthetic_frame(700, 700)
        with self.assertRaises(PositiveTileError):
            crop_tile(frame, [0, 0, 100, 100])
        with self.assertRaises(PositiveTileError):
            crop_tile(frame, [100, 100, 740, 740])

    def test_bbox_transform_is_exact(self) -> None:
        crop = [960, 400, 1600, 1040]
        box = [1000, 500, 1100, 600]
        tile = source_to_tile(box, crop)
        self.assertEqual(tile, [40.0, 100.0, 140.0, 200.0])
        yolo = tile_to_yolo(tile)
        self.assertEqual(yolo, [round(90 / 640, 6), round(150 / 640, 6),
                                round(100 / 640, 6), round(100 / 640, 6)])
        for value in yolo:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_yolo_values_are_clamped_into_range(self) -> None:
        yolo = tile_to_yolo([0, 0, 640, 640])
        self.assertEqual(yolo, [0.5, 0.5, 1.0, 1.0])
        self.assertTrue(all(0.0 <= v <= 1.0 for v in tile_to_yolo([-5, -5, 645, 645])))

    def test_label_txt_lines_use_class_zero(self) -> None:
        label = build_label(["ge-1"], [1000, 500, 1100, 600], [960, 400, 1600, 1040])
        lines = label_txt_lines([label])
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith(f"{CLASS_ID} "))
        self.assertEqual(len(lines[0].split()), 5)

    def test_short_side_buckets(self) -> None:
        self.assertEqual(short_side_px([0, 0, 6, 40]), 6.0)
        self.assertEqual(size_bucket(6), "<10")
        self.assertEqual(size_bucket(15), "10-19")
        self.assertEqual(size_bucket(30), "20-39")
        self.assertEqual(size_bucket(60), "40-79")
        self.assertEqual(size_bucket(120), "80+")


# --------------------------------------------------------------------------- #
# §43 Multiple labels (§12/§13/§14/§35)
# --------------------------------------------------------------------------- #


class MultiLabelTest(Base):
    def test_same_frame_verified_target_is_added_automatically(self) -> None:
        data, f = self.data_with(
            dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760], offset=10),
            dict(episode_id="ge-b", bbox=[1500, 680, 1560, 750], offset=10),
        )
        result = self.generate(data, f)
        self.assertEqual(result["pre_dedup_count"], 2)
        self.assertEqual(len(result["candidates"]), 1)      # §34: same crop, same pixels
        row = result["candidates"][0]
        self.assertEqual(row["label_count"], 2)
        self.assertEqual(row["candidate_generation_status"], STATUS_READY)
        ids = {e for label in row["labels"] for e in label["episode_ids"]}
        self.assertEqual(ids, {"ge-a", "ge-b"})
        self.assertEqual(row["primary_episode_ids"], ["ge-a", "ge-b"])
        self.assertEqual(len(row["merged_from_tile_ids"]), 1)
        self.assertIn("ge-b", row["frame_mate_episode_ids"])
        for label in row["labels"]:
            self.assertEqual(label["class_id"], CLASS_ID)
            self.assertEqual(len(label_txt_lines([label])), 1)

    def test_different_frame_target_is_never_added(self) -> None:
        data, f = self.data_with(
            dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760], offset=10),
            dict(episode_id="ge-b", bbox=[1260, 690, 1340, 770], offset=70),
        )
        result = self.generate(data, f)
        for row in result["candidates"]:
            ids = {e for label in row["labels"] for e in label["episode_ids"]}
            self.assertEqual(ids, {row["primary_episode_id"]}, row)
            self.assertEqual(row["label_count"], 1)

    def test_same_ps_different_timestamp_is_never_added(self) -> None:
        """The §13 trap: one PS file is not one frame."""
        data, f = self.data_with(
            dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760], offset=10,
                 source_file="ps-same"),
            dict(episode_id="ge-b", bbox=[1250, 690, 1330, 770], offset=200,
                 source_file="ps-same"),
        )
        self.assertEqual(len(data["episodes"]), 2)
        result = self.generate(data, f)
        for row in result["candidates"]:
            ids = {e for label in row["labels"] for e in label["episode_ids"]}
            self.assertEqual(ids, {row["primary_episode_id"]})
        self.assertEqual({c["source_file_id"] for c in result["candidates"]},
                         {"ps-same"})

    def test_same_second_episodes_are_preclustered_but_must_confirm_one_frame(self) -> None:
        f = self.fixture()
        f.add_episode("ge-a", bbox=[1240, 680, 1320, 760],
                      decoded_timestamp="2026-09-20 10:00:10.000")
        f.add_episode("ge-b", bbox=[1250, 690, 1330, 770],
                      decoded_timestamp="2026-09-20 10:00:10.020")
        data = f.write()
        groups = cluster_same_frame(data["episodes"], {"ps-a": 0.04})
        self.assertEqual(len(groups), 1)          # 20 ms apart: one pre-cluster
        groups = cluster_same_frame(data["episodes"], {"ps-a": 0.01})
        self.assertEqual(len(groups), 2)          # a 10 ms frame interval separates them
        # the two offsets decode to different achieved frames, so no label is shared
        result = self.generate(
            data, f, decoder=FakeDecoder(interval=0.04, offset_step=0.02))
        for row in result["candidates"]:
            ids = {e for label in row["labels"] for e in label["episode_ids"]}
            self.assertEqual(ids, {row["primary_episode_id"]}, row)
        self.assertEqual(len(result["candidates"]), 2)

    def test_two_offsets_that_reach_the_same_frame_do_share_labels(self) -> None:
        f = self.fixture()
        f.add_episode("ge-a", bbox=[1240, 680, 1320, 760],
                      decoded_timestamp="2026-09-20 10:00:10.000")
        f.add_episode("ge-b", bbox=[1500, 680, 1560, 750],
                      decoded_timestamp="2026-09-20 10:00:10.020")
        data = f.write()
        # the fake decoder snaps every offset onto a 40 ms grid, so both requests decode
        # the very same frame and the labels must be shared
        result = self.generate(data, f, decoder=FakeDecoder(interval=0.04, snap=0.04))
        self.assertEqual(len(result["candidates"]), 1)
        row = result["candidates"][0]
        self.assertEqual(row["label_count"], 2)
        ids = {e for label in row["labels"] for e in label["episode_ids"]}
        self.assertEqual(ids, {"ge-a", "ge-b"})

    def test_identical_boxes_collapse_to_one_label_with_two_episode_ids(self) -> None:
        data, f = self.data_with(
            dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760], offset=10),
            dict(episode_id="ge-b", bbox=[1240, 680, 1320, 760], offset=10),
        )
        result = self.generate(data, f)
        row = result["candidates"][0]
        self.assertEqual(row["label_count"], 1)
        self.assertEqual(row["labels"][0]["episode_ids"], ["ge-a", "ge-b"])
        self.assertEqual(len(label_txt_lines(row["labels"])), 1)

    def test_label_dedup_requires_iou_095(self) -> None:
        near = [build_label(["a"], [100, 100, 200, 200], [0, 0, 640, 640]),
                build_label(["b"], [100, 100, 201, 200], [0, 0, 640, 640])]
        far = [build_label(["a"], [100, 100, 200, 200], [0, 0, 640, 640]),
               build_label(["b"], [100, 100, 260, 200], [0, 0, 640, 640])]
        self.assertEqual(len(dedup_labels(near)), 1)
        self.assertEqual(len(dedup_labels(far)), 2)
        self.assertGreater(bbox_iou(near[0]["source_xyxy"], near[1]["source_xyxy"]), 0.95)

    def test_unlocalized_required_touching_the_crop_blocks_the_tile(self) -> None:
        f = self.fixture()
        f.add_episode("ge-a", bbox=[1240, 680, 1320, 760], offset=10)
        f.add_episode("ge-keep", bbox=None, offset=10, eligible=False,
                      truth="REQUIRED_LITTER", localization="TRUTH_REVIEW_REQUIRED",
                      screening_bbox=[1300, 700, 1360, 760], decision="KEEP_REQUIRED")
        data = f.write()
        result = self.generate(data, f)
        row = result["candidates"][0]
        self.assertEqual(row["candidate_generation_status"],
                         "KNOWN_UNLOCALIZED_REQUIRED_PRESENT")
        self.assertTrue(row["known_unlocalized_required_present"])
        self.assertIn("ge-keep", row["known_unlocalized_required_in_crop_ids"])
        self.assertIsNone(row["annotation_review_status"])

    def test_unlocalized_required_provably_outside_the_crop_does_not_block(self) -> None:
        f = self.fixture()
        f.add_episode("ge-a", bbox=[400, 400, 460, 460], offset=10)
        f.add_episode("ge-keep", bbox=None, offset=10, eligible=False,
                      truth="REQUIRED_LITTER", localization="TRUTH_REVIEW_REQUIRED",
                      screening_bbox=[1900, 1000, 1960, 1060], decision="KEEP_REQUIRED")
        data = f.write()
        row = self.generate(data, f)["candidates"][0]
        self.assertEqual(row["candidate_generation_status"], STATUS_READY)
        self.assertFalse(row["known_unlocalized_required_present"])
        self.assertEqual(row["known_unlocalized_required_same_frame_ids"], ["ge-keep"])
        self.assertEqual(row["known_unlocalized_required_in_crop_ids"], [])

    def test_unlocalized_required_without_any_geometry_blocks_the_tile(self) -> None:
        f = self.fixture()
        f.add_episode("ge-a", bbox=[1240, 680, 1320, 760], offset=10)
        f.add_episode("ge-nowhere", bbox=None, offset=10, eligible=False,
                      truth="REQUIRED_LITTER", localization="LOCALIZATION_UNRESOLVED")
        data = f.write()
        row = self.generate(data, f)["candidates"][0]
        self.assertEqual(row["candidate_generation_status"],
                         "KNOWN_UNLOCALIZED_REQUIRED_PRESENT")
        self.assertEqual(row["known_unlocalized_required_without_geometry_ids"],
                         ["ge-nowhere"])

    def test_ignore_small_inside_the_crop_is_not_labeled_and_not_blocking(self) -> None:
        f = self.fixture()
        f.add_episode("ge-a", bbox=[1240, 680, 1320, 760], offset=10)
        f.add_episode("ge-small", bbox=None, offset=10, eligible=False,
                      truth="IGNORE_SMALL", bucket="IGNORE_SMALL",
                      screening_bbox=[1300, 700, 1310, 710], decision="IGNORE_SMALL")
        data = f.write()
        row = self.generate(data, f)["candidates"][0]
        self.assertEqual(row["candidate_generation_status"], STATUS_READY)
        self.assertEqual(row["label_count"], 1)
        self.assertEqual(row["ignore_small_in_crop_ids"], ["ge-small"])
        self.assertNotIn("ge-small",
                         {e for label in row["labels"] for e in label["episode_ids"]})

    def test_non_litter_is_never_a_positive_label(self) -> None:
        f = self.fixture()
        f.add_episode("ge-a", bbox=[1240, 680, 1320, 760], offset=10)
        f.add_episode("ge-nl", bbox=None, offset=10, eligible=False, truth="NON_LITTER",
                      bucket="NON_LITTER", screening_bbox=[1300, 700, 1400, 760],
                      decision="NON_LITTER")
        data = f.write()
        row = self.generate(data, f)["candidates"][0]
        self.assertEqual(row["candidate_generation_status"], STATUS_READY)
        self.assertEqual(row["non_litter_same_frame_ids"], ["ge-nl"])
        self.assertEqual(row["label_count"], 1)

    def test_uncertain_truth_in_crop_blocks_the_tile(self) -> None:
        f = self.fixture()
        f.add_episode("ge-a", bbox=[1240, 680, 1320, 760], offset=10)
        f.add_episode("ge-unc", bbox=None, offset=10, eligible=False, truth="UNCERTAIN",
                      bucket="UNCERTAIN", screening_bbox=[1300, 700, 1400, 760],
                      decision="UNCERTAIN")
        data = f.write()
        row = self.generate(data, f)["candidates"][0]
        self.assertEqual(row["candidate_generation_status"], "UNCERTAIN_TRUTH_IN_CROP")
        self.assertEqual(row["uncertain_truth_in_crop_ids"], ["ge-unc"])


# --------------------------------------------------------------------------- #
# §43 Source frame identity + idempotency + dedup
# --------------------------------------------------------------------------- #


class GenerationTest(Base):
    def test_frame_mismatch_is_reported_and_never_tiled(self) -> None:
        data, f = self.data_with(dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760]))
        decoder = FakeDecoder(mismatch_offsets=(10.0,))
        result = self.generate(data, f, decoder=decoder)
        row = result["candidates"][0]
        self.assertEqual(row["candidate_generation_status"], "SOURCE_FRAME_MISMATCH")
        self.assertFalse(row["frame_identity_confirmed"])
        self.assertEqual(row["timestamp_delta_ms"], 3000.0)
        self.assertIsNone(row["image_path"])

    def test_decode_failure_is_reported(self) -> None:
        data, f = self.data_with(dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760]))
        decoder = FakeDecoder(fail_offsets=(10.0,))
        row = self.generate(data, f, decoder=decoder)["candidates"][0]
        self.assertEqual(row["candidate_generation_status"], "DECODE_FAILED")

    def test_frame_is_decoded_once_per_frame_not_once_per_episode(self) -> None:
        data, f = self.data_with(
            dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760], offset=10),
            dict(episode_id="ge-b", bbox=[1500, 680, 1560, 750], offset=10),
        )
        decoder = FakeDecoder()
        self.generate(data, f, decoder=decoder)
        self.assertEqual(len(decoder.decodes), 1)

    def test_generate_is_idempotent_and_reuses_the_recorded_image(self) -> None:
        data, f = self.data_with(
            dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760], offset=10),
            dict(episode_id="ge-b", bbox=[1500, 680, 1560, 750], offset=10),
        )
        first = self.generate(data, f)
        decoder = FakeDecoder()
        self.assertEqual(first["pre_dedup_count"], 2)
        self.assertEqual(len(first["candidates"]), 1)     # the two share one crop
        second = generate_candidates(
            data, decoder, f.out / "candidate_tiles" / "images",
            existing=first["candidates"],
            input_fingerprint=candidate_fingerprint(data),
            existing_fingerprint=candidate_fingerprint(data))
        self.assertEqual(decoder.decodes, [])
        self.assertEqual(decoder.probes, [])   # §46: no probe either
        self.assertEqual(len(second["candidates"]), 1)
        for before, after in zip(first["candidates"], second["candidates"]):
            # a re-run must reproduce the record byte for byte, including the merge
            # provenance, and must never list a tile as merged into itself
            self.assertEqual(before, after)
            self.assertNotIn(after["tile_id"], after["merged_from_tile_ids"])

    def test_changed_inputs_refuse_to_overwrite_the_artifact(self) -> None:
        data, f = self.data_with(dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760]))
        first = self.generate(data, f)
        with self.assertRaises(PositiveTileError):
            generate_candidates(data, FakeDecoder(),
                                f.out / "candidate_tiles" / "images",
                                existing=first["candidates"],
                                input_fingerprint="new-fingerprint",
                                existing_fingerprint="old-fingerprint")

    def test_tile_id_is_deterministic_and_depends_on_the_crop(self) -> None:
        one = canonical_tile_id("ge-a", "ps", "2026-09-20 10:00:10", [960, 400, 1600, 1040])
        again = canonical_tile_id("ge-a", "ps", "2026-09-20 10:00:10", [960, 400, 1600, 1040])
        other = canonical_tile_id("ge-a", "ps", "2026-09-20 10:00:10", [0, 0, 640, 640])
        self.assertEqual(one, again)
        self.assertNotEqual(one, other)
        self.assertTrue(one.startswith("pt-"))

    def test_image_sha_is_stable_across_two_real_generations(self) -> None:
        data, f = self.data_with(dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760]))
        first = self.generate(data, f)["candidates"][0]
        shutil.rmtree(f.out)
        second = self.generate(data, f)["candidates"][0]
        self.assertEqual(first["image_sha256"], second["image_sha256"])
        self.assertEqual(first["tile_id"], second["tile_id"])
        self.assertEqual(first["size_bytes"], second["size_bytes"])

    def test_generated_image_bytes_match_the_recorded_sha(self) -> None:
        data, f = self.data_with(dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760]))
        row = self.generate(data, f)["candidates"][0]
        path = Path(row["image_path"])
        self.assertTrue(path.name.endswith(".png"))
        self.assertEqual(sha256_file(path), row["image_sha256"])
        self.assertEqual(path.stat().st_size, row["size_bytes"])

    def test_generated_png_is_640x640_and_matches_the_source_crop(self) -> None:
        import numpy as np
        import struct
        import zlib

        data, f = self.data_with(dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760]))
        row = self.generate(data, f)["candidates"][0]
        raw = Path(row["image_path"]).read_bytes()
        width = int.from_bytes(raw[16:20], "big")
        height = int.from_bytes(raw[20:24], "big")
        self.assertEqual((width, height), (640, 640))
        # inflate the IDAT stream and compare a few pixels against the source frame
        start = raw.index(b"IDAT") + 4
        end = raw.index(b"IEND") - 4
        scan = zlib.decompress(raw[start:end])
        self.assertEqual(len(scan), 640 * (1 + 640 * 3))
        frame = synthetic_frame(W, H)
        crop = row["source_crop_xyxy"]
        for y, x in ((0, 0), (5, 9), (320, 640 - 1), (639, 0), (639, 639)):
            offset = y * (1 + 640 * 3) + 1 + x * 3
            rgb = list(scan[offset:offset + 3])
            bgr = frame[crop[1] + y, crop[0] + x]
            self.assertEqual(rgb, [int(bgr[2]), int(bgr[1]), int(bgr[0])],
                             f"pixel {x},{y} differs")

    def test_exact_duplicate_tiles_are_merged(self) -> None:
        data, f = self.data_with(
            dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760], offset=10),
            dict(episode_id="ge-b", bbox=[1240, 680, 1320, 760], offset=10),
        )
        result = self.generate(data, f)
        self.assertEqual(result["pre_dedup_count"], 2)
        self.assertEqual(len(result["candidates"]), 1)
        row = result["candidates"][0]
        self.assertEqual(row["primary_episode_ids"], ["ge-a", "ge-b"])
        self.assertEqual(len(row["merged_from_tile_ids"]), 1)
        self.assertEqual(len(result["merges"]), 1)

    def test_different_crops_that_produce_different_images_are_not_merged(self) -> None:
        data, f = self.data_with(
            dict(episode_id="ge-a", bbox=[400, 400, 460, 460], offset=10),
            dict(episode_id="ge-b", bbox=[1500, 900, 1560, 960], offset=10),
        )
        result = self.generate(data, f)
        self.assertEqual(len(result["candidates"]), 2)

    def test_dedup_only_merges_identical_images_and_crop(self) -> None:
        one = {"tile_id": "pt-a", "primary_episode_id": "ge-a",
               "primary_episode_ids": ["ge-a"], "camera_id": "01021",
               "source_file_id": "ps", "step1c2_decoded_timestamp": "t",
               "source_crop_xyxy": [0, 0, 640, 640], "image_sha256": "sha1",
               "labels": [build_label(["ge-a"], [10, 10, 60, 60], [0, 0, 640, 640])],
               "label_count": 1}
        same = json.loads(json.dumps(one))
        same.update({"tile_id": "pt-b", "primary_episode_id": "ge-b",
                     "primary_episode_ids": ["ge-b"],
                     "labels": [build_label(["ge-b"], [10, 10, 60, 60], [0, 0, 640, 640])]})
        other_crop = json.loads(json.dumps(one))
        other_crop.update({"tile_id": "pt-c", "source_crop_xyxy": [1, 1, 641, 641]})
        other_sha = json.loads(json.dumps(one))
        other_sha.update({"tile_id": "pt-d", "image_sha256": "sha2"})
        rows, merges = dedup_candidates([one, same, other_crop, other_sha])
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(merges), 1)


# --------------------------------------------------------------------------- #
# §43 Annotation-complete gating + resume (§28/§29)
# --------------------------------------------------------------------------- #


class ReviewTest(Base):
    def _generated(self):
        data, f = self.data_with(
            dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760], offset=10),
            dict(episode_id="ge-b", bbox=[400, 400, 460, 460], offset=100),
        )
        result = self.generate(data, f)
        self.assertEqual(len(result["candidates"]), 2)
        return data, f, result["candidates"]

    def test_only_annotation_complete_sets_training_ready(self) -> None:
        data, f, candidates = self._generated()
        state = TileReviewState.load(f.state, candidate_count=len(candidates))
        for decision in REVIEW_DECISIONS:
            state.decide(candidates[0], decision, note="why")
            rows = apply_review(candidates, state)
            row = [r for r in rows if r["tile_id"] == candidates[0]["tile_id"]][0]
            self.assertEqual(row["annotation_review_status"], decision)
            self.assertEqual(row["positive_training_ready"],
                             decision == "ANNOTATION_COMPLETE")
        with self.assertRaises(ReviewError):
            state.decide(candidates[0], "COMPLETE")

    def test_unreviewed_tiles_are_pending_and_not_ready(self) -> None:
        data, f, candidates = self._generated()
        state = TileReviewState.load(f.state, candidate_count=len(candidates))
        rows = apply_review(candidates, state)
        for row in rows:
            self.assertEqual(row["annotation_review_status"], "PENDING")
            self.assertFalse(row["positive_training_ready"])
        self.assertEqual(state.progress(candidates),
                         {"reviewable": 2, "reviewed": 0, "pending": 2, "skipped": 0})

    def test_non_reviewable_candidates_cannot_be_decided(self) -> None:
        f = self.fixture()
        f.add_episode("ge-big", bbox=[10, 10, 700, 700], offset=10)
        data = f.write()
        result = self.generate(data, f)
        row = result["candidates"][0]
        self.assertEqual(row["candidate_generation_status"], "TARGET_EXCEEDS_TILE")
        state = TileReviewState.load(f.state, candidate_count=1)
        with self.assertRaises(ReviewError):
            state.decide(row, "ANNOTATION_COMPLETE")
        self.assertEqual(state.progress(result["candidates"])["pending"], 0)

    def test_review_state_resumes_with_an_audit_trail(self) -> None:
        data, f, candidates = self._generated()
        state = TileReviewState.load(f.state, candidate_count=len(candidates))
        state.decide(candidates[0], "ANNOTATION_COMPLETE", note="looks complete")
        state.decide(candidates[1], "MISSING_REQUIRED", note="another bag on the right")
        restarted = TileReviewState.load(f.state, candidate_count=len(candidates))
        self.assertEqual(restarted.progress(candidates)["reviewed"], 2)
        self.assertEqual(restarted.get(candidates[0]["tile_id"])["annotation_review_status"],
                         "ANNOTATION_COMPLETE")
        changed = restarted.decide(candidates[0], "BOX_PROBLEM", note="box is loose")
        self.assertEqual(changed["revision"], 2)
        self.assertEqual(changed["positive_training_ready"], False)
        self.assertEqual(changed["annotation_review_note"], "box is loose")
        self.assertEqual(len(restarted.audit_trail), 3)
        restarted.reset(candidates[0]["tile_id"])
        self.assertIsNone(restarted.get(candidates[0]["tile_id"]))
        self.assertEqual(TileReviewState.load(f.state).audit_trail[-1]["action"], "reset")

    def test_state_from_a_different_candidate_set_is_refused(self) -> None:
        data, f, candidates = self._generated()
        state = TileReviewState.load(f.state, candidate_count=len(candidates),
                                     input_fingerprint="fingerprint-a")
        state.decide(candidates[0], "ANNOTATION_COMPLETE")
        with self.assertRaises(ReviewError):
            TileReviewState.load(f.state, input_fingerprint="fingerprint-b")

    def test_skip_is_audited_and_counted(self) -> None:
        data, f, candidates = self._generated()
        state = TileReviewState.load(f.state, candidate_count=len(candidates))
        state.skip(candidates[0]["tile_id"])
        self.assertEqual(state.progress(candidates)["skipped"], 1)
        self.assertEqual(state.audit_trail[-1]["action"], "skip")


# --------------------------------------------------------------------------- #
# §30-§33 Accepted dataset
# --------------------------------------------------------------------------- #


class AcceptedTest(Base):
    def _reviewed(self, decisions):
        """Three episodes -> two tiles (ge-a/ge-b merge into one multi-label tile)."""
        data, f = self.data_with(
            dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760], offset=10),
            dict(episode_id="ge-b", bbox=[1500, 680, 1560, 750], offset=10),
            dict(episode_id="ge-c", bbox=[400, 400, 460, 460], offset=100),
        )
        candidates = self.generate(data, f)["candidates"]
        state = TileReviewState.load(f.state, candidate_count=len(candidates))
        for candidate, decision in zip(candidates, decisions):
            state.decide(candidate, decision, note="x")
        return data, f, candidates, state

    def test_build_refuses_while_anything_is_pending(self) -> None:
        data, f, candidates, state = self._reviewed(
            ["ANNOTATION_COMPLETE", "MISSING_REQUIRED"])
        self.assertEqual(len(candidates), 2)
        state.reset(candidates[1]["tile_id"])          # back to PENDING
        with self.assertRaises(PositiveTileError) as caught:
            build_accepted(candidates, f.out / "candidate_tiles" / "images",
                           f.out / "accepted", state=state)
        self.assertIn("pending=1", str(caught.exception))
        self.assertFalse((f.out / "accepted").exists())

    def test_build_refuses_when_a_tile_was_skipped(self) -> None:
        data, f, candidates, state = self._reviewed(
            ["ANNOTATION_COMPLETE", "ANNOTATION_COMPLETE"])
        state.skip(candidates[1]["tile_id"])
        with self.assertRaises(PositiveTileError) as caught:
            build_accepted(candidates, f.out / "candidate_tiles" / "images",
                           f.out / "accepted", state=state)
        self.assertIn("skipped=1", str(caught.exception))

    def test_build_copies_bytes_verbatim_and_writes_yolo_labels(self) -> None:
        data, f, candidates, state = self._reviewed(
            ["ANNOTATION_COMPLETE", "MISSING_REQUIRED"])
        accepted = build_accepted(candidates, f.out / "candidate_tiles" / "images",
                                  f.out / "accepted", state=state)
        self.assertEqual(accepted["accepted_tile_count"], 1)
        self.assertEqual(accepted["accepted_label_count"], 2)
        self.assertEqual(len(accepted["rows"][0]["labels"]), 2)
        self.assertEqual(accepted["rows"][0]["primary_episode_ids"], ["ge-a", "ge-b"])
        by_id = {c["tile_id"]: c for c in candidates}
        for row in accepted["rows"]:
            source = by_id[row["tile_id"]]
            image = Path(row["image_path"])
            label = Path(row["label_path"])
            self.assertTrue(image.name.endswith(".png"))
            self.assertEqual(sha256_file(image), source["image_sha256"])   # §31
            self.assertEqual(sha256_file(image), row["image_sha256"])
            lines = label.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), row["label_count"])
            self.assertGreaterEqual(len(lines), 1)                         # §33
            for line in lines:
                class_id, cx, cy, w, h = line.split()
                self.assertEqual(class_id, str(CLASS_ID))
                for value in (cx, cy, w, h):
                    self.assertTrue(0.0 <= float(value) <= 1.0)
            episode_ids = {e for label_row in row["labels"]
                           for e in label_row["episode_ids"]}
            self.assertIn(row["primary_episode_id"], episode_ids)          # §32

    def test_accepted_never_re_encodes_the_reviewed_tile(self) -> None:
        data, f, candidates, state = self._reviewed(
            ["ANNOTATION_COMPLETE", "ANNOTATION_COMPLETE"])
        accepted = build_accepted(candidates, f.out / "candidate_tiles" / "images",
                                  f.out / "accepted", state=state)
        by_id = {c["tile_id"]: c for c in candidates}
        for row in accepted["rows"]:
            self.assertEqual(Path(row["image_path"]).read_bytes(),
                             Path(by_id[row["tile_id"]]["image_path"]).read_bytes())
            self.assertEqual(by_id[row["tile_id"]]["image_sha256"], row["image_sha256"])

    def test_hard_fail_when_the_candidate_image_drifted(self) -> None:
        data, f, candidates, state = self._reviewed(
            ["ANNOTATION_COMPLETE", "MISSING_REQUIRED"])
        Path(candidates[0]["image_path"]).write_bytes(b"\x89PNG\r\n\x1a\nfake")
        with self.assertRaises(PositiveTileError):
            build_accepted(candidates, f.out / "candidate_tiles" / "images",
                           f.out / "accepted", state=state)

    def test_hard_fail_when_a_complete_tile_has_no_label(self) -> None:
        data, f, candidates, state = self._reviewed(
            ["ANNOTATION_COMPLETE", "MISSING_REQUIRED"])
        broken = [dict(c) for c in candidates]
        for row in broken:
            if row["tile_id"] == candidates[0]["tile_id"]:
                row["labels"] = []
        with self.assertRaises(PositiveTileError):
            build_accepted(broken, f.out / "candidate_tiles" / "images",
                           f.out / "accepted", state=state)


# --------------------------------------------------------------------------- #
# SUMMARY / MANIFEST (§47-§49)
# --------------------------------------------------------------------------- #


class SummaryTest(Base):
    def test_summary_reports_the_generate_state(self) -> None:
        data, f = self.data_with(
            dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760], offset=10, camera="01021"),
            dict(episode_id="ge-b", bbox=[1500, 680, 1560, 750], offset=10, camera="01021",
                 source_file="ps-a"),
            dict(episode_id="ge-c", bbox=[400, 400, 460, 460], offset=100, camera="01022",
                 source_file="ps-b"),
        )
        result = self.generate(data, f)
        preflight = verify_preflight(data)
        state = TileReviewState.load(f.state, candidate_count=len(result["candidates"]))
        summary = build_summary(data, result["candidates"], preflight, state=state,
                                merges=result["merges"], decoded=True,
                                stats={**result["stats"],
                                       "pre_dedup_count": result["pre_dedup_count"]})
        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["tile_size"], TILE_SIZE)
        self.assertEqual(summary["class_mapping"], {"0": CLASS_NAME})
        self.assertEqual(summary["input"]["training_eligible_episode_count"], 3)
        self.assertEqual(summary["generate"]["candidate_tile_count"], 3)
        self.assertEqual(summary["generate"]["deduplicated_tile_count"], 2)
        self.assertEqual(summary["generate"]["merged_duplicate_tile_count"], 1)
        self.assertEqual(summary["generate"]["merge_count"], 1)
        self.assertEqual(summary["generate"]["generation_failed_count"], 0)
        self.assertEqual(summary["generate"]["multi_label_tile_count"], 1)
        self.assertEqual(summary["generate"]["tiles_with_2plus_labels"], 1)
        self.assertEqual(summary["generate"]["tiles_with_1_label"], 1)
        self.assertEqual(summary["generate"]["decoded_frame_count"], 2)
        self.assertEqual(summary["review"]["reviewable"], 2)
        self.assertEqual(summary["review"]["reviewed"], 0)
        self.assertEqual(summary["review"]["pending"], 2)
        self.assertEqual(summary["accepted"]["accepted_positive_tile_count"], 0)
        self.assertEqual(summary["per_camera"]["01021"]["input_eligible"], 2)
        self.assertEqual(summary["per_camera"]["01022"]["input_eligible"], 1)
        buckets = summary["generate"]["bbox_short_side_buckets"]
        self.assertEqual(sum(buckets.values()), 3)          # 2 tiles, one with 2 labels
        for value in summary["boundaries"].values():
            self.assertIs(value, False)

    def test_summary_after_build_reports_accepted_and_lost(self) -> None:
        data, f = self.data_with(
            dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760], offset=10),
            dict(episode_id="ge-b", bbox=[400, 400, 460, 460], offset=100),
        )
        candidates = self.generate(data, f)["candidates"]
        state = TileReviewState.load(f.state, candidate_count=len(candidates))
        state.decide(candidates[0], "ANNOTATION_COMPLETE", note="")
        state.decide(candidates[1], "BOX_PROBLEM", note="box covers the whole tile")
        accepted = build_accepted(candidates, f.out / "candidate_tiles" / "images",
                                  f.out / "accepted", state=state)
        summary = build_summary(data, candidates, verify_preflight(data), state=state,
                                decoded=True, accepted=accepted,
                                stats={"pre_dedup_count": len(candidates)})
        self.assertEqual(summary["review"]["counts"]["ANNOTATION_COMPLETE"], 1)
        self.assertEqual(summary["review"]["counts"]["BOX_PROBLEM"], 1)
        self.assertEqual(summary["accepted"]["accepted_positive_tile_count"], 1)
        self.assertEqual(summary["accepted"]["unique_episode_ids_represented"], 1)
        self.assertEqual(summary["accepted"]["episodes_lost_due_annotation_incomplete"], 1)
        self.assertIn(candidates[1]["primary_episode_id"],
                      summary["accepted"]["episodes_lost_ids"])

    def test_manifest_records_the_frozen_inputs_and_flags(self) -> None:
        data, f = self.data_with(dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760]))
        result = self.generate(data, f)
        preflight = verify_preflight(data)
        state = TileReviewState.load(f.state, candidate_count=len(result["candidates"]))
        summary = build_summary(data, result["candidates"], preflight, state=state,
                                decoded=True,
                                stats={"pre_dedup_count": result["pre_dedup_count"]})
        manifest = build_manifest(
            data, summary, code_commit="0" * 40, generated_at="2026-09-23T00:00:00Z",
            config={"images_dir": "x", "accepted_root": "y"},
            artifact_root=f.out, candidate_index_path=f.out / "tile_candidates.jsonl",
            review_state_path=f.state, provenance={"note": "n"})
        self.assertEqual(manifest["tile_size"], 640)
        self.assertTrue(manifest["source_native"])
        self.assertFalse(manifest["resize"])
        self.assertEqual(manifest["class_mapping"], {"0": CLASS_NAME})
        for key in ("gold_sha256", "recovery_evidence_sha256", "localization_sha256",
                    "truth_reconciliation_sha256",
                    "training_episode_manifest_sha256"):
            self.assertIn(key, manifest["input_sha256"])
        self.assertEqual(manifest["counts"]["candidate_tile_count"], 1)
        self.assertEqual(manifest["counts"]["reviewed"], 0)
        for value in manifest["boundaries"].values():
            self.assertIs(value, False)


# --------------------------------------------------------------------------- #
# §43 Safety + UI
# --------------------------------------------------------------------------- #


class SafetyTest(unittest.TestCase):
    def test_logic_module_has_no_model_or_training_dependency(self) -> None:
        source = (ROOT / "rtsp_annotator"
                  / "ground_litter_positive_tiles.py").read_text(encoding="utf-8")
        import ast

        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for module in ("cv2", "torch", "torchvision", "ultralytics", "tensorrt"):
            self.assertNotIn(module, imported, module)
        for forbidden in ("model.train(", "yolo.train(", "hard_negative_mining",
                          "generate_negatives", "augment_dataset"):
            self.assertNotIn(forbidden, source, forbidden)
        # numpy may only be imported inside the two pixel helpers
        self.assertEqual(source.count("import numpy"), 2)

    def test_decode_adapter_imports_nothing_heavy_at_module_level(self) -> None:
        import ast

        source = (ROOT / "rtsp_annotator"
                  / "ground_litter_positive_tile_decode.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level = set()
        for node in tree.body:                      # module level only
            if isinstance(node, ast.Import):
                top_level.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.add(node.module)
        self.assertNotIn("av", top_level)
        self.assertNotIn("cv2", top_level)
        self.assertIn("import av", source)          # imported lazily inside methods
        self.assertNotIn("cv2", source)

    def test_cli_exposes_exactly_the_five_commands(self) -> None:
        source = CLI.read_text(encoding="utf-8")
        self.assertEqual(source.count("sub.add_parser("), 5)
        for command in ("plan", "generate", "serve", "status", "build"):
            self.assertIn(f'sub.add_parser("{command}")', source)
        self.assertNotIn("/api/proposal", source)

    def test_review_ui_cannot_add_or_move_a_box(self) -> None:
        html = (TOOLS / "index.html").read_text(encoding="utf-8")
        app = (TOOLS / "app.js").read_text(encoding="utf-8")
        server = (TOOLS / "serve.py").read_text(encoding="utf-8")
        for decision in REVIEW_DECISIONS:
            self.assertIn(decision, html)
            self.assertIn(decision, app)
        for forbidden in ("BOX_OK", "BOX_BAD", "/api/proposal", "proposal",
                          "POINT_OK", "point_click", "draw_box", "add_box",
                          "new_bbox", "SAM"):
            self.assertNotIn(forbidden, server, forbidden)
            self.assertNotIn(forbidden, html, forbidden)
            self.assertNotIn(forbidden, app, forbidden)
        for route in ('path == "/api/meta"', 'path == "/api/tile"',
                      'path == "/api/image"', 'parsed.path == "/api/review"',
                      'parsed.path == "/api/reset"', 'parsed.path == "/api/skip"'):
            self.assertIn(route, server)
        self.assertEqual(server.count('parsed.path == "/api/'), 3)

    def test_review_ui_overlay_never_touches_the_image_bytes(self) -> None:
        app = (TOOLS / "app.js").read_text(encoding="utf-8")
        server = (TOOLS / "serve.py").read_text(encoding="utf-8")
        # overlays are CSS boxes over an untouched <img>
        self.assertNotIn("canvas", app.lower())
        self.assertNotIn("toDataURL", app)
        self.assertNotIn("drawImage", app)
        self.assertIn("class=\"box", app)
        # the image route only ever reads the recorded PNG
        self.assertIn('"image/png"', server)
        self.assertIn("image.read_bytes()", server)
        for forbidden in ("ImageDraw", "cv2.imwrite", "putText", "rectangle("):
            self.assertNotIn(forbidden, server, forbidden)

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
        self.assertLessEqual({"tilewrap", "zoomwrap", "note"}, used)
        actions = set(re.findall(r'data-act="([^"]+)"', app))
        self.assertEqual(actions, set(REVIEW_DECISIONS) | {"SKIP", "RESET"})

    def test_build_never_writes_outside_the_output_tree(self) -> None:
        source = (ROOT / "rtsp_annotator"
                  / "ground_litter_positive_tiles.py").read_text(encoding="utf-8")
        self.assertNotIn("shutil", source)
        self.assertNotIn("os.remove", source)
        self.assertNotIn("rmtree", source)


# --------------------------------------------------------------------------- #
# CLI end to end
# --------------------------------------------------------------------------- #


class CliTest(Base):
    def _run(self, *argv):
        import subprocess

        return subprocess.run([sys.executable, str(CLI), *argv], cwd=str(ROOT),
                              capture_output=True, text=True)

    def _args(self, f: Fixture, out: Path) -> list[str]:
        return ["--training-manifest", str(f.root / "training_episode_manifest.jsonl"),
                "--truth-overlay", str(f.root / "truth_reconciliation.jsonl"),
                "--localization", str(f.root / "localizations.jsonl"),
                "--recovery-evidence", str(f.root / "episode_source_evidence.jsonl"),
                "--source-files", str(f.root / "source_files.jsonl"),
                "--gold", str(f.gold), "--gold-manifest", str(f.gold_manifest),
                "--output", str(out), "--repo-root", str(ROOT)]

    def test_plan_is_read_only_and_writes_plan_json(self) -> None:
        data, f = self.data_with(
            dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760], offset=10),
            dict(episode_id="ge-b", bbox=[1500, 680, 1560, 750], offset=10),
        )
        out = f.root / "artifact"
        result = self._run(*self._args(f, out), "plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["eligible_episode_count"], 2)
        self.assertEqual(report["distinct_frames"], 1)
        self.assertEqual(report["multi_label_frame_count"], 1)
        self.assertFalse(report["preflight"]["expected_count_ok"])
        self.assertTrue((out / "plan.json").is_file())
        self.assertFalse((out / "candidate_tiles").exists())

    def _generate_via_logic(self, f: Fixture, out: Path) -> list[dict]:
        """The CLI's own generate path is covered by the real joint test."""
        from rtsp_annotator.ground_litter_positive_tiles import (
            candidate_fingerprint as fingerprint,
            generate_candidates as generate,
            write_jsonl as dump,
        )

        data = f.write()
        args = self._args(f, out)
        decoder = FakeDecoder()
        result = generate(data, decoder, out / "candidate_tiles" / "images",
                          input_fingerprint=fingerprint(data))
        dump(out / "tile_candidates.jsonl", result["candidates"])
        args_state = out / "review_state.json"
        TileReviewState.load(args_state,
                             candidate_count=len(result["candidates"])).save()
        return result["candidates"]

    def test_status_and_build_refuse_before_any_review(self) -> None:
        data, f = self.data_with(dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760]))
        out = f.root / "artifact"
        out.mkdir(parents=True, exist_ok=True)
        self._generate_via_logic(f, out)
        self.assertTrue((out / "tile_candidates.jsonl").is_file())
        status = self._run(*self._args(f, out), "status")
        self.assertEqual(status.returncode, 0, status.stderr)
        payload = json.loads(status.stdout)
        self.assertEqual(payload["reviewable"], 1)
        self.assertEqual(payload["reviewed"], 0)
        self.assertFalse(payload["build_allowed"])
        built = self._run(*self._args(f, out), "build")
        self.assertEqual(built.returncode, 4)
        self.assertIn("REFUSED", built.stderr)
        self.assertFalse((out / "accepted").exists())

    def test_build_succeeds_after_every_tile_is_reviewed(self) -> None:
        data, f = self.data_with(dict(episode_id="ge-a", bbox=[1240, 680, 1320, 760]))
        out = f.root / "artifact"
        out.mkdir(parents=True, exist_ok=True)
        candidates = self._generate_via_logic(f, out)
        state = TileReviewState.load(out / "review_state.json")
        state.decide(candidates[0], "ANNOTATION_COMPLETE", note="complete")
        built = self._run(*self._args(f, out), "build")
        self.assertEqual(built.returncode, 0, built.stderr)
        payload = json.loads(built.stdout)
        self.assertEqual(payload["accepted_positive_tile_count"], 1)
        self.assertTrue((out / "SUMMARY.json").is_file())
        self.assertTrue((out / "MANIFEST.json").is_file())
        self.assertEqual(payload["accepted_label_count"], 1)
        self.assertTrue((out / "positive_training_manifest.jsonl").is_file())
        image = next((out / "accepted" / "images").glob("*.png"))
        label = next((out / "accepted" / "labels").glob("*.txt"))
        self.assertEqual(sha256_file(image), candidates[0]["image_sha256"])
        self.assertTrue(label.read_text(encoding="utf-8").startswith("0 "))


if __name__ == "__main__":
    unittest.main()
