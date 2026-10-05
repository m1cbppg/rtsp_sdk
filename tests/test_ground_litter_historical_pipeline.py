"""End-to-end tests for the v2 coarse -> review-group pipeline (no cv2/torch needed)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rtsp_annotator.ground_litter_historical import CAMERAS
from rtsp_annotator.ground_litter_production_scale import (
    merge_cross_model_candidates, tile_plan, tile_starts,
)

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_ground_litter_historical_review.py"


class TilePlanTests(unittest.TestCase):
    def test_tile_starts_full_coverage_without_duplicates(self):
        starts = tile_starts(2560)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 2560 - 640)
        self.assertEqual(len(starts), len(set(starts)))
        for index in range(len(starts) - 1):
            self.assertLessEqual(starts[index + 1] - starts[index], 512)
        covered = set()
        for start in starts:
            covered.update(range(start, start + 640))
        self.assertEqual(covered, set(range(2560)))

    def test_tile_plan_shape(self):
        plan = tile_plan(2560, 1440)
        self.assertEqual(len(plan), len(tile_starts(2560)) * len(tile_starts(1440)))

    def test_small_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            tile_starts(320)


class MergeTests(unittest.TestCase):
    def _box(self, source, xyxy, conf, cls=0, name="litter"):
        return {"source": source, "xyxy": xyxy, "confidence": conf,
                "class_id": cls, "class_name": name, "tile_xy": [0, 0]}

    def test_overlapping_models_merge_into_both(self):
        merged = merge_cross_model_candidates([
            self._box("turhancan", [100, 100, 140, 140], 0.8),
            self._box("yolo", [104, 102, 144, 142], 0.3),
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual(set(merged[0]["confidence_by_source"]), {"turhancan", "yolo"})
        self.assertEqual(set(merged[0]["bbox_by_source"]), {"turhancan", "yolo"})

    def test_disjoint_models_stay_separate(self):
        merged = merge_cross_model_candidates([
            self._box("turhancan", [100, 100, 140, 140], 0.8),
            self._box("yolo", [900, 900, 940, 940], 0.3),
        ])
        self.assertEqual(len(merged), 2)
        labels = sorted(next(iter(item["confidence_by_source"])) for item in merged)
        self.assertEqual(labels, ["turhancan", "yolo"])

    def test_secondary_only_box_is_kept(self):
        merged = merge_cross_model_candidates([
            self._box("turhancan", [100, 100, 140, 140], 0.8),
            self._box("yolo", [104, 102, 144, 142], 0.3),
            self._box("yolo", [600, 600, 640, 640], 0.2),
        ])
        self.assertEqual(len(merged), 2)
        self.assertTrue(any(set(item["confidence_by_source"]) == {"yolo"} for item in merged))


def _synth_artifact(root: Path) -> None:
    frames = []
    observations = []
    frame_index = 0
    for camera_index, camera in enumerate(CAMERAS):
        for window in range(3):
            for offset in (30.0, 150.0, 270.0):
                frame_index += 1
                frame_id = f"{camera}_w{window}_t{int(offset):03d}"
                frames.append({
                    "frame_id": frame_id,
                    "camera_id": camera,
                    "window_id": f"{camera}_w{window}",
                    "date": f"2026-09-2{3 + window}",
                    "day_split": "TRAIN",
                    "selection_bucket": ["early", "mid", "late"][window],
                    "offset_seconds": offset,
                    "frame_index": int(offset * 25),
                    "width": 2560,
                    "height": 1440,
                    "record_start": f"2026-09-2{3 + window} 0{6 + window}:00:00",
                    "record_end": f"2026-09-2{3 + window} 0{6 + window}:05:00",
                    "image": f"frames/{frame_id}.png",
                    "image_sha256": "0" * 64,
                    "roi": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                    "roi_geometry_version": "test",
                })
                # two fixed litter locations, one moving on the third window
                x = 200.0 + camera_index * 10
                y = 300.0
                timestamp = 1_700_000_000.0 + window * 3600 + offset
                if window == 0:
                    observations.append(_observation(frame_id, camera, timestamp, [x, y, x + 40, y + 40],
                                                     {"turhancan": 0.5}, window, offset))
                elif window == 1:
                    observations.append(_observation(frame_id, camera, timestamp, [x, y, x + 40, y + 40],
                                                     {"yolo": 0.22}, window, offset))
                else:
                    observations.append(_observation(frame_id, camera, timestamp,
                                                     [x + 500, y + 300, x + 540, y + 340],
                                                     {"turhancan": 0.4, "yolo": 0.2}, window, offset))
    (root / "coarse_frames.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in frames) + "\n", encoding="utf-8")
    (root / "candidate_observations.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in observations) + "\n",
        encoding="utf-8")


def _observation(frame_id, camera, timestamp, box, confidences, window, offset):
    return {
        "observation_id": f"{frame_id}_o00",
        "frame_id": frame_id,
        "camera_id": camera,
        "window_id": f"{camera}_w{window}",
        "date": f"2026-09-2{3 + window}",
        "day_split": "TRAIN",
        "selection_bucket": ["early", "mid", "late"][window],
        "offset_seconds": offset,
        "timestamp": timestamp,
        "bbox_xyxy": [float(v) for v in box],
        "bbox_by_source": {name: [float(v) for v in box] for name in confidences},
        "sampling": "coarse",
        "confidence_by_source": confidences,
        "class_name_by_source": {name: "litter" for name in confidences},
        "appearance": [120.0, 118.0, 115.0],
        "background_key": "B1T0",
    }


class BuilderEndToEndTests(unittest.TestCase):
    def test_builder_produces_review_units_and_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp)
            _synth_artifact(artifact)
            result = subprocess.run(
                [sys.executable, "-B", str(BUILDER), "--artifact", str(artifact),
                 "--no-classical", "--blind-per-camera", "2"],
                capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            units = [json.loads(line) for line in
                     (artifact / "review_units.jsonl").read_text(encoding="utf-8").splitlines() if line]
            queue = json.loads((artifact / "queue.json").read_text(encoding="utf-8"))
            summary = json.loads((artifact / "review_build_summary.json").read_text(encoding="utf-8"))

            self.assertTrue(units)
            self.assertEqual(summary["episodes"] > 0, True)
            self.assertEqual(summary["review_groups"], len(units) - summary["blind_units"])
            self.assertEqual(summary["blind_units"], 2 * len(CAMERAS))
            self.assertEqual(queue["total_units"], len(units))
            self.assertTrue(queue["order"])

            candidate_units = [u for u in units if u["kind"] == "candidate"]
            blind_units = [u for u in units if u["kind"] == "blind"]
            self.assertTrue(candidate_units)
            self.assertEqual(len(blind_units), 2 * len(CAMERAS))
            for unit in candidate_units:
                self.assertTrue(unit["observations"])
                self.assertEqual(unit["observations"][0]["role"], "normal")
                self.assertIn(unit["tier"], {"P1", "P2", "P3", "P3b", "LOW"})
                self.assertLessEqual(len(unit["observations"]), 3)
            for unit in blind_units:
                self.assertTrue(unit["blind"])
                self.assertEqual(unit["candidates"], [])
                self.assertEqual(unit["tier"], "BLIND")

    def test_duplicate_background_is_collapsed_to_one_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp)
            _synth_artifact(artifact)
            rows = [json.loads(line) for line in
                    (artifact / "candidate_observations.jsonl").read_text(encoding="utf-8").splitlines() if line]
            sys.path.insert(0, str(ROOT / "scripts"))
            try:
                import importlib
                builder = importlib.import_module("build_ground_litter_historical_review")
            finally:
                sys.path.pop(0)
            stats = builder.mark_duplicate_background(rows)
            self.assertGreater(stats["clusters"], 0)
            primaries = [row for row in rows if not row["duplicate_background"]]
            self.assertEqual(len(primaries), stats["clusters"])


if __name__ == "__main__":
    unittest.main()
