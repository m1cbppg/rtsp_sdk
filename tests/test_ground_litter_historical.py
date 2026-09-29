"""Unit tests for the Rapid Dataset v2 historical mining core (no cv2/torch needed)."""
from __future__ import annotations

from datetime import datetime, timedelta
import unittest

from rtsp_annotator.ground_litter_historical import (
    BLIND_FRAMES_PER_CAMERA, CAMERAS, DAY_SPLIT, EXPECTED_WINDOW_TOTAL, Observation,
    QueueEntry, RecordingSlot, choose_representatives, cluster_episodes,
    day1_selection, greedy_diversity_select, link_review_groups, manifest_sha256,
    select_windows, blind_frame_selection,
)


def _slots(camera: str, date: str, *, start_hour: int = 6, count: int = 12) -> list[RecordingSlot]:
    slots = []
    for index in range(count):
        start = datetime.fromisoformat(f"{date} {start_hour:02d}:00:00") + timedelta(minutes=5 * index)
        end = start + timedelta(minutes=5)
        slots.append(RecordingSlot(
            file_id=f"{camera}-{date}-{index}",
            record_start=start.strftime("%Y-%m-%d %H:%M:%S"),
            record_end=end.strftime("%Y-%m-%d %H:%M:%S"),
            file_size=88_000_000,
            file_name=f"{index}.ps",
        ))
    return slots


def _all_slots() -> dict:
    out = {}
    for camera in CAMERAS:
        for date in DAY_SPLIT:
            out[(camera, date)] = _slots(camera, date, start_hour=6, count=144)
    return out


def _obs(obs_id: str, camera: str, seconds: float, box, *, sampling="coarse",
         sources=None, appearance=None, background=None) -> Observation:
    return Observation(
        observation_id=obs_id,
        camera_id=camera,
        frame_id=f"frame-{obs_id}",
        window_id=f"w-{obs_id}",
        timestamp=1_700_000_000.0 + seconds,
        bbox_xyxy=[float(v) for v in box],
        sampling=sampling,
        confidence_by_source=sources or {"turhancan": 0.4},
        appearance=appearance,
        background_key=background,
    )


class WindowSelectionTests(unittest.TestCase):
    def test_counts_and_determinism(self):
        slots = _all_slots()
        first = select_windows(slots)
        second = select_windows(slots)
        self.assertEqual(len(first), EXPECTED_WINDOW_TOTAL)
        self.assertEqual([r["window_id"] for r in first], [r["window_id"] for r in second])
        by_split = {}
        for row in first:
            by_split.setdefault(row["day_split"], []).append(row)
        self.assertEqual(len(by_split["TRAIN"]), 60)
        self.assertEqual(len(by_split["DEV"]), 16)
        self.assertEqual(len(by_split["FINAL"]), 16)
        for row in by_split["TRAIN"]:
            self.assertIn(row["selection_bucket"], {"early", "mid", "late"})

    def test_hour_varies_across_days(self):
        rows = select_windows(_all_slots())
        seen = {}
        for row in rows:
            if row["day_split"] != "TRAIN":
                continue
            key = (row["camera_id"], row["selection_bucket"])
            seen.setdefault(key, []).append(row["start_time"][11:13])
        for key, hours in seen.items():
            # each bucket spans four clock hours over five train days
            self.assertGreaterEqual(len(set(hours)), 4, f"too repetitive for {key}: {hours}")

    def test_final_day_boundary_is_respected(self):
        rows = select_windows(_all_slots(), day7_usable_end="12:00:00")
        final = [r for r in rows if r["day_split"] == "FINAL"]
        self.assertTrue(final)
        for row in final:
            self.assertLess(row["start_time"][11:19], "12:00:00")

    def test_day1_selection_has_twenty_windows(self):
        rows = select_windows(_all_slots())
        chosen = day1_selection(rows)
        self.assertEqual(len(chosen), 20)
        self.assertEqual(len(set(chosen)), 20)

    def test_manifest_hash_ignores_download_state(self):
        rows = select_windows(_all_slots())
        before = manifest_sha256(rows)
        for row in rows:
            row["download_status"] = "done"
            row["local_path"] = "/tmp/x.ps"
            row["source_sha256"] = "deadbeef"
        self.assertEqual(before, manifest_sha256(rows))


class EpisodeTests(unittest.TestCase):
    def test_gap_below_coarse_limit_merges(self):
        rows = [
            _obs("o1", "01021", 0, [100, 100, 130, 130]),
            _obs("o2", "01021", 120, [103, 101, 133, 129]),
            _obs("o3", "01021", 240, [99, 102, 128, 132]),
        ]
        episodes = cluster_episodes(rows)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].observation_count if hasattr(episodes[0], "observation_count")
                         else len(episodes[0].observation_ids), 3)

    def test_gap_above_coarse_limit_starts_new_episode(self):
        rows = [
            _obs("o1", "01021", 0, [100, 100, 130, 130]),
            _obs("o2", "01021", 400, [100, 100, 130, 130]),
        ]
        episodes = cluster_episodes(rows)
        self.assertEqual(len(episodes), 2)

    def test_far_jump_starts_new_episode(self):
        rows = [
            _obs("o1", "01021", 0, [100, 100, 130, 130]),
            _obs("o2", "01021", 60, [800, 900, 830, 930]),
        ]
        self.assertEqual(len(cluster_episodes(rows)), 2)

    def test_size_ratio_above_three_starts_new_episode(self):
        rows = [
            _obs("o1", "01021", 0, [100, 100, 110, 110]),
            _obs("o2", "01021", 60, [100, 100, 140, 140]),
        ]
        self.assertEqual(len(cluster_episodes(rows)), 2)

    def test_small_target_jitter_is_not_a_split(self):
        # 12x12 target: 4 px of jitter is far more than half the box, centre distance
        # keeps it together because r = clamp(0.6*12, 8, 32) = 8.
        rows = [
            _obs("o1", "01021", 0, [100, 100, 112, 112]),
            _obs("o2", "01021", 30, [104, 104, 116, 116]),
        ]
        self.assertEqual(len(cluster_episodes(rows)), 1)

    def test_cross_camera_never_merges(self):
        rows = [
            _obs("o1", "01021", 0, [100, 100, 130, 130]),
            _obs("o2", "01022", 30, [100, 100, 130, 130]),
        ]
        self.assertEqual(len(cluster_episodes(rows)), 2)

    def test_ambiguous_pair_is_not_auto_merged(self):
        # Two open episodes 20 px apart (r = 12) and a third observation exactly
        # half-way between them is equidistant, so it must not silently merge.
        rows = [
            _obs("o1", "01021", 0, [100, 100, 120, 120]),
            _obs("o2", "01021", 10, [120, 100, 140, 120]),
            _obs("o3", "01021", 20, [110, 100, 130, 120]),
        ]
        episodes = cluster_episodes(rows)
        self.assertGreaterEqual(len(episodes), 2)
        self.assertTrue(any(e.ambiguous_parents for e in episodes))


class ReviewGroupTests(unittest.TestCase):
    def test_long_gap_same_place_becomes_suspected_group(self):
        rows = [
            _obs("o1", "01021", 0, [100, 100, 130, 130]),
            _obs("o2", "01021", 4000, [104, 102, 134, 132]),
        ]
        episodes = cluster_episodes(rows)
        groups = link_review_groups(episodes, rows)
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0].suspected_same_object)

    def test_long_gap_far_place_is_new_object(self):
        rows = [
            _obs("o1", "01021", 0, [100, 100, 130, 130]),
            _obs("o2", "01021", 4000, [900, 900, 930, 930]),
        ]
        episodes = cluster_episodes(rows)
        groups = link_review_groups(episodes, rows)
        self.assertEqual(len(groups), 2)
        self.assertFalse(any(g.suspected_same_object for g in groups))

    def test_appearance_mismatch_blocks_linking(self):
        rows = [
            _obs("o1", "01021", 0, [100, 100, 130, 130], appearance=[10, 10, 10]),
            _obs("o2", "01021", 4000, [104, 102, 134, 132], appearance=[250, 250, 250]),
        ]
        episodes = cluster_episodes(rows)
        groups = link_review_groups(episodes, rows)
        self.assertEqual(len(groups), 2)


class RepresentativeTests(unittest.TestCase):
    def test_normal_representative_is_not_the_largest_or_most_confident(self):
        rows = [
            _obs("o1", "01021", 0, [100, 100, 120, 120], sources={"turhancan": 0.9}),
            _obs("o2", "01021", 60, [100, 100, 130, 130], sources={"turhancan": 0.2}),
            _obs("o3", "01021", 120, [100, 100, 240, 240], sources={"turhancan": 0.95}),
        ]
        picked = choose_representatives(rows)
        self.assertEqual(picked[0]["role"], "normal")
        self.assertEqual(picked[0]["observation_id"], "o2")

    def test_state_change_adds_second_frame(self):
        rows = [
            _obs("o1", "01021", 0, [100, 100, 130, 130], appearance=[10, 10, 10], background="a"),
            _obs("o2", "01021", 60, [100, 100, 130, 130], appearance=[200, 200, 200], background="b"),
            _obs("o3", "01021", 120, [100, 100, 130, 130], appearance=[10, 10, 10], background="a"),
        ]
        picked = choose_representatives(rows)
        self.assertGreaterEqual(len(picked), 2)
        self.assertLessEqual(len(picked), 3)


class DiversityTests(unittest.TestCase):
    def test_greedy_prefers_new_buckets(self):
        entries = [
            QueueEntry("rg-1", "P1", ["camera:01021", "time:early", "size:tiny"]),
            QueueEntry("rg-2", "P1", ["camera:01021", "time:early", "size:tiny"]),
            QueueEntry("rg-3", "P3", ["camera:01022", "time:late", "size:large"]),
        ]
        ordered = greedy_diversity_select(entries, limit=2, random_reserve_fraction=0.0)
        self.assertEqual(ordered[0].review_group_id, "rg-1")
        self.assertEqual(ordered[1].review_group_id, "rg-3")

    def test_random_reserve_is_deterministic(self):
        entries = [QueueEntry(f"rg-{i}", "P3", [f"b{i}"]) for i in range(20)]
        first = [e.review_group_id for e in greedy_diversity_select(entries, limit=5)]
        entries = [QueueEntry(f"rg-{i}", "P3", [f"b{i}"]) for i in range(20)]
        second = [e.review_group_id for e in greedy_diversity_select(entries, limit=5)]
        self.assertEqual(first, second)


class BlindTests(unittest.TestCase):
    def test_blind_selection_per_camera(self):
        frames = []
        for camera in CAMERAS:
            for index in range(20):
                frames.append({
                    "frame_id": f"{camera}-f{index}",
                    "camera_id": camera,
                    "window_id": f"w{index}",
                    "date": f"2026-09-2{3 + index % 5}",
                    "selection_bucket": ["early", "mid", "late"][index % 3],
                    "offset_seconds": 30.0 * (index % 3),
                })
        chosen = blind_frame_selection(frames, per_camera=BLIND_FRAMES_PER_CAMERA)
        self.assertEqual(len(chosen), BLIND_FRAMES_PER_CAMERA * len(CAMERAS))
        for camera in CAMERAS:
            subset = [c for c in chosen if c["camera_id"] == camera]
            self.assertEqual(len(subset), BLIND_FRAMES_PER_CAMERA)
            self.assertEqual(sum(1 for c in subset if c["blind_kind"] == "truly_random"), 4)
            self.assertEqual(len({c["frame_id"] for c in subset}), BLIND_FRAMES_PER_CAMERA)


if __name__ == "__main__":
    unittest.main()
