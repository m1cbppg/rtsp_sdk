import unittest

from scripts.deduplicate_ground_litter_labels import deduplicate


class DeduplicateGroundLitterLabelsTests(unittest.TestCase):
    def test_repeated_frames_become_one_item_per_spatial_cluster(self):
        labels = [{"item_id": f"f{i}", "label": "true_litter"} for i in range(3)]
        rows = [
            {"sample_id": "f0", "candidates": [{"box": [100, 100, 120, 120]}]},
            {"sample_id": "f1", "candidates": [{"box": [102, 101, 122, 121]},
                                                    {"box": [500, 400, 520, 420]}]},
            {"sample_id": "f2", "candidates": [{"box": [99, 102, 119, 122]}]},
        ]
        result = deduplicate(labels, rows, radius_px=30)
        self.assertEqual(result["input_frame_labels"], 3)
        self.assertEqual(result["input_candidate_observations"], 4)
        self.assertEqual(result["deduplicated_item_count"], 2)
        self.assertEqual(result["items"][0]["frame_count"], 3)
        self.assertEqual(result["items"][1]["frame_count"], 1)

    def test_frame_without_candidate_does_not_create_item(self):
        result = deduplicate([{"item_id": "f0", "label": "true_litter"}],
                             [{"sample_id": "f0", "candidates": []}])
        self.assertEqual(result["deduplicated_item_count"], 0)


if __name__ == "__main__":
    unittest.main()
