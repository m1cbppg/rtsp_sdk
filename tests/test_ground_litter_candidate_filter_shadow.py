import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


shadow = load("shadow", "scripts/run_ground_litter_candidate_filter_shadow.py")
select = load("shadow_select", "scripts/select_ground_litter_shadow_review.py")
evaluate = load("shadow_evaluate", "scripts/evaluate_ground_litter_candidate_filter_shadow.py")


class LatestWinsTests(unittest.TestCase):
    def test_fast_worker_drops_nothing(self):
        result = shadow.simulate_latest_wins([0.1] * 5, 0.5)
        self.assertEqual(result["completed"], 5)
        self.assertEqual(result["dropped"], 0)
        self.assertEqual(result["maximum_queue_depth"], 0)

    def test_slow_worker_replaces_old_queue_item(self):
        result = shadow.simulate_latest_wins([1.2] * 6, 0.5)
        self.assertLess(result["completed"], 6)
        self.assertGreater(result["dropped"], 0)
        self.assertEqual(result["maximum_queue_depth"], 1)

    def test_invalid_service_time_is_rejected(self):
        with self.assertRaises(ValueError):
            shadow.simulate_latest_wins([float("nan")], 0.5)


class ShadowSelectionTests(unittest.TestCase):
    def rows(self, passed, count):
        return [{
            "proposal_id": f"p-{passed}-{index}", "passed": passed,
            "margin": (index + 1) * (1 if passed else -1),
        } for index in range(count)]

    def test_strata_are_disjoint_and_exhaustive(self):
        rows = self.rows(False, 7) + self.rows(True, 8)
        groups = select.semantic_strata(rows)
        ids = [row["proposal_id"] for group in groups.values() for row in group]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {row["proposal_id"] for row in rows})
        self.assertEqual(len(groups["filtered_near"]), 4)
        self.assertEqual(len(groups["passed_near"]), 4)

    def test_even_sampling_is_deterministic(self):
        rows = self.rows(True, 50)
        first = select.evenly(rows, 12, "x")
        second = select.evenly(list(reversed(rows)), 12, "x")
        self.assertEqual(
            [row["proposal_id"] for row in first],
            [row["proposal_id"] for row in second],
        )
        self.assertEqual(len(first), 12)


class ShadowDecisionTests(unittest.TestCase):
    def test_go_requires_recall_and_useful_reduction(self):
        overall = {"recall": 0.96, "negative_reduction": 0.30}
        cameras = {"01021": {"positive_rows": 5, "recall": 1.0}}
        decision, _ = evaluate.decide(overall, cameras, 20)
        self.assertEqual(decision, "GO_TO_TEMPORAL_SHADOW")

    def test_low_recall_is_no_go(self):
        overall = {"recall": 0.89, "negative_reduction": 0.40}
        cameras = {"01021": {"positive_rows": 10, "recall": 0.89}}
        decision, reasons = evaluate.decide(overall, cameras, 25)
        self.assertEqual(decision, "NO_GO")
        self.assertIn("OVERALL_RECALL_BELOW_0_90", reasons)

    def test_too_few_positives_is_inconclusive(self):
        overall = {"recall": 1.0, "negative_reduction": 0.40}
        cameras = {"01021": {"positive_rows": 3, "recall": 1.0}}
        decision, _ = evaluate.decide(overall, cameras, 10)
        self.assertEqual(decision, "INSUFFICIENT_POSITIVES")


if __name__ == "__main__":
    unittest.main()
