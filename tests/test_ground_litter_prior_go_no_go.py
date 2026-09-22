"""背景先验 Go/No-Go 实验脚本的最小必要测试。

只测试实验脚本自身的口径与缓存，不修改生产模块。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "gonogo", REPO / "scripts" / "evaluate_ground_litter_prior_go_no_go.py",
)
gonogo = importlib.util.module_from_spec(spec)
sys.modules["gonogo"] = gonogo
spec.loader.exec_module(gonogo)


def _profile_row(pid: str, *, matched: int = 0, candidate_count: int = 0,
                 enter: bool = False, score: float = 0.0) -> dict:
    return {
        "profile_id": pid, "score": score, "enter_eligible": enter,
        "hold_eligible": enter, "matched_candidates": matched,
        "candidate_support_at_target": matched * 10,
        "raw_seed_pixels": 2, "raw_support_pixels": 9,
        "candidate_count": candidate_count, "size_filtered": 0,
        "availability_fraction": 0.9,
    }


class PairingTests(unittest.TestCase):
    """成对原图/注入图不能串状态；oracle/active/output 严格分开。"""

    def test_oracle_active_output_are_separate(self):
        core = {
            "groups": [{"name": "A", "version": "v3"}],
            "results": [
                {
                    "group_version": "v3", "oracle_hit": True,
                    "baseline_ambiguous": False,
                    "native_side": 16,
                    "profiles": {"p0001": _profile_row("p0001", matched=0)},
                },
                {
                    "group_version": "v3", "oracle_hit": True,
                    "baseline_ambiguous": False,
                    "native_side": 16,
                    "profiles": {"p0001": _profile_row("p0001", matched=1)},
                },
            ],
        }
        replay = {"v3": {"active_profile_id": "p0001"}}
        metrics = gonogo.stage_metrics(core, replay)
        row = metrics["v3"]
        # oracle 看任意参考；active 看生效参考；两者在这份数据里不同。
        self.assertEqual(row["oracle_hits"], 2)
        self.assertEqual(row["active_hits"], 1)
        # 没有能力标注（legacy_unknown）时按"不门禁"处理，output 允许等于 active。
        self.assertEqual(row["prior_output_hits"], 1)

    def test_unsuitable_active_profile_is_not_an_output_hit(self):
        core = {
            "groups": [
                {"name": "B", "version": "v4", "prior_capable_ids": []},
            ],
            "results": [
                {
                    "group_version": "v4", "oracle_hit": True,
                    "baseline_ambiguous": False, "native_side": 16,
                    "profiles": {"p0003": _profile_row("p0003", matched=1)},
                },
            ],
        }
        replay = {"v4": {"active_profile_id": "p0003"}}
        row = gonogo.stage_metrics(core, replay)["v4"]
        self.assertEqual(row["active_hits"], 1)
        # active 命中了，但该参考 prior_suitable=false → 不算可输出。
        self.assertEqual(row["prior_output_hits"], 0)

    def test_suitable_active_profile_is_an_output_hit(self):
        core = {
            "groups": [
                {"name": "C", "version": "v3", "prior_capable_ids": ["p0011"]},
            ],
            "results": [
                {
                    "group_version": "v3", "oracle_hit": True,
                    "baseline_ambiguous": False, "native_side": 16,
                    "profiles": {"p0011": _profile_row("p0011", matched=1)},
                },
            ],
        }
        replay = {"v3": {"active_profile_id": "p0011"}}
        row = gonogo.stage_metrics(core, replay)["v3"]
        self.assertEqual(row["prior_output_hits"], 1)

    def test_baseline_ambiguity_is_flagged_and_counted(self):
        core = {
            "groups": [{"name": "A", "version": "v3"}],
            "results": [
                {
                    "group_version": "v3", "oracle_hit": True,
                    "baseline_ambiguous": True, "native_side": 16,
                    "profiles": {"p0001": _profile_row("p0001", matched=1)},
                },
                {
                    "group_version": "v3", "oracle_hit": True,
                    "baseline_ambiguous": False, "native_side": 16,
                    "profiles": {"p0001": _profile_row("p0001", matched=1)},
                },
            ],
        }
        metrics = gonogo.aggregate_metrics(core)["v3"]
        self.assertEqual(metrics["baseline_ambiguous"], 1)
        self.assertEqual(metrics["per_size"]["16"]["baseline_ambiguous"], 1)


class CacheTests(unittest.TestCase):
    """同一缓存键必须得到一致结果；键包含帧/Profile/配置/注入身份。"""

    def test_same_key_returns_identical_result(self):
        cache = gonogo.ProfileResultCache()
        key = (f"frame|cfg", "p0001")
        value = {"raw_support_pixels": 9, "candidates": [1, 2]}
        cache.put(key, value)
        first = cache.get(key)
        second = cache.get(key)
        self.assertEqual(first, value)
        self.assertIs(first, second)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 2)

    def test_different_injection_identity_is_a_different_key(self):
        cache = gonogo.ProfileResultCache()
        cache.put(("frame:a|cfg", "p0001"), {"v": 1})
        cache.put(("frame:b|cfg", "p0001"), {"v": 2})
        self.assertEqual(cache.get(("frame:a|cfg", "p0001"))["v"], 1)
        self.assertEqual(cache.get(("frame:b|cfg", "p0001"))["v"], 2)
        self.assertEqual(cache.misses, 2)

    def test_cache_hit_matches_fresh_computation(self):
        """缓存命中与重新计算必须一致（同一输入重复 put/get）。"""
        cache = gonogo.ProfileResultCache()
        row = _profile_row("p0001", matched=1, candidate_count=1)
        cache.put(("k|cfg", "p0001"), row)
        self.assertEqual(cache.get(("k|cfg", "p0001")), row)


class TemplateAndPositionTests(unittest.TestCase):
    def test_templates_are_deterministic_and_bounded(self):
        for name in gonogo.TEMPLATES:
            first = gonogo.build_template(
                name, 16, np.random.default_rng(1),
            )
            second = gonogo.build_template(
                name, 16, np.random.default_rng(1),
            )
            self.assertEqual(first.shape, (16, 16, 3))
            np.testing.assert_array_equal(first, second)

    def test_positions_are_inside_roi(self):
        roi = np.zeros((200, 320), np.uint8)
        roi[20:180, 20:300] = 255
        picks = gonogo.pick_positions(roi, np.random.default_rng(2))
        self.assertEqual(set(picks), set(gonogo.ROI_POSITIONS))
        for name, (x, y) in picks.items():
            self.assertEqual(roi[y, x], 255, name)

    def test_injection_box_is_inside_frame(self):
        frame = np.zeros((200, 320, 3), np.uint8)
        template = gonogo.build_template("light_paper", 24,
                                         np.random.default_rng(3))
        injected, truth = gonogo.inject_native(frame, (5, 5), template)
        box = truth["native_box"]
        self.assertGreaterEqual(box[0], 0)
        self.assertGreaterEqual(box[1], 0)
        self.assertLessEqual(box[2], 320)
        self.assertLessEqual(box[3], 200)
        self.assertGreater(float(injected.sum()), 0.0)


class DecisionTests(unittest.TestCase):
    """预注册判定：只用冻结门槛。"""

    def test_low_oracle_is_no_go(self):
        metrics = {"v3": {"per_size": {}}}
        stage = {"v3": {"oracle_hit_rate": 0.40, "prior_output_hit_rate": 0.30}}
        out = gonogo.decide(metrics, stage, {"groups": {}}, primary_version="v3")
        self.assertEqual(out["decision"], "NO_GO")

    def test_good_oracle_but_weak_output_is_conditional_selector(self):
        metrics = {"v3": {"per_size": {}}}
        stage = {"v3": {"oracle_hit_rate": 0.80, "prior_output_hit_rate": 0.30}}
        out = gonogo.decide(metrics, stage, {"groups": {}}, primary_version="v3")
        self.assertEqual(out["decision"], "CONDITIONAL_SELECTOR")

    def test_high_false_positives_give_verifier_or_no_go(self):
        metrics = {"v3": {"per_size": {}}}
        stage = {"v3": {"oracle_hit_rate": 0.80, "prior_output_hit_rate": 0.60}}
        negative = {"groups": {"v3": {"observed_seconds": 1800.0,
                                      "confirmed_events": 20}}}
        out = gonogo.decide(metrics, stage, negative, primary_version="v3")
        self.assertEqual(out["decision"], "NO_GO")
        negative2 = {"groups": {"v3": {"observed_seconds": 1800.0,
                                       "confirmed_events": 3}}}
        out2 = gonogo.decide(metrics, stage, negative2, primary_version="v3")
        self.assertEqual(out2["decision"], "CONDITIONAL_VERIFIER")

    def test_clean_result_is_go(self):
        metrics = {"v3": {"per_size": {}}}
        stage = {"v3": {"oracle_hit_rate": 0.80, "prior_output_hit_rate": 0.60}}
        negative = {"groups": {"v3": {"observed_seconds": 1800.0,
                                      "confirmed_events": 0}}}
        out = gonogo.decide(metrics, stage, negative, primary_version="v3")
        self.assertEqual(out["decision"], "GO")
        # 门槛没有被结果修改。
        self.assertEqual(
            out["thresholds"]["oracle_hit_rate_go"], 0.70,
        )


if __name__ == "__main__":
    unittest.main()
