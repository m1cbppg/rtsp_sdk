"""C1–C4 复核后定向修复的回归测试（2026-09-21）。

对应 `docs/plans/2026-09-21-profile-factory-v2-c1c4-repair-plan.md`。
评审反例脚本 `output/profile_factory_v2_review_20260920/reproduce_remaining.py`
保持只读；这里给出**修复后**的新断言，不改写旧脚本。
"""
from __future__ import annotations

import contextlib
import datetime as _datetime
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time as _time
import unittest

import cv2
import numpy as np

from rtsp_annotator.ground_litter_profile_background import (
    estimate_noise, select_calibration_observations,
)
from rtsp_annotator.ground_litter_profile_match import MatchEnvelope
from tests.test_ground_litter_profile_repairs import (
    SharedFrameReplayTests, _matcher,
)


def _candidates(count: int = 3, size: int = 160, *,
                prior_suitable: bool = True) -> list[dict]:
    """测试候选默认带 prior 能力（v4 契约后剪枝会检查它）。"""
    helper = SharedFrameReplayTests()
    rows = [helper._candidate(f"p{index + 1}") for index in range(count)]
    for row in rows:
        row["noise_calibration"] = {
            "source": (
                "calibration_day" if prior_suitable else "reference_self"
            ),
            "source_independent": True,
            "appearance_matched": bool(prior_suitable),
            "calibration_sufficient": bool(prior_suitable),
            "prior_suitable": bool(prior_suitable),
            "degradation_reason": None if prior_suitable
            else "no_appearance_match",
            "independent_blocks": 3 if prior_suitable else 0,
        }
    return rows


def _reference_frames(candidate: dict, count: int = 20) -> list[dict]:
    return [
        {
            "frame": candidate["reference"].copy(),
            "source_time": 1000.0 + index * 43.4,
            "tick_interval_seconds": 43.4,
        }
        for index in range(count)
    ]


class C1FairComparisonTests(unittest.TestCase):
    """C1：基线/留一法同帧同轴，保守逐次删除，终选与资源裁剪都复核。"""

    def test_baseline_and_loo_use_identical_frames(self):
        from scripts import build_ground_litter_profile_bank as build

        candidates = _candidates(3)
        frames = _reference_frames(candidates[0])
        report: dict = {}
        result = build.replay_all_candidates(
            SimpleNamespace(bank_id="b", version="v", max_profiles=24),
            candidates, frames, {"canvas_size": [160, 160], "roi": []},
            _matcher(), report, replay_w=160, replay_h=160, loo_stride=4,
        )
        # C1 反例：v2 的 baseline 用 stride=1、LOO 用 stride=4，帧不一致。
        self.assertEqual(report["replay"]["loo_stride"], 1)
        self.assertTrue(report["replay"]["loo_stride_ignored"])
        self.assertTrue(report["replay"]["loo_frames_identical_to_baseline"])
        digests = list(result["baseline_frame_digests"])
        self.assertEqual(len(digests), len(frames))
        for stats in result["leave_one_out"].values():
            self.assertEqual(stats["scored_frames"], len(frames))

    def test_score_matrix_caches_per_frame_per_candidate(self):
        from scripts import build_ground_litter_profile_bank as build

        candidates = _candidates(3)
        frames = _reference_frames(candidates[0], count=6)
        calls: list[tuple[int, str]] = []
        original = build.score_profile

        def traced(frame, *args, profile_id="", **kwargs):
            calls.append((int(frame[0, 0, 0]), profile_id))
            return original(frame, *args, profile_id=profile_id, **kwargs)

        with unittest.mock.patch.object(build, "score_profile", side_effect=traced):
            build.replay_all_candidates(
                SimpleNamespace(bank_id="b", version="v", max_profiles=24),
                candidates, frames, {"canvas_size": [160, 160], "roi": []},
                _matcher(), {}, replay_w=160, replay_h=160, leave_one_out=True,
            )
        # 3 候选 × 6 帧 = 18 次评分；不得因 baseline+LOO 重复评分而翻倍。
        self.assertEqual(len(calls), 18)

    def test_identical_reference_pair_deletes_at_most_one(self):
        from scripts import build_ground_litter_profile_bank as build

        candidates = _candidates(2)
        frames = _reference_frames(candidates[0])
        report: dict = {}
        config = SimpleNamespace(bank_id="b", version="v", max_profiles=24)
        result = build.replay_all_candidates(
            config, candidates, frames,
            {"canvas_size": [160, 160], "roi": []}, _matcher(), report,
            replay_w=160, replay_h=160, loo_stride=1,
        )
        # 反例核心：删掉任意一个完全相同的参考都不能让覆盖归零。
        self.assertGreater(result["baseline"]["effective_fraction"], 0.5)
        for stats in result["leave_one_out"].values():
            self.assertGreater(stats["effective_fraction_without"], 0.5)
        pruning: dict = {}
        kept = build.select_profiles_dynamically(config, candidates, result, pruning)
        self.assertEqual(
            pruning["pruning"]["method"],
            "conservative_one_at_a_time_dynamic_replay",
        )
        # C1（v3 复核）：两个完全可互相替代的参考必须**恰好**留下一个。
        # 早期版本用启动期覆盖做提前退出，导致一个都删不掉；这里用删减记录
        # 非空来验证真的发生了剪枝，而不是"至少保留一个"这种空断言。
        self.assertEqual(len(kept), 1, [item["profile_id"] for item in kept])
        removed = pruning["pruning"]["removed"]
        self.assertTrue(removed, "重复参考必须产生至少一条删除记录")
        self.assertEqual([row["reason"] for row in removed],
                         ["DYNAMICALLY_REDUNDANT"])
        self.assertTrue(pruning["pruning"]["deletion_order"])
        # 保留后的完整回放指标必须与删除前一致（重复参考被删不影响能力）。
        survivor = kept[0]
        timeline = build.project_selection_timeline(
            result["score_matrix"], [survivor["profile_id"]],
            selector_config=result["selector_config"],
            bank_id="b", bank_version="review", view_id="view_0",
            nominal_tick=result["nominal_tick_seconds"],
            join_gap_seconds=result["selector_config"]["join_gap_seconds"],
        )
        self.assertAlmostEqual(
            timeline["effective_fraction"],
            result["baseline"]["effective_fraction"],
            places=5,
        )
        self.assertEqual(
            timeline["pause_max"], result["baseline"]["pause_max"],
        )

    def test_conservative_pruning_removes_one_at_a_time_and_reverifies(self):
        from scripts import build_ground_litter_profile_bank as build

        # p1/p2 完全相同（互为冗余），p3 是唯一不同外观：逐次删除必须只删一个
        # 冗余项，且每轮都在剩余集合上重算 LOO。
        candidates = _candidates(2)
        third = _candidates(1)[0]
        third["profile_id"] = "p3"
        third["group_id"] = "g3"
        third["reference"] = np.clip(
            third["reference"].astype(np.int32) + 60, 0, 255,
        ).astype(np.uint8)
        third["context"] = type(third["context"])(
            "p3", third["reference"], third["valid"], {}, {},
        )
        candidates.append(third)
        frames = _reference_frames(candidates[0])
        config = SimpleNamespace(bank_id="b", version="v", max_profiles=24)
        replay: dict = {}
        result = build.replay_all_candidates(
            config, candidates, frames,
            {"canvas_size": [160, 160], "roi": []}, _matcher(), replay,
            replay_w=160, replay_h=160,
        )
        pruning: dict = {}
        kept = build.select_profiles_dynamically(config, candidates, result, pruning)
        order = pruning["pruning"]["deletion_order"]
        kept_ids = [item["profile_id"] for item in kept]
        # 互补参考必须保留，重复参考必须被删（删除记录非空）。
        self.assertIn("p3", kept_ids, kept_ids)
        self.assertTrue(pruning["pruning"]["removed"], "冗余参考必须被删除")
        removed_ids = [row["profile_id"] for row in pruning["pruning"]["removed"]]
        self.assertTrue(set(removed_ids) <= {"p1", "p2"}, removed_ids)
        self.assertEqual(sorted(removed_ids + kept_ids),
                         ["p1", "p2", "p3"])
        # 每一轮只能删一个，并记录删除后的复核指标。
        evaluated = [row["candidates_evaluated"] for row in order]
        self.assertTrue(all(len(row) >= 2 for row in evaluated))
        for removal in pruning["pruning"]["removed"]:
            self.assertIn("summary_after", removal)
            self.assertLess(removal["set_size_after"], len(candidates))
        # 删除的候选必须是在**当时剩余集合**上代价最小的那个。
        for row in order:
            self.assertEqual(row["chosen_reason"], "min_loo_cost_below_threshold")
            self.assertIn(row["removed"], row["candidates_evaluated"])

    def test_final_and_trimmed_sets_are_reverified_on_shared_matrix(self):
        from scripts import build_ground_litter_profile_bank as build

        candidates = _candidates(3)
        frames = _reference_frames(candidates[0])
        config = SimpleNamespace(bank_id="b", version="v", max_profiles=2)
        replay: dict = {}
        result = build.replay_all_candidates(
            config, candidates, frames,
            {"canvas_size": [160, 160], "roi": []}, _matcher(), replay,
            replay_w=160, replay_h=160,
        )
        pruning: dict = {}
        kept = build.select_profiles_dynamically(config, candidates, result, pruning)
        verification = build._verify_selected_subsets(
            config, candidates, kept, result,
            {"canvas_size": [160, 160], "roi": []}, _matcher(),
        )
        self.assertTrue(verification["final_set"]["verified_on_shared_score_matrix"])
        self.assertTrue(
            verification["resource_trimmed"]["verified_on_shared_score_matrix"]
        )
        self.assertTrue(
            verification["final_set"]["frames_identical_to_baseline"]
        )
        self.assertTrue(
            verification["resource_trimmed"]["frames_identical_to_baseline"]
        )
        self.assertLessEqual(len(verification["resource_trimmed_ids"]), 2)
        self.assertIn("coverage_delta_vs_baseline", verification["final_set"])


class PriorCapabilityContractTests(unittest.TestCase):
    """v4 复核：prior 能力必须进入剪枝目标与运行时门禁，不能只是报告字段。"""

    # -- 剪枝：两类覆盖同时约束 ------------------------------------------- #

    def _replay(self, build, candidates, frames, *, max_profiles=24):
        config = SimpleNamespace(bank_id="b", version="v", max_profiles=max_profiles)
        replay: dict = {}
        result = build.replay_all_candidates(
            config, candidates, frames,
            {"canvas_size": [160, 160], "roi": []}, _matcher(), replay,
            replay_w=160, replay_h=160,
        )
        return config, result

    def test_unique_prior_suitable_reference_is_never_deleted(self):
        """环境覆盖完全冗余、但唯一 prior_suitable=true 的参考必须留下。"""
        from scripts import build_ground_litter_profile_bank as build

        # p1/p2/p3 完全相同（环境上互相冗余），只有 p3 允许 prior。
        candidates = _candidates(3)
        for row in candidates[:2]:
            row["noise_calibration"]["prior_suitable"] = False
            row["noise_calibration"]["calibration_sufficient"] = False
            row["noise_calibration"]["source"] = "reference_self"
        self.assertEqual(
            build.prior_suitable_ids_of(candidates), ["p3"],
        )
        frames = _reference_frames(candidates[0])
        config, result = self._replay(build, candidates, frames)
        pruning: dict = {}
        kept = build.select_profiles_dynamically(config, candidates, result, pruning)
        kept_ids = {item["profile_id"] for item in kept}
        self.assertIn("p3", kept_ids, kept_ids)
        self.assertTrue(pruning["prior_bank"]["available"])
        self.assertEqual(
            pruning["prior_bank"]["prior_suitable_profiles_kept"], ["p3"],
        )
        # 环境覆盖冗余的两个 match-only 参考可以被删。
        self.assertTrue(pruning["pruning"]["removed"])

    def test_two_identical_prior_suitable_references_delete_one(self):
        """两个完全重复且都适合 prior：可以删一个，prior 覆盖不变。"""
        from scripts import build_ground_litter_profile_bank as build

        candidates = _candidates(2)
        frames = _reference_frames(candidates[0])
        config, result = self._replay(build, candidates, frames)
        pruning: dict = {}
        kept = build.select_profiles_dynamically(config, candidates, result, pruning)
        self.assertEqual(len(kept), 1)
        base_prior = result["baseline"]["prior"]["prior_effective_coverage"]
        after_prior = pruning["pruning"]["prior_effective_coverage"]
        self.assertAlmostEqual(base_prior, after_prior, places=5)
        self.assertEqual(pruning["prior_bank"]["prior_suitable_profiles_kept"],
                         [kept[0]["profile_id"]])

    def test_match_only_candidate_is_kept_when_constraints_allow(self):
        """match-only 参考不会被"prior 约束"或误判无条件删除。

        这一项不依赖轨迹打分器的动态（那部分由
        ``test_conservative_pruning_removes_one_at_a_time_and_reverifies``
        覆盖），而是直接检查剪枝决策记录：match-only 候选被删除时，
        **必然**是因为它在当时剩余集合上两类约束都满足，而不是被当成
        "没有 prior 能力"就无条件清理。
        """
        from scripts import build_ground_litter_profile_bank as build

        candidates = _candidates(3, prior_suitable=False)
        candidates[0]["noise_calibration"]["prior_suitable"] = True
        candidates[0]["noise_calibration"]["calibration_sufficient"] = True
        frames = _reference_frames(candidates[0])
        config, result = self._replay(build, candidates, frames)
        pruning: dict = {}
        kept = build.select_profiles_dynamically(config, candidates, result, pruning)
        decisions = {row["removed"]: row for row in pruning["pruning"]["deletion_order"]}
        for pid, decision in decisions.items():
            cost = decision["loo_cost"]
            # 每条删除都必须同时满足环境与 prior 两类阈值。
            self.assertLessEqual(cost["effective_fraction_delta"], 0.005)
            self.assertLessEqual(cost["pause_max_delta"], 5.0)
            self.assertLessEqual(cost["prior_coverage_delta"], 0.0)
            self.assertLessEqual(cost["prior_pause_delta"], 0.0)
            self.assertTrue(cost["prior_suitable_kept"])
        # 只要还有 prior-capable 候选，就不会被删空。
        self.assertTrue(pruning["prior_bank"]["available"])

    def test_no_prior_capable_candidate_is_reported_not_hidden(self):
        """候选阶段一个 suitable 都没有：必须明确 prior unavailable。"""
        from scripts import build_ground_litter_profile_bank as build

        candidates = _candidates(2, prior_suitable=False)
        self.assertEqual(build.prior_suitable_ids_of(candidates), [])
        frames = _reference_frames(candidates[0])
        config, result = self._replay(build, candidates, frames)
        pruning: dict = {}
        kept = build.select_profiles_dynamically(config, candidates, result, pruning)
        self.assertTrue(kept)
        self.assertFalse(pruning["prior_bank"]["available"])
        self.assertEqual(
            pruning["prior_bank"]["reason"], "NO_PRIOR_SUITABLE_CANDIDATE",
        )
        self.assertTrue(pruning.get("semantic_only"))
        self.assertTrue(pruning.get("pruning_warnings"))

    def test_resource_trim_keeps_a_prior_capable_profile(self):
        """资源裁剪不能把 prior-capable 集合删空。"""
        from scripts import build_ground_litter_profile_bank as build

        candidates = _candidates(4, prior_suitable=False)
        # 只有 p4 允许 prior。
        last = candidates[-1]["noise_calibration"]
        last["prior_suitable"] = True
        last["calibration_sufficient"] = True
        last["source"] = "calibration_day"
        frames = _reference_frames(candidates[0])
        config, result = self._replay(build, candidates, frames, max_profiles=2)
        pruning: dict = {}
        kept = build.select_profiles_dynamically(config, candidates, result, pruning)
        self.assertLessEqual(len(kept), 2)
        self.assertIn("p4", {item["profile_id"] for item in kept})
        self.assertTrue(pruning["prior_bank"]["available"])

    def test_pruning_report_exposes_both_capabilities(self):
        from scripts import build_ground_litter_profile_bank as build

        candidates = _candidates(3)
        frames = _reference_frames(candidates[0])
        config, result = self._replay(build, candidates, frames)
        pruning: dict = {}
        build.select_profiles_dynamically(config, candidates, result, pruning)
        report = pruning["pruning"]
        for key in ("match_coverage", "prior_effective_coverage",
                    "prior_pause_max", "semantic_only_intervals",
                    "prior_suitable_profiles_kept"):
            self.assertIn(key, report)
        self.assertIn("prior_bank_available", report)


class PriorRuntimeGateTests(unittest.TestCase):
    """运行时门禁：unsuitable active Profile 不能产生 prior-only 候选。"""

    def _selector(self, *, prior_ids, config=None):
        from rtsp_annotator.ground_litter_profile_selector import ProfileSelector

        return ProfileSelector(
            bank_id="b", bank_version="v", view_id="view_0",
            profile_ids=["p1", "p2"], config=config or {
                "tick_interval_seconds": 2.0, "result_validity_seconds": 4.0,
                "join_gap_seconds": 300.0,
            },
            prior_suitable_profile_ids=prior_ids,
        )

    def _feed(self, selector, profile_id, ticks=4, step=2.0, start=0.0):
        from rtsp_annotator.ground_litter_profile_selector import CandidateMatch

        decisions = []
        for index in range(ticks):
            timestamp = start + index * step
            current_id = selector.selected_profile_id
            match = CandidateMatch(
                profile_id, 0.1, True, True, verified=True,
            )
            current = (
                match if current_id == profile_id
                else (CandidateMatch(current_id, 0.1, True, True, verified=True)
                      if current_id else None)
            )
            decision = selector.observe(
                timestamp=timestamp, current=current, candidates=[match],
                tested_profile_ids=[profile_id],
            )
            if decision.commit_requested:
                selector.commit(
                    profile_id=decision.commit_profile_id, timestamp=timestamp,
                )
            decisions.append(decision)
        return decisions

    def test_unsuitable_active_profile_blocks_prior_output(self):
        selector = self._selector(prior_ids=["p2"])
        decisions = self._feed(selector, "p1", ticks=6)
        self.assertEqual(selector.selected_profile_id, "p1")
        last = decisions[-1]
        self.assertTrue(last.prior_allowed)          # 环境匹配仍然成立
        self.assertFalse(last.prior_available)       # 但 prior 被门禁
        self.assertIsNone(last.prior_profile_id)
        self.assertEqual(
            last.prior_unavailable_reason, "PROFILE_PRIOR_UNSUITABLE",
        )
        self.assertTrue(last.profile_match_available)
        summary = selector.prior_summary()
        self.assertEqual(summary["prior_effective_coverage"], 0.0)
        self.assertGreater(summary["semantic_only_seconds"], 0.0)
        reasons = {row["reason"] for row in summary["semantic_only_intervals"]}
        self.assertIn("PROFILE_PRIOR_UNSUITABLE", reasons)
        # 启动窗口还没有 active 参考，reason 必须是"无匹配"而不是"不适合"。
        self.assertIn("NO_MATCHED_PROFILE", reasons)

    def test_suitable_active_profile_allows_prior_output(self):
        selector = self._selector(prior_ids=["p1"])
        decisions = self._feed(selector, "p1", ticks=6)
        last = decisions[-1]
        self.assertTrue(last.prior_available)
        self.assertEqual(last.prior_profile_id, "p1")
        self.assertIsNone(last.prior_unavailable_reason)
        summary = selector.prior_summary()
        self.assertGreater(summary["prior_effective_coverage"], 0.0)

    def test_no_matched_profile_is_distinct_from_unsuitable(self):
        selector = self._selector(prior_ids=[])
        decisions = self._feed(selector, "p1", ticks=1)
        first = decisions[0]
        self.assertFalse(first.profile_match_available)
        self.assertEqual(
            first.prior_unavailable_reason, "NO_MATCHED_PROFILE",
        )

    def test_suitability_transition_advances_prior_generation(self):
        """suitable → unsuitable 后 prior 立即暂停，并推进 prior 代际。"""
        selector = self._selector(prior_ids=["p1"])
        self._feed(selector, "p1", ticks=4)
        generation_before = selector.prior_generation
        self.assertTrue(selector._prior_suitability_last)
        # 同一 Profile，但能力集合变化（模拟参考切到 unsuitable 参考）。
        selector.prior_suitable_profile_ids = frozenset()
        decisions = self._feed(selector, "p1", ticks=1, start=8.0)
        self.assertFalse(decisions[-1].prior_available)
        self.assertGreater(selector.prior_generation, generation_before)

    def test_missing_capability_set_keeps_legacy_behaviour(self):
        """没有能力集合（老调用/离线对照）时不加门禁，避免静默改变历史结果。"""
        selector = self._selector(prior_ids=None)
        decisions = self._feed(selector, "p1", ticks=6)
        self.assertTrue(decisions[-1].prior_available)


class LegacyBankCapabilityTests(unittest.TestCase):
    """历史 Bank 缺能力字段：默认拒绝，显式开关下保守按 false 读取。"""

    def test_legacy_bank_requires_explicit_compatibility_flag(self):
        import tempfile as _tempfile

        from rtsp_annotator.ground_litter_profile_bank import BankError, load_bank
        from tests.profile_bank_fixtures import build_synthetic_bank

        root = Path(_tempfile.mkdtemp()) / "banks"
        build_synthetic_bank(root, "camera_legacy", "v1", profiles=2)
        # 手工抹掉能力字段，模拟 v3 及更早的产物。
        import json as _json
        for pid in ("p0001", "p0002"):
            path = root / "camera_legacy" / "v1" / "profiles" / pid / "profile.json"
            payload = _json.loads(path.read_text(encoding="utf-8"))
            payload.pop("prior_suitable", None)
            payload.pop("calibration_state", None)
            path.write_text(_json.dumps(payload), encoding="utf-8")
        with self.assertRaises(BankError):
            load_bank(root, "camera_legacy", "v1", verify=False)
        bank = load_bank(
            root, "camera_legacy", "v1", verify=False,
            allow_legacy_profile_capabilities=True,
        )
        self.assertTrue(bank.legacy_capabilities)
        self.assertEqual(bank.prior_suitable_ids, ())
        for record in bank.profiles:
            self.assertFalse(record.prior_suitable)
            self.assertEqual(record.capability_source, "legacy_conservative")


class C2BoundedPipelineTests(unittest.TestCase):
    """C2：逐文件"准备→消费→释放"；背压不是失败；峰值真实受控。"""

    # -- 假远端素材 --------------------------------------------------------- #

    def _fake_remote(self, payload_frames: int = 14):
        """远程来源桩：清单客户端 + 可下载、可解码的假 PS。"""

        class FakeListClient:
            def __init__(self, files=(), **_kwargs):
                self.files = list(files)

        class FakeDownloader:
            def __init__(self):
                self.downloaded: list[str] = []

            def fetch_url_for_file(self, client, query, file_id, **kwargs):
                return SimpleNamespace(url="fake://" + file_id)

            def download(self, url, target, **kwargs):
                file_id = url.rsplit("/", 1)[-1]
                self.downloaded.append(file_id)
                target.parent.mkdir(parents=True, exist_ok=True)
                # 受管缓存落盘名是 *.bin，OpenCV 只按扩展名选后端：先写 .mp4。
                staging = target.with_name(target.name + ".tmp.mp4")
                writer = cv2.VideoWriter(
                    str(staging), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120),
                )
                rng = np.random.default_rng(7)
                for _ in range(payload_frames):
                    writer.write(rng.integers(60, 180, (120, 160, 3), dtype=np.uint8))
                writer.release()
                staging.replace(target)
                payload = target.read_bytes()
                import hashlib as _hashlib
                return SimpleNamespace(
                    size=len(payload), sha256=_hashlib.sha256(payload).hexdigest(),
                    range_supported=False, elapsed_seconds=0.01,
                )

        return FakeListClient, FakeDownloader

    @staticmethod
    def _reserve_bytes() -> int:
        from rtsp_annotator.ground_litter_recording_cache import (
            ManagedRecordingCache,
        )
        return int(ManagedRecordingCache.unknown_size_reserve)

    def _materializer(self, build, cache, tmp, files, *, budget, downloader):
        config = build.FactoryConfig(
            camera_id="cam_c2", bank_id="cam_c2", version="vc2",
            output_root=Path(tmp) / "banks", work_dir=Path(tmp) / "work",
            geometry_path=None, analysis_size=None,
            device_code="00000000000000000000",
            raw_cache_budget=budget,
        )
        args = SimpleNamespace(source="ctseelink-file-urls",
                               auth_token=None, api_key=None)
        return build.FileMaterializer(config, args, cache, {}, local_paths={}), config

    def test_materializer_consumes_and_releases_one_file_at_a_time(self):
        """4 个文件、配额只够 1 个：全部必须被消费，不能记 BUDGET_REJECTED。

        这就是 v3 复核复现出的缺陷（除第一个外全被标预算拒绝）的回归。
        """
        from scripts import build_ground_litter_profile_bank as build
        from rtsp_annotator.ground_litter_recording_cache import (
            ManagedRecordingCache,
        )
        from rtsp_annotator.ground_litter_recording_source import RecordingFile

        files = [
            RecordingFile(
                file_id=f"f{index}", file_name=f"f{index}.ps",
                record_start=f"2026-09-{14 + index} 10:00:00",
                record_end=f"2026-09-{14 + index} 10:05:00", file_size=None,
            )
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            client_cls, downloader_cls = self._fake_remote()
            downloader = downloader_cls()
            # 先量出一个文件的真实大小，再把配额压到一个多文件都放不下的量级。
            with ManagedRecordingCache(
                Path(tmp) / "probe", raw_cache_budget=10 ** 9,
            ) as probe:
                probe.register("00000000000000000000", files[:1])
                with unittest.mock.patch.object(
                    build, "RecordingListClient", client_cls,
                ), unittest.mock.patch.object(
                    build, "RecordingDownloader", lambda: downloader,
                ):
                    _, config = self._materializer(
                        build, probe, tmp, files, budget=10 ** 9,
                        downloader=downloader,
                    )
                    path = build.FileMaterializer(
                        config,
                        SimpleNamespace(source="ctseelink-file-urls",
                                        auth_token=None, api_key=None),
                        probe, {}, local_paths={},
                    ).need(files[0])
                one_size = path.stat().st_size
                self.assertGreater(one_size, 0)
            # 声明大小未知时，缓存按 unknown_size_reserve 预留；配额只够
            # 一个文件的预留（严格小于两个）。
            reserve = self._reserve_bytes()
            self.assertGreater(one_size, 0)
            budget = int(reserve * 1.5)
            downloader.downloaded.clear()
            # 量尺寸用的是独立缓存目录；正式流程从空缓存开始，四个文件都必须
            # 真的走一次下载（reused_cache 必须为 0）。
            with ManagedRecordingCache(
                Path(tmp) / "run", raw_cache_budget=budget,
            ) as cache:
                cache.register("00000000000000000000", files)
                materializer, config = self._materializer(
                    build, cache, tmp, files, budget=budget,
                    downloader=downloader,
                )
                with unittest.mock.patch.object(
                    build, "RecordingListClient", client_cls,
                ), unittest.mock.patch.object(
                    build, "RecordingDownloader", lambda: downloader,
                ):
                    for item in files:
                        path = materializer.need(item)
                        self.assertIsNotNone(path, f"{item.file_id} 未被消费")
                        self.assertTrue(Path(path).is_file())
                        # 立刻释放：下一个文件准备时缓存里只有它自己。
                        materializer.release(item)
                summary = materializer.summary()
            self.assertEqual(summary["need"], 4)
            self.assertEqual(summary["reused_cache"], 0)
            self.assertEqual(summary["downloaded"], 4)
            self.assertEqual(summary["failed"], [])
            self.assertEqual(summary["failure_kinds"], {})
            self.assertEqual(summary["released"], 4)
            self.assertLessEqual(summary["peak_raw_bytes"], budget)
            self.assertLess(config.raw_cache_budget, reserve * 2)
            self.assertEqual(sorted(downloader.downloaded), [f"f{i}" for i in range(4)])

    def test_backpressure_drives_consumption_instead_of_failure(self):
        """4 个文件、配额只够 2 个：全部必须被消费，不能记 BUDGET_REJECTED。

        这是 v3 复核复现出的缺陷（除第一个外全被标预算拒绝）的直接回归：
        背压必须先释放已消费条目再取下一个，而不是把后续文件记成失败。
        """
        from scripts import build_ground_litter_profile_bank as build
        from rtsp_annotator.ground_litter_recording_cache import (
            ManagedRecordingCache,
        )
        from rtsp_annotator.ground_litter_recording_source import RecordingFile

        files = [
            RecordingFile(
                file_id=f"f{index}", file_name=f"f{index}.ps",
                record_start=f"2026-09-{14 + index} 10:00:00",
                record_end=f"2026-09-{14 + index} 10:05:00", file_size=600,
            )
            for index in range(4)
        ]
        client_cls, downloader_cls = self._fake_remote()
        downloader = downloader_cls()
        with tempfile.TemporaryDirectory() as tmp:
            with ManagedRecordingCache(
                Path(tmp) / "run", raw_cache_budget=10 ** 9,
            ) as probe:
                probe.register("00000000000000000000", files[:1])
                probe_cfg = build.FactoryConfig(
                    camera_id="c", bank_id="c", version="v",
                    output_root=Path(tmp) / "banks", work_dir=Path(tmp) / "work",
                    geometry_path=None, analysis_size=None,
                    device_code="00000000000000000000",
                    raw_cache_budget=10 ** 9,
                )
                with unittest.mock.patch.object(
                    build, "RecordingListClient", client_cls,
                ), unittest.mock.patch.object(
                    build, "RecordingDownloader", lambda: downloader,
                ):
                    path = build.FileMaterializer(
                        probe_cfg,
                        SimpleNamespace(source="ctseelink-file-urls",
                                        auth_token=None, api_key=None),
                        probe, {}, local_paths={},
                    ).need(files[0])
                one_size = path.stat().st_size
            # 配额只够约 2 个真实文件（严格小于 3 个）。
            budget = int(one_size * 2.5)
            downloader.downloaded.clear()
            with ManagedRecordingCache(
                Path(tmp) / "run2", raw_cache_budget=budget,
            ) as cache:
                cache.register("00000000000000000000", files)
                config = build.FactoryConfig(
                    camera_id="c", bank_id="c", version="v",
                    output_root=Path(tmp) / "banks", work_dir=Path(tmp) / "work",
                    geometry_path=None, analysis_size=None,
                    device_code="00000000000000000000",
                    raw_cache_budget=budget,
                )
                materializer = build.FileMaterializer(
                    config,
                    SimpleNamespace(source="ctseelink-file-urls",
                                    auth_token=None, api_key=None),
                    cache, {}, local_paths={}, wait_timeout=5.0, max_retries=2,
                )
                with unittest.mock.patch.object(
                    build, "RecordingListClient", client_cls,
                ), unittest.mock.patch.object(
                    build, "RecordingDownloader", lambda: downloader,
                ):
                    for item in files:
                        path = materializer.need(item)
                        self.assertIsNotNone(
                            path, f"{item.file_id} 因容量被判失败",
                        )
                        self.assertTrue(Path(path).is_file())
                        # 消费完立刻释放：这就是"拉取→处理→删除"。
                        materializer.release(item)
                summary = materializer.summary()
            self.assertEqual(summary["need"], 4)
            self.assertEqual(summary["downloaded"], 4, str(summary["failed"]))
            self.assertEqual(summary["failure_kinds"], {}, str(summary["failed"]))
            self.assertEqual(summary["released"], 4)
            self.assertLess(budget, one_size * 3)
            self.assertLessEqual(summary["peak_raw_bytes"], budget)
            self.assertEqual(sorted(downloader.downloaded),
                             [f"f{i}" for i in range(4)])

    def test_remote_source_failure_is_classified_not_material_quality(self):
        from scripts import build_ground_litter_profile_bank as build
        from rtsp_annotator.ground_litter_recording_cache import (
            ManagedRecordingCache,
        )
        from rtsp_annotator.ground_litter_recording_source import RecordingFile

        item = RecordingFile(
            file_id="f0", file_name="f0.ps",
            record_start="2026-09-14 10:00:00",
            record_end="2026-09-14 10:05:00", file_size=1000,
        )

        class BrokenDownloader:
            def fetch_url_for_file(self, client, query, file_id, **kwargs):
                raise RuntimeError("link expired")

            def download(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("fetch 失败时不应继续下载")

        budget = int(self._reserve_bytes() * 1.5)
        with tempfile.TemporaryDirectory() as tmp:
            with ManagedRecordingCache(tmp, raw_cache_budget=budget) as cache:
                config = build.FactoryConfig(
                    camera_id="c", bank_id="c", version="v",
                    output_root=Path(tmp) / "banks", work_dir=Path(tmp) / "work",
                    geometry_path=None, analysis_size=None,
                    device_code="00000000000000000000",
                    raw_cache_budget=budget,
                )
                materializer = build.FileMaterializer(
                    config,
                    SimpleNamespace(source="ctseelink-file-urls",
                                    auth_token=None, api_key=None),
                    cache, {}, local_paths={},
                )
                with unittest.mock.patch.object(
                    build, "RecordingListClient", lambda **kwargs: object(),
                ), unittest.mock.patch.object(
                    build, "RecordingDownloader", BrokenDownloader,
                ):
                    path = materializer.need(item)
            self.assertIsNone(path)
            self.assertEqual(materializer.summary()["failure_kinds"],
                             {"URL_REFRESH_FAILED": 1})
            for row in materializer.summary()["failed"]:
                self.assertNotEqual(row["kind"], "MATERIAL_INVALID")

    def test_whole_factory_chain_consumes_every_needed_file_above_quota(self):
        """全链路、真实受管缓存、远端假下载器、配额只够约 1 个文件。

        必须覆盖高清、校准、回放三个阶段，并断言**所有本应成功的文件都被真的
        消费**（不是"有终态"），且没有文件因缓存容量被漏掉。
        """
        from scripts import build_ground_litter_profile_bank as build
        from rtsp_annotator.ground_litter_recording_cache import (
            ManagedRecordingCache,
        )

        client_cls, downloader_cls = self._fake_remote(payload_frames=12)
        inventory = []
        for index, (day, hour) in enumerate((
            (14, 10), (14, 14), (15, 10), (15, 14),
            (16, 10), (17, 10), (18, 10), (19, 10),
        )):
            inventory.append(SimpleNamespace(
                file_id=f"f{index:02d}", file_name=f"f{index:02d}.ps",
                record_start=f"2026-09-{day} {hour:02d}:00:00",
                record_end=f"2026-09-{day} {hour:02d}:05:00",
                file_size=512, file_type="ps",
                as_dict=lambda self=None: {},
            ))

        class FakeListClient:
            def __init__(self, **kwargs):
                pass

            def query(self, query):  # pragma: no cover - 走清单缓存
                return SimpleNamespace(entries=[])

        def _fake_inventory_stage(config, client, cache, report, entries=None):
            """替真实清单阶段：把清单写进 report 并返回文件列表。"""
            report["inventory"] = {
                "mode": "fake_remote",
                "files": len(inventory),
                "total_declared_bytes": sum(
                    int(item.file_size or 0) for item in inventory
                ),
            }
            return list(inventory)
        downloader = downloader_cls()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            work.mkdir(parents=True)
            # 先量单文件大小，再按"只够约一个"设配额。
            with ManagedRecordingCache(
                root / "probe", raw_cache_budget=10 ** 9,
            ) as probe:
                probe.register("44180209031322001030", inventory[:1])
                probe_cfg = build.FactoryConfig(
                    camera_id="cam_chain", bank_id="cam_chain", version="vchain",
                    output_root=root / "banks", work_dir=root / "work",
                    geometry_path=None, analysis_size=None,
                    device_code="44180209031322001030",
                    raw_cache_budget=10 ** 9,
                )
                with unittest.mock.patch.object(
                    build, "RecordingListClient", FakeListClient,
                ), unittest.mock.patch.object(
                    build, "RecordingDownloader", lambda: downloader,
                ):
                    first = build.FileMaterializer(
                        probe_cfg,
                        SimpleNamespace(source="ctseelink-file-urls",
                                        auth_token=None, api_key=None),
                        probe, {}, local_paths={},
                    ).need(inventory[0])
                one_size = first.stat().st_size
            # 8 个文件的真实总输入约 1.1MB；配额只够一个文件（放不下第二个）。
            budget = int(one_size * 1.5)
            self.assertGreater(one_size, 0)
            self.assertGreater(len(inventory) * one_size, budget * 4)
            downloader.downloaded.clear()

            config = build.FactoryConfig(
                camera_id="cam_chain", bank_id="cam_chain", version="vchain",
                output_root=root / "banks", work_dir=root / "work",
                geometry_path=None, analysis_size=None,
                device_code="44180209031322001030",
                raw_cache_budget=budget,
                work_budget=512 * 1024 ** 2,
                max_profiles=6,
                calibration_day="2026-09-18",
                max_files=8,
            )
            args = SimpleNamespace(
                input=None, source="ctseelink-file-urls",
                device_code=config.device_code, start="", end="",
                auth_token=None, api_key=None,
                prefetch_slots=1, max_replay_frames=12, loo_stride=1,
                resume=False, supersede=False, seed=config.seed,
                calibration_day="2026-09-18", per_day_hours=0, max_files=8,
            )
            with unittest.mock.patch.object(
                build, "RecordingListClient", FakeListClient,
            ), unittest.mock.patch.object(
                build, "RecordingDownloader", lambda: downloader,
            ), unittest.mock.patch.object(
                build, "find_first_decodable",
                lambda *a, **k: ((160, 120), "stub"),
            ), unittest.mock.patch.object(
                build, "stage_inventory_remote",
                _fake_inventory_stage,
            ), contextlib.redirect_stdout(io.StringIO()):
                report = build.run_factory(config, args)

            # 配额严格小于两个文件，而全部输入是 8 个文件。
            self.assertLess(budget, one_size * 2)
            material = report["materialization_summary"]
            self.assertIn("composite", material)
            self.assertIn("replay_collect", material)
            # 所有"需要"的构建+校准文件都必须真的被消费过。
            needed = set(report["material_plan"]["needed_file_ids"])
            composite = material["composite"]
            self.assertEqual(composite["failure_kinds"], {},
                             str(composite["failed"]))
            self.assertEqual(report["hd_repull"], composite)
            # 合成阶段的素材来自采样期已有的高清帧；按需取用时要么命中缓存、
            # 要么现拉，绝不允许"没拿到就跳过"。
            self.assertEqual(composite["reused_cache"] + composite["reused_local"]
                             + composite["downloaded"], composite["need"])
            # 回放阶段逐文件消费，不能因容量漏文件。
            replay_mat = report["replay_materialize"]
            self.assertEqual(replay_mat["failure_kinds"], {},
                             str(replay_mat["failed"]))
            self.assertEqual(replay_mat["consumed_files"],
                             replay_mat["planned_files"])
            self.assertGreater(replay_mat["planned_files"], 1)
            self.assertGreater(replay_mat["released"], 0)
            # 校准阶段逐文件消费。
            cal = report["calibration_material"]
            self.assertEqual(cal["planned"], 1)
            self.assertEqual(cal["consumed"], 1, cal)
            # 盲测日文件不在 needed 集合里，也从未被下载。
            blind = set(report["material_plan"]["blind_files_excluded_from_hd"])
            self.assertFalse(blind & needed)
            self.assertFalse(blind & set(downloader.downloaded))
            # 实际峰值受控（不是只看记账字段）。
            envelope = report["resource_envelope"]
            self.assertLessEqual(envelope["peak_raw_bytes"], budget)
            self.assertLessEqual(envelope["peak_work_bytes"], config.work_budget)
            self.assertEqual(envelope["final_raw_bytes"], 0)
            # 逐文件释放必须真的把临时盘清空，而不是只写记账。
            self.assertEqual(report["space"]["budget"]["raw_bytes"], 0)

    def test_collect_replay_frames_consumes_each_file_once(self):
        """回放逐文件：准备→抽帧→释放，同一文件不重复下载。"""
        from scripts import build_ground_litter_profile_bank as build
        from rtsp_annotator.ground_litter_recording_cache import (
            ManagedRecordingCache,
        )
        from rtsp_annotator.ground_litter_recording_source import RecordingFile

        files = [
            RecordingFile(
                file_id=f"r{index}", file_name=f"r{index}.ps",
                record_start=f"2026-09-1{4 + index} 00:00:00",
                record_end=f"2026-09-1{4 + index} 00:05:00", file_size=None,
            )
            for index in range(3)
        ]
        client_cls, downloader_cls = self._fake_remote()
        downloader = downloader_cls()
        reserve = self._reserve_bytes()
        budget = int(reserve * 1.5)
        with tempfile.TemporaryDirectory() as tmp:
            with ManagedRecordingCache(tmp, raw_cache_budget=budget) as cache:
                config = build.FactoryConfig(
                    camera_id="cam_replay", bank_id="cam_replay", version="vreplay",
                    output_root=Path(tmp) / "banks", work_dir=Path(tmp) / "work",
                    geometry_path=None, analysis_size=None,
                    device_code="00000000000000000000",
                    raw_cache_budget=budget,
                )
                args = SimpleNamespace(
                    input=None, source="ctseelink-file-urls",
                    auth_token=None, api_key=None,
                )
                with unittest.mock.patch.object(
                    build, "RecordingListClient", client_cls,
                ), unittest.mock.patch.object(
                    build, "RecordingDownloader", lambda: downloader,
                ):
                    report: dict = {}
                    rows = build._collect_replay_frames(
                        cache, config, files,
                        {"canvas_size": [160, 120], "roi": []}, frame_budget=8,
                        args=args, local_paths={}, report=report,
                    )
                self.assertEqual(cache.managed_bytes(), 0)
            self.assertGreater(len(rows), 0)
            materialize = report["replay_materialize"]
            self.assertEqual(materialize["planned_files"], 3)
            self.assertEqual(materialize["consumed_files"], 3)
            self.assertEqual(materialize["downloaded"], 3)
            self.assertEqual(materialize["released"], 3)
            self.assertEqual(materialize["failed"], [])
            self.assertEqual(sorted(downloader.downloaded), ["r0", "r1", "r2"])
            # 实测峰值只包含一个文件，而不是三个。
            self.assertLessEqual(materialize["peak_raw_bytes"], budget)
            self.assertLess(materialize["peak_raw_bytes"], 3 * 100000)


class C3NoiseCalibrationTests(unittest.TestCase):
    """C3：噪声来自独立校准观测；外观匹配；退化原因必须显式。"""

    def _group(self, group_id: str, blocks: list[str]) -> dict:
        return {
            "group_id": group_id, "representative_key": blocks[0],
            "members": list(blocks), "time_blocks": list(blocks),
            "support": {}, "low_support": False, "days": ["2026-09-14"],
        }

    def test_calibration_sources_are_separate_from_synthesis(self):
        # 同一外观的校准观测 → 分配给同一组。
        reference = np.full((64, 64, 3), 90, np.uint8)
        descriptor = {
            "grid_luminance": np.full((9, 16), 90.0, np.float32),
            "grid_chroma": np.zeros((9, 16), np.float32),
            "grid_structure": np.zeros((9, 16), np.float32),
            "grid_weight": np.ones((9, 16), np.float32),
        }
        calibration = [
            {"time_block": f"cal@0000000{index}", "descriptor": descriptor}
            for index in range(3)
        ]
        groups = [self._group("g001", ["build@00000001"])]
        assignment = select_calibration_observations(
            groups, calibration, descriptors={
                "build@00000001": descriptor,
                "cal@00000000": descriptor, "cal@00000001": descriptor,
                "cal@00000002": descriptor,
            },
            group_members={"g001": ["build@00000001"]},
        )
        payload = assignment["per_group"]["g001"]
        self.assertEqual(len(payload["blocks"]), 3)
        self.assertIsNone(payload["reason"])
        self.assertEqual(assignment["unassigned_observations"], 0)

    def test_unmatched_calibration_is_reported_not_silently_used(self):
        reference = np.zeros((9, 16), np.float32)
        far = {
            "grid_luminance": np.full((9, 16), 250.0, np.float32),
            "grid_chroma": np.zeros((9, 16), np.float32),
            "grid_structure": np.zeros((9, 16), np.float32),
            "grid_weight": np.ones((9, 16), np.float32),
        }
        near = dict(far, grid_luminance=np.full((9, 16), 10.0, np.float32))
        groups = [{
            "group_id": "g001", "representative_key": "build@1",
            "members": ["build@1"], "time_blocks": ["build@1"],
        }]
        assignment = select_calibration_observations(
            groups,
            [{"time_block": "cal@1", "descriptor": far},
             {"time_block": "cal@2", "descriptor": far}],
            descriptors={"build@1": near, "cal@1": far, "cal@2": far},
            group_members={"g001": ["build@1"]},
        )
        payload = assignment["per_group"]["g001"]
        self.assertEqual(payload["blocks"], [])
        self.assertEqual(payload["reason"], "no_appearance_match")
        self.assertEqual(assignment["unassigned_observations"], 2)
        self.assertTrue(payload["near_misses"])

    def test_unmatched_calibration_cannot_change_normal_noise_thresholds(self):
        """外观不匹配的校准帧不能改变该参考的正常噪声阈值（C3）。

        夹具：参考是均匀冷色调，校准帧带明显暖色偏移 + 大块亮斑。
        如果这段素材被用来学容差，亮度阈值会被整体抬高；正确行为是
        退化为参考自身观测 + 基础阈值，并把该参考标成暂不适合 prior。
        """
        from scripts import build_ground_litter_profile_bank as build
        from rtsp_annotator.ground_litter_profile_background import (
            NOISE_DEFAULTS, estimate_noise,
        )
        from rtsp_annotator.ground_litter_profile_match import (
            apply_compensation, residual_maps, robust_color_compensation,
        )

        rng = np.random.default_rng(17)
        reference = np.clip(
            rng.integers(70, 110, (128, 128, 3), dtype=np.int32)
            + np.array([20, 0, -20], np.int32),
            0, 255,
        ).astype(np.uint8)
        valid = np.full((128, 128), 255, np.uint8)
        warm = np.clip(
            reference.astype(np.int32) + np.array([-40, 0, 40], np.int32),
            0, 255,
        ).astype(np.uint8)
        warm[40:88, 40:88] = 250

        # 参考自身观测（基础阈值路径）
        baseline = estimate_noise(
            reference, [reference, reference], [valid, valid], ["b1", "b2"],
            stride=1, valid=valid,
        )
        # 不匹配的校准素材若被错误使用，会把这些阈值抬高：
        unmatched = estimate_noise(
            reference, [warm, warm], [valid, valid], ["cal1", "cal2"],
            stride=1, valid=valid,
        )
        base_median = float(np.median(baseline.payload["seed_luminance_median"]))
        unmatched_median = float(
            np.median(unmatched.payload["seed_luminance_median"])
        )
        self.assertGreater(
            unmatched_median, base_median * 3.0,
            "夹具必须真的能演示'跨外观素材会显著抬高残差中心'",
        )
        # 正确行为：判定逻辑必须把"来源独立"和"外观匹配"分开，并且在外观
        # 不匹配时退回参考自身观测（不是拿这段暖色素材学容差）。
        state = build.classify_calibration_state(
            matched_blocks=[], matched_frame_count=0,
            calibration_available=True,
            assignment_reason="no_appearance_match",
        )
        self.assertEqual(state["source"], "reference_self")
        self.assertTrue(state["source_independent"])
        self.assertFalse(state["appearance_matched"])
        self.assertFalse(state["calibration_sufficient"])
        self.assertFalse(state["prior_suitable"])
        self.assertEqual(state["degradation_reason"], "no_appearance_match")
        # 有匹配素材（>=2 块、>=2 帧）时才允许用外部素材学容差。
        ok = build.classify_calibration_state(
            matched_blocks=["c1", "c2"], matched_frame_count=2,
            calibration_available=True,
        )
        self.assertTrue(ok["calibration_sufficient"])
        self.assertTrue(ok["prior_suitable"])
        self.assertEqual(ok["source"], "calibration_day")
        # 只有一块匹配素材也算不够：单块校准学不出尾部分位。
        single = build.classify_calibration_state(
            matched_blocks=["c1"], matched_frame_count=3,
            calibration_available=True,
        )
        self.assertFalse(single["calibration_sufficient"])
        self.assertEqual(single["degradation_reason"],
                         "single_matched_observation")

    def test_noise_residual_uses_shared_compensation(self):
        """运行时前景支持与离线噪声估计必须用同一残差定义（C3）。"""
        from rtsp_annotator.ground_litter_profile_analysis import (
            BankPriorContext, evaluate_bank_frame,
        )
        from rtsp_annotator.ground_litter_profile_match import residual_maps

        rng = np.random.default_rng(11)
        reference = rng.integers(70, 130, (128, 128, 3), dtype=np.uint8)
        frame = reference.copy()
        frame[64:76, 64:76] = 240
        # 注入区不算入共同拟合区：这样 fit_mask 仍然足够大（否则会
        # BankError("共同拟合区域不足")），但注入块确实是可用的前景证据。
        valid = np.full((128, 128), 255, np.uint8)
        valid[64:76, 64:76] = 0
        signature, luminance = residual_maps(reference, frame)
        estimate = estimate_noise(
            reference, [frame], [valid], ["cal@1"], stride=1,
            config={"_independent_calibration": True},
        )
        self.assertIn("seed_signature_median", estimate.payload)
        # 残差中心必须来自 residual_maps 的同一输出，而不是另一套减法。
        context = BankPriorContext("p", reference, valid, {}, {})
        evaluation = evaluate_bank_frame(
            context, frame,
            envelope=MatchEnvelope(1e6, 1e6, True, 10, "unit"),
            config=_matcher(), roi_mask=valid,
        )
        self.assertGreater(float(signature.max()), 0.0)
        self.assertGreater(float(luminance.max()), 0.0)
        # 运行时评分必须暴露与离线估计同一套受限补偿；residual_maps 在两边
        # 都作用在全局补偿后的图像上，因此这里的 gains/biases 必须存在。
        self.assertIsNotNone(evaluation.score)
        self.assertIn("compensation", evaluation.score)
        self.assertIn("gains", evaluation.score["compensation"])
        self.assertIn("biases", evaluation.score["compensation"])
        self.assertIn("diagnostics", evaluation.score)

    def test_offline_and_online_residuals_match_after_shared_compensation(self):
        """离线噪声与在线前景必须给出同一套残差（C3 核心）。

        夹具带全局光照偏移：未补偿时离线亮度残差中位数约 2.31，在线（补偿后）
        约 0.50——两者不是同一个量。修复后离线估计也必须走
        ``robust_color_compensation + apply_compensation``，于是同一帧的
        残差分布必须与在线路径一致（逐元素相等，不是近似）。
        """
        from rtsp_annotator.ground_litter_profile_analysis import (
            BankPriorContext, evaluate_bank_frame,
        )
        from rtsp_annotator.ground_litter_profile_background import estimate_noise
        from rtsp_annotator.ground_litter_profile_match import (
            MatchEnvelope, apply_compensation, residual_maps,
            robust_color_compensation,
        )

        rng = np.random.default_rng(23)
        reference = rng.integers(80, 140, (160, 160, 3), dtype=np.uint8)
        # 整体亮度上移：参考自身观测的"未补偿残差"会明显大于补偿后的残差。
        shifted = np.clip(reference.astype(np.int32) + 14, 0, 255).astype(np.uint8)
        valid = np.full((160, 160), 255, np.uint8)

        # 旧口径（直接对原始参考做差）只用于展示差异量级。
        _, raw_lum = residual_maps(reference, shifted)
        raw_median = float(np.median(raw_lum))

        offline = estimate_noise(
            reference, [shifted], [valid], ["cal@1"], stride=1, valid=valid,
        )
        diag = offline.diagnostics["shared_compensation"]
        self.assertTrue(diag["applied"])
        offline_median = float(
            np.median(offline.payload["seed_luminance_median"])
        )

        # 在线路径：评分暴露的补偿参考 + 同一 residual_maps。
        context = BankPriorContext("p", reference, valid, {}, {})
        evaluation = evaluate_bank_frame(
            context, shifted,
            envelope=MatchEnvelope(1e6, 1e6, True, 10, "unit"),
            config=_matcher(), roi_mask=valid,
        )
        self.assertIsNotNone(evaluation.score)
        compensation = evaluation.score["compensation"]
        online_reference = apply_compensation(
            reference,
            robust_color_compensation(
                reference, shifted,
                np.where(valid > 0, 255, 0).astype(np.uint8),
            ),
        )
        _, online_lum = residual_maps(online_reference, shifted)
        # 补偿系数必须与在线一致（同一拟合函数、同一拟合区）。
        self.assertAlmostEqual(compensation["gains"][0], 1.0, places=3)
        self.assertAlmostEqual(compensation["biases"][0], 14.0, places=1)

        # 关键断言：离线学到的残差中心 == 在线补偿后的残差中心。
        self.assertLess(offline_median, raw_median * 0.25,
                        "修复必须把离线残差拉到补偿后的量级")
        self.assertAlmostEqual(
            offline_median, float(np.median(online_lum)), places=4,
        )
        # 每个像素都必须一致（不是只有中位数像）。
        np.testing.assert_allclose(
            np.asarray(offline.payload["seed_luminance_median"]),
            cv2.resize(
                online_lum, (160, 160), interpolation=cv2.INTER_AREA,
            ),
            atol=1e-3, rtol=0,
        )

    def test_low_support_profiles_keep_base_thresholds(self):
        reference = np.full((48, 48, 3), 100, np.uint8)
        reason = {
            "grid_luminance": np.full((9, 16), 100.0, np.float32),
            "grid_chroma": np.zeros((9, 16), np.float32),
            "grid_structure": np.zeros((9, 16), np.float32),
            "grid_weight": np.ones((9, 16), np.float32),
        }
        groups = [{
            "group_id": "g1", "representative_key": "b@1", "members": ["b@1"],
            "time_blocks": ["b@1"],
        }]
        assignment = select_calibration_observations(
            groups, [], descriptors={"b@1": reason},
            group_members={"g1": ["b@1"]},
        )
        self.assertEqual(
            assignment["per_group"]["g1"]["reason"], "no_calibration_samples",
        )


class SmallTargetDiagnosticTests(unittest.TestCase):
    """第六项：小目标损失定位诊断必须真的跑起来并给出四类对照。"""

    def test_diagnostic_reports_four_comparisons_and_loss_stage(self):
        import tempfile as _tempfile

        import cv2 as _cv2

        from scripts import diagnose_ground_litter_small_target as diag
        from tests.profile_bank_fixtures import build_synthetic_bank

        root = Path(_tempfile.mkdtemp())
        build_synthetic_bank(
            root / "banks", "camera_diag", "v1", profiles=3,
            width=256, height=192,
        )
        media = root / "ps"
        media.mkdir()
        rng = np.random.default_rng(9)
        frame = rng.integers(80, 150, (192, 256, 3), dtype=np.uint8)
        frame = _cv2.GaussianBlur(frame, (0, 0), 1.2)
        frame[::12, :] = (frame[::12, :] // 2 + 40).astype(np.uint8)
        _cv2.imwrite(str(media / "2026-09-19T100000.png"), frame)
        output = root / "out"
        args = diag.parse_args_from([
            "--bank-root", str(root / "banks"), "--bank-id", "camera_diag",
            "--version", "v1", "--media", str(media), "--frame-index", "0",
            "--native-size-px", "8,24", "--online-size", "128x96",
            "--output", str(output),
        ])
        report = diag.diagnose(args)
        modes = {row["mode"] for row in report["comparisons"]}
        self.assertEqual(
            modes, {"A_native_prior_adapter", "B_online_analysis_path"},
        )
        self.assertEqual(len(report["comparisons"]), 4)
        for row in report["comparisons"]:
            # 每个对照都有同帧未注入基线。
            self.assertIn("baseline_candidates_any", row)
            self.assertIn("search_candidates", row)
            self.assertIn(row["loss_stage"], diag.STAGES)
            self.assertTrue(row["injection"]["inside_roi"])
            self.assertTrue(row["best_reference"])
            # 原生注入与在线注入记录的是同一物理尺寸。
            self.assertAlmostEqual(
                row["injection"]["scaled_size_px"],
                row["native_size_px"]
                * row["canvas_size"][0] / 256.0, places=3,
            )
            self.assertIn("seed_pixels_at_target", row["profiles"][
                row["best_reference"]["profile_id"]
            ])
        # 在线画布的那个对照必须记录"缩放到几个像素"。
        online = [
            row for row in report["comparisons"]
            if row["canvas"] == "online" and row["native_size_px"] == 8
        ][0]
        self.assertEqual(online["injected_side_on_canvas"], 4)
        self.assertFalse(online["injection"]["sub_pixel_at_canvas"])
        self.assertTrue((output / "small_target_diagnosis.json").is_file())


class C4HitReportingTests(unittest.TestCase):
    """C4：潜力/可用命中分离、prior_allowed 门槛、同帧成对基线。"""

    def _tick(self, boxes, *, profile_id="p1", prior_allowed=True, count=None):
        return {
            "profile_id": profile_id,
            "prior_allowed": prior_allowed,
            "candidate_boxes": len(boxes) if count is None else count,
            "candidate_boxes_detail": [
                {"box": list(box), "profile_id": pid} for box, pid in boxes
            ],
            "availability_max": 1.0,
            "blocked_availability": {},
        }

    def test_potential_and_effective_hits_are_separate(self):
        from scripts import evaluate_ground_litter_profile_bank as evaluate

        truth = {"box": [40, 40, 48, 48], "native_side": 8,
                 "injection_scale": "native_then_resize"}
        # 候选来自"非生效参考"，只能算潜力命中。
        tick = self._tick([([41, 41, 49, 49], "p2")], profile_id="p1")
        row = evaluate._match_target(
            truth, tick, baseline_tick=self._tick([]), tick_index=0,
            file_id="f", source_time=1.0, active_profile_id="p1",
        )
        self.assertTrue(row["potential_hit"])
        self.assertFalse(row["effective_hit"])
        self.assertEqual(row["potential_hit_profiles"], ["p2"])

        # 候选来自生效参考 → 两个口径都成立。
        good = self._tick([([41, 41, 49, 49], "p1")], profile_id="p1")
        row = evaluate._match_target(
            truth, good, baseline_tick=self._tick([]), tick_index=0,
            file_id="f", source_time=1.0, active_profile_id="p1",
        )
        self.assertTrue(row["potential_hit"])
        self.assertTrue(row["effective_hit"])

    def test_effective_hits_require_prior_allowed(self):
        from scripts import evaluate_ground_litter_profile_bank as evaluate

        truth = {"box": [10, 10, 18, 18]}
        tick = self._tick([([11, 11, 19, 19], "p1")], profile_id="p1",
                          prior_allowed=False)
        row = evaluate._match_target(
            truth, tick, baseline_tick=self._tick([], prior_allowed=False),
            tick_index=0, file_id="f", source_time=1.0,
            active_profile_id="p1",
        )
        self.assertTrue(row["potential_hit"])
        self.assertFalse(row["effective_hit"])

    def test_summary_counts_all_rows_and_reports_both_fractions(self):
        from scripts import evaluate_ground_litter_profile_bank as evaluate

        rows = [
            {"potential_hit": True, "effective_hit": True,
             "baseline_candidate_boxes": 0, "any_candidate": True,
             "prior_allowed": True, "injection_scale": "native_then_resize",
             "active_profile_id": "p1", "effective_hit_profiles": ["p1"],
             "paired_baseline_available": True},
            {"potential_hit": True, "effective_hit": False,
             "baseline_candidate_boxes": None, "any_candidate": True,
             "prior_allowed": False, "injection_scale": "native_then_resize",
             "active_profile_id": "p2", "effective_hit_profiles": [],
             "paired_baseline_available": False},
            {"potential_hit": False, "effective_hit": False,
             "baseline_candidate_boxes": 1, "any_candidate": False,
             "prior_allowed": True, "injection_scale": "canvas",
             "active_profile_id": "p1", "effective_hit_profiles": [],
             "paired_baseline_available": True},
        ]
        summary = evaluate._small_target_summary(rows)
        self.assertEqual(summary["rows_total"], 3)
        self.assertEqual(summary["rows_scored"], 3)
        self.assertEqual(summary["rows_dropped"], 1)
        self.assertEqual(summary["potential_hits"], 2)
        self.assertEqual(summary["effective_hits"], 1)
        self.assertEqual(summary["hits_when_prior_not_allowed"], 1)
        self.assertEqual(summary["potential_but_not_effective"], 1)
        self.assertEqual(summary["native_scale_injections"], 2)

    def test_native_scale_injection_is_bigger_than_canvas_block(self):
        from scripts import evaluate_ground_litter_profile_bank as evaluate

        rng = np.random.default_rng(3)
        native = rng.integers(60, 140, (240, 320, 3), dtype=np.uint8)
        roi_native = np.full((240, 320), 255, np.uint8)
        frame, _injected_native, truth = evaluate.inject_small_target_native_scale(
            native, (80, 60), roi_native, 16, rng,
        )
        self.assertIsNotNone(truth)
        self.assertEqual(truth["injection_scale"], "native_then_resize")
        self.assertEqual(truth["native_side"], 16)
        self.assertEqual(frame.shape[:2], (60, 80))
        # 画布上的注入框边长应约为 16 × (80/320) = 4px，而不是 16px。
        canvas_side = truth["box"][2] - truth["box"][0]
        self.assertLess(canvas_side, 6.0)
        self.assertGreater(canvas_side, 2.0)

    def test_paired_baseline_helper_uses_same_frame(self):
        """成对基线必须来自同一帧：这里核对评估循环确实复制 selector。"""
        import inspect
        from scripts import evaluate_ground_litter_profile_bank as evaluate

        source = inspect.getsource(evaluate.main)
        self.assertIn("copy.deepcopy(selector)", source)
        self.assertIn("cv2.resize(captured.frame, size", source)
        self.assertNotIn("baseline_tick = ticks[-1]", source)


if __name__ == "__main__":
    unittest.main()
