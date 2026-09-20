"""R1～R11 实施评审缺陷的回归测试。

每个测试对应评审文档中的一条发现，并尽量直接复现评审给出的反例。
命名规则：``test_rN_*``。这些测试检查的是**具体实现缺陷**，不代表现场准确率。
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from rtsp_annotator.ground_litter_profile_analysis import (
    BankPriorContext, _support_candidates, compute_mask_set, evaluate_bank_frame,
    roi_mask_from_geometry,
)
from rtsp_annotator.ground_litter_profile_background import (
    estimate_noise, observation_mask_from_frames, temporal_median_composite,
)
from rtsp_annotator.ground_litter_profile_bank import (
    BankError, default_matcher_config, load_bank, validate_calibration,
)
from rtsp_annotator.ground_litter_profile_match import (
    MatchEnvelope, analyze_residual_support, coerce_envelope, evaluate_match,
    score_profile,
)
from rtsp_annotator.ground_litter_profile_selector import (
    CandidateMatch, ProfileSelector,
)
from tests.profile_bank_fixtures import (
    build_synthetic_bank, synthetic_reference, synthetic_valid, write_test_video,
)


def _matcher() -> dict:
    return default_matcher_config()


def _envelope() -> MatchEnvelope:
    return MatchEnvelope(enter=5.0, hold=6.0, calibrated=True, samples=20,
                         source="unit_test")


class FrozenCalibrationTests(unittest.TestCase):
    """R1：盲测数据不得改变冻结的 Bank 参数。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "banks"
        build_synthetic_bank(self.root, "camera_01", "v1", profiles=2)

    def test_r1_bank_loader_requires_frozen_envelopes(self):
        bank = load_bank(self.root, "camera_01", "v1")
        self.assertEqual(validate_calibration(bank.matcher, list(bank.ids())), [])
        # 抹掉校准声明后必须被拒绝（历史未校准资产只能显式放行）。
        matcher = dict(bank.matcher)
        matcher.pop("profiles", None)
        matcher["calibration"] = {"source": "uncalibrated_defaults"}
        problems = validate_calibration(matcher, list(bank.ids()))
        self.assertIn("BANK_UNCALIBRATED", problems)

    def test_r1_evaluation_does_not_refit_envelope_on_blind_data(self):
        from scripts import evaluate_ground_litter_profile_bank as ev

        media = Path(self.tmp.name) / "blind"
        media.mkdir()
        write_test_video(media / "blind1.mp4", frames=12, width=160, height=120)
        bank = load_bank(self.root, "camera_01", "v1")
        before = json.dumps(bank.matcher, sort_keys=True)
        out = Path(self.tmp.name) / "eval"
        code = ev.main([
            "--bank-root", str(self.root), "--bank-id", "camera_01",
            "--version", "v1", "--input", str(media), "--output", str(out),
            "--analysis-fps", "1.0", "--small-target-trials", "2",
        ])
        self.assertEqual(code, 0)
        report = json.loads((out / "evaluation.json").read_text("utf-8"))
        self.assertEqual(report["envelope_source"], "frozen_in_bank")
        self.assertEqual(report["envelope_refit_warning"], "")
        after = json.dumps(load_bank(self.root, "camera_01", "v1").matcher,
                           sort_keys=True)
        self.assertEqual(before, after)
        # 报告的包络必须与 Bank 冻结值一致（不能出现“另一套阈值”）。
        for pid, envelope in report["envelopes"].items():
            frozen = bank.matcher["profiles"][pid]["envelope"]
            self.assertAlmostEqual(envelope["enter"], frozen["enter"], places=5)
            self.assertAlmostEqual(envelope["hold"], frozen["hold"], places=5)

    def test_r1_refit_flag_is_explicit_and_flagged(self):
        from scripts import evaluate_ground_litter_profile_bank as ev

        media = Path(self.tmp.name) / "blind2"
        media.mkdir()
        write_test_video(media / "b.mp4", frames=10, width=160, height=120)
        out = Path(self.tmp.name) / "eval2"
        code = ev.main([
            "--bank-root", str(self.root), "--bank-id", "camera_01",
            "--version", "v1", "--input", str(media), "--output", str(out),
            "--analysis-fps", "1.0", "--small-target-trials", "0",
            "--refit-envelope-on-input",
        ])
        self.assertEqual(code, 0)
        report = json.loads((out / "evaluation.json").read_text("utf-8"))
        self.assertEqual(report["envelope_source"], "refit_on_evaluation_input")
        self.assertTrue(report["envelope_refit_warning"])

    def test_r1_uncalibrated_bank_refused_without_override(self):
        # 直接构造一个未校准 matcher 并校验 loader 行为。
        matcher = default_matcher_config()
        problems = validate_calibration(matcher, ["p0001"])
        self.assertIn("BANK_UNCALIBRATED", problems)
        self.assertIn("MISSING_ENVELOPE:p0001", problems)


class SharedFrameReplayTests(unittest.TestCase):
    """R2：同一 tick 的所有候选面对同一真实帧与源时间。"""

    SIZE = 160

    def _candidate(self, profile_id: str, value: int = 80) -> dict:
        ramp = np.linspace(-20, 20, self.SIZE, dtype=np.float32)
        base = np.full((self.SIZE, self.SIZE, 3), float(value), np.float32)
        reference = np.clip(
            base + ramp[None, :, None], 0, 255,
        ).astype(np.uint8)
        valid = np.full((self.SIZE, self.SIZE), 255, np.uint8)
        return {
            "profile_id": profile_id, "group_id": f"g{profile_id}",
            "reference": reference, "valid": valid,
            "noise": {}, "descriptor": {},
            "envelope": {"enter": 100.0, "hold": 100.0, "calibrated": True,
                         "samples": 10, "source": "unit"},
            "context": BankPriorContext(profile_id, reference, valid, {}, {}),
            "support": {}, "low_support": False, "days": ["2026-09-01"],
            "samples": 3, "observation": {}, "valid_fraction_of_roi": 1.0,
        }

    def test_r2_all_candidates_see_same_frame_and_source_time(self):
        from scripts import build_ground_litter_profile_bank as build

        candidates = [self._candidate("p1"), self._candidate("p2")]
        frames = []
        for index, value in enumerate([60, 70, 80, 90, 100, 110]):
            frame = np.full((self.SIZE, self.SIZE, 3), value, np.uint8)
            frames.append({
                "file_id": "f1", "record_start": "2026-09-01 00:00:00",
                "offset_seconds": float(index), "source_time": 1000.0 + index,
                "frame": frame, "frame_sha256": f"sha{value}",
            })
        seen: list[tuple[float, int, str]] = []
        original = build.score_profile

        def traced(frame, *args, profile_id="", **kwargs):
            seen.append((float(frame[0, 0, 0]), profile_id))
            return original(frame, *args, profile_id=profile_id, **kwargs)

        config = SimpleNamespace(
            bank_id="b", version="v1", max_profiles=24,
        )
        report: dict = {}
        with patch.object(build, "score_profile", side_effect=traced):
            with contextlib.redirect_stdout(io.StringIO()):
                build.replay_all_candidates(
                    config, candidates, frames,
                    {"canvas_size": [self.SIZE, self.SIZE], "roi": []},
                    _matcher(), report,
                    replay_w=self.SIZE, replay_h=self.SIZE, leave_one_out=False,
                )
        # 每个 tick 的两个候选必须看到同一帧内容
        by_value: dict[int, set[int]] = {}
        for index, (value, _pid) in enumerate(seen):
            tick = index // len(candidates)
            by_value.setdefault(tick, set()).add(int(value))
            self.assertAlmostEqual(
                value, float([60, 70, 80, 90, 100, 110][tick]), places=4,
            )
        for tick, values in by_value.items():
            self.assertEqual(len(values), 1,
                             f"tick {tick} saw different frames {values}")
        self.assertEqual(len(by_value), 6)

    def test_r2_dynamic_pruning_uses_leave_one_out_not_static(self):
        from scripts import build_ground_litter_profile_bank as build

        candidates = [self._candidate("p1"), self._candidate("p2")]
        config = SimpleNamespace(max_profiles=24)
        replay = {
            "baseline": {"effective_fraction": 0.9, "pause_max": 4.0},
            "leave_one_out": {
                "p1": {"effective_fraction_delta": 0.0, "pause_max_delta": 0.0},
                "p2": {"effective_fraction_delta": 0.4, "pause_max_delta": 30.0},
            },
        }
        report: dict = {}
        kept = build.select_profiles_dynamically(config, candidates, replay, report)
        self.assertEqual([item["profile_id"] for item in kept], ["p2"])
        self.assertEqual(report["pruning"]["method"], "leave_one_out_dynamic_replay")
        removed = {row["profile_id"]: row["reason"]
                   for row in report["pruning"]["removed"]}
        self.assertEqual(removed.get("p1"), "DYNAMICALLY_REDUNDANT")

    def test_r2_n_not_capped_by_static_ranking(self):
        """静态覆盖为 0 但动态有贡献的候选必须保留。"""
        from scripts import build_ground_litter_profile_bank as build

        candidate = self._candidate("p1")
        candidate["static_coverage"] = 0.0
        config = SimpleNamespace(max_profiles=24)
        replay = {
            "baseline": {"effective_fraction": 0.5, "pause_max": 10.0},
            "leave_one_out": {
                "p1": {"effective_fraction_delta": 0.3, "pause_max_delta": 20.0},
            },
        }
        report: dict = {}
        kept = build.select_profiles_dynamically(config, [candidate], replay, report)
        self.assertEqual([item["profile_id"] for item in kept], ["p1"])


class CoverageIntegrationTests(unittest.TestCase):
    """R3：覆盖率按真实观测区间积分，不把未观测时间算成有效。"""

    def test_r3_no_forward_fill_over_long_gap(self):
        selector = ProfileSelector(
            bank_id="b", bank_version="v1", view_id="v",
            profile_ids=["p"],
            config={"tick_interval_seconds": 2.0, "result_validity_seconds": 4.0,
                    "join_gap_seconds": 300.0},
        )
        match = CandidateMatch("p", 0.1, True, True, verified=True)
        # 前四点正常，然后停 10 分钟（超过 join_gap）再回来。
        for timestamp in (0.0, 2.0, 4.0, 6.0, 606.0):
            decision = selector.observe(
                timestamp=timestamp,
                current=match if selector.selected_profile_id else None,
                candidates=[match],
            )
            if decision.commit_requested:
                selector.commit(profile_id="p", timestamp=timestamp)
        summary = selector.summarise()
        # 评审反例：旧实现把整段无观测时间算成有效；现在只算到计划观察点。
        self.assertLessEqual(summary["effective_seconds"], 8.0)
        self.assertGreaterEqual(summary["off_air_seconds"], 290.0)

    def test_r3_explicit_gap_is_not_coverage(self):
        selector = ProfileSelector(
            bank_id="b", bank_version="v1", view_id="v", profile_ids=["p"],
            config={"tick_interval_seconds": 2.0, "result_validity_seconds": 4.0},
        )
        match = CandidateMatch("p", 0.1, True, True, verified=True)
        # 中间 100s 完全没有观测：登记为不可判断，不得算成有效识别。
        selector.observe(timestamp=0.0, current=None, candidates=[match])
        selector.mark_non_observable(seconds=100.0, reason="DECODE_FAILED")
        selector.observe(timestamp=102.0, current=None, candidates=[match])
        summary = selector.summarise()
        self.assertAlmostEqual(summary["non_observable_seconds"], 100.0, places=3)
        self.assertLessEqual(summary["effective_seconds"], 4.0)
        # 100s 无观测只能贡献到计划观察点，不能贡献 100s 可判断时间。
        self.assertLessEqual(summary["observed_seconds"], 4.0)


class SearchBudgetTests(unittest.TestCase):
    """R4：粗排以外的候选必须能被公平检查到。"""

    def _selector(self, ids):
        return ProfileSelector(
            bank_id="b", bank_version="v1", view_id="v", profile_ids=ids,
            config={"top_k": 3, "max_small_matches_per_tick": 4,
                    "expanded_new_candidates_per_tick": 1,
                    "recovery_min_samples": 2, "recovery_min_span_seconds": 2.0},
        )

    def test_r4_run_tick_reaches_kth_plus_one(self):
        from scripts import evaluate_ground_litter_profile_bank as ev

        ids = [f"p{i}" for i in range(1, 7)]
        selector = self._selector(ids)
        contexts = {pid: SimpleNamespace(profile_id=pid) for pid in ids}
        ranks = {pid: {"rank": float(index)} for index, pid in enumerate(ids)}
        envelopes = {pid: _envelope() for pid in ids}

        def fake_eval(context, *args, **kwargs):
            good = context.profile_id == "p5"
            return SimpleNamespace(
                outcome={"score": 0.1, "enter_eligible": good,
                         "hold_eligible": good, "reason": "MATCHED"},
                candidates=(), availability_fraction=1.0,
            )

        checked: list[str] = []
        with patch.object(ev, "extract_grid_descriptor", return_value={}), \
             patch.object(ev, "descriptor_coarse_distance",
                          side_effect=lambda a, b, scale: b["rank"]), \
             patch.object(ev, "evaluate_bank_frame", side_effect=fake_eval):
            for tick in range(20):
                row = ev._run_tick(
                    selector, contexts, ranks, {}, envelopes, _matcher(),
                    np.zeros((64, 64, 3), np.uint8), np.full((64, 64), 255, np.uint8),
                    (64, 64), {"roi": []},
                    source_time=float(tick) * 2.0, tick_index=tick,
                    tick_interval=2.0,
                )
                checked.extend(row["tested"])
        self.assertIn("p5", checked)
        self.assertEqual(selector.selected_profile_id, "p5")

    def test_r4_budget_is_still_bounded(self):
        from scripts import evaluate_ground_litter_profile_bank as ev

        ids = [f"p{i}" for i in range(1, 9)]
        selector = self._selector(ids)
        contexts = {pid: SimpleNamespace(profile_id=pid) for pid in ids}
        envelopes = {pid: _envelope() for pid in ids}

        def fake_eval(context, *args, **kwargs):
            return SimpleNamespace(
                outcome={"score": 1.0, "enter_eligible": False,
                         "hold_eligible": False, "reason": "SCORE_ABOVE_HOLD"},
                candidates=(), availability_fraction=1.0,
            )

        row = None
        with patch.object(ev, "extract_grid_descriptor", return_value={}), \
             patch.object(ev, "descriptor_coarse_distance",
                          side_effect=lambda a, b, scale: b["rank"]), \
             patch.object(ev, "evaluate_bank_frame", side_effect=fake_eval):
            row = ev._run_tick(
                selector, contexts, {pid: {"rank": i} for i, pid in enumerate(ids)},
                {}, envelopes, _matcher(), np.zeros((64, 64, 3), np.uint8),
                np.full((64, 64), 255, np.uint8), (64, 64), {"roi": []},
                source_time=0.0, tick_index=0, tick_interval=2.0,
            )
        self.assertLessEqual(row["tested_budget"], row["budget"])


class NoiseSupportTests(unittest.TestCase):
    """R8：只有实际有效的观测才进入支持计数与分位统计。"""

    def test_r8_support_counts_only_valid_observations(self):
        base = np.full((32, 32, 3), 80, np.uint8)
        left = np.zeros((32, 32), np.uint8)
        left[:, :16] = 255
        right = np.zeros((32, 32), np.uint8)
        right[:, 16:] = 255
        noise = estimate_noise(
            base, [base, base], [left, right], ["b1", "b2"], stride=1,
        )
        counts = noise.payload["support_blocks"]
        self.assertEqual(float(counts[8, 8]), 1.0)
        self.assertEqual(float(counts[8, 24]), 1.0)

    def test_r8_repeated_block_does_not_inflate_support(self):
        base = np.full((32, 32, 3), 80, np.uint8)
        mask = np.full((32, 32), 255, np.uint8)
        noise = estimate_noise(
            base, [base, base, base], [mask, mask, mask], ["b1", "b1", "b1"],
            stride=1,
        )
        counts = noise.payload["support_blocks"]
        self.assertEqual(float(counts[8, 8]), 1.0)


class AdmissionTests(unittest.TestCase):
    """R9：零可用性/坏帧不得通过准入，也不得累计清走证据。"""

    def _context(self):
        frame = np.random.default_rng(1).integers(50, 190, (192, 192, 3),
                                                  dtype=np.uint8)
        valid = np.full((192, 192), 255, np.uint8)
        return BankPriorContext("p", frame.copy(), valid, {}, {})

    def test_r9_bad_frame_blocks_admission(self):
        context = self._context()
        evaluation = evaluate_bank_frame(
            context, context.reference,
            envelope=MatchEnvelope(100, 100, True, 20, "review"),
            config=_matcher(), roi_mask=np.full((192, 192), 255, np.uint8),
            corruption="known_bad_frame",
        )
        self.assertFalse(evaluation.outcome["enter_eligible"])
        self.assertFalse(evaluation.outcome["hold_eligible"])
        self.assertEqual(evaluation.availability_fraction, 0.0)
        self.assertTrue(evaluation.outcome["reason"].startswith("BLOCKED_"))

    def test_r9_geometry_invalid_blocks_admission(self):
        context = self._context()
        evaluation = evaluate_bank_frame(
            context, context.reference,
            envelope=MatchEnvelope(100, 100, True, 20, "review"),
            config=_matcher(), roi_mask=np.full((192, 192), 255, np.uint8),
            geometry_valid=False,
        )
        self.assertFalse(evaluation.outcome["enter_eligible"])
        self.assertIn("GEOMETRY_INVALID", evaluation.outcome["reason"])

    def test_r9_zero_availability_from_roi_blocks_admission(self):
        context = self._context()
        evaluation = evaluate_bank_frame(
            context, context.reference,
            envelope=MatchEnvelope(100, 100, True, 20, "review"),
            config=_matcher(), roi_mask=np.zeros((192, 192), np.uint8),
        )
        self.assertFalse(evaluation.outcome["enter_eligible"])
        self.assertLess(evaluation.availability_fraction, 0.15)


class SharedCompensationTests(unittest.TestCase):
    """R10：评分与前景必须用同一套补偿后的残差，并保留候选几何过滤。"""

    def test_r10_foreground_uses_same_compensation_as_score(self):
        """前景提取必须使用评分那一步的补偿参考（R10）。

        做法：分别用「原始参考」和「补偿参考」直接调用共享的残差分析，
        证明两者输出不同；再验证 adapter 的结果等于**补偿参考**那一个。
        这样即使某些场景下两种参考恰好给出相同支持，测试也不会失去区分力
        （它比较的是调用链实际使用的输入）。
        """
        rng = np.random.default_rng(7)
        size = 192
        reference = rng.integers(60, 160, (size, size, 3), dtype=np.uint8)
        valid = np.full((size, size), 255, np.uint8)
        gains = np.array([1.20, 0.85, 1.10], np.float32)
        base = np.clip(
            reference.astype(np.float32) * gains[None, None, :], 0, 255,
        ).astype(np.uint8)
        frame = base.copy()
        frame[80:96, 80:96] = 250
        context = BankPriorContext("p", reference, valid, {}, {})
        score = score_profile(frame, reference, valid, valid,
                              profile_id="p", config=_matcher())
        self.assertIsNotNone(score.compensated_reference)
        thresholds = {
            "seed_signature_threshold": np.full((size, size), 3.0, np.float32),
            "seed_luminance_threshold": np.full((size, size), 100.0, np.float32),
            "support_signature_threshold": np.full((size, size), 1.2, np.float32),
            "support_luminance_threshold": np.full((size, size), 25.0, np.float32),
        }
        with_compensated = analyze_residual_support(
            frame, score.compensated_reference, valid, valid, thresholds,
        )
        with_raw = analyze_residual_support(
            frame, reference, valid, valid, thresholds,
        )
        # 二者必须可区分（否则这个测试没有意义，应修正夹具）。
        self.assertFalse(
            np.array_equal(with_compensated.support, with_raw.support),
            "夹具无法区分补偿与未补偿：请加强光照变化",
        )
        from rtsp_annotator.ground_litter_profile_analysis import (
            compute_mask_set,
        )
        masks = compute_mask_set(
            context, frame, roi_mask=valid,
            compensated_reference=score.compensated_reference,
            fit_mask=score.fit_mask,
        )
        self.assertTrue(
            np.array_equal(masks.foreground_support, with_compensated.support),
            "adapter 没有使用评分传入的补偿参考",
        )

    def test_r10_single_pixel_noise_and_large_blob_are_not_candidates(self):
        support = np.zeros((160, 160), np.uint8)
        support[5, 5] = 1
        support[20:140, 20:140] = 1
        rows, rejected = _support_candidates(
            support, np.full_like(support, 255),
        )
        self.assertEqual(rows, [])
        self.assertGreaterEqual(rejected["too_small"], 1)
        self.assertGreaterEqual(rejected["too_large"], 1)

    def test_r10_small_target_survives_geometry_filter(self):
        support = np.zeros((160, 160), np.uint8)
        support[40:48, 40:48] = 1  # 8x8 小目标
        rows, rejected = _support_candidates(
            support, np.full_like(support, 255),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["box"], [40, 40, 48, 48])
        self.assertEqual(rejected["too_small"], 0)


class FrozenCanvasTests(unittest.TestCase):
    """R11：所有文件必须配准到同一冻结画布。"""

    def test_r11_sampler_requires_frozen_canvas(self):
        from rtsp_annotator.ground_litter_profile_sampling import (
            BoundedPreviewSampler,
        )
        from rtsp_annotator.ground_litter_recording_cache import (
            ManagedRecordingCache,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache = ManagedRecordingCache(tmp)
            self.addCleanup(cache.close)
            with self.assertRaises(BankError):
                BoundedPreviewSampler(
                    cache, analysis_size=(64, 48),
                    roi_mask=np.full((48, 64), 255, np.uint8),
                )

    def test_r11_cross_file_shift_registers_to_same_canvas(self):
        from rtsp_annotator.ground_litter_profile_sampling import CanvasRegistrar

        rng = np.random.default_rng(3)
        # 同一高分辨率源的两个重叠裁剪：位移是真实的整数像素，无插值伪影。
        big = rng.integers(60, 180, (240, 240, 3), dtype=np.uint8)
        big = cv2.GaussianBlur(big, (0, 0), 1.0)
        big[::12, :] = 40
        big[:, ::12] = 40
        canvas = np.ascontiguousarray(big[8:168, 8:168])
        shifted = np.ascontiguousarray(big[12:172, 14:174])
        registrar = CanvasRegistrar(canvas)
        aligned, diagnostics = registrar.register(shifted)
        raw = float(np.abs(
            shifted.astype(np.float32) - canvas.astype(np.float32)
        ).mean())
        if diagnostics["registration"] == "homography":
            difference = float(np.abs(
                aligned.astype(np.float32) - canvas.astype(np.float32)
            ).mean())
            self.assertLessEqual(difference, raw + 1e-6)
            self.assertTrue(diagnostics["geometry_ok"])
        else:
            # 特征不足时允许恒等回退，但必须如实标注。
            self.assertEqual(diagnostics["registration"], "identity_fallback")

    def test_r11_observation_mask_excludes_movers(self):
        base = synthetic_reference(160, 120, value=120)
        frames = [base.copy() for _ in range(5)]
        for index in (0, 2):
            frames[index] = base.copy()
            frames[index][40:70, 60:90] = 250  # 行人/车辆经过
        roi = synthetic_valid(160, 120)
        availability, diagnostics, masks = observation_mask_from_frames(
            frames, roi, block_ids=[f"b{i}" for i in range(5)],
        )
        self.assertEqual(len(masks), 5)
        # 该区域在多数帧里被瞬态物体占据：所有帧都不应用它做合成。
        for mask in masks:
            self.assertEqual(
                int(np.count_nonzero(mask[45:65, 65:85])), 0,
            )
        # 未被移动物体影响的区域仍然可用。
        self.assertGreater(int(np.count_nonzero(masks[1][:30, :30])), 0)
        self.assertLess(diagnostics["available_fraction_of_roi"], 1.0)

    def test_r11_moving_objects_do_not_enter_background(self):
        base = synthetic_reference(160, 120, value=120)
        frames = [base.copy() for _ in range(6)]
        for index in (0, 1):
            frames[index][40:70, 60:90] = 250
        roi = synthetic_valid(160, 120)
        availability, _diag, masks = observation_mask_from_frames(
            frames, roi, block_ids=[f"b{i}" for i in range(6)],
        )
        composite = temporal_median_composite(
            frames, masks, availability, [f"b{i}" for i in range(6)],
            stride=1, min_observations=2,
        )
        patch = composite.reference[50:60, 70:80].astype(np.float32)
        reference_patch = base[50:60, 70:80].astype(np.float32)
        self.assertLess(float(np.abs(patch - reference_patch).mean()), 30.0)


class BoundedPipelineTests(unittest.TestCase):
    """R5：总素材超过缓存配额时仍要有界处理全部计划文件。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.media = self.root / "ps"
        self.media.mkdir()
        self.work = self.root / "work"

    def _write_media(self, count: int) -> list[Path]:
        """写有纹理的合成剪辑：过于平坦的画面会被质量过滤判为 BLURRY。"""
        paths = []
        rng = np.random.default_rng(11)
        for index in range(count):
            path = self.media / f"clip_{index}.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120),
            )
            if not writer.isOpened():  # pragma: no cover - 编解码器缺失
                self.skipTest("无法创建测试视频")
            try:
                for frame_index in range(40):
                    frame = rng.integers(70, 170, (120, 160, 3), dtype=np.uint8)
                    frame = cv2.GaussianBlur(frame, (0, 0), 0.8)
                    frame[::8, :] = 30
                    frame[:, ::8] = 30
                    writer.write(frame)
            finally:
                writer.release()
            paths.append(path)
        return paths

    def test_r5_all_planned_files_consumed_under_tight_budget(self):
        """冷启动 + 极小预算：全部计划文件都必须最终被消费，且峰值不超预算。"""
        from rtsp_annotator.ground_litter_recording_cache import (
            ManagedRecordingCache,
        )
        from rtsp_annotator.ground_litter_recording_source import (
            RecordingDownloader, RecordingFile,
        )
        from scripts import build_ground_litter_profile_bank as build

        self._write_media(4)
        sources = sorted(self.media.glob("*.mp4"))
        largest = max(path.stat().st_size for path in sources)
        # 预算只够「当前处理中的一个文件」+ 少量余量。
        budget = int(largest * 1.4)
        cache = ManagedRecordingCache(
            self.work, raw_cache_budget=budget, work_budget=8 * 1024 ** 3,
        )
        self.addCleanup(cache.close)

        class LocalDownloader(RecordingDownloader):
            """把本地文件当远端下载：只测流水线的拉取/释放，不测网络。"""

            def fetch_url_for_file(self, client, query, file_id, **kwargs):
                return SimpleNamespace(url=f"file://{file_id}", file=None)

            def download(self, url, target, *, expected_size=None, **kwargs):
                source = self.sources[url.removeprefix("file://")]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                size = target.stat().st_size
                return SimpleNamespace(
                    size=size, sha256="a" * 64, range_supported=False,
                    elapsed_seconds=0.01, resumed=False,
                )

        downloader = LocalDownloader()
        downloader.sources = {f"f{index}": path for index, path in enumerate(sources)}

        files = []
        for index, path in enumerate(sources):
            files.append(RecordingFile(
                file_id=f"f{index}", file_name=str(path),
                record_start="2026-09-01 10:00:00",
                record_end="2026-09-01 10:05:00",
                file_size=path.stat().st_size,
            ))
        cache.register("00000000000000000000", files)
        config = SimpleNamespace(
            device_code="00000000000000000000", seed=1, preview_width=160,
            use_seek=False,
            work_dir=self.work, bank_id="b", version="v1", max_profiles=24,
        )
        report: dict = {}
        geometry = {
            "canvas_size": [160, 120], "roi": [],
            "exclude_zones": [], "overlay_exclude_zones": [], "view_id": "v",
        }
        samples, _details, _hd, _desc = build.stream_materialize_and_sample(
            config, None, downloader, cache, files, geometry, report,
            prefetch_slots=2, wait_timeout=60.0,
        )
        states = {row["file_id"]: row["state"]
                  for row in report["pipeline"]["final_states"]}
        self.assertEqual(len(states), len(files), states)
        self.assertTrue(all(state == "consumed" for state in states.values()),
                        states)
        self.assertFalse(report["pipeline"]["truncated_plan"])
        self.assertTrue(samples)
        # 退出 lease 后必须真的释放（旧实现在 lease 内调用会被拒绝）。
        self.assertGreater(report["pipeline"]["released_files"], 0)
        self.assertLessEqual(cache.budget_report().raw_bytes, budget)

    def test_r5_wait_for_capacity_reports_timeout(self):
        from rtsp_annotator.ground_litter_recording_cache import (
            CacheError, ManagedRecordingCache,
        )

        cache = ManagedRecordingCache(
            self.root / "tight", raw_cache_budget=1024, work_budget=1024 ** 3,
        )
        self.addCleanup(cache.close)
        ok, reason = cache.wait_for_capacity(
            10 * 1024 * 1024, timeout=0.1, poll_seconds=0.05,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "raw_cache_budget")
        del CacheError


class SmallTargetTests(unittest.TestCase):
    """R6：小目标必须在 ROI 内、按原图尺度注入，并逐目标空间匹配。"""

    def test_r6_injection_lands_inside_roi_at_native_scale(self):
        from scripts import evaluate_ground_litter_profile_bank as ev

        geometry = json.loads((
            Path(__file__).resolve().parents[1]
            / "config/ground_litter_01030_geometry.json"
        ).read_text(encoding="utf-8"))
        native_w, native_h = geometry["canvas_size"]
        roi_native = roi_mask_from_geometry(geometry, native_w, native_h)
        canvas = (960, 540)
        side = max(2, int(round(8 * canvas[0] / native_w)))
        rng = np.random.default_rng(20260920)
        frame = np.zeros((canvas[1], canvas[0], 3), np.uint8)
        canvas_roi = roi_mask_from_geometry(geometry, canvas[0], canvas[1])
        inside = 0
        for _ in range(12):
            _target, truth = ev._inject_small_target_in_roi(
                frame, roi_native, canvas, side, rng,
            )
            self.assertIsNotNone(truth)
            x0, y0, x1, y1 = truth["box"]
            self.assertGreater(
                int(np.count_nonzero(canvas_roi[y0:y1, x0:x1])), 0,
                f"注入框 {truth['box']} 不在 ROI 内",
            )
            inside += 1
        self.assertEqual(inside, 12)

    def test_r6_target_match_requires_spatial_overlap(self):
        from scripts import evaluate_ground_litter_profile_bank as ev

        truth = {"box": [40, 40, 48, 48], "native_side": 8}
        far_tick = {
            "candidate_boxes": 3,
            "candidate_boxes_detail": [{"box": [100, 100, 108, 108]}],
            "prior_allowed": True, "availability_max": 1.0,
        }
        near_tick = {
            "candidate_boxes": 1,
            "candidate_boxes_detail": [{"box": [41, 41, 49, 49]}],
            "prior_allowed": True, "availability_max": 1.0,
        }
        far = ev._match_target(truth, far_tick, baseline_tick=None, tick_index=0,
                               file_id="f", source_time=0.0)
        near = ev._match_target(truth, near_tick, baseline_tick=None, tick_index=1,
                                file_id="f", source_time=2.0)
        self.assertFalse(far["target_detected"])
        self.assertTrue(near["target_detected"])
        self.assertGreater(near["iou_hits"], 0)

    def test_r6_paired_baseline_is_reported(self):
        from scripts import evaluate_ground_litter_profile_bank as ev

        truth = {"box": [40, 40, 48, 48], "native_side": 8}
        baseline = {"candidate_boxes": 0, "candidate_boxes_detail": []}
        tick = {
            "candidate_boxes": 1,
            "candidate_boxes_detail": [{"box": [40, 40, 48, 48]}],
            "prior_allowed": True, "availability_max": 1.0,
        }
        row = ev._match_target(truth, tick, baseline_tick=baseline, tick_index=0,
                               file_id="f", source_time=0.0)
        self.assertEqual(row["baseline_candidate_boxes"], 0)
        summary = ev._small_target_summary([row])
        self.assertEqual(summary["trials"], 1)
        self.assertEqual(summary["detected"], 1)
        self.assertEqual(summary["paired_baselines"], 1)


class EnvelopeCoercionTests(unittest.TestCase):
    """冻结包络从 JSON 读回来是 dict，必须与 MatchEnvelope 走同一判定。"""

    def test_coerce_envelope_from_json(self):
        envelope = coerce_envelope(
            {"enter": 8.5, "hold": 8.7, "calibrated": True, "samples": 28,
             "source": "calibration_quantile"}
        )
        self.assertAlmostEqual(envelope.enter, 8.5)
        self.assertTrue(envelope.calibrated)
        score = SimpleNamespace(score=1.0, profile_id="p", diagnostics={},
                                anchor_fraction=1.0, distinct_regions=9)
        outcome = evaluate_match(score, envelope, _matcher())
        self.assertTrue(outcome.enter_eligible)

    def test_coerce_envelope_rejects_garbage(self):
        with self.assertRaises(BankError):
            coerce_envelope(object())


class CalibrationSeparationTests(unittest.TestCase):
    """R1/R7：校准块必须与构建块隔离，且噪声来源可核对。"""

    def test_noise_records_independent_calibration(self):
        base = synthetic_reference(160, 120, value=120)
        mask = synthetic_valid(160, 120)
        frames = [base.copy() for _ in range(4)]
        noise = estimate_noise(
            base, frames, [mask] * 4, [f"c{i}" for i in range(4)], stride=1,
            config={"_calibration_block_count": 4, "_independent_calibration": True},
        )
        self.assertEqual(noise.diagnostics["calibration_blocks"], 4)
        self.assertTrue(noise.diagnostics["independent_of_reference"])

    def test_composite_uses_observation_masks(self):
        base = synthetic_reference(160, 120, value=120)
        valid = synthetic_valid(160, 120)
        frames = [base.copy() for _ in range(4)]
        for index in (0, 1):
            frames[index][50:70, 60:80] = 250
        availability, _diag, masks = observation_mask_from_frames(
            frames, valid, block_ids=[f"b{i}" for i in range(4)],
        )
        result = temporal_median_composite(
            frames, masks, availability, [f"b{i}" for i in range(4)],
        )
        # 瞬态物体不应进入合成背景。
        self.assertLess(int(result.reference[60, 70, 0]), 200)


if __name__ == "__main__":
    unittest.main()
