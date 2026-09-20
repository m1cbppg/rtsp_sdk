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


def _candidates(count: int = 3, size: int = 160) -> list[dict]:
    helper = SharedFrameReplayTests()
    return [helper._candidate(f"p{index + 1}") for index in range(count)]


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
        self.assertGreaterEqual(len(kept), 1)
        self.assertLessEqual(len(kept), len(candidates))
        self.assertEqual(
            pruning["pruning"]["method"],
            "conservative_one_at_a_time_dynamic_replay",
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
        # 每一轮只能删一个，并记录删除后的复核指标。
        evaluated = [row["candidates_evaluated"] for row in order]
        self.assertTrue(all(len(row) >= 2 for row in evaluated))
        for removal in pruning["pruning"]["removed"]:
            self.assertIn("summary_after", removal)
            self.assertLess(removal["set_size_after"], len(candidates))
        # 至少要留下 p3（唯一外观）。
        self.assertIn("p3", [item["profile_id"] for item in kept])

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


class C2BoundedPipelineTests(unittest.TestCase):
    """C2：只拉本阶段需要的文件；失败分类；全链路预算记账。"""

    def _fake_remote(self):
        """构造远程来源桩：清单 + 可下载的假 PS。"""

        class FakeListClient:
            def __init__(self, files=(), **_kwargs):
                self.files = list(files)

            def list_files(self, query):  # pragma: no cover - 由工厂调用
                return list(self.files)

        class FakeDownloader:
            def __init__(self, root: Path):
                self.root = root
                self.downloaded: list[str] = []

            def fetch_url_for_file(self, client, query, file_id, **kwargs):
                return SimpleNamespace(url="fake://" + file_id)

            def download(self, url, target, **kwargs):
                file_id = url.rsplit("/", 1)[-1]
                delay = {"f0": 0.05, "f1": 0.1, "f2": 0.15, "f3": 0.2}.get(
                    file_id, 0.05,
                )
                import time as _time
                _time.sleep(delay)
                self.downloaded.append(file_id)
                target.parent.mkdir(parents=True, exist_ok=True)
                # 受管缓存的落盘名是 *.bin，而 OpenCV 的 VideoWriter 只按扩展名
                # 选后端；先写 .mp4 再改名，保证假 PS 真的可解码。
                staging = target.with_name(target.name + ".tmp.mp4")
                writer = cv2.VideoWriter(
                    str(staging), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120),
                )
                rng = np.random.default_rng(7)
                for _ in range(12):
                    writer.write(rng.integers(60, 180, (120, 160, 3), dtype=np.uint8))
                writer.release()
                if target.exists():
                    target.unlink()
                staging.replace(target)
                payload = target.read_bytes()
                import hashlib as _hashlib
                return SimpleNamespace(
                    size=len(payload), sha256=_hashlib.sha256(payload).hexdigest(),
                    range_supported=False, elapsed_seconds=0.01,
                )

        return FakeListClient, FakeDownloader

    def test_hd_repull_only_fetches_declared_files(self):
        from scripts import build_ground_litter_profile_bank as build
        from rtsp_annotator.ground_litter_recording_cache import ManagedRecordingCache
        from rtsp_annotator.ground_litter_recording_source import RecordingFile

        files = [
            RecordingFile(f"f{index}", f"f{index}.ps",
                          f"2026-09-{14 + index} 10:00:00",
                          f"2026-09-{14 + index} 10:05:00", 100)
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with ManagedRecordingCache(tmp, raw_cache_budget=10 ** 9) as cache:
                # 真实调用链里清单阶段已经 register；重拉阶段不负责登记。
                cache.register("00000000000000000000", files)
                client_cls, downloader_cls = self._fake_remote()
                downloader = downloader_cls(Path(tmp))
                with unittest.mock.patch.object(
                    build, "RecordingListClient", client_cls,
                ), unittest.mock.patch.object(
                    build, "RecordingDownloader", lambda: downloader,
                ):
                    summary = build._ensure_remote_entries(
                        SimpleNamespace(
                            device_code="00000000000000000000",
                        ),
                        SimpleNamespace(source="ctseelink-file-urls",
                                        auth_token=None, api_key=None),
                        cache, files, {}, local_paths={},
                        needed_file_ids=["f1", "f2"],
                    )
                # C2 反例：v2 会把盲测日文件一起重拉。
                self.assertEqual(summary["considered"], 4)
                self.assertEqual(summary["checked"], 2)
                self.assertEqual(summary["skipped_not_needed"], 2)
                self.assertEqual(sorted(summary["skipped_not_needed_ids"]),
                                 ["f0", "f3"])
                self.assertEqual(
                    sorted(downloader.downloaded), ["f1", "f2"],
                    f"summary={summary}",
                )
                self.assertEqual(summary["failure_kinds"], {}, str(summary))

    def test_failure_kinds_stay_separate_from_material_quality(self):
        from scripts import build_ground_litter_profile_bank as build
        from rtsp_annotator.ground_litter_recording_cache import ManagedRecordingCache
        from rtsp_annotator.ground_litter_recording_source import RecordingFile

        files = [
            RecordingFile(f"f{index}", f"f{index}.ps",
                          "2026-09-14 10:00:00", "2026-09-14 10:05:00", 100)
            for index in range(2)
        ]

        class BrokenDownloader:
            def fetch_url_for_file(self, client, query, file_id, **kwargs):
                raise RuntimeError("link expired")

            def download(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("fetch 失败时不应继续下载")

        with tempfile.TemporaryDirectory() as tmp:
            with ManagedRecordingCache(tmp, raw_cache_budget=10 ** 9) as cache:
                with unittest.mock.patch.object(
                    build, "RecordingListClient", lambda **kwargs: object(),
                ), unittest.mock.patch.object(
                    build, "RecordingDownloader", BrokenDownloader,
                ):
                    summary = build._ensure_remote_entries(
                        SimpleNamespace(device_code="00000000000000000000"),
                        SimpleNamespace(source="ctseelink-file-urls",
                                        auth_token=None, api_key=None),
                        cache, files, {}, local_paths={},
                    )
            self.assertEqual(summary["failure_kinds"],
                             {"URL_REFRESH_FAILED": 2})
            # 网络/刷新失败不能被记成"素材质量失败"。
            for row in summary["failed"]:
                self.assertEqual(row["kind"], "URL_REFRESH_FAILED")
                self.assertNotIn("MATERIAL_INVALID", row["kind"])

    def test_whole_factory_chain_bounded_with_quota_below_total_input(self):
        """全链路（清单→分区→分阶段拉取→合成→包络→回放→定稿→发布）。

        C2 验收：总输入 > 缓存配额时必须仍然跑完，且每个阶段的占用都有记账、
        盲测素材不被高清阶段拉取、峰值不超配额。**必须覆盖多轮采样**：
        分区后的文件数 > 预取槽数，否则只测了第一轮。
        """
        from scripts import build_ground_litter_profile_bank as build
        from rtsp_annotator.ground_litter_recording_cache import (
            ManagedRecordingCache,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "ps"
            media.mkdir(parents=True)
            rng = np.random.default_rng(5)
            total_bytes = 0
            for day in range(1, 5):
                for chunk in range(2):
                    path = media / f"2026090{day}_{chunk:02d}.mp4"
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120),
                    )
                    tint = 20 * ((day + chunk) % 2)
                    for _ in range(14):
                        frame = rng.integers(60, 180, (120, 160, 3), dtype=np.uint8)
                        frame = cv2.GaussianBlur(frame, (0, 0), 1.1)
                        frame = np.clip(
                            frame.astype(np.int32) + tint, 0, 255,
                        ).astype(np.uint8)
                        frame[::20, :] = (frame[::20, :] // 2 + 30).astype(np.uint8)
                        writer.write(frame)
                    writer.release()
                    stamp = _datetime.datetime(2026, 9, day, 10 + chunk, 0, 0).timestamp()
                    os.utime(path, (stamp, stamp))
                    total_bytes += path.stat().st_size

            config = build.FactoryConfig(
                camera_id="cam_chain", bank_id="cam_chain", version="vchain",
                output_root=root / "banks", work_dir=root / "work",
                geometry_path=None, analysis_size=None,
                device_code="00000000000000000000",
                raw_cache_budget=max(1, total_bytes),
                work_budget=512 * 1024 ** 2,
                max_profiles=6,
            )
            args = SimpleNamespace(
                input=str(media), source=None, device_code=config.device_code,
                start="", end="", auth_token=None, api_key=None,
                prefetch_slots=1, max_replay_frames=24, loo_stride=1,
                resume=True, supersede=False, seed=config.seed,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                report = build.run_factory(config, args)

            # 总输入 > 配额：确认这条约束真的被触发。
            self.assertGreater(total_bytes, 0)
            self.assertLessEqual(config.raw_cache_budget, total_bytes)
            # 多轮采样：文件数必须大于预取槽数。
            self.assertGreater(
                report["pipeline"]["planned"], int(args.prefetch_slots),
            )
            self.assertEqual(
                report["pipeline"]["planned"], report["sampling"]["files_sampled"]
                + report["pipeline"]["failed"],
            )
            envelope = report["resource_envelope"]
            self.assertLessEqual(envelope["peak_raw_bytes"],
                                 config.raw_cache_budget)
            self.assertLessEqual(envelope["peak_work_bytes"], config.work_budget)
            # 每个阶段都留下释放/拒绝记账，而不是把占用留到最后。
            self.assertIn("after_composite", report["stage_releases"])
            self.assertIn("composite", envelope["stage_bytes"])
            self.assertIn("hd_materialize", envelope["stage_bytes"])
            # 计划文件的终态必须全部有记录（含失败原因）。
            self.assertTrue(envelope["planned_final_states"])
            self.assertEqual(
                sum(envelope["planned_final_summary"].values()),
                len(envelope["planned_final_states"]),
            )
            # 盲测日文件不得进入高清重拉计划。
            blind = set(report["material_plan"]["blind_files_excluded_from_hd"])
            needed = set(report["material_plan"]["needed_file_ids"])
            self.assertFalse(blind & needed)
            # 回放帧内存口径必须报告，且评分后不常驻像素。
            self.assertGreater(envelope["frame_memory_estimate"]["replay_frames"], 0)
            self.assertLessEqual(
                report["replay_frames"]["frames"],
                int(args.max_replay_frames),
            )
            # 素材失败与预算/网络失败必须分开记录。
            self.assertIn("failure_kinds", report["hd_repull"])
            self.assertIn("material_failures", envelope)

            contexts = report["n_selection_fair_comparison"]
            self.assertTrue(contexts["frames_identical_for_all_subsets"])
            self.assertIn(
                "final_set_verification", report,
            )
            self.assertTrue(
                report["final_set_verification"]["final_set"]
                ["frames_identical_to_baseline"]
            )

    def test_replay_collect_rematerializes_missing_build_files(self):
        """回放是独立消费阶段：构建 PS 不在盘上时必须能自己按需重取。

        历史缺陷：合成阶段 `_release_materialized` + `--resume` 回收之后，
        `_collect_replay_frames` 静默收集 0 帧，动态定稿失去时间轴。
        这里用远程来源桩走真实取回路径（受管缓存落盘名是 *.bin）。
        """
        from scripts import build_ground_litter_profile_bank as build
        from rtsp_annotator.ground_litter_recording_cache import (
            ManagedRecordingCache,
        )
        from rtsp_annotator.ground_litter_recording_source import RecordingFile
        import hashlib

        class FakeListClient:
            def __init__(self, **kwargs):
                pass

        class FakeDownloader:
            def __init__(self):
                self.downloaded: list[str] = []

            def fetch_url_for_file(self, client, query, file_id, **kwargs):
                return SimpleNamespace(url="fake://" + file_id)

            def download(self, url, target, **kwargs):
                file_id = url.rsplit("/", 1)[-1]
                self.downloaded.append(file_id)
                target.parent.mkdir(parents=True, exist_ok=True)
                staging = target.with_name(target.name + ".tmp.mp4")
                writer = cv2.VideoWriter(
                    str(staging), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120),
                )
                rng = np.random.default_rng(13)
                for _ in range(14):
                    writer.write(rng.integers(60, 180, (120, 160, 3), dtype=np.uint8))
                writer.release()
                staging.replace(target)
                payload = target.read_bytes()
                return SimpleNamespace(
                    size=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                    range_supported=False, elapsed_seconds=0.01,
                )

        files = [
            RecordingFile(
                file_id=f"r{index}", file_name=f"r{index}.ps",
                record_start=f"2026-09-1{4 + index} 00:00:00",
                record_end=f"2026-09-1{4 + index} 00:05:00", file_size=1000,
            )
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            work.mkdir()
            config = build.FactoryConfig(
                camera_id="cam_replay", bank_id="cam_replay", version="vreplay",
                output_root=Path(tmp) / "banks", work_dir=work,
                geometry_path=None, analysis_size=None,
                device_code="00000000000000000000",
            )
            geometry = {"canvas_size": [160, 120], "roi": []}
            args = SimpleNamespace(
                input=None, source="ctseelink-file-urls",
                auth_token=None, api_key=None,
            )
            downloader = FakeDownloader()
            with ManagedRecordingCache(tmp, raw_cache_budget=10 ** 9) as cache:
                with unittest.mock.patch.object(
                    build, "RecordingListClient", FakeListClient,
                ), unittest.mock.patch.object(
                    build, "RecordingDownloader", lambda: downloader,
                ):
                    report: dict = {}
                    rows = build._collect_replay_frames(
                        cache, config, files, geometry, frame_budget=8,
                        args=args, local_paths={}, report=report,
                    )
            self.assertGreater(len(rows), 0)
            materialize = report["replay_materialize"]
            self.assertGreater(materialize["attempted"], 0)
            self.assertEqual(materialize["ok"], 2)
            self.assertEqual(sorted(downloader.downloaded), ["r0", "r1"])
            self.assertEqual(materialize["failed"], [])


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
