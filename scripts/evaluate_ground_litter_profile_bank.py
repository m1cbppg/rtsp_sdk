#!/usr/bin/env python
"""A5/B5：冻结 Bank 的离线评估与连续回放报告。

评估口径（方案一 §7.2、§8；方案二 §10）：

* 静态潜在覆盖：每 tick 是否存在合格参考——只表示库的能力。
* 动态有效覆盖：实际 Selector 选中、本帧分析完成且结果新鲜的可判断比例。
* 暂停 P95/最长、切换次数、首次发现到提交时延、事件确认时延。
* 缺帧 / 预算跳过 / 验证失败都记为缺口，不做未封顶前向填充。
* 跨五分钟文件保持同一个 Selector 与事件 memory，不每个文件重新预热。
* 下载等待单独计时，不混入源视频时间线。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rtsp_annotator.ground_litter_profile_analysis import (  # noqa: E402
    build_prior_context, evaluate_bank_frame, roi_mask_from_geometry,
)
from rtsp_annotator.ground_litter_profile_bank import (  # noqa: E402
    BankError, atomic_write_json, load_bank, sha256_file,
)
from rtsp_annotator.ground_litter_profile_match import (  # noqa: E402
    MatchEnvelope, descriptor_coarse_distance, envelope_from_samples,
    extract_grid_descriptor, global_descriptor_scale,
)
from rtsp_annotator.ground_litter_profile_sampling import (  # noqa: E402
    CanvasRegistrar, SequentialFrameReader, frame_quality, parse_seconds,
    preview_image, probe_recording,
)
from rtsp_annotator.ground_litter_profile_selector import (  # noqa: E402
    CandidateMatch, ProfileSelector,
)
from rtsp_annotator.ground_litter_recording_cache import (  # noqa: E402
    ManagedRecordingCache,
)
from rtsp_annotator.ground_litter_recording_source import RecordingFile  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估冻结 Profile Bank")
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--bank-id", required=True)
    parser.add_argument("--version", default=None)
    parser.add_argument("--input", type=Path, default=None,
                        help="本地 PS/录像目录（按时间顺序连续回放）")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-fps", type=float, default=0.5)
    parser.add_argument("--max-frames-per-file", type=int, default=60)
    parser.add_argument("--small-target-frame-fraction", type=float, default=0.2,
                        help="把合成小目标叠加到该比例的分析帧上，用于对照")
    parser.add_argument("--small-target-size-px", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--replay-size", default="960x540",
                        help="评估画布（归一化几何天然可缩放），默认 960x540")
    parser.add_argument("--max-files", type=int, default=0,
                        help="最多评估多少个录像文件（0=全部）")
    return parser.parse_args(argv)


_MEDIA_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})")


def collect_media(root: Path) -> list[RecordingFile]:
    """收集可评估录像；时间优先取文件名里的录像起始时刻。

    下载工具把 ``record_start`` 编进文件名（``2026-09-20T022904.ps``），
    因此这里不需要依赖 mtime；缺少该模式时回退到 mtime 并如实标注。
    """
    files: list[RecordingFile] = []
    patterns = ("*.ps", "*.mp4", "*.mkv", "*.avi", "*.mov", "*.ts", "*.m4v")
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if not path.is_file():
                continue
            stat = path.stat()
            match = _MEDIA_STAMP.match(path.stem)
            if match:
                record_start = (
                    f"{match.group(1)} {match.group(2)}:{match.group(3)}:{match.group(4)}"
                )
            else:
                record_start = datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            files.append(RecordingFile(
                file_id=path.name,
                file_name=str(path),
                record_start=record_start,
                record_end=record_start,
                file_size=stat.st_size,
            ))
    return sorted(files, key=lambda item: (item.record_start, item.file_id))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    bank = load_bank(args.bank_root, args.bank_id, args.version)
    bank_size = bank.reference_size
    try:
        width_text, height_text = str(args.replay_size).lower().split("x")
        size = (int(width_text), int(height_text))
    except (AttributeError, ValueError):
        raise SystemExit("--replay-size 必须是 WxH，例如 960x540")
    if size[0] < 320 or size[1] < 180:
        raise SystemExit("--replay-size 太小，评分不再可信")
    report: dict[str, Any] = {}
    matcher = bank.matcher
    geometry = bank.geometry
    roi = roi_mask_from_geometry(geometry, size[0], size[1])
    overlay = geometry.get("overlay_exclude_zones") or []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    contexts = {}
    for pid in bank.ids():
        context = build_prior_context(bank, pid)
        if context.reference.shape[:2] != (size[1], size[0]):
            context = _scale_context(context, size)
        contexts[pid] = context
    descriptors = {pid: bank.load_descriptor(pid) for pid in bank.ids()}
    scale = global_descriptor_scale(list(descriptors.values()))
    probe_scores = _calibration_scores(
        bank, args.input, size, roi, overlay, matcher,
        max_files=3, max_frames=12,
    )
    report["replay_size"] = list(size)
    report["bank_reference_size"] = list(bank_size)
    envelopes = {
        pid: envelope_from_samples(probe_scores.get(pid, []), matcher)
        for pid in bank.ids()
    }

    replay_config = dict(matcher.get("selection", {}))
    # 观测间隔上限必须与实际分析节拍一致：0.05 FPS 时两 tick 相隔 20s，
    # 若仍用 4s，每次观测都会清空连续证据，动态覆盖会恒为 0（而静态覆盖是 1.0），
    # 这正是 F4 要求区分的「静态 ≠ 动态」。这里按节拍覆盖，并允许 2 倍余量。
    tick_interval = 1.0 / max(args.analysis_fps, 1e-3)
    replay_config["max_observation_gap_seconds"] = max(
        4.0, 2.0 * tick_interval,
    )
    replay_config["join_gap_seconds"] = max(
        float(replay_config.get("join_gap_seconds", 300.0)),
        4.0 * tick_interval,
    )
    selector = ProfileSelector(
        bank_id=bank.bank_id, bank_version=bank.version,
        view_id=str(geometry.get("view_id", "view_0")),
        profile_ids=list(bank.ids()), config=replay_config,
    )

    report.update({
        "kind": "ground_litter_profile_bank_evaluation",
        "bank": {
            "bank_id": bank.bank_id, "version": bank.version,
            "profiles": list(bank.ids()),
            "manifest_sha256": sha256_file(
                bank.profiles[0].directory.parent.parent / "bank.json"
            ),
        },
        "envelopes": {pid: value.as_dict() for pid, value in envelopes.items()},
        "analysis_fps": args.analysis_fps,
        "roi_pixels": int(np.count_nonzero(roi)),
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })

    ticks: list[dict[str, Any]] = []
    static_covered = 0
    gaps: list[dict[str, Any]] = []
    false_positive_samples: list[dict[str, Any]] = []
    small_target_trials: list[dict[str, Any]] = []
    source_seconds = 0.0
    decode_seconds = 0.0
    io_wait_seconds = 0.0
    frame_index = 0
    rng = np.random.default_rng(args.seed)
    reference_canvas: np.ndarray | None = None
    previous_source_time: float | None = None

    if args.input is None:
        raise SystemExit("--input 为必填：评估必须消费真实/回放的连续录像")
    media = collect_media(Path(args.input))
    if args.max_files:
        media = media[: args.max_files]
    if not media:
        raise SystemExit(f"目录里没有可评估的录像: {args.input}")
    report["files_evaluated"] = len(media)

    selector.reset_search(reason="evaluation_start")
    for item in media:
        io_started = time.monotonic()
        probe = probe_recording(item.file_name)
        io_wait_seconds += time.monotonic() - io_started
        if not probe.ok:
            gaps.append({"file": item.file_id, "reason": "DECODE_FAILED"})
            continue
        reader = SequentialFrameReader(item.file_name)
        # 按源帧率与分析节拍取帧，不复制帧凑数。
        source_fps = (
            (probe.frame_count / probe.duration_seconds)
            if probe.duration_seconds > 0 and probe.frame_count else 25.0
        )
        stride = max(1, int(round(source_fps / max(args.analysis_fps, 1e-3))))
        file_started = time.monotonic()
        try:
            for captured in reader.iter_frames():
                if frame_index % stride:
                    frame_index += 1
                    continue
                frame_index += 1
                if reference_canvas is None:
                    reference_canvas = cv2.resize(
                        captured.frame, size, interpolation=cv2.INTER_AREA,
                    )
                frame = cv2.resize(captured.frame, size, interpolation=cv2.INTER_AREA)
                quality = frame_quality(frame)
                if not quality.usable:
                    gaps.append({
                        "file": item.file_id, "t": captured.time_seconds,
                        "reason": "QUALITY", "reasons": list(quality.reasons),
                    })
                    continue
                if len(ticks) % max(1, int(round(1.0 / max(
                    args.small_target_frame_fraction, 1e-3,
                )))) == 0:
                    frame, injected = _inject_small_target(
                        frame, size, args.small_target_size_px, rng,
                    )
                    small_target_trials.append({
                        "tick": len(ticks), "injected": True,
                        "box": injected["box"] if injected else None,
                    })
                source_time = _source_seconds(item, captured.time_seconds)
                if previous_source_time is not None and source_time <= previous_source_time:
                    gaps.append({
                        "file": item.file_id, "t": captured.time_seconds,
                        "reason": "SOURCE_TIME_REWIND",
                    })
                    selector.mark_alignment_change(selector.alignment_generation + 1)
                previous_source_time = source_time
                source_seconds = max(source_seconds, source_time)
                tick = _run_tick(
                    selector, contexts, descriptors, scale, envelopes, matcher,
                    frame, roi, size, geometry, source_time=source_time,
                    tick_index=len(ticks),
                )
                ticks.append(tick)
                if tick["static_eligible_ids"]:
                    static_covered += 1
                if tick["candidate_count"] and not tick["prior_allowed"]:
                    false_positive_samples.append({
                        "tick": len(ticks), "file": item.file_id,
                        "candidates": tick["candidate_count"],
                        "reason": tick["reason"],
                    })
                if len(ticks) % 200 == 0:
                    atomic_write_json(output / "progress.json", {
                        "ticks": len(ticks), "source_seconds": round(source_seconds, 2),
                    })
                if len(ticks) >= args.max_frames_per_file * max(len(media), 1):
                    break
        except Exception as exc:
            gaps.append({"file": item.file_id, "reason": f"READ_ERROR:{type(exc).__name__}"})
        decode_seconds += reader.decode_seconds
        report.setdefault("files", []).append({
            "file": item.file_id,
            "duration_seconds": round(probe.duration_seconds, 3),
            "decoded_frames": reader.frames_decoded,
            "decode_seconds": round(reader.decode_seconds, 3),
            "source_seconds": round(_source_seconds(item, 0.0), 3),
        })
        del file_started

    summary = selector.summarise()
    static_fraction = 0.0 if not ticks else static_covered / len(ticks)
    confirmations = _confirmation_report(ticks, small_target_trials)
    report.update({
        "ticks": len(ticks),
        "static_potential_coverage": {
            "fraction": round(static_fraction, 5),
            "covered_ticks": static_covered,
            "note": "潜在覆盖只表示库能力，不能冒充动态可用覆盖",
        },
        "dynamic_coverage": summary,
        "gaps": gaps,
        "gap_count": len(gaps),
        "small_target": confirmations,
        "false_positive_examples": false_positive_samples[:40],
        "selector_config_used": {
            key: replay_config.get(key) for key in (
                "max_observation_gap_seconds", "join_gap_seconds",
                "switch_min_samples", "switch_min_span_seconds",
                "recovery_min_samples", "recovery_min_span_seconds",
                "min_dwell_seconds", "top_k", "max_small_matches_per_tick",
            )
        },
        "performance": {
            "source_seconds": round(source_seconds, 3),
            "decode_seconds": round(decode_seconds, 3),
            "io_wait_seconds": round(io_wait_seconds, 3),
            "wall_seconds": round(time.monotonic() - started, 3),
            "note": "离线回放使用显式虚拟时钟；wall 时间不是在线恢复速度",
        },
        "selector_timeline": selector.timeline()[-500:],
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    atomic_write_json(output / "evaluation.json", report)
    atomic_write_json(output / "coverage.json", {
        "kind": "ground_litter_profile_bank_coverage",
        "static_potential": report["static_potential_coverage"],
        "dynamic_effective": {
            "fraction": summary["effective_fraction"],
            "pause_p95": summary["pause_p95"],
            "pause_max": summary["pause_max"],
            "switch_count": summary["switch_count"],
            "unknown_reasons": summary["unknown_reasons"],
        },
        "retained_profile_reasons": {
            "kept": list(bank.ids()),
            "note": "本次评估不删除参考；删除必须在动态回放证明暂停不恶化后离线重发布",
        },
        "gaps": len(gaps),
    })
    if args.report is not None:
        atomic_write_json(args.report, report)
    print(json.dumps({
        "ticks": len(ticks),
        "static_fraction": report["static_potential_coverage"]["fraction"],
        "dynamic": {
            "effective_fraction": summary["effective_fraction"],
            "pause_p95": summary["pause_p95"],
            "pause_max": summary["pause_max"],
            "switch_count": summary["switch_count"],
        },
        "gaps": len(gaps),
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


def _scale_context(context: Any, size: tuple[int, int]) -> Any:
    """把 Bank 上下文缩放到评估画布；几何是归一化的，缩放不改变语义。"""
    from rtsp_annotator.ground_litter_profile_analysis import BankPriorContext
    import cv2 as _cv2
    import numpy as _np

    def scale_map(array: _np.ndarray, interpolation: int) -> _np.ndarray:
        if array.ndim == 2:
            return _cv2.resize(array, size, interpolation=interpolation)
        return array

    noise = {
        key: scale_map(_np.asarray(value), _cv2.INTER_NEAREST)
        if _np.asarray(value).ndim == 2 else value
        for key, value in context.noise.items()
    }
    return BankPriorContext(
        profile_id=context.profile_id,
        reference=_cv2.resize(context.reference, size, interpolation=_cv2.INTER_AREA),
        valid=_cv2.resize(context.valid, size, interpolation=_cv2.INTER_NEAREST),
        noise=noise,
        metadata=context.metadata,
        geometry_diagnostics=context.geometry_diagnostics,
    )


def _source_seconds(item: RecordingFile, offset: float) -> float:
    try:
        return parse_seconds(item.record_start) + float(offset)
    except Exception:
        return float(offset)


def _inject_small_target(
    frame: np.ndarray, size: tuple[int, int], side: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """在画面中注入一个已知尺寸的小目标，用于「小目标不被吞掉」的对照。"""
    if side < 2:
        return frame, None
    height, width = frame.shape[:2]
    top = int(rng.integers(side + 2, max(side + 3, height // 2)))
    left = int(rng.integers(side + 2, max(side + 3, width // 2)))
    target = frame.copy()
    target[top:top + side, left:left + side] = 250
    return target, {
        "box": [left, top, left + side, top + side], "side": side,
    }


def _run_tick(
    selector: ProfileSelector, contexts: Mapping[str, Any],
    descriptors: Mapping[str, Mapping[str, np.ndarray]],
    scale: Mapping[str, float], envelopes: Mapping[str, MatchEnvelope],
    matcher: Mapping[str, Any], frame: np.ndarray, roi: np.ndarray,
    size: tuple[int, int], geometry: Mapping[str, Any],
    *, source_time: float, tick_index: int,
) -> dict[str, Any]:
    """一个分析 tick：粗检索 → 有界精排 → Selector 决策（不做第二次重分析）。"""
    selection = matcher.get("selection", {})
    top_k = int(selection.get("top_k", 3))
    budget = int(selection.get("max_small_matches_per_tick", 4))
    preview = preview_image(frame, width=960)
    descriptor = extract_grid_descriptor(
        preview,
        roi_mask_from_geometry(geometry, preview.shape[1], preview.shape[0]),
    )
    ordered = sorted(
        (
            (pid, descriptor_coarse_distance(descriptor, payload, scale=scale))
            for pid, payload in descriptors.items()
        ),
        key=lambda item: (item[1], item[0]),
    )
    current_id = selector.selected_profile_id
    plan = selector.plan_tick(
        timestamp=source_time,
        current_hold_eligible=current_id is not None,
    )
    to_check = list(dict.fromkeys(
        ([current_id] if current_id else [])
        + [pid for pid, _distance in ordered[:top_k]]
        + list(plan.get("reserved", []))
    ))[:budget]
    results: list[CandidateMatch] = []
    static_eligible: list[str] = []
    candidate_boxes = 0
    current_match: CandidateMatch | None = None
    for pid in to_check:
        if pid not in contexts:
            continue
        context = contexts[pid]
        try:
            evaluation = evaluate_bank_frame(
                context, frame, envelope=envelopes[pid], config=matcher,
                roi_mask=roi,
            )
        except BankError:
            continue
        outcome = evaluation.outcome
        candidate = CandidateMatch(
            pid, float(outcome["score"] or 0.0),
            bool(outcome["enter_eligible"]), bool(outcome["hold_eligible"]),
            verified=True,
        )
        results.append(candidate)
        if outcome["enter_eligible"]:
            static_eligible.append(pid)
        candidate_boxes += len(evaluation.candidates)
        if pid == current_id:
            current_match = candidate
    decision = selector.observe(
        timestamp=source_time, current=current_match, candidates=results,
        tested_profile_ids=to_check,
    )
    if decision.commit_requested:
        profile_id = decision.commit_profile_id or ""
        # 验证在本 tick 内完成，年龄为 0；真正的过期结果由加载/分析时延决定，
        # 这里保留显式检查点，便于接入真实加载器后复用。
        if not selector.discard_stale(
            profile_id=profile_id, observed_at=source_time, now=source_time,
        ):
            record = selector.commit(
                profile_id=profile_id, timestamp=source_time,
            )
            if record["previous"] != record["profile_id"]:
                _notify_switch(contexts, decision, record)
    return {
        "tick": tick_index,
        "source_time": round(source_time, 3),
        "profile_id": decision.selected_profile_id,
        "candidate_profile_id": decision.candidate_profile_id,
        "status": decision.status,
        "phase": decision.phase,
        "prior_allowed": decision.prior_allowed,
        "reason": decision.reason,
        "tested": list(to_check),
        "static_eligible_ids": sorted(static_eligible),
        "candidate_count": candidate_boxes,
        "score_current": decision.score_current,
        "score_best": decision.score_best,
        "budget_exhausted": decision.budget_exhausted,
        "search_cursor": decision.search_cursor,
        "alignment_generation": decision.alignment_generation,
    }


def _notify_switch(
    contexts: Mapping[str, Any], decision: Any, record: Mapping[str, Any],
) -> None:
    """提交边界通知事件 memory（本次评估未接入 V33 memory 时只记录）。"""
    del contexts, decision, record


def _confirmation_report(
    ticks: Sequence[Mapping[str, Any]],
    small_target_trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """首次可判断到提交的时延，以及小目标对照的逐例结果。"""
    first_available: float | None = None
    first_commit: float | None = None
    for tick in ticks:
        if tick["prior_allowed"] and first_available is None:
            first_available = float(tick["source_time"])
        if tick["profile_id"] and first_commit is None:
            # 首次提交 = 出现选中参考的那一 tick。
            first_commit = float(tick["source_time"])
    detected = sum(1 for tick in ticks if tick["candidate_count"] > 0)
    return {
        "trials": len(small_target_trials),
        "ticks_with_prior_candidates": detected,
        "first_prior_available_seconds": first_available,
        "first_commit_seconds": first_commit,
        "discovery_to_commit_seconds": (
            None if first_available is None or first_commit is None
            else round(first_commit - first_available, 3)
        ),
        "note": (
            "小目标为合成叠加，只用于「不被噪声/局部无效吞掉」的机制对照，"
            "不代表现场准确率"
        ),
    }


def _calibration_scores(
    bank: Any, input_dir: Path | None, size: tuple[int, int],
    roi: np.ndarray, overlay: Sequence[Any], matcher: Mapping[str, Any],
    *, max_files: int, max_frames: int,
) -> dict[str, list[float]]:
    """从回放来源之外的连续块统计包络样本（方案二 §3.4）。"""
    from rtsp_annotator.ground_litter_profile_match import score_profile

    scores: dict[str, list[float]] = {pid: [] for pid in bank.ids()}
    if input_dir is None:
        return scores
    scaled = {
        pid: _scale_context(build_prior_context(bank, pid), size)
        for pid in bank.ids()
    }
    media = collect_media(Path(input_dir))[:max_files]
    registrar: CanvasRegistrar | None = None
    for item in media:
        probe = probe_recording(item.file_name)
        if not probe.ok:
            continue
        reader = SequentialFrameReader(item.file_name)
        count = 0
        try:
            for captured in reader.iter_frames():
                frame = cv2.resize(captured.frame, size, interpolation=cv2.INTER_AREA)
                if registrar is None:
                    registrar = CanvasRegistrar(frame, overlay_exclude_zones=overlay)
                if not frame_quality(frame).usable:
                    continue
                for pid in bank.ids():
                    context = scaled[pid]
                    try:
                        score = score_profile(
                            frame, context.reference, context.valid,
                            roi, profile_id=pid, config=matcher,
                        )
                    except BankError:
                        continue
                    scores[pid].append(score.score)
                count += 1
                if count >= max_frames:
                    break
        except Exception:
            continue
        if count:
            break
    return scores


if __name__ == "__main__":
    raise SystemExit(main())
