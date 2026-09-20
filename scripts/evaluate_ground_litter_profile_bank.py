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
    BankError, atomic_write_json, bank_envelope, load_bank, sha256_file,
    validate_calibration,
)
from rtsp_annotator.ground_litter_profile_match import (  # noqa: E402
    MatchEnvelope, coerce_envelope, descriptor_coarse_distance,
    envelope_from_samples, extract_grid_descriptor, global_descriptor_scale,
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
    parser.add_argument("--days", default="",
                        help="只评估这些日期（逗号分隔 YYYY-MM-DD）；用于把独立"
                             "验证集与训练/校准集分开")
    parser.add_argument("--record-split", action="store_true",
                        help="在报告中记录本次评估所属的分区与来源")
    parser.add_argument("--allow-uncalibrated-bank", action="store_true",
                        help="允许加载缺少冻结包络的历史 Bank（仅用于回归对照）")
    parser.add_argument("--refit-envelope-on-input", action="store_true",
                        help="用待评数据重新拟合包络（会破坏盲测独立性，仅回归对照）")
    parser.add_argument("--small-target-trials", type=int, default=12,
                        help="注入并逐目标验证的小目标数量（0=关闭）")
    parser.add_argument("--small-target-size-native-px", type=int, default=8,
                        help="按原图画布计的小目标边长（像素）")
    parser.add_argument("--event-memory", action="store_true",
                        help="接入 V33 事件 memory 做离线生命周期验证")
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
    bank = load_bank(
        args.bank_root, args.bank_id, args.version,
        require_calibration=not args.allow_uncalibrated_bank,
    )
    bank_size = bank.reference_size
    calibration_problems = validate_calibration(bank.matcher, list(bank.ids()))
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
    report["replay_size"] = list(size)
    report["bank_reference_size"] = list(bank_size)
    # R1：默认只读 Bank 里冻结的包络。只有显式 --refit-envelope-on-input
    # 才在待评数据上重新拟合，并在报告里标红。
    frozen_envelopes = {
        pid: bank_envelope(bank.matcher, pid) for pid in bank.ids()
    }
    if args.refit_envelope_on_input:
        probe_scores = _calibration_scores(
            bank, args.input, size, roi, overlay, matcher,
            max_files=3, max_frames=12,
        )
        envelopes = {
            pid: envelope_from_samples(probe_scores.get(pid, []), matcher)
            for pid in bank.ids()
        }
        envelope_source = "refit_on_evaluation_input"
        refit_warning = (
            "本次评估在待评数据上重新拟合了包络：该目录不再是独立盲测，"
            "结果只能作为回归对照。"
        )
    else:
        if any(value is None for value in frozen_envelopes.values()):
            raise SystemExit(
                "Bank 缺少冻结包络；请用 --allow-uncalibrated-bank 读取历史产物，"
                "或重建 Bank。禁止在待评数据上悄悄补拟合。"
            )
        envelopes = {
            pid: coerce_envelope(value) for pid, value in frozen_envelopes.items()
        }
        envelope_source = "frozen_in_bank"
        refit_warning = ""
    report["envelope_source"] = envelope_source
    report["envelope_refit_warning"] = refit_warning
    report["bank_calibration"] = dict(bank.matcher.get("calibration") or {})
    report["bank_calibration_problems"] = calibration_problems

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
    gaps: list[dict[str, Any]] = []
    false_positive_samples: list[dict[str, Any]] = []
    per_target: list[dict[str, Any]] = []
    source_seconds = 0.0
    decode_seconds = 0.0
    io_wait_seconds = 0.0
    frame_index = 0
    rng = np.random.default_rng(args.seed)
    previous_source_time: float | None = None
    file_states: list[dict[str, Any]] = []

    if args.input is None:
        raise SystemExit("--input 为必填：评估必须消费真实/回放的连续录像")
    media = collect_media(Path(args.input))
    if args.days:
        wanted_days = {
            value.strip() for value in str(args.days).split(",") if value.strip()
        }
        media = [item for item in media
                 if str(item.record_start)[:10] in wanted_days]
    planned_files = len(media)
    if args.max_files:
        media = media[: args.max_files]
    if not media:
        raise SystemExit(f"目录里没有可评估的录像: {args.input}")
    report["split"] = {
        "days": sorted({str(item.record_start)[:10] for item in media}),
        "source_root": str(args.input),
        "declared_role": "independent_validation",
        "note": ("独立验证集必须与训练/校准集按日期隔离；本报告只记录本次实际"
                 "消费的日期，不声称覆盖训练集"),
    }
    report["files_planned"] = planned_files
    report["files_attempted"] = len(media)
    report["files_evaluated"] = len(media)

    # R6：按**原图画布**尺寸换算小目标边长，并只在 ROI 内注入。
    native_w, native_h = bank_size
    canvas_scale = size[0] / max(native_w, 1)
    inject_side = max(
        2, int(round(args.small_target_size_native_px * canvas_scale)),
    )
    roi_native = roi_mask_from_geometry(geometry, native_w, native_h)
    event_memory = _build_event_memory(matcher) if args.event_memory else None
    trials_per_file = max(
        0, int(round(args.small_target_trials / max(1, len(media)))),
    )

    selector.reset_search(reason="evaluation_start")
    for item in media:
        io_started = time.monotonic()
        probe = probe_recording(item.file_name)
        io_wait_seconds += time.monotonic() - io_started
        if not probe.ok:
            gaps.append({"file": item.file_id, "reason": "DECODE_FAILED"})
            file_states.append({"file": item.file_id, "state": "decode_failed"})
            selector.mark_non_observable(
                seconds=_expected_file_seconds(item, args, probe),
                reason="DECODE_FAILED",
            )
            continue
        reader = SequentialFrameReader(item.file_name)
        source_fps = (
            (probe.frame_count / probe.duration_seconds)
            if probe.duration_seconds > 0 and probe.frame_count else 25.0
        )
        stride = max(1, int(round(source_fps / max(args.analysis_fps, 1e-3))))
        ticks_this_file = 0
        planned_ticks = 0
        try:
            for captured in reader.iter_frames():
                if frame_index % stride:
                    frame_index += 1
                    continue
                frame_index += 1
                frame = cv2.resize(captured.frame, size, interpolation=cv2.INTER_AREA)
                quality = frame_quality(frame)
                if not quality.usable:
                    gaps.append({
                        "file": item.file_id, "t": captured.time_seconds,
                        "reason": "QUALITY", "reasons": list(quality.reasons),
                    })
                    selector.mark_non_observable(
                        seconds=1.0 / max(args.analysis_fps, 1e-3),
                        reason="QUALITY",
                    )
                    continue
                source_time = _source_seconds(item, captured.time_seconds)
                if previous_source_time is not None and source_time <= previous_source_time:
                    gaps.append({
                        "file": item.file_id, "t": captured.time_seconds,
                        "reason": "SOURCE_TIME_REWIND",
                    })
                    selector.mark_alignment_change(selector.alignment_generation + 1)
                gap_seconds = (
                    0.0 if previous_source_time is None
                    else max(0.0, source_time - previous_source_time)
                )
                previous_source_time = source_time
                source_seconds = max(source_seconds, source_time)
                planned_ticks += 1
                # 无目标基线：第一个 tick；带目标：后续按配额注入（成对检查）。
                trial_index = planned_ticks - 1
                inject = (
                    trials_per_file > 0
                    and trial_index % 2 == 1
                    and len([row for row in per_target
                             if row["file"] == item.file_id]) < trials_per_file
                )
                if inject:
                    frame, truth = _inject_small_target_in_roi(
                        frame, roi_native, size, inject_side, rng,
                    )
                    baseline_tick = ticks[-1] if ticks else None
                else:
                    truth = None
                    baseline_tick = None
                tick = _run_tick(
                    selector, contexts, descriptors, scale, envelopes, matcher,
                    frame, roi, size, geometry, source_time=source_time,
                    tick_index=len(ticks), tick_interval=gap_seconds or None,
                    event_memory=event_memory,
                )
                tick["file"] = item.file_id
                tick["gap_seconds"] = round(gap_seconds, 3)
                ticks.append(tick)
                ticks_this_file += 1
                if truth is not None:
                    per_target.append(_match_target(
                        truth, tick, baseline_tick=baseline_tick,
                        tick_index=len(ticks) - 1, file_id=item.file_id,
                        source_time=source_time,
                    ))
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
            file_states.append({
                "file": item.file_id, "state": "read_error",
                "error": type(exc).__name__,
            })
            selector.mark_non_observable(
                seconds=_expected_file_seconds(item, args, probe),
                reason="READ_ERROR",
            )
            continue
        decode_seconds += reader.decode_seconds
        report.setdefault("files", []).append({
            "file": item.file_id,
            "duration_seconds": round(probe.duration_seconds, 3),
            "decoded_frames": reader.frames_decoded,
            "decode_seconds": round(reader.decode_seconds, 3),
            "source_seconds": round(_source_seconds(item, 0.0), 3),
            "ticks": ticks_this_file,
        })
        file_states.append({
            "file": item.file_id, "state": "ok", "ticks": ticks_this_file,
        })

    summary = selector.summarise(
        join_gap_seconds=replay_config.get("join_gap_seconds"),
    )
    planned_files = report["files_planned"]
    attempted_files = report["files_attempted"]
    succeeded = [row for row in file_states if row.get("state") == "ok"]
    failed = [row for row in file_states if row.get("state") != "ok"]
    static_ticks = sum(1 for tick in ticks if tick["static_eligible_ids"])
    static_fraction = 0.0 if not ticks else static_ticks / len(ticks)
    small_target = _small_target_summary(per_target)
    report.update({
        "ticks": len(ticks),
        "files": {
            "planned": planned_files,
            "attempted": attempted_files,
            "succeeded": len(succeeded),
            "failed": failed,
            "per_file": file_states,
        },
        "static_potential_coverage": {
            "fraction": round(static_fraction, 5),
            "covered_ticks": static_ticks,
            "ticks": len(ticks),
            "note": "潜在覆盖只表示库能力，不能冒充动态可用覆盖",
        },
        "dynamic_coverage": summary,
        "observation_accounting": {
            "observed_seconds": summary["observed_seconds"],
            "effective_seconds": summary["effective_seconds"],
            "off_air_seconds": summary["off_air_seconds"],
            "non_observable_seconds": summary["non_observable_seconds"],
            "non_observable_reasons": summary["non_observable_reasons"],
            "unobservable_ticks": summary["unobservable_ticks"],
            "wall_span_seconds": summary["wall_span_seconds"],
            "note": ("有效 = 可判断且当前参考仍合格的区间；录像空缺与不可判断"
                     "分别记账，不做前向填充"),
        },
        "gaps": gaps,
        "gap_count": len(gaps),
        "small_target": small_target,
        "availability_checks": {
            "ticks_with_zero_availability": sum(
                1 for tick in ticks if tick["availability_max"] == 0.0
            ),
            "blocked_reasons": sorted({
                reason for tick in ticks
                for reason in (tick.get("blocked_availability") or {}).values()
            }),
        },
        "false_positive_examples": false_positive_samples[:40],
        "selector_config_used": {
            key: replay_config.get(key) for key in (
                "max_observation_gap_seconds", "join_gap_seconds",
                "switch_min_samples", "switch_min_span_seconds",
                "recovery_min_samples", "recovery_min_span_seconds",
                "min_dwell_seconds", "top_k", "max_small_matches_per_tick",
                "tick_interval_seconds", "result_validity_seconds",
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
            "effective_seconds": summary["effective_seconds"],
            "observed_seconds": summary["observed_seconds"],
            "non_observable_seconds": summary["non_observable_seconds"],
            "off_air_seconds": summary["off_air_seconds"],
            "pause_p95": summary["pause_p95"],
            "pause_max": summary["pause_max"],
            "switch_count": summary["switch_count"],
            "unknown_reasons": summary["unknown_reasons"],
        },
        "envelope_source": report.get("envelope_source"),
        "files": report["files"],
        "small_target": {
            "trials": small_target["trials"],
            "detected": small_target["detected"],
            "detection_fraction": small_target["detection_fraction"],
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
        "files": {
            "planned": planned_files, "attempted": attempted_files,
            "succeeded": len(succeeded),
        },
        "static_fraction": report["static_potential_coverage"]["fraction"],
        "dynamic": {
            "effective_fraction": summary["effective_fraction"],
            "effective_seconds": summary["effective_seconds"],
            "observed_seconds": summary["observed_seconds"],
            "off_air_seconds": summary["off_air_seconds"],
            "non_observable_seconds": summary["non_observable_seconds"],
            "pause_p95": summary["pause_p95"],
            "pause_max": summary["pause_max"],
            "switch_count": summary["switch_count"],
        },
        "small_target": {
            "trials": small_target["trials"],
            "detected": small_target["detected"],
        },
        "envelope_source": report.get("envelope_source"),
        "gaps": len(gaps),
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


def _run_tick(
    selector: ProfileSelector, contexts: Mapping[str, Any],
    descriptors: Mapping[str, Mapping[str, np.ndarray]],
    scale: Mapping[str, float], envelopes: Mapping[str, MatchEnvelope],
    matcher: Mapping[str, Any], frame: np.ndarray, roi: np.ndarray,
    size: tuple[int, int], geometry: Mapping[str, Any],
    *, source_time: float, tick_index: int,
    tick_interval: float | None = None,
    event_memory: Any = None,
) -> dict[str, Any]:
    """一个分析 tick：粗检索 → 有界精排 → Selector 决策。

    R4：预算必须让「扩展检索预留」优先占位，粗排 Top-K 次之，当前参考保底；
    否则 Selector 推进了游标而预留候选仍被粗排挤掉，第 K+1 个可用参考永远
    查不到。当前是否可保持用**真实评分**判断，不能只看“有当前参考”。
    """
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
    current_hold_known = selector.last_current_hold_eligible(current_id)
    plan = selector.plan_tick(
        timestamp=source_time, current_hold_eligible=current_hold_known,
    )
    reserved = [pid for pid in plan.get("reserved", []) if pid in contexts]
    top = [pid for pid, _distance in ordered[:top_k] if pid in contexts]
    to_check: list[str] = []
    for pid in reserved + top + ([current_id] if current_id else []):
        if pid and pid in contexts and pid not in to_check:
            to_check.append(pid)
        if len(to_check) >= budget:
            break
    results: list[CandidateMatch] = []
    static_eligible: list[str] = []
    candidate_boxes = 0
    current_match: CandidateMatch | None = None
    availability: dict[str, float] = {}
    blocked: dict[str, str] = {}
    candidate_box_rows: list[dict[str, Any]] = []
    for pid in to_check:
        context = contexts[pid]
        try:
            evaluation = evaluate_bank_frame(
                context, frame, envelope=envelopes[pid], config=matcher,
                roi_mask=roi,
            )
        except BankError:
            continue
        outcome = evaluation.outcome
        availability[pid] = round(evaluation.availability_fraction, 5)
        if str(outcome.get("reason", "")).startswith("BLOCKED_"):
            blocked[pid] = str(outcome["reason"])
        verified = bool(outcome["enter_eligible"] or outcome["hold_eligible"])
        candidate = CandidateMatch(
            pid, float(outcome["score"] or 0.0),
            bool(outcome["enter_eligible"]), bool(outcome["hold_eligible"]),
            verified=verified,
        )
        results.append(candidate)
        if outcome["enter_eligible"]:
            static_eligible.append(pid)
        candidate_boxes += len(evaluation.candidates)
        candidate_box_rows.extend(
            {"box": list(row["box"]), "profile_id": pid}
            for row in evaluation.candidates
        )
        if pid == current_id:
            current_match = candidate
    decision = selector.observe(
        timestamp=source_time, current=current_match, candidates=results,
        tested_profile_ids=to_check, observable=True,
        tick_interval_seconds=tick_interval,
    )
    if decision.commit_requested:
        profile_id = decision.commit_profile_id or ""
        if not selector.discard_stale(
            profile_id=profile_id, observed_at=source_time, now=source_time,
        ):
            record = selector.commit(
                profile_id=profile_id, timestamp=source_time,
            )
            if record["previous"] != record["profile_id"]:
                _notify_switch(event_memory, record)
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
        "tested_budget": len(to_check),
        "budget": budget,
        "static_eligible_ids": sorted(static_eligible),
        "candidate_boxes": int(candidate_boxes),
        "candidate_count": int(candidate_boxes),
        "candidate_boxes_detail": candidate_box_rows,
        "availability_fraction": availability,
        "blocked_availability": blocked,
        "availability_max": (
            round(max(availability.values()), 5) if availability else 0.0
        ),
        "score_current": decision.score_current,
        "score_best": decision.score_best,
        "budget_exhausted": decision.budget_exhausted,
        "search_cursor": decision.search_cursor,
        "alignment_generation": decision.alignment_generation,
        "observation_span_seconds": decision.as_dict().get(
            "observation_span_seconds"
        ),
    }


def _notify_switch(event_memory: Any, record: Mapping[str, Any]) -> None:
    """提交边界通知事件 memory：保留身份、清空跨参考证据窗口（R6）。"""
    if event_memory is None:
        return
    try:
        event_memory.notify_reference_switch(
            previous_profile_id=str(record.get("previous") or ""),
            profile_id=str(record.get("profile_id") or ""),
            generation=int(record.get("generation", 0)),
            same_profile_id=bool(record.get("same_profile_id", False)),
        )
    except Exception:  # pragma: no cover - memory 异常不得影响评估主链
        return


def _build_event_memory(matcher: Mapping[str, Any]) -> Any:
    """构造离线 V33 事件 memory，用于切换期间证据冻结/恢复的生命周期验证。"""
    from types import SimpleNamespace
    from rtsp_annotator.ground_litter_v33 import V33EventMemory

    options = SimpleNamespace(
        analysis_fps=0.5,
        semantic_scan_interval_seconds=4.0,
        semantic_confirm_span_seconds=6.0,
        prior_confirm_span_seconds=6.0,
        semantic_hit_window=6,
        prior_hit_window=6,
        fused_hit_window=6,
        semantic_hit_count=4,
        prior_hit_count=4,
        fused_hit_count=4,
        semantic_clear_min_misses=3,
        semantic_clear_seconds=8.0,
        clear_confirm_seconds=5.0,
        pending_expire_seconds=15.0,
        prior_suspend_expire_seconds=120.0,
        min_clean_valid_fraction=0.8,
        actor_overlap_threshold=0.2,
        maximum_closed_events=200,
    )
    del matcher
    return V33EventMemory(options, pixel_scale=1.0)


def _expected_file_seconds(item: RecordingFile, args: Any, probe: Any) -> float:
    """一个文件的预期可观测秒数，用于把“没读到”的时间记为不可判断。"""
    del args
    try:
        duration = float(probe.duration_seconds)
        if duration > 0:
            return duration
    except Exception:
        pass
    try:
        return max(0.0, parse_seconds(item.record_end) - parse_seconds(item.record_start))
    except Exception:
        return 0.0


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
        reference_canvas_width=int(context.reference.shape[1]),
    )


def _source_seconds(item: RecordingFile, offset: float) -> float:
    try:
        return parse_seconds(item.record_start) + float(offset)
    except Exception:
        return float(offset)


def _inject_small_target_in_roi(
    frame: np.ndarray, roi_native: np.ndarray, size: tuple[int, int],
    side: int, rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """在 ROI 内按**原图尺度**注入小目标，并返回真值框（R6）。

    选点先**在原图画布**上完成（只在 ROI 像素里抽样），再映射到评估画布，
    因此注入位置一定落在有效区域内；``side`` 已经是按画布缩放后的边长。
    """
    if side < 2:
        return frame, None
    native_h, native_w = roi_native.shape[:2]
    rows, cols = np.nonzero(roi_native > 0)
    if rows.size == 0:
        return frame, None
    half = side // 2
    for _ in range(32):
        pick = int(rng.integers(0, rows.size))
        native_y, native_x = int(rows[pick]), int(cols[pick])
        scale_x = size[0] / max(native_w, 1)
        scale_y = size[1] / max(native_h, 1)
        left = int(round(native_x * scale_x)) - half
        top = int(round(native_y * scale_y)) - half
        left = max(0, min(left, size[0] - side))
        top = max(0, min(top, size[1] - side))
        box = [left, top, left + side, top + side]
        canvas_roi = roi_mask_from_geometry(
            {"roi": [[0, 0], [1, 0], [1, 1], [0, 1]]}, size[0], size[1],
        )
        if int(np.count_nonzero(canvas_roi[top:top + side, left:left + side])) > 0:
            target = frame.copy()
            target[top:top + side, left:left + side] = 250
            return target, {
                "box": box, "side": side,
                "native_side": int(round(side / max(scale_x, 1e-6))),
                "native_center": [native_x, native_y],
                "injected": True,
            }
    return frame, None


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(v) for v in left)
    rx1, ry1, rx2, ry2 = (float(v) for v in right)
    inter_w = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    inter_h = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    inter = inter_w * inter_h
    union = (
        max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
        + max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
        - inter
    )
    return 0.0 if union <= 0 else inter / union


def _match_target(
    truth: Mapping[str, Any], tick: Mapping[str, Any],
    *, baseline_tick: Mapping[str, Any] | None, tick_index: int,
    file_id: str, source_time: float,
) -> dict[str, Any]:
    """把注入目标与该 tick 的候选做**空间匹配**（R6）。

    区分三层：任意候选、目标命中（IoU≥0.1 或中心落入真值框）、事件确认。
    同时给出无目标基线的候选数，便于判断“命中”是不是噪声造成的。
    """
    boxes = [row["box"] for row in tick.get("candidate_boxes_detail", [])] or []
    truth_box = list(truth["box"])
    hits = [box for box in boxes if _box_iou(box, truth_box) >= 0.1]
    center_x = (truth_box[0] + truth_box[2]) / 2.0
    center_y = (truth_box[1] + truth_box[3]) / 2.0
    centers = [
        box for box in boxes
        if box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]
    ]
    return {
        "tick": tick_index,
        "file": file_id,
        "source_time": round(float(source_time), 3),
        "truth_box": truth_box,
        "native_side": truth.get("native_side"),
        "available": bool(tick.get("prior_allowed")),
        "availability_max": tick.get("availability_max"),
        "any_candidate": bool(tick.get("candidate_boxes")),
        "candidate_boxes": tick.get("candidate_boxes"),
        "iou_hits": len(hits),
        "center_hits": len(centers),
        "target_detected": bool(hits or centers),
        "baseline_candidate_boxes": (
            None if baseline_tick is None else baseline_tick.get("candidate_boxes")
        ),
        "blocked_reason": next(iter(
            (tick.get("blocked_availability") or {}).values()
        ), None),
    }


def _small_target_summary(per_target: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_target = list(per_target)
    detected = [row for row in by_target if row["target_detected"]]
    paired = [row for row in by_target if row["baseline_candidate_boxes"] is not None]
    paired_noise = [
        row for row in paired
        if not row["target_detected"] and (row["baseline_candidate_boxes"] or 0) > 0
    ]
    return {
        "trials": len(by_target),
        "detected": len(detected),
        "detection_fraction": (
            0.0 if not by_target else round(len(detected) / len(by_target), 5)
        ),
        "ticks_with_any_candidate": sum(
            1 for row in by_target if row["any_candidate"]
        ),
        "trials_with_target_but_no_candidate": sum(
            1 for row in by_target
            if not row["any_candidate"]
        ),
        "paired_baselines": len(paired),
        "paired_baseline_with_candidates": len(paired_noise),
        "per_target": by_target,
        "discovery_to_commit_seconds": None,
        "note": (
            "小目标为原图尺度合成叠加，逐目标空间匹配（IoU≥0.1 或中心命中）；"
            "这是机制验证，不代表现场识别准确率"
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
