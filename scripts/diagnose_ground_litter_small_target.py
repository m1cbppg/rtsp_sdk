"""小目标损失定位诊断：同一帧、同一注入位置、同一 Bank 的四种对照（只读）。

对应用户 v3 复核后的第六项要求：

* A. 原分辨率共享 prior adapter（真正以原尺寸执行，不缩小冒充）
* B. 当前在线 960×540 分析路径
* C. A/B 中实际生效的参考
* D. 离线逐参考比较得到的最佳候选参考（仅作诊断上界）

每个对照都有**同帧未注入**基线；注入位置在 ROI 内并标明是否落在参考有效区域；
在目标位置比较注入/未注入差异（不用全图任意候选当基线误报）；记录目标在哪一步
丢失（ROI/valid、残差、阈值、形态学、尺寸过滤、参考选择、prior_allowed）；
分开报告搜索候选、生效参考候选与允许输出的候选。**不修改任何阈值。**

用法::

    python scripts/diagnose_ground_litter_small_target.py \
        --bank-root out/bank --bank-id camera_01030 --version v3 \
        --media /path/to/ps_or_mp4 --frame-index 30 \
        --native-size-px 8,24 --output out/c1c4_diag/small_target
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rtsp_annotator.ground_litter_profile_analysis import (  # noqa: E402
    BankPriorContext, build_prior_context_at, evaluate_bank_frame,
    roi_mask_from_geometry,
)
from rtsp_annotator.ground_litter_profile_bank import (  # noqa: E402
    BankError, atomic_write_json, load_bank,
)
from rtsp_annotator.ground_litter_profile_match import (  # noqa: E402
    analyze_residual_support, apply_compensation, descriptor_coarse_distance,
    evaluate_match, extract_grid_descriptor, global_descriptor_scale,
    residual_maps, robust_color_compensation, score_profile,
)
from rtsp_annotator.ground_litter_profile_sampling import (  # noqa: E402
    SequentialFrameReader, frame_quality, preview_image,
)
from rtsp_annotator.ground_litter_profile_selector import (  # noqa: E402
    CandidateMatch, ProfileSelector,
)

STAGES = (
    "FRAME_QUALITY", "ROI_OR_VALID", "RESIDUAL_BELOW_THRESHOLD",
    "THRESHOLD_BELOW_NOISE_FLOOR", "MORPHOLOGY_EMPTY",
    "SIZE_FILTERED", "REFERENCE_NOT_SELECTED", "PRIOR_NOT_ALLOWED",
    "SURVIVED",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="小目标损失定位诊断（只读）")
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--bank-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, default=None,
                        help="复用 Bank 冻结几何（缺省时从 bank.json 读）")
    parser.add_argument("--frame-index", type=int, default=30)
    parser.add_argument("--native-size-px", default="8,24",
                        help="按原图画布计的目标边长（逗号分隔）")
    parser.add_argument("--native-position", default="",
                        help="可选：注入中心，原图像素 'x,y'；缺省时取 ROI 中心")
    parser.add_argument("--online-size", default="960x540",
                        help="在线分析画布（与评估器一致）")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-visuals", action="store_true")
    return parser.parse_args()


def _parse_size(text: str) -> tuple[int, int]:
    width, _, height = str(text).lower().partition("x")
    return int(width), int(height)


def _read_first_frame(path: Path, index: int) -> np.ndarray | None:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return None if image is None else image
    reader = SequentialFrameReader(path)
    for position, captured in enumerate(reader.iter_frames()):
        if position >= index:
            return captured.frame.copy()
    return None


def _collect_media(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    allowed = {".mp4", ".ps", ".mkv", ".avi", ".mov",
               ".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )


def _load_geometry(bank_root: Path, bank_id: str, version: str,
                   path: Path | None) -> dict[str, Any]:
    """读冻结几何；`--geometry` 只用于覆盖 ROI/排除区，画布尺寸仍以 Bank 为准。"""
    directory = bank_root / bank_id / version
    payload: dict[str, Any] = {}
    frozen = directory / "camera_geometry.json"
    if frozen.is_file():
        payload = json.loads(frozen.read_text(encoding="utf-8"))
    if path is not None:
        override = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("roi", "exclude_zones", "overlay_exclude_zones", "view_id"):
            if key in override:
                payload[key] = override[key]
    if not payload:
        raise SystemExit(f"没有可用的冻结几何: {frozen}")
    return payload


def _inject(frame: np.ndarray, center: tuple[int, int], side: int,
            roi: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = frame.shape[:2]
    half = side // 2
    left = max(0, min(int(center[0]) - half, width - side))
    top = max(0, min(int(center[1]) - half, height - side))
    target = frame.copy()
    patch = target[top:top + side, left:left + side]
    fill = int(np.clip(float(patch.mean()) + 90.0, 0, 255))
    target[top:top + side, left:left + side] = fill
    box = [left, top, left + side, top + side]
    return target, {
        "box": box, "side": int(side), "fill_value": fill,
        "center": [int(round((left + box[2]) / 2)), int(round((top + box[3]) / 2))],
        "inside_roi": bool(np.count_nonzero(
            roi[top:top + side, left:left + side]
        )),
    }


def _inside(box: Sequence[float], frame_shape: Sequence[int]) -> bool:
    return bool(
        0 <= box[0] < frame_shape[1] and 0 <= box[1] < frame_shape[0]
    )


def _match_position(candidates: Sequence[Mapping[str, Any]],
                    box: Sequence[float]) -> list[dict[str, Any]]:
    matched = []
    for row in candidates:
        candidate = row.get("box") or []
        if len(candidate) < 4:
            continue
        if (float(candidate[0]) <= float(box[2])
                and float(candidate[2]) >= float(box[0])
                and float(candidate[1]) <= float(box[3])
                and float(candidate[3]) >= float(box[1])):
            matched.append(dict(row))
    return matched


def _stage_verdict(
    *, quality_ok: bool, inside_roi: bool, inside_valid: bool,
    seed_pixels: int, support_pixels: int, candidates: int,
    selected: bool, prior_allowed: bool,
    rejected: Mapping[str, int], residual_at_target: float,
    threshold_at_target: float,
) -> str:
    if not quality_ok:
        return "FRAME_QUALITY"
    if not inside_roi or not inside_valid:
        return "ROI_OR_VALID"
    if seed_pixels <= 0 and support_pixels <= 0:
        if residual_at_target < threshold_at_target:
            return "THRESHOLD_BELOW_NOISE_FLOOR"
        return "RESIDUAL_BELOW_THRESHOLD"
    if support_pixels > 0 and candidates <= 0:
        return "SIZE_FILTERED" if rejected.get("too_small") else "MORPHOLOGY_EMPTY"
    if candidates > 0 and not selected:
        return "REFERENCE_NOT_SELECTED"
    if candidates > 0 and not prior_allowed:
        return "PRIOR_NOT_ALLOWED"
    return "SURVIVED"


def _target_diagnostics(
    context: BankPriorContext, frame: np.ndarray, box: Sequence[float],
    roi: np.ndarray, matcher: Mapping[str, Any],
) -> dict[str, Any]:
    """在注入位置局部量化"残差 / 阈值 / 形态学"三件事。"""
    height, width = frame.shape[:2]
    left, top, right, bottom = (int(round(float(value))) for value in box)
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    # 评分走与运行时相同的补偿路径，拿到补偿后参考。
    try:
        score = score_profile(
            frame, context.reference, context.valid, roi,
            profile_id=context.profile_id, config=matcher,
        )
        compensated = score.compensated_reference
        if compensated is None:
            compensated = context.reference
    except BankError as exc:
        return {"error": str(exc)[:160], "available": False}
    signature, luminance = residual_maps(compensated, frame)
    support = analyze_residual_support(
        frame, compensated, context.valid,
        np.where(context.valid > 0, 255, 0).astype(np.uint8),
        context.thresholds(frame.shape[:2]),
    )
    thresholds = context.thresholds(frame.shape[:2])
    region = np.s_[top:bottom, left:right]
    return {
        "available": True,
        "luminance_median_at_target": round(float(np.median(luminance[region])), 4),
        "luminance_max_at_target": round(float(luminance[region].max()), 4),
        "signature_max_at_target": round(float(signature[region].max()), 4),
        "support_seed_luminance_threshold": round(
            float(np.median(thresholds["seed_luminance_threshold"][region])), 4,
        ),
        "support_seed_signature_threshold": round(
            float(np.median(thresholds["seed_signature_threshold"][region])), 4,
        ),
        "seed_pixels_at_target": int(np.count_nonzero(
            support.seed[region]
        )),
        "support_pixels_at_target": int(np.count_nonzero(
            support.support[region]
        )),
        "raw_before_compensation_luminance_median": round(
            float(np.median(residual_maps(context.reference, frame)[1][region])), 4,
        ),
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    bank = load_bank(
        args.bank_root, args.bank_id, args.version,
        require_calibration=True,
    )
    geometry = _load_geometry(args.bank_root, args.bank_id, args.version,
                              args.geometry)
    native_w, native_h = (int(value) for value in bank.reference_size)
    online_w, online_h = _parse_size(args.online_size)
    matcher = dict(bank.matcher)
    media = _collect_media(Path(args.media))
    if not media:
        raise SystemExit(f"没有可读素材: {args.media}")
    source = media[0]
    frame = _read_first_frame(source, int(args.frame_index))
    if frame is None:
        raise SystemExit(f"{source} 没有第 {args.frame_index} 帧")
    native_frame = cv2.resize(
        frame, (native_w, native_h), interpolation=cv2.INTER_AREA,
    )
    if frame.shape[1] != native_w or frame.shape[0] != native_h:
        # 素材本身可能不是原生分辨率：记录以便解释。
        pass
    quality = frame_quality(frame)
    roi_native = roi_mask_from_geometry(geometry, native_w, native_h)
    rows, cols = np.nonzero(roi_native > 0)
    if rows.size == 0:
        raise SystemExit("ROI 内没有像素")
    if args.native_position:
        px, _, py = str(args.native_position).partition(",")
        center = (int(px), int(py))
    else:
        center = (int(np.median(cols)), int(np.median(rows)))

    descriptors = {pid: bank.load_descriptor(pid) for pid in bank.ids()}
    scales = global_descriptor_scale(list(descriptors.values()))
    online_scale = online_w / max(native_w, 1)

    comparisons: list[dict[str, Any]] = []
    visuals: dict[str, Any] = {}
    for native_side in (int(value) for value in str(args.native_size_px).split(",")):
        for canvas, size, mode in (
            ("native", (native_w, native_h), "A_native_prior_adapter"),
            ("online", (online_w, online_h), "B_online_analysis_path"),
        ):
            contexts = {
                pid: build_prior_context_at(bank, pid, size)
                for pid in bank.ids()
            }
            roi = roi_mask_from_geometry(geometry, size[0], size[1])
            scale = size[0] / max(native_w, 1)
            side = max(2, int(round(native_side * scale)))
            # 同一物理位置：原生坐标映射到该画布。
            detail_center = (
                int(round(center[0] * size[0] / native_w)),
                int(round(center[1] * size[1] / native_h)),
            )
            work = (
                cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                if (size[0], size[1]) != (native_w, native_h) else native_frame
            )
            injected, truth = _inject(work, detail_center, side, roi)
            truth["requested_native_size_px"] = int(native_side)
            truth["scaled_size_px"] = round(float(native_side * scale), 3)
            truth["sub_pixel_at_canvas"] = bool(scale * native_side < 2.0)
            truth["native_side"] = int(native_side)
            truth["canvas"] = canvas
            truth["canvas_size"] = [int(size[0]), int(size[1])]
            truth["native_center"] = [int(center[0]), int(center[1])]
            truth["injected_side_on_canvas"] = int(side)

            baseline_tick = _evaluate_all(
                contexts, work, roi, matcher, truth, valid_key="baseline",
            )
            injected_tick = _evaluate_all(
                contexts, injected, roi, matcher, truth, valid_key="injected",
            )
            best = _best_reference(injected_tick, truth["box"])
            search = _search_candidates(
                injected, contexts, size, roi, matcher, descriptors, scales,
            )
            effective = (injected_tick.get("profiles") or {}).get(
                best["profile_id"], {},
            ) if best else {}
            # C：真实共享 Selector 的决策（不是"最佳参考"）。为了让状态机
            # 稳定，先用同一批候选吃若干帧预热，再看注入帧的决策。
            runtime = _runtime_decision(
                contexts, work, injected, roi, matcher, size, geometry,
                bank_id=args.bank_id, version=args.version,
            )
            selected_matches = [
                pid for pid, payload in (injected_tick.get("profiles") or {}).items()
                if int(payload.get("matched_candidate_count") or 0) > 0
            ]
            selected_by_runtime = (
                runtime.get("effective_profile_id") in selected_matches
                if runtime.get("effective_profile_id") else False
            )
            verdict = _stage_verdict(
                quality_ok=quality.usable,
                inside_roi=bool(truth["inside_roi"]),
                inside_valid=bool(best["inside_valid"]) if best else False,
                seed_pixels=int(effective.get("seed_pixels_at_target") or 0),
                support_pixels=int(effective.get("support_pixels_at_target") or 0),
                candidates=int(effective.get("candidate_count") or 0),
                selected=bool(selected_by_runtime),
                prior_allowed=bool(runtime.get("prior_allowed")),
                rejected=effective.get("rejected") or {},
                residual_at_target=float(
                    effective.get("luminance_median_at_target") or 0.0
                ),
                threshold_at_target=float(
                    effective.get("support_seed_luminance_threshold") or 0.0
                ),
            )
            comparisons.append({
                "mode": mode,
                "canvas": canvas,
                "canvas_size": [int(size[0]), int(size[1])],
                "native_size_px": int(native_side),
                "injected_side_on_canvas": int(side),
                "online_scale": round(float(online_scale), 5),
                "injection": truth,
                "baseline_candidates_any": baseline_tick["candidate_count"],
                "injected_candidates_any": injected_tick["candidate_count"],
                "search_candidates": search,
                "best_reference": best,
                "runtime_selector": runtime,
                "selected_matches_at_target": selected_matches,
                "loss_stage": verdict,
                "profiles": injected_tick.get("profiles"),
            })
            if args.save_visuals:
                visuals[f"{mode}_{native_side}"] = {
                    "work": work, "injected": injected, "truth": truth,
                    "best_reference_id": best["profile_id"] if best else None,
                }
    report = {
        "bank": {
            "bank_root": str(args.bank_root), "bank_id": args.bank_id,
            "version": args.version, "profiles": list(bank.ids()),
            "reference_size": [int(native_w), int(native_h)],
        },
        "source": {
            "media": str(source), "frame_index": int(args.frame_index),
            "frame_quality": quality.as_dict(),
            "native_center": [int(center[0]), int(center[1])],
        },
        "comparisons": comparisons,
        "loss_stage_vocabulary": list(STAGES),
        "note": ("诊断不修改任何阈值，也不把'最佳参考'当成实际 Selector；"
                 "所有候选都来自真实共享 adapter。"),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "small_target_diagnosis.json", report)
    if args.save_visuals:
        for name, payload in visuals.items():
            _write_visual(output / f"{name}.png", payload)
    return report


def _runtime_decision(
    contexts: Mapping[str, BankPriorContext], baseline: np.ndarray,
    injected: np.ndarray, roi: np.ndarray, matcher: Mapping[str, Any],
    size: tuple[int, int], geometry: Mapping[str, Any], *,
    bank_id: str, version: str, warmup: int = 6,
) -> dict[str, Any]:
    """用真实共享 Selector 跑一次（含预热），返回注入帧的决策。

    D 对照是"离线最佳参考"，只是诊断上界；C 必须是真实 Selector 的选择，
    否则不能回答"是哪一步丢的"。
    """
    replay_config = dict(matcher.get("selection", {}))
    nominal = 2.0
    replay_config["tick_interval_seconds"] = nominal
    replay_config["result_validity_seconds"] = max(nominal, 4.0)
    replay_config["max_observation_gap_seconds"] = max(4.0, 2.0 * nominal)
    replay_config.setdefault("join_gap_seconds", 300.0)
    selector = ProfileSelector(
        bank_id=bank_id, bank_version=version,
        view_id=str(geometry.get("view_id", "view_0")),
        profile_ids=list(contexts), config=replay_config,
    )

    def step(frame: np.ndarray, timestamp: float) -> dict[str, Any]:
        current_id = selector.selected_profile_id
        current = None
        matches: list[CandidateMatch] = []
        envelope_map = {
            pid: _envelope_for(context, matcher)
            for pid, context in contexts.items()
        }
        for pid, context in contexts.items():
            evaluation = evaluate_bank_frame(
                context, frame, envelope=envelope_map[pid], config=matcher,
                roi_mask=roi,
            )
            candidate = CandidateMatch(
                pid, float(evaluation.outcome.get("score") or 0.0),
                bool(evaluation.outcome.get("enter_eligible")),
                bool(evaluation.outcome.get("hold_eligible")),
                verified=bool(
                    evaluation.outcome.get("enter_eligible")
                    or evaluation.outcome.get("hold_eligible")
                ),
            )
            matches.append(candidate)
            if pid == current_id:
                current = candidate
        decision = selector.observe(
            timestamp=timestamp, current=current, candidates=matches,
            tested_profile_ids=list(contexts), observable=True,
            tick_interval_seconds=nominal,
        )
        if decision.commit_requested:
            selector.commit(
                profile_id=decision.commit_profile_id, timestamp=timestamp,
            )
        return {
            "status": decision.status, "reason": decision.reason,
            "prior_allowed": bool(decision.prior_allowed),
            "selected_profile_id": decision.selected_profile_id,
            "candidate_profile_id": decision.candidate_profile_id,
        }

    for index in range(max(1, int(warmup))):
        step(baseline, float(index) * nominal)
    state = step(injected, float(max(1, int(warmup))) * nominal)
    return {
        "effective_profile_id": state["selected_profile_id"],
        "prior_allowed": state["prior_allowed"],
        "status": state["status"],
        "reason": state["reason"],
        "candidate_profile_id": state["candidate_profile_id"],
        "warmup_frames": int(warmup),
    }


def _evaluate_all(
    contexts: Mapping[str, BankPriorContext], frame: np.ndarray,
    roi: np.ndarray, matcher: Mapping[str, Any], truth: Mapping[str, Any],
    *, valid_key: str,
) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    total = 0
    for pid, context in contexts.items():
        evaluation = evaluate_bank_frame(
            context, frame, envelope=_envelope_for(context, matcher),
            config=matcher, roi_mask=roi,
        )
        matched = _match_position(evaluation.candidates, truth["box"])
        total += len(evaluation.candidates)
        detail = _target_diagnostics(context, frame, truth["box"], roi, matcher)
        profiles[pid] = {
            "matched_candidates_at_target": matched,
            "candidate_count": len(evaluation.candidates),
            "matched_candidate_count": len(matched),
            "availability_fraction": round(evaluation.availability_fraction, 5),
            "enter_eligible": bool(evaluation.outcome.get("enter_eligible")),
            "hold_eligible": bool(evaluation.outcome.get("hold_eligible")),
            "reason": evaluation.outcome.get("reason"),
            "inside_valid": bool(np.count_nonzero(
                context.valid[
                    truth["box"][1]:truth["box"][3],
                    truth["box"][0]:truth["box"][2],
                ] > 0
            )),
            "rejected": evaluation.mask_diagnostics.get(
                "candidate_rejections", {},
            ),
            **detail,
        }
    return {"candidate_count": total, "profiles": profiles}


def _envelope_for(context: BankPriorContext, matcher: Mapping[str, Any]) -> Any:
    """从冻结的 noise.npz 与 matcher 里取该候选的包络（只读）。"""
    from rtsp_annotator.ground_litter_profile_match import coerce_envelope

    profile = matcher.get("profiles", {}).get(context.profile_id) or {}
    return coerce_envelope(profile.get("envelope"))


def _best_reference(tick: Mapping[str, Any], box: Sequence[float]) -> dict[str, Any] | None:
    rows = []
    for pid, payload in (tick.get("profiles") or {}).items():
        rows.append({
            "profile_id": pid,
            "matched": int(payload.get("matched_candidate_count") or 0),
            "inside_valid": bool(payload.get("inside_valid")),
            "enter": bool(payload.get("enter_eligible")),
            "support_pixels": int(payload.get("support_pixels_at_target") or 0),
            "seed_pixels": int(payload.get("seed_pixels_at_target") or 0),
            "candidate_count": int(payload.get("candidate_count") or 0),
            "luminance_median_at_target": payload.get("luminance_median_at_target"),
            "luminance_max_at_target": payload.get("luminance_max_at_target"),
            "support_seed_luminance_threshold":
                payload.get("support_seed_luminance_threshold"),
            "rejected": payload.get("rejected") or {},
        })
    if not rows:
        return None
    rows.sort(key=lambda row: (
        -int(row["matched"]), -int(row["support_pixels"]),
        -int(row["seed_pixels"]), row["profile_id"],
    ))
    best = dict(rows[0])
    best["all_profiles"] = [
        {key: row[key] for key in (
            "profile_id", "matched", "inside_valid", "support_pixels",
            "seed_pixels", "candidate_count",
        )} for row in rows
    ]
    return best


def _search_candidates(
    frame: np.ndarray, contexts: Mapping[str, BankPriorContext],
    size: tuple[int, int], roi: np.ndarray, matcher: Mapping[str, Any],
    descriptors: Mapping[str, Mapping[str, np.ndarray]],
    scales: Mapping[str, float],
) -> list[dict[str, Any]]:
    """参考选择阶段：粗距离排序 + 逐参考是否 enter（不含 Selector 状态）。"""
    preview = preview_image(frame, width=960)
    descriptor = extract_grid_descriptor(
        preview,
        roi_mask_from_geometry(
            {"roi": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            preview.shape[1], preview.shape[0],
        ),
    )
    ordered = sorted(
        (
            (pid, descriptor_coarse_distance(descriptor, payload, scale=scales))
            for pid, payload in descriptors.items()
        ),
        key=lambda item: (item[1], item[0]),
    )
    rows = []
    for pid, distance in ordered:
        try:
            score = score_profile(
                frame, contexts[pid].reference, contexts[pid].valid, roi,
                profile_id=pid, config=matcher,
            )
        except BankError as exc:
            rows.append({"profile_id": pid, "coarse_distance": round(distance, 5),
                         "error": str(exc)[:120]})
            continue
        outcome = evaluate_match(score, _envelope_for(contexts[pid], matcher), matcher)
        rows.append({
            "profile_id": pid, "coarse_distance": round(float(distance), 5),
            "enter_eligible": bool(outcome.enter_eligible),
            "hold_eligible": bool(outcome.hold_eligible),
            "score": round(float(score.score), 5),
        })
    return rows


def _write_visual(path: Path, payload: Mapping[str, Any]) -> None:
    frame = np.asarray(payload["work"]).copy()
    truth = payload["truth"]
    box = truth["box"]
    cv2.rectangle(
        frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
        (0, 0, 255), 2,
    )
    label = str(payload.get("best_reference_id") or "-")
    cv2.putText(frame, label, (max(0, int(box[0])), max(12, int(box[1]) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args() if argv is None else parse_args_from(argv)
    report = diagnose(args)
    print(json.dumps({
        "comparisons": [
            {
                "mode": row["mode"], "native_size_px": row["native_size_px"],
                "canvas_size": row["canvas_size"],
                "injected_side_on_canvas": row["injected_side_on_canvas"],
                "loss_stage": row["loss_stage"],
                "baseline_candidates_any": row["baseline_candidates_any"],
                "injected_candidates_any": row["injected_candidates_any"],
                "best_reference": {
                    "profile_id": row["best_reference"]["profile_id"],
                    "matched": row["best_reference"]["matched"],
                    "inside_valid": row["best_reference"]["inside_valid"],
                    "support_pixels": row["best_reference"]["support_pixels"],
                } if row["best_reference"] else None,
            }
            for row in report["comparisons"]
        ],
        "output": str(Path(args.output) / "small_target_diagnosis.json"),
    }, ensure_ascii=False, indent=1))
    return 0


def parse_args_from(argv: Sequence[str]) -> argparse.Namespace:
    original = sys.argv
    try:
        sys.argv = ["diagnose", *argv]
        return parse_args()
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
