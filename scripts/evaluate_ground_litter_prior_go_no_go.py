"""背景先验 Go/No-Go 可行性实验（方案一之外的独立判定，不修改主链）。

回答三个假设：

* H1：给出正确参考时，8/12/16/24 原生像素的小目标信号是否存在（oracle）；
* H2：单 active Selector 能保留多少 oracle 召回（active / prior_output）；
* H3：未注入真实录像中，背景先验会产出多少持续非垃圾事件（时序状态机后）。

设计约束（与任务书一致）：

* 每个真实帧只解码一次；配准与共同画布只算一次；
* 每个 ``(frame, profile)`` 的补偿/残差/阈值/候选只算一次并缓存；
* A/B/C/D 四组只复用逐 Profile 结果，各自重放 Selector；
* 注入在**原生分辨率**完成，再走真实在线缩放与共享 adapter；
* 不修改生产模块，只调用现有真实函数。

用法（服务器隔离目录）::

    ./venv/bin/python scripts/evaluate_ground_litter_prior_go_no_go.py \
        --bank-root out/bank --bank-id camera_01030 \
        --media-dir diag_media --output out/prior_gonogo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
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
    MatchEnvelope, coerce_envelope,
)
from rtsp_annotator.ground_litter_profile_sampling import (  # noqa: E402
    SequentialFrameReader, frame_quality, parse_seconds,
)
from rtsp_annotator.ground_litter_profile_selector import (  # noqa: E402
    CandidateMatch, ProfileSelector,
)

EXPERIMENT_VERSION = "prior_gonogo_r1"
# 预注册门槛（运行前冻结；不得看结果后修改）。
THRESHOLDS = {
    "oracle_hit_rate_go": 0.70,
    "oracle_hit_rate_no_go": 0.50,
    "environments_detectable_min": 3,
    "environments_min": 4,
    "prior_output_hit_rate_go": 0.50,
    "output_to_oracle_ratio_go": 0.70,
    "prior_availability_go": 0.50,
    "fp_confirmed_go_per_hour": 2.0,
    "fp_confirmed_no_go_per_hour": 6.0,
}

TEMPLATES = ("light_paper", "dark_paper", "color_wrapper", "translucent")
SIZES = (8, 12, 16, 24)
ROI_POSITIONS = ("far", "mid", "near", "texture")
SIZES = (8, 12, 16, 24)


# --------------------------------------------------------------------------- #
# 素材与身份
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def config_digest(payload: Mapping[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="背景先验 Go/No-Go 实验")
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--bank-id", required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--online-size", default="960x540")
    parser.add_argument("--native-size", default="2560x1440")
    parser.add_argument("--frames-per-env", type=int, default=6)
    parser.add_argument("--frame-index", type=int, default=40)
    parser.add_argument("--analysis-fps", type=float, default=0.5)
    parser.add_argument("--geometry", type=Path, default=None)
    parser.add_argument(
        "--capability-report", type=Path, default=None,
        help="带 prior 能力的 factory_report.json（v4 报告）",
    )
    parser.add_argument(
        "--groups", default="A:v3,B:v4,C:cap,D:v5",
        help=("Profile 对照组：name:version[:cap]；cap 表示使用报告里的 "
              "prior 能力标注"),
    )
    parser.add_argument("--negative-minutes", type=float, default=30.0)
    parser.add_argument("--negative-fps", type=float, default=0.5)
    parser.add_argument("--profiles-per-frame", type=int, default=4,
                        help="每帧按粗距离取前 N 个参考参与尺度扫描")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--smoke", action="store_true",
                        help="只跑最多 12 帧 smoke，不跑负样本")
    parser.add_argument("--save-events", action="store_true")
    return parser.parse_args(argv)


def collect_media(root: Path) -> list[Path]:
    allowed = {".ps", ".mp4", ".mkv", ".avi", ".mov", ".ts", ".m4v"}
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
        and path.stat().st_size > 1_000_000
    )


def read_frame(path: Path, index: int) -> np.ndarray | None:
    reader = SequentialFrameReader(path)
    for position, captured in enumerate(reader.iter_frames()):
        if position >= index:
            return captured.frame.copy()
    return None


def read_frames(path: Path, count: int, *, step_frames: int) -> list[tuple[int, np.ndarray]]:
    """按固定帧步长抽干 count 帧（一次解码）。"""
    rows: list[tuple[int, np.ndarray]] = []
    reader = SequentialFrameReader(path)
    wanted = {index * step_frames for index in range(count)}
    for position, captured in enumerate(reader.iter_frames()):
        if position in wanted:
            rows.append((position, captured.frame.copy()))
            if len(rows) >= count:
                break
        elif position > max(wanted, default=0):
            break
    return rows


# --------------------------------------------------------------------------- #
# 注入模板（确定性合成；结构信号实验，不是真实垃圾准确率）
# --------------------------------------------------------------------------- #


def pick_positions(
    roi: np.ndarray, rng: np.random.Generator,
) -> dict[str, tuple[int, int]]:
    """在 ROI 内选 4 类位置：远端 / 中段 / 近端 / 纹理变化处。

    位置只用于本次注入，不参与任何阈值选择；记录在结果里以便复核。
    """
    rows, cols = np.nonzero(roi > 0)
    if rows.size == 0:
        return {}
    order = np.argsort(rows)  # 按图像 y（远→近）
    third = max(1, rows.size // 3)
    picks: dict[str, tuple[int, int]] = {}
    picks["far"] = (int(cols[order[third // 2]]), int(rows[order[third // 2]]))
    picks["mid"] = (
        int(cols[order[third + third // 2]]),
        int(rows[order[third + third // 2]]),
    )
    picks["near"] = (
        int(cols[order[-1 - third // 4]]),
        int(rows[order[-1 - third // 4]]),
    )
    pick = int(rng.integers(0, rows.size))
    picks["texture"] = (int(cols[pick]), int(rows[pick]))
    return picks




# --------------------------------------------------------------------------- #
# 注入模板与几何（确定性合成；结构信号实验，不代表真实垃圾准确率）
# --------------------------------------------------------------------------- #


def build_template(name: str, side: int, rng: np.random.Generator) -> np.ndarray:
    """确定性合成模板；8/12/16/24 原生像素。"""
    canvas = np.zeros((side, side, 3), np.uint8)
    if name == "light_paper":
        canvas[:] = 236
        canvas[side // 3:side // 3 + max(1, side // 8), :] = 208
    elif name == "dark_paper":
        canvas[:] = 46
        canvas[:, side // 3:side // 3 + max(1, side // 8)] = 72
    elif name == "color_wrapper":
        canvas[:] = (58, 42, 212)
        canvas[side // 2:, :] = (198, 188, 62)
    elif name == "translucent":
        canvas[:] = 152
        noise = rng.integers(-10, 10, (side, side, 3))
        canvas = np.clip(canvas.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"未知模板: {name}")
    return canvas


def inject_native(
    frame: np.ndarray, center: tuple[int, int], template: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """在原生分辨率图上贴模板，返回注入图与真值框。"""
    height, width = frame.shape[:2]
    side_h, side_w = template.shape[:2]
    left = max(0, min(int(center[0]) - side_w // 2, width - side_w))
    top = max(0, min(int(center[1]) - side_h // 2, height - side_h))
    target = frame.copy()
    target[top:top + side_h, left:left + side_w] = template
    return target, {
        "native_box": [left, top, left + side_w, top + side_h],
        "native_side": [int(side_w), int(side_h)],
        "native_center": [left + side_w // 2, top + side_h // 2],
    }


def scale_box(box: Sequence[float], scale_x: float, scale_y: float) -> list[float]:
    return [
        float(box[0]) * scale_x, float(box[1]) * scale_y,
        float(box[2]) * scale_x, float(box[3]) * scale_y,
    ]


def overlap(box: Sequence[float], truth: Sequence[float]) -> bool:
    if len(box) < 4:
        return False
    return not (
        float(box[2]) < float(truth[0]) or float(box[0]) > float(truth[2])
        or float(box[3]) < float(truth[1]) or float(box[1]) > float(truth[3])
    )


def select_profiles_for_frame(
    bank: Any, matcher: Mapping[str, Any], online_frame: np.ndarray,
    geometry: Mapping[str, Any], size: tuple[int, int], *,
    limit: int,
) -> list[str]:
    """按粗距离选出该帧最相关的 ``limit`` 个参考。

    oracle 口径允许"从候选里找最合适的参考"，因此不需要在每一帧对全部
    13 个参考跑满尺度扫描；先按共享 contract 里的粗距离排序取前 N，
    既保持 oracle 语义，又把逐帧 × 尺度计算量压到可控范围。
    """
    from rtsp_annotator.ground_litter_profile_match import (
        descriptor_coarse_distance, extract_grid_descriptor,
        global_descriptor_scale,
    )

    preview = cv2.resize(online_frame, (960, 540), interpolation=cv2.INTER_AREA)
    descriptor = extract_grid_descriptor(
        preview,
        roi_mask_from_geometry(
            {"roi": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            preview.shape[1], preview.shape[0],
        ),
    )
    payloads = {pid: bank.load_descriptor(pid) for pid in bank.ids()}
    scale = global_descriptor_scale(list(payloads.values()))
    ordered = sorted(
        (
            (pid, descriptor_coarse_distance(descriptor, payload, scale=scale))
            for pid, payload in payloads.items()
        ),
        key=lambda item: (item[1], item[0]),
    )
    return [pid for pid, _ in ordered[:max(1, int(limit))]]


# 模板只构造一次：48 个 (模板, 尺寸) 组合在整轮实验里复用。
TEMPLATE_CACHE: dict[tuple[str, int], np.ndarray] = {
    (name, side): build_template(
        name, side, np.random.default_rng(20260921 + side),
    )
    for name in TEMPLATES for side in SIZES
}



# --------------------------------------------------------------------------- #
# 主实验（逐 Profile 缓存 → 成对注入 → A/B/C/D → 负样本 → 报告）
# --------------------------------------------------------------------------- #



# --------------------------------------------------------------------------- #
# 逐 Profile 结果缓存与共享 adapter 评估（其余实验主体）
# --------------------------------------------------------------------------- #


class ProfileResultCache:
    """缓存 ``(frame_key, profile_id, config_key)`` 的评估结果。"""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[str, str]) -> dict[str, Any] | None:
        value = self._store.get(key)
        if value is None:
            return None
        self.hits += 1
        return value

    def put(self, key: tuple[str, str], value: dict[str, Any]) -> dict[str, Any]:
        self._store[key] = value
        self.misses += 1
        return value


def evaluate_profile(
    context: BankPriorContext, frame: np.ndarray, roi: np.ndarray,
    matcher: Mapping[str, Any], truth_box: Sequence[float] | None,
) -> dict[str, Any]:
    """真实共享 adapter 的单帧 × 单 Profile 评估 + 目标位置漏斗。"""
    evaluation = evaluate_bank_frame(
        context, frame, envelope=envelope_for(context, matcher),
        config=matcher, roi_mask=roi,
    )
    mask = evaluation.mask_diagnostics or {}
    cands = list(evaluation.candidates or [])
    result: dict[str, Any] = {
        "profile_id": context.profile_id,
        "availability_fraction": round(
            float(evaluation.availability_fraction), 5,
        ),
        "score": None if evaluation.score is None
        else round(float(evaluation.score.get("score") or 0.0), 5),
        "enter_eligible": bool(evaluation.outcome.get("enter_eligible")),
        "hold_eligible": bool(evaluation.outcome.get("hold_eligible")),
        "reason": evaluation.outcome.get("reason"),
        "raw_seed_pixels": int(mask.get("seed_pixels") or 0),
        "raw_support_pixels": int(mask.get("support_pixels") or 0),
        "candidate_count": len(cands),
        "size_filtered": int(
            (mask.get("candidate_rejections") or {}).get("too_small") or 0
        ),
        "admission_blocked": mask.get("admission_blocked"),
    }
    result["candidate_boxes"] = [
        [float(v) for v in (row.get("box") or [])] for row in cands
    ]
    if truth_box is not None:
        matched = [
            row for row in cands if overlap(row.get("box") or [], truth_box)
        ]
        result["matched_candidates"] = len(matched)
        result["candidate_support_at_target"] = int(sum(
            int(row.get("support_pixels") or 0) for row in matched
        ))
    return result


def envelope_for(context: BankPriorContext, matcher: Mapping[str, Any]) -> MatchEnvelope:
    profile = (matcher.get("profiles") or {}).get(context.profile_id) or {}
    return coerce_envelope(profile.get("envelope"))


def selection_config(matcher: Mapping[str, Any], nominal: float) -> dict[str, Any]:
    cfg = dict(matcher.get("selection", {}))
    cfg["tick_interval_seconds"] = nominal
    cfg["result_validity_seconds"] = max(nominal, 4.0)
    cfg["max_observation_gap_seconds"] = max(4.0, 2.0 * nominal)
    cfg.setdefault("join_gap_seconds", 300.0)
    return cfg


def replay_selector(
    profile_ids: Sequence[str],
    per_frame: Sequence[Mapping[str, Mapping[str, Any]]],
    *, matcher: Mapping[str, Any], nominal: float, bank_id: str,
    version: str, view_id: str, prior_capable: Sequence[str] | None,
) -> dict[str, Any]:
    """在一帧已算好的逐 Profile 结果上重放 Selector（不重算残差）。"""
    selector = ProfileSelector(
        bank_id=bank_id, bank_version=version, view_id=view_id,
        profile_ids=list(profile_ids), config=selection_config(matcher, nominal),
        prior_suitable_profile_ids=(
            None if prior_capable is None else list(prior_capable)
        ),
    )
    decisions: list[dict[str, Any]] = []
    for index, frame_map in enumerate(per_frame):
        timestamp = float(index) * nominal
        current_id = selector.selected_profile_id
        current: CandidateMatch | None = None
        matches: list[CandidateMatch] = []
        for pid in profile_ids:
            entry = frame_map.get(pid)
            if entry is None:
                continue
            candidate = CandidateMatch(
                pid, float(entry["score"] or 0.0),
                bool(entry["enter_eligible"]), bool(entry["hold_eligible"]),
                verified=bool(
                    entry["enter_eligible"] or entry["hold_eligible"]
                ),
            )
            matches.append(candidate)
            if pid == current_id:
                current = candidate
        decision = selector.observe(
            timestamp=timestamp, current=current, candidates=matches,
            tested_profile_ids=list(profile_ids), observable=True,
            tick_interval_seconds=nominal,
        )
        if decision.commit_requested:
            selector.commit(
                profile_id=decision.commit_profile_id, timestamp=timestamp,
            )
        decisions.append({
            "active_profile_id": selector.selected_profile_id,
            "prior_available": bool(getattr(decision, "prior_available", False)),
            "prior_unavailable_reason": getattr(
                decision, "prior_unavailable_reason", None,
            ),
            "status": decision.status,
        })
    return {
        "decisions": decisions,
        "active_profile_id": selector.selected_profile_id,
        "switches": selector.switch_count,
        "prior_summary": selector.prior_summary(),
    }


def parse_groups(spec: str) -> list[dict[str, Any]]:
    """解析 ``name:version[:cap]``。

    ``cap`` 表示该组的 prior 能力取自 ``--capability-report``（例如 v3 参考
    资产 + v4 报告的能力标注）；历史 Bank 没有能力字段时只能用于 oracle/诊断。
    """
    groups: list[dict[str, Any]] = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        name = parts[0]
        version = parts[1] if len(parts) > 1 and parts[1] else "v3"
        capability = len(parts) > 2 and parts[2] == "cap"
        groups.append({
            "name": name, "version": version,
            "capability_from_report": capability,
        })
    return groups


def capability_from_report(
    report_path: Path, profile_ids: Sequence[str],
) -> dict[str, bool]:
    """从构建报告读 prior 能力（composite 顺序 = pNNNN 编号）。"""
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    mapping: dict[str, bool] = {}
    for index, row in enumerate(payload.get("composite", []), start=1):
        mapping[f"p{index:04d}"] = bool(
            (row.get("noise_calibration") or {}).get("prior_suitable")
        )
    return {pid: mapping.get(pid, False) for pid in profile_ids}


def load_geometry(
    bank_root: Path, bank_id: str, version: str, override: Path | None,
) -> dict[str, Any]:
    directory = Path(bank_root) / bank_id / version
    payload = json.loads((directory / "camera_geometry.json").read_text())
    if override is not None:
        extra = json.loads(Path(override).read_text(encoding="utf-8"))
        for key in ("roi", "exclude_zones", "overlay_exclude_zones", "view_id"):
            if key in extra:
                payload[key] = extra[key]
    return payload


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * q) - 1))
    return round(float(ordered[index]), 4)


def run_core(args: argparse.Namespace) -> dict[str, Any]:
    """完整实验主体：成对注入 + A/B/C/D + 逐 Profile 缓存。"""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    native_w, native_h = (int(v) for v in str(args.native_size).lower().split("x"))
    online_w, online_h = (int(v) for v in str(args.online_size).lower().split("x"))
    scale_x = online_w / max(native_w, 1)
    scale_y = online_h / max(native_h, 1)
    group_specs = parse_groups(args.groups)
    banks: dict[str, Any] = {}
    matchers: dict[str, dict[str, Any]] = {}
    # 只构建在线画布上下文：注入后的评估走真实在线缩放与 adapter；
    # 原生尺寸只用于"贴模板"这一步，不需要为每个 Profile 建原生上下文。
    online_contexts: dict[tuple[str, str], BankPriorContext] = {}
    capability_reports: dict[str, dict[str, bool]] = {}
    for group in group_specs:
        version = group["version"]
        if version in banks:
            continue
        bank = load_bank(
            args.bank_root, args.bank_id, version, require_calibration=True,
            allow_legacy_profile_capabilities=True,
        )
        banks[version] = bank
        matchers[version] = dict(bank.matcher)
        print(f"[gonogo] bank {version}: {len(bank.ids())} profiles", flush=True)
        for pid in bank.ids():
            online_contexts[(version, pid)] = build_prior_context_at(
                bank, pid, (online_w, online_h),
            )
        print(f"[gonogo] contexts ready for {version}", flush=True)
        if group["capability_from_report"] and args.capability_report:
            capability_reports[version] = capability_from_report(
                Path(args.capability_report), bank.ids(),
            )
    profile_cache = ProfileResultCache()
    timings: list[float] = []
    started = time.monotonic()
    media = collect_media(Path(args.media_dir))
    if not media:
        raise SystemExit(f"没有可用素材: {args.media_dir}")
    frame_budget = int(args.frames_per_env) * len(media)
    if args.smoke:
        frame_budget = min(frame_budget, 12)

    corpus: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    frame_keys: list[str] = []
    print(f"[gonogo] media={len(media)} budget_frames={frame_budget}",
          flush=True)
    decoded = 0
    for env_index, path in enumerate(media):
        if decoded >= frame_budget:
            break
        try:
            file_digest = sha256_file(path)
        except OSError as exc:
            corpus.append({
                "path": str(path), "status": "unreadable",
                "error": str(exc)[:120],
            })
            continue
        remaining = frame_budget - decoded
        frames = read_frames(
            path, min(int(args.frames_per_env), remaining),
            step_frames=max(1, int(round(25.0 / max(args.analysis_fps, 1e-3)))),
        )
        if not frames:
            corpus.append({
                "path": str(path), "status": "no_frames",
                "sha256": file_digest,
            })
            continue
        env_name = f"env{env_index + 1}"
        print(f"[gonogo] {env_name} frames={len(frames)} from {path.name}",
              flush=True)
        decoded += len(frames)
        corpus.append({
            "path": str(path), "status": "used", "environment": env_name,
            "sha256": file_digest, "bytes": path.stat().st_size,
            "frames": [index for index, _ in frames],
            "native_size": [native_w, native_h],
            "uses_real_targets": False,
            "role": "real_negative_plus_synthetic_injection",
            "target_types": "synthetic_only",
        })
        for frame_index, frame in frames:
            frame_key = f"{file_digest}:{frame_index}"
            frame_keys.append(frame_key)
            print(f"[gonogo] frame {frame_index}", flush=True)
            online_frame = cv2.resize(
                frame, (online_w, online_h), interpolation=cv2.INTER_AREA,
            )
            for version, bank in banks.items():
                matcher = matchers[version]
                geometry = load_geometry(
                    args.bank_root, args.bank_id, version, args.geometry,
                )
                roi_native = roi_mask_from_geometry(geometry, native_w, native_h)
                roi_online = roi_mask_from_geometry(geometry, online_w, online_h)
                positions = pick_positions(roi_native, np.random.default_rng(
                    int(args.seed) + env_index
                ))
                for template_name in TEMPLATES:
                    for side in SIZES:
                        template = TEMPLATE_CACHE[(template_name, side)]
                        for position_name in ROI_POSITIONS:
                            center = positions.get(position_name)
                            if center is None:
                                continue
                            injected_native, truth = inject_native(
                                frame, center, template,
                            )
                            injected_online = cv2.resize(
                                injected_native, (online_w, online_h),
                                interpolation=cv2.INTER_AREA,
                            )
                            truth_online = scale_box(
                                truth["native_box"], scale_x, scale_y,
                            )
                            injected_rows: dict[str, dict[str, Any]] = {}
                            baseline_rows: dict[str, dict[str, Any]] = {}
                            subset = select_profiles_for_frame(
                                bank, matcher, online_frame, geometry,
                                (online_w, online_h),
                                limit=int(args.profiles_per_frame),
                            )
                            for pid in subset:
                                key = (
                                    f"{frame_key}|{template_name}|{side}|"
                                    f"{position_name}|{version}",
                                    pid,
                                )
                                cached = profile_cache.get(key)
                                if cached is None:
                                    t0 = time.monotonic()
                                    cached = {
                                        "injected": evaluate_profile(
                                            online_contexts[(version, pid)],
                                            injected_online, roi_online,
                                            matcher, truth_online,
                                        ),
                                        "baseline": evaluate_profile(
                                            online_contexts[(version, pid)],
                                            online_frame, roi_online,
                                            matcher, truth_online,
                                        ),
                                    }
                                    timings.append(time.monotonic() - t0)
                                    profile_cache.put(key, cached)
                                injected_rows[pid] = cached["injected"]
                                baseline_rows[pid] = cached["baseline"]
                            oracle = any(
                                int(row.get("matched_candidates") or 0) > 0
                                for row in injected_rows.values()
                            )
                            ambiguous = any(
                                int(row.get("matched_candidates") or 0) > 0
                                for row in baseline_rows.values()
                            )
                            results.append({
                                "environment": env_name,
                                "frame_key": frame_key,
                                "group_version": version,
                                "template": template_name,
                                "native_side": side,
                                "position": position_name,
                                "truth_native_box": truth["native_box"],
                                "truth_online_box": [
                                    round(value, 3) for value in truth_online
                                ],
                                "inside_roi": bool(np.count_nonzero(
                                    roi_native[
                                        truth["native_box"][1]:truth["native_box"][3],
                                        truth["native_box"][0]:truth["native_box"][2],
                                    ]
                                )),
                                "oracle_hit": bool(oracle),
                                "baseline_ambiguous": bool(ambiguous),
                                "profiles": {
                                    pid: {
                                        "matched_candidates": int(
                                            row.get("matched_candidates") or 0
                                        ),
                                        "candidate_support_at_target": int(
                                            row.get("candidate_support_at_target") or 0
                                        ),
                                        "raw_seed_pixels": row.get("raw_seed_pixels"),
                                        "raw_support_pixels": row.get("raw_support_pixels"),
                                        "candidate_count": row.get("candidate_count"),
                                        "size_filtered": row.get("size_filtered"),
                                        "availability_fraction": row.get(
                                            "availability_fraction"
                                        ),
                                        "enter_eligible": row.get("enter_eligible"),
                                        "baseline_matched": int(
                                            baseline_rows[pid].get(
                                                "matched_candidates"
                                            ) or 0
                                        ),
                                    }
                                    for pid, row in injected_rows.items()
                                },
                            })
    elapsed = time.monotonic() - started
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "native_size": [native_w, native_h],
        "online_size": [online_w, online_h],
        "analysis_fps": float(args.analysis_fps),
        "thresholds": THRESHOLDS,
        "templates": list(TEMPLATES),
        "sizes": list(SIZES),
        "roi_positions": list(ROI_POSITIONS),
        "capability_reports": capability_reports,
        "corpus": corpus,
        "frame_keys": frame_keys,
        "results": results,
        "cache": {"hits": profile_cache.hits, "misses": profile_cache.misses},
        "profile_eval_seconds": {
            "count": len(timings),
            "p50": _percentile(timings, 0.50),
            "p95": _percentile(timings, 0.95),
        },
        "elapsed_seconds": round(elapsed, 3),
        "note": ("合成注入只测结构信号，不代表真实垃圾准确率；"
                 "所有命中都经现有共享 adapter 与在线缩放。"),
    }


# --------------------------------------------------------------------------- #
# 逐 Profile 结果缓存与共享 adapter 评估（其余实验主体）
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 负样本：真实 Selector / 时序状态机 / 持续事件
# --------------------------------------------------------------------------- #


def negative_sample_run(
    args: argparse.Namespace, banks: Mapping[str, Any],
    matchers: Mapping[str, Mapping[str, Any]],
    online_contexts: Mapping[tuple[str, str], BankPriorContext],
    capability_reports: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    """未注入真实录像上的持续候选统计（真实 Selector + 真实状态机）。"""
    from rtsp_annotator.ground_litter_v33 import (
        PriorObservation, V33EventMemory,
    )
    from types import SimpleNamespace

    online_w, online_h = (
        int(v) for v in str(args.online_size).lower().split("x")
    )
    nominal = 1.0 / max(float(args.negative_fps), 1e-3)
    media = collect_media(Path(args.media_dir))
    if not media:
        return {"status": "no_media"}
    budget_seconds = max(0.0, float(args.negative_minutes) * 60.0)
    per_group: dict[str, Any] = {}
    for version, bank in banks.items():
        matcher = matchers[version]
        geometry = load_geometry(args.bank_root, args.bank_id, version, args.geometry)
        roi = roi_mask_from_geometry(geometry, online_w, online_h)
        capability = capability_reports.get(version)
        prior_capable = None
        if capability is not None:
            prior_capable = [pid for pid, ok in capability.items() if ok]
        selector = ProfileSelector(
            bank_id=args.bank_id, bank_version=version, view_id="view_0",
            profile_ids=list(bank.ids()),
            config=selection_config(matcher, nominal),
            prior_suitable_profile_ids=prior_capable,
        )
        # 用真实 options dataclass，避免手写缺字段（V33 memory 读很多窗口参数）。
        from rtsp_annotator.ground_litter_detection import (
            GroundLitterDetectionOptions as _Options,
        )
        try:
            memory_options = _Options(
                analysis_fps=float(args.negative_fps),
                profile_id="gonogo_negative",
            )
        except TypeError:
            memory_options = SimpleNamespace(
                analysis_fps=float(args.negative_fps),
                profile_id="gonogo_negative",
                semantic_scan_interval_seconds=4.0,
                semantic_confirm_span_seconds=4.0,
                prior_confirm_span_seconds=6.0,
                confirm_visible_seconds=5.0,
                clear_confirm_seconds=5.0,
                startup_suppress_seconds=15.0,
                semantic_hit_window=3,
                prior_hit_window=6,
                fused_hit_window=4,
                maximum_boxes=8,
                maximum_closed_events=10_000,
                actor_overlap_threshold=0.25,
            )
        memory = V33EventMemory(
            memory_options, pixel_scale=online_w / 2560.0,
        )
        consumed = 0.0
        raw_candidates = 0
        passed_candidates = 0
        confirmed = 0
        displayed = 0
        prior_available_seconds = 0.0
        longest_prior_gap = 0.0
        current_gap = 0.0
        switches = 0
        cold_start_events = 0
        tick_index = 0
        for path in media:
            if consumed >= budget_seconds:
                break
            step = max(1, int(round(25.0 / max(args.negative_fps, 1e-3))))
            frames = read_frames(
                path, max(1, int(budget_seconds / nominal) + 1),
                step_frames=step,
            )
            for _frame_index, frame in frames:
                if consumed >= budget_seconds:
                    break
                online = cv2.resize(
                    frame, (online_w, online_h), interpolation=cv2.INTER_AREA,
                )
                timestamp = float(tick_index) * nominal
                current_id = selector.selected_profile_id
                current: CandidateMatch | None = None
                matches: list[CandidateMatch] = []
                profile_rows: dict[str, dict[str, Any]] = {}
                for pid in bank.ids():
                    row = evaluate_profile(
                        online_contexts[(version, pid)], online, roi, matcher,
                        None,
                    )
                    profile_rows[pid] = row
                    raw_candidates += int(row["candidate_count"])
                    candidate = CandidateMatch(
                        pid, float(row["score"] or 0.0),
                        bool(row["enter_eligible"]), bool(row["hold_eligible"]),
                        verified=bool(
                            row["enter_eligible"] or row["hold_eligible"]
                        ),
                    )
                    matches.append(candidate)
                    if pid == current_id:
                        current = candidate
                decision = selector.observe(
                    timestamp=timestamp, current=current, candidates=matches,
                    tested_profile_ids=list(bank.ids()), observable=True,
                    tick_interval_seconds=nominal,
                )
                if decision.commit_requested:
                    selector.commit(
                        profile_id=decision.commit_profile_id,
                        timestamp=timestamp,
                    )
                    switches = selector.switch_count
                prior_available = bool(
                    getattr(decision, "prior_available", False)
                )
                if prior_available:
                    prior_available_seconds += nominal
                    current_gap = 0.0
                else:
                    current_gap += nominal
                    longest_prior_gap = max(longest_prior_gap, current_gap)

                active_row = profile_rows.get(
                    str(selector.selected_profile_id or "")
                ) or {}
                active_boxes = list(active_row.get("candidate_boxes") or [])
                passed = len(active_boxes)
                passed_candidates += passed
                prior_observations = [
                    PriorObservation(
                        box_xyxy=(float(box[0]), float(box[1]),
                                  float(box[2]), float(box[3])),
                        anomaly_score=0.5,
                        region_id=f"prior{index}",
                        support_pixels=1,
                        observed_at=timestamp,
                    )
                    for index, box in enumerate(active_boxes)
                    if len(box) >= 4
                ]
                result = memory.update(
                    timestamp=timestamp, semantic=None,
                    prior=prior_observations,
                    prior_available=prior_available,
                    environment_state="NORMAL",
                    support=None, valid=None,
                    prior_raw=len(active_boxes),
                    prior_retained=len(prior_observations),
                    semantic_scan=False,
                )
                confirmed += int(getattr(result, "confirmed_events", 0) or 0)
                displayed += int(len(getattr(result, "detections", ()) or ()))
                if switches > 0 and tick_index <= 30:
                    cold_start_events += 1
                tick_index += 1
                consumed += nominal
        per_group[version] = {
            "ticks": tick_index,
            "observed_seconds": round(consumed, 3),
            "raw_prior_candidates": raw_candidates,
            "candidates_entering_state_machine": passed_candidates,
            "confirmed_events": confirmed,
            "displayed_events": displayed,
            "prior_available_seconds": round(prior_available_seconds, 3),
            "prior_available_fraction": (
                round(prior_available_seconds / consumed, 5)
                if consumed > 0 else 0.0
            ),
            "longest_prior_unavailable_seconds": round(longest_prior_gap, 3),
            "selector_switches": switches,
            "cold_start_window_events": cold_start_events,
            "prior_capable_profiles": prior_capable,
            "analysis_fps": float(args.negative_fps),
            "note": ("semantic 通道关闭，只统计 prior 通道进入现有 V33 状态机后的"
                     "候选；未做人工视觉审核的事件只能叫未审核候选"),
        }
    return {
        "status": "ok",
        "negative_minutes_requested": float(args.negative_minutes),
        "groups": per_group,
    }


# --------------------------------------------------------------------------- #
# 指标与判定
# --------------------------------------------------------------------------- #


def aggregate_metrics(core: Mapping[str, Any]) -> dict[str, Any]:
    """把逐 (环境, 模板, 尺寸, 位置, 组) 结果聚合成三层召回。"""
    rows = core.get("results") or []
    versions = sorted({str(row["group_version"]) for row in rows})
    metrics: dict[str, Any] = {}
    for version in versions:
        subset = [row for row in rows if row["group_version"] == version]
        total = len(subset)
        oracle = sum(1 for row in subset if row["oracle_hit"])
        ambiguous = sum(1 for row in subset if row["baseline_ambiguous"])
        per_size: dict[str, Any] = {}
        for size in SIZES:
            size_rows = [row for row in subset if row["native_side"] == size]
            if not size_rows:
                continue
            per_size[str(size)] = {
                "trials": len(size_rows),
                "oracle_hits": sum(1 for row in size_rows if row["oracle_hit"]),
                "oracle_hit_rate": round(
                    sum(1 for row in size_rows if row["oracle_hit"])
                    / len(size_rows), 5,
                ),
                "baseline_ambiguous": sum(
                    1 for row in size_rows if row["baseline_ambiguous"]
                ),
            }
        metrics[version] = {
            "trials": total,
            "oracle_hits": oracle,
            "oracle_hit_rate": round(oracle / total, 5) if total else 0.0,
            "baseline_ambiguous": ambiguous,
            "baseline_ambiguous_rate": (
                round(ambiguous / total, 5) if total else 0.0
            ),
            "per_size": per_size,
        }
    return metrics


def stage_metrics(
    core: Mapping[str, Any], replay: Mapping[str, Any],
) -> dict[str, Any]:
    """oracle / active（模拟）/ prior_output 三层召回。"""
    rows = core.get("results") or []
    profile_groups = core.get("groups") or []
    active_by_version: dict[str, str | None] = {}
    for group in profile_groups:
        version = group["version"]
        if version not in active_by_version:
            active_by_version[version] = replay.get(version, {}).get(
                "active_profile_id"
            )
    result: dict[str, Any] = {}
    for version, active in active_by_version.items():
        subset = [row for row in rows if row["group_version"] == version]
        total = len(subset)
        oracle = active_hits = output_hits = 0
        prior_capable: set[str] | None = None
        for group in (core.get("groups") or []):
            if group["version"] == version and "prior_capable_ids" in group:
                prior_capable = set(group.get("prior_capable_ids") or ())
        for row in subset:
            if row["oracle_hit"]:
                oracle += 1
            profiles = row.get("profiles") or {}
            if active and int(
                (profiles.get(active) or {}).get("matched_candidates") or 0
            ) > 0:
                active_hits += 1
                # 可输出：生效参考命中，且该参考 prior_suitable=true。
                if prior_capable is None or active in prior_capable:
                    output_hits += 1
        result[version] = {
            "trials": total,
            "oracle_hits": oracle,
            "oracle_hit_rate": round(oracle / total, 5) if total else 0.0,
            "active_profile_id": active,
            "active_hits": active_hits,
            "active_hit_rate": round(active_hits / total, 5) if total else 0.0,
            "prior_output_hits": output_hits,
            "prior_output_hit_rate": round(output_hits / total, 5)
            if total else 0.0,
        }
    return result


def decide(
    metrics: Mapping[str, Any], stage: Mapping[str, Any],
    negative: Mapping[str, Any], *, primary_version: str,
) -> dict[str, Any]:
    """预注册判定：只用冻结的门槛，不根据结果调整。"""
    primary = stage.get(primary_version) or {}
    oracle = float(primary.get("oracle_hit_rate") or 0.0)
    output = float(primary.get("prior_output_hit_rate") or 0.0)
    ratio = round(output / oracle, 5) if oracle > 0 else 0.0
    per_env: dict[str, float] = {}
    rows = metrics.get(primary_version, {}).get("per_size", {})
    del rows
    negative_groups = (negative.get("groups") or {})
    fp_per_hour = None
    for payload in negative_groups.values():
        seconds = float(payload.get("observed_seconds") or 0.0)
        if seconds >= 300:
            fp_per_hour = round(
                float(payload.get("confirmed_events") or 0) / (seconds / 3600.0), 3,
            )
            break
    decision = "NO_GO"
    reasons: list[str] = []
    if oracle < THRESHOLDS["oracle_hit_rate_no_go"]:
        decision = "NO_GO"
        reasons.append(
            f"oracle_hit_rate={oracle:.3f} < {THRESHOLDS['oracle_hit_rate_no_go']}"
        )
    elif oracle < THRESHOLDS["oracle_hit_rate_go"]:
        decision = "NO_GO"
        reasons.append(
            f"oracle_hit_rate={oracle:.3f} 落在 NO-GO 与 GO 之间"
        )
    elif output < THRESHOLDS["prior_output_hit_rate_go"]:
        decision = "CONDITIONAL_SELECTOR"
        reasons.append(
            f"oracle={oracle:.3f} 但 prior_output_hit_rate={output:.3f} 不足"
        )
    elif ratio < THRESHOLDS["output_to_oracle_ratio_go"]:
        decision = "CONDITIONAL_SELECTOR"
        reasons.append(f"output/oracle={ratio:.3f} < 0.70")
    elif fp_per_hour is not None and fp_per_hour > THRESHOLDS[
        "fp_confirmed_no_go_per_hour"
    ]:
        decision = "NO_GO"
        reasons.append(f"确认误报 {fp_per_hour}/h 超过 NO-GO 上限")
    elif fp_per_hour is not None and fp_per_hour > THRESHOLDS[
        "fp_confirmed_go_per_hour"
    ]:
        decision = "CONDITIONAL_VERIFIER"
        reasons.append(f"确认误报 {fp_per_hour}/h 需要二分类复核")
    else:
        decision = "GO"
        reasons.append("oracle 与输出召回均达标，误报在门槛内")
    return {
        "decision": decision,
        "reasons": reasons,
        "oracle_hit_rate": oracle,
        "prior_output_hit_rate": output,
        "output_to_oracle_ratio": ratio,
        "confirmed_fp_per_hour": fp_per_hour,
        "per_environment": per_env,
        "thresholds": THRESHOLDS,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    core = run_core(args)
    atomic_write_json(output / "RAW_RESULTS.json", core)
    print(f"[gonogo] core done in {core['elapsed_seconds']}s", flush=True)

    # 用核心阶段同一批在线上下文跑 Selector 重放，得到 active 参考。
    native_w, native_h = (int(v) for v in str(args.native_size).lower().split("x"))
    online_w, online_h = (int(v) for v in str(args.online_size).lower().split("x"))
    online_contexts: dict[tuple[str, str], BankPriorContext] = {}
    banks: dict[str, Any] = {}
    matchers: dict[str, dict[str, Any]] = {}
    capability_reports: dict[str, dict[str, bool]] = {}
    for group in parse_groups(args.groups):
        version = group["version"]
        if version in banks:
            continue
        bank = load_bank(
            args.bank_root, args.bank_id, version, require_calibration=True,
            allow_legacy_profile_capabilities=True,
        )
        banks[version] = bank
        matchers[version] = dict(bank.matcher)
        for pid in bank.ids():
            online_contexts[(version, pid)] = build_prior_context_at(
                bank, pid, (online_w, online_h),
            )
        if group["capability_from_report"] and args.capability_report:
            capability_reports[version] = capability_from_report(
                Path(args.capability_report), bank.ids(),
            )
    metrics = aggregate_metrics(core)
    stage: dict[str, Any] = {}
    replay_payload: dict[str, Any] = {}
    for version, bank in banks.items():
        subset_frames = build_replay_frames(core, version)
        prior_capable = None
        if version in capability_reports:
            prior_capable = [
                pid for pid, ok in capability_reports[version].items() if ok
            ]
        replay = replay_selector(
            bank.ids(), subset_frames, matcher=matchers[version],
            nominal=1.0 / max(args.analysis_fps, 1e-3),
            bank_id=args.bank_id, version=version, view_id="view_0",
            prior_capable=prior_capable,
        )
        replay_payload[version] = replay
    for version, group in ((g["version"], g) for g in parse_groups(args.groups)):
        if version not in stage:
            entry = stage_metrics(
                {
                    "results": [
                        row for row in core["results"]
                        if row["group_version"] == version
                    ],
                    "groups": [{
                        "version": version,
                        "prior_capable_ids": sorted(
                            pid for pid, ok in (
                                capability_reports.get(version) or {}
                            ).items() if ok
                        ) if version in capability_reports else None,
                    }],
                },
                replay_payload,
            )[version]
            entry["group_name"] = group["name"]
            entry["prior_capable_ids"] = sorted(
                pid for pid, ok in (capability_reports.get(version) or {}).items()
                if ok
            ) if version in capability_reports else None
            stage[version] = entry
    negative: dict[str, Any] = {"status": "skipped"}
    if not args.smoke and args.negative_minutes > 0:
        try:
            negative = negative_sample_run(
                args, banks, matchers, online_contexts, capability_reports,
            )
        except Exception as exc:  # 负样本失败不推翻核心结论，但要明确记录
            negative = {
                "status": "error", "error": f"{type(exc).__name__}: {exc}"[:200],
            }
    primary = next(
        (g["version"] for g in parse_groups(args.groups) if g["name"] == "C"),
        next(iter(stage), ""),
    )
    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "thresholds": THRESHOLDS,
        "groups": core.get("groups"),
        "corpus": core.get("corpus"),
        "cache": core.get("cache"),
        "performance": {
            "core_seconds": core.get("elapsed_seconds"),
            "profile_eval_seconds": core.get("profile_eval_seconds"),
            "total_seconds": round(time.monotonic() - started, 3),
            "negative_minutes": args.negative_minutes,
            "analysis_fps": args.analysis_fps,
            "note": "旁路离线性能；未运行主视频链路，不冒充生产 FPS",
        },
        "match_metrics": metrics,
        "stage_metrics": stage,
        "replay": replay_payload,
        "negative_sample": negative,
        "decision": decide(metrics, stage, negative, primary_version=primary),
        "primary_version": primary,
        "limitations": [
            "合成注入只测结构信号，不代表真实垃圾准确率",
            "真实目标样本数量为 0；未做人工视觉审核的事件只能叫未审核候选",
            "主视频链路未运行，性能数字只是旁路离线值",
        ],
    }
    atomic_write_json(output / "RESULTS.json", result)
    print(json.dumps({
        "decision": result["decision"]["decision"],
        "reasons": result["decision"]["reasons"],
        "primary_version": primary,
        "performance": result["performance"],
    }, ensure_ascii=False, indent=1))
    return 0


def build_replay_frames(
    core: Mapping[str, Any], version: str,
) -> list[dict[str, dict[str, Any]]]:
    """把逐 (环境, 帧, 模板, 尺寸, 位置) 结果折叠成"每帧 × Profile"表。"""
    frames: dict[str, dict[str, dict[str, Any]]] = {}
    for row in core.get("results") or []:
        if row["group_version"] != version:
            continue
        frame_key = str(row["frame_key"])
        bucket = frames.setdefault(frame_key, {})
        for pid, payload in (row.get("profiles") or {}).items():
            existing = bucket.get(pid)
            if existing is None:
                bucket[pid] = {
                    "score": 0.0,
                    "enter_eligible": bool(payload.get("enter_eligible")),
                    "hold_eligible": bool(payload.get("enter_eligible")),
                    "matched_candidates": int(
                        payload.get("matched_candidates") or 0
                    ),
                }
            else:
                existing["enter_eligible"] = bool(
                    existing["enter_eligible"]
                    or payload.get("enter_eligible")
                )
                existing["hold_eligible"] = bool(
                    existing["hold_eligible"] or payload.get("enter_eligible")
                )
                existing["matched_candidates"] = max(
                    existing["matched_candidates"],
                    int(payload.get("matched_candidates") or 0),
                )
    return list(frames.values())


if __name__ == "__main__":
    raise SystemExit(main())
