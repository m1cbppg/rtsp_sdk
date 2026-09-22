#!/usr/bin/env python3
"""Measure the V3.3 dual-recall budgets on a real DeepStream host.

This script runs **on the server** (host or a side container that can reach the
API and `nvidia-smi`). It samples one live ``hybrid_v33`` stream and writes a
machine-readable measurement file that ``scripts/report_ground_litter_v33.py``
folds into the performance report.

Design rules:

* Nothing here is simulated. Every number comes from the running API response or
  from ``nvidia-smi``; a missing key is recorded as missing instead of defaulted.
* The GPU-memory verdict needs a *baseline* (the same pipeline without the V3.3
  side branch), because the budget is "V3.3 新增 GPU 显存 ≤1.5GiB". Without
  ``--baseline-vram-mib`` the verdict stays ``not_evaluable`` rather than
  pretending the absolute usage is the increment.
* The API key is read from the environment or a file; it is never written into
  the output, the logs or the command line.

Pure helpers (percentile, sampling, budget evaluation) are unit-tested in
``tests/test_ground_litter_v33_server_measure.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT / "output" / "ground_litter_v33_perf_20260918" / "server_measurement.json"
)

# Architecture doc §10.2. Read the budget mapping as follows:
#   normal_tick_p95   -> last_total_ms on ticks that ran no full ROI scan
#   full_scan_p95     -> last_full_scan_ms on ticks that did run a full scan
#   crop_batch_p95    -> last_crop_batch_ms on normal ticks that ran crops
NORMAL_TICK_P95_MAX_MS = 2000.0
FULL_SCAN_P95_MAX_MS = 1500.0
CROP_BATCH_P95_MAX_MS = 800.0
INPUT_FRAME_AGE_P95_MAX_MS = 4000.0
ADDED_VRAM_MAX_MIB = 1536.0  # 1.5 GiB
MIN_PUBLISH_FPS = 20.0

CHAIN_KEYS = (
    "publish_fps",
    "unique_publish_fps",
    "duplicate_publish_fps",
    "pipeline_healthy",
)
HYBRID_KEYS = (
    "last_prior_ms",
    "last_full_scan_ms",
    "last_crop_batch_ms",
    "last_total_ms",
    "input_frame_age_ms",
    "dropped_analysis_frames",
    "semantic_model_runs_full",
    "semantic_model_runs_crop",
    "branch_state",
    "prior_environment_state",
    "semantic_raw_candidates",
    "semantic_retained_candidates",
    "prior_raw_candidates",
    "prior_retained_candidates",
    "semantic_only_active",
    "semantic_only_confirmed",
    "prior_only_active",
    "prior_only_confirmed",
    "fused_active",
    "fused_confirmed",
    "cross_source_merges",
    "semantic_crop_raw_candidates",
    "semantic_crop_unmatched_candidates",
)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile; returns None for an empty sample."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = fraction * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_sample(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalise one ``GET /v1/streams/{id}`` payload into a measurement row.

    Unknown/missing values are kept as ``None`` so the summary can distinguish
    "not reported" from "reported as zero" — the two have very different meaning
    for a budget audit.
    """
    ground_litter = payload.get("ground_litter") or {}
    hybrid = ground_litter.get("hybrid") or {}
    metrics = payload.get("metrics") or {}

    sample: dict[str, Any] = {
        "state": ground_litter.get("state"),
        "count": ground_litter.get("count"),
        "hybrid_present": bool(hybrid),
    }
    for key in CHAIN_KEYS:
        sample[key] = metrics.get(key)
    for key in HYBRID_KEYS:
        sample[key] = hybrid.get(key)
    return sample


def stage_series(samples: Iterable[dict[str, Any]]) -> dict[str, list[float]]:
    """Split the raw samples into the stage series the budgets are defined on.

    A tick is classified as a full-scan tick when its current snapshot reports
    a positive ``last_full_scan_ms``. ``semantic_model_runs_full`` is a
    cumulative counter and therefore cannot classify the current tick after the
    first scan. Crop batches are counted only where the crop timing is positive,
    so an idle tick does not dilute the p95.
    """
    normal_totals: list[float] = []
    scan_totals: list[float] = []
    scans: list[float] = []
    crops: list[float] = []
    ages: list[float] = []
    for sample in samples:
        scan = _number(sample.get("last_full_scan_ms"))
        is_scan = scan is not None and scan > 0
        total = _number(sample.get("last_total_ms"))
        if total is not None:
            (scan_totals if is_scan else normal_totals).append(total)
        if scan is not None and is_scan:
            scans.append(scan)
        crop = _number(sample.get("last_crop_batch_ms"))
        if crop is not None and crop > 0:
            crops.append(crop)
        age = _number(sample.get("input_frame_age_ms"))
        if age is not None:
            ages.append(age)
    return {
        "normal_tick_total_ms": normal_totals,
        "scan_tick_total_ms": scan_totals,
        "full_scan_ms": scans,
        "crop_batch_ms": crops,
        "input_frame_age_ms": ages,
    }


def summarize_stage(series: dict[str, list[float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, values in series.items():
        summary[name] = {
            "count": len(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "max": max(values) if values else None,
        }
    return summary


def _verdict(
    metric: str, observed: float | bool | None, threshold: Any, *, passed: bool | None
) -> dict[str, Any]:
    return {
        "metric": metric,
        "observed": observed,
        "threshold": threshold,
        "passed": passed,
    }


def evaluate_budget(
    *,
    stages: dict[str, Any],
    chain: dict[str, Any],
    added_vram_mib: float | None,
) -> list[dict[str, Any]]:
    """Turn the raw summaries into explicit pass/fail rows under §10.2.

    ``None`` observed values produce ``passed: None`` (not evaluable) — an
    absent measurement must never be reported as compliant.
    """
    verdicts: list[dict[str, Any]] = []

    def stage_p95(name: str) -> float | None:
        return (stages.get(name) or {}).get("p95")

    normal = stage_p95("normal_tick_total_ms")
    verdicts.append(
        _verdict(
            "normal_tick_p95_ms",
            normal,
            f"<{NORMAL_TICK_P95_MAX_MS:.0f}",
            passed=None if normal is None else normal < NORMAL_TICK_P95_MAX_MS,
        )
    )
    scan = stage_p95("full_scan_ms")
    verdicts.append(
        _verdict(
            "full_scan_p95_ms",
            scan,
            f"<{FULL_SCAN_P95_MAX_MS:.0f}",
            passed=None if scan is None else scan < FULL_SCAN_P95_MAX_MS,
        )
    )
    crop = stage_p95("crop_batch_ms")
    verdicts.append(
        _verdict(
            "crop_batch_p95_ms",
            crop,
            f"<{CROP_BATCH_P95_MAX_MS:.0f}",
            passed=None if crop is None else crop < CROP_BATCH_P95_MAX_MS,
        )
    )
    age = stage_p95("input_frame_age_ms")
    verdicts.append(
        _verdict(
            "input_frame_age_p95_ms",
            age,
            f"<{INPUT_FRAME_AGE_P95_MAX_MS:.0f}",
            passed=None if age is None else age < INPUT_FRAME_AGE_P95_MAX_MS,
        )
    )
    verdicts.append(
        _verdict(
            "added_vram_mib",
            added_vram_mib,
            f"<={ADDED_VRAM_MAX_MIB:.0f}",
            passed=(
                None
                if added_vram_mib is None
                else added_vram_mib <= ADDED_VRAM_MAX_MIB
            ),
        )
    )

    publish = chain.get("publish_fps_min")
    verdicts.append(
        _verdict(
            "publish_fps_min",
            publish,
            f">={MIN_PUBLISH_FPS:.0f}",
            passed=None if publish is None else publish >= MIN_PUBLISH_FPS,
        )
    )
    unique = chain.get("unique_publish_fps_min")
    verdicts.append(
        _verdict(
            "unique_publish_fps_min",
            unique,
            f">={MIN_PUBLISH_FPS:.0f}",
            passed=None if unique is None else unique >= MIN_PUBLISH_FPS,
        )
    )
    duplicate = chain.get("duplicate_publish_fps_max")
    verdicts.append(
        _verdict(
            "duplicate_publish_fps_max",
            duplicate,
            "==0",
            passed=None if duplicate is None else duplicate == 0.0,
        )
    )
    healthy = chain.get("pipeline_healthy_all")
    verdicts.append(
        _verdict(
            "pipeline_healthy_all",
            healthy,
            "all true",
            passed=healthy,
        )
    )
    return verdicts


def summarize_chain(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def series(key: str) -> list[float]:
        return [
            value
            for value in (_number(sample.get(key)) for sample in samples)
            if value is not None
        ]

    publish = series("publish_fps")
    unique = series("unique_publish_fps")
    duplicate = series("duplicate_publish_fps")
    health_flags = [
        sample.get("pipeline_healthy")
        for sample in samples
        if sample.get("pipeline_healthy") is not None
    ]
    return {
        "samples": len(samples),
        "publish_fps_min": min(publish) if publish else None,
        "publish_fps_mean": statistics.fmean(publish) if publish else None,
        "unique_publish_fps_min": min(unique) if unique else None,
        "duplicate_publish_fps_max": max(duplicate) if duplicate else None,
        "pipeline_healthy_all": (
            all(bool(flag) for flag in health_flags) if health_flags else None
        ),
        "pipeline_healthy_samples": len(health_flags),
    }


def summarize_branch_states(samples: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        state = sample.get("branch_state")
        key = str(state) if state is not None else "missing"
        counts[key] = counts.get(key, 0) + 1
    return counts


def fetch_stream_payload(
    base_url: str, stream_id: str, api_key: str, timeout: float = 10.0
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/streams/{stream_id}"
    request = urllib.request.Request(url, headers={"X-API-Key": api_key})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def read_gpu_memory_mib(run: Callable[..., Any] = subprocess.run) -> float | None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    first = (result.stdout or "").strip().splitlines()
    if not first:
        return None
    try:
        return float(first[0].strip())
    except ValueError:
        return None


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key_file:
        return Path(args.api_key_file).read_text(encoding="utf-8").strip()
    key = os.environ.get(args.api_key_env, "").strip()
    if not key:
        raise SystemExit(
            f"未提供 API Key：请设置环境变量 {args.api_key_env} 或使用 --api-key-file"
            "（不要写在命令行上，避免进入 shell 历史）"
        )
    return key


def run_measurement(
    *,
    base_url: str,
    stream_id: str,
    api_key: str,
    samples: int,
    interval_seconds: float,
    baseline_vram_mib: float | None,
    fetch: Callable[[str, str, str, float], dict[str, Any]] = fetch_stream_payload,
    gpu_reader: Callable[[], float | None] = read_gpu_memory_mib,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    vram_samples: list[float] = []
    errors: list[str] = []
    for index in range(samples):
        try:
            payload = fetch(base_url, stream_id, api_key, 10.0)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            progress(f"[{index + 1}/{samples}] 采样失败: {type(exc).__name__}")
        else:
            sample = extract_sample(payload)
            rows.append(sample)
            progress(
                f"[{index + 1}/{samples}] state={sample.get('state')} "
                f"branch={sample.get('branch_state')} "
                f"pub={sample.get('publish_fps')} age={sample.get('input_frame_age_ms')}"
            )
        vram = gpu_reader()
        if vram is not None:
            vram_samples.append(vram)
        if index + 1 < samples:
            sleep(interval_seconds)

    series = stage_series(rows)
    stages = summarize_stage(series)
    chain = summarize_chain(rows)
    peak_vram = max(vram_samples) if vram_samples else None
    added_vram = (
        None
        if peak_vram is None or baseline_vram_mib is None
        else peak_vram - baseline_vram_mib
    )

    doc: dict[str, Any] = {
        "kind": "ground_litter_v33_server_measurement",
        "mode": "hybrid_v33",
        "stream_id": stream_id,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "samples_requested": samples,
        "samples_collected": len(rows),
        "interval_seconds": interval_seconds,
        "budget_source": "docs/plans/2026-09-18-ground-litter-v33-dual-recall-architecture.md §10.2",
        "budget_mapping": {
            "normal_tick_p95_ms": "last_total_ms on ticks with last_full_scan_ms == 0",
            "full_scan_p95_ms": "last_full_scan_ms on ticks with semantic_model_runs_full > 0",
            "crop_batch_p95_ms": "last_crop_batch_ms where > 0",
            "input_frame_age_p95_ms": "input_frame_age_ms over all ticks",
            "added_vram_mib": "peak nvidia-smi memory.used minus --baseline-vram-mib",
        },
        "stage_summary": stages,
        "chain_summary": chain,
        "branch_state_counts": summarize_branch_states(rows),
        "vram": {
            "baseline_mib": baseline_vram_mib,
            "peak_mib": peak_vram,
            "samples_mib": vram_samples,
            "added_mib": added_vram,
            "baseline_provided": baseline_vram_mib is not None,
        },
        "samples": rows,
        "errors": errors,
        "verdicts": evaluate_budget(
            stages=stages, chain=chain, added_vram_mib=added_vram
        ),
    }
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--interval-seconds", type=float, default=4.0)
    parser.add_argument(
        "--baseline-vram-mib",
        type=float,
        default=None,
        help="同管线但不启用 V3.3 旁路时的 nvidia-smi memory.used，用于计算新增显存",
    )
    parser.add_argument(
        "--api-key-env", default="RTSP_API_KEY", help="存放 API Key 的环境变量名"
    )
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查脚本与输出路径可写，不访问 API",
    )
    args = parser.parse_args(argv)

    output = Path(args.output)
    if args.dry_run:
        print(f"输出路径: {output}")
        print("dry-run：未访问 API，未读取 GPU")
        return 0

    api_key = resolve_api_key(args)
    doc = run_measurement(
        base_url=args.base_url,
        stream_id=args.stream_id,
        api_key=api_key,
        samples=args.samples,
        interval_seconds=args.interval_seconds,
        baseline_vram_mib=args.baseline_vram_mib,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已写出 {output}")
    not_evaluable = [v for v in doc["verdicts"] if v["passed"] is None]
    failed = [v for v in doc["verdicts"] if v["passed"] is False]
    print(f"不达标 {len(failed)} 项，不可判定 {len(not_evaluable)} 项")
    for verdict in failed + not_evaluable:
        print(f"  {verdict['metric']}: 观测={verdict['observed']} 门槛={verdict['threshold']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
