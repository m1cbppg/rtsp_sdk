"""阶段 2：用 v4 真实候选集合的评分矩阵离线重跑**双目标剪枝**（不连服务器）。

输入：`export_profile_bank_score_matrix.py` 导出的 JSON（矩阵 + 每候选能力）。
输出：预计最终集合、两类覆盖、每轮删除前后的指标，以及 prior 是否可用。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import build_ground_litter_profile_bank as build  # noqa: E402


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else
                "output/c1c4_v3fix_20260921/replay_score_matrix.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = build.import_replay_score_matrix(payload["matrix"])
    capability = {
        str(row["profile_id"]): bool(row.get("prior_suitable"))
        for row in payload.get("candidates", [])
    }
    candidates = []
    for pid in matrix.profile_ids:
        candidates.append({
            "profile_id": pid,
            "group_id": capability.get(pid, pid) and pid or pid,
            "noise_calibration": {
                "prior_suitable": bool(capability.get(pid, False)),
                "calibration_sufficient": bool(capability.get(pid, False)),
                "source": (
                    "calibration_day" if capability.get(pid, False)
                    else "reference_self"
                ),
            },
        })
    replay = {
        "score_matrix": matrix,
        "selector_config": dict(payload["selector_config"]),
        "nominal_tick_seconds": float(payload["nominal_tick_seconds"]),
        "view_id": str(payload["view_id"]),
        "score_matrix_profile_ids": list(matrix.profile_ids),
    }
    config = SimpleNamespace(bank_id="camera_01030", version="v4", max_profiles=24)
    report: dict = {}
    kept = build.select_profiles_dynamically(config, candidates, replay, report)
    pruning = report["pruning"]
    summary = {
        "frames": matrix.frame_count,
        "candidates": list(matrix.profile_ids),
        "prior_suitable_candidates": build.prior_suitable_ids_of(candidates),
        "kept": [item["profile_id"] for item in kept],
        "prior_suitable_profiles_kept": pruning["prior_suitable_profiles_kept"],
        "prior_bank_available": pruning["prior_bank_available"],
        "match_coverage": pruning["match_coverage"],
        "prior_effective_coverage": pruning["prior_effective_coverage"],
        "prior_pause_max": pruning["prior_pause_max"],
        "semantic_only_intervals": pruning["semantic_only_intervals"],
        "deletion_rounds": [
            {
                "round": row["round"], "removed": row["removed"],
                "metrics_before": {
                    "match_coverage": row["metrics_before"].get("match_coverage"),
                    "prior_effective_coverage":
                        row["metrics_before"].get("prior_effective_coverage"),
                },
                "metrics_after": {
                    "match_coverage": row["metrics_after"].get("match_coverage"),
                    "prior_effective_coverage":
                        row["metrics_after"].get("prior_effective_coverage"),
                },
            }
            for row in pruning["deletion_order"]
        ],
        "warnings": report.get("pruning_warnings") or [],
        "note": (
            "离线预演：只重跑选择/剪枝，不重做下载、合成与噪声；"
            "参考资产取自同一分区、同一分组的完整候选集合。"
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
