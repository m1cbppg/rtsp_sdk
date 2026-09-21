"""导出 v4 工厂的评分矩阵（只读：复用 work_v4 已落盘构建 PS，不重建 Bank）。

阶段 2 只用它做离线重新剪枝预演，因此默认只取时间线上最早的若干文件，
避免为了一次预检重新下载 1GB+ 素材。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rtsp_annotator.ground_litter_profile_analysis import BankPriorContext
from rtsp_annotator.ground_litter_profile_bank import load_bank
from rtsp_annotator.ground_litter_recording_cache import ManagedRecordingCache
from rtsp_annotator.ground_litter_recording_source import RecordingFile
from scripts import build_ground_litter_profile_bank as build

MAX_FILES = int(os.environ.get("MATRIX_MAX_FILES", "6"))
BANK_ROOT = Path("out/bank")
BANK_ID = "camera_01030"
VERSION = os.environ.get("MATRIX_VERSION", "v4")
# 参考资产版本：v4 最终只发布 2 个 Profile，做"候选集合预演"必须用仍有全部
# 候选的版本取参考图（v3）；prior 能力始终取自 VERSION 的报告。
ASSET_VERSION = os.environ.get("MATRIX_ASSET_VERSION", "v3")
WORK = Path(f"work_{VERSION}")
DEVICE = "44180209031322001030"
OUT = Path("out/c1c4_v4_matrix/replay_score_matrix.json")


def credential_headers() -> dict[str, str]:
    """回放鉴权：从服务器运行配置读取，不回显、不落盘。"""
    path = os.path.expanduser("~/rtsp-deepstream/config/api.json")
    try:
        payload = json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}
    key = (payload.get("api") or {}).get("key")
    return {"X-API-Key": str(key)} if key else {}


def main() -> int:
    # v4 是契约生效前发布的 Bank（没有能力字段）：显式打开兼容开关按保守值
    # 读取，再用报告里的真实 prior 能力覆盖，以便离线预演新剪枝目标。
    bank = load_bank(
        BANK_ROOT, BANK_ID, VERSION, require_calibration=True,
        allow_legacy_profile_capabilities=True,
    )
    matcher = dict(bank.matcher)
    directory = BANK_ROOT / BANK_ID / VERSION
    geometry = json.loads((directory / "camera_geometry.json").read_text())
    report = json.loads((directory / "reports" / "factory_report.json").read_text())
    config = build.FactoryConfig(
        camera_id=BANK_ID, bank_id=BANK_ID, version=VERSION,
        output_root=BANK_ROOT, work_dir=WORK, geometry_path=None,
        analysis_size=None, device_code=DEVICE,
        raw_cache_budget=8 * 1024 ** 3, work_budget=60 * 1024 ** 3,
    )

    prior_by_pid = {
        f"p{index:04d}": row["noise_calibration"]
        for index, row in enumerate(report["composite"], start=1)
    }
    assets = load_bank(
        BANK_ROOT, BANK_ID, ASSET_VERSION, require_calibration=True,
        allow_legacy_profile_capabilities=True,
    )
    asset_matcher = dict(assets.matcher)
    candidates = []
    for pid in assets.ids():
        entry = assets.profile(pid)
        capability = dict(prior_by_pid.get(pid) or {})
        reference = assets.load_reference(pid)
        valid = assets.load_valid(pid)
        candidates.append({
            "profile_id": pid,
            "group_id": str(entry.metadata.get("group_id") or pid),
            "reference": reference, "valid": valid, "noise": {},
            "descriptor": assets.load_descriptor(pid),
            "envelope": (
                asset_matcher.get("profiles", {}).get(pid) or {}
            ).get("envelope"),
            "noise_calibration": capability,
            "context": BankPriorContext(
                profile_id=pid, reference=reference, valid=valid, noise={},
                metadata=dict(entry.metadata),
            ),
        })

    inventory = json.loads(
        (WORK / "inventory_5307975a77b599f5.json").read_text()
    )
    needed = set(report["material_plan"]["needed_file_ids"])
    files = [RecordingFile(
        file_id=row["file_id"], file_name=row.get("file_name", ""),
        record_start=row["record_start"], record_end=row["record_end"],
        file_size=row.get("file_size"), file_type=row.get("file_type", ""),
    ) for row in inventory["files"] if row["file_id"] in needed]
    files.sort(key=lambda item: (item.record_start, item.file_id))
    files = files[:MAX_FILES]

    headers = credential_headers()
    args = SimpleNamespace(
        source="ctseelink-file-urls", auth_token=None,
        api_key=headers.get("X-API-Key"), input=None,
    )
    print(
        f"files={len(files)} candidates={len(candidates)} "
        f"capability={build.prior_suitable_ids_of(candidates)} "
        f"auth={'yes' if headers else 'no'}",
        flush=True,
    )
    with ManagedRecordingCache(
        WORK, raw_cache_budget=8 * 1024 ** 3, work_budget=60 * 1024 ** 3,
    ) as cache:
        cache.recover()
        started = time.monotonic()
        frames = build._collect_replay_frames(
            cache, config, files, geometry, frame_budget=64,
            args=args, local_paths={}, report={},
        )
        print(f"frames={len(frames)} in {time.monotonic()-started:.1f}s", flush=True)
        if len(frames) < 10:
            raise SystemExit("素材不足，无法导出矩阵")
        started = time.monotonic()
        replay: dict = {}
        result = build.replay_all_candidates(
            config, candidates, frames, geometry, matcher, replay,
            leave_one_out=False, loo_stride=1,
        )
        print(f"matrix in {time.monotonic()-started:.1f}s", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "matrix": build.export_replay_score_matrix(result["score_matrix"]),
        "selector_config": dict(result["selector_config"]),
        "nominal_tick_seconds": result["nominal_tick_seconds"],
        "view_id": result["view_id"],
        "prior_suitable": build.prior_suitable_ids_of(candidates),
        "files_used": [item.file_id for item in files],
        "candidates": [
            {
                "profile_id": item["profile_id"],
                "prior_suitable": bool(
                    (item["noise_calibration"] or {}).get("prior_suitable")
                ),
                "group_id": item["group_id"],
            }
            for item in candidates
        ],
    }, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} {OUT.stat().st_size} bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
