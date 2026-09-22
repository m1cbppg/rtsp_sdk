#!/usr/bin/env python3
"""Evaluate a V3.2 online replay against reviewed fixed-view events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_ground_litter_online_replay_v31 import (  # noqa: E402
    evaluate as evaluate_v31,
    render_report as render_report_v31,
)


def evaluate(online_root: Path, reviewed_root: Path) -> dict[str, Any]:
    result = evaluate_v31(online_root, reviewed_root)
    result["kind"] = "ground_litter_online_replay_v32_human_evaluation"
    result["lifecycle_version"] = "3.2"
    return result


def render_report(result: dict[str, Any]) -> str:
    report = render_report_v31(result).replace(
        "# Ground Litter V3.1 在线回放评估",
        "# Ground Litter V3.2 在线回放评估",
        1,
    )
    hardening = (
        "\n## V3.2 生命周期保护\n\n"
        "- `CLEAN_PENDING` 只累计时间连续、间隔受控的有效地面观测。\n"
        "- 环境不可用、人员/结构遮挡、残差仍存在或地面有效覆盖不足时，清洁计时归零。\n"
        "- 可见异常证据仍允许跨遮挡累计；诊断历史与已关闭事件保留数量均有上限。\n"
    )
    marker = "\n## 是否可以进入多摄像头\n"
    return report.replace(marker, hardening + marker, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--reviewed-root", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.online_root, args.reviewed_root)
    (args.online_root / "human_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.online_root / "REPORT.md").write_text(
        render_report(result), encoding="utf-8"
    )
    print(json.dumps(result["online_negative"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
