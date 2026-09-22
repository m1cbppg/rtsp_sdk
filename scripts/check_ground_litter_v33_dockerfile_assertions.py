#!/usr/bin/env python3
"""Run the V3.3 Dockerfile's build-time assertions locally.

The candidate image can only be built on the server, so a broken assertion inside
``Dockerfile.deepstream.ground-litter-v33-update`` would not be discovered until
that authorised build runs. This script extracts the ``RUN python3 - <<'PY'``
blocks from the Dockerfile **verbatim**, rewrites the in-image ``/app`` prefix to
this checkout, and executes them here against the real reviewed weight.

It therefore checks the actual shipped build logic, not a copy of it. Requires
the local venv (torch/ultralytics/cv2) and the reviewed weight; it never talks to
the network and never builds an image.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile.deepstream.ground-litter-v33-update"
IMAGE_ROOT = "/app"

# ``RUN python3 - <<'PY' ... PY`` heredocs are the build-time checks.
HEREDOC = re.compile(
    r"RUN python3 - <<'PY'\n(?P<body>.*?)\nPY\n",
    re.DOTALL,
)


def extract_blocks(text: str) -> list[str]:
    return [match.group("body") for match in HEREDOC.finditer(text)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the rewritten temporary scripts for inspection",
    )
    args = parser.parse_args()

    text = DOCKERFILE.read_text(encoding="utf-8")
    blocks = extract_blocks(text)
    if not blocks:
        print("未在 Dockerfile 中找到 RUN python3 构建期断言块", file=sys.stderr)
        return 2

    weight = REPO_ROOT / "models" / "litter" / "turhancan_yolov8m_seg_trash.pt"
    if not weight.is_file():
        print(f"缺少已复核权重，无法本地复现构建断言: {weight}", file=sys.stderr)
        return 2

    failures = 0
    for index, body in enumerate(blocks, start=1):
        rewritten = body.replace(IMAGE_ROOT + "/", str(REPO_ROOT) + "/")
        # Defensive: a bare "/app" reference (no trailing slash) would otherwise
        # survive and fail with a confusing FileNotFoundError.
        rewritten = re.sub(r'"/app"', f'"{REPO_ROOT}"', rewritten)
        print(f"=== Dockerfile 构建期断言块 {index}/{len(blocks)} ===", flush=True)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=not args.keep, encoding="utf-8"
        ) as handle:
            handle.write(rewritten)
            handle.flush()
            print(f"   临时脚本: {handle.name}" if args.keep else "", flush=True)
            result = subprocess.run(
                [sys.executable, handle.name],
                cwd=REPO_ROOT,
                text=True,
            )
        if result.returncode != 0:
            failures += 1
            print(f"块 {index} 失败，退出码 {result.returncode}", file=sys.stderr)

    if failures:
        print(f"构建期断言本地复现失败: {failures}/{len(blocks)}", file=sys.stderr)
        return 1
    print(f"全部 {len(blocks)} 个构建期断言块本地通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
