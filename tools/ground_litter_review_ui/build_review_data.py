#!/usr/bin/env python3
"""Build the static review dataset from an offline VLM review manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "output/ground_litter_prior_region_v3_20260917/vlm_review/negative_v3_manifest.jsonl"
)


def browser_path(value: str) -> str:
    path = Path(value).resolve()
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"review asset is outside repository root: {path}") from exc
    return "/" + relative.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=HERE / "dist/review-data.js")
    args = parser.parse_args()

    raw = args.manifest.read_bytes()
    records = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records.append({
            **row,
            "clean_reference_crop": browser_path(row["clean_reference_crop"]),
            "current_crop": browser_path(row["current_crop"]),
            "context_crop": browser_path(row["context_crop"]),
        })

    payload = {
        "dataset": "ground_litter_negative_v3",
        "title": "零散垃圾候选人工审核",
        "fingerprint": hashlib.sha256(raw).hexdigest()[:16],
        "manifest": "/" + args.manifest.resolve().relative_to(REPO_ROOT).as_posix(),
        "count": len(records),
        "items": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "window.REVIEW_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "count": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
