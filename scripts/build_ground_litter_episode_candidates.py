#!/usr/bin/env python3
"""Step 1A: build episode *candidates* from historical Silver review cards.

Read-only over the original Silver review directories.  Emits a new, deletable,
deterministic artifact set.  Never writes labels, images, timestamps or boxes back.

Example:

    python scripts/build_ground_litter_episode_candidates.py \
        --output output/ground_litter_episode_candidates_20260923
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_episode_candidates import (  # noqa: E402
    DEFAULT_CONFIG,
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    GroupingConfig,
    build_manifest,
    build_summary,
    git_commit_of,
    group_cards,
    group_to_record,
    load_batches,
    load_clean_gaps,
    write_artifacts,
)

#: The six historical human-review rounds whose labels make up the Silver corpus
#: (189 LITTER + 2,083 NON_LITTER + 153 UNCERTAIN + 35 BOX_WRONG = 2,460 cards).
DEFAULT_BATCHES: dict[str, str] = {
    "active_day3": "output/ground_litter_audit_active_day3_20260918/review",
    "holdout": "output/ground_litter_audit_holdout_20260918",
    "formal_day2": "output/ground_litter_audit_formal_day2",
    "formal_day1": "output/ground_litter_audit_formal_day1",
    "pilot_01030": "output/ground_litter_audit_pilot_01030_20260920/review",
    "shadow_20260921": "output/ground_litter_candidate_filter_shadow_20260921/review",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path,
                   default=ROOT / "output" / "ground_litter_episode_candidates_20260923")
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--config", type=Path, default=None,
                   help="optional JSON override of the centralised grouping thresholds")
    p.add_argument("--clean-gap", type=Path, default=None,
                   help="optional JSON with independent 'position observed empty' evidence")
    p.add_argument("--generated-at", default=None,
                   help="override the metadata timestamp (for reproducible reruns)")
    p.add_argument("--print-config", action="store_true")
    return p


def _load_config(path: Path | None) -> GroupingConfig:
    if path is None:
        return GroupingConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("--config must be a JSON object")
    if "config" in payload and isinstance(payload["config"], dict):
        payload = payload["config"]
    return GroupingConfig.from_dict(payload)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = _load_config(args.config)

    if args.print_config:
        print(json.dumps(config.as_dict(), ensure_ascii=False, indent=2))
        return 0

    batch_dirs = {key: ROOT / rel for key, rel in DEFAULT_BATCHES.items()}
    missing = [f"{k}={v}" for k, v in batch_dirs.items() if not v.is_dir()]
    if missing:
        print("missing Silver batch directories:\n  " + "\n  ".join(missing),
              file=sys.stderr)
        return 2

    cards, batch_diagnostics = load_batches(batch_dirs, config=config)

    clean_gaps = []
    clean_gap_used = False
    if args.clean_gap is not None:
        payload = json.loads(args.clean_gap.read_text(encoding="utf-8"))
        clean_gaps = load_clean_gaps(payload)
        clean_gap_used = bool(clean_gaps)

    groups, unresolved = group_cards(cards, config, clean_gaps)
    lookup = {c.card_id: c for c in cards}
    records = [group_to_record(g, config, lookup) for g in groups]

    summary = build_summary(
        groups, records, cards, config=config, unresolved=unresolved,
        batch_diagnostics=batch_diagnostics, clean_gap_evidence_used=clean_gap_used)
    summary["input"]["clean_gap_count"] = len(clean_gaps)

    generated_at = args.generated_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    manifest = build_manifest(
        records=records, cards=cards, batch_diagnostics=batch_diagnostics,
        config=config, git_commit=git_commit_of(args.repo_root), generated_at=generated_at)
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["generator_version"] = GENERATOR_VERSION
    manifest["inputs"] = {
        "batches": DEFAULT_BATCHES,
        "clean_gap_file": str(args.clean_gap) if args.clean_gap else None,
    }

    paths = write_artifacts(args.output, records=records, summary=summary, manifest=manifest)

    print(json.dumps({
        "episode_candidate_count": summary["output"]["candidate_group_count"],
        "singleton_groups": summary["output"]["singleton_group_count"],
        "multi_card_groups": summary["output"]["multi_card_group_count"],
        "ambiguous_groups": summary["output"]["ambiguous_group_count"],
        "input_LITTER_cards": summary["input"]["LITTER_cards"],
        "input_BOX_WRONG_cards": summary["input"]["BOX_WRONG_cards"],
        "artifacts": paths,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
