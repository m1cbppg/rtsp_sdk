#!/usr/bin/env python3
"""Step 1B: human-confirm historical Silver episodes and audit trainability.

Subcommands
-----------
serve   run the local (offline) review UI
status  print review progress, risk coverage and input-lineage readiness
build   materialise gold_episodes.jsonl + SUMMARY.json + MANIFEST.json

The Step 1A artifact is an immutable input: it is hashed and recorded, never written.
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

from rtsp_annotator.ground_litter_episode_candidates import git_commit_of  # noqa: E402
from rtsp_annotator.ground_litter_gold_episode_review import (  # noqa: E402
    SCHEMA_VERSION,
    ReviewState,
    TrainabilityPolicy,
    build_gold_records,
    build_manifest,
    build_queue,
    build_summary,
    candidate_lineage_readiness,
    load_step1a,
    write_review_outputs,
)

DEFAULT_ARTIFACT = ROOT / "output" / "ground_litter_episode_candidates_20260923" / "episode_candidates.jsonl"
DEFAULT_STEP1A_MANIFEST = ROOT / "output" / "ground_litter_episode_candidates_20260923" / "MANIFEST.json"
DEFAULT_OUTPUT = ROOT / "output" / "ground_litter_gold_episode_review_20260923"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    p.add_argument("--step1a-manifest", type=Path, default=DEFAULT_STEP1A_MANIFEST)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--state", type=Path, default=None,
                   help="defaults to <output>/review_state.json")
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--retention-days", type=float, default=7.0,
                   help="declared remote recording retention used by the lineage audit")
    p.add_argument("--source-evidence", type=Path, default=None,
                   help="optional JSON proving PS reachability or full-frame assets")
    p.add_argument("--generated-at", default=None)
    p.add_argument("--port", type=int, default=8771)
    p.add_argument("--host", default="127.0.0.1")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("serve")
    sub.add_parser("status")
    sub.add_parser("build")
    return p


def _state_path(args: argparse.Namespace) -> Path:
    return args.state or (args.output / "review_state.json")


def _policy(args: argparse.Namespace) -> TrainabilityPolicy:
    return TrainabilityPolicy(
        retention_days=args.retention_days,
        reference_time=datetime.strptime(args.generated_at, "%Y-%m-%dT%H:%M:%SZ")
        if args.generated_at else None,
    )


def _load_evidence(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    step1a = load_step1a(args.artifact, args.step1a_manifest, repo_root=args.repo_root)
    queue = build_queue(step1a)
    state_path = _state_path(args)
    policy = _policy(args)
    evidence = _load_evidence(args.source_evidence)

    if args.command == "serve":
        tools = ROOT / "tools" / "ground_litter_gold_review_ui"
        sys.path.insert(0, str(tools))
        import serve as ui  # type: ignore  # noqa: PLC0415

        args.state = str(state_path)
        args.manifest = str(args.step1a_manifest)
        ui.configure(args)
        server = ui.create_server(ui.ReviewHandler, args.host, args.port)
        print(f"Step 1B gold-episode review: http://{args.host}:{args.port}/", flush=True)
        print(f"queue={len(ui.ReviewHandler.queue)} candidates "
              f"state={state_path}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("stopped", flush=True)
        return 0

    state = ReviewState.load(state_path, step1a=step1a)

    if args.command == "status":
        print(json.dumps({
            "step1a_artifact": str(step1a.artifact_path),
            "step1a_artifact_sha256": step1a.artifact_sha256,
            "step1a_code_commit": step1a.step1a_code_commit,
            "candidate_count": step1a.candidate_count,
            "member_card_count": step1a.member_card_count,
            "review_state": str(state_path),
            "progress": state.progress(step1a),
            "risk_coverage": {
                "grouping_risk": sum(1 for q in queue if q["grouping_risk"]),
                "lineage_only": sum(1 for q in queue if q["lineage_only"]),
                "multi_card": sum(1 for q in queue if q["member_count"] > 1),
            },
            "input_lineage_readiness": candidate_lineage_readiness(
                step1a, policy=policy, source_evidence=evidence),
        }, ensure_ascii=False, indent=2))
        return 0

    # Materialise review_state.json so the reviewer can resume even before the
    # first decision; loading already read any existing decisions, so this is
    # idempotent and never discards progress.
    state.save()

    records, conflicts = build_gold_records(
        step1a, state, policy=policy, source_evidence=evidence)
    summary = build_summary(step1a, state, records, conflicts, queue=queue)
    summary["input_lineage_readiness"] = candidate_lineage_readiness(
        step1a, policy=policy, source_evidence=evidence)

    generated_at = args.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = build_manifest(
        step1a, state,
        step1b_code_commit=git_commit_of(args.repo_root),
        config={
            "artifact": str(args.artifact),
            "step1a_manifest": str(args.step1a_manifest),
            "output": str(args.output),
            "state": str(state_path),
            "retention_days": args.retention_days,
            "source_evidence": str(args.source_evidence) if args.source_evidence else None,
            "schema_version": SCHEMA_VERSION,
        },
        generated_at=generated_at,
        review_state_path=state_path,
        gold_path=args.output / "gold_episodes.jsonl",
        summary_path=args.output / "SUMMARY.json",
    )
    paths = write_review_outputs(args.output, records=records, summary=summary, manifest=manifest)
    print(json.dumps({
        "reviewed": summary["human_review"]["reviewed"],
        "pending": summary["human_review"]["pending"],
        "gold_episode_count": summary["gold_episodes"]["gold_episode_count"],
        "conflicts": len(conflicts),
        "artifacts": paths,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
