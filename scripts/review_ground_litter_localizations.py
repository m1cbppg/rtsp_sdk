#!/usr/bin/env python3
"""Step 1C-1: localization review for the 180 REQUIRED_LITTER Gold episodes.

Subcommands
-----------
plan    read-only: counts, origin breakdown, screening and expected proposal workload
serve   run the local review UI (BOX_OK / BOX_BAD + A/B/C proposal selection)
status  report progress from an existing review state (no images, no proposals)
build   write localizations.jsonl + SUMMARY.json + MANIFEST.json

Gold and the Step 1C-0 recovery evidence are immutable inputs: both are hashed and
never written.  Only episodes a human marked VERIFIED_BBOX become training-ready.
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
from rtsp_annotator.ground_litter_localization_review import (  # noqa: E402
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    LocalizationError,
    ReviewState,
    Reviewer,
    SealedAssetError,
    build_manifest,
    build_summary,
    build_upstream_provenance,
    load_localization_input,
    sha256_file,
    write_outputs,
)

GOLD = ROOT / "output" / "ground_litter_gold_episode_review_20260923"
RECOVERY = ROOT / "output" / "ground_litter_gold_source_recovery_20260923"
STEP1A = ROOT / "output" / "ground_litter_episode_candidates_20260923" / "episode_candidates.jsonl"
DEFAULT_OUTPUT = ROOT / "output" / "ground_litter_gold_localization_20260923"

#: Step 1C-0 provenance, recorded so no downstream step has to guess (§27).
STEP1C0_EXECUTION_COMMIT = "2e5da415c20feda3384e080a3ff9217c97c4e83c"
STEP1C0_MANIFEST_COMMIT = "81c3dde27f7ed759248330e3c332d78907a137e3"
STEP1C0_EVIDENCE_COMMIT = "bfdecd5"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", type=Path, default=GOLD / "gold_episodes.jsonl")
    p.add_argument("--gold-manifest", type=Path, default=GOLD / "MANIFEST.json")
    p.add_argument("--recovery-root", type=Path, default=RECOVERY)
    p.add_argument("--step1a-artifact", type=Path, default=STEP1A)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--episode", action="append", default=None)
    p.add_argument("--camera", action="append", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--generated-at", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8772)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("serve")
    sub.add_parser("status")
    sub.add_parser("build")
    return p


def _state_path(args: argparse.Namespace) -> Path:
    return args.state or (args.output / "review_state.json")


def _load(args: argparse.Namespace):
    return load_localization_input(
        args.gold, args.recovery_root, step1a_artifact=args.step1a_artifact,
        gold_manifest=args.gold_manifest)


def _select(args: argparse.Namespace, data):
    episodes = list(data.required)
    if args.episode:
        wanted = set(args.episode)
        episodes = [e for e in episodes if e.episode_id in wanted]
        unknown = wanted - {e.episode_id for e in episodes}
        if unknown:
            raise SystemExit(f"unknown episode ids: {sorted(unknown)}")
    if args.camera:
        wanted = set(args.camera)
        episodes = [e for e in episodes if e.camera_id in wanted]
    if args.limit is not None:
        episodes = episodes[:max(0, args.limit)]
    return episodes


def _stamp(args: argparse.Namespace) -> str:
    return args.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_plan(args: argparse.Namespace, data) -> int:
    import collections
    episodes = _select(args, data)
    origins = collections.Counter(e.origin for e in episodes)
    reasons = collections.Counter(r for e in episodes for r in e.screen["screen_reasons"])
    report = {
        "gold_artifact": str(data.gold_path),
        "gold_sha256": data.gold_sha256,
        "recovery_evidence_sha256": data.recovery_evidence_sha256,
        "required_episode_count": data.required_count,
        "selected_episode_count": len(episodes),
        "origin": dict(sorted(origins.items())),
        "screen_reasons": dict(sorted(reasons.items())),
        "likely_box_bad": sum(1 for e in episodes if e.screen["likely_box_bad"]),
        "likely_box_ok": sum(1 for e in episodes if not e.screen["likely_box_bad"]),
        "episodes_with_original_bbox": sum(1 for e in episodes if e.original_bbox),
        "episodes_without_original_bbox": sum(1 for e in episodes if not e.original_bbox),
        "manual_point_mapping_verified": sum(
            1 for e in episodes if e.point_source and e.point_source.get("ok")),
        "proposal_workload_if_every_flagged_episode_needs_it": sum(
            1 for e in episodes if e.screen["likely_box_bad"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


def cmd_serve(args: argparse.Namespace, data) -> int:
    tools = ROOT / "tools" / "ground_litter_localization_ui"
    sys.path.insert(0, str(tools))
    import serve as ui  # type: ignore  # noqa: PLC0415

    args.state = str(_state_path(args))
    args.recovery_root = str(args.recovery_root)
    ui.configure(args, data)
    server = ui.create_server(ui.ReviewHandler, args.host, args.port)
    print(f"Step 1C-1 localization review: http://{args.host}:{args.port}/", flush=True)
    print(f"queue={len(ui.ReviewHandler.queue)} episodes state={args.state}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    return 0


def cmd_status(args: argparse.Namespace, data) -> int:
    state = ReviewState.load(_state_path(args), data=data)
    progress = state.progress(data)
    counts: dict[str, int] = {}
    for e in data.required:
        status = state.status_of(e.episode_id)
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({
        "artifact_root": str(args.output),
        "review_state": str(_state_path(args)),
        "required_episode_count": data.required_count,
        **progress,
        "by_status": dict(sorted(counts.items())),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_build(args: argparse.Namespace, data) -> int:
    state = ReviewState.load(_state_path(args), data=data)
    state.save()   # materialise the state file so a first run is resumable
    reviewer = Reviewer(data=data, state=state, engine=None)
    records = reviewer.build_records()
    summary = build_summary(data, records, state)

    after = sha256_file(data.gold_path)
    if after != data.gold_sha256:
        raise LocalizationError("Gold artifact changed during this step")

    provenance = build_upstream_provenance(
        step1c0_reported_execution_commit=STEP1C0_EXECUTION_COMMIT,
        step1c0_manifest_commit=STEP1C0_MANIFEST_COMMIT,
        step1c0_evidence_commit=STEP1C0_EVIDENCE_COMMIT,
        code_equivalence_verified=True,
        note="verified with git diff that the recovery modules are byte-identical "
             "between the reported execution commit and the commit recorded in the "
             "Step 1C-0 MANIFEST")
    manifest = build_manifest(
        data, summary, code_commit=git_commit_of(args.repo_root),
        generated_at=_stamp(args),
        config={"gold": str(args.gold), "recovery_root": str(args.recovery_root),
                "step1a_artifact": str(args.step1a_artifact),
                "output": str(args.output), "state": str(_state_path(args)),
                "schema_version": SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION},
        artifact_root=args.output,
        localization_path=args.output / "localizations.jsonl",
        review_state_path=_state_path(args), provenance=provenance)
    paths = write_outputs(args.output, records=records, summary=summary, manifest=manifest)
    print(json.dumps({
        "required": data.required_count,
        "reviewed": summary["review"]["reviewed"],
        "pending": summary["review"]["pending"],
        "VERIFIED_BBOX": summary["localization"]["VERIFIED_BBOX"],
        "LOCALIZATION_UNRESOLVED": summary["localization"]["LOCALIZATION_UNRESOLVED"],
        "TRUTH_REVIEW_REQUIRED": summary["localization"]["TRUTH_REVIEW_REQUIRED"],
        "training_localization_ready": summary["training_localization_ready"],
        "artifacts": paths,
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        data = _load(args)
    except (SealedAssetError, LocalizationError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    if args.command == "plan":
        return cmd_plan(args, data)
    if args.command == "serve":
        return cmd_serve(args, data)
    if args.command == "status":
        return cmd_status(args, data)
    return cmd_build(args, data)


if __name__ == "__main__":
    raise SystemExit(main())
