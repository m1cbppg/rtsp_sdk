#!/usr/bin/env python3
"""Step 1C-1R: reconcile the truth of the 60 TRUTH_REVIEW_REQUIRED episodes.

Subcommands
-----------
plan    read-only: preflight counts, in-scope episodes, prior reason breakdown
serve   run the minimal local review UI (5 reconciliation buttons only)
status  report progress from an existing review state
build   write truth_reconciliation.jsonl + training_episode_manifest.jsonl
        + SUMMARY.json + MANIFEST.json

Gold, the Step 1C-0 recovery evidence and the Step 1C-1 localization artifact are all
immutable inputs.  This step changes no bbox and re-reviews no episode outside the 60.
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
from rtsp_annotator.ground_litter_truth_reconciliation import (  # noqa: E402
    DECISIONS,
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    ReconciliationError,
    ReviewState,
    SealedAssetError,
    build_manifest,
    build_overlay_rows,
    build_summary,
    build_upstream_provenance,
    load_reconciliation_input,
    sha256_file,
    verify_preflight,
    write_outputs,
)

GOLD = ROOT / "output" / "ground_litter_gold_episode_review_20260923"
RECOVERY = ROOT / "output" / "ground_litter_gold_source_recovery_20260923"
LOCALIZATION = ROOT / "output" / "ground_litter_gold_localization_20260923"
DEFAULT_OUTPUT = ROOT / "output" / "ground_litter_truth_reconciliation_20260923"

#: Upstream commits, recorded so no step has to infer them (§27 pattern).
#: ``*_EXECUTION`` is the commit that introduced the code, ``*_MANIFEST`` is the commit a
#: step's own MANIFEST records as its ``code_commit`` (git HEAD when the evidence was
#: written), and ``*_EVIDENCE`` is the commit that actually added that evidence file.
STEP1C0_EXECUTION = "2e5da415c20feda3384e080a3ff9217c97c4e83c"
STEP1C0_MANIFEST = "81c3dde27f7ed759248330e3c332d78907a137e3"
STEP1C0_EVIDENCE = "bfdecd5"
STEP1C1_EXECUTION = "809ce49196dadc639fe90438da3c27057baa34bd"
STEP1C1_MANIFEST = "806f8de4cb8b750c1c68aa262571d59ea1c8d0ba"
STEP1C1_EVIDENCE = "1a1f65c14a5e971baa00eb36774e9c85911124d8"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", type=Path, default=GOLD / "gold_episodes.jsonl")
    p.add_argument("--gold-manifest", type=Path, default=GOLD / "MANIFEST.json")
    p.add_argument("--recovery-root", type=Path, default=RECOVERY)
    p.add_argument("--localization-root", type=Path, default=LOCALIZATION)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--generated-at", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8773)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("serve")
    sub.add_parser("status")
    sub.add_parser("build")
    return p


def _state_path(args: argparse.Namespace) -> Path:
    return args.state or (args.output / "review_state.json")


def _load(args: argparse.Namespace):
    return load_reconciliation_input(args.gold, args.recovery_root,
                                     args.localization_root,
                                     gold_manifest=args.gold_manifest)


def _stamp(args: argparse.Namespace) -> str:
    return args.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_plan(args: argparse.Namespace, data) -> int:
    import collections
    scope = data.in_scope
    report = {
        "gold_artifact": str(data.gold_path),
        "gold_sha256": data.gold_sha256,
        "recovery_evidence_sha256": data.recovery_evidence_sha256,
        "localization_artifact": str(data.localization_path),
        "localization_sha256": data.localization_sha256,
        "localization_summary_sha256": data.localization_summary_sha256,
        "localization_manifest_sha256": data.localization_manifest_sha256,
        "step1c1_code_commit_recorded_in_manifest": data.step1c1_code_commit,
        "preflight": verify_preflight(data),
        "in_scope_count": len(scope),
        "in_scope_by_origin": dict(sorted(collections.Counter(e.origin for e in scope).items())),
        "in_scope_by_camera": dict(sorted(collections.Counter(e.camera_id for e in scope).items())),
        "prior_reason_present": sum(1 for e in scope if e.prior_truth_review_reason),
        "prior_reason_missing": sum(1 for e in scope if not e.prior_truth_review_reason),
        "prior_reason_histogram": dict(collections.Counter(
            e.prior_truth_review_reason or "NO_PRIOR_REASON" for e in scope).most_common()),
        "note": "prior reasons are displayed for context only and are never auto-classified",
        "frozen_out_of_scope": {
            "verified_bbox": sum(1 for e in data.episodes
                                 if e.localization_status == "VERIFIED_BBOX"),
            "localization_unresolved": sum(1 for e in data.episodes
                                           if e.localization_status == "LOCALIZATION_UNRESOLVED"),
        },
        "options": list(DECISIONS),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


def cmd_serve(args: argparse.Namespace, data) -> int:
    tools = ROOT / "tools" / "ground_litter_truth_ui"
    sys.path.insert(0, str(tools))
    import serve as ui  # type: ignore  # noqa: PLC0415

    args.state = str(_state_path(args))
    ui.configure(args, data)
    server = ui.create_server(ui.ReviewHandler, args.host, args.port)
    print(f"Step 1C-1R truth reconciliation: http://{args.host}:{args.port}/", flush=True)
    print(f"in_scope={len(ui.ReviewHandler.queue)} state={args.state}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    return 0


def cmd_status(args: argparse.Namespace, data) -> int:
    state = ReviewState.load(_state_path(args), data=data)
    progress = state.progress(data)
    decisions: dict[str, int] = {d: 0 for d in DECISIONS}
    for row in state.decisions.values():
        key = str(row.get("reconciliation_decision"))
        if key in decisions:
            decisions[key] += 1
    overlay, manifest = build_overlay_rows(data, state)
    eligible = [r["episode_id"] for r in manifest if r["training_eligible"]]
    print(json.dumps({
        "artifact_root": str(args.output),
        "review_state": str(_state_path(args)),
        "preflight": verify_preflight(data),
        **progress,
        "decisions": decisions,
        "training_eligible_episode_count": len(eligible),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_build(args: argparse.Namespace, data) -> int:
    state = ReviewState.load(_state_path(args), data=data)
    state.save()
    overlay, manifest_rows = build_overlay_rows(data, state)
    preflight = verify_preflight(data)
    summary = build_summary(data, state, overlay, manifest_rows, preflight)

    if sha256_file(data.gold_path) != data.gold_sha256:
        raise ReconciliationError("Gold artifact changed during this step")
    if sha256_file(data.localization_path) != data.localization_sha256:
        raise ReconciliationError("localization artifact changed during this step")

    provenance = build_upstream_provenance(
        step1c0_execution_commit=STEP1C0_EXECUTION,
        step1c0_manifest_commit=STEP1C0_MANIFEST,
        step1c0_evidence_commit=STEP1C0_EVIDENCE,
        step1c1_execution_commit=STEP1C1_EXECUTION,
        step1c1_manifest_commit=STEP1C1_MANIFEST,
        step1c1_evidence_commit=STEP1C1_EVIDENCE,
        code_equivalence_verified=True,
        note="both commit pairs verified with git diff to be byte-identical for the "
             "relevant module files")
    manifest = build_manifest(
        data, summary, code_commit=git_commit_of(args.repo_root),
        generated_at=_stamp(args),
        config={"gold": str(args.gold), "recovery_root": str(args.recovery_root),
                "localization_root": str(args.localization_root),
                "output": str(args.output), "state": str(_state_path(args)),
                "schema_version": SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION},
        artifact_root=args.output,
        overlay_path=args.output / "truth_reconciliation.jsonl",
        manifest_path=args.output / "training_episode_manifest.jsonl",
        review_state_path=_state_path(args), provenance=provenance)
    paths = write_outputs(args.output, overlay=overlay, manifest_rows=manifest_rows,
                          summary=summary, manifest=manifest)
    written = json.loads((args.output / "MANIFEST.json").read_text(encoding="utf-8"))
    for key, path in (("truth_reconciliation_sha256", args.output / "truth_reconciliation.jsonl"),
                      ("training_episode_manifest_sha256",
                       args.output / "training_episode_manifest.jsonl"),
                      ("summary_sha256", args.output / "SUMMARY.json")):
        if written.get(key) != sha256_file(path):
            raise ReconciliationError(f"artifact hash mismatch for {path}")
    print(json.dumps({
        "required_before_reconciliation": summary["input"]["required_before_reconciliation"],
        "in_scope": summary["review"]["total_in_scope"],
        "reviewed": summary["review"]["reviewed"],
        "pending": summary["review"]["pending"],
        "reconciliation": summary["reconciliation"],
        "effective_truth_totals": summary["effective_truth_totals"],
        "pending_reconciliation_count": summary["pending_reconciliation_count"],
        "training_eligible_episode_count":
            summary["training"]["training_eligible_episode_count"],
        "truth_reconciliation_sha256": written["truth_reconciliation_sha256"],
        "artifacts": paths,
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        data = _load(args)
    except (SealedAssetError, ReconciliationError) as exc:
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
