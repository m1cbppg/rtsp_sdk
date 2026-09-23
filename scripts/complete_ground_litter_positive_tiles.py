#!/usr/bin/env python3
"""Step 1C-2M: complete multi-target positive tiles (Missing-Required salvage).

Subcommands
-----------
plan    read-only: the 62 MISSING_REQUIRED tiles, expected counts and blockers
serve   point-click completion UI: click a missed Required, pick an A/B/C proposal
status  report completion progress and proposal statistics
build   only when every queued tile is reviewed: emit accepted_v2 (29 frozen
        positives + the salvaged tiles) with byte-identical tile images

Only the MISSING_REQUIRED tiles are touched.  The 29 accepted tiles and the 2
BOX_PROBLEM tiles are never re-reviewed, the crop never moves, the tile PNG is never
re-encoded, no original bbox is modified, and nothing is auto-labelled: every
supplemental bbox comes from an explicit human pick followed by an explicit recheck.
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
    SealedAssetError,
)
from rtsp_annotator.ground_litter_tile_completion import (  # noqa: E402
    FINAL_TILE_STATUSES,
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    TILE_STATUSES,
    CompletionError,
    CompletionState,
    apply_state,
    build_accepted_v2,
    build_manifest,
    build_summary,
    load_completion_input,
    proposal_statistics,
    queue_rows,
    read_jsonl,
    sha256_file,
    verify_preflight,
    write_jsonl,
    write_outputs,
)

STEP1C2_ROOT = ROOT / "output" / "ground_litter_positive_tiles_20260923"
DEFAULT_OUTPUT = ROOT / "output" / "ground_litter_positive_tile_completion_20260923"

#: Upstream commits (the same §47-style record used by the earlier steps).
STEP1C0_EXECUTION = "2e5da415c20feda3384e080a3ff9217c97c4e83c"
STEP1C0_MANIFEST = "81c3dde27f7ed759248330e3c332d78907a137e3"
STEP1C0_EVIDENCE = "bfdecd5"
STEP1C1_EXECUTION = "809ce49196dadc639fe90438da3c27057baa34bd"
STEP1C1_MANIFEST = "806f8de4cb8b750c1c68aa262571d59ea1c8d0ba"
STEP1C1_EVIDENCE = "1a1f65c14a5e971baa00eb36774e9c85911124d8"
STEP1C1R_EXECUTION = "5399c4ff972b3f6bff02ba66bc861bc0b684b778"
STEP1C1R_MANIFEST = "d841864f3a997347bf53db9cf24c2d1940052ac6"
STEP1C1R_EVIDENCE = "6c60bb8120161f9f735b00a465cc49cb426369c2"
STEP1C2_EXECUTION = "7944dadf248089deca23f097c2f9442770cc5635"
STEP1C2_SECTION17_CORRECTION = "34339737413f1843bab0c3a130c47a4b0bcee3a8"
STEP1C2_MANIFEST = "abe131ec6fe4895dfdcd4c7909eff4f3b65b4b48"
STEP1C2_EVIDENCE = "c548f27e816e4cf80ffd8b7e09a862428e6c9398"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step1c2-root", type=Path, default=STEP1C2_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--generated-at", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8775)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("serve")
    sub.add_parser("status")
    sub.add_parser("build")
    return p


def _state_path(args: argparse.Namespace) -> Path:
    return args.state or (args.output / "completion_state.json")


def _load(args: argparse.Namespace) -> dict:
    return load_completion_input(args.step1c2_root)


def _stamp(args: argparse.Namespace) -> str:
    return args.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fingerprint(data: dict) -> str:
    import hashlib

    material = "|".join([GENERATOR_VERSION, data["summary_sha256"],
                         data["manifest_sha256"], data["tiles_sha256"],
                         data["review_state_sha256"]])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _provenance() -> dict:
    return {
        "step1c0_reported_execution_commit": STEP1C0_EXECUTION,
        "step1c0_manifest_commit": STEP1C0_MANIFEST,
        "step1c0_evidence_commit": STEP1C0_EVIDENCE,
        "step1c1_reported_execution_commit": STEP1C1_EXECUTION,
        "step1c1_manifest_commit": STEP1C1_MANIFEST,
        "step1c1_evidence_commit": STEP1C1_EVIDENCE,
        "step1c1r_reported_execution_commit": STEP1C1R_EXECUTION,
        "step1c1r_manifest_commit": STEP1C1R_MANIFEST,
        "step1c1r_evidence_commit": STEP1C1R_EVIDENCE,
        "step1c2_reported_execution_commit": STEP1C2_EXECUTION,
        "step1c2_section17_correction_commit": STEP1C2_SECTION17_CORRECTION,
        "step1c2_manifest_commit": STEP1C2_MANIFEST,
        "step1c2_evidence_commit": STEP1C2_EVIDENCE,
        "note": ("Step 1C-2M only reads the Step 1C-2 artifacts recorded here; their "
                 "SHA-256 is re-verified before anything is written"),
    }


def cmd_plan(args: argparse.Namespace, data: dict) -> int:
    import collections

    preflight = verify_preflight(data)
    queue = queue_rows(data)
    report = {
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "step1c2_root": str(data["root"]),
        "missing_required_input": len(queue),
        "old_accepted": len(data["accepted_rows"]),
        "box_problem_excluded": len(data["box_problem"]),
        "preflight": preflight,
        "per_camera_missing": dict(sorted(collections.Counter(
            row["camera_id"] for row in queue).items())),
        "existing_label_count_histogram": dict(sorted(collections.Counter(
            str(row["existing_label_count"]) for row in queue).items())),
        "risk_flagged_missing": sum(
            1 for row in queue if row["known_unlocalized_required_present"]),
        "candidate_png_count": preflight["candidate_png_count"],
        "hashes": preflight["hashes"],
        "input_fingerprint": _fingerprint(data),
        "blockers": ([{"kind": "preflight_counts_mismatch",
                       "value": preflight}]
                     if not preflight["counts_match"] else [])
        + list(preflight["problems"]),
        "note": ("read-only: no image is decoded, no crop moves, and the 29 accepted "
                 "tiles plus the 2 BOX_PROBLEM tiles stay untouched"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


def cmd_serve(args: argparse.Namespace, data: dict) -> int:
    tools = ROOT / "tools" / "ground_litter_completion_ui"
    sys.path.insert(0, str(tools))
    import serve as ui  # type: ignore  # noqa: PLC0415

    args.state = str(_state_path(args))
    args.input_fingerprint = _fingerprint(data)
    ui.configure(args, data)
    handler = ui.CompletionHandler
    server = ui.create_server(handler, args.host, args.port)
    print(f"Step 1C-2M supplemental completion: http://{args.host}:{args.port}/",
          flush=True)
    print(f"queue={len(handler.queue)} state={args.state}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    return 0


def _state(args: argparse.Namespace, data: dict) -> CompletionState:
    return CompletionState.load(_state_path(args), input_fingerprint=_fingerprint(data))


def _preview(args: argparse.Namespace, data: dict, state: CompletionState) -> dict:
    queue = queue_rows(data)
    applied = apply_state(queue, state)
    return {
        "queue": len(queue),
        "progress": state.progress(queue),
        "status_counts": _counter(row["completion_status"] for row in applied),
        "proposal": proposal_statistics(applied),
        "supplemental_targets": sum(len(row["supplemental_targets"]) for row in applied),
    }


def cmd_status(args: argparse.Namespace, data: dict) -> int:
    state = _state(args, data)
    queue = queue_rows(data)
    preview = _preview(args, data, state)
    would_be = 0
    applied = apply_state(queue, state)
    for row in applied:
        if row["completion_status"] == TILE_STATUSES[1]:
            would_be += 1
    print(json.dumps({
        "artifact_root": str(args.output),
        "state": str(_state_path(args)),
        **preview,
        "would_be_salvaged_tile_count": would_be,
        "would_be_accepted_v2_tile_count": len(data["accepted_rows"]) + would_be,
        "build_allowed": (preview["progress"]["pending"] == 0
                          and preview["progress"]["skipped"] == 0),
        "old_accepted_tile_count": len(data["accepted_rows"]),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_build(args: argparse.Namespace, data: dict) -> int:
    state = _state(args, data)
    queue = queue_rows(data)
    accepted_root = args.output / "accepted_v2"
    accepted = build_accepted_v2(data, queue, accepted_root, state=state)
    applied = apply_state(queue, state)
    preflight = verify_preflight(data)

    for path, recorded in ((data["tiles_path"], data["tiles_sha256"]),
                           (data["summary_path"], data["summary_sha256"]),
                           (data["manifest_path"], data["manifest_sha256"]),
                           (data["review_state_path"], data["review_state_sha256"])):
        if sha256_file(path) != recorded:
            raise CompletionError(f"Step 1C-2 input changed during this step: {path}")

    immutability = {"checked": 0, "mismatch": [], "step1c2_sha_checked": True}
    for row in accepted["rows"]:
        source = Path(str(row["image_path"]))
        if sha256_file(source) != row["image_sha256"]:
            immutability["mismatch"].append(row["tile_id"])
        immutability["checked"] += 1
    if immutability["mismatch"]:
        raise CompletionError(
            f"accepted_v2 image bytes differ from Step 1C-2: {immutability['mismatch']}")

    summary = build_summary(data, queue, preflight, state=state, accepted_v2=accepted,
                            proposal_stats=proposal_statistics(applied),
                            image_immutability=immutability)
    manifest = build_manifest(
        data, summary, code_commit=git_commit_of(args.repo_root),
        generated_at=_stamp(args),
        config={"step1c2_root": str(args.step1c2_root), "output": str(args.output),
                "state": str(_state_path(args)), "accepted_root": str(accepted_root),
                "schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION},
        artifact_root=args.output, provenance=_provenance())
    manifest["input_fingerprint"] = _fingerprint(data)
    paths = write_outputs(args.output, summary=summary, manifest=manifest)
    write_jsonl(args.output / "tile_completion_manifest.jsonl", applied)
    write_jsonl(args.output / "positive_training_manifest_v2.jsonl", accepted["rows"])
    print(json.dumps({
        "old_accepted_positive_tile_count":
            summary["final"]["old_accepted_positive_tile_count"],
        "salvaged_positive_tile_count": summary["final"]["salvaged_positive_tile_count"],
        "accepted_v2_positive_tile_count":
            summary["final"]["accepted_v2_positive_tile_count"],
        "accepted_v2_label_count": summary["final"]["accepted_v2_label_count"],
        "status_counts": summary["completion"]["status_counts"],
        "supplemental": summary["supplemental"],
        "proposal": summary["proposal"],
        "per_camera": summary["per_camera"],
        "image_immutability": summary["image_immutability"],
        "artifacts": {**paths,
                      "accepted_v2_root": str(accepted_root),
                      "tile_completion_manifest":
                          str(args.output / "tile_completion_manifest.jsonl"),
                      "positive_training_manifest_v2":
                          str(args.output / "positive_training_manifest_v2.jsonl")},
    }, ensure_ascii=False, indent=2))
    return 0


def _counter(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        data = _load(args)
    except (SealedAssetError, CompletionError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    if args.command == "plan":
        return cmd_plan(args, data)
    if args.command == "serve":
        return cmd_serve(args, data)
    if args.command == "status":
        return cmd_status(args, data)
    try:
        return cmd_build(args, data)
    except CompletionError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
