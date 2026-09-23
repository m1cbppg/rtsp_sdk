#!/usr/bin/env python3
"""Step 1C-2: source-native 640x640 positive training tiles + annotation completeness.

Subcommands
-----------
plan      read-only: eligibility preflight, per camera, expected candidates, blockers
generate  decode the raw PS and write the 640x640 source-native candidate tiles
serve     run the annotation-completeness review UI (no way to draw a box)
status    report review progress from an existing review state
build     only when every reviewable tile is reviewed: copy the accepted tiles
          byte-for-byte and emit YOLO one-class labels

Every upstream artifact (Gold, recovery evidence, localization, truth
reconciliation, training manifest) is read-only.  Nothing in this step creates,
moves, clips or scales a bbox, and no detector / segmentation / training runs.
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
from rtsp_annotator.ground_litter_positive_tiles import (  # noqa: E402
    GENERATOR_VERSION,
    MIN_LABEL_MARGIN_PX,
    REVIEW_DECISIONS,
    SCHEMA_VERSION,
    STATUS_READY,
    TILE_SIZE,
    PositiveTileError,
    SealedAssetError,
    TileReviewState,
    apply_review,
    build_accepted,
    build_manifest,
    build_summary,
    candidate_fingerprint,
    cluster_same_frame,
    generate_candidates,
    load_positive_tile_input,
    read_jsonl,
    sha256_file,
    verify_preflight,
    write_jsonl,
    write_outputs,
)

TRUTH_ROOT = ROOT / "output" / "ground_litter_truth_reconciliation_20260923"
RECOVERY_ROOT = ROOT / "output" / "ground_litter_gold_source_recovery_20260923"
LOCALIZATION_ROOT = ROOT / "output" / "ground_litter_gold_localization_20260923"
GOLD_ROOT = ROOT / "output" / "ground_litter_gold_episode_review_20260923"
DEFAULT_OUTPUT = ROOT / "output" / "ground_litter_positive_tiles_20260923"

#: Upstream commits, recorded so no step has to infer them (§47 pattern).
#: ``*_EXECUTION`` introduced the code, ``*_MANIFEST`` is what that step's own MANIFEST
#: records as its ``code_commit``, ``*_EVIDENCE`` added the evidence file.
STEP1C0_EXECUTION = "2e5da415c20feda3384e080a3ff9217c97c4e83c"
STEP1C0_MANIFEST = "81c3dde27f7ed759248330e3c332d78907a137e3"
STEP1C0_EVIDENCE = "bfdecd5"
STEP1C1_EXECUTION = "809ce49196dadc639fe90438da3c27057baa34bd"
STEP1C1_MANIFEST = "806f8de4cb8b750c1c68aa262571d59ea1c8d0ba"
STEP1C1_EVIDENCE = "1a1f65c14a5e971baa00eb36774e9c85911124d8"
STEP1C1R_EXECUTION = "5399c4ff972b3f6bff02ba66bc861bc0b684b778"
STEP1C1R_MANIFEST = "d841864f3a997347bf53db9cf24c2d1940052ac6"
STEP1C1R_EVIDENCE = "6c60bb8120161f9f735b00a465cc49cb426369c2"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--training-manifest",
                   type=Path, default=TRUTH_ROOT / "training_episode_manifest.jsonl")
    p.add_argument("--truth-overlay", type=Path,
                   default=TRUTH_ROOT / "truth_reconciliation.jsonl")
    p.add_argument("--localization", type=Path,
                   default=LOCALIZATION_ROOT / "localizations.jsonl")
    p.add_argument("--recovery-evidence", type=Path,
                   default=RECOVERY_ROOT / "episode_source_evidence.jsonl")
    p.add_argument("--source-files", type=Path,
                   default=RECOVERY_ROOT / "source_files.jsonl")
    p.add_argument("--gold", type=Path, default=GOLD_ROOT / "gold_episodes.jsonl")
    p.add_argument("--gold-manifest", type=Path, default=GOLD_ROOT / "MANIFEST.json")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--generated-at", default=None)
    p.add_argument("--tile-size", type=int, default=TILE_SIZE)
    p.add_argument("--margin", type=float, default=MIN_LABEL_MARGIN_PX)
    p.add_argument("--limit", type=int, default=None,
                   help="generate only the first N eligible episodes (small-scale test)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8774)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("generate")
    sub.add_parser("serve")
    sub.add_parser("status")
    sub.add_parser("build")
    return p


def _images_dir(args: argparse.Namespace) -> Path:
    return args.output / "candidate_tiles" / "images"


def _accepted_root(args: argparse.Namespace) -> Path:
    return args.output / "accepted"


def _state_path(args: argparse.Namespace) -> Path:
    return args.state or (args.output / "review_state.json")


def _load(args: argparse.Namespace) -> dict:
    return load_positive_tile_input(
        training_manifest_path=args.training_manifest,
        truth_overlay_path=args.truth_overlay,
        localization_path=args.localization,
        recovery_evidence_path=args.recovery_evidence,
        source_files_path=args.source_files,
        gold_path=args.gold,
        gold_manifest=args.gold_manifest,
    )


def _stamp(args: argparse.Namespace) -> str:
    return args.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _provenance(note: str = "") -> dict:
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
        "note": note or ("upstream commit pairs recorded so no downstream step infers "
                         "them; each step's MANIFEST records the git HEAD at the moment "
                         "its evidence was written"),
    }


def _existing_candidates(args: argparse.Namespace) -> tuple[list, str, dict]:
    path = args.output / "tile_candidates.jsonl"
    manifest_path = args.output / "MANIFEST.json"
    if not path.is_file():
        return [], "", {}
    fingerprint = ""
    intervals: dict = {}
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            fingerprint = str(payload.get("input_fingerprint") or "")
            intervals = dict(payload.get("source_frame_intervals") or {})
        except json.JSONDecodeError:
            fingerprint = ""
    return read_jsonl(path), fingerprint, intervals


def cmd_plan(args: argparse.Namespace, data: dict) -> int:
    import collections

    preflight = verify_preflight(data)
    episodes = data["episodes"]
    groups = cluster_same_frame(episodes)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    screening_by_frame: dict[tuple[str, str], list] = {}
    for target in data["screening"]:
        key = (str(target["source_file_id"]), str(target["step1c0_decoded_timestamp"]))
        screening_by_frame.setdefault(key, []).append(target)
    unlocalized_frames = {k for k, v in screening_by_frame.items()
                          if any(t["kind"] == "UNLOCALIZED_REQUIRED" for t in v)}
    blockers = [
        {"kind": "eligibility", **problem} for problem in preflight["eligibility_problems"]
    ]
    if preflight["training_eligible_episode_count"] != 109:
        blockers.append({"kind": "unexpected_eligible_count",
                         "value": preflight["training_eligible_episode_count"]})
    report = {
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "gold_sha256": data["gold_sha256"],
        "recovery_evidence_sha256": data["recovery_evidence_sha256"],
        "localization_sha256": data["localization_sha256"],
        "truth_reconciliation_sha256": data["truth_overlay_sha256"],
        "training_episode_manifest_sha256": data["training_manifest_sha256"],
        "source_files_sha256": data["source_files_sha256"],
        "input_fingerprint": candidate_fingerprint(data, size=args.tile_size),
        "preflight": preflight,
        "tile_size": args.tile_size,
        "min_label_margin_px": args.margin,
        "class_mapping": {"0": "ground_litter"},
        "eligible_episode_count": len(episodes),
        "unique_source_ps_count": preflight["unique_source_ps_count"],
        "per_camera_eligible": preflight["per_camera"],
        "expected_candidate_count": len(episodes),
        "distinct_frames": len(groups),
        "multi_label_frame_count": len(multi),
        "multi_label_frame_sizes": dict(sorted(collections.Counter(
            len(v) for v in groups.values()).items())),
        "frames_with_unlocalized_required": len(unlocalized_frames),
        "screening_by_kind": preflight["screening_by_kind"],
        "source_resolutions": preflight["source_resolutions"],
        "blockers": blockers,
        "note": ("plan is read-only: no image is decoded and no candidate is written; "
                 "the 35 REQUIRED episodes without a verified bbox are not processed"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


def cmd_generate(args: argparse.Namespace, data: dict) -> int:
    from rtsp_annotator.ground_litter_positive_tile_decode import load_default_decoder

    if args.limit:
        data = dict(data)
        data["episodes"] = data["episodes"][: args.limit]
    fingerprint = candidate_fingerprint(data, size=args.tile_size)
    existing, existing_fingerprint, known_intervals = _existing_candidates(args)
    result = generate_candidates(
        data, load_default_decoder(), _images_dir(args), size=args.tile_size,
        margin=args.margin, existing=existing, input_fingerprint=fingerprint,
        existing_fingerprint=existing_fingerprint or None,
        known_intervals=known_intervals)
    candidates = result["candidates"]
    preflight = verify_preflight(data)
    state = TileReviewState.load(_state_path(args), candidate_count=len(candidates),
                                 input_fingerprint=fingerprint)
    state.save()
    summary = build_summary(data, candidates, preflight, state=state,
                            merges=result["merges"], decoded=True,
                            stats={**result["stats"],
                                   "pre_dedup_count": result["pre_dedup_count"]})
    manifest = build_manifest(
        data, summary, code_commit=git_commit_of(args.repo_root),
        generated_at=_stamp(args),
        config={"tile_size": args.tile_size, "margin": args.margin,
                "output": str(args.output), "state": str(_state_path(args)),
                "images_dir": str(_images_dir(args)),
                "accepted_root": str(_accepted_root(args)),
                "limit": args.limit,
                "schema_version": SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION},
        artifact_root=args.output,
        candidate_index_path=args.output / "tile_candidates.jsonl",
        review_state_path=_state_path(args), provenance=_provenance())
    manifest["input_fingerprint"] = fingerprint
    manifest["source_frame_intervals"] = {
        key: round(value, 6) for key, value in sorted(result["intervals"].items())}
    paths = write_outputs(args.output, candidates=candidates, summary=summary,
                          manifest=manifest)
    print(json.dumps({
        "input_training_eligible_episode_count":
            summary["input"]["training_eligible_episode_count"],
        "candidate_tile_count": summary["generate"]["candidate_tile_count"],
        "deduplicated_tile_count": summary["generate"]["deduplicated_tile_count"],
        "merged_duplicate_tile_count": summary["generate"]["merged_duplicate_tile_count"],
        "generation_failed_count": summary["generate"]["generation_failed_count"],
        "multi_label_tile_count": summary["generate"]["multi_label_tile_count"],
        "status_counts": summary["generate"]["status_counts"],
        "review": summary["review"],
        "per_camera": summary["per_camera"],
        "artifacts": paths,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace, data: dict) -> int:
    tools = ROOT / "tools" / "ground_litter_tile_ui"
    sys.path.insert(0, str(tools))
    import serve as ui  # type: ignore  # noqa: PLC0415

    candidates_path = args.output / "tile_candidates.jsonl"
    if not candidates_path.is_file():
        print("REFUSED: run generate first", file=sys.stderr)
        return 3
    args.state = str(_state_path(args))
    ui.configure(args, data, read_jsonl(candidates_path))
    server = ui.create_server(ui.ReviewHandler, args.host, args.port)
    print(f"Step 1C-2 annotation-complete review: http://{args.host}:{args.port}/",
          flush=True)
    print(f"reviewable={len(ui.ReviewHandler.reviewable)} state={args.state}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    return 0


def _candidate_context(args: argparse.Namespace) -> tuple[list, TileReviewState]:
    candidates = read_jsonl(args.output / "tile_candidates.jsonl")
    state = TileReviewState.load(_state_path(args), candidate_count=len(candidates))
    return candidates, state


def cmd_status(args: argparse.Namespace, data: dict) -> int:
    candidates, state = _candidate_context(args)
    rows = apply_review(candidates, state)
    progress = state.progress(candidates)
    counts = {}
    for row in rows:
        if row.get("candidate_generation_status") == STATUS_READY:
            key = str(row.get("annotation_review_status"))
            counts[key] = counts.get(key, 0) + 1
    ready = sum(1 for row in rows if row.get("positive_training_ready"))
    print(json.dumps({
        "artifact_root": str(args.output),
        "review_state": str(_state_path(args)),
        **progress,
        "review_status_counts": dict(sorted(counts.items())),
        "would_be_accepted_tile_count": ready,
        "build_allowed": progress["pending"] == 0 and progress["skipped"] == 0,
        "generation_status_counts": _counter(
            str(row.get("candidate_generation_status")) for row in rows),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_build(args: argparse.Namespace, data: dict) -> int:
    candidates, state = _candidate_context(args)
    accepted = build_accepted(candidates, _images_dir(args), _accepted_root(args),
                              state=state)
    rows = apply_review(candidates, state)
    preflight = verify_preflight(data)
    summary = build_summary(data, rows, preflight, state=state, decoded=True,
                            accepted=accepted, stats={"pre_dedup_count": len(rows)})
    manifest = build_manifest(
        data, summary, code_commit=git_commit_of(args.repo_root),
        generated_at=_stamp(args),
        config={"tile_size": args.tile_size, "margin": args.margin,
                "output": str(args.output), "state": str(_state_path(args)),
                "images_dir": str(_images_dir(args)),
                "accepted_root": str(_accepted_root(args)),
                "schema_version": SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION},
        artifact_root=args.output,
        candidate_index_path=args.output / "tile_candidates.jsonl",
        review_state_path=_state_path(args), provenance=_provenance())
    manifest["input_fingerprint"] = candidate_fingerprint(data, size=args.tile_size)
    paths = write_outputs(args.output, candidates=rows, summary=summary,
                          manifest=manifest)
    manifest_path = args.output / "positive_training_manifest.jsonl"
    write_jsonl(manifest_path, accepted["rows"])
    accepted_images = sorted((_accepted_root(args) / "images").glob("*.png"))
    accepted_labels = sorted((_accepted_root(args) / "labels").glob("*.txt"))
    if len(accepted_images) != accepted["accepted_tile_count"] or \
            len(accepted_labels) != accepted["accepted_tile_count"]:
        raise PositiveTileError("accepted image/label count does not match the manifest")
    for row in accepted["rows"]:
        if sha256_file(row["image_path"]) != row["image_sha256"]:
            raise PositiveTileError(f"{row['tile_id']} accepted image hash mismatch")
    print(json.dumps({
        "reviewed": summary["review"]["reviewed"],
        "pending": summary["review"]["pending"],
        "accepted_positive_tile_count": accepted["accepted_tile_count"],
        "accepted_label_count": accepted["accepted_label_count"],
        "review_counts": summary["review"]["counts"],
        "episodes_lost_due_annotation_incomplete":
            summary["accepted"]["episodes_lost_due_annotation_incomplete"],
        "per_camera": summary["per_camera"],
        "artifacts": {**paths, "positive_training_manifest": str(manifest_path)},
    }, ensure_ascii=False, indent=2))
    return 0


def _counter(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        data = _load(args)
    except (SealedAssetError, PositiveTileError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    if args.command == "plan":
        return cmd_plan(args, data)
    if args.command == "generate":
        return cmd_generate(args, data)
    if args.command == "serve":
        return cmd_serve(args, data)
    if args.command == "status":
        return cmd_status(args, data)
    try:
        return cmd_build(args, data)
    except PositiveTileError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
