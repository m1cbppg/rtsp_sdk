#!/usr/bin/env python3
"""Step 1D: hard negative pool from historically human-confirmed NON_LITTER cards.

Subcommands
-----------
plan      read-only: candidate universe, sampling rule, expected counts, blockers
generate  decode the raw PS and write the source-native 640x640 negative candidates
serve     run the negative-completeness review UI
status    report review progress from an existing review state
build     only when every reviewable candidate is reviewed: copy the accepted tiles
          byte-for-byte and write 0-byte YOLO labels

The candidate universe is historical human labels (NON_LITTER) plus the Step 1C-1R
reconciled NON_LITTER targets.  Nothing here creates a bbox, runs a detector, modifies
the positive pool / truth, or touches Sealed data.
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
    sha256_file,
)
from rtsp_annotator.ground_litter_hard_negatives import (  # noqa: E402
    GENERATOR_VERSION,
    PENDING,
    READY,
    REVIEW_DECISIONS,
    SCHEMA_VERSION,
    STATUS_READY,
    HardNegativeError,
    NegativeReviewState,
    apply_review,
    build_accepted,
    build_manifest,
    build_summary,
    candidate_fingerprint,
    generate_candidates,
    load_historical_input,
    load_truth_context,
    load_verified_required_boxes,
    plan_candidates,
    read_jsonl,
    verify_preflight,
    write_jsonl,
    write_outputs,
)
from rtsp_annotator.ground_litter_positive_tile_decode import (  # noqa: E402
    load_default_decoder,
)

#: Historical human review batches (Step 1A inputs) and the frozen upstream artifacts.
BATCH_DIRS = (
    ROOT / "output/ground_litter_audit_active_day3_20260918/review",
    ROOT / "output/ground_litter_audit_formal_day1",
    ROOT / "output/ground_litter_audit_formal_day2",
    ROOT / "output/ground_litter_audit_holdout_20260918",
    ROOT / "output/ground_litter_audit_pilot_01030_20260920/review",
    ROOT / "output/ground_litter_candidate_filter_shadow_20260921/review",
)
RECOVERY_ROOT = ROOT / "output/ground_litter_gold_source_recovery_20260923"
LOCALIZATION_ROOT = ROOT / "output/ground_litter_gold_localization_20260923"
TRUTH_ROOT = ROOT / "output/ground_litter_truth_reconciliation_20260923"
STEP1C2_ROOT = ROOT / "output/ground_litter_positive_tiles_20260923"
STEP1C2M_ROOT = ROOT / "output/ground_litter_positive_tile_completion_20260923"
GOLD_ROOT = ROOT / "output/ground_litter_gold_episode_review_20260923"
DEFAULT_OUTPUT = ROOT / "output/ground_litter_hard_negatives_20260923"

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
STEP1C2_MANIFEST = "abe131ec6fe4895dfdcd4c7909eff4f3b65b4b48"
STEP1C2_EVIDENCE = "c548f27e816e4cf80ffd8b7e09a862428e6c9398"
STEP1C2M_EXECUTION = "b767b01e1e11ac55dbd013296cdfd8303310b472"
STEP1C2M_MANIFEST = "8013300"
STEP1C2M_EVIDENCE = "8457ad4"

#: ultralytics' own label parser tolerates a zero-row label file; recorded verbatim so
#: the empty-label contract is auditable without importing the package (no venv here has
#: both cv2 and ultralytics).
EMPTY_LABEL_GUARD = "if nl := len(lb):"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recovery-root", type=Path, default=RECOVERY_ROOT)
    p.add_argument("--localization-root", type=Path, default=LOCALIZATION_ROOT)
    p.add_argument("--truth-root", type=Path, default=TRUTH_ROOT)
    p.add_argument("--step1c2-root", type=Path, default=STEP1C2_ROOT)
    p.add_argument("--step1c2m-root", type=Path, default=STEP1C2M_ROOT)
    p.add_argument("--gold-root", type=Path, default=GOLD_ROOT)
    p.add_argument("--batch-dir", type=Path, action="append", default=None,
                   help="historical review batch directory (repeatable)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--generated-at", default=None)
    p.add_argument("--hard-target", type=int, default=130)
    p.add_argument("--easy-target", type=int, default=40)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8776)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("generate")
    sub.add_parser("serve")
    sub.add_parser("status")
    sub.add_parser("build")
    return p


def _state_path(args: argparse.Namespace) -> Path:
    return args.state or (args.output / "review_state.json")


def _batch_dirs(args: argparse.Namespace) -> list[Path]:
    return list(args.batch_dir) if args.batch_dir else list(BATCH_DIRS)


def _load(args: argparse.Namespace) -> dict:
    return load_historical_input(
        _batch_dirs(args),
        source_files_path=args.recovery_root / "source_files.jsonl",
        reconciled_overlay_path=args.truth_root / "truth_reconciliation.jsonl",
        localization_path=args.localization_root / "localizations.jsonl",
        recovery_evidence_path=args.recovery_root / "episode_source_evidence.jsonl")


def _required_sets(args: argparse.Namespace):
    return load_verified_required_boxes(
        tile_candidates_path=args.step1c2_root / "tile_candidates.jsonl",
        completion_manifest_path=args.step1c2m_root / "tile_completion_manifest.jsonl")


def _context(args: argparse.Namespace):
    return load_truth_context(overlay_path=args.truth_root / "truth_reconciliation.jsonl",
                              localization_path=args.localization_root / "localizations.jsonl")


def _plan(args: argparse.Namespace, data: dict) -> dict:
    by_frame, by_file = _required_sets(args)
    return plan_candidates(data, required_by_frame=by_frame, required_by_file=by_file,
                           truth_context=_context(args), hard_target=args.hard_target,
                           easy_target=args.easy_target)


def _fingerprint(args: argparse.Namespace, data: dict) -> str:
    extra = [sha256_file(args.step1c2_root / "tile_candidates.jsonl")]
    steps = args.step1c2m_root / "tile_completion_manifest.jsonl"
    if steps.is_file():
        extra.append(sha256_file(steps))
    accepted = args.step1c2m_root / "positive_training_manifest_v2.jsonl"
    if accepted.is_file():
        extra.append(sha256_file(accepted))
    return candidate_fingerprint(data, extra_material=extra)


def _stamp(args: argparse.Namespace) -> str:
    return args.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty_label_check() -> dict:
    """Prove a 0-byte label file is a valid background sample for the YOLO pipeline.

    No venv in this checkout has both cv2 and ultralytics, so the installed package
    cannot be imported here.  Instead the installed source is inspected (no import) and
    its own parsing semantics are replayed on a real 0-byte file with numpy.
    """
    import importlib.util
    import re
    import tempfile

    import numpy as np

    result: dict = {
        "method": "installed_ultralytics_source_inspection_and_parser_replay",
        "guard": EMPTY_LABEL_GUARD,
        "note": ("verify_image_label() validates the label array only when it has at "
                 "least one row, so a 0-byte .txt yields 0 instances and the image is a "
                 "valid background sample"),
    }
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or not spec.origin:
        result["available"] = False
        result["reason"] = "ultralytics is not installed in this interpreter"
    else:
        package = Path(spec.origin).parent
        init = (package / "__init__.py").read_text(encoding="utf-8")
        version = re.search(r'__version__\s*=\s*"([^"]+)"', init)
        utils = package / "data" / "utils.py"
        source = utils.read_text(encoding="utf-8") if utils.is_file() else ""
        result.update({
            "available": True,
            "ultralytics_version": version.group(1) if version else "unknown",
            "verify_image_label_source": str(utils),
            "guard_present": EMPTY_LABEL_GUARD in source,
        })
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        empty_path = Path(handle.name)
    try:
        size = empty_path.stat().st_size
        # replay ultralytics' own label parsing on the real empty file
        rows = [row.split() for row in
                empty_path.read_text(encoding="utf-8").strip().splitlines() if len(row)]
        array = np.array(rows, dtype=np.float32)
        result.update({
            "empty_label_bytes": size,
            "parsed_label_shape": list(array.shape),
            "parsed_label_rows": int(len(array)),
            "validation_block_entered": bool(len(array)),
            "empty_label_is_background": size == 0 and len(array) == 0,
        })
    finally:
        empty_path.unlink(missing_ok=True)
    return result


def cmd_plan(args: argparse.Namespace, data: dict) -> int:
    plan = _plan(args, data)
    preflight = verify_preflight(data, plan)
    report = {
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "batch_dirs": [str(path) for path in _batch_dirs(args)],
        "historical_non_litter_input_count": preflight["historical_non_litter_count"],
        "reconciled_non_litter_input_count": preflight["reconciled_non_litter_count"],
        "historical_label_counts": preflight["label_counts"],
        "recoverable_count": preflight["historical_recoverable_count"],
        "recoverable_rule": ("the card (camera, timestamp) falls inside a raw PS window "
                             "already preserved by Step 1C-0"),
        "sampling_rule": {
            "hard_target": args.hard_target, "easy_target": args.easy_target,
            "easy_max_fraction": 0.30,
            "dedup_unit": "camera + source_file + minute + 64px spatial grid",
            "max_per_ten_minutes": 2,
            "hardness_counts": preflight["hardness_counts"],
        },
        "sampled_candidate_count": preflight["sampled_candidate_count"],
        "candidate_tile_count": preflight["candidate_tile_count"],
        "ready_candidate_count": preflight["ready_candidate_count"],
        "known_required_overlap_excluded":
            preflight["known_required_overlap_excluded"],
        "status_counts": preflight["status_counts"],
        "per_camera_candidate": preflight["per_camera_candidate"],
        "per_camera_recoverable": preflight["per_camera_recoverable"],
        "unique_source_ps": preflight["unique_source_ps"],
        "unique_source_frames": preflight["unique_source_frames"],
        "input_fingerprint": _fingerprint(args, data),
        "hashes": preflight["hashes"],
        "empty_label_loader_check": _empty_label_check(),
        "blockers": ([{"kind": "no_reviewable_candidate"}]
                     if not preflight["ready_candidate_count"] else []),
        "note": ("read-only: no frame is decoded, no PS is downloaded, the positive pool "
                 "and every truth artifact stay untouched"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


def cmd_generate(args: argparse.Namespace, data: dict) -> int:
    from rtsp_annotator.ground_litter_positive_tiles import read_jsonl as read_tiles

    plan = _plan(args, data)
    fingerprint = _fingerprint(args, data)
    candidates_path = args.output / "negative_candidates.jsonl"
    manifest_path = args.output / "MANIFEST.json"
    existing: list = []
    existing_fingerprint = None
    if candidates_path.is_file():
        existing = read_tiles(candidates_path)
        if manifest_path.is_file():
            existing_fingerprint = str(json.loads(
                manifest_path.read_text(encoding="utf-8")).get("input_fingerprint") or "")
    result = generate_candidates(
        data, plan, load_default_decoder(),
        args.output / "candidate_tiles" / "images",
        existing=existing, input_fingerprint=fingerprint,
        existing_fingerprint=existing_fingerprint)
    preflight = verify_preflight(data, plan)
    preflight["exact_duplicate_removed"] = result["stats"].get(
        "exact_duplicate_removed", 0)
    preflight["candidate_tile_count"] = len(result["candidates"])
    preflight["ready_candidate_count"] = sum(
        1 for row in result["candidates"]
        if row["candidate_generation_status"] == STATUS_READY)
    state = NegativeReviewState.load(_state_path(args),
                                     candidate_count=len(result["candidates"]),
                                     input_fingerprint=fingerprint)
    state.save()
    summary = build_summary(data, result["candidates"], preflight, state=state,
                            empty_label_check=_empty_label_check())
    manifest = build_manifest(
        data, summary, code_commit=git_commit_of(args.repo_root),
        generated_at=_stamp(args),
        config={"output": str(args.output), "state": str(_state_path(args)),
                "batch_dirs": [str(path) for path in _batch_dirs(args)],
                "hard_target": args.hard_target, "easy_target": args.easy_target,
                "schema_version": SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION},
        artifact_root=args.output,
        provenance=_provenance(), **_positive_pool_hashes(args))
    manifest["input_fingerprint"] = fingerprint
    paths = write_outputs(args.output, candidates=result["candidates"],
                          summary=summary, manifest=manifest)
    print(json.dumps({
        "historical_non_litter_input_count":
            summary["input"]["historical_non_litter_input_count"],
        "sampled_candidate_count": summary["input"]["sampled_candidate_count"],
        "candidate_tile_count": summary["generate"]["candidate_tile_count"],
        "ready_candidate_count": summary["generate"]["ready_candidate_count"],
        "known_required_overlap_excluded":
            summary["generate"]["known_required_overlap_excluded"],
        "exact_duplicate_removed": summary["generate"]["exact_duplicate_removed"],
        "status_counts": summary["generate"]["status_counts"],
        "risk_flag_counts": summary["generate"]["risk_flag_counts"],
        "per_camera": summary["per_camera"],
        "review": summary["review"],
        "decode_stats": result["stats"],
        "artifacts": paths,
    }, ensure_ascii=False, indent=2))
    return 0


def _positive_pool_hashes(args: argparse.Namespace) -> dict:
    manifest = args.step1c2m_root / "MANIFEST.json"
    summary = args.step1c2m_root / "SUMMARY.json"
    return {
        "positive_pool_manifest_sha256": sha256_file(manifest) if manifest.is_file() else "",
        "positive_pool_summary_sha256": sha256_file(summary) if summary.is_file() else "",
    }


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
        "step1c2_manifest_commit": STEP1C2_MANIFEST,
        "step1c2_evidence_commit": STEP1C2_EVIDENCE,
        "step1c2m_reported_execution_commit": STEP1C2M_EXECUTION,
        "step1c2m_manifest_commit": STEP1C2M_MANIFEST,
        "step1c2m_evidence_commit": STEP1C2M_EVIDENCE,
        "note": ("Step 1D only reads the artifacts recorded here; the historical review "
                 "labels are human NON_LITTER decisions, never detector output"),
    }


def cmd_serve(args: argparse.Namespace, data: dict) -> int:
    tools = ROOT / "tools" / "ground_litter_negative_ui"
    sys.path.insert(0, str(tools))
    import serve as ui  # type: ignore  # noqa: PLC0415

    candidates_path = args.output / "negative_candidates.jsonl"
    if not candidates_path.is_file():
        print("REFUSED: run generate first", file=sys.stderr)
        return 3
    args.state = str(_state_path(args))
    args.input_fingerprint = _fingerprint(args, data)
    ui.configure(args, plan=_plan(args, data), candidates=read_json(candidates_path),
                 positive_rows=_positive_rows(args))
    handler = ui.NegativeHandler
    server = ui.create_server(handler, args.host, args.port)
    print(f"Step 1D hard negative review: http://{args.host}:{args.port}/", flush=True)
    print(f"queue={len(handler.reviewable)} state={args.state}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    return 0


def read_json(path: Path) -> list:
    return read_jsonl(path)


def _positive_rows(args: argparse.Namespace) -> list:
    path = args.step1c2m_root / "positive_training_manifest_v2.jsonl"
    return read_jsonl(path) if path.is_file() else []


def cmd_status(args: argparse.Namespace, data: dict) -> int:
    plan = _plan(args, data)
    candidates_path = args.output / "negative_candidates.jsonl"
    if not candidates_path.is_file():
        print("REFUSED: run generate first", file=sys.stderr)
        return 3
    candidates = read_jsonl(candidates_path)
    state = NegativeReviewState.load(_state_path(args), candidate_count=len(candidates))
    rows = apply_review(candidates, state)
    counts = {decision: 0 for decision in REVIEW_DECISIONS}
    accepted = 0
    for row in rows:
        if row.get("candidate_generation_status") != STATUS_READY:
            continue
        status = row.get("review_status") or PENDING
        if status in counts:
            counts[status] += 1
        if row.get("hard_negative_ready"):
            accepted += 1
    progress = state.progress(candidates)
    print(json.dumps({
        "artifact_root": str(args.output),
        "state": str(_state_path(args)),
        **progress,
        "review_status_counts": counts,
        "would_be_accepted_tile_count": accepted,
        "build_allowed": progress["pending"] == 0 and progress["skipped"] == 0,
        "candidate_total": len(candidates),
        "planned_ready_candidate_count": plan["status_counts"].get(STATUS_READY, 0),
        "generation_status_counts": _counter(
            str(row.get("candidate_generation_status")) for row in rows),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_build(args: argparse.Namespace, data: dict) -> int:
    candidates = read_jsonl(args.output / "negative_candidates.jsonl")
    state = NegativeReviewState.load(_state_path(args), candidate_count=len(candidates))
    accepted = build_accepted(
        candidates, args.output / "accepted", state=state,
        positive_rows=_positive_rows(args),
        positive_tiles_path=args.step1c2_root / "tile_candidates.jsonl")
    rows = apply_review(candidates, state)
    plan = _plan(args, data)
    preflight = verify_preflight(data, plan)
    summary = build_summary(data, candidates, preflight, state=state,
                            accepted=accepted, empty_label_check=_empty_label_check())
    manifest = build_manifest(
        data, summary, code_commit=git_commit_of(args.repo_root),
        generated_at=_stamp(args),
        config={"output": str(args.output), "state": str(_state_path(args)),
                "schema_version": SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION},
        artifact_root=args.output, provenance=_provenance(),
        **_positive_pool_hashes(args))
    manifest["input_fingerprint"] = _fingerprint(args, data)
    paths = write_outputs(args.output, candidates=rows, summary=summary,
                          manifest=manifest)
    manifest_rows = args.output / "hard_negative_training_manifest.jsonl"
    write_jsonl(manifest_rows, accepted["rows"])
    images = sorted((args.output / "accepted" / "images").glob("*.png"))
    labels = sorted((args.output / "accepted" / "labels").glob("*.txt"))
    if len(images) != accepted["accepted_tile_count"] or \
            len(labels) != accepted["accepted_tile_count"]:
        raise HardNegativeError("accepted image/label count mismatch")
    for row in accepted["rows"]:
        if sha256_file(row["image_path"]) != row["image_sha256"]:
            raise HardNegativeError(f"{row['negative_tile_id']}: accepted image changed")
        if Path(row["label_path"]).stat().st_size != 0:
            raise HardNegativeError(f"{row['negative_tile_id']}: label must be 0 bytes")
    print(json.dumps({
        "accepted_hard_negative_tile_count": accepted["accepted_tile_count"],
        "accepted_label_count": 0,
        "review_counts": summary["review"]["counts"],
        "positive_conflict_count": summary["accepted"]["positive_conflict_count"],
        "per_camera": summary["per_camera"],
        "unique_source_ps": summary["generate"]["unique_source_ps"],
        "empty_label_verified_count":
            summary["accepted"]["empty_label_verified_count"],
        "artifacts": {**paths, "accepted_root": str(args.output / "accepted"),
                      "hard_negative_training_manifest": str(manifest_rows)},
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
    except (SealedAssetError, HardNegativeError) as exc:
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
    except HardNegativeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
