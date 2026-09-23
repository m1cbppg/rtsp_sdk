#!/usr/bin/env python3
"""Step 1C-0: recover the original PS behind human-confirmed Gold episodes.

Subcommands
-----------
plan    read-only: how many Required episodes, known file_ids, expected downloads
run     resolve -> download -> SHA256 -> decode -> verification frame -> evidence
status  report progress from an existing artifact root (no network)

Gold (``gold_episodes.jsonl``) is an immutable input: it is hashed at the start and
re-hashed at the end of every run.  The recovered PS are artifact-owned and are never
released or deleted by this tool.
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
from rtsp_annotator.ground_litter_gold_source_recovery import (  # noqa: E402
    RECOVERY_STATUSES,
    GoldChangedError,
    GoldSourceRecovery,
    RecoveryConfig,
    RecoveryError,
    SealedAssetError,
    build_manifest,
    build_summary,
    load_device_codes,
    load_gold,
    resolve_episode,
    sha256_file,
)

DEFAULT_GOLD = ROOT / "output" / "ground_litter_gold_episode_review_20260923" / "gold_episodes.jsonl"
DEFAULT_GOLD_MANIFEST = ROOT / "output" / "ground_litter_gold_episode_review_20260923" / "MANIFEST.json"
DEFAULT_ROI_CONFIG_DIR = ROOT / "output" / "ground_litter_final_roi_20260922" / "config"
DEFAULT_OUTPUT = ROOT / "output" / "ground_litter_gold_source_recovery_20260923"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--gold-manifest", type=Path, default=DEFAULT_GOLD_MANIFEST)
    p.add_argument("--device-map-source", type=Path, action="append", default=None,
                   help="file or directory carrying camera_id + device_code "
                        "(default: the final-ROI config directory)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--endpoint", default=None,
                   help="override the recording file-urls endpoint")
    p.add_argument("--episode", action="append", default=None,
                   help="restrict to specific episode_id (repeatable)")
    p.add_argument("--camera", action="append", default=None,
                   help="restrict to specific camera_id (repeatable)")
    p.add_argument("--limit", type=int, default=None, help="process at most N episodes")
    p.add_argument("--max-downloads", type=int, default=None)
    p.add_argument("--verify-hashes-on-resume", action="store_true")
    p.add_argument("--generated-at", default=None)
    sub = p.add_subparsers(dest="command", required=True)
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--resolve", action="store_true",
                        help="also query remote metadata (read-only, no downloads)")
    p_run = sub.add_parser("run")
    p_run.add_argument("--force", action="store_true",
                       help="re-derive episodes that are already recovered")
    sub.add_parser("status")
    return p


def _stamp(args: argparse.Namespace) -> str:
    return args.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _select(args: argparse.Namespace, episodes):
    chosen = list(episodes)
    if args.episode:
        wanted = set(args.episode)
        chosen = [e for e in chosen if e.episode_id in wanted]
        unknown = wanted - {e.episode_id for e in chosen}
        if unknown:
            raise SystemExit(f"unknown episode ids: {sorted(unknown)}")
    if args.camera:
        wanted = set(args.camera)
        chosen = [e for e in chosen if e.camera_id in wanted]
    chosen.sort(key=lambda e: (e.camera_id, e.episode_id))
    if args.limit is not None:
        chosen = chosen[:max(0, args.limit)]
    return chosen


def _config(args: argparse.Namespace) -> RecoveryConfig:
    return RecoveryConfig(
        artifact_root=args.output.resolve(),
        gold_path=args.gold,
        gold_manifest=args.gold_manifest,
        endpoint=args.endpoint,
        max_downloads=args.max_downloads,
        verify_hashes_on_resume=args.verify_hashes_on_resume,
    )


def cmd_plan(args: argparse.Namespace, gold) -> int:
    chosen = _select(args, gold.required)
    known = [e for e in chosen if e.has_known_file_id]
    unknown = [e for e in chosen if not e.has_known_file_id]
    unique_ids = sorted({f for e in known for f in e.hypothesised_file_ids})
    report = {
        "gold_artifact": str(gold.path),
        "gold_sha256": gold.sha256,
        "gold_code_commit": gold.code_commit,
        "required_episode_count": gold.required_count,
        "selected_episode_count": len(chosen),
        "known_file_id_episodes": len(known),
        "missing_file_id_episodes": len(unknown),
        "unique_known_file_ids": len(unique_ids),
        "manual_missing_targets_in_scope": sum(1 for e in chosen if e.is_manual_missing_target),
        "per_camera": {},
        "expected_downloads_if_all_resolved": len(unique_ids),
    }
    for episode in chosen:
        bucket = report["per_camera"].setdefault(
            episode.camera_id, {"required": 0, "known_file_id": 0, "missing_file_id": 0})
        bucket["required"] += 1
        bucket["known_file_id" if episode.has_known_file_id else "missing_file_id"] += 1

    if args.resolve:
        from rtsp_annotator.ground_litter_gold_source_recovery import HttpListing
        listing = HttpListing(args.endpoint)
        resolved: dict[str, dict[str, int]] = {}
        unique_downloads: set[str] = set()
        for episode in chosen:
            outcome = resolve_episode(episode, listing)
            resolved[outcome.status] = resolved.get(outcome.status, 0) + 1
            if outcome.recording is not None:
                unique_downloads.add(outcome.recording.file_id)
        report["resolution_without_download"] = dict(sorted(resolved.items()))
        report["expected_unique_downloads"] = len(unique_downloads)

    plan_path = args.output / "plan.json"
    from rtsp_annotator.ground_litter_gold_source_recovery import atomic_write_json
    atomic_write_json(plan_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace, gold) -> int:
    from rtsp_annotator.ground_litter_gold_source_decode import load_default_adapter
    from rtsp_annotator.ground_litter_gold_source_recovery import HttpDownloader, HttpListing

    chosen = _select(args, gold.required)
    if not chosen:
        raise SystemExit("no episodes selected")

    listing = HttpListing(args.endpoint)
    downloader = HttpDownloader(listing)
    recovery = GoldSourceRecovery(
        _config(args), listing, fetcher=downloader.fetch, decoder=load_default_adapter())

    def report(episode, evidence) -> None:
        print(f"{episode.episode_id} {episode.camera_id} "
              f"{evidence.source_recovery_status} {evidence.reason_code}", flush=True)

    # run() applies the resume policy: an episode already backed by a preserved PS
    # and an on-disk verification frame is never re-derived.
    recovery.run(chosen, on_episode=report, force=args.force)
    recovery.save()

    # §4: prove this step did not modify the Gold artifact
    after = sha256_file(gold.path)
    if after != gold.sha256:
        raise GoldChangedError(
            f"Gold artifact changed during the run ({gold.sha256[:12]} -> {after[:12]})")

    frames = len(list((args.output / "verification_frames").rglob("*.jpg")))
    evidence = list(recovery.evidence.values())
    summary = build_summary(gold, evidence, recovery.files,
                            extra={"selected_episode_count": len(chosen),
                                   "gold_sha256_after_run": after})
    manifest = build_manifest(
        gold, summary, code_commit=git_commit_of(args.repo_root),
        generated_at=_stamp(args), config=_config(args).as_dict(),
        artifact_root=args.output, evidence_path=recovery.evidence_path,
        verification_frame_count=frames)
    from rtsp_annotator.ground_litter_gold_source_recovery import atomic_write_json
    from rtsp_annotator.ground_litter_gold_source_recovery import SCHEMA_VERSION, GENERATOR_VERSION
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["generator_version"] = GENERATOR_VERSION
    atomic_write_json(args.output / "SUMMARY.json", summary)
    atomic_write_json(args.output / "MANIFEST.json", manifest)

    print(json.dumps({
        "required": summary["gold"]["required_episode_count"],
        "completed": summary["episodes"]["completed_episodes"],
        "recovered": summary["episodes"]["RECOVERED_SOURCE_NATIVE"],
        "unique_source_files_downloaded": summary["files"]["unique_source_files_downloaded"],
        "total_ps_bytes_preserved": summary["files"]["total_ps_bytes_preserved"],
        "gold_sha256_unchanged": after == gold.sha256,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_status(args: argparse.Namespace, gold) -> int:
    from rtsp_annotator.ground_litter_gold_source_recovery import read_jsonl

    root = args.output
    evidence = read_jsonl(root / "episode_source_evidence.jsonl")
    files = read_jsonl(root / "source_files.jsonl")
    counts = {status: 0 for status in RECOVERY_STATUSES}
    for row in evidence:
        key = str(row.get("source_recovery_status"))
        counts[key] = counts.get(key, 0) + 1
    preserved = sum(int(f.get("size_bytes") or 0) for f in files)
    frames = len(list((root / "verification_frames").rglob("*.jpg")))
    print(json.dumps({
        "artifact_root": str(root),
        "gold_required": gold.required_count,
        "completed_episodes": len(evidence),
        "remaining_episodes": gold.required_count - len(evidence),
        "unique_source_files": len(files),
        "unique_ps_preserved": sum(1 for f in files
                                   if f.get("recovery_status") in ("DOWNLOADED", "PRESERVED")),
        "total_ps_bytes_preserved": preserved,
        "verification_frames": frames,
        **counts,
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    sources = args.device_map_source or [DEFAULT_ROI_CONFIG_DIR]
    try:
        device_codes = load_device_codes(*sources)
        gold = load_gold(args.gold, manifest_path=args.gold_manifest,
                         device_codes=device_codes)
    except (SealedAssetError, RecoveryError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    missing_cameras = sorted({e.camera_id for e in gold.required} - set(device_codes))
    if missing_cameras:
        print(f"REFUSED: no device_code for cameras {missing_cameras}; "
              f"searched {[str(s) for s in sources]}", file=sys.stderr)
        return 3
    if args.command == "plan":
        return cmd_plan(args, gold)
    if args.command == "run":
        try:
            return cmd_run(args, gold)
        except GoldChangedError as exc:
            print(f"HARD FAIL: {exc}", file=sys.stderr)
            return 4
        except SealedAssetError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 3
    return cmd_status(args, gold)


if __name__ == "__main__":
    raise SystemExit(main())
