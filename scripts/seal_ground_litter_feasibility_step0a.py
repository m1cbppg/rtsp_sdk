#!/usr/bin/env python3
"""Detector-blind Step 0A: freeze fixed windows, then permanently seal raw PS."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_recording_source import (  # noqa: E402
    DEFAULT_FILE_URLS_ENDPOINT,
    ListQuery,
    RecordingDownloader,
    RecordingFile,
    RecordingListClient,
    RecordingSourceError,
    UrlRefreshPolicy,
    deduplicate_files,
    detect_truncation,
    file_looks_like_media,
)

TFMT = "%Y-%m-%d %H:%M:%S"
SPLITS = ("development", "sealed_test")
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, TFMT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def safe_name(value: str) -> str:
    return SAFE_NAME.sub("_", value.strip())[:220] or "recording.ps"


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema") != "ground_litter_feasibility_step0a_preregistration_v1":
        raise SystemExit("unsupported preregistration schema")
    cameras = payload.get("cameras") or []
    if len(cameras) != 5:
        raise SystemExit("Step 0A requires exactly five cameras")
    if len({row["device_code"] for row in cameras}) != 5:
        raise SystemExit("duplicated device_code")
    for row in cameras:
        if not re.fullmatch(r"\d{20}", str(row["device_code"])):
            raise SystemExit("device_code must be 20 digits")
    for split in SPLITS:
        window = payload["selection_protocol"][split]
        if parse_time(window["end"]) <= parse_time(window["start"]):
            raise SystemExit(f"invalid window for {split}")
    payload["_config_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def auth_headers(args: argparse.Namespace) -> dict[str, str]:
    headers: dict[str, str] = {}
    token = os.environ.get(args.auth_token_env, "").strip()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if token:
        headers["Authorization"] = token
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def overlaps(item: RecordingFile, start: datetime, end: datetime) -> bool:
    return parse_time(item.record_start) < end and start < parse_time(item.record_end)


def coverage(files: Sequence[RecordingFile], start: datetime, end: datetime) -> dict[str, Any]:
    spans: list[tuple[datetime, datetime]] = []
    for item in files:
        left = max(start, parse_time(item.record_start))
        right = min(end, parse_time(item.record_end))
        if right > left:
            spans.append((left, right))
    spans.sort()
    merged: list[list[datetime]] = []
    for left, right in spans:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    requested = (end - start).total_seconds()
    covered = sum((right - left).total_seconds() for left, right in merged)
    gaps: list[dict[str, Any]] = []
    cursor = start
    for left, right in merged:
        if left > cursor:
            gaps.append({
                "start": cursor.strftime(TFMT),
                "end": left.strftime(TFMT),
                "seconds": round((left - cursor).total_seconds(), 3),
            })
        cursor = max(cursor, right)
    if cursor < end:
        gaps.append({
            "start": cursor.strftime(TFMT),
            "end": end.strftime(TFMT),
            "seconds": round((end - cursor).total_seconds(), 3),
        })
    return {
        "requested_seconds": requested,
        "covered_seconds": covered,
        "coverage_fraction": round(covered / requested, 6) if requested else 0.0,
        "max_gap_seconds": max((x["seconds"] for x in gaps), default=0.0),
        "gaps": gaps,
    }


def query_fixed_window(
    client: RecordingListClient, device_code: str, start_text: str, end_text: str,
) -> tuple[list[RecordingFile], dict[str, Any]]:
    start, end = parse_time(start_text), parse_time(end_text)
    midpoint = start + (end - start) / 2
    full = client.query(ListQuery(device_code, start_text, end_text))
    left = client.query(ListQuery(device_code, start_text, midpoint.strftime(TFMT)))
    right = client.query(ListQuery(device_code, midpoint.strftime(TFMT), end_text))
    truncation = detect_truncation(full.files(), [*left.files(), *right.files()])
    files = [
        item for item in deduplicate_files([full, left, right], device_code)
        if overlaps(item, start, end)
    ]
    return files, {
        "coverage": coverage(files, start, end),
        "truncation": truncation,
        "full_count": len(full.entries),
        "half_union_count": len({x.file.file_id for x in (*left.entries, *right.entries)}),
    }


def freeze(config: dict[str, Any], args: argparse.Namespace) -> Path:
    state = args.state_dir.resolve()
    target = state / "FROZEN_MANIFEST.json"
    if target.exists():
        if args.verify_existing:
            existing = json.loads(target.read_text())
            if existing.get("preregistration_sha256") != config["_config_sha256"]:
                raise SystemExit("existing frozen manifest belongs to another config")
            return target
        raise SystemExit("FROZEN_MANIFEST.json already exists; refusing reselection")

    client = RecordingListClient(args.endpoint, headers=auth_headers(args))
    protocol = config["selection_protocol"]
    gate = protocol["coverage_gate"]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for camera in config["cameras"]:
        for split in SPLITS:
            window = protocol[split]
            try:
                files, diagnostics = query_fixed_window(
                    client, camera["device_code"], window["start"], window["end"]
                )
            except RecordingSourceError as exc:
                failures.append({
                    "camera_id": camera["camera_id"],
                    "split": split,
                    "error": str(exc)[:200],
                })
                continue
            cov = diagnostics["coverage"]
            cov_pass = (
                cov["coverage_fraction"] >= float(gate["minimum_fraction"])
                and cov["max_gap_seconds"] <= float(gate["maximum_gap_seconds"])
            )
            pad = int(protocol["training_exclusion_pad_seconds"])
            rows.append({
                "camera_id": camera["camera_id"],
                "device_code": camera["device_code"],
                "split": split,
                "window": {
                    "start": window["start"], "end": window["end"],
                    "timezone": config["timezone"], "purpose": window["purpose"],
                },
                "training_exclusion_window": {
                    "start": (parse_time(window["start"]) - timedelta(seconds=pad)).strftime(TFMT),
                    "end": (parse_time(window["end"]) + timedelta(seconds=pad)).strftime(TFMT),
                    "timezone": config["timezone"],
                    "pad_seconds": pad,
                },
                "scene_version": camera["scene_version"],
                "scene_version_basis": camera["scene_version_basis"],
                "roi": {
                    "config_path": camera["roi_config_path"],
                    "config_blob_sha": camera["roi_config_blob_sha"],
                    "geometry_version": camera["geometry_version"],
                    "reference_frame_sha256": camera["reference_frame_sha256"],
                    "canvas_size": camera["canvas_size"],
                },
                "selected_files": [x.as_dict() for x in files],
                "selected_file_count": len(files),
                "selected_declared_bytes": sum(int(x.file_size or 0) for x in files),
                "coverage": cov,
                "coverage_gate_pass": cov_pass,
                "truncation_check": diagnostics["truncation"],
                "inventory_counts": {
                    "full": diagnostics["full_count"],
                    "half_union": diagnostics["half_union_count"],
                },
            })

    if failures:
        atomic_json(state / "FREEZE_ATTEMPT_REPORT.json", {
            "schema": "ground_litter_feasibility_step0a_freeze_attempt_v1",
            "experiment_id": config["experiment_id"],
            "created_at_utc": utc_now(),
            "preregistration_sha256": config["_config_sha256"],
            "fixed_windows_unchanged": True,
            "failures": failures,
            "freeze_gate": "NO_GO_INVENTORY_QUERY",
            "overall_detector_go_no_go": "NOT_EVALUATED",
        })
        raise SystemExit("inventory failed; retry the same preregistered windows")

    seen: dict[tuple[str, str], str] = {}
    conflicts: list[dict[str, str]] = []
    for row in rows:
        for item in row["selected_files"]:
            key = (row["device_code"], item["file_id"])
            old = seen.get(key)
            if old and old != row["split"]:
                conflicts.append({
                    "device_code": key[0], "file_id": key[1],
                    "first_split": old, "second_split": row["split"],
                })
            seen[key] = row["split"]

    freeze_pass = (
        len(rows) == 10
        and all(x["coverage_gate_pass"] for x in rows)
        and not any(x["truncation_check"].get("suspected_truncation") for x in rows)
        and not conflicts
    )
    manifest = {
        "schema": "ground_litter_feasibility_step0a_frozen_manifest_v1",
        "experiment_id": config["experiment_id"],
        "frozen_at_utc": utc_now(),
        "preregistration_sha256": config["_config_sha256"],
        "baseline": config["baseline"],
        "selection_protocol": protocol,
        "detector_outputs_consulted": False,
        "rows": rows,
        "cross_split_file_conflicts": conflicts,
        "freeze_gate": "PASS" if freeze_pass else "NO_GO_DATA_CAPTURE",
        "overall_detector_go_no_go": "NOT_EVALUATED",
        "notes": [
            "Sealed Test must never be used for tuning in this round.",
            "File-level split isolation is enforced here; episode-level isolation is checked after blind truth.",
            "If one physical litter episode crosses splits, quarantine it from scored sets; never move Sealed into Training.",
            "IGNORE_SMALL contributes neither TP nor FP."
        ],
    }
    atomic_json(target, manifest)
    atomic_json(state / "FREEZE_REPORT.json", {
        "experiment_id": config["experiment_id"],
        "freeze_gate": manifest["freeze_gate"],
        "rows": [{
            "camera_id": x["camera_id"], "split": x["split"],
            "selected_file_count": x["selected_file_count"],
            "selected_declared_bytes": x["selected_declared_bytes"],
            "coverage": x["coverage"],
            "coverage_gate_pass": x["coverage_gate_pass"],
            "suspected_truncation": x["truncation_check"].get("suspected_truncation", False),
        } for x in rows],
        "cross_split_file_conflicts": conflicts,
    })
    print(json.dumps({
        "freeze_gate": manifest["freeze_gate"],
        "frozen_manifest": str(target),
        "rows": len(rows),
    }, ensure_ascii=False, indent=2))
    return target


def archive_path(root: Path, experiment: str, row: Mapping[str, Any],
                 item: Mapping[str, Any]) -> Path:
    return (
        root / experiment / row["split"] / row["camera_id"] / "raw"
        / f"{safe_name(str(item['file_id']))}__{safe_name(str(item.get('file_name') or 'recording.ps'))}"
    )


def probe_source(path: Path) -> dict[str, Any]:
    try:
        from rtsp_annotator.ground_litter_profile_sampling import probe_recording
        probe = probe_recording(path)
        return {
            "ok": bool(probe.ok), "width": int(probe.width), "height": int(probe.height),
            "duration_seconds": round(float(probe.duration_seconds), 3),
            "frame_count": int(probe.frame_count), "codec": str(probe.codec),
            "error": str(probe.error or "")[:200],
        }
    except Exception as exc:
        return {"ok": False, "error": f"probe_exception:{type(exc).__name__}"}


def materialize(config: dict[str, Any], frozen_path: Path,
                args: argparse.Namespace) -> Path:
    frozen = json.loads(frozen_path.read_text())
    if frozen["preregistration_sha256"] != config["_config_sha256"]:
        raise SystemExit("frozen manifest/config mismatch")
    if frozen["freeze_gate"] != "PASS" and not args.allow_incomplete_freeze:
        raise SystemExit("freeze_gate is not PASS")

    root = args.archive_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected_rows = [
        x for x in frozen["rows"]
        if args.split == "all" or x["split"] == args.split
    ]
    expected_bytes = sum(
        int(item.get("file_size") or 0)
        for row in selected_rows for item in row["selected_files"]
    )
    if expected_bytes and shutil.disk_usage(root).free < int(expected_bytes * 1.05):
        raise SystemExit("insufficient archive disk for declared bytes + 5% margin")

    client = RecordingListClient(args.endpoint, headers=auth_headers(args))
    downloader = RecordingDownloader(
        timeout=args.download_timeout, max_attempts=args.download_attempts
    )
    refresh_policy = UrlRefreshPolicy()
    files_out: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row in selected_rows:
        for item in row["selected_files"]:
            target = archive_path(root, frozen["experiment_id"], row, item)
            try:
                if target.exists():
                    actual_bytes = target.stat().st_size
                    if item.get("file_size") is not None and actual_bytes != int(item["file_size"]):
                        raise RuntimeError("existing archive file size mismatch")
                    digest = sha256_file(target)
                    status = "verified_existing"
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    start = parse_time(item["record_start"]) - timedelta(seconds=10)
                    end = parse_time(item["record_end"]) + timedelta(seconds=10)
                    entry = downloader.fetch_url_for_file(
                        client,
                        ListQuery(row["device_code"], start.strftime(TFMT), end.strftime(TFMT)),
                        item["file_id"],
                        policy=refresh_policy,
                    )
                    result = downloader.download(
                        entry.url, target,
                        expected_size=int(item["file_size"]) if item.get("file_size") is not None else None,
                        allow_resume=True,
                    )
                    actual_bytes, digest, status = result.size, result.sha256, "downloaded"
                if not file_looks_like_media(target):
                    raise RuntimeError("archive object does not look like media")
                identity = {
                    "schema": "ground_litter_feasibility_raw_ps_identity_v1",
                    "experiment_id": frozen["experiment_id"],
                    "split": row["split"],
                    "camera_id": row["camera_id"],
                    "device_code": row["device_code"],
                    "scene_version": row["scene_version"],
                    "roi": row["roi"],
                    "file_id": item["file_id"],
                    "file_name": item.get("file_name", ""),
                    "record_start": item["record_start"],
                    "record_end": item["record_end"],
                    "declared_bytes": item.get("file_size"),
                    "actual_bytes": actual_bytes,
                    "sha256": digest,
                    "source_probe": probe_source(target),
                    "signed_url_persisted": False,
                    "archive_relative_path": str(target.relative_to(root)),
                    "materialized_at_utc": utc_now(),
                }
                atomic_json(target.with_suffix(target.suffix + ".identity.json"), identity)
                os.chmod(target, 0o444)
                files_out.append({
                    "camera_id": row["camera_id"], "split": row["split"],
                    "file_id": item["file_id"], "status": status,
                    "archive_relative_path": identity["archive_relative_path"],
                    "bytes": actual_bytes, "sha256": digest,
                    "source_probe": identity["source_probe"],
                })
            except Exception as exc:
                failures.append({
                    "camera_id": row["camera_id"], "split": row["split"],
                    "file_id": item["file_id"], "error": str(exc)[:240],
                })

    expected_count = sum(len(x["selected_files"]) for x in selected_rows)
    complete = not failures and len(files_out) == expected_count
    state = args.state_dir.resolve()
    report = {
        "schema": "ground_litter_feasibility_step0a_materialization_report_v1",
        "experiment_id": frozen["experiment_id"],
        "created_at_utc": utc_now(),
        "frozen_manifest_sha256": sha256_file(frozen_path),
        "requested_split": args.split,
        "expected_file_count": expected_count,
        "expected_declared_bytes": expected_bytes,
        "materialized_file_count": len(files_out),
        "materialized_bytes": sum(x["bytes"] for x in files_out),
        "files": files_out,
        "failures": failures,
        "step0a_gate": "PASS" if complete else "NO_GO_DATA_CAPTURE",
        "overall_detector_go_no_go": "NOT_EVALUATED",
    }
    report_path = state / "MATERIALIZATION_REPORT.json"
    atomic_json(report_path, report)
    atomic_json(state / "SHA256SUMS.json", {
        "schema": "ground_litter_feasibility_step0a_sha256s_v1",
        "experiment_id": frozen["experiment_id"],
        "files": [{
            "sha256": x["sha256"],
            "split": x["split"],
            "camera_id": x["camera_id"],
            "file_id": x["file_id"],
            "archive_relative_path": x["archive_relative_path"],
        } for x in sorted(files_out, key=lambda x: (x["split"], x["camera_id"], x["file_id"]))],
    })

    if complete and args.split in ("all", "sealed_test"):
        marker = root / frozen["experiment_id"] / "sealed_test" / "SEALED_DO_NOT_TUNE.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            "Sealed Test: do not use recordings, frames, tiles, labels, errors, or model outputs "
            "for tuning/training/threshold/overlap/fusion selection in this round.\n"
            f"experiment_id={frozen['experiment_id']}\n"
            f"frozen_manifest_sha256={report['frozen_manifest_sha256']}\n"
        )
        os.chmod(marker, 0o444)

    print(json.dumps({
        "step0a_gate": report["step0a_gate"],
        "expected_file_count": expected_count,
        "materialized_file_count": len(files_out),
        "failures": len(failures),
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    return report_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, required=True)
    common.add_argument("--state-dir", type=Path, required=True)
    common.add_argument("--endpoint", default=DEFAULT_FILE_URLS_ENDPOINT)
    common.add_argument("--auth-token-env", default="GROUND_LITTER_RECORDING_AUTH_TOKEN")
    common.add_argument("--api-key-env", default="GROUND_LITTER_RECORDING_API_KEY")

    f = sub.add_parser("freeze", parents=[common])
    f.add_argument("--verify-existing", action="store_true")

    m = sub.add_parser("materialize", parents=[common])
    m.add_argument("--archive-root", type=Path, required=True)
    m.add_argument("--split", choices=("all", *SPLITS), default="all")
    m.add_argument("--download-timeout", type=float, default=45.0)
    m.add_argument("--download-attempts", type=int, default=3)
    m.add_argument("--allow-incomplete-freeze", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "freeze":
        freeze(config, args)
        return 0
    frozen_path = args.state_dir.resolve() / "FROZEN_MANIFEST.json"
    if not frozen_path.exists():
        raise SystemExit("run freeze first")
    materialize(config, frozen_path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
