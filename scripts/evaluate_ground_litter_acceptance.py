"""Evaluate explicitly reviewed ground-litter truth against immutable log snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtsp_annotator.ground_litter_acceptance import evaluate, prepare_review, read_logs, render_report, sha256_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--logs", type=Path, nargs="+", required=True,
                        help="Complete JSONL snapshots, oldest first; include numeric rotations")
    parser.add_argument("--media", type=Path, help="Original local recording for SHA-256 verification")
    parser.add_argument("--prepare-review", action="store_true",
                        help="Import log IDs as uncertain and produce a draft; does not label truth")
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    args = parser.parse_args(argv)
    rows, fingerprints = read_logs(args.logs)
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    if args.prepare_review:
        truth = prepare_review(truth, rows, fingerprints)
    report = evaluate(truth, rows, fingerprints,
                      media_sha256=sha256_file(args.media) if args.media else None)
    report["input_fingerprints"] = {"logs": fingerprints, "truth_sha256": sha256_file(args.truth)}
    args.output.mkdir(parents=True, exist_ok=False)
    if args.prepare_review:
        (args.output / "review-draft.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.md")}, ensure_ascii=False))
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
