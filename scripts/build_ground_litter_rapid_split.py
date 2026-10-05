#!/usr/bin/env python3
"""Freeze the Ground Litter Rapid Eval v1 split and review-frame manifest.

Development-only, four cameras (01021/01022/01027/01030).  Reads the frozen Development
inventory and the official Step 2C blind-truth artifact **read-only**, and writes nothing
outside ``--output-dir``.

    python scripts/build_ground_litter_rapid_split.py \
        --manifest ~/step2c1-blind-truth/artifact/development_manifest.json \
        --truth ~/step2c1-blind-truth/artifact/truth_objects.jsonl \
        --review-state ~/step2c1-blind-truth/artifact/review_state.json \
        --exploratory-selection output/fourcam-exploratory-20260929/selection.json \
        --output-dir ~/ground-litter-rapid-v1/artifact
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtsp_annotator.ground_litter_rapid import (  # noqa: E402
    CAMERAS,
    SEED,
    TRAIN_SPLIT,
    RapidError,
    assert_writable_root,
    build_frame_manifest,
    build_split,
    canonical_sha256,
    read_jsonl,
    verify_split,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="frozen Development inventory (read-only)")
    parser.add_argument("--truth", type=Path, required=True,
                        help="official truth_objects.jsonl (read-only)")
    parser.add_argument("--review-state", type=Path, default=None,
                        help="official review_state.json (read-only)")
    parser.add_argument("--exploratory-selection", type=Path, required=True,
                        help="output/fourcam-exploratory-20260929/selection.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default=SEED)
    return parser.parse_args(argv)


def load_prior_inference(selection_path: Path) -> list[str]:
    rows = json.loads(selection_path.read_text(encoding="utf-8"))
    file_ids = []
    for row in rows:
        file_id = row.get("file_id")
        if file_id and file_id not in file_ids:
            file_ids.append(file_id)
    if not file_ids:
        raise RapidError(f"no file_id found in {selection_path}")
    return file_ids


def bonus_frames_from_truth(truth_rows, by_file, split) -> list[dict]:
    """Existing human REQUIRED_LITTER points on Rapid-Train PS become bonus train frames."""
    grouped: dict[tuple[str, int], dict] = {}
    for mark in truth_rows:
        if mark.get("truth_class") != "REQUIRED_LITTER":
            continue
        if not mark.get("in_roi", True):
            continue
        camera = mark.get("camera_id")
        file_id = mark.get("source_file_id")
        if camera not in CAMERAS or file_id not in by_file:
            continue
        row = by_file[file_id]
        if row["split"] != TRAIN_SPLIT:
            raise RapidError(
                f"human REQUIRED point {mark.get('truth_id')} sits on a Rapid-Eval PS: "
                f"{file_id}"
            )
        key = (file_id, int(mark["frame_index"]))
        entry = grouped.setdefault(key, {
            "file_id": file_id,
            "frame_index": int(mark["frame_index"]),
            "decoded_timestamp": float(mark["decoded_timestamp"]),
            "reason": "existing_human_required_point",
            "truth_ids": [],
        })
        entry["truth_ids"].append(mark.get("truth_id"))
    return [grouped[key] for key in sorted(grouped, key=lambda k: (k[0], k[1]))]


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = assert_writable_root(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("split") not in (None, "development"):
        raise RapidError(f"manifest is not the Development split: {manifest.get('split')}")
    files = [row for row in manifest["files"] if row["camera_id"] in CAMERAS]
    prior = load_prior_inference(args.exploratory_selection)
    split = build_split(files, prior, seed=args.seed)
    verify_split(split)

    by_file = {row["file_id"]: row for row in split["rows"]}
    truth_rows = read_jsonl(args.truth)
    bonus = bonus_frames_from_truth(truth_rows, by_file, split)
    frame_manifest = build_frame_manifest(split, bonus)

    write_json(out_dir / "split.json", split)
    with open(out_dir / "frame_manifest.jsonl", "w", encoding="utf-8") as handle:
        for row in frame_manifest["frames"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(out_dir / "frame_manifest.meta.json", {
        "schema_version": frame_manifest["schema_version"],
        "seed": frame_manifest["seed"],
        "split_sha256": frame_manifest["split_sha256"],
        "manifest_sha256": frame_manifest["manifest_sha256"],
        "counts": frame_manifest["counts"],
        "bonus_frames": [f["frame_id"] for f in frame_manifest["frames"]
                         if f["kind"] == "bonus_train"],
    })

    review_note = None
    if args.review_state:
        state = json.loads(args.review_state.read_text(encoding="utf-8"))
        reviewed = sum(1 for v in (state.get("files") or {}).values()
                       if v.get("status") == "REVIEWED")
        review_note = {"official_reviewed_ps": reviewed,
                       "official_total_ps": len(state.get("files") or {})}

    # Seed every bonus train frame with the human Required points already recorded during the
    # official Discovery session, so the operator only has to confirm and localize them.
    inherited_dir = out_dir / "inherited_truth"
    inherited_dir.mkdir(parents=True, exist_ok=True)
    for frame in frame_manifest["frames"]:
        if frame["kind"] != "bonus_train":
            continue
        marks = [m for m in truth_rows
                 if m.get("source_file_id") == frame["file_id"]
                 and int(m.get("frame_index", -1)) == frame["nominal_frame_index"]
                 and m.get("truth_class") == "REQUIRED_LITTER"
                 and m.get("in_roi", True)]
        with open(inherited_dir / f'{frame["frame_id"]}.jsonl', "w", encoding="utf-8") as handle:
            for mark in marks:
                handle.write(json.dumps({
                    "truth_id": mark.get("truth_id"),
                    "source_xy": mark.get("source_point"),
                    "truth_class": mark.get("truth_class"),
                    "in_roi": bool(mark.get("in_roi", True)),
                    "created_at": mark.get("created_at"),
                    "note": mark.get("note"),
                    "origin": "official_discovery_inherited",
                }, ensure_ascii=False) + "\n")

    write_json(out_dir / "split_provenance.json", {
        "seed": split["seed"],
        "split_sha256": split["split_sha256"],
        "manifest_source": str(args.manifest),
        "truth_source": str(args.truth),
        "exploratory_selection_source": str(args.exploratory_selection),
        "manifest_canonical_sha256": canonical_sha256(files),
        "prior_inference_file_ids": split["excluded_from_holdout_due_prior_inference"],
        "official_review_state": review_note,
        "sealed_accessed": False,
        "official_writes": 0,
    })

    counts = split["counts"]
    print(f"seed={split['seed']}")
    print(f"rapid_train={counts['rapid_train']} PS  rapid_eval={counts['rapid_eval']} PS")
    for camera in CAMERAS:
        per = split["per_camera"][camera]
        print(f"  {camera}: train={per['rapid_train']} eval={per['rapid_eval']} "
              f"forced_to_train={per['forced_to_train']}")
    print(f"split_sha256={split['split_sha256']}")
    print(f"frames: fixed_train={frame_manifest['counts']['fixed_train']} "
          f"fixed_eval={frame_manifest['counts']['fixed_eval']} "
          f"bonus_train={frame_manifest['counts']['bonus_train']}")
    print(f"manifest_sha256={frame_manifest['manifest_sha256']}")
    print(f"written to {out_dir}")
    print("\nRapid-Eval Holdout PS:")
    for row in split["rows"]:
        if row["split"] != TRAIN_SPLIT:
            print(f"  {row['camera_id']} {row['file_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
