#!/usr/bin/env python3
"""Step 1C-2 §42: real small-scale joint test for the positive-tile pipeline.

Picks diverse real training-eligible episodes from the Step 1C-1R manifest (single
target, small target, target near the frame edge, several verified Required in one tile,
multi-target frame) and runs the real path:

    raw PS -> decode -> 640x640 source crop -> PNG -> YOLO label -> review UI

and then proves, per case:

* the frame identity agrees with the Step 1C-0 decoded timestamp;
* the tile is a genuine source-native slice: every PNG pixel equals the decoded source
  crop, so nothing was resized, letterboxed or resampled;
* the labels line up: the PNG region at ``tile_xyxy`` is identical to the source frame
  region at ``source_xyxy``;
* the primary bbox is fully inside the crop, with the reported margin;
* the review server serves the untouched bytes and refuses to offer any way to draw,
  move or re-propose a box;
* no upstream artifact was modified.

Writes only under the work directory (default ``/tmp/step1c2_joint``), including a drawn
overlay contact sheet for human eyes.  Run it with the profile virtualenv:

    .venv-profile/bin/python scripts/verify_ground_litter_positive_tiles.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_positive_tiles import (  # noqa: E402
    build_candidate_for_episode,
    candidate_fingerprint,
    generate_candidates,
    load_positive_tile_input,
    png_bytes,
    sha256_file,
    write_jsonl,
    _offset_seconds,
    _screening_index,
)
from rtsp_annotator.ground_litter_positive_tile_decode import (  # noqa: E402
    load_default_decoder,
)
from rtsp_annotator.ground_litter_localization_review import read_jsonl  # noqa: E402

TRUTH = ROOT / "output/ground_litter_truth_reconciliation_20260923"
RECOVERY = ROOT / "output/ground_litter_gold_source_recovery_20260923"
LOCALIZATION = ROOT / "output/ground_litter_gold_localization_20260923"
GOLD = ROOT / "output/ground_litter_gold_episode_review_20260923"

FAILURES: list[str] = []


def check(condition, label, extra=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'' if not extra else '  ' + extra}",
          flush=True)
    if not condition:
        FAILURES.append(label)


def load_inputs():
    return load_positive_tile_input(
        training_manifest_path=TRUTH / "training_episode_manifest.jsonl",
        truth_overlay_path=TRUTH / "truth_reconciliation.jsonl",
        localization_path=LOCALIZATION / "localizations.jsonl",
        recovery_evidence_path=RECOVERY / "episode_source_evidence.jsonl",
        source_files_path=RECOVERY / "source_files.jsonl",
        gold_path=GOLD / "gold_episodes.jsonl",
        gold_manifest=GOLD / "MANIFEST.json",
    )


def pick_cases(data):
    """Metadata-only selection (no decode) of reviewable, diverse real cases."""
    episodes = data["episodes"]
    screening = _screening_index(data)
    frames: dict[tuple, dict] = {}
    for episode in episodes:
        key = (episode["source_file_id"], episode["step1c0_decoded_timestamp"])
        frames.setdefault(key, {"members": [], "key": key})["members"].append(episode)

    def plan_of(frame, episode):
        return build_candidate_for_episode(episode, frame["members"],
                                          screening.get(frame["key"], []))

    for frame in frames.values():
        plans = [plan_of(frame, member) for member in frame["members"]]
        frame["ok"] = all(p["candidate_generation_status"] == "READY_FOR_REVIEW"
                          for p in plans)
        frame["max_labels"] = max((p["label_count"] for p in plans), default=0)
        frame["plans"] = plans

    ordered = sorted(frames.values(), key=lambda f: (f["key"][0], f["key"][1]))
    ready = [f for f in ordered if f["ok"]]
    singles = [f for f in ready if len(f["members"]) == 1]
    multi = [f for f in ready if f["max_labels"] >= 2]
    complex_frames = [f for f in ready if len(f["members"]) >= 3]

    def short(episode):
        box = episode["verified_bbox"]
        return min(box[2] - box[0], box[3] - box[1])

    def edge_distance(episode):
        box = episode["verified_bbox"]
        return min(box[0], box[1], episode["source_width"] - box[2],
                   episode["source_height"] - box[3])

    cases = {
        "single": min(singles, key=lambda f: abs(short(f["members"][0]) - 30))["members"][0],
        "small": min(ready, key=lambda f: min(short(m) for m in f["members"]))["members"][0],
        "edge": min(ready, key=lambda f: min(edge_distance(m) for m in f["members"]))
        ["members"][0],
        "multi": max(multi, key=lambda f: (f["max_labels"], len(f["members"])))
        ["members"][0],
        "complex": max(complex_frames or multi, key=lambda f: len(f["members"]))
        ["members"][0],
    }
    blocked = [f for f in ordered
               if any(p["candidate_generation_status"]
                      == "KNOWN_UNLOCALIZED_REQUIRED_PRESENT" for p in f["plans"])]
    return cases, {f["key"]: f["members"] for f in ordered}, blocked


def png_pixels(path: Path):
    import numpy as np

    raw = Path(path).read_bytes()
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    scan = zlib.decompress(raw[raw.index(b"IDAT") + 4:raw.index(b"IEND") - 4])
    rows = np.frombuffer(scan, dtype="uint8").reshape(height, 1 + width * 3)
    return rows[:, 1:].reshape(height, width, 3), (width, height)


def draw_copy(tile_rgb, labels):
    """Draw an overlay on a COPY, for human inspection only."""
    canvas = tile_rgb.copy()

    def rect(box, colour):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(639, x2), min(639, y2)
        for thickness in range(3):
            if y1 + thickness <= 639:
                canvas[y1 + thickness, x1:x2 + 1] = colour
            if y2 - thickness >= 0:
                canvas[y2 - thickness, x1:x2 + 1] = colour
            if x1 + thickness <= 639:
                canvas[y1:y2 + 1, x1 + thickness] = colour
            if x2 - thickness >= 0:
                canvas[y1:y2 + 1, x2 - thickness] = colour

    for index, label in enumerate(labels):
        rect(label["tile_xyxy"], (0, 255, 0) if index == 0 else (60, 130, 255))
    return canvas


def contact_sheet(canvases, out_path: Path):
    import numpy as np

    cols, rows = 3, 2
    sheet = np.full((rows * 660, cols * 660, 3), 24, dtype="uint8")
    for index, canvas in enumerate(canvases[:cols * rows]):
        row, col = divmod(index, cols)
        y, x = row * 660 + 10, col * 660 + 10
        sheet[y:y + 640, x:x + 640] = canvas[:640, :640]
    out_path.write_bytes(png_bytes(np.ascontiguousarray(sheet)))


def hash_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/tmp/step1c2_joint"))
    parser.add_argument("--port", type=int, default=8797)
    parser.add_argument("--keep", action="store_true",
                        help="keep the previous work directory")
    args = parser.parse_args(argv)
    work = args.work
    if work.exists():
        if args.keep:
            shutil.rmtree(work)
        else:
            shutil.rmtree(work)
    work.mkdir(parents=True)

    data = load_inputs()
    upstream = {
        "gold": GOLD / "gold_episodes.jsonl",
        "recovery": RECOVERY / "episode_source_evidence.jsonl",
        "localization": LOCALIZATION / "localizations.jsonl",
        "truth_overlay": TRUTH / "truth_reconciliation.jsonl",
        "training_manifest": TRUTH / "training_episode_manifest.jsonl",
    }
    before = {name: hash_of(path) for name, path in upstream.items()}

    cases, frames, blocked = pick_cases(data)
    decoder = load_default_decoder()
    canvases = []
    union: set[str] = set()

    for name, primary in cases.items():
        key = (primary["source_file_id"], primary["step1c0_decoded_timestamp"])
        members = frames[key]
        subset = dict(data)
        subset["episodes"] = members
        print(f"\n--- case {name}: {primary['episode_id']} "
              f"({len(members)} episode(s) in the frame)")
        # Each case is generated in its own directory: §34 pruning is per artifact root,
        # so sharing one directory between cases would delete the other cases' tiles.
        result = generate_candidates(
            subset, decoder, work / "cases" / name / "images",
            input_fingerprint=candidate_fingerprint(subset))
        union.update((m["episode_id"] for m in members))
        candidates = [r for r in result["candidates"]
                      if primary["episode_id"] in r["primary_episode_ids"]]
        check(bool(candidates), f"{name}: a candidate tile was produced")
        if not candidates:
            continue
        row = candidates[0]
        if row["image_path"] is None:
            check(row["candidate_generation_status"] != "READY_FOR_REVIEW",
                  f"{name}: non-reviewable candidates carry no image")
            print(f"      skipped pixel checks: {row['candidate_generation_status']} "
                  f"({row['candidate_generation_detail']})")
            continue
        check(row["candidate_generation_status"] == "READY_FOR_REVIEW",
              f"{name}: generation status is reviewable", row["candidate_generation_status"])
        check(row["frame_identity_confirmed"] is True,
              f"{name}: decoded frame matches the Step 1C-0 timestamp",
              f"delta={row['timestamp_delta_ms']}ms interval={row['frame_interval_ms']}ms")
        image = Path(row["image_path"])
        check(image.is_file() and image.name == f"{row['tile_id']}.png",
              f"{name}: PNG written at the deterministic tile id")
        check(sha256_file(image) == row["image_sha256"],
              f"{name}: recorded image sha matches the bytes on disk")
        pixels, (width, height) = png_pixels(image)
        check((width, height) == (640, 640), f"{name}: PNG header is 640x640",
              f"{width}x{height}")
        crop = row["source_crop_xyxy"]
        check(row["crop_size"] == 640 and crop is not None,
              f"{name}: crop size recorded as 640")
        check(crop[2] - crop[0] == 640 and crop[3] - crop[1] == 640,
              f"{name}: crop is exactly 640x640 source pixels", json.dumps(crop))
        check(crop[0] >= 0 and crop[1] >= 0 and crop[2] <= row["source_width"]
              and crop[3] <= row["source_height"],
              f"{name}: crop inside the source frame")

        decoded = decoder.decode_frame(Path(row["source_ps_path"]),
                                       _offset_seconds(primary))
        frame_rgb = decoded["frame"][:, :, ::-1]
        expected = frame_rgb[crop[1]:crop[3], crop[0]:crop[2]]
        check(bool((pixels == expected).all()),
              f"{name}: every tile pixel equals the source crop (no resize, no resample)")

        for label in row["labels"]:
            sx = [int(round(v)) for v in label["source_xyxy"]]
            tx = [int(round(v)) for v in label["tile_xyxy"]]
            check(bool((pixels[tx[1]:tx[3], tx[0]:tx[2]]
                        == frame_rgb[sx[1]:sx[3], sx[0]:sx[2]]).all()),
                  f"{name}: label {label['episode_ids']} tile_xyxy maps onto source_xyxy",
                  f"tile={tx} source={sx}")
            check(sx[0] - crop[0] == tx[0] and sx[1] - crop[1] == tx[1],
                  f"{name}: label transform is exactly source - crop origin")
        primary_labels = [label for label in row["labels"]
                          if primary["episode_id"] in label["episode_ids"]]
        check(bool(primary_labels), f"{name}: primary episode has its own label")
        if primary_labels:
            check(all(0 <= v <= 1 for v in primary_labels[0]["yolo_xywh_norm"]),
                  f"{name}: YOLO values inside [0,1]",
                  json.dumps(primary_labels[0]["yolo_xywh_norm"]))
            margin = row["min_label_margin_px"]
            check(margin is not None and margin >= 0,
                  f"{name}: primary bbox fully inside the crop", f"margin={margin}px")
            at_edge = min(primary["verified_bbox"][0], primary["verified_bbox"][1],
                          primary["source_width"] - primary["verified_bbox"][2],
                          primary["source_height"] - primary["verified_bbox"][3]) < 32
            check(margin >= 32 or at_edge,
                  f"{name}: margin >= 32px unless the target sits at the frame edge",
                  f"margin={margin}px")
        check(row["label_count"] == len(row["labels"]) >= 1,
              f"{name}: label count matches", f"{row['label_count']}")
        print(f"      episode    : {primary['episode_id']} cam={primary['camera_id']}")
        print(f"      PS         : {Path(row['source_ps_path']).name}")
        print(f"      decoded    : {row['step1c0_decoded_timestamp']} -> "
              f"{row['step1c2_decoded_timestamp']}")
        print(f"      source bbox: {[round(v) for v in row['primary_source_bbox']]}")
        print(f"      crop xyxy  : {crop}")
        for label in row["labels"]:
            print(f"      label      : {label['episode_ids']} tile="
                  f"{[round(v) for v in label['tile_xyxy']]} yolo="
                  f"{label['yolo_xywh_norm']} short={label['source_short_side_px']}px")
        print(f"      sha256     : {row['image_sha256']}")
        canvases.append(draw_copy(pixels, row["labels"]))

    contact_sheet(canvases, work / "contact_sheet.png")

    # One artifact over the union of the case frames, for the review-UI joint test.
    subset = dict(data)
    subset["episodes"] = [e for e in data["episodes"] if e["episode_id"] in union]
    unified = generate_candidates(
        subset, decoder, work / "candidate_tiles" / "images",
        input_fingerprint=candidate_fingerprint(subset))
    rows_all = unified["candidates"]
    write_jsonl(work / "tile_candidates.jsonl", rows_all)

    print(f"\n--- §17 rule on real data: {len(blocked)} frame(s) blocked by a same-frame "
          f"Required without a verified bbox (reported, not a failure)")
    for frame in blocked[:5]:
        member = frame["members"][0]
        plan = frame["plans"][0]
        print(f"      {member['episode_id']} cam={member['camera_id']} "
              f"{member['step1c0_decoded_timestamp']} "
              f"status={plan['candidate_generation_status']} "
              f"in_crop={plan['known_unlocalized_required_in_crop_ids']} "
              f"same_frame={len(plan['known_unlocalized_required_same_frame_ids'])}")

    check(len(rows_all) >= 5, "at least five candidate tiles for the UI joint test",
          f"n={len(rows_all)} (a multi-target frame can yield several crops)")

    port = args.port
    state = work / "review_state.json"
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts/build_ground_litter_positive_tiles.py"),
         "--output", str(work), "--state", str(state), "--port", str(port), "serve"],
        cwd=str(ROOT), stdout=open(work / "serve.log", "wb"),
        stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        meta = None
        for _ in range(80):
            time.sleep(0.25)
            try:
                meta = json.loads(urllib.request.urlopen(base + "/api/meta",
                                                         timeout=10).read())
                break
            except Exception:
                continue
        check(meta is not None, "review server answered /api/meta")
        check(meta["reviewable_count"] == len(rows_all)
              and meta["progress"]["reviewed"] == 0,
              f"server shows {len(rows_all)} reviewable tiles, 0 reviewed",
              json.dumps(meta["progress"]))
        first = meta["queue"][0]["tile_id"]
        tile = json.loads(urllib.request.urlopen(base + "/api/tile?id=" + first,
                                                 timeout=10).read())
        check(len(tile["labels"]) >= 1 and tile["crop_size"] == 640,
              "server exposes labels and the 640 crop")
        check(tile["options"] == ["ANNOTATION_COMPLETE", "MISSING_REQUIRED",
                                  "BOX_PROBLEM", "UNCERTAIN_COMPLETENESS"],
              "server offers exactly the four review decisions")
        body = urllib.request.urlopen(base + "/api/image?tile_id=" + first,
                                      timeout=30).read()
        recorded = [row for row in rows_all if row["tile_id"] == first][0]
        check(hashlib.sha256(body).hexdigest() == recorded["image_sha256"],
              "served image bytes are the untouched candidate PNG")
        request = urllib.request.Request(
            base + "/api/review",
            data=json.dumps({"tile_id": first, "decision": "ANNOTATION_COMPLETE",
                             "note": "joint test"}).encode(),
            method="POST", headers={"Content-Type": "application/json"})
        reply = json.loads(urllib.request.urlopen(request, timeout=10).read())
        check(reply["annotation_review_status"] == "ANNOTATION_COMPLETE"
              and reply["positive_training_ready"] is True,
              "COMPLETE marks the tile training-ready")
        for forbidden in ("/api/proposal", "/api/box", "/api/new_bbox"):
            try:
                urllib.request.urlopen(base + forbidden, timeout=5)
                check(False, f"{forbidden} must not exist")
            except urllib.error.HTTPError as exc:
                check(exc.code == 404, f"{forbidden} is not served ({exc.code})")
        page = urllib.request.urlopen(base + "/app.js", timeout=10).read()
        check(b"ANNOTATION_COMPLETE" in page and b"canvas" not in page.lower(),
              "page draws CSS overlays only (no canvas)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    after = {name: hash_of(path) for name, path in upstream.items()}
    check(before == after, "all five upstream artifacts are byte-identical",
          json.dumps({k: before[k] == after[k] for k in before}))

    print(f"\ncontact sheet: {work / 'contact_sheet.png'}")
    print(f"{len(FAILURES)} failures" if FAILURES else "ALL JOINT CHECKS PASSED")
    for failure in FAILURES:
        print(" -", failure)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
