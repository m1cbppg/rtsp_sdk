#!/usr/bin/env python3
"""Step 1C-2M §37/§38: real joint test for the supplemental completion pipeline.

Part A drives the real review UI over HTTP against the real 62-tile queue with a /tmp
state (the production state stays untouched): point click -> A/B/C -> select -> multiple
supplemental targets -> final recheck -> accepted_v2.

Part B writes contact sheets so a human can look at real MISSING_REQUIRED tiles, choose
points on visible unlabeled REQUIRED_LITTER, and then see the A/B/C proposals drawn on a
copy.  Nothing from this script is written into the production completion state.

    .venv-profile/bin/python scripts/verify_ground_litter_tile_completion.py --part a
    .venv-profile/bin/python scripts/verify_ground_litter_tile_completion.py --part b
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

from rtsp_annotator.ground_litter_point_proposal import (  # noqa: E402
    PointProposalEngine,
    size_prior_from_labels,
)
from rtsp_annotator.ground_litter_positive_tiles import (  # noqa: E402
    png_bytes,
    write_jsonl,
)
from rtsp_annotator.ground_litter_tile_completion import (  # noqa: E402
    TILE_STATUS_COMPLETE,
    TILE_STATUS_STILL_MISSING,
    TILE_STATUS_UNCERTAIN,
    CompletionState,
    build_accepted_v2,
    load_completion_input,
    queue_rows,
    sha256_file,
    verify_preflight,
)

STEP1C2 = ROOT / "output" / "ground_litter_positive_tiles_20260923"
FAILURES: list[str] = []

#: Tiles chosen for the §37 joint test (existing 1/2/3 labels + the §17 risk tile) and
#: for the §38 contact sheet.  Filled by --list.
CASES = ("pt-missing",)


def check(condition, label, extra=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'' if not extra else '  ' + extra}",
          flush=True)
    if not condition:
        FAILURES.append(label)


def png_pixels(path: Path):
    import numpy as np

    raw = Path(path).read_bytes()
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    scan = zlib.decompress(raw[raw.index(b"IDAT") + 4:raw.index(b"IEND") - 4])
    rows = np.frombuffer(scan, dtype="uint8").reshape(height, 1 + width * 3)
    return rows[:, 1:].reshape(height, width, 3), (width, height)


def draw(tile_rgb, boxes):
    """Draw coloured outlines on a COPY (never written into a tile PNG)."""
    canvas = tile_rgb.copy()

    def rect(box, colour, thickness=2):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(639, x2), min(639, y2)
        for offset in range(thickness):
            if y1 + offset <= 639:
                canvas[y1 + offset, x1:x2 + 1] = colour
            if y2 - offset >= 0:
                canvas[y2 - offset, x1:x2 + 1] = colour
            if x1 + offset <= 639:
                canvas[y1:y2 + 1, x1 + offset] = colour
            if x2 - offset >= 0:
                canvas[y1:y2 + 1, x2 - offset] = colour

    for coords, colour in boxes:
        rect(coords, colour)
    return canvas


def contact_sheet(canvases, out_path: Path, cols=3):
    import numpy as np

    rows = max(1, (len(canvases) + cols - 1) // cols)
    sheet = np.full((rows * 660, cols * 660, 3), 24, dtype="uint8")
    for index, canvas in enumerate(canvases):
        row, col = divmod(index, cols)
        y, x = row * 660 + 10, col * 660 + 10
        sheet[y:y + 640, x:x + 640] = canvas[:640, :640]
    out_path.write_bytes(png_bytes(np.ascontiguousarray(sheet)))


def point(tile_id: str, x: float, y: float) -> dict:
    return {"tile_id": tile_id, "tile_x": x, "tile_y": y}


class Ui:
    def __init__(self, output: Path, state: Path, port: int) -> None:
        self.base = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            [sys.executable,
             str(ROOT / "scripts/complete_ground_litter_positive_tiles.py"),
             "--step1c2-root", str(STEP1C2), "--output", str(output),
             "--state", str(state), "--port", str(port), "serve"],
            cwd=str(ROOT), stdout=open(output / "serve.log", "wb"),
            stderr=subprocess.STDOUT)

    def __enter__(self) -> "Ui":
        for _ in range(120):
            time.sleep(0.25)
            try:
                self.get("/api/meta")
                return self
            except Exception:
                continue
        raise CompletionError("review server did not start")

    def __exit__(self, *exc) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:                # pragma: no cover
            self.proc.kill()

    def get(self, path: str) -> dict:
        return json.loads(urllib.request.urlopen(self.base + path, timeout=30).read())

    def post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(request, timeout=60).read())
        except urllib.error.HTTPError as exc:
            return json.loads(exc.read())


class CompletionError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# part A: the real UI joint test
# --------------------------------------------------------------------------- #


def part_a(work: Path, cases: list[dict], port: int) -> int:
    data = load_completion_input(STEP1C2)
    preflight = verify_preflight(data)
    check(preflight["counts_match"], "preflight matches 29/62/2/0/93",
          json.dumps({key: preflight[key] for key in (
              "step1c2_annotation_complete", "step1c2_missing_required",
              "step1c2_box_problem", "step1c2_total")}))
    check(preflight["problem_count"] == 0, "no input problem",
          json.dumps(preflight["problems"]))

    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    state_path = work / "completion_state.json"
    before = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
        ("gold", ROOT / "output/ground_litter_gold_episode_review_20260923/gold_episodes.jsonl"),
        ("recovery", ROOT / "output/ground_litter_gold_source_recovery_20260923/episode_source_evidence.jsonl"),
        ("localization", ROOT / "output/ground_litter_gold_localization_20260923/localizations.jsonl"),
        ("truth", ROOT / "output/ground_litter_truth_reconciliation_20260923/truth_reconciliation.jsonl"),
        ("step1c2_tiles", STEP1C2 / "tile_candidates.jsonl"),
        ("step1c2_summary", STEP1C2 / "SUMMARY.json"),
        ("step1c2_review_state", STEP1C2 / "review_state.json"),
    )}

    with Ui(work, state_path, port) as ui:
        meta = ui.get("/api/meta")
        check(meta["queue_count"] == 62, "the UI queue holds exactly the 62 tiles",
              str(meta["queue_count"]))
        check(meta["progress"]["pending"] == 62, "0/62 reviewed at start",
              json.dumps(meta["progress"]))
        check(not any(row["primary_episode_id"] == "pt-complete-0"
                      for row in meta["queue"]), "accepted tiles stay out of the queue")
        check(all(row["existing_label_count"] >= 1 for row in meta["queue"]),
              "every queue tile has its original verified label(s)")
        print(f"      size prior used for proposal C: {json.dumps(meta['size_prior'])}")

        for case in cases:
            name, tile_id = case["name"], case["tile_id"]
            print(f"\n--- case {name}: {tile_id} ({case['why']})")
            tile = ui.get("/api/tile?id=" + tile_id)
            before_labels = json.dumps(tile["existing_labels"], sort_keys=True)
            before_sha = tile["image_sha256"]
            crop = tile["source_crop_xyxy"]
            made = truncated = 0
            for index, (tx, ty) in enumerate(case["points"]):
                reply = ui.post("/api/point", point(tile_id, tx, ty))
                check(reply.get("ok") is True, f"{name}: point {index + 1} generated a result",
                      json.dumps({key: reply.get(key) for key in ("error_code", "message")}))
                if not reply.get("ok"):
                    continue
                check(reply["elapsed_ms"] < 2000,
                      f"{name}: proposal under 2 s", f"{reply['elapsed_ms']} ms")
                candidates = reply["candidates"]
                check(0 < len(candidates) <= 3, f"{name}: at most A/B/C",
                      json.dumps([c["letter"] for c in candidates]))
                chosen = candidates[0]
                sel = ui.post("/api/select", {
                    "tile_id": tile_id, "target_id": reply["supplemental_target_id"],
                    "letter": chosen["letter"]})
                if sel.get("warning"):
                    print(f"      point {index + 1}: {sel['warning']} "
                          f"({sel.get('reason')}) — confirmed as an independent target")
                    sel = ui.post("/api/select", {
                        "tile_id": tile_id, "target_id": reply["supplemental_target_id"],
                        "letter": chosen["letter"], "confirm_independent": True})
                check(sel.get("ok") is True, f"{name}: proposal {chosen['letter']} saved",
                      json.dumps({key: sel.get(key) for key in ("error_code", "message")}))
                tile = ui.get("/api/tile?id=" + tile_id)
                target = [t for t in tile["supplemental_targets"]
                          if t["supplemental_target_id"] == reply["supplemental_target_id"]][0]
                if target["localization_status"] == "VERIFIED_BBOX":
                    made += 1
                    sb = target["verified_source_xyxy"]
                    tb = target["verified_tile_xyxy"]
                    check(abs(sb[0] - (tb[0] + crop[0])) < 0.01
                          and abs(sb[1] - (tb[1] + crop[1])) < 0.01,
                          f"{name}: source = tile + crop origin",
                          f"tile={tb} source={sb} crop={crop}")
                    supplement = [label for label in tile["merged_labels"]
                                  if label["origin"] == "step1c2m_supplemental"]
                    check(all(not label["episode_ids"] for label in supplement),
                          f"{name}: a supplemental label carries no episode identity")
                    check(any(reply["supplemental_target_id"]
                              in label["supplemental_target_ids"] for label in supplement),
                          f"{name}: supplemental provenance is recorded")
                    print(f"      point {index + 1} ({tx},{ty}) -> {chosen['letter']} "
                          f"[{chosen['method']}] tile={tb} source={sb}")
                else:
                    if target["localization_status"] == "TARGET_TRUNCATED_BY_TILE":
                        truncated += 1
                    print(f"      point {index + 1}: {target['localization_status']} "
                          "(kept via the reviewer path)")
            check(made >= case["min_saved"],
                  f"{name}: at least {case['min_saved']} supplemental target(s) located",
                  f"located={made} truncated={truncated}")
            if case.get("expect_truncated"):
                check(truncated >= 1,
                      f"{name}: the tile-boundary guard fired as expected",
                      f"truncated={truncated}")
            check(json.dumps(tile["existing_labels"], sort_keys=True) == before_labels,
                  f"{name}: original verified labels unchanged")
            check(tile["image_sha256"] == before_sha, f"{name}: tile image untouched")

        # the failure path: reject the proposals -> unresolved -> cannot complete
        tile_id = cases[0]["tile_id"]
        tile = ui.get("/api/tile?id=" + tile_id)
        reply = ui.post("/api/point", point(tile_id, 320, 60))
        if reply.get("ok"):
            ui.post("/api/reject", {"tile_id": tile_id,
                                    "target_id": reply["supplemental_target_id"],
                                    "reason": "joint test: all three wrong"})
            weird = ui.post("/api/recheck", {"tile_id": tile_id,
                                            "decision": TILE_STATUS_COMPLETE, "note": ""})
            check(weird.get("ok") is False,
                  "a tile with an unresolved supplemental target cannot COMPLETE",
                  json.dumps(weird)[:120])

        # complete one case, then verify accepted_v2 for a subset build
        target_case = cases[-1]
        tile_id = target_case["tile_id"]
        done = ui.post("/api/recheck", {"tile_id": tile_id,
                                       "decision": TILE_STATUS_COMPLETE,
                                       "note": "joint test complete"})
        check(done.get("ok") is True and done.get("positive_training_ready") is True,
              "recheck COMPLETE marks the tile training-ready", json.dumps(done)[:120])
        still = ui.post("/api/recheck", {"tile_id": cases[0]["tile_id"],
                                        "decision": TILE_STATUS_STILL_MISSING,
                                        "note": "joint test still missing"})
        check(still.get("ok") is True and still.get("positive_training_ready") is False,
              "STILL_MISSING excludes the tile", json.dumps(still)[:120])

    # resume: the /tmp state survives a restart
    reloaded = CompletionState.load(state_path)
    check(len(reloaded.tiles) >= 2, "completion state persisted", str(len(reloaded.tiles)))
    check(bool(reloaded.audit_trail), "audit trail persisted",
          str(len(reloaded.audit_trail)))

    # accepted_v2 over a two-tile subset (the 29 frozen tiles are always included)
    subset = [row for row in queue_rows(data)
              if row["tile_id"] in {cases[-1]["tile_id"], cases[0]["tile_id"]}]
    accepted = build_accepted_v2(data, subset, work / "accepted_v2", state=reloaded)
    check(accepted["frozen_tile_count"] == 29, "all 29 frozen positives are carried over")
    check(accepted["salvaged_tile_count"] == 1, "the completed tile is salvaged",
          str(accepted["salvaged_tile_count"]))
    check(accepted["accepted_v2_tile_count"] == 30, "accepted_v2 = 29 + 1",
          str(accepted["accepted_v2_tile_count"]))
    for row in accepted["rows"]:
        image = Path(row["image_path"])
        source = (STEP1C2 / "tile_candidates.jsonl")
        original = (data["by_id"][row["tile_id"]]["image_path"])
        check(sha256_file(image) == row["image_sha256"], f"{row['tile_id']}: accepted sha")
        check(sha256_file(image) == sha256_file(Path(original)),
              f"{row['tile_id']}: accepted_v2 bytes == Step 1C-2 candidate bytes")
        lines = Path(row["label_path"]).read_text(encoding="utf-8").strip().splitlines()
        check(len(lines) == row["label_count"] >= 1, f"{row['tile_id']}: label count",
              str(len(lines)))
        for line in lines:
            parts = line.split()
            check(parts[0] == "0" and len(parts) == 5
                  and all(0.0 <= float(v) <= 1.0 for v in parts[1:]),
                  f"{row['tile_id']}: one-class YOLO line", line)
    salvaged = [row for row in accepted["rows"] if row["origin"] == "step1c2m_salvaged"]
    if salvaged:
        check(salvaged[0]["label_count"] >= 2,
              "the salvaged tile carries original + supplemental labels",
              str(salvaged[0]["label_count"]))
        check(any(label["origin"] == "step1c2m_supplemental"
                  for label in salvaged[0]["labels"]),
              "supplemental provenance is preserved")
    write_jsonl(work / "joint_accepted_v2.jsonl", accepted["rows"])

    after = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
        ("gold", ROOT / "output/ground_litter_gold_episode_review_20260923/gold_episodes.jsonl"),
        ("recovery", ROOT / "output/ground_litter_gold_source_recovery_20260923/episode_source_evidence.jsonl"),
        ("localization", ROOT / "output/ground_litter_gold_localization_20260923/localizations.jsonl"),
        ("truth", ROOT / "output/ground_litter_truth_reconciliation_20260923/truth_reconciliation.jsonl"),
        ("step1c2_tiles", STEP1C2 / "tile_candidates.jsonl"),
        ("step1c2_summary", STEP1C2 / "SUMMARY.json"),
        ("step1c2_review_state", STEP1C2 / "review_state.json"),
    )}
    check(before == after, "every upstream artifact is byte-identical",
          json.dumps({key: before[key] == after[key] for key in before}))
    real_state = STEP1C2.parent / "ground_litter_positive_tile_completion_20260923"
    if (real_state / "completion_state.json").is_file():
        real = json.loads((real_state / "completion_state.json").read_text())
        check(not real.get("tiles"),
              "the production completion state was not touched by this test")

    print(f"\n{len(FAILURES)} failures" if FAILURES else "\nPART A: ALL CHECKS PASSED")
    for failure in FAILURES:
        print(" -", failure)
    return 1 if FAILURES else 0


# --------------------------------------------------------------------------- #
# part B: contact sheets for the human-driven proposal sanity sample
# --------------------------------------------------------------------------- #


def part_b(work: Path, limit: int, points: list[dict] | None) -> int:
    from rtsp_annotator.ground_litter_positive_tiles import read_jsonl

    work.mkdir(parents=True, exist_ok=True)
    rows = {row["tile_id"]: row for row in read_jsonl(STEP1C2 / "tile_candidates.jsonl")}
    queue = sorted(queue_rows(load_completion_input(STEP1C2)),
                   key=lambda row: (-row["existing_label_count"], row["tile_id"]))
    # a mix of the dense multi-label tiles (§31) and simpler single-label tiles
    dense = [row for row in queue if row["existing_label_count"] >= 2]
    single = [row for row in queue if row["existing_label_count"] == 1]
    chosen = dense[: limit - limit // 3] + single[: limit // 3]
    chosen = chosen[:limit]
    write_jsonl(work / "sample_tiles.jsonl", [
        {key: row[key] for key in ("tile_id", "primary_episode_id", "camera_id",
                                   "existing_label_count", "source_crop_xyxy",
                                   "image_path", "known_unlocalized_required_present")}
        for row in chosen])

    canvases = []
    for row in chosen:
        pixels, _ = png_pixels(Path(row["image_path"]))
        boxes = [(label["tile_xyxy"], (0, 255, 0)) for label in row["existing_labels"]]
        canvases.append(draw(pixels, boxes))
    contact_sheet(canvases, work / "sample_sheet.png")
    print(f"sample sheet ({len(chosen)} tiles, green = existing verified labels): "
          f"{work / 'sample_sheet.png'}")

    if not points:
        return 0

    engine = PointProposalEngine()
    prior = size_prior_from_labels([label for row in chosen
                                    for label in row["existing_labels"]])
    outcomes = []
    canvases = []
    for entry in points:
        row = rows[entry["tile_id"]]
        pixels, _ = png_pixels(Path(row["image_path"]))
        result = engine.propose(tile_path=Path(row["image_path"]),
                                point_tile_xy=[entry["tile_x"], entry["tile_y"]],
                                size=640, size_prior=prior, revision=1)
        boxes = [(label["tile_xyxy"], (0, 255, 0)) for label in row["labels"]]
        for candidate in result.get("candidates") or []:
            # PointProposalEngine returns "bbox"; keep both spellings working here
            boxes.append((candidate.get("bbox_tile_xyxy") or candidate["bbox"],
                          (60, 130, 255)))
        boxes.append(([entry["tile_x"] - 12, entry["tile_y"] - 1,
                       entry["tile_x"] + 12, entry["tile_y"] + 1], (255, 209, 102)))
        boxes.append(([entry["tile_x"] - 1, entry["tile_y"] - 12,
                       entry["tile_x"] + 1, entry["tile_y"] + 12], (255, 209, 102)))
        canvases.append(draw(pixels, boxes))
        outcomes.append({
            "tile_id": entry["tile_id"], "point": [entry["tile_x"], entry["tile_y"]],
            "ok": result.get("ok"), "error_code": result.get("error_code"),
            "candidates": [{"letter": c["letter"],
                            "bbox": c.get("bbox_tile_xyxy") or c["bbox"],
                            "method": c["method"], "contains_point": c["contains_point"],
                            "touches_border": c["touches_border"]}
                           for c in result.get("candidates") or []],
            "note": entry.get("note", ""),
        })
        print(f"  {entry['tile_id']} point=({entry['tile_x']},{entry['tile_y']}) "
              f"ok={result.get('ok')} n={len(result.get('candidates') or [])}")
    contact_sheet(canvases, work / "proposal_sheet.png")
    write_jsonl(work / "proposal_sample.jsonl", outcomes)
    print(f"proposal sheet: {work / 'proposal_sheet.png'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("a", "b"), default="a")
    parser.add_argument("--work", type=Path, default=Path("/tmp/step1c2m_joint"))
    parser.add_argument("--port", type=int, default=8796)
    parser.add_argument("--cases", type=Path, default=None,
                        help="JSON file with the §37 cases")
    parser.add_argument("--sample-limit", type=int, default=12)
    parser.add_argument("--points", type=Path, default=None,
                        help="JSON file with the §38 sample points")
    args = parser.parse_args(argv)
    if args.part == "a":
        cases = json.loads(Path(args.cases).read_text()) if args.cases else None
        if not cases:
            raise SystemExit("--cases is required for part a")
        return part_a(args.work, cases, args.port)
    points = json.loads(Path(args.points).read_text()) if args.points else None
    return part_b(args.work, args.sample_limit, points)


if __name__ == "__main__":
    sys.exit(main())
