#!/usr/bin/env python3
"""Step 1D §37: real joint test for the hard-negative pipeline.

Picks five real ready candidates covering distinct hardness types (detector
false positive, surface texture, temporal candidate, random-grid background, and a tile
that carries the UNCERTAIN warning) plus one candidate the generator excluded because a
verified REQUIRED box overlaps its crop, then drives the real review UI over HTTP with a
/tmp state:

    historical NON_LITTER card
      -> local raw PS (already on disk)
      -> decoded source frame
      -> 640x640 source-native crop (anchor kept inside)
      -> PNG + SHA-256
      -> UI review (NEGATIVE_OK / REQUIRED_PRESENT / UNCERTAIN / BAD_CROP)
      -> accepted tile with a 0-byte YOLO label

Nothing from this script is written into the production review state.

    .venv-profile/bin/python scripts/verify_ground_litter_hard_negatives.py
    .venv-profile/bin/python scripts/verify_ground_litter_hard_negatives.py --sheet
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

from rtsp_annotator.ground_litter_hard_negatives import (  # noqa: E402
    RISK_UNCERTAIN,
    STATUS_KNOWN_REQUIRED,
    STATUS_READY,
    check_positive_conflict,
    load_historical_input,
    load_truth_context,
    load_verified_required_boxes,
    plan_candidates,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from rtsp_annotator.ground_litter_positive_tiles import png_bytes  # noqa: E402
from scripts.build_ground_litter_hard_negatives import (  # noqa: E402
    BATCH_DIRS,
    GOLD_ROOT,
    LOCALIZATION_ROOT,
    RECOVERY_ROOT,
    STEP1C2_ROOT,
    STEP1C2M_ROOT,
    TRUTH_ROOT,
)

OUTPUT = ROOT / "output/ground_litter_hard_negatives_20260923"
FAILURES: list[str] = []
UPSTREAM = (
    ("gold", GOLD_ROOT / "gold_episodes.jsonl"),
    ("recovery", RECOVERY_ROOT / "episode_source_evidence.jsonl"),
    ("localization", LOCALIZATION_ROOT / "localizations.jsonl"),
    ("truth", TRUTH_ROOT / "truth_reconciliation.jsonl"),
    ("positive_manifest", STEP1C2M_ROOT / "positive_training_manifest_v2.jsonl"),
    ("step1c2_tiles", STEP1C2_ROOT / "tile_candidates.jsonl"),
)


def check(condition, label, extra=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'' if not extra else '  ' + extra}",
          flush=True)
    if not condition:
        FAILURES.append(label)


def hash_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def png_pixels(path: Path):
    import numpy as np

    raw = Path(path).read_bytes()
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    scan = zlib.decompress(raw[raw.index(b"IDAT") + 4:raw.index(b"IEND") - 4])
    rows = np.frombuffer(scan, dtype="uint8").reshape(height, 1 + width * 3)
    return rows[:, 1:].reshape(height, width, 3)


def draw(tile_rgb, boxes):
    canvas = tile_rgb.copy()

    def rect(box, colour):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(639, x2), min(639, y2)
        for offset in range(3):
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


def sheet(canvases, out_path: Path, cols=3):
    import numpy as np

    rows = max(1, (len(canvases) + cols - 1) // cols)
    canvas = np.full((rows * 660, cols * 660, 3), 24, dtype="uint8")
    for index, tile in enumerate(canvases):
        row, col = divmod(index, cols)
        y, x = row * 660 + 10, col * 660 + 10
        canvas[y:y + 640, x:x + 640] = tile[:640, :640]
    out_path.write_bytes(png_bytes(np.ascontiguousarray(canvas)))


class CompletionError(RuntimeError):
    pass


def pick_cases(plan: dict) -> tuple[list[dict], dict | None]:
    ready = [row for row in plan["candidates"]
             if row["candidate_generation_status"] == STATUS_READY]
    excluded = [row for row in plan["candidates"]
                if row["candidate_generation_status"] == STATUS_KNOWN_REQUIRED]
    uncertain = [row for row in ready if RISK_UNCERTAIN in (row["risk_flags"] or [])]
    cases: list[dict] = []
    seen: set[str] = set()

    def take(pool, name, why):
        for row in pool:
            if row["negative_tile_id"] not in seen:
                seen.add(row["negative_tile_id"])
                cases.append({"name": name, "why": why, "row": row})
                return True
        return False

    take([row for row in ready if row["hardness_source"] == "historical_false_positive"],
         "detector-false-positive", "historical detector false positive (semantic)")
    take([row for row in ready if row["hardness_source"] == "surface_texture"],
         "surface-texture", "路面纹理 / texture-mined non-litter")
    take([row for row in ready
          if row["hardness_source"] in ("temporal_candidate", "other")],
         "temporal-candidate", "temporal / other-sourced non-litter")
    take([row for row in ready if row["hardness_source"] == "random_grid_background"],
         "random-grid", "random-grid background (easy negative)")
    take(uncertain, "uncertain-warning", "tile with an UNCERTAIN truth warning")
    for row in ready:
        if len(cases) >= 5:
            break
        take([row], f"extra-{len(cases)}", "additional ready candidate")
    return cases, (excluded[0] if excluded else None)


class Ui:
    def __init__(self, state: Path, port: int) -> None:
        self.base = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            [sys.executable,
             str(ROOT / "scripts/build_ground_litter_hard_negatives.py"),
             "--output", str(OUTPUT), "--state", str(state), "--port", str(port),
             "serve"],
            cwd=str(ROOT), stdout=open(OUTPUT.parent / "serve_1d.log", "wb"),
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
            return json.loads(urllib.request.urlopen(request, timeout=30).read())
        except urllib.error.HTTPError as exc:
            return json.loads(exc.read())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/tmp/step1d_joint"))
    parser.add_argument("--port", type=int, default=8798)
    parser.add_argument("--sheet", action="store_true",
                        help="write contact sheets for human review and stop")
    args = parser.parse_args(argv)
    work = args.work
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    state_path = work / "review_state.json"

    data = load_historical_input(
        BATCH_DIRS, source_files_path=RECOVERY_ROOT / "source_files.jsonl",
        reconciled_overlay_path=TRUTH_ROOT / "truth_reconciliation.jsonl",
        localization_path=LOCALIZATION_ROOT / "localizations.jsonl",
        recovery_evidence_path=RECOVERY_ROOT / "episode_source_evidence.jsonl")
    by_frame, by_file = load_verified_required_boxes(
        tile_candidates_path=STEP1C2_ROOT / "tile_candidates.jsonl",
        completion_manifest_path=STEP1C2M_ROOT / "tile_completion_manifest.jsonl")
    context = load_truth_context(
        overlay_path=TRUTH_ROOT / "truth_reconciliation.jsonl",
        localization_path=LOCALIZATION_ROOT / "localizations.jsonl")
    plan = plan_candidates(data, required_by_frame=by_frame, required_by_file=by_file,
                           truth_context=context)
    cases, excluded = pick_cases(plan)
    # the UI serves the generated index (with image paths and hashes), not the plan
    generated = {row["negative_tile_id"]: row
                 for row in read_jsonl(OUTPUT / "negative_candidates.jsonl")}
    for case in cases:
        case["row"] = {**case["row"], **generated.get(case["row"]["negative_tile_id"], {})}
    if excluded:
        excluded = {**excluded, **generated.get(excluded["negative_tile_id"], {})}
    print("=== plan (replayed read-only) ===")
    print(f"  historical NON_LITTER {data['historical_non_litter_count']} + reconciled "
          f"{data['reconciled_non_litter_count']} | recoverable {data['recoverable_count']}")
    print(f"  candidates {len(plan['candidates'])} | ready "
          f"{plan['status_counts'].get(STATUS_READY)} | §31 excluded "
          f"{plan['status_counts'].get(STATUS_KNOWN_REQUIRED)}")
    print("  cases:", ", ".join(f"{c['name']}={c['row']['negative_tile_id']}"
                               for c in cases))

    if args.sheet:
        generated = {row["negative_tile_id"]: row
                     for row in read_jsonl(OUTPUT / "negative_candidates.jsonl")}
        cases = [{"name": c["name"], "why": c["why"],
                  "row": {**c["row"], **generated.get(c["row"]["negative_tile_id"], {})}}
                 for c in cases]
        if excluded:
            excluded = {**excluded, **generated.get(excluded["negative_tile_id"], {})}
        canvases = []
        for case in cases:
            row = case["row"]
            canvases.append(draw(png_pixels(Path(row["image_path"])),
                                 [(row["anchor_tile_xyxy"], (255, 159, 28))]))
        if excluded and excluded.get("image_path"):
            row = excluded
            boxes = [(row["anchor_tile_xyxy"], (255, 159, 28))]
            for box in by_frame.get((row["source_file_id"], row["timestamp"]), []):
                crop = row["source_crop_xyxy"]
                boxes.append(([box["box"][0] - crop[0], box["box"][1] - crop[1],
                               box["box"][2] - crop[0], box["box"][3] - crop[1]],
                              (224, 96, 58)))
            canvases.append(draw(png_pixels(Path(excluded["image_path"])), boxes))
        sheet(canvases, work / "case_sheet.png")
        print(f"case sheet: {work / 'case_sheet.png'}")
        if excluded and not excluded.get("image_path"):
            print(f"  note: the §31-excluded candidate {excluded['negative_tile_id']} has "
                  f"no tile image by design (excluded from metadata alone)")
        return 0

    print("\n=== exact duplicate / conflict screening ===")
    positive = read_jsonl(STEP1C2M_ROOT / "positive_training_manifest_v2.jsonl")
    conflict = check_positive_conflict(
        plan["candidates"], positive, STEP1C2_ROOT / "tile_candidates.jsonl")
    check(conflict["ok"], "no candidate collides with the positive pool",
          json.dumps({k: conflict[k] for k in ("sha_conflict_count",
                                               "crop_conflict_count")}))

    before = {name: hash_of(path) for name, path in UPSTREAM}
    case_state = work / "review_state.json"
    decisions = ["NEGATIVE_OK", "NEGATIVE_OK", "REQUIRED_PRESENT", "NEGATIVE_OK",
                 "UNCERTAIN"]
    with Ui(case_state, args.port) as ui:
        meta = ui.get("/api/meta")
        check(meta["reviewable_count"] == plan["status_counts"].get(STATUS_READY),
              "the UI queue holds exactly the ready candidates",
              f"{meta['reviewable_count']} vs {plan['status_counts'].get(STATUS_READY)}")
        check(meta["excluded_by_generation_count"] ==
              plan["status_counts"].get(STATUS_KNOWN_REQUIRED),
              "the §31 exclusions are kept out of the queue",
              str(meta["excluded_by_generation_count"]))
        check(meta["progress"]["pending"] == meta["reviewable_count"],
              "0 reviewed at the start", json.dumps(meta["progress"]))

        for case, decision in zip(cases, decisions):
            row = case["row"]
            tile = ui.get("/api/candidate?id=" + row["negative_tile_id"])
            check(tile["negative_tile_id"] == row["negative_tile_id"],
                  f"{case['name']}: served by the UI")
            check(tile["anchor_tile_xyxy"] is not None and
                  tile["source_crop_xyxy"] == row["source_crop_xyxy"],
                  f"{case['name']}: crop + anchor exposed",
                  json.dumps(tile["anchor_tile_xyxy"]))
            check(tile["review_status"] == "PENDING",
                  f"{case['name']}: starts pending")
            body = urllib.request.urlopen(ui.base + tile["image_url"],
                                          timeout=30).read()
            check(hashlib.sha256(body).hexdigest() == row["image_sha256"],
                  f"{case['name']}: served image bytes are the untouched PNG")
            weird = ui.post("/api/review", {"candidate_id": row["negative_tile_id"],
                                            "decision": decision, "reason": ""})
            if decision == "NEGATIVE_OK":
                check(weird.get("ok") is True and weird.get("hard_negative_ready") is True,
                      f"{case['name']}: NEGATIVE_OK -> ready", json.dumps(weird)[:100])
            else:
                check("error" in weird,
                      f"{case['name']}: {decision} without a reason is refused",
                      json.dumps(weird)[:100])
                reply = ui.post("/api/review",
                                {"candidate_id": row["negative_tile_id"],
                                 "decision": decision, "reason": "joint test: " + decision})
                check(reply.get("ok") is True and reply.get("hard_negative_ready") is False,
                      f"{case['name']}: {decision} accepted and excluded",
                      json.dumps(reply)[:100])
            print(f"      {case['name']}: {row['negative_tile_id']} cam={row['camera_id']} "
                  f"{row['timestamp']} crop={row['source_crop_xyxy']} "
                  f"anchor={row['anchor_tile_xyxy']} -> {decision}")

        if excluded:
            reply = ui.post("/api/review",
                            {"candidate_id": excluded["negative_tile_id"],
                             "decision": "NEGATIVE_OK", "reason": ""})
            check("error" in reply,
                  "a §31-excluded candidate cannot be reviewed",
                  json.dumps(reply)[:120])
            print(f"      §31 excluded: {excluded['negative_tile_id']} "
                  f"detail={excluded['candidate_generation_detail']}")

    # ---- accepted pool for the reviewed subset -------------------------------- #
    rows = read_jsonl(OUTPUT / "negative_candidates.jsonl")
    from rtsp_annotator.ground_litter_hard_negatives import (
        NegativeReviewState, build_accepted,
    )

    subset = [row for row in rows
              if row["negative_tile_id"] in {c["row"]["negative_tile_id"] for c in cases}]
    state = NegativeReviewState.load(case_state)
    accepted = build_accepted(subset, work / "accepted", state=state,
                              positive_rows=positive,
                              positive_tiles_path=STEP1C2_ROOT / "tile_candidates.jsonl")
    check(accepted["accepted_tile_count"] == 3,
          "exactly the NEGATIVE_OK tiles are accepted",
          str(accepted["accepted_tile_count"]))
    check(accepted["conflict"]["ok"], "accepted pool has no positive conflict")
    for row in accepted["rows"]:
        source = [c for c in cases if c["row"]["negative_tile_id"] ==
                  row["negative_tile_id"]][0]["row"]
        image = Path(row["image_path"])
        label = Path(row["label_path"])
        check(sha256_file(image) == source["image_sha256"],
              f"{row['negative_tile_id']}: accepted bytes == candidate bytes")
        check(label.stat().st_size == 0 and label.read_bytes() == b"",
              f"{row['negative_tile_id']}: YOLO label is a 0-byte file")
    write_jsonl(work / "accepted_subset.jsonl", accepted["rows"])

    after = {name: hash_of(path) for name, path in UPSTREAM}
    check(before == after, "every upstream artifact is byte-identical",
          json.dumps({name: before[name] == after[name] for name in before}))
    real_state = OUTPUT / "review_state.json"
    if real_state.is_file():
        check(not json.loads(real_state.read_text()).get("decisions"),
              "the production review state was not touched by this test")

    print(f"\n{len(FAILURES)} failures" if FAILURES else "\nALL JOINT CHECKS PASSED")
    for failure in FAILURES:
        print(" -", failure)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
