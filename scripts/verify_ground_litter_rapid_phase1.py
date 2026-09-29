#!/usr/bin/env python3
"""Phase-1 smoke test for the Rapid v1 review UI, on a throwaway copy.

Verifies, over real HTTP and with the real server code, that:

1. Stage A is blind — no model output, not even a candidate count, for a frame whose truth
   is not complete;
2. a truth point round-trips to ``truth_points.jsonl``;
3. Stage B is gated — predictions appear only after ``truth_complete``;
4. prediction review writes verdicts and the ``M`` verdict reopens Stage A;
5. localization exposes A/B/C candidates and records the selection;
6. a Rapid-Eval Holdout frame is refused by the training-export guard (HTTP 409);
7. state survives a reload (resume).

The smoke artifact is a separate directory; the real Rapid artifact is hashed before and
after and must be unchanged.  Run on the machine that holds the extracted frames.

    python scripts/verify_ground_litter_rapid_phase1.py \
        --artifact ~/ground-litter-rapid-v1/artifact \
        --smoke-dir ~/ground-litter-rapid-v1/smoke/artifact --port 8811
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rtsp_annotator.ground_litter_rapid import (  # noqa: E402
    EVAL_SPLIT,
    TRAIN_SPLIT,
    read_jsonl,
    write_json,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--smoke-dir", type=Path, required=True)
    p.add_argument("--port", type=int, default=8811)
    p.add_argument("--no-semantic", action="store_true")
    p.add_argument("--keep", action="store_true")
    return p.parse_args(argv)


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return "absent"
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


class Client:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def get(self, path: str, **params):
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return self._call(url, None)

    def post(self, path: str, payload: dict):
        return self._call(self.base + path, json.dumps(payload).encode("utf-8"))

    def raw(self, path: str, **params):
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.status, response.read(), response.headers.get("Content-Type")

    def _call(self, url: str, body):
        request = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"} if body else {})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8")
            try:
                return error.code, json.loads(text)
            except json.JSONDecodeError:
                return error.code, {"raw": text}


def build_smoke_artifact(artifact: Path, smoke: Path) -> dict:
    if smoke.exists():
        shutil.rmtree(smoke)
    (smoke / "frames").mkdir(parents=True)
    (smoke / "baseline").mkdir(parents=True)
    shutil.copy2(artifact / "split.json", smoke / "split.json")

    manifest = read_jsonl(artifact / "frame_manifest.jsonl")
    predictions_all = {row["frame_id"]: row for row in
                       read_jsonl(artifact / "baseline" / "predictions.jsonl")}

    def pick(split: str, minimum: int = 2) -> dict:
        """Prefer a frame with >=2 predictions, so Stage B can exercise both M and Y."""
        fixed = [f for f in manifest if f["split"] == split and f["kind"] == "fixed"]
        counts = {f["frame_id"]: len(predictions_all.get(f["frame_id"], {})
                                         .get("predictions") or []) for f in fixed}
        for floor in (minimum, 1):
            candidates = [f for f in fixed if counts[f["frame_id"]] >= floor]
            if candidates:
                return candidates[0]
        raise SystemExit(f"no {split} frame with predictions; run the baseline first")

    train = pick(TRAIN_SPLIT, 2)
    # A second frame on the same PS: the copy-previous case the operator hits most often.
    same_ps = next(f for f in manifest
                   if f["split"] == TRAIN_SPLIT and f["kind"] == "fixed"
                   and f["file_id"] == train["file_id"] and f["frame_id"] != train["frame_id"])
    evaluation = pick(EVAL_SPLIT, 1)
    chosen = [train, same_ps, evaluation]

    extraction = json.loads((artifact / "extraction_manifest.json").read_text(encoding="utf-8"))
    records = {r["frame_id"]: r for r in extraction["records"]}
    smoke_records = []
    for frame in chosen:
        record = records[frame["frame_id"]]
        shutil.copy2(artifact / "frames" / record["image"], smoke / "frames" / record["image"])
        smoke_records.append(record)
    (smoke / "extraction_manifest.json").write_text(json.dumps({
        "records": smoke_records, "decode_failure_count": 0, "missing": [],
        "frames_requested": len(smoke_records), "frames_extracted": len(smoke_records),
        "split_sha256": json.loads((artifact / "split.json").read_text())["split_sha256"],
        "counts": {}, "anomaly_count": 0, "anomalies": [],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(smoke / "frame_manifest.jsonl", "w", encoding="utf-8") as handle:
        for frame in chosen:
            handle.write(json.dumps(frame, ensure_ascii=False) + "\n")

    predictions = predictions_all
    with open(smoke / "baseline" / "predictions.jsonl", "w", encoding="utf-8") as handle:
        for frame in chosen:
            handle.write(json.dumps(predictions[frame["frame_id"]], ensure_ascii=False) + "\n")
    return {"train_frame": train["frame_id"], "copy_frame": same_ps["frame_id"],
            "eval_frame": evaluation["frame_id"],
            "train_predictions": predictions[train["frame_id"]]["predictions"],
            "copy_predictions": predictions[same_ps["frame_id"]]["predictions"],
            "eval_predictions": predictions[evaluation["frame_id"]]["predictions"]}


def wait_port(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = args.artifact.resolve()
    smoke = args.smoke_dir.resolve()
    if smoke == artifact:
        raise SystemExit("smoke dir must not be the real artifact")

    import serve_ground_litter_rapid_review as server
    from http.server import ThreadingHTTPServer

    before = tree_digest(artifact)
    fixture = build_smoke_artifact(artifact, smoke)

    store, images, localizer = server.build(smoke, enable_semantic=not args.no_semantic)
    handler = server.make_handler(store, images, localizer, repo_root=Path(__file__).resolve().parents[1])
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    if not wait_port(args.port):
        raise SystemExit("smoke server did not start")
    client = Client(f"http://127.0.0.1:{args.port}")

    checks: list[dict] = []

    def check(name: str, ok: bool, detail=None):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f'{"PASS" if ok else "FAIL"}  {name}' + (f"  {detail}" if detail else ""), flush=True)

    train_frame = fixture["train_frame"]
    copy_frame = fixture["copy_frame"]
    eval_frame = fixture["eval_frame"]
    train_predictions = fixture["train_predictions"]
    copy_predictions = fixture["copy_predictions"]
    eval_predictions = fixture["eval_predictions"]

    try:
        status, health = client.get("/api/health")
        check("health", status == 200 and health.get("ok"), health)

        status, body, _content_type = client.raw("/api/frame_image", frame_id=train_frame)
        check("frame_image_is_jpeg", status == 200 and body[:2] == b"\xff\xd8",
              {"status": status, "bytes": len(body)})

        # --- 1. Stage A blindness ---------------------------------------- #
        status, state = client.get("/api/state", frame_id=train_frame)
        raw = json.dumps(state, ensure_ascii=False)
        leaked = [p["prediction_id"] for p in train_predictions if p["prediction_id"] in raw]
        coords_leaked = any(str(round(p["xyxy"][0], 3)) in raw for p in train_predictions)
        check("stage_a_predictions_hidden",
              state.get("predictions") is None and not leaked and not coords_leaked
              and "prediction_count" not in state,
              {"predictions": state.get("predictions"), "ids_leaked": leaked,
               "coords_leaked": coords_leaked})
        check("stage_a_stage_label", state.get("stage") == "truth", state.get("stage"))

        # --- 2. truth point round-trip ----------------------------------- #
        if train_predictions:
            first = train_predictions[0]["xyxy"]
            px, py = (first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0
        else:
            px, py = 1280.0, 720.0
        status, created = client.post("/api/truth", {
            "frame_id": train_frame, "truth_class": "REQUIRED_LITTER", "x": px, "y": py})
        truth_id = (created.get("point") or {}).get("truth_id")
        on_disk = read_jsonl(smoke / "review" / "truth_points.jsonl")
        check("truth_point_persisted",
              status == 200 and truth_id and any(p["truth_id"] == truth_id for p in on_disk),
              {"status": status, "truth_id": truth_id, "count": len(on_disk)})

        status, state = client.get("/api/state", frame_id=train_frame)
        check("stage_a_still_blind_with_point", state.get("predictions") is None
              and len(state.get("truth_points") or []) == 1, None)

        # --- 3/4. Stage B gating and review ------------------------------ #
        status, body = client.post("/api/prediction_review", {
            "frame_id": train_frame, "prediction_id": train_predictions[0]["prediction_id"],
            "verdict": "Y"})
        check("prediction_review_refused_before_truth_complete", status == 409, body)

        status, body = client.post("/api/truth_complete", {"frame_id": train_frame,
                                                           "complete": True})
        check("truth_complete", status == 200 and body["entry"]["truth_complete"], status)
        status, state = client.get("/api/state", frame_id=train_frame)
        check("stage_b_predictions_visible",
              state.get("predictions") is not None
              and len(state["predictions"]) == len(train_predictions)
              and state["stage"] == "prediction",
              {"count": len(state.get("predictions") or []), "stage": state["stage"]})

        if train_predictions:
            prediction_id = train_predictions[0]["prediction_id"]
            status, body = client.post("/api/prediction_review", {
                "frame_id": train_frame, "prediction_id": prediction_id, "verdict": "Y"})
            check("prediction_review_recorded", status == 200, body.get("error"))
        else:
            prediction_id = None
            check("prediction_review_recorded", True, "no predictions on this frame")

        # --- 5. localization A/B/C --------------------------------------- #
        status, body = client.post("/api/localize", {"truth_id": truth_id,
                                                     "frame_id": train_frame})
        candidates = body.get("candidates") or []
        sources = [c["proposal_source"] for c in candidates]
        check("localization_candidates_returned", status == 200 and len(candidates) >= 1,
              {"sources": sources, "semantic": body.get("semantic_note")})
        check("localization_candidate_A_from_judged_Y",
              "baseline_judged_Y" in sources or not train_predictions,
              {"sources": sources})
        if candidates:
            status, raw_bytes, content_type = client.raw(
                "/api/candidate_image", truth_id=truth_id, idx=0)
            check("candidate_crop_is_jpeg", status == 200 and raw_bytes[:2] == b"\xff\xd8",
                  {"bytes": len(raw_bytes), "content_type": content_type})
            status, body = client.post("/api/localize_select", {
                "truth_id": truth_id, "frame_id": train_frame, "choice": 1})
            check("localization_selection_recorded",
                  status == 200 and body["selection"]["status"] == "LOCALIZED",
                  body.get("selection"))
            check("localization_frame_scoped",
                  body.get("frame_id") == train_frame
                  and body["selection"]["frame_id"] == train_frame, body.get("frame_id"))
            check("localization_response_contract",
                  all(key in body for key in ("ok", "truth_id", "frame_id",
                                              "localization_complete", "remaining",
                                              "next_truth_id")),
                  {k: body.get(k) for k in ("ok", "remaining", "localization_complete",
                                            "next_truth_id")})
            # Idempotency: submitting the same choice again must not add a second decision.
            status, again = client.post("/api/localize_select", {
                "truth_id": truth_id, "frame_id": train_frame, "choice": 1})
            rows = [r for r in read_jsonl(smoke / "review" / "localization_reviews.jsonl")
                    if r["truth_id"] == truth_id]
            check("localization_duplicate_submit_idempotent",
                  status == 200 and again.get("already_recorded") is True and len(rows) == 1,
                  {"already_recorded": again.get("already_recorded"), "rows": len(rows)})
        else:
            status, body = client.post("/api/localize_select", {"truth_id": truth_id,
                                                               "choice": 0})
            check("localization_selection_recorded",
                  status == 200 and body["selection"]["status"] == "UNLOCALIZED_SKIP",
                  body.get("selection"))

        on_disk = read_jsonl(smoke / "review" / "localization_reviews.jsonl")
        check("localization_review_persisted", len(on_disk) == 1, on_disk)

        # --- 6. training-export guard ------------------------------------ #
        status, body = client.post("/api/train_export_probe", {"frame_id": eval_frame})
        check("rapid_eval_export_refused_409",
              status == 409 and "rapid_eval" in json.dumps(body), body)
        status, body = client.post("/api/train_export_probe", {"frame_id": train_frame})
        check("rapid_train_export_allowed", status == 200 and body.get("exported"), body)

        # --- 7. resume --------------------------------------------------- #
        resumed = server.RapidReviewStore(smoke)
        resumed.load_predictions()
        check("resume_truth_persisted",
              resumed.frame_state(train_frame)["truth_complete"]
              and len(resumed.points_for(train_frame)) == 1, None)
        check("resume_review_persisted",
              resumed.reviews_for(train_frame).get(prediction_id) == "Y" or not prediction_id,
              resumed.reviews_for(train_frame))
        status, progress = client.get("/api/progress")
        check("progress_shape", status == 200
              and progress["truth_review"]["total"] == 3
              and progress["rapid_eval"]["total"] == 1, progress["truth_review"])

        # --- 8. copy previous (C) ---------------------------------------- #
        status, state = client.get("/api/state", frame_id=copy_frame)
        source = (state.get("copy") or {}).get("source") or {}
        check("copy_available_same_ps", source.get("allowed") is True
              and source.get("reason") == "same_ps", source)

        status, body = client.post("/api/copy_previous", {"frame_id": copy_frame})
        copy_info = body.get("copy") or {}
        check("copy_returns_pending_not_complete",
              status == 200 and copy_info.get("copy_state") == "COPIED_PENDING_CONFIRM"
              and copy_info.get("truth_complete") is False, copy_info.get("copy_state"))

        status, state = client.get("/api/state", frame_id=copy_frame)
        check("copied_frame_stays_blind", state.get("predictions") is None
              and state.get("truth_complete") is False
              and len(state.get("truth_points") or []) == 1,
              {"points": len(state.get("truth_points") or []),
               "predictions": state.get("predictions")})
        check("copied_point_marked", all(p.get("origin") == "copied_from_previous_frame"
                                        for p in state.get("truth_points") or []), None)

        on_disk = read_jsonl(smoke / "review" / "truth_points.jsonl")
        copied_on_disk = [p for p in on_disk if p["frame_id"] == copy_frame]
        source_on_disk = [p for p in on_disk
                          if p["frame_id"] == train_frame
                          and p.get("origin") == "rapid_v1_review"]
        check("copied_coordinates_verbatim",
              len(copied_on_disk) == len(source_on_disk) == 1
              and copied_on_disk[0]["source_xy"] == source_on_disk[0]["source_xy"]
              and copied_on_disk[0]["truth_class"] == source_on_disk[0]["truth_class"],
              {"source": source_on_disk[0]["source_xy"],
               "copied": copied_on_disk[0]["source_xy"]})

        status, body = client.post("/api/truth_complete", {"frame_id": copy_frame,
                                                           "complete": True})
        check("enter_confirms_copy",
              status == 200 and body["entry"]["truth_complete"] is True
              and body["entry"].get("copy_state") == "CONFIRMED_COPY",
              body["entry"].get("copy_state"))
        status, state = client.get("/api/state", frame_id=copy_frame)
        check("stage_b_opens_after_confirm", state.get("predictions") is not None
              and state["stage"] in ("prediction", "localization", "done"), state["stage"])

        # --- 9. training export plan ------------------------------------- #
        status, plan = client.post("/api/export_plan", {})
        summary = plan.get("summary") or {}
        check("export_plan_excludes_eval",
              status == 200 and plan.get("rapid_eval_denominator", {}).get("frames") == 1
              and plan.get("rapid_eval_denominator", {})
              .get("excluded_from_training") is True, summary)
        check("export_plan_has_dedup_parameters",
              status == 200 and "positives" in plan and "hard_negatives" in plan
              and plan["positives"]["parameters"]["max_per_cluster"] >= 1,
              plan.get("max_per_cluster"))

        # --- M verdict reopens Stage A ----------------------------------- #
        if eval_predictions:
            status, body = client.post("/api/truth", {
                "frame_id": eval_frame, "truth_class": "REQUIRED_LITTER",
                "x": (eval_predictions[0]["xyxy"][0] + eval_predictions[0]["xyxy"][2]) / 2.0,
                "y": (eval_predictions[0]["xyxy"][1] + eval_predictions[0]["xyxy"][3]) / 2.0})
            client.post("/api/truth_complete", {"frame_id": eval_frame, "complete": True})
            status, body = client.post("/api/prediction_review", {
                "frame_id": eval_frame,
                "prediction_id": eval_predictions[0]["prediction_id"], "verdict": "M"})
            status, state = client.get("/api/state", frame_id=eval_frame)
            check("m_verdict_reopens_stage_a",
                  state["truth_complete"] is False and state["predictions"] is None
                  and state["stage"] == "truth", state["stage"])
    finally:
        httpd.shutdown()
        httpd.server_close()

    after = tree_digest(artifact)
    check("real_artifact_untouched", before == after,
          "unchanged" if before == after else "REAL ARTIFACT CHANGED")

    failures = [c for c in checks if not c["ok"]]
    report = {
        "phase": "ground-litter-rapid-v1-phase1-smoke",
        "smoke_dir": str(smoke),
        "real_artifact": str(artifact),
        "real_artifact_digest_before": before,
        "real_artifact_digest_after": after,
        "fixture": {"train_frame": train_frame, "copy_frame": copy_frame,
                    "eval_frame": eval_frame,
                    "train_predictions": len(train_predictions),
                    "copy_predictions": len(copy_predictions),
                    "eval_predictions": len(eval_predictions)},
        "checks": checks,
        "failed": [c["check"] for c in failures],
        "result": "PASS" if not failures else "FAIL",
    }
    write_json(smoke / "smoke_report.json", report)
    print(f'\nSMOKE {report["result"]}: {len(checks) - len(failures)}/{len(checks)} checks '
          f'passed; report at {smoke / "smoke_report.json"}')
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
