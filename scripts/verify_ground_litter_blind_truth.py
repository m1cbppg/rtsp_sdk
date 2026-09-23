#!/usr/bin/env python3
"""Step 2C-1 joint test: real Development PS -> blind point truth -> episode -> sampling ->
localization -> frozen artifact, through the real HTTP UI, with no detector anywhere.

This is the §31 small-scale integration check.  It is meant to run on the machine that
holds the Development PS.  It deliberately uses a *pilot* output directory: the points it
records are geometric smoke-test points (the ROI centroid), never the operator's truth,
and the pilot state stays separate from the official Blind Truth artifact.

    python scripts/verify_ground_litter_blind_truth.py \
        --development-root <dir containing .../development/<cam>/raw/*.ps> \
        --roi-dir output/ground_litter_final_roi_20260922/config \
        --output /tmp/pilot-blinde-truth --port 8811
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_blind_truth import (  # noqa: E402
    IGNORE_SMALL,
    LOCALIZATION_BOX_OK,
    LOCALIZATION_POINT,
    LOCALIZATION_PROPOSAL,
    REQUIRED_LITTER,
    REVIEW_DONE,
    SCHEMA_VERSION,
    assert_development_asset,
    read_jsonl,
)

CLI = ROOT / "scripts" / "serve_ground_litter_blind_truth.py"
JPEG_MAGIC = b"\xff\xd8\xff"


def get(base: str, path: str) -> tuple[int, bytes, str]:
    request = urllib.request.Request(base + path)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, response.read(), response.headers.get("content-type", "")
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers.get("content-type", "")


def post(base: str, path: str, payload: dict) -> tuple[int, bytes]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(base + path, data=data,
                                     headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def wait_for_port(port: int, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.4)
    raise SystemExit(f"UI server did not open port {port}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--roi-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8811)
    args = parser.parse_args(argv)

    assert_development_asset(args.development_root)
    args.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"schema_version": SCHEMA_VERSION,
                                  "development_root": str(args.development_root),
                                  "output": str(args.output), "steps": {}}
    server_log = args.output / "joint_test_server.log"
    log_handle = server_log.open("w", encoding="utf-8")
    server = subprocess.Popen(
        [sys.executable, "-B", str(CLI), "--development-root", str(args.development_root),
         "--roi-dir", str(args.roi_dir), "--output", str(args.output),
         "--bind", "127.0.0.1", "--port", str(args.port), "serve"],
        stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{args.port}"
    try:
        wait_for_port(args.port)
        status, body, _ = get(base, "/api/inventory")
        inventory = json.loads(body)
        results["steps"]["inventory"] = {"status": status, "ps_count": inventory["ps_count"],
                                         "cameras": inventory["cameras"]}
        assert status == 200 and inventory["ps_count"] >= 1, inventory

        file_id = inventory["files"][0]["file_id"]
        camera_id = inventory["files"][0]["camera_id"]
        status, body, _ = get(base, f"/api/frame-meta?file_id={file_id}&t=60")
        meta = json.loads(body)
        results["steps"]["frame_meta"] = {"status": status, **{
            key: meta[key] for key in ("source_width", "source_height", "decoded_timestamp")}}
        assert status == 200 and meta["source_width"] == 2560 and meta["source_height"] == 1440

        status, body, content_type = get(base, f"/api/frame?file_id={file_id}&t=60&w=1280")
        results["steps"]["frame"] = {"status": status, "bytes": len(body),
                                     "content_type": content_type,
                                     "jpeg": body.startswith(JPEG_MAGIC)}
        assert status == 200 and body.startswith(JPEG_MAGIC) and len(body) > 20000

        roi = inventory["files"][0]["roi"]
        cx = sum(point[0] for point in roi) / len(roi) * 2560
        cy = sum(point[1] for point in roi) / len(roi) * 1440
        status, body, _ = get(base, f"/api/crop?file_id={file_id}&t=60&cx={cx}&cy={cy}"
                                    f"&half=220")
        results["steps"]["crop"] = {"status": status, "bytes": len(body),
                                    "jpeg": body.startswith(JPEG_MAGIC),
                                    "source_point": [round(cx, 1), round(cy, 1)]}
        assert status == 200 and body.startswith(JPEG_MAGIC)

        status, body, _ = get(base, f"/api/proposals?file_id={file_id}&t=60&x={cx}&y={cy}")
        proposals = json.loads(body)["proposals"]
        results["steps"]["classic_cv_proposals"] = {
            "status": status, "count": len(proposals),
            "labels": [row["label"] for row in proposals],
            "rule": json.loads(body)["rule"]}
        assert status == 200 and len(proposals) <= 3

        # geometric smoke-test point (the ROI centroid), explicitly not operator truth
        status, body = post(base, "/api/truth", {
            "action": "add", "truth_class": REQUIRED_LITTER, "camera_id": camera_id,
            "source_file_id": file_id, "decoded_timestamp": 60.0,
            "source_point": [cx, cy], "note": "JOINT_TEST smoke point (ROI centroid)"})
        required = json.loads(body)
        results["steps"]["truth_required"] = {"status": status,
                                              "truth_id": required["truth_id"],
                                              "localization_status": required["localization_status"],
                                              "in_roi": required["in_roi"],
                                              "timestamp": required["timestamp"]}
        assert status == 200 and required["in_roi"] is True
        assert required["localization_status"] == LOCALIZATION_POINT

        status, body = post(base, "/api/truth", {
            "action": "add", "truth_class": IGNORE_SMALL, "camera_id": camera_id,
            "source_file_id": file_id, "decoded_timestamp": 62.0,
            "source_point": [cx + 40, cy + 30],
            "note": "JOINT_TEST smoke point (ignore semantics)"})
        ignore = json.loads(body)
        results["steps"]["truth_ignore"] = {"status": status,
                                            "truth_id": ignore["truth_id"],
                                            "enters_recall_denominator":
                                                ignore["enters_recall_denominator"],
                                            "enters_ignore_set": ignore["enters_ignore_set"]}
        assert ignore["enters_recall_denominator"] is False
        assert ignore["enters_ignore_set"] is True

        status, body = post(base, "/api/episode",
                            {"action": "new", "truth_id": required["truth_id"]})
        episode = json.loads(body)
        status2, body2 = post(base, "/api/episode", {"action": "confirm",
                                                     "episode_id": episode["episode_id"]})
        results["steps"]["episode"] = {"new_status": status, "confirm_status": status2,
                                       "episode_id": episode["episode_id"]}
        assert status2 == 200

        status, body = post(base, "/api/truth", {"action": "set_interval",
                                                 "truth_id": ignore["truth_id"],
                                                 "episode_id": episode["episode_id"],
                                                 "which": "end"})
        results["steps"]["interval"] = {"status": status, "reply": json.loads(body)}

        if proposals:
            status, body = post(base, "/api/truth", {
                "action": "update", "truth_id": required["truth_id"],
                "source_bbox_xyxy": proposals[0]["bbox_xyxy"],
                "localization_status": LOCALIZATION_PROPOSAL})
            updated = json.loads(body)
        else:
            status, body = post(base, "/api/truth", {
                "action": "update", "truth_id": required["truth_id"],
                "localization_status": LOCALIZATION_PROPOSAL})
            updated = json.loads(body)
        results["steps"]["localization"] = {
            "status": status, "localization_status": updated["localization_status"],
            "has_bbox": bool(updated.get("source_bbox_xyxy"))}
        assert status == 200

        for row in inventory["files"]:
            post(base, "/api/review", {"file_id": row["file_id"], "status": REVIEW_DONE})
        status, body, _ = get(base, "/api/state")
        state = json.loads(body)
        results["steps"]["review"] = {"status": status,
                                      "coverage": state["coverage"]["reviewed_fraction"],
                                      "complete": state["coverage"]["complete"]}
        assert state["coverage"]["complete"] is True

        status, body = post(base, "/api/sample", {})
        samples = json.loads(body)
        results["steps"]["sample"] = {"status": status, **samples}
        assert samples["visible_frames"] >= 1 and samples["global_frames"] >= 1

        status, body = post(base, "/api/freeze", {"code_commit": "joint-test"})
        frozen = json.loads(body)
        results["steps"]["freeze"] = {"status": status,
                                      "truth_sha256": frozen["truth_sha256"],
                                      "artifacts": sorted(frozen["artifact_sha256"])}
        assert status == 200 and len(frozen["truth_sha256"]) == 64

        status, body = post(base, "/api/truth", {
            "action": "add", "truth_class": REQUIRED_LITTER, "camera_id": camera_id,
            "source_file_id": file_id, "decoded_timestamp": 70.0, "source_point": [cx, cy]})
        results["steps"]["post_freeze_rejected"] = {"status": status,
                                                    "body": body[:160].decode("utf-8",
                                                                              "replace")}
        assert status == 409, (status, body)

        visible = read_jsonl(args.output / "visible_frame_manifest.jsonl")
        global_frames = read_jsonl(args.output / "global_roi_frame_manifest.jsonl")
        truth = read_jsonl(args.output / "truth_objects.jsonl")
        results["steps"]["artifacts"] = {
            "truth_objects": len(truth), "visible_frames": len(visible),
            "global_frames": len(global_frames),
            "visible_sample_reason": sorted({row["sample_reason"] for row in visible}),
            "global_sample_reason": sorted({row["sample_reason"] for row in global_frames}),
            "visible_source_size": sorted({(row["source_width"], row["source_height"])
                                           for row in visible}),
            "read_only": {path.name: oct(path.stat().st_mode & 0o777)
                          for path in sorted(args.output.glob("*.json"))}}
        for path in args.output.glob("truth_objects.jsonl"):
            assert path.stat().st_mode & 0o222 == 0, "truth is not read-only"
        results["ok"] = True
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:                    # pragma: no cover
            server.kill()
        log_handle.close()
        text = server_log.read_text(encoding="utf-8", errors="replace") if server_log.is_file() else ""
        results["server_log_tail"] = text[-1200:]
        results["server_log_bytes"] = len(text)
    (args.output / "joint_test.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if results.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
