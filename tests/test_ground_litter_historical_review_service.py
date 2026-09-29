"""Tests for the Ground Litter Historical Review v2 web service.

Deliberately free of cv2/numpy/torch and of any network access: the whole HTTP
contract is exercised on a synthetic artifact served from an ephemeral port in a
background thread.  PNG frames are written with a tiny pure-python encoder so the
suite runs in the plain repo venv (which has no importable cv2).

pytest command:
    .venv/bin/python -m pytest tests/test_ground_litter_historical_review_service.py -q
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import struct
import sys
import tempfile
import threading
import unittest
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import serve_ground_litter_historical_review as server  # noqa: E402

PNG_WIDTH = 320
PNG_HEIGHT = 180
ROI = [[0.10, 0.20], [0.90, 0.20], [0.90, 0.85], [0.10, 0.85]]


# --------------------------------------------------------------------------- #
# Pure-python PNG writer (no cv2)
# --------------------------------------------------------------------------- #


def write_png(path: Path, width: int = PNG_WIDTH, height: int = PNG_HEIGHT,
              rgb: tuple[int, int, int] = (28, 62, 96)) -> None:
    raw = bytearray()
    row = bytes(rgb) * width
    for _ in range(height):
        raw.append(0)  # filter type 0
        raw += row

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit truecolor
    payload = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
               chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


# --------------------------------------------------------------------------- #
# Synthetic artifact
# --------------------------------------------------------------------------- #


def _frame(frame_id: str, image: str, camera: str = "01021") -> dict:
    return {
        "frame_id": frame_id,
        "camera_id": camera,
        "window_id": "w-1",
        "date": "2026-09-23",
        "day_split": "TRAIN",
        "selection_bucket": "mid",
        "offset_seconds": 12.5,
        "frame_index": 100,
        "width": PNG_WIDTH,
        "height": PNG_HEIGHT,
        "record_start": "2026-09-23 06:00:00",
        "image": image,
        "image_sha256": "0" * 64,
        "roi": ROI,
        "roi_geometry_version": "test-v1",
    }


def _observation(observation_id: str, frame_id: str, role: str = "normal",
                 bbox=(120.0, 80.0, 160.0, 120.0), ) -> dict:
    return {
        "observation_id": observation_id,
        "frame_id": frame_id,
        "role": role,
        "offset_seconds": 12.5,
        "bbox_xyxy": list(bbox),
        "source_label": "both",
        "confidence_by_source": {"turhancan": 0.42},
        "contains_point": True,
    }


def build_artifact(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)

    frames = [
        _frame("f-0001", "frames/f-0001.png"),
        _frame("f-0002a", "frames/f-0002a.png"),
        _frame("f-0002b", "frames/f-0002b.png"),
        _frame("f-0003", "frames/f-0003.png"),
        _frame("f-0004", "frames/f-0004.png"),
    ]
    with open(root / "coarse_frames.jsonl", "w", encoding="utf-8") as handle:
        for row in frames:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in frames:
        write_png(root / row["image"])

    units = [
        {
            "unit_id": "rg-00001", "kind": "candidate", "camera_id": "01021",
            "day_split": "TRAIN", "date": "2026-09-23",
            "timestamp": "2026-09-23 06:15:30", "tier": "P1", "batch": 1,
            "queue_index": 0, "blind": False, "suspected_same_object": False,
            "episode_count": 1, "representative_observation_id": "ob-00001",
            "observations": [_observation("ob-00001", "f-0001")],
            "candidates": [
                {"candidate_id": "A", "source": "semantic", "label": "turhancan semantic",
                 "bbox_xyxy": [118.0, 78.0, 162.0, 122.0], "contains_point": True},
                {"candidate_id": "B", "source": "yolo", "label": "yolo litter",
                 "bbox_xyxy": [110.0, 70.0, 170.0, 130.0], "contains_point": False},
            ],
            "buckets": ["camera:01021"],
        },
        {
            "unit_id": "rg-00002", "kind": "candidate", "camera_id": "01021",
            "day_split": "TRAIN", "date": "2026-09-23",
            "timestamp": "2026-09-23 06:35:00", "tier": "P2", "batch": 1,
            "queue_index": 1, "blind": False, "suspected_same_object": True,
            "episode_count": 2, "representative_observation_id": "ob-00012",
            "observations": [
                _observation("ob-00011", "f-0002a", "normal"),
                _observation("ob-00012", "f-0002b", "shadow_change",
                             bbox=(130.0, 90.0, 150.0, 110.0)),
            ],
            "candidates": [
                {"candidate_id": "C", "source": "classical", "label": "classical change",
                 "bbox_xyxy": [126.0, 86.0, 154.0, 114.0], "contains_point": True},
            ],
            "buckets": ["camera:01021"],
        },
        {
            "unit_id": "rg-00003", "kind": "blind", "camera_id": "01021",
            "day_split": "TRAIN", "date": "2026-09-23",
            "timestamp": "2026-09-23 07:10:00", "tier": "BLIND", "batch": 1,
            "queue_index": 2, "blind": True, "suspected_same_object": False,
            "episode_count": 1, "representative_observation_id": "ob-00021",
            "observations": [_observation("ob-00021", "f-0003")],
            "candidates": [
                {"candidate_id": "A", "source": "semantic", "label": "turhancan semantic",
                 "bbox_xyxy": [115.0, 75.0, 165.0, 125.0], "contains_point": True},
                {"candidate_id": "B", "source": "yolo", "label": "yolo litter",
                 "bbox_xyxy": [112.0, 72.0, 168.0, 128.0], "contains_point": False},
                {"candidate_id": "C", "source": "classical", "label": "classical change",
                 "bbox_xyxy": [120.0, 80.0, 160.0, 120.0], "contains_point": False},
            ],
            "buckets": ["camera:01021"],
        },
        {
            "unit_id": "rg-00004", "kind": "candidate", "camera_id": "01021",
            "day_split": "DEV", "date": "2026-09-24",
            "timestamp": "2026-09-24 08:05:00", "tier": "LOW", "batch": 2,
            "queue_index": 3, "blind": False, "suspected_same_object": False,
            "episode_count": 1, "representative_observation_id": "ob-00031",
            "observations": [_observation("ob-00031", "f-0004")],
            "candidates": [],
            "buckets": ["camera:01021"],
        },
    ]
    with open(root / "review_units.jsonl", "w", encoding="utf-8") as handle:
        for row in units:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# HTTP harness
# --------------------------------------------------------------------------- #


class ServiceTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "artifact"
        build_artifact(self.root)
        self.store = server.ReviewStore(self.root)
        self.renderer = server.MediaRenderer(self.store)
        handler = server.make_handler(self.store, self.renderer)
        from http.server import ThreadingHTTPServer
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        self._tmp.cleanup()

    def request(self, method: str, path: str, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Content-Type": "application/json"} if body is not None else {}
        payload = json.dumps(body) if body is not None else None
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data

    def get_json(self, path: str):
        status, data = self.request("GET", path)
        self.assertEqual(status, 200, msg=data[:300])
        return json.loads(data.decode("utf-8"))

    def post_json(self, path: str, body):
        status, data = self.request("POST", path, body)
        return status, (json.loads(data.decode("utf-8")) if data else None)

    def decide(self, unit_id: str, verdict: str, expect: int = 200):
        status, payload = self.post_json("/api/decide", {"unit_id": unit_id, "verdict": verdict})
        self.assertEqual(status, expect, msg=payload)
        return payload

    def bbox(self, unit_id: str, choice: int, expect: int = 200):
        status, payload = self.post_json("/api/bbox", {"unit_id": unit_id, "choice": choice})
        self.assertEqual(status, expect, msg=payload)
        return payload

    def link(self, unit_id: str, decision: str, expect: int = 200):
        status, payload = self.post_json("/api/link",
                                         {"unit_id": unit_id, "decision": decision})
        self.assertEqual(status, expect, msg=payload)
        return payload


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class QueueTests(ServiceTestCase):
    def test_queue_ordering_and_batch_filter(self):
        payload = self.get_json("/api/queue?batch=1&offset=0&limit=10")
        self.assertEqual(payload["batch"], 1)
        self.assertEqual(payload["total"], 3)
        self.assertEqual([u["unit_id"] for u in payload["units"]],
                         ["rg-00001", "rg-00002", "rg-00003"])
        self.assertEqual([u["queue_index"] for u in payload["units"]], [0, 1, 2])
        self.assertEqual(payload["units"][0]["thumb_url"], "/media/rg-00001/context.jpg")
        self.assertIsNone(payload["units"][0]["verdict"])
        self.assertFalse(payload["units"][0]["complete"])

        page = self.get_json("/api/queue?batch=1&offset=1&limit=1")
        self.assertEqual(page["total"], 3)
        self.assertEqual([u["unit_id"] for u in page["units"]], ["rg-00002"])

        batch2 = self.get_json("/api/queue?batch=2&offset=0&limit=10")
        self.assertEqual(batch2["total"], 1)
        self.assertEqual(batch2["units"][0]["unit_id"], "rg-00004")

    def test_state_progress_shape(self):
        state = self.get_json("/api/state")
        for key in ("total", "decided", "bbox_done", "complete", "remaining", "model_revealed"):
            self.assertIn(key, state["progress"])
        self.assertEqual(state["progress"]["total"], 4)
        self.assertEqual(state["progress"]["remaining"], 4)
        self.assertEqual(state["counts_by_verdict"]["R"], 0)
        self.assertEqual(state["counts_by_verdict"]["undecided"], 4)

    def test_favicon_is_204(self):
        status, data = self.request("GET", "/favicon.ico")
        self.assertEqual(status, 204)
        self.assertEqual(data, b"")


class BlindnessTests(ServiceTestCase):
    def test_blind_hides_candidates_before_decision(self):
        payload = self.get_json("/api/unit?unit_id=rg-00003")
        self.assertTrue(payload["blind"])
        self.assertFalse(payload["model_revealed"])
        self.assertEqual(payload["candidates"], [])
        self.assertEqual(payload["images"]["candidate_urls"], {})
        # observation/context URLs are not candidate-derived and stay visible
        self.assertTrue(payload["images"]["observation_urls"])

    def test_blind_candidate_media_404_until_revealed(self):
        status, _ = self.request("GET", "/media/rg-00003/cand_A.jpg")
        self.assertEqual(status, 404)

        self.decide("rg-00003", "I")
        payload = self.get_json("/api/unit?unit_id=rg-00003")
        self.assertTrue(payload["model_revealed"])
        self.assertEqual(len(payload["candidates"]), 3)
        self.assertIn("A", payload["images"]["candidate_urls"])

        after, _ = self.request("GET", "/media/rg-00003/cand_A.jpg")
        # 200 with cv2, explicit "unavailable" otherwise. Never a silent 404.
        self.assertIn(after, (200, 503))

    def test_non_blind_unit_always_revealed(self):
        payload = self.get_json("/api/unit?unit_id=rg-00001")
        self.assertTrue(payload["model_revealed"])
        self.assertEqual([c["candidate_id"] for c in payload["candidates"]], ["A", "B"])


class MediaRenderingTests(ServiceTestCase):
    def test_media_contract_renders_or_degrades_clearly(self):
        # rg-00002 is non-blind, so every media kind is reachable without a verdict.
        paths = [
            "/media/rg-00002/context.jpg",
            "/media/rg-00002/crop.jpg",
            "/media/rg-00002/obs_ob-00011.jpg",
            "/media/rg-00002/cand_C.jpg",
        ]
        for path in paths:
            status, data = self.request("GET", path)
            if self.renderer.available():
                self.assertEqual(status, 200, msg=f"{path}: {data[:200]!r}")
                self.assertTrue(data.startswith(b"\xff\xd8\xff"), msg=path)  # JPEG SOI
            else:
                self.assertEqual(status, 503, msg=path)
                payload = json.loads(data.decode("utf-8"))
                self.assertTrue(payload["unavailable"])
                self.assertFalse(payload["cv2"])

    def test_context_caching_is_deterministic(self):
        first = self.request("GET", "/media/rg-00001/context.jpg")
        second = self.request("GET", "/media/rg-00001/context.jpg")
        self.assertEqual(first, second)


class StageTransitionTests(ServiceTestCase):
    def test_verdict_r_requires_bbox(self):
        payload = self.decide("rg-00001", "R")
        self.assertEqual(payload["stage"], "bbox")
        self.assertFalse(payload["complete"])

        unit = self.get_json("/api/unit?unit_id=rg-00001")
        self.assertEqual(unit["decision"], "R")
        self.assertEqual(unit["stage"], "bbox")
        self.assertIsNone(unit["bbox_choice"])

        done = self.bbox("rg-00001", 1)
        self.assertEqual(done["stage"], "done")
        self.assertTrue(done["complete"])
        self.assertEqual(done["bbox_choice"], 1)

    def test_bbox_zero_is_unlocalized_and_completes(self):
        self.decide("rg-00002", "R")
        payload = self.bbox("rg-00002", 0)
        self.assertEqual(payload["bbox_choice"], 0)
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["stage"], "done")

    def test_i_u_n_complete_immediately(self):
        for unit_id, verdict in (("rg-00001", "I"), ("rg-00002", "U"), ("rg-00003", "N")):
            payload = self.decide(unit_id, verdict)
            self.assertEqual(payload["stage"], "done")
            self.assertTrue(payload["complete"])

    def test_bbox_before_verdict_is_409(self):
        status, payload = self.post_json("/api/bbox", {"unit_id": "rg-00001", "choice": 1})
        self.assertEqual(status, 409)
        self.assertFalse(payload["ok"])
        self.assertIn("REQUIRED", payload["error"])

    def test_invalid_verdict_is_400(self):
        status, payload = self.post_json("/api/decide", {"unit_id": "rg-00001", "verdict": "X"})
        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])

    def test_bbox_choice_beyond_candidates_is_409(self):
        self.decide("rg-00004", "R")  # rg-00004 has no candidates
        status, payload = self.post_json("/api/bbox", {"unit_id": "rg-00004", "choice": 2})
        self.assertEqual(status, 409)
        self.assertFalse(payload["ok"])


class LinkTests(ServiceTestCase):
    def test_link_requires_suspected_same_object(self):
        status, payload = self.post_json("/api/link",
                                         {"unit_id": "rg-00001", "decision": "SAME"})
        self.assertEqual(status, 409)
        self.assertFalse(payload["ok"])

    def test_link_records_for_suspected_unit(self):
        payload = self.link("rg-00002", "SAME")
        self.assertEqual(payload["link_decision"], "SAME")
        unit = self.get_json("/api/unit?unit_id=rg-00002")
        self.assertEqual(unit["link_decision"], "SAME")
        self.assertTrue(unit["suspected_same_object"])


class PersistenceTests(ServiceTestCase):
    def test_redecide_is_idempotent_and_single_row(self):
        self.decide("rg-00001", "I")
        self.decide("rg-00001", "I")
        first = (self.root / "review" / "decisions.jsonl").read_text(encoding="utf-8")
        self.assertEqual(len([ln for ln in first.splitlines() if ln.strip()]), 1)

        self.decide("rg-00001", "N")
        rows = [json.loads(ln) for ln in
                (self.root / "review" / "decisions.jsonl").read_text(
                    encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["unit_id"], "rg-00001")
        self.assertEqual(rows[0]["verdict"], "N")

    def test_state_survives_store_reload(self):
        self.decide("rg-00001", "I")
        self.decide("rg-00002", "R")
        self.bbox("rg-00002", 1)

        reloaded = server.ReviewStore(self.root)
        self.assertEqual(reloaded.decision_of("rg-00001"), "I")
        self.assertTrue(reloaded.is_complete("rg-00001"))
        self.assertEqual(reloaded.decision_of("rg-00002"), "R")
        self.assertEqual(reloaded.bbox_choice_of("rg-00002"), 1)
        self.assertTrue(reloaded.is_complete("rg-00002"))
        self.assertEqual(reloaded.stage_of("rg-00002"), "done")

        progress = reloaded.progress()
        self.assertEqual(progress["decided"], 2)
        self.assertEqual(progress["bbox_done"], 1)
        self.assertEqual(progress["complete"], 2)
        self.assertEqual(progress["remaining"], 2)

    def test_writes_stay_inside_artifact_root(self):
        self.decide("rg-00001", "R")
        self.bbox("rg-00001", 2)
        self.link("rg-00002", "NEW")
        review_files = sorted(p.name for p in (self.root / "review").glob("*.jsonl"))
        self.assertEqual(review_files,
                         ["bbox_decisions.jsonl", "decisions.jsonl", "link_decisions.jsonl"])

    def test_bbox_decision_records_candidate_id(self):
        self.decide("rg-00002", "R")
        self.bbox("rg-00002", 1)
        rows = [json.loads(ln) for ln in
                (self.root / "review" / "bbox_decisions.jsonl").read_text(
                    encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["choice"], 1)
        self.assertEqual(rows[0]["candidate_id"], "C")


class IsolationTests(ServiceTestCase):
    def test_media_traversal_rejected(self):
        for path in ("/media/../secret.jpg", "/media/rg-00001/../../secret.jpg"):
            status, _ = self.request("GET", path)
            self.assertEqual(status, 403, msg=path)

    def test_unknown_unit_and_name_404(self):
        status, _ = self.request("GET", "/api/unit?unit_id=does-not-exist")
        self.assertEqual(status, 404)
        status, _ = self.request("GET", "/media/rg-99999/context.jpg")
        self.assertEqual(status, 404)
        status, _ = self.request("GET", "/media/rg-00001/bogus.jpg")
        self.assertEqual(status, 404)
        status, _ = self.request("GET", "/media/rg-00001/obs_missing.jpg")
        self.assertEqual(status, 404)
        status, _ = self.request("GET", "/media/rg-00001/cand_Z.jpg")
        self.assertEqual(status, 404)

    def test_forbidden_roots_refused(self):
        for forbidden in ("/home/sf01/step2c1-blind-truth/artifact", "/tmp/sealed-data"):
            with self.assertRaises(server.ReviewError):
                server.assert_artifact_root(forbidden)

    def test_resolve_under_refuses_escapes(self):
        with self.assertRaises(server.ClientError):
            server.resolve_under(self.root, "../elsewhere.jsonl")
        with self.assertRaises(server.ClientError):
            server.resolve_under(self.root, "/etc/passwd")


class SelfTestCommandTests(unittest.TestCase):
    def test_selftest_subcommand_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifact"
            build_artifact(root)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = server.main(["--artifact", str(root), "selftest"])
            self.assertEqual(code, 0, msg=buffer.getvalue())
            result = json.loads(buffer.getvalue())
            self.assertTrue(result["ok"])
            self.assertEqual(result["failures"], [])
            self.assertEqual(result["units"], 4)
            self.assertEqual(result["blind_units"], 1)

    def test_status_subcommand_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifact"
            build_artifact(root)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = server.main(["--artifact", str(root), "status"])
            self.assertEqual(code, 0, msg=buffer.getvalue())
            payload = json.loads(buffer.getvalue())
            self.assertEqual(payload["progress"]["total"], 4)


if __name__ == "__main__":
    unittest.main()
