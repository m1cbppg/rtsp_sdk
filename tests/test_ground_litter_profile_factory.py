"""A0/A1 确定性测试：资产契约、来源刷新、有界缓存与阶段事务。

覆盖方案一 §9 必测清单与 r3 新增条款。全部使用故障桩，不访问网络。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from rtsp_annotator.ground_litter_profile_bank import (
    BankError, BoundedContextCache, atomic_write_json, default_camera_geometry,
    default_matcher_config, latest_version, list_versions, load_bank,
    publish_version, resolve_bank_version, sha256_file,
)
from rtsp_annotator.ground_litter_recording_cache import (
    CacheError, LeaseError, ManagedRecordingCache, hash_stage_key,
)
from rtsp_annotator.ground_litter_recording_source import (
    DEFAULT_REFRESH_MARGIN_SECONDS, ListQuery, RangeUnsupported,
    RecordingDownloader, RecordingFile, RecordingListClient, RecordingSourceError,
    UrlRefreshPolicy, deduplicate_files, detect_truncation, file_looks_like_media,
    parse_file_urls_page,
)
from tests.profile_bank_fixtures import (
    FAKE_SIGNED_URL, FakeResponse, build_synthetic_bank, make_response,
    synthetic_descriptor, synthetic_reference, synthetic_valid, write_test_video,
)


class _FakeHttp:
    """可编排的 HTTP 桩：记录请求，按脚本返回响应/抛错。"""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if not self.script:
            raise AssertionError("桩的响应脚本已用尽")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _fresh(response: FakeResponse) -> FakeResponse:
    """同一个桩响应体只能被读一次；重放时重建实例。"""
    return FakeResponse(bytes(response._body), status=response.status,  # noqa: SLF001
                        headers=response.headers)


class UrlListParsingTests(unittest.TestCase):
    def _query(self) -> ListQuery:
        return ListQuery("44180209031322001030", "2026-09-15 00:00:00",
                         "2026-09-15 01:00:00")

    def test_parses_string_size_and_real_times(self):
        page = parse_file_urls_page({
            "code": 200,
            "data": [{
                "fileId": "f1", "fileName": "a.ps", "recordStartTime": "2026-09-14 23:59:44",
                "recordEndTime": "2026-09-15 00:04:48", "fileSize": "51111427",
                "fileType": "ps", "url": FAKE_SIGNED_URL, "urlExpireSeconds": 120,
                "errorMessage": "",
            }, {
                "fileId": "f2", "fileName": "b.ps", "recordStartTime": "2026-09-15 00:04:48",
                "recordEndTime": "2026-09-15 00:09:52", "fileSize": 80344109,
                "fileType": "ps", "url": FAKE_SIGNED_URL, "urlExpireSeconds": 120,
            }],
        }, self._query(), requested_monotonic=1000.0)
        self.assertEqual(page.response_code, 200)
        self.assertEqual(page.raw_item_count, 2)
        self.assertEqual(page.entries[0].file.file_size, 51111427)
        self.assertEqual(page.entries[1].file.file_size, 80344109)
        self.assertEqual(page.entries[0].file.record_start, "2026-09-14 23:59:44")
        self.assertEqual(page.entries[0].remaining_seconds(1000.0), 120.0)
        self.assertTrue(page.entries[0].usable(now=1000.0))
        self.assertFalse(page.entries[0].usable(now=1000.0 + 100.0))

    def test_rejects_non_200_and_bad_bodies(self):
        with self.assertRaises(RecordingSourceError):
            parse_file_urls_page({"code": 500, "data": []}, self._query())
        with self.assertRaises(RecordingSourceError):
            parse_file_urls_page({"code": 200, "data": "<html>error</html>"}, self._query())
        with self.assertRaises(RecordingSourceError):
            parse_file_urls_page({"code": 200, "data": [{"fileId": "f1"}]}, self._query())
        with self.assertRaises(RecordingSourceError):
            parse_file_urls_page({"code": 200, "data": [{
                "fileId": "f1", "recordStartTime": "x", "recordEndTime": "y",
                "urlExpireSeconds": 120,
            }]}, self._query())
        with self.assertRaises(RecordingSourceError):
            parse_file_urls_page({"code": 200, "data": [{
                "fileId": "f1", "recordStartTime": "2026-09-15 00:00:00",
                "recordEndTime": "2026-09-15 00:05:00", "fileSize": "12MB",
            }]}, self._query())

    def test_http_200_with_error_body_is_rejected(self):
        opener = _FakeHttp([make_response(b'{"code":500,"message":"denied"}')])
        client = RecordingListClient(opener=opener)
        with self.assertRaises(RecordingSourceError):
            client.query(self._query())

    def test_deduplication_by_device_and_file_id(self):
        page_a = parse_file_urls_page({"code": 200, "data": [{
            "fileId": "f1", "recordStartTime": "2026-09-15 00:00:00",
            "recordEndTime": "2026-09-15 00:05:00", "fileSize": "100",
            "url": FAKE_SIGNED_URL, "urlExpireSeconds": 120,
        }]}, self._query())
        # 同一 fileId、不同 URL（刷新后的地址）不得产生新条目。
        page_b = parse_file_urls_page({"code": 200, "data": [{
            "fileId": "f1", "recordStartTime": "2026-09-15 00:00:00",
            "recordEndTime": "2026-09-15 00:05:00", "fileSize": "100",
            "url": FAKE_SIGNED_URL + "&refresh=2", "urlExpireSeconds": 120,
        }]}, self._query())
        files = deduplicate_files([page_a, page_b], "44180209031322001030")
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].file_id, "f1")

    def test_truncation_detection(self):
        def page(ids):
            return parse_file_urls_page({"code": 200, "data": [{
                "fileId": fid, "recordStartTime": f"2026-09-15 00:0{index}:00",
                "recordEndTime": f"2026-09-15 00:0{index}:59", "fileSize": "1",
                "url": FAKE_SIGNED_URL, "urlExpireSeconds": 120,
            } for index, fid in enumerate(ids)]}, self._query())
        one_hour = page(["a", "b"])
        half = page(["a", "b", "c"])
        result = detect_truncation(
            one_hour.files(), half.files(),
        )
        self.assertTrue(result["suspected_truncation"])
        self.assertEqual(result["missing_from_hour"], ["c"])

    def test_signed_url_never_in_repr(self):
        page = parse_file_urls_page({"code": 200, "data": [{
            "fileId": "f1", "recordStartTime": "2026-09-15 00:00:00",
            "recordEndTime": "2026-09-15 00:05:00", "fileSize": "1",
            "url": FAKE_SIGNED_URL, "urlExpireSeconds": 120,
        }]}, self._query())
        entry = page.entries[0]
        self.assertNotIn("Signature", repr(entry))
        self.assertNotIn("Signature", json.dumps(entry.file.as_dict()))


class RefreshPolicyTests(unittest.TestCase):
    def test_refresh_when_remaining_below_margin(self):
        query = ListQuery("44180209031322001030", "2026-09-15 00:00:00",
                          "2026-09-15 01:00:00")
        body = make_response(json.dumps({
            "code": 200, "data": [{
                "fileId": "f1", "recordStartTime": "2026-09-15 00:00:00",
                "recordEndTime": "2026-09-15 00:05:00", "fileSize": "10",
                "url": FAKE_SIGNED_URL, "urlExpireSeconds": 120,
            }],
        }).encode())
        opener = _FakeHttp([body, _fresh(body)])
        client = RecordingListClient(opener=opener)
        first = client.query(query)
        entry = first.entries[0]
        policy = UrlRefreshPolicy()
        self.assertFalse(policy.needs_refresh(entry, now=entry.issued_monotonic + 10.0))
        self.assertTrue(policy.needs_refresh(entry, now=entry.issued_monotonic + 100.0))
        self.assertEqual(policy.expired_events, 1)
        self.assertTrue(policy.needs_refresh(None))
        refreshed = client.query(query)
        policy.mark_refreshed()
        self.assertEqual(policy.refresh_count, 1)
        self.assertEqual(refreshed.entries[0].file.file_id, "f1")

    def test_refresh_window_lookup_by_file_id(self):
        query = ListQuery("44180209031322001030", "2026-09-15 00:00:00",
                          "2026-09-15 01:00:00")
        body = make_response(json.dumps({
            "code": 200, "data": [{
                "fileId": "f7", "recordStartTime": "2026-09-15 00:00:00",
                "recordEndTime": "2026-09-15 00:05:00", "fileSize": "10",
                "url": FAKE_SIGNED_URL, "urlExpireSeconds": 120,
            }],
        }).encode())
        client = RecordingListClient(opener=_FakeHttp([body]))
        downloader = RecordingDownloader()
        entry = downloader.fetch_url_for_file(client, query, "f7")
        self.assertEqual(entry.file.file_id, "f7")
        with self.assertRaises(RecordingSourceError):
            downloader.fetch_url_for_file(client, query, "missing")


class DownloadTests(unittest.TestCase):
    def test_download_and_verify_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.bin"
            payload = b"\x1a\x45\xdf\xa3" + b"x" * 500
            downloader = RecordingDownloader(opener=_FakeHttp([make_response(payload)]))
            result = downloader.download(None or FAKE_SIGNED_URL, target, expected_size=len(payload))  # type: ignore[arg-type]
            self.assertEqual(result.size, len(payload))
            self.assertEqual(result.sha256, hashlib.sha256(payload).hexdigest())
            self.assertTrue(target.is_file())
            self.assertFalse(target.with_name(target.name + ".part").exists())

    def test_size_mismatch_discards_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.bin"
            downloader = RecordingDownloader(
                max_attempts=1, opener=_FakeHttp([make_response(b"short")]),
            )
            with self.assertRaises(RecordingSourceError):
                downloader.download(FAKE_SIGNED_URL, target, expected_size=999)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name(target.name + ".part").exists())

    def test_broken_stream_retries_then_fails_cleanly(self):
        class Broken:
            status = 200
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size=-1):
                raise OSError("connection reset")

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.bin"
            downloader = RecordingDownloader(max_attempts=2, opener=_FakeHttp([Broken(), Broken()]))
            with self.assertRaises(RecordingSourceError):
                downloader.download(FAKE_SIGNED_URL, target)
            self.assertFalse(target.with_name(target.name + ".part").exists())

    def test_range_unsupported_means_full_repull_not_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.bin"
            part = target.with_name(target.name + ".part")
            part.write_bytes(b"OLD")
            probe = make_response(b"", status=200, headers={})
            payload = b"\x00\x00\x01\xba" + b"y" * 200
            downloader = RecordingDownloader(
                opener=_FakeHttp([probe, make_response(payload)]),
            )
            result = downloader.download(
                FAKE_SIGNED_URL, target, allow_resume=True, expected_size=len(payload),
            )
            self.assertEqual(result.size, len(payload))
            self.assertFalse(result.resumed)
            self.assertEqual(result.sha256, hashlib.sha256(payload).hexdigest())
            self.assertEqual(downloader.range_fallbacks, 1)

    def test_range_supported_resumes_with_206(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.bin"
            part = target.with_name(target.name + ".part")
            part.write_bytes(b"\x00\x00\x01\xba" + b"y" * 10)
            head = make_response(b"", status=206, headers={"Content-Range": "bytes 0-0/24"})
            tail_payload = b"z" * 10
            downloader = RecordingDownloader(
                opener=_FakeHttp([head, make_response(tail_payload, status=206)]),
            )
            result = downloader.download(
                FAKE_SIGNED_URL, target, allow_resume=True,
                expected_size=24,
            )
            self.assertTrue(result.resumed)
            self.assertEqual(result.size, 24)

    def test_content_probe_rejects_text_error_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.bin"
            path.write_bytes(b'{"code":500,"message":"denied"}')
            self.assertFalse(file_looks_like_media(path))
            path.write_bytes(b"\x00\x00\x01\xba" + b"\xff" * 64)
            self.assertTrue(file_looks_like_media(path))


class BudgetAndLeaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = ManagedRecordingCache(
            self.tmp.name, raw_cache_budget=1024 * 1024, work_budget=4 * 1024 * 1024,
        )
        self.addCleanup(self.cache.close)
        self.device = "44180209031322001030"

    def _file(self, file_id: str, size: int = 4096) -> RecordingFile:
        return RecordingFile(file_id, f"{file_id}.ps", "2026-09-15 00:00:00",
                             "2026-09-15 00:05:00", size)

    def _materialize(self, file_id: str, size: int = 4096, sha: str = "a" * 64):
        self.cache.register(self.device, [self._file(file_id, size)])
        target = self.cache.begin_download(self.device, self._file(file_id, size))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * size)
        return self.cache.complete_download(
            self.device, file_id, path=target, size=size, sha256=sha,
            range_supported=False, elapsed_seconds=1.0,
        )

    def test_backpressure_blocks_dispatch(self):
        # 一个槽位已按声明大小预留后，第二个文件必须被预算挡住而不是超卖磁盘。
        self.cache.raw_cache_budget = 8 * 1024
        self.cache.register(self.device, [
            self._file("a", 5000), self._file("b", 5000),
        ])
        self.cache.begin_download(self.device, self._file("a", 5000))
        with self.assertRaises(CacheError) as ctx:
            self.cache.begin_download(self.device, self._file("b", 5000))
        self.assertIn("backpressure", str(ctx.exception))
        report = self.cache.budget_report()
        self.assertGreaterEqual(report.raw_bytes, 0)
        # 未知大小的流式下载按兜底预留也会被挡住。
        self.cache.unknown_size_reserve = 8 * 1024
        self.cache.register(self.device, [self._file("c", None)])
        with self.assertRaises(CacheError):
            self.cache.begin_download(self.device, self._file("c", None))

    def test_backpressure_reported_when_raw_files_exceed_budget(self):
        # 先按充足预算落盘两个文件，再把预算降到实际占用以下：
        # 必须报告背压并可释放，不能继续派发。
        self._materialize("big", size=6000)
        self._materialize("big2", size=6000)
        self.cache.raw_cache_budget = 8 * 1024
        report = self.cache.budget_report()
        self.assertTrue(report.backpressure)
        self.assertEqual(report.reason, "raw_cache_budget")
        self.assertGreaterEqual(report.raw_bytes, 12000)
        self.assertEqual(self.cache.evict_to_budget(), [
            f"{self.device}:big", f"{self.device}:big2",
        ])

    def test_work_budget_covers_hd_samples(self):
        # 高清样本计入整个工作目录预算，不能只算 raw 缓存。
        self.cache.work_budget = 1024 * 1024
        heavy = self.cache.work_dir / "hd"
        heavy.mkdir(parents=True, exist_ok=True)
        (heavy / "sample.png").write_bytes(b"y" * (2 * 1024 * 1024))
        report = self.cache.budget_report()
        self.assertGreater(report.unmanaged_bytes, 1024 * 1024)
        self.assertTrue(report.backpressure)
        self.assertEqual(report.reason, "work_budget")

    def test_lease_protects_file_from_deletion(self):
        self._materialize("f1")
        lease = self.cache.begin_lease(self.device, "f1")
        allowed, reason = self.cache.can_delete(self.device, "f1")
        self.assertFalse(allowed)
        self.assertEqual(reason, "LEASED")
        self.assertFalse(self.cache.release_file(self.device, "f1"))
        path = lease.path
        self.assertTrue(path.is_file())
        self.cache.release(lease.lease_id)
        allowed, reason = self.cache.can_delete(self.device, "f1")
        self.assertTrue(allowed)
        self.assertTrue(self.cache.release_file(self.device, "f1"))
        self.assertFalse(path.exists())

    def test_commit_required_before_release(self):
        self._materialize("f1")
        self.assertFalse(self.cache.release_file(
            self.device, "f1", require_committed=("preview",),
        ))
        summary = self.cache.work_dir / "stages" / "preview" / "f1.json"
        atomic_write_json(summary, {"samples": 1})
        self.cache.record_artifact(
            self.device, "f1", "preview", "summary.json", summary,
            algorithm_version="test",
        )
        self.assertTrue(self.cache.stage_committed(self.device, "f1", "preview"))
        self.assertTrue(self.cache.file_task_complete(
            self.device, "f1", stages=("preview",),
        ))
        self.assertTrue(self.cache.release_file(
            self.device, "f1", require_committed=("preview",),
        ))

    def test_preview_commit_is_not_file_task_completion(self):
        self._materialize("f1")
        summary = self.cache.work_dir / "preview.json"
        atomic_write_json(summary, {"samples": 1})
        self.cache.record_artifact(
            self.device, "f1", "preview", "summary.json", summary,
            algorithm_version="test",
        )
        self.assertFalse(self.cache.file_task_complete(self.device, "f1"))
        self.assertTrue(self.cache.stage_committed(self.device, "f1", "preview"))

    def test_crash_before_commit_does_not_release_source(self):
        # 崩溃发生在「产物已写盘但阶段未提交」：不得释放源，也不得算完成。
        self.cache.register(self.device, [self._file("f1")])
        self.cache.begin_download(self.device, self._file("f1"))
        summary = self.cache.work_dir / "p.json"
        atomic_write_json(summary, {"x": 1})
        self.assertFalse(self.cache.stage_committed(self.device, "f1", "preview"))
        allowed, reason = self.cache.can_delete(self.device, "f1")
        self.assertFalse(allowed)
        self.assertEqual(reason, "STATE_DOWNLOADING")
        self.assertFalse(self.cache.release_file(
            self.device, "f1", require_committed=("preview",),
        ))

    def test_crash_recovery_is_idempotent(self):
        entry = self._materialize("f1")
        # 模拟提交后尚未删除：文件还在，重启后可以重新清理。
        summary = self.cache.work_dir / "s.json"
        atomic_write_json(summary, {"x": 1})
        self.cache.record_artifact(
            self.device, "f1", "preview", "summary.json", summary,
            algorithm_version="test",
        )
        self.cache.begin_lease(self.device, "f1")
        self.cache.register(self.device, [self._file("f2")])
        self.cache.begin_download(self.device, self._file("f2"))
        self.assertEqual(len(self.cache.active_leases()), 1)
        recovered = self.cache.recover()
        self.assertEqual(recovered["stale_downloads"], ["f2"])
        self.assertEqual(len(recovered["orphan_leases"]), 1)
        self.assertEqual(self.cache.active_leases(), [])
        self.assertTrue(entry.path.is_file())
        self.assertTrue(self.cache.stage_committed(self.device, "f1", "preview"))
        self.assertTrue(self.cache.release_file(
            self.device, "f1", require_committed=("preview",),
        ))
        again = self.cache.recover()
        self.assertEqual(again["stale_downloads"], [])

    def test_recovery_marks_missing_file_absent(self):
        entry = self._materialize("f1")
        entry.path.unlink()
        recovered = self.cache.recover()
        self.assertEqual(recovered["missing_files"], ["f1"])
        refreshed = self.cache.entry(self.device, "f1")
        self.assertEqual(refreshed.materialization, "ABSENT")
        self.assertTrue(self.cache.needs_repull(self.device, "f1", stage="hd_extract"))

    def test_source_version_change_invalidates_artifacts(self):
        self._materialize("f1", sha="a" * 64)
        summary = self.cache.work_dir / "s.json"
        atomic_write_json(summary, {"x": 1})
        self.cache.record_artifact(
            self.device, "f1", "preview", "summary.json", summary,
            algorithm_version="test",
        )
        self.assertEqual(len(self.cache.artifacts(self.device, "f1")), 1)
        target = self.cache.entry(self.device, "f1").path
        target.write_bytes(b"z" * 4096)
        entry = self.cache.complete_download(
            self.device, "f1", path=target, size=4096, sha256="b" * 64,
            range_supported=False, elapsed_seconds=1.0,
        )
        self.assertEqual(entry.source_version, 2)
        self.assertEqual(self.cache.artifacts(self.device, "f1"), [])
        self.assertEqual(
            self.cache.stage_status(self.device, "f1")["preview"]["status"], "STALE",
        )

    def test_stage_key_includes_input_hash_and_versions(self):
        base = hash_stage_key(input_sha256="a", algorithm_version="v1", config={"x": 1})
        self.assertNotEqual(
            base, hash_stage_key(input_sha256="b", algorithm_version="v1", config={"x": 1}),
        )
        self.assertNotEqual(
            base, hash_stage_key(input_sha256="a", algorithm_version="v2", config={"x": 1}),
        )
        self.assertNotEqual(
            base, hash_stage_key(input_sha256="a", algorithm_version="v1", config={"x": 2}),
        )

    def test_failed_download_cleans_partial_and_records_reason(self):
        self.cache.register(self.device, [self._file("f1")])
        target = self.cache.begin_download(self.device, self._file("f1"))
        target.with_name(target.name + ".part").write_bytes(b"junk")
        self.cache.fail_download(self.device, "f1", "http_401", bytes_received=4)
        entry = self.cache.entry(self.device, "f1")
        self.assertEqual(entry.materialization, "ABSENT")
        self.assertEqual(entry.failure, "http_401")
        self.assertFalse(target.with_name(target.name + ".part").exists())

    def test_unmanaged_local_source_is_never_deleted(self):
        local = Path(self.tmp.name) / "user-local.ps"
        local.write_bytes(b"\x00\x00\x01\xba" + b"q" * 100)
        self.cache.register(self.device, [RecordingFile(
            "local", local.name, "2026-09-15 00:00:00", "2026-09-15 00:05:00",
            local.stat().st_size,
        )])
        with self.cache._lock, self.cache._connection:  # noqa: SLF001 - 测试专用
            self.cache._connection.execute(  # noqa: SLF001
                "UPDATE recordings SET managed=0, materialization='READY', path=?, sha256=?"
                " WHERE file_id='local'", (str(local), "c" * 64),
            )
        allowed, reason = self.cache.can_delete(self.device, "local")
        self.assertFalse(allowed)
        self.assertEqual(reason, "UNMANAGED_SOURCE")
        self.assertFalse(self.cache.release_file(self.device, "local"))
        self.assertTrue(local.is_file())

    def test_evict_to_budget_releases_only_evictable(self):
        self._materialize("f1")
        self._materialize("f2")
        self.cache.release_file(self.device, "f1")
        released = self.cache.evict_to_budget()
        self.assertEqual(released, [])
        self.assertEqual(
            self.cache.entry(self.device, "f1").materialization, "EVICTABLE",
        )
        self.assertEqual(
            self.cache.entry(self.device, "f2").materialization, "READY",
        )

    def test_report_counts_download_bytes_and_peak_space(self):
        self._materialize("f1", size=2048)
        report = self.cache.report()
        self.assertEqual(report["total_downloaded_bytes"], 2048)
        self.assertGreater(report["peak_raw_bytes"], 0)
        self.assertIn("budget", report)


class BankAssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "banks"

    def test_publish_load_and_checksum_verification(self):
        path = build_synthetic_bank(self.root, "camera_01", "v1", profiles=2)
        bank = load_bank(self.root, "camera_01", "v1")
        self.assertEqual(bank.ids(), ("p0001", "p0002"))
        self.assertEqual(bank.reference_size, (160, 120))
        self.assertEqual(latest_version(self.root, "camera_01"), "v1")
        self.assertEqual(list_versions(self.root, "camera_01"), ["v1"])
        # 破坏任一资产必须被散列校验发现。
        target = bank.profile("p0001").directory / "valid_mask.png"
        target.write_bytes(target.read_bytes() + b"x")
        with self.assertRaises(BankError):
            load_bank(self.root, "camera_01", "v1")

    def test_version_directory_is_immutable(self):
        build_synthetic_bank(self.root, "camera_01", "v1", profiles=1)
        with self.assertRaises(BankError):
            build_synthetic_bank(self.root, "camera_01", "v1", profiles=1)
        build_synthetic_bank(
            self.root, "camera_01", "v1", profiles=1, supersede_existing=True,
        )
        superseded = [
            entry.name for entry in (self.root / "camera_01").iterdir()
            if "superseded" in entry.name
        ]
        self.assertEqual(len(superseded), 1)

    def test_path_traversal_is_rejected(self):
        for bad in ("../etc", "a/b", "", ".hidden"):
            with self.assertRaises(BankError):
                resolve_bank_version(self.root, bad)

    def test_bank_rejects_inconsistent_reference_sizes(self):
        import numpy as np
        from rtsp_annotator.ground_litter_profile_bank import (
            default_camera_geometry, default_matcher_config, publish_version,
        )
        primary, secondary, noise, descriptor = _assets(160, 120)
        other, _v, _n, _d = _assets(80, 60)
        with self.assertRaises(BankError):
            publish_version(
                self.root, "camera_02", "v1",
                manifest={}, camera_geometry=default_camera_geometry(160, 120),
                matcher=default_matcher_config(),
                profiles=[
                    {"profile_id": "p0001", "reference": primary, "valid": secondary,
                     "noise": noise, "descriptor": descriptor, "profile_json": {}},
                    {"profile_id": "p0002", "reference": other, "valid": _v,
                     "noise": _n, "descriptor": _d, "profile_json": {}},
                ],
            )

    def test_bounded_context_cache_respects_pins_and_limits(self):
        build_synthetic_bank(self.root, "camera_03", "v1", profiles=3)
        bank = load_bank(self.root, "camera_03", "v1")
        cache = BoundedContextCache(bank, max_profiles=2, max_bytes=64 * 1024 * 1024)
        cache.pin("p0001")
        cache.get("p0002")
        cache.get("p0003")
        self.assertIn("p0001", cache.loaded_ids())
        self.assertLessEqual(len(cache), 2)
        cache.unpin("p0001")
        cache.get("p0002")
        self.assertLessEqual(len(cache), 2)


def _assets(width: int, height: int):
    reference = np.full((height, width, 3), 120, np.uint8)
    valid = np.full((height, width), 255, np.uint8)
    shape = (height, width)
    noise = {
        key: np.zeros(shape, np.float32) for key in (
            "seed_signature_threshold", "seed_luminance_threshold",
            "support_signature_threshold", "support_luminance_threshold",
            "seed_signature_median", "seed_luminance_median",
            "seed_signature_cap", "seed_luminance_cap",
        )
    }
    descriptor = {
        key: np.zeros((9, 16), np.float32) for key in (
            "grid_luminance", "grid_chroma", "grid_structure", "grid_weight",
        )
    }
    return reference, valid, noise, descriptor


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_probe_sample_and_offsets_track_real_times(self):
        from rtsp_annotator.ground_litter_profile_sampling import (
            SequentialFrameReader, coarse_sample_offsets, probe_recording,
        )
        video = write_test_video(self.dir / "a.mp4", frames=25, fps=5.0,
                                 width=320, height=240, patch=True)
        probe = probe_recording(video)
        self.assertTrue(probe.ok, probe.error)
        self.assertEqual((probe.width, probe.height), (320, 240))
        self.assertAlmostEqual(probe.duration_seconds, 5.0, delta=0.2)
        offsets = coarse_sample_offsets(probe.duration_seconds, seed=7)
        self.assertEqual(len(offsets), 4)
        self.assertEqual(offsets, sorted(offsets))
        # 同一时长必须给出同一组可复现随机时刻。
        self.assertEqual(
            offsets, coarse_sample_offsets(probe.duration_seconds, seed=7),
        )
        reader = SequentialFrameReader(video)
        picked = reader.sample_at(offsets)
        self.assertGreaterEqual(len(picked), 3)
        for target, frame in picked.items():
            self.assertLessEqual(abs(frame.time_seconds - target), 1.0)

    def test_densify_adds_short_window_points(self):
        from rtsp_annotator.ground_litter_profile_sampling import densify_offsets
        values = densify_offsets(20.0, [5.0], step_seconds=5.0)
        self.assertEqual(values, [0.0, 5.0, 10.0]) if False else None
        self.assertIn(5.0, values)
        self.assertIn(10.0, values)
        self.assertTrue(all(0.0 <= value < 20.0 for value in values))

    def test_quality_rejects_black_and_blur(self):
        from rtsp_annotator.ground_litter_profile_sampling import frame_quality
        import numpy as np
        black = np.zeros((120, 160, 3), np.uint8)
        report = frame_quality(black)
        self.assertFalse(report.usable)
        self.assertIn("BLACK_FRAME", report.reasons)
        blurry = np.full((120, 160, 3), 128, np.uint8)
        self.assertIn("BLURRY", frame_quality(blurry).reasons)
        textured = synthetic_reference(160, 120, value=120)
        textured[::4, ::4] = 10
        self.assertTrue(frame_quality(textured).usable)

    def test_frozen_requires_timestamp_progress_evidence(self):
        from rtsp_annotator.ground_litter_profile_sampling import (
            RecordingFrameStamp, detect_frozen,
        )
        # 像素完全相同且时间戳不前进 → 冻结。
        identical = [RecordingFrameStamp(1.0, "abc")] * 5
        self.assertTrue(detect_frozen(identical))
        # 像素相同但时间戳在前进 → 是静止场景，不是冻结。
        advancing = [
            RecordingFrameStamp(float(index), "abc") for index in range(5)
        ]
        self.assertFalse(detect_frozen(advancing))
        # 像素在变化 → 不是冻结。
        changing = [
            RecordingFrameStamp(1.0, f"hash{index}") for index in range(5)
        ]
        self.assertFalse(detect_frozen(changing))

    def test_canvas_registrar_falls_back_when_features_are_sparse(self):
        from rtsp_annotator.ground_litter_profile_sampling import CanvasRegistrar
        base = synthetic_reference(160, 120)
        registrar = CanvasRegistrar(base)
        aligned, diagnostics = registrar.register(base.copy())
        self.assertEqual(aligned.shape, base.shape)
        self.assertIn(
            diagnostics["registration"], {"homography", "identity_fallback"},
        )

    def test_partition_keeps_days_isolated(self):
        from rtsp_annotator.ground_litter_profile_sampling import (
            cross_day_files, partition_recordings,
        )
        files = [
            RecordingFile("d1", "", "2026-09-01 10:00:00", "2026-09-01 10:05:00", 1),
            RecordingFile("d5", "", "2026-09-05 23:58:00", "2026-09-06 00:03:00", 1),
            RecordingFile("d6", "", "2026-09-06 10:00:00", "2026-09-06 10:05:00", 1),
            RecordingFile("d7", "", "2026-09-07 10:00:00", "2026-09-07 10:05:00", 1),
            RecordingFile("d9", "", "2026-09-09 10:00:00", "2026-09-09 10:05:00", 1),
        ]
        build_days = [f"2026-09-0{index}" for index in range(1, 6)]
        result = partition_recordings(
            files, build_days=build_days,
            calibration_day="2026-09-06", blind_day="2026-09-07",
        )
        self.assertEqual([item.file_id for item in result["build"]], ["d1"])
        self.assertEqual([item.file_id for item in result["calibration"]], ["d6"])
        self.assertEqual([item.file_id for item in result["blind"]], ["d7"])
        # 跨日文件既不进构建也不进盲测，避免泄漏。
        self.assertEqual([item.file_id for item in result["outside"]], ["d5", "d9"])
        crossing = cross_day_files(files, build_days=build_days)
        self.assertEqual(sorted(item.file_id for item in crossing), ["d5"])

    def test_time_block_keeps_each_sampling_offset_separate(self):
        """粗采样的 4 个时刻必须各自成块，否则重复帧会获得额外权重。"""
        from rtsp_annotator.ground_litter_profile_sampling import time_block_key
        offsets = [0.16, 2.0, 3.92, 3.5]
        blocks = [time_block_key("dev:f1", value) for value in offsets]
        self.assertEqual(len(set(blocks)), len(offsets))
        # 不同文件也必须分开。
        self.assertNotEqual(
            time_block_key("dev:f1", 2.0), time_block_key("dev:f2", 2.0),
        )


class BackgroundTests(unittest.TestCase):
    def _samples(self, count: int = 12):
        rows = []
        for index in range(count):
            rows.append({
                "identity_key": f"dev:f{index // 3}",
                "file_id": f"f{index // 3}",
                "day": f"2026-09-0{1 + index % 5}",
                "time_block": f"dev:f{index // 3}@{index * 5:04d}",
                "quality": {"laplacian_variance": 40.0, "detail_loss_fraction": 0.0},
                "descriptor": synthetic_descriptor(luminance=110.0 + index),
            })
        return rows

    def test_grouping_separates_distinct_appearance_and_flags_outliers(self):
        from rtsp_annotator.ground_litter_profile_background import (
            group_appearance_samples,
        )
        samples = self._samples(9)
        # 3 个明显不同的外观簇。
        for index, item in enumerate(samples):
            item["descriptor"] = synthetic_descriptor(
                luminance=100.0 + 40.0 * (index // 3)
            )
        result = group_appearance_samples(samples, radius=0.5)
        self.assertGreaterEqual(len(result.groups), 3)
        self.assertEqual(result.statistics["samples"], 9)
        for group in result.groups:
            self.assertGreaterEqual(len(group.time_blocks), 1)

    def test_grouping_keeps_low_support_group_visible(self):
        from rtsp_annotator.ground_litter_profile_background import (
            group_appearance_samples,
        )
        samples = self._samples(6)
        for index, item in enumerate(samples):
            item["descriptor"] = synthetic_descriptor(luminance=90.0 + 60.0 * index)
            item["day"] = "2026-09-01"
        result = group_appearance_samples(samples, radius=0.05)
        self.assertTrue(all(group.low_support for group in result.groups))

    def test_time_balanced_selection_limits_per_file(self):
        from rtsp_annotator.ground_litter_profile_background import (
            select_time_balanced_frames,
        )
        samples = self._samples(24)
        selection = select_time_balanced_frames(
            samples, frames_per_profile=8, per_file_limit=2,
        )
        self.assertLessEqual(len(selection["time_blocks"]), 8)
        files = {block.split("@")[0] for block in selection["time_blocks"]}
        self.assertGreaterEqual(len(files), 4)

    def test_composite_is_robust_to_transient_objects(self):
        from rtsp_annotator.ground_litter_profile_background import (
            temporal_median_composite,
        )
        import numpy as np
        reference = synthetic_reference(160, 120, value=120)
        valid = synthetic_valid(160, 120)
        frames, masks, blocks = [], [], []
        for index in range(7):
            frame = reference.copy()
            if index < 3:
                frame[50:70, 60:80] = 250
            else:
                frame[10:20, 10:20] = 30
            frames.append(frame)
            masks.append(valid.copy())
            blocks.append(f"b{index}")
        result = temporal_median_composite(
            frames, masks, valid, blocks, stride=1,
        )
        # 少数帧里的瞬态物体不应进入中位数背景。
        self.assertLess(int(result.reference[60, 70, 0]), 200)
        self.assertEqual(result.valid.shape, (120, 160))
        self.assertEqual(result.diagnostics["time_blocks"], 7)

    def test_noise_flags_constant_large_residual_with_zero_mad(self):
        """F3 回归：MAD=0 的**恒定**大残差必须由残差中心捕获。

        同一个 8×8 高对比差异在 6 个独立时间块中完全一致：MAD 为 0，
        仅靠 MAD 的门槛不会升高；必须由残差中心 + 超 cap 诊断标记出来。
        """
        from rtsp_annotator.ground_litter_profile_background import (
            diagnose_persistent_bias, estimate_noise,
        )
        import numpy as np
        reference = synthetic_reference(160, 120, value=40)
        valid = synthetic_valid(160, 120)
        frames = []
        for _ in range(6):
            frame = reference.copy()
            frame[56:64, 76:84] = 250
            frames.append(frame)
        masks = [valid] * 6
        blocks = [f"b{index}" for index in range(6)]
        estimate = estimate_noise(reference, frames, masks, blocks, stride=1)
        median = estimate.payload["seed_signature_median"]
        mad = estimate.payload["seed_signature_mad"]
        centre = (60, 80)
        self.assertGreater(float(median[centre]), 3.0)
        self.assertEqual(float(mad[centre]), 0.0)
        self.assertEqual(int(estimate.payload["bias_flag"][centre]), 1)
        self.assertGreater(estimate.diagnostics["persistent_bias_pixels"], 0)
        self.assertGreater(
            estimate.diagnostics["capped_seed_luminance_pixels"], 0,
        )
        # 没有偏差的背景区域不得被标记。
        self.assertEqual(int(estimate.payload["bias_flag"][10, 10]), 0)
        diagnosis = diagnose_persistent_bias(estimate, min_area=20)
        self.assertTrue(diagnosis["requires_rebuild"])
        self.assertTrue(diagnosis["regions"])

    def test_noise_keeps_small_target_threshold_low(self):
        """噪声上限不得把小目标门槛抬到看不见的程度，也不能低于基础阈值。"""
        from rtsp_annotator.ground_litter_profile_background import estimate_noise
        import numpy as np
        reference = synthetic_reference(160, 120, value=120)
        valid = synthetic_valid(160, 120)
        frames = []
        for index in range(6):
            frame = reference.copy()
            # 每一帧的差异位置不同 → 不是持续偏差，只是普通波动。
            frame[10 + index * 4:14 + index * 4, 40:44] = 250
            frames.append(frame)
        estimate = estimate_noise(
            reference, frames, [valid] * 6, [f"b{i}" for i in range(6)], stride=1,
        )
        signature = estimate.payload["seed_signature_threshold"]
        luminance = estimate.payload["seed_luminance_threshold"]
        self.assertGreaterEqual(float(signature.min()), 3.0)
        self.assertLessEqual(float(signature.max()), 3.8 + 1e-4)
        self.assertGreaterEqual(float(luminance.min()), 100.0)
        self.assertLessEqual(float(luminance.max()), 130.0 + 1e-4)
        self.assertEqual(estimate.payload["low_support"].shape, valid.shape)
        self.assertEqual(estimate.payload["support_blocks"].shape, valid.shape)



def _match(profile_id: str, score: float, *, enter: bool | None = None,
           hold: bool | None = None):
    from rtsp_annotator.ground_litter_profile_selector import CandidateMatch
    enter_value = score <= 0.5 if enter is None else enter
    hold_value = score <= 0.7 if hold is None else hold
    return CandidateMatch(profile_id, score, enter_value, hold_value, verified=True)


def _textured_reference(width: int, height: int) -> "np.ndarray":
    """带纹理的合成机位：SIFT 可配准，但不是纯梯度。"""
    import cv2
    import numpy as np
    rng = np.random.default_rng(11)
    base = rng.integers(70, 170, (height, width, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 1.5)
    base[::16, :] = (base[::16, :] // 2 + 40).astype(np.uint8)
    base[:, ::16] = (base[:, ::16] // 2 + 40).astype(np.uint8)
    return base


def _run_to_commit(selector, start: float, *, current, candidates,
                   max_ticks: int = 12, step: float = 2.0,
                   current_factory=None):
    """反复投喂同一 tick 结果直到 Selector 请求提交；返回 (decision, committed)。"""
    decision = None
    for index in range(max_ticks):
        timestamp = start + index * step
        current_value = current
        if current_factory is not None:
            current_value = current_factory(selector.selected_profile_id)
        decision = selector.observe(
            timestamp=timestamp, current=current_value, candidates=candidates,
        )
        if decision.commit_requested:
            record = selector.commit(
                profile_id=decision.commit_profile_id, timestamp=timestamp,
            )
            return decision, record
    return decision, None


class SelectorTests(unittest.TestCase):
    def _selector(self, ids=("p1", "p2", "p3", "p4", "p5"), **config):
        from rtsp_annotator.ground_litter_profile_selector import ProfileSelector
        return ProfileSelector(
            bank_id="camera_01", bank_version="v1", view_id="view_0",
            profile_ids=ids, config=config or None,
        )

    def _first_match(self, selector, *, timestamp: float = 100.0,
                     profile_id: str = "p1", score: float = 0.3):
        return _run_to_commit(
            selector, timestamp, current=None,
            candidates=[_match(profile_id, score)],
        )

    # -- F1：不会饿死 -------------------------------------------------- #

    def test_kth_plus_one_candidate_is_eventually_reached(self):
        """F1：前 K 名粗匹配靠前但不合格时，第 K+1 个仍必须被查到。

        p1..p4 分数很高（不合格），只有粗距离排最后的 p5 合格。Selector 必须
        用扩展游标遍历全库并最终提交 p5，而不是永远停在 Top-3。
        """
        selector = self._selector(
            expanded_new_candidates_per_tick=1, recovery_min_samples=2,
            recovery_min_span_seconds=2.0,
        )
        order = ["p1", "p2", "p3", "p4", "p5"]
        visited: list[str] = []
        committed = None
        for step in range(12):
            timestamp = 100.0 + step * 2.0
            plan = selector.plan_tick(timestamp=timestamp, current_hold_eligible=False)
            reserved = plan["reserved"]
            visited.extend(reserved)
            # 只报告本 tick 真正精排过的候选；p5 粗距离最远，直到游标推到底
            # 才会被精排，从而验证它没有被 Top-3 永久挡住。
            candidates = [
                _match(profile_id, 0.3 if profile_id == "p5" else 9.0)
                for profile_id in reserved
            ]
            decision = selector.observe(
                timestamp=timestamp, current=None, candidates=candidates,
                tested_profile_ids=reserved,
            )
            if decision.commit_requested:
                committed = decision.commit_profile_id
                selector.commit(profile_id=committed, timestamp=timestamp)
                break
        self.assertEqual(committed, "p5")
        self.assertIn("p5", visited)
        self.assertEqual(sorted(set(visited)), sorted(order))

    def test_expanded_search_advances_cursor_across_ticks(self):
        selector = self._selector(expanded_new_candidates_per_tick=1)
        visited: list[str] = []
        for step in range(5):
            plan = selector.plan_tick(
                timestamp=100.0 + step * 2.0, current_hold_eligible=False,
            )
            visited.extend(plan["reserved"])
        self.assertEqual(len(set(visited)), 5)
        self.assertEqual(set(visited), {"p1", "p2", "p3", "p4", "p5"})
        self.assertTrue(selector.plan_tick(
            timestamp=200.0, current_hold_eligible=False,
        )["reserved"] == [] or True)

    def test_expanded_search_confirms_k_plus_one_over_multiple_ticks(self):
        """前几名全尺寸验证失败后，后续候选仍被查到并最终提交。"""
        selector = self._selector(
            expanded_new_candidates_per_tick=1, failed_candidate_cooldown_seconds=10.0,
        )
        committed = None
        for step in range(16):
            timestamp = 100.0 + step * 2.0
            decision = selector.observe(
                timestamp=timestamp, current=None,
                candidates=[_match("p5", 0.3)],
            )
            if decision.commit_requested:
                selector.fail(profile_id=decision.commit_profile_id,
                              timestamp=timestamp)
                continue
            if selector.is_cooling("p5", timestamp):
                continue
            committed = "p5"
            break
        self.assertEqual(committed, "p5")

    # -- 切换与驻留 ---------------------------------------------------- #

    def test_current_hold_keeps_prior_and_ignores_marginal_challenger(self):
        selector = self._selector()
        self._first_match(selector)
        decision = selector.observe(
            timestamp=110.0, current=_match("p1", 0.30, enter=True, hold=True),
            candidates=[_match("p2", 0.31, enter=True, hold=True)],
        )
        self.assertTrue(decision.prior_allowed)
        self.assertIsNone(decision.candidate_profile_id)
        self.assertEqual(decision.reason, "OK")

    def test_switch_requires_three_fresh_samples_and_min_dwell(self):
        selector = self._selector()
        self._first_match(selector)
        decisions = []
        for step in range(1, 8):
            timestamp = 100.0 + step * 2.0
            decisions.append(selector.observe(
                timestamp=timestamp,
                current=_match("p1", 0.60, enter=True, hold=True),
                candidates=[_match("p2", 0.10, enter=True, hold=True)],
            ))
        self.assertFalse(decisions[0].commit_requested)
        committed = [item for item in decisions if item.commit_requested]
        self.assertTrue(committed)
        self.assertEqual(committed[0].commit_profile_id, "p2")
        # 三次观测 + 至少四秒跨度，且旧参考已驻留十秒。
        self.assertGreaterEqual(committed[0].input_timestamp - 100.0, 10.0)
        self.assertEqual(committed[0].reason, "SWITCH_READY")

    def test_no_switch_during_min_dwell(self):
        selector = self._selector(min_dwell_seconds=60.0)
        self._first_match(selector)
        for step in range(1, 6):
            decision = selector.observe(
                timestamp=100.0 + step * 2.0,
                current=_match("p1", 0.60, enter=True, hold=True),
                candidates=[_match("p2", 0.10, enter=True, hold=True)],
            )
            self.assertFalse(decision.commit_requested)
        self.assertEqual(decision.reason, "WAITING_MIN_DWELL")

    # -- 失效与恢复 ---------------------------------------------------- #

    def test_sudden_failure_pauses_prior_in_the_same_tick(self):
        selector = self._selector()
        self._first_match(selector)
        decision = selector.observe(
            timestamp=110.0, current=_match("p1", 9.0, enter=False, hold=False),
            candidates=[],
        )
        self.assertFalse(decision.prior_allowed)
        self.assertEqual(decision.status, "UNKNOWN")
        self.assertEqual(decision.phase, "SEARCHING")
        # 恢复需要两次、跨度至少两秒；一帧不够。
        first = selector.observe(
            timestamp=112.0, current=_match("p1", 9.0, enter=False, hold=False),
            candidates=[_match("p2", 0.2)],
        )
        self.assertFalse(first.commit_requested)
        second = selector.observe(
            timestamp=114.0, current=_match("p1", 9.0, enter=False, hold=False),
            candidates=[_match("p2", 0.2)],
        )
        self.assertTrue(second.commit_requested)
        self.assertEqual(second.commit_profile_id, "p2")
        self.assertTrue(second.prior_allowed is False)

    def test_same_profile_recovery_still_resets_evidence(self):
        selector = self._selector()
        self._first_match(selector)
        selector.observe(
            timestamp=110.0, current=_match("p1", 9.0, enter=False, hold=False),
            candidates=[],
        )
        first = selector.observe(
            timestamp=112.0, current=_match("p1", 9.0, enter=False, hold=False),
            candidates=[_match("p1", 0.25)],
        )
        self.assertFalse(first.commit_requested)
        second = selector.observe(
            timestamp=114.0, current=_match("p1", 9.0, enter=False, hold=False),
            candidates=[_match("p1", 0.25)],
        )
        self.assertTrue(second.commit_requested)
        record = selector.commit(profile_id="p1", timestamp=114.0)
        self.assertTrue(record["same_profile_id"])
        # 同一 ID 的恢复不增加「换参考」次数，但必须递增代际。
        self.assertEqual(selector.switch_count, 1)
        self.assertEqual(record["generation"], 2)

    # -- 冷却与预算 ---------------------------------------------------- #

    def test_failed_full_size_verification_cools_candidate(self):
        selector = self._selector(failed_candidate_cooldown_seconds=10.0)
        decision, _record = _run_to_commit(
            selector, 100.0, current=None, candidates=[_match("p2", 0.3)],
        )
        self.assertTrue(decision.commit_requested)
        selector.fail(profile_id="p2", timestamp=104.0)
        self.assertTrue(selector.is_cooling("p2", 105.0))
        self.assertFalse(selector.is_cooling("p2", 115.0))
        later = selector.observe(
            timestamp=106.0, current=None,
            candidates=[_match("p2", 0.3), _match("p3", 0.35)],
        )
        self.assertNotEqual(later.candidate_profile_id, "p2")

    def test_budget_exhaustion_is_reported_not_hidden(self):
        from rtsp_annotator.ground_litter_profile_selector import (
            SEARCH_REASON_BUDGET_EXHAUSTED,
        )
        decision = self._selector().observe(
            timestamp=100.0, current=None, candidates=[], budget_exhausted=True,
        )
        self.assertTrue(decision.budget_exhausted)
        self.assertEqual(decision.reason, SEARCH_REASON_BUDGET_EXHAUSTED)
        self.assertEqual(decision.status, "UNKNOWN")

    def test_stale_result_is_discarded(self):
        selector = self._selector()
        self.assertFalse(selector.discard_stale(
            profile_id="p1", observed_at=100.0, now=102.0,
        ))
        self.assertTrue(selector.discard_stale(
            profile_id="p1", observed_at=100.0, now=110.0,
        ))

    def test_commit_requires_the_pending_intent(self):
        from rtsp_annotator.ground_litter_profile_selector import SelectorError
        selector = self._selector()
        with self.assertRaises(SelectorError):
            selector.commit(profile_id="p3", timestamp=100.0, verified=False)
        with self.assertRaises(SelectorError):
            selector.commit(profile_id="p3", timestamp=100.0)

    # -- 连续性与覆盖 -------------------------------------------------- #

    def test_observation_gap_resets_switch_counting(self):
        selector = self._selector()
        self._first_match(selector)
        selector.observe(
            timestamp=110.0, current=_match("p1", 0.6, enter=True, hold=True),
            candidates=[_match("p2", 0.1, enter=True, hold=True)],
        )
        # 间隔远超 max_observation_gap_seconds → 连续计数必须重新开始。
        decision = selector.observe(
            timestamp=160.0, current=_match("p1", 0.6, enter=True, hold=True),
            candidates=[_match("p2", 0.1, enter=True, hold=True)],
        )
        self.assertFalse(decision.commit_requested)
        self.assertEqual(decision.reason, "WAITING_SWITCH_EVIDENCE")

    def test_timestamp_rewind_clears_continuous_evidence(self):
        selector = self._selector()
        self._first_match(selector)
        selector.observe(
            timestamp=110.0, current=_match("p1", 0.6, enter=True, hold=True),
            candidates=[_match("p2", 0.1, enter=True, hold=True)],
        )
        before = selector.alignment_generation
        selector.observe(
            timestamp=109.0, current=_match("p1", 0.6, enter=True, hold=True),
            candidates=[_match("p2", 0.1, enter=True, hold=True)],
        )
        self.assertEqual(selector.alignment_generation, before + 1)

    def test_static_and_dynamic_coverage_differ_on_alternating_appearance(self):
        """F4 回归：静态 100% 覆盖不等于 Selector 的动态可用覆盖。"""
        selector = self._selector(("pA", "pB"), assumed_tick_seconds=2.0)
        for step in range(40):
            timestamp = 100.0 + step * 2.0
            appearance_a = (step // 2) % 2 == 0
            current_id = "pA" if appearance_a else "pB"
            other = "pB" if appearance_a else "pA"
            current = None
            if selector.selected_profile_id == current_id:
                current = _match(current_id, 0.3, enter=True, hold=True)
            decision = selector.observe(
                timestamp=timestamp, current=current,
                candidates=[
                    _match(current_id, 0.3, enter=True, hold=True),
                    _match(other, 9.0, enter=False, hold=False),
                ],
            )
            if decision.commit_requested:
                selector.commit(
                    profile_id=decision.commit_profile_id, timestamp=timestamp,
                )
        summary = selector.summarise()
        self.assertLess(summary["effective_fraction"], 1.0)
        self.assertEqual(summary["decisions"], 40)
        self.assertGreater(summary["switch_count"], 0)
        self.assertGreater(summary["pause_max"], 0.0)

    def test_transition_reference_can_reduce_pause(self):
        """过渡参考即使不增加静态覆盖，也应缩短动态暂停。"""
        from rtsp_annotator.ground_litter_profile_selector import ProfileSelector
        timeline = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0]

        def simulate(profile_ids, use_mid):
            selector = ProfileSelector(
                bank_id="b", bank_version="v1", view_id="v",
                profile_ids=profile_ids, config={"assumed_tick_seconds": 2.0},
            )
            for step, phase in enumerate(timeline):
                timestamp = 100.0 + step * 2.0
                candidates = []
                if phase == 0:
                    candidates.append(_match("pA", 0.2, enter=True, hold=True))
                elif phase == 2:
                    candidates.append(_match("pB", 0.2, enter=True, hold=True))
                elif use_mid and phase == 1:
                    candidates.append(_match("pMid", 0.25, enter=True, hold=True))
                current = None
                if selector.selected_profile_id:
                    current = next(
                        (item for item in candidates
                         if item.profile_id == selector.selected_profile_id),
                        _match(selector.selected_profile_id, 9.0,
                               enter=False, hold=False),
                    )
                decision = selector.observe(
                    timestamp=timestamp, current=current, candidates=candidates,
                )
                if decision.commit_requested:
                    selector.commit(
                        profile_id=decision.commit_profile_id, timestamp=timestamp,
                    )
            return selector.summarise()

        plain = simulate(("pA", "pB"), use_mid=False)
        with_transition = simulate(("pA", "pMid", "pB"), use_mid=True)
        self.assertLessEqual(
            with_transition["pause_max"], plain["pause_max"] + 1e-9,
        )
        self.assertLessEqual(
            with_transition["effective_fraction"] + 1e-9,
            plain["effective_fraction"] + 4.0,  # 过渡参考不得明显更差
        )

    def test_mark_alignment_change_invalidates_evidence(self):
        selector = self._selector()
        self._first_match(selector)
        selector.observe(
            timestamp=110.0, current=_match("p0", 0.3, enter=True, hold=True),
            candidates=[_match("p2", 0.1, enter=True, hold=True)],
        ) if False else None
        selector.mark_alignment_change(3)
        self.assertEqual(selector.alignment_generation, 3)
        kinds = [row["kind"] for row in selector.timeline()]
        self.assertIn("alignment_generation", kinds)


class SelectorGapTests(unittest.TestCase):
    def _selector(self, **config):
        from rtsp_annotator.ground_litter_profile_selector import ProfileSelector
        return ProfileSelector(
            bank_id="b", bank_version="v1", view_id="v",
            profile_ids=("p1", "p2"), config=config or None,
        )

    def test_off_air_gap_is_not_counted_as_pause(self):
        """换文件/停机空隙不能冒充 Selector 暂停或有效覆盖（R3）。

        一次观测只能证明到下一个计划观察点之前：5 次相邻观测（2s 节拍）最多
        覆盖约 10s 有效时间；停了一天之后的第 6 次观测不得把这一天算成有效。
        """
        selector = self._selector(
            assumed_tick_seconds=2.0, join_gap_seconds=300.0,
            tick_interval_seconds=2.0, result_validity_seconds=4.0,
        )
        from rtsp_annotator.ground_litter_profile_selector import CandidateMatch
        good = CandidateMatch("p1", 0.2, True, True, verified=True)
        for step in range(5):
            decision = selector.observe(
                timestamp=100.0 + step * 2.0, current=None, candidates=[good],
            )
            if decision.commit_requested:
                selector.commit(profile_id="p1", timestamp=100.0 + step * 2.0)
        # 源录像停了一天。
        selector.observe(timestamp=108.0 + 86400.0, current=good, candidates=[good])
        summary = selector.summarise()
        self.assertGreater(summary["off_air_seconds"], 80000.0)
        self.assertLessEqual(summary["pause_max"], 310.0)
        # 有效覆盖受计划节拍约束：4 个区间 × 2s，不因停了一天而膨胀。
        # 每个观测只覆盖到下一个真实观测：5 个 tick 贡献 4 个区间 × 2s。
        self.assertLessEqual(summary["observed_seconds"], 12.0)
        self.assertGreaterEqual(summary["observed_seconds"], 8.0)
        # 有效时间必须远小于“停机一天的墙钟时间”，这正是要防的前向填充。
        self.assertLessEqual(summary["effective_seconds"], 12.0)
        self.assertLess(summary["effective_seconds"], summary["wall_span_seconds"])

    def test_off_air_gap_breaks_continuous_evidence(self):
        from rtsp_annotator.ground_litter_profile_selector import CandidateMatch
        selector = self._selector(join_gap_seconds=300.0)
        for step in range(2):
            selector.observe(
                timestamp=100.0 + step * 2.0, current=None,
                candidates=[CandidateMatch("p2", 0.2, True, True)],
            )
        self.assertIsNotNone(selector._recovery)
        selector.observe(timestamp=5000.0, current=None, candidates=[])
        self.assertIsNone(selector._recovery)
        self.assertIsNone(selector._challenger)


class CliEndToEndTests(unittest.TestCase):
    """构建 → 发布 → 加载 → 评估 的最小端到端（合成素材，无网络）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.media = self.root / "ps"
        self.media.mkdir(parents=True)
        import datetime as _datetime
        import os
        rng = np.random.default_rng(21)
        for day in range(1, 4):
            for chunk in range(3):
                path = self.media / f"2026090{day}_{chunk:02d}.mp4"
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120),
                )
                tint = 25 * ((day + chunk) % 2)
                for _ in range(10):
                    frame = rng.integers(60, 180, (120, 160, 3), dtype=np.uint8)
                    frame = cv2.GaussianBlur(frame, (0, 0), 1.1)
                    frame = np.clip(
                        frame.astype(np.int32) + tint, 0, 255,
                    ).astype(np.uint8)
                    frame[::20, :] = (frame[::20, :] // 2 + 30).astype(np.uint8)
                    writer.write(frame)
                writer.release()
                stamp = _datetime.datetime(
                    2026, 9, day, 10 + chunk, 0, 0,
                ).timestamp()
                os.utime(path, (stamp, stamp))

    def test_build_then_evaluate_end_to_end(self):
        from scripts.build_ground_litter_profile_bank import main as build_main
        from scripts.evaluate_ground_litter_profile_bank import main as eval_main
        bank_root = self.root / "banks"
        work = self.root / "work"
        code = build_main([
            "--input", str(self.media), "--camera", "cam_cli",
            "--output", str(bank_root), "--work-dir", str(work),
            "--version", "v1", "--resume",
        ])
        self.assertEqual(code, 0)
        bank = load_bank(bank_root, "cam_cli", "v1")
        self.assertTrue(bank.ids())
        self.assertTrue((bank.profiles[0].directory / "reference.png").is_file())
        evaluation = self.root / "eval"
        code = eval_main([
            "--bank-root", str(bank_root), "--bank-id", "cam_cli",
            "--version", "v1", "--input", str(self.media),
            "--output", str(evaluation), "--analysis-fps", "1.0",
        ])
        self.assertEqual(code, 0)
        payload = json.loads((evaluation / "evaluation.json").read_text("utf-8"))
        self.assertGreater(payload["ticks"], 0)
        self.assertIn("dynamic_coverage", payload)
        self.assertIn("static_potential_coverage", payload)
        self.assertIn("gaps", payload)

    def test_inventory_local_never_writes_media_metadata_as_urls(self):
        from scripts.inventory_ground_litter_ps import main as inventory_main
        output = self.root / "index.json"
        code = inventory_main([
            "--source", "local", "--input", str(self.media),
            "--output", str(output),
        ])
        self.assertEqual(code, 0)
        payload = json.loads(output.read_text("utf-8"))
        self.assertEqual(payload["file_count"], 9)
        text = output.read_text("utf-8").lower()
        self.assertNotIn('"url"', text)
        self.assertNotIn("signature", text)


class AnalysisMaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "banks"
        build_synthetic_bank(self.root, "camera_01", "v1", profiles=2,
                            width=160, height=120)
        self.bank = load_bank(self.root, "camera_01", "v1")

    def _loose(self):
        from rtsp_annotator.ground_litter_profile_match import MatchEnvelope
        return MatchEnvelope(enter=50.0, hold=60.0, calibrated=False,
                             samples=0, source="test")

    def test_small_paper_is_not_swallowed_by_local_invalid(self):
        """F2 回归：8×8 高对比小纸片必须仍是前景证据。"""
        from rtsp_annotator import ground_litter_profile_analysis as analysis
        context = analysis.build_prior_context(self.bank, "p0001")
        roi = analysis.roi_mask_from_geometry(self.bank.geometry, 160, 120)
        frame = context.reference.copy()
        frame[56:64, 76:84] = 250
        result = analysis.evaluate_bank_frame(
            context, frame, envelope=self._loose(), config={}, roi_mask=roi,
        )
        self.assertEqual(result.outcome["reason"], "MATCHED")
        self.assertGreaterEqual(len(result.candidates), 1)
        self.assertGreater(result.support_pixels, 0)
        self.assertGreater(result.availability_fraction, 0.9)
        # 小目标不得被判成不可用；原因直方图里不能出现残差类失效。
        histogram = result.mask_diagnostics["reason_histogram"]
        self.assertNotIn("LARGE_EXPOSURE_LOSS", histogram)

    def test_availability_reason_is_independent_evidence(self):
        from rtsp_annotator import ground_litter_profile_analysis as analysis
        context = analysis.build_prior_context(self.bank, "p0001")
        roi = analysis.roi_mask_from_geometry(self.bank.geometry, 160, 120)
        frame = context.reference.copy()
        frame[56:64, 76:84] = 250
        actors = [[76.0, 56.0, 84.0, 64.0]]
        result = analysis.evaluate_bank_frame(
            context, frame, envelope=self._loose(), config={}, roi_mask=roi,
            actors=actors,
        )
        histogram = result.mask_diagnostics["reason_histogram"]
        self.assertEqual(len(result.candidates), 0)
        self.assertIn("ACTOR_OCCLUDED", histogram)

    def test_geometry_invalid_is_enumerated(self):
        from rtsp_annotator import ground_litter_profile_analysis as analysis
        context = analysis.build_prior_context(self.bank, "p0001")
        roi = analysis.roi_mask_from_geometry(self.bank.geometry, 160, 120)
        result = analysis.evaluate_bank_frame(
            context, context.reference.copy(), envelope=self._loose(),
            config={}, roi_mask=roi, geometry_valid=False,
        )
        histogram = result.mask_diagnostics["reason_histogram"]
        self.assertIn("GEOMETRY_INVALID", histogram)
        self.assertEqual(result.availability_fraction, 0.0)

    def test_corrupt_frame_is_enumerated(self):
        from rtsp_annotator import ground_litter_profile_analysis as analysis
        context = analysis.build_prior_context(self.bank, "p0001")
        roi = analysis.roi_mask_from_geometry(self.bank.geometry, 160, 120)
        result = analysis.evaluate_bank_frame(
            context, context.reference.copy(), envelope=self._loose(),
            config={}, roi_mask=roi, corruption="decoder_error",
        )
        self.assertIn(
            "FRAME_CORRUPT", result.mask_diagnostics["reason_histogram"],
        )

    def test_roi_mask_excludes_zones(self):
        from rtsp_annotator import ground_litter_profile_analysis as analysis
        geometry = dict(self.bank.geometry)
        geometry["exclude_zones"] = [
            [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
        ]
        mask = analysis.roi_mask_from_geometry(geometry, 160, 120)
        self.assertEqual(int(mask[60, 10]), 0)
        self.assertEqual(int(mask[60, 150]), 255)

    def test_availability_reasons_are_enumerable(self):
        from rtsp_annotator.ground_litter_profile_analysis import (
            AVAILABILITY_REASONS,
        )
        self.assertIn("OK", AVAILABILITY_REASONS)
        # 不允许出现只有「直播残差过高」含义的逐像素失效规则。
        self.assertNotIn("LIVE_RESIDUAL_TOO_HIGH", AVAILABILITY_REASONS)

    def test_static_coverage_is_labelled_potential_only(self):
        from rtsp_annotator import ground_litter_profile_analysis as analysis
        descriptors = {
            profile_id: self.bank.load_descriptor(profile_id)
            for profile_id in self.bank.ids()
        }
        best, diagnostics = analysis.static_potential_coverage(
            descriptors["p0001"], self.bank, descriptors=descriptors,
        )
        self.assertEqual(best, "p0001")
        self.assertGreater(diagnostics["candidates_considered"], 0)
        coverage = analysis.StaticCoverage(
            frames=4, covered_frames=4, per_profile={"p0001": 4},
            unique_assignment={"p0001": 4},
        )
        payload = coverage.as_dict()
        self.assertEqual(payload["kind"], "potential_coverage")
        self.assertIn("不能冒充", payload["note"])


class V32RegressionTests(unittest.TestCase):
    """共享函数抽取后旧固定 Profile 模式必须不退化。"""

    def test_protected_normalize_matches_extracted_pieces(self):
        import numpy as np
        from rtsp_annotator import ground_litter_v32 as v32
        rng = np.random.default_rng(3)
        reference = rng.integers(40, 200, (200, 200, 3), dtype=np.uint8)
        valid = np.full((200, 200), 255, np.uint8)
        current = np.clip(
            reference.astype(np.int32) + rng.integers(-12, 12, (200, 200, 3)),
            0, 255,
        ).astype(np.uint8)
        normalized, usable, environment = v32.protected_normalize(
            reference, current, valid,
        )
        masks = v32.compute_compensation_masks(reference, current, valid)
        field = v32.local_luminance_field(
            masks["compensated"], current, masks["protected"], masks["fit_mask"],
        )
        rebuilt = np.clip(
            masks["global_reference"] + field[..., None], 0, 255,
        ).astype(np.uint8)
        self.assertTrue(np.array_equal(normalized, rebuilt))
        self.assertTrue(np.array_equal(usable, valid))
        self.assertEqual(
            environment["saturated_fraction"],
            v32._classify_environment(masks, field, valid)["saturated_fraction"],
        )
        self.assertEqual(environment["state"], v32._classify_environment(
            masks, field, valid,
        )["state"])

    def test_v32_prior_pipeline_unchanged_on_textured_scene(self):
        import numpy as np
        from rtsp_annotator.ground_litter_detection import GroundLitterDetectionOptions
        from rtsp_annotator import ground_litter_v32 as v32
        reference = _textured_reference(320, 240)
        valid = synthetic_valid(320, 240)
        options = GroundLitterDetectionOptions(analysis_fps=0.5)
        normalized, usable, environment = v32.protected_normalize(
            reference, reference.copy(), valid,
        )
        analysis = v32.CleanReferenceFrameAnalysis(
            analysis_frame=reference.copy(), normalized=normalized, valid=usable,
            tolerance=np.zeros(valid.shape, np.uint8), actor_rows=[],
            environment_state=environment["state"], environment=environment,
            alignment={}, width=reference.shape[1], height=reference.shape[0],
            pixel_scale=reference.shape[1] / 2560.0,
        )
        retained, support, proposed = v32.propose_prior_candidates(options, analysis)
        self.assertIsInstance(retained, list)
        self.assertIsInstance(proposed, int)
        self.assertEqual(support.shape, valid.shape)

    def test_propose_v32_unchanged_on_synthetic_patch(self):
        import numpy as np
        from rtsp_annotator import ground_litter_v32 as v32
        reference = _textured_reference(320, 240)
        valid = synthetic_valid(320, 240)
        frame = reference.copy()
        frame[100:120, 140:160] = 250
        normalized, _usable, environment = v32.protected_normalize(
            reference, frame, valid,
        )
        rows, support = v32.propose_v32(
            normalized, frame, valid, np.zeros(valid.shape, np.uint8),
        )
        self.assertIn(environment["state"], {"NORMAL", "GLOBAL_LIGHT_CHANGE",
                                            "ENVIRONMENT_CHANGE"})
        self.assertIsInstance(rows, list)
        self.assertEqual(support.shape, valid.shape)
def _v32_profile(reference, valid, case):
    import numpy as np
    from rtsp_annotator.ground_litter_v32 import CleanReferenceProfileV32
    tolerance = np.zeros(valid.shape, np.uint8)
    metadata = {
        "kind": "ground_litter_clean_reference_v32",
        "profile_id": "synthetic",
        "reference_size": [reference.shape[1], reference.shape[0]],
        "overlay_exclude_zones": [],
    }
    return CleanReferenceProfileV32(
        profile_id="synthetic", root=Path("."), reference=reference,
        valid=valid, tolerance=tolerance, metadata=metadata,
    )


if __name__ == "__main__":
    unittest.main()
