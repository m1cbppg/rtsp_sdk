"""受管 PS 缓存：空间预算、租约、阶段事务与提交后清理（方案一 §4.5～§4.8）。

设计要点（对应文档条款）：

* **材料化状态与阶段状态分开**：``ABSENT/DOWNLOADING/READY/LEASED/EVICTABLE`` 与
  ``preview/hd_extract/calibration_replay/blind_replay`` 四类阶段键互不覆盖。
  ``preview`` 完成**不**等于 ``hd_extract`` 或整个文件任务完成。
* **空间预算**：``raw_cache_budget`` 只统计受管下载缓存（含 ``.part``、预取、处理中、
  失败残留），按声明的 ``fileSize`` 预留；``work_budget`` 统计整个工作目录（含高清样本、
  资产、报告）。二者都是硬上限——超限时拒绝派发并报告背压，不做静默超卖。
* **租约**：只有被显式 ``acquire`` 的文件才可以被消费；持租约期间绝不允许删除。
* **删除条件必须全部满足**：受管缓存内 + 无租约 + 阶段产物已落盘并校验 +
  阶段已事务提交 + 有重拉路径。绝不调用远端删除接口，绝不删除用户本地源文件。
* **崩溃恢复幂等**：提交后未删除 → 可重新清理；未提交 → 不释放源；源已删且阶段完成 →
  跳过；新阶段 → 按稳定 fileId 重拉。重拉得到不同 SHA-256 视为源版本变化，
  相关产物全部失效。
* 签名的临时 URL 不进入本模块：``acquire`` 返回 ``RecordingFile`` 与路径，
  刷新与下载由调用方持有 URL。
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .ground_litter_profile_bank import atomic_write_json, sha256_file
from .ground_litter_recording_source import RecordingFile

MATERIALIZATION_STATES = ("ABSENT", "DOWNLOADING", "READY", "LEASED", "EVICTABLE")
STAGE_NAMES = ("preview", "hd_extract", "calibration_replay", "blind_replay")
TERMINAL_STAGES = STAGE_NAMES  # 用于说明：这些阶段都提交后才算文件任务完成

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
  identity_key   TEXT PRIMARY KEY,
  device_code    TEXT NOT NULL,
  file_id        TEXT NOT NULL,
  file_name      TEXT NOT NULL DEFAULT '',
  record_start   TEXT NOT NULL,
  record_end     TEXT NOT NULL,
  declared_size  INTEGER,
  materialization TEXT NOT NULL DEFAULT 'ABSENT',
  path           TEXT,
  bytes          INTEGER NOT NULL DEFAULT 0,
  sha256         TEXT,
  source_version INTEGER NOT NULL DEFAULT 1,
  managed        INTEGER NOT NULL DEFAULT 1,
  range_supported INTEGER,
  failure        TEXT,
  attempts       INTEGER NOT NULL DEFAULT 0,
  downloaded_bytes INTEGER NOT NULL DEFAULT 0,
  download_seconds REAL NOT NULL DEFAULT 0,
  updated        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS leases (
  lease_id   TEXT PRIMARY KEY,
  identity_key TEXT NOT NULL,
  owner      TEXT NOT NULL,
  acquired   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS stages (
  identity_key TEXT NOT NULL,
  stage        TEXT NOT NULL,
  status       TEXT NOT NULL,
  input_hash   TEXT,
  config_hash  TEXT,
  algorithm_version TEXT,
  artifact     TEXT,
  artifact_sha256 TEXT,
  updated      TEXT NOT NULL,
  detail       TEXT,
  PRIMARY KEY (identity_key, stage)
);
CREATE TABLE IF NOT EXISTS artifacts (
  identity_key TEXT NOT NULL,
  stage        TEXT NOT NULL,
  name         TEXT NOT NULL,
  path         TEXT NOT NULL,
  sha256       TEXT NOT NULL,
  bytes        INTEGER NOT NULL,
  invalidated  INTEGER NOT NULL DEFAULT 0,
  created      TEXT NOT NULL,
  PRIMARY KEY (identity_key, stage, name)
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  identity_key TEXT,
  kind TEXT NOT NULL,
  detail TEXT NOT NULL,
  created TEXT NOT NULL
);
"""


class CacheError(RuntimeError):
    """缓存/预算/租约失败。"""


class LeaseError(CacheError):
    """文件被其他消费者占用或租约不存在。"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=str)


def hash_stage_key(*, input_sha256: str | None, algorithm_version: str,
                   config: Any) -> str:
    """阶段键必须包含输入散列、算法版本与配置版本（方案一 §4.6）。"""
    from .ground_litter_profile_bank import sha256_bytes
    return sha256_bytes(
        f"{input_sha256 or 'unknown'}|{algorithm_version}|{_canonical(config)}".encode()
    )


@dataclass(frozen=True, slots=True)
class RecordingEntry:
    identity_key: str
    device_code: str
    file: RecordingFile
    materialization: str
    path: Path | None
    bytes: int
    sha256: str | None
    source_version: int
    managed: bool
    range_supported: bool | None
    failure: str | None
    attempts: int
    downloaded_bytes: int
    download_seconds: float
    updated: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity_key": self.identity_key,
            "file_id": self.file.file_id,
            "record_start": self.file.record_start,
            "record_end": self.file.record_end,
            "materialization": self.materialization,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "source_version": self.source_version,
            "managed": self.managed,
            "range_supported": self.range_supported,
            "failure": self.failure,
            "attempts": self.attempts,
            "downloaded_bytes": self.downloaded_bytes,
            "download_seconds": round(self.download_seconds, 3),
            "updated": self.updated,
        }


@dataclass(slots=True)
class BudgetReport:
    raw_bytes: int = 0
    part_bytes: int = 0
    managed_file_bytes: int = 0
    unmanaged_bytes: int = 0
    leased_count: int = 0
    total_work_bytes: int = 0
    raw_budget: int = 0
    work_budget: int = 0
    backpressure: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_bytes": self.raw_bytes,
            "part_bytes": self.part_bytes,
            "managed_file_bytes": self.managed_file_bytes,
            "unmanaged_bytes": self.unmanaged_bytes,
            "leased_count": self.leased_count,
            "total_work_bytes": self.total_work_bytes,
            "raw_budget": self.raw_budget,
            "work_budget": self.work_budget,
            "backpressure": self.backpressure,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class Lease:
    lease_id: str
    identity_key: str
    entry: RecordingEntry
    acquired_monotonic: float

    @property
    def path(self) -> Path:
        if self.entry.path is None:
            raise LeaseError("租约没有可用文件路径")
        return self.entry.path


class ManagedRecordingCache:
    """SQLite 记账 + 受管文件目录。进程内加锁，跨进程用 SQLite 事务。"""

    def __init__(
        self, work_dir: str | Path, *,
        raw_cache_budget: int = 1024 * 1024 * 1024,
        work_budget: int = 20 * 1024 * 1024 * 1024,
    ) -> None:
        if raw_cache_budget < 1 or work_budget < 1:
            raise CacheError("空间预算必须为正")
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.raw_dir = self.work_dir / "raw-cache"
        self.db_path = self.work_dir / "cache.sqlite3"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.raw_cache_budget = int(raw_cache_budget)
        self.work_budget = int(work_budget)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.executescript(_CACHE_SCHEMA)
        self.peak_raw_bytes = 0
        self.peak_work_bytes = 0

    # -- 基础 -------------------------------------------------------------- #

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "ManagedRecordingCache":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _log(self, identity_key: str | None, kind: str, detail: Mapping[str, Any]) -> None:
        self._connection.execute(
            "INSERT INTO events(identity_key, kind, detail, created) VALUES (?,?,?,?)",
            (identity_key, kind, _canonical(detail), _now()),
        )

    def events(self, *, limit: int = 500) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
        return [dict(row) for row in rows]

    # -- 清单登记 ---------------------------------------------------------- #

    def register(self, device_code: str, files: Iterable[RecordingFile]) -> int:
        """登记/更新稳定身份；URL 变化不产生新条目。"""
        added = 0
        with self._lock, self._connection:
            for item in files:
                key = RecordingFile.identity_key(device_code, item.file_id)
                row = self._connection.execute(
                    "SELECT source_version, declared_size, sha256 FROM recordings "
                    "WHERE identity_key=?", (key,),
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        "INSERT INTO recordings(identity_key, device_code, file_id, file_name,"
                        " record_start, record_end, declared_size, updated)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        (key, device_code, item.file_id, item.file_name,
                         item.record_start, item.record_end, item.file_size, _now()),
                    )
                    added += 1
                else:
                    self._connection.execute(
                        "UPDATE recordings SET file_name=?, record_start=?, record_end=?,"
                        " declared_size=?, updated=? WHERE identity_key=?",
                        (item.file_name, item.record_start, item.record_end,
                         item.file_size, _now(), key),
                    )
        return added

    def entries(self) -> list[RecordingEntry]:
        rows = self._connection.execute(
            "SELECT * FROM recordings ORDER BY record_start, file_id"
        ).fetchall()
        return [self._entry_from_row(row) for row in rows]

    def entry(self, device_code: str, file_id: str) -> RecordingEntry | None:
        key = RecordingFile.identity_key(device_code, file_id)
        row = self._connection.execute(
            "SELECT * FROM recordings WHERE identity_key=?", (key,)
        ).fetchone()
        return None if row is None else self._entry_from_row(row)

    def _entry_from_row(self, row: sqlite3.Row) -> RecordingEntry:
        return RecordingEntry(
            identity_key=row["identity_key"],
            device_code=row["device_code"],
            file=RecordingFile(
                file_id=row["file_id"], file_name=row["file_name"],
                record_start=row["record_start"], record_end=row["record_end"],
                file_size=row["declared_size"],
            ),
            materialization=row["materialization"],
            path=Path(row["path"]) if row["path"] else None,
            bytes=int(row["bytes"] or 0),
            sha256=row["sha256"],
            source_version=int(row["source_version"]),
            managed=bool(row["managed"]),
            range_supported=(
                None if row["range_supported"] is None else bool(row["range_supported"])
            ),
            failure=row["failure"],
            attempts=int(row["attempts"] or 0),
            downloaded_bytes=int(row["downloaded_bytes"] or 0),
            download_seconds=float(row["download_seconds"] or 0.0),
            updated=row["updated"],
        )

    # -- 预算 -------------------------------------------------------------- #

    def managed_bytes(self) -> int:
        """受管缓存**实际占用**：数据库 `bytes` 之和 + 无记录的 .part 残留。"""
        rows = self._connection.execute(
            "SELECT COALESCE(SUM(bytes), 0) AS total FROM recordings"
        ).fetchone()
        total = int(rows["total"] or 0)
        recorded = {
            Path(row["path"]).name for row in self._connection.execute(
                "SELECT path FROM recordings WHERE path IS NOT NULL"
            ).fetchall()
        }
        for path in self.raw_dir.glob("*.part"):
            if path.name not in recorded and path.is_file():
                total += path.stat().st_size
        return total

    def work_bytes(self) -> int:
        total = 0
        for path in self.work_dir.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:  # pragma: no cover
                    continue
        return total

    def reserved_bytes(self, *, exclude: Sequence[str] = ()) -> int:
        """尚未落盘但必须预留的空间：正在下载的声明大小。"""
        rows = self._connection.execute(
            "SELECT declared_size, bytes, path FROM recordings WHERE materialization IN"
            " ('DOWNLOADING','READY','LEASED')"
        ).fetchall()
        excluded = set(exclude)
        total = 0
        for row in rows:
            path = row["path"]
            if path and Path(path).name.split(".")[0] in excluded:
                continue
            on_disk = int(row["bytes"] or 0)
            if on_disk > 0:
                continue
            total += int(row["declared_size"] or 0)
        return total

    def budget_report(self) -> BudgetReport:
        managed = self.managed_bytes()
        work = self.work_bytes()
        part = sum(
            path.stat().st_size for path in self.raw_dir.glob("*.part")
            if path.is_file()
        )
        report = BudgetReport(
            raw_bytes=managed,
            part_bytes=part,
            managed_file_bytes=max(0, managed - part),
            unmanaged_bytes=max(0, work - managed),
            leased_count=int(self._connection.execute(
                "SELECT COUNT(*) AS n FROM leases"
            ).fetchone()["n"]),
            total_work_bytes=work,
            raw_budget=self.raw_cache_budget,
            work_budget=self.work_budget,
        )
        self.peak_raw_bytes = max(self.peak_raw_bytes, report.raw_bytes)
        self.peak_work_bytes = max(self.peak_work_bytes, report.total_work_bytes)
        if report.raw_bytes > self.raw_cache_budget:
            report.backpressure = True
            report.reason = "raw_cache_budget"
        elif report.total_work_bytes > self.work_budget:
            report.backpressure = True
            report.reason = "work_budget"
        return report

    def can_reserve(self, size: int | None) -> tuple[bool, str]:
        """按声明大小预留空间；未知大小使用 ``unknown_size_reserve`` 兜底。

        账号口径：受管缓存**实际占用** + 尚未落盘的下载预留 + 本次需求。
        """
        report = self.budget_report()
        need = int(size) if size is not None and size > 0 else self.unknown_size_reserve
        projected_raw = report.raw_bytes + self.reserved_bytes() + need
        if projected_raw > self.raw_cache_budget:
            return False, "raw_cache_budget"
        if report.total_work_bytes + need > self.work_budget:
            return False, "work_budget"
        return True, ""

    unknown_size_reserve = 80 * 1024 * 1024

    # -- 下载生命周期 ------------------------------------------------------ #

    def begin_download(self, device_code: str, file: RecordingFile) -> Path:
        """把条目置为 DOWNLOADING 并返回目标路径；期间计入预算。"""
        ok, reason = self.can_reserve(file.file_size)
        if not ok:
            raise CacheError(f"backpressure:{reason}")
        key = RecordingFile.identity_key(device_code, file.file_id)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT materialization FROM recordings WHERE identity_key=?", (key,)
            ).fetchone()
            if row is None:
                raise CacheError("必须先 register 再下载")
            if row["materialization"] == "LEASED":
                raise LeaseError("文件正在被消费，不能重新下载")
            target = self.raw_dir / f"{key.replace(':', '_')}.bin"
            self._connection.execute(
                "UPDATE recordings SET materialization='DOWNLOADING', path=?, failure=NULL,"
                " bytes=0, sha256=NULL, updated=? WHERE identity_key=?",
                (str(target), _now(), key),
            )
            self._log(key, "download_begin", {"file_size": file.file_size})
        return target

    def complete_download(
        self, device_code: str, file_id: str, *, path: Path, size: int, sha256: str,
        range_supported: bool | None, elapsed_seconds: float,
    ) -> RecordingEntry:
        key = RecordingFile.identity_key(device_code, file_id)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT sha256, source_version, downloaded_bytes, download_seconds"
                " FROM recordings WHERE identity_key=?", (key,),
            ).fetchone()
            if row is None:
                raise CacheError("条目不存在")
            previous = row["sha256"]
            version = int(row["source_version"] or 1)
            invalidated = 0
            if previous and previous != sha256:
                # 源版本变化：相关产物失效，不能覆盖后沿用旧成绩。
                version += 1
                cursor = self._connection.execute(
                    "UPDATE artifacts SET invalidated=1 WHERE identity_key=?", (key,)
                )
                invalidated = cursor.rowcount or 0
                self._connection.execute(
                    "UPDATE stages SET status='STALE', updated=? WHERE identity_key=?",
                    (_now(), key),
                )
                self._log(key, "source_version_changed", {
                    "previous_sha256": previous, "sha256": sha256,
                    "invalidated_artifacts": invalidated,
                })
            self._connection.execute(
                "UPDATE recordings SET materialization='READY', path=?, bytes=?, sha256=?,"
                " source_version=?, range_supported=?, failure=NULL, attempts=attempts+1,"
                " downloaded_bytes=downloaded_bytes+?, download_seconds=download_seconds+?,"
                " updated=? WHERE identity_key=?",
                (str(path), int(size), sha256, version,
                 None if range_supported is None else int(range_supported),
                 int(size), float(elapsed_seconds), _now(), key),
            )
            self._log(key, "download_complete", {
                "size": size, "sha256": sha256,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "range_supported": range_supported, "source_version": version,
            })
        return self.entry(device_code, file_id)  # type: ignore[return-value]

    def fail_download(self, device_code: str, file_id: str, reason: str,
                      *, bytes_received: int = 0, elapsed_seconds: float = 0.0) -> None:
        key = RecordingFile.identity_key(device_code, file_id)
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE recordings SET materialization='ABSENT', failure=?, attempts=attempts+1,"
                " downloaded_bytes=downloaded_bytes+?, download_seconds=download_seconds+?,"
                " updated=? WHERE identity_key=?",
                (str(reason)[:200], int(bytes_received), float(elapsed_seconds), _now(), key),
            )
            self._log(key, "download_failed", {
                "reason": str(reason)[:200], "bytes_received": bytes_received,
            })
        # 失败残留必须清掉，否则会永久占用 raw 预算。
        self.discard_partial(device_code, file_id)

    def discard_partial(self, device_code: str, file_id: str) -> None:
        key = RecordingFile.identity_key(device_code, file_id)
        row = self._connection.execute(
            "SELECT path FROM recordings WHERE identity_key=?", (key,)
        ).fetchone()
        if row is None or not row["path"]:
            return
        target = Path(row["path"])
        for candidate in (target, target.with_name(target.name + ".part")):
            if candidate.is_file():
                candidate.unlink(missing_ok=True)

    # -- 租约 -------------------------------------------------------------- #

    @contextmanager
    def acquire(self, device_code: str, file_id: str, *, owner: str = "worker",
                lease_id: str | None = None) -> Iterator[Lease]:
        lease = self.begin_lease(device_code, file_id, owner=owner, lease_id=lease_id)
        try:
            yield lease
        finally:
            self.release(lease.lease_id)

    def begin_lease(self, device_code: str, file_id: str, *, owner: str = "worker",
                    lease_id: str | None = None) -> Lease:
        key = RecordingFile.identity_key(device_code, file_id)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM recordings WHERE identity_key=?", (key,)
            ).fetchone()
            if row is None:
                raise CacheError("条目不存在")
            if row["materialization"] == "EVICTABLE" and row["path"] and Path(
                row["path"]
            ).is_file():
                # EVICTABLE 只是「可以删」，不代表文件已经不在；本地只读源
                # 在两次消费之间会停在这个状态。
                with self._lock, self._connection:
                    self._connection.execute(
                        "UPDATE recordings SET materialization='READY', updated=?"
                        " WHERE identity_key=?", (_now(), key),
                    )
                row = self._connection.execute(
                    "SELECT * FROM recordings WHERE identity_key=?", (key,)
                ).fetchone()
            if row["materialization"] not in ("READY", "LEASED"):
                raise LeaseError(
                    f"文件不可消费（materialization={row['materialization']}）"
                )
            if row["path"] is None or not Path(row["path"]).is_file():
                raise LeaseError("文件已不在磁盘上，需要重拉")
            identifier = lease_id or uuid.uuid4().hex
            self._connection.execute(
                "INSERT INTO leases(lease_id, identity_key, owner, acquired) VALUES (?,?,?,?)",
                (identifier, key, owner, time.monotonic()),
            )
            self._connection.execute(
                "UPDATE recordings SET materialization='LEASED', updated=? WHERE identity_key=?",
                (_now(), key),
            )
            self._log(key, "lease_begin", {"owner": owner, "lease_id": identifier})
        entry = self.entry(device_code, file_id)
        assert entry is not None
        return Lease(identifier, key, entry, time.monotonic())

    def release(self, lease_id: str) -> None:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT identity_key FROM leases WHERE lease_id=?", (lease_id,)
            ).fetchone()
            if row is None:
                return
            key = row["identity_key"]
            self._connection.execute("DELETE FROM leases WHERE lease_id=?", (lease_id,))
            remaining = self._connection.execute(
                "SELECT COUNT(*) AS n FROM leases WHERE identity_key=?", (key,)
            ).fetchone()["n"]
            if not remaining:
                self._connection.execute(
                    "UPDATE recordings SET materialization='EVICTABLE', updated=?"
                    " WHERE identity_key=? AND materialization='LEASED'",
                    (_now(), key),
                )
            self._log(key, "lease_release", {"lease_id": lease_id})

    def active_leases(self) -> list[dict[str, Any]]:
        rows = self._connection.execute("SELECT * FROM leases").fetchall()
        return [dict(row) for row in rows]

    # -- 阶段事务 ---------------------------------------------------------- #

    def record_artifact(
        self, device_code: str, file_id: str, stage: str, name: str, path: str | Path,
        *, expected_sha256: str | None = None, commit: bool = True,
        input_hash: str | None = None, config: Any = None,
        algorithm_version: str = "", detail: Mapping[str, Any] | None = None,
    ) -> str:
        """把阶段产物落盘校验后**事务提交**；返回产物 sha256。

        产物已存在且校验一致时视为幂等重放，不重复写。
        """
        if stage not in STAGE_NAMES:
            raise CacheError(f"未知阶段: {stage}")
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise CacheError(f"阶段产物不存在: {artifact_path.name}")
        digest = sha256_file(artifact_path)
        if expected_sha256 and digest != expected_sha256.lower():
            raise CacheError("阶段产物校验失败")
        key = RecordingFile.identity_key(device_code, file_id)
        size = artifact_path.stat().st_size
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT sha256, invalidated FROM artifacts WHERE identity_key=? AND stage=?"
                " AND name=?", (key, stage, name),
            ).fetchone()
            if existing and existing["sha256"] == digest and not existing["invalidated"]:
                pass
            else:
                self._connection.execute(
                    "INSERT OR REPLACE INTO artifacts(identity_key, stage, name, path, sha256,"
                    " bytes, invalidated, created) VALUES (?,?,?,?,?,?,0,?)",
                    (key, stage, name, str(artifact_path), digest, size, _now()),
                )
            if commit:
                stage_key = hash_stage_key(
                    input_sha256=input_hash, algorithm_version=algorithm_version, config=config,
                )
                self._connection.execute(
                    "INSERT OR REPLACE INTO stages(identity_key, stage, status, input_hash,"
                    " config_hash, algorithm_version, artifact, artifact_sha256, updated, detail)"
                    " VALUES (?,?,'COMMITTED',?,?,?,?,?,?,?)",
                    (key, stage, input_hash, stage_key, algorithm_version,
                     str(artifact_path), digest, _now(), _canonical(detail or {})),
                )
                self._log(key, "stage_commit", {"stage": stage, "artifact": artifact_path.name})
        return digest

    def stage_status(self, device_code: str, file_id: str) -> dict[str, dict[str, Any]]:
        key = RecordingFile.identity_key(device_code, file_id)
        rows = self._connection.execute(
            "SELECT * FROM stages WHERE identity_key=?", (key,)
        ).fetchall()
        return {row["stage"]: dict(row) for row in rows}

    def stage_committed(self, device_code: str, file_id: str, stage: str) -> bool:
        return self.stage_status(device_code, file_id).get(stage, {}).get("status") == "COMMITTED"

    def file_task_complete(self, device_code: str, file_id: str,
                           *, stages: Sequence[str] = TERMINAL_STAGES) -> bool:
        """只有全部目标阶段都提交才算文件任务完成——preview 不等于完成。"""
        status = self.stage_status(device_code, file_id)
        return all(status.get(stage, {}).get("status") == "COMMITTED" for stage in stages)

    def artifacts(self, device_code: str, file_id: str, *,
                  include_invalidated: bool = False) -> list[dict[str, Any]]:
        key = RecordingFile.identity_key(device_code, file_id)
        query = "SELECT * FROM artifacts WHERE identity_key=?"
        if not include_invalidated:
            query += " AND invalidated=0"
        return [dict(row) for row in self._connection.execute(query, (key,)).fetchall()]

    # -- 删除 -------------------------------------------------------------- #

    def can_delete(self, device_code: str, file_id: str) -> tuple[bool, str]:
        """检查全部删除条件；返回 (是否可删, 原因)。"""
        key = RecordingFile.identity_key(device_code, file_id)
        row = self._connection.execute(
            "SELECT * FROM recordings WHERE identity_key=?", (key,)
        ).fetchone()
        if row is None:
            return False, "NOT_MANAGED"
        if not row["managed"]:
            return False, "UNMANAGED_SOURCE"
        if row["materialization"] in ("ABSENT", "EVICTABLE"):
            return True, "ALREADY_RELEASED"
        if row["materialization"] == "LEASED":
            return False, "LEASED"
        if row["materialization"] != "READY":
            return False, f"STATE_{row['materialization']}"
        leases = self._connection.execute(
            "SELECT COUNT(*) AS n FROM leases WHERE identity_key=?", (key,)
        ).fetchone()["n"]
        if leases:
            return False, "LEASED"
        if not row["sha256"]:
            return False, "NEVER_VERIFIED"
        return True, "OK"

    def release_file(
        self, device_code: str, file_id: str, *, require_committed: Sequence[str] = (),
    ) -> bool:
        """释放受管临时 PS。返回 True 表示本次真的删除了文件。

        传入 ``require_committed`` 时，任一阶段未提交即拒绝删除。
        """
        key = RecordingFile.identity_key(device_code, file_id)
        allowed, reason = self.can_delete(device_code, file_id)
        if not allowed:
            self._log(key, "release_denied", {"reason": reason})
            return False
        for stage in require_committed:
            if not self.stage_committed(device_code, file_id, stage):
                self._log(key, "release_denied", {"reason": f"stage_not_committed:{stage}"})
                return False
        row = self._connection.execute(
            "SELECT path FROM recordings WHERE identity_key=?", (key,)
        ).fetchone()
        removed = False
        if row and row["path"]:
            target = Path(row["path"])
            for candidate in (target, target.with_name(target.name + ".part")):
                if candidate.is_file():
                    candidate.unlink(missing_ok=True)
                    removed = True
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE recordings SET materialization='EVICTABLE', bytes=0, path=NULL,"
                " updated=? WHERE identity_key=?", (_now(), key),
            )
            self._log(key, "released", {"removed": removed})
        return removed

    def evict_to_budget(self, *, allow_ready: bool | None = None) -> list[str]:
        """按需释放受管临时副本。

        默认只在超预算时扩大到 ``READY`` 且无租约的文件（它们是可重拉的缓存，
        不是后续必需产物）；``allow_ready=False`` 时只清理已释放条目。
        """
        backpressure = self.budget_report().backpressure
        if allow_ready is None:
            allow_ready = backpressure
        states = ["EVICTABLE"]
        if allow_ready:
            states.append("READY")
        placeholders = ",".join("?" for _ in states)
        rows = self._connection.execute(
            f"SELECT device_code, file_id, materialization FROM recordings"
            f" WHERE materialization IN ({placeholders}) AND managed=1 AND path IS NOT NULL",
            tuple(states),
        ).fetchall()
        released: list[str] = []
        for row in rows:
            if row["materialization"] == "READY" and not allow_ready:
                continue
            if self.release_file(row["device_code"], row["file_id"]):
                released.append(f"{row['device_code']}:{row['file_id']}")
        return released

    # -- 崩溃恢复 ---------------------------------------------------------- #

    def recover(self) -> dict[str, Any]:
        """幂等恢复：DOWNLOADING 残留作废、孤儿租约清理、已删源标记 ABSENT。"""
        summary: dict[str, Any] = {
            "stale_downloads": [], "orphan_leases": [], "missing_files": [],
        }
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT identity_key, device_code, file_id, path FROM recordings"
                " WHERE materialization='DOWNLOADING'"
            ).fetchall()
            for row in rows:
                self._connection.execute(
                    "UPDATE recordings SET materialization='ABSENT', failure='interrupted',"
                    " updated=? WHERE identity_key=?", (_now(), row["identity_key"]),
                )
                self._log(row["identity_key"], "recover_stale_download", {})
                summary["stale_downloads"].append(row["file_id"])
            leases = self._connection.execute(
                "SELECT lease_id, identity_key FROM leases"
            ).fetchall()
            if leases:
                self._connection.execute("DELETE FROM leases")
                summary["orphan_leases"] = [row["lease_id"] for row in leases]
            ready = self._connection.execute(
                "SELECT identity_key, device_code, file_id, path FROM recordings"
                " WHERE materialization IN ('READY','LEASED','EVICTABLE') AND path IS NOT NULL"
            ).fetchall()
            for row in ready:
                if not Path(row["path"]).is_file():
                    self._connection.execute(
                        "UPDATE recordings SET materialization='ABSENT', path=NULL, bytes=0,"
                        " failure='missing_after_crash', updated=? WHERE identity_key=?",
                        (_now(), row["identity_key"]),
                    )
                    self._log(row["identity_key"], "recover_missing_file", {})
                    summary["missing_files"].append(row["file_id"])
                elif row["identity_key"] in leases:
                    continue
            for row in self._connection.execute(
                "SELECT identity_key FROM recordings WHERE materialization='LEASED'"
            ).fetchall():
                self._connection.execute(
                    "UPDATE recordings SET materialization='READY', updated=?"
                    " WHERE identity_key=?", (_now(), row["identity_key"]),
                )
        # 未被任何记录引用的 .part 属于失败残留，直接清理。
        referenced = {
            Path(row["path"]).name for row in self._connection.execute(
                "SELECT path FROM recordings WHERE path IS NOT NULL"
            ).fetchall()
        }
        for part in self.raw_dir.glob("*.part"):
            if part.name.removesuffix(".part") not in referenced:
                part.unlink(missing_ok=True)
        return summary

    def needs_repull(self, device_code: str, file_id: str, *, stage: str) -> bool:
        """preview 完成后仍需要高清的阶段必须按稳定身份重拉。"""
        entry = self.entry(device_code, file_id)
        if entry is None:
            return True
        if entry.materialization in ("ABSENT", "EVICTABLE", "DOWNLOADING"):
            return True
        if self.stage_committed(device_code, file_id, stage):
            return False
        return True

    def report(self) -> dict[str, Any]:
        entries = self.entries()
        budget = self.budget_report()
        by_state: dict[str, int] = {}
        for entry in entries:
            by_state[entry.materialization] = by_state.get(entry.materialization, 0) + 1
        return {
            "files": len(entries),
            "by_state": by_state,
            "total_downloaded_bytes": sum(entry.downloaded_bytes for entry in entries),
            "total_download_seconds": round(
                sum(entry.download_seconds for entry in entries), 3
            ),
            "peak_raw_bytes": self.peak_raw_bytes,
            "peak_work_bytes": self.peak_work_bytes,
            "budget": budget.as_dict(),
            "failures": [
                {"file_id": entry.file.file_id, "failure": entry.failure}
                for entry in entries if entry.failure
            ],
        }


__all__ = [
    "BudgetReport", "CacheError", "Lease", "LeaseError", "MATERIALIZATION_STATES",
    "ManagedRecordingCache", "RecordingEntry", "STAGE_NAMES", "TERMINAL_STAGES",
    "hash_stage_key",
]
