from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

from .event_engine import NormalizedRect
from .vessel_detection import (
    EvidenceValidationResult,
    SMALL_TARGET_PROPOSAL_CLASS_ID,
    VesselDetection,
    VesselSnapshot,
)


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")
_ACTIVE_JOB_STATES = ("claimed", "running", "capturing", "returning_home")
_DEFAULT_ZOOM_GAIN_PER_DELTA = math.log(2.0) / 2.0
_ZOOM_GAIN_ALPHA = 0.25
_MIN_ZOOM_GAIN_PER_DELTA = 0.02
_MAX_ZOOM_GAIN_PER_DELTA = 1.0
_REACQUIRE_AMBIGUITY_MARGIN = 0.03
_MAX_STALLED_ZOOM_ROUNDS = 2
_STRICT_REACQUIRE_FRACTION = 0.60


@dataclass(frozen=True, slots=True)
class PtzVerificationOptions:
    """Optional close-up verification driven by high-recall vessel candidates."""

    enabled: bool = False
    camera_id: str = ""
    camera_control_url: str = "http://camera-control:8080"
    camera_control_key_env: str = "CAMERA_CONTROL_API_KEY"
    zoom_strategy: Literal["adaptive", "fixed"] = "adaptive"
    zoom_steps: tuple[int, ...] = (4, 4)
    adaptive_target_width_ratio: float = 0.25
    adaptive_target_height_ratio: float = 0.18
    adaptive_min_step: int = 1
    adaptive_max_step: int = 6
    adaptive_max_rounds: int = 3
    adaptive_max_total_zoom_delta: int = 12
    adaptive_min_scale_growth_ratio: float = 1.12
    confirmed_target_fallback_zoom_rounds: int = 1
    confirmed_target_fallback_zoom_step: int = 3
    command_timeout_seconds: float = 12.0
    reacquire_timeout_seconds: float = 4.0
    settle_seconds: float = 0.5
    maximum_off_home_seconds: float = 25.0
    capture_quality: int = 1
    evidence_validation_required: bool = True
    evidence_capture_attempts: int = 2
    evidence_minimum_sharpness: float = 12.0
    evidence_target_scale_ratio: float = 0.70
    monitoring_interval_seconds: float = 0.25
    home_frame_delay_seconds: float = 1.5
    home_stable_frames: int = 2
    minimum_target_observations: int = 3
    proposal_merge_radius: float = 0.04
    proposal_minimum_interval_seconds: float = 30.0
    proposal_maximum_verifications_per_hour: int = 12
    recent_target_seconds: float = 1_200.0
    confirmed_cooldown_seconds: float = 1_200.0
    negative_cooldown_seconds: float = 3_600.0
    lost_retry_seconds: float = 180.0
    dedup_base_radius: float = 0.018
    dedup_uncertainty_per_second: float = 0.0015
    dedup_maximum_radius: float = 0.08
    reacquire_strict_center_radius: float = 0.22
    reacquire_center_radius: float = 0.45
    reacquire_cluster_radius: float = 0.18

    @property
    def shutdown_timeout_seconds(self) -> float:
        """Time reserved for an active verification to return HOME and finish."""
        return self.maximum_off_home_seconds + 2.0

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.camera_id or not _SAFE_IDENTIFIER.fullmatch(self.camera_id):
            raise ValueError("ptz_verification.camera_id格式无效")
        parsed = urlsplit(self.camera_control_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("ptz_verification.camera_control_url必须是HTTP(S)地址")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("camera_control_url中不能包含账号或密钥")
        if not _SAFE_IDENTIFIER.fullmatch(self.camera_control_key_env):
            raise ValueError("camera_control_key_env格式无效")
        if self.zoom_strategy not in {"adaptive", "fixed"}:
            raise ValueError("zoom_strategy必须是adaptive或fixed")
        if not self.zoom_steps or len(self.zoom_steps) > 3:
            raise ValueError("ptz_verification.zoom_steps必须包含1到3步")
        if any(step < 1 or step > 16 for step in self.zoom_steps):
            raise ValueError("ptz_verification.zoom_steps每步必须在[1,16]")
        if not 0.03 <= self.adaptive_target_width_ratio <= 0.8:
            raise ValueError("adaptive_target_width_ratio必须在[0.03,0.8]")
        if not 0.03 <= self.adaptive_target_height_ratio <= 0.8:
            raise ValueError("adaptive_target_height_ratio必须在[0.03,0.8]")
        if not 1 <= self.adaptive_min_step <= self.adaptive_max_step <= 16:
            raise ValueError("adaptive变焦步长必须满足1<=min<=max<=16")
        if not 1 <= self.adaptive_max_rounds <= 6:
            raise ValueError("adaptive_max_rounds必须在[1,6]")
        if not 1 <= self.adaptive_max_total_zoom_delta <= 48:
            raise ValueError("adaptive_max_total_zoom_delta必须在[1,48]")
        if not 1.0 <= self.adaptive_min_scale_growth_ratio <= 3.0:
            raise ValueError("adaptive_min_scale_growth_ratio必须在[1,3]")
        if not 0 <= self.confirmed_target_fallback_zoom_rounds <= 2:
            raise ValueError(
                "confirmed_target_fallback_zoom_rounds必须在[0,2]"
            )
        if not 1 <= self.confirmed_target_fallback_zoom_step <= 8:
            raise ValueError(
                "confirmed_target_fallback_zoom_step必须在[1,8]"
            )
        if not 1 <= self.capture_quality <= 6:
            raise ValueError("ptz_verification.capture_quality必须在[1,6]")
        if not 1 <= self.evidence_capture_attempts <= 3:
            raise ValueError("evidence_capture_attempts必须在[1,3]")
        if not 0 <= self.evidence_minimum_sharpness <= 10_000:
            raise ValueError("evidence_minimum_sharpness必须在[0,10000]")
        if not 0.30 <= self.evidence_target_scale_ratio <= 1.0:
            raise ValueError("evidence_target_scale_ratio必须在[0.30,1.0]")
        if not 2 <= self.command_timeout_seconds <= 60:
            raise ValueError("command_timeout_seconds必须在[2,60]")
        if not 1 <= self.reacquire_timeout_seconds <= 30:
            raise ValueError("reacquire_timeout_seconds必须在[1,30]")
        if not 0 <= self.settle_seconds <= 5:
            raise ValueError("settle_seconds必须在[0,5]")
        if not 5 <= self.maximum_off_home_seconds <= 120:
            raise ValueError("maximum_off_home_seconds必须在[5,120]")
        if not 0.05 <= self.monitoring_interval_seconds <= 5:
            raise ValueError("monitoring_interval_seconds必须在[0.05,5]")
        if not 0 <= self.home_frame_delay_seconds <= 10:
            raise ValueError("home_frame_delay_seconds必须在[0,10]")
        if not 1 <= self.home_stable_frames <= 10:
            raise ValueError("home_stable_frames必须在[1,10]")
        if not 1 <= self.minimum_target_observations <= 10:
            raise ValueError("minimum_target_observations必须在[1,10]")
        if not 0 <= self.proposal_merge_radius <= 0.10:
            raise ValueError("proposal_merge_radius必须在[0,0.10]")
        if not 0 <= self.proposal_minimum_interval_seconds <= 3_600:
            raise ValueError("proposal_minimum_interval_seconds必须在[0,3600]")
        if not 1 <= self.proposal_maximum_verifications_per_hour <= 3_600:
            raise ValueError(
                "proposal_maximum_verifications_per_hour必须在[1,3600]"
            )
        if min(
            self.recent_target_seconds,
            self.confirmed_cooldown_seconds,
            self.negative_cooldown_seconds,
            self.lost_retry_seconds,
        ) <= 0:
            raise ValueError("PTZ验证记忆和冷却时间必须大于0")
        if not 0 < self.dedup_base_radius <= self.dedup_maximum_radius <= 0.5:
            raise ValueError("PTZ验证去重半径无效")
        if not 0 <= self.dedup_uncertainty_per_second <= 0.05:
            raise ValueError("dedup_uncertainty_per_second无效")
        if not (
            0.05
            <= self.reacquire_strict_center_radius
            <= self.reacquire_center_radius
            <= 0.75
        ):
            raise ValueError(
                "重捕获半径必须满足0.05<=strict<=maximum<=0.75"
            )
        if not 0.02 <= self.reacquire_cluster_radius <= 0.30:
            raise ValueError("reacquire_cluster_radius必须在[0.02,0.30]")

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "camera_id": self.camera_id,
            "camera_control_url": self.camera_control_url,
            "camera_control_key_env": self.camera_control_key_env,
            "zoom_strategy": self.zoom_strategy,
            "zoom_steps": list(self.zoom_steps),
            "adaptive_target_width_ratio": self.adaptive_target_width_ratio,
            "adaptive_target_height_ratio": self.adaptive_target_height_ratio,
            "adaptive_min_step": self.adaptive_min_step,
            "adaptive_max_step": self.adaptive_max_step,
            "adaptive_max_rounds": self.adaptive_max_rounds,
            "adaptive_max_total_zoom_delta": (
                self.adaptive_max_total_zoom_delta
            ),
            "adaptive_min_scale_growth_ratio": (
                self.adaptive_min_scale_growth_ratio
            ),
            "confirmed_target_fallback_zoom_rounds": (
                self.confirmed_target_fallback_zoom_rounds
            ),
            "confirmed_target_fallback_zoom_step": (
                self.confirmed_target_fallback_zoom_step
            ),
            "command_timeout_seconds": self.command_timeout_seconds,
            "reacquire_timeout_seconds": self.reacquire_timeout_seconds,
            "settle_seconds": self.settle_seconds,
            "maximum_off_home_seconds": self.maximum_off_home_seconds,
            "capture_quality": self.capture_quality,
            "evidence_validation_required": (
                self.evidence_validation_required
            ),
            "evidence_capture_attempts": self.evidence_capture_attempts,
            "evidence_minimum_sharpness": self.evidence_minimum_sharpness,
            "evidence_target_scale_ratio": self.evidence_target_scale_ratio,
            "monitoring_interval_seconds": self.monitoring_interval_seconds,
            "home_frame_delay_seconds": self.home_frame_delay_seconds,
            "home_stable_frames": self.home_stable_frames,
            "minimum_target_observations": self.minimum_target_observations,
            "proposal_merge_radius": self.proposal_merge_radius,
            "proposal_minimum_interval_seconds": (
                self.proposal_minimum_interval_seconds
            ),
            "proposal_maximum_verifications_per_hour": (
                self.proposal_maximum_verifications_per_hour
            ),
            "recent_target_seconds": self.recent_target_seconds,
            "confirmed_cooldown_seconds": self.confirmed_cooldown_seconds,
            "negative_cooldown_seconds": self.negative_cooldown_seconds,
            "lost_retry_seconds": self.lost_retry_seconds,
            "dedup_base_radius": self.dedup_base_radius,
            "dedup_uncertainty_per_second": (
                self.dedup_uncertainty_per_second
            ),
            "dedup_maximum_radius": self.dedup_maximum_radius,
            "reacquire_strict_center_radius": (
                self.reacquire_strict_center_radius
            ),
            "reacquire_center_radius": self.reacquire_center_radius,
            "reacquire_cluster_radius": self.reacquire_cluster_radius,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any] | None,
    ) -> "PtzVerificationOptions":
        values = dict(payload or {})
        if "zoom_steps" in values:
            values["zoom_steps"] = tuple(int(item) for item in values["zoom_steps"])
        options = cls(**values)
        options.validate()
        return options


@dataclass(frozen=True, slots=True)
class ObservedTarget:
    target_id: str
    source_track_id: int
    x: float
    y: float
    width: float
    height: float
    cooldown_until: float
    last_result: str | None
    observations: int


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    job_id: str
    sequence: int
    target: ObservedTarget


@dataclass(frozen=True, slots=True)
class _TargetSelection:
    detection: VesselDetection | None = None
    saw_candidates: bool = False
    competing_groups: bool = False


class PtzVerificationRepository:
    """SQLite/WAL target memory and evidence index shared by worker and API."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.image_root = self.root / "images"
        self.database_path = self.root / "ptz-verification.sqlite3"
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS targets (
                    target_id TEXT PRIMARY KEY,
                    stream_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    source_track_id INTEGER NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    width REAL NOT NULL,
                    height REAL NOT NULL,
                    vx REAL NOT NULL DEFAULT 0,
                    vy REAL NOT NULL DEFAULT 0,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    cooldown_until REAL NOT NULL DEFAULT 0,
                    last_result TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    observations INTEGER NOT NULL DEFAULT 1,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_targets_recent
                    ON targets(stream_id, camera_id, last_seen DESC);
                CREATE TABLE IF NOT EXISTS jobs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL UNIQUE,
                    target_id TEXT NOT NULL REFERENCES targets(target_id),
                    stream_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    source_track_id INTEGER NOT NULL,
                    trigger_x REAL NOT NULL,
                    trigger_y REAL NOT NULL,
                    state TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    home_returned INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_list
                    ON jobs(sequence DESC, stream_id, result);
                CREATE TABLE IF NOT EXISTS evidence_images (
                    image_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    kind TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    captured_at REAL NOT NULL,
                    is_derived INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_job
                    ON evidence_images(job_id, captured_at);
                CREATE TABLE IF NOT EXISTS completion_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
                    completed_at REAL NOT NULL
                );
                """
            )
            target_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(targets)")
            }
            if "observations" not in target_columns:
                connection.execute(
                    "ALTER TABLE targets ADD COLUMN observations "
                    "INTEGER NOT NULL DEFAULT 1"
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO completion_events(job_id,completed_at)
                SELECT job_id,COALESCE(finished_at,created_at) FROM jobs
                WHERE state='completed' ORDER BY sequence
                """
            )

    def observe(
        self,
        *,
        stream_id: str,
        camera_id: str,
        detection: VesselDetection,
        now: float,
        options: PtzVerificationOptions,
        exclude_target_ids: set[str] | None = None,
    ) -> ObservedTarget:
        rectangle = detection.rectangle
        x = rectangle.left + rectangle.width / 2.0
        y = rectangle.top + rectangle.height / 2.0
        cutoff = now - options.recent_target_seconds
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM targets
                WHERE stream_id=? AND camera_id=? AND last_seen>=?
                ORDER BY last_seen DESC
                LIMIT 500
                """,
                (stream_id, camera_id, cutoff),
            ).fetchall()
            # A completed review is a fixed spatial memory, not an ordinary
            # tracker state.  During the cooldown, never extrapolate its old
            # velocity or let nearby proposal fragments drag the anchor
            # across the scene.  This also intentionally ignores
            # ``exclude_target_ids`` so several fragments in one snapshot are
            # all suppressed by the same reviewed location.
            reviewed_match: sqlite3.Row | None = None
            reviewed_distance = math.inf
            for row in rows:
                if float(row["cooldown_until"]) <= now:
                    continue
                distance = math.hypot(x - float(row["x"]), y - float(row["y"]))
                size_allowance = min(
                    0.5
                    * max(
                        rectangle.width,
                        rectangle.height,
                        float(row["width"]),
                        float(row["height"]),
                    ),
                    options.proposal_merge_radius,
                )
                radius = min(
                    max(options.proposal_merge_radius, size_allowance),
                    options.dedup_maximum_radius,
                )
                if distance <= radius and distance < reviewed_distance:
                    reviewed_match = row
                    reviewed_distance = distance

            if reviewed_match is not None:
                target_id = str(reviewed_match["target_id"])
                connection.execute(
                    """
                    UPDATE targets SET source_track_id=?,last_seen=?,updated_at=?,
                        observations=observations+1 WHERE target_id=?
                    """,
                    (detection.object_id, now, now, target_id),
                )
                return ObservedTarget(
                    target_id=target_id,
                    source_track_id=detection.object_id,
                    x=float(reviewed_match["x"]),
                    y=float(reviewed_match["y"]),
                    width=float(reviewed_match["width"]),
                    height=float(reviewed_match["height"]),
                    cooldown_until=float(reviewed_match["cooldown_until"]),
                    last_result=(
                        str(reviewed_match["last_result"])
                        if reviewed_match["last_result"] is not None
                        else None
                    ),
                    observations=int(reviewed_match["observations"]) + 1,
                )

            match: sqlite3.Row | None = None
            match_distance = math.inf
            for row in rows:
                if float(row["cooldown_until"]) > now:
                    continue
                if (
                    exclude_target_ids is not None
                    and str(row["target_id"]) in exclude_target_ids
                ):
                    continue
                elapsed = max(now - float(row["last_seen"]), 0.0)
                predicted_x = float(row["x"]) + float(row["vx"]) * elapsed
                predicted_y = float(row["y"]) + float(row["vy"]) * elapsed
                distance = math.hypot(x - predicted_x, y - predicted_y)
                uncertainty = min(
                    options.dedup_uncertainty_per_second * elapsed,
                    options.dedup_maximum_radius - options.dedup_base_radius,
                )
                size_allowance = min(
                    0.5
                    * max(
                        rectangle.width,
                        rectangle.height,
                        float(row["width"]),
                        float(row["height"]),
                    ),
                    options.dedup_maximum_radius / 2.0,
                )
                radius = min(
                    options.dedup_base_radius + uncertainty + size_allowance,
                    options.dedup_maximum_radius,
                )
                same_source = int(row["source_track_id"]) == detection.object_id
                if same_source:
                    radius = min(radius * 1.5, options.dedup_maximum_radius)
                if distance <= radius and distance < match_distance:
                    match = row
                    match_distance = distance

            if match is None:
                target_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO targets(
                        target_id,stream_id,camera_id,source_track_id,
                        x,y,width,height,first_seen,last_seen,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        target_id,
                        stream_id,
                        camera_id,
                        detection.object_id,
                        x,
                        y,
                        rectangle.width,
                        rectangle.height,
                        now,
                        now,
                        now,
                    ),
                )
                cooldown_until = 0.0
                last_result = None
                observations = 1
            else:
                target_id = str(match["target_id"])
                elapsed = max(now - float(match["last_seen"]), 0.0)
                vx = float(match["vx"])
                vy = float(match["vy"])
                if elapsed >= 0.05:
                    instantaneous_vx = (x - float(match["x"])) / elapsed
                    instantaneous_vy = (y - float(match["y"])) / elapsed
                    if int(match["observations"]) <= 1:
                        vx = instantaneous_vx
                        vy = instantaneous_vy
                    else:
                        vx = vx * 0.7 + instantaneous_vx * 0.3
                        vy = vy * 0.7 + instantaneous_vy * 0.3
                connection.execute(
                    """
                    UPDATE targets SET
                        source_track_id=?,x=?,y=?,width=?,height=?,
                        vx=?,vy=?,last_seen=?,updated_at=?,
                        observations=observations+1
                    WHERE target_id=?
                    """,
                    (
                        detection.object_id,
                        x,
                        y,
                        rectangle.width,
                        rectangle.height,
                        vx,
                        vy,
                        now,
                        now,
                        target_id,
                    ),
                )
                cooldown_until = float(match["cooldown_until"])
                last_result = match["last_result"]
                observations = int(match["observations"]) + 1

        return ObservedTarget(
            target_id=target_id,
            source_track_id=detection.object_id,
            x=x,
            y=y,
            width=rectangle.width,
            height=rectangle.height,
            cooldown_until=cooldown_until,
            last_result=str(last_result) if last_result is not None else None,
            observations=observations,
        )

    def claim(
        self,
        *,
        stream_id: str,
        camera_id: str,
        target: ObservedTarget,
        now: float,
    ) -> ClaimedJob | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT cooldown_until FROM targets WHERE target_id=?",
                (target.target_id,),
            ).fetchone()
            if row is None or float(row["cooldown_until"]) > now:
                connection.rollback()
                return None
            placeholders = ",".join("?" for _ in _ACTIVE_JOB_STATES)
            active = connection.execute(
                f"SELECT 1 FROM jobs WHERE target_id=? AND state IN ({placeholders}) LIMIT 1",
                (target.target_id, *_ACTIVE_JOB_STATES),
            ).fetchone()
            if active is not None:
                connection.rollback()
                return None
            job_id = uuid.uuid4().hex
            cursor = connection.execute(
                """
                INSERT INTO jobs(
                    job_id,target_id,stream_id,camera_id,source_track_id,
                    trigger_x,trigger_y,state,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    target.target_id,
                    stream_id,
                    camera_id,
                    target.source_track_id,
                    target.x,
                    target.y,
                    "claimed",
                    now,
                ),
            )
            connection.execute(
                "UPDATE targets SET attempts=attempts+1,updated_at=? WHERE target_id=?",
                (now, target.target_id),
            )
            connection.commit()
            return ClaimedJob(
                job_id=job_id,
                sequence=int(cursor.lastrowid),
                target=target,
            )
        finally:
            connection.close()

    def mark_running(self, job_id: str, now: float) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET state='running',started_at=? WHERE job_id=?",
                (now, job_id),
            )

    def recover_incomplete(
        self,
        *,
        stream_id: str,
        camera_id: str,
        now: float,
        options: PtzVerificationOptions,
    ) -> int:
        """Close jobs abandoned by a previous worker after HOME is restored."""
        placeholders = ",".join("?" for _ in _ACTIVE_JOB_STATES)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT job_id,target_id FROM jobs
                WHERE stream_id=? AND camera_id=?
                  AND state IN ({placeholders})
                """,
                (stream_id, camera_id, *_ACTIVE_JOB_STATES),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE jobs SET state='completed',result='target_lost',
                        error=?,home_returned=1,finished_at=?
                    WHERE job_id=?
                    """,
                    (
                        "worker restarted; monitoring preset restored",
                        now,
                        str(row["job_id"]),
                    ),
                )
                connection.execute(
                    """
                    UPDATE targets SET last_result='target_lost',
                        cooldown_until=?,updated_at=? WHERE target_id=?
                    """,
                    (
                        now + options.lost_retry_seconds,
                        now,
                        str(row["target_id"]),
                    ),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO completion_events(job_id,completed_at)
                    VALUES(?,?)
                    """,
                    (str(row["job_id"]), now),
                )
            return len(rows)

    def store_evidence(
        self,
        *,
        job_id: str,
        content: bytes,
        mime_type: str,
        captured_at: float,
        kind: str = "closeup_raw",
        is_derived: bool = False,
    ) -> str:
        if not content:
            raise ValueError("evidence image is empty")
        image_id = uuid.uuid4().hex
        suffix = ".jpg" if mime_type == "image/jpeg" else ".bin"
        relative_path = Path(job_id[:2]) / job_id / f"{image_id}{suffix}"
        path = self.image_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        digest = hashlib.sha256(content).hexdigest()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evidence_images(
                    image_id,job_id,kind,relative_path,mime_type,size_bytes,
                    sha256,captured_at,is_derived
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    image_id,
                    job_id,
                    kind,
                    str(relative_path),
                    mime_type,
                    len(content),
                    digest,
                    captured_at,
                    int(is_derived),
                ),
            )
        return image_id

    def finish(
        self,
        *,
        job_id: str,
        result: str,
        error: str | None,
        home_returned: bool,
        now: float,
        options: PtzVerificationOptions,
    ) -> None:
        if result == "boat_confirmed":
            cooldown = options.confirmed_cooldown_seconds
        elif result == "candidate_not_confirmed":
            cooldown = options.negative_cooldown_seconds
        elif result == "recovery_failed":
            cooldown = max(options.negative_cooldown_seconds, 3_600.0)
        else:
            cooldown = options.lost_retry_seconds
        with self._connect() as connection:
            row = connection.execute(
                "SELECT target_id,trigger_x,trigger_y FROM jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            connection.execute(
                """
                UPDATE jobs SET state='completed',result=?,error=?,
                    home_returned=?,finished_at=? WHERE job_id=?
                """,
                (result, error, int(home_returned), now, job_id),
            )
            connection.execute(
                """
                UPDATE targets SET last_result=?,cooldown_until=?,updated_at=?,
                    x=?,y=?,vx=0,vy=0
                WHERE target_id=?
                """,
                (
                    result,
                    now + cooldown,
                    now,
                    float(row["trigger_x"]),
                    float(row["trigger_y"]),
                    str(row["target_id"]),
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO completion_events(job_id,completed_at)
                VALUES(?,?)
                """,
                (job_id, now),
            )

    def list_jobs(
        self,
        *,
        stream_id: str | None = None,
        result: str | None = None,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit必须在[1,1000]")
        clauses = ["e.sequence>?"]
        values: list[Any] = [after_sequence]
        if stream_id is not None:
            clauses.append("j.stream_id=?")
            values.append(stream_id)
        if result is not None:
            clauses.append("j.result=?")
            values.append(result)
        values.append(limit)
        query = f"""
            SELECT j.*,e.sequence AS completion_sequence FROM completion_events e
            JOIN jobs j ON j.job_id=e.job_id
            WHERE {' AND '.join(clauses)}
            ORDER BY e.sequence ASC LIMIT ?
        """
        with self._connect() as connection:
            jobs = [dict(row) for row in connection.execute(query, values)]
            for job in jobs:
                job["creation_sequence"] = int(job["sequence"])
                job["sequence"] = int(job.pop("completion_sequence"))
                job["home_returned"] = bool(job["home_returned"])
                images = connection.execute(
                    """
                    SELECT image_id,kind,mime_type,size_bytes,sha256,
                           captured_at,is_derived
                    FROM evidence_images WHERE job_id=? ORDER BY captured_at
                    """,
                    (job["job_id"],),
                ).fetchall()
                job["images"] = [
                    {
                        **dict(image),
                        "is_derived": bool(image["is_derived"]),
                        "download_url": (
                            "/v1/vessel-verifications/"
                            f"{job['job_id']}/images/{image['image_id']}"
                        ),
                    }
                    for image in images
                ]
            return jobs

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT e.sequence FROM completion_events e
                JOIN jobs j ON j.job_id=e.job_id WHERE j.job_id=?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        jobs = self.list_jobs(after_sequence=int(row["sequence"]) - 1, limit=1)
        if not jobs or jobs[0]["job_id"] != job_id:
            raise KeyError(job_id)
        return jobs[0]

    def media_path(self, job_id: str, image_id: str) -> tuple[Path, str]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT relative_path,mime_type FROM evidence_images
                WHERE image_id=? AND job_id=?
                """,
                (image_id, job_id),
            ).fetchone()
        if row is None:
            raise KeyError(image_id)
        path = (self.image_root / str(row["relative_path"])).resolve()
        if self.image_root not in path.parents or not path.is_file():
            raise KeyError(image_id)
        return path, str(row["mime_type"])


class CameraControlClient:
    """Small authenticated HTTP client; secrets never enter stream payloads."""

    def __init__(
        self,
        options: PtzVerificationOptions,
        *,
        api_key: str | None = None,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        options.validate()
        self.options = options
        self.base_url = options.camera_control_url.rstrip("/") + "/"
        self.api_key = api_key or os.getenv(options.camera_control_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"环境变量{options.camera_control_key_env}未配置"
            )
        self._opener = opener
        self._lease_token: str | None = None

    def _json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            urljoin(self.base_url, path.lstrip("/")),
            data=body,
            method=method,
            headers={
                "X-Camera-Control-Key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                **(
                    {"X-Camera-Control-Lease": self._lease_token}
                    if self._lease_token is not None
                    else {}
                ),
            },
        )
        try:
            with self._opener(
                request,
                timeout=timeout or self.options.command_timeout_seconds,
            ) as response:
                content = response.read()
                if not content:
                    return {}
                return json.loads(content.decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"camera-control HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError("camera-control连接失败") from exc

    def _submit_and_wait(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        submitted = self._json_request("POST", path, payload)
        command_id = str(submitted.get("command_id", ""))
        if not command_id:
            raise RuntimeError("camera-control未返回command_id")
        deadline = time.monotonic() + self.options.command_timeout_seconds
        while True:
            current = self._json_request(
                "GET",
                f"/v1/commands/{command_id}",
                timeout=min(self.options.command_timeout_seconds, 5.0),
            )
            state = current.get("state")
            if state == "completed":
                return current
            if state == "failed":
                raise RuntimeError(str(current.get("error") or "camera command failed"))
            if time.monotonic() >= deadline:
                raise RuntimeError("camera-control命令轮询超时")
            time.sleep(0.1)

    def locate(self, x: float, y: float, zoom_delta: int) -> None:
        self._submit_and_wait(
            f"/v1/cameras/{self.options.camera_id}/commands/locate",
            {
                "x": x,
                "y": y,
                "zoom_delta": zoom_delta,
                "timeout_seconds": self.options.command_timeout_seconds,
                "autofocus": False,
            },
        )

    def acquire_lease(self, owner: str, ttl_seconds: float) -> None:
        lease = self._json_request(
            "POST",
            f"/v1/cameras/{self.options.camera_id}/lease",
            {"owner": owner, "ttl_seconds": ttl_seconds},
        )
        token = str(lease.get("token", ""))
        if not token:
            raise RuntimeError("camera-control未返回摄像头租约令牌")
        self._lease_token = token

    def release_lease(self) -> None:
        if self._lease_token is None:
            return
        try:
            self._json_request(
                "DELETE",
                f"/v1/cameras/{self.options.camera_id}/lease",
            )
        finally:
            self._lease_token = None

    def autofocus(self) -> None:
        self._submit_and_wait(
            f"/v1/cameras/{self.options.camera_id}/commands/autofocus",
            {"timeout_seconds": self.options.command_timeout_seconds},
        )

    def home(self) -> None:
        self._submit_and_wait(
            f"/v1/cameras/{self.options.camera_id}/commands/home",
            {"timeout_seconds": self.options.command_timeout_seconds},
        )

    def stop(self) -> None:
        self._json_request(
            "POST",
            f"/v1/cameras/{self.options.camera_id}/stop",
            {},
        )

    def capture(self) -> tuple[bytes, str, float]:
        command = self._submit_and_wait(
            f"/v1/cameras/{self.options.camera_id}/commands/capture",
            {
                "quality": self.options.capture_quality,
                "timeout_seconds": self.options.command_timeout_seconds,
            },
        )
        result = command.get("result") or {}
        download_url = str(result.get("download_url", ""))
        if not download_url:
            raise RuntimeError("camera-control抓图结果没有download_url")
        artifact_url = urljoin(self.base_url, download_url)
        base = urlsplit(self.base_url)
        artifact = urlsplit(artifact_url)
        if (artifact.scheme, artifact.netloc) != (base.scheme, base.netloc):
            raise RuntimeError("camera-control返回了不受信任的图片地址")
        request = Request(
            artifact_url,
            method="GET",
            headers={"X-Camera-Control-Key": self.api_key},
        )
        try:
            with self._opener(
                request,
                timeout=self.options.command_timeout_seconds,
            ) as response:
                content = response.read()
                mime_type = response.headers.get_content_type()
        except (HTTPError, URLError) as exc:
            raise RuntimeError("camera-control图片下载失败") from exc
        if not content:
            raise RuntimeError("camera-control返回空图片")
        return content, mime_type, time.time()


def _merge_proposal_fragments(
    detections: tuple[VesselDetection, ...],
    radius: float,
) -> tuple[VesselDetection, ...]:
    """Collapse nearby unclassified fragments without merging real boats."""
    confirmed = [
        item
        for item in detections
        if item.class_id != SMALL_TARGET_PROPOSAL_CLASS_ID
    ]
    proposals = sorted(
        (
            item
            for item in detections
            if item.class_id == SMALL_TARGET_PROPOSAL_CLASS_ID
        ),
        key=lambda item: (
            item.hits,
            item.rectangle.width * item.rectangle.height,
            item.confidence,
        ),
        reverse=True,
    )
    selected: list[VesselDetection] = list(confirmed)
    selected_proposals: list[VesselDetection] = []
    for proposal in proposals:
        x, y = proposal.rectangle.center
        if any(
            math.hypot(x - current.rectangle.center[0], y - current.rectangle.center[1])
            <= radius
            for current in (*confirmed, *selected_proposals)
        ):
            continue
        selected_proposals.append(proposal)
    selected.extend(selected_proposals)
    return tuple(selected)


class PtzVerificationCoordinator:
    """One serial verifier per camera; never called from the video callback."""

    def __init__(
        self,
        *,
        stream_id: str,
        options: PtzVerificationOptions,
        snapshot_provider: Callable[[], VesselSnapshot],
        repository: PtzVerificationRepository,
        camera_client: CameraControlClient,
        evidence_validator: Callable[
            [bytes, float], EvidenceValidationResult
        ]
        | None = None,
    ) -> None:
        options.validate()
        if options.evidence_validation_required and evidence_validator is None:
            raise ValueError("启用PTZ证据保存时必须配置证据图片复检器")
        self.stream_id = stream_id
        self.options = options
        self._snapshot_provider = snapshot_provider
        self._repository = repository
        self._camera = camera_client
        self._evidence_validator = evidence_validator
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_monitoring_version = 0
        self._monitoring_not_before = 0.0
        self._recovery_required = False
        self._verifying = threading.Event()
        self._proposal_verification_times: deque[float] = deque()
        self._zoom_gain_per_delta = _DEFAULT_ZOOM_GAIN_PER_DELTA
        self._view_generation = 0
        self._lease_owner = f"rtsp:{stream_id}:{uuid.uuid4().hex}"
        self._lease_ttl_seconds = min(
            max(
                options.maximum_off_home_seconds
                + options.command_timeout_seconds * 2.0
                + 15.0,
                60.0,
            ),
            600.0,
        )
        self._lease_renew_after = 0.0
        self.last_error: str | None = None

    def _proposal_budget_available(self, now: float) -> bool:
        cutoff = now - 3_600.0
        while (
            self._proposal_verification_times
            and self._proposal_verification_times[0] <= cutoff
        ):
            self._proposal_verification_times.popleft()
        if (
            self._proposal_verification_times
            and now - self._proposal_verification_times[-1]
            < self.options.proposal_minimum_interval_seconds
        ):
            return False
        return (
            len(self._proposal_verification_times)
            < self.options.proposal_maximum_verifications_per_hour
        )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"ptz-verification-{self.options.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        self.request_shutdown()
        if self._verifying.is_set():
            try:
                self._camera.stop()
            except Exception:
                # The in-flight command and the mandatory HOME attempt below
                # remain the source of truth for the final job result.
                pass
        if self._thread is not None:
            self._thread.join(timeout=max(timeout, 0.0))

    def request_shutdown(self) -> None:
        """Prevent new PTZ work without blocking the worker signal handler."""
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._renew_lease(force=True)
                break
            except Exception as exc:
                self.last_error = (
                    "等待摄像头控制租约: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._stop_event.wait(2.0)
        if self._stop_event.is_set():
            return
        self._verifying.set()
        try:
            before_home_version = self._snapshot_provider().result_version
            home_started_at = time.monotonic()
            self._advance_view_generation()
            self._camera.home()
            self._monitoring_not_before = (
                home_started_at + self.options.home_frame_delay_seconds
            )
            # The coordinator starts before the DeepStream pipeline. Physical
            # HOME completion is verified by camera_control, and run_once()
            # already rejects this pre-HOME version, so startup must not wait
            # for frames that cannot exist until pipeline.start().
            self._last_monitoring_version = max(
                self._last_monitoring_version,
                before_home_version,
            )
            self._repository.recover_incomplete(
                stream_id=self.stream_id,
                camera_id=self.options.camera_id,
                now=time.time(),
                options=self.options,
            )
        except Exception as exc:
            self._recovery_required = True
            self.last_error = f"启动回全景失败: {type(exc).__name__}: {exc}"
            try:
                self._camera.release_lease()
            except Exception:
                pass
            return
        self._verifying.clear()
        while not self._stop_event.is_set():
            try:
                self._renew_lease()
                self.run_once()
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            self._stop_event.wait(self.options.monitoring_interval_seconds)
        try:
            self._camera.release_lease()
        except Exception as exc:
            self.last_error = f"释放摄像头控制租约失败: {exc}"

    def _renew_lease(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now < self._lease_renew_after:
            return
        self._camera.acquire_lease(
            self._lease_owner,
            self._lease_ttl_seconds,
        )
        self._lease_renew_after = now + self._lease_ttl_seconds / 3.0

    def _advance_view_generation(self) -> int:
        self._view_generation += 1
        return self._view_generation

    @property
    def view_generation(self) -> int:
        return self._view_generation

    @property
    def state(self) -> str:
        if self._recovery_required:
            return "recovery_required"
        if self._verifying.is_set():
            return "verifying"
        if self.last_error:
            return "degraded"
        if self._thread is not None and self._thread.is_alive():
            return "running"
        return "stopped"

    @property
    def is_busy(self) -> bool:
        """Whether the camera view may differ from its monitoring preset."""
        return self._verifying.is_set() or self._recovery_required

    def run_once(self) -> str | None:
        if self._recovery_required:
            return None
        snapshot = self._snapshot_provider()
        if snapshot.result_version <= self._last_monitoring_version:
            return None
        self._last_monitoring_version = snapshot.result_version
        if (
            snapshot.updated_at is None
            or snapshot.updated_at < self._monitoring_not_before
        ):
            return None
        if snapshot.state != "running" or not snapshot.detections:
            return None
        now = time.time()
        candidates: list[tuple[VesselDetection, ObservedTarget]] = []
        observed_target_ids: set[str] = set()
        for detection in _merge_proposal_fragments(
            snapshot.detections,
            self.options.proposal_merge_radius,
        ):
            if (
                detection.class_id == SMALL_TARGET_PROPOSAL_CLASS_ID
                and not self._proposal_budget_available(now)
            ):
                continue
            target = self._repository.observe(
                stream_id=self.stream_id,
                camera_id=self.options.camera_id,
                detection=detection,
                now=now,
                options=self.options,
                exclude_target_ids=observed_target_ids,
            )
            observed_target_ids.add(target.target_id)
            candidates.append((detection, target))
        candidates.sort(
            key=lambda item: (
                item[0].class_id != SMALL_TARGET_PROPOSAL_CLASS_ID,
                item[0].hits,
                item[0].rectangle.width * item[0].rectangle.height,
                item[0].confidence,
            ),
            reverse=True,
        )
        for detection, target in candidates:
            if target.observations < self.options.minimum_target_observations:
                continue
            job = self._repository.claim(
                stream_id=self.stream_id,
                camera_id=self.options.camera_id,
                target=target,
                now=now,
            )
            if job is None:
                continue
            if detection.class_id == SMALL_TARGET_PROPOSAL_CLASS_ID:
                self._proposal_verification_times.append(now)
            self._execute(job, detection)
            return job.job_id
        return None

    def _adaptive_target_reached(self, detection: VesselDetection) -> bool:
        return (
            detection.rectangle.width
            >= self.options.adaptive_target_width_ratio
            or detection.rectangle.height
            >= self.options.adaptive_target_height_ratio
        )

    def _capture_target_centered(self, detection: VesselDetection) -> bool:
        x, y = detection.rectangle.center
        return math.hypot(x - 0.5, y - 0.5) <= min(
            self.options.reacquire_center_radius,
            0.15,
        )

    @staticmethod
    def _visual_scale(detection: VesselDetection) -> float:
        return math.sqrt(
            max(
                detection.rectangle.width * detection.rectangle.height,
                1e-9,
            )
        )

    def _adaptive_zoom_step(
        self,
        detection: VesselDetection,
        total_zoom_delta: int,
    ) -> int:
        remaining = (
            self.options.adaptive_max_total_zoom_delta - total_zoom_delta
        )
        if remaining <= 0:
            return 0
        size_score = max(
            detection.rectangle.width
            / self.options.adaptive_target_width_ratio,
            detection.rectangle.height
            / self.options.adaptive_target_height_ratio,
        )
        required_scale = max(1.0, 1.0 / max(size_score, 1e-6))
        estimated = int(
            math.ceil(
                math.log(required_scale)
                / max(self._zoom_gain_per_delta, _MIN_ZOOM_GAIN_PER_DELTA)
            )
        )
        step = max(self.options.adaptive_min_step, estimated)
        step = min(step, self.options.adaptive_max_step, remaining)
        return max(step, 0)

    def _execute(self, job: ClaimedJob, detection: VesselDetection) -> None:
        self._verifying.set()
        result = "target_lost"
        error: str | None = None
        departed_home = False
        home_returned = False
        point_x = detection.rectangle.left + detection.rectangle.width / 2.0
        point_y = detection.rectangle.top + detection.rectangle.height / 2.0
        started = time.monotonic()
        try:
            self._repository.mark_running(job.job_id, time.time())
            current = detection
            total_zoom_delta = 0
            fallback_zoom_rounds_used = 0
            stalled_rounds = 0
            maximum_rounds = (
                len(self.options.zoom_steps)
                if self.options.zoom_strategy == "fixed"
                else self.options.adaptive_max_rounds
            )
            for step_index in range(maximum_rounds):
                if self._stop_event.is_set():
                    raise InterruptedError(
                        "PTZ verification interrupted by shutdown"
                    )
                if (
                    time.monotonic() - started
                    >= self.options.maximum_off_home_seconds
                ):
                    raise RuntimeError("PTZ验证超过最大离开全景时间")
                if self.options.zoom_strategy == "fixed":
                    zoom_step = self.options.zoom_steps[step_index]
                    allow_proposal = step_index < maximum_rounds - 1
                else:
                    if (
                        current.class_id != SMALL_TARGET_PROPOSAL_CLASS_ID
                        and self._adaptive_target_reached(current)
                        and self._capture_target_centered(current)
                    ):
                        break
                    if (
                        current.class_id != SMALL_TARGET_PROPOSAL_CLASS_ID
                        and self._adaptive_target_reached(current)
                    ):
                        # The target is large enough but too close to an edge:
                        # recenter without consuming any optical zoom budget.
                        zoom_step = 0
                    else:
                        zoom_step = self._adaptive_zoom_step(
                            current,
                            total_zoom_delta,
                        )
                        if zoom_step <= 0:
                            break
                    after_step_total = total_zoom_delta + zoom_step
                    allow_proposal = (
                        step_index < maximum_rounds - 1
                        and after_step_total
                        < self.options.adaptive_max_total_zoom_delta
                    )
                before_version = self._snapshot_provider().result_version
                self._renew_lease(force=True)
                # A locate command can move the camera and then time out. Mark
                # the view unsafe before issuing it so the finally block always
                # attempts HOME even when the control request raises.
                departed_home = True
                self._advance_view_generation()
                self._camera.locate(point_x, point_y, zoom_step)
                total_zoom_delta += zoom_step
                minimum_updated_at = time.monotonic()
                if self.options.settle_seconds:
                    self._stop_event.wait(self.options.settle_seconds)
                reacquired = self._wait_for_centered_detection(
                    after_version=before_version,
                    minimum_updated_at=minimum_updated_at,
                    allow_proposal=allow_proposal,
                    reference=current,
                )
                closeup = reacquired.detection
                fallback_zoom_delta = 0
                if (
                    closeup is None
                    and not reacquired.saw_candidates
                    and self.options.zoom_strategy == "adaptive"
                    and current.class_id != SMALL_TARGET_PROPOSAL_CLASS_ID
                    and not self._adaptive_target_reached(current)
                    and fallback_zoom_rounds_used
                    < self.options.confirmed_target_fallback_zoom_rounds
                ):
                    closeup, fallback_zoom_delta, fallback_rounds = (
                        self._fallback_zoom_confirmed_target(
                            reference=current,
                            total_zoom_delta=total_zoom_delta,
                            started=started,
                            maximum_rounds=(
                                self.options.confirmed_target_fallback_zoom_rounds
                                - fallback_zoom_rounds_used
                            ),
                        )
                    )
                    total_zoom_delta += fallback_zoom_delta
                    fallback_zoom_rounds_used += fallback_rounds
                if closeup is None:
                    if self._stop_event.is_set():
                        raise InterruptedError(
                            "PTZ verification interrupted by shutdown"
                        )
                    if current.class_id == SMALL_TARGET_PROPOSAL_CLASS_ID:
                        result = "candidate_not_confirmed"
                    elif reacquired.competing_groups:
                        result = "target_ambiguous"
                    else:
                        result = "target_lost"
                    return
                previous = current
                previous_scale = self._visual_scale(previous)
                current = closeup
                point_x = closeup.rectangle.left + closeup.rectangle.width / 2.0
                point_y = closeup.rectangle.top + closeup.rectangle.height / 2.0
                effective_zoom_step = zoom_step + fallback_zoom_delta
                if (
                    self.options.zoom_strategy == "adaptive"
                    and effective_zoom_step > 0
                ):
                    growth = self._visual_scale(closeup) / previous_scale
                    if (
                        previous.class_id != SMALL_TARGET_PROPOSAL_CLASS_ID
                        and closeup.class_id
                        != SMALL_TARGET_PROPOSAL_CLASS_ID
                        and growth > 1.0
                    ):
                        observed_gain = math.log(growth) / effective_zoom_step
                        observed_gain = min(
                            max(
                                observed_gain,
                                _MIN_ZOOM_GAIN_PER_DELTA,
                            ),
                            _MAX_ZOOM_GAIN_PER_DELTA,
                        )
                        self._zoom_gain_per_delta = (
                            self._zoom_gain_per_delta
                            * (1.0 - _ZOOM_GAIN_ALPHA)
                            + observed_gain * _ZOOM_GAIN_ALPHA
                        )
                    if (
                        growth
                        < self.options.adaptive_min_scale_growth_ratio
                    ):
                        stalled_rounds += 1
                    else:
                        stalled_rounds = 0
                    if stalled_rounds >= _MAX_STALLED_ZOOM_ROUNDS:
                        result = (
                            "candidate_not_confirmed"
                            if closeup.class_id
                            == SMALL_TARGET_PROPOSAL_CLASS_ID
                            else "insufficient_resolution"
                        )
                        return
            if current.class_id == SMALL_TARGET_PROPOSAL_CLASS_ID:
                result = "candidate_not_confirmed"
                return
            if (
                self.options.zoom_strategy == "adaptive"
                and (
                    not self._adaptive_target_reached(current)
                    or not self._capture_target_centered(current)
                )
            ):
                result = "insufficient_resolution"
                return
            if self._stop_event.is_set():
                raise InterruptedError(
                    "PTZ verification interrupted by shutdown"
                )
            self._renew_lease(force=True)
            self._camera.autofocus()
            validation_error = ""
            for capture_attempt in range(
                self.options.evidence_capture_attempts
            ):
                if self._stop_event.is_set():
                    raise InterruptedError(
                        "PTZ verification interrupted by shutdown"
                    )
                self._renew_lease(force=True)
                content, mime_type, captured_at = self._camera.capture()
                valid, validation_error = self._validate_evidence_capture(
                    content,
                    current,
                )
                if valid:
                    self._repository.store_evidence(
                        job_id=job.job_id,
                        content=content,
                        mime_type=mime_type,
                        captured_at=captured_at,
                    )
                    result = "boat_confirmed"
                    break
                if (
                    capture_attempt + 1
                    < self.options.evidence_capture_attempts
                ):
                    self._stop_event.wait(0.3)
            else:
                result = "evidence_not_confirmed"
                error = validation_error or "证据图片未通过二次验真"
                return
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = "target_lost"
        finally:
            if departed_home:
                home_errors: list[str] = []
                for _attempt in range(2):
                    try:
                        before_home_version = (
                            self._snapshot_provider().result_version
                        )
                        home_started_at = time.monotonic()
                        self._advance_view_generation()
                        # Refresh immediately before the safety-critical HOME
                        # call. A long multi-round verification must not let
                        # its exclusive camera lease expire while off preset.
                        self._renew_lease(force=True)
                        self._camera.home()
                        self._monitoring_not_before = (
                            home_started_at
                            + self.options.home_frame_delay_seconds
                        )
                        self._synchronize_monitoring_view(
                            before_home_version,
                            required=not self._stop_event.is_set(),
                        )
                        # A successful HOME command alone is not enough while
                        # the stream is live: downstream detections must also
                        # prove that fresh monitoring-view frames arrived.
                        # Otherwise the camera can remain at a close-up view
                        # while the coordinator incorrectly accepts new work.
                        home_returned = True
                        break
                    except Exception as exc:
                        home_errors.append(f"{type(exc).__name__}: {exc}")
                        try:
                            self._camera.stop()
                        except Exception:
                            pass
                        self._stop_event.wait(0.2)
                if not home_returned:
                    self._recovery_required = True
                    result = "recovery_failed"
                    recovery_error = "; ".join(home_errors)
                    error = f"{error}; {recovery_error}" if error else recovery_error
            else:
                home_returned = True
            if not departed_home or home_returned:
                self._verifying.clear()
            self._repository.finish(
                job_id=job.job_id,
                result=result,
                error=error,
                home_returned=home_returned,
                now=time.time(),
                options=self.options,
            )

    def _validate_evidence_capture(
        self,
        content: bytes,
        reference: VesselDetection,
    ) -> tuple[bool, str]:
        if not self.options.evidence_validation_required:
            return True, ""
        if self._evidence_validator is None:
            return False, "证据图片复检器不可用"
        validation = self._evidence_validator(
            content,
            self.options.command_timeout_seconds,
        )
        if validation.state != "running":
            return False, validation.message or "证据图片复检失败"
        selected = self._select_centered(
            validation.detections,
            reference,
            center_radius=min(self.options.reacquire_center_radius, 0.30),
        )
        target = selected.detection
        if target is None:
            if selected.competing_groups:
                return False, "证据图片中存在无法归属的竞争船群"
            return False, "证据图片中未重新识别到居中的船舶"
        if not (
            target.rectangle.width
            >= self.options.adaptive_target_width_ratio
            * self.options.evidence_target_scale_ratio
            or target.rectangle.height
            >= self.options.adaptive_target_height_ratio
            * self.options.evidence_target_scale_ratio
        ):
            return False, "证据图片中的船舶像素尺寸不足"
        sharpness = validation.sharpness_for(target.object_id)
        if sharpness < self.options.evidence_minimum_sharpness:
            return (
                False,
                "证据图片中的船舶区域清晰度不足: "
                f"{sharpness:.2f}<{self.options.evidence_minimum_sharpness:.2f}",
            )
        return True, ""

    def _fallback_zoom_confirmed_target(
        self,
        *,
        reference: VesselDetection,
        total_zoom_delta: int,
        started: float,
        maximum_rounds: int,
    ) -> tuple[VesselDetection | None, int, int]:
        """Give a confirmed distant boat a bounded zoom chance after a miss.

        A PTZ move changes the complete image coordinate system.  A tiny boat
        can therefore disappear from the detector for a few close-up frames
        even when FASTGOTO moved in the right direction.  Requiring a fresh
        detection before every zoom makes the verifier return HOME before the
        target becomes large enough to recognize.  Once (and only once) the
        HOME view has already confirmed a real boat, keep the current optical
        centre and apply a small bounded zoom before trying to reacquire it.
        """
        consumed_zoom_delta = 0
        attempted_rounds = 0
        for _round in range(maximum_rounds):
            if self._stop_event.is_set():
                break
            if (
                time.monotonic() - started
                >= self.options.maximum_off_home_seconds
            ):
                break
            remaining = (
                self.options.adaptive_max_total_zoom_delta
                - total_zoom_delta
                - consumed_zoom_delta
            )
            zoom_step = min(
                self.options.confirmed_target_fallback_zoom_step,
                remaining,
            )
            if zoom_step <= 0:
                break
            before_version = self._snapshot_provider().result_version
            # FASTGOTO coordinates are relative to the current view.  After
            # the first locate, (0.5, 0.5) preserves the acquired direction
            # and requests optical zoom without another blind pan/tilt move.
            self._advance_view_generation()
            self._renew_lease(force=True)
            self._camera.locate(0.5, 0.5, zoom_step)
            attempted_rounds += 1
            consumed_zoom_delta += zoom_step
            minimum_updated_at = time.monotonic()
            if self.options.settle_seconds:
                self._stop_event.wait(self.options.settle_seconds)
            reacquired = self._wait_for_centered_detection(
                after_version=before_version,
                minimum_updated_at=minimum_updated_at,
                allow_proposal=True,
                reference=reference,
            )
            if reacquired.detection is not None:
                return (
                    reacquired.detection,
                    consumed_zoom_delta,
                    attempted_rounds,
                )
            if reacquired.saw_candidates:
                # Something is visible but it cannot be associated safely
                # with the locked target.  Additional blind zoom could switch
                # to another group, so stop the fallback immediately.
                break
        return None, consumed_zoom_delta, attempted_rounds

    def _wait_for_centered_detection(
        self,
        *,
        after_version: int,
        minimum_updated_at: float,
        allow_proposal: bool,
        reference: VesselDetection,
    ) -> _TargetSelection:
        started = time.monotonic()
        deadline = started + self.options.reacquire_timeout_seconds
        strict_until = started + (
            self.options.reacquire_timeout_seconds
            * _STRICT_REACQUIRE_FRACTION
        )
        last_checked_version = after_version
        latest_confirmed: tuple[VesselDetection, ...] = ()
        latest_proposals: tuple[VesselDetection, ...] = ()
        saw_candidates = False
        competing_groups = False
        while time.monotonic() < deadline and not self._stop_event.is_set():
            snapshot = self._snapshot_provider()
            if (
                snapshot.state == "running"
                and snapshot.result_version > last_checked_version
                and snapshot.updated_at is not None
                and snapshot.updated_at >= minimum_updated_at
            ):
                last_checked_version = snapshot.result_version
                latest_confirmed = tuple(
                    item
                    for item in snapshot.detections
                    if item.class_id != SMALL_TARGET_PROPOSAL_CLASS_ID
                )
                latest_proposals = tuple(
                    item
                    for item in snapshot.detections
                    if item.class_id == SMALL_TARGET_PROPOSAL_CLASS_ID
                )
            center_radius = (
                self.options.reacquire_strict_center_radius
                if time.monotonic() < strict_until
                else self.options.reacquire_center_radius
            )
            confirmed = self._select_centered(
                latest_confirmed,
                reference,
                center_radius=center_radius,
            )
            saw_candidates = saw_candidates or confirmed.saw_candidates
            competing_groups = (
                competing_groups or confirmed.competing_groups
            )
            if confirmed.detection is not None:
                return confirmed
            if allow_proposal and not confirmed.competing_groups:
                proposal = self._select_centered(
                    latest_proposals,
                    reference,
                    center_radius=center_radius,
                )
                saw_candidates = saw_candidates or proposal.saw_candidates
                competing_groups = (
                    competing_groups or proposal.competing_groups
                )
                if proposal.detection is not None:
                    return proposal
            self._stop_event.wait(0.1)
        return _TargetSelection(
            saw_candidates=saw_candidates,
            competing_groups=competing_groups,
        )

    def _nearest_centered(
        self,
        detections: tuple[VesselDetection, ...],
        reference: VesselDetection,
    ) -> VesselDetection | None:
        """Compatibility wrapper used by focused association tests."""
        return self._select_centered(
            detections,
            reference,
            center_radius=self.options.reacquire_center_radius,
        ).detection

    def _select_centered(
        self,
        detections: tuple[VesselDetection, ...],
        reference: VesselDetection,
        *,
        center_radius: float,
    ) -> _TargetSelection:
        if not detections:
            return _TargetSelection()
        eligible = tuple(
            item
            for item in detections
            if math.hypot(
                item.rectangle.center[0] - 0.5,
                item.rectangle.center[1] - 0.5,
            )
            <= center_radius
        )
        if not eligible:
            return _TargetSelection(saw_candidates=True)
        groups = self._cluster_reacquire_detections(eligible)
        scored = [
            self._score_reacquire_group(group, reference)
            for group in groups
        ]
        scored.sort(key=lambda pair: pair[0])
        if (
            len(scored) > 1
            and scored[1][0] - scored[0][0]
            < _REACQUIRE_AMBIGUITY_MARGIN
        ):
            return _TargetSelection(
                saw_candidates=True,
                competing_groups=True,
            )
        return _TargetSelection(
            detection=scored[0][1],
            saw_candidates=True,
        )

    def _cluster_reacquire_detections(
        self,
        detections: tuple[VesselDetection, ...],
    ) -> tuple[tuple[VesselDetection, ...], ...]:
        """Group adjacent boats without chaining across the whole frame."""
        groups: list[list[VesselDetection]] = []
        ordered = sorted(
            detections,
            key=lambda item: math.hypot(
                item.rectangle.center[0] - 0.5,
                item.rectangle.center[1] - 0.5,
            ),
        )
        for item in ordered:
            item_x, item_y = item.rectangle.center
            best_index: int | None = None
            best_distance = math.inf
            for index, group in enumerate(groups):
                center_x = statistics.fmean(
                    member.rectangle.center[0] for member in group
                )
                center_y = statistics.fmean(
                    member.rectangle.center[1] for member in group
                )
                distance = math.hypot(
                    item_x - center_x,
                    item_y - center_y,
                )
                nearest_box_gap = min(
                    self._rectangle_gap(item, member)
                    for member in group
                )
                if (
                    (
                        distance <= self.options.reacquire_cluster_radius
                        or (
                            nearest_box_gap
                            <= self.options.reacquire_cluster_radius * 0.25
                            and distance
                            <= self.options.reacquire_cluster_radius * 2.0
                        )
                    )
                    and distance < best_distance
                ):
                    best_index = index
                    best_distance = distance
            if best_index is None:
                groups.append([item])
            else:
                groups[best_index].append(item)
        return tuple(tuple(group) for group in groups)

    @staticmethod
    def _rectangle_gap(
        first: VesselDetection,
        second: VesselDetection,
    ) -> float:
        first_right = first.rectangle.left + first.rectangle.width
        first_bottom = first.rectangle.top + first.rectangle.height
        second_right = second.rectangle.left + second.rectangle.width
        second_bottom = second.rectangle.top + second.rectangle.height
        horizontal = max(
            first.rectangle.left - second_right,
            second.rectangle.left - first_right,
            0.0,
        )
        vertical = max(
            first.rectangle.top - second_bottom,
            second.rectangle.top - first_bottom,
            0.0,
        )
        return math.hypot(horizontal, vertical)

    def _score_reacquire_group(
        self,
        group: tuple[VesselDetection, ...],
        reference: VesselDetection,
    ) -> tuple[float, VesselDetection]:
        """Build a group-centred target whose scale remains one-boat scale."""
        center_x = statistics.fmean(
            item.rectangle.center[0] for item in group
        )
        center_y = statistics.fmean(
            item.rectangle.center[1] for item in group
        )
        representative = min(
            group,
            key=lambda item: (
                math.hypot(
                    item.rectangle.center[0] - 0.5,
                    item.rectangle.center[1] - 0.5,
                )
                + self._reacquire_continuity_penalty(item, reference)
            ),
        )
        width = float(
            statistics.median(item.rectangle.width for item in group)
        )
        height = float(
            statistics.median(item.rectangle.height for item in group)
        )
        width = min(max(width, 1e-6), 1.0)
        height = min(max(height, 1e-6), 1.0)
        left = min(max(center_x - width / 2.0, 0.0), 1.0 - width)
        top = min(max(center_y - height / 2.0, 0.0), 1.0 - height)
        target = VesselDetection(
            object_id=representative.object_id,
            rectangle=NormalizedRect(left, top, width, height),
            confidence=max(item.confidence for item in group),
            class_id=representative.class_id,
            hits=max(item.hits for item in group),
        )
        score = (
            math.hypot(center_x - 0.5, center_y - 0.5)
            + self._reacquire_continuity_penalty(
                representative,
                reference,
            )
        )
        return score, target

    def _reacquire_continuity_penalty(
        self,
        item: VesselDetection,
        reference: VesselDetection,
    ) -> float:
        reference_scale = self._visual_scale(reference)
        reference_aspect = reference.rectangle.width / max(
            reference.rectangle.height,
            1e-6,
        )
        if item.class_id != reference.class_id:
            return 0.0
        penalty = 0.0
        item_scale = self._visual_scale(item)
        if item_scale < reference_scale * 0.70:
            penalty += 0.10
        item_aspect = item.rectangle.width / max(
            item.rectangle.height,
            1e-6,
        )
        penalty += min(
            abs(math.log(max(item_aspect / reference_aspect, 1e-6)))
            * 0.04,
            0.08,
        )
        return penalty

    def _synchronize_monitoring_view(
        self,
        after_version: int,
        *,
        required: bool = True,
    ) -> bool:
        """Drain zoom-view frames until a post-HOME monitoring frame arrives."""
        deadline = time.monotonic() + max(
            self.options.reacquire_timeout_seconds,
            self.options.home_frame_delay_seconds + 1.0,
        )
        stable_versions = 0
        previous_version = after_version
        while time.monotonic() < deadline and not self._stop_event.is_set():
            snapshot = self._snapshot_provider()
            if snapshot.result_version > self._last_monitoring_version:
                self._last_monitoring_version = snapshot.result_version
            if (
                snapshot.result_version > after_version
                and snapshot.updated_at is not None
                and snapshot.updated_at >= self._monitoring_not_before
            ):
                if snapshot.result_version > previous_version:
                    stable_versions += 1
                    previous_version = snapshot.result_version
                if stable_versions >= self.options.home_stable_frames:
                    return True
            self._stop_event.wait(0.1)
        if required and not self._stop_event.is_set():
            raise TimeoutError("HOME后未收到足够的稳定全景帧")
        return False
