import json
import sqlite3
import threading
import time
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.request import Request

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.ptz_verification import (
    CameraControlClient,
    PtzVerificationCoordinator,
    PtzVerificationOptions,
    PtzVerificationRepository,
    _merge_proposal_fragments,
)
from rtsp_annotator.vessel_detection import (
    EvidenceValidationResult,
    VesselDetection,
    VesselSnapshot,
)
from rtsp_annotator.vessel_detection import SMALL_TARGET_PROPOSAL_CLASS_ID


def detection(
    object_id: int,
    x: float,
    y: float,
    *,
    width: float = 0.03,
    height: float = 0.02,
    hits: int = 4,
    class_id: int = 8,
) -> VesselDetection:
    return VesselDetection(
        object_id=object_id,
        rectangle=NormalizedRect(
            left=x - width / 2,
            top=y - height / 2,
            width=width,
            height=height,
        ),
        confidence=0.2,
        class_id=class_id,
        hits=hits,
    )


class SnapshotSource:
    def __init__(self, wide_detections: tuple[VesselDetection, ...]) -> None:
        self.version = 1
        self.wide_detections = wide_detections
        self.current = VesselSnapshot(
            state="running",
            detections=wide_detections,
            result_version=self.version,
            updated_at=time.monotonic(),
        )
        self.pending_closeup = False
        self.closeup_confirmed = True
        self.closeup_results: list[tuple[VesselDetection, ...]] = []

    def snapshot(self) -> VesselSnapshot:
        if self.pending_closeup:
            self.pending_closeup = False
            self.version += 1
            if self.closeup_results:
                closeup = self.closeup_results.pop(0)
            else:
                closeup = (
                    (
                        detection(
                            900 + self.version,
                            0.5,
                            0.5,
                            width=0.3,
                            height=0.2,
                        ),
                    )
                    if self.closeup_confirmed
                    else ()
                )
            self.current = VesselSnapshot(
                state="running",
                detections=closeup,
                result_version=self.version,
                updated_at=time.monotonic(),
            )
        return self.current

    def return_home(self, detections: tuple[VesselDetection, ...] | None = None) -> None:
        self.version += 1
        self.current = VesselSnapshot(
            state="running",
            detections=detections or self.wide_detections,
            result_version=self.version,
            updated_at=time.monotonic(),
        )


class FakeCameraClient:
    def __init__(self, source: SnapshotSource) -> None:
        self.source = source
        self.locates: list[tuple[float, float, int]] = []
        self.home_calls = 0
        self.autofocus_calls = 0
        self.capture_calls = 0
        self.stop_calls = 0
        self.fail_home = False
        self.fail_locate = False
        self.update_source_on_home = True
        self.lease_acquires = 0
        self.lease_releases = 0

    def acquire_lease(self, _owner: str, _ttl_seconds: float) -> None:
        self.lease_acquires += 1

    def release_lease(self) -> None:
        self.lease_releases += 1

    def locate(self, x: float, y: float, zoom_delta: int) -> None:
        self.locates.append((x, y, zoom_delta))
        if self.fail_locate:
            raise RuntimeError("locate timed out after movement")
        self.source.pending_closeup = True

    def autofocus(self) -> None:
        self.autofocus_calls += 1

    def capture(self) -> tuple[bytes, str, float]:
        self.capture_calls += 1
        return b"\xff\xd8close-up\xff\xd9", "image/jpeg", time.time()

    def home(self) -> None:
        self.home_calls += 1
        if self.fail_home:
            raise RuntimeError("home failed")
        if not self.update_source_on_home:
            return
        moved = tuple(
            detection(
                item.object_id + 100,
                item.rectangle.center[0] + 0.005,
                item.rectangle.center[1],
                width=item.rectangle.width,
                height=item.rectangle.height,
            )
            for item in self.source.wide_detections
        )
        self.source.return_home(moved)

    def stop(self) -> None:
        self.stop_calls += 1


class ContinuousTrackingSource(SnapshotSource):
    def __init__(
        self,
        wide_detections: tuple[VesselDetection, ...],
        tracking_results: list[tuple[VesselDetection, ...]],
        *,
        stop_when_exhausted: bool = True,
    ) -> None:
        super().__init__(wide_detections)
        self.tracking_results = tracking_results
        self.tracking_armed = False
        self.tracking = False
        self.stop_when_exhausted = stop_when_exhausted

    def start_tracking(self) -> None:
        self.tracking_armed = True

    def snapshot(self) -> VesselSnapshot:
        if self.pending_closeup:
            self.pending_closeup = False
            self.version += 1
            detections = (
                self.tracking_results.pop(0)
                if self.tracking and self.tracking_results
                else (
                    detection(
                        900 + self.version,
                        0.5,
                        0.5,
                        width=0.25,
                        height=0.18,
                    ),
                )
            )
            self.current = VesselSnapshot(
                state="running",
                detections=detections,
                result_version=self.version,
                updated_at=time.monotonic(),
            )
            return self.current
        if self.tracking_armed:
            self.tracking_armed = False
            self.tracking = True
            return self.current
        if self.tracking:
            self.version += 1
            if self.tracking_results:
                self.current = VesselSnapshot(
                    state="running",
                    detections=self.tracking_results.pop(0),
                    result_version=self.version,
                    updated_at=time.monotonic(),
                )
            elif self.stop_when_exhausted:
                self.current = VesselSnapshot(
                    state="stopped",
                    result_version=self.version,
                    updated_at=time.monotonic(),
                )
        return self.current


class ContinuousTrackingCamera(FakeCameraClient):
    def capture(self) -> tuple[bytes, str, float]:
        result = super().capture()
        assert isinstance(self.source, ContinuousTrackingSource)
        self.source.start_tracking()
        return result


class PtzVerificationTests(unittest.TestCase):
    def options(self, **overrides):
        values = {
            "enabled": True,
            "camera_id": "camera-01",
            "zoom_steps": (4, 4),
            "adaptive_target_width_ratio": 0.25,
            "adaptive_target_height_ratio": 0.18,
            "settle_seconds": 0,
            "reacquire_timeout_seconds": 1,
            "monitoring_interval_seconds": 0.05,
            "home_frame_delay_seconds": 0,
            "home_stable_frames": 1,
            "minimum_target_observations": 1,
            "evidence_validation_required": False,
        }
        values.update(overrides)
        return PtzVerificationOptions(**values)

    def test_adaptive_options_round_trip_to_worker_payload(self) -> None:
        options = self.options(
            trace_logging_enabled=True,
            zoom_strategy="adaptive",
            adaptive_target_width_ratio=0.3,
            adaptive_target_height_ratio=0.2,
            adaptive_min_step=2,
            adaptive_max_step=7,
            adaptive_max_rounds=4,
            adaptive_max_total_zoom_delta=18,
            adaptive_min_scale_growth_ratio=1.08,
            confirmed_target_fallback_zoom_rounds=2,
            confirmed_target_fallback_zoom_step=4,
            reacquire_strict_center_radius=0.18,
            reacquire_center_radius=0.40,
            reacquire_cluster_radius=0.16,
            proposal_minimum_interval_seconds=45,
            proposal_maximum_verifications_per_hour=6,
            continuous_tracking=True,
            tracking_center_deadband=0.08,
            tracking_command_interval_seconds=0.3,
            tracking_settle_seconds=0.1,
            tracking_recovery_enabled=True,
            tracking_recovery_interval_seconds=1.5,
            tracking_recovery_zoom_out_step=2,
            tracking_recovery_max_attempts=4,
            tracking_lost_timeout_seconds=3,
            tracking_max_duration_seconds=0,
            tracking_zoom_hysteresis_ratio=0.15,
            tracking_zoom_step=2,
            tracking_initial_extra_zoom_step=1,
            vessel_number_recognition_enabled=True,
            vessel_number_fallback="10032",
        )

        restored = PtzVerificationOptions.from_payload(options.to_payload())

        self.assertEqual(restored, options)

    def test_tracking_recovery_zooms_out_and_reacquires_target(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            reference = detection(
                1,
                0.55,
                0.5,
                width=0.25,
                height=0.18,
            )
            source = SnapshotSource((reference,))
            camera = FakeCameraClient(source)
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    continuous_tracking=True,
                    tracking_recovery_enabled=True,
                    tracking_recovery_interval_seconds=0.5,
                    tracking_recovery_zoom_out_step=2,
                    tracking_recovery_max_attempts=1,
                    tracking_lost_timeout_seconds=2,
                ),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(Path(directory)),
                camera_client=camera,  # type: ignore[arg-type]
            )

            observation = coordinator._wait_for_tracking_detection(
                after_version=source.version,
                reference=reference,
            )

        self.assertIsNotNone(observation.detection)
        self.assertEqual(camera.locates, [(0.55, 0.5, -2)])
        self.assertEqual(
            coordinator.tracking_metrics["tracking_corrections"],
            1,
        )

    def test_zero_tracking_max_duration_means_no_time_limit(self) -> None:
        options = self.options(tracking_max_duration_seconds=0)

        options.validate()
        self.assertEqual(options.tracking_max_duration_seconds, 0)

    def test_tracking_keeps_same_detector_id_at_edge_of_frame(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            reference = detection(7, 0.5, 0.5)
            edge = detection(7, 0.98, 0.5)
            source = SnapshotSource((edge,))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(reacquire_center_radius=0.45),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(Path(directory)),
                camera_client=FakeCameraClient(source),  # type: ignore[arg-type]
            )

            selected = coordinator._select_tracking_target(
                (edge,),
                reference,
                center_radius=0.45,
            )

        self.assertIs(selected.detection, edge)

    def test_demo_reacquire_does_not_switch_to_centered_competing_boat(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            reference = detection(1, 0.30, 0.50, width=0.12, height=0.08)
            competing = detection(2, 0.50, 0.50, width=0.12, height=0.08)
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    tracking_profile="demo_continuous",
                    continuous_tracking=True,
                    tracking_max_duration_seconds=0,
                ),
                snapshot_provider=lambda: VesselSnapshot(),
                repository=PtzVerificationRepository(Path(directory)),
                camera_client=FakeCameraClient(SnapshotSource((reference,))),  # type: ignore[arg-type]
            )
            selected = coordinator._select_tracking_target(
                (competing,), reference, center_radius=0.45,
                allow_center_fallback=False,
            )
        self.assertIsNone(selected.detection)

    def test_vessel_number_uses_configured_fallback_when_ocr_is_unavailable(
        self,
    ) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            item = detection(1, 0.5, 0.5, width=0.4, height=0.3)
            source = SnapshotSource((item,))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    vessel_number_recognition_enabled=True,
                    vessel_number_fallback="10032",
                ),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(Path(directory)),
                camera_client=FakeCameraClient(source),  # type: ignore[arg-type]
            )

            with patch(
                "rtsp_annotator.ptz_verification._ocr_vessel_number",
                return_value=None,
            ):
                value = coordinator._recognize_vessel_number(b"jpeg", item)

        self.assertEqual(value, "10032")

    def test_task_trace_logging_is_disabled_by_default(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = SnapshotSource((detection(1, 0.7, 0.5),))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(root),
                camera_client=FakeCameraClient(source),  # type: ignore[arg-type]
            )

            self.assertIsNotNone(coordinator.run_once())
            self.assertFalse((root / "task-traces").exists())

    def test_task_trace_records_one_file_per_job_and_camera_action(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = SnapshotSource((detection(1, 0.7, 0.5),))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(trace_logging_enabled=True),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(root),
                camera_client=FakeCameraClient(source),  # type: ignore[arg-type]
            )

            job_id = coordinator.run_once()

            self.assertIsNotNone(job_id)
            trace_files = list((root / "task-traces").glob("*.jsonl"))
            self.assertEqual(trace_files, [root / "task-traces" / f"{job_id}.jsonl"])
            records = [
                json.loads(line)
                for line in trace_files[0].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["event"], "task.started")
            self.assertEqual(records[-1]["event"], "task.finished")
            self.assertEqual(records[-1]["result"], "boat_confirmed")
            self.assertTrue(all(item["task_id"] == job_id for item in records))
            completed_actions = [
                item["action"]
                for item in records
                if item["event"] == "camera_control.completed"
            ]
            self.assertEqual(
                completed_actions,
                ["locate", "autofocus", "capture", "home"],
            )
            capture = next(
                item
                for item in records
                if item["event"] == "camera_control.completed"
                and item["action"] == "capture"
            )
            self.assertEqual(capture["result"]["size_bytes"], 12)
            self.assertNotIn("content", capture["result"])

    def test_task_trace_records_failed_camera_action(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = SnapshotSource((detection(1, 0.7, 0.5),))
            camera = FakeCameraClient(source)
            camera.fail_locate = True
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(trace_logging_enabled=True),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(root),
                camera_client=camera,  # type: ignore[arg-type]
            )

            job_id = coordinator.run_once()

            trace_path = root / "task-traces" / f"{job_id}.jsonl"
            records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            failed = [
                item
                for item in records
                if item["event"] == "camera_control.failed"
            ]
            self.assertEqual(failed[0]["action"], "locate")
            self.assertEqual(failed[0]["error_type"], "RuntimeError")
            self.assertEqual(records[-1]["result"], "target_lost")

    def test_overlay_state_keeps_target_and_last_four_camera_operations(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            item = detection(1, 0.7, 0.5)
            source = SnapshotSource((item,))
            camera = FakeCameraClient(source)
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(Path(directory)),
                camera_client=camera,  # type: ignore[arg-type]
            )
            coordinator._set_overlay_target(item)
            coordinator._camera_control(
                "locate",
                lambda: camera.locate(0.7, 0.5, 2),
                reason="continuous_tracking",
                zoom_delta=2,
            )
            for _ in range(4):
                coordinator._camera_control("autofocus", camera.autofocus)

            overlay = coordinator.overlay_state()

        self.assertEqual(overlay.target_rectangle, item.rectangle)
        self.assertEqual(len(overlay.operation_lines), 4)
        self.assertTrue(
            all(line.endswith("自动对焦") for line in overlay.operation_lines)
        )

    def test_tracking_can_add_one_final_zoom_step_before_capture(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = ContinuousTrackingSource(
                (detection(1, 0.7, 0.5),),
                [],
            )
            camera = ContinuousTrackingCamera(source)
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    continuous_tracking=True,
                    tracking_initial_extra_zoom_step=1,
                ),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(Path(directory)),
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(len(camera.locates), 2)
            self.assertEqual(camera.locates[1], (0.5, 0.5, 1))
            self.assertEqual(camera.capture_calls, 1)
            self.assertEqual(camera.home_calls, 1)

    def test_green_primary_detection_can_trigger_ptz_without_sidecar_box(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource(())
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    minimum_target_observations=3,
                    primary_target_minimum_observations=1,
                ),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )
            coordinator.publish_primary_detections(
                (detection(-12, 0.7, 0.5),),
                updated_at=time.monotonic(),
            )

            job_id = coordinator.run_once()

            self.assertIsNotNone(job_id)
            self.assertTrue(camera.locates)
            self.assertEqual(camera.capture_calls, 1)
            self.assertEqual(
                coordinator.tracking_metrics["ptz_trigger_status"],
                "triggered",
            )
            self.assertEqual(
                coordinator.tracking_metrics[
                    "ptz_trigger_required_observations"
                ],
                1,
            )

    def test_tracking_loss_uses_short_retry_even_after_confirmed_capture(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = ContinuousTrackingSource(
                (detection(1, 0.7, 0.5),),
                [],
                stop_when_exhausted=False,
            )
            camera = ContinuousTrackingCamera(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    continuous_tracking=True,
                    tracking_lost_timeout_seconds=1,
                    confirmed_cooldown_seconds=1_200,
                    lost_retry_seconds=15,
                ),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(
                coordinator.tracking_metrics["tracking_last_end_reason"],
                "target_lost",
            )
            job = repository.list_jobs()[0]
            self.assertEqual(job["result"], "boat_confirmed")
            with sqlite3.connect(repository.database_path) as connection:
                cooldown_until, updated_at = connection.execute(
                    "SELECT cooldown_until,updated_at FROM targets"
                ).fetchone()
            self.assertAlmostEqual(cooldown_until - updated_at, 15, delta=0.1)

    def test_tracking_zoom_uses_hysteresis_around_target_size(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.5, 0.5),))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    continuous_tracking=True,
                    tracking_zoom_hysteresis_ratio=0.2,
                    tracking_zoom_step=2,
                ),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(Path(directory)),
                camera_client=FakeCameraClient(source),  # type: ignore[arg-type]
            )

            self.assertEqual(
                coordinator._tracking_zoom_delta(
                    detection(2, 0.5, 0.5, width=0.15, height=0.08)
                ),
                2,
            )
            self.assertEqual(
                coordinator._tracking_zoom_delta(
                    detection(3, 0.5, 0.5, width=0.25, height=0.18)
                ),
                0,
            )
            self.assertEqual(
                coordinator._tracking_zoom_delta(
                    detection(4, 0.5, 0.5, width=0.35, height=0.25)
                ),
                -2,
            )

    def test_tracking_prefers_same_track_over_a_different_centered_boat(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            reference = detection(
                7,
                0.62,
                0.5,
                width=0.25,
                height=0.18,
            )
            source = SnapshotSource((reference,))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(continuous_tracking=True),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(Path(directory)),
                camera_client=FakeCameraClient(source),  # type: ignore[arg-type]
            )

            selected = coordinator._select_tracking_target(
                (
                    detection(99, 0.5, 0.5, width=0.25, height=0.18),
                    detection(7, 0.64, 0.5, width=0.25, height=0.18),
                ),
                reference,
                center_radius=0.45,
            )

            self.assertIsNotNone(selected.detection)
            self.assertEqual(selected.detection.object_id, 7)

    def test_reacquire_ignores_stale_scale_after_zoom(self) -> None:
        from tempfile import TemporaryDirectory

        reference = detection(
            1,
            0.5,
            0.5,
            width=0.10,
            height=0.05,
        )
        snapshots = iter(
            (
                VesselSnapshot(
                    state="running",
                    detections=(
                        detection(
                            2,
                            0.5,
                            0.5,
                            width=0.10,
                            height=0.05,
                        ),
                    ),
                    result_version=2,
                    updated_at=time.monotonic(),
                ),
                VesselSnapshot(
                    state="running",
                    detections=(
                        detection(
                            3,
                            0.5,
                            0.5,
                            width=0.20,
                            height=0.10,
                        ),
                    ),
                    result_version=3,
                    updated_at=time.monotonic(),
                ),
            )
        )
        latest = VesselSnapshot()

        def snapshot() -> VesselSnapshot:
            nonlocal latest
            try:
                latest = next(snapshots)
            except StopIteration:
                pass
            return latest

        with TemporaryDirectory() as directory:
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(),
                snapshot_provider=snapshot,
                repository=PtzVerificationRepository(Path(directory)),
                camera_client=FakeCameraClient(  # type: ignore[arg-type]
                    SnapshotSource((reference,))
                ),
            )
            selected = coordinator._wait_for_centered_detection(
                after_version=1,
                minimum_updated_at=0,
                allow_proposal=False,
                reference=reference,
                minimum_scale_ratio=1.5,
            )

        self.assertIsNotNone(selected.detection)
        self.assertEqual(selected.detection.object_id, 3)

    def test_continuous_tracking_recenters_rescales_then_returns_home(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = ContinuousTrackingSource(
                (detection(1, 0.7, 0.5),),
                [
                    (detection(10, 0.70, 0.5, width=0.25, height=0.18),),
                    (detection(11, 0.50, 0.5, width=0.15, height=0.08),),
                    (detection(12, 0.50, 0.5, width=0.35, height=0.25),),
                ],
            )
            camera = ContinuousTrackingCamera(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    continuous_tracking=True,
                    settle_seconds=0.1,
                    tracking_command_interval_seconds=0.1,
                ),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            tracking_commands = camera.locates[1:]
            self.assertEqual(
                [item[2] for item in tracking_commands],
                [0, 1, -1],
            )
            self.assertEqual(camera.capture_calls, 1)
            self.assertEqual(camera.home_calls, 1)
            self.assertEqual(coordinator.state, "stopped")
            self.assertEqual(
                coordinator.tracking_metrics["tracking_corrections"],
                3,
            )
            self.assertEqual(
                coordinator.tracking_metrics["tracking_last_end_reason"],
                "stream_unavailable",
            )
            jobs = repository.list_jobs()
            self.assertEqual(jobs[0]["result"], "boat_confirmed")
            self.assertTrue(jobs[0]["home_returned"])

    def test_tracking_locate_timeout_still_forces_home(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = ContinuousTrackingSource(
                (
                    detection(
                        1,
                        0.5,
                        0.5,
                        width=0.25,
                        height=0.18,
                    ),
                ),
                [
                    (detection(10, 0.70, 0.5, width=0.25, height=0.18),),
                ],
            )
            camera = ContinuousTrackingCamera(source)
            camera.fail_locate = True
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(continuous_tracking=True),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(len(camera.locates), 1)
            self.assertEqual(camera.home_calls, 1)
            job = repository.list_jobs()[0]
            self.assertEqual(job["result"], "target_lost")
            self.assertTrue(job["home_returned"])

    def test_evidence_rejection_does_not_disable_continuous_tracking(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = ContinuousTrackingSource(
                (
                    detection(
                        1,
                        0.5,
                        0.5,
                        width=0.25,
                        height=0.18,
                    ),
                ),
                [],
            )
            camera = ContinuousTrackingCamera(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    continuous_tracking=True,
                    evidence_validation_required=True,
                ),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
                evidence_validator=lambda _content, _timeout: (
                    EvidenceValidationResult(
                        state="running",
                        detections=(),
                    )
                ),
            )

            coordinator.run_once()

            self.assertEqual(
                coordinator.tracking_metrics["tracking_last_end_reason"],
                "stream_unavailable",
            )
            job = repository.list_jobs()[0]
            self.assertEqual(job["result"], "evidence_not_confirmed")
            self.assertIn("未重新识别", job["error"])

    def test_manual_return_home_interrupts_tracking_and_restores_preset(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = ContinuousTrackingSource(
                (detection(1, 0.7, 0.5),),
                [],
                stop_when_exhausted=False,
            )
            camera = ContinuousTrackingCamera(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    continuous_tracking=True,
                    tracking_lost_timeout_seconds=5,
                ),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )
            worker = threading.Thread(target=coordinator.run_once)
            worker.start()
            deadline = time.monotonic() + 2
            while coordinator.state != "tracking" and time.monotonic() < deadline:
                time.sleep(0.01)

            coordinator.request_return_home("manual-home-1")
            worker.join(timeout=2)

            self.assertFalse(worker.is_alive())
            self.assertEqual(camera.stop_calls, 1)
            self.assertEqual(camera.home_calls, 1)
            self.assertEqual(
                coordinator.tracking_metrics["tracking_last_end_reason"],
                "manual_return_home",
            )
            self.assertEqual(
                coordinator.tracking_metrics["ptz_last_return_home_request_id"],
                "manual-home-1",
            )
            self.assertTrue(
                coordinator.tracking_metrics["ptz_manual_hold"]
            )
            self.assertEqual(coordinator.state, "manual_hold")
            self.assertTrue(repository.list_jobs()[0]["home_returned"])

            # Emergency HOME is latched. A fresh detection must not start a
            # new movement until the stream is explicitly recreated/updated.
            source.return_home((detection(2, 0.7, 0.5),))
            self.assertIsNone(coordinator.run_once())
            self.assertEqual(camera.home_calls, 1)

    def test_continuous_mode_tracks_confirmed_boat_before_target_scale(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = ContinuousTrackingSource(
                (detection(1, 0.7, 0.5),),
                [],
            )
            camera = ContinuousTrackingCamera(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    zoom_strategy="adaptive",
                    adaptive_target_width_ratio=0.50,
                    adaptive_target_height_ratio=0.50,
                    adaptive_max_rounds=1,
                    continuous_tracking=True,
                ),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(camera.capture_calls, 1)
            self.assertEqual(
                coordinator.tracking_metrics["tracking_last_end_reason"],
                "stream_unavailable",
            )
            self.assertEqual(repository.list_jobs()[0]["result"], "boat_confirmed")

    def test_sdk_capture_is_saved_only_after_boat_and_sharpness_validation(
        self,
    ) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource(
                (detection(1, 0.5, 0.5, width=0.30, height=0.20),)
            )
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(evidence_validation_required=True),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
                evidence_validator=lambda _content, _timeout: (
                    EvidenceValidationResult(
                        state="running",
                        detections=(
                            detection(
                                77,
                                0.5,
                                0.5,
                                width=0.28,
                                height=0.19,
                            ),
                        ),
                        sharpness_by_object_id=((77, 80.0),),
                    )
                ),
            )

            coordinator.run_once()

            job = repository.list_jobs()[0]
            self.assertEqual(job["result"], "boat_confirmed")
            self.assertEqual(camera.capture_calls, 1)
            self.assertEqual(len(job["images"]), 1)

    def test_invalid_sdk_capture_is_retried_but_never_saved_as_evidence(
        self,
    ) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource(
                (detection(1, 0.5, 0.5, width=0.30, height=0.20),)
            )
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(evidence_validation_required=True),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
                evidence_validator=lambda _content, _timeout: (
                    EvidenceValidationResult(state="running")
                ),
            )

            coordinator.run_once()

            job = repository.list_jobs()[0]
            self.assertEqual(job["result"], "evidence_not_confirmed")
            self.assertEqual(camera.capture_calls, 2)
            self.assertEqual(job["images"], [])

    def test_confirmed_distant_boat_gets_bounded_fallback_zoom_after_miss(
        self,
    ) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.8, 0.5),))
            source.closeup_results = [
                (),
                (detection(102, 0.5, 0.5, width=0.3, height=0.2),),
            ]
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(zoom_strategy="adaptive"),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(
                camera.locates,
                [(0.8, 0.5, 6), (0.5, 0.5, 3)],
            )
            self.assertEqual(camera.capture_calls, 1)
            self.assertEqual(camera.home_calls, 1)
            self.assertEqual(
                repository.list_jobs()[0]["result"],
                "boat_confirmed",
            )

    def test_confirmed_distant_boat_stops_after_fallback_zoom_is_lost(
        self,
    ) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.8, 0.5),))
            source.closeup_results = [(), ()]
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(zoom_strategy="adaptive"),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(
                camera.locates,
                [(0.8, 0.5, 6), (0.5, 0.5, 3)],
            )
            self.assertEqual(camera.capture_calls, 0)
            self.assertEqual(camera.home_calls, 1)
            self.assertEqual(
                repository.list_jobs()[0]["result"],
                "target_lost",
            )

    def test_unconfirmed_proposal_does_not_use_fallback_zoom(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            proposal = detection(
                1,
                0.8,
                0.5,
                class_id=SMALL_TARGET_PROPOSAL_CLASS_ID,
            )
            source = SnapshotSource((proposal,))
            source.closeup_results = [()]
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(zoom_strategy="adaptive"),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(camera.locates, [(0.8, 0.5, 6)])
            self.assertEqual(camera.capture_calls, 0)
            self.assertEqual(camera.home_calls, 1)

    def test_proposal_budget_enforces_interval_and_hourly_cap(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource(())
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    proposal_minimum_interval_seconds=30,
                    proposal_maximum_verifications_per_hour=2,
                ),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(Path(directory)),
                camera_client=FakeCameraClient(source),
            )
            coordinator._proposal_verification_times.extend((100.0, 150.0))

            self.assertFalse(coordinator._proposal_budget_available(160.0))
            self.assertFalse(coordinator._proposal_budget_available(200.0))
            self.assertTrue(coordinator._proposal_budget_available(3_701.0))

    def test_nearby_proposal_fragments_merge_but_real_boats_stay_distinct(self) -> None:
        proposals = (
            detection(
                1,
                0.50,
                0.50,
                hits=12,
                class_id=SMALL_TARGET_PROPOSAL_CLASS_ID,
            ),
            detection(
                2,
                0.51,
                0.50,
                hits=8,
                class_id=SMALL_TARGET_PROPOSAL_CLASS_ID,
            ),
            detection(
                3,
                0.59,
                0.50,
                hits=6,
                class_id=SMALL_TARGET_PROPOSAL_CLASS_ID,
            ),
            detection(4, 0.505, 0.50, hits=4, class_id=8),
            detection(5, 0.51, 0.50, hits=4, class_id=8),
        )

        merged = _merge_proposal_fragments(proposals, 0.02)

        self.assertEqual([item.object_id for item in merged], [4, 5, 3])

    def test_reviewed_location_is_frozen_and_suppresses_new_track_ids(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            repository = PtzVerificationRepository(Path(directory))
            options = self.options(proposal_merge_radius=0.04)
            original = repository.observe(
                stream_id="stream-1",
                camera_id="camera-01",
                detection=detection(1, 0.50, 0.50),
                now=100.0,
                options=options,
            )
            job = repository.claim(
                stream_id="stream-1",
                camera_id="camera-01",
                target=original,
                now=101.0,
            )
            self.assertIsNotNone(job)
            assert job is not None
            with repository._connect() as connection:
                connection.execute(
                    "UPDATE targets SET x=0.53,y=0.52,vx=0.5,vy=-0.5 "
                    "WHERE target_id=?",
                    (original.target_id,),
                )
            repository.finish(
                job_id=job.job_id,
                result="boat_confirmed",
                error=None,
                home_returned=True,
                now=102.0,
                options=options,
            )

            reviewed_again = repository.observe(
                stream_id="stream-1",
                camera_id="camera-01",
                detection=detection(99, 0.515, 0.50),
                now=110.0,
                options=options,
                exclude_target_ids={original.target_id},
            )
            distinct_neighbour = repository.observe(
                stream_id="stream-1",
                camera_id="camera-01",
                detection=detection(100, 0.58, 0.50),
                now=110.0,
                options=options,
                exclude_target_ids={original.target_id},
            )

            self.assertEqual(reviewed_again.target_id, original.target_id)
            self.assertGreater(reviewed_again.cooldown_until, 110.0)
            self.assertNotEqual(distinct_neighbour.target_id, original.target_id)
            with repository._connect() as connection:
                row = connection.execute(
                    "SELECT x,y,vx,vy FROM targets WHERE target_id=?",
                    (original.target_id,),
                ).fetchone()
            assert row is not None
            self.assertAlmostEqual(float(row["x"]), 0.50)
            self.assertAlmostEqual(float(row["y"]), 0.50)
            self.assertEqual(float(row["vx"]), 0.0)
            self.assertEqual(float(row["vy"]), 0.0)

    def test_adaptive_zoom_uses_feedback_and_reduces_later_step(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.7, 0.5),))
            source.closeup_results = [
                (detection(101, 0.5, 0.5, width=0.08, height=0.05),),
                (detection(102, 0.5, 0.5, width=0.3, height=0.2),),
            ]
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(zoom_strategy="adaptive"),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual([item[2] for item in camera.locates], [6, 4])
            self.assertEqual(camera.capture_calls, 1)
            self.assertEqual(repository.list_jobs()[0]["result"], "boat_confirmed")

    def test_confirmed_boat_is_prioritized_over_older_proposal(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            proposal = detection(
                1,
                0.2,
                0.5,
                hits=100,
                class_id=SMALL_TARGET_PROPOSAL_CLASS_ID,
            )
            boat = detection(2, 0.8, 0.5, hits=4, class_id=8)
            source = SnapshotSource((proposal, boat))
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(zoom_strategy="adaptive"),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(len(camera.locates), 1)
            self.assertAlmostEqual(camera.locates[0][0], 0.8)

    def test_adaptive_zoom_stops_as_soon_as_confirmed_box_is_large(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.7, 0.5),))
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(zoom_strategy="adaptive"),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(len(camera.locates), 1)
            self.assertEqual(camera.autofocus_calls, 1)
            self.assertEqual(camera.capture_calls, 1)

    def test_adaptive_zoom_does_not_capture_undersized_non_growing_boat(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.7, 0.5),))
            source.closeup_results = [
                (detection(101, 0.5, 0.5, width=0.031, height=0.021),),
                (detection(102, 0.5, 0.5, width=0.032, height=0.022),),
            ]
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(zoom_strategy="adaptive"),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(len(camera.locates), 2)
            self.assertEqual(camera.capture_calls, 0)
            self.assertEqual(
                repository.list_jobs()[0]["result"],
                "insufficient_resolution",
            )

    def test_adaptive_zoom_never_confirms_non_growing_motion_proposal(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            proposal = detection(
                1,
                0.7,
                0.5,
                class_id=SMALL_TARGET_PROPOSAL_CLASS_ID,
            )
            source = SnapshotSource((proposal,))
            source.closeup_results = [
                (
                    detection(
                        101,
                        0.5,
                        0.5,
                        width=0.031,
                        height=0.021,
                        class_id=SMALL_TARGET_PROPOSAL_CLASS_ID,
                    ),
                ),
                (
                    detection(
                        102,
                        0.5,
                        0.5,
                        width=0.032,
                        height=0.022,
                        class_id=SMALL_TARGET_PROPOSAL_CLASS_ID,
                    ),
                ),
            ]
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(zoom_strategy="adaptive"),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(len(camera.locates), 2)
            self.assertEqual(camera.capture_calls, 0)
            self.assertEqual(
                repository.list_jobs()[0]["result"],
                "candidate_not_confirmed",
            )

    def test_reacquire_groups_adjacent_near_center_boats(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource(())
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(reacquire_center_radius=0.25),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(Path(directory)),
                camera_client=FakeCameraClient(source),
            )
            reference = detection(1, 0.7, 0.5)
            candidates = (
                detection(2, 0.48, 0.5, width=0.08, height=0.05),
                detection(3, 0.52, 0.5, width=0.08, height=0.05),
            )

            selected = coordinator._nearest_centered(candidates, reference)

            self.assertIsNotNone(selected)
            assert selected is not None
            self.assertAlmostEqual(selected.rectangle.center[0], 0.5)
            self.assertAlmostEqual(selected.rectangle.width, 0.08)

    def test_reacquire_rejects_two_separated_competing_groups(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource(())
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    reacquire_strict_center_radius=0.30,
                    reacquire_center_radius=0.30,
                    reacquire_cluster_radius=0.12,
                ),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(Path(directory)),
                camera_client=FakeCameraClient(source),
            )
            reference = detection(1, 0.7, 0.5)
            candidates = (
                detection(2, 0.25, 0.5, width=0.08, height=0.05),
                detection(3, 0.75, 0.5, width=0.08, height=0.05),
            )

            self.assertIsNone(
                coordinator._nearest_centered(candidates, reference)
            )

    def test_adjacent_boat_group_is_zoomed_until_individuals_are_large(
        self,
    ) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.75, 0.5),))
            source.closeup_results = [
                (
                    detection(101, 0.46, 0.5, width=0.06, height=0.04),
                    detection(102, 0.54, 0.5, width=0.06, height=0.04),
                ),
                (
                    detection(201, 0.35, 0.5, width=0.30, height=0.20),
                    detection(202, 0.65, 0.5, width=0.30, height=0.20),
                ),
            ]
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    reacquire_cluster_radius=0.18,
                ),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,
            )

            coordinator.run_once()

            self.assertEqual(len(camera.locates), 2)
            self.assertAlmostEqual(camera.locates[1][0], 0.5)
            self.assertGreater(camera.locates[1][2], 0)
            self.assertEqual(camera.capture_calls, 1)
            self.assertEqual(
                repository.list_jobs()[0]["result"],
                "boat_confirmed",
            )

    def test_competing_groups_block_confirmed_target_fallback_zoom(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.75, 0.5),))
            source.closeup_results = [
                (
                    detection(101, 0.25, 0.5, width=0.08, height=0.05),
                    detection(102, 0.75, 0.5, width=0.08, height=0.05),
                ),
            ]
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    reacquire_strict_center_radius=0.30,
                    reacquire_center_radius=0.30,
                    reacquire_cluster_radius=0.12,
                ),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,
            )

            coordinator.run_once()

            self.assertEqual(len(camera.locates), 1)
            self.assertEqual(camera.capture_calls, 0)
            self.assertEqual(camera.home_calls, 1)
            self.assertEqual(
                repository.list_jobs()[0]["result"],
                "target_ambiguous",
            )

    def test_reacquire_prefers_continuous_shape_when_centers_are_close(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource(())
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(reacquire_center_radius=0.25),
                snapshot_provider=source.snapshot,
                repository=PtzVerificationRepository(Path(directory)),
                camera_client=FakeCameraClient(source),
            )
            reference = detection(
                1,
                0.7,
                0.5,
                width=0.06,
                height=0.02,
            )
            wrong_shape = detection(
                2,
                0.50,
                0.5,
                width=0.04,
                height=0.08,
            )
            continuous = detection(
                3,
                0.54,
                0.5,
                width=0.12,
                height=0.04,
            )

            selected = coordinator._nearest_centered(
                (wrong_shape, continuous),
                reference,
            )

            self.assertIsNotNone(selected)
            assert selected is not None
            self.assertEqual(selected.object_id, continuous.object_id)

    def test_adaptive_zoom_can_capture_already_large_boat_without_motion(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource(
                (detection(1, 0.5, 0.5, width=0.3, height=0.2),)
            )
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(zoom_strategy="adaptive"),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(camera.locates, [])
            self.assertEqual(camera.home_calls, 0)
            self.assertEqual(camera.capture_calls, 1)
            self.assertTrue(repository.list_jobs()[0]["home_returned"])

    def test_adaptive_zoom_recenters_large_edge_boat_without_extra_zoom(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource(
                (detection(1, 0.8, 0.5, width=0.3, height=0.2),)
            )
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(zoom_strategy="adaptive"),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,
            )

            coordinator.run_once()

            self.assertEqual([item[2] for item in camera.locates], [0])
            self.assertEqual(camera.capture_calls, 1)
            self.assertEqual(
                repository.list_jobs()[0]["result"],
                "boat_confirmed",
            )

    def test_fixed_zoom_strategy_preserves_configured_steps(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.7, 0.5),))
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(
                    zoom_strategy="fixed",
                    zoom_steps=(3, 5),
                ),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual([item[2] for item in camera.locates], [3, 5])
            self.assertEqual(camera.capture_calls, 1)

    def test_confirmed_target_is_captured_and_not_revisited_after_home(self) -> None:
        with self.subTest("repository and coordinator"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                source = SnapshotSource((detection(1, 0.72, 0.55),))
                camera = FakeCameraClient(source)
                repository = PtzVerificationRepository(Path(directory))
                coordinator = PtzVerificationCoordinator(
                    stream_id="stream-1",
                    options=self.options(),
                    snapshot_provider=source.snapshot,
                    repository=repository,
                    camera_client=camera,  # type: ignore[arg-type]
                )

                job_id = coordinator.run_once()
                self.assertIsNotNone(job_id)
                self.assertEqual(camera.capture_calls, 1)
                self.assertEqual(camera.home_calls, 1)
                self.assertGreaterEqual(camera.lease_acquires, 4)
                jobs = repository.list_jobs()
                self.assertEqual(jobs[0]["result"], "boat_confirmed")
                self.assertTrue(jobs[0]["home_returned"])
                self.assertEqual(len(jobs[0]["images"]), 1)

                coordinator.run_once()
                self.assertEqual(camera.capture_calls, 1)
                self.assertEqual(len(repository.list_jobs()), 1)

    def test_intermediate_zoom_can_follow_proposal_until_boat_is_confirmed(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            proposal = detection(
                1,
                0.72,
                0.55,
                class_id=SMALL_TARGET_PROPOSAL_CLASS_ID,
            )
            source = SnapshotSource((proposal,))
            source.closeup_results = [
                (
                    detection(
                        101,
                        0.5,
                        0.5,
                        width=0.12,
                        height=0.08,
                        class_id=SMALL_TARGET_PROPOSAL_CLASS_ID,
                    ),
                ),
                (detection(102, 0.5, 0.5, width=0.3, height=0.2),),
            ]
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(len(camera.locates), 2)
            self.assertEqual(camera.capture_calls, 1)
            self.assertEqual(
                repository.list_jobs()[0]["result"],
                "boat_confirmed",
            )

    def test_stale_closeup_frame_after_home_enters_recovery_failed(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.72, 0.55),))
            camera = FakeCameraClient(source)
            camera.update_source_on_home = False
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()
            locate_count = len(camera.locates)
            coordinator.run_once()

            self.assertEqual(len(camera.locates), locate_count)
            self.assertEqual(len(repository.list_jobs()), 1)
            self.assertEqual(
                repository.list_jobs()[0]["result"],
                "recovery_failed",
            )
            self.assertTrue(coordinator.is_busy)

    def test_candidate_must_be_observed_repeatedly_before_ptz_claim(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.70, 0.5),))
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(minimum_target_observations=3),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            self.assertIsNone(coordinator.run_once())
            source.return_home((detection(1, 0.705, 0.5),))
            self.assertIsNone(coordinator.run_once())
            source.return_home((detection(1, 0.710, 0.5),))
            self.assertIsNotNone(coordinator.run_once())

            self.assertEqual(camera.capture_calls, 1)

    def test_unconfirmed_candidate_returns_home_then_next_candidate_runs(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            first = detection(1, 0.35, 0.55, width=0.04)
            second = detection(2, 0.75, 0.55, width=0.03)
            source = SnapshotSource((first, second))
            source.closeup_confirmed = False
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()
            source.return_home()
            coordinator.run_once()
            jobs = repository.list_jobs()
            self.assertEqual(len(jobs), 2)
            self.assertTrue(
                all(item["result"] == "target_lost" for item in jobs)
            )
            self.assertEqual(camera.home_calls, 2)

    def test_shutdown_interrupts_reacquire_and_completes_job_after_home(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.7, 0.5),))
            source.closeup_confirmed = False
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(reacquire_timeout_seconds=5),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.start()
            deadline = time.monotonic() + 2
            while camera.home_calls < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            source.return_home()
            while not camera.locates and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(camera.locates)

            coordinator.shutdown(timeout=2)

            self.assertEqual(coordinator.state, "stopped")
            # Interrupt the active operation, then stop once more immediately
            # before the final shutdown HOME command.
            self.assertEqual(camera.stop_calls, 2)
            # Startup HOME, interrupted-job HOME, and final shutdown HOME.
            self.assertEqual(camera.home_calls, 3)
            jobs = repository.list_jobs()
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["result"], "target_lost")
            self.assertTrue(jobs[0]["home_returned"])
            self.assertIn("shutdown", jobs[0]["error"])

    def test_shutdown_reasserts_home_while_idle(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource(())
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.start()
            deadline = time.monotonic() + 2
            while camera.home_calls < 1 and time.monotonic() < deadline:
                time.sleep(0.01)

            coordinator.shutdown(timeout=2)

            self.assertEqual(coordinator.state, "stopped")
            self.assertEqual(camera.home_calls, 2)
            self.assertGreaterEqual(camera.stop_calls, 1)
            self.assertGreaterEqual(camera.lease_acquires, 2)
            self.assertGreaterEqual(camera.lease_releases, 2)

    def test_shutdown_reserves_time_for_final_home(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource(())
            camera = FakeCameraClient(source)
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(command_timeout_seconds=20),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )
            thread = Mock()
            coordinator._thread = thread

            coordinator.shutdown(timeout=70)

            thread.join.assert_called_once_with(timeout=22.0)
            self.assertEqual(camera.home_calls, 1)
            self.assertEqual(camera.lease_releases, 1)

    def test_failed_home_stops_new_verification_work(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.7, 0.5),))
            camera = FakeCameraClient(source)
            camera.fail_home = True
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()
            self.assertEqual(
                repository.list_jobs()[0]["result"],
                "recovery_failed",
            )
            source.return_home((detection(2, 0.3, 0.5),))
            self.assertIsNone(coordinator.run_once())

    def test_locate_error_still_forces_home_and_clears_busy_state(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = SnapshotSource((detection(1, 0.7, 0.5),))
            camera = FakeCameraClient(source)
            camera.fail_locate = True
            repository = PtzVerificationRepository(Path(directory))
            coordinator = PtzVerificationCoordinator(
                stream_id="stream-1",
                options=self.options(),
                snapshot_provider=source.snapshot,
                repository=repository,
                camera_client=camera,  # type: ignore[arg-type]
            )

            coordinator.run_once()

            self.assertEqual(camera.home_calls, 1)
            self.assertFalse(coordinator.is_busy)
            self.assertEqual(repository.list_jobs()[0]["result"], "target_lost")

    def test_close_targets_in_same_frame_keep_distinct_memories(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            repository = PtzVerificationRepository(Path(directory))
            options = self.options(dedup_base_radius=0.03)
            excluded: set[str] = set()
            first = repository.observe(
                stream_id="stream-1",
                camera_id="camera-01",
                detection=detection(1, 0.50, 0.50),
                now=100.0,
                options=options,
                exclude_target_ids=excluded,
            )
            excluded.add(first.target_id)
            second = repository.observe(
                stream_id="stream-1",
                camera_id="camera-01",
                detection=detection(2, 0.515, 0.50),
                now=100.0,
                options=options,
                exclude_target_ids=excluded,
            )

            self.assertNotEqual(first.target_id, second.target_id)

    def test_restart_recovery_closes_abandoned_job_after_home(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            repository = PtzVerificationRepository(Path(directory))
            options = self.options(lost_retry_seconds=180)
            target = repository.observe(
                stream_id="stream-1",
                camera_id="camera-01",
                detection=detection(1, 0.5, 0.5),
                now=100.0,
                options=options,
            )
            job = repository.claim(
                stream_id="stream-1",
                camera_id="camera-01",
                target=target,
                now=100.0,
            )
            assert job is not None
            repository.mark_running(job.job_id, 101.0)

            recovered = repository.recover_incomplete(
                stream_id="stream-1",
                camera_id="camera-01",
                now=200.0,
                options=options,
            )

            self.assertEqual(recovered, 1)
            item = repository.get_job(job.job_id)
            self.assertEqual(item["result"], "target_lost")
            self.assertTrue(item["home_returned"])
            refreshed = repository.observe(
                stream_id="stream-1",
                camera_id="camera-01",
                detection=detection(99, 0.502, 0.5),
                now=201.0,
                options=options,
            )
            self.assertIsNone(
                repository.claim(
                    stream_id="stream-1",
                    camera_id="camera-01",
                    target=refreshed,
                    now=201.0,
                )
            )

    def test_completion_cursor_is_created_only_when_job_finishes(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            repository = PtzVerificationRepository(Path(directory))
            options = self.options()
            target = repository.observe(
                stream_id="stream-1",
                camera_id="camera-01",
                detection=detection(1, 0.5, 0.5),
                now=100.0,
                options=options,
            )
            job = repository.claim(
                stream_id="stream-1",
                camera_id="camera-01",
                target=target,
                now=100.0,
            )
            assert job is not None

            self.assertEqual(repository.list_jobs(), [])
            repository.finish(
                job_id=job.job_id,
                result="boat_confirmed",
                error=None,
                home_returned=True,
                now=101.0,
                options=options,
            )
            completed = repository.list_jobs(after_sequence=0)

            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]["sequence"], 1)
            self.assertEqual(completed[0]["creation_sequence"], job.sequence)
            self.assertEqual(
                repository.list_jobs(after_sequence=completed[0]["sequence"]),
                [],
            )

    def test_completion_cursor_follows_finish_order_across_cameras(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            repository = PtzVerificationRepository(Path(directory))
            options = self.options()

            def claim(camera_id: str, x: float):
                target = repository.observe(
                    stream_id=f"stream-{camera_id}",
                    camera_id=camera_id,
                    detection=detection(1, x, 0.5),
                    now=100.0,
                    options=options,
                )
                job = repository.claim(
                    stream_id=f"stream-{camera_id}",
                    camera_id=camera_id,
                    target=target,
                    now=100.0,
                )
                assert job is not None
                return job

            first_created = claim("camera-01", 0.2)
            second_created = claim("camera-02", 0.8)
            repository.finish(
                job_id=second_created.job_id,
                result="boat_confirmed",
                error=None,
                home_returned=True,
                now=101.0,
                options=options,
            )
            first_batch = repository.list_jobs(after_sequence=0)
            repository.finish(
                job_id=first_created.job_id,
                result="boat_confirmed",
                error=None,
                home_returned=True,
                now=102.0,
                options=options,
            )
            second_batch = repository.list_jobs(
                after_sequence=first_batch[0]["sequence"]
            )

            self.assertEqual(first_batch[0]["job_id"], second_created.job_id)
            self.assertEqual(second_batch[0]["job_id"], first_created.job_id)
            self.assertGreater(
                second_batch[0]["sequence"],
                first_batch[0]["sequence"],
            )

    def test_repository_migrates_existing_target_database(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "ptz-verification.sqlite3"
            with sqlite3.connect(database_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE targets (
                        target_id TEXT PRIMARY KEY,
                        stream_id TEXT NOT NULL,
                        camera_id TEXT NOT NULL,
                        source_track_id INTEGER NOT NULL,
                        x REAL NOT NULL,y REAL NOT NULL,
                        width REAL NOT NULL,height REAL NOT NULL,
                        vx REAL NOT NULL DEFAULT 0,vy REAL NOT NULL DEFAULT 0,
                        first_seen REAL NOT NULL,last_seen REAL NOT NULL,
                        cooldown_until REAL NOT NULL DEFAULT 0,
                        last_result TEXT,attempts INTEGER NOT NULL DEFAULT 0,
                        updated_at REAL NOT NULL
                    )
                    """
                )

            repository = PtzVerificationRepository(root)
            with repository._connect() as connection:
                target_columns = {
                    str(row["name"])
                    for row in connection.execute("PRAGMA table_info(targets)")
                }
                event_table = connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type='table' AND name='completion_events'
                    """
                ).fetchone()

            self.assertIn("observations", target_columns)
            self.assertIsNotNone(event_table)

    def test_multiple_observations_preserve_moving_target_identity(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            repository = PtzVerificationRepository(Path(directory))
            options = self.options(minimum_target_observations=3)
            first = repository.observe(
                stream_id="stream-1",
                camera_id="camera-01",
                detection=detection(1, 0.20, 0.5),
                now=100.0,
                options=options,
            )
            second = repository.observe(
                stream_id="stream-1",
                camera_id="camera-01",
                detection=detection(1, 0.21, 0.5),
                now=101.0,
                options=options,
            )
            third = repository.observe(
                stream_id="stream-1",
                camera_id="camera-01",
                detection=detection(1, 0.22, 0.5),
                now=102.0,
                options=options,
            )
            moved = repository.observe(
                stream_id="stream-1",
                camera_id="camera-01",
                detection=detection(99, 0.28, 0.5),
                now=112.0,
                options=options,
            )

            self.assertEqual(first.target_id, second.target_id)
            self.assertEqual(second.target_id, third.target_id)
            self.assertEqual(third.observations, 3)
            self.assertEqual(moved.target_id, first.target_id)

    def test_camera_client_authenticates_and_downloads_same_origin_artifact(self) -> None:
        class Response:
            def __init__(self, content: bytes, content_type: str) -> None:
                self.content = content
                self.headers = Message()
                self.headers["Content-Type"] = content_type

            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return self.content

        requests = []
        responses = iter(
            (
                Response(json.dumps({"command_id": "cmd-1"}).encode(), "application/json"),
                Response(
                    json.dumps(
                        {
                            "state": "completed",
                            "result": {"download_url": "/v1/artifacts/image-1"},
                        }
                    ).encode(),
                    "application/json",
                ),
                Response(b"jpeg-bytes", "image/jpeg"),
            )
        )

        def opener(request, **_kwargs):
            requests.append(request)
            return next(responses)

        client = CameraControlClient(
            self.options(camera_control_url="http://camera-control:8080"),
            api_key="internal-secret",
            opener=opener,
        )

        content, mime_type, _captured_at = client.capture()

        self.assertEqual(content, b"jpeg-bytes")
        self.assertEqual(mime_type, "image/jpeg")
        self.assertTrue(
            all(
                request.get_header("X-camera-control-key")
                == "internal-secret"
                for request in requests
            )
        )

    def test_camera_client_rejects_cross_origin_artifact(self) -> None:
        class Response:
            headers = Message()

            def __init__(self, payload: dict[str, object]) -> None:
                self.content = json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return self.content

        responses = iter(
            (
                Response({"command_id": "cmd-1"}),
                Response(
                    {
                        "state": "completed",
                        "result": {"download_url": "https://evil.invalid/x"},
                    }
                ),
            )
        )
        client = CameraControlClient(
            self.options(camera_control_url="http://camera-control:8080"),
            api_key="internal-secret",
            opener=lambda *_args, **_kwargs: next(responses),
        )

        with self.assertRaisesRegex(RuntimeError, "不受信任"):
            client.capture()

    def test_camera_client_renews_and_releases_control_lease(self) -> None:
        class Response:
            def __init__(self, payload: dict[str, object] | None) -> None:
                self.content = (
                    json.dumps(payload).encode() if payload is not None else b""
                )

            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return self.content

        responses = iter(
            (
                Response({"token": "lease-token"}),
                Response({"token": "lease-token"}),
                Response(None),
            )
        )
        requests: list[Request] = []

        def opener(request: Request, **_kwargs):
            requests.append(request)
            return next(responses)

        client = CameraControlClient(
            self.options(camera_control_url="http://camera-control:8080"),
            api_key="internal-secret",
            opener=opener,
        )

        client.acquire_lease("rtsp:test", 60)
        client.acquire_lease("rtsp:test", 60)
        client.release_lease()

        self.assertIsNone(requests[0].get_header("X-camera-control-lease"))
        self.assertEqual(
            requests[1].get_header("X-camera-control-lease"),
            "lease-token",
        )
        self.assertEqual(
            requests[2].get_header("X-camera-control-lease"),
            "lease-token",
        )


if __name__ == "__main__":
    unittest.main()
