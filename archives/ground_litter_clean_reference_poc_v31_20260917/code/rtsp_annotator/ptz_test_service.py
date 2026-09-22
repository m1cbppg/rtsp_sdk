from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class TestServiceSettings:
    source_path: Path
    publish_url: str
    api_key: str
    camera_id: str = "virtual-river-01"
    output_width: int = 1920
    output_height: int = 1080
    output_fps: float = 10.0
    source_start_seconds: float = 0.0
    source_end_seconds: float | None = None
    zoom_gain_per_delta: float = math.log(2.0) / 2.0
    maximum_zoom: float = 64.0
    artifact_root: Path = Path("/app/ptz-test-data/artifacts")
    ffmpeg_path: str = "ffmpeg"
    gstreamer_path: str = "gst-launch-1.0"
    encoder: str = "libx264"
    mirror_real_camera: bool = False
    real_camera_control_url: str = "http://camera-control:8080"
    real_camera_id: str = "river-ptz-01"
    real_camera_control_key: str = ""

    @classmethod
    def from_environment(cls) -> "TestServiceSettings":
        end_value = os.getenv("PTZ_TEST_SOURCE_END_SECONDS", "").strip()
        return cls(
            source_path=Path(
                os.getenv("PTZ_TEST_SOURCE", "/fixtures/boat.mp4")
            ),
            publish_url=os.environ["PTZ_TEST_PUBLISH_URL"],
            api_key=os.environ["PTZ_TEST_CONTROL_KEY"],
            camera_id=os.getenv("PTZ_TEST_CAMERA_ID", "virtual-river-01"),
            output_width=int(os.getenv("PTZ_TEST_WIDTH", "1920")),
            output_height=int(os.getenv("PTZ_TEST_HEIGHT", "1080")),
            output_fps=float(os.getenv("PTZ_TEST_FPS", "10")),
            source_start_seconds=float(
                os.getenv("PTZ_TEST_SOURCE_START_SECONDS", "0")
            ),
            source_end_seconds=(float(end_value) if end_value else None),
            zoom_gain_per_delta=float(
                os.getenv(
                    "PTZ_TEST_ZOOM_GAIN_PER_DELTA",
                    str(math.log(2.0) / 2.0),
                )
            ),
            maximum_zoom=float(os.getenv("PTZ_TEST_MAXIMUM_ZOOM", "64")),
            artifact_root=Path(
                os.getenv(
                    "PTZ_TEST_ARTIFACT_ROOT",
                    "/app/ptz-test-data/artifacts",
                )
            ),
            ffmpeg_path=os.getenv("PTZ_TEST_FFMPEG", "ffmpeg"),
            gstreamer_path=os.getenv(
                "PTZ_TEST_GSTREAMER",
                "gst-launch-1.0",
            ),
            encoder=os.getenv("PTZ_TEST_ENCODER", "libx264"),
            mirror_real_camera=os.getenv(
                "PTZ_TEST_MIRROR_REAL_CAMERA", "false"
            ).lower()
            in {"1", "true", "yes"},
            real_camera_control_url=os.getenv(
                "REAL_CAMERA_CONTROL_URL",
                "http://camera-control:8080",
            ),
            real_camera_id=os.getenv(
                "REAL_CAMERA_ID", "river-ptz-01"
            ),
            real_camera_control_key=os.getenv(
                "REAL_CAMERA_CONTROL_API_KEY", ""
            ),
        )

    def validate(self) -> None:
        if not self.source_path.is_file():
            raise ValueError(f"测试视频不存在: {self.source_path}")
        if not self.publish_url.startswith(("rtsp://", "rtsps://")):
            raise ValueError("PTZ_TEST_PUBLISH_URL必须是RTSP地址")
        if len(self.api_key) < 16:
            raise ValueError("PTZ_TEST_CONTROL_KEY至少需要16个字符")
        if self.output_width < 320 or self.output_height < 180:
            raise ValueError("测试输出分辨率过低")
        if not 0.1 <= self.output_fps <= 60:
            raise ValueError("PTZ_TEST_FPS必须在[0.1,60]")
        if self.source_start_seconds < 0:
            raise ValueError("测试片段起始时间不能为负数")
        if (
            self.source_end_seconds is not None
            and self.source_end_seconds <= self.source_start_seconds
        ):
            raise ValueError("测试片段结束时间必须晚于起始时间")
        if not 0.01 <= self.zoom_gain_per_delta <= 1.0:
            raise ValueError("虚拟变焦增益必须在[0.01,1.0]")
        if not 1.0 <= self.maximum_zoom <= 256.0:
            raise ValueError("最大虚拟变焦倍率必须在[1,256]")
        if self.mirror_real_camera and not self.real_camera_control_key:
            raise ValueError("镜像真实摄像头时必须配置控制密钥")


@dataclass(slots=True)
class VirtualViewport:
    zoom_gain_per_delta: float
    maximum_zoom: float
    center_x: float = 0.5
    center_y: float = 0.5
    zoom: float = 1.0

    def locate(self, x: float, y: float, zoom_delta: int) -> None:
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("x/y必须在[0,1]")
        if not -16 <= zoom_delta <= 16:
            raise ValueError("zoom_delta必须在[-16,16]")
        view_size = 1.0 / self.zoom
        self.center_x += (x - 0.5) * view_size
        self.center_y += (y - 0.5) * view_size
        self.zoom = min(
            max(
                self.zoom * math.exp(self.zoom_gain_per_delta * zoom_delta),
                1.0,
            ),
            self.maximum_zoom,
        )
        half = 0.5 / self.zoom
        self.center_x = min(max(self.center_x, half), 1.0 - half)
        self.center_y = min(max(self.center_y, half), 1.0 - half)

    def home(self) -> None:
        self.center_x = 0.5
        self.center_y = 0.5
        self.zoom = 1.0

    def render(
        self,
        frame: np.ndarray,
        *,
        width: int,
        height: int,
    ) -> np.ndarray:
        import cv2

        source_height, source_width = frame.shape[:2]
        view_width = max(int(round(source_width / self.zoom)), 2)
        view_height = max(int(round(source_height / self.zoom)), 2)
        center_x = int(round(self.center_x * source_width))
        center_y = int(round(self.center_y * source_height))
        left = min(max(center_x - view_width // 2, 0), source_width - view_width)
        top = min(max(center_y - view_height // 2, 0), source_height - view_height)
        crop = frame[top : top + view_height, left : left + view_width]
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


class LocateRequest(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    zoom_delta: int = Field(default=0, ge=-16, le=16)
    timeout_seconds: float = Field(default=8, ge=0.5, le=60)
    autofocus: bool = False


class TimeoutRequest(BaseModel):
    timeout_seconds: float = Field(default=8, ge=0.5, le=60)


class CaptureRequest(TimeoutRequest):
    quality: int = Field(default=1, ge=1, le=6)


class LeaseRequest(BaseModel):
    owner: str = Field(min_length=1, max_length=200)
    ttl_seconds: float = Field(default=120, ge=10, le=600)


@dataclass(slots=True)
class _Command:
    command_id: str
    camera_id: str
    action: str
    state: str = "queued"
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "camera_id": self.camera_id,
            "action": self.action,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
        }


class _CommandExecutor:
    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="virtual-ptz")
        self._commands: dict[str, _Command] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        camera_id: str,
        action: str,
        operation: Callable[[], dict[str, Any] | None],
    ) -> dict[str, Any]:
        command = _Command(uuid.uuid4().hex, camera_id, action)
        with self._lock:
            self._commands[command.command_id] = command
        self._pool.submit(self._run, command.command_id, operation)
        return command.payload()

    def _run(
        self,
        command_id: str,
        operation: Callable[[], dict[str, Any] | None],
    ) -> None:
        with self._lock:
            command = self._commands[command_id]
            command.state = "running"
            command.started_at = _utc_now()
        try:
            result = operation()
        except Exception as exc:
            with self._lock:
                command = self._commands[command_id]
                command.state = "failed"
                command.error = f"{type(exc).__name__}: {exc}"
                command.finished_at = _utc_now()
        else:
            with self._lock:
                command = self._commands[command_id]
                command.state = "completed"
                command.result = result
                command.finished_at = _utc_now()

    def get(self, command_id: str) -> dict[str, Any]:
        with self._lock:
            if command_id not in self._commands:
                raise KeyError(command_id)
            return self._commands[command_id].payload()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class _ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, tuple[Path, str, str]] = {}
        self._lock = threading.Lock()

    def store(self, content: bytes, mime_type: str, kind: str) -> dict[str, Any]:
        artifact_id = uuid.uuid4().hex
        suffix = ".jpg" if mime_type == "image/jpeg" else ".bin"
        path = self.root / f"{artifact_id}{suffix}"
        path.write_bytes(content)
        with self._lock:
            self._items[artifact_id] = (path, mime_type, kind)
        return {
            "artifact_id": artifact_id,
            "kind": kind,
            "mime_type": mime_type,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "download_url": f"/v1/artifacts/{artifact_id}",
        }

    def get(self, artifact_id: str) -> tuple[Path, str, str]:
        with self._lock:
            try:
                return self._items[artifact_id]
            except KeyError as exc:
                raise KeyError(artifact_id) from exc


class _MirrorCameraClient:
    def __init__(self, settings: TestServiceSettings) -> None:
        self.base_url = settings.real_camera_control_url.rstrip("/") + "/"
        self.camera_id = settings.real_camera_id
        self.api_key = settings.real_camera_control_key

    def _json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 20.0,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(
            urljoin(self.base_url, path.lstrip("/")),
            data=body,
            method=method,
            headers={
                "X-Camera-Control-Key": self.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except HTTPError as exc:
            raise RuntimeError(f"真实camera_control HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError("无法连接真实camera_control") from exc

    def command(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        submitted = self._json(
            "POST",
            f"/v1/cameras/{self.camera_id}/commands/{action}",
            payload,
            timeout,
        )
        command_id = str(submitted["command_id"])
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = self._json("GET", f"/v1/commands/{command_id}", timeout=5)
            if current.get("state") == "completed":
                return current
            if current.get("state") == "failed":
                raise RuntimeError(str(current.get("error") or "真实摄像头命令失败"))
            time.sleep(0.1)
        raise RuntimeError("真实摄像头命令等待超时")

    def download(self, path: str, timeout: float = 10.0) -> tuple[bytes, str]:
        request = Request(
            urljoin(self.base_url, path.lstrip("/")),
            headers={"X-Camera-Control-Key": self.api_key},
        )
        with urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get_content_type()

    def stop(self) -> None:
        self._json(
            "POST",
            f"/v1/cameras/{self.camera_id}/stop",
            {},
            5,
        )


class _RawRtspPublisher:
    def __init__(self, settings: TestServiceSettings) -> None:
        self.settings = settings
        self._process: subprocess.Popen[bytes] | None = None
        self.backend: str | None = None

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        ffmpeg = shutil.which(self.settings.ffmpeg_path)
        gstreamer = shutil.which(self.settings.gstreamer_path)
        if ffmpeg is not None:
            command = self._ffmpeg_command(ffmpeg)
            self.backend = "ffmpeg"
        elif gstreamer is not None:
            command = self._gstreamer_command(gstreamer)
            self.backend = "gstreamer"
        else:
            raise RuntimeError(
                "虚拟RTSP发布失败：容器内既没有ffmpeg，也没有gst-launch-1.0"
            )
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def _ffmpeg_command(self, executable: str) -> list[str]:
        return [
            executable,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{self.settings.output_width}x{self.settings.output_height}",
            "-r",
            str(self.settings.output_fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            self.settings.encoder,
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(max(int(round(self.settings.output_fps)), 1)),
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            self.settings.publish_url,
        ]

    def _gstreamer_command(self, executable: str) -> list[str]:
        fps = Fraction(self.settings.output_fps).limit_denominator(1_000)
        keyframe_interval = max(int(round(self.settings.output_fps)), 1)
        return [
            executable,
            "-q",
            "fdsrc",
            "fd=0",
            "!",
            "rawvideoparse",
            f"width={self.settings.output_width}",
            f"height={self.settings.output_height}",
            "format=bgr",
            f"framerate={fps.numerator}/{fps.denominator}",
            "!",
            "videoconvert",
            "!",
            "nvvideoconvert",
            "gpu-id=0",
            "!",
            "video/x-raw(memory:NVMM),format=I420",
            "!",
            "nvv4l2h264enc",
            "bitrate=4000000",
            f"iframeinterval={keyframe_interval}",
            f"idrinterval={keyframe_interval}",
            "insert-sps-pps=true",
            "!",
            "h264parse",
            "config-interval=-1",
            "!",
            "video/x-h264,stream-format=byte-stream,alignment=au",
            "!",
            "rtspclientsink",
            f"location={self.settings.publish_url}",
            "protocols=tcp",
            "latency=0",
        ]

    def write(self, frame: np.ndarray) -> None:
        self.start()
        assert self._process is not None
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            self.stop()
            raise

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


class VirtualPtzTestController:
    def __init__(
        self,
        settings: TestServiceSettings,
        *,
        publisher: _RawRtspPublisher | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.viewport = VirtualViewport(
            settings.zoom_gain_per_delta,
            settings.maximum_zoom,
        )
        self.publisher = publisher or _RawRtspPublisher(settings)
        self.artifacts = _ArtifactStore(settings.artifact_root)
        self.mirror = (
            _MirrorCameraClient(settings)
            if settings.mirror_real_camera
            else None
        )
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_frame: np.ndarray | None = None
        self._last_action = "home"
        self._last_frame_at: float | None = None
        self._last_publish_at: float | None = None
        self._last_error: str | None = None
        self._history: list[dict[str, Any]] = []
        self._started_at = _utc_now()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._render_loop,
            name="virtual-ptz-render",
            daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self.publisher.stop()

    def locate(self, request: LocateRequest) -> dict[str, Any]:
        before = self.status()
        with self._lock:
            self.viewport.locate(request.x, request.y, request.zoom_delta)
            self._last_action = "locate"
            current = self._viewport_payload()
        mirror_result = None
        if self.mirror is not None:
            mirror_result = self.mirror.command(
                "locate",
                request.model_dump(),
                request.timeout_seconds,
            )
        self._record("locate", request.model_dump(), before, current)
        return {"virtual_status": current, "real_command": mirror_result}

    def home(self, request: TimeoutRequest) -> dict[str, Any]:
        before = self.status()
        with self._lock:
            self.viewport.home()
            self._last_action = "home"
            current = self._viewport_payload()
        mirror_result = None
        if self.mirror is not None:
            mirror_result = self.mirror.command(
                "home",
                request.model_dump(),
                request.timeout_seconds,
            )
        self._record("home", request.model_dump(), before, current)
        return {"virtual_status": current, "real_command": mirror_result}

    def autofocus(self, request: TimeoutRequest) -> dict[str, Any]:
        mirror_result = None
        if self.mirror is not None:
            mirror_result = self.mirror.command(
                "autofocus",
                request.model_dump(),
                request.timeout_seconds,
            )
        self._record("autofocus", request.model_dump(), self.status(), self.status())
        return {"virtual_status": self.status(), "real_command": mirror_result}

    def capture(self, request: CaptureRequest) -> dict[str, Any]:
        import cv2

        with self._lock:
            if self._latest_frame is None:
                raise RuntimeError("虚拟RTSP尚未产生画面")
            ok, encoded = cv2.imencode(
                ".jpg",
                self._latest_frame,
                [cv2.IMWRITE_JPEG_QUALITY, max(95 - request.quality * 5, 65)],
            )
        if not ok:
            raise RuntimeError("虚拟截图JPEG编码失败")
        virtual_artifact = self.artifacts.store(
            bytes(encoded), "image/jpeg", "virtual_boat_capture"
        )
        result: dict[str, Any] = dict(virtual_artifact)
        if self.mirror is not None:
            real_command = self.mirror.command(
                "capture",
                request.model_dump(),
                request.timeout_seconds,
            )
            real_result = real_command.get("result") or {}
            download_url = str(real_result.get("download_url", ""))
            if not download_url:
                raise RuntimeError("真实摄像头截图没有download_url")
            real_content, real_mime = self.mirror.download(download_url)
            result["real_capture"] = self.artifacts.store(
                real_content,
                real_mime,
                "real_camera_capture",
            )
        self._record(
            "capture",
            request.model_dump(),
            self.status(),
            self.status(),
            result=result,
        )
        return result

    def stop(self) -> None:
        if self.mirror is not None:
            self.mirror.stop()
        self._record("stop", {}, self.status(), self.status())

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._viewport_payload()

    def _viewport_payload(self) -> dict[str, Any]:
        return {
            "camera_id": self.settings.camera_id,
            "provider": "virtual_ptz_test",
            "online": (
                self._thread is not None
                and self._thread.is_alive()
                and self._last_frame_at is not None
            ),
            "stream_ready": (
                self._last_publish_at is not None
                and time.monotonic() - self._last_publish_at < 5.0
            ),
            "ptz_state": "idle",
            "zoom_state": "idle",
            "focus_state": "idle",
            "pan_degrees": (self.viewport.center_x - 0.5) * 180.0,
            "tilt_degrees": (self.viewport.center_y - 0.5) * 90.0,
            "zoom_level": self.viewport.zoom,
            "zoom_ratio": self.viewport.zoom,
            "preset_id": 1 if self.viewport.zoom == 1.0 else None,
            "last_error": self._last_error,
            "updated_at": _utc_now(),
            "center_x": self.viewport.center_x,
            "center_y": self.viewport.center_y,
            "last_action": self._last_action,
            "mirror_real_camera": self.settings.mirror_real_camera,
            "publisher_backend": getattr(self.publisher, "backend", "injected"),
        }

    def report(self) -> dict[str, Any]:
        with self._lock:
            history = list(self._history)
            current = self._viewport_payload()
        locates = [item for item in history if item["action"] == "locate"]
        captures = [item for item in history if item["action"] == "capture"]
        homes = [item for item in history if item["action"] == "home"]
        return {
            "test_mode": True,
            "started_at": self._started_at,
            "state": "running" if current["online"] else "stopped",
            "current": current,
            "rounds": locates,
            "captures": captures,
            "checks": {
                "at_least_one_locate": bool(locates),
                "virtual_zoom_changed": any(
                    item["after"]["zoom_level"] > 1.0 for item in locates
                ),
                "capture_returned": bool(captures),
                "home_returned": bool(homes) and current["preset_id"] == 1,
                "real_camera_mirrored": self.settings.mirror_real_camera,
            },
        }

    def _record(
        self,
        action: str,
        request: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
        *,
        result: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._history.append(
                {
                    "sequence": len(self._history) + 1,
                    "timestamp": _utc_now(),
                    "action": action,
                    "request": request,
                    "before": before,
                    "after": after,
                    "result": result,
                }
            )

    def _render_loop(self) -> None:
        import cv2

        capture = cv2.VideoCapture(str(self.settings.source_path))
        if not capture.isOpened():
            return
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        start_frame = int(round(self.settings.source_start_seconds * source_fps))
        end_frame = (
            int(round(self.settings.source_end_seconds * source_fps))
            if self.settings.source_end_seconds is not None
            else None
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        next_due = time.monotonic()
        try:
            while not self._stop.is_set():
                position = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                if end_frame is not None and position >= end_frame:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                ok, frame = capture.read()
                if not ok:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    continue
                with self._lock:
                    rendered = self.viewport.render(
                        frame,
                        width=self.settings.output_width,
                        height=self.settings.output_height,
                    )
                    self._draw_test_overlay(rendered)
                    self._latest_frame = rendered.copy()
                    self._last_frame_at = time.monotonic()
                try:
                    self.publisher.write(rendered)
                    with self._lock:
                        self._last_publish_at = time.monotonic()
                        self._last_error = None
                except Exception as exc:
                    with self._lock:
                        self._last_error = f"{type(exc).__name__}: {exc}"
                    self._stop.wait(0.5)
                next_due += 1.0 / self.settings.output_fps
                self._stop.wait(max(next_due - time.monotonic(), 0.0))
                if next_due < time.monotonic() - 1.0:
                    next_due = time.monotonic()
        finally:
            capture.release()

    def _draw_test_overlay(self, frame: np.ndarray) -> None:
        import cv2

        lines = [
            "ISOLATED PTZ TEST - NOT PRODUCTION EVIDENCE",
            (
                f"zoom={self.viewport.zoom:.2f} "
                f"center=({self.viewport.center_x:.3f},{self.viewport.center_y:.3f}) "
                f"action={self._last_action}"
            ),
            f"mirror_real_camera={self.settings.mirror_real_camera}",
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (20, 35 + index * 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )


def create_test_app(
    settings: TestServiceSettings | None = None,
    controller: VirtualPtzTestController | None = None,
) -> FastAPI:
    settings = settings or TestServiceSettings.from_environment()
    settings.validate()
    controller = controller or VirtualPtzTestController(settings)
    commands = _CommandExecutor()
    lease_lock = threading.Lock()
    lease: dict[str, Any] = {}
    app = FastAPI(title="Isolated PTZ Test Camera", version="1.0.0")

    @app.on_event("startup")
    def startup() -> None:
        controller.start()

    @app.on_event("shutdown")
    def shutdown() -> None:
        commands.shutdown()
        controller.shutdown()

    def authorize(x_camera_control_key: str | None) -> None:
        import hmac

        if x_camera_control_key is None or not hmac.compare_digest(
            x_camera_control_key,
            settings.api_key,
        ):
            raise HTTPException(status_code=401, detail="invalid X-Camera-Control-Key")

    def require_lease(token: str | None) -> None:
        with lease_lock:
            if lease and float(lease["expires_at"]) <= time.time():
                lease.clear()
            if lease and token != lease["token"]:
                raise HTTPException(
                    status_code=409,
                    detail=f"camera is leased by {lease['owner']}",
                )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "test_mode": True,
            "source": settings.source_path.name,
        }

    @app.get("/v1/cameras")
    def cameras(
        x_camera_control_key: str | None = Header(
            default=None, alias="X-Camera-Control-Key"
        ),
    ) -> list[dict[str, Any]]:
        authorize(x_camera_control_key)
        return [
            {
                "camera_id": settings.camera_id,
                "provider": "virtual_ptz_test",
                "channel": 0,
                "home_preset": 1,
            }
        ]

    @app.get("/v1/cameras/{camera_id}/status")
    def camera_status(
        camera_id: str,
        x_camera_control_key: str | None = Header(
            default=None, alias="X-Camera-Control-Key"
        ),
    ) -> dict[str, Any]:
        authorize(x_camera_control_key)
        _ensure_camera(camera_id, settings)
        return controller.status()

    @app.post("/v1/cameras/{camera_id}/lease")
    def acquire_lease(
        camera_id: str,
        request: LeaseRequest,
        x_camera_control_key: str | None = Header(
            default=None, alias="X-Camera-Control-Key"
        ),
        x_camera_control_lease: str | None = Header(
            default=None, alias="X-Camera-Control-Lease"
        ),
    ) -> dict[str, Any]:
        authorize(x_camera_control_key)
        _ensure_camera(camera_id, settings)
        with lease_lock:
            if lease and float(lease["expires_at"]) <= time.time():
                lease.clear()
            if lease and x_camera_control_lease != lease["token"]:
                raise HTTPException(
                    status_code=409,
                    detail=f"camera is leased by {lease['owner']}",
                )
            if not lease:
                lease.update(
                    owner=request.owner,
                    token=uuid.uuid4().hex,
                )
            elif request.owner != lease["owner"]:
                raise HTTPException(status_code=409, detail="lease owner mismatch")
            lease["expires_at"] = time.time() + request.ttl_seconds
            return {
                "camera_id": camera_id,
                "owner": lease["owner"],
                "token": lease["token"],
                "expires_at": lease["expires_at"],
            }

    @app.delete(
        "/v1/cameras/{camera_id}/lease",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def release_lease(
        camera_id: str,
        x_camera_control_key: str | None = Header(
            default=None, alias="X-Camera-Control-Key"
        ),
        x_camera_control_lease: str | None = Header(
            default=None, alias="X-Camera-Control-Lease"
        ),
    ) -> None:
        authorize(x_camera_control_key)
        _ensure_camera(camera_id, settings)
        with lease_lock:
            if lease and x_camera_control_lease != lease["token"]:
                raise HTTPException(status_code=409, detail="lease token mismatch")
            lease.clear()

    @app.post(
        "/v1/cameras/{camera_id}/commands/locate",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def locate(
        camera_id: str,
        request: LocateRequest,
        x_camera_control_key: str | None = Header(
            default=None, alias="X-Camera-Control-Key"
        ),
        x_camera_control_lease: str | None = Header(
            default=None, alias="X-Camera-Control-Lease"
        ),
    ) -> dict[str, Any]:
        authorize(x_camera_control_key)
        require_lease(x_camera_control_lease)
        _ensure_camera(camera_id, settings)
        return commands.submit(camera_id, "locate", lambda: controller.locate(request))

    @app.post(
        "/v1/cameras/{camera_id}/commands/home",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def home(
        camera_id: str,
        request: TimeoutRequest | None = None,
        x_camera_control_key: str | None = Header(
            default=None, alias="X-Camera-Control-Key"
        ),
        x_camera_control_lease: str | None = Header(
            default=None, alias="X-Camera-Control-Lease"
        ),
    ) -> dict[str, Any]:
        authorize(x_camera_control_key)
        require_lease(x_camera_control_lease)
        _ensure_camera(camera_id, settings)
        value = request or TimeoutRequest()
        return commands.submit(camera_id, "home", lambda: controller.home(value))

    @app.post(
        "/v1/cameras/{camera_id}/commands/autofocus",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def autofocus(
        camera_id: str,
        request: TimeoutRequest | None = None,
        x_camera_control_key: str | None = Header(
            default=None, alias="X-Camera-Control-Key"
        ),
        x_camera_control_lease: str | None = Header(
            default=None, alias="X-Camera-Control-Lease"
        ),
    ) -> dict[str, Any]:
        authorize(x_camera_control_key)
        require_lease(x_camera_control_lease)
        _ensure_camera(camera_id, settings)
        value = request or TimeoutRequest()
        return commands.submit(
            camera_id,
            "autofocus",
            lambda: controller.autofocus(value),
        )

    @app.post(
        "/v1/cameras/{camera_id}/commands/capture",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def capture(
        camera_id: str,
        request: CaptureRequest,
        x_camera_control_key: str | None = Header(
            default=None, alias="X-Camera-Control-Key"
        ),
        x_camera_control_lease: str | None = Header(
            default=None, alias="X-Camera-Control-Lease"
        ),
    ) -> dict[str, Any]:
        authorize(x_camera_control_key)
        require_lease(x_camera_control_lease)
        _ensure_camera(camera_id, settings)
        return commands.submit(camera_id, "capture", lambda: controller.capture(request))

    @app.post("/v1/cameras/{camera_id}/stop")
    def stop(
        camera_id: str,
        x_camera_control_key: str | None = Header(
            default=None, alias="X-Camera-Control-Key"
        ),
    ) -> dict[str, str]:
        authorize(x_camera_control_key)
        _ensure_camera(camera_id, settings)
        controller.stop()
        return {"status": "ok"}

    @app.get("/v1/commands/{command_id}")
    def command_status(
        command_id: str,
        x_camera_control_key: str | None = Header(
            default=None, alias="X-Camera-Control-Key"
        ),
    ) -> dict[str, Any]:
        authorize(x_camera_control_key)
        try:
            return commands.get(command_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="command not found") from exc

    @app.get("/v1/artifacts/{artifact_id}")
    def artifact(
        artifact_id: str,
        x_camera_control_key: str | None = Header(
            default=None, alias="X-Camera-Control-Key"
        ),
    ) -> Response:
        authorize(x_camera_control_key)
        try:
            path, mime_type, _kind = controller.artifacts.get(artifact_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="artifact not found") from exc
        return FileResponse(path, media_type=mime_type)

    @app.get("/v1/test/report")
    def report(
        x_camera_control_key: str | None = Header(
            default=None, alias="X-Camera-Control-Key"
        ),
    ) -> dict[str, Any]:
        authorize(x_camera_control_key)
        return controller.report()

    return app


def _ensure_camera(camera_id: str, settings: TestServiceSettings) -> None:
    if camera_id != settings.camera_id:
        raise HTTPException(status_code=404, detail="camera not found")


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_test_app(),
        host="0.0.0.0",
        port=8080,
        reload=False,
    )


if __name__ == "__main__":
    main()
