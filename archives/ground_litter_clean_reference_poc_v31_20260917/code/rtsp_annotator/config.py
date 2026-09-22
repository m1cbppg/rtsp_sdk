from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


VALID_RTSP_TRANSPORTS = ("tcp", "udp")
RoiPoint = tuple[float, float]
RoiPolygon = tuple[RoiPoint, ...]


def parse_classes(value: str | None) -> tuple[int, ...] | None:
    """Parse a comma-separated class list, preserving order without duplicates."""
    if value is None or not value.strip():
        return None

    parsed: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        class_id = int(item)
        if class_id < 0:
            raise ValueError("类别 ID 不能为负数")
        if class_id not in parsed:
            parsed.append(class_id)
    return tuple(parsed) or None


def parse_roi(value: str | None) -> RoiPolygon | None:
    """Parse normalized polygon points such as '0.1,0.2;0.9,0.2;0.5,0.9'."""
    if value is None or not value.strip():
        return None

    points: list[RoiPoint] = []
    for raw_point in value.split(";"):
        coordinates = [item.strip() for item in raw_point.split(",")]
        if len(coordinates) != 2 or not all(coordinates):
            raise ValueError(
                "识别区域格式应为 x,y;x,y;x,y，例如 "
                "0.1,0.1;0.9,0.1;0.9,0.9;0.1,0.9"
            )
        try:
            x, y = (float(item) for item in coordinates)
        except ValueError as exc:
            raise ValueError("识别区域坐标必须是数字") from exc
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("识别区域坐标必须在 0 到 1 之间")
        points.append((x, y))

    if len(points) < 3:
        raise ValueError("识别区域至少需要 3 个点")
    if len(set(points)) < 3:
        raise ValueError("识别区域至少需要 3 个不同的点")

    doubled_area = abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                points,
                points[1:] + points[:1],
            )
        )
    )
    if doubled_area <= 1e-9:
        raise ValueError("识别区域不能是零面积或共线多边形")
    return tuple(points)


def redact_url(value: str) -> str:
    """Hide credentials from an RTSP URL before it is written to logs."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    try:
        if parsed.username is None:
            return value

        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        redacted = parsed._replace(netloc=f"***:***@{host}")
        return redacted.geturl()
    except ValueError:
        return "<invalid-rtsp-url>"


@dataclass(frozen=True, slots=True)
class Settings:
    input_url: str
    output_url: str
    model_path: Path
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    classes: tuple[int, ...] | None = None
    roi: RoiPolygon | None = None
    roi_line_width: int = 3
    device: str = "auto"
    half: bool = False
    show_labels: bool = True
    label_map_path: Path | None = None
    font_path: Path | None = None
    line_width: int | None = None
    output_fps: float | None = None
    fallback_fps: float = 25.0
    bitrate: str = "2500k"
    encoder: str = "libx264"
    preset: str = "ultrafast"
    gop_seconds: float = 1.0
    input_rtsp_transport: str = "tcp"
    output_rtsp_transport: str = "tcp"
    ffmpeg_path: str = "ffmpeg"
    open_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 10.0
    reconnect_delay_seconds: float = 2.0
    stats_interval_seconds: float = 10.0
    log_level: str = "INFO"

    def validate(self) -> None:
        errors: list[str] = []
        if not self.input_url.lower().startswith(("rtsp://", "rtsps://")):
            errors.append("输入地址必须以 rtsp:// 或 rtsps:// 开头")
        if not self.output_url.lower().startswith(("rtsp://", "rtsps://")):
            errors.append("输出地址必须以 rtsp:// 或 rtsps:// 开头")
        if self.input_url == self.output_url:
            errors.append("输入和输出 RTSP 地址不能相同")
        if not self.model_path.is_file():
            errors.append(f"模型文件不存在: {self.model_path}")
        if self.label_map_path is not None and not self.label_map_path.is_file():
            errors.append(f"中文标签映射文件不存在: {self.label_map_path}")
        if self.font_path is not None and not self.font_path.is_file():
            errors.append(f"中文字体文件不存在: {self.font_path}")
        if not 0.0 < self.conf <= 1.0:
            errors.append("conf 必须在 (0, 1] 范围内")
        if not 0.0 < self.iou <= 1.0:
            errors.append("iou 必须在 (0, 1] 范围内")
        if self.imgsz <= 0:
            errors.append("imgsz 必须大于 0")
        if self.roi is not None:
            if len(self.roi) < 3:
                errors.append("roi 至少需要 3 个点")
            elif any(
                not 0.0 <= coordinate <= 1.0
                for point in self.roi
                for coordinate in point
            ):
                errors.append("roi 坐标必须在 [0, 1] 范围内")
        if self.roi_line_width <= 0:
            errors.append("roi_line_width 必须大于 0")
        if self.output_fps is not None and not 0.1 <= self.output_fps <= 120.0:
            errors.append("output_fps 必须在 [0.1, 120] 范围内")
        if not 0.1 <= self.fallback_fps <= 120.0:
            errors.append("fallback_fps 必须在 [0.1, 120] 范围内")
        if self.line_width is not None and self.line_width <= 0:
            errors.append("line_width 必须大于 0")
        if self.gop_seconds <= 0:
            errors.append("gop_seconds 必须大于 0")
        if self.input_rtsp_transport not in VALID_RTSP_TRANSPORTS:
            errors.append(
                f"input_rtsp_transport 必须是: {', '.join(VALID_RTSP_TRANSPORTS)}"
            )
        if self.output_rtsp_transport not in VALID_RTSP_TRANSPORTS:
            errors.append(
                f"output_rtsp_transport 必须是: {', '.join(VALID_RTSP_TRANSPORTS)}"
            )
        if self.open_timeout_seconds <= 0 or self.read_timeout_seconds <= 0:
            errors.append("拉流超时必须大于 0")
        if self.reconnect_delay_seconds <= 0:
            errors.append("重连间隔必须大于 0")
        if self.stats_interval_seconds <= 0:
            errors.append("统计间隔必须大于 0")
        if errors:
            raise ValueError("；".join(errors))
