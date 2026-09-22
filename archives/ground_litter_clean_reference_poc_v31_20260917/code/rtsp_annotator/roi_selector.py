from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

from .config import parse_roi


def _format_coordinate(value: float) -> str:
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


def format_normalized_roi(
    points: Sequence[tuple[int, int]],
    width: int,
    height: int,
) -> str:
    """Convert pixel points selected in a frame into the CLI ROI format."""
    if width <= 1 or height <= 1:
        raise ValueError("画面宽高必须大于 1")
    if len(points) < 3:
        raise ValueError("识别区域至少需要 3 个点")

    normalized: list[str] = []
    for x, y in points:
        normalized_x = min(1.0, max(0.0, x / (width - 1)))
        normalized_y = min(1.0, max(0.0, y / (height - 1)))
        normalized.append(
            f"{_format_coordinate(normalized_x)},"
            f"{_format_coordinate(normalized_y)}"
        )
    value = ";".join(normalized)
    parse_roi(value)
    return value


def read_preview_frame(
    source: str,
    transport: str,
    open_timeout: float,
    read_timeout: float,
) -> Any:
    """Read one BGR frame from an RTSP URL or a local image."""
    if source.lower().startswith(("rtsp://", "rtsps://")):
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("缺少 PyAV，请先安装 requirements.txt") from exc

        container = av.open(
            source,
            mode="r",
            options={
                "rtsp_transport": transport,
                "fflags": "nobuffer",
                "flags": "low_delay",
                "max_delay": "0",
                "reorder_queue_size": "0",
                "probesize": "32",
                "analyzeduration": "0",
            },
            timeout=(open_timeout, read_timeout),
        )
        try:
            video = next(
                stream for stream in container.streams if stream.type == "video"
            )
            video.thread_type = "SLICE"
            video.codec_context.thread_count = 1
            decoded = next(container.decode(video=0))
            return decoded.to_ndarray(format="bgr24")
        except StopIteration as exc:
            raise RuntimeError("输入中没有可读取的视频画面") from exc
        finally:
            container.close()

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("缺少 OpenCV，请先安装 requirements.txt") from exc

    frame = cv2.imread(str(Path(source).expanduser()))
    if frame is None:
        raise RuntimeError(f"无法读取图片: {source}")
    return frame


def select_roi(frame: Any, window_name: str = "Draw detection ROI") -> str | None:
    """Open an OpenCV window and let the user draw a polygon with the mouse."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("缺少 OpenCV/NumPy，请先安装 requirements.txt") from exc

    source_height, source_width = frame.shape[:2]
    preview_scale = min(1.0, 1280 / source_width, 720 / source_height)
    if preview_scale < 1.0:
        frame = cv2.resize(
            frame,
            (
                round(source_width * preview_scale),
                round(source_height * preview_scale),
            ),
            interpolation=cv2.INTER_AREA,
        )
    height, width = frame.shape[:2]
    points: list[tuple[int, int]] = []

    def on_mouse(
        event: int,
        x: int,
        y: int,
        _flags: int,
        _parameter: object,
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            canvas = frame.copy()
            if points:
                polygon = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    canvas,
                    [polygon],
                    isClosed=len(points) >= 3,
                    color=(0, 255, 255),
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
                for index, point in enumerate(points, start=1):
                    cv2.circle(canvas, point, 5, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.putText(
                        canvas,
                        str(index),
                        (point[0] + 7, point[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.rectangle(canvas, (0, 0), (width, 38), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                "Left: add  Right/Backspace: undo  R: reset  Enter: save  Esc: cancel",
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13) and len(points) >= 3:
                return format_normalized_roi(points, width, height)
            if key == 27:
                return None
            if key in (8, 127) and points:
                points.pop()
            elif key in (ord("r"), ord("R")):
                points.clear()
    finally:
        cv2.destroyWindow(window_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rtsp_annotator.roi_selector",
        description="从 RTSP 首帧或本地图片中用鼠标绘制多边形识别区域。",
    )
    parser.add_argument(
        "--input",
        default=os.environ.get("RTSP_INPUT_URL"),
        required=os.environ.get("RTSP_INPUT_URL") is None,
        help="RTSP 地址或本地图片路径，也可使用 RTSP_INPUT_URL",
    )
    parser.add_argument(
        "--transport",
        choices=("tcp", "udp"),
        default="tcp",
        help="读取 RTSP 时使用的传输方式",
    )
    parser.add_argument("--open-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        frame = read_preview_frame(
            args.input,
            args.transport,
            args.open_timeout,
            args.read_timeout,
        )
        roi = select_roi(frame)
    except Exception as exc:
        parser.error(str(exc))

    if roi is None:
        print("已取消，没有生成识别区域。")
        return 130
    print("\n复制下面的参数到识别程序：")
    print(f"--roi '{roi}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
