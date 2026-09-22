from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Sequence

from .config import Settings, parse_classes, parse_roi
from .pipeline import run_pipeline


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value is not None else default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rtsp_annotator",
        description="读取 RTSP，实时运行 YOLO 并把叠框画面发布为新的 RTSP。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=_env("RTSP_INPUT_URL"),
        help="输入 RTSP 地址，也可用 RTSP_INPUT_URL",
    )
    parser.add_argument(
        "--output",
        default=_env("RTSP_OUTPUT_URL"),
        help="输出 RTSP 地址，也可用 RTSP_OUTPUT_URL",
    )
    parser.add_argument(
        "--model",
        default=_env("YOLO_MODEL_PATH"),
        help="本地 Ultralytics YOLO .pt 模型，也可用 YOLO_MODEL_PATH",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸")
    parser.add_argument(
        "--classes",
        default=None,
        metavar="IDS",
        help="只保留这些类别 ID，例如 0 或 0,2,3；默认全部类别",
    )
    parser.add_argument(
        "--roi",
        default=_env("YOLO_ROI"),
        metavar="POINTS",
        help=(
            "归一化多边形识别区域，例如 "
            "0.1,0.1;0.9,0.1;0.9,0.9;0.1,0.9；"
            "只保留中心点在区域内的检测框"
        ),
    )
    parser.add_argument(
        "--roi-line-width",
        type=int,
        default=3,
        help="输出画面中的识别区域边界宽度",
    )
    parser.add_argument(
        "--device",
        default=_env("YOLO_DEVICE", "auto"),
        help=(
            "推理设备：auto 自动选择 CUDA>MPS>CPU；"
            "也可指定 cpu、mps、cuda、cuda:0 或 0；"
            "也可使用 YOLO_DEVICE"
        ),
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="CUDA 推理使用 FP16；MPS 和 CPU 不支持此参数",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="框上不显示类别名称",
    )
    parser.add_argument(
        "--label-map",
        default=None,
        help="自定义模型的中文标签 JSON；键可为原类别名或类别 ID",
    )
    parser.add_argument(
        "--font",
        default=None,
        help="中文字体 .ttf/.ttc/.otf；默认自动寻找系统中文字体",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=None,
        help="框线宽度；默认由 YOLO 按画面尺寸计算",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="输出 FPS；默认读取源流 FPS",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=25.0,
        help="源流未提供可靠 FPS 时的回退值",
    )
    parser.add_argument("--bitrate", default="2500k", help="H.264 目标码率")
    parser.add_argument("--encoder", default="libx264", help="FFmpeg 视频编码器")
    parser.add_argument("--preset", default="ultrafast", help="编码器 preset")
    parser.add_argument(
        "--gop-seconds",
        type=float,
        default=1.0,
        help="关键帧间隔（秒）",
    )
    parser.add_argument(
        "--input-transport",
        choices=("tcp", "udp"),
        default="tcp",
        help="拉取源流使用的 RTSP 下层传输",
    )
    parser.add_argument(
        "--output-transport",
        choices=("tcp", "udp"),
        default="tcp",
        help="向 RTSP Server 发布时使用的下层传输",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg 可执行文件")
    parser.add_argument(
        "--open-timeout",
        type=float,
        default=10.0,
        help="打开输入流超时（秒）",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=10.0,
        help="读取输入帧超时（秒）",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=2.0,
        help="输入或输出失败后的重试间隔（秒）",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=10.0,
        help="运行统计打印间隔（秒）",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="日志级别",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    missing = [
        flag
        for flag, value in (
            ("--input", args.input),
            ("--output", args.output),
            ("--model", args.model),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"缺少必需参数: {', '.join(missing)}")

    settings = Settings(
        input_url=args.input,
        output_url=args.output,
        model_path=Path(args.model).expanduser().resolve(),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=parse_classes(args.classes),
        roi=parse_roi(args.roi),
        roi_line_width=args.roi_line_width,
        device=args.device,
        half=args.half,
        show_labels=not args.no_labels,
        label_map_path=(
            Path(args.label_map).expanduser().resolve()
            if args.label_map
            else None
        ),
        font_path=(
            Path(args.font).expanduser().resolve() if args.font else None
        ),
        line_width=args.line_width,
        output_fps=args.output_fps,
        fallback_fps=args.fallback_fps,
        bitrate=args.bitrate,
        encoder=args.encoder,
        preset=args.preset,
        gop_seconds=args.gop_seconds,
        input_rtsp_transport=args.input_transport,
        output_rtsp_transport=args.output_transport,
        ffmpeg_path=args.ffmpeg,
        open_timeout_seconds=args.open_timeout,
        read_timeout_seconds=args.read_timeout,
        reconnect_delay_seconds=args.reconnect_delay,
        stats_interval_seconds=args.stats_interval,
        log_level=args.log_level,
    )
    settings.validate()
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = settings_from_args(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
    )
    try:
        run_pipeline(settings)
    except KeyboardInterrupt:
        return 130
    except Exception:
        logging.getLogger(__name__).exception("流水线异常退出")
        return 1
    return 0
