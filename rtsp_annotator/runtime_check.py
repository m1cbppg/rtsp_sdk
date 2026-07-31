from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import math
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .pipeline import (
    ResolvedDevice,
    resolve_inference_device,
    validate_inference_precision,
)


SUPPORTED_PYTHON_MIN = (3, 10)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 13)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "未安装"


def _resolve_executable(value: str) -> str:
    executable = shutil.which(value)
    if executable is None and Path(value).is_file():
        executable = str(Path(value).resolve())
    if executable is None:
        raise RuntimeError(f"找不到可执行文件: {value}")
    return executable


def _check_ffmpeg(ffmpeg_path: str) -> tuple[str, str]:
    executable = _resolve_executable(ffmpeg_path)
    result = subprocess.run(
        [executable, "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    encoders = f"{result.stdout}\n{result.stderr}"
    if "libx264" not in encoders:
        raise RuntimeError(f"FFmpeg 不包含 libx264 编码器: {executable}")
    version_result = subprocess.run(
        [executable, "-version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    version_line = version_result.stdout.splitlines()[0]
    return executable, version_line


def _synchronize(torch_module: Any, device: ResolvedDevice) -> None:
    if device.backend == "cuda":
        index = int(device.value.split(":", 1)[1])
        torch_module.cuda.synchronize(index)
    elif device.backend == "mps":
        torch_module.mps.synchronize()


def _probe_accelerator(
    torch_module: Any,
    device: ResolvedDevice,
) -> float:
    """Run real work on the selected backend; this is a health check, not a benchmark."""
    size = 1024 if device.backend != "cpu" else 256
    left = torch_module.ones((size, size), device=device.value)
    right = torch_module.ones((size, size), device=device.value)
    _ = left @ right
    _synchronize(torch_module, device)

    started = time.perf_counter()
    result = left @ right
    _synchronize(torch_module, device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    expected = float(size)
    observed = float(result[0, 0].item())
    if observed != expected:
        raise RuntimeError(
            f"设备计算结果异常: expected={expected}, actual={observed}"
        )
    return elapsed_ms


def _device_details(torch_module: Any, device: ResolvedDevice) -> list[str]:
    if device.backend != "cuda":
        return []

    index = int(device.value.split(":", 1)[1])
    properties = torch_module.cuda.get_device_properties(index)
    major, minor = torch_module.cuda.get_device_capability(index)
    total_gib = properties.total_memory / (1024**3)
    return [
        f"PyTorch CUDA runtime: {torch_module.version.cuda}",
        f"CUDA capability: {major}.{minor}",
        f"GPU memory: {total_gib:.1f} GiB",
    ]


def _benchmark_yolo(
    torch_module: Any,
    ultralytics_module: Any,
    device: ResolvedDevice,
    model_path: Path,
    imgsz: int,
    half: bool,
    iterations: int,
) -> tuple[float, float]:
    if not model_path.is_file():
        raise RuntimeError(f"模型文件不存在: {model_path}")
    if imgsz <= 0:
        raise ValueError("--imgsz 必须大于 0")
    if iterations <= 0:
        raise ValueError("--iterations 必须大于 0")
    validate_inference_precision(half, device)

    numpy_module = importlib.import_module("numpy")
    frame = numpy_module.zeros((1080, 1920, 3), dtype=numpy_module.uint8)
    model = ultralytics_module.YOLO(str(model_path))
    predict_kwargs = {
        "source": frame,
        "device": device.value,
        "imgsz": imgsz,
        "verbose": False,
    }
    if half:
        predict_kwargs["quantize"] = 16

    for _ in range(3):
        model.predict(**predict_kwargs)
    _synchronize(torch_module, device)

    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        model.predict(**predict_kwargs)
        _synchronize(torch_module, device)
        samples_ms.append((time.perf_counter() - started) * 1000)

    ordered = sorted(samples_ms)
    p95_index = min(
        len(ordered) - 1,
        max(0, math.ceil(len(ordered) * 0.95) - 1),
    )
    return statistics.fmean(samples_ms), ordered[p95_index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rtsp_annotator.runtime_check",
        description="检查 RTSP + YOLO 运行依赖，并在指定设备上执行真实张量计算。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto、cpu、mps、cuda、cuda:N 或数字 GPU 编号",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="FFmpeg 可执行文件",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="可选：加载该 .pt 模型并执行预热后的推理基准",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="模型基准的 YOLO 推理尺寸",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="模型基准在 CUDA 上使用 FP16",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="模型预热后计时的推理次数",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("RTSP + YOLO 运行环境自检")
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")

    version = sys.version_info[:2]
    if not SUPPORTED_PYTHON_MIN <= version < SUPPORTED_PYTHON_MAX_EXCLUSIVE:
        print("失败: 本项目要求 Python 3.10–3.12", file=sys.stderr)
        return 1

    try:
        torch_module = importlib.import_module("torch")
        importlib.import_module("av")
        ultralytics_module = importlib.import_module("ultralytics")
        device = resolve_inference_device(args.device, torch_module)
        ffmpeg, ffmpeg_version = _check_ffmpeg(args.ffmpeg)
        elapsed_ms = _probe_accelerator(torch_module, device)
        benchmark: tuple[float, float] | None = None
        if args.model is not None:
            benchmark = _benchmark_yolo(
                torch_module,
                ultralytics_module,
                device,
                args.model.expanduser().resolve(),
                args.imgsz,
                args.half,
                args.iterations,
            )
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"失败: {exc}", file=sys.stderr)
        return 1

    print(f"PyTorch: {_distribution_version('torch')}")
    print(f"Ultralytics: {_distribution_version('ultralytics')}")
    print(f"PyAV: {_distribution_version('av')}")
    print(f"Device: {device.value} ({device.name})")
    for detail in _device_details(torch_module, device):
        print(detail)
    print(f"设备计算自检: 通过 ({elapsed_ms:.2f} ms，仅用于健康检查)")
    print(f"FFmpeg: {ffmpeg}")
    print(f"FFmpeg version: {ffmpeg_version}")
    print("libx264: 可用")
    if benchmark is not None:
        average_ms, p95_ms = benchmark
        print(
            "YOLO 模型基准: "
            f"平均 {average_ms:.2f} ms, P95 {p95_ms:.2f} ms, "
            f"理论上限 {1000 / average_ms:.1f} FPS"
        )
        print("说明: 模型基准不包含 RTSP 拉流、画框、H.264 编码和播放器缓存")
    print("结论: 基础运行环境通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
