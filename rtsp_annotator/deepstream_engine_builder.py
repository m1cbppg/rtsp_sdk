from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path


LOGGER = logging.getLogger("rtsp_annotator.engine_builder")


def engine_path_for(
    onnx_path: Path,
    engine_root: Path,
    *,
    imgsz: int,
    batch_size: int,
    gpu_id: int,
) -> Path:
    return engine_root / (
        f"{onnx_path.stem}_{imgsz}_b{batch_size}_gpu{gpu_id}_fp16.engine"
    )


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    imgsz: int,
    batch_size: int,
    gpu_id: int = 0,
    trtexec: str = "trtexec",
) -> None:
    if engine_path.is_file() and engine_path.stat().st_size > 0:
        LOGGER.info("复用TensorRT引擎: %s", engine_path.name)
        return
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = engine_path.with_suffix(".engine.building")
    temporary.unlink(missing_ok=True)
    batch_one = f"input:1x3x{imgsz}x{imgsz}"
    batch_max = f"input:{batch_size}x3x{imgsz}x{imgsz}"
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={temporary}",
        "--fp16",
        f"--device={gpu_id}",
        f"--minShapes={batch_one}",
        f"--optShapes={batch_max}",
        f"--maxShapes={batch_max}",
        "--memPoolSize=workspace:2048",
        "--builderOptimizationLevel=3",
        "--skipInference",
    ]
    LOGGER.info(
        "首次构建TensorRT引擎（可能需要数分钟）: %s",
        engine_path.name,
    )
    try:
        subprocess.run(command, check=True)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("trtexec未生成有效引擎")
        temporary.replace(engine_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_auxiliary_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    input_name: str,
    input_shape: tuple[int, int, int],
    batch_size: int,
    gpu_id: int = 0,
    trtexec: str = "trtexec",
) -> None:
    if engine_path.is_file() and engine_path.stat().st_size > 0:
        LOGGER.info("复用TensorRT引擎: %s", engine_path.name)
        return
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = engine_path.with_suffix(".engine.building")
    temporary.unlink(missing_ok=True)
    channels, height, width = input_shape

    def shape(batch: int) -> str:
        # trtexec uses ":" as the separator between an input name and its
        # dimensions. TensorFlow-style ONNX names also contain ":", so quote
        # those names explicitly to keep the profile unambiguous.
        profile_name = f"'{input_name}'" if ":" in input_name else input_name
        return f"{profile_name}:{batch}x{channels}x{height}x{width}"

    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={temporary}",
        "--fp16",
        f"--device={gpu_id}",
        f"--minShapes={shape(1)}",
        f"--optShapes={shape(min(4, batch_size))}",
        f"--maxShapes={shape(batch_size)}",
        "--memPoolSize=workspace:1024",
        "--builderOptimizationLevel=3",
        "--skipInference",
    ]
    LOGGER.info("构建辅助TensorRT引擎: %s", engine_path.name)
    try:
        subprocess.run(command, check=True)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("trtexec未生成有效辅助引擎")
        temporary.replace(engine_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_license_plate_engines(
    model_root: Path,
    engine_root: Path,
    *,
    batch_size: int = 16,
    gpu_id: int = 0,
    trtexec: str = "trtexec",
) -> None:
    lpr_root = model_root / "lpr"
    detector = lpr_root / "LPDNet_CCPD_pruned_tao5.onnx"
    recognizer = lpr_root / "ch_lprnet_baseline18_deployable.onnx"
    if not detector.is_file() and not recognizer.is_file():
        return
    if not detector.is_file() or not recognizer.is_file():
        raise RuntimeError("中国车牌模型不完整")
    build_auxiliary_engine(
        detector,
        engine_root
        / f"lpdnet_ch_b{batch_size}_gpu{gpu_id}_fp16.engine",
        input_name="input_1:0",
        input_shape=(3, 1168, 720),
        batch_size=batch_size,
        gpu_id=gpu_id,
        trtexec=trtexec,
    )
    build_auxiliary_engine(
        recognizer,
        engine_root
        / f"lprnet_ch_b{batch_size}_gpu{gpu_id}_fp16.engine",
        input_name="image_input",
        input_shape=(3, 48, 96),
        batch_size=batch_size,
        gpu_id=gpu_id,
        trtexec=trtexec,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="为DeepStream预构建TensorRT FP16引擎",
    )
    parser.add_argument("--models", type=Path, default=Path("/app/models"))
    parser.add_argument("--engines", type=Path, default=Path("/app/engines"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--lpr-batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    models = sorted(args.models.glob("*.onnx"))
    if not models:
        raise RuntimeError(f"没有找到ONNX模型: {args.models}")
    for onnx_path in models:
        labels_path = onnx_path.with_name(
            f"{onnx_path.stem}.labels.txt"
        )
        if not labels_path.is_file():
            raise RuntimeError(f"缺少标签文件: {labels_path}")
        target = engine_path_for(
            onnx_path,
            args.engines,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            gpu_id=args.gpu_id,
        )
        build_engine(
            onnx_path,
            target,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            gpu_id=args.gpu_id,
            trtexec=args.trtexec,
        )
    build_license_plate_engines(
        args.models,
        args.engines,
        batch_size=args.lpr_batch_size,
        gpu_id=args.gpu_id,
        trtexec=args.trtexec,
    )


if __name__ == "__main__":
    main()
