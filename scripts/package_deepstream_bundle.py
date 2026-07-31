from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path


FILES = {
    "docker-compose.deepstream.api.yml": "docker-compose.deepstream.api.yml",
    "DEEPSTREAM_DEPLOY.md": "DEEPSTREAM_DEPLOY.md",
    "LICENSE_PLATE.md": "LICENSE_PLATE.md",
    "NIGHT_VISION.md": "NIGHT_VISION.md",
    "HTTP_API.md": "HTTP_API.md",
    "config/api.deepstream.example.json": (
        "config/api.deepstream.example.json"
    ),
    "config/labels.zh.example.json": "config/labels.zh.example.json",
    "config/mediamtx.api.yml": "config/mediamtx.api.yml",
}


def package(
    project_dir: Path,
    output_path: Path,
    images: list[str],
) -> None:
    if output_path.exists():
        raise FileExistsError(
            f"输出已存在，请先移动或改用新文件名: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.building")
    temporary.unlink(missing_ok=True)
    process: subprocess.Popen[bytes] | None = None
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=3,
            allowZip64=True,
        ) as archive:
            process = subprocess.Popen(
                ["docker", "image", "save", *images],
                stdout=subprocess.PIPE,
                stderr=None,
            )
            assert process.stdout is not None
            with archive.open(
                "images.tar",
                mode="w",
                force_zip64=True,
            ) as target:
                while chunk := process.stdout.read(8 * 1024 * 1024):
                    target.write(chunk)
            process.stdout.close()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"docker image save失败，退出码={return_code}"
                )
            for source, destination in FILES.items():
                archive.write(project_dir / source, destination)
            archive.writestr("engines/.keep", "")
        temporary.replace(output_path)
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="流式生成DeepStream离线ZIP，不落盘中间images.tar",
    )
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", action="append", required=True)
    args = parser.parse_args()
    package(
        args.project_dir.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.image,
    )


if __name__ == "__main__":
    main()
