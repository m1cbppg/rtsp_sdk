#!/usr/bin/env python3
"""Assemble the Ground Litter V3.3 dual-recall candidate-image build context.

This script only prepares and verifies a local upload bundle. It never talks to
the server, never builds an image and never touches a production tag. The server
side build is a separate, explicitly authorised step documented in the generated
``GROUND_LITTER_V33_DEPLOY_20260918.md``.

Outputs (under ``dist/ground-litter-v33-dual-recall-20260918/``):

* ``<name>/``                    build context, ready for ``docker build``
* ``<name>-context.tar.gz``      the same context as a single upload file
* ``MANIFEST.json``              SHA-256 of every source file + the tarball
* ``SHA256SUMS.txt``             ``sha256sum -c`` style list
* ``GROUND_LITTER_V33_DEPLOY_20260918.md``  build, verify and rollback steps
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "ground-litter-v33-dual-recall-20260918"
DOCKERFILE_NAME = "Dockerfile.deepstream.ground-litter-v33-update"

# The candidate is a separate tag by construction. ROLLBACK_IMAGE is the tag the
# hardened V3.2 image is known by; both are asserted in the bundle tests so a
# future edit cannot quietly point the candidate at a production tag.
CANDIDATE_IMAGE = (
    "rtsp-yolo-annotator:deepstream8-ground-litter-v33-dual-recall-20260918"
)
ROLLBACK_IMAGE = (
    "rtsp-yolo-annotator:deepstream8-ground-litter-v32-hardening-20260918"
)

# Everything the Dockerfile COPYs, plus the contract documents and the request
# example an operator needs when creating the first hybrid_v33 stream.
CONTEXT_FILES: tuple[str, ...] = (
    DOCKERFILE_NAME,
    "THIRD_PARTY_MODEL_NOTICES.md",
    "config/ground_litter_v33_hybrid_stream_request.example.json",
    "docs/plans/2026-09-18-ground-litter-v33-dual-recall-architecture.md",
    "rtsp_annotator/api.py",
    "rtsp_annotator/deepstream_manager.py",
    "rtsp_annotator/deepstream_worker.py",
    "rtsp_annotator/stream_manager.py",
    "rtsp_annotator/ground_litter_detection.py",
    "rtsp_annotator/ground_litter_process.py",
    "rtsp_annotator/ground_litter_v32.py",
    "rtsp_annotator/ground_litter_v33.py",
    "rtsp_annotator/ground_litter_geometry.py",
    "models/litter/turhancan_yolov8m_seg_trash.pt",
)

# Profile directories are copied wholesale by the Dockerfile; enumerate them so
# the manifest pins exactly what shipped instead of "whatever was on disk".
CONTEXT_DIRS: tuple[str, ...] = ("models/litter/profiles",)

# The very same modules must be installable in the running container. The V3.2
# hardening deploy compared these byte for byte, so keep that list here too.
CONTAINER_COMPARE_FILES: tuple[str, ...] = (
    "rtsp_annotator/api.py",
    "rtsp_annotator/deepstream_manager.py",
    "rtsp_annotator/deepstream_worker.py",
    "rtsp_annotator/stream_manager.py",
    "rtsp_annotator/ground_litter_detection.py",
    "rtsp_annotator/ground_litter_process.py",
    "rtsp_annotator/ground_litter_v32.py",
    "rtsp_annotator/ground_litter_v33.py",
    "rtsp_annotator/ground_litter_geometry.py",
)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def collect_context_files() -> list[Path]:
    files = [REPO_ROOT / name for name in CONTEXT_FILES]
    for directory in CONTEXT_DIRS:
        root = REPO_ROOT / directory
        if not root.is_dir():
            raise SystemExit(f"构建上下文目录缺失: {root}")
        files.extend(sorted(p for p in root.rglob("*") if p.is_file()))
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise SystemExit("以下文件不存在，无法打包:\n  " + "\n  ".join(missing))
    return files


def build_context(files: list[Path], staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    for source in files:
        relative = source.relative_to(REPO_ROOT)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def make_tarball(staging: Path, tar_path: Path) -> None:
    if tar_path.exists():
        tar_path.unlink()
    # Deterministic archive: sorted entries, root ownership, zeroed mtimes and a
    # zeroed gzip header time, so re-running the packager on unchanged sources
    # reproduces the exact SHA-256 recorded in SHA256SUMS.txt.
    with tar_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for path in sorted(staging.rglob("*")):
                    relative = path.relative_to(staging.parent)
                    info = archive.gettarinfo(str(path), arcname=str(relative))
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)


def dockerfile_base_image() -> str:
    for line in (REPO_ROOT / DOCKERFILE_NAME).read_text(encoding="utf-8").splitlines():
        if line.startswith("ARG BASE_IMAGE="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"{DOCKERFILE_NAME} 缺少 BASE_IMAGE 默认值")


def git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def write_deploy_doc(
    path: Path,
    *,
    manifest: dict,
    tarball_name: str,
    tarball_sha: str,
    base_image: str,
) -> None:
    candidate = manifest["candidate_image"]
    file_lines = "\n".join(
        f"| `{name}` | `{digest[:16]}…` |" for name, digest in sorted(manifest["files"].items())
    )
    path.write_text(
        f"""# 零散垃圾 V3.3 双通道召回 — 候选镜像构建与回滚单

日期：{manifest['created']}　契约：`docs/plans/2026-09-18-ground-litter-v33-dual-recall-architecture.md`

> **当前状态：候选镜像尚未构建，本单尚未执行。** 本地只完成了代码、测试、真实权重
> smoke、分来源误报报告、性能报告和本候选构建包。服务器构建与实测需要单独授权，
> 且**不得**用本镜像替换生产标签、**不得**部署。

## 1. 构建包指纹

- 包名：`{tarball_name}`
- 包 SHA-256：`{tarball_sha}`
- Git HEAD：`{manifest['git_head']}`
- 构建上下文文件数：{manifest['file_count']}
- 基础镜像（`{DOCKERFILE_NAME}` 的 `ARG BASE_IMAGE` 默认值）：
  `{base_image}` ← 即线上运行中的 V3.2 硬化镜像，服务器本地已有，**不需要 pull**

完整逐文件 SHA-256：

| 文件 | SHA-256（前16位） |
| --- | --- |
{file_lines}

## 2. 构建（需授权，且只打候选标签）

```bash
cd /home/sf01/rtsp-deepstream
mkdir -p releases/{BUNDLE_NAME}
# 上传前先核对本文件记录的 SHA-256
sha256sum -c SHA256SUMS.txt
tar -xzf {tarball_name} -C releases/{BUNDLE_NAME}
cd releases/{BUNDLE_NAME}/{BUNDLE_NAME}

docker build --platform linux/amd64 \\
  -f {DOCKERFILE_NAME} \\
  -t {candidate} \\
  .
```

构建期断言（失败即中止，不会等到运行期）：

1. `py_compile` 全部 9 个模块；
2. 基础镜像必须存在 `cv2` / `numpy` / `torch` / `ultralytics`，否则打印缺失项退出；
3. `build_ground_litter_tiles` 真实构图；
4. `YOLO()` 真实解析 turhancan 权重并读出类别表；
5. 用假检测器驱动一次 `_analyse_hybrid` tick，断言 `state ∈ (running, warming_up)`。

构建完成后记录实际镜像 ID：

```bash
docker image inspect {candidate} --format '{{{{.Id}}}} {{{{.Created}}}}'
```

## 3. 停止线（本次不执行）

- ❌ 不 `docker compose up`、不重建 `rtsp-yolo-api`；
- ❌ 不改 `docker-compose.*.override.yml`、不改 `config/api.json`；
- ❌ 不给候选镜像打生产标签，不覆盖 `{base_image}`；
- ❌ 不删除任何现有镜像、engine、事件或日志。

## 4. 回滚步骤（仅在将来授权部署后才需要）

当前生产标签与回退标签（{manifest['created']} 记录，部署前必须重新核对）：

| 角色 | 标签 | 镜像 ID 前缀 |
| --- | --- | --- |
| 当前生产 | `{base_image}` | `7aa71d92…` |
| 上一版 V3.2 | `deepstream8-before-ground-litter-v32-hardening-20260918` | `841ca526317f` |
| V3.2 之前 | `deepstream8-before-ground-litter-v32-20260918` | `7563a79fb12b` |
| 本次候选 | `{candidate}` | 构建后填写 |

回滚 = 去掉将来可能加入的 V3.3 override，恢复**当前线上这条 6 文件链**后只重建 API。
部署前先用下面两条只读命令核对真实链与镜像，**不要照抄本文件的历史值**：

```bash
cd /home/sf01/rtsp-deepstream
docker inspect rtsp-yolo-api \\
  --format '{{{{index .Config.Labels "com.docker.compose.project.config_files"}}}}'
docker inspect rtsp-yolo-api --format '{{{{.Image}}}} {{{{.State.StartedAt}}}} {{{{.RestartCount}}}}'
```

核对一致后的回滚命令（当前记录的 6 文件链，最后一个是硬化 override）：

```bash
docker compose -f docker-compose.deepstream.api.yml \\
  -f docker-compose.ptz-v12.override.yml \\
  -f docker-compose.demo-continuous.override.yml \\
  -f docker-compose.ground-litter.override.yml \\
  -f docker-compose.ground-litter-v32.override.yml \\
  -f docker-compose.ground-litter-v32-hardening.override.yml \\
  up -d --no-deps api
```

回滚只需改 Compose 引用，**不必重建镜像**；MediaMTX / camera-control /
web-gateway 不重启。API 容器重启会停掉进程内全部流任务，业务方需重建流。

## 5. 部署前必须补齐的实测（当前为空白）

- 主链 `publish_fps` / `unique_publish_fps` / `duplicate_publish_fps` /
  `pipeline_healthy`、`input_frame_age_ms` 与显存占用；
- 真实 RTSP 下的 semantic-only / prior-only / fused 三源遥测；
- 与本次离线报告同口径的现场误报观察（离线 semantic-only 约 215 事件/小时，
  收紧阈值约 25 事件/小时，**不达标**）；
- 先验通道需要与目标时段匹配的 profile，否则会持续 `abstaining`。
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist",
        default="dist",
        help="输出根目录（相对仓库根，默认 dist）",
    )
    args = parser.parse_args()

    dist_root = REPO_ROOT / args.dist
    bundle_dir = dist_root / BUNDLE_NAME
    staging = bundle_dir / BUNDLE_NAME
    tarball = bundle_dir / f"{BUNDLE_NAME}-context.tar.gz"

    files = collect_context_files()
    build_context(files, staging)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    make_tarball(staging, tarball)

    manifest = {
        "kind": "ground_litter_v33_dual_recall_candidate_bundle",
        "created": "2026-09-18",
        "contract": "docs/plans/2026-09-18-ground-litter-v33-dual-recall-architecture.md",
        "git_head": git_head(),
        "candidate_image": CANDIDATE_IMAGE,
        "rollback_image": ROLLBACK_IMAGE,
        "base_image": dockerfile_base_image(),
        "built": False,
        "deployed": False,
        "file_count": len(files),
        "files": {
            str(source.relative_to(REPO_ROOT)): sha256_file(source) for source in files
        },
        "tarball": {
            "name": tarball.name,
            "sha256": sha256_file(tarball),
            "bytes": tarball.stat().st_size,
        },
        "container_compare_files": list(CONTAINER_COMPARE_FILES),
    }

    (bundle_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (bundle_dir / "SHA256SUMS.txt").write_text(
        f"{manifest['tarball']['sha256']}  {tarball.name}\n", encoding="utf-8"
    )
    write_deploy_doc(
        bundle_dir / "GROUND_LITTER_V33_DEPLOY_20260918.md",
        manifest=manifest,
        tarball_name=tarball.name,
        tarball_sha=manifest["tarball"]["sha256"],
        base_image=manifest["base_image"],
    )

    print(f"bundle dir      : {bundle_dir.relative_to(REPO_ROOT)}")
    print(f"context files   : {len(files)}")
    print(f"context tarball : {tarball.name}  {manifest['tarball']['bytes']} bytes")
    print(f"tarball sha256  : {manifest['tarball']['sha256']}")
    print(f"candidate image : {manifest['candidate_image']} (NOT built)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
