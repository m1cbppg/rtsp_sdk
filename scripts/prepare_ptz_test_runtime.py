#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import secrets
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="准备独立的船舶识别/PTZ闭环测试目录",
    )
    parser.add_argument("video", type=Path, help="含船回放视频")
    parser.add_argument(
        "--public-host",
        default="127.0.0.1",
        help="测试输出RTSP中使用的服务器地址",
    )
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=Path("/home/sf01/rtsp-deepstream/engines"),
        help="现有TensorRT engine目录",
    )
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--force", action="store_true", help="覆盖已有测试视频")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"视频不存在: {video}")
    if args.start < 0 or (args.end is not None and args.end <= args.start):
        raise SystemExit("--start/--end时间范围无效")
    if not args.engine_dir.expanduser().is_dir():
        raise SystemExit(f"engine目录不存在: {args.engine_dir}")

    root = Path("ptz-test-runtime")
    config_dir = root / "config"
    fixture_dir = root / "fixtures"
    for path in (
        config_dir,
        fixture_dir,
        root / "data",
        root / "virtual-data",
    ):
        path.mkdir(parents=True, exist_ok=True)

    destination = fixture_dir / "boat.mp4"
    if destination.exists() and not args.force:
        raise SystemExit(f"{destination}已存在；确认覆盖请加--force")
    if video != destination.resolve():
        shutil.copy2(video, destination)

    api_key = secrets.token_hex(24)
    control_key = secrets.token_hex(24)
    publish_password = secrets.token_urlsafe(24)
    read_password = secrets.token_urlsafe(24)
    publish_user = "ptztestpub"
    read_user = "ptztestview"

    config = json.loads(Path("config/api.deepstream.example.json").read_text())
    config["api"].update({"key": api_key, "host": "0.0.0.0", "port": 8080})
    config["rtsp"].update(
        {
            "internal_base_url": "rtsp://ptz-test-mediamtx:8554",
            "public_base_url": f"rtsp://{args.public_host}:18554",
            "publish_user": publish_user,
            "publish_password": publish_password,
            "read_user": read_user,
            "read_password": read_password,
        }
    )
    config["inference"]["max_streams"] = 1
    config["events"]["storage_root"] = "/app/data/events"
    (config_dir / "api.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    env_lines = [
        f"PTZ_TEST_API_KEY={api_key}",
        f"PTZ_TEST_CONTROL_KEY={control_key}",
        f"PTZ_TEST_RTSP_PUBLISH_USER={publish_user}",
        f"PTZ_TEST_RTSP_PUBLISH_PASSWORD={publish_password}",
        f"PTZ_TEST_RTSP_READ_USER={read_user}",
        f"PTZ_TEST_RTSP_READ_PASSWORD={read_password}",
        f"PTZ_TEST_ENGINE_DIR={args.engine_dir.expanduser().resolve()}",
        f"PTZ_TEST_SOURCE_START_SECONDS={args.start}",
        f"PTZ_TEST_SOURCE_END_SECONDS={'' if args.end is None else args.end}",
        "PTZ_TEST_MIRROR_REAL_CAMERA=false",
        "REAL_CAMERA_CONTROL_URL=http://camera-control:8080",
        "REAL_CAMERA_ID=river-ptz-01",
        "REAL_CAMERA_CONTROL_API_KEY=",
    ]
    env_path = Path(".env.ptz-test")
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    env_path.chmod(0o600)

    print("隔离测试目录已准备完成")
    print(f"视频: {destination}")
    print(f"配置: {config_dir / 'api.json'}")
    print(f"环境变量: {env_path} (权限0600)")
    print("下一步: docker compose --env-file .env.ptz-test -f docker-compose.ptz-test.yml up -d --build")


if __name__ == "__main__":
    main()
