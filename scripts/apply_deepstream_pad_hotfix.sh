#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd -- "$script_dir/.." && pwd)"
deploy_dir="${1:-$HOME/rtsp-cuda}"
compose_file="$deploy_dir/docker-compose.deepstream.api.yml"
image="rtsp-yolo-annotator:deepstream8-amd64"
backup_image="rtsp-yolo-annotator:deepstream8-amd64-before-padfix"

if [[ ! -f "$compose_file" ]]; then
    echo "错误：找不到Compose文件：$compose_file" >&2
    echo "用法：$0 [服务器部署目录]" >&2
    exit 1
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "错误：服务器尚未加载镜像：$image" >&2
    exit 1
fi

if ! docker image inspect "$backup_image" >/dev/null 2>&1; then
    docker tag "$image" "$backup_image"
fi

docker build \
    --pull=false \
    --network=none \
    --build-arg "BASE_IMAGE=$backup_image" \
    --file "$package_dir/Dockerfile.deepstream.pad-hotfix" \
    --tag "$image" \
    "$package_dir"

# MediaMTX does not contain application code and does not need to be
# interrupted. Recreate only the API container; its worker subprocesses will
# be terminated, so callers must register their streams again afterwards.
docker compose -f "$compose_file" up -d mediamtx
docker compose \
    -f "$compose_file" \
    up -d --no-deps --force-recreate api

docker compose -f "$compose_file" ps
