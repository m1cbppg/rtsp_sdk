#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
output_path="${1:-$project_dir/dist/rtsp-yolo-cuda-amd64.tar.gz}"
image_name="rtsp-yolo-annotator:cuda-amd64"
mediamtx_image="bluenviron/mediamtx:1"
reuse_existing_image="${REUSE_EXISTING_IMAGE:-0}"
base_cache_image="rtsp-yolo-annotator:cuda-amd64-base-cache"
ubuntu_mirror="${UBUNTU_MIRROR:-http://mirrors.aliyun.com/ubuntu/}"
ubuntu_mirror="${ubuntu_mirror%/}/"
ubuntu_mirror_secure="${UBUNTU_MIRROR_SECURE:-${ubuntu_mirror/http:/https:}}"
ubuntu_mirror_secure="${ubuntu_mirror_secure%/}/"

if ! command -v docker >/dev/null 2>&1; then
    echo "错误：找不到 docker，请先启动 Docker Desktop。" >&2
    exit 1
fi

if ! compgen -G "$project_dir/models/*.pt" >/dev/null; then
    echo "错误：models 目录下没有 .pt 模型，离线镜像将无法识别。" >&2
    exit 1
fi

mkdir -p "$(dirname -- "$output_path")"
bundle_dir="$(mktemp -d)"
cleanup() {
    rm -rf -- "$bundle_dir"
    if docker image inspect "$base_cache_image" >/dev/null 2>&1; then
        docker image rm "$base_cache_image" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "构建 linux/amd64 CUDA 推理镜像（Apple Silicon 会使用模拟构建）..."
echo "Ubuntu 引导软件源：$ubuntu_mirror"
echo "Ubuntu HTTPS软件源：$ubuntu_mirror_secure"
if [[ "$reuse_existing_image" == "1" ]]; then
    if ! docker image inspect "$image_name" >/dev/null 2>&1; then
        echo "错误：REUSE_EXISTING_IMAGE=1，但本机没有$image_name" >&2
        exit 1
    fi
    echo "复用现有CUDA依赖层，仅更新代码和models目录..."
    docker tag "$image_name" "$base_cache_image"
    docker buildx build \
        --platform linux/amd64 \
        --file "$project_dir/Dockerfile.cuda.incremental" \
        --build-arg "BASE_IMAGE=$base_cache_image" \
        --tag "$image_name" \
        --load \
        "$project_dir"
else
    docker buildx build \
        --platform linux/amd64 \
        --file "$project_dir/Dockerfile.cuda" \
        --build-arg "UBUNTU_MIRROR=$ubuntu_mirror" \
        --build-arg "UBUNTU_MIRROR_SECURE=$ubuntu_mirror_secure" \
        --tag "$image_name" \
        --load \
        "$project_dir"
fi

if docker image inspect "$mediamtx_image" >/dev/null 2>&1; then
    echo "复用本机已有MediaMTX镜像。"
else
    echo "拉取 linux/amd64 MediaMTX 镜像..."
    docker pull --platform linux/amd64 "$mediamtx_image"
fi

annotator_arch="$(docker image inspect "$image_name" --format '{{.Architecture}}')"
mediamtx_arch="$(docker image inspect "$mediamtx_image" --format '{{.Architecture}}')"
if [[ "$annotator_arch" != "amd64" || "$mediamtx_arch" != "amd64" ]]; then
    echo "错误：镜像架构不正确：annotator=$annotator_arch mediamtx=$mediamtx_arch" >&2
    exit 1
fi

echo "导出离线镜像..."
docker image save \
    --output "$bundle_dir/images.tar" \
    "$image_name" \
    "$mediamtx_image"

cp "$project_dir/docker-compose.cuda.yml" "$bundle_dir/"
cp "$project_dir/docker-compose.cuda.api.yml" "$bundle_dir/"
cp "$project_dir/docker-compose.cuda.public.yml" "$bundle_dir/"
cp "$project_dir/HTTP_API.md" "$bundle_dir/"
cp "$project_dir/OFFLINE_CUDA.md" "$bundle_dir/"
cp "$project_dir/PUBLIC_RTSP.md" "$bundle_dir/"
mkdir -p "$bundle_dir/config"
cp "$project_dir/config/api.cuda.example.json" "$bundle_dir/config/"
cp "$project_dir/config/labels.zh.example.json" "$bundle_dir/config/"
cp "$project_dir/config/mediamtx.api.yml" "$bundle_dir/config/"

echo "压缩离线包..."
tar -C "$bundle_dir" -czf "$output_path" .

echo "完成：$output_path"
du -h "$output_path"
