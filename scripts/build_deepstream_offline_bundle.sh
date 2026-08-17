#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
python_path="${PYTHON_PATH:-$project_dir/.venv/bin/python}"
output_path="${1:-$project_dir/dist/rtsp-yolo-deepstream8-amd64.zip}"
image_name="rtsp-yolo-annotator:deepstream8-amd64"
base_cache_image="rtsp-yolo-annotator:deepstream8-amd64-base-cache"
mediamtx_image="bluenviron/mediamtx:1"
reuse_existing_image="${REUSE_EXISTING_IMAGE:-0}"

if ! command -v docker >/dev/null 2>&1; then
    echo "错误：找不到docker，请先启动Docker Desktop。" >&2
    exit 1
fi
if [[ ! -x "$python_path" ]]; then
    echo "错误：找不到Python：$python_path" >&2
    exit 1
fi

gas_model="$project_dir/models/gas/yoloe-26l-seg.pt"
gas_profile="$project_dir/models/gas/profiles/camera_01_ir.json"
gas_reference="$project_dir/models/gas/profiles/camera_01_ir.jpg"
for required_gas_asset in "$gas_model" "$gas_profile" "$gas_reference"; do
    if [[ ! -f "$required_gas_asset" ]]; then
        echo "错误：燃气瓶离线资源缺失：$required_gas_asset" >&2
        exit 1
    fi
done

"$script_dir/export_deepstream_models.sh"

cleanup() {
    if docker image inspect "$base_cache_image" >/dev/null 2>&1; then
        docker image rm "$base_cache_image" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ "$reuse_existing_image" == "1" ]]; then
    if ! docker image inspect "$image_name" >/dev/null 2>&1; then
        echo "错误：REUSE_EXISTING_IMAGE=1，但本机没有$image_name" >&2
        exit 1
    fi
    echo "复用已有DeepStream依赖和解析器，仅更新代码与模型..."
    docker tag "$image_name" "$base_cache_image"
    docker buildx build \
        --platform linux/amd64 \
        --file "$project_dir/Dockerfile.deepstream.incremental" \
        --build-arg "BASE_IMAGE=$base_cache_image" \
        --tag "$image_name" \
        --load \
        "$project_dir"
else
    echo "构建DeepStream 8 linux/amd64镜像..."
    docker buildx build \
        --platform linux/amd64 \
        --file "$project_dir/Dockerfile.deepstream" \
        --tag "$image_name" \
        --load \
        "$project_dir"
fi

if ! docker image inspect "$mediamtx_image" >/dev/null 2>&1; then
    docker pull --platform linux/amd64 "$mediamtx_image"
fi

annotator_arch="$(
    docker image inspect "$image_name" --format '{{.Architecture}}'
)"
mediamtx_arch="$(
    docker image inspect "$mediamtx_image" --format '{{.Architecture}}'
)"
if [[ "$annotator_arch" != "amd64" || "$mediamtx_arch" != "amd64" ]]; then
    echo "错误：镜像架构不正确：api=$annotator_arch mediamtx=$mediamtx_arch" >&2
    exit 1
fi

echo "流式导出并压缩离线镜像（只占用一份压缩结果空间）..."
"$python_path" "$script_dir/package_deepstream_bundle.py" \
    --project-dir "$project_dir" \
    --output "$output_path" \
    --image "$image_name" \
    --image "$mediamtx_image"
echo "完成：$output_path"
du -h "$output_path"
