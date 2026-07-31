#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
python_path="${PYTHON_PATH:-$project_dir/.venv/bin/python}"
image_size="${DEEPSTREAM_IMGSZ:-640}"
export_cache_dir="$(mktemp -d)"
cleanup() {
    rm -rf -- "$export_cache_dir"
}
trap cleanup EXIT
export MPLCONFIGDIR="$export_cache_dir/matplotlib"
export XDG_CACHE_HOME="$export_cache_dir/xdg"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

if [[ ! -x "$python_path" ]]; then
    echo "错误：找不到Python：$python_path" >&2
    echo "请先在项目目录创建.venv，或设置PYTHON_PATH。" >&2
    exit 1
fi

if ! "$python_path" -c \
    'import torch, ultralytics, onnx, onnxslim, onnxscript' \
    >/dev/null 2>&1; then
    echo "错误：模型导出依赖不完整。" >&2
    echo "执行：$project_dir/.venv/bin/pip install onnx onnxslim onnxscript" >&2
    exit 1
fi

shopt -s nullglob
models=("$project_dir"/models/*.pt)
if (( ${#models[@]} == 0 )); then
    echo "错误：models目录没有.pt模型。" >&2
    exit 1
fi

for model in "${models[@]}"; do
    onnx_path="${model%.pt}.onnx"
    labels_path="${model%.pt}.labels.txt"
    if [[ -s "$onnx_path" && -s "$labels_path" \
        && "$onnx_path" -nt "$model" \
        && "$labels_path" -nt "$model" ]]; then
        echo "复用已导出的模型：$(basename -- "$onnx_path")"
        continue
    fi
    echo "导出：$(basename -- "$model") -> ONNX（动态batch 1/2）"
    "$python_path" "$script_dir/export_yolo26_deepstream.py" \
        --weights "$model" \
        --size "$image_size" \
        --simplify
done
