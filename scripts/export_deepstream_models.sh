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

export_dependencies_checked=0
require_export_dependencies() {
    if [[ "$export_dependencies_checked" == "1" ]]; then
        return
    fi
    if ! "$python_path" -c \
        'import torch, ultralytics, onnx, onnxslim, onnxscript' \
        >/dev/null 2>&1; then
        echo "错误：模型需要重新导出，但导出依赖不完整。" >&2
        echo "执行：$project_dir/.venv/bin/pip install onnx onnxslim onnxscript" >&2
        exit 1
    fi
    export_dependencies_checked=1
}

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
    require_export_dependencies
    echo "导出：$(basename -- "$model") -> ONNX（动态batch 1/2）"
    "$python_path" "$script_dir/export_yolo26_deepstream.py" \
        --weights "$model" \
        --size "$image_size" \
        --simplify
done

world_weights="$project_dir/models/events/yolov8s-worldv2.pt"
garbage_onnx="$project_dir/models/events/yolo_world_garbage.onnx"
garbage_labels="$project_dir/models/events/yolo_world_garbage.labels.txt"
if [[ -f "$world_weights" ]] && {
    [[ ! -s "$garbage_onnx" ]] \
        || [[ ! -s "$garbage_labels" ]] \
        || [[ "$garbage_onnx" -ot "$world_weights" ]];
}; then
    require_export_dependencies
    if ! "$python_path" -c 'import clip' >/dev/null 2>&1; then
        echo "错误：垃圾模型导出需要Ultralytics CLIP。" >&2
        echo "执行：$project_dir/.venv/bin/pip install 'git+https://github.com/ultralytics/CLIP.git'" >&2
        exit 1
    fi
    echo "导出固定垃圾词表的YOLO-World模型..."
    "$python_path" "$script_dir/export_yolo_world_garbage.py" \
        --weights "$world_weights" \
        --output "$garbage_onnx" \
        --size "$image_size" \
        --simplify
fi

if [[ ! -s "$garbage_onnx" || ! -s "$garbage_labels" ]]; then
    echo "错误：缺少垃圾识别ONNX或标签文件：$project_dir/models/events" >&2
    exit 1
fi

pile_weights="$project_dir/models/events/street_garbage_pile.pt"
pile_onnx="$project_dir/models/events/street_garbage_pile.onnx"
pile_labels="$project_dir/models/events/street_garbage_pile.labels.txt"
if [[ -f "$pile_weights" ]] && {
    [[ ! -s "$pile_onnx" ]] \
        || [[ ! -s "$pile_labels" ]] \
        || [[ "$pile_onnx" -ot "$pile_weights" ]];
}; then
    require_export_dependencies
    echo "导出街景垃圾堆YOLO11模型..."
    "$python_path" "$script_dir/export_street_garbage_deepstream.py" \
        --weights "$pile_weights" \
        --output "$pile_onnx" \
        --size "$image_size" \
        --simplify
fi

if [[ ! -s "$pile_onnx" || ! -s "$pile_labels" ]]; then
    echo "错误：缺少街景垃圾堆ONNX或标签文件：$project_dir/models/events" >&2
    exit 1
fi
