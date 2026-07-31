from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


COCO_LABELS_ZH: dict[str, str] = {
    "person": "人员",
    "bicycle": "自行车",
    "car": "汽车",
    "motorcycle": "摩托车",
    "airplane": "飞机",
    "bus": "公交车",
    "train": "火车",
    "truck": "卡车",
    "boat": "船",
    "traffic light": "交通信号灯",
    "fire hydrant": "消防栓",
    "stop sign": "停止标志",
    "parking meter": "停车计时器",
    "bench": "长椅",
    "bird": "鸟",
    "cat": "猫",
    "dog": "狗",
    "horse": "马",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "大象",
    "bear": "熊",
    "zebra": "斑马",
    "giraffe": "长颈鹿",
    "backpack": "背包",
    "umbrella": "雨伞",
    "handbag": "手提包",
    "tie": "领带",
    "suitcase": "行李箱",
    "frisbee": "飞盘",
    "skis": "滑雪板",
    "snowboard": "单板滑雪板",
    "sports ball": "球",
    "kite": "风筝",
    "baseball bat": "棒球棒",
    "baseball glove": "棒球手套",
    "skateboard": "滑板",
    "surfboard": "冲浪板",
    "tennis racket": "网球拍",
    "bottle": "瓶子",
    "wine glass": "酒杯",
    "cup": "杯子",
    "fork": "叉子",
    "knife": "刀",
    "spoon": "勺子",
    "bowl": "碗",
    "banana": "香蕉",
    "apple": "苹果",
    "sandwich": "三明治",
    "orange": "橙子",
    "broccoli": "西兰花",
    "carrot": "胡萝卜",
    "hot dog": "热狗",
    "pizza": "披萨",
    "donut": "甜甜圈",
    "cake": "蛋糕",
    "chair": "椅子",
    "couch": "沙发",
    "potted plant": "盆栽",
    "bed": "床",
    "dining table": "餐桌",
    "toilet": "马桶",
    "tv": "电视",
    "laptop": "笔记本电脑",
    "mouse": "鼠标",
    "remote": "遥控器",
    "keyboard": "键盘",
    "cell phone": "手机",
    "microwave": "微波炉",
    "oven": "烤箱",
    "toaster": "烤面包机",
    "sink": "水槽",
    "refrigerator": "冰箱",
    "book": "书",
    "clock": "时钟",
    "vase": "花瓶",
    "scissors": "剪刀",
    "teddy bear": "泰迪熊",
    "hair drier": "吹风机",
    "toothbrush": "牙刷",
}

DEFAULT_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("C:/Windows/Fonts/msyh.ttc"),
)


def contains_chinese(value: str) -> bool:
    return any(
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
        for character in value
    )


def load_label_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"中文标签映射文件无效: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("中文标签映射必须是 JSON 对象")

    mapping: dict[str, str] = {}
    for raw_key, raw_value in payload.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise ValueError("中文标签映射的键和值都必须是字符串")
        key = raw_key.strip()
        value = raw_value.strip()
        if not key or not value:
            raise ValueError("中文标签映射的键和值不能为空")
        if not contains_chinese(value):
            raise ValueError(f"标签 {key!r} 的显示名称必须包含中文")
        mapping[key] = value
        mapping.setdefault(key.casefold(), value)
    return mapping


def chinese_label(
    class_id: int,
    original_name: str,
    custom_mapping: Mapping[str, str] | None = None,
) -> str:
    name = str(original_name).strip()
    mapping = custom_mapping or {}
    for key in (str(class_id), name, name.casefold()):
        translated = mapping.get(key)
        if translated:
            return translated
    if contains_chinese(name):
        return name
    translated = COCO_LABELS_ZH.get(name.casefold())
    return translated if translated is not None else f"类别{class_id}"


def translated_names(
    names: Mapping[Any, Any] | Sequence[Any] | None,
    custom_mapping: Mapping[str, str] | None = None,
) -> dict[int, str]:
    if isinstance(names, Mapping):
        items = names.items()
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        items = enumerate(names)
    else:
        items = ()
    return {
        int(class_id): chinese_label(
            int(class_id),
            str(original_name),
            custom_mapping,
        )
        for class_id, original_name in items
    }


def resolve_chinese_font(configured_path: Path | None = None) -> Path:
    candidates: list[Path] = []
    if configured_path is not None:
        candidates.append(configured_path.expanduser())
    candidates.extend(DEFAULT_FONT_CANDIDATES)

    windows_directory = os.environ.get("WINDIR")
    if windows_directory:
        candidates.append(Path(windows_directory, "Fonts", "msyh.ttc"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if configured_path is not None:
        raise RuntimeError(f"中文字体文件不存在: {configured_path}")
    raise RuntimeError(
        "未找到中文字体。macOS 通常自带苹方；Ubuntu 请安装 fonts-noto-cjk，"
        "或用 --font 指定中文 .ttf/.ttc/.otf 文件"
    )
