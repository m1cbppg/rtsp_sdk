"""Export a fixed-vocabulary YOLO-World v2 model for DeepStream-Yolo.

The text encoder is used only while this script runs.  Its embeddings are
stored in the exported graph, so the deployed container needs neither CLIP
nor network access.
"""

from __future__ import annotations

import argparse
import warnings
from copy import deepcopy
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from ultralytics import YOLOWorld
from ultralytics.nn.modules import C2f, Detect
import ultralytics.utils.tal as ultralytics_tal


DEFAULT_PROMPTS = (
    "plastic bottle",
    "garbage bag",
    "plastic bag",
    "cardboard box",
    "paper waste",
    "can",
    "trash pile",
    "waste",
)


def _dist2bbox(
    distance: torch.Tensor,
    anchor_points: torch.Tensor,
    xywh: bool = False,
    dim: int = -1,
) -> torch.Tensor:
    """Decode boxes as xyxy, matching NvDsInferParseYolo's Nx6 ABI."""
    del xywh
    left_top, right_bottom = distance.chunk(2, dim)
    top_left = anchor_points - left_top
    bottom_right = anchor_points + right_bottom
    return torch.cat((top_left, bottom_right), dim)


# WorldDetect and Detect reference the same function object imported from tal.
# Replacing its code before export makes the first four ONNX output values
# x1/y1/x2/y2, which is the format expected by the pinned DeepStream-Yolo
# parser. Without this, boxes are silently interpreted as malformed xyxy.
ultralytics_tal.dist2bbox.__code__ = _dist2bbox.__code__


class DeepStreamOutput(nn.Module):
    """Convert [box channels, class scores] to DeepStream-Yolo Nx6."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.transpose(1, 2)
        boxes = value[:, :, :4]
        scores, labels = torch.max(
            value[:, :, 4:],
            dim=-1,
            keepdim=True,
        )
        return torch.cat(
            [boxes, scores, labels.to(boxes.dtype)],
            dim=-1,
        )


def _load_model(
    weights: Path,
    prompts: tuple[str, ...],
) -> nn.Module:
    wrapper = YOLOWorld(str(weights))
    # Avoid retaining the 338 MB CLIP encoder. Only the generated text
    # embeddings are needed after this call.
    wrapper.model.set_classes(list(prompts), cache_clip_model=False)
    wrapper.model.names = list(prompts)
    model = deepcopy(wrapper.model).to(torch.device("cpu"))
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for module in model.modules():
        if isinstance(module, Detect):
            module.dynamic = True
            module.export = True
            module.format = "onnx"
        elif isinstance(module, C2f):
            module.forward = module.forward_split
    return model


def export(
    weights: Path,
    *,
    output: Path,
    prompts: tuple[str, ...],
    size: int,
    opset: int,
    simplify: bool,
) -> tuple[Path, Path]:
    if not prompts or any(not item.strip() for item in prompts):
        raise ValueError("垃圾识别提示词不能为空")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = _load_model(weights, prompts)
    labels_path = output.with_name(f"{output.stem}.labels.txt")
    labels_path.write_text(
        "".join(f"{item}\n" for item in prompts),
        encoding="utf-8",
    )
    wrapped = nn.Sequential(model, DeepStreamOutput())
    example = torch.zeros(2, 3, size, size)
    torch.onnx.export(
        wrapped,
        example,
        str(output),
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        external_data=False,
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
    )
    if simplify:
        import onnxslim

        simplified = onnxslim.slim(onnx.load(str(output)))
        onnx.save(simplified, str(output))
    output.with_suffix(".onnx.data").unlink(missing_ok=True)
    return output, labels_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="导出固定垃圾词表的YOLO-World DeepStream ONNX",
    )
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument(
        "--output",
        default=Path("models/events/yolo_world_garbage.onnx"),
        type=Path,
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="可重复传入；未传时使用项目默认垃圾词表",
    )
    parser.add_argument("--size", default=640, type=int)
    parser.add_argument("--opset", default=18, type=int)
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args(argv)
    weights = args.weights.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not weights.is_file() or weights.suffix != ".pt":
        raise RuntimeError(f"无效YOLO-World v2模型: {weights}")
    prompts = tuple(args.prompts or DEFAULT_PROMPTS)
    warnings.filterwarnings("ignore")
    onnx_path, labels_path = export(
        weights,
        output=output,
        prompts=prompts,
        size=args.size,
        opset=args.opset,
        simplify=args.simplify,
    )
    print(f"ONNX: {onnx_path}")
    print(f"标签: {labels_path}")


if __name__ == "__main__":
    main()
