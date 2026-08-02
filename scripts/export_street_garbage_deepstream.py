"""Export a YOLO11 street-garbage detector for DeepStream-Yolo.

The deployed graph emits ``x1,y1,x2,y2,score,class_id`` rows so it can use
the same pinned DeepStream-Yolo parser as the main detector.  The source
checkpoint is only required while exporting and is not shipped in the image.
"""

from __future__ import annotations

import argparse
import warnings
from copy import deepcopy
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect
import ultralytics.utils.tal as ultralytics_tal


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


ultralytics_tal.dist2bbox.__code__ = _dist2bbox.__code__


class DeepStreamOutput(nn.Module):
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


def _load_model(weights: Path) -> tuple[nn.Module, dict[int, str]]:
    wrapper = YOLO(str(weights))
    model = deepcopy(wrapper.model).to(torch.device("cpu"))
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for module in model.modules():
        if isinstance(module, Detect):
            # YOLO11 uses the regular dense one-to-many head.  Let its native
            # export path produce [B, 4 + classes, anchors], then collapse it
            # to the parser's Nx6 layout in DeepStreamOutput.
            module.dynamic = True
            module.export = True
            module.format = "onnx"
        elif isinstance(module, C2f):
            module.forward = module.forward_split
    names = {int(key): str(value) for key, value in model.names.items()}
    return model, names


def export(
    weights: Path,
    *,
    output: Path,
    size: int,
    opset: int,
    simplify: bool,
) -> tuple[Path, Path]:
    model, names = _load_model(weights)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels_path = output.with_name(f"{output.stem}.labels.txt")
    labels_path.write_text(
        "".join(f"{names[index]}\n" for index in sorted(names)),
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
        # The legacy exporter is deliberate: the project pins an ONNX graph
        # shape already exercised by DeepStream-Yolo/TensorRT.
        dynamo=False,
    )
    if simplify:
        import onnxslim

        simplified = onnxslim.slim(onnx.load(str(output)))
        onnx.save(simplified, str(output))
    output.with_suffix(".onnx.data").unlink(missing_ok=True)
    return output, labels_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="导出街景垃圾堆YOLO11 DeepStream动态批次ONNX",
    )
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument(
        "--output",
        default=Path("models/events/street_garbage_pile.onnx"),
        type=Path,
    )
    parser.add_argument("--size", default=640, type=int)
    parser.add_argument("--opset", default=18, type=int)
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args(argv)
    weights = args.weights.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not weights.is_file() or weights.suffix != ".pt":
        raise RuntimeError(f"无效YOLO11模型: {weights}")
    warnings.filterwarnings("ignore")
    onnx_path, labels_path = export(
        weights,
        output=output,
        size=args.size,
        opset=args.opset,
        simplify=args.simplify,
    )
    print(f"ONNX: {onnx_path}")
    print(f"标签: {labels_path}")


if __name__ == "__main__":
    main()
