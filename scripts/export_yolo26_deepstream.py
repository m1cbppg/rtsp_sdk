"""Export Ultralytics YOLO26 weights for the DeepStream-Yolo parser.

Adapted from DeepStream-Yolo's MIT-licensed utils/export_yolo26.py at commit
2894babce8e75c49115dbe0c7b516289ed853565.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect, v10Detect
import ultralytics.models.yolo
import ultralytics.utils
import ultralytics.utils.tal as ultralytics_tal


sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
sys.modules["ultralytics.yolo.utils"] = ultralytics.utils


def _dist2bbox(
    distance: torch.Tensor,
    anchor_points: torch.Tensor,
    xywh: bool = False,
    dim: int = -1,
) -> torch.Tensor:
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


def _forward_deepstream(self: Any, values: list[torch.Tensor]) -> Any:
    detached = [value.detach() for value in values]
    if hasattr(self, "inference"):
        one_to_one = [
            torch.cat(
                (
                    self.one2one_cv2[index](detached[index]),
                    self.one2one_cv3[index](detached[index]),
                ),
                1,
            )
            for index in range(self.nl)
        ]
        return self.inference(one_to_one)
    one_to_one = self.forward_head(detached, **self.one2one)
    return self._inference(one_to_one)


def _load_model(weights: Path) -> tuple[nn.Module, dict[int, str]]:
    wrapper = YOLO(str(weights))
    model = deepcopy(wrapper.model).to(torch.device("cpu"))
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for module in model.modules():
        if isinstance(module, (Detect, v10Detect)):
            module.dynamic = False
            module.export = True
            module.format = "onnx"
            if module.__class__.__name__ == "Detect":
                module.forward = types.MethodType(
                    _forward_deepstream,
                    module,
                )
        elif isinstance(module, C2f):
            module.forward = module.forward_split
    return model, {int(key): str(value) for key, value in model.names.items()}


def export(
    weights: Path,
    *,
    size: int,
    opset: int,
    simplify: bool,
) -> tuple[Path, Path]:
    model, names = _load_model(weights)
    labels_path = weights.with_name(f"{weights.stem}.labels.txt")
    labels_path.write_text(
        "".join(f"{names[index]}\n" for index in sorted(names)),
        encoding="utf-8",
    )
    wrapped = nn.Sequential(model, DeepStreamOutput())
    example = torch.zeros(2, 3, size, size)
    onnx_path = weights.with_suffix(".onnx")
    torch.onnx.export(
        wrapped,
        example,
        str(onnx_path),
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

        simplified = onnxslim.slim(onnx.load(str(onnx_path)))
        onnx.save(simplified, str(onnx_path))
    legacy_external_data = onnx_path.with_suffix(".onnx.data")
    legacy_external_data.unlink(missing_ok=True)
    return onnx_path, labels_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="导出YOLO26 DeepStream动态批次ONNX",
    )
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--size", default=640, type=int)
    parser.add_argument("--opset", default=18, type=int)
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args(argv)
    weights = args.weights.expanduser().resolve()
    if not weights.is_file() or weights.suffix != ".pt":
        raise RuntimeError(f"无效.pt模型: {weights}")
    warnings.filterwarnings("ignore")
    onnx_path, labels_path = export(
        weights,
        size=args.size,
        opset=args.opset,
        simplify=args.simplify,
    )
    print(f"ONNX: {onnx_path}")
    print(f"标签: {labels_path}")


if __name__ == "__main__":
    main()
