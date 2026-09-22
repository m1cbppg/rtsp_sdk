#!/usr/bin/env python3
"""Small runtime probe for the frozen-feature candidate classifier experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    model = YOLO(args.model)
    rows = model.embed(source=[str(args.image)], imgsz=320, device=0, verbose=False)
    print("container", type(rows).__name__, "count", len(rows))
    for index, row in enumerate(rows):
        print(index, type(row).__name__, tuple(row.shape), row.dtype, row.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
