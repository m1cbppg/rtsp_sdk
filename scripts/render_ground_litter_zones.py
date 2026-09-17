#!/usr/bin/env python3
"""Render reviewed ground-litter zones and the native-pixel tile plan.

The overlay is produced by the same ``ground_litter_geometry.prepare`` call the
runtime uses, so the picture always matches the tile plan that will actually be
analysed. Output is local-only: no camera address, credential or network access
is involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from rtsp_annotator.ground_litter_detection import (
    GroundLitterDetectionOptions,
    build_ground_litter_tiles,
    ground_litter_camera_view,
)
from rtsp_annotator.ground_litter_geometry import polygon_points, prepare

ZONE_COLOR = (0, 220, 220)
EXCLUSION_COLOR = (0, 0, 255)
OVERLAY_EXCLUSION_COLOR = (0, 140, 255)
TILE_COLOR = (255, 200, 0)
MASK_COLOR = (0, 90, 0)


def resolve(profiles_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for base in (profiles_path.parent, profiles_path.parent.parent, Path.cwd()):
        resolved = base / candidate
        if resolved.is_file():
            return resolved
    return Path(value)


def render(
    image_path: Path,
    camera: dict,
    options: GroundLitterDetectionOptions,
    output: Path,
    *,
    draw_masks: bool,
) -> dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"无法读取参考图: {image_path}")
    height, width = frame.shape[:2]
    if [width, height] != list(camera["reference_size"]):
        raise ValueError(
            f"{image_path.name} 分辨率 {width}x{height} 与 profile "
            f"{camera['reference_size']} 不一致"
        )
    camera_view = ground_litter_camera_view(options, width, height)
    masks, tiles, coverage = prepare(
        camera_view,
        frame,
        int(options.tile_size_px),
        float(options.tile_overlap),
    )
    annotated = frame.copy()
    if draw_masks:
        tint = np.zeros_like(annotated)
        for mask in masks.values():
            tint[mask.astype(bool)] = MASK_COLOR
        annotated = cv2.addWeighted(annotated, 0.75, tint, 0.25, 0)
    for zone in options.zones:
        points = polygon_points(
            [list(point) for point in zone.polygon],
            width,
            height,
        )
        cv2.polylines(annotated, [points], True, ZONE_COLOR, 6)
        for exclusion in zone.exclude_zones:
            cv2.polylines(
                annotated,
                [
                    polygon_points(
                        [list(point) for point in exclusion],
                        width,
                        height,
                    )
                ],
                True,
                EXCLUSION_COLOR,
                5,
            )
        label_x = int(min(point[0] for point in zone.polygon) * width)
        label_y = int(min(point[1] for point in zone.polygon) * height)
        cv2.putText(
            annotated,
            f"{zone.region_id} >={zone.minimum_short_side_px}px/"
            f"{zone.minimum_box_area_px}px2",
            (max(label_x - 10, 4), max(label_y - 12, 26)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            ZONE_COLOR,
            3,
            cv2.LINE_AA,
        )
    for exclusion in options.overlay_exclude_zones:
        cv2.polylines(
            annotated,
            [
                polygon_points(
                    [list(point) for point in exclusion],
                    width,
                    height,
                )
            ],
            True,
            OVERLAY_EXCLUSION_COLOR,
            5,
        )
    for index, (x, y, right, bottom) in enumerate(tiles):
        cv2.rectangle(annotated, (x, y), (right, bottom), TILE_COLOR, 3)
        cv2.putText(
            annotated,
            f"tile{index}",
            (x + 12, y + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            TILE_COLOR,
            3,
            cv2.LINE_AA,
        )
    header = (
        f"{image_path.name} | {width}x{height} | tiles={len(tiles)} "
        f"| ground={coverage}px ({coverage / (width * height):.2%})"
    )
    cv2.rectangle(annotated, (0, 0), (width, 70), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        header,
        (16, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"无法写出 {output}")
    return {
        "image": str(image_path),
        "rendered": str(output),
        "size": [width, height],
        "tiles": [list(tile) for tile in tiles],
        "tile_count": len(tiles),
        "ground_pixels": coverage,
        "ground_fraction": round(coverage / (width * height), 6),
        "regions": [
            {
                "region_id": zone.region_id,
                "mask_pixels": int(masks[zone.region_id].sum()),
                "minimum_short_side_px": zone.minimum_short_side_px,
                "minimum_box_area_px": zone.minimum_box_area_px,
            }
            for zone in options.zones
        ],
    }


def contact_sheet(images: list[Path], output: Path, *, columns: int = 1) -> None:
    loaded = [cv2.imread(str(path)) for path in images]
    loaded = [item for item in loaded if item is not None]
    if not loaded:
        raise ValueError("没有可拼接的图片")
    width = min(item.shape[1] for item in loaded)
    resized = [
        cv2.resize(
            item,
            (width, round(item.shape[0] * width / item.shape[1])),
            interpolation=cv2.INTER_AREA,
        )
        for item in loaded
    ]
    rows = [
        resized[index : index + columns]
        for index in range(0, len(resized), columns)
    ]
    heights = [max(item.shape[0] for item in row) for row in rows]
    canvas = np.zeros((sum(heights) + 8 * (len(rows) - 1), width, 3), np.uint8)
    offset = 0
    for row, row_height in zip(rows, heights):
        for item in row:
            canvas[offset : offset + item.shape[0], : item.shape[1]] = item
        offset += row_height + 8
    if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"无法写出 {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--single-camera",
        action="store_true",
        help="profile 是单机位文件（cameras[0]），忽略 --device",
    )
    args = parser.parse_args()
    payload = json.loads(args.profiles.read_text(encoding="utf-8"))
    cameras = payload.get("cameras") or []
    if args.single_camera:
        camera = cameras[0]
    else:
        camera = next(
            (item for item in cameras if item["device_code"] == args.device),
            None,
        )
    if camera is None:
        parser.error("profile 中找不到该设备号")
    model = payload["model"]
    results = []
    rendered: list[Path] = []
    for mode in ("day", "night"):
        settings = camera.get(mode) or {}
        reference = settings.get("reference_image") or camera.get(
            "reference_image"
        )
        if not reference:
            continue
        options = GroundLitterDetectionOptions(
            enabled=True,
            tile_size_px=int(model.get("tile_size_px", 640)),
            tile_overlap=float(model.get("tile_overlap", 0.2)),
            nms_iou=float(model.get("nms_iou", 0.5)),
            zones=tuple(
                _zone_from_profile(zone) for zone in camera["zones"]
            ),
            overlay_exclude_zones=tuple(
                tuple((float(point[0]), float(point[1])) for point in polygon)
                for polygon in camera.get("overlay_exclude_zones", [])
            ),
        )
        options.validate()
        path = resolve(args.profiles, reference)
        output = args.output / (
            f"{camera['device_code']}-{mode}-zones.jpg"
        )
        results.append(
            render(path, camera, options, output, draw_masks=mode == "day")
        )
        rendered.append(output)
    sheet = args.output / f"{camera['device_code']}-zones-contact.jpg"
    contact_sheet(rendered, sheet)
    summary = {
        "profiles": str(args.profiles),
        "device_code": camera["device_code"],
        "profile_status": camera.get("calibration_status"),
        "notes": camera.get("notes", ""),
        "contact_sheet": str(sheet),
        "renders": results,
    }
    (args.output / "zones_render.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _zone_from_profile(zone: dict):
    from rtsp_annotator.ground_litter_detection import GroundLitterZone

    return GroundLitterZone(
        region_id=zone["region_id"],
        polygon=tuple(
            (float(point[0]), float(point[1])) for point in zone["polygon"]
        ),
        name=zone.get("name", ""),
        exclude_zones=tuple(
            tuple(
                (float(point[0]), float(point[1])) for point in polygon
            )
            for polygon in zone.get("exclude_zones", [])
        ),
        minimum_short_side_px=int(zone.get("minimum_short_side_px", 12)),
        minimum_box_area_px=int(zone.get("minimum_box_area_px", 160)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
