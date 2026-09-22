"""Build the reviewed afternoon Clean Reference profile for camera 01."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


PROFILE_KIND = "ground_litter_clean_reference_v32"
PROFILE_ID = "camera_01_v32_afternoon_1080p"
OUTPUT_SIZE = (1920, 1080)

# Clockwise boundary of the visible sidewalk. The left edge follows the curb;
# the right edge follows the wall/floor seam instead of using a trapezoid.
GROUND_ROI = (
    (0.500, 0.160),
    (0.625, 0.160),
    (0.650, 0.220),
    (0.670, 0.320),
    (0.690, 0.430),
    (0.720, 0.560),
    (0.750, 0.700),
    (0.780, 0.840),
    (0.810, 0.960),
    (0.370, 0.960),
    (0.390, 0.840),
    (0.410, 0.700),
    (0.430, 0.560),
    (0.450, 0.430),
    (0.470, 0.320),
    (0.488, 0.220),
)

# This fixture is merchant equipment rather than observable ground. It moved
# between the morning reference and afternoon acceptance stream.
ZONE_EXCLUDES = (
    (
        (0.585, 0.145),
        (0.665, 0.145),
        (0.675, 0.315),
        (0.595, 0.315),
    ),
)

OVERLAY_EXCLUDES = (
    ((0.0, 0.0), (0.44, 0.0), (0.44, 0.12), (0.0, 0.12)),
    ((0.70, 0.835), (0.88, 0.835), (0.88, 0.94), (0.70, 0.94)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def points(polygon: tuple[tuple[float, float], ...]) -> np.ndarray:
    width, height = OUTPUT_SIZE
    return np.round(
        np.asarray(polygon, np.float32) * np.asarray([width, height])
    ).astype(np.int32)


def read_resized(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)


def build(args: argparse.Namespace) -> Path:
    reference = read_resized(args.reference)
    comparison = [read_resized(path) for path in args.comparison]

    valid = np.zeros(reference.shape[:2], np.uint8)
    cv2.fillPoly(valid, [points(GROUND_ROI)], 255)
    for polygon in (*ZONE_EXCLUDES, *OVERLAY_EXCLUDES):
        cv2.fillPoly(valid, [points(polygon)], 0)

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    maximum_delta = np.zeros(reference_gray.shape, np.uint8)
    for image in comparison:
        delta = cv2.absdiff(
            reference_gray,
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        )
        maximum_delta = np.maximum(maximum_delta, delta)
    # Only mark strong temporal changes (people/vehicles/merchant movement).
    # Ordinary afternoon illumination drift is handled by protected_normalize;
    # placing it in this mask would desensitize nearly the whole sidewalk.
    tolerance = (maximum_delta >= 96).astype(np.uint8) * 255
    tolerance = cv2.dilate(tolerance, np.ones((7, 7), np.uint8))
    tolerance[valid == 0] = 0

    output = args.output / PROFILE_ID
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "reference.png": reference,
        "valid_mask.png": valid,
        "daylight_tolerance.png": tolerance,
    }
    for name, image in paths.items():
        if not cv2.imwrite(str(output / name), image):
            raise RuntimeError(f"cannot write {output / name}")

    metadata = {
        "kind": PROFILE_KIND,
        "profile_id": PROFILE_ID,
        "reference_size": list(OUTPUT_SIZE),
        "created": "2026-09-18",
        "source_reference": str(args.reference),
        "source_reference_sha256": sha256(args.reference),
        "comparison_sha256": {
            path.name: sha256(path) for path in args.comparison
        },
        "ground_roi": [list(point) for point in GROUND_ROI],
        "zone_exclude_zones": [
            [list(point) for point in polygon] for polygon in ZONE_EXCLUDES
        ],
        "overlay_exclude_zones": [
            [list(point) for point in polygon] for polygon in OVERLAY_EXCLUDES
        ],
        "sha256": {
            name: sha256(output / name) for name in paths
        },
        "baseline": (
            "Camera 01 afternoon production hardening reference; reviewed "
            "sidewalk polygon; startup evidence suppression required"
        ),
    }
    (output / "profile.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    preview = reference.copy()
    shade = preview.copy()
    cv2.fillPoly(shade, [points(GROUND_ROI)], (0, 190, 255))
    preview = cv2.addWeighted(shade, 0.18, preview, 0.82, 0)
    cv2.polylines(preview, [points(GROUND_ROI)], True, (0, 230, 255), 4)
    for polygon in ZONE_EXCLUDES:
        cv2.polylines(preview, [points(polygon)], True, (0, 0, 255), 4)
    preview_path = args.preview / "camera_01_v32_afternoon_roi_review.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"cannot write {preview_path}")

    print(json.dumps({
        "profile": str(output),
        "preview": str(preview_path),
        "valid_fraction": round(float(np.count_nonzero(valid) / valid.size), 4),
        "tolerance_fraction_of_valid": round(float(
            np.count_nonzero(tolerance) / max(np.count_nonzero(valid), 1)
        ), 4),
    }, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("models/litter/profiles")
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("output/ground_litter_v32_production_hardening_20260918"),
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
