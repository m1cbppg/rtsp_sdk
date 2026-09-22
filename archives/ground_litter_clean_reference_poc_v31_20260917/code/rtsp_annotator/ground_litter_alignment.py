"""Fixed-view verification; never updates the reference from unknown live frames."""
from __future__ import annotations

import cv2
import numpy as np

from .ground_litter_geometry import polygon_points
from .ground_litter_quality import DetailLossGuard


class ViewAlignment:
    """Match distributed scene features while excluding camera text overlays.

    SIFT improves matching across illumination and foreground changes. The
    displacement limit is the original 3 pixels at width 640 (12 at 2560).
    ORB is retained solely for reproducible comparisons with earlier runs.
    """

    def __init__(self, reference, camera: dict):
        self.method = camera.get("view_alignment", "sift")
        if self.method not in {"orb", "sift"}:
            raise ValueError("view_alignment must be orb or sift")
        self.shape = reference.shape
        self.quality=DetailLossGuard(reference)
        self.width = 640 if self.method == "orb" else 960
        self.height = round(reference.shape[0] * self.width / reference.shape[1])
        self.size = (self.width, self.height)
        self.detector = (cv2.ORB_create(nfeatures=600) if self.method == "orb"
                         else cv2.SIFT_create(nfeatures=2500, contrastThreshold=.02))
        self.mask = np.full((self.height, self.width), 255, np.uint8)
        if self.method == "sift":
            for polygon in camera.get("overlay_exclude_zones", []):
                cv2.fillPoly(self.mask, [polygon_points(polygon, *self.size)], 0)
        self.reference_keys, self.reference_desc = self.detector.detectAndCompute(
            self.gray(reference), self.mask)
        self.diagnostics: dict = {}

    def gray(self, frame):
        return cv2.cvtColor(cv2.resize(frame, self.size), cv2.COLOR_BGR2GRAY)

    def check(self, frame) -> None:
        self.diagnostics = {"method": self.method, "matches": 0, "inliers": 0}
        if frame.shape != self.shape:
            raise ValueError("resolution_changed")
        self.diagnostics['quality']=self.quality.inspect(frame)
        if not self.diagnostics['quality']['usable']:
            raise ValueError('image_quality_unknown')
        gray = self.gray(frame)
        if gray.mean() < 8 or gray.mean() > 247 or cv2.Laplacian(gray, cv2.CV_32F).var() < 3:
            raise ValueError("image_quality_unknown")
        keys, desc = self.detector.detectAndCompute(gray, self.mask)
        if desc is None or self.reference_desc is None or len(desc) < 2:
            raise ValueError("view_alignment_unknown")
        norm = cv2.NORM_HAMMING if self.method == "orb" else cv2.NORM_L2
        pairs = cv2.BFMatcher(norm).knnMatch(self.reference_desc, desc, k=2)
        matches = [a for pair in pairs if len(pair) == 2 for a, b in [pair]
                   if a.distance < .7 * b.distance]
        # Different reference descriptors must not all vote for one feature.
        if self.method == "sift":
            unique = {}
            for match in sorted(matches, key=lambda m: m.distance):
                unique.setdefault(match.trainIdx, match)
            matches = list(unique.values())
        self.diagnostics["matches"] = len(matches)
        if len(matches) < 12:
            raise ValueError("view_alignment_unknown")
        source = np.float32([self.reference_keys[m.queryIdx].pt for m in matches])
        target = np.float32([keys[m.trainIdx].pt for m in matches])
        matrix, mask = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=2)
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            raise ValueError("view_alignment_unknown")
        selected = source[mask.ravel().astype(bool)]
        count = len(selected)
        self.diagnostics["inliers"] = count
        if count < (10 if self.method == "orb" else 12):
            raise ValueError("view_alignment_unknown")
        fraction = count / len(matches)
        span = np.ptp(selected, axis=0) / np.array(self.size)
        hull_fraction = cv2.contourArea(cv2.convexHull(selected)) / (self.width * self.height)
        self.diagnostics.update(inlier_fraction=round(fraction, 4),
                                hull_fraction=round(hull_fraction, 4))
        if self.method == "sift" and (fraction < .5 or min(span) < .15 or hull_fraction < .015):
            raise ValueError("view_alignment_unknown")
        corners = np.float32([[[0, 0], [self.width, 0],
                               [self.width, self.height], [0, self.height]]])
        displacement = float(np.linalg.norm(cv2.transform(corners, matrix) - corners, axis=2).max())
        self.diagnostics["displacement_native_px"] = round(displacement * self.shape[1] / self.width, 3)
        if displacement > 3 * self.width / 640:
            raise ValueError("view_changed_recalibration_required")
