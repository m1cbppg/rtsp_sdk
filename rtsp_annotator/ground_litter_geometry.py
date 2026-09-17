"""Native-pixel calibration geometry shared by pilot and review tools."""
import cv2
import numpy as np

def polygon_points(points, width, height):
    values = np.asarray(points, np.float32)
    if (values.ndim != 2 or values.shape[1] != 2 or len(values) < 3
            or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()):
        raise ValueError("Invalid normalized polygon")
    result = np.rint(values * [width, height]).astype(np.int32)
    if abs(cv2.contourArea(result)) < 1:
        raise ValueError("Degenerate polygon")
    return result


def prepare(camera, frame, tile_size, overlap):
    h, w = frame.shape[:2]
    if camera["reference_size"] != [w, h]:
        raise ValueError("Image/reference resolution mismatch")
    masks = {}
    for zone in camera["zones"]:
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [polygon_points(zone["polygon"], w, h)], 1)
        for exclusion in zone["exclude_zones"]:
            cv2.fillPoly(mask, [polygon_points(exclusion, w, h)], 0)
        for exclusion in camera.get("overlay_exclude_zones", []):
            cv2.fillPoly(mask, [polygon_points(exclusion, w, h)], 0)
        if not mask.any():
            raise ValueError("Region has no visible ground")
        if zone["region_id"] in masks:
            raise ValueError("Duplicate region ID")
        masks[zone["region_id"]] = mask
    combined = np.maximum.reduce(list(masks.values()))
    def starts(low, high, size):
        low = max(0, min(low, size-tile_size))
        end = max(low, high-tile_size)
        return sorted(set(list(range(low,end+1,round(tile_size*(1-overlap))))+[end]))
    def grid(mask):
        ys,xs=np.nonzero(mask)
        return {(x,y,min(w,x+tile_size),min(h,y+tile_size))
                for y in starts(int(ys.min()),int(ys.max()+1),h)
                for x in starts(int(xs.min()),int(xs.max()+1),w)
                if mask[y:y+tile_size,x:x+tile_size].any()}
    # Choose the smaller complete covering: joint bounds or individual ROI
    # bounds. Never rescale the original image before extracting these tiles.
    joint=grid(combined)
    separate=set().union(*(grid(mask) for mask in masks.values()))
    tiles=sorted(min((joint,separate),key=len))
    coverage=np.zeros_like(combined)
    for x,y,r,b in tiles:
        coverage[y:b,x:r]=1
    if np.any(combined & ~coverage):
        raise ValueError('Tile plan omitted ground pixels')
    return masks, tiles, int(combined.sum())


def box_overlap_fraction(box, actor):
    x,y,r,b = box
    a,c,d,e = actor
    area = max(0, r-x) * max(0, b-y)
    if area <= 0:
        return 0.0
    return max(0,min(r,d)-max(x,a))*max(0,min(b,e)-max(y,c))/area
