#!/usr/bin/env python3
"""Local candidate screening on cached frames; counts are NOT accuracy metrics."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/litter_eval/ultralytics")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROI = [[0.02, 0.08], [0.37, 0.08], [0.46, 0.96], [0.02, 0.96]]


class NativeYOLOv9:
    """Adapt the upstream dual detection head without changing the weights."""

    def __init__(self, weights, source):
        sys.path.insert(0, str(source.resolve()))
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes
        from ultralytics.engine.results import Results

        self.letterbox = letterbox
        self.nms = non_max_suppression
        self.scale_boxes = scale_boxes
        self.results = Results
        # Legacy upstream checkpoints serialize their architecture. Local,
        # downloaded evaluation checkpoint only; never accept request inputs here.
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        self.model = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
        self.names = self.model.names
        self.metadata = {"epoch": checkpoint.get("epoch"),
                         "best_fitness": np.asarray(checkpoint.get("best_fitness", 0)).item(),
                         "source": str(source), "head": "second dual head, as detect_dual.py"}

    @torch.inference_mode()
    def predict(self, frame, imgsz, conf, **kwargs):
        padded = self.letterbox(frame, new_shape=imgsz, auto=True, stride=32)[0]
        tensor = torch.from_numpy(np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))).float()[None] / 255
        prediction = self.model(tensor)[0][1]
        boxes = self.nms(prediction, conf_thres=conf, iou_thres=.7)[0]
        boxes[:, :4] = self.scale_boxes(tensor.shape[2:], boxes[:, :4], frame.shape)
        return [self.results(orig_img=frame, path="", names=self.names, boxes=boxes)]


def tiles(mode, w, h, edge=640, overlap=0.20):
    if mode == "full":
        return [(0, 0, w, h)]
    # Four overlapping crops cover the whole provisional ROI, including curb.
    x1, x2 = int(w * .02), int(w * .46)
    top, bottom = int(h * .08), int(h * .96)
    edge = max(160, min(int(edge), max(w, h)))
    stride = max(1, int(round(edge * (1.0 - float(overlap)))))
    xs = list(range(x1, max(x1 + 1, x2 - edge + 1), stride))
    ys = list(range(top, max(top + 1, bottom - edge + 1), stride))
    # Always include the far edge so the ROI boundary is covered.
    xs.append(max(x1, x2 - edge))
    ys.append(max(top, bottom - edge))
    return [(x, y, min(w, x + edge), min(h, y + edge))
            for y in sorted(set(ys)) for x in sorted(set(xs))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=["full", "tiles"], default="tiles")
    ap.add_argument("--confidence", type=float, default=.15)
    ap.add_argument("--tile-size", type=int, default=640,
                    help="native-pixel tile edge for --mode tiles")
    ap.add_argument("--tile-overlap", type=float, default=.20,
                    help="tile overlap in [0, .8)")
    ap.add_argument("--limit-per-video", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--yolov9-source", type=Path, help="Upstream WongKinYiu/yolov9 source directory")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    model = NativeYOLOv9(a.model, a.yolov9_source) if a.yolov9_source else YOLO(str(a.model))
    paths = []
    for name in ["day", "night"]:
        # The first FFmpeg decoded day frame has missing HEVC reference data.
        # Exclude frame 001 from BOTH videos before applying any model.
        part = sorted(a.frames.glob(f"{name}_*.jpg"))[1:]
        if a.limit_per_video:
            part = part[::max(1, len(part) // a.limit_per_video)][:a.limit_per_video]
        paths.extend(part)
    if not paths:
        raise ValueError("No day/night cached frames found after excluding frame 001")
    report = {"model": str(a.model), "sha256": hashlib.sha256(a.model.read_bytes()).hexdigest(),
              "classes": model.names, "roi": ROI, "exclusion_zones": [], "mode": a.mode,
              "imgsz": 640 if a.mode == "tiles" else 1280, "confidence": a.confidence,
              "tile_size": a.tile_size, "tile_overlap": a.tile_overlap,
              "device": "cpu", "nms_iou": .5,
              "sampling": "FFmpeg decoded frame indices 250,500,...; 001 excluded; nominal 10s interval",
              "frames": []}
    if a.yolov9_source:
        report["native_yolov9"] = model.metadata
    for index, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Cannot read {path}")
        h, w = frame.shape[:2]
        detections = []
        start = time.monotonic()
        for x0, y0, xend, yend in tiles(a.mode, w, h, a.tile_size, a.tile_overlap):
            result = model.predict(frame[y0:yend, x0:xend], imgsz=report["imgsz"],
                                   conf=a.confidence, verbose=False, device="cpu")[0]
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                x1 += x0; x2 += x0; y1 += y0; y2 += y0
                center = ((x1+x2)/(2*w), (y1+y2)/(2*h))
                if cv2.pointPolygonTest(np.array(ROI, np.float32), center, False) < 0:
                    continue
                detections.append({"label": result.names[int(box.cls[0])],
                                   "confidence": float(box.conf[0]),
                                   "box": [round(x1), round(y1), round(x2), round(y2)]})
        if detections:
            boxes = [[d["box"][0], d["box"][1], d["box"][2]-d["box"][0],
                      d["box"][3]-d["box"][1]] for d in detections]
            keep = cv2.dnn.NMSBoxes(boxes, [d["confidence"] for d in detections], a.confidence, .5)
            detections = [detections[int(k)] for k in np.asarray(keep).reshape(-1)]
        report["frames"].append({"image": path.name, "nominal_time_s": (int(path.stem.split('_')[-1])-1)*10,
                                 "inference_seconds": round(time.monotonic()-start, 3),
                                 "detections": detections})
        ann = frame.copy()
        cv2.polylines(ann, [np.array([[round(x*w),round(y*h)] for x,y in ROI])], True, (0,255,255), 2)
        for d in detections:
            x1,y1,x2,y2 = d["box"]
            cv2.rectangle(ann, (x1,y1), (x2,y2), (0,200,0), 2)
            cv2.putText(ann, f'{d["label"]} {d["confidence"]:.2f}', (x1,max(20,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (0,200,0), 2)
        cv2.imwrite(str(a.output/path.name), ann)
        (a.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(f"{index+1}/{len(paths)} {path.name}: {len(detections)} boxes", flush=True)
    report["summary"] = {}
    for part in ["day", "night"]:
        rows = [f for f in report["frames"] if f["image"].startswith(part)]
        counts = {}
        for f in rows:
            for d in f["detections"]:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
        report["summary"][part] = {"sampled_frames": len(rows),
            "frames_with_detections": sum(bool(f["detections"]) for f in rows),
            "counts_by_label": counts, "detection_count": sum(counts.values())}
    (a.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
