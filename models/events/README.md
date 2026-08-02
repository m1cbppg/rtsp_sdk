# Garbage analysis model assets

- Source weight: Ultralytics `yolov8s-worldv2.pt`, downloaded from the
  official `ultralytics/assets` v8.4.0 release.
- Source SHA256:
  `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792`
- Export: `scripts/export_yolo_world_garbage.py` with the fixed vocabulary in
  `yolo_world_garbage.labels.txt`, dynamic batch 1–2, input 640×640, output
  8400×6.
- ONNX SHA256:
  `8c470f286ae2937e4d2e8f48d2265a9a843a43781f17b57dd0ee2a9554cb40b4`
- Output coordinates are `x1,y1,x2,y2,score,class_id`, matching the pinned
  DeepStream-Yolo parser ABI.

The source `.pt` is used only when regenerating the ONNX and is excluded from
the Docker image. Review the upstream Ultralytics/YOLO-World licenses before
commercial distribution.

## Street garbage pile model

- Source checkpoint: `Vansh180/PotholeNet-V1` (`Vision Classification.pt`),
  a YOLO11m detector trained on street-level civic imagery with a `garbage`
  class. The upstream model card declares MIT licensing.
- Source revision: `c30e125895cd546f5ea2e94d81daa56d19b21d2e`.
- Source SHA256:
  `f380cd373f61f2bc71f7fcc1b0ec072194dc2cd933fd05bc1ae5ad136a333b78`.
- Export: `scripts/export_street_garbage_deepstream.py`, dynamic batch 1–2,
  input 640×640, output 8400×6.
- ONNX SHA256:
  `34553ebe1fb1afc4d4055f5e6103f3935bfc2872eba2c36b2ddb947e6538c75d`.

The pile profile accepts only the model's `garbage` class, clusters nearby
component detections, and draws one Chinese `垃圾堆` rectangle per cluster.
The original YOLO-World item profile remains available and unchanged.
