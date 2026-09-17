# Ground litter stability fix deployment (2026-09-16)

Local validation: 480 unittest cases, compileall and git diff --check passed. DeepStream ServiceMaker contract could not run locally because pyservicemaker is not installed.

Changes:
- `inference_imgsz` separates native crop size from model input size; `320 + 640` is supported.
- bounded `local_actor_max_crops` performs optional local vehicle/person rechecks.
- `box_smoothing_alpha` smooths displayed boxes without changing candidate matching geometry.
- confirmed boxes survive short inference misses until `hold_seconds`; occlusion resets confirmation; duplicate/old timestamps do not count.
- DeepStream group signature includes all new options to prevent incompatible sharing.

Deployment:
- Uploaded package SHA-256: `1e717d82bfd17b5b6b6f03e4d18f23cbe7ebba3fe5c7a8e750a959492e905109`.
- New image: `rtsp-yolo-annotator:deepstream8-ground-litter-fix-20260916`.
- Image ID: `sha256:66e78508a861289e60efa2574db8c23cd83ee54a3f076736e4b34fa888ff39e8`.
- Rollback tag: `rtsp-yolo-annotator:deepstream8-before-fix-20260916`.
- Only `rtsp-yolo-api` was recreated. MediaMTX, camera-control and web gateway were not restarted.
- Existing API streams were cleared by the API restart and must be recreated by callers.
- `/health` returned `{"status":"ok"}` and API restart count is 0.

Recommended first test parameters for this camera:
- model: `turhancan_yolov8m_seg_trash.pt`
- tile_size_px: `320`
- inference_imgsz: `640`
- tile_overlap: `0.25`
- confidence: `0.15`
- analysis_fps: `2.0`
- minimum_hits: `3`, hit_window: `5`, hold_seconds: `5`
- local_actor_max_crops: `2`, box_smoothing_alpha: `0.5`

These parameters require real-stream A/B validation; they are not an accuracy guarantee.

## Index correction

The first image build used an expanded ground-litter signature but one old index when reading `actor_model`; with `local_actor_max_crops=2` this resolved the actor model as filename `2`. The API correctly rejected the request as `零散垃圾模型不存在: 2`, so no bad stream was started. The corrected signature reads actor model at index 7 and was deployed as `...ground-litter-fix-20260916-r1` (image ID `sha256:7d1f67767895edf74444f2baef6e35be7940d660e97c7fbe68c2fa4b5a8fcdc3`).

## Zone-threshold and NMS-order revision

The API accepts optional `confidence` and `night_confidence` on each zone. Inference uses the
lowest configured zone threshold for recall, then applies the candidate's zone threshold before
cross-tile NMS. This prevents an out-of-zone high-confidence box from suppressing an in-zone
candidate. The latest server image is `rtsp-yolo-annotator:deepstream8-zone-threshold-20260916-r2`;
the previous image is tagged `deepstream8-before-nms-order-20260916`. Local and container source
hashes matched after restart. Accuracy still requires a fresh source-stream replay.
