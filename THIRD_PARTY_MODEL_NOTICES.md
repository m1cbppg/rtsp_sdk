# Third-party model notices

## Local litter screening weights

The 2026-09-09 offline evaluation downloaded public checkpoints into `models/litter/`.
Their source URLs, observed revisions, publisher license declarations and local SHA-256
digests are recorded in [models/litter/manifest.json](models/litter/manifest.json), with
usage limitations in [models/litter/README.md](models/litter/README.md).

Only `turhancan_yolov8m_seg_trash.pt` is retained on disk. As of 2026-09-15 that single
checkpoint is referenced by the API `ground_litter` scenario and is copied into the
incremental image `Dockerfile.deepstream.ground-litter-update`. That is a local image
build only: the weight is not yet running in the production deployment, and enabling the
scenario in production needs separate upload/deploy authorization. The other eight
evaluated checkpoints remain historical provenance records and are not on disk.

Model-card MIT/Apache declarations do not establish the licenses of all training data,
base weights or inference code. In particular, review Ultralytics and upstream YOLOv9
terms separately. Baraa Lazkani's checkpoint is explicitly noncommercial without written
permission; Jhandry's OpenRAIL variant remains unverified.

## OpenMMLab UPerNet ConvNeXt Tiny

The optional vessel-camera calibration tool embeds the following pretrained
semantic-segmentation checkpoint in the DeepStream image:

- Model: `openmmlab/upernet-convnext-tiny`
- Pinned revision: `876ffc5`
- Source: <https://huggingface.co/openmmlab/upernet-convnext-tiny>
- License declared by the model publisher: MIT

Copyright (c) OpenMMLab and model contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
