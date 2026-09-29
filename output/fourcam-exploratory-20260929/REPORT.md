# Four-camera exploratory detector probe — 2026-09-29

## Decision

The frozen Step 2B `last.pt` model is not a useful standalone candidate generator on these selected four-camera frames. The Turhancan semantic baseline proposes boxes near most marked points, but also proposes many other boxes, including visually implausible candidates. It is a possible candidate generator for further development, not a deployment-ready detector. This is an exploratory observation, **not** a formal recall, precision, false-positive rate, or Step 0B Go/No-Go result.

## Fixed scope and method

- Development only: cameras `01021`, `01022`, `01027`, `01030`; `01028` is deferred for severe occlusion. Sealed was not accessed.
- The frozen Development and audit manifests were verified against SHA256 `66f908f475f8e3972e371f09987b060481632b9361d5404e4dd00c5beb178496` and `2f8ea4b3a2c7552cadf1a586afa51631e16566dbb1e0dce5a06b0ee7f97c274b`.
- Nine previously recorded point marks occupy five distinct frames. One extra fixed, **unlabelled** frame at 150 seconds from each camera's first reviewed PS gives four context frames. Nine frames in six original PS files were decoded sequentially at 25 fps. All six PS SHA256 values matched their frozen manifest entries.
- Original 2560×1440 frames were tiled into 640×640 windows at 512-pixel stride with full right/bottom coverage. Only ROI-intersecting tiles were inferred; predicted box centers must lie inside the ROI. Per-class NMS at IoU 0.5 was applied across tiles. Models were run on CPU at confidence 0.01, and contact sheets show boxes at confidence ≥0.25.
- Frozen Step 2B `last.pt` SHA256: `4852392aeae9a68669a50752eb1f7466fbcc9a426524faba86fb20a3b6351a94`. Turhancan semantic baseline SHA256: `a2f8de0c7f714e2ab8b70c62490e2a41fd4a6681ca4a8dd442797c809a140278`.
- `selection.json`, `predictions.json`, nine side-by-side contact sheets, and `SHA256SUMS.json` are in this directory. The source, Step 2B, and semantic panels are ordered left to right.

## Observations

| Model / threshold | Boxes across nine frames | Point locations inside any box, of nine | Boxes on four unlabelled context frames |
|---|---:|---:|---:|
| Step 2B `last.pt`, 0.01 | 3 | 1 | 1 |
| Step 2B `last.pt`, 0.25 | 1 | 0 | 1 |
| Turhancan semantic, 0.01 | 317 | 8 | 89 |
| Turhancan semantic, 0.25 | 45 | 5 | 17 |
| Turhancan semantic, 0.50 | 10 | 3 | 2 |

“Point inside box” is a geometric diagnostic. The nine human marks are point truth, not complete bounding boxes or final Episode QA; a box covering a point is not automatically a correct detection. Likewise, context frames have not been exhaustively labelled, so their candidates are **not** counted as false positives. The one 01027 context frame has a small ROI and no point mark, so zero candidates there is not evidence of camera-wide performance.

At confidence 0.25, the semantic baseline produced 9 boxes on the 01021 anchor frame, 2/3/11 across three 01022 anchor frames, and 3 on the 01030 anchor frame. It also produced 9/3/0/5 boxes on the fixed context frames for 01021/01022/01027/01030. Visual inspection of the [later 01022 anchor](01022_1790066839000_f5911.jpg) shows overlapping Plastic and Metal candidates on the same object and a candidate on a person/held item. This supports a false-candidate concern but does not quantify its frequency.

## Limits and next engineering move

This sample was intentionally biased toward existing positive point marks and contains only one arbitrary context frame per camera. Five episodes are still active, B blind audit and C Episode QA were skipped by user choice, and 01028 remains deferred. A production decision or formal score is blocked by those missing labels and review stages.

For the next development iteration, use the semantic baseline as the candidate source and visually filter its ROI candidates on a small, fixed Development sample; collect proper box truth for ambiguous matches and distinguish ground litter from people, merchandise, bags, and fixtures. Keep the frozen Step 2B checkpoint as a comparison, not the sole detector. Do not tune on Sealed.

## Integrity and isolation

The successful run wrote only to `/home/sf01/step2c1-blind-truth/exploratory-fourcam-20260929/results-v2` and this copied local directory. `SHA256SUMS.json` verified all 11 result files after transfer. The official artifact's two frozen manifests, `review_state.json`, and `truth_objects.jsonl` retained their pre-run hashes. The 8801 review service was not called or restarted. An earlier run failed while drawing a rectangle; its partial result and log are isolated under `results` and `run.log`, and the corrected run is `results-v2` / `run-v2.log`.
