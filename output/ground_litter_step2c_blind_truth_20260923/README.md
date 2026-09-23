# Step 2C-1 — Development Unlock + Blind Truth

**BLIND TRUTH MODE — DETECTOR OUTPUT DISABLED.**
This step builds the human truth for the Step 0A **Development** split before any model
output exists. It never loads `last.pt` / `best.pt`, never imports torch/ultralytics and
never runs inference; `scripts/serve_ground_litter_blind_truth.py selftest` enforces that
(loaded modules, source tokens, checkpoint arguments) and the run aborts if it fails.

## Where things live

| Thing | Location |
| --- | --- |
| Development PS (65, 5.53 GiB, 5 cameras) | `/home/sf01/ground-litter-feasibility/20260923-r1/archive/ground-litter-detector-feasibility-20260923-r1/development/<camera>/raw/` |
| Sealed PS (64) | the sibling `sealed_test/` split — **never touched**; `assert_development_asset` hard-fails on it |
| Official Blind Truth artifact | `/home/sf01/step2c1-blind-truth/artifact` (review not started: 0/65) |
| Joint-test pilot artifact | `/home/sf01/step2c1-blind-truth/pilot-artifact` (smoke-test points only) |
| Code bundle | `/home/sf01/step2c1-blind-truth/code` (SHA-256 in `code_bundle_sha256.json`) |

## Running the review UI

On the machine that holds the Development PS (it needs the raw video; no GPU, no model):

```bash
cd /home/sf01/step2c1-blind-truth/code
/home/sf01/ground_litter_train/.venv/bin/python -B scripts/serve_ground_litter_blind_truth.py \
  --development-root /home/sf01/ground-litter-feasibility/20260923-r1/archive/ground-litter-detector-feasibility-20260923-r1/development \
  --roi-dir $PWD/output/ground_litter_final_roi_20260922/config \
  --output /home/sf01/step2c1-blind-truth/artifact \
  --bind 127.0.0.1 --port 8801 serve
```

Then from a workstation: `ssh -N -L 8801:127.0.0.1:8801 sf01@14.21.88.97 -p 21002` and open
`http://127.0.0.1:8801/`.

Keys: `R` = REQUIRED_LITTER, `I` = IGNORE_SMALL, `U` = UNCERTAIN, `D` = NON_LITTER,
`X` = delete the selected truth, `N` = new episode, `E` = assign to the selected episode,
`C` = confirm episode, `M` = mark this PS reviewed, `←/→` = ±1 frame
(`Shift` = ±1 s), `space` = play/pause. Mouse: click a target centre (no bbox drawing),
`经典CV定位候选` proposes up to three classic-CV boxes **after** you confirmed the object;
picking one records `PROPOSAL_SELECTED`, leaving the point truth records
`LOCALIZATION_UNRESOLVED` for manual adjudication.

Other subcommands: `manifest`, `sample`, `status`, `freeze`, `erratum`.

## Sampling (frozen)

* visible frames: fixed **5 s grid** inside each confirmed Required episode, **max 5 per
  episode**, first frame = first confirmable timestamp. Deterministic; no content or model
  input.
* global ROI frames: fixed **30 s grid** over every Development PS, independent of content.

## Freeze

`freeze` refuses until **all 65 PS are reviewed**, then SHA-256s the truth objects, the
episode manifest, the visible-frame manifest, the global-frame manifest, the Development
manifest, the localization state and the summary/manifest, writes `FREEZE.json` and strips
the write bits (`0444`). After that nothing may be edited in place: corrections go through
`erratum`, which appends to `TRUTH_ERRATA.jsonl` with the original value, the reason, the
time and whether it happened after inference.

## Evidence files in this directory

| File | What it is |
| --- | --- |
| `development_manifest.json` | the official 65 PS Development inventory (camera, scene_version, timestamps, byte size, SHA-256, frozen ROI) |
| `official_state.json` | official artifact state: 0/65 reviewed, not frozen, no agent truth |
| `pilot_joint_test.json` | §31 small-scale joint test on real Development video through the real HTTP UI |
| `pilot_SUMMARY.json` / `pilot_MANIFEST.json` / `pilot_FREEZE.json` / `pilot_localization_state.json` | the pilot artifact's frozen records (smoke-test points only) |
| `blindness_selftest.json` | detector-free proof: local + server self-test, plus the pilot's decode/proposal checks |
| `code_bundle_sha256.json` | uploaded code bundle archive and per-file SHA-256 |

The pilot truth itself (`truth_objects.jsonl`, `episodes.jsonl`, the frame manifests) stays
on the server: it is a geometric smoke test (ROI centroid), not operator truth, and this
repository does not carry truth JSONL.
