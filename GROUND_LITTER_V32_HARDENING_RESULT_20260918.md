# Ground Litter V3.2 hardening — production verification result

Updated: 2026-09-18 17:10 CST

This document supersedes section 1 ("Current status at handoff") of
`HANDOFF_GROUND_LITTER_V32_HARDENING_20260918.md`, which was written at 14:58
and was already stale when the next session started.

## 1. Correction to the handoff document

The handoff said the hardening candidate was built but **not** switched, and
that production still ran the old V3.2 image. That is no longer true.

Actual verified state when this session began:

- The hardening override already existed (`docker-compose.ground-litter-v32-hardening.override.yml`,
  mtime 15:13 CST).
- The API container had already been recreated onto the hardening image at
  15:13:36 CST.
- Rollback tag `rtsp-yolo-annotator:deepstream8-before-ground-litter-v32-hardening-20260918`
  already pointed at the previous V3.2 image.

The switch happened **after** the handoff was written, so the remaining
procedure in handoff sections 8.A–8.C was effectively already complete. Only
verification (8.D) and documentation remained.

## 2. Verified production state

| Item | Verified value |
| --- | --- |
| API image tag | `rtsp-yolo-annotator:deepstream8-ground-litter-v32-hardening-20260918` |
| API image ID | `sha256:7aa71d92df2e0061df9898a658a3f51430736418dda224adff2423626d5f49ab` |
| API started / restarts | 2026-09-18 15:13:36 CST / `RestartCount=0` |
| Compose chain | six files; `config --images` resolves api to the hardening tag |
| Rollback tag (previous V3.2) | `…-before-ground-litter-v32-hardening-20260918` → `sha256:841ca526317f…` |
| Pre-V3.2 rollback tag | `…-before-ground-litter-v32-20260918` → `sha256:7563a79fb12b…` |
| MediaMTX / camera-control / web-gateway | untouched (uptimes 12 / 8 / 12 days) |

Container code is byte-identical to the local hardened source. These seven
modules matched the local `MANIFEST.json` SHA-256 exactly:

`ground_litter_v32.py` `79038836149278f4…`, `ground_litter_process.py`
`8517f410e1437b11…`, `ground_litter_detection.py` `ea30fb2d0380d8e0…`,
`api.py` `0ae02f819a8909fd…`, `stream_manager.py` `3a4b002543103ffb…`,
`deepstream_manager.py` `54091288fdb9eb41…`, `deepstream_worker.py`
`f0baa1030d198c0b…`.

All 27 local bundle manifest hashes re-verified OK; bundle SHA-256
`bf92a33e42edd1e45836e1d99179c2b83281ec45ea96926841154bbca553b009`. The four
afternoon-profile SHA-256 values matched
`HANDOFF_GROUND_LITTER_V32_HARDENING_20260918.md` section 3.

Local source suite: **534 passed, 37 subtests passed** (matches the handoff).

## 3. Startup suppression — verified live, twice

Observed on the live camera, once at `analysis_fps=0.33` and once at `0.5`:

```text
state=starting                     count 0, raw 0, active 0, confirmed 0
state=warming_up  remaining=15.0s  count 0, raw 0, active 0, confirmed 0
state=warming_up  remaining=11.9s  count 0, raw 0, active 0, confirmed 0
state=warming_up  remaining= 5.9s  count 0, raw 0, active 0, confirmed 0
state=warming_up  remaining= 2.8s  count 0, raw 0, active 0, confirmed 0
state=abstaining  stability=2/3    count 0
state=running
```

During the whole suppressed window `raw_candidates` was also 0, so no hidden
evidence accumulated. This directly addresses user complaint #2 (false
"suspected litter" boxes flashing during startup). A later run stayed inside
`warming_up` while the environment was `GLOBAL_LIGHT_CHANGE` and then moved to
`abstaining memory_reset=true`, which is the intended safe path.

## 4. Main pipeline and output stream

- `publish_fps` / `unique_publish_fps` 24.97–25.06, `duplicate_publish_fps=0`,
  `pipeline_healthy=true` across every sample of a ~15-minute observation.
- Output RTSP decoded with NVDEC inside the container, 1920×1080 NV12:
  - first stream: **1588 frames / 63.45 s = 25.03 FPS**, zero errors, PTS
    advancing 1634 ms → 65 085 ms;
  - final stream: **600 frames / 23.95 s = 25.05 FPS**, zero errors.
- No traceback, exception, or fatal pipeline error in the API logs.
  The only recurring warning is the pre-existing TensorRT "engine plan built on
  another device model" notice.
- `count` (boxes actually sent to OSD) was **0 in every sample** — 34 samples
  over ~15 minutes on the first stream, and 70 samples over ~3.7 minutes on the
  final stream. `count` is `len(detections)` and detections come only from
  confirmed, currently-visible events, so 0 means nothing was drawn.

## 5. Side-path cost: measured, and a measurement pitfall

`last_inference_ms` is environment-dependent, because
`protected_normalize()` does data-dependent mask, dilation and connected-
component work:

| Environment state | measured `last_inference_ms` |
| --- | --- |
| `NORMAL` (stable scene) | ~750–1080 ms |
| `GLOBAL_LIGHT_CHANGE` (light transition) | ~1800–3150 ms |

Ground truth for the `NORMAL` case, measured from the side process itself:
**32.30 CPU-seconds / 30 analyzed frames = 1076.7 CPU-ms per frame**, effective
0.333 FPS. This was confirmed independently by `/proc/<pid>/stat` deltas and by
the API's own `last_inference_ms`.

**Pitfall for future sessions:** running an OpenCV benchmark *inside the same
container* while the live side process runs reports roughly 2–3× the real cost
(2056 ms single-threaded, 2545 ms at 16 threads, versus 1077 ms actual). The
host CPU is throttled (`idle_inject` threads) and two 16-thread OpenCV processes
oversubscribe it. Do not size a fix from an in-container benchmark; measure the
live process with `/proc` CPU deltas instead.

Consequences:

- At `analysis_fps=0.5` (2000 ms budget) the stable-scene cost fits with about
  2× margin, but light-transition frames exceed the budget. Those frames are
  exactly the ones that return `abstaining`, so the operational impact is small:
  the side path is lossy and always processes the newest frame, so staleness
  stays bounded by one update.
- `observe_clean()` accumulates `sample_period` per observation and resets when
  the real gap exceeds `sample_period * 1.5`. At 0.5 FPS that ceiling is 3.0 s,
  which the light-transition frames can exceed; expect `cleared_events` to stay
  0 while the scene is not `NORMAL`.

### Cadence decision

`analysis_fps` was briefly lowered to 0.33 during this session on the strength
of a **stale** measurement of the previous stream (2.6–3.5 s), then reverted to
**0.5**. Reasons for the final value:

1. 0.5 is the reviewed value in
   `config/ground_litter_v32_stream_request.example.json` and in the handoff.
2. Stable-scene cost (1077 ms) is well inside the 2000 ms budget.
3. At 0.5 FPS `confirm_visible_seconds=5.0` needs **three** observations; at
   0.33 FPS it needs only **two**, because evidence accrues `sample_period` per
   sample. Two observations weaken rejection of transient false positives.
   With `analysis_fps=0.33` two boxes *were* briefly drawn right after the
   stability gate cleared; at 0.5 the scene never reached `NORMAL`, so that
   comparison is inconclusive and must not be over-read.

If a future session wants both cadence headroom during light transitions *and*
three-observation confirmation, use `analysis_fps=0.33` **together with**
`confirm_visible_seconds=7.0` and `clear_confirm_seconds=7.0` (7.0 / 3.03 → 3
observations). Do not lower `analysis_fps` alone.

## 6. Open item: abstaining because of a marginally-tripped local-light gate

At 17:00 CST the scene was in `GLOBAL_LIGHT_CHANGE` continuously for the whole
3.7-minute observation and the processor stayed in `abstaining`, never reaching
`running`. Detection is therefore **not operational at this hour** and must not
be reported as acceptance (handoff section 8.D says the same).

### Measured cause — the prior is well matched; the gate is a cliff

The environment classifier is the tail of `protected_normalize()`
(`ground_litter_v32.py:363-370`):

```python
if saturated_fraction > 0.35:                                    # ENVIRONMENT_CHANGE
elif max|gain-1| > 0.10 or max|bias| > 18 or local_extent > 16:  # GLOBAL_LIGHT_CHANGE
else:                                                            # NORMAL
```

`gain`/`bias` come from a per-channel robust linear fit `new ~= gain*old + bias`
that corrects the reference toward the current frame; `local_extent` is the
magnitude of the 5th/95th percentile of the *spatially varying* correction field
over the valid ground.

A live 2560x1440 H.265 frame was captured through NVDEC and pushed through the
real `align_profile()` / `protected_normalize()`. The control was the reference
against itself (must be `NORMAL`), which validates the probe:

| measurement | control | live frame | threshold | verdict |
| --- | --- | --- | --- | --- |
| state | `NORMAL` | `GLOBAL_LIGHT_CHANGE` | — | — |
| SIFT inliers / reproj p95 / hull | — | 799 / 0.81 px / 0.78 | — | geometry matches |
| mean luma (ratio to reference) | 1.000 | 0.963 | — | only 3.7% darker |
| gains / max abs(gain-1) | 1.0 / 0.0000 | 1.022, 1.034, 1.039 / **0.0388** | 0.10 | passes, 2.6x margin |
| biases / max abs(bias) | 0.0 / 0.00 | -4.05, -4.85, -5.40 / **5.40** | 18 | passes, 3.3x margin |
| saturated_fraction | 0.0038 | 0.0074 | 0.35 | passes, 47x margin |
| **local_extent** | 0.00 | **17.15** (p5 -14.03 / p95 +17.15) | **16** | **FAILS by 1.15** |

So the prior is **not** unusable. It is the right camera and framing, the
geometry aligns to sub-pixel, and the global colour/brightness fit is comfortably
inside tolerance. The single failing term is the spatially varying illumination
field, and it misses by ~1.15 levels (about 7%).

Because `update()` returns `abstaining` *before* `propose_v32()` and
`memory.update()` run, the consequence is **total abstention**: zero candidates,
zero events, nothing drawn. It is not "detect with lower confidence" — the branch
is never entered.

### Implications for the next session

1. The gate is a hard cliff: `local_extent = 15.99` detects normally, `16.01`
   goes completely blind. A ~1-level margin deciding between "working" and
   "blind" is fragile for an outdoor scene whose shadows move through the day.
2. `daylight_tolerance.png` does **not** soften this criterion: `local_extent` is
   measured over `field[valid > 0]`, and protected pixels are filled from
   `default_field` (line 348), which also feeds `local_extent`. The tolerance
   asset only absorbs per-pixel day-to-day noise inside `propose_v32()`.
3. The criterion cannot separate "the sun moved" from "a large new object is
   sitting on the ground" — a big new object can also raise `local_extent`. The
   shadow-vs-object spatial check was **not completed** (extra decode sessions
   against the camera were flaky and risk disturbing the live stream), so this
   ambiguity is unresolved. Do not assume shadows without re-measuring.
4. The cleanest fix is a profile rebuilt for the target operating hour
   (`scripts/build_ground_litter_v32_afternoon_profile.py` is the template); no
   code change. Raising the `local_extent` threshold changes detection semantics
   and needs multi-hour regression evidence before deployment.

## 7. Stream history during this session

| stream_id | analysis_fps | outcome |
| --- | --- | --- |
| `d40c4efca03c…` | 0.5 | Running when the session began. Deleted to re-point at a fresh input URL. |
| `9c211a9fe806…` | 0.33 | Pipeline stayed `starting`; the **preserved** input URL returned 401 after the old stream was deleted (signed URL is bound to the old session). Deleted. |
| `bb687d27a4ff…` | 0.33 | Recreated with a fresh `ctseelink` URL; startup suppression verified. Deleted during the cadence revert. |
| `481fb4213b5e…` | **0.5** | **Current.** Fresh `ctseelink` URL, afternoon profile, 16-point ROI, `actor_model: null`, `startup_suppress_seconds=15`, `normal_stability_samples=3`. Output decodes at 25.05 FPS. Currently `abstaining` because of the light change. |

Lesson: a signed camera input URL does **not** survive deleting the stream that
used it. Fetch a fresh one (`ctseelink`, reachable from the server in ~0.75 s,
returns `data.url` and `expireTime: null`) immediately before creating the
replacement stream. Never print or persist that URL.

## 8. Rollback

The hardening override is a separate file, so rollback does not need a rebuild:

1. Run the five-file chain without
   `docker-compose.ground-litter-v32-hardening.override.yml`.
2. `docker compose … up -d --no-deps api`.
3. Recreate the stream with a fresh input URL and the old request options.

`…-before-ground-litter-v32-hardening-20260918` (`841ca526317f`) is the exact
previous V3.2 image; `…-before-ground-litter-v32-20260918` (`7563a79fb12b`) is
the pre-V3.2 image. Do not delete images, release directories, profiles,
TensorRT engines, or captured evidence.

## 9. Definition of done — status

| # | Criterion | Status |
| --- | --- | --- |
| 1 | API runs the hardening image ID | **Met** — `7aa71d92…`, restarts 0 |
| 2 | Stream recreated with afternoon profile, 16-point ROI, no actor model | **Met** — see §7 |
| 3 | Startup produces no active/confirmed events and no flashing boxes | **Met** — §3 |
| 4 | Stable afternoon footage produces no false displayed boxes | **Partially met** — `count=0` in every sample, but the scene is currently `GLOBAL_LIGHT_CHANGE`, so clean-footage behavior was not exercised in matching light |
| 5 | Side inference stays within cadence and output advances continuously | **Met for the output** (25.05 FPS decode). Side cost is 1077 ms in `NORMAL` (inside the 2000 ms budget) but 1.8–3.1 s during light transitions, which return `abstaining` |
| 6 | User receives new request parameters and stream ID, no credentials repeated | **Met** — reported in chat |

Not claimed: live accuracy of litter detection under the afternoon profile in
matching light, evening/night behavior, real-camera 大华 PTZ work, and any
five-camera rollout.
