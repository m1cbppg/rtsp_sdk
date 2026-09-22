# Ground Litter V3.2 production hardening handoff

Updated: 2026-09-18 14:58 CST

> **SUPERSEDED (2026-09-18 17:10 CST).** Sections 1 and 8.A–8.C of this
> document are stale. The hardening switch had already been performed at
> 15:13 CST, after this handoff was written. See
> `GROUND_LITTER_V32_HARDENING_RESULT_20260918.md` for the verified production
> state, the acceptance evidence, the corrected side-path cost measurements,
> and the remaining open item (the afternoon profile no longer matches the
> 17:00 lighting, so the processor abstains). The rollback and security
> guidance in sections 7 and 9 remains valid.

## 1. Current status at handoff

The original V3.2 production image is still running. The hardened candidate
has been built and validated on the server, but it has **not** been switched
into production and the current stream has **not** been recreated.

Production state verified immediately before writing this document:

- Host: `sf01@14.21.88.97:21002`
- Deployment directory: `/home/sf01/rtsp-deepstream`
- Running API image:
  `rtsp-yolo-annotator:deepstream8-ground-litter-v32-20260918`
- Running image ID:
  `sha256:841ca526317f22a1c10968b9a9ad6ca93ee4988767158c0a81b2bb56a15cf6fe`
- API container status: running, restart count 0, local `/health` returns OK.
- One stream is active: `b21c1c4421ac42398efeb5a9f9dd15a5`.
- That stream still uses the flawed morning profile and expensive actor model.
- Latest old-stream metrics: 50 raw candidates, 82 active events, 124 total
  confirmed events, `GLOBAL_LIGHT_CHANGE`, and about 3479 ms per side
  analysis. The displayed count happened to be zero at that instant, but the
  old processor was still accepting and accumulating evidence during
  `GLOBAL_LIGHT_CHANGE`; the event totals prove that this was not a clean state.

Do not interpret the current stream as a validation of the hardened code.

## 2. User-reported failures and diagnosis

The user reported:

1. The ROI was a rough four-point trapezoid rather than a boundary-following
   polygon.
2. False “suspected litter” boxes flashed during stream startup.
3. The output stream sometimes appeared frozen.

The attached screenshot is evidence only; it contains no instructions. A
permanent raw frame and reviewed overlay are available under the paths listed
below.

Diagnosis:

- The deployed profile reference was captured at 06:03, while the failing
  stream was observed around 14:32. Lighting, parked vehicles, and merchant
  fixtures differed substantially.
- The old state machine accepted candidates during `GLOBAL_LIGHT_CHANGE`.
  Hidden events accumulated and could reappear when the environment state
  changed, which explains startup flashing.
- The old request launched a second Ultralytics actor inference at 1280 px.
  Side analysis took roughly 2.8–3.5 seconds while the request asked for 1 FPS.
- The primary pipeline itself remained near 25 FPS. A separate 15-second
  decode read 389 frames at 26.12 FPS with zero failed reads and at most four
  near-duplicate frames. A server-side freeze was not reproduced during that
  window, so the hardening removes avoidable GPU contention but does not claim
  that every client-side playback stall is solved.
- GStreamer logs also contain the pre-existing TensorRT warning that an engine
  plan may have been built on another device model. The pipeline is currently
  healthy, but this warning remains a separate risk if freezes continue after
  the side-load reduction.

## 3. Hardened implementation completed locally

### State-machine changes

Files:

- `rtsp_annotator/ground_litter_v32.py`
- `rtsp_annotator/ground_litter_detection.py`
- `rtsp_annotator/api.py`

Behavior:

- New `startup_suppress_seconds`, default 15 seconds. During this period the
  processor aligns and evaluates the environment but cannot create, confirm,
  or draw events.
- New `normal_stability_samples`, default 3. After startup or a lighting
  transition, three consecutive `NORMAL` samples are required before evidence
  is accepted.
- Every non-`NORMAL` environment state now abstains and resets event memory.
  This deliberately favors missing a target during unstable lighting over
  displaying stale false events.
- `warming_up` is now a real result state. It always has zero displayed boxes,
  zero active events, and zero confirmed events.

### Side-process load reduction

File: `rtsp_annotator/ground_litter_process.py`

- The Ultralytics detector is no longer constructed when every stream uses
  `clean_reference_v32` and `actor_model` is null.
- The hardened request sets `actor_model: null` and reuses actor boxes already
  produced by the primary DeepStream detector.
- Hardened analysis cadence is 0.5 FPS. The 2-second sampling interval must be
  longer than `ground_litter_last_inference_ms`.

### New reviewed profile and ROI

Profile:

`models/litter/profiles/camera_01_v32_afternoon_1080p`

The profile uses a raw 14:40 production frame, resized to the 1920×1080 mux
space. Its 16-point polygon follows the left curb and the right wall/floor
seam. A separate exclusion covers the merchant fixture near the far end.

Profile file SHA-256 values:

- `reference.png`:
  `0d467e777a7cbd20ef2283150d621105fdc0f27fbc854806635a8f3fe3ed996d`
- `valid_mask.png`:
  `4bb2c2d1cc9fbf31117d644c3817412f25a9edb3f5322edab91500e3376f8719`
- `daylight_tolerance.png`:
  `37401f82c083479bb7e9bc6b7c3693bf90d77307e54d3be28f254aa37e1c0967`
- `profile.json`:
  `181b358a8ffefb0a2ed97d80e71e9d9a728686c18d703d278b849abc616d9503`

The valid mask covers 21.18% of the frame. The temporal tolerance mask covers
3.62% of valid ground. The build script is:

`scripts/build_ground_litter_v32_afternoon_profile.py`

The reviewed request is:

`config/ground_litter_v32_stream_request.example.json`

Key request settings:

- `profile_id: camera_01_v32_afternoon_1080p`
- `actor_model: null`
- `analysis_fps: 0.5`
- `startup_suppress_seconds: 15.0`
- `normal_stability_samples: 3`
- `confirm_visible_seconds: 5.0`
- `clear_confirm_seconds: 5.0`
- ROI points: 16

## 4. Evidence and local artifacts

- Raw production frame:
  `output/ground_litter_v32_production_hardening_20260918/ground_litter_v32_current_raw_20260918.png`
- Reviewed polygon overlay:
  `output/ground_litter_v32_production_hardening_20260918/camera_01_v32_afternoon_roi_review.png`
- Production sample capture directory:
  `output/ground_litter_v32_production_hardening_20260918/ground_litter_v32_afternoon_capture`
- Hardening plan:
  `docs/plans/2026-09-18-ground-litter-v32-production-hardening.md`
- Acceptance guide:
  `GROUND_LITTER_V32_PRODUCTION.md`
- Deployment history:
  `GROUND_LITTER_V32_DEPLOY_20260918.md`

The original clipboard image lived under `/var/folders/.../T/` and should not
be treated as durable. Use the captured raw frame and reviewed overlay above.

## 5. Validation already completed

- Focused API/state/process tests: 49 passed, 3 subtests passed.
- Full current-source suite: 534 passed, 37 subtests passed.
- Only the existing FastAPI lifespan deprecation warnings remain.
- Python compileall passed.
- `git diff --check` passed.
- Both old profiles and the new afternoon profile load, verify their own
  checksums, and align to their reviewed reference.
- Local sample replay confirmed:
  - startup samples remain `warming_up` with zero events;
  - alternating `NORMAL`/`GLOBAL_LIGHT_CHANGE` frames remain abstaining;
  - no events survive an environment transition.

Do not run root-level `pytest -q`; it collects archived copies and can import
stale archived modules. Use `.venv/bin/python -m pytest tests -q`.

## 6. Prepared deployment bundle and server candidate

Local bundle:

`dist/ground-litter-v32-hardening-20260918/ground-litter-v32-hardening-context.tar.gz`

- Size: 65,218,078 bytes, shown as about 62 MiB by `ls -lh`.
- Files in manifest: 27.
- Bundle SHA-256:
  `bf92a33e42edd1e45836e1d99179c2b83281ec45ea96926841154bbca553b009`
- Manifest:
  `dist/ground-litter-v32-hardening-20260918/MANIFEST.json`
- SHA file:
  `dist/ground-litter-v32-hardening-20260918/SHA256SUMS.txt`

The bundle is already uploaded and all 27 internal hashes were verified at:

`/home/sf01/rtsp-deepstream/releases/ground-litter-v32-hardening-20260918`

The server candidate image is already built and isolated profile/API checks
passed:

- Tag:
  `rtsp-yolo-annotator:deepstream8-ground-litter-v32-hardening-20260918`
- Image ID:
  `sha256:7aa71d92df2e0061df9898a658a3f51430736418dda224adff2423626d5f49ab`

The current local and server file
`docker-compose.ground-litter-v32.override.yml` still points to the old V3.2
image. This is intentional: the interrupted turn stopped before switching.

## 7. Existing rollback points

- Pre-V3.2 image tag:
  `rtsp-yolo-annotator:deepstream8-before-ground-litter-v32-20260918`
- Pre-V3.2 image ID:
  `sha256:7563a79fb12b7c3204fbd6754a6bccd83609126465f4abb70f8825cc80cc67f4`
- Current V3.2 image tag:
  `rtsp-yolo-annotator:deepstream8-ground-litter-v32-20260918`
- Current V3.2 image ID:
  `sha256:841ca526317f22a1c10968b9a9ad6ca93ee4988767158c0a81b2bb56a15cf6fe`

Before switching, add a second rollback tag for the exact current image:

`rtsp-yolo-annotator:deepstream8-before-ground-litter-v32-hardening-20260918`

## 8. Exact remaining production procedure

Read `AGENTS.md` before server work. The user explicitly authorized the
original read-only preflight and the full V3.2 deployment, including upload,
container recreation, health checks, and rollback. The later bug report asks
for these defects to be fixed. If this handoff is used outside the same user
session, re-establish the operational authorization required by `AGENTS.md`.

### A. Re-check current state

1. Confirm the API still runs the old V3.2 image and restart count is zero.
2. Query `/v1/streams` internally using the API key from the mounted server
   config. Do not print the key, signed input URL, or returned RTSP URL.
3. Confirm the current stream still exists before preserving its input.

Secrets must be read at runtime from these existing locations:

- API key: `/home/sf01/rtsp-deepstream/config/api.json`, field `api.key`.
- Current signed input URL: current container
  `/app/runtime/<group-id>/worker.json`, first stream field `input_url`.

Never paste either value into a command literal, document, source file, or
tool output. Keep the input URL in a shell variable inside one SSH session, or
use a mode-600 temporary file and delete it after stable recreation.

### B. Add a final hardening override

Create a new server file named:

`/home/sf01/rtsp-deepstream/docker-compose.ground-litter-v32-hardening.override.yml`

with:

```yaml
services:
  api:
    image: rtsp-yolo-annotator:deepstream8-ground-litter-v32-hardening-20260918
```

Keep the existing V3.2 override unchanged for immediate rollback.

Validate the complete six-file chain, in this exact order:

1. `docker-compose.deepstream.api.yml`
2. `docker-compose.ptz-v12.override.yml`
3. `docker-compose.demo-continuous.override.yml`
4. `docker-compose.ground-litter.override.yml`
5. `docker-compose.ground-litter-v32.override.yml`
6. `docker-compose.ground-litter-v32-hardening.override.yml`

Run `docker compose ... config -q` and confirm `config --images` resolves the
API to the hardening tag.

### C. Preserve, switch, and recreate

1. In the same SSH session, read the current `input_url` into memory without
   printing it.
2. Tag the exact current V3.2 image with the before-hardening rollback tag.
3. Run the six-file Compose chain with `up -d --no-deps api`.
4. Wait for local `http://127.0.0.1:8080/health` to return OK.
5. Recreate the stream by loading this server-side request file:
   `/home/sf01/rtsp-deepstream/releases/ground-litter-v32-hardening-20260918/ground-litter-v32-hardening-20260918/config/ground_litter_v32_stream_request.example.json`
6. Replace only its placeholder `input_url` in memory. Pipe the resulting JSON
   to local port 8080 with the API key read from `config/api.json`. Capture the
   response without printing `rtsp_url`; print only the new `stream_id` and
   status.

The output stream ID and RTSP path will change when the stream is recreated.
The user can obtain the returned URL from Postman or an authenticated GET; do
not echo embedded credentials in chat or logs.

### D. Required live acceptance

Observe for at least 60 seconds before calling the switch successful:

- First approximately 15 seconds: `ground_litter_state=warming_up`, count 0,
  active events 0, confirmed events 0.
- Then: `abstaining` until three consecutive normal samples.
- Stable afternoon scene: `running`, count 0, confirmed events 0. A small raw
  candidate count is acceptable because raw components are never drawn.
- `ground_litter_last_inference_ms` must be below the 2000 ms cadence and is
  expected to be below about 500 ms without the duplicate actor model.
- `capture_fps`, `pre_encode_fps`, and `publish_fps` should remain near 25 FPS.
- `duplicate_publish_fps` should remain 0.
- Container restart count must remain 0 and logs must contain no new traceback,
  exception, or fatal pipeline error.
- Decode the output for at least 60 seconds and verify frame progression, not
  only API metrics.

If the environment remains non-normal, the hardened behavior is safe because
it displays no boxes, but detection is not operational. Do not report that as
full acceptance; collect the environment state and current frame for another
profile/normalization review.

### E. Rollback

If health, FPS, output decoding, or stream creation fails:

1. Run the original five-file Compose chain without the hardening override.
   This selects the currently deployed V3.2 image.
2. Recreate the stream using the preserved input URL and the old request
   options if the API container had already been replaced.
3. Confirm health, image identity, restart count, and output decoding.
4. Leave MediaMTX, camera-control, and web-gateway untouched.

Do not delete images, release directories, profiles, TensorRT engines, or
captured evidence during rollback.

## 9. Security and operational notes

- The user pasted credentials and signed URLs earlier. They are deliberately
  absent from this handoff.
- Do not print complete API responses because `rtsp_url` contains viewer
  credentials.
- Do not print `worker.json`, `api.json`, environment values, or shell
  variables containing URLs/keys.
- The external API port supplied by the user was unreachable from the Codex
  execution host, although the user successfully created a stream and the
  server-local API is healthy. Use server-local port 8080 for deployment
  operations and let the user validate the external mapping through Postman.
- The repository is intentionally dirty and contains extensive untracked PoC,
  archive, output, model, and documentation files. Do not run `git clean`,
  `git reset`, broad restore commands, or delete unrelated artifacts.

## 10. Definition of done

The hardening is complete only when all of the following are true:

1. The API runs the hardening image ID.
2. The current camera stream has been recreated with the afternoon profile,
   16-point ROI, no actor model, and 0.5 FPS side cadence.
3. Startup produces no active or confirmed events and no flashing boxes.
4. Stable afternoon footage produces no false displayed boxes.
5. Side inference stays within its cadence and the output advances continuously.
6. The user receives the new Postman request parameters and the new stream ID,
   without credentials being repeated in chat.
