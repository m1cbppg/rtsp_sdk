# Ground Litter V3.2 production deployment sheet

## Prepared candidate

- Bundle: `dist/ground-litter-v32-20260918/ground-litter-v32-context.tar.gz`
- Bundle SHA-256: `c16f403bea3e3279ff6fd1ac09df74147d89dd386faa3da5e8e0463ef651a902`
- Contents: 19 files, including the existing reviewed litter model, production
  modules, incremental Dockerfile, 1440p audit profile, and 1080p production
  profile.
- Production request: `config/ground_litter_v32_stream_request.example.json`
- Acceptance guide: `GROUND_LITTER_V32_PRODUCTION.md`

Local evidence:

- Current-source tests: 530 passed, 37 subtests passed.
- Compileall and `git diff --check`: passed.
- 1080p production-path lifecycle replay: 65 samples in 64.02 seconds, passed.
- First event: seen 10s, confirmed 20s, actor-occluded, cleared 34s.
- Second same-location event: seen 46s, confirmed 50s with a new event ID.
- Confirmed off-target events after reviewed ROI/exclusion: zero.

## Step 1: read-only server preflight

Target recorded by repository operations history:

- SSH: `sf01@14.21.88.97:21002`
- directory: `/home/sf01/rtsp-deepstream`

The preflight will only read:

- current compose files and resolved API image;
- API/MediaMTX/camera-control/web-gateway container identity, health, restart
  count, and active streams;
- disk and GPU capacity;
- current ground-litter model/profile files and Python dependency versions.

No file is uploaded and no process is restarted during preflight.

### Observed on 2026-09-18 14:18 CST

- Host and directory matched the recorded target: `sf01-B460M-HDV` and
  `/home/sf01/rtsp-deepstream`.
- The running API image is
  `rtsp-yolo-annotator:deepstream8-context-20260916`
  (`sha256:7563a79fb12b7c3204fbd6754a6bccd83609126465f4abb70f8825cc80cc67f4`).
- The API container had been up for two days with zero restarts and returned
  `{"status":"ok"}` from its local health endpoint.
- The exact running Compose chain is:
  `docker-compose.deepstream.api.yml`,
  `docker-compose.ptz-v12.override.yml`,
  `docker-compose.demo-continuous.override.yml`, and
  `docker-compose.ground-litter.override.yml`.
- The stream API reported zero active streams. Recreating the API container
  therefore will not interrupt a currently registered inference stream.
- The RTX 3060 Ti had about 7.75 GiB free and 0% utilization. The deployment
  filesystem had about 323 GiB free.
- The existing image contains the reviewed 54.8 MB litter model and the
  expected Python/CUDA stack. V3.2 profiles are not yet present, as expected.
- The API listens locally on port 8080. A connection to the user-supplied
  public port 22198 was refused during this preflight, and neither 22198 nor
  38080 was a local listener. Public routing must be checked again after the
  candidate is switched; it is independent of the V3.2 inference code.

## Step 2: upload and candidate build

After a successful preflight:

1. Upload the 60.6 MB tarball to a new release path.
2. Verify SHA-256 before extraction.
3. Build a new image from the actual running API image as `BASE_IMAGE`:
   `rtsp-yolo-annotator:deepstream8-ground-litter-v32-20260918`.
4. Run imports, profile checksum loading, API request validation, and the
   packaged build check inside the candidate image.

This does not delete or rebuild TensorRT engines.

## Step 3: production switch

Before switching:

1. Record active streams and stop if any cannot be recreated.
2. Tag the exact running image as
   `rtsp-yolo-annotator:deepstream8-before-ground-litter-v32-20260918`.
3. Add a small compose override selecting the candidate image.
4. Validate the complete four-file Compose chain observed in preflight with
   `docker compose config -q`.
5. Run `up -d --no-deps api`.

Impact: the API container is recreated. All process-owned stream tasks stop and
must be recreated. MediaMTX, camera-control, and web-gateway are not restarted.

## Step 4: smoke check and acceptance handoff

After switching:

- check API health, container restart count, logs, and dependency/profile load;
- leave the server with no automatically created test stream;
- provide the credential-free curl template to the user, who inserts the
  current signed RTSP URL and API key locally;
- validate `ground_litter.state`, `last_inference_ms`, lifecycle counters,
  output decoding, publish FPS, and duplicate FPS after the user creates it.

The first stream uses `analysis_fps: 1.0`. If target-server
`last_inference_ms` is repeatedly at or above 1000 ms, delete that test stream,
change to `0.5`, and repeat. Raw candidates never enter OSD.

### Deployment result on 2026-09-18 14:25 CST

- Uploaded archive SHA-256 and all 19 manifest entries matched the prepared
  local release.
- Candidate build completed from the exact running base image. Dependency,
  Python compile, tile planning, API request parsing, SIFT availability, and
  both Clean Reference profile checks passed in an isolated container.
- Saved rollback tag
  `rtsp-yolo-annotator:deepstream8-before-ground-litter-v32-20260918` at image
  ID `sha256:7563a79fb12b7c3204fbd6754a6bccd83609126465f4abb70f8825cc80cc67f4`.
- Switched only the API service to
  `rtsp-yolo-annotator:deepstream8-ground-litter-v32-20260918`, image ID
  `sha256:841ca526317f22a1c10968b9a9ad6ca93ee4988767158c0a81b2bb56a15cf6fe`.
- First local health probe passed, restart count remained zero, recent logs had
  no error/exception/traceback entries, and the V3.2 API fields were present.
- MediaMTX, camera-control, and web-gateway retained their original running
  containers. Active streams remained zero.
- Public ports 22198, 8080, and 38080 were unreachable from the deployment
  execution host. Acceptance must first confirm that the user's network can
  reach the supplied 22198 mapping, or restore that external mapping without
  changing the healthy API container.

## Rollback

1. Point the final compose override back to the saved pre-V3.2 image.
2. Run `docker compose ... up -d --no-deps api`.
3. Confirm health and container identity.
4. Recreate any previously active streams from the recorded preflight list.

The candidate image, release directory, prior image, models, engines, and
profile assets remain available for audit. No destructive cleanup is part of
the deployment.

## Production hardening after first live review

The first live stream exposed three issues that the offline fixture did not:

- the four-point ROI did not follow the real sidewalk boundary;
- the 06:03 profile produced more than 60 active and more than 40 confirmed
  changes in the 14:32 scene;
- the optional 1280 px actor model made each side analysis take roughly
  2.8–3.1 seconds, despite a requested 1 FPS cadence.

The hardening release adds a reviewed 16-point afternoon profile, suppresses
startup evidence, resets event memory across non-normal lighting, requires
stable normal samples before recovery, removes the redundant actor-model
inference, and changes the first acceptance cadence to 0.5 FPS. The main
pipeline itself remained healthy at about 25 FPS, and an independent 15-second
decode read 389 frames without a failed read; the reduced side load addresses
the avoidable contention risk while longer live observation continues.
