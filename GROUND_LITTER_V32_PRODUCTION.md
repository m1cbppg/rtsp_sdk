# Ground Litter V3.2 production acceptance

V3.2 is an explicit fixed-camera mode of the existing `ground_litter` side
path. The hardened camera-01 request uses the immutable
`camera_01_v32_afternoon_1080p` Clean Reference profile and does not learn a
new background from live startup frames.

The output RTSP draws only confirmed lifecycle events. Raw change components,
pending events, actor/context occlusion, insufficient visible ground,
environment-change abstention, and clean-confirmation frames are not drawn.

The 1080p profile matches the production DeepStream mux size. It was built
from a reviewed raw 14:40 production frame after the original 06:03 profile
created dozens of false afternoon events. Its 16-point polygon follows the
curb and wall/floor boundary, and a separate exclusion covers the merchant
fixture near the far end of the sidewalk.

The processor suppresses all event evidence for its first 15 seconds. It also
abstains for every non-normal environment state, clears event memory after a
lighting transition, and waits for three consecutive normal samples before
accepting new evidence. This prevents hidden startup events from flashing when
the environment state changes.

The hardened request runs at 0.5 FPS and does not launch a second Ultralytics
actor model. It reuses person/vehicle boxes already produced by the main
DeepStream detector, keeping the side path lossy and avoiding redundant GPU
load.

Read credentials interactively so the API key and signed RTSP URL do not enter
shell history files or repository artifacts. Confirm the public mapping before
creating a stream:

```bash
export RTSP_API_BASE_URL='http://<api-host>:<port>'
printf 'API key: ' >&2
IFS= read -r -s RTSP_API_KEY
printf '\nSigned RTSP URL: ' >&2
IFS= read -r INPUT_RTSP_URL
export RTSP_API_KEY INPUT_RTSP_URL

curl --fail-with-body "$RTSP_API_BASE_URL/health"

jq --arg input_url "$INPUT_RTSP_URL" '.input_url = $input_url' \
  config/ground_litter_v32_stream_request.example.json \
  | curl --fail-with-body --location \
      "$RTSP_API_BASE_URL/v1/streams" \
      --header "X-API-Key: $RTSP_API_KEY" \
      --header 'Content-Type: application/json' \
      --data-binary @-
```

The create response contains `stream_id` and `rtsp_url`. Poll status with:

```bash
curl --fail-with-body \
  --header "X-API-Key: $RTSP_API_KEY" \
  "$RTSP_API_BASE_URL/v1/streams/<stream_id>"
```

Acceptance fields under `ground_litter`:

- `state=warming_up`: startup evidence is deliberately suppressed.
- `state=running`: the side process is producing current decisions.
- `state=abstaining`: a broad environment change is present; boxes are hidden.
- `count`: boxes currently sent to OSD. This must stay zero for ordinary clean
  footage and actor occlusion.
- `raw_candidates`: current internal components. These are never drawn.
- `active_events`, `confirmed_events`, `cleared_events`: lifecycle counters.
- `environment_state`: `NORMAL`, `GLOBAL_LIGHT_CHANGE`, or
  `ENVIRONMENT_CHANGE`.
- `last_inference_ms`: use this to compare the requested cadence with actual
  capacity. At 0.5 FPS the sampling period is 2000 ms.

  **Corrected 2026-09-18.** An earlier revision claimed this "should normally be
  below 500 ms without the duplicate actor model". That was wrong. Measured on
  the production RTX 3060 Ti host at 1920×1080:

  - `NORMAL` (stable scene): ~750–1080 ms. Ground truth from `/proc` CPU deltas
    on the side process was **1076.7 CPU-ms per frame**.
  - `GLOBAL_LIGHT_CHANGE`: ~1800–3150 ms, because `protected_normalize()` does
    data-dependent mask, dilation and connected-component work that becomes
    dense during a light transition.

  So stable-scene inference fits the 2000 ms budget with about 2× margin, while
  light-transition frames exceed it. Those frames return `abstaining` anyway,
  and the side path is lossy and always processes the newest frame, so staleness
  stays bounded by one update.

  Do **not** measure this with an OpenCV benchmark run inside the API container
  while the live side process is running: the throttled host CPU oversubscribes
  and the benchmark reports roughly 2–3× the real cost. Measure the live side
  process instead (`/proc/<pid>/stat` utime+stime divided by analyzed frames).

  Note also that evidence accrues `1/analysis_fps` seconds per observation, so
  lowering `analysis_fps` reduces the number of observations needed to reach
  `confirm_visible_seconds`. At `analysis_fps=0.33` a 5.0 s confirmation is
  reached after only **two** observations instead of three. If a lower cadence
  is ever needed, raise `confirm_visible_seconds` and `clear_confirm_seconds` to
  7.0 at the same time.

For first acceptance, observe at least 30 minutes of ordinary traffic before
staging one removable test object if that becomes practical. The output must
not show raw-frame box flicker. A real target should require five seconds of
cumulative visible evidence, disappear while occluded or while clean evidence
is being confirmed, close after five consecutive valid clean seconds, and get
a new ID if it appears again after closure.

Because evidence accrues one `sample_period` per observation, "five seconds"
means three observations at 0.5 FPS (3 × 2.0 s = 6.0 s ≥ 5.0 s), not five.

## Acceptance status on 2026-09-18

The hardening release was switched into production and verified. Full evidence
and the correction to the handoff document are in
`GROUND_LITTER_V32_HARDENING_RESULT_20260918.md`. In short:

- startup suppression and the three-sample stability gate were both observed
  live, with zero raw candidates and zero events throughout;
- `count` stayed 0 in every sample;
- the main chain held ~25 FPS with `duplicate_publish_fps=0`, and the output
  stream decoded at 25.05 FPS with zero errors;
- the camera was in `GLOBAL_LIGHT_CHANGE` at 17:00 CST, so the processor stayed
  in `abstaining` and **detection was not operational**. The 14:40 profile does
  not cover that hour. Re-run this acceptance in matching light, or build a
  profile for the target hour, before claiming accuracy.

