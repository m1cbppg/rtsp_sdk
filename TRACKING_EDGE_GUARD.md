# Tracking Edge Guard

2026-09-07. Experimental, local implementation only. Not a completed
`demo_continuous` profile and not device-validated.

## Confirmed User Priority

Maintain close-up following in ordinary sailing. Do not zoom out merely
because the boat is larger than the nominal scale. If sudden acceleration or
imminent clipping makes continued close-up tracking unsafe, one small zoom-out
is allowed to preserve the target. Follow at the safer scale before restoring
close-up. Number readability is a separate image/OCR requirement, not implied
by reaching a fixed boat-box width.

This supersedes the original plan's unconditional negative-zoom prohibition,
but does not authorize blind zoom-out after loss or automatic HOME.

## Implemented Scope

The opt-in guard replaces the zoom decision **inside the existing continuous
tracking stage**, not the preceding verification/capture lifecycle. It uses
source timestamps, current position evidence and screen-space velocity to
estimate remaining edge time and the position at the end of a control window.
The window includes observation age, assumed device response, assumed motion
duration, and upstream video/jitter uncertainty separately.

Two consistent displacement intervals are required. Acceleration without edge
risk is insufficient. A predicted imminent exit can also trigger protection
without acceleration, since holding a dangerously narrow view would defeat the
user's priority. Each escape request is exactly `zoom_delta=-1`. It is not an
assertion about the physical optical ratio of one SDK step.

Following an acknowledged escape, default cooldown is 8 seconds; no repeated
shrinking during cooldown and no immediately opposing zoom-in. Zoom-in resumes
one step at a time only after 3 seconds of stable centering and adequate edge
margin. Median scale and a target-reached latch prevent size jitter from
repeatedly restarting zoom. Position correction continues when eligible.
Successful pure position corrections preserve the stability window; unsafe or
unreliable updates and zoom actions reset it. An escape decision bypasses the
ordinary command-interval throttle, but cannot replace an in-flight action.

Motion history is reset after camera actions, source/ID changes, gaps and large
scale discontinuities. There is no calibrated world-speed or image-motion
compensation claim. Pure held, predicted, unknown-provenance, duplicate and
stale observations cannot authorize risk zoom. Primary observations are tagged
as image tracker updates only with explicit usable tracker confidence; otherwise
the motion guard uses reliable sidecar observations instead of masking them
with unknown-provenance primary boxes.

## Parameters

Merge these fields into the existing `ptz_verification` request; retain the
camera ID, credentials environment, ROI and other stream configuration.

```json
{
  "enabled": true,
  "continuous_tracking": true,
  "tracking_edge_guard_enabled": true,
  "tracking_edge_response_seconds": 1.0,
  "tracking_edge_motion_seconds": 0.5,
  "tracking_edge_uncertainty_seconds": 0.25,
  "tracking_edge_cooldown_seconds": 8.0,
  "tracking_edge_stable_seconds": 3.0,
  "tracking_edge_maximum_age_seconds": 0.5
}
```

All delays are assumptions until measured. The original user's perceived
0.5-1 second response may already include video delay: do not add that delay
twice when replacing these assumptions with measurements.

The default is disabled, preserving existing API callers. Enabling requires
both PTZ enabled and continuous tracking. Worker serialization and the stream
detail `ptz_verification.edge_guard` expose the parameters and effective policy.
In this stage the guard ignores old bidirectional scale-step/hysteresis settings
and all blind recovery zoom settings. Initial zoom, capture, evidence, duration
and HOME lifecycle are **unchanged**, so this is not yet the requested full demo.

Trace event `tracking.edge_guard` records the reason, requested step, motion
reliability, predicted margin and edge time. Actual dispatch/completion use the
existing camera-control trace. A decision log is not proof of physical motion.

## Local Verification

From the RTSP repository, run:

```bash
.venv/bin/python -m unittest discover -s tests -p test_tracking_edge_guard.py -v
.venv/bin/python -m scripts.evaluate_tracking_edge_guard --output /tmp/edge-guard-policy-v1.json
```

The simulator has 48 fixed cases of 60 seconds: left/right sailing, HOME-frame
speeds 1%/3% per second, 1% to 6% bursts for 2 seconds, and a separate predeclared
1% to 12% stress burst. Mechanical start delay is 0.5/0.75/1 second; video delay
is 0.1/0.3 seconds; motion lasts 0.2 seconds independently. It begins in close-up
at 4x and models a nonreplaceable in-flight action. It is a policy-only detector
stub test, NOT HTTP/DeepStream/OSD/identity or real image validation. It does not
prove successful initial zoom or actual number recognition.

Current results: ordinary sailing 24/24 passed with zero clipping and zero
zoom-out; burst 6/12 passed (failed cases include clipping up to 0.85 seconds);
stress burst 0/12 passed. The report command deliberately exits 1 because
non-stress cases still fail. No cases were relabelled to hide failure.
The final RTSP unit-test suite passed 271 tests; this does not override the
separate performance acceptance failures above.

## Remaining Risks

The production coordinator still waits synchronously for each physical action.
This guard cannot replace an in-flight nonreplaceable action, infer arbitrary
acceleration before it is observed, or guarantee that an emergency zoom arrives
before exit. Cooldown also deliberately prevents repeated emergency shrink.
Uncalibrated optical response and stale upstream video remain real limitations.

The larger unified-session, asynchronous latest-intent, compensation and
single-target-display work in the implementation plan remains incomplete.
Both tracking-update Dockerfiles now include the new module and observation
dependencies; the older code-update and pad-hotfix entrypoints also carry these
dependencies. No image was built or deployed in this change. Do not
treat passing unit tests or the partial simulator results as L1/L2 acceptance.
