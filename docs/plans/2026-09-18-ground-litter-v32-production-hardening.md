# Ground Litter V3.2 production hardening plan

## Observed failures

1. The first acceptance ROI is a four-point trapezoid that cuts off part of
   the sidewalk and includes unstable shop/parking boundaries.
2. The frozen reference is from 06:03 while the production report is from
   14:32. The live stream accumulated more than 60 active events and more than
   40 confirmed events during broad daylight change, so hidden events can
   flash when the environment state changes.
3. The side process performs a second 1280 px actor-model inference. Live
   `ground_litter_last_inference_ms` is about 2.8–3.1 seconds for a requested
   1 FPS cadence. The main pipeline currently reports 25 FPS and a direct
   15-second output decode did not freeze, but the redundant GPU load is an
   avoidable stall risk.

## Implementation

1. Replace the acceptance ROI with a reviewed multi-point polygon following
   the visible curb and wall/floor boundary. Exclude merchant fixtures and OSD
   areas from the valid ground mask.
2. Build a new immutable afternoon profile from multiple current raw input
   samples. Store source hashes and the review polygon in profile metadata.
3. Add a startup suppression interval. During this interval the processor may
   align and normalize, but it cannot create, confirm, or draw events.
4. Treat every non-`NORMAL` environment state as abstaining. Reset event
   memory after an environment transition and require consecutive normal
   samples before accepting new evidence.
5. Remove `actor_model` from this acceptance stream and reuse actor boxes from
   the primary DeepStream detector. Set the initial analysis cadence to 0.5
   FPS. Raw analysis remains on the existing lossy queue.

## Verification

1. Unit-test warm-up suppression, non-normal abstention, memory reset, and
   recovery after stable normal samples.
2. Overlay the polygon on the raw production frame and retain the review image.
3. Replay the current production samples through the packaged processor. Clean
   footage must produce zero displayed boxes and zero confirmed events after
   warm-up.
4. Build an incremental image, run profile checksum/API validation, recreate
   the acceptance stream, and observe metrics for at least 60 seconds.
5. Require main publish FPS near source FPS, zero duplicate publish FPS,
   side-process latency below its cadence, and no box flicker. Roll back the
   API image if the main pipeline becomes unhealthy.
