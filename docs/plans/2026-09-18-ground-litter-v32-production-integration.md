# Ground Litter V3.2 production integration plan

## Goal

Expose the validated fixed-camera Clean Reference V3.2 lifecycle through the
existing `/v1/streams` `ground_litter` configuration while keeping the current
YOLO-only behavior compatible.

The production picture must contain only confirmed, currently supported
events. Raw change components and pending events are diagnostics and must never
be sent to OSD.

## Design

1. Add an explicit `ground_litter.mode`:
   - `yolo`: existing detector and display tracker.
   - `clean_reference_v32`: reviewed reference profile, V3 change proposals,
     actor/context occlusion checks, and V3.2 event lifecycle.
2. Resolve `profile_id` only below `models/litter/profiles`; reject absolute
   paths, traversal, missing files, malformed metadata, and checksum mismatch.
3. Package the reviewed camera profile as immutable reference image, valid
   ground mask, daylight tolerance mask, and metadata.
4. Align the reviewed reference once to the native-resolution stream. If the
   resolution or alignment is invalid, publish an error/abstention snapshot and
   leave the main RTSP pipeline running.
5. Run Clean Reference inference in the existing lossy side process. Preserve
   the native-resolution frame and existing actor detector.
6. Convert only confirmed active V3.2 events to `GroundLitterDetection`:
   - show `VISIBLE_ANOMALY` and residual-backed `ANOMALY_PENDING`;
   - hide `OCCLUDED`, `ENVIRONMENT_CHANGE`, `GROUND_UNAVAILABLE`, and
     `CLEAN_PENDING`;
   - remove `CLEARED`, `EXPIRED_PENDING`, and merged events.
7. Expose mode/profile and lifecycle counters in API status. Never expose raw
   frame data or credentials.

## Verification

1. Unit test request validation, payload round-trip, safe profile resolution,
   checksum/shape checks, lifecycle display gating, timestamp handling, and
   legacy-mode compatibility.
2. Exercise the real independent process with fakes for both modes.
3. Replay the existing 65-second lifecycle fixture through the production
   processor and assert confirm, occlusion, clear, and new-ID behavior.
4. Run focused tests, then the complete repository suite.
5. Produce two credential-free acceptance payloads: shadow/status validation
   and confirmed-event OSD validation.

## Rollout boundary

Local code, tests, profile assets, and a deployable change set are prepared
first. Server upload, image build, container replacement, and live stream
creation require a separate concrete deployment authorization under the
repository operations policy.
