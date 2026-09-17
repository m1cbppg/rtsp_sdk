# Continuous Tracking Implementation Progress

2026-09-07 follow-up: the user now permits rare small protective zoom-out for
sudden acceleration/imminent exit, while ordinary sailing should remain in
close-up. An opt-in guard is implemented in the existing tracking stage; see
`TRACKING_EDGE_GUARD.md` for API fields, tests and explicit failed delayed
scenarios. This updates only that scope; full L1 is still incomplete.

Follow-up verification: all 271 RTSP unittest cases passed, including 12 guard
tests; compileall and diff whitespace checks passed. The separate physical
policy simulation passed 30/48 scenarios: ordinary sailing 24/24 with no escape
zoom, burst 6/12, stress burst 0/12. Its exit status remains 1; those performance
failures are not hidden by the unit-test result. No device or server was used.

Date: 2026-09-07. Status: in progress, NOT L1 complete; L2 unverified.

## 2026-09-07 local implementation continuation

The following local fixes were applied on top of the existing dirty workspace:

- `LatestIntentDispatcher` now exposes lifecycle/fault state, clears pending
  intents after executor errors, and reports completion/error callbacks.
- Demo sessions reject observations from an older PTZ view generation and pass
  the worker's generation into primary snapshots. Held/unknown timestamps are
  not fabricated as current control observations.
- Demo reacquisition uses position/scale continuity and does not fall back to
  the screen-center vessel. A lost session remains in `lost_hold`, cancels
  pending control, hides the stale overlay box, and waits for the same target;
  it does not HOME or start a replacement job.
- Edge guard receives `action_completed` only after a locate operation returns.
  Evidence autofocus/capture is bounded to an idle control window and is
  skipped with a trace event when tracking remains busy.
- All PTZ incremental Dockerfiles copy and compile
  `continuous_tracking.py`. `scripts/export_demo_bundle.py` recursively
  redacts structured JSON secrets and RTSP credentials, with an end-to-end
  synthetic test.

Local evidence after these changes: `python -m unittest discover -s tests -q`
passes 280 tests; `compileall` and `git diff --check` pass. The deterministic
closed-loop matrix remains 6/6 for the predeclared normal low-video-buffer
cases (response assumptions 0.5/0.75/1.0 s), 2/15 pressure cases. The edge
guard policy simulation remains 30/48 and exits non-zero by design because
delayed/high-acceleration cases still fail. These are detector-stub/policy
simulations, not DeepStream, SDK, image-quality, or camera acceptance.

The DeepStream runtime contract script was not executable in this macOS
environment because `pyservicemaker` is unavailable. No server, camera,
container, or deployment operation was performed. L1 remains pending until
the full delayed HTTP/PTZ/OSD integration and build/runtime checks are
completed; L2 remains an explicit on-site validation item.

Local follow-up implementation (2026-09-07): added the explicit
`tracking_profile=demo_continuous` contract, a unified observation/intent
dispatcher, discovery-to-follow coordinator path, asynchronous evidence work,
and LAN check/collect/report/export scripts. The deterministic closed-loop
matrix reports 6/6 predeclared normal low-buffer cases passing (0.5/0.75/1.0 s
response assumptions) and pressure/delayed-video failures separately; it uses
a detector stub and does not prove DeepStream, SDK, image accuracy, or real
camera behavior. Full local unittest discovery now passes 275 tests.

## Baseline

The workspace started on main with 17 modified tracked files, an untracked
implementation plan and an untracked tracking Dockerfile. Existing changes
are preserved. No server or real camera was contacted.

Pre-edit SHA-256:

```text
9b81b9efa3dd43a055126d9e480fb07037a5109bb72372cf3c738584a6706cb3  rtsp_annotator/ptz_verification.py
b7723951f9ddf5bc575f19b74d65fe5068ce8795beca5877e22142eab63de748  rtsp_annotator/vessel_detection.py
a64f90f347d2071441396204d3436c30ebff93afb98f45a0d9f95e8d42b23280  rtsp_annotator/vessel_detection_process.py
173332daaddfa0e497f1a3db28319691928549a3da06c37eb1669402a15a724d  rtsp_annotator/api.py
5cfb71fa65cf1ad9c8df57df2c4693173e9cd08a726e38e3faa3d53690084564  rtsp_annotator/deepstream_worker.py
d907b7e0a1e941851117dbd10a77c7068756d74c37899372948d202fbb373107  tests/test_ptz_verification.py
886f5689254bb41d87592e204503164bd1f74c13eb81080a2c521241a294e9ee  DEMO_CONTINUOUS_TRACKING_IMPLEMENTATION_PLAN.md
```

## Execution Plan

1. A: preserve baseline, reproduce stale observations, define profile policy
   and fixed simulation scenarios before tuning control behavior.
2. B: distinguish measured, tracker-updated, predicted and held observations;
   reject stale source results and stale view generations independently.
3. C: introduce one continuous session, bounded latest-intent control,
   progressive zoom and explicit fault/manual/stop lifecycle. Integrate
   server cancellation before enabling the profile for device use.
4. D: associate display tracks with session identity, isolate optional
   evidence work and preserve independently visible vessels.
5. E: delayed physical closed-loop and HTTP integration, standard regression,
   passive LAN collection/report tools, build context and rollback guidance.

## Confirmed Defects

- `_snapshot`: fresh empty primary frames retained 20-second-old sidecar boxes.
- `VesselTrackManager.update`: empty detection frames republished static tracks
  with no last-position timestamp or evidence-kind distinction.
- Sidecar output cache accepted inference completed for an obsolete view.
- Close-up options unconditionally disabled configured large-box protection.
- `camera_control/commands.py`: STOP does not invalidate queued work closures.
  The SDK stop call alone is not a queue cancellation guarantee.
- `_execute` still serializes zoom, evidence and following; ordinary finally
  HOME is not an appropriate continuous-demo session lifecycle.

## Red/Green Evidence

`test_tracking_observation_contract.py` initially ran 3 tests: 2 failures
(stale box and held box accepted) and 1 error (missing observation metadata).
After the observation patch all 3 passed. Original PTZ regression then ran
60 tests successfully after preserving non-running source timestamps for
shutdown synchronization while rejecting their boxes for control.

No physical delayed closed-loop, demo profile, LAN toolkit, image build or
real device result is claimed by these observation tests.

## Implemented Foundation

- Per-sidecar-target evidence kind, last reliable position timestamp, source
  and source update identifier. Static held boxes remain available to ordinary
  display consumers but are excluded from PTZ control snapshots.
- Independent source freshness checks, including future-time rejection;
  non-running source timestamps remain usable by existing cleanup logic.
- View generation is carried through sidecar inference output and checked at
  cache ingress. The cache rejects older views and out-of-order source time.
- Close-up inference preserves the configured large-box confidence/area
  protections instead of overriding them to 1.0.
- With separately approved local write access, camera_control now invalidates
  queued commands at STOP and serializes cancellation against short SDK
  dispatches. Locate, preset/HOME, focus and diagnostic motion dispatches check
  the cancellation generation. Physical completion waits remain outside this
  dispatch lock. Interrupted work is failed, never reported completed.

Cross-repository files changed in camera_control: `camera_control/commands.py`,
`camera_control/api.py`, `tests/test_command_cancellation.py`. Existing dirty
changes in that repository were preserved. No deployment was performed.

## Remaining Acceptance Gaps

- No demo profile is exposed yet; standard behavior remains the active path.
- Unified session identity, compensation, latest-intent dispatch, progressive
  demo zoom and unified display are NOT implemented by this foundation patch.
- No session/sequence/deadline HTTP contract, restart recovery or interruptible
  physical idle wait is delivered. HOME still uses the existing serial command
  execution lock. STOP cancellation does not guarantee a bounded HOME latency.
- SDK calls themselves may block; their duration remains unmeasured. Native
  capture is not a short dispatch and optional-task isolation is still pending.
- Main-pipeline NvDCF evidence distinction and per-source exactly-once
  observation counting remain pending. Legacy observations remain explicitly
  `legacy_unknown`; this patch does not invent measured detector evidence.
- No fixed delayed physical scenario matrix, old/new performance comparison,
  HTTP-to-OSD closed loop, LAN tools or build artifact has been delivered.
- L1 remains incomplete. L2 remains unverified. These changes must not be
  presented as having achieved the requested camera demonstration.

Final regression evidence: RTSP unittest discovery passed 259 tests;
camera_control pytest passed 30 tests. RTSP compileall and both repositories'
`git diff --check` passed.
The camera_control test environment uses Python 3.14; RTSP uses Python 3.12.
Neither is a substitute for Linux/NVIDIA/SDK device validation.
