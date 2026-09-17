# Continuous PTZ demo: LAN validation

This runbook is for the operator inside the camera network. It does not
authorize remote deployment and never places credentials in shell history.

1. Run `python scripts/check_demo_environment.py --config config/api.json` and
   record the JSON. Add `--base-url http://127.0.0.1:38080` only when the API
   is locally reachable. A successful check does not calibrate PTZ direction,
   zoom ratio, HOME, or capture.
2. Build the DeepStream bundle with
   `python scripts/package_deepstream_bundle.py --help`; inspect the generated
   manifest and SHA-256 before loading it. Keep the existing config and image
   as rollback material. Restarting the API stops in-memory streams.
3. Create one stream using the redacted request in
   `DEMO_CONTINUOUS_TRACKING_IMPLEMENTATION_PLAN.md`, adding
   `"tracking_profile": "demo_continuous"` and injecting `input_url` and keys
   through the local secret mechanism. Verify the response contains
   `effective_policy.profile=demo_continuous`.
4. Start `python scripts/collect_demo_session.py --base-url ... --stream-id ...
   --output ./demo-collection`; this is passive and does not move the camera.
   Keep the complete output video and trace files locally for review.
5. Generate `python scripts/build_demo_report.py --input ./demo-collection
   --output ./demo-report.md`. Export only after manual review with
   `python scripts/export_demo_bundle.py --input ./demo-collection --output
   ./demo-export`; inspect the manifest and ensure recordings do not show
   credentials.
6. End the run with the existing DELETE or explicit PTZ return-home endpoint.
   Confirm `manual_hold` and that no old command resumes afterward.

The offline evaluator is `python scripts/evaluate_demo_continuous.py --output
/tmp/demo-continuous.json`. It reports normal and pressure cases separately;
the simulator uses a detector stub and does not prove image accuracy, GPU
runtime, SDK behavior, or vessel-number readability.
