"""Unit tests for the server-side V3.3 budget measurement script.

The script only runs for real on the DeepStream host, so its logic must be
provable locally with injected payloads. The important contract under test is
honesty: a metric that was never reported must come back as ``passed: None``
(not evaluable) and must never be counted as compliant.
"""

from __future__ import annotations

import subprocess
import unittest

from scripts.measure_ground_litter_v33_server import (
    ADDED_VRAM_MAX_MIB,
    CROP_BATCH_P95_MAX_MS,
    FULL_SCAN_P95_MAX_MS,
    NORMAL_TICK_P95_MAX_MS,
    evaluate_budget,
    extract_sample,
    main,
    percentile,
    read_gpu_memory_mib,
    run_measurement,
    stage_series,
    summarize_branch_states,
    summarize_chain,
    summarize_stage,
)


def payload(
    *,
    publish_fps=24.9,
    unique_publish_fps=24.9,
    duplicate_publish_fps=0.0,
    pipeline_healthy=True,
    state="running",
    hybrid=True,
    **hybrid_values,
):
    hybrid_block = {
        "last_prior_ms": 540.0,
        "last_full_scan_ms": 0.0,
        "last_crop_batch_ms": 0.0,
        "last_total_ms": 560.0,
        "input_frame_age_ms": 700.0,
        "dropped_analysis_frames": 0,
        "semantic_model_runs_full": 0,
        "semantic_model_runs_crop": 0,
        "branch_state": "ok",
        "prior_environment_state": "NORMAL",
    }
    hybrid_block.update(hybrid_values)
    document = {
        "stream_id": "stream-1",
        "ground_litter": {"state": state, "count": 0},
        "metrics": {
            "publish_fps": publish_fps,
            "unique_publish_fps": unique_publish_fps,
            "duplicate_publish_fps": duplicate_publish_fps,
            "pipeline_healthy": pipeline_healthy,
        },
    }
    if hybrid:
        document["ground_litter"]["hybrid"] = hybrid_block
    return document


class ExtractSampleTests(unittest.TestCase):
    def test_reads_chain_and_hybrid_fields(self):
        sample = extract_sample(payload())
        self.assertTrue(sample["hybrid_present"])
        self.assertEqual(sample["publish_fps"], 24.9)
        self.assertEqual(sample["last_prior_ms"], 540.0)
        self.assertEqual(sample["branch_state"], "ok")

    def test_absent_hybrid_block_is_reported_as_absent(self):
        sample = extract_sample(payload(hybrid=False))
        self.assertFalse(sample["hybrid_present"])
        self.assertIsNone(sample["last_prior_ms"])

    def test_absent_metrics_stay_none_instead_of_zero(self):
        sample = extract_sample({"ground_litter": {"state": "running"}})
        for key in ("publish_fps", "duplicate_publish_fps", "pipeline_healthy"):
            self.assertIsNone(sample[key], key)

    def test_boolean_metrics_are_not_coerced_to_numbers(self):
        sample = extract_sample(payload(pipeline_healthy=True))
        self.assertIs(sample["pipeline_healthy"], True)


class PercentileTests(unittest.TestCase):
    def test_empty_sample_has_no_percentile(self):
        self.assertIsNone(percentile([], 0.95))

    def test_single_value_is_its_own_percentile(self):
        self.assertEqual(percentile([12.0], 0.95), 12.0)

    def test_interpolates_between_ranks(self):
        self.assertAlmostEqual(percentile([0.0, 10.0], 0.5), 5.0)


class StageSeriesTests(unittest.TestCase):
    def test_splits_normal_and_scan_ticks(self):
        samples = [
            extract_sample(payload(last_total_ms=500.0)),
            extract_sample(
                payload(
                    last_total_ms=1400.0,
                    last_full_scan_ms=1300.0,
                    semantic_model_runs_full=1,
                )
            ),
        ]
        series = stage_series(samples)
        self.assertEqual(series["normal_tick_total_ms"], [500.0])
        self.assertEqual(series["scan_tick_total_ms"], [1400.0])
        self.assertEqual(series["full_scan_ms"], [1300.0])

    def test_cumulative_scan_counter_does_not_turn_later_ticks_into_scans(self):
        samples = [
            extract_sample(
                payload(
                    last_total_ms=1400.0,
                    last_full_scan_ms=1300.0,
                    semantic_model_runs_full=1,
                )
            ),
            extract_sample(
                payload(
                    last_total_ms=500.0,
                    last_full_scan_ms=0.0,
                    semantic_model_runs_full=1,
                )
            ),
        ]
        series = stage_series(samples)
        self.assertEqual(series["scan_tick_total_ms"], [1400.0])
        self.assertEqual(series["normal_tick_total_ms"], [500.0])

    def test_zero_crop_timing_is_not_counted_as_a_crop_batch(self):
        samples = [
            extract_sample(payload(last_crop_batch_ms=0.0)),
            extract_sample(payload(last_crop_batch_ms=420.0)),
        ]
        self.assertEqual(stage_series(samples)["crop_batch_ms"], [420.0])

    def test_missing_values_are_skipped_not_zero_filled(self):
        series = stage_series([extract_sample({"ground_litter": {}})])
        self.assertEqual(series["input_frame_age_ms"], [])
        self.assertEqual(series["normal_tick_total_ms"], [])

    def test_summary_reports_counts_and_p95(self):
        series = {"x": [100.0] * 19 + [10_000.0]}
        summary = summarize_stage(series)
        self.assertEqual(summary["x"]["count"], 20)
        self.assertLess(summary["x"]["p50"], 200.0)
        self.assertEqual(summary["x"]["max"], 10_000.0)


class BudgetVerdictTests(unittest.TestCase):
    def _stages(self, *, normal, scan, crop, age):
        return {
            "normal_tick_total_ms": {"p95": normal},
            "full_scan_ms": {"p95": scan},
            "crop_batch_ms": {"p95": crop},
            "input_frame_age_ms": {"p95": age},
        }

    def test_compliant_measurement_passes_every_row(self):
        verdicts = evaluate_budget(
            stages=self._stages(normal=1200.0, scan=900.0, crop=300.0, age=1500.0),
            chain={
                "publish_fps_min": 24.9,
                "unique_publish_fps_min": 24.9,
                "duplicate_publish_fps_max": 0.0,
                "pipeline_healthy_all": True,
            },
            added_vram_mib=800.0,
        )
        self.assertTrue(all(row["passed"] for row in verdicts), verdicts)

    def test_over_budget_stage_fails(self):
        verdicts = evaluate_budget(
            stages=self._stages(
                normal=NORMAL_TICK_P95_MAX_MS + 1.0,
                scan=FULL_SCAN_P95_MAX_MS + 1.0,
                crop=CROP_BATCH_P95_MAX_MS + 1.0,
                age=1.0,
            ),
            chain={
                "publish_fps_min": 24.9,
                "unique_publish_fps_min": 24.9,
                "duplicate_publish_fps_max": 0.0,
                "pipeline_healthy_all": True,
            },
            added_vram_mib=ADDED_VRAM_MAX_MIB + 1.0,
        )
        failed = {row["metric"] for row in verdicts if row["passed"] is False}
        self.assertEqual(
            failed,
            {
                "normal_tick_p95_ms",
                "full_scan_p95_ms",
                "crop_batch_p95_ms",
                "added_vram_mib",
            },
        )

    def test_unreported_metrics_are_not_evaluable_never_compliant(self):
        verdicts = evaluate_budget(
            stages=self._stages(normal=None, scan=None, crop=None, age=None),
            chain={
                "publish_fps_min": None,
                "unique_publish_fps_min": None,
                "duplicate_publish_fps_max": None,
                "pipeline_healthy_all": None,
            },
            added_vram_mib=None,
        )
        self.assertTrue(all(row["passed"] is None for row in verdicts), verdicts)

    def test_vram_without_baseline_is_not_evaluable(self):
        # The budget is the *added* VRAM; without a baseline the absolute peak
        # must not be treated as the increment.
        verdicts = evaluate_budget(
            stages=self._stages(normal=1000.0, scan=900.0, crop=300.0, age=1000.0),
            chain={
                "publish_fps_min": 25.0,
                "unique_publish_fps_min": 25.0,
                "duplicate_publish_fps_max": 0.0,
                "pipeline_healthy_all": True,
            },
            added_vram_mib=None,
        )
        vram = next(r for r in verdicts if r["metric"] == "added_vram_mib")
        self.assertIsNone(vram["passed"])

    def test_duplicate_frames_fail_even_when_fps_is_high(self):
        verdicts = evaluate_budget(
            stages=self._stages(normal=1000.0, scan=900.0, crop=300.0, age=1000.0),
            chain={
                "publish_fps_min": 25.0,
                "unique_publish_fps_min": 18.0,
                "duplicate_publish_fps_max": 0.4,
                "pipeline_healthy_all": True,
            },
            added_vram_mib=100.0,
        )
        failed = {row["metric"] for row in verdicts if row["passed"] is False}
        self.assertEqual(failed, {"unique_publish_fps_min", "duplicate_publish_fps_max"})


class ChainSummaryTests(unittest.TestCase):
    def test_minimums_and_health_across_samples(self):
        samples = [
            extract_sample(payload(publish_fps=25.0)),
            extract_sample(payload(publish_fps=21.5, pipeline_healthy=False)),
        ]
        summary = summarize_chain(samples)
        self.assertEqual(summary["publish_fps_min"], 21.5)
        self.assertEqual(summary["duplicate_publish_fps_max"], 0.0)
        self.assertFalse(summary["pipeline_healthy_all"])
        self.assertEqual(summary["pipeline_healthy_samples"], 2)

    def test_branch_state_counts_are_explicit(self):
        samples = [
            extract_sample(payload(branch_state="ok")),
            extract_sample(payload(branch_state="ok")),
            extract_sample(payload(branch_state="prior_degraded")),
            extract_sample({"ground_litter": {}}),
        ]
        self.assertEqual(
            summarize_branch_states(samples),
            {"ok": 2, "prior_degraded": 1, "missing": 1},
        )


class RunMeasurementTests(unittest.TestCase):
    def test_document_records_vram_delta_and_verdicts(self):
        sequence = [payload(publish_fps=24.5), payload(last_prior_ms=600.0)]
        vram_sequence = iter([2000.0, 2100.0])

        doc = run_measurement(
            base_url="http://api",
            stream_id="s1",
            api_key="super-secret",
            samples=2,
            interval_seconds=0.0,
            baseline_vram_mib=1500.0,
            fetch=lambda *_: sequence.pop(0),
            gpu_reader=lambda: next(vram_sequence),
            sleep=lambda _: None,
            progress=lambda _: None,
        )
        self.assertEqual(doc["samples_collected"], 2)
        self.assertEqual(doc["vram"]["peak_mib"], 2100.0)
        self.assertEqual(doc["vram"]["added_mib"], 600.0)
        self.assertEqual(doc["branch_state_counts"], {"ok": 2})
        self.assertIn("added_vram_mib", {v["metric"] for v in doc["verdicts"]})

    def test_api_key_never_appears_in_the_measurement_document(self):
        doc = run_measurement(
            base_url="http://api",
            stream_id="s1",
            api_key="super-secret",
            samples=1,
            interval_seconds=0.0,
            baseline_vram_mib=None,
            fetch=lambda *_: payload(),
            gpu_reader=lambda: 100.0,
            sleep=lambda _: None,
            progress=lambda _: None,
        )
        self.assertNotIn("super-secret", repr(doc))

    def test_sampling_errors_are_recorded_and_do_not_abort(self):
        def failing_fetch(*_args):
            raise TimeoutError("no answer")

        doc = run_measurement(
            base_url="http://api",
            stream_id="s1",
            api_key="k",
            samples=2,
            interval_seconds=0.0,
            baseline_vram_mib=None,
            fetch=failing_fetch,
            gpu_reader=lambda: None,
            sleep=lambda _: None,
            progress=lambda _: None,
        )
        self.assertEqual(doc["samples_collected"], 0)
        self.assertEqual(len(doc["errors"]), 2)
        self.assertTrue(all(row["passed"] is None for row in doc["verdicts"]))


class GpuReaderTests(unittest.TestCase):
    def test_parses_first_gpu_value(self):
        class Result:
            stdout = "2100\n"

        self.assertEqual(read_gpu_memory_mib(lambda *a, **k: Result()), 2100.0)

    def test_missing_nvidia_smi_returns_none(self):
        def boom(*_args, **_kwargs):
            raise FileNotFoundError("nvidia-smi")

        self.assertIsNone(read_gpu_memory_mib(boom))

    def test_garbage_output_returns_none(self):
        class Result:
            stdout = "N/A\n"

        self.assertIsNone(read_gpu_memory_mib(lambda *a, **k: Result()))

    def test_nonzero_exit_returns_none(self):
        def failing(*_args, **_kwargs):
            raise subprocess.CalledProcessError(1, ["nvidia-smi"])

        self.assertIsNone(read_gpu_memory_mib(failing))


class DryRunTests(unittest.TestCase):
    def test_dry_run_does_not_require_an_api_key(self):
        self.assertEqual(main(["--stream-id", "s1", "--dry-run"]), 0)


if __name__ == "__main__":
    unittest.main()
