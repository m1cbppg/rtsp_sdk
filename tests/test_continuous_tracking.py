import time
import unittest

from rtsp_annotator.continuous_tracking import (
    ClosedLoopCase,
    DemoTrackingController,
    LatestIntentDispatcher,
    ObservationKind,
    TrackingObservation,
    simulate_closed_loop,
)


class ContinuousTrackingTests(unittest.TestCase):
    def test_first_observation_decides_follow_and_zoom(self):
        obs = TrackingObservation(
            "boat-1", .44, .42, .10, .12, "primary", 1, 0, 1.0, 1.0
        )
        decision = DemoTrackingController().decide(obs, now=1.0, view_epoch=0)
        self.assertIsNotNone(decision)
        self.assertGreaterEqual(decision[2], 1)

    def test_held_observation_never_drives_control(self):
        obs = TrackingObservation(
            "boat-1", .44, .42, .3, .3, "sidecar", 1, 0, 1.0, 1.0,
            kind=ObservationKind.HELD,
        )
        self.assertIsNone(DemoTrackingController().decide(obs, now=1.1, view_epoch=0))

    def test_dispatcher_keeps_only_latest_pending_intent(self):
        seen = []
        gate = __import__("threading").Event()

        def execute(intent):
            seen.append(intent.sequence)
            gate.wait(.2)

        dispatcher = LatestIntentDispatcher(execute)
        dispatcher.submit(target_id="b", x=.4, y=.5, zoom_delta=0, reason="a", view_epoch=0)
        time.sleep(.03)
        dispatcher.submit(target_id="b", x=.6, y=.5, zoom_delta=0, reason="b", view_epoch=0)
        dispatcher.submit(target_id="b", x=.7, y=.5, zoom_delta=0, reason="c", view_epoch=0)
        gate.set()
        time.sleep(.05)
        dispatcher.close()
        self.assertGreaterEqual(len(seen), 1)
        self.assertEqual(seen[-1], 3)

    def test_dispatcher_error_faults_and_cancels_pending_intents(self):
        """A failed SDK action must not leave a live queue of stale moves."""
        import threading

        failed = threading.Event()
        errors = []
        seen = []

        def execute(intent):
            seen.append(intent.sequence)
            raise RuntimeError("synthetic camera failure")

        def on_error(intent, exc):
            errors.append((intent.sequence, str(exc)))
            failed.set()

        dispatcher = LatestIntentDispatcher(execute, on_error=on_error)
        try:
            dispatcher.submit(
                target_id="b", x=.4, y=.5, zoom_delta=0,
                reason="position_correction", view_epoch=0,
            )
            # This one may be pending while the first action fails.  It must
            # be discarded when the worker enters the faulted state.
            dispatcher.submit(
                target_id="b", x=.6, y=.5, zoom_delta=0,
                reason="position_correction", view_epoch=0,
            )
            self.assertTrue(failed.wait(1.0))
            self.assertEqual(dispatcher.status, "faulted")
            self.assertTrue(dispatcher.failed)
            self.assertIsNone(dispatcher.pending)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0][0], seen[0])

            # Once faulted, submit is a no-op and cannot resurrect control.
            dispatcher.submit(
                target_id="b", x=.7, y=.5, zoom_delta=0,
                reason="position_correction", view_epoch=0,
            )
            time.sleep(.05)
            self.assertEqual(seen, [seen[0]])
        finally:
            dispatcher.close()
        self.assertEqual(dispatcher.status, "closed")

    def test_dispatcher_reports_action_completion(self):
        import threading

        completed = []
        done = threading.Event()

        def execute(intent):
            return {"accepted": True, "sequence": intent.sequence}

        def on_complete(intent, result):
            completed.append((intent.sequence, result))
            done.set()

        dispatcher = LatestIntentDispatcher(execute, on_complete=on_complete)
        try:
            dispatcher.submit(
                target_id="b", x=.5, y=.5, zoom_delta=1,
                reason="progressive_zoom", view_epoch=0,
            )
            self.assertTrue(done.wait(1.0))
            self.assertEqual(completed[0][0], 1)
            self.assertEqual(completed[0][1]["accepted"], True)
            # Completion is followed by an idle state once the worker clears
            # its in-flight marker.
            for _ in range(20):
                if dispatcher.status == "idle":
                    break
                time.sleep(.01)
            self.assertEqual(dispatcher.status, "idle")
        finally:
            dispatcher.close()

    def test_closed_loop_reaches_scale_under_normal_delays(self):
        result = simulate_closed_loop(ClosedLoopCase(
            "normal", .5, .1, .2, .001, direction=1,
        ))
        self.assertTrue(result["quality_pass"], result)
        self.assertGreater(result["zoom_in_commands"], 0)
        self.assertGreaterEqual(result["zoom_out_commands"], 0)


if __name__ == "__main__":
    unittest.main()
