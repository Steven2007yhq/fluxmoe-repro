import unittest

from fluxmoe.events import Timeline


class TimelineTests(unittest.TestCase):
    def test_stream_serialization_and_cross_stream_overlap(self):
        timeline = Timeline()
        load = timeline.submit("load", 4, "copy")
        decode = timeline.submit("decode", 3, "decode")
        copy2 = timeline.submit("copy2", 2, "copy")
        compute = timeline.submit("compute", 5, "compute", [load, decode])
        self.assertEqual((copy2.start_ms, copy2.end_ms), (4, 6))
        self.assertEqual((compute.start_ms, compute.end_ms), (4, 9))
        self.assertEqual(timeline.total_ms, 9)
        timeline.run_until()
        self.assertEqual(timeline.now_ms, 9)
        self.assertTrue(all(row["completed"] for row in timeline.trace()))

    def test_callbacks_are_deferred_and_dependencies_complete_first(self):
        timeline = Timeline()
        observed = []
        first = timeline.submit("first", 2, "a", action=lambda: observed.append("first"))
        second = timeline.submit("second", 0, "b", [first], lambda: observed.append("second"))
        later = timeline.submit("later", 8, "c", action=lambda: observed.append("later"))
        self.assertEqual(observed, [])
        timeline.run_until(second)
        self.assertEqual(observed, ["first", "second"])
        self.assertFalse(later.completed)
        timeline.run_until(second)
        self.assertEqual(observed, ["first", "second"])
        timeline.run_until()
        self.assertEqual(observed, ["first", "second", "later"])

    def test_shared_link_does_not_receive_unlimited_bandwidth(self):
        timeline = Timeline()
        first = timeline.submit("a", 5, "copy0", resource="pcie")
        second = timeline.submit("b", 7, "copy1", resource="pcie")
        decode = timeline.submit("c", 4, "decode", resource="decoder")
        self.assertEqual(second.start_ms, first.end_ms)
        self.assertEqual(decode.start_ms, 0)
        self.assertEqual(timeline.total_ms, 12)

    def test_rejects_invalid_times_and_foreign_dependencies(self):
        timeline = Timeline()
        for duration in [-1, float("nan"), float("inf")]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                timeline.submit("bad", duration, "a")
        other = Timeline().submit("other", 1, "a")
        with self.assertRaises(ValueError):
            timeline.submit("bad", 1, "a", [other])
        with self.assertRaises(ValueError):
            timeline.run_until(other)

    def test_submit_after_progress_cannot_schedule_in_past(self):
        timeline = Timeline()
        first = timeline.submit("a", 5, "a")
        timeline.run_until(first)
        self.assertEqual(timeline.submit("b", 1, "b").start_ms, 5)

    def test_callback_failure_stops_dependent_operations(self):
        timeline = Timeline()
        def fail():
            raise ValueError("materialization failed")
        first = timeline.submit("failed", 1, "a", action=fail)
        second = timeline.submit("dependent", 1, "a")
        with self.assertRaisesRegex(ValueError, "materialization failed"):
            timeline.run_until()
        self.assertFalse(first.completed)
        self.assertFalse(second.completed)
        with self.assertRaises(RuntimeError):
            timeline.run_until()
        self.assertEqual(timeline.trace()[0]["error"], "materialization failed")


if __name__ == "__main__":
    unittest.main()
