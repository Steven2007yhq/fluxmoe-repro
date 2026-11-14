import unittest

from fluxmoe.events import Timeline
from fluxmoe.paging import HazardError, PagedTensorPool


def sizes(layers=5):
    return {layer: {"gate_up": [4, 6], "down": [2, 3]} for layer in range(layers)}


class PagedTensorTests(unittest.TestCase):
    def test_real_aliasing_preserves_unique_virtual_identity(self):
        pool = PagedTensorPool(sizes())
        first = pool.tensor(0, 0, "gate_up")
        alias = pool.tensor(2, 0, "gate_up")
        self.assertNotEqual(first.virtual_address, alias.virtual_address)
        self.assertIs(pool.block(0, "gate_up"), pool.block(2, "gate_up"))
        self.assertIsNot(pool.block(0, "gate_up"), pool.block(1, "gate_up"))
        pool.write(0, 0, "gate_up", b"abcd")
        self.assertEqual(pool.read(0, 0, "gate_up"), b"abcd")
        pool.write(2, 0, "gate_up", b"wxyz")
        self.assertEqual(pool.block(0, "gate_up").data[:4], b"wxyz")
        with self.assertRaises(HazardError):
            pool.read(0, 0, "gate_up")
        self.assertIs(pool.tensor(0, 0, "gate_up"), first)
        self.assertEqual(pool.physical_bytes, 30)
        self.assertEqual(pool.logical_bytes, 75)

    def test_resident_layers_do_not_alias_pageable_or_other_residents(self):
        pool = PagedTensorPool(sizes(), resident_layers={0, 2})
        pool.write(0, 0, "gate_up", b"keep")
        pool.write(1, 0, "gate_up", b"old!")
        pool.write(4, 0, "gate_up", b"new!")
        self.assertEqual(pool.read(0, 0, "gate_up"), b"keep")
        self.assertIs(pool.block(1, "gate_up"), pool.block(4, "gate_up"))
        self.assertIsNot(pool.block(0, "gate_up"), pool.block(2, "gate_up"))
        self.assertEqual(pool.physical_bytes, 60)

    def test_raw_read_before_materialization_and_partial_expert_load(self):
        pool = PagedTensorPool(sizes())
        timeline = Timeline()
        load = timeline.submit("load", 3, "copy", action=lambda: pool.write(0, 0, "gate_up", b"data"))
        with self.assertRaises(HazardError):
            pool.read(0, 0, "gate_up")
        timeline.run_until(load)
        self.assertEqual(pool.read(0, 0, "gate_up"), b"data")
        with self.assertRaises(HazardError):
            pool.read(0, 1, "gate_up")

    def test_war_detected_until_last_reader_finishes(self):
        pool = PagedTensorPool(sizes())
        timeline = Timeline()
        first = timeline.submit("read", 5, "compute")
        pool.mark_read(0, "gate_up", first)
        with self.assertRaises(HazardError):
            pool.assert_write_safe(2, "gate_up", 4.99)
        pool.assert_write_safe(2, "gate_up", 5)
        # A separate double-buffer slot can be filled concurrently.
        pool.assert_write_safe(1, "gate_up", 0)
        load = timeline.submit("write alias", 1, "copy", [first])
        pool.assert_write_safe(2, "gate_up", load.start_ms)

    def test_last_reader_and_cross_timeline_validation(self):
        pool = PagedTensorPool(sizes())
        timeline = Timeline()
        earlier = timeline.submit("early", 2, "a")
        later = timeline.submit("late", 4, "b")
        pool.mark_read(0, "down", later)
        pool.mark_read(0, "down", earlier)
        self.assertIs(pool.block(2, "down").last_read, later)
        with self.assertRaises(ValueError):
            pool.mark_read(0, "down", Timeline().submit("foreign", 5, "a"))

    def test_validates_shapes_and_write_sizes(self):
        with self.assertRaises(ValueError):
            PagedTensorPool({})
        bad = sizes()
        bad[2]["down"] = [1, 2]
        with self.assertRaises(ValueError):
            PagedTensorPool(bad)
        with self.assertRaises(ValueError):
            PagedTensorPool(sizes(), resident_layers={12})
        pool = PagedTensorPool(sizes())
        with self.assertRaises(ValueError):
            pool.write(0, 0, "down", b"bad size")
        self.assertEqual(len(pool.block(0, "down").data), 5)
        for at in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                pool.assert_write_safe(0, "down", at)


if __name__ == "__main__":
    unittest.main()
