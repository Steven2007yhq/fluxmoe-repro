"""Threshold, placement and delayed-reclamation checks; no GPU dependencies."""

import math
import unittest

from fluxmoe.planner import (
    Planner,
    ResidencyState,
    TensorKey,
    TensorSpec,
    VersionedBackends,
    initial_bandwidth_placement,
)


def specs(layers=3, experts=4):
    return [TensorSpec(TensorKey(layer, expert, projection), size, size // 2)
            for layer in range(layers) for expert in range(experts)
            for projection, size in (("gate_up", 200), ("down", 100))]


def observe(planner, state, **kwargs):
    defaults = dict(iteration=16, active_batch=2, remaining_batch=8,
                    forward_ms=1.0, load_ms=[1.0, 1.0])
    defaults.update(kwargs)
    return planner.observe(state, **defaults)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = Planner()
        self.state = ResidencyState(specs())

    def apply_and_publish(self, decision, **kwargs):
        self.planner.apply(self.state, decision, **kwargs)
        return self.state.publish_iteration()

    def test_dead_zone_and_boundary_inequalities(self):
        for ratio in (0.9, 0.95, 1.0):
            decision = observe(self.planner, self.state, forward_ms=ratio / 4, load_ms=1)
            self.assertEqual(decision.action, "none")
            self.assertAlmostEqual(decision.rho, ratio)
        decision = observe(self.planner, self.state, forward_ms=1.00001 / 4, load_ms=1)
        self.assertEqual(decision.action, "offload")
        self.apply_and_publish(decision)
        decision = observe(self.planner, self.state, forward_ms=0.89999 / 4, load_ms=1)
        self.assertEqual(decision.action, "restore")

    def test_control_interval_and_completed_decode_profiles_only(self):
        cases = [({"iteration": 0}, "not_control_interval"),
                 ({"iteration": 15}, "not_control_interval"),
                 ({"phase": "prefill"}, "prefill_disabled"),
                 ({"profiles_complete": False}, "incomplete_profiles"),
                 ({"active_batch": 0}, "no_active_sequences"),
                 ({"active_batch": 0, "remaining_batch": 0}, "finished")]
        for kwargs, reason in cases:
            with self.subTest(kwargs=kwargs):
                decision = observe(self.planner, self.state, **kwargs)
                self.assertEqual(decision.action, "none")
                self.assertEqual(decision.reason, reason)
        self.assertEqual(observe(self.planner, self.state, iteration=32).action, "offload")

    def test_zero_load_is_infinite_but_needs_batch_pressure(self):
        self.assertTrue(math.isinf(observe(self.planner, self.state, load_ms=0).rho))
        decision = observe(self.planner, self.state, load_ms=0, remaining_batch=2)
        self.assertEqual(decision.action, "none")

    def test_layer_reclamation_precedes_tensor_offload_and_needs_relaunch(self):
        self.state = ResidencyState(specs(), resident_layers=(0, 2))
        decision = observe(self.planner, self.state)
        self.assertEqual(decision.action, "reclaim_layer")
        self.assertTrue(decision.requires_relaunch)
        self.assertEqual({key.layer for key in decision.keys}, {2})
        with self.assertRaises(RuntimeError):
            self.planner.apply(self.state, decision)
        self.apply_and_publish(decision, between_launches=True)
        self.assertEqual(self.state.resident_layers, {0})
        self.apply_and_publish(observe(self.planner, self.state), between_launches=True)
        self.assertEqual(observe(self.planner, self.state).action, "offload")
        restore = observe(self.planner, self.state, load_ms=100)
        self.assertTrue(restore.requires_relaunch)
        self.apply_and_publish(restore, between_launches=True)
        self.assertEqual(self.state.resident_layers, {0})

    def test_down_first_balanced_ties_and_lifo(self):
        sequence = []
        # AG .25 and 4 experts select one down tensor at a time.
        for _ in range(12):
            decision = observe(self.planner, self.state)
            self.assertEqual(len(decision.keys), 1)
            self.assertEqual(decision.keys[0].projection, "down")
            sequence.append(decision.keys[0])
            self.apply_and_publish(decision)
        self.assertEqual(sequence[:6], [TensorKey(layer, expert, "down")
                                       for expert in (3, 2) for layer in (2, 1, 0)])
        gate = observe(self.planner, self.state)
        self.assertEqual(gate.keys, (TensorKey(2, 3, "gate_up"),))
        self.apply_and_publish(gate)
        restored = observe(self.planner, self.state, load_ms=100)
        self.assertEqual(restored.keys, gate.keys)
        self.apply_and_publish(restored)
        restored = observe(self.planner, self.state, load_ms=100)
        self.assertEqual(restored.keys, (sequence[-1],))

    def test_granularity_preserves_approximate_raw_byte_volume(self):
        state = ResidencyState(specs(layers=1, experts=8))
        planner = Planner(adjustment_granularity=0.5)
        for _ in range(2):
            decision = observe(planner, state)
            self.assertEqual(len(decision.keys), 4)
            planner.apply(state, decision)
            state.publish_iteration()
        decision = observe(planner, state)
        self.assertEqual(len(decision.keys), 2)
        self.assertTrue(all(key.projection == "gate_up" for key in decision.keys))
        self.assertEqual(sum(state.specs[key].raw_bytes for key in decision.keys), 400)

    def test_cpu_budget_uses_raw_bytes_and_blocks_additional_offload(self):
        self.state = ResidencyState(specs(), cpu_budget_bytes=150)
        self.apply_and_publish(observe(self.planner, self.state))
        self.assertEqual(self.state.cpu_bytes, 100)
        decision = observe(self.planner, self.state)
        self.assertEqual(decision.action, "none")
        self.assertEqual(decision.reason, "no_offload_within_budget")
        with self.assertRaises(ValueError):
            ResidencyState(specs(), cpu_keys=(TensorKey(0, 0, "gate_up"),), cpu_budget_bytes=150)

    def test_staging_next_iteration_and_stale_decision(self):
        previous = self.state.snapshot()
        decision = observe(self.planner, self.state)
        self.planner.apply(self.state, decision)
        self.assertIs(self.state.snapshot(), previous)
        self.assertEqual(self.state.cpu_bytes, 0)
        self.assertEqual(observe(self.planner, self.state).reason, "metadata_pending_publication")
        current = self.state.publish_iteration()
        self.assertEqual(current.version, previous.version + 1)
        self.assertEqual(previous.backends[decision.keys[0]], "compressed")
        self.assertEqual(current.backends[decision.keys[0]], "cpu")
        with self.assertRaises(TypeError):
            previous.backends[decision.keys[0]] = "cpu"
        with self.assertRaises(RuntimeError):
            self.planner.apply(self.state, decision)

    def test_partial_layer_residency_rejected_and_initial_floor_explicit(self):
        with self.assertRaises(ValueError):
            self.state.stage({TensorKey(0, 0, "down"): "resident"}, between_launches=True)
        self.assertEqual(observe(self.planner, self.state, load_ms=100).reason,
                         "at_initial_residency_floor")

    def test_invalid_timings_do_not_trigger_a_reclamation(self):
        for value in (-1, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                observe(self.planner, self.state, load_ms=[value])
        with self.assertRaises(ValueError):
            observe(self.planner, self.state, remaining_batch=1)


class VersionedLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.key = TensorKey(0, 0, "down")
        self.metadata = VersionedBackends({self.key: "compressed"})

    def test_all_final_readers_must_complete_before_free(self):
        snapshot = self.metadata.snapshot()
        reader1 = self.metadata.begin_read(snapshot, self.key)
        reader2 = self.metadata.begin_read(snapshot, self.key)
        self.metadata.stage({self.key: "cpu"})
        self.metadata.publish()
        self.assertEqual(len(self.metadata.retired_pending), 1)
        self.assertEqual(self.metadata.drain_reclaimable(), ())
        self.metadata.complete_read(reader1)
        self.assertEqual(self.metadata.drain_reclaimable(), ())
        self.metadata.complete_read(reader2)
        freed = self.metadata.drain_reclaimable()
        self.assertEqual(len(freed), 1)
        self.assertEqual(freed[0].backend, "compressed")
        self.assertEqual(self.metadata.drain_reclaimable(), ())
        self.assertEqual(snapshot.backends[self.key], "compressed")
        with self.assertRaises(RuntimeError):
            self.metadata.begin_read(snapshot, self.key)
        with self.assertRaises(ValueError):
            self.metadata.complete_read(reader2)

    def test_unchanged_copy_reader_spans_multiple_metadata_versions(self):
        other = TensorKey(0, 1, "down")
        meta = VersionedBackends({self.key: "compressed", other: "compressed"})
        read = meta.begin_read(meta.snapshot(), self.key)
        meta.stage({other: "cpu"})
        meta.publish()
        meta.drain_reclaimable()
        meta.stage({self.key: "cpu"})
        meta.publish()
        self.assertEqual(meta.drain_reclaimable(), ())
        meta.complete_read(read)
        self.assertEqual([copy.key for copy in meta.drain_reclaimable()], [self.key])

    def test_restore_uses_new_generation_while_old_copy_waits(self):
        old = self.metadata.begin_read(self.metadata.snapshot(), self.key)
        self.metadata.stage({self.key: "cpu"})
        self.metadata.publish()
        self.metadata.stage({self.key: "compressed"})
        new_snapshot = self.metadata.publish()
        new = self.metadata.begin_read(new_snapshot, self.key)
        self.assertNotEqual(old.generation, new.generation)
        self.assertEqual([copy.backend for copy in self.metadata.drain_reclaimable()], ["cpu"])
        self.metadata.complete_read(old)
        old_copy = self.metadata.drain_reclaimable()
        self.assertEqual(old_copy[0].generation, old.generation)
        self.assertEqual(self.metadata.retired_pending, ())
        self.metadata.complete_read(new)


class BandwidthPlacementTests(unittest.TestCase):
    def test_whole_tensor_round_up_per_layer(self):
        items = specs(layers=2, experts=8)
        keys = initial_bandwidth_placement(items, cpu_bandwidth=32, gpu_bandwidth=400)
        # ceil(3 * 8 * 32/432) = 2 down tensors for every layer.
        self.assertEqual(keys, (TensorKey(1, 7, "down"), TensorKey(0, 7, "down"),
                                TensorKey(1, 6, "down"), TensorKey(0, 6, "down")))
        self.assertEqual(initial_bandwidth_placement(items, 0, 400), ())

    def test_budget_cap_and_outside_paper_rule_domain(self):
        keys = initial_bandwidth_placement(specs(layers=2, experts=8), 32, 400, 250)
        self.assertEqual(len(keys), 2)
        self.assertEqual({key.layer for key in keys}, {0, 1})
        with self.assertRaises(ValueError):
            initial_bandwidth_placement(specs(), 400, 400)


if __name__ == "__main__":
    unittest.main()
