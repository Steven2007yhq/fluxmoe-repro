from dataclasses import replace
import unittest
from unittest.mock import patch

from fluxmoe.model import ModelConfig
from fluxmoe.paging import HazardError
from fluxmoe.runtime import (
    BudgetExceeded, FluxMoESimulator, SimulationConfig, STRATEGIES,
    run_simulation,
)


def small_config(**overrides):
    values = {
        "model": ModelConfig(layers=6, experts=4, hidden=4, intermediate=8),
        "batch": 2,
        "iterations": 2,
        "remaining_batch": 16,
        "cpu_fraction": 0.5,
    }
    values.update(overrides)
    return SimulationConfig(**values)


class RuntimeTests(unittest.TestCase):
    def assert_causal_trace(self, simulator, result):
        trace = result["trace"]
        by_id = {event["id"]: event for event in trace}
        loads = {}
        last_reader = {}
        for event in trace:
            self.assertTrue(event["completed"])
            for predecessor in event["dependencies"]:
                self.assertLessEqual(by_id[predecessor]["end_ms"], event["start_ms"])
            metadata = event["metadata"]
            key = (metadata["iteration"], metadata["layer"])
            if metadata["kind"] == "load":
                loads.setdefault(key, []).append(event)
                previous = last_reader.get(metadata["block_id"])
                if previous is not None:
                    # A physical alias may be reused only after its reader.
                    self.assertIn(previous["id"], event["dependencies"])
                    self.assertLessEqual(previous["end_ms"], event["start_ms"])
            elif metadata["kind"] == "compute":
                for load in loads.get(key, ()):
                    self.assertIn(load["id"], event["dependencies"])
                    self.assertLessEqual(load["end_ms"], event["start_ms"])
                for projection in ("gate_up", "down"):
                    block_id = simulator.pool.block(metadata["layer"], projection).id
                    last_reader[block_id] = event

    def test_strategy_prefetch_and_layer_count_matrix_matches_resident_oracle(self):
        # One layer, two slots, first alias reuse, and repeated alias reuse.
        for layers in (1, 2, 3, 6):
            model = ModelConfig(layers=layers, experts=4, hidden=4, intermediate=8)
            reference = run_simulation(small_config(model=model, strategy="resident"))
            for strategy in STRATEGIES:
                for prefetch in (False, True):
                    with self.subTest(layers=layers, strategy=strategy, prefetch=prefetch):
                        simulator = FluxMoESimulator(small_config(
                            model=model, strategy=strategy, prefetch=prefetch,
                        ))
                        result = simulator.run()
                        self.assertEqual(result["status"], "passed")
                        self.assertEqual(result["output_sha256"], reference["output_sha256"])
                        self.assertEqual(result["correctness"]["max_abs_error"], 0)
                        self.assertTrue(result["correctness"]["stable_virtual_identities"])
                        self.assert_causal_trace(simulator, result)

    def test_shared_links_serialize_and_obey_configured_service_time(self):
        config = small_config(strategy="hybrid", iterations=1)
        result = run_simulation(config)
        resources = {}
        for event in result["trace"]:
            if event["resource"] is None:
                continue
            resources.setdefault(event["resource"], []).append(event)
            metadata = event["metadata"]
            if event["resource"] == "h2d_link":
                transfer_ms = metadata["raw_bytes"] / (config.cpu_bandwidth_gbps * 1e6)
            else:
                transfer_ms = max(
                    metadata["raw_bytes"] / (config.decompression_bandwidth_gbps * 1e6),
                    metadata["source_bytes"] / (config.compressed_read_bandwidth_gbps * 1e6),
                )
            self.assertAlmostEqual(event["duration_ms"], transfer_ms + config.launch_ms)
        self.assertEqual(set(resources), {"h2d_link", "decoder"})
        for events in resources.values():
            ordered = sorted(events, key=lambda event: (event["start_ms"], event["id"]))
            for before, after in zip(ordered, ordered[1:]):
                self.assertLessEqual(before["end_ms"], after["start_ms"])
            self.assertLessEqual(
                sum(event["duration_ms"] for event in ordered),
                ordered[-1]["end_ms"] - ordered[0]["start_ms"] + 1e-12,
            )

    def test_layer_materialization_latency_includes_resource_queue(self):
        config = small_config(strategy="compressed", iterations=1, prefetch=True)
        result = run_simulation(config)
        trace = result["trace"]
        loads = [event for event in trace if event["metadata"].get("kind") == "load"]
        compute = {
            event["metadata"]["layer"]: event
            for event in trace if event["metadata"].get("kind") == "compute"
        }
        for layer, measured in enumerate(result["iterations"][0]["layer_materialization_ms"]):
            layer_events = [event for event in loads if event["metadata"]["layer"] == layer]
            issue_ms = 0 if layer < 2 else compute[layer - 2]["end_ms"]
            self.assertAlmostEqual(measured, max(event["end_ms"] for event in layer_events) - issue_ms)
        second = [event for event in loads if event["metadata"]["layer"] == 1]
        queue_delay = min(event["start_ms"] for event in second)
        self.assertGreater(queue_delay, 0)
        active_span = max(event["end_ms"] for event in second) - min(event["start_ms"] for event in second)
        self.assertAlmostEqual(
            result["iterations"][0]["layer_materialization_ms"][1] - active_span,
            queue_delay,
        )

    def test_planner_changes_publish_next_iteration_without_growing_kv(self):
        config = small_config(
            planner_enabled=True, control_interval=1,
            iterations=5, remaining_batch=32,
        )
        result = run_simulation(config)
        reference = run_simulation(replace(config, strategy="resident", planner_enabled=False))
        self.assertEqual(result["output_sha256"], reference["output_sha256"])
        self.assertTrue(result["planner"][0]["applied_for_next_iteration"])
        versions = [record["metadata_version"] for record in result["iterations"]]
        self.assertEqual(versions[0], 0)
        self.assertGreater(versions[-1], 0)
        self.assertEqual(versions, sorted(versions))
        for event in result["trace"]:
            metadata = event["metadata"]
            if "metadata_version" in metadata:
                self.assertEqual(metadata["metadata_version"], versions[metadata["iteration"] - 1])
        memories = result["memory_by_iteration"]
        self.assertTrue(all(memory["kv_fixed_reservation"] == config.kv_bytes for memory in memories))
        self.assertLess(memories[-1]["device_total"], memories[0]["device_total"])
        self.assertGreater(memories[-1]["host_experts"], memories[0]["host_experts"])
        self.assertTrue(result["reclaimed_copies"])
        self.assertFalse(result["planner"][-1]["applied_for_next_iteration"])

    def test_oom_is_preserved_as_failure_without_fabricated_performance(self):
        config = small_config(gpu_budget_bytes=1)
        with self.assertRaises(BudgetExceeded):
            FluxMoESimulator(config)
        result = run_simulation(config)
        self.assertEqual(result["status"], "oom")
        self.assertIsNone(result["metrics"])
        self.assertIn("budget", result["reason"])
        self.assertEqual(result["config"]["gpu_budget_bytes"], 1)

    def test_omitting_war_dependency_is_detected_before_alias_overwrite(self):
        simulator = FluxMoESimulator(small_config(
            strategy="compressed", iterations=1,
            expert_ms_per_token=10,
        ))
        submit = simulator.timeline.submit

        def omit_alias_barrier(*args, **kwargs):
            metadata = kwargs.get("metadata", {})
            if metadata.get("kind") == "load" and metadata.get("layer") == 2:
                kwargs["wait_for"] = ()
            return submit(*args, **kwargs)

        with patch.object(simulator.timeline, "submit", side_effect=omit_alias_barrier):
            with self.assertRaisesRegex(HazardError, "WAR"):
                simulator.run()

    def test_omitting_raw_dependency_is_detected_before_numerical_use(self):
        simulator = FluxMoESimulator(small_config(
            strategy="compressed", iterations=1,
            decompression_bandwidth_gbps=0.00001,
            compressed_read_bandwidth_gbps=0.00001,
            expert_ms_per_token=0.001, nonexpert_ms=0.001,
        ))
        submit = simulator.timeline.submit

        def omit_materialization_barrier(*args, **kwargs):
            if kwargs.get("metadata", {}).get("kind") == "compute":
                kwargs["wait_for"] = tuple(kwargs["wait_for"])[:1]
            return submit(*args, **kwargs)

        with patch.object(simulator.timeline, "submit", side_effect=omit_materialization_barrier):
            with self.assertRaisesRegex(HazardError, "RAW"):
                simulator.run()

    def test_independent_runs_are_deterministic_but_objects_are_single_use(self):
        config = small_config(iterations=1)
        first = run_simulation(config)
        second = run_simulation(config)
        self.assertEqual(first, second)
        simulator = FluxMoESimulator(config)
        simulator.run()
        with self.assertRaisesRegex(RuntimeError, "new simulator"):
            simulator.run()


if __name__ == "__main__":
    unittest.main()
