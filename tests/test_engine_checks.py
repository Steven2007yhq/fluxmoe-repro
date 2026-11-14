"""CPU rejection tests for engine reports and independent KV observation."""
import copy
import importlib.util
from types import SimpleNamespace
import unittest

from fluxmoe.engine_checks import compare_results, serialize_request_output


def run_fixture():
    return {
        "checkpoint_id": "sha256-fixture",
        "engine_config": {"kv_cache_memory_bytes": 1048576, "dtype": "bfloat16"},
        "scenarios": [{"name": "mixed", "sampling": {"temperature": 0, "max_tokens": 2,
            "ignore_eos": True, "prompt_logprobs": 5}, "requests": [{
            "request_id": "0", "prompt_token_ids": [1, 2, 3], "token_ids": [7, 8],
            "finished": True, "logprobs": [-0.1, -0.2],
            "prompt_logprobs": [None, -0.5, -0.3], "finish_reason": "length",
            "stop_reason": None, "num_cached_tokens": 0,
        }]}],
        "kv_evidence": {"allocated_bytes": 1048576, "num_blocks": 64,
            "prefill_forwards": 1, "decode_forwards": 1,
            "cache_write_checks": 4, "cache_write_failures": 0,
            "history_read_checks": 1, "history_read_failures": 0},
    }


class EngineChecksTests(unittest.TestCase):
    def test_exact_and_tolerated_probability_comparisons(self):
        reference = run_fixture()
        paged = copy.deepcopy(reference)
        paged["scenarios"][0]["requests"][0]["request_id"] = "other-process-99"
        paged["scenarios"][0]["requests"][0]["logprobs"][1] += 1e-6
        result = compare_results(reference, paged)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["compared_generated_tokens"], 2)
        self.assertEqual(result["compared_prompt_logprobs"], 2)
        self.assertGreater(result["max_logprob_abs_error"], 0)

    def test_token_stopping_and_request_assignment_must_match(self):
        for field, value in (("token_ids", [7, 9]), ("prompt_token_ids", [1, 4, 3]),
                             ("finish_reason", "stop"), ("stop_reason", 8), ("finished", False)):
            with self.subTest(field=field):
                reference, paged = run_fixture(), run_fixture()
                paged["scenarios"][0]["requests"][0][field] = value
                self.assertEqual(compare_results(reference, paged)["status"], "failed")

    def test_nonfinite_missing_or_inaccurate_logprobs_are_not_accepted(self):
        for value in (float("nan"), float("inf"), None, -0.7):
            with self.subTest(value=value):
                reference, paged = run_fixture(), run_fixture()
                paged["scenarios"][0]["requests"][0]["logprobs"][0] = value
                self.assertEqual(compare_results(reference, paged)["status"], "failed")
        reference, paged = run_fixture(), run_fixture()
        paged["scenarios"][0]["requests"][0]["logprobs"] = []
        self.assertEqual(compare_results(reference, paged)["status"], "failed")

    def test_empty_or_incomplete_experiments_fail(self):
        for field in ("checkpoint_id", "engine_config", "scenarios", "kv_evidence"):
            with self.subTest(field=field):
                reference, paged = run_fixture(), run_fixture()
                del paged[field]
                self.assertEqual(compare_results(reference, paged)["status"], "failed")
        self.assertEqual(compare_results({}, {})["status"], "failed")

    def test_same_outputs_without_numerical_kv_evidence_fail(self):
        for field, value in (("history_read_checks", 0), ("cache_write_failures", 1),
                             ("cache_write_checks", True), ("decode_forwards", 0)):
            with self.subTest(field=field):
                reference, paged = run_fixture(), run_fixture()
                paged["kv_evidence"][field] = value
                result = compare_results(reference, paged)
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["kv_evidence_valid"])

    def test_checkpoint_budget_and_order_mismatches_fail(self):
        for field, value in (("checkpoint_id", "different"),
                             ("engine_config", {"kv_cache_memory_bytes": 2})):
            reference, paged = run_fixture(), run_fixture()
            paged[field] = value
            self.assertEqual(compare_results(reference, paged)["status"], "failed")
        reference, paged = run_fixture(), run_fixture()
        paged["kv_evidence"]["num_blocks"] = 128
        self.assertEqual(compare_results(reference, paged)["status"], "failed")
        reference = run_fixture()
        second = copy.deepcopy(reference["scenarios"][0])
        second["name"] = "prefix_repeat"
        reference["scenarios"].append(second)
        paged = copy.deepcopy(reference)
        paged["scenarios"].reverse()
        self.assertEqual(compare_results(reference, paged)["status"], "failed")

    def test_prompt_logprobs_optional_per_scenario_but_required_somewhere(self):
        reference, paged = run_fixture(), run_fixture()
        for run in (reference, paged):
            run["scenarios"][0]["requests"][0]["prompt_logprobs"] = None
            run["scenarios"][0]["sampling"]["prompt_logprobs"] = None
        self.assertEqual(compare_results(reference, paged)["status"], "failed")
        self.assertEqual(compare_results(reference, paged, require_prompt_logprobs=False)["status"], "passed")

    def test_same_truncated_outputs_do_not_pass(self):
        reference, paged = run_fixture(), run_fixture()
        for run in (reference, paged):
            request = run["scenarios"][0]["requests"][0]
            request["token_ids"] = request["token_ids"][:1]
            request["logprobs"] = request["logprobs"][:1]
        result = compare_results(reference, paged)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("did not reach max_tokens" in error for error in result["errors"]))

    def test_stopping_rules_and_invalid_generation_budgets_fail(self):
        for budget in (None, 0, -1, True, 2.0, "2"):
            reference, paged = run_fixture(), run_fixture()
            for run in (reference, paged):
                run["scenarios"][0]["sampling"]["max_tokens"] = budget
            with self.subTest(budget=budget):
                self.assertEqual(compare_results(reference, paged)["status"], "failed")
        reference, paged = run_fixture(), run_fixture()
        for run in (reference, paged):
            run["scenarios"][0]["requests"][0]["finish_reason"] = "stop"
        self.assertEqual(compare_results(reference, paged)["status"], "failed")
        for run in (reference, paged):
            run["scenarios"][0]["sampling"]["ignore_eos"] = False
        self.assertEqual(compare_results(reference, paged)["status"], "passed")

    def test_requested_prompt_logprobs_cannot_be_missing_from_one_scenario(self):
        reference, paged = run_fixture(), run_fixture()
        for run in (reference, paged):
            second = copy.deepcopy(run["scenarios"][0])
            second["name"] = "requested_but_missing"
            second["sampling"]["prompt_logprobs"] = 0  # 0 still requests chosen-token probabilities.
            second["requests"][0]["prompt_logprobs"] = None
            run["scenarios"].append(second)
        result = compare_results(reference, paged)
        self.assertEqual(result["status"], "failed")
        self.assertGreater(result["compared_prompt_logprobs"], 0)
        self.assertTrue(any("requested prompt_logprobs are missing" in error for error in result["errors"]))

    def test_serialize_extracts_chosen_token_not_top_rank(self):
        logprob = lambda value: SimpleNamespace(logprob=value)
        output = SimpleNamespace(request_id="7", prompt_token_ids=[1, 2],
            prompt_logprobs=[None, {2: logprob(-3), 9: logprob(-1)}], finished=True,
            num_cached_tokens=0, outputs=[SimpleNamespace(text="ok", token_ids=[4],
                logprobs=[{9: logprob(-1), 4: logprob(-2)}], finish_reason="length", stop_reason=None)])
        result = serialize_request_output(output)
        self.assertEqual(result["logprobs"], [-2.0])
        self.assertEqual(result["prompt_logprobs"], [None, -3.0])
        del output.outputs[0].logprobs[0][4]
        with self.assertRaisesRegex(ValueError, "chosen token"):
            serialize_request_output(output)

    def test_invalid_tolerance_rejected(self):
        for value in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                compare_results(run_fixture(), run_fixture(), atol=value)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency")
class KVObserverTests(unittest.TestCase):
    """Exercise real CPU tensor indexing and SDPA without importing vLLM/GPU."""

    def setUp(self):
        import torch
        from fluxmoe.kv_observer import KVObserver
        self.torch = torch
        layer = torch.nn.Module()
        layer.impl = type("TritonAttentionImpl", (), {})()
        for key, value in dict(alibi_slopes=None, sliding_window=(-1, -1),
                               logits_soft_cap=0, scale=2 ** -0.5).items():
            setattr(layer.impl, key, value)
        layer.kv_cache = torch.zeros((4, 2, 4, 4), dtype=torch.bfloat16)
        layer.layer_name = "fixture.attn"
        layer.num_heads, layer.num_kv_heads = 4, 2
        layer.head_size = layer.head_size_v = 2
        self.layer = layer
        self.observer = KVObserver(layer)
        self.addCleanup(self.observer.remove)

    def step(self, slots, *, seq_len, values, output_value=0, corrupt_write=False):
        torch = self.torch
        count = len(slots)
        key = torch.zeros((count, 2, 2), dtype=torch.bfloat16)
        value = torch.tensor(values, dtype=torch.bfloat16)[:, None, None].expand(count, 2, 2).clone()
        for index, slot in enumerate(slots):
            self.layer.kv_cache[slot // 4, :, slot % 4, :] = torch.cat((key[index], value[index]), dim=-1)
        if corrupt_write:
            self.layer.kv_cache[slots[0] // 4, :, slots[0] % 4, 0] = 1
        self.observer.pending = {"query": torch.zeros((count, 4, 2), dtype=torch.bfloat16),
            "key": key, "value": value, "slots": slots, "query_starts": [0, count],
            "seq_lens": [seq_len], "block_table": [[2, 1]]}
        output = torch.full((count, 8), output_value, dtype=torch.bfloat16)
        self.observer.after(self.layer, (), {}, output)

    def test_noncontiguous_blocks_gqa_and_history_reference(self):
        self.step([8, 9, 10, 11], seq_len=4, values=[1, 2, 3, 4])
        self.step([4], seq_len=5, values=[5], output_value=3)
        data = self.observer.data
        self.assertEqual(data["cache_write_checks"], 5)
        self.assertEqual(data["cache_write_failures"], 0)
        self.assertEqual(data["history_read_checks"], 1)
        self.assertEqual(data["history_read_failures"], 0)
        self.assertEqual(data["steps"][1]["active_block_ids"], [[2, 1]])
        self.assertEqual(data["max_attention_abs_error"], 0)

    def test_current_token_only_attention_fails_history_comparison(self):
        self.step([8, 9, 10, 11], seq_len=4, values=[1, 2, 3, 4])
        self.step([4], seq_len=5, values=[5], output_value=5)
        self.assertEqual(self.observer.data["history_read_failures"], 1)
        self.assertEqual(self.observer.data["max_attention_abs_error"], 2)

    def test_write_corruption_and_unobserved_history_fail(self):
        self.step([8], seq_len=1, values=[1], corrupt_write=True)
        self.assertEqual(self.observer.data["cache_write_failures"], 1)
        self.step([4], seq_len=5, values=[5], output_value=1.2)
        self.assertGreater(self.observer.data["history_missing_slots"], 0)
        self.assertEqual(self.observer.data["history_read_failures"], 1)


if __name__ == "__main__":
    unittest.main()
