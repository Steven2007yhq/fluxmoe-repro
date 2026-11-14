"""Configuration and root-hook lifecycle tests; real engines run separately."""
from contextlib import contextmanager
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fluxmoe.integrations.engine import (
    EnginePagingOptions, FluxMoEModelController, validate_engine_config,
)


class EngineOptionsTests(unittest.TestCase):
    def test_options_reject_silent_fallback_and_invalid_budgets(self):
        self.assertEqual(EnginePagingOptions.from_dict({}).checkpoint_format, "auto")
        for values in ({"cpu_fraction": float("nan")}, {"cpu_fraction": -0.1},
                       {"cpu_fraction": True}, {"cpu_fraction": "0.1"},
                       {"prefetch": 1}, {"max_trace_events": -1},
                       {"max_trace_events": True}, {"checkpoint_format": "dummy"},
                       {"ignored_typo": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                EnginePagingOptions.from_dict(values)

    def config(self):
        return SimpleNamespace(
            parallel_config=SimpleNamespace(ubatch_size=0),
            compilation_config=SimpleNamespace(mode=0, cudagraph_mode=0),
            kernel_config=SimpleNamespace(moe_backend="triton")), SimpleNamespace(
                enforce_eager=True, quantization=None,
                hf_config=SimpleNamespace(architectures=["MixtralForCausalLM"]))

    def test_config_accepts_only_supported_eager_single_rank_model(self):
        config, model = self.config()
        validate_engine_config(config, model)
        for owner, field, value in (("model", "enforce_eager", False),
                                     ("model", "quantization", "awq"),
                                     ("model", "dtype", "torch.float16"),
                                     ("model", "enable_sleep_mode", True),
                                     ("parallel", "tensor_parallel_size", 2),
                                     ("parallel", "pipeline_parallel_size", 2),
                                     ("parallel", "ubatch_size", 2),
                                     ("parallel", "enable_expert_parallel", True),
                                     ("compilation", "mode", 3),
                                     ("compilation", "cudagraph_mode", 1),
                                     ("config", "speculative_config", object())):
            config, model = self.config()
            target = {"config": config, "model": model, "parallel": config.parallel_config,
                      "compilation": config.compilation_config}[owner]
            setattr(target, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_engine_config(config, model)

    def test_rejects_storage_changing_or_unverified_engine_modes(self):
        overrides = [
            ("scheduler_config", SimpleNamespace(async_scheduling=True)),
            ("cache_config", SimpleNamespace(cache_dtype="fp8")),
            ("cache_config", SimpleNamespace(kv_offloading_size=1)),
            ("offload_config", SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=1))),
            ("offload_config", SimpleNamespace(prefetch=SimpleNamespace(offload_group_size=2))),
            ("load_config", SimpleNamespace(device="cpu")),
            ("kv_transfer_config", object()), ("weight_transfer_config", object()),
        ]
        for field, value in overrides:
            config, model = self.config()
            setattr(config, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_engine_config(config, model)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency")
class EngineRootHookTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        events = self.events = []

        class Model(torch.nn.Module):
            fail = False

            def forward(self, input_ids):
                events.append("model")
                if self.fail:
                    raise ArithmeticError("original forward failure")
                return input_ids + 1

        class Adapter:
            @contextmanager
            def iteration(self):
                events.append("begin")
                try:
                    yield
                except BaseException:
                    events.append("abort")
                    raise
                else:
                    events.append("end")

            def detach(self, *, restore_to_device):
                self.restore_to_device = restore_to_device
                events.append("detach")

        class Pager:
            def close(self):
                events.append("close")

        self.model, self.adapter, self.pager = Model(), Adapter(), Pager()
        self.observer = patch("fluxmoe.integrations.engine._forward_observation", return_value={
            "tokens": 3, "phase": "single_token_query", "max_query_len": 1})
        self.observer.start()
        self.addCleanup(self.observer.stop)

    def test_every_root_forward_uses_one_complete_iteration(self):
        controller = FluxMoEModelController(self.model, adapter=self.adapter, pager=self.pager)
        x = self.torch.tensor([1, 2, 3])
        self.assertTrue(self.torch.equal(self.model(input_ids=x), x + 1))
        self.assertTrue(self.torch.equal(self.model(x), x + 1))
        self.assertEqual(self.events, ["begin", "model", "end"] * 2)
        self.assertEqual(controller.forward_counts["completed"], 2)
        controller.close()
        controller.close()
        self.assertEqual(self.events[-2:], ["detach", "close"])
        self.assertFalse(self.adapter.restore_to_device)

    def test_failure_aborts_iteration_and_preserves_original_error(self):
        controller = FluxMoEModelController(self.model, adapter=self.adapter, pager=self.pager)
        self.model.fail = True
        with self.assertRaisesRegex(ArithmeticError, "original forward failure"):
            self.model(self.torch.tensor([1]))
        self.assertEqual(self.events, ["begin", "model", "abort"])
        self.assertFalse(controller._executing)
        self.assertIsNone(controller._iteration)
        self.assertEqual(controller.forward_counts["failed"], 1)
        controller.close()

    def test_resident_observer_preserves_results_and_bounds_history(self):
        controller = FluxMoEModelController(self.model, max_observations=1)
        x = self.torch.tensor([1])
        for _ in range(3):
            self.assertTrue(self.torch.equal(self.model(x), x + 1))
        self.assertEqual(controller.forward_counts["completed"], 3)
        self.assertEqual(len(controller.observations), 1)
        self.assertEqual(controller.status()["mode"], "resident")
        controller.close()
        self.assertTrue(self.torch.equal(self.model(x), x + 1))

    def test_failed_begin_does_not_leave_root_active(self):
        class FailingAdapter:
            @contextmanager
            def iteration(self):
                raise RuntimeError("pager begin failure")
                yield

        controller = FluxMoEModelController(self.model, adapter=FailingAdapter(), pager=self.pager)
        with self.assertRaisesRegex(RuntimeError, "pager begin failure"):
            self.model(self.torch.tensor([1]))
        self.assertFalse(controller._executing)
        self.assertIsNone(controller._iteration)
        self.assertEqual(self.events, [])

    def test_partial_hook_install_is_rolled_back(self):
        with patch.object(self.model, "register_forward_hook", side_effect=RuntimeError("hook failure")):
            with self.assertRaisesRegex(RuntimeError, "hook failure"):
                FluxMoEModelController(self.model)
        self.assertEqual(len(self.model._forward_pre_hooks), 0)
        self.assertEqual(len(self.model._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
