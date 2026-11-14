"""CPU validation tests plus opt-in real installed-vLLM runner integration."""
import importlib.util
import os
from types import SimpleNamespace
import unittest

from fluxmoe.integrations.vllm import inspect_runner, validate_weight_layout


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency")
class VllmLayoutTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.w13 = torch.empty((4, 32, 8), dtype=torch.bfloat16)
        self.w2 = torch.empty((4, 8, 16), dtype=torch.bfloat16)

    def runner(self):
        method_class = type("UnquantizedFusedMoEMethod", (), {})
        method = method_class()
        method.unquantized_backend = SimpleNamespace(name="TRITON")
        method.moe_kernel = object()
        routed = SimpleNamespace(quant_method=method, quant_config=None,
                                 activation=SimpleNamespace(value="silu"),
                                 w13_weight=self.w13, w2_weight=self.w2)
        parallel = SimpleNamespace(tp_size=1, dp_size=1, ep_size=1, pcp_size=1, sp_size=1,
                                   use_ep=False, enable_eplb=False, is_sequence_parallel=False,
                                   use_all2all_kernels=False)
        return SimpleNamespace(routed_experts=routed, moe_config=SimpleNamespace(moe_parallel_config=parallel),
                               layer_name="test")

    def test_bf16_layout_and_cuda_requirement(self):
        self.assertEqual(validate_weight_layout(self.w13, self.w2), (4, 8, 16))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            validate_weight_layout(self.w13, self.w2, require_cuda=True)
        with self.assertRaisesRegex(ValueError, "BF16"):
            validate_weight_layout(self.w13.float(), self.w2.float())
        with self.assertRaisesRegex(ValueError, "expected"):
            validate_weight_layout(self.w13, self.w2[..., :-1])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_weight_layout(self.w13.transpose(1, 2).contiguous().transpose(1, 2), self.w2)

    def test_runner_must_be_postprocessed_triton(self):
        runner = self.runner()
        self.assertEqual(inspect_runner(runner)["backend"], "TRITON")
        runner.routed_experts.quant_method.moe_kernel = None
        with self.assertRaisesRegex(ValueError, "process_weights_after_loading"):
            inspect_runner(runner)
        runner.routed_experts.quant_method.moe_kernel = object()
        runner.routed_experts.quant_method.unquantized_backend.name = "FLASHINFER_CUTLASS"
        with self.assertRaisesRegex(ValueError, "TRITON"):
            inspect_runner(runner)

    def test_rejects_parallel_eplb_quantized_and_overlap_modes(self):
        for field, value in (("tp_size", 2), ("ep_size", 2), ("enable_eplb", True), ("use_all2all_kernels", True)):
            runner = self.runner()
            setattr(runner.moe_config.moe_parallel_config, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                inspect_runner(runner)
        runner = self.runner()
        runner.routed_experts.quant_config = object()
        with self.assertRaisesRegex(ValueError, "quantized"):
            inspect_runner(runner)
        runner = self.runner()
        runner._shared_experts = object()
        with self.assertRaisesRegex(ValueError, "shared"):
            inspect_runner(runner)


@unittest.skipUnless(os.environ.get("FLUXMOE_TEST_VLLM") == "1", "set FLUXMOE_TEST_VLLM=1 in vllm_env for real runner test")
class VllmRealRunnerTests(unittest.TestCase):
    def test_actual_vllm_runner_uses_vmm_with_identical_bf16_outputs(self):
        from fluxmoe.integrations.vllm import run_vllm_smoke
        result = run_vllm_smoke(layers=4, iterations=2, batch=8)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["stable_vmm_pointers"])
        self.assertTrue(all(row["bitwise_equal"] for row in result["comparisons"]))
        self.assertIn("MoERunner", result["binding"]["layers"][0]["runner_type"])


if __name__ == "__main__":
    unittest.main()
