import unittest

from fluxmoe.capacity import GPU_PRESETS, estimate_capacity


class CapacityTests(unittest.TestCase):
    def test_two_a40_mixtral_pageable_weight_floor(self):
        result = estimate_capacity("mixtral", 2, 48)
        self.assertAlmostEqual(result["used_gb_per_gpu"], 31.43)
        self.assertAlmostEqual(result["headroom_gb_per_gpu"], 11.77)
        self.assertTrue(result["fits_static_budget"])
        self.assertAlmostEqual(result["host_expert_gb"], 11.25)
        self.assertEqual(result["host_kv_budget_gb"], 512)

    def test_four_a40_glm_fails_even_without_runtime_allowances(self):
        result = estimate_capacity("glm", 4, 48)
        self.assertFalse(result["fits_static_budget"])
        self.assertLess(result["headroom_gb_per_gpu"], 0)
        self.assertAlmostEqual(result["host_expert_gb"], 84.5)

    def test_paper_eight_h20_resident_glm_weights_fit(self):
        result = estimate_capacity("glm", 8, GPU_PRESETS["h20"]["gpu_gb"], resident=True)
        self.assertTrue(result["fits_static_budget"])
        self.assertAlmostEqual(result["used_gb_per_gpu"], 710 / 8)
        self.assertAlmostEqual(result["budget_gb_per_gpu"], 92.34)
        self.assertAlmostEqual(result["headroom_gb_per_gpu"], 3.59)
        self.assertEqual(result["effective_cpu_fraction"], 0)
        self.assertEqual(result["host_expert_gb"], 0)
        self.assertEqual(result["breakdown"]["double_layer_buffer_gb_per_gpu"], 0)
        self.assertEqual(result["breakdown"]["compressed_expert_gb_per_gpu"], 0)
        # Weight-only fit is not a runnable-deployment guarantee.
        exceeds = estimate_capacity("glm", 8, 102.6, resident=True, kv_gb_per_gpu=4)
        self.assertFalse(exceeds["fits_static_budget"])

    def test_eight_80gb_gpus_do_not_fit_resident_glm(self):
        self.assertFalse(estimate_capacity("glm", 8, 80, resident=True)["fits_static_budget"])

    def test_runtime_allowances_are_added_without_hidden_discount(self):
        floor = estimate_capacity("mixtral", 2, 48)
        deployed = estimate_capacity("mixtral", 2, 48, kv_gb_per_gpu=10, workspace_gb_per_gpu=2)
        self.assertAlmostEqual(deployed["used_gb_per_gpu"] - floor["used_gb_per_gpu"], 12)
        self.assertFalse(deployed["fits_static_budget"])

    def test_full_model_aliases_and_qwen_specific_caveat(self):
        a = estimate_capacity("Mixtral-8×7B-Instruct", 2, 48)
        self.assertEqual(a["model_key"], "mixtral")
        b = estimate_capacity("Qwen3-Next-80B-A3B-Instruct", 4, 48)
        self.assertTrue(any("KV-head replication" in item for item in b["caveats"]))

    def test_rejects_invalid_inputs(self):
        bad_values = [
            {"model": "unknown"}, {"gpus": 0}, {"gpus": -1},
            {"gpus": 2.0}, {"gpus": True}, {"gpu_gb": 0},
            {"gpu_gb": float("inf")}, {"gpu_gb": float("nan")},
            {"gpu_gb": "48"}, {"cpu_fraction": -0.1}, {"cpu_fraction": 1.1},
            {"cpu_fraction": float("nan")}, {"budget_fraction": 0},
            {"budget_fraction": 1.01}, {"kv_gb_per_gpu": -1},
            {"workspace_gb_per_gpu": -1}, {"resident": "yes"},
        ]
        for bad in bad_values:
            arguments = {"model": "mixtral", "gpus": 2, "gpu_gb": 48}
            arguments.update(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                estimate_capacity(**arguments)


if __name__ == "__main__":
    unittest.main()
