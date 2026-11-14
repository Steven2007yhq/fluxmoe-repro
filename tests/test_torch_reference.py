"""Optional independent matrix oracle; CPU-only even on a CUDA-equipped host."""

import importlib.util
import math
import struct
import unittest

from fluxmoe.codec import floats_to_bf16
from fluxmoe.model import ModelConfig, ToyMoE
from fluxmoe.planner import TensorKey
from fluxmoe.torch_reference import (
    bf16_bytes_to_tensor,
    tensor_to_bf16_bytes,
    torch_forward,
    verify_torch_reference,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional PyTorch reference: torch is not installed")
class TorchReferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = ToyMoE(ModelConfig(layers=4, experts=5, hidden=7, intermediate=11, top_k=3, seed=412))

    def test_multiple_layers_top_k_and_random_inputs(self):
        result = verify_torch_reference(self.model, batch=13)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["dtype"], "torch.float64")
        self.assertLessEqual(result["max_abs_error"], result["tolerance"])

    def test_hot_experts_and_tied_router_scores(self):
        # All positive inputs route to experts 0 and 1, leaving others empty.
        model = ToyMoE(ModelConfig(layers=3, experts=6, hidden=5, intermediate=9, top_k=2, seed=81))
        for layer in range(model.config.layers):
            model.routers[layer] = [[10.0] * 5, [10.0] * 5] + [[-10.0] * 5 for _ in range(4)]
        inputs = [[1.0 + i / 10] * 5 for i in range(6)]
        expected = model.forward(inputs)
        actual = torch_forward(model, inputs)
        error = max(abs(a - b) for ar, br in zip(actual, expected) for a, b in zip(ar, br))
        self.assertLess(error, 1e-10)
        _, counts = model.forward_layer(0, inputs, model.reader)
        self.assertEqual(counts, [6, 6, 0, 0, 0, 0])

    def test_tied_top_one_chooses_lower_index_expert(self):
        model = ToyMoE(ModelConfig(layers=1, experts=2, hidden=1, intermediate=1, top_k=1))
        model.routers[0] = [[0.0], [0.0]]
        model.weights[TensorKey(0, 0, "gate_up")] = floats_to_bf16([1.0, 2.0])
        model.weights[TensorKey(0, 0, "down")] = floats_to_bf16([3.0])
        model.weights[TensorKey(0, 1, "gate_up")] = floats_to_bf16([10.0, 20.0])
        model.weights[TensorKey(0, 1, "down")] = floats_to_bf16([30.0])
        actual = torch_forward(model, [[1.0]])[0][0]
        self.assertAlmostEqual(actual, 1 + 6 / (1 + math.exp(-1)), places=13)

    def test_empty_batch_and_invalid_inputs(self):
        self.assertEqual(torch_forward(self.model, []), [])
        report = verify_torch_reference(self.model, batch=0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["max_abs_error"], 0)
        for inputs in ([[1]], [[float("nan")] * 7], [[float("inf")] * 7]):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                torch_forward(self.model, inputs)
        with self.assertRaises(ValueError):
            torch_forward(self.model, [], device="not-a-device")

    def test_all_bf16_words_exact_bit_round_trip(self):
        raw = struct.pack("<65536H", *range(65536))
        tensor = bf16_bytes_to_tensor(raw)
        self.assertEqual(tensor_to_bf16_bytes(tensor), raw)
        for raw in self.model.weights.values():
            self.assertEqual(tensor_to_bf16_bytes(bf16_bytes_to_tensor(raw)), raw)
        with self.assertRaises(ValueError):
            bf16_bytes_to_tensor(b"\x00")

    def test_cuda_request_without_device_does_not_fall_back(self):
        # Simulate unavailability without touching the actual GPU.
        from unittest.mock import patch
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no usable CUDA device"):
                torch_forward(self.model, self.model.inputs(1), device="cuda")


if __name__ == "__main__":
    unittest.main()
