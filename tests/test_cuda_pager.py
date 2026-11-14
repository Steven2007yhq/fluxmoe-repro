"""Opt-in integration tests for real asynchronous VMM-backed expert paging."""
from __future__ import annotations

import gc
import importlib.util
import os
import unittest
from unittest.mock import patch

from fluxmoe.cuda_pager import CudaMoEPager, PagingProtocolError
from fluxmoe.planner import TensorKey


@unittest.skipUnless(os.environ.get("FLUXMOE_TEST_CUDA") == "1",
                     "set FLUXMOE_TEST_CUDA=1 for real asynchronous GPU paging tests")
class CudaMoEPagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("CUDA pager tests requested but PyTorch is not installed")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA pager tests requested but no CUDA device is available")
        cls.torch = torch

    def weights(self, layers=5, experts=2, hidden=8, intermediate=8):
        """Use integer-generated BF16 storage, including NaNs and signed zero."""
        torch = self.torch
        cursor = 0
        outputs = []
        for shape in ((experts, 2 * intermediate, hidden), (experts, hidden, intermediate)):
            tensors = []
            count = shape[0] * shape[1] * shape[2]
            for _ in range(layers):
                words = torch.arange(cursor, cursor + count, dtype=torch.int64).bitwise_and(65535).to(torch.uint16)
                tensors.append(words.view(torch.bfloat16).reshape(shape))
                cursor += count
            outputs.append(tensors)
        return outputs

    def execute(self, pager, iterations=3, capture=True):
        torch = self.torch
        compute = torch.cuda.Stream()
        captures = []
        # The host submits many iterations without host-side/device-wide sync.
        # Explicit synchronization occurs only after this guarded submit region.
        with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("hot-path device sync")), \
             patch.object(torch.cuda.Stream, "synchronize", side_effect=AssertionError("hot-path stream sync")):
            for iteration in range(iterations):
                pager.begin_iteration()
                for layer in range(pager.layers):
                    with torch.cuda.stream(compute):
                        gate_up, down = pager.acquire(layer, compute)
                        if capture:
                            captures.append((iteration, layer, gate_up.clone().view(torch.uint16), down.clone().view(torch.uint16)))
                        else:
                            # Still submit a real consumer operation.
                            gate_up.clone()
                            down.clone()
                        pager.release(layer, compute)
                    del gate_up, down
                pager.end_iteration()
        pager.synchronize()
        return captures

    def assert_captures(self, captures, weights):
        torch = self.torch
        for iteration, layer, gate_up, down in captures:
            with self.subTest(iteration=iteration, layer=layer):
                self.assertTrue(torch.equal(gate_up.cpu(), weights[0][layer].view(torch.uint16)))
                self.assertTrue(torch.equal(down.cpu(), weights[1][layer].view(torch.uint16)))

    def assert_gpu_trace_dependencies(self, trace):
        # Timing events on different streams share the hardware clock. Allow
        # 3 microseconds for event timestamp quantization/float elapsed-time ULP.
        tolerance_ms = 0.003
        loads = {}
        prior_compute = {}
        for event in trace:
            self.assertGreaterEqual(event["duration_ms"], 0)
            self.assertGreaterEqual(event["end_ms"] + tolerance_ms, event["start_ms"])
            identity = event["iteration"], event["layer"]
            if event["kind"] == "load":
                loads.setdefault(identity, []).append(event)
                previous = prior_compute.get(event["slot"])
                if previous is not None:
                    self.assertTrue(event["war_dependency"])
                    self.assertGreaterEqual(event["start_ms"] + tolerance_ms, previous["end_ms"])
            else:
                self.assertEqual(event["kind"], "compute")
                self.assertEqual(event["raw_dependencies"], len(loads[identity]))
                for load in loads[identity]:
                    self.assertGreaterEqual(event["start_ms"] + tolerance_ms, load["end_ms"])
                prior_compute[event["slot"]] = event

    def test_compressed_cpu_and_mixed_with_and_without_prefetch(self):
        weights = self.weights(layers=5)
        for fraction in (0.0, 0.25, 1.0):
            for prefetch in (False, True):
                with self.subTest(cpu_fraction=fraction, prefetch=prefetch):
                    pager = CudaMoEPager(*weights, cpu_fraction=fraction, prefetch=prefetch)
                    try:
                        addresses = {name: tuple(tensor.data_ptr() for tensor in views)
                                     for name, views in pager._views.items()}
                        captures = self.execute(pager, iterations=3)
                        self.assert_captures(captures, weights)
                        self.assert_gpu_trace_dependencies(pager.trace())
                        self.assertEqual(pager.counters["iterations"], 3)
                        self.assertEqual(addresses, {name: tuple(tensor.data_ptr() for tensor in views)
                                                    for name, views in pager._views.items()})
                        self.assertLessEqual(pager.memory()["pinned_host_expert_bytes"], pager.cpu_budget_bytes)
                        self.assertTrue(all(tensor.is_pinned() for tensor in pager._host.values()))
                        if fraction == 0:
                            self.assertEqual(pager.counters["h2d_copies"], 0)
                            self.assertGreater(pager.counters["decode_launches"], 0)
                        elif fraction == 1:
                            self.assertEqual(pager.counters["decode_launches"], 0)
                            self.assertGreater(pager.counters["h2d_copies"], 0)
                        else:
                            self.assertGreater(pager.counters["decode_launches"], 0)
                            self.assertGreater(pager.counters["h2d_copies"], 0)
                    finally:
                        pager.close()

    def test_all_bf16_bit_patterns_decode_directly_into_vmm(self):
        # Three gate/up layers alone span 49152 words; down layers extend the
        # contiguous pattern sequence beyond all 65536 BF16 bit patterns.
        weights = self.weights(layers=3, experts=8, hidden=32, intermediate=32)
        self.assertGreater(sum(t.numel() for group in weights for t in group), 65536)
        pager = CudaMoEPager(*weights, cpu_fraction=0)
        try:
            captures = self.execute(pager, iterations=2)
            self.assert_captures(captures, weights)
            self.assertEqual(pager.counters["h2d_copies"], 0)
            self.assertEqual(pager.counters["decode_launches"], 3 * 8 * 2 * 2)
        finally:
            pager.close()

    def test_one_layer_many_iterations_retains_war_dependency(self):
        weights = self.weights(layers=1)
        pager = CudaMoEPager(*weights, cpu_fraction=0.5)
        try:
            captures = self.execute(pager, iterations=9)
            self.assert_captures(captures, weights)
            self.assert_gpu_trace_dependencies(pager.trace())
            self.assertGreater(pager.counters["war_waits"], 0)
        finally:
            pager.close()

    def test_out_of_order_and_unreleased_alias_are_rejected(self):
        torch = self.torch
        pager = CudaMoEPager(*self.weights(layers=4), cpu_fraction=0.25)
        stream = torch.cuda.Stream()
        try:
            with self.assertRaises(PagingProtocolError):
                pager.acquire(0, stream)
            pager.begin_iteration()
            with self.assertRaises(PagingProtocolError):
                pager.begin_iteration()
            with self.assertRaises(PagingProtocolError):
                pager.acquire(1, stream)
            with torch.cuda.stream(stream):
                gate_up, down = pager.acquire(0, stream)
                with self.assertRaises(PagingProtocolError):
                    pager.acquire(0, stream)
                with self.assertRaises(PagingProtocolError):
                    pager.prefetch(2)  # physical slot0 is still acquired
                with self.assertRaises(PagingProtocolError):
                    pager.release(1, stream)
                with self.assertRaises(PagingProtocolError):
                    pager.end_iteration()
                gate_up.clone()
                down.clone()
                pager.release(0, stream)
            del gate_up, down
            for layer in range(1, pager.layers):
                with torch.cuda.stream(stream):
                    gate_up, down = pager.acquire(layer, stream)
                    gate_up.clone()
                    down.clone()
                    pager.release(layer, stream)
                del gate_up, down
            pager.end_iteration()
            pager.synchronize()
        finally:
            pager.close()

    def test_external_tensor_blocks_close_then_retry_succeeds(self):
        pager = CudaMoEPager(*self.weights(layers=3), cpu_fraction=0.25)
        external = pager.layer_tensors(0)[0][:, :2].detach()
        with self.assertRaisesRegex(PagingProtocolError, "external owners"):
            pager.close()
        self.assertFalse(pager._closed)
        with self.assertRaises(PagingProtocolError):
            pager.begin_iteration()
        self.assertTrue(any(arena.live_tensor_leases for arena in pager._arenas.values()))
        del external
        gc.collect()
        pager.close()
        self.assertTrue(pager._closed)
        pager.close()

    def test_close_fences_queued_readers_before_freeing_sources(self):
        torch = self.torch
        weights = self.weights(layers=3)
        pager = CudaMoEPager(*weights, cpu_fraction=0)
        stream = torch.cuda.Stream()
        copies = []
        pager.begin_iteration()
        for layer in range(pager.layers):
            with torch.cuda.stream(stream):
                gate_up, down = pager.acquire(layer, stream)
                # Copies are queued consumers, without per-layer host waits.
                copies.append((0, layer, gate_up.clone().view(torch.uint16), down.clone().view(torch.uint16)))
                pager.release(layer, stream)
            del gate_up, down
        pager.end_iteration()
        # close itself must wait for all issued readers before freeing sources.
        pager.close()
        self.assert_captures(copies, weights)
        self.assertTrue(pager._closed)
        self.assertEqual(pager._sources, {})

    def test_explicit_cpu_placement_respects_whole_tensor_budget(self):
        weights = self.weights(layers=3)
        selected = {TensorKey(0, 0, "down"), TensorKey(2, 1, "gate_up")}
        pager = CudaMoEPager(*weights, cpu_fraction=0.5, cpu_keys=selected)
        try:
            self.assertEqual(pager.cpu_keys, selected)
            self.assert_captures(self.execute(pager, iterations=2), weights)
        finally:
            pager.close()
        with self.assertRaisesRegex(ValueError, "budget"):
            CudaMoEPager(*weights, cpu_fraction=0, cpu_keys=selected)

    def test_partial_submission_failure_poisons_pager_but_can_close(self):
        pager = CudaMoEPager(*self.weights(layers=3, experts=2), cpu_fraction=0)
        # The first expert has already enqueued GPU work when expert1 fails.
        source = pager._sources[TensorKey(0, 1, "gate_up")]
        with patch.object(source, "decode_into", side_effect=RuntimeError("injected submit failure")):
            with self.assertRaisesRegex(RuntimeError, "injected submit failure"):
                pager.begin_iteration()
        self.assertTrue(pager._failed)
        self.assertGreater(pager.counters["decode_launches"], 0)
        for operation in (pager.begin_iteration, lambda: pager.acquire(0),
                          lambda: pager.prefetch(0), lambda: pager.layer_tensors(0)):
            with self.assertRaisesRegex(PagingProtocolError, "failed during submission"):
                operation()
        pager.close()
        self.assertTrue(pager._closed)
        self.assertTrue(source.closed)

    def test_validation_error_during_close_still_releases_allocations(self):
        pager = CudaMoEPager(*self.weights(layers=3), cpu_fraction=0.25)
        self.execute(pager, iterations=2)
        sources = list(pager._sources.values())
        arenas = list(pager._arenas.values())
        with patch.object(sources[0], "check", side_effect=RuntimeError("injected validation failure")):
            with self.assertRaisesRegex(RuntimeError, "cleanup completed with errors.*injected validation failure"):
                pager.close()
        self.assertTrue(pager._closed)
        self.assertEqual(pager._sources, {})
        self.assertEqual(pager._arenas, {})
        self.assertTrue(all(source.closed for source in sources))
        self.assertTrue(all(arena.closed for arena in arenas))
        pager.close()  # Cleanup remains idempotent after reporting the error.

    def test_trace_retention_is_bounded_without_changing_outputs(self):
        weights = self.weights(layers=3)
        for limit in (0, 3):
            with self.subTest(max_trace_events=limit):
                pager = CudaMoEPager(*weights, cpu_fraction=0.25, max_trace_events=limit)
                try:
                    self.assert_captures(self.execute(pager, iterations=4), weights)
                    trace = pager.trace()
                    self.assertEqual(len(trace), limit)
                    self.assertEqual(len(pager._records), limit)
                    self.assertGreater(pager.counters["trace_events_dropped"], 0)
                    self.assertEqual(pager.counters["iterations"], 4)
                finally:
                    pager.close()


if __name__ == "__main__":
    unittest.main()
