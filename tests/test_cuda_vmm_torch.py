"""Real VMM-backed Torch storage and owner lifetimes; GPU tests are opt-in."""
import gc
import importlib.util
import os
import unittest
import weakref

from fluxmoe.cuda_vmm import CudaPagedArena, _view_geometry, cuda_vmm_available


class VmmViewGeometryTests(unittest.TestCase):
    def test_dense_views_scalar_and_offsets(self):
        self.assertEqual(_view_geometry(64, (2, 4), 2, 16), ((2, 4), 16))
        self.assertEqual(_view_geometry(8, (), 8), ((), 8))
        self.assertEqual(_view_geometry(64, [8], 4, 32), ((8,), 32))

    def test_invalid_shapes_alignment_and_bounds(self):
        for shape, itemsize, offset in [
            ((0,), 2, 0), ((-1,), 2, 0), ((True,), 2, 0), ((1.5,), 2, 0),
            (8, 2, 0), ((4,), 2, 1), ((33,), 2, 0), ((1,), 2, 64),
            ((2**63,), 1, 0), ((2**62, 8), 1, 0), ((1,), 2, -1),
        ]:
            with self.subTest(shape=shape, itemsize=itemsize, offset=offset), self.assertRaises(ValueError):
                _view_geometry(64, shape, itemsize, offset)


@unittest.skipUnless(os.environ.get("FLUXMOE_TEST_CUDA") == "1",
                     "set FLUXMOE_TEST_CUDA=1 for VMM Torch view hardware tests")
class CudaVmmTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("real CUDA Torch view tests were requested but torch is not installed")
        import torch
        cls.torch = torch
        supported, reason = cuda_vmm_available()
        if not supported or not torch.cuda.is_available():
            raise RuntimeError(f"real VMM Torch view tests cannot run: {reason}")

    def test_zero_copy_bf16_alias_and_pointer_stability(self):
        torch = self.torch
        arena = CudaPagedArena(4, 64)
        first = arena.tensor(0, (4, 8))
        alias = arena.tensor(2, (4, 8), torch.bfloat16)
        second = arena.tensor(1, (32,), torch.bfloat16)
        try:
            self.assertEqual(first.data_ptr(), arena.addresses[0])
            self.assertEqual(alias.data_ptr(), arena.addresses[2])
            self.assertNotEqual(first.data_ptr(), alias.data_ptr())
            first.copy_(torch.arange(32, dtype=torch.bfloat16, device="cuda").reshape(4, 8))
            second.fill_(-9)
            self.assertTrue(torch.equal(first, alias))
            alias.add_(2)
            self.assertTrue(torch.equal(first, torch.arange(32, dtype=torch.bfloat16, device="cuda").reshape(4, 8) + 2))
            self.assertTrue(torch.equal(second, torch.full_like(second, -9)))
            self.assertEqual(first.data_ptr(), arena.addresses[0])
        finally:
            del first, alias, second
            gc.collect()
            self.assertEqual(arena.live_tensor_leases, 0)
            arena.close()

    def test_slice_detach_and_storage_keep_arena_alive(self):
        torch = self.torch
        arena = CudaPagedArena(3, 128)
        arena_ref = weakref.ref(arena)
        tensor = arena.tensor(0, (8, 8))
        derived = tensor[2:4].detach()
        del tensor
        gc.collect()
        self.assertEqual(arena.live_tensor_leases, 1)
        with self.assertRaisesRegex(RuntimeError, "live Torch storage"):
            arena.close()
        self.assertFalse(arena.closed)
        derived.fill_(3)
        torch.cuda.synchronize()
        del arena
        gc.collect()
        self.assertIsNotNone(arena_ref())
        self.assertTrue(torch.equal(derived, torch.full_like(derived, 3)))
        del derived
        gc.collect()
        self.assertIsNone(arena_ref())

    def test_live_storage_object_survives_tensor_deletion(self):
        arena = CudaPagedArena(3, 64)
        value = arena.tensor(0, (32,))
        storage = value.untyped_storage()
        del value
        gc.collect()
        self.assertEqual(arena.live_tensor_leases, 1)
        with self.assertRaisesRegex(RuntimeError, "live Torch storage"):
            arena.close()
        del storage
        gc.collect()
        self.assertEqual(arena.live_tensor_leases, 0)
        arena.close()

    def test_offset_dtype_and_rejected_views_do_not_leak(self):
        torch = self.torch
        arena = CudaPagedArena(4, 128)
        value = arena.tensor(0, (3, 4), torch.float32, offset_bytes=32)
        try:
            self.assertEqual(value.data_ptr(), arena.addresses[0] + 32)
            value.fill_(1.5)
            self.assertTrue(torch.equal(value, torch.full_like(value, 1.5)))
            for shape, dtype, offset in [((16,), torch.float64, 8), ((1,), torch.complex64, 0),
                                         ((1,), "bfloat16", 0), ((4,), torch.float32, 2)]:
                with self.subTest(shape=shape, dtype=dtype, offset=offset), self.assertRaises(ValueError):
                    arena.tensor(0, shape, dtype, offset_bytes=offset)
            self.assertEqual(arena.live_tensor_leases, 1)
        finally:
            del value
            gc.collect()
            arena.close()

    def test_explicit_stream_events_order_nonblocking_copies_and_compute(self):
        torch = self.torch
        arena = CudaPagedArena(4, 8192)
        weights = arena.tensor(0, (64, 64))
        alias = arena.tensor(2, (64, 64))
        copy_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()
        ready, read_done, rewritten = (torch.cuda.Event() for _ in range(3))
        source = torch.arange(4096, dtype=torch.float32).to(torch.bfloat16).reshape(64, 64).pin_memory()
        next_source = torch.ones_like(source).pin_memory()
        try:
            with torch.cuda.stream(copy_stream):
                weights.copy_(source, non_blocking=True)
                ready.record()
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(ready)
                result = weights.float().sum()
                read_done.record()
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(read_done)
                alias.copy_(next_source, non_blocking=True)
                rewritten.record()
            rewritten.synchronize()
            self.assertEqual(result.item(), source.float().sum().item())
            self.assertTrue(torch.equal(weights, torch.ones_like(weights)))
            self.assertEqual(weights.data_ptr(), arena.addresses[0])
            self.assertEqual(alias.data_ptr(), arena.addresses[2])
        finally:
            copy_stream.synchronize()
            compute_stream.synchronize()
            del weights, alias
            gc.collect()
            arena.close()


if __name__ == "__main__":
    unittest.main()
