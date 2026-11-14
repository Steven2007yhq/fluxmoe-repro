"""Opt-in physical CUDA VMM alias tests, separate from the CPU event scheduler."""

import os
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fluxmoe import cuda_vmm
from fluxmoe.cuda_vmm import CudaPagedArena, cuda_vmm_available


class CudaVmmInputTests(unittest.TestCase):
    def test_invalid_sizes_fail_before_loading_cuda(self):
        for layers, size, device in ((0, 4, 0), (2, 0, 0), (-1, 4, 0), (2, 4, -1),
                                    (True, 4, 0), (2, 1.5, 0), (2**63, 4, 0)):
            with self.subTest(layers=layers, size=size, device=device), self.assertRaises(ValueError):
                CudaPagedArena(layers, size, device)

    def test_missing_build_metadata_requests_rebuild_without_loading_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "libfluxmoe_vmm.so"
            library.write_bytes(b"not a native library")
            with patch.dict(os.environ, {"FLUXMOE_VMM_LIBRARY": str(library)}):
                with self.assertRaisesRegex(RuntimeError, "metadata missing.*rebuild"):
                    cuda_vmm._library()

    def test_source_change_is_rejected_before_loading_stale_library(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "libfluxmoe_vmm.so"
            content = b"old library fixture"
            library.write_bytes(content)
            library.with_suffix(".json").write_text(json.dumps({
                "returncode": 0, "source_sha256": "old-source-hash",
                "library_sha256": hashlib.sha256(content).hexdigest(),
            }))
            with patch.dict(os.environ, {"FLUXMOE_VMM_LIBRARY": str(library)}):
                with self.assertRaisesRegex(RuntimeError, "source SHA256 differs.*rebuild"):
                    cuda_vmm._library()


@unittest.skipUnless(os.environ.get("FLUXMOE_TEST_CUDA") == "1",
                     "optional real CUDA VMM tests: set FLUXMOE_TEST_CUDA=1")
class CudaVmmHardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        available, reason = cuda_vmm_available()
        if not available:
            raise RuntimeError(f"explicit CUDA VMM validation requested but unavailable: {reason}")

    def test_real_aliases_keep_distinct_stable_virtual_addresses(self):
        size = 1031  # Exercise allocation-granularity padding.
        first = bytes(index % 251 for index in range(size))
        second = bytes((index + 41) % 251 for index in range(size))
        with CudaPagedArena(layers=6, layer_bytes=size) as arena:
            addresses = arena.addresses
            self.assertEqual(len(set(addresses)), 6)
            self.assertTrue(all(address > 0 for address in addresses))
            self.assertEqual(arena.logical_bytes, 6 * size)
            aligned = ((size + arena.granularity - 1) // arena.granularity) * arena.granularity
            self.assertEqual(arena.physical_bytes, 2 * aligned)
            arena.write(0, first)
            self.assertEqual(arena.read(2), first)
            self.assertEqual(arena.read(4), first)
            arena.write(1, second)
            self.assertEqual(arena.read(3), second)
            self.assertEqual(arena.read(5), second)
            self.assertEqual(arena.read(0), first)
            # Demonstrate why scheduling must protect old readers: aliases overwrite.
            arena.write(2, second)
            self.assertEqual(arena.read(0), second)
            self.assertEqual(arena.addresses, addresses)
            self.assertEqual(arena.read(2, 17), second[:17])
            self.assertEqual(arena.read(2, 0), b"")
        self.assertTrue(arena.closed)
        arena.close()  # Idempotent and does not double-release the primary context.
        with self.assertRaises(RuntimeError):
            arena.read(0)

    def test_two_arenas_can_close_independently_and_reopen(self):
        with CudaPagedArena(3, 64) as first:
            first.write(0, b"a" * 64)
            with CudaPagedArena(4, 64) as second:
                second.write(0, b"b" * 64)
                self.assertEqual(first.read(0), b"a" * 64)
            self.assertEqual(first.read(2), b"a" * 64)
        with CudaPagedArena(1, 64) as third:
            third.write(0, b"c" * 64)
            self.assertEqual(third.read(0), b"c" * 64)

    def test_bounds_and_invalid_device_do_not_break_existing_context(self):
        with CudaPagedArena(4, 16) as arena:
            arena.write(0, b"q" * 16)
            for operation in (lambda: arena.read(4), lambda: arena.write(-1, b"q" * 16),
                              lambda: arena.read(0, 17), lambda: arena.write(0, b"short")):
                with self.assertRaises(ValueError):
                    operation()
            with self.assertRaises(RuntimeError):
                CudaPagedArena(4, 16, device=2**31 - 1)
            self.assertEqual(arena.read(2), b"q" * 16)


if __name__ == "__main__":
    unittest.main()
