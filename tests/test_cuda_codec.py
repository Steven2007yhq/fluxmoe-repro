"""CPU index tests plus explicitly opted-in real CUDA correctness tests.

Run GPU tests with FLUXMOE_TEST_CUDA=1. No test installs packages.
"""
import os
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import random
import struct
import unittest

from fluxmoe.codec import compress_bf16, floats_to_bf16
from fluxmoe.cuda_codec import (CudaCodecError, _prepare, _validate_library_provenance,
                                available, decode_cuda, decode_cuda_with_metrics)


class CudaIndexTests(unittest.TestCase):
    def test_chunk_offsets_cover_exact_exponent_bits(self):
        raw = floats_to_bf16([0, 1, 2, 4, 8, -1, 0.5] * 73)
        encoded = compress_bf16(raw)
        for chunk_size in [1, 7, 256, 65536]:
            result = _prepare(encoded, chunk_size)
            self.assertEqual(result["reference"], raw)
            self.assertEqual(result["offsets"][0], 0)
            self.assertEqual(result["offsets"][-1], result["bit_count"])
            self.assertEqual(len(result["offsets"]), (len(raw) // 2 + chunk_size - 1) // chunk_size + 1)
            self.assertTrue(all(a < b for a, b in zip(result["offsets"], result["offsets"][1:])))

    def test_empty_does_not_launch_kernel(self):
        output, metrics = decode_cuda_with_metrics(compress_bf16(b""))
        self.assertEqual(output, b"")
        self.assertFalse(metrics["kernel_launched"])

    def test_corrupt_input_rejected_before_cuda(self):
        with self.assertRaises(ValueError):
            decode_cuda(b"invalid")
        for size in [0, -1, 65537, 1.5]:
            with self.assertRaises(ValueError):
                _prepare(compress_bf16(b"\0\0"), size)


class CudaProvenanceTests(unittest.TestCase):
    def test_source_and_artifact_hashes_are_checked_without_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "native" / "huffman_decode.cu"
            source.parent.mkdir()
            source.write_bytes(b"reference source")
            library = root / "decoder.so"
            library.write_bytes(b"reference artifact")
            metadata = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "artifact_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}
            library.with_suffix(".json").write_text(json.dumps(metadata))
            with patch("fluxmoe.cuda_codec._ROOT", root):
                found, digest = _validate_library_provenance(library, required=True)
                self.assertEqual(found, metadata)
                self.assertEqual(digest, metadata["artifact_sha256"])
                source.write_bytes(b"edited source")
                with self.assertRaisesRegex(CudaCodecError, "source changed"):
                    _validate_library_provenance(library, required=True)
                source.write_bytes(b"reference source")
                library.write_bytes(b"tampered artifact")
                with self.assertRaisesRegex(CudaCodecError, "artifact SHA256 mismatch"):
                    _validate_library_provenance(library, required=True)

    def test_default_requires_metadata_and_custom_legacy_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "decoder.so"
            library.write_bytes(b"legacy artifact")
            with self.assertRaisesRegex(CudaCodecError, "lacks source/artifact provenance"):
                _validate_library_provenance(library, required=True)
            metadata, digest = _validate_library_provenance(library, required=False)
            self.assertEqual(metadata, {})
            self.assertEqual(digest, hashlib.sha256(library.read_bytes()).hexdigest())
            library.with_suffix(".json").write_text("[]")
            with self.assertRaisesRegex(CudaCodecError, "JSON object"):
                _validate_library_provenance(library, required=False)


_GPU_OPT_IN = os.environ.get("FLUXMOE_TEST_CUDA") == "1"


@unittest.skipUnless(_GPU_OPT_IN, "Set FLUXMOE_TEST_CUDA=1 for bounded real-GPU tests")
class CudaDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not available():
            raise RuntimeError("CUDA tests explicitly requested but CUDA driver/device is unavailable")

    def test_all_65536_bf16_patterns_bitwise(self):
        raw = b"".join(struct.pack("<H", word) for word in range(65536))
        actual, metrics = decode_cuda_with_metrics(compress_bf16(raw))
        self.assertEqual(actual, raw)
        self.assertTrue(metrics["kernel_launched"])
        self.assertTrue(metrics["bitwise_matches_cpu"])
        self.assertGreaterEqual(metrics["kernel_ms"], 0)
        self.assertIn("SM=", metrics["device"])

    def test_singleton_partial_chunk_and_signed_specials(self):
        for raw in [b"\x81\x7f", b"\0\x80" * 257,
                    b"\0\0\0\x80\x80\x7f\x80\xff\xff\x7f\xff\xff"]:
            self.assertEqual(decode_cuda(compress_bf16(raw)), raw)

    def test_long_codes_and_chunk_boundaries(self):
        frequencies = [1, 1]
        for _ in range(12):
            frequencies.append(frequencies[-1] + frequencies[-2])
        raw = b"".join(struct.pack("<H", exponent << 7)
                       for exponent, count in enumerate(frequencies) for _ in range(count))
        encoded = compress_bf16(raw)
        self.assertGreater(max(encoded[32:288]), 8)
        for size in [1, 31, 256, 65536]:
            actual, _ = decode_cuda_with_metrics(encoded, chunk_size=size)
            self.assertEqual(actual, raw)

    def test_nonuniform_512kib_input(self):
        rng = random.Random(260402715)
        raw = b"".join(struct.pack("<H", (rng.randrange(2) << 15) |
                                   (rng.choice([124, 125, 126, 126, 126, 127, 128]) << 7) |
                                   rng.randrange(128)) for _ in range(256 * 1024))
        encoded = compress_bf16(raw)
        self.assertLess(len(encoded), len(raw))
        self.assertEqual(decode_cuda(encoded), raw)


if __name__ == "__main__":
    unittest.main()
