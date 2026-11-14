"""Bit-level and malformed-container tests for the independent BF16 codec."""

import math
import random
import struct
import unittest
import zlib

from fluxmoe.codec import (
    FORMAT_VERSION,
    HEADER_BYTES,
    bf16_to_float,
    bf16_to_floats,
    compress_bf16,
    decompress_bf16,
    float_to_bf16,
    floats_to_bf16,
)


def words_to_bytes(words):
    return b"".join(struct.pack("<H", word) for word in words)


def fix_container_checksum(container):
    """Recompute integrity checksum so structural validation is exercised."""
    data = bytearray(container)
    struct.pack_into("<I", data, 28, 0)
    struct.pack_into("<I", data, 28, zlib.crc32(data))
    return bytes(data)


def make_container(count, bit_count, lengths, payload, raw_crc=0):
    return fix_container_checksum(
        struct.pack("<4sBBHQQII256s", b"FMBF", FORMAT_VERSION, 0,
                    HEADER_BYTES, count, bit_count, raw_crc, 0, bytes(lengths))
        + payload
    )


class CodecRoundTripTests(unittest.TestCase):
    def test_every_possible_bf16_bit_pattern(self):
        raw = words_to_bytes(range(65536))
        encoded = compress_bf16(raw)
        self.assertEqual(decompress_bf16(encoded), raw)
        # Uniform exponent entropy is eight bits, plus the full header.
        self.assertEqual(len(encoded), len(raw) + HEADER_BYTES)

    def test_empty_and_all_single_symbol_exponents(self):
        self.assertEqual(decompress_bf16(compress_bf16(b"")), b"")
        self.assertEqual(len(compress_bf16(b"")), HEADER_BYTES)
        for exponent in range(256):
            raw = words_to_bytes((sign << 15) | (exponent << 7) | mantissa
                                 for sign in range(2) for mantissa in range(128))
            self.assertEqual(decompress_bf16(compress_bf16(raw)), raw)

    def test_special_values_roundtrip_without_canonicalization(self):
        raw = words_to_bytes([0, 0x8000, 0x7F80, 0xFF80, 0x7F81,
                              0x7FC0, 0x7FFF, 0xFF81, 0xFFFF, 1, 0x807F])
        self.assertEqual(decompress_bf16(compress_bf16(raw)), raw)

    def test_nonuniform_random_data_and_determinism(self):
        rng = random.Random(27415)
        raw = words_to_bytes((rng.randrange(2) << 15) |
                             (rng.choice([124, 125, 126, 126, 126, 127, 128]) << 7) |
                             rng.randrange(128) for _ in range(12000))
        encoded = compress_bf16(raw)
        self.assertLess(len(encoded), len(raw))
        self.assertEqual(encoded, compress_bf16(raw))
        self.assertEqual(decompress_bf16(encoded), raw)

    def test_header_overhead_can_make_compression_unprofitable(self):
        raw = floats_to_bf16([1.0])
        encoded = compress_bf16(raw)
        self.assertEqual(len(encoded), HEADER_BYTES + 2)
        self.assertGreater(len(encoded), len(raw))
        self.assertEqual(decompress_bf16(encoded), raw)

    def test_deep_codebook_and_non_byte_aligned_stream(self):
        # Fibonacci frequencies create longer-than-eight-bit codes.
        frequencies = [1, 1]
        for _ in range(12):
            frequencies.append(frequencies[-1] + frequencies[-2])
        raw = words_to_bytes(exponent << 7
                             for exponent, count in enumerate(frequencies)
                             for _ in range(count))
        encoded = compress_bf16(raw)
        self.assertGreater(max(encoded[32:288]), 8)
        self.assertEqual(decompress_bf16(encoded), raw)
        for count in range(1, 18):
            raw = b"\0\0" * count
            self.assertEqual(decompress_bf16(compress_bf16(raw)), raw)


class FloatConversionTests(unittest.TestCase):
    def test_all_finite_words_are_exactly_representable(self):
        for word in range(65536):
            if (word & 0x7F80) != 0x7F80:
                self.assertEqual(float_to_bf16(bf16_to_float(word)), word)

    def test_known_values_signed_zero_infinities_and_nan(self):
        for value, word in [(0.0, 0), (-0.0, 0x8000), (1.0, 0x3F80),
                            (-2.0, 0xC000), (math.inf, 0x7F80),
                            (-math.inf, 0xFF80)]:
            self.assertEqual(float_to_bf16(value), word)
        self.assertEqual(math.copysign(1.0, bf16_to_float(0x8000)), -1.0)
        for word in [0x7F81, 0x7FFF, 0xFF81, 0xFFFF]:
            result = float_to_bf16(bf16_to_float(word))
            self.assertEqual(result & 0x7F80, 0x7F80)
            self.assertNotEqual(result & 0x7F, 0)
            self.assertTrue(math.isnan(bf16_to_float(result)))

    def test_ties_to_even_and_double_rounding_boundaries(self):
        self.assertEqual(float_to_bf16(1.0 + 2.0 ** -8), 0x3F80)
        self.assertEqual(float_to_bf16(1.0 + 3 * 2.0 ** -8), 0x3F82)
        midpoint = 1.0 + 2.0 ** -8
        self.assertEqual(float_to_bf16(math.nextafter(midpoint, math.inf)), 0x3F81)
        self.assertEqual(float_to_bf16(math.nextafter(midpoint, -math.inf)), 0x3F80)
        self.assertEqual(float_to_bf16(-midpoint), 0xBF80)

    def test_underflow_normal_transition_and_overflow(self):
        self.assertEqual(float_to_bf16(2.0 ** -134), 0)
        self.assertEqual(float_to_bf16(-2.0 ** -134), 0x8000)
        self.assertEqual(float_to_bf16(math.nextafter(2.0 ** -134, math.inf)), 1)
        self.assertEqual(float_to_bf16(3 * 2.0 ** -134), 2)
        self.assertEqual(float_to_bf16(127.5 * 2.0 ** -133), 0x0080)
        overflow_midpoint = 255.5 * 2.0 ** 120
        self.assertEqual(float_to_bf16(math.nextafter(overflow_midpoint, -math.inf)), 0x7F7F)
        self.assertEqual(float_to_bf16(overflow_midpoint), 0x7F80)
        self.assertEqual(float_to_bf16(-1e300), 0xFF80)
        self.assertEqual(float_to_bf16(1e-300), 0)

    def test_iterable_helpers_and_argument_validation(self):
        self.assertEqual(floats_to_bf16(x for x in [1, -2, 0]), b"\x80\x3f\x00\xc0\x00\x00")
        self.assertEqual(bf16_to_floats(b"\x80\x3f\x00\xc0"), [1.0, -2.0])
        for word in [-1, 65536]:
            with self.assertRaises(ValueError):
                bf16_to_float(word)
        with self.assertRaises(TypeError):
            bf16_to_float(1.5)
        for function in [compress_bf16, bf16_to_floats]:
            with self.assertRaises(ValueError):
                function(b"\0")
        for function in [compress_bf16, decompress_bf16, bf16_to_floats]:
            with self.assertRaises(TypeError):
                function("not bytes")


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        self.encoded = compress_bf16(floats_to_bf16([1, -2, 3, 0] * 32))

    def test_every_single_byte_corruption_is_rejected(self):
        for index in range(len(self.encoded)):
            corrupted = bytearray(self.encoded)
            corrupted[index] ^= 1
            with self.assertRaises(ValueError, msg=f"byte {index}"):
                decompress_bf16(corrupted)

    def test_every_truncation_and_trailing_data_is_rejected(self):
        for size in range(len(self.encoded)):
            with self.assertRaises(ValueError, msg=f"length {size}"):
                decompress_bf16(self.encoded[:size])
        with self.assertRaises(ValueError):
            decompress_bf16(self.encoded + b"\0")

    def test_raw_checksum_is_verified_independently(self):
        corrupted = bytearray(self.encoded)
        corrupted[HEADER_BYTES] ^= 1
        with self.assertRaisesRegex(ValueError, "decoded BF16 checksum"):
            decompress_bf16(fix_container_checksum(corrupted))

    def test_prefix_code_validation(self):
        for test_lengths, message in [([1, 1, 1], "oversubscribed"),
                                      ([2, 2], "incomplete"),
                                      ([2], "singleton")]:
            lengths = test_lengths + [0] * (256 - len(test_lengths))
            encoded = make_container(3, 3, lengths, b"\0\0\0\0")
            with self.assertRaisesRegex(ValueError, message):
                decompress_bf16(encoded)

    def test_nonzero_padding_and_invalid_singleton_code(self):
        encoded = bytearray(compress_bf16(b"\0\0"))
        encoded[-1] = 1
        with self.assertRaisesRegex(ValueError, "padding"):
            decompress_bf16(fix_container_checksum(encoded))
        encoded[-1] = 128
        with self.assertRaisesRegex(ValueError, "invalid Huffman"):
            decompress_bf16(fix_container_checksum(encoded))

    def test_unused_symbols_and_excess_decoded_symbols(self):
        lengths = [1, 1] + [0] * 254
        encoded = make_container(2, 2, lengths, b"\0\0\0")
        with self.assertRaisesRegex(ValueError, "unused symbols"):
            decompress_bf16(encoded)
        encoded = make_container(2, 3, lengths, b"\0\0\x40")
        with self.assertRaisesRegex(ValueError, "excess symbols"):
            decompress_bf16(encoded)

    def test_partial_code_missing_symbols_and_bad_empty_container(self):
        lengths = [1, 2, 2] + [0] * 253
        # Exponents 0, 1 followed by an incomplete 1-bit prefix of 2.
        encoded = make_container(3, 4, lengths, b"\0\0\0\x50")
        with self.assertRaisesRegex(ValueError, "incomplete exponent"):
            decompress_bf16(encoded)
        encoded = make_container(0, 0, [0] * 256, b"", raw_crc=1)
        with self.assertRaisesRegex(ValueError, "invalid empty"):
            decompress_bf16(encoded)
        encoded = make_container(1, 0, [1] + [0] * 255, b"\0")
        with self.assertRaisesRegex(ValueError, "invalid exponent"):
            decompress_bf16(encoded)
        encoded = make_container(1, 1, [0] * 256, b"\0\0")
        with self.assertRaisesRegex(ValueError, "empty codebook"):
            decompress_bf16(encoded)

    def test_untrusted_huge_count_rejected_before_allocation(self):
        encoded = make_container(2 ** 63, 2 ** 63, [1] + [0] * 255, b"")
        with self.assertRaisesRegex(ValueError, "payload"):
            decompress_bf16(encoded)


if __name__ == "__main__":
    unittest.main()
