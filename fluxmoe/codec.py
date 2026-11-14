"""Lossless BF16 exponent Huffman codec, implemented on the CPU.

This is an independent, inspectable reference format, not the unpublished
FluxMoE wire format or a GPU decoder. A BF16 word is split into one verbatim
sign/mantissa byte and an eight-bit exponent. Only exponents are compressed.

FMBF version 1, all multibyte header integers little endian::

    4s magic = b"FMBF"
    B  version = 1
    B  flags = 0
    H  header_bytes = 288
    Q  element_count
    Q  exponent_bit_count (excludes byte padding)
    I  CRC32 of original little-endian BF16 bytes
    I  CRC32 of entire container, with this field set to zero
    256B code lengths, indexed by exponent; zero means absent
    element_count B sign/mantissa bytes: bit 7 sign, bits 6:0 mantissa
    ceil(exponent_bit_count / 8) B exponent codes, most-significant bit first

Canonical codes are ordered by (length, exponent). Nonempty codebooks must
be complete prefix codes; a singleton uses code 0 of length 1. Unused low
bits of the last byte must be zero. Empty input has no codes or payload.
All codebook symbols must occur. CRC32 detects accidental corruption; it is
not an authentication mechanism. Small or high-entropy inputs can expand.
"""

from __future__ import annotations

import heapq
import math
import struct
import zlib
from collections.abc import Iterable

MAGIC = b"FMBF"
FORMAT_VERSION = 1
_HEADER = struct.Struct("<4sBBHQQII256s")
HEADER_BYTES = _HEADER.size
_CONTAINER_CRC_OFFSET = 28


def _bytes(data: bytes, name: str) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like")
    return bytes(data)


def _code_lengths(frequencies: list[int]) -> bytes:
    # Distinct minimum-symbol tie breakers give deterministic trees even
    # when several frequencies are equal; heap entries never compare nodes.
    heap = [(n, symbol, symbol) for symbol, n in enumerate(frequencies) if n]
    heapq.heapify(heap)
    lengths = bytearray(256)
    if not heap:
        return bytes(lengths)
    if len(heap) == 1:
        lengths[heap[0][1]] = 1
        return bytes(lengths)
    while len(heap) > 1:
        na, sa, a = heapq.heappop(heap)
        nb, sb, b = heapq.heappop(heap)
        heapq.heappush(heap, (na + nb, min(sa, sb), (a, b)))
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, int):
            lengths[node] = depth
        else:
            stack.append((node[0], depth + 1))
            stack.append((node[1], depth + 1))
    return bytes(lengths)


def _canonical_codes(lengths: bytes) -> dict[int, tuple[int, int]]:
    ordered = sorted((length, symbol) for symbol, length in enumerate(lengths) if length)
    if not ordered:
        return {}
    if len(ordered) == 1:
        if ordered[0][0] != 1:
            raise ValueError("singleton Huffman code must have length 1")
        return {ordered[0][1]: (0, 1)}
    code = 0
    previous_length = 0
    codes = {}
    for length, symbol in ordered:
        code <<= length - previous_length
        if code >= (1 << length):
            raise ValueError("oversubscribed Huffman codebook")
        codes[symbol] = (code, length)
        code += 1
        previous_length = length
    if code != (1 << previous_length):
        raise ValueError("incomplete Huffman codebook")
    return codes


def _container_crc(data: bytes) -> int:
    crc = zlib.crc32(data[:_CONTAINER_CRC_OFFSET])
    crc = zlib.crc32(b"\0\0\0\0", crc)
    return zlib.crc32(data[_CONTAINER_CRC_OFFSET + 4 :], crc)


def compress_bf16(raw: bytes) -> bytes:
    """Encode little-endian BF16 words without changing any of their bits."""
    raw = _bytes(raw, "raw")
    if len(raw) % 2:
        raise ValueError("BF16 data must contain an even number of bytes")
    count = len(raw) // 2
    frequencies = [0] * 256
    exponents = bytearray(count)
    sign_mantissa = bytearray(count)
    for index in range(count):
        low, high = raw[2 * index], raw[2 * index + 1]
        exponent = ((high & 0x7F) << 1) | (low >> 7)
        exponents[index] = exponent
        frequencies[exponent] += 1
        sign_mantissa[index] = (high & 0x80) | (low & 0x7F)
    lengths = _code_lengths(frequencies)
    codes = _canonical_codes(lengths)
    exponent_bits = bytearray()
    pending = 0
    pending_count = 0
    bit_count = 0
    for exponent in exponents:
        code, width = codes[exponent]
        pending = (pending << width) | code
        pending_count += width
        bit_count += width
        while pending_count >= 8:
            pending_count -= 8
            exponent_bits.append((pending >> pending_count) & 0xFF)
        pending &= (1 << pending_count) - 1
    if pending_count:
        exponent_bits.append(pending << (8 - pending_count))
    header = _HEADER.pack(
        MAGIC, FORMAT_VERSION, 0, HEADER_BYTES, count, bit_count,
        zlib.crc32(raw), 0, lengths,
    )
    result = bytearray(header + sign_mantissa + exponent_bits)
    struct.pack_into("<I", result, _CONTAINER_CRC_OFFSET, _container_crc(result))
    return bytes(result)


def decompress_bf16(encoded: bytes) -> bytes:
    """Decode a complete FMBF container, rejecting malformed or corrupt data.

    ValueError reports invalid containers; TypeError reports non-byte input.
    Counts are validated against the supplied payload before allocating output.
    """
    encoded = _bytes(encoded, "encoded")
    if len(encoded) < HEADER_BYTES:
        raise ValueError("truncated FMBF header")
    magic, version, flags, header_bytes, count, bit_count, raw_crc, crc, lengths = _HEADER.unpack_from(encoded)
    if magic != MAGIC or version != FORMAT_VERSION:
        raise ValueError("unrecognized FMBF magic or version")
    if flags != 0 or header_bytes != HEADER_BYTES:
        raise ValueError("unsupported FMBF flags or header size")
    expected_size = HEADER_BYTES + count + (bit_count + 7) // 8
    if len(encoded) != expected_size:
        raise ValueError("FMBF payload is truncated or has trailing data")
    if _container_crc(encoded) != crc:
        raise ValueError("FMBF container checksum mismatch")
    codes = _canonical_codes(lengths)
    if count == 0:
        if bit_count or codes or raw_crc:
            raise ValueError("invalid empty FMBF container")
        return b""
    if not codes or not count <= bit_count <= 255 * count:
        raise ValueError("invalid exponent bit count or empty codebook")
    if len(codes) > count:
        raise ValueError("codebook has more symbols than elements")
    stream = encoded[HEADER_BYTES + count :]
    padding_bits = (-bit_count) % 8
    if padding_bits and stream[-1] & ((1 << padding_bits) - 1):
        raise ValueError("nonzero exponent padding bits")
    decode = {(width, code): symbol for symbol, (code, width) in codes.items()}
    max_width = max(lengths)
    output = bytearray()
    seen = set()
    code = 0
    width = 0
    decoded_count = 0
    for bit_index in range(bit_count):
        code = (code << 1) | ((stream[bit_index // 8] >> (7 - bit_index % 8)) & 1)
        width += 1
        exponent = decode.get((width, code))
        if exponent is not None:
            if decoded_count == count:
                raise ValueError("exponent stream contains excess symbols")
            sm = encoded[HEADER_BYTES + decoded_count]
            output.append((sm & 0x7F) | ((exponent & 1) << 7))
            output.append((sm & 0x80) | (exponent >> 1))
            decoded_count += 1
            seen.add(exponent)
            code = 0
            width = 0
        elif width >= max_width:
            raise ValueError("invalid Huffman code in exponent stream")
    if width or decoded_count != count:
        raise ValueError("incomplete exponent stream")
    if seen != codes.keys():
        raise ValueError("codebook contains unused symbols")
    if zlib.crc32(output) != raw_crc:
        raise ValueError("decoded BF16 checksum mismatch")
    return bytes(output)


def _round_ratio_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = remainder * 2
    return quotient + (twice > denominator or (twice == denominator and quotient % 2 == 1))


def float_to_bf16(value: float) -> int:
    """Round a Python float directly to BF16, nearest with ties to even.

    Avoids an intermediate float32 conversion and its double-rounding cases.
    Preserves signed zero and infinity. NaNs remain NaNs and are quieted;
    conversion through Python floats does not promise NaN-payload preservation.
    For bit-exact handling of every NaN encoding use the raw-byte codec.
    """
    value = float(value)
    binary64 = struct.unpack("<Q", struct.pack("<d", value))[0]
    sign = (binary64 >> 63) << 15
    if math.isnan(value):
        return sign | 0x7FC0 | ((binary64 >> 45) & 0x3F)
    if math.isinf(value):
        return sign | 0x7F80
    if value == 0:
        return sign
    numerator, denominator = abs(value).as_integer_ratio()
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent > 127:
        return sign | 0x7F80
    quantum_exponent = max(exponent, -126) - 7
    if quantum_exponent >= 0:
        mantissa = _round_ratio_even(numerator, denominator << quantum_exponent)
    else:
        mantissa = _round_ratio_even(numerator << -quantum_exponent, denominator)
    if exponent < -126:
        return sign | mantissa  # 128 naturally encodes the smallest normal.
    if mantissa == 256:
        mantissa = 128
        exponent += 1
    if exponent > 127:
        return sign | 0x7F80
    return sign | ((exponent + 127) << 7) | (mantissa - 128)


def bf16_to_float(word: int) -> float:
    """Convert a 16-bit BF16 word to a Python float (exact for finite values)."""
    if not isinstance(word, int):
        raise TypeError("BF16 word must be an integer")
    if not 0 <= word <= 0xFFFF:
        raise ValueError("BF16 word must lie in [0, 65535]")
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


def floats_to_bf16(values: Iterable[float]) -> bytes:
    """Round values to little-endian BF16 words."""
    output = bytearray()
    for value in values:
        word = float_to_bf16(value)
        output.extend((word & 0xFF, word >> 8))
    return bytes(output)


def bf16_to_floats(raw: bytes) -> list[float]:
    """Read little-endian BF16 words as Python floats."""
    raw = _bytes(raw, "raw")
    if len(raw) % 2:
        raise ValueError("BF16 data must contain an even number of bytes")
    return [bf16_to_float(low | (high << 8)) for low, high in zip(raw[::2], raw[1::2])]
