// Fast CPU preparation for the independent FMBF format. Not a GPU benchmark.
#include <cstdint>
#include <cstddef>
extern "C" int fluxmoe_cpu_pack(const uint8_t* raw, uint64_t count,
    const uint64_t* codes, const uint8_t* lengths, uint8_t* sm,
    uint8_t* stream, uint64_t stream_bytes) {
    uint64_t byte = 0; unsigned used = 0; uint8_t pending = 0;
    for (uint64_t i = 0; i < count; ++i) {
        const uint8_t lo = raw[2*i], hi = raw[2*i+1];
        const unsigned exp = ((hi & 127u) << 1) | (lo >> 7);
        sm[i] = (hi & 128u) | (lo & 127u);
        if (!lengths[exp] || lengths[exp] > 63) return 1;
        for (int bit = lengths[exp]-1; bit >= 0; --bit) {
            pending = static_cast<uint8_t>((pending << 1) | ((codes[exp] >> bit) & 1u));
            if (++used == 8) {
                if (byte >= stream_bytes) return 2;
                stream[byte++] = pending; pending = 0; used = 0;
            }
        }
    }
    if (used) {
        if (byte >= stream_bytes) return 3;
        stream[byte++] = static_cast<uint8_t>(pending << (8-used));
    }
    return byte == stream_bytes ? 0 : 4;
}
extern "C" int fluxmoe_cpu_unpack(const uint8_t* sm, const uint8_t* stream,
    uint64_t count, uint64_t bits, const int32_t* left, const int32_t* right,
    const int32_t* symbols, uint32_t nodes, uint32_t chunk_size,
    uint16_t* output, uint64_t* offsets, uint8_t* seen) {
    if (!nodes || nodes > 512 || !chunk_size) return 1;
    uint64_t cursor = 0, chunk = 0;
    for (uint64_t i = 0; i < count; ++i) {
        if (i % chunk_size == 0) offsets[chunk++] = cursor;
        int32_t node = 0; unsigned depth = 0;
        while (symbols[node] < 0) {
            if (cursor >= bits || ++depth > 255) return 2;
            const unsigned b = (stream[cursor >> 3] >> (7-(cursor & 7))) & 1u;
            ++cursor; node = b ? right[node] : left[node];
            if (node < 0 || static_cast<uint32_t>(node) >= nodes) return 3;
        }
        const int32_t e = symbols[node];
        if (e < 0 || e > 255) return 4;
        seen[e] = 1;
        output[i] = static_cast<uint16_t>(((sm[i] & 128u) << 8) | (e << 7) | (sm[i] & 127u));
    }
    offsets[chunk] = cursor;
    return cursor == bits ? 0 : 5;
}
