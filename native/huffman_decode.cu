// Independent FMBF reference decoder. CPU builds validated prefix-tree and
// chunk bit offsets; baseline and optimized paths retain bit-exact
// reconstruction and async stream semantics. This is not the author's kernel.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <limits.h>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr unsigned BLOCK_THREADS = 128;
constexpr unsigned WARP_THREADS = 32;
constexpr unsigned BLOCK_WARPS = BLOCK_THREADS / WARP_THREADS;
static_assert(BLOCK_THREADS % WARP_THREADS == 0 && BLOCK_THREADS <= 1024);
template <class T> struct DeviceBuffer {
    T* pointer = nullptr;
    ~DeviceBuffer() { if (pointer) cudaFree(pointer); }
    cudaError_t allocate(size_t count) {
        return cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T));
    }
};
struct Event {
    cudaEvent_t handle = nullptr;
    ~Event() { if (handle) cudaEventDestroy(handle); }
};
void message(char* error, size_t size, const char* where, cudaError_t status) {
    if (error && size) snprintf(error, size, "%s: %s (%d)", where,
                               cudaGetErrorString(status), static_cast<int>(status));
}
__global__ __launch_bounds__(BLOCK_THREADS) void decode_chunks(
    const uint8_t* sign_mantissa, const uint8_t* stream,
    uint64_t count, uint64_t bit_count, const uint64_t* offsets,
    uint64_t chunks, uint32_t chunk_size, const int32_t* left,
    const int32_t* right, const int32_t* symbols, uint32_t nodes,
    uint16_t* output, int* failed) {
    const uint64_t chunk = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (chunk >= chunks) return;
    const uint64_t first = chunk * chunk_size;
    const uint64_t end = first + chunk_size < count ? first + chunk_size : count;
    uint64_t cursor = offsets[chunk];
    const uint64_t chunk_end = offsets[chunk + 1];
    if (cursor > chunk_end || chunk_end > bit_count) {
        atomicExch(failed, 1); return;
    }
    for (uint64_t index = first; index < end; ++index) {
        int32_t node = 0;
        uint32_t depth = 0;
        while (symbols[node] < 0) {
            if (cursor >= chunk_end || ++depth > 255) {
                atomicExch(failed, 2); return;
            }
            const uint8_t bit = (stream[cursor >> 3] >> (7 - (cursor & 7))) & 1;
            ++cursor;
            node = bit ? right[node] : left[node];
            if (node < 0 || static_cast<uint32_t>(node) >= nodes) {
                atomicExch(failed, 3); return;
            }
        }
        const int32_t exponent = symbols[node];
        if (exponent > 255) { atomicExch(failed, 4); return; }
        const uint8_t sm = sign_mantissa[index];
        output[index] = static_cast<uint16_t>(((sm & 0x80) << 8) |
                                             (exponent << 7) | (sm & 0x7f));
    }
    if (cursor != chunk_end) atomicExch(failed, 5);
}

// Immutable 10-bit prefix table: low byte is code length, high bits are either
// the exponent or (length=0) a continuation trie node. Long codes retain exact
// tree fallback. The final zero padding permits two aligned u32 lookahead loads.
constexpr unsigned LOOK_BITS = 10;
constexpr unsigned LOOK_SIZE = 1u << LOOK_BITS;
// Two MSB-first words in registers. Normal refills consume only one new word
// per 32 bits; a long-code fallback may jump several words and reload both.
// This keeps the on-disk FMBF format and all chunk offsets unchanged.
struct BitWindow {
    uint64_t word = UINT64_MAX;
    uint32_t high = 0, low = 0;
    __device__ __forceinline__ unsigned prefix(const uint8_t* bytes, uint64_t cursor) {
        const auto* words = reinterpret_cast<const uint32_t*>(bytes);
        const uint64_t next = cursor >> 5;
        if (next != word) {
            high = word != UINT64_MAX && next == word + 1
                ? low : __byte_perm(words[next], 0, 0x0123);
            low = __byte_perm(words[next + 1], 0, 0x0123);
            word = next;
        }
        return __funnelshift_l(low, high, cursor & 31) >> (32 - LOOK_BITS);
    }
};
__device__ __forceinline__ unsigned next_exponent(
    const uint8_t* bytes, uint64_t& cursor, uint64_t end,
    const uint32_t* table, const int32_t* left, const int32_t* right,
    const int32_t* symbols, uint32_t nodes, bool& valid) {
    const auto* words = reinterpret_cast<const uint32_t*>(bytes);
    const uint64_t word = cursor >> 5;
    const uint32_t a = __byte_perm(words[word], 0, 0x0123);
    const uint32_t b = __byte_perm(words[word + 1], 0, 0x0123);
    const unsigned prefix = __funnelshift_l(b, a, cursor & 31) >> (32 - LOOK_BITS);
    const uint32_t entry = table[prefix];
    const unsigned width = entry & 255;
    if (entry == UINT32_MAX || cursor >= end) { valid = false; return 0; }
    if (width) {
        cursor += width;
        valid = cursor <= end;
        return entry >> 8;
    }
    cursor += LOOK_BITS;
    int32_t node = entry >> 8;
    unsigned depth = LOOK_BITS;
    while (node >= 0 && static_cast<uint32_t>(node) < nodes && symbols[node] < 0) {
        if (cursor >= end || ++depth > 255) { valid = false; return 0; }
        const unsigned bit = (bytes[cursor >> 3] >> (7 - (cursor & 7))) & 1;
        ++cursor; node = bit ? right[node] : left[node];
    }
    if (cursor > end || node < 0 || static_cast<uint32_t>(node) >= nodes ||
        symbols[node] < 0 || symbols[node] > 255) { valid = false; return 0; }
    return symbols[node];
}

// Lookup-only ablation keeps the original strided IO; tiled=true additionally
// transposes each warp's 32x32 output tile through padded shared memory. The
// stores then cover consecutive BF16 words instead of 512-byte lane strides.
template <bool Tiled, bool Bulk = false, bool Buffered = false, bool VectorIO = false>
__global__ __launch_bounds__(BLOCK_THREADS) void decode_prefix_chunks(
    const uint8_t* sign_mantissa, const uint8_t* bytes, uint64_t count,
    uint64_t bit_count, const uint64_t* offsets, uint64_t chunks,
    uint32_t chunk_size, const int32_t* left, const int32_t* right,
    const int32_t* symbols, uint32_t nodes, const uint32_t* global_table, const uint64_t* global_bulk,
    uint16_t* output, int* failed) {
    static_assert(!Bulk || Tiled);
    static_assert(!Buffered || Bulk);
    static_assert(!VectorIO || Bulk);
    __shared__ uint32_t table[LOOK_SIZE];
    __shared__ uint64_t bulk_table[Bulk ? LOOK_SIZE : 1];
    // 34 uint16 elements = 17 banks per row; packed uint32 accesses are aligned.
    // Retain the previous 33-column layout in the old ablations.
    __shared__ __align__(4) uint16_t tile[Tiled ? BLOCK_WARPS : 1][32][Tiled ? (VectorIO ? 34 : 33) : 1];
    for (unsigned i = threadIdx.x; i < LOOK_SIZE; i += blockDim.x) table[i] = global_table[i];
    if constexpr (Bulk) for (unsigned i = threadIdx.x; i < LOOK_SIZE; i += blockDim.x) bulk_table[i] = global_bulk[i];
    __syncthreads();
    const uint64_t chunk = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint64_t first = chunk * chunk_size;
    uint64_t cursor = chunk < chunks ? offsets[chunk] : 0;
    const uint64_t end = chunk < chunks ? offsets[chunk + 1] : 0;
    bool valid = chunk < chunks && cursor <= end && end <= bit_count;
    uint32_t pending = 0, widths = 0;
    BitWindow window;
    if (chunk < chunks && !valid) atomicExch(failed, 1);
    if constexpr (!Tiled) {
        if (chunk >= chunks || !valid) return;
        for (uint32_t i = 0; i < chunk_size && first + i < count; ++i) {
            const unsigned e = next_exponent(bytes, cursor, end, table, left, right, symbols, nodes, valid);
            if (!valid) { atomicExch(failed, 6); return; }
            const uint8_t sm = sign_mantissa[first + i];
            output[first + i] = ((sm & 128u) << 8) | (e << 7) | (sm & 127u);
        }
    } else {
        // Every lane participates in both warp barriers, including tail chunks
        // and failed chunks. No early exit can strand another lane at syncwarp.
        for (uint32_t base = 0; base < chunk_size; base += 32) {
            const uint64_t warp_first = chunk - lane;
            const bool full_tile = warp_first + 32 <= chunks && base + 32 <= chunk_size &&
                (warp_first + 31) * chunk_size + base + 32 <= count;
            const bool vector_io = VectorIO && full_tile && chunk_size % 4 == 0 &&
                reinterpret_cast<uintptr_t>(output) % 4 == 0;
            if constexpr (Bulk) {
              if (vector_io) {
                #pragma unroll 1
                for (unsigned group = 0; group < 8; ++group) {
                    const unsigned row = group * 4 + lane / 8, col = (lane % 8) * 4;
                    const uint64_t target = (warp_first + row) * chunk_size + base + col;
                    const uint32_t sm = *reinterpret_cast<const uint32_t*>(sign_mantissa + target);
                    auto* pair = reinterpret_cast<uint32_t*>(&tile[warp][row][col]);
                    pair[0] = (sm & 255) | ((sm & 0xff00) << 8);
                    pair[1] = ((sm >> 16) & 255) | ((sm >> 8) & 0xff0000);
                }
              } else {
                #pragma unroll 1
                for (unsigned row = 0; row < 32; ++row) {
                    const uint64_t target = (warp_first + row) * chunk_size + base + lane;
                    tile[warp][row][lane] = (warp_first + row < chunks && base + lane < chunk_size && target < count)
                        ? sign_mantissa[target] : 0;
                }
              }
                __syncwarp();
                // Compute the tail limit once; full tiles drain 32 symbols.
                // The old bulk ablation retains its original per-symbol tests.
                const unsigned emit_limit = first + base < count
                    ? static_cast<unsigned>(min(uint64_t(32), min(uint64_t(chunk_size - base), count - first - base))) : 0;
                unsigned i = 0;
                while (valid && (VectorIO ? i < emit_limit :
                        (i < 32 && base + i < chunk_size && first + base + i < count))) {
                    if (!widths) {
                        unsigned prefix;
                        if constexpr (Buffered) prefix = window.prefix(bytes, cursor);
                        else {
                        const auto* words = reinterpret_cast<const uint32_t*>(bytes);
                        const uint64_t word = cursor >> 5;
                        const uint32_t a = __byte_perm(words[word], 0, 0x0123);
                        const uint32_t b = __byte_perm(words[word + 1], 0, 0x0123);
                        prefix = __funnelshift_l(b,a,cursor & 31) >> (32 - LOOK_BITS);
                        }
                        const uint64_t entry = bulk_table[prefix];
                        pending = static_cast<uint32_t>(entry); widths = entry >> 32;
                        if (!widths) {
                            const unsigned e = next_exponent(bytes,cursor,end,table,left,right,symbols,nodes,valid);
                            if (valid) {
                                const uint8_t sm = tile[warp][lane][i];
                                tile[warp][lane][i] = ((sm & 128u) << 8) | (e << 7) | (sm & 127u);
                                ++i;
                            } else atomicExch(failed,6);
                            continue;
                        }
                    }
                    // Drain a whole lookup result before refilling. Refilling
                    // per output causes divergent lanes to execute a lookup
                    // on almost every warp instruction even with a queue.
                    #pragma unroll 1
                    for (unsigned j = 0; j < 4; ++j) {
                        if (widths && valid && (VectorIO ? i < emit_limit :
                                (i < 32 && base + i < chunk_size && first + base + i < count))) {
                            cursor += widths & 255;
                            const unsigned e = pending & 255;
                            widths >>= 8; pending >>= 8;
                            if (cursor <= end) {
                                const uint8_t sm = tile[warp][lane][i];
                                tile[warp][lane][i] = ((sm & 128u) << 8) | (e << 7) | (sm & 127u);
                                ++i;
                            } else { valid = false; atomicExch(failed,6); }
                        }
                    }
                }
            } else {
              #pragma unroll 1
              for (unsigned i = 0; i < 32; ++i) {
                uint16_t value = 0;
                if (valid && base + i < chunk_size && first + base + i < count) {
                    const unsigned e = next_exponent(bytes, cursor, end, table, left, right, symbols, nodes, valid);
                    if (valid) {
                        const uint8_t sm = sign_mantissa[first + base + i];
                        value = ((sm & 128u) << 8) | (e << 7) | (sm & 127u);
                    } else atomicExch(failed, 6);
                }
                tile[warp][lane][i] = value;
              }
            }
            __syncwarp();
          if (vector_io) {
            #pragma unroll 1
            for (unsigned group = 0; group < 16; ++group) {
                const unsigned row = group * 2 + lane / 16, col = (lane % 16) * 2;
                const uint64_t target = (warp_first + row) * chunk_size + base + col;
                *reinterpret_cast<uint32_t*>(output + target) =
                    *reinterpret_cast<const uint32_t*>(&tile[warp][row][col]);
            }
          } else {
            #pragma unroll 1
            for (unsigned row = 0; row < 32; ++row) {
                const uint64_t target = (warp_first + row) * chunk_size + base + lane;
                if (warp_first + row < chunks && base + lane < chunk_size && target < count)
                    output[target] = tile[warp][row][lane];
            }
          }
            __syncwarp();
        }
    }
    if (chunk < chunks && valid && cursor != end) atomicExch(failed, 5);
}
} // namespace

extern "C" int fluxmoe_cuda_device_info(char* output, size_t size) {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) { message(output, size, "cudaGetDeviceCount", status); return -1; }
    if (count == 0) { if (output && size) snprintf(output, size, "No CUDA devices"); return 0; }
    int current = 0, runtime = 0, driver = 0;
    status = cudaGetDevice(&current);
    if (status != cudaSuccess) { message(output, size, "cudaGetDevice", status); return -1; }
    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, current);
    if (status != cudaSuccess) { message(output, size, "cudaGetDeviceProperties", status); return -1; }
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    if (output && size) snprintf(output, size,
        "%s; device=%d; SM=%d.%d; memory=%llu; runtime=%d; driver=%d",
        prop.name, current, prop.major, prop.minor,
        static_cast<unsigned long long>(prop.totalGlobalMem), runtime, driver);
    return 1;
}

// All pointers supplied here refer to HOST memory. The output is constructed
// in cudaMalloc device storage by decode_chunks, then copied to host_output.
extern "C" int fluxmoe_cuda_decode(
    const uint8_t* host_sm, const uint8_t* host_stream,
    uint64_t count, uint64_t bit_count, const uint64_t* host_offsets,
    uint64_t chunks, uint32_t chunk_size, const int32_t* host_left,
    const int32_t* host_right, const int32_t* host_symbols, uint32_t nodes,
    uint16_t* host_output, float* kernel_ms, char* error, size_t error_size) {
    if (kernel_ms) *kernel_ms = 0.0f;
    if (error && error_size) error[0] = 0;
    if (count == 0) return 0;
    if (!chunk_size || !nodes || nodes > 512 ||
        chunks != count / chunk_size + (count % chunk_size != 0) ||
        count > SIZE_MAX / sizeof(uint16_t) || bit_count > UINT64_MAX - 7 ||
        chunks > static_cast<uint64_t>(INT_MAX) * BLOCK_THREADS ||
        !host_sm || !host_stream || !host_offsets || !host_left ||
        !host_right || !host_symbols || !host_output) {
        if (error && error_size) snprintf(error, error_size, "invalid native decoder arguments");
        return -1;
    }
    DeviceBuffer<uint8_t> sm, stream;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<int32_t> left, right, symbols;
    DeviceBuffer<uint16_t> output;
    DeviceBuffer<int> failed;
    Event start, stop;
#define CHECK(call) do { cudaError_t status = (call); if (status != cudaSuccess) { \
    message(error, error_size, #call, status); return -1; } } while (0)
    CHECK(sm.allocate(count));
    CHECK(stream.allocate((bit_count + 7) / 8));
    CHECK(offsets.allocate(chunks + 1));
    CHECK(left.allocate(nodes)); CHECK(right.allocate(nodes)); CHECK(symbols.allocate(nodes));
    CHECK(output.allocate(count)); CHECK(failed.allocate(1));
    CHECK(cudaMemcpy(sm.pointer, host_sm, count, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(stream.pointer, host_stream, (bit_count + 7) / 8, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(offsets.pointer, host_offsets, (chunks + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(left.pointer, host_left, nodes * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(right.pointer, host_right, nodes * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(symbols.pointer, host_symbols, nodes * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(failed.pointer, 0, sizeof(int)));
    CHECK(cudaEventCreate(&start.handle)); CHECK(cudaEventCreate(&stop.handle));
    CHECK(cudaEventRecord(start.handle));
    decode_chunks<<<static_cast<unsigned>((chunks + BLOCK_THREADS - 1) / BLOCK_THREADS), BLOCK_THREADS>>>(
        sm.pointer, stream.pointer, count, bit_count, offsets.pointer, chunks,
        chunk_size, left.pointer, right.pointer, symbols.pointer, nodes,
        output.pointer, failed.pointer);
    CHECK(cudaPeekAtLastError());
    CHECK(cudaEventRecord(stop.handle));
    CHECK(cudaEventSynchronize(stop.handle));
    if (kernel_ms) CHECK(cudaEventElapsedTime(kernel_ms, start.handle, stop.handle));
    int failure = 0;
    CHECK(cudaMemcpy(&failure, failed.pointer, sizeof(int), cudaMemcpyDeviceToHost));
    if (failure) {
        if (error && error_size) snprintf(error, error_size, "GPU Huffman validation failed: code %d", failure);
        return -1;
    }
    CHECK(cudaMemcpy(host_output, output.pointer, count * sizeof(uint16_t), cudaMemcpyDeviceToHost));
#undef CHECK
    return 0;
}

// Persistent compressed backing for asynchronous decode into a caller-owned
// device address (including CUDA VMM). Preparation/upload is outside the hot
// path. Stream ordering and destination WAR/RAW protection belong to the caller.
// Runtime stream/event rules:
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
namespace {
void require_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
}
struct DeviceScope {
    int previous = -1;
    explicit DeviceScope(int requested) {
        require_cuda(cudaGetDevice(&previous), "cudaGetDevice");
        if (previous != requested) require_cuda(cudaSetDevice(requested), "cudaSetDevice");
    }
    ~DeviceScope() { if (previous >= 0) cudaSetDevice(previous); }
};
template <typename Function>
int guarded_source(char* error, size_t capacity, Function&& fn) noexcept {
    if (error && capacity) error[0] = 0;
    try { fn(); return 0; }
    catch (const std::exception& exception) {
        if (error && capacity) snprintf(error, capacity, "%s", exception.what());
    } catch (...) {
        if (error && capacity) snprintf(error, capacity, "unknown persistent decoder error");
    }
    return -1;
}
struct CompressedSource {
    int device;
    uint64_t count, bit_count, chunks;
    uint32_t chunk_size, nodes;
    size_t bytes = 0;
    uint8_t *sm = nullptr, *stream = nullptr;
    uint64_t* offsets = nullptr;
    int32_t *left = nullptr, *right = nullptr, *symbols = nullptr;
    int* failed = nullptr;
    uint32_t* lookup = nullptr;
    uint64_t* bulk_lookup = nullptr;
    int decoder = 6;
    bool poisoned = false; // Set after an explicit check observes a GPU failure.
    std::unordered_map<uintptr_t, cudaEvent_t> last_reads;
    std::mutex mutex;
    CompressedSource(int d, uint64_t c, uint64_t b, uint64_t k, uint32_t s, uint32_t n)
        : device(d), count(c), bit_count(b), chunks(k), chunk_size(s), nodes(n) {}
    template <typename T> void upload(T*& destination, const T* host, size_t length) {
        require_cuda(cudaMalloc(reinterpret_cast<void**>(&destination), length * sizeof(T)), "allocate compressed source");
        bytes += length * sizeof(T);
        require_cuda(cudaMemcpy(destination, host, length * sizeof(T), cudaMemcpyHostToDevice), "upload compressed source");
    }
    void wait() {
        for (const auto& entry : last_reads)
            require_cuda(cudaEventSynchronize(entry.second), "wait for source's final GPU reader");
    }
    // Cleanup is a lifecycle boundary. It waits only for this source's readers,
    // then frees allocations; ordinary decode_into never waits or allocates.
    ~CompressedSource() noexcept {
        int previous = -1;
        cudaGetDevice(&previous);
        if (cudaSetDevice(device) == cudaSuccess) {
            for (const auto& entry : last_reads) cudaEventSynchronize(entry.second);
            for (const auto& entry : last_reads) cudaEventDestroy(entry.second);
            for (void* pointer : {static_cast<void*>(sm), static_cast<void*>(stream),
                                 static_cast<void*>(offsets), static_cast<void*>(left),
                                 static_cast<void*>(right), static_cast<void*>(symbols),
                                 static_cast<void*>(failed), static_cast<void*>(lookup), static_cast<void*>(bulk_lookup)})
                if (pointer) cudaFree(pointer);
        }
        if (previous >= 0) cudaSetDevice(previous);
    }
};
CompressedSource& source_from(void* handle) {
    if (!handle) throw std::invalid_argument("compressed source is closed");
    return *static_cast<CompressedSource*>(handle);
}
}  // namespace

extern "C" int fluxmoe_cuda_source_create(
    int device, const uint8_t* host_sm, const uint8_t* host_stream,
    uint64_t count, uint64_t bit_count, const uint64_t* host_offsets,
    uint64_t chunks, uint32_t chunk_size, const int32_t* host_left,
    const int32_t* host_right, const int32_t* host_symbols, uint32_t nodes,
    void** output, size_t* device_bytes, char* error, size_t error_size) {
    if (output) *output = nullptr;
    return guarded_source(error, error_size, [&] {
        if (!output || !device_bytes || device < 0 || !count || !chunk_size || !nodes || nodes > 512 ||
            chunks != count / chunk_size + (count % chunk_size != 0) ||
            count > SIZE_MAX / 2 || bit_count > UINT64_MAX - 7 ||
            chunks > static_cast<uint64_t>(INT_MAX) * BLOCK_THREADS ||
            !host_sm || !host_stream || !host_offsets || !host_left || !host_right || !host_symbols)
            throw std::invalid_argument("invalid persistent compressed source arguments");
        // The FMBF encoder assigns even a single symbol a nonzero code length.
        // Reject a zero-bit root leaf at the C ABI boundary instead of building
        // an ambiguous (width=0) prefix table.
        if (host_symbols[0] >= 0)
            throw std::invalid_argument("FMBF requires a non-leaf root and nonzero code lengths");
        for (uint32_t node = 0; node < nodes; ++node) {
            if (host_symbols[node] > 255 || host_symbols[node] < -1 ||
                host_left[node] < -1 || host_right[node] < -1 ||
                host_left[node] >= static_cast<int32_t>(nodes) || host_right[node] >= static_cast<int32_t>(nodes))
                throw std::invalid_argument("invalid FMBF prefix tree");
        }
        DeviceScope scope(device);
        auto source = std::make_unique<CompressedSource>(device, count, bit_count, chunks, chunk_size, nodes);
        source->upload(source->sm, host_sm, count);
        const size_t payload_bytes = (bit_count + 7) / 8;
        require_cuda(cudaMalloc(reinterpret_cast<void**>(&source->stream), payload_bytes + 8), "allocate padded bitstream");
        source->bytes += payload_bytes + 8;
        require_cuda(cudaMemcpy(source->stream, host_stream, payload_bytes, cudaMemcpyHostToDevice), "upload bitstream");
        require_cuda(cudaMemset(source->stream + payload_bytes, 0, 8), "pad bitstream");
        source->upload(source->offsets, host_offsets, chunks + 1);
        source->upload(source->left, host_left, nodes);
        source->upload(source->right, host_right, nodes);
        source->upload(source->symbols, host_symbols, nodes);
        std::vector<uint32_t> table(LOOK_SIZE, UINT32_MAX);
        for (unsigned prefix = 0; prefix < LOOK_SIZE; ++prefix) {
            int node = 0; unsigned depth = 0;
            while (node >= 0 && static_cast<unsigned>(node) < nodes && host_symbols[node] < 0 && depth < LOOK_BITS) {
                node = ((prefix >> (LOOK_BITS - 1 - depth)) & 1) ? host_right[node] : host_left[node];
                ++depth;
            }
            if (node >= 0 && static_cast<unsigned>(node) < nodes)
                table[prefix] = host_symbols[node] >= 0 ? (host_symbols[node] << 8) | depth : node << 8;
        }
        source->upload(source->lookup, table.data(), table.size());
        std::vector<uint64_t> bulk(LOOK_SIZE,0);
        for (unsigned prefix = 0; prefix < LOOK_SIZE; ++prefix) {
            unsigned position = 0;
            for (unsigned item = 0; item < 4; ++item) {
                int node = 0; unsigned depth = 0;
                while (node >= 0 && static_cast<unsigned>(node) < nodes && host_symbols[node] < 0 && position + depth < LOOK_BITS) {
                    node = ((prefix >> (LOOK_BITS - 1 - position - depth)) & 1) ? host_right[node] : host_left[node];
                    ++depth;
                }
                if (node < 0 || static_cast<unsigned>(node) >= nodes || host_symbols[node] < 0) break;
                bulk[prefix] |= static_cast<uint64_t>(host_symbols[node]) << (8 * item);
                bulk[prefix] |= static_cast<uint64_t>(depth) << (32 + 8 * item);
                position += depth;
            }
        }
        source->upload(source->bulk_lookup, bulk.data(), bulk.size());
        const int zero = 0;
        source->upload(source->failed, &zero, 1);
        *device_bytes = source->bytes;
        *output = source.release();
    });
}

extern "C" int fluxmoe_cuda_source_set_decoder(void* handle, int decoder, char* error, size_t error_size) {
    return guarded_source(error, error_size, [&] {
        auto& source = source_from(handle);
        std::lock_guard<std::mutex> lock(source.mutex);
        if (decoder < 0 || decoder > 6) throw std::invalid_argument("unknown CUDA decoder");
        DeviceScope scope(source.device); source.wait(); source.decoder = decoder;
    });
}

extern "C" int fluxmoe_cuda_source_decode(
    void* handle, uintptr_t destination, uintptr_t stream_handle, size_t output_bytes,
    char* error, size_t error_size) {
    return guarded_source(error, error_size, [&] {
        auto& source = source_from(handle);
        std::lock_guard<std::mutex> lock(source.mutex);
        if (!destination || destination % alignof(uint16_t) || output_bytes < source.count * sizeof(uint16_t))
            throw std::invalid_argument("invalid/undersized BF16 device destination");
        if (source.poisoned)
            throw std::runtime_error("compressed source failed validation; recreate it before decoding");
        DeviceScope scope(source.device);
        auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
        int stream_device = -1;
        require_cuda(cudaStreamGetDevice(stream, &stream_device), "query destination stream device");
        if (stream_device != source.device)
            throw std::invalid_argument("stream belongs to a different CUDA device");
        cudaPointerAttributes attributes{};
        require_cuda(cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(destination)), "query destination pointer");
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != source.device)
            throw std::invalid_argument("destination is not device memory on the compressed source's device");
        cudaStreamCaptureStatus capture;
        require_cuda(cudaStreamIsCapturing(stream, &capture), "query capture status");
        if (capture != cudaStreamCaptureStatusNone)
            throw std::invalid_argument("persistent decoder CUDA Graph capture is not supported yet");
        // One completion event per stream is reused. First use creates an event,
        // subsequent calls allocate no CUDA storage or events.
        auto found = source.last_reads.find(stream_handle);
        if (found == source.last_reads.end()) {
            cudaEvent_t event = nullptr;
            require_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create source read event");
            try { found = source.last_reads.emplace(stream_handle, event).first; }
            catch (...) { cudaEventDestroy(event); throw; }
        }
        const unsigned grid = static_cast<unsigned>((source.chunks + BLOCK_THREADS - 1) / BLOCK_THREADS);
        if (source.decoder == 0) {
            decode_chunks<<<grid, BLOCK_THREADS, 0, stream>>>(source.sm, source.stream, source.count, source.bit_count,
                source.offsets, source.chunks, source.chunk_size, source.left, source.right, source.symbols,
                source.nodes, reinterpret_cast<uint16_t*>(destination), source.failed);
        } else if (source.decoder == 1) {
            decode_prefix_chunks<false><<<grid, BLOCK_THREADS, 0, stream>>>(source.sm, source.stream, source.count, source.bit_count,
                source.offsets, source.chunks, source.chunk_size, source.left, source.right, source.symbols,
                source.nodes, source.lookup, source.bulk_lookup, reinterpret_cast<uint16_t*>(destination), source.failed);
        } else if (source.decoder == 2) {
            decode_prefix_chunks<true><<<grid, BLOCK_THREADS, 0, stream>>>(source.sm, source.stream, source.count, source.bit_count,
                source.offsets, source.chunks, source.chunk_size, source.left, source.right, source.symbols,
                source.nodes, source.lookup, source.bulk_lookup, reinterpret_cast<uint16_t*>(destination), source.failed);
        } else if (source.decoder == 3) {
            decode_prefix_chunks<true,true><<<grid, BLOCK_THREADS, 0, stream>>>(source.sm, source.stream, source.count, source.bit_count,
                source.offsets, source.chunks, source.chunk_size, source.left, source.right, source.symbols,
                source.nodes, source.lookup, source.bulk_lookup, reinterpret_cast<uint16_t*>(destination), source.failed);
        } else if (source.decoder == 4) {
            decode_prefix_chunks<true,true,true><<<grid, BLOCK_THREADS, 0, stream>>>(source.sm, source.stream, source.count, source.bit_count,
                source.offsets, source.chunks, source.chunk_size, source.left, source.right, source.symbols,
                source.nodes, source.lookup, source.bulk_lookup, reinterpret_cast<uint16_t*>(destination), source.failed);
        } else if (source.decoder == 5) {
            decode_prefix_chunks<true,true,true,true><<<grid, BLOCK_THREADS, 0, stream>>>(source.sm, source.stream, source.count, source.bit_count,
                source.offsets, source.chunks, source.chunk_size, source.left, source.right, source.symbols,
                source.nodes, source.lookup, source.bulk_lookup, reinterpret_cast<uint16_t*>(destination), source.failed);
        } else {
            decode_prefix_chunks<true,true,false,true><<<grid, BLOCK_THREADS, 0, stream>>>(source.sm, source.stream, source.count, source.bit_count,
                source.offsets, source.chunks, source.chunk_size, source.left, source.right, source.symbols,
                source.nodes, source.lookup, source.bulk_lookup, reinterpret_cast<uint16_t*>(destination), source.failed);
        }
        cudaError_t launched = cudaPeekAtLastError();
        cudaError_t recorded = cudaEventRecord(found->second, stream);
        if (recorded != cudaSuccess) {
            // Error recovery must not free source storage while this launch uses it.
            cudaStreamSynchronize(stream);
            require_cuda(recorded, "record final source reader");
        }
        // Record a lifetime barrier even when last-error reports an earlier
        // asynchronous failure: this launch may still have been queued.
        require_cuda(launched, "launch persistent BF16 decoder");
    });
}

extern "C" int fluxmoe_cuda_source_check(void* handle, char* error, size_t error_size) {
    return guarded_source(error, error_size, [&] {
        auto& source = source_from(handle);
        std::lock_guard<std::mutex> lock(source.mutex);
        DeviceScope scope(source.device);
        source.wait();
        int failed = 0;
        require_cuda(cudaMemcpy(&failed, source.failed, sizeof(int), cudaMemcpyDeviceToHost), "read source error flag");
        if (failed) {
            source.poisoned = true;
            throw std::runtime_error("GPU Huffman validation failed: code " + std::to_string(failed) +
                                     "; source is poisoned, recreate it before decoding");
        }
    });
}

extern "C" int fluxmoe_cuda_source_destroy(void* handle, char* error, size_t error_size) {
    return guarded_source(error, error_size, [&] {
        if (!handle) return;
        std::unique_ptr<CompressedSource> source(static_cast<CompressedSource*>(handle));
        DeviceScope scope(source->device);
        source->wait();
    });
}
