// Minimal CUDA Driver VMM arena, independent of the PyTorch C++ ABI.
// Fixed VA aliases demonstrate storage semantics only. All copies synchronize;
// callers must not treat this as an asynchronous, hazard-safe paging scheduler.
// API references:
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html

#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string cuda_message(CUresult result, const char* operation) {
    const char *name = nullptr, *detail = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &detail);
    return std::string(operation) + ": " + (name ? name : "CUDA error") + " (" +
           std::to_string(static_cast<int>(result)) + ") " + (detail ? detail : "");
}

void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) throw std::runtime_error(cuda_message(result, operation));
}

void error_text(char* buffer, size_t capacity, const std::string& message) noexcept {
    if (buffer && capacity) std::snprintf(buffer, capacity, "%s", message.c_str());
}

template <typename Function>
int guarded(char* error, size_t capacity, Function&& function) noexcept {
    if (error && capacity) error[0] = '\0';
    try {
        function();
        return 0;
    } catch (const std::exception& exception) {
        error_text(error, capacity, exception.what());
    } catch (...) {
        error_text(error, capacity, "unknown native VMM error");
    }
    return 1;
}

struct ContextScope {
    bool pushed = false;
    explicit ContextScope(CUcontext context) {
        check(cuCtxPushCurrent(context), "cuCtxPushCurrent");
        pushed = true;
    }
    void pop() {
        if (pushed) {
            CUcontext previous = nullptr;
            check(cuCtxPopCurrent(&previous), "cuCtxPopCurrent");
            pushed = false;
        }
    }
    ~ContextScope() {
        if (pushed) {
            CUcontext previous = nullptr;
            cuCtxPopCurrent(&previous);
        }
    }
};

CUdevice supported_device(int ordinal) {
    if (ordinal < 0) throw std::invalid_argument("device must be nonnegative");
    check(cuInit(0), "cuInit");
    CUdevice device;
    check(cuDeviceGet(&device, ordinal), "cuDeviceGet");
    int supported = 0;
    check(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                               device), "query CUDA VMM support");
    if (!supported) throw std::runtime_error("device does not support CUDA virtual memory management");
    return device;
}

CUmemAllocationProp allocation_properties(CUdevice device) {
    CUmemAllocationProp properties{};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;
    return properties;
}

struct Arena {
    size_t layers;
    size_t layer_bytes;
    size_t aligned_bytes = 0;
    size_t granularity = 0;
    CUdevice device = 0;
    CUcontext context = nullptr;
    bool retained = false;
    CUmemGenericAllocationHandle handles[2]{};
    bool allocated[2]{false, false};
    std::vector<CUdeviceptr> addresses;
    std::vector<bool> reserved;
    std::vector<bool> mapped;

    Arena(size_t layer_count, size_t bytes)
        : layers(layer_count), layer_bytes(bytes), addresses(layer_count, 0),
          reserved(layer_count, false), mapped(layer_count, false) {}

    void initialize(int ordinal) {
        device = supported_device(ordinal);
        auto properties = allocation_properties(device);
        check(cuMemGetAllocationGranularity(&granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "cuMemGetAllocationGranularity");
        if (granularity == 0 || layer_bytes > std::numeric_limits<size_t>::max() - (granularity - 1))
            throw std::overflow_error("aligned layer size overflows size_t");
        aligned_bytes = ((layer_bytes + granularity - 1) / granularity) * granularity;
        if (aligned_bytes > std::numeric_limits<size_t>::max() / 2)
            throw std::overflow_error("physical allocation size overflows size_t");
        check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
        retained = true;
        ContextScope scope(context);
        for (size_t block = 0; block < 2; ++block) {
            check(cuMemCreate(&handles[block], aligned_bytes, &properties, 0), "cuMemCreate");
            allocated[block] = true;
        }
        CUmemAccessDesc access{};
        access.location = properties.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        for (size_t layer = 0; layer < layers; ++layer) {
            check(cuMemAddressReserve(&addresses[layer], aligned_bytes, 0, 0, 0), "cuMemAddressReserve");
            reserved[layer] = true;
            check(cuMemMap(addresses[layer], aligned_bytes, 0, handles[layer % 2], 0), "cuMemMap");
            mapped[layer] = true;
            check(cuMemSetAccess(addresses[layer], aligned_bytes, &access, 1), "cuMemSetAccess");
        }
        for (size_t block = 0; block < 2 && block < layers; ++block)
            check(cuMemsetD8(addresses[block], 0, aligned_bytes), "initialize physical buffer");
        check(cuCtxSynchronize(), "synchronize initialization");
        scope.pop();
    }

    // Best-effort cleanup continues after an error and reports the first one.
    // Never reset/destroy a shared primary context; release only our retained ref.
    CUresult cleanup() noexcept {
        CUresult first = CUDA_SUCCESS;
        auto capture = [&first](CUresult result) {
            if (first == CUDA_SUCCESS && result != CUDA_SUCCESS) first = result;
        };
        bool pushed = false;
        if (retained) {
            CUresult result = cuCtxPushCurrent(context);
            capture(result);
            pushed = result == CUDA_SUCCESS;
            if (pushed) capture(cuCtxSynchronize());
        }
        for (size_t layer = layers; layer-- > 0;) {
            if (mapped[layer]) {
                CUresult result = cuMemUnmap(addresses[layer], aligned_bytes);
                capture(result);
                if (result == CUDA_SUCCESS) mapped[layer] = false;
            }
            if (reserved[layer] && !mapped[layer]) {
                CUresult result = cuMemAddressFree(addresses[layer], aligned_bytes);
                capture(result);
                if (result == CUDA_SUCCESS) reserved[layer] = false;
            }
        }
        for (size_t block = 0; block < 2; ++block) {
            if (allocated[block]) {
                CUresult result = cuMemRelease(handles[block]);
                capture(result);
                if (result == CUDA_SUCCESS) allocated[block] = false;
            }
        }
        if (pushed) {
            CUcontext popped = nullptr;
            capture(cuCtxPopCurrent(&popped));
        }
        if (retained) {
            CUresult result = cuDevicePrimaryCtxRelease(device);
            capture(result);
            if (result == CUDA_SUCCESS) retained = false;
        }
        return first;
    }

    ~Arena() { cleanup(); }

    void validate(size_t layer, size_t bytes) const {
        if (layer >= layers) throw std::out_of_range("layer index out of range");
        if (bytes > layer_bytes) throw std::out_of_range("copy exceeds logical layer size");
    }
};

Arena& arena_from(void* handle) {
    if (!handle) throw std::invalid_argument("arena is closed or null");
    return *static_cast<Arena*>(handle);
}

}  // namespace

extern "C" {

int fluxmoe_vmm_probe(int ordinal, size_t* granularity, char* error, size_t capacity) {
    return guarded(error, capacity, [&] {
        if (!granularity) throw std::invalid_argument("granularity output is null");
        auto properties = allocation_properties(supported_device(ordinal));
        check(cuMemGetAllocationGranularity(granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "cuMemGetAllocationGranularity");
    });
}

int fluxmoe_vmm_create(size_t layers, size_t layer_bytes, int ordinal, void** output,
                       char* error, size_t capacity) {
    if (output) *output = nullptr;
    return guarded(error, capacity, [&] {
        if (!output || !layers || !layer_bytes)
            throw std::invalid_argument("nonzero layers, layer_bytes and an output pointer are required");
        if (layers > std::numeric_limits<size_t>::max() / layer_bytes)
            throw std::overflow_error("logical allocation size overflows size_t");
        auto arena = std::make_unique<Arena>(layers, layer_bytes);
        arena->initialize(ordinal);
        *output = arena.release();
    });
}

int fluxmoe_vmm_info(void* handle, uint64_t* addresses, size_t count, size_t* physical_bytes,
                     size_t* logical_bytes, size_t* granularity, char* error, size_t capacity) {
    return guarded(error, capacity, [&] {
        auto& arena = arena_from(handle);
        if (!addresses || count != arena.layers || !physical_bytes || !logical_bytes || !granularity)
            throw std::invalid_argument("invalid arena info outputs");
        for (size_t index = 0; index < count; ++index) addresses[index] = arena.addresses[index];
        *physical_bytes = 2 * arena.aligned_bytes;
        *logical_bytes = arena.layers * arena.layer_bytes;
        *granularity = arena.granularity;
    });
}

int fluxmoe_vmm_write(void* handle, size_t layer, const void* data, size_t bytes,
                      char* error, size_t capacity) {
    return guarded(error, capacity, [&] {
        auto& arena = arena_from(handle);
        arena.validate(layer, bytes);
        if (!data || bytes != arena.layer_bytes)
            throw std::invalid_argument("write requires exactly layer_bytes of input");
        ContextScope scope(arena.context);
        check(cuMemcpyHtoD(arena.addresses[layer], data, bytes), "cuMemcpyHtoD");
        check(cuCtxSynchronize(), "synchronize write");
        scope.pop();
    });
}

int fluxmoe_vmm_read(void* handle, size_t layer, void* data, size_t bytes,
                     char* error, size_t capacity) {
    return guarded(error, capacity, [&] {
        auto& arena = arena_from(handle);
        arena.validate(layer, bytes);
        if (bytes && !data) throw std::invalid_argument("read output is null");
        if (!bytes) return;
        ContextScope scope(arena.context);
        check(cuMemcpyDtoH(data, arena.addresses[layer], bytes), "cuMemcpyDtoH");
        check(cuCtxSynchronize(), "synchronize read");
        scope.pop();
    });
}

int fluxmoe_vmm_destroy(void* handle, char* error, size_t capacity) {
    return guarded(error, capacity, [&] {
        if (!handle) return;
        std::unique_ptr<Arena> arena(static_cast<Arena*>(handle));
        CUresult result = arena->cleanup();
        check(result, "CUDA VMM arena cleanup");
    });
}

}  // extern "C"
