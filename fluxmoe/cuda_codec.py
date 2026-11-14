"""Optional real CUDA decoder for the CPU reference FMBF format.

No torch extension or package installation is needed. nvcc compiles a C ABI
shared object, linked to one explicit CUDA runtime library. Input validation
and chunk-index construction execute on CPU; only exponent decoding and BF16
word reconstruction execute in the GPU kernel. The complete CPU validation
necessarily decodes the input once. This deliberately favors validation over
speed and is not the paper's optimized decoder or a throughput benchmark.

Build: python3 -m fluxmoe.cuda_codec build --nvcc /path/to/nvcc
Check: FLUXMOE_TEST_CUDA=1 python3 -m unittest tests.test_cuda_codec -v
Demo:  python3 -m fluxmoe.cuda_codec verify --elements 65536

Environment overrides: FLUXMOE_NVCC, FLUXMOE_CUDART, FLUXMOE_CUDA_LIBRARY,
FLUXMOE_CUDA_ARCH (default native), FLUXMOE_CUDA_DRIVER (default libcuda.so.1).
Compiler source: ../native/huffman_decode.cu. Build logs and metadata remain
under ../native/build. No existing virtual environment is modified.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import threading
import time

from .codec import HEADER_BYTES, _canonical_codes, decompress_bf16, compress_bf16

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LIBRARY = _ROOT / "native" / "build" / "libfluxmoe_huffman.so"
_LIBRARIES: dict[str, ctypes.CDLL] = {}
_LIBRARY_HASHES: dict[str, str] = {}


class CudaCodecError(RuntimeError):
    """Readable build, loader, driver, or device execution failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_library_provenance(path: Path, required: bool) -> tuple[dict, str]:
    """Verify source/artifact identity before loading or returning a cached .so."""
    metadata_path = path.with_suffix(".json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        if not isinstance(metadata, dict):
            raise ValueError("build metadata must be a JSON object")
        if required and not all(metadata.get(field) for field in ("source_sha256", "artifact_sha256")):
            raise CudaCodecError(f"CUDA decoder {path} lacks source/artifact provenance; rebuild it.")
        expected_source = metadata.get("source_sha256")
        if expected_source and expected_source != _sha256(_ROOT / "native" / "huffman_decode.cu"):
            raise CudaCodecError(f"CUDA decoder source changed since {path} was built; rebuild it before decoding.")
        artifact_hash = _sha256(path)
        expected_artifact = metadata.get("artifact_sha256")
        if expected_artifact and expected_artifact != artifact_hash:
            raise CudaCodecError(f"CUDA decoder artifact SHA256 mismatch for {path}; rebuild it.")
        return metadata, artifact_hash
    except (OSError, ValueError, TypeError) as exc:
        raise CudaCodecError(f"Cannot verify CUDA decoder provenance for {path}: {exc}; rebuild it.") from exc


def available() -> bool:
    """Whether a CUDA driver/device is accessible; does not compile or load torch.

    This does not imply a CUDA toolkit is installed. build_cuda_library reports
    compiler/runtime failures separately. This is a driver probe, not a full
    runtime-initialization check; explicit empty/-1 device masks return False.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        return False
    try:
        driver = ctypes.CDLL(os.environ.get("FLUXMOE_CUDA_DRIVER", "libcuda.so.1"))
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDeviceGetCount.restype = ctypes.c_int
        count = ctypes.c_int()
        return driver.cuInit(0) == 0 and driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
    except (OSError, AttributeError):
        return False


def _find_nvcc(explicit: str | None = None) -> Path:
    requested = explicit or os.environ.get("FLUXMOE_NVCC")
    if requested:
        found = shutil.which(requested)
        candidate = Path(found or requested).expanduser().resolve()
        if not candidate.is_file():
            raise CudaCodecError(f"nvcc not found: {candidate}")
        return candidate
    candidates = [shutil.which("nvcc")]
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(variable):
            candidates.append(str(Path(os.environ[variable]) / "bin" / "nvcc"))
    for prefix in (Path(sys.prefix), Path.home() / "vllm_env", Path.home() / "torch_env"):
        for pattern in ("lib/python*/site-packages/nvidia/cu13/bin/nvcc",
                        "lib/python*/site-packages/nvidia/cuda_nvcc/bin/nvcc"):
            candidates.extend(sorted(glob.glob(str(prefix / pattern))))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise CudaCodecError("No nvcc found. Set FLUXMOE_NVCC=/absolute/path/to/nvcc; no packages were installed.")


def _find_cudart(nvcc: Path) -> Path:
    explicit = os.environ.get("FLUXMOE_CUDART")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise CudaCodecError(f"FLUXMOE_CUDART does not exist: {candidate}")
    root = nvcc.parent.parent
    for directory in (root / "lib", root / "lib64", root / "targets/x86_64-linux/lib",
                      root.parent / "cuda_runtime/lib"):
        # Prefer the unversioned linker name if installed; otherwise use the
        # exact major SONAME from this toolkit, never a torch runtime lookup.
        matches = sorted(directory.glob("libcudart.so*"), key=lambda path: (len(path.name), path.name))
        if matches:
            return matches[0].resolve()
    raise CudaCodecError(f"No CUDA runtime beside {nvcc}. Set FLUXMOE_CUDART to its matching libcudart.so.")


def build_cuda_library(nvcc: str | None = None, output_dir: str | Path | None = None,
                       architecture: str | None = None) -> Path:
    """Build the decoder and save exact command/stdout/stderr in build.log."""
    compiler = _find_nvcc(nvcc)
    runtime = _find_cudart(compiler)
    directory = Path(output_dir) if output_dir is not None else _DEFAULT_LIBRARY.parent
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "libfluxmoe_huffman.so"
    source = _ROOT / "native" / "huffman_decode.cu"
    if not source.is_file():
        raise CudaCodecError(f"CUDA source missing: {source}")
    source_hash = _sha256(source)
    arch = architecture or os.environ.get("FLUXMOE_CUDA_ARCH", "native")
    command = [str(compiler), "-std=c++17", "-O3", "-shared", "-Xcompiler=-fPIC",
               "--cudart=none", f"-arch={arch}", "-L", str(runtime.parent),
               str(source), "-Xlinker", str(runtime), "-Xlinker", "-rpath", "-Xlinker",
               str(runtime.parent), "-o", str(library)]
    log = directory / "build.log"
    start = time.time()
    version = subprocess.run([str(compiler), "--version"], text=True, capture_output=True, check=False)
    log.write_text("$ " + shlex.join(command) + "\n\n" + version.stdout + version.stderr, encoding="utf-8")
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        with log.open("a", encoding="utf-8") as output:
            output.write(f"\nCould not start compiler: {exc}\n")
        raise CudaCodecError(f"Could not run nvcc; see {log}: {exc}") from exc
    with log.open("a", encoding="utf-8") as output:
        output.write("\nstdout:\n" + result.stdout + "\nstderr:\n" + result.stderr)
        output.write(f"\nexit_code={result.returncode}; elapsed_s={time.time() - start:.6f}\n")
    if result.returncode:
        raise CudaCodecError(f"CUDA compilation failed (exit {result.returncode}); see {log}\n{result.stderr[-2000:]}")
    if _sha256(source) != source_hash:
        raise CudaCodecError("CUDA source changed during compilation; rebuild before loading this artifact.")
    metadata = {"source": str(source), "source_sha256": source_hash,
                "artifact_sha256": _sha256(library),
                "nvcc": str(compiler), "cudart": str(runtime),
                "architecture": arch, "command": command, "build_log": str(log),
                "compiled_at_unix": start, "nvcc_version": version.stdout.strip(),
                "python": sys.executable}
    library.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return library


def _load_library() -> ctypes.CDLL:
    requested = os.environ.get("FLUXMOE_CUDA_LIBRARY")
    path = Path(requested).expanduser().resolve() if requested else _DEFAULT_LIBRARY
    if not path.is_file():
        if requested:
            raise CudaCodecError(f"FLUXMOE_CUDA_LIBRARY not found: {path}")
        path = build_cuda_library()
    metadata, artifact_hash = _validate_library_provenance(
        path, required=path.resolve() == _DEFAULT_LIBRARY.resolve())
    key = str(path)
    if key in _LIBRARIES:
        if _LIBRARY_HASHES[key] != artifact_hash:
            raise CudaCodecError(f"CUDA decoder {path} changed after loading; start a new Python process.")
        return _LIBRARIES[key]
    try:
        # Explicit matching runtime load also yields a clear missing-path error.
        if metadata:
            runtime_path = metadata["cudart"]
            ctypes.CDLL(runtime_path, mode=ctypes.RTLD_LOCAL)
        library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise CudaCodecError(f"Cannot load CUDA decoder {path}: {exc}. Rebuild with matching FLUXMOE_NVCC/FLUXMOE_CUDART.") from exc
    try:
        for name in ("fluxmoe_cuda_device_info", "fluxmoe_cuda_decode", "fluxmoe_cuda_source_create",
                     "fluxmoe_cuda_source_decode", "fluxmoe_cuda_source_check", "fluxmoe_cuda_source_destroy",
                     "fluxmoe_cuda_source_set_decoder"):
            getattr(library, name)
    except AttributeError as exc:
        raise CudaCodecError(f"CUDA decoder {path} has an incompatible C ABI; rebuild it: {exc}") from exc
    pbyte = ctypes.POINTER(ctypes.c_uint8)
    pint = ctypes.POINTER(ctypes.c_int32)
    library.fluxmoe_cuda_device_info.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    library.fluxmoe_cuda_device_info.restype = ctypes.c_int
    library.fluxmoe_cuda_decode.argtypes = [
        pbyte, pbyte, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64, ctypes.c_uint32,
        pint, pint, pint, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float), ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.fluxmoe_cuda_decode.restype = ctypes.c_int
    library.fluxmoe_cuda_source_create.argtypes = [
        ctypes.c_int, pbyte, pbyte, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64, ctypes.c_uint32,
        pint, pint, pint, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t), ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.fluxmoe_cuda_source_decode.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.fluxmoe_cuda_source_set_decoder.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_size_t]
    library.fluxmoe_cuda_source_set_decoder.restype = ctypes.c_int
    for name in ("fluxmoe_cuda_source_check", "fluxmoe_cuda_source_destroy"):
        getattr(library, name).argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    for name in ("fluxmoe_cuda_source_create", "fluxmoe_cuda_source_decode",
                 "fluxmoe_cuda_source_check", "fluxmoe_cuda_source_destroy"):
        getattr(library, name).restype = ctypes.c_int
    _LIBRARIES[key] = library
    _LIBRARY_HASHES[key] = artifact_hash
    return library


def _prepare(encoded: bytes, chunk_size: int) -> dict:
    if not isinstance(chunk_size, int) or not 1 <= chunk_size <= 65536:
        raise ValueError("chunk_size must be an integer in [1, 65536]")
    # Strict CPU validation checks checksums, prefix codes and exact bit count.
    reference = decompress_bf16(encoded)
    count = len(reference) // 2
    bit_count = struct.unpack_from("<Q", encoded, 16)[0]
    lengths = encoded[32:HEADER_BYTES]
    codes = _canonical_codes(lengths)
    left, right, symbols = [-1], [-1], [-1]
    for exponent, (code, width) in codes.items():
        node = 0
        for shift in range(width - 1, -1, -1):
            children = right if (code >> shift) & 1 else left
            child = children[node]
            if child == -1:
                child = len(symbols)
                children[node] = child
                left.append(-1); right.append(-1); symbols.append(-1)
            node = child
        symbols[node] = exponent
    offsets = []
    offset = 0
    for index in range(count):
        if index % chunk_size == 0:
            offsets.append(offset)
        low, high = reference[2 * index], reference[2 * index + 1]
        exponent = ((high & 0x7F) << 1) | (low >> 7)
        offset += lengths[exponent]
    offsets.append(offset)
    if offset != bit_count:
        raise ValueError("internal CPU index length does not match container")
    return {"reference": reference, "count": count, "bit_count": bit_count,
            "sign_mantissa": encoded[HEADER_BYTES:HEADER_BYTES + count],
            "stream": encoded[HEADER_BYTES + count:], "offsets": offsets,
            "left": left, "right": right, "symbols": symbols}


def decode_cuda_with_metrics(encoded: bytes, *, chunk_size: int = 256) -> tuple[bytes, dict]:
    """Decode on GPU, copy back, and compare to a strictly validated CPU result.

    kernel_ms excludes preprocessing, allocations, host/device transfers and
    comparison. End-to-end time includes them and possible first-call build or
    context initialization. Neither number is a steady-state benchmark.
    """
    started = time.perf_counter()
    if not isinstance(encoded, (bytes, bytearray, memoryview)):
        raise TypeError("encoded must be bytes-like")
    encoded = bytes(encoded)
    prepare_start = time.perf_counter()
    prepared = _prepare(encoded, chunk_size)
    prepare_ms = (time.perf_counter() - prepare_start) * 1000
    count = prepared["count"]
    base_metrics = {"backend": "cuda_chunked_huffman_reference", "elements": count,
                    "container_bytes": len(encoded), "output_bytes": count * 2,
                    "chunk_size": chunk_size, "cpu_validation_and_index_ms": prepare_ms,
                    "performance_claim": "correctness smoke only; not paper decoder throughput"}
    if not count:
        base_metrics.update({"kernel_launched": False, "kernel_ms": 0.0,
                             "end_to_end_ms": (time.perf_counter() - started) * 1000})
        return b"", base_metrics
    library = _load_library()
    info = ctypes.create_string_buffer(1024)
    if library.fluxmoe_cuda_device_info(info, len(info)) != 1:
        raise CudaCodecError(info.value.decode(errors="replace"))
    sm = (ctypes.c_uint8 * count).from_buffer_copy(prepared["sign_mantissa"])
    stream = (ctypes.c_uint8 * len(prepared["stream"])).from_buffer_copy(prepared["stream"])
    offsets = (ctypes.c_uint64 * len(prepared["offsets"]))(*prepared["offsets"])
    nodes = len(prepared["symbols"])
    left = (ctypes.c_int32 * nodes)(*prepared["left"])
    right = (ctypes.c_int32 * nodes)(*prepared["right"])
    symbols = (ctypes.c_int32 * nodes)(*prepared["symbols"])
    output = (ctypes.c_uint16 * count)()
    kernel_ms = ctypes.c_float()
    error = ctypes.create_string_buffer(1024)
    native_started = time.perf_counter()
    status = library.fluxmoe_cuda_decode(
        sm, stream, count, prepared["bit_count"], offsets, len(offsets) - 1,
        chunk_size, left, right, symbols, nodes, output,
        ctypes.byref(kernel_ms), error, len(error),
    )
    native_ms = (time.perf_counter() - native_started) * 1000
    if status:
        raise CudaCodecError(error.value.decode(errors="replace") or f"CUDA decoder failed: {status}")
    result = bytes(output)
    if sys.byteorder != "little":
        result = b"".join(struct.pack("<H", word) for word in output)
    if result != prepared["reference"]:
        raise CudaCodecError("GPU decoder output differs from CPU validated BF16 bytes")
    base_metrics.update({"kernel_launched": True, "kernel_ms": kernel_ms.value,
                         "native_allocate_transfer_kernel_copyback_ms": native_ms,
                         "end_to_end_ms": (time.perf_counter() - started) * 1000,
                         "device": info.value.decode(errors="replace"), "chunks": len(offsets) - 1,
                         "index_bytes": len(offsets) * 8, "decode_tree_bytes": nodes * 12,
                         "gpu_allocated_bytes": count * 3 + len(stream) + len(offsets) * 8 + nodes * 12 + 4,
                         "bitwise_matches_cpu": True})
    return result, base_metrics


def decode_cuda(encoded: bytes) -> bytes:
    """Return byte-exact decoded BF16 data; no silent CPU-only fallback."""
    return decode_cuda_with_metrics(encoded)[0]


def _native_call(function, *arguments) -> None:
    error = ctypes.create_string_buffer(2048)
    status = function(*arguments, error, len(error))
    if status:
        raise CudaCodecError(error.value.decode(errors="replace") or f"CUDA operation failed: {status}")


def _pointer_integer(name: str, value: int, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= ctypes.c_size_t(-1).value:
        raise ValueError(f"{name} must be a pointer-sized integer >= {minimum}")
    return value


class CudaCompressedTensor:
    """Persistent compressed GPU backing with direct asynchronous materialization.

    Construction validates the FMBF container, builds chunk indices on CPU, and
    uploads compressed payload/tree/indices/prefix tables once. The CPU decoded reference is
    discarded after preparation. ``decode_into`` launches directly into a caller
    owned CUDA address on its supplied stream: no CPU decode, allocation,
    transfer, readback, or synchronization occurs in the normal launch path.
    One completion event is allocated when a stream is first used and reused.

    The caller must preserve destination storage and insert WAR/RAW dependencies.
    Distinct streams may concurrently read the same immutable compressed source;
    they must use disjoint destinations or an explicit external ordering. Error
    flags are sticky atomic writes, never reset by concurrent launches. ``check``
    and ``close`` wait for this source's last readers on every registered stream.
    A failed ``check`` poisons the source; later submissions require recreating
    it. No error is cleared by concurrent launches. Call ``check`` at validation
    boundaries before consuming untrusted output; this is not per-launch CRC.
    Device memory excludes caller-owned output buffers; CUDA Graph capture is
    currently rejected explicitly. This class never imports PyTorch.

    preparation_backend='native' uses the optional C++ CPU codec to perform the
    same container validation and chunk indexing on large pretrained tensors.
    The default Python preparation remains unchanged. Both run before timing.
    decoder='packed' adds aligned uint32 input/output transactions and a
    34-column uint16 tile to the previous 'bulk' decoder. 'buffered' caches a
    two-word bit window; 'vectorized' combines it with packed IO. These and
    'tree', 'lookup', 'tiled', 'bulk' remain explicit measurement ablations. Codes
    longer than the 10-bit table prefix use the exact trie fallback. GPU source
    byte accounting includes the tables and eight bytes of lookahead padding.
    """

    def __init__(self, encoded: bytes, device: int = 0, chunk_size: int = 256,
                 preparation_backend: str = "python", decoder: str = "packed"):
        if decoder not in {"tree","lookup","tiled","bulk","buffered","vectorized","packed"}:
            raise ValueError("decoder must be tree, lookup, tiled, bulk, buffered, vectorized or packed")
        if isinstance(device, bool) or not isinstance(device, int) or not 0 <= device < 2**31:
            raise ValueError("device must be a nonnegative CUDA ordinal")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= 65536:
            raise ValueError("chunk_size must be an integer in [1, 65536]")
        if not isinstance(encoded, (bytes, bytearray, memoryview)):
            raise TypeError("encoded must be bytes-like")
        self._lock = threading.RLock()
        self._handle = ctypes.c_void_p()
        self._closed = False
        self._library = None
        self.device = device
        self.decoder = decoder
        self.chunk_size = chunk_size
        started = time.perf_counter()
        encoded = bytes(encoded)
        self.container_sha256 = hashlib.sha256(encoded).hexdigest()
        if preparation_backend == "python":
            prepared = _prepare(encoded, chunk_size)
        elif preparation_backend == "native":
            from .fast_codec import prepare
            prepared = prepare(encoded, chunk_size)
        else:
            raise ValueError("preparation_backend must be python or native")
        self.preparation_backend = preparation_backend
        self.count = prepared["count"]
        self.output_bytes = 2 * self.count
        self.device_bytes = 0
        self.container_bytes = len(encoded)
        self.decode_submissions = 0
        self.preparation_ms = (time.perf_counter() - started) * 1000
        if self.count == 0:
            return
        self._library = _load_library()
        sm = (ctypes.c_uint8 * self.count).from_buffer_copy(prepared["sign_mantissa"])
        stream = (ctypes.c_uint8 * len(prepared["stream"])).from_buffer_copy(prepared["stream"])
        offsets = (ctypes.c_uint64 * len(prepared["offsets"]))(*prepared["offsets"])
        nodes = len(prepared["symbols"])
        left = (ctypes.c_int32 * nodes)(*prepared["left"])
        right = (ctypes.c_int32 * nodes)(*prepared["right"])
        symbols = (ctypes.c_int32 * nodes)(*prepared["symbols"])
        allocated = ctypes.c_size_t()
        _native_call(self._library.fluxmoe_cuda_source_create, device, sm, stream,
                     self.count, prepared["bit_count"], offsets, len(offsets) - 1,
                     chunk_size, left, right, symbols, nodes,
                     ctypes.byref(self._handle), ctypes.byref(allocated))
        self.device_bytes = allocated.value
        _native_call(self._library.fluxmoe_cuda_source_set_decoder,
                     self._handle, {"tree":0,"lookup":1,"tiled":2,"bulk":3,"buffered":4,"vectorized":5,"packed":6}[decoder])
        self.preparation_ms = (time.perf_counter() - started) * 1000

    @property
    def closed(self) -> bool:
        return self._closed

    def _assert_open(self) -> None:
        if self.closed:
            raise CudaCodecError("compressed source is closed")

    def decode_into(self, destination: int, stream: int, output_bytes: int | None = None) -> None:
        """Enqueue BF16 decode into caller-owned device storage; do not wait.

        ``output_bytes`` states the caller's available extent, defaulting to the
        exact required size. The native layer validates pointer device/type, but
        cannot infer the tensor extent inside a larger VMM mapping.
        """
        with self._lock:
            self._assert_open()
            _pointer_integer("destination", destination, allow_zero=self.count == 0)
            _pointer_integer("stream", stream, allow_zero=True)
            if destination % 2:
                raise ValueError("BF16 destination must be two-byte aligned")
            size = self.output_bytes if output_bytes is None else output_bytes
            _pointer_integer("output_bytes", size, allow_zero=True)
            if size < self.output_bytes:
                raise ValueError("output_bytes is smaller than the decoded BF16 tensor")
            if self.count:
                _native_call(self._library.fluxmoe_cuda_source_decode,
                             self._handle, destination, stream, size)
                self.decode_submissions += 1

    def check(self) -> None:
        """At an explicit validation boundary, await readers and check errors."""
        with self._lock:
            self._assert_open()
            if self.count:
                _native_call(self._library.fluxmoe_cuda_source_check, self._handle)

    def close(self) -> None:
        """Wait for every registered stream's final read, then free backing."""
        with self._lock:
            if self.closed:
                return
            self._closed = True
            handle, self._handle = self._handle, ctypes.c_void_p()
            if handle.value:
                _native_call(self._library.fluxmoe_cuda_source_destroy, handle)

    def __enter__(self):
        self._assert_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--nvcc")
    build_parser.add_argument("--output-dir")
    build_parser.add_argument("--architecture")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--elements", type=int, default=65536)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build_cuda_library(args.nvcc, args.output_dir, args.architecture)
            print(json.dumps({"library": str(result), "build_log": str(result.parent / "build.log")}, indent=2))
        else:
            if not 1 <= args.elements <= 2 * 1024 * 1024:
                parser.error("--elements must be in [1, 2097152] for this bounded smoke test")
            raw = b"".join(struct.pack("<H", index % 65536) for index in range(args.elements))
            decoded, metrics = decode_cuda_with_metrics(compress_bf16(raw))
            metrics["bitwise_matches_input"] = decoded == raw
            print(json.dumps(metrics, indent=2))
        return 0
    except (CudaCodecError, ValueError) as exc:
        print(f"CUDA decoder error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
