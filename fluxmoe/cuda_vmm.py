"""Optional real CUDA VMM alias arena, built against the CUDA Driver API only.

Fixed CUDA virtual addresses map to two reusable physical buffers. Synchronous
byte-copy methods support primitive checks; optional zero-copy Torch views let
external streams consume the real storage. Writing layer 2 overwrites layer 0's
aliased storage, so the caller must supply WAR/RAW event coordination.

Build once with ``build_cuda_vmm(cuda_include='/path/containing/cuda.h')``.
Torch is imported lazily by tensor(). No automatic GPU fallback, package
download, or context reset occurs.
"""

from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LIBRARY = _ROOT / "build" / "libfluxmoe_vmm.so"
_SIZE_MAX = ctypes.c_size_t(-1).value
_LIBRARIES: dict[str, tuple[Any, str, str]] = {}
_LOAD_LOCK = threading.Lock()


def _integer(name: str, value: int, minimum: int = 0, maximum: int = _SIZE_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_build_record(output: Path, record: dict[str, Any]) -> None:
    metadata = output.with_suffix(".json")
    serialized = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=".vmm-metadata-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(serialized)
    try:
        os.replace(temporary, metadata)
    finally:
        temporary.unlink(missing_ok=True)
    (output.parent / "vmm_build.log").write_text(
        f"started_at_utc: {record['started_at_utc']}\n"
        f"source: {record['source_path']}\nsource_sha256: {record['source_sha256']}\n"
        f"cuda_include: {record['cuda_include']}\n"
        f"compiler --version:\n{record['compiler_version']}\n"
        f"command: {shlex.join(record['command'])}\nreturncode: {record['returncode']}\n"
        f"stdout:\n{record['stdout']}\nstderr:\n{record['stderr']}\n",
        encoding="utf-8",
    )


def build_cuda_vmm(cuda_include: str | os.PathLike[str] | None = None,
                   output_path: str | os.PathLike[str] | None = None,
                   compiler: str | None = None) -> Path:
    """Compile a shared library without initializing CUDA or using the GPU.

    Header discovery accepts FLUXMOE_CUDA_INCLUDE, CUDA_HOME/CUDA_PATH, standard
    toolkit locations and NVIDIA wheel include directories in sys.path.
    The linker needs the installed driver libcuda.so.1 (no CUDA runtime ABI).
    """
    candidates: list[Path] = []
    if cuda_include is not None:
        candidates.append(Path(cuda_include))
    else:
        if os.environ.get("FLUXMOE_CUDA_INCLUDE"):
            candidates.append(Path(os.environ["FLUXMOE_CUDA_INCLUDE"]))
        for variable in ("CUDA_HOME", "CUDA_PATH"):
            if os.environ.get(variable):
                candidates.append(Path(os.environ[variable]) / "include")
        candidates += [Path("/usr/local/cuda/include"), Path("/usr/include")]
        for directory in sys.path:
            if directory:
                candidates.extend(sorted((Path(directory) / "nvidia").glob("*/include")))
    include = next((path.resolve() for path in candidates if (path / "cuda.h").is_file()), None)
    if include is None:
        raise RuntimeError("cuda.h not found; pass cuda_include or set FLUXMOE_CUDA_INCLUDE")
    selected_compiler = compiler or os.environ.get("CXX", "g++")
    executable = shutil.which(selected_compiler)
    if executable is None:
        raise RuntimeError(f"C++ compiler not found: {selected_compiler}")
    output = Path(output_path or _DEFAULT_LIBRARY).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = _ROOT / "native" / "paged_arena.cpp"
    source_sha256 = _sha256(source)
    compiler_info = subprocess.run([executable, "--version"], capture_output=True, text=True,
                                   timeout=10, check=False)
    with tempfile.TemporaryDirectory(prefix="fluxmoe-vmm-build-", dir=output.parent) as directory:
        temporary_output = Path(directory) / "libfluxmoe_vmm.so"
        command = [executable, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
                   "-I", str(include), str(source),
                   "-Wl,-z,defs", "-l:libcuda.so.1", "-o", str(temporary_output)]
        record: dict[str, Any] = {
            "schema_version": 1,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(source), "source_sha256": source_sha256,
            "cuda_include": str(include), "cuda_header_sha256": _sha256(include / "cuda.h"),
            "library_path": str(output), "library_sha256": None,
            "compiler": executable,
            "compiler_version": compiler_info.stdout + compiler_info.stderr,
            "compiler_version_returncode": compiler_info.returncode,
            "command": command,
        }
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
            record.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        except subprocess.TimeoutExpired as exc:
            def decoded(value: str | bytes | None) -> str:
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            record.update(returncode=-1, stdout=decoded(exc.stdout),
                          stderr=decoded(exc.stderr) + "\nCompilation exceeded the 120-second timeout.")
        if record["returncode"] == 0 and _sha256(source) != source_sha256:
            record.update(returncode=-1, stderr=record["stderr"] + "\nSource changed during compilation; rebuild required.")
        if record["returncode"] == 0:
            record["library_sha256"] = _sha256(temporary_output)
            os.replace(temporary_output, output)
        _write_build_record(output, record)
        if record["returncode"]:
            raise RuntimeError(f"CUDA VMM build failed; see {output.parent / 'vmm_build.log'}:\n"
                               f"{record['stdout']}{record['stderr']}")
    return output


def _library() -> Any:
    path = str(Path(os.environ.get("FLUXMOE_VMM_LIBRARY", str(_DEFAULT_LIBRARY))).resolve())
    with _LOAD_LOCK:
        if not Path(path).is_file():
            raise RuntimeError("CUDA VMM library is not built; call build_cuda_vmm(cuda_include=...)")
        metadata_path = Path(path).with_suffix(".json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("CUDA VMM build metadata missing or invalid; rebuild with build_cuda_vmm(...)") from exc
        if not isinstance(metadata, dict) or metadata.get("returncode") != 0:
            raise RuntimeError("CUDA VMM has no successful recorded build; rebuild with build_cuda_vmm(...)")
        source_sha256 = _sha256(_ROOT / "native" / "paged_arena.cpp")
        if metadata.get("source_sha256") != source_sha256:
            raise RuntimeError("CUDA VMM source SHA256 differs from the compiled source; rebuild with build_cuda_vmm(...)")
        library_sha256 = _sha256(Path(path))
        if metadata.get("library_sha256") != library_sha256:
            raise RuntimeError("CUDA VMM library SHA256 differs from its build metadata; rebuild with build_cuda_vmm(...)")
        if path in _LIBRARIES:
            library, loaded_source, loaded_library = _LIBRARIES[path]
            if loaded_source != source_sha256 or loaded_library != library_sha256:
                raise RuntimeError("CUDA VMM was rebuilt after loading; start a new Python process before using it")
            return library
        try:
            library = ctypes.CDLL(path)
        except OSError as exc:
            raise RuntimeError(f"cannot load CUDA VMM library: {exc}") from exc
        size, pointer, error = ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)
        signatures = {
            "fluxmoe_vmm_probe": [ctypes.c_int, ctypes.POINTER(size), error, size],
            "fluxmoe_vmm_create": [size, size, ctypes.c_int, ctypes.POINTER(pointer), error, size],
            "fluxmoe_vmm_info": [pointer, ctypes.POINTER(ctypes.c_uint64), size, ctypes.POINTER(size),
                                  ctypes.POINTER(size), ctypes.POINTER(size), error, size],
            "fluxmoe_vmm_write": [pointer, size, pointer, size, error, size],
            "fluxmoe_vmm_read": [pointer, size, pointer, size, error, size],
            "fluxmoe_vmm_destroy": [pointer, error, size],
        }
        for name, arguments in signatures.items():
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, ctypes.c_int
        _LIBRARIES[path] = (library, source_sha256, library_sha256)
        return library


def _call(function: Any, *arguments: Any) -> None:
    error = ctypes.create_string_buffer(2048)
    if function(*arguments, error, len(error)):
        raise RuntimeError(error.value.decode("utf-8", errors="replace"))


def cuda_vmm_available(device: int = 0) -> tuple[bool, str]:
    """Check the built library, driver, device and VMM support without allocation."""
    _integer("device", device, maximum=2**31 - 1)
    try:
        library = _library()
        granularity = ctypes.c_size_t()
        _call(library.fluxmoe_vmm_probe, device, ctypes.byref(granularity))
    except RuntimeError as exc:
        return False, str(exc)
    return True, f"CUDA VMM supported; minimum granularity {granularity.value} bytes"


def _view_geometry(layer_bytes: int, shape: Any, itemsize: int,
                   offset_bytes: int = 0) -> tuple[tuple[int, ...], int]:
    """Validate a dense, nonempty view without importing Torch or CUDA."""
    if not isinstance(shape, (tuple, list)):
        raise ValueError("shape must be a tuple or list of positive dimensions")
    dimensions = tuple(shape)
    elements = 1
    for dimension in dimensions:
        _integer("shape dimension", dimension, minimum=1, maximum=2**63 - 1)
        if elements > _SIZE_MAX // dimension:
            raise ValueError("shape element count overflows size_t")
        elements *= dimension
    _integer("itemsize", itemsize, minimum=1)
    _integer("offset_bytes", offset_bytes, maximum=layer_bytes)
    if offset_bytes % itemsize:
        raise ValueError("offset_bytes must be aligned to the requested dtype")
    if elements > (layer_bytes - offset_bytes) // itemsize:
        raise ValueError("tensor view exceeds logical layer_bytes")
    return dimensions, elements * itemsize


class _CudaArrayLease:
    """Owner retained by Torch's from_blob storage, including derived views.

    PyTorch's CUDA-array importer Py_INCREFs this object and decrefs it from the
    storage deleter (tensor_numpy.cpp). Export uint8, then reinterpret BF16 in
    Torch so no nonstandard NumPy BF16 typestr is required. The arena does not
    retain this lease, avoiding an arena→lease→arena reference cycle.
    """

    def __init__(self, arena: "CudaPagedArena", pointer: int, size: int):
        self._arena = arena
        self.__cuda_array_interface__ = {
            "shape": (size,), "strides": None, "typestr": "|u1",
            "data": (pointer, False), "version": 3,
            # Storage is initialized when the arena is constructed. Later
            # producers/consumers establish their own CUDA event dependencies.
            "stream": None,
        }
        arena._live_tensor_leases += 1

    def __del__(self) -> None:
        try:
            arena = self._arena
            self._arena = None
            if arena is not None:
                with arena._lock:
                    arena._live_tensor_leases -= 1
        except Exception:
            # Best effort during interpreter teardown; the strong reference
            # protects the native allocation during normal storage lifetimes.
            pass


class CudaPagedArena:
    """Two real physical allocations, stable per-layer CUDA VA, ordinal%2 alias.

    Sizes include driver-required alignment in physical_bytes.  logical_bytes is
    layers*layer_bytes. Address integers are valid until close. The synchronous
    read/write methods serialize their own calls; tensor() leaves asynchronous
    CUDA access ordering to the caller. Torch storages retain an arena lease and
    prevent explicit close until released. Never use detached raw pointers after
    their owning arena/storage lifetime ends.
    """

    def __init__(self, layers: int, layer_bytes: int, device: int = 0):
        _integer("layers", layers, minimum=1)
        _integer("layer_bytes", layer_bytes, minimum=1)
        _integer("device", device, maximum=2**31 - 1)
        if layers > _SIZE_MAX // layer_bytes:
            raise ValueError("logical allocation size overflows size_t")
        self.layers, self.layer_bytes, self.device = layers, layer_bytes, device
        self._lock = threading.RLock()
        self._live_tensor_leases = 0
        self._handle = ctypes.c_void_p()
        self._library = _library()
        _call(self._library.fluxmoe_vmm_create, layers, layer_bytes, device, ctypes.byref(self._handle))
        try:
            addresses = (ctypes.c_uint64 * layers)()
            physical, logical, granularity = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t()
            _call(self._library.fluxmoe_vmm_info, self._handle, addresses, layers,
                  ctypes.byref(physical), ctypes.byref(logical), ctypes.byref(granularity))
            self.addresses = tuple(addresses)
            self.physical_bytes = physical.value
            self.logical_bytes = logical.value
            self.granularity = granularity.value
        except BaseException:
            self.close()
            raise

    @property
    def closed(self) -> bool:
        return not bool(self._handle.value)

    @property
    def live_tensor_leases(self) -> int:
        """Number of externally owned Torch storages, not number of views."""
        with self._lock:
            return self._live_tensor_leases

    def _validate(self, layer: int) -> None:
        if self.closed:
            raise RuntimeError("CUDA VMM arena is closed")
        _integer("layer", layer, maximum=self.layers - 1)

    def write(self, layer: int, data: bytes) -> None:
        """Synchronously replace one complete layer, affecting all its aliases."""
        with self._lock:
            self._validate(layer)
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise TypeError("data must be bytes-like")
            payload = bytes(data)
            if len(payload) != self.layer_bytes:
                raise ValueError("write requires exactly layer_bytes of input")
            buffer = ctypes.create_string_buffer(payload, len(payload))
            _call(self._library.fluxmoe_vmm_write, self._handle, layer, buffer, len(payload))

    def read(self, layer: int, size: int | None = None) -> bytes:
        """Synchronously copy the requested prefix of a layer to host bytes."""
        with self._lock:
            self._validate(layer)
            size = self.layer_bytes if size is None else _integer("size", size, maximum=self.layer_bytes)
            buffer = ctypes.create_string_buffer(size)
            _call(self._library.fluxmoe_vmm_read, self._handle, layer, buffer, size)
            return buffer.raw

    def tensor(self, layer: int, shape: tuple[int, ...], dtype: Any = None,
               *, offset_bytes: int = 0) -> Any:
        """Return a non-owning Torch tensor at the real CUDA VA, without a copy.

        ``dtype=None`` means torch.bfloat16. Shapes are contiguous and nonempty;
        ``()`` is a scalar. BF16/FP16/FP32/FP64, integer and bool storage are
        supported. Zero-sized dimensions, quantized, packed and complex dtypes
        are rejected. The Tensor's storage retains an arena lease; slicing,
        detach and dtype reinterpretation keep that lease alive.

        This method neither loads weights nor inserts waits. Callers must order
        GPU reads/writes, including aliases at layer+2, using their CUDA streams
        and events. The native close operation synchronizes before unmapping,
        and explicit close rejects any still-live Tensor storage.
        """
        with self._lock:
            self._validate(layer)
            # Check dimension/offset types before importing the optional module.
            _view_geometry(self.layer_bytes, shape, 1, offset_bytes)
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("Torch VMM views require the optional PyTorch environment") from exc
            if dtype is None:
                dtype = torch.bfloat16
            supported = {
                torch.bool: 1, torch.uint8: 1, torch.int8: 1,
                torch.bfloat16: 2, torch.float16: 2, torch.int16: 2, torch.uint16: 2,
                torch.float32: 4, torch.int32: 4, torch.uint32: 4,
                torch.float64: 8, torch.int64: 8, torch.uint64: 8,
            }
            if not isinstance(dtype, torch.dtype) or dtype not in supported:
                raise ValueError("unsupported Torch dtype for VMM tensor view")
            dimensions, nbytes = _view_geometry(self.layer_bytes, shape, supported[dtype], offset_bytes)
            pointer = self.addresses[layer] + offset_bytes
            lease = _CudaArrayLease(self, pointer, nbytes)
            # The CUDA array interface importer keeps a reference to lease in
            # its storage deleter; no Python tensor attributes/finalizers needed.
            raw = torch.as_tensor(lease, dtype=torch.uint8, device=torch.device("cuda", self.device))
            value = raw.view(dtype).view(dimensions)
            if value.data_ptr() != pointer or value.device != torch.device("cuda", self.device):
                raise RuntimeError("Torch CUDA array import did not preserve the requested VMM pointer/device")
            return value

    def close(self) -> None:
        with self._lock:
            if not self.closed:
                if self._live_tensor_leases:
                    raise RuntimeError(
                        f"cannot close CUDA VMM arena with {self._live_tensor_leases} live Torch storage lease(s); "
                        "release all tensors and derived views before close"
                    )
                handle = self._handle
                self._handle = ctypes.c_void_p()
                _call(self._library.fluxmoe_vmm_destroy, handle)

    def __enter__(self) -> "CudaPagedArena":
        if self.closed:
            raise RuntimeError("CUDA VMM arena is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Explicit close/context-manager reports failures; interpreter teardown
            # may already have unloaded CUDA and must not raise from a finalizer.
            pass
