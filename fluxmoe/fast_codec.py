"""C++ CPU preparation for large real weights; byte-compatible with codec.py.

No device operations occur here. GPU measurements must exclude this preparation.
The Python codec remains the independent oracle for tests and unusual codebooks.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zlib
from . import codec

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "native/cpu_codec.cpp"
LIBRARY = ROOT / "native/build/libfluxmoe_cpu_codec.so"
_CACHE = None

def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(compiler=None):
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    cmd = [compiler or os.environ.get("CXX", "g++"), "-std=c++17", "-O3", "-Wall", "-Wextra",
           "-Werror", "-shared", "-fPIC", str(SOURCE), "-o", str(LIBRARY)]
    source_hash = _hash(SOURCE)
    p = subprocess.run(cmd, text=True, capture_output=True)
    log = LIBRARY.parent / "cpu_codec_build.log"
    log.write_text(shlex.join(cmd) + "\n" + p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError(f"CPU codec build failed: {log}\n{p.stderr}")
    if source_hash != _hash(SOURCE):
        raise RuntimeError("CPU codec source changed during compilation")
    version = subprocess.run([cmd[0],"--version"],text=True,capture_output=True,check=False).stdout
    LIBRARY.with_suffix(".json").write_text(json.dumps({"command":cmd,"compiler_version":version,
        "source_sha256":source_hash,"artifact_sha256":_hash(LIBRARY)},indent=2)+"\n")
    return LIBRARY

def _library():
    global _CACHE
    if sys.byteorder != "little":
        raise RuntimeError("native preparation currently requires a little-endian host")
    if not LIBRARY.is_file():
        build()
    record = json.loads(LIBRARY.with_suffix(".json").read_text())
    hashes = (_hash(SOURCE), _hash(LIBRARY))
    if hashes != (record["source_sha256"],record["artifact_sha256"]):
        raise RuntimeError("CPU codec source/artifact changed; rebuild and restart Python")
    if _CACHE is not None:
        if _CACHE[0] != hashes:
            raise RuntimeError("CPU codec rebuilt in a running process; restart Python")
        return _CACHE[1]
    lib = ctypes.CDLL(str(LIBRARY))
    ptr = ctypes.c_void_p; u64 = ctypes.c_uint64; u32 = ctypes.c_uint32
    lib.fluxmoe_cpu_pack.argtypes = [ptr,u64,ptr,ptr,ptr,ptr,u64]
    lib.fluxmoe_cpu_unpack.argtypes = [ptr,ptr,u64,u64,ptr,ptr,ptr,u32,u32,ptr,ptr,ptr]
    lib.fluxmoe_cpu_pack.restype = lib.fluxmoe_cpu_unpack.restype = ctypes.c_int
    _CACHE = (hashes,lib)
    return lib

def histograms(raw):
    import numpy as np
    if len(raw) % 2:
        raise ValueError("BF16 bytes must have even length")
    words = np.frombuffer(raw,dtype="<u2")
    exponent = np.zeros(256,dtype=np.int64); mantissa = np.zeros(128,dtype=np.int64)
    for start in range(0,len(words),4*1024*1024):
        part = words[start:start+4*1024*1024]
        exponent += np.bincount((part >> 7) & 255,minlength=256)
        mantissa += np.bincount(part & 127,minlength=128)
    return exponent, mantissa

def compress(raw):
    import numpy as np
    raw = bytes(raw)
    exponent,_ = histograms(raw)
    lengths = codec._code_lengths(exponent.tolist())
    codes = codec._canonical_codes(lengths)
    if max(lengths,default=0) > 63:
        raise ValueError("native pack supports Huffman widths <=63; use Python reference for this codebook")
    n = len(raw)//2; bits = sum(int(exponent[e])*width for e,(_,width) in codes.items())
    code_array = np.zeros(256,dtype=np.uint64)
    for e,(code,_) in codes.items(): code_array[e] = code
    raw_array = np.frombuffer(raw,dtype=np.uint8)
    length_array = np.frombuffer(lengths,dtype=np.uint8)
    sm = np.empty(n,dtype=np.uint8); stream = np.zeros((bits+7)//8,dtype=np.uint8)
    status = _library().fluxmoe_cpu_pack(raw_array.ctypes.data,n,code_array.ctypes.data,
        length_array.ctypes.data,sm.ctypes.data,stream.ctypes.data,len(stream))
    if status: raise ValueError(f"native FMBF encoding error {status}")
    result = bytearray(codec._HEADER.pack(codec.MAGIC,1,0,codec.HEADER_BYTES,n,bits,
        zlib.crc32(raw),0,lengths) + sm.tobytes() + stream.tobytes())
    codec.struct.pack_into("<I",result,28,codec._container_crc(result))
    return bytes(result)

def prepare(encoded,chunk_size=256):
    import numpy as np
    if type(chunk_size) is not int or not 1 <= chunk_size <= 65536:
        raise ValueError("chunk_size must be in [1,65536]")
    encoded = bytes(encoded)
    if len(encoded) < codec.HEADER_BYTES: raise ValueError("truncated FMBF header")
    magic,version,flags,header,n,bits,raw_crc,crc,lengths = codec._HEADER.unpack_from(encoded)
    if (magic,version,flags,header) != (codec.MAGIC,1,0,codec.HEADER_BYTES):
        raise ValueError("unsupported FMBF header")
    if len(encoded) != header+n+(bits+7)//8 or codec._container_crc(encoded) != crc:
        raise ValueError("FMBF length/checksum mismatch")
    codes = codec._canonical_codes(lengths)
    if n == 0:
        if bits or codes or raw_crc: raise ValueError("invalid empty FMBF")
    elif not codes or not n <= bits <= 255*n or len(codes)>n:
        raise ValueError("invalid FMBF codebook/bit count")
    sm = encoded[header:header+n]; stream = encoded[header+n:]
    if bits%8 and stream[-1] & ((1 << (8-bits%8))-1): raise ValueError("nonzero FMBF padding")
    left,right,symbols = [-1],[-1],[-1]
    for symbol,(code,width) in codes.items():
        node = 0
        for bit in range(width-1,-1,-1):
            branch = right if (code >> bit)&1 else left
            if branch[node] == -1:
                branch[node] = len(symbols); left.append(-1);right.append(-1);symbols.append(-1)
            node = branch[node]
        symbols[node] = symbol
    arrays = [np.array(a,dtype=np.int32) for a in (left,right,symbols)]
    output = np.empty(n,dtype=np.uint16)
    offsets = np.empty((n+chunk_size-1)//chunk_size+1,dtype=np.uint64)
    seen = np.zeros(256,dtype=np.uint8)
    sm_array = np.frombuffer(sm,dtype=np.uint8); stream_array = np.frombuffer(stream,dtype=np.uint8)
    status = _library().fluxmoe_cpu_unpack(sm_array.ctypes.data,stream_array.ctypes.data,n,bits,
        *(a.ctypes.data for a in arrays),len(symbols),chunk_size,
        output.ctypes.data,offsets.ctypes.data,seen.ctypes.data)
    if status:
        raise ValueError(f"FMBF native decoding error {status}")
    reference = output.tobytes()
    if zlib.crc32(reference)!=raw_crc or set(np.flatnonzero(seen))!=set(codes):
        raise ValueError(f"FMBF decoded contents/CRC/codebook mismatch (native status {status})")
    return {"reference":reference,"count":n,"bit_count":bits,"sign_mantissa":sm,
        "stream":stream,"offsets":offsets.tolist(),"left":left,"right":right,"symbols":symbols}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["build"])
    p.add_argument("--compiler",default=None)
    args = p.parse_args()
    print(build(args.compiler))
