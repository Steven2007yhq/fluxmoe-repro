"""Persistent-source CUDA tests: direct writes, non-default streams, lifetime."""
import os
import struct
import unittest
import zlib
import copy
from unittest.mock import patch

from fluxmoe.codec import compress_bf16
from fluxmoe.cuda_codec import CudaCodecError, CudaCompressedTensor


class PersistentSourceInputTests(unittest.TestCase):
    def test_invalid_inputs_fail_without_device_work(self):
        encoded = compress_bf16(b"\0\0")
        for kwargs in ({"device": -1}, {"device": True}, {"device": 2**31},
                       {"chunk_size": 0}, {"chunk_size": True}, {"chunk_size": 65537},
                       {"decoder":"unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CudaCompressedTensor(encoded, **kwargs)
        with self.assertRaises(ValueError):
            CudaCompressedTensor(b"corrupt")

    def test_empty_source_requires_no_gpu_and_has_explicit_lifetime(self):
        with patch("fluxmoe.cuda_codec._load_library", side_effect=AssertionError("must not load CUDA")):
            with CudaCompressedTensor(compress_bf16(b"")) as source:
                self.assertEqual(source.device_bytes, 0)
                self.assertEqual(source.output_bytes, 0)
                source.decode_into(0, 0, 0)
                source.check()
            source.close()
            with self.assertRaisesRegex(CudaCodecError, "closed"):
                source.decode_into(0, 0, 0)


@unittest.skipUnless(os.environ.get("FLUXMOE_TEST_CUDA") == "1",
                     "Set FLUXMOE_TEST_CUDA=1 for persistent CUDA decoder tests")
class PersistentSourceCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("persistent CUDA tests explicitly requested without a CUDA device")
        cls.torch = torch

    def assert_words(self, tensor, raw):
        words = tensor.cpu().tolist()
        self.assertEqual(struct.pack(f"<{len(words)}H", *words), raw)

    def test_single_symbol_tree_and_two_byte_aligned_destination(self):
        torch = self.torch
        # >=32 chunks exercises the vector full-warp path; odd output pointer
        # alignment (legal for BF16) must select the scalar store fallback.
        raw = b"\x81\x3f" * 8193
        encoded = compress_bf16(raw)
        self.assertEqual(max(encoded[32:288]), 1)
        for mode in ["tree", "lookup", "tiled", "bulk", "buffered", "vectorized", "packed"]:
            for shift in [0, 1]:
                with self.subTest(mode=mode, shift=shift):
                    output = torch.full((8195,), 0x5a5a, dtype=torch.uint16, device="cuda")
                    with CudaCompressedTensor(encoded, decoder=mode) as source:
                        source.decode_into(output[shift:].data_ptr(), torch.cuda.current_stream().cuda_stream)
                        source.check()
                    self.assert_words(output[shift:shift+8193], raw)
                    self.assertEqual(output[shift+8193:].cpu().tolist(), [0x5a5a] * (2-shift))
                    if shift: self.assertEqual(output[0].item(), 0x5a5a)

    def test_native_rejects_zero_bit_root_leaf(self):
        from fluxmoe.cuda_codec import _prepare
        encoded = compress_bf16(b"\x81\x3f" * 4)
        prepared = _prepare(encoded, 256)
        prepared.update(left=[-1], right=[-1], symbols=[127])
        with patch("fluxmoe.cuda_codec._prepare", return_value=prepared):
            with self.assertRaisesRegex(CudaCodecError, "non-leaf root"):
                CudaCompressedTensor(encoded)

    def test_gpu_malformed_stream_and_offsets_are_sticky_and_poison_source(self):
        from fluxmoe.cuda_codec import _prepare
        torch = self.torch
        encoded = compress_bf16(b"\x81\x3f" * 8193)
        original = _prepare(encoded, 256)
        for mode in ["tree", "lookup", "tiled", "bulk", "buffered", "vectorized", "packed"]:
            for fault in ["invalid_code", "offset_past_payload", "truncated_chunk"]:
                prepared = copy.deepcopy(original)
                if fault == "invalid_code":
                    prepared["stream"] = b"\xff" * len(prepared["stream"])
                elif fault == "offset_past_payload":
                    prepared["offsets"][0] = prepared["bit_count"] + 1
                else:
                    prepared["offsets"][-1] -= 1
                with self.subTest(mode=mode, fault=fault):
                    # Fault injection deliberately bypasses strict CPU checksum
                    # validation to exercise GPU bounds/error paths themselves.
                    with patch("fluxmoe.cuda_codec._prepare", return_value=prepared):
                        source = CudaCompressedTensor(encoded, decoder=mode)
                    with source:
                        output = torch.full((8209,), 0x5a5a, dtype=torch.uint16, device="cuda")
                        stream = torch.cuda.current_stream().cuda_stream
                        source.decode_into(output.data_ptr(), stream)
                        for _ in range(2):
                            with self.assertRaisesRegex(CudaCodecError, "GPU Huffman validation failed"):
                                source.check()
                        with self.assertRaisesRegex(CudaCodecError, "recreate"):
                            source.decode_into(output.data_ptr(), stream)
                        self.assertEqual(output[8193:].cpu().tolist(), [0x5a5a] * 16)

    def test_all_bf16_words_direct_to_nondefault_stream(self):
        torch = self.torch
        raw = struct.pack("<65536H", *range(65536))
        stream = torch.cuda.Stream()
        output = torch.empty(65536, dtype=torch.uint16, device="cuda")
        with CudaCompressedTensor(compress_bf16(raw)) as source:
            with patch("fluxmoe.cuda_codec._prepare", side_effect=AssertionError("hot path CPU decode")):
                source.decode_into(output.data_ptr(), stream.cuda_stream)
            ready = torch.cuda.Event()
            ready.record(stream)
            torch.cuda.current_stream().wait_event(ready)
            self.assert_words(output, raw)
            self.assertEqual(source.decode_submissions, 1)
            self.assertGreater(source.device_bytes, 0)
            source.check()

    def test_all_decoders_match_all_bf16_patterns_and_unaligned_chunk_tails(self):
        torch=self.torch
        from fluxmoe.fast_codec import compress
        all_words=struct.pack("<65536H",*range(65536))
        for mode in ["tree","lookup","tiled","bulk","buffered","vectorized","packed"]:
            for count,chunk in [(65536,256),(1,256),(31,31),(32,32),(33,31),
                                (255,33),(257,256),(8193,257),(65537,65536)]:
                raw=(all_words*2)[:2*count]
                with self.subTest(mode=mode,count=count,chunk=chunk):
                    with CudaCompressedTensor(compress(raw),chunk_size=chunk,
                            preparation_backend="native",decoder=mode) as source:
                        output=torch.full((count+16,),0x5a5a,dtype=torch.uint16,device="cuda")
                        source.decode_into(output.data_ptr(),torch.cuda.current_stream().cuda_stream)
                        source.check();self.assert_words(output[:count],raw)
                        self.assertEqual(output[count:].cpu().tolist(),[0x5a5a]*16)

    def test_lookup_continuation_for_codes_longer_than_ten_bits(self):
        from fluxmoe.fast_codec import compress
        frequencies=[1,1]
        for _ in range(18):frequencies.append(sum(frequencies[-2:]))
        # Fibonacci skew creates long codes; include signs and mantissa changes.
        raw=b"".join(struct.pack("<H",(e<<7)|(e&127)|((e%2)<<15))*n
                     for e,n in enumerate(frequencies))
        encoded=compress(raw)
        self.assertGreater(max(encoded[32:288]),10)
        torch=self.torch
        for mode in ["lookup","tiled","bulk","buffered","vectorized","packed"]:
            for chunk in [1,31,256,257]:
                with self.subTest(mode=mode,chunk=chunk):
                    with CudaCompressedTensor(encoded,chunk_size=chunk,decoder=mode,
                            preparation_backend="native") as source:
                        output=torch.empty(source.count,dtype=torch.uint16,device="cuda")
                        source.decode_into(output.data_ptr(),torch.cuda.current_stream().cuda_stream)
                        source.check();self.assert_words(output,raw)

    def test_longest_255_bit_codes_have_exact_tree_fallback(self):
        from fluxmoe import codec
        lengths=bytes(list(range(1,255))+[255,255])
        codes=codec._canonical_codes(lengths)
        words=[(e<<7)|((e*19)&127)|((e%2)<<15) for e in range(256)]
        raw=struct.pack("<256H",*words)
        bits="".join(f"{codes[e][0]:0{codes[e][1]}b}" for e in range(256))
        padded=bits+"0"*((-len(bits))%8)
        payload=int(padded,2).to_bytes(len(padded)//8,"big")
        sm=bytes(((w>>8)&128)|(w&127) for w in words)
        encoded=bytearray(codec._HEADER.pack(codec.MAGIC,1,0,codec.HEADER_BYTES,
            256,len(bits),zlib.crc32(raw),0,lengths)+sm+payload)
        struct.pack_into("<I",encoded,28,codec._container_crc(encoded))
        self.assertEqual(codec.decompress_bf16(encoded),raw)
        torch=self.torch
        for mode in ["lookup","tiled","bulk","buffered","vectorized","packed"]:
            with CudaCompressedTensor(encoded,chunk_size=31,decoder=mode,
                    preparation_backend="native") as source:
                output=torch.empty(source.count,dtype=torch.uint16,device="cuda")
                source.decode_into(output.data_ptr(),torch.cuda.current_stream().cuda_stream)
                source.check();self.assert_words(output,raw)

    def test_launch_does_not_wait_for_stream_and_event_orders_consumer(self):
        torch = self.torch
        raw = struct.pack("<4096H", *range(4096))
        stream, consume = torch.cuda.Stream(), torch.cuda.Stream()
        output = torch.empty(4096, dtype=torch.uint16, device="cuda")
        with CudaCompressedTensor(compress_bf16(raw)) as source:
            source.decode_into(output.data_ptr(), stream.cuda_stream)
            source.check()  # Warm kernel/module and create stream completion event.
            with torch.cuda.stream(stream):
                torch.cuda._sleep(50_000_000)
            source.decode_into(output.data_ptr(), stream.cuda_stream)
            ready = torch.cuda.Event()
            ready.record(stream)
            self.assertFalse(ready.query(), "decode_into unexpectedly waited for the GPU stream")
            with torch.cuda.stream(consume):
                consume.wait_event(ready)
                copied = output.clone()
            consume.synchronize()
            self.assert_words(copied, raw)

    def test_multiple_streams_and_close_waits_for_final_readers(self):
        torch = self.torch
        raw = struct.pack("<8192H", *[i % 65536 for i in range(8192)])
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        outputs = [torch.empty(8192, dtype=torch.uint16, device="cuda") for _ in streams]
        source = CudaCompressedTensor(compress_bf16(raw), chunk_size=31)
        for index, (stream, output) in enumerate(zip(streams, outputs)):
            with torch.cuda.stream(stream):
                torch.cuda._sleep(3_000_000 * (index + 1))
            source.decode_into(output.data_ptr(), stream.cuda_stream, output.numel() * 2)
            source.decode_into(output.data_ptr(), stream.cuda_stream, output.numel() * 2)
        source.close()  # Must protect all source readers without caller synchronization.
        self.assertTrue(source.closed)
        for output in outputs:
            self.assert_words(output, raw)

    def test_destination_checks_and_close_reject_reuse(self):
        torch = self.torch
        output = torch.empty(16, dtype=torch.uint16, device="cuda")
        source = CudaCompressedTensor(compress_bf16(b"\x81\x7f" * 16))
        stream = torch.cuda.current_stream().cuda_stream
        for pointer, stream_pointer, size in ((0, stream, 32), (output.data_ptr() + 1, stream, 32),
                                             (output.data_ptr(), -1, 32), (output.data_ptr(), stream, 31)):
            with self.subTest(pointer=pointer, size=size), self.assertRaises(ValueError):
                source.decode_into(pointer, stream_pointer, size)
        source.close()
        with self.assertRaises(CudaCodecError):
            source.decode_into(output.data_ptr(), stream)


if __name__ == "__main__":
    unittest.main()
