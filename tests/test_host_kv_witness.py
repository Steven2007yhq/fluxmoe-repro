"""CPU fixtures for ordered host-KV evidence and rejection of false matches."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fluxmoe.host_kv_witness import PAYLOAD_BYTES, Journal, audit_witness, digest


class HostKVWitnessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = dict(
            request_id="request-1", salt_sha256=digest(b"salt"), native_key="45" * 33,
            logical_block=2, token_sha256=digest(b"tokens"), prompt_tokens=29, host_block=9,
        )
        self.data = bytes(range(256)) * (PAYLOAD_BYTES // 256)

    def build(self, *, corrupt=False, cross_request=False, rewritten=False, release=True):
        with patch("fluxmoe.host_kv_witness.os.getpid", return_value=40):
            scheduler = Journal(self.root, "scheduler")
            scheduler.record("pool", host_cap=256 * PAYLOAD_BYTES)
            scheduler.record("request", **{key: self.identity[key] for key in
                                            ("request_id", "salt_sha256", "prompt_tokens")})
            scheduler.record("store_planned", job=15, gpu_block=5, **self.identity)
            scheduler.record("transfer_done", job=15, direction="store", ranks_completed=4)
            released = scheduler.record(
                "gpu_released", request_id="request-1", gpu_block=5, logical_block=2,
                refs_before=1, refs_after=0 if release else 1, preemptions=1, table_after=[],
                free_queue_linked=release, free_queue_neighbors=[4, 6],
            )
            identity = dict(self.identity)
            if cross_request:
                identity["request_id"] = "request-2"
            scheduler.record("load_planned", job=29, gpu_block=19, preemptions=1, **identity)
            scheduler.record("transfer_done", job=29, direction="load", ranks_completed=4)
            scheduler.record(
                "scheduled_read", load_job=29, store_job=15, release_index=released["index"],
                gpu_block=19, preemptions=1, query_position=52, query_count=1,
                block_table=[6, 10, 19, 21], **self.identity,
            )

        for rank in range(4):
            with patch("fluxmoe.host_kv_witness.os.getpid", return_value=rank + 41):
                worker = Journal(self.root, f"rank-{rank}")
                first_layer = rank // 2 * 16
                layers = list(range(first_layer, first_layer + 16))
                worker.record(
                    "pool", rank=rank, pp_rank=rank // 2, tp_rank=rank % 2, uuid=f"GPU-{rank}",
                    gpu_blocks=32, cpu_blocks=64, page_bytes=PAYLOAD_BYTES, pinned=True,
                    host_pool_bytes=256 * PAYLOAD_BYTES, layers=layers,
                )
                for event, job, block, payload_kind in (
                    ("store_source", 15, 5, "gpu-original"),
                    ("store_done", 15, 5, "host-after-store"),
                    ("load_source", 29, 19, "host-before-load"),
                    ("load_done", 29, 19, "gpu-restored"),
                    ("attention_read", 29, 19, "gpu-attention"),
                ):
                    data = self.data
                    if corrupt and rank == 2 and event == "load_done":
                        data = bytes([data[0] ^ 1]) + data[1:]
                    extra = {}
                    if event == "attention_read":
                        extra = dict(query_position=52, query_count=1, rows=[
                            dict(layer=layer, row=0, query_start=0, query_stop=1, seq_len=53,
                                 block_table=[6, 10, 19, 21],
                                 write_slots=[(19 if rewritten else 21) * 16 + 4])
                            for layer in layers
                        ])
                    worker.record(
                        event, job=job, gpu_block=block, job_blocks=1, transfer_bytes=PAYLOAD_BYTES,
                        payload=worker.payload(job, payload_kind, data), **self.identity, **extra,
                    )

    def test_actual_auditor_recomputes_raw_evidence(self):
        self.build()
        result = audit_witness(self.root)
        self.assertEqual(result["anchors"], [{"request_id": "request-1", "input_position": 52}])
        self.assertEqual(result["witnesses"][0]["gpu_after"], 19)
        self.assertEqual(result["witnesses"][0]["rank_payload_sha256"]["3"], digest(self.data))

    def test_one_bit_corruption_with_valid_own_hash(self):
        self.build(corrupt=True)
        with self.assertRaisesRegex(RuntimeError, "raw BF16 bits differ at gpu-restored"):
            audit_witness(self.root)

    def test_unreleased_gpu_block(self):
        self.build(release=False)
        with self.assertRaisesRegex(RuntimeError, "GPU release"):
            audit_witness(self.root)

    def test_cross_request_hit(self):
        self.build(cross_request=True)
        with self.assertRaisesRegex(RuntimeError, "differ at request_id"):
            audit_witness(self.root)

    def test_attention_recomputed_block(self):
        self.build(rewritten=True)
        with self.assertRaisesRegex(RuntimeError, "restored history read"):
            audit_witness(self.root)


if __name__ == "__main__":
    unittest.main()
