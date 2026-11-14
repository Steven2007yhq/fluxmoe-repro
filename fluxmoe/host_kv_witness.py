"""Configure and independently audit the vLLM host-KV diagnostic."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

BLOCK_SIZE = 16
LAYER_BYTES = 4 * BLOCK_SIZE * 256 * 2
LAYERS_PER_STAGE = 16
MAX_REQUESTS = 32
PAYLOAD_BYTES = LAYER_BYTES * LAYERS_PER_STAGE
IDENTITY_FIELDS = ("request_id", "salt_sha256", "native_key", "logical_block",
                   "token_sha256", "prompt_tokens", "host_block")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def connector_config(evidence_dir, host_bytes):
    """Use native CPU OffloadingConnector with diagnostic hooks."""
    require(host_bytes > 0, "host KV requires a positive byte cap")
    return {
        "kv_connector": "WitnessOffloadingConnector",
        "kv_connector_module_path": "fluxmoe.host_kv_connector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use": host_bytes,
            "blocks_per_chunk": 1,
            "offload_prompt_only": False,
            "witness_dir": str(Path(evidence_dir).resolve()),
        },
    }


def pending_read_requests():
    """Return request IDs and first query positions for witnessed restores."""
    from vllm.distributed.kv_transfer.kv_transfer_state import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )

    if not has_kv_transfer_group():
        return {}
    pending = getattr(get_kv_transfer_group(), "witness_reads", {})
    return {row["request_id"]: row["query_position"] for row in pending.values()}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


class Journal:
    """Commit each process's records independently."""

    def __init__(self, root, name):
        self.root = root
        self.path = root / (name + ".jsonl")
        with self.path.open("x"):
            pass
        self.index = 0

    def record(self, kind, **fields):
        row = {"event": kind, "index": self.index, "pid": os.getpid(), **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.index += 1
        return row

    def payload(self, job, kind, data):
        require(len(data) == PAYLOAD_BYTES, "wrong witness payload length")
        path = self.root / f"{self.path.stem}.{job}.{kind}.bin"
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return {"path": path.name, "bytes": len(data), "sha256": digest(data)}


def _read_journal(root, name):
    rows = [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines()]
    require(bool(rows) and all(row["index"] == i and row["pid"] == rows[0]["pid"]
                               for i, row in enumerate(rows)), f"invalid journal {name}")
    return rows


def _match(rows, event, **keys):
    matches = [row for row in rows if row["event"] == event
               and all(row.get(key) == value for key, value in keys.items())]
    require(len(matches) == 1, f"expected one {event} record for {keys}")
    return matches[0]


def _payload(root, name, job, kind, record):
    filename = f"{name}.{job}.{kind}.bin"
    require(record["path"] == filename and record["bytes"] == PAYLOAD_BYTES,
            "unexpected witness payload path or size")
    path = root / filename
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == PAYLOAD_BYTES,
            f"missing or malformed witness payload {filename}")
    data = path.read_bytes()
    require(digest(data) == record["sha256"], f"witness payload hash mismatch: {filename}")
    return data


def _audit_scheduler(rows, read):
    store_job, load_job = read["store_job"], read["load_job"]
    stored = _match(rows, "store_planned", job=store_job)
    loaded = _match(rows, "load_planned", job=load_job)
    released = _match(rows, "gpu_released", index=read["release_index"])
    store_done = _match(rows, "transfer_done", job=store_job, direction="store")
    load_done = _match(rows, "transfer_done", job=load_job, direction="load")
    request = _match(rows, "request", request_id=stored["request_id"])
    require(store_job != load_job and
            request["index"] < stored["index"] < store_done["index"] < released["index"] <
            loaded["index"] < load_done["index"] < read["index"],
            "wrong scheduler transfer/release/read order")
    for field in IDENTITY_FIELDS:
        require(stored[field] == loaded[field] == read[field],
                f"store, load and read differ at {field}")
    require(request["salt_sha256"] == stored["salt_sha256"] and
            request["prompt_tokens"] == stored["prompt_tokens"] and
            stored["logical_block"] * BLOCK_SIZE >= stored["prompt_tokens"] and
            read["query_position"] >= (stored["logical_block"] + 1) * BLOCK_SIZE and
            stored["request_id"] == released["request_id"] and
            stored["gpu_block"] == released["gpu_block"] and
            stored["logical_block"] == released["logical_block"] and
            loaded["gpu_block"] == read["gpu_block"] and
            read["block_table"][read["logical_block"]] == loaded["gpu_block"] and
            released["refs_before"] == 1 and released["refs_after"] == 0 and
            released["table_after"] == [] and
            released["free_queue_linked"] is True and len(released["free_queue_neighbors"]) == 2 and
            released["gpu_block"] not in released["free_queue_neighbors"] and
            loaded["preemptions"] == read["preemptions"] == released["preemptions"] and
            store_done["ranks_completed"] == load_done["ranks_completed"] == 4,
            "request identity, GPU release, or decode-history coverage differs")
    return stored, loaded


def _audit_worker(root, rank, rows, pool, read, stored, loaded):
    name = f"rank-{rank}"
    store_job, load_job = read["store_job"], read["load_job"]
    before = _match(rows, "store_source", job=store_job)
    host = _match(rows, "store_done", job=store_job)
    load = _match(rows, "load_source", job=load_job)
    restored = _match(rows, "load_done", job=load_job)
    used = _match(rows, "attention_read", job=load_job)
    require(before["index"] < host["index"] < load["index"] < restored["index"] < used["index"],
            f"wrong rank {rank} transfer/read order")
    require(all(row[field] == read[field] for row in (before, host, load, restored, used)
                for field in IDENTITY_FIELDS) and
            all(row["gpu_block"] == stored["gpu_block"] for row in (before, host)) and
            all(row["gpu_block"] == loaded["gpu_block"] for row in (load, restored, used)),
            f"rank {rank} transfer endpoints or logical identity differ")
    require(all(row["transfer_bytes"] == row["job_blocks"] * PAYLOAD_BYTES
                and row["job_blocks"] > 0 for row in (host, restored)),
            f"rank {rank} native completion has invalid transfer size")
    layer_start = (rank // 2) * LAYERS_PER_STAGE
    layers = list(range(layer_start, layer_start + LAYERS_PER_STAGE))
    require(pool["layers"] == layers and [row["layer"] for row in used["rows"]] == layers
            and used["query_position"] == read["query_position"]
            and used["query_count"] == read["query_count"],
            f"rank {rank} actual attention coverage differs")
    logical = read["logical_block"]
    for row in used["rows"]:
        query_count = row["query_stop"] - row["query_start"]
        table = row["block_table"]
        require(query_count == read["query_count"] > 0
                and row["seq_len"] - query_count == read["query_position"]
                and len(table) == (row["seq_len"] + BLOCK_SIZE - 1) // BLOCK_SIZE
                and 0 <= logical < len(table) and table[logical] == loaded["gpu_block"]
                and table == read["block_table"][:len(table)]
                and len(row["write_slots"]) == query_count
                and all(slot >= 0 and slot // BLOCK_SIZE != loaded["gpu_block"]
                        for slot in row["write_slots"]),
                f"rank {rank} recorded Triton row does not prove a restored history read")
    source_data = _payload(root, name, store_job, "gpu-original", before["payload"])
    captures = (
        (store_job, "host-after-store", host),
        (load_job, "host-before-load", load),
        (load_job, "gpu-restored", restored),
        (load_job, "gpu-attention", used),
    )
    for job, kind, row in captures:
        require(_payload(root, name, job, kind, row["payload"]) == source_data,
                f"rank {rank} raw BF16 bits differ at {kind}")
    return digest(source_data)


def audit_witness(evidence_dir, expected_ranks=(0, 1, 2, 3)):
    """Recompute the same-request swap and actual attention-read proof from disk."""
    root = Path(evidence_dir)
    require(tuple(expected_ranks) == (0, 1, 2, 3), "witness requires four TP2/PP2 ranks")
    sched = _read_journal(root, "scheduler")
    workers = {rank: _read_journal(root, f"rank-{rank}") for rank in expected_ranks}
    pools = [_match(rows, "pool", rank=rank, pp_rank=rank // 2, tp_rank=rank % 2)
             for rank, rows in workers.items()]
    cap = _match(sched, "pool")["host_cap"]
    require(len({pool["uuid"] for pool in pools}) == 4 and all(pool["uuid"] for pool in pools),
            "expected four distinct GPU identities")
    require(len({pool["pid"] for pool in pools} | {sched[0]["pid"]}) == 5,
            "expected separate scheduler and worker processes")
    require(len({pool["host_pool_bytes"] for pool in pools}) == 1
            and len({pool["gpu_blocks"] for pool in pools}) == 1
            and len({pool["cpu_blocks"] for pool in pools}) == 1
            and all(pool["host_pool_bytes"] > 0 and pool["host_pool_bytes"] <= cap
                    and pool["pinned"] is True and pool["page_bytes"] == PAYLOAD_BYTES
                    and pool["cpu_blocks"] * PAYLOAD_BYTES * 4 <= pool["host_pool_bytes"]
                    for pool in pools),
            "invalid actual pinned host-KV pool")
    requests = [row for row in sched if row["event"] == "request"]
    require(0 < len(requests) <= MAX_REQUESTS
            and len({row["request_id"] for row in requests}) == len(requests)
            and len({row["salt_sha256"] for row in requests}) == len(requests),
            "request IDs or salts were reused")
    completed = []
    for read in (row for row in sched if row["event"] == "scheduled_read"):
        stored, loaded = _audit_scheduler(sched, read)
        rank_digests = {
            str(rank): _audit_worker(root, rank, rows, pools[rank], read, stored, loaded)
            for rank, rows in workers.items()
        }
        completed.append({
            **{field: read[field] for field in IDENTITY_FIELDS},
            "store_job": read["store_job"], "load_job": read["load_job"],
            "gpu_before": stored["gpu_block"], "gpu_after": loaded["gpu_block"],
            "query_position": read["query_position"], "query_count": read["query_count"],
            "rank_payload_sha256": rank_digests,
        })
    require(bool(completed), "no complete same-request decode block swap-out/restore witness")
    anchors = [{"request_id": row["request_id"], "input_position": row["query_position"]}
               for row in completed]
    require(len({row["request_id"] for row in anchors}) == len(anchors),
            "multiple first read witnesses for one request")
    return {"status": "SAME_DECODE_BLOCK_BITWISE_PASSED", "witnesses": completed,
            "anchors": anchors, "pools": pools}
