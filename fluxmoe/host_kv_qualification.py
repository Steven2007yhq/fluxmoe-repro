"""One four-rank BF16 host-KV qualification and fresh same-layout replay."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import traceback

GIB = 1 << 30
BLOCK_SIZE = 16
PAGE_BYTES_PER_RANK = 512 * 1024  # 16 layers × 4 KV heads × 128 × K/V × BF16 × 16 tokens
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
REVISION = "eba92302a2861cdc0098cc54bc9f17cb2c47eb61"
WORKLOAD_SHA256 = "a4013c4615a9c8f35383c2bff4d2438d06b6319f6258ce27c9e243a2d4c923a1"
SOURCE_CONTRACT_SHA256 = "74707afe16bfc834cbc679c9885e0c49a636c73db450ee085fc0217d9aa738c5"
# Private vLLM interfaces used by this qualification in addition to the prior
# native KV readback. Fingerprints come from the local 0.29.0 source readback.
REPLAY_SOURCES = {
    "v1/worker/gpu/pp_utils.py": "79cd7d52f7c9fd3bc3f3a3bf61aec3dfe72ec6b4938d1b8b11845a0d75ae0ce4",
    "v1/worker/gpu/input_batch.py": "3929c92e42ae90e4410bb4537dcbdcd171a662e44afc1189a9a3e19037a84410",
    "v1/executor/multiproc_executor.py": "3ed0149a8a8e1550c4f062af40e645d1b734e70f99981bc5bf15f0360507648a",
    "model_executor/models/mixtral.py": "4575826e8258e2ed6c8212558e6dba9e94416a3689275d0b836ea44ea0fff307",
    "model_executor/layers/attention/attention.py": "64a1b218f04a7178d9d41c12fc6ae08f6cff648031bf118216019a48d15572c7",
}
CHECKPOINTS = (1, 512, 1024, 2048, 3072, 4096)
PASS = "HOST_KV_LONG_QUALIFICATION_PASSED"
ARMS = ("treatment", "reference")
EVIDENCE_FILES = (
    "preflight.json", "comparison.json", "request-manifest.json", "vllm-source-contract.json",
    *(f"{arm}/{name}.json" for arm in ARMS for name in
      ("arm-result", "outputs", "fences", "installed", "engine-options",
       "captures", "kv-audit", "worker-exits", "admission")),
    *(f"{arm}-{name}.json" for arm in ARMS for name in ("launch", "started", "exit")),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    write_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def write_bytes(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _absolute_path(text):
    path = Path(text)
    require(path.is_absolute(), f"expected an absolute path: {text}")
    return path


def _bound_json(path, expected, label):
    content = _absolute_path(path).read_bytes()
    require(hashlib.sha256(content).hexdigest() == expected, f"{label} SHA256 differs")
    return json.loads(content)


def validate_config(config):
    required = {
        "schema", "gpu_uuids", "checkpoint", "request_manifest", "source_contract",
        "gpu_blocks", "host_kv_bytes", "owned_rss_cap_bytes", "work_cap_bytes",
    }
    require(set(config) == required and config["schema"] == 1, "unknown or missing config fields")
    uuids = config["gpu_uuids"]
    require(isinstance(uuids, list) and len(set(uuids)) == len(uuids) == 4
            and all(re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid)
                    for uuid in uuids), "four ordered NVIDIA GPU UUIDs required")
    for key in ("checkpoint", "request_manifest", "source_contract"):
        _absolute_path(config[key])
    require(type(config["gpu_blocks"]) is int and 267 <= config["gpu_blocks"] <= 4096,
            "GPU KV allocation must hold one complete 4,234-token request and retain batch pressure")
    require(type(config["host_kv_bytes"]) is int
            and 17 * GIB <= config["host_kv_bytes"] <= 18 * GIB
            and config["host_kv_bytes"] % (4 * PAGE_BYTES_PER_RANK) == 0,
            "aggregate host KV pool must be 17–18 GiB and page-aligned")
    require(config["owned_rss_cap_bytes"] == 160 * GIB and config["work_cap_bytes"] == 16 * GIB,
            "incorrect RSS or work-directory cap")


def read_workload(config):
    workload = _bound_json(config["request_manifest"], WORKLOAD_SHA256, "ShareGPT workload")
    requests = workload["requests"]
    require(workload["all_requests_available_initially"] is True
            and len(requests) == workload["request_count"] == 32
            and workload["output_tokens_per_request"] == CHECKPOINTS[-1]
            and workload["total_output_tokens"] == 32 * CHECKPOINTS[-1],
            "expected the frozen 32 × 4096 request manifest")
    for ordinal, request in enumerate(requests):
        tokens = request["prompt_token_ids"]
        require(request["ordinal"] == ordinal and request["request_id"] == f"sharegpt-p32-{ordinal:02d}"
                and len(tokens) == request["prompt_tokens"] and 17 <= len(tokens) <= 138
                and all(type(token) is int and 0 <= token < 32000 for token in tokens)
                and request["output_tokens"] == CHECKPOINTS[-1],
                f"invalid workload row {ordinal}")
    return workload


def check_installation(contract):
    versions = {}
    for name, expected in contract["target_versions"].items():
        actual = metadata.version(name)
        require(actual.split("+")[0] == expected, f"expected {name} {expected}; found {actual}")
        versions[name] = actual
    spec = util.find_spec("vllm")
    require(spec is not None and spec.origin is not None, "vLLM source unavailable")
    root = Path(spec.origin).parent
    for relative, expected in (contract["readback_files"] | REPLAY_SOURCES).items():
        require(digest(root / relative) == expected, f"installed vLLM source differs: {relative}")
    return versions


def checkpoint_inventory(checkpoint, workload, *, verify_shards=False):
    root = Path(checkpoint)
    manifest_path = root / "download_manifest.json"
    record = read_json(manifest_path)
    require(record["repo_id"] == MODEL and record["revision"] == REVISION
            and record["status"] in {"offline_import_verified", "downloaded_verified"},
            "verified offline Mixtral checkpoint required")
    files = record["files"]
    index = read_json(root / "model.safetensors.index.json")["weight_map"]
    shards = {f"model-{i:05d}-of-00019.safetensors" for i in range(1, 20)}
    require(len(index) == 995 and set(index.values()) == shards, "unexpected Mixtral checkpoint index")
    model_config = read_json(root / "config.json")
    geometry = {"model_type": "mixtral", "hidden_size": 4096, "num_hidden_layers": 32,
                "num_attention_heads": 32, "num_key_value_heads": 8,
                "num_local_experts": 8, "vocab_size": 32000}
    require(all(model_config.get(key) == value for key, value in geometry.items()),
            "unexpected Mixtral architecture")
    required_files = shards | {"config.json", "model.safetensors.index.json"}
    for name, expected in workload["tokenizer"]["files"].items():
        require(files[name]["sha256"] == expected, f"tokenizer differs: {name}")
        required_files.add(name)
    if (root / "generation_config.json").exists():
        required_files.add("generation_config.json")
    stats = {}
    for name in sorted(required_files):
        path = root / name
        info = path.stat()
        expected = files[name]
        require(path.is_file() and info.st_size == expected["bytes"]
                and re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]), f"invalid checkpoint file: {name}")
        if verify_shards or name not in shards:
            require(digest(path) == expected["sha256"], f"checkpoint file differs: {name}")
        stats[name] = {"device": info.st_dev, "inode": info.st_ino, "bytes": info.st_size,
                       "mtime_ns": info.st_mtime_ns, "sha256": expected["sha256"]}
    return {"model": MODEL, "revision": REVISION, "manifest_sha256": digest(manifest_path),
            "files": stats}


def _launch_env(config):
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(config["gpu_uuids"]),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "VLLM_USE_SIMPLE_KV_OFFLOAD": "0",
        "VLLM_KV_CACHE_LAYOUT": "LBHNC",
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "20",
        "NCCL_CUMEM_ENABLE": "0",
        "CUDA_MODULE_LOADING": "EAGER",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "27415",
    }


def engine_options(config, connector):
    return {
        "model": config["checkpoint"],
        "tokenizer": config["checkpoint"],
        "trust_remote_code": False,
        "load_format": "safetensors",
        "generation_config": "vllm",
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 2,
        "data_parallel_size": 1,
        "enable_expert_parallel": True,
        "distributed_executor_backend": "mp",
        "disable_custom_all_reduce": True,
        "moe_backend": "triton",
        "attention_config": {"backend": "TRITON_ATTN"},
        "enforce_eager": True,
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "max_model_len": 4256,  # 4,096 outputs plus the longest frozen prompt (138).
        "max_num_batched_tokens": 128,
        "max_num_seqs": 32,
        "block_size": BLOCK_SIZE,
        "num_gpu_blocks_override": config["gpu_blocks"],
        "kv_cache_memory_bytes": config["gpu_blocks"] * PAGE_BYTES_PER_RANK,
        "gpu_memory_utilization": 0.90,
        "kv_transfer_config": connector,
        "shutdown_timeout": 20,
        "seed": 27415,
    }


def generate(engine, requests, sampling, directory):
    """Admit every request before the first in-process EngineCore step."""
    release = time.time()
    expected = {request["request_id"]: request["prompt_token_ids"] for request in requests}
    for request_id, tokens in expected.items():
        salt = f"{WORKLOAD_SHA256}/{request_id}"
        internal_id = engine.add_request(request_id,
                                         {"prompt_token_ids": tokens, "cache_salt": salt},
                                         sampling, arrival_time=release)
        require(internal_id == request_id, "vLLM request ID mapping differs")
    write_json(directory / "admission.json",
               {"request_ids": list(expected), "arrived_together": True,
                "salt_namespace": WORKLOAD_SHA256, "release_unix": release})
    completed = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            if not output.finished:
                continue
            request_id = output.request_id
            require(request_id in expected and request_id not in completed
                    and list(output.prompt_token_ids) == expected[request_id]
                    and len(output.outputs) == 1 and output.outputs[0].index == 0,
                    "output request ownership differs")
            row = output.outputs[0]
            tokens = list(row.token_ids)
            receipt = {"request_id": request_id, "token_ids": tokens,
                       "finish_reason": row.finish_reason, "stop_reason": row.stop_reason}
            write_json(directory / f"{request_id}.json", receipt)
            require(row.finish_reason == "length" and row.stop_reason is None
                    and len(tokens) == CHECKPOINTS[-1]
                    and all(type(token) is int and 0 <= token < 32000 for token in tokens),
                    f"{request_id} did not complete the exact output workload")
            completed[request_id] = receipt
    require(set(completed) == set(expected), "requests did not fully drain")
    return [completed[request_id] for request_id in expected]


def _step_engine_without_pp_queue(engine):
    # With asynchronous PP batches disabled, scheduler transitions see native
    # completion in a fixed order. PP model execution and communication stay on.
    from vllm.v1.engine.core_client import InprocClient
    require(isinstance(engine.engine_core, InprocClient), "in-process scheduler required")
    core = engine.engine_core.engine_core
    require(not core.async_scheduling and not core.batch_queue, "engine already has queued work")
    core.batch_queue = None
    core.batch_queue_size = 1
    core.step_fn = core.step
    return core


def _close_engine(engine, directory):
    executor = engine.model_executor
    workers = [worker.proc for worker in executor.workers]
    try:
        engine.engine_core.shutdown()
    finally:
        for worker in workers:
            worker.join(timeout=5)
        exits = [{"pid": worker.pid, "exitcode": worker.exitcode} for worker in workers]
        write_json(directory / "worker-exits.json", exits)
    require(len(exits) == 4 and all(row["exitcode"] == 0 for row in exits),
            "one or more vLLM workers lacked a clean original-parent exit")


def run_arm(config, work, arm):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from fluxmoe import host_kv_reference as reference
    from fluxmoe import host_kv_witness as witness

    workload = read_workload(config)
    requests = workload["requests"]
    directory = work / arm
    directory.mkdir()
    transfer = directory / "kv"
    transfer.mkdir()
    capture = directory / "capture"
    capture.mkdir()
    anchors = None
    if arm == "reference":
        anchors = read_json(work / "treatment" / "kv-audit.json")["anchors"]
        require(read_json(work / "treatment" / "arm-result.json")["status"] == "arm_completed",
                "reference requires treatment completion and fences")
    llm = None
    try:
        options = engine_options(config, witness.connector_config(transfer, config["host_kv_bytes"]))
        write_json(directory / "engine-options.json", options)
        llm = LLM(**options)
        engine = llm.llm_engine
        core = _step_engine_without_pp_queue(engine)
        installed = llm.collective_rpc(reference.install, args=(
            str(capture), requests, CHECKPOINTS, CHECKPOINTS[-1], anchors))
        write_json(directory / "installed.json", installed)
        sampling = SamplingParams(temperature=0.0, n=1, seed=27415, ignore_eos=True,
                                  min_tokens=0, max_tokens=CHECKPOINTS[-1],
                                  skip_reading_prefix_cache=False, detokenize=False,
                                  output_kind=RequestOutputKind.FINAL_ONLY)
        outputs = generate(engine, requests, sampling, directory)
        write_json(directory / "outputs.json", outputs)
        # EngineCore can still have finished IDs/connector jobs to retire after
        # the frontend has consumed the last token.
        drain_steps = 0
        while core.scheduler.has_requests():
            require(drain_steps < 128, "scheduler or connector failed to drain")
            engine.step()
            drain_steps += 1
        fences = llm.collective_rpc(reference.fence)
        write_json(directory / "fences.json", fences)
        reference.validate_fences(fences, config["gpu_uuids"], initial=installed)
        captures = llm.collective_rpc(reference.finish)
        write_json(directory / "captures.json", captures)
        raw_witness = witness.audit_witness(transfer)
        require(raw_witness["anchors"], "same-request post-restore anchor is missing")
        write_json(directory / "kv-audit.json", raw_witness)
    finally:
        if llm is not None:
            original = sys.exception()
            try:
                _close_engine(llm.llm_engine, directory)
            except Exception as cleanup:
                if original is None:
                    raise
                original.add_note(f"vLLM shutdown also failed: {cleanup!r}")
    write_json(directory / "arm-result.json",
               {"status": "arm_completed", "arm": arm, "requests": len(outputs),
                "output_tokens": sum(len(row["token_ids"]) for row in outputs),
                "drain_steps": drain_steps})


def _run_child(config_path, work, arm, environment):
    argv = [sys.executable, "-B", "-m", "fluxmoe.host_kv_qualification",
            "--config", str(config_path), "--work", str(work), "--arm", arm]
    write_json(work / f"{arm}-launch.json", {"argv": argv, "env_overrides": environment})
    process = subprocess.Popen(argv, env={**os.environ, **environment})
    write_json(work / f"{arm}-started.json", {"pid": process.pid, "started_unix": time.time()})
    code = process.wait()
    write_json(work / f"{arm}-exit.json", {"pid": process.pid, "returncode": code})
    require(code == 0, f"{arm} exited with status {code}")
    receipt = read_json(work / arm / "arm-result.json")
    require(receipt["status"] == "arm_completed" and receipt["requests"] == 32
            and receipt["output_tokens"] == 131072, f"{arm} completion receipt differs")


def run(config_path, work):
    config = read_json(config_path)
    validate_config(config)
    workload = read_workload(config)
    contract = _bound_json(config["source_contract"], SOURCE_CONTRACT_SHA256, "vLLM source contract")
    versions = check_installation(contract)
    checkpoint = checkpoint_inventory(config["checkpoint"], workload, verify_shards=True)
    write_json(work / "preflight.json", {
        "versions": versions,
        "python": sys.version,
        "python_executable": sys.executable,
        "checkpoint": checkpoint,
        "all_checkpoint_files_hashed": True,
        "request_manifest_sha256": WORKLOAD_SHA256,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "replay_source_sha256": REPLAY_SOURCES,
        "config_sha256": digest(config_path),
    })
    for field, name, expected in (
        ("request_manifest", "request-manifest.json", WORKLOAD_SHA256),
        ("source_contract", "vllm-source-contract.json", SOURCE_CONTRACT_SHA256),
    ):
        raw = Path(config[field]).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, f"{field} changed after preflight")
        write_bytes(work / name, raw)
    environment = _launch_env(config)
    for arm in ARMS:
        require(checkpoint_inventory(config["checkpoint"], workload) == checkpoint,
                "checkpoint inventory changed during qualification")
        _run_child(config_path, work, arm, environment)
    require(checkpoint_inventory(config["checkpoint"], workload) == checkpoint,
            "checkpoint inventory changed during replay")
    comparison = audit(work, config)
    write_json(work / "comparison.json", comparison)
    write_json(work / "result.json", {
        "status": PASS, "formal_performance_allowed": False, "full_model_audit_passed": False,
        "config_sha256": digest(config_path), "requests_per_arm": 32,
        "output_tokens_per_arm": 131072, "diagnostic_synchronization": True,
        "evidence": {name: digest(work / name) for name in EVIDENCE_FILES},
    })


def audit(work, config):
    """Recompute numeric and transfer checks directly from the original files."""
    from fluxmoe import host_kv_reference as reference
    from fluxmoe import host_kv_witness as witness

    workload = _bound_json(str(work / "request-manifest.json"), WORKLOAD_SHA256, "recovered workload")
    _bound_json(str(work / "vllm-source-contract.json"), SOURCE_CONTRACT_SHA256, "recovered source contract")
    requests = workload["requests"]
    outputs = {}
    proofs = {}
    for arm in ARMS:
        directory = work / arm
        arm_result = read_json(directory / "arm-result.json")
        require(arm_result["status"] == "arm_completed" and arm_result["requests"] == len(requests)
                and arm_result["output_tokens"] == len(requests) * CHECKPOINTS[-1],
                f"{arm} did not complete the full workload")
        fences = read_json(directory / "fences.json")
        reference.validate_fences(fences, config["gpu_uuids"], initial=read_json(directory / "installed.json"))
        capture_receipts = read_json(directory / "captures.json")
        for fence in fences:
            rank = fence["rank"]
            metadata = read_json(directory / "capture" / f"rank{rank}" / "capture.json")
            receipt = next(row for row in capture_receipts if row["rank"] == rank)
            require(all(fence[field] == metadata[field] == receipt[field] for field in
                        ("rank", "pid", "uuid", "device_index", "tp_rank", "pp_rank", "ep_rank")),
                    f"{arm} capture rank {rank} differs from its completion fence")
        proof = witness.audit_witness(directory / "kv")
        require(proof == read_json(directory / "kv-audit.json"), f"{arm} KV audit differs from raw evidence")
        require([pool["uuid"] for pool in proof["pools"]] == config["gpu_uuids"]
                and all(pool["gpu_blocks"] == config["gpu_blocks"]
                        and pool["host_pool_bytes"] <= config["host_kv_bytes"] for pool in proof["pools"])
                and {row["pid"] for row in proof["pools"]} == {row["pid"] for row in fences},
                f"{arm} KV pool and worker fences are not bound to the frozen device allocation")
        exits = read_json(directory / "worker-exits.json")
        require({row["pid"] for row in exits} == {row["pid"] for row in fences}
                and all(row["exitcode"] == 0 for row in exits)
                and read_json(work / f"{arm}-exit.json")["returncode"] == 0,
                f"{arm} original-parent exit coverage differs")
        rows = read_json(directory / "outputs.json")
        require([row["request_id"] for row in rows] == [request["request_id"] for request in requests],
                f"{arm} output order or coverage differs")
        outputs[arm] = {}
        for row in rows:
            request_id = row["request_id"]
            require(row == read_json(directory / f"{request_id}.json")
                    and row["finish_reason"] == "length" and row["stop_reason"] is None
                    and len(row["token_ids"]) == CHECKPOINTS[-1]
                    and all(type(token) is int and 0 <= token < 32000 for token in row["token_ids"]),
                    f"{arm} output receipt differs for {request_id}")
            outputs[arm][request_id] = row["token_ids"]
        prompt_by_id = {request["request_id"]: request["prompt_token_ids"] for request in requests}
        for block in proof["witnesses"]:
            request_id = block["request_id"]
            prompt = prompt_by_id[request_id]
            history = prompt + outputs[arm][request_id]
            first = block["logical_block"] * BLOCK_SIZE
            tokens = history[first:first + BLOCK_SIZE]
            salt = f"{WORKLOAD_SHA256}/{request_id}".encode()
            require(len(tokens) == BLOCK_SIZE and block["prompt_tokens"] == len(prompt)
                    and block["salt_sha256"] == hashlib.sha256(salt).hexdigest()
                    and block["token_sha256"] == hashlib.sha256(struct.pack("<16i", *tokens)).hexdigest(),
                    f"{arm} KV block differs from the frozen request and actual token history")
        proofs[arm] = proof
    require(outputs["treatment"] == outputs["reference"],
            "complete greedy output history differs from fresh BF16 replay")
    anchors = proofs["treatment"]["anchors"]
    require(sorted(proofs["reference"]["anchors"], key=lambda row: row["request_id"])
            == sorted(anchors, key=lambda row: row["request_id"]),
            "same-request restore anchors differ from the reference run")
    logical_blocks = {}
    for arm in ARMS:
        logical_blocks[arm] = {
            row["request_id"]: {field: row[field] for field in
                                ("logical_block", "query_position", "query_count",
                                 "token_sha256", "rank_payload_sha256")}
            for row in proofs[arm]["witnesses"]
        }
    require(logical_blocks["treatment"] == logical_blocks["reference"],
            "witnessed logical KV blocks differ from the fresh BF16 replay")
    return reference.compare(work / "treatment" / "capture", work / "reference" / "capture",
                             requests, CHECKPOINTS, CHECKPOINTS[-1], anchors,
                             outputs=outputs["treatment"])


def prepare(args):
    inputs = args.inputs.resolve(strict=True)
    config = {
        "schema": 1, "gpu_uuids": args.gpu_uuids, "checkpoint": str(args.checkpoint.resolve()),
        "request_manifest": str(inputs / "REQUEST_MANIFEST.json"),
        "source_contract": str(inputs / "TARGET_KV_CAPABILITY.json"),
        "gpu_blocks": args.gpu_blocks, "host_kv_bytes": 18 * GIB,
        "owned_rss_cap_bytes": 160 * GIB, "work_cap_bytes": 16 * GIB,
    }
    validate_config(config)
    read_workload(config)
    _bound_json(config["source_contract"], SOURCE_CONTRACT_SHA256, "vLLM source contract")
    with args.output.open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")


def main():
    if sys.argv[1:2] == ["prepare"]:
        parser = argparse.ArgumentParser(description="Prepare the fixed long workload configuration.")
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--gpu-uuids", nargs=4, required=True)
        parser.add_argument("--inputs", type=Path, default=Path("configs/host_kv_long"))
        parser.add_argument("--gpu-blocks", type=int, default=4096)
        parser.add_argument("--output", type=Path, required=True)
        prepare(parser.parse_args(sys.argv[2:]))
        return
    if sys.argv[1:2] == ["audit"]:
        parser = argparse.ArgumentParser(description="Recompute the checks on completed raw evidence.")
        parser.add_argument("--work", type=Path, required=True)
        work = parser.parse_args(sys.argv[2:]).work.resolve(strict=True)
        result = read_json(work / "result.json")
        config = read_json(work / "config.json")
        execution = read_json(work / "control" / "execution.json")
        validate_config(config)
        require(execution["status"] == "success"
                and execution["result_sha256"] == digest(work / "result.json")
                and result["status"] == PASS and result["formal_performance_allowed"] is False
                and result["config_sha256"] == digest(work / "config.json")
                and set(result["evidence"]) == set(EVIDENCE_FILES)
                and all(digest(work / name) == expected for name, expected in result["evidence"].items()),
                "qualification result or evidence SHA256 differs")
        comparison = audit(work, config)
        require(comparison == read_json(work / "comparison.json"), "recomputed comparison differs")
        print(json.dumps({"status": PASS, "comparison": comparison}, indent=2, allow_nan=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--arm", choices=("treatment", "reference"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    work = args.work.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    try:
        if args.arm:
            config = read_json(config_path)
            validate_config(config)
            require(all(os.environ.get(k) == v for k, v in _launch_env(config).items()),
                    "arm must use the frozen environment from the supervisor")
            run_arm(config, work, args.arm)
        else:
            run(config_path, work)
    except Exception as exc:
        name = f"{args.arm}-failure.json" if args.arm else "failure.json"
        write_json(work / name, {"status": "failed", "error": repr(exc),
                                 "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
