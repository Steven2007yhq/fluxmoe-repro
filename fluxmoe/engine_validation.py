"""Real vLLM checkpoint/KV/generation correctness, with isolated GPU runs.

Run with the vLLM environment: ``python -m fluxmoe.engine_validation --output
results/engine_validation``. Diagnostic attention hooks synchronize deliberately;
elapsed times from this program are NOT inference performance measurements.
Use a fresh output directory for new runs. Rechecking legacy JSON without the
new environment provenance reports ``incomplete`` for that additional check;
it does not invalidate the historical numerical observations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback


def engine_config() -> dict:
    return dict(dtype="bfloat16", tensor_parallel_size=1, enforce_eager=True,
                seed=27415, max_model_len=128, max_num_seqs=4,
                max_num_batched_tokens=32, enable_chunked_prefill=True,
                enable_prefix_caching=True, kv_cache_memory_bytes=1048576,
                gpu_memory_utilization=0.4, block_size=16,
                async_scheduling=False, disable_log_stats=False,
                attention_config={"backend": "TRITON_ATTN"},
                kernel_config={"moe_backend": "triton"},
                compilation_config={"mode": 0, "cudagraph_mode": "NONE"})


def workload() -> list[dict]:
    long_prompt = [1] + [4 + (i * 7) % 210 for i in range(94)]
    prefix = [1] + [4 + (i * 11) % 210 for i in range(63)]
    return [
        dict(name="mixed_text", prompts=["hello world", "the quick brown fox jumps over the lazy dog",
                                         "one two three four five"], prompt_logprobs=True, max_tokens=16),
        dict(name="chunked_prefill", prompts=[{"prompt_token_ids": long_prompt},
                                              {"prompt_token_ids": [1, 10, 22, 33, 45]}],
             prompt_logprobs=True, max_tokens=16),
        dict(name="prefix_cold", prompts=[{"prompt_token_ids": prefix}],
             prompt_logprobs=False, max_tokens=16),
        dict(name="prefix_warm", prompts=[{"prompt_token_ids": prefix}],
             prompt_logprobs=False, max_tokens=16),
        dict(name="block_reuse", prompts=[{"prompt_token_ids": [1] + [20 + j] * (8 + j * 9)} for j in range(4)],
             prompt_logprobs=True, max_tokens=16),
    ]


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def compare_execution_environments(reference: dict, paged: dict) -> dict:
    """Require a same-host/device/software A/B pair, not cross-GPU equivalence.

    Older artifacts remain readable: absent provenance is ``incomplete``, not
    a numerical failure. Known conflicting provenance is a failed A/B check.
    """
    fields = ("execution_environment_schema_version", "hostname", "gpu", "gpu_compute_capability", "torch_cuda",
              "versions", "process_environment")
    errors, missing = [], []
    required_versions = ("python", "numpy", "triton", "vllm", "torch", "transformers")
    for label, run in (("resident", reference), ("paged", paged)):
        if type(run.get("execution_environment_schema_version")) is not int or run["execution_environment_schema_version"] != 1:
            missing.append(f"{label}: execution environment schema v1 provenance is unavailable")
        for field in ("hostname", "gpu", "torch_cuda"):
            if not isinstance(run.get(field), str) or not run[field]:
                missing.append(f"{label}: missing execution environment {field}")
        capability = run.get("gpu_compute_capability")
        if (not isinstance(capability, list) or len(capability) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in capability)):
            missing.append(f"{label}: missing/invalid GPU compute capability")
        versions = run.get("versions", {})
        if (not isinstance(versions, dict)
                or any(not isinstance(versions.get(name), str) or not versions[name] for name in required_versions)):
            missing.append(f"{label}: Python/NumPy/Triton/framework versions are required")
        environment = run.get("process_environment", {})
        if (not isinstance(environment, dict)
                or any(name not in environment for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"))):
            missing.append(f"{label}: CUDA device visibility/order provenance is required")
    for field in fields:
        left, right = reference.get(field), paged.get(field)
        if field in ("versions", "process_environment"):
            if isinstance(left, dict) and isinstance(right, dict):
                for name in sorted(left.keys() & right.keys()):
                    if left[name] != right[name]:
                        errors.append(f"execution environment {field}.{name}: resident/paged mismatch; compare each platform separately")
        elif left is not None and right is not None and left != right:
            errors.append(f"execution environment {field}: resident/paged mismatch; compare each platform separately")
    return {"status": "failed" if errors else "incomplete" if missing else "passed",
            "errors": errors, "missing_evidence": missing, "incomplete_metadata": bool(missing),
            "resident": {field: reference.get(field) for field in fields},
            "paged": {field: paged.get(field) for field in fields}}


def run_single(mode: str, model: Path, output: Path) -> dict:
    # Must be set before importing vLLM. These apply only to this child process.
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_NO_USAGE_STATS="1",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", VLLM_USE_FLASHINFER_SAMPLER="0")
    import torch
    from vllm import LLM, SamplingParams
    from .checkpoint import verify_tiny_moe_checkpoint, expected_vllm_expert_manifest
    from .engine_checks import serialize_request_output
    from .integrations.engine import register_engine_loaders
    from .kv_observer import install_kv_observer, kv_observer_status, remove_kv_observer

    manifest = verify_tiny_moe_checkpoint(model)
    register_engine_loaders()
    result = dict(mode=mode, status="running", purpose="correctness_only_random_tiny_mixtral",
                  execution_environment_schema_version=1,
                  checkpoint_id=hashlib.sha256((model / "fixture_manifest.json").read_bytes()).hexdigest(),
                  checkpoint_files=manifest["files"], engine_config=engine_config(), scenarios=[],
                  versions={"python": platform.python_version(), **{
                      name: importlib.metadata.version(name)
                      for name in ("vllm", "torch", "transformers", "numpy", "triton")}},
                  hostname=platform.node(),
                  gpu=torch.cuda.get_device_name(0), torch_cuda=torch.version.cuda,
                  gpu_compute_capability=list(torch.cuda.get_device_capability(0)),
                  workload=workload(), diagnostic_hooks_synchronize=True,
                  process_environment={key: os.environ.get(key) for key in (
                      "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_FLASHINFER_SAMPLER",
                      "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                      "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")})
    llm = None
    observer_installed = False
    try:
        llm = LLM(model=str(model), tokenizer=str(model),
                  load_format="fluxmoe" if mode == "paged" else "fluxmoe_resident",
                  model_loader_extra_config={"cpu_fraction": 0.25, "max_trace_events": 10000},
                  worker_extension_cls="fluxmoe.integrations.engine.FluxMoEWorkerExtension",
                  **engine_config())
        result["initial_worker"] = llm.collective_rpc("fluxmoe_status")[0]
        result["expected_expert_weights"] = expected_vllm_expert_manifest(model)
        result["loaded_expert_weights_match_checkpoint"] = (
            result["initial_worker"]["loading"]["expert_weights"] == result["expected_expert_weights"])
        if not result["loaded_expert_weights_match_checkpoint"]:
            raise AssertionError("vLLM loaded expert weights differ from the on-disk checkpoint")
        result["observer_install"] = llm.collective_rpc(install_kv_observer)[0]
        observer_installed = True
        for case in workload():
            sampling = dict(temperature=0, max_tokens=case["max_tokens"], ignore_eos=True,
                            logprobs=5, prompt_logprobs=5 if case["prompt_logprobs"] else None)
            outputs = llm.generate(case["prompts"], SamplingParams(**sampling), use_tqdm=False)
            result["scenarios"].append(dict(name=case["name"], sampling=sampling,
                                            requests=[serialize_request_output(row) for row in outputs]))
            _save(output, result)
        torch.cuda.synchronize()
        result["kv_evidence"] = llm.collective_rpc(kv_observer_status)[0]
        llm.collective_rpc(remove_kv_observer)
        observer_installed = False
        # Run a known workload after removing every synchronizing attention hook.
        # Prompt logprobs force the same full prefill despite the warm prefix cache.
        case = workload()[0]
        sampling = result["scenarios"][0]["sampling"]
        outputs = llm.generate(case["prompts"], SamplingParams(**sampling), use_tqdm=False)
        result["uninstrumented_scenario"] = dict(name=case["name"], sampling=sampling,
            requests=[serialize_request_output(row) for row in outputs])
        result["final_worker"] = llm.collective_rpc("fluxmoe_status", kwargs={"include_trace": True})[0]
        kv = result["final_worker"]["kv_cache"]
        result["kv_evidence"].update(allocated_bytes=kv["tensor_bytes"], num_blocks=kv["num_blocks"])
        result["kv_evidence"]["allocation_stable"] = (result["initial_worker"]["kv_cache"]["tensors"] == kv["tensors"])
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if llm is not None:
            actions = []
            if observer_installed:
                actions.append(("observer_remove", lambda: llm.collective_rpc(remove_kv_observer)))
            actions.extend([("teardown", lambda: llm.collective_rpc("fluxmoe_close")[0]),
                            ("engine_shutdown", lambda: llm.llm_engine.engine_core.shutdown())])
            for label, action in actions:
                try:
                    result[label] = action()
                except BaseException as exc:
                    result[label + "_error"] = repr(exc)
                    result["status"] = "failed"
        _save(output, result)
    if result["status"] != "completed":
        raise RuntimeError(f"engine teardown failed; see {output}")
    return result


def compare_saved(directory: Path) -> dict:
    from .engine_checks import compare_results
    reference, paged = [json.loads((directory / f"{mode}.json").read_text()) for mode in ("resident", "paged")]
    report = compare_results(reference, paged)
    report["execution_environment"] = compare_execution_environments(reference, paged)
    report["errors"].extend(report["execution_environment"]["errors"])
    report["incomplete_reasons"] = report["execution_environment"]["missing_evidence"]
    report["coverage"] = {}
    report["uninstrumented_checks"] = {}
    for mode, run in (("resident", reference), ("paged", paged)):
        if run.get("status") != "completed" or not run.get("teardown", {}).get("closed"):
            report["errors"].append(f"{mode}: engine run or clean teardown failed")
        if not run.get("kv_evidence", {}).get("allocation_stable"):
            report["errors"].append(f"{mode}: KV allocation changed during generation")
        if not run.get("loaded_expert_weights_match_checkpoint"):
            report["errors"].append(f"{mode}: loaded experts do not match on-disk checkpoint")
        forwards = run.get("final_worker", {}).get("forward_counts", {})
        if forwards.get("failed", 0) != 0 or forwards.get("started", 0) <= 0 or forwards.get("started") != forwards.get("completed"):
            report["errors"].append(f"{mode}: incomplete or failed root model forwards")
        cases = {case["name"]: case for case in run.get("scenarios", [])}
        expected = workload()
        if run.get("workload") != expected or [case.get("name") for case in run.get("scenarios", [])] != [case["name"] for case in expected]:
            report["errors"].append(f"{mode}: prescribed workload is missing or changed")
        for case in expected:
            actual = cases.get(case["name"], {})
            if len(actual.get("requests", [])) != len(case["prompts"]):
                report["errors"].append(f"{mode}.{case['name']}: prescribed request count mismatch")
            expected_sampling = dict(temperature=0, max_tokens=case["max_tokens"], ignore_eos=True,
                                     logprobs=5, prompt_logprobs=5 if case["prompt_logprobs"] else None)
            if actual.get("sampling") != expected_sampling:
                report["errors"].append(f"{mode}.{case['name']}: prescribed sampling mismatch")
        cold = cases.get("prefix_cold", {}).get("requests", [{}])[0]
        warm = cases.get("prefix_warm", {}).get("requests", [{}])[0]
        kv = run.get("kv_evidence", {})
        coverage = dict(chunked_prefill=kv.get("chunked_prefill_forwards", 0) > 0,
                        reused_cache_slots=kv.get("rewritten_slots", 0) > 0,
                        prefix_cache_hit=(warm.get("num_cached_tokens") or 0) > 0,
                        cold_warm_tokens_equal=bool(cold.get("token_ids")) and cold.get("token_ids") == warm.get("token_ids"),
                        all_history_slots_observed=kv.get("history_missing_slots") == 0)
        report["coverage"][mode] = coverage
        for key, observed in coverage.items():
            if not observed:
                report["errors"].append(f"{mode}: required workload coverage missing: {key}")
        original = dict(run, scenarios=[cases["mixed_text"]] if "mixed_text" in cases else [])
        uninstrumented = dict(run, scenarios=[run["uninstrumented_scenario"]] if "uninstrumented_scenario" in run else [])
        check = compare_results(original, uninstrumented)
        report["uninstrumented_checks"][mode] = check
        if check["status"] != "passed":
            report["errors"].append(f"{mode}: generation without KV diagnostic hooks failed comparison")
    weights = [run.get("final_worker", {}).get("loading", {}).get("expert_weights") for run in (reference, paged)]
    report["loaded_expert_weights_equal"] = bool(weights[0]) and weights[0] == weights[1]
    if not report["loaded_expert_weights_equal"]:
        report["errors"].append("loaded expert weight manifests differ or are missing")
    counters = paged.get("final_worker", {}).get("pager_counters", {})
    adapter = paged.get("final_worker", {}).get("adapter", {})
    if adapter.get("attached") is not True or adapter.get("failed") is not False:
        report["errors"].append("paged: adapter was detached or failed during inference")
    report["pager_counters"] = counters
    for field in ("decode_launches", "h2d_copies", "raw_waits", "war_waits", "iterations"):
        if counters.get(field, 0) <= 0:
            report["errors"].append(f"paged: no actual {field} observed")
    if counters.get("iterations") != paged.get("final_worker", {}).get("forward_counts", {}).get("completed"):
        report["errors"].append("paged: root forward count does not match completed paging iterations")
    report["status"] = "failed" if report["errors"] else "incomplete" if report["incomplete_reasons"] else "passed"
    report["scope"] = (f"single GPU per run (resident: {reference.get('gpu', 'unknown')}; "
                       f"paged: {paged.get('gpu', 'unknown')}), random tiny Mixtral, real eager LLM.generate; "
                       "not model quality, throughput or cross-platform numerical equivalence")
    _save(directory / "comparison.json", report)
    _write_report(directory, reference, paged, report)
    return report


def _write_report(directory: Path, reference: dict, paged: dict, report: dict) -> None:
    """Generate the human report from the same raw JSON as the machine checks."""
    rows = ["# vLLM 端到端正确性验证", "",
            f"判定：**{report['status']}**。随机微型 Mixtral；每次运行使用单卡。", "",
            "| 模式 | GPU | vLLM / PyTorch | Python / NumPy / Triton |", "| --- | --- | --- | --- |"]
    for mode, run in (("resident", reference), ("paged", paged)):
        versions = run.get("versions", {})
        rows.append(f"| {mode} | {run.get('gpu', 'unknown')} | "
                    f"{versions.get('vllm', 'unknown')} / {versions.get('torch', 'unknown')} | "
                    f"{versions.get('python', 'unknown')} / {versions.get('numpy', 'unknown')} / "
                    f"{versions.get('triton', 'unknown')} |")
    rows += ["", "设备、主机与软件环境须在两侧一致；不同平台的验证结果单独报告。", "",
            "| 对照指标 | 实测值 |", "| --- | --- |",
            f"| 请求 / 生成 token | {report['compared_requests']} / {report['compared_generated_tokens']} |",
            f"| 生成 / prompt logprob 核对数 | {report['compared_logprobs']} / {report['compared_prompt_logprobs']} |",
            f"| 最大生成 / prompt logprob 差值 | {report['max_logprob_abs_error']} / {report['max_prompt_logprob_abs_error']} |", "",
            "| 模式 | KV 写入核对 / 失败 | 历史 attention 核对 / 失败 | 移除诊断钩子后生成 token |", "| --- | --- | --- | --- |"]
    for mode, run in (("resident", reference), ("paged", paged)):
        kv = run.get("kv_evidence", {})
        normal = report["uninstrumented_checks"].get(mode, {})
        rows.append(f"| {mode} | {kv.get('cache_write_checks')} / {kv.get('cache_write_failures')} | "
                    f"{kv.get('history_read_checks')} / {kv.get('history_read_failures')} | {normal.get('compared_generated_tokens')} |")
    rows += ["", f"实际分页计数：`{json.dumps(report['pager_counters'], sort_keys=True)}`。", "",
             "KV 独立数值核验覆盖首层 attention；历史输出采用 CPU FP32 SDPA，容差为 atol=1e-3、rtol=1e-2。"
             "缓存分配、前缀命中、分块 prefill 与槽复用的原始证据在两侧 JSON 中。",
             "", "范围：验证磁盘加载、真实 KV 和生成一致性。随机权重不评估文本质量；"
             "诊断过程包含同步，不报告性能。初次加载仍需完整模型驻留，未验证大模型或多卡容量。", "",
             "原始文件：[resident.json](resident.json)、[paged.json](paged.json)、[comparison.json](comparison.json)。", "",
             "重算报告：`python -m fluxmoe.engine_validation --compare-only --output " + str(directory) + "`", ""]
    if report["errors"]:
        rows += ["失败项：", ""] + ["- " + error for error in report["errors"]]
    if report.get("incomplete_reasons"):
        rows += ["", "环境证据不完整：旧实验的原始数值结论不因此改判失败；"
                 "新增的同机环境核验尚不能完成。新入口需使用新的输出目录重新采集。", ""]
        rows += ["- " + reason for reason in report["incomplete_reasons"]]
    (directory / "report.md").write_text("\n".join(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/tiny_mixtral"))
    parser.add_argument("--output", type=Path, default=Path("results/engine_validation"))
    parser.add_argument("--single", choices=("resident", "paged"))
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    if args.single:
        result = run_single(args.single, args.model.resolve(), args.output)
        print(json.dumps({"status": result["status"], "mode": args.single, "output": str(args.output)}))
        return
    if not args.compare_only:
        from .checkpoint import build_tiny_moe_checkpoint
        build_tiny_moe_checkpoint(args.model)
        args.output.mkdir(parents=True, exist_ok=True)
        # Never replace prior runs: retain failed attempts and their full logs.
        for mode in ("resident", "paged"):
            output = args.output / f"{mode}.json"
            log = args.output / f"{mode}.log"
            if output.exists() or log.exists():
                raise FileExistsError(f"Use a new output directory to retain existing evidence: {args.output}")
            command = [sys.executable, "-m", "fluxmoe.engine_validation", "--single", mode,
                       "--model", str(args.model.resolve()), "--output", str(output.resolve())]
            with log.open("w") as handle:
                child = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
            if child.returncode:
                raise RuntimeError(f"{mode} failed ({child.returncode}); see {log}")
    report = compare_saved(args.output)
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
