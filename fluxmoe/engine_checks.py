"""Pure-data checks for the resident/paged vLLM engine correctness experiment.

The run schema is deliberately independent of vLLM and torch::

    {"checkpoint_id": "manifest SHA256 (weights, config, tokenizer)",
     "engine_config": {"same semantic options on both sides": "..."},
     "scenarios": [{"name": "mixed", "sampling": {"temperature": 0,
                    "max_tokens": 16, "ignore_eos": True, "prompt_logprobs": 5},
                    "requests": [serialize_request_output(output), ...]}],
     "kv_evidence": {"allocated_bytes": 67108864, "num_blocks": 1024,
         "prefill_forwards": 2, "decode_forwards": 8,
         "cache_write_checks": 10, "cache_write_failures": 0,
         "history_read_checks": 8, "history_read_failures": 0}}

KV checks must come from actual attention observation. ``cache_write_checks``
counts comparisons of the K/V inputs with their assigned slots after attention.
``history_read_checks`` counts independent numerical attention comparisons using
the historical cache (query length < sequence length). A cache allocation, hash
change, or prefix hit alone cannot supply either numerical check. Diagnostic
hooks synchronize and their run times must not be reported as performance.
"""
from __future__ import annotations

import math
from typing import Any


def _chosen_logprobs(token_ids: list[int], values: Any, *, prompt: bool) -> list[float | None] | None:
    if values is None:
        return None
    if len(values) != len(token_ids):
        raise ValueError("logprobs length does not match token_ids")
    result: list[float | None] = []
    for position, (token_id, row) in enumerate(zip(token_ids, values)):
        if row is None and prompt and position == 0:
            result.append(None)
        elif row is None or token_id not in row:
            raise ValueError(f"chosen token {token_id} is missing at logprob position {position}")
        else:
            item = row[token_id]
            result.append(float(item.logprob if hasattr(item, "logprob") else item))
    return result


def serialize_request_output(output: Any) -> dict[str, Any]:
    """Extract chosen-token probabilities from an n=1 vLLM RequestOutput.

    Request IDs and timing fields are retained only where useful for diagnosis;
    they are not cross-process correctness criteria. No vLLM import is needed.
    """
    if len(output.outputs) != 1:
        raise ValueError("engine correctness checks require exactly one completion per request (n=1)")
    completion = output.outputs[0]
    prompt_ids = list(output.prompt_token_ids)
    token_ids = list(completion.token_ids)
    return {
        "request_id": str(output.request_id),
        "prompt_token_ids": prompt_ids,
        "token_ids": token_ids,
        "text": completion.text,
        "finished": bool(output.finished),
        "logprobs": _chosen_logprobs(token_ids, completion.logprobs, prompt=False),
        "prompt_logprobs": _chosen_logprobs(prompt_ids, output.prompt_logprobs, prompt=True),
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
        "num_cached_tokens": getattr(output, "num_cached_tokens", None),
    }


def _count(value: Any, *, positive: bool = False) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= (1 if positive else 0)


def _valid_tokens(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_count(token) for token in value)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare_results(reference: dict[str, Any], paged: dict[str, Any], *,
                    atol: float = 1e-5, rtol: float = 1e-5,
                    require_prompt_logprobs: bool = True) -> dict[str, Any]:
    """Check exact greedy IDs, stopping, finite logprobs and genuine KV evidence.

    Logprob acceptance is ``abs(paged-reference) <= atol + rtol*abs(reference)``.
    The default is a strict same-kernel paging check, not a general BF16 accuracy
    tolerance. Both errors and raw run artifacts should be preserved on failure.
    Missing evidence fails instead of silently accepting an empty experiment.
    This bounded workload requires positive max_tokens. With ignore_eos=True,
    each completion must reach that budget and terminate with reason "length".
    """
    if not _finite(atol) or not _finite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be finite and nonnegative")
    errors: list[str] = []
    report: dict[str, Any] = {
        "status": "failed", "errors": errors, "atol": atol, "rtol": rtol,
        "compared_requests": 0, "compared_generated_tokens": 0,
        "compared_logprobs": 0, "compared_prompt_logprobs": 0,
        "max_logprob_abs_error": 0.0, "max_prompt_logprob_abs_error": 0.0,
        "kv_evidence_valid": False,
    }
    for label, run in (("reference", reference), ("paged", paged)):
        if not isinstance(run, dict):
            errors.append(f"{label}: run must be a dictionary")
            return report
        if not isinstance(run.get("checkpoint_id"), str) or not run["checkpoint_id"]:
            errors.append(f"{label}: missing checkpoint_id")
        if not isinstance(run.get("engine_config"), dict) or not run["engine_config"]:
            errors.append(f"{label}: missing engine_config")
    for field in ("checkpoint_id", "engine_config"):
        if reference.get(field) != paged.get(field):
            errors.append(f"{field}: resident/paged mismatch")

    def scenarios(run: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
        rows = run.get("scenarios")
        if not isinstance(rows, list) or not rows:
            errors.append(f"{label}: scenarios must be a nonempty list")
            return {}
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"]:
                errors.append(f"{label}: scenario needs a nonempty name")
                continue
            name = row["name"]
            if name in result:
                errors.append(f"{label}: duplicate scenario {name}")
            result[name] = row
        return result

    left, right = scenarios(reference, "reference"), scenarios(paged, "paged")
    if list(left) != list(right):
        errors.append("scenarios: names/order mismatch (cache state depends on execution order)")

    def probabilities(a: Any, b: Any, expected_len: int, path: str, *, prompt: bool) -> None:
        field = "prompt_logprobs" if prompt else "logprobs"
        if prompt and a is None and b is None:
            return  # Prefix-cache scenarios may explicitly disable prompt logprobs.
        if not isinstance(a, list) or not isinstance(b, list) or len(a) != expected_len or len(b) != expected_len:
            errors.append(f"{path}.{field}: missing or incorrect length")
            return
        for index, (x, y) in enumerate(zip(a, b)):
            if prompt and index == 0 and x is None and y is None:
                continue
            if not _finite(x) or not _finite(y):
                errors.append(f"{path}.{field}[{index}]: nonfinite or missing value")
                continue
            error = abs(y - x)
            counter = "compared_prompt_logprobs" if prompt else "compared_logprobs"
            maximum = "max_prompt_logprob_abs_error" if prompt else "max_logprob_abs_error"
            report[counter] += 1
            report[maximum] = max(report[maximum], error)
            if error > atol + rtol * abs(x):
                errors.append(f"{path}.{field}[{index}]: {x!r} != {y!r} within tolerance")

    for name, scenario in left.items():
        if name not in right:
            continue
        other = right[name]
        if scenario.get("sampling") != other.get("sampling"):
            errors.append(f"{name}: sampling mismatch")
        for label, item in (("reference", scenario), ("paged", other)):
            sampling = item.get("sampling", {})
            if not isinstance(sampling, dict):
                errors.append(f"{label}.{name}: sampling must be a dictionary")
                continue
            if sampling.get("temperature") != 0:
                errors.append(f"{label}.{name}: greedy temperature=0 required")
            if not _count(sampling.get("max_tokens"), positive=True):
                errors.append(f"{label}.{name}: a positive integer max_tokens budget is required")
        requests, others = scenario.get("requests"), other.get("requests")
        if not isinstance(requests, list) or not requests or not isinstance(others, list) or not others:
            errors.append(f"{name}: requests must be nonempty lists")
            continue
        if len(requests) != len(others):
            errors.append(f"{name}: request count mismatch")
        for index, (a, b) in enumerate(zip(requests, others)):
            path = f"{name}.requests[{index}]"
            if not isinstance(a, dict) or not isinstance(b, dict):
                errors.append(f"{path}: request must be a dictionary")
                continue
            report["compared_requests"] += 1
            for label, row in (("reference", a), ("paged", b)):
                sampling = scenario.get("sampling") if label == "reference" else other.get("sampling")
                sampling = sampling if isinstance(sampling, dict) else {}
                for field in ("prompt_token_ids", "token_ids"):
                    if not _valid_tokens(row.get(field)):
                        errors.append(f"{label}.{path}.{field}: nonempty token IDs required")
                if row.get("finished") is not True:
                    errors.append(f"{label}.{path}: request did not finish")
                if row.get("finish_reason") not in ("length", "stop"):
                    errors.append(f"{label}.{path}: unsupported/missing finish_reason")
                if "stop_reason" not in row:
                    errors.append(f"{label}.{path}: missing stop_reason")
                budget, generated = sampling.get("max_tokens"), row.get("token_ids")
                if _count(budget, positive=True) and _valid_tokens(generated):
                    if len(generated) > budget:
                        errors.append(f"{label}.{path}: completion exceeds max_tokens budget")
                    if (sampling.get("ignore_eos") is True or row.get("finish_reason") == "length") and len(generated) != budget:
                        errors.append(f"{label}.{path}: completion did not reach max_tokens budget")
                if sampling.get("ignore_eos") is True and row.get("finish_reason") != "length":
                    errors.append(f"{label}.{path}: ignore_eos workload must finish with reason 'length'")
                if sampling.get("prompt_logprobs") is not None and not isinstance(row.get("prompt_logprobs"), list):
                    errors.append(f"{label}.{path}: requested prompt_logprobs are missing")
            for field in ("prompt_token_ids", "token_ids", "finish_reason", "stop_reason"):
                if a.get(field) != b.get(field):
                    errors.append(f"{path}.{field}: mismatch")
            if _valid_tokens(a.get("token_ids")):
                report["compared_generated_tokens"] += len(a["token_ids"])
                probabilities(a.get("logprobs"), b.get("logprobs"), len(a["token_ids"]), path, prompt=False)
            if _valid_tokens(a.get("prompt_token_ids")):
                probabilities(a.get("prompt_logprobs"), b.get("prompt_logprobs"),
                              len(a["prompt_token_ids"]), path, prompt=True)

    if report["compared_logprobs"] == 0:
        errors.append("no generated-token logprobs compared")
    if require_prompt_logprobs and report["compared_prompt_logprobs"] == 0:
        errors.append("no prompt-token logprobs compared")

    before_kv = len(errors)
    for label, run in (("reference", reference), ("paged", paged)):
        evidence = run.get("kv_evidence")
        if not isinstance(evidence, dict):
            errors.append(f"{label}: missing kv_evidence")
            continue
        for field in ("allocated_bytes", "num_blocks", "prefill_forwards", "decode_forwards",
                      "cache_write_checks", "history_read_checks"):
            if not _count(evidence.get(field), positive=True):
                errors.append(f"{label}.kv_evidence.{field}: positive observed count required")
        for field in ("cache_write_failures", "history_read_failures"):
            if not _count(evidence.get(field)) or evidence[field] != 0:
                errors.append(f"{label}.kv_evidence.{field}: zero observed failures required")
    if isinstance(reference.get("kv_evidence"), dict) and isinstance(paged.get("kv_evidence"), dict):
        for field in ("allocated_bytes", "num_blocks"):
            if reference["kv_evidence"].get(field) != paged["kv_evidence"].get(field):
                errors.append(f"kv_evidence.{field}: resident/paged budget mismatch")
    report["kv_evidence_valid"] = len(errors) == before_kv
    report["status"] = "failed" if errors else "passed"
    return report
