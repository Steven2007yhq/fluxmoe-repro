"""Reproducible command-line entry points; no implicit package/model downloads."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from .capacity import GPU_PRESETS, MODEL_SPECS, estimate_capacity
from .model import ModelConfig, ToyMoE
from .report import chrome_trace, load_results, render_report
from .runtime import STRATEGIES, SimulationConfig, run_simulation


ROOT = Path(__file__).resolve().parent.parent


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def provenance() -> dict:
    source_hashes = {}
    for folder in ("fluxmoe", "native", "tests", "configs"):
        for path in sorted((ROOT / folder).glob("**/*")):
            if path.is_file() and path.suffix in (".py", ".cu", ".cpp", ".json") and "__pycache__" not in path.parts:
                source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "executable": sys.executable, "platform": platform.platform(), "source_sha256": source_hashes}


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    config = SimulationConfig()
    if args.config:
        config = SimulationConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    overrides = {key: getattr(args, key) for key in ("strategy", "batch", "iterations")
                 if hasattr(args, key) and getattr(args, key) is not None}
    if getattr(args, "no_prefetch", False):
        overrides["prefetch"] = False
    return replace(config, **overrides)


def save_run(output: Path, result: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "result.json", result)
    write_json(output / "trace.json", chrome_trace(result))
    write_json(output / "manifest.json", provenance())
    (output / "report.md").write_text(render_report([result]), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FluxMoE CPU reference and explicitly modeled event simulator")
    commands = parser.add_subparsers(dest="command", required=True)
    simulate = commands.add_parser("simulate", help="run a real small MoE with modeled asynchronous weight paging")
    simulate.add_argument("--config")
    simulate.add_argument("--strategy", choices=STRATEGIES)
    simulate.add_argument("--batch", type=int)
    simulate.add_argument("--iterations", type=int)
    simulate.add_argument("--no-prefetch", action="store_true")
    simulate.add_argument("--output", default="results/simulation")
    sweep = commands.add_parser("sweep", help="compare strategies using identical synthetic weights and inputs")
    sweep.add_argument("--config")
    sweep.add_argument("--batches", type=int, nargs="+", default=[2, 4])
    sweep.add_argument("--output", default="results/sweep")
    report = commands.add_parser("report", help="regenerate a report from saved JSON/JSONL")
    report.add_argument("--input", required=True)
    report.add_argument("--output", required=True)
    validate = commands.add_parser("validate", help="run correctness tests and save raw unittest output")
    validate.add_argument("--output", default="results/validation")
    capacity = commands.add_parser("capacity", help="static large-model capacity estimate, not a deployment guarantee")
    capacity.add_argument("--model", choices=MODEL_SPECS, required=True)
    capacity.add_argument("--gpu", choices=GPU_PRESETS, default="a40")
    capacity.add_argument("--gpus", type=int, default=4)
    capacity.add_argument("--gpu-gb", type=float)
    capacity.add_argument("--cpu-fraction", type=float, default=0.125)
    capacity.add_argument("--kv-gb", type=float, default=0)
    capacity.add_argument("--workspace-gb", type=float, default=0)
    capacity.add_argument("--resident", action="store_true")
    capacity.add_argument("--output")
    reference = commands.add_parser("torch-check", help="optional independent CPU/CUDA PyTorch numerical oracle")
    reference.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    reference.add_argument("--output", default="results/torch_reference.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "simulate":
            result = run_simulation(config_from_args(args))
            result["experiment_id"] = "synthetic-single-rank"
            save_run(Path(args.output), result)
            print(json.dumps({"status": result["status"], "measurement_kind": result["measurement_kind"],
                              "metrics": result.get("metrics"), "output": str(Path(args.output).resolve())}, ensure_ascii=False))
            return 0 if result["status"] == "passed" else 2
        if args.command == "sweep":
            config = config_from_args(args)
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            records = []
            with (output / "runs.jsonl").open("w", encoding="utf-8") as stream:
                for batch in args.batches:
                    for strategy in STRATEGIES:
                        for prefetch in (False, True):
                            run_config = replace(config, batch=batch, remaining_batch=max(batch, config.remaining_batch),
                                                 strategy=strategy, prefetch=prefetch, planner_enabled=False)
                            result = run_simulation(run_config)
                            result["experiment_id"] = f"b{batch}-{strategy}-prefetch{int(prefetch)}"
                            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                            stream.flush()
                            write_json(output / f"{result['experiment_id']}.trace.json", chrome_trace(result))
                            records.append(result)
                            print(result["experiment_id"], result["status"], flush=True)
            write_json(output / "manifest.json", provenance())
            (output / "report.md").write_text(render_report(records), encoding="utf-8")
            return 0
        if args.command == "report":
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(render_report(load_results(args.input)), encoding="utf-8")
            print(str(target.resolve()))
            return 0
        if args.command == "validate":
            output = Path(args.output).resolve()
            output.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                       cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (output / "unittest.log").write_text(completed.stdout, encoding="utf-8")
            write_json(output / "validation.json", {"status": "passed" if completed.returncode == 0 else "failed",
                                                     "returncode": completed.returncode,
                                                     "command": [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                                     "evidence": "unittest.log", "provenance": provenance()})
            print(completed.stdout, end="")
            return completed.returncode
        if args.command == "capacity":
            preset = GPU_PRESETS[args.gpu]
            result = estimate_capacity(args.model, args.gpus, args.gpu_gb if args.gpu_gb is not None else preset["gpu_gb"],
                                       args.cpu_fraction, args.kv_gb, args.workspace_gb, args.resident)
            result["gpu_capacity_source"] = "user override" if args.gpu_gb is not None else preset["source"]
            if args.output:
                write_json(Path(args.output), result)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.command == "torch-check":
            from .torch_reference import verify_torch_reference
            result = verify_torch_reference(ToyMoE(ModelConfig()), device=args.device)
            write_json(Path(args.output), {**result, "provenance": provenance()})
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 0 if result["passed"] else 1
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0
