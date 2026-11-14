"""Real BF16 weight distributions and steady-state two-stream decoder measures.

plan/fetch/analyze are CPU-only. benchmark is explicitly run by the operator on
one GPU. Logical EP shards are measured sequentially, not distributed inference.
No model is instantiated and no checkpoint dtype conversion is permitted.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

MODELS = {
    "mixtral": {"repo":"mistralai/Mixtral-8x7B-Instruct-v0.1", "degree":2, "figure_layer":0, "paper_layers":32, "experts":8},
    "qwen": {"repo":"Qwen/Qwen3-Next-80B-A3B-Instruct", "degree":4, "figure_layer":20, "paper_layers":48, "experts":512},
    "glm": {"repo":"zai-org/GLM-4.5", "degree":8, "figure_layer":3, "paper_layers":89, "experts":160},
}
PATTERN = re.compile(r"model\.layers\.(\d+)\.(?:block_sparse_moe|mlp)\.experts\.(\d+)\."
                     r"(w1|w2|w3|gate_proj|up_proj|down_proj)\.weight")
KINDS = {"w1":"gate", "w2":"down", "w3":"up", "gate_proj":"gate", "up_proj":"up", "down_proj":"down"}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(8*1024*1024),b""): h.update(part)
    return h.hexdigest()

def save(path,value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")

def fresh(path):
    path = Path(path)
    path.mkdir(parents=True,exist_ok=False)
    return path

def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or not p.parts:
        raise ValueError(f"unsafe checkpoint filename: {name}")
    return name

def groups(weight_map):
    result = defaultdict(lambda:defaultdict(dict))
    for name in weight_map:
        m = PATTERN.fullmatch(name)
        if m:
            layer,expert,kind = m.groups()
            kind = KINDS[kind]
            if kind in result[int(layer)][int(expert)]: raise ValueError("duplicate expert projection")
            result[int(layer)][int(expert)][kind] = name
    if not result: raise ValueError("no recognized routed expert tensors in this checkpoint")
    for layer,experts in result.items():
        if sorted(experts) != list(range(len(experts))): raise ValueError(f"non-contiguous expert IDs in layer {layer}")
        if any(set(tensors)!={"gate","up","down"} for tensors in experts.values()):
            raise ValueError(f"incomplete expert tensors in layer {layer}")
    return dict(result)

def select_layers(text,available):
    available = sorted(int(i) for i in available)
    selected = available if text=="all" else sorted(set(int(i) for i in text.split(",")))
    if not selected or not set(selected)<=set(available): raise ValueError(f"layers must be drawn from {available}")
    return selected

def main_model_groups(weight_map,config):
    result=groups(weight_map)
    n=config.get("num_hidden_layers")
    if type(n) is not int or n<=0:raise ValueError("checkpoint num_hidden_layers is missing/invalid")
    # GLM includes a next-token-prediction module at model.layers.92. It is not
    # one of the 89 MoE layers in the 92-layer main model (layers 3..91).
    excluded=sorted(i for i in result if i>=n)
    return {i:g for i,g in result.items() if i<n},excluded

def make_plan(model,layers,output,revision="main"):
    from huggingface_hub import HfApi,hf_hub_download
    if Path(output).exists(): raise FileExistsError(output)
    spec = MODELS[model]
    info = HfApi().model_info(spec["repo"],revision=revision,files_metadata=True)
    commit = info.sha
    if not re.fullmatch(r"[0-9a-f]{40}",commit): raise ValueError("Hub did not return an immutable commit SHA")
    index_path = hf_hub_download(spec["repo"],"model.safetensors.index.json",revision=commit)
    config_path = hf_hub_download(spec["repo"],"config.json",revision=commit)
    index = json.loads(Path(index_path).read_text())
    config=json.loads(Path(config_path).read_text())
    all_groups,excluded = main_model_groups(index["weight_map"],config)
    selected = select_layers(layers,all_groups)
    if len(all_groups)!=spec["paper_layers"] or any(len(all_groups[i])!=spec["experts"] for i in selected):
        raise ValueError("checkpoint MoE structure differs from the selected paper model")
    names = [name for layer in selected for tensors in all_groups[layer].values() for name in tensors.values()]
    weight_map = {name:safe_name(index["weight_map"][name]) for name in names}
    files = {s.rfilename:s for s in info.siblings}
    shards = []
    for name in sorted(set(weight_map.values())):
        s = files[name]
        if not isinstance(s.size,int) or s.size<=0: raise ValueError(f"missing file size: {name}")
        lfs = s.lfs
        expected = lfs.get("sha256") if isinstance(lfs,dict) else getattr(lfs,"sha256",None)
        shards.append({"name":name,"bytes":s.size,"expected_sha256":expected})
    result = {"schema":1,"model":model,"repo_id":spec["repo"],"revision":commit,
        "selected_layers":selected,"available_layers":sorted(all_groups),
        "logical_ep_degree":spec["degree"],"weight_map":weight_map,"shards":shards,
        "required_download_bytes":sum(s["bytes"] for s in shards),
        "config":config,"excluded_auxiliary_expert_layers":excluded,"status":"planned_not_downloaded",
        "origin":"huggingface_pretrained",
        "note":"Whole safetensors shards covering selected tensors; not the whole model unless --layers all. No weights downloaded by plan."}
    save(output,result)
    print(json.dumps({k:result[k] for k in ["repo_id","revision","selected_layers","required_download_bytes","shards"]},indent=2),flush=True)
    return result

def fetch(plan_path,directory,max_download_gb):
    from huggingface_hub import hf_hub_download
    plan = json.loads(Path(plan_path).read_text())
    if plan["repo_id"] != MODELS[plan["model"]]["repo"] or not re.fullmatch(r"[0-9a-f]{40}",plan["revision"]):
        raise ValueError("unrecognized model/revision")
    names=[safe_name(s["name"]) for s in plan["shards"]]
    if (len(set(names))!=len(names) or set(names)!=set(plan["weight_map"].values())
            or any(type(s["bytes"]) is not int or s["bytes"]<=0 for s in plan["shards"])
            or sum(s["bytes"] for s in plan["shards"])!=plan["required_download_bytes"]):
        raise ValueError("inconsistent shard names/sizes/download total in plan")
    if not math.isfinite(max_download_gb) or max_download_gb<=0 or plan["required_download_bytes"]>max_download_gb*1e9:
        raise ValueError("download exceeds --max-download-gb; inspect plan and choose an explicit adequate cap")
    directory = Path(directory).expanduser().resolve()
    if not directory.parent.is_dir() or not os.access(directory.parent,os.W_OK):
        raise PermissionError(f"Create/provide a writable data parent first: {directory.parent}")
    directory.mkdir(exist_ok=True)
    manifest_path = directory/"download_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old.get(k)!=plan[k] for k in ["repo_id","revision","weight_map"]):
            raise FileExistsError("data directory belongs to a different plan; use a new directory")
    needed = sum(s["bytes"] for s in plan["shards"] if not (directory/safe_name(s["name"])).is_file())
    if shutil.disk_usage(directory).free < needed + max(1024**3,int(needed*.05)):
        raise OSError("insufficient disk space for planned shards plus download margin")
    result = dict(plan,status="downloading",files={})
    save(manifest_path,result)
    for name in ["config.json","model.safetensors.index.json"]+[s["name"] for s in plan["shards"]]:
        safe_name(name)
        print(f"Fetching {name} at {plan['revision']}",flush=True)
        p = Path(hf_hub_download(plan["repo_id"],name,revision=plan["revision"],local_dir=directory))
        if not p.resolve().is_relative_to(directory): raise ValueError("download escaped data directory")
        digest = sha256(p)
        expected = next((s for s in plan["shards"] if s["name"]==name),None)
        if expected and (p.stat().st_size!=expected["bytes"] or (expected["expected_sha256"] and expected["expected_sha256"]!=digest)):
            raise ValueError(f"download size/SHA256 mismatch: {name}")
        result["files"][name] = {"sha256":digest,"bytes":p.stat().st_size}
        save(manifest_path,result)
    result["status"] = "downloaded_verified";save(manifest_path,result)
    print(f"Verified manifest: {manifest_path}",flush=True)

def load_manifest(path):
    path = Path(path).resolve(); data = json.loads(path.read_text())
    if data.get("status")!="downloaded_verified": raise ValueError("checkpoint download not complete/verified")
    if data.get("origin") not in {"huggingface_pretrained","unit_test_fixture"}:
        raise ValueError("weight origin not recorded")
    for name,entry in data["files"].items():
        p = path.parent/safe_name(name)
        if not p.resolve().is_relative_to(path.parent) or p.stat().st_size!=entry["bytes"] or sha256(p)!=entry["sha256"]:
            raise ValueError(f"checkpoint file integrity mismatch: {name}")
    index = json.loads((path.parent/"model.safetensors.index.json").read_text())["weight_map"]
    for name,shard in data["weight_map"].items():
        if index.get(name)!=shard or shard not in data["files"]:
            raise ValueError("selected tensor is not covered by the verified checkpoint index/files")
    return data,path.parent

def tensor_bytes(directory,manifest,name):
    import torch
    from safetensors import safe_open
    path = directory/safe_name(manifest["weight_map"][name])
    with safe_open(path,framework="pt",device="cpu") as f:
        tensor = f.get_tensor(name)
        if tensor.dtype != torch.bfloat16: raise ValueError(f"{name}: expected original BF16, got {tensor.dtype}; no conversion allowed")
        if tensor.ndim!=2 or min(tensor.shape)<=0: raise ValueError(f"invalid expert matrix: {name}")
        shape = list(tensor.shape)
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return raw,shape

def expert_bytes(directory,manifest,tensors):
    gate,gs = tensor_bytes(directory,manifest,tensors["gate"])
    up,us = tensor_bytes(directory,manifest,tensors["up"])
    down,ds = tensor_bytes(directory,manifest,tensors["down"])
    if gs!=us or ds!=gs[::-1]: raise ValueError("gate/up/down expert shape mismatch")
    return gate+up,down,gs

def entropy(counts):
    n = sum(counts)
    return -sum((c/n)*math.log2(c/n) for c in counts if c) if n else 0.0

def analyze(manifest_path,layers,output):
    import numpy as np
    from .fast_codec import histograms
    manifest,directory = load_manifest(manifest_path)
    selected = select_layers(layers,manifest["selected_layers"])
    out = fresh(output)
    result = {"status":"running","paper_ref":"Figure 11","model":manifest["model"],
        "repo_id":manifest["repo_id"],"revision":manifest["revision"],"layers":[],
        "download_manifest_sha256":sha256(manifest_path),"dtype":"original_BF16_no_cast",
        "weight_origin":manifest["origin"]}
    all_groups = groups(manifest["weight_map"])
    for layer in selected:
        e = np.zeros(256,dtype=np.int64); m = np.zeros(128,dtype=np.int64); tensor_records = []
        for expert,tensors in sorted(all_groups[layer].items()):
            for projection,name in sorted(tensors.items()):
                raw,shape = tensor_bytes(directory,manifest,name)
                eh,mh = histograms(raw);e+=eh;m+=mh
                tensor_records.append({"name":name,"shape":shape,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
        row = {"layer":layer,"experts":len(all_groups[layer]),"elements":int(e.sum()),
            "exponent_counts":e.tolist(),"mantissa_counts":m.tolist(),
            "exponent_entropy_bits":entropy(e.tolist()),"mantissa_entropy_bits":entropy(m.tolist()),"tensors":tensor_records}
        result["layers"].append(row)
        print(f"Analyzed {manifest['model']} layer {layer}: {row['experts']} experts, {row['elements']} BF16 words",flush=True)
        save(out/"figure11.json",result)
    result["status"]="passed";save(out/"figure11.json",result)

def included(model,rank,expert,first_expert,projection):
    # Paper Mixtral placement: one fused gate/up offloaded on rank 0 and one
    # down offloaded on rank 1. Exact unpublished expert IDs are assumed here.
    return not (model=="mixtral" and expert==first_expert and
                ((rank==0 and projection=="gate_up") or (rank==1 and projection=="down")))

def gpu_snapshot():
    p = subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,pstate,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory",
                        "--format=csv,noheader"],capture_output=True,text=True)
    return {"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}

def sample_statistics(samples,raw_bytes):
    import numpy as np
    if not samples or any(not math.isfinite(x) or x<=0 for x in samples): raise ValueError("invalid GPU event samples")
    return {"samples":len(samples),"avg_latency_ms":float(np.mean(samples)),
        "p99_latency_ms":float(np.percentile(samples,99)),
        "throughput_gb_s":float(np.mean([raw_bytes/x/1e6 for x in samples]))}

def equivalent(a,b):
    """Allow roundoff when replotting the same samples on another NumPy/CPU."""
    if isinstance(a,bool) or isinstance(b,bool):return type(a) is type(b) and a==b
    if isinstance(a,int) and isinstance(b,int):return a==b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    return a==b

def summarize_cases(completed,metadata):
    if not completed or any(c["status"]!="passed" for c in completed):
        raise ValueError("cannot summarize absent/failed cases")
    expected={(l,r) for l in metadata["selected_layers"] for r in range(MODELS[metadata["model"]]["degree"])}
    identities=[(c["layer"],c["logical_rank"]) for c in completed]
    if len(set(identities))!=len(identities) or set(identities)!=expected:
        raise ValueError("missing/duplicate layer-rank cases")
    summaries={}
    recomputed=[sample_statistics(c["samples_ms"],c["raw_bytes"]) for c in completed]
    for c,stats in zip(completed,recomputed):
        if not equivalent(c["statistics"],stats) or c["bitwise_equal_before_and_after"] is not True:
            raise ValueError("raw samples/statistics/correctness evidence mismatch")
    for key in ["avg_latency_ms","p99_latency_ms","throughput_gb_s"]:
        values=[s[key] for s in recomputed]
        summaries[key]={"mean":statistics.mean(values),"std":statistics.stdev(values) if len(values)>1 else 0.0}
    return {"status":"passed","cases":len(completed),"metadata":metadata,"statistics":summaries,
        "mean_uncompressed_gb":statistics.mean(c["raw_bytes"]/1e9 for c in completed),
        "mean_storage_saved_percent":statistics.mean(100*(1-c["container_bytes"]/c["raw_bytes"]) for c in completed),
        "mean_device_source_saved_percent":statistics.mean(100*(1-c["device_source_bytes"]/c["raw_bytes"]) for c in completed)}

def measure_case(manifest,directory,layer,rank,warmup,samples,chunk_size,decoder="packed"):
    import torch
    from .fast_codec import compress
    from .cuda_codec import CudaCompressedTensor
    model = manifest["model"];degree=MODELS[model]["degree"]
    experts = groups(manifest["weight_map"])[layer]
    if len(experts)%degree: raise ValueError("expert count not divisible by logical EP degree")
    n = len(experts)//degree; ids = list(range(rank*n,(rank+1)*n))
    sources=[]; outputs=[]; expected=[]; records=[]; projection_ids=[]
    started = time.perf_counter()
    try:
        for expert in ids:
            gate_up,down,shape = expert_bytes(directory,manifest,experts[expert])
            for projection,raw in [("gate_up",gate_up),("down",down)]:
                if not included(model,rank,expert,ids[0],projection): continue
                encoded=compress(raw)
                source=CudaCompressedTensor(encoded,device=0,chunk_size=chunk_size,preparation_backend="native",decoder=decoder)
                sources.append(source);outputs.append(torch.empty(source.count,dtype=torch.uint16,device="cuda"))
                expected.append(raw);projection_ids.append(0 if projection=="gate_up" else 1)
                records.append({"expert":expert,"projection":projection,"gate_shape":shape,
                    "raw_bytes":len(raw),"container_bytes":len(encoded),"device_source_bytes":source.device_bytes,
                    "raw_sha256":hashlib.sha256(raw).hexdigest(),"container_sha256":source.container_sha256})
                print(f"Prepared L{layer}/R{rank}/E{expert}/{projection}",flush=True)
        streams=[torch.cuda.Stream(),torch.cuda.Stream()];control=torch.cuda.Stream()
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        done=[torch.cuda.Event(),torch.cuda.Event()]
        def enqueue():
            for source,output,pid in zip(sources,outputs,projection_ids):
                source.decode_into(output.data_ptr(),streams[pid].cuda_stream,source.output_bytes)
        # Establish initial allocation/source-upload dependencies before using
        # non-default streams. This sync is outside every measured window.
        torch.cuda.synchronize();enqueue()
        for stream in streams:stream.synchronize()
        for source,output,reference in zip(sources,outputs,expected):
            source.check()
            if output.cpu().view(torch.uint8).numpy().tobytes()!=reference:
                raise AssertionError("real-weight GPU decode differs from original BF16 bits")
        for _ in range(warmup):enqueue()
        for stream in streams:stream.synchronize()
        prepare_seconds=time.perf_counter()-started
        before=gpu_snapshot(); times=[]
        for _ in range(samples):
            begin.record(control)
            for stream in streams:stream.wait_event(begin)
            enqueue()
            for stream,event in zip(streams,done):event.record(stream);control.wait_event(event)
            end.record(control);end.synchronize()
            times.append(begin.elapsed_time(end))
        for source,output,reference in zip(sources,outputs,expected):
            source.check()
            if output.cpu().view(torch.uint8).numpy().tobytes()!=reference:
                raise AssertionError("post-measurement GPU output differs from original BF16 bits")
        raw_bytes=sum(r["raw_bytes"] for r in records);container_bytes=sum(r["container_bytes"] for r in records)
        device_bytes=sum(r["device_source_bytes"] for r in records)
        return {"status":"passed","layer":layer,"logical_rank":rank,"logical_ep_degree":degree,"decoder":decoder,
            "raw_bytes":raw_bytes,"container_bytes":container_bytes,"device_source_bytes":device_bytes,
            "output_buffer_bytes":raw_bytes,"storage_saved_container_percent":100*(1-container_bytes/raw_bytes),
            "storage_saved_device_source_percent":100*(1-device_bytes/raw_bytes),
            "statistics":sample_statistics(times,raw_bytes),"samples_ms":times,"warmup":warmup,
            "preparation_seconds_excluded":prepare_seconds,"bitwise_equal_before_and_after":True,
            "tensors":records,"gpu_before":before,"gpu_after":gpu_snapshot(),
            "timing_scope":"CUDA event window from common start to both projection streams joined; includes host launch gaps; excludes encoding, source upload, allocation, correctness copies and checks",
            "torch_allocated_bytes":torch.cuda.memory_allocated(),"torch_reserved_bytes":torch.cuda.memory_reserved()}
    finally:
        for source in sources:source.close()

def benchmark(manifest_path,output,warmup,samples,chunk_size,decoder="packed"):
    import torch
    if os.environ.get("CUDA_LAUNCH_BLOCKING") not in (None,"","0"):
        raise ValueError("unset CUDA_LAUNCH_BLOCKING for the two-stream measurement")
    if warmup<1 or samples<2:raise ValueError("warmup>=1 and samples>=2 required")
    if type(chunk_size) is not int or not 1<=chunk_size<=65536:raise ValueError("invalid decoder chunk size")
    if not torch.cuda.is_available():raise RuntimeError("CUDA device required; no CPU timing fallback")
    manifest,directory=load_manifest(manifest_path);out=fresh(output)
    spec=MODELS[manifest["model"]]; selected=manifest["selected_layers"]
    metadata={"model":manifest["model"],"repo_id":manifest["repo_id"],"revision":manifest["revision"],
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"hostname":platform.node(),
        "paper_ref":"Table 3 / Exp#3","implementation":f"independent FMBF {decoder} decoder, not author code","decoder":decoder,
        "native_source_sha256":sha256(Path(__file__).parent.parent/"native/huffman_decode.cu"),
        "weight_origin":manifest["origin"],"measurement_source_sha256":sha256(__file__),
        "implementation_source_sha256":{name:sha256(Path(__file__).parent/name)
            for name in ["weight_experiments.py","fast_codec.py","cuda_codec.py","codec.py"]},
        "gpu":torch.cuda.get_device_name(0),"capability":list(torch.cuda.get_device_capability(0)),
        "python":platform.python_version(),"torch":torch.__version__,"torch_cuda":torch.version.cuda,
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"chunk_size":chunk_size,
        "selected_layers":selected,"selected_cases":len(selected)*spec["degree"],
        "available_layers":manifest["available_layers"],
        "paper_expected_cases":spec["paper_layers"]*spec["degree"],
        "coverage":"full_layer_rank_grid" if selected==manifest["available_layers"] and len(selected)==spec["paper_layers"] else "partial_layers_pilot",
        "download_manifest_sha256":sha256(manifest_path),
        "placement_note":"Mixtral: rank0 first local expert gate/up on CPU, rank1 first local expert down on CPU; unpublished original IDs assumed. Qwen/GLM all routed tensors compressed.",
        "statistics_note":"Per-case empirical P99 over samples_ms; table mean/std across selected layer/rank case summaries. This is not a confidence interval. Logical ranks run sequentially on one GPU.",
        "versions":{n:importlib.metadata.version(n) for n in ["numpy","safetensors"]}}
    save(out/"metadata.json",metadata)
    completed=[]
    for layer in selected:
        for rank in range(spec["degree"]):
            try:
                case=measure_case(manifest,directory,layer,rank,warmup,samples,chunk_size,decoder)
            except Exception as exc:
                case={"status":"failed","layer":layer,"logical_rank":rank,"error":str(exc),"traceback":traceback.format_exc()}
                with (out/"cases.jsonl").open("a") as f:f.write(json.dumps(case)+"\n")
                save(out/"summary.json",{"status":"failed","failed_case":case,"completed_cases":len(completed)})
                raise
            with (out/"cases.jsonl").open("a") as f:f.write(json.dumps(case,allow_nan=False)+"\n")
            completed.append(case)
            print(f"Measured L{layer}/R{rank}: {case['statistics']}",flush=True)
            torch.cuda.empty_cache()
    result=summarize_cases(completed,metadata)
    result["raw_cases_sha256"]=sha256(out/"cases.jsonl")
    result["metadata_sha256"]=sha256(out/"metadata.json")
    save(out/"summary.json",result)
    # Capture the actual target-built native records alongside the raw samples.
    root=Path(__file__).resolve().parent.parent
    records=out/"build_records";records.mkdir()
    for name in ["libfluxmoe_huffman.json","build.log","libfluxmoe_cpu_codec.json","cpu_codec_build.log"]:
        p=root/"native/build"/name
        if p.exists():shutil.copy2(p,records/name)
    print(json.dumps(result,indent=2),flush=True)

def verified_summary(path):
    path=Path(path);stored=json.loads(path.read_text())
    if stored["status"]!="passed":raise ValueError(f"benchmark not passed: {path}")
    raw=path.parent/"cases.jsonl";meta=path.parent/"metadata.json"
    if sha256(raw)!=stored["raw_cases_sha256"] or sha256(meta)!=stored["metadata_sha256"]:
        raise ValueError("benchmark raw data hashes differ from saved summary")
    cases=[json.loads(s) for s in raw.read_text().splitlines()]
    d=summarize_cases(cases,json.loads(meta.read_text()))
    if any(not equivalent(stored[k],d[k]) for k in d):raise ValueError("benchmark summary differs from raw samples")
    return d,cases

def merge(paths,output):
    if not paths:raise ValueError("at least one benchmark required")
    loaded=[verified_summary(p) for p in paths]
    base=loaded[0][0]["metadata"]
    keys=["model","repo_id","revision","implementation","decoder","native_source_sha256","weight_origin","measurement_source_sha256","implementation_source_sha256",
          "gpu","capability","hostname","python","torch","torch_cuda","cuda_visible_devices",
          "chunk_size","available_layers","versions","placement_note"]
    for d,_ in loaded:
        if any(d["metadata"].get(k)!=base.get(k) for k in keys):
            raise ValueError("merge requires identical model revision, implementation, device and software")
    cases=[c for _,rows in loaded for c in rows]
    if len({(c["warmup"],len(c["samples_ms"])) for c in cases})!=1:
        raise ValueError("merge requires identical warmup/sample protocol")
    metadata=dict(base,selected_layers=sorted({c["layer"] for c in cases}),selected_cases=len(cases),
        merged_sources=[{"path":str(p),"sha256":sha256(p)} for p in paths])
    spec=MODELS[base["model"]]
    metadata["coverage"]=("full_layer_rank_grid" if metadata["selected_layers"]==base["available_layers"]
                          and len(metadata["selected_layers"])==spec["paper_layers"] else "partial_layers_pilot")
    result=summarize_cases(cases,metadata)
    out=fresh(output);save(out/"metadata.json",metadata)
    (out/"cases.jsonl").write_text("".join(json.dumps(c,allow_nan=False)+"\n" for c in cases))
    result.update(raw_cases_sha256=sha256(out/"cases.jsonl"),metadata_sha256=sha256(out/"metadata.json"))
    save(out/"summary.json",result)
    return result

def stream_layers(plan_path,data_root,output,per_layer_gb,total_gb,warmup,samples,chunk_size):
    """Download one layer's shards at a time; remove only tool-created staging.

    Keeps every layer's manifest and raw benchmark samples in the result tree.
    It does not delete an existing user checkpoint or infer access to /data.
    """
    plan=json.loads(Path(plan_path).read_text());g=groups(plan["weight_map"])
    subsets=[]
    for layer in plan["selected_layers"]:
        names={n for tensors in g[layer].values() for n in tensors.values()}
        mapping={n:plan["weight_map"][n] for n in names}
        shards=[s for s in plan["shards"] if s["name"] in set(mapping.values())]
        subsets.append(dict(plan,selected_layers=[layer],weight_map=mapping,shards=shards,
                            required_download_bytes=sum(s["bytes"] for s in shards)))
    total=sum(p["required_download_bytes"] for p in subsets)
    if (not math.isfinite(total_gb) or total_gb<=0 or total>total_gb*1e9
            or not math.isfinite(per_layer_gb) or per_layer_gb<=0
            or any(p["required_download_bytes"]>per_layer_gb*1e9 for p in subsets)):
        raise ValueError(f"stream download cap exceeded: total {total/1e9:.3f}GB; inspect per-layer plans")
    root=Path(data_root).expanduser().resolve()
    if not root.is_dir() or not os.access(root,os.W_OK):raise PermissionError("data root must exist and be writable")
    # local_dir downloads write weights directly into each staging directory,
    # rather than populate the Hub's central model cache. The manual command
    # also sets HF_HUB_DISABLE_XET before import for predictable disk usage
    # across installed hf_xet versions (the old chunk-cache knob is obsolete).
    out=fresh(output);paths=[]
    save(out/"stream_plan.json",{"total_download_bytes_including_repeated_shards":total,
        "peak_layer_download_bytes":max(p["required_download_bytes"] for p in subsets),
        "cleanup":"only each tool-created temporary staging directory; manifests/results retained"})
    print(f"Streaming {len(subsets)} layers; planned network volume {total/1e9:.3f}GB",flush=True)
    for subset in subsets:
        layer=subset["selected_layers"][0];layer_out=out/f"layer-{layer}";layer_out.mkdir()
        pp=layer_out/"download_plan.json";save(pp,subset)
        with tempfile.TemporaryDirectory(prefix=f"fluxmoe-{plan['model']}-l{layer}-",dir=root) as staging:
            checkpoint=Path(staging)/"checkpoint"
            try:
                fetch(pp,checkpoint,per_layer_gb)
                benchmark(checkpoint/"download_manifest.json",layer_out/"benchmark",warmup,samples,chunk_size)
            finally:
                manifest=checkpoint/"download_manifest.json"
                if manifest.exists():shutil.copy2(manifest,layer_out/manifest.name)
        paths.append(layer_out/"benchmark/summary.json")
    merge(paths,out/"combined")
    print(f"Complete raw grid: {out/'combined/summary.json'}",flush=True)

def report(analyses,benchmarks,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=fresh(output)
    figure_rows=[]
    for file in analyses:
        d=json.loads(Path(file).read_text())
        if d["status"]!="passed":raise ValueError(f"analysis not passed: {file}")
        for layer in d["layers"]:figure_rows.append((d,layer))
    if figure_rows:
        fig,axes=plt.subplots(len(figure_rows),2,figsize=(11,3*len(figure_rows)),squeeze=False)
        with (out/"figure11.csv").open("w",newline="") as f:
            writer=csv.writer(f);writer.writerow(["model","revision","layer","field","symbol","count","probability"])
            for row,(d,layer) in enumerate(figure_rows):
                for col,field in enumerate(["exponent","mantissa"]):
                    counts=layer[field+"_counts"];total=sum(counts)
                    probabilities=[n/total for n in counts]
                    axes[row,col].plot(range(len(counts)),probabilities,lw=1.3)
                    axes[row,col].set(title=f"{d['model']} layer {layer['layer']} — {field}",xlabel=f"BF16 {field} symbol",ylabel="Probability")
                    axes[row,col].set_xlim(0,len(counts)-1)
                    for symbol,(count,p) in enumerate(zip(counts,probabilities)):
                        writer.writerow([d["model"],d["revision"],layer["layer"],field,symbol,count,p])
        origins=sorted({d.get("weight_origin","unknown") for d,_ in figure_rows})
        fig.suptitle("Figure 11 correspondence — BF16 expert weights / "+", ".join(origins))
        fig.tight_layout();fig.savefig(out/"figure11.png",dpi=180);fig.savefig(out/"figure11.pdf");plt.close(fig)
    rows=[]
    for path in benchmarks:
        d,_=verified_summary(path)
        m=d["metadata"];s=d["statistics"]
        rows.append({"model":m["model"],"gpu":m["gpu"],"decoder":m.get("decoder","legacy"),"chunk_size":m["chunk_size"],"coverage":m["coverage"],"cases":d["cases"],
            "raw_GB":d["mean_uncompressed_gb"],"saved_container_pct":d["mean_storage_saved_percent"],
            "saved_device_source_pct":d["mean_device_source_saved_percent"],
            **{key+"_"+stat:s[key][stat] for key in s for stat in ["mean","std"]}})
    if rows:
        with (out/"table3.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        fig,axes=plt.subplots(1,4,figsize=(16,4))
        def label(r):
            gpu=r['gpu'].replace('NVIDIA GeForce ','').replace('NVIDIA ','').replace(' Laptop GPU',' Laptop').replace(' GPU','')
            coverage='full' if r['coverage']=='full_layer_rank_grid' else 'pilot'
            return f"{r['model']} / {r['decoder']}\n{gpu}\n{r['cases']} cases / {coverage}"
        labels=[label(r) for r in rows]
        for ax,metric,title in zip(axes,["saved_container_pct","avg_latency_ms_mean","p99_latency_ms_mean","throughput_gb_s_mean"],
                                   ["Storage saved (%)","Mean latency (ms)","P99 latency (ms)","Decoded output (GB/s)"]):
            error=[r.get(metric.replace("_mean","_std"),0) for r in rows] if metric.endswith("_mean") else None
            ax.bar(labels,[r[metric] for r in rows],yerr=error,capsize=4);ax.set_title(title)
            ax.tick_params(axis='x',labelsize=8)
        origins=sorted({json.loads(Path(p).read_text())["metadata"].get("weight_origin","unknown") for p in benchmarks})
        fig.suptitle("Table 3 — independent decoder / "+", ".join(origins))
        fig.text(.5,.015,"Error bars: std across layer/rank cases; pilot samples do not establish stable P99",ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(out/"table3.png",dpi=180);fig.savefig(out/"table3.pdf");plt.close(fig)
    save(out/"sources.json",{"analyses":[{"path":str(p),"sha256":sha256(p)} for p in analyses],
        "benchmarks":[{"path":str(p),"sha256":sha256(p)} for p in benchmarks]})
    print(f"Figures and CSV: {out}")

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    s=sub.add_parser("plan");s.add_argument("--model",choices=MODELS,required=True);s.add_argument("--layers",required=True);s.add_argument("--revision",default="main");s.add_argument("--output",required=True)
    s=sub.add_parser("fetch");s.add_argument("--plan",required=True);s.add_argument("--directory",required=True);s.add_argument("--max-download-gb",type=float,required=True)
    s=sub.add_parser("analyze");s.add_argument("--manifest",required=True);s.add_argument("--layers",required=True);s.add_argument("--output",required=True)
    s=sub.add_parser("benchmark");s.add_argument("--manifest",required=True);s.add_argument("--output",required=True);s.add_argument("--warmup",type=int,default=10);s.add_argument("--samples",type=int,default=200);s.add_argument("--chunk-size",type=int,default=256);s.add_argument("--decoder",choices=["tree","lookup","tiled","bulk","buffered","vectorized","packed"],default="packed")
    s=sub.add_parser("report");s.add_argument("--analysis",nargs="*",default=[]);s.add_argument("--benchmark",nargs="*",default=[]);s.add_argument("--output",required=True)
    s=sub.add_parser("merge");s.add_argument("--inputs",nargs="+",required=True);s.add_argument("--output",required=True)
    s=sub.add_parser("stream");s.add_argument("--plan",required=True);s.add_argument("--data-root",required=True);s.add_argument("--output",required=True);s.add_argument("--max-layer-download-gb",type=float,required=True);s.add_argument("--max-total-download-gb",type=float,required=True);s.add_argument("--warmup",type=int,default=20);s.add_argument("--samples",type=int,default=500);s.add_argument("--chunk-size",type=int,default=256)
    a=p.parse_args()
    if a.command=="plan":make_plan(a.model,a.layers,a.output,a.revision)
    elif a.command=="fetch":fetch(a.plan,a.directory,a.max_download_gb)
    elif a.command=="analyze":analyze(a.manifest,a.layers,a.output)
    elif a.command=="benchmark":benchmark(a.manifest,a.output,a.warmup,a.samples,a.chunk_size,a.decoder)
    elif a.command=="merge":merge(a.inputs,a.output)
    elif a.command=="stream":stream_layers(a.plan,a.data_root,a.output,a.max_layer_download_gb,a.max_total_download_gb,a.warmup,a.samples,a.chunk_size)
    else:report(a.analysis,a.benchmark,a.output)

if __name__=="__main__":main()
