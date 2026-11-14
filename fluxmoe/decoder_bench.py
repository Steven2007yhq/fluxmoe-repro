"""One real expert projection: decoder ablations and device-copy control.

This is an isolated diagnostic, not Table 3's full layer/two-stream workload.
Preparation is outside CUDA-event timing. Every run checks original BF16 bits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from datetime import datetime, timezone
import shutil
import traceback

from .weight_experiments import (load_manifest, groups, expert_bytes, fresh, save,
    sha256, gpu_snapshot, sample_statistics)


def run(manifest_path, output, cache, layer=0, expert=0, projection="gate_up",
        modes=("tree","lookup","tiled","bulk","buffered","vectorized","packed"), warmup=5, samples=20, runs=3, chunk_size=256):
    import torch
    from .fast_codec import compress
    from .cuda_codec import CudaCompressedTensor
    if warmup<1 or samples<2 or runs<1:raise ValueError("invalid sample protocol")
    manifest,directory=load_manifest(manifest_path)
    raw_pair=expert_bytes(directory,manifest,groups(manifest["weight_map"])[layer][expert])
    raw=raw_pair[0 if projection=="gate_up" else 1];del raw_pair
    digest=hashlib.sha256(raw).hexdigest()
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    blob=cache/(digest+".fmbf");record=blob.with_suffix(".json")
    if blob.exists():
        expected=json.loads(record.read_text())
        if expected["raw_sha256"]!=digest or expected["container_sha256"]!=sha256(blob):
            raise ValueError("diagnostic encoded cache hash mismatch")
        encoded=blob.read_bytes()
    else:
        encoded=compress(raw);blob.write_bytes(encoded)
        save(record,{"raw_sha256":digest,"container_sha256":sha256(blob)})
    out=fresh(output)
    metadata={"scope":"one real expert projection; single stream diagnostic; not Table 3",
        "pid":os.getpid(),"hostname":platform.node(),"python":platform.python_version(),
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "model":manifest["model"],"revision":manifest["revision"],"layer":layer,"expert":expert,
        "projection":projection,"raw_bytes":len(raw),"raw_sha256":digest,"container_bytes":len(encoded),
        "container_sha256":sha256(blob),"weight_origin":manifest["origin"],"chunk_size":chunk_size,
        "gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"torch_cuda":torch.version.cuda,
        "warmup":warmup,"samples":samples,"runs":runs,"modes":list(modes),
        "source_sha256":sha256(__file__),"native_source_sha256":sha256(Path(__file__).parent.parent/"native/huffman_decode.cu")}
    save(out/"result.json",{"status":"running","metadata":metadata})
    records=[]
    stream=torch.cuda.Stream()
    output_tensor=torch.empty(len(raw)//2,dtype=torch.uint16,device="cuda")
    def check():
        if output_tensor.cpu().view(torch.uint8).numpy().tobytes()!=raw:
            raise AssertionError("diagnostic output differs from original BF16 bits")
    def measure(enqueue,label):
        torch.cuda.synchronize()
        enqueue();stream.synchronize();check()
        for _ in range(warmup):enqueue()
        stream.synchronize()
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        times=[];before=gpu_snapshot()
        torch.cuda.nvtx.range_push("decoder_bench/"+label)
        try:
            for _ in range(samples):
                begin.record(stream);enqueue();end.record(stream);end.synchronize()
                times.append(begin.elapsed_time(end))
        finally:torch.cuda.nvtx.range_pop()
        check()
        return {"status":"passed","mode":label,"samples_ms":times,"statistics":sample_statistics(times,len(raw)),
                "bitwise_equal_before_and_after":True,"gpu_before":before,"gpu_after":gpu_snapshot()}
    try:
        for repeat in range(runs):
            # Reverse alternate runs to expose order/clock effects, not hide them.
            order=list(modes) if repeat%2==0 else list(reversed(modes))
            for mode in order:
                with CudaCompressedTensor(encoded,chunk_size=chunk_size,preparation_backend="native",decoder=mode) as source:
                    row=measure(lambda:source.decode_into(output_tensor.data_ptr(),stream.cuda_stream),mode)
                    source.check();row.update(run=repeat,device_source_bytes=source.device_bytes)
                records.append(row)
                with (out/"samples.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
                print(mode,repeat,row["statistics"],flush=True)
        resident=output_tensor.clone();torch.cuda.synchronize()
        def copy_control():
            with torch.cuda.stream(stream):output_tensor.copy_(resident)
        row=measure(copy_control,"device_copy_control");row["run"]=0;records.append(row)
        with (out/"samples.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        print(row["mode"],row["statistics"],flush=True)
        save(out/"result.json",{"status":"passed","metadata":metadata,"records":records,
            "samples_sha256":sha256(out/"samples.jsonl")})
    except Exception:
        save(out/"result.json",{"status":"failed","metadata":metadata,"completed":records,"error":traceback.format_exc()})
        raise
    builds=out/"build_records";builds.mkdir()
    for name in ["libfluxmoe_huffman.json","build.log","libfluxmoe_cpu_codec.json","cpu_codec_build.log"]:
        p=Path(__file__).parent.parent/"native/build"/name
        if p.exists():shutil.copy2(p,builds/name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);p.add_argument("--cache",required=True)
    p.add_argument("--layer",type=int,default=0);p.add_argument("--expert",type=int,default=0)
    p.add_argument("--projection",choices=["gate_up","down"],default="gate_up")
    p.add_argument("--modes",nargs="+",choices=["tree","lookup","tiled","bulk","buffered","vectorized","packed"],default=["tree","lookup","tiled","bulk","buffered","vectorized","packed"])
    p.add_argument("--warmup",type=int,default=5);p.add_argument("--samples",type=int,default=20);p.add_argument("--runs",type=int,default=3)
    p.add_argument("--chunk-size",type=int,default=256)
    a=p.parse_args();run(a.manifest,a.output,a.cache,a.layer,a.expert,a.projection,a.modes,a.warmup,a.samples,a.runs,a.chunk_size)


if __name__=="__main__":main()
