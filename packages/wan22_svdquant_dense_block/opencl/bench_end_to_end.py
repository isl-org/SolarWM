"""End-to-end timing of one implementation on the two task workloads, same operation everywhere:
   bf16 x -> dynamic INT4 group-64 quantization -> LoRA-A -> INT4 GEMM -> LoRA-B -> bias -> bf16 out (no activation).

  v    : x (M = 1215) with the v projection weights  (N = 3072,  K = 3072)
  wide : x (M = 1215) with the ffn_up weights        (N = 14336, K = 3072), plain epilogue

Usage (one implementation per process; the SYCL builds share a module name):
  python opencl/bench_end_to_end.py sycl-current              # autotuned pool (the shipped behaviour)
  python opencl/bench_end_to_end.py sycl-static               # KF_AUTOTUNE=0: static default only
  python opencl/bench_end_to_end.py sycl-original  /tmp/pkg_orig   # the kernel as received (build dir given)
  python opencl/bench_end_to_end.py ocl-fixed                 # first OpenCL port: 4x4 sub-groups, 256 GRF, no pipeline
  python opencl/bench_end_to_end.py ocl-tuned                 # static dispatch + runtime autotuning (shipped)
Prints one JSON line: {"impl", "ms": {"v": ..., "wide": ...}, "variant": {...}}. Times are wall clock per call over
20 back-to-back calls with one synchronization at the end (the same method for SYCL and OpenCL); minimum of 7 batches.
Inputs come from prepare_v_inputs.py (/tmp/ocl_probe/v_inputs2.npz).
"""
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
NPZ = os.environ.get("KF_INPUTS", "/tmp/ocl_probe/v_inputs2.npz")
CALLS, BATCHES = 20, 7
WORKLOADS = {"v": "v", "wide": "wide"}


def bench(fn, sync):
    fn(); sync()                       # warm-up (also triggers any one-time tuning)
    best = float("inf")
    for _ in range(BATCHES):
        sync()
        t0 = time.perf_counter()
        for _ in range(CALLS):
            fn()
        sync()
        best = min(best, (time.perf_counter() - t0) / CALLS)
    return best * 1e3


def run_sycl(impl, build_dir):
    if impl == "sycl-static":
        os.environ["KF_AUTOTUNE"] = "0"
    sys.path.insert(0, str(build_dir))
    import torch
    mod = importlib.import_module("svdquant_dense_kernel_grf128_n_tile64")
    assert Path(mod.__file__).resolve().parent == Path(build_dir).resolve(), mod.__file__
    d = np.load(NPZ)
    dev = torch.device("xpu")
    bf16 = lambda a: torch.from_numpy(a.view(np.int16)).to(dev).view(torch.bfloat16)
    x = bf16(d["x"])
    out, variant = {}, {}
    for name in WORKLOADS:
        wb = torch.from_numpy(d[f"{name}_w_dpas_b"].view(np.int32)).to(dev)
        ws = torch.from_numpy(d[f"{name}_w_scale"]).to(dev)
        la, lb, bias = bf16(d[f"{name}_lora_a"]), bf16(d[f"{name}_lora_b"]), bf16(d[f"{name}_bias"])
        call = lambda: mod.svdquant_w4a4_gemm(x, wb, ws, bias=bias, lora_a=la, lora_b=lb)
        out[name] = bench(call, torch.xpu.synchronize)
        if hasattr(mod, "last_selection"):
            variant[name] = mod.last_selection().get("winner")
    return out, variant


def run_ocl(impl):
    import pyopencl as cl
    sys.path.insert(0, str(HERE))
    import ocl_svdquant as L
    d = np.load(NPZ)
    dev = L.intel_gpu()
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    out, variant = {}, {}
    for name in WORKLOADS:
        weights = L.Weights(ctx, d[f"{name}_w_dpas_b"], d[f"{name}_w_scale"], d[f"{name}_lora_a"],
                            d[f"{name}_lora_b"], d[f"{name}_bias"])
        layer = L.Layer(ctx, queue, weights, d["x"])
        if impl == "ocl-fixed":
            values = {"KF_SGM": 4, "KF_SGN": 4, "KF_PIPE": 0, "KF_GRF": 256}
            run = L.make_runner(L.StaticDispatch(ctx, dev).program(values), queue, values)
            call = lambda: run(layer)
            variant[name] = values
        else:
            op = L.SvdquantLinear(ctx, queue, cache_dir=Path(tempfile.mkdtemp(prefix="bench-")))
            call = lambda: op(layer)
            call(); queue.finish()
            variant[name] = op.selected
        out[name] = bench(call, queue.finish)
    return out, variant


def main():
    impl = sys.argv[1]
    if impl.startswith("sycl"):
        build_dir = sys.argv[2] if len(sys.argv) > 2 else PKG
        ms, variant = run_sycl(impl, build_dir)
    else:
        ms, variant = run_ocl(impl)
    print(json.dumps({"impl": impl, "ms": ms, "variant": variant}))


if __name__ == "__main__":
    main()
