"""Benchmark per-layer + full dense block timings of the compiled extension (median of N, ms)."""
import sys, os, time, importlib, json, statistics
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import task
from qdq_utils import quantize_activation_int4


def timeit(fn, warm=5, iters=30, reps=5):
    for _ in range(warm): fn()
    torch.xpu.synchronize()
    meds = []
    for _ in range(reps):
        ts = []
        for _ in range(iters):
            s = torch.xpu.Event(enable_timing=True); e = torch.xpu.Event(enable_timing=True)
            s.record(); fn(); e.record(); e.synchronize()
            ts.append(s.elapsed_time(e))
        meds.append(statistics.median(ts))
    return min(meds)


def layer_fns(mod, bd):
    dev = torch.device("xpu")
    x = bd["x"]; q, v, up, dn = bd["q_p"], bd["v_p"], bd["ffn_up_p"], bd["ffn_down_p"]
    h = torch.zeros((1215, 14336), dtype=torch.bfloat16, device=dev)
    h_q, h_s = quantize_activation_int4(h, 64)
    g = mod.svdquant_w4a4_gemm
    return {
        "q": lambda: g(x, q["w_dpas_b"], q["w_scale_dpas"], bias=q["bias"], lora_a=q["lora_a"], lora_b=q["lora_b"], norm_weight=q["norm_w"], activation="none"),
        "v": lambda: g(x, v["w_dpas_b"], v["w_scale_dpas"], bias=v["bias"], lora_a=v["lora_a"], lora_b=v["lora_b"], activation="none"),
        "ffn_up": lambda: g(x, up["w_dpas_b"], up["w_scale_dpas"], bias=up["bias"], lora_a=up["lora_a"], lora_b=up["lora_b"], activation="gelu_tanh", fuse_quant_out=True),
        "ffn_down": lambda: g(h_q, dn["w_dpas_b"], dn["w_scale_dpas"], bias=dn["bias"], lora_a=dn["lora_a"], lora_b=dn["lora_b"], x_scales=h_s, x_bf16=h, activation="none"),
        "block": (lambda f=task.make_sycl_dense_block_fn(mod): f(*bd["sycl_args"])),
    }


if __name__ == "__main__":
    mod = importlib.import_module(task.extension_name(task.load_hparams()))
    bd = task.block_data.__wrapped__(torch.device("xpu"))
    res = {k: timeit(f) for k, f in layer_fns(mod, bd).items()}
    print(json.dumps({k: round(v, 3) for k, v in res.items()}))
    if "--ref" in sys.argv:
        ref = timeit(lambda: task.reference_dense_block(*bd["ref_args"]))
        print("bf16 reference block ms:", round(ref, 3))
