"""Per-stage XPU kernel time of sage_causal.forward (quantize_q / k_centre / quantize_k / attention) vs SDPA."""
import sys, torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import os; sys.path.insert(0, os.environ.get("SAGE_BUILD_DIR", "/home/gta/omni_bench/kernel/build")); import sage_causal
H, HKV, D = 32, 4, 128
torch.xpu.set_per_process_memory_fraction(0.92)
for S in map(int, sys.argv[1:]):
    q = torch.randn(1, S, H, D, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(1, S, HKV, D, device="xpu", dtype=torch.bfloat16); v = torch.randn_like(k)
    for _ in range(3): sage_causal.forward(q, k, v, is_causal=True)
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        for _ in range(3): sage_causal.forward(q, k, v, is_causal=True)
        torch.xpu.synchronize()
    print(f"== S={S}")
    evs = [e for e in prof.key_averages() if e.device_type.name == "XPU" or getattr(e, "self_device_time_total", 0) > 0]
    tot = sum(e.self_device_time_total for e in evs)
    for e in sorted(evs, key=lambda e: -e.self_device_time_total)[:8]:
        print(f"  {e.self_device_time_total/3/1e3:8.2f} ms  {100*e.self_device_time_total/tot:5.1f}%  {e.key[:110]}")
    del q, k, v; torch.xpu.empty_cache()
