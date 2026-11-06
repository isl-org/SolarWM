"""Peak extra device memory of one sage_causal call at S tokens (bf16, 32/4 heads, D=128, causal).
    ./env.sh python kernel/peak_memory.py BUILD_DIR [S]"""
import sys, torch
sys.path.insert(0, sys.argv[1]); import sage_causal as m
S = int(sys.argv[2]) if len(sys.argv) > 2 else 63488
q = torch.randn(1, S, 32, 128, device="xpu", dtype=torch.bfloat16); k = torch.randn(1, S, 4, 128, device="xpu", dtype=torch.bfloat16); v = torch.randn_like(k)
m.forward(q, k, v, True); torch.xpu.synchronize(); torch.xpu.empty_cache()
base = torch.xpu.memory_allocated(); torch.xpu.reset_peak_memory_stats()
o = m.forward(q, k, v, True); torch.xpu.synchronize()
print(f"{sys.argv[1]}: S={S} peak extra {(torch.xpu.max_memory_allocated() - base) / 2**20:.0f} MiB (incl. {o.numel() * 2 / 2**20:.0f} MiB output)")
