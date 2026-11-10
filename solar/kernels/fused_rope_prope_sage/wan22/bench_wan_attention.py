"""Standalone attention benchmark on Wan2.2 self-attention shapes (non-causal MHA, D=128, bf16):
torch SDPA (default Wan backend on XPU) vs Sage INT8 (sage_causal) vs bf16 flash (flash_causal).
    ./env.sh python wan/bench_wan_attention.py"""
import sys, time, statistics as st, torch, torch.nn.functional as F
import os
sys.path.insert(0, os.environ.get("SAGE_BUILD_DIR", "/home/gta/omni_bench/kernel/build"))
sys.path.insert(0, os.environ.get("FLASH_BUILD_DIR", "/home/gta/omni_bench/kernel/build_flash"))
import sage_causal, flash_causal
torch.xpu.set_per_process_memory_fraction(0.9)
SHAPES = [("TI2V-5B 81f/153f (window 21f)", 8505, 24), ("I2V-A14B 81f", 32760, 40), ("I2V-A14B 153f", 60840, 40)]
def bench(fn, n):
    fn(); torch.xpu.synchronize(); ts = []
    for _ in range(n):
        a = torch.xpu.Event(enable_timing=True); b = torch.xpu.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
    return st.median(ts)
for name, S, H in SHAPES:
    g = torch.Generator(device="xpu").manual_seed(0)
    q = torch.randn(1, S, H, 128, device="xpu", generator=g, dtype=torch.float32).bfloat16()
    k = torch.randn(1, S, H, 128, device="xpu", generator=g, dtype=torch.float32).bfloat16()
    v = torch.randn(1, S, H, 128, device="xpu", generator=g, dtype=torch.float32).bfloat16()
    n = 5 if S > 40000 else 10
    flops = 4 * H * 128 * S * S
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # Wan calls SDPA on [B,H,S,D] views
    t_sdpa = bench(lambda: F.scaled_dot_product_attention(qt, kt, vt), n)
    t_sage = bench(lambda: sage_causal.forward(q, k, v, False), n)
    pad = (-S) % 64
    t_flash = bench(lambda: flash_causal.forward(q, k, v, False), n)
    # exact fp32 reference on 512 sampled query rows, 4 heads (no full score matrix)
    rows = torch.linspace(0, S - 1, 512, device="xpu").long()
    o_s = sage_causal.forward(q, k, v, False)[0, rows, :4].float()
    ref = []
    for h in range(4):
        sc = (q[0, rows, h].float() @ k[0, :, h].float().T) * 128 ** -0.5
        ref.append(torch.softmax(sc, -1) @ v[0, :, h].float())
    cos = F.cosine_similarity(o_s.transpose(0, 1).flatten(), torch.stack(ref).flatten(), dim=0).item()
    print(f"{name:32s} S={S:6d} H={H}: SDPA {t_sdpa:8.1f} ms ({flops/t_sdpa/1e9:5.0f} TF) | Sage INT8 {t_sage:8.1f} ms ({flops/t_sage/1e9:5.0f} TF, {t_sdpa/t_sage:.2f}x) | "
          f"flash bf16 {t_flash:8.1f} ms ({flops/t_flash/1e9:5.0f} TF, {t_sdpa/t_flash:.2f}x) | Sage cos vs fp32 (4 heads) {cos:.5f}", flush=True)
    del q, k, v, qt, kt, vt; torch.xpu.empty_cache()
