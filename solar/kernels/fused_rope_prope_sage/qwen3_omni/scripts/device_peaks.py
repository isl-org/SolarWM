"""Measured device peaks used as utilisation denominators: bf16 GEMM TFLOPS, int8 GEMM TOPS (if
available), and device copy bandwidth (read + write bytes)."""
import json, time, torch
torch.xpu.set_per_process_memory_fraction(0.6)
def t(fn, it=20):
    for _ in range(3): fn()
    torch.xpu.synchronize(); s = time.perf_counter()
    for _ in range(it): fn()
    torch.xpu.synchronize(); return (time.perf_counter() - s) / it
res = {"device": torch.xpu.get_device_name(0)}
best = 0
for n in (4096, 8192):
    a = torch.randn(n, n, device="xpu", dtype=torch.bfloat16); b = torch.randn(n, n, device="xpu", dtype=torch.bfloat16)
    best = max(best, 2 * n**3 / t(lambda: a @ b) / 1e12)
res["bf16_gemm_tflops"] = best
try:
    a = torch.randint(-8, 8, (8192, 8192), device="xpu", dtype=torch.int8); b = torch.randint(-8, 8, (8192, 8192), device="xpu", dtype=torch.int8)
    res["int8_gemm_tops"] = 2 * 8192**3 / t(lambda: torch._int_mm(a, b)) / 1e12
except Exception as e:
    res["int8_gemm_tops"] = None; res["int8_note"] = f"{type(e).__name__}: {str(e)[:80]}"
# Random data: never-written buffers are zero and get memory compression (measured 1144 GB/s on
# an uninitialised uint8 copy vs ~530 GB/s with real data).
n = 2**30
x = torch.randn(n // 4, device="xpu", dtype=torch.float32); y = torch.empty_like(x)
res["read_bw_gbs"] = n / t(lambda: x.sum(), 10) / 1e9
res["copy_bw_gbs"] = 2 * n / t(lambda: y.copy_(x), 10) / 1e9
res["add_bw_gbs"] = 3 * n / t(lambda: torch.add(x, x, out=y), 10) / 1e9
res["peak_bw_gbs"] = max(res["read_bw_gbs"], res["copy_bw_gbs"], res["add_bw_gbs"])
print(json.dumps(res, indent=1)); json.dump(res, open("/home/gta/omni_bench/logs/device_peaks.json", "w"), indent=1)
