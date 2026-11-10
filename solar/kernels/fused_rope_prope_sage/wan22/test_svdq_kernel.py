"""SVDQuant kernel on real Wan2.2-A14B weights (block 20, high-noise expert) at M = 32,130 tokens:
accuracy vs fp32 (kernel, its emulation, and rank 0 = plain W4A4) and speed vs W8A16 / bf16 linears."""
import json, os, sys, statistics as st, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wan_svdq as S
from safetensors import safe_open
D = os.environ.get("WAN_INT8_DIR", "/home/gta/models/wan22-i2v-a14b-int8") + "/transformer"
idx = json.load(open(f"{D}/int8_index.json"))["weight_map"]
def get(k):
    with safe_open(f"{D}/{idx[k]}", "pt", device="xpu") as f: return f.get_tensor(k)
dev, M = torch.device("xpu"), 32130
def lin(name):
    return get(f"{name}.weight").float() * get(f"{name}.weight_scale").float()[:, None], get(f"{name}.bias").float()
def bench(fn, n=10):
    fn(); torch.xpu.synchronize(); ts = []
    for _ in range(n):
        a, b = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
    return st.median(ts)
torch.manual_seed(0)
cos = lambda a, b: F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()
b = "blocks.20"
cases = [("qkv", torch.cat([lin(f"{b}.attn1.to_{p}")[0] for p in "qkv"]), torch.cat([lin(f"{b}.attn1.to_{p}")[1] for p in "qkv"])),
         ("ffn_up", *lin(f"{b}.ffn.net.0.proj")), ("ffn_down", *lin(f"{b}.ffn.net.2"))]
for name, w, bias in cases:
    n, k = w.shape
    x = torch.randn(M, k, device=dev)
    x[:, torch.randperm(k, device=dev)[:k // 100]] *= 20          # 1% outlier channels, as in DiT activations
    if name == "ffn_down": x = F.gelu(x, approximate="tanh")
    x = x.bfloat16()
    truth = x.float() @ w.t() + bias
    res = {}
    for rank in (32, 0):
        m = S.SVDQLinear(k, n, rank)
        for kk, vv in S.quantise_linear(w, bias, x.float().abs().amax(0), rank=rank).items():
            setattr(m, kk, vv)
        y = m(x)
        res[rank] = cos(y, truth)
        if rank == 32:
            sub = slice(0, 2048)
            emu = cos(y[sub], m.reference(x[sub]))
            t_svdq = bench(lambda: m(x))
    wb = w.bfloat16(); bb = bias.bfloat16()
    w8 = get(f"{b}.ffn.net.0.proj.weight") if name == "ffn_up" else None
    q8 = torch.round(w / (w.abs().amax(1, keepdim=True) / 127)).to(torch.int8); s8 = (w.abs().amax(1) / 127).bfloat16()
    t_bf16 = bench(lambda: F.linear(x, wb, bb))
    t_w8 = bench(lambda: F.linear(x, q8.to(torch.bfloat16) * s8[:, None], bb))
    tf = 2 * M * n * k / 1e9
    print(f"{name:8s} N={n:5d} K={k:5d} | cos vs fp32: SVDQ r32 {res[32]:.5f}, W4A4 r0 {res[0]:.5f}, bf16 {cos(F.linear(x, wb, bb), truth):.5f} | "
          f"kernel vs emulation {emu:.5f} | bf16 {t_bf16:6.2f} ms ({tf / t_bf16:4.0f} TF) W8A16 {t_w8:6.2f} ms SVDQ {t_svdq:6.2f} ms "
          f"({tf / t_svdq:4.0f} TOPS) -> {t_w8 / t_svdq:.2f}x vs W8A16", flush=True)
    del w, x, truth; torch.xpu.empty_cache()
