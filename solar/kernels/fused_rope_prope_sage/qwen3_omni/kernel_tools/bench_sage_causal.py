"""Causal attention microbenchmark at the Qwen3-Omni thinker shape (H=32, Hkv=4, D=128, bf16).

Variants per sequence length (prefill from an empty cache, Sq == Skv):
  sdpa         F.scaled_dot_product_attention(is_causal=True, enable_gqa=True) on [B,H,S,D]
  sage_kernel  sage_causal.forward on pre-laid-out [B,S,H,D] tensors with K/V already expanded
  sage_e2e     what the HF integration runs. In HF prefill Q/K/V are physically [B,S,H,D]
               (projection view + transpose; RoPE keeps strides; an empty cache stores K as is),
               so [B,H,S,D] -> [B,S,H,D] is a free view and native GQA needs no repeat_kv.
  sage_e2e_rep the same but expanding K/V to 32 heads (the pre-GQA integration)

Reports median time, causal-FLOP TFLOPS (2 GEMMs x 2 FLOP x H x S^2/2 x D) and extra peak VRAM.

    /home/gta/omni_bench/env.sh python bench_sage_causal.py [--lens 2048 8192 ...] [--json out.json]
"""
import argparse, json, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/gta/omni_bench/kernel/build")
import sage_causal  # noqa: E402

H, HKV, D = 32, 4, 128


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def peak_extra(fn):
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    base = torch.xpu.memory_allocated()
    torch.xpu.reset_peak_memory_stats()
    out = fn()
    torch.xpu.synchronize()
    extra = torch.xpu.max_memory_allocated() - base
    del out
    return extra / 2**20


def sage_e2e(q, k, v):
    # q [B,H,S,D], k/v [B,Hkv,S,D] views as HF hands them over; .contiguous() is a no-op here
    return sage_causal.forward(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                               v.transpose(1, 2).contiguous(), is_causal=True)


def sage_e2e_rep(q, k, v):
    qs = q.transpose(1, 2).contiguous()
    rep = H // HKV
    ks = k[:, :, None].expand(-1, -1, rep, -1, -1).reshape(1, H, -1, D).transpose(1, 2).contiguous()
    vs = v[:, :, None].expand(-1, -1, rep, -1, -1).reshape(1, H, -1, D).transpose(1, 2).contiguous()
    return sage_causal.forward(qs, ks, vs, is_causal=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[2048, 8192, 16384, 32768, 63488])
    ap.add_argument("--json")
    a = ap.parse_args()
    torch.xpu.set_per_process_memory_fraction(0.92)
    rows = []
    print(f"{'S':>6} | {'sdpa ms':>8} {'TF':>6} {'MiB':>7} | {'kern ms':>8} {'TF':>6} {'MiB':>7} | "
          f"{'e2e ms':>8} {'TF':>6} {'MiB':>7} | {'speedup kern':>12} {'e2e':>6}")
    for S in a.lens:
        g = torch.Generator(device="xpu").manual_seed(0)
        # physical [B,S,H,D], viewed as [B,H,S,D] -- exactly what HF passes to the attention fn
        q = torch.randn(1, S, H, D, device="xpu", dtype=torch.bfloat16, generator=g).transpose(1, 2)
        k = torch.randn(1, S, HKV, D, device="xpu", dtype=torch.bfloat16, generator=g).transpose(1, 2)
        v = torch.randn(1, S, HKV, D, device="xpu", dtype=torch.bfloat16, generator=g).transpose(1, 2)
        qs = q.transpose(1, 2).contiguous()
        ks = k.repeat_interleave(H // HKV, dim=1).transpose(1, 2).contiguous()
        vs = v.repeat_interleave(H // HKV, dim=1).transpose(1, 2).contiguous()
        f_sdpa = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        f_kern = lambda: sage_causal.forward(qs, ks, vs, is_causal=True)
        f_e2e = lambda: sage_e2e(q, k, v)
        f_rep = lambda: sage_e2e_rep(q, k, v)
        flops = 4 * H * D * S * (S + 1) / 2
        r = {"S": S, "causal_flops": flops}
        for name, f in (("sdpa", f_sdpa), ("sage_kernel", f_kern), ("sage_e2e", f_e2e), ("sage_e2e_rep", f_rep)):
            ms = timeit(f, iters=10 if S <= 16384 else 5)
            r[name] = {"ms": ms, "tflops": flops / ms / 1e9, "extra_mib": peak_extra(f)}
        rows.append(r)
        s, kn, e = r["sdpa"], r["sage_kernel"], r["sage_e2e"]
        print(f"{S:>6} | {s['ms']:8.2f} {s['tflops']:6.1f} {s['extra_mib']:7.0f} | {kn['ms']:8.2f} {kn['tflops']:6.1f} "
              f"{kn['extra_mib']:7.0f} | {e['ms']:8.2f} {e['tflops']:6.1f} {e['extra_mib']:7.0f} | "
              f"{s['ms'] / kn['ms']:11.2f}x {s['ms'] / e['ms']:5.2f}x", flush=True)
        del q, k, v, qs, ks, vs
        torch.xpu.empty_cache()
    if a.json:
        json.dump({"device": torch.xpu.get_device_name(0), "torch": torch.__version__, "rows": rows},
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
