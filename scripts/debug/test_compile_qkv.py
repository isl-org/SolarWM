#!/usr/bin/env python3
"""Test torch.compile on Baseline vs Fused QKV."""

import time
import torch
import torch.nn as nn

from solarwm.backends.wan22.runtime.modeling.model import WanRMSNorm


def synchronize():
    torch.xpu.synchronize()


def time_fn(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


class BaselineModule(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim)
        self.norm_k = WanRMSNorm(dim)

    def forward(self, x):
        b, s = x.shape[:2]
        q = self.norm_q(self.q(x)).view(b, s, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(x)).view(b, s, self.num_heads, self.head_dim)
        v = self.v(x).view(b, s, self.num_heads, self.head_dim)
        return q, k, v


class FusedModule(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.norm_q = WanRMSNorm(dim)
        self.norm_k = WanRMSNorm(dim)

    def forward(self, x):
        b, s = x.shape[:2]
        q_raw, k_raw, v_raw = self.qkv(x).chunk(3, dim=-1)
        q = self.norm_q(q_raw).view(b, s, self.num_heads, self.head_dim)
        k = self.norm_k(k_raw).view(b, s, self.num_heads, self.head_dim)
        v = v_raw.contiguous().view(b, s, self.num_heads, self.head_dim)
        return q, k, v


def main():
    device = "xpu"
    dtype = torch.bfloat16
    dim = 3072
    num_heads = 24
    s = 1215

    x = torch.randn(1, s, dim, device=device, dtype=dtype)
    base = BaselineModule(dim, num_heads).to(device=device, dtype=dtype)
    fused = FusedModule(dim, num_heads).to(device=device, dtype=dtype)

    with torch.no_grad():
        fused.qkv.weight.copy_(torch.cat([base.q.weight, base.k.weight, base.v.weight], dim=0))
        fused.qkv.bias.copy_(torch.cat([base.q.bias, base.k.bias, base.v.bias], dim=0))

    print("Uncompiled:")
    t_base_eager = time_fn(lambda: base(x))
    t_fused_eager = time_fn(lambda: fused(x))
    print(f"  Baseline: {t_base_eager:.3f} ms")
    print(f"  Fused:    {t_fused_eager:.3f} ms")

    try:
        print("Compiling Baseline...")
        compiled_base = torch.compile(base)
        t_base_compiled = time_fn(lambda: compiled_base(x))
        print(f"  Compiled Baseline: {t_base_compiled:.3f} ms")
    except Exception as e:
        print("  Compiled Baseline failed:", e)

    try:
        print("Compiling Fused...")
        compiled_fused = torch.compile(fused)
        t_fused_compiled = time_fn(lambda: compiled_fused(x))
        print(f"  Compiled Fused:    {t_fused_compiled:.3f} ms")
    except Exception as e:
        print("  Compiled Fused failed:", e)


if __name__ == "__main__":
    main()
