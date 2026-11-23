"""Run the exact SYCL source on a binary fixture produced by bench_opencl.cpp."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prefix")
    p.add_argument("--build-dir", default="/home/gta/omni_bench/kernel/build")
    p.add_argument("--b", type=int, default=1)
    p.add_argument("--sq", type=int, required=True)
    p.add_argument("--sk", type=int, required=True)
    p.add_argument("--hq", type=int, default=32)
    p.add_argument("--hkv", type=int, default=4)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--noncausal", action="store_true")
    a = p.parse_args()

    sys.path.insert(0, a.build_dir)
    import sage_causal

    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16

    def load(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        bits = torch.from_numpy(np.fromfile(f"{a.prefix}.{name}.bits", dtype=np.uint16).copy())
        t = bits.view(torch.int16).view(dtype)
        return t.reshape(shape).to("xpu").contiguous()

    q = load("q", (a.b, a.sq, a.hq, a.d))
    k = load("k", (a.b, a.sk, a.hkv, a.d))
    v = load("v", (a.b, a.sk, a.hkv, a.d))
    sycl = sage_causal.forward(q, k, v, is_causal=not a.noncausal)
    torch.xpu.synchronize()
    expected = load("opencl", (a.b, a.sq, a.hq, a.d))

    delta = (sycl.float() - expected.float()).abs()
    rel = torch.linalg.vector_norm(sycl.float() - expected.float()) / torch.linalg.vector_norm(sycl.float())
    cosine = torch.nn.functional.cosine_similarity(sycl.float().flatten(), expected.float().flatten(), dim=0)
    print(f"SYCL vs OpenCL: max_abs={delta.max().item():.8g} mean_abs={delta.mean().item():.8g} rel={rel.item():.8g} cosine={cosine.item():.9f}")
    print(f"nonzero_bits={(sycl.view(torch.int16) != expected.view(torch.int16)).sum().item()} / {sycl.numel()}")
    out = sycl.view(torch.int16).cpu().numpy().astype(np.uint16, copy=False)
    out.tofile(f"{a.prefix}.sycl.bits")


if __name__ == "__main__":
    main()
