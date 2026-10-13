"""Check the SYCL SVDQuant paths (unfused and fused) against the PyTorch reference.

Usage: python verify_svdquant_fused.py [checkpoint_dir] [block]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from solarwm.backends.wan22.runtime.modeling.model import WanRMSNorm
from solarwm.backends.wan22.runtime.modeling.svdquant_linear import SVDQuantModelManager

CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/ssheorey/models/SolarWM/SolarWM-5B-sgf-stage2-81f/svdquant_int4/sdpa_k4"
)
BLOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 17
DIM, FFN_DIM, M = 3072, 14336, 1215
DEV = torch.device("xpu")


class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q, self.k, self.v = nn.Identity(), nn.Identity(), nn.Identity()
        self.norm_q = WanRMSNorm(DIM, eps=1e-6)
        self.norm_k = WanRMSNorm(DIM, eps=1e-6)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attn()
        self.ffn = nn.Sequential(nn.Identity(), nn.GELU(approximate="tanh"), nn.Identity())


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(30)])


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def timed(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    torch.manual_seed(0)
    model = _Model().to(DEV, torch.bfloat16)
    for blk in model.blocks:
        for norm in (blk.self_attn.norm_q, blk.self_attn.norm_k):
            norm.weight.data.uniform_(0.5, 1.5)
    manager = SVDQuantModelManager(model)

    x = torch.randn(1, M, DIM, device=DEV, dtype=torch.bfloat16)
    x[..., torch.randperm(DIM)[:16]] *= 20.0  # outlier channels
    h_in = torch.randn(1, M, DIM, device=DEV, dtype=torch.bfloat16)

    with torch.no_grad():
        manager.load_svdquant_checkpoint(CKPT, device=DEV, fused=False)
        blk = model.blocks[BLOCK]
        attn = blk.self_attn
        for bin_name in ("0", "commit"):
            manager.set_current_bin(bin_name)
            print(f"--- block {BLOCK}, bin {bin_name}: rho q/k/v/up/down = "
                  f"{attn.q.rho}/{attn.k.rho}/{attn.v.rho}/{blk.ffn[0].rho}/{blk.ffn[2].rho}")
            for name, layer, inp in (("q", attn.q, x), ("k", attn.k, x), ("v", attn.v, x),
                                     ("ffn.0", blk.ffn[0], h_in)):
                out = layer(inp)
                ref = layer.reference(inp.reshape(-1, DIM)).view_as(out)
                print(f"  unfused {name:6s} rel L2 vs reference: {rel(out, ref):.4f}")
        unfused = {}
        manager.set_current_bin("0")
        unfused["qkv"] = [attn.norm_q(attn.q(x)), attn.norm_k(attn.k(x)), attn.v(x)]
        unfused["ffn"] = blk.ffn(h_in)
        t_unfused_qkv = timed(lambda: (attn.norm_q(attn.q(x)), attn.norm_k(attn.k(x)), attn.v(x)))
        t_unfused_ffn = timed(lambda: blk.ffn(h_in))

        manager.load_svdquant_checkpoint(CKPT, device=DEV, fused=True)
        blk = model.blocks[BLOCK]
        attn = blk.self_attn
        assert attn.q.shared_input is not None and attn.q.norm is not None
        for bin_name in ("0", "commit"):
            manager.set_current_bin(bin_name)
            print(f"--- fused, bin {bin_name}")
            outs = [attn.norm_q(attn.q(x)), attn.norm_k(attn.k(x)), attn.v(x)]
            for name, layer, out in zip(("q+norm", "k+norm", "v"), (attn.q, attn.k, attn.v), outs):
                ref = layer.reference(x.reshape(-1, DIM)).view_as(out)
                print(f"  fused {name:7s} rel L2 vs reference: {rel(out, ref):.4f}")
            y = blk.ffn(h_in)
            up, down = blk.ffn.up, blk.ffn.down
            h_ref = torch.nn.functional.gelu(up.reference(h_in.reshape(-1, DIM)).float(), approximate="tanh")
            y_ref = down.reference(h_ref.to(torch.bfloat16)).view_as(y)
            print(f"  fused ffn     rel L2 vs reference: {rel(y, y_ref):.4f}")

        manager.set_current_bin("0")
        fused_qkv = [attn.norm_q(attn.q(x)), attn.norm_k(attn.k(x)), attn.v(x)]
        for name, a, b in zip(("q+norm", "k+norm", "v"), fused_qkv, unfused["qkv"]):
            print(f"  bin 0 fused vs unfused {name:7s}: {rel(a, b):.4f}")
        print(f"  bin 0 fused vs unfused ffn    : {rel(blk.ffn(h_in), unfused['ffn']):.4f}")
        t_fused_qkv = timed(lambda: (attn.norm_q(attn.q(x)), attn.norm_k(attn.k(x)), attn.v(x)))
        t_fused_ffn = timed(lambda: blk.ffn(h_in))
        print(f"--- timing (bin 0, M={M}): QKV(+norm) unfused {t_unfused_qkv:.3f} ms, fused {t_fused_qkv:.3f} ms;"
              f" FFN unfused {t_unfused_ffn:.3f} ms, fused {t_fused_ffn:.3f} ms")


if __name__ == "__main__":
    main()
