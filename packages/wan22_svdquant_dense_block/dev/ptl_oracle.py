"""pytest plugin (-p ptl_oracle): replaces the Triton correctness oracle with an equivalent pure-torch one.

The Triton XPU backend fails on Panther Lake drivers (SPV_INTEL_predicated_io unsupported), so
task.py's `triton_svdquant_w4a4_gemm` is substituted by a torch implementation of the same math.
"""
import sys
import types
import torch
import torch.nn.functional as F


def torch_svdquant_ref(x, w_packed, w_scale, s_in=None, bias=None, lora_a=None, lora_b=None,
                       group_size=64, activation="none", norm_weight=None, norm_eps=1e-5, **_):
    from qdq_utils import unpack_int4_signed
    xg = x.float().view(x.shape[0], -1, group_size)
    xs = torch.clamp(xg.abs().amax(-1, keepdim=True) / 7.0, min=1e-8)  # fp32 scale, like the fused kernels
    sc = xg / xs
    sc = torch.where(sc >= 0, sc + 0.5, sc - 0.5)  # round half away from zero, as the Triton kernel
    xdq = (torch.clamp(sc.trunc(), -8, 7) * xs).view(x.shape)
    wdq = unpack_int4_signed(w_packed).float() * w_scale.float().repeat_interleave(group_size, 1)
    y = xdq @ wdq.t()
    if lora_a is not None and lora_b is not None:
        xl = (x.float() @ lora_a.float().t()).to(torch.bfloat16)
        y = y + xl.float() @ lora_b.float().t()
    if bias is not None:
        y = y + bias.float()
    if activation == "gelu_tanh":
        y = F.gelu(y, approximate="tanh")
    elif activation == "silu":
        y = F.silu(y)
    y = y.to(torch.bfloat16)
    if norm_weight is not None:
        yf = y.float()
        yf = yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + norm_eps) * norm_weight.float()
        y = yf.to(torch.bfloat16)
    return y


def pytest_configure(config):
    mod = types.ModuleType("triton_w4a4_gemm")
    mod.triton_svdquant_w4a4_gemm = torch_svdquant_ref
    sys.modules["triton_w4a4_gemm"] = mod
