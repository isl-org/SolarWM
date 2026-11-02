"""Fused operations for Wan2.2 DiT on Intel Arc XPU.

This module provides four memory-bound fusion primitives:
1. `triton_layernorm_adaln`:
   Fuses un-affine LayerNorm + AdaLN scale and shift modulation:
     y = LayerNorm(x) * (1 + scale) + shift
   Avoids materializing intermediate normalized tensors in DRAM.

2. `triton_rmsnorm`:
   Fuses sum-of-squares reduction + rsqrt + affine weight multiplication:
     y = WanRMSNorm(x) * weight
   Eliminates intermediate pow(2) and reduction buffer round-trips.

3. `fused_gated_residual_add_`:
   In-place vector fused multiply-accumulate for token-modulated residual updates:
     x += y * gate
   Uses native oneAPI vector instructions on Intel XPU via torch.addcmul_.

4. `onednn_linear_gelu_tanh`:
   Fuses Linear projection GEMM + GELU(tanh) activation epilogue:
     y = GELU_tanh(x @ W^T + b)
   Dispatches to oneDNN post-op accumulator epilogue via torch.ops.mkldnn._linear_pointwise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _layernorm_adaln_kernel(
    X_ptr,
    SCALE_ptr,
    SHIFT_ptr,
    OUT_ptr,
    M: int,
    S: int,
    C: int,
    mod_seqlen: int,
    eps: float,
    stride_xb: int,
    stride_xs: int,
    stride_xc: int,
    stride_sb: int,
    stride_stok: int,
    stride_sc: int,
    stride_shiftb: int,
    stride_shifttok: int,
    stride_shiftc: int,
    stride_outb: int,
    stride_outs: int,
    stride_outc: int,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    b = row // S
    s = row % S
    tok = s // mod_seqlen

    cols = tl.arange(0, BLOCK_C)
    c_mask = cols < C

    x_ptrs = X_ptr + b * stride_xb + s * stride_xs + cols * stride_xc
    x = tl.load(x_ptrs, mask=c_mask, other=0.0).to(tl.float32)

    # LayerNorm reduction across C (mean and variance in FP32)
    mean = tl.sum(x, axis=0) / C
    diff = tl.where(c_mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * rstd

    # Load token-specific AdaLN modulation parameters
    s_ptrs = SCALE_ptr + b * stride_sb + tok * stride_stok + cols * stride_sc
    shift_ptrs = SHIFT_ptr + b * stride_shiftb + tok * stride_shifttok + cols * stride_shiftc
    scale = tl.load(s_ptrs, mask=c_mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptrs, mask=c_mask, other=0.0).to(tl.float32)

    out = x_norm * (1.0 + scale) + shift
    out_ptrs = OUT_ptr + b * stride_outb + s * stride_outs + cols * stride_outc
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=c_mask)


@triton.jit
def _rmsnorm_kernel(
    X_ptr,
    WEIGHT_ptr,
    OUT_ptr,
    M: int,
    C: int,
    eps: float,
    stride_xm: int,
    stride_xc: int,
    stride_outm: int,
    stride_outc: int,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_C)
    c_mask = cols < C

    x_ptrs = X_ptr + row * stride_xm + cols * stride_xc
    x = tl.load(x_ptrs, mask=c_mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    rstd = 1.0 / tl.sqrt(sum_sq / C + eps)

    w = tl.load(WEIGHT_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)
    out = (x * rstd) * w

    out_ptrs = OUT_ptr + row * stride_outm + cols * stride_outc
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=c_mask)


@torch.compiler.disable
def triton_layernorm_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused LayerNorm + AdaLN modulation for DiT blocks on Intel XPU.

    Computes:
        out = LayerNorm(x) * (1 + scale) + shift
    directly in SRAM registers, eliminating DRAM round-trips for the intermediate
    un-affine normalized tensor and intermediate un-shifted scaled tensor.
    """
    if not (x.is_xpu and x.dtype == torch.bfloat16):
        orig_scale = scale if scale.ndim == 4 else scale.unsqueeze(2)
        orig_shift = shift if shift.ndim == 4 else shift.unsqueeze(2)
        num_tokens = scale.shape[1]
        mod_seqlen = x.shape[1] // num_tokens
        norm = F.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype)
        return (
            norm.unflatten(1, (num_tokens, mod_seqlen)) * (1.0 + orig_scale) + orig_shift
        ).flatten(1, 2)

    orig_shape = x.shape
    if x.ndim == 2:
        x = x.unsqueeze(0)
    b, s, c = x.shape

    if scale.ndim == 4:
        scale = scale.squeeze(2)
        shift = shift.squeeze(2)

    num_tokens = scale.shape[1]
    mod_seqlen = s // num_tokens
    out = torch.empty_like(x)

    m = b * s
    block_c = 1 << (c - 1).bit_length()
    grid = (m,)

    _layernorm_adaln_kernel[grid](
        x,
        scale,
        shift,
        out,
        m,
        s,
        c,
        mod_seqlen,
        float(eps),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        shift.stride(0),
        shift.stride(1),
        shift.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_C=block_c,
    )
    return out.view(orig_shape)


@torch.compiler.disable
def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused WanRMSNorm on Intel XPU.

    Computes:
        out = (x / sqrt(mean(x^2) + eps)) * weight
    in a single pass over SRAM registers.
    """
    if not (x.is_xpu and x.dtype == torch.bfloat16):
        return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)).type_as(x) * weight

    orig_shape = x.shape
    c = orig_shape[-1]
    x_2d = x.reshape(-1, c)
    m = x_2d.shape[0]
    out_2d = torch.empty_like(x_2d)

    block_c = 1 << (c - 1).bit_length()
    grid = (m,)
    _rmsnorm_kernel[grid](
        x_2d,
        weight,
        out_2d,
        m,
        c,
        float(eps),
        x_2d.stride(0),
        x_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        BLOCK_C=block_c,
    )
    return out_2d.view(orig_shape)


def fused_gated_residual_add_(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    num_tokens: int | None = None,
    mod_seqlen: int | None = None,
) -> torch.Tensor:
    """In-place token-modulated residual accumulation: x += y * gate.

    Uses native oneAPI vector fused multiply-accumulate on Intel XPU.
    """
    if gate.ndim == 4:
        gate = gate.squeeze(2)
    b, s, c = x.shape
    if num_tokens is None:
        num_tokens = gate.shape[1]
    if mod_seqlen is None:
        mod_seqlen = s // num_tokens

    gate_expanded = gate.unsqueeze(2).expand(b, num_tokens, mod_seqlen, c).reshape(b, s, c)
    return x.addcmul_(y, gate_expanded)


@torch.compiler.disable
def onednn_linear_gelu_tanh(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused Linear GEMM + GELU(tanh) post-op epilogue via oneDNN.

    Computes:
        out = GELU_tanh(x @ weight.T + bias)
    directly in oneDNN accumulator registers, eliminating intermediate DRAM write-back.
    """
    if x.is_xpu and hasattr(torch.ops.mkldnn, "_linear_pointwise"):
        return torch.ops.mkldnn._linear_pointwise(x, weight, bias, "gelu", [], "tanh")
    return F.gelu(F.linear(x, weight, bias), approximate="tanh")
