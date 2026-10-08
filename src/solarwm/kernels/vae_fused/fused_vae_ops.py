"""Fused BF16 kernels for Wan2.2 VAE decoder on Intel XPU / Triton.

This module provides two memory-bound fusion primitives:
1. `triton_rms_norm_silu`:
   Fuses RMS_norm + SiLU in a single streaming pass:
     y = SiLU(RMS_norm(x) * gamma + bias)
   Avoids materializing unnormalized and pre-activation tensors in DRAM.

2. `triton_add_rms_norm_silu`:
   Fuses Residual Addition + RMS_norm + SiLU in a single streaming pass:
     sum = x + residual
     y = SiLU(RMS_norm(sum) * gamma + bias)
   Performs the addition directly in registers before reduction, writing
   the skip shortcut tensor `sum` and the activated tensor `y` in one kernel.

Both kernels operate natively on bfloat16 inputs with fp32 accumulation and
support both 5D (channels_last_3d) and 4D (channels_last) tensors.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rms_norm_silu_kernel(
    X_ptr,
    GAMMA_ptr,
    BIAS_ptr,
    OUT_ptr,
    M: int,
    C: int,
    SCALE: float,
    stride_xm: int,
    stride_xc: int,
    stride_outm: int,
    stride_outc: int,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    cols = tl.arange(0, BLOCK_C)
    c_mask = cols < C
    mask_2d = m_mask[:, None] & c_mask[None, :]

    # 1. Load weights
    gamma = tl.load(GAMMA_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(BIAS_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)

    # 2. Load input activations
    x_ptrs = X_ptr + m_offsets[:, None] * stride_xm + cols[None, :] * stride_xc
    x = tl.load(x_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

    # 3. Sum of squares reduction across channels in float32
    sum_sq = tl.sum(x * x, axis=1)[:, None]
    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, 1e-12)

    # 4. Affine transform
    normed = (x / norm) * SCALE * gamma[None, :]
    if HAS_BIAS:
        normed = normed + bias[None, :]

    # 5. SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    silu = normed / (1.0 + tl.exp(-normed))

    # 6. Store output
    out_ptrs = OUT_ptr + m_offsets[:, None] * stride_outm + cols[None, :] * stride_outc
    tl.store(out_ptrs, silu.to(tl.bfloat16), mask=mask_2d)


@triton.jit
def _fused_add_rms_norm_silu_kernel(
    X_ptr,
    RESIDUAL_ptr,
    GAMMA_ptr,
    BIAS_ptr,
    OUT_ptr,
    OUT_SUM_ptr,
    M: int,
    C: int,
    SCALE: float,
    stride_xm: int,
    stride_xc: int,
    stride_rm: int,
    stride_rc: int,
    stride_outm: int,
    stride_outc: int,
    stride_sm: int,
    stride_sc: int,
    HAS_BIAS: tl.constexpr,
    STORE_SUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    cols = tl.arange(0, BLOCK_C)
    c_mask = cols < C
    mask_2d = m_mask[:, None] & c_mask[None, :]

    # 1. Load weights
    gamma = tl.load(GAMMA_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(BIAS_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)

    # 2. Load x and residual
    x_ptrs = X_ptr + m_offsets[:, None] * stride_xm + cols[None, :] * stride_xc
    res_ptrs = RESIDUAL_ptr + m_offsets[:, None] * stride_rm + cols[None, :] * stride_rc
    x = tl.load(x_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
    res = tl.load(res_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

    # 3. In-register residual sum
    y = x + res

    # 4. Optionally write residual sum to DRAM (for downstream shortcut)
    if STORE_SUM:
        out_sum_ptrs = OUT_SUM_ptr + m_offsets[:, None] * stride_sm + cols[None, :] * stride_sc
        tl.store(out_sum_ptrs, y.to(tl.bfloat16), mask=mask_2d)

    # 5. Sum of squares reduction across channels in float32
    sum_sq = tl.sum(y * y, axis=1)[:, None]
    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, 1e-12)

    # 6. Affine transform
    normed = (y / norm) * SCALE * gamma[None, :]
    if HAS_BIAS:
        normed = normed + bias[None, :]

    # 7. SiLU
    silu = normed / (1.0 + tl.exp(-normed))

    # 8. Store activated output
    out_ptrs = OUT_ptr + m_offsets[:, None] * stride_outm + cols[None, :] * stride_outc
    tl.store(out_ptrs, silu.to(tl.bfloat16), mask=mask_2d)


def _get_strides(tensor: torch.Tensor, C: int) -> tuple[int, int]:
    """Return (stride_token, stride_channel) for contiguous or channels_last tensor."""
    if tensor.is_contiguous(memory_format=torch.channels_last_3d) or tensor.is_contiguous(memory_format=torch.channels_last):
        return C, 1
    if tensor.stride(1) == 1:
        # Channels are contiguous along dim 1
        stride_m = tensor.stride(0) // (tensor.shape[0] * tensor.shape[2:].numel()) if tensor.ndim > 2 else tensor.stride(0)
        return C, 1
    # Fallback for standard contiguous: reshape/view needed or pass raw strides
    raise ValueError(f"Tensor with shape {tensor.shape} and strides {tensor.stride()} is not in channels-first layout with contiguous channels.")


@torch.compiler.disable
def triton_rms_norm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute SiLU(RMS_norm(x) * gamma + bias) in a single fused Triton kernel.

    Parameters:
        x: Input tensor in channels_last_3d (5D) or channels_last (4D) format, bfloat16.
        gamma: Channel scale parameter of shape [C, 1, ...], bfloat16 or float32.
        bias: Optional channel bias parameter of shape [C, 1, ...], bfloat16 or float32.
        scale: Normalization factor sqrt(C). If None, computed as sqrt(C).
    """
    if x.device.type != "xpu":
        # Fallback for CPU / CUDA
        scale_val = scale if scale is not None else float(x.shape[1] ** 0.5)
        norm = torch.nn.functional.normalize(x, dim=1)
        res = norm * scale_val * gamma
        if bias is not None and isinstance(bias, torch.Tensor):
            res = res + bias
        return torch.nn.functional.silu(res)

    C = x.shape[1]
    M = x.numel() // C
    scale_val = float(scale) if scale is not None else float(C ** 0.5)

    has_bias = bias is not None and isinstance(bias, torch.Tensor)
    bias_tensor = bias.flatten().contiguous() if has_bias else gamma.flatten()

    out = torch.empty_like(x)

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_M = 4 if C <= 512 else 2
    num_warps = 8 if C >= 512 else 4
    grid = (triton.cdiv(M, BLOCK_M),)

    stride_xm, stride_xc = _get_strides(x, C)
    stride_outm, stride_outc = _get_strides(out, C)

    _fused_rms_norm_silu_kernel[grid](
        x,
        gamma.flatten().contiguous(),
        bias_tensor,
        out,
        M,
        C,
        scale_val,
        stride_xm,
        stride_xc,
        stride_outm,
        stride_outc,
        HAS_BIAS=has_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    return out


@torch.compiler.disable
def triton_add_rms_norm_silu(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale: float | None = None,
    store_sum: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute y = x + residual, and SiLU(RMS_norm(y) * gamma + bias) in one fused kernel.

    Parameters:
        x: Conv output tensor, bfloat16.
        residual: Residual shortcut tensor, bfloat16.
        gamma: Channel scale parameter.
        bias: Optional channel bias parameter.
        scale: Normalization factor sqrt(C). If None, computed as sqrt(C).
        store_sum: Whether to materialize y = x + residual for downstream skip connection.

    Returns:
        (out, out_sum) if store_sum is True, else (out, None).
    """
    if x.device.type != "xpu":
        scale_val = scale if scale is not None else float(x.shape[1] ** 0.5)
        y = x + residual
        norm = torch.nn.functional.normalize(y, dim=1)
        res = norm * scale_val * gamma
        if bias is not None and isinstance(bias, torch.Tensor):
            res = res + bias
        out = torch.nn.functional.silu(res)
        return out, y if store_sum else None

    C = x.shape[1]
    M = x.numel() // C
    scale_val = float(scale) if scale is not None else float(C ** 0.5)

    has_bias = bias is not None and isinstance(bias, torch.Tensor)
    bias_tensor = bias.flatten().contiguous() if has_bias else gamma.flatten()

    out = torch.empty_like(x)
    out_sum = torch.empty_like(x) if store_sum else out

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_M = 4 if C <= 512 else 2
    num_warps = 8 if C >= 512 else 4
    grid = (triton.cdiv(M, BLOCK_M),)

    stride_xm, stride_xc = _get_strides(x, C)
    stride_rm, stride_rc = _get_strides(residual, C)
    stride_outm, stride_outc = _get_strides(out, C)
    stride_sm, stride_sc = _get_strides(out_sum, C)

    _fused_add_rms_norm_silu_kernel[grid](
        x,
        residual,
        gamma.flatten().contiguous(),
        bias_tensor,
        out,
        out_sum,
        M,
        C,
        scale_val,
        stride_xm,
        stride_xc,
        stride_rm,
        stride_rc,
        stride_outm,
        stride_outc,
        stride_sm,
        stride_sc,
        HAS_BIAS=has_bias,
        STORE_SUM=store_sum,
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    return out, out_sum if store_sum else None
