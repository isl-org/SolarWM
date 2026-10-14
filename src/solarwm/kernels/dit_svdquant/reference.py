"""Reference PyTorch implementations for SVDQuant W4A4 + BF16 LoRA and BF16 baseline kernels.

These reference kernels serve as the golden numerical truth and contract specification
for performance engineering and optimization agents targeting Intel Arc XPU or other accelerators.

Specifications:
1. BF16 Reference Kernels:
   - bf16_linear: Standard linear projection Y = X @ W.T + bias.
   - bf16_linear_activation: Linear projection + fused activation (none, gelu_tanh, silu, gelu).
   - bf16_ffn: Standard two-layer FFN (FFN0 + gelu_tanh -> FFN2).
   - bf16_attention_projections: Grouped or individual Q, K, V, O projections.

2. SVDQuant W4A4 + BF16 Reference Kernels:
   - E8M0 scale support: 8-bit unsigned exponent (scale = 2^(E - 127)) implemented via bit-shift.
   - Dynamic symmetric & unsigned INT4 activation quantization (group_size = 64).
   - Channel smoothing (X / s_in).
   - Rank-16 BF16 LoRA residual branch (X / s_in @ lora_a.T @ lora_b.T).
   - Fused activation epilogue (gelu_tanh, silu, gelu, none).
   - Optional fused output INT4 quantization (qout, oscales) for chaining into subsequent layers.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from .qdq_utils import pack_int4_signed, unpack_int4_signed


# ============================================================================
# 1. E8M0 Scale Utilities (Power-of-2 Exponent Scales via Bit Shifts)
# ============================================================================

E8M0_BIAS = 127
MBITS_F32 = 23


def e8m0_to_f32(e8m0: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 uint8 exponent bytes into IEEE 754 float32 via a single bit shift.

    Mathematical definition:
      scale = 2^(E - 127) for E > 0, and 0.0 for E == 0.
    In IEEE 754 float32:
      Bit 31: sign (0)
      Bits 30-23: biased exponent (E)
      Bits 22-0: mantissa (0)
    Therefore:
      float32_bits = (E << 23)
    """
    assert e8m0.dtype == torch.uint8, "E8M0 tensor must be torch.uint8"
    biased_exp = e8m0.to(torch.int32)
    bits = biased_exp << MBITS_F32
    f32 = bits.view(torch.float32)
    # Zero out when exponent is 0
    return torch.where(biased_exp == 0, torch.zeros_like(f32), f32)


def f32_to_e8m0(x: torch.Tensor) -> torch.Tensor:
    """Quantize positive float32 scales to nearest power-of-two E8M0 exponent bytes."""
    assert x.dtype in (torch.float32, torch.bfloat16, torch.float16)
    x_f32 = x.to(torch.float32)
    # scale = 2^(E - 127) => log2(scale) = E - 127 => E = round(log2(scale)) + 127
    log2_scale = torch.log2(torch.clamp(x_f32, min=2.0**-126))
    biased_exp = torch.round(log2_scale).to(torch.int32) + E8M0_BIAS
    biased_exp = torch.clamp(biased_exp, 0, 255)
    return biased_exp.to(torch.uint8)


# ============================================================================
# 2. Existing BF16 PyTorch Baseline Kernels
# ============================================================================


def bf16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference BF16 Linear layer: Y = X @ weight.T + bias."""
    return F.linear(x, weight, bias)


def bf16_linear_activation(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str = "none",
) -> torch.Tensor:
    """Reference BF16 Linear layer with fused activation epilogue."""
    out = F.linear(x, weight, bias)
    if activation == "gelu_tanh":
        return F.gelu(out, approximate="tanh")
    elif activation == "silu":
        return F.silu(out)
    elif activation == "gelu":
        return F.gelu(out, approximate="none")
    return out


def bf16_ffn(
    x: torch.Tensor,
    w_up: torch.Tensor,
    b_up: torch.Tensor | None,
    w_down: torch.Tensor,
    b_down: torch.Tensor | None,
    activation: str = "gelu_tanh",
) -> torch.Tensor:
    """Reference BF16 two-layer Feed-Forward Network: down(activation(up(x)))."""
    h = bf16_linear_activation(x, w_up, b_up, activation=activation)
    return bf16_linear(h, w_down, b_down)


def bf16_attention_projections(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    bias_q: torch.Tensor | None = None,
    bias_k: torch.Tensor | None = None,
    bias_v: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference BF16 Q, K, V attention projections."""
    q = F.linear(x, w_q, bias_q)
    k = F.linear(x, w_k, bias_k)
    v = F.linear(x, w_v, bias_v)
    return q, k, v


# ============================================================================
# 3. Dynamic INT4 Activation Quantization
# ============================================================================


def quantize_activation_int4(
    x: torch.Tensor,
    group_size: int = 64,
    unsigned: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize activations to INT4 with per-token, per-group scaling.

    Args:
        x: [M, K] tensor in BF16/FP32.
        group_size: Number of channels per scale group (default 64).
        unsigned: If True, quantize to unsigned [0, 15] (useful after GELU shift).
                  If False, quantize to signed [-8, 7].

    Returns:
        packed: [M, K // 2] uint8 or int8 packed INT4 tensor.
        scales: [M, K // group_size] scale factors in input dtype.
    """
    orig_shape = x.shape
    k = orig_shape[-1]
    x_2d = x.reshape(-1, k).to(torch.float32)
    m = x_2d.shape[0]
    num_groups = k // group_size

    x_grouped = x_2d.view(m, num_groups, group_size)

    if unsigned:
        group_max = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=True)
        scales = torch.clamp(group_max / 15.0, min=1e-10)
        x_q = torch.clamp(torch.round(x_grouped / scales), 0, 15).to(torch.int8)
    else:
        group_max = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=True)
        scales = torch.clamp(group_max / 7.0, min=1e-10)
        x_q = torch.clamp(torch.round(x_grouped / scales), -8, 7).to(torch.int8)

    x_q_2d = x_q.view(m, k)
    packed = pack_int4_signed(x_q_2d)
    scales_out = scales.view(m, num_groups).to(x.dtype)
    return packed, scales_out


# ============================================================================
# 4. SVDQuant W4A4 + BF16 LoRA Reference Kernel
# ============================================================================


def svdquant_w4a4_linear_ref(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    s_in: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    group_size: int = 64,
    activation: str = "none",
    fuse_quant_out: bool = False,
    unsigned_quant_out: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation of SVDQuant W4A4 Linear + LoRA + Activation + optional QOut.

    Formula:
      Y = activation( (QDQ(X / s_in) @ QDQ(W).T) + ((X / s_in) @ lora_a.T @ lora_b.T) + bias )

    Features:
      - Compatible with both standard float scales (BF16/FP16) and E8M0 uint8 scales.
      - Optional channel smoothing s_in.
      - Rank-16 LoRA branch.
      - Epilogue activation: 'none', 'gelu_tanh', 'silu', 'gelu'.
      - Optional fused output quantization: returns (out_bf16, qout, oscales).
    """
    orig_shape = x.shape
    k = orig_shape[-1]
    x_2d = x.reshape(-1, k).to(torch.float32)
    m = x_2d.shape[0]

    # Step 1: Input channel smoothing
    if s_in is not None:
        x_s = x_2d / s_in.to(torch.float32)
    else:
        x_s = x_2d

    # Step 2: Dynamic groupwise activation quantize & dequantize (simulates A4)
    num_groups = k // group_size
    x_grouped = x_s.view(m, num_groups, group_size)
    x_amax = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=True)
    x_scale = torch.clamp(x_amax / 7.0, min=1e-10)
    x_q = torch.clamp(torch.round(x_grouped / x_scale), -8, 7)
    x_dequant = (x_q * x_scale).view(m, k)

    # Step 3: Weight unpack & dequantize (simulates W4, supports E8M0 via bit shifts)
    w_unpacked = unpack_int4_signed(w_packed).to(torch.float32)  # [N, K]
    n = w_unpacked.shape[0]
    w_grouped = w_unpacked.view(n, num_groups, group_size)

    if w_scale.dtype == torch.uint8:
        # E8M0 scale: bit-shift to float32
        w_scale_f32 = e8m0_to_f32(w_scale)
    else:
        w_scale_f32 = w_scale.to(torch.float32)

    # Support both [N, num_groups] and [num_groups, N] scale layouts
    if w_scale_f32.shape == (n, num_groups):
        w_scale_expanded = w_scale_f32.unsqueeze(-1)  # [N, num_groups, 1]
    elif w_scale_f32.shape == (num_groups, n):
        w_scale_expanded = w_scale_f32.t().unsqueeze(-1)  # [N, num_groups, 1]
    else:
        raise ValueError(f"Unexpected w_scale shape: {w_scale.shape}")

    w_dequant = (w_grouped * w_scale_expanded).view(n, k)

    # Step 4: Main INT4 matmul
    out = torch.matmul(x_dequant, w_dequant.t())

    # Step 5: Low-Rank BF16 branch
    if lora_a is not None and lora_b is not None:
        lora_out = torch.matmul(x_s, lora_a.to(torch.float32).t())
        lora_out = torch.matmul(lora_out, lora_b.to(torch.float32).t())
        out = out + lora_out

    # Step 6: Bias addition
    if bias is not None:
        out = out + bias.to(torch.float32)

    # Step 7: Nonlinearity activation function
    if activation == "gelu_tanh":
        out = F.gelu(out, approximate="tanh")
    elif activation == "silu":
        out = F.silu(out)
    elif activation == "gelu":
        out = F.gelu(out, approximate="none")

    out_bf16 = out.to(x.dtype).view(*orig_shape[:-1], n)

    # Step 8: Optional fused output INT4 quantization
    if fuse_quant_out:
        qout, oscales = quantize_activation_int4(
            out_bf16, group_size=group_size, unsigned=unsigned_quant_out
        )
        return out_bf16, qout, oscales

    return out_bf16


# ============================================================================
# 5. SVDQuant Fused FFN (MLP) Reference
# ============================================================================


def svdquant_fused_ffn_ref(
    x: torch.Tensor,
    # FFN0 (Up-projection) parameters
    w0_packed: torch.Tensor,
    w0_scale: torch.Tensor,
    s0_in: torch.Tensor | None,
    b0: torch.Tensor | None,
    l0_a: torch.Tensor | None,
    l0_b: torch.Tensor | None,
    # FFN2 (Down-projection) parameters
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    s2_in: torch.Tensor | None,
    b2: torch.Tensor | None,
    l2_a: torch.Tensor | None,
    l2_b: torch.Tensor | None,
    group_size: int = 64,
    fuse_intermediate_quant: bool = True,
) -> torch.Tensor:
    """Reference execution of a 2-layer SVDQuant FFN block.

    If fuse_intermediate_quant is True, simulates chaining where the FFN0
    activation epilogue directly quantizes the hidden activations to INT4
    and passes them to FFN2 without full BF16 memory round-trips.
    """
    if fuse_intermediate_quant:
        # FFN0 with fused GELU + INT4 quantization
        h_bf16, qout, oscales = svdquant_w4a4_linear_ref(
            x=x,
            w_packed=w0_packed,
            w_scale=w0_scale,
            s_in=s0_in,
            bias=b0,
            lora_a=l0_a,
            lora_b=l0_b,
            group_size=group_size,
            activation="gelu_tanh",
            fuse_quant_out=True,
        )
        # FFN2 takes the quantized hidden state or h_bf16
        out = svdquant_w4a4_linear_ref(
            x=h_bf16,
            w_packed=w2_packed,
            w_scale=w2_scale,
            s_in=s2_in,
            bias=b2,
            lora_a=l2_a,
            lora_b=l2_b,
            group_size=group_size,
            activation="none",
            fuse_quant_out=False,
        )
    else:
        h = svdquant_w4a4_linear_ref(
            x=x,
            w_packed=w0_packed,
            w_scale=w0_scale,
            s_in=s0_in,
            bias=b0,
            lora_a=l0_a,
            lora_b=l0_b,
            group_size=group_size,
            activation="gelu_tanh",
            fuse_quant_out=False,
        )
        out = svdquant_w4a4_linear_ref(
            x=h,
            w_packed=w2_packed,
            w_scale=w2_scale,
            s_in=s2_in,
            bias=b2,
            lora_a=l2_a,
            lora_b=l2_b,
            group_size=group_size,
            activation="none",
            fuse_quant_out=False,
        )

    return out


__all__ = [
    "E8M0_BIAS",
    "e8m0_to_f32",
    "f32_to_e8m0",
    "bf16_linear",
    "bf16_linear_activation",
    "bf16_ffn",
    "bf16_attention_projections",
    "quantize_activation_int4",
    "svdquant_w4a4_linear_ref",
    "svdquant_fused_ffn_ref",
]
