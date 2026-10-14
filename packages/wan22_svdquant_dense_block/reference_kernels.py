"""Reference implementations for Wan2.2 TI2V-5B Camera Transformer Block.

Provides:
1. BF16 PyTorch baseline kernels (the runtime performance reference).
2. SVDQuant W4A4 + BF16 LoRA reference implementations.
3. Wan2.2 RMSNorm normalization for query and key projections.
"""

from __future__ import annotations

from typing import Tuple, Optional, Union
import torch
import torch.nn.functional as F

try:
    from .qdq_utils import unpack_int4_signed, e8m0_to_f32, quantize_activation_int4
except ImportError:
    from qdq_utils import unpack_int4_signed, e8m0_to_f32, quantize_activation_int4


def wan_rmsnorm(x: torch.Tensor, weight: Optional[torch.Tensor] = None, eps: float = 1e-5) -> torch.Tensor:
    """Wan2.2 RMSNorm: x * rsqrt(mean(x^2, dim=-1) + eps) * weight."""
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    normed = x_f32 * torch.rsqrt(variance + eps)
    if weight is not None:
        normed = normed * weight.to(torch.float32)
    return normed.to(x.dtype)


def bf16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    lora_a: Optional[torch.Tensor] = None,
    lora_b: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-5,
) -> torch.Tensor:
    """Reference BF16 Linear layer: Y = X @ weight.T + bias (+ LoRA if present) (+ RMSNorm if present)."""
    out = F.linear(x, weight, bias)
    if lora_a is not None and lora_b is not None:
        lora_act = F.linear(x, lora_a)  # [M, r]
        out = out + F.linear(lora_act, lora_b)  # [M, N]
    if norm_weight is not None:
        out = wan_rmsnorm(out, norm_weight, eps=norm_eps)
    return out


def bf16_linear_activation(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    lora_a: Optional[torch.Tensor] = None,
    lora_b: Optional[torch.Tensor] = None,
    activation: str = "none",
) -> torch.Tensor:
    """Reference BF16 Linear layer with fused activation epilogue."""
    out = bf16_linear(x, weight, bias, lora_a, lora_b)
    if activation == "gelu_tanh":
        return F.gelu(out, approximate="tanh")
    elif activation == "silu":
        return F.silu(out)
    elif activation == "gelu":
        return F.gelu(out, approximate="none")
    return out


def bf16_dense_block(
    x: torch.Tensor,
    q_w: torch.Tensor, q_b: Optional[torch.Tensor], q_la: Optional[torch.Tensor], q_lb: Optional[torch.Tensor], q_norm_w: Optional[torch.Tensor],
    k_w: torch.Tensor, k_b: Optional[torch.Tensor], k_la: Optional[torch.Tensor], k_lb: Optional[torch.Tensor], k_norm_w: Optional[torch.Tensor],
    v_w: torch.Tensor, v_b: Optional[torch.Tensor], v_la: Optional[torch.Tensor], v_lb: Optional[torch.Tensor],
    ffn_up_w: torch.Tensor, ffn_up_b: Optional[torch.Tensor], ffn_up_la: Optional[torch.Tensor], ffn_up_lb: Optional[torch.Tensor],
    ffn_down_w: torch.Tensor, ffn_down_b: Optional[torch.Tensor], ffn_down_la: Optional[torch.Tensor], ffn_down_lb: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the full BF16 dense block: 3 projections (Q/K with RMSNorm, V) + 2 FFN layers."""
    q = bf16_linear(x, q_w, q_b, q_la, q_lb, norm_weight=q_norm_w)
    k = bf16_linear(x, k_w, k_b, k_la, k_lb, norm_weight=k_norm_w)
    v = bf16_linear(x, v_w, v_b, v_la, v_lb)
    ffn_up = bf16_linear_activation(x, ffn_up_w, ffn_up_b, ffn_up_la, ffn_up_lb, activation="gelu_tanh")
    ffn_down = bf16_linear(ffn_up, ffn_down_w, ffn_down_b, ffn_down_la, ffn_down_lb)
    return q, k, v, ffn_down


def svdquant_w4a4_linear_ref(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    lora_a: Optional[torch.Tensor] = None,
    lora_b: Optional[torch.Tensor] = None,
    group_size: int = 64,
    activation: str = "none",
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-5,
    x_scales: Optional[torch.Tensor] = None,
    x_bf16: Optional[torch.Tensor] = None,
    fuse_quant_out: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Mathematical emulation of SVDQuant INT4 linear with BF16 LoRA, optional RMSNorm, and optional output quantization."""
    # 1. Determine input representation (already INT4 or BF16)
    if x.dtype == torch.int8:
        x_q_packed = x
        assert x_scales is not None, "x_scales required when x is int8"
        x_unpacked = unpack_int4_signed(x_q_packed).to(torch.float32)
        x_scales_expanded = x_scales.to(torch.float32).repeat_interleave(group_size, dim=-1)
        x_dequant = x_unpacked * x_scales_expanded
        lora_input = x_bf16 if x_bf16 is not None else x_dequant.to(torch.bfloat16)
    else:
        # Dynamic quantization of input
        x_q_packed, calc_scales = quantize_activation_int4(x, group_size=group_size)
        x_unpacked = unpack_int4_signed(x_q_packed).to(torch.float32)
        x_scales_expanded = calc_scales.to(torch.float32).repeat_interleave(group_size, dim=-1)
        x_dequant = x_unpacked * x_scales_expanded
        lora_input = x.to(torch.bfloat16)

    # 2. Dequantize weights
    w_unpacked = unpack_int4_signed(w_packed).to(torch.float32)
    if w_scale.dtype == torch.uint8:
        w_scale_f32 = e8m0_to_f32(w_scale)
    else:
        w_scale_f32 = w_scale.to(torch.float32)
    w_scales_expanded = w_scale_f32.repeat_interleave(group_size, dim=-1)
    w_dequant = w_unpacked * w_scales_expanded

    # 3. INT4 GEMM
    out = torch.matmul(x_dequant, w_dequant.T)

    # 4. Low-rank BF16 branch
    if lora_a is not None and lora_b is not None:
        x_lora = torch.matmul(lora_input, lora_a.to(torch.bfloat16).T)
        out_lora = torch.matmul(x_lora, lora_b.to(torch.bfloat16).T)
        out = out + out_lora.to(torch.float32)

    # 5. Bias
    if bias is not None:
        out = out + bias.to(torch.float32)

    # 6. Activation
    if activation == "gelu_tanh":
        out = F.gelu(out, approximate="tanh")
    elif activation == "silu":
        out = F.silu(out)
    elif activation == "gelu":
        out = F.gelu(out, approximate="none")

    # 7. Optional RMSNorm
    if norm_weight is not None:
        out = wan_rmsnorm(out, norm_weight, eps=norm_eps)

    out_bf16 = out.to(torch.bfloat16)

    # 8. Optional Fused INT4 Output Quantization
    if fuse_quant_out:
        q_out_packed, out_scales = quantize_activation_int4(out_bf16, group_size=group_size)
        return out_bf16, q_out_packed, out_scales

    return out_bf16
