"""Pack, unpack, and pseudo-QDQ reference utilities for SVDQuant INT4.

Fail-closed contract:
- Symmetric signed 4-bit representation in [-8, 7].
- Packed into int8 with little-endian nibbles (element 0 in low nibble, element 1 in high nibble).
- Group size G = 64 default.
- Channel smoothing and rank-16 residual LoRA branch.
- E8M0 scale support (power-of-2 scales via bit shift).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def pack_int4_signed(unpacked_int8: torch.Tensor) -> torch.Tensor:
    """Pack signed int8 values in [-8, 7] into packed int8 (2 values per byte).

    Input: [..., K], where K must be even.
    Output: [..., K // 2], int8.
    """
    assert unpacked_int8.shape[-1] % 2 == 0, "Dimension must be divisible by 2"
    clamped = torch.clamp(unpacked_int8, -8, 7).to(torch.int8)

    # Convert to unsigned 4-bit nibbles [0, 15] for bitwise OR
    u4 = clamped & 0x0F
    even = u4[..., 0::2]
    odd = u4[..., 1::2]

    packed_u8 = even | (odd << 4)
    return packed_u8.to(torch.int8)


def unpack_int4_signed(packed_int8: torch.Tensor) -> torch.Tensor:
    """Unpack packed int8 (2 signed 4-bit values per byte) into signed int8 in [-8, 7].

    Input: [..., K // 2], int8.
    Output: [..., K], int8.
    """
    orig_shape = list(packed_int8.shape)
    orig_shape[-1] *= 2

    # Bit manipulation with sign extension
    even = (packed_int8 << 4) >> 4  # Arithmetic shift preserves sign of low 4 bits
    odd = packed_int8 >> 4          # Arithmetic shift preserves sign of high 4 bits

    unpacked = torch.empty(orig_shape, dtype=torch.int8, device=packed_int8.device)
    unpacked[..., 0::2] = even
    unpacked[..., 1::2] = odd
    return unpacked


def convert_int4_weights_to_dpas_b(packed_int4: torch.Tensor) -> torch.Tensor:
    """Pre-pack W4 weights as the XMX INT4 DPAS B operand register image.

    Output shape is ``[N // 16, K // 64, 8, 16]`` int32: row-wise VNNI-8 per
    64-wide K step.  Dword ``d`` of lane ``n`` holds K ``[8d, 8d + 8)`` of
    output column ``n`` (low nibble first), so the kernel fetches each tile
    with one subgroup block read.  Produce it once while loading a quantized
    model, not during ``forward``.
    """
    if packed_int4.ndim != 2:
        raise ValueError(f"expected [N, K//2] packed weights, got {tuple(packed_int4.shape)}")
    n, k_half = packed_int4.shape
    k = k_half * 2
    if n % 16 or k % 64:
        raise ValueError(f"DPAS B packing requires N % 16 == 0 and K % 64 == 0, got N={n}, K={k}")

    row_dwords = packed_int4.contiguous().view(torch.uint8).view(n, k // 8, 4).contiguous().view(torch.int32)
    vnni8 = row_dwords.view(n // 16, 16, k // 8).permute(0, 2, 1).contiguous()
    return vnni8.view(n // 16, k // 64, 8, 16)


def e8m0_to_f32(e8m0: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 uint8 exponent bytes into IEEE 754 float32 via a single bit shift."""
    assert e8m0.dtype == torch.uint8, "E8M0 tensor must be torch.uint8"
    biased_exp = e8m0.to(torch.int32)
    bits = biased_exp << 23
    f32 = bits.view(torch.float32)
    return torch.where(biased_exp == 0, torch.zeros_like(f32), f32)


def convert_w_scales_to_dpas(w_scale: torch.Tensor) -> torch.Tensor:
    """Pre-transpose per-group weight scales ``[N, K // 64]`` to f32 ``[K // 64, N]``.

    Lets the XMX kernel fetch every output-column scale of a K step with one
    subgroup block read.  E8M0 (uint8) scales are decoded to f32 here, once.
    """
    if w_scale.ndim != 2:
        raise ValueError(f"expected [N, K//64] weight scales, got {tuple(w_scale.shape)}")
    scale_f32 = e8m0_to_f32(w_scale) if w_scale.dtype == torch.uint8 else w_scale.to(torch.float32)
    return scale_f32.t().contiguous()


def f32_to_e8m0(x: torch.Tensor) -> torch.Tensor:
    """Quantize positive float32 scales to nearest power-of-two E8M0 exponent bytes."""
    x_f32 = x.to(torch.float32)
    log2_scale = torch.log2(torch.clamp(x_f32, min=2.0**-126))
    biased_exp = torch.round(log2_scale).to(torch.int32) + 127
    biased_exp = torch.clamp(biased_exp, 0, 255)
    return biased_exp.to(torch.uint8)


def pseudo_qdq_linear(
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
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pseudo-quantized reference emulation of SVDQuant linear layer.

    Used as numerical ground truth to verify kernel parity.
    Y = Activation((QDQ(X / s_in) @ QDQ(W).T) + ((X / s_in) @ A.T @ B.T) + bias)
    """
    orig_shape = x.shape
    k = orig_shape[-1]
    x_2d = x.reshape(-1, k).to(torch.float32)

    # 1. Channel smoothing
    if s_in is not None:
        x_2d = x_2d / s_in.to(torch.float32)

    # 2. Dynamic groupwise activation quantize & dequantize
    m = x_2d.shape[0]
    num_groups = k // group_size
    x_grouped = x_2d.view(m, num_groups, group_size)
    x_amax = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=True)
    x_scale = torch.clamp(x_amax / 7.0, min=1e-8)
    x_q = torch.clamp(torch.round(x_grouped / x_scale), -8, 7)
    x_dequant = (x_q * x_scale).view(m, k)

    # 3. Weight unpack & dequantize (supports E8M0 scales via bit shift)
    w_unpacked = unpack_int4_signed(w_packed).to(torch.float32)  # [N, K]
    n = w_unpacked.shape[0]
    w_grouped = w_unpacked.view(n, num_groups, group_size)

    if w_scale.dtype == torch.uint8:
        w_scale_f32 = e8m0_to_f32(w_scale)
    else:
        w_scale_f32 = w_scale.to(torch.float32)

    if w_scale_f32.shape == (n, num_groups):
        w_scale_expanded = w_scale_f32.unsqueeze(-1)
    elif w_scale_f32.shape == (num_groups, n):
        w_scale_expanded = w_scale_f32.t().unsqueeze(-1)
    else:
        w_scale_expanded = w_scale_f32.unsqueeze(-1)

    w_dequant = (w_grouped * w_scale_expanded).view(n, k)

    # 4. Main INT4 matmul
    out = torch.matmul(x_dequant, w_dequant.t())

    # 5. Low-rank branch
    if lora_a is not None and lora_b is not None:
        lora_out = torch.matmul(x_2d, lora_a.to(torch.float32).t())
        lora_out = torch.matmul(lora_out, lora_b.to(torch.float32).t())
        out = out + lora_out

    # 6. Bias
    if bias is not None:
        out = out + bias.to(torch.float32)

    # 7. Nonlinearity activation epilogue
    if activation == "gelu_tanh":
        out = F.gelu(out, approximate="tanh")
    elif activation == "silu":
        out = F.silu(out)
    elif activation == "gelu":
        out = F.gelu(out, approximate="none")

    out_bf16 = out.to(x.dtype).view(*orig_shape[:-1], n)

    if fuse_quant_out:
        num_out_groups = n // group_size
        out_grouped = out.view(m, num_out_groups, group_size)
        out_amax = torch.amax(torch.abs(out_grouped), dim=-1, keepdim=True)
        out_scale = torch.clamp(out_amax / 7.0, min=1e-8)
        out_q = torch.clamp(torch.round(out_grouped / out_scale), -8, 7).to(torch.int8)
        packed_out = pack_int4_signed(out_q.view(m, n))
        return out_bf16, packed_out, out_scale.view(m, num_out_groups).to(x.dtype)

    return out_bf16
