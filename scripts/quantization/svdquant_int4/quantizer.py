"""Symmetric INT4 group-64 quantizer and packing utilities compatible with B70 XMX ABI."""

from __future__ import annotations

import torch


def pack_int4_signed(unpacked_int8: torch.Tensor) -> torch.Tensor:
    """Pack signed int8 in [-8, 7] into int8 (2 values per byte, low nibble first).

    Input: [..., K], where K is even.
    Output: [..., K // 2], int8.
    """
    assert unpacked_int8.shape[-1] % 2 == 0, "Last dimension must be even"
    clamped = torch.clamp(unpacked_int8, -8, 7).to(torch.int8)
    u4 = clamped & 0x0F
    even = u4[..., 0::2]
    odd = u4[..., 1::2]
    packed = even | (odd << 4)
    return packed.to(torch.int8)


def unpack_int4_signed(packed_int8: torch.Tensor) -> torch.Tensor:
    """Unpack packed int8 (2 values per byte) into signed int8 in [-8, 7].

    Input: [..., K // 2], int8.
    Output: [..., K], int8.
    """
    orig_shape = list(packed_int8.shape)
    orig_shape[-1] *= 2
    even = (packed_int8 << 4) >> 4  # Sign extension from bit 3
    odd = packed_int8 >> 4          # Sign extension from bit 7
    unpacked = torch.empty(orig_shape, dtype=torch.int8, device=packed_int8.device)
    unpacked[..., 0::2] = even
    unpacked[..., 1::2] = odd
    return unpacked


def quantize_weights_rtn_g64(
    weight: torch.Tensor, group_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize 2D weight [OC, IC] to symmetric INT4 group-wise RTN.

    Returns:
        qweight: [OC, IC // 2] int8 packed (low nibble first)
        w_scale: [OC, IC // group_size] bfloat16
    """
    oc, ic = weight.shape
    assert ic % group_size == 0, f"IC ({ic}) must be divisible by group_size ({group_size})"
    w_fp = weight.float()
    w_grouped = w_fp.view(oc, ic // group_size, group_size)
    amax = w_grouped.abs().amax(dim=-1).clamp(min=1e-8)
    scale = amax / 7.0  # symmetric [-7, 7], max code 7

    # Round-to-nearest in [-8, 7]
    q_unpacked = torch.round(w_grouped / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    q_unpacked = q_unpacked.view(oc, ic)

    qweight = pack_int4_signed(q_unpacked)
    w_scale = scale.to(torch.bfloat16)
    return qweight, w_scale


def dequantize_int4_weights(
    qweight: torch.Tensor, w_scale: torch.Tensor, group_size: int = 64, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Dequantize packed INT4 weights back to continuous tensor [OC, IC]."""
    q_unpacked = unpack_int4_signed(qweight).to(dtype)
    oc, ic = q_unpacked.shape
    q_grouped = q_unpacked.view(oc, ic // group_size, group_size)
    scale_grouped = w_scale.to(dtype).view(oc, ic // group_size, 1)
    w_dequant = (q_grouped * scale_grouped).view(oc, ic)
    return w_dequant


def quantize_activations_dynamic(
    x: torch.Tensor, group_size: int = 64, clip_ratio: float = 1.0
) -> torch.Tensor:
    """Dynamic per-row per-group-64 symmetric INT4 pseudo-quantization.

    Applied along inner feature dimension.
    Formula:
        clip_amax = clip_ratio * absmax(x_group)
        scale = clip_amax / 7.0
        x_q = clamp(round(x / scale), -8, 7) * scale
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1]).float()
    tokens, ic = x_flat.shape
    assert ic % group_size == 0, f"IC ({ic}) must be divisible by group_size ({group_size})"

    xg = x_flat.view(tokens, ic // group_size, group_size)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (amax * clip_ratio) / 7.0
    scale = scale.clamp(min=1e-8)

    q = torch.round(xg / scale).clamp(-8, 7)
    qdq = (q * scale).view(tokens, ic)
    return qdq.reshape(orig_shape).to(x.dtype)
