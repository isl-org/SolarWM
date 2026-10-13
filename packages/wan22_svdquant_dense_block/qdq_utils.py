"""Pack, unpack, and scale utilities for SVDQuant INT4.

Fail-closed contract:
- Symmetric signed 4-bit representation in [-8, 7].
- Packed into int8 with little-endian nibbles (element 0 in low nibble, element 1 in high nibble).
- Group size G = 64 default.
- Dynamic activation quantization.
- E8M0 scale support (power-of-2 scales via bit shift).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

E8M0_BIAS: int = 127


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

    even = (packed_int8 << 4) >> 4  # Arithmetic shift preserves sign of low 4 bits
    odd = packed_int8 >> 4          # Arithmetic shift preserves sign of high 4 bits

    unpacked = torch.empty(orig_shape, dtype=torch.int8, device=packed_int8.device)
    unpacked[..., 0::2] = even
    unpacked[..., 1::2] = odd
    return unpacked


def convert_int4_weights_to_vnni8(packed_int4: torch.Tensor) -> torch.Tensor:
    """Convert row-major packed INT4 weights into XMX VNNI-8 dwords.

    The input stores a weight matrix as ``[N, K // 2]`` little-endian signed
    nibbles.  The output has shape ``[N // 16, K // 8, 16]`` and dtype
    ``int32``.  Each dword contains eight consecutive K nibbles for one output
    column; the final dimension is the 16-column XMX subgroup tile.

    This is a pure layout transform: ``unpack_vnni8_weights`` reconstructs
    exactly the original packed tensor.
    """
    if packed_int4.ndim != 2:
        raise ValueError(f"expected [N, K//2] packed weights, got {tuple(packed_int4.shape)}")
    n, k_half = packed_int4.shape
    k = k_half * 2
    if n % 16 or k % 64:
        raise ValueError(f"VNNI-8 requires N % 16 == 0 and K % 64 == 0, got N={n}, K={k}")

    # Each group of four packed bytes is eight K nibbles.  Preserve byte/nibble
    # order by viewing only after making the source byte-contiguous.
    packed_u8 = packed_int4.contiguous().view(torch.uint8)
    dwords_per_row = k // 8
    row_dwords = packed_u8.view(n, dwords_per_row, 4).contiguous().view(torch.int32)
    return row_dwords.view(n // 16, 16, dwords_per_row).permute(0, 2, 1).contiguous()


def unpack_vnni8_weights(vnni8: torch.Tensor) -> torch.Tensor:
    """Invert :func:`convert_int4_weights_to_vnni8` to packed signed INT4."""
    if vnni8.ndim != 3 or vnni8.shape[2] != 16:
        raise ValueError(f"expected [N//16, K//8, 16] int32 VNNI-8 tensor, got {tuple(vnni8.shape)}")
    if vnni8.dtype != torch.int32:
        raise TypeError(f"VNNI-8 tensor must be int32, got {vnni8.dtype}")

    n_tiles, dwords_per_row, _ = vnni8.shape
    row_dwords = vnni8.permute(0, 2, 1).contiguous().view(n_tiles * 16, dwords_per_row)
    return row_dwords.view(torch.uint8).view(torch.int8).reshape(n_tiles * 16, dwords_per_row * 4)


def verify_int4_weights_vnni8(packed_int4: torch.Tensor) -> bool:
    """Return whether VNNI-8 conversion round-trips packed INT4 weights exactly."""
    return torch.equal(packed_int4.contiguous(), unpack_vnni8_weights(convert_int4_weights_to_vnni8(packed_int4)))

def convert_int4_weights_to_dpas_b(packed_int4: torch.Tensor) -> torch.Tensor:
    """Pre-pack INT4 weights as the XMX DPAS B operand register image.

    Returns ``[N // 16, K // 64, 8, 16]`` int32: row-wise VNNI-8 grouped per
    64-wide K step.  Dword ``d`` of lane ``n`` holds K ``[8d, 8d + 8)`` of
    output column ``n``, so each 512-byte tile is fetched with one subgroup
    block read straight into the DPAS operand registers.
    """
    n, k_half = packed_int4.shape
    return convert_int4_weights_to_vnni8(packed_int4).view(n // 16, (2 * k_half) // 64, 8, 16)


def unpack_dpas_b_weights(dpas_b: torch.Tensor) -> torch.Tensor:
    """Invert :func:`convert_int4_weights_to_dpas_b` exactly."""
    if dpas_b.ndim != 4 or dpas_b.shape[2:] != (8, 16):
        raise ValueError(f"expected [N//16, K//64, 8, 16] int32 tensor, got {tuple(dpas_b.shape)}")
    n_tiles, k_groups, _, _ = dpas_b.shape
    return unpack_vnni8_weights(dpas_b.reshape(n_tiles, k_groups * 8, 16))


def verify_int4_weights_dpas_b(packed_int4: torch.Tensor) -> bool:
    """Return whether the DPAS B layout round-trips packed INT4 weights exactly."""
    return torch.equal(
        packed_int4.contiguous(),
        unpack_dpas_b_weights(convert_int4_weights_to_dpas_b(packed_int4)),
    )


def e8m0_to_f32(e8m0: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 uint8 exponent bytes into IEEE 754 float32 via bit shift."""
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
    biased_exp = torch.round(log2_scale).to(torch.int32) + E8M0_BIAS
    biased_exp = torch.clamp(biased_exp, 0, 255)
    return biased_exp.to(torch.uint8)


def quantize_activation_int4(
    x: torch.Tensor,
    group_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize activations to signed INT4 [-8, 7] with per-token, per-group scaling.

    Args:
        x: [M, K] tensor in BF16/FP32.
        group_size: Number of channels per scale group (default 64).

    Returns:
        packed: [M, K // 2] int8 packed INT4 tensor.
        scales: [M, K // group_size] scale factors in input dtype.
    """
    orig_shape = x.shape
    k = orig_shape[-1]
    x_2d = x.reshape(-1, k).to(torch.float32)
    m = x_2d.shape[0]
    num_groups = k // group_size

    x_grouped = x_2d.view(m, num_groups, group_size)
    group_max = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=True)
    scales = torch.clamp(group_max / 7.0, min=1e-10)
    x_q = torch.clamp(torch.round(x_grouped / scales), -8, 7).to(torch.int8)

    x_q_2d = x_q.view(m, k)
    packed = pack_int4_signed(x_q_2d)
    scales_out = scales.view(m, num_groups).to(x.dtype)
    return packed, scales_out
