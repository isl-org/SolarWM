"""Reference implementation of the fused RoPE + PRoPE + SDPA attention pipeline.

This module provides a clean, self-contained reference for:
1. Multi-coordinate RoPE (EchoRoPE / block-relativistic RoPE) with configurable
   internal precision (float64/complex128, float32/complex64, float16/complex32).
2. Projective Positional Encoding (PRoPE) camera transformations:
   - SE(3) inversion and intrinsics lifting/inversion.
   - Per-token/per-camera projection matrices P_T (for Q), P_inv (for K, V), P (for O).
   - Application across the 32 contiguous 4-channel blocks of head_dim=128.
3. Scaled Dot-Product Attention (SDPA).
4. Output projection P applied to the attention accumulator O.
5. End-to-end composite pipeline: (Q, K, V, cameras, freqs) -> O_prope.

Layout conventions:
- Q: [batch, seqlen_q, num_heads, head_dim] (bfloat16)
- K: [batch, seqlen_kv, num_heads, head_dim] (bfloat16)
- V: [batch, seqlen_kv, num_heads, head_dim] (bfloat16)
- Output: [batch, seqlen_q, num_heads, head_dim] (bfloat16)
- Cameras: viewmats [batch, num_cams, 4, 4], Ks [batch, num_cams, 3, 3]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


# Supported RoPE precisions and their real/complex dtype pairs
ROPE_DTYPE_MAP = {
    "float64": (torch.float64, torch.complex128),
    "fp64": (torch.float64, torch.complex128),
    torch.float64: (torch.float64, torch.complex128),
    "float32": (torch.float32, torch.complex64),
    "fp32": (torch.float32, torch.complex64),
    torch.float32: (torch.float32, torch.complex64),
    "float16": (torch.float16, torch.complex32),
    "fp16": (torch.float16, torch.complex32),
    torch.float16: (torch.float16, torch.complex32),
}


def normalize_rope_precision(
    dtype_spec: Union[str, torch.dtype] = "float64",
) -> Tuple[torch.dtype, torch.dtype]:
    """Return (real_dtype, complex_dtype) for the requested RoPE precision."""
    if dtype_spec in ROPE_DTYPE_MAP:
        return ROPE_DTYPE_MAP[dtype_spec]
    if isinstance(dtype_spec, str):
        key = dtype_spec.strip().lower()
        if key in ROPE_DTYPE_MAP:
            return ROPE_DTYPE_MAP[key]
    raise ValueError(
        f"Unsupported rope_dtype: {dtype_spec}. Expected float64, float32, or float16."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. RoPE (Rotary Positional Encoding) Reference
# ─────────────────────────────────────────────────────────────────────────────

def rope_params(
    max_seq_len: int,
    dim: int,
    theta: float = 10000.0,
    complex_dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Precompute complex RoPE frequency tables [max_seq_len, dim / 2]."""
    assert dim % 2 == 0, f"dim={dim} must be even"
    pos = torch.arange(max_seq_len, dtype=torch.float64)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    )
    freqs = torch.outer(pos, inv_freq)
    # Return polar form: cos(freqs) + i * sin(freqs) = exp(i * freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    freqs_complex = torch.complex(cos, sin)
    if complex_dtype != torch.complex128:
        freqs_complex = freqs_complex.to(complex_dtype)
    return freqs_complex


def build_wan22_freqs(
    head_dim: int = 128,
    max_len: int = 1024,
    complex_dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Build the standard 3-split Wan2.2 RoPE frequency table."""
    dim_t = head_dim - 4 * (head_dim // 6)
    dim_h = 2 * (head_dim // 6)
    dim_w = 2 * (head_dim // 6)
    return torch.cat(
        [
            rope_params(max_len, dim_t, complex_dtype=complex_dtype),
            rope_params(max_len, dim_h, complex_dtype=complex_dtype),
            rope_params(max_len, dim_w, complex_dtype=complex_dtype),
        ],
        dim=1,
    )


def apply_rope(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
    rope_dtype: Union[str, torch.dtype] = "float64",
    grid_list: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> torch.Tensor:
    """Apply 3D window-relative RoPE to tokens in [B, L, H, D] layout.

    Args:
        x: Input tokens of shape [batch, seqlen, num_heads, head_dim], typically bfloat16.
        grid_sizes: [batch, 3] tensor containing (frames, height, width).
        freqs: Precomputed complex frequencies [max_frames, head_dim / 2].
        start_frame: Window-relative start frame offset.
        rope_dtype: Intermediate rotation precision (float64, float32, or float16).
        grid_list: Optional list of (frames, height, width) tuples to avoid tensor-read graph breaks.

    Returns:
        Rotated tensor matching x shape and x dtype.
    """
    rope_real_dtype, rope_complex_dtype = normalize_rope_precision(rope_dtype)
    if freqs.dtype != rope_complex_dtype:
        freqs = freqs.to(rope_complex_dtype)

    b, seq_len_total, n, d = x.shape
    c = d // 2  # number of complex pairs (e.g. 64 for d=128)
    split_t = c - 2 * (c // 3)
    split_h = c // 3
    split_w = c // 3
    freqs_t, freqs_h, freqs_w = freqs.split([split_t, split_h, split_w], dim=1)

    sizes = grid_list if grid_list is not None else grid_sizes.tolist()
    output = []

    for i, (f, h, w) in enumerate(sizes):
        seq_len = f * h * w
        valid_len = min(seq_len_total, seq_len)

        temporal_idx = torch.arange(
            start_frame, start_frame + f, device=x.device, dtype=torch.long
        )

        # Broadcast 3D frequencies: [F, 1, 1, Ct], [1, H, 1, Ch], [1, 1, W, Cw] -> [F, H, W, C]
        freqs_i_full = torch.cat(
            [
                freqs_t[temporal_idx].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)  # [seq_len, 1, C]

        if valid_len > 0:
            # Pair channels into complex numbers
            x_i = torch.view_as_complex(
                x[i, :valid_len].to(rope_real_dtype).reshape(valid_len, n, -1, 2)
            )
            freqs_i = freqs_i_full[:valid_len]
            # Complex multiplication = 2D rotation
            x_i_rot = x_i * freqs_i
            # View back as real and flatten
            x_i_real = torch.view_as_real(x_i_rot).flatten(2)
        else:
            x_i_real = x[i, :0].to(rope_real_dtype)

        if seq_len_total > valid_len:
            x_i_real = torch.cat([x_i_real, x[i, valid_len:].to(rope_real_dtype)], dim=0)

        output.append(x_i_real)

    return torch.stack(output, dim=0).type_as(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRoPE (Projective Positional Encoding) Matrix Prep & Application
# ─────────────────────────────────────────────────────────────────────────────

def transform_relative_viewmats(
    viewmats: torch.Tensor,
    transform: str = "linear",
) -> torch.Tensor:
    """Apply linear or logd4 translation compression to view matrices."""
    if transform == "linear" or transform is None:
        return viewmats
    if transform != "logd4":
        raise ValueError(f"Unknown camera translation transform: {transform}")

    translation = viewmats[..., :3, 3]
    compute_dtype = torch.float64 if viewmats.dtype == torch.float64 else torch.float32
    working = translation.to(dtype=compute_dtype)
    norm = torch.linalg.vector_norm(working, dim=-1, keepdim=True)
    safe_norm = norm.clamp_min(torch.finfo(compute_dtype).tiny)
    scale = torch.where(
        norm > 0,
        torch.log1p(norm) / (4.0 * safe_norm),
        torch.zeros_like(norm),
    )
    compressed = working * scale
    out = viewmats.clone()
    out[..., :3, 3] = compressed.to(dtype=viewmats.dtype)
    return out


def invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a batch of 4x4 SE(3) transformation matrices."""
    assert transforms.shape[-2:] == (4, 4)
    R = transforms[..., :3, :3]
    t = transforms[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = R_inv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", R_inv, t)
    out[..., 3, 3] = 1.0
    return out.to(dtype=transforms.dtype)


def lift_k(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 camera intrinsic matrices to 4x4 homogeneous matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def invert_k(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 camera intrinsic matrices (assumes zero skew)."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out.to(dtype=Ks.dtype)


def prepare_prope_matrices(
    viewmats: torch.Tensor,
    Ks: Optional[torch.Tensor] = None,
    target_seqlen: Optional[int] = None,
    camera_translation_transform: str = "linear",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-token or per-camera PRoPE projection matrices (P_T, P_inv, P).

    Args:
        viewmats: [batch, cameras, 4, 4] camera-to-world view matrices.
        Ks: Optional [batch, cameras, 3, 3] camera intrinsic matrices.
        target_seqlen: If provided and greater than cameras, expands matrices to [batch, seqlen, 4, 4].
        camera_translation_transform: "linear" or "logd4".

    Returns:
        P_T: Transpose of P, applied to Query [batch, seqlen/cams, 4, 4].
        P_inv: Inverse of P, applied to Key and Value [batch, seqlen/cams, 4, 4].
        P: Forward projection matrix, applied to Attention Output [batch, seqlen/cams, 4, 4].
    """
    batch, cameras, _, _ = viewmats.shape
    v = transform_relative_viewmats(viewmats, camera_translation_transform)

    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0]
        Ks_norm[..., 1, 1] = Ks[..., 1, 1]
        Ks_norm[..., 0, 2] = 0.0
        Ks_norm[..., 1, 2] = 0.0
        Ks_norm[..., 2, 2] = 1.0

        P = torch.einsum("...ij,...jk->...ik", lift_k(Ks_norm), v)
        P_T = P.transpose(-1, -2).to(dtype=viewmats.dtype)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            invert_se3(v),
            lift_k(invert_k(Ks_norm)),
        ).to(dtype=viewmats.dtype)
    else:
        P = v
        P_T = P.transpose(-1, -2)
        P_inv = invert_se3(v)

    # Optionally expand per-camera matrices to per-token matrices
    if target_seqlen is not None and target_seqlen > cameras:
        assert target_seqlen % cameras == 0, f"seqlen={target_seqlen} not divisible by cameras={cameras}"
        repeat = target_seqlen // cameras
        P_T = P_T.repeat_interleave(repeat, dim=1)
        P_inv = P_inv.repeat_interleave(repeat, dim=1)
        P = P.repeat_interleave(repeat, dim=1)

    return P_T, P_inv, P


def apply_prope(
    feats: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply 4x4 PRoPE matrix across all 32 contiguous 4-channel blocks in head_dim.

    Args:
        feats: Features in [batch, seqlen, num_heads, head_dim] or [batch, num_heads, seqlen, head_dim].
        matrix: Projection matrix [batch, seqlen, 4, 4] or [batch, cameras, 4, 4].

    Returns:
        Transformed features in the same layout and dtype as feats.
    """
    has_heads_at_dim1 = feats.ndim == 4 and feats.shape[1] > feats.shape[2]
    # Standardize to [batch, num_heads, seqlen, head_dim] for calculation
    if feats.ndim == 4 and feats.shape[2] != matrix.shape[1] and feats.shape[1] == matrix.shape[1]:
        # feats is [batch, seqlen, num_heads, head_dim]
        b, s, n, d = feats.shape
        x = feats.transpose(1, 2)  # [b, n, s, d]
        restore_transpose = True
    else:
        b, n, s, d = feats.shape
        x = feats
        restore_transpose = False

    D = matrix.shape[-1]  # 4
    assert d % D == 0, f"head_dim={d} must be divisible by {D}"
    num_blocks = d // D  # 32
    matrix_bf16 = matrix.to(dtype=x.dtype)

    if matrix.shape[1] == s:
        # Per-token projection: [B, S, 4, 4]
        x_reshaped = x.view(b, n, s, num_blocks, D)
        out = torch.einsum("btij,bntpj->bntpi", matrix_bf16, x_reshaped)
    else:
        # Per-camera projection: [B, C, 4, 4]
        cams = matrix.shape[1]
        assert s % cams == 0
        x_reshaped = x.view(b, n, cams, s // cams, num_blocks, D)
        out = torch.einsum("bcij,bncpkj->bncpki", matrix_bf16, x_reshaped)

    out = out.reshape(b, n, s, d)
    if restore_transpose:
        out = out.transpose(1, 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Scaled Dot-Product Attention (SDPA)
# ─────────────────────────────────────────────────────────────────────────────

def scaled_dot_product_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute Scaled Dot-Product Attention matching PyTorch's native SDPA.

    Args:
        q: Query tensor [batch, seqlen_q, num_heads, head_dim] (bfloat16).
        k: Key tensor [batch, seqlen_kv, num_heads, head_dim] (bfloat16).
        v: Value tensor [batch, seqlen_kv, num_heads, head_dim] (bfloat16).
        scale: Softmax temperature scale (defaults to 1.0 / sqrt(head_dim)).

    Returns:
        Attention output [batch, seqlen_q, num_heads, head_dim] (bfloat16).
    """
    # Transpose to [B, H, L, D] for attention calculation
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    # Use native PyTorch SDPA (which runs FlashAttention / efficient SDPA kernel)
    out_t = F.scaled_dot_product_attention(
        q_t, k_t, v_t, scale=scale, is_causal=False
    )
    return out_t.transpose(1, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Composite Fused RoPE + PRoPE + SDPA Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RoPEPRoPEConfig:
    """Configuration for RoPE, PRoPE, and Attention."""
    head_dim: int = 128
    num_heads: int = 24
    rope_dtype: str = "float64"  # "float64", "float32", or "float16"
    camera_translation_transform: str = "linear"  # "linear" or "logd4"
    frame_seqlen: int = 405  # 15 * 27 tokens per frame
    use_echorope: bool = False  # False for block-relativistic RoPE (production Stage2)


def fused_rope_prope_sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    freqs: torch.Tensor,
    q_grid_sizes: torch.Tensor,
    k_grid_sizes: torch.Tensor,
    q_viewmats: torch.Tensor,
    q_Ks: Optional[torch.Tensor] = None,
    kv_viewmats: Optional[torch.Tensor] = None,
    kv_Ks: Optional[torch.Tensor] = None,
    q_start_frame: int = 0,
    k_start_frame: int = 0,
    rope_dtype: Union[str, torch.dtype] = "float64",
    camera_translation_transform: str = "linear",
    return_intermediates: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """End-to-end reference implementation of the complete attention sequence.

    Pipeline:
        1. RoPE: Rotates Q with q_start_frame and K with k_start_frame in rope_dtype.
        2. PRoPE: Multiplies Q by P_q^T, K by P_kv^-1, V by P_kv^-1.
        3. SDPA: Scaled dot-product attention over Q_prope, K_prope, V_prope.
        4. Output PRoPE: Multiplies attention accumulator O by P_q.

    Args:
        q: [B, Lq, H, D] raw query features (bfloat16).
        k: [B, Lk, H, D] raw key features (bfloat16, sliced from KV cache).
        v: [B, Lk, H, D] raw value features (bfloat16, sliced from KV cache).
        freqs: Complex RoPE frequency table.
        q_grid_sizes: [B, 3] grid sizes for Q (frames, height, width).
        k_grid_sizes: [B, 3] grid sizes for K (frames, height, width).
        q_viewmats: [B, Cams_q, 4, 4] or [B, Lq, 4, 4] query view matrices.
        q_Ks: Optional [B, Cams_q, 3, 3] or [B, Lq, 3, 3] query intrinsics.
        kv_viewmats: Optional [B, Cams_kv, 4, 4] or [B, Lk, 4, 4] key/value view matrices.
        kv_Ks: Optional [B, Cams_kv, 3, 3] or [B, Lk, 3, 3] key/value intrinsics.
        q_start_frame: Temporal frame offset for query RoPE.
        k_start_frame: Temporal frame offset for key RoPE.
        rope_dtype: Internal RoPE precision ("float64", "float32", or "float16").
        camera_translation_transform: "linear" or "logd4".
        return_intermediates: If True, returns a dict of intermediate tensors.

    Returns:
        O_prope: Final output tensor [B, Lq, H, D] (bfloat16).
        intermediates (optional): Dictionary of all intermediate stages.
    """
    b, lq, h, d = q.shape
    _, lk, _, _ = k.shape

    # Step 1: RoPE rotation on Q and K
    q_rope = apply_rope(
        q,
        grid_sizes=q_grid_sizes,
        freqs=freqs,
        start_frame=q_start_frame,
        rope_dtype=rope_dtype,
    )
    k_rope = apply_rope(
        k,
        grid_sizes=k_grid_sizes,
        freqs=freqs,
        start_frame=k_start_frame,
        rope_dtype=rope_dtype,
    )

    # Step 2: Prepare PRoPE matrices
    P_q_T, _, P_q = prepare_prope_matrices(
        q_viewmats,
        Ks=q_Ks,
        target_seqlen=lq,
        camera_translation_transform=camera_translation_transform,
    )
    if kv_viewmats is not None:
        _, P_kv_inv, _ = prepare_prope_matrices(
            kv_viewmats,
            Ks=kv_Ks,
            target_seqlen=lk,
            camera_translation_transform=camera_translation_transform,
        )
    else:
        # Self-attention over identical Q and KV context
        _, P_kv_inv, _ = prepare_prope_matrices(
            q_viewmats,
            Ks=q_Ks,
            target_seqlen=lk,
            camera_translation_transform=camera_translation_transform,
        )

    # Step 3: Apply PRoPE
    # Q: P_q^T
    q_prope = apply_prope(q_rope, P_q_T)
    # K: P_kv^-1
    k_prope = apply_prope(k_rope, P_kv_inv)
    # V: P_kv^-1 (V does not receive RoPE)
    v_prope = apply_prope(v, P_kv_inv)

    # Step 4: SDPA
    o_sdpa = scaled_dot_product_attention_reference(q_prope, k_prope, v_prope)

    # Step 5: Output PRoPE (apply P_q to attention output)
    o_prope = apply_prope(o_sdpa, P_q)

    if return_intermediates:
        intermediates = {
            "q_rope": q_rope,
            "k_rope": k_rope,
            "P_q_T": P_q_T,
            "P_kv_inv": P_kv_inv,
            "P_q": P_q,
            "q_prope": q_prope,
            "k_prope": k_prope,
            "v_prope": v_prope,
            "o_sdpa": o_sdpa,
            "o_prope": o_prope,
        }
        return o_prope, intermediates

    return o_prope
