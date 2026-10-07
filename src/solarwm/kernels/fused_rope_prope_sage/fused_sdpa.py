"""The reference kernel: fused RoPE -> PRoPE -> SDPA -> output PRoPE.

``fused_rope_prope_sdpa_split`` implements the fusion boundary described in the
RoPE-PRoPE-attention fusion guide, with the attention itself left to the vendor
kernel. RoPE and PRoPE are fused into one streaming pass per tensor and the
output projection into one pass over the attention result, so every
``roped_*`` and ``*_prope`` intermediate disappears: what remains is one read
and one write per tensor feeding ``F.scaled_dot_product_attention``.

The channel de-interleave that RoPE and PRoPE both need is done with a bit
trick rather than a register shuffle: four contiguous bfloat16 channels are
exactly one int64, so a dense 2D block load of the int64 view hands the kernel
all four PRoPE lanes, and shifts and masks unpack them with no layout
conversion at all.

The guide also describes a single kernel that keeps Q/K/V in registers across
an online-softmax loop. That was implemented and measured at ~10-25x slower on
this backend -- ``tl.split`` / ``tl.join`` on a register tile lowers to
per-element shared-memory traffic -- so it is not kept here. ``FUSION_NOTES.md``
records the measurements and the alternatives that were tried.

Following the guide's "WS18 numerical trap" warning, ``P`` and ``R`` are never
pre-multiplied: RoPE and PRoPE stay two sequential register operations, and the
unfused pipeline's bfloat16 rounding boundary between them (and between SDPA
and the output projection) is preserved exactly.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .unfused_kernel import normalize_rope_precision, prepare_prope_matrices


# ─────────────────────────────────────────────────────────────────────────────
# RoPE (cos, sin) table construction
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _rope_table_kernel(
    FR, OUT, GRID,
    seq_len, start_frame, max_rows,
    stride_ob, stride_gb,
    C: tl.constexpr, SPLIT_T: tl.constexpr, SPLIT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Expand the 3-split Wan2.2 frequency table into a per-token (cos, sin) table.

    ``OUT[b, n]`` is laid out as four contiguous 32-wide planes::

        [ cos(pair 2p) | sin(pair 2p) | cos(pair 2p+1) | sin(pair 2p+1) ]

    which is exactly what the transform kernel's four block pointers read.
    ``(frames, height, width)`` is read from ``GRID`` on the device so the host
    never has to synchronise to learn the grid shape.
    """
    pid = tl.program_id(0)
    b = tl.program_id(1)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)

    gb = GRID + b * stride_gb
    f = tl.load(gb + 0)
    h_size = tl.load(gb + 1)
    w = tl.load(gb + 2)
    hw = h_size * w
    live = rows < tl.minimum(seq_len, f * hw)

    fi = tl.minimum(rows // hw + start_frame, max_rows - 1)
    hi = (rows // w) % h_size
    wi = rows % w

    p = tl.arange(0, C // 2)
    out_base = OUT + b * stride_ob + rows[:, None] * (2 * C)

    for half in tl.static_range(2):
        c = 2 * p + half
        row = tl.where(c[None, :] < SPLIT_T, fi[:, None],
                       tl.where(c[None, :] < SPLIT_T + SPLIT_H, hi[:, None], wi[:, None]))
        idx = (row * C + c[None, :]) * 2
        co = tl.where(live[:, None], tl.load(FR + idx, mask=live[:, None], other=1.0), 1.0)
        si = tl.where(live[:, None], tl.load(FR + idx + 1, mask=live[:, None], other=0.0), 0.0)
        store = rows < seq_len
        # plane-major: [cos(2p) | sin(2p) | cos(2p+1) | sin(2p+1)], C // 2 wide each
        tl.store(out_base + half * C + p[None, :],
                 co.to(OUT.dtype.element_ty), mask=store[:, None])
        tl.store(out_base + half * C + C // 2 + p[None, :],
                 si.to(OUT.dtype.element_ty), mask=store[:, None])


def build_rope_table(
    freqs_real: torch.Tensor,
    grid_sizes: torch.Tensor,
    seq_len: int,
    start_frame: int,
) -> torch.Tensor:
    """``[batch, seq_len, 2 * C]`` plane-major (cos, sin) table.

    ``freqs_real`` is ``torch.view_as_real(freqs)``, already carrying the
    reference's rotation precision.
    """
    c = freqs_real.shape[1]
    batch = grid_sizes.shape[0]
    out = torch.empty(batch, seq_len, 2 * c, dtype=freqs_real.dtype,
                      device=freqs_real.device)
    grid_sizes = grid_sizes.to(device=freqs_real.device).contiguous()
    block_t = 64
    _rope_table_kernel[(triton.cdiv(seq_len, block_t), batch)](
        freqs_real, out, grid_sizes,
        seq_len, start_frame, freqs_real.shape[0],
        out.stride(0), grid_sizes.stride(0),
        C=c, SPLIT_T=c - 2 * (c // 3), SPLIT_H=c // 3, BLOCK_T=block_t,
        num_warps=4,
    )
    return out


def _freqs_real(freqs: torch.Tensor, rope_dtype: Union[str, torch.dtype]) -> torch.Tensor:
    """``view_as_real(freqs)`` in the reference's rotation precision."""
    real_dtype, complex_dtype = normalize_rope_precision(rope_dtype)
    if freqs.dtype != complex_dtype:
        freqs = freqs.to(complex_dtype)
    table_dtype = torch.float64 if real_dtype == torch.float64 else torch.float32
    return torch.view_as_real(freqs).to(table_dtype).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Fused RoPE + PRoPE transform (one streaming pass, no shuffles)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _unpack4(x):
    """int64 tile holding four contiguous bfloat16 channels -> four fp32 planes.

    A bfloat16 bit pattern shifted into the high half of an int32 *is* the
    float32 with the same value, so the widening conversion is free.
    """
    y0 = ((x & 0xFFFF) << 16).to(tl.int32).to(tl.float32, bitcast=True)
    y1 = (((x >> 16) & 0xFFFF) << 16).to(tl.int32).to(tl.float32, bitcast=True)
    y2 = (((x >> 32) & 0xFFFF) << 16).to(tl.int32).to(tl.float32, bitcast=True)
    y3 = (((x >> 48) & 0xFFFF) << 16).to(tl.int32).to(tl.float32, bitcast=True)
    return y0, y1, y2, y3


@triton.jit
def _round_fp16(x):
    """Round a float32 to the nearest float16, result kept in float32.

    ``x.to(tl.float16).to(tl.float32)`` is folded away by the backend, so the
    round-to-nearest-even is done on the bit pattern. The shift widens for
    float16 subnormals and values that overflow float16 saturate to infinity,
    which is what the reference's complex32 store does.
    """
    u = x.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    mag = u & 0x7FFFFFFF
    sign = u & 0x80000000
    exp = (mag >> 23) - 127
    shift = tl.minimum(13 + tl.maximum(0, -14 - exp), 31)
    lsb = (mag >> shift) & 1
    rounded = ((mag + (1 << (shift - 1)) - 1 + lsb) >> shift) << shift
    y = (sign | rounded).to(tl.int32).to(tl.float32, bitcast=True)
    overflow = (mag >= 0x477FF000) & (mag <= 0x7F800000)
    return tl.where(overflow, tl.where(x > 0, float("inf"), -float("inf")), y)


@triton.jit
def _pack4(z0, z1, z2, z3):
    """Four fp32 planes -> int64 tile of four round-to-nearest-even bfloat16."""
    u0 = z0.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int64) & 0xFFFF
    u1 = z1.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int64) & 0xFFFF
    u2 = z2.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int64) & 0xFFFF
    u3 = z3.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int64) & 0xFFFF
    return u0 | (u1 << 16) | (u2 << 32) | (u3 << 48)


@triton.jit
def rope4(y0, y1, y2, y3, c0, s0, c1, s1, ROPE: tl.constexpr):
    """Rotate the four plane tiles and round back to bfloat16, as the reference does.

    ``ROPE`` selects the rotation precision: 1 = float32, 2 = float16,
    3 = float64. torch's complex multiply contracts into an FMA -- reproducing
    that contraction (rather than two rounded products) is what makes the
    rotation bit-identical to the reference.
    """
    if ROPE == 3:
        # rope_dtype="float64": the reference rotates in complex128.
        d0, d1 = y0.to(tl.float64), y1.to(tl.float64)
        d2, d3 = y2.to(tl.float64), y3.to(tl.float64)
        r0 = tl.math.fma(d0, c0, -(d1 * s0)).to(tl.float32)
        r1 = tl.math.fma(d0, s0, d1 * c0).to(tl.float32)
        r2 = tl.math.fma(d2, c1, -(d3 * s1)).to(tl.float32)
        r3 = tl.math.fma(d2, s1, d3 * c1).to(tl.float32)
    else:
        r0 = tl.math.fma(y0, c0, -(y1 * s0))
        r1 = tl.math.fma(y0, s0, y1 * c0)
        r2 = tl.math.fma(y2, c1, -(y3 * s1))
        r3 = tl.math.fma(y2, s1, y3 * c1)
        if ROPE == 2:
            # rope_dtype="float16": the reference stores the rotation as
            # complex32 before converting to bfloat16.
            r0 = _round_fp16(r0)
            r1 = _round_fp16(r1)
            r2 = _round_fp16(r2)
            r3 = _round_fp16(r3)
    # The reference stores the rotation back to bfloat16 before PRoPE.
    return (r0.to(tl.bfloat16).to(tl.float32), r1.to(tl.bfloat16).to(tl.float32),
            r2.to(tl.bfloat16).to(tl.float32), r3.to(tl.bfloat16).to(tl.float32))


@triton.jit
def prope4(y0, y1, y2, y3, P, cam):
    """Apply the per-token 4x4 projection gathered from ``P`` to the plane tiles."""
    pb = P + cam * 16
    m00 = tl.load(pb + 0)[:, None]
    m01 = tl.load(pb + 1)[:, None]
    m02 = tl.load(pb + 2)[:, None]
    m03 = tl.load(pb + 3)[:, None]
    m10 = tl.load(pb + 4)[:, None]
    m11 = tl.load(pb + 5)[:, None]
    m12 = tl.load(pb + 6)[:, None]
    m13 = tl.load(pb + 7)[:, None]
    m20 = tl.load(pb + 8)[:, None]
    m21 = tl.load(pb + 9)[:, None]
    m22 = tl.load(pb + 10)[:, None]
    m23 = tl.load(pb + 11)[:, None]
    m30 = tl.load(pb + 12)[:, None]
    m31 = tl.load(pb + 13)[:, None]
    m32 = tl.load(pb + 14)[:, None]
    m33 = tl.load(pb + 15)[:, None]
    return (m00 * y0 + m01 * y1 + m02 * y2 + m03 * y3,
            m10 * y0 + m11 * y1 + m12 * y2 + m13 * y3,
            m20 * y0 + m21 * y1 + m22 * y2 + m23 * y3,
            m30 * y0 + m31 * y1 + m32 * y2 + m33 * y3)


# The transform is bandwidth bound, so the useful configuration space is flat:
# every BLOCK_T in 16..64 lands within ~3% of the memory roof, while BLOCK_T
# >= 128 falls off a cliff (18 ms at 256). The sweep is kept to three points so
# autotuning cannot pick a pathological config out of run-to-run timing noise.
_TRANSFORM_CONFIGS = [
    triton.Config({"BLOCK_T": bt, "grf_mode": "default"}, num_warps=8, num_stages=2)
    for bt in (16, 32, 64)
]


@triton.autotune(configs=_TRANSFORM_CONFIGS, key=["seq_len"])
@triton.jit
def _transform_kernel(
    X, Y, CS, P,
    stride_xb, stride_xn, stride_xh,
    stride_yb, stride_yn, stride_yh,
    stride_cs, stride_p,
    seq_len, tpc, ncam,
    PLANES: tl.constexpr, ROPE: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """``y = PRoPE(RoPE(x))`` for one head of one token block, in one pass.

    ``ROPE`` selects the rotation precision the reference used: 0 = no rotation
    (V and the output projection), 1 = float32, 2 = float16, 3 = float64.
    """
    h = tl.program_id(0)        # fastest axis: neighbours share the CS tile
    pid = tl.program_id(1)
    b = tl.program_id(2)
    start = pid * BLOCK_T

    X_bp = tl.make_block_ptr(X + b * stride_xb + h * stride_xh, (seq_len, PLANES),
                             (stride_xn, 1), (start, 0), (BLOCK_T, PLANES), (1, 0))
    y0, y1, y2, y3 = _unpack4(tl.load(X_bp, boundary_check=(0,)))

    if ROPE > 0:
        cs = CS + b * stride_cs
        c0 = tl.load(tl.make_block_ptr(cs, (seq_len, 4 * PLANES), (4 * PLANES, 1),
                                       (start, 0), (BLOCK_T, PLANES), (1, 0)),
                     boundary_check=(0,))
        s0 = tl.load(tl.make_block_ptr(cs, (seq_len, 4 * PLANES), (4 * PLANES, 1),
                                       (start, PLANES), (BLOCK_T, PLANES), (1, 0)),
                     boundary_check=(0,))
        c1 = tl.load(tl.make_block_ptr(cs, (seq_len, 4 * PLANES), (4 * PLANES, 1),
                                       (start, 2 * PLANES), (BLOCK_T, PLANES), (1, 0)),
                     boundary_check=(0,))
        s1 = tl.load(tl.make_block_ptr(cs, (seq_len, 4 * PLANES), (4 * PLANES, 1),
                                       (start, 3 * PLANES), (BLOCK_T, PLANES), (1, 0)),
                     boundary_check=(0,))
        y0, y1, y2, y3 = rope4(y0, y1, y2, y3, c0, s0, c1, s1, ROPE)

    rows = start + tl.arange(0, BLOCK_T)
    cam = tl.minimum(rows // tpc, ncam - 1)
    z0, z1, z2, z3 = prope4(y0, y1, y2, y3, P + b * stride_p, cam)

    Y_bp = tl.make_block_ptr(Y + b * stride_yb + h * stride_yh, (seq_len, PLANES),
                             (stride_yn, 1), (start, 0), (BLOCK_T, PLANES), (1, 0))
    tl.store(Y_bp, _pack4(z0, z1, z2, z3), boundary_check=(0,))


ROPE_MODE = {torch.float32: 1, torch.float16: 2, torch.float64: 3}


def supports(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Whether the Triton path can handle these tensors.

    ``head_dim // 4`` is a Triton block width, so it has to be a power of two,
    and the int64 view of the bfloat16 channels needs a contiguous head_dim.
    """
    if q.device.type != "xpu":
        return False
    if not all(t.dtype == torch.bfloat16 for t in (q, k, v)):
        return False
    d = q.shape[-1]
    if d % 4 or d != k.shape[-1] or d != v.shape[-1]:
        return False
    planes = d // 4
    return planes & (planes - 1) == 0


def rope_prope_transform(
    x: torch.Tensor,
    matrix: torch.Tensor,
    cos_sin: Optional[torch.Tensor] = None,
    rope_mode: int = 0,
    in_layout: str = "NHD",
    out_layout: str = "NHD",
) -> torch.Tensor:
    """Fused RoPE (optional) + PRoPE over bfloat16 features.

    ``in_layout`` / ``out_layout`` select ``[B, L, H, D]`` ("NHD") or
    ``[B, H, L, D]`` ("HND"); the kernel is stride-driven, so re-laying the
    tensor out costs nothing and lets SDPA see contiguous ``[B, H, L, D]``.
    """
    if x.dtype != torch.bfloat16:
        raise TypeError("the fused transform expects bfloat16 features")
    if not x.is_contiguous():
        x = x.contiguous()        # the int64 view needs packed head_dim
    tok, head = (1, 2) if in_layout == "NHD" else (2, 1)
    b, seq_len, nh, d = x.shape[0], x.shape[tok], x.shape[head], x.shape[3]
    if d % 4:
        raise ValueError(f"head_dim={d} must be a multiple of 4")
    ncam = matrix.shape[1]
    if seq_len % ncam:
        raise ValueError("sequence length must be divisible by the camera count")

    out_shape = (b, seq_len, nh, d) if out_layout == "NHD" else (b, nh, seq_len, d)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    x64 = x.view(torch.int64)
    y64 = out.view(torch.int64)
    otok, ohead = (1, 2) if out_layout == "NHD" else (2, 1)
    planes = d // 4

    cs = cos_sin if cos_sin is not None else x  # unused when ROPE is 0
    grid = lambda meta: (nh, triton.cdiv(seq_len, meta["BLOCK_T"]), b)
    _transform_kernel[grid](
        x64, y64, cs, matrix,
        x64.stride(0), x64.stride(tok), x64.stride(head),
        y64.stride(0), y64.stride(otok), y64.stride(ohead),
        cs.stride(0) if cos_sin is not None else 0, matrix.stride(0),
        seq_len, seq_len // ncam, ncam,
        PLANES=planes, ROPE=rope_mode if cos_sin is not None else 0,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Host-side preparation
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _prope_matrix_kernel(
    VM, KS, ROT, PT, PINV, PO,
    stride_vb, stride_kb, stride_ob,
    HAS_K: tl.constexpr, HAS_ROT: tl.constexpr,
):
    """One program per camera: build ``(P^T, P^-1, P)`` for the "linear" transform.

    Mirrors ``prepare_prope_matrices`` operation for operation -- the 4x4
    contractions accumulate in ascending ``k`` exactly as the reference's
    ``einsum`` does, and the zero entries of the lifted intrinsics are kept
    rather than folded away -- so the bfloat16-rounded result is bit-identical
    while costing one launch instead of ~30.

    ``HAS_ROT`` left-multiplies the two matrices that meet in a dot product
    (``P^T`` for Q, ``P^-1`` for K) by a 4x4 orthogonal matrix. The rotation
    cancels in ``Q_eff . K_eff`` but spreads channel outliers inside each
    4-channel block, which is worth real accuracy to an int8 attention and
    costs nothing at run time. ``P`` (the output projection) is never rotated.
    """
    c = tl.program_id(0)
    b = tl.program_id(1)
    vb = VM + b * stride_vb + c * 16
    v00 = tl.load(vb + 0); v01 = tl.load(vb + 1); v02 = tl.load(vb + 2); v03 = tl.load(vb + 3)
    v10 = tl.load(vb + 4); v11 = tl.load(vb + 5); v12 = tl.load(vb + 6); v13 = tl.load(vb + 7)
    v20 = tl.load(vb + 8); v21 = tl.load(vb + 9); v22 = tl.load(vb + 10); v23 = tl.load(vb + 11)
    v30 = tl.load(vb + 12); v31 = tl.load(vb + 13); v32 = tl.load(vb + 14); v33 = tl.load(vb + 15)

    z = v00 * 0.0
    one = z + 1.0

    # invert_se3(v): R^T on the 3x3 block, -R^T t in the last column.
    r00 = v00; r01 = v10; r02 = v20
    r10 = v01; r11 = v11; r12 = v21
    r20 = v02; r21 = v12; r22 = v22
    t0 = -(r00 * v03 + r01 * v13 + r02 * v23)
    t1 = -(r10 * v03 + r11 * v13 + r12 * v23)
    t2 = -(r20 * v03 + r21 * v13 + r22 * v23)
    n30 = z; n31 = z; n32 = z; n33 = one
    n00 = r00; n01 = r01; n02 = r02; n03 = t0
    n10 = r10; n11 = r11; n12 = r12; n13 = t1
    n20 = r20; n21 = r21; n22 = r22; n23 = t2

    if HAS_K:
        kb = KS + b * stride_kb + c * 9
        fx = tl.load(kb + 0)
        fy = tl.load(kb + 4)
        # lift_k(Ks_norm): Ks_norm keeps only (fx, fy, 1) on the diagonal.
        p00 = fx * v00 + z * v10 + z * v20 + z * v30
        p01 = fx * v01 + z * v11 + z * v21 + z * v31
        p02 = fx * v02 + z * v12 + z * v22 + z * v32
        p03 = fx * v03 + z * v13 + z * v23 + z * v33
        p10 = z * v00 + fy * v10 + z * v20 + z * v30
        p11 = z * v01 + fy * v11 + z * v21 + z * v31
        p12 = z * v02 + fy * v12 + z * v22 + z * v32
        p13 = z * v03 + fy * v13 + z * v23 + z * v33
        p20 = z * v00 + z * v10 + one * v20 + z * v30
        p21 = z * v01 + z * v11 + one * v21 + z * v31
        p22 = z * v02 + z * v12 + one * v22 + z * v32
        p23 = z * v03 + z * v13 + one * v23 + z * v33
        p30 = z * v00 + z * v10 + z * v20 + one * v30
        p31 = z * v01 + z * v11 + z * v21 + one * v31
        p32 = z * v02 + z * v12 + z * v22 + one * v32
        p33 = z * v03 + z * v13 + z * v23 + one * v33
        # lift_k(invert_k(Ks_norm)): the principal point of Ks_norm is zero,
        # so the (0, 2) and (1, 2) entries are -0 / fx and -0 / fy.
        ikx = one / fx; iky = one / fy; ik02 = -z / fx; ik12 = -z / fy
        q00 = n00 * ikx + n01 * z + n02 * z + n03 * z
        q01 = n00 * z + n01 * iky + n02 * z + n03 * z
        q02 = n00 * ik02 + n01 * ik12 + n02 * one + n03 * z
        q03 = n00 * z + n01 * z + n02 * z + n03 * one
        q10 = n10 * ikx + n11 * z + n12 * z + n13 * z
        q11 = n10 * z + n11 * iky + n12 * z + n13 * z
        q12 = n10 * ik02 + n11 * ik12 + n12 * one + n13 * z
        q13 = n10 * z + n11 * z + n12 * z + n13 * one
        q20 = n20 * ikx + n21 * z + n22 * z + n23 * z
        q21 = n20 * z + n21 * iky + n22 * z + n23 * z
        q22 = n20 * ik02 + n21 * ik12 + n22 * one + n23 * z
        q23 = n20 * z + n21 * z + n22 * z + n23 * one
        q30 = n30 * ikx + n31 * z + n32 * z + n33 * z
        q31 = n30 * z + n31 * iky + n32 * z + n33 * z
        q32 = n30 * ik02 + n31 * ik12 + n32 * one + n33 * z
        q33 = n30 * z + n31 * z + n32 * z + n33 * one
    else:
        p00 = v00; p01 = v01; p02 = v02; p03 = v03
        p10 = v10; p11 = v11; p12 = v12; p13 = v13
        p20 = v20; p21 = v21; p22 = v22; p23 = v23
        p30 = v30; p31 = v31; p32 = v32; p33 = v33
        q00 = n00; q01 = n01; q02 = n02; q03 = n03
        q10 = n10; q11 = n11; q12 = n12; q13 = n13
        q20 = n20; q21 = n21; q22 = n22; q23 = n23
        q30 = n30; q31 = n31; q32 = n32; q33 = n33

    # P^T, before any rotation
    u00 = p00; u01 = p10; u02 = p20; u03 = p30
    u10 = p01; u11 = p11; u12 = p21; u13 = p31
    u20 = p02; u21 = p12; u22 = p22; u23 = p32
    u30 = p03; u31 = p13; u32 = p23; u33 = p33

    if HAS_ROT:
        g00 = tl.load(ROT + 0); g01 = tl.load(ROT + 1); g02 = tl.load(ROT + 2); g03 = tl.load(ROT + 3)
        g10 = tl.load(ROT + 4); g11 = tl.load(ROT + 5); g12 = tl.load(ROT + 6); g13 = tl.load(ROT + 7)
        g20 = tl.load(ROT + 8); g21 = tl.load(ROT + 9); g22 = tl.load(ROT + 10); g23 = tl.load(ROT + 11)
        g30 = tl.load(ROT + 12); g31 = tl.load(ROT + 13); g32 = tl.load(ROT + 14); g33 = tl.load(ROT + 15)
        a00 = g00 * u00 + g01 * u10 + g02 * u20 + g03 * u30
        a01 = g00 * u01 + g01 * u11 + g02 * u21 + g03 * u31
        a02 = g00 * u02 + g01 * u12 + g02 * u22 + g03 * u32
        a03 = g00 * u03 + g01 * u13 + g02 * u23 + g03 * u33
        a10 = g10 * u00 + g11 * u10 + g12 * u20 + g13 * u30
        a11 = g10 * u01 + g11 * u11 + g12 * u21 + g13 * u31
        a12 = g10 * u02 + g11 * u12 + g12 * u22 + g13 * u32
        a13 = g10 * u03 + g11 * u13 + g12 * u23 + g13 * u33
        a20 = g20 * u00 + g21 * u10 + g22 * u20 + g23 * u30
        a21 = g20 * u01 + g21 * u11 + g22 * u21 + g23 * u31
        a22 = g20 * u02 + g21 * u12 + g22 * u22 + g23 * u32
        a23 = g20 * u03 + g21 * u13 + g22 * u23 + g23 * u33
        a30 = g30 * u00 + g31 * u10 + g32 * u20 + g33 * u30
        a31 = g30 * u01 + g31 * u11 + g32 * u21 + g33 * u31
        a32 = g30 * u02 + g31 * u12 + g32 * u22 + g33 * u32
        a33 = g30 * u03 + g31 * u13 + g32 * u23 + g33 * u33
        c00 = g00 * q00 + g01 * q10 + g02 * q20 + g03 * q30
        c01 = g00 * q01 + g01 * q11 + g02 * q21 + g03 * q31
        c02 = g00 * q02 + g01 * q12 + g02 * q22 + g03 * q32
        c03 = g00 * q03 + g01 * q13 + g02 * q23 + g03 * q33
        c10 = g10 * q00 + g11 * q10 + g12 * q20 + g13 * q30
        c11 = g10 * q01 + g11 * q11 + g12 * q21 + g13 * q31
        c12 = g10 * q02 + g11 * q12 + g12 * q22 + g13 * q32
        c13 = g10 * q03 + g11 * q13 + g12 * q23 + g13 * q33
        c20 = g20 * q00 + g21 * q10 + g22 * q20 + g23 * q30
        c21 = g20 * q01 + g21 * q11 + g22 * q21 + g23 * q31
        c22 = g20 * q02 + g21 * q12 + g22 * q22 + g23 * q32
        c23 = g20 * q03 + g21 * q13 + g22 * q23 + g23 * q33
        c30 = g30 * q00 + g31 * q10 + g32 * q20 + g33 * q30
        c31 = g30 * q01 + g31 * q11 + g32 * q21 + g33 * q31
        c32 = g30 * q02 + g31 * q12 + g32 * q22 + g33 * q32
        c33 = g30 * q03 + g31 * q13 + g32 * q23 + g33 * q33
    else:
        a00 = u00; a01 = u01; a02 = u02; a03 = u03
        a10 = u10; a11 = u11; a12 = u12; a13 = u13
        a20 = u20; a21 = u21; a22 = u22; a23 = u23
        a30 = u30; a31 = u31; a32 = u32; a33 = u33
        c00 = q00; c01 = q01; c02 = q02; c03 = q03
        c10 = q10; c11 = q11; c12 = q12; c13 = q13
        c20 = q20; c21 = q21; c22 = q22; c23 = q23
        c30 = q30; c31 = q31; c32 = q32; c33 = q33

    out = b * stride_ob + c * 16
    # apply_prope casts the matrix to the feature dtype before multiplying.
    tl.store(PO + out + 0, p00.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 0, a00.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 0, c00.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 1, p01.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 1, a01.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 1, c01.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 2, p02.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 2, a02.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 2, c02.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 3, p03.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 3, a03.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 3, c03.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 4, p10.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 4, a10.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 4, c10.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 5, p11.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 5, a11.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 5, c11.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 6, p12.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 6, a12.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 6, c12.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 7, p13.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 7, a13.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 7, c13.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 8, p20.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 8, a20.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 8, c20.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 9, p21.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 9, a21.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 9, c21.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 10, p22.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 10, a22.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 10, c22.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 11, p23.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 11, a23.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 11, c23.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 12, p30.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 12, a30.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 12, c30.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 13, p31.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 13, a31.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 13, c31.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 14, p32.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 14, a32.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 14, c32.to(tl.bfloat16).to(tl.float32))
    tl.store(PO + out + 15, p33.to(tl.bfloat16).to(tl.float32))
    tl.store(PT + out + 15, a33.to(tl.bfloat16).to(tl.float32))
    tl.store(PINV + out + 15, c33.to(tl.bfloat16).to(tl.float32))


def _prope_tables_triton(
    viewmats: torch.Tensor,
    Ks: Optional[torch.Tensor],
    rotation: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-launch ``(P^T, P^-1, P)`` as ``[batch, cams, 16]`` float32.

    ``rotation`` is an optional 4x4 orthogonal matrix left-multiplied into
    ``P^T`` and ``P^-1`` only; see ``_prope_matrix_kernel``.
    """
    batch, ncam = viewmats.shape[:2]
    vm = viewmats.to(torch.float32).contiguous()
    ks = None if Ks is None else Ks.to(torch.float32).contiguous()
    rot = None if rotation is None else rotation.reshape(16).contiguous().to(torch.float32)
    shape = (batch, ncam, 16)
    p_t = torch.empty(shape, dtype=torch.float32, device=viewmats.device)
    p_inv = torch.empty(shape, dtype=torch.float32, device=viewmats.device)
    p = torch.empty(shape, dtype=torch.float32, device=viewmats.device)
    _prope_matrix_kernel[(ncam, batch)](
        vm, ks if ks is not None else vm, rot if rot is not None else vm,
        p_t, p_inv, p,
        vm.stride(0), 0 if ks is None else ks.stride(0), p_t.stride(0),
        HAS_K=ks is not None, HAS_ROT=rot is not None, num_warps=1,
    )
    return p_t, p_inv, p


def _flat16(m: torch.Tensor) -> torch.Tensor:
    """``[B, C, 4, 4]`` -> ``[B, C, 16]`` float32, rounded through bfloat16.

    ``apply_prope`` casts the matrix to the feature dtype before multiplying,
    so carrying that rounding here is what makes the kernels track the
    reference bit pattern.
    """
    return (m.to(torch.bfloat16).to(torch.float32)
             .reshape(m.shape[0], m.shape[1], 16).contiguous())


def _prope_tables(
    viewmats: torch.Tensor,
    Ks: Optional[torch.Tensor],
    camera_translation_transform: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-camera ``(P^T, P^-1, P)`` flattened to ``[batch, cams, 16]`` float32."""
    p_t, p_inv, p = prepare_prope_matrices(
        viewmats, Ks=Ks, target_seqlen=None,
        camera_translation_transform=camera_translation_transform,
    )
    return _flat16(p_t), _flat16(p_inv), _flat16(p)


def _prope_tables_batched(
    q_viewmats: torch.Tensor,
    q_Ks: Optional[torch.Tensor],
    kv_viewmats: torch.Tensor,
    kv_Ks: Optional[torch.Tensor],
    camera_translation_transform: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(P_q^T, P_kv^-1, P_q)`` with the two camera sets prepared in one pass.

    ``prepare_prope_matrices`` is ~30 tiny launches; running the query and
    key/value cameras as a single concatenated batch halves the launch count
    without changing a single floating-point operation.
    """
    n_q = q_viewmats.shape[1]
    same = kv_viewmats is q_viewmats and kv_Ks is q_Ks
    if (camera_translation_transform in (None, "linear")
            and q_viewmats.device.type == "xpu"
            and (q_Ks is None) == (kv_Ks is None)):
        if same:
            p_t, p_inv, p = _prope_tables_triton(q_viewmats, q_Ks)
            return p_t, p_inv, p
        p_t, _, p = _prope_tables_triton(q_viewmats, q_Ks)
        _, p_inv, _ = _prope_tables_triton(kv_viewmats, kv_Ks)
        return p_t, p_inv, p

    if same:
        viewmats, Ks = q_viewmats, q_Ks
    elif (q_Ks is None) != (kv_Ks is None):
        # Mixed intrinsics: fall back to two independent preparations.
        p_t, _, p = _prope_tables(q_viewmats, q_Ks, camera_translation_transform)
        _, p_inv, _ = _prope_tables(kv_viewmats, kv_Ks, camera_translation_transform)
        return p_t, p_inv, p
    else:
        viewmats = torch.cat([q_viewmats, kv_viewmats], dim=1)
        Ks = None if q_Ks is None else torch.cat([q_Ks, kv_Ks], dim=1)

    p_t, p_inv, p = prepare_prope_matrices(
        viewmats, Ks=Ks, target_seqlen=None,
        camera_translation_transform=camera_translation_transform,
    )
    if same:
        return _flat16(p_t), _flat16(p_inv), _flat16(p)
    return _flat16(p_t[:, :n_q]), _flat16(p_inv[:, n_q:]), _flat16(p[:, :n_q])


def _prepare(q, k, freqs, q_grid_sizes, k_grid_sizes, q_viewmats, q_Ks,
             kv_viewmats, kv_Ks, q_start_frame, k_start_frame, rope_dtype,
             camera_translation_transform):
    fr = _freqs_real(freqs, rope_dtype)
    cs_q = build_rope_table(fr, q_grid_sizes, q.shape[1], q_start_frame)
    cs_k = build_rope_table(fr, k_grid_sizes, k.shape[1], k_start_frame)
    if kv_viewmats is None:
        kv_viewmats, kv_Ks = q_viewmats, q_Ks
    p_q_t, p_kv_inv, p_q = _prope_tables_batched(
        q_viewmats, q_Ks, kv_viewmats, kv_Ks, camera_translation_transform)
    return cs_q, cs_k, p_q_t, p_kv_inv, p_q


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────

def fused_rope_prope_sdpa_split(
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
    scale: Optional[float] = None,
) -> torch.Tensor:
    """RoPE+PRoPE fused into the Q/K/V stream, SDPA, then fused output PRoPE."""
    cs_q, cs_k, p_q_t, p_kv_inv, p_q = _prepare(
        q, k, freqs, q_grid_sizes, k_grid_sizes, q_viewmats, q_Ks,
        kv_viewmats, kv_Ks, q_start_frame, k_start_frame, rope_dtype,
        camera_translation_transform)

    # The transforms emit contiguous [B, H, L, D] directly: SDPA is measurably
    # faster on that layout than on a transposed view of [B, L, H, D], and the
    # epilogue reads it back while writing the caller's [B, L, H, D].
    mode = ROPE_MODE[normalize_rope_precision(rope_dtype)[0]]
    q_eff = rope_prope_transform(q, p_q_t, cs_q, mode, out_layout="HND")
    k_eff = rope_prope_transform(k, p_kv_inv, cs_k, mode, out_layout="HND")
    v_eff = rope_prope_transform(v, p_kv_inv, None, out_layout="HND")

    o = F.scaled_dot_product_attention(q_eff, k_eff, v_eff, scale=scale, is_causal=False)

    return rope_prope_transform(o, p_q, None, in_layout="HND", out_layout="NHD")


fused_rope_prope_sdpa = fused_rope_prope_sdpa_split
