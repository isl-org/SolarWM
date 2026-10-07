"""Fused RoPE + PRoPE + SageAttention (int8) + output PRoPE.

Same fusion boundary as :func:`.fused_sdpa.fused_rope_prope_sdpa_split`, with
the bfloat16 FlashAttention from ``F.scaled_dot_product_attention`` replaced by
an int8 SageAttention core. Tuned for Intel Arc B70 (Battlemage, Xe2, 32 Xe
cores), Triton 3.8.

Five launches per call, against nine for the reference kernel:

1. ``_sage_tables_kernel``    -- every camera matrix the call needs: rotated
                                 ``P_q^T`` and ``P_kv^-1`` for Q and K, plain
                                 ``P_kv^-1`` for V, plain ``P`` for the output.
2. ``_sage_prologue_kernel``  -- RoPE + PRoPE of Q, and in the same launch the
                                 per-head centre of K (sampled, see below).
3. ``_sage_kv_kernel``        -- per 32-token block: RoPE + PRoPE of K, centre,
                                 int8 with one scale per block; PRoPE of V.
                                 The bfloat16 ``K_eff`` never touches memory.
4. ``_sage_attn_kernel``      -- online-softmax attention, int8 QK dot and
                                 bfloat16 PV dot.
5. ``_transform_kernel``      -- the output projection, shared with the SDPA
                                 path.

The camera matrices come from the reference's own matrix kernel, so V_eff and
the output projection are bit-identical to the SDPA path. Q and K are not: they
are quantised to int8 next, so their rotation runs on fp16 ``(cos, sin)`` with
fp32 FMAs for every ``rope_dtype`` and skips the reference's bfloat16 stores
(after RoPE, and of K_eff before int8). That changes Q_eff by bfloat16-sized
rounding, ~2.5e-3 relative, and the final error not at all (1.509% vs 1.504%
against float64), while the float64 and float16 modes stop paying for fp64
arithmetic and emulated fp16 rounding (-230 / -110 us per call).

What makes this fast on B70, in order of how much it matters:

* **Tensor descriptors, not block pointers.** With Triton 3.8 on this part
  ``tl.make_block_ptr`` no longer lowers to 2D block IO in the attention: the
  loads become tensor-of-pointer gathers in a blocked layout, Q is round-tripped
  through shared memory every iteration, and the kernel spills tens of KB per
  thread (17 ms for the attention alone). ``tl.make_tensor_descriptor`` restores
  the block loads: 0.97 ms, against 1.33 ms for the vendor bfloat16 SDPA.
* **Eight query rows per warp.** DPAS holds the whole K and V tile in every
  warp's registers, so 16 rows per warp spills on this part at any ``BLOCK_N``,
  with or without splitting the PV dot. ``BLOCK_M == 8 * num_warps`` always.
* **A real ``int8 x int8 -> int32`` dot**, with K kept ``[L, D]`` and fed
  through ``tl.trans`` -- a pre-transposed ``[D, L]`` K loses the fast
  transposed block load and is 4-50x slower.
* **Few launches.** At the smaller test shapes the GPU work is ~0.1 ms and the
  call is bound by host-side launch overhead, so the pipeline is cut from 13
  launches (two RoPE tables, three camera-matrix builds, a torch reduction, ...)
  to five, and the attention config is picked by a heuristic instead of the
  autotuner.
* **A 4x4 orthogonal rotation folded into the camera matrices.** PRoPE already
  multiplies every contiguous 4-channel block by a 4x4 matrix, and
  ``(H P_q^T q) . (H P_kv^-1 k) == (P_q^T q) . (P_kv^-1 k)`` for orthogonal
  ``H``. Rotating those two matrices therefore leaves the attention unchanged
  while spreading channel outliers inside each block before quantisation:
  free, and 1.17x lower error. ``P`` (the output projection) and the V-side
  ``P_kv^-1`` are left alone -- the rotation has nothing to cancel against
  there.

``FUSION_NOTES.md`` records the variants that were measured and rejected.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from .fused_sdpa import (
    _freqs_real,
    _pack4,
    _prope_matrix_kernel,
    _prope_tables_batched,
    _transform_kernel,
    _unpack4,
    prope4,
)

#: K is quantised in blocks of this many tokens, which is also the attention's
#: inner tile width, so one scale is loaded per iteration.
SAGE_BLOCK_N = 32

#: The K centre is the mean of one ``_MEAN_BLOCK_T``-token tile in every
#: ``_MEAN_STRIDE`` tokens (~1/8 of the window; windows under 2048 tokens are
#: averaged in full). Any vector that is constant
#: across keys is an exact centre -- softmax ignores the shift -- so the mean
#: only has to be a good estimate of where K sits; the RMS error is unchanged to
#: +-0.002% against the full mean on the test fixtures.
_MEAN_STRIDE = 512
_MEAN_BLOCK_T = 64
_MEAN_SPLITS_MAX = 8

#: Token tile of the Q / V / output transforms. Bandwidth bound, flat from 16
#: to 64 on this part.
_TRANSFORM_BLOCK_T = 32

_PROLOGUE_BLOCK_T = 16
_PROLOGUE_WARPS = 8
_TABLE_BLOCK_T = 8
_TABLE_WARPS = 8


@triton.jit
def _quantise_i8(x, scale):
    """Round-half-away-from-zero int8 quantisation of ``x / scale``, as int32."""
    xv = x / scale
    return tl.clamp(xv + 0.5 * tl.where(xv >= 0.0, 1.0, -1.0), -128.0, 127.0).to(tl.int32)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Camera matrices and RoPE tables, one launch
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _rope_cs(FR, GRID, rows, p, seq_len, start_frame, max_rows,
             C: tl.constexpr, SPLIT_T: tl.constexpr, SPLIT_H: tl.constexpr):
    """``(cos, sin)`` of RoPE pairs ``2p`` and ``2p + 1`` for each of ``rows``.

    The same gather, indices and masking as ``fused_sdpa._rope_table_kernel``,
    so the table holds the same values. Gathered once per token here rather
    than inside the transforms: there it would be repeated for every head, and
    on this part that costs more than the transform itself.
    """
    f = tl.load(GRID + 0)
    h_size = tl.load(GRID + 1)
    w = tl.load(GRID + 2)
    hw = h_size * w
    live = (rows < tl.minimum(seq_len, f * hw))[:, None]
    fi = tl.minimum(rows // hw + start_frame, max_rows - 1)[:, None]
    hi = ((rows // w) % h_size)[:, None]
    wi = (rows % w)[:, None]

    c = 2 * p[None, :]
    row = tl.where(c < SPLIT_T, fi, tl.where(c < SPLIT_T + SPLIT_H, hi, wi))
    idx = (row * C + c) * 2
    c0 = tl.where(live, tl.load(FR + idx, mask=live, other=1.0), 1.0)
    s0 = tl.where(live, tl.load(FR + idx + 1, mask=live, other=0.0), 0.0)
    c = c + 1
    row = tl.where(c < SPLIT_T, fi, tl.where(c < SPLIT_T + SPLIT_H, hi, wi))
    idx = (row * C + c) * 2
    c1 = tl.where(live, tl.load(FR + idx, mask=live, other=1.0), 1.0)
    s1 = tl.where(live, tl.load(FR + idx + 1, mask=live, other=0.0), 0.0)
    return c0, s0, c1, s1


@triton.jit
def _sage_tables_kernel(
    VMQ, KSQ, VMK, KSK, ROT,
    PTQ, POQ, PIK, PIKR, SCRQ, SCRK,
    FR, GQ, GK, CSQ, CSK,
    nq, nkv, n_qb, seq_q, seq_k, start_q, start_k, max_rows,
    stride_vq, stride_kq, stride_vk, stride_kk, stride_oq, stride_ok, stride_gb,
    HAS_K: tl.constexpr, ROTATE: tl.constexpr,
    PLANES: tl.constexpr, C: tl.constexpr, SPLIT_T: tl.constexpr, SPLIT_H: tl.constexpr,
    TB: tl.constexpr,
):
    """Every table the call needs, in one launch.

    Programs ``[0, nq)`` build the query cameras and ``[nq, nq + nkv)`` the
    key/value ones. Each of those is the reference's ``_prope_matrix_kernel``
    itself -- inlined, with its pointers shifted so that its ``program_id``
    lands on the right camera -- so the matrices are bit-identical to
    ``_prope_tables_triton``; outputs a branch does not need go to scratch.
    The remaining programs expand the RoPE ``(cos, sin)`` tables of Q, then K,
    ``TB`` tokens each, in ``build_rope_table``'s plane-major layout.
    """
    c = tl.program_id(0)
    b = tl.program_id(1)
    if c < nq:
        # P^T (rotated if ROTATE) for Q and the plain P for the output
        _prope_matrix_kernel(VMQ, KSQ, ROT, PTQ, SCRQ, POQ,
                             stride_vq, stride_kq, stride_oq, HAS_K, ROTATE)
    elif c < nq + nkv:
        shift = nq * 16
        vm = VMK - shift
        ks = KSK - nq * 9
        scr = SCRK - shift
        # plain P^-1 for V
        _prope_matrix_kernel(vm, ks, ROT, scr, PIK - shift, scr,
                             stride_vk, stride_kk, stride_ok, HAS_K, False)
        if ROTATE:
            # rotated P^-1 for K
            _prope_matrix_kernel(vm, ks, ROT, scr, PIKR - shift, scr,
                                 stride_vk, stride_kk, stride_ok, HAS_K, True)
    else:
        t = c - nq - nkv
        if t < n_qb:
            CS = CSQ
            grid = GQ
            seq_len = seq_q
            start_frame = start_q
        else:
            t -= n_qb
            CS = CSK
            grid = GK
            seq_len = seq_k
            start_frame = start_k
        start = t * TB
        c0, s0, c1, s1 = _rope_cs(FR, grid + b * stride_gb, start + tl.arange(0, TB),
                                  tl.arange(0, PLANES), seq_len, start_frame, max_rows,
                                  C, SPLIT_T, SPLIT_H)
        csd = tl.make_tensor_descriptor(CS + b * seq_len * 4 * PLANES, (seq_len, 4 * PLANES),
                                        (4 * PLANES, 1), (TB, PLANES))
        csd.store([start, 0], c0.to(tl.float16))
        csd.store([start, PLANES], s0.to(tl.float16))
        csd.store([start, 2 * PLANES], c1.to(tl.float16))
        csd.store([start, 3 * PLANES], s1.to(tl.float16))


def _build_tables(fr, q_grid_sizes, k_grid_sizes, seq_q, seq_k, q_start_frame,
                  k_start_frame, q_viewmats, q_Ks, kv_viewmats, kv_Ks, rot, cameras):
    """``(cs_q, cs_k, P_q^T, P_kv^-1 for K, P_kv^-1 for V, P_q)``.

    ``cameras=False`` builds only the RoPE tables, for camera transforms the
    matrix kernel does not implement; the caller supplies the matrices.
    """
    batch, nq = q_viewmats.shape[:2]
    nkv = kv_viewmats.shape[1]
    dev = fr.device
    c = fr.shape[1]
    planes = c // 2
    # fp16 (cos, sin) for the Q/K rotation -- see _transform_tile
    cs_q = torch.empty((batch, seq_q, 2 * c), dtype=torch.float16, device=dev)
    cs_k = torch.empty((batch, seq_k, 2 * c), dtype=torch.float16, device=dev)
    grid_q = q_grid_sizes.to(device=dev).contiguous()
    grid_k = k_grid_sizes.to(device=dev).contiguous()
    tb = _TABLE_BLOCK_T
    n_qb, n_kb = triton.cdiv(seq_q, tb), triton.cdiv(seq_k, tb)

    vmq = q_viewmats.to(torch.float32).contiguous()
    vmk = kv_viewmats.to(torch.float32).contiguous()
    has_k = q_Ks is not None and cameras
    ksq = q_Ks.to(torch.float32).contiguous() if has_k else vmq
    ksk = kv_Ks.to(torch.float32).contiguous() if has_k else vmk
    rot_f32 = rot.to(torch.float32) if rot is not None else None
    f32 = dict(dtype=torch.float32, device=dev)
    q_tabs = torch.empty((3, batch, nq, 16), **f32)       # P^T, P, scratch
    k_tabs = torch.empty((3, batch, nkv, 16), **f32)      # P^-1, rotated P^-1, scratch
    p_kv_k = k_tabs[1] if rot is not None else k_tabs[0]
    ncam = (nq, nkv) if cameras else (0, 0)
    _sage_tables_kernel[(ncam[0] + ncam[1] + n_qb + n_kb, batch)](
        vmq, ksq, vmk, ksk, rot_f32 if rot_f32 is not None else vmq,
        q_tabs[0], q_tabs[1], k_tabs[0], p_kv_k, q_tabs[2], k_tabs[2],
        fr, grid_q, grid_k, cs_q, cs_k,
        ncam[0], ncam[1], n_qb, seq_q, seq_k, q_start_frame, k_start_frame, fr.shape[0],
        vmq.stride(0), ksq.stride(0) if has_k else 0,
        vmk.stride(0), ksk.stride(0) if has_k else 0,
        nq * 16, nkv * 16, grid_q.stride(0),
        HAS_K=has_k, ROTATE=rot is not None,
        PLANES=planes, C=c, SPLIT_T=c - 2 * (c // 3), SPLIT_H=c // 3, TB=tb, num_warps=_TABLE_WARPS,
    )
    return cs_q, cs_k, q_tabs[0], p_kv_k, k_tabs[0], q_tabs[1]


# ─────────────────────────────────────────────────────────────────────────────
# Shared transform pieces
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _transform_tile(xd, csd, start, col, P, tpc, ncam,
                    PLANES: tl.constexpr, ROPE: tl.constexpr):
    """``PRoPE(RoPE(x))`` of the int64 tile of ``xd``'s block shape at ``(start, col)``.

    Four fp32 planes, plane ``j`` holding channels ``4 * (col + p) + j``; the
    ``(cos, sin)`` tile comes from the same columns of each plane of ``csd``.

    ``ROPE`` is a flag here, not the reference's precision mode: the rotation
    feeds int8 quantisation, so it runs on fp16 ``(cos, sin)`` with fp32 FMAs
    whatever ``rope_dtype`` is. A bfloat16 input is exact in fp32 and an fp16
    factor is too, so the products are exact and the only new rounding is the
    fp16 ``(cos, sin)`` (~2^-11), 8x below an int8 step; the reference's
    bfloat16 store between RoPE and PRoPE is skipped as a pointless double
    rounding. PRoPE is ``prope4`` itself.
    """
    y0, y1, y2, y3 = _unpack4(xd.load([start, col]))
    if ROPE:
        c0 = csd.load([start, col]).to(tl.float32)
        s0 = csd.load([start, PLANES + col]).to(tl.float32)
        c1 = csd.load([start, 2 * PLANES + col]).to(tl.float32)
        s1 = csd.load([start, 3 * PLANES + col]).to(tl.float32)
        y0, y1 = tl.math.fma(y0, c0, -(y1 * s0)), tl.math.fma(y0, s0, y1 * c0)
        y2, y3 = tl.math.fma(y2, c1, -(y3 * s1)), tl.math.fma(y2, s1, y3 * c1)
    rows = start + tl.arange(0, xd.block_shape[0])
    cam = tl.minimum(rows // tpc, ncam - 1)
    return prope4(y0, y1, y2, y3, P, cam)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Q transform + K centre, one launch
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _sage_prologue_kernel(
    Q, QE, K, KM, CSQ, CSK, PQ, PK,
    seq_q, seq_k, tpc_q, ncam_q, tpc_k, ncam_k, inv_count,
    H: tl.constexpr, PLANES: tl.constexpr, ROPE: tl.constexpr,
    BLOCK_T: tl.constexpr, MEAN_SPLITS: tl.constexpr,
    MEAN_STRIDE: tl.constexpr, MEAN_BT: tl.constexpr,
):
    """The first ``H * MEAN_SPLITS`` programs centre K, the rest transform Q.

    The centre programs go first so their longer serial loop starts at once and
    hides behind the Q tiles. Each owns ``PLANES / MEAN_SPLITS`` of a head's
    4-channel PRoPE blocks over the sampled tokens; the blocks are independent,
    so each program's mean is final and needs no cross-program reduction. The
    mean is over ``K_eff`` rounded to bfloat16, i.e. over the values the
    quantiser sees, and is stored plane-major (``[plane j | block p]`` holds
    channel ``4p + j``).
    """
    pid = tl.program_id(0)
    b = tl.program_id(1)
    if pid < H * MEAN_SPLITS:
        h = pid // MEAN_SPLITS
        NP: tl.constexpr = PLANES // MEAN_SPLITS
        col = (pid % MEAN_SPLITS) * NP
        mkd = tl.make_tensor_descriptor(K + (b * seq_k * H + h) * PLANES, (seq_k, PLANES),
                                       (H * PLANES, 1), (MEAN_BT, NP))
        mcsd = tl.make_tensor_descriptor(CSK + b * seq_k * 4 * PLANES, (seq_k, 4 * PLANES),
                                        (4 * PLANES, 1), (MEAN_BT, NP))
        # accumulate tile-shaped and reduce once at the end: a cross-warp
        # reduction per sampled tile made this loop barrier-bound (46 -> 22 us)
        t0 = tl.zeros((MEAN_BT, NP), tl.float32)
        t1 = tl.zeros((MEAN_BT, NP), tl.float32)
        t2 = tl.zeros((MEAN_BT, NP), tl.float32)
        t3 = tl.zeros((MEAN_BT, NP), tl.float32)
        for start in tl.range(0, seq_k, MEAN_STRIDE, loop_unroll_factor=2):
            # padding rows are zero and transform to zero, so they add nothing
            z0, z1, z2, z3 = _transform_tile(mkd, mcsd, start, col, PK + b * ncam_k * 16,
                                             tpc_k, ncam_k, PLANES, ROPE)
            t0 += z0
            t1 += z1
            t2 += z2
            t3 += z3
        a0 = tl.sum(t0, 0)
        a1 = tl.sum(t1, 0)
        a2 = tl.sum(t2, 0)
        a3 = tl.sum(t3, 0)
        out = KM + (b * H + h) * 4 * PLANES + col + tl.arange(0, NP)
        tl.store(out, a0 * inv_count)
        tl.store(out + PLANES, a1 * inv_count)
        tl.store(out + 2 * PLANES, a2 * inv_count)
        tl.store(out + 3 * PLANES, a3 * inv_count)
    else:
        t = pid - H * MEAN_SPLITS
        h = t % H                   # fastest: neighbours share the RoPE tile
        start = (t // H) * BLOCK_T
        xd = tl.make_tensor_descriptor(Q + (b * seq_q * H + h) * PLANES, (seq_q, PLANES),
                                       (H * PLANES, 1), (BLOCK_T, PLANES))
        csd = tl.make_tensor_descriptor(CSQ + b * seq_q * 4 * PLANES, (seq_q, 4 * PLANES),
                                        (4 * PLANES, 1), (BLOCK_T, PLANES))
        z0, z1, z2, z3 = _transform_tile(xd, csd, start, 0, PQ + b * ncam_q * 16,
                                         tpc_q, ncam_q, PLANES, ROPE)
        yd = tl.make_tensor_descriptor(QE + (b * H + h) * seq_q * PLANES, (seq_q, PLANES),
                                       (PLANES, 1), (BLOCK_T, PLANES))
        yd.store([start, 0], _pack4(z0, z1, z2, z3))


# ─────────────────────────────────────────────────────────────────────────────
# 3. K quantisation + V transform, one launch
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _sage_kv_kernel(
    K, V, KI, KS, VE, KM, CSK, PKR, PK,
    seq_k, tpc, ncam,
    H: tl.constexpr, PLANES: tl.constexpr, ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One ``BLOCK_N``-token block of one head: int8 K plus its scale, or V_eff.

    Even programs do K, odd ones the same block of V: half the live state per
    program, ~3% faster than doing both in one.

    Subtracting a vector that is constant across keys shifts every logit in a
    row by the same amount, which softmax ignores -- so the centring is exact
    and costs the attention nothing, while shrinking what the int8 range has to
    cover. The four int8 channels of a PRoPE block are packed into one int32,
    the mirror image of the int64 trick on the load side.
    """
    t = tl.program_id(0)
    b = tl.program_id(1)
    is_v = t % 2
    t = t // 2
    h = t % H                       # fastest: neighbours share the RoPE tile
    n = t // H
    start = n * BLOCK_N
    src = (b * seq_k * H + h) * PLANES
    dst = (b * H + h) * seq_k * PLANES
    csd = tl.make_tensor_descriptor(CSK + b * seq_k * 4 * PLANES, (seq_k, 4 * PLANES),
                                    (4 * PLANES, 1), (BLOCK_N, PLANES))

    if is_v:
        vd = tl.make_tensor_descriptor(V + src, (seq_k, PLANES), (H * PLANES, 1),
                                       (BLOCK_N, PLANES))
        w0, w1, w2, w3 = _transform_tile(vd, csd, start, 0, PK + b * ncam * 16, tpc, ncam,
                                         PLANES, False)
        ved = tl.make_tensor_descriptor(VE + dst, (seq_k, PLANES), (PLANES, 1),
                                        (BLOCK_N, PLANES))
        ved.store([start, 0], _pack4(w0, w1, w2, w3))
        return

    kd = tl.make_tensor_descriptor(K + src, (seq_k, PLANES), (H * PLANES, 1), (BLOCK_N, PLANES))
    z0, z1, z2, z3 = _transform_tile(kd, csd, start, 0, PKR + b * ncam * 16, tpc, ncam,
                                     PLANES, ROPE)

    mb = KM + (b * H + h) * 4 * PLANES + tl.arange(0, PLANES)
    # centre; padding rows would become -mean, so zero them to keep them out
    # of the scale. K_eff is not rounded to bfloat16 first: int8 follows, and
    # the extra rounding only adds error
    live = ((start + tl.arange(0, BLOCK_N)) < seq_k)[:, None]
    z0 = tl.where(live, z0 - tl.load(mb)[None, :], 0.0)
    z1 = tl.where(live, z1 - tl.load(mb + PLANES)[None, :], 0.0)
    z2 = tl.where(live, z2 - tl.load(mb + 2 * PLANES)[None, :], 0.0)
    z3 = tl.where(live, z3 - tl.load(mb + 3 * PLANES)[None, :], 0.0)
    # one reduction over the elementwise max of the planes, not four
    amax = tl.max(tl.maximum(tl.maximum(tl.abs(z0), tl.abs(z1)),
                             tl.maximum(tl.abs(z2), tl.abs(z3))))
    scale = tl.maximum(amax * (1.0 / 127.0), 1.0e-8)
    tl.store(KS + (b * H + h) * tl.cdiv(seq_k, BLOCK_N) + n, scale)
    packed = ((_quantise_i8(z0, scale) & 0xFF)
              | ((_quantise_i8(z1, scale) & 0xFF) << 8)
              | ((_quantise_i8(z2, scale) & 0xFF) << 16)
              | (_quantise_i8(z3, scale) << 24))
    kid = tl.make_tensor_descriptor(KI + dst, (seq_k, PLANES), (PLANES, 1), (BLOCK_N, PLANES))
    kid.store([start, 0], packed)


# ─────────────────────────────────────────────────────────────────────────────
# 4. int8 attention
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _sage_attn_kernel(
    Q, KI8, V, KS, O,
    stride_qb, stride_qh, stride_qn,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_sb, stride_sh,
    stride_ob, stride_oh, stride_on,
    qo_len, kv_len, sm_scale,
    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    start_m = pid_m * BLOCK_M

    qd = tl.make_tensor_descriptor(Q + b * stride_qb + h * stride_qh, (qo_len, D),
                                   (stride_qn, 1), (BLOCK_M, D))
    kd = tl.make_tensor_descriptor(KI8 + b * stride_kb + h * stride_kh, (kv_len, D),
                                   (stride_kn, 1), (BLOCK_N, D))
    vd = tl.make_tensor_descriptor(V + b * stride_vb + h * stride_vh, (kv_len, D),
                                   (stride_vn, 1), (BLOCK_N, D))
    od = tl.make_tensor_descriptor(O + b * stride_ob + h * stride_oh, (qo_len, D),
                                   (stride_on, 1), (BLOCK_M, D))

    q = qd.load([start_m, 0]).to(tl.float32)
    q_scale = tl.maximum(tl.max(tl.abs(q)) * (1.0 / 127.0), 1.0e-8)
    q_i8 = _quantise_i8(q, q_scale).to(tl.int8)

    acc = tl.zeros((BLOCK_M, D), tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    qk_scale = q_scale * sm_scale * 1.44269504
    offs_n = tl.arange(0, BLOCK_N)
    scales = KS + b * stride_sb + h * stride_sh

    for start_n in tl.range(0, kv_len, BLOCK_N):
        k_scale = tl.load(scales + start_n // BLOCK_N)
        # int8 x int8 -> int32 DPAS: twice the bfloat16 K per instruction.
        # tl.trans of the [BLOCK_N, D] tile folds into a transposed block load;
        # both scales are scalars -- a per-row Q scale is 1.17x more accurate
        # but makes IGC spill, doubling the time.
        s = tl.dot(q_i8, tl.trans(kd.load([start_n, 0])), out_dtype=tl.int32).to(tl.float32)
        s = s * (qk_scale * k_scale)
        s = tl.where((start_n + offs_n)[None, :] < kv_len, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), vd.load([start_n, 0]), acc)
        m_i = m_new

    od.store([start_m, 0], (acc / l_i[:, None]).to(O.dtype.element_ty))


def _attn_config(qo_len: int, heads: int, batch: int) -> Tuple[int, int]:
    """``(BLOCK_M, num_warps)`` for the attention, ``BLOCK_M == 8 * num_warps``.

    A heuristic rather than ``triton.autotune``: the autotuner's per-call
    dispatch is a measurable share of the small shapes' host-bound runtime.
    Bigger tiles reuse each K/V tile across more rows; smaller ones only pay
    when the grid would otherwise leave B70's 32 Xe cores short of work.
    Measured on the production chunk: 128/16 and 256/32 at 0.97-1.0 ms (128
    keeps the per-tile Q scale finer), 64/8 1.05 ms, 32/4 1.16 ms.
    """
    for block_m, warps, min_programs in ((128, 16, 128), (64, 8, 96)):
        if triton.cdiv(qo_len, block_m) * heads * batch >= min_programs:
            return block_m, warps
    return 32, 4


def _attend_int8(
    q: torch.Tensor,
    k_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    block_n: int = SAGE_BLOCK_N,
) -> torch.Tensor:
    """The attention core on pre-quantised K, all ``[B, H, L, D]``."""
    batch, heads, qo_len, d = q.shape
    kv_len = k_int8.shape[2]
    block_m, warps = _attn_config(qo_len, heads, batch)
    out = torch.empty_like(q)
    _sage_attn_kernel[(triton.cdiv(qo_len, block_m), heads, batch)](
        q, k_int8, v, k_scale, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        qo_len, kv_len, d ** -0.5, D=d, BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=warps, num_stages=2, grf_mode="256",
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Standalone int8 attention on an already-transformed bfloat16 K
# (and the two-pass quantiser the tests compare the fused one against)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _k_mean_partial_kernel(
    K, OUT, seq_len,
    stride_kb, stride_kh, stride_kn,
    stride_ob, stride_oh, stride_os,
    D: tl.constexpr, SPLITS: tl.constexpr, BLOCK_L: tl.constexpr,
):
    """Partial column sums of ``K[b, h]`` over the KV window.

    ``k.mean(dim=2)`` in torch reduces over a strided middle axis and is several
    times slower; this streams the same bytes with 2D block loads and splits the
    window so the grid is wide enough to fill the device.
    """
    h = tl.program_id(0)
    s = tl.program_id(1)
    b = tl.program_id(2)

    chunk = tl.cdiv(tl.cdiv(seq_len, BLOCK_L), SPLITS) * BLOCK_L
    begin = s * chunk
    end = tl.minimum(begin + chunk, seq_len)

    kd = tl.make_tensor_descriptor(K + b * stride_kb + h * stride_kh, (seq_len, D),
                                   (stride_kn, 1), (BLOCK_L, D))
    acc = tl.zeros((D,), tl.float32)
    for start in range(begin, end, BLOCK_L):
        acc += tl.sum(kd.load([start, 0]).to(tl.float32), axis=0)
    tl.store(OUT + b * stride_ob + h * stride_oh + s * stride_os + tl.arange(0, D), acc)


def k_window_mean(k: torch.Tensor, splits: int = 16, block_l: int = 64) -> torch.Tensor:
    """Per-head mean of ``[B, H, L, D]`` over L, as float32 ``[B, H, D]``."""
    batch, heads, seq_len, d = k.shape
    partial = torch.empty((batch, heads, splits, d), dtype=torch.float32, device=k.device)
    _k_mean_partial_kernel[(heads, splits, batch)](
        k, partial, seq_len,
        k.stride(0), k.stride(1), k.stride(2),
        partial.stride(0), partial.stride(1), partial.stride(2),
        D=d, SPLITS=splits, BLOCK_L=block_l, num_warps=8,
    )
    return partial.sum(dim=2) / seq_len


@triton.jit
def _quantise_k_kernel(
    K, KM, KI8, KS,
    stride_kb, stride_kh, stride_kn,
    stride_mb, stride_mh,
    stride_ib, stride_ih, stride_in,
    stride_sb, stride_sh,
    kv_len,
    D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Centre ``K`` by the per-head mean and quantise it to int8."""
    n = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    start = n * BLOCK_N

    km = tl.load(KM + b * stride_mb + h * stride_mh + tl.arange(0, D))
    kd = tl.make_tensor_descriptor(K + b * stride_kb + h * stride_kh, (kv_len, D),
                                   (stride_kn, 1), (BLOCK_N, D))
    k = kd.load([start, 0]).to(tl.float32) - km[None, :]

    # descriptor loads pad with zeros, which become -km after centring; mask
    # them out so the padding cannot inflate the block's scale
    rows = start + tl.arange(0, BLOCK_N)
    k = tl.where(rows[:, None] < kv_len, k, 0.0)
    scale = tl.maximum(tl.max(tl.abs(k)) * (1.0 / 127.0), 1.0e-8)
    tl.store(KS + b * stride_sb + h * stride_sh + n, scale)

    od = tl.make_tensor_descriptor(KI8 + b * stride_ib + h * stride_ih, (kv_len, D),
                                   (stride_in, 1), (BLOCK_N, D))
    od.store([start, 0], _quantise_i8(k, scale).to(tl.int8))


def quantise_k(
    k: torch.Tensor,
    km: Optional[torch.Tensor] = None,
    block_n: int = SAGE_BLOCK_N,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Centre and quantise ``[B, H, L, D]`` K to int8 with one scale per block."""
    batch, heads, kv_len, d = k.shape
    if km is None:
        km = k_window_mean(k)
    km = km.contiguous()
    blocks = triton.cdiv(kv_len, block_n)
    k_scale = torch.empty((batch, heads, blocks), dtype=torch.float32, device=k.device)
    k_int8 = torch.empty((batch, heads, kv_len, d), dtype=torch.int8, device=k.device)
    _quantise_k_kernel[(blocks, heads, batch)](
        k, km, k_int8, k_scale,
        k.stride(0), k.stride(1), k.stride(2),
        km.stride(0), km.stride(1),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        kv_len, D=d, BLOCK_N=block_n, num_warps=8,
    )
    return k_int8, k_scale


def sage_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    km: Optional[torch.Tensor] = None,
    block_n: int = SAGE_BLOCK_N,
) -> torch.Tensor:
    """int8 SageAttention over ``[B, H, L, D]`` tensors (non-causal)."""
    k_int8, k_scale = quantise_k(k, km, block_n)
    return _attend_int8(q, k_int8, k_scale, v, block_n)


# ─────────────────────────────────────────────────────────────────────────────
# The free rotation
# ─────────────────────────────────────────────────────────────────────────────

_HADAMARD4 = (
    (0.5, 0.5, 0.5, 0.5),
    (0.5, -0.5, 0.5, -0.5),
    (0.5, 0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5, 0.5),
)

_HADAMARD4_CACHE = {}


def hadamard4(device, dtype=torch.float32) -> torch.Tensor:
    """The 4x4 normalised Hadamard matrix (orthogonal, entries +-1/2).

    Cached per device: building it is a host-to-device copy, which would
    otherwise sit on every call's critical path.
    """
    key = (torch.device(device), dtype)
    if key not in _HADAMARD4_CACHE:
        _HADAMARD4_CACHE[key] = torch.tensor(_HADAMARD4, device=device, dtype=dtype)
    return _HADAMARD4_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end
# ─────────────────────────────────────────────────────────────────────────────

def fused_rope_prope_sage(
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
    rotate: bool = True,
) -> torch.Tensor:
    """RoPE+PRoPE fused into the Q/K/V stream, int8 attention, fused output PRoPE.

    ``rotate=False`` drops the Hadamard rotation, for A/B measurement; it is
    free, so there is no reason to turn it off in production.
    """
    if kv_viewmats is None:
        kv_viewmats, kv_Ks = q_viewmats, q_Ks
    # rope_dtype still selects the frequency values (complex32 / 64 / 128), but
    # not the rotation's arithmetic -- see _transform_tile. An fp64 table is
    # narrowed on the host: gathering fp64 in the table kernel costs 3x.
    fr = _freqs_real(freqs, rope_dtype)
    if fr.dtype == torch.float64:
        fr = fr.float()
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    batch, seq_q, heads, d = q.shape
    seq_k = k.shape[1]
    planes = d // 4
    ncam_q, ncam_k = q_viewmats.shape[1], kv_viewmats.shape[1]
    if seq_q % ncam_q or seq_k % ncam_k:
        raise ValueError("sequence length must be divisible by the camera count")

    rot = hadamard4(q.device) if rotate else None
    linear = (camera_translation_transform in (None, "linear")
              and (q_Ks is None) == (kv_Ks is None))
    cs_q, cs_k, p_q_t, p_kv_k, p_kv_v, p_q = _build_tables(
        fr, q_grid_sizes, k_grid_sizes, seq_q, seq_k, q_start_frame, k_start_frame,
        q_viewmats, q_Ks, kv_viewmats, kv_Ks, rot, cameras=linear)
    if not linear:
        p_q_t, p_kv_v, p_q = _prope_tables_batched(
            q_viewmats, q_Ks, kv_viewmats, kv_Ks, camera_translation_transform)
        p_kv_k = p_kv_v
        if rotate:
            # rotate the matrices of *this* transform; the Triton matrix kernel
            # only implements "linear", and using it here would hand Q and K a
            # different camera transform than V and the output (3.8% error)
            p_q_t, p_kv_k = (
                (rot @ m.reshape(*m.shape[:2], 4, 4)).reshape(m.shape)
                .to(torch.bfloat16).to(torch.float32)
                for m in (p_q_t, p_kv_v))

    q64, k64, v64 = q.view(torch.int64), k.view(torch.int64), v.view(torch.int64)
    dev = q.device

    # 2. Q_eff, and the K centre
    q_eff = torch.empty((batch, heads, seq_q, d), dtype=q.dtype, device=dev)
    k_mean = torch.empty((batch, heads, d), dtype=torch.float32, device=dev)
    # descriptor rows must be >= 16 bytes: at least 8 fp16 (cos, sin) columns
    mean_splits = max(1, min(_MEAN_SPLITS_MAX, planes // 8))
    # short windows are cheap to average in full, and a 1-in-8 sample of them
    # would be only a handful of tiles
    stride = _MEAN_STRIDE if seq_k >= 8 * _MEAN_STRIDE else _MEAN_BLOCK_T
    n_mean = triton.cdiv(seq_k, stride)
    count = (n_mean - 1) * _MEAN_BLOCK_T + min(_MEAN_BLOCK_T, seq_k - (n_mean - 1) * stride)
    n_q_tiles = triton.cdiv(seq_q, _PROLOGUE_BLOCK_T)
    _sage_prologue_kernel[(heads * (mean_splits + n_q_tiles), batch)](
        q64, q_eff.view(torch.int64), k64, k_mean, cs_q, cs_k, p_q_t, p_kv_k,
        seq_q, seq_k, seq_q // ncam_q, ncam_q, seq_k // ncam_k, ncam_k, 1.0 / count,
        H=heads, PLANES=planes, ROPE=True, BLOCK_T=_PROLOGUE_BLOCK_T,
        MEAN_SPLITS=mean_splits, MEAN_STRIDE=stride, MEAN_BT=_MEAN_BLOCK_T,
        num_warps=_PROLOGUE_WARPS,
    )

    # 3. int8 K and V_eff
    blocks = triton.cdiv(seq_k, SAGE_BLOCK_N)
    k_int8 = torch.empty((batch, heads, seq_k, d), dtype=torch.int8, device=dev)
    k_scale = torch.empty((batch, heads, blocks), dtype=torch.float32, device=dev)
    v_eff = torch.empty((batch, heads, seq_k, d), dtype=v.dtype, device=dev)
    _sage_kv_kernel[(2 * heads * blocks, batch)](
        k64, v64, k_int8.view(torch.int32), k_scale, v_eff.view(torch.int64), k_mean,
        cs_k, p_kv_k, p_kv_v, seq_k, seq_k // ncam_k, ncam_k,
        H=heads, PLANES=planes, ROPE=True, BLOCK_N=SAGE_BLOCK_N, num_warps=8,
    )

    # 4. attention
    o = _attend_int8(q_eff, k_int8, k_scale, v_eff)

    # 5. output projection, back to [B, L, H, D]; the reference's own kernel,
    # launched without its autotuner
    out = torch.empty((batch, seq_q, heads, d), dtype=q.dtype, device=dev)
    o64, y64 = o.view(torch.int64), out.view(torch.int64)
    _transform_kernel.fn[(heads, triton.cdiv(seq_q, _TRANSFORM_BLOCK_T), batch)](
        o64, y64, o64, p_q,
        o64.stride(0), o64.stride(2), o64.stride(1),
        y64.stride(0), y64.stride(1), y64.stride(2),
        0, p_q.stride(0),
        seq_q, seq_q // ncam_q, ncam_q,
        PLANES=planes, ROPE=0, BLOCK_T=_TRANSFORM_BLOCK_T, num_warps=8, num_stages=2,
    )
    return out
