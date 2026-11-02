"""Fused SVDQuant W4A4 + BF16 Low-Rank GEMM Kernel for Intel Arc (XPU).

Mathematical formulation:
  Y = Activation(Linear(QDQ_int4(X / s_in), QDQ_int4(W_res)) + Linear(X / s_in, L_A) @ L_B + bias)

Fusions included in this kernel:
1. Dynamic group-64 activation quantization on-the-fly.
2. Dynamic signed 4-bit weight unpacking & group-64 scaling in registers.
3. Compatibility with Nunchaku / DeepCompressor E8M0 power-of-2 scales via direct bit-shift:
   scale = 2^(E - 127) <=> (E << 23).to(float32, bitcast=True).
4. INT4 GEMM via Intel Arc XMX INT8-to-FP32 tensor processing units (`tl.dot`).
5. Low-rank BF16 GEMM fusion (fused `x_lora @ lora_b.T` into the GEMM register accumulator).
6. Bias addition in the epilogue.
7. Nonlinearity activation function epilogue (`gelu_tanh`, `silu`, `gelu`, or none).
8. Optional fused output INT4 quantization & nibble-packing (similar to Nunchaku's `qout`).
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _svdquant_w4a4_gemm_kernel(
    X_ptr,
    W_packed_ptr,
    W_scale_ptr,
    S_in_ptr,
    XLora_ptr,
    LoraB_ptr,
    Bias_ptr,
    Out_ptr,
    Qout_ptr,
    Oscale_ptr,
    M: int,
    N: int,
    K: int,
    stride_xm: int,
    stride_xk: int,
    stride_wn: int,
    stride_wk: int,
    stride_ws_n: int,
    stride_ws_k: int,
    stride_xl_m: int,
    stride_xl_r: int,
    stride_lb_n: int,
    stride_lb_r: int,
    stride_out_m: int,
    stride_out_n: int,
    stride_qo_m: int,
    stride_qo_n: int,
    stride_os_m: int,
    stride_os_n: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RANK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SMOOTH: tl.constexpr,
    HAS_LORA: tl.constexpr,
    ACT_MODE: tl.constexpr,  # 0: none, 1: gelu_tanh, 2: silu, 3: gelu
    IS_E8M0: tl.constexpr,  # True if scales are 8-bit unsigned exponents
    FUSE_QUANT_OUT: tl.constexpr,  # True to quantize and pack output to INT4
    WRITE_BF16_OUT: tl.constexpr,  # True to write full BF16 tensor to Out_ptr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_half = tl.arange(0, BLOCK_K // 2)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Base pointers
    x_base = X_ptr + offs_m[:, None] * stride_xm
    w_base = W_packed_ptr + offs_n[:, None] * stride_wn
    ws_base = W_scale_ptr + offs_n[:, None] * stride_ws_n

    num_k_blocks = tl.cdiv(K, BLOCK_K)

    for k_idx in range(0, num_k_blocks):
        k_start = k_idx * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # 1. Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = x_base + offs_k[None, :] * stride_xk
        x_tile = tl.load(
            x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)

        # Apply static input channel smoothing if provided: x = x / s_in
        if HAS_SMOOTH:
            s_in = tl.load(S_in_ptr + offs_k, mask=k_mask, other=1.0).to(
                tl.float32
            )
            x_tile = x_tile / s_in[None, :]

        # 2. Dynamic symmetric INT4 quantization of activation tile
        x_amax = tl.max(tl.abs(x_tile), axis=1)[:, None]
        x_scale = tl.maximum(x_amax / 7.0, 1e-8)
        scaled = x_tile / x_scale
        rounded = tl.where(scaled >= 0.0, scaled + 0.5, scaled - 0.5)
        x_q = tl.clamp(rounded, -8.0, 7.0).to(tl.int8)

        # 3. Load packed weights: [BLOCK_N, BLOCK_K // 2]
        offs_k_h = (k_start // 2) + offs_k_half
        w_ptrs = w_base + offs_k_h[None, :] * stride_wk
        w_packed = tl.load(
            w_ptrs,
            mask=n_mask[:, None] & (offs_k_h[None, :] < (K // 2)),
            other=0,
        )

        # 4. Unpack 4-bit weights
        w_even = (w_packed << 4) >> 4  # [BLOCK_N, BLOCK_K // 2]
        w_odd = w_packed >> 4          # [BLOCK_N, BLOCK_K // 2]
        w_q = tl.interleave(w_even, w_odd)  # [BLOCK_N, BLOCK_K]

        # 5. Load weight scale for this group: [BLOCK_N, 1]
        # Supports both standard floating-point scales and E8M0 power-of-2 scales via bit shift
        group_idx = k_start // GROUP_SIZE
        if IS_E8M0:
            # E8M0 is uint8 exponent: scale = 2^(e - 127)
            # In IEEE float32, exponent field is at bits 23..30 with bias 127
            e_val = tl.load(
                ws_base + group_idx * stride_ws_k,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            bits = e_val << 23
            w_scale = bits.to(tl.float32, bitcast=True)
        else:
            w_scale = tl.load(
                ws_base + group_idx * stride_ws_k,
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float32)

        # 6. Tensor Core Dot product using Intel XMX INT8 instructions
        int_dot = tl.dot(x_q, tl.trans(w_q))

        # 7. Apply scales: dot * x_scale [BLOCK_M, 1] * w_scale.T [1, BLOCK_N]
        acc += int_dot * x_scale * tl.trans(w_scale)

    # 8. Fused BF16 Low-Rank LoRA GEMM: acc += x_lora @ lora_b.T
    if HAS_LORA:
        offs_r = tl.arange(0, RANK)
        x_lora_ptrs = (
            XLora_ptr
            + offs_m[:, None] * stride_xl_m
            + offs_r[None, :] * stride_xl_r
        )
        lora_b_ptrs = (
            LoraB_ptr
            + offs_n[:, None] * stride_lb_n
            + offs_r[None, :] * stride_lb_r
        )

        x_lora_tile = tl.load(x_lora_ptrs, mask=m_mask[:, None], other=0.0).to(
            tl.float32
        )
        lora_b_tile = tl.load(lora_b_ptrs, mask=n_mask[:, None], other=0.0).to(
            tl.float32
        )

        acc += tl.dot(x_lora_tile, tl.trans(lora_b_tile))

    # 9. Fused Bias addition
    if HAS_BIAS:
        b = tl.load(Bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += b[None, :]

    # 10. Fused Nonlinearity activation epilogue
    if ACT_MODE == 1:
        # GELU tanh: 0.5 * x * (1 + tanh(u)) = x * sigmoid(2u)
        two_u = 1.5957691216057308 * (acc + 0.044715 * acc * acc * acc)
        acc = acc * tl.sigmoid(two_u)
    elif ACT_MODE == 2:
        # SiLU: x * sigmoid(x)
        acc = acc * tl.sigmoid(acc)
    elif ACT_MODE == 3:
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865475))

    # 11. Optional store BF16 output
    if WRITE_BF16_OUT:
        out_ptrs = (
            Out_ptr
            + offs_m[:, None] * stride_out_m
            + offs_n[None, :] * stride_out_n
        )
        tl.store(
            out_ptrs,
            acc.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    # 12. Optional Fused INT4 Output Quantization (like Nunchaku qout)
    if FUSE_QUANT_OUT:
        # Dynamic symmetric INT4 quantization of output tile
        out_amax = tl.max(tl.abs(acc), axis=1)[:, None]
        out_scale = tl.maximum(out_amax / 7.0, 1e-8)
        scaled_out = acc / out_scale
        rounded_out = tl.where(
            scaled_out >= 0.0, scaled_out + 0.5, scaled_out - 0.5
        )
        q_out = tl.clamp(rounded_out, -8.0, 7.0).to(tl.int8)

        # Pack into nibbles using tl.split on last dimension = 2
        q_3d = tl.reshape(q_out, (BLOCK_M, BLOCK_N // 2, 2))
        q_even, q_odd = tl.split(q_3d)
        packed_out = (q_even.to(tl.int8) & 0x0F) | (
            (q_odd.to(tl.int8) & 0x0F) << 4
        )

        offs_n_half = tl.arange(0, BLOCK_N // 2)
        qout_ptrs = (
            Qout_ptr
            + offs_m[:, None] * stride_qo_m
            + offs_n_half[None, :] * stride_qo_n
        )
        qout_mask = m_mask[:, None] & (
            (pid_n * (BLOCK_N // 2) + offs_n_half)[None, :] < (N // 2)
        )
        tl.store(qout_ptrs, packed_out, mask=qout_mask)

        # Store output scale: [M, num_n_blocks]
        oscale_ptrs = (
            Oscale_ptr
            + offs_m[:, None] * stride_os_m
            + pid_n * stride_os_n
        )
        tl.store(oscale_ptrs, out_scale.to(tl.bfloat16), mask=m_mask[:, None])


def triton_svdquant_w4a4_gemm(
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
    return_bf16: bool = True,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute fused SVDQuant W4A4 + BF16 Low-Rank GEMM + Activation on Intel XPU.

    Fuses:
    - INT4 GEMM
    - E8M0 power-of-2 scales via bit shifts (when w_scale.dtype is uint8)
    - BF16 LoRA GEMM (precomputes X @ lora_a.T, fuses X_lora @ lora_b.T in registers)
    - Bias addition
    - Nonlinearity epilogue ('none', 'gelu_tanh', 'silu', 'gelu')
    - Optional fused output INT4 quantization & packing (returning out, qout, oscales)
    """
    orig_shape = x.shape
    k = orig_shape[-1]
    x_2d = x.reshape(-1, k)
    m = x_2d.shape[0]
    n = w_packed.shape[0]

    # Output tensors
    out = (
        torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        if return_bf16
        else None
    )

    if fuse_quant_out:
        qout = torch.empty((m, n // 2), device=x.device, dtype=torch.int8)
        num_scale_cols = triton.cdiv(n, 64)
        oscale = torch.empty(
            (m, num_scale_cols), device=x.device, dtype=torch.bfloat16
        )
    else:
        qout = None
        oscale = None

    has_lora = lora_a is not None and lora_b is not None
    if has_lora:
        rank = lora_a.shape[0]
        # Precompute X_lora = (X / s_in) @ lora_a.T
        x_s = (x_2d / s_in) if s_in is not None else x_2d
        x_lora = torch.matmul(x_s, lora_a.t()).to(torch.bfloat16)
        lora_b_bf16 = lora_b.to(torch.bfloat16)
    else:
        rank = 16
        x_lora = x_2d
        lora_b_bf16 = x_2d

    act_modes = {"none": 0, "gelu_tanh": 1, "silu": 2, "gelu": 3}
    act_mode = act_modes.get(activation.lower(), 0)

    # Detect E8M0 scale dtype
    is_e8m0 = (
        w_scale.dtype == torch.uint8
        or (hasattr(torch, "float8_e8m0fnu") and w_scale.dtype == torch.float8_e8m0fnu)
    )

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = group_size

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))

    _svdquant_w4a4_gemm_kernel[grid](
        x_2d,
        w_packed,
        w_scale,
        s_in if s_in is not None else x_2d,
        x_lora,
        lora_b_bf16,
        bias if bias is not None else x_2d,
        out if out is not None else x_2d,
        qout if qout is not None else x_2d,
        oscale if oscale is not None else x_2d,
        m,
        n,
        k,
        x_2d.stride(0),
        x_2d.stride(1),
        w_packed.stride(0),
        w_packed.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        x_lora.stride(0),
        x_lora.stride(1) if has_lora else 0,
        lora_b_bf16.stride(0),
        lora_b_bf16.stride(1) if has_lora else 0,
        out.stride(0) if out is not None else 0,
        out.stride(1) if out is not None else 0,
        qout.stride(0) if qout is not None else 0,
        qout.stride(1) if qout is not None else 0,
        oscale.stride(0) if oscale is not None else 0,
        oscale.stride(1) if oscale is not None else 0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        RANK=rank,
        GROUP_SIZE=group_size,
        HAS_BIAS=bias is not None,
        HAS_SMOOTH=s_in is not None,
        HAS_LORA=has_lora,
        ACT_MODE=act_mode,
        IS_E8M0=is_e8m0,
        FUSE_QUANT_OUT=fuse_quant_out,
        WRITE_BF16_OUT=return_bf16,
    )

    if fuse_quant_out:
        if return_bf16:
            return out.view(*orig_shape[:-1], n), qout, oscale
        return qout, oscale

    return out.view(*orig_shape[:-1], n)
