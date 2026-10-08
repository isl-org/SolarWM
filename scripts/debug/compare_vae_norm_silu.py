import time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _rms_norm_silu_kernel(
    X, GAMMA, BIAS, OUT,
    M, C,
    SCALE,
    stride_xm, stride_xc,
    stride_outm, stride_outc,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    cols = tl.arange(0, BLOCK_C)
    c_mask = cols < C

    gamma = tl.load(GAMMA + cols, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(BIAS + cols, mask=c_mask, other=0.0).to(tl.float32)

    ptrs = X + m_offsets[:, None] * stride_xm + cols[None, :] * stride_xc
    mask_2d = m_mask[:, None] & c_mask[None, :]
    x = tl.load(ptrs, mask=mask_2d, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=1)[:, None]
    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, 1e-12)

    normed = (x / norm) * SCALE * gamma[None, :] + bias[None, :]
    silu = normed / (1.0 + tl.exp(-normed))

    out_ptrs = OUT + m_offsets[:, None] * stride_outm + cols[None, :] * stride_outc
    tl.store(out_ptrs, silu.to(tl.bfloat16), mask=mask_2d)


@triton.jit
def _add_rms_norm_silu_kernel(
    X, RESIDUAL, GAMMA, BIAS, OUT, OUT_SUM,
    M, C,
    SCALE,
    stride_xm, stride_xc,
    stride_rm, stride_rc,
    stride_outm, stride_outc,
    stride_sm, stride_sc,
    STORE_SUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    cols = tl.arange(0, BLOCK_C)
    c_mask = cols < C

    gamma = tl.load(GAMMA + cols, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(BIAS + cols, mask=c_mask, other=0.0).to(tl.float32)

    mask_2d = m_mask[:, None] & c_mask[None, :]

    x_ptrs = X + m_offsets[:, None] * stride_xm + cols[None, :] * stride_xc
    res_ptrs = RESIDUAL + m_offsets[:, None] * stride_rm + cols[None, :] * stride_rc
    x = tl.load(x_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
    res = tl.load(res_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

    # In-register sum
    y = x + res

    if STORE_SUM:
        out_sum_ptrs = OUT_SUM + m_offsets[:, None] * stride_sm + cols[None, :] * stride_sc
        tl.store(out_sum_ptrs, y.to(tl.bfloat16), mask=mask_2d)

    sum_sq = tl.sum(y * y, axis=1)[:, None]
    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, 1e-12)

    normed = (y / norm) * SCALE * gamma[None, :] + bias[None, :]
    silu = normed / (1.0 + tl.exp(-normed))

    out_ptrs = OUT + m_offsets[:, None] * stride_outm + cols[None, :] * stride_outc
    tl.store(out_ptrs, silu.to(tl.bfloat16), mask=mask_2d)


def triton_rms_norm_silu(x, gamma, bias):
    C = x.shape[1]
    M = x.numel() // C
    out = torch.empty_like(x)
    scale = float(C ** 0.5)

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_M = 4 if C <= 512 else 2
    grid = (triton.cdiv(M, BLOCK_M),)

    x_2d = x.view(-1, C) if x.is_contiguous() else x
    stride_xm = C if x.is_contiguous(memory_format=torch.channels_last_3d) or x.is_contiguous() else x.stride(0)
    stride_xc = 1

    _rms_norm_silu_kernel[grid](
        x, gamma.flatten(), bias.flatten(), out,
        M, C, scale,
        stride_xm, stride_xc,
        stride_xm, stride_xc,
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
        num_warps=8 if C >= 512 else 4,
    )
    return out


def triton_add_rms_norm_silu(x, residual, gamma, bias, store_sum=True):
    C = x.shape[1]
    M = x.numel() // C
    out = torch.empty_like(x)
    out_sum = torch.empty_like(x) if store_sum else None
    scale = float(C ** 0.5)

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_M = 4 if C <= 512 else 2
    grid = (triton.cdiv(M, BLOCK_M),)

    stride_xm = C
    stride_xc = 1

    _add_rms_norm_silu_kernel[grid](
        x, residual, gamma.flatten(), bias.flatten(), out, out_sum if store_sum else x,
        M, C, scale,
        stride_xm, stride_xc,
        stride_xm, stride_xc,
        stride_xm, stride_xc,
        stride_xm, stride_xc,
        STORE_SUM=store_sum,
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
        num_warps=8 if C >= 512 else 4,
    )
    return out, out_sum


# Compiled baselines
@torch.compile(dynamic=False)
def compiled_rms_norm_silu(x, gamma, bias):
    scale = float(x.shape[1] ** 0.5)
    return F.silu(F.normalize(x, dim=1) * scale * gamma + bias)


@torch.compile(dynamic=False)
def compiled_add_rms_norm_silu(x, residual, gamma, bias):
    scale = float(x.shape[1] ** 0.5)
    y = x + residual
    out = F.silu(F.normalize(y, dim=1) * scale * gamma + bias)
    return out, y


def eager_rms_norm_silu(x, gamma, bias):
    scale = float(x.shape[1] ** 0.5)
    return F.silu(F.normalize(x, dim=1) * scale * gamma + bias)


def eager_add_rms_norm_silu(x, residual, gamma, bias):
    scale = float(x.shape[1] ** 0.5)
    y = x + residual
    return F.silu(F.normalize(y, dim=1) * scale * gamma + bias), y


def benchmark_op(fn, *args, repeat=50):
    # Warmup
    for _ in range(5):
        fn(*args)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    torch.xpu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000 / repeat


if __name__ == "__main__":
    dev = "xpu"
    test_cases = [
        ("Middle block", 1024, 1, 30, 54),
        ("Upsample 1",   1024, 4, 120, 216),
        ("Upsample 2",    512, 4, 120, 216),
        ("Upsample 3",    256, 4, 240, 432),
    ]

    print("=" * 85)
    print("KERNEL 1: RMS_norm + SiLU (Standalone Microbenchmark)")
    print("=" * 85)
    print(f"{'Layer / Shape':<25} {'Eager (ms)':>12} {'Compiled (ms)':>14} {'Triton (ms)':>12} {'Speedup vs Eager':>18}")
    print("-" * 85)

    for name, C, T, H, W in test_cases:
        x = torch.randn(1, C, T, H, W, device=dev, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last_3d)
        gamma = torch.randn(C, 1, 1, 1, device=dev, dtype=torch.bfloat16)
        bias = torch.randn(C, 1, 1, 1, device=dev, dtype=torch.bfloat16)

        # compile warmup
        compiled_rms_norm_silu(x, gamma, bias)
        torch.xpu.synchronize()

        ms_eager = benchmark_op(eager_rms_norm_silu, x, gamma, bias)
        ms_comp = benchmark_op(compiled_rms_norm_silu, x, gamma, bias)
        ms_tri = benchmark_op(triton_rms_norm_silu, x, gamma, bias)
        sp_eager = ms_eager / ms_tri
        sp_comp = ms_comp / ms_tri

        shape_str = f"{name} ({C}x{T}x{H}x{W})"
        print(f"{shape_str:<25} {ms_eager:12.3f} {ms_comp:14.3f} {ms_tri:12.3f} {sp_eager:17.2f}x (vs comp: {sp_comp:.2f}x)")

    print("\n" + "=" * 85)
    print("KERNEL 2: Add + RMS_norm + SiLU (Standalone Microbenchmark)")
    print("=" * 85)
    print(f"{'Layer / Shape':<25} {'Eager (ms)':>12} {'Compiled (ms)':>14} {'Triton (ms)':>12} {'Speedup vs Eager':>18}")
    print("-" * 85)

    for name, C, T, H, W in test_cases:
        x = torch.randn(1, C, T, H, W, device=dev, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last_3d)
        res = torch.randn(1, C, T, H, W, device=dev, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last_3d)
        gamma = torch.randn(C, 1, 1, 1, device=dev, dtype=torch.bfloat16)
        bias = torch.randn(C, 1, 1, 1, device=dev, dtype=torch.bfloat16)

        # compile warmup
        compiled_add_rms_norm_silu(x, res, gamma, bias)
        torch.xpu.synchronize()

        ms_eager = benchmark_op(eager_add_rms_norm_silu, x, res, gamma, bias)
        ms_comp = benchmark_op(compiled_add_rms_norm_silu, x, res, gamma, bias)
        ms_tri = benchmark_op(triton_add_rms_norm_silu, x, res, gamma, bias)
        sp_eager = ms_eager / ms_tri
        sp_comp = ms_comp / ms_tri

        shape_str = f"{name} ({C}x{T}x{H}x{W})"
        print(f"{shape_str:<25} {ms_eager:12.3f} {ms_comp:14.3f} {ms_tri:12.3f} {sp_eager:17.2f}x (vs comp: {sp_comp:.2f}x)")
