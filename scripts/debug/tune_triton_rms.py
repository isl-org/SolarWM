import time
import torch
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

    # 2D tile: [BLOCK_M, BLOCK_C]
    ptrs = X + m_offsets[:, None] * stride_xm + cols[None, :] * stride_xc
    mask_2d = m_mask[:, None] & c_mask[None, :]
    x = tl.load(ptrs, mask=mask_2d, other=0.0).to(tl.float32)

    # Sum of squares across dim 1
    sum_sq = tl.sum(x * x, axis=1)[:, None]
    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, 1e-12)

    normed = (x / norm) * SCALE * gamma[None, :] + bias[None, :]
    silu = normed / (1.0 + tl.exp(-normed))

    out_ptrs = OUT + m_offsets[:, None] * stride_outm + cols[None, :] * stride_outc
    tl.store(out_ptrs, silu.to(tl.bfloat16), mask=mask_2d)


def benchmark_config(M, C, BLOCK_M, num_warps=8):
    dev = "xpu"
    x = torch.randn(M, C, device=dev, dtype=torch.bfloat16)
    gamma = torch.randn(C, device=dev, dtype=torch.bfloat16)
    bias = torch.randn(C, device=dev, dtype=torch.bfloat16)
    out = torch.empty_like(x)

    BLOCK_C = triton.next_power_of_2(C)
    grid = (triton.cdiv(M, BLOCK_M),)

    for _ in range(5):
        _rms_norm_silu_kernel[grid](
            x, gamma, bias, out,
            M, C, float(C**0.5),
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
            num_warps=num_warps,
        )
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    N = 30
    for _ in range(N):
        _rms_norm_silu_kernel[grid](
            x, gamma, bias, out,
            M, C, float(C**0.5),
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
            num_warps=num_warps,
        )
    torch.xpu.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000 / N
    bytes_transferred = M * C * 2 * 2
    bw = bytes_transferred / (ms * 1e-3) / 1e9
    return ms, bw


if __name__ == "__main__":
    print(f"{'M':>8} {'C':>6} {'BLOCK_M':>8} {'warps':>6} {'ms':>10} {'BW (GB/s)':>12}")
    for C in [256, 512, 1024]:
        for M in [1620, 103680]:
            best_bw = 0
            best_cfg = None
            for bm in [1, 2, 4, 8]:
                for nw in [4, 8]:
                    ms, bw = benchmark_config(M, C, bm, nw)
                    if bw > best_bw:
                        best_bw = bw
                        best_cfg = (bm, nw, ms)
            bm, nw, ms = best_cfg
            print(f"{M:8d} {C:6d} {bm:8d} {nw:6d} {ms:10.3f} {best_bw:12.2f}")
