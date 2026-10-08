import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _rms_norm_silu_kernel(
    X, GAMMA, BIAS, OUT,
    M, C,
    SCALE,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    x_ptrs = X + row * C + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # 1. Compute L2 norm across C
    sum_sq = tl.sum(x * x, axis=0)
    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, 1e-12)

    # 2. Normalize and scale
    gamma = tl.load(GAMMA + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(BIAS + cols, mask=mask, other=0.0).to(tl.float32)

    normed = (x / norm) * SCALE * gamma + bias
    # 3. SiLU
    silu = normed / (1.0 + tl.exp(-normed))

    out_ptrs = OUT + row * C + cols
    tl.store(out_ptrs, silu.to(tl.bfloat16), mask=mask)


@triton.jit
def _add_rms_norm_silu_kernel(
    X, RESIDUAL, GAMMA, BIAS, OUT, OUT_SUM,
    M, C,
    SCALE,
    STORE_SUM: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    x_ptrs = X + row * C + cols
    res_ptrs = RESIDUAL + row * C + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Residual addition
    y = x + res

    if STORE_SUM:
        out_sum_ptrs = OUT_SUM + row * C + cols
        tl.store(out_sum_ptrs, y.to(tl.bfloat16), mask=mask)

    # 1. Compute L2 norm across C
    sum_sq = tl.sum(y * y, axis=0)
    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, 1e-12)

    # 2. Normalize and scale
    gamma = tl.load(GAMMA + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(BIAS + cols, mask=mask, other=0.0).to(tl.float32)

    normed = (y / norm) * SCALE * gamma + bias
    # 3. SiLU
    silu = normed / (1.0 + tl.exp(-normed))

    out_ptrs = OUT + row * C + cols
    tl.store(out_ptrs, silu.to(tl.bfloat16), mask=mask)


def triton_rms_norm_silu(x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    C = x.shape[1]
    M = x.numel() // C
    out = torch.empty_like(x)
    scale = float(C ** 0.5)

    BLOCK_C = triton.next_power_of_2(C)
    grid = (M,)
    _rms_norm_silu_kernel[grid](
        x, gamma.flatten(), bias.flatten(), out,
        M, C, scale,
        BLOCK_C=BLOCK_C,
    )
    return out


def triton_add_rms_norm_silu(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor, store_sum: bool = True
) -> tuple[torch.Tensor, torch.Tensor | None]:
    C = x.shape[1]
    M = x.numel() // C
    out = torch.empty_like(x)
    out_sum = torch.empty_like(x) if store_sum else None
    scale = float(C ** 0.5)

    BLOCK_C = triton.next_power_of_2(C)
    grid = (M,)
    _add_rms_norm_silu_kernel[grid](
        x, residual, gamma.flatten(), bias.flatten(), out, out_sum if store_sum else x,
        M, C, scale,
        STORE_SUM=store_sum,
        BLOCK_C=BLOCK_C,
    )
    return out, out_sum


if __name__ == "__main__":
    dev = "xpu"
    for C in [256, 512, 1024]:
        T, H, W = 2, 30, 54
        x = torch.randn(1, C, T, H, W, device=dev, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last_3d)
        res = torch.randn(1, C, T, H, W, device=dev, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last_3d)
        gamma = torch.randn(C, 1, 1, 1, device=dev, dtype=torch.bfloat16)
        bias = torch.randn(C, 1, 1, 1, device=dev, dtype=torch.bfloat16)
        scale = float(C ** 0.5)

        # PyTorch Eager reference
        norm_ref = F.normalize(x, dim=1)
        ref_norm_silu = F.silu(norm_ref * scale * gamma + bias)

        tri_norm_silu = triton_rms_norm_silu(x, gamma, bias)
        diff1 = (tri_norm_silu.float() - ref_norm_silu.float()).abs().max().item()

        # Add + Norm + SiLU reference
        y_ref = x + res
        norm_y = F.normalize(y_ref, dim=1)
        ref_add_norm_silu = F.silu(norm_y * scale * gamma + bias)

        tri_add_silu, tri_sum = triton_add_rms_norm_silu(x, res, gamma, bias, store_sum=True)
        diff2 = (tri_add_silu.float() - ref_add_norm_silu.float()).abs().max().item()
        diff_sum = (tri_sum.float() - y_ref.float()).abs().max().item()

        print(f"C={C}: diff(norm_silu)={diff1:.4e}, diff(add_norm_silu)={diff2:.4e}, diff(sum)={diff_sum:.4e}")
