"""Unit tests for fused Triton VAE kernels with ULP-tolerance validation."""

import pytest
import torch
import torch.nn.functional as F

from solarwm.kernels.vae_fused import triton_add_rms_norm_silu, triton_rms_norm_silu


def bf16_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute exact IEEE 754 ULP distance between two bfloat16 tensors."""
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert a.shape == b.shape
    i_a = a.view(torch.int16).long() & 0xFFFF
    i_b = b.view(torch.int16).long() & 0xFFFF
    ord_a = torch.where(i_a >= 0x8000, 0x8000 - (i_a & 0x7FFF), 0x8000 + i_a)
    ord_b = torch.where(i_b >= 0x8000, 0x8000 - (i_b & 0x7FFF), 0x8000 + i_b)
    return (ord_a - ord_b).abs()


def _fp32_rms_norm_silu_ref(x_f: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """High-precision FP32 reference with float accumulator."""
    C = x_f.shape[1]
    scale = float(C ** 0.5)
    norm = x_f.norm(2, dim=1, keepdim=True).clamp_min(1e-12)
    normed = (x_f / norm) * scale * gamma.float()
    if bias is not None:
        normed = normed + bias.float()
    return F.silu(normed).bfloat16()


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required for Triton kernels")
@pytest.mark.parametrize("C", [256, 512, 1024])
@pytest.mark.parametrize("with_bias", [False, True])
def test_triton_rms_norm_silu_ulp_parity_5d(C: int, with_bias: bool):
    torch.manual_seed(42)
    dev = "xpu"
    T, H, W = 2, 16, 24
    x = torch.randn(1, C, T, H, W, device=dev, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last_3d)
    gamma = torch.randn(C, 1, 1, 1, device=dev, dtype=torch.bfloat16)
    bias = torch.randn(C, 1, 1, 1, device=dev, dtype=torch.bfloat16) if with_bias else None

    # Reference
    ref = _fp32_rms_norm_silu_ref(x.float(), gamma, bias)
    # Triton kernel
    out = triton_rms_norm_silu(x, gamma, bias)

    ulp = bf16_ulp_distance(ref, out)
    normal_mask = ref.abs() > 1e-4

    # In normal IEEE 754 range, Triton kernel matches within 2 ULPs
    assert ulp[normal_mask].max().item() <= 2, f"Normal range max ULP > 2 for C={C}, with_bias={with_bias}"
    assert ulp.float().mean().item() < 0.01, f"Mean ULP drift {ulp.float().mean().item()} too large for C={C}"


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required for Triton kernels")
@pytest.mark.parametrize("C", [256, 512])
def test_triton_rms_norm_silu_4d(C: int):
    torch.manual_seed(42)
    dev = "xpu"
    H, W = 32, 48
    x = torch.randn(1, C, H, W, device=dev, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    gamma = torch.randn(C, 1, 1, device=dev, dtype=torch.bfloat16)

    ref = _fp32_rms_norm_silu_ref(x.float(), gamma, None)
    out = triton_rms_norm_silu(x, gamma, None)

    ulp = bf16_ulp_distance(ref, out)
    normal_mask = ref.abs() > 1e-4
    assert ulp[normal_mask].max().item() <= 2


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required for Triton kernels")
@pytest.mark.parametrize("C", [256, 512, 1024])
@pytest.mark.parametrize("store_sum", [True, False])
def test_triton_add_rms_norm_silu_ulp_parity(C: int, store_sum: bool):
    torch.manual_seed(42)
    dev = "xpu"
    T, H, W = 2, 16, 24
    x = torch.randn(1, C, T, H, W, device=dev, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last_3d)
    res = torch.randn(1, C, T, H, W, device=dev, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last_3d)
    gamma = torch.randn(C, 1, 1, 1, device=dev, dtype=torch.bfloat16)

    # Reference with in-register FP32 addition
    y_f = x.float() + res.float()
    ref_out = _fp32_rms_norm_silu_ref(y_f, gamma, None)

    out, out_sum = triton_add_rms_norm_silu(x, res, gamma, None, store_sum=store_sum)

    ulp = bf16_ulp_distance(ref_out, out)
    normal_mask = ref_out.abs() > 1e-4

    assert ulp[normal_mask].max().item() <= 2, f"Max ULP drift > 2 for C={C}"
    assert ulp.float().mean().item() < 0.01, f"Mean ULP drift {ulp.float().mean().item()} too large for C={C}"

    if store_sum:
        assert out_sum is not None
        # Sum is stored as bfloat16: check it matches (x + res)
        y_ref_bf16 = (x.float() + res.float()).bfloat16()
        sum_diff = (out_sum.float() - y_ref_bf16.float()).abs().max().item()
        assert sum_diff == 0.0, f"Stored sum diff {sum_diff} is non-zero"
    else:
        assert out_sum is None


def test_triton_fused_cpu_fallback():
    C = 64
    x = torch.randn(1, C, 2, 8, 8, dtype=torch.bfloat16)
    gamma = torch.ones(C, 1, 1, 1, dtype=torch.bfloat16)
    out = triton_rms_norm_silu(x, gamma)
    assert out.shape == x.shape

    res = torch.randn(1, C, 2, 8, 8, dtype=torch.bfloat16)
    out_add, sum_add = triton_add_rms_norm_silu(x, res, gamma, store_sum=True)
    assert out_add.shape == x.shape
    assert sum_add.shape == x.shape
