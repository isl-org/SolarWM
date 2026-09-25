"""Numerical and dtype tests for RoPE precision variations (float64, float32, float16)."""

from __future__ import annotations

import pytest
import torch

from solarwm.backends.wan22.runtime.modeling.causal_model import echorope_apply
from solarwm.backends.wan22.runtime.modeling.model import rope_params


@pytest.mark.parametrize("rope_dtype,expected_real,expected_complex", [
    ("float64", torch.float64, torch.complex128),
    ("float32", torch.float32, torch.complex64),
    ("float16", torch.float16, torch.complex32),
])
def test_echorope_apply_dtypes(rope_dtype, expected_real, expected_complex):
    batch, seq_len, num_heads, head_dim = 1, 12, 4, 16
    c = head_dim // 2
    x = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
    grid_sizes = torch.tensor([[1, 3, 4]], dtype=torch.long)

    base_freqs = torch.cat(
        [
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
        ],
        dim=1,
    )
    freqs = base_freqs.to(expected_complex)

    out = echorope_apply(x, grid_sizes, freqs, rope_dtype=rope_dtype)
    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16


def test_echorope_float32_close_to_float64():
    torch.manual_seed(42)
    batch, seq_len, num_heads, head_dim = 1, 12, 4, 16
    c = head_dim // 2
    x = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
    grid_sizes = torch.tensor([[1, 3, 4]], dtype=torch.long)

    base_freqs = torch.cat(
        [
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
        ],
        dim=1,
    )

    out_fp64 = echorope_apply(x, grid_sizes, base_freqs, rope_dtype="float64")
    out_fp32 = echorope_apply(x, grid_sizes, base_freqs.to(torch.complex64), rope_dtype="float32")
    out_fp16 = echorope_apply(x, grid_sizes, base_freqs.to(torch.complex32), rope_dtype="float16")

    # Float32 should be virtually identical to Float64 in bfloat16 output
    max_diff_fp32 = (out_fp64.float() - out_fp32.float()).abs().max().item()
    assert max_diff_fp32 < 1e-2, f"fp32 diff too high: {max_diff_fp32}"

    # Float16 will have slight half-precision quantization differences
    max_diff_fp16 = (out_fp64.float() - out_fp16.float()).abs().max().item()
    assert max_diff_fp16 < 5e-2, f"fp16 diff too high: {max_diff_fp16}"


def test_echorope_on_device():
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
    device = "xpu"
    batch, seq_len, num_heads, head_dim = 1, 1215, 24, 128
    x = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    grid_sizes = torch.tensor([[3, 15, 27]], dtype=torch.long, device=device)
    base_freqs = torch.cat(
        [
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
        ],
        dim=1,
    ).to(device)

    import time
    for dt, c_dt in [("float64", torch.complex128), ("float32", torch.complex64), ("float16", torch.complex32)]:
        freqs = base_freqs.to(c_dt)
        torch.xpu.synchronize()
        t0 = time.time()
        for _ in range(10):
            out = echorope_apply(x, grid_sizes, freqs, rope_dtype=dt)
        torch.xpu.synchronize()
        print(f"\n{dt}: {(time.time() - t0)*100:.2f} ms per 10 calls")
        assert out.shape == x.shape
        assert out.device == x.device
