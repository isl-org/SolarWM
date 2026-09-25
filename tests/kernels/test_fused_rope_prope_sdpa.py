"""Tests for fused RoPE + PRoPE + SDPA reference implementation and ULP equivalence verification."""

import pytest
import torch

from solarwm.backends.wan22.runtime.modeling.attention import attention
from solarwm.backends.wan22.runtime.modeling.camera_prope import (
    _apply_tiled_projmat,
    prope_apply_fns_separate_cached,
)
from solarwm.backends.wan22.runtime.modeling.causal_model import block_relativistic_rope
from solarwm.kernels.fused_rope_prope_sdpa import (
    apply_prope,
    apply_rope,
    assert_bit_exact,
    assert_close_ulp,
    compare_tensors_bit_exact,
    compare_tensors_ulp,
    compute_ulp_diff,
    create_test_inputs,
    fused_rope_prope_sdpa_reference,
)


# =========================================================================
# 1. ULP Distance & Equivalence Unit Tests
# =========================================================================

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_compute_ulp_diff_identical(dtype):
    """Verify that identical tensors produce 0 ULPs distance."""
    x = torch.tensor([1.0, -2.5, 0.0, 100.0], dtype=dtype)
    ulp = compute_ulp_diff(x, x)
    assert torch.all(ulp == 0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_compute_ulp_diff_signed_zeros(dtype):
    """Verify that +0.0 and -0.0 produce 0 ULPs distance."""
    pos_zero = torch.tensor([0.0], dtype=dtype)
    neg_zero = torch.tensor([-0.0], dtype=dtype)
    ulp = compute_ulp_diff(pos_zero, neg_zero)
    assert ulp.item() == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_compute_ulp_diff_single_ulp_step(dtype):
    """Verify that 1.0 + eps produces exactly 1 ULP distance from 1.0."""
    eps = torch.finfo(dtype).eps
    v1 = torch.tensor([1.0], dtype=dtype)
    v2 = torch.tensor([1.0 + eps], dtype=dtype)
    ulp = compute_ulp_diff(v1, v2)
    assert ulp.item() == 1


def test_ulp_reporter_tolerance_pass_and_fail():
    """Verify that assert_close_ulp passes within tolerance and raises on larger discrepancies."""
    t1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    t2 = t1.clone()

    # Perturb index 1 by 1 ULP (16-bit integer + 1)
    t2_int = t2.view(torch.int16)
    t2_int[1] += 1

    # Should pass with max_ulp >= 1
    report = compare_tensors_ulp(t2, t1, max_ulp=1)
    assert report.is_equivalent
    assert not report.is_exact
    assert report.max_ulp == 1
    assert report.mismatched_elements == 0
    assert_close_ulp(t2, t1, max_ulp=1)

    # Should fail with max_ulp == 0 (strict bit-exact)
    report_strict = compare_tensors_ulp(t2, t1, max_ulp=0)
    assert not report_strict.is_equivalent
    assert report_strict.mismatched_elements == 1
    assert report_strict.first_mismatch_idx == (1,)
    assert report_strict.first_mismatch_ulp == 1

    with pytest.raises(AssertionError, match="within allowed ULPs"):
        assert_close_ulp(t2, t1, max_ulp=0)


# =========================================================================
# 2. Pipeline Equivalence Tests (Reference vs In-tree Model)
# =========================================================================

@pytest.mark.parametrize("rope_dtype", ["float64", "float32", "float16"])
def test_reference_equivalent_against_in_tree_model(rope_dtype):
    """Verify that fused_rope_prope_sdpa_reference produces equivalent results

    within a few ULPs (and in fact 0 ULPs) against in-tree code across all RoPE dtypes.
    """
    inputs = create_test_inputs(
        batch=1,
        num_heads=4,
        head_dim=16,
        q_frames=2,
        kv_frames=4,
        h_patches=3,
        w_patches=4,
        rope_dtype=rope_dtype,
        seed=123,
    )

    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    freqs = inputs["freqs"]
    q_grid = inputs["q_grid_sizes"]
    k_grid = inputs["k_grid_sizes"]
    q_viewmats = inputs["q_viewmats"]
    q_Ks = inputs["q_Ks"]
    kv_viewmats = inputs["kv_viewmats"]
    kv_Ks = inputs["kv_Ks"]
    q_start = inputs["q_start_frame"]
    k_start = inputs["k_start_frame"]

    # 1. Run via in-tree functions
    roped_q_intree = block_relativistic_rope(
        q, q_grid, freqs, start_frame=q_start, rope_dtype=rope_dtype
    )
    roped_k_intree = block_relativistic_rope(
        k, k_grid, freqs, start_frame=k_start, rope_dtype=rope_dtype
    )

    cache = {}
    apply_fn_q, apply_fn_kv, apply_fn_o = prope_apply_fns_separate_cached(
        cache,
        head_dim=q.shape[-1],
        q_viewmats=q_viewmats,
        q_Ks=q_Ks,
        kv_viewmats=kv_viewmats,
        kv_Ks=kv_Ks,
        kv_window=(0, k.shape[1]),
    )

    attn_q = apply_fn_q(roped_q_intree.transpose(1, 2)).transpose(1, 2)
    attn_k = apply_fn_kv(roped_k_intree.transpose(1, 2)).transpose(1, 2)
    attn_v = apply_fn_kv(v.transpose(1, 2)).transpose(1, 2)

    sdpa_out = attention(attn_q, attn_k, attn_v)
    expected_out = apply_fn_o(sdpa_out.transpose(1, 2)).transpose(1, 2)

    # 2. Run via standalone reference
    actual_out = fused_rope_prope_sdpa_reference(
        q,
        k,
        v,
        freqs=freqs,
        q_grid_sizes=q_grid,
        k_grid_sizes=k_grid,
        q_viewmats=q_viewmats,
        q_Ks=q_Ks,
        kv_viewmats=kv_viewmats,
        kv_Ks=kv_Ks,
        q_start_frame=q_start,
        k_start_frame=k_start,
        rope_dtype=rope_dtype,
    )

    # 3. Assert equivalence within tolerance (max_ulp <= 2)
    assert_close_ulp(
        actual_out,
        expected_out,
        max_ulp=2,
        message=f"Standalone reference differs from in-tree code for rope_dtype={rope_dtype}",
    )


def test_rope_standalone_equivalence():
    """Verify standalone apply_rope matches block_relativistic_rope within <= 2 ULPs."""
    inputs = create_test_inputs(batch=1, num_heads=4, head_dim=16, q_frames=3, h_patches=2, w_patches=3)
    q = inputs["q"]
    freqs = inputs["freqs"]
    grid = inputs["q_grid_sizes"]

    for dtype in ["float64", "float32", "float16"]:
        expected = block_relativistic_rope(q, grid, freqs, start_frame=2, rope_dtype=dtype)
        actual = apply_rope(q, grid, freqs, start_frame=2, rope_dtype=dtype)
        assert_close_ulp(actual, expected, max_ulp=2)


def test_prope_standalone_equivalence():
    """Verify standalone apply_prope matches in-tree _apply_tiled_projmat within <= 2 ULPs."""
    B, L, H, D = 1, 24, 4, 16
    feats = torch.randn(B, L, H, D, dtype=torch.bfloat16)
    matrix = torch.randn(B, L, 4, 4, dtype=torch.float32)

    # In-tree expects [B, H, L, D]
    expected = _apply_tiled_projmat(feats.transpose(1, 2), matrix).transpose(1, 2)
    # Standalone accepts native [B, L, H, D]
    actual = apply_prope(feats, matrix)

    assert_close_ulp(actual, expected, max_ulp=2)


def test_production_chunk4_geometry_equivalence():
    """Verify numerical equivalence on authentic Stage2 production chunk 4 shapes

    (Lq=1215, Lk=6075, num_heads=24, head_dim=128).
    """
    inputs = create_test_inputs(
        batch=1,
        num_heads=24,
        head_dim=128,
        q_frames=3,
        kv_frames=15,
        h_patches=15,
        w_patches=27,
        rope_dtype="float32",
        seed=999,
    )

    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    freqs = inputs["freqs"]
    q_grid = inputs["q_grid_sizes"]
    k_grid = inputs["k_grid_sizes"]
    q_viewmats = inputs["q_viewmats"]
    q_Ks = inputs["q_Ks"]
    kv_viewmats = inputs["kv_viewmats"]
    kv_Ks = inputs["kv_Ks"]
    q_start = inputs["q_start_frame"]
    k_start = inputs["k_start_frame"]

    # In-tree
    roped_q_intree = block_relativistic_rope(q, q_grid, freqs, start_frame=q_start, rope_dtype="float32")
    roped_k_intree = block_relativistic_rope(k, k_grid, freqs, start_frame=k_start, rope_dtype="float32")

    cache = {}
    apply_fn_q, apply_fn_kv, apply_fn_o = prope_apply_fns_separate_cached(
        cache,
        head_dim=128,
        q_viewmats=q_viewmats,
        q_Ks=q_Ks,
        kv_viewmats=kv_viewmats,
        kv_Ks=kv_Ks,
        kv_window=(0, k.shape[1]),
    )

    attn_q = apply_fn_q(roped_q_intree.transpose(1, 2)).transpose(1, 2)
    attn_k = apply_fn_kv(roped_k_intree.transpose(1, 2)).transpose(1, 2)
    attn_v = apply_fn_kv(v.transpose(1, 2)).transpose(1, 2)

    sdpa_out = attention(attn_q, attn_k, attn_v)
    expected_out = apply_fn_o(sdpa_out.transpose(1, 2)).transpose(1, 2)

    # Reference
    actual_out = fused_rope_prope_sdpa_reference(
        q,
        k,
        v,
        freqs=freqs,
        q_grid_sizes=q_grid,
        k_grid_sizes=k_grid,
        q_viewmats=q_viewmats,
        q_Ks=q_Ks,
        kv_viewmats=kv_viewmats,
        kv_Ks=kv_Ks,
        q_start_frame=q_start,
        k_start_frame=k_start,
        rope_dtype="float32",
    )

    assert_close_ulp(actual_out, expected_out, max_ulp=2)


def test_production_chunk4_on_xpu_if_available():
    """Verify numerical equivalence on Intel XPU device if available."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    device = "xpu"
    inputs = create_test_inputs(
        batch=1,
        num_heads=24,
        head_dim=128,
        q_frames=3,
        kv_frames=15,
        h_patches=15,
        w_patches=27,
        device=device,
        rope_dtype="float32",
        seed=42,
    )

    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    freqs = inputs["freqs"]
    q_grid = inputs["q_grid_sizes"]
    k_grid = inputs["k_grid_sizes"]
    q_viewmats = inputs["q_viewmats"]
    q_Ks = inputs["q_Ks"]
    kv_viewmats = inputs["kv_viewmats"]
    kv_Ks = inputs["kv_Ks"]
    q_start = inputs["q_start_frame"]
    k_start = inputs["k_start_frame"]

    roped_q_intree = block_relativistic_rope(q, q_grid, freqs, start_frame=q_start, rope_dtype="float32")
    roped_k_intree = block_relativistic_rope(k, k_grid, freqs, start_frame=k_start, rope_dtype="float32")

    cache = {}
    apply_fn_q, apply_fn_kv, apply_fn_o = prope_apply_fns_separate_cached(
        cache,
        head_dim=128,
        q_viewmats=q_viewmats,
        q_Ks=q_Ks,
        kv_viewmats=kv_viewmats,
        kv_Ks=kv_Ks,
        kv_window=(0, k.shape[1]),
    )

    attn_q = apply_fn_q(roped_q_intree.transpose(1, 2)).transpose(1, 2)
    attn_k = apply_fn_kv(roped_k_intree.transpose(1, 2)).transpose(1, 2)
    attn_v = apply_fn_kv(v.transpose(1, 2)).transpose(1, 2)

    sdpa_out = attention(attn_q, attn_k, attn_v)
    expected_out = apply_fn_o(sdpa_out.transpose(1, 2)).transpose(1, 2)

    actual_out = fused_rope_prope_sdpa_reference(
        q,
        k,
        v,
        freqs=freqs,
        q_grid_sizes=q_grid,
        k_grid_sizes=k_grid,
        q_viewmats=q_viewmats,
        q_Ks=q_Ks,
        kv_viewmats=kv_viewmats,
        kv_Ks=kv_Ks,
        q_start_frame=q_start,
        k_start_frame=k_start,
        rope_dtype="float32",
    )

    assert_close_ulp(actual_out, expected_out, max_ulp=2)


def test_bit_exact_backward_compatibility():
    """Verify that assert_bit_exact and compare_tensors_bit_exact continue to work."""
    t1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    t2 = t1.clone()
    assert_bit_exact(t2, t1)

    t2_int = t2.view(torch.int16)
    t2_int[1] ^= 1

    report = compare_tensors_bit_exact(t2, t1)
    assert not report.is_exact
    assert report.mismatched_elements == 1
    assert report.total_elements == 3
    assert report.first_mismatch_idx == (1,)
