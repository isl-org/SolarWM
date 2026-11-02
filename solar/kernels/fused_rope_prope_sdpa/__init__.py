"""Fused RoPE + PRoPE + SDPA kernel reference and verification package.

This package provides:
- Clean standalone reference implementation (`fused_rope_prope_sdpa_reference`)
- Configurable RoPE precision (float16, float32, float64)
- PRoPE camera projection and matrix prep helpers
- ULP numerical equivalence testing utilities (`assert_close_ulp`, `compare_tensors_ulp`, `compute_ulp_diff`)
- Bit-exact testing and verification utilities (`assert_bit_exact`, `compare_tensors_bit_exact`)
- Realistic input generators matching Wan2.2 Stage2 chunk geometry
"""

from .reference import (
    ROPE_DTYPE_MAP,
    RoPEPRoPEConfig,
    apply_prope,
    apply_rope,
    build_wan22_freqs,
    fused_rope_prope_sdpa_reference,
    invert_k,
    invert_se3,
    lift_k,
    normalize_rope_precision,
    prepare_prope_matrices,
    rope_params,
    scaled_dot_product_attention_reference,
    transform_relative_viewmats,
)
from .testing import (
    BitExactReport,
    EquivalenceReport,
    ULPEquivalenceReport,
    assert_bit_exact,
    assert_close_ulp,
    assert_equivalent,
    compare_equivalence,
    compare_tensors_bit_exact,
    compare_tensors_ulp,
    compute_ulp_diff,
    create_test_inputs,
)

__all__ = [
    "ROPE_DTYPE_MAP",
    "RoPEPRoPEConfig",
    "apply_prope",
    "apply_rope",
    "build_wan22_freqs",
    "fused_rope_prope_sdpa_reference",
    "invert_k",
    "invert_se3",
    "lift_k",
    "normalize_rope_precision",
    "prepare_prope_matrices",
    "rope_params",
    "scaled_dot_product_attention_reference",
    "transform_relative_viewmats",
    "BitExactReport",
    "EquivalenceReport",
    "ULPEquivalenceReport",
    "compute_ulp_diff",
    "compare_tensors_ulp",
    "assert_close_ulp",
    "compare_equivalence",
    "assert_equivalent",
    "compare_tensors_bit_exact",
    "assert_bit_exact",
    "create_test_inputs",
]
