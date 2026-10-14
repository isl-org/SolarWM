from __future__ import annotations

from .qdq_utils import (
    convert_int4_weights_to_dpas_b,
    convert_w_scales_to_dpas,
    e8m0_to_f32,
    f32_to_e8m0,
    pack_int4_signed,
    pseudo_qdq_linear,
    unpack_int4_signed,
)
from .reference import (
    bf16_attention_projections,
    bf16_ffn,
    bf16_linear,
    bf16_linear_activation,
    quantize_activation_int4,
    svdquant_fused_ffn_ref,
    svdquant_w4a4_linear_ref,
)
from .triton_w4a4_gemm import triton_svdquant_w4a4_gemm

__all__ = [
    # BF16 Reference Baseline Kernels
    "bf16_linear",
    "bf16_linear_activation",
    "bf16_ffn",
    "bf16_attention_projections",
    # Quantization Utilities & Reference Kernels
    "pack_int4_signed",
    "unpack_int4_signed",
    "convert_int4_weights_to_dpas_b",
    "convert_w_scales_to_dpas",
    "e8m0_to_f32",
    "f32_to_e8m0",
    "quantize_activation_int4",
    "pseudo_qdq_linear",
    "svdquant_w4a4_linear_ref",
    "svdquant_fused_ffn_ref",
    # Triton Fused Kernel
    "triton_svdquant_w4a4_gemm",
]
