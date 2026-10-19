"""SVDQuant INT4 group-64 calibration and export toolkit for Wan2.2 DiT on Intel B70."""

from .act_policy import search_act_clip_ratio
from .attn_dispatch import ATTENTION_BACKENDS, resolve_attention_backend, with_attention_backend
from .capture import ActivationCaptureHook, HessianCollector
from .exporter import export_svdquant_safetensors, load_svdquant_manifest, reshard_svdquant_directory
from .gptq import gptq_quantize_residual
from .quantizer import (
    dequantize_int4_weights,
    pack_int4_signed,
    quantize_activations_dynamic,
    quantize_weights_rtn_g64,
    unpack_int4_signed,
)
from .smoothing import compute_joint_smoothing_scales, compute_smoothing_scale
from .svd import extract_low_rank_svd

__all__ = [
    "quantize_weights_rtn_g64",
    "quantize_activations_dynamic",
    "pack_int4_signed",
    "unpack_int4_signed",
    "dequantize_int4_weights",
    "compute_smoothing_scale",
    "compute_joint_smoothing_scales",
    "extract_low_rank_svd",
    "gptq_quantize_residual",
    "search_act_clip_ratio",
    "ATTENTION_BACKENDS",
    "with_attention_backend",
    "resolve_attention_backend",
    "ActivationCaptureHook",
    "HessianCollector",
    "export_svdquant_safetensors",
    "load_svdquant_manifest",
    "reshard_svdquant_directory",
]
