"""Python wrapper for SageAttention v2 SYCL extension."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
from solarwm.kernels.sycl_jit import load_sycl_extension

_EXTENSION_NAME = "sageattn_v2_sycl"
_MODULE: Any = None
_LOAD_FAILED = False


def _get_module(gpu_arch: str = "bmg-g31") -> Any:
    global _MODULE, _LOAD_FAILED
    if _MODULE is not None:
        return _MODULE
    if _LOAD_FAILED:
        return None

    src_file = Path(__file__).parent / "sycl_tla_opt.sycl"
    if not src_file.is_file():
        warnings.warn(f"SageAttention v2 source file not found: {src_file}", stacklevel=2)
        _LOAD_FAILED = True
        return None

    try:
        _MODULE = load_sycl_extension(
            extension_name=_EXTENSION_NAME,
            sources=[src_file],
            gpu_arch=gpu_arch,
            include_sycl_tla=True,
        )
        return _MODULE
    except Exception as exc:
        warnings.warn(f"Failed to compile or load SageAttention v2 extension: {exc}", stacklevel=2)
        _LOAD_FAILED = True
        return None


def is_available(gpu_arch: str = "bmg-g31") -> bool:
    """Return True if the SageAttention v2 extension is loaded or can be compiled."""
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return False
    return _get_module(gpu_arch) is not None


def ensure_sageattn_v2(gpu_arch: str = "bmg-g31") -> bool:
    """Eagerly build or load the extension."""
    return _get_module(gpu_arch) is not None


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Run SageAttention v2 forward pass.

    Expects contiguous [B, L, H, D] XPU tensors where D is 64 or 128 and dtype is FP16 or BF16.
    """
    mod = _get_module()
    if mod is None:
        raise RuntimeError("SageAttention v2 SYCL extension is not available.")
    return mod.forward(q, k, v)


def quantize_q(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize Q: returns (q4, q_scale, q_mean)."""
    mod = _get_module()
    if mod is None:
        raise RuntimeError("SageAttention v2 SYCL extension is not available.")
    res = mod.quantize_q(q)
    return res[0], res[1], res[2]


def k_centre(k: torch.Tensor) -> torch.Tensor:
    """Compute K centre: returns [B, H, D] float32 tensor."""
    mod = _get_module()
    if mod is None:
        raise RuntimeError("SageAttention v2 SYCL extension is not available.")
    return mod.k_centre(k)


def quantize_k(
    k: torch.Tensor,
    k_mean: torch.Tensor,
    q_mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize K given k_mean and q_mean: returns (k4, k_scale, delta_s)."""
    mod = _get_module()
    if mod is None:
        raise RuntimeError("SageAttention v2 SYCL extension is not available.")
    res = mod.quantize_k(k, k_mean, q_mean)
    return res[0], res[1], res[2]


def pack_k(
    k: torch.Tensor,
    k_mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K to INT4 DPAS-B layout and compute k_scale: returns (k4, k_scale)."""
    mod = _get_module()
    if mod is None:
        raise RuntimeError("SageAttention v2 SYCL extension is not available.")
    res = mod.pack_k(k, k_mean)
    return res[0], res[1]


def compute_delta_s(
    k: torch.Tensor,
    k_mean: torch.Tensor,
    q_mean: torch.Tensor,
) -> torch.Tensor:
    """Compute delta_s from k, k_mean, and q_mean: returns delta_s tensor."""
    mod = _get_module()
    if mod is None:
        raise RuntimeError("SageAttention v2 SYCL extension is not available.")
    return mod.compute_delta_s(k, k_mean, q_mean)


def attention(
    q4: torch.Tensor,
    q_scale: torch.Tensor,
    q_mean: torch.Tensor,
    k4: torch.Tensor,
    k_scale: torch.Tensor,
    delta_s: torch.Tensor,
    v: torch.Tensor,
    q_len: int,
) -> torch.Tensor:
    """Run attention core given quantized Q/K components and BF16/FP16 V."""
    mod = _get_module()
    if mod is None:
        raise RuntimeError("SageAttention v2 SYCL extension is not available.")
    return mod.attention(q4, q_scale, q_mean, k4, k_scale, delta_s, v, q_len)
