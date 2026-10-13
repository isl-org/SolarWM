"""Select the Stage2 DiT self-attention backend on a loaded Wan model.

The backend only chooses which fused RoPE/PRoPE kernel the circular-cache
self-attention runs; DiT cross-attention and VAE attention always use SDPA.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from solarwm.backends.wan22.contracts import STAGE2_ATTENTION_KERNELS

ATTENTION_BACKENDS = tuple(STAGE2_ATTENTION_KERNELS)


def resolve_attention_backend(name: str) -> str:
    val = str(name).strip().lower()
    if val not in STAGE2_ATTENTION_KERNELS:
        raise ValueError(f"attention backend must be one of {ATTENTION_BACKENDS}; got {name!r}")
    return val


def _wan_model(model: Any) -> Any:
    while not hasattr(model, "set_fused_attention_kernel"):
        inner = getattr(model, "module", None) or getattr(model, "model", None)
        if inner is None or inner is model:
            raise TypeError(f"{type(model).__name__} does not wrap a CausalWanModel")
        model = inner
    return model


@contextlib.contextmanager
def with_attention_backend(model: Any, backend: str) -> Iterator[None]:
    """Run ``model``'s Stage2 self-attention on ``backend`` (sdpa | sageattnv1)."""
    root = _wan_model(model)
    previous = root.fused_kernel
    root.set_fused_attention_kernel(STAGE2_ATTENTION_KERNELS[resolve_attention_backend(backend)])
    try:
        yield
    finally:
        root.set_fused_attention_kernel(previous)
