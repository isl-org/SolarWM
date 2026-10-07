"""Fused RoPE + PRoPE + attention kernels for Intel Arc (Xe2), in Triton.

* :func:`fused_rope_prope_sage`           -- int8 SageAttention core (fused_sage.py)
* :func:`fused_rope_prope_sdpa_split`     -- the reference kernel, SDPA core (fused_sdpa.py)
* :func:`fused_rope_prope_sdpa_reference` -- the unfused PyTorch pipeline (unfused_kernel.py)
"""

from .fused_sdpa import fused_rope_prope_sdpa_split
from .fused_sage import fused_rope_prope_sage
from .unfused_kernel import fused_rope_prope_sdpa_reference

fused_rope_prope_sdpa = fused_rope_prope_sdpa_split

__all__ = [
    "fused_rope_prope_sage",
    "fused_rope_prope_sdpa",
    "fused_rope_prope_sdpa_split",
    "fused_rope_prope_sdpa_reference",
]
