"""Fused RoPE + PRoPE + attention kernels for Intel Arc (Xe2), in Triton.

* :func:`fused_rope_prope_sage`           -- int8 SageAttention core (fused_sage.py)
* :func:`fused_rope_prope_sdpa_split`     -- the reference kernel, SDPA core (fused_sdpa.py)
* :func:`fused_rope_prope_sdpa_reference` -- the unfused PyTorch pipeline (unfused_kernel.py)
"""

from .fused_sdpa import (
    fused_rope_prope_sdpa_split,
    rope_prope_transform,
    ROPE_MODE,
    _prope_tables_triton,
    _prope_tables,
    build_rope_table,
    _freqs_real,
)
from .fused_sage import fused_rope_prope_sage
from .radial_attention import (
    RadialAttentionConfig,
    build_radial_block_mask,
    build_sage_block_indices,
    radial_density,
)
from .unfused_kernel import (
    fused_rope_prope_sdpa_reference,
    normalize_rope_precision,
)

fused_rope_prope_sdpa = fused_rope_prope_sdpa_split

__all__ = [
    "fused_rope_prope_sage",
    "fused_rope_prope_sdpa",
    "fused_rope_prope_sdpa_split",
    "fused_rope_prope_sdpa_reference",
    "rope_prope_transform",
    "ROPE_MODE",
    "normalize_rope_precision",
    "_prope_tables_triton",
    "_prope_tables",
    "build_rope_table",
    "_freqs_real",
    "RadialAttentionConfig",
    "build_radial_block_mask",
    "build_sage_block_indices",
    "radial_density",
]
