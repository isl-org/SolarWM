"""ULP-based numerical equivalence and verification harness for fused RoPE + PRoPE + SDPA kernels.

This module provides:
1. ULP (Unit in the Last Place) equivalence checking between candidate and reference kernels.
2. Configurable ULP tolerance (e.g. <= 2 ULPs or <= 4 ULPs) to accommodate fused arithmetic variations.
3. Detailed ULP distribution and percentile reporting (0 ULP, <=1 ULP, <=2 ULPs, <=4 ULPs, max ULP, mean ULP).
4. Bit-level binary inspections showing exact integer/hex representation of first discrepancies.
5. Realistic input fixtures matching production Wan2.2 Stage2 chunk geometry.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from .reference import (
    build_wan22_freqs,
    fused_rope_prope_sdpa_reference,
    normalize_rope_precision,
)


@dataclass(frozen=True)
class ULPEquivalenceReport:
    """Detailed report comparing candidate kernel output against reference within ULP tolerance."""

    is_equivalent: bool
    is_exact: bool
    shape_matches: bool
    dtype_matches: bool
    max_ulp: int
    mean_ulp: float
    p99_ulp: float
    p999_ulp: float
    max_allowed_ulp: int
    ulp_counts: Dict[str, int] = field(default_factory=dict)
    ulp_fractions: Dict[str, float] = field(default_factory=dict)
    max_absolute_error: float = 0.0
    max_relative_error: float = 0.0
    total_elements: int = 0
    mismatched_elements: int = 0
    mismatch_fraction: float = 0.0
    first_mismatch_idx: Optional[Tuple[int, ...]] = None
    first_mismatch_val_candidate: Optional[float] = None
    first_mismatch_val_reference: Optional[float] = None
    first_mismatch_ulp: Optional[int] = None
    first_mismatch_bits_candidate: Optional[str] = None
    first_mismatch_bits_reference: Optional[str] = None

    def __str__(self) -> str:
        if not self.shape_matches or not self.dtype_matches:
            return (
                f"✗ INCOMPATIBLE TENSORS:\n"
                f"  - Shape matches: {self.shape_matches}\n"
                f"  - Dtype matches: {self.dtype_matches}"
            )

        lines = []
        if self.is_equivalent:
            status = "✓ ULP EQUIVALENCE VERIFIED"
            if self.is_exact:
                lines.append(f"{status}: All {self.total_elements:,} elements are 100% bitwise identical (0 ULPs).")
            else:
                lines.append(f"{status} (tolerance: <= {self.max_allowed_ulp} ULPs):")
                lines.append(f"  - Max ULP observed: {self.max_ulp} ULP(s)")
                lines.append(f"  - Mean ULP: {self.mean_ulp:.4f} | P99: {self.p99_ulp:.1f} | P99.9: {self.p999_ulp:.1f}")
        else:
            lines.append(f"✗ ULP EQUIVALENCE FAILED (tolerance: <= {self.max_allowed_ulp} ULPs):")
            lines.append(f"  - Max ULP observed: {self.max_ulp} ULP(s)")
            lines.append(f"  - Mean ULP: {self.mean_ulp:.4f} | P99: {self.p99_ulp:.1f} | P99.9: {self.p999_ulp:.1f}")
            lines.append(
                f"  - Elements exceeding tolerance: {self.mismatched_elements:,} / {self.total_elements:,} "
                f"({self.mismatch_fraction * 100:.4f}%)"
            )

        if self.ulp_counts:
            lines.append("  - ULP Distribution:")
            for key in ["0_ulp", "le_1_ulp", "le_2_ulp", "le_4_ulp", "le_8_ulp", "gt_8_ulp"]:
                if key in self.ulp_counts:
                    label = {
                        "0_ulp": "    0 ULP (bit-exact):",
                        "le_1_ulp": "    <= 1 ULP:         ",
                        "le_2_ulp": "    <= 2 ULPs:        ",
                        "le_4_ulp": "    <= 4 ULPs:        ",
                        "le_8_ulp": "    <= 8 ULPs:        ",
                        "gt_8_ulp": "    > 8 ULPs:         ",
                    }[key]
                    cnt = self.ulp_counts[key]
                    frac = self.ulp_fractions.get(key, 0.0) * 100
                    lines.append(f"{label} {cnt:,} / {self.total_elements:,} ({frac:.4f}%)")

        lines.append(f"  - Max absolute error: {self.max_absolute_error:.6e}")
        lines.append(f"  - Max relative error: {self.max_relative_error:.6e}")

        if not self.is_equivalent and self.first_mismatch_idx is not None:
            lines.extend([
                f"  - First out-of-tolerance element at index {self.first_mismatch_idx}:",
                f"      Candidate: {self.first_mismatch_val_candidate} (bits: {self.first_mismatch_bits_candidate})",
                f"      Reference: {self.first_mismatch_val_reference} (bits: {self.first_mismatch_bits_reference})",
                f"      ULP diff:  {self.first_mismatch_ulp} ULPs",
            ])
        return "\n".join(lines)


# Backward-compatibility aliases
BitExactReport = ULPEquivalenceReport
EquivalenceReport = ULPEquivalenceReport


def _float_to_bits(val: float, dtype: torch.dtype) -> str:
    """Format a scalar float value as hex and binary bit string."""
    if dtype in (torch.bfloat16, torch.float16):
        t = torch.tensor(val, dtype=dtype)
        raw_int = int(t.view(torch.int16).item()) & 0xFFFF
        return f"0x{raw_int:04x} (b{raw_int:016b})"
    elif dtype == torch.float32:
        raw_int = struct.unpack(">I", struct.pack(">f", float(val)))[0]
        return f"0x{raw_int:08x} (b{raw_int:032b})"
    elif dtype == torch.float64:
        raw_int = struct.unpack(">Q", struct.pack(">d", float(val)))[0]
        return f"0x{raw_int:016x} (b{raw_int:064b})"
    return str(val)


def compute_ulp_diff(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Compute the element-wise ULP (Units in the Last Place) distance between two float tensors.

    Supports:
        - torch.bfloat16 (1 sign, 8 exponent, 7 mantissa)
        - torch.float16  (1 sign, 5 exponent, 10 mantissa)
        - torch.float32  (1 sign, 8 exponent, 23 mantissa)
        - torch.float64  (1 sign, 11 exponent, 52 mantissa)

    Returns:
        torch.Tensor of dtype int64 with identical shape containing ULP distances.
    """
    if candidate.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {candidate.shape} vs {reference.shape}")
    if candidate.dtype != reference.dtype:
        raise ValueError(f"Dtype mismatch: {candidate.dtype} vs {reference.dtype}")

    dtype = candidate.dtype
    if dtype in (torch.bfloat16, torch.float16):
        int_type = torch.int16
    elif dtype == torch.float32:
        int_type = torch.int32
    elif dtype == torch.float64:
        int_type = torch.int64
    else:
        raise ValueError(f"Unsupported dtype for ULP computation: {dtype}")

    c = candidate.detach()
    r = reference.detach()

    c_sign = torch.signbit(c)
    r_sign = torch.signbit(r)

    # torch.abs() clears the sign bit, leaving exact magnitude bits
    c_mag = torch.abs(c).view(int_type).to(torch.int64)
    r_mag = torch.abs(r).view(int_type).to(torch.int64)

    same_sign = (c_sign == r_sign)
    diff_same = (c_mag - r_mag).abs()
    diff_diff = c_mag + r_mag
    ulp_diff = torch.where(same_sign, diff_same, diff_diff)

    # Handle NaNs and infinities
    c_nan = torch.isnan(c)
    r_nan = torch.isnan(r)
    both_nan = c_nan & r_nan
    one_nan = c_nan ^ r_nan
    ulp_diff = torch.where(both_nan, torch.zeros_like(ulp_diff), ulp_diff)
    ulp_diff = torch.where(one_nan, torch.full_like(ulp_diff, 10**9), ulp_diff)
    return ulp_diff


def compare_tensors_ulp(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    max_ulp: int = 2,
    max_mismatch_fraction: float = 0.0,
) -> ULPEquivalenceReport:
    """Compare two tensors for numerical equivalence within a specified ULP tolerance.

    Args:
        candidate: Output tensor from candidate kernel.
        reference: Ground truth tensor from reference implementation.
        max_ulp: Maximum allowable difference in Units in the Last Place (default: 2).
        max_mismatch_fraction: Maximum allowed fraction of elements exceeding max_ulp (default: 0.0).

    Returns:
        ULPEquivalenceReport with distribution statistics and discrepancy diagnostics.
    """
    shape_matches = candidate.shape == reference.shape
    dtype_matches = candidate.dtype == reference.dtype

    if not shape_matches or not dtype_matches:
        return ULPEquivalenceReport(
            is_equivalent=False,
            is_exact=False,
            shape_matches=shape_matches,
            dtype_matches=dtype_matches,
            max_ulp=10**9,
            mean_ulp=float("inf"),
            p99_ulp=float("inf"),
            p999_ulp=float("inf"),
            max_allowed_ulp=max_ulp,
            max_absolute_error=float("inf"),
            max_relative_error=float("inf"),
            total_elements=reference.numel(),
            mismatched_elements=candidate.numel(),
            mismatch_fraction=1.0,
        )

    total_elements = candidate.numel()
    if total_elements == 0:
        return ULPEquivalenceReport(
            is_equivalent=True,
            is_exact=True,
            shape_matches=True,
            dtype_matches=True,
            max_ulp=0,
            mean_ulp=0.0,
            p99_ulp=0.0,
            p999_ulp=0.0,
            max_allowed_ulp=max_ulp,
            max_absolute_error=0.0,
            max_relative_error=0.0,
            total_elements=0,
            mismatched_elements=0,
            mismatch_fraction=0.0,
        )

    cand_cpu = candidate.detach().cpu()
    ref_cpu = reference.detach().cpu()

    ulp_diff = compute_ulp_diff(cand_cpu, ref_cpu)
    max_ulp_val = int(ulp_diff.max().item())
    mean_ulp_val = float(ulp_diff.float().mean().item())

    # Quantiles
    ulp_float = ulp_diff.float().flatten()
    p99_val = float(torch.quantile(ulp_float, 0.99).item()) if total_elements > 100 else float(max_ulp_val)
    p999_val = float(torch.quantile(ulp_float, 0.999).item()) if total_elements > 1000 else float(max_ulp_val)

    # Distribution counts
    c0 = int((ulp_diff == 0).sum().item())
    cle1 = int((ulp_diff <= 1).sum().item())
    cle2 = int((ulp_diff <= 2).sum().item())
    cle4 = int((ulp_diff <= 4).sum().item())
    cle8 = int((ulp_diff <= 8).sum().item())
    cgt8 = int((ulp_diff > 8).sum().item())

    ulp_counts = {
        "0_ulp": c0,
        "le_1_ulp": cle1,
        "le_2_ulp": cle2,
        "le_4_ulp": cle4,
        "le_8_ulp": cle8,
        "gt_8_ulp": cgt8,
    }
    ulp_fractions = {k: v / total_elements for k, v in ulp_counts.items()}

    # Floating point error metrics
    c_flt = cand_cpu.float()
    r_flt = ref_cpu.float()
    abs_diff = (c_flt - r_flt).abs()
    max_abs_err = float(abs_diff.max().item())

    denominator = torch.maximum(c_flt.abs(), r_flt.abs())
    rel_diff = torch.where(denominator > 0, abs_diff / denominator, torch.zeros_like(abs_diff))
    max_rel_err = float(rel_diff.max().item())

    # Check against tolerance
    out_of_tol_mask = ulp_diff > max_ulp
    mismatched_elements = int(out_of_tol_mask.sum().item())
    mismatch_fraction = mismatched_elements / total_elements

    is_exact = (max_ulp_val == 0)
    is_equivalent = (mismatch_fraction <= max_mismatch_fraction)

    first_idx = None
    val_cand = None
    val_ref = None
    bits_cand = None
    bits_ref = None
    first_ulp = None

    if mismatched_elements > 0:
        mismatch_indices = torch.nonzero(out_of_tol_mask)
        first_idx = tuple(int(x) for x in mismatch_indices[0])
        val_cand = float(cand_cpu[first_idx].item())
        val_ref = float(ref_cpu[first_idx].item())
        first_ulp = int(ulp_diff[first_idx].item())
        bits_cand = _float_to_bits(val_cand, candidate.dtype)
        bits_ref = _float_to_bits(val_ref, reference.dtype)

    return ULPEquivalenceReport(
        is_equivalent=is_equivalent,
        is_exact=is_exact,
        shape_matches=True,
        dtype_matches=True,
        max_ulp=max_ulp_val,
        mean_ulp=mean_ulp_val,
        p99_ulp=p99_val,
        p999_ulp=p999_val,
        max_allowed_ulp=max_ulp,
        ulp_counts=ulp_counts,
        ulp_fractions=ulp_fractions,
        max_absolute_error=max_abs_err,
        max_relative_error=max_rel_err,
        total_elements=total_elements,
        mismatched_elements=mismatched_elements,
        mismatch_fraction=mismatch_fraction,
        first_mismatch_idx=first_idx,
        first_mismatch_val_candidate=val_cand,
        first_mismatch_val_reference=val_ref,
        first_mismatch_ulp=first_ulp,
        first_mismatch_bits_candidate=bits_cand,
        first_mismatch_bits_reference=bits_ref,
    )


def assert_close_ulp(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    max_ulp: int = 2,
    max_mismatch_fraction: float = 0.0,
    message: str = "Candidate kernel output is not within allowed ULPs of reference",
) -> None:
    """Assert that candidate tensor is numerically equivalent to reference within max_ulp."""
    report = compare_tensors_ulp(
        candidate,
        reference,
        max_ulp=max_ulp,
        max_mismatch_fraction=max_mismatch_fraction,
    )
    if not report.is_equivalent:
        raise AssertionError(f"{message}\n{report}")


# Equivalence aliases
compare_equivalence = compare_tensors_ulp
assert_equivalent = assert_close_ulp


def compare_tensors_bit_exact(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> ULPEquivalenceReport:
    """Compare two tensors for 100% bit-for-bit exactness (0 ULPs)."""
    return compare_tensors_ulp(candidate, reference, max_ulp=0, max_mismatch_fraction=0.0)


def assert_bit_exact(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    message: str = "Candidate kernel output is not bit-exact to reference",
) -> None:
    """Assert that candidate tensor is bit-exact to reference (0 ULPs difference)."""
    assert_close_ulp(candidate, reference, max_ulp=0, max_mismatch_fraction=0.0, message=message)


def create_test_inputs(
    batch: int = 1,
    num_heads: int = 24,
    head_dim: int = 128,
    q_frames: int = 3,
    kv_frames: int = 15,
    h_patches: int = 15,
    w_patches: int = 27,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    rope_dtype: str = "float64",
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate realistic inputs matching Stage2 chunk geometry (e.g. chunk 4).

    Defaults:
        - Q: 3 frames * 405 tokens = 1215 tokens
        - K/V: 15 frames * 405 tokens = 6075 tokens
        - Heads: 24, Head Dim: 128
    """
    torch.manual_seed(seed)
    tokens_per_frame = h_patches * w_patches  # 405
    lq = q_frames * tokens_per_frame  # 1215
    lkv = kv_frames * tokens_per_frame  # 6075

    q = torch.randn(batch, lq, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, lkv, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, lkv, num_heads, head_dim, dtype=dtype, device=device)

    q_grid = torch.tensor([[q_frames, h_patches, w_patches]], dtype=torch.long, device=device)
    k_grid = torch.tensor([[kv_frames, h_patches, w_patches]], dtype=torch.long, device=device)

    _, c_dt = normalize_rope_precision(rope_dtype)
    freqs = build_wan22_freqs(head_dim=head_dim, max_len=1024, complex_dtype=c_dt).to(device)

    # Realistic camera matrices
    # Identity view matrices with subtle translations
    q_viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(batch, q_frames, 1, 1)
    q_viewmats[..., 0, 3] = torch.linspace(0.1, 0.3, q_frames, device=device)

    kv_viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(batch, kv_frames, 1, 1)
    kv_viewmats[..., 0, 3] = torch.linspace(0.0, 0.3, kv_frames, device=device)

    # Realistic intrinsic matrices
    q_Ks = torch.eye(3, device=device).unsqueeze(0).repeat(batch, q_frames, 1, 1)
    q_Ks[..., 0, 0] = 500.0 / 864.0  # fx normalized
    q_Ks[..., 1, 1] = 500.0 / 480.0  # fy normalized

    kv_Ks = torch.eye(3, device=device).unsqueeze(0).repeat(batch, kv_frames, 1, 1)
    kv_Ks[..., 0, 0] = 500.0 / 864.0
    kv_Ks[..., 1, 1] = 500.0 / 480.0

    q_start_frame = kv_frames - q_frames  # e.g. 15 - 3 = 12
    k_start_frame = 0

    return {
        "q": q,
        "k": k,
        "v": v,
        "freqs": freqs,
        "q_grid_sizes": q_grid,
        "k_grid_sizes": k_grid,
        "q_viewmats": q_viewmats,
        "q_Ks": q_Ks,
        "kv_viewmats": kv_viewmats,
        "kv_Ks": kv_Ks,
        "q_start_frame": q_start_frame,
        "k_start_frame": k_start_frame,
        "rope_dtype": rope_dtype,
        "camera_translation_transform": "linear",
    }
