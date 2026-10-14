"""Symmetric INT4 group-64 GPTQ residual quantizer."""

from __future__ import annotations

import torch

from .quantizer import pack_int4_signed


def gptq_quantize_residual(
    residual: torch.Tensor,
    hessian: torch.Tensor,
    smooth_scale: torch.Tensor | None = None,
    group_size: int = 64,
    damping: float = 0.01,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize the SVD low-rank residual matrix using GPTQ for INT4 group-64.

    Args:
        residual: [OC, IC] residual matrix R (operates on smoothed activations)
        hessian: [IC, IC] activation covariance matrix H = X^T X
        smooth_scale: [IC] smoothing factor s (if given, H' = diag(1/s) @ H @ diag(1/s))
        group_size: quantization group size (64)
        damping: diagonal damping factor
        block_size: column block update size (128)

    Returns:
        qweight: [OC, IC // 2] int8 packed weights (low nibble first)
        w_scale: [OC, IC // group_size] bfloat16 scales
        w_quant_dequant: [OC, IC] reconstructed residual in bfloat16
    """
    oc, ic = residual.shape
    device = residual.device
    dtype = residual.dtype

    assert ic % group_size == 0, f"IC ({ic}) must be divisible by group_size ({group_size})"
    assert hessian.shape == (ic, ic), f"Hessian shape {hessian.shape} must match (IC, IC) = ({ic}, {ic})"

    # 1. Transform Hessian to smoothed space if smooth_scale provided:
    # Since X' = X * diag(1 / s), H' = X'^T X' = diag(1/s) @ H @ diag(1/s)
    h = hessian.float().clone()
    if smooth_scale is not None:
        s_inv = (1.0 / smooth_scale.float().clamp(min=1e-8)).to(device)
        h = h * s_inv.unsqueeze(0) * s_inv.unsqueeze(1)

    # 2. Add diagonal damping and zero-handling
    diag_idx = torch.arange(ic, device=device)
    diag = h[diag_idx, diag_idx]
    dead = diag <= 1e-7
    h[dead, dead] = 1.0
    mean_diag = torch.mean(h[diag_idx, diag_idx]).clamp(min=1e-6)

    # 3. Robust inversion of Hessian using Cholesky with adaptive damping
    h_inv_chol = None
    curr_damp = damping * mean_diag
    for attempt in range(6):
        h_try = h.clone()
        h_try[diag_idx, diag_idx] += curr_damp
        try:
            chol = torch.linalg.cholesky(h_try)
            h_inv = torch.cholesky_inverse(chol)
            h_inv_chol = torch.linalg.cholesky(h_inv, upper=True)
            break
        except torch.linalg.LinAlgError:
            curr_damp = max(curr_damp * 10.0, 1e-3 * mean_diag * (10.0 ** attempt))

    if h_inv_chol is None:
        try:
            h_inv = torch.linalg.pinv(h + 0.1 * mean_diag * torch.eye(ic, device=device))
            h_inv = 0.5 * (h_inv + h_inv.T)
            vals, vecs = torch.linalg.eigh(h_inv)
            vals = vals.clamp(min=1e-8)
            h_inv_pos = (vecs * vals.unsqueeze(0)) @ vecs.T
            h_inv_chol = torch.linalg.cholesky(h_inv_pos, upper=True)
        except Exception:
            # Safe fallback: identity (reduces GPTQ to exact RTN INT4)
            h_inv_chol = torch.eye(ic, device=device, dtype=torch.float32)

    # 4. Compute static group scales across residual
    w = residual.float().clone()
    w_grouped = w.view(oc, ic // group_size, group_size)
    amax = w_grouped.abs().amax(dim=-1).clamp(min=1e-8)
    scales = (amax / 7.0).to(torch.bfloat16).float()  # [OC, num_groups]

    q_matrix = torch.zeros((oc, ic), dtype=torch.int8, device=device)
    w_recon = torch.zeros((oc, ic), dtype=torch.float32, device=device)

    # 5. Block-wise GPTQ column updates
    for i in range(0, ic, block_size):
        end_idx = min(i + block_size, ic)
        count = end_idx - i

        w_block = w[:, i:end_idx].clone()
        err_block = torch.zeros_like(w_block)
        h_inv_chol_block = h_inv_chol[i:end_idx, i:end_idx]

        for j in range(count):
            col_idx = i + j
            group_idx = col_idx // group_size
            col_scale = scales[:, group_idx]  # [OC]

            # Current column weights
            w_col = w_block[:, j]
            # Quantize to INT4 [-8, 7]
            q_col = torch.round(w_col / col_scale).clamp(-8, 7).to(torch.int8)
            q_matrix[:, col_idx] = q_col

            w_hat = q_col.float() * col_scale
            w_recon[:, col_idx] = w_hat

            err = w_col - w_hat
            err_block[:, j] = err

            # Intra-block column update
            diag_val = h_inv_chol_block[j, j]
            if j < count - 1 and diag_val.abs() > 1e-12:
                step = h_inv_chol_block[j, j + 1 :] / diag_val
                w_block[:, j + 1 :] -= err.unsqueeze(1) * step.unsqueeze(0)

        # Inter-block update of subsequent columns
        if end_idx < ic:
            # err_block @ (U_blk^-1 @ U_rem)
            u_rem = h_inv_chol[i:end_idx, end_idx:]
            step_rem = torch.linalg.solve_triangular(
                h_inv_chol_block, u_rem, upper=True, left=True
            )
            w[:, end_idx:] -= err_block @ step_rem

    qweight = pack_int4_signed(q_matrix)
    w_scale = scales.to(torch.bfloat16)
    w_quant_dequant = w_recon.to(dtype)

    return qweight, w_scale, w_quant_dequant
