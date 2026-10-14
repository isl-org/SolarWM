"""Low-rank SVD extraction on smoothed weights for SVDQuant."""

from __future__ import annotations

import torch


def extract_low_rank_svd(
    weight_smoothed: torch.Tensor,
    smooth_scale: torch.Tensor,
    rank: int = 32,
    niter: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform rank-r SVD decomposition on smoothed weights and fold smoothing into lora_a.

    Args:
        weight_smoothed: [OC, IC] smoothed weight matrix W' = W * diag(s)
        smooth_scale: [IC] per-channel smoothing scale s
        rank: low-rank branch dimension (32 for B70 kernel)
        niter: randomized SVD power iterations

    Returns:
        lora_a: [rank, IC] BF16 matrix consuming raw (unsmoothed) activations
        lora_b: [OC, rank] BF16 matrix producing layer output
        residual: [OC, IC] BF16 matrix to be quantized with GPTQ/RTN (acts on smoothed input X')
    """
    oc, ic = weight_smoothed.shape
    w_fp = weight_smoothed.float()
    q = min(rank + 16, min(oc, ic))

    # Fast randomized SVD low-rank
    u, s, v = torch.svd_lowrank(w_fp, q=q, niter=niter)

    u_r = u[:, :rank]           # [OC, rank]
    s_r = s[:rank]              # [rank]
    v_r = v[:, :rank]           # [IC, rank]

    # lora_b absorbs singular values
    lora_b = (u_r * s_r.unsqueeze(0)).to(torch.bfloat16)  # [OC, rank]
    down = v_r.T.to(torch.bfloat16)                       # [rank, IC]

    # lora_a folds (1 / s) so that X @ lora_a.T == (X / s) @ down.T
    # This allows the SYCL x-prologue to compute X @ lora_a.T concurrently with quantization!
    s_inv = (1.0 / smooth_scale.float().clamp(min=1e-8)).unsqueeze(0)  # [1, IC]
    lora_a = (down.float() * s_inv).to(torch.bfloat16)                 # [rank, IC]

    # Residual matrix: R = W' - lora_b @ down
    lora_recon = lora_b.float() @ down.float()
    residual = (w_fp - lora_recon).to(torch.bfloat16)

    return lora_a, lora_b, residual
