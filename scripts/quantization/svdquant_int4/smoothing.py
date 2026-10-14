"""Input activation outlier smoothing for Wan2.2 SVDQuant.

Computes channel-wise smoothing scales:
    s_i = (max|X_i|)^alpha / (max|W_{:, i}|)^alpha
Such that:
    X' = X * diag(1 / s)
    W' = W * diag(s)
and X' @ W'^T == X @ W^T.
"""

from __future__ import annotations

import torch


def compute_smoothing_scale(
    act_max: torch.Tensor,
    weight_max: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-5,
    clamp_min: float = 1e-2,
    clamp_max: float = 1e2,
) -> torch.Tensor:
    """Compute per-channel smoothing scale s_i for a single linear layer.

    Args:
        act_max: [IC] maximum absolute activation magnitude per channel
        weight_max: [IC] maximum absolute weight magnitude per channel
        alpha: outlier migration strength (default 0.5)
        eps: floor guard against division by zero
        clamp_min: minimum allowed smoothing scale
        clamp_max: maximum allowed smoothing scale
    Returns:
        s: [IC] smoothing scale in bfloat16
    """
    a = act_max.float().clamp(min=eps)
    w = weight_max.float().clamp(min=eps)
    scale = (a.pow(alpha) / w.pow(1.0 - alpha)).clamp(min=clamp_min, max=clamp_max)
    return scale.to(torch.bfloat16)


def compute_joint_smoothing_scales(
    act_max: torch.Tensor,
    weights: list[torch.Tensor],
    alpha: float = 0.5,
    eps: float = 1e-5,
    clamp_min: float = 1e-2,
    clamp_max: float = 1e2,
) -> torch.Tensor:
    """Compute a single joint smoothing scale s across multiple layers sharing the same input X.

    In Wan2.2 dense blocks, self_attn.q, self_attn.k, self_attn.v and ffn.0
    all consume the same input tensor x. They must therefore share identical
    smoothing factors s to avoid de-fusing the prologue kernel.

    Args:
        act_max: [IC] maximum absolute activation magnitude across tokens
        weights: list of [OC_l, IC] weight tensors for layers sharing this input
    """
    ic = act_max.shape[0]
    w_max_combined = torch.zeros(ic, dtype=torch.float32, device=act_max.device)
    for w in weights:
        w_max = w.float().abs().amax(dim=0)
        w_max_combined = torch.maximum(w_max_combined, w_max)

    return compute_smoothing_scale(
        act_max, w_max_combined, alpha=alpha, eps=eps, clamp_min=clamp_min, clamp_max=clamp_max
    )
