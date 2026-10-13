"""Activation clipping ratio policy search per layer and timestep."""

from __future__ import annotations

import torch

from .quantizer import quantize_activations_dynamic


def search_act_clip_ratio(
    act_sample: torch.Tensor,
    weight_orig: torch.Tensor,
    bias: torch.Tensor | None,
    residual_recon: torch.Tensor,
    smooth_scale: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    candidates: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 1.0),
    group_size: int = 64,
) -> tuple[float, dict[float, float]]:
    """Find the optimal activation clipping ratio rho minimizing end-to-end layer reconstruction error.

    Args:
        act_sample: [tokens, IC] subsample of activations for this layer and timestep bin
        weight_orig: [OC, IC] original BF16 weight
        bias: [OC] bias or None
        residual_recon: [OC, IC] reconstructed residual weight hat(R) from GPTQ/RTN
        smooth_scale: [IC] smoothing scale s
        lora_a: [rank, IC] folded LoRA down projection (consumes raw X)
        lora_b: [OC, rank] LoRA up projection
        candidates: clipping ratios to evaluate (default [0.6, 0.7, 0.8, 0.9, 1.0])
        group_size: activation group size (64)

    Returns:
        best_rho: optimal clipping ratio
        errors: dict mapping rho -> relative MSE
    """
    x = act_sample.to(device=weight_orig.device, dtype=torch.float32)
    if x.shape[0] > 512:
        x = x[:512]
    w = weight_orig.float()
    b = bias.float() if bias is not None else 0.0

    # Ground truth reference output
    y_true = x @ w.T + b
    y_norm_sq = torch.sum(y_true**2).clamp(min=1e-8)

    # Low-rank branch consumes raw X
    la = lora_a.to(device=weight_orig.device, dtype=torch.float32)
    lb = lora_b.to(device=weight_orig.device, dtype=torch.float32)
    lora_out = (x @ la.T) @ lb.T

    # Smoothed input for quantized residual branch
    s = smooth_scale.to(device=weight_orig.device, dtype=torch.float32)
    s_inv = 1.0 / s.clamp(min=1e-8)
    x_smooth = x * s_inv.unsqueeze(0)
    r_hat = residual_recon.to(device=weight_orig.device, dtype=torch.float32)

    best_rho = 1.0
    best_err = float("inf")
    errors: dict[float, float] = {}

    for rho in candidates:
        x_qdq = quantize_activations_dynamic(x_smooth, group_size=group_size, clip_ratio=rho)
        y_pred = x_qdq @ r_hat.T + lora_out + b

        rel_err = (torch.sum((y_true - y_pred) ** 2) / y_norm_sq).item()
        errors[rho] = rel_err

        if rel_err < best_err:
            best_err = rel_err
            best_rho = rho

    return best_rho, errors
