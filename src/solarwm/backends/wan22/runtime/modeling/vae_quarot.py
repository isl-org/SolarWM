"""Block-256 QuaRot (Hadamard-rotated) W8A8 dynamic quantization for Wan2.2 VAE decoder.

Provides data-free, calibration-free dynamic INT8 quantization for the primary 3D convolutions
in the Wan2.2 VAE decoder on Intel Arc GPUs (XPU) using native oneDNN qconv_pointwise.tensor.

Key mathematical properties:
1. Sylvester-Hadamard orthogonal rotation matrix H_256 (H H^T = I).
2. Weights are pre-rotated offline along input channels: W' = R_{IC}^T W.
3. Activations are rotated at runtime along channels: X' = X R_{IC}.
4. Output is mathematically invariant: Y = X' W' = X (R R^T) W = X W in standard unrotated coordinates.
5. Preserves exact zero padding and temporal causal cache (feat_cache) continuity.
"""

from __future__ import annotations

import logging
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import CausalConv3d

logger = logging.getLogger(__name__)

_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def get_hadamard_matrix(
    n: int = 256,
    device: torch.device | str = "xpu",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generate or retrieve a normalized Sylvester-Hadamard orthogonal matrix of size n.

    H is symmetric (H = H^T) and orthogonal (H @ H = I).
    """
    dev = torch.device(device)
    key = (n, dev, dtype)
    if key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[key]

    if (n & (n - 1)) != 0 or n <= 0:
        raise ValueError(f"Hadamard order n must be a positive power of 2, got {n}")

    # Recursive Sylvester construction on CPU in float32 for precision
    h = torch.tensor([[1.0]], dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)

    # Normalize by 1 / sqrt(n) so H @ H.T = I
    h = h / (float(n) ** 0.5)
    h_out = h.to(device=dev, dtype=dtype)
    _HADAMARD_CACHE[key] = h_out
    return h_out


def rotate_channels_last(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Rotate channels in blocks of H.shape[0] (256) for 4D or 5D channels-first tensors.

    Parameters:
        x: Input tensor of shape [N, C, T, H, W] or [N, C, H, W].
           Memory format is expected to be channels_last_3d or channels_last.
        H: Orthogonal rotation matrix of shape [256, 256].

    Returns:
        Rotated tensor with the same shape and memory format.
    """
    orig_shape = x.shape
    C = orig_shape[1]
    block_size = H.shape[0]

    if C % block_size != 0:
        raise ValueError(
            f"Channel count C={C} must be an integer multiple of block size {block_size}"
        )

    if x.ndim == 5:
        N, _, T, H_dim, W_dim = orig_shape
        # Permute channels to the innermost dimension: [N, T, H, W, C]
        # In channels_last_3d layout, this view is physically contiguous in memory.
        x_last = x.permute(0, 2, 3, 4, 1)
        x_flat = x_last.reshape(-1, block_size)
        x_rot = x_flat @ H
        x_rot_mc = x_rot.view(N, T, H_dim, W_dim, C)
        return x_rot_mc.permute(0, 4, 1, 2, 3).contiguous(
            memory_format=torch.channels_last_3d
        )
    elif x.ndim == 4:
        N, _, H_dim, W_dim = orig_shape
        x_last = x.permute(0, 2, 3, 1)
        x_flat = x_last.reshape(-1, block_size)
        x_rot = x_flat @ H
        x_rot_mc = x_rot.view(N, H_dim, W_dim, C)
        return x_rot_mc.permute(0, 3, 1, 2).contiguous(
            memory_format=torch.channels_last
        )
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got ndim={x.ndim}")


def _eager_quantize_mc(x_rot_mc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scalar dynamic absmax scale and quantize 2D [M, C] activation matrix to int8."""
    # Two-stage reduction: spatial/temporal reduction first, then channel reduction
    x_absmax = x_rot_mc.abs().amax(dim=0).amax()
    x_scale = (x_absmax / 127.0).clamp(min=1e-8)
    qx_mc = (x_rot_mc / x_scale).round().clamp(-127, 127).to(torch.int8)
    return qx_mc, x_scale


# Compiled dynamic-shape quantizer for Intel XPU Inductor
try:
    _compiled_quantize_mc = torch.compile(_eager_quantize_mc, dynamic=True)
except Exception:
    _compiled_quantize_mc = _eager_quantize_mc


def quantize_conv_weight_quarot(
    conv: nn.Conv3d | CausalConv3d,
    H: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-rotate weights along input channels and compute symmetric per-OC INT8 weights and scales.

    Returns:
        (qw, w_scale, zero_w)
    """
    w = conv.weight.data
    OC, IC = w.shape[0], w.shape[1]
    block_size = H.shape[0]

    if IC % block_size != 0:
        raise ValueError(
            f"Input channels IC={IC} must be a multiple of block size {block_size}"
        )

    device = w.device
    dtype = w.dtype

    # Permute IC to last dimension: [OC, kT, kH, kW, IC]
    w_perm = w.permute(0, 2, 3, 4, 1)
    w_flat = w_perm.reshape(-1, block_size)
    # W' = W @ H
    w_rot_flat = w_flat @ H
    w_rot = w_rot_flat.view(OC, w.shape[2], w.shape[3], w.shape[4], IC).permute(
        0, 4, 1, 2, 3
    )

    # Per-OC symmetric scale: max(|w|) / 127
    w_absmax = w_rot.abs().amax(dim=(1, 2, 3, 4))
    w_scale = (w_absmax / 127.0).clamp(min=1e-8).to(dtype=torch.float32)

    # Quantize to int8
    qw = (
        (w_rot / w_scale[:, None, None, None, None])
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
        .contiguous(memory_format=torch.channels_last_3d)
    )

    zero_w = torch.zeros((OC,), dtype=torch.int32, device=device)
    return qw, w_scale, zero_w


@torch.compiler.disable
def _quarot_causal_conv3d_forward(
    self: CausalConv3d,
    x: torch.Tensor,
    cache_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Patched dynamic INT8 forward for CausalConv3d with Block-256 QuaRot."""
    if not getattr(self, "_quarot_enabled", False):
        orig_fwd = getattr(self, "_orig_forward", None)
        if orig_fwd is not None:
            return orig_fwd(x, cache_x=cache_x)
        return super(CausalConv3d, self).forward(x)

    target_pad_t = getattr(self, "_quarot_target_pad_t", self._padding[4])
    spatial_pad = getattr(
        self, "_quarot_spatial_pad", [0, self._padding[2], self._padding[0]]
    )

    if cache_x is not None and target_pad_t > 0:
        cache_x = cache_x.to(x.device)
        if cache_x.shape[2] == 2 and x.shape[2] == 1:
            b, c, _, h_in, w_in = x.shape
            buf = getattr(self, "_fast_cache_buffer", None)
            if (
                buf is None
                or buf.shape != (b, c, 3, h_in, w_in)
                or buf.device != x.device
                or buf.dtype != x.dtype
            ):
                buf = torch.empty(
                    (b, c, 3, h_in, w_in),
                    device=x.device,
                    dtype=x.dtype,
                    memory_format=torch.channels_last_3d,
                )
                self._fast_cache_buffer = buf
            buf[:, :, :2, :, :].copy_(cache_x)
            buf[:, :, 2:3, :, :].copy_(x)
            x = buf
        else:
            x = torch.cat([cache_x, x], dim=2)
        pad_t = max(0, target_pad_t - cache_x.shape[2])
    else:
        pad_t = target_pad_t

    # Spatial padding is handled in hardware by oneDNN qconv; only pad temporal dimension if needed
    if pad_t > 0:
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))

    N, C, T, H_dim, W_dim = x.shape
    block_size = self._quarot_H.shape[0]

    # Fast in-place rotated activation flattening:
    # x is channels_last_3d, so permute to [N, T, H, W, C] is contiguous in memory.
    x_last = x.permute(0, 2, 3, 4, 1)
    x_flat = x_last.reshape(-1, block_size)
    x_rot_flat = x_flat @ self._quarot_H
    x_rot_mc = x_rot_flat.view(-1, C)

    # Dynamic scalar scale & INT8 quantization
    quant_fn = getattr(self, "_quarot_quant_fn", _eager_quantize_mc)
    qx_mc, x_scale = quant_fn(x_rot_mc)

    # View as 5D channels_last_3d tensor
    qx = (
        qx_mc.view(N, T, H_dim, W_dim, C)
        .permute(0, 4, 1, 2, 3)
        .contiguous(memory_format=torch.channels_last_3d)
    )

    # Execute native oneDNN INT8 qconv with hardware spatial padding
    out = torch.ops.onednn.qconv_pointwise.tensor(
        qx,
        x_scale,
        self._quarot_zero_x,
        self._quarot_qw,
        self._quarot_w_scale,
        self._quarot_zero_w,
        self._quarot_bias_fp32,
        list(self.stride),
        spatial_pad,
        list(self.dilation),
        self.groups,
        1.0,
        0,
        torch.bfloat16,
        "none",
        [],
        "",
    )
    return out


def install_quarot_conv(
    conv: CausalConv3d,
    H: torch.Tensor,
    use_compile: bool = True,
) -> None:
    """Patch a CausalConv3d instance with QuaRot W8A8 dynamic oneDNN execution."""
    if getattr(conv, "_quarot_installed", False):
        conv._quarot_enabled = True
        return

    qw, w_scale, zero_w = quantize_conv_weight_quarot(conv, H)
    device = conv.weight.device

    conv._quarot_H = H
    conv._quarot_qw = qw
    conv._quarot_w_scale = w_scale
    conv._quarot_zero_w = zero_w
    conv._quarot_zero_x = torch.tensor(0, dtype=torch.int32, device=device)
    conv._quarot_bias_fp32 = (
        conv.bias.data.float() if conv.bias is not None else None
    )
    conv._quarot_spatial_pad = [0, conv._padding[2], conv._padding[0]]
    conv._quarot_target_pad_t = conv._padding[4]
    conv._quarot_quant_fn = (
        _compiled_quantize_mc if use_compile else _eager_quantize_mc
    )
    conv._orig_forward = conv.forward
    conv.forward = types.MethodType(_quarot_causal_conv3d_forward, conv)
    conv._quarot_enabled = True
    conv._quarot_installed = True


def uninstall_quarot_conv(conv: CausalConv3d) -> None:
    """Restore the original BF16 forward method of a patched CausalConv3d."""
    if not getattr(conv, "_quarot_installed", False):
        return
    conv._quarot_enabled = False
    orig_fwd = getattr(conv, "_orig_forward", None)
    if orig_fwd is not None:
        conv.forward = orig_fwd
        delattr(conv, "_orig_forward")
    conv._quarot_installed = False


def is_quarot_candidate(name: str, module: nn.Module) -> bool:
    """Return True if the module is one of the 28 primary 3x3x3 3D convolutions in ResidualBlock."""
    if not isinstance(module, CausalConv3d):
        return False
    if getattr(module, "kernel_size", None) != (3, 3, 3):
        return False
    if module.in_channels % 256 != 0:
        return False
    # Only quantize residual.2 and residual.6 inside ResidualBlock
    if not ("residual.2" in name or "residual.6" in name):
        return False
    if "head" in name or "conv1" in name:
        return False
    return True


def apply_vae_quarot(
    decoder: nn.Module,
    enabled: bool = True,
    use_compile: bool = True,
    block_size: int = 256,
) -> list[str]:
    """Enable or disable Block-256 QuaRot W8A8 dynamic quantization on the VAE decoder.

    Parameters:
        decoder: The Decoder3d instance from WanVAE_.
        enabled: If True, install dynamic INT8 execution; if False, restore BF16.
        use_compile: If True, use torch.compile on the dynamic activation quantizer.
        block_size: Channel block size for orthogonal rotation (default 256).

    Returns:
        List of layer names affected.
    """
    affected_layers: list[str] = []
    candidates: list[tuple[str, CausalConv3d]] = []

    for name, module in decoder.named_modules():
        if is_quarot_candidate(name, module):
            candidates.append((name, module))

    if not candidates:
        logger.warning("No QuaRot candidate convolutions found in decoder.")
        return affected_layers

    if not enabled:
        for name, conv in candidates:
            uninstall_quarot_conv(conv)
            affected_layers.append(name)
        logger.info("Uninstalled QuaRot W8A8 on %d decoder layers.", len(affected_layers))
        return affected_layers

    # Determine device and dtype from first candidate
    first_conv = candidates[0][1]
    device = first_conv.weight.device
    dtype = first_conv.weight.dtype

    if device.type != "xpu":
        logger.warning(
            "QuaRot W8A8 dynamic oneDNN execution is only supported on Intel XPU (got %s). Skipping.",
            device.type,
        )
        return affected_layers

    H = get_hadamard_matrix(n=block_size, device=device, dtype=dtype)

    for name, conv in candidates:
        install_quarot_conv(conv, H, use_compile=use_compile)
        affected_layers.append(name)

    logger.info(
        "Successfully applied Block-%d QuaRot W8A8 dynamic quantization on %d decoder layers.",
        block_size,
        len(affected_layers),
    )
    return affected_layers
