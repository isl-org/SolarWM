"""SVDQuant linear module and portable pseudo-QDQ reference for SolarWM Wan2.2 Stage2."""

from __future__ import annotations

import torch
import torch.nn as nn

from solarwm.kernels.dit_svdquant.qdq_utils import (
    convert_int4_weights_to_dpas_b,
    convert_w_scales_to_dpas,
    pseudo_qdq_linear,
)
from solarwm.kernels.dit_svdquant.triton_w4a4_gemm import triton_svdquant_w4a4_gemm


class SVDQuantLinear(nn.Module):
    """Inference-only linear layer executing SVDQuant W4A4 + BF16 Low-Rank residual branch.

    Supports:
    - INT4 packed weights [out_features, in_features // 2]
    - BF16/FP16 scales or E8M0 power-of-2 scales (uint8)
    - Channel smoothing [in_features]
    - Rank-16 BF16 LoRA branch (lora_a, lora_b)
    - Epilogue activation ('none', 'gelu_tanh', 'silu', 'gelu')
    - Optional fused INT4 output quantization for chained execution
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 64,
        rank: int = 16,
        has_bias: bool = False,
        has_smooth: bool = True,
        activation: str = "none",
        use_e8m0_scale: bool = False,
        use_triton_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.rank = rank
        self.activation = activation
        self.use_triton_kernel = use_triton_kernel
        self.use_e8m0_scale = use_e8m0_scale

        scale_dtype = torch.uint8 if use_e8m0_scale else torch.bfloat16
        self.register_buffer(
            "qweight",
            torch.zeros((out_features, in_features // 2), dtype=torch.int8),
        )
        # Generated once by _load_from_state_dict.  The SYCL XMX kernel block-loads
        # this directly into its INT4 DPAS B operand registers.
        self.register_buffer("dpas_b", None, persistent=False)
        self.register_buffer("dpas_w_scale", None, persistent=False)
        self.register_buffer(
            "w_scale",
            torch.ones((out_features, in_features // group_size), dtype=scale_dtype),
        )

        if has_smooth:
            self.register_buffer("smooth_scale", torch.ones((in_features,), dtype=torch.bfloat16))
        else:
            self.register_buffer("smooth_scale", None)

        if has_bias:
            self.register_buffer("bias", torch.zeros((out_features,), dtype=torch.bfloat16))
        else:
            self.register_buffer("bias", None)

        if rank > 0:
            self.register_buffer("lora_a", torch.zeros((rank, in_features), dtype=torch.bfloat16))
            self.register_buffer("lora_b", torch.zeros((out_features, rank), dtype=torch.bfloat16))
        else:
            self.register_buffer("lora_a", None)
            self.register_buffer("lora_b", None)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Prepack the DPAS B operand and its scales once after the weights are loaded."""
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self.dpas_b = convert_int4_weights_to_dpas_b(self.qweight)
        self.dpas_w_scale = convert_w_scales_to_dpas(self.w_scale)

    def forward(
        self,
        x: torch.Tensor,
        fuse_quant_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_triton_kernel and x.is_xpu:
            return triton_svdquant_w4a4_gemm(
                x=x,
                w_packed=self.qweight,
                w_scale=self.w_scale,
                s_in=self.smooth_scale,
                bias=self.bias,
                lora_a=self.lora_a,
                lora_b=self.lora_b,
                group_size=self.group_size,
                activation=self.activation,
                fuse_quant_out=fuse_quant_out,
            )
        return pseudo_qdq_linear(
            x=x,
            w_packed=self.qweight,
            w_scale=self.w_scale,
            s_in=self.smooth_scale,
            bias=self.bias,
            lora_a=self.lora_a,
            lora_b=self.lora_b,
            group_size=self.group_size,
            activation=self.activation,
            fuse_quant_out=fuse_quant_out,
        )
