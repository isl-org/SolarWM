"""Benchmark comparing:
1. Our Triton SVDQuant kernel (fused INT4 + BF16 LoRA + GELU_tanh epilogue).
2. llm-scaler's omni_xpu_kernel SVDQuant (oneDNN u4 matmul, append_sum, ESIMD kernels).
3. PyTorch native oneDNN BF16 Linear baseline (+ GELU_tanh).
"""

from __future__ import annotations

import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ssheorey/code/SolarWM-quant/.venv-xpu-quant/lib/python3.12/site-packages")
import omni_xpu_kernel
from omni_xpu_kernel import svdq

from solarwm.kernels.dit_svdquant.qdq_utils import pack_int4_signed
from solarwm.kernels.dit_svdquant.triton_w4a4_gemm import triton_svdquant_w4a4_gemm


def sync():
    torch.xpu.synchronize()


def time_fn(fn, warmup=15, runs=50):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    sync()
    t1 = time.perf_counter()
    return (t1 - t0) / runs * 1000.0  # ms


def benchmark_layer(M: int, K: int, N: int, name: str, with_gelu: bool = False):
    print(f"\n=======================================================")
    print(f"Layer: {name} (M={M}, K={K}, N={N}, with_gelu={with_gelu})")
    print(f"=======================================================")

    device = "xpu"
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    x_f16 = x.to(torch.float16)

    # Weights
    w_raw = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=device)
    w_packed = pack_int4_signed(w_raw)
    w_scale = torch.rand(N, K // 64, device=device, dtype=torch.bfloat16) * 0.1
    s_in = torch.rand(K, device=device, dtype=torch.bfloat16) + 0.5
    bias = torch.randn(N, device=device, dtype=torch.bfloat16)

    lora_a = torch.randn(16, K, device=device, dtype=torch.bfloat16) * 0.01
    lora_b = torch.randn(N, 16, device=device, dtype=torch.bfloat16) * 0.01

    # Weights for PyTorch BF16 baseline
    w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    # Weights prepared for llm-scaler oneDNN u4 GEMM
    # Note: omni_xpu_kernel takes scales of shape [K//64, N]
    w_scale_trans = w_scale.t().contiguous()
    packed_u4, scales_f16 = svdq.prepare_onednn_weights(w_packed.view(torch.uint8), w_scale_trans)

    # -----------------------------------------------------------
    # 1. Baseline: PyTorch Native BF16 Linear
    # -----------------------------------------------------------
    if with_gelu:
        def pt_bf16():
            out = F.linear(x, w_bf16, bias)
            return F.gelu(out, approximate="tanh")
    else:
        def pt_bf16():
            return F.linear(x, w_bf16, bias)

    ms_pt_bf16 = time_fn(pt_bf16)
    print(f"  [1] PyTorch Native BF16 Linear:                 {ms_pt_bf16:6.3f} ms (1.00x Ref)")

    # -----------------------------------------------------------
    # 2. llm-scaler: oneDNN u4 GEMM (preconverted)
    # -----------------------------------------------------------
    def llm_scaler_onednn_gemm():
        return svdq.onednn_int4_gemm_preconverted(x_f16, packed_u4, scales_f16)

    ms_onednn_gemm = time_fn(llm_scaler_onednn_gemm)
    print(f"  [2] llm-scaler oneDNN u4 GEMM (core matmul):     {ms_onednn_gemm:6.3f} ms ({ms_pt_bf16 / ms_onednn_gemm:4.2f}x)")

    # -----------------------------------------------------------
    # 3. llm-scaler: oneDNN u4 GEMM + append_sum (LoRA + bias prefill)
    # -----------------------------------------------------------
    dst = torch.empty((M, N), device=device, dtype=torch.bfloat16)
    x_s = x / s_in

    def llm_scaler_fused_sum():
        # Pre-fill dst with LoRA + bias
        x_lora = torch.matmul(x_s, lora_a.t())
        torch.matmul(x_lora, lora_b.t(), out=dst)
        dst.add_(bias)
        # Fused oneDNN u4 GEMM accumulate into dst
        svdq.onednn_int4_gemm_add_to_output(x_f16, packed_u4, scales_f16, dst)
        if with_gelu:
            return F.gelu(dst, approximate="tanh")
        return dst

    ms_llm_scaler_sum = time_fn(llm_scaler_fused_sum)
    print(f"  [3] llm-scaler oneDNN u4 + append_sum (fused):   {ms_llm_scaler_sum:6.3f} ms ({ms_pt_bf16 / ms_llm_scaler_sum:4.2f}x)")

    # -----------------------------------------------------------
    # 4. Our Triton SVDQuant kernel (fused INT4 + BF16 LoRA + GELU)
    # -----------------------------------------------------------
    act_str = "gelu_tanh" if with_gelu else "none"

    def our_triton():
        return triton_svdquant_w4a4_gemm(
            x=x,
            w_packed=w_packed,
            w_scale=w_scale,
            s_in=s_in,
            bias=bias,
            lora_a=lora_a,
            lora_b=lora_b,
            group_size=64,
            activation=act_str,
        )

    ms_our_triton = time_fn(our_triton)
    print(f"  [4] Our Triton SVDQuant (fused INT4+LoRA+Epil): {ms_our_triton:6.3f} ms ({ms_pt_bf16 / ms_our_triton:4.2f}x)")


def main():
    print("=" * 60)
    print("SVDQUANT KERNEL BENCHMARK ON INTEL ARC B70 / B580 (Xe2)")
    print("=" * 60)

    # 1. Attention Projection: M=1215, K=3072, N=3072
    benchmark_layer(1215, 3072, 3072, "Self/Cross-Attention Proj (Q/K/V/O)", with_gelu=False)

    # 2. FFN Up-projection + GELU: M=1215, K=3072, N=14336
    benchmark_layer(1215, 3072, 14336, "FFN 0 (3072 -> 14336) + GELU_tanh", with_gelu=True)

    # 3. FFN Down-projection: M=1215, K=14336, N=3072
    benchmark_layer(1215, 14336, 3072, "FFN 2 (14336 -> 3072)", with_gelu=False)


if __name__ == "__main__":
    main()
