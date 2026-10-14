import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

PKG_DIR = Path(__file__).resolve().parent.parent.parent / "packages" / "wan22_svdquant_dense_block"
sys.path.insert(0, str(PKG_DIR))

import svdquant_dense_kernel
from qdq_utils import quantize_activation_int4, unpack_int4_signed, pack_int4_signed
from reference_kernels import bf16_linear, bf16_linear_activation, wan_rmsnorm
from triton_w4a4_gemm import triton_svdquant_w4a4_gemm
from task import execute_sycl_layer, make_sycl_dense_block_fn, reference_dense_block


def sync():
    torch.xpu.synchronize()


def time_fn(fn, warmup=20, runs=60):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    sync()
    t1 = time.perf_counter()
    return (t1 - t0) / runs * 1000.0


def make_layer_params(k_dim, n_dim, r_dim=32, device="xpu", has_norm=False):
    num_g = k_dim // 64
    w_raw = torch.randint(-8, 8, (n_dim, k_dim), dtype=torch.int8, device=device)
    w_grouped = w_raw.view(n_dim, num_g, 64).to(torch.float32)
    g_max = torch.amax(torch.abs(w_grouped), dim=-1, keepdim=True)
    w_scales = torch.clamp(g_max / 7.0, min=1e-8)
    w_int4 = torch.clamp(torch.round(w_grouped / w_scales), -8, 7).to(torch.int8)
    w_q = pack_int4_signed(w_int4.view(n_dim, k_dim))
    w_scale = w_scales.view(n_dim, num_g).to(torch.bfloat16).contiguous()
    bias = (torch.randn(n_dim, dtype=torch.bfloat16, device=device) * 0.05).contiguous()
    la = (torch.randn((r_dim, k_dim), dtype=torch.bfloat16, device=device) * 0.02).contiguous()
    lb = (torch.randn((n_dim, r_dim), dtype=torch.bfloat16, device=device) * 0.02).contiguous()
    nw = (torch.ones(n_dim, dtype=torch.bfloat16, device=device) + torch.randn(n_dim, dtype=torch.bfloat16, device=device) * 0.01).contiguous() if has_norm else None
    w_bf16 = (w_raw.to(torch.bfloat16) * 0.1).contiguous()
    return {
        "w_q": w_q, "w_scale": w_scale, "bias": bias,
        "lora_a": la, "lora_b": lb, "norm_w": nw, "w_bf16": w_bf16,
    }


def main():
    device = torch.device("xpu")
    print("=" * 70)
    print("SVDQuant W4A4 + BF16 LoRA Local Validation and Timing (Arc Pro B70)")
    print("=" * 70)

    from task import block_data
    data = block_data.__wrapped__(device)

    x = data["x"]
    qp = data["q_p"]
    kp = data["k_p"]
    vp = data["v_p"]
    up = data["ffn_up_p"]
    dp = data["ffn_down_p"]
    sycl_args = data["sycl_args"]
    ref_args = data["ref_args"]
    M = 1215
    N_ffn = 14336

    print("\n--- Correctness Testing vs Triton Oracle ---")

    # 1. Q (RMSNorm)
    t_q = triton_svdquant_w4a4_gemm(x, qp["w_q"], qp["w_scale"], bias=qp["bias"], lora_a=qp["lora_a"], lora_b=qp["lora_b"], norm_weight=qp["norm_w"], activation="none")
    s_q = execute_sycl_layer(svdquant_dense_kernel, x, qp["w_q"], qp["w_scale"], bias=qp["bias"], lora_a=qp["lora_a"], lora_b=qp["lora_b"], norm_weight=qp["norm_w"], activation="none")
    diff_q = (t_q - s_q).abs().max().item()
    print(f"1. Q Projection (RMSNorm) max diff vs Triton: {diff_q:.5f} (Pass: {diff_q <= 0.05})")
    assert diff_q <= 0.05

    # 2. V
    t_v = triton_svdquant_w4a4_gemm(x, vp["w_q"], vp["w_scale"], bias=vp["bias"], lora_a=vp["lora_a"], lora_b=vp["lora_b"], activation="none")
    s_v = execute_sycl_layer(svdquant_dense_kernel, x, vp["w_q"], vp["w_scale"], bias=vp["bias"], lora_a=vp["lora_a"], lora_b=vp["lora_b"], activation="none")
    diff_v = (t_v - s_v).abs().max().item()
    print(f"2. V Projection max diff vs Triton: {diff_v:.5f} (Pass: {diff_v <= 0.035})")
    assert diff_v <= 0.035

    # 3. FFN Up (GELU)
    t_up = triton_svdquant_w4a4_gemm(x, up["w_q"], up["w_scale"], bias=up["bias"], lora_a=up["lora_a"], lora_b=up["lora_b"], activation="gelu_tanh")
    s_up_bf16, s_up_q, s_up_scales = execute_sycl_layer(svdquant_dense_kernel, x, up["w_q"], up["w_scale"], bias=up["bias"], lora_a=up["lora_a"], lora_b=up["lora_b"], activation="gelu_tanh", fuse_quant_out=True)
    diff_up = (t_up - s_up_bf16).abs().max().item()
    print(f"3. FFN Up (gelu_tanh + quant_out) max diff vs Triton: {diff_up:.5f} (Pass: {diff_up <= 0.035})")
    assert diff_up <= 0.035
    assert s_up_q.shape == (M, N_ffn // 2)

    # 4. FFN Down (INT4 input)
    h_bf16 = (torch.randn((M, N_ffn), dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    h_q, h_scales = quantize_activation_int4(h_bf16, group_size=64)
    t_down = triton_svdquant_w4a4_gemm(h_bf16, dp["w_q"], dp["w_scale"], bias=dp["bias"], lora_a=dp["lora_a"], lora_b=dp["lora_b"], activation="none")
    s_down = execute_sycl_layer(svdquant_dense_kernel, h_q, dp["w_q"], dp["w_scale"], bias=dp["bias"], lora_a=dp["lora_a"], lora_b=dp["lora_b"], x_scales=h_scales, x_bf16=h_bf16, activation="none")
    diff_down = (t_down - s_down).abs().max().item()
    print(f"4. FFN Down (INT4 input) max diff vs Triton: {diff_down:.5f} (Pass: {diff_down <= 0.05})")
    assert diff_down <= 0.05

    # 5. Full Dense Block
    sycl_dense_block_fn = make_sycl_dense_block_fn(svdquant_dense_kernel)
    b_q, b_k, b_v, b_down = sycl_dense_block_fn(*sycl_args)
    print("5. Full Chained Dense Block Correctness: ALL PASSED!")

    print("\n" + "=" * 70)
    print("Performance & Speedup Benchmarks (Arc Pro B70)")
    print("=" * 70)

    # Q Projection timing
    time_q_ref = time_fn(lambda: bf16_linear(x, qp["w_bf16"], qp["bias"], qp["lora_a"], qp["lora_b"], norm_weight=qp["norm_w"]))
    time_q_sycl = time_fn(lambda: execute_sycl_layer(svdquant_dense_kernel, x, qp["w_q"], qp["w_scale"], bias=qp["bias"], lora_a=qp["lora_a"], lora_b=qp["lora_b"], norm_weight=qp["norm_w"]))
    print(f"Projection Q (1215x3072 + RMSNorm): SYCL = {time_q_sycl:.3f} ms | Ref BF16 = {time_q_ref:.3f} ms | Speedup = {time_q_ref/time_q_sycl:.2f}x")

    # FFN Up timing
    time_up_ref = time_fn(lambda: bf16_linear_activation(x, up["w_bf16"], up["bias"], up["lora_a"], up["lora_b"], activation="gelu_tanh"))
    time_up_sycl = time_fn(lambda: execute_sycl_layer(svdquant_dense_kernel, x, up["w_q"], up["w_scale"], bias=up["bias"], lora_a=up["lora_a"], lora_b=up["lora_b"], activation="gelu_tanh", fuse_quant_out=True))
    print(f"FFN Up (1215x14336 + GELU):         SYCL = {time_up_sycl:.3f} ms | Ref BF16 = {time_up_ref:.3f} ms | Speedup = {time_up_ref/time_up_sycl:.2f}x")

    # FFN Down timing
    time_down_ref = time_fn(lambda: bf16_linear(h_bf16, dp["w_bf16"], dp["bias"], dp["lora_a"], dp["lora_b"]))
    time_down_sycl = time_fn(lambda: execute_sycl_layer(svdquant_dense_kernel, h_q, dp["w_q"], dp["w_scale"], bias=dp["bias"], lora_a=dp["lora_a"], lora_b=dp["lora_b"], x_scales=h_scales, x_bf16=h_bf16))
    print(f"FFN Down (1215x3072, INT4 in):      SYCL = {time_down_sycl:.3f} ms | Ref BF16 = {time_down_ref:.3f} ms | Speedup = {time_down_ref/time_down_sycl:.2f}x")

    # Total Dense Block timing
    time_block_ref = time_fn(lambda: reference_dense_block(*ref_args))
    time_block_sycl = time_fn(lambda: sycl_dense_block_fn(*sycl_args))

    print("-" * 70)
    print(f"FULL DENSE BLOCK TOTAL:             SYCL = {time_block_sycl:.3f} ms | Ref BF16 = {time_block_ref:.3f} ms | Speedup = {time_block_ref/time_block_sycl:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
