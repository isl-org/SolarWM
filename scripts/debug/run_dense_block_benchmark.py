import sys
sys.path.insert(0, "/home/ssheorey/code/SolarWM/packages/wan22_svdquant_dense_block")
import torch
import task
import svdquant_dense_kernel
from task import bf16_linear, bf16_linear_activation, execute_sycl_layer, quantize_activation_int4

def benchmark_fn(fn, warmup=10, iters=20):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    start_evt = [torch.xpu.Event(enable_timing=True) for _ in range(iters)]
    end_evt = [torch.xpu.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_evt[i].record()
        fn()
        end_evt[i].record()
    torch.xpu.synchronize()

    times = [start_evt[i].elapsed_time(end_evt[i]) for i in range(iters)]
    return sum(times) / len(times)

def main():
    device = torch.device("xpu")
    data = task.block_data.__wrapped__(device)
    x = data["x"]
    qp = data["q_p"]
    kp = data["k_p"]
    vp = data["v_p"]
    up = data["ffn_up_p"]
    dp = data["ffn_down_p"]

    print("=== Wan2.2 TI2V-5B Camera Transformer Block Benchmark (Intel Arc Pro B70 BMG) ===")
    
    # 1. Full Dense Block
    ref_block = task.reference_dense_block
    ref_args = data["ref_args"]
    t_ref_block = benchmark_fn(lambda: ref_block(*ref_args))

    sycl_block = task.make_sycl_dense_block_fn(svdquant_dense_kernel)
    sycl_args = data["sycl_args"]
    t_sycl_block = benchmark_fn(lambda: sycl_block(*sycl_args))

    print(f"\n1. Full Dense Block (3 Projections + 2 FFN layers + LoRA + RMSNorm + GELU):")
    print(f"   - PyTorch BF16 Reference: {t_ref_block:.3f} ms")
    print(f"   - SYCL Fused SVDQuant:   {t_sycl_block:.3f} ms")

    # 2. Q Projection (M=1215, K=3072, N=3072, Rank 32, RMSNorm)
    ref_q = lambda: bf16_linear(x, qp["w_bf16"], qp["bias"], qp["lora_a"], qp["lora_b"], norm_weight=qp["norm_w"])
    sycl_q = lambda: execute_sycl_layer(svdquant_dense_kernel, x, qp["w_q"], qp["w_scale"], bias=qp["bias"], lora_a=qp["lora_a"], lora_b=qp["lora_b"], norm_weight=qp["norm_w"], activation="none")
    t_ref_q = benchmark_fn(ref_q)
    t_sycl_q = benchmark_fn(sycl_q)
    print(f"\n2. Q Projection (with RMSNorm):")
    print(f"   - PyTorch BF16 Reference: {t_ref_q:.3f} ms")
    print(f"   - SYCL Fused SVDQuant:   {t_sycl_q:.3f} ms")

    # 3. FFN Up (M=1215, K=3072, N=14336, Rank 32, gelu_tanh + INT4 output quantization)
    ref_up = lambda: bf16_linear_activation(x, up["w_bf16"], up["bias"], up["lora_a"], up["lora_b"], activation="gelu_tanh")
    sycl_up = lambda: execute_sycl_layer(svdquant_dense_kernel, x, up["w_q"], up["w_scale"], bias=up["bias"], lora_a=up["lora_a"], lora_b=up["lora_b"], activation="gelu_tanh", fuse_quant_out=True)
    t_ref_up = benchmark_fn(ref_up)
    t_sycl_up = benchmark_fn(sycl_up)
    print(f"\n3. FFN Up-Projection (with gelu_tanh + fused INT4 quantize):")
    print(f"   - PyTorch BF16 Reference: {t_ref_up:.3f} ms")
    print(f"   - SYCL Fused SVDQuant:   {t_sycl_up:.3f} ms")

    # 4. FFN Down (M=1215, K=14336, N=3072, Rank 32)
    h_bf16 = (torch.zeros((1215, 14336), dtype=torch.bfloat16, device=device)).contiguous()
    h_q, h_scales = quantize_activation_int4(h_bf16, group_size=64)
    ref_down = lambda: bf16_linear(h_bf16, dp["w_bf16"], dp["bias"], dp["lora_a"], dp["lora_b"])
    sycl_down = lambda: execute_sycl_layer(svdquant_dense_kernel, h_q, dp["w_q"], dp["w_scale"], bias=dp["bias"], lora_a=dp["lora_a"], lora_b=dp["lora_b"], x_scales=h_scales, x_bf16=h_bf16, activation="none")
    t_ref_down = benchmark_fn(ref_down)
    t_sycl_down = benchmark_fn(sycl_down)
    print(f"\n4. FFN Down-Projection (consuming INT4 directly):")
    print(f"   - PyTorch BF16 Reference: {t_ref_down:.3f} ms")
    print(f"   - SYCL Fused SVDQuant:   {t_sycl_down:.3f} ms")

if __name__ == "__main__":
    main()
