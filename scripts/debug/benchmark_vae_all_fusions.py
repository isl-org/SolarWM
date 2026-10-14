import time
import torch
import torch.nn as nn
from solarwm.backends.wan22.runtime.components import Wan5BVAE
from solarwm.backends.wan22.runtime.modeling.vae_quarot import apply_vae_quarot

def run_comprehensive_benchmark(N=10):
    dev = "xpu"
    vae_path = "/home/ssheorey/models/SolarWM/SolarWM-5B-base/vae/Wan2.2_VAE.pth"
    print("=" * 70)
    print("COMPREHENSIVE END-TO-END VAE DECODE BENCHMARK (12 frames, 480x864)")
    print("=" * 70)

    tile_latents = torch.randn(1, 3, 48, 30, 54, device=dev, dtype=torch.bfloat16)

    # 1. Baseline: Eager BF16 channels_last_3d
    print("1. Running Baseline (Eager BF16)...")
    vae_base = Wan5BVAE(vae_path, xpu_channels_last=True).to(dev, dtype=torch.bfloat16)
    with vae_base.streaming_decode_session() as decode_tile:
        out_base_0 = decode_tile(tile_latents) # Warmup
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for _ in range(N):
            out_base = decode_tile(tile_latents)
        torch.xpu.synchronize()
        base_ms = (time.perf_counter() - t0) * 1000 / N
    print(f"   Baseline Latency: {base_ms:.2f} ms")

    # 2. WS20: Fused Triton Norm + SiLU
    print("2. Running WS20 (Fused Triton Norm/SiLU)...")
    vae_fused = Wan5BVAE(vae_path, xpu_channels_last=True).to(dev, dtype=torch.bfloat16)
    vae_fused.module.configure_xpu_fused_decode(True)
    with vae_fused.streaming_decode_session() as decode_tile:
        out_fused_0 = decode_tile(tile_latents) # Warmup
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for _ in range(N):
            out_fused = decode_tile(tile_latents)
        torch.xpu.synchronize()
        fused_ms = (time.perf_counter() - t0) * 1000 / N
    print(f"   WS20 Latency:     {fused_ms:.2f} ms")

    # 3. WS23: QuaRot W8A8 with Target 1 (append_sum) + Target 2 (direct quant & buffer reuse)
    print("3. Running WS23 (QuaRot INT8 + Hardware Epilogue Accumulation)...")
    vae_opt = Wan5BVAE(vae_path, xpu_channels_last=True).to(dev, dtype=torch.bfloat16)
    vae_opt.module.configure_xpu_fused_decode(True)
    apply_vae_quarot(vae_opt.module.decoder, enabled=True, use_compile=True)

    with vae_opt.streaming_decode_session() as decode_tile:
        out_opt_0 = decode_tile(tile_latents) # Warmup
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for _ in range(N):
            out_opt = decode_tile(tile_latents)
        torch.xpu.synchronize()
        opt_ms = (time.perf_counter() - t0) * 1000 / N
    print(f"   WS23 Latency:     {opt_ms:.2f} ms")

    print("\n" + "=" * 70)
    print("SUMMARY RESULTS (Intel Arc B580 - 12 frames, 480x864):")
    print("=" * 70)
    print(f"Baseline (Eager BF16):                           {base_ms:8.2f} ms  (1.00x)")
    print(f"WS20 (BF16 + Fused Triton Norm/SiLU):            {fused_ms:8.2f} ms  ({base_ms/fused_ms:5.2f}x)")
    print(f"WS23 (QuaRot W8A8 + Hardware Epilogue):          {opt_ms:8.2f} ms  ({base_ms/opt_ms:5.2f}x)")
    print("-" * 70)
    saved_vs_base = base_ms - opt_ms
    saved_vs_fused = fused_ms - opt_ms
    print(f"Net Savings vs Baseline:   {saved_vs_base:8.2f} ms ({(saved_vs_base/base_ms)*100:4.1f}% reduction)")
    print(f"Net Savings vs WS20 Fused: {saved_vs_fused:8.2f} ms ({(saved_vs_fused/fused_ms)*100:4.1f}% reduction)")
    print("=" * 70)

    # Pixel Parity Check:
    diff_vs_base = (out_base.float() - out_opt.float()).abs()
    print(f"Output Pixel Max Diff vs Baseline:  {diff_vs_base.max().item():.5f}")
    print(f"Output Pixel Mean Diff vs Baseline: {diff_vs_base.mean().item():.6f}")

    diff_vs_fused = (out_fused.float() - out_opt.float()).abs()
    print(f"Output Pixel Max Diff vs WS20:      {diff_vs_fused.max().item():.5f}")
    print(f"Output Pixel Mean Diff vs WS20:     {diff_vs_fused.mean().item():.6f}")
    print("=" * 70)

if __name__ == "__main__":
    run_comprehensive_benchmark(N=10)
