import time
import torch
import torch.nn as nn
from solarwm.backends.wan22.runtime.components import Wan5BVAE
from solarwm.kernels.vae_fused import triton_rms_norm_silu
from solarwm.backends.wan22.runtime.modeling.vae import RMS_norm

class FusedNormSiLU(nn.Module):
    def __init__(self, norm_module: RMS_norm):
        super().__init__()
        self.gamma = norm_module.gamma
        self.bias = getattr(norm_module, "bias", None)
        self.scale = getattr(norm_module, "scale", None)

    def forward(self, x):
        return triton_rms_norm_silu(x, self.gamma, self.bias, self.scale)


def patch_vae_with_fused_kernels(vae_module):
    """Replace (RMS_norm, SiLU) pairs in decoder with FusedNormSiLU + nn.Identity()."""
    count = 0
    decoder = vae_module.decoder

    def replace_norm_silu_in_seq(seq):
        nonlocal count
        for i in range(len(seq) - 1):
            if isinstance(seq[i], RMS_norm) and isinstance(seq[i + 1], nn.SiLU):
                seq[i] = FusedNormSiLU(seq[i])
                seq[i + 1] = nn.Identity()
                count += 1

    # 1. Middle blocks
    for m in decoder.middle:
        if hasattr(m, "residual") and isinstance(m.residual, nn.Sequential):
            replace_norm_silu_in_seq(m.residual)

    # 2. Upsample blocks
    for up in decoder.upsamples:
        for block in up.upsamples:
            if hasattr(block, "residual") and isinstance(block.residual, nn.Sequential):
                replace_norm_silu_in_seq(block.residual)

    # 3. Head
    replace_norm_silu_in_seq(decoder.head)

    print(f"Patched {count} (RMS_norm + SiLU) pairs with fused Triton kernels")


def benchmark_full_tile(N=10):
    dev = "xpu"
    vae_path = "/home/ssheorey/models/SolarWM/SolarWM-5B-base/vae/Wan2.2_VAE.pth"
    tile_latents = torch.randn(1, 3, 48, 30, 54, device=dev, dtype=torch.bfloat16)

    # 1. Baseline
    vae_base = Wan5BVAE(vae_path, xpu_channels_last=True).to(dev, dtype=torch.bfloat16)

    with vae_base.streaming_decode_session() as decode_tile:
        out_base_0 = decode_tile(tile_latents) # Warmup
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for _ in range(N):
            out_base = decode_tile(tile_latents)
        torch.xpu.synchronize()
        t1 = time.perf_counter()
        base_ms = (t1 - t0) * 1000 / N

    # 2. Fused Triton
    vae_fused = Wan5BVAE(vae_path, xpu_channels_last=True).to(dev, dtype=torch.bfloat16)
    patch_vae_with_fused_kernels(vae_fused.module)

    with vae_fused.streaming_decode_session() as decode_tile:
        out_fused_0 = decode_tile(tile_latents) # Warmup
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for _ in range(N):
            out_fused = decode_tile(tile_latents)
        torch.xpu.synchronize()
        t1 = time.perf_counter()
        fused_ms = (t1 - t0) * 1000 / N

    print("\n" + "=" * 65)
    print("END-TO-END VAE DECODE TILE BENCHMARK (12 frames, 480x864)")
    print("=" * 65)
    print(f"Baseline (Eager BF16 channels_last_3d): {base_ms:8.2f} ms")
    print(f"Fused Triton Kernels:                 {fused_ms:8.2f} ms")
    diff_ms = base_ms - fused_ms
    pct = (diff_ms / base_ms) * 100
    print(f"Delta:                                {diff_ms:8.2f} ms ({pct:.1f}% reduction)")
    print("=" * 65)

    max_pixel_diff = (out_base.float() - out_fused.float()).abs().max().item()
    mean_pixel_diff = (out_base.float() - out_fused.float()).abs().mean().item()
    print(f"Decoded pixel parity: Max diff: {max_pixel_diff:.5f}, Mean diff: {mean_pixel_diff:.6f}")


if __name__ == "__main__":
    benchmark_full_tile(N=10)
