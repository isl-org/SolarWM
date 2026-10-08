import time
import torch
from solarwm.backends.wan22.runtime.components import Wan5BVAE

def profile_vae_decode():
    dev = "xpu"
    vae_path = "/home/ssheorey/models/SolarWM/SolarWM-5B-base/vae/Wan2.2_VAE.pth"
    vae = Wan5BVAE(vae_path, xpu_channels_last=True).to(dev, dtype=torch.bfloat16)
    decoder = vae.module.decoder

    # Input latents for 1 chunk: 1 latent frame -> 4 video frames, or 3 latent frames -> 12 video frames
    # Real decode chunk is [1, 48, 3, 30, 54] -> produces [1, 12, 12, 480, 864] (actually 16 channels, conv2 out)
    z = torch.randn(1, 48, 3, 30, 54, device=dev, dtype=torch.bfloat16)

    # Warmup
    feat_cache = [None] * 50
    feat_idx = [0]
    out = decoder(z, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=True)
    torch.xpu.synchronize()

    # Time total decoder forward
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        feat_cache = [None] * 50
        feat_idx = [0]
        out = decoder(z, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=True)
    torch.xpu.synchronize()
    t1 = time.perf_counter()
    total_tile_ms = (t1 - t0) * 1000 / N

    print(f"Total VAE Decoder Tile (12 frames): {total_tile_ms:.2f} ms")

if __name__ == "__main__":
    profile_vae_decode()
