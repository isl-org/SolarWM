import torch
from solarwm.backends.wan22.runtime.components import Wan5BVAE

def profile_kernel_elimination():
    dev = "xpu"
    vae_path = "/home/ssheorey/models/SolarWM/SolarWM-5B-base/vae/Wan2.2_VAE.pth"
    print("Loading Wan5BVAE...")
    vae = Wan5BVAE(vae_path, xpu_channels_last=True).to(dev, dtype=torch.bfloat16)
    decoder = vae.module.decoder

    # Latent tensor for 1 frame: [1, 1024, 1, 30, 54] into decoder.middle
    # Or full latent tile: [1, 3, 48, 30, 54]
    tile_latents = torch.randn(1, 3, 48, 30, 54, device=dev, dtype=torch.bfloat16)

    with vae.streaming_decode_session() as decode_tile:
        # Warmup
        out = decode_tile(tile_latents)
        torch.xpu.synchronize()

        # Profile with torch.profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU],
            record_shapes=True,
        ) as prof:
            out = decode_tile(tile_latents)
            torch.xpu.synchronize()

    # Analyze profiler events
    key_averages = prof.key_averages()
    add_events = [evt for evt in key_averages if "aten::add" in evt.key]

    print("\n" + "=" * 60)
    print("PROFILER KERNEL AUDIT: aten::add Occurrences")
    print("=" * 60)
    for evt in add_events:
        print(f"Key: {evt.key:30s} | Count: {evt.count:4d} | Device Time: {evt.device_time_total / 1000.0:8.3f} ms")
    print("=" * 60)

    total_add_count = sum(evt.count for evt in add_events)
    print(f"Total aten::add calls in 1 tile (3 latent frames): {total_add_count}")

if __name__ == "__main__":
    profile_kernel_elimination()
