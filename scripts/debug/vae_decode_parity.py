"""Streaming Wan2.2 VAE decode parity and timing on XPU, as used by the stage-2 tile decoder.

Decodes pre-encoded dataset latents in 3-latent-frame tiles through
``Wan5BVAE.streaming_decode_session`` (continuous feature cache), then either
saves the decoded frames as a baseline or compares them against one.

Usage:
  python vae_decode_parity.py save    OUT.pt [--quarot]
  python vae_decode_parity.py compare BASE.pt [--quarot]
"""

from __future__ import annotations

import argparse
import io
import tarfile
import time
from pathlib import Path

import torch
from safetensors.torch import load as load_safetensors

from solarwm.backends.wan22.runtime.components import Wan5BVAE

LATENT_TAR = Path(
    "/home/ssheorey/data/SolarWM-Data/releases-v1/example/latent/wan22-ti2v5b-153f-480p-v1.tar"
)
VAE_WEIGHTS = Path("/home/ssheorey/models/SolarWM/SolarWM-5B-base/vae/Wan2.2_VAE.pth")
TILE = 3
NUM_TILES = 6


def load_latents() -> torch.Tensor:
    with tarfile.open(LATENT_TAR) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(".safetensors"))
        tensors = load_safetensors(tar.extractfile(member).read())
    return tensors["latents"][None, : TILE * NUM_TILES].contiguous()  # B T C H W


def decode(quarot: bool) -> tuple[torch.Tensor, list[float]]:
    vae = Wan5BVAE(VAE_WEIGHTS, xpu_channels_last=True, xpu_vae_int8_quarot=quarot)
    vae.to(torch.device("xpu"), dtype=torch.bfloat16)
    latents = load_latents().to("xpu", torch.bfloat16)
    frames, times = [], []
    with torch.no_grad():
        for _ in range(2):  # pass 0 warms up kernels/autotune; pass 1 is measured
            frames, times = [], []
            with vae.streaming_decode_session() as decode_tile:
                for i in range(NUM_TILES):
                    torch.xpu.synchronize()
                    t0 = time.perf_counter()
                    out = decode_tile(latents[:, i * TILE : (i + 1) * TILE])
                    torch.xpu.synchronize()
                    times.append((time.perf_counter() - t0) * 1e3)
                    frames.append(out.float().cpu())
    return torch.cat(frames, dim=1), times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("save", "compare"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--quarot", action="store_true")
    args = parser.parse_args()

    video, times = decode(args.quarot)
    steady = sum(times[1:]) / len(times[1:])
    print(f"tile decode ms: {[round(t, 1) for t in times]} | steady mean {steady:.1f} ms")
    if args.mode == "save":
        torch.save({"video": video.to(torch.bfloat16), "times": times}, args.path)
        print(f"saved {tuple(video.shape)} -> {args.path}")
        return
    base = torch.load(args.path)
    ref = base["video"].float()
    diff = (video - ref).abs()
    mse = diff.pow(2).mean().item()
    psnr = 10 * torch.log10(torch.tensor(4.0 / max(mse, 1e-20))).item()
    u8 = lambda v: ((v.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    u8_exact = (u8(video) == u8(ref)).float().mean().item()
    base_steady = sum(base["times"][1:]) / len(base["times"][1:])
    print(
        f"vs baseline: max|diff| {diff.max().item():.4g} mean|diff| {diff.mean().item():.3g} "
        f"PSNR {psnr:.1f} dB | uint8 identical {100 * u8_exact:.3f}% "
        f"| steady {base_steady:.1f} -> {steady:.1f} ms"
    )


if __name__ == "__main__":
    main()
