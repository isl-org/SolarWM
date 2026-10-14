"""Compare Stage2 batch generation with num_frame_per_block=3 vs num_frame_per_block=1.

Tests whether the noise drift happens in standard batch Stage2 inference.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from PIL import Image

from solarwm.backends.wan22.runtime.stage2 import build_stage2_generation_provider, _stage2_self_forcing_latents
from solarwm.backends.wan22.runtime.interactive_camera import FreeFlyCamera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=3, choices=[1, 3])
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="outputs/debug_batch_chunk3")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config_path = Path("configs/examples/wan22_ti2v_5b/infer_stage2_sgf_interactive_xpu.local.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["num_frame_per_block"] = args.chunk

    print(f"Building provider with chunk={args.chunk} on {config['inference'].get('device', 'cpu')}...")
    provider = build_stage2_generation_provider(config)
    provider._load_role("model")
    device = provider.device

    image = Image.open(Path("assets/eastshade.webp")).convert("RGB")
    width = int(config["data"]["width"])
    height = int(config["data"]["height"])
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    pixels = (
        torch.from_numpy(np.asarray(image).copy())
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .div(127.5)
        .sub(1.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device, dtype=torch.bfloat16)
    )

    with torch.no_grad():
        first_latent = provider.vae.encode(
            pixels[:, :1].permute(0, 2, 1, 3, 4).contiguous()
        ).to(torch.bfloat16)
        condition = provider.text_encoder(["A moving camera through the scene"])
        prompt_embeds = condition["prompt_embeds"]
        if prompt_embeds.device != device:
            condition = {
                **condition,
                "prompt_embeds": prompt_embeds.to(device=device, dtype=torch.bfloat16),
            }

    # Static camera for all frames
    frame_seqlen = int(config["model"]["frame_sequence_length"])
    total_tokens = args.frames * frame_seqlen
    viewmats = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(total_tokens, 1, 1).unsqueeze(0)
    intrinsics = torch.zeros((1, total_tokens, 3, 3), dtype=torch.float32, device=device)
    intrinsics[0, :, 0, 0] = 1.0  # FX
    intrinsics[0, :, 1, 1] = 1.0  # FY
    intrinsics[0, :, 0, 2] = 0.5
    intrinsics[0, :, 1, 2] = 0.5
    intrinsics[0, :, 2, 2] = 1.0
    camera = {"viewmats": viewmats, "K": intrinsics}

    generation_pass = replace(
        provider.plan.passes[0],
        rollout_latent_frames=args.frames,
        min_rollout_latent_frames=args.frames,
    )
    rng = torch.Generator(device=device).manual_seed(args.seed)

    print(f"Running _stage2_self_forcing_latents with chunk={args.chunk} for {args.frames} frames...")
    t0 = time.perf_counter()
    with torch.no_grad():
        latents, schedule = _stage2_self_forcing_latents(
            provider,
            generation_pass,
            first_latent,
            condition,
            camera,
            rng,
        )
    print(f"Rollout took {time.perf_counter() - t0:.2f}s, latents shape: {latents.shape}")

    # Decode sample frames
    decode_context = provider.vae.streaming_decode_session()
    with decode_context as decode_tile:
        for i in [0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 59]:
            if i >= args.frames:
                continue
            tile = latents[:, i:i+1]
            pix = decode_tile(tile)
            frame_np = pix[0, 0].detach().float().cpu().numpy()
            if frame_np.shape[0] == 3:
                frame_np = np.transpose(frame_np, (1, 2, 0))
            frame_uint8 = np.clip((frame_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
            img_path = outdir / f"batch_frame_{i:03d}.jpg"
            Image.fromarray(frame_uint8).save(img_path, quality=90)
            gray = np.mean(frame_uint8.astype(float), axis=-1)
            grad_y = np.abs(np.diff(gray, axis=0))
            grad_x = np.abs(np.diff(gray, axis=1))
            hf_energy = float(np.mean(grad_y) + np.mean(grad_x))
            print(f"Decoded frame {i:2d}: hf_energy={hf_energy:.2f} -> saved {img_path.name}")

    provider.close()


if __name__ == "__main__":
    main()
