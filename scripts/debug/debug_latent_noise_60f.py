"""Debug standalone Stage2 rollout for 60 latent frames with static camera.

Checks whether latents or decoded frames become noisy over time and saves sample images.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from PIL import Image

from solarwm.backends.wan22.runtime.stage2 import build_stage2_generation_provider
from solarwm.backends.wan22.runtime.interactive import Stage2InteractiveSession
from solarwm.backends.wan22.runtime.interactive_camera import FreeFlyCamera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/examples/wan22_ti2v_5b/infer_stage2_sgf_interactive_xpu.local.yaml",
    )
    parser.add_argument("--image", default="assets/eastshade.webp")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="outputs/debug_noise_60f")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    print(f"Building provider on {config['inference'].get('device', 'cpu')}...")
    provider = build_stage2_generation_provider(config)
    provider._load_role("model")
    device = provider.device

    image = Image.open(Path(args.image)).convert("RGB")
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

    print(f"first_latent shape: {first_latent.shape}, mean: {first_latent.mean().item():.4f}, std: {first_latent.std().item():.4f}")

    camera = FreeFlyCamera()
    decode_context = provider.vae.streaming_decode_session()
    decode_tile = decode_context.__enter__()

    try:
        session = Stage2InteractiveSession.from_provider(
            provider,
            first_latent=first_latent,
            condition=condition,
            seed=args.seed,
            max_latent_frames=args.frames,
            decode_tile=decode_tile,
            first_pose=camera.pose,
        )

        print(f"\nRunning {args.frames} frames with static camera (no keypresses)...")
        for i in range(args.frames):
            t0 = time.perf_counter()
            # No keys pressed -> static camera
            pose = camera.update(set(), 0.05)
            cam_tokens = camera.model_tokens(
                frame_sequence_length=session.frame_sequence_length,
                device=device,
                dtype=session.dtype,
            )
            frame = session.step(cam_tokens, pose=pose)
            dt = time.perf_counter() - t0

            pix = frame.pixels
            pix_mean = pix.float().mean().item()
            pix_std = pix.float().std().item()

            # Save sample images
            if i in [0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 59] or i % 10 == 0:
                frame_np = pix[0, 0].detach().float().cpu().numpy()
                if frame_np.shape[0] == 3:
                    frame_np = np.transpose(frame_np, (1, 2, 0))
                frame_uint8 = np.clip((frame_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
                img_path = outdir / f"frame_{i:03d}.jpg"
                Image.fromarray(frame_uint8).save(img_path, quality=90)

                # Measure noise/high-frequency energy using Laplacian or gradient
                gray = np.mean(frame_uint8.astype(float), axis=-1)
                grad_y = np.abs(np.diff(gray, axis=0))
                grad_x = np.abs(np.diff(gray, axis=1))
                hf_energy = float(np.mean(grad_y) + np.mean(grad_x))

                print(
                    f"Frame {i:3d}: diff={frame.diffusion_seconds*1000:.1f}ms, "
                    f"dec={frame.decode_seconds*1000:.1f}ms, total={dt*1000:.1f}ms | "
                    f"pix mean={pix_mean:+.3f}, std={pix_std:.3f}, hf_energy={hf_energy:.2f} -> saved {img_path.name}"
                )
    finally:
        decode_context.__exit__(None, None, None)
        provider.close()


if __name__ == "__main__":
    main()
