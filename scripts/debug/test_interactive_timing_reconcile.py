"""Benchmark Stage2InteractiveSession on XPU with synchronized timing for Chunk 4."""

from pathlib import Path
import time
import torch
import yaml
from PIL import Image
import numpy as np

from solarwm.backends.wan22.runtime.stage2 import build_stage2_generation_provider
from solarwm.backends.wan22.runtime.interactive import Stage2InteractiveSession
from solarwm.backends.wan22.runtime.interactive_camera import FreeFlyCamera, camera_tokens_from_poses


def main():
    config_path = "configs/examples/wan22_ti2v_5b/infer_stage2_sgf_interactive_xpu.local.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["model"]["num_frame_per_block"] = 3
    config.setdefault("runtime", {})["stage2_vae_quant"] = "quarot_w8a8"

    print(f"Loading model on {config['inference'].get('device', 'xpu')}...")
    provider = build_stage2_generation_provider(config)
    provider._load_role("model")

    # Load initial anchor image
    image_path = "assets/eastshade.webp"
    image = Image.open(image_path).convert("RGB").resize(
        (int(config["data"]["width"]), int(config["data"]["height"])),
        Image.Resampling.BICUBIC,
    )
    pixels = (
        torch.from_numpy(np.asarray(image))
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .div(127.5)
        .sub(1.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=provider.device, dtype=torch.bfloat16)
    )

    with torch.no_grad():
        first_latent = provider.vae.encode(
            pixels[:, :1].permute(0, 2, 1, 3, 4).contiguous()
        ).to(torch.bfloat16)
        condition = provider.text_encoder([""])
        prompt_embeds = condition["prompt_embeds"]
        if prompt_embeds.device != provider.device:
            condition = {
                **condition,
                "prompt_embeds": prompt_embeds.to(
                    device=provider.device,
                    dtype=torch.bfloat16,
                ),
            }

    camera = FreeFlyCamera()
    decode_context = provider.vae.streaming_decode_session()
    decode_tile = decode_context.__enter__()

    try:
        session = Stage2InteractiveSession.from_provider(
            provider,
            first_latent=first_latent,
            condition=condition,
            seed=42,
            max_latent_frames=18,  # 6 chunks = 18 latent frames (chunk 4 is step 4)
            decode_tile=decode_tile,
            first_pose=camera.pose,
        )

        print("\nRunning 6 chunks (18 latent frames) with device synchronization...")
        step_idx = 0
        while not session.done:
            poses = [camera.pose.copy() for _ in range(session.chunk)]
            camera_tokens = camera_tokens_from_poses(
                poses,
                origin=camera._origin,
                frame_sequence_length=session.frame_sequence_length,
                device=session.device,
                dtype=session.dtype,
            )

            frame = session.step(camera_tokens, pose=camera.pose)
            print(
                f"Chunk {step_idx:2d} (latent {frame.index:2d}..{frame.index+session.chunk:2d}): "
                f"DiT={frame.diffusion_seconds*1000:6.1f} ms | "
                f"VAE={frame.decode_seconds*1000:6.1f} ms | "
                f"Total={ (frame.diffusion_seconds + frame.decode_seconds)*1000:6.1f} ms | "
                f"emitted={frame.pixels.shape[1]} pix"
            )
            step_idx += 1

    finally:
        decode_context.__exit__(None, None, None)
        provider.close()


if __name__ == "__main__":
    main()
