"""Test Stage2InteractiveSession with chunk=3 for 60 frames."""

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

    # Set num_frame_per_block to 3!
    config["model"]["num_frame_per_block"] = 3

    print("Building provider with num_frame_per_block=3...")
    provider = build_stage2_generation_provider(config)
    provider._load_role("model")

    # Load initial image
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

    outdir = Path("outputs/debug_interactive_chunk3")
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        session = Stage2InteractiveSession.from_provider(
            provider,
            first_latent=first_latent,
            condition=condition,
            seed=42,
            max_latent_frames=60,
            decode_tile=decode_tile,
            first_pose=camera.pose,
        )

        print(f"Session initialized with chunk={session.chunk}. Running 60 frames...")
        start_time = time.perf_counter()

        step_idx = 0
        fps = float(config.get("data", {}).get("fps", 16.0))
        dt_sub = 4.0 / max(fps, 1.0)
        while not session.done:
            # Update camera with default motion (moving forward, turning right after 4s)
            poses = []
            if session.latent_index == 0:
                poses.append(camera.pose.copy())
                for _ in range(session.chunk - 1):
                    pose = camera.update(set(), dt_sub)
                    poses.append(pose)
            else:
                for _ in range(session.chunk):
                    pose = camera.update(set(), dt_sub)
                    poses.append(pose)
            camera_tokens = camera_tokens_from_poses(
                poses,
                origin=camera._origin,
                frame_sequence_length=session.frame_sequence_length,
                device=session.device,
                dtype=session.dtype,
            )

            frame = session.step(camera_tokens, pose=poses[-1])
            pos = poses[-1][:3, 3]
            fwd = poses[-1][:3, :3] @ np.array([0.0, 0.0, -1.0])
            print(
                f"Step {step_idx:02d}: latent {frame.index:02d}..{frame.index+session.chunk:02d} | "
                f"pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | "
                f"fwd=[{fwd[0]:.2f}, {fwd[1]:.2f}, {fwd[2]:.2f}] | "
                f"time={frame.diffusion_seconds + frame.decode_seconds:.2f}s"
            )
            
            # Save the last decoded frame of this step
            last_pix = frame.pixels[0, -1].detach().float().cpu().numpy()
            if last_pix.shape[0] in (1, 3, 4) and last_pix.shape[-1] not in (1, 3, 4):
                last_pix = np.transpose(last_pix, (1, 2, 0))
            last_pix = np.clip((last_pix + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
            img = Image.fromarray(last_pix)
            img_path = outdir / f"step_{step_idx:02d}_latent_{frame.index:02d}.jpg"
            img.save(img_path)

            step_idx += 1

        print(f"Done in {time.perf_counter() - start_time:.2f}s. Total frames: {session.latent_index}")

    finally:
        decode_context.__exit__(None, None, None)
        provider.close()


if __name__ == "__main__":
    main()
