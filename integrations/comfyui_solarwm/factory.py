"""Default file-based bootstrap for the ComfyUI session manager."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def build_provider_inputs(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load a Stage2 provider and one TI2V anchor from a local image.

    ComfyUI normally supplies image tensors to nodes, but the interactive
    session is intentionally outside the prompt queue.  A local image path
    keeps the HTTP route simple and avoids copying a full image through JSON.
    Deployments that already have tensors can replace this factory with
    ``configure_provider_factory``.
    """

    import torch
    import yaml
    from PIL import Image

    from solarwm.backends.wan22.runtime.stage2 import build_stage2_generation_provider

    config_path = Path(str(payload["config_path"])).expanduser()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("SolarWM interactive config must be a mapping")
    provider = build_stage2_generation_provider(config)
    try:
        provider._load_role("model")
        image_path = payload.get("initial_image")
        if not image_path:
            raise ValueError("interactive session requires initial_image")
        image = Image.open(Path(str(image_path))).convert("RGB")
        image = image.resize(
            (int(config["data"]["width"]), int(config["data"]["height"])),
            Image.Resampling.BICUBIC,
        )
        pixels = (
            torch.from_numpy(__import__("numpy").asarray(image))
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
            condition = provider.text_encoder([str(payload.get("prompt", ""))])
            prompt_embeds = condition["prompt_embeds"]
            if prompt_embeds.device != provider.device:
                condition = {
                    **condition,
                    "prompt_embeds": prompt_embeds.to(
                        device=provider.device,
                        dtype=torch.bfloat16,
                    ),
                }
        return {
            "provider": provider,
            "first_latent": first_latent,
            "condition": condition,
        }
    except Exception:
        provider.close()
        raise


__all__ = ["build_provider_inputs"]
