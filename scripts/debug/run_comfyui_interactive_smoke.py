"""Run the ComfyUI interactive provider without starting its HTTP server."""

from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path
import queue
import sys

import os

# Intel Level-Zero and SYCL relaxed allocation limits (must be set before importing torch / Level Zero driver)
os.environ.setdefault("NEOReadDebugKeys", "1")
os.environ.setdefault("ZE_ENABLE_RELAXED_ALLOCATION_LIMITS", "1")
os.environ.setdefault("UR_L0_USE_RELAXED_ALLOCATION_LIMITS", "1")
os.environ.setdefault("SYCL_PI_LEVEL_ZERO_RELAXED_ALLOCATION_LIMITS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("PYTORCH_XPU_ALLOC_CONF", "max_split_size_mb:128")

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from integrations.comfyui_solarwm.factory import build_provider_inputs, clear_cached_provider
from integrations.comfyui_solarwm.manager import manager


def _default_config() -> Path:
    candidates = (
        REPO_ROOT / "configs/examples/wan22_ti2v_5b/infer_stage2_sgf_interactive_xpu.local.yaml",
        REPO_ROOT / "configs/examples/wan22_ti2v_5b/infer_stage2_sgf_interactive.local.yaml",
        REPO_ROOT / "configs/examples/wan22_ti2v_5b/infer_stage2_sgf_interactive.yaml",
    )
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config(),
    )
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "assets/eastshade.webp")
    parser.add_argument("--prompt", default="A natural outdoor scene with a steady camera.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-latent-frames", type=int, default=3)
    parser.add_argument(
        "--resolution",
        choices=["360p (320x576)", "480p (480x864)", "360p", "480p"],
        default="360p (320x576)",
        help="Resolution to generate (default: 360p (320x576)).",
    )
    parser.add_argument(
        "--linear-quant",
        choices=["full_precision", "svdquant_k1", "svdquant_k2", "svdquant_k4"],
        default="full_precision",
        help="Linear layer quantization backend.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["sdpa", "sageattnv1", "sageattnv2"],
        default="sdpa",
        help="Attention implementation backend.",
    )
    parser.add_argument(
        "--stage2-vae-quant",
        choices=["full_precision", "quarot_w8a8"],
        default="full_precision",
        help="Stage2 VAE quantization mode.",
    )
    args = parser.parse_args()
    if args.max_latent_frames < 1:
        parser.error("--max-latent-frames must be positive")
    if not args.config.is_file():
        parser.error(f"Config does not exist: {args.config}")
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    faulthandler.enable(all_threads=True)
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    manager.set_provider_factory(build_provider_inputs)
    session = None
    frames = 0
    resolution_str = args.resolution
    if resolution_str == "360p":
        resolution_str = "360p (320x576)"
    elif resolution_str == "480p":
        resolution_str = "480p (480x864)"
    try:
        session = manager.create({
            "config_path": str(args.config.resolve()),
            "initial_image": str(args.image.resolve()),
            "prompt": args.prompt,
            "seed": args.seed,
            "max_latent_frames": args.max_latent_frames,
            "resolution": resolution_str,
            "attention_backend": args.attention_backend,
            "linear_quant": args.linear_quant,
            "stage2_vae_quant": args.stage2_vae_quant,
            "camera_motion": "manual_only",
        })
        while True:
            try:
                event = session.events.get(timeout=60)
            except queue.Empty:
                if session.worker is None or not session.worker.is_alive():
                    raise RuntimeError("Worker exited without completion")
                print("Still generating...", flush=True)
                continue
            if not isinstance(event, dict):
                continue
            print(event, flush=True)
            if event.get("type") == "error":
                raise RuntimeError(str(event))
            if event.get("type") == "frame":
                frames += 1
            if event.get("type") == "stopped":
                if event.get("reason") != "horizon" or frames == 0:
                    raise RuntimeError("Session stopped without completing decoded frames")
                print(f"PASS: {frames} decoded frames; recording: {session.recording_path}", flush=True)
                break
    finally:
        try:
            if session is not None:
                manager.delete(session.session_id)
        finally:
            clear_cached_provider()
            torch.set_default_dtype(original_dtype)


if __name__ == "__main__":
    main()
