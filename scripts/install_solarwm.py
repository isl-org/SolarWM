#!/usr/bin/env python3
"""Install SolarWM's Wan2.2 runtime into the active Python environment."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any, NoReturn


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path.home() / "SolarWM-Data" / "releases-v1"
HF_MODEL_REPO = "junchaoh-cs/SolarWM"

BASE_PATTERNS = (
    "SolarWM-5B-base/LICENSE.txt",
    "SolarWM-5B-base/UPSTREAM_README.md",
    "SolarWM-5B-base/assets-manifest.json",
    "SolarWM-5B-base/conditioning/**",
    "SolarWM-5B-base/text_encoder/**",
    "SolarWM-5B-base/tokenizer/**",
    "SolarWM-5B-base/vae/**",
)
STAGE2_PATTERN = "SolarWM-5B-sgf-stage2-81f/**"
WAN_RUNTIME_PACKAGES = (
    "diffusers==0.38.0",
    "decord>=0.6",
    "einops>=0.8",
    "ftfy>=6.2",
    "omegaconf>=2.3",
    "regex>=2024.0",
    "safetensors>=0.4",
    "transformers==5.12.1",
    "peft==0.20.0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install SolarWM in the active virtual environment and download "
            "the Wan2.2-5B Stage2 inference assets."
        )
    )
    parser.add_argument(
        "--comfyui",
        type=Path,
        help="ComfyUI installation whose extra_model_paths.yaml should be updated",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        help="Optional local model directory; default is Hugging Face's cache",
    )
    parser.add_argument(
        "--svdquant-repo",
        help="Optional private Hugging Face repo containing a svdquant_int4/ directory",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Install runtime dependencies without downloading model weights",
    )
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def ensure_virtual_environment() -> None:
    if sys.prefix == sys.base_prefix:
        fail("activate a Python virtual environment before running this script")


def run_pip(*arguments: str) -> None:
    command = [sys.executable, "-m", "pip", *arguments]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def detect_device() -> tuple[str, Any]:
    try:
        import torch
    except ImportError as exc:
        fail(f"torch must already be installed in the active environment: {exc}")

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu", torch
    except RuntimeError:
        pass
    if torch.cuda.is_available():
        return "cuda", torch
    return "cpu", torch


def install_runtime(device: str) -> None:
    if device == "cuda":
        run_pip("install", "--no-deps", "-e", ".[wan]")
        run_pip("install", *WAN_RUNTIME_PACKAGES)
        run_pip(
            "install",
            "--no-deps",
            "--no-build-isolation",
            "flash-attn==2.8.3",
        )
    else:
        run_pip("install", "-e", ".[wan-xpu]")


def ensure_huggingface_hub() -> Any:
    if importlib.util.find_spec("huggingface_hub") is None:
        run_pip("install", "huggingface_hub")
    from huggingface_hub import snapshot_download

    return snapshot_download


def model_weights_are_present(model_root: Path) -> bool:
    required_files = (
        model_root
        / "SolarWM-5B-base"
        / "text_encoder"
        / "models_t5_umt5-xxl-enc-bf16.pth",
        model_root / "SolarWM-5B-base" / "tokenizer",
        model_root / "SolarWM-5B-base" / "vae" / "Wan2.2_VAE.pth",
        model_root / "SolarWM-5B-sgf-stage2-81f" / "model.pt",
    )
    return all(path.exists() for path in required_files)


def svdquant_weights_are_present(model_root: Path) -> bool:
    root = model_root / "SolarWM-5B-sgf-stage2-81f" / "svdquant_int4"
    return root.is_dir() and any(root.rglob("*.safetensors"))


def download_weights(
    local_dir: Path | None,
    svdquant_repo: str | None,
) -> Path:
    if local_dir is not None:
        local_dir = local_dir.expanduser().resolve()
        local_dir.mkdir(parents=True, exist_ok=True)

    if local_dir is not None and model_weights_are_present(local_dir):
        model_root = local_dir
        print(f"Model weights already present: {model_root}")
    else:
        snapshot_download = ensure_huggingface_hub()
        download_args = {
            "repo_id": HF_MODEL_REPO,
            "allow_patterns": (*BASE_PATTERNS, STAGE2_PATTERN),
        }
        if local_dir is not None:
            download_args["local_dir"] = local_dir
            print(f"Downloading {HF_MODEL_REPO} to {local_dir}")
        else:
            print(f"Downloading {HF_MODEL_REPO} to the Hugging Face cache")
        model_root = Path(snapshot_download(**download_args))

    if svdquant_repo:
        svdquant_root = model_root / "SolarWM-5B-sgf-stage2-81f"
        if local_dir is not None and svdquant_weights_are_present(model_root):
            print("SVDQuant weights already present; skipping private repository download")
        else:
            snapshot_download = ensure_huggingface_hub()
            if local_dir is not None:
                svdquant_root.mkdir(parents=True, exist_ok=True)
                print(f"Downloading {svdquant_repo} to {svdquant_root}")
                snapshot_download(repo_id=svdquant_repo, local_dir=svdquant_root)
            else:
                print(f"Downloading {svdquant_repo} to the Hugging Face cache")
                snapshot_download(repo_id=svdquant_repo)
    return model_root


def configure_comfyui(comfyui_root: Path, model_root: Path | None) -> None:
    if not comfyui_root.is_dir():
        fail(f"ComfyUI directory does not exist: {comfyui_root}")

    import yaml

    config_path = comfyui_root / "extra_model_paths.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            fail(f"ComfyUI config must contain a mapping: {config_path}")
    else:
        config = {}

    desired = {
        "solarwm": {
            "base_path": str(REPO_ROOT),
            "custom_nodes": "integrations",
            "datasets": str(DATA_ROOT),
        },
    }
    if model_root is not None:
        desired["solarwm_models"] = {
            "base_path": str(model_root),
            "vae": "SolarWM-5B-base/vae",
            "text_encoders": "SolarWM-5B-base/text_encoder",
            "diffusion_models": "SolarWM-5B-sgf-stage2-81f",
        }
    if all(config.get(key) == value for key, value in desired.items()):
        print(f"ComfyUI paths already configured: {config_path}")
        return

    config.update(desired)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"Updated {config_path}")


def main() -> None:
    args = parse_args()
    ensure_virtual_environment()
    device, torch = detect_device()
    print(f"Using active Python: {sys.executable}")
    print(f"Detected device: {device} (torch {torch.__version__})")

    install_runtime(device)
    model_root = None
    if not args.skip_weights:
        model_root = download_weights(args.local_dir, args.svdquant_repo)
    if args.comfyui:
        configure_comfyui(args.comfyui.expanduser().resolve(), model_root)

    print("SolarWM installation complete.")
    if args.comfyui:
        print("Restart ComfyUI to load its updated extra_model_paths.yaml.")


if __name__ == "__main__":
    main()
