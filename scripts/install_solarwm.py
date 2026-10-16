#!/usr/bin/env python3
"""Install SolarWM's Wan2.2 runtime into the active Python environment."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, NoReturn


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
DATA_ROOT = Path.home() / "SolarWM-Data" / "releases-v1"
HF_MODEL_REPO = "junchaoh-cs/SolarWM"
INTERACTIVE_CONFIG = (
    REPO_ROOT
    / "configs"
    / "examples"
    / "wan22_ti2v_5b"
    / "infer_stage2_sgf_interactive.yaml"
)

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
BASE_PACKAGES = ("numpy>=1.24", "PyYAML>=6.0")
PACKAGE_NAME = re.compile(r"^[A-Za-z0-9_.-]+")
WINDOWS_DRIVE_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")
CYGDRIVE_PATH = re.compile(r"^/cygdrive/([A-Za-z])(?:/(.*))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install SolarWM in the active virtual environment and download "
            "the Wan2.2-5B Stage2 inference assets."
        )
    )
    parser.add_argument(
        "--comfyui",
        type=str,
        help="ComfyUI installation whose extra_model_paths.yaml should be updated",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help=f"Local model directory (default: {DEFAULT_MODEL_DIR})",
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


def normalize_path(value: str | Path) -> Path:
    raw = os.fspath(value)
    cygdrive_match = CYGDRIVE_PATH.match(raw)
    if cygdrive_match and os.name == "nt":
        drive, suffix = cygdrive_match.groups()
        raw = f"{drive.upper()}:/{suffix or ''}"
    elif os.name != "nt":
        windows_match = WINDOWS_DRIVE_PATH.match(raw)
        if windows_match and Path("/cygdrive").is_dir():
            drive, suffix = windows_match.groups()
            raw = "/cygdrive/" + drive.lower() + "/" + suffix.replace(chr(92), "/")
    return Path(raw).expanduser().resolve()


def ensure_virtual_environment() -> None:
    if sys.prefix == sys.base_prefix:
        fail("activate a Python virtual environment before running this script")


def run_install(*arguments: str) -> None:
    if shutil.which("uv"):
        command = ["uv", "pip", "install", "--python", sys.executable, *arguments]
    else:
        command = [sys.executable, "-m", "pip", "install", *arguments]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def package_name(requirement: str) -> str:
    match = PACKAGE_NAME.match(requirement)
    if match is None:
        fail(f"cannot determine package name from requirement: {requirement}")
    return match.group(0)


def ensure_package(requirement: str, *install_options: str) -> None:
    name = package_name(requirement)
    try:
        installed_version = version(name)
    except PackageNotFoundError:
        print(f"Installing missing package: {requirement}")
        run_install(*install_options, requirement)
    else:
        print(f"Keeping installed package: {name} {installed_version}")


def ensure_solarwm_package() -> None:
    try:
        installed_version = version("solarwm")
    except PackageNotFoundError:
        print("Installing SolarWM in editable mode")
        run_install("--no-deps", "-e", ".")
    else:
        print(f"Keeping installed package: solarwm {installed_version}")


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
    ensure_solarwm_package()
    for requirement in BASE_PACKAGES:
        ensure_package(requirement)
    for requirement in WAN_RUNTIME_PACKAGES:
        ensure_package(requirement)
    if device == "cuda":
        ensure_package(
            "flash-attn==2.8.3",
            "--no-deps",
            "--no-build-isolation",
        )


def ensure_huggingface_hub() -> Any:
    if importlib.util.find_spec("huggingface_hub") is None:
        ensure_package("huggingface_hub")
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
    local_dir: Path,
    svdquant_repo: str | None,
) -> Path:
    local_dir = normalize_path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if model_weights_are_present(local_dir):
        model_root = local_dir
        print(f"Model weights already present: {model_root}")
    else:
        snapshot_download = ensure_huggingface_hub()
        download_args = {
            "repo_id": HF_MODEL_REPO,
            "allow_patterns": (*BASE_PATTERNS, STAGE2_PATTERN),
            "local_dir": local_dir,
        }
        print(f"Downloading {HF_MODEL_REPO} to {local_dir}")
        model_root = Path(snapshot_download(**download_args))

    if svdquant_repo:
        svdquant_root = model_root / "SolarWM-5B-sgf-stage2-81f"
        if svdquant_weights_are_present(model_root):
            print("SVDQuant weights already present; skipping private repository download")
        else:
            snapshot_download = ensure_huggingface_hub()
            svdquant_root.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {svdquant_repo} to {svdquant_root}")
            snapshot_download(repo_id=svdquant_repo, local_dir=svdquant_root)
    return model_root


def configure_interactive_config(model_root: Path) -> None:
    import yaml

    local_configs = [
        INTERACTIVE_CONFIG.with_name(f"{INTERACTIVE_CONFIG.stem}_xpu.local.yaml"),
        INTERACTIVE_CONFIG.with_name(f"{INTERACTIVE_CONFIG.stem}.local.yaml"),
    ]
    targets = [path for path in local_configs if path.exists()]
    if not targets:
        targets = [local_configs[-1]]

    for config_path in targets:
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        else:
            config = yaml.safe_load(INTERACTIVE_CONFIG.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            fail(f"SolarWM config must contain a mapping: {config_path}")

        model = config.setdefault("model", {})
        checkpoint = config.setdefault("checkpoint", {})
        if not isinstance(model, dict) or not isinstance(checkpoint, dict):
            fail(f"SolarWM config model/checkpoint sections must be mappings: {config_path}")
        model["base_path"] = str(model_root / "SolarWM-5B-base")
        checkpoint["path"] = str(model_root / "SolarWM-5B-sgf-stage2-81f")

        updated = yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
        if config_path.exists() and config_path.read_text(encoding="utf-8") == updated:
            print(f"SolarWM config already points to {model_root}: {config_path}")
            continue
        config_path.write_text(updated, encoding="utf-8")
        print(f"Updated SolarWM config: {config_path}")


def configure_comfyui(comfyui_root: Path, model_root: Path) -> None:
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
    model_root = normalize_path(args.local_dir)
    if not args.skip_weights:
        model_root = download_weights(args.local_dir, args.svdquant_repo)
    configure_interactive_config(model_root)
    if args.comfyui:
        configure_comfyui(normalize_path(args.comfyui), model_root)

    print("SolarWM installation complete.")
    if args.comfyui:
        print("Restart ComfyUI to load its updated extra_model_paths.yaml.")


if __name__ == "__main__":
    main()
