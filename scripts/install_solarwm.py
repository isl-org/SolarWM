#!/usr/bin/env python3
"""Install SolarWM's Wan2.2 runtime into the active Python environment."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile
from typing import Any, NoReturn


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
DATA_ROOT = Path.home() / "SolarWM-Data" / "releases-v1"
HF_MODEL_REPO = "junchaoh-cs/SolarWM"
DEFAULT_SVDQUANT_RELEASE_REPO = "isl-org/SolarWM"
DEFAULT_SVDQUANT_RELEASE_TAG = "ptc"
DEFAULT_SVDQUANT_VARIANT = "sdpa_k1"
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
    "av",
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
LEVEL_ZERO_VERSION = "1.34.0"
LEVEL_ZERO_SHA256 = "3050280cc94b7a5db7fb173391fcced9b7284ae196bcaf05a566d2911baa7733"
LEVEL_ZERO_URL = (
    "https://github.com/oneapi-src/level-zero/releases/download/v1.34.0/"
    "level-zero-win-sdk-1.34.0.zip"
)


def setup_triton_environment() -> None:
    """Install development files and a startup hook in this venv, not globally."""
    if os.name != "nt":
        return
    sdk = Path(sys.prefix) / "level-zero-sdk" / LEVEL_ZERO_VERSION
    required = (sdk / "include" / "level_zero" / "ze_api.h", sdk / "lib" / "ze_loader.lib")
    if not all(path.is_file() for path in required):
        with tempfile.TemporaryDirectory(prefix="solarwm-level-zero-") as temporary:
            archive = Path(temporary) / "sdk.zip"
            print(f"Downloading Level Zero Windows SDK {LEVEL_ZERO_VERSION}")
            certificate = os.environ.get("SSL_CERT_FILE") or os.environ.get("NODE_EXTRA_CA_CERTS")
            context = ssl.create_default_context(cafile=certificate or None)
            with urllib.request.urlopen(LEVEL_ZERO_URL, context=context, timeout=120) as response:
                with archive.open("wb") as output:
                    shutil.copyfileobj(response, output)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest != LEVEL_ZERO_SHA256:
                fail(f"Level Zero SDK checksum mismatch: {digest}")
            with zipfile.ZipFile(archive) as contents:
                for entry in contents.infolist():
                    target = (sdk / entry.filename).resolve()
                    if not target.is_relative_to(sdk.resolve()):
                        fail(f"Unsafe SDK archive entry: {entry.filename}")
                contents.extractall(sdk)
        if not all(path.is_file() for path in required):
            fail("Level Zero SDK extraction did not produce headers and ze_loader.lib")

    from importlib.metadata import distribution

    sycl_files = distribution("intel-sycl-rt").files or ()
    headers = [path.locate() for path in sycl_files if path.name == "sycl.hpp"]
    libraries = [path.locate() for path in sycl_files if path.name == "sycl.lib"]
    if not any(path.is_file() for path in headers) or not any(
        path.is_file() for path in libraries
    ):
        fail(
            "intel-sycl-rt lacks SYCL development files. Restore the matching wheel, "
            "or install the matching oneAPI DPC++ component and configure ONEAPI_ROOT."
        )
    print(f"SYCL headers: {next(path for path in headers if path.is_file())}")
    print(f"SYCL import library: {next(path for path in libraries if path.is_file())}")
    hook = Path(sysconfig.get_path("purelib")) / "solarwm_triton_environment.pth"
    hook.write_text(
        "import solarwm.runtime.triton_environment; "
        "solarwm.runtime.triton_environment.configure()\n",
        encoding="utf-8",
    )
    from solarwm.runtime.triton_environment import configure

    configure()
    print(f"Installed venv-local Triton environment hook: {hook}")


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
        "--svdquant-variant",
        default=DEFAULT_SVDQUANT_VARIANT,
        help=(
            f"SVDQuant variant to download from GitHub release (default: {DEFAULT_SVDQUANT_VARIANT}; "
            "options: sdpa_k1, sdpa_k2, sdpa_k4, sageattnv1_k1, sageattnv1_k2, sageattnv1_k4, none)"
        ),
    )
    parser.add_argument(
        "--svdquant-release-repo",
        default=DEFAULT_SVDQUANT_RELEASE_REPO,
        help=f"GitHub repository for SVDQuant release assets (default: {DEFAULT_SVDQUANT_RELEASE_REPO})",
    )
    parser.add_argument(
        "--svdquant-release-tag",
        default=DEFAULT_SVDQUANT_RELEASE_TAG,
        help=f"GitHub release tag for SVDQuant assets (default: {DEFAULT_SVDQUANT_RELEASE_TAG})",
    )
    parser.add_argument(
        "--svdquant-repo",
        help="Optional private Hugging Face repo containing a svdquant_int4/ directory (fallback)",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Install runtime dependencies without downloading model weights",
    )
    parser.add_argument(
        "--build-kernels",
        action="store_true",
        help="Build the optional Intel XPU SVDQuant and oneDNN kernels",
    )
    parser.add_argument(
        "--gpu-arch",
        default="bmg-g31",
        help="Intel GPU architecture passed to the SVDQuant kernel build",
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


def svdquant_kernel_is_present() -> bool:
    package_dir = REPO_ROOT / "packages" / "wan22_svdquant_dense_block"
    return any(
        path.is_file()
        for suffix in (".pyd", ".so", ".dll")
        for path in package_dir.glob(f"svdquant_dense_kernel_*{suffix}")
    )


def build_svdquant_kernel(gpu_arch: str) -> None:
    package_dir = REPO_ROOT / "packages" / "wan22_svdquant_dense_block"
    build_script = package_dir / "build.py"
    if svdquant_kernel_is_present():
        print("SVDQuant SYCL kernel already built; skipping")
        return
    if not build_script.is_file():
        fail(f"SVDQuant kernel build script is missing: {build_script}")
    print(f"Building SVDQuant SYCL kernel for {gpu_arch}")
    subprocess.run(
        [sys.executable, str(build_script), "--gpu-arch", gpu_arch],
        cwd=package_dir,
        check=True,
    )
    if not svdquant_kernel_is_present():
        fail("SVDQuant kernel build completed without producing a native extension")


def build_qconv_kernel() -> None:
    print("Building or loading the oneDNN qconv XPU kernel")
    try:
        from solarwm.kernels.qconv_onednn import is_available

        if is_available():
            print("oneDNN qconv XPU kernel is available")
        else:
            print(
                "Warning: oneDNN qconv XPU kernel is unavailable; "
                "the native PyTorch oneDNN fallback will be used"
            )
    except Exception as exc:
        print(f"Warning: oneDNN qconv kernel check failed; using fallback: {exc}")


def build_sageattn_v2_kernel(gpu_arch: str) -> None:
    print(f"Building or verifying SageAttention v2 SYCL kernel for {gpu_arch}")
    try:
        from solarwm.kernels.fused_rope_prope_sage.sageattn_v2 import ensure_sageattn_v2

        if ensure_sageattn_v2(gpu_arch=gpu_arch):
            print("SageAttention v2 SYCL kernel is ready")
        else:
            print("Warning: SageAttention v2 SYCL kernel build failed")
    except Exception as exc:
        print(f"Warning: SageAttention v2 SYCL kernel check failed: {exc}")


def build_xpu_kernels(device: str, gpu_arch: str) -> None:
    if device != "xpu":
        print(f"Skipping Intel XPU kernel builds for detected device: {device}")
        return
    build_svdquant_kernel(gpu_arch)
    build_sageattn_v2_kernel(gpu_arch)
    build_qconv_kernel()


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


def svdquant_weights_are_present(model_root: Path, variant: str | None = None) -> bool:
    root = model_root / "SolarWM-5B-sgf-stage2-81f" / "svdquant_int4"
    if not root.is_dir():
        return False
    if variant and variant.lower() not in ("none", "skip", "all"):
        v_dir = root / variant
        if not v_dir.is_dir() or not (v_dir / "manifest.json").is_file():
            return False
        return any(v_dir.glob("*.safetensors"))
    return any(root.rglob("*.safetensors"))


def download_file_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(f"{dest.suffix}.part")
    req = urllib.request.Request(url, headers={"User-Agent": "SolarWM-Installer"})
    with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8 * 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = downloaded * 100.0 / total
                print(
                    f"\rDownloading {dest.name}: {downloaded / (1024*1024):.1f}/{total / (1024*1024):.1f} MB ({percent:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\rDownloading {dest.name}: {downloaded / (1024*1024):.1f} MB",
                    end="",
                    flush=True,
                )
    print()
    tmp_path.replace(dest)


def download_svdquant_release_variant(
    model_root: Path,
    variant: str = DEFAULT_SVDQUANT_VARIANT,
    repo: str = DEFAULT_SVDQUANT_RELEASE_REPO,
    tag: str = DEFAULT_SVDQUANT_RELEASE_TAG,
) -> Path:
    variant_dir = model_root / "SolarWM-5B-sgf-stage2-81f" / "svdquant_int4" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = variant_dir / "manifest.json"

    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            shards = manifest_data.get("shards", [])
            if shards and all((variant_dir / s).is_file() and (variant_dir / s).stat().st_size > 0 for s in shards):
                print(f"SVDQuant {variant} weights already present in {variant_dir}")
                return variant_dir
        except Exception:
            pass

    manifest_url = f"https://github.com/{repo}/releases/download/{tag}/{variant}_manifest.json"
    print(f"Downloading SVDQuant {variant} manifest from {manifest_url}")
    download_file_with_progress(manifest_url, manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    shards = manifest_data.get("shards", [])
    if not shards:
        fail(f"No shards found in manifest for SVDQuant {variant}")

    for shard in shards:
        shard_dest = variant_dir / shard
        if shard_dest.is_file() and shard_dest.stat().st_size > 0:
            print(f"Shard {shard} already exists, skipping")
            continue
        asset_name = f"{variant}_{shard}"
        shard_url = f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"
        print(f"Downloading SVDQuant asset {asset_name} -> {shard_dest.name}")
        download_file_with_progress(shard_url, shard_dest)

    print(f"SVDQuant {variant} weights downloaded to {variant_dir}")
    return variant_dir


def download_weights(
    local_dir: Path,
    svdquant_repo: str | None = None,
    svdquant_variant: str | None = DEFAULT_SVDQUANT_VARIANT,
    svdquant_release_repo: str = DEFAULT_SVDQUANT_RELEASE_REPO,
    svdquant_release_tag: str = DEFAULT_SVDQUANT_RELEASE_TAG,
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
    elif svdquant_variant and svdquant_variant.lower() not in ("none", "skip"):
        download_svdquant_release_variant(
            model_root,
            variant=svdquant_variant,
            repo=svdquant_release_repo,
            tag=svdquant_release_tag,
        )
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
        ema_root = model_root / "SolarWM-5B-sgf-stage2-81f-ema-bf16"
        checkpoint["path"] = str(
            ema_root if (ema_root / "COMPLETE.json").is_file()
            else model_root / "SolarWM-5B-sgf-stage2-81f"
        )

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
    if device == "xpu":
        setup_triton_environment()
    if args.build_kernels:
        build_xpu_kernels(device, args.gpu_arch)
    model_root = normalize_path(args.local_dir)
    if not args.skip_weights:
        model_root = download_weights(
            args.local_dir,
            svdquant_repo=args.svdquant_repo,
            svdquant_variant=args.svdquant_variant,
            svdquant_release_repo=args.svdquant_release_repo,
            svdquant_release_tag=args.svdquant_release_tag,
        )
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "prepare_stage2_ema.py"),
             str(model_root / "SolarWM-5B-sgf-stage2-81f"),
             str(model_root / "SolarWM-5B-sgf-stage2-81f-ema-bf16")],
            check=True,
        )
    configure_interactive_config(model_root)
    if args.comfyui:
        configure_comfyui(normalize_path(args.comfyui), model_root)

    print("SolarWM installation complete.")
    if args.comfyui:
        print("Restart ComfyUI to load its updated extra_model_paths.yaml.")


if __name__ == "__main__":
    main()
