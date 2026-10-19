"""Default file-based bootstrap for the ComfyUI session manager."""

from __future__ import annotations

from collections.abc import Mapping
import os
from os.path import expandvars
from pathlib import Path
import threading
from typing import Any


_ATTENTION_BACKENDS = {"sdpa", "sageattnv1", "sageattnv2"}
_LINEAR_QUANTS = {"full_precision", "svdquant_k1", "svdquant_k2", "svdquant_k4"}
_VAE_QUANTS = {"full_precision", "quarot_w8a8"}

_CACHE_LOCK = threading.Lock()
_CACHED_PROVIDER: Any | None = None
_CACHED_PROVIDER_KEY: tuple[Any, ...] | None = None


def _resolve_repo_path(value: str) -> str:
    expanded = expandvars(value)
    if os.name == "nt" and expanded.startswith("/") and not expanded.startswith("//"):
        return expanded
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return str(path)


def resolve_image_path(image_value: str | Path) -> Path:
    """Resolve a ComfyUI input image or a SolarWM repository asset."""
    repo_root = Path(__file__).resolve().parents[2]
    image_name = str(image_value)
    for annotation in (" [input]", " [output]", " [temp]"):
        if image_name.endswith(annotation):
            image_name = image_name[: -len(annotation)]
            break
    image_path = Path(image_name).expanduser()
    if image_path.is_absolute():
        return image_path
    try:
        import folder_paths

        comfy_image_path = Path(
            folder_paths.get_annotated_filepath(str(image_value))
        )
        if comfy_image_path.exists():
            return comfy_image_path
    except (ImportError, OSError, ValueError, AttributeError):
        pass
    repo_image_path = repo_root / image_path
    return repo_image_path if repo_image_path.exists() else image_path


def load_dataset_camera_metadata(image_value: str | Path) -> dict[str, Any] | None:
    """Load prompt and trajectory sidecars associated with a selected image."""
    import numpy as np

    image_path = resolve_image_path(image_value)
    prompt_path = image_path.with_suffix(".prompt.txt")
    camera_path = image_path.with_suffix(".camera.npz")
    if not prompt_path.is_file() or not camera_path.is_file():
        return None
    with np.load(camera_path, allow_pickle=False) as values:
        trajectory = np.asarray(values["c2w"], dtype=np.float64)
        fps = float(np.asarray(values.get("fps", 16.0)).item())
        sample_id = str(np.asarray(values.get("sample_id", "")).item())
    return {
        "prompt": prompt_path.read_text(encoding="utf-8").strip(),
        "trajectory": trajectory,
        "fps": fps,
        "sample_id": sample_id,
        "image_path": str(image_path),
    }


def _interactive_config(payload: Mapping[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Apply ComfyUI-only runtime choices without rewriting the YAML file."""

    model = config.setdefault("model", {})
    runtime = config.setdefault("runtime", {})
    if not isinstance(model, dict) or not isinstance(runtime, dict):
        raise ValueError("interactive config model/runtime sections must be mappings")

    base_path = str(model.get("base_path", "")).strip()
    if base_path and not base_path.startswith("/path/to/"):
        model["base_path"] = _resolve_repo_path(base_path)
    checkpoint = config.get("checkpoint", {})
    if isinstance(checkpoint, dict):
        checkpoint_path = str(checkpoint.get("path", "")).strip()
        if checkpoint_path:
            checkpoint["path"] = _resolve_repo_path(checkpoint_path)

    attention_backend = str(payload.get("attention_backend", "sdpa")).strip().lower()
    if attention_backend not in _ATTENTION_BACKENDS:
        raise ValueError(f"unsupported attention_backend: {attention_backend}")
    linear_quant = str(payload.get("linear_quant", "full_precision")).strip().lower()
    if linear_quant not in _LINEAR_QUANTS:
        raise ValueError(f"unsupported linear_quant: {linear_quant}")
    stage2_vae_quant = str(payload.get("stage2_vae_quant", "full_precision")).strip().lower()
    if stage2_vae_quant not in _VAE_QUANTS:
        raise ValueError(f"unsupported stage2_vae_quant: {stage2_vae_quant}")

    model["attention_backend"] = attention_backend
    model["linear_quant"] = linear_quant
    runtime["stage2_vae_quant"] = stage2_vae_quant

    resolution = str(payload.get("resolution", "")).strip().lower()
    if "360p" in resolution:
        data = config.setdefault("data", {})
        data["height"] = 320
        data["width"] = 576
        data["latent_shape"] = [21, 48, 20, 36]
        model["frame_sequence_length"] = 180
    elif "480p" in resolution:
        data = config.setdefault("data", {})
        data["height"] = 480
        data["width"] = 864
        data["latent_shape"] = [21, 48, 30, 54]
        model["frame_sequence_length"] = 405

    weights_root = str(payload.get("svdquant_weights_root", "")).strip()
    if weights_root:
        model["svdquant_weights_root"] = _resolve_repo_path(weights_root)
    elif linear_quant != "full_precision":
        checkpoint_path = Path(str(config.get("checkpoint", {}).get("path", ""))).expanduser()
        candidate = checkpoint_path / "svdquant_int4"
        if not candidate.exists() and (
            checkpoint_path.parent / "SolarWM-5B-sgf-stage2-81f" / "svdquant_int4"
        ).exists():
            candidate = checkpoint_path.parent / "SolarWM-5B-sgf-stage2-81f" / "svdquant_int4"
        model["svdquant_weights_root"] = str(candidate)
    return config


def get_cached_provider(payload: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Retrieve or construct a cached Stage2 provider retained in VRAM."""
    global _CACHED_PROVIDER, _CACHED_PROVIDER_KEY

    import torch
    import yaml
    from solarwm.backends.wan22.runtime.stage2 import build_stage2_generation_provider

    raw_config_path = Path(str(payload["config_path"])).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    if not raw_config_path.is_absolute() and not raw_config_path.exists():
        if (repo_root / raw_config_path).exists():
            config_path = repo_root / raw_config_path
        else:
            config_path = raw_config_path
    else:
        config_path = raw_config_path

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("SolarWM interactive config must be a mapping")

    # If the selected config has dummy placeholder paths but a local sibling exists, prefer the local one
    base_path = str(config.get("model", {}).get("base_path", ""))
    if base_path.startswith("/path/to/"):
        for suffix in ("_xpu.local.yaml", ".local.yaml"):
            candidate = config_path.parent / (config_path.stem + suffix)
            if candidate.exists():
                config = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                break

    config = _interactive_config(payload, config)
    cache_key = (
        str(config_path.resolve()) if config_path.exists() else str(config_path),
        str(config.get("model", {}).get("base_path", "")),
        str(config.get("checkpoint", {}).get("path", "")),
        str(config.get("model", {}).get("attention_backend", "")),
        str(config.get("model", {}).get("linear_quant", "")),
        str(config.get("model", {}).get("svdquant_weights_root", "")),
        str(config.get("runtime", {}).get("stage2_vae_quant", "")),
        int(config.get("data", {}).get("height", 480)),
        int(config.get("data", {}).get("width", 864)),
    )

    with _CACHE_LOCK:
        if _CACHED_PROVIDER is not None and _CACHED_PROVIDER_KEY == cache_key:
            return _CACHED_PROVIDER, config

        if _CACHED_PROVIDER is not None:
            try:
                _CACHED_PROVIDER.close()
            except Exception:
                pass
            _CACHED_PROVIDER = None
            _CACHED_PROVIDER_KEY = None
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        provider = build_stage2_generation_provider(config)
        try:
            provider._load_role("model")
        except Exception:
            try:
                provider.close()
            except Exception:
                pass
            raise
        _CACHED_PROVIDER = provider
        _CACHED_PROVIDER_KEY = cache_key
        return provider, config


def clear_cached_provider() -> None:
    """Explicitly release the cached Stage2 model provider from VRAM."""
    global _CACHED_PROVIDER, _CACHED_PROVIDER_KEY

    with _CACHE_LOCK:
        provider = _CACHED_PROVIDER
        _CACHED_PROVIDER = None
        _CACHED_PROVIDER_KEY = None
        if provider is not None:
            try:
                provider.close()
            except Exception:
                pass
        try:
            import torch
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def is_provider_cached() -> bool:
    """Return True if a model provider is currently resident in VRAM."""
    with _CACHE_LOCK:
        return _CACHED_PROVIDER is not None


def build_provider_inputs(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load or reuse a cached Stage2 provider and encode one TI2V anchor from a local image.

    ComfyUI normally supplies image tensors to nodes, but the interactive
    session is intentionally outside the prompt queue.  A local image path
    keeps the HTTP route simple and avoids copying a full image through JSON.
    Deployments that already have tensors can replace this factory with
    ``configure_provider_factory``.
    """

    import torch
    from PIL import Image

    provider, config = get_cached_provider(payload)
    image_val = payload.get("initial_image")
    if not image_val:
        raise ValueError("interactive session requires initial_image")
    image_path = resolve_image_path(str(image_val))
    image = Image.open(image_path).convert("RGB")
    image = image.resize(
        (int(config["data"]["width"]), int(config["data"]["height"])),
        Image.Resampling.BICUBIC,
    )
    pixels = (
        torch.from_numpy(__import__("numpy").asarray(image).copy())
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
        prompt = str(payload.get("prompt", ""))
        if str(payload.get("camera_motion", "")) == "dataset_trajectory":
            dataset_metadata = load_dataset_camera_metadata(str(image_val))
            if dataset_metadata is not None:
                prompt = dataset_metadata["prompt"]
        condition = provider.text_encoder([prompt])
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


__all__ = [
    "build_provider_inputs",
    "clear_cached_provider",
    "get_cached_provider",
    "is_provider_cached",
]
