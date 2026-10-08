#!/usr/bin/env python3
"""Compare Wan 2.2 VAE decode across device/dtype configs against a CPU fp32 oracle.

Answers, without needing the CUDA host:
  * Is XPU decode deterministic (same inputs twice -> same pixels)?
  * Does XPU decode match a device-independent CPU fp32 reference?
  * Is the gap caused by bf16 (reproducible on CPU bf16) or by XPU kernels?
  * Which primitive (conv3d / F.normalize / nearest-exact upsample / SiLU) diverges?

Latents are cropped so CPU fp32 stays cheap; the VAE is convolutional so a crop is
still a valid cross-device comparison (both sides see the identical crop).

Usage
-----
  python scripts/debug/wan22_vae_decode_device_probe.py --device xpu \
    --latents outputs/wan22-cuda-decode-reference/latents_bf16.pt \
    --latent-frames 2 --crop 16 --output outputs/debug/vae-device-probe/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solarwm.backends.wan22.runtime.components import Wan5BVAE  # noqa: E402
from solarwm.backends.wan22.runtime.modeling import vae as vae_module  # noqa: E402

Z_DIM = 48


def _patch_vae_attention() -> None:
    """Replace the VAE AttentionBlock's SDPA call with explicit fp32 softmax matmul."""

    import math

    from einops import rearrange

    def forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        scale = 1.0 / math.sqrt(int(q.shape[-1]))
        weights = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
        weights = weights.softmax(dim=-1)
        x = torch.matmul(weights, v.float()).to(v.dtype)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity

    vae_module.AttentionBlock.forward = forward


def _pick_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda", 0)
    if name == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise SystemExit("xpu requested but torch.xpu is not available")
        return torch.device("xpu", 0)
    if name == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"unknown device: {name}")


def _vae_weights() -> Path:
    model_root = Path(os.environ.get("SOLAR_MODEL_ROOT", REPO_ROOT / "models/SolarWM")).expanduser()
    path = model_root / "SolarWM-5B-base" / "vae" / "Wan2.2_VAE.pth"
    if not path.is_file():
        raise SystemExit(f"missing VAE weights at {path} (set SOLAR_MODEL_ROOT)")
    return path


def _fixed_latents(latent_frames: int, height: int, width: int) -> torch.Tensor:
    """Deterministic [1, T, C, H, W] latents (no RNG) as a fallback input."""

    elements = latent_frames * Z_DIM * height * width
    base = torch.arange(elements, dtype=torch.float32)
    pattern = torch.sin(base * 0.011) + 0.3 * torch.cos(base * 0.005)
    return pattern.view(1, latent_frames, Z_DIM, height, width).to(torch.bfloat16)


def _compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()
    if ref.shape != cand.shape:
        return {"ok": False, "reason": f"shape {list(cand.shape)} != {list(ref.shape)}"}
    diff = (ref - cand).abs()
    per_frame = diff.flatten(2).mean(dim=-1).flatten().tolist() if diff.ndim >= 3 else []
    return {
        "ok": True,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "candidate_mean": float(cand.mean().item()),
        "reference_mean": float(ref.mean().item()),
        "per_frame_mean_abs_diff": [round(value, 5) for value in per_frame],
    }


def _decode(
    vae: Wan5BVAE,
    latents_btchw: torch.Tensor,
    *,
    device: torch.device,
    weight_dtype: torch.dtype,
    autocast: bool,
    master_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    vae.module.load_state_dict(master_state)
    vae.to(device, dtype=weight_dtype)
    latents = latents_btchw.to(device=device, dtype=weight_dtype)
    clips = latents.permute(0, 2, 1, 3, 4)
    outputs = []
    with torch.no_grad():
        for clip in clips:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast,
            ):
                decoded = vae.module.decode(clip.unsqueeze(0), vae._scale(clip))
            outputs.append(decoded.float().clamp_(-1, 1).squeeze(0))
    return torch.stack(outputs, dim=0).permute(0, 2, 1, 3, 4).cpu()


def _primitive_probes(device: torch.device) -> dict[str, Any]:
    """Fixed-input single-op comparisons: XPU/CUDA kernel vs CPU fp64 reference."""

    results: dict[str, Any] = {}

    # 1. F.normalize over channels (RMS_norm in vae.py uses this).
    channels = 384
    elements = channels * 8 * 8
    base = torch.arange(elements, dtype=torch.float32)
    pattern = (torch.sin(base * 0.017) * 3.0).view(1, channels, 1, 8, 8)
    exact = F.normalize(pattern.double(), dim=1)
    for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        value = F.normalize(pattern.to(device=device, dtype=dtype), dim=1)
        results[f"F_normalize_dim1_{label}"] = _compare(exact.float(), value.float())

    # 2. Causal-style conv3d.
    conv = torch.nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=True)
    with torch.no_grad():
        weight_elements = conv.weight.numel()
        conv.weight.copy_(
            torch.cos(torch.arange(weight_elements, dtype=torch.float32) * 0.003).view_as(
                conv.weight
            )
            * 0.05
        )
        conv.bias.copy_(torch.linspace(-0.1, 0.1, conv.bias.numel()))
    x_elements = 64 * 4 * 16 * 16
    x = torch.sin(torch.arange(x_elements, dtype=torch.float32) * 0.021).view(1, 64, 4, 16, 16)
    with torch.no_grad():
        exact_conv = conv.double()(x.double())
        conv.float()
        for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            module = conv.to(device=device, dtype=dtype)
            value = module(x.to(device=device, dtype=dtype))
            results[f"conv3d_{label}"] = _compare(exact_conv.float(), value.float())
            conv.to(device="cpu", dtype=torch.float32)

    # 3. nearest-exact upsample (Upsample in vae.py casts to float first).
    up_elements = 32 * 2 * 8 * 8
    up_in = torch.cos(torch.arange(up_elements, dtype=torch.float32) * 0.03).view(1, 32 * 2, 8, 8)
    exact_up = F.interpolate(up_in.double(), scale_factor=(2.0, 2.0), mode="nearest-exact")
    for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        source = up_in.to(device=device, dtype=dtype)
        value = F.interpolate(source.float(), scale_factor=(2.0, 2.0), mode="nearest-exact").to(
            dtype
        )
        results[f"interpolate_nearest_exact_{label}"] = _compare(exact_up.float(), value.float())

    # 4. SiLU.
    silu_in = torch.linspace(-8.0, 8.0, 4096)
    exact_silu = F.silu(silu_in.double())
    for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        value = F.silu(silu_in.to(device=device, dtype=dtype))
        results[f"silu_{label}"] = _compare(exact_silu.float(), value.float())
    return results


def _conv_sweep(device: torch.device) -> dict[str, Any]:
    """Fixed-input conv sweep over the decoder's real channel/spatial shapes."""

    results: dict[str, Any] = {}
    cases_2d = [
        (384, 60, 108),
        (384, 120, 216),
        (192, 240, 432),
        (96, 480, 864),
        (384, 48, 48),
        (384, 96, 96),
        (192, 192, 192),
        (96, 192, 192),
    ]
    for channels, height, width in cases_2d:
        conv = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        with torch.no_grad():
            conv.weight.copy_(
                torch.cos(torch.arange(conv.weight.numel(), dtype=torch.float32) * 0.0007).view_as(
                    conv.weight
                )
                * 0.02
            )
            conv.bias.zero_()
            x = torch.sin(
                torch.arange(channels * height * width, dtype=torch.float32) * 0.0031
            ).view(1, channels, height, width)
            reference = conv(x)
            for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
                module = conv.to(device=device, dtype=dtype)
                value = module(x.to(device=device, dtype=dtype)).float().cpu()
                key = f"conv2d_c{channels}_{height}x{width}_{label}"
                results[key] = _compare(reference, value)
                results[key].pop("per_frame_mean_abs_diff", None)
                conv.to(device="cpu", dtype=torch.float32)
    cases_3d = [
        (384, 1, 60, 108),
        (384, 2, 60, 108),
        (192, 1, 120, 216),
        (96, 1, 240, 432),
        (384, 1, 24, 24),
        (384, 1, 48, 48),
    ]
    for channels, frames, height, width in cases_3d:
        conv = torch.nn.Conv3d(channels, channels, 3, padding=1, bias=True)
        with torch.no_grad():
            conv.weight.copy_(
                torch.cos(torch.arange(conv.weight.numel(), dtype=torch.float32) * 0.0005).view_as(
                    conv.weight
                )
                * 0.01
            )
            conv.bias.zero_()
            x = torch.sin(
                torch.arange(channels * frames * height * width, dtype=torch.float32) * 0.0023
            ).view(1, channels, frames, height, width)
            reference = conv(x)
            for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
                module = conv.to(device=device, dtype=dtype)
                value = module(x.to(device=device, dtype=dtype)).float().cpu()
                key = f"conv3d_c{channels}_{frames}x{height}x{width}_{label}"
                results[key] = _compare(reference, value)
                results[key].pop("per_frame_mean_abs_diff", None)
                conv.to(device="cpu", dtype=torch.float32)
    return results


def _sdpa_sweep(device: torch.device) -> dict[str, Any]:
    """Probe the VAE ``AttentionBlock`` geometry: 1 head, seq=h*w, head_dim=channels."""

    results: dict[str, Any] = {}
    shapes = [
        (16, 16),
        (20, 20),
        (24, 24),
        (28, 28),
        (30, 30),
        (30, 54),
        (60, 108),
    ]
    # 640 is the real Wan 2.2 VAE middle-block width (dim=160, dim_mult[-1]=4).
    for channels in (384, 640, 768):
        for height, width in shapes:
            seq = height * width
            elements = seq * channels
            base = torch.arange(elements, dtype=torch.float32)
            q = (torch.sin(base * 0.0013) * 0.5).view(1, 1, seq, channels)
            k = (torch.cos(base * 0.0017) * 0.5).view(1, 1, seq, channels)
            v = (torch.sin(base * 0.0011 + 1.0) * 0.5).view(1, 1, seq, channels)
            reference = F.scaled_dot_product_attention(q.double(), k.double(), v.double())
            for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
                value = F.scaled_dot_product_attention(
                    q.to(device=device, dtype=dtype),
                    k.to(device=device, dtype=dtype),
                    v.to(device=device, dtype=dtype),
                )
                key = f"sdpa_c{channels}_seq{seq}_{height}x{width}_{label}"
                results[key] = _compare(reference.float(), value.float())
                results[key].pop("per_frame_mean_abs_diff", None)
    return results


def _matmul_sweep(device: torch.device) -> dict[str, Any]:
    """Raw batched GEMM at the VAE attention shapes: [1,1,S,C] @ [1,1,C,S] and [1,1,S,S] @ [1,1,S,C]."""

    results: dict[str, Any] = {}
    for channels in (640, 768):
        for seq in (256, 400, 576, 784, 900, 1620):
            a = (
                torch.sin(torch.arange(seq * channels, dtype=torch.float32) * 0.0013) * 0.5
            ).view(1, 1, seq, channels)
            b = (
                torch.cos(torch.arange(seq * channels, dtype=torch.float32) * 0.0017) * 0.5
            ).view(1, 1, seq, channels)
            scores_ref = torch.matmul(a.double(), b.double().transpose(-1, -2))
            for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
                value = torch.matmul(
                    a.to(device=device, dtype=dtype),
                    b.to(device=device, dtype=dtype).transpose(-1, -2),
                )
                key = f"matmul_qk_c{channels}_seq{seq}_{label}"
                results[key] = _compare(scores_ref.float(), value.float())
                results[key].pop("per_frame_mean_abs_diff", None)
            probs = torch.softmax(scores_ref / (channels**0.5), dim=-1)
            av_ref = torch.matmul(probs, b.double())
            for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
                value = torch.matmul(
                    probs.to(device=device, dtype=dtype),
                    b.to(device=device, dtype=dtype),
                )
                key = f"matmul_av_c{channels}_seq{seq}_{label}"
                results[key] = _compare(av_ref.float(), value.float())
                results[key].pop("per_frame_mean_abs_diff", None)
    return results


def _softmax_sweep(device: torch.device) -> dict[str, Any]:
    """Softmax over the last dim at the VAE attention score shapes [1, 1, S, S]."""

    results: dict[str, Any] = {}
    for seq in (256, 400, 576, 784, 900, 1620):
        scores = (
            torch.sin(torch.arange(seq * seq, dtype=torch.float32) * 0.0007) * 4.0
        ).view(1, 1, seq, seq)
        reference = torch.softmax(scores.double(), dim=-1)
        for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            value = torch.softmax(scores.to(device=device, dtype=dtype), dim=-1)
            key = f"softmax_seq{seq}_{label}"
            results[key] = _compare(reference.float(), value.float())
            results[key].pop("per_frame_mean_abs_diff", None)
    return results


def _normalize_sweep(device: torch.device) -> dict[str, Any]:
    """Probe ``RMS_norm``'s ``F.normalize(x, dim=1)`` at the decoder's real shapes."""

    results: dict[str, Any] = {}
    cases = [
        (384, 1, 16, 16),
        (384, 1, 24, 24),
        (384, 1, 30, 54),
        (384, 1, 60, 108),
        (192, 1, 120, 216),
        (96, 1, 240, 432),
        (96, 1, 480, 864),
    ]
    for channels, frames, height, width in cases:
        elements = channels * frames * height * width
        base = torch.arange(elements, dtype=torch.float32)
        x = (torch.sin(base * 0.0019) * 3.0).view(1, channels, frames, height, width)
        reference = F.normalize(x.double(), dim=1)
        for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            value = F.normalize(x.to(device=device, dtype=dtype), dim=1)
            key = f"normalize_c{channels}_{frames}x{height}x{width}_{label}"
            results[key] = _compare(reference.float(), value.float())
            results[key].pop("per_frame_mean_abs_diff", None)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "xpu", "cpu"), required=True)
    parser.add_argument(
        "--latents",
        type=Path,
        default=None,
        help="Optional [1,T,C,H,W] latents_bf16.pt; fixed synthetic latents when omitted",
    )
    parser.add_argument("--latent-frames", type=int, default=2)
    parser.add_argument("--crop", type=int, default=16, help="Crop latent H/W to this size (0=off)")
    parser.add_argument("--skip-cpu-fp32", action="store_true")
    parser.add_argument("--skip-primitives", action="store_true")
    parser.add_argument(
        "--conv-sweep",
        action="store_true",
        help="Only run the conv2d/conv3d shape sweep (skips VAE decode entirely)",
    )
    parser.add_argument(
        "--op-sweep",
        action="store_true",
        help="Only run SDPA + F.normalize shape sweeps (skips VAE decode entirely)",
    )
    parser.add_argument(
        "--patch-vae-attention",
        action="store_true",
        help="Swap the VAE AttentionBlock SDPA call for explicit fp32 softmax matmul",
    )
    parser.add_argument(
        "--cuda-decode",
        type=Path,
        default=None,
        help="CUDA pixel tensor [1,F,3,H,W]; compared against the leading 4T-3 frames "
        "(decode is causal in time, so a T-latent prefix decodes those frames exactly)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = _pick_device(args.device)
    if args.patch_vae_attention:
        _patch_vae_attention()

    if args.op_sweep:
        sweep = {
            "schema": "solarwm.wan22-vae-op-sweep.v1",
            "device": device.type,
            "sdpa_sweep": _sdpa_sweep(device),
            "matmul_sweep": _matmul_sweep(device),
            "softmax_sweep": _softmax_sweep(device),
            "normalize_sweep": _normalize_sweep(device),
        }
        text = json.dumps(sweep, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        print(text)
        return 0

    if args.conv_sweep:
        sweep = {
            "schema": "solarwm.wan22-vae-conv-sweep.v1",
            "device": device.type,
            "conv_sweep": _conv_sweep(device),
        }
        text = json.dumps(sweep, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        print(text)
        return 0

    if args.latents is not None:
        latents = torch.load(args.latents.expanduser().resolve(), map_location="cpu", weights_only=True)
        source = str(args.latents)
    else:
        latents = _fixed_latents(args.latent_frames, 30, 54)
        source = "fixed_synthetic"
    latents = latents[:, : args.latent_frames]
    if args.crop:
        latents = latents[..., : args.crop, : args.crop]
    latents = latents.contiguous().to(torch.bfloat16)

    vae = Wan5BVAE(_vae_weights())
    master_state = {key: value.detach().clone() for key, value in vae.module.state_dict().items()}

    report: dict[str, Any] = {
        "schema": "solarwm.wan22-vae-decode-device-probe.v1",
        "device": device.type,
        "latents_source": source,
        "latents_shape": list(latents.shape),
        "patched_vae_attention": bool(args.patch_vae_attention),
        "configs": {},
    }

    configs: list[tuple[str, torch.device, torch.dtype, bool]] = []
    if not args.skip_cpu_fp32:
        configs.append(("cpu_fp32_no_autocast", torch.device("cpu"), torch.float32, False))
        configs.append(("cpu_bf16_autocast", torch.device("cpu"), torch.bfloat16, True))
    configs.append((f"{device.type}_bf16_autocast", device, torch.bfloat16, True))
    configs.append((f"{device.type}_bf16_autocast_repeat", device, torch.bfloat16, True))
    configs.append((f"{device.type}_fp32_no_autocast", device, torch.float32, False))

    pixels: dict[str, torch.Tensor] = {}
    for name, cfg_device, cfg_dtype, cfg_autocast in configs:
        try:
            pixels[name] = _decode(
                vae,
                latents,
                device=cfg_device,
                weight_dtype=cfg_dtype,
                autocast=cfg_autocast,
                master_state=master_state,
            )
            report["configs"][name] = {
                "ok": True,
                "shape": list(pixels[name].shape),
                "mean": float(pixels[name].mean().item()),
                "std": float(pixels[name].std().item()),
            }
        except Exception as exc:  # noqa: BLE001 - probe should report, not crash
            report["configs"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    oracle = "cpu_fp32_no_autocast" if "cpu_fp32_no_autocast" in pixels else None
    comparisons: dict[str, Any] = {}
    if oracle is not None:
        for name, tensor in pixels.items():
            if name == oracle:
                continue
            comparisons[f"{name}_vs_cpu_fp32"] = _compare(pixels[oracle], tensor)
    device_bf16 = f"{device.type}_bf16_autocast"
    repeat = f"{device_bf16}_repeat"
    if device_bf16 in pixels and repeat in pixels:
        comparisons["determinism_same_device_two_runs"] = _compare(
            pixels[device_bf16],
            pixels[repeat],
        )
    if device_bf16 in pixels and "cpu_bf16_autocast" in pixels:
        comparisons[f"{device_bf16}_vs_cpu_bf16"] = _compare(
            pixels["cpu_bf16_autocast"],
            pixels[device_bf16],
        )
    if args.cuda_decode is not None:
        cuda_path = args.cuda_decode.expanduser().resolve()
        if not cuda_path.is_file():
            comparisons["vs_cuda_decode"] = {"ok": False, "reason": f"missing {cuda_path}"}
        else:
            cuda_pixels = torch.load(cuda_path, map_location="cpu", weights_only=True)
            frames = next(iter(pixels.values())).shape[1] if pixels else 0
            cuda_prefix = cuda_pixels[:, :frames]
            report["cuda_decode_file"] = cuda_path.name
            report["cuda_decode_full_shape"] = list(cuda_pixels.shape)
            for name, tensor in pixels.items():
                if name.endswith("_repeat"):
                    continue
                comparisons[f"{name}_vs_cuda_decode"] = _compare(cuda_prefix, tensor)
    report["comparisons"] = comparisons

    if not args.skip_primitives:
        report["primitive_probes"] = _primitive_probes(device)

    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
