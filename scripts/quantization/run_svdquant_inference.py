"""Inference evaluation of SolarWM Stage2 DiT using SVDQuant INT4 weights across attention backends and hardware."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

# Add local package and quantization directory to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "quantization"))

from scripts.quantization.svdquant_int4.attn_dispatch import with_attention_backend
from scripts.quantization.svdquant_int4.quantizer import unpack_int4_signed


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def get_peak_mem_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**3)
    elif device.type == "xpu":
        return torch.xpu.max_memory_allocated(device) / (1024**3)
    return 0.0


def reset_peak_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "xpu":
        torch.xpu.reset_peak_memory_stats(device)


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()


from solarwm.backends.wan22.runtime.modeling.svdquant_linear import (
    CUDASVDQuantLinear,
    SVDQuantLinear,
    SVDQuantModelManager,
    SYCLSVDQuantLinear,
    TimestepTrackingDiffusionWrapper,
)


class _NoTextEncoder:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def to(self, *args: Any, **kwargs: Any) -> _NoTextEncoder:
        return self

    def __call__(self, prompts: Sequence[str]) -> Mapping[str, Any]:
        raise RuntimeError("run_svdquant_inference uses preencoded prompt embeddings only")


def iter_latent_samples(tars: Sequence[Path], *, per_shard: int = 1) -> Iterator[dict[str, Any]]:
    import tarfile
    from safetensors.torch import load as load_safetensors

    suffixes = (".preencoded.safetensors", ".camera.npz", ".manifest.json")
    for path in tars:
        found = 0
        pending: dict[str, dict[str, bytes]] = {}
        with tarfile.open(path) as tar:
            for member in tar:
                suffix = next((s for s in suffixes if member.name.endswith(s)), None)
                if suffix is None:
                    continue
                key = member.name[: -len(suffix)]
                entry = pending.setdefault(key, {})
                entry[suffix] = tar.extractfile(member).read()
                if len(entry) < len(suffixes):
                    continue
                del pending[key]
                tensors = load_safetensors(entry[".preencoded.safetensors"])
                manifest = json.loads(entry[".manifest.json"])
                yield {
                    "key": key,
                    "shard": str(path),
                    "latents": tensors["latents"],
                    "prompt_embeds": tensors["prompt_embeds"],
                    "camera_payload": entry[".camera.npz"],
                    "manifest": manifest,
                    "start_frame": int(manifest["preencoding"]["start_frame"]),
                    "pixel_frames": int(manifest["preencoding"]["pixel_frames"]),
                }
                found += 1
                if found >= per_shard:
                    break


def sample_inputs(provider: Any, sample: Mapping[str, Any], latent_frames: int) -> tuple[Any, ...]:
    from solarwm.backends.wan22.runtime.data import CameraGuards, build_camera_tokens

    config = provider.config
    data = config["data"]
    frame_tokens = int(config["model"]["frame_sequence_length"])
    start = sample["start_frame"]
    video = sample["manifest"]["video"]
    camera = build_camera_tokens(
        sample["camera_payload"],
        tuple(range(start, start + sample["pixel_frames"])),
        sample["manifest"],
        source_fps=float(video.get("fps") or 16.0),
        output_fps=float(data.get("fps", 16.0)),
        frame_sequence_length=frame_tokens,
        guards=CameraGuards(
            max_rel_translation=float(data["max_rel_translation"]),
            max_camera_abs=float(data["max_camera_abs"]),
        ),
        configured_array_key=str(data["camera_array_key"]),
    )
    tokens = latent_frames * frame_tokens
    device = provider.device
    latents = sample["latents"][:latent_frames].unsqueeze(0).to(device, torch.bfloat16)
    first_latent = latents[:, :1].contiguous()
    condition = {"prompt_embeds": sample["prompt_embeds"].unsqueeze(0).to(device, torch.bfloat16)}
    camera = {
        "viewmats": camera["viewmats"][:tokens].unsqueeze(0).to(device, torch.float32),
        "K": camera["K"][:tokens].unsqueeze(0).to(device, torch.float32),
    }
    return first_latent, condition, camera, latents


def build_provider(config_args: Sequence[str]) -> Any:
    from solarwm.backends.wan22.runtime import stage2
    from solarwm.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parsed = parser.parse_args(list(config_args))
    values = load_config(parsed.config, parsed.set).values
    stage2.WanTextEncoder = _NoTextEncoder
    provider = stage2.build_stage2_generation_provider(values)
    provider._load_role(str(provider.plan.passes[0].weights))
    return provider


def run_stage2_rollout(
    provider: Any,
    sample: Mapping[str, Any],
    latent_frames: int,
    seed: int = 1234,
) -> tuple[torch.Tensor, float, float]:
    from solarwm.backends.wan22.runtime import stage2

    generation_pass = replace(
        provider.plan.passes[0],
        rollout_latent_frames=latent_frames,
        min_rollout_latent_frames=latent_frames,
    )
    first_latent, condition, camera, _ = sample_inputs(provider, sample, latent_frames)
    rng = torch.Generator(device=provider.device)
    rng.manual_seed(seed)

    reset_peak_mem(provider.device)
    sync_device(provider.device)
    t0 = time.perf_counter()

    with torch.no_grad():
        latents, _ = stage2._stage2_self_forcing_latents(
            provider, generation_pass, first_latent, condition, camera, rng
        )
    sync_device(provider.device)
    elapsed = time.perf_counter() - t0
    peak_vram_gb = get_peak_mem_gb(provider.device)

    return latents.clone(), elapsed, peak_vram_gb


def _round(value: float | None, ndigits: int = 2) -> float | None:
    return None if value is None else round(value, ndigits)


def _fmt(value: float | None, unit: str = "") -> str:
    return "n/a (cached)" if value is None else f"{value:.2f}{unit}"


def compute_rel_l2(ref: torch.Tensor, test: torch.Tensor) -> float:
    diff = (ref.float() - test.float()).norm()
    norm = ref.float().norm().clamp_min(1e-8)
    return (diff / norm).item()


def decode_latents_to_u8(provider: Any, latents: torch.Tensor, device: torch.device) -> tuple[np.ndarray, float]:
    """Decodes [1, T_lat, C, H_lat, W_lat] into uint8 [T_pix, H, W, 3] numpy array."""
    sync_device(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        pix = provider.vae.decode_streaming(latents.to(device), chunk_latent_frames=4)
    sync_device(device)
    vae_time = time.perf_counter() - t0

    u8 = ((pix[0].float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    res = u8.permute(0, 2, 3, 1).cpu().numpy()
    del pix, u8
    empty_device_cache(device)
    return res, vae_time


def compute_psnr(ref: np.ndarray, test: np.ndarray) -> float:
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def write_video(path: Path | str, frames: np.ndarray, fps: int = 16) -> None:
    """Writes uint8 [T, H, W, C] frames to MP4 using av (PyAV)."""
    import av

    path = str(path)
    T, H, W, C = frames.shape
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"

    for frame in frames:
        f = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(f):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def read_video(path: Path | str) -> np.ndarray:
    """Reads MP4 file into uint8 [T, H, W, C] numpy array using av (PyAV)."""
    import av

    container = av.open(str(path), mode="r")
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    return np.stack(frames, axis=0)


def annotate_and_resize(frame_np: np.ndarray, label: str, target_size: tuple[int, int] = (432, 240)) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame_np).resize(target_size, Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((10, 10), label)
    draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill=(0, 0, 0))
    draw.text((10, 10), label, fill=(255, 255, 255))
    return np.array(img)


def main() -> None:
    # Ensure stdout is unbuffered
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Run SolarWM Stage2 SVDQuant inference across attention backends")
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=Path("/home/ssheorey/models/SolarWM/SolarWM-5B-sgf-stage2-81f/svdquant_int4"),
        help="Root directory containing SVDQuant checkpoints",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["sdpa", "sageattnv1"],
        help="Attention backends to evaluate",
    )
    parser.add_argument(
        "--k-tags",
        nargs="+",
        default=["k1", "k2", "k4"],
        help="Weight set tags to evaluate (e.g. k1 k2 k4)",
    )
    parser.add_argument(
        "--linear-quant",
        type=str,
        default=None,
        help="Optional linear quant setting (e.g. svdquant_k1, svdquant_k2, svdquant_k4)",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-running rollout and video decoding even if cached latents or videos exist",
    )
    parser.add_argument("--latent-frames", type=int, default=39, help="Number of rollout latent frames (multiples of 3)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for generation")
    parser.add_argument("--shards", type=Path, nargs="+", required=True, help="Path to latent shard tar files")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path for results")
    parser.add_argument("--save-videos", action="store_true", help="Decode latents to uint8 video using Wan VAE and save MP4/GIF artifacts")
    parser.add_argument("--save-latents-dir", type=Path, default=None, help="Directory to save generated latent .pt files")
    parser.add_argument("--compare-latents-dir", type=Path, default=None, help="Directory containing reference (e.g. CUDA) latent .pt files to compare against")
    parser.add_argument("--use-quant-vae", action="store_true", help="Apply Block-256 QuaRot W8A8 quantized VAE (XPU only)")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("/home/ssheorey/models/SolarWM/SolarWM-5B-sgf-stage2-81f/svdquant_int4/videos"),
        help="Directory to save videos",
    )
    args, rest = parser.parse_known_args()
    if args.linear_quant:
        k_part = args.linear_quant.split("_")[-1]
        args.k_tags = [k_part]
    if args.compare_latents_dir:
        args.compare_latents_dir = Path(args.compare_latents_dir)

    print("=" * 80)
    print("SOLARWM STAGE 2 SVDQUANT FULL-HORIZON INFERENCE")
    print(f"Backends: {args.backends}")
    print(f"Weight Sets: {args.k_tags}")
    print(f"Latent frames: {args.latent_frames} (target pixel frames: {(args.latent_frames - 1) * 4 + 1})")
    print(f"Quantized VAE: {args.use_quant_vae}")
    print("=" * 80)

    # 1. Build provider
    print("Building Stage 2 generation provider...")
    provider = build_provider(rest)
    device = provider.device
    print(f"Provider initialized on device: {device}")

    if args.use_quant_vae:
        print("Enabling Block-256 QuaRot W8A8 VAE on device...")
        from solarwm.backends.wan22.runtime.modeling.vae_quarot import apply_vae_quarot
        apply_vae_quarot(provider.vae.module.decoder, enabled=True)

    # 2. Get sample
    sample = next(iter_latent_samples(args.shards, per_shard=1))
    print(f"Sample key: {sample['key']} (pixel frames: {sample['pixel_frames']}, start: {sample['start_frame']})")

    if args.save_latents_dir:
        args.save_latents_dir.mkdir(parents=True, exist_ok=True)

    manager = SVDQuantModelManager(provider.diffusion.module)
    orig_diffusion = provider.diffusion
    provider.diffusion = TimestepTrackingDiffusionWrapper(orig_diffusion, manager)

    # 3. Run full-precision BF16 baseline first
    print("\n--- Running Baseline BF16 (sdpa) ---")
    saved_bf16_file = args.save_latents_dir / "bf16_sdpa.pt" if args.save_latents_dir else None
    if saved_bf16_file and saved_bf16_file.exists():
        print(f"Loading cached baseline BF16 latents from {saved_bf16_file}...")
        bf16_latents = torch.load(saved_bf16_file, map_location="cpu")
        bf16_time = None
        bf16_vram = None
    else:
        with with_attention_backend(provider.diffusion, "sdpa"):
            bf16_latents, bf16_time, bf16_vram = run_stage2_rollout(
                provider, sample, latent_frames=args.latent_frames, seed=args.seed
            )
        bf16_latents = bf16_latents.cpu()
        if args.save_latents_dir:
            torch.save(bf16_latents, saved_bf16_file)
    bf16_finite = bool(torch.isfinite(bf16_latents).all().item())
    print(f"Baseline BF16 (sdpa): Rollout {_fmt(bf16_time, 's')}, Peak Mem: {_fmt(bf16_vram, ' GB')}, Finite: {bf16_finite}")

    _, _, _, gt_latents = sample_inputs(provider, sample, args.latent_frames)
    gt_latents = gt_latents.cpu()
    bf16_gt_rel_l2 = compute_rel_l2(gt_latents, bf16_latents)
    bf16_last_rel_l2 = compute_rel_l2(gt_latents[:, -1:], bf16_latents[:, -1:])
    print(f"Baseline BF16 vs Ground Truth: Full rel L2: {bf16_gt_rel_l2:.4f} | Last frame rel L2: {bf16_last_rel_l2:.4f}")

    bf16_cuda_rel_l2 = None
    bf16_cuda_last_rel_l2 = None
    if args.compare_latents_dir:
        ref_file = args.compare_latents_dir / "bf16_sdpa.pt"
        if ref_file.exists():
            ref_lat = torch.load(ref_file, map_location="cpu")
            bf16_cuda_rel_l2 = round(compute_rel_l2(ref_lat, bf16_latents), 4)
            bf16_cuda_last_rel_l2 = round(compute_rel_l2(ref_lat[:, -1:], bf16_latents[:, -1:]), 4)
            print(f"Baseline BF16 vs Reference CUDA: Full rel L2: {bf16_cuda_rel_l2:.4f} | Last frame rel L2: {bf16_cuda_last_rel_l2:.4f}")

    if args.save_latents_dir:
        torch.save(gt_latents, args.save_latents_dir / "ground_truth.pt")
        torch.save(bf16_latents, args.save_latents_dir / "bf16_sdpa.pt")

    latents_to_decode: dict[str, torch.Tensor] = {}
    gt_latents_cpu = gt_latents.clone()
    bf16_latents_cpu = bf16_latents.clone()
    if args.save_videos:
        vdir = Path(args.video_dir)
        if not (vdir / "ground_truth.mp4").exists() or not (vdir / "last_frame_ground_truth.png").exists():
            latents_to_decode["ground_truth"] = gt_latents_cpu
        if not (vdir / "bf16_sdpa.mp4").exists() or not (vdir / "last_frame_bf16_sdpa.png").exists():
            latents_to_decode["bf16_sdpa"] = bf16_latents_cpu

    empty_device_cache(device)

    results: list[dict[str, Any]] = [
        {
            "backend": "sdpa",
            "quant": "none (BF16)",
            "k_tag": "none",
            "run_key": "bf16_sdpa",
            "rollout_time_s": _round(bf16_time),
            "vae_time_s": None,
            "total_time_s": None,
            "peak_mem_gb": _round(bf16_vram),
            "rel_l2_vs_bf16": 0.0,
            "rel_l2_vs_gt": round(bf16_gt_rel_l2, 4),
            "last_latent_rel_l2_vs_bf16": 0.0,
            "last_latent_rel_l2_vs_gt": round(bf16_last_rel_l2, 4),
            "rel_l2_vs_cuda": bf16_cuda_rel_l2,
            "last_latent_rel_l2_vs_cuda": bf16_cuda_last_rel_l2,
            "psnr_vs_bf16": float("inf"),
            "psnr_vs_gt": None,
            "last_psnr_vs_bf16": float("inf"),
            "last_psnr_vs_gt": None,
            "finite": bf16_finite,
        }
    ]

    # 4. Evaluate each (backend, k_tag) combination
    for k_tag in args.k_tags:
        for backend in args.backends:
            run_key = f"{backend}_{k_tag}"
            saved_latent_file = args.save_latents_dir / f"{run_key}.pt" if args.save_latents_dir else None

            # If latents already exist from a previous run and not force_rerun, reuse them
            if not args.force_rerun and saved_latent_file and saved_latent_file.exists():
                print(f"\n--- Loading existing latents for [{run_key}] from {saved_latent_file} ---")
                latents_cpu = torch.load(saved_latent_file, map_location="cpu")
                elapsed, peak_vram = None, None
            else:
                ckpt_dir = args.weights_root / run_key
                print(f"\n--- Running SVDQuant INT4 [{backend}] ({k_tag}) from {ckpt_dir} ---")
                if not ckpt_dir.exists():
                    print(f"ERROR: Checkpoint dir does not exist: {ckpt_dir}")
                    continue

                manager.unload_bf16_linears_to_cpu()
                empty_device_cache(device)
                manager.load_svdquant_checkpoint(ckpt_dir, device=device)
                print(f"Successfully loaded 150 SVDQuant linear layers into DiT on {device}.")

                try:
                    with with_attention_backend(provider.diffusion, backend):
                        latents, elapsed, peak_vram = run_stage2_rollout(
                            provider, sample, latent_frames=args.latent_frames, seed=args.seed
                        )
                finally:
                    manager.restore_bf16()
                    manager.unload_bf16_linears_to_cpu()
                    empty_device_cache(device)

                latents_cpu = latents.cpu()
                del latents
                empty_device_cache(device)

                if args.save_latents_dir:
                    torch.save(latents_cpu, saved_latent_file)

            # Ensure latents are finite
            is_finite = bool(torch.isfinite(latents_cpu).all().item())
            if not is_finite:
                print(f"WARNING: [{run_key}] latents contain NaN or Inf!")

            rel_l2_bf16 = compute_rel_l2(bf16_latents_cpu, latents_cpu)
            rel_l2_gt = compute_rel_l2(gt_latents_cpu, latents_cpu)
            last_rel_l2_bf16 = compute_rel_l2(bf16_latents_cpu[:, -1:], latents_cpu[:, -1:])
            last_rel_l2_gt = compute_rel_l2(gt_latents_cpu[:, -1:], latents_cpu[:, -1:])

            print(f"[{run_key}] Rollout Time: {_fmt(elapsed, 's')} | Peak Mem: {_fmt(peak_vram, ' GB')} | Finite: {is_finite}")
            print(f"[{run_key}] Full rel L2: vs BF16={rel_l2_bf16:.4f}, vs GT={rel_l2_gt:.4f}")
            print(f"[{run_key}] Last frame rel L2: vs BF16={last_rel_l2_bf16:.4f}, vs GT={last_rel_l2_gt:.4f}")

            cuda_rel_l2 = None
            cuda_last_rel_l2 = None
            if args.compare_latents_dir:
                ref_file = args.compare_latents_dir / f"{run_key}.pt"
                if ref_file.exists():
                    ref_lat = torch.load(ref_file, map_location="cpu")
                    cuda_rel_l2 = round(compute_rel_l2(ref_lat, latents_cpu), 4)
                    cuda_last_rel_l2 = round(compute_rel_l2(ref_lat[:, -1:], latents_cpu[:, -1:]), 4)
                    print(f"[{run_key}] vs Reference CUDA: Full rel L2={cuda_rel_l2:.4f}, Last frame rel L2={cuda_last_rel_l2:.4f}")

            results.append(
                {
                    "backend": backend,
                    "quant": f"SVDQuant-INT4 ({k_tag})",
                    "k_tag": k_tag,
                    "run_key": run_key,
                    "rollout_time_s": _round(elapsed),
                    "vae_time_s": None,
                    "total_time_s": None,
                    "peak_mem_gb": _round(peak_vram),
                    "rel_l2_vs_bf16": round(rel_l2_bf16, 4),
                    "rel_l2_vs_gt": round(rel_l2_gt, 4),
                    "last_latent_rel_l2_vs_bf16": round(last_rel_l2_bf16, 4),
                    "last_latent_rel_l2_vs_gt": round(last_rel_l2_gt, 4),
                    "rel_l2_vs_cuda": cuda_rel_l2,
                    "last_latent_rel_l2_vs_cuda": cuda_last_rel_l2,
                    "psnr_vs_bf16": None,
                    "psnr_vs_gt": None,
                    "last_psnr_vs_bf16": None,
                    "last_psnr_vs_gt": None,
                    "finite": is_finite,
                }
            )

            # If video was already decoded from a prior run, avoid decoding it again
            mp4_file = (Path(args.video_dir) / f"{run_key}.mp4") if args.video_dir else None
            png_file = (Path(args.video_dir) / f"last_frame_{run_key}.png") if args.video_dir else None
            if args.save_videos:
                if mp4_file and mp4_file.exists() and png_file and png_file.exists():
                    print(f"  Note: {mp4_file} already exists, will load existing video.")
                else:
                    latents_to_decode[run_key] = latents_cpu.clone()

            del latents_cpu

    # Restore original diffusion
    provider.diffusion = orig_diffusion
    manager.restore_bf16()
    manager.restore_bf16_linears_to_device(device)
    empty_device_cache(device)

    # Save video artifacts
    if args.save_videos:
        from PIL import Image

        # Free student/critic diffusion models from device memory to leave full 32GB VRAM available for VAE decoding
        del provider.diffusion, orig_diffusion, manager
        if hasattr(provider, "student"):
            del provider.student
        if hasattr(provider, "critic"):
            del provider.critic
        empty_device_cache(device)

        vdir = Path(args.video_dir)
        vdir.mkdir(parents=True, exist_ok=True)
        print(f"\nDecoding latents to video artifacts (saving to {vdir})...")

        decoded_videos: dict[str, np.ndarray] = {}
        vae_times: dict[str, float] = {}
        for name, lat in latents_to_decode.items():
            mp4_file = vdir / f"{name}.mp4"
            png_file = vdir / f"last_frame_{name}.png"
            if mp4_file.exists() and png_file.exists():
                print(f"  Loading existing decoded video for {name} from {mp4_file}...")
                vid = read_video(mp4_file)
                decoded_videos[name] = vid
                vae_times[name] = None
            else:
                print(f"  Decoding {name} with Wan VAE...")
                lat_xpu = lat.to(device)
                vid, v_time = decode_latents_to_u8(provider, lat_xpu, device)
                del lat_xpu
                decoded_videos[name] = vid
                vae_times[name] = round(v_time, 2)
                write_video(mp4_file, vid, fps=16)
                Image.fromarray(vid[-1]).save(str(png_file))
                print(f"    Decoded {vid.shape[0]} frames in {v_time:.2f}s (saved to {mp4_file})")
                empty_device_cache(device)

        # Also queue any missing MP4s for runs present in results
        for r in results:
            rk = r.get("run_key")
            if rk and (rk not in decoded_videos or args.force_rerun):
                mp4_file = vdir / f"{rk}.mp4"
                png_file = vdir / f"last_frame_{rk}.png"
                if not args.force_rerun and mp4_file.exists() and png_file.exists():
                    print(f"  Loading existing decoded video for {rk} from {mp4_file}...")
                    decoded_videos[rk] = read_video(mp4_file)
                    vae_times[rk] = None
                else:
                    saved_lat = args.save_latents_dir / f"{rk}.pt" if args.save_latents_dir else None
                    if saved_lat and saved_lat.exists():
                        print(f"  Decoding video for {rk} from {saved_lat}...")
                        lat_cpu = torch.load(saved_lat, map_location="cpu")
                        lat_xpu = lat_cpu.to(device)
                        vid, v_time = decode_latents_to_u8(provider, lat_xpu, device)
                        del lat_xpu, lat_cpu
                        decoded_videos[rk] = vid
                        vae_times[rk] = round(v_time, 2)
                        write_video(mp4_file, vid, fps=16)
                        Image.fromarray(vid[-1]).save(str(png_file))
                        print(f"    Decoded {vid.shape[0]} frames in {v_time:.2f}s (saved to {mp4_file})")
                        empty_device_cache(device)

        gt_video = decoded_videos.get("ground_truth")
        if gt_video is None and (vdir / "ground_truth.mp4").exists():
            gt_video = read_video(vdir / "ground_truth.mp4")
            decoded_videos["ground_truth"] = gt_video

        bf16_video = decoded_videos.get("bf16_sdpa")
        if bf16_video is None and (vdir / "bf16_sdpa.mp4").exists():
            bf16_video = read_video(vdir / "bf16_sdpa.mp4")
            decoded_videos["bf16_sdpa"] = bf16_video

        # XPU QuaRot W8A8 Quantized VAE comparison
        if device.type == "xpu" and not args.use_quant_vae and bf16_video is not None:
            q_mp4 = vdir / "bf16_quarot_vae.mp4"
            q_png = vdir / "last_frame_bf16_quarot_vae.png"
            if q_mp4.exists() and q_png.exists():
                print(f"  Loading existing QuaRot W8A8 VAE video from {q_mp4}...")
                q_vid = read_video(q_mp4)
                q_time = None
            else:
                print("\n  [XPU VAE Benchmark] Testing Block-256 QuaRot W8A8 Quantized VAE vs Standard BF16 VAE...")
                from solarwm.backends.wan22.runtime.modeling.vae_quarot import apply_vae_quarot
                empty_device_cache(device)
                apply_vae_quarot(provider.vae.module.decoder, enabled=True)
                bf16_lat_xpu = bf16_latents_cpu.to(device)
                q_vid, q_time = decode_latents_to_u8(provider, bf16_lat_xpu, device)
                del bf16_lat_xpu
                apply_vae_quarot(provider.vae.module.decoder, enabled=False)
                empty_device_cache(device)
                write_video(q_mp4, q_vid, fps=16)
                Image.fromarray(q_vid[-1]).save(str(q_png))
            bf16_vtime = vae_times.get("bf16_sdpa")
            q_speedup = bf16_vtime / q_time if bf16_vtime is not None and q_time is not None else None
            q_psnr = compute_psnr(bf16_video, q_vid)
            last_q_psnr = compute_psnr(bf16_video[-1:], q_vid[-1:])
            print(f"  [XPU VAE Benchmark] Standard BF16 VAE: {_fmt(bf16_vtime, 's')}")
            speedup_s = f" ({q_speedup:.2f}x speedup)" if q_speedup is not None else ""
            print(f"  [XPU VAE Benchmark] QuaRot W8A8 VAE:   {_fmt(q_time, 's')}{speedup_s}")
            print(f"  [XPU VAE Benchmark] QuaRot VAE PSNR vs BF16 VAE: Full={q_psnr:.2f} dB, Last Frame={last_q_psnr:.2f} dB")
            decoded_videos["bf16_quarot_vae"] = q_vid
            vae_times["bf16_quarot_vae"] = _round(q_time)
            for r in results:
                r["quarot_vae_time_s"] = _round(q_time)
                r["quarot_vae_speedup"] = _round(q_speedup)
                r["quarot_vae_psnr_vs_bf16"] = round(q_psnr, 2)
                r["quarot_last_psnr_vs_bf16"] = round(last_q_psnr, 2)

        for r in results:
            rk = r.get("run_key")
            if rk in decoded_videos:
                b_vid = decoded_videos[rk]
                r["vae_time_s"] = vae_times.get(rk)
                if r["vae_time_s"] is not None and r["rollout_time_s"] is not None:
                    r["total_time_s"] = round(r["rollout_time_s"] + r["vae_time_s"], 2)
                if bf16_video is not None:
                    r["psnr_vs_bf16"] = round(compute_psnr(bf16_video, b_vid), 2)
                    r["last_psnr_vs_bf16"] = round(compute_psnr(bf16_video[-1:], b_vid[-1:]), 2)
                if gt_video is not None:
                    r["psnr_vs_gt"] = round(compute_psnr(gt_video, b_vid), 2)
                    r["last_psnr_vs_gt"] = round(compute_psnr(gt_video[-1:], b_vid[-1:]), 2)
            else:
                # If video wasn't decoded in this session, attempt to load from disk to compute PSNR
                mp4_file = vdir / f"{rk}.mp4"
                if mp4_file.exists():
                    b_vid = read_video(mp4_file)
                    decoded_videos[rk] = b_vid
                    if bf16_video is not None:
                        r["psnr_vs_bf16"] = round(compute_psnr(bf16_video, b_vid), 2)
                        r["last_psnr_vs_bf16"] = round(compute_psnr(bf16_video[-1:], b_vid[-1:]), 2)
                    if gt_video is not None:
                        r["psnr_vs_gt"] = round(compute_psnr(gt_video, b_vid), 2)
                        r["last_psnr_vs_gt"] = round(compute_psnr(gt_video[-1:], b_vid[-1:]), 2)

        # 1. Save individual MP4s and last frame PNGs
        for name, vid in decoded_videos.items():
            out_mp4 = vdir / f"{name}.mp4"
            if not out_mp4.exists():
                write_video(out_mp4, vid, fps=16)
            last_png = vdir / f"last_frame_{name}.png"
            if not last_png.exists():
                last_img = Image.fromarray(vid[-1])
                last_img.save(str(last_png))
            print(f"  Verified individual MP4: {out_mp4} and last frame: {last_png}")

        # Also save ground truth and bf16 baseline last frames explicitly
        if gt_video is not None and not (vdir / "last_frame_ground_truth.png").exists():
            Image.fromarray(gt_video[-1]).save(str(vdir / "last_frame_ground_truth.png"))
        if bf16_video is not None and not (vdir / "last_frame_bf16_sdpa.png").exists():
            Image.fromarray(bf16_video[-1]).save(str(vdir / "last_frame_bf16_sdpa.png"))
        if "bf16_quarot_vae" in decoded_videos and not (vdir / "last_frame_bf16_quarot_vae.png").exists():
            Image.fromarray(decoded_videos["bf16_quarot_vae"][-1]).save(str(vdir / "last_frame_bf16_quarot_vae.png"))

        # 2. Build comparison videos per k-tag if backends present
        for k_tag in ["k1", "k2", "k4"]:
            sdpa_k = f"sdpa_{k_tag}"
            s1_k = f"sageattnv1_{k_tag}"
            s2_k = f"sageattnv2_{k_tag}"
            # 2x2 grid if both sage1 and sage2 present
            if all(k in decoded_videos for k in ["bf16_sdpa", sdpa_k, s1_k, s2_k]):
                T = decoded_videos["bf16_sdpa"].shape[0]
                grid_frames = []
                sdpa_res = next((r for r in results if r.get("run_key") == sdpa_k), {})
                s1_res = next((r for r in results if r.get("run_key") == s1_k), {})
                s2_res = next((r for r in results if r.get("run_key") == s2_k), {})

                for t in range(T):
                    f_bf16 = annotate_and_resize(decoded_videos["bf16_sdpa"][t], "BF16 Baseline (SDPA)", (432, 240))
                    p_sdpa = sdpa_res.get("psnr_vs_bf16", 0)
                    f_sdpa = annotate_and_resize(decoded_videos[sdpa_k][t], f"SVDQuant SDPA ({p_sdpa} dB)", (432, 240))
                    p_s1 = s1_res.get("psnr_vs_bf16", 0)
                    f_s1 = annotate_and_resize(decoded_videos[s1_k][t], f"SVDQuant Sage1 ({p_s1} dB)", (432, 240))
                    p_s2 = s2_res.get("psnr_vs_bf16", 0)
                    f_s2 = annotate_and_resize(decoded_videos[s2_k][t], f"SVDQuant Sage2 ({p_s2} dB)", (432, 240))

                    top = np.concatenate([f_bf16, f_sdpa], axis=1)
                    bot = np.concatenate([f_s1, f_s2], axis=1)
                    grid = np.concatenate([top, bot], axis=0)
                    grid_frames.append(grid)

                grid_np = np.stack(grid_frames, axis=0)
                out_grid = vdir / f"compare_2x2_{k_tag}.mp4"
                if args.force_rerun or not out_grid.exists():
                    write_video(out_grid, grid_np, fps=16)
                    print(f"  Verified 2x2 comparison MP4: {out_grid}")
                last_2x2_png = vdir / f"last_frame_compare_2x2_{k_tag}.png"
                if args.force_rerun or not last_2x2_png.exists():
                    Image.fromarray(grid_frames[-1]).save(str(last_2x2_png))
            # 1x3 horizontal comparison if sdpa and sage1 present (without sage2)
            elif all(k in decoded_videos for k in ["bf16_sdpa", sdpa_k, s1_k]):
                T = decoded_videos["bf16_sdpa"].shape[0]
                grid_frames = []
                sdpa_res = next((r for r in results if r.get("run_key") == sdpa_k), {})
                s1_res = next((r for r in results if r.get("run_key") == s1_k), {})

                for t in range(T):
                    f_bf16 = annotate_and_resize(decoded_videos["bf16_sdpa"][t], "BF16 Baseline (SDPA)", (432, 240))
                    p_sdpa = sdpa_res.get("psnr_vs_bf16", 0)
                    f_sdpa = annotate_and_resize(decoded_videos[sdpa_k][t], f"SVDQuant SDPA ({p_sdpa} dB)", (432, 240))
                    p_s1 = s1_res.get("psnr_vs_bf16", 0)
                    f_s1 = annotate_and_resize(decoded_videos[s1_k][t], f"SVDQuant Sage1 ({p_s1} dB)", (432, 240))

                    grid = np.concatenate([f_bf16, f_sdpa, f_s1], axis=1)
                    grid_frames.append(grid)

                grid_np = np.stack(grid_frames, axis=0)
                out_grid = vdir / f"compare_1x3_{k_tag}.mp4"
                if args.force_rerun or not out_grid.exists():
                    write_video(out_grid, grid_np, fps=16)
                    print(f"  Verified 1x3 comparison MP4: {out_grid}")
                last_1x3_png = vdir / f"last_frame_compare_1x3_{k_tag}.png"
                if args.force_rerun or not last_1x3_png.exists():
                    Image.fromarray(grid_frames[-1]).save(str(last_1x3_png))

                # Also generate 1x2 side-by-side comparison between SDPA and SageAttn1
                grid_1x2_frames = []
                for t in range(T):
                    p_sdpa = sdpa_res.get("psnr_vs_bf16", 0)
                    f_sdpa = annotate_and_resize(decoded_videos[sdpa_k][t], f"SVDQuant SDPA {k_tag.upper()} ({p_sdpa} dB)", (432, 240))
                    p_s1 = s1_res.get("psnr_vs_bf16", 0)
                    f_s1 = annotate_and_resize(decoded_videos[s1_k][t], f"SVDQuant Sage1 {k_tag.upper()} ({p_s1} dB)", (432, 240))
                    grid_1x2 = np.concatenate([f_sdpa, f_s1], axis=1)
                    grid_1x2_frames.append(grid_1x2)
                grid_1x2_np = np.stack(grid_1x2_frames, axis=0)
                out_1x2 = vdir / f"compare_1x2_{k_tag}.mp4"
                if args.force_rerun or not out_1x2.exists():
                    write_video(out_1x2, grid_1x2_np, fps=16)
                    print(f"  Verified 1x2 comparison MP4: {out_1x2}")
                last_1x2_png = vdir / f"last_frame_compare_1x2_{k_tag}.png"
                if args.force_rerun or not last_1x2_png.exists():
                    Image.fromarray(grid_1x2_frames[-1]).save(str(last_1x2_png))
            # 1x2 comparison if only sdpa is present for this k_tag
            elif all(k in decoded_videos for k in ["bf16_sdpa", sdpa_k]):
                T = decoded_videos["bf16_sdpa"].shape[0]
                grid_frames = []
                sdpa_res = next((r for r in results if r.get("run_key") == sdpa_k), {})

                for t in range(T):
                    f_bf16 = annotate_and_resize(decoded_videos["bf16_sdpa"][t], "BF16 Baseline (SDPA)", (432, 240))
                    p_sdpa = sdpa_res.get("psnr_vs_bf16", 0)
                    f_sdpa = annotate_and_resize(decoded_videos[sdpa_k][t], f"SVDQuant SDPA ({p_sdpa} dB)", (432, 240))

                    grid = np.concatenate([f_bf16, f_sdpa], axis=1)
                    grid_frames.append(grid)

                grid_np = np.stack(grid_frames, axis=0)
                out_grid = vdir / f"compare_1x2_{k_tag}.mp4"
                if args.force_rerun or not out_grid.exists():
                    write_video(out_grid, grid_np, fps=16)
                    print(f"  Verified 1x2 comparison MP4: {out_grid}")
                last_1x2_png = vdir / f"last_frame_compare_1x2_{k_tag}.png"
                if args.force_rerun or not last_1x2_png.exists():
                    Image.fromarray(grid_frames[-1]).save(str(last_1x2_png))

    # Print summary table
    print("\n" + "=" * 140)
    print(f"{'Run / Backend':<24} | {'Rollout(s)':<10} | {'VAE(s)':<8} | {'Total(s)':<8} | {'Rel L2 (BF16)':<14} | {'PSNR (BF16)':<12} | {'Last Rel L2':<12} | {'vs CUDA L2':<12} | {'Last vs CUDA':<12}")
    print("-" * 140)
    for r in results:
        label = f"{r['backend']} ({r['k_tag']})"
        rollout_s = f"{r['rollout_time_s']:.2f}" if r['rollout_time_s'] is not None else "-"
        vae_s = f"{r['vae_time_s']:.2f}" if r['vae_time_s'] is not None else "-"
        total_s = f"{r['total_time_s']:.2f}" if r['total_time_s'] is not None else "-"
        rel_l2 = f"{r['rel_l2_vs_bf16']:.4f}"
        psnr = f"{r['psnr_vs_bf16']:.2f} dB" if r['psnr_vs_bf16'] is not None else "-"
        last_l2 = f"{r['last_latent_rel_l2_vs_bf16']:.4f}"
        vs_cuda = f"{r['rel_l2_vs_cuda']:.4f}" if r.get('rel_l2_vs_cuda') is not None else "-"
        last_vs_cuda = f"{r['last_latent_rel_l2_vs_cuda']:.4f}" if r.get('last_latent_rel_l2_vs_cuda') is not None else "-"
        print(f"{label:<24} | {rollout_s:<10} | {vae_s:<8} | {total_s:<8} | {rel_l2:<14} | {psnr:<12} | {last_l2:<12} | {vs_cuda:<12} | {last_vs_cuda:<12}")
    print("=" * 140)

    if results and "quarot_vae_time_s" in results[0]:
        print("\n--- Intel XPU QuaRot W8A8 VAE Acceleration Summary ---")
        print(f"BF16 VAE Decode Time:    {results[0].get('vae_time_s', '-')}s")
        print(f"QuaRot VAE Decode Time:  {results[0].get('quarot_vae_time_s', '-')}s ({results[0].get('quarot_vae_speedup', '-')}x speedup)")
        print(f"QuaRot VAE Fidelity:     Full Video PSNR = {results[0].get('quarot_vae_psnr_vs_bf16', '-')} dB | Last Frame PSNR = {results[0].get('quarot_last_psnr_vs_bf16', '-')} dB")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved inference evaluation results to: {args.output_json}")


if __name__ == "__main__":
    main()
