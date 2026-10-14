"""SVDQuant INT4 group-64 calibration runner for Wan2.2 TI2V-5B.

Executes on NVIDIA GPU (CUDA) on isl-gpu59 using latent-WDS shards.
Supports SDPA, SageAttention v1, and SageAttention v2 attention backends.
Produces quantized INT4 weights + rank-32 LoRA + smoothing vectors + activation clip policies.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import sys
import tarfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Ensure solarwm and svdquant_int4 are importable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "quant"))

from scripts.quantization.svdquant_int4 import (
    ATTENTION_BACKENDS,
    compute_joint_smoothing_scales,
    compute_smoothing_scale,
    export_svdquant_safetensors,
    extract_low_rank_svd,
    gptq_quantize_residual,
    search_act_clip_ratio,
    with_attention_backend,
)

SUFFIXES = (".preencoded.safetensors", ".camera.npz", ".manifest.json")
TIMESTEP_BINS = ("0", "1", "2", "3", "commit")


class _NoTextEncoder:
    """Mock text encoder: prompt embeddings are already precomputed in latent-WDS."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def to(self, *args: Any, **kwargs: Any) -> _NoTextEncoder:
        return self

    def __call__(self, prompts: Sequence[str]) -> Mapping[str, Any]:
        raise RuntimeError("stage2_latent_driver uses preencoded prompt embeddings only")


def iter_latent_samples(
    tars: Sequence[Path], *, per_shard: int, keys: set[str] | None = None
) -> Iterator[dict[str, Any]]:
    from safetensors.torch import load as load_safetensors

    for path in tars:
        found = 0
        pending: dict[str, dict[str, bytes]] = {}
        with tarfile.open(path) as tar:
            for member in tar:
                suffix = next((s for s in SUFFIXES if member.name.endswith(s)), None)
                if suffix is None:
                    continue
                key = member.name[: -len(suffix)]
                if keys is not None and key not in keys:
                    continue
                entry = pending.setdefault(key, {})
                entry[suffix] = tar.extractfile(member).read()
                if len(entry) < len(SUFFIXES):
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
    if int(camera["viewmats"].shape[0]) < tokens:
        raise ValueError(f"{sample['key']}: camera covers fewer than {latent_frames} latent frames")
    device = provider.device
    latents = sample["latents"][:latent_frames].unsqueeze(0).to(device, torch.bfloat16)
    first_latent = latents[:, :1].contiguous()
    condition = {"prompt_embeds": sample["prompt_embeds"].unsqueeze(0).to(device, torch.bfloat16)}
    camera = {
        "viewmats": camera["viewmats"][:tokens].unsqueeze(0).to(device, torch.float32),
        "K": camera["K"][:tokens].unsqueeze(0).to(device, torch.float32),
    }
    return first_latent, condition, camera, latents


def guarded_samples(provider: Any, samples: Iterator[dict[str, Any]], latent_frames: int) -> Iterator[dict[str, Any]]:
    from solarwm.data.camera import CameraGuardError

    for sample in samples:
        try:
            sample_inputs(provider, sample, latent_frames)
        except CameraGuardError as exc:
            print(json.dumps({"skipped": sample["key"], "reason": str(exc)}), flush=True)
            continue
        yield sample


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


def rollout(
    provider: Any,
    sample: Mapping[str, Any],
    *,
    latent_frames: int,
    seed: int,
) -> torch.Tensor:
    from solarwm.backends.wan22.runtime import stage2

    generation_pass = replace(
        provider.plan.passes[0],
        rollout_latent_frames=latent_frames,
        min_rollout_latent_frames=latent_frames,
    )
    first_latent, condition, camera, _ = sample_inputs(provider, sample, latent_frames)
    rng = torch.Generator(device=provider.device)
    rng.manual_seed(seed)
    with torch.no_grad():
        latents, _ = stage2._stage2_self_forcing_latents(
            provider, generation_pass, first_latent, condition, camera, rng
        )
    return latents


class ActivationCollector:
    """Manages Hessian and absmax collection across blocks and timestep bins."""

    def __init__(
        self,
        num_blocks: int = 30,
        subsample_limit: int = 4096,
        target_blocks: list[int] | None = None,
    ) -> None:
        self.num_blocks = num_blocks
        self.target_blocks = target_blocks
        self.current_bin = "0"
        self.subsample_limit = subsample_limit
        self.stats: dict[tuple[int, str, str], dict[str, Any]] = {}
        self._handles: list[Any] = []

    def set_bin(self, bin_name: str) -> None:
        self.current_bin = bin_name

    def _add(self, block_idx: int, target: str, x: torch.Tensor) -> None:
        key = (block_idx, target, self.current_bin)
        stat = self.stats.get(key)
        in_feat = x.shape[-1]
        if stat is None:
            stat = {
                "hessian": torch.zeros((in_feat, in_feat), dtype=torch.float32),
                "absmax": torch.zeros(in_feat, dtype=torch.float32),
                "tokens": 0,
                "subsamples": [],
                "subsample_count": 0,
            }
            self.stats[key] = stat

        x_flat = x.detach().reshape(-1, in_feat).float()
        tokens = x_flat.shape[0]
        if tokens == 0:
            return

        batch_amax = x_flat.abs().amax(dim=0).cpu()
        stat["absmax"] = torch.maximum(stat["absmax"], batch_amax)
        stat["hessian"] += (x_flat.T @ x_flat).cpu()
        stat["tokens"] += tokens

        if stat["subsample_count"] < self.subsample_limit:
            take = min(self.subsample_limit - stat["subsample_count"], tokens)
            stat["subsamples"].append(x_flat[:take].cpu().clone())
            stat["subsample_count"] += take

    def attach(self, model: nn.Module) -> None:
        self.detach()
        blocks = getattr(model, "blocks", [])

        # 1. Pre-hook on whole model to intercept radial_step and update current_bin
        bin_names = {0: "0", 1: "1", 2: "2", 3: "3", 4: "commit"}

        def _model_pre_hook(m: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            radial_step = kwargs.get("radial_step")
            if radial_step is None and len(args) > 13:
                radial_step = args[13]
            self.set_bin(bin_names.get(radial_step, "0"))

        self._handles.append(model.register_forward_pre_hook(_model_pre_hook, with_kwargs=True))

        # 2. Block pre-hooks
        for b_idx, block in enumerate(blocks):
            if self.target_blocks is not None and b_idx not in self.target_blocks:
                continue
            def _make_block_hook(b_idx=b_idx):
                def _hook(mod: nn.Module, inp: tuple[Any, ...]) -> None:
                    x = inp[0]
                    if isinstance(x, torch.Tensor):
                        self._add(b_idx, "x", x)
                return _hook
            self._handles.append(block.register_forward_pre_hook(_make_block_hook()))

            ffn = getattr(block, "ffn", None)
            if ffn is not None and hasattr(ffn, "2"):
                ffn_down = getattr(ffn, "2")
                def _make_ffn_hook(b_idx=b_idx):
                    def _hook(mod: nn.Module, inp: tuple[Any, ...]) -> None:
                        h = inp[0]
                        if isinstance(h, torch.Tensor):
                            self._add(b_idx, "ffn_down", h)
                    return _hook
                self._handles.append(ffn_down.register_forward_pre_hook(_make_ffn_hook()))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def quantize_block_layers(
    block: nn.Module,
    block_idx: int,
    collector: ActivationCollector,
    bins: tuple[str, ...],
    rank: int = 32,
    group_size: int = 64,
    device: torch.device = torch.device("cuda:0"),
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float]]]:
    """Quantize the 5 kernel layers of one block for the specified bins."""
    tensors: dict[str, torch.Tensor] = {}
    act_policies: dict[str, dict[str, float]] = {b: {} for b in bins}

    # 1. Pool stats for 'x' input (q, k, v, ffn.0)
    x_in_feat = block.self_attn.q.in_features
    h_x_sum = torch.zeros((x_in_feat, x_in_feat), dtype=torch.float32)
    amax_x_combined = torch.zeros(x_in_feat, dtype=torch.float32)
    subsamples_x: dict[str, torch.Tensor] = {}

    for b in bins:
        stat = collector.stats.get((block_idx, "x", b))
        if stat is not None:
            h_x_sum += stat["hessian"]
            amax_x_combined = torch.maximum(amax_x_combined, stat["absmax"])
            if stat["subsamples"]:
                subsamples_x[b] = torch.cat(stat["subsamples"], dim=0).to(device)

    # Joint smoothing scale across q, k, v, ffn.0
    layers_x = [
        ("self_attn.q", block.self_attn.q),
        ("self_attn.k", block.self_attn.k),
        ("self_attn.v", block.self_attn.v),
        ("ffn.0", getattr(block.ffn, "0")),
    ]
    weights_x = [layer.weight.data.to(device) for _, layer in layers_x]
    s_x = compute_joint_smoothing_scales(amax_x_combined.to(device), weights_x, alpha=0.5)

    for name, layer in layers_x:
        w_orig = layer.weight.data.to(device)
        b_orig = layer.bias.data.to(device) if layer.bias is not None else None

        # SVD low-rank branch
        ws = (w_orig.float() * s_x.float().unsqueeze(0)).to(torch.bfloat16)
        lora_a, lora_b, resid = extract_low_rank_svd(ws, s_x, rank=rank)

        # GPTQ on residual
        qweight, w_scale, r_recon = gptq_quantize_residual(
            resid, h_x_sum.to(device), smooth_scale=s_x, group_size=group_size
        )

        pfx = f"blocks.{block_idx}.{name}"
        tensors[f"{pfx}.qweight"] = qweight.cpu()
        tensors[f"{pfx}.w_scale"] = w_scale.cpu()
        tensors[f"{pfx}.smooth_scale"] = s_x.cpu()
        tensors[f"{pfx}.lora_a"] = lora_a.cpu()
        tensors[f"{pfx}.lora_b"] = lora_b.cpu()
        if b_orig is not None:
            tensors[f"{pfx}.bias"] = b_orig.cpu()

        # Search act clip ratio per bin
        for b in bins:
            sub = subsamples_x.get(b)
            if sub is not None and sub.shape[0] > 0:
                best_rho, _ = search_act_clip_ratio(
                    sub, w_orig, b_orig, r_recon, s_x, lora_a, lora_b, group_size=group_size
                )
            else:
                best_rho = 0.8  # empirical default from paper
            act_policies[b][name] = best_rho

    # 2. Stats for 'ffn.2' (down-projection)
    ffn_down = getattr(block.ffn, "2")
    down_in_feat = ffn_down.in_features
    h_down_sum = torch.zeros((down_in_feat, down_in_feat), dtype=torch.float32)
    amax_down_combined = torch.zeros(down_in_feat, dtype=torch.float32)
    subsamples_down: dict[str, torch.Tensor] = {}

    for b in bins:
        stat = collector.stats.get((block_idx, "ffn_down", b))
        if stat is not None:
            h_down_sum += stat["hessian"]
            amax_down_combined = torch.maximum(amax_down_combined, stat["absmax"])
            if stat["subsamples"]:
                subsamples_down[b] = torch.cat(stat["subsamples"], dim=0).to(device)

    w_down = ffn_down.weight.data.to(device)
    b_down = ffn_down.bias.data.to(device) if ffn_down.bias is not None else None
    s_down = compute_smoothing_scale(amax_down_combined.to(device), w_down.float().abs().amax(dim=0), alpha=0.5)

    ws_down = (w_down.float() * s_down.float().unsqueeze(0)).to(torch.bfloat16)
    lora_a, lora_b, resid = extract_low_rank_svd(ws_down, s_down, rank=rank)

    qweight, w_scale, r_recon = gptq_quantize_residual(
        resid, h_down_sum.to(device), smooth_scale=s_down, group_size=group_size
    )

    pfx = f"blocks.{block_idx}.ffn.2"
    tensors[f"{pfx}.qweight"] = qweight.cpu()
    tensors[f"{pfx}.w_scale"] = w_scale.cpu()
    tensors[f"{pfx}.smooth_scale"] = s_down.cpu()
    tensors[f"{pfx}.lora_a"] = lora_a.cpu()
    tensors[f"{pfx}.lora_b"] = lora_b.cpu()
    if b_down is not None:
        tensors[f"{pfx}.bias"] = b_down.cpu()

    for b in bins:
        sub = subsamples_down.get(b)
        if sub is not None and sub.shape[0] > 0:
            best_rho, _ = search_act_clip_ratio(
                sub, w_down, b_down, r_recon, s_down, lora_a, lora_b, group_size=group_size
            )
        else:
            best_rho = 0.8
        act_policies[b]["ffn.2"] = best_rho

    return tensors, act_policies


def run_calibration(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"Loading Stage2 provider on {device}...")
    provider = build_provider(args.provider_args)
    model = getattr(provider.diffusion, "module", getattr(provider.diffusion, "model", provider.diffusion))

    print(f"Gathering calibration samples from: {args.shards}")
    all_shards = [Path(s) for s in args.shards]
    samples = list(iter_latent_samples(all_shards, per_shard=args.per_shard))
    samples = list(guarded_samples(provider, samples, latent_frames=args.latent_frames))
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"Collected {len(samples)} valid calibration samples.")

    # Build weight sets definitions: K=1, K=2, K=4 configurations
    # Bins: ("0", "1", "2", "3", "commit")
    configs_to_build: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    if "1" in args.k_sets:
        configs_to_build["k1"] = [("set0", TIMESTEP_BINS)]
    if "2" in args.k_sets:
        configs_to_build["k2"] = [
            ("set0", ("0", "1")),                  # high noise
            ("set1", ("2", "3", "commit")),        # low noise
        ]
    if "4" in args.k_sets:
        configs_to_build["k4"] = [
            ("set0", ("0",)),
            ("set1", ("1",)),
            ("set2", ("2",)),
            ("set3", ("3", "commit")),
        ]

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    backends = args.attn_backends
    print(f"Calibrating across attention backends: {backends}")

    for b_idx_loop, attn_backend in enumerate(backends):
        print("\n" + "=" * 80)
        print(f"CALIBRATION BACKEND [{b_idx_loop + 1}/{len(backends)}]: {attn_backend}")
        print("=" * 80)

        collector = ActivationCollector(num_blocks=len(model.blocks))
        collector.attach(model)

        print(f"Running rollouts with attention backend: {attn_backend}...", flush=True)
        t0 = time.time()
        with with_attention_backend(model, attn_backend):
            for idx, sample in enumerate(samples):
                st = time.time()
                rollout(provider, sample, latent_frames=args.latent_frames, seed=args.seed + idx)
                print(f"[{attn_backend}] [{idx + 1}/{len(samples)}] {sample['key']} completed in {time.time() - st:.2f}s", flush=True)

        collector.detach()
        print(f"Activation capture for {attn_backend} completed in {time.time() - t0:.2f}s.")

        for k_tag, sets_def in configs_to_build.items():
            print(f"\n=== Quantizing [{attn_backend}] configuration: {k_tag} ({len(sets_def)} weight sets) ===")
            quantized_sets: dict[str, dict[str, torch.Tensor]] = {}
            full_act_policy: dict[str, Any] = {}

            for set_name, set_bins in sets_def:
                print(f"Generating {set_name} for bins: {set_bins}...", flush=True)
                set_tensors: dict[str, torch.Tensor] = {}
                for b_idx in range(len(model.blocks)):
                    t_dict, act_p = quantize_block_layers(
                        model.blocks[b_idx],
                        b_idx,
                        collector,
                        bins=set_bins,
                        rank=args.rank,
                        group_size=args.group_size,
                        device=device,
                    )
                    set_tensors.update(t_dict)
                    for b_name, policies in act_p.items():
                        full_act_policy.setdefault(b_name, {})[f"blocks.{b_idx}"] = policies
                quantized_sets[set_name] = set_tensors

            manifest = {
                "format_version": "solarwm-svdquant-v1",
                "rank": args.rank,
                "group_size": args.group_size,
                "num_sets": len(sets_def),
                "k_tag": k_tag,
                "attn_backend": attn_backend,
                "timesteps": list(TIMESTEP_BINS),
                "set_mapping": {
                    b: set_name for set_name, s_bins in sets_def for b in s_bins
                },
                "activation_policy": full_act_policy,
                "calibration_samples": [s["key"] for s in samples],
            }

            k_out_dir = output_base / f"{attn_backend}_{k_tag}"
            manifest_path = export_svdquant_safetensors(k_out_dir, quantized_sets, manifest)
            print(f"Wrote {attn_backend} {k_tag} checkpoint to {manifest_path}", flush=True)

    print("\nCalibration pipeline completed successfully across all requested backends and sets.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan2.2 SVDQuant Calibration")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--shards", nargs="+", required=True, help="List of latent-WDS .tar shards")
    parser.add_argument("--per-shard", type=int, default=4, help="Samples to take per shard")
    parser.add_argument("--max-samples", type=int, default=None, help="Max total calibration samples")
    parser.add_argument("--latent-frames", type=int, default=21, help="Latent frames per sample")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--group-size", type=int, default=64, help="Weight group size")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device (single GPU)")
    parser.add_argument(
        "--attn-backends",
        nargs="+",
        default=list(ATTENTION_BACKENDS),
        choices=list(ATTENTION_BACKENDS),
        help="Attention backend(s) for calibration rollouts",
    )
    parser.add_argument(
        "--k-sets",
        nargs="+",
        default=["1", "2", "4"],
        help="Which K weight-set configurations to build (e.g. 1 2 4)",
    )
    args, rest = parser.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    args.provider_args = rest
    return args


if __name__ == "__main__":
    run_calibration(parse_args())
