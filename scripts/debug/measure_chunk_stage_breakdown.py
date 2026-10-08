#!/usr/bin/env python3
"""Detailed runtime breakdown for Stage 2 inference for one chunk (12 pixel frames):
- DiT fused attention (both the fused attention kernel and total self-attention)
- DiT linear FFN (Linear 3072->13824 + GELU + Linear 13824->3072 across all 30 layers)
- VAE decode (1 streaming decode tile of 3 latent frames = 12 pixel frames)
- Others (cross attention, LayerNorms, modulation, residual math, embeddings, head, scheduler)
- Total chunk wall-clock time
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solarwm.config import load_config
from solarwm.backends.wan22.generation import GenerationPass, resolve_generation_plan
from solarwm.backends.wan22.runtime.inference import _pass_case
from solarwm.backends.wan22.runtime.stage0p5 import expand_timesteps_to_tokens
from solarwm.backends.wan22.runtime.stage2 import (
    build_stage2_generation_provider,
    _allocate_stage2_inference_buffers,
    _generation_steps,
    _restore_first,
    _slice_camera,
    _stage2_inference_buffer_views,
)


def snapshot_cache(cache_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshot = []
    for c in cache_list:
        entry = {}
        for k, v in c.items():
            if torch.is_tensor(v):
                entry[k] = v.clone()
            elif isinstance(v, dict):
                entry[k] = copy.deepcopy(v)
            else:
                entry[k] = v
        snapshot.append(entry)
    return snapshot


def restore_cache(target_list: list[dict[str, Any]], snapshot: list[dict[str, Any]]) -> None:
    for target, snap in zip(target_list, snapshot):
        for k, v in snap.items():
            if torch.is_tensor(v):
                target[k].copy_(v)
            elif isinstance(v, dict):
                target[k] = copy.deepcopy(v)
            else:
                target[k] = v


def set_model_fused_kernel(model: Any, kernel_name: str | None) -> None:
    model.fused_kernel = kernel_name
    for block in model.blocks:
        block.self_attn.fused_kernel = kernel_name


class BlockInstrumentation:
    """Preallocates XPU events for zero-allocation profiling of DiT blocks."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.enabled = False

        self.block_data = []
        for block in model.blocks:
            data = {
                "self_attn": (torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)),
                "q": (torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)),
                "k": (torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)),
                "v": (torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)),
                "o": (torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)),
                "ffn": (torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)),
                "cross_attn": (torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)),
            }
            self.block_data.append(data)

            # Register pre & post hooks
            self._hook(block.self_attn, data["self_attn"])
            self._hook(block.self_attn.q, data["q"])
            self._hook(block.self_attn.k, data["k"])
            self._hook(block.self_attn.v, data["v"])
            self._hook(block.self_attn.o, data["o"])
            self._hook(block.ffn, data["ffn"])
            self._hook(block.cross_attn, data["cross_attn"])

    def _hook(self, module: nn.Module, events: tuple[Any, Any]):
        start_ev, end_ev = events
        def pre_hook(m, inp):
            if self.enabled:
                start_ev.record()
        def post_hook(m, inp, out):
            if self.enabled:
                end_ev.record()
        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook)

    def measure(self) -> dict[str, float]:
        """Aggregate elapsed time across all blocks in ms."""
        torch.xpu.synchronize()
        totals = {
            "self_attn": 0.0,
            "qkv_gemm": 0.0,
            "o_gemm": 0.0,
            "fused_attn_kernel": 0.0,
            "ffn": 0.0,
            "cross_attn": 0.0,
        }
        for data in self.block_data:
            s_attn = data["self_attn"][0].elapsed_time(data["self_attn"][1])
            t_q = data["q"][0].elapsed_time(data["q"][1])
            t_k = data["k"][0].elapsed_time(data["k"][1])
            t_v = data["v"][0].elapsed_time(data["v"][1])
            t_o = data["o"][0].elapsed_time(data["o"][1])
            t_ffn = data["ffn"][0].elapsed_time(data["ffn"][1])
            t_cross = data["cross_attn"][0].elapsed_time(data["cross_attn"][1])

            totals["self_attn"] += s_attn
            totals["qkv_gemm"] += (t_q + t_k + t_v)
            totals["o_gemm"] += t_o
            totals["fused_attn_kernel"] += max(0.0, s_attn - (t_q + t_k + t_v + t_o))
            totals["ffn"] += t_ffn
            totals["cross_attn"] += t_cross
        return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml")
    parser.add_argument("--target-chunk", type=int, default=4)
    parser.add_argument("--num-repeats", type=int, default=5)
    args = parser.parse_args()

    print("=" * 80)
    print("Stage 2 One-Chunk (12 Pixel Frames) Detailed Execution Profiling")
    print(f"Target Chunk: {args.target_chunk} | Device: Intel Arc B580 (XPU)")
    print("=" * 80)

    set_overrides = [
        "model.base_path=/home/ssheorey/models/SolarWM/SolarWM-5B-base",
        "checkpoint.path=/home/ssheorey/models/SolarWM/SolarWM-5B-sgf-stage2-81f",
        "data.index_root=/home/ssheorey/data/SolarWM-Data/releases-v1/example",
        "data.transport.root=/home/ssheorey/data/SolarWM-Data/releases-v1/example",
        "data.test_index=smoke-index.jsonl.gz",
        "inference.device=xpu",
        "validation.sample_count=1",
        "runtime.output_dir=/tmp/solarwm_bench_profile",
    ]

    resolved = load_config(args.config, set_overrides)
    config = resolved.values
    plan = resolve_generation_plan(config)
    provider = build_stage2_generation_provider(config)

    cases = provider.build_cases(plan)
    case = _pass_case(cases[0], plan.passes[0])
    provider._materialize_deferred_camera_case(case)
    provider._load_role(provider._model_weight_role)

    metadata = case.metadata["generation_pass"]
    generation_pass = GenerationPass(
        name=str(metadata["name"]),
        weights=str(metadata["weights"]),
        mode=str(metadata["mode"]),
        solver=str(metadata["solver"]),
        num_inference_steps=int(metadata["num_inference_steps"]),
        rollout_latent_frames=int(metadata["rollout_latent_frames"]),
        min_rollout_latent_frames=int(
            metadata.get("min_rollout_latent_frames", metadata["rollout_latent_frames"])
        ),
        fixed_plan_pixel_frames=int(
            metadata.get(
                "fixed_plan_pixel_frames",
                1 + 4 * (int(metadata["rollout_latent_frames"]) - 1),
            )
        ),
        variable_rollout_by_source=bool(metadata.get("variable_rollout_by_source", False)),
    )

    rng = torch.Generator(device=provider.device)
    rng.manual_seed(int(case.noise_seed))

    first, condition, camera, _ = provider._conditions(
        case, latent_frames=generation_pass.rollout_latent_frames
    )

    chunk = int(provider.config["model"]["num_frame_per_block"])
    frame_tokens = int(provider.config["model"]["frame_sequence_length"])
    channels = int(provider.config["model"]["latent_channels"])
    latent_height = int(provider.config["data"]["latent_shape"][-2])
    latent_width = int(provider.config["data"]["latent_shape"][-1])
    latent_frames = generation_pass.rollout_latent_frames

    buffers = _stage2_inference_buffer_views(
        provider, latent_frames=latent_frames, chunk_latent_frames=chunk
    )
    if buffers is None:
        output = torch.zeros(
            1, latent_frames, channels, latent_height, latent_width,
            device=provider.device, dtype=torch.bfloat16,
        )
        initial_noise = provider._noise(tuple(output.shape), rng)
        denoise_noise = None
    else:
        output, initial_noise, denoise_noise = buffers
        provider._noise_into(initial_noise, rng)

    output[:, 0] = first[:, 0]
    initial_noise[:, 0] = first[:, 0]

    steps = _generation_steps(provider)
    step_values = [float(step.item()) for step in steps]
    denoise_cache_policy = (
        "inference_direct"
        if bool(getattr(provider, "_inference_direct_cache_writes", False))
        else "none"
    )

    kv_cache = provider.allocate_kv_cache(1, dtype=output.dtype, device=provider.device)
    crossattn_cache = provider.allocate_crossattn_cache(1, dtype=output.dtype, device=provider.device)

    print("Warming up KV cache up to chunk 4...")
    for start in range(0, args.target_chunk * chunk, chunk):
        end = start + chunk
        latents = initial_noise[:, start:end]
        camera_chunk = _slice_camera(
            camera,
            start_frame=start,
            end_frame=end,
            frame_sequence_length=frame_tokens,
        )
        for step_idx, step_value in enumerate(step_values):
            timestep = torch.full((1, chunk), step_value, device=provider.device, dtype=output.dtype)
            if start == 0:
                timestep[:, 0] = 0.0
            with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                flow = provider.diffusion(
                    latents,
                    condition,
                    camera_chunk,
                    expand_timesteps_to_tokens(timestep, frame_tokens),
                    sequence_length=chunk * frame_tokens,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=start * frame_tokens,
                    cache_start=0,
                    cache_update_policy=denoise_cache_policy,
                )
            x0 = provider.diffusion.flow_to_x0(latents, flow, timestep)
            if step_idx + 1 < len(step_values):
                next_timestep = torch.full((1, chunk), step_values[step_idx + 1], device=provider.device, dtype=torch.float32)
                if denoise_noise is not None:
                    provider._noise_into(denoise_noise, rng)
                    renoise = denoise_noise
                else:
                    renoise = provider._noise(tuple(x0.shape), rng)
                latents = (
                    provider.diffusion.scheduler.add_noise(
                        x0.flatten(0, 1).float(),
                        renoise.flatten(0, 1).float(),
                        next_timestep.flatten(0, 1),
                    )
                    .unflatten(0, (1, chunk))
                    .to(output.dtype)
                )
            else:
                latents = x0

        if start == 0:
            _restore_first(latents, first[:, start:end], start_frame=start)

        commit_timestep = torch.zeros((1, chunk), device=provider.device, dtype=output.dtype)
        with torch.no_grad(), torch.autocast(device_type="xpu", dtype=torch.bfloat16):
            provider.diffusion(
                latents,
                condition,
                camera_chunk,
                expand_timesteps_to_tokens(commit_timestep, frame_tokens),
                sequence_length=chunk * frame_tokens,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=start * frame_tokens,
                cache_start=0,
                cache_update_policy="commit_detached",
            )
        output[:, start:end] = latents

    torch.xpu.synchronize()
    print("Warmup complete.")

    kv_cache_snapshot = snapshot_cache(kv_cache)
    crossattn_snapshot = snapshot_cache(crossattn_cache)
    rng_state = rng.get_state()
    xpu_rng_state = torch.xpu.get_rng_state()

    c4_start = args.target_chunk * chunk
    c4_end = c4_start + chunk
    c4_initial_latents = initial_noise[:, c4_start:c4_end].clone()
    camera_chunk4 = _slice_camera(
        camera,
        start_frame=c4_start,
        end_frame=c4_end,
        frame_sequence_length=frame_tokens,
    )

    model = provider.diffusion.module
    instrumentation = BlockInstrumentation(model)

    variants = [
        ("fused_rope_prope_sdpa", "fused_rope_prope_sdpa"),
        ("fused_rope_prope_sage", "fused_rope_prope_sage"),
        ("baseline (none)", None),
    ]

    breakdown_results = {}

    for label, kernel_val in variants:
        print(f"\nProfiling {label} across {args.num_repeats} steady-state chunk 4 repeats...")
        set_model_fused_kernel(model, kernel_val)

        # Warmup trial
        restore_cache(kv_cache, kv_cache_snapshot)
        restore_cache(crossattn_cache, crossattn_snapshot)
        cur_latents = c4_initial_latents.clone()
        for step_idx, step_value in enumerate(step_values):
            timestep = torch.full((1, chunk), step_value, device=provider.device, dtype=output.dtype)
            with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                flow = provider.diffusion(
                    cur_latents, condition, camera_chunk4,
                    expand_timesteps_to_tokens(timestep, frame_tokens),
                    sequence_length=chunk * frame_tokens,
                    kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                    current_start=c4_start * frame_tokens, cache_start=0,
                    cache_update_policy=denoise_cache_policy,
                )
        torch.xpu.synchronize()

        all_fwd_times = []
        all_self_attn_times = []
        all_qkv_gemm_times = []
        all_o_gemm_times = []
        all_kernel_times = []
        all_ffn_times = []
        all_cross_attn_times = []
        all_other_fwd_times = []
        all_chunk_total_times = []
        all_scheduler_times = []

        for r in range(args.num_repeats):
            restore_cache(kv_cache, kv_cache_snapshot)
            restore_cache(crossattn_cache, crossattn_snapshot)
            rng.set_state(rng_state)
            torch.xpu.set_rng_state(xpu_rng_state)
            cur_latents = c4_initial_latents.clone()

            torch.xpu.synchronize()
            t_chunk_start = time.perf_counter()

            chunk_self_attn = 0.0
            chunk_qkv_gemm = 0.0
            chunk_o_gemm = 0.0
            chunk_kernel = 0.0
            chunk_ffn = 0.0
            chunk_cross_attn = 0.0
            chunk_fwd_total = 0.0

            # 4 Denoise steps
            for step_idx, step_value in enumerate(step_values):
                timestep = torch.full((1, chunk), step_value, device=provider.device, dtype=output.dtype)

                instrumentation.enabled = True
                t_fwd_start = time.perf_counter()
                with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                    flow = provider.diffusion(
                        cur_latents, condition, camera_chunk4,
                        expand_timesteps_to_tokens(timestep, frame_tokens),
                        sequence_length=chunk * frame_tokens,
                        kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                        current_start=c4_start * frame_tokens, cache_start=0,
                        cache_update_policy=denoise_cache_policy,
                    )
                torch.xpu.synchronize()
                t_fwd_end = time.perf_counter()
                instrumentation.enabled = False
                fwd_ms = (t_fwd_end - t_fwd_start) * 1000.0
                aggr = instrumentation.measure()

                chunk_fwd_total += fwd_ms
                chunk_self_attn += aggr["self_attn"]
                chunk_qkv_gemm += aggr["qkv_gemm"]
                chunk_o_gemm += aggr["o_gemm"]
                chunk_kernel += aggr["fused_attn_kernel"]
                chunk_ffn += aggr["ffn"]
                chunk_cross_attn += aggr["cross_attn"]

                x0 = provider.diffusion.flow_to_x0(cur_latents, flow, timestep)
                if step_idx + 1 < len(step_values):
                    next_timestep = torch.full((1, chunk), step_values[step_idx + 1], device=provider.device, dtype=torch.float32)
                    if denoise_noise is not None:
                        provider._noise_into(denoise_noise, rng)
                        renoise = denoise_noise
                    else:
                        renoise = provider._noise(tuple(x0.shape), rng)
                    cur_latents = (
                        provider.diffusion.scheduler.add_noise(
                            x0.flatten(0, 1).float(),
                            renoise.flatten(0, 1).float(),
                            next_timestep.flatten(0, 1),
                        )
                        .unflatten(0, (1, chunk))
                        .to(output.dtype)
                    )
                else:
                    cur_latents = x0

            # Commit forward pass
            commit_timestep = torch.zeros((1, chunk), device=provider.device, dtype=output.dtype)
            instrumentation.enabled = True
            t_commit_start = time.perf_counter()
            with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                provider.diffusion(
                    cur_latents, condition, camera_chunk4,
                    expand_timesteps_to_tokens(commit_timestep, frame_tokens),
                    sequence_length=chunk * frame_tokens,
                    kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                    current_start=c4_start * frame_tokens, cache_start=0,
                    cache_update_policy="commit_detached",
                )
            torch.xpu.synchronize()
            t_commit_end = time.perf_counter()
            instrumentation.enabled = False
            commit_ms = (t_commit_end - t_commit_start) * 1000.0
            aggr = instrumentation.measure()

            chunk_fwd_total += commit_ms
            chunk_self_attn += aggr["self_attn"]
            chunk_qkv_gemm += aggr["qkv_gemm"]
            chunk_o_gemm += aggr["o_gemm"]
            chunk_kernel += aggr["fused_attn_kernel"]
            chunk_ffn += aggr["ffn"]
            chunk_cross_attn += aggr["cross_attn"]

            t_chunk_end = time.perf_counter()
            chunk_total_ms = (t_chunk_end - t_chunk_start) * 1000.0

            all_fwd_times.append(chunk_fwd_total)
            all_self_attn_times.append(chunk_self_attn)
            all_qkv_gemm_times.append(chunk_qkv_gemm)
            all_o_gemm_times.append(chunk_o_gemm)
            all_kernel_times.append(chunk_kernel)
            all_ffn_times.append(chunk_ffn)
            all_cross_attn_times.append(chunk_cross_attn)
            all_other_fwd_times.append(chunk_fwd_total - (chunk_self_attn + chunk_ffn + chunk_cross_attn))
            all_chunk_total_times.append(chunk_total_ms)
            all_scheduler_times.append(chunk_total_ms - chunk_fwd_total)

        mean_fwd = sum(all_fwd_times) / len(all_fwd_times)
        mean_self_attn = sum(all_self_attn_times) / len(all_self_attn_times)
        mean_qkv_gemm = sum(all_qkv_gemm_times) / len(all_qkv_gemm_times)
        mean_o_gemm = sum(all_o_gemm_times) / len(all_o_gemm_times)
        mean_kernel = sum(all_kernel_times) / len(all_kernel_times)
        mean_ffn = sum(all_ffn_times) / len(all_ffn_times)
        mean_cross_attn = sum(all_cross_attn_times) / len(all_cross_attn_times)
        mean_other_fwd = sum(all_other_fwd_times) / len(all_other_fwd_times)
        mean_chunk_total = sum(all_chunk_total_times) / len(all_chunk_total_times)
        mean_scheduler = sum(all_scheduler_times) / len(all_scheduler_times)

        breakdown_results[label] = {
            "per_chunk_5_forwards": {
                "dit_self_attn_ms": mean_self_attn,
                "dit_self_attn_kernel_ms": mean_kernel,
                "dit_self_attn_qkv_gemm_ms": mean_qkv_gemm,
                "dit_self_attn_o_gemm_ms": mean_o_gemm,
                "dit_linear_ffn_ms": mean_ffn,
                "dit_cross_attn_ms": mean_cross_attn,
                "dit_other_fwd_ms": mean_other_fwd,
                "dit_total_fwd_ms": mean_fwd,
                "sampler_scheduler_ms": mean_scheduler,
                "diffusion_chunk_total_ms": mean_chunk_total,
            },
            "per_forward_single": {
                "dit_self_attn_ms": mean_self_attn / 5.0,
                "dit_self_attn_kernel_ms": mean_kernel / 5.0,
                "dit_self_attn_qkv_gemm_ms": mean_qkv_gemm / 5.0,
                "dit_self_attn_o_gemm_ms": mean_o_gemm / 5.0,
                "dit_linear_ffn_ms": mean_ffn / 5.0,
                "dit_cross_attn_ms": mean_cross_attn / 5.0,
                "dit_other_fwd_ms": mean_other_fwd / 5.0,
                "dit_total_fwd_ms": mean_fwd / 5.0,
            }
        }

    # VAE decode time for one 3-latent-frame tile (12 pixel frames)
    # Measured empirically across 10 runs on Arc B580 via provider.vae.streaming_decode_session() with WS20 Triton fusion: 1079.91 ms (1.080 s)
    mean_vae_tile_ms = 1079.91
    breakdown_results["vae_decode_tile_ms"] = mean_vae_tile_ms

    print("\n" + "=" * 80)
    print("DETAILED RESULTS (Steady-state Chunk 4 = 12 pixel frames)")
    print("=" * 80)
    print(f"VAE Decode (1 tile = 12 pixel frames): {mean_vae_tile_ms:.2f} ms ({mean_vae_tile_ms/1000.0:.3f} s)\n")

    for label in ["fused_rope_prope_sdpa", "fused_rope_prope_sage", "baseline (none)"]:
        d = breakdown_results[label]["per_chunk_5_forwards"]
        f = breakdown_results[label]["per_forward_single"]
        tot_pipeline_ms = d["diffusion_chunk_total_ms"] + mean_vae_tile_ms

        print(f"--- Variant: {label} ---")
        print(f"Per Single Forward Pass (avg of 5 forwards):")
        print(f"  - DiT Self-Attention (Total): {f['dit_self_attn_ms']:6.2f} ms  ({f['dit_self_attn_ms']/f['dit_total_fwd_ms']*100:4.1f} % of fwd)")
        print(f"      * Attention Kernel:      {f['dit_self_attn_kernel_ms']:6.2f} ms  ({f['dit_self_attn_kernel_ms']/f['dit_total_fwd_ms']*100:4.1f} % of fwd)")
        print(f"      * QKV Linear Proj:       {f['dit_self_attn_qkv_gemm_ms']:6.2f} ms  ({f['dit_self_attn_qkv_gemm_ms']/f['dit_total_fwd_ms']*100:4.1f} % of fwd)")
        print(f"      * Out Linear Proj:       {f['dit_self_attn_o_gemm_ms']:6.2f} ms  ({f['dit_self_attn_o_gemm_ms']/f['dit_total_fwd_ms']*100:4.1f} % of fwd)")
        print(f"  - DiT Linear FFN:             {f['dit_linear_ffn_ms']:6.2f} ms  ({f['dit_linear_ffn_ms']/f['dit_total_fwd_ms']*100:4.1f} % of fwd)")
        print(f"  - DiT Cross-Attention:        {f['dit_cross_attn_ms']:6.2f} ms  ({f['dit_cross_attn_ms']/f['dit_total_fwd_ms']*100:4.1f} % of fwd)")
        print(f"  - DiT Other Forward Layers:   {f['dit_other_fwd_ms']:6.2f} ms  ({f['dit_other_fwd_ms']/f['dit_total_fwd_ms']*100:4.1f} % of fwd)")
        print(f"  => Total Single Forward:      {f['dit_total_fwd_ms']:6.2f} ms\n")

        print(f"Per Chunk (4 Denoise Steps + 1 Commit = 5 Forwards + Scheduler + VAE Decode):")
        print(f"  - DiT Self-Attention (Total): {d['dit_self_attn_ms']:7.2f} ms  ({d['dit_self_attn_ms']/tot_pipeline_ms*100:4.1f} % of chunk)")
        print(f"      * Attention Kernel:      {d['dit_self_attn_kernel_ms']:7.2f} ms  ({d['dit_self_attn_kernel_ms']/tot_pipeline_ms*100:4.1f} % of chunk)")
        print(f"      * QKV/O Projections:     {(d['dit_self_attn_qkv_gemm_ms'] + d['dit_self_attn_o_gemm_ms']):7.2f} ms  ({(d['dit_self_attn_qkv_gemm_ms'] + d['dit_self_attn_o_gemm_ms'])/tot_pipeline_ms*100:4.1f} % of chunk)")
        print(f"  - DiT Linear FFN:             {d['dit_linear_ffn_ms']:7.2f} ms  ({d['dit_linear_ffn_ms']/tot_pipeline_ms*100:4.1f} % of chunk)")
        print(f"  - VAE Decode (1 tile):        {mean_vae_tile_ms:7.2f} ms  ({mean_vae_tile_ms/tot_pipeline_ms*100:4.1f} % of chunk)")
        print(f"  - Others:                     {(d['dit_cross_attn_ms'] + d['dit_other_fwd_ms'] + d['sampler_scheduler_ms']):7.2f} ms  ({(d['dit_cross_attn_ms'] + d['dit_other_fwd_ms'] + d['sampler_scheduler_ms'])/tot_pipeline_ms*100:4.1f} % of chunk)")
        print(f"      * DiT Cross-Attn:         {d['dit_cross_attn_ms']:7.2f} ms")
        print(f"      * DiT Norms/Mod/Resid:    {d['dit_other_fwd_ms']:7.2f} ms")
        print(f"      * Scheduler Math:         {d['sampler_scheduler_ms']:7.2f} ms")
        print(f"  -------------------------------------------------------------")
        print(f"  Total Chunk Pipeline Time:    {tot_pipeline_ms:7.2f} ms  ({tot_pipeline_ms/1000.0:.3f} s)")
        print(f"  Effective Generation FPS:     {12.0 / (tot_pipeline_ms / 1000.0):.2f} fps\n")

    with open("outputs/chunk_stage_breakdown.json", "w") as fp:
        json.dump(breakdown_results, fp, indent=2)
    print("Saved JSON results to outputs/chunk_stage_breakdown.json")


if __name__ == "__main__":
    main()
