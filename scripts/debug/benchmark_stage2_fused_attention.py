#!/usr/bin/env python3
"""Benchmark runtime comparison for Stage2 attention kernels:
1. Baseline (unfused sequential PyTorch pipeline)
2. fused_rope_prope_sdpa (fused RoPE + PRoPE with vendor SDPA)
3. fused_rope_prope_sage (fused RoPE + PRoPE with int8 SageAttention)

Measures:
- Per-step denoising forward latency (4 steps per chunk)
- Commit forward latency (1 step per chunk)
- Mean single forward pass latency
- Total chunk 4 steady-state rollout latency
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

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


def set_model_radial_attention(model: Any, enabled: bool) -> None:
    model.radial_attention = bool(enabled)
    for block in model.blocks:
        block.self_attn.radial_attention = bool(enabled)


def run_benchmark(
    config_path: str,
    target_chunk: int = 4,
    num_repeats: int = 5,
) -> dict[str, Any]:
    print("=" * 75)
    print(f"Stage2 Attention Kernel Benchmark: Steady-State Chunk {target_chunk}")
    print("=" * 75)

    set_overrides = [
        "model.base_path=/home/ssheorey/models/SolarWM/SolarWM-5B-base",
        "checkpoint.path=/home/ssheorey/models/SolarWM/SolarWM-5B-sgf-stage2-81f",
        "data.index_root=/home/ssheorey/data/SolarWM-Data/releases-v1/example",
        "data.transport.root=/home/ssheorey/data/SolarWM-Data/releases-v1/example",
        "data.test_index=smoke-index.jsonl.gz",
        "inference.device=xpu",
        "validation.sample_count=1",
        "runtime.output_dir=/tmp/solarwm_bench_fused",
    ]

    print("Loading config and building Stage2 generation provider...")
    resolved = load_config(config_path, set_overrides)
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

    print(f"Rolling out warmup chunks 0 to {target_chunk - 1} to reach realistic steady-state KV cache...")
    torch.xpu.synchronize()
    t_warmup_start = time.perf_counter()

    for start in range(0, target_chunk * chunk, chunk):
        end = start + chunk
        chunk_idx = start // chunk
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
        torch.xpu.synchronize()
        print(f"  Warmup chunk {chunk_idx} done.")

    torch.xpu.synchronize()
    print(f"Warmup complete in {time.perf_counter() - t_warmup_start:.2f} s. Ready at chunk {target_chunk}.")

    kv_cache_snapshot = snapshot_cache(kv_cache)
    crossattn_snapshot = snapshot_cache(crossattn_cache)
    rng_state = rng.get_state()
    xpu_rng_state = torch.xpu.get_rng_state()

    c4_start = target_chunk * chunk
    c4_end = c4_start + chunk
    c4_initial_latents = initial_noise[:, c4_start:c4_end].clone()
    camera_chunk4 = _slice_camera(
        camera,
        start_frame=c4_start,
        end_frame=c4_end,
        frame_sequence_length=frame_tokens,
    )

    model = provider.diffusion.module

    variants = [
        ("baseline (none)", None),
        ("fused_rope_prope_sdpa (dense)", "fused_rope_prope_sdpa", False),
        ("fused_rope_prope_sdpa (radial)", "fused_rope_prope_sdpa", True),
        ("fused_rope_prope_sage (dense)", "fused_rope_prope_sage", False),
        ("fused_rope_prope_sage (radial)", "fused_rope_prope_sage", True),
    ]

    chunk4_results = {}

    for variant in variants:
        label = variant[0]
        kernel_val = variant[1]
        radial_enabled = bool(variant[2]) if len(variant) > 2 else False
        print(f"\n" + "-" * 60)
        print(f"--- Benchmarking: {label} ---")
        print("-" * 60)
        set_model_fused_kernel(model, kernel_val)
        set_model_radial_attention(model, radial_enabled)

        # Warmup trial for chunk 4
        restore_cache(kv_cache, kv_cache_snapshot)
        restore_cache(crossattn_cache, crossattn_snapshot)
        rng.set_state(rng_state)
        torch.xpu.set_rng_state(xpu_rng_state)

        for step_idx, step_value in enumerate(step_values):
            timestep = torch.full((1, chunk), step_value, device=provider.device, dtype=output.dtype)
            with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                flow = provider.diffusion(
                    c4_initial_latents,
                    condition,
                    camera_chunk4,
                    expand_timesteps_to_tokens(timestep, frame_tokens),
                    sequence_length=chunk * frame_tokens,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=c4_start * frame_tokens,
                    cache_start=0,
                    cache_update_policy=denoise_cache_policy,
                    radial_step=step_idx,
                )
        commit_timestep = torch.zeros((1, chunk), device=provider.device, dtype=output.dtype)
        with torch.no_grad(), torch.autocast(device_type="xpu", dtype=torch.bfloat16):
            provider.diffusion(
                c4_initial_latents,
                condition,
                camera_chunk4,
                expand_timesteps_to_tokens(commit_timestep, frame_tokens),
                sequence_length=chunk * frame_tokens,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=c4_start * frame_tokens,
                cache_start=0,
                cache_update_policy="commit_detached",
                radial_step=len(step_values),
            )
        torch.xpu.synchronize()

        # Timed repeats
        denoise_step_times = [[] for _ in range(len(step_values))]
        commit_step_times = []
        chunk_total_times = []

        for repeat in range(num_repeats):
            restore_cache(kv_cache, kv_cache_snapshot)
            restore_cache(crossattn_cache, crossattn_snapshot)
            rng.set_state(rng_state)
            torch.xpu.set_rng_state(xpu_rng_state)
            cur_latents = c4_initial_latents.clone()

            torch.xpu.synchronize()
            t_chunk_start = time.perf_counter()

            for step_idx, step_value in enumerate(step_values):
                timestep = torch.full((1, chunk), step_value, device=provider.device, dtype=output.dtype)

                torch.xpu.synchronize()
                t_fwd_start = time.perf_counter()
                with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                    flow = provider.diffusion(
                        cur_latents,
                        condition,
                        camera_chunk4,
                        expand_timesteps_to_tokens(timestep, frame_tokens),
                        sequence_length=chunk * frame_tokens,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=c4_start * frame_tokens,
                        cache_start=0,
                        cache_update_policy=denoise_cache_policy,
                        radial_step=step_idx,
                    )
                    x0 = provider.diffusion.flow_to_x0(cur_latents, flow, timestep)
                torch.xpu.synchronize()
                fwd_ms = (time.perf_counter() - t_fwd_start) * 1000.0
                denoise_step_times[step_idx].append(fwd_ms)

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

            # Commit step
            commit_timestep = torch.zeros((1, chunk), device=provider.device, dtype=output.dtype)
            torch.xpu.synchronize()
            t_commit_start = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                provider.diffusion(
                    cur_latents,
                    condition,
                    camera_chunk4,
                    expand_timesteps_to_tokens(commit_timestep, frame_tokens),
                    sequence_length=chunk * frame_tokens,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=c4_start * frame_tokens,
                    cache_start=0,
                    cache_update_policy="commit_detached",
                    radial_step=len(step_values),
                )
            torch.xpu.synchronize()
            commit_ms = (time.perf_counter() - t_commit_start) * 1000.0
            commit_step_times.append(commit_ms)

            chunk_total_ms = (time.perf_counter() - t_chunk_start) * 1000.0
            chunk_total_times.append(chunk_total_ms)

            mean_denoise = sum(denoise_step_times[i][-1] for i in range(len(step_values))) / len(step_values)
            print(f"  [Repeat {repeat+1}/{num_repeats}] Denoise mean: {mean_denoise:6.2f} ms | Commit: {commit_ms:6.2f} ms | Chunk 4 Total: {chunk_total_ms:7.2f} ms")

        # Aggregate metrics
        avg_denoise_per_step = [sum(denoise_step_times[i]) / num_repeats for i in range(len(step_values))]
        all_denoise_flattened = [t for step in denoise_step_times for t in step]
        mean_denoise_fwd = sum(all_denoise_flattened) / len(all_denoise_flattened)
        mean_commit_fwd = sum(commit_step_times) / len(commit_step_times)
        mean_all_fwd = (sum(all_denoise_flattened) + sum(commit_step_times)) / (len(all_denoise_flattened) + len(commit_step_times))
        mean_chunk_total = sum(chunk_total_times) / len(chunk_total_times)

        chunk4_results[label] = {
            "avg_denoise_per_step_ms": avg_denoise_per_step,
            "mean_denoise_fwd_ms": mean_denoise_fwd,
            "mean_commit_fwd_ms": mean_commit_fwd,
            "mean_all_fwd_ms": mean_all_fwd,
            "mean_chunk_total_ms": mean_chunk_total,
        }

    provider.close()

    base_fwd = chunk4_results["baseline (none)"]["mean_denoise_fwd_ms"]
    base_chunk = chunk4_results["baseline (none)"]["mean_chunk_total_ms"]
    for label in chunk4_results:
        if label == "baseline (none)":
            continue
        chunk4_results[label]["fwd_speedup"] = base_fwd / chunk4_results[label]["mean_denoise_fwd_ms"]
        chunk4_results[label]["fwd_savings_ms"] = base_fwd - chunk4_results[label]["mean_denoise_fwd_ms"]
        chunk4_results[label]["chunk_speedup"] = base_chunk / chunk4_results[label]["mean_chunk_total_ms"]
        chunk4_results[label]["chunk_savings_ms"] = base_chunk - chunk4_results[label]["mean_chunk_total_ms"]

    print("\n" + "=" * 80)
    print(" SUMMARY: Chunk 4 Steady-State Runtime & Speedups")
    print("=" * 80)
    print(f"{'Variant':<24} | {'Denoise Fwd':>12} | {'Commit Fwd':>12} | {'Chunk 4 Total':>14} | {'Speedup':>9}")
    print("-" * 80)
    for label in chunk4_results:
        res = chunk4_results[label]
        sp = res.get("chunk_speedup", 1.0)
        print(f"{label:<24} | {res['mean_denoise_fwd_ms']:9.2f} ms | {res['mean_commit_fwd_ms']:9.2f} ms | {res['mean_chunk_total_ms']:11.2f} ms | {sp:7.2f}x")
    print("=" * 80)

    return chunk4_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml")
    parser.add_argument("--target-chunk", type=int, default=4)
    parser.add_argument("--num-repeats", type=int, default=5)
    parser.add_argument("--output-json", default="bench_stage2_fused_attention.json")
    args = parser.parse_args()

    results = run_benchmark(
        config_path=args.config,
        target_chunk=args.target_chunk,
        num_repeats=args.num_repeats,
    )
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")
