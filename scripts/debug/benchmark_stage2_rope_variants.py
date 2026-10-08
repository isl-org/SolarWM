#!/usr/bin/env python3
"""Benchmark runtime comparison for RoPE precision variants (float64, float32, float16).

Measures:
1. RoPE microbenchmark latency (Q and K kernel alone on XPU with chunk 4 shapes).
2. End-to-end single diffusion forward pass latency (denoise steps and commit step)
   at chunk 4 on the authentic 5B model with real KV cache and attention buffers.
3. Total chunk 4 rollout latency across multiple repeated trials.
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
from solarwm.backends.wan22.runtime.modeling.model import rope_params
from solarwm.backends.wan22.runtime.modeling.causal_model import echorope_apply
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


def run_rope_microbenchmark(device: str = "xpu") -> dict[str, Any]:
    print("\n" + "=" * 70)
    print(" 1. RoPE Kernel Microbenchmark (Chunk 4 shapes: Q=1215, K=6075 tokens)")
    print("=" * 70)

    batch, num_heads, head_dim = 1, 24, 128
    seq_len_q = 1215
    seq_len_k = 6075

    q = torch.randn(batch, seq_len_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch, seq_len_k, num_heads, head_dim, dtype=torch.bfloat16, device=device)

    grid_q = torch.tensor([[3, 15, 27]], dtype=torch.long, device=device)
    grid_k = torch.tensor([[15, 15, 27]], dtype=torch.long, device=device)

    base_freqs = torch.cat(
        [
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
        ],
        dim=1,
    ).to(device)

    results = {}
    variants = [
        ("float64", torch.complex128),
        ("float32", torch.complex64),
        ("float16", torch.complex32),
    ]

    for name, c_dt in variants:
        freqs = base_freqs.to(c_dt)
        # Warmup
        for _ in range(15):
            _ = echorope_apply(q, grid_q, freqs, rope_dtype=name)
            _ = echorope_apply(k, grid_k, freqs, rope_dtype=name)
        torch.xpu.synchronize()

        # Timed benchmark: 50 iterations
        N = 50
        t0 = time.perf_counter()
        for _ in range(N):
            _ = echorope_apply(q, grid_q, freqs, rope_dtype=name)
            _ = echorope_apply(k, grid_k, freqs, rope_dtype=name)
        torch.xpu.synchronize()
        per_layer_ms = (time.perf_counter() - t0) / N * 1000.0
        per_pass_ms = per_layer_ms * 30.0  # 30 layers in Wan2.2 5B
        results[name] = {
            "per_layer_ms": per_layer_ms,
            "per_pass_30_layers_ms": per_pass_ms,
        }
        print(f"  {name:8s}: {per_layer_ms:6.3f} ms / layer (Q+K)  |  {per_pass_ms:6.2f} ms per 30-layer forward pass")

    base_ms = results["float64"]["per_pass_30_layers_ms"]
    for name in ["float32", "float16"]:
        speedup = base_ms / results[name]["per_pass_30_layers_ms"]
        savings = base_ms - results[name]["per_pass_30_layers_ms"]
        results[name]["speedup_vs_fp64"] = speedup
        results[name]["savings_per_pass_ms"] = savings

    return results


def run_chunk4_benchmark(
    config_path: str,
    target_chunk: int = 4,
    num_repeats: int = 3,
) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print(f" 2. Full Model Single Diffusion Forward Pass & Chunk {target_chunk} Benchmark")
    print("=" * 70)

    set_overrides = [
        "model.base_path=/home/ssheorey/models/SolarWM/SolarWM-5B-base",
        "checkpoint.path=/home/ssheorey/models/SolarWM/SolarWM-5B-sgf-stage2-81f",
        "data.index_root=/home/ssheorey/data/SolarWM-Data/releases-v1/example",
        "data.transport.root=/home/ssheorey/data/SolarWM-Data/releases-v1/example",
        "data.test_index=smoke-index.jsonl.gz",
        "inference.device=xpu",
        "validation.sample_count=1",
        "runtime.output_dir=/tmp/solarwm_bench_chunk4",
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
            if start == 0:
                x0 = _restore_first(x0, first, start_frame=0)
            if step_idx + 1 < len(step_values):
                next_timestep = torch.full((1, chunk), step_values[step_idx + 1], device=provider.device, dtype=torch.float32)
                if start == 0:
                    next_timestep[:, 0] = 0.0
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
                if start == 0:
                    latents = _restore_first(latents, first, start_frame=0)
            else:
                latents = x0

        output[:, start:end] = latents
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

    # Snapshot KV cache, cross-attn cache, and RNG state before chunk 4
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
    dtype_map = {
        "float64": (torch.float64, torch.complex128),
        "float32": (torch.float32, torch.complex64),
        "float16": (torch.float16, torch.complex32),
    }

    chunk4_results = {}

    for variant_name in ["float64", "float32", "float16"]:
        real_dt, complex_dt = dtype_map[variant_name]
        print(f"\n--- Benchmarking Variant: {variant_name} ({complex_dt}) ---")
        model.rope_dtype = real_dt
        model.rope_complex_dtype = complex_dt
        model.freqs = model.freqs.to(complex_dt)

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

        chunk4_results[variant_name] = {
            "denoise_step_times_ms": denoise_step_times,
            "avg_denoise_per_step_ms": avg_denoise_per_step,
            "commit_step_times_ms": commit_step_times,
            "chunk_total_times_ms": chunk_total_times,
            "mean_denoise_fwd_ms": mean_denoise_fwd,
            "mean_commit_fwd_ms": mean_commit_fwd,
            "mean_all_fwd_ms": mean_all_fwd,
            "mean_chunk_total_ms": mean_chunk_total,
        }

    provider.close()

    # Calculate relative speedups vs float64
    base_fwd = chunk4_results["float64"]["mean_denoise_fwd_ms"]
    base_chunk = chunk4_results["float64"]["mean_chunk_total_ms"]
    for name in ["float32", "float16"]:
        chunk4_results[name]["fwd_speedup"] = base_fwd / chunk4_results[name]["mean_denoise_fwd_ms"]
        chunk4_results[name]["fwd_savings_ms"] = base_fwd - chunk4_results[name]["mean_denoise_fwd_ms"]
        chunk4_results[name]["chunk_speedup"] = base_chunk / chunk4_results[name]["mean_chunk_total_ms"]
        chunk4_results[name]["chunk_savings_ms"] = base_chunk - chunk4_results[name]["mean_chunk_total_ms"]

    return chunk4_results


def main() -> int:
    parser = argparse.ArgumentParser(description="RoPE precision runtime benchmark")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml"),
        help="Path to inference config YAML",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed repeats per variant")
    parser.add_argument("--chunk", type=int, default=4, help="Chunk index to benchmark")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    micro_results = run_rope_microbenchmark("xpu")
    chunk4_results = run_chunk4_benchmark(args.config, target_chunk=args.chunk, num_repeats=args.repeats)

    print("\n" + "=" * 85)
    print(f" FINAL BENCHMARK SUMMARY (Wan2.2 Stage2 5B on Intel XPU, Chunk {args.chunk})")
    print("=" * 85)
    print(f"{'Metric':<35} | {'Float64 (Baseline)':<16} | {'Float32':<14} | {'Float16':<14}")
    print("-" * 85)

    # 1. RoPE microbenchmark
    r64_layer = micro_results["float64"]["per_layer_ms"]
    r32_layer = micro_results["float32"]["per_layer_ms"]
    r16_layer = micro_results["float16"]["per_layer_ms"]
    print(f"{'RoPE Kernel Latency (1 Layer, Q+K)':<35} | {r64_layer:6.3f} ms        | {r32_layer:6.3f} ms      | {r16_layer:6.3f} ms")

    r64_pass = micro_results["float64"]["per_pass_30_layers_ms"]
    r32_pass = micro_results["float32"]["per_pass_30_layers_ms"]
    r16_pass = micro_results["float16"]["per_pass_30_layers_ms"]
    print(f"{'RoPE Kernel Total (30 Layers)':<35} | {r64_pass:6.2f} ms        | {r32_pass:6.2f} ms      | {r16_pass:6.2f} ms")

    r32_rope_sp = micro_results["float32"]["speedup_vs_fp64"]
    r16_rope_sp = micro_results["float16"]["speedup_vs_fp64"]
    print(f"{'  -> RoPE Kernel Speedup':<35} | {'1.00x (ref)':<16} | {r32_rope_sp:5.2f}x         | {r16_rope_sp:5.2f}x")

    print("-" * 85)

    # 2. Diffusion forward pass
    f64_d0 = chunk4_results["float64"]["avg_denoise_per_step_ms"][0]
    f32_d0 = chunk4_results["float32"]["avg_denoise_per_step_ms"][0]
    f16_d0 = chunk4_results["float16"]["avg_denoise_per_step_ms"][0]
    print(f"{'Diffusion Fwd Pass (Step 0)':<35} | {f64_d0:6.2f} ms        | {f32_d0:6.2f} ms      | {f16_d0:6.2f} ms")

    f64_mean = chunk4_results["float64"]["mean_denoise_fwd_ms"]
    f32_mean = chunk4_results["float32"]["mean_denoise_fwd_ms"]
    f16_mean = chunk4_results["float16"]["mean_denoise_fwd_ms"]
    print(f"{'Diffusion Fwd Pass (Mean Denoise)':<35} | {f64_mean:6.2f} ms        | {f32_mean:6.2f} ms      | {f16_mean:6.2f} ms")

    f64_commit = chunk4_results["float64"]["mean_commit_fwd_ms"]
    f32_commit = chunk4_results["float32"]["mean_commit_fwd_ms"]
    f16_commit = chunk4_results["float16"]["mean_commit_fwd_ms"]
    print(f"{'Diffusion Fwd Pass (Commit Step)':<35} | {f64_commit:6.2f} ms        | {f32_commit:6.2f} ms      | {f16_commit:6.2f} ms")

    f32_fwd_sp = chunk4_results["float32"]["fwd_speedup"]
    f16_fwd_sp = chunk4_results["float16"]["fwd_speedup"]
    print(f"{'  -> Forward Pass Speedup':<35} | {'1.00x (ref)':<16} | {f32_fwd_sp:5.2f}x         | {f16_fwd_sp:5.2f}x")

    f32_fwd_sav = chunk4_results["float32"]["fwd_savings_ms"]
    f16_fwd_sav = chunk4_results["float16"]["fwd_savings_ms"]
    print(f"{'  -> Forward Pass Savings':<35} | {'-':<16} | {f32_fwd_sav:+5.2f} ms       | {f16_fwd_sav:+5.2f} ms")

    print("-" * 85)

    # 3. Full chunk 4
    c64_total = chunk4_results["float64"]["mean_chunk_total_ms"]
    c32_total = chunk4_results["float32"]["mean_chunk_total_ms"]
    c16_total = chunk4_results["float16"]["mean_chunk_total_ms"]
    print(f"{'Chunk 4 Total Rollout Latency':<35} | {c64_total:7.2f} ms       | {c32_total:7.2f} ms     | {c16_total:7.2f} ms")

    c32_sp = chunk4_results["float32"]["chunk_speedup"]
    c16_sp = chunk4_results["float16"]["chunk_speedup"]
    print(f"{'  -> Chunk 4 Total Speedup':<35} | {'1.00x (ref)':<16} | {c32_sp:5.2f}x         | {c16_sp:5.2f}x")

    c32_sav = chunk4_results["float32"]["chunk_savings_ms"]
    c16_sav = chunk4_results["float16"]["chunk_savings_ms"]
    print(f"{'  -> Chunk 4 Total Savings':<35} | {'-':<16} | {c32_sav:+6.2f} ms      | {c16_sav:+6.2f} ms")
    print("=" * 85)

    if args.out_json:
        full_report = {
            "microbenchmark": micro_results,
            "chunk4": chunk4_results,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(full_report, indent=2))
        print(f"\nSaved full benchmark report to: {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
