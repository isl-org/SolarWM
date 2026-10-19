"""Benchmark the full 360p (and 480p) world model end-to-end on Intel Arc XPU.
Measures actual latency across all stages:
1. Condition encoding: VAE anchor image encode + text encoder
2. DiT diffusion steps per chunk: Step 0, Step 1, Step 2, Step 3, Commit forward
3. VAE streaming decoding per chunk
4. Total latency per chunk and generated FPS
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import yaml

from solarwm.backends.wan22.runtime.stage2 import build_stage2_generation_provider
from solarwm.backends.wan22.runtime.interactive import Stage2InteractiveSession
from solarwm.backends.wan22.runtime.interactive_camera import FreeFlyCamera, camera_tokens_from_poses


def sync(device):
    if device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(
    config_path: str,
    attention_backend: str = "sdpa",
    linear_quant: str = "svdquant_k4",
    stage2_vae_quant: str = "quarot_w8a8",
    num_chunks: int = 6,
) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["model"]["attention_backend"] = attention_backend
    config["model"]["linear_quant"] = linear_quant
    config.setdefault("runtime", {})["stage2_vae_quant"] = stage2_vae_quant
    config["model"]["num_frame_per_block"] = 3

    print(f"\n=======================================================")
    print(f"Running: {config['data']['height']}x{config['data']['width']} "
          f"({config['model']['frame_sequence_length']} tokens/frame)")
    print(f"Backend: linear={linear_quant}, attn={attention_backend}, vae_quant={stage2_vae_quant}")
    print(f"=======================================================")

    t0_load = time.perf_counter()
    provider = build_stage2_generation_provider(config)
    provider._load_role("model")
    device = provider.device
    sync(device)
    load_time_sec = time.perf_counter() - t0_load
    print(f"Model loaded in {load_time_sec:.2f}s")

    # 1. Condition Encoding Benchmark
    image_path = "assets/eastshade.webp"
    image = Image.open(image_path).convert("RGB").resize(
        (int(config["data"]["width"]), int(config["data"]["height"])),
        Image.Resampling.BICUBIC,
    )
    pixels = (
        torch.from_numpy(np.asarray(image))
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .div(127.5)
        .sub(1.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=provider.device, dtype=torch.bfloat16)
    )

    sync(device)
    t0_cond = time.perf_counter()
    with torch.no_grad():
        t0_vae_enc = time.perf_counter()
        first_latent = provider.vae.encode(
            pixels[:, :1].permute(0, 2, 1, 3, 4).contiguous()
        ).to(torch.bfloat16)
        sync(device)
        vae_enc_ms = (time.perf_counter() - t0_vae_enc) * 1000.0

        t0_t5 = time.perf_counter()
        condition = provider.text_encoder(["Walk forward smoothly."])
        prompt_embeds = condition["prompt_embeds"]
        if prompt_embeds.device != provider.device:
            condition = {
                **condition,
                "prompt_embeds": prompt_embeds.to(
                    device=provider.device,
                    dtype=torch.bfloat16,
                ),
            }
        sync(device)
        t5_ms = (time.perf_counter() - t0_t5) * 1000.0
    sync(device)
    cond_total_ms = (time.perf_counter() - t0_cond) * 1000.0
    print(f"Stage 1 (Condition Encoding): VAE encode={vae_enc_ms:.2f}ms, T5 prompt={t5_ms:.2f}ms, Total={cond_total_ms:.2f}ms")

    # 2. Instrument provider.diffusion to record exact per-step timings
    current_chunk = [0]
    chunk_step_times: dict[int, dict[str | int, float]] = {}

    orig_diffusion = provider.diffusion
    class StepTimingWrapper:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __call__(self, *args, **kwargs):
            radial_step = kwargs.get("radial_step", None)
            step_key = "commit" if radial_step == 4 else radial_step
            sync(device)
            t0 = time.perf_counter()
            out = self.inner(*args, **kwargs)
            sync(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            c_idx = current_chunk[0]
            chunk_step_times.setdefault(c_idx, {})[step_key] = elapsed_ms
            return out

    provider.diffusion = StepTimingWrapper(provider.diffusion)

    camera = FreeFlyCamera()
    decode_context = provider.vae.streaming_decode_session()
    decode_tile = decode_context.__enter__()

    max_latent_frames = num_chunks * 3
    session = Stage2InteractiveSession.from_provider(
        provider,
        first_latent=first_latent,
        condition=condition,
        seed=42,
        max_latent_frames=max_latent_frames,
        decode_tile=decode_tile,
        first_pose=camera.pose,
    )

    chunk_records = []
    chunk_idx = 0
    try:
        while not session.done and chunk_idx < num_chunks:
            current_chunk[0] = chunk_idx
            poses = [camera.pose.copy() for _ in range(session.chunk)]
            camera_tokens = camera_tokens_from_poses(
                poses,
                origin=camera._origin,
                frame_sequence_length=session.frame_sequence_length,
                device=session.device,
                dtype=session.dtype,
            )

            sync(device)
            t0_step = time.perf_counter()
            frame = session.step(camera_tokens, pose=camera.pose)
            sync(device)
            step_total_ms = (time.perf_counter() - t0_step) * 1000.0

            dit_ms = frame.diffusion_seconds * 1000.0
            vae_ms = frame.decode_seconds * 1000.0
            emitted_frames = frame.pixels.shape[1]
            fps = emitted_frames / (step_total_ms / 1000.0)

            steps_breakdown = chunk_step_times.get(chunk_idx, {})

            chunk_records.append({
                "chunk_idx": chunk_idx,
                "latent_range": [frame.index, frame.index + session.chunk],
                "emitted_frames": emitted_frames,
                "dit_ms": dit_ms,
                "vae_decode_ms": vae_ms,
                "total_chunk_ms": step_total_ms,
                "fps": fps,
                "steps_breakdown": steps_breakdown,
            })

            steps_str = " | ".join(f"s{s}:{t:.1f}ms" for s, t in sorted(steps_breakdown.items(), key=lambda x: str(x[0])))
            print(
                f"Chunk {chunk_idx}: DiT={dit_ms:6.1f}ms ({steps_str}) | "
                f"VAE={vae_ms:5.1f}ms | Total={step_total_ms:6.1f}ms | FPS={fps:4.1f}"
            )
            chunk_idx += 1

    finally:
        decode_context.__exit__(None, None, None)
        provider.close()
        del provider
        gc.collect()
        if device.type == "xpu":
            torch.xpu.empty_cache()

    # Steady state (chunks 3 to end, typically chunk 4)
    steady_chunks = chunk_records[2:] if len(chunk_records) > 2 else chunk_records
    steady_dit_ms = float(np.median([c["dit_ms"] for c in steady_chunks]))
    steady_vae_ms = float(np.median([c["vae_decode_ms"] for c in steady_chunks]))
    steady_total_ms = float(np.median([c["total_chunk_ms"] for c in steady_chunks]))
    steady_fps = float(np.median([c["fps"] for c in steady_chunks]))

    print(f"\n--- Steady State (Median of Chunks 2-{len(chunk_records)-1}) ---")
    print(f"DiT:   {steady_dit_ms:6.1f} ms")
    print(f"VAE:   {steady_vae_ms:6.1f} ms")
    print(f"Total: {steady_total_ms:6.1f} ms")
    print(f"FPS:   {steady_fps:6.1f} frames/sec")

    return {
        "config_path": config_path,
        "resolution": f"{config['data']['height']}x{config['data']['width']}",
        "frame_sequence_length": config["model"]["frame_sequence_length"],
        "linear_quant": linear_quant,
        "attention_backend": attention_backend,
        "stage2_vae_quant": stage2_vae_quant,
        "condition_encoding": {
            "vae_enc_ms": vae_enc_ms,
            "t5_ms": t5_ms,
            "total_ms": cond_total_ms,
        },
        "chunks": chunk_records,
        "steady_state": {
            "dit_ms": steady_dit_ms,
            "vae_decode_ms": steady_vae_ms,
            "total_chunk_ms": steady_total_ms,
            "fps": steady_fps,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-360p", default="configs/examples/wan22_ti2v_5b/infer_stage2_sgf_interactive_360p_xpu.local.yaml")
    parser.add_argument("--config-480p", default="configs/examples/wan22_ti2v_5b/infer_stage2_sgf_interactive_xpu.local.yaml")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--quant", default="svdquant_k4")
    parser.add_argument("--vae-quant", default="quarot_w8a8")
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--compare-480p", action="store_true")
    parser.add_argument("--output-json", default="outputs/benchmark_360p_vs_480p.json")
    args = parser.parse_args()

    results = {}
    # Run 360p
    res_360p = run_benchmark(
        args.config_360p,
        attention_backend=args.attn,
        linear_quant=args.quant,
        stage2_vae_quant=args.vae_quant,
        num_chunks=args.chunks,
    )
    results["360p"] = res_360p

    if args.compare_480p:
        res_480p = run_benchmark(
            args.config_480p,
            attention_backend=args.attn,
            linear_quant=args.quant,
            stage2_vae_quant=args.vae_quant,
            num_chunks=args.chunks,
        )
        results["480p"] = res_480p

        print("\n=======================================================")
        print("Comparison Summary (Steady State):")
        print(f"480p DiT: {res_480p['steady_state']['dit_ms']:.1f}ms | VAE: {res_480p['steady_state']['vae_decode_ms']:.1f}ms | Total: {res_480p['steady_state']['total_chunk_ms']:.1f}ms | FPS: {res_480p['steady_state']['fps']:.1f}")
        print(f"360p DiT: {res_360p['steady_state']['dit_ms']:.1f}ms | VAE: {res_360p['steady_state']['vae_decode_ms']:.1f}ms | Total: {res_360p['steady_state']['total_chunk_ms']:.1f}ms | FPS: {res_360p['steady_state']['fps']:.1f}")
        speedup = res_480p['steady_state']['total_chunk_ms'] / res_360p['steady_state']['total_chunk_ms']
        print(f"Speedup: {speedup:.2f}x")
        print("=======================================================")

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote full benchmark report to {out_file}")
