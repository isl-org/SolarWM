#!/usr/bin/env python3
"""Run one self-forcing diffusion forward from a CUDA rollout trace and compare flow/x0.

Uses the same provider.diffusion path as production Stage2 rollout (shared with
generate_wan22_stage2_cuda_reference.py). Intended to isolate the smallest forward
that diverges between CUDA (trace) and XPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEBUG_SCRIPTS = REPO_ROOT / "scripts" / "debug"
for path in (str(REPO_ROOT), str(DEBUG_SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from solarwm.backends.wan22.generation import resolve_generation_plan
from solarwm.backends.wan22.runtime.inference import _pass_case
from solarwm.backends.wan22.runtime.stage0p5 import expand_timesteps_to_tokens
from solarwm.backends.wan22.runtime.modeling.attention import wan22_attention_backend
from solarwm.backends.wan22.runtime.stage2 import (
    CudaWanStage2GenerationAdapter,
    _generation_steps,
    _restore_first,
    _slice_camera,
)

# Reuse trace helpers from the reference generator script.
from generate_wan22_stage2_cuda_reference import (
    DEFAULT_CONFIG,
    _compare_tensors,
    _pass_for_weights,
    _pick_device,
    _resolve_config,
    _restore_cache_tensors,
)


def _load_step_tensor(trace_dir: Path, chunk: int, step: int, suffix: str) -> torch.Tensor:
    path = trace_dir / "steps" / f"chunk{chunk:02d}_step{step:02d}_{suffix}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _run_single_forward(
    provider: Any,
    *,
    trace_dir: Path,
    chunk: int,
    step: int,
    inject_kv: bool,
    weights_role: str,
) -> dict[str, Any]:
    payload = torch.load(trace_dir / "conditions.pt", map_location="cpu", weights_only=True)
    first_latent = payload["first_latent"].to(device=provider.device, dtype=torch.bfloat16)
    condition = {
        "prompt_embeds": payload["prompt_embeds"].to(
            device=provider.device,
            dtype=torch.bfloat16,
        ),
    }
    camera = {
        "viewmats": payload["camera_viewmats"].to(device=provider.device),
        "K": payload["camera_K"].to(device=provider.device),
    }
    chunk_size = int(provider.config["model"]["num_frame_per_block"])
    frame_tokens = int(provider.config["model"]["frame_sequence_length"])
    start = chunk * chunk_size
    end = start + chunk_size
    latents = _load_step_tensor(trace_dir, chunk, step, "latents_in").to(
        device=provider.device,
        dtype=torch.bfloat16,
    )
    camera_chunk = _slice_camera(
        camera,
        start_frame=start,
        end_frame=end,
        frame_sequence_length=frame_tokens,
    )
    steps = _generation_steps(provider)
    if step < 0 or step >= len(steps):
        raise ValueError(f"step must be in [0, {len(steps) - 1}]")
    timestep_value = float(steps[step].item())
    timestep = torch.full((1, chunk_size), timestep_value, device=provider.device, dtype=torch.bfloat16)
    if start == 0:
        timestep[:, 0] = 0.0

    dtype = torch.bfloat16
    kv_cache = provider.allocate_kv_cache(1, dtype=dtype, device=provider.device)
    crossattn_cache = provider.allocate_crossattn_cache(1, dtype=dtype, device=provider.device)
    if inject_kv and chunk > 0:
        prev = chunk - 1
        kv_payload = torch.load(
            trace_dir / f"chunk{prev:02d}_kv_cache.pt",
            map_location="cpu",
            weights_only=True,
        )
        cross_payload = torch.load(
            trace_dir / f"chunk{prev:02d}_crossattn_cache.pt",
            map_location="cpu",
            weights_only=True,
        )
        _restore_cache_tensors(kv_payload, kv_cache, provider.device)
        _restore_cache_tensors(cross_payload, crossattn_cache, provider.device)

    provider._load_role(weights_role)
    with torch.no_grad(), torch.autocast(
        device_type=provider.device.type,
        dtype=torch.bfloat16,
        enabled=provider.device.type in ("cuda", "xpu"),
    ):
        flow = provider.diffusion(
            latents,
            condition,
            camera_chunk,
            expand_timesteps_to_tokens(timestep, frame_tokens),
            sequence_length=chunk_size * frame_tokens,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=start * frame_tokens,
            cache_start=0,
            cache_update_policy="none",
        )
        x0 = provider.diffusion.flow_to_x0(latents, flow, timestep)
    if start == 0:
        x0 = _restore_first(x0, first_latent, start_frame=0)

    latents_in_ref = _load_step_tensor(trace_dir, chunk, step, "latents_in")
    report: dict[str, Any] = {
        "device": provider.device.type,
        "attention_backend": wan22_attention_backend(),
        "chunk": chunk,
        "step": step,
        "timestep": timestep_value,
        "inject_kv": inject_kv and chunk > 0,
        "latents_in_vs_cuda_trace": _compare_tensors(latents_in_ref, latents.cpu()),
        "latents_in_stats": {
            "shape": list(latents.shape),
            "mean": float(latents.float().mean().item()),
        },
    }
    for suffix, tensor in (("flow", flow), ("x0", x0)):
        ref_path = trace_dir / "steps" / f"chunk{chunk:02d}_step{step:02d}_{suffix}.pt"
        if ref_path.is_file():
            reference = torch.load(ref_path, map_location="cpu", weights_only=True)
            report[f"{suffix}_vs_cuda_trace"] = _compare_tensors(reference, tensor.cpu())
        else:
            report[f"{suffix}_vs_cuda_trace"] = {"match": False, "reason": f"missing {ref_path}"}
    del kv_cache, crossattn_cache
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "xpu", "cpu"), required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--inject-kv", action="store_true", help="Restore CUDA KV before chunk>=1")
    parser.add_argument("--weights", choices=("live", "ema"), default="ema")
    parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON report here")
    args = parser.parse_args()

    trace_dir = args.trace_dir.expanduser().resolve()
    if not (trace_dir / "manifest.json").is_file():
        raise SystemExit(f"error: missing manifest in {trace_dir}")

    device = _pick_device(args.device)
    overrides = list(args.overrides)
    overrides.append(f"inference.device={device.type}")
    config = _resolve_config(overrides)
    plan = resolve_generation_plan(config)
    provider = CudaWanStage2GenerationAdapter(config, plan)
    try:
        report = _run_single_forward(
            provider,
            trace_dir=trace_dir,
            chunk=int(args.chunk),
            step=int(args.step),
            inject_kv=bool(args.inject_kv),
            weights_role=args.weights,
        )
        text = json.dumps(report, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        print(text)
        worst = max(
            (
                report.get("flow_vs_cuda_trace", {}).get("max_abs_diff", 0.0),
                report.get("x0_vs_cuda_trace", {}).get("max_abs_diff", 0.0),
            )
        )
        return 0 if worst < 1e-2 else 1
    finally:
        provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
