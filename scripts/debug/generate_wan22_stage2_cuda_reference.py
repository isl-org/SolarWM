#!/usr/bin/env python3
"""Build or compare Wan2.2 Stage2 rollout latents and VAE decode references.

CUDA machine (reference + full rollout trace for XPU replay):
  python scripts/debug/generate_wan22_stage2_cuda_reference.py --device cuda --output-dir /path/out \\
    --save-rollout-trace

XPU machine (probe + optional compare):
  python scripts/debug/generate_wan22_stage2_cuda_reference.py --device xpu --output-dir /path/out \\
    --compare-reference /path/to/cuda/reference \\
    --replay-rollout-trace /path/to/cuda/reference/rollout_trace --replay-mode kv_chunks

Environment (same as inference):
  SOLAR_MODEL_ROOT, SOLAR_DATA_ROOT (releases-v1), optional SOLAR_REPO
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solarwm.config import load_config
from solarwm.backends.wan22.generation import GenerationPass, resolve_generation_plan
from solarwm.inference.engine import InferenceCase
from solarwm.backends.wan22.runtime.inference import _pass_case
from solarwm.backends.wan22.runtime.stage0p5 import expand_timesteps_to_tokens
from solarwm.backends.wan22.runtime.stage2 import (
    CudaWanStage2GenerationAdapter,
    _generation_steps,
    _restore_first,
    _slice_camera,
    _stage2_self_forcing_latents,
)

TRACE_SCHEMA = "solarwm.wan22-stage2-rollout-trace.v1"

DEFAULT_CONFIG = REPO_ROOT / "configs/examples/wan22_ti2v_5b/infer_stage2_sgf_81f.yaml"
STREAMING_CHUNK = 60


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(tensor)
    finite_fraction = float(finite.float().mean().item()) if tensor.numel() else 1.0
    safe = tensor.detach().float().cpu()
    if finite_fraction < 1.0:
        safe = safe[finite.cpu()]
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "finite_fraction": finite_fraction,
        "min": float(safe.min().item()) if safe.numel() else None,
        "max": float(safe.max().item()) if safe.numel() else None,
        "mean": float(safe.mean().item()) if safe.numel() else None,
        "std": float(safe.std().item()) if safe.numel() > 1 else 0.0,
    }


def _resolve_config(overrides: list[str]) -> dict[str, Any]:
    model_root = _env_path("SOLAR_MODEL_ROOT", str(REPO_ROOT / "models/SolarWM"))
    data_root = _env_path(
        "SOLAR_DATA_ROOT",
        str(REPO_ROOT / "data/SolarWM-Data/releases-v1"),
    )
    base = [
        f"model.base_path={model_root / 'SolarWM-5B-base'}",
        f"checkpoint.path={model_root / 'SolarWM-5B-sgf-stage2-81f'}",
        f"data.index_root={data_root / 'example'}",
        f"data.transport.root={data_root / 'example'}",
        "data.test_index=smoke-index.jsonl.gz",
        "validation.sample_count=1",
        "data.num_workers=1",
    ]
    resolved = load_config(DEFAULT_CONFIG, overrides=base + overrides)
    return resolved.mutable_copy()


def _pick_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("error: --device cuda but torch.cuda.is_available() is False")
        return torch.device("cuda", 0)
    if requested == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise SystemExit("error: --device xpu but torch.xpu is not available")
        return torch.device("xpu", 0)
    if requested == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"unknown device: {requested}")


def _output_latent_frames(case_metadata: Mapping[str, Any], generation_pass: Any) -> int:
    metadata = case_metadata.get("generation_pass", {})
    if isinstance(metadata, Mapping) and "output_rollout_latent_frames" in metadata:
        return int(metadata["output_rollout_latent_frames"])
    return int(generation_pass.rollout_latent_frames)


@dataclass
class _RolloutTraceReplay:
    """Inject CUDA-captured tensors so the probe device never draws RNG."""

    trace_dir: Path
    mode: str  # noise | conditions | kv_chunks

    def _load(self, relative: str) -> torch.Tensor:
        return torch.load(self.trace_dir / relative, map_location="cpu", weights_only=True)

    def load_conditions(self) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        payload = torch.load(self.trace_dir / "conditions.pt", map_location="cpu", weights_only=True)
        first = payload["first_latent"]
        condition = {"prompt_embeds": payload["prompt_embeds"]}
        camera = {"viewmats": payload["camera_viewmats"], "K": payload["camera_K"]}
        return first, condition, camera

    def initial_noise(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._load("initial_noise.pt").to(device=device, dtype=dtype)

    def renoise(self, chunk: int, step: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        path = self.trace_dir / "steps" / f"chunk{chunk:02d}_step{step:02d}_renoise.pt"
        return torch.load(path, map_location="cpu", weights_only=True).to(device=device, dtype=dtype)

    def compare_step_tensor(self, chunk: int, step: int, suffix: str, tensor: torch.Tensor) -> dict[str, Any] | None:
        path = self.trace_dir / "steps" / f"chunk{chunk:02d}_step{step:02d}_{suffix}.pt"
        if not path.is_file():
            return None
        reference = torch.load(path, map_location="cpu", weights_only=True)
        return {
            "chunk": chunk,
            "step": step,
            "tensor": suffix,
            **_compare_tensors(reference, tensor.cpu()),
        }

    def load_kv_snapshot(self, chunk: int, kv_cache: list, crossattn_cache: list, device: torch.device) -> None:
        if self.mode != "kv_chunks":
            return
        if chunk <= 0:
            return
        prev = chunk - 1
        kv_payload = torch.load(
            self.trace_dir / f"chunk{prev:02d}_kv_cache.pt",
            map_location="cpu",
            weights_only=True,
        )
        cross_payload = torch.load(
            self.trace_dir / f"chunk{prev:02d}_crossattn_cache.pt",
            map_location="cpu",
            weights_only=True,
        )
        _restore_cache_tensors(kv_payload, kv_cache, device)
        _restore_cache_tensors(cross_payload, crossattn_cache, device)


def _serialize_cache_blocks(caches: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    payload: dict[str, torch.Tensor] = {}
    for block_index, block in enumerate(caches):
        for key, value in block.items():
            if key.startswith("_"):
                if isinstance(value, Mapping):
                    for sub_key, tensor in value.items():
                        if torch.is_tensor(tensor):
                            payload[f"b{block_index}_{key}_{sub_key}"] = tensor.detach().cpu()
                continue
            if torch.is_tensor(value):
                payload[f"b{block_index}_{key}"] = value.detach().cpu()
    return payload


def _restore_cache_tensors(
    payload: Mapping[str, torch.Tensor],
    caches: list[dict[str, Any]],
    device: torch.device,
) -> None:
    for block_index, block in enumerate(caches):
        for key, value in block.items():
            if key.startswith("_"):
                if isinstance(value, Mapping):
                    for sub_key, tensor in value.items():
                        if not torch.is_tensor(tensor):
                            continue
                        name = f"b{block_index}_{key}_{sub_key}"
                        if name in payload:
                            tensor.copy_(
                                payload[name].to(device=device, dtype=tensor.dtype),
                            )
                continue
            if torch.is_tensor(value):
                name = f"b{block_index}_{key}"
                if name in payload:
                    value.copy_(payload[name].to(device=device, dtype=value.dtype))


def _self_forcing_rollout(
    provider: Any,
    generation_pass: Any,
    first_latent: torch.Tensor,
    condition: Mapping[str, Any],
    camera: Mapping[str, Any],
    generator: torch.Generator | None,
    *,
    trace_dir: Path | None = None,
    replay: _RolloutTraceReplay | None = None,
    output_latent_frames: int | None = None,
) -> tuple[torch.Tensor, Mapping[str, Any], dict[str, Any]]:
    """Self-forcing NFE4 rollout with optional CUDA trace capture or XPU replay."""

    if (
        str(generation_pass.solver) != "self_forcing"
        or int(generation_pass.num_inference_steps) != 4
    ):
        raise RuntimeError("trace rollout supports self_forcing NFE4 only")

    latent_frames = int(generation_pass.rollout_latent_frames)
    chunk = int(provider.config["model"]["num_frame_per_block"])
    channels = int(provider.config["model"]["latent_channels"])
    latent_height = int(provider.config["data"]["latent_shape"][-2])
    latent_width = int(provider.config["data"]["latent_shape"][-1])
    frame_tokens = int(provider.config["model"]["frame_sequence_length"])
    device = provider.device
    dtype = torch.bfloat16

    output = torch.zeros(1, latent_frames, channels, latent_height, latent_width, device=device, dtype=dtype)
    output[:, 0] = first_latent[:, 0]
    if replay is not None:
        initial_noise = replay.initial_noise(device, dtype)
    else:
        if generator is None:
            raise RuntimeError("generator required when not replaying trace noise")
        initial_noise = provider._noise(tuple(output.shape), generator)
    initial_noise[:, 0] = first_latent[:, 0]

    steps = _generation_steps(provider)
    kv_cache = provider.allocate_kv_cache(1, dtype=dtype, device=device)
    crossattn_cache = provider.allocate_crossattn_cache(1, dtype=dtype, device=device)
    if kv_cache and "_fused_prope_camera_metadata" not in kv_cache[0]:
        raise RuntimeError("Stage2 rollout trace requires preallocated fused camera metadata")

    step_compare: list[dict[str, Any]] = []
    chunk_compare: list[dict[str, Any]] = []
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "steps").mkdir(exist_ok=True)
        torch.save(initial_noise.detach().cpu(), trace_dir / "initial_noise.pt")
        torch.save(
            {
                "first_latent": first_latent.detach().cpu(),
                "prompt_embeds": condition["prompt_embeds"].detach().cpu(),
                "camera_viewmats": camera["viewmats"].detach().cpu(),
                "camera_K": camera["K"].detach().cpu(),
            },
            trace_dir / "conditions.pt",
        )

    chunk_index = 0
    try:
        for start in range(0, latent_frames, chunk):
            end = start + chunk
            if replay is not None:
                replay.load_kv_snapshot(chunk_index, kv_cache, crossattn_cache, device)
            latents = initial_noise[:, start:end].clone()
            camera_chunk = _slice_camera(
                camera,
                start_frame=start,
                end_frame=end,
                frame_sequence_length=frame_tokens,
            )
            for step_index, step in enumerate(steps):
                timestep = torch.full((1, chunk), float(step.item()), device=device, dtype=dtype)
                if start == 0:
                    timestep[:, 0] = 0.0
                step_prefix = trace_dir / "steps" if trace_dir else None
                if step_prefix is not None:
                    torch.save(
                        latents.detach().cpu(),
                        step_prefix / f"chunk{chunk_index:02d}_step{step_index:02d}_latents_in.pt",
                    )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type in ("cuda", "xpu"),
                ):
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
                        cache_update_policy="none",
                    )
                    x0 = provider.diffusion.flow_to_x0(latents, flow, timestep)
                if start == 0:
                    x0 = _restore_first(x0, first_latent, start_frame=0)
                if step_prefix is not None:
                    torch.save(
                        flow.detach().cpu(),
                        step_prefix / f"chunk{chunk_index:02d}_step{step_index:02d}_flow.pt",
                    )
                    torch.save(
                        x0.detach().cpu(),
                        step_prefix / f"chunk{chunk_index:02d}_step{step_index:02d}_x0.pt",
                    )
                elif replay is not None:
                    for suffix, value in (("flow", flow), ("x0", x0)):
                        entry = replay.compare_step_tensor(chunk_index, step_index, suffix, value)
                        if entry is not None:
                            step_compare.append(entry)
                if step_index + 1 < len(steps):
                    next_timestep = torch.full(
                        (1, chunk),
                        float(steps[step_index + 1].item()),
                        device=device,
                        dtype=torch.float32,
                    )
                    if start == 0:
                        next_timestep[:, 0] = 0.0
                    if replay is not None:
                        renoise = replay.renoise(chunk_index, step_index, device, dtype)
                    else:
                        renoise = provider._noise(tuple(x0.shape), generator)
                    if step_prefix is not None:
                        torch.save(
                            renoise.detach().cpu(),
                            step_prefix / f"chunk{chunk_index:02d}_step{step_index:02d}_renoise.pt",
                        )
                    latents = (
                        provider.diffusion.scheduler.add_noise(
                            x0.flatten(0, 1).float(),
                            renoise.flatten(0, 1).float(),
                            next_timestep.flatten(0, 1),
                        )
                        .unflatten(0, (1, chunk))
                        .to(dtype)
                    )
                    if start == 0:
                        latents = _restore_first(latents, first_latent, start_frame=0)
                else:
                    latents = x0
                if step_prefix is not None:
                    torch.save(
                        latents.detach().cpu(),
                        step_prefix / f"chunk{chunk_index:02d}_step{step_index:02d}_latents_out.pt",
                    )
                elif replay is not None:
                    entry = replay.compare_step_tensor(
                        chunk_index,
                        step_index,
                        "latents_out",
                        latents,
                    )
                    if entry is not None:
                        step_compare.append(entry)
            output[:, start:end] = latents
            commit_timestep = torch.zeros((1, chunk), device=device, dtype=dtype)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type in ("cuda", "xpu"),
            ):
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
            if replay is not None:
                committed_ref = replay.trace_dir / f"chunk{chunk_index:02d}_committed_latents.pt"
                if committed_ref.is_file():
                    reference = torch.load(committed_ref, map_location="cpu", weights_only=True)
                    chunk_compare.append(
                        {
                            "chunk": chunk_index,
                            "committed_latents": _compare_tensors(reference, latents.cpu()),
                        },
                    )
            if trace_dir is not None:
                torch.save(
                    latents.detach().cpu(),
                    trace_dir / f"chunk{chunk_index:02d}_committed_latents.pt",
                )
                torch.save(
                    _serialize_cache_blocks(kv_cache),
                    trace_dir / f"chunk{chunk_index:02d}_kv_cache.pt",
                )
                torch.save(
                    _serialize_cache_blocks(crossattn_cache),
                    trace_dir / f"chunk{chunk_index:02d}_crossattn_cache.pt",
                )
            chunk_index += 1
    finally:
        del kv_cache, crossattn_cache

    schedule = {
        "schema": "solarwm.wan22-stage2-self-forcing-schedule.v1",
        "solver": "self_forcing",
        "timesteps": [float(value.item()) for value in steps],
        "chunk_latent_frames": chunk,
    }
    trace_meta: dict[str, Any] = {}
    if output_latent_frames is not None:
        trimmed = output[:, :output_latent_frames].contiguous()
        if trace_dir is not None:
            torch.save(trimmed.detach().cpu(), trace_dir / "output_latents.pt")
            manifest = {
                "schema": TRACE_SCHEMA,
                "latent_frames": latent_frames,
                "output_latent_frames": output_latent_frames,
                "chunk_latent_frames": chunk,
                "num_chunks": chunk_index,
                "num_steps_per_chunk": len(steps),
                "sample_fields": [
                    "conditions.pt",
                    "initial_noise.pt",
                    "output_latents.pt",
                    "chunk##_committed_latents.pt",
                    "chunk##_kv_cache.pt",
                    "chunk##_crossattn_cache.pt",
                    "steps/chunk##_step##_*.pt",
                ],
            }
            (trace_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        trace_meta["trace_dir"] = str(trace_dir)
    else:
        trimmed = output
    if replay is not None:
        trace_meta["replay_mode"] = replay.mode
        trace_meta["step_compare"] = step_compare
        trace_meta["chunk_compare"] = chunk_compare
        if step_compare:
            worst = max(step_compare, key=lambda row: row.get("max_abs_diff", 0.0))
            trace_meta["step_compare_worst"] = worst
    return trimmed, schedule, trace_meta


def _rollout_trace_report(
    probe_output: torch.Tensor,
    trace_meta: Mapping[str, Any],
    trace_dir: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "trace_dir": str(trace_dir),
        "replay_mode": trace_meta.get("replay_mode"),
    }
    ref_out = trace_dir / "output_latents.pt"
    if ref_out.is_file():
        reference = torch.load(ref_out, map_location="cpu", weights_only=True)
        probe_cpu = probe_output.cpu()
        if reference.shape != probe_cpu.shape and reference.ndim >= 2 and probe_cpu.ndim >= 2:
            frames = min(int(reference.shape[1]), int(probe_cpu.shape[1]))
            reference = reference[:, :frames]
            probe_cpu = probe_cpu[:, :frames]
        report["output_latents_vs_cuda_trace"] = _compare_tensors(reference, probe_cpu)
    if "step_compare" in trace_meta:
        report["per_step_vs_cuda_trace"] = trace_meta["step_compare"]
        report["per_step_worst"] = trace_meta.get("step_compare_worst")
    if "chunk_compare" in trace_meta:
        report["per_chunk_committed_vs_cuda_trace"] = trace_meta["chunk_compare"]
    return report


def _run_rollout(
    provider: Any,
    case: Any,
    generation_pass: Any,
    *,
    trace_dir: Path | None = None,
    replay: _RolloutTraceReplay | None = None,
) -> tuple[torch.Tensor, Mapping[str, Any], torch.Tensor, dict[str, Any]]:
    generator: torch.Generator | None = None
    trace_meta: dict[str, Any] = {}
    if replay is None:
        generator = torch.Generator(device=provider.device)
        generator.manual_seed(int(case.noise_seed))
        first, condition, camera, model_y = provider._conditions(
            case,
            latent_frames=int(generation_pass.rollout_latent_frames),
        )
    elif replay.mode in ("conditions", "kv_chunks"):
        first_cpu, condition_cpu, camera_cpu = replay.load_conditions()
        first = first_cpu.to(device=provider.device, dtype=torch.bfloat16)
        condition = {
            "prompt_embeds": condition_cpu["prompt_embeds"].to(
                device=provider.device,
                dtype=torch.bfloat16,
            ),
        }
        camera = {
            "viewmats": camera_cpu["viewmats"].to(device=provider.device),
            "K": camera_cpu["K"].to(device=provider.device),
        }
        model_y = None
    else:
        first, condition, camera, model_y = provider._conditions(
            case,
            latent_frames=int(generation_pass.rollout_latent_frames),
        )
    if model_y is not None:
        raise RuntimeError("Stage2 TI2V rollout does not expect I2V y")
    output_frames = _output_latent_frames(case.metadata, generation_pass)
    with torch.no_grad():
        if trace_dir is not None or replay is not None:
            latents, schedule, trace_meta = _self_forcing_rollout(
                provider,
                generation_pass,
                first,
                condition,
                camera,
                generator,
                trace_dir=trace_dir,
                replay=replay,
                output_latent_frames=output_frames,
            )
        else:
            latents, schedule = _stage2_self_forcing_latents(
                provider,
                generation_pass,
                first,
                condition,
                camera,
                generator,
            )
            latents = latents[:, :output_frames].contiguous()
    return latents, schedule, first, trace_meta


def _decode_direct(
    vae: Any,
    latents_btchw: torch.Tensor,
    *,
    autocast: bool,
    autocast_dtype: torch.dtype,
) -> torch.Tensor:
    clips = latents_btchw.permute(0, 2, 1, 3, 4)
    decode_fn = vae.module.decode
    weight_dtype = next(vae.module.parameters()).dtype
    outputs = []
    for clip in clips:
        if autocast:
            with torch.autocast(device_type=clip.device.type, dtype=autocast_dtype):
                decoded = decode_fn(clip.unsqueeze(0), vae._scale(clip))
        else:
            clip_matched = clip.to(device=clip.device, dtype=weight_dtype)
            decoded = decode_fn(clip.unsqueeze(0), vae._scale(clip_matched))
        outputs.append(decoded.float().clamp_(-1, 1).squeeze(0))
    return torch.stack(outputs, dim=0).permute(0, 2, 1, 3, 4)


def _decode_streaming(
    vae: Any,
    latents_btchw: torch.Tensor,
    *,
    chunk_latent_frames: int,
) -> torch.Tensor:
    return vae.decode_streaming(latents_btchw, chunk_latent_frames=chunk_latent_frames)


def _probe_decodes(vae: Any, latents: torch.Tensor) -> dict[str, Any]:
    results: dict[str, Any] = {}
    variants = [
        ("bf16_autocast_direct", {"autocast": True, "autocast_dtype": torch.bfloat16}),
        ("fp32_no_autocast_direct", {"autocast": False, "autocast_dtype": torch.float32}),
    ]
    for name, kwargs in variants:
        try:
            pixels = _decode_direct(vae, latents, **kwargs)
            results[name] = _tensor_stats(pixels)
            results[name]["ok"] = results[name]["finite_fraction"] == 1.0
        except Exception as exc:
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if int(latents.shape[1]) > STREAMING_CHUNK:
        try:
            pixels = _decode_streaming(vae, latents, chunk_latent_frames=STREAMING_CHUNK)
            results["bf16_streaming_cached"] = _tensor_stats(pixels)
            results["bf16_streaming_cached"]["ok"] = (
                results["bf16_streaming_cached"]["finite_fraction"] == 1.0
            )
        except Exception as exc:
            results["bf16_streaming_cached"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return results


def _apply_rollout_horizon(
    generation_pass: GenerationPass,
    case: InferenceCase,
    *,
    rollout_latent_frames: int,
    output_latent_frames: int | None,
) -> tuple[GenerationPass, InferenceCase]:
    """Shrink internal rollout and trim output for partial CUDA trace replay."""

    chunk = 3
    if rollout_latent_frames <= 0 or rollout_latent_frames % chunk:
        raise ValueError(f"rollout_latent_frames must be a positive multiple of {chunk}")
    out_frames = output_latent_frames if output_latent_frames is not None else rollout_latent_frames
    if out_frames > rollout_latent_frames:
        raise ValueError("output_latent_frames cannot exceed rollout_latent_frames")
    generation_pass = replace(
        generation_pass,
        rollout_latent_frames=rollout_latent_frames,
        min_rollout_latent_frames=min(
            generation_pass.min_rollout_latent_frames,
            rollout_latent_frames,
        ),
        fixed_plan_pixel_frames=1 + 4 * (rollout_latent_frames - 1),
        variable_rollout_by_source=False,
    )
    metadata = dict(case.metadata)
    pass_metadata = dict(metadata.get("generation_pass", {}))
    pass_metadata.update(
        {
            "rollout_latent_frames": rollout_latent_frames,
            "output_rollout_latent_frames": out_frames,
            "min_rollout_latent_frames": generation_pass.min_rollout_latent_frames,
            "fixed_plan_pixel_frames": generation_pass.fixed_plan_pixel_frames,
            "variable_rollout_by_source": False,
        },
    )
    metadata["generation_pass"] = pass_metadata
    rollouts = metadata.get("rollout_latent_frames_by_pass")
    if isinstance(rollouts, Mapping):
        metadata["rollout_latent_frames_by_pass"] = {
            str(name): rollout_latent_frames for name in rollouts
        }
    return generation_pass, replace(case, metadata=metadata)


def _pass_for_weights(plan: Any, weights: str) -> Any:
    for item in plan.passes:
        if str(item.weights) == weights:
            return item
    raise RuntimeError(f"no generation pass with weights={weights!r}")


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    ref = a.detach().float().cpu()
    cand = b.detach().float().cpu()
    if ref.shape != cand.shape:
        return float("nan")
    return float(((ref - cand) ** 2).mean().item())


def _save_rgb_frame(tensor_tchw: torch.Tensor, path: Path) -> None:
    """Save one TCHW frame in [-1, 1] as PNG via torchvision or raw uint8."""

    import numpy as np
    from PIL import Image

    frame = tensor_tchw.detach().float().cpu()
    if frame.ndim != 3 or int(frame.shape[0]) != 3:
        raise ValueError(f"expected CHW RGB, got {tuple(frame.shape)}")
    rgb = ((frame.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def _diagnose_frame0(
    provider: Any,
    case: Any,
    generation_pass: Any,
    latents: torch.Tensor,
    rollout_first_latent: torch.Tensor,
    out: Path,
) -> dict[str, Any]:
    """Latent pin, VAE round-trip, and temporal-decode coupling for pixel frame 0."""

    prepared = provider._prepared.get(case.slot)
    if prepared is None:
        raise RuntimeError("case was not prepared; call build_cases before diagnose")

    first_latent = rollout_first_latent
    pin_diff = (latents[:, :1].float() - first_latent[:, :1].float()).abs()
    pin = {
        "first_latent_shape": list(first_latent.shape),
        "latents_shape": list(latents.shape),
        "max_abs_diff": float(pin_diff.max().item()),
        "mean_abs_diff": float(pin_diff.mean().item()),
        "matches_tol_1e-2": bool(pin_diff.max().item() < 1e-2),
        "matches_tol_1e-3": bool(pin_diff.max().item() < 1e-3),
    }

    source = prepared.pixels[0].to(device=provider.device, dtype=torch.bfloat16)
    pixels_btchw = prepared.pixels[:1].unsqueeze(0).to(
        device=provider.device,
        dtype=torch.bfloat16,
    )
    pixels_bcthw = pixels_btchw[:, :1].permute(0, 2, 1, 3, 4).contiguous()
    with torch.no_grad():
        encoded0 = provider.vae.encode(pixels_bcthw).to(torch.bfloat16)
        post_rollout_reencode_diff = float(
            (encoded0[:, :1].float() - first_latent[:, :1].float()).abs().max().item()
        )
        roundtrip = provider.vae.decode(encoded0, use_cache=False)
    pin["post_rollout_reencode_vs_rollout_first_max"] = post_rollout_reencode_diff
    roundtrip_mse = _mse(source, roundtrip[0, 0])

    with torch.no_grad():
        full_decode = provider.vae.decode(latents, use_cache=False)
        single_latent = latents[:, :1].contiguous()
        decode_one = provider.vae.decode(single_latent, use_cache=False)

    gen_f0 = full_decode[0, 0]
    one_f0 = decode_one[0, 0]
    source_mse = _mse(source, gen_f0)
    coupling_mse = _mse(one_f0, gen_f0)
    encode_vs_first_latent = _mse(encoded0[:, :1], first_latent[:, :1])

    frames_dir = out / "frame0_diagnose"
    _save_rgb_frame(source.cpu(), frames_dir / "source_t0.png")
    _save_rgb_frame(gen_f0.cpu(), frames_dir / "rollout_decode_t0.png")
    _save_rgb_frame(one_f0.cpu(), frames_dir / "latent0_only_decode_t0.png")
    _save_rgb_frame(roundtrip[0, 0].cpu(), frames_dir / "vae_roundtrip_t0.png")

    per_frame_mse: list[float] = []
    n = min(int(prepared.pixels.shape[0]), int(full_decode.shape[1]))
    for t in range(n):
        ref = prepared.pixels[t].to(device=provider.device)
        per_frame_mse.append(_mse(ref, full_decode[0, t]))
    import statistics

    return {
        "latent_pin_vs_first_latent": pin,
        "encode_pixel0_vs_first_latent_mse": encode_vs_first_latent,
        "vae_roundtrip_pixel0_mse": roundtrip_mse,
        "rollout_decode_pixel0_vs_source_mse": source_mse,
        "temporal_coupling_pixel0_mse": coupling_mse,
        "per_frame_mse_vs_source": {
            "count": n,
            "frame0": per_frame_mse[0] if per_frame_mse else None,
            "min": min(per_frame_mse) if per_frame_mse else None,
            "max": max(per_frame_mse) if per_frame_mse else None,
            "mean": statistics.fmean(per_frame_mse) if per_frame_mse else None,
        },
        "frames_dir": str(frames_dir),
    }


def _vae_isolation_report(
    vae: Any,
    ref_dir: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Decode CUDA reference latents on the probe device (isolates VAE from rollout)."""

    latents_path = ref_dir / "latents_bf16.pt"
    if not latents_path.is_file():
        return {"ok": False, "reason": f"missing {latents_path}"}
    cuda_latents = torch.load(latents_path, map_location="cpu", weights_only=True)
    latents = cuda_latents.to(device=device, dtype=torch.bfloat16)
    report: dict[str, Any] = {
        "cuda_latents_path": str(latents_path),
        "latent_stats_on_device": _tensor_stats(latents),
    }
    production_ref = ref_dir / "decode_cuda_production.pt"
    cuda_pixels_path = (
        production_ref if production_ref.is_file() else _decode_pixels_path(ref_dir, "cuda")
    )
    if cuda_pixels_path is not None and cuda_pixels_path.is_file():
        cuda_pixels = torch.load(cuda_pixels_path, map_location="cpu", weights_only=True)
        report["cuda_reference_decode_file"] = cuda_pixels_path.name
    else:
        cuda_pixels = None
        report["cuda_reference_decode_file"] = None

    with torch.no_grad():
        production = vae.decode(latents, use_cache=False)
    report["production_wan5b_decode"] = _tensor_stats(production)
    if cuda_pixels is not None:
        report["production_vs_cuda_saved_decode"] = _compare_tensors(
            cuda_pixels,
            production.cpu(),
        )

    for name, kwargs in (
        ("bf16_autocast_direct", {"autocast": True, "autocast_dtype": torch.bfloat16}),
        ("weight_dtype_direct", {"autocast": False, "autocast_dtype": torch.float32}),
    ):
        try:
            pixels = _decode_direct(vae, latents, **kwargs)
            entry: dict[str, Any] = {"stats": _tensor_stats(pixels)}
            if cuda_pixels is not None:
                entry["vs_cuda_saved_decode"] = _compare_tensors(cuda_pixels, pixels.cpu())
            report[name] = entry
        except Exception as exc:
            report[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    report["ok"] = True
    return report


def _decode_pixels_path(directory: Path, device_type: str) -> Path | None:
    """Resolve a saved direct-decode tensor (CUDA reference naming varies)."""

    for name in (
        f"decode_{device_type}_fp32_direct.pt",
        f"decode_{device_type}_weight_dtype_direct.pt",
    ):
        path = directory / name
        if path.is_file():
            return path
    return None


def _compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()
    if ref.shape != cand.shape:
        return {"match": False, "reason": f"shape {list(cand.shape)} != {list(ref.shape)}"}
    finite = torch.isfinite(ref) & torch.isfinite(cand)
    if not bool(finite.all()):
        return {
            "match": False,
            "reason": "non-finite values in reference or candidate",
            "ref_finite": float(torch.isfinite(ref).float().mean()),
            "cand_finite": float(torch.isfinite(cand).float().mean()),
        }
    diff = (ref - cand).abs()
    return {
        "match": True,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cuda", "xpu", "cpu"),
        required=True,
        help="Device for rollout and VAE decode probes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for reference or probe artifacts",
    )
    parser.add_argument(
        "--compare-reference",
        type=Path,
        default=None,
        help="CUDA reference directory (metadata + latents + decode_cuda_fp32_direct.pt)",
    )
    parser.add_argument(
        "--weights",
        choices=("live", "ema"),
        default="ema",
        help="Checkpoint role for rollout (default: ema)",
    )
    parser.add_argument(
        "--frame0-diagnose",
        action="store_true",
        help="Latent pin + VAE round-trip + save frame-0 PNGs under output-dir",
    )
    parser.add_argument(
        "--decode-reference-latents",
        action="store_true",
        help="Skip diffusion rollout; decode latents_bf16.pt from --compare-reference only",
    )
    parser.add_argument(
        "--save-rollout-trace",
        action="store_true",
        help="After rollout, write rollout_trace/ with noise, conditions, per-step tensors, and KV snapshots",
    )
    parser.add_argument(
        "--replay-rollout-trace",
        type=Path,
        default=None,
        metavar="DIR",
        help="CUDA rollout_trace directory; inject saved tensors instead of device RNG",
    )
    parser.add_argument(
        "--replay-mode",
        choices=("noise", "conditions", "kv_chunks"),
        default="kv_chunks",
        help="noise: CUDA initial_noise+renoise only; conditions: +prompt/camera/first_latent; "
        "kv_chunks: also restore KV/cross-attn before each chunk after the first (default)",
    )
    parser.add_argument(
        "--rollout-latent-frames",
        type=int,
        default=None,
        metavar="N",
        help="Override rollout horizon (multiple of 3) for partial trace replay",
    )
    parser.add_argument(
        "--output-latent-frames",
        type=int,
        default=None,
        metavar="N",
        help="Override trimmed output latent count (default: same as --rollout-latent-frames)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Extra solarwm config overrides (e.g. inference.device=xpu)",
    )
    args = parser.parse_args()

    device = _pick_device(args.device)
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    overrides = list(args.overrides)
    overrides.append(f"inference.device={device.type}")
    config = _resolve_config(overrides)

    plan = resolve_generation_plan(config)
    provider = CudaWanStage2GenerationAdapter(config, plan)
    try:
        cases = provider.build_cases(plan)
        if not cases:
            raise SystemExit("error: no inference cases built from smoke index")
        gen_pass = _pass_for_weights(plan, args.weights)
        case = _pass_case(cases[0], gen_pass)
        if args.rollout_latent_frames is not None:
            gen_pass, case = _apply_rollout_horizon(
                gen_pass,
                case,
                rollout_latent_frames=int(args.rollout_latent_frames),
                output_latent_frames=args.output_latent_frames,
            )
        provider._load_role(args.weights)
        rollout_first: torch.Tensor | None = None
        rollout_trace_meta: dict[str, Any] = {}
        replay: _RolloutTraceReplay | None = None
        if args.replay_rollout_trace is not None:
            trace_root = args.replay_rollout_trace.expanduser().resolve()
            if trace_root.is_dir() and (trace_root / "manifest.json").is_file():
                replay_dir = trace_root
            elif (trace_root / "rollout_trace" / "manifest.json").is_file():
                replay_dir = trace_root / "rollout_trace"
            else:
                raise SystemExit(
                    f"error: --replay-rollout-trace must point at rollout_trace/ (manifest.json missing in {trace_root})",
                )
            replay = _RolloutTraceReplay(replay_dir, args.replay_mode)
        trace_dir: Path | None = None
        if args.save_rollout_trace and not args.decode_reference_latents:
            trace_dir = out / "rollout_trace"
        if args.decode_reference_latents:
            if args.compare_reference is None:
                raise SystemExit("error: --decode-reference-latents requires --compare-reference")
            ref_latents_path = args.compare_reference.resolve() / "latents_bf16.pt"
            if not ref_latents_path.is_file():
                raise SystemExit(f"error: missing CUDA latents at {ref_latents_path}")
            latents = torch.load(ref_latents_path, map_location="cpu", weights_only=True).to(
                device=device,
                dtype=torch.bfloat16,
            )
            schedule = {"schema": "skipped", "reason": "decode_reference_latents"}
        else:
            latents, schedule, rollout_first, rollout_trace_meta = _run_rollout(
                provider,
                case,
                gen_pass,
                trace_dir=trace_dir,
                replay=replay,
            )

        latent_stats = _tensor_stats(latents)
        if not args.decode_reference_latents:
            torch.save(latents.detach().cpu(), out / "latents_bf16.pt")
        schedule_payload = schedule if isinstance(schedule, Mapping) else {"value": str(schedule)}
        (out / "rollout_schedule.json").write_text(
            json.dumps(dict(schedule_payload), indent=2, default=str) + "\n"
        )

        decode_probes = _probe_decodes(provider.vae, latents)
        with torch.no_grad():
            production_pixels = provider.vae.decode(latents, use_cache=False)
        torch.save(
            production_pixels.detach().cpu(),
            out / f"decode_{device.type}_production.pt",
        )
        if device.type == "cuda":
            torch.save(production_pixels.detach().cpu(), out / "decode_cuda_production.pt")
        ref_save_error: str | None = None
        try:
            ref_pixels = _decode_direct(
                provider.vae,
                latents,
                autocast=False,
                autocast_dtype=torch.float32,
            )
            torch.save(ref_pixels.detach().cpu(), out / f"decode_{device.type}_weight_dtype_direct.pt")
            if device.type == "cuda":
                torch.save(ref_pixels.detach().cpu(), out / "decode_cuda_fp32_direct.pt")
        except Exception as exc:
            ref_save_error = f"{type(exc).__name__}: {exc}"

        frame0_diagnose: dict[str, Any] | None = None
        if args.frame0_diagnose:
            frame0_diagnose = _diagnose_frame0(
                provider, case, gen_pass, latents, rollout_first, out
            )

        metadata = {
            "schema": "solarwm.wan22-stage2-decode-reference.v1",
            "device": device.type,
            "sample_id": case.sample_id,
            "noise_seed": int(case.noise_seed),
            "weights_role": args.weights,
            "mode": "decode_reference_latents" if args.decode_reference_latents else "rollout",
            "rollout_trace_dir": str(trace_dir) if trace_dir is not None else None,
            "replay_rollout_trace": str(replay.trace_dir) if replay is not None else None,
            "replay_mode": replay.mode if replay is not None else None,
            "latent_stats": latent_stats,
            "decode_probes": decode_probes,
            "reference_decode_error": ref_save_error,
            "frame0_diagnose": frame0_diagnose,
            "config_path": str(DEFAULT_CONFIG),
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        if args.compare_reference is not None or replay is not None:
            ref_dir = (
                args.compare_reference.resolve()
                if args.compare_reference is not None
                else (replay.trace_dir.parent if replay is not None else out)
            )
            report: dict[str, Any] = {"reference_dir": str(ref_dir)}
            if args.compare_reference is not None:
                report["vae_isolation"] = _vae_isolation_report(
                    provider.vae,
                    ref_dir,
                    device=device,
                )
            cuda_latents_path = ref_dir / "latents_bf16.pt"
            if cuda_latents_path.is_file() and not args.decode_reference_latents:
                cuda_latents = torch.load(cuda_latents_path, map_location="cpu", weights_only=True)
                report["rollout_latents_vs_cuda"] = _compare_tensors(cuda_latents, latents.cpu())
            elif not args.decode_reference_latents:
                report["rollout_latents_vs_cuda"] = {
                    "match": False,
                    "reason": f"missing {cuda_latents_path}",
                }
            cuda_pixels_path = _decode_pixels_path(ref_dir, "cuda")
            probe_pixels_path = (
                None if args.decode_reference_latents else _decode_pixels_path(out, device.type)
            )
            if cuda_pixels_path is not None and probe_pixels_path is not None:
                cuda_pixels = torch.load(cuda_pixels_path, map_location="cpu", weights_only=True)
                probe_pixels = torch.load(probe_pixels_path, map_location="cpu", weights_only=True)
                report["rollout_decode_pixels_vs_cuda"] = {
                    "reference_file": str(cuda_pixels_path.name),
                    "candidate_file": str(probe_pixels_path.name),
                    "note": "Each side decoded its own rollout latents; use vae_isolation for same latents.",
                    **_compare_tensors(cuda_pixels, probe_pixels),
                }
            elif not args.decode_reference_latents:
                report["rollout_decode_pixels_vs_cuda"] = {
                    "match": False,
                    "reason": "missing decode tensors",
                    "reference_decode": str(cuda_pixels_path) if cuda_pixels_path else None,
                    "candidate_decode": str(probe_pixels_path) if probe_pixels_path else None,
                }
            if replay is not None:
                report["rollout_trace_replay"] = _rollout_trace_report(
                    latents,
                    rollout_trace_meta,
                    replay.trace_dir,
                )
            (out / "probe_report.json").write_text(json.dumps(report, indent=2) + "\n")

        print(json.dumps(metadata, indent=2))
        return 0 if latent_stats["finite_fraction"] == 1.0 else 1
    finally:
        provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
