"""Profile one steady-state chunk-4 DiT forward pass (denoise step 1) and one VAE chunk decode."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile

from scripts.quantization.run_svdquant_inference import (
    _NoTextEncoder,
    iter_latent_samples,
    sample_inputs,
)
from scripts.quantization.svdquant_int4.attn_dispatch import with_attention_backend
from solarwm.backends.wan22.runtime import stage2
from solarwm.backends.wan22.runtime.modeling.vae_quarot import apply_vae_quarot
from solarwm.config import load_config

stage2.WanTextEncoder = _NoTextEncoder

CONFIG = "configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml"
SHARD = Path(
    "/home/ssheorey/data/SolarWM-Data/releases-v1/latent-wds/wan22-ti2v5b-153f-480p-v1/abot/shards/kept-high-000008.tar"
)
OUT = REPO_ROOT / "outputs/svdquant_eval/eval_153f/kernel_profiles"


def table(prof, n=30):
    return prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=n, max_name_column_width=90)


def run(linear_quant: str, attn: str, profile_vae: bool | str) -> None:
    # Normalize inputs
    if linear_quant in ("bf16", "fp16", "full_precision"):
        linear_quant = "full_precision"
    elif linear_quant in ("svdquant", "svdquant_k4", "k4"):
        linear_quant = "svdquant_k4"

    if attn in ("sageattn", "sageattnv1"):
        attn = "sageattnv1"
    elif attn in ("sdpa",):
        attn = "sdpa"

    tag = f"{linear_quant}_{attn}"
    values = load_config(
        CONFIG,
        [
            "model.base_path=/home/ssheorey/models/SolarWM/SolarWM-5B-base",
            "checkpoint.path=/home/ssheorey/models/SolarWM/SolarWM-5B-sgf-stage2-81f",
            "data.index_root=/home/ssheorey/data/SolarWM-Data/releases-v1",
            "data.transport.root=/home/ssheorey/data/SolarWM-Data/releases-v1",
            "inference.device=xpu",
            f"model.linear_quant={linear_quant}",
            f"model.attention_backend={attn}",
        ],
    ).values
    provider = stage2.build_stage2_generation_provider(values)
    provider._load_role(str(provider.plan.passes[0].weights))
    device = provider.device

    sample = next(iter_latent_samples([SHARD], per_shard=1))
    latent_frames = 15
    gp = replace(provider.plan.passes[0], rollout_latent_frames=latent_frames, min_rollout_latent_frames=latent_frames)
    first, cond, cam, _ = sample_inputs(provider, sample, latent_frames)
    rng = torch.Generator(device=device).manual_seed(1234)

    state = {"chunk": 0, "prof": None}
    inner = provider.diffusion

    class Wrap:
        def __getattr__(self, name):
            return getattr(inner, name)

        def __call__(self, *a, **kw):
            if state["chunk"] == 4 and kw.get("radial_step") == 1:
                torch.xpu.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as p:
                    out = inner(*a, **kw)
                    torch.xpu.synchronize()
                state["prof"] = p
                return out
            return inner(*a, **kw)

    provider.diffusion = Wrap()
    orig_slice = stage2._slice_camera

    def hooked(*a, **kw):
        state["chunk"] = kw.get("start_frame", 0) // 3
        return orig_slice(*a, **kw)

    stage2._slice_camera = hooked
    try:
        with with_attention_backend(provider.diffusion, attn), torch.no_grad():
            latents, _ = stage2._stage2_self_forcing_latents(provider, gp, first, cond, cam, rng)
    finally:
        stage2._slice_camera = orig_slice

    OUT.mkdir(parents=True, exist_ok=True)
    table_str = table(state["prof"])
    (OUT / f"dit_{tag}.txt").write_text(table_str)
    state["prof"].export_chrome_trace(str(OUT / f"dit_{tag}.json"))
    if tag == "full_precision_sdpa":
        (OUT / "dit_bf16_sdpa.txt").write_text(table_str)
        (OUT / "dit_bf16_sdpa.json").write_text((OUT / f"dit_{tag}.json").read_text())
    print(f"\n===== DiT chunk4 step1: {tag} =====")
    print(table(state["prof"], 25))

    if profile_vae:
        chunk = latents[:, 12:15].contiguous()
        vae_modes: list[tuple[str, bool]] = []
        val_str = str(profile_vae).lower()
        if val_str in ("true", "1", "vae", "all", "both"):
            vae_modes = [("vae_bf16", False), ("vae_quarot", True)]
        elif val_str in ("vae_bf16", "bf16"):
            vae_modes = [("vae_bf16", False)]
        elif val_str in ("vae_quarot", "quarot", "w8a8"):
            vae_modes = [("vae_quarot", True)]
        else:
            vae_modes = [("vae_bf16", False), ("vae_quarot", True)]

        for name, enabled in vae_modes:
            apply_vae_quarot(provider.vae.module.decoder, enabled=enabled)
            with torch.no_grad():
                provider.vae.decode_streaming(chunk, chunk_latent_frames=4)
                torch.xpu.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as p:
                    provider.vae.decode_streaming(chunk, chunk_latent_frames=4)
                    torch.xpu.synchronize()
            (OUT / f"{name}.txt").write_text(table(p))
            p.export_chrome_trace(str(OUT / f"{name}.json"))
            print(f"\n===== {name} =====")
            print(table(p, 20))
        apply_vae_quarot(provider.vae.module.decoder, enabled=False)


if __name__ == "__main__":
    vae_flag = sys.argv[3] if len(sys.argv) > 3 else False
    run(sys.argv[1], sys.argv[2], vae_flag)
