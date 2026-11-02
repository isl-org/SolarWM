"""Kernel-level relative RMS error of each quantized DiT kernel against its BF16 path.

Runs the BF16 benchmark rollout to the first denoise forward of chunk 5 (the first with the full
7,290-token KV window) and captures, for a few layers, the self-attention Q/K/V that SDPA receives
and the inputs of the five SVDQuant linears. Then, on those tensors:

* Sage v1 (INT8 Q/K, with the Hadamard-4 rotation production folds into PRoPE) and Sage v2 (INT4 Q/K)
  vs BF16 SDPA; BF16 SDPA vs FP32 SDPA gives the BF16 floor.
* SVDQuant (real `sdpa_k4` weights, bin 0) vs the BF16 linears; Q/K/V per linear and the FFN.

Each is also run on random Gaussian inputs of the same shapes (SVDQuant: random activations, the
layer's real weights), which is what `bench_isolated_blocks.py attention` uses.

Relative RMS = ||y - y_ref|| / ||y_ref||.

    python scripts/report/kernel_errors.py --output-json outputs/report_chain/isolated/kernel_errors.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

import bench_chunk
from bench_isolated_blocks import SVDQUANT_DIR
from solarwm.backends.wan22.runtime import stage2
from solarwm.backends.wan22.runtime.modeling.svdquant_linear import SVDQuantFFN, SYCLSVDQuantLinear
from solarwm.config import load_config
from solarwm.kernels import dit_fused
from solarwm.kernels.fused_rope_prope_sage import hadamard4, k_window_mean, sageattn_v2
from solarwm.kernels.fused_rope_prope_sage.fused_sage import sage_attention

LAYERS = (0, 15, 29)
CAPTURE_CHUNK = 5
CHUNKS = CAPTURE_CHUNK + 1


def rel_rms(y, ref):
    y, ref = y.float(), ref.float()
    return ((y - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def capture(overrides):
    stage2.WanTextEncoder = bench_chunk._NoTextEncoder
    values = load_config("configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml", bench_chunk.ASSET_OVERRIDES + overrides).values
    provider = stage2.build_stage2_generation_provider(values)
    provider._load_role(str(provider.plan.passes[0].weights))
    fields = {f.name for f in dataclasses.fields(provider.plan.passes[0])}
    generation_pass = dataclasses.replace(provider.plan.passes[0], **{k: 3 * CHUNKS for k in ("rollout_latent_frames", "min_rollout_latent_frames") if k in fields})
    first_latent, condition, camera = bench_chunk._inputs(provider, bench_chunk._first_sample(Path(bench_chunk.DEFAULT_SHARD)), 3 * CHUNKS)

    blocks = next(m for name, m in provider.diffusion.module.named_modules() if name.endswith("blocks") and isinstance(m, torch.nn.ModuleList))
    state = {"forward": 0, "attn_layer": 0}
    captured = {layer: {} for layer in LAYERS}
    target = CAPTURE_CHUNK * bench_chunk.FORWARDS_PER_CHUNK

    def armed():
        return state["forward"] == target

    sdpa = F.scaled_dot_product_attention

    def sdpa_capture(q, k, v, *args, **kwargs):
        if armed() and k.shape[-2] > q.shape[-2]:
            layer = state["attn_layer"]
            if layer in captured:
                captured[layer]["attn"] = (q.clone(), k.clone(), v.clone())
            state["attn_layer"] += 1
        return sdpa(q, k, v, *args, **kwargs)

    F.scaled_dot_product_attention = sdpa_capture
    up_weights = {blocks[layer].ffn[0].weight.data_ptr(): layer for layer in LAYERS}
    gelu = dit_fused.onednn_linear_gelu_tanh

    def gelu_capture(x, weight, bias):
        if armed() and weight.data_ptr() in up_weights:
            captured[up_weights[weight.data_ptr()]]["ffn.0.input"] = x.clone()
        return gelu(x, weight, bias)

    dit_fused.onednn_linear_gelu_tanh = gelu_capture
    for layer in LAYERS:
        block = blocks[layer]
        for suffix in SYCLSVDQuantLinear.TARGET_SUFFIXES:
            module = block.get_submodule(suffix)
            captured[layer][suffix + ".weight"] = module.weight.detach().clone()
            captured[layer][suffix + ".bias"] = module.bias.detach().clone()

            def hook(_, args, layer=layer, suffix=suffix):
                if armed():
                    captured[layer][suffix + ".input"] = args[0].clone()

            module.register_forward_pre_hook(hook)

    inner = provider.diffusion

    class Counting:
        def __getattr__(self, name):
            return getattr(inner, name)

        def __call__(self, *args, **kwargs):
            out = inner(*args, **kwargs)
            state["forward"] += 1
            return out

    provider.diffusion = Counting()
    rng = torch.Generator(device=provider.device).manual_seed(1234)
    with torch.no_grad():
        stage2._stage2_self_forcing_latents(provider, generation_pass, first_latent, condition, camera, rng)
    F.scaled_dot_product_attention = sdpa
    dit_fused.onednn_linear_gelu_tanh = gelu
    del provider
    torch.xpu.empty_cache()
    return captured


def attention_errors(q, k, v):
    ref = F.scaled_dot_product_attention(q, k, v)
    fp32 = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    h = hadamard4(q.device, q.dtype)
    rot = lambda t: (t.unflatten(-1, (-1, 4)) @ h).flatten(-2).contiguous()
    n_hist = k.shape[-2] - q.shape[-2]
    k_rot = rot(k)
    v1 = sage_attention(rot(q), k_rot, v, km=k_window_mean(k_rot[:, :, :n_hist]))
    nhd = lambda t: t.transpose(1, 2).contiguous()
    q_n, k_n = nhd(q), nhd(k)
    km = sageattn_v2.k_centre(k_n[:, :n_hist])
    k4, k4_scale = sageattn_v2.pack_k(k_n, km)
    q4, q_scale, q_mean = sageattn_v2.quantize_q(q_n)
    delta_s = sageattn_v2.compute_delta_s(k_n, km, q_mean)
    v2 = sageattn_v2.attention(q4, q_scale, q_mean, k4, k4_scale, delta_s, nhd(v), q_n.size(1)).transpose(1, 2)
    return {"bf16_sdpa_vs_fp32": rel_rms(ref, fp32), "sage_v1": rel_rms(v1, ref), "sage_v2": rel_rms(v2, ref)}


def svdquant_linears(layer, device):
    manifest = json.loads((SVDQUANT_DIR / "manifest.json").read_text())
    sets, prefix = {}, f"blocks.{layer}."
    for shard in manifest["shards"]:
        with safe_open(str(SVDQUANT_DIR / shard), "pt") as f:
            for key in f.keys():
                set_name, rest = key.split(".", 1)
                if rest.startswith(prefix):
                    module, param = rest.removeprefix(prefix).rsplit(".", 1)
                    sets.setdefault(module, {}).setdefault(set_name, {})[param] = f.get_tensor(key)
    policy = manifest["activation_policy"]
    return {suffix: SYCLSVDQuantLinear(sets[suffix], manifest["set_mapping"], {b: float(p[f"blocks.{layer}"][suffix]) for b, p in policy.items()}, device=device)
            for suffix in SYCLSVDQuantLinear.TARGET_SUFFIXES}


def svdquant_errors(layer, c, device):
    lin = svdquant_linears(layer, device)
    out = {}
    for suffix in ("self_attn.q", "self_attn.k", "self_attn.v"):
        x = c[suffix + ".input"]
        out[suffix] = rel_rms(lin[suffix](x), F.linear(x, c[suffix + ".weight"], c[suffix + ".bias"]))
    x = c["ffn.0.input"]
    ref = F.linear(dit_fused.onednn_linear_gelu_tanh(x, c["ffn.0.weight"], c["ffn.0.bias"]), c["ffn.2.weight"], c["ffn.2.bias"])
    out["ffn"] = rel_rms(SVDQuantFFN(lin["ffn.0"], lin["ffn.2"])(x), ref)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    device = torch.device("xpu")
    captured = capture([])
    result = {"real": {"attention": {}, "svdquant": {}}, "random": {"attention": {}, "svdquant": {}}}
    torch.manual_seed(0)
    with torch.no_grad():
        for layer in LAYERS:
            c = captured[layer]
            rand = {k: torch.randn_like(t) if k.endswith(".input") else t for k, t in c.items() if k != "attn"}
            rand_attn = tuple(torch.randn_like(t) for t in c["attn"])
            for kind, attn, inputs in (("real", c["attn"], c), ("random", rand_attn, rand)):
                result[kind]["attention"][layer] = attention_errors(*attn)
                result[kind]["svdquant"][layer] = svdquant_errors(layer, inputs, device)
                print(f"layer {layer} {kind}: attention {result[kind]['attention'][layer]}  svdquant {result[kind]['svdquant'][layer]}", flush=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
