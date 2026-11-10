"""INT8 weight-only (W8A16) storage for Wan2.2 transformer experts, and lazy per-expert loading.

The fp32 experts (53 GiB each) do not fit a 32 GB GPU even in bf16 (28 GiB); in INT8 they are about
14 GiB. Every 2-D weight inside the transformer blocks (attention q/k/v/out, cross-attention, FFN) is
stored as int8 with one bf16 scale per output channel (symmetric absmax / 127, round half to even) and
dequantised to bf16 right before its matmul. Everything else (embeddings, norms, biases, modulation,
patch / head projections) stays bf16. The same weights serve both attention arms, so a default-vs-Sage
comparison isolates the attention kernel.

    python wan/wan_int8.py convert <diffusers_dir> <out_dir>      # transformer + transformer_2
"""
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SHARD_BYTES = 4 * 2**30


def _quantise_row(w):
    """w [out, in] fp32 -> (int8 [out, in], bf16 scale [out])."""
    amax = w.abs().amax(dim=1).clamp_min(1e-12)
    scale = amax / 127.0
    q = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale.to(torch.bfloat16)


def _is_block_linear_weight(key, t):
    return ".blocks." in f".{key}" and key.endswith(".weight") and t.ndim == 2


def convert_expert(src_dir, out_dir):
    """Stream fp32 safetensors shards -> int8 (block linears) / bf16 (rest) shards, one tensor at a time."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    index = json.load(open(os.path.join(src_dir, "diffusion_pytorch_model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))
    out_map, buf, buf_bytes, n_out, n_q, n_keep = {}, {}, 0, 0, 0, 0

    def flush():
        nonlocal buf, buf_bytes, n_out
        if not buf:
            return
        name = f"int8-{n_out:05d}.safetensors"
        save_file(buf, os.path.join(out_dir, name))
        for k in buf:
            out_map[k] = name
        n_out += 1
        buf, buf_bytes = {}, 0

    for shard in shards:
        with safe_open(os.path.join(src_dir, shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if _is_block_linear_weight(key, t):
                    q, s = _quantise_row(t.float())
                    buf[key] = q.contiguous()
                    buf[key[:-len(".weight")] + ".weight_scale"] = s.contiguous()
                    buf_bytes += q.numel() + 2 * s.numel()
                    n_q += 1
                else:
                    buf[key] = t.to(torch.bfloat16).contiguous()
                    buf_bytes += 2 * t.numel()
                    n_keep += 1
                del t
                if buf_bytes >= SHARD_BYTES:
                    flush()
    flush()
    json.dump({"weight_map": out_map, "format": "wan_w8a16_rowwise"},
              open(os.path.join(out_dir, "int8_index.json"), "w"), indent=1)
    import shutil
    shutil.copy(os.path.join(src_dir, "config.json"), os.path.join(out_dir, "config.json"))
    print(f"[convert] {src_dir} -> {out_dir}: {n_q} int8 linear weights, {n_keep} bf16 tensors, {n_out} shards", flush=True)


class Int8Linear(nn.Module):
    """W8A16 linear: int8 weight [out, in] + bf16 per-output-channel scale; dequantised per call."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8, device="meta"))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.bfloat16, device="meta"))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device="meta")) if bias else None

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype)[:, None]
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))


def _replace_block_linears(module, prefix=""):
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and ".blocks." in f".{full}.":
            setattr(module, name, Int8Linear(child.in_features, child.out_features, child.bias is not None))
        else:
            _replace_block_linears(child, full)


def build_expert(int8_dir):
    """WanTransformer3DModel on the meta device with block linears replaced by Int8Linear."""
    from diffusers import WanTransformer3DModel
    cfg = WanTransformer3DModel.load_config(int8_dir)
    with torch.device("meta"):
        model = WanTransformer3DModel.from_config(cfg).to(torch.bfloat16)
    # bf16 on meta so model.dtype (which the pipeline reads before the lazy load) matches the weights
    _replace_block_linears(model)
    return model.eval().requires_grad_(False)


def load_expert_weights(model, int8_dir, device):
    """Materialise every parameter / buffer of `model` from the shards (int8_index.json or svdq_index.json)
    directly on `device`."""
    from safetensors.torch import load_file
    name = "int8_index.json" if os.path.isfile(os.path.join(int8_dir, "int8_index.json")) else "svdq_index.json"
    index = json.load(open(os.path.join(int8_dir, name)))
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        state.update(load_file(os.path.join(int8_dir, shard), device=str(device)))
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    missing = [k for k in missing if not k.endswith(("freqs_cos", "freqs_sin"))]
    if missing or unexpected:
        raise RuntimeError(f"int8 load mismatch: missing={missing[:6]} unexpected={unexpected[:6]}")
    # RoPE tables are non-persistent buffers computed in the constructor, so they are still on meta:
    # rebuild the module on the CPU (float64 tables, as diffusers does) and move it to the device.
    rope = model.rope
    model.rope = type(rope)(rope.attention_head_dim, rope.patch_size, rope.max_seq_len).to(device)
    for name, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            raise RuntimeError(f"buffer {name} still on meta")
    return model


def unload_expert(model):
    """Free the expert's device memory (parameters and buffers back to meta)."""
    model.to("meta")
    torch.xpu.empty_cache()


if __name__ == "__main__":
    if sys.argv[1] == "convert":
        src, out = sys.argv[2], sys.argv[3]
        for sub in ("transformer", "transformer_2"):
            convert_expert(os.path.join(src, sub), os.path.join(out, sub))
