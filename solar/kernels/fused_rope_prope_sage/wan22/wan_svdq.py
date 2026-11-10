"""SVDQuant W4A4 (SolarWM `dit_svdquant` SYCL kernel) for the Wan2.2-I2V-A14B transformer blocks.

Per block, the self-attention Q/K/V projections (fused into one `to_qkv`, N = 3 x 5120) and both FFN
projections run on `svdquant_dense_kernel_grf128_n_tile64.svdquant_w4a4_gemm`:

    y = Q4(x * smooth_inv) @ Q4(R)^T + (x @ lora_a^T) @ lora_b^T + bias,
    W * s = L_B L_A + R   (rank-r SVD of the smoothed weight, residual R in INT4)

with group-64 symmetric INT4 for both activations (dynamic, per token) and weights, smoothing
s = max|x|^alpha / max|W|^(1 - alpha) per input channel from calibration, and lora_a = L_A / s
(the kernel applies the low-rank branch to the unsmoothed x). The FFN up projection fuses GELU(tanh)
and emits INT4 already smoothed for the down projection. Everything else in the block (attention
output, cross-attention, norms, modulation) keeps the W8A16 storage of `wan_int8.py`.

    python wan/wan_svdq.py convert <int8_dir> <calib.pt> <out_dir> [--rank 32] [--alpha 0.5]
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wan_int8  # noqa: E402

SVDQ_BUILD = os.environ.get("SVDQ_BUILD_DIR", "/home/gta/omni_bench/kernel/build_svdq")
_KERNEL = None


def kernel():
    global _KERNEL
    if _KERNEL is None:
        sys.path.insert(0, SVDQ_BUILD)
        import svdquant_dense_kernel_grf128_n_tile64 as k
        _KERNEL = k
    return _KERNEL


# ---------------------------------------------------------------- packing (as solar/kernels/dit_svdquant)
def pack_int4(q):
    """int8 values in [-8, 7], [N, K] -> [N, K/2] int8, element 0 in the low nibble."""
    u = q.to(torch.int8) & 0x0F
    return (u[:, 0::2] | (u[:, 1::2] << 4)).to(torch.int8)


def to_dpas_b(packed):
    """[N, K/2] packed INT4 -> DPAS B register image [N/16, K/64, 8, 16] int32."""
    n, kh = packed.shape
    k = 2 * kh
    rows = packed.contiguous().view(torch.uint8).view(n, k // 8, 4).contiguous().view(torch.int32)
    return rows.view(n // 16, 16, k // 8).permute(0, 2, 1).contiguous().view(n // 16, k // 64, 8, 16)


def from_dpas_b(w_dpas_b, w_scales_dpas):
    """Inverse of the prepack: dequantised residual [N, K] float32 (for the reference path)."""
    n16, kg = w_dpas_b.shape[:2]
    n, k = n16 * 16, kg * 64
    packed = w_dpas_b.reshape(n16, k // 8, 16).permute(0, 2, 1).reshape(n, k // 8).contiguous().view(torch.uint8)
    packed = packed.view(n, k // 2)
    lo = (packed & 0x0F).to(torch.int8); lo = torch.where(lo >= 8, lo - 16, lo)
    hi = ((packed >> 4) & 0x0F).to(torch.int8); hi = torch.where(hi >= 8, hi - 16, hi)
    w = torch.stack([lo, hi], -1).flatten(-2).float()
    return w * w_scales_dpas.t().repeat_interleave(64, dim=-1)


# ---------------------------------------------------------------- conversion of one linear
@torch.no_grad()
def quantise_linear(w, bias, x_amax, rank=32, alpha=0.5, niter=6):
    """w [N, K] float32 (device), x_amax [K] -> tensors for SVDQLinear."""
    n, k = w.shape
    w_amax = w.abs().amax(0).clamp_min(1e-5)
    s = (x_amax.float().clamp_min(1e-5) ** alpha / w_amax ** (1 - alpha)).clamp_min(1e-5)
    ws = w * s[None, :]
    if rank > 0:
        u, sv, v = torch.svd_lowrank(ws, q=rank + 16, niter=niter)
        lb = u[:, :rank] * sv[:rank]            # [N, r]
        la = v[:, :rank].t()                     # [r, K]
        r = ws - lb @ la
    else:
        r = ws
    g = r.view(n, k // 64, 64)
    scale = (g.abs().amax(-1) / 7.0).clamp_min(1e-8).to(torch.bfloat16)   # stored precision
    q = torch.round(g / scale.float()[..., None]).clamp_(-8, 7).view(n, k)
    out = {
        "w_dpas_b": to_dpas_b(pack_int4(q)),
        "w_scales_dpas": scale.float().t().contiguous(),                    # [K/64, N]
        "smooth_inv": (1.0 / s).float().contiguous(),
        "bias": (bias if bias is not None else torch.zeros(n, device=w.device)).to(torch.bfloat16).contiguous(),
    }
    if rank > 0:
        out["lora_a"] = (la / s[None, :]).to(torch.bfloat16).contiguous()
        out["lora_b"] = lb.to(torch.bfloat16).contiguous()
    return out


# ---------------------------------------------------------------- modules
class SVDQLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        meta = dict(device="meta")
        self.register_buffer("w_dpas_b", torch.empty(out_features // 16, in_features // 64, 8, 16, dtype=torch.int32, **meta))
        self.register_buffer("w_scales_dpas", torch.empty(in_features // 64, out_features, dtype=torch.float32, **meta))
        self.register_buffer("smooth_inv", torch.empty(in_features, dtype=torch.float32, **meta))
        self.register_buffer("bias", torch.empty(out_features, dtype=torch.bfloat16, **meta))
        if rank > 0:
            self.register_buffer("lora_a", torch.empty(rank, in_features, dtype=torch.bfloat16, **meta))
            self.register_buffer("lora_b", torch.empty(out_features, rank, dtype=torch.bfloat16, **meta))
        else:
            self.lora_a = self.lora_b = None

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        y = kernel().svdquant_w4a4_gemm(x2, self.w_dpas_b, self.w_scales_dpas, lora_a=self.lora_a, lora_b=self.lora_b,
                                        bias=self.bias, x_smooth_inv=self.smooth_inv, x_rho=1.0)
        return y.view(*shape[:-1], self.out_features)

    def reference(self, x):
        """fp32 emulation with the kernel's quantisation semantics."""
        x2 = x.reshape(-1, x.shape[-1]).float()
        g = (x2 * self.smooth_inv).view(x2.shape[0], -1, 64)
        sc = (g.abs().amax(-1, keepdim=True) / 7.0).clamp_min(1e-8)
        q = g / sc
        q = torch.where(q >= 0, torch.floor(q + 0.5), torch.ceil(q - 0.5)).clamp(-8, 7)
        y = (q * sc).view_as(x2) @ from_dpas_b(self.w_dpas_b, self.w_scales_dpas).t()
        if self.lora_a is not None:
            y = y + (x2 @ self.lora_a.float().t()) @ self.lora_b.float().t()
        return (y + self.bias.float()).view(*x.shape[:-1], self.out_features)


class SVDQFFN(nn.Module):
    """Wan FeedForward (Linear -> GELU tanh -> Linear) on two SVDQLinear; INT4 hand-off between them."""

    def __init__(self, dim, hidden, rank):
        super().__init__()
        self.up, self.down = SVDQLinear(dim, hidden, rank), SVDQLinear(hidden, dim, rank)

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        u, d, k = self.up, self.down, kernel()
        h, h_q, h_s = k.svdquant_w4a4_gemm(x2, u.w_dpas_b, u.w_scales_dpas, lora_a=u.lora_a, lora_b=u.lora_b, bias=u.bias,
                                           activation="gelu_tanh", fuse_quant_out=True, x_smooth_inv=u.smooth_inv,
                                           x_rho=1.0, out_smooth_inv=d.smooth_inv, out_rho=1.0)
        y = k.svdquant_w4a4_gemm(h_q, d.w_dpas_b, d.w_scales_dpas, x_scales=h_s, x_bf16=h, lora_a=d.lora_a,
                                 lora_b=d.lora_b, bias=d.bias)
        return y.view(*shape[:-1], d.out_features)


# ---------------------------------------------------------------- model surgery, load, convert
def _patch_block(block, rank):
    attn = block.attn1
    dim = attn.to_q.in_features
    for name in ("to_q", "to_k", "to_v"):
        delattr(attn, name)
    attn.to_qkv = SVDQLinear(dim, 3 * dim, rank)
    attn.fused_projections = True
    hidden = block.ffn.net[0].proj.out_features
    block.ffn = SVDQFFN(dim, hidden, rank)


def build_expert(svdq_dir):
    """INT8 expert skeleton (meta, bf16) with Q/K/V and FFN of every block replaced by SVDQuant modules."""
    meta = json.load(open(os.path.join(svdq_dir, "svdq_index.json")))
    model = wan_int8.build_expert(svdq_dir)
    for block in model.blocks:
        _patch_block(block, meta["rank"])
    return model


load_expert_weights = wan_int8.load_expert_weights
unload_expert = wan_int8.unload_expert


@torch.no_grad()
def convert_expert(int8_dir, calib, out_dir, rank, alpha, prefix):
    """int8 expert + calibration -> SVDQuant expert directory (same shard / index layout as wan_int8)."""
    from safetensors.torch import load_file, save_file
    os.makedirs(out_dir, exist_ok=True)
    index = json.load(open(os.path.join(int8_dir, "int8_index.json")))
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        state.update(load_file(os.path.join(int8_dir, shard)))
    dev = torch.device("xpu")

    def deq(name):
        return state.pop(f"{name}.weight").to(dev).float() * state.pop(f"{name}.weight_scale").to(dev).float()[:, None]

    def emit(name, t):
        for key, val in t.items():
            state[f"{name}.{key}"] = val.cpu()

    n_blocks = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks."))
    for i in range(n_blocks):
        b = f"blocks.{i}"
        w = torch.cat([deq(f"{b}.attn1.to_{p}") for p in "qkv"], 0)
        bias = torch.cat([state.pop(f"{b}.attn1.to_{p}.bias").to(dev).float() for p in "qkv"], 0)
        emit(f"{b}.attn1.to_qkv", quantise_linear(w, bias, calib[f"{prefix}.{i}.attn1"].to(dev), rank, alpha))
        w = deq(f"{b}.ffn.net.0.proj"); bias = state.pop(f"{b}.ffn.net.0.proj.bias").to(dev).float()
        emit(f"{b}.ffn.up", quantise_linear(w, bias, calib[f"{prefix}.{i}.ffn_up"].to(dev), rank, alpha))
        w = deq(f"{b}.ffn.net.2"); bias = state.pop(f"{b}.ffn.net.2.bias").to(dev).float()
        emit(f"{b}.ffn.down", quantise_linear(w, bias, calib[f"{prefix}.{i}.ffn_down"].to(dev), rank, alpha))
        del w, bias
        torch.xpu.empty_cache()
    shards, buf, size, out_map = 0, {}, 0, {}
    for key in sorted(state):
        buf[key] = state[key].contiguous(); size += buf[key].numel() * buf[key].element_size()
        if size >= wan_int8.SHARD_BYTES:
            name = f"svdq-{shards:05d}.safetensors"; save_file(buf, os.path.join(out_dir, name))
            out_map.update({k: name for k in buf}); shards, buf, size = shards + 1, {}, 0
    if buf:
        name = f"svdq-{shards:05d}.safetensors"; save_file(buf, os.path.join(out_dir, name))
        out_map.update({k: name for k in buf}); shards += 1
    json.dump({"weight_map": out_map, "format": "wan_svdq_w4a4", "rank": rank, "alpha": alpha},
              open(os.path.join(out_dir, "svdq_index.json"), "w"), indent=1)
    import shutil
    shutil.copy(os.path.join(int8_dir, "config.json"), os.path.join(out_dir, "config.json"))
    print(f"[svdq] {int8_dir} -> {out_dir}: {n_blocks} blocks, rank {rank}, alpha {alpha}, {shards} shards", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert"])
    ap.add_argument("int8_dir"); ap.add_argument("calib"); ap.add_argument("out_dir")
    ap.add_argument("--rank", type=int, default=32); ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args()
    calib = torch.load(a.calib)
    for prefix, sub in (("transformer", "transformer"), ("transformer_2", "transformer_2")):
        convert_expert(os.path.join(a.int8_dir, sub), calib, os.path.join(a.out_dir, sub), a.rank, a.alpha, prefix)
