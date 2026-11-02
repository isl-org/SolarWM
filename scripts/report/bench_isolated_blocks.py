"""Isolated block timings for the report, using the production call sequence at HEAD.

    python scripts/report/bench_isolated_blocks.py attention
    python scripts/report/bench_isolated_blocks.py quarot

``attention`` replays one DiT self-attention layer of a steady-state 480p chunk the
way ``CausalWanSelfAttention`` runs it with the RoPEd KV ring: RoPE/PRoPE on the
1,215 new tokens only, the 6,075-token history taken from the ring, and for Sage
v1/v2 the history K taken from the per-chunk quantized cache (INT8 for v1, INT4
DPAS-B images for v2). History work is done once per chunk and reported
separately, amortised over the chunk's 5 forwards.

``quarot`` decodes 3 latent chunks with the production VAE (bf16, channels-last,
Triton norms) and, on the last one, times every QuaRot candidate convolution on
its real input: the BF16 oneDNN forward vs the QuaRot W8A8 forward.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from solarwm.kernels.fused_rope_prope_sage import (
    ROPE_MODE,
    _attend_int8,
    _freqs_real,
    _prope_tables_triton,
    build_rope_table,
    hadamard4,
    k_window_mean,
    normalize_rope_precision,
    quantise_k,
    rope_prope_transform,
    sageattn_v2,
)
from solarwm.kernels.fused_rope_prope_sage.unfused_kernel import build_wan22_freqs

HEADS, HEAD_DIM, H_PATCH, W_PATCH = 24, 128, 15, 27
NEW_FRAMES, HIST_FRAMES = 3, 15
FORWARDS_PER_CHUNK = 5
DIT_LAYERS = 30
VAE_WEIGHTS = "/home/ssheorey/models/SolarWM/SolarWM-5B-base/vae/Wan2.2_VAE.pth"
SVDQUANT_DIR = Path("/home/ssheorey/models/SolarWM/SolarWM-5B-sgf-stage2-81f/svdquant_int4/sdpa_k4")


def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _cameras(n, device, x_shift):
    viewmats = torch.eye(4, device=device).repeat(1, n, 1, 1)
    viewmats[..., 0, 3] = torch.linspace(*x_shift, n, device=device)
    Ks = torch.eye(3, device=device).repeat(1, n, 1, 1)
    Ks[..., 0, 0] = 500.0 / 864.0
    Ks[..., 1, 1] = 500.0 / 480.0
    return viewmats, Ks


class AttentionLayer:
    """State of one layer's KV ring for one chunk, plus the per-forward call."""

    def __init__(self, backend, device):
        self.backend = backend
        self.is_sage = backend == "sage"
        self.qo_layout = "NHD" if backend == "sagev2" else "HND"
        tokens = H_PATCH * W_PATCH
        self.n_new, self.n_hist = NEW_FRAMES * tokens, HIST_FRAMES * tokens
        self.n_win = self.n_new + self.n_hist
        bf = dict(dtype=torch.bfloat16, device=device)
        torch.manual_seed(0)
        self.q = torch.randn(1, self.n_new, HEADS, HEAD_DIM, **bf)
        self.k = torch.randn(1, self.n_new, HEADS, HEAD_DIM, **bf)
        self.v = torch.randn(1, self.n_new, HEADS, HEAD_DIM, **bf)
        self.ring_k = torch.randn(1, self.n_win, HEADS, HEAD_DIM, **bf)
        self.ring_v = torch.randn(1, self.n_win, HEADS, HEAD_DIM, **bf)
        self.ring_k_enc = torch.zeros_like(self.ring_k)
        self.ring_v_enc = torch.zeros_like(self.ring_v)
        if self.is_sage:
            self.k8_buf = torch.empty(1, HEADS, self.n_win, HEAD_DIM, dtype=torch.int8, device=device)
            self.ks_buf = torch.empty(1, HEADS, -(-self.n_win // 32), dtype=torch.float32, device=device)
        if self.backend == "sagev2":
            win_pad = -(-self.n_win // 64) * 64
            self.k4_buf = torch.empty(1, HEADS, win_pad // 16, HEAD_DIM // 64, 8, 16, dtype=torch.int32, device=device)
            self.k4s_buf = torch.empty(1, HEADS, win_pad, dtype=torch.float32, device=device)

        # tables are memoised per forward in production, so they are built once here
        fr = _freqs_real(build_wan22_freqs(HEAD_DIM, 1024).to(device), "float64")
        self.mode = ROPE_MODE[normalize_rope_precision("float64")[0]]
        grid = lambda f: torch.tensor([[f, H_PATCH, W_PATCH]], dtype=torch.long, device=device)
        self.cs_hist = build_rope_table(fr, grid(HIST_FRAMES), self.n_hist, start_frame=0)
        self.cs_new = build_rope_table(fr, grid(NEW_FRAMES), self.n_new, start_frame=HIST_FRAMES)
        rot = hadamard4(device) if self.is_sage else None
        hv, hk = _cameras(HIST_FRAMES, device, (0.0, 0.25))
        nv, nk = _cameras(NEW_FRAMES, device, (0.25, 0.3))
        _, self.p_hist_k_inv, _ = _prope_tables_triton(hv, hk, rotation=rot)
        _, self.p_hist_v_inv, _ = _prope_tables_triton(hv, hk, rotation=None)
        self.p_q_t, self.p_new_k_inv, self.p_q = _prope_tables_triton(nv, nk, rotation=rot)
        _, self.p_new_v_inv, _ = _prope_tables_triton(nv, nk, rotation=None)

    def history(self):
        """Once per chunk: RoPE the history into the ring and quantize its K."""
        h = self.n_hist
        k_hist_enc = rope_prope_transform(self.ring_k[:, :h], self.p_hist_k_inv, self.cs_hist, self.mode, out_layout="NHD")
        v_hist_enc = rope_prope_transform(self.ring_v[:, :h], self.p_hist_v_inv, None, out_layout="NHD")
        self.ring_k_enc[:, :h].copy_(k_hist_enc)
        self.ring_v_enc[:, :h].copy_(v_hist_enc)
        if self.backend == "sagev2":
            self.km = sageattn_v2.k_centre(k_hist_enc)
            aligned = (h // 64) * 64
            sageattn_v2.pack_k_into(k_hist_enc[:, :aligned], self.km, self.k4_buf, self.k4s_buf, 0)
        elif self.is_sage:
            self.km = k_window_mean(k_hist_enc.transpose(1, 2))
            aligned = (h // 32) * 32
            quantise_k(k_hist_enc.transpose(1, 2)[:, :, :aligned], km=self.km,
                       out=(self.k8_buf[:, :, :aligned], self.ks_buf[:, :, : aligned // 32]))

    def _transform_new(self):
        self.ring_k[:, self.n_hist:].copy_(self.k)
        self.ring_v[:, self.n_hist:].copy_(self.v)
        q_eff = rope_prope_transform(self.q, self.p_q_t, self.cs_new, self.mode, out_layout=self.qo_layout)
        k_new = rope_prope_transform(self.k, self.p_new_k_inv, self.cs_new, self.mode, out_layout="NHD")
        v_new = rope_prope_transform(self.v, self.p_new_v_inv, None, out_layout="NHD")
        self.ring_k_enc[:, self.n_hist:].copy_(k_new)
        self.ring_v_enc[:, self.n_hist:].copy_(v_new)
        return q_eff

    def _sagev2_core_inputs(self, q_eff):
        q_v2 = q_eff
        k_v2 = self.ring_k_enc[:, : self.n_win]
        aligned = (self.n_hist // 64) * 64
        sageattn_v2.pack_k_into(k_v2[:, aligned:], self.km, self.k4_buf, self.k4s_buf, aligned)
        q4, q_scale, q_mean = sageattn_v2.quantize_q(q_v2)
        delta_s = sageattn_v2.compute_delta_s(k_v2, self.km, q_mean)
        return q4, q_scale, q_mean, self.k4_buf, self.k4s_buf, delta_s, self.ring_v_enc[:, : self.n_win], q_v2.size(1)

    def _sage_core_inputs(self, q_eff):
        attn_k = self.ring_k_enc[:, : self.n_win].transpose(1, 2)
        aligned = (self.n_hist // 32) * 32
        quantise_k(attn_k[:, :, aligned:], km=self.km,
                   out=(self.k8_buf[:, :, aligned:], self.ks_buf[:, :, aligned // 32:]))
        return q_eff, self.k8_buf, self.ks_buf, self.ring_v_enc[:, : self.n_win].transpose(1, 2)

    def forward(self):
        """Every forward: new-token RoPE/PRoPE, ring writes, attention, output PRoPE."""
        q_eff = self._transform_new()
        if self.backend == "sagev2":
            o = sageattn_v2.attention(*self._sagev2_core_inputs(q_eff))
        elif self.is_sage:
            o = _attend_int8(*self._sage_core_inputs(q_eff))
        else:
            attn_k = self.ring_k_enc[:, : self.n_win].transpose(1, 2)
            attn_v = self.ring_v_enc[:, : self.n_win].transpose(1, 2)
            o = F.scaled_dot_product_attention(q_eff, attn_k, attn_v, is_causal=False)
        return rope_prope_transform(o, self.p_q, None, in_layout=self.qo_layout, out_layout="NHD")

    def core(self):
        """The attention kernel alone, on inputs already prepared (K already quantized)."""
        q_eff = self._transform_new()
        if self.backend == "sagev2":
            args = self._sagev2_core_inputs(q_eff)
            return lambda: sageattn_v2.attention(*args)
        if self.is_sage:
            args = self._sage_core_inputs(q_eff)
            return lambda: _attend_int8(*args)
        attn_k = self.ring_k_enc[:, : self.n_win].transpose(1, 2)
        attn_v = self.ring_v_enc[:, : self.n_win].transpose(1, 2)
        return lambda: F.scaled_dot_product_attention(q_eff, attn_k, attn_v, is_causal=False)


def bench_attention(device):
    rows = {}
    outputs = {}
    for backend in ("sdpa", "sage", "sagev2"):
        layer = AttentionLayer(backend, device)
        layer.history()
        outputs[backend] = layer.forward().float()
        hist_ms = timeit(layer.history)
        fwd_ms = timeit(layer.forward)
        core_ms = timeit(layer.core())
        chunk_ms = (FORWARDS_PER_CHUNK * fwd_ms + hist_ms) * DIT_LAYERS
        rows[backend] = dict(core_ms=core_ms, forward_ms=fwd_ms, history_ms=hist_ms, dit_chunk_ms=chunk_ms)
        del layer
        torch.xpu.empty_cache()
    for backend in ("sage", "sagev2"):
        err = (outputs[backend] - outputs["sdpa"]).pow(2).mean().sqrt() / outputs["sdpa"].pow(2).mean().sqrt()
        rows[backend]["rel_rms_vs_sdpa"] = err.item()

    ref = rows["sdpa"]
    print(f"\n{torch.xpu.get_device_name(device)}, torch {torch.__version__}; one layer, Lq 1215 / Lkv 7290, median of 50")
    print(f"{'backend':8s} {'core ms':>9s} {'x':>6s} {'forward ms':>11s} {'x':>6s} {'history ms':>11s} {'per chunk ms':>13s} {'x':>6s}")
    for name, r in rows.items():
        print(f"{name:8s} {r['core_ms']:9.3f} {ref['core_ms'] / r['core_ms']:5.2f}x "
              f"{r['forward_ms']:11.3f} {ref['forward_ms'] / r['forward_ms']:5.2f}x "
              f"{r['history_ms']:11.3f} {r['dit_chunk_ms']:13.1f} {ref['dit_chunk_ms'] / r['dit_chunk_ms']:5.2f}x")
    return rows


def bench_quarot(device):
    from solarwm.backends.wan22.runtime.components import Wan5BVAE
    from solarwm.backends.wan22.runtime.modeling.vae_quarot import (
        _quarot_causal_conv3d_forward,
        is_quarot_candidate,
    )

    vae = Wan5BVAE(VAE_WEIGHTS, xpu_channels_last=True, xpu_vae_int8_quarot=True).to(device, dtype=torch.bfloat16)
    rows = []
    state = {"time": False}

    def wrap(name, conv):
        quarot_forward = conv.forward

        def forward(x, cache_x=None, accumulate_into=None):
            if state["time"]:
                acc = None if accumulate_into is None else accumulate_into.clone()
                bf16 = timeit(lambda: conv._orig_forward(x, cache_x=cache_x, accumulate_into=acc), 3, 20)
                quarot = timeit(lambda: _quarot_causal_conv3d_forward(conv, x, cache_x=cache_x, accumulate_into=acc), 3, 20)
                rows.append(dict(layer=name, shape=list(x.shape), cache_t=0 if cache_x is None else cache_x.shape[2],
                                 accumulate=accumulate_into is not None, bf16_ms=bf16, quarot_ms=quarot))
            return quarot_forward(x, cache_x=cache_x, accumulate_into=accumulate_into)

        conv.forward = forward

    for name, module in vae.module.decoder.named_modules():
        if is_quarot_candidate(name, module):
            wrap(name, module)

    torch.manual_seed(0)
    latents = torch.randn(1, 3, 48, 30, 54, device=device, dtype=torch.bfloat16)
    with vae.streaming_decode_session() as decode_tile:
        for i in range(3):
            state["time"] = i == 2
            decode_tile(latents)
    torch.xpu.synchronize()

    bf16 = sum(r["bf16_ms"] for r in rows)
    quarot = sum(r["quarot_ms"] for r in rows)
    print(f"\n{torch.xpu.get_device_name(device)}; QuaRot candidate convolutions in one steady-state 480p decode tile")
    for r in rows:
        print(f"  {r['layer']:40s} {str(r['shape']):26s} {r['bf16_ms']:8.3f} -> {r['quarot_ms']:8.3f} ms  {r['bf16_ms'] / r['quarot_ms']:.2f}x")
    print(f"  {len(rows)} layers: BF16 {bf16:.1f} ms -> QuaRot {quarot:.1f} ms per tile ({bf16 / quarot:.2f}x)")
    return dict(layers=rows, bf16_ms=bf16, quarot_ms=quarot)


def _block0_svdquant(device):
    """Block 0's real `sdpa_k4` tensors per linear, and the production `SYCLSVDQuantLinear`s."""
    from safetensors import safe_open

    from solarwm.backends.wan22.runtime.modeling.svdquant_linear import SYCLSVDQuantLinear

    manifest = json.loads((SVDQUANT_DIR / "manifest.json").read_text())
    sets = {}
    for shard in manifest["shards"]:
        with safe_open(str(SVDQUANT_DIR / shard), "pt") as f:
            for key in f.keys():
                set_name, rest = key.split(".", 1)
                if rest.startswith("blocks.0."):
                    module, param = rest.removeprefix("blocks.0.").rsplit(".", 1)
                    sets.setdefault(module, {}).setdefault(set_name, {})[param] = f.get_tensor(key)
    policy = manifest["activation_policy"]
    lin = {
        suffix: SYCLSVDQuantLinear(
            sets[suffix], manifest["set_mapping"],
            {b: float(p["blocks.0"][suffix]) for b, p in policy.items()}, device=device)
        for suffix in SYCLSVDQuantLinear.TARGET_SUFFIXES
    }
    return sets, lin


def bench_svdquant(device):
    from solarwm.backends.wan22.runtime.modeling.model import WanRMSNorm
    from solarwm.backends.wan22.runtime.modeling.svdquant_linear import SharedInputQuant, SVDQuantFFN
    from solarwm.kernels.dit_fused import onednn_linear_gelu_tanh, triton_rmsnorm

    _, lin = _block0_svdquant(device)
    # same wiring as SVDQuantModelManager.load_svdquant_checkpoint(fused=True)
    qkv = [lin["self_attn.q"], lin["self_attn.k"], lin["self_attn.v"]]
    norm_q, norm_k = WanRMSNorm(3072, eps=1e-6).to(device), WanRMSNorm(3072, eps=1e-6).to(device)
    qkv[0].norm, qkv[1].norm = norm_q, norm_k
    shared = SharedInputQuant(qkv)
    for i, m in enumerate(qkv):
        m.shared_input, m.shared_index = shared, i
    ffn = SVDQuantFFN(lin["ffn.0"], lin["ffn.2"])

    bf = dict(dtype=torch.bfloat16, device=device)
    torch.manual_seed(0)
    x = torch.randn(1, NEW_FRAMES * H_PATCH * W_PATCH, 3072, **bf)
    w = {n: torch.randn(o, i, **bf) * i ** -0.5 for n, (o, i) in
         dict(q=(3072, 3072), k=(3072, 3072), v=(3072, 3072), up=(14336, 3072), down=(3072, 14336)).items()}
    b = {n: torch.zeros(t.shape[0], **bf) for n, t in w.items()}
    gq, gk = norm_q.weight.to(torch.bfloat16), norm_k.weight.to(torch.bfloat16)

    def bf16_qkv():
        triton_rmsnorm(F.linear(x, w["q"], b["q"]), gq, eps=1e-6)
        triton_rmsnorm(F.linear(x, w["k"], b["k"]), gk, eps=1e-6)
        F.linear(x, w["v"], b["v"])

    def svd_qkv():
        qkv[0](x), qkv[1](x), qkv[2](x)

    def bf16_ffn():
        F.linear(onednn_linear_gelu_tanh(x, w["up"], b["up"]), w["down"], b["down"])

    rows = {}
    for name, ref, opt in (("qkv", bf16_qkv, svd_qkv), ("ffn", bf16_ffn, lambda: ffn(x))):
        rows[name] = dict(bf16_ms=timeit(ref), svdquant_ms=timeit(opt))
    bf16 = sum(r["bf16_ms"] for r in rows.values())
    svd = sum(r["svdquant_ms"] for r in rows.values())
    rows["block"] = dict(bf16_ms=bf16, svdquant_ms=svd)
    print(f"\n{torch.xpu.get_device_name(device)}; one DiT block's quantized linears, M=1215, bin 0, median of 50")
    for name, r in rows.items():
        print(f"  {name:6s} BF16 {r['bf16_ms']:7.3f} ms -> SVDQuant {r['svdquant_ms']:7.3f} ms  {r['bf16_ms'] / r['svdquant_ms']:.2f}x")
    print(f"  per chunk (30 layers x 5 forwards): {bf16 * 150:.1f} -> {svd * 150:.1f} ms")
    return rows


def bench_svdquant_llm_scaler(device):
    """Each SVDQuant linear of block 0, ours vs llm-scaler's, on the same weights and input.

    llm-scaler runs SVDQuant through comfy-kitchen-xpu's XPU adapter on omni_xpu_kernel
    (`quantize_svdquant_w4a4` + `scaled_mm_svdquant_w4a4_preconverted`, the path its ComfyUI
    uses after load-time weight conversion). Needs omni_xpu_kernel and comfy-kitchen-xpu on
    PYTHONPATH.
    """
    from comfy_kitchen.backends.xpu import svdquant as ck
    from omni_xpu_kernel import svdq

    sets, lin = _block0_svdquant(device)
    torch.manual_seed(0)
    x = torch.randn(NEW_FRAMES * H_PATCH * W_PATCH, 3072, dtype=torch.bfloat16, device=device)
    rows = {}
    for suffix, ours in lin.items():
        t = {k: v.to(device) for k, v in sets[suffix]["set0"].items()}
        k_in = t["qweight"].shape[1] * 2
        xin = x if k_in == 3072 else torch.randn(x.shape[0], k_in, dtype=torch.bfloat16, device=device)
        packed_u4, scales_f16 = svdq.prepare_onednn_weights(t["qweight"].view(torch.uint8), t["w_scale"].to(torch.bfloat16).t().contiguous())
        smooth = t["smooth_scale"].to(torch.bfloat16)
        lora_down = t["lora_a"].to(torch.bfloat16).t().contiguous()
        lora_up = t["lora_b"].to(torch.bfloat16).contiguous()
        bias = t["bias"].to(torch.bfloat16) if "bias" in t else None

        def theirs():
            q, ascales, lora_act = ck.quantize_svdquant_w4a4(xin, smooth, lora_down)
            return ck.scaled_mm_svdquant_w4a4_preconverted(q, packed_u4, ascales, scales_f16, lora_act, lora_up, bias)[: xin.shape[0]]

        y_ours, y_theirs = ours(xin).float(), theirs().float()
        rows[suffix] = dict(shape=[xin.shape[0], k_in, y_ours.shape[-1]], ours_ms=timeit(lambda: ours(xin)), llm_scaler_ms=timeit(theirs),
                            rel_diff=((y_ours - y_theirs).pow(2).mean().sqrt() / y_theirs.pow(2).mean().sqrt()).item())
    ours_ms = sum(r["ours_ms"] for r in rows.values())
    theirs_ms = sum(r["llm_scaler_ms"] for r in rows.values())
    rows["block"] = dict(ours_ms=ours_ms, llm_scaler_ms=theirs_ms)
    print(f"\n{torch.xpu.get_device_name(device)}; one SVDQuant linear at a time (act quant + LoRA + INT4 GEMM), median of 50")
    for name, r in rows.items():
        extra = f"  M,K,N {r['shape']}  rel diff {r['rel_diff']:.4f}" if "shape" in r else ""
        print(f"  {name:12s} llm-scaler {r['llm_scaler_ms']:7.3f} ms -> ours {r['ours_ms']:7.3f} ms  {r['llm_scaler_ms'] / r['ours_ms']:.2f}x{extra}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("block", choices=("attention", "quarot", "svdquant", "svdquant-llm-scaler"))
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    device = torch.device("xpu")
    result = {"attention": bench_attention, "quarot": bench_quarot, "svdquant": bench_svdquant,
              "svdquant-llm-scaler": bench_svdquant_llm_scaler}[args.block](device)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
