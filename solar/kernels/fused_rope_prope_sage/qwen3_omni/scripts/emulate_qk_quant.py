"""Emulate Q*K^T quantisation schemes on captured real Qwen3-Omni activations (scripts/capture_qkv.py).

For each captured layer: R query rows (the last 256 + evenly spread rows), all heads, causal attention
over the full prefix. Every scheme is compared with exact fp32 attention (cosine / relative L2 of
the attention output). P is rounded to bf16 before P*V and the row sum, as in the kernels.

Schemes:
  bf16         bf16 Q/K/V, fp32 scores (flash/SDPA-like floor)
  int4_ours    sage_causal: per-token INT4, K centred by a sampled mean, Q centred per 256-row block
               with the exact fp32 correction dS = q_mean . (K - k_mean)
  int4_ksmooth per-token INT4, exact K mean only (no Q smoothing)
  int8_ksmooth per-token INT8, exact K mean only           (upstream smoothing, our granularity)
  int8_qksmooth per-token INT8, Q per-block smoothing + exact K mean
  int8_upstream INT8, Q scale per 32 rows (per warp), K scale per 64 tokens (per block), exact K mean
  kernel       the real sage_causal kernel on the full sequence (validates int4_ours)

    ./env.sh python scripts/emulate_qk_quant.py --json logs/emulate_qk_quant.json
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.environ.get("SAGE_BUILD_DIR", "/home/gta/omni_bench/kernel/build"))
import sage_causal  # noqa: E402

CAP = "/home/gta/omni_bench/act_capture"
BQ = 256
K_MEAN_FULL, K_MEAN_STRIDE, K_MEAN_TILE = 2048, 512, 64


def quant(x, levels, group_rows=1):
    """Symmetric quantisation with one scale per `group_rows` consecutive rows (dim 0); x [N, D] fp32.
    Returns the dequantised tensor (round half to even, as the kernels do)."""
    n, d = x.shape
    pad = (-n) % group_rows
    xp = torch.cat([x, x.new_zeros(pad, d)]) if pad else x
    g = xp.view(-1, group_rows, d)
    scale = g.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-12) / levels
    return (torch.round(g / scale) * scale).view(-1, d)[:n]


def k_mean_sampled(k):
    s = k.shape[0]
    if s < K_MEAN_FULL:
        return k.mean(0)
    idx = torch.arange(s, device=k.device)
    return k[(idx % K_MEAN_STRIDE) < K_MEAN_TILE].mean(0)


def block_means(q):
    s, d = q.shape
    nb = (s + BQ - 1) // BQ
    pad = nb * BQ - s
    qp = torch.cat([q, q.new_zeros(pad, d)]) if pad else q
    sums = qp.view(nb, BQ, d).sum(1)
    counts = torch.full((nb, 1), BQ, device=q.device, dtype=q.dtype)
    counts[-1] = BQ - pad
    return sums / counts            # [nb, D]


def attend(scores, v, rows):
    """scores [R, S] fp32 (scaled), causal over keys <= row; v [S, D] fp32. P rounded to bf16."""
    keys = torch.arange(scores.shape[1], device=scores.device)
    scores = scores.masked_fill(keys[None, :] > rows[:, None], float("-inf"))
    p = torch.exp(scores - scores.amax(1, keepdim=True)).to(torch.bfloat16).float()
    return (p @ v) / p.sum(1, keepdim=True)


def run_layer(q, k, v, rows, kernel_out):
    """q [S, Hq, D], k/v [S, Hkv, D] (bf16 on device). Returns {scheme: (cos, rel_l2, worst_head_cos)}."""
    s, hq, d = q.shape
    hkv = k.shape[1]
    scale = d ** -0.5
    acc = {}

    def add(name, h, out, ref):
        a = acc.setdefault(name, {"dot": 0.0, "nn": 0.0, "rr": 0.0, "err": 0.0, "worst": 1.0})
        o, r = out.flatten().double(), ref.flatten().double()
        a["dot"] += float(o @ r); a["nn"] += float(o @ o); a["rr"] += float(r @ r); a["err"] += float((o - r) @ (o - r))
        a["worst"] = min(a["worst"], float(torch.nn.functional.cosine_similarity(o, r, dim=0)))

    for hk in range(hkv):
        kf, vf = k[:, hk].float(), v[:, hk].float()
        km_s, km_e = k_mean_sampled(kf), kf.mean(0)
        kc_s, kc_e = kf - km_s, kf - km_e
        k4_s, k4_e = quant(kc_s, 7), quant(kc_e, 7)
        k8_e, k8_blk = quant(kc_e, 127), quant(kc_e, 127, group_rows=64)
        kb = k[:, hk].float()            # bf16 values
        for g in range(hq // hkv):
            h = hk * (hq // hkv) + g
            qf = q[:, h].float()
            qr = qf[rows]
            ref = attend((qr @ kf.T) * scale, vf, rows)
            add("bf16", h, attend((q[rows, h].float() @ kb.T) * scale, vf, rows), ref)
            # Q block smoothing (kernel: blocks of 256 rows, exact fp32 dS = qm . (K - km))
            qm = block_means(qf)[rows // BQ]                               # [R, D]
            q4_sm = quant(qr - qm, 7)
            add("int4_ours", h, attend((q4_sm @ k4_s.T + qm @ kc_s.T) * scale, vf, rows), ref)
            add("int4_ksmooth", h, attend((quant(qr, 7) @ k4_e.T) * scale, vf, rows), ref)
            add("int8_ksmooth", h, attend((quant(qr, 127) @ k8_e.T) * scale, vf, rows), ref)
            add("int8_qksmooth", h, attend((quant(qr - qm, 127) @ k8_e.T + qm @ kc_e.T) * scale, vf, rows), ref)
            q8_warp = quant(qf, 127, group_rows=32)[rows]
            add("int8_upstream", h, attend((q8_warp @ k8_blk.T) * scale, vf, rows), ref)
            add("kernel", h, kernel_out[rows, h].float(), ref)
    return {n: dict(cos=a["dot"] / (a["nn"] * a["rr"]) ** 0.5, rel_l2=(a["err"] / a["rr"]) ** 0.5,
                    worst_head_cos=a["worst"]) for n, a in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--json")
    args = ap.parse_args()
    torch.xpu.set_per_process_memory_fraction(0.9)
    results = []
    for pdir in sorted(glob.glob(f"{CAP}/*")):
        layers = sorted({int(os.path.basename(f)[1:].split("_")[0]) for f in glob.glob(f"{pdir}/L*_q.pt")})
        for layer in layers:
            q, k, v = (torch.load(f"{pdir}/L{layer}_{n}.pt").to("xpu")[0] for n in "qkv")
            s = q.shape[0]
            last = torch.arange(s - 256, s, device="xpu")
            spread = torch.linspace(2048, s - 257, args.rows - 256, device="xpu").long()
            rows = torch.cat([spread, last])
            kernel_out = sage_causal.forward(q[None].contiguous(), k[None].contiguous(), v[None].contiguous(), True)[0]
            res = run_layer(q, k, v, rows, kernel_out)
            row = dict(prompt=os.path.basename(pdir), layer=layer, tokens=s, **{f"{n}_cos": round(m["cos"], 5) for n, m in res.items()},
                       **{f"{n}_worst": round(m["worst_head_cos"], 4) for n, m in res.items()},
                       **{f"{n}_rel": round(m["rel_l2"], 4) for n, m in res.items()})
            results.append(row)
            print(f"{row['prompt']:8s} L{layer:<2d} " + "  ".join(f"{n}={m['cos']:.4f}/{m['worst_head_cos']:.3f}" for n, m in res.items()), flush=True)
            del q, k, v, kernel_out
            torch.xpu.empty_cache()
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
