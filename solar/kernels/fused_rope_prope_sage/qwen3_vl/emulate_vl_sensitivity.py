"""Per-layer / per-head sensitivity of Qwen3-VL attention to Q*K^T quantisation (captured activations).

For every captured layer (capture_vl_qkv.py) and KV head: attention output of each scheme vs exact
fp32, as cosine per query head. Schemes:
  int8         installed kernel: per-token INT8, K centred, no Q smoothing
  int4         per-token INT4, K centred, no Q smoothing
  int4_qs      INT4 + Q smoothing (256-row block mean, exact correction)
  int4_had     INT4, Hadamard rotation of Q and K
  int4_had_qs  INT4, Hadamard rotation + Q smoothing

    ./env.sh python vl8b/emulate_vl_sensitivity.py --dir act_capture_vl/needle_65536 --json logs/vl_sensitivity_64k.json
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qk_quant_lib import causal_attend, cos_per_head, hadamard, reference, scheme_scores  # noqa: E402

SCHEMES = {
    "int8": dict(bits=8),
    "int4": dict(bits=4),
    "int4_qs": dict(bits=4, q_smooth=True),
    "int4_had": dict(bits=4, rot="had"),
    "int4_had_qs": dict(bits=4, rot="had", q_smooth=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--json")
    ap.add_argument("--max-rows", type=int, default=768)
    args = ap.parse_args()
    torch.xpu.set_per_process_memory_fraction(0.9)
    files = sorted(glob.glob(f"{args.dir}/L*.pt"), key=lambda f: int(os.path.basename(f)[1:-3]))
    results = []
    for f in files:
        layer = int(os.path.basename(f)[1:-3])
        d = torch.load(f)
        q, qpos, qmean_blk, k, v = d["q"], d["qpos"], d["qmean"], d["k"], d["v"]
        if q.shape[0] > args.max_rows:   # keep the last 256 rows, subsample the rest evenly
            keep = torch.cat([torch.linspace(0, q.shape[0] - 257, args.max_rows - 256).long(), torch.arange(q.shape[0] - 256, q.shape[0])])
            q, qpos = q[keep], qpos[keep]
        hq, hkv, dd = q.shape[1], k.shape[1], q.shape[2]
        g = hq // hkv
        dev = "xpu"
        qpos = qpos.to(dev)
        rot_h = hadamard(dd, dev)
        per = {n: torch.zeros(hq, dtype=torch.float64) for n in SCHEMES}
        for h in range(hkv):
            qh = q[:, h * g:(h + 1) * g].to(dev).float()
            qm = qmean_blk[(qpos // 256).cpu(), h * g:(h + 1) * g].to(dev).float()
            kh, vh = k[:, h].to(dev).float(), v[:, h].to(dev).float()
            ref = reference(qh, kh, vh, qpos)
            for name, cfg in SCHEMES.items():
                rot = rot_h if cfg.get("rot") == "had" else None
                s = scheme_scores(qh, kh, cfg["bits"], qpos, qm, rot, cfg.get("q_smooth", False))
                per[name][h * g:(h + 1) * g] = cos_per_head(causal_attend(s, vh, qpos), ref).cpu()
            del ref
        row = dict(layer=layer, tokens=k.shape[0], rows=q.shape[0],
                   **{f"{n}_mean": round(float(c.mean()), 5) for n, c in per.items()},
                   **{f"{n}_min": round(float(c.min()), 4) for n, c in per.items()},
                   **{f"{n}_heads": [round(float(x), 4) for x in c] for n, c in per.items()})
        results.append(row)
        print(f"L{layer:<2d} " + "  ".join(f"{n}={row[n + '_mean']:.4f}/{row[n + '_min']:.3f}" for n in SCHEMES), flush=True)
        torch.xpu.empty_cache()
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
