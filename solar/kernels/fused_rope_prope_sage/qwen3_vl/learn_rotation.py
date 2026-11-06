"""Learn per-KV-head orthogonal rotations that make INT4 Q*K^T accurate (offline, captured activations).

R = H matrix_exp(A - A^T) (orthogonal; H = normalised Hadamard, A starts at 0), applied to centred K and to
Q after RoPE: (qR)(kR)^T = qk^T, so only the quantisation error changes. A is to maximise the cosine between INT4 attention output (straight-through rounding) and exact
fp32 attention, on the even-indexed captured query rows; the odd rows are held out for evaluation.

    ./env.sh python vl8b/learn_rotation.py --dir act_capture_vl/needle_65536 --layers 0 1 2 --json logs/vl_rotation.json
Saves the learned rotations to <dir>/rotations.pt ({layer: [Hkv, D, D]}).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qk_quant_lib import causal_attend, cos_per_head, hadamard, reference, scheme_scores  # noqa: E402


def learn_head(qh, qm, kh, vh, qpos, rot0, steps, lr, train_mask, q_smooth):
    a = torch.zeros_like(rot0, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    ref = reference(qh[train_mask], kh, vh, qpos[train_mask]).detach()
    for _ in range(steps):
        rot = rot0 @ torch.linalg.matrix_exp(a - a.T)
        s = scheme_scores(qh[train_mask], kh, 4, qpos[train_mask], qm[train_mask], rot, q_smooth, ste=True)
        out = causal_attend_diff(s, vh, qpos[train_mask])
        loss = 1 - torch.nn.functional.cosine_similarity(out.flatten(1), ref.flatten(1), dim=1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (rot0 @ torch.linalg.matrix_exp(a - a.T)).detach()


def causal_attend_diff(scores, v, qpos):
    """Differentiable version of causal_attend (no bf16 rounding of P)."""
    keys = torch.arange(scores.shape[-1], device=scores.device)
    scores = scores.masked_fill(keys[None, None, :] > qpos[:, None, None], float("-inf"))
    p = torch.softmax(scores, -1)
    return torch.einsum("rgs,sd->rgd", p, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--q-smooth", action="store_true", help="train and evaluate with Q smoothing as well")
    ap.add_argument("--json")
    args = ap.parse_args()
    torch.xpu.set_per_process_memory_fraction(0.9)
    dev = "xpu"
    out_rot = {}
    results = []
    for layer in args.layers:
        d = torch.load(f"{args.dir}/L{layer}.pt")
        q, qpos, qmean_blk, k, v = d["q"], d["qpos"], d["qmean"], d["k"], d["v"]
        keep = torch.cat([torch.linspace(0, q.shape[0] - 257, args.rows - 256).long(), torch.arange(q.shape[0] - 256, q.shape[0])])
        q, qpos = q[keep], qpos[keep].to(dev)
        hq, hkv, dd = q.shape[1], k.shape[1], q.shape[2]
        g = hq // hkv
        train = (torch.arange(q.shape[0], device=dev) % 2) == 0
        test = ~train
        had = hadamard(dd, dev)
        rots = []
        cos = {n: torch.zeros(hq, dtype=torch.float64) for n in ("int8", "int4", "int4_had", "int4_learned")}
        for h in range(hkv):
            qh = q[:, h * g:(h + 1) * g].to(dev).float()
            qm = qmean_blk[(qpos // 256).cpu(), h * g:(h + 1) * g].to(dev).float()
            kh, vh = k[:, h].to(dev).float(), v[:, h].to(dev).float()
            r = learn_head(qh, qm, kh, vh, qpos, had, args.steps, args.lr, train, args.q_smooth)
            rots.append(r.cpu())
            with torch.no_grad():
                ref = reference(qh[test], kh, vh, qpos[test])
                for name, bits, rot in (("int8", 8, None), ("int4", 4, None), ("int4_had", 4, had), ("int4_learned", 4, r)):
                    s = scheme_scores(qh[test], kh, bits, qpos[test], qm[test], rot, args.q_smooth and bits == 4)
                    cos[name][h * g:(h + 1) * g] = cos_per_head(causal_attend(s, vh, qpos[test]), ref).cpu()
            torch.xpu.empty_cache()
        out_rot[layer] = torch.stack(rots)
        row = dict(layer=layer, q_smooth=args.q_smooth, **{f"{n}_mean": round(float(c.mean()), 5) for n, c in cos.items()},
                   **{f"{n}_min": round(float(c.min()), 4) for n, c in cos.items()})
        results.append(row)
        print(f"L{layer:<2d} (held-out rows) " + "  ".join(f"{n}={row[n + '_mean']:.4f}/{row[n + '_min']:.3f}" for n in cos), flush=True)
    torch.save(out_rot, f"{args.dir}/rotations{'_qs' if args.q_smooth else ''}.pt")
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
