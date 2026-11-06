"""Capture real post-RoPE Q/K/V of every Qwen3-VL-8B text layer during a chunked long-context prefill.

Runs the bf16 flash arm (reference trajectory) on the needle-in-a-haystack prompt from mem_ramp.py.
Per layer, saved to OUT/L<layer>.pt:
  q       [R, Hq, D] bf16   sampled query rows (ROWS_PER_CHUNK per chunk, evenly spread, + the last 256)
  qpos    [R] int64         absolute positions of those rows
  qmean   [S/256, Hq, D]    fp32 means of every 256-row Q block (for Q-smoothing variants)
  k, v    [S, Hkv, D] bf16  all keys / values (the dequantised INT8-cache values the kernel sees)

    scripts/rss_guard.sh 24 logs/capture_vl.log -- ./env.sh python vl8b/capture_vl_qkv.py --length 65536
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vl_attention  # noqa: E402
import vl_model      # noqa: E402
import mem_ramp      # noqa: E402

BQ = 256
ROWS_PER_CHUNK = 48
LAST_ROWS = 256
STATE = {}


def capture(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, is_causal=None, **kw):
    sq = query.shape[2]
    layer = getattr(module, "layer_idx", None)
    if STATE.get("on") and sq > 1 and layer is not None:
        st = STATE.setdefault(layer, {"q": [], "qpos": [], "qsum": None})
        skv = key.shape[2]
        p0 = skv - sq                                              # absolute position of this chunk's row 0
        q = query[0].transpose(0, 1)                               # [sq, Hq, D]
        idx = torch.linspace(0, sq - 1, min(ROWS_PER_CHUNK, sq), device=q.device).long()
        if skv == STATE["total"]:                                  # last chunk: also keep the final rows
            idx = torch.unique(torch.cat([idx, torch.arange(max(0, sq - LAST_ROWS), sq, device=q.device)]))
        st["q"].append(q[idx].cpu()); st["qpos"].append((idx + p0).cpu())
        # block sums for 256-row Q-block means (chunks are multiples of 256 except possibly the last)
        nb = (sq + BQ - 1) // BQ
        qf = q.float()
        pad = nb * BQ - sq
        if pad:
            qf = torch.cat([qf, qf.new_zeros(pad, *qf.shape[1:])])
        sums = qf.view(nb, BQ, *qf.shape[1:]).sum(1).cpu()
        counts = torch.full((nb,), float(BQ)); counts[-1] = BQ - pad
        st.setdefault("sums", []).append(sums); st.setdefault("counts", []).append(counts)
        if skv == STATE["total"]:
            torch.save(dict(q=torch.cat(st["q"]), qpos=torch.cat(st["qpos"]),
                            qmean=torch.cat(st["sums"]) / torch.cat(st["counts"])[:, None, None],
                            k=key[0].transpose(0, 1).contiguous().cpu(), v=value[0].transpose(0, 1).contiguous().cpu()),
                       f"{STATE['out']}/L{layer}.pt")
            del STATE[layer]
    return vl_attention.attention_forward(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out", default="/home/gta/omni_bench/act_capture_vl")
    args = ap.parse_args()
    out = f"{args.out}/needle_{args.length}"
    os.makedirs(out, exist_ok=True)
    model, proc = vl_model.load()
    from transformers import AttentionInterface
    AttentionInterface.register(vl_attention.NAME, capture)
    vl_attention.set_arm("flash")
    ids = torch.tensor([mem_ramp.build_prompt(proc.tokenizer, args.length)], device="xpu")
    STATE.update(on=True, total=ids.shape[1], out=out)
    cache = vl_model.Int8KVCache(model, ids.shape[1])
    vl_model.prefill(model, ids, cache, args.chunk)
    torch.xpu.synchronize()
    print(f"[capture] {ids.shape[1]} tokens, {len(os.listdir(out))} layers -> {out}", flush=True)


if __name__ == "__main__":
    main()
