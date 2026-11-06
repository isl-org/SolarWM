"""RULER at 128K / 256K on Qwen3-VL-8B (INT4 weights, INT8 KV cache, chunked prefill): flash vs Sage.

Arms: flash (bf16 flash_causal, the baseline) and sage (sage_causal, the installed kernel). Each prompt
runs through both arms, the order alternating per sample; samples are interleaved across tasks so a
stopped run stays balanced. Rows already in logs/ruler_vl_<tag>.jsonl are skipped (resume).

    scripts/rss_guard.sh 24 logs/ruler_vl_128k.log -- ./env.sh python vl8b/bench_ruler_vl.py \
        --lengths 129024 --per-task 10 --tag rvl128k
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/gta/omni_bench/ruler")
import vl_attention  # noqa: E402
import vl_model      # noqa: E402
from score import score  # noqa: E402

DATA = "/home/gta/omni_bench/ruler/data_vl"
LOGS = "/home/gta/omni_bench/logs"


def run_one(model, tok, r, arm, chunk, eos):
    text = tok.apply_chat_template([{"role": "user", "content": r["input"]}], add_generation_prompt=True,
                                   tokenize=False) + r.get("answer_prefix", "")
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device="xpu")
    vl_attention.set_arm(arm)
    torch.xpu.synchronize(); torch.xpu.empty_cache(); torch.xpu.reset_peak_memory_stats()
    cache = vl_model.Int8KVCache(model, ids.shape[1] + r["max_new_tokens"])
    vl_attention.set_timing(True)
    t0 = time.perf_counter()
    logits = vl_model.prefill(model, ids, cache, chunk)
    torch.xpu.synchronize()
    ttft = time.perf_counter() - t0
    vl_attention.set_timing(False, clear=False)
    attn = vl_attention.attention_time_ms() / 1e3
    peak = torch.xpu.max_memory_allocated() / 2**30
    reserved = torch.xpu.max_memory_reserved() / 2**30
    out_ids, steps = vl_model.greedy_decode(model, logits, cache, r["max_new_tokens"], eos)
    e2e = time.perf_counter() - t0
    pred = tok.decode([t for t in out_ids if t not in eos], skip_special_tokens=True)
    del cache, logits
    torch.xpu.empty_cache()
    return dict(task=r["task"], index=r["index"], length=r["length_bucket"], tokens=ids.shape[1], arm=arm,
                ttft_s=round(ttft, 2), attn_s=round(attn, 2), e2e_s=round(e2e, 2), n_gen=len(out_ids),
                tpot_ms=round(1e3 * sum(steps) / max(len(steps), 1), 1), peak_prefill_gib=round(peak, 2),
                peak_reserved_gib=round(reserved, 2), pred=pred, score=score(r["task"], pred, r["outputs"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[129024])
    ap.add_argument("--per-task", type=int, default=10)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--arms", nargs="+", default=["flash", "sage"])
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    model, proc = vl_model.load()
    tok = proc.tokenizer
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    out_path = f"{LOGS}/ruler_vl_{a.tag}.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {(r["length"], r["task"], r["index"], r["arm"]) for r in map(json.loads, open(out_path))}
    vl_attention.set_arm("flash")   # warm-up (Triton JIT, first launches)
    warm = vl_model.Int8KVCache(model, 8192)
    vl_model.prefill(model, torch.randint(1000, 50000, (1, 8192), device="xpu"), warm, a.chunk)
    del warm; torch.xpu.empty_cache()
    for L in a.lengths:
        rows = [json.loads(l) for l in open(f"{DATA}/ruler_{L}.jsonl")]
        tasks = a.tasks or sorted({r["task"] for r in rows})
        by_task = {t: [r for r in rows if r["task"] == t][:a.per_task] for t in tasks}
        for i in range(a.per_task):
            for t in tasks:
                if i >= len(by_task[t]):
                    continue
                r = by_task[t][i]
                arms = a.arms if i % 2 == 0 else list(reversed(a.arms))
                for arm in arms:
                    if (L, t, r["index"], arm) in done:
                        continue
                    row = run_one(model, tok, r, arm, a.chunk, eos)
                    print(f"L={L} {t:16s} #{r['index']:<3d} {arm:6s} tok={row['tokens']} ttft={row['ttft_s']}s "
                          f"attn={row['attn_s']}s e2e={row['e2e_s']}s gen={row['n_gen']} peak={row['peak_prefill_gib']}G "
                          f"res={row['peak_reserved_gib']}G score={row['score']:.2f}", flush=True)
                    with open(out_path, "a") as f:
                        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
