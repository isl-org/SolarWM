"""Memory / speed ramp for Qwen3-VL-8B (INT4) with an INT8 KV cache and chunked prefill.

For each target length and attention arm: a needle-in-a-haystack prompt (essay haystack, a magic
number at 10% depth, the question at the end, Qwen chat template), chunked prefill, greedy decode.
Records TTFT, prefill attention time, peak VRAM, TPOT and whether the needle was retrieved.

    scripts/rss_guard.sh 24 logs/vl_ramp.log -- ./env.sh python vl8b/mem_ramp.py \
        --lengths 32768 65536 131072 262144 --arms sage flash --json logs/vl_ramp.jsonl
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vl_attention  # noqa: E402
import vl_model      # noqa: E402

# Any RULER jsonl with niah_single_2 rows (Paul Graham essay haystack) works as the haystack source.
RULER = os.environ.get("HAYSTACK_JSONL", "/home/gta/omni_bench/ruler/data/ruler_63488.jsonl")
NEEDLE = "The special magic number for the blue lighthouse is 7381946."
QUESTION = "What is the special magic number for the blue lighthouse mentioned in the text? Answer with the number only."
ANSWER = "7381946"


def haystack_ids(tok, n_tokens):
    essays = []
    for line in open(RULER):
        d = json.loads(line)
        if d["task"] == "niah_single_2":
            essays.append(d["input"])
    ids = tok("\n".join(essays), add_special_tokens=False)["input_ids"]
    reps = n_tokens // len(ids) + 1
    return (ids * reps)[:n_tokens]


def build_prompt(tok, length):
    head = tok.apply_chat_template([{"role": "user", "content": "@@"}], tokenize=False, add_generation_prompt=True)
    pre, post = head.split("@@")
    pre_ids = tok(pre + "Read the following text carefully.\n", add_special_tokens=False)["input_ids"]
    post_ids = tok("\n\n" + QUESTION + post, add_special_tokens=False)["input_ids"]
    needle_ids = tok("\n" + NEEDLE + "\n", add_special_tokens=False)["input_ids"]
    body = length - len(pre_ids) - len(post_ids) - len(needle_ids)
    hay = haystack_ids(tok, body)
    cut = body // 10
    return pre_ids + hay[:cut] + needle_ids + hay[cut:] + post_ids


def gib(x):
    return round(x / 2**30, 2)


def run(model, tok, length, arm, chunk, max_new):
    ids = torch.tensor([build_prompt(tok, length)], device="xpu")
    vl_attention.set_arm(arm)
    torch.xpu.synchronize(); torch.xpu.empty_cache(); torch.xpu.reset_peak_memory_stats()
    cache = vl_model.Int8KVCache(model, ids.shape[1] + max_new)
    torch.xpu.synchronize()
    after_cache = torch.xpu.memory_allocated()
    vl_attention.set_timing(True)
    t0 = time.perf_counter()
    logits = vl_model.prefill(model, ids, cache, chunk)
    torch.xpu.synchronize()
    ttft = time.perf_counter() - t0
    vl_attention.set_timing(False, clear=False)
    attn_ms = vl_attention.attention_time_ms()
    peak_prefill, reserved = torch.xpu.max_memory_allocated(), torch.xpu.max_memory_reserved()
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    out_ids, step_times = vl_model.greedy_decode(model, logits, cache, max_new, eos)
    text = tok.decode(out_ids, skip_special_tokens=True)
    row = dict(length=length, tokens=ids.shape[1], arm=arm, chunk=chunk, ttft_s=round(ttft, 2),
               attn_s=round(attn_ms / 1e3, 2), kv_cache_gib=gib(after_cache - weights_bytes),
               peak_prefill_gib=gib(peak_prefill), peak_reserved_gib=gib(reserved),
               peak_total_gib=gib(torch.xpu.max_memory_allocated()),
               tpot_ms=round(1e3 * sum(step_times) / max(len(step_times), 1), 1),
               answer=text.strip()[:60], correct=ANSWER in text)
    del cache, logits
    torch.xpu.empty_cache()
    return row


def main():
    global weights_bytes
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[32768, 65536, 131072, 262144])
    ap.add_argument("--arms", nargs="+", default=["sage", "flash"])
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--json")
    args = ap.parse_args()
    model, proc = vl_model.load()
    tok = proc.tokenizer
    torch.xpu.synchronize()
    weights_bytes = torch.xpu.memory_allocated()
    print(f"[load] weights {gib(weights_bytes)} GiB allocated, {gib(torch.xpu.memory_reserved())} GiB reserved", flush=True)
    for arm in args.arms:  # warm-up: Triton JIT for the INT4 GEMMs, kernel first-launch costs
        run(model, tok, 2 * args.chunk, arm, args.chunk, 2)
    for length in args.lengths:
        need = vl_model.Int8KVCache.bytes_for(model, length + args.max_new)
        print(f"[plan] L={length}: KV cache + dequant buffers {gib(need)} GiB", flush=True)
        for arm in args.arms:
            try:
                row = run(model, tok, length, arm, args.chunk, args.max_new)
            except torch.OutOfMemoryError as e:
                row = dict(length=length, arm=arm, chunk=args.chunk, error="OOM", detail=str(e)[:200],
                           peak_prefill_gib=gib(torch.xpu.max_memory_allocated()))
                torch.xpu.empty_cache()
            print("ROW " + json.dumps(row), flush=True)
            if args.json:
                with open(args.json, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
