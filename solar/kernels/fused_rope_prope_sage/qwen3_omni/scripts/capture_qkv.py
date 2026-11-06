"""Capture real post-RoPE Q/K/V of selected Qwen3-Omni thinker layers during a 64K RULER prefill.

The model runs on SDPA (reference trajectory); for each selected layer the attention inputs are saved
as bf16 [1, S, H, D] tensors to OUT/<task>_<index>/L<layer>_{q,k,v}.pt. Prefill only, no KV cache.

    scripts/rss_guard.sh 24 logs/capture_qkv.log -- ./env.sh python scripts/capture_qkv.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/home/gta/omni_bench/scripts")
import sage_attention as SA  # noqa: E402
from model_loader import load  # noqa: E402

OUT = "/home/gta/omni_bench/act_capture"
DATA = "/home/gta/omni_bench/ruler/data/ruler_63488.jsonl"
LAYERS = {0, 3, 12, 24, 36, 47}
PROMPTS = [("cwe", 0), ("qa_2", 0)]
_dst = None


def capture_forward(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None,
                    is_causal=None, **kwargs):
    layer = getattr(module, "layer_idx", None)
    if (_dst and query.shape[2] > 1 and layer in LAYERS
            and type(module).__name__.endswith("ThinkerTextAttention")):
        for name, t in (("q", query), ("k", key), ("v", value)):
            torch.save(t.transpose(1, 2).contiguous().cpu(), f"{_dst}/L{layer}_{name}.pt")
    return SA.sage_attention_forward(module, query, key, value, attention_mask, dropout, scaling,
                                     is_causal, **kwargs)


def main():
    global _dst
    SA.register()
    model, proc = load(SA.NAME)
    from transformers import AttentionInterface
    AttentionInterface.register(SA.NAME, capture_forward)
    SA.set_force_sdpa(True)
    tok = proc.tokenizer
    rows = {(d["task"], d["index"]): d for d in map(json.loads, open(DATA))}
    for task, idx in PROMPTS:
        r = rows[(task, idx)]
        text = tok.apply_chat_template([{"role": "user", "content": r["input"]}], add_generation_prompt=True,
                                       tokenize=False) + r.get("answer_prefix", "")
        enc = tok(text, return_tensors="pt").to("xpu")
        _dst = f"{OUT}/{task}_{idx}"
        os.makedirs(_dst, exist_ok=True)
        torch.xpu.empty_cache()
        with torch.no_grad():
            model.thinker(**enc, use_cache=False)
        torch.xpu.synchronize()
        _dst = None
        print(f"[capture] {task} #{idx}: {enc['input_ids'].shape[1]} tokens -> {OUT}/{task}_{idx}", flush=True)


if __name__ == "__main__":
    main()
