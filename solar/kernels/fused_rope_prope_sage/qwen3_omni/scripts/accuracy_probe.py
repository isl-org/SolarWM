"""Phase 1 real-activation accuracy: Sage vs SDPA inside Qwen3-Omni prefill.

(1) per attention call: Sage output vs SDPA on the same Q/K/V (compare mode)
(2) end to end: next-token logits with all-Sage vs all-SDPA (top-1 / top-5 / KL)

Inputs: OmniBench image+audio samples, plus long natural-language prompts built from
OmniBench's human-written image/audio captions, truncated to target token lengths.
"""
import json, sys, time
import pyarrow.parquet as pq
import torch, torch.nn.functional as F
sys.path.insert(0, "/home/gta/omni_bench/scripts")
import sage_attention as SA
from model_loader import load
from omnibench_data import iter_samples, build_inputs, PARQUETS

LENS = [int(x) for x in sys.argv[1:]] or [8192, 16384, 32768]
SA.register()
model, proc = load(SA.NAME)
tok = proc.tokenizer

def caption_corpus():
    parts = []
    for f in PARQUETS:
        t = pq.read_table(f, columns=["image content", "audio content", "question"]).to_pylist()
        parts += [f"Scene: {r['image content']} Sound: {r['audio content']} Question: {r['question']}" for r in t]
    return "\n".join(parts)

def text_inputs(n_tokens, corpus_ids):
    body = tok.decode(corpus_ids[: n_tokens - 64])
    msg = [{"role": "user", "content": [{"type": "text", "text": body + "\n\nSummarise the recurring themes above in one sentence."}]}]
    txt = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    return proc(text=txt, return_tensors="pt").to("xpu")

@torch.no_grad()
def last_logits(inputs):
    out = model.thinker(**inputs, use_cache=False, logits_to_keep=1)
    return out.logits[0, -1].float()

def run(tag, inputs):
    L = inputs["input_ids"].shape[1]
    SA.STATS.clear(); SA.set_compare(True); SA.set_force_sdpa(False)
    t = time.perf_counter(); ls = last_logits(inputs); torch.xpu.synchronize(); t_cmp = time.perf_counter() - t
    stats = list(SA.STATS)
    SA.set_compare(False); SA.set_force_sdpa(True)
    lr = last_logits(inputs)
    SA.set_force_sdpa(False)
    p_s, p_r = F.log_softmax(ls, -1), F.log_softmax(lr, -1)
    kl = F.kl_div(p_s, p_r, log_target=True, reduction="sum").item()
    t5s, t5r = set(ls.topk(5).indices.tolist()), set(lr.topk(5).indices.tolist())
    by_mod = {}
    for s in stats:
        by_mod.setdefault(s["module"], []).append(s)
    summ = {m: dict(calls=len(v), cos_min=min(x["cos"] for x in v), cos_mean=sum(x["cos"] for x in v) / len(v),
                    rel_mean=sum(x["rel_l2"] for x in v) / len(v), rel_max=max(x["rel_l2"] for x in v),
                    worst_layer=min(v, key=lambda x: x["cos"])["layer"]) for m, v in by_mod.items()}
    res = dict(tag=tag, tokens=L, kl=kl, top1_match=ls.argmax().item() == lr.argmax().item(),
               top5_overlap=len(t5s & t5r), top1_sage=tok.decode(ls.argmax()), top1_sdpa=tok.decode(lr.argmax()),
               per_module=summ, compare_pass_s=t_cmp, calls=stats)
    print(f"[{tag}] tokens={L} top1_match={res['top1_match']} top5_overlap={res['top5_overlap']}/5 KL={kl:.4f} "
          f"({res['top1_sage']!r} vs {res['top1_sdpa']!r})", flush=True)
    for m, d in summ.items():
        print(f"    {m:34s} calls={d['calls']:3d} cos min/mean={d['cos_min']:.4f}/{d['cos_mean']:.4f} "
              f"rel_l2 mean/max={d['rel_mean']:.3f}/{d['rel_max']:.3f} worst_layer={d['worst_layer']}", flush=True)
    return res

results = []
for s in iter_samples(limit=3):
    results.append(run(f"omnibench_{s['index']}", build_inputs(proc, s, "xpu")))
corpus_ids = tok(caption_corpus())["input_ids"]
print("caption corpus tokens:", len(corpus_ids), flush=True)
for n in LENS:
    if n <= len(corpus_ids):
        results.append(run(f"text_{n}", text_inputs(n, corpus_ids)))
        torch.xpu.empty_cache()
print("fallback counts:", dict(SA.COUNTS))
json.dump(results, open("/home/gta/omni_bench/logs/accuracy_probe.json", "w"), indent=1)
