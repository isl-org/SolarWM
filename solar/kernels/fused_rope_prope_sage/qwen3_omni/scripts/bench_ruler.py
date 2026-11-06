"""RULER context-length sweep: SDPA vs SageAttention inside Qwen3-Omni (thinker, text only).

One model load; arms are switched per call (same weights), alternating order per sample:
  sdpa      every attention call on torch SDPA
  sage      thinker (causal, GQA) + audio encoder on the Sage kernel; vision / decode on SDPA
  sage_l3   sage, but thinker layer 3 kept on SDPA (its INT4 error grows with context)

Per (sample, arm) row in the output JSONL:
  ttft_s            generate() start -> first logits processor call (after device sync)
  e2e_s, n_gen, tpot_s
  peak_prefill_gib  max_memory_allocated at the first token (prefill peak); act_prefill_gib = peak - weights
  attn_ms           summed XPU-event time of the 48 thinker attention calls in prefill
  prefill/attention FLOPs and minimum-traffic bytes (analytic), score (RULER metric)

    env.sh python bench_ruler.py --lengths 2048 8192 --per-task 2 --arms sdpa sage --tag pilot
"""
import argparse, json, math, os, sys, time
import torch
from transformers import LogitsProcessor, LogitsProcessorList

sys.path.insert(0, "/home/gta/omni_bench/scripts")
sys.path.insert(0, "/home/gta/omni_bench/ruler")
import sage_attention as SA  # noqa: E402
from model_loader import load  # noqa: E402
from score import score  # noqa: E402

DATA = "/home/gta/omni_bench/ruler/data/ruler_{}.jsonl"
LOGS = "/home/gta/omni_bench/logs"
EOS_IDS = None


def set_arm(arm):
    SA.set_force_sdpa(arm == "sdpa")
    SA.set_sdpa_layers({3} if arm == "sage_l3" else set())


class FirstToken(LogitsProcessor):
    """Called once per generated token, right after the logits of that step exist."""
    def __init__(self):
        self.t = None
        self.peak = None

    def __call__(self, input_ids, scores):
        if self.t is None:
            torch.xpu.synchronize()
            self.t = time.perf_counter()
            self.peak = torch.xpu.max_memory_allocated()
            SA.set_timing_active(False)
        return scores


def active_params_per_token(model):
    """Logical weights touched per token in the thinker text model (experts scaled by top_k / E)."""
    tc = model.config.thinker_config.text_config
    frac = tc.num_experts_per_tok / tc.num_experts
    total = 0
    for name, m in model.thinker.model.language_model.named_modules():
        fin = getattr(m, "in_features", None) or getattr(m, "infeatures", None)
        fout = getattr(m, "out_features", None) or getattr(m, "outfeatures", None)
        if fin and fout and not any(True for _ in m.children()):
            total += fin * fout * (frac if ".experts." in name else 1.0)
    return total


def thinker_weight_bytes(model):
    lm = model.thinker.model.language_model
    return sum(t.numel() * t.element_size() for t in list(lm.parameters()) + list(lm.buffers()))


def attn_flops(S, H, D, layers):
    return layers * 4 * H * D * S * (S + 1) / 2


def attn_bytes(S, H, Hkv, D, layers, arm, sage_layers):
    e = 2  # bf16
    sdpa = S * H * D * e * 2 + 2 * S * Hkv * D * e            # Q read + O write, K/V read
    q_blocks = math.ceil(S / 256)
    kv_pad = math.ceil(S / 64) * 64
    ds = H * q_blocks * kv_pad * 4 * 0.5                         # causal half of dS, fp32
    sage = (S * H * D * e + S * Hkv * D * e                      # quantize: read Q, K (bf16)
            + (S * H * D // 2 + S * Hkv * D // 2) * 2            # Q4 / K4 write + read
            + (S * H + S * Hkv) * 4 * 2                          # scales write + read
            + ds * 2                                             # dS write + read
            + S * Hkv * D * e + S * H * D * e)                   # V read, O write
    if arm == "sdpa":
        return layers * sdpa
    return sage_layers * sage + (layers - sage_layers) * sdpa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 16384, 32768, 63488])
    ap.add_argument("--arms", nargs="+", default=["sdpa", "sage", "sage_l3"])
    ap.add_argument("--per-task", type=int, default=20)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--max-new-tokens", type=int, help="override RULER's per-task value")
    a = ap.parse_args()

    SA.register()
    model, proc = load(SA.NAME)
    tok = proc.tokenizer
    global EOS_IDS
    EOS_IDS = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    tc = model.config.thinker_config.text_config
    H, Hkv, D, NL = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, tc.num_hidden_layers
    act_params = active_params_per_token(model)
    w_bytes = thinker_weight_bytes(model)
    weights_gib = torch.xpu.memory_allocated() / 2**30
    print(f"active params/token={act_params/1e9:.3f}B thinker weight bytes={w_bytes/2**30:.2f} GiB "
          f"resident={weights_gib:.2f} GiB", flush=True)

    out_path = f"{LOGS}/ruler_{a.tag}.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {(r["length"], r["task"], r["index"], r["arm"]) for r in map(json.loads, open(out_path))}
    fout = open(out_path, "a")

    def run_one(r, arm, record=True):
        set_arm(arm)
        text = tok.apply_chat_template([{"role": "user", "content": r["input"]}], add_generation_prompt=True,
                                       tokenize=False) + r.get("answer_prefix", "")
        enc = tok(text, return_tensors="pt").to("xpu")
        L = enc["input_ids"].shape[1]
        mnt = a.max_new_tokens or r["max_new_tokens"]
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()
        ft = FirstToken()
        SA.set_timing(True)
        t0 = time.perf_counter()
        with torch.no_grad():
            # The checkpoint's generation_config has no eos_token_id: without this, generation never
            # stops and runs to max_new_tokens emitting junk after the answer.
            out = model.thinker.generate(**enc, max_new_tokens=mnt, do_sample=False,
                                         eos_token_id=EOS_IDS, pad_token_id=EOS_IDS[1],
                                         logits_processor=LogitsProcessorList([ft]))
        torch.xpu.synchronize()
        t1 = time.perf_counter()
        attn_ms = SA.attention_time_ms()
        n_attn_calls = len(SA.TIMED)
        SA.set_timing(False)
        n_gen = out.shape[1] - L
        pred = tok.decode(out[0, L:], skip_special_tokens=True)
        if not record:
            return None
        sage_layers = 0 if arm == "sdpa" else NL - (1 if arm == "sage_l3" else 0)
        ttft = ft.t - t0
        a_fl = attn_flops(L, H, D, NL)
        p_fl = 2 * act_params * L + a_fl
        a_by = attn_bytes(L, H, Hkv, D, NL, arm, sage_layers)
        row = dict(tag=a.tag, length=r["length_bucket"], tokens=L, task=r["task"], index=r["index"], arm=arm,
                   ttft_s=ttft, e2e_s=t1 - t0, n_gen=n_gen, tpot_s=(t1 - ft.t) / max(n_gen - 1, 1),
                   peak_prefill_gib=ft.peak / 2**30, act_prefill_gib=ft.peak / 2**30 - weights_gib,
                   peak_total_gib=torch.xpu.max_memory_allocated() / 2**30,
                   attn_ms=attn_ms, attn_calls=n_attn_calls,
                   prefill_flops=p_fl, attn_flops=a_fl, attn_bytes=a_by,
                   prefill_bytes=w_bytes + a_by + L * NL * 2 * Hkv * D * 2,
                   score=score(r["task"], pred, r["outputs"]), pred=pred[:200])
        fout.write(json.dumps(row) + "\n")
        fout.flush()
        return row

    for Lb in a.lengths:
        rows = [json.loads(l) for l in open(DATA.format(Lb))]
        by_task = {}
        for r in rows:
            if a.tasks and r["task"] not in a.tasks:
                continue
            by_task.setdefault(r["task"], []).append(r)
        # round-robin over tasks (sample 0 of every task, then sample 1, ...) so a run stopped early
        # still covers all tasks evenly; re-running with the same --tag resumes where it stopped
        todo = [by_task[t][i] for i in range(a.per_task) for t in by_task if i < len(by_task[t])]
        if not todo:
            continue
        # warm-up: Triton / SYCL JIT for this length's shapes, once per arm, discarded
        for arm in a.arms:
            run_one(todo[0], arm, record=False)
        for i, r in enumerate(todo):
            arms = a.arms if i % 2 == 0 else a.arms[::-1]
            for arm in arms:
                if (r["length_bucket"], r["task"], r["index"], arm) in done:
                    continue
                row = run_one(r, arm)
                print(f"L={Lb} {r['task']:16s} #{r['index']:<3d} {arm:8s} tok={row['tokens']} ttft={row['ttft_s']:.2f}s "
                      f"attn={row['attn_ms']:.0f}ms e2e={row['e2e_s']:.2f}s gen={row['n_gen']} "
                      f"peak={row['peak_prefill_gib']:.2f}G score={row['score']:.2f}", flush=True)


if __name__ == "__main__":
    main()
