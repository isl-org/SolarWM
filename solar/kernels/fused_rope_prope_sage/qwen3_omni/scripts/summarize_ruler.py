"""Aggregate ruler_<tag>.jsonl into per (length, arm) metrics; writes CSV + markdown and prints it.

    env.sh python summarize_ruler.py <tag> [<tag> ...]
"""
import csv, json, statistics as st, sys
from collections import defaultdict

LOGS = "/home/gta/omni_bench/logs"
peaks = json.load(open(f"{LOGS}/device_peaks.json"))
PEAK_TF, PEAK_BW = peaks["bf16_gemm_tflops"], peaks["peak_bw_gbs"]

rows = [r for t in sys.argv[1:] for r in map(json.loads, open(f"{LOGS}/ruler_{t}.jsonl"))]
groups = defaultdict(list)
for r in rows:
    groups[(r["length"], r["arm"])].append(r)

med = lambda xs: st.median(xs) if xs else float("nan")
out = []
for (L, arm), g in sorted(groups.items()):
    ttft = med([r["ttft_s"] for r in g])
    attn_s = med([r["attn_ms"] for r in g]) / 1e3
    a_fl = med([r["attn_flops"] for r in g])
    a_by = med([r["attn_bytes"] for r in g])
    p_fl = med([r["prefill_flops"] for r in g])
    p_by = med([r["prefill_bytes"] for r in g])
    out.append(dict(
        length=L, arm=arm, n=len(g), tokens=round(med([r["tokens"] for r in g])),
        ttft_s=ttft, e2e_s=med([r["e2e_s"] for r in g]), tpot_ms=1e3 * med([r["tpot_s"] for r in g]),
        n_gen=med([r["n_gen"] for r in g]),
        peak_prefill_gib=max(r["peak_prefill_gib"] for r in g), act_prefill_gib=max(r["act_prefill_gib"] for r in g),
        attn_s=attn_s, attn_share=attn_s / ttft,
        attn_tflops=a_fl / attn_s / 1e12, attn_tf_util=a_fl / attn_s / 1e12 / PEAK_TF,
        attn_gbs=a_by / attn_s / 1e9, attn_bw_util=a_by / attn_s / 1e9 / PEAK_BW,
        prefill_tflops=p_fl / ttft / 1e12, prefill_tf_util=p_fl / ttft / 1e12 / PEAK_TF,
        prefill_gbs=p_by / ttft / 1e9, prefill_bw_util=p_by / ttft / 1e9 / PEAK_BW,
        score=100 * st.mean(r["score"] for r in g)))

base = {r["length"]: r for r in out if r["arm"] == "sdpa"}
for r in out:
    b = base.get(r["length"])
    r["ttft_speedup"] = b["ttft_s"] / r["ttft_s"] if b else float("nan")
    r["attn_speedup"] = b["attn_s"] / r["attn_s"] if b else float("nan")
    r["e2e_speedup"] = b["e2e_s"] / r["e2e_s"] if b else float("nan")

tag = "_".join(sys.argv[1:])
with open(f"{LOGS}/summary_{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

hdr = ("| L | arm | n | TTFT s | ×SDPA | attn s | attn share | ×SDPA | attn TFLOPS (util) | attn GB/s (util) | "
       "prefill TFLOPS (util) | prefill GB/s (util) | peak prefill GiB | act GiB | E2E s | TPOT ms | RULER |")
lines = [f"Peaks: bf16 GEMM {PEAK_TF:.0f} TFLOPS, bandwidth {PEAK_BW:.0f} GB/s ({peaks['device']})", "", hdr,
         "|" + "---|" * hdr.count("|")[:-1] if False else "|" + "---|" * (hdr.count("|") - 1)]
for r in out:
    lines.append(
        f"| {r['length']} | {r['arm']} | {r['n']} | {r['ttft_s']:.2f} | {r['ttft_speedup']:.2f} | {r['attn_s']:.2f} | "
        f"{100*r['attn_share']:.0f}% | {r['attn_speedup']:.2f} | {r['attn_tflops']:.0f} ({100*r['attn_tf_util']:.0f}%) | "
        f"{r['attn_gbs']:.0f} ({100*r['attn_bw_util']:.0f}%) | {r['prefill_tflops']:.1f} ({100*r['prefill_tf_util']:.0f}%) | "
        f"{r['prefill_gbs']:.0f} ({100*r['prefill_bw_util']:.0f}%) | {r['peak_prefill_gib']:.2f} | {r['act_prefill_gib']:.2f} | "
        f"{r['e2e_s']:.2f} | {r['tpot_ms']:.0f} | {r['score']:.1f} |")
md = "\n".join(lines)
open(f"{LOGS}/summary_{tag}.md", "w").write(md + "\n")
print(md)
