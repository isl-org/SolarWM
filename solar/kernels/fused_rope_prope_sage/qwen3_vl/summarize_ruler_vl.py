"""Summarise ruler_vl_<tag>.jsonl (bench_ruler_vl.py): per-task scores per arm, paired Sage - flash
difference with a 95% CI, and median per-prompt speedups (TTFT, prefill attention, E2E) and peak VRAM.

    python3 vl8b/summarize_ruler_vl.py rvl128k [rvl256k ...]
"""
import collections
import json
import math
import statistics as st
import sys

LOGS = "/home/gta/omni_bench/logs"


def main():
    rows = [json.loads(l) for t in sys.argv[1:] for l in open(f"{LOGS}/ruler_vl_{t}.jsonl")]
    for L in sorted({r["length"] for r in rows}):
        by = collections.defaultdict(dict)
        for r in rows:
            if r["length"] == L:
                by[(r["task"], r["index"])][r["arm"]] = r
        keys = [k for k in by if "flash" in by[k] and "sage" in by[k]]
        if not keys:
            continue
        print(f"\n## L={L}  ({len(keys)} paired prompts)\n")
        print("| task | n | flash | sage | sage - flash |")
        print("|---|---|---|---|---|")
        for t in sorted({k[0] for k in keys}):
            ks = [k for k in keys if k[0] == t]
            f = [by[k]["flash"]["score"] for k in ks]; s = [by[k]["sage"]["score"] for k in ks]
            print(f"| {t} | {len(ks)} | {100*st.mean(f):.1f} | {100*st.mean(s):.1f} | {100*(st.mean(s)-st.mean(f)):+.1f} |")
        d = [by[k]["sage"]["score"] - by[k]["flash"]["score"] for k in keys]
        ci = 100 * 1.96 * st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"| **all** | {len(keys)} | {100*st.mean(by[k]['flash']['score'] for k in keys):.1f} | "
              f"{100*st.mean(by[k]['sage']['score'] for k in keys):.1f} | {100*st.mean(d):+.1f} ± {ci:.1f} |\n")
        for name, f in (("TTFT", "ttft_s"), ("prefill attention", "attn_s"), ("E2E", "e2e_s")):
            fl = [by[k]["flash"][f] for k in keys]; sg = [by[k]["sage"][f] for k in keys]
            print(f"- {name}: flash {st.median(fl):.1f} s, sage {st.median(sg):.1f} s, "
                  f"median per-prompt speedup {st.median(a / b for a, b in zip(fl, sg)):.3f}x")
        for arm in ("flash", "sage"):
            print(f"- peak VRAM {arm}: allocated {max(by[k][arm]['peak_prefill_gib'] for k in keys):.2f} GiB, "
                  f"reserved {max(by[k][arm]['peak_reserved_gib'] for k in keys):.2f} GiB")


if __name__ == "__main__":
    main()
