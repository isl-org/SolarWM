"""Validate data/ruler_{L}.jsonl: count, schema, token windows, gold presence, disk usage.

python validate.py [--recount]   # --recount re-tokenizes every input with apply_chat_template
"""
import argparse
import json
import math
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LENGTHS = [2048, 8192, 16384, 32768, 63488]
TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
         "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]
FIELDS = ["task", "index", "length_bucket", "input", "n_tokens_templated", "outputs", "max_new_tokens",
          "answer_prefix"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recount", action="store_true")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data"))
    args = ap.parse_args()
    tok = None
    if args.recount:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound")

    stats = defaultdict(list)
    bad, total, gold_missing = [], 0, []
    niah_examples = []
    for L in LENGTHS:
        path = os.path.join(args.data_dir, f"ruler_{L}.jsonl")
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                total += 1
                for k in FIELDS:
                    assert k in r, (path, k)
                assert r["length_bucket"] == L
                n = r["n_tokens_templated"]
                if tok is not None:
                    enc = tok.apply_chat_template([{"role": "user", "content": r["input"]}], tokenize=True,
                                                  add_generation_prompt=True)
                    n2 = len(enc["input_ids"] if hasattr(enc, "keys") else enc)
                    if n2 != n:
                        bad.append((r["task"], L, r["index"], f"recount {n2} != {n}"))
                    n = n2
                if not (math.floor(0.97 * L) <= n <= L):
                    bad.append((r["task"], L, r["index"], n))
                stats[(r["task"], L)].append(n)
                if r["task"].startswith(("niah", "vt", "cwe", "fwe")):
                    # every gold answer must appear verbatim in the input (fwe/cwe: as words)
                    miss = [o for o in r["outputs"] if o not in r["input"]]
                    if miss:
                        gold_missing.append((r["task"], L, r["index"], miss))
                if r["task"].startswith("qa"):
                    if not any(o.lower() in r["input"].lower() for o in r["outputs"]):
                        gold_missing.append((r["task"], L, r["index"], r["outputs"]))
                if r["task"].startswith("niah"):
                    niah_examples.append(r)

    print(f"{'task':16s}" + "".join(f"{L:>22}" for L in LENGTHS))
    print(f"{'':16s}" + "".join(f"{'min/mean/max (n)':>22}" for _ in LENGTHS))
    for t in TASKS:
        cells = []
        for L in LENGTHS:
            v = stats.get((t, L), [])
            cells.append(f"{min(v)}/{sum(v)/len(v):.0f}/{max(v)} ({len(v)})" if v else "-")
        print(f"{t:16s}" + "".join(f"{c:>22}" for c in cells))
    print(f"\nwindow per L: " + ", ".join(f"{L}: [{math.floor(0.97*L)}, {L}]" for L in LENGTHS))
    print(f"total samples: {total}; out of window: {len(bad)}; gold-not-in-input: {len(gold_missing)}")
    for b in bad[:20]:
        print("  OUT OF RANGE", b)
    for g in gold_missing[:20]:
        print("  GOLD MISSING", g)

    print("\nspot-check niah needles:")
    rng = random.Random(0)
    for r in rng.sample(niah_examples, 6):
        o = r["outputs"][0]
        i = r["input"].find(o)
        s = r["input"][max(0, i - 70): i + len(o) + 5].replace("\n", " ")
        print(f"  {r['task']:16s} L={r['length_bucket']:5d} idx={r['index']:2d} gold={r['outputs']} "
              f"pos={i / len(r['input']):.2f} ...{s}")


if __name__ == "__main__":
    main()
