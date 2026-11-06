"""RULER scorer (port of third_party/RULER/scripts/eval/{evaluate.py,synthetic/constants.py}).

Metrics, per sample, as a fraction in [0, 1]:
  string_match_all  (niah_*, vt, cwe, fwe): share of gold strings found in the prediction
                                            (case-insensitive substring match)
  string_match_part (qa_*):                 1 if any gold string is in the prediction, else 0
RULER reports the mean over samples x 100. The CLI below does the same.
The prediction is cleaned first, as RULER's postprocess_pred does: strip, then replace
control characters [\\x00-\\x1f] with '\\n'.

Library use:
  from score import score
  score("niah_single_1", "The number is 1234567.", ["1234567"])  # -> 1.0

CLI:
  python score.py preds.jsonl [more.jsonl ...] [--pred_field pred] [--data_dir data] [--out summary.json]
Each prediction line needs `task`, `pred` (or --pred_field), and either `outputs` or
(`index`, `length_bucket`) to look the gold answers up in data/ruler_{L}.jsonl.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

TASK_METRIC = {
    "niah_single_1": "all", "niah_single_2": "all", "niah_single_3": "all",
    "niah_multikey_1": "all", "niah_multikey_2": "all", "niah_multikey_3": "all",
    "niah_multivalue": "all", "niah_multiquery": "all",
    "vt": "all", "cwe": "all", "fwe": "all",
    "qa_1": "part", "qa_2": "part",
}
_NP = re.compile(r"[\x00-\x1f]")


def postprocess_pred(pred: str) -> str:
    pred = (pred or "").strip()
    return _NP.sub("\n", pred).strip()


def string_match_all(pred: str, refs) -> float:
    p = pred.lower()
    return sum(1.0 if r.lower() in p else 0.0 for r in refs) / len(refs)


def string_match_part(pred: str, refs) -> float:
    p = pred.lower()
    return max(1.0 if r.lower() in p else 0.0 for r in refs)


def metric_for(task: str) -> str:
    if task in TASK_METRIC:
        return TASK_METRIC[task]
    if task.startswith("qa"):
        return "part"
    return "all"


def score(task: str, prediction: str, outputs) -> float:
    """Per-sample RULER score in [0, 1]."""
    if isinstance(outputs, str):
        outputs = [outputs]
    pred = postprocess_pred(prediction)
    fn = string_match_part if metric_for(task) == "part" else string_match_all
    return fn(pred, outputs)


def _load_gold(data_dir, L, cache):
    if L not in cache:
        path = os.path.join(data_dir, f"ruler_{L}.jsonl")
        cache[L] = {}
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                cache[L][(r["task"], r["index"])] = r["outputs"]
    return cache[L]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--pred_field", default="pred")
    ap.add_argument("--data_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--out", help="write summary JSON here")
    args = ap.parse_args()

    acc = defaultdict(list)  # (task, L) -> [scores]
    gold_cache = {}
    for path in args.preds:
        with open(path) as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                task, L = r["task"], r.get("length_bucket")
                outputs = r.get("outputs")
                if outputs is None:
                    outputs = _load_gold(args.data_dir, L, gold_cache)[(task, r["index"])]
                pred = r.get(args.pred_field)
                if pred is None:
                    print(f"warning: {path}:{ln} has no '{args.pred_field}', scored 0", file=sys.stderr)
                    pred = ""
                acc[(task, L)].append(score(task, pred, outputs))

    tasks = sorted({t for t, _ in acc}, key=lambda t: (list(TASK_METRIC).index(t) if t in TASK_METRIC else 99, t))
    lengths = sorted({L for _, L in acc}, key=lambda x: (x is None, x))
    table = {t: {L: round(100 * sum(v) / len(v), 2) for (tt, L), v in acc.items() if tt == t} for t in tasks}
    counts = {f"{t}|{L}": len(v) for (t, L), v in acc.items()}
    avg = {}
    for L in lengths:
        vals = [table[t][L] for t in tasks if L in table[t]]
        avg[L] = round(sum(vals) / len(vals), 2) if vals else None

    w = max(16, max(len(t) for t in tasks) + 1)
    print("task".ljust(w) + "".join(f"{str(L):>9}" for L in lengths))
    for t in tasks:
        print(t.ljust(w) + "".join(f"{table[t].get(L, float('nan')):9.2f}" for L in lengths))
    print("avg (task mean)".ljust(w) + "".join(f"{avg[L]:9.2f}" for L in lengths))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"per_task_length": {t: {str(L): s for L, s in d.items()} for t, d in table.items()},
                       "avg_per_length": {str(L): s for L, s in avg.items()},
                       "n_samples": counts}, f, indent=1)


if __name__ == "__main__":
    main()
