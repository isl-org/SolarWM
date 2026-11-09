"""Merge RULER sweep files into one file with distinct arm names, for summarize_ruler.py.

bench_ruler.py names the Sage arm `sage` whatever kernel is built, so sweeps run with different
kernels must be relabelled before they are summarised together. Example (how
logs/ruler_sweep64k_final.jsonl was built):

    python3 scripts/merge_sweeps.py sweep64k_final sweep64k:sdpa=sdpa,sage=sage_int4 sweep64k_int8:sage=sage_int8
    ./env.sh python scripts/summarize_ruler.py sweep64k_final

Arguments: OUT_TAG, then TAG:FROM=TO[,FROM=TO...] per input (rows of other arms are dropped).
"""
import json
import sys

LOGS = "/home/gta/omni_bench/logs"


def main():
    out_tag, specs = sys.argv[1], sys.argv[2:]
    n = 0
    with open(f"{LOGS}/ruler_{out_tag}.jsonl", "w") as out:
        for spec in specs:
            tag, mapping = spec.split(":", 1)
            relabel = dict(pair.split("=", 1) for pair in mapping.split(","))
            for line in open(f"{LOGS}/ruler_{tag}.jsonl"):
                row = json.loads(line)
                if row["arm"] in relabel:
                    row["arm"], row["tag"] = relabel[row["arm"]], out_tag
                    out.write(json.dumps(row) + "\n")
                    n += 1
    print(f"wrote {n} rows to {LOGS}/ruler_{out_tag}.jsonl")


if __name__ == "__main__":
    main()
