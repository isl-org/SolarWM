"""Compare VTune ``hotspots.csv`` exports (from run_wan22_stage2_vtune.sh) by kernel name.

Usage: python vtune_hotspots_compare.py RESULT_DIR_A RESULT_DIR_B [--top N]
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import re
from pathlib import Path


def load(result_dir: Path) -> dict[str, list[float]]:
    text = (result_dir / "hotspots.csv").read_text()
    text = text[text.index("Computing Task\tWork Size") :]
    agg: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0, 0.0, 0.0])
    for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
        try:
            seconds = float(row["Computing Task:Total Time"])
        except (TypeError, ValueError):
            continue
        name = re.sub(r"<.*", "", row["Computing Task"]).replace("at::native::xpu::", "")[:48]
        entry = agg[name]
        entry[0] += seconds
        entry[1] += int(row["Computing Task:Instance Count"] or 0)
        entry[2] += seconds * float(row["XVE Pipelines:XMX active(%)"] or 0)
        entry[3] += seconds * (
            float(row["GPU Memory Bandwidth, GB/sec:Read"] or 0)
            + float(row["GPU Memory Bandwidth, GB/sec:Write"] or 0)
        )
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    a, b = load(args.a), load(args.b)
    ta, tb = sum(v[0] for v in a.values()), sum(v[0] for v in b.values())
    print(f"total GPU kernel time: A {ta * 1e3:.1f} ms | B {tb * 1e3:.1f} ms")
    print(f"{'kernel':48s} {'A ms':>8} {'n':>5} {'XMX%':>5} {'GB/s':>5} | {'B ms':>8} {'n':>5} {'XMX%':>5} {'GB/s':>5} | {'B-A':>7}")
    keys = sorted(set(a) | set(b), key=lambda k: -max(a.get(k, [0])[0], b.get(k, [0])[0]))
    for key in keys[: args.top]:
        cells = []
        for agg in (a, b):
            t, n, xmx, bw = agg.get(key, [0.0, 0, 0.0, 0.0])
            cells.append(
                f"{t * 1e3:8.1f} {n:5d} {xmx / t if t else 0:5.1f} {bw / t if t else 0:5.0f}"
            )
        delta = (b.get(key, [0])[0] - a.get(key, [0])[0]) * 1e3
        print(f"{key:48s} {cells[0]} | {cells[1]} | {delta:+7.1f}")


if __name__ == "__main__":
    main()
