"""Measure every legal OpenCL candidate on the task workloads (full operation, interleaved medians) and write the
measured static ranking (static_priority_ocl.json: hardware class -> shape class -> best-first (SGM, SGN, PIPE)).

The sweep uses the runtime tuner itself: finalists = pool size, so every candidate gets a median. Run it again after
kernel or driver changes; the ranking is evidence for the static policy, not a constant.
Usage: python opencl/sweep_ocl.py [inputs.npz] [rounds]
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyopencl as cl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from autotune import SelectionSettings  # noqa: E402
import ocl_svdquant as L  # noqa: E402


def load_workloads(npz_path):
    d = np.load(npz_path)
    x = d["x"]
    specs = {
        "v": ("v_w_dpas_b", "v_w_scale", "v_lora_a", "v_lora_b", "v_bias"),
        "wide": ("wide_w_dpas_b", "wide_w_scale", "wide_lora_a", "wide_lora_b", "wide_bias"),
    }
    return d, x, specs


def main(npz_path, rounds):
    d, x, specs = load_workloads(npz_path)
    dev = L.intel_gpu()
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    results = {"device": dev.name, "driver": dev.driver_version, "hardware": L.hardware_class(dev), "shapes": {}}
    for name, (wb, ws, la, lb, bi) in specs.items():
        weights = L.Weights(ctx, d[wb], d[ws], d[la], d[lb], d[bi])
        layer = L.Layer(ctx, queue, weights, x)
        per_round = []
        for r in range(rounds):
            with tempfile.TemporaryDirectory(prefix="sweep-") as cache:
                op = L.SvdquantLinear(ctx, queue, cache_dir=Path(cache), rank_file=Path("/nonexistent"),
                                      settings=SelectionSettings(finalists=64))
                op(layer)                       # cold search over the full legal pool
                records = next(iter(op.tuned.timings.values()))
                per_round.append({L.cfg_key(rec["candidate"]): rec for rec in records})
                print(f"{name} round {r}: selected {op.selected}", flush=True)
        # per candidate: minimum over rounds of the median (fallback: screening time)
        ids = set(per_round[0])
        best = {}
        for cid in ids:
            meds = [pr[cid]["median_seconds"] for pr in per_round if pr.get(cid) and pr[cid]["median_seconds"]]
            scr = [pr[cid]["screen_seconds"] for pr in per_round if pr.get(cid) and pr[cid]["screen_seconds"]]
            best[cid] = min(meds) if meds else (min(scr) if scr else float("inf"))
        order = sorted(best, key=lambda c: best[c])
        results["shapes"][name] = {
            "N": int(weights.N), "K": int(weights.K), "shape_class": L.shape_class(weights.N, weights.K),
            "ms": {str(dict(c)): best[c] * 1e3 for c in order},
            "order": [[dict(c)["KF_SGM"], dict(c)["KF_SGN"], dict(c)["KF_PIPE"], dict(c)["KF_GRF"]] for c in order],
        }
        print(name, "best:", dict(order[0]), f"{best[order[0]]*1e3:.3f} ms", flush=True)
    out = HERE / "sweep_ocl_results.json"
    out.write_text(json.dumps(results, indent=2))
    write_ranking(results)
    print("wrote", out, "and static_priority_ocl.json")


def write_ranking(results):
    """Static priority file: hardware class -> shape class -> best-first [SGM, SGN, PIPE, GRF] from measured medians."""
    ranking = {results["hardware"]: {}}
    for s in results["shapes"].values():
        ranking[results["hardware"]][s["shape_class"]] = s["order"]
    (HERE / "static_priority_ocl.json").write_text(json.dumps(ranking, indent=2))


def ranking_from_results(path=None):
    """Regenerate the static priority file from a saved sweep (no re-measurement)."""
    results = json.loads((HERE / "sweep_ocl_results.json").read_text())
    for s in results["shapes"].values():
        s["order"] = [[c["KF_SGM"], c["KF_SGN"], c["KF_PIPE"], c["KF_GRF"]]
                      for c in (dict(eval(k)) for k in s["ms"])]
    write_ranking(results)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--ranking-only"]:
        ranking_from_results()
        sys.exit(0)
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ocl_probe/v_inputs2.npz", int(sys.argv[2]) if len(sys.argv) > 2 else 2)
