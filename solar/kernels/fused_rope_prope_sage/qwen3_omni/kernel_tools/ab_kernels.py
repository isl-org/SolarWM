"""Interleaved A/B benchmark for Sage causal extension builds.

Each variant must be built with a unique extension name and stored either as a
shared library or in a directory containing exactly one ``*.so`` file.
"""
import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

H, HKV, D = 32, 4, 128
DEFAULT_LENGTHS = [2048, 8192, 16384, 32768, 63488]


def load_variant(spec):
    if "=" not in spec:
        raise ValueError(f"expected NAME=PATH, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if path.is_dir():
        libraries = sorted(path.glob("*.so"))
        if len(libraries) != 1:
            raise ValueError(f"{path} must contain exactly one .so, found {len(libraries)}")
        path = libraries[0]
    if not path.is_file() or path.suffix != ".so":
        raise ValueError(f"not a shared library: {path}")
    module_name = path.stem
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load extension {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return label, module


def make_inputs(length):
    generator = torch.Generator(device="xpu").manual_seed(0)
    q = torch.randn((1, length, H, D), device="xpu", dtype=torch.bfloat16, generator=generator)
    k = torch.randn((1, length, HKV, D), device="xpu", dtype=torch.bfloat16, generator=generator)
    v = torch.randn_like(k)
    return q, k, v


def measure(fn, iters):
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        result = fn()
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
        del result
    return statistics.median(samples)


def profile_stages(label, module, q, k, v, repeats):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        for _ in range(repeats):
            result = module.forward(q, k, v, is_causal=True)
            del result
        torch.xpu.synchronize()
    events = [event for event in prof.key_averages() if event.self_device_time_total > 0]
    total = sum(event.self_device_time_total for event in events)
    print(f"  stages {label} (mean of {repeats} calls):")
    for event in sorted(events, key=lambda item: item.self_device_time_total, reverse=True)[:12]:
        elapsed_ms = event.self_device_time_total / repeats / 1e3
        share = 100 * event.self_device_time_total / total if total else 0
        print(f"    {elapsed_ms:8.3f} ms {share:5.1f}% {event.key[:120]}")


def run_worker(spec, length, warmup, iters, stages):
    label, module = load_variant(spec)
    torch.xpu.set_per_process_memory_fraction(0.92)
    q, k, v = make_inputs(length)
    fn = lambda: module.forward(q, k, v, is_causal=True)
    for _ in range(warmup):
        result = fn()
        del result
    torch.xpu.synchronize()
    elapsed = measure(fn, iters)
    if stages:
        profile_stages(label, module, q, k, v, repeats=3)
    print(json.dumps({"label": label, "median_ms": elapsed}))


def run_isolated(spec, length, round_index, warmup, iters, stages):
    label = spec.split("=", 1)[0]
    guard_log = Path(__file__).resolve().parents[1] / "logs" / f"ab_{label}_{length}_r{round_index}.log"
    command = ["scripts/rss_guard.sh", "24", str(guard_log), "--", sys.executable,
               str(Path(__file__).resolve()), "--worker", spec,
               "--worker-length", str(length), "--worker-warmup", str(warmup),
               "--worker-iters", str(iters)]
    if stages:
        command.append("--worker-stages")
    completed = subprocess.run(command, check=True, text=True, capture_output=True,
                               env=os.environ.copy())
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    worker_output = guard_log.read_text() if guard_log.exists() else completed.stdout
    records = [line for line in worker_output.splitlines() if line.startswith("{")]
    if not records:
        raise RuntimeError(f"worker returned no measurement: {worker_output}\n{completed.stdout}")
    return json.loads(records[-1])["median_ms"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", metavar="NAME=PATH", help=argparse.SUPPRESS)
    parser.add_argument("--worker-length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-warmup", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--worker-iters", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--worker-stages", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variants", nargs="+", metavar="NAME=PATH",
                        help="variant labels and .so paths (or directories containing one .so)")
    parser.add_argument("--lens", type=int, nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5, help="timed calls per variant and round")
    parser.add_argument("--warmup", type=int, default=2, help="warmup calls per variant and length")
    parser.add_argument("--stages", action="store_true", help="profile per-op XPU time after timing")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.worker_length is None:
            parser.error("--worker-length is required in worker mode")
        run_worker(args.worker, args.worker_length, args.worker_warmup,
                   args.worker_iters, args.worker_stages)
        return
    if not args.variants:
        parser.error("--variants is required outside worker mode")
    if len(args.variants) < 2:
        parser.error("--variants requires at least two variants")
    if args.rounds < 1 or args.iters < 1 or args.warmup < 0:
        parser.error("rounds and iters must be positive; warmup cannot be negative")

    variants = []
    for spec in args.variants:
        if "=" not in spec:
            parser.error(f"expected NAME=PATH, got {spec!r}")
        variants.append((spec.split("=", 1)[0], spec.split("=", 1)[1]))
    torch.xpu.set_per_process_memory_fraction(0.92)
    results = {label: {} for label, _ in variants}
    print(f"device: {torch.xpu.get_device_name(0)} | rounds={args.rounds} iters={args.iters}")
    print("S       " + "  ".join(f"{label:>14}" for label, _ in variants) + "  ratios vs first")

    for length in args.lens:
        round_times = {label: [] for label, _ in variants}
        for round_index in range(args.rounds):
            order = variants[round_index % len(variants):] + variants[:round_index % len(variants)]
            for label, path in order:
                spec = f"{label}={path}"
                round_times[label].append(run_isolated(spec, length, round_index, args.warmup,
                                                        args.iters, args.stages))
        medians = {label: statistics.median(times) for label, times in round_times.items()}
        for label, median_ms in medians.items():
            results[label][str(length)] = {"rounds_ms": round_times[label], "median_ms": median_ms}
        baseline = medians[variants[0][0]]
        values = "  ".join(f"{medians[label]:14.3f}" for label, _ in variants)
        ratios = "  ".join(f"{label}={baseline / medians[label]:.3f}x" for label, _ in variants[1:])
        print(f"{length:<7} {values}  {ratios}", flush=True)

    output = {
        "device": torch.xpu.get_device_name(0),
        "torch": torch.__version__,
        "rounds": args.rounds,
        "iters": args.iters,
        "variants": {label: {"path": str(Path(path).resolve()), "results": results[label]}
                     for label, path in (item.split("=", 1) for item in args.variants)},
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()