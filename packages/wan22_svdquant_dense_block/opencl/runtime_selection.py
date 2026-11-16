"""Copy beside a backend's autotune.py; no GPU or compiler emulation lives here."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
import sys

# Shape regimes belong to static dispatch: one canonical `bucket_dim` (re-exported here for the autotune wrappers).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "intel-gpu-kernel-dispatch" / "assets"))
from dispatch_metadata import bucket_dim  # noqa: E402,F401


@dataclass(frozen=True)
class SelectionSettings:
    finalists: int = 3
    batches: int = 5
    relative_noise: float = 0.02
    slow_factor: float = 3.0

    def __post_init__(self):
        if type(self.finalists) is not int or self.finalists < 2:
            raise ValueError("finalists must be an integer >= 2")
        if type(self.batches) is not int or self.batches < 3:
            raise ValueError("batches must be an integer >= 3")
        if not math.isfinite(self.relative_noise) or not 0 <= self.relative_noise < 1:
            raise ValueError("relative_noise must be finite and in [0, 1)")
        if not math.isfinite(self.slow_factor) or self.slow_factor <= 1:
            raise ValueError("slow_factor must be finite and > 1")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fnv1a64(value: str) -> str:
    result = 0xCBF29CE484222325
    for byte in value.encode("utf-8"):
        result = ((result ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{result:016x}"


def candidate_id(values):
    """Canonical candidate identity (what the evaluators compare): sorted items of the integer config.

    Pool entries are flat integer configs, or ``{"name": ..., "parameters": {integer config}}`` wrappers.
    """
    values = dict(values)
    return tuple(sorted(dict(values["parameters"] if "parameters" in values else values).items()))


class SelectionStats:
    """Work actually done by a selector: counted at the real search/validate/benchmark boundaries.

    ``snapshot()`` is exactly the counter dictionary the evaluators read, so adapters need no instrumentation.
    ``builds`` counts calls to the backend's build hook (a compile for OpenCL, Triton and Vulkan; a runner lookup for
    the ahead-of-time SYCL wrapper, which does not compile at run time).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.builds = self.searches = self.benchmarks = self.validations = 0
        self.benchmarked, self.validated = set(), set()

    def validated_candidate(self, values):
        self.validations += 1
        self.validated.add(candidate_id(values))

    def benchmarked_candidate(self, values):
        self.benchmarks += 1
        self.benchmarked.add(candidate_id(values))

    def snapshot(self):
        return {
            "builds": self.builds,
            "searches": self.searches,
            "benchmarks": self.benchmarks,
            "validations": self.validations,
            "benchmarked": set(self.benchmarked),
            "validated": set(self.validated),
        }


def eligible_pool(default, candidates):
    pool = []
    for candidate in candidates:
        values = dict(candidate)
        if values not in pool:
            pool.append(values)
    default = dict(default)
    if default not in pool:
        raise ValueError("Policy default must belong to the eligible pool")
    return [default] + [values for values in pool if values != default]


def select(pool, build, measure, settings=SelectionSettings(), report=None):
    """Screen the whole pool, then re-time finalists in alternating batches (incumbent wins within the noise margin).

    ``report`` (optional dict) receives ``timings``: one record per pool member, in pool order, with the
    screening seconds and, for finalists, the median seconds. Set ``finalists`` to ``len(pool)`` for a full sweep.
    """
    observations = []

    def finite_measure(handle, values):
        elapsed = float(measure(handle, values))
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Benchmark must return finite positive seconds")
        return elapsed

    for values in pool:
        try:
            handle = build(values)
            elapsed = finite_measure(handle, values)
            observations.append((elapsed, values, handle))
        except Exception as exc:
            logging.debug("Candidate %s failed: %s", values, exc)
    if not observations:
        raise RuntimeError("Every eligible GPU candidate failed")
    incumbent = observations[0]
    finalists = sorted(observations, key=lambda item: item[0])[: settings.finalists]
    if incumbent not in finalists:
        finalists[-1] = incumbent
    samples = {id(item): [] for item in finalists}
    failed = set()
    for batch in range(settings.batches):
        for item in finalists if batch % 2 == 0 else reversed(finalists):
            if id(item) in failed:
                continue
            try:
                samples[id(item)].append(finite_measure(item[2], item[1]))
            except Exception as exc:
                failed.add(id(item))
                logging.debug("Finalist %s failed: %s", item[1], exc)
    ranked = [(median(samples[id(item)]), item) for item in finalists if id(item) not in failed]
    if not ranked:
        raise RuntimeError("Every GPU finalist failed")
    if report is not None:
        screen = {id(values): seconds for seconds, values, _ in observations}
        median_of = {id(item[1]): duration for duration, item in ranked}
        report["timings"] = [
            {
                "candidate": dict(values),
                "screen_seconds": screen.get(id(values)),
                "median_seconds": median_of.get(id(values)),
            }
            for values in pool
        ]
    elapsed, winner = min(ranked, key=lambda item: item[0])
    for duration, item in ranked:
        if item is incumbent and duration <= elapsed * (1 + settings.relative_noise):
            elapsed, winner = duration, item
            break
    return winner[2], winner[1], elapsed


def gpu_identity():
    import torch

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("An actual XPU GPU is required; supply explicit GPU device identity for another backend")
    props = torch.xpu.get_device_properties(torch.xpu.current_device())
    driver = getattr(props, "driver_version", None)
    if driver is None:
        raise RuntimeError("Supply device_identity with actual device and driver_version")
    return {"name": props.name, "driver_version": str(driver)}


class RuntimeSelection:
    def __init__(
        self,
        identity,
        cache_dir,
        build,
        sync,
        *,
        warmup_iters=2,
        bench_iters=5,
        selection=SelectionSettings(),
        stats=None,
    ):
        if warmup_iters < 1 or bench_iters < 1:
            raise ValueError("Iteration counts must be positive")
        self.identity = identity
        self.cache_dir = Path(cache_dir)
        self._build = build
        self.sync = sync
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.selection = selection
        self.stats = stats if stats is not None else SelectionStats()
        self.build = self._counted_build   # every call site (disk hit, cold search) goes through the counter
        self.timings = {}  # canonical workload -> per-candidate timings of its last cold search
        self.retained = {}
        self.lock = threading.Lock()

    def _counted_build(self, values):
        self.stats.builds += 1
        return self._build(values)

    def resolve(self, workload, resolve_policy, args, kwargs, make_validator, make_reset=None):
        key = canonical(workload)
        with self.lock:
            if key in self.retained:
                return self.retained[key]
            default, candidates = resolve_policy()
            pool = eligible_pool(default, candidates)
            identity = canonical(
                {
                    "runtime": self.identity,
                    "workload": workload,
                    "pool": pool,
                    "default": default,
                    "selection": vars(self.selection),
                    "protocol": 3,
                }
            )
            path = self.cache_dir / (hashlib.sha256(identity.encode()).hexdigest() + ".json")
            validate = make_validator(*args, **kwargs) if make_validator else None
            reset = make_reset(*args, **kwargs) if make_reset else None

            def check(runner, values):
                if reset is not None:
                    reset()
                self.sync()
                output = runner(*args, **kwargs)
                self.sync()
                if validate is not None:
                    validate(output)
                self.stats.validated_candidate(values)

            winner = None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                values = payload["parameters"]
                if payload.get("identity") == identity and values in pool:
                    runner = self.build(values)
                    check(runner, values)
                    winner = (runner, values)
            except Exception as exc:
                logging.debug("Loaded GPU selection rejected: %s", exc)
                pass
            if winner is None:
                incumbent_time = [None]

                def measure(runner, values):
                    if reset is not None:
                        reset()
                    self.sync()
                    output = runner(*args, **kwargs)
                    self.sync()
                    if validate is not None:
                        validate(output)
                    self.stats.validated_candidate(values)
                    for _ in range(self.warmup_iters - 1):
                        if reset is not None:
                            reset()
                        runner(*args, **kwargs)
                    self.sync()
                    if reset is None:
                        start = time.perf_counter()
                        for _ in range(self.bench_iters):
                            runner(*args, **kwargs)
                        self.sync()
                        elapsed = (time.perf_counter() - start) / self.bench_iters
                    else:
                        total = 0.0
                        for _ in range(self.bench_iters):
                            reset()
                            self.sync()
                            start = time.perf_counter()
                            runner(*args, **kwargs)
                            self.sync()
                            total += time.perf_counter() - start
                        elapsed = total / self.bench_iters
                    self.stats.benchmarked_candidate(values)
                    if incumbent_time[0] is not None and elapsed > self.selection.slow_factor * incumbent_time[0]:
                        raise RuntimeError("Slow candidate rejected after completed synchronized batch")
                    if math.isfinite(elapsed) and elapsed > 0 and incumbent_time[0] is None:
                        incumbent_time[0] = elapsed
                    return elapsed

                self.stats.searches += 1
                report = {}
                runner, values, elapsed = select(pool, self.build, measure, self.selection, report)
                self.timings[key] = report.get("timings", [])
                winner = (runner, values)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps({"identity": identity, "parameters": values, "elapsed_seconds": elapsed}),
                    encoding="utf-8",
                )
                temporary.replace(path)
            if reset is not None:
                reset()
                self.sync()
            self.retained[key] = winner
            return winner
