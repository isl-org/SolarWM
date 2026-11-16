"""Reusable OpenCL autotuning engine driven by metadata embedded in kernel source.

Each kernel provides an AUTOTUNE_METADATA block. Static dispatch owns metadata parsing,
parameter domains, hardware constraints, and legal pool construction
(``intel-gpu-kernel-dispatch/assets/dispatch_metadata.py``); this module compiles
candidates and delegates timing selection and cache bookkeeping to ``runtime_selection.py``.

Preferred API::

    kernel = AutoTunedKernel(source, queue, make_runner, workload_key,
                             compiler_options=("-cl-std=CL3.0",),
                             make_validator=make_validator,
                             make_reset=make_reset)
    kernel(*args)  # First call tunes/validates or loads a tuned selection.
    kernel(*args)  # Same workload bucket: dispatches the retained program directly.

Metadata is a version-1 comment block delimited by AUTOTUNE_METADATA_BEGIN/END:
``param NAME 8,16,32 default=16`` implies ``-DNAME=<selected>``. Fixed build flags
belong to compiler_options; ``option-if NAME=32 <flag>`` handles exceptions.
Constraint lines are ANDed, never evaluated as arbitrary Python expressions:
``le A B``, ``ge A B``, ``divisible A B``, and ``product_le A B C D LIMIT``.
Operands are parameter names, integer literals, or ``DEVICE_<LIMIT>`` tokens.
product_le checks exact positive-divisor quotients (A/B)*(C/D) <= LIMIT.

Lazy disk selections are scoped by workload and fixed compiler options, then by
source, platform, device, vendor and driver. Default-only entries do not suppress
first-call tuning. The wrapper holds strong references to programs per workload;
new instances use PyOpenCL's build/binary cache rather than sharing context-bound
program objects. PyOpenCL determines its own binary cache location. Our JSON files
live below its user cache root (normally ~/.cache/pyopencl/kernelfoundry-autotune).
PYOPENCL_CACHE_DIR overrides the *selection* root here; it is not a portable OpenCL
ICD binary-cache setting. PYOPENCL_NO_CACHE may disable PyOpenCL binary reuse.

Use tune_opencl for an explicit benchmark callback; use load_or_compile_default
to opt out of tuning. Candidate compilation/validation/timing failures are skipped;
if none succeeds, tuning raises and does not silently return an untested default.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

import platformdirs
import pyopencl as cl

# Static dispatch owns metadata, legality and the pool (intel-gpu-kernel-dispatch/assets/dispatch_metadata.py);
# the shared selector measures it. Copy both beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "intel-gpu-kernel-dispatch" / "assets"))
from dispatch_metadata import (
    Constraint,
    KernelMetadata,
    Parameter,
    check_candidate,
    constraint_is_valid as _constraint_is_valid,
    parse_metadata,
    static_pool,
)
from runtime_selection import (
    RuntimeSelection,
    SelectionSettings,
    SelectionStats,
    bucket_dim,
    eligible_pool,
    fnv1a64,
    select,
)

BEGIN_MARKER = "AUTOTUNE_METADATA_BEGIN"
END_MARKER = "AUTOTUNE_METADATA_END"


def default_cache_dir() -> Path:
    """Return the PyOpenCL user cache root, honoring XDG_CACHE_HOME on Linux."""
    override = os.environ.get("PYOPENCL_CACHE_DIR") or os.environ.get("KF_AUTOTUNE_CACHE_DIR")
    if override:
        return Path(override).expanduser() / "kernelfoundry-autotune"
    return Path(platformdirs.user_cache_dir("pyopencl", "pyopencl")) / "kernelfoundry-autotune"


def _cache_path(source: str, device: cl.Device, cache_dir: Path | None = None) -> Path:
    identity = "\0".join((source, device.platform.name, device.name, device.vendor, device.driver_version))
    return (cache_dir or default_cache_dir()) / f"{fnv1a64(identity)}.json"


def load_cached_parameters(
    source: str, device: cl.Device, cache_dir: Path | None = None, *, require_autotuned: bool = False
) -> dict[str, int] | None:
    path = _cache_path(source, device, cache_dir)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict) or metadata.get("source_hash") != fnv1a64(source):
        return None
    if require_autotuned and metadata.get("autotuned") is not True:
        return None
    try:
        values = dict(metadata["parameters"])
        schema = parse_metadata(source)
        check_candidate(schema, values, device)
        return values
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def save_cached_parameters(
    source: str,
    device: cl.Device,
    values: Mapping[str, int],
    *,
    cache_dir: Path | None = None,
    autotuned: bool,
    elapsed_seconds: float | None = None,
) -> Path:
    path = _cache_path(source, device, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_hash": fnv1a64(source),
        "platform": device.platform.name,
        "device": device.name,
        "vendor": device.vendor,
        "driver_version": device.driver_version,
        "autotuned": autotuned,
        "elapsed_seconds": elapsed_seconds,
        "parameters": dict(values),
    }
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)
    return path


def compile_candidate(
    context: cl.Context,
    source: str,
    metadata: KernelMetadata,
    values: Mapping[str, int],
    compiler_options: tuple[str, ...] = (),
) -> cl.Program:
    """Compile a candidate. PyOpenCL's built-in source cache stores/reuses its device binary."""
    return cl.Program(context, source).build(options=[*compiler_options, *metadata.build_options(values)])


def _resolve_pool(metadata, device, candidates, default):
    """Default + ordered eligible pool. Without a supplied pool, static dispatch's metadata pool is used."""
    if candidates is None:
        static_default, candidates = static_pool(metadata, device)
        default = static_default if default is None else default
    elif default is None:
        default = metadata.defaults if metadata.defaults in candidates else candidates[0]
    for values in (default, *candidates):
        check_candidate(metadata, values, device)  # a pool member outside the metadata is a dispatch bug
    pool = eligible_pool(default, candidates)
    return pool[0], pool


def tune_opencl(
    source: str,
    context: cl.Context,
    benchmark: Callable[[cl.Program, Mapping[str, int]], float],
    *,
    cache_dir: Path | None = None,
    candidates: list[dict[str, int]] | None = None,
    compiler_options: tuple[str, ...] = (),
    default: Mapping[str, int] | None = None,
    selection: SelectionSettings = SelectionSettings(),
    stats: SelectionStats | None = None,
    report: dict | None = None,
) -> tuple[cl.Program, dict[str, int], float]:
    """Screen the whole pool (default first), then compare alternating median batches (shared ``select``).

    ``stats`` counts the search, each build and each completed benchmark; ``report['timings']`` gets per-candidate seconds.
    """
    if not context.devices:
        raise ValueError("OpenCL context has no devices")
    device = context.devices[0]
    if not device.type & cl.device_type.GPU:
        raise ValueError("OpenCL autotuning requires an actual GPU context")
    metadata = parse_metadata(source)
    default, pool = _resolve_pool(metadata, device, candidates, default)

    def measure(program, values):
        seconds = benchmark(program, values)
        if stats is not None:
            stats.benchmarked_candidate(values)
        return seconds

    def build(values):
        if stats is not None:
            stats.builds += 1
        return compile_candidate(context, source, metadata, values, compiler_options)

    if stats is not None:
        stats.searches += 1
    best_program, best_values, best_seconds = select(
        pool,
        build,
        measure,
        selection,
        report,
    )
    save_cached_parameters(
        source, device, best_values, cache_dir=cache_dir, autotuned=True, elapsed_seconds=best_seconds
    )
    return best_program, best_values, best_seconds


def load_or_compile_default(
    source: str,
    context: cl.Context,
    *,
    cache_dir: Path | None = None,
    compiler_options: tuple[str, ...] = (),
) -> tuple[cl.Program, dict[str, int]]:
    """Load tuned parameters when available, else compile/cache metadata defaults."""
    if not context.devices:
        raise ValueError("OpenCL context has no devices")
    device = context.devices[0]
    metadata = parse_metadata(source)
    values = load_cached_parameters(source, device, cache_dir)
    autotuned = values is not None
    if values is None:
        values = metadata.defaults
    program = compile_candidate(context, source, metadata, values, compiler_options)
    if not autotuned:
        save_cached_parameters(source, device, values, cache_dir=cache_dir, autotuned=False)
    return program, values


class AutoTunedKernel:
    """Lazy, workload-aware callable retaining each winning program and runner.

    Construction does not compile. The first ``kernel(*args)`` loads a *tuned* disk
    selection or searches the metadata grid via ``RuntimeSelection``, then dispatches
    the winner. Later calls for that workload reuse the in-memory program without
    build/cache I/O. A new instance loads selected parameters and lets PyOpenCL reuse
    its binary cache.

    ``make_runner(program, queue, values)`` owns kernel arguments and launch geometry,
    deriving launch grids from the live call arguments rather than bucketed keys.
    ``workload_key(*args, **kwargs)`` returns a JSON-serializable identity containing
    bucketed dynamic dimensions (via ``bucket_dim``) and exact static dimensions,
    dtypes, layouts, and execution modes, not buffers.
    Optional ``make_validator(*args, **kwargs)`` prepares a zero-argument output check
    once per search or disk load.
    Optional ``make_reset(*args, **kwargs)`` prepares a zero-argument state-restoration
    callback invoked before validation and before each warmup/timed launch when kernels
    mutate inputs or accumulate into outputs in-place.
    """

    def __init__(
        self,
        source: str,
        queue: cl.CommandQueue,
        make_runner: Callable,
        workload_key: Callable,
        *,
        compiler_options: tuple[str, ...] = (),
        cache_dir: Path | None = None,
        candidates: list[dict[str, int]] | None = None,
        warmup_iters: int = 2,
        bench_iters: int = 5,
        make_validator: Callable | None = None,
        make_reset: Callable | None = None,
        policy: Callable[[Any], tuple[Mapping[str, int], list[dict[str, int]]]] | None = None,
        policy_version: str = "",
        selection: SelectionSettings = SelectionSettings(),
    ):
        self.source = source
        self.queue = queue
        self.make_runner = make_runner
        self.workload_key = workload_key
        self.compiler_options = tuple(compiler_options)
        self.cache_dir = cache_dir or default_cache_dir()
        self.candidates = candidates
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.make_validator = make_validator
        self.make_reset = make_reset
        self.policy = policy
        self.policy_version = policy_version
        self.selection = selection
        self.stats = SelectionStats()  # searches/benchmarks/validations actually performed
        self.timings: dict[str, list] = {}  # workload identity -> per-candidate timings of its last cold search
        self._selection_runtime: RuntimeSelection | None = None
        self._programs: dict[int, cl.Program] = {}
        self.hyperparameters: dict[str, int] | None = None
        self.program: cl.Program | None = None
        self._lock = threading.Lock()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        workload = self.workload_key(*args, **kwargs)
        with self._lock:
            if self._selection_runtime is None:
                device = self.queue.device
                if not device.type & cl.device_type.GPU:
                    raise ValueError("OpenCL autotuning requires an actual GPU queue")
                metadata = parse_metadata(self.source)

                def build(values: Mapping[str, int]) -> Callable:
                    program = compile_candidate(
                        self.queue.context, self.source, metadata, values, self.compiler_options
                    )
                    runner = self.make_runner(program, self.queue, values)
                    self._programs[id(runner)] = program
                    return runner

                self._selection_runtime = RuntimeSelection(
                    {
                        "source": self.source,
                        "device": {
                            "platform": device.platform.name,
                            "name": device.name,
                            "vendor": device.vendor,
                            "driver_version": device.driver_version,
                        },
                        "flags": self.compiler_options,
                        "policy_version": self.policy_version,
                    },
                    self.cache_dir,
                    build,
                    self.queue.finish,
                    warmup_iters=self.warmup_iters,
                    bench_iters=self.bench_iters,
                    selection=self.selection,
                    stats=self.stats,
                )
                self.timings = self._selection_runtime.timings

        def resolve_policy():
            device = self.queue.device
            metadata = parse_metadata(self.source)
            default, pool = self.policy(workload) if self.policy else (None, self.candidates)
            if self.policy is not None and (default is None or pool is None):
                raise ValueError("Policy must return an explicit default and eligible pool")
            return _resolve_pool(metadata, device, pool, default)

        def validator_factory(*call_args, **call_kwargs):
            validate = self.make_validator(*call_args, **call_kwargs)
            return lambda output: validate()

        runner, values = self._selection_runtime.resolve(
            workload,
            resolve_policy,
            args,
            kwargs,
            validator_factory if self.make_validator else None,
            make_reset=self.make_reset,
        )
        self.program = self._programs[id(runner)]
        self.hyperparameters = dict(values)
        return runner(*args, **kwargs)
