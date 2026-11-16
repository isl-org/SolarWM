"""SVDQuant v-layer on OpenCL: static dispatch and runtime autotuning. PyOpenCL and numpy only (no torch).

Pieces (mirroring the SYCL port in kernel.sycl):
  StaticDispatch  legality (device limits + the compiled kernel's CL_KERNEL_WORK_GROUP_SIZE), ranking from measured
                  orders (static_priority_ocl.json, per hardware class and shape class), deterministic default.
  SvdquantLinear  runtime autotuning with the skill's OpenCL tuner (autotune.py / runtime_selection.py): every
                  eligible candidate is validated against the static default's output and timed as the full
                  operation; the winner is cached per exact workload (bucketed M, exact N, K) on disk and retained.
  Layer           device buffers for one call (x, workspaces, output) over shared Weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyopencl as cl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from autotune import AutoTunedKernel, SelectionSettings, compile_candidate, default_cache_dir  # noqa: E402
from dispatch_metadata import bucket_key, parse_metadata, rank_pool  # noqa: E402

SOURCE_PATH = HERE / "svdquant_w4a4_v_layer.cl"
SOURCE = SOURCE_PATH.read_text()
COMPILER_OPTIONS = ("-cl-std=CL3.0",)   # register mode is a metadata alternative (KF_GRF), not a fixed flag
POLICY_VERSION = "svdq-ocl-v1"
RANK_FILE = HERE / "static_priority_ocl.json"
R = 32
GROUP = 64
GEMM = "gemm_int4_lora_bias"


def intel_gpu(ctx_device=None):
    if ctx_device is not None:
        return ctx_device
    plat = next(p for p in cl.get_platforms() if "Intel" in p.name)
    return plat.get_devices(cl.device_type.GPU)[0]


def hardware_class(device) -> str:
    """Xe3-LPG class by compute-unit (XVE) count: 96 on Panther Lake 12Xe. Other counts get their own key."""
    return f"xve{device.max_compute_units}"


def shape_class(n: int, k: int) -> str:
    if k > 2 * n:
        return "deep_k"
    return "wide" if n > k else "square"


def cfg_key(values) -> tuple:
    return tuple(sorted(values.items()))


def gemm_geometry(values, M, N):
    """Global and local sizes of the GEMM for one candidate (sub-groups per work-group: SGM along M, SGN along N)."""
    sgm, sgn = values["KF_SGM"], values["KF_SGN"]
    wg_n = -(-(N // GROUP) // sgn)
    wg_m = -(-(-(-M // 8)) // sgm)
    return (wg_n * 16 * sgn, wg_m * sgm), (16 * sgn, sgm)


class Weights:
    """Device buffers of one projection (pre-packed by the model load: w_dpas_b, w_scale_dpas, lora, bias)."""

    def __init__(self, ctx, wb: np.ndarray, wscale: np.ndarray, lora_a: np.ndarray, lora_b: np.ndarray, bias: np.ndarray):
        flags = cl.mem_flags
        self.K = wscale.shape[0] * GROUP
        self.N = wscale.shape[1]
        self.wb = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(wb, np.uint32).reshape(-1))
        self.wscale = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(wscale, np.float32).reshape(-1))
        self.lora_a = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(lora_a, np.uint16))
        self.lora_b = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(lora_b, np.uint16))
        self.bias = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(bias, np.uint16))


class Layer:
    """One call: activation rows x (bf16 [M][K]) and output buffers over shared weights."""

    def __init__(self, ctx, queue, weights: Weights, x: np.ndarray):
        flags = cl.mem_flags
        self.queue, self.w = queue, weights
        self.M, self.K, self.N = x.shape[0], weights.K, weights.N
        assert x.shape[1] == self.K
        self.x = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(x, np.uint16))
        self.xq = cl.Buffer(ctx, flags.READ_WRITE, self.M * self.K // 2)
        self.xs = cl.Buffer(ctx, flags.READ_WRITE, (self.K // GROUP) * self.M * 4)
        self.lora_act = cl.Buffer(ctx, flags.READ_WRITE, self.M * R * 4)
        self.out = cl.Buffer(ctx, flags.WRITE_ONLY, self.M * self.N * 2)
        self.oracle = None   # host copy of the static default's output (validation reference)

    def read_out(self) -> np.ndarray:
        host = np.zeros((self.M, self.N), dtype=np.uint16)
        cl.enqueue_copy(self.queue, host, self.out).wait()
        return host


def workload_key(layer: Layer) -> dict:
    return bucket_key({"M": layer.M}, N=layer.N, K=layer.K, R=R, epilogue="plain_lora_bias", dtype="bf16",
                      x="bf16", x_int4=False)


def make_runner(program, queue, values):
    """Full operation for one candidate: quantize -> LoRA-A -> INT4 GEMM + LoRA-B + bias. Returns the output buffer."""
    k_quant = cl.Kernel(program, "quantize_x_group64")
    k_lora = cl.Kernel(program, "lora_a_dpas")
    k_gemm = cl.Kernel(program, GEMM)

    def run(layer: Layer):
        M, N, K = layer.M, layer.N, layer.K
        w = layer.w
        k_quant.set_args(layer.x, layer.xq, layer.xs, np.int32(M), np.int32(K))
        k_lora.set_args(layer.x, w.lora_a, layer.lora_act, np.int32(M), np.int32(K))
        k_gemm.set_args(layer.xq, w.wb, w.wscale, layer.xs, layer.lora_act, w.lora_b, w.bias, layer.out,
                        np.int32(M), np.int32(N), np.int32(K // GROUP))
        cl.enqueue_nd_range_kernel(queue, k_quant, (K // GROUP, M), None)
        cl.enqueue_nd_range_kernel(queue, k_lora, (16 * (-(-M // 8)), 1), (16, 1))
        g, l = gemm_geometry(values, M, N)
        cl.enqueue_nd_range_kernel(queue, k_gemm, g, l)
        return layer.out

    return run


def device_facts(device) -> dict:
    """Queried device facts with state and source (dispatch guidance: unavailable queries stay unknown)."""
    queries = {
        "name": ("name", "CL_DEVICE_NAME"),
        "driver": ("driver_version", "CL_DRIVER_VERSION"),
        "max_work_group_size": ("max_work_group_size", "CL_DEVICE_MAX_WORK_GROUP_SIZE"),
        "max_work_item_sizes": ("max_work_item_sizes", "CL_DEVICE_MAX_WORK_ITEM_SIZES"),
        "local_mem_size": ("local_mem_size", "CL_DEVICE_LOCAL_MEM_SIZE"),
        "compute_units": ("max_compute_units", "CL_DEVICE_MAX_COMPUTE_UNITS (XVE count on Xe)"),
        "extensions": ("extensions", "CL_DEVICE_EXTENSIONS"),
        "ip_version": (None, "CL_DEVICE_IP_VERSION_INTEL"),
    }
    facts = {}
    for key, (attr, source) in queries.items():
        try:
            value = getattr(device, attr) if attr else None
            if attr is None:
                facts[key] = {"value": None, "state": "unknown", "source": f"{source}: not exposed by PyOpenCL"}
            else:
                facts[key] = {"value": value, "state": "verified", "source": f"clGetDeviceInfo({source})"}
        except Exception as error:  # noqa: BLE001 - an unavailable query stays unknown, it is not inferred
            facts[key] = {"value": None, "state": "unknown", "source": f"{source}: {error}"}
    try:
        facts["sub_group_sizes"] = {"value": _query_sub_group_sizes(device), "state": "verified",
                                    "source": "clGetDeviceInfo(CL_DEVICE_SUB_GROUP_SIZES_INTEL) via libOpenCL (not exposed by PyOpenCL)"}
    except Exception as error:  # noqa: BLE001 - unknown stays unknown; the sub-group legality check then rejects
        facts["sub_group_sizes"] = {"value": [], "state": "unknown",
                                    "source": f"CL_DEVICE_SUB_GROUP_SIZES_INTEL: {error}"}
    return facts


def _query_sub_group_sizes(device) -> list:
    """CL_DEVICE_SUB_GROUP_SIZES_INTEL (0x4108) through the C library: PyOpenCL has no enum for it."""
    import ctypes
    lib = ctypes.CDLL("libOpenCL.so")
    lib.clGetDeviceInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_size_t)]
    buffer, size = (ctypes.c_size_t * 16)(), ctypes.c_size_t()
    status = lib.clGetDeviceInfo(ctypes.c_void_p(int(device.int_ptr)), 0x4108, ctypes.sizeof(buffer), buffer,
                                 ctypes.byref(size))
    if status != 0:
        raise RuntimeError(f"clGetDeviceInfo returned {status}")
    return [int(buffer[i]) for i in range(size.value // ctypes.sizeof(ctypes.c_size_t))]


class StaticDispatch:
    """Legal pool, default and ranking for one device, following intel-gpu-kernel-dispatch (opencl.md).

    Legality per candidate (each failure is recorded with its source):
      device          local size per axis and total against the queried device limits
      compiled kernel total against CL_KERNEL_WORK_GROUP_SIZE; compile-time work-group size; static local memory
      sub-group       sub-group size 16 offered by the device and the kernel's max sub-group size for the local size
      workload tails  N and K multiples of the 64-column quantization group; PIPE=1 needs an even group count
    Candidates that pass keep their order; ranking (measured, per hardware and shape class) only reorders them.
    Compiling a candidate is not runtime search, benchmarking or validation.
    """

    def __init__(self, ctx, device, rank_file: Path = RANK_FILE):
        self.ctx, self.device = ctx, device
        self.metadata = parse_metadata(SOURCE)
        self.facts = device_facts(device)
        self.hardware = hardware_class(device)
        self.rank = json.loads(rank_file.read_text()) if rank_file.exists() else {}
        self._programs: dict = {}
        self._kernels: dict = {}
        self._selections: dict = {}

    def program(self, values):
        key = cfg_key(values)
        if key not in self._programs:
            self._programs[key] = compile_candidate(self.ctx, SOURCE, self.metadata, values, COMPILER_OPTIONS)
        return self._programs[key]

    def kernel(self, values):
        key = cfg_key(values)
        if key not in self._kernels:
            self._kernels[key] = cl.Kernel(self.program(values), GEMM)
        return self._kernels[key]

    def check(self, values, n: int, k: int) -> list:
        """Return the list of failed checks (empty = legal) for one candidate and workload."""
        dev = self.device
        failures = []
        local = (16 * values["KF_SGN"], values["KF_SGM"], 1)
        threads = local[0] * local[1]
        if any(l > m for l, m in zip(local, dev.max_work_item_sizes)):
            failures.append(("local size exceeds a device per-axis limit", "CL_DEVICE_MAX_WORK_ITEM_SIZES"))
        if threads > dev.max_work_group_size:
            failures.append(("work-group exceeds the device limit", "CL_DEVICE_MAX_WORK_GROUP_SIZE"))
        kernel = self.kernel(values)
        limit = kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, dev)
        if threads > limit:
            failures.append((f"work-group exceeds the compiled kernel limit ({limit} work-items for this build)",
                             "CL_KERNEL_WORK_GROUP_SIZE of the compiled candidate"))
        required = tuple(kernel.get_work_group_info(cl.kernel_work_group_info.COMPILE_WORK_GROUP_SIZE, dev))
        if any(required) and required != local:
            failures.append((f"compiled work-group size {required} differs from the launch {local}",
                             "CL_KERNEL_COMPILE_WORK_GROUP_SIZE"))
        static_local = kernel.get_work_group_info(cl.kernel_work_group_info.LOCAL_MEM_SIZE, dev)
        if static_local > dev.local_mem_size:
            failures.append(("static local memory exceeds the device", "CL_KERNEL_LOCAL_MEM_SIZE"))
        if 16 not in self.facts["sub_group_sizes"]["value"]:
            failures.append(("sub-group size 16 is not offered by the device", "CL_DEVICE_SUB_GROUP_SIZES_INTEL"))
        else:
            try:
                sg = kernel.get_sub_group_info(dev, cl.kernel_sub_group_info.MAX_SUB_GROUP_SIZE_FOR_NDRANGE, local)
                if int(sg) != 16:
                    failures.append((f"kernel runs at sub-group size {sg}, the DPAS builtins need 16",
                                     "CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE"))
            except cl.Error as error:
                failures.append((f"sub-group size unknown: {error}", "CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE"))
        if n % GROUP:
            failures.append((f"N={n} is not a multiple of the 64-column group", "workload shape"))
        if k % GROUP:
            failures.append((f"K={k} is not a multiple of the 64-element group", "workload shape"))
        elif values["KF_PIPE"] and (k // GROUP) % 2:
            failures.append(("pipelined build needs an even group count", "workload shape"))
        return failures

    def select(self, workload: dict) -> dict:
        n, k = int(workload["N"]), int(workload["K"])
        cls = shape_class(n, k)
        cache_key = (self.hardware, cls, n, k)
        if cache_key in self._selections:
            return self._selections[cache_key]
        legal, rejected, decisions = [], [], []
        for values in self.metadata.candidates(self.device):
            name = dict(values)
            failures = self.check(values, n, k)
            if failures:
                why = "; ".join(reason for reason, _ in failures)
                source = "; ".join(src for _, src in failures)
                rejected.append(name)
                decisions.append({"candidate": name, "legal": False, "reason": why, "source": source})
            else:
                legal.append(name)
                decisions.append({"candidate": name, "legal": True,
                                  "reason": "within device, compiled-kernel, sub-group and tail limits",
                                  "source": "queried device facts and the compiled candidate's work-group limit"})
        order = [dict(zip(("KF_SGM", "KF_SGN", "KF_PIPE", "KF_GRF"), c))
                 for c in self.rank.get(self.hardware, {}).get(cls, [])]
        ranked = rank_pool(legal, order) if order else legal
        if not ranked:
            raise RuntimeError("no legal OpenCL GEMM candidate for this device and workload")
        # Measured order first: the default is the best measured legal candidate. The declared default is used only
        # when no measurement exists for this hardware and shape class (dispatch: first priority group is the default).
        default = ranked[0] if order else (self.metadata.defaults if self.metadata.defaults in ranked else ranked[0])
        pool = [default] + [c for c in ranked if c != default]
        sel = {"shape_class": cls, "hardware": self.hardware, "default": default, "pool": pool,
               "input": legal + rejected, "rejected": rejected, "decisions": decisions,
               "ranked_by": "measured order" if order else "declared default first",
               "facts": self.facts}
        self._selections[cache_key] = sel
        return sel


class SvdquantLinear:
    """The autotuned v-layer op. ``op(layer)`` returns the output buffer of the tuned full operation."""

    def __init__(self, ctx, queue, *, cache_dir=None, eligible=None, stats=None,
                 settings: SelectionSettings = SelectionSettings(), rank_file: Path = RANK_FILE):
        self.ctx, self.queue = ctx, queue
        self.dispatch = StaticDispatch(ctx, queue.device, rank_file)
        self.eligible = None if eligible is None else {tuple(sorted(e)) for e in eligible}
        self.tuned = AutoTunedKernel(
            SOURCE, queue, make_runner, workload_key,
            compiler_options=COMPILER_OPTIONS,
            cache_dir=cache_dir or default_cache_dir(),
            make_validator=self._make_validator,
            policy=self._policy,
            policy_version=POLICY_VERSION,
            selection=settings,
        )
        if stats is not None:
            self.tuned.stats = stats

    # --- hooks used by the tuner -------------------------------------------------------------------------
    def _policy(self, workload):
        sel = self.dispatch.select(workload)
        pool = sel["pool"]
        if self.eligible is not None:
            pool = [c for c in pool if cfg_key(c) in self.eligible]
            if not pool:
                raise RuntimeError("eligibility restriction removes every candidate")
        default = sel["default"] if sel["default"] in pool else pool[0]
        return default, pool

    def _make_validator(self, layer: Layer):
        def validate():
            got = layer.read_out()
            if not np.array_equal(got, layer.oracle):
                raise AssertionError("candidate output differs from the static default's output")
        return validate

    # --- public API ---------------------------------------------------------------------------------------
    def reference(self, layer: Layer) -> np.ndarray:
        """Static default's output (the oracle for validation). Built and run outside the tuner."""
        default = self.dispatch.select(workload_key(layer))["default"]
        run = make_runner(self.dispatch.program(default), self.queue, default)
        run(layer)
        self.queue.finish()
        return layer.read_out()

    def __call__(self, layer: Layer):
        if layer.oracle is None:
            layer.oracle = self.reference(layer)
        return self.tuned(layer)

    @property
    def selected(self):
        return dict(self.tuned.hyperparameters) if self.tuned.hyperparameters else None
