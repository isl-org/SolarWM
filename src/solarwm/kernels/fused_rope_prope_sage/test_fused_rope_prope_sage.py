"""Accuracy tests and benchmark for the fused RoPE + PRoPE + attention kernels.

    pytest test_fused_rope_prope_sage.py              # accuracy tests + benchmark
    pytest test_fused_rope_prope_sage.py -s -k bench  # benchmark only, with its table
    python test_fused_rope_prope_sage.py              # the benchmark table alone

Three kernels, three roles:

* ``unfused``   -- :func:`fused_rope_prope_sdpa_reference`, the original
  stage-by-stage PyTorch pipeline. The oracle.
* ``reference`` -- :func:`fused_rope_prope_sdpa_split`, fused RoPE + PRoPE +
  SDPA + output PRoPE. Bit-exact against the oracle, so those tests assert zero
  differing elements rather than a tolerance. This is what new work is measured
  against, for speed and accuracy.
* ``sage``      -- :func:`fused_rope_prope_sage`, the same fusion around an int8
  SageAttention core. Accurate rather than exact; the gates below say how much.
  It is measured against the reference kernel, which shares every stage but
  the attention, so the bound is on the attention core alone.

Requires an Intel XPU (developed on Arc B70, torch 2.14.1+xpu, Triton 3.8).
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import pytest
import torch

# this folder is the package; make it importable when run as a script too
PARENT = str(Path(__file__).resolve().parent.parent)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from fused_rope_prope_sage import fused_sdpa as fk
from fused_rope_prope_sage import fused_sage
from fused_rope_prope_sage.fused_sdpa import fused_rope_prope_sdpa_split
from fused_rope_prope_sage.fused_sage import fused_rope_prope_sage, hadamard4
from fused_rope_prope_sage.unfused_kernel import (
    apply_prope,
    apply_rope,
    build_wan22_freqs,
    fused_rope_prope_sdpa_reference,
    normalize_rope_precision,
    prepare_prope_matrices,
)


pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Intel XPU is required",
)

ROPE_DTYPES = ("float64", "float32", "float16")


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────

# The production Wan2.2 Stage2 chunk: Q [1, 3*15*27 = 1215, 24, 128] and
# K/V [1, 18*15*27 = 7290, 24, 128], bfloat16, 3 query and 18 KV cameras.
GEOMETRY = dict(batch=1, num_heads=24, head_dim=128, q_frames=3, kv_frames=18,
                h_patches=15, w_patches=27)


def create_test_inputs(
    device,
    rope_dtype: str = "float64",
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
    **overrides,
) -> dict:
    """Random Q/K/V with realistic cameras, in the production geometry by default.

    Any key of ``GEOMETRY`` can be overridden. Returns every keyword argument
    of the three kernels, plus ``q``, ``k`` and ``v``.
    """
    g = {**GEOMETRY, **overrides}
    batch, frames_q, frames_kv = g["batch"], g["q_frames"], g["kv_frames"]
    tokens = g["h_patches"] * g["w_patches"]
    torch.manual_seed(seed)
    shape_q = (batch, frames_q * tokens, g["num_heads"], g["head_dim"])
    shape_kv = (batch, frames_kv * tokens, g["num_heads"], g["head_dim"])
    q = torch.randn(shape_q, dtype=dtype, device=device)
    k = torch.randn(shape_kv, dtype=dtype, device=device)
    v = torch.randn(shape_kv, dtype=dtype, device=device)

    grid = lambda f: torch.tensor([[f, g["h_patches"], g["w_patches"]]],
                                  dtype=torch.long, device=device)
    _, complex_dtype = normalize_rope_precision(rope_dtype)
    freqs = build_wan22_freqs(head_dim=g["head_dim"], max_len=1024,
                              complex_dtype=complex_dtype).to(device)

    def cameras(n, x_shift):
        # identity view matrices with a small translation, normalised intrinsics
        viewmats = torch.eye(4, device=device).repeat(batch, n, 1, 1)
        viewmats[..., 0, 3] = torch.linspace(*x_shift, n, device=device)
        Ks = torch.eye(3, device=device).repeat(batch, n, 1, 1)
        Ks[..., 0, 0] = 500.0 / 864.0
        Ks[..., 1, 1] = 500.0 / 480.0
        return viewmats, Ks

    q_viewmats, q_Ks = cameras(frames_q, (0.1, 0.3))
    kv_viewmats, kv_Ks = cameras(frames_kv, (0.0, 0.3))
    return dict(
        q=q, k=k, v=v, freqs=freqs,
        q_grid_sizes=grid(frames_q), k_grid_sizes=grid(frames_kv),
        q_viewmats=q_viewmats, q_Ks=q_Ks, kv_viewmats=kv_viewmats, kv_Ks=kv_Ks,
        q_start_frame=frames_kv - frames_q, k_start_frame=0,
        rope_dtype=rope_dtype, camera_translation_transform="linear",
    )


def _kwargs(inputs):
    return {k: v for k, v in inputs.items() if k not in {"q", "k", "v"}}


def invoke(kernel, inputs, **extra):
    return kernel(inputs["q"], inputs["k"], inputs["v"], **_kwargs(inputs), **extra)


def float64_truth(inputs):
    """The whole pipeline in float64: an absolute yardstick, below the reference."""
    kw = _kwargs(inputs)
    q = apply_rope(inputs["q"], kw["q_grid_sizes"], kw["freqs"], kw["q_start_frame"],
                   "float64").double()
    k = apply_rope(inputs["k"], kw["k_grid_sizes"], kw["freqs"], kw["k_start_frame"],
                   "float64").double()
    v = inputs["v"].double()
    kv_viewmats, kv_Ks = kw["kv_viewmats"], kw["kv_Ks"]
    if kv_viewmats is None:
        kv_viewmats, kv_Ks = kw["q_viewmats"], kw["q_Ks"]
    as64 = lambda t: None if t is None else t.double()
    p_q_t, _, p_q = prepare_prope_matrices(as64(kw["q_viewmats"]), as64(kw["q_Ks"]), q.shape[1])
    _, p_kv, _ = prepare_prope_matrices(as64(kv_viewmats), as64(kv_Ks), k.shape[1])
    qp = apply_prope(q, p_q_t).transpose(1, 2)
    kp = apply_prope(k, p_kv).transpose(1, 2)
    vp = apply_prope(v, p_kv).transpose(1, 2)
    s = (qp @ kp.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    # softmax spelled out: torch.softmax in float64 on XPU (torch 2.14.1) is
    # wrong for some row lengths (810, 1620, 2430 -- right at 7290)
    p = (s - s.amax(dim=-1, keepdim=True)).exp()
    p = p / p.sum(dim=-1, keepdim=True)
    return apply_prope((p @ vp).transpose(1, 2), p_q)


# ─────────────────────────────────────────────────────────────────────────────
# Comparisons
# ─────────────────────────────────────────────────────────────────────────────

def rel_rms(actual, reference):
    """RMS of the difference, relative to the reference's own RMS."""
    reference = reference.double()
    return ((actual.double() - reference).pow(2).mean().sqrt()
            / reference.pow(2).mean().sqrt()).item()


def assert_bit_exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    differ = (actual != expected) & ~(actual.isnan() & expected.isnan())
    n = int(differ.sum())
    assert n == 0, f"{n} / {actual.numel()} elements differ"


# Accuracy gates for the int8 kernel, against the reference kernel.
#
# These are regression tripwires, not a model-level correctness criterion (that
# has to come from video PSNR). They are set from what int8 Q/K can actually
# produce: sweeping the K-scale granularity from per-token to one scale per
# head moves the error only between 1.28% and 1.68%, because the floor is int8
# itself and not the scaling. Measured today: ~1.5%.
#
# So anything past ~1.8% is not a quantiser tuning choice -- it is a bug or a
# deliberate regime change (quantising V, a narrower format). For reference,
# the bugs this caught while the kernel was being written read 17.9%, 26.7%
# and 69%, so the gate has plenty of room.
SAGE_MAX_RMS = 0.018          # global
SAGE_MAX_HEAD_RMS = 0.025     # per head, so one bad head cannot hide in the mean
SAGE_MIN_COSINE = 0.9995      # direction, which RMS alone does not pin
# The two below protect the property that makes the RMS gate meaningful at all:
# the error is zero-mean noise, so it grows as sqrt(N) over a chunk's 150
# attentions instead of accumulating. A biased quantiser (round-toward-zero
# instead of round-half-away, say) can barely move the RMS and still compound.
SAGE_MAX_GAIN = 0.005         # error proportional to the signal
SAGE_MAX_CHANNEL_BIAS = 0.001  # error surviving an average over tokens and heads


def assert_sage_gates(sage, reference):
    sage, reference = sage.double(), reference.double()
    delta = sage - reference
    signal = reference.pow(2).mean().sqrt()

    rms = (delta.pow(2).mean().sqrt() / signal).item()
    assert rms < SAGE_MAX_RMS, f"RMS delta {rms * 100:.2f}% of output RMS"

    per_head = (delta.pow(2).mean(dim=(0, 1, 3)).sqrt()
                / reference.pow(2).mean(dim=(0, 1, 3)).sqrt())
    assert per_head.max() < SAGE_MAX_HEAD_RMS, (
        f"worst head {per_head.max().item() * 100:.2f}% "
        f"(median {per_head.median().item() * 100:.2f}%)")

    cos = torch.nn.functional.cosine_similarity(
        sage.reshape(-1, sage.shape[-1]),
        reference.reshape(-1, reference.shape[-1]), dim=-1).mean()
    assert cos > SAGE_MIN_COSINE, f"mean cosine similarity {cos.item():.6f}"

    gain = (delta * reference).sum() / reference.pow(2).sum()
    assert gain.abs() < SAGE_MAX_GAIN, f"systematic gain {gain.item() * 100:+.3f}%"

    bias = delta.mean(dim=(0, 1, 2)).pow(2).mean().sqrt() / signal
    assert bias < SAGE_MAX_CHANNEL_BIAS, f"per-channel bias {bias.item() * 100:.4f}%"


@pytest.fixture(scope="module")
def device():
    return torch.device("xpu")


# Shapes small enough to keep the suite quick but still exercise every branch:
# partial token tiles, a KV window shorter than Q's, and non-128 head dims.
SHAPES = [
    pytest.param({}, id="production-chunk"),
    pytest.param({"head_dim": 64}, id="head_dim-64"),
    pytest.param({"head_dim": 32, "num_heads": 4}, id="head_dim-32"),
    pytest.param({"kv_frames": 4}, id="short-kv"),
    pytest.param({"q_frames": 1}, id="single-query-frame"),
    pytest.param({"h_patches": 8, "w_patches": 8}, id="8x8-patches"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Reference kernel: bit-exact against the unfused pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _reference_vs_unfused(inputs):
    assert_bit_exact(invoke(fused_rope_prope_sdpa_split, inputs),
                     invoke(fused_rope_prope_sdpa_reference, inputs))


@pytest.mark.parametrize("rope_dtype", ROPE_DTYPES)
def test_bit_exact_per_rope_precision(device, rope_dtype):
    _reference_vs_unfused(create_test_inputs(device, rope_dtype))


@pytest.mark.parametrize("overrides", SHAPES)
def test_bit_exact_per_shape(device, overrides):
    _reference_vs_unfused(create_test_inputs(device, "float32", **overrides))


def test_no_intrinsics(device):
    inputs = create_test_inputs(device, "float32")
    inputs["q_Ks"] = inputs["kv_Ks"] = None
    _reference_vs_unfused(inputs)


def test_shared_query_and_kv_cameras(device):
    inputs = create_test_inputs(device, "float32", kv_frames=3)
    inputs["kv_viewmats"] = inputs["kv_Ks"] = None
    _reference_vs_unfused(inputs)


def test_logd4_camera_transform(device):
    inputs = create_test_inputs(device, "float32")
    inputs["camera_translation_transform"] = "logd4"
    _reference_vs_unfused(inputs)


def test_non_contiguous_query(device):
    inputs = create_test_inputs(device, "float32")
    inputs["q"] = inputs["q"].transpose(1, 2).contiguous().transpose(1, 2)
    assert not inputs["q"].is_contiguous()
    _reference_vs_unfused(inputs)


def _assert_batched_matches_independent_runs(device, kernel):
    # The unfused pipeline's apply_prope only supports batch=1, so batching is
    # checked against two independent single-batch runs instead.
    small = dict(kv_frames=4, h_patches=5, w_patches=6, num_heads=4)
    a = create_test_inputs(device, "float32", **small)
    b = create_test_inputs(device, "float32", seed=99, **small)
    batched = {key: torch.cat([a[key], b[key]], dim=0)
               for key in ("q", "k", "v", "q_grid_sizes", "k_grid_sizes",
                           "q_viewmats", "q_Ks", "kv_viewmats", "kv_Ks")}
    out = invoke(kernel, {**a, **batched})
    assert torch.equal(out[0], invoke(kernel, a)[0])
    assert torch.equal(out[1], invoke(kernel, b)[0])


def test_batched_matches_independent_runs(device):
    _assert_batched_matches_independent_runs(device, fused_rope_prope_sdpa_split)


def test_prope_matrix_kernel_matches_reference(device):
    """The single-launch camera-matrix kernel must be bit-exact, not just close."""
    torch.manual_seed(7)
    for _ in range(5):
        n = 7
        rot, _ = torch.linalg.qr(torch.randn(1, n, 3, 3, device=device).double())
        viewmats = torch.zeros(1, n, 4, 4, device=device)
        viewmats[..., :3, :3] = rot.float()
        viewmats[..., :3, 3] = torch.randn(1, n, 3, device=device) * 2.0
        viewmats[..., 3, 3] = 1.0
        Ks = torch.zeros(1, n, 3, 3, device=device)
        Ks[..., 0, 0] = torch.rand(1, n, device=device) * 2 + 0.2
        Ks[..., 1, 1] = torch.rand(1, n, device=device) * 2 + 0.2
        Ks[..., 0, 2] = torch.randn(1, n, device=device)
        Ks[..., 1, 2] = torch.randn(1, n, device=device)
        Ks[..., 2, 2] = 1.0

        for ks in (Ks, None):
            got = fk._prope_tables_triton(viewmats, ks)
            want = prepare_prope_matrices(viewmats, Ks=ks, target_seqlen=None)
            for g, w in zip(got, want):
                assert torch.equal(g, fk._flat16(w))


# ─────────────────────────────────────────────────────────────────────────────
# Sage kernel: accurate, against the reference kernel
# ─────────────────────────────────────────────────────────────────────────────

def _sage_vs_reference(inputs):
    assert_sage_gates(invoke(fused_rope_prope_sage, inputs),
                      invoke(fused_rope_prope_sdpa_split, inputs))


@pytest.mark.parametrize("rope_dtype", ROPE_DTYPES)
def test_sage_accuracy_per_rope_precision(device, rope_dtype):
    _sage_vs_reference(create_test_inputs(device, rope_dtype))


def test_sage_logd4_camera_transform(device):
    """The fallback camera path must rotate *its own* matrices."""
    inputs = create_test_inputs(device, "float32")
    inputs["camera_translation_transform"] = "logd4"
    _sage_vs_reference(inputs)


def test_sage_batched_matches_independent_runs(device):
    _assert_batched_matches_independent_runs(device, fused_rope_prope_sage)


def test_hadamard_rotation_earns_its_place(device):
    """The rotation is free, so it has to actually reduce the error."""
    inputs = create_test_inputs(device, "float32")
    reference = invoke(fused_rope_prope_sdpa_split, inputs)
    err = {rotate: rel_rms(invoke(fused_rope_prope_sage, inputs, rotate=rotate), reference)
           for rotate in (False, True)}
    assert err[True] < err[False] * 0.95, err


def test_hadamard_rotation_cancels_in_the_dot_product(device):
    """The rotation must leave the bfloat16 attention essentially unchanged.

    If it did not cancel, the "free accuracy" claim would be a bug instead.
    """
    h = hadamard4(device, torch.float64)
    assert torch.allclose(h @ h.T, torch.eye(4, dtype=torch.float64, device=device))

    inputs = create_test_inputs(device, "float32")
    mode = fk.ROPE_MODE[torch.float32]
    fr = fk._freqs_real(inputs["freqs"], "float32")
    cs_q = fk.build_rope_table(fr, inputs["q_grid_sizes"], inputs["q"].shape[1],
                               inputs["q_start_frame"])
    cs_k = fk.build_rope_table(fr, inputs["k_grid_sizes"], inputs["k"].shape[1],
                               inputs["k_start_frame"])
    plain = fk._prope_tables_batched(inputs["q_viewmats"], inputs["q_Ks"],
                                     inputs["kv_viewmats"], inputs["kv_Ks"], "linear")
    rot = hadamard4(device)
    q_t_rot, _, _ = fk._prope_tables_triton(inputs["q_viewmats"], inputs["q_Ks"], rot)
    _, kv_rot, _ = fk._prope_tables_triton(inputs["kv_viewmats"], inputs["kv_Ks"], rot)

    def attend(p_q_t, p_kv):
        q = fk.rope_prope_transform(inputs["q"], p_q_t, cs_q, mode, out_layout="HND")
        k = fk.rope_prope_transform(inputs["k"], p_kv, cs_k, mode, out_layout="HND")
        v = fk.rope_prope_transform(inputs["v"], plain[1], None, out_layout="HND")
        return torch.nn.functional.scaled_dot_product_attention(q, k, v).double()

    # only the bfloat16 rounding of the rotated matrices survives
    rel = rel_rms(attend(q_t_rot, kv_rot), attend(plain[0], plain[1]))
    assert rel < 5e-3, f"rotation changed the attention by {rel * 100:.3f}%"


# The Sage path's own transform kernels. V and the output projection must be
# the reference kernel's arithmetic bit for bit; Q and K, which are quantised
# to int8 next, rotate on fp16 (cos, sin) with fp32 FMAs and skip the
# reference's intermediate bfloat16 stores, so they are bounded instead of exact.

def _sage_intermediates(inputs, monkeypatch):
    """Run the Sage path and capture what it hands the attention core."""
    captured = {}
    attend = fused_sage._attend_int8

    def spy(q_eff, k_int8, k_scale, v_eff, *args):
        captured.update(q_eff=q_eff, k_int8=k_int8, k_scale=k_scale, v_eff=v_eff)
        return attend(q_eff, k_int8, k_scale, v_eff, *args)

    monkeypatch.setattr(fused_sage, "_attend_int8", spy)
    invoke(fused_rope_prope_sage, inputs)
    return captured


def _reference_transforms(inputs, rope_dtype):
    mode = fk.ROPE_MODE[normalize_rope_precision(rope_dtype)[0]]
    fr = fk._freqs_real(inputs["freqs"], rope_dtype)
    cs_q = fk.build_rope_table(fr, inputs["q_grid_sizes"], inputs["q"].shape[1],
                               inputs["q_start_frame"])
    cs_k = fk.build_rope_table(fr, inputs["k_grid_sizes"], inputs["k"].shape[1],
                               inputs["k_start_frame"])
    rot = hadamard4(inputs["q"].device)
    p_q_t, _, p_q = fk._prope_tables_triton(inputs["q_viewmats"], inputs["q_Ks"], rot)
    _, p_kv_k, _ = fk._prope_tables_triton(inputs["kv_viewmats"], inputs["kv_Ks"], rot)
    _, p_kv_v, _ = fk._prope_tables_triton(inputs["kv_viewmats"], inputs["kv_Ks"])
    q_eff = fk.rope_prope_transform(inputs["q"], p_q_t, cs_q, mode, out_layout="HND")
    k_eff = fk.rope_prope_transform(inputs["k"], p_kv_k, cs_k, mode, out_layout="HND")
    v_eff = fk.rope_prope_transform(inputs["v"], p_kv_v, None, out_layout="HND")
    return dict(tables=(cs_q, cs_k, p_q_t, p_kv_k, p_kv_v, p_q),
                q_eff=q_eff, k_eff=k_eff, v_eff=v_eff, fr=fr)


@pytest.mark.parametrize("rope_dtype", ROPE_DTYPES)
def test_sage_tables_match_the_reference_kernels(device, rope_dtype):
    """One launch builds both RoPE tables (fp16) and every camera matrix (exact)."""
    inputs = create_test_inputs(device, rope_dtype)
    ref = _reference_transforms(inputs, rope_dtype)
    fr = ref["fr"].float() if ref["fr"].dtype == torch.float64 else ref["fr"]
    ours = fused_sage._build_tables(
        fr, inputs["q_grid_sizes"], inputs["k_grid_sizes"],
        inputs["q"].shape[1], inputs["k"].shape[1],
        inputs["q_start_frame"], inputs["k_start_frame"],
        inputs["q_viewmats"], inputs["q_Ks"], inputs["kv_viewmats"], inputs["kv_Ks"],
        hadamard4(device), cameras=True)
    names = ("cs_q", "cs_k", "P_q^T", "P_kv^-1 (K)", "P_kv^-1 (V)", "P_q")
    for name, a, b in zip(names, ours, ref["tables"]):
        if name.startswith("cs"):
            # the same values, rounded once to fp16 (fp64 is narrowed to fp32 first)
            b = b.float().to(torch.float16)
        assert torch.equal(a, b), name


@pytest.mark.parametrize("rope_dtype", ROPE_DTYPES)
@pytest.mark.parametrize("overrides", SHAPES)
def test_sage_q_and_v_track_the_reference_transforms(device, monkeypatch, overrides,
                                                     rope_dtype):
    """V_eff is exact. Q_eff differs only by bfloat16-sized rounding (~2.5e-3
    relative RMS measured), far below the int8 step (~3% of a tile's RMS) it
    is quantised to next; a wrong rotation or camera reads 10-100x higher."""
    inputs = create_test_inputs(device, rope_dtype, **overrides)
    ref = _reference_transforms(inputs, rope_dtype)
    got = _sage_intermediates(inputs, monkeypatch)
    assert torch.equal(got["v_eff"], ref["v_eff"])
    assert rel_rms(got["q_eff"], ref["q_eff"]) < 5e-3


def test_sage_fused_quantiser_tracks_the_two_pass_one(device, monkeypatch):
    """With the centre averaged in full (short windows), quantising K inside its
    own RoPE+PRoPE pass agrees with quantising the reference's bfloat16 K_eff to
    within one int8 step (92% identical measured; the rest are rounding ties
    moved by the skipped bfloat16 store)."""
    inputs = create_test_inputs(device, "float32", q_frames=1, kv_frames=3,
                                h_patches=8, w_patches=8)
    ref = _reference_transforms(inputs, "float32")
    got = _sage_intermediates(inputs, monkeypatch)
    k_int8, k_scale = fused_sage.quantise_k(ref["k_eff"], fused_sage.k_window_mean(ref["k_eff"]))
    diff = (got["k_int8"].int() - k_int8.int()).abs()
    assert diff.max() <= 1
    assert (diff == 0).float().mean() > 0.85
    assert ((got["k_scale"] - k_scale).abs() / k_scale).max() < 1e-2


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

# The production chunk, then three small shapes where the call is bound by
# host-side launch overhead rather than the GPU.
BENCH_SHAPES = {
    "production-chunk": {},
    "shorter-64d": dict(head_dim=64, num_heads=8, q_frames=2, kv_frames=6),
    "small-32d": dict(head_dim=32, num_heads=4, q_frames=1, kv_frames=4),
    "partial-tile": dict(q_frames=1, kv_frames=3, h_patches=8, w_patches=8),
}
KERNELS = {
    "reference": fused_rope_prope_sdpa_split,
    "sage": fused_rope_prope_sage,
    "unfused": fused_rope_prope_sdpa_reference,
}


def timeit(fn, warmup=10, iters=30):
    """``(median, min)`` in ms over ``iters`` event-timed runs."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    samples = []
    for _ in range(iters):
        torch.xpu.synchronize()
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples)


def benchmark(device, rope_dtype="float32", seed=17):
    """Time all three kernels on every ``BENCH_SHAPES`` entry and print a table.

    Returns ``{shape: {kernel: (median_ms, rms_vs_reference, rms_vs_float64)}}``.
    """
    results = {}
    for shape, overrides in BENCH_SHAPES.items():
        inputs = create_test_inputs(device, rope_dtype, seed=seed, **overrides)
        reference = invoke(fused_rope_prope_sdpa_split, inputs)
        truth = float64_truth(inputs)
        results[shape] = {}
        for name, kernel in KERNELS.items():
            out = invoke(kernel, inputs)
            med, _ = timeit(lambda: invoke(kernel, inputs))
            results[shape][name] = (med, rel_rms(out, reference), rel_rms(out, truth))

    print(f"\n{torch.xpu.get_device_name(device)}, torch {torch.__version__}, "
          f"rope_dtype={rope_dtype}, median of 30 runs; RMS is relative to the output RMS")
    print(f"  {'shape':18s} {'kernel':10s} {'time':>9s} {'vs reference':>13s} "
          f"{'vs unfused':>11s} {'RMS vs ref':>11s} {'RMS vs fp64':>12s}")
    for shape, row in results.items():
        for name, (med, err_ref, err_truth) in row.items():
            print(f"  {shape:18s} {name:10s} {med:7.3f}ms "
                  f"{row['reference'][0] / med:12.2f}x {row['unfused'][0] / med:10.2f}x "
                  f"{err_ref * 100:10.3f}% {err_truth * 100:11.3f}%")
            shape = ""
    return results


def test_benchmark(device):
    """Prints the table (run with ``-s``). Asserts only what holds on any run:
    the reference stays exact, Sage stays within its RMS gate, and on the
    production chunk -- GPU-bound, not launch-bound -- Sage beats the reference."""
    results = benchmark(device)
    for row in results.values():
        assert row["reference"][1] == 0.0
        assert row["unfused"][1] == 0.0
        assert row["sage"][1] < SAGE_MAX_RMS
    production = results["production-chunk"]
    assert production["sage"][0] < production["reference"][0]


if __name__ == "__main__":
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise SystemExit("Intel XPU is required")
    benchmark(torch.device("xpu"))
