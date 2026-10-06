# Fused RoPE + PRoPE + SageAttention

Fused Triton kernels for the Wan2.2 Stage2 geometric-attention pipeline on
Intel Arc (Xe2; tuned for the Arc B70). The pipeline is

```
Q, K, V  ->  RoPE  ->  PRoPE  ->  attention  ->  output PRoPE  ->  O
```

which the original PyTorch code runs as eight separate stages. The fused paths
collapse RoPE + PRoPE into one streaming pass per tensor and the output
projection into one pass over the attention result.

Workload: Q `[1, 1215, 24, 128]`, K/V `[1, 7290, 24, 128]`, bfloat16, 3 query
cameras and 18 KV cameras — the production Stage2 chunk. It is defined once,
as `GEOMETRY` at the top of
[`test_fused_rope_prope_sage.py`](test_fused_rope_prope_sage.py); the tests
override it per case to cover smaller head dims, short KV windows and partial
token tiles.

## Requirements

An Intel XPU, `torch` with XPU support and the Intel XPU Triton backend
(developed with torch 2.14.1+xpu and Triton 3.8.0), plus `pytest`:

```bash
pip install torch==2.14.1+xpu --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple
pip install pytest
```

## Roles

| | what it is | file | entry point |
| --- | --- | --- | --- |
| **candidate** | fused RoPE + PRoPE + **int8 Sage** + output PRoPE | [`fused_sage.py`](fused_sage.py) | `fused_rope_prope_sage` |
| **reference kernel** | fused RoPE + PRoPE + **SDPA** + output PRoPE | [`fused_sdpa.py`](fused_sdpa.py) | `fused_rope_prope_sdpa_split` |
| unfused | the original stage-by-stage PyTorch pipeline | [`unfused_kernel.py`](unfused_kernel.py) | `fused_rope_prope_sdpa_reference` |

**The reference kernel is what new work is measured against**, for both speed
and accuracy. It is bit-exact against the unfused pipeline, which is kept as
the oracle that proves that and as context for what the fusion was worth.

The Sage kernel shares the reference's V PRoPE and output projection bit for
bit. Its Q/K RoPE+PRoPE runs at lower precision (fp16 `(cos, sin)`, fp32
accumulation, see change 5 below) because Q and K are quantised to int8 right
after; that moves the final error by under 0.01%, so the difference between
the two kernels is still, to measurement precision, the attention core alone.

```python
from fused_rope_prope_sage import (
    fused_rope_prope_sage,             # the int8 candidate
    fused_rope_prope_sdpa_split,       # the reference kernel
    fused_rope_prope_sdpa_reference,   # the unfused PyTorch pipeline
)

out = fused_rope_prope_sage(q, k, v, freqs=freqs, q_grid_sizes=q_grid,
                            k_grid_sizes=k_grid, q_viewmats=q_viewmats, q_Ks=q_Ks,
                            kv_viewmats=kv_viewmats, kv_Ks=kv_Ks,
                            q_start_frame=q_start_frame, rope_dtype="float32")
```

This folder is the package itself, so put its *parent* directory on the import
path (or install it there). All three take the same arguments (`q`, `k`, `v` are `[B, L, H, D]` bfloat16
on the XPU) and return `[B, Lq, H, D]`. The fused kernels need `head_dim / 4`
to be a power of two. The guide's literal single-kernel design was
implemented, measured at ~10–25x slower on this backend, and removed;
`FUSION_NOTES.md` has the numbers and the reason.

## Running the tests and the benchmark

```bash
pytest test_fused_rope_prope_sage.py              # 45 tests: accuracy + benchmark
pytest test_fused_rope_prope_sage.py -s -k bench  # benchmark only, printing its table
python test_fused_rope_prope_sage.py              # the benchmark table alone
```

The accuracy tests check that the reference kernel is **bit-exact** against
the unfused pipeline (all three RoPE precisions, six shapes, with and without
intrinsics, shared cameras, `logd4`, batching, non-contiguous input), and that
the Sage kernel stays within its accuracy gates against the reference kernel
(see "Accuracy gates" below). The Sage path's own transform kernels are also
checked: V_eff and every camera matrix bit-exact, Q_eff and the int8 K bounded.

`test_benchmark` times all three kernels on four shapes and reports speed and
RMS error, both against the reference kernel and against the whole pipeline
computed in float64. It asserts only what holds on any run: the reference
stays exact, Sage stays within its RMS gate, and Sage beats the reference on
the production chunk.

## Results

### Intel Arc B70 (current target)

Measured on `Intel(R) Graphics [0xe223]` (Battlemage / `bmg`, 32 Xe cores,
discrete), torch 2.14.1+xpu, Triton 3.8.0, `rope_dtype=float32`, with
`test_benchmark` (median of 30 event-timed runs; RMS relative to the output
RMS):

| shape | kernel | time | vs reference | vs unfused | RMS vs reference | RMS vs float64 |
| --- | --- | --- | --- | --- | --- | --- |
| **production chunk** | **Sage (int8)** | **1.65 ms** | **1.23x** | **2.90x** | 1.53% | 1.51% |
| | reference | 2.02 ms | 1.00x | 2.36x | 0 (bit-exact) | 0.41% |
| | unfused | 4.77 ms | 0.42x | 1.00x | 0 (bit-exact) | 0.41% |
| shorter-64d | Sage (int8) | 0.33 ms | 1.37x | 4.66x | 1.46% | 1.44% |
| | reference | 0.45 ms | 1.00x | 3.41x | 0 | 0.42% |
| small-32d | Sage (int8) | 0.33 ms | 1.36x | 4.66x | 1.39% | 1.37% |
| | reference | 0.45 ms | 1.00x | 3.42x | 0 | 0.42% |
| partial-tile | Sage (int8) | 0.30 ms | 1.44x | 5.25x | 1.41% | 1.40% |
| | reference | 0.44 ms | 1.00x | 3.65x | 0 | 0.40% |

The three small shapes are bound by host-side launch overhead, not by the GPU
(~0.1 ms of kernel time per call), which is why the Sage path is five launches
against the reference's nine. Event timing includes that host overhead, so
these small-shape times are about 2x what the GPU-side KernelFoundry harness
used during development reported (0.15 ms Sage / 0.24 ms reference). As
inherited from the LNL work, the same `fused_sage.py` ran the production chunk
at 18.2 ms here (0.11x) -- see "Porting to B70" below for why.

### Intel Arc integrated (`lnl`, where the kernels were first written)

Measured on `Intel(R) Arc(TM) Graphics` (Xe2 / `lnl`, 8 Xe cores), torch
2.12.1+xpu, Triton 3.7.1. Median of 30 timed runs; `rope_dtype=float32`.
These numbers predate the B70 port.

| path | time | vs reference | accuracy vs reference |
| --- | --- | --- | --- |
| **fused + SDPA (reference)** | **9.0 ms** | **1.00x** | — |
| fused + Sage (int8) | 9.7 ms | 0.89–1.05x | 1.49% RMS, 36.5 dB SNR, cos 0.99989 |
| unfused PyTorch | 23.5 ms | 0.37–0.43x | bit-exact (0 ULP) |

So on LNL the fusion itself is worth **~2.5x** over the unfused pipeline, and
the int8 path traded ~1.5% RMS error for roughly break-even speed. Run-to-run
spread on that (shared, integrated) GPU is
about ±5%, which is wider than the current gap between the two fused paths;
take the `min` column as well as the median when comparing them.

The reference kernel's bit-exactness holds for all three RoPE precisions
(`float64`, `float32`, `float16`) — not "within tolerance", but the identical
bit pattern.

### Where the int8 path stands (B70)

Each launch of the Sage call timed on its own (median of 30; the small
launches carry ~40 us of event overhead):

| launch | what it does | time |
| --- | --- | --- |
| `_sage_tables_kernel` | both RoPE tables + every camera matrix | 0.01 ms* |
| `_sage_prologue_kernel` | RoPE+PRoPE of Q, and the K centre | 0.04 ms* |
| `_sage_kv_kernel` | RoPE+PRoPE+centre+int8 of K, PRoPE of V | 0.295 ms |
| `_sage_attn_kernel` | the int8 attention | 1.02 ms |
| `_transform_kernel` | output projection (the reference's kernel) | 0.02 ms* |
| `F.scaled_dot_product_attention`, the reference core | | 1.33 ms |

\* `triton.testing.do_bench`, which does not carry the event overhead.

The KV launch moves ~157 MB and runs at the ~560 GB/s this part sustains, so
the non-attention launches are within ~15% of the memory floor. What is left is
the attention core: 1.3x faster than SDPA, but still well short of the DPAS
peak -- every warp re-loads the full K and V tile for its 8 rows, and 16 rows
per warp does not fit in 256 GRF (see FUSION_NOTES.md, "Porting to B70").

## What was optimised in `fused + Sage`

This is the history of the candidate, for whoever picks it up next. The LNL
work comes first; the B70 port is at the end of this section.

The int8 kernels in the supplied `SageAttention_Triton_Optimized.py` (not part
of this repo), used as supplied, run
the pipeline at **26.5 ms — slower even than the unfused pipeline**. Four
changes bring it to ~9 ms, a **3x speedup over the supplied kernel**, while
also *reducing* its error by 1.17x. `fused_sage.py` now contains only the
result, so the supplied kernels are no longer imported at all.

**1. A correctness fix first.** `SageAttention_Triton_Optimized.forward` sizes
its launch grid with a hard-coded `grid_block_m = 128`, but the kernel derives
its block index from the *autotuned* `BLOCK_M` — which is 64 on this device. The
launch therefore covers 10 of the 19 query blocks per head and mis-maps the
rest: **47% of the output is wrong**, by up to 343x the output RMS. Nothing
below would have meant anything without this.

**2. 2D block loads instead of masked pointer arithmetic (the big one).** The
supplied kernels address memory with masked pointer arithmetic, which never
lowers to Xe2's 2D block-load hardware. Rewriting the identical algorithm with
block pointers takes the attention stage from 25.5 ms to 6.8 ms.

**3. A real `int8 x int8 -> int32` dot.** The supplied kernel widens its int8
values to float16 before `tl.dot`, which runs on the same DPAS path as
bfloat16 — so quantising to int8 bought no arithmetic throughput at all, only
the halved K footprint. Using the int8 DPAS doubles the K per instruction.

**4. A 4x4 Hadamard rotation folded into the camera matrices — free accuracy.**
PRoPE already multiplies every contiguous 4-channel block by a 4x4 matrix, and
for any orthogonal `H`

```
(H·P_qᵀ q) · (H·P_kv⁻¹ k)  =  (P_qᵀ q) · (P_kv⁻¹ k)
```

so replacing `P_qᵀ` and `P_kv⁻¹` with `H·P_qᵀ` and `H·P_kv⁻¹` **on the host**
changes nothing about the attention, while spreading channel outliers inside
each block before quantisation. `P` (the output projection) and the V-side
`P_kv⁻¹` are left alone — there is nothing for the rotation to cancel against
there. Zero runtime cost, error 1.72% -> 1.47% (1.17x). Pass `rotate=False` to
`fused_rope_prope_sage` to A/B it.

Two supporting fixes: the per-head K-centring vector was being computed with
`k_eff.mean(dim=2)`, which reduces over a strided middle axis at ~19 GB/s
(2.4 ms); `fused_sage.k_window_mean` streams the same bytes with block loads in
0.49 ms. And the Q/K/V transforms emit contiguous `[B, H, L, D]` directly,
which both attention cores prefer.

### Porting to B70: 18.2 ms -> 1.57 ms

The LNL-tuned `fused_sage.py` ran at 18.2 ms here, 0.11x the reference, with
17.3 ms of that in the attention kernel alone. Changes, in order of impact:

**1. Tensor descriptors instead of block pointers (the big one).** With Triton
3.8 on BMG, `tl.make_block_ptr` in the attention loop no longer lowers to 2D
block IO: the TTGIR shows tensor-of-pointer loads in a `#blocked` layout, Q is
re-read through shared memory every iteration, and the kernel spills 13-48 KB
per thread. Rewriting the same loop with `tl.make_tensor_descriptor` takes it
from 17.3 ms to 0.97 ms, against SDPA's 1.33 ms. (The streaming transform
kernels are unaffected -- block pointers there still run at ~560 GB/s.)

**2. Eight query rows per warp.** The LNL autotune list (16+ rows per warp)
is exactly the set of configs that spill here: DPAS keeps the whole K and V
tile in each warp's registers. `BLOCK_M == 8 * num_warps` everywhere, picked by
a heuristic instead of the autotuner.

**3. K quantised inside its own transform.** `_sage_kv_kernel` does RoPE,
PRoPE, bfloat16 rounding, centring and int8 for a 32-token block in one pass
from the raw K (and V's PRoPE in the same launch), so the bfloat16 K_eff is
never written or re-read. The centre is the mean of one 64-token tile in every
512 (any constant centre is exact under softmax; the error is unchanged to
+-0.002%), computed by the prologue launch with each program owning a slice of
4-channel blocks so that no cross-program reduction is needed. Windows under
2048 tokens are averaged in full, and there the int8 K matched the old
two-pass quantiser bit for bit (since change 5 it agrees to within one int8
step; tested).

**4. Five launches instead of thirteen.** At the small test shapes the call is
host-bound, so launch count is runtime. Both RoPE tables and every camera
matrix are built in one launch -- the matrix programs inline the reference's
own `_prope_matrix_kernel`, so the camera matrices are bit-identical (tested) --
and the Q transform shares a launch with the K centre. V_eff stays
bit-identical to the reference path's.

**5. RoPE for Q and K in fp16 with fp32 accumulation, for every `rope_dtype`.**
Q and K are quantised to int8 right after their transform, so reproducing the
reference's rotation precision there buys nothing. They now rotate on an fp16
`(cos, sin)` table with fp32 FMAs (bf16 inputs and fp16 factors are both exact
in fp32, so the products are exact) and skip the reference's two bfloat16
stores -- after RoPE, and of K_eff before int8 -- which are only double
rounding. V and the output projection are untouched and stay bit-exact.

| `rope_dtype` | before | after | error vs fp64, before -> after |
| --- | --- | --- | --- |
| float32 | 1.39 ms | 1.40 ms | 1.504% -> 1.509% |
| float64 (the function default) | 1.64 ms | 1.42 ms | 1.504% -> 1.509% |
| float16 | 1.51 ms | 1.42 ms | 1.516% -> 1.509% |

A bf16 `(cos, sin)` table was measured too: same speed, but +0.01% RMS and a
systematic gain of 0.04% instead of 0.03% -- small, but bias rather than noise,
so fp16 it is. Pure fp16 math is no faster and can overflow above 65504.

Also fixed: with `camera_translation_transform="logd4"` and `rotate=True`, the
old code built the rotated Q/K matrices with the linear-only Triton kernel,
handing Q and K a different camera transform than V and the output (3.8% RMS).
The fallback now rotates the logd4 matrices themselves (1.44%).

Measured on B70 and rejected (numbers in FUSION_NOTES.md): computing S^T so
the softmax reductions stay in-lane, pre-transposed `[D, L]` int8 K, software
pipelining QK(j+1) against softmax(j), FMA-folded softmax with a tail-only
mask, splitting the PV dot to fit 16 rows per warp, int8 PV, per-row Q scales
(1.17x more accurate, but IGC spills and the attention doubles), and
gathering RoPE values inside the transforms instead of building a table.

### Accuracy gates for the int8 candidate

`test_sage_accuracy_against_reference_kernel` enforces these against the
reference kernel. They are **regression tripwires, not a model-level
correctness criterion** — that has to come from video PSNR on the real model.

| gate | today | limit | headroom |
| --- | --- | --- | --- |
| global RMS | 1.49% | **1.8%** | 1.2x |
| worst-head RMS | 1.54% | 2.5% | 1.6x |
| systematic gain | 0.032% | 0.5% | 15.6x |
| per-channel bias | 0.015% | 0.1% | 6.5x |
| mean cosine | 0.999890 | 0.9995 | 4.5x |

The 1.8% comes from what int8 Q/K can actually produce. Sweeping the K-scale
granularity from per-token to a single scale per head moves the error only
between **1.28% and 1.68%** — the floor is int8 itself, not the scaling:

| K scale granularity | RMS | effective bits |
| --- | --- | --- |
| per token (finest possible) | 1.28% | 6.3 |
| per 32 tokens — today | 1.44% | 6.1 |
| per 512 tokens | 1.56% | 6.0 |
| one scale per head (coarsest) | 1.68% | 5.9 |

So past ~1.8% you are not tuning a quantiser any more: you have quantised V,
moved to a narrower format, or introduced a bug. For calibration, the bugs this
caught while the kernel was being written read 17.9%, 26.7% and 69%.

The gain and bias gates matter more than their headroom suggests. Sage's error
today is **zero-mean noise** — bias is ~1% of the total, kurtosis 3.2, no hot
head (worst/median 1.04x) — which is why it grows as `sqrt(N)` over a chunk's
150 attentions instead of accumulating. A biased quantiser (round-toward-zero
instead of round-half-away, say) can barely move the RMS and still compound 150
times. Those two gates are what keep the RMS gate meaningful.

Context for the numbers: the reference kernel is *itself* an approximation at
0.42% RMS against float64, and bfloat16 output rounding alone costs 0.17%. In
effective mantissa bits that is 7.9 for the reference, 6.1 for Sage today, and
5.6 at a 2% budget — so 1.5% to 2% is about half a bit.

### Measured, and deliberately not in the tree

Two further arithmetic changes are free in operation count and do reduce the
error, but this backend lowers them badly: both need a *per-row* value broadcast
across the dot's accumulator layout, which goes through shared memory here.

| | error vs reference | end-to-end |
| --- | --- | --- |
| current | 1.47% | 9.0 ms |
| + per-token Q and K scales | 1.15% | 16.8 ms |
| + per-block centres, quantisation fused into the transform | 1.13% | 60.6 ms |

In isolation: a per-token K scale costs +0.3 ms, a per-token Q scale +5 ms, and
the rank-1 centre correction +26 ms — against an inner loop that should see ~3%
more arithmetic each. They were removed from `fused_sage.py` to keep it to one
kernel; `FUSION_NOTES.md` has the derivations, in case the backend's row
broadcasts ever get cheaper.

## Layout

| file | what it is |
| --- | --- |
| [`__init__.py`](__init__.py) | makes this folder the `fused_rope_prope_sage` package |
| [`fused_sage.py`](fused_sage.py) | fused + SageAttention (int8), the candidate |
| [`fused_sdpa.py`](fused_sdpa.py) | **the reference kernel** (fused + SDPA) |
| [`unfused_kernel.py`](unfused_kernel.py) | unfused PyTorch pipeline, the oracle (unmodified) |
| [`test_fused_rope_prope_sage.py`](test_fused_rope_prope_sage.py) | accuracy tests and benchmark |
| [`FUSION_NOTES.md`](FUSION_NOTES.md) | why the fusion boundary is where it is, and the backend gotchas |
