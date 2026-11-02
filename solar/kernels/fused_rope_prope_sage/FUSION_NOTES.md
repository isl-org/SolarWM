# Fusing RoPE + PRoPE + SDPA on Intel Arc (Xe2 / `lnl`)

Written for: engineers continuing this kernel work.

Device under test: `Intel(R) Arc(TM) Graphics` (`arch=lnl`, 8 Xe cores, 64 EUs,
`has_2d_block_io`, DPAS), torch 2.12.1+xpu, Triton 3.7.1 (Intel XPU backend).

## Result

The reference kernel (`fused_sdpa.py`) is what the rest of the work is measured
against; `unfused_kernel.py` is the unfused PyTorch pipeline it replaced.

| path | median | vs unfused | numerics |
| --- | --- | --- | --- |
| `unfused_kernel.py` (unfused PyTorch) | 23.6 ms | 1.00x | — |
| `fused_sdpa.py` (fused Triton) | 9.4 ms | **2.5x** | **bit-exact (0 ULP)** |
| the guide's literal single kernel (removed, see below) | ~100–240 ms | 0.1–0.2x | ≤ 9.8e-4 abs |

KernelFoundry harness numbers for the same workload: reference 20.9 ms median,
fused 8.4 ms median (**2.48x**).

Geometry is the production Stage2 chunk: Q `[1, 1215, 24, 128]`,
K/V `[1, 7290, 24, 128]`, bfloat16, 3 query cameras and 18 KV cameras.

## Why the guide's single kernel loses here

The guide asks for one kernel that loads Q/K/V once and does RoPE, PRoPE,
FlashAttention and the output projection in registers. It was implemented and
it was correct, but ~25x slower than the split form on this backend, so it has
been removed from the tree (it is in git history as `fused_geometric_attention`
in `fused_sdpa.py`). The reason is a single lowering issue:

RoPE pairs channels `(2c, 2c+1)`; PRoPE mixes the four channels of each block
`(4p .. 4p+3)`. Both need the `[BLOCK, 128]` register tile de-interleaved into
four `[BLOCK, 32]` planes and re-interleaved afterwards. Expressed the natural
way — `tl.reshape` + `tl.split` / `tl.join` — the Intel backend lowers each one
to per-element shared-memory traffic. Measured on a plain flash-attention
kernel at `BLOCK_M=64, BLOCK_N=32`:

| K/V loop body | time |
| --- | --- |
| plain attention, transposed block load | 11.8 ms |
| plain attention, `tl.trans(k)` | 9.4 ms |
| + `tl.split` / `tl.join` round trip only | **268 ms** |
| + full RoPE and PRoPE on top | 270 ms |

`tl.trans` is free (it is actually *faster* than a transposed block load — the
compiler folds it into the DPAS operand layout). The de-interleave is not.

Two alternatives were measured and rejected:

- **Four narrow dots.** Decompose into four `[M,32] x [32,N]` DPAS operations so
  no re-interleave is needed. 184 ms — 32-wide operands destroy the DPAS tiling.
- **Dense `128x128` PRoPE matrix per K/V tile.** Expressible as `tl.dot`, but
  applying a dense matrix to each K/V token costs 128 MACs per element against
  attention's `BLOCK_M`, so the overhead is `128 / BLOCK_M` ≥ 50% at any
  `BLOCK_M` that fits in registers.

There is also no way to express the two-stage form (RoPE, round to bfloat16,
then PRoPE) with shifted *memory* loads, because the intermediate only exists in
registers — and the guide explicitly forbids collapsing `P·R` into one matrix
(the WS18 PSNR trap). So the fusion boundary has to move.

Separately, a from-scratch Triton flash attention peaks at **10.4 TFLOPS**
(10.5 ms) on this part against torch's **15.1 TFLOPS** (7.2 ms), so there is no
headroom to pay for fusion out of a faster attention core either.

## What the fused path does instead

Four Triton kernels around the vendor SDPA:

1. `_rope_table_kernel` — expands the 3-split Wan2.2 frequency table into a
   per-token `(cos, sin)` table, plane-major so the transform kernel reads it
   with four dense block pointers. Grid sizes are read from the device tensor,
   so there is no host synchronisation.
2. `_prope_matrix_kernel` — one program per camera builds `(P^T, P^-1, P)`.
3. `_transform_kernel` — fused RoPE + PRoPE, one streaming pass per tensor
   (Q, K, V), writing contiguous `[B, H, L, D]`.
4. `F.scaled_dot_product_attention`.
5. `_transform_kernel` again for the output projection, converting back to
   `[B, L, H, D]`.

The de-interleave problem is solved by **not doing it in registers**: four
contiguous bfloat16 channels are exactly one int64, so the kernel block-loads
the int64 view of the tensor and unpacks the four PRoPE lanes with shifts and
masks. A bfloat16 bit pattern shifted into the high half of an int32 *is* the
float32 of the same value, so the widening conversion is free too:

```python
y0 = ((x & 0xFFFF) << 16).to(tl.int32).to(tl.float32, bitcast=True)
```

The repack is the mirror image. No layout conversion, no shared memory,
dense 2D block loads and stores throughout.

### Stage breakdown

| stage | unfused | fused |
| --- | --- | --- |
| RoPE(Q) | 1.54 | — |
| RoPE(K) | 7.17 | — |
| camera matrices | 0.97 | 0.25 |
| RoPE tables | — | 0.13 |
| PRoPE(Q) / fused Q | 0.60 | 0.22 |
| PRoPE(K) / fused K | 2.85 | 0.90 |
| PRoPE(V) / fused V | 2.89 | 0.87 |
| SDPA | 7.92 | 7.21 |
| PRoPE(O) | 0.53 | 0.17 |
| **sum** | **24.45** | **9.76** |

Everything except attention went from 16.5 ms to 2.55 ms (6.5x). The K and V
transforms run at ~110 GB/s against a ~136 GB/s LPDDR5X peak, so they are at
the memory roof and the remaining 7.2 ms is torch's oneDNN attention.

## Bit-exactness

The fused path is **bit-identical** to `unfused_kernel.py` for all three RoPE
precisions, which matters more than it sounds: the task's tolerance is
`max_ulp=8` with *zero* elements allowed to exceed it, and for bfloat16 outputs
that is unreachable for any reimplementation of the attention itself. A control
run — replacing only the attention with an exact float32 computation and leaving
everything else as the reference — puts **2.2%** of elements beyond 8 ULP, and
the single-kernel flash variant puts 0.71% beyond it. Near-zero outputs after
the `P_q` projection amplify any attention-level difference without bound.

Keeping the vendor SDPA sidesteps that entirely, but only if the tensors fed to
it are bit-identical. Getting there needed three fixes that are worth recording:

- **torch's complex multiply contracts into an FMA.** `(a*c) - (b*d)` as two
  rounded products leaves 45 of 3.7M elements wrong; `fma(a, c, -(b*d))` is
  exact. Same for the imaginary part.
- **Triton folds `x.to(tl.float16).to(tl.float32)` away entirely** (the float16
  rounding simply does not happen — this is silent, and a bitcast in between
  does not stop it). `rope_dtype="float16"` therefore rounds on the bit pattern
  in `_round_fp16`, with the shift widened for float16 subnormals. The bfloat16
  equivalent is *not* folded, so `.to(tl.bfloat16).to(tl.float32)` is fine.
- **`rope_dtype="float64"` needs real float64 in the kernel.** float32 leaves
  ~45 elements of `q_rope` off by up to 35 ULP, and a single wrong input element
  perturbs a whole attention row — which showed up as ~400 out-of-tolerance
  outputs. The float64 → float32 → bfloat16 double rounding matches what torch
  does for the same conversion.

The 4x4 PRoPE contraction needed no special care: `v_prope` came out bit-exact
on the first try, which is also what proves the float32-accumulate-then-round
sequence matches torch's bfloat16 einsum.

`_prope_matrix_kernel` replaces ~30 tiny torch launches with one, and is checked
bit-for-bit against `prepare_prope_matrices` on randomised cameras (real
rotations, translations, off-centre intrinsics) in
`test_fused_rope_prope_sage.py`.

## Things that cost real time and are easy to miss

- `grid_sizes.tolist()` is a **0.32 ms device synchronisation** per call. The
  reference pays it twice; the fused path reads the grid shape on the device.
- Feeding SDPA contiguous `[B, H, L, D]` instead of a transposed view of
  `[B, L, H, D]` is worth **~1.1 ms** (8.5 → 7.4 ms) and returns identical bits.
  The transform kernels emit that layout directly.
- `BLOCK_T >= 128` in the transform kernel falls off a cliff (18 ms at 256 vs
  0.82 ms at 16–64). The autotune list is pinned to the flat region so run-to-run
  timing noise cannot select a pathological config.
- `grf_mode` is an Intel-only compile option and is not a `triton.Config`
  keyword in this build; it has to be passed inside the config's kwargs dict.

## SageAttention (int8) as the attention core

`fused_sage.py` is the same fusion with `F.scaled_dot_product_attention`
replaced by the int8 kernels from the supplied `SageAttention_Triton_Optimized.py`
(not part of this repo). RoPE,
PRoPE and the output projection are shared and stay bit-exact, so every
difference below is the attention core alone.

| attention core | end-to-end | attention stage | RMS delta vs float64 | SNR | mean cosine |
| --- | --- | --- | --- | --- | --- |
| `F.scaled_dot_product_attention` (bf16) | **9.3 ms** | 7.2 ms | 0.42% | 47.6 dB | 0.999993 |
| SageAttention, as given | 27.4 ms | 25.5 ms | 1.74% | 35.2 dB | 0.999851 |
| SageAttention, block-pointer rewrite | 9.1 ms | 6.8 ms | 1.72% | 35.3 dB | 0.999855 |

Sage against the bfloat16 path: RMS delta 1.73% of output RMS, mean |delta|
1.36%, max |delta| 16.5%, 99.3% of elements within 5% of RMS. That is normal
for int8 attention and stable across head dims, RoPE precisions and KV lengths
(1.3–1.7%).

Three things are worth knowing before reusing that kernel:

- **`SageAttention_Triton_Optimized.forward` is broken on this device.** It
  sizes its grid with a hard-coded `grid_block_m = 128` while the kernel derives
  its block index from the autotuned `BLOCK_M`. The autotuner picks
  `BLOCK_M=64` here, so the launch covers 10 of the 19 query blocks per head and
  mis-maps the rest: **47% of the output is wrong**, by up to 343x the output
  RMS. `fused_sage.py` sizes its grid from the tile it actually launches
  (`_attn_config`), so it cannot drift from the kernel's `BLOCK_M`.
- **Quantising to int8 buys no arithmetic here as written.** The kernel widens
  the int8 values to float16 before `tl.dot`, which runs on the same DPAS path
  as bfloat16. The only saving is the halved K footprint.
- **Masked pointer-arithmetic loads do not become 2D block loads**, which is
  what costs the 3.5x. Rewriting the identical algorithm — same centring, same
  per-block K scales, same per-tile Q scale, same online softmax — with block
  pointers and a real `int8 x int8 -> int32` dot takes the attention stage from
  25.5 ms to 6.8 ms, slightly *faster* than torch's bfloat16 FlashAttention.

So on this part int8 attention is roughly a wash on speed and costs ~12 dB of
accuracy: the quantisation pre-pass (centre, scale, write 22 MB of int8) eats
the bandwidth that the narrower K loads save. `k_eff.mean(dim=2)` in torch is
also worth avoiding — it reduces over a strided middle axis at ~19 GB/s
(2.4 ms); `fused_sage.k_window_mean` does it in 0.49 ms.

## Making the int8 attention more accurate for free

Four changes to the quantisation arithmetic, none of which adds meaningful work
to the inner loop. Measured end to end against the bfloat16 path, same fixture:

| variant | RMS vs fp64 | SNR | gain |
| --- | --- | --- | --- |
| Sage as given | 1.74% | 35.2 dB | — |
| block-pointer rewrite (same arithmetic) | 1.72% | 35.3 dB | 1.01x |
| + 4x4 Hadamard in the PRoPE matrices | 1.47% | 36.6 dB | 1.17x |
| + per-token Q and K scales | 1.15% | 38.8 dB | 1.50x |
| + per-block centres | 1.13% | 38.9 dB | 1.52x |

Stable to ±0.01% across seeds and RoPE precisions. In order of value:

- **A 4x4 orthogonal rotation folded into the camera matrices.** PRoPE already
  multiplies every contiguous 4-channel block by a 4x4 matrix, and
  `(H P_q^T q) . (H P_kv^-1 k) = (P_q^T q) . (P_kv^-1 k)` for any orthogonal
  `H`. Replacing `P_q^T` and `P_kv^-1` with `H P_q^T` and `H P_kv^-1` on the
  host therefore changes nothing about the attention while spreading channel
  outliers inside each block before quantisation. `P` (the output projection)
  and the V-side `P_kv^-1` are left alone, so the rotation has nothing to
  cancel against there. **Zero runtime cost, 1.17x lower error.**
- **Per-token scales instead of per-tile / per-32-block scalars.** A per-row Q
  scale and a per-column K scale both factor straight out of the dot:
  `q_hat . k_hat = qs[m] * ks[n] * (q_i8 . k_i8)`. The given kernel takes one
  `amax` over a whole `[BLOCK_M, 128]` Q tile and one over a `[32, 128]` K
  block, so the quietest token in a tile is quantised against the loudest one.
  Per-token costs one extra vector per tile. **1.28x lower error on top.**
- **Per-block centres instead of one mean per head.** Softmax is invariant to a
  shift that is constant across *all* keys, which is what makes the per-head
  mean free; a per-block centre needs a correction, but the correction is
  exactly rank one — `q . (c_t + ks*k_i8) = q . c_t + ks * (q . k_i8)` — at
  `BLOCK_M*D` against the dot's `BLOCK_M*BLOCK_N*D`. It is never worse than the
  per-head mean and is 1.18x better when K carries positional structure, which
  RoPE produces: a slowly varying channel offset that one global mean cannot
  remove. It also makes quantisation *local*, which lets it fuse into the
  RoPE+PRoPE pass and removes two of the three passes over K.
- Smoothing Q by its channel mean (SageAttention-2's `smooth_q`) was measured
  and does nothing here — after RoPE and PRoPE, Q's channel means are 0.9% of
  its RMS.

### ...and what that costs on this particular part

Free in operation count is not free in wall clock here:

| variant | end-to-end | vs SDPA-fused |
| --- | --- | --- |
| fused + SDPA (bfloat16) | 9.1 ms | 1.00x |
| fused + Sage, block-ptr | 8.7 ms | 1.05x |
| + Hadamard | 9.0 ms | 1.01x |
| + per-token scales | 16.2 ms | 0.56x |
| + per-block centres, quantisation fused | 58.6 ms | 0.16x |

The Hadamard is genuinely free — it is two extra 4x4 matrices on the host. The
other two are not, for the same reason the single-kernel fusion failed: both
need a *per-row* value broadcast across the dot's accumulator layout every
iteration, and this backend lowers that through shared memory. Measured in
isolation on the attention stage: a per-token K scale costs +0.3 ms, a
per-token Q scale +5 ms, and the rank-1 centre correction +26 ms — against an
inner loop that should see 3%, 3% and 3% more arithmetic respectively.
Hoisting the broadcast out of the loop does not help.

The practical configuration on this part is therefore **block-pointer + the
Hadamard rotation**: 1.47% error at 9.0 ms, i.e. 1.17x lower error than the
given kernel at 3x its speed. That is what `fused_sage.py` contains.

## Porting to B70 (Battlemage, Triton 3.8)

Device: `Intel(R) Graphics [0xe223]` (`arch=bmg`, 32 Xe cores, 256 XVEs,
discrete), torch 2.14.1+xpu, Triton 3.8.0. Triton detects DPAS and 2D block IO
here (`has_subgroup_matrix_multiply_accumulate`, `has_2d_block_io`); torch's
own `get_device_capability()` reports both as False, but Triton queries the
extensions itself, so that is a red herring.

### Block pointers stop lowering to block IO

The inherited Sage pipeline ran at 18.2 ms (reference 1.94 ms), 17.3 ms of it
in the attention. Every attention config spilled; the TTGIR shows why:

```
%q   = tt.load %ptrs, %mask : tensor<64x128x!tt.ptr<bf16>, #blocked>
%q_s = ttg.local_alloc %q          // Q round-trips through SLM ...
...  = ttg.local_load %q_s         // ... every iteration
%k   = ttg.convert_layout %kk : #blocked -> #linear
```

`tl.make_block_ptr` is rewritten to tensor-of-pointer loads. The same loop with
`tl.make_tensor_descriptor` gets `ttig.block_io` loads straight into the DPAS
operand layouts (`column_major` for K via `tl.trans`, `row_major` + VNNI
transform for V):

| attention, production chunk | time | spills |
| --- | --- | --- |
| block pointers, best config | 16.98 ms | 17-23 KB |
| descriptors, BLOCK_M=128, 16 warps | 0.97-0.99 ms | 0 |
| descriptors, BLOCK_M=256, 32 warps | 0.97 ms | 0 |
| `F.scaled_dot_product_attention` | 1.33 ms | -- |

The streaming transforms are *not* affected: `_transform_kernel` with block
pointers moves K at ~560 GB/s, the same as a descriptor rewrite.

### Rows per warp

Only `BLOCK_M == 8 * num_warps` avoids spilling (256 GRF; 512 GRF is not
supported on this part). At 16 rows/warp the K tile (64 GRF), V tile (128 GRF)
and accumulator (128 GRF) cannot all be resident:

| BLOCK_M / warps | rows/warp | BLOCK_N=32 | spills |
| --- | --- | --- | --- |
| 64 / 8, 128 / 16, 256 / 32 | 8 | 1.0-1.15 ms | 0 |
| 64 / 4, 128 / 8, 256 / 16 | 16 | 6.6-8.2 ms | 12 KB |
| 128 / 8 with PV split into two D-halves | 16 | 1.4-2.2 ms | 2-5 KB |

### Where the attention time goes (BLOCK_M=128/16 warps, int8 Q, K)

| loop body | time |
| --- | --- |
| full online softmax | 0.98 ms |
| QK + PV, no softmax at all | 0.88 ms |
| PV only | 0.37 ms |
| QK dot only, K loaded per iteration, BLOCK_N=32 | 0.49 ms |
| QK dot only, K tile loaded once | 0.20 ms (270 TOPS) |

Softmax is ~10% of the loop. The cost is the dots and, above all, re-loading
the K and V tiles: each warp owns 8 rows but loads the full 4 KB + 8 KB per
iteration, ~10 GB of L1 traffic per call.

### Measured on B70 and rejected

| variant | attention | why |
| --- | --- | --- |
| S^T = K Q^T, so row reductions stay in-lane | 7-63 ms / build failure | spills; IGC lacks the transposed d32 read it needs at BLOCK_M>=256 |
| K stored transposed `[D, L]` (row-major B operand) | 4-50 ms | int8 VNNI-transform load path spills |
| QK of block j+1 issued before softmax/PV of block j | 1.03-1.07 ms | no overlap gained |
| FMA-folded softmax + mask only on the tail block | 1.11-1.17 ms | duplicated loop body spills |
| FMA-folded softmax, single loop | 0.98 ms | no change |
| BLOCK_N = 16 or 64 | 1.1-1.3 ms | slower |
| int8 P and V | 1.34 ms | slower (and less accurate) |
| per-row Q scale (1.27% -> attn error vs 1.49%) | 1.6-2.9 ms | IGC spills 3-4.5 KB in every formulation tried |
| per-token K scale (1.34% vs 1.49%) | +5% | kept out; the speed is the target |

### The rest of the pipeline

Five launches, each timed with `do_bench`:

| launch | B70 | notes |
| --- | --- | --- |
| tables (RoPE x2 + all camera matrices) | 12.6 us | was 3 matrix + 2 table launches (~24 us); `TB=8`, 8 warps -- `TB>=32` spills |
| prologue (Q transform + K centre) | 36 us | the centre's serial loop is ~22 us of it and hides under the Q tiles |
| KV (K int8 + V transform) | 295 us | ~157 MB at ~530 GB/s, ~88% of the ~600 GB/s copy roof |
| attention | 1025 us | |
| output projection | 21 us | the reference's `_transform_kernel`, launched without its autotuner |

Two traps, both measured:

- **Gathering the RoPE `(cos, sin)` inside the transforms** instead of reading a
  table removes two launches but repeats the gather for every head: the KV
  launch went to 0.80 ms and the prologue to 0.22 ms. The table is built once,
  in the tables launch.
- **Finishing the K mean from per-split partial sums inside the quantiser**
  (5472 programs each reducing `[16, 128]`) cost +80 us -- more than half the
  quantisation itself. The centre is instead split across 4-channel blocks,
  which are independent, so every program's mean is final.

The centre is sampled (one 64-token tile per 512 tokens). Against the full mean
the RMS error moves by at most 0.002% on the fixtures; even a single tile only
moves it by 0.01%, but sampling every camera keeps it robust to real data.

### RoPE precision for Q and K

The reference reproduces each `rope_dtype` exactly: real fp64 FMAs for
`float64`, fp16 rounding emulated on the bit pattern for `float16` (Triton
folds `x.to(fp16).to(fp32)` away), and a bfloat16 store between RoPE and PRoPE.
On B70 fp64 is slow and the emulation is ~15 integer ops per element, so the
Sage path paid +230 us (`float64`) and +110 us (`float16`) per call for
precision that int8 quantisation throws away. Measured, production chunk:

| Q/K rotation | float32 / float64 / float16 e2e | vs fp64 | worst head vs ref | gain vs ref |
| --- | --- | --- | --- | --- |
| reference arithmetic, both bf16 rounds | 1386 / 1638 / 1511 us | 1.504% | 1.591% | +0.032% |
| fp32 FMA, fp32 table, no bf16 rounds | 1410 / 1419 / 1405 us | 1.509% | -- | -- |
| **fp16 table, fp32 FMA, no bf16 rounds** (kept) | 1396 / 1406 / 1398 us | 1.509% | 1.600% | +0.029% |
| fp16 table, bf16 rounds kept | 1385 / 1423 / 1408 us | 1.521% | -- | -- |
| bf16 table, fp32 FMA | 1440 / -- / -- us | 1.517% | 1.608% | +0.039% |
| pure fp16 math | 1385 / 1415 / 1419 us | 1.509% | 1.599% | +0.029% |
| pure bf16 math | 1416 / -- / -- us | 1.518% | 1.603% | +0.043% |

The fp64 frequency table is narrowed to fp32 on the host before the fp16 table
is built: gathering fp64 inside the table kernel took 47 us instead of 13.

### Fusing the Q transform and output projection into the attention: rejected

With Q's rotation no longer bound to the reference (previous section), the Q
transform could in principle move into the attention prologue and the output
projection into its epilogue, removing the Q_eff and O round trips (~45 us) and
two launches. Four split K=32 dots would avoid any re-interleave, but int8 DPAS
needs a reduction depth of 32, so that only works at head_dim 128. The general
form interleaves once per program instead: `tl.join` of the four int8 planes
into natural channel order in the prologue, `tl.reshape` + `tl.split` of the
accumulator into planes in the epilogue (then `prope4` + `_pack4`, bit-exact).

| attention, production chunk, BLOCK_M=128 | time | note |
| --- | --- | --- |
| current (+ separate output pass) | 1034 + 22 us | |
| output projection in the epilogue | 1433 us | bit-identical output; 32 KB SLM for the one-time split |
| Q transform in the prologue | 3775 us | Q tile kept in SLM and `local_load`ed every iteration; 6-7 KB spills |
| both | 4228 us | |

The loop IR is unchanged by the epilogue; the one-time reshape/split through
shared memory alone outweighs the pass it replaces (still +255 us at
BLOCK_M=256, where occupancy cannot drop further). On this toolchain the only
efficient bridge between the de-interleaved int64 planes and the DPAS layout is
memory (2D block store + load), which is exactly what the separate Q and output
passes are -- so a scratch round trip inside the attention would move the same
bytes and save at most a launch.

### Profiling with unitrace on this machine

`~/pti-gpu/tools/unitrace/build/unitrace` works, but needs four things that are
not obvious:

```bash
# 1. Triton launches through UR's Level Zero V2 adapter (counter-based events,
#    immediate command lists); unitrace records only the *first* launch of each
#    Triton kernel there. The V1 adapter with regular command lists records all.
export UR_LOADER_USE_LEVEL_ZERO_V2=0 UR_L0_USE_IMMEDIATE_COMMANDLISTS=0
# 2. The integrated UHD 730 (i915) makes unitrace's own zeInit fail; hide it.
export ZE_AFFINITY_MASK=0
# 3. Hardware metrics need this (set it back to 1 afterwards).
echo 0 | sudo tee /proc/sys/dev/xe/observation_paranoid
unitrace -d --stall-sampling python driver.py
```

4. Synchronise the driver with `torch.xpu.synchronize()`: an event-based sync
   (`torch.xpu.Event().synchronize()`) hangs under unitrace here.

Only the `EuStallSampling` metric group is exposed on this driver -- no OA
bandwidth or occupancy groups -- so bandwidth comes from bytes / time against
a measured roof: a descriptor copy kernel moves ~600 GB/s (incompressible data;
small-integer test data compresses and reads as 1.2 TB/s).

Stall profile of the RoPE side (share of samples):

| kernel | active | send (memory) | sbid (dependency) | sync (barrier) |
| --- | --- | --- | --- | --- |
| KV | 36% | 33% | 17% | 2% |
| prologue, before the fix below | 41% | 10% | 21% | 9% |
| reference `_transform_kernel` | 29% | 35% | 20% | 3% |

What that led to:

| change | before | after | bit-exact |
| --- | --- | --- | --- |
| K centre: tile-shaped accumulators, one reduction after the loop | 46 us | 22 us | centre moves 4.5e-8 rel. (summation order) |
| KV: K and V blocks in separate programs | 303 us | 295 us | yes |
| KV: camera matrix as 16 scalars when a tile is inside one camera | 303 us | 303 us | yes -- rejected, the gathers are not the cost |
| KV: num_warps 2 / 4 / 16 | 506 / 332 / 297 us | | kept 8 |

What is left on the RoPE side is ~50-60 us in total: the KV launch is ~35 us
from the roof, and the only structural savings left are fusing the Q transform
into the attention prologue (saves writing and re-reading Q_eff, ~25 us) and
the output projection into its epilogue (~15 us). Both need the 4-channel
mixing done in DPAS layout -- 4 split K=32 dots, or a block-diagonal 128x128
matrix per camera -- inside the one kernel whose register budget is already
exhausted, so neither has been attempted.

## Running it

```bash
pytest test_fused_rope_prope_sage.py              # accuracy tests + benchmark
pytest test_fused_rope_prope_sage.py -s -k bench  # benchmark only, printing its table
python test_fused_rope_prope_sage.py              # the benchmark table alone
```
