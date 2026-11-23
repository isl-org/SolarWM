# SageAttention Causal Inference — Intel OpenCL Port

This directory contains an OpenCL C port targeting the exact behavior and data-layout contract of `solar/kernels/fused_rope_prope_sage/sage_causal.sycl`.

## Files

- `sage_causal.cl` — Intel OpenCL 3.0 quantization and attention kernels.
- `bench_opencl.cpp` — native OpenCL runner with event timing, fixed fixtures, and dump support.
- `compare_sycl.py` — direct output-bit comparison with the installed SYCL extension.
- `check_intermediates.py` — validation for centering, quantization, packing, and score intermediates.
- `accuracy_real.py` — accuracy and speed of OpenCL build variants against SYCL and exact FP32 on captured Qwen3-Omni activations (`/home/gta/omni_bench/act_capture`, from `capture_qkv.py`). Usage: `env.sh python accuracy_real.py out.json name="-DOPTS" ...`.
- `regress.sh` — bit-exact OpenCL-vs-SYCL regression matrix, built in the SYCL-exact configuration (`-DSAGE_FMA_SCORE=0 -DSAGE_MAX_SKIP=0`) (B>1, GQA, D=64/128, FP16/BF16, causal/non-causal, ragged and tile-aligned lengths, Skv ≥ 4096 for the work-group-sync path).

## Scope

The kernel implements forward scaled dot-product attention for inference:

$$
O = \operatorname{softmax}\left(\frac{QK^{T}}{\sqrt D}\right)V
$$

with optional bottom-right-aligned causal masking:

$$
j \le i + S_{KV} - S_Q
$$

Supported features:

- FP16 or BF16 input bits, selected at compile time
- Head dimension 64 or 128
- Grouped Query Attention (GQA), where `Hq % Hkv == 0`
- Contiguous `[B, S, H, D]` Q/K/V layout
- Causal or non-causal operation
- INT8 Q·Kᵀ using Intel XMX/DPAS
- FP16/BF16 P·V with FP32 accumulation
- Online tiled softmax without a materialized attention matrix

Despite the original directory name, this source does not compute RoPE or ProPE. Q and K are expected to have already received any positional transformations.

## Overall Algorithm

### 1. Estimate a K center

`sage_k_centre` estimates one center vector per batch/KV-head (16 subgroups per work-group, partial sums reduced through local memory in the same order as SYCL, so `k_mean` is bit-identical):

$$
k_m = \operatorname{mean}_j K_j
$$

Sequences shorter than 2048 use every key. Longer sequences sample a 64-row tile every 512 rows, matching the sampling strategy of the source SYCL path.

K is centered before quantization:

$$
K'_j = K_j - k_m
$$

For a particular query row:

$$
Q_i K_j^T = Q_i(K_j-k_m)^T + Q_i k_m^T
$$

The second term is constant across keys. Softmax cancels a row-constant offset:

$$
\operatorname{softmax}(s_j+c)=\operatorname{softmax}(s_j)
$$

K smoothing therefore improves INT8 range utilization without requiring a separate score correction.

### 2. Quantize Q per token

`sage_quantize_q` computes each Q token's absolute maximum:

$$
a_{\max,i}=\max_d |Q_{i,d}|
$$

and symmetric INT8 values:

$$
Q_{8,i,d}=\operatorname{rint}\left(Q_{i,d}\frac{127}{a_{\max,i}}\right)
$$

The reconstruction scale is:

$$
q_{s,i}=\frac{a_{\max,i}}{127}
$$

The ordinary softmax scale and base-2 exponential conversion are folded into this scale:

$$
q_{s,i}\leftarrow q_{s,i}\frac{\log_2(e)}{\sqrt D}
$$

The attention loop can consequently use `exp2` directly.

### 3. Quantize and pack K per token

`sage_quantize_k` centers each key, finds its own absolute maximum, and quantizes symmetrically to INT8:

$$
K_{8,j,d}=\operatorname{rint}\left((K_{j,d}-k_m[d])\frac{127}{a_{\max,j}}\right)
$$

The resulting values are packed directly into Intel INT8 DPAS B-operand form:

- 16 key columns per tile
- 32 channels per DPAS K step
- Four adjacent INT8 channels packed into each `uint`
- Eight `uint` values per DPAS key column

The packed shape is represented as:

```text
[B, Hkv, SkvPad / 16, D / 32, 8, 16] uint
```

### 4. Stream attention blocks

One 16-lane subgroup owns 16 query rows. Each D=128 work-group has 16 subgroups, so the Q tile is 256 rows; D=64 uses 8 subgroups and a 128-row Q tile. For causal attention each subgroup stops after the last KV tile its own rows can see. For every 32-key block (64 keys at D=64) it:

1. Computes an INT8 DPAS score fragment:

   $$
   I_{ij}=Q_{8,i}\cdot K'_{8,j}
   $$

2. Reconstructs scores in FP32:

   $$
   S_{ij}=q_{s,i} k_{s,j} I_{ij}
   $$

3. Applies padding and optional causal masking.
4. Updates online-softmax maximum and denominator state:

   $$
   m_i=\max(m_i,\max_j S_{ij})
   $$

   $$
   P_{ij}=2^{S_{ij}-m_i}
   $$

5. Rescales existing P·V accumulators when the running maximum increases.
6. Accumulates P·V with FP16/BF16 DPAS and FP32 accumulators.
7. Loads raw V through Intel's 2D transform block read, which produces the VNNI packing required by P·V DPAS without a separate `sage_pack_v` stage.
8. Normalizes after the KV loop:

   $$
   O_i=\frac{\sum_j P_{ij}V_j}{\sum_j P_{ij}}
   $$

No complete score or probability matrix is materialized in global memory.

## Kernel and Buffer Contracts

| Buffer | Shape | Type | Purpose |
|---|---:|---|---|
| Q | `[B,Sq,Hq,D]` | FP16/BF16 bits in `ushort` | Query input |
| K | `[B,Skv,Hkv,D]` | FP16/BF16 bits in `ushort` | Key input |
| V | `[B,Skv,Hkv,D]` | FP16/BF16 bits in `ushort` | Value input |
| `k_mean` | `[B,Hkv,D]` | `float` | Sampled key center |
| `q8` | `[B,Hq,SqPad,D]` | `char` | Quantized Q |
| `q_scale` | `[B,Hq,SqPad]` | `float` | Q reconstruction/log2-attention scale |
| `k8_packed` | `[B,Hkv,SkvPad/16,D/32,8,16]` | `uint` | INT8 DPAS-packed K |
| `k_scale` | `[B,Hkv,SkvPad]` | `float` | K reconstruction scale |
| `q_bound` | `[B,Hq,SqPad]` | `float` | `q_scale · ‖q8‖` per row (max-skip bound) |
| `k_bound` | `[B,Hkv,SkvPad/16]` | `float` | max of `k_scale · ‖k8‖` per 16-key fragment (max-skip bound) |
| V | `[B,Skv,Hkv,D]` | FP16/BF16 bits in `ushort` | Raw V; transformed on load |
| output | `[B,Sq,Hq,D]` | FP16/BF16 bits in `ushort` | Attention result |

Required padding:

- `SqPad` is rounded up to the full Q tile: 256 rows for D=128 or 128 rows for D=64.
- `SkvPad` is rounded up from `Skv` to a multiple of 64, covering the K tile and 16-key packed-K fragments.

Shape constraints:

- `D` is 64 or 128.
- `Hq` is an exact multiple of `Hkv`.
- For causal attention, `Skv >= Sq`.
- Input tensors must be contiguous in `[B,S,H,D]` order.

Kernel arguments added for the max skip, appended after the original ones so earlier indices are unchanged: `sage_quantize_q` arg 7 `q_bound` (out), `sage_quantize_k` arg 8 `k_bound` (out), `sage_attention` args 14 `q_bound` and 15 `k_bound` (in).

## Launch Sequence

The kernels must run in dependency order on the same queue:

### 1. `sage_k_centre`

```text
global = (B * Hkv * 256)
local  = (256)
```

One work-group of 16 subgroups handles one batch/KV-head. Subgroup `s` sums sampled rows `s, s+16, ...`; lane `l` owns channels `CH*l .. CH*l+CH-1` (`CH = D/16`).

### 2. `sage_quantize_q`

```text
global = (B * Hq * QTile, SqPad / QTile)
local  = (QTile, 1)
```

One work-group quantizes one Q tile; each 16-lane subgroup owns 16 rows.

### 3. `sage_quantize_k`

```text
global = (B * Hkv * ceil(SkvPad / 128) * 128)
local  = (128)
```

Eight 16-lane subgroups process up to eight 16-key tiles per work-group and write the packed K layout.

### 4. `sage_attention`

```text
global = (ceil(Sq / QTile) * QTile, B * Hq) work-items
local  = (QTile, 1)
```

The work-group size is 256 for D=128 and 128 for D=64. One required 16-lane subgroup per 16 Q rows performs the DPAS mainloop.

The host should enqueue or retain an ordering dependency between stages 1 and 3, and between quantization and stage 4. In-order OpenCL queues already preserve this order; out-of-order queues need events.

## Compile-Time Configuration

Build the source as OpenCL 3.0 with FP16/BF16 bit interpretation selected by macros.

Add `-cl-intel-256-GRF-per-thread` to every build; the attention kernel is tuned for 256 GRF. Optional macros: `SAGE_Q_RELOAD`, `SAGE_K8_PREFETCH` (2), `SAGE_V_PREFETCH` (0), `SAGE_WG_SYNC` (1), `SAGE_WG_SYNC_MIN_KV` (4096), `SAGE_V_ROWS` (32), `SAGE_HOIST_ONES` (0), and the two numerics switches `SAGE_FMA_SCORE` (1), `SAGE_MAX_SKIP` (1) with `SAGE_SKIP_T` (80 for BF16, 12 for FP16). `-DSAGE_FMA_SCORE=0 -DSAGE_MAX_SKIP=0` gives output bit-identical to `sage_causal.sycl`.

For D=128, FP16:

```text
-cl-std=CL3.0 -cl-mad-enable -DSAGE_HEAD_DIM=128 -DSAGE_INPUT_BF16=0
```

For D=128, BF16:

```text
-cl-std=CL3.0 -cl-mad-enable -DSAGE_HEAD_DIM=128 -DSAGE_INPUT_BF16=1
```

The same options apply for `SAGE_HEAD_DIM=64`.

The target device must support the Intel subgroup matrix multiply-accumulate APIs and the required 16-lane subgroup operations used by:

- `intel_sub_group_i8_i8_matrix_mad_k32`
- `intel_sub_group_f16_f16_matrix_mad_k16`
- `intel_sub_group_bf16_bf16_matrix_mad_k16`
- `intel_subgroup_block_read_transform_u16_k16` and `intel_sub_group_2d_block_prefetch_16b_32r16x2c` (`cl_intel_subgroup_2d_block_io`)
- `intel_work_group_barrier_arrive/wait` (`cl_intel_split_work_group_barrier`)
- `intel_convert_bfloat16_as_ushort` (`cl_intel_bfloat16_conversions`)
- subgroup shuffles, votes and required subgroup sizing

A host implementation should query `CL_DEVICE_EXTENSIONS` before selecting this kernel and provide a non-DPAS fallback for devices without the relevant Intel extensions.

## Optimization Strategies and Intuition

### INT8 Q·Kᵀ

The expensive score GEMM uses INT8 DPAS. On Xe, XMX/DPAS can perform the integer matrix operation at higher effective throughput than FP16 while requiring fewer bytes.

INT8 quantization is used only for scores. P and V remain FP16/BF16 because output-vector errors can accumulate more visibly than score approximation errors after softmax.

### Per-token scales

Each Q and K token receives an independent absolute-maximum scale. A tensor-wide scale could be dominated by one outlier and leave most channels poorly represented. Per-token scaling preserves useful INT8 dynamic range while needing only two scalar reconstruction factors per score.

### K smoothing

K centering removes a common component that does not affect softmax rankings. This concentrates K values around zero and makes limited INT8 resolution more useful.

For long contexts, sampling the center avoids scanning the full K sequence in the center-only stage. Quantization still scans every K token later.

### DPAS-specific operand packing

K and V are written in layouts that allow 16-lane subgroup DPAS operands to be consumed directly. This reduces gather instructions, transposes, and register shuffles in the repeated attention loop.

### Subgroup matrix ownership

A subgroup lane owns one output key column in an 8×16 score tile. For INT8 DPAS, each lane contributes two adjacent channels from every query row and consumes a complete 32-channel fragment for one K column.

The consistent fragment mapping is what permits the result to be used directly for softmax and P·V.

### Online softmax / FlashAttention structure

The kernel stores only row maximum, row sum, one score tile, and output accumulators. Memory use is proportional to a tile rather than the full `Sq × Skv` score matrix.

Processing scores and P·V in the same streaming loop also keeps P fragments hot in registers.

### FP32 accumulation

Although P and V use FP16/BF16 DPAS, output accumulators and normalization state remain FP32. This is important for long-context numerical stability.

### Fused scale and base-2 softmax

The factor `log2(e) / sqrt(D)` is folded into the Q scale. Scores therefore use `exp2` directly, avoiding a repeated scale conversion in the score loop.

### Padding and zero initialization

Q and K padding are explicit. V is not padded: the 2D block read is given the real height `Skv`, so rows past the end read as zero, and padded score columns are assigned negative infinity. This keeps edge tiles structurally identical to full tiles and avoids divergent DPAS execution.

### Causal mask and bottom-right alignment

A query row `i` can only see key `j` when:

```text
j <= i + Skv - Sq
```

This convention supports chunked prefill or an inference path where the KV sequence includes earlier context but Q represents a trailing chunk.

### Compile-time specialization

Head dimension and input precision are macros rather than runtime values. This permits static allocation, loop unrolling, dead-code elimination, and shape-specialized address calculations.

## Differences from the SYCL Version

With `-DSAGE_FMA_SCORE=0 -DSAGE_MAX_SKIP=0` the output is bit-identical to `sage_causal.sycl` (see `regress.sh`). The default build makes two arithmetic changes in the attention loop (see "Attention-loop numerics" below); measured accuracy against exact FP32 is unchanged. The implementation differs in how it gets there:

- OpenCL subgroup DPAS builtins instead of inline Gen ISA `.decl` blocks; `cl_intel_subgroup_2d_block_io` for the V transform loads and V prefetch.
- Causal early exit is per **subgroup** (the SYCL loop is barrier-synchronised, so its bound is per work-group and the leading subgroups run extra fully-masked tiles).
- At D=128 the Q8 A-operands are re-read from L1 for every KV tile instead of staying resident (`SAGE_Q_RELOAD`); keeping them pushes IGC past 256 GRF. D=64 keeps them in registers.
- Work-group lockstep (split barrier per KV tile, `SAGE_WG_SYNC`) is applied only when `Skv >= SAGE_WG_SYNC_MIN_KV` (4096) and only over the tiles every subgroup executes.
- The 256-GRF mode is a host build option (`-cl-intel-256-GRF-per-thread`), not a kernel attribute.
- No Python/PyTorch launcher in this directory.

Native GQA maps each Q head to `Hq / Hkv` KV-head groups. V is consumed raw from `[B,S,H,D]`: the 2D base is the batch/KV-head channel start, the pitch is `Hkv * D` elements and the height is `Skv`.

## Validation Performed

The source was offline-compiled with Intel `ocloc` for the Battlemage target in four configurations:

- D=64, FP16
- D=64, BF16
- D=128, FP16
- D=128, BF16

Typical command shape:

```bash
ocloc compile \
  -file sage_causal.cl \
  -device bmg \
  -options '-cl-std=CL3.0 -cl-mad-enable -cl-intel-256-GRF-per-thread -DSAGE_HEAD_DIM=128 -DSAGE_INPUT_BF16=1'
```

`bench_opencl --check` compares against an FP64 CPU reference. At `B=1, Sq=300, Skv=384, Hq=8, Hkv=4`, BF16, non-causal, the max absolute error is 2.6e-4 and the RMS error is 5.1e-5. (The previous version gave 1.1e-2 here because of the dropped-last-V-row bug below.) Absolute error is the useful measure: BF16 output rounding and the INT8 score approximation inflate relative error near zero.

The native benchmark is built and run without PyTorch (it reads `sage_causal.cl` from the working directory; link `libOpenCL.so.1` directly if the `-dev` symlink is missing):

```bash
g++ -O2 -std=c++17 bench_opencl.cpp -lOpenCL -o /tmp/bench_opencl   # or .../libOpenCL.so.1
/tmp/bench_opencl --sq 2048 --sk 2048 --hq 32 --hkv 4 --large-grf --warmup 100 --iters 200
/tmp/bench_opencl --b 2 --sq 300 --sk 333 --hq 8 --hkv 2 --large-grf --dump /tmp/fx
python compare_sycl.py --b 2 --sq 300 --sk 333 --hq 8 --hkv 2 /tmp/fx
```

### Exactness

`regress.sh` dumps identical binary Q/K/V fixtures from `bench_opencl`, runs the installed SYCL `sage_causal.so` on them and requires **0 differing output bits**. It covers 14 cases: B=1/2, Hkv=1/2/4, D=64/128, FP16/BF16, causal and non-causal, ragged (`Skv=257, 333, 4500`) and 64-aligned (`Skv=1024, 2048, 4096`) lengths, and `Skv ≥ 4096` for the work-group-sync path. All 14 pass.

```bash
./regress.sh [bench binary] [python with torch + XPU]
```

Two bugs in the earlier port were hidden because the published fixtures only used `B=1` and `Skv=257`:

- The V 2D reads passed `width-1, height-1, pitch-1`, but the builtin takes real sizes. This dropped the last V row, which only matters when `Skv` is a multiple of 64. At `Skv=2048` the CPU-reference error was about 30× higher than it should be.
- The V batch offset used `SkvPad` instead of `Skv`. Every batch after the first read the wrong rows, so at `B=2` about half the output bits were wrong.

A third difference was the D=64 FP16 1-ULP mismatch. It came from `k_mean` being summed in a different order from SYCL.

### Performance

Arc Pro B70, BF16, Hq=32, Hkv=4, `-cl-intel-256-GRF-per-thread`, all four stages end-to-end (OpenCL event time) against SYCL `sage_causal.forward` (XPU event time, including its quantizers). Medians of 100 runs after 100 warm-up runs (15/15 at S ≥ 8K):

| S | D | causal | OpenCL ms | SYCL ms |
|---:|---:|:---:|---:|---:|
| 2048 | 128 | yes | 0.40 | 0.44 |
| 2048 | 128 | no  | 0.56 | 0.67 |
| 2048 | 64  | yes | 0.22 | 0.23 |
| 2048 | 64  | no  | 0.33 | 0.42 |
| 4096 | 128 | yes | 1.30 | 1.27 |
| 4096 | 128 | no  | 2.26 | 2.42 |
| 4096 | 64  | yes | 0.78 | 0.73 |
| 4096 | 64  | no  | 1.34 | 1.67 |
| 8192 | 128 | yes | 4.24 | 4.53 |
| 8192 | 128 | no  | 8.16 | 8.98 |
| 16384 | 128 | yes | 17.9 | 17.2 |
| 16384 | 128 | no  | 34.6 | 34.9 |
| 16384 | 64  | yes | 11.1 | 10.0 |
| 16384 | 64  | no  | 21.5 | 26.1 |

The previous version took 1.64 ms (attention 1.46 ms) for the first row. Stage split there is now `k_centre 0.005`, `quant_q 0.076`, `quant_k 0.013`, `attention 0.31` ms.

Short benchmarks are clock-sensitive: 40-iteration runs of sub-millisecond kernels varied by up to 3× before the GPU clock settled. Use ≥100 warm-up iterations.

What moved the attention stage (D=128 causal 2K, in order):

| Change | attention ms |
|---|---:|
| previous version | 1.46 |
| per-subgroup causal KV bound (previously all tiles were processed, the upper half fully masked) | 0.88 |
| lane-distributed row max/rescale, 16×16 transpose-reduce, hardware BF16 conversion, uniform mask branches, block-read Q8 and block-write O | 0.71 |
| float16 vector for the transpose-reduce (an array became select-of-address, so it went to scratch) | 0.59 |
| V 2D prefetch for the current tile + in-loop K8 prefetch | 0.39 |
| Q8 re-read per KV tile at D=128 (spill 2.5 KB → 0) | 0.31 |
| work-group split-barrier lockstep for Skv ≥ 4096 | 16K: 21.1 → 17.9 ms total |

No kernel spills in any of the four `(D, dtype)` specializations.

### Attention-loop numerics (default build)

Two changes reduce the vector instructions per KV tile; both are switchable:

- **`SAGE_FMA_SCORE`**: scores are kept as `x = I·ks`; the row scale goes into the exponent, `P = exp2(fma(x, qs, −max))`, and the block max is taken on `x` (`max qs·x = qs·max x`, with `qs` clamped to `FLT_MIN` for zero rows). This saves a multiply and a subtract per score. Rounding differs from SYCL by one FMA.
- **`SAGE_MAX_SKIP`**: `S_ij = qs_i·ks_j·(q8_i·k8_j) ≤ q_bound_i · k_bound_t` holds exactly (Cauchy–Schwarz on the INT8 vectors). When every row of the subgroup has `bound ≤ running max + SAGE_SKIP_T`, the exact block max, 16×16 transpose-reduce and lazy-rescale check are skipped (about 100 instructions), and P is computed against the running max (`P ≤ 2^T`). Softmax is invariant to the reference, so only rounding changes. With T=80 the FP32 sums stay below 2^116 for Skv ≤ 2^20 and |V| ≤ 2^16. FP16 uses T=12 to keep P below 65504.

**Accuracy on real activations** (Qwen3-Omni thinker, post-RoPE Q/K/V captured with `omni_bench/scripts/capture_qkv.py`, cwe and qa_2 RULER prompts at 62.7K tokens, layers 0/3/12/24/36/47; 1,024 query rows per layer against exact FP32 causal attention):

- Relative L2 error against FP32 is within 4e-6 of SYCL's on every layer (for example L3: 0.12536 vs 0.12536; L24: 0.01657 vs 0.01657).
- Worst per-row cosine drops by at most 2e-4.
- Difference from SYCL output: 2e-5 to 1.2e-3 relative L2.

**Speed on the same layers** (sum over the 12 layers, end-to-end ms; random-data speedups are larger because the bound almost always passes):

| build | sum ms | vs SYCL |
|---|---:|---:|
| SYCL Sage | 2996 | 1.000 |
| SYCL-exact OpenCL (32-row V reads) | 3093 | 1.032 |
| + FMA fold | 3036 | 1.013 |
| + max skip, T=64 | 3025 | 1.010 |
| + max skip, T=80 (default) | 2988 | 0.997 |

The bound is loose on some heads (layer 3; layer 47 at T=64), where the failed check costs about 3%. Random data, previous loop → default: 63,488 Omni 265.0 → 246.9 ms, VL 4K×64K 31.6 → 30.1 ms, VL 4K×256K 139.8 → 132.4 ms.

**Tried and rejected:** hoisting the DPAS "ones" operand out of the loop (−32 instructions per tile but 20% slower at 63K, with no spill), and loading the next tile's `k_bound` one iteration ahead (30% slower, no spill). On this IGC, static instruction counts do not predict long-context time; check every change at 63K.

## Suggested Next Optimization Steps

1. **Causal at D=64 and at 16K (5–10% behind SYCL).** The split-barrier lockstep is limited to subgroup 0's causal bound. Try a per-work-group bound with masking, as SYCL does, above the sync threshold, and measure it against the extra masked tiles.
2. **`quant_q` is now about 20% of a 2K D=128 call (0.076 ms, about 330 GB/s).** Coalesced 2D block loads, or fusing Q quantization into the attention prologue (each Q tile is read by exactly one work-group), would remove a pass over Q.
3. **ALU volume around the DPAS work.** When the max is skipped, a tile now takes about 400 instructions for 52 DPAS. 160 of them are the five per-score operations (int→float, ×ks, fma, exp2, float→BF16), which is close to the floor for this algorithm. The rest is scalar address setup for the K8 loads and Q8 re-reads, plus `sync`. Further large gains would need a different P·V scheme, for example INT8 P·V (twice the DPAS rate, with P quantized per row and V per channel), which needs a new accuracy study.
4. **A tighter skip bound** for the heads where Cauchy–Schwarz is loose (layer 3), or a per-subgroup adaptive check that stops testing after repeated failures.
5. Add malformed-shape and extension checks (`cl_intel_subgroup_2d_block_io`, `cl_intel_split_work_group_barrier`, `cl_intel_bfloat16_conversions`, `cl_intel_subgroup_matrix_multiply_accumulate`) to a permanent launcher.
6. Run `regress.sh` after every change (exact mode) and the real-activation accuracy check for numerics changes; it is the guard for the fixture-blind bugs listed above.

## Important Numerical Notes

This kernel is designed as an inference approximation path, not an exact FP32 implementation. Expected error sources are:

- input values represented as FP16/BF16,
- Q and K reconstructed from per-token INT8,
- P rounded back to FP16/BF16 before P·V,
- output rounded to FP16/BF16,
- sampled rather than full K-center estimation for long sequences.

For validation, use mixed-magnitude inputs, zero and near-zero queries, long K dimensions, GQA, ragged sequence lengths, and causal cases near tile boundaries. Exact equality is not appropriate; use error thresholds and task-level quality checks.

## OpenCL Runtime Requirements

A host-side integration should:

1. Query the Intel OpenCL platform/device and matrix extension support.
2. Build one program per `(SAGE_HEAD_DIM, SAGE_INPUT_BF16)` specialization.
3. Round and allocate padded buffers correctly.
4. Allocate all temporaries from reusable workspace storage.
5. Set each kernel argument with the declared buffer type and shape.
6. Respect stage dependencies.
7. Retain 16-lane subgroup compatibility for `sage_attention`.
8. Verify the build log for register spills before performance claims.
9. Validate numerical output before enabling the approximation in production.
