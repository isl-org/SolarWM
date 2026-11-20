# SageAttention Causal Inference — Intel OpenCL Port

This directory contains an OpenCL C port targeting the exact behavior and data-layout contract of `solar/kernels/fused_rope_prope_sage/sage_causal.sycl`.

## Files

- `sage_causal.cl` — Intel OpenCL 3.0 quantization and attention kernels.
- `bench_opencl.cpp` — native OpenCL runner with event timing, fixed fixtures, and dump support.
- `compare_sycl.py` — direct output-bit comparison with the installed SYCL extension.
- `check_intermediates.py` — validation for centering, quantization, packing, and score intermediates.

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

`sage_k_centre` estimates one center vector per batch/KV-head:

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

One 16-lane subgroup owns 16 query rows. Each D=128 work-group has 16 subgroups, so the Q tile is 256 rows; D=64 uses 8 subgroups and a 128-row Q tile. For every 32-key block (64 keys at D=64) it:

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

## Launch Sequence

The kernels must run in dependency order on the same queue:

### 1. `sage_k_centre`

```text
global = (B * Hkv * D)
local  = (D)
```

One work-group handles one batch/KV-head. Each local work-item owns one channel.

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
- subgroup reductions and required subgroup sizing

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

Q and K padding are explicit. Padded V rows are packed as zeros, and padded score columns are assigned negative infinity. This keeps edge tiles structurally identical to full tiles and avoids divergent DPAS execution.

### Causal mask and bottom-right alignment

A query row `i` can only see key `j` when:

```text
j <= i + Skv - Sq
```

This convention supports chunked prefill or an inference path where the KV sequence includes earlier context but Q represents a trailing chunk.

### Compile-time specialization

Head dimension and input precision are macros rather than runtime values. This permits static allocation, loop unrolling, dead-code elimination, and shape-specialized address calculations.

## Differences from the SYCL Version

The OpenCL port now follows the exact SYCL data layouts and arithmetic path, but is still implemented in OpenCL C rather than CUTLASS templates or inline Gen ISA.

Current differences include:

- OpenCL subgroup DPAS builtins instead of inline Gen ISA aliases and `.decl` blocks
- OpenCL source-level transformations rely on IGC rather than SYCL/CUTLASS compile-time tensor layouts
- No effective explicit K/V software pipeline is retained; OpenCL exposes no equivalent 2D V prefetch intrinsic for the transformed load path
- The 256-GRF property is supplied by the host as `-cl-intel-256-GRF-per-thread` rather than embedded in the kernel
- No integrated CUTLASS epilogue or Python/PyTorch launcher in this directory

The retained B70 launch uses 16 subgroups for D=128 and 8 for D=64. Native GQA maps each Q head to `Hq / Hkv` KV-head groups. The benchmark uses raw `[B,S,H,D]` V input; the V 2D base starts at the selected batch/KV-head channel while its pitch remains `Hkv * D` elements.

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
  -options '-cl-std=CL3.0 -cl-mad-enable -DSAGE_HEAD_DIM=128 -DSAGE_INPUT_BF16=1'
```

A small causal GQA BF16 case was also run on the available Intel OpenCL GPU and compared against an FP32 CPU reference:

```text
max absolute error: 8.49828e-4
max relative error: 0.629368 (dominated by values close to zero)
```

The low absolute error is the more useful indicator here because BF16 output quantization and INT8 score approximation amplify relative error near zero.

The native benchmark is built and run without PyTorch:

```bash
g++ -O2 -std=c++17 bench_opencl.cpp -lOpenCL -o /tmp/bench_opencl
/tmp/bench_opencl --sq 2048 --sk 2048 --hq 32 --hkv 4 --large-grf --warmup 12 --iters 40
python compare_sycl.py --sq 256 --sk 257 --hq 8 --hkv 4 /tmp/opencl-fixture
```

### Exactness

Direct comparisons use identical dumped binary Q/K/V fixtures and the installed `sage_causal.so`:

- `Sq=256, Skv=257, Hq=8, Hkv=1`, BF16, causal: 0 differing bits / 262144.
- `Sq=256, Skv=257, Hq=8, Hkv=4`, BF16, causal: 0 differing bits / 262144.

The Hkv>1 test specifically validates the raw V base and interleaved-head pitch. `check_intermediates.py` also validates K centering, Q/K scales, packed Q/K operands, and raw DPAS scores.

### Performance and ISA snapshot

On Arc Pro B70 with BF16, D=128, 32 Q heads, 4 KV heads, causal `S=2048`, and 256-GRF mode, the retained version measured approximately:

```text
k_centre   0.087 ms
quant_q    0.078 ms
quant_k    0.017 ms
attention  1.461 ms
total      1.643 ms
```

The reference SYCL kernel is approximately 0.50 ms end-to-end in this environment, so the OpenCL port remains slower. Replicating CUTLASS/TLA source and lowering details produced a useful reduction but did not close the gap:

- Coalesced K8 subgroup loads: 160 → 112 `load.ugm` instructions and spill 1280 → 1152 bytes.
- Hoisted row reciprocals: 131 → 19 `math.inv` instructions.
- Attention shader: about 19.6K → 14.6K ISA lines.
- Retained 2K timing improved from about 1.74 ms to 1.64–1.65 ms total, depending on run conditions.

VTune still reports roughly 100% theoretical occupancy; throughput is therefore primarily limited by generated instruction volume and dependency structure rather than launch occupancy.

## Suggested Next Optimization Steps

1. Reduce exp2/conversion expansion and compare probability-fragment layouts.
2. Analyze the remaining D=128 spill set rather than broadly shrinking the kernel tile.
3. Reproduce CUTLASS's V 2D copy scheduling only where OpenCL can emit the same block messages.
4. Re-check exactness for D=64, FP16, and non-causal cases after any arithmetic-layout change.
5. Add malformed-shape and extension checks to a permanent launcher.
6. Keep strict Hkv=1 and Hkv>1 bit-comparison fixtures in the regression workflow.

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
