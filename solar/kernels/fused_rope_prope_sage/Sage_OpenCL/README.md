# SageAttention Causal Inference — Intel OpenCL Port

This directory contains an OpenCL C implementation of SageAttention-style causal inference derived from the SYCL kernel previously located at `solar/kernels/fused_rope_prope_sage/sage_causal.sycl`.

## Files

- `sage_causal.cl` — Intel OpenCL 3.0 kernels for quantization, packing, and attention.

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

### 4. Pack V for P·V DPAS

`sage_pack_v` packs two adjacent FP16/BF16 V values into each `uint`, producing the B-operand structure expected by FP16/BF16 DPAS:

```text
[B, Hkv, SkvPad / 2, D] uint
```

Padding rows contain zeros.

### 5. Stream attention blocks

One 16-lane subgroup owns 8 query rows. For every 16-key tile it:

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
7. Normalizes after the KV loop:

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
| `v_packed` | `[B,Hkv,SkvPad/2,D]` | `uint` | FP16/BF16 DPAS-packed V |
| output | `[B,Sq,Hq,D]` | FP16/BF16 bits in `ushort` | Attention result |

Required padding:

- `SqPad` is rounded up from `Sq` to a multiple of 8 because the attention tile has 8 Q rows.
- `SkvPad` is rounded up from `Skv` to a multiple of 16 because K and V packing use 16-key/2-row packing.
- Allocating `SkvPad` as a multiple of 16 also keeps DPAS K/V tile launches uniform.

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
global = (B * Hq * SqPad)
local  = unspecified/implementation-selected
```

One work-item handles one query token.

### 3. `sage_quantize_k`

```text
global = (B * Hkv * SkvPad)
local  = unspecified/implementation-selected
```

One work-item handles one key token and writes its portion of the packed K layout.

### 4. `sage_pack_v`

```text
global = (Hkv * D, B, SkvPad / 2)
local  = implementation-selected
```

Each work-item packs two adjacent V rows for one batch/head/channel.

### 5. `sage_attention`

```text
global = (ceil(Sq / 8) * 16, B * Hq)
local  = (16, 1)
```

A required subgroup size of 16 is essential because the DPAS interfaces produce 8 rows by 16 columns per call.

The host should enqueue or retain an ordering dependency between stages 1 and 3, and between quantization/packing and stage 5. In-order OpenCL queues already preserve this order; out-of-order queues need events.

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

This OpenCL port intentionally prioritizes a clear and buildable baseline. It does not reproduce every low-level optimization in `sage_causal.sycl`.

Current differences include:

- OpenCL subgroup DPAS builtins instead of inline Gen ISA aliases and `.decl` blocks
- Straightforward per-token quantization rather than subgroup-packed quantization kernels with every token resident in registers
- Direct online-softmax rescaling on each K tile rather than the SYCL lazy-rescale threshold
- No explicit K/V prefetch loop or double-buffer software pipeline
- No explicit 256-GRF launch property
- No large 128/256-row Q work-group tile; this port uses 8-row subgroup tiles
- No integrated CUTLASS epilogue
- No Python/PyTorch launcher included in this folder

These differences make the OpenCL version easier to inspect and validate, while retaining the main SageAttention architecture and Intel XMX compute path.

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

The existing compiler reported spills in some initial D=128 quantization/attention variants. All variants still compiled successfully, but performance tuning should begin by inspecting these spills and either:

- increasing register allocation where supported,
- reducing private per-work-item storage,
- splitting quantization work across subgroups,
- or introducing bounded preprocessing tiles.

## Suggested Next Optimization Steps

1. Add subgroup-oriented Q/K quantization to reduce private arrays and global traffic.
2. Add `-cl-intel-256-GRF-per-thread` as an optional build variant and compare spills.
3. Add explicit two-stage K/V software prefetching or block-read scheduling.
4. Add lazy online-softmax rescaling with a tested threshold.
5. Increase V tiles per subgroup after measuring register pressure.
6. Specialize or tile D=64 and D=128 launch configurations independently.
7. Add malformed-shape checks and extension checks to the host launcher.
8. Benchmark edge sizes, GQA ratios, causal/non-causal paths, and sequence lengths.
9. Compare end-to-end latency and numerical quality against `sage_causal.sycl` or a full-precision FlashAttention baseline.
10. Add permanent CPU-reference tests to the repository rather than relying on an external temporary harness.

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
