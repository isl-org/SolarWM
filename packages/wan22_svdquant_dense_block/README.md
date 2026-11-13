# Wan2.2 SVDQuant Dense Block Kernel Optimization Task

## Workload Description
This task optimizes the core dense GEMM runtime of the **SolarWM Wan2.2 TI2V-5B Stage 2 Camera Transformer Block**.

The target runtime metric is the **sum of execution times across 5 layers**:
1. **Projection 0 (Q)**: \( M = 1215, K = 3072, N = 3072 \), LoRA rank 32, with fused `WanRMSNorm(dim=3072, eps=1e-5)`
2. **Projection 1 (K)**: \( M = 1215, K = 3072, N = 3072 \), LoRA rank 32, with fused `WanRMSNorm(dim=3072, eps=1e-5)`
3. **Projection 2 (V)**: \( M = 1215, K = 3072, N = 3072 \), LoRA rank 32, no normalization
4. **FFN Up-projection**: \( M = 1215, K = 3072, N = 14336 \), LoRA rank 32, activation `gelu_tanh`, with optional fused INT4 output quantization (`fuse_quant_out`)
5. **FFN Down-projection**: \( M = 1215, K = 14336, N = 3072 \), LoRA rank 32, consuming pre-quantized INT4 activation directly

## Key Architectural Directives & Fusions
- **Optional Quantization**: If input \( X \) is already in INT4 (from a preceding layer with fused output quantization, such as FFN Up \(\to\) FFN Down), quantization is bypassed. If input \( X \) is BF16, dynamic INT4 group-64 quantization is performed on the fly inside the SYCL kernel (similar to Nunchaku SVDQuant).
- **Fused LoRA Calculation**: The residual low-rank branch \( (X \times L_A^T) \times L_B^T \) is computed entirely inside the SYCL kernel on the fly in registers, without intermediate allocations or separate PyTorch kernel dispatches.
- **Dual GEMM Execution**: Mainloop executes dual GEMM (INT4 MMA + BF16 LoRA MMA), or XMX/DPAS for INT4 with vector engines for BF16 LoRA in the epilogue.
- **Epilogue Fusions**:
  - Optional `WanRMSNorm` epilogue for query and key projections.
  - Optional non-linearities (`gelu_tanh`, `silu`).
  - Optional fused output INT4 quantization (`qout`, `oscales`) to enable zero-overhead inter-layer chaining.

## Oracles & Reference Implementations
- **Correctness Oracle**: High-precision Triton kernel (`triton_svdquant_w4a4_gemm`) with dynamic INT4 quantization, register LoRA fusion, and `gelu_tanh` epilogue.
- **Comparison Metric**: Direct comparison in `torch.bfloat16` precision.
  - Empirical measurements on Intel Arc B70:
    - Projections (\( K = 3072 \)): max diff in BF16 \(\approx 0.011\) (mean \(\approx 0.001\)).
    - FFN Up (\( K = 3072 \), with `gelu_tanh`): max diff in BF16 \(\approx 0.006\) (mean \(\approx 0.0005\)).
    - FFN Down (\( K = 14336 \)): max diff in BF16 \(\approx 0.018\) (mean \(\approx 0.002\)).
  - Threshold: \( 0.035 \sim 0.05 \) in bfloat16 accounts for IEEE 754 summation reordering across tile boundaries (~1-2 ULPs) while strictly guarding against functional regressions.
- **Runtime Reference Baseline**: PyTorch native BF16 operations (`bf16_linear` / `torch.nn.functional.linear` + `wan_rmsnorm`).

## Portability & Build Requirements
- Download `sycl-tla` headers and applications explicitly before building:
  `git clone --depth 1 --branch v0.9.3 https://github.com/intel/sycl-tla.git sycl-tla`.
  This release resolves to commit `7b56b936b87f799f80dde60c2fd8d11c1128cebb`.
- Install `kernelfoundry` in the target Python environment for the build harness.
- The optional VAE qconv extension needs the matching oneAPI Toolkit oneDNN
  development component (`intel.oneapi.win.dnnl`), including `dnnl.lib` and
  `dnnl.dll` on Windows; PyTorch's bundled headers are not a substitute.
- No machine-specific hardcoded paths are used; include directories are resolved relative to the task root.
- The build harness assumes Intel oneAPI's `icpx` (or Windows `icx`) compiler is
  present in `PATH`.
- Run `python build.py --gpu-arch bmg-g31`; the extension is emitted with
  Python's native-module suffix (`.pyd` on Windows, `.so` on Linux) in this
  package directory.

## GEMM variant pool, static dispatch and runtime autotuning

`kernel.sycl` compiles a pool of GEMM configurations (`KF_GEMM_VARIANTS`) in one extension. Each variant is a
template instantiation of the same fused kernel (INT4 DPAS main loop, LoRA, bias/activation, RMSNorm partial
sums or fused INT4 output quantization) described by `Cfg<MT, NG, SGM, SGN, SWZ, GRF, PIPE, PFD>`:

| Field | Meaning |
| --- | --- |
| `MT` / `NG` | 8-row DPAS blocks / 64-column quantization groups per sub-group |
| `SGM` x `SGN` | sub-groups per work-group along M / N (shared B / A tiles) |
| `SWZ` | work-group order swizzle panel height |
| `GRF` | register mode (128 or 256) |
| `PIPE` / `PFD` | next-group operand prefetch into registers / cache prefetch distance (K-groups) |

The pool kernels are specialized for rank-32 LoRA and the three epilogue families of this task (plain,
RMSNorm partial sums, GELU + INT4 output). Anything else (other ranks, SiLU, ...) runs the generic legacy
configuration.

**Static dispatch** (`static_select`, skill: intel-gpu-kernel-dispatch) queries the device (sub-group sizes,
work-group limit, XVE count) and the *compiled kernel's* work-group limit, drops illegal variants with a
reason (N not a multiple of the column tile, work-group too large, ...), and ranks the rest by hardware class and
shape (deep-K vs wide). The first entry is the deterministic default. Only the order is estimated; legality is
checked, never inferred. Ranking evidence: the Xe3-LPG (96 XVE) order was measured on Panther Lake 12Xe;
other devices (including Arc Pro B70) start from variant 0, the configuration tuned on B70 before this pool
existed. **No B70 was available while this was written**, so B70 behaviour relies on the runtime autotuner.

**Runtime autotuning** (`run_tuned`, skill: runtime-autotune, native `autotune.hpp`) validates and times every
eligible variant as a full operation (GEMM + RMSNorm pass) on the first call of an exact workload, caches the
winner on disk (`~/.cache/sycl/kernelfoundry-autotune`, or `$SYCL_AUTOTUNE_CACHE_DIR`; keyed by build, device,
driver, workload, policy and pool) and memoizes it in-process. Cold selection costs about 0.2-1.4 s per
workload; a disk hit about 20 ms; warm calls pay nothing extra. Validation compares against the static
default's output. If selection itself fails the default is used.

Environment: `KF_AUTOTUNE=0` disables runtime selection (static default only), `KF_VARIANT=<i>` forces variant `i`.
Python diagnostics on the extension module: `gemm_variants()`, `gemm_variant_params()`, `gemm_pool(M, N, K, R, kind)`
(device, default, ranked pool, rejected variants with reasons), `last_selection()` (winner, pool, finalists' median ms),
`selection_stats()` / `reset_selection_stats()` (searches, benchmarks, validations counted by the tuner),
`set_eligible(ids)` (restrict the pool; `[]` clears), `last_used_variant()`, `clear_selection_memo()`.

**Per-shape specialization.** Dispatch and caching are per workload (M, N, K, rank, epilogue, activation, bias, qkv,
INT4 input): the static ranking differs for deep-K versus wide shapes, and each exact workload gets its own cached
winner. On Panther Lake 12Xe the same variant (`mt1_ng1_wg4x4_swz4_grf256_p1_pf0`) is the default and the autotune
winner for all five task shapes, within 0.5% of the best measured variant on each; variant 0 (the previous tuned configuration) was
6-27% slower per shape. Arc Pro B70 was not available: it starts from variant 0 and relies on the runtime autotuner.

## Development tools (`dev/`)

- `ptl_oracle.py`: pytest plugin replacing the Triton oracle with an equivalent torch implementation (the Triton
  XPU backend aborts on Panther Lake drivers: `SPV_INTEL_predicated_io`). Use:
  `PYTHONPATH=dev pytest task.py -p ptl_oracle -k "not benchmark"`.
  Two tests (`..._projection_q_rmsnorm`, `..._dense_block`) exceed the 0.05 tolerance on the Q/K RMSNorm outputs
  with the original kernel and the current one alike: the GEMM error is ~0.01 (same as the oracle's distance to
  exact fp64 math), and the RMSNorm amplifies it by 1/rms(y) ~ 9 on this data.
- `bench.py` (per-layer + block timings), `prof.py`, `run_loop.py` + `vt_table.py` (VTune `gpu-hotspots` tables),
  `asm_stats.py` (instruction mix of the main loop from IGC dumps:
  `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir> python build.py --gpu-arch ptl`).
- The per-layer `test_benchmark_*` tests need a `reference_kernel.sycl` that is not part of this package.
