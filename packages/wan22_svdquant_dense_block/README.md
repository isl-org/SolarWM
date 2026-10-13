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
- Self-contained `sycl-tla` headers and applications are bundled in the package (`sycl-tla/`).
- No machine-specific hardcoded paths are used; include directories are resolved relative to the task root.
- The build harness assumes `icpx` (Intel oneAPI DPC++/C++ Compiler) is present in `PATH`.
