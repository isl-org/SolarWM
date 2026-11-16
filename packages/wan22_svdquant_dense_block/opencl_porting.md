# OpenCL port of the SVDQuant W4A4 projection: results and status

Device: Intel(R) Graphics [0xb080] (Panther Lake, 12 Xe cores, 96 XVE), driver 25.35.35096.9. Shared GPU: run-to-run
spread is 10 to 20 %, so every number below is a range over repeated runs, not a single measurement.

## Result in one table

Operation timed everywhere (identical): bf16 x (M = 1215) -> dynamic INT4 group-64 quantization -> LoRA-A (rank 32) ->
INT4 GEMM -> LoRA-B -> bias -> bf16 output. No activation, no fused output quantization.
Method: wall-clock per call over 20 back-to-back calls with one synchronization at the end, minimum of 7 batches,
repeated in separate processes (`opencl/bench_end_to_end.py`). Inputs: `opencl/prepare_v_inputs.py`.

| Implementation | Final variant | v: N = K = 3072 | wide: N = 14336, K = 3072 |
|---|---|---|---|
| SYCL original (kernel as received, commit `0c74e96`) | one configuration (4 x 4 sub-groups, 128 GRF, no pipelining) | 0.81 to 0.83 ms (2 runs) | 2.11 to 2.17 ms |
| SYCL current (variant pool, static default = autotune winner) | `mt1_ng1_wg4x4_swz4_grf256_p1_pf0` (4 x 4 sub-groups, swizzle 4, 256 GRF, pipelined) for both shapes | 0.64 to 0.79 ms (7 runs) | 1.98 to 2.12 ms |
| OpenCL first port (one fixed variant) | SGM 4, SGN 4, PIPE 0, GRF 256 | 0.70 to 0.72 ms (3 runs) | 8.26 to 8.48 ms |
| **OpenCL, static dispatch + runtime autotuning** | v: SGM 4, SGN 1, PIPE 0, GRF 128. wide: SGM 8, SGN 1, PIPE 0, GRF 128 | **0.555 to 0.565 ms** (2 runs) | **4.55 ms** (2 runs) |

How to read it:
- v shape: the tuned OpenCL result is below every SYCL run (0.56 vs 0.64 to 0.79 ms). The SYCL figure includes torch-side
  output allocation and Python dispatch on every call; the OpenCL path pre-allocates. Treat the gap as real but not
  a clean kernel-to-kernel comparison.
- wide shape: **OpenCL is 2.1 to 2.3 times slower than current SYCL** (4.55 vs 1.98 to 2.12 ms), and the single
  fixed variant is 4 times slower. The OpenCL port does not match SYCL on the wide shape.
- SYCL static default and SYCL autotune choose the same variant on both shapes; differences between the two modes in
  the table are run-to-run noise.

Optimal variant definitions. SYCL `Cfg<MT, NG, SGM, SGN, SWZ, GRF, PIPE, PFD>`: MT = 1 (8-row blocks per sub-group),
NG = 1 (64-column groups per sub-group), SGM x SGN = 4 x 4 sub-groups per work-group, SWZ = 4 (work-group order
swizzle panel), GRF = 256, PIPE = 1 (next-group operand loads before the current DPAS), PFD = 0 (no cache prefetch).
OpenCL `KF_SGM, KF_SGN, KF_PIPE, KF_GRF` (`svdquant_w4a4_v_layer.cl`): sub-groups per work-group along M and N, pipelining,
register mode (256 = `-cl-intel-256-GRF-per-thread`, 128 = default allocation).

## Correctness

- INT4 GEMM alone (plain epilogue): bit-exact against SYCL on the v shape (max |difference| = 0).
- Full operation, tuned OpenCL variants against the saved SYCL outputs: 99.56 % of bf16 values bit-identical on both shapes;
  largest difference 0.0039 (v) and 0.0044 (wide) at a largest output magnitude of 0.625, about one bf16 step. The
  difference comes from the LoRA-A summation order, not from the INT4 GEMM. Quantization equality with the SYCL prologue was not
  checked separately.

## Static dispatch and autotuning (intel-gpu-kernel-dispatch, runtime-autotune)

- Pool (metadata block in the kernel): KF_SGM {1,2,4,8} x KF_SGN {1,2,4,8} x KF_PIPE {0,1} x KF_GRF {128,256} = 64
  builds; **61 legal**. The 3 rejected (8 x 8 sub-groups) exceed the compiled kernel's work-group limit
  (`CL_KERNEL_WORK_GROUP_SIZE`: 512 for the 256-register build, 768 for the default build). Compiling all 64 takes about 12 s.
- Legality checks (each with its source): per-axis and total work-group size against the device, compiled work-group
  limit, compile-time work-group size, static local memory, sub-group size 16, N and K multiples of 64, even group count
  for the pipelined build.
- Measured ranking (full operation, finalists = pool, 2 rounds, `sweep_ocl_results.json`, tuner timing method so absolute
  values differ slightly from the table above):

| Variant (SGM x SGN, PIPE, GRF) | v | wide |
|---|---|---|
| best: v 4 x 1, wide 8 x 1, PIPE 0, GRF 128 | 0.576 ms | 5.11 ms |
| 4 x 4, PIPE 0, GRF 128 | 0.591 ms | 9.08 ms |
| 4 x 4, PIPE 0, GRF 256 (first port) | 0.731 ms | 9.22 ms |
| best with GRF 256 | 0.661 ms | 5.57 ms |
| best with PIPE 1 | 0.873 ms | 7.51 ms |
| slowest of 61 | 2.96 ms | 24.1 ms |

- The default register allocation without pipelining was best on both shapes. The sub-group shape that wins differs by
  shape (4 x 1 square, 8 x 1 wide), so a single default is wrong for one of them; ranking is stored per shape class in
  `opencl/static_priority_ocl.json` (hardware class `xve96`).
- Scorecard (run with the skills' evaluator; the adapter and result files are not kept in this repository): c1 to c5 all passed (61 legal candidates run correctly when forced alone,
  full-pool cold search with every candidate benchmarked and validated, lifecycle, bucket reuse: M = 1100 and 1215 share a
  search, M = 900 searches separately). Self-test: all 8 injected faults detected.

## SPIR-V and ISA comparison (OpenCL first-port variant vs SYCL `Cfg<1,1,4,4,4,256,1,0>`, plain epilogue)

Both modules were built with `ocloc compile -device ptl` and disassembled with `ocloc disasm`. **The binaries are not identical.**

| | OpenCL | SYCL |
|---|---|---|
| SPIR-V | 1,585 instructions (3 kernels) | 23,778 instructions (42 kernels; not comparable in size) |
| Matrix instruction source | `intel_sub_group_*_matrix_mad` builtin calls | inline vISA asm blocks |
| GEMM kernel, ISA instructions | 1,305 | 1,178 |
| INT4 / bf16 DPAS | 4 / 8 | 8 / 8 (SYCL unrolls two pipelined stages) |
| mov / add / mach+macl | 201 / 282 / 134 | 436 / 145 / 33 |

Same `dpas.8x8 ... :s4` instruction and operand layouts; different scheduling and address arithmetic. The OpenCL variant
compared here is not the final tuned one.

## Not ported or not measured

- Only the plain projection is ported (rank 32): the Q/K RMSNorm epilogue, GELU + fused INT4 output, the INT4-input
  FFN-down path and the fused QKV GEMM are not.
- Work-group swizzle and cache-prefetch distance have no OpenCL counterpart yet. The prefetch builtins compile but were
  not used. Missing swizzle and prefetch are a plausible cause of the wide-shape gap (the 8 x 1 variants that share B
  across M are 1.8 times faster than 4 x 4 there), but this was **not tested**.
- Cold selection cost (first call) and warm cost were not measured separately.
- Validation oracle for the autotuner is the static default's output (as in the SYCL port); the scorecard checks
  correctness against the SYCL reference.
- Arc Pro B70 was not available: nothing was measured there.

## Files

See `opencl/README.md`. Reproduce: `python opencl/prepare_v_inputs.py`, `python opencl/sweep_ocl.py`,
and `python opencl/bench_end_to_end.py <impl>` (commands in `opencl/README.md`).
