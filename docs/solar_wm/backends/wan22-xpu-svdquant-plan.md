# Wan2.2 TI2V-5B SVDQuant W4A4+BF16 Optimization Plan & Progress

**Status:** Completed Phase 1 (Anchors & Sidecar Format) and Phase 2 (B70 Triton Feasibility). Rejection gate triggered; offline calibration and runtime integration halted on null result.
**Target Device:** Intel Arc B580 / B70 (`Intel(R) Graphics [0xe223]`), PyTorch 2.12.1+xpu.
**Calibration Host:** `isl-gpu59.rr.intel.com`, NVIDIA RTX 6000 Ada (48 GB), PyTorch 2.5.1+cu124.
**Model Target:** SolarWM Wan2.2 TI2V-5B Stage2 camera-length inference (`CausalWanModel`).
**Precision Contract:** Symmetric INT4 weights + dynamic INT4 activations + rank-16 BF16 residual branch (`W4A4+BF16`).
**Base Checkpoint:** `/home/ssheorey/models/SolarWM/SolarWM-5B-sgf-stage2-81f` (`model.pt`, 39,998,928,500 bytes).

---

## 1. Scope & Invariants

- **Architecture:** Wan2.2 TI2V-5B DiT (`dim=3072`, `ffn_dim=14336`, `num_heads=24`, `head_dim=128`, 30 blocks).
- **Target Linears:** Exactly 300 block `nn.Linear` layers:
  - Per block (10): `self_attn.{q,k,v,o}`, `cross_attn.{q,k,v,o}`, `ffn.0`, `ffn.2`.
  - Excluded from quant (BF16): Global embeddings (`text_embedding`, `time_embedding`, `time_projection`, `head`), LayerNorm/RMSNorm, modulation, patch embedding (`Conv3d`), attention cores.
- **Rollout Schedule:** NFE4 (`denoising_step_list = [1000, 750, 500, 250]`) + 1 commit pass per chunk. Total 5 forward passes per chunk.
- **Active Shapes:**
  - Token sequence length $S = 1215$ (3 latent frames $\times 405$ tokens/frame).
  - Attention projections: $[1215, 3072] \times [3072, 3072]$.
  - FFN projections: $[1215, 3072] \times [3072, 14336]$ and $[1215, 14336] \times [14336, 3072]$.

---

## 2. Workstreams & Stages at a Glance

| Stage | Description | Gate / Target | Status | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **WS1** | Anchors, exact timesteps, sidecar format | Fail-closed specification | **Done and verified** | Manifest schema & ABI locked |
| **WS2** | B70 Triton INT4+BF16 kernel feasibility | Speedup $\ge 1.4\times$ on GEMM path | **Rejected** | Triton 3.4ms vs oneDNN 0.17ms (0.05x); HW ceiling 1.40x |
| **WS3** | NVIDIA DeepCompressor INT4+GPTQ calibration | SVD r=16 + balanced Hessian | **Blocked / Halted** | Pre-calibration feasibility gate failed |
| **WS4** | Pseudo-QDQ reference & safetensors loader | Parity vs exported sidecar | Not started | - |
| **WS5** | SolarWM Stage2 runtime integration | Config flag + fail-closed hook | Not started | - |
| **WS6** | Sensitive layer search & quality gates | PSNR $\ge 30\text{ dB}$, SSIM $\ge 0.95$ | Not started | - |
| **WS7** | Fresh-process B70 validation & documentation | $\ge 5\%$ chunk speedup, $-6.6\text{ GB}$ VRAM | Not started | - |

---

## 3. Sidecar Specification & Fail-Closed ABI

The SVDQuant sidecar uses sharded `safetensors` alongside a top-level `manifest.json`.

### Tensor Naming Convention per Linear Layer (`<module_name>`):
- `<module_name>.qweight`: `torch.int8` or `torch.uint8` tensor of shape `[OC, IC // 2]`. Signed 4-bit values packed 2 per byte (low nibble = first element).
- `<module_name>.w_scale`: `torch.bfloat16` tensor of shape `[OC, IC // group_size]`. Default `group_size = 64`.
- `<module_name>.smooth_scale`: `torch.bfloat16` tensor of shape `[IC]`. Per-channel static input smoothing factor ($X_s = X / d$, $W_s = W \cdot d$).
- `<module_name>.lora_a`: `torch.bfloat16` tensor of shape `[rank, IC]`, where `rank = 16`.
- `<module_name>.lora_b`: `torch.bfloat16` tensor of shape `[OC, rank]`.
- `<module_name>.bias`: `torch.bfloat16` tensor of shape `[OC]` (if original linear has bias).

### Activation Policy (`manifest.json`):
- Timestep bins: 4 denoise steps mapped to step indices `[0, 1, 2, 3]` and commit pass `step = 4`.
- Per-step clipping ratio `clip_ratio[step] \in {0.6, 0.7, 0.8, 0.9, 1.0}`.
- Exponent offset `exp_offset[step] \in {-1, 0, 1}`.

---

## 4. B70 Triton Kernel Feasibility Findings (WS2)

Detailed hardware measurements on Intel Arc B580 / B70 (`Intel(R) Graphics [0xe223]`) with `PyTorch 2.12.1+xpu` and `Triton 3.2.0`:

1. **Bare Hardware Matmul (M=1215, K=3072, N=3072):**
   - Native oneDNN BF16 GEMM: **0.170 ms**
   - Native oneDNN INT8 GEMM: **0.122 ms** (Theoretical maximum hardware speedup: **1.39×**)
   - Pure Triton BF16 GEMM: **0.952 ms** (5.6× slower than oneDNN)
   - Pure Triton INT8 GEMM: **0.890 ms** (7.3× slower than oneDNN)

2. **Custom SVDQuant Triton Kernel with On-the-Fly INT4 Unpacking & Scaling:**
   - Unoptimized fused kernel (with inner-loop dynamic quant): **5.850 ms** (0.03× vs BF16)
   - Fully optimized fused kernel (pre-quantized INT8 activations + bitwise unpack + group-64 scale accumulation): **3.399 ms** (**0.05× vs BF16 — 20× slower**)

3. **Auxiliary Pipeline Overheads per Linear:**
   - Dynamic INT4 per-token activation quantization: **0.110 ms** (PyTorch) / **0.373 ms** (Triton)
   - Rank-16 BF16 Low-Rank residual branch: **0.029 ms**
   - Sum of minimum auxiliary overheads alone ($0.110 + 0.029 = 0.139\text{ ms}$) is **82% of the entire 0.170 ms budget of the BF16 Linear layer**.

4. **Conclusion & Decision:**
   - **Gate Failed:** The required $\ge 1.4\times$ GEMM path speedup is physically unreachable through Triton on Intel Arc Xe2 because Triton’s tile compiler for Intel XPU does not match oneDNN assembly, taking ~3.4 ms vs 0.17 ms.
   - Even assuming a theoretical 0-cost Triton compiler, the bare hardware INT8 speedup (1.39×) minus the quantization and rank-16 overheads results in net negative performance.
   - Under the PyTorch Pipeline Optimization measurement contract and Phase 2 Go/No-Go gate ("*Proceed to DeepCompressor only if the projected five-forward saving is at least 134 ms (weighted projection path about ≥1.4×)... Record and stop on a null result rather than calculating quantized weights for a nonviable runtime.*"), this path is definitively rejected and stopped on this null result.

---

## 5. Decision & Rejection Documentation

In accordance with Phase 2 Go/No-Go rules:
- **Baseline GEMM Cost (300 linears x 5 forward passes):** $453.5\text{ ms}$
- **Target Saving:** $\ge 134.0\text{ ms}$ (threshold $\ge 5\%$ chunk speedup)
- **Triton SVDQuant Measured Latency (300 linears x 5 passes):** $\approx 9,145\text{ ms}$ ($20\times$ slower than BF16 baseline)
- **Status:** WS2 completed with verified benchmark evidence; subsequent calibration (WS3) and runtime integration (WS4-WS7) correctly aborted as infeasible on Arc Xe2 Triton.

