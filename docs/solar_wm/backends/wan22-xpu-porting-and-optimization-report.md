# SolarWM Wan 2.2 - 5B: Intel XPU Porting and Optimization Report

Porting and optimizing the **SolarWM Wan 2.2 TI2V-5B** camera-controlled world model [1–4] on one
**Intel Arc Pro B70** with native PyTorch XPU. No other GPU was used. The XPU port is the `xpu`
branch of [isl-org/SolarWM](https://github.com/isl-org/SolarWM/tree/xpu) [3].

1. **Part I — Summary**: one section per slide.
2. **Part II — Detailed report**: how things are measured, the porting steps, each optimization,
   quantization methods, and what did not work.
3. **Part III — Reproduction**: scripts, commits, and what could not be reproduced.
4. **References.**

# Part I: Summary

## Summary

* **Goal:** port and optimize SolarWM Wan 2.2-5B on one Intel Arc Pro B70 with native PyTorch
  XPU, toward real time: 12 frames per chunk at 16 FPS = **750 ms per chunk**.
* **480×864:** 3,617 → **1,628 ms per chunk (2.22×)**, 3.32 → 7.37 FPS. Real time needs 2.17× more.
* **320×576:** **779 ms per chunk, 15.40 FPS**, within 4 % of real time.
* **The GPU was already saturated,** so the gains came from removing redundant GPU work, fusing
  bandwidth-bound operations, and low precision, not from host-side scheduling.
* **Our SVDQuant W4A4 kernel is 2.15× faster** than llm-scaler's on the same weights.
* **Compilers did not help:** `torch.compile` and OpenVINO were slower or barely faster.

---

## The model: SolarWM Wan 2.2-5B

* Autoregressive camera-controlled world model [1–4]: first frame + text prompt + 6-DoF camera
  trajectory → video, generated chunk by chunk.
* **DiT** (5B, 30 layers): per chunk, 4 denoise forwards + 1 KV-cache commit forward over 3
  latent frames.
* **Causal 3D VAE:** decodes each chunk's 3 latent frames into 12 pixel frames.
* **Camera conditioning:** Projective RoPE (PRoPE) on attention over an 18-latent-frame KV window
  (7,290 tokens at 480p).

---

## Initial port: CUDA → XPU

* XPU-only environment: no CUDA packages; `torch.cuda` → XPU device selection and sync.
* `flash-attn` (CUDA-only) → PyTorch SDPA.
* UMT5-XXL text encoder on the host CPU, so the DiT and VAE fit in 32 GB.
* **Result: DiT 1,898 ms + VAE 1,720 ms = 3,617 ms per chunk (3.32 FPS).**

---

## Optimizations: speedup per step

Speedup of DiT or VAE time over the previous step (480×864). Steps that change that network by
≤ 1.01× are left out. Bar length is proportional to (speedup − 1).

**DiT: 1,898 → 827 ms (2.30×)**

| Optimization | Speedup | |
| --- | ---: | --- |
| Host-int KV-cache indices (no `.item()` syncs) | 1.02× | █ |
| Circular KV cache + PRoPE matrix memo | 1.12× | █████ |
| RoPEd KV ring: RoPE/PRoPE history once per chunk | 1.25× | ██████████ |
| Fused RoPE + PRoPE + SDPA kernel | 1.13× | █████ |
| DiT operator fusions (AdaLN, RMSNorm, GELU, residual) | 1.18× | ███████ |
| Drop per-forward raw KV window copy | 1.10× | ████ |
| Other changes merged with the VAE epilogue fusion (`b66f630`) | 1.03× | █ |
| SVDQuant W4A4 k4 linears | 1.07× | ███ |

**VAE: 1,720 → 817 ms (2.11×)**

| Optimization | Speedup | |
| --- | ---: | --- |
| Elementwise in bf16 | 1.19× | ████████ |
| Channels-last convolutions | 1.06× | ██ |
| Fused Triton RMSNorm + SiLU | 1.27× | ███████████ |
| QuaRot dynamic W8A8 convolutions | 1.18× | ███████ |
| oneDNN residual-add epilogue fusion | 1.11× | ████ |

---

## Final results

| | DiT ms | VAE ms | Chunk ms | FPS | vs 750 ms target |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain port, 480×864 | 1,897.7 | 1,719.5 | 3,617.2 | 3.32 | 4.82× over |
| **Optimized, 480×864** | **811.2** | **816.7** | **1,627.9** | **7.37** | 2.17× over |
| Speedup | 2.34× | 2.11× | **2.22×** | | |
| **Optimized, 320×576** | 374.7 | 404.6 | **779.2** | **15.40** | 1.04× over |

* Intel Arc Pro B70 (Xe2, 32 GB), `torch 2.12.1+xpu`[^softmax], Triton XPU, oneDNN, SYCL.
* Steady-state chunk, median of chunks 5–7 of an 8-chunk rollout; same input and seed for every
  step (Part II §2). 320×576 is a different computation, not a step in the chain.

---

## Quantization: speed and quality

Speed: one 153-frame (13-chunk) rollout at 480×864, GPU locked at 2,000 MHz. Quality: DOVER++
no-reference video quality, overall and its aesthetic and technical scores (0–1, higher is
better), mean over 117 test samples, with the change relative to BF16 in parentheses, for the first chunk
(frames 0–8) and the last chunk (frames 141–152) of 153-frame rollouts. SVDQuant is k4 (4 timestep
weight sets) unless noted. Sage v2 has no SVDQuant weights calibrated for it, so it is shown only
with BF16 linears.

| Configuration | DiT ms | VAE ms | Chunk ms | Chunk × | Overall first | Overall last | Aesthetic first | Aesthetic last | Technical first | Technical last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 reference | 968.9 | 1,138.2 | 2,107.2 | 1.00× | 0.421 | 0.466 | 0.947 | 0.956 | 0.087 | 0.098 |
| SageAttention v1 | 896.4 | 1,136.8 | 2,033.2 | 1.04× | 0.421 (+0.1 %) | 0.463 (−0.5 %) | 0.947 (+0.0 %) | 0.954 (−0.3 %) | 0.087 (+0.2 %) | 0.099 (+0.5 %) |
| SageAttention v2 | 888.6 | 1,137.2 | 2,025.8 | 1.04× | 0.414 (−1.6 %) | **0.299 (−35.8 %)** | 0.947 (+0.0 %) | **0.911 (−4.7 %)** | 0.083 (−4.0 %) | **0.051 (−48.7 %)** |
| SVDQuant W4A4 k1 | 916.9 | 1,136.5 | 2,053.4 | 1.03× | 0.415 (−1.3 %) | 0.445 (−4.5 %) | 0.947 (+0.0 %) | 0.952 (−0.5 %) | 0.084 (−3.3 %) | 0.092 (−6.3 %) |
| SVDQuant W4A4 k2 | 916.1 | 1,136.3 | 2,052.4 | 1.03× | 0.417 (−1.0 %) | 0.437 (−6.1 %) | 0.947 (+0.0 %) | 0.951 (−0.6 %) | 0.085 (−2.4 %) | 0.091 (−7.7 %) |
| SVDQuant W4A4 k4 | 915.6 | 1,136.9 | 2,052.5 | 1.03× | 0.416 (−1.1 %) | 0.438 (−5.9 %) | 0.947 (+0.0 %) | 0.951 (−0.6 %) | 0.084 (−3.0 %) | 0.091 (−7.2 %) |
| QuaRot dynamic W8A8 VAE | 957.0 | 946.1 | 1,903.0 | 1.11× | 0.420 (−0.1 %) | 0.466 (−0.1 %) | 0.947 (−0.0 %) | 0.956 (+0.0 %) | 0.086 (−0.1 %) | 0.098 (−0.1 %) |
| SVDQuant + W8A8 | 916.7 | 937.3 | 1,854.0 | 1.14× | 0.416 (−1.2 %) | 0.438 (−5.9 %) | 0.947 (−0.0 %) | 0.951 (−0.6 %) | 0.084 (−3.1 %) | 0.091 (−7.2 %) |
| Sage v1 + SVDQuant | 841.4 | 1,137.0 | 1,978.4 | 1.07× | 0.417 (−1.0 %) | 0.433 (−6.9 %) | 0.947 (+0.0 %) | 0.948 (−0.8 %) | 0.084 (−2.5 %) | 0.089 (−9.4 %) |
| Sage v1 + SVDQuant + W8A8 | 851.3 | 946.0 | 1,797.3 | 1.17× | 0.416 (−1.1 %) | 0.433 (−7.1 %) | 0.947 (−0.0 %) | 0.948 (−0.9 %) | 0.084 (−2.6 %) | 0.089 (−9.5 %) |

* **Kernel alone** (2,000 MHz) and its relative RMS error vs the BF16 kernel, on real
  steady-state inputs | random Gaussian inputs:
  * Sage v1 attention 1.41×: 2.2–3.6 % | 1.35 %.
  * Sage v2 attention 1.83×: 31–55 % | 17.1 %.
  * SVDQuant, a block's linears 1.16× (Q/K/V 1.16×, FFN 1.16×): Q/K/V 3.5–11 % | 16–22 %, FFN
    11–22 % | 14–22 %.

  Real Q/K are harder for Sage than random data; real activations are easier for SVDQuant, whose
  smoothing and clipping are calibrated on them.
* All speeds here are with the GPU locked at 2,000 MHz. Actual speedups may be lower at the
  default clock because of GPU throttling.
* **vs llm-scaler's SVDQuant kernel:** same weights and input, one complete linear at a time, GPU
  locked at 2,000 MHz: ours is **2.15× faster** on a block's five linears (6.29 → 2.92 ms).
* **Sage v1 and QuaRot W8A8 match BF16** in all three scores. SVDQuant lowers the last chunk
  slightly: overall −0.027 (−5.9 %), aesthetic −0.006 (−0.6 %), technical −0.007 (−7.2 %).
* **Sage v2 collapses over the rollout:** last chunk overall 0.299 vs 0.466 (−35.8 %), aesthetic
  0.911 vs 0.956 (−4.7 %), technical 0.051 vs 0.098 (−48.7 %); 79 of 117 samples are more than 0.1 lower overall. SDPA stays
  the default.
* Ground truth (the dataset's own latents), first / last chunk: overall 0.422 / 0.455, aesthetic
  0.947 / 0.951, technical 0.087 / 0.103. Details: Part II §6.1.

---

## What did not work

| Approach | Result | Why |
| --- | --- | --- |
| Host-sync removal, DiT/VAE stream overlap | 1.00× | The GPU was already saturated, and it serializes the two compute streams |
| `torch.compile` DiT | DiT 0.21× | Recompiles every chunk because cache indices change |
| `torch.compile` VAE | VAE 1.05× | Small gain for a long JIT warmup |
| OpenVINO GPU VAE | VAE 0.71× | Its 3D convolutions lack oneDNN's DPAS `channels_last_3d` path |
| SageAttention v1 | DiT 1.01× | Kernel gain mostly lost to a lower GPU clock |
| Radial sparse attention | DiT 1.01× | No gain at 1,215 queries / 7,290 keys |
| EchoRoPE folded into PRoPE | rejected | Failed the video quality gate |

# Part II: Detailed Report

## 1. SolarWM Wan 2.2 - 5B

### 1.1 Model
SolarWM [1] generates video autoregressively from a first frame, a prompt, and a camera
trajectory (per-frame extrinsics and intrinsics). The backbone is Wan 2.2 TI2V-5B: a causal
diffusion transformer over latents from a causal 3D VAE.

Training route: Stage 0.5 bidirectional flow matching → Stage 1 → Stage 2 DMD distillation via
Self-Gradient Forcing (SGF). Stage 2 samples each 3-latent chunk in 4 denoise steps, then runs one
commit forward at t = 0. The commit writes the chunk into the KV cache for later chunks.

| Parameter | Value |
| --- | --- |
| DiT | 5B parameters, 30 layers, hidden 3072, FFN 14336, 24 heads × 128 |
| Text conditioning | UMT5-XXL, 512 × 4096 |
| VAE | causal 3D, 48 latent channels, 16× spatial, 4× temporal |
| Patch | (1, 2, 2) |
| 480×864 | latent 30 × 54, 405 tokens/frame, 1,215 query tokens, 7,290-token KV window |
| 320×576 | latent 20 × 36, 180 tokens/frame, 540 query tokens, 3,240-token KV window |
| Chunk | 3 latent frames → 12 pixel frames; 5 DiT forwards |
| KV window | 18 latent frames (`local_attn_size`), no sink |
| Real-time target | 12 frames at 16 FPS = 750 ms per chunk |
| Weights | BF16 on Hugging Face [4]; SVDQuant W4A4 in the `ptc` release [5] |

### 1.2 Positional and camera conditioning
* **EchoRoPE:** 3D rotary embedding over (t, y, x). The original code rotates in `complex128`.
* **Projective RoPE (PRoPE):** each token's camera projection matrix is built from the frame's
  extrinsics and normalized intrinsics. Its transpose is applied to queries, and its inverse to
  keys and values.

Both are applied to the whole KV window. Positions are window-relative, so RoPEd history cannot be
reused across chunks, only across the 5 forwards within one chunk.

## 2. What is measured

**Every chunk-level runtime and speedup comes from one measured chain** (Part III,
`scripts/report/run_chain.py`). Each step is the previous step plus one change, measured with the
same script, input sample, seed, and workload. Numbers that are not on this chain say so.

* **Workload:** one steady-state generation chunk at 480×864.
  * **DiT:** 4 denoise forwards + 1 KV-cache commit forward over 3 latent frames.
  * **VAE:** streaming decode of those 3 latent frames into 12 pixel frames.
* **Steady state:** median of chunks 5–7 of an 8-chunk rollout, when the 18-latent KV window is
  full.
* **Same input everywhere:** one preencoded dataset sample (prompt embedding, first latent,
  camera trajectory), seed 1234.
* **Chunk ms = DiT ms + VAE ms**, because the two run serially on the device.
  FPS = 12 frames / chunk time.
* **Speedup** = the reference step's time / this step's time, separately for DiT, VAE, and the
  whole chunk. The reference is the previous step. Alternatives are compared against the step
  they would replace.
* **GPU clock:** default (unlocked) everywhere, except the columns and tables marked
  "@ 2,000 MHz" or "locked at 2,000 MHz".
* **System:** Intel Arc Pro B70 (Xe2, 32 GB), Core Ultra 5 245K, 30 GiB host RAM, Linux 6.18,
  `torch 2.12.1+xpu`, Triton XPU, oneDNN, SYCL. Nothing else runs on the GPU.

## 3. Plain Port (S0, commit `001e586`)

1. **Environment.** `.venv-wan22-xpu` with `torch 2.12.1+xpu`[^softmax] and Triton XPU; no CUDA
   wheels, `flash-attn`, or NCCL.
2. **Device abstraction.** `torch.cuda` usage replaced with XPU device selection and
   `torch.xpu.synchronize()`. Single-GPU inference skips process-group setup.
3. **Attention.** `flash-attn` (CUDA-only) replaced with
   `torch.nn.functional.scaled_dot_product_attention`.
4. **Memory.** The UMT5-XXL encoder runs on the host CPU so the DiT, KV cache and VAE fit on the
   device.
5. **Route.** Production inference uses the camera-length route with a single EMA pass. The
   original route also generated a second pass that was discarded.

Result: **DiT 1,897.7 ms + VAE 1,719.5 ms = 3,617.2 ms per chunk (3.32 FPS).**

## 4. Profiling

The XPU trace showed the DiT and VAE queues busy back to back, and their sum matched the chunk
wall time. The idle gaps in the trace were profiler overhead, not host stalls. Two conclusions
followed:

* Work that only removes host stalls or reorders submission cannot raise throughput. S1 (1.00×),
  S2 (1.01×), and the rejected DiT/VAE stream overlap confirmed this.
* The DiT forward spent more time in RoPE (complex128), PRoPE over the KV window, and PRoPE
  matrix construction than in its GEMMs. All three recompute values that do not change within a
  chunk.

## 5. Optimizations, step by step

Every number is from the chain (Part III). Speedups are over the previous step.

| Step | Commit | Change | DiT ms | DiT × | VAE ms | VAE × | Chunk ms | Chunk × | FPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | `001e586` | Plain port | 1897.7 | | 1719.5 | | 3617.2 | | 3.32 |
| S1 | `b0491d0` | Host-sync removal, streaming VAE decode | 1899.0 | 1.00× | 1719.1 | 1.00× | 3618.1 | 1.00× | 3.32 |
| S2 | `0352a3c` | Host-int cache indices, buffers, uint8 D2H, cached grids (KV cache `clone`) | 1860.0 | 1.02× | 1718.7 | 1.00× | 3578.7 | 1.01× | 3.35 |
| S3 | `0352a3c` | Circular KV cache + PRoPE memo | 1660.5 | 1.12× | 1721.8 | 1.00× | 3382.3 | 1.06× | 3.55 |
| S4 | `4d0c2f9` | VAE elementwise in bf16 | 1661.9 | 1.00× | 1445.9 | 1.19× | 3107.8 | 1.09× | 3.86 |
| S5 | `4d0c2f9` | VAE channels-last convolutions | 1667.5 | 1.00× | 1365.3 | 1.06× | 3032.8 | 1.02× | 3.96 |
| S6 | `4d0c2f9` | Fused Triton VAE RMSNorm+SiLU | 1670.2 | 1.00× | 1077.8 | 1.27× | 2748.0 | 1.10× | 4.37 |
| S7 | `4d0c2f9` | RoPEd KV ring | 1332.7 | 1.25× | 1079.6 | 1.00× | 2412.3 | 1.14× | 4.97 |
| S8 | `4d0c2f9` | Fused RoPE+PRoPE+SDPA kernel | 1180.2 | 1.13× | 1081.1 | 1.00× | 2261.3 | 1.07× | 5.31 |
| S9 | `4d0c2f9` | DiT operator fusions | 997.6 | 1.18× | 1078.8 | 1.00× | 2076.4 | 1.09× | 5.78 |
| S10 | `4d0c2f9` | QuaRot dynamic W8A8 VAE | 985.1 | 1.01× | 912.0 | 1.18× | 1897.1 | 1.09× | 6.33 |
| S11 | `32b1dd9` | Drop per-forward raw KV window copy | 899.5 | 1.10× | 907.8 | 1.00× | 1807.3 | 1.05× | 6.64 |
| S12 | `b66f630` | VAE oneDNN `append_sum` epilogue, QuaRot prologue reuse | 870.5 | 1.03× | 817.0 | 1.11× | 1687.5 | 1.07× | 7.11 |
| S13 | HEAD | HEAD, same configuration | 870.5 | 1.00× | 815.0 | 1.00× | 1685.5 | 1.00× | 7.12 |
| S14 | HEAD | SVDQuant W4A4 DiT linears (KernelFoundry kernel) | 811.2 | 1.07× | 816.7 | 1.00× | 1627.9 | 1.04× | 7.37 |
| | | **Total S0 → S14** | | **2.34×** | | **2.11×** | | **2.22×** | |
| | | **Target** | | | | | **750** | | **16.0** |

S10 and S14 are described with the other quantization methods in §6.

Grouped by kind of change:

| Group | Steps | Speedup per step |
| --- | --- | --- |
| Remove redundant DiT work: circular KV cache, PRoPE memo, RoPEd KV ring, no raw-window copy | S3, S7, S11 | DiT 1.12×, 1.25×, 1.10× |
| VAE bandwidth: bf16, channels-last, fused RMSNorm+SiLU, oneDNN epilogue fusion | S4, S5, S6, S12 | VAE 1.19×, 1.06×, 1.27×, 1.11× |
| Fused DiT kernels: RoPE+PRoPE+SDPA, AdaLN / RMSNorm / GELU | S8, S9 | DiT 1.13×, 1.18× |
| Quantization: QuaRot VAE, SVDQuant DiT | S10, S14 | VAE 1.18×, DiT 1.07× |

### 5.1 Host-side cleanup (S1, S2)
* **S1:** removes per-chunk host/device synchronizations and streams VAE decode per chunk.
* **S2** spans `b0491d0..0352a3c` with the KV cache forced back to `clone` mode, so only the
  mode-independent changes are counted:
  * Cache indices as host ints, and the remaining `.item()` syncs in `echorope_apply` removed
    (WS3, WS16).
  * Preallocated rollout and decode buffers (WS5).
  * GPU-side uint8 quantization of decoded pixels before the device-to-host copy, plus a reusable
    decode session (WS6).
  * Cached scheduler grids and VAE `_scale` tensors (WS17).
* On a saturated device these give 1.02× DiT and no VAE change. Their value is lower memory and
  removed host stalls.

### 5.2 Circular KV cache and PRoPE memo (S3), DiT 1.12×
* **Circular KV cache (WS4).** The original attention cloned the entire KV cache, about 2.5 GiB,
  on every forward so that training could defer cache writes. Inference now writes denoise tokens
  speculatively into a mirrored ring and advances the pointer only on the commit forward. The
  mirror keeps the wrapped window one contiguous slice for SDPA.
* **PRoPE memo (WS12).** The PRoPE projection matrices depend only on the chunk's cameras, yet
  `_prepare_apply_fns_all_dim` rebuilt them twice per layer, 30 layers deep. They are now memoized
  once per forward, keyed on the window bounds.
* `0352a3c` only enables the memo for circular caches, so S3 measures both together.
* Both are bit-exact.

### 5.3 VAE decode: bf16, channels-last, Triton fusion (S4–S6)
* **bf16 elementwise (S4, WS14), VAE 1.19×.** The decoder ran convolutions under bf16 autocast,
  but RMSNorm, the `_scale` reciprocal and nearest-exact upsampling round-tripped through fp32,
  and each tile was returned as `.float()`. Now:
  * `RMS_norm` narrows `F.normalize` back to bf16.
  * The `_scale` reciprocal is narrowed so autocast cannot reintroduce an fp32 stream.
  * Upsampling uses XPU-native bf16 `nearest-exact` instead of an fp32 round trip.
  * fp16 was rejected because decoder activations overflow it.
  * S4 is measured at `4d0c2f9` with all later toggles off, so it also includes code changes
    between `0352a3c` and `4d0c2f9` that have no toggle. DiT time is unchanged, so the gain is in
    the VAE.
* **Channels-last (S5, WS14), VAE 1.06×.** `channels_last_3d` for decoder `Conv3d` and
  `channels_last` for the resampling `Conv2d` remove oneDNN layout reorders.
* **Triton RMSNorm+SiLU (S6, WS20), VAE 1.27×.** `triton_rms_norm_silu` and
  `triton_add_rms_norm_silu` fuse normalization, the optional residual add, scale/bias and SiLU
  into one pass.

### 5.4 RoPEd KV ring (S7, WS13), DiT 1.25×
Within a chunk, the window bounds are identical across all 5 forwards: only the 1,215 new tokens
change, and the 6,075 history tokens do not.

* The RoPEd ring stores EchoRoPE/PRoPE-transformed K and V in a persistent buffer.
* History is transformed once per chunk; each forward transforms only the new tokens and attends
  over a contiguous slice.
* Dropping the raw ring's mirror pays for the RoPEd buffer, so device memory is unchanged.
* The change is bit-exact. (Config key: `runtime.stage2_encoded_kv_ring`.)

### 5.5 Fused attention (S8) and the raw-window copy (S11)
* **S8, DiT 1.13×:** a Triton kernel fuses EchoRoPE, the PRoPE transform and vendor SDPA
  (`fused_rope_prope_sdpa`).
* **S11, DiT 1.10×:** at `4d0c2f9` the fused path read the whole raw K/V window from the
  unmirrored ring on every forward (`_read_ring_slice`, a `torch.cat` whenever the window wraps),
  even though attention reads the RoPEd ring.
  * A one-chunk XPU profile at S10's config shows 93.6 ms in `aten::cat` (300 calls, 2 per layer
    per forward).
  * `32b1dd9`, which rewrote the attention path with one kernel-selection seam
    (`model.attention_backend`), does not read it. Its profile has no `aten::cat`, and DiT time
    drops by 86 ms.

### 5.6 DiT operator fusions (S9, WS22), DiT 1.18×
| Fusion | Isolated eager ms | Isolated fused ms |
| --- | ---: | ---: |
| AdaLN-modulated LayerNorm (Triton) | 0.347 | 0.109 |
| Gated residual (`addcmul_`) | 0.035 | 0.016 |
| Q/K WanRMSNorm (Triton) | 0.061 | 0.027 |
| FFN Linear + GELU (oneDNN pointwise post-op) | 0.984 | 0.868 |
| Whole attention block forward | 5.292 | 5.052 |

These isolated numbers come from `scripts/debug/benchmark_dit_fusions.py` at HEAD. The block
saving (0.24 ms × 30 layers × 5 forwards ≈ 36 ms) under-predicts the 183 ms chunk-level gain at
`4d0c2f9`; the difference was not isolated.

### 5.7 VAE epilogue fusion (S12, WS23), VAE 1.11×
The residual add after each `ResidualBlock`'s second convolution is done by oneDNN
`post_ops::append_sum` into the shortcut buffer, and QuaRot prologue buffers are reused.
`b66f630` also merges other branch changes, which account for the 1.03× DiT change in this step.

## 6. Quantization

Three methods, applied to different parts of the model:

| Method | Applied to | Weights | Activations | Calibration |
| --- | --- | --- | --- | --- |
| SVDQuant W4A4 | DiT linears: Q/K/V, FFN up, FFN down | INT4 + BF16 rank-32 low-rank branch | INT4, per group of 64 | Offline: SVDQuant + timestep-aware GPTQ |
| QuaRot dynamic W8A8 | VAE decoder 3×3×3 convolutions | INT8, per output channel | INT8, dynamic per-tensor scale at run time | None (data-free) |
| SageAttention v1 / v2 | DiT self-attention Q·Kᵀ | — | Q and K in INT8 (v1) / INT4 (v2) | None |

### 6.1 How quantization is measured
* **Speed, chunk-level:** DiT or VAE time from the chain.
* **Speed, isolated:** only the quantized block, timed with the production call sequence at HEAD
  (`scripts/report/bench_isolated_blocks.py`).
* **Speed, quantization table:** one 13-chunk (153-frame) rollout of the benchmark workload per
  configuration at HEAD, with the GPU locked at 2,000 MHz (`scripts/report/quality.py`,
  `outputs/report_chain/quality_153f/metrics.json`). Steady-state median of chunks 5–12, one timed
  VAE pass. The chain timings elsewhere in this report use the default clock, so these absolute
  times are higher.
* **Quality:** DOVER++ [15], a no-reference video quality model. A reference comparison such as
  PSNR does not measure quality here: each chunk is generated from the previous ones, so a small
  difference becomes a different but still plausible continuation, which scores as low as a broken
  one.
  * **Samples:** the clean-153f test split
    (`data/releases-v1/recipes/clean-153f/.../test-index.jsonl.gz`) restricted to the shards
    present locally: 134 samples from 8 sources. 17 of them (11 omniworld, 3 spatialvid,
    2 sekai_game, 1 realcam_vid) are rejected by the model's camera guard (relative translation
    too large) in every configuration, leaving **117**: abot 50, realcam_vid 16, miradata 12,
    sekai_walking 11, dl3dv 10, spatialvid 8, sekai_game 6, omniworld 4.
  * **Rollout:** each sample's first latent, camera and prompt; 153 frames (13 chunks) at
    480×864, seed 1234, the same for all configurations. Ground truth is the sample's own latents
    decoded by the BF16 VAE.
  * **Clips:** first chunk (frames 0–8), one chunk's error; last chunk (frames 141–152, chunk
    13), which includes autoregressive drift. Encoded lossless (libx264 `-qp 0`, yuv444p) at
    16 FPS.
  * **Scores:** DOVER++ (`DOVER_plus_plus.pth`) aesthetic and technical head scores and the
    official fused overall score, each rescaled to 0–1 as in DOVER's `evaluate_a_set_of_videos.py`;
    higher is better. DOVER++'s heads sit near the ends of that rescaling (aesthetic ≈ 0.95,
    technical ≈ 0.09, ground truth included), so compare differences, not absolute values. DOVER samples 32-frame clips, so the 9- and 12-frame
    clips are looped by its frame sampler; the same seed per sample and chunk is used for every
    configuration.
  * **vs BF16:** the score tables give the change of each mean relative to BF16 in parentheses.
    The difference table gives the per-sample difference from BF16 on the same sample and chunk,
    mean with a 95 % bootstrap interval over the 117 samples.
* **SVDQuant weights:** k1, k2 and k4 are 1, 2 and 4 timestep-bin weight/clip sets. The weights are
  calibrated per attention backend: SDPA loads `sdpa_k<n>` and Sage v1 loads `sageattnv1_k4`
  (calibrated with Sage v1 attention). There are no weights calibrated with Sage v2, so Sage v2 is
  measured only with BF16 linears.
* QuaRot changes only the VAE, so its DiT latents are identical and any change is from the decode.

| Configuration | DiT ms | VAE ms | Chunk ms | Chunk × | Overall first | Overall last | Aesthetic first | Aesthetic last | Technical first | Technical last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 reference | 968.9 | 1,138.2 | 2,107.2 | 1.00× | 0.421 | 0.466 | 0.947 | 0.956 | 0.087 | 0.098 |
| Ground truth (dataset latents) | — | — | — | — | 0.422 (+0.3 %) | 0.455 (−2.3 %) | 0.947 (+0.0 %) | 0.951 (−0.6 %) | 0.087 (+1.0 %) | 0.103 (+4.9 %) |
| SageAttention v1 | 896.4 | 1,136.8 | 2,033.2 | 1.04× | 0.421 (+0.1 %) | 0.463 (−0.5 %) | 0.947 (+0.0 %) | 0.954 (−0.3 %) | 0.087 (+0.2 %) | 0.099 (+0.5 %) |
| SageAttention v2 | 888.6 | 1,137.2 | 2,025.8 | 1.04× | 0.414 (−1.6 %) | **0.299 (−35.8 %)** | 0.947 (+0.0 %) | **0.911 (−4.7 %)** | 0.083 (−4.0 %) | **0.051 (−48.7 %)** |
| SVDQuant W4A4 k1 | 916.9 | 1,136.5 | 2,053.4 | 1.03× | 0.415 (−1.3 %) | 0.445 (−4.5 %) | 0.947 (+0.0 %) | 0.952 (−0.5 %) | 0.084 (−3.3 %) | 0.092 (−6.3 %) |
| SVDQuant W4A4 k2 | 916.1 | 1,136.3 | 2,052.4 | 1.03× | 0.417 (−1.0 %) | 0.437 (−6.1 %) | 0.947 (+0.0 %) | 0.951 (−0.6 %) | 0.085 (−2.4 %) | 0.091 (−7.7 %) |
| SVDQuant W4A4 k4 | 915.6 | 1,136.9 | 2,052.5 | 1.03× | 0.416 (−1.1 %) | 0.438 (−5.9 %) | 0.947 (+0.0 %) | 0.951 (−0.6 %) | 0.084 (−3.0 %) | 0.091 (−7.2 %) |
| QuaRot dynamic W8A8 VAE | 957.0 | 946.1 | 1,903.0 | 1.11× | 0.420 (−0.1 %) | 0.466 (−0.1 %) | 0.947 (−0.0 %) | 0.956 (+0.0 %) | 0.086 (−0.1 %) | 0.098 (−0.1 %) |
| SVDQuant + W8A8 | 916.7 | 937.3 | 1,854.0 | 1.14× | 0.416 (−1.2 %) | 0.438 (−5.9 %) | 0.947 (−0.0 %) | 0.951 (−0.6 %) | 0.084 (−3.1 %) | 0.091 (−7.2 %) |
| Sage v1 + SVDQuant | 841.4 | 1,137.0 | 1,978.4 | 1.07× | 0.417 (−1.0 %) | 0.433 (−6.9 %) | 0.947 (+0.0 %) | 0.948 (−0.8 %) | 0.084 (−2.5 %) | 0.089 (−9.4 %) |
| Sage v1 + SVDQuant + W8A8 | 851.3 | 946.0 | 1,797.3 | 1.17× | 0.416 (−1.1 %) | 0.433 (−7.1 %) | 0.947 (−0.0 %) | 0.948 (−0.9 %) | 0.084 (−2.6 %) | 0.089 (−9.5 %) |

Paired difference vs BF16 on the same sample and chunk, mean [95 % interval]:

| Configuration | Overall first | Overall last | Aesthetic first | Aesthetic last | Technical first | Technical last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ground truth (dataset latents) | +0.001 [−0.002, +0.004] | −0.011 [−0.022, +0.002] | +0.000 [+0.000, +0.000] | −0.005 [−0.010, +0.001] | +0.001 [−0.001, +0.002] | +0.005 [+0.001, +0.009] |
| SageAttention v1 | +0.000 [−0.000, +0.001] | −0.002 [−0.010, +0.005] | +0.000 [+0.000, +0.000] | −0.003 [−0.007, +0.002] | +0.000 [−0.000, +0.000] | +0.001 [−0.002, +0.003] |
| SageAttention v2 | −0.007 [−0.008, −0.006] | **−0.167 [−0.187, −0.146]** | +0.000 [+0.000, +0.000] | **−0.045 [−0.056, −0.034]** | −0.003 [−0.004, −0.003] | **−0.048 [−0.054, −0.042]** |
| SVDQuant W4A4 k1 | −0.006 [−0.007, −0.004] | −0.021 [−0.030, −0.012] | +0.000 [+0.000, +0.000] | −0.004 [−0.008, −0.001] | −0.003 [−0.004, −0.002] | −0.006 [−0.009, −0.003] |
| SVDQuant W4A4 k2 | −0.004 [−0.005, −0.003] | −0.028 [−0.037, −0.020] | +0.000 [+0.000, +0.000] | −0.005 [−0.010, −0.000] | −0.002 [−0.003, −0.001] | −0.008 [−0.011, −0.005] |
| SVDQuant W4A4 k4 | −0.005 [−0.006, −0.004] | −0.027 [−0.035, −0.019] | +0.000 [+0.000, +0.000] | −0.006 [−0.009, −0.002] | −0.003 [−0.003, −0.002] | −0.007 [−0.010, −0.004] |
| QuaRot dynamic W8A8 VAE | −0.000 [−0.001, −0.000] | −0.000 [−0.001, +0.000] | −0.000 [−0.000, −0.000] | +0.000 [−0.000, +0.000] | −0.000 [−0.000, −0.000] | −0.000 [−0.000, −0.000] |
| SVDQuant + W8A8 | −0.005 [−0.006, −0.004] | −0.028 [−0.035, −0.020] | −0.000 [−0.000, −0.000] | −0.006 [−0.009, −0.002] | −0.003 [−0.003, −0.002] | −0.007 [−0.010, −0.005] |
| Sage v1 + SVDQuant | −0.004 [−0.005, −0.003] | −0.032 [−0.041, −0.024] | +0.000 [+0.000, +0.000] | −0.008 [−0.014, −0.003] | −0.002 [−0.003, −0.002] | −0.009 [−0.013, −0.006] |
| Sage v1 + SVDQuant + W8A8 | −0.005 [−0.006, −0.004] | −0.033 [−0.042, −0.024] | −0.000 [−0.000, −0.000] | −0.008 [−0.015, −0.003] | −0.002 [−0.003, −0.002] | −0.009 [−0.013, −0.006] |

* **Sage v2 breaks long rollouts.** Its first chunk is close to BF16, but by chunk 13 it is lower on
  110 of 117 samples, and more than 0.1 lower on 79; the frames degrade into texture. Both heads
  drop: aesthetic 0.956 → 0.911 (lower on 99 samples), technical 0.098 → 0.051 (lower on 111).
* **Sage v1 and QuaRot W8A8 are not distinguishable from BF16** at either chunk, in all three
  scores.
* **SVDQuant costs a small, consistent amount** that grows with the rollout: −0.005 at the first
  chunk, −0.027 at the last (lower on 94 of 117 samples, more than 0.1 lower on 6), with aesthetic
  −0.006 and technical −0.007 at the last chunk. k1, k2 and k4 are within each other's intervals. Adding Sage v1 or W8A8 to it changes little.
* BF16 and ground truth score the same at the first chunk. At the last chunk the generated video
  scores slightly above the real data overall and in aesthetic, and slightly below in technical
  (−0.005): DOVER++ rates perceptual quality, not fidelity to the
  scene.
* Plain DOVER gives the same ordering. Last chunk vs BF16, overall / aesthetic / technical:
  Sage v2 −0.237 / −0.274 / −0.036, SVDQuant k4 −0.042 / −0.012 / −0.004, Sage v1
  −0.001 / +0.000 / −0.000, W8A8 −0.000 / +0.001 / −0.000. Per-clip scores, sub-scores and per-source means are in
  `outputs/report_chain/vqa_153f/scores.json`.
* **Examples:** 4 random samples (seed 20261024; abot, miradata, sekai_game, spatialvid), full
  153-frame videos of ground truth and all 12 configurations, tiled per sample in
  `outputs/report_chain/vqa_153f/examples/grid_<source>.mp4`.

### 6.2 SVDQuant W4A4 DiT linears (S14, WS25), DiT 1.05×
* **Method:** SVDQuant [6] splits each linear into a BF16 low-rank branch (rank 32), which
  absorbs activation and weight outliers, and an INT4 residual. Calibration follows the SVDQuant
  recipe of DeepCompressor [7] with the timestep-aware SVDQuant-GPTQ extension [8]:
  * per-channel smoothing between activations and weights;
  * a rank-32 SVD of the smoothed weight as the low-rank branch;
  * GPTQ [9] for the INT4 group-64 residual instead of round-to-nearest;
  * a per-layer, per-timestep activation clipping ratio (from 0.6–1.0), searched to minimize the
    layer's output error. The production weights (**k4**) have 4 weight/clip sets, one per denoise
    step; the commit forward uses the last. k1 and k2 (1 and 2 sets) are compared in §6.1.
  * Calibrated on CUDA with `scripts/quantization/calibrate_svdquant.py`; weights in [5]
    (`sdpa_k4`; `sageattnv1_k4` when combined with Sage v1).
* **Kernel:** a SYCL dense-block kernel tuned with KernelFoundry: INT4 × INT4 DPAS with prepacked
  DPAS-B weights, `grf128`, N-tile 64.
  * Activation quantization and LoRA-down run in the prologue; LoRA-up, bias and the Q/K RMSNorm
    run in the epilogue.
* **Production wiring** (`load_svdquant_checkpoint(fused=True)`):
  * Q/K/V share one activation-quantization + LoRA-down pass, and the Q/K RMSNorms run in the GEMM
    epilogue.
  * FFN up fuses GELU and emits the INT4 activations that FFN down consumes.
* **Isolated, at HEAD, GPU locked at 2,000 MHz:** one block's quantized linears with real
  `sdpa_k4` weights (bin 0), vs the BF16 path the block runs at HEAD (oneDNN linears + Triton QK
  RMSNorm; oneDNN Linear+GELU for FFN up):
  * Q/K/V: 0.771 → 0.658 ms (**1.17×**). With fused QKV kernel dispatch, panel swizzling, and parallel dual RMSNorm, SVDQuant now beats oneDNN's BF16 GEMMs.
  * FFN: 2.084 → 1.800 ms (**1.16×**).
  * Together: 2.855 → 2.458 ms (**1.16×**), 60 ms per chunk, translating to a 53.3 ms DiT saving of the
    whole rollout at the same clock (§6.1). The DiT gain is smaller than 1.16× because attention,
    cross-attention, the output projection and elementwise work are not quantized.
* **Kernel error:** relative RMS vs the BF16 linears, on the real inputs of layers 0 / 15 / 29 in
  the first denoise forward of a steady-state chunk (bin 0), and on random Gaussian activations
  with the same real weights (Part III §3):

  | Linear | Real, layer 0 | Real, layer 15 | Real, layer 29 | Random, layers 0 / 15 / 29 |
  | --- | ---: | ---: | ---: | ---: |
  | Q | 6.4 % | 8.4 % | 8.3 % | 20.0 / 16.5 / 17.4 % |
  | K | 3.9 % | 7.0 % | 3.5 % | 19.0 / 15.9 / 15.9 % |
  | V | 7.5 % | 11.3 % | 11.2 % | 22.2 / 17.1 / 18.3 % |
  | FFN (up + GELU + down) | 11.0 % | 21.6 % | 11.6 % | 15.3 / 21.6 / 13.9 % |

  Random activations have no channel structure for the smoothing and clipping calibrated on
  real activations to exploit, so they are 2–5× worse on Q/K/V than real inputs.

* **vs llm-scaler's SVDQuant kernel.** llm-scaler [10] runs SVDQuant in ComfyUI through
  comfy-kitchen-xpu [11], whose XPU backend calls llm-scaler's `omni_xpu_kernel`:
  * activations are quantized to INT4 by an ESIMD kernel, then dequantized back to BF16;
  * a oneDNN INT4-weight × BF16-activation GEMM follows, with M padded to a multiple of 256;
  * the LoRA branch runs as separate fp32 matmuls.

  Both kernels were timed on the same real block-0 weights and input, one complete SVDQuant
  linear at a time (activation quantization + LoRA + INT4 GEMM), with the GPU clock locked at
  2,000 MHz:

  | Linear (M × K × N) | llm-scaler ms | Ours ms | Speedup |
  | --- | ---: | ---: | ---: |
  | Q, K, V (1215 × 3072 × 3072), each | 0.656–0.662 | 0.283–0.297 | **2.21–2.34×** |
  | FFN up (1215 × 3072 × 14336) | 2.043 | 0.873 | **2.34×** |
  | FFN down (1215 × 14336 × 3072) | 2.268 | 1.185 | **1.91×** |
  | **All five linears of a block** | **6.286** | **2.922** | **2.15×** |

  * Our kernel's own Q/K/V sharing and FFN fusion are not used here; per linear, they would add
    to the gap.
  * The outputs agree to 2.3–2.7 % relative RMS when our activation clipping is set to 1.0;
    llm-scaler's quantizer has no clipping.
* **Quality vs BF16 (k4, §6.1):** DOVER++ first / last chunk: overall −0.005 / −0.027, aesthetic
  +0.000 / −0.006, technical −0.003 / −0.007.

### 6.3 QuaRot dynamic W8A8 VAE (S10, WS21), VAE 1.18×
* **Method:** QuaRot [12]-style rotation with dynamic INT8 quantization of the 28 primary 3×3×3
  convolutions in the decoder's residual blocks (`residual.2` / `residual.6`).
  * Input channels are rotated by a block-256 Sylvester-Hadamard matrix H. Hᵀ is folded into the
    weights offline, so X·W = (X·H)·(Hᵀ·W). The rotation spreads channel outliers, which makes a
    single activation scale usable.
  * Weights: symmetric INT8 per output channel, quantized once at load.
  * Activations: **dynamically quantized** at run time, with one absmax scale per tensor per call.
    No calibration data.
  * The convolution runs in oneDNN INT8 with fp32 accumulation and BF16 output, through a small
    SYCL extension (`qconv_onednn`). The extension avoids a host sync per call and fuses the
    residual add (`append_sum`, §5.7).
* **Isolated, at HEAD, GPU locked at 2,000 MHz:** each quantized convolution (84 calls per
  3-latent chunk) is timed on its input from a steady-state decode, BF16 vs QuaRot:

  | Decoder stage (channels, resolution) | Calls | BF16 ms | QuaRot ms | Speedup |
  | --- | ---: | ---: | ---: | ---: |
  | 1024 ch, 30×54 | 30 | 70.8 | 39.6 | 1.79× |
  | 1024 ch, 60×108 | 18 | 159.6 | 116.0 | 1.38× |
  | 1024 / 512 ch, 120×216 | 18 | 308.1 | 220.8 | 1.40× |
  | 512 / 256 ch, 240×432 | 18 | 337.7 | 302.2 | 1.12× |
  | **All** | **84** | **876.2** | **678.5** | **1.29×** |

  At full output resolution, QuaRot gives little gain: the rotation and activation quantization
  passes cost almost as much as the INT8 convolution saves.
* **Quality vs BF16 (§6.1):** DOVER++ overall, aesthetic and technical unchanged at both chunks
  (−0.000). The DiT latents are
  identical, so any change is from the decode.

### 6.4 SageAttention v1 / v2 (S8a, S14a)
* **Method:** SageAttention [13, 14] quantizes Q and K for Q·Kᵀ; the softmax and P·V stay BF16.
  * **v1:** INT8 Q and K, per-block scales (32 tokens for K). K is smoothed by subtracting its mean
    over the window, which does not change the softmax. Q and K are rotated by a 4×4 Hadamard
    matrix folded into the PRoPE camera matrices (no extra pass; Q·Kᵀ unchanged). Triton kernel
    with INT8 DPAS.
  * **v2:** INT4 Q and K. Q is centered by its mean, and the mean's contribution is added back
    exactly (`compute_delta_s`). SYCL kernel with INT4 DPAS.

All timings with the GPU locked at 2,000 MHz (min = max frequency; it held under load). DiT and
chunk times for v1 are with SVDQuant (chain steps S14 = SDPA, S14a = Sage v1). Sage v2 has no
SVDQuant weights calibrated for it, so its DiT and chunk times are with BF16 linears, vs the BF16
reference (§6.1).

| Measurement | SDPA ms | Sage v1 ms | v1 × | Sage v2 ms | v2 × |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention kernel alone (one call) | 1.760 | 1.247 | **1.41×** | 0.964 | **1.83×** |
| DiT time per chunk, v1 (with SVDQuant) | 916.7 | 841.4 | **1.09×** | — | — |
| DiT time per chunk, v2 (BF16 linears) | 968.9 | — | — | 888.6 | **1.09×** |
| End-to-end chunk, v1 (with SVDQuant) | 1,854.0 | 1,778.7 | **1.04×** | — | — |
| End-to-end chunk, v2 (BF16 linears) | 2,107.2 | — | — | 2,025.8 | **1.04×** |

What each timing covers:

* **Attention kernel alone:** one call of the attention kernel for one layer: 1,215 queries over
  the 7,290-token KV window (24 heads × 128).
  * Inputs are already prepared: RoPEd Q, and for Sage, K already quantized (INT8 for v1, INT4
    DPAS images for v2).
* **DiT time per chunk:** the whole DiT for one chunk: 5 forwards (4 denoise + 1 KV-cache commit)
  × 30 layers.
  * It includes everything else in the DiT: linears, norms, cross-attention, RoPE/PRoPE of new
    tokens, ring writes, and new-tail K quantization.
  * Once per chunk, it also RoPEs and (for Sage) quantizes the 6,075 history tokens.
* **End-to-end chunk:** DiT time plus the VAE decode of the chunk's 12 frames. The VAE is the same
  in all runs.
* **Actual speedups may be lower** at the default clock because of GPU throttling: under
  sustained load the clock drops to about 2,090 MHz during the Sage v1 kernel vs about 2,470 MHz
  during SDPA, so Sage v1's kernel gain shrinks (at the default clock the S14a chain step is DiT
  1.01×).
* **Kernel error:** relative RMS of the attention output vs BF16 SDPA (Part III §3):
  * on the real Q/K/V of layers 0 / 15 / 29 in the first denoise forward of a steady-state chunk;
  * on random Gaussian Q/K/V of the same shapes (all three layers agree to 0.01 %).

  Sage v1 uses the Hadamard-4 rotation and both use the history K mean, as in production.

  | | Real, layer 0 | Real, layer 15 | Real, layer 29 | Random |
  | --- | ---: | ---: | ---: | ---: |
  | BF16 SDPA vs FP32 SDPA | 0.18 % | 0.17 % | 0.12 % | 0.23 % |
  | Sage v1 (INT8 Q/K) | 3.6 % | 2.7 % | 2.2 % | 1.35 % |
  | Sage v2 (INT4 Q/K) | 39.9 % | 30.7 % | 54.6 % | 17.1 % |

  Real Q/K have outlier channels and peaked softmax rows, so both kernels are 1.6–3.2× worse on
  real data than on random data. Sage v2's per-call error is 11–25× Sage v1's and accumulates over
  the rollout (quality below).

Implementation:

* **Quantized K history cache:** when history is RoPEd once per chunk, its K is also quantized
  once:
  * v1: `quantise_k(..., out=...)` into the persistent window buffers `_sage_k_int8` /
    `_sage_k_scale`.
  * v2: `pack_k_into` into the persistent window buffers `_sagev2_k4` / `_sagev2_k_scale`.

  Each forward quantizes only the 1,215 new tokens and writes them into the same buffers after
  the history, so the kernel reads one contiguous window.
* **Stored layouts match what the kernels read:**
  * v1 keeps INT8 K as row-major `[B, H, L, D]`. The Triton kernel loads `[32, 128]` tiles with a
    transposed 2D block load straight into INT8 DPAS.
  * v2 stores K as DPAS-B register images, `K4 [B, H, L/16, D/64, 8 dwords, 16 lanes]`, which the
    SYCL kernel fills with one sub-group block read per operand. Q and the output stay in
    `[B, L, H, D]`.
* **Per-forward work around the kernel:** new-token RoPE/PRoPE, new-tail K quantization, and for
  v2, `compute_delta_s` over the full BF16 K window (0.10 ms), which depends on the current
  `q_mean`. No Triton kernel compiles in the steady-state chunks.
* S8a (Sage v1 at `4d0c2f9`, DiT 1.11× over S7 vs 1.13× for fused SDPA) has no K history cache:
  it RoPEs and quantizes the whole 7,290-token K window on every forward.

**Quality vs BF16 (§6.1), DOVER++ over 117 test samples:**

* v1 matches SDPA. First / last chunk: overall +0.000 / −0.002, aesthetic +0.000 / −0.003,
  technical +0.000 / +0.001. With SVDQuant k4 the last chunk is overall −0.032, aesthetic −0.008,
  technical −0.009, close to SVDQuant alone (−0.027 / −0.006 / −0.007).
* v2 degrades over the rollout. First / last chunk: overall −0.007 / −0.167 (79 of 117 samples
  more than 0.1 lower), aesthetic +0.000 / −0.045, technical −0.003 / −0.048. SDPA stays the default, and v2 is not usable for long rollouts.

## 7. Alternatives and What Did Not Work

| Step | Alternative | vs | DiT ms | DiT × | VAE ms | VAE × | Chunk ms | Chunk × |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S3a | `torch.compile` DiT blocks (WS7) | S3 | 7821.4 | 0.21× | 1719.5 | 1.00× | 9540.9 | 0.35× |
| S8a | Fused RoPE+PRoPE+SageAttention v1 | S7 | 1196.9 | 1.11× | 1078.7 | 1.00× | 2275.6 | 1.06× |
| S9a | `torch.compile` VAE decoder (WS24) | S9 | 999.9 | 1.00× | 1031.5 | 1.05× | 2031.4 | 1.02× |
| S14a | SVDQuant + SageAttention v1 | S14 | 814.5 | 1.01× | 817.0 | 1.00× | 1631.5 | 1.01× |
| S14c | SVDQuant + radial sparse attention | S14 | 818.9 | 1.01× | 816.5 | 1.00× | 1635.4 | 1.00× |

* **`torch.compile` DiT (S3a): DiT time 1,660.5 → 7,821.4 ms (0.21×).** Dynamo guards on cache
  indices and the per-forward PRoPE memo, so it recompiles every chunk until it hits the recompile
  limit. Per-chunk DiT time falls from 24.5 s (chunk 0) to 6.6 s (chunk 7), and the measured
  chunks 5–7 still include recompilation.
* **`torch.compile` VAE (S9a): VAE time 1,078.8 → 1,031.5 ms (1.05×)**, with a long JIT warmup.
  Not worth a compile step in an interactive application. `torch-xpu` uses oneDNN kernels that
  TorchInductor cannot fuse with other operations.
* **SageAttention v1/v2 (S8a, S14a).** See §6.4.
* **Radial sparse attention (S14c).** Within noise of dense attention at these sequence lengths.
* **OpenVINO export of the VAE (WS24).** These numbers come from the plan
  (`wan22-xpu-inference-optimization-plan.md`, Workstream 24), measured on the B70 with that
  section's own harness: BF16 + Triton norm, no QuaRot, 480×864, 3 latent frames. They are not on
  the chain, and the export script is not in the repository.

  | Engine | Per chunk | VAE × vs eager |
  | --- | ---: | ---: |
  | PyTorch eager (BF16 + Triton norm/SiLU) | 736.6 ms | 1.00× |
  | `torch.compile` (Inductor XPU) | 720.7 ms | 1.02× |
  | OpenVINO GPU, FP16 IR | 1,037.6 ms | 0.71× |
  | OpenVINO CPU (Core Ultra 5 245K), FP32 IR | ~87.4 s | 0.008× |

  The plan attributes the GPU regression to OpenVINO's 3D convolution kernels, which lack the
  oneDNN DPAS `channels_last_3d` path that PyTorch uses.
* **Overlapping DiT and VAE on two streams (WS6).** No throughput gain, because the device
  serializes the two compute streams. The pipeline config key no longer exists, so this harness
  cannot time it.
* **Folding EchoRoPE into PRoPE (WS18).** Failed the video quality gate.
* **Plain W8A8 VAE without rotation (WS19).** The gain was too small for its complexity; QuaRot
  (S10) is used instead.

## 8. Lower resolution (R320)
* With the S14 stack at 320×576 (180 tokens per frame), DiT takes 374.7 ms and VAE 404.6 ms: a
  779.2 ms chunk, **15.40 FPS**, 1.04× over the 750 ms target.
* This is a different computation, so it is reported separately rather than as a speedup.
* The ComfyUI viewer adds GPU-side uint8 conversion and a background encode thread so host
  post-processing does not add to the chunk time; that path is not timed by this harness.

## 9. Takeaways

1. **Measure where GPU time goes first.** The GPU was saturated, so scheduling fixes gave 1.00×.
2. **Delete redundant work.** Caching and reusing RoPE/PRoPE work (S3, S7, S11) gave DiT 1.12×,
   1.25× and 1.10×.
3. **Fuse bandwidth-bound work.** bf16, channels-last and Triton fusion (S4–S6) took the VAE from
   1,722 to 1,078 ms without quantization.
4. **Low-precision speedups shrink in real use.** At 2,000 MHz SVDQuant is 1.16× on the quantized
   linears and 1.06× on DiT time. Sage v2's INT4 kernel is 1.83× faster than SDPA, but the chunk
   only 1.04×.
5. **The kernel implementation matters as much as the format.** With the same SVDQuant W4A4
   weights, the INT4 × INT4 DPAS kernel is 2.15× faster than an INT4-weight × BF16 oneDNN path.
6. **Compilers did not help on this stack:** `torch.compile` gave 0.21× on DiT time and 1.05× on
   VAE time; `torch-xpu` uses oneDNN kernels that TorchInductor cannot fuse with other
   operations. The OpenVINO runtime is slower than PyTorch eager for the VAE (0.71×).
7. **480p real time needs another 2.19×.** At 320p the same stack is within 4 % of the 750 ms
   target.

# Part III: Reproduction

All runs: Intel Arc Pro B70, Linux 6.18, `torch 2.12.1+xpu`, `.venv-wan22-xpu`, nothing else on
the GPU.

QuaRot's `qconv_onednn` extension is compiled on first use and needs `icpx` on `PATH`
(`/opt/intel/oneapi/compiler/latest/bin`). The `b66f630` worktree (S12) loads it as
`qconv_onednn_ops.so`.

## 1. Chain benchmark

* `scripts/report/bench_chunk.py` measures one configuration of one commit.
  * It uses only APIs that are unchanged from `001e586` to HEAD
    (`build_stage2_generation_provider`, `_load_role`, `_stage2_self_forcing_latents`,
    `decode_streaming_chunks`).
  * It stubs the text encoder and feeds one preencoded sample from
    `latent-wds/wan22-ti2v5b-153f-480p-v1/abot/shards/kept-high-000008.tar`.
  * **DiT time:** each diffusion forward is wrapped with device syncs; a chunk is the sum of its
    5 forwards.
  * **VAE time:** all chunks are streamed through one continuous causal cache, as in production.
    Each tile is timed including the copy of its pixels to the host. The first full pass is
    warmup; the next 2 passes are timed.
  * **Steady state:** median over chunks 5–7.
  * `--profile-chunk N` stores a DiT XPU kernel table for chunk N (used for S11).
  * `--save-frames PATH` saves the decoded video of the last pass (used for the error metrics).
* `scripts/report/run_chain.py` defines every step (commit + config overrides + reference step).
  It creates detached worktrees under `outputs/report_worktrees/<commit>`, runs `bench_chunk.py`
  in each, and prints the table with DiT, VAE and chunk speedups.

```bash
source .venv-wan22-xpu/bin/activate
python scripts/report/run_chain.py               # all steps, 30-40 min; reuses saved results
python scripts/report/run_chain.py --steps S7 S8 --rerun
python scripts/report/run_chain.py --table       # print outputs/report_chain/*.json
```

## 2. Commits used

| Commit | Steps | Why this commit |
| --- | --- | --- |
| `001e586` | S0 | First XPU port |
| `b0491d0` | S1 | Sync removal and streaming decode only |
| `0352a3c` | S2, S3, S3a | Circular cache and PRoPE memo; `kv_cache_mode=clone` and DiT compile still toggleable |
| `4d0c2f9` | S4–S10, S8a, S9a | Last commit where unfused attention, the RoPEd ring, DiT fusions, VAE channels-last, VAE Triton fusions and QuaRot are all config toggles. `32b1dd9` retired `stage2_fused_kernel`, `stage2_encoded_kv_ring`, `stage2_dit_fused_ops` and `stage2_vae_fused_kernels`. |
| `32b1dd9` | S11 | Attention-path rewrite; first commit with `model.attention_backend` / `model.linear_quant` |
| `b66f630` | S12 | WS23 VAE epilogue fusion |
| HEAD of `xpu` | S13, S14, S14a–c, R320 | Current code |

Toggles at `4d0c2f9`, all off for S4: `runtime.stage2_fused_kernel=none`,
`runtime.stage2_encoded_kv_ring=false`, `runtime.stage2_dit_fused_ops=false`,
`runtime.stage2_vae_channels_last=false`, `runtime.stage2_vae_fused_kernels=false`,
`runtime.stage2_vae_int8_quarot=false`. Each later step turns one on (see `STEPS` in
`run_chain.py`).

## 3. Isolated block measurements (not the chunk)

`scripts/report/bench_isolated_blocks.py` times one quantized block at HEAD with the production
call sequence, inputs and layouts. The quantization timings in this report were run with the GPU
min and max frequency both set to 2,000 MHz (check
`/sys/class/drm/card0/device/tile0/gt0/freq0/{min,max,act}_freq` under load) and saved in
`outputs/report_chain/freq2000/`.

```bash
python scripts/report/bench_isolated_blocks.py attention --output-json outputs/report_chain/freq2000/attention.json
python scripts/report/bench_isolated_blocks.py svdquant  --output-json outputs/report_chain/freq2000/svdquant.json
python scripts/report/bench_isolated_blocks.py quarot    --output-json outputs/report_chain/freq2000/quarot.json
python scripts/report/kernel_errors.py --output-json outputs/report_chain/isolated/kernel_errors.json
PYTHONPATH=<omni_xpu_kernel install>:<comfy-kitchen-xpu checkout> \
  python scripts/report/bench_isolated_blocks.py svdquant-llm-scaler --output-json outputs/report_chain/isolated/svdquant_llm_scaler.json
```

* **`attention`:** one DiT self-attention layer of a steady-state 480p chunk for SDPA, Sage v1 and
  Sage v2.
  * The RoPEd ring holds 6,075 history tokens; RoPE/PRoPE runs on the 1,215 new tokens only.
  * For Sage, history K comes from the once-per-chunk quantized cache, and the new tail is
    quantized on each forward.
  * The report uses its kernel-alone time: the attention kernel on prepared inputs (RoPEd Q,
    quantized K for Sage). It also prints per-layer forward and history times, which the report
    does not use.
  * `outputs/report_chain/sage_debug/clock_under_load.py` measures the sustained kernel time and
    GPU clock for SDPA and Sage v1.
* **`svdquant`:** one block's Q/K/V and FFN linears with real `sdpa_k4` weights, compared with
  the BF16 path at HEAD.
* **`svdquant-llm-scaler`:** each of block 0's SVDQuant linears, ours vs llm-scaler's, on the
  same weights and input.
  * llm-scaler `51fc819`, `omni/omni_xpu_kernel`, built as a wheel for `torch 2.12.1+xpu` with
    `OMNI_XPU_DEVICE=bmg` against oneAPI 2025.3 oneDNN 3.9.1.
  * Its Linux build pins oneDNN 3.11.2 (oneAPI 2026.0), which needs the SYCL 9 runtime and
    cannot load next to `torch 2.12.1+xpu`. For this build the version check in `setup.py` was
    set to 3.9.1; the `onednn_int4_gemm` source is unchanged since it last validated 3.9.1
    (`ce0ccb9`).
  * comfy-kitchen-xpu `950214a` (`comfy_kitchen/backends/xpu/svdquant.py`).
* **`quarot`:** decodes 3 chunks and, on the third, times every QuaRot convolution call on its
  real input (BF16 vs QuaRot).
* **`kernel_errors.py`:** relative RMS error of the Sage and SVDQuant kernels vs their BF16 path,
  on real and on random data (clock-independent).
  * Runs the BF16 benchmark rollout to chunk 5's first denoise forward (full 7,290-token window).
    It captures the self-attention Q/K/V and the five SVDQuant linear inputs of layers 0, 15
    and 29.
  * On those tensors: Sage v1 / v2 vs SDPA (production quantization calls, Sage v1 with the
    Hadamard-4 rotation), and SVDQuant (`sdpa_k4`, bin 0) vs the BF16 linears.
  * Repeats both on random Gaussian tensors of the same shapes (SVDQuant: random activations,
    real weights).

Two other isolated measurements:

| Operator | Reference | Optimized | Speedup | Command |
| --- | --- | --- | ---: | --- |
| RoPE+PRoPE+SDPA over the full window, Lq 1215 / Lkv 7290 | unfused 4.595 ms | fused 2.003 ms | 2.29× | `pytest src/solarwm/kernels/fused_rope_prope_sage/test_fused_rope_prope_sage.py -k benchmark -s` |
| DiT attention block forward, WS22 fusions | eager 5.292 ms | fused 5.052 ms | 1.05× | `python scripts/debug/benchmark_dit_fusions.py` |

## 4. Quantization speed and quality

```bash
cd scripts/report && python quality.py          # speed of each configuration at HEAD; reuses saved videos
python quality_gifs.py                           # first | last chunk GIFs for the summary deck
```

DOVER++ quality (§6.1), from the repository root; generation uses the SolarWM environment
and scoring the separate DOVER environment in `/home/ssheorey/code/DOVER/.venv`:

```bash
for c in bf16 sage1 sage2 svdq_k1 svdq_k2 svdq w8a8 svdq_w8a8 sage1_svdq sage1_svdq_w8a8; do
  PYTHONPATH=scripts/report python scripts/report/vqa_generate.py --configs $c   # one process per configuration
done
PYTHONPATH=scripts/report python scripts/report/vqa_generate.py --ground-truth
cd scripts/report && PYTHONPATH=/home/ssheorey/code/DOVER /home/ssheorey/code/DOVER/.venv/bin/python vqa_score.py
PYTHONPATH=. python vqa_examples.py              # labelled grids of the 4 example videos
```

`vqa_generate.py` loads the model once per configuration and skips clips that already exist. Each
configuration runs in its own process: device memory is not fully released between
configurations, and the seventh one loaded in a single process runs out of memory.

`quality.py` runs each configuration of §6.1 at HEAD with `bench_chunk.py --save-frames` for
13 chunks (153 frames) with one timed VAE pass (`--chunks 13 --vae-repeats 1`), with the GPU
locked at 2,000 MHz, and writes the timings, videos and `metrics.json` to
`outputs/report_chain/quality_153f/`. Its `metrics.json` also holds reference-based PSNR, which
the report does not use.

## 5. Not reproducible, so no chain numbers are given

* **WS25 kernel tuning in isolation.** At `0825790`, just before the KernelFoundry kernel landed in
  `dd357e3`, the older `svdquant_dense_kernel_grf128_n_tile64` extension is not built, and the
  fallback path crashes the device (`UR_RESULT_ERROR_DEVICE_LOST`). That kernel has to be rebuilt
  at that commit; until then, WS25 is included in S14.
* **OpenVINO VAE (WS24).** The export script is not in the repository; Part II §7 quotes
  the plan's numbers.
* **DiT/VAE stream overlap (WS6).** The pipeline key no longer exists. Timing it needs a
  whole-run harness at `f2c3912` with `inference.stage2_xpu_vae_pipeline`.
* **ComfyUI viewer turnaround.** It includes browser and encode work outside this harness.

## 6. Raw results

* `outputs/report_chain/<step>.json` holds, for each step: the commit, overrides, per-chunk DiT
  times, per-repeat per-tile VAE times, the steady-state medians, and peak device memory.
* `outputs/report_chain/bisect/` holds:
  * the commit bisect between `4d0c2f9` and HEAD that split S11 and S12;
  * the S11 kernel profiles (`prof_4d0c2f9.json`, `prof_32b1dd9.json`).
* `outputs/report_chain/isolated/` holds the isolated block results (`attention.json`,
  `svdquant.json`, `svdquant_llm_scaler.json`, `quarot.json`).
* `outputs/report_chain/quality_153f/` holds the 153-frame decoded videos and `metrics.json`.
* `outputs/report_chain/vqa_153f/` holds the first and last chunk clips per configuration and
  sample, `scores.csv` / `scores.json` (DOVER++ and DOVER), `samples.json`, and in `examples/` the
  full 153-frame videos and grids of 4 random samples (seed 20261024, in `examples.json`).
* `outputs/report_chain/freq2000/` holds the clock-locked reruns: `attention.json`,
  `svdquant.json`, `quarot.json`, and `S14.json` and `S14a.json` with `chain.log`. These were run with the GPU min and max
  frequency both set to 2,000 MHz; `clock_under_load.py` confirms the clock holds.

# References

1. J. Huang et al., "SolarWM: Open Data and Scalable Training for Long-Horizon Video World
   Models," arXiv:2609.02886, 2026. <https://arxiv.org/abs/2609.02886>
2. SolarWM project page: <https://junchao-cs.github.io/SolarWM-Web/>
3. SolarWM code: <https://github.com/isl-org/SolarWM>. The Intel XPU port is the `xpu` branch:
   <https://github.com/isl-org/SolarWM/tree/xpu>
4. SolarWM model weights (Hugging Face): <https://huggingface.co/collections/junchaoh-cs/solarwm>
5. SolarWM Wan 2.2-5B SVDQuant W4A4 weights, release `ptc`:
   <https://github.com/isl-org/SolarWM/releases/tag/ptc>
6. M. Li et al., "SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion
   Models," ICLR 2025, arXiv:2411.05007. <https://arxiv.org/abs/2411.05007>
7. DeepCompressor (MIT HAN Lab), model compression toolbox implementing SVDQuant:
   <https://github.com/mit-han-lab/deepcompressor>
8. Junhao Wu, Dezhong Yao, Hai Jin, "Timestep-Aware SVDQuant-GPTQ for W4A4 Quantization of Wan2.2-I2V,"
   arXiv:2605.27003, 2026. <https://arxiv.org/abs/2605.27003>
9. E. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
   Transformers," ICLR 2023, arXiv:2210.17323. <https://arxiv.org/abs/2210.17323>
10. Intel llm-scaler (Omni, `omni_xpu_kernel`): <https://github.com/intel/llm-scaler>
11. comfy-kitchen-xpu: <https://github.com/xiangyuT/comfy-kitchen-xpu>
12. S. Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs," NeurIPS 2024,
    arXiv:2404.00456. <https://arxiv.org/abs/2404.00456>
13. J. Zhang et al., "SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference
    Acceleration," ICLR 2025, arXiv:2410.02367. <https://arxiv.org/abs/2410.02367>
14. J. Zhang et al., "SageAttention2: Efficient Attention with Thorough Outlier Smoothing and
    Per-thread INT4 Quantization," ICML 2025, arXiv:2411.10958. <https://arxiv.org/abs/2411.10958>
15. H. Wu et al., "Exploring Video Quality Assessment on User Generated Contents from Aesthetic
    and Technical Perspectives," ICCV 2023, arXiv:2211.04894. DOVER and DOVER++:
    <https://github.com/VQAssessment/DOVER>

[^softmax]: **Not verified on production hardware.** On a development setup, `torch.softmax` on
    Linux XPU in `2.13.0+xpu` and `2.14.0+xpu` returned wrong values (fp32) or NaN/Inf (bf16) for
    reduction sizes 576, 640, 768, 1215, 1620, 2430 and 3240, which include Wan's attention
    lengths; `2.12.1+xpu` matched the float64 CPU reference, so the environment is pinned to it.
    Reproducer: `scripts/debug/xpu_softmax_bug_mre.py`; notes: `docs/xpu-softmax-bug-report.md`.
