# Wan2.2-I2V-A14B with SageAttention on one Arc Pro B70: runbook

Use this to run, reproduce or extend the Wan2.2 image-to-video work. Start a new session in this directory with:
*"Read WAN22_I2V_RUNBOOK.md and continue the Wan2.2 I2V work."*
Last updated 2026-11-10. Companion runbooks: `WAN22_T2V_VBENCH2_RUNBOOK.md` (VBench-2.0 on Wan2.2-T2V-A14B, same integration), `QWEN3_OMNI_RUNBOOK.md`, `QWEN3_VL_256K_RUNBOOK.md`.

## 1. Summary

[Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) generates video on a single Arc Pro B70 (32 GB VRAM, 30 GB host RAM) with:

- both 14B experts stored as INT8 weight-only (W8A16, one bf16 scale per output channel), 14 GiB each, with only one expert on the GPU at a time (swap ≈ 3 s);
- the transformer's self-attention routed either to **torch SDPA** (the default diffusers backend on XPU) or to **Sage** (`sage_causal.sycl`, smoothed INT8 Q·Kᵀ, non-causal). Cross-attention, weights, scheduler, seed and everything else are identical between the two arms;
- optionally, the self-attention Q/K/V and FFN linears of every block on **SVDQuant W4A4** (SolarWM's `wan22_svdquant_dense_block` SYCL kernel, rank-32 low-rank branch, section 4.1), 10.7 GiB per expert on the GPU.

**End-to-end result** (official example image and prompt, 544×720, 81 frames, 40 UniPC steps, CFG 3.5, seed 42; 32,130 tokens per self-attention call, 3,200 calls per video; `wan22/results/cat480_{sdpa,sage}.json`):

| linears / attention | denoising, excl. expert loads | speedup | whole run | self-attention | peak VRAM |
|---|---|---|---|---|---|
| INT8 / SDPA (default) | 1648.6 s (41.2 s/step) | — | 1686.9 s | 801.7 s | 17.8 GiB |
| INT8 / Sage INT8 | 1443.5 s (36.1 s/step) | **1.14×** | 1481.7 s (1.14×) | 607.5 s (**1.32×**) | 17.8 GiB |
| INT8 / Sage INT4 | 1323.8 s (33.1 s/step) | **1.25×** | 1363.1 s (1.24×) | 467.2 s (**1.72×**) | 18.0 GiB |
| SVDQuant / SDPA | 1602.3 s (40.1 s/step) | **1.03×** | 1638.0 s (1.03×) | 802.1 s | **14.3 GiB** |
| SVDQuant / Sage INT8 | 1397.6 s (34.9 s/step) | **1.18×** | 1433.2 s (1.18×) | 609.9 s (1.31×) | **14.3 GiB** |

Sage INT8 is the installed `sage_causal.sycl` (smoothed INT8 Q·Kᵀ, K smoothing). Sage INT4 is the smoothed INT4 Q·Kᵀ kernel from commit `c2c1cce` (`QWEN3_OMNI_RUNBOOK.md` 6.6), built into its own directory and selected with `SAGE_BUILD_DIR`; results in `wan22/results/cat480_int4_sage.json`. SVDQuant runs: `cat480_{sdpa,sage}_svdq.json`.

- Self-attention is 49% of denoising time with SDPA; the remaining ≈ 840 s (INT8 linears with per-call dequantisation, cross-attention, norms) is the same in both arms, which caps the end-to-end gain.
- The in-model attention speedup matches the standalone kernel benchmark at this shape (section 3).
- SVDQuant cuts the non-attention time by only 5.5% (847 → 800 s): at M = 32,130 tokens the kernel reaches ≈ 140 TOPS and is 1.27× / 1.23× / 1.04× faster than the W8A16 linears on QKV / FFN up / FFN down (`wan22/results/test_svdq_kernel.log`). Its main gain here is memory (−3.5 GiB per expert).

**Quality, same seed** (`wan22/results/cat480_compare.json`, `cat480_int4_vs_sdpa_compare.json`, `cat480_int4_vs_int8_compare.json`):

| pair | PSNR mean / worst | PSNR frame 0 → 80 | SSIM mean / worst | mean abs diff |
|---|---|---|---|---|
| Sage INT8 vs SDPA | 26.2 / 23.1 dB | 39.0 → 23.1 dB | 0.861 / 0.787 | 6.1 / 255 |
| Sage INT4 vs SDPA | 20.3 / 17.2 dB | 30.4 → 17.2 dB | 0.712 / 0.638 | 15.1 / 255 |
| Sage INT4 vs Sage INT8 | 20.6 / 17.7 dB | 30.4 → 17.7 dB | 0.720 / 0.666 | 14.4 / 255 |
| SVDQuant / SDPA vs SDPA | 18.3 / 15.6 dB | 25.2 → 16.6 dB | 0.655 / 0.586 | 18.7 / 255 |
| SVDQuant / Sage INT8 vs SDPA | 19.0 / 17.1 dB | 25.2 → 19.0 dB | 0.668 / 0.600 | 16.9 / 255 |

Comparison files: `cat480_compare.json` (Sage INT8), `cat480_int4_vs_{sdpa,int8}_compare.json`, `cat480_svdq_vs_sdpa_compare.json`, `cat480_svdq_sage_vs_sdpa_compare.json`.

Kernel-level error at the Wan shape (random inputs with a per-channel K bias, cosine vs bf16 SDPA): INT8 0.99995, INT4 0.986.

- **INT8:** PSNR is 39 dB on the first frame (the conditioning image) and drifts down to ≈ 23–24 dB by the last frame. Visually the two videos show the same scene, subject, composition and motion; neither has artifacts, blur or colour shift. The difference maps concentrate on high-frequency edges (whiskers, fur, sunglasses rims, water texture), i.e. small spatial shifts.
- **INT4:** frame 0 is already 30 dB (texture-level differences on the conditioning frame), and by the end the video follows a visibly different camera path (horizon lower, more spray, the cat framed slightly further right). Each frame is still clean: no artifacts, banding, smearing or colour shift, and the subject is intact. INT4 is about equally far from SDPA and from INT8, so it is a separate trajectory, not a perturbation of either. Mean gradient magnitude (a sharpness proxy, every 8th frame) is 9.46 SDPA / 9.39 INT8 / 8.99 INT4: a 5% drop for INT4 that may be content (more motion blur and spray) rather than detail loss; one video cannot separate the two.
- **SVDQuant:** frame 0 (the conditioning image, where every arm should agree) is only 25 dB, and the difference is visible: grainier, over-textured fur and more speckled water. That is quantisation error in the picture, not trajectory divergence. Later frames follow a different but plausible path (the cat faces the camera at frame 40 instead of turning; more surfboard by frame 80) with no structural artifacts or colour shift; the grain persists slightly. Sharpness proxy 8.70 (SDPA 9.46); with Sage on top 9.07. Adding Sage INT8 does not make it worse (19.0 vs 18.3 dB): the W4A4 linears dominate the error. Per-layer cosine vs fp32 on real weights: 0.995 (rank 0, plain W4A4: 0.994).
- Sage INT8 and INT4 are trajectory divergence: a numerical change in early denoising steps is amplified by the sampler into a different but valid sample. Pixel metrics count that as difference, not as degradation. The size of the divergence follows the kernel error (INT8 ≪ INT4).
- There is no ground truth for image-to-video, so a same-seed comparison measures *divergence from the default*, not quality. Section 6 lists the controls that separate the two.

## 2. Model facts and memory budget

- Two experts of 14B each (`transformer` = high-noise, `transformer_2` = low-noise), switched at `boundary_ratio` 0.9 of the timestep range. Each: 40 layers, 40 heads, head dim 128, multi-head attention (Hq = Hkv), FFN 13,824, `qk_norm` `rms_norm_across_heads`, text dim 4,096 (UMT5). No image encoder in the I2V-A14B checkpoint; the image enters through the VAE latent (36 input channels).
- Tokens per attention call = latent frames × (H/16) × (W/16) with latent frames = (frames − 1)/4 + 1: 81 frames at 480p-area → 32,130–32,760 tokens; at 720p-area → 75,600.
- fp32 checkpoint: 53 GiB per expert, so neither bf16 (28 GiB) nor two experts fit. INT8 W8A16: 14.2 GiB per expert on the GPU.
- Denoising peak: 17.8 GiB (one expert + activations at 32K tokens). UMT5 (bf16, 11 GB) runs once before denoising and is freed; the VAE (fp32) decodes after the expert is unloaded.
- Host RAM peak ≈ 8 GB (weights are loaded straight to the GPU with `safetensors.load_file(device="xpu")`).

## 3. Attention kernel on Wan shapes

Standalone benchmark (non-causal MHA, D = 128, bf16, random inputs; `wan22/results/bench_wan_attention.log`):

| shape | tokens × heads | SDPA | Sage INT8 | flash bf16 | Sage cos vs fp32 |
|---|---|---|---|---|---|
| TI2V-5B | 8,505 × 24 | 9.7 ms | 7.2 ms (1.35×) | 7.6 ms (1.27×) | 0.99995 |
| I2V-A14B 81 frames | 32,760 × 40 | 256.0 ms | 193.6 ms (1.32×) | 205.4 ms (1.25×) | 0.99995 |
| I2V-A14B 153 frames | 60,840 × 40 | 927.0 ms | 663.0 ms (1.40×) | 708.0 ms (1.31×) | 0.99995 |

SDPA sustains 82–92 TFLOPS on these shapes, Sage INT8 114–124 TFLOPS. The speedup grows with sequence length, so 720p (75,600 tokens) should give a larger end-to-end gain than 480p, because attention becomes a larger share of each step.

## 4. How the integration works

- **Attention hook** (`run_wan_i2v.py:install_attention_hook`): replaces `diffusers.models.transformers.transformer_wan.dispatch_attention_fn`. A call is treated as self-attention when `attn_mask is None`, q and k have the same length > 4,096, D = 128 and dtype bf16; with `--arm sage` it goes to `sage_causal.forward(q, k, v, False)` on the `[B, S, H, D]` tensors diffusers already provides (no transposes). Everything else (cross-attention to 512 text tokens) falls through to the original function. Self-attention is timed with XPU events in both arms.
- **INT8 experts** (`wan_int8.py`): `convert` streams the fp32 shards one tensor at a time; every 2-D weight under `blocks.` (attention q/k/v/out, cross-attention, FFN; 400 per expert) becomes int8 + bf16 row scale (absmax / 127), every other tensor becomes bf16. `Int8Linear` dequantises to bf16 right before `F.linear`. `build_expert` builds the model on the meta device in bf16 (the pipeline reads `transformer.dtype` before the lazy load), `load_expert_weights` loads shards directly to the GPU and rebuilds the RoPE module (its tables are non-persistent buffers), `unload_expert` moves everything back to meta.
- **SVDQuant linears** (`wan_svdq.py`): see section 4.1.
- **Lazy experts** (`LazyExperts`): a forward pre-hook loads the expert the pipeline is about to call and unloads the other.
- **Pipeline**: `WanImageToVideoPipeline` subclass with `_execution_device` pinned to XPU; prompt embeddings computed once with the official negative prompt, then UMT5 is freed; `output_type="latent"`, then manual VAE decode (`latents / std + mean`).
- **Outputs** per run in `--out` (default `omni_bench/wan/outputs`): `<tag>_<arm>.npy` (uint8 frames `[F, H, W, 3]`), `.mp4` (system `ffmpeg`, libx264 crf 16, 16 fps) and `.json` (timings, peak VRAM, call count, Sage build directory).

### 4.1 SVDQuant W4A4 linears

- Kernel: `SolarWM/packages/wan22_svdquant_dense_block/kernel.sycl` (module `svdquant_dense_kernel_grf128_n_tile64`, `svdquant_w4a4_gemm`): group-64 symmetric INT4 activations (dynamic, per token, with `x_smooth_inv` and clipping `rho`) and weights on XMX, a bf16 low-rank branch `(x @ lora_a^T) @ lora_b^T` on the **unsmoothed** x, bias, optional GELU(tanh) epilogue and fused INT4 output (`fuse_quant_out`, smoothed with the next layer's `out_smooth_inv`). Weights are passed pre-packed as the DPAS B image `[N/16, K/64, 8, 16]` int32 and scales as fp32 `[K/64, N]`. Requires N % 64 == 0, K % 64 == 0, LoRA rank in {16, 32, 64, 96, 128}. Built with sycl-tla v0.9.3 (`wan22/build_svdq.sh`, section 5.2).
- Layers (the same set SolarWM's `svdquant_linear.py` replaces): per block, Q/K/V fused into one `to_qkv` (N = 15,360, one shared input smoothing; diffusers' `fused_projections` path, q/k RMSNorm unchanged) and the FFN as `SVDQFFN` (up projection with GELU + INT4 output feeding the down projection). Attention output, cross-attention, norms and modulation stay W8A16.
- Conversion (`wan_svdq.py convert`), per linear W [N, K]: smoothing s = max|x|^α / max|W|^(1−α) per input channel (α = 0.5) from calibration maxima; rank-32 SVD of W·s (`torch.svd_lowrank`, q = 48, 6 iterations) → `lora_b = U Σ`, `lora_a = Vᵀ / s`; residual in group-64 INT4 with bf16 scales (absmax / 7). `rho` = 1.0 (no clipping) everywhere. The source weights are the INT8 experts (the fp32 originals had been deleted); the comparison baseline uses the same INT8 weights.
- Calibration (`run_wan_i2v.py --calib-out`): forward pre-hooks record the per-channel max |x| at every block's `attn1.to_q` input, FFN input and FFN-down input (after GELU), over all steps and both CFG passes; run on the default SDPA arm with seed 7 and 33 frames, so the test video is not used.
- Checks: `test_tiny_svdq.py` (tiny model: hooks → conversion → surgery → load; output cos 0.99997 vs INT8), `test_svdq_kernel.py` (real block-20 weights at M = 32,130: kernel = fp32 emulation exactly; per-layer cos vs fp32 and timings vs W8A16 / bf16).

## 5. Code and reproduction

### 5.1 Code in `wan22/`

| file | purpose |
|---|---|
| `wan_svdq.py` | SVDQuant converter (`convert <int8_dir> <calib.pt> <out_dir> [--rank] [--alpha]`), `SVDQLinear`, `SVDQFFN`, expert surgery |
| `build_svdq.sh` | `build.sh` without its sycl-tla v0.8 check, for the SVDQuant kernel (sycl-tla v0.9.3) |
| `test_tiny_svdq.py`, `test_svdq_kernel.py` | SVDQuant self-test on the tiny model; kernel accuracy / speed on real A14B weights |
| `run_wan_t2v_batch.py`, `vbench/` | Wan2.2-T2V-A14B batch generation for VBench-2.0 (see `WAN22_T2V_VBENCH2_RUNBOOK.md`) |
| `wan_int8.py` | INT8 converter (`convert <diffusers_dir> <out_dir>`), `Int8Linear`, build / load / unload of an expert |
| `run_wan_i2v.py` | end-to-end I2V run, `--arm sdpa\|sage`, `--linear int8\|svdq`, `--calib-out`; writes frames, MP4 and timing JSON |
| `compare_videos.py` | per-frame PSNR, SSIM (Gaussian 11×11, σ 1.5, luma) and mean abs diff; side-by-side MP4 (A \| B \| \|A − B\| × 8) |
| `bench_wan_attention.py` | standalone SDPA / Sage / flash benchmark on Wan shapes with an exact fp32 check on 512 sampled rows |
| `test_tiny.py` | self-test on a 2-layer random Wan transformer: conversion, INT8 vs bf16, Sage vs SDPA, unload / reload |
| `verify_and_prune.py` | checks every fp32 source tensor is present in the INT8 index, then deletes the fp32 expert folders |
| `results/` | timing JSONs, the comparison JSON and the kernel benchmark log behind sections 1 and 3 |

### 5.2 Setup

Environment: `~/omni_env` (torch 2.14.1+xpu), the `omni_bench` workspace at `/home/gta/omni_bench` (restore steps in `qwen3_omni/README.md`); copy `wan22/*.py` to `omni_bench/wan/`. diffusers is installed in an isolated directory so `omni_env` is not modified. Machine safety rules are in `QWEN3_OMNI_RUNBOOK.md` section 5: run GPU jobs through `rss_guard.sh`, one GPU job at a time, and do not use `expandable_segments`.

```bash
# 1. diffusers 0.42.1 + ftfy (prompt cleaning), isolated; every Wan command below needs this PYTHONPATH
~/omni_env/bin/pip install --no-deps --target /home/gta/pylibs/diffusers diffusers==0.42.1 ftfy wcwidth
export PYTHONPATH=/home/gta/pylibs/diffusers

# 2. weights: 117 GB download (experts are fp32), then INT8 (2 x 14 GB; disk-bound, a few minutes)
hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers --local-dir /home/gta/models/wan22-i2v-a14b-diffusers
cd /home/gta/omni_bench
./env.sh python wan/wan_int8.py convert /home/gta/models/wan22-i2v-a14b-diffusers /home/gta/models/wan22-i2v-a14b-int8
# optional: verify, then delete the 2 x 53 GB fp32 experts (text encoder, VAE, tokenizer, scheduler stay)
python3 wan/verify_and_prune.py /home/gta/models/wan22-i2v-a14b-diffusers /home/gta/models/wan22-i2v-a14b-int8

# 3. Sage kernel (AOT for bmg-g31, about 45 s); run from this folder
B=/home/gta/omni_bench/kernel
bash $B/build.sh $B/build $PWD/sage_causal.sycl sage_causal
```

```bash
# 4. SVDQuant kernel (sycl-tla v0.9.3: git clone --depth 1 --branch v0.9.3 https://github.com/intel/sycl-tla.git ~/.cache/sycl-tla-v0.9.3)
cp wan22/build_svdq.sh $B/; CUTLASS_DIR=$HOME/.cache/sycl-tla-v0.9.3 bash $B/build_svdq.sh $B/build_svdq \
  /home/gta/SolarWM/packages/wan22_svdquant_dense_block/kernel.sycl svdquant_dense_kernel_grf128_n_tile64
```

Environment variables read by the scripts: `WAN_DIR` (diffusers checkpoint), `WAN_INT8_DIR` (INT8 experts), `WAN_SVDQ_DIR` (SVDQuant experts, default `/home/gta/models/wan22-i2v-a14b-svdq`), `SVDQ_BUILD_DIR` (default `omni_bench/kernel/build_svdq`), `SAGE_BUILD_DIR` / `FLASH_BUILD_DIR` (kernel builds, default `omni_bench/kernel/build{,_flash}`).

### 5.3 Reproduce sections 1 and 3

```bash
cd /home/gta/omni_bench; export PYTHONPATH=/home/gta/pylibs/diffusers
# self-test (1 min) and short smoke run (17 frames, 4 steps, both experts used; about 1 min)
./env.sh python wan/test_tiny.py
scripts/rss_guard.sh 24 logs/wan_smoke.log -- ./env.sh python wan/run_wan_i2v.py --arm sage --tag smoke --frames 17 --steps 4

# kernel benchmark on Wan shapes (section 3; needs the flash build from QWEN3_VL_256K_RUNBOOK.md 5.2)
scripts/rss_guard.sh 24 logs/bench_wan_attention.log -- ./env.sh python wan/bench_wan_attention.py

# end-to-end, both arms (about 28 + 25 min), then the comparison (section 1)
for arm in sdpa sage; do
  scripts/rss_guard.sh 24 logs/wan_i2v_cat480_$arm.log -- ./env.sh python wan/run_wan_i2v.py --arm $arm --tag cat480
done
./env.sh python wan/compare_videos.py wan/outputs/cat480_sdpa.npy wan/outputs/cat480_sage.npy --out wan/outputs/cat480

# INT4 arm (about 23 min): build the c2c1cce kernel into its own directory and select it
mkdir -p kernel/int4_src
git -C /home/gta/SolarWM show c2c1cce:solar/kernels/fused_rope_prope_sage/sage_causal.sycl > kernel/int4_src/sage_causal.sycl
bash kernel/build.sh kernel/build_int4 $PWD/kernel/int4_src/sage_causal.sycl sage_causal
SAGE_BUILD_DIR=$PWD/kernel/build_int4 scripts/rss_guard.sh 24 logs/wan_i2v_cat480_int4.log -- \
  ./env.sh python wan/run_wan_i2v.py --arm sage --tag cat480_int4
./env.sh python wan/compare_videos.py wan/outputs/cat480_sdpa.npy wan/outputs/cat480_int4_sage.npy --out wan/outputs/cat480_int4_vs_sdpa
./env.sh python wan/compare_videos.py wan/outputs/cat480_sage.npy wan/outputs/cat480_int4_sage.npy --out wan/outputs/cat480_int4_vs_int8
```

SVDQuant arms (calibration about 10 min, conversion a few minutes, each run about 27 / 24 min):

```bash
./env.sh python wan/test_tiny.py && ./env.sh python wan/test_tiny_svdq.py
scripts/rss_guard.sh 24 logs/test_svdq_kernel.log -- ./env.sh python wan/test_svdq_kernel.py
scripts/rss_guard.sh 24 logs/wan_calib.log -- ./env.sh python wan/run_wan_i2v.py --arm sdpa --tag calib --seed 7 --frames 33 \
  --calib-out wan/outputs/calib_amax.pt
scripts/rss_guard.sh 24 logs/wan_svdq_convert.log -- ./env.sh python wan/wan_svdq.py convert /home/gta/models/wan22-i2v-a14b-int8 \
  wan/outputs/calib_amax.pt /home/gta/models/wan22-i2v-a14b-svdq          # 2 x 11 GB, peak host RAM 19 GB
for arm in sdpa sage; do
  scripts/rss_guard.sh 24 logs/wan_i2v_cat480_${arm}_svdq.log -- ./env.sh python wan/run_wan_i2v.py --arm $arm --linear svdq --tag cat480
done
./env.sh python wan/compare_videos.py wan/outputs/cat480_sdpa.npy wan/outputs/cat480_sdpa_svdq.npy --out wan/outputs/cat480_svdq_vs_sdpa
./env.sh python wan/compare_videos.py wan/outputs/cat480_sdpa.npy wan/outputs/cat480_sage_svdq.npy --out wan/outputs/cat480_svdq_sage_vs_sdpa
```

Side-by-side MP4 layout for every comparison: first video | second video | |first − second| × 8.

Two builds of `sage_causal` cannot be compared in one Python process: both are module `sage_causal`, and the second import silently returns the first. Run each build in its own process.

Other settings: `--max-area $((720*1280))` for 720p (75,600 tokens; about 1.5 h per arm), `--image`, `--prompt`, `--frames`, `--steps`, `--guidance`, `--seed`. Another kernel build is selected with `SAGE_BUILD_DIR`; use a distinct `--tag` so outputs are not overwritten (the JSON records the build directory).

## 6. Known limitations and next steps

- **Same-seed divergence is not a quality score.** To show how much of the 26 dB divergence is specific to Sage, run a control arm that is exact up to bf16 rounding (e.g. SDPA math backend or a different reduction order) and compare it with the default the same way. If it lands near the same PSNR, the divergence is the sampler's sensitivity rather than INT8 error.
- **Reference-free quality metrics** (CLIP text–frame similarity, frame-quality or temporal-consistency scores) and more prompts / seeds are needed to report a quality delta with an uncertainty.
- **INT4 vs INT8 trade-off:** INT4 is 1.24× end to end against 1.14× for INT8 and produces a clean but clearly different video (closer to a different seed than to the default). Whether it loses quality on average (sharpness, prompt adherence, temporal consistency) needs the reference-free metrics above over several prompts and seeds; on Qwen3-Omni the same INT4 kernel lost 7.4 RULER points at 64K.
- **SVDQuant quality and speed:** the W4A4 error is visible on the conditioning frame. Candidates, each one 27-minute run: α = 0.6–0.7, rank 64, FFN down kept at W8A16 (its post-GELU input is the hardest to quantise), per-layer clipping `rho` from calibration (SolarWM's checkpoints use per-timestep-bin `rho`), conversion from the fp32 weights instead of INT8. Speed: the kernel was tuned for SolarWM's 5B Stage-2 shapes (M = 1,215); at M = 32K it needs larger M tiles to approach the XMX INT4 peak.
- **720p** (75,600 tokens) should show a larger end-to-end gain (kernel 1.40× at 60,840 tokens) and has not been run.
- The INT8 linears dequantise to bf16 on every call; that cost is identical in both arms but dilutes the end-to-end speedup. A fused W8A16 GEMM would make attention a larger share of each step.
- Wan2.2 recommends fp32 for `time_embedder`, `scale_shift_table` and the norms; this setup keeps them in bf16 (the same in both arms).
