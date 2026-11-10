# Wan2.2-I2V-A14B with SageAttention on one Arc Pro B70: runbook

Use this to run, reproduce or extend the Wan2.2 image-to-video work. Start a new session in this directory with:
*"Read WAN22_I2V_RUNBOOK.md and continue the Wan2.2 I2V work."*
Last updated 2026-11-10. Companion runbooks: `QWEN3_OMNI_RUNBOOK.md`, `QWEN3_VL_256K_RUNBOOK.md`.

## 1. Summary

[Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) generates video on a single Arc Pro B70 (32 GB VRAM, 30 GB host RAM) with:

- both 14B experts stored as INT8 weight-only (W8A16, one bf16 scale per output channel), 14 GiB each, with only one expert on the GPU at a time (swap ≈ 3 s);
- the transformer's self-attention routed either to **torch SDPA** (the default diffusers backend on XPU) or to **Sage** (`sage_causal.sycl`, smoothed INT8 Q·Kᵀ, non-causal). Cross-attention, weights, scheduler, seed and everything else are identical between the two arms.

**End-to-end result** (official example image and prompt, 544×720, 81 frames, 40 UniPC steps, CFG 3.5, seed 42; 32,130 tokens per self-attention call, 3,200 calls per video; `wan22/results/cat480_{sdpa,sage}.json`):

| | SDPA (default) | Sage INT8 | speedup |
|---|---|---|---|
| self-attention | 801.7 s | 607.5 s | **1.32×** |
| denoising, excluding expert loads | 1648.6 s (41.2 s/step) | 1443.5 s (36.1 s/step) | **1.14×** |
| whole run (text encode + denoise + VAE decode) | 1686.9 s | 1481.7 s | **1.14×** |
| peak VRAM during denoising | 17.8 GiB | 17.8 GiB | — |

- Self-attention is 49% of denoising time with SDPA; the remaining ≈ 840 s (INT8 linears with per-call dequantisation, cross-attention, norms) is the same in both arms, which caps the end-to-end gain.
- The in-model attention speedup matches the standalone kernel benchmark at this shape (section 3).

**Quality, Sage vs SDPA, same seed** (`wan22/results/cat480_compare.json`):

| metric | mean | worst frame |
|---|---|---|
| PSNR | 26.2 dB | 23.1 dB |
| SSIM (luma) | 0.861 | 0.787 |
| mean abs difference | 6.1 / 255 | — |

- PSNR is 39 dB on the first frame (the conditioning image) and drifts down to ≈ 23–24 dB by the last frame.
- Visually the two videos show the same scene, subject, composition and motion; neither has artifacts, blur or colour shift. The difference maps concentrate on high-frequency edges (whiskers, fur, sunglasses rims, water texture), i.e. small spatial shifts. This is trajectory divergence: a small numerical change in early denoising steps is amplified by the sampler into a slightly different but equally valid sample. Pixel metrics count that as difference, not as degradation.
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
- **Lazy experts** (`LazyExperts`): a forward pre-hook loads the expert the pipeline is about to call and unloads the other.
- **Pipeline**: `WanImageToVideoPipeline` subclass with `_execution_device` pinned to XPU; prompt embeddings computed once with the official negative prompt, then UMT5 is freed; `output_type="latent"`, then manual VAE decode (`latents / std + mean`).
- **Outputs** per run in `--out` (default `omni_bench/wan/outputs`): `<tag>_<arm>.npy` (uint8 frames `[F, H, W, 3]`), `.mp4` (system `ffmpeg`, libx264 crf 16, 16 fps) and `.json` (timings, peak VRAM, call count, Sage build directory).

## 5. Code and reproduction

### 5.1 Code in `wan22/`

| file | purpose |
|---|---|
| `wan_int8.py` | INT8 converter (`convert <diffusers_dir> <out_dir>`), `Int8Linear`, build / load / unload of an expert |
| `run_wan_i2v.py` | end-to-end I2V run, `--arm sdpa\|sage`, writes frames, MP4 and timing JSON |
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

Environment variables read by the scripts: `WAN_DIR` (diffusers checkpoint), `WAN_INT8_DIR` (INT8 experts), `SAGE_BUILD_DIR` / `FLASH_BUILD_DIR` (kernel builds, default `omni_bench/kernel/build{,_flash}`).

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
```

Other settings: `--max-area $((720*1280))` for 720p (75,600 tokens; about 1.5 h per arm), `--image`, `--prompt`, `--frames`, `--steps`, `--guidance`, `--seed`. Another kernel build is selected with `SAGE_BUILD_DIR`; use a distinct `--tag` so outputs are not overwritten (the JSON records the build directory).

## 6. Known limitations and next steps

- **Same-seed divergence is not a quality score.** To show how much of the 26 dB divergence is specific to Sage, run a control arm that is exact up to bf16 rounding (e.g. SDPA math backend or a different reduction order) and compare it with the default the same way. If it lands near the same PSNR, the divergence is the sampler's sensitivity rather than INT8 error.
- **Reference-free quality metrics** (CLIP text–frame similarity, frame-quality or temporal-consistency scores) and more prompts / seeds are needed to report a quality delta with an uncertainty.
- **INT4 Q·Kᵀ** (the `c2c1cce` kernel from `QWEN3_OMNI_RUNBOOK.md` 6.6) has cosine 0.986 vs SDPA at the Wan shape, against 0.99995 for INT8; an end-to-end run with it shows whether that error becomes visible in the video.
- **720p** (75,600 tokens) should show a larger end-to-end gain (kernel 1.40× at 60,840 tokens) and has not been run.
- The INT8 linears dequantise to bf16 on every call; that cost is identical in both arms but dilutes the end-to-end speedup. A fused W8A16 GEMM would make attention a larger share of each step.
- Wan2.2 recommends fp32 for `time_embedder`, `scale_shift_table` and the norms; this setup keeps them in bf16 (the same in both arms).
