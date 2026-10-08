# Wan2.2 Stage2 VAE decode debug plan (Intel XPU / CUDA reference)

Cross-platform investigation for `Stage2 VAE decode produced non-finite pixels` on
XPU. CUDA produces a **reference bundle** on a separate machine; XPU runs the same
script in probe/compare mode.

## Scripts

| Script | Where to run |
|--------|----------------|
| [`scripts/debug/generate_wan22_stage2_cuda_reference.py`](../../scripts/debug/generate_wan22_stage2_cuda_reference.py) | CUDA GPU (reference) or XPU (probe) |

### CUDA reference machine

```bash
source ~/.bashrc   # proxy if needed
cd /path/to/SolarWM
source .venv-wan22-xpu/bin/activate   # or CUDA Wan venv with flash-attn

export SOLAR_MODEL_ROOT=/path/to/SolarWM-models
export SOLAR_DATA_HOME=/path/to/SolarWM-Data
export SOLAR_DATA_ROOT="$SOLAR_DATA_HOME/releases-v1"
export SOLAR_OUTPUT_ROOT=/path/to/wan22-decode-reference

python scripts/debug/generate_wan22_stage2_cuda_reference.py \
  --device cuda \
  --output-dir "$SOLAR_OUTPUT_ROOT" \
  --save-rollout-trace
```

`--save-rollout-trace` writes `rollout_trace/` (conditions, `initial_noise`, per-step
`flow` / `x0` / `renoise` / `latents_*`, per-chunk committed latents and KV /
cross-attn snapshots). Use this to bisect XPU diffusion without device RNG.

Copy the output directory to the XPU host (e.g. `rsync` / `scp`).

### XPU probe machine

**Full rollout + compare** (diffusion + VAE; latents will differ device-to-device):

```bash
export SOLAR_MODEL_ROOT=...
export SOLAR_DATA_ROOT=.../releases-v1

python scripts/debug/generate_wan22_stage2_cuda_reference.py \
  --device xpu \
  --weights ema \
  --output-dir outputs/debug/vae-xpu-probe \
  --compare-reference outputs/wan22-cuda-decode-reference
```

Read `probe_report.json`:

- `rollout_latents_vs_cuda` — diffusion rollout parity (same seed; expect small diffs only if stack matches).
- `rollout_decode_pixels_vs_cuda` — **misleading** if latents differ; each side decoded its own rollout.
- `vae_isolation` — **same** `latents_bf16.pt` from CUDA reference decoded on XPU (true VAE test).

**VAE-only (fast)** — skip diffusion; load CUDA `latents_bf16.pt`:

```bash
python scripts/debug/generate_wan22_stage2_cuda_reference.py \
  --device xpu \
  --decode-reference-latents \
  --output-dir outputs/debug/vae-isolation-xpu \
  --compare-reference outputs/wan22-cuda-decode-reference
```

In `probe_report.json` → `vae_isolation.production_vs_cuda_saved_decode` (and `bf16_autocast_direct`).
`match: true` only means shapes/finite; use **`mean_abs_diff`** / **`max_abs_diff`** on pixels in `[-1, 1]`.

**Rollout trace replay** (remove XPU RNG; bisect numerics):

```bash
python scripts/debug/generate_wan22_stage2_cuda_reference.py \
  --device xpu \
  --weights ema \
  --output-dir outputs/debug/rollout-trace-replay \
  --compare-reference outputs/wan22-cuda-decode-reference \
  --replay-rollout-trace outputs/wan22-cuda-decode-reference/rollout_trace \
  --replay-mode kv_chunks
```

`--replay-mode`:

| Mode | Injected from CUDA trace |
|------|---------------------------|
| `noise` | `initial_noise` + per-step `renoise` (conditions still from XPU `_conditions`) |
| `conditions` | above + `first_latent`, `prompt_embeds`, camera |
| `kv_chunks` | above + KV / cross-attn state before each chunk after the first |

`probe_report.json` → `rollout_trace_replay` lists per-step and final latent diffs vs the trace.

**Determinism:** replay removes **RNG** (and, with `conditions` / `kv_chunks`, VAE-encode and
cache drift inputs). It does **not** guarantee bit-identical CUDA vs XPU outputs: bf16
kernels, FlashAttention vs SDPA, and accumulation order still differ. Expect **close**
tensors if the stack matches; use per-step `max_abs_diff` in `rollout_trace_replay` to find
the first diverging forward.

### Partial trace replay (9 latent frames / chunks 0–2)

```bash
python scripts/debug/generate_wan22_stage2_cuda_reference.py \
  --device xpu --weights ema \
  --output-dir outputs/debug/rollout-chunk0-2-replay \
  --compare-reference outputs/wan22-cuda-decode-reference \
  --replay-rollout-trace outputs/wan22-cuda-decode-reference/rollout_trace \
  --replay-mode kv_chunks \
  --rollout-latent-frames 9 --output-latent-frames 9
```

**Smallest forward bisect** (one `provider.diffusion` call, same as rollout):

```bash
python scripts/debug/wan22_stage2_trace_forward_bisect.py \
  --device xpu \
  --trace-dir outputs/wan22-cuda-decode-reference/rollout_trace \
  --chunk 0 --step 0
```

With `kv_chunks` replay, divergence shows up on **chunk 0 step 0 `flow`** (~0.4 max abs
diff in bf16) — the first diffusion forward with empty KV. Inputs (`latents_in`) match the
trace; the gap is inside the model forward.

**CUDA FlashAttention vs XPU SDPA:** `attention()` uses FlashAttention on CUDA when
installed, and `torch.nn.functional.scaled_dot_product_attention` elsewhere
(`src/solarwm/backends/wan22/runtime/modeling/attention.py`). A CUDA trace captured with
Flash is not an apples-to-apples reference for XPU. For parity testing, set
`SOLARWM_WAN22_ATTENTION=sdpa` on **both** machines (re-capture the trace on CUDA with SDPA,
or run `wan22_stage2_trace_forward_bisect.py` on CUDA with the same env and compare to XPU).

Measured on XPU (`torch 2.14.0+xpu`), SDPA at the **diffusion** shapes (24 heads,
`head_dim=128`, `Lq=1215`, `Lk` 512/1215/2430/3240/8505) matches a float64 reference to
`max_abs_diff` ~1e-4 in bf16, so attention is **not** the source of the rollout gap. The
remaining ~0.018 mean `flow` difference is consistent with accumulated bf16 kernel
differences (Flash vs SDPA) across ~30 blocks. bf16 GEMM accumulation on XPU is identical to
CPU (same `mean_abs_diff` vs float64), so it is not a lower-precision-accumulate issue.

## 8. Root cause of bad XPU pixels: `softmax` on XPU is wrong for some reduction sizes

`scripts/debug/wan22_vae_decode_device_probe.py` compares VAE decode against a
**device-independent CPU fp32 oracle**, so it needs no CUDA host:

```bash
python scripts/debug/wan22_vae_decode_device_probe.py --device xpu \
  --latents outputs/wan22-cuda-decode-reference/latents_bf16.pt \
  --latent-frames 2 --crop 0 --skip-primitives \
  --cuda-decode outputs/wan22-cuda-decode-reference/decode_cuda_production.pt

# shape sweeps for the failing primitive
python scripts/debug/wan22_vae_decode_device_probe.py --device xpu --op-sweep
python scripts/debug/wan22_vae_decode_device_probe.py --device xpu --conv-sweep
```

Findings (decode is causal in time, so a 2-latent prefix decodes the first 5 pixel frames
exactly and can be compared against the full CUDA decode):

| Check | Result |
|-------|--------|
| CPU fp32 decode vs `decode_cuda_production.pt` | mean **0.0015** — the CUDA bundle is self-consistent |
| XPU bf16 decode vs CPU fp32, full `30x54` latents | mean **0.214**, max **2.0** |
| XPU fp32 (no autocast) vs CPU fp32, full latents | mean **0.069** — not a bf16 problem |
| XPU decode repeated twice | `max_abs_diff` **0.0** — fully deterministic |
| Latent crop sweep (2 frames) | `16`, `20`, `28`, `30` clean (~0.0013); **`24` broken** (0.185); full `30x54` broken |
| `conv2d` / `conv3d` sweep at decoder shapes | clean (bf16 ~1e-3, fp32 ~1e-6) |
| `F.normalize` (`RMS_norm`) sweep | clean |
| bf16 GEMM vs CPU bf16 GEMM | identical |

The latent crop that breaks (`24x24`) and the full size (`30x54`) correspond to VAE
`AttentionBlock` sequence lengths **576** and **1620** with `head_dim = 640` (`dim=160`,
`dim_mult[-1]=4`); the clean crops are 256/400/784/900. Isolating the primitive:

```text
torch.softmax(x, dim=-1) on XPU vs float64, [1, 1, S, S]
  S =  128  256  320  384  400  405  512  784  810  900 1024 2048 4096  -> fp32 error ~1e-9  (clean)
  S =  576  640  768 1215 1620 2430 3240                                -> fp32 mean error 3e-4 .. 1e-3, max up to 1.4e-2
  bf16 additionally returns NaN for S = 1215, 2430, 3240, 8505
```

So **`torch.softmax` on XPU is incorrect for particular last-dim sizes**, deterministically,
in fp32 as well as bf16. SDPA inherits it: at `head_dim=640`, SDPA is clean at `seq=900` and
broken at `seq=576`/`1620`, matching the VAE decode bisect exactly.

Replacing the VAE `AttentionBlock` SDPA call with an explicit matmul+softmax
(`--patch-vae-attention`) does **not** help, because the manual path hits the same XPU
`softmax` bug (and adds bf16 error in the QK matmul under autocast). A workaround has to
avoid the bad reduction sizes or run that softmax off-XPU.

### Standalone reproducer and torch version bisect

[`scripts/debug/xpu_softmax_bug_mre.py`](../../scripts/debug/xpu_softmax_bug_mre.py) needs
only `torch` (fixed inputs, float64 CPU reference, exit code 1 on failure):

```bash
python scripts/debug/xpu_softmax_bug_mre.py                   # softmax size sweep, fp32 + bf16
python scripts/debug/xpu_softmax_bug_mre.py --also-sdpa       # + SDPA at the VAE shape
python scripts/debug/xpu_softmax_bug_mre.py --include-large    # + sizes 4096, 8505
```

Bisected across the Intel XPU wheels on `https://download.pytorch.org/whl/xpu`
(clean `uv` venvs, Python 3.12, same B-series device):

| torch | `torch.version.xpu` | softmax sweep | SDPA (1 head, `d=640`) |
|-------|---------------------|---------------|------------------------|
| 2.14.0+xpu | 20260100 | **FAIL** 576, 640, 768, 1215, 1620, 2430, 3240 (bf16 NaN at 1215/2430/3240/8505) | **FAIL** seq 576, 1620 |
| 2.13.0+xpu | 20260000 | **FAIL**, byte-identical to 2.14.0 | not run |
| **2.12.1+xpu** | 20250302 | **PASS** all 21 sizes incl. 4096/8505 | **PASS** (fp32 ~1e-7, bf16 ~1e-4) |
| 2.11.0+xpu | 20250302 | PASS | not run |
| 2.10.0+xpu | 20250301 | PASS | not run |

The regression landed between **2.12.1 and 2.13.0**, which is also where the bundled oneAPI
version jumps from `2025.x` to `2026.x` — consistent with a new SYCL/oneDNN softmax kernel.

**Newest known-good version: `torch==2.12.1+xpu`.** Bisect venvs live under
`~/xpu-torch-bisect/` (`v213`, `v2121`, `v2110`, `v2100`).

## Goal

Determine whether bad pixels come from **latents**, **VAE numerics on XPU**, or
**decode path** (autocast / streaming / dtype). Fix in shared `components.py` /
`stage2.py` where possible so CUDA behavior is unchanged.

## Steps

### 1. Pin down the failure surface

On the failing run, before the check in `_stage2_generated_sample` (~937–939):

- `torch.isfinite(output_latents).float().mean()`
- `output_latents` shape, dtype, min/max/absmax
- Decode path: `direct` vs `continuous_cached_tiles` (`output_latent_frames > 60`)

If latents are not finite → debug diffusion / self-forcing (§4).
If latents are finite → VAE only (§2–3).

### 2. Isolate the VAE (cross-platform)

1. **Round-trip**: example shard → `encode` → `decode` on CPU, CUDA, XPU.
2. **Saved latents**: decode the same `latents_bf16.pt` on CPU (reference) vs XPU.
3. **Bisect** `modeling/vae.py` if XPU-only failure.

### 3. Likely levers (CUDA-safe)

1. **VAE decode autocast** (`Wan5BVAE.decode` in `components.py`): uses
   `torch.autocast(device_type=clip.device.type, dtype=clip.dtype)`; try fp32 decode on XPU only if probes show finite-but-wrong pixels.
2. **Stage2 self-forcing autocast** (`stage2.py`): must include **xpu** —
   `enabled=provider.device.type in ("cuda", "xpu")` (fixed 2026-09-17).
3. **Streaming vs direct**: smoke rollout (39 latents) uses **direct** decode (`< 60` latent chunk). Longer runs use `decode_streaming` — bisect if artifacts appear only above the threshold.
4. **Optional CPU VAE decode** behind a config flag if XPU VAE remains broken.

### 4. If latents are non-finite

- Log latent stats; verify first-frame VAE encode on XPU.
- Compare same seed on CUDA (reference bundle latents vs XPU latents).

### 5. Done when

- `finite_fraction == 1.0` after decode on XPU for smoke config.
- Full `solarwm infer` writes `COMPLETE.json` and `video.mp4`.
- CUDA regression unchanged (FlashAttention path via `attention()` router).
- Session log updated in [`intel-b580-wan22-stage2.md`](intel-b580-wan22-stage2.md).

## B70 probe snapshot (2026-09-17)

First XPU run: `outputs/debug/vae-xpu-probe/` (see `metadata.json`).

- Rollout latents: **39** frames, `finite_fraction` ≈ 1.0 (not bit-exact).
- **`bf16_autocast_direct`**: pixels **fully finite** (`ok: true`) — matches production `Wan5BVAE.decode` autocast path.
- **`fp32_no_autocast_direct`**: mostly non-finite — weight dtype bf16 without autocast is not viable on XPU.
- CUDA reference still needed to compare latent values and pixel diffs (`--compare-reference`).

**Resolution (2026-09-17):** Full infer failed until Stage2 self-forcing enabled `torch.autocast` on
**XPU** (it was CUDA-only). Unstable latents → bad VAE pixels. After
`enabled=provider.device.type in ("cuda", "xpu")` in `stage2.py`, smoke infer wrote MP4s
(`RUN_ID=wan22-b70-smoke-20260917-212926`).

## 6. Regional blue noise (finite pixels, wrong appearance)

**Symptom:** Scene matches the caption globally but **patchy saturated blue noise** appears
in the middle of the frame; **`video.mp4` frame 0** is affected. `compare.mp4` is
side-by-side **reference pixels (left)** vs **decoded generation (right)** (`_encode_compare_mp4`).

Latent TI2V pinning (`_restore_first`, `output[:, 0] = first_latent`) fixes **latent
index 0**, not necessarily **pixel frame 0** after a **temporal** VAE decode of a full
latent sequence — test that explicitly.

### Decision tree (run in order)

| Step | What to check | If positive → |
|------|----------------|---------------|
| A | `compare.mp4` frame 0: left OK, right blue | Generation/decode path (not source video) |
| B | `torch.allclose(output_latents[:, 0], first_latent)` immediately before decode | If false → self-forcing pin bug on XPU; if true → VAE or temporal coupling |
| C | `decode(output_latents[:, :1])` vs `decode(full)[:, :1]` on XPU | If they differ → temporal VAE uses neighbors; compare to CUDA |
| D | `encode(pixels[:, :1])` → `decode` only (no diffusion) | If blue here → **VAE encode/decode on XPU**; skip diffusion audit |
| E | Decode `latents_bf16.pt` from probe on CUDA vs XPU | Latent vs VAE isolation |
| F | Later frames only (frame 0 clean) | Diffusion / KV cache / camera — not first-frame pin |

### Recommended commands

**VAE-only round-trip** (extend probe script or one-off REPL with same `Wan5BVAE` weights):

```bash
# Full probe (latents + decode variants); add a small REPL block for encode→decode
# on pixels[:, :1] if you need VAE-only without diffusion.
python scripts/debug/generate_wan22_stage2_cuda_reference.py \
  --device xpu \
  --output-dir outputs/debug/vae-xpu-first-frame
```

**Frame extract for eyeball + PSNR** (requires `ffmpeg`):

```bash
ffmpeg -y -i outputs/.../ema_self_forcing_nfe4/slot-000000/compare.mp4 \
  -frames:v 1 /tmp/compare-f0.png
```

**CUDA-path audit (infer)** — start here, then spot-check hits outside training:

```bash
rg 'device\.type == "cuda"|torch\.cuda|enabled=.*cuda' \
  src/solarwm/backends/wan22/runtime/stage2.py \
  src/solarwm/backends/wan22/runtime/inference.py \
  src/solarwm/backends/wan22/runtime/components.py \
  src/solarwm/backends/wan22/runtime/distributed.py \
  src/solarwm/backends/wan22/runtime/modeling/
```

Known infer-relevant sites to verify XPU parity: `attention()` router (SDPA on XPU),
`Wan5BVAE.decode` autocast `device_type`, `distributed` / `inference` `empty_cache`
(XPU should call `torch.xpu.empty_cache()` where CUDA is cleared today), Stage2 adapter
device construction (`distributed.resolve_inference_device`).

### B70 result (2026-09-17, compare step A)

- Left (source) clean; right blue on frame 0; blue **oscillates** over time.
- **Probe hygiene:** pin checks must use `first_latent` from the **same** rollout, not a second
  `_conditions()` (repeat encode on XPU can disagree with the first).
- **Final `_restore_first` experiment (reverted):** training-parity restore on inference output
  made EMA video **worse** visually. Reverted in `stage2.py`; keep in-loop `_restore_first` only.

**Post-fix PNGs** (`outputs/debug/frame0-diagnose-post-fix/frame0_diagnose/`):

- `vae_roundtrip_t0.png` — blue patches throughout (pure VAE encode→decode on XPU).
- `latent0_only_decode_t0.png` — partial blue; scene still readable.
- `rollout_decode_t0.png` — heavy noise; source not visible.

→ Prioritize **VAE / full-sequence temporal decode** on XPU and CUDA reference; not another
restore-first change without CUDA baseline.

### Other hypotheses (lower priority until A–D ruled out)

- **Mild non-finite latents** (`finite_fraction` ≈ 1 but not bit-exact) → speckle after VAE.
- **bf16 diffusion** with good enough `isfinite` checks → try fp32 flow on chunk 0 only (experiment).
- **Wrong color range** in MP4 encode (`minus_one_one` in compare path) — usually global tint, not patchy blue.
- **EMA vs live** — for Intel work, judge **ema** only; re-run live only when checking checkpoint parity.

## 7. Live status — CUDA reference vs XPU (2026-09-19)

CUDA reference: `outputs/wan22-cuda-decode-reference/` (good infer video on CUDA host).
XPU probes: `outputs/debug/vae-xpu-probe/`, `outputs/debug/vae-isolation-xpu/`.

| Check | `mean_abs_diff` (pixels) | `max_abs_diff` | Interpretation |
|-------|--------------------------|----------------|----------------|
| Rollout latents CUDA vs XPU (`vae-xpu-probe`) | **0.74** (latent space) | **8.6** | **Diffusion / self-forcing on XPU ≠ CUDA** (same `noise_seed`, `ema`). |
| Rollout decode pixels (each side’s latents) | **0.64** | **2.0** | Confounded by latent mismatch; not a VAE-only signal. |
| **VAE isolation** — XPU decode of **CUDA** `latents_bf16.pt` vs CUDA `decode_cuda_fp32_direct.pt` | **0.43** (`production_wan5b_decode`) | **2.0** | **VAE on XPU does not match CUDA** on identical latents. |
| Same, `bf16_autocast_direct` probe | **0.44** | **2.0** | Same conclusion (production path uses bf16 autocast). |

**Conclusion:** Two independent gaps:

1. **Rollout latents** — fix XPU diffusion (attention, autocast, dtype, missed CUDA-only ops).
2. **VAE decode** — fix or workaround (CUDA reference latents → XPU `Wan5BVAE.decode`); consider CPU VAE decode for XPU infer, fp32 VAE, or layer bisect in `modeling/vae.py`.

**Next actions**

1. On CUDA: decode `latents_bf16.pt` with `Wan5BVAE.decode` (production path) and save pixels; compare to `decode_cuda_fp32_direct.pt` to confirm reference tensor baseline.
2. On XPU: frame-0 / per-frame `mean_abs_diff` from isolation decode vs CUDA (script or `--frame0-diagnose` with CUDA latents).
3. Diffusion: single self-forcing step dump (flow / x0) CUDA vs XPU with frozen weights and same caches — after VAE workaround if needed for usable video.

## Reference bundle layout

```text
<output-dir>/
  metadata.json           # config digest, sample_id, seeds, tensor stats
  latents_bf16.pt         # CPU, rollout output
  rollout_schedule.json
  decode_cuda_production.pt      # Wan5BVAE.decode (infer path); preferred VAE reference
  decode_cuda_fp32_direct.pt   # weight-dtype direct decode (CUDA alias)
  decode_<device>_weight_dtype_direct.pt
  decode_<device>_production.pt
  probe_report.json       # vae_isolation + rollout_* metrics (XPU run)
  rollout_trace/          # optional (--save-rollout-trace on CUDA)
    manifest.json
    conditions.pt
    initial_noise.pt
    output_latents.pt
    chunk##_committed_latents.pt
    chunk##_kv_cache.pt
    chunk##_crossattn_cache.pt
    steps/chunk##_step##_*.pt
```
