# Wan2.2 Stage2 XPU inference optimization plan

**Status:** WS1–WS5, WS9, WS12, and WS16 done and verified; WS6, WS7, WS8, and WS17 rejected on
measurement (WS6's overlap machinery reverted; WS17's caching kept for correctness, not speed);
**WS13 is the only remaining lever with a projected gain**
**Last updated:** 2026-09-21 (B70)
**Device:** Intel B70, 30.3 GiB VRAM, `torch 2.12.1+xpu`
**Output anchor:** `md5 380d9cf74a9018d76408a3e615558fab`,
`sha256 2da7fa11bc10faf9d8309f84e7f5365708d05f190baaf71e6d2b4b3e27023a5c`
(camera route, 42 latent / 160 published frames) — **re-anchored by WS9** on 2026-09-21 and
reproduced by two fresh processes. The prior anchor `md5 92beab461308361a3110364da8cbf97f` was set
by WS1 and held through WS6; it remains the reference for every numerics-neutral change made
before WS9.
**Scope:** the Stage2 **inference** path only (`infer` / validation generation). Training, FSDP,
and multi-GPU paths are out of scope.
**Source guidance:** [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

> **The central finding, added 2026-09-21.** The device is **~99 % busy** during generation:
> measured GPU union-busy time is 1.693 s per diffusion chunk plus 1.695 s per VAE tile against a
> 3.44 s measured chunk wall time. There was never any scheduling overhead to recover, which is
> why WS3–WS6 produced no throughput gain. **53 % of every diffusion forward is positional and
> camera encoding recomputed redundantly** over the whole KV window — see
> [Per-forward budget](#per-forward-budget-measured). That is where the remaining time is, and
> no amount of streaming, pipelining, sync removal, or graph capture reaches it.

Workstreams are numbered **WS1…WS17 in the order they were identified**, and for WS1–WS11 that
was also the execution order. **WS12–WS17 were added after kernel attribution and take priority
over the unfinished earlier numbers.** WS12 and WS16 have since landed, and WS7 and WS17 were
rejected, so the recommended order from here is **WS13 → WS14 → WS15 → WS10**, with WS11 still
blocked on a quality decision rather than on engineering.

> **Number convention.** Every figure in this document is labeled **measured** (observed on the
> B70 in a run that can be pointed to), **projected** (computed from measured inputs under a
> stated assumption), or **estimated** (from experience or specs; nothing was run). Projected
> figures always carry their assumption and are never evidence that the assumption holds. WS6 is
> the cautionary case: a projected 6.3 fps derived from two correct measurements was treated as
> an outcome before the assumption behind it was tested, and it turned out to be false.

## Workstreams at a glance

| WS | Change | Projected gain | Measured gain | Complexity | Gate | Status |
| --- | --- | --- | --- | --- | --- | --- |
| WS1 | Production route → camera-length, single EMA pass | ~2× wall clock *(assumes the discarded pass dominates)* | **~300 s → 187 s (1.6×)** — not isolated, horizon also grew 39 → 42 latents | config | re-anchor | **Done and verified** |
| WS2 | Measurement harness | none (enables the rest) | — | small | — | **Done and verified** |
| WS3 | Remove host/device synchronizations | launch overhead across every forward *(estimated)* | **not isolated** — digest held, no separate timing captured | small | exact | **Done and verified** |
| WS4 | Configurable KV cache: mirrored circular default, clone fallback | removes rolling full-cache clones | **chunk 3.5881 s clone → 3.3890 s circular (−199 ms, 5.5 %)**, A/B measured 2026-09-21; rollout reserved +0.84 GiB | medium | exact | **Done and verified — keep** |
| WS5 | Memory: streaming decode, preallocation, reuse | ~0.3–1 GiB from decode *(estimated)*, horizon-independent peak | **reserved 13.93 → 11.69 GiB between rollout and decode**; peak now horizon-independent | medium | exact | **Done and verified** |
| WS6 | Pipeline diffusion against VAE decode | 50.8 s → ~28 s *(projected, **assumes the XPU overlaps two compute streams**)* | **48.49 s — no gain**; two-stream microbenchmark measures **1.001×** overlap | high | exact | **Rejected and reverted** — device does not overlap compute; per-chunk tiling and uint8 D2H kept |
| WS7 | `torch.compile` / inductor autotune | largest potential *(estimated)* | **3.3609 s eager → 3.3602 s compiled — no gain**, with 0 graph breaks and 0 eager fallback; costs 79 s of warmup and breaks the exact gate | high | PSNR | **Rejected** — see WS7 result |
| WS8 | XPU graphs | launch overhead only *(estimated)* | — launch overhead is ~1 % of the chunk | high | PSNR | **Rejected** — no overhead to recover |
| WS9 | Text encoder → host bf16 | −10.6 GiB host resident, −6.6 GiB load spike, encode 6.8 s → 5.1 s *(measured in isolation)* | **`conditions` 58.69 s → 27.56 s (−31.1 s)**; re-anchored, PSNR 28.27 dB mean vs old anchor (trajectory drift, visually clean) | small | PSNR | **Done and verified** |
| WS10 | fp32 matmul precision knobs | small *(estimated)* | — | small | PSNR | Not started |
| WS11 | Chunk size 1 latent frame | latency 3.24 s → 1.56 s, −40 % throughput *(both measured in isolation)* | — not adopted | high | quality experiment | **In progress** — plumbing only |
| WS12 | Hoist PRoPE projection-matrix construction out of the per-layer loop | 16.5 → 0.55 ms/forward ⇒ −80 ms/chunk *(projected from measured per-layer cost; assumes the saving is exposed, and the queue is 99 % busy so it is)* | **chunk 3.4447 → 3.3890 s (−55.7 ms, 1.6 %)**; bit-exact | small | exact | **Done and verified** |
| WS13 | Encoded KV ring: stop re-encoding the window every forward | 594 → 198 ms/chunk ⇒ chunk 3.44 → ~3.04 s *(projected from measured per-call costs; assumes a persistent encoded ring, not a `cat`)* | — | medium | exact | **Not started — largest remaining win** |
| WS14 | VAE decode elementwise in bf16 | 745 → ~250 ms/tile ⇒ tile 1.695 → ~1.14 s *(projected; assumes those kernels are bandwidth-bound, measured 548 vs 1814 GB/s)* | — | medium | PSNR | Not started |
| WS15 | bf16 RoPE rotation instead of complex128 | ~26 → ~11 ms/forward *after WS13* *(projected; **a naive fp32 rewrite measures 0.47×, i.e. slower** — only a minimal-traffic bf16 form wins)* | — | medium | PSNR | Not started — after WS13 |
| WS16 | Remove the surviving `.item()` syncs in `echorope_apply` | none *(estimated — the queue is 99 % busy)* | **chunk 3.3890 → 3.3609 s (−28.1 ms, 0.83 %)**; bit-exact | small | exact | **Done and verified** |
| WS17 | Cache scheduler grids and VAE `_scale` tensors instead of re-uploading them every call | ~16 % of chunk *(estimated from raw `Memcpy M2D` attribution — later shown to be the wrong reading, see WS17 result)* | **~4 ms/chunk, within run-to-run noise (3.7–7.7 ms stdev)** | small | exact | **Rejected on speed, kept for a correctness fix it uncovered** |

WS1 is first because it is the largest single win, carries no new code risk, and is independent of
everything else — and because doing it first means the harness and every later gate measure the
route we actually intend to ship. WS9 and WS10 sit late because both change numerics, so each
re-anchors the exact-match digest.

**Read the two gain columns together.** Three workstreams (WS3, WS4, WS5) were accepted on a
correctness and memory basis without an isolated timing measurement, and their measured cells say
so rather than borrowing the projection. WS6 is the case where the two columns disagree
decisively.

**What the completed rows add up to.** WS1 and WS9 moved the wall clock (route change and text
encoder, ~113 s + ~31 s). WS4 moved throughput slightly (1.91 → 1.75 s/chunk). WS3, WS5, WS6, and
WS8 moved throughput not at all, because all four targeted scheduling, host syncs, memory
traffic, or launch overhead on a device that is already ~99 % busy. **Generation throughput has
never improved beyond WS4** until WS12, which is the first item aimed at the work the device is
actually doing and the first to move the chunk since. Stacked, WS12–WS15 project to
**chunk 3.44 → ~2.33 s, i.e. 3.30 → ~4.9 published fps (1.47×)**; WS12 delivered 70 % of its
share, so discount the rest accordingly.

### Blockers and ordering constraints

Nothing is blocked on anything external except where noted. The real constraints:

| WS | Constraint | Kind |
| --- | --- | --- |
| WS1 | none — config, weights, and contract all permit it; B70 smoke baseline is recorded | — |
| WS2 | none — `ProfilerActivity.XPU`, `max_memory_allocated`, `max_memory_reserved`, `Stream`, `Event` all verified present on `2.12.1+xpu` | — |
| WS3 | complete | — |
| WS4 | complete; circular mode is restricted to camera inference with `sink_size=0` | — |
| WS5 | none, now that the weight offload is dropped (it was the only item needing WS9's host RAM) | — |
| WS6 | needs WS5 (two tiles resident at 9.2 GiB working set each); WS3 is complete | ordering |
| WS7 | **rejected** — compiles cleanly (0 breaks, 0 fallbacks) and measures 0.02 % against eager | — |
| WS8 | **rejected** — the device is 99 % busy, so there is no launch overhead to recover | — |
| WS9 | complete and verified | — |
| WS10 | none; one-line change, gated on WS2 proving it buys anything | — |
| WS11 | strongest form of the quality gate needs remote CUDA capacity | external |
| WS12 | none — pure redundancy removal, no new device behaviour assumed | — |
| WS13 | **must drop the raw ring's mirror in the same change** or it does not fit in 0.399 GiB of headroom | memory |
| WS14 | none, independent of WS12/WS13 | — |
| WS15 | needs WS13, else the same code gets written twice | ordering |

**The numbered order is a valid execution order.** The one prior inversion — WS5's weight offload
needing WS9's host RAM — is gone with the offload itself.

WS9 and WS10 stay late for a **gate** reason rather than a memory reason: both change numerics, so
each one re-anchors the exact-match digest. Keeping them after WS3–WS6 means all the
numerics-neutral work is validated against a single stable anchor first.

**Gate data availability.** `outputs/wan22-cuda-decode-reference/` holds a CUDA EMA reference for
one sample at the production shape (`[1, 39, 48, 30, 54]`, seed 1602803860) plus full decoded
pixels, so PSNR gates that compare against CUDA can run locally. The rollout trace only covers
chunks 00–02, so per-chunk CUDA bisection beyond chunk 2 needs a new remote run. WS9 is the happy
case: CUDA already runs the encoder in bf16 on device, so the existing reference *is* the
bf16-encoder reference and no new CUDA run is needed. WS11 is the exception — no CUDA reference
exists at block size 1, and generating one needs cluster access. A local block-3-vs-block-1
comparison is a reasonable proxy, since block 3 is already CUDA-validated.

## Where these changes go

Keep the model and the rollout in-tree and extend the seams that already exist. The codebase is
mixed training/inference, but it is not undifferentiated, and the three relevant layers each want
a different treatment.

**Inference-only construction — `CudaWanStage2GenerationAdapter` (`stage2.py:2444-2495`).** A
~110-line class only standalone inference ever builds, and it already makes device-specific
decisions (the XPU→CPU text encoder fallback lives there). WS7, WS8, WS9, and WS10 all belong
here. No new file is needed.

**Shared model code — `modeling/causal_model.py`.** WS3 and WS4 touch attention and KV-cache code
that training's `stage2_camera_rollout` also calls. Do not fork it: forking `CausalWanModel` drags
in checkpoint loading, fused-PRoPE metadata, and the sequence-parallel paths, and the edit is the
same either way. The seam to use instead is the existing `cache_update_policy` enum, already
threaded `stage2.py` → `components.py` → `causal_model.py` with values `"commit_detached"` and
`"none"`; WS4 adds a third value.

**The validation-identical core — `_stage2_generated_sample` / `_stage2_self_forcing_latents`.**
Called both by training-time validation (`stage2.py:1786`) and by standalone inference
(`stage2.py:2548`). Every gate in this plan is "byte-identical MP4 against the recorded
baseline," which only means something while these are the same code. Note the rollout is *already*
split — training uses `stage2_camera_rollout` (`stage2.py:413`), inference uses
`_stage2_self_forcing_latents` (`stage2.py:632`) — so inference-specific rollout work needs no new
fork.

**A thin deployment entrypoint is justified, but for a different reason.** Not to make the
workstreams below easier; they get easier from the seams above. It is justified because the
interactive long-rollout product does not exist yet — there is no viewer in the repo — and because
`inference.py` is a 3000-line batch-evaluation harness (dataset index rows, manifests,
publication, `COMPLETE.json`) that an interactive loop does not want. Such an entrypoint would
call the provider directly, run the single WS1 pass, and stream frames. That is new code, not a
fork.

## Environment and baseline

| Item | Value |
| --- | --- |
| Device | Intel B70 `[0xe223]`, 30.3 GiB VRAM, 256 EUs, driver `1.14.37020+3` |
| Integrated GPU | **none** — one render node, `torch.xpu.device_count() == 1` |
| CPU | Core Ultra 5 245K (Arrow Lake-S), 14 threads, AVX2 + AVX-VNNI; **no AVX-512, no AMX** |
| Host RAM | 30 GiB |
| Torch | `2.12.1+xpu`, `triton-xpu 3.7.1`, inductor available |
| XPU graph API | `XPUGraph`, `torch.xpu.graph`, `graph_pool_handle`, `make_graphed_callables` present |
| fp32 matmul precision | `highest` (default); `mkldnn.allow_tf32` is `False` |
| Attention backend | SDPA (FlashAttention is CUDA-only) |

Baseline workload is `infer_stage2_sgf_81f.yaml`, one sample, `live` + `ema` passes. **WS1
replaces this route**, so these numbers are the pre-WS1 anchor; per-chunk costs carry over
unchanged (the camera config has identical model geometry) but the pass count and horizon do not.

| Quantity | Value |
| --- | --- |
| Rollout | 39 latent frames, `num_frame_per_block=3` → 13 chunks per pass |
| Forwards per pass | 13 × (4 denoise + 1 commit) = **65**; 130 per run |
| Tokens per forward | 3 × `frame_sequence_length` 405 = **1215** queries; KV window bounded by `local_attn_size=18` |
| Model | `dim` 3072, 30 layers, 24 heads × 128, bf16 |
| Decode | 39 latent frames → 153 pixel frames at 480×864 |
| Wall clock | ≈5 min, **dominated by weight loading and the CPU text encoder**, not by generation |
| Peak device memory | **26.0 GiB / 30.3 GiB** |
| Peak host RSS | **28.2 GiB / 30 GiB** |

Device memory by phase (`/proc/<pid>/fdinfo`, `drm-total-vram0`):

| Phase | Device memory |
| --- | --- |
| Diffusion weights resident (5B bf16) | 10.0 GiB |
| + VAE and conditioning | 12.0 GiB |
| Rollout steady state (KV cache filled) | 16.7 GiB |
| VAE decode | 26.0 GiB (decode working set alone 9.2 GiB) |

## Latency and throughput budget

Timed by wrapping `provider.diffusion` on the real rollout path (EMA role) and timing
`decode_streaming_chunks` per tile. Per-forward cost ramps while the attention window fills and
saturates from chunk 6, which is `local_attn_size=18` latent frames.

| Steady-state quantity | Value | Class |
| --- | --- | --- |
| Denoise forward | 0.378 s (chunk 0 is 0.517 s including warmup) | measured |
| Commit forward | 0.397 s | measured |
| Diffusion per chunk (4 denoise + 1 commit) | **1.91 s** | measured |
| VAE decode per 3-latent-frame tile (12 pixel frames) | **1.72 s** (first tile 2.58 s) | measured |
| Total per chunk, serial | 3.63 s per 12 pixel frames ⇒ **3.3 generated fps** | projected from the two rows above; later **confirmed measured** at 48.49 s / 14 chunks |
| Same, decode overlapped against the next chunk | 1.91 s ⇒ 6.3 generated fps | **projected — assumes the XPU overlaps two compute streams. WS6 measured this assumption to be false.** |

At 16 fps playback a chunk is 0.75 s of video, so generation is ≈4.8× slower than real time
serially (measured). The ≈2.5× pipelined figure was projected under the same overlap assumption
and was **not** achieved — see WS6.

**Interactive latency** (camera pose in → corresponding pixels out) is the number that matters for
a viewer. The chunk's W2C matrices must be known before its first denoise step, so the critical
path is 4 denoise forwards + one decode tile = 1.51 s + 1.72 s ≈ **3.2 s** (projected from
measured per-forward and per-tile times; not separately measured end to end). The commit forward
is off that path (it only has to finish before the next chunk) but still costs throughput. Camera
input is additionally quantized to one chunk: a pose applies to all 3 latent frames, so direction
can only change every 0.75 s of generated video.

A 39-frame pass is ~23 s of rollout plus ~22 s of decode, so the ~5 min run is mostly the 38 GiB
checkpoint load, the second pass's role reload, and the CPU text encoder — which is why WS1 and
WS9 move the wall clock more than any kernel-level work here.

## Per-forward budget (measured)

Added 2026-09-21 from the chunks 4–8 XPU profile plus isolated timings of the in-tree helpers at
production shapes (`Lq` 1215, `Lk` 7290, 24 heads × 128, 30 layers). **This section is the
attribution that WS2 deferred as a "manual follow-up", and its absence is why WS3–WS8 were
mis-prioritized.** Build it before ranking any further work.

Splitting the trace's kernel time by GPU queue reconciles with the wall clock exactly:

| Quantity | Value | Class |
| --- | --- | --- |
| Diffusion queue, union-busy per chunk | **1.693 s** | measured |
| VAE queue, union-busy per tile | **1.695 s** | measured |
| Sum | 3.388 s | measured |
| Chunk wall time, unprofiled steady state | **3.44 s** | measured |
| **Device occupancy** | **~99 %** | measured |

So the sum of the two queues *is* the wall clock. The 16 % GPU idle visible inside the trace is
profiler overhead (profiled chunks cost 4.04 s against 3.44 s unprofiled), not a host-side stall.

Per diffusion forward, against 338.6 ms of measured GPU-busy time:

| Component | Time | Share | Kernel launches |
| --- | --- | --- | --- |
| Model GEMMs (q/k/v/o, cross q/o, FFN) | 90.7 ms | 27 % | ~240 |
| **RoPE, in complex128** | **79.3 ms** | **23 %** | 60 |
| **PRoPE feature transform over the whole KV window** | **56.1 ms** | **17 %** | ~300 |
| SDPA (self + cross) | 43.5 ms | 13 % | 60 |
| **PRoPE matrix prep, rebuilt identically in all 30 layers** | **16.5 ms** | **5 %** | ~1,700 |
| norms, modulation, time embedding, residuals, cache writes, misc | ~52 ms | 15 % | ~3,700 |
| **total** | **338.6 ms** | | **~6,150** |

The three bold rows are **152 ms per forward, more than the GEMMs and SDPA combined**. Two
distinct kinds of waste are stacked:

**Per-layer redundancy.** `_prepare_apply_fns_all_dim` (`camera_prope.py:241`) rebuilds `P`,
`P_T`, and `P_inv` from the camera view matrices twice per layer, 30 layers deep — 60 times per
forward — from inputs that are constant for the whole chunk. Measured 0.551 ms per layer where
0.55 ms per *forward* would do. This is also where most of the ~1,700 tiny
`fill`/`zeros_like`/`reciprocal`/`scalar_tensor` launches come from. WS12 fixes it.

**Per-forward redundancy.** `causal_model.py:1177-1204` slices the full 7290-token K window out
of the circular cache every forward and re-applies RoPE to all of it (2.203 ms/layer) plus PRoPE
to all of K and V (0.881 ms each). But `local_end_index` advances only on the commit forward, so
across the 4 denoise forwards **and** the commit, `window_start`/`window_end` are identical and
the 6,075 history tokens are untouched — only the current chunk's 1,215 slots change. 83 % of the
window is re-encoded five times to produce the same bits. Measured 594 ms per chunk, of which
~396 ms is recomputation. WS13 fixes it.

The GEMMs are not a target: measured at 129–142 TFLOP/s against a 150 TFLOP/s ceiling
(`8192³` bf16 square), and fusing q/k/v into one `3072→9216` GEMM buys **1.01×**. Whatever
`torch.compile` is worth here is elementwise fusion, not matmul autotuning.

VAE decode, per 1,695 ms tile: convolution 837 ms (49 %), elementwise copy/cast 374 ms (22 %),
elementwise math 370 ms (22 %), fp32 SiLU 59 ms (3 %). WS14 targets the 745 ms of fp32
elementwise work; `streaming_decode_session` converts each tile to fp32 on device
(`components.py:310`).

### The real-time ceiling

Worth stating so it is not rediscovered. Summing only the irreducible work — model GEMMs
5 × 90.7 ms, SDPA 5 × 43.5 ms, and VAE convolution 837 ms — gives **1.51 s per chunk, or
7.6 published fps, at 100 % efficiency with zero overhead** (projected from measured kernel
times; assumes every other kernel is eliminated, which is not achievable).

**16 fps real-time playback is therefore out of reach on this device in this configuration**, and
WS12–WS15 at best reach ~4.9 fps. Closing the remaining gap needs *less work*, not
better-executed work: fewer denoise steps than NFE4, lower resolution, int8/fp8 on the XMX units,
or a cheaper decoder. Those are model and product decisions and are deliberately out of scope
here — see [Out of scope](#out-of-scope).

## Regression gate

Three consecutive 81f runs on 2.12.1 produced **byte-identical** `ema` MP4s
(`md5 2985664b344b975588f70020668ec66e`,
`sha256 c54da3c8dfa33f1bec843abb2196b9b2d1ef54476345e81aba38a21e3e3434cf`), so the XPU path is
deterministic run-to-run. Therefore:

1. **Exact-match gate** — the run must reproduce the anchor digest bit for bit. Applies to every
   numerics-neutral workstream.
2. **PSNR gate** — for workstreams that change numerics by design: compare against the
   digest-matched baseline with a per-change PSNR / max-abs-diff threshold, plus visual
   inspection of frame 0, where the softmax bug showed first.
3. Keep the `torch==2.12.1+xpu` pin. Do not fix a performance problem by moving to a build that
   fails `scripts/debug/xpu_softmax_bug_mre.py`, and re-run that reproducer after any change that
   could re-route into a different softmax kernel.

**Camera-route anchor (WS1, verified 2026-09-20).** The 81f digest above remains the historical
record of the 2.12.1 fix. Production now uses the camera-length route; two independent B70 runs
of the local smoke sample produced an identical 160-frame MP4:
`md5 92beab461308361a3110364da8cbf97f`,
`sha256 74908cfed9cc78c174a912c67afa81e6e788bd0d882bdddf24b69fd3c0e91502`.
The route resolved 42 latent frames (165 model frames, trimmed to 160 published frames) and the
release manifest selected EMA. Every exact-match gate from WS2 through WS6 used this digest.

**Superseded as the production anchor by WS9 (2026-09-21).** WS9 changes the prompt embeddings by
design, so exact-match gates from WS12 onward use
`md5 380d9cf74a9018d76408a3e615558fab` instead. The digest above remains the correct reference
for re-verifying any pre-WS9 numerics-neutral change in isolation.

## WS1 — production route: camera-length, single EMA pass

**Decided: switch production inference to the camera-length route.** Largest single win in this
document, no new code risk, independent of every other workstream, and it sets the baseline the
rest of the plan is measured against — so it goes first.

### Why the current route costs double

The 81f route runs `live` then `ema`, doubling the work for an output we discard, and that is
deliberate rather than a config accident. `resolve_generation_plan` refuses inference-only sampler
blocks so standalone inference cannot drift from the validated contract (`generation.py:62-92`),
and `sgf.py:187-206` then picks the allowed pass list by route: `["live"]` or `["live", "ema"]`
for training, `["model"]` for camera-length inference, `["live", "ema"]` for standalone
fixed-length. Role switching itself is `_load_role` reloading from the 38 GiB `model.pt` into the
same module, so the second pass costs **load time, not peak memory** — there is only ever one
resident copy.

### Why camera-length is the right route

`configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml` already ships exactly one
pass, `{name: model_self_forcing_nfe4, weights: model}`, and `_published_default_weight_role`
resolves `weights: model` from the release manifest, which for `SolarWM-5B-sgf-stage2-81f` reads
`default_weights: ema` (`weight_role: live+ema`). **So this route already runs a single EMA pass
with no contract edit and no sign-off**, and it is the route `scripts/run_wan22_stage2_xpu.sh`
already uses. It is also the route that can express an unbounded horizon, which the long-rollout
goal needs anyway, so it settles the pass count and the horizon together.

The alternative — relaxing `sgf.py` to allow `["ema"]` on the fixed route — would have preserved
the 81f digest as the regression anchor, but it edits a deliberate provenance contract and needs
sign-off. Not chosen.

### What changes, and what has to be re-established

The horizon is derived differently, so this is **not** a byte-identical change:

- Fixed route: `variable_rollout_by_source: true` with `min_rollout_latent_frames: 21` and
  `rollout_latent_frames: 60`, so the horizon is the source clip clamped to [21, 60] — 39 for the
  measured sample.
- Camera route: `_camera_length_rollout_latents` covers the **full** source clip
  (`floor((num_frames-1) × output_fps / source_fps) + 1` pixels → `1 + ceil((pixels-1)/4)`
  latents, rounded up to a whole block), with **no 60 cap**. `_pass_case` then overrides the
  pass's declared `rollout_latent_frames` with that resolved value, which is why the config's
  placeholder of one block is harmless.

So the win is "one pass instead of two, and one fewer 38 GiB role reload," but the horizon may
grow for long sources. Per-chunk cost is unchanged — the camera config carries identical
`num_frame_per_block`, `frame_sequence_length`, and `local_attn_size` — so the latency budget
above still applies per chunk.

**Completed 2026-09-20:** production and the launcher use the camera-length config; two B70
smoke runs completed in 190.7 s and 184.0 s, respectively, with the same 42-latent / 160-published
frame result and the digest recorded above. `AGENTS.local.md` now points to
`generation/model_self_forcing_nfe4/`. WS2 will collect the per-phase time and memory baseline.

## WS2 — measurement harness

Without per-phase numbers the later workstreams cannot be ranked or verified. Run it against the
WS1 route so the numbers describe production.

1. Per-phase wall-clock timers around weight load, `_conditions`, rollout (per chunk), VAE decode,
   and MP4 encode. Emit as a JSON event so runs are comparable.
2. Device memory per phase via `torch.xpu.max_memory_allocated()` / `max_memory_reserved()`, plus
   the external `drm-total-vram0` poller for true driver-side usage. The two disagree in a way
   that matters — see WS5.
3. Kernel attribution with `torch.profiler` (XPU activity) for one chunk, to see whether time sits
   in SDPA, the linear layers, the norms, or host-side launch overhead. This bounds what WS7 and
   WS8 can buy.

**Implemented and verified (2026-09-20).** Enable the opt-in recorder with
`runtime.stage2_inference_measurements: true`. It writes one JSONL record per phase to
`runs/<run-id>/reports/stage2-inference-measurements.rank-00000.jsonl`, synchronizing only in
measurement mode and reporting XPU peak allocation and reservation per phase. A B70 camera-route
smoke retained the WS1 MP4 digest and reported: 70.92 s model load, 5.46 s weight load, 49.62 s
conditions (including the CPU fp32 text encoder), 1.91 s steady-state rollout chunk, 24.87 s VAE
decode, and 24.95 GiB peak XPU reservation during VAE decode.

**Kernel attribution was deferred here as "a manual follow-up when required", and that deferral
is the root cause of the WS3–WS8 mis-prioritization.** It was finally done on 2026-09-21 and
immediately rejected WS6 and WS8 and reordered everything else — see
[Per-forward budget](#per-forward-budget-measured). **Treat per-kernel attribution as part of the
harness, not as an optional follow-up: it is what tells you whether the time is in the kernels or
in the scheduling, and every later ranking depends on the answer.**

**Final B70 camera smoke (2026-09-21, 42 latent / 160 published frames).** The retained
three-latent VAE pipeline takes **48.49 s** from rollout start through the final VAE drain:
**3.30 published fps**. This is effectively the serial result, not the 6.3 fps overlap model:
14 × (1.91 s diffusion + 1.72 s decode) predicts 50.82 s, while ideal overlap predicts roughly
26.7 s plus final drain. The decode stream records **24.35 s** across 14 tiles with 0.0001 s
buffer-reuse wait, proving correct asynchronous submission but not concurrent XPU compute. The
process-wide high-water mark is **23.67 GiB allocated** and **27.84 GiB reserved**. Its
identifiable persistent portion is approximately 10.9 GiB diffusion+VAE weights, 5.01 GiB
mirrored circular self-KV, 0.176 GiB cross-attention KV, 0.261 GiB 900-latent output/noise
buffers, and 0.001 GiB camera-cache metadata; the balance is transient diffusion workspace and
activations. Enable `runtime.stage2_xpu_profiler: true` to write a single
`reports/stage2-xpu-profile.slot-*.json` Chrome/Perfetto trace for steady-state chunks **4–8
inclusive**, with CPU/XPU events, memory, shapes, Python stacks, and concurrently executing VAE
work. Override the inclusive bounds with
`runtime.stage2_xpu_profiler_start_chunk` and `runtime.stage2_xpu_profiler_end_chunk`. This
captures complete attribution and kernel launches while avoiding the full rollout's
hundreds of thousands of events.

**Captured (2026-09-21):**
`outputs/wan22-ti2v-5b-stage2-xpu-profiler/runs/wan22-camera-xpu-profiler-20260921-071914/reports/stage2-xpu-profile.slot-000000.rank-00000.json`.
Open this JSON directly in `chrome://tracing` or Perfetto. It contains the complete chunks 4–8
CPU/XPU trace, including Python stacks, operator shapes and memory events, and XPU kernel/launch
events.

## WS3 — remove host/device synchronizations

Numerics-neutral: **exact-match gate** applies strictly.

The guide's "avoid unnecessary CPU-GPU synchronization" item is the largest structural issue in
this path. Each `.item()` flushes the XPU queue, and the KV-cache bookkeeping does it inside every
attention layer of every forward.

1. **KV-cache indices as Python ints.** `causal_model.py:1150-1151` and `:1572-1573` called
   `kv_cache["global_end_index"].item()` and `["local_end_index"].item()` to drive Python `if`
   branches. With 30 blocks and several reads per block that is O(100) syncs per forward, O(10k)
   per run. They are per-rank scalars in SP1 and already effectively host state; keep device
   tensors only where a collective needs them. This is also the precondition for WS4, WS6, WS7,
   and WS8. **Done** — both sites now read through the `_cache_index` host-int helper
   (`causal_model.py:92`); no `.item()` remains on this path.
2. **Precompute the schedule.** `float(step.item())` per denoise step, though `steps` is known
   before the loop.
3. **One finiteness check per rollout**, not per chunk — a full-tensor reduction plus a sync, once
   per chunk. Same for the decode-side reduction; note `decode_streaming_chunks`
   (`components.py:338`) also syncs once per tile, which matters for WS6.
4. **Pass `frame_seqlen` in** rather than deriving it from `grid_sizes` on device
   (`math.prod(grid_sizes[0][1:]).item()` recomputes a constant from a device tensor).

## WS4 — eliminate the per-forward KV-cache clone

Numerics-neutral: **exact-match gate**. Prerequisite: WS3.

Every block forward clones the entire KV cache: `temp_k = kv_cache["k"].detach().clone()` at
`causal_model.py:1236-1237` (rolling mode) and `:1294-1295` (direct insert, pre-WS4). The block attends
over the clone, records a deferred `cache_update_info`, and `_apply_cache_updates` writes back
into the real cache after all blocks have run.

Derived from the shapes above — this is arithmetic from config, **not yet measured**, so WS2
should confirm it before the work is scheduled:

| Quantity | Value |
| --- | --- |
| Cache tokens | `local_attn_size` 18 × `frame_sequence_length` 405 = 7290 |
| `k` per layer | 7290 × 3072 × 2 B = 42.7 MiB |
| `k` + `v` per layer | 85.4 MiB |
| **Cloned per forward** (30 layers) | **≈2.5 GiB** |
| Per 130-forward run | ≈325 GiB of clone allocation and traffic |

The clone-and-defer pattern exists for training: the gradient-carrying replay must not mutate the
cache the no-grad pass built, and the `is_recompute` branch replays chunks already cached.
Inference needs neither. `global_end_index` only advances on the commit forward, so `is_recompute`
is structurally always `False` in `_stage2_self_forcing_latents`. Worse, the clone is
unconditional while `cache_update_policy` only gates the *write-back*, so on the four denoise
forwards per chunk (policy `"none"`) the 2.5 GiB clone is built and then discarded.

Add an inference `cache_update_policy` value that writes `k`/`v` in place and commits the index
bookkeeping only on the commit forward. The enum is already threaded through `stage2.py` →
`components.py` → `causal_model.py`, so training's `"commit_detached"` default is untouched.

**Rolling mode makes this harder than it looks, and rolling mode is the common case.** In direct
insert (cache not yet full) the clone can simply be dropped: nothing moves, and each forward
rewrites the same slots. In rolling mode the eviction *shift* is applied to the clone
(`causal_model.py:1238-1248`) and re-applied to the real cache by `_apply_cache_updates`, which
only runs on the commit forward. So the shift must be **visible to attention on all four denoise
forwards but committed exactly once** — writing it in place per forward would shift four times and
destroy the history. The fix is to stop materializing a shifted copy and instead build the visible
window from the unshifted cache with offset indexing, which is a rework of the window construction
rather than a small deletion.

With `kv_cache_size` 7290 tokens and 1215 new tokens per chunk, rolling begins once
`local_end_index` exceeds 6075, so on a 13-chunk rollout chunks 0–4 are direct insert and chunks
5–12 roll. The harder half is ~62 % of forwards, and it only grows with the horizon, so landing
only the direct-insert case captures a minority of the win. Staging it that way is still
reasonable, but do not book the full saving until rolling mode is done.

**Implemented and verified (2026-09-20):** standalone inference now uses
`cache_update_policy="inference_direct"` for direct-insert denoise forwards. It writes speculative
KV slots in place without advancing the host indices; the detached commit overwrites those slots
and remains the sole index advance. Rolling forwards intentionally retain the clone-based shifted
window, preventing four evictions per chunk. Training and validation retain their original
clone-and-defer behavior. Focused tests pass and the B70 camera-route MP4 exactly matches the WS1
digest.

**Superseded by the circular implementation (2026-09-20):** camera-length standalone inference
now defaults to `inference.kv_cache_mode: circular`. It allocates mirrored two-capacity K/V and
fused-camera buffers, so the logical wrapped window is one contiguous physical slice for SDPA:
no full-cache clone or concatenation is needed in either direct-insert or rolling mode. Denoise
writes are speculative and commit is the only pointer advance. `inference.kv_cache_mode: clone`
retains the accepted clone path as a fallback. Circular mode is deliberately restricted to the
production no-sink (`sink_size=0`) route; training, nonzero-sink, and recompute paths remain on
the clone implementation.

The B70 circular run exactly matches the camera-route digest. Measured steady-state chunk time is
**1.75 s** versus **1.91 s** before circular mode; its peak reservation is **17.32 GiB** versus
16.49 GiB during rollout (+0.84 GiB). Decode peaks at 26.92 GiB reserved, still below the 30.3
GiB device limit.

## WS5 — memory: streaming decode, preallocation, reuse

Numerics-neutral: **exact-match gate**.

### Streaming decode is bit-exact (verified)

Measured on production-shaped latents (`[1, 39, 48, 30, 54]`, XPU bf16 VAE):

| Path | Time | Peak `memory_allocated` | vs direct decode |
| --- | --- | --- | --- |
| `decode(use_cache=False)` | 23.20 s | 9.22 GiB | reference |
| `decode_streaming(chunk_latent_frames=39)` | 21.96 s | 9.22 GiB | **bit-exact** (max_abs 0) |
| `decode_streaming(chunk_latent_frames=12)` | 22.07 s | 8.96 GiB | **bit-exact** (max_abs 0) |
| `decode_streaming(chunk_latent_frames=6)` | 22.03 s | 8.90 GiB | **bit-exact** (max_abs 0) |

Cached decode over temporal tiles sharing one continuous cache reproduces single-shot decode
exactly, at every chunk size tested, and is marginally faster. The memory win at a 39-frame
horizon is small (~0.3 GiB): the decode working set is dominated by per-tile activations at full
480×864 resolution, not by temporal extent. Spatial tiling could cut it but would introduce seams
and forfeit the exact-match gate.

Adopt it anyway because the fp32 output accumulates on the **host** instead of the device
(≈0.8 GiB at 39 frames) and peak memory becomes **independent of the rollout horizon** — which
matters more now that WS1 removes the 60-latent cap. For the long-rollout goal this is the
enabling piece: tiles are yielded one at a time and the KV window is already bounded, so the only
remaining horizon-dependent term is the `output` latent buffer allocated for the whole rollout,
which an unbounded run would need to make a rolling window.

### Cache release does not lower the driver high-water mark

The 26.0 GiB peak looked like fragmentation, so it was tested directly: `torch.xpu.empty_cache()`
between rollout and decode lowers torch's reserved pool by 2.2 GiB (13.93 → 11.69 GiB) but leaves
`drm-total-vram0` unchanged (14.44 → 14.55 GiB). The driver does not return pages. **Expect no
peak relief from cache release**, and treat `torch.xpu.memory_reserved()` as a poor proxy for what
the card holds.

### Rejected: offloading the diffusion weights during decode

**Decided against.** Decode does not need the ~10 GiB of diffusion weights, so evicting them
during decode was the largest available peak-memory lever. It is incompatible with the streaming,
interactive target. Measured 10 GiB transfer cost on this host:

| Host buffer | Offload (D2H) | Reload (H2D) | Round trip |
| --- | --- | --- | --- |
| pageable | 1.96 s (5.1 GiB/s) | 1.73 s (5.8 GiB/s) | 3.7 s |
| **pinned** | 0.38 s (26.2 GiB/s) | **0.21 s (48.3 GiB/s)** | **0.6 s** |

0.6 s is acceptable only against a single whole-rollout decode (~22 s, ~3 % overhead). Streaming
decode interleaves per chunk, so the round trip is paid every cycle, and the cycle is short: the
real-time target is 4 latent frames per second, i.e. a decode every 0.25 s of generated video per
latent frame, so at a 3-frame block the whole per-chunk budget is 0.75 s and at WS11's 1-frame
block it is 0.25 s. A 0.6 s weight round trip is most of the first budget and **larger than the
second one entirely** — it would dominate the pipeline it is supposed to help. Batching chunks into
large decode windows would amortize it, but that directly opposes the low-latency goal.

Dropping this also removes the plan's only host-RAM dependency: the 10 GiB pinned staging buffer
was the sole reason WS5 needed WS9 to land first.

### The remaining levers

1. **Reuse the noise buffers.** Implemented for standalone camera inference: each prior draw now
   fills an exact-shaped view of a persistent buffer with `torch.randn(..., out=view)`, preserving
   generator call ordering and shapes. Required for WS8, where addresses must be stable.
2. **Preallocate maximum-shape buffers up front.** Implemented for standalone camera inference:
   `inference.max_rollout_latent_frames` is a positive, chunk-aligned capacity. Output, initial
   noise, and denoise-noise storage are allocated once after model load and each source slices them.
   Sources over the configured capacity fail before denoising.
3. **Drop the avoidable copies** — the pre-decode `.contiguous()` and the per-chunk noise slice
   clone.

With the offload gone, WS5's peak-memory contribution is only the streaming-decode win (~0.3 GiB
plus horizon independence). The 26.0 GiB peak therefore stands for now, which matters for WS7 and
WS8: autotune and graph pools have to fit in ~4 GiB of headroom, not in headroom this workstream
was going to create.

## WS6 — pipeline diffusion against VAE decode

**Exact-match gate** — this reorders execution without changing arithmetic. Prerequisites: WS3 (a
host sync anywhere in the overlapped region collapses it) and WS5 (same double-buffered
allocations).

Today decode runs strictly after the rollout, so the device alternates between phases.
`torch.xpu.Stream`, `torch.xpu.Event`, the `torch.xpu.stream` context manager, and
`non_blocking=True` copies into pinned host memory are all available in `2.12.1+xpu`, so while
chunk *n+1* denoises, chunk *n* can decode and copy back.

| Schedule | Per 12 pixel frames | Generated fps | vs 16 fps real time | Class |
| --- | --- | --- | --- | --- |
| serial | 1.91 s diffusion + 1.72 s decode = 3.63 s | 3.3 | 4.8× slower | projected, later **confirmed measured** |
| pipelined | max(1.91, 1.72) = 1.91 s | 6.3 | 2.5× slower | **projected — assumption disproved, never achieved** |

The 6.3 fps figure is a theoretical ideal that assumes the XPU fully overlaps the two kernels.
That assumption was the whole workstream and it was never tested before implementation; WS6's
measurement below shows it does not hold on this device. Even had it held, it would **not**
improve interactive latency: a chunk's pixels still need that chunk's 4 denoise forwards followed
by its decode. Pipelining hides decode behind the *next* chunk's work.

1. Put decode on a second XPU stream, with an event recorded after the rollout writes its output
   slice and waited on by the decode stream, so a tile is read only once the diffusion writes are
   visible.
2. Double-buffer the decode input and output tiles; one scratch buffer would re-serialize the
   streams.
3. Move the D2H copy of finished tiles to pinned memory with `non_blocking=True`, and keep MP4
   encoding on a host worker thread so the codec never blocks the device.
4. Attribute peak memory carefully: two streams in flight means two tiles resident, which eats
   some of the headroom WS5 frees.

### WS6 result

- **Projection.** 14 chunks × 1.91 s diffusion + 1.72 s decode (both measured, steady state).
  Serial = 50.8 s. Full overlap = 14 × 1.91 = 26.7 s plus drain ⇒ target ~28 s.
  **Assumption: the XPU executes two compute-heavy streams concurrently.**
- **Pre-check.** *Not performed before implementation.* **Performed 2026-09-21, after the fact,
  and it takes seven seconds to run.** Two independent bf16 `4096³` GEMMs on separate
  `torch.xpu.Stream`s: one workload alone 37.56 ms, both serial 78.70 ms, both on two streams
  **78.64 ms — 1.001× (measured)**. A small `conv3d` against a large GEMM, which is much closer
  to the VAE-versus-diffusion mix, gives **0.995×**. **The B70 does not overlap compute streams
  at all.** The trace confirms it independently: per-queue union-busy times are 8.464 s and
  8.474 s, and the union across both queues is 16.939 s — their sum to four significant figures,
  i.e. literally zero concurrent execution.
- **Implementation.** Camera XPU inference defaults to
  `inference.stage2_xpu_vae_pipeline: true`. It retains a continuous VAE decode cache and submits
  **every completed three-latent chunk** to a second XPU stream after its detached-commit event;
  two pinned host buffers preserve output order for the existing CPU encoder. Unsupported routes
  and an explicit `false` use the serial decoder.
- **Measured (B70, 42-latent smoke).** Pipeline interval **48.49 s** against a **50.82 s** serial
  projection and a **~26.7 s** overlapped projection. Scheduling telemetry: all 14 tiles
  submitted asynchronously, **24.45 s** device decode-plus-copy inside the interval, **0.0003 s**
  total buffer-reuse wait, zero blocked submissions. Throughput **3.30 published fps (measured)**
  against **6.3 fps (projected)**.
- **Gate.** Passes. The 42-latent smoke exercises 14 ordered VAE tiles and reproduces the exact
  anchor digest for both the pipeline and the serial fallback.
- **Decision (revised 2026-09-21).** **Rejected.** The overlap assumption is false on this device
  and the microbenchmark above settles it for the whole platform, not just this workstream. The
  second stream, the `Event` pairs, and the two pinned double buffers exist only to attempt
  overlap and should be removed.
- **Reconsider only if** a future device or driver reports more than one compute engine, i.e. the
  two-stream microbenchmark measures materially above 1.0×. Re-run that microbenchmark first;
  do not re-derive the conclusion from scheduling telemetry.
- **Do not revert the whole class.** Two things implemented alongside WS6 are independent of the
  failed overlap claim and are worth keeping: **per-chunk incremental decode** (pixels are
  available per chunk instead of after the full rollout, which the interactive target requires
  and which keeps host pixel accumulation independent of the horizon) and **GPU-side uint8
  quantization before the host copy**. Both work unchanged on a single stream.

**Why the telemetry misled.** Asynchronous submission, near-zero buffer waits, and zero blocked
submissions are all true and all establish *scheduling* correctness only. None of them is
evidence of concurrent kernel execution. The measurement that settled it was arithmetic: 48.49 s
sits within 5 % of the serial projection and at 1.8× the overlapped one. **Proof of overlap is a
measured total near the overlapped projection, nothing else.**

The chunks 4–8 XPU profile (see WS2) is the evidence needed to determine whether event
dependencies or single-engine stream scheduling cause the serialization.

**One-latent decoder microtile experiment (2026-09-20): rejected.** A BF16 VAE-only decode falls
from **8.86 GiB** peak allocated at three latents to **3.79 GiB** at one, and the end-to-end
42-latent smoke is byte-exact, including both MP4s. This does not reduce the production peak:
fresh-process A/B runs with one process-wide peak counter (not reset between asynchronous chunks)
report **23.67 → 23.64 GiB allocated** and **27.84 → 27.84 GiB reserved**, because
diffusion/KV-cache allocations already dominate it.
It also adds 14.52 s of buffer-reuse waits and changes the pipeline interval from 48.70 to
48.81 s. Keep the three-latent submissions and two-buffer pipeline. Reconsider microtiles only
if diffusion weights/cache no longer coexist with decode or a lower-workspace XPU Conv3d algorithm
is available.

**GPU encoder quantization (2026-09-20):** the retained three-latent pipeline now quantizes VAE
pixels on XPU before its pinned-host copies. It keeps two uint8 representations because the
existing artifacts have distinct rules: `video.mp4` rounds `[-1, 1]` to RGB, while `compare.mp4`
truncates after its comparison normalization. The B70 smoke reproduces both prior MP4 digests,
while reducing each D2H tile from float32 to uint8. VAE convolution remains bf16-autocast, but its
public pixel tensor is intentionally fp32 before this final quantization.

## WS7 — `torch.compile`

**PSNR gate.** Prerequisites: WS3 and WS4 — every `.item()`-driven branch in the attention cache
path is a hard graph break, so compiling first buys almost nothing.

**Implemented staged interface (2026-09-20):** set
`inference.stage2_xpu_compile_blocks: true` only on the standalone Stage2
camera-length XPU configuration. It defaults to `false`. After the selected
checkpoint role has loaded and the diffusion module is already on XPU, the
runtime wraps each `CausalWanAttentionBlock` with
`torch.compile(block, dynamic=False)`. It intentionally does not compile the
diffusion wrapper, VAE, text encoder, sampler, or WS8 graphs.

This is an opt-in, no-fallback setting. It rejects non-XPU execution and a
process with `torch._dynamo.config.suppress_errors=true`; a compile error
during wrapping leaves the original block list intact, and lazy compile errors
on the first denoise forward propagate to the caller. Circular inference omits
the unused `current_start` and `cache_start` block arguments; its committed
cache end is authoritative. No dynamic shape or graph-capture work was added.

**B70 retry (2026-09-20): rejected at the time.** WS3's remaining
RoPE `.item()` was removed from the supplied `frame_seqlen` path, and circular
blocks no longer receive the unused varying global offset. Dynamo no longer
reports either as the recompile cause. It instead specializes the required
host-resident `kv_cache["local_end_index"]` values (0, 1215, 2430, …), again
hitting the eight-variant limit. Raising the limit was believed to compile one graph per rollout
chunk (300 for the configured 900-frame horizon). **Both of those conclusions are superseded by
the 2026-09-21 pre-check below.**

### WS7 re-diagnosis (2026-09-21) — the blocker is smaller than recorded

A full `stage2_xpu_compile_blocks=true` run with `TORCH_LOGS=recompiles,graph_breaks`, plus a
synthetic variant-count harness, produced the complete cause list. Two recorded claims were
wrong.

**Wrong claim 1: the variant count is unbounded.** It is not. `_circular_capacity` is 7290 and
the ring advances 1215 tokens per chunk, so `_circular_ring_start` cycles through exactly **6**
residues and `local_end_index` saturates at 7290. Measured over 14 chunks: **11 distinct
`(ring_start, local_end_index)` states, and it stays 11 for any horizon.** Not 300.

**Wrong claim 2: everything falls back.** Exactly **one** code object hits the limit —
`CausalWanSelfAttention.forward` (`causal_model.py:989`), on
`kv_cache['local_end_index'] == 3645`. The rest of the block compiles and runs. Measured effect:
**steady-state chunk 3.44 s → 3.32 s (3.5 %)**, with the 53 % of the forward that is RoPE and
PRoPE still running eager. The compiled output digest is
`md5 aa2718c9333f27f85e8f07f82d43b10a`, so it remains outside the exact gate.

The full measured cause list, by frequency over one 14-chunk run (42 recompiles):

| Cause | Count | Fixed by |
| --- | --- | --- |
| `tensor 'x' size mismatch at index 1` in `echorope_apply` / `block_relativistic_rope` | 45 | WS13 (fixed-shape window) |
| `tensor 'k_window' size mismatch at index 1` | 18 | WS13 |
| `cache_update_policy == 'inference_direct'` | 14 | deliberate 2 variants, or hoist the cache write out |
| `kv_cache['local_end_index'] == {0, 1215, 2430, 3645}` | 26 | WS13 (slice outside the compiled region) |
| `start_frame == {0, 3, 6}` | 12 | WS13 |
| `KeyError on kv_cache['_fused_prope_camera_metadata']` | 2 | pass camera metadata as an argument, not on `kv_cache[0]` only |
| `crossattn_cache['is_init'] == False` | 1 | initialize the cross-attention cache before the rollout instead of lazily |
| **graph break** at `causal_model.py:199` (`grid_sizes.tolist()` in `echorope_apply`'s Python batch loop) | 33 | pass `(f, h, w)` as Python ints |

**`.item()` also survived WS3.** `echorope_apply` (`causal_model.py:242-243`) calls
`temporal_idx.min().item()` and `.max().item()` as a bounds diagnostic on every RoPE application:
2 per layer × 30 layers = **120 device syncs per forward**, 3,030 in the trace window, surfacing
as 3,000 D2H copies averaging 851 µs. The bounds are already validated from Python ints at
`causal_model.py:818-862` (`_rope_q_and_window_k`) before the call. Removing them is worth **~nothing in wall clock** —
the queue is 99 % busy — but it is a hard prerequisite for compile and it removes a whole guard
class. Do not book a throughput gain for it.

### WS7 result (2026-09-21) — rejected on measurement, after the blockers were removed

The re-diagnosis above was acted on and **the compile blockers are now genuinely gone**, so this
is a clean negative result rather than another blocked attempt.

Three changes got there:

1. **Raised Dynamo's variant ceiling.** `_STAGE2_XPU_DYNAMO_VARIANTS = 32` in
   `_compile_stage2_xpu_transformer_blocks`, against a default of 8. The real steady-state count
   is **27 variants per block**, not the 11 estimated above — that estimate counted
   `(ring_start, local_end_index)` states but missed the ramp's shape specializations. **Zero
   eager fallbacks** after the raise.
2. **Removed the last graph break.** `grid_sizes.tolist()` in `echorope_apply` was the *only*
   break site in the entire block (83 occurrences, one per rope call). `grid_sizes` is a CPU
   tensor built from Python shapes, so `CausalWanModel.forward` now reads it once into
   `grid_list` and threads Python int tuples down to the rope helpers. **Zero graph breaks**
   afterwards.
3. **Removed the surviving device syncs** (WS16 below), which were also `_local_scalar_dense`
   breaks under compile.

**Measured with zero breaks, zero fallbacks, and a warm `TORCHINDUCTOR_CACHE_DIR`:**

| | steady-state chunk | warmup |
| --- | --- | --- |
| eager | **3.3609 s** | — |
| compiled | **3.3602 s** | **+79.0 s** over the first 8 chunks |

**That is a 0.7 ms difference, i.e. 0.02 % — noise.** Break-even against the 79 s warmup would
take ~113,000 chunks. The compiled output is also outside the exact gate
(`md5 47878fe2f752e2763db41bdeafeb10dd` against the anchor
`380d9cf74a9018d76408a3e615558fab`), so adopting it would additionally cost a re-anchor.

**Why it produced nothing, and why this was predictable.** Compile pays in three currencies and
this pipeline has no balance in any of them:

- *Launch overhead*: the device is 99 % busy, so there is ~1 % to recover. Same reason WS8 was
  rejected.
- *Matmul selection*: the model GEMMs already run at 129–142 TFLOP/s against a measured
  150 TFLOP/s ceiling (86–95 %), so inductor has at most 5–14 % of 90.7 ms to find.
- *Elementwise fusion*: this is the one real target, and it is dominated by the **redundant**
  RoPE/PRoPE work. Fusing redundant work is strictly worse than deleting it, which is what
  WS12 did and WS13 will do.

**Interesting null result: inductor produced exactly eager-speed code on a 30-block
transformer with no breaks.** That is worth remembering as a property of this stack — it argues
against reaching for `torch.compile` again on the B70 without first showing the profile has
overhead or unfused elementwise work to recover.

**What was kept.** The variant-limit raise and the `grid_list` plumbing stay in tree. The limit
raise is inert unless `stage2_xpu_compile_blocks=true`, which remains **false** by default. The
`grid_list` change is a small eager win in its own right and is **bit-exact** (verified: eager
run after the change reproduced `380d9cf74a9018d76408a3e615558fab`).

**`max-autotune` was not run, and here is the arithmetic for skipping it.** Its entire
addressable surface is the GEMM gap: 90.7 ms of GEMM per forward at 86–95 % of a *measured*
ceiling gives at most 12.7 ms/forward ≈ 63 ms/chunk (1.9 %), and only if autotune found kernels
at 100 % of a ceiling that was itself measured from the best observed case. Against 27 variants
× 30 blocks of autotuning per process, that is not a favourable trade. **Reconsider only
alongside WS13**, when the kernel mix has changed enough that the profile must be re-taken
anyway.

### WS7 plan, revised (superseded by the result above)

The prerequisite is architectural, as previously recorded, and it is **WS13**: compile a
stateless core that receives an already-sliced, fixed-shape encoded window and contains no Python
integers derived from cache state. Then:

1. **Leave the 5 ramp chunks eager.** In steady state the window is already exactly 7290 tokens,
   so fixed shapes need no mask. On a 300-chunk production horizon the ramp is 1.7 % of forwards.
2. **Do not use masks to force fixed shapes.** Measured: an additive mask costs **1.45×** on SDPA
   at `Lk` 7290 (1.357 → 1.962 ms) and 2.44× at `Lk` 1215. Over 30 layers that is +18 ms per
   forward — more than the ramp chunks it would save.
3. Expect the gain in elementwise fusion, not matmuls (the GEMMs are already at 86–95 % of the
   measured ceiling). Budget `max-autotune` accordingly, i.e. do not expect much.

**VAE-only probe (2026-09-20): rejected.** Compiling `WanVAE_.cached_decode` on B70 hit Dynamo's
eight-variant limit inside causal decoder residual blocks as their temporal dimension changed
(1, 2, and 4). First execution took 109.6 s and differed from eager output by max-abs 1.332, so
there is no safe VAE compile setting.

Shapes are favorable: 1215 query tokens per forward, a bounded KV window, and identically-shaped
chunks. The only varying input is the integer `current_start`.

1. Start with `torch.compile(module, dynamic=False)` on the transformer **block**, not the whole
   model, so a graph break costs one block rather than the pass.
2. Mark `current_start` dynamic (or bucket it) to avoid one recompile per chunk.
3. Progress `default` → `mode="reduce-overhead"` → `mode="max-autotune"`, measuring each. Autotune
   and graph capture increase memory use, which is why WS5 comes first.
4. Budget the per-process warmup: a 60–90 s compile must pay for itself against the run length.
   Point `TORCHINDUCTOR_CACHE_DIR` at persistent storage so repeat runs skip codegen.
5. Two compiled variants will appear, one per `cache_update_policy`. Confirm both compile rather
   than silently falling back.
6. Keep `SOLARWM_COMPILE_FLEX` off — FlexAttention compilation is a Stage1 training concern.

Risk: inductor fusion changes numerics, and XPU inductor coverage on this module is unverified.

## WS16 — remove the surviving `echorope_apply` device syncs

**Done and verified 2026-09-21. Exact-match gate passed.**

WS3 removed the rollout-loop syncs but missed these. `echorope_apply` ran a bounds diagnostic —
`temporal_idx.min().item()` and `.max().item()` — on *every* rope application: 2 per call,
2 calls per layer, 30 layers = **120 device syncs per forward**.

The fix is that the check was redundant on the path that paid for it. When `frame_indices is
None` the function has already proved `0 <= start_frame` and `start_frame + f <=
freqs_t.shape[0]` from Python ints, and then builds `temporal_idx` as exactly
`arange(start_frame, start_frame + f)` — so re-deriving the same bound from device scalars
could not fail. The diagnostic now runs only in the `frame_indices is not None` branch, where
a caller supplies arbitrary indices and the check is real. Stage1 keeps it; KV-cache inference
never reaches it.

- **Measured: chunk 3.3890 s → 3.3609 s, −28.1 ms (0.83 %).** Ranges do not overlap
  (3.3762–3.3992 against 3.3566–3.3660).
- **Correcting the WS7 re-diagnosis, which predicted "worth ~nothing in wall clock".** That
  reasoning — the queue is 99 % busy, so removing host-side waits cannot help — was wrong in a
  specific and reusable way: **a sync does not just make the host wait, it drains the queue.**
  Each `.item()` forces the device to retire everything in flight before the next launch, so
  120 of them per forward punch 120 bubbles into an otherwise saturated pipeline. The 99 %-busy
  figure is measured *across* the chunk and does not exclude many small gaps.
- **Do not generalize the 99 %-busy argument to syncs.** It correctly rejects launch-overhead
  work (WS8) and, as it turned out, compile (WS7). It does not reject sync removal.

## WS17 — cache scheduler grids and VAE `_scale` tensors (rejected on speed)

**Exact-match gate.** No prerequisites. Motivated by a raw profiler read that turned out to be
wrong in an instructive way — read this section together with the correction it produced, not in
isolation.

`WanDiffusion.flow_to_x0` (`components.py:632-633`) re-uploads the scheduler's `timesteps` and
`sigmas` — 1000-element fp64 host tensors, because `FlowMatchScheduler()` defaults to
`num_train_timesteps=1000` — on every call. `Wan5BVAE._scale` (`components.py:203-206`)
re-uploads the 48-channel `_mean`/`_std` grids on every encode call and every decode tile. A
chunks 4–5 profile (`ZET_ENABLE_METRICS=1`, `with_flops=True`) attributed **1369.0 ms** of
`Memcpy M2D` to exactly these two sites over 42 copies — 36.7 % of total device time in the
window, and the single largest line item, ahead of GEMMs (1292.4 ms) and convolution
(1675.0 ms combined but split across many kernels).

### WS17 result — the fix landed, cost nothing, and found a real bug

**Both grids were cached** — a `grid_on(name, device, dtype)` accessor on `FlowMatchScheduler`
and a `(device, dtype)`-keyed cache in `_scale` — cutting `Memcpy M2D` count from 42 to 10 per
profiled window.

- **Speed: no gain.** Two full runs against the pre-WS17 baseline (3.3613 s, stdev 3.7 ms):
  3.3562 s and 3.3584 s (stdev 6.2–7.7 ms). **~4 ms, or 0.12 % — inside run-to-run noise.** Peak
  memory was identical to the byte (26.545 GiB reserved).
- **First cut broke exactness.** The naive `(device, dtype)`-keyed `_scale` cache changed output
  pixels (hash `6510782a…` → `a573e177…`, reproduced twice, so deterministic rather than flaky).
  Cause: `_scale` is called both inside `torch.autocast` (the decode-tile path,
  `components.py:301-310`) and outside it (encode). `1.0 / std.to(bf16)` is promoted to fp32
  under autocast but stays bf16 outside it, so the same `(device, dtype)` key silently holds two
  different results depending on which call site populated it first. **Fixed** by adding
  `torch.is_autocast_enabled()` / `torch.get_autocast_dtype()` to the cache key, restoring the
  exact-match digest (`6510782a…`) with the caching kept.
- **Re-profiled after the fix: `Memcpy M2D` count fell 42 → 10 and its attributed time fell
  1369.0 → 1074.2 ms, while wall clock did not move.** That gap is the real finding.

### Why the attributed time was never recoverable

The 10 surviving copies transfer **10 KiB in total** (5×`[1000]` fp64 + a few `[48]` bf16
grids not yet warm) while being attributed **1074.2 ms** — the trace's own bandwidth field
reports this directly: several copies show ~1e-5 GB/s, physically impossible for a real
transfer, next to two copies of the same size showing a plausible ~0.3 GB/s and costing 0.00 ms.
At a realistic 8 GB/s, 10 KiB takes 1.3 µs; the attributed time is roughly 840,000× that.

**`Memcpy M2D` duration in this profiler is queue-completion latency
(`completion_timestamp − append_timestamp`) on an in-order device queue, not transfer cost.**
When a copy is enqueued behind a deep compute backlog, its completion timestamp — and therefore
its reported `dur` — includes all of that backlog, regardless of how large the copy actually is.
That time is real on the device clock, but it was never a cost the copy caused, and nothing
about removing 32 of the 42 copies could touch it, because removing a copy doesn't remove the
backlog it was waiting behind.

**Two follow-up checks confirmed this is not a submission-mode artifact.** `ZE_DEBUG=1` was
tried first, expecting it to force synchronous kernel launches; it instead turned out to be the
Level Zero loader's verbose API-call tracer (`[ze_loader] [trace] zeCommandListAppendLaunchKernel(...)`),
which had emitted 8,145 log lines / 946 KB in 60 s of weight loading alone and was killed before
it could run a profiled chunk. `UR_L0_USE_IMMEDIATE_COMMANDLISTS` was tried next (`=1`, then `=0`
with `EVENTS_PER_BATCH=1`): total device time and `Memcpy M2D` attribution stayed within noise
across all three configurations (15.217–15.299 s total; 1042.6–1074.2 ms M2D over the same 10
events), so command-list batching mode is not the mechanism either. Getting genuine per-kernel
attribution on this stack would require inserting `torch.xpu.synchronize()` around each profiled
op boundary in code — not an environment variable — and was not pursued, since wall-clock A/B
already answers the questions that matter here.

- **Decision: rejected on speed, kept for correctness and profile clarity.** The caching removes
  32 blocking host-to-device transfers per profiled window and closes a real (if latent) autocast
  precision hazard in `_scale`. It is bit-exact and lint-clean. It does not move the chunk.
- **Generalizable finding, add to how every future profile in this document is read:** treat any
  op whose reported bandwidth is physically implausible (orders of magnitude below realistic
  transfer rates) as a queue-drain wait marker, not as workload, and gate every change on
  wall-clock A/B rather than on attributed device time for that op. This refines rather than
  contradicts WS16: a host-side `.item()` sync drains the queue and *is* recoverable (WS16
  measured −28.1 ms), because the host was genuinely blocked; an async `Memcpy M2D` enqueued
  behind a backlog is not, because nothing was waiting on it.

## WS8 — XPU graphs

**Rejected 2026-09-21, before implementation, on the WS2 profile.**

The original plan was to capture one denoise step and one commit step as separate graphs and
replay them 4× and 1× per chunk, with `current_start` and the timestep fed through preallocated
device tensors updated in place. It was gated on "only hand-roll graphs if the WS2 profile still
shows launch overhead dominating." **The profile shows the opposite.**

Graphs address host-side launch overhead only. Measured: the device is **~99 % busy**
(1.693 s diffusion + 1.695 s VAE queue union-busy against a 3.44 s chunk), so there is roughly
**1 % of launch overhead to recover**, and the host has ~2.5× headroom on its launch budget
(6,150 launches per forward at 22 µs of `urEnqueueKernelLaunch` each = 135 ms of host time
against a 339 ms forward). A perfect graph capture cannot produce a measurable speedup.

This also disposes of the hypothesis that the ~6,150 kernel launches per forward are themselves
the problem. They are a *symptom* of the redundant per-layer work; WS12 and WS13 delete the work
and the launches together, which is strictly better than replaying them faster.

**Reconsider only if** a future profile shows device occupancy materially below 90 % after
WS12–WS15 land — i.e. once the kernels are few enough and fast enough that the host becomes the
constraint.

## WS9 — text encoder → host bf16

**PSNR gate.** **Placement is settled: the encoder stays on the host, in bf16.** GPU placement is
not pursued (see below), so this workstream is now just two host-side changes — the dtype and the
load path — with no device-memory question and no dependency on anything above it.

It stays at this position because it is PSNR-gated: it changes the embeddings, so it re-anchors the
exact-match digest. Running it after WS3–WS6 keeps all the numerics-neutral work on one anchor. It
could move earlier at the cost of an extra re-anchor, and nothing technical prevents that.

### bf16 on the host

The intent was bf16 on an integrated GPU, else CPU with AMX. Neither exists here: no iGPU is
exposed, and the Core Ultra 5 245K has no AMX and no AVX-512. So "CPU bf16" means oneDNN on
AVX2/VNNI. Measured on UMT5-XXL, 512-token prompt, 14 threads:

| Placement | Resident | Encode (cold / warm) |
| --- | --- | --- |
| CPU fp32 (production today) | 21.8 GiB host | 13.7 s / 6.8 s |
| **CPU bf16** | **11.2 GiB host** | **5.1 s / 5.1 s** |
| XPU bf16 | 10.58 GiB device | 1.4 s |

CPU bf16 wins on both axes against today's fp32 — ~10.6 GiB less host RAM and ~25 % faster — even
without AMX, and it needs no device memory. **Implemented (2026-09-21):** XPU Stage2 constructs
UMT5 directly as CPU bf16 before loading weights, avoiding fp32 parameter materialization and its
post-load conversion. This intentionally changes prompt embeddings; the next B70 smoke must
record its new output digest and visual/PSNR comparison before it is accepted as the production
anchor.

### Then fix the load path, which is the real host-RAM risk

| Stage | Host RSS |
| --- | --- |
| after `WanTextEncoder(...)` construction | 17.9 GiB (**parameters materialize as fp32**) |
| after `.to(cpu, bf16)` | 11.2 GiB |
| after `.to(cpu, fp32)` (production today) | 21.8 GiB |
| transient peak during load | **28.4 GiB of 30 GiB** |

The checkpoint on disk is already bf16, but construction upcasts to fp32, so the load spikes to
within ~1.6 GiB of the host limit regardless of the final dtype. Construct in bf16 directly
(meta-device init plus `load_state_dict(assign=True)`, or an explicit construction dtype). This is
the riskiest number in the baseline and it is independent of everything else here.

### GPU placement: not pursued

Recorded so the question does not get reopened without new information. Resident on GPU is a clear
no: 26.0 GiB peak + 10.58 GiB = 36.6 GiB against 30.3 GiB, and with WS5's weight offload dropped
there is no longer a path to the headroom it would need. A transient GPU encode (load → encode all
prompts → free) needs ≈21.6 GiB at that phase and would fit today, but it buys only ~3.7 s per case
over CPU bf16 while adding an offload path — not worth the machinery. Revisit only if the encoder
becomes a measured bottleneck after WS2.

**Gate note:** bf16 embeddings differ from fp32 (`max_abs 1.83`, `mean_abs 2.4e-4`, ≈7 % of mean
magnitude), so the exact-match gate cannot apply. The CUDA path runs the encoder in bf16 on
device — fp32-on-CPU is the XPU-specific deviation — so this should *improve* CUDA parity.
Validate against the CUDA reference as well as the XPU baseline.

### WS9 verification (2026-09-21) — accepted and re-anchored

- **Measured, in-pipeline, same sample and same code path as the pre-WS9 profiler run:**
  `conditions` **58.69 s → 27.56 s (−31.1 s)**. A second run recorded 32.24 s, so call it
  **27.6–32.2 s** against 58.7 s; the spread is CPU text-encoder scheduling variance. Saving is
  larger than the 1.7 s encode delta measured in isolation, so most of it comes from the load
  path fix (constructing in bf16 instead of materializing fp32 and converting) rather than from
  the encode itself. **Both halves of WS9 paid.**
- **Throughput unchanged, as expected:** the VAE pipeline interval is **48.499 s** against
  48.49 s before WS9, and steady-state chunk time is 3.44 s. WS9 is upstream of the rollout.
- **Determinism:** two fresh processes produced byte-identical MP4s,
  `md5 380d9cf74a9018d76408a3e615558fab`,
  `sha256 2da7fa11bc10faf9d8309f84e7f5365708d05f190baaf71e6d2b4b3e27023a5c`. **This is the new
  anchor.**
- **Decision: accepted.**

#### PSNR against the old anchor is not a usable gate here, and this generalizes

Measured against the WS1 anchor, frame by frame: **45.79 dB at frame 1 decaying monotonically to
15.71 dB at frame 160, mean 28.27 dB.** That decay curve is the signature of an autoregressive
trajectory diverging from a tiny initial perturbation, not of degraded output. Visual inspection
of frames 0, 80, and 159 confirms it: same scene, same camera path, both sharp and coherent, with
the two runs drifting to slightly different camera offsets by the end.

**Therefore: for this pipeline, per-frame PSNR against an XPU eager baseline is a weak gate for
any numerics-changing workstream.** Self-forcing amplifies any perturbation, so WS10, WS14, and
WS15 will all show the same decay and a low tail PSNR regardless of whether they help or hurt.
The usable gates are:

1. **Exact match**, for numerics-neutral changes — which is why WS12 and WS13 are ranked first.
2. **Early-frame PSNR** (frames 0–20, before divergence dominates) as a sanity check that the
   change is a small perturbation rather than a bug. WS9 scores 45.8–38.0 dB there.
3. **Visual inspection of late frames** for coherence and sharpness, not for pixel agreement.
4. **CUDA parity**, where a reference exists at the right shape. Note the reference in
   `outputs/wan22-cuda-decode-reference/` is at the 81f fixed-route shape
   (`[1, 39, 48, 30, 54]`, seed 1602803860), *not* the camera route, so it cannot be compared to
   this run directly. Generating a camera-route CUDA reference is the missing asset for gates 4.

## WS10 — fp32 matmul precision knobs

**PSNR gate.** Last deliberately: most likely to cost accuracy for the least gain, so evaluate it
only once the structural work is done and WS2 can prove whether it buys anything.

1. `torch.set_float32_matmul_precision("high")` once at inference setup. Most of the rollout is
   already bf16 under autocast, so the reachable surface is the fp32 residue only: the scheduler's
   `add_noise`, VAE fp32 sections, RoPE and norm math.
2. Re-check `torch.backends.mkldnn.allow_tf32` (currently `False`) for the CPU-side ops.
3. Do **not** widen autocast coverage or add fp8 here. FP8 is Phase 2 in the port plan, and the
   softmax history on this stack argues for one precision knob at a time.

## WS11 — chunk size 1 latent frame (blocked: needs a quality experiment)

Measured on the real module with random latents (timing-valid only):

| Block | Denoise/forward | 4 denoise | Decode/tile | **Latency** | Per 12 frames | fps (serial) |
| --- | --- | --- | --- | --- | --- | --- |
| 3 (today) | 0.379 s | 1.52 s | 1.72 s (12 frames) | **3.24 s** | 3.63 s | 3.3 |
| 1 | 0.248 s | 0.99 s | 0.57 s (4 frames) | **1.56 s** | 5.49 s | 2.2 |

A 1-frame block **halves interactive latency** and cuts camera-input quantization from 0.75 s to
0.25 s of video, but costs ~51 % more compute per output frame. Per-forward cost falls only 1.53×,
not 3×, because the fixed overhead (30 blocks of launches, attention against a KV window that does
not shrink) does not scale with the smaller query — which means WS3, WS4, WS7, and WS8 would
improve the small-block case the most, so re-measure after them.

The plumbing is easy and mostly landed. The blocker is **model validity**: block size is a trained
property. `attention_block_size = frame_seqlen * num_frame_per_block` (`causal_model.py:360`)
defines the chunkwise-causal geometry the weights were trained under — within a block frames are
denoised jointly under a shared timestep and attend to each other, and only across blocks is
attention causal. A 1-frame block removes that intra-block coupling, which is out of distribution
for this checkpoint, and the surrounding invariants (`local_attn_size=18`,
`max_prior_clean_chunks=5`, `score_local_attn_size=21`, asserted in `sgf.py:125-127`) are expressed
against 3-frame chunks.

Gate the *default* on generating the same case at block 1 and block 3 and comparing against both
the block-3 baseline and the CUDA reference. Expect degradation; if it is unacceptable, a 1-frame
block needs a fine-tune at that block size, which is out of scope for an inference-only plan. Do
not ship a latency win that quietly changes generation quality.

## WS12 — hoist the PRoPE projection-matrix construction out of the per-layer loop

**Exact-match gate** — the values are identical, they are just computed once instead of 60 times.
No prerequisites. **Do this first: it is the cheapest item in the document.**

`_prepare_apply_fns_all_dim` (`camera_prope.py:241`) builds `P`, `P_T`, and `P_inv` from
`transform_relative_viewmats(viewmats)` and the normalized intrinsics. Nothing in it depends on
the layer, the hidden state, or the timestep — only on the camera tensors, which are constant for
the whole chunk. `_apply_fused_prope` (`causal_model.py:661`) nevertheless calls it twice per
layer (once for the query span, once for the KV window), so it runs **60 times per forward**.

- **Projection.** Measured 0.551 ms per layer (0.276 ms query side + 0.271 ms KV side + 0.004 ms
  for the two `transform_relative_viewmats` calls). Currently 30 × 0.551 = **16.5 ms per
  forward**; hoisted, **0.55 ms per forward**. Saving 16.0 ms/forward = **80 ms per chunk**,
  chunk 3.44 → ~3.36 s (projected; assumes the saving is exposed rather than hidden, and at 99 %
  device occupancy it is).
- **Secondary effect, unquantified but real:** it removes 59 of 60 calls' worth of tiny kernels —
  most of the ~1,700 `fill`/`zeros_like`/`reciprocal`/`scalar_tensor` launches per forward, from
  `_invert_SE3`, `_lift_K`, and `_invert_K` building 4×4 matrices elementwise.
- **Pre-check.** None needed beyond the measurement above; this is redundant computation, not a
  bet on device behaviour.
- **Where.** The natural seam is to compute the three matrices once in `CausalWanModel.forward`
  alongside `_stage_fused_camera_cache` and thread them through `block_kwargs`, the same way `e0`
  and `freqs` already are. That also removes the
  `KeyError on kv_cache['_fused_prope_camera_metadata']` recompile guard for WS7, since the
  metadata stops being read off `kv_cache[0]` inside the block.
- **Risk.** Low. Keep the per-layer path for the training and Stage1 callers that do not have a
  per-chunk-constant camera, and restrict the hoisted path to standalone camera inference, the
  way WS4 restricted circular mode.

### WS12 result (2026-09-21) — done, bit-exact, 70 % of projection

Implemented as a per-forward memo rather than as threading the matrices through `block_kwargs`.
`CausalWanModel.forward` creates an empty `prope_cache` dict when the cache is circular and the
mode is `fused_prope`, passes it down through `CausalWanAttentionBlock.forward` to
`CausalWanSelfAttention.forward`, and `prope_apply_fns_separate_cached` (new, in
`camera_prope.py`) memoizes the query-side transforms under `"q"` and the K/V transforms under
`("kv", (window_start, window_end))`. **Why the memo instead of the hoist:** the window bounds
are derived inside the attention module from the circular cache's own pointers, so computing
them in `forward` would mean duplicating the ring arithmetic in a second place — a correctness
hazard for a 56 ms win. The memo gets the same 60→2 call reduction without moving that logic,
and keying the K/V half on the window makes a stale slice impossible rather than merely unlikely.
Passing `prope_cache=None` restores the per-layer path for every training and non-circular caller.

- **Measured: steady-state chunk 3.4447 s → 3.3890 s, −55.7 ms (1.6 %).** Within-run spreads do
  not overlap (3.4391–3.4526 against 3.3762–3.3992), so the delta is well outside run noise.
  VAE pipeline interval 48.574 → 47.735 s.
- **Gate: passed exactly.** `video.mp4 md5 380d9cf74a9018d76408a3e615558fab` and
  `compare.mp4 md5 fb3686583a1c5237ca5f60b91a86f0f7`, both matching the WS9 anchor.
- **Projection accuracy: 56 of 80 ms, i.e. 70 %.** The isolated per-layer measurement
  (0.551 ms) overstates the in-situ cost by ~30 %; the honest reading is that
  microbenchmarked helper costs are an upper bound on what deleting the call returns, because
  some of those tiny kernels were already interleaving with neighbouring work. **Apply the same
  discount when reading the WS13 and WS14 projections**, which were derived the same way.

## WS13 — encoded KV ring: stop re-encoding the window every forward

**Exact-match gate.** Prerequisite: WS12 (so the matrices being applied are already hoisted).
**This is the largest remaining win and the architectural prerequisite for WS7.**

`causal_model.py:1177-1204` slices the whole logical window out of the circular cache and
re-encodes all of it on every forward: RoPE over 7290 K tokens plus the PRoPE feature transform
over 7290 K and 7290 V tokens. But `local_end_index` advances only on the commit forward
(`_apply_cache_updates`), so `window_start` and `window_end` are **identical across all five
forwards of a chunk**, and the denoise writes land in the same 1,215 slots each time. The 6,075
history tokens are bit-identical on all five passes.

- **Projection.** Measured per layer: RoPE on the 7290-token window 2.203 ms, PRoPE on K 0.881 ms,
  on V 0.881 ms ⇒ 3.965 ms/layer ⇒ 118.9 ms/forward ⇒ **594 ms per chunk**. Split at
  6075 history / 1215 new: history once per chunk 99.1 ms, new tokens 5 × 19.8 ms = 99.1 ms ⇒
  **198 ms per chunk**. Saving **396 ms per chunk**; diffusion 1.693 → ~1.30 s, chunk
  3.44 → **~3.04 s** (projected from measured per-call costs; assumes a persistent encoded ring,
  see the trap below).
- **Why it is bit-exact.** RoPE is elementwise and PRoPE's `_apply_tiled_projmat` is a per-token
  einsum over a 4-element index, so splitting along the token axis cannot change any reduction
  order. The exact-match gate applies and must pass.
- **Why the encoding cannot be cached *across* chunks.** Positions are window-relative:
  `_rope_q_and_window_k` assigns `local_start=0` to the oldest window frame, so when the window
  slides by one chunk every retained frame's position shifts by −3 frames. Re-encoding the
  history once per chunk is therefore the correct granularity, and it is exactly the 5× reuse
  costed above. (Making positions absolute would allow cross-chunk caching but is out of
  distribution for this checkpoint — that is a WS11-class quality experiment, not an
  optimization.)
- **The trap that decides whether this pays.** Do **not** implement it as
  `torch.cat([encoded_history, encoded_new])`. That is 44.8 MiB per tensor per layer, so
  ~2.7 GB of traffic per forward, ~45 ms at the measured 1814 GB/s — it would eat most of the
  saving. Write encoded tokens **into a persistent encoded ring** so the window slice stays a
  view, exactly as WS4 did for the raw cache.
- **Memory pre-check (performed 2026-09-21).** Exact tensor arithmetic, confirmed against the
  5.01 GiB figure already recorded for the mirrored self-KV cache:

  | Buffer | Size |
  | --- | --- |
  | raw ring, mirrored at 2× capacity (current) | 5.006 GiB |
  | raw ring, unmirrored | 2.503 GiB |
  | encoded window buffer (1× capacity, K+V, 30 layers, bf16) | 2.503 GiB |
  | **naive: keep the mirror and add the encoded ring** | **+2.503 GiB — does not fit** |
  | **drop the mirror, add the encoded ring** | **±0.000 GiB** |

  Measured current peak is **25.413 GiB allocated / 29.897 GiB reserved** of 30.296 GiB, i.e.
  **0.399 GiB of headroom**, so the naive form is not an option. **Dropping the mirror pays for
  the encoded ring exactly.** That is also the right design: the mirror exists only so the
  wrapped logical window is one contiguous slice for SDPA, and once SDPA reads the *encoded*
  buffer — which is written in window order and contiguous by construction — the raw ring never
  needs a contiguous window again. The once-per-chunk re-encode reads the raw ring as two
  slices across the wrap.
- **WS7 payoff.** The compiled region then receives a fixed-shape `[1, 7290, 24, 128]` encoded
  window and no cache-derived Python integers, which the synthetic pre-check measured as
  **12 compiled graphs → 1**.

## WS14 — VAE decode elementwise in bf16

**PSNR gate** (see the WS9 note on why PSNR against the XPU baseline is weak here; use
early-frame PSNR plus visual inspection). No prerequisites, independent of WS12/WS13.

VAE decode is 1,695 ms per tile (measured): convolution 837 ms, elementwise copy/cast 374 ms,
elementwise math 370 ms, fp32 SiLU 59 ms, other 55 ms. Convolution is irreducible; the **745 ms
of fp32 elementwise work is not**. `streaming_decode_session`'s `decode_tile`
(`components.py:301-310`) runs the convolutions under bf16 autocast but returns
`decoded.float()`, and the surrounding residual/normalization math is fp32 throughout.

- **Projection.** 745 → ~250 ms per tile, so tile 1.695 → **~1.14 s** (projected; assumes these
  kernels are bandwidth-bound, which the measured elementwise throughput supports:
  548 GB/s at fp32 versus 1814 GB/s at bf16, a 3.3× ratio).
- **Pre-check.** Confirm the kernels are bandwidth-bound rather than launch-bound by comparing
  the per-kernel times in the VAE queue against their tensor sizes; 374 ms across 661 launches
  and 370 ms across 525 launches per tile is ~0.6 ms per kernel, far above launch cost, so this
  is very likely right — but measure it rather than assume it.
- **Risk.** This is softmax-bug-adjacent territory on this stack. Re-run
  `scripts/debug/xpu_softmax_bug_mre.py --also-sdpa` after the change, per the regression gate.
  Preserve each artifact's existing rounding rule exactly (`video.mp4` rounds, `compare.mp4`
  truncates) or the comparison will fail for reasons unrelated to dtype.

## WS15 — bf16 RoPE rotation instead of complex128

**PSNR gate.** Prerequisite: **WS13** — do not do this first. Before WS13 the RoPE cost is
79.3 ms/forward and worth attacking; after WS13 it is ~26 ms, and the remaining win is ~15 ms.
Sequencing it second means one implementation instead of two.

`echorope_apply` (`causal_model.py:245-257`) casts activations to `torch.float64`, forms a
`complex128` view, multiplies by a `complex128` frequency table, and casts back. The frequency
table itself is built as `complex128` by `rope_params`.

- **Projection.** ~26 → ~11 ms per forward after WS13 (projected from the microbenchmark below;
  assumes a minimal-traffic implementation).
- **Pre-check (performed 2026-09-21), and it contains the trap.** A naive fp32 real-valued
  rewrite is **slower than the complex128 path**: at L=7290, complex128 1.428 ms versus fp32
  3.027 ms, i.e. **0.47×**, because the fp32 version materializes more intermediates. Only a
  minimal-traffic bf16 form wins: 0.942 ms at L=7290 (1.5×) and 0.120 ms versus 0.248 ms at
  L=1215 (2.06×). **The reason fp64 costs what it does is bandwidth, not FLOPs** — measured
  elementwise `mul` throughput is 527 GB/s at fp64 versus 548 GB/s at fp32 and 1814 GB/s at
  bf16, so fp64 costs ~2× fp32 purely in bytes moved. Anyone implementing this on the assumption
  that "fp32 is faster than fp64" will ship a regression; implement against the microbenchmark.
- **Accuracy.** Against the complex128 reference on random activations, max-abs deviation is
  1.56e-2 for fp32 and 3.12e-2 for bf16 — both dominated by bf16 input quantization rather than
  by the rotation arithmetic. Group this with WS10 and WS14 so the anchor moves once.

## Out of scope

From the tuning guide, these apply to training or to models we do not run: `DataLoader`
worker/`pin_memory` tuning, `set_to_none` gradient handling, activation checkpointing, DDP/FSDP
gradient-bucket and `no_sync` tuning, rank load balancing, and bias removal in conv-then-norm
blocks (a weight-layout change that would invalidate the checkpoints). `torch.no_grad()` is
already in place and the diffusion module is already `eval().requires_grad_(False)`;
`torch.inference_mode()` is a possible small upgrade but must be checked against the cache tensors
being mutated in place.

**Work-reduction levers are deliberately out of scope for this document.** Per the real-time
ceiling above, the irreducible kernel work is 1.51 s per chunk (7.6 published fps), so reaching
16 fps requires generating less work rather than executing the same work better: fewer denoise
steps than NFE4, lower resolution, int8/fp8 on the XMX units (measured bf16 ceiling 150 TFLOP/s;
int8 throughput not measured), or a cheaper decoder. Each changes what the pipeline produces or
requires retraining, so each belongs to a model/product plan with its own quality gates. They
are recorded here only so the ceiling is not rediscovered as an optimization failure.

## Implementation status

Statuses are in the glance table above. Only the partially-done workstreams need detail.

### WS2 — done

The opt-in camera-route measurement harness is implemented and its JSONL report was verified on
the B70 without changing the camera-route MP4 digest. See WS2 for the measured baseline.

### WS3 — done

The schedule is materialized once per rollout, finiteness is accumulated on device, and standalone
Stage2 KV-cache indices are now host integers. Both causal-attention paths retain support for the
legacy tensor indices needed by training and sequence-parallel callers. The model threads the
precomputed physical frame sequence length to cache attention; low-level callers that omit it
retain the legacy `grid_sizes` fallback. The B70 camera-route smoke reproduced the WS1 digest
exactly after this change.

### WS4 — done

Camera-length inference defaults to `inference.kv_cache_mode: circular`, an inference-only
mirrored cache whose wrapped logical window is contiguous for SDPA. Denoise writes are speculative
and commits alone advance the ring pointer. `inference.kv_cache_mode: clone` preserves the
accepted clone implementation as a fallback. Circular mode is restricted to `sink_size=0`;
training, nonzero-sink, and recompute paths remain clone-based. The B70 exact-match and memory
gates pass.

### WS5 — done

- **Done:** streaming decode is the unconditional production path — `_stage2_generated_sample`
  always calls `vae.decode_streaming` and the single-shot `vae.decode(use_cache=False)` branch is
  gone. The per-chunk noise-slice clone is also removed, which is half of lever 3; the pre-decode
  `.contiguous()` remains.
- **Done:** standalone camera inference preallocates output, initial-noise, and denoise-noise
  buffers to `inference.max_rollout_latent_frames` (900 in the production example). Every
  random draw fills an exact-shaped view, retaining the previous generator draw ordering and
  shapes. Training and fixed-length inference still use the allocation-based path.
- **Dropped:** the diffusion-weight offload during decode, on latency grounds — see the rejection
  note in WS5.

### WS9 — done and verified

XPU Stage2 constructs UMT5 directly as CPU bf16. Verified in-pipeline on 2026-09-21: `conditions`
58.69 s → 27.56 s, output re-anchored and reproduced by two fresh processes. See
[WS9 verification](#ws9-verification-2026-09-21--accepted-and-re-anchored).

### WS11 — plumbing only

`_stage2_self_forcing_latents` accepts any `num_frame_per_block` that evenly divides the rollout
horizon, not just 3. Shipped configs are unchanged at 3. **No quality validation has been run at
other block sizes** — do not change the default without the experiment described in WS11.

### WS12 — done

Per-forward PRoPE transform memo in `camera_prope.prope_apply_fns_separate_cached`, enabled only
for circular `fused_prope` inference. Measured −55.7 ms/chunk, bit-exact.

### WS16 — done

`echorope_apply`'s bounds diagnostic is scoped to the arbitrary-index branch, removing 120
device syncs per forward. Measured −28.1 ms/chunk, bit-exact.

### WS17 — rejected on speed, kept for correctness

Scheduler grid and VAE `_scale` caching landed and is bit-exact (autocast state is part of the
`_scale` cache key, fixing a real precision hazard the naive cache introduced). Measured ~4 ms/
chunk, within noise. See the WS17 section above for the profiler-attribution finding it produced.

### WS13–WS15 — not started

All three are specified with their projections, assumptions, and pre-check results in their own
sections. Recommended order: **WS13 → WS14 → WS15**. WS13 is numerics-neutral and carries the
exact-match gate against the WS9 anchor, so it can land and be verified without moving it;
WS14 and WS15 change numerics and should be grouped with WS10 so the anchor moves once.

**Cumulative measured position: chunk 3.4447 s → 3.3609 s** across WS12 and WS16, both
bit-exact against the WS9 anchor. WS17 landed on top of that (also bit-exact) but moved nothing
measurable (~4 ms, within noise), so the cumulative figure is unchanged. WS13 is now the only
unexplored item with a projected gain above 1 %.

### Verification

The full Wan22 suite passes (318 passed, 1 pre-existing CUDA-only skip). The B70 camera-route
smoke after WS5 reproduces the WS1 anchor exactly (`md5 92beab461308361a3110364da8cbf97f`), so
the camera preallocation and reusable RNG buffers hold the exact-match gate.

### Incidental bug fix found while implementing WS5

The VAE-decode finiteness gate compared a float32 `torch.isfinite(decoded).float().mean()` against
exactly `1.0`. Streaming decode concatenates per-tile results before that reduction, changing the
summation order relative to a single-shot decode; on the production tensor size it rounds to
`1.0 + 1 ulp` even though every pixel is finite, and the run failed with
`finite_fraction=1.0000001192092896`. Fixed by gating on the exact boolean reduction
(`torch.isfinite(decoded).all()`) and computing the float mean only for the error message. The
pre-existing check was already fragile — a few non-finite pixels among ~4×10⁷ elements could
round-trip through the mean and go undetected — so this is a correctness fix independent of WS5.

## Status log

Append-only. Correct an earlier entry by adding a new one that supersedes it; do not edit history.

### 2026-09-21 — Citation audit: file:line references throughout this document were re-verified

Every `file.py:line` citation in the document was checked against the current source and 12 were
found stale, drifted by the WS12/WS13-diagnosis/WS16/WS17 edits (`causal_model.py`, `camera_prope.py`,
`components.py` all grew or shifted lines). All were corrected to current line numbers except
citations inside earlier dated entries in this status log, which are left as literal records of
what was true when written, per the append-only rule above. No code changed.

### 2026-09-21 — WS17: scheduler/VAE host-constant caching rejected on speed, kept for correctness

A chunks 4–5 profile attributed 1369.0 ms (36.7 % of device time in the window) to `Memcpy M2D`,
traced to two sites re-uploading host constants every call: `flow_to_x0`'s scheduler
`timesteps`/`sigmas` (1000-element fp64, `components.py:632-633`) and `Wan5BVAE._scale`'s
mean/std grids (`components.py:203-206`). Cached both. **Measured: ~4 ms/chunk gain, within
noise (3.7–7.7 ms run-to-run stdev).** Re-profiling after the fix showed `Memcpy M2D` count fall
42 → 10 and its attributed time fall 1369.0 → 1074.2 ms while wall clock did not move — the
attributed time was never recoverable. The 10 surviving copies transfer 10 KiB total while
"costing" 1074.2 ms; the trace's own bandwidth field for several of them reports ~1e-5 GB/s,
physically impossible for a transfer. **`Memcpy M2D` `dur` on this profiler is queue-completion
latency on an in-order device queue (`completion_timestamp − append_timestamp`), not transfer
cost** — a copy queued behind a compute backlog inherits that backlog's time whether or not
anything is waiting on the copy. Ruled out two alternate explanations by measurement rather than
assumption: `ZE_DEBUG=1` (requested as a "force synchronous launch" flag) is actually the Level
Zero loader's verbose call tracer and was killed after emitting 946 KB of log in 60 s without
reaching a profiled chunk; `UR_L0_USE_IMMEDIATE_COMMANDLISTS=1` and `=0` (with
`EVENTS_PER_BATCH=1`) both left total device time and `Memcpy M2D` attribution within noise of
the default (15.217–15.299 s total across all three), ruling out command-list batching mode as
the mechanism. Getting genuine per-kernel attribution here needs `torch.xpu.synchronize()`
inserted in code around each op, not an environment variable; not pursued, since wall-clock A/B
already settles what matters. **The caching also fixed a real bug it introduced along the way:**
a naive `(device, dtype)`-keyed `_scale` cache broke exactness (deterministically, verified by
two identical reruns) because `1.0 / std.to(bf16)` is promoted to fp32 under `torch.autocast` but
stays bf16 outside it, so the same key silently held two different results depending on call
order. Fixed by adding autocast state to the cache key. **Decision: rejected on speed, kept for
correctness and profile-reading clarity.** Generalizable addition to how this document reads
profiles: an op with a physically implausible reported bandwidth is a queue-drain wait marker,
not workload, and every change must still be gated on wall-clock A/B. This refines rather than
contradicts WS16 — a host `.item()` sync drains the queue and is measurably recoverable because
the host was genuinely blocked (WS16: −28.1 ms), while an async copy waiting behind a backlog is
not, because nothing was waiting on it.

### 2026-09-21 — WS7 rejected on measurement; compile blockers removed first

Made compile actually work, then measured that it does not pay. Raised Dynamo's per-block
variant ceiling to 32 (real count is **27**, not the 11 previously estimated — that estimate
missed the ramp's shape specializations), giving **zero eager fallbacks**, and removed the sole
graph break in the whole transformer block (`grid_sizes.tolist()` in `echorope_apply`, 83
occurrences) by threading Python int grids from `CausalWanModel.forward`, giving **zero graph
breaks**. Result with a warm inductor cache: **compiled 3.3602 s/chunk against eager
3.3609 s** — 0.7 ms, noise — plus **79 s of warmup** and a digest outside the exact gate.
Break-even ≈113,000 chunks. Root cause is the same 99 %-busy finding: no launch overhead to
recover, GEMMs already at 86–95 % of ceiling, and the only fusible elementwise work is the
redundant RoPE/PRoPE that WS12/WS13 delete instead. `max-autotune` skipped on arithmetic (≤1.9 %
addressable). **Kept in tree:** the variant-limit raise (inert, compile defaults off) and the
`grid_list` plumbing (bit-exact, verified by an eager control run reproducing the anchor).

### 2026-09-21 — WS16: last device syncs removed, and the "99 % busy" argument gets a limit

Removed the 120 `.item()` syncs per forward in `echorope_apply`'s bounds diagnostic by scoping
it to the arbitrary-index branch; the contiguous branch had already proved the same bound from
Python ints. **Chunk 3.3890 → 3.3609 s (−28.1 ms, 0.83 %)**, bit-exact. This **contradicts the
WS7 re-diagnosis**, which predicted ~nothing on the grounds that the queue is saturated. The
correction is reusable: a sync does not merely make the host wait, it **drains the queue**, so
120 per forward punch 120 bubbles into a 99 %-busy pipeline. The 99 %-busy argument still
correctly rejects launch-overhead work; it does not reject sync removal.

### 2026-09-21 — WS4 A/B measured: keep it, it is worth 199 ms/chunk

WS4 had been accepted on correctness and memory grounds with "no isolated time delta" recorded,
so it was re-examined for possible revert. A direct A/B via `inference.kv_cache_mode`:
**clone 3.5881 s/chunk against circular 3.3890 s — circular is 199 ms (5.5 %) faster**, ranges
non-overlapping, same digest from both. The clone arm did not carry WS12 (the memo is gated on
circular), so crediting it the same 56 ms still leaves WS4 worth **~143 ms (4.0 %)**.
**WS4 is kept.** It is also a prerequisite for WS13, which needs the ring's fixed addresses.

### 2026-09-21 — WS12 landed: first throughput gain since WS4

Per-forward memoization of the PRoPE projection matrices, 60 constructions per forward down to
2. **Steady-state chunk 3.4447 s → 3.3890 s (−55.7 ms, 1.6 %)**, bit-exact against the WS9
anchor on both `video.mp4` and `compare.mp4`, full suite 325 passed. Realized 70 % of the 80 ms
projection — recorded as a general caution that isolated helper timings are an upper bound on
what deleting the call returns, which applies to the WS13 and WS14 projections too. Implemented
as a memo keyed on `(window_start, window_end)` rather than as a hoist into
`CausalWanModel.forward`, to avoid duplicating the circular ring arithmetic in a second place.

### 2026-09-21 — WS6 partially reverted; the overlap machinery is gone, the tiling stays

Removed the second `torch.xpu.Stream`, both `Event` pairs, the double pinned buffers, the
out-of-order check, and `_drain` from `_Stage2XpuVaePipeline` — about 120 lines whose only
purpose was overlap that measures 1.001×. **Kept** per-chunk incremental decode and GPU-side
uint8 quantization, which are independent of the overlap claim and load-bearing at the
configured `max_rollout_latent_frames: 900` horizon: ~3,597 pixel frames at 480×864 accumulate
**17.9 GB on the host in fp32 versus 4.5 GB in uint8** against 30 GiB of host RAM. Decode now
runs on the caller's stream with one synchronization per tile before the pinned buffer is reused.
**Verified bit-exact:** `video.mp4 md5 380d9cf74a9018d76408a3e615558fab` and
`compare.mp4 md5 fb3686583a1c5237ca5f60b91a86f0f7`, both matching the WS9 anchor run. Chunk time
unchanged at 3.44 s, VAE pipeline interval 48.574 s against 48.499 s (+0.15 %, noise), so the
second stream was confirmed to buy nothing in-pipeline as well as in the microbenchmark. The
`vae_decode.mode` provenance string `continuous_cached_xpu_pipeline_tiles` is unchanged and still
accurate; the telemetry record drops the now-meaningless `wait_seconds` and `blocked_submissions`
in favour of `decode_seconds`. Full Wan22 suite: 325 passed, 1 pre-existing CUDA-only skip.

### 2026-09-21 — Kernel attribution captured: the redundant-encoding finding

Aggregated the chunks 4–8 trace (1.4 GB, 4.79 M events) by GPU queue and cross-checked it against
isolated timings of the in-tree helpers at production shapes. **The device is ~99 % busy**
(1.693 s diffusion + 1.695 s VAE queue union-busy against a 3.44 s measured chunk), so WS3–WS6
were competing for ~1 % of scheduling overhead. **53 % of each diffusion forward is RoPE and
PRoPE re-encoding the whole KV window**, 152 ms against 90.7 ms of model GEMMs and 43.5 ms of
SDPA. Two redundancies: `_prepare_apply_fns_all_dim` runs 60× per forward on per-chunk-constant
camera tensors (WS12), and the 6,075 history tokens of the window are re-encoded on all five
forwards of a chunk although they are bit-identical (WS13). Added a
[Per-forward budget](#per-forward-budget-measured) section, the real-time ceiling, and WS12–WS15.
Also measured: model GEMMs run at 129–142 TFLOP/s against a 150 TFLOP/s ceiling, so matmul
autotuning has nothing to give; q/k/v fusion buys 1.01×. No code changed.

### 2026-09-21 — WS6 rejected and WS8 rejected on the overlap microbenchmark

Ran the two-stream pre-check that WS6 skipped. Two independent bf16 `4096³` GEMMs on separate
`torch.xpu.Stream`s: serial 78.70 ms, two streams **78.64 ms — 1.001×**. A small `conv3d` against
a large GEMM gives **0.995×**. The trace agrees independently: per-queue union-busy 8.464 s and
8.474 s, union across both 16.939 s — their exact sum. **The B70 does not overlap compute.** WS6
moves from "kept as default, no gain" to **rejected**; the second stream, `Event` pairs, and
double pinned buffers should be removed, while per-chunk incremental decode and GPU-side uint8
quantization are kept because they are independent of the overlap claim. **WS8 (XPU graphs) is
rejected before implementation** on its own stated gate: graphs address launch overhead, and at
99 % occupancy there is ~1 % of it. Total cost of both decisions: seven seconds of benchmarking.

### 2026-09-21 — WS7 re-diagnosed; two recorded claims were wrong

Ran a full compiled rollout with `TORCH_LOGS=recompiles,graph_breaks` plus a synthetic
variant-count harness. **Correction 1:** the variant count is **bounded at 11**, not 300 —
`_circular_capacity` is 7290 and the ring advances 1215/chunk, so `ring_start` cycles through 6
residues and `local_end_index` saturates. **Correction 2:** only **one** code object hits the
limit, `CausalWanSelfAttention.forward` on `local_end_index == 3645`; the rest of the block
compiles and runs, giving a measured **3.44 → 3.32 s/chunk (3.5 %)** with the RoPE/PRoPE 53 %
still eager. Recorded the complete eight-cause list, including a previously unrecorded graph
break at `causal_model.py:188` (`grid_sizes.tolist()`, 33 occurrences) and **120 surviving
`.item()` syncs per forward** in `echorope_apply`'s bounds diagnostic — which WS3 marked done.
The sync removal is a prerequisite, **not** a throughput win: the queue is already saturated.
Measured that forcing fixed shapes with an additive mask costs 1.45× on SDPA at `Lk` 7290, so the
ramp chunks should stay eager instead. WS7 is now blocked on WS13 rather than rejected.

### 2026-09-21 — WS9 verified, accepted, and re-anchored

Two fresh-process B70 camera smoke runs produced byte-identical output,
`md5 380d9cf74a9018d76408a3e615558fab`,
`sha256 2da7fa11bc10faf9d8309f84e7f5365708d05f190baaf71e6d2b4b3e27023a5c`. **This supersedes the
WS1 anchor `md5 92beab461308361a3110364da8cbf97f` as the production anchor.** Measured
`conditions` **58.69 s → 27.56 s** (second run 32.24 s) on the same sample and code path,
−31.1 s; the saving exceeds the 1.7 s isolated encode delta, so the load-path fix carried most of
it. Throughput unchanged as expected: pipeline interval 48.499 s versus 48.49 s, chunk 3.44 s.
PSNR against the old anchor decays 45.79 dB at frame 1 to 15.71 dB at frame 160 (mean 28.27 dB);
visual inspection of frames 0/80/159 shows the same scene and camera path, both sharp, drifting
to slightly different offsets. **Recorded a general gate finding:** because self-forcing
amplifies any perturbation, per-frame PSNR against an XPU eager baseline is a weak gate for every
numerics-changing workstream here, so WS10/WS14/WS15 must use early-frame PSNR plus visual
coherence plus CUDA parity instead. Note the existing CUDA reference is at the 81f fixed-route
shape, not the camera route, so a camera-route CUDA reference is a missing gate asset.

### 2026-09-21 — Memory headroom re-measured; the recorded peak was stale

Measured peak is **25.413 GiB allocated / 29.897 GiB reserved** of 30.296 GiB — **0.399 GiB of
headroom**, 98.7 % of the card. This supersedes the 23.67 GiB / 27.84 GiB recorded in the
one-latent-microtile entry below. Consequence for WS13: a 2.503 GiB encoded window ring does not
fit as an addition, but dropping the raw ring's 2.503 GiB mirror pays for it **exactly**, and the
mirror becomes unnecessary once SDPA reads the encoded buffer instead of the raw window.

### 2026-09-21 — Number-class audit of this document

Relabeled every runtime and memory figure as measured, projected, or estimated, and split the
at-a-glance table into separate projected and measured gain columns. This surfaced that WS3, WS4,
and WS5 were accepted without isolated timing measurements — their measured cells now say so
instead of implying the projection was achieved. No code changed.

### 2026-09-21 — WS9 implemented: text encoder constructed as CPU bf16

The XPU Stage2 adapter now constructs UMT5 directly in bf16 rather than materializing fp32 and
converting, which also removes the transient load spike. Focused tests pass. **Not yet verified:**
this changes prompt embeddings by design, so the next B70 smoke must record a new output digest
and a PSNR/visual comparison. The WS1 anchor does not apply to WS9 output.

### 2026-09-21 — WS6 throughput claim corrected

Previously recorded as "overlap verified" on the strength of asynchronous submission and
near-zero buffer waits. That was wrong: those establish scheduling correctness, not concurrent
execution. The measured 48.49 s pipeline interval sits within 5 % of the 50.82 s serial
projection and at 1.8× the 26.7 s overlapped projection, so **the kernels serialize and WS6
delivers no throughput gain**. Kept as default because it is correct and removes host-side
stalls. Root cause pending the chunks 4–8 profile.

### 2026-09-21 — Steady-state XPU profile captured

`runtime.stage2_xpu_profiler: true` writes one Chrome/Perfetto trace for chunks 4–8, with CPU/XPU
events, Python stacks, shapes, memory, and kernel launches. Two earlier attempts were killed
during trace export: a full-rollout capture with stacks exhausts host memory. Trace at
`outputs/wan22-ti2v-5b-stage2-xpu-profiler/runs/wan22-camera-xpu-profiler-20260921-071914/reports/stage2-xpu-profile.slot-000000.rank-00000.json`.

### 2026-09-21 — WS7 rejected after second attempt

Removed the RoPE `.item()` and stopped passing the unused varying offset to circular-cache
blocks. Dynamo then specialized on `kv_cache["local_end_index"]` instead and hit the eight-variant
limit again; output still failed the exact gate. Raising the limit would compile one graph per
chunk (300 at the configured horizon). VAE-only compile also rejected: 109.6 s compile, max-abs
1.332 against eager. Prerequisite is architectural — isolate mutable cache assembly from a
stateless compiled region — not a flag.

### 2026-09-20 — One-latent microtiles rejected

VAE-local peak allocated fell 8.86 → 3.79 GiB (measured) and output stayed byte-exact, but
process-wide peak moved only 23.67 → 23.64 GiB allocated and 27.84 → 27.84 GiB reserved, because
diffusion and the KV cache dominate it. Added 14.52 s of buffer-reuse waits. Reverted. Correct
measurement, wrong scope.

### 2026-09-20 — WS6 GPU-side uint8 quantization

VAE pixels are quantized on XPU before the pinned-host copies, cutting each D2H tile from float32
to uint8. Two representations retained because `video.mp4` rounds and `compare.mp4` truncates.
Both prior digests reproduced.

### 2026-09-20 — WS1–WS5 completed and anchored

Camera-length single-EMA route became production (WS1), establishing anchor
`md5 92beab461308361a3110364da8cbf97f`. Measurement harness (WS2), sync removal (WS3), circular KV
cache (WS4), and streaming decode plus preallocation (WS5) all landed and reproduced that anchor
exactly.
