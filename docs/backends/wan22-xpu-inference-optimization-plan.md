# Wan2.2 Stage2 XPU inference optimization plan

**Status:** partially implemented (see checklist below)
**Last updated:** 2026-09-20 (B70)
**Scope:** the Stage2 **inference** path only (`infer` / validation generation). Training,
FSDP, and multi-GPU paths are explicitly out of scope.
**Source guidance:** [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

Workstream identities (WS1…WS6) are stable; the sections below are ordered by **execution
order**, which is not the same as the numbering:

| Step | Workstream | Gate |
| --- | --- | --- |
| 0 | WS0 — measurement harness | — |
| 1 | WS1 — remove host/device synchronizations | exact match |
| 2 | WS3 — memory: streaming decode, preallocation, reuse | exact match |
| 3 | WS7 — pipeline diffusion against VAE decode | exact match |
| 4 | WS5 — `torch.compile` / inductor autotune | PSNR |
| 5 | WS6 — XPU graphs | PSNR |
| 6 | WS4 — text encoder precision and placement | PSNR |
| 7 | WS2 — fp32 matmul precision knobs | PSNR |
| — | WS8 — configurable chunk size (latency vs throughput) | quality experiment |

WS4 and WS2 sit last deliberately: both change numerics, and WS4's GPU variant can only be
decided once WS5/WS6 have claimed their memory.

## 1. Environment and baseline

| Item | Value |
| --- | --- |
| Device | Intel B70 `[0xe223]`, 30.3 GiB VRAM, 256 EUs, driver `1.14.37020+3` |
| Integrated GPU | **none available** — one render node, `lspci` shows only the discrete `e223`, `torch.xpu.device_count() == 1` |
| CPU | Intel Core Ultra 5 245K (Arrow Lake-S), 14 threads, AVX2 + AVX-VNNI + F16C; **no AVX-512, no AMX** |
| Host | 30 GiB RAM total |
| Torch | `2.12.1+xpu`, `triton-xpu 3.7.1`, inductor available |
| XPU graph API | `torch.xpu.XPUGraph`, `torch.xpu.graph`, `graph_pool_handle`, `make_graphed_callables` all present |
| fp32 matmul precision | `highest` (default); `torch.backends.mkldnn.allow_tf32` is `False` |
| Attention backend | SDPA (`wan22_attention_backend()` → `sdpa`; FlashAttention is CUDA-only) |

Workload for `infer_stage2_sgf_81f.yaml`, one sample, `live` + `ema` passes:

| Quantity | Value |
| --- | --- |
| Rollout | 39 latent frames, `num_frame_per_block=3` → 13 chunks per pass |
| Forwards per pass | 13 × (4 denoise + 1 commit) = **65**; 130 per run |
| Tokens per forward | 3 × `frame_sequence_length` 405 = **1215** query tokens, KV window bounded by `local_attn_size=18` |
| Decode | 39 latent frames → 153 pixel frames at 480×864 |
| Wall clock | ≈ 5 min total, **dominated by weight loading and the CPU text encoder**, not by generation (see §1.4) |
| Peak device memory | **26.0 GiB / 30.3 GiB** |
| Peak host RSS | **28.2 GiB / 30 GiB** |

Device memory by phase (from `/proc/<pid>/fdinfo` `drm-total-vram0`):

| Phase | Device memory |
| --- | --- |
| Diffusion weights resident (5B bf16) | 10.0 GiB |
| + VAE and conditioning | 12.0 GiB |
| Rollout steady state (KV cache filled) | 16.7 GiB |
| VAE decode | 26.0 GiB (decode working set alone measures 9.2 GiB) |

## 1.4. Measured latency budget

Timed by wrapping `provider.diffusion` on the real rollout path (30 latent frames, EMA role) and
by timing `decode_streaming_chunks` per tile. Per-forward cost ramps while the attention window
fills and saturates from chunk 6, which is `local_attn_size=18` latent frames:

| Chunk | Denoise (per forward) | Commit |
| --- | --- | --- |
| 0 | 0.517 s (includes warmup) | 0.23 s |
| 1–5 | 0.251 → 0.359 s | 0.25 → 0.36 s |
| **6+ (steady state)** | **0.378 s** | **0.397 s** |

| Steady-state quantity | Value |
| --- | --- |
| Diffusion per chunk (4 denoise + 1 commit) | **1.91 s** |
| VAE decode per 3-latent-frame tile (12 pixel frames) | **1.72 s** (first tile 2.58 s) |
| Total per chunk, serial | 3.63 s for 12 pixel frames ⇒ **3.3 generated fps** |
| Same, if decode overlaps the next chunk's diffusion | 1.91 s per 12 frames ⇒ **6.3 generated fps** |

At 16 fps playback one chunk is 0.75 s of video, so generation is ≈4.8× slower than real time
serially and ≈2.5× slower if decode is pipelined against diffusion.

**Interactive latency** (camera pose in → corresponding pixels out), which is the number that
matters for a viewer: the chunk's W2C matrices must be known before the first denoise step, so
the critical path is 4 denoise forwards + one decode tile = **1.51 s + 1.72 s ≈ 3.2 s**. The
commit forward (0.40 s) is not on that path — it only has to finish before the *next* chunk —
but it does count against throughput. On top of the 3.2 s, camera input is quantized to one
chunk: a pose applies to all 3 latent frames, so direction can only change every 0.75 s of
generated video.

For reference, a 39-frame pass is ~23 s of rollout plus ~22 s of decode, so the ~5 min run is
mostly the 38 GiB checkpoint load, the role reload for the second pass, and the CPU text
encoder — which is why §1.5 (dropping the `live` pass) and §8 (the encoder) move the wall clock
more than any kernel-level work here.

## 1.5. Production pass policy: EMA only

The baseline above runs `live` then `ema`, which roughly doubles the work for an output we
discard. That is not an accident of this host's config:

- The pass list comes from `validation.passes`, and `resolve_generation_plan` deliberately
  refuses inference-only sampler blocks so "standalone inference cannot drift from the
  validated contract" (`generation.py:62-92`). Stage2 SGF training validation compares the live
  student against its EMA copy, and standalone infer reuses that same list so a standalone run
  is provenance-identical to a validation run.
- `sgf.py:187-206` then **hard-requires** `["live", "ema"]` for standalone fixed-length
  inference: an EMA-only pass list raises `BackendContractError`. So dropping the live pass on
  the 81f route is a **contract change in `sgf.py`**, not a config edit.
- Role switching itself is `_load_role` reloading the selected payload from the 38 GiB
  `model.pt` into the same module, so it costs **load time, not peak memory** — there is only
  ever one resident copy of the weights.

Given the XPU weights policy (EMA is the only pass used for quality judgment) and the goal of
very long rollouts, the live pass is pure overhead here. Two ways forward, and the second is
probably the right one:

1. Relax `sgf.py` to allow `["ema"]` for standalone fixed-length inference. Cheap, but it edits
   a deliberate provenance contract, so it needs sign-off rather than a quiet patch.
2. Use the **camera-length route**, which already expresses exactly one pass with
   `weights: model` and no role switch (`generation.py:137`, `sgf.py:193-202`). It is also not
   pinned to the fixed route's mandatory 60-latent horizon (`sgf.py:214`), which the 81f
   contract enforces on every pass. For "very long / infinite rollout" work this is the route
   that can actually express the horizon, so aligning production inference with it solves the
   pass-count and horizon questions together.

Either way this is a ~2× wall-clock win on the whole run and it is independent of every
workstream below, so it belongs ahead of them in priority even though it is not an
"optimization" in the tuning-guide sense. It does not change the `ema` output bytes, so the
exact-match gate applies.

## 2. Regression gate (applies to every change below)

Three consecutive 81f runs on 2.12.1 produced **byte-identical** `ema` MP4s
(`md5 2985664b344b975588f70020668ec66e`, `sha256 c54da3c8dfa33f1bec843abb2196b9b2d1ef54476345e81aba38a21e3e3434cf`),
so the XPU path is currently deterministic run-to-run. Therefore:

1. **Exact-match gate** (WS1, WS3): the run must reproduce that digest bit for bit.
2. **PSNR gate** (WS5, WS6, WS4, WS2): these change numerics by design. Compare against the
   digest-matched baseline video with a per-change PSNR / max-abs-diff threshold, plus visual
   inspection of frame 0 (where the softmax bug showed first).
3. Keep the `torch==2.12.1+xpu` pin. Do not "fix" a performance problem by moving to a build
   that fails `scripts/debug/xpu_softmax_bug_mre.py`, and re-run that reproducer after any
   change that could re-route into a different softmax kernel.

## 3. Step 0 — WS0: measurement harness (do first)

Without per-phase numbers the later workstreams cannot be ranked or verified.

- Per-phase wall-clock timers around: weight load, `_conditions`, rollout (per chunk), VAE
  decode, MP4 encode. Emit as a JSON event so runs are comparable.
- Device memory: `torch.xpu.max_memory_allocated()` / `max_memory_reserved()` per phase, plus
  the external `drm-total-vram0` poller for true driver-side usage. The two disagree in a way
  that matters — see §5.
- Kernel-level attribution: `torch.profiler` with XPU activity for one chunk, to see whether
  time sits in SDPA, the linear layers, the norms, or host-side launch overhead. This decides
  how much WS5 and WS6 can possibly buy.

## 4. Step 1 — WS1: remove host/device synchronizations

Numerics-neutral, so the **exact-match gate** applies strictly.

The guide's "Avoid unnecessary CPU-GPU synchronization" item is the largest structural issue
in this path. Each `.item()` flushes the XPU queue, and the KV-cache bookkeeping does it
*inside every attention layer of every forward*:

- `modeling/causal_model.py:1034-1101` and `:1364-1390` — `kv_cache["global_end_index"].item()`
  and `kv_cache["local_end_index"].item()` drive Python `if` branches. With ~30 blocks and
  several reads per block this is O(100) syncs per forward, O(10k) per run.
- `stage2.py:686` and `:715` — `float(step.item())` per denoise step, though `steps` is known
  before the loop and can be materialized once on the host.
- `stage2.py:735` — `torch.isfinite(latents).all().item()` per chunk: a full-tensor reduction
  plus a sync, 13 times per pass.
- `stage2.py:937` (and `inference.py:1953`) — the decode finiteness reduction. Note
  `decode_streaming_chunks` (`components.py:286`) also syncs once per tile.
- `modeling/causal_model.py:1002`, `:1364` — `math.prod(grid_sizes[0][1:]).item()` recomputes a
  constant from a device tensor.

Plan:

1. Keep cache indices as **Python ints** in the cache dict (they are per-rank scalars in SP1 and
   already effectively host state), leaving device tensors only where a collective needs them.
   This removes the syncs *and* is the precondition for WS5 and WS6.
2. Precompute `steps` and the timestep tensors before the chunk loop; build the per-step
   `torch.full` timesteps once and mutate in place.
3. Replace per-chunk finiteness checks with a single check at the end of the rollout, or keep
   them behind a debug flag. The contract needs to fail the run, not fail per chunk.
4. Pass `frame_seqlen` in rather than deriving it from `grid_sizes` on device.

## 5. Step 2 — WS3: memory — streaming decode, preallocation, reuse

Numerics-neutral: **exact-match gate**, and the streaming-decode part has been verified
bit-exact (below).

### 5.1 Switch production decode to the streaming path — yes, it is safe

`stage2.py:917-931` only streams when `output_latent_frames > _STREAMING_VAE_LATENT_CHUNK` (60).
The production rollout is 39 frames, so it takes the single-shot `vae.decode` branch. Measured
on the production-shaped latents `outputs/wan22-cuda-decode-reference/latents_bf16.pt`
(`[1, 39, 48, 30, 54]`, XPU bf16 VAE):

| Path | Time | Peak `memory_allocated` | vs direct decode |
| --- | --- | --- | --- |
| `decode(use_cache=False)` (production today) | 23.20 s | 9.22 GiB | reference |
| `decode_streaming(chunk_latent_frames=39)` | 21.96 s | 9.22 GiB | **bit-exact** (max_abs 0) |
| `decode_streaming(chunk_latent_frames=12)` | 22.07 s | 8.96 GiB | **bit-exact** (max_abs 0) |
| `decode_streaming(chunk_latent_frames=6)` | 22.03 s | 8.90 GiB | **bit-exact** (max_abs 0) |

So the switch is safe under the exact-match gate: `cached_decode` over temporal tiles with one
continuous cache reproduces `decode` exactly, at every chunk size tested, for all 153 output
frames. It is also marginally faster.

Two honest caveats that change the earlier expectation:

- **The memory win at this horizon is small** (9.22 → 8.90 GiB, ~0.3 GiB). The decode working
  set is dominated by the VAE's per-tile cache and activations at full 480×864 resolution, not
  by temporal extent, so temporal chunking cannot cut it much. Spatial tiling could, but that
  introduces seams and would forfeit the exact-match gate.
- The real reasons to adopt it anyway: the fp32 output accumulates on the **host** instead of
  the device (≈0.8 GiB at 39 frames, ≈1.2 GiB at the configured 60-frame horizon), peak memory
  becomes **independent of the rollout horizon**, and the code path is then the same one the
  file-streaming branch already exercises.

For the very-long-rollout goal this is the enabling piece: `decode_streaming_chunks` already
yields finished tiles one at a time, and the KV window is already bounded by `local_attn_size`,
so device memory can be made flat in the horizon. The remaining horizon-dependent term is the
`output` latent buffer allocated for the whole rollout (`stage2.py:652-660`), which would need
to become a rolling window for an unbounded run.

### 5.2 Cache release does not lower the driver high-water mark

Tested explicitly, since the 26.0 GiB peak looked like fragmentation (KV-cache blocks freed but
retained by the allocator, then a differently-shaped decode allocating fresh blocks):

| Between rollout and decode | torch `reserved` after decode | driver `drm-total-vram0` |
| --- | --- | --- |
| nothing | 13.93 GiB | 14.44 GiB |
| `torch.xpu.empty_cache()` | 11.69 GiB | 14.55 GiB |

`empty_cache()` lowers torch's reserved pool by 2.2 GiB but leaves the driver-side high-water
mark unchanged: the driver does not return pages. **Do not expect peak relief from cache
release**, and treat `torch.xpu.memory_reserved()` as a poor proxy for what the card actually
holds. The corollary for `empty_accelerator_cache()` on role switch (`inference.py:1175`) is
that it costs pool rebuild time without buying headroom — measure and probably drop it.

### 5.3 The remaining levers, in order of size

1. **Offload the diffusion weights during decode** (~10 GiB) — **only with a preallocated
   pinned host buffer, and only if decode windows stay large.** Decode does not need the
   weights, so this is the one change that meaningfully lowers the run's peak and the only way
   to create room for WS4's GPU variant. Measured 10 GiB transfer cost on this host:

   | Host buffer | Offload (D2H) | Reload (H2D) | Round trip |
   | --- | --- | --- | --- |
   | pageable | 1.96 s (5.1 GiB/s) | 1.73 s (5.8 GiB/s) | 3.7 s |
   | **pinned** | 0.38 s (26.2 GiB/s) | **0.21 s (48.3 GiB/s)** | **0.6 s** |

   Pinned memory is 8× faster and makes the round trip negligible against a whole-rollout
   ~22 s decode; pageable at 3.7 s is not acceptable. Two conditions therefore apply. First,
   allocate the pinned staging buffer **once** and reuse it — a per-decode `pin_memory()`
   allocation would give back the win. Second, for long or unbounded rollouts decode
   interleaves with the rollout, and against the measured per-chunk budget of 3.63 s (§1.4) a
   0.6 s round trip per chunk is **~17 % overhead**; batching four chunks into one 12-frame
   decode window (~14.5 s) brings it to ~4 %. Size the decode window accordingly, or skip the
   offload entirely if VRAM allows.
   Note that with the EMA-only policy in §1.5 there is no role switch, so the offload buys
   headroom only — it no longer saves any reload the run was doing anyway.
2. **Reuse the noise buffers.** `provider._noise` (`stage2.py:1707`, `inference.py:1637`)
   allocates a fresh `torch.randn` per denoise step (4 per chunk). Draw into a preallocated
   buffer. Required for WS6, where addresses must be stable.
3. **Preallocate the maximum-shape buffers up front** (guide: "Preallocate memory in case of
   variable input length"). `rollout_latent_frames` varies per source (39 here, 60 configured),
   so allocate for the configured maximum once and slice.
4. Avoid the avoidable copies: `latents[:, :output_latent_frames].contiguous()` (`stage2.py:838`)
   and `initial_noise[:, start:end].clone()` (`stage2.py:676`).

## 6. Step 3 — WS5: `torch.compile`

**PSNR gate.** Prerequisite: WS1. Every `.item()`-driven branch in the attention cache path is
a hard graph break, so compiling before that lands buys almost nothing.

Shapes are favorable: 1215 query tokens per forward, a bounded KV window, 13 chunks with
identical shapes. The only varying input is the integer `current_start`.

1. Start with `torch.compile(module, dynamic=False)` on the transformer block, not the whole
   model, so a graph break costs one block rather than the pass.
2. Mark `current_start` as a dynamic int (or bucket it) to avoid 13 recompiles per pass.
3. Progress `default` → `mode="reduce-overhead"` → `mode="max-autotune"`, measuring each. The
   guide notes autotune and graph capture increase memory use, which is why WS3 comes first.
4. Expect a real warmup cost per process (the whole run is ~5 min, so a 60–90 s compile must pay
   for itself). Enable the inductor cache (`TORCHINDUCTOR_CACHE_DIR` on persistent storage) so
   repeat runs skip codegen.
5. Two compiled variants will appear: `cache_update_policy="none"` (denoise) and
   `"commit_detached"` (commit). Confirm both compile rather than silently falling back.
6. Keep `SOLARWM_COMPILE_FLEX` off. FlexAttention compilation is a Stage1 training concern
   (`causal_model.py:41`) and the inference path does not use it.

Risk: inductor fusion changes numerics, and XPU inductor coverage is less proven than CUDA's.

## 7. Step 4 — WS6: XPU graphs

**PSNR gate.** `torch.xpu.XPUGraph`, `torch.xpu.graph`, `graph_pool_handle`, and
`make_graphed_callables` all exist in `2.12.1+xpu`, so the guide's "Use CUDA Graphs" advice has
a direct XPU analogue. This is late because it requires everything above: no host syncs inside
the captured region (WS1), stable addresses for inputs, outputs, noise, and KV cache (WS3), and
headroom for the graph's private pool (WS3).

Capture one denoise step and one commit step as separate graphs, replay them 4× and 1× per
chunk, with `current_start` and the timestep fed through preallocated device tensors updated in
place. Compare against `torch.compile(mode="reduce-overhead")`, which may already deliver most
of the launch-overhead win with far less machinery; only hand-roll graphs if the profile still
shows launch overhead dominating.

## 8. Step 5 — WS4: text encoder precision and placement

**PSNR gate** (see the parity note below). Ordered after WS5/WS6 because those two consume the
memory that decides whether GPU placement is possible at all.

### 8.1 First: bf16 on the host, which is where it has to run

The instruction was to try bf16 on an integrated GPU, else CPU with AMX. Neither is available
here: there is no iGPU exposed (single render node, one `torch.xpu` device), and the Core Ultra
5 245K has no AMX and no AVX-512 — only AVX2, AVX-VNNI, and F16C. So "CPU bf16" on this host
means oneDNN on AVX2/VNNI. Measured anyway (UMT5-XXL, 512-token prompt, 14 threads):

| Placement | Resident | Encode (cold / warm) |
| --- | --- | --- |
| CPU fp32 (production today) | 21.8 GiB host | 13.7 s / 6.8 s |
| **CPU bf16** | **11.2 GiB host** | **5.1 s / 5.1 s** |
| XPU bf16 | 10.58 GiB device | 1.4 s |

CPU bf16 wins on both axes against the current fp32 placement — ~10.6 GiB less host RAM and
~25 % faster — even without AMX. **Do this first; it needs no device memory at all.**

### 8.2 Fix the load path, which is the actual host-RAM risk

RSS traced through the load stages:

| Stage | Host RSS |
| --- | --- |
| after `WanTextEncoder(...)` construction | 17.9 GiB (**parameters materialize as fp32**) |
| after `.to(cpu, bf16)` | 11.2 GiB |
| after `.to(cpu, fp32)` (production today) | 21.8 GiB |
| transient peak during load | **28.4 GiB of 30 GiB** |

The checkpoint on disk is already bf16, but construction upcasts to fp32, so the load spikes to
within ~1.6 GiB of the host limit regardless of the final dtype. Constructing in bf16 directly
(meta-device init plus `load_state_dict(assign=True)`, or an explicit construction dtype)
removes the spike. This is the single riskiest number in the baseline and it is independent of
everything else in this plan.

### 8.3 Only then consider the GPU, and only if headroom remains

- Resident on GPU today: **no.** 26.0 GiB peak + 10.58 GiB = 36.6 GiB against 30.3 GiB.
- Resident on GPU after WS3/WS5/WS6: decide from measurement, not from this document. WS3's
  realistic gain is the ~10 GiB diffusion-weight offload (§5.3), while WS5's autotune and WS6's
  graph pool both consume headroom. Re-measure peak memory after those land, and require the
  margin to hold at the configured 60-frame horizon, not just at 39.
- Transient GPU encode (load → encode all prompts → free) needs ≈ 21.6 GiB at that phase and
  would fit today, but it only buys ~3.7 s per case over CPU bf16 while adding an offload path.
  Park it as the fallback if §8.1 and §8.2 prove insufficient.

### 8.4 Parity note on the gate

bf16 embeddings differ from fp32: `max_abs 1.83`, `mean_abs 2.4e-4` (≈7 % of mean magnitude),
so the output video changes and the exact-match gate cannot apply. Note that the CUDA path runs
the encoder in bf16 on device — fp32-on-CPU is the XPU-specific deviation — so this change is
expected to *improve* CUDA parity rather than degrade quality. Validate against the CUDA
reference as well as the XPU baseline.

## 8.5. WS7: pipeline diffusion against VAE decode (latency and throughput)

**Exact-match gate** — this only reorders execution, it does not change any arithmetic. Best
scheduled right after WS3, since it needs the same double-buffered output allocations.

Today decode runs strictly after the rollout (or after each tile, serially), so the device
alternates between the two phases. `torch.xpu.Stream`, `torch.xpu.Event`, the `torch.xpu.stream`
context manager, and `non_blocking=True` copies into pinned host memory are all available in
`2.12.1+xpu`, so the two phases can overlap: while chunk *n+1* denoises, chunk *n* decodes and
its pixels copy back to the host.

Measured effect on the §1.4 budget:

| Schedule | Per 12 pixel frames | Generated fps | vs 16 fps real time |
| --- | --- | --- | --- |
| serial (today) | 1.91 s diffusion + 1.72 s decode = 3.63 s | 3.3 | 4.8× slower |
| pipelined | max(1.91, 1.72) = 1.91 s | 6.3 | 2.5× slower |

So pipelining is worth ~1.9× throughput, and it is the single largest structural win available
short of making the forwards themselves faster. Note what it does **not** buy: interactive
latency stays ≈3.2 s, because a given chunk's pixels still require that chunk's 4 denoise
forwards *followed by* its decode. Pipelining hides decode behind the *next* chunk's work; it
cannot shorten the dependency chain for the current one.

Implementation notes:

1. Put decode on a second XPU stream, with a `torch.xpu.Event` recorded after the rollout writes
   `output[:, start:end]` and waited on by the decode stream, so the tile is only read once the
   diffusion writes are visible.
2. Double-buffer the decode input and output tiles; a single scratch buffer would serialize the
   two streams again.
3. Move the D2H copy of finished tiles to pinned memory with `non_blocking=True`, and keep the
   MP4 encode on a host worker thread so the codec never blocks the device.
4. This requires WS1 first: `decode_streaming_chunks` currently calls
   `torch.isfinite(decoded).all().item()` per tile (`components.py:286`), which synchronizes and
   would collapse the overlap.
5. Attribute peak memory carefully — two streams in flight means two tiles resident, which eats
   some of the headroom WS3 frees.

## 8.6. WS8: configurable chunk size (1 latent frame) — feasible, but not free

**Requires a quality decision, not just a gate.** Measured both block sizes on the real module
(random latents, so timing-valid only):

| Block | Denoise/forward | 4 denoise | Decode/tile | **Latency (4 denoise + decode)** | Per 12 frames | Generated fps (serial) |
| --- | --- | --- | --- | --- | --- | --- |
| 3 (today) | 0.379 s | 1.52 s | 1.72 s (12 frames) | **3.24 s** | 3.63 s | 3.3 |
| 1 | 0.248 s | 0.99 s | 0.57 s (4 frames) | **1.56 s** | 5.49 s | 2.2 |

So a 1-frame block **halves interactive latency** (3.24 s → 1.56 s) and cuts camera-input
quantization from 0.75 s to 0.25 s of video, but costs ~51 % more compute per output frame
(5.49 s vs 3.63 s per 12 frames). The per-forward cost only falls 1.53×, not 3×, because the
fixed per-forward overhead (≈30 blocks of kernel launches, plus attention against a KV window
that does not shrink) does not scale with the 3× smaller query. That also means WS1/WS5/WS6,
which attack exactly that fixed overhead, would improve the small-block case the most — so
revisit this measurement after them.

Two distinct kinds of work are involved:

- **Plumbing (easy).** `stage2.py:651` hard-codes `chunk != 3` as an error, and the model
  contract asserts `num_frame_per_block == 3` (`contracts.py:498`). A 1-frame block runs through
  the existing KV-cache path without modification — the sweep above completed with no errors —
  so the runtime already tolerates it.
- **Model validity (the real question).** The block size is a **trained** property, not a
  runtime knob. `attention_block_size = frame_seqlen * num_frame_per_block`
  (`causal_model.py:360`) defines the chunkwise-causal geometry the weights were trained under:
  within a block the frames are denoised jointly with a shared timestep and attend to each
  other, and only across blocks is attention causal. Running with a 1-frame block removes that
  intra-block coupling, which is out of distribution for this checkpoint, and the surrounding
  invariants (`local_attn_size=18`, `max_prior_clean_chunks=5`, `score_local_attn_size=21`, all
  asserted in `sgf.py:125-127`) are expressed against 3-frame chunks.

Recommendation: treat the config knob as cheap and do it, but gate the *default* on a quality
experiment — generate the same case at block 1 and 3 and compare against the block-3 baseline
and the CUDA reference. Expect degradation; if it is unacceptable, a 1-frame block needs a
fine-tune at that block size, which is out of scope for this inference-only plan. Do not ship a
latency win that quietly changes generation quality.

## 9. Step 6 — WS2: fp32 matmul precision knobs (last)

**PSNR gate.** Deliberately last: it is the change most likely to cost accuracy for the least
gain, so it should only be evaluated once the structural work is done and the measurement
harness can prove whether it buys anything.

1. `torch.set_float32_matmul_precision("high")` once at inference setup. Most of the rollout is
   already bf16 under autocast, so the reachable surface is only the fp32 residue: the
   scheduler's `add_noise` (`stage2.py:723-729` upcasts to fp32), VAE fp32 sections, RoPE and
   norm math.
2. Re-check `torch.backends.mkldnn.allow_tf32` (currently `False`) for the CPU-side ops.
3. Do **not** widen autocast coverage or add fp8 here. FP8 is Phase 2 in the port plan, and the
   softmax history on this stack argues for changing one precision knob at a time.

## 10. Explicitly out of scope

From the tuning guide, these apply to training or to models we do not run: `DataLoader`
worker/`pin_memory` tuning for the training loader, `set_to_none` gradient handling, activation
checkpointing, DDP/FSDP gradient-bucket and `no_sync` tuning, load balancing across ranks, and
bias removal in conv-then-norm blocks (a weight-layout change that would invalidate the
checkpoints). `torch.no_grad()` is already in place on the inference path (`stage2.py:823`) and
the diffusion module is already `eval().requires_grad_(False)`; `torch.inference_mode()` is a
possible small upgrade but must be checked against the cache tensors being mutated in place.

## 11. Summary of expected outcomes

| Step | Change | Expected gain | Risk | Gate |
| --- | --- | --- | --- | --- |
| — | EMA-only production pass (§1.5) | ~2× wall clock | low, but edits a provenance contract | exact |
| 0 | WS0 measurement harness | none (enables the rest) | none | — |
| 1 | WS1 sync removal | launch-overhead reduction across 130 forwards | low | exact |
| 2 | WS3 streaming decode (verified bit-exact) + buffer reuse + weight offload during decode | ~0.3–1 GiB from decode, up to ~10 GiB from the offload, horizon-independent peak | low | exact |
| 3 | WS7 pipeline decode against diffusion | throughput 3.3 → 6.3 generated fps; no latency change | medium | exact |
| 4 | WS5 `torch.compile` → `reduce-overhead` → `max-autotune` | largest potential | medium | PSNR |
| 5 | WS6 XPU graphs | launch overhead only | high | PSNR |
| 6 | WS4 encoder CPU bf16 + bf16 construction | −10.6 GiB host resident, −6.6 GiB load spike, encode 6.8 s → 5.1 s | low | PSNR |
| 7 | WS2 fp32 matmul precision | small | low | PSNR |
| — | WS8 chunk size 1 | latency 3.24 s → 1.56 s, but −40 % throughput and out-of-distribution | high | quality experiment |

Re-measure peak device memory at every step: steps 3 and 4 both consume the headroom step 2
creates, and step 5's GPU variant competes for the same budget.

## 12. Implementation status

| Item | Status |
| --- | --- |
| WS1 sync removal — precompute `step_values` once, single device-side finite AND per rollout instead of a `.item()` sync per chunk | **Done** in `_stage2_self_forcing_latents` |
| WS1 sync removal — KV-cache `global_end_index`/`local_end_index` as Python ints (`causal_model.py`) | **Not done.** Highest remaining sync count, but touches the roll/direct-insert cache-update paths, `cache_update_info`, and both `forward()` overloads; needs its own focused pass and exact-match validation, not bundled here. |
| WS1 sync removal — `frame_seqlen = math.prod(grid_sizes[0][1:]).item()` | **Not done.** Requires threading a precomputed value through the forward signature; deferred with WS1's KV-cache item. |
| WS3.1 streaming decode as the unconditional production path | **Done** — `_stage2_generated_sample` now always calls `vae.decode_streaming`; the single-shot `vae.decode(use_cache=False)` branch is removed. |
| WS3.3 remove the `initial_noise[:, start:end].clone()` per chunk | **Done.** |
| WS3 — pinned-buffer diffusion-weight offload during decode, noise-buffer reuse, max-shape preallocation | **Not done.** Offload needs new pinned-buffer plumbing; noise-buffer reuse and preallocation are mainly preconditions for WS6 (not attempted this pass). |
| WS8 configurable chunk size | **Done, plumbing only.** `_stage2_self_forcing_latents` accepts any `num_frame_per_block` that evenly divides the rollout horizon; the shipped configs are unchanged at 3. **No quality validation has been run at other block sizes** — do not change the default without the experiment described in §8.6. |
| WS7 pipeline diffusion against decode | **Not done.** Needs the KV-cache sync removal above as a precondition (a host sync anywhere in the captured region collapses the overlap) plus new stream/event plumbing. |
| WS5 `torch.compile`, WS6 XPU graphs, WS4 encoder placement, WS2 matmul precision, WS0 harness | **Not done.** All PSNR-gated or require new infrastructure; not attempted this pass. |
| §1.5 EMA-only production pass / camera-length route | **Not done — needs sign-off.** Either option edits a deliberate provenance contract (`sgf.py`); left for an explicit decision rather than a quiet patch. |

**Verified this pass:** full test suite (769 passed, 1 pre-existing CUDA-only skip); a bug this
work exposed — see below; and an end-to-end 81f smoke run whose `ema` MP4 is **byte-identical**
to the recorded baseline (`md5 2985664b344b975588f70020668ec66e`), so WS1 and WS3.1/3.3 hold the
exact-match gate.

**Bug found and fixed along the way (not part of the plan, but blocking WS3.1):** the VAE-decode
finiteness gate compared a float32 `torch.isfinite(decoded).float().mean()` against exactly
`1.0`. Streaming decode concatenates per-tile results before this reduction, which changes the
summation order relative to a single-shot decode; on the production tensor size this rounds to
`1.0 + 1 ulp` even though every pixel is finite, and the run failed with
`finite_fraction=1.0000001192092896`. Fixed by gating on the exact boolean reduction
(`torch.isfinite(decoded).all()`) and only computing the float mean for the error message when
that reduction is false. This also means the pre-existing check was already fragile — a small
number of non-finite pixels among ~4×10⁷ elements could round-trip through the mean similarly
and go undetected — so the fix is a correctness improvement independent of WS3.
