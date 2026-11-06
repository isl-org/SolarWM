# Qwen3-VL-8B-Instruct at 256K context on one Arc Pro B70: runbook

Use this to run, reproduce or extend the long-context Qwen3-VL work. Start a new session in this directory with:
*"Read QWEN3_VL_256K_RUNBOOK.md and continue the Qwen3-VL 256K work."*
Last updated 2026-11-04. Companion runbook for the Qwen3-Omni work: `QWEN3_OMNI_RUNBOOK.md`.

## 1. Summary

[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) runs its full native 262,144-token context on a single Arc Pro B70 (32 GB VRAM) with:

- INT4 weights (W4A16, group 128) for the text decoder; the vision tower and `lm_head` stay bf16;
- an INT8 KV cache with one scale per (token, KV head);
- chunked prefill (4,096-token chunks), so attention buffers stay bounded;
- two attention arms for prefill: **Sage** (`sage_causal.sycl`, smoothed INT4 Q·Kᵀ) and a **bf16 flash** baseline (`flash_causal.sycl`). Both use the same sycl-tla work-group tiling.

Measured on one needle-in-a-haystack prompt per run (essay haystack, needle at 10% depth). The Sage rows used the **INT4 Q·Kᵀ** version of `sage_causal.sycl`; the current kernel is INT8 (see `QWEN3_OMNI_RUNBOOK.md` section 3: INT4 loses accuracy on long-context aggregation tasks, INT8 does not) and has not been re-measured here yet:

| context | arm | TTFT | prefill attention | peak VRAM allocated / reserved | decode / token | needle |
|---|---|---|---|---|---|---|
| 64K | Sage | 21.3 s | 9.1 s | 12.2 / 14.1 GiB | 185 ms | found |
| 64K | flash | 24.6 s | 12.1 s | 12.1 / 12.3 GiB | 186 ms | found |
| 128K | Sage | 62.3 s | 36.3 s | 17.4 / 25.1 GiB | 303 ms | found |
| 128K | flash | 74.9 s | 48.1 s | 16.9 / 17.2 GiB | 303 ms | found |
| 256K | Sage | **203.8 s** | 145.7 s (139 TFLOPS) | 27.9 / **30.0 GiB** | – | **missed** (first token was end-of-text) |
| 256K | flash | **252.9 s** | 192.7 s (105 TFLOPS) | 26.5 / 26.8 GiB | 539 ms | found |

- 256K is feasible: with the flash arm the model retrieves the needle at the full context, with about 3.5 GiB of VRAM headroom.
- Sage is 1.24–1.33× faster on prefill attention and 1.16–1.24× on TTFT. The INT4 linear layers take about 60 s of TTFT at 256K for both arms.
- **Two open issues with Sage at 256K** (section 7): it missed the needle (one sample, so not yet a measured accuracy loss), and reserved VRAM reached the 30 GiB cap because of allocator fragmentation.
- Throughput planning: one 256K prompt through both arms takes about 8 minutes. A RULER run with 12 tasks × 1 sample at 256K is about 1.6 h; 12 × 2 is about 3.2 h.

## 2. Model facts and memory budget

Text decoder: 36 layers, hidden 4096, MLP 12288, 32 Q heads / 8 KV heads, head_dim 128, vocab 151,936, untied `lm_head`, `max_position_embeddings` 262,144, interleaved MRoPE (text-only prompts reduce to 1-D positions). The bf16 checkpoint is 16.3 GiB.

KV cache per token: 36 layers × 2 × 8 heads × 128 × 2 bytes = 144 KiB in bf16, which is 36 GiB at 256K. Usable VRAM is about 30 GiB (94% cap).

| item at 256K | size |
|---|---|
| weights: INT4 text decoder + bf16 embeddings, `lm_head`, vision | 6.75 GiB |
| INT8 KV cache + per-(token, head) bf16 scales | 18.3 GiB |
| bf16 dequant buffers for one layer's K and V (shared by all layers) | 1.0 GiB |
| per-call transients: flash / Sage (Sage's dS buffer and centred-K GEMM temporaries grow with the key count) | 0.5 / 1.8 GiB |
| **peak allocated: flash / Sage** | **26.5 / 27.9 GiB** |

Combinations that do not fit: bf16 weights + INT8 KV (about 37 GiB), INT4 weights + bf16 KV (about 46 GiB).

## 3. Attention for chunked prefill on XPU

A prefill chunk of Sq queries attends to Skv ≥ Sq keys with a bottom-right aligned causal mask (query i sees keys j ≤ i + Skv − Sq).

- **torch SDPA cannot do this efficiently on XPU:**
  - With an explicit mask (what transformers builds for a chunk), SDPA falls back to a materialising path: at 4,096 × 262,144 it tries to allocate 192 GiB.
  - `is_causal=True` is top-left aligned when Sq ≠ Skv, which is the wrong mask for a chunk.
  - The fast kernel behind `F.scaled_dot_product_attention` (86 TFLOPS) never returns the log-sum-exp, so "chunk vs past" and "chunk vs itself" cannot be merged. `aten._scaled_dot_product_flash_attention` does return it but runs at about 18 TFLOPS.
- **`flash_causal.sycl` is the bf16 baseline.** It wraps sycl-tla v0.8's `XeFMHAFwdKernel` (XeDefault mainloop, `CausalMask=true`), which has native GQA and the bottom-right mask, with the same tiling as `sage_causal.sycl`.
  - One 4,096-token chunk against 262,144 keys: 164 ms, 107 TFLOPS. Torch's fast kernel without a mask takes 200 ms.
  - **Constraint:** causal calls need (Skv − Sq) to be a multiple of the key tile (32 at D=128, 64 at D=64). sycl-tla applies the causal mask only on each sub-group's last KV tile, so a misaligned diagonal leaks future keys. The wrapper rejects misaligned calls. Chunked prefill always satisfies the constraint when every chunk except the last is a multiple of the tile.
- **`sage_causal.sycl` handles chunks natively:** one 4,096-token chunk against 262,144 keys takes 121 ms (144 TFLOPS) with 1.7 GiB of transients.
- **Decode** (q_len = 1) uses `F.scaled_dot_product_attention(..., enable_gqa=True)` for both arms. The default transformers SDPA path would expand K/V 4× for GQA (2 GiB per layer at 256K).
- **No mask is ever materialised:** the backend registers a mask builder that returns `None`, and the kernels apply the causal mask themselves.

## 4. Code in this folder

| file | purpose |
|---|---|
| `sage_causal.sycl` | Sage kernel: smoothed INT8 Q·Kᵀ, bf16 P·V, bottom-right causal, native GQA (`QWEN3_OMNI_RUNBOOK.md` section 4) |
| `flash_causal.sycl` | bf16 flash baseline, same API: `forward(q, k, v, is_causal=False)` on contiguous `[B,S,H,D]` |
| `qwen3_vl/vl_attention.py` | transformers attention backend `vl_xpu` with arms `sage` / `flash` (`set_arm`), XPU-event timing of prefill attention, mask-free registration |
| `qwen3_vl/vl_model.py` | INT4 loader, `Int8KVCache`, `prefill()` (chunked, last-position logits only), `greedy_decode()` |
| `qwen3_vl/mem_ramp.py` | needle-in-a-haystack ramp: TTFT, attention time, peak VRAM, decode time and needle retrieval for each length and arm; writes JSONL |

Design notes:

- **`Int8KVCache`** preallocates int8 K/V `[1, Smax, Hkv, D]` per layer, plus bf16 scales `[1, Smax, Hkv, 1]` (symmetric absmax / 127 per token and head).
  - On each `update`, the layer quantises the new tokens, then dequantises the filled prefix into one bf16 buffer pair shared by all layers.
  - It returns `[1, Hkv, S, D]` views of `[1, S, Hkv, D]` storage, so the kernels' `.transpose(1, 2).contiguous()` is free.
- **Prefill** calls `model.model(...)` per chunk with `cache_position`. The model computes its MRoPE position deltas on the first chunk and reuses them for later chunks. `lm_head` runs on the last position only.
- **INT4 linears** use auto-round's `auto_round:tritonv2_zp` backend, pinned because `auto_round_kernel_xpu` uses AVX-512 host code and SIGILLs on this CPU. Small INT4 scale tensors are packed per layer to avoid allocator slack.

## 5. Reproduce

Environment: `~/omni_env` (torch 2.14.1+xpu, transformers 5.22.1, auto-round 0.16.1); sycl-tla v0.8 headers in `/home/gta/sycl_tla_opt/third_party/sycl-tla-v0.8`. Machine safety rules are in `QWEN3_OMNI_RUNBOOK.md` section 5: run GPU jobs through `rss_guard.sh`, one GPU job at a time, and do not use `expandable_segments`.

```bash
# 1. Weights: download the bf16 model (16.3 GiB), then quantise the text decoder (about 4 min on the B70)
hf download Qwen/Qwen3-VL-8B-Instruct
auto-round --model Qwen/Qwen3-VL-8B-Instruct --scheme W4A16 --iters 0 --format auto_round \
    --device xpu --low_cpu_mem_usage --output_dir /home/gta/models/qwen3vl8b-w4a16
#    -> /home/gta/models/qwen3vl8b-w4a16/Qwen3-VL-8B-Instruct-w4g128 (6.8 GB; auto_round:auto_gptq packing, sym)
#    --iters 0 is auto-round's optimised RTN; drop it for full tuning (slower, better absolute accuracy).

# 2. Kernels (AOT for bmg-g31, about 45 s each); run from this folder; build.sh args: OUT_DIR SOURCE MODULE
export CUTLASS_DIR=/home/gta/sycl_tla_opt/third_party/sycl-tla-v0.8 PY=$HOME/omni_env/bin/python
B=/home/gta/omni_bench/kernel
bash $B/build.sh $B/build       $PWD/sage_causal.sycl  sage_causal
bash $B/build.sh $B/build_flash $PWD/flash_causal.sycl flash_causal

# 3. Ramp (about 20 min for 64K + 128K + 256K, both arms)
cd /home/gta/omni_bench
scripts/rss_guard.sh 24 logs/vl_ramp.log -- ./env.sh python \
    /home/gta/SolarWM/solar/kernels/fused_rope_prope_sage/qwen3_vl/mem_ramp.py \
    --lengths 65536 131072 262144 --arms sage flash --json logs/vl_ramp.jsonl
grep -E "^ROW|OOM|rss_guard" logs/vl_ramp.log
```

Environment variables read by the scripts:

- `QWEN3VL_INT4_DIR`: quantised checkpoint.
- `SAGE_BUILD_DIR`, `FLASH_BUILD_DIR`: kernel builds (default `omni_bench/kernel/build{,_flash}`).
- `HAYSTACK_JSONL`: RULER jsonl with `niah_single_2` rows, used as the essay source.

`mem_ramp.py` options: `--chunk` (default 4096; keep it a multiple of 256), `--max-new` (decode tokens, default 16). Each length/arm run appends one JSON row (`ttft_s`, `attn_s`, `kv_cache_gib`, `peak_prefill_gib`, `peak_reserved_gib`, `tpot_ms`, `answer`, `correct`). An OOM is recorded as a row instead of aborting the ramp.

## 6. Known limitations

- **Text only, batch 1.** The `vl_xpu` backend asserts batch 1 and no attention mask. Image/video inputs would also need the vision tower to run with its own attention (it is configured as `sdpa`), and MRoPE positions for the mixed sequence.
- **Decode is slow at long context:** 539 ms/token at 256K, because every layer dequantises the whole INT8 cache on every token. This is acceptable for short answers (RULER); an attention kernel that reads INT8 K/V directly would remove the dequantisation.
- **The flash baseline needs a tile-aligned diagonal offset** (section 3).
- **Evidence so far is one needle prompt per length and arm**, not a benchmark.

## 7. Next steps

1. **Re-measure the Sage arm with the INT8 kernel** (64K / 128K / 256K ramp, then several needles and depths). The INT4 miss at 256K is consistent with the INT4 accuracy loss measured on Qwen3-Omni; the steps below apply if INT8 still misses.
2. **Sage accuracy at 256K.**
   - Prefill with the flash arm. On the final chunk, also run Sage and compare its attention output with flash per layer (cosine / relative L2).
   - If one or two layers break down at 256K (on Qwen3-Omni, thinker layer 3 degraded with length: cosine 0.94 at 8K to 0.81 at 32K), keep those layers on flash. This costs little speed.
   - Then repeat the needle test at several depths and lengths.
3. **Sage memory at 256K:** the current kernel no longer allocates the per-call dS buffer that grew with the key count (its per-call memory is Q8, K8, scales and output), which should remove most of the fragmentation; if reserved memory still approaches the cap, call `torch.xpu.empty_cache()` once per chunk (or per layer) in the Sage arm. Reserved memory then stays near the 27.9 GiB actually allocated instead of reaching the 30 GiB cap. Alternatively, preallocate Sage's per-call buffers at their 256K size.
4. **RULER at 131,072 and 262,144 tokens:** regenerate the data with the Qwen3-VL tokenizer and chat template, using the same generator as the Omni harness (`QWEN3_OMNI_RUNBOOK.md` section 6.5). Budget 2–8 min per prompt and arm.
5. **Faster decode:** INT8-aware decode attention, or dequantise only the new tokens per layer if a per-layer bf16 cache fits at shorter contexts.
6. **Absolute accuracy:** re-quantise with full auto-round tuning (without `--iters 0`) if RULER scores matter beyond the Sage-vs-flash comparison.
