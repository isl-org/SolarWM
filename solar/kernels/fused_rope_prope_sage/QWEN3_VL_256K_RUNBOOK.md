# Qwen3-VL-8B-Instruct at 256K context on one Arc Pro B70: runbook

Use this to run, reproduce or extend the long-context Qwen3-VL work. Start a new session in this directory with:
*"Read QWEN3_VL_256K_RUNBOOK.md and continue the Qwen3-VL 256K work."*
Last updated 2026-11-06. Companion runbook for the Qwen3-Omni work: `QWEN3_OMNI_RUNBOOK.md`.

## 1. Summary

[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) runs its full native 262,144-token context on a single Arc Pro B70 (32 GB VRAM) with:

- INT4 weights (W4A16, group 128) for the text decoder; the vision tower and `lm_head` stay bf16;
- an INT8 KV cache with one scale per (token, KV head);
- chunked prefill (4,096-token chunks) with `torch.xpu.empty_cache()` after each chunk;
- two attention arms for prefill: **Sage** (`sage_causal.sycl`, smoothed INT8 Q·Kᵀ) and a **bf16 flash** baseline (`flash_causal.sycl`). Both use the same sycl-tla work-group tiling.

Needle-in-a-haystack ramp, one prompt per length and arm (essay haystack, needle at 10% depth; `qwen3_vl/results/vl_ramp_int8.jsonl`, 256K rows with the per-chunk `empty_cache` in `vl_ramp_memfix.jsonl`):

| context | arm | TTFT | prefill attention | peak VRAM allocated / reserved | decode / token | needle |
|---|---|---|---|---|---|---|
| 64K | Sage | 21.9 s | 9.9 s | 12.1 / 12.3 GiB | 186 ms | found |
| 64K | flash | 24.2 s | 11.9 s | 12.1 / 12.3 GiB | 185 ms | found |
| 128K | Sage | 65.4 s | 39.7 s | 16.9 / 18.1 GiB | 302 ms | found |
| 128K | flash | 74.4 s | 47.7 s | 16.9 / 17.2 GiB | 302 ms | found |
| 256K | **Sage** | **216.4 s (1.16×)** | **158.9 s (1.20×)** | 26.5 / 27.0 GiB | 537 ms | found |
| 256K | flash | 250.4 s | 190.5 s | 26.5 / 26.9 GiB | 534 ms | found |

- 256K fits with about 3 GiB of VRAM headroom in both arms.
- Sage is about 1.20× faster on prefill attention at every length and 1.11–1.16× on TTFT. The bf16 flash baseline is strong here (about 107 TFLOPS on a 4K × 256K chunk), so the gain is smaller than on Qwen3-Omni, where the baseline was torch SDPA (about 80 TFLOPS).
- The INT4 linear layers take about 58 s of TTFT at 256K in both arms.
- Throughput planning: one 256K prompt through both arms takes about 8–9 minutes including decode; one 128K prompt about 3 minutes.
- **RULER at 128K** (cwe, qa_2, niah_multivalue, niah_multiquery; 10 samples each, 127–129K tokens; `qwen3_vl/results/ruler_vl_rvl128k.jsonl`):

  | task | flash | Sage | Sage − flash |
  |---|---|---|---|
  | cwe | 6.0 | 6.0 | 0.0 |
  | qa_2 | 20.0 | 20.0 | 0.0 |
  | niah_multivalue | 97.5 | 97.5 | 0.0 |
  | niah_multiquery | 85.0 | 87.5 | +2.5 |
  | **all (40 paired prompts)** | **52.1** | **52.8** | **+0.6 ± 2.6** |

  No measurable accuracy loss. Median per-prompt speedups: TTFT 71.6 → 63.2 s (1.13×), prefill attention 45.5 → 37.9 s (1.20×), E2E 92.8 → 84.2 s (1.11×); peak VRAM identical (16.7 GiB). cwe and qa_2 are low in both arms: this INT4-weight model is weak at aggregation and multi-hop QA at 128K regardless of the attention kernel.
- **RULER at 256K** (same four tasks, 3 samples each, 256–259K tokens; `qwen3_vl/results/ruler_vl_rvl256k.jsonl`):

  | task | flash | Sage | Sage − flash |
  |---|---|---|---|
  | cwe | 0.0 | 0.0 | 0.0 |
  | qa_2 | 0.0 | 0.0 | 0.0 |
  | niah_multivalue | 41.7 | 41.7 | 0.0 |
  | niah_multiquery | 50.0 | 66.7 | +16.7 |
  | **all (12 paired prompts)** | **22.9** | **27.1** | **+4.2 ± 8.2** |

  No measurable accuracy loss (12 prompts: indicative, not conclusive). Median per-prompt speedups: TTFT 245.8 → 211.8 s (1.16×), prefill attention 186.6 → 155.0 s (1.21×), E2E 297.2 → 265.5 s (1.13×); peak VRAM 26.35 GiB allocated in both arms, 26.6 GiB reserved. The model itself degrades strongly at 256K in both arms (multi-needle 42–67 vs 85–98 at 128K; cwe and qa_2 at 0).

## 2. Model facts and memory budget

Text decoder: 36 layers, hidden 4096, MLP 12288, 32 Q heads / 8 KV heads, head_dim 128, vocab 151,936, untied `lm_head`, `max_position_embeddings` 262,144, interleaved MRoPE (text-only prompts reduce to 1-D positions). The bf16 checkpoint is 16.3 GiB.

KV cache per token: 36 layers × 2 × 8 heads × 128 × 2 bytes = 144 KiB in bf16, which is 36 GiB at 256K. Usable VRAM is about 30 GiB (94% cap).

| item at 256K | size |
|---|---|
| weights: INT4 text decoder + bf16 embeddings, `lm_head`, vision | 6.75 GiB |
| INT8 KV cache + per-(token, head) bf16 scales | 18.3 GiB |
| bf16 dequant buffers for one layer's K and V (shared by all layers) | 1.0 GiB |
| per-call attention transients (Q8 / K8 / scales for Sage, workspace for flash) and activations | about 0.5 GiB |
| **peak allocated (both arms)** | **26.5 GiB** |

Combinations that do not fit: bf16 weights + INT8 KV (about 37 GiB), INT4 weights + bf16 KV (about 46 GiB).

Sage's per-call buffers grow with the key count every chunk; without the per-chunk `empty_cache` the caching allocator strands the previous chunk's blocks and reserved memory reaches the 30 GiB cap at 256K (29.9 GiB reserved for 26.5 GiB allocated). With it, reserved stays at 27.0 GiB and TTFT is unchanged.

## 3. Attention for chunked prefill on XPU

A prefill chunk of Sq queries attends to Skv ≥ Sq keys with a bottom-right aligned causal mask (query i sees keys j ≤ i + Skv − Sq).

- **torch SDPA cannot do this efficiently on XPU:**
  - With an explicit mask (what transformers builds for a chunk), SDPA falls back to a materialising path: at 4,096 × 262,144 it tries to allocate 192 GiB.
  - `is_causal=True` is top-left aligned when Sq ≠ Skv, which is the wrong mask for a chunk.
  - The fast kernel behind `F.scaled_dot_product_attention` (86 TFLOPS) never returns the log-sum-exp, so "chunk vs past" and "chunk vs itself" cannot be merged. `aten._scaled_dot_product_flash_attention` does return it but runs at about 18 TFLOPS.
- **`flash_causal.sycl` is the bf16 baseline.** It wraps sycl-tla v0.8's `XeFMHAFwdKernel` (XeDefault mainloop, `CausalMask=true`), which has native GQA and the bottom-right mask, with the same tiling as `sage_causal.sycl`.
  - One 4,096-token chunk against 262,144 keys: 164 ms, 107 TFLOPS. Torch's fast kernel without a mask takes 200 ms.
  - **Constraint:** causal calls need (Skv − Sq) to be a multiple of the key tile (32 at D=128, 64 at D=64). sycl-tla applies the causal mask only on each sub-group's last KV tile, so a misaligned diagonal leaks future keys. The wrapper rejects misaligned calls. Chunked prefill always satisfies the constraint when every chunk except the last is a multiple of the tile.
- **`sage_causal.sycl` handles chunks natively** (bottom-right mask, native GQA); see `QWEN3_OMNI_RUNBOOK.md` section 4 for its design.
- **Decode** (q_len = 1) uses `F.scaled_dot_product_attention(..., enable_gqa=True)` for both arms. The default transformers SDPA path would expand K/V 4× for GQA (2 GiB per layer at 256K).
- **No mask is ever materialised:** the backend registers a mask builder that returns `None`, and the kernels apply the causal mask themselves.

## 4. INT4 Q·Kᵀ on Qwen3-VL (offline analysis)

INT4 Q·Kᵀ would be about 9% faster on prefill attention than INT8 at 256K (the INT4 kernel measured 145.7 s vs INT8's 158.9 s), but on Qwen3-Omni it lost 7.4 RULER points at 64K. Real activations of all 36 layers were captured at 64K (`qwen3_vl/capture_vl_qkv.py`) and every scheme compared with exact fp32 attention (`emulate_vl_sensitivity.py`; `results/vl_sensitivity_64k.json`). Attention-output cosine, per layer mean / worst head:

| scheme | range over the 36 layers | worst layer |
|---|---|---|
| INT8 (current kernel) | 0.998–0.9999 / ≥ 0.994 | L10 0.9982 |
| INT4, K smoothing only | 0.71–0.98 / down to 0.33 | L10 0.712 |
| INT4 + Q smoothing | 0.80–0.99 / down to 0.54 | L0 0.801 |
| INT4 + Hadamard rotation of Q and K | 0.95–0.99 / down to 0.82 | L0 0.946 |
| INT4 + Hadamard + Q smoothing (best INT4) | 0.983–0.997 / down to 0.86 | L0 0.963 |

- No layer gets INT4 to INT8 level, so per-layer mixed precision only changes how much accuracy is traded. On Qwen3-Omni, INT4 at about 0.98–0.99 per layer already cost 7.4 RULER points.
- **Learned rotations do not beat Hadamard.** Per-KV-head orthogonal rotations R = H·exp(A − Aᵀ), trained with straight-through rounding to maximise INT4 attention cosine (`learn_rotation.py`; `results/vl_rotation_qs.json`), scored 0.956 / 0.982 / 0.986 / 0.988 / 0.996 on held-out rows of layers 0 / 5 / 10 / 34 / 17, against Hadamard's 0.962 / 0.984 / 0.989 / 0.990 / 0.997.
- The rotation is exact for Q·Kᵀ when applied after RoPE ((qR)(kR)ᵀ = qkᵀ), so a Hadamard-rotated INT4 kernel is possible; it would need a long-context accuracy check (RULER, section 5.3) before use.

## 5. Code and reproduction

### 5.1 Code in this folder

| file | purpose |
|---|---|
| `sage_causal.sycl` | Sage kernel: smoothed INT8 Q·Kᵀ, bf16 P·V, bottom-right causal, native GQA |
| `flash_causal.sycl` | bf16 flash baseline, same API: `forward(q, k, v, is_causal=False)` on contiguous `[B,S,H,D]` |
| `qwen3_vl/vl_attention.py` | transformers attention backend `vl_xpu` with arms `sage` / `flash` (`set_arm`), XPU-event timing of prefill attention, mask-free registration |
| `qwen3_vl/vl_model.py` | INT4 loader, `Int8KVCache`, `prefill()` (chunked, per-chunk `empty_cache`, last-position logits only), `greedy_decode()` |
| `qwen3_vl/mem_ramp.py` | needle-in-a-haystack ramp: TTFT, attention time, peak VRAM, decode time and needle retrieval per length and arm |
| `qwen3_vl/bench_ruler_vl.py`, `summarize_ruler_vl.py` | RULER at 128K / 256K, flash vs Sage per prompt (resumable by tag), and its summary |
| `qwen3_vl/capture_vl_qkv.py` | captures post-RoPE Q rows, Q block means and full K/V of every layer during a chunked prefill |
| `qwen3_vl/qk_quant_lib.py`, `emulate_vl_sensitivity.py`, `learn_rotation.py` | Q·Kᵀ quantisation schemes (INT4/INT8, Q smoothing, rotations) on captured activations; learned rotations |
| `qwen3_vl/results/` | ramp, sensitivity and rotation results |
| `qwen3_omni/ruler/generate.py` | RULER generator; `--tokenizer` selects the tokenizer and chat template used for the length fit |

Design notes:

- **`Int8KVCache`** preallocates int8 K/V `[1, Smax, Hkv, D]` per layer, plus bf16 scales `[1, Smax, Hkv, 1]` (symmetric absmax / 127 per token and head).
  - On each `update`, the layer quantises the new tokens, then dequantises the filled prefix into one bf16 buffer pair shared by all layers.
  - It returns `[1, Hkv, S, D]` views of `[1, S, Hkv, D]` storage, so the kernels' `.transpose(1, 2).contiguous()` is free.
- **Prefill** calls `model.model(...)` per chunk with `cache_position`. The model computes its MRoPE position deltas on the first chunk and reuses them for later chunks. `lm_head` runs on the last position only.
- **INT4 linears** use auto-round's `auto_round:tritonv2_zp` backend, pinned because `auto_round_kernel_xpu` uses AVX-512 host code and SIGILLs on this CPU. Small INT4 scale tensors are packed per layer to avoid allocator slack.

### 5.2 Setup and ramp

Environment: `~/omni_env` (torch 2.14.1+xpu, transformers 5.22.1, auto-round 0.16.1); sycl-tla v0.8 headers in `/home/gta/sycl_tla_opt/third_party/sycl-tla-v0.8`. The scripts expect the `omni_bench` workspace at `/home/gta/omni_bench` (restore steps in `qwen3_omni/README.md`; the Qwen3-VL scripts go to `omni_bench/vl8b/`). Machine safety rules are in `QWEN3_OMNI_RUNBOOK.md` section 5: run GPU jobs through `rss_guard.sh`, one GPU job at a time, and do not use `expandable_segments`.

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
scripts/rss_guard.sh 24 logs/vl_ramp.log -- ./env.sh python vl8b/mem_ramp.py \
    --lengths 65536 131072 262144 --arms sage flash --json logs/vl_ramp.jsonl
grep -E "^ROW|OOM|rss_guard" logs/vl_ramp.log
```

Environment variables read by the scripts: `QWEN3VL_INT4_DIR` (quantised checkpoint), `SAGE_BUILD_DIR` / `FLASH_BUILD_DIR` (kernel builds, default `omni_bench/kernel/build{,_flash}`), `HAYSTACK_JSONL` (RULER jsonl with `niah_single_2` rows, the essay source for the ramp).

### 5.3 RULER at 128K and 256K

Data: the four tasks most sensitive to Q·Kᵀ error on Qwen3-Omni (cwe, qa_2, niah_multivalue, niah_multiquery), fitted with the Qwen3-VL tokenizer and chat template. Buckets leave 2,048 tokens for generation: 129,024 (10 samples per task, 127–129K tokens) and 260,096 (3 samples per task, 256–259K tokens). cwe at 256K needs RULER's `english_words.json`, a git-lfs object (fetch it into `third_party/RULER/scripts/data/synthetic/json/` and check its SHA-256 against the pointer).

```bash
cd /home/gta/omni_bench/ruler; T=/home/gta/models/qwen3vl8b-w4a16/Qwen3-VL-8B-Instruct-w4g128
../env.sh python generate.py --tokenizer $T --out_dir $PWD/data_vl --lengths 129024 \
    --tasks cwe qa_2 niah_multivalue niah_multiquery --num_samples 10
../env.sh python generate.py --tokenizer $T --out_dir $PWD/data_vl --lengths 260096 \
    --tasks cwe qa_2 niah_multivalue niah_multiquery --num_samples 3
cd ..
# about 2 h (128K, 40 prompts x 2 arms) and 1.8 h (256K, 12 prompts x 2 arms); resumable by --tag
nohup scripts/rss_guard.sh 24 logs/ruler_vl_128k.log -- ./env.sh python vl8b/bench_ruler_vl.py \
    --lengths 129024 --per-task 10 --tag rvl128k > /dev/null 2>&1 &
# then: --lengths 260096 --per-task 3 --tag rvl256k  (log ruler_vl_256k.log)
python3 vl8b/summarize_ruler_vl.py rvl128k rvl256k     # per-task scores, paired diff + CI, speedups, VRAM
```

Each prompt runs through both arms (order alternating per sample; samples interleaved across tasks); rows go to `logs/ruler_vl_<tag>.jsonl` with TTFT, prefill attention time, E2E, decode time, peak VRAM, prediction and RULER score.

## 6. Known limitations

- **Text only, batch 1.** The `vl_xpu` backend asserts batch 1 and no attention mask. Image/video inputs would also need the vision tower to run with its own attention (it is configured as `sdpa`), and MRoPE positions for the mixed sequence.
- **Decode is slow at long context:** about 535 ms/token at 256K, because every layer dequantises the whole INT8 cache on every token. This is acceptable for short answers (RULER); an attention kernel that reads INT8 K/V directly would remove the dequantisation.
- **The flash baseline needs a tile-aligned diagonal offset** (section 3).
- **Generation is not bit-reproducible run to run** (seen on Qwen3-Omni even for the SDPA arm), so compare arms with paired statistics.

## 7. Next steps

1. **More 256K samples** if the 256K accuracy comparison needs to be conclusive (3 per task now; each prompt takes about 9 minutes through both arms).
2. **INT4 + Hadamard + Q smoothing kernel**, only if RULER shows headroom worth chasing: the rotation is exact and can be fused into the quantise stage. Expected gain over INT8 is about 9% of prefill attention (about 5% of TTFT at 256K), and it needs the RULER check above to rule out the accuracy loss INT4 showed on Qwen3-Omni.
3. **Faster decode:** INT8-aware decode attention, or dequantise only the new tokens per layer if a per-layer bf16 cache fits at shorter contexts.
4. **Absolute accuracy:** re-quantise with full auto-round tuning (without `--iters 0`) if RULER scores matter beyond the Sage-vs-flash comparison.
