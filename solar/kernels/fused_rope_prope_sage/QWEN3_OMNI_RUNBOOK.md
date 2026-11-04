# Qwen3-Omni + SageAttention (this kernel) on Arc Pro B70: project runbook

Use this to restart the work from scratch. Start a new session in this directory with:
*"Read QWEN3_OMNI_RUNBOOK.md and continue the Qwen3-Omni Sage benchmark."*
Last updated 2026-11-04.

## 1. What the project is

`sycl_tla_opt.sycl` (this folder) is a SageAttention2 kernel in SYCL: smoothed INT4 Q·Kᵀ on Xe2 DPAS, then FP16/BF16 P·V. The project integrates it into **Qwen3-Omni-30B-A3B-Instruct** and benchmarks it end to end against torch SDPA. The metrics are:

- Time to first token (TTFT)
- Peak VRAM during prefill
- Prefill TFLOPS and bandwidth utilisation
- End-to-end latency vs sequence length

**Benchmarks:**

- **RULER** (main context-length sweep: 2K, 8K, 16K, 32K, and "64K" = 63,488 tokens, the model maximum).
- **Video-MME long with audio** (later).
- **OmniBench** (short-context sanity check).

**Decision taken:** causal masking was added to the kernel itself, rather than using Sage only in the non-causal audio encoder.

## 2. Where everything lives

| What | Path |
|---|---|
| Base kernel the work started from (non-causal, Hq == Hkv only) | `/home/gta/omni_bench/kernel/sage_orig.sycl`, the pre-`23b7cac` version of `./sycl_tla_opt.sycl` (`git show 050d413^:src/solarwm/kernels/fused_rope_prope_sage/sycl_tla_opt.sycl`) |
| Newer upstream kernel (staged API + in-place KV-cache packing, still no causal mask) | `./sycl_tla_opt.sycl` on branch `xpu` (commit `23b7cac`) |
| Benchmark workspace: kernel work copy, harness, data, logs | `/home/gta/omni_bench/` (**not version-controlled**) |
| **Session runbook: 64K sweep and kernel-candidate workflow** | `/home/gta/omni_bench/RUNBOOK_64K.md` |
| **Current kernel (causal + GQA)** | `./sage_causal.sycl` (this folder) |
| Benchmark work copy of the kernel, and its build | `/home/gta/omni_bench/kernel/sage_causal.sycl`, `.../kernel/build/sage_causal.so`; keep it identical to `./sage_causal.sycl` |
| Kernel tools: tests, Sage-vs-SDPA bench, interleaved A/B, KernelFoundry gate, KV-loop instruction mix | `/home/gta/omni_bench/kernel/` (`test_sage_causal.py`, `bench_sage_causal.py`, `ab_kernels.py`, `kf_gate.sh`, `isa_hotpath.py`) |
| KernelFoundry task package (optimise the kernel for the Omni shapes) | `/home/gta/kf_sage_omni/` |
| KernelFoundry source, and the gate workspace used by `kf_gate.sh` | `/home/gta/kernelfoundry_private/`, `/home/gta/kf_task_run/` |
| Python env | `/home/gta/omni_env` (torch 2.14.1+xpu, transformers 5.22.1, auto-round 0.16.1) |
| sycl-tla v0.8 headers + build script | `/home/gta/sycl_tla_opt/third_party/sycl-tla-v0.8`, `/home/gta/sycl_tla_opt/build.sh` |
| Model | HF cache: `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound` (25 GiB) |
| OmniBench | HF cache: `m-a-p/OmniBench` |

**If `/home/gta/omni_bench` still exists:** go straight to its `RUNBOOK_64K.md`. **If it is gone:** use section 6.

## 3. Results so far

**Standalone kernel vs SDPA** (thinker shape: bf16, 32 Q heads / 4 KV heads, D=128, causal, called as from HF; full call including quantisation; `omni_bench/logs/bench_sage_causal.log`):

| tokens | 8K | 16K | 32K | 63K |
|---|---|---|---|---|
| SDPA → Sage | 6.38 → 3.81 ms | 27.7 → 15.2 ms | 109.9 → 60.8 ms | 417 → 224 ms |
| speedup | 1.67× | 1.82× | 1.81× | 1.86× |
| Sage TFLOPS (SDPA ≈ 80) | 144 | 145 | 145 | 147 |
| extra VRAM, SDPA → Sage | 64 → 116 MiB | 128 → 295 MiB | 256 → 846 MiB | 0.5 → 2.5 GiB |

At 2K the call is about 0.5 ms and dominated by launch overhead; timings there are noisy.

**Inside Qwen3-Omni** (RULER niah_single_1, 1 sample, final loader settings; `omni_bench/logs/smoke64k.log`):

| tokens | TTFT SDPA → Sage | thinker attention SDPA → Sage | peak prefill VRAM SDPA → Sage | E2E SDPA → Sage | answer |
|---|---|---|---|---|---|
| 63K (62,852) | 29.7 → 20.7 s (1.44×) | 19.5 → 9.9 s (1.96×; 80 vs 157 TFLOPS) | 26.83 → 27.61 GiB | 32.1 → 23.0 s | correct (both) |

At 2–16K the INT4 MoE path dominates prefill, so TTFT gains there are small.

**Accuracy on real activations** (`omni_bench/logs/accuracy_probe.json`, 3 OmniBench prompts + 8K/16K/32K text):

- The top-1 next token matched SDPA on all 6 prompts; KL divergence was 0.0002–0.20.
- Audio encoder: cosine ≥ 0.996.
- Thinker layers: mean cosine about 0.98 over the 48 layers. **Layer 3** is the weakest and degrades with length: 0.94 at 8K, 0.90 at 16K, 0.81 at 32K. The `sage_l3` arm keeps layer 3 on SDPA.

**Device peaks measured** (B70): bf16 GEMM 154 TFLOPS, int8 GEMM 291 TOPS, read 594 GB/s, add (2r+w) 812 GB/s. Memory benchmarks need random data; zero buffers get compressed and read about 2× too fast.

## 4. Kernel design: base `sage_orig.sycl` → `sage_causal.sycl`

API: `forward(q, k, v, is_causal=False)`. Inputs are contiguous `[B,S,H,D]` tensors, fp16 or bf16, D = 64 or 128. Differences from the base kernel:

1. **Causal mask, bottom-right aligned:** query i sees keys j ≤ i + Skv − Sq. Each work-group stops at its diagonal KV tile, and the loop bound is per work-group because of the barrier-synced loop. Heavy Q blocks are launched first. Key 0 is always visible, so the online softmax never sees an all −inf row.
   - The −inf select is a separate, sub-group-uniform branch taken only on tiles that cross the sub-group's diagonal. Folded into the score expression, the compiler predicates it and issues 2 cmp + 1 sel per element on every tile.
2. **dS by GEMM:** `quantize_k` only centres and quantises K (K4, k_scale). The correction term dS[qb, j] = q_mean[qb] · (K_j − k_mean) is one fp32 `at::bmm` per KV head, written straight into the dS layout; padded keys get −inf.
3. **Native GQA:** K4/k_scale are per KV head and dS is per Q head; the attention kernel indexes KV head = h / (Hq/Hkv). There is no `repeat_kv`. In HF prefill, Q/K/V are already physically `[B,S,H,D]`, so the transposes are free views.
4. **Row sums of P** stay on DPAS (P × ones), fed from the P fragment in place.
5. **Build-unique namespace:** the outer namespace is derived from the extension name, so two builds can be loaded in one process (KernelFoundry loads candidate and reference together) without sharing SYCL kernel names.
6. **Numerics:** non-causal output matches the original kernel within 1 bf16 ulp (the dS GEMM changes the fp32 summation order); accuracy against fp32 SDPA is unchanged.

**Performance character:** the KV loop is limited by vector (XVE) instruction issue, not by XMX or memory: about 480 always-executed instructions per 16×32 tile per sub-group, against 44 DPAS. The main remaining items are INT4 dequantisation, softmax, and the P conversion to bf16. `omni_bench/kernel/isa_hotpath.py` prints this mix from an IGC assembly dump (see `RUNBOOK_64K.md`); `unitrace` does not see this PyTorch build's kernels.

**Still open:** the dS buffer is fp32 `[B, Hq, ceil(Sq/256), Skv_pad]`, about 1.9 GiB at 63K, and it grows as S². At 256K it would be about 34 GB. Sharing one Q mean across larger blocks shrinks it, but coarser smoothing lowers real-activation accuracy at thinker layer 3 (a 1024-row block failed the gate).

## 5. Machine constraints

These caused crashes before. Do not skip them.

- 30 GB host RAM, 32 GB VRAM, Core Ultra 5 245K (**no AVX-512**).
- **When VRAM overflows, the Xe driver spills it into host RAM instead of torch raising OOM.** The global OOM killer then fires and has killed the Claude session itself. Always:
  - run GPU jobs through `omni_bench/scripts/rss_guard.sh` (it kills the job when system free RAM drops below 4 GB);
  - call `torch.xpu.set_per_process_memory_fraction(0.94)`;
  - run one GPU job at a time (a model run uses about 28 GB of VRAM).
- `PYTORCH_ALLOC_CONF=expandable_segments:True` works on XPU but made prefill 2.4× slower and decode 5× slower. Don't use it.
- Prebuilt wheels with AVX-512 host code SIGILL. auto-round-lib's `auto_round_kernel_xpu` is one of them.
- Network goes through the Intel proxy (env vars already set). NLTK downloads need `NLTK_ALLOW_PROXIED_URLOPEN=1`.

## 6. Rebuild from scratch (if `omni_bench` or `omni_env` was lost)

### 6.1 Environment

```bash
/usr/bin/python3 -m venv ~/omni_env
~/omni_env/bin/pip install torch==2.14.1+xpu torchvision --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple
~/omni_env/bin/pip install "transformers>=4.57" accelerate qwen-omni-utils datasets soundfile librosa av \
    auto-round auto-round-lib "mkl==2026.1.0" pytest pybind11 ninja psutil huggingface_hub \
    nltk wonderwords tenacity html2text beautifulsoup4
# run wrapper (env.sh): export LD_LIBRARY_PATH=$HOME/omni_env/lib:$LD_LIBRARY_PATH; export HF_DEACTIVATE_ASYNC_LOAD=1
hf download Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound
hf download m-a-p/OmniBench --repo-type dataset
```

### 6.2 Kernel build

The kernel is `./sage_causal.sycl` in this folder (section 4 describes it). Copy it to `omni_bench/kernel/` and build it there as module `sage_causal`. Build every candidate to its own output directory and module name.

```bash
CUTLASS_DIR=/home/gta/sycl_tla_opt/third_party/sycl-tla-v0.8 PY=$HOME/omni_env/bin/python \
  bash /home/gta/sycl_tla_opt/build.sh <out_dir> <source.sycl> <module_name>     # AOT for bmg-g31, ~45 s
```

### 6.3 Model loader: required fixes

Every one of these was needed; they apply identically to both arms.

1. Load with `enable_audio_output=False` (no talker), `dtype=bf16`, `device_map="xpu"`.
2. **Remap the quantisation config:** `block_name_to_quantize` and the `extra_config` regexes say `thinker.model.layers`, but transformers 5 uses `thinker.model.language_model.layers`. Without the remap nothing is quantised, about 58 GB gets allocated, and the GPU spills to host → OOM kill.
3. **Pin the backend** to `qc["backend"] = "auto_round:tritonv2_zp"`. Delete the `triton` requirement from `BackendInfos["auto_round:tritonv2_zp"].requirements`, because the XPU wheel is called `triton-xpu`.
4. Set the VRAM fraction to 0.94 **before** loading. While loading, call `torch.xpu.empty_cache()` whenever reserved − allocated > 1 GiB (wrap `transformers.core_model_loading._materialize_copy`).
5. **Pack small tensors:** for each decoder layer, copy every parameter/buffer under 2 MB into one buffer per dtype (256-byte aligned) and swap in views. The ~0.75 MB expert tensors otherwise waste about 4 GiB of 2 MB allocator segments, and 64K OOMs. A CPU round trip does not help.
6. **`lm_head` on the last position only.** The thinker ignores `logits_to_keep`; full logits at 64K are about 19 GiB.
7. **Lean MoE forward**, replacing auto-round's `experts_forward`: stable-argsort the routing once, take one `bincount().tolist()`, then per expert gather → MLP → × weight → `index_add_` into an fp32 accumulator. It is exact, about 1.4× faster, and avoids the (tokens·top_k, hidden) buffers (about 6 GiB at 63K). Token chunking was worse: chunk 4096 added 7.5 s to TTFT at 32K.
8. Forward pre-hook on each thinker decoder layer: `torch.xpu.empty_cache()` when the sequence is longer than 16K.
9. In `thinker.generate()`, pass `eos_token_id=[<|im_end|> 151645, <|endoftext|> 151643]`. The checkpoint has no EOS, so generation never stops.

### 6.4 Attention backend

Register `sage_xpu` through `transformers.AttentionInterface`, and register `AttentionMaskInterface` with `sdpa_mask` (batch-1 causal prefill then gets mask=None).

**Use Sage only when:**

- q_len ≥ 2
- D ∈ {64, 128}
- the dtype is fp16 or bf16
- the mask is None
- scaling == D^-0.5

Otherwise fall back to `sdpa_attention_forward`. The causal flag comes from `module.is_causal`. Call `forward(q.transpose(1,2).contiguous(), k..., v..., is_causal)`; the result is already `[B,S,H,D]`.

**Extras:**

- A compare mode (run SDPA too and log cosine/relative-L2 per call).
- A per-layer SDPA override.
- XPU-event timing around thinker prefill attention calls.

### 6.5 RULER data and harness

- **RULER:** clone NVIDIA/RULER (commit db43f92). Generate 12 tasks:
  - niah_single_1/2/3, niah_multikey_1/2, niah_multivalue, niah_multiquery
  - vt, cwe, fwe, qa_1, qa_2

  Use 20 samples each at {2048, 8192, 16384, 32768, 63488}. Fit lengths on the **Qwen chat-templated** token count within [0.97L, L]. Score with RULER's `string_match_all` / `string_match_part`. Append `answer_prefix` after the generation prompt.
- **Per (sample, arm) the harness records:**
  - TTFT = time to the first logits-processor call, after a sync
  - peak VRAM at that point
  - attention ms from the XPU events
  - E2E, TPOT, score
  - analytic FLOPs and minimum-traffic bytes (active params/token = 2.718B for the thinker text model)
- **Run conditions:** arms alternate order per sample, with one warm-up per length, and the output is resumable JSONL.

## 7. Next steps

0. **Reconcile with the upstream kernel.** Branch `xpu` updated `./sycl_tla_opt.sycl` (`23b7cac`: SageAttention v2 optimisations with in-place KV-cache packing, a staged `quantize_q` / `pack_k` / `attention` API). Our causal + GQA changes were made on the older version. Either port them onto the new kernel or benchmark the two against each other before the 64K sweep.

1. **64K RULER sweep** (about 3.5–4 h, SDPA vs Sage, 15 per task): follow `/home/gta/omni_bench/RUNBOOK_64K.md`. The 64K smoke test passes with the current kernel (section 3).
2. **Further kernel work**, only behind the gates in `RUNBOOK_64K.md` ("Working on the kernel"): local tests, KernelFoundry task gates, interleaved A/B (≥ 3% at 63K), real-activation accuracy probe.
   - KernelFoundry (`/home/gta/kf_sage_omni`) times its reference once per job or reuses a cached result, so its reported speedups can be inflated. Confirm every result with the local A/B before using it.
   - Since the loop is vector-bound, look for instructions to remove from the always-executed path; moving work from XMX to vector units, or adding vector work per tile, makes it slower.
3. **Remaining lengths** 2K–32K with the same harness.
4. **Video-MME** long split with audio (101 GB download; needs a video loader).
5. **Dense Qwen3-VL-8B-Instruct up to 256K:** see `QWEN3_VL_256K_RUNBOOK.md`. It runs at the full 262,144-token context with INT4 weights, an INT8 KV cache and chunked prefill (TTFT 3.4 min with Sage, 4.2 min with the bf16 flash baseline).
