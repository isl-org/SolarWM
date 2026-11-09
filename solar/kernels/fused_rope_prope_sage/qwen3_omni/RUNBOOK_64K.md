# Runbook: Qwen3-Omni 64K RULER sweep, SDPA vs SageAttention (Arc Pro B70)

Start a new session with: *"Read /home/gta/omni_bench/RUNBOOK_64K.md and run the 64K sweep."*

## Goal (about 4 h of machine time for both arms)

Measure SDPA vs Sage at the 64K bucket (prompts of 62.5–63.5K tokens) on RULER with Qwen3-Omni-30B-A3B. The metrics are:

- TTFT
- Peak VRAM during prefill
- Prefill and attention TFLOPS, and bandwidth utilisation
- End-to-end latency
- RULER accuracy

## Current state

- **Kernel:** causal SageAttention, source `kernel/sage_causal.sycl`, built as `kernel/build/sage_causal.so`.
  - Smoothed **INT8** Q·Kᵀ on DPAS: K centred by its (sampled) mean, Q and K quantised per token to INT8, bf16 P·V with fp32 accumulation, online softmax with lazy rescaling. This is the SageAttention2 8-bit recipe (K smoothing only).
  - Causal masking (bottom-right aligned) and native GQA. A shared per-key bias (0 / −inf) masks the padded KV tail.
  - Q·Kᵀ runs one 16-key block per step: two blocks per step spill registers (about 5% slower).
  - `kernel/test_sage_causal.py` passes (56 tests).
- **Why INT8 without Q smoothing** (real activations captured at 64K, `scripts/emulate_qk_quant.py`; attention-output cosine vs exact fp32):

  | Q·Kᵀ scheme | thinker layer 3 | other layers | RULER, 4 weak tasks (60 prompts) |
  |---|---|---|---|
  | INT4, Q+K smoothing | 0.61 | 0.977–0.994 | 45.8 (SDPA 64.8) |
  | INT8, Q+K smoothing | 0.997 | ≥ 0.9998 | 65.7 |
  | **INT8, K smoothing (current)** | **0.992–0.994** | **≥ 0.9992** | **64.2** |

  Q smoothing on INT8 needs a dS GEMM and a [Hq, S/256, S] fp32 buffer (1.9 GiB at 63K) for no measurable RULER gain (−1.5 ± 4.7 points without it).
- **Kernel speed** (`logs/bench_sage_causal.log`; bf16, Hq=32, Hkv=4, D=128, causal; full call including quantisation):

  | S | SDPA | Sage | speedup |
  |---|---|---|---|
  | 8,192 | 6.37 ms | 4.61 ms | 1.38× |
  | 16,384 | 27.5 ms | 17.6 ms | 1.56× |
  | 32,768 | 108.7 ms | 68.9 ms | 1.58× |
  | 63,488 | 412 ms | 254 ms | 1.62× |

  Extra device memory per call at 63K: 0.78 GiB including the 0.48 GiB output (SDPA: 0.48 GiB).
- **HF integration:** `scripts/sage_attention.py` registers the `sage_xpu` attention backend.
  - Thinker prefill (causal, 32/4 heads, D=128) and the audio encoder (D=64) run on Sage.
  - The vision encoder (D=72) and decode steps (q_len=1) use SDPA.
- **End-to-end at 64K** (current kernel, 12 prompts, one per RULER task; `logs/ruler_e2e_final.jsonl`):

  | arm | TTFT | thinker attention | E2E | peak prefill VRAM |
  |---|---|---|---|---|
  | sdpa | 29.8 s | 19.4 s | 32.3 s | 26.9 GiB |
  | **sage** | **21.8 s (1.37×)** | **11.3 s (1.72×)** | **24.4 s (1.32×)** | **26.9 GiB** |

  Decode is on SDPA in both arms (235 ms/token), so the E2E gain comes from prefill.
- **Speedup vs context length** (current kernel, 12 prompts per length; `logs/ruler_e2e_short.jsonl`, `logs/ruler_e2e_final.jsonl`):

  | context | TTFT SDPA → Sage | TTFT speedup | thinker attention SDPA → Sage | attention speedup | E2E speedup | attention share of SDPA TTFT |
  |---|---|---|---|---|---|---|
  | 8K | 3.36 → 3.24 s | 1.04× | 0.32 → 0.21 s | 1.57× | 1.01× | 10% |
  | 16K | 5.00 → 4.54 s | 1.10× | 1.17 → 0.73 s | 1.60× | 1.07× | 24% |
  | 32K | 10.59 → 8.53 s | 1.24× | 5.00 → 2.92 s | 1.71× | 1.18× | 47% |
  | 64K | 29.79 → 21.80 s | **1.37×** | 19.40 → 11.25 s | **1.72×** | **1.32×** | 65% |

  Attention itself is 1.57–1.72× faster at every length; the end-to-end gain grows with context because attention's share of prefill grows (the INT4 MoE layers dominate below about 16K). Peak prefill VRAM is identical in both arms at every length.
- **RULER accuracy at 64K** (12 tasks × 15 samples; `logs/summary_sweep64k_final.md`, `logs/ruler_sweep64k_final.jsonl`): SDPA 68.8, INT8 with Q smoothing 68.9 (+0.1 ± 2.4 paired), INT4 61.4 (−7.4 ± 4.5). The current kernel (INT8, K smoothing only) was checked on the four tasks most sensitive to Q·Kᵀ error (cwe, qa_2, niah_multivalue, niah_multiquery; `logs/ruler_sweep64k_noqsm.jsonl`): 64.2 vs SDPA 64.8 (−0.7 ± 5.4). A full 12-task sweep of the current kernel has not been run.
  - Generation is not bit-reproducible run to run even on the SDPA arm (a few predictions change between runs), so compare arms with paired statistics over many prompts.
- **Real-activation accuracy** (`logs/accuracy_probe.json`, 3 OmniBench prompts + 8K/16K/32K text): top-1 next token matched SDPA on all 6 prompts; KL ≤ 0.004; thinker attention cosine vs SDPA: mean ≥ 0.999 over the 48 layers, worst layer ≥ 0.993.

## Machine constraints

These caused crashes before. Do not skip them.

- 30 GB host RAM and 32 GB VRAM. **When VRAM overflows, the Xe driver spills it into host RAM, and the global OOM killer has killed the Claude session itself.** The following safeguards handle this:
  - Always launch GPU jobs through `scripts/rss_guard.sh LIMIT_GB LOG -- cmd`. It kills the job when system free RAM drops below 4 GB, and writes `LOG.mem.csv`.
  - Always use `./env.sh`. It sets `LD_LIBRARY_PATH` for MKL and turns off async weight loading.
  - Run only one GPU job at a time. A model run uses about 28 GB of VRAM, so a benchmark started next to it can overflow.
  - Do **not** set `PYTORCH_ALLOC_CONF=expandable_segments:True`. It made TTFT 2.4× slower and decode 5× slower.
- `scripts/model_loader.py` applies these fixes automatically. All of them are identical in both arms.
  - Remaps the INT4 AutoRound checkpoint to transformers 5 module names.
  - Uses the `auto_round:tritonv2_zp` backend. The ARK XPU kernel uses AVX-512 and SIGILLs on this CPU.
  - Caps torch at 94% of VRAM.
  - Packs small weight tensors. This recovers 4 GiB of allocator slack; without it, 64K OOMs.
  - Projects only the last-token logits.
  - Uses a lean MoE forward: exact, about 1.4× faster, and no 6 GiB buffers.
  - Calls `empty_cache` per layer above 16K tokens.
- `bench_ruler.py` passes explicit EOS ids. The checkpoint's `generation_config` has none, so without them generation never stops.

## Step 0: pre-flight (about 5 min)

```bash
cd /home/gta/omni_bench
./env.sh python -c "import torch;print(torch.xpu.get_device_name(0), torch.xpu.mem_get_info())"   # GPU idle?
free -g                                                                              # ~28 GB available?
pgrep -af "python|bench" | grep -v pgrep                                             # no other GPU job
./env.sh python -m pytest -q kernel/test_sage_causal.py                              # 56 passed
```

If `kernel/sage_causal.sycl` has changed since `kernel/build/sage_causal.so` was built, rebuild it first:

```bash
cd kernel && CUTLASS_DIR=/home/gta/sycl_tla_opt/third_party/sycl-tla-v0.8 PY=/home/gta/omni_env/bin/python \
    bash build.sh build sage_causal.sycl sage_causal && cd ..
```

## Step 1: smoke test at 64K (about 3 min)

```bash
scripts/rss_guard.sh 24 logs/smoke64k.log -- ./env.sh python scripts/bench_ruler.py \
    --lengths 63488 --per-task 1 --tasks niah_single_1 --arms sdpa sage --tag smoke64k
grep -E "defragment|^L=|OutOfMemory|rss_guard" logs/smoke64k.log
```

What to expect:

- The `[defragment]` line reports a slack of about 0.26 GiB.
- TTFT is about 30 s for sdpa and about 21.8 s for sage; thinker attention about 19.5 s vs 11.3 s.
- Rows already present under a tag are skipped; use a fresh `--tag` to re-run the smoke test.
- `score=1.00`.

## Step 2: the sweep (runs in the background and survives the session)

Recommended: SDPA vs Sage, 15 samples per task, 12 tasks = 180 prompts × 2 arms. At about 70–80 s per prompt (both arms plus decode), that is about 3.5–4 h.

```bash
cd /home/gta/omni_bench
nohup scripts/rss_guard.sh 24 logs/sweep64k.log -- ./env.sh python scripts/bench_ruler.py \
    --lengths 63488 --per-task 15 --arms sdpa sage --tag sweep64k > /dev/null 2>&1 &
```

- **Order:** samples are interleaved across tasks (index 0 of all 12 tasks, then index 1, …), and the arm order alternates per sample. A run that is stopped early is still balanced.
- **Resume:** re-run the same command with the same `--tag`. Rows already in `logs/ruler_sweep64k.jsonl` are skipped. To extend, raise `--per-task` (max 20).
- **Progress:** `grep -c '^L=' logs/sweep64k.log` counts rows (2 per prompt). Use `tail -3 logs/sweep64k.log` for the latest. Check `logs/sweep64k.log.mem.csv` if memory is a concern.
- **Finished** when the log ends with `[rss_guard] exit=0 ...`.
- **Kernel changed after the sweep?** The SDPA rows do not depend on the Sage kernel. Re-run only the Sage arm under a new tag, for example `--arms sage --tag sweep64k_sage2` (about 1.2 h). Use a new tag, because rows already present under a tag are skipped.

## Step 3: summarise

```bash
./env.sh python scripts/summarize_ruler.py sweep64k            # -> logs/summary_sweep64k.{md,csv}
./env.sh python ruler/score.py --help                          # per-task accuracy table (RULER metrics)
```

The summary table is one row per arm with these columns:

- Median TTFT, and its ratio to SDPA
- Attention time, attention share of TTFT, and ratio to SDPA
- Attention TFLOPS and GB/s, with utilisation against the measured peaks in `logs/device_peaks.json` (bf16 GEMM 154 TFLOPS, 812 GB/s)
- Prefill TFLOPS and GB/s
- Peak prefill VRAM, total and activations only
- E2E, TPOT, and RULER score

Metric definitions are in the docstrings of `scripts/bench_ruler.py` and `scripts/summarize_ruler.py`.

## Working on the kernel (only if a new candidate is to be evaluated)

Build every candidate to its own directory; never overwrite `kernel/build/` until the candidate has passed all gates. All commands run from `/home/gta/omni_bench`.

```bash
export CUTLASS_DIR=/home/gta/sycl_tla_opt/third_party/sycl-tla-v0.8 PY=/home/gta/omni_env/bin/python
# 1. Build: once as sage_causal (local tools) and once as sage_kernel (KernelFoundry gate)
(cd kernel && bash build.sh build_cand cand.sycl sage_causal && bash build.sh build_cand_kf cand.sycl sage_kernel)
# 2. Local correctness (56 tests); SAGE_BUILD_DIR selects the build
SAGE_BUILD_DIR=$PWD/kernel/build_cand ./env.sh python -m pytest -q kernel/test_sage_causal.py
# 3. KernelFoundry task gates: correctness vs reference, accuracy not worse, exact mask, GQA, peak memory
kernel/kf_gate.sh kernel/build_cand_kf/sage_kernel.so kf_cand_tests          # expect 32 passed
# 4. Speed: interleaved A/B, each variant in its own process
scripts/rss_guard.sh 8 logs/ab_cand.log -- ./env.sh python kernel/ab_kernels.py \
    --variants cur=$PWD/kernel/build/sage_causal.so cand=$PWD/kernel/build_cand/sage_causal.so \
    --rounds 3 --iters 10 --json logs/ab_cand.json
# 5. Real-activation accuracy (loads the model; overwrites logs/accuracy_probe.json, so copy it away first)
SAGE_BUILD_DIR=$PWD/kernel/build_cand scripts/rss_guard.sh 24 logs/accuracy_probe_cand.log -- \
    ./env.sh python scripts/accuracy_probe.py 8192 16384 32768
```

- **Adoption gate:** at least 3% faster at 63K in step 4 (or a measured accuracy gain), all gates in steps 2–3 green, and in step 5 a top-1 match on all prompts with thinker layer 3 cosine no lower than the values above. Then copy the candidate to `kernel/sage_causal.sycl`, rebuild `kernel/build/`, and update the "Current state" numbers.
- **Kernel names must be unique per build.** The source derives an outer namespace from the extension name (`SAGE_NS`). Keep that: two builds loaded into one process (the KernelFoundry gate loads candidate and reference together) otherwise share SYCL kernel names, and the runtime may run the wrong binary or hang.
- **KernelFoundry-reported speedups:** KernelFoundry times its reference once per job, or reuses a cached result, rather than next to each candidate. Always confirm a reported speedup with step 4 before using a kernel.
- **Profiling:** `unitrace` does not see the kernels of this PyTorch build (no per-kernel metrics, even with `dev.xe.observation_paranoid=0`). Use the compiler's assembly instead:

  ```bash
  (cd kernel && IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/tmp/isa_cand bash build.sh /tmp/build_isa cand.sycl sage_causal)
  python3 kernel/isa_hotpath.py /tmp/isa_cand    # instruction mix of the attention KV loop, always-executed vs conditional
  ```

  The KV loop is limited by vector (XVE) instructions, not by XMX or memory. Every instruction added to the always-executed path costs about 0.2 ms at 63K.
- `kernel/profile_stages.py 63488` gives per-stage time (quantize, dS GEMM, attention) via `torch.profiler`.
- **Quantisation-scheme emulation on real activations:** `scripts/capture_qkv.py` saves post-RoPE Q/K/V of selected thinker layers during a 64K prefill (about 3.6 GB per prompt, to `act_capture/`); `scripts/emulate_qk_quant.py` compares INT4/INT8 schemes and the real kernel (`SAGE_BUILD_DIR`) against exact fp32 per layer. Use it before building a new numerics variant.
- **KernelFoundry gate:** `/home/gta/kf_task_run` holds `task.py` and the current kernel built as `sage_reference.so` (from `reference.sycl`, a copy of `kernel/sage_causal.sycl`); the installed kernel passes all 32 tests against itself. After adopting a new kernel, copy it to `reference.sycl` and rebuild `sage_reference.so` there so later candidates are compared with it.

## If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `[rss_guard] system MemAvailable < 4 GB` | The driver spilled VRAM into host RAM. Check that nothing else is using the GPU or RAM, and that `env.sh` was used. |
| `XPU out of memory` at 64K | Look at the `[defragment]` line. If slack is far above 0.3 GiB, the packing did not run. Check that `_defragment` is called in `model_loader.load`. |
| `Illegal instruction` while loading | The ARK XPU backend got selected. `model_loader` pins `auto_round:tritonv2_zp`; keep it pinned. |
| No layers quantised or about 58 GB allocated | The module-name remap in `model_loader` is missing (transformers upgrade?). |
| Generation always hits `max_new_tokens` | The EOS ids are missing from `generate()`. |
| Session died mid-sweep | The job keeps running under `nohup`. Check `pgrep -af bench_ruler` and the log. Re-launch the same command to resume. |
| A test hangs on its first case, or `UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX` | Two builds with the same kernel names are loaded in one process. Check that the candidate keeps the `SAGE_NS` namespace. |

## File map (`/home/gta/omni_bench`)

- `env.sh` (run wrapper) and `RUNBOOK_64K.md` (this file).
- `kernel/`:
  - `sage_causal.sycl`: the kernel; `build/sage_causal.so` is its build.
  - `sage_orig.sycl` (`build/sage_orig.so`): the original non-causal SolarWM kernel, used as a reference by the tests.
  - `build.sh`: `bash build.sh OUT_DIR SOURCE MODULE_NAME` (needs `CUTLASS_DIR` and `PY`, see above).
  - `test_sage_causal.py`: 56 correctness tests (`SAGE_BUILD_DIR` selects the build).
  - `bench_sage_causal.py`: Sage vs SDPA speed and memory.
  - `ab_kernels.py`: interleaved A/B between builds, one process per variant.
  - `kf_gate.sh`: KernelFoundry task gates for a build named `sage_kernel` (uses `/home/gta/kf_task_run`).
  - `profile_stages.py`, `isa_hotpath.py`: per-stage time and KV-loop instruction mix.
  - `dump_outputs.py`, `peak_memory.py`: outputs on fixed inputs (bitwise comparison of two builds) and peak memory per call.
  - `flash_causal.sycl` (`build_flash/`): bf16 flash attention baseline with bottom-right causal mask (Qwen3-VL long-context work).
- `scripts/`:
  - `model_loader.py`, `sage_attention.py` (`SAGE_BUILD_DIR` selects the build), `bench_ruler.py`, `summarize_ruler.py`.
  - `accuracy_probe.py`, `device_peaks.py`, `rss_guard.sh`, `capture_qkv.py`, `emulate_qk_quant.py`.
  - `omnibench_data.py` and `smoke_test.py` (OmniBench).
- `ruler/`:
  - `data/ruler_{2048,8192,16384,32768,63488}.jsonl`: 12 tasks × 20 samples each, lengths fitted with the Qwen chat template.
  - `score.py`, `generate.py`, `README.md`.
- `logs/`: all results. The 64K results are `ruler_sweep64k_final.jsonl` / `summary_sweep64k_final.md` (RULER), `ruler_sweep64k_noqsm.jsonl` and `ruler_e2e_final.jsonl` (current kernel); device peaks are in `device_peaks.json`.
- Elsewhere:
  - Python env: `~/omni_env` (torch 2.14.1+xpu, transformers 5.22.1, auto-round 0.16.1).
  - Model: `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound` (HF cache).
  - KernelFoundry task package: `/home/gta/kf_sage_omni`; KernelFoundry source: `/home/gta/kernelfoundry_private`.
  - KernelFoundry gate workspace: `/home/gta/kf_task_run` (task files plus the reference built as `sage_reference.so`).

## Later

- Run the remaining lengths (2K–32K) with the same harness: `--lengths 2048 8192 16384 32768`.
- Video-MME, long split with audio (`lmms-eval/Video-MME`, 101 GB; check disk space first). It needs a video data loader.
- Qwen3-VL-8B (dense) uses the same backend and harness: it needs a new loader and RULER data regenerated with its tokenizer. 128K needs INT4 weights plus chunked prefill. 256K additionally needs an INT8 KV cache, and Sage's dS buffer grows with S²/256 (34 GB for a full 256K call).
