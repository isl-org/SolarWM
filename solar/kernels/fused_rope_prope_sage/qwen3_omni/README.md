# Qwen3-Omni Sage benchmark harness and results

Harness, kernel tools and results for the Qwen3-Omni-30B-A3B SageAttention work on Arc Pro B70. The kernel itself is `../sage_causal.sycl`; the project overview is `../QWEN3_OMNI_RUNBOOK.md`, and the operational runbook (64K sweep, kernel-candidate workflow) is `RUNBOOK_64K.md` in this folder.

The scripts expect the workspace at `/home/gta/omni_bench` (absolute paths in `env.sh`, `scripts/*` and `kernel_tools/*`) and the Python env at `~/omni_env`.

## Layout

| folder | contents | goes to |
|---|---|---|
| `scripts/` | `bench_ruler.py` (RULER sweep, SDPA vs Sage arms), `summarize_ruler.py`, `model_loader.py` (INT4 AutoRound loader with the memory fixes), `sage_attention.py` (`sage_xpu` attention backend), `accuracy_probe.py` (real-activation accuracy), `capture_qkv.py` + `emulate_qk_quant.py` (Q·Kᵀ quantisation schemes on captured activations), `device_peaks.py`, `omnibench_data.py`, `smoke_test.py`, `rss_guard.sh` (host-RAM guard for every GPU job) | `omni_bench/scripts/` |
| `kernel_tools/` | `build.sh` (AOT build: `OUT_DIR SOURCE MODULE`), `test_sage_causal.py` (56 tests; needs `sage_orig.sycl` built as `sage_orig`), `bench_sage_causal.py` (Sage vs SDPA), `ab_kernels.py` (interleaved A/B, one process per build), `kf_gate.sh` (KernelFoundry task gates), `isa_hotpath.py` (KV-loop instruction mix from an IGC dump), `profile_stages.py`, `dump_outputs.py`, `peak_memory.py` | `omni_bench/kernel/` |
| `ruler/` | RULER data generator, scorer and validator (`README.md` has the steps) | `omni_bench/ruler/` |
| `results/` | key results (below) | `omni_bench/logs/` |
| `env.sh`, `RUNBOOK_64K.md` | run wrapper, runbook | `omni_bench/` |

## Restore a working copy

```bash
B=/home/gta/omni_bench; H=/home/gta/SolarWM/solar/kernels/fused_rope_prope_sage
mkdir -p $B/{scripts,kernel,ruler,logs}
cp $H/qwen3_omni/scripts/* $B/scripts/;  cp $H/qwen3_omni/kernel_tools/* $B/kernel/
cp $H/qwen3_omni/ruler/* $B/ruler/;      cp $H/qwen3_omni/results/* $B/logs/
cp $H/qwen3_omni/env.sh $H/qwen3_omni/RUNBOOK_64K.md $B/;  cp $H/sage_causal.sycl $H/flash_causal.sycl $B/kernel/
cd $B/kernel && export CUTLASS_DIR=/home/gta/sycl_tla_opt/third_party/sycl-tla-v0.8 PY=$HOME/omni_env/bin/python
bash build.sh build sage_causal.sycl sage_causal && bash build.sh build sage_orig.sycl sage_orig
```

The RULER data (`ruler/data/ruler_{2048,...,63488}.jsonl`, 222 MB) is not stored here; regenerate it with `ruler/generate.py` (steps in `ruler/README.md`; needs NVIDIA/RULER at commit db43f92 in `third_party/RULER`). Environment setup and the model loader fixes are in `../QWEN3_OMNI_RUNBOOK.md` section 6.

## Results

| file | what |
|---|---|
| `ruler_sweep64k_final.jsonl`, `summary_sweep64k_final.{md,csv}` | 64K RULER, 12 tasks × 15 samples: `sdpa`, `sage_int4` (INT4 Q·Kᵀ), `sage_int8` (INT8 with Q+K smoothing) |
| `ruler_sweep64k_int8.jsonl` | the raw INT8 (Q+K smoothing) sweep rows (arm `sage`) |
| `ruler_sweep64k_noqsm.jsonl` | current kernel (INT8, K smoothing only) on cwe, qa_2, niah_multivalue, niah_multiquery |
| `ruler_sweep64k_l3.jsonl` | INT4 with thinker layer 3 on SDPA (`sage_l3`), same four tasks |
| `ruler_e2e_final.jsonl` | current kernel vs SDPA, one prompt per task: TTFT / attention / E2E timing |
| `accuracy_probe*.json` | real-activation accuracy (per-layer cosine vs SDPA, KL, top-1): current kernel, INT4, INT8 with Q smoothing |
| `emulate_*.json` | per-layer attention-output cosine vs fp32 on captured 64K activations for INT4/INT8 schemes and the real kernels |
| `bench_sage_causal.{json,log}`, `device_peaks.json` | standalone kernel vs SDPA; measured B70 GEMM and bandwidth peaks |
| `vl_ramp*.jsonl` | Qwen3-VL-8B 8K–256K needle ramp (see `../QWEN3_VL_256K_RUNBOOK.md`; Sage rows measured with the INT4 kernel) |
