# Wan2.2-T2V-A14B on VBench-2.0 (SDPA vs Sage INT8 vs Sage INT4 vs SVDQuant): runbook

Use this to run, resume or finish the VBench-2.0 comparison. Start a new session in this directory with:
*"Read WAN22_T2V_VBENCH2_RUNBOOK.md and continue the VBench-2.0 work."*
Last updated 2026-11-10. The integration itself (INT8 experts, Sage attention hook, SVDQuant linears, lazy expert
loading) is described in `WAN22_I2V_RUNBOOK.md` sections 4 and 4.1; this runbook only covers what differs for T2V
and the benchmark.

## 1. Goal and status

Score four configurations of [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
on a VBench-2.0 subset, on one Arc Pro B70:

| arm (output folder) | block linears | self-attention |
|---|---|---|
| `sdpa` (default) | INT8 W8A16 | torch SDPA |
| `sage` | INT8 W8A16 | Sage INT8 (`sage_causal.sycl`, installed kernel) |
| `sage_int4` | INT8 W8A16 | Sage INT4 (`sage_causal.sycl` from commit `c2c1cce`) |
| `sdpa_svdq` | SVDQuant W4A4 for QKV + FFN, rest INT8 | torch SDPA |

| step | status |
|---|---|
| T2V download, INT8 conversion of both experts, fp32 experts deleted | done on the original machine (`/home/gta/models/wan22-t2v-a14b-int8`, 2 × 14 GB) |
| subset selection (`wan22/vbench/subset.json`) | done |
| batch generator (`wan22/run_wan_t2v_batch.py`) and chain (`wan22/vbench/run_chain.sh`) | written; the chain was launched (smoke test → SVDQuant calibration + conversion → 4 arms in 3 rounds) but no T2V video had been verified yet |
| evaluator environment `~/vbench_env` | torch 2.14.1+xpu, transformers 4.51 and PyPI deps installed; **LLaVA-NeXT and the VBench-2.0 package not installed** (section 5) |
| evaluator port to XPU, scoring, report | not started (section 6) |

On a new machine, start at section 3; on the original machine, check section 4.4 first (the chain resumes).

## 2. Benchmark subset

VBench-2.0 has 18 dimensions in 5 categories, 9–119 prompts per dimension, and asks for 3 videos per prompt
(20 for Diversity). At ≈ 25 min per 480p video per arm on one B70, the full suite is out of reach, so the subset is
**2 dimensions per category, 3 prompts per dimension, 1 seed per prompt; Diversity: 1 prompt × 6 seeds** =
33 videos per arm, 132 in total.

| category | dimensions | evaluator models (VBench-2.0) |
|---|---|---|
| Human Fidelity | Human_Anatomy, Human_Identity | YOLO-World + human/face/hand anomaly ViTs; ArcFace + CoTracker |
| Creativity | Composition, Diversity | LLaVA-Video-7B-Qwen2; feature diversity across seeds |
| Controllability | Camera_Motion, Dynamic_Attribute | CoTracker; LLaVA-Video-7B-Qwen2 |
| Physics | Mechanics, Multi-View_Consistency | LLaVA-Video-7B-Qwen2; RAFT + CoTracker + CLIP |
| Commonsense | Motion_Rationality, Instance_Preservation | LLaVA-Video-7B-Qwen2; YOLO-World tracking |

Creativity and Commonsense have exactly two dimensions. The others were chosen to be sensitive to numerical error
in attention or linears: anatomy and identity drift, global camera motion, attribute change over time and 3D
consistency under an orbiting camera. Qwen2.5-7B-Instruct is not needed by any chosen dimension.

Prompts are chosen by rule, not by hand (`vbench/select_subset.py`): the prompts at positions n/6, n/2 and 5n/6 of
each dimension's list in `prompts/prompt/<Dimension>.txt`; for Diversity the middle prompt. Each video has its own
seed (1000–1032), identical across arms. Rows are ordered in 3 rounds of 11 (one prompt of every dimension and 2
Diversity seeds per round), so `--limit 11 / 22 / 33` gives balanced partial results. Video files use the
VBench-2.0 naming `<Dimension>/<prompt[:180]>-<index>.mp4`.

## 3. Setup from scratch

Requires the `omni_bench` workspace at `/home/gta/omni_bench` (restore steps in `qwen3_omni/README.md`), `~/omni_env`
(torch 2.14.1+xpu) and the isolated diffusers install, Sage kernels and SVDQuant kernel from `WAN22_I2V_RUNBOOK.md`
section 5.2 (steps 1, 3 and 4), plus the INT4 Sage build from its section 5.3. Copy `wan22/*.py` to
`omni_bench/wan/` and `wan22/vbench/*` to `omni_bench/wan/vbench/`. Machine safety rules: `QWEN3_OMNI_RUNBOOK.md`
section 5 (GPU jobs through `rss_guard.sh`, one GPU job at a time, no `expandable_segments`).

```bash
cd /home/gta/omni_bench; export PYTHONPATH=/home/gta/pylibs/diffusers
# 1. weights: 117 GB download, then INT8 one expert at a time, deleting each fp32 expert after verification
#    (needs ~70 GB free at peak instead of ~145 GB; peak host RAM ~8 GB)
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local-dir /home/gta/models/wan22-t2v-a14b-diffusers
S=/home/gta/models/wan22-t2v-a14b-diffusers; D=/home/gta/models/wan22-t2v-a14b-int8
for sub in transformer transformer_2; do
  ./env.sh python -c "import sys; sys.path.insert(0, 'wan'); import wan_int8; wan_int8.convert_expert('$S/$sub', '$D/$sub')"
  python3 wan/verify_and_prune.py $S $D $sub
done

# 2. subset (needs a VBench clone: git clone --depth 1 https://github.com/Vchitect/VBench.git /home/gta/ext/vbench_src/VBench)
python3 wan/vbench/select_subset.py /home/gta/ext/vbench_src/VBench/VBench-2.0 wan/vbench/subset.json
```

`run_chain.sh` waits for the line `[t2v] done` in `logs/wan_t2v_convert.log` (written by the original queued
conversion). When converting by hand as above, create it first: `echo "[t2v] done" >> logs/wan_t2v_convert.log`.

Disk budget: T2V INT8 28 GB, T2V SVDQuant 22 GB, text encoder + VAE 12 GB, videos ≈ 1–2 MB each, evaluator models
≈ 20 GB (LLaVA-Video-7B 16 GB, the rest small).

## 4. Generation

### 4.1 Settings

Model card: 81 frames, 40 UniPC steps (flow shift 3.0), guidance 4.0 for the high-noise expert and 3.0 for the
low-noise expert (`boundary_ratio` 0.875), the official negative prompt, 16 fps; resolution 832×480 (32,760 tokens
per self-attention call). Peak VRAM ≈ 18 GiB (INT8 experts), ≈ 14 GiB (SVDQuant).

### 4.2 Batch generator

`run_wan_t2v_batch.py --arm sdpa|sage [--linear int8|svdq] [--arm-name NAME] [--limit N]` loads tokenizer, UMT5,
VAE and both expert skeletons once, encodes every prompt of the subset and frees UMT5, then for each video:
denoise (one expert on the GPU at a time, ≈ 3 s per swap), unload the expert, VAE decode, write the MP4 atomically
(`.part.mp4` then rename) to `wan/vbench/videos/<arm>/<Dimension>/...`, and append a row to
`wan/vbench/videos/<arm>/timings.jsonl` (denoise time excluding expert loads, self-attention time, decode time,
peak VRAM, Sage build). Finished videos are skipped, so any run can be restarted. `--prompt "..." --seed S`
runs a single prompt; `--calib-out file.pt` records SVDQuant calibration maxima (as in the I2V runner).

### 4.3 The chain

`wan/vbench/run_chain.sh` (resumable; rerun to continue):

1. smoke test: 17 frames, 4 steps, Sage, model-card example prompt → `wan/vbench/smoke/` (about 2 min);
2. SVDQuant calibration on the default SDPA arm with the model-card example prompt *"Two anthropomorphic cats in
   comfy boxing gear and bright gloves fight intensely on a spotlighted stage."* (not in the subset), seed 7,
   33 frames → `wan/outputs/t2v_calib_amax.pt` (about 10 min); then
   `wan_svdq.py convert` → `/home/gta/models/wan22-t2v-a14b-svdq` (rank 32, α 0.5; peak host RAM 19 GB);
3. for `--limit` 11, 22, 33: the four arms in order `sdpa`, `sage`, `sage_int4` (`SAGE_BUILD_DIR=kernel/build_int4`),
   `sdpa_svdq`.

```bash
cd /home/gta/omni_bench; nohup wan/vbench/run_chain.sh > logs/vb_chain.log 2>&1 &
grep "\[chain\]" logs/vb_chain.log                       # progress markers and failures
grep "\[batch\]" logs/vb_{sdpa,sage,sage_int4,sdpa_svdq}.log | tail
```

Time estimate (from the I2V runs at the same token count): ≈ 27 min per video for `sdpa`, 24 for `sage`, 22 for
`sage_int4`, 27 for `sdpa_svdq`; one round (44 videos) ≈ 18 h, all three ≈ 55 h. A round finishes all four arms
before the next starts, so the first comparable results exist after round 1.

### 4.4 Checks before trusting the run

- Smoke video `wan/vbench/smoke/smoke/custom/Two anthropomorphic cats...-0.mp4` exists and shows the prompt.
- First `sdpa` video: look at a frame (`ffmpeg -i <mp4> -vf "select=eq(n\,40)" -vframes 1 /tmp/f.png`); check
  `timings.jsonl` has `self_attention_calls` = 3,200 (40 steps × 2 CFG passes × 40 layers).
- The four arms of the same video use the same seed; `sage_build` in the `sage_int4` rows is `.../kernel/build_int4`.

## 5. Evaluator environment

`~/vbench_env` (Python 3.12 venv): torch 2.14.1+xpu and torchvision 0.29.2+xpu from
`https://download.pytorch.org/whl/xpu`, then from PyPI: transformers 4.51.0, tokenizers 0.21.1, accelerate 1.2.1,
einops, timm 0.4.12, numpy < 2, opencv-python-headless, scenedetect, av, decord, open_clip_torch 2.30.0, ftfy,
regex, sentencepiece, qwen-vl-utils, easydict, scikit-image, scipy, pandas, imageio(-ffmpeg), omegaconf, fvcore,
iopath, peft 0.14.0, huggingface-hub 0.30.0. Kept separate from `omni_env` (transformers 5.22) on purpose.

**Still to install** (needs explicit approval: these run third-party setup code from GitHub clones):

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git /home/gta/ext/llava_src/LLaVA-NeXT
git -C /home/gta/ext/llava_src/LLaVA-NeXT checkout 79ef45a6d8b89b92d7a8525f077c3a3a9894a87d   # VBench-2.0's pin
~/vbench_env/bin/pip install --no-deps -e /home/gta/ext/llava_src/LLaVA-NeXT
~/vbench_env/bin/pip install --no-deps -e /home/gta/ext/vbench_src/VBench/VBench-2.0
```

Then the evaluator weights (VBench-2.0 `pretrained/*/download.sh`, into `~/.cache/vbench2`): LLaVA-Video-7B-Qwen2
(`hf download lmms-lab/LLaVA-Video-7B-Qwen2`), RAFT (`models.zip` from Dropbox), ArcFace, YOLO-World and the three
anomaly-detector weights (Google Drive ids in `vbench2/utils.py:init_submodules`), CoTracker via `torch.hub`
(`facebookresearch/co-tracker`). Human_Anatomy and Instance_Preservation additionally need the mmdet / mmyolo /
YOLO-World stack (`mmdet==3.0.0`, `mmyolo==0.6.0`, mmcv, YOLO-World at `4826695`); mmcv's compiled ops are
CUDA-oriented and may only run on the CPU.

## 6. Remaining work

1. **Port the evaluators to XPU.** About 80 places in `VBench-2.0/vbench2` hard-code CUDA (`.cuda()`,
   `device='cuda'`). Run each chosen dimension with `device=torch.device("xpu")`, patching the hard-coded calls;
   LLaVA-Video-7B in bf16 on XPU (16 GB) fits alongside nothing else, so evaluate only when no generation job runs.
   Fall back to the CPU for the mmcv-based detectors if needed (33 × 4 videos is small).
2. **Subset evaluation wrapper.** `VBench2.build_full_info_json` raises if any of the 3 standard videos per prompt
   (20 for Diversity) is missing. Write a wrapper that builds the full-info JSON from `subset.json` (its
   `auxiliary_info` field is copied from `VBench2_full_info.json`) with only the generated videos, then calls the
   per-dimension `compute_<dimension>` functions. Check that Diversity accepts 6 videos.
3. **Score** every arm × dimension; aggregate per category as VBench-2.0 does (category = mean of its dimensions)
   and report per-arm category scores with the speed numbers from `timings.jsonl` (median denoise / self-attention
   time per video, speedup vs `sdpa`). With 3 prompts per dimension, report per-prompt paired differences vs `sdpa`
   rather than treating small score differences as significant.
4. **Same-seed divergence** (optional, cheap): `compare_videos.py` can compare each arm against `sdpa` per video
   (convert the MP4s to `.npy` first) to relate VBench score changes to pixel divergence.
5. **Document** results in this runbook and copy `timings.jsonl` and the evaluation JSONs to `wan22/vbench/results/`.

## 7. Known limitations

- The subset is small (3 prompts × 1 seed per dimension): enough to detect large regressions between arms, not to
  reproduce leaderboard scores (which use all prompts and 3 seeds).
- 480p instead of the model card's 720p example; the attention share and Sage speedup are larger at 720p.
- SVDQuant uses INT8-derived weights, α 0.5, rank 32 and no activation clipping; on I2V it showed visible grain on
  the conditioning frame (`WAN22_I2V_RUNBOOK.md` section 1). T2V calibration uses one prompt.
