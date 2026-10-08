# Intel B580/B70 Wan2.2 Stage2 Inference Port

**Status:** Phase 1 — smoke infer **passed**; **Phase 1D** quality **resolved** (XPU softmax
regression in PyTorch ≥ 2.13; environment pinned to `2.12.1+xpu`); **Phase 2 Fused Attention**
integrated and validated on B580 (`fused_rope_prope_sdpa` / `fused_rope_prope_sage` deliver
1.45×–1.47× diffusion forward speedup; 160-frame generation verified).
**Last updated:** 2026-10-06 (B580 / B70)
**Scope:** Inference only

**Weights policy (XPU / Intel):** Treat **`ema`** as the only pass used for quality
judgment and publication. The 81f infer contract still runs **live then ema** (see
`sgf.py`); ignore `live_self_forcing_nfe4` artifacts unless debugging parity.
Checkpoint role loading for standalone infer already defaults to **`inference.weights:
ema`** in `infer_stage2_sgf_81f.yaml`.

Session numbering: **Session 1** = Intel B580 development machine (log entries 1–4 below).
**Session 2** = Intel B70 (32 GiB VRAM), this host.

The reproducible environment entry point is
[`scripts/setup_wan22_xpu.sh`](../../scripts/setup_wan22_xpu.sh); its non-PyTorch
runtime pins are listed in
[`environments/wan22-xpu-requirements.txt`](../../environments/wan22-xpu-requirements.txt).

## Locked Goal

Port the SolarWM Wan2.2 TI2V-5B Stage2 DMD/SGF inference route to run on Intel
discrete GPUs through PyTorch XPU (validated on **B580** and **B70** class cards).

The first supported deployment targets are:

- One Intel B580 or B70 GPU (32 GiB class VRAM on B70).
- One Python 3.12 virtual environment.
- Latest PyTorch XPU release that passes the kernel-correctness gate, currently
  **PyTorch 2.12.1** (`2.13.0+xpu` and newer regress `torch.softmax`; see
  "Phase 1D resolution" below).
- Native PyTorch XPU APIs, including modern `torch.accelerator` device discovery where available.
- BF16 model execution as the initial precision path.
- No CUDA packages, CUDA wheels, FlashAttention package, training, or multi-GPU execution in the Intel environment.
- Keep the fixes cross platform, so that the same code can run unchanged on CUDA GPUs as well.

FP8 is explicitly deferred to **Phase 2**. It must not affect the Phase 1 environment or become an implicit fallback.

## Acceptance Criteria

Phase 1 is complete when all of the following are true:

1. A clean Python 3.12 virtual environment can install the Intel/XPU runtime without installing CUDA dependencies or the standalone `flash-attn` package.
2. PyTorch reports the Intel GPU through its XPU/accelerator APIs, and a small tensor operation executes on the selected device.
3. `solarwm environment probe` reports an inference-capable XPU environment without requiring CUDA.
4. The Wan2.2 Stage2 route accepts an explicit XPU device and does not unconditionally initialize CUDA or NCCL for one process.
5. Wan attention uses a PyTorch-native XPU-supported path. The plan assumes PyTorch provides internal optimized attention/SDPA kernels, but this must be confirmed by runtime probes and not confused with the separate `flash-attn` Python package.
6. Existing CUDA behavior and CPU/mock tests remain intact.
7. A released Wan2.2 5B Stage2 checkpoint produces a valid minimal inference artifact on Intel XPU, or a precise unsupported operator/memory blocker is recorded here.

## Explicit Non-Goals

- Training or checkpoint resume on XPU.
- Multi-GPU XPU execution, FSDP, CCL, or distributed sequence parallelism.
- Porting LTX-2.5, MiniMax-H3, or Wan2.2 I2V-A14B.
- Installing or adapting CUDA-only FlashAttention.
- Quantized weights in Phase 1.
- Claiming FP8 support before the Phase 2 capability and quality checks pass.

## Quantization and Weight Availability

### Current repository state

The SolarWM Wan2.2 runtime has no Wan-specific int4, int8, or FP8 checkpoint loader, quantization configuration, or quantized weight contract. Existing Wan2.2 configs use ordinary model checkpoints with BF16/FP32 runtime settings. The FP8 reference found during discovery belongs to the LTX data pipeline and is not reusable as Wan inference support.

The published SolarWM collection advertises Wan2.2 5B base and staged checkpoints, including Stage2, but no quantized Wan2.2 artifact was identified from the public collection listing. The exact local/authenticated checkpoint inventory must be recorded when the model directory is available.

### Phase 1 support matrix

| Format/path | Phase 1 status | Notes |
|---|---|---|
| BF16 model weights and execution | Target | Primary B580 path; validate operator coverage and memory use. |
| FP32 weights or selected FP32 operations | Compatibility fallback | Use only where required by an operator or numerical check. |
| Native PyTorch XPU SDPA/attention kernels | Target | Validate on the installed PyTorch 2.14 XPU build. |
| Standalone `flash-attn` package | Excluded | Do not install; its CUDA assumptions are incompatible with this environment. |
| int8/int4 quantized Wan weights | Unsupported | No loader or compatible SolarWM weights identified. |
| FP8 weights/execution | Deferred to Phase 2 | No Phase 1 implementation or dependency. |

## Environment Plan

Create the environment separately from the repository's CUDA-oriented Wan environment. The repository's existing release-tested baseline is Python 3.10, PyTorch 2.5.1/CUDA 12.4, and FlashAttention 2.8.3; that baseline is not the Intel target.

Planned environment name:

```text
.venv-wan22-xpu
```

Planned setup shape, to be finalized against the available Intel wheel/index:

```bash
cd /home/gta/code/SolarWM
python3.12 -m venv .venv-wan22-xpu
source .venv-wan22-xpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Install the latest available PyTorch 2.14 XPU build using the official Intel/XPU index.
# Do not install a CUDA wheel and do not install flash-attn.
python -m pip install torch torchvision torchaudio --index-url <verified-xpu-index>
python -m pip install -e .
python -m pip install \
  diffusers==0.38.0 \
  transformers==5.12.1 \
  peft==0.20.0 \
  'ftfy>=6.2' 'omegaconf>=2.3' 'regex>=2024.0'
solarwm environment probe
```

For recreation, run the checked-in script from the repository root:

```bash
export https_proxy=http://proxy-dmz.intel.com:912
./scripts/setup_wan22_xpu.sh
```

The script recreates `.venv-wan22-xpu`, installs PyTorch `2.14.0+xpu` from the
PyTorch XPU index, installs SolarWM without the `[wan]` extra, installs the
runtime pins from `environments/wan22-xpu-requirements.txt`, and runs the
accelerator and SolarWM environment probes. `VENV_DIR`, `PYTHON_VERSION`,
`PYTORCH_VERSION`, and `XPU_INDEX_URL` may be overridden for a local mirror or
future compatible wheel.

The exact PyTorch 2.14 XPU wheel/index and compatible versions of the Wan runtime dependencies must be captured from the successful installation. Do not substitute a CUDA index or silently downgrade PyTorch. Intel Extension for PyTorch is not a planned dependency because its upstream project is archived/retired and the maintained direction is native PyTorch XPU.

### Environment validation checklist

Record the output of these checks in the session log:

```bash
python - <<'PY'
import torch

print('torch:', torch.__version__)
print('has torch.xpu:', hasattr(torch, 'xpu'))
print('xpu available:', hasattr(torch, 'xpu') and torch.xpu.is_available())
print('xpu count:', torch.xpu.device_count() if hasattr(torch, 'xpu') else 0)
print('has torch.accelerator:', hasattr(torch, 'accelerator'))
if hasattr(torch, 'accelerator'):
    print('accelerator:', torch.accelerator.current_accelerator())
PY
```

Then run a small device operation and probe attention/autocast behavior. The result must identify whether `torch.accelerator` is available in the selected release and which compatibility fallback is required when it is not.

## Implementation Plan

### Phase 1A: Device and runtime contract

1. Add a small device selection utility for inference that prefers an explicit configured device and otherwise uses modern accelerator discovery where available.
2. Preserve explicit `cuda`, `cpu`, and `xpu` behavior; do not make device selection depend on CUDA availability.
3. Keep Stage2 topology at world size 1 and sequence-parallel size 1. Avoid process-group initialization for the single-device route.
4. Update `stage2.py` and shared inference adapter construction to stop assuming `torch.device('cuda', local_rank)`.
5. Make `torch.autocast`, cleanup, synchronization, and RNG handling use the selected device type. Preserve CUDA-specific behavior for the existing CUDA environment.
6. Update readiness/probe reporting so inference can pass with XPU while training continues to report its existing unsupported status.

### Phase 1B: Attention and model execution

1. Ensure the XPU path does not import or select the standalone FlashAttention package.
2. Use PyTorch-native scaled-dot-product attention or another native XPU-supported implementation selected by capability detection.
3. Keep CUDA FlashAttention fast paths unchanged where possible.
4. Add tests for device selection, no-CUDA single-rank initialization, attention fallback selection, autocast device type, and XPU-safe cleanup/RNG behavior using mocks where hardware is unavailable.
5. Add an actual B580 smoke test for tensor operations and the smallest feasible Stage2 inference before attempting long-horizon generation.

### Phase 1C: Stage2 validation

1. Use the existing `infer_stage2_sgf_camera_length.yaml` contract with one process and an explicit XPU device override.
2. Validate the released `SolarWM-5B-sgf-stage2-81f` checkpoint and record whether the checkpoint is BF16/FP32 and whether any conversion is necessary.
3. Start with the smallest valid frame/chunk settings available for a smoke run.
4. Confirm the generated artifact is readable, has the expected dimensions/frame count, and does not contain NaN/Inf output.
5. Record memory use, runtime, unsupported operators, and any required workarounds.
6. Expand to the documented 81-frame Stage2 path only after the smoke run passes.

### Phase 1D: Visual quality (XPU)

1. **EMA-only evaluation** — compare and ship `generation/ema_self_forcing_nfe4/` only;
   optional follow-up: relax `sgf.py` to allow a single `ema` pass on standalone infer
   (saves ~half diffusion+VAE time on B70).
2. **Regional blue noise** — see session log (frame-0 experiment **reverted**); continue
   [`wan22-stage2-vae-decode-debug.md`](wan22-stage2-vae-decode-debug.md) §6 and Phase 1E audit.
3. **CUDA reference bundle** — `generate_wan22_stage2_cuda_reference.py` on CUDA; XPU compare.

### Phase 1E: CUDA → XPU port audit (active)

Systematic grep: `device.type == "cuda"`, `torch.cuda`, `device="cuda"`, `enabled=...cuda`
under `src/solarwm/` (Wan22 infer path first; other backends out of scope unless shared).

| Location | Role | XPU / infer status |
|----------|------|-------------------|
| `runtime/distributed.py` | `resolve_device`, init, cleanup | **Done** — XPU set_device; `synchronize_accelerator` / `empty_accelerator_cache` |
| `runtime/inference.py` | post–checkpoint-load cache | **Done** — uses `empty_accelerator_cache` |
| `runtime/stage2.py` adapter | `resolve_device`, XPU text on CPU | **Done** |
| `runtime/stage2.py` self-forcing | `torch.autocast` cuda+xpu | **Done** |
| `runtime/stage2.py` training init | hard-coded `cuda`, `manual_seed_all` | Training only — defer |
| `runtime/modeling/attention.py` | FlashAttn vs SDPA | **Done** — XPU uses SDPA via `attention()` |
| `runtime/components.py` `Wan5BVAE` | encode/decode autocast `device_type` | **Done**; `clear_cache` on entry (investigate) |
| `runtime/modeling/vae.py` | legacy `amp`, default `device="cuda"` | Legacy helper — not `Wan5BVAE` path; audit |
| `runtime/modeling/t5.py` | default device | **Done** — `resolve_device()` fallback |
| `runtime/sequence_parallel.py` | scratch tensor device | **Done** — `resolve_device()` |
| `runtime/checkpoint.py` / `stage2.py` RNG | CUDA RNG in checkpoints | Training — add XPU RNG later if needed |
| `runtime/preencode.py` | CUDA-only preencode | Out of scope (not infer smoke) |
| `runtime/readiness.py` | CUDA probe | Infer uses `require_cuda=False`; optional XPU probe |
| `solarwm/runtime/randomness.py` | `seed_process` | **Done** — `torch.xpu.manual_seed_all` |
| Other backends (`minimax_h3`, `ltx25`, …) | CUDA training/infer | **Not in Wan22 XPU scope** |

Next: CUDA reference latents/pixels; VAE encode idempotency on XPU (second `_conditions` encode
≠ first); bisect blue patches with `latent0_only_decode` vs full-sequence decode.

## Phase 2: FP8 Follow-Up

FP8 is intentionally postponed until Phase 1 BF16 inference is stable.

Phase 2 will:

1. Inventory the exact FP8 dtypes and kernels exposed by the installed PyTorch XPU release.
2. Test FP8 linear, matmul, convolution/VAE, attention, conversion, and autocast operations independently on the B580.
3. Decide whether FP8 is weight-only, activation-only, mixed, or unsupported for each Wan component.
4. Define an explicit configuration field and reject unsupported combinations instead of silently converting weights.
5. Establish checkpoint format and scale metadata requirements. Do not call BF16 weights FP8 without a reproducible conversion contract.
6. Compare output quality and numerical stability against the BF16 baseline on fixed seeds.
7. Measure memory and throughput, then document whether FP8 is beneficial on the B580.

FP8 Phase 2 is complete only when a fixed-seed Stage2 comparison passes artifact validity and defined quality tolerances. Until then, the supported quantization statement remains: BF16/FP32 runtime only; no quantized Wan weights available in the project.

## Session Log

### B580 — Session 1 (log 1/4) - 2026-09-16 - Planning and scope lock

**Requests locked**

- Target only Wan2.2 5B Stage2 inference on the Intel B580.
- Use Python 3.12.
- Use the latest available PyTorch release; the current plan target is PyTorch 2.14 with XPU support.
- Omit CUDA dependencies.
- Defer FP8 to Phase 2.
- Do not begin source code changes in this session.

**Discovery performed**

- Read the repository environment guide, Wan2.2 Stage2 guide, Stage2 config, attention implementation, inference adapter, and distributed initialization.
- Confirmed the existing Stage2 code hard-codes CUDA in adapter construction, device setup, autocast, cleanup, and RNG handling.
- Confirmed the attention module has an SDPA fallback but the standalone FlashAttention path contains CUDA-only assumptions.
- Searched for Wan quantization support and found no Wan int4/int8/FP8 loader or quantized checkpoint contract.
- Checked the public SolarWM collection listing; it lists Wan2.2 5B base/stage checkpoints but no quantized artifact was identified.
- Checked current upstream PyTorch documentation. The XPU start guide redirects to the 2.14 documentation, so PyTorch 2.14 is the current plan target. `torch.accelerator` availability must still be checked against the installed wheel.
- Checked Intel Extension for PyTorch and found the upstream project archived/retired; it is excluded from the planned environment.

**Results**

- No source files changed.
- No virtual environment created yet.
- No PyTorch/XPU package probe run yet.
- No quantized weights identified.
- FP8 moved from a Phase 1 possibility to Phase 2.
- Phase 1 precision target is BF16, with FP32 compatibility fallback only where required.

**Next steps**

1. Verify the exact available PyTorch 2.14 XPU wheel/index and Intel B580 driver visibility.
2. Create `.venv-wan22-xpu` with Python 3.12, no CUDA packages, and no standalone FlashAttention.
3. Record `torch.accelerator`, `torch.xpu`, native attention, BF16, and autocast probe results.
4. Begin Phase 1 implementation only after this plan is reviewed and the environment is reproducible.
5. Maintain this document after every implementation session with commands, results, blockers, and next steps.

### B580 — Session 1 (log 2/4) - 2026-09-16 - Environment recreation

**Changes**

- Recreated the lost Wan XPU implementation in the working tree for device
  resolution, single-device initialization/cleanup, Stage2/inference adapters,
  device-aware autocast, and non-CUDA attention fallback.
- Added `scripts/setup_wan22_xpu.sh` as the repeatable Python 3.12/PyTorch
  `2.14.0+xpu` environment installer.
- Added `environments/wan22-xpu-requirements.txt` for the non-PyTorch Wan
  inference dependencies and pytest.

**Validation**

- `bash -n scripts/setup_wan22_xpu.sh` passed.
- `git diff --check` passed.
- The recreated environment installed successfully with Python 3.12.14,
  PyTorch `2.14.0+xpu`, torchvision/torchaudio XPU wheels, and no CUDA wheel or
  standalone `flash-attn` package.
- PyTorch reports `torch.xpu.is_available()=True`, one XPU device, and
  `torch.accelerator.current_accelerator()=xpu`.
- BF16 matrix multiplication executed on `xpu:0`.
- Native XPU scaled-dot-product attention under BF16 autocast executed on
  `xpu:0` with finite output.
- `python -m solarwm environment probe` reports Python 3.12.14, PyTorch
  `2.14.0+xpu`, Diffusers 0.38.0, Transformers 5.12.1, and PEFT 0.20.0.
- Focused validation passed: `92 passed` in
  `tests/backends/wan22/test_inference_runtime.py` and
  `tests/backends/wan22/test_stage2_runtime.py`.

**Remaining work**

- Run the smallest real Wan2.2 Stage2 checkpoint inference on the B580.
- Validate model/VAE operators and memory use, then record the artifact result.
- Add or refine readiness reporting for XPU if the real inference command
  exposes a CUDA-only readiness gate.

**Current blocker**

- The new machine has no local model or data artifacts. Hugging Face
  authentication is now valid as `ssheorey-intel`, but authenticated dry-runs
  for both `junchaoh-cs/SolarWM` and `junchaoh-cs/SolarWM-Data` still return
  `Access denied. This repository requires approval.` The inference data guide
  also requires accepting access terms before downloading the standalone test
  payload.
- The machine has sufficient storage for the artifacts: approximately 404 GiB
  free on `/home`/`/tmp` at the time of the check.
- Resume after access is granted with:

  ```bash
  export https_proxy=http://proxy-dmz.intel.com:912
  .venv-wan22-xpu/bin/hf auth whoami  # currently succeeds as ssheorey-intel
  export SOLAR_MODEL_ROOT=/home/gta/models/SolarWM
  export SOLAR_DATA_HOME=/home/gta/data/SolarWM-Data
  export SOLAR_DATA_ROOT="$SOLAR_DATA_HOME/releases-v1"
  export SOLAR_OUTPUT_ROOT=/home/gta/outputs/solarwm
  .venv-wan22-xpu/bin/hf download junchaoh-cs/SolarWM \
    --include 'SolarWM-5B-*/**' --local-dir "$SOLAR_MODEL_ROOT"
  .venv-wan22-xpu/bin/hf download junchaoh-cs/SolarWM-Data \
    --repo-type dataset --local-dir "$SOLAR_DATA_HOME"
  ```

  Before rerunning those commands, request/accept access for the model and
  dataset repositories in the Hugging Face browser while logged in as
  `ssheorey-intel`. The raw-WDS/test payload required by the selected inference index must also
  be available under `$SOLAR_DATA_ROOT`; the main controls repository alone is
  not sufficient for online Wan inference. Do not send a token through chat;
  authenticate directly in the terminal.

### B580 — Session 1 (log 3/4) - 2026-09-16 - Approved weight download and payload selection

  **Results**

  - Hugging Face approval now permits authenticated dry-runs for the model and
    release-controls repositories.
  - The exact inference weight set was confirmed before download: `SolarWM-5B-base`
    (34.2 GiB) plus `SolarWM-5B-sgf-stage2-81f` (40.0 GiB). No Stage0.5, Stage1,
    or training-only checkpoint directories are being downloaded.
  - The small `releases-v1/test-set` controls were downloaded (581 KiB).
  - The first selected `abot` test row requires raw shard
    `raw-wds/abot/shards/kept-xhigh-000704.tar` (1.62 GiB).
  - That raw shard is not present in the controls repository. The separate
    `SolarWM-Data_test-set-v1` repository exposes standalone `abot` shards but
    currently still returns an approval error for this account.
  - The limited model download is active under `/home/gta/models/SolarWM`; the
    download cache had written approximately 4.3 GiB at the last check.

  **Next steps**

  1. Let the base and Stage2 inference weights finish downloading.
  2. Obtain approval for `junchaoh-cs/SolarWM-Data_test-set-v1`, or obtain the
     documented raw-WDS shard through the data-access process.
  3. Download only the selected standalone `abot` shard and matching index if
     available, or choose a sample from an accessible payload.
  4. Adapt the Stage2 config paths and run the smallest valid B580 inference.
  5. Verify and record the generated MP4 and update this session log.

### B580 — Session 1 (log 4/4) - 2026-09-16 - Inference launcher and machine handoff

**Repository state**

The working tree has not been committed. Transfer all repository changes,
including the untracked files listed below, before continuing:

Modified tracked files:

- `src/solarwm/backends/wan22/backend.py`
- `src/solarwm/backends/wan22/runtime/distributed.py`
- `src/solarwm/backends/wan22/runtime/inference.py`
- `src/solarwm/backends/wan22/runtime/modeling/attention.py`
- `src/solarwm/backends/wan22/runtime/stage2.py`
- `docs/backends/wan22-ti2v-5b.md`

Untracked files that are part of the implementation:

- `docs/backends/intel-b580-wan22-stage2.md`
- `environments/wan22-xpu-requirements.txt`
- `scripts/setup_wan22_xpu.sh`
- `scripts/run_wan22_stage2_xpu.sh`

No `git commit` has been created. Use the normal repository transfer/commit
workflow on the source machine, or copy these files explicitly. Do not use a
reset or checkout that would discard the uncommitted runtime changes.

**Implemented runtime changes**

- Device resolution accepts explicit `xpu` and uses `torch.accelerator` when
  available; one-process XPU execution avoids CUDA/NCCL initialization.
- Inference and Stage2 paths use the selected device for autocast, movement,
  synchronization, cleanup, and RNG handling.
- Standalone FlashAttention is selected only for CUDA; XPU uses native PyTorch
  SDPA.
- The inference readiness gate no longer requires CUDA. Training remains
  outside the XPU scope.
- Stage2 builds the transformer architecture and loads the Stage2 role directly
  without requiring base transformer safetensor shards.
- `scripts/run_wan22_stage2_xpu.sh` performs asset preflight and launches the
  one-process Stage2 camera-length route with `inference.device=xpu`.

**Validation completed**

- Python `3.12.14` environment: `.venv-wan22-xpu`.
- PyTorch `2.14.0+xpu`; `torch.xpu.is_available()` is true; one XPU device.
- `torch.accelerator.current_accelerator()` returns `xpu`.
- BF16 XPU matmul passed.
- Native XPU SDPA under BF16 autocast passed with finite output.
- `python -m solarwm environment probe` passed.
- Focused tests passed: `92 passed` in
  `tests/backends/wan22/test_inference_runtime.py` and
  `tests/backends/wan22/test_stage2_runtime.py`.
- `bash -n scripts/setup_wan22_xpu.sh` and
  `bash -n scripts/run_wan22_stage2_xpu.sh` passed.
- `git diff --check` passed.

**Model assets completed on the source machine**

Local root: `/home/gta/models/SolarWM`

- `SolarWM-5B-base/`: approximately `14G`, containing the required shared
  runtime assets:
  - `text_encoder/models_t5_umt5-xxl-enc-bf16.pth`
  - `tokenizer/`
  - `vae/Wan2.2_VAE.pth`
  - `conditioning/wan_negemb_cn.pth`
- `SolarWM-5B-sgf-stage2-81f/`: approximately `38G` on disk.
- `SolarWM-5B-sgf-stage2-81f/model.pt`: exactly `39,998,928,500` bytes and
  present after the download completed.

Do not download the approximately 20G base transformer safetensor shards for
this inference-only route. The Stage2 checkpoint contains the complete trained
transformer and the config uses the built-in architecture definition. The
Stage2 checkpoint contains both live and EMA roles; published inference
defaults to EMA.

**Current hard blocker**

The source machine has no raw-WDS test data. The launcher stops before model
loading with:

```text
Missing required inference asset:
/home/gta/models/SolarWM-Data/releases-v1/recipes/clean-81f/raw-wds/test-index.jsonl.gz
```

The selected first `abot` test row previously required:

```text
raw-wds/abot/shards/kept-xhigh-000704.tar  (~1.62 GiB)
```

The release controls/index repository does not contain that raw shard. The
separate `junchaoh-cs/SolarWM-Data_test-set-v1` repository exposes standalone
test shards but required separate Hugging Face approval at the time of this
session. The next machine must obtain approval for that repository, or obtain
the documented raw-WDS test payload by another approved route. A controls-only
download is insufficient for online Wan inference because the VAE, video, and
camera members are read from the raw shard.

**Continuation on the next machine**

1. Transfer the repository changes and enter the repository:

   ```bash
   cd /path/to/SolarWM
   git status --short
   ```

2. Recreate the XPU environment. The setup script requires `uv` and uses the
   Intel proxy by default:

   ```bash
   export https_proxy=http://proxy-dmz.intel.com:912
   export HTTPS_PROXY="$https_proxy"
   ./scripts/setup_wan22_xpu.sh
   source .venv-wan22-xpu/bin/activate
   ```

   Do not install `.[wan]`; that extra installs CUDA FlashAttention. Do not
   install standalone `flash-attn` or CUDA PyTorch wheels.

3. Set paths. Keep model and data roots separate:

   ```bash
   export SOLAR_MODEL_ROOT=/path/to/SolarWM-models
   export SOLAR_DATA_HOME=/path/to/SolarWM-Data
   export SOLAR_DATA_ROOT="$SOLAR_DATA_HOME/releases-v1"
   export SOLAR_OUTPUT_ROOT=/path/to/solarwm-outputs
   ```

4. Ensure the model closure exists under `$SOLAR_MODEL_ROOT` as listed above.
   If transferring instead of redownloading, preserve the directory names and
   verify `model.pt` has size `39998928500` bytes.

5. Obtain the raw test index and matching shard(s), then verify that this path
   exists:

   ```bash
   test -s "$SOLAR_DATA_ROOT/recipes/clean-81f/raw-wds/test-index.jsonl.gz"
   ```

   Download only the shard(s) referenced by the selected test index when the
   standalone payload manifest permits it; do not download training data.

6. Run the launcher. It checks the Python environment, text encoder,
   tokenizer, VAE, Stage2 checkpoint, and test index before model allocation:

   ```bash
   scripts/run_wan22_stage2_xpu.sh
   ```

   Override `SOLAR_RUN_ID` for each new create-only camera inference run. Do
   not reuse a completed run ID.

7. Verify the output transaction beneath
   `$SOLAR_OUTPUT_ROOT/wan22-ti2v-5b-stage2-xpu/`, including the run-level
   `COMPLETE.json`, generated MP4, comparison MP4, camera `.npy`, and finite
   output metrics. Record peak XPU memory, runtime, unsupported operators, and
   any required workaround here.

**Camera/control context**

Camera conditioning currently comes from each raw sample's camera NPZ, usually
the `c2w` array. SolarWM selects the requested frame window, rebases it to the
first pose, converts it to relative view matrices, expands each pose across
the model token sequence, and supplies `viewmats` plus normalized intrinsics
`K` to the PRoPE camera-attention path. There is no keyboard, mouse, joystick,
or live interactive control adapter yet. Interactive controls would need to be
converted into future camera poses before being passed through this same
`viewmats`/`K` interface.

**Immediate next milestone (B580 handoff)**

Acquire the standalone raw test payload, run the smallest real Stage2 XPU
inference, and record whether the first failure is an operator, memory,
checkpoint-loading, data-format, or output-encoding issue. Only after that
smoke run should the route be expanded or interactive controls be designed.

### B70 — Session 2 - 2026-09-16 - Weights, memory fit, smoke inference

**Machine**

- Intel **B70**, 32 GiB VRAM, one XPU device (`torch.xpu.is_available()=True`).
- Repo: `/home/ssheorey/code/SolarWM`; venv: `.venv-wan22-xpu` (Python 3.12.3,
  PyTorch `2.14.0+xpu`).
- Proxy: `https_proxy` from `~/.bashrc` (`http://proxy-dmz.intel.com:912`).
- Local operator notes: `AGENTS.local.md` (not committed).

**Artifacts downloaded (`hf`, minimal Stage2 closure)**

```bash
export SOLAR_MODEL_ROOT=/home/ssheorey/models/SolarWM
export SOLAR_DATA_HOME=/home/ssheorey/data/SolarWM-Data
export SOLAR_DATA_ROOT="$SOLAR_DATA_HOME/releases-v1"
hf download junchaoh-cs/SolarWM \
  --include "SolarWM-5B-base/conditioning/**" \
  --include "SolarWM-5B-base/text_encoder/**" \
  --include "SolarWM-5B-base/tokenizer/**" \
  --include "SolarWM-5B-base/vae/**" \
  --include "SolarWM-5B-sgf-stage2-81f/**" \
  --local-dir "$SOLAR_MODEL_ROOT"
hf download junchaoh-cs/SolarWM-Data --repo-type dataset \
  --include "releases-v1/recipes/clean-81f/raw-wds/test-index.jsonl.gz" \
  --include "releases-v1/release.json" \
  --include "releases-v1/example/**" \
  --local-dir "$SOLAR_DATA_HOME"
```

- `SolarWM-5B-sgf-stage2-81f/model.pt`: `39,998,928,500` bytes (verified).
- `SolarWM-5B-base/`: ~14 GiB (text encoder, tokenizer, VAE, conditioning).
- Raw `raw-wds/.../*.tar` shards are **not** on the main `SolarWM-Data` HF repo;
  `junchaoh-cs/SolarWM-Data_test-set-v1` still returns access denied for
  `ssheorey-intel`. Smoke path uses `releases-v1/example/raw/xhigh.tar` (~5.4 MiB).

**Runtime changes (this session)**

- `environments/wan22-xpu-requirements.txt`: added `einops`, `decord`.
- `readiness.py`: `flash_attention_missing` is a **warning** when `require_cuda=False`
  (inference on XPU without standalone FlashAttention).
- `stage2.py` Stage2 inference adapter: UMT5 stays on **CPU** on XPU; init order is
  text encoder → diffusion on XPU → VAE on XPU (avoids process kill during VAE load
  when diffusion weights were still on CPU).
- `inference.py`: move `prompt_embeds` to inference device after CPU encoding;
  `torch.xpu.empty_cache()` after checkpoint role load.
- `modeling/model.py`: cross/self blocks call `attention()` (SDPA on XPU) instead of
  `flash_attention()` (CUDA-only assert).
- `scripts/run_wan22_stage2_xpu.sh`: default paths → `/home/ssheorey/models/SolarWM`,
  `/home/ssheorey/data/SolarWM-Data/releases-v1`.

**Validation**

- `CudaWanStage2GenerationAdapter` constructs successfully; ~**11.4 GiB** XPU allocated
  after load (diffusion + VAE; text encoder on host RAM).
- Earlier failures on this host (resolved or understood):
  - Missing `einops` / `decord` in venv → install via `uv pip`.
  - Full `example/index.jsonl.gz` fails canonical validation (duplicate `sample_id`s) →
    single-row `example/smoke-index.jsonl.gz`.
  - Exit **137** during adapter init when VAE was constructed before diffusion `.to(xpu)`.
  - First smoke run after adapter fix: `validation row lacks source num_frames` →
    enriched smoke index from shard manifest (`num_frames=160`, caption, camera members).
  - Generation `AssertionError` in `modeling/model.py` cross-attn: direct
    `flash_attention()` CUDA assert → fixed to `attention()` SDPA path.

**Smoke inference command (81f, one sample, example data)**

```bash
export SOLAR_RUN_ID="wan22-b70-smoke-$(date +%Y%m%d-%H%M%S)"
python -m solarwm infer \
  --config configs/examples/wan22_ti2v_5b/infer_stage2_sgf_81f.yaml \
  --set "model.base_path=$SOLAR_MODEL_ROOT/SolarWM-5B-base" \
  --set "checkpoint.path=$SOLAR_MODEL_ROOT/SolarWM-5B-sgf-stage2-81f" \
  --set "data.index_root=$SOLAR_DATA_ROOT/example" \
  --set "data.transport.root=$SOLAR_DATA_ROOT/example" \
  --set "data.test_index=smoke-index.jsonl.gz" \
  --set "inference.device=xpu" \
  --set "validation.sample_count=1" \
  --set "data.num_workers=1" \
  --set "inference.run_id=$SOLAR_RUN_ID" \
  --set "runtime.output_dir=$SOLAR_OUTPUT_ROOT/wan22-ti2v-5b-stage2-xpu"
```

Log: `outputs/wan22-b70-infer.log`. Require run-level `COMPLETE.json` under
`runtime.output_dir` before treating MP4 outputs as valid.

**B70 smoke infer (2026-09-17)**

- `RUN_ID=wan22-b70-smoke-20260917-212926`, config `infer_stage2_sgf_81f.yaml`, example `smoke-index.jsonl.gz`.
- Fix: Stage2 self-forcing `torch.autocast` enabled for **`xpu`** as well as `cuda` (was CUDA-only).
- Artifacts:
  - `outputs/wan22-ti2v-5b-stage2-xpu/generation/live_self_forcing_nfe4/slot-000000/video.mp4`
  - `outputs/wan22-ti2v-5b-stage2-xpu/generation/ema_self_forcing_nfe4/slot-000000/video.mp4` (and `compare.mp4` per pass).

**B70 frame-0 `_restore_first` experiment (2026-09-17) — reverted**

Added training-parity final `_restore_first` on inference self-forcing output + pre-decode
restore. EMA re-infer (`outputs/wan22-ti2v-5b-stage2-xpu-frame0-fix/`) looked **worse** than
pre-change smoke (`outputs/wan22-ti2v-5b-stage2-xpu/`). **Reverted** those `stage2.py` hunks.

Frame-0 PNGs (`outputs/debug/frame0-diagnose-post-fix/frame0_diagnose/`):

| File | Observation |
|------|-------------|
| `source_t0.png` | Clean reference |
| `vae_roundtrip_t0.png` | **Blue patches** across the frame (encode→decode on XPU) |
| `latent0_only_decode_t0.png` | Some blue patches; **kitchen still recognizable** |
| `rollout_decode_t0.png` | **Very noisy**; source not recognizable |

Probe: latent pin `max_abs_diff: 0.0` with rollout `first_latent` (metric only; visuals regressed).

**CUDA reference vs XPU (2026-09-19)** — see [`wan22-stage2-vae-decode-debug.md`](wan22-stage2-vae-decode-debug.md) §7.

- CUDA reference: `outputs/wan22-cuda-decode-reference/` (good video on CUDA).
- `vae-xpu-probe`: rollout latents **max_abs_diff ≈ 8.6** vs CUDA → **diffusion/self-forcing** mismatch.
- `vae-isolation-xpu` (`--decode-reference-latents`): same CUDA latents → XPU `Wan5BVAE.decode` vs CUDA saved decode **mean_abs_diff ≈ 0.43** → **VAE on XPU** also wrong.
- Debug script now writes `decode_*_production.pt` (infer path); re-run CUDA reference once to add `decode_cuda_production.pt` for apples-to-apples VAE compare.

**Open items (Phase 1D / 1E)**

1. **VAE** — bisect XPU decode vs CUDA on shared `latents_bf16.pt`; consider CPU VAE decode fallback for XPU infer.
2. **Diffusion** — rollout parity (attention/SDPA, autocast, step dumps) after VAE baseline is clear.
3. Re-run CUDA reference with new `decode_cuda_production.pt` artifact.
4. Phase 1E CUDA-path audit (table in plan); test-set HF access optional.
5. Optional: EMA-only pass in `sgf.py` for faster XPU iteration.

## Phase 1D resolution — XPU softmax regression (2026-09-20)

**Root cause.** `torch.softmax` on XPU returns wrong values (fp32) or NaNs (bf16) for
several last-dimension sizes in `2.13.0+xpu` and `2.14.0+xpu`. The VAE `AttentionBlock`
hits those sizes at production latent resolution, which produced the regional blue noise.
Reproducer: `scripts/debug/xpu_softmax_bug_mre.py` (fails on 2.13.0/2.14.0, all sizes pass
on 2.12.1 and earlier). Bisect across `2.10.0 → 2.14.0` puts the regression between
`2.12.1+xpu` and `2.13.0+xpu`, alongside the bundled oneAPI version jump.

**Fix.** Pin the environment to `torch==2.12.1+xpu` with `torchvision==0.27.1+xpu`
(`scripts/setup_wan22_xpu.sh`). `torchaudio` is dropped: it has no build for torch 2.12
and nothing in the Wan22 path imports it.

**Verification.** `docs/backends/wan22-stage2-vae-decode-debug.md` §7, plus three
end-to-end 81f smoke runs on 2.12.1 (`inference.device=xpu`, `validation.selection_seed=42`):
clean video, `run-result.json` status `complete`, and **byte-identical** `ema` MP4s across
all three runs (`md5 2985664b344b975588f70020668ec66e`), so the XPU path is now
deterministic run-to-run.

**Baseline resource envelope** (81f config, 39-latent-frame rollout, one sample,
`live` + `ema` passes): peak device memory **26.0 GiB of 30.3 GiB** available, peak host
RSS **28.2 GiB of 30 GiB**, wall clock ≈ 5 min. Host RSS is dominated by the CPU fp32
text encoder (~20 GiB); see
[`wan22-xpu-inference-optimization-plan.md`](wan22-xpu-inference-optimization-plan.md).

## Stage 2 Fused Attention Kernels & Pipeline Breakdown (2026-10-06)

### 1. Integration & Runtime Kernel Selection

Stage 2 camera-length inference supports selecting custom Triton attention kernels via
`runtime.stage2_fused_kernel`:

- **`none`**: Unfused sequential PyTorch pipeline (baseline: `complex128` RoPE followed by `torch.einsum` PRoPE and vendor SDPA).
- **`reference`**: In-tree reference package (`solarwm.kernels.fused_rope_prope_sdpa.reference`).
- **`fused_rope_prope_sdpa`**: Custom Triton kernel fusing multi-coordinate RoPE, PRoPE camera transformations, and vendor SDPA (`fused_rope_prope_sdpa_split`).
- **`fused_rope_prope_sage`**: Custom Triton kernel fusing RoPE, PRoPE, and int8 SageAttention (`fused_rope_prope_sage`).

**XPU Triton Lowering & Window Fixes:**
1. **FP32 Camera Matrices:** Triton Intel XPU lowering asserts `inElemTy.isF32()`. Camera intrinsics and extrinsics (`cam_viewmats`, `cam_K`) are explicitly cast to `torch.float32` before kernel launch.
2. **Device Tensors:** Grid size metadata (`q_grid_sizes`, `k_grid_sizes`) are cast to XPU device tensors prior to launch.
3. **KV Cache Window Alignment:** In `CausalWanSelfAttention.forward`, temporal grid dimensions are unconditionally set (`q_grid[:, 0] = num_new_frames`, `k_grid[:, 0] = num_window_frames`). Previously, `k_grid` was clamped to query frames when `grid_list` was present, causing Triton's `_rope_table_kernel` to treat historical KV cache tokens as inactive.

---

### 2. Steady-State Chunk 4 Benchmarks (Intel Arc B580)

Measured across 5 timed repeats at steady-state chunk 4 (4 denoise forward passes + 1 commit forward pass; 12 pixel frames = 3 latent frames):

| Variant | Denoise Fwd Latency | Commit Fwd Latency | Mean Single Fwd | Chunk 4 Diffusion Total | Speedup vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`baseline` (`none`)** | 309.90 ms | 310.03 ms | 309.93 ms | 1,550.56 ms | 1.00× (baseline) |
| **`fused_rope_prope_sdpa`** | 214.13 ms | 214.10 ms | 214.13 ms | 1,071.43 ms | **1.45×** (~479 ms saved / chunk) |
| **`fused_rope_prope_sage`** | 210.33 ms | 210.13 ms | 210.29 ms | 1,052.21 ms | **1.47×** (~498 ms saved / chunk) |

Raw benchmark JSON: `outputs/bench_stage2_fused_attention.json`.

---

### 3. Full 160-Frame Video Quality & Numerical Parity

Full 160-frame autoregressive inference (14 chunks $\times$ 4 NFE self-forcing steps) was validated on B580 against baseline (`outputs/rope-prope-baseline-check/`):

| Comparison | Total Frames | Mean Video PSNR | Min PSNR | Max PSNR | Validation Result |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`reference` vs `baseline`** | 160 | **$\infty$ dB** | $\infty$ dB | $\infty$ dB | **100% bit-exact SHA-256 match** across all 160 frames |
| **`fused_rope_prope_sdpa` vs `baseline`** | 160 | **22.16 dB** | 9.95 dB | 41.65 dB | Clean, artifact-free video (1,006,536 B vs 1,048,490 B baseline) |
| **`fused_rope_prope_sage` vs `baseline`** | 160 | **22.29 dB** | 10.71 dB | 41.52 dB | Clean, artifact-free video (1,001,195 B) |
| **`fused_rope_prope_sage` vs `fused_sdpa`** | 160 | **21.30 dB** | 12.13 dB | 41.56 dB | High mutual trajectory agreement |

Visual inspection across rollout checkpoints (frames 0, 40, 80, 120) confirmed scene geometry, lighting, and textures are preserved without numerical instability.

---

### 4. Per-Chunk Pipeline Runtime Breakdown (12 Pixel Frames)

For one steady-state chunk (4 denoise forward passes + 1 commit forward pass = 5 DiT forwards + 1 streaming VAE decode tile) on Intel Arc B580 using `runtime.stage2_fused_kernel=fused_rope_prope_sdpa`:

| Stage / Component | Runtime | Share of Chunk | Details |
| :--- | :---: | :---: | :--- |
| **1. DiT Fused Attention** | **~390 ms** | **14.1 %** | 5 forwards $\times$ 30 layers:<br>• *Fused Attention Kernel*: **~250 ms** (9.0 %)<br>• *QKV & Out GEMMs*: **~140 ms** (5.1 %) |
| **2. DiT Linear FFN** | **~328 ms** | **11.9 %** | 30 layers of `Linear(3072→13824) → GELU → Linear(13824→3072)` across 5 forwards |
| **3. VAE Decode (1 tile)** | **1,695 ms** | **61.3 %** | 1 streaming decode tile of 3 latent frames $\to$ 12 pixel frames at $480 \times 864$ |
| **4. Others** | **~353 ms** | **12.7 %** | • *Cross-Attention*: **~152 ms** (5.5 %)<br>• *Norms, AdaLN Modulation, Residuals, Head*: **~175 ms** (6.3 %)<br>• *Scheduler / Sampler Math*: **~6 ms** (0.2 %) |
| **Total Chunk Pipeline** | **~2,767 ms** | **100.0 %** | **4.34 generated fps** (up from 3.69 fps baseline; 2.77 s vs 3.25 s) |

#### Per Single Forward Pass (avg across 5 forwards in chunk):
- **Total Single Forward Pass**: **214.1 ms** (down from **309.9 ms** in baseline)
  - DiT Self-Attention (Total): **~78.0 ms** (36.4 % of forward)
    - Fused RoPE + PRoPE + SDPA kernel: ~50.0 ms
    - Q, K, V Linear Projections: ~21.0 ms
    - Out Linear Projection: ~7.0 ms
  - DiT Linear FFN: **~65.7 ms** (30.7 % of forward)
  - DiT Cross-Attention: **~30.4 ms** (14.2 % of forward)
  - DiT Norms, AdaLN Modulation, Residuals, Head: **~35.0 ms** (16.3 % of forward)

#### Optimization Implications:
1. **VAE Decode Bottleneck:** With attention fusion reducing DiT forward latency from 309.9 ms to 214.1 ms, VAE decode now accounts for **61.3 % of total per-chunk time** (1.70 s out of 2.77 s).
2. **Sequential Compute Concurrency:** Because Intel Arc B580 hardware does not overlap concurrent compute streams (measured $1.001\times$ overlap in WS6 microbenchmarks), VAE decode runs serially after diffusion commits. Future performance gains will require accelerating VAE decode (e.g. WS14: bf16 elementwise operations to reduce the 745 ms of fp32 copy/cast overhead).

---

## References

- `environments/README.md` - existing CUDA runtime matrix and installation ordering.
- `docs/backends/wan22-ti2v-5b.md` - released Wan2.2 5B checkpoints and Stage2 inference command.
- `configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml` - one-process Stage2 inference configuration.
- `src/solarwm/backends/wan22/runtime/stage2.py` - Stage2 runtime and adapter.
- `src/solarwm/backends/wan22/runtime/inference.py` - shared Wan inference adapter.
- `src/solarwm/backends/wan22/runtime/distributed.py` - device/process-group setup and cleanup.
- `src/solarwm/backends/wan22/runtime/modeling/attention.py` - FlashAttention and SDPA selection.
- PyTorch XPU documentation: https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html
- PyTorch installation selector: https://pytorch.org/get-started/locally/
