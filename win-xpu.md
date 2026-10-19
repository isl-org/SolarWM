# Windows XPU port: live status

Last updated: 2026-10-19.

**Status:** installation, custom kernel builds, and selected kernel tests work.
The reduced EMA checkpoint loads in a real interactive session. End-to-end
interactive video generation is not yet verified.

## Host and target environment

- Native Windows 11 x64; Python 3.12.10.
- ComfyUI Python environment:
  `C:\Users\gta\AppData\Local\Programs\AI Playground\resources\ComfyUI\.venv`.
- SolarWM checkout: `C:\Users\gta\Documents\SolarWM`.
- PyTorch `2.13.0+xpu`, Triton XPU `3.7.2`, oneAPI Toolkit `2026.0.0`.
- Active GPU: Intel Arc Pro B70, reported capacity approximately 27.25 GiB.
  Earlier testing used an Arc Pro B65; it subsequently appeared as an inactive
  PnP device. Do not assume those two device identities are interchangeable.
- Host RAM approximately 31.5 GiB. Windows commit headroom varies with other
  applications; it was approximately 36.95 GiB during checkpoint diagnostics.
- VS 2022 C++ Build Tools, Windows SDK, and VS Code Python/C++/CMake extensions
  were installed. VS Code is not required for runtime inference.

## Work completed

### Toolchain and runtime

1. Created a separate Torch development venv and installed the exact XPU wheel.
   Verified GPU discovery, XPU matmul, a standalone SYCL kernel, and a Triton
   vector-add kernel.
2. Installed oneAPI 2026.0 and MSVC Build Tools. In this release, oneAPI Toolkit
   replaces the separate Base/HPC installers.
3. Diagnosed proxy routing: the environment uses
   `HTTPS_PROXY=http://proxy-dmz.intel.com:912`, while `.intel.com` in
   `NO_PROXY` can bypass the proxy for Intel downloads. Windows installers can
   use WinINet rather than these environment variables.
4. Diagnosed compiler temporary-file failures. A dedicated writable compiler
   temporary directory worked where the existing directory failed.

### SolarWM and ComfyUI

1. Installed SolarWM editable into the ComfyUI venv using
   `scripts/install_solarwm.py` and `uv`; this venv initially had no pip.
2. Configured `extra_model_paths.yaml` and linked ComfyUI's
   `custom_nodes\SolarWM` to `integrations\comfyui_solarwm`.
3. Fixed POSIX-rooted path handling in the integration on Windows.
4. The existing ComfyUI integration suite passed all 10 tests. ComfyUI's own
   `nodes.load_custom_node` loaded and registered `SolarWMInteractive`.
5. The running AI Playground API required authentication (HTTP 401).
   The active server was left untouched; live browser/API operation was not
   verified by these tests. Seeing `C:\Python312\python.exe` as a child process
   alone is not proof that a venv launcher has lost its environment.

### Weights

1. Downloaded only the installer's Wan2.2-5B base-conditioning, text-encoder,
   tokenizer, VAE, and Stage2 release files: 17 model files, approximately
   50.48 GiB. The base transformer shards were not needed for this Stage2 path.
2. Hugging Face access initially failed with 401; the updated local token was
   accepted. Tokens must remain local and excluded from Git.
3. Diagnosed the original `model.pt`: approximately 37.25 GiB, containing live
   and EMA FP32 states of approximately 18.63 GiB each.
4. Added `scripts/prepare_stage2_ema.py`. It preserves the original release and
   writes a separate verified EMA-only BF16 inference transaction:
   `models\SolarWM-5B-sgf-stage2-81f-ema-bf16\model.pt`, approximately 9.31 GiB.
   All 825 EMA tensors are retained; live weights are omitted. BF16 matches
   the inference model's existing dtype, but is not an FP32 archival copy.
5. Wired processing into the download step and made local interactive config
   select the derived transaction. Stage2 validation now accepts EMA-only
   metadata as well as the original live+EMA transaction.

### Kernels and minimal Triton setup

1. Downloaded Intel SYCL TLA v0.9.3 explicitly:
   `https://github.com/intel/sycl-tla.git`, commit
   `7b56b936b87f799f80dde60c2fd8d11c1128cebb`.
   Installed `kernelfoundry` for the SVDQuant build harness.
2. Built and loaded `svdquant_dense_kernel_grf128_n_tile64.pyd` for `bmg-g31`.
   Loading succeeded; full SVDQuant numerical parity and model inference with
   quantized weights have not been established by this work.
3. Installed the matching toolkit oneDNN component, `intel.oneapi.win.dnnl`,
   retaining the existing toolkit components. PyTorch wheels are not a
   replacement for this custom extension's oneDNN development library.
4. Built `qconv_onednn_ops.cp312-win_amd64.pyd`. Fixed Windows linking to
   Python's import library and registered the oneDNN runtime DLL directory.
   A small custom SYCL qconv numerical check passed.
5. SolarWM Triton LayerNorm/AdaLN and RMSNorm parity tests passed six XPU cases.
6. Added a checksum-verified Level Zero SDK 1.34.0 download to the installer,
   extracted under the active venv. The installed `intel-sycl-rt` wheel already
   supplies SYCL headers under `.venv\Library\include\sycl` and `sycl.lib`
   under `.venv\Library\lib`, not under `site-packages`.
7. Added a venv-local `.pth` startup hook that sets the Level Zero SDK path and
   initializes installed MSVC tools when no compiler is configured.
   Fresh-cache Triton RMSNorm compilation passed with MSVC, no icpx on PATH,
   no `ONEAPI_ROOT`, and the downloaded Level Zero SDK.

Prebuilt custom SYCL binaries do not require SYCL TLA or compiler headers at
runtime. Triton's native helper JIT still needs a host compiler, Python
development files, and SYCL/Level Zero development files. A warmed cache is not
a portable substitute for those prerequisites.

## Failures and current blockers

### Original checkpoint private mapping

`torch.load(..., weights_only=True, mmap=True)` loaded metadata but accessing
tensor data caused Windows `0xC0000005` access violations. This reproduced on
the first tensor without constructing a model or transferring to GPU.

The first ZIP storage passed its CRC check; ordinary buffer CPU/BF16/XPU
copies passed. Python read-only mmap could read the complete file, including
offsets beyond 32 GiB. PyTorch's full-size private storage mapping was invalid,
whereas smaller private mappings and a full-size shared mapping worked.
Insufficient Windows commit for a whole-file copy-on-write mapping is the
leading explanation, not a proven general PyTorch size limit.

The processor uses a read-only mapping, never a writable shared mapping of the
original release. The 9.31 GiB derived checkpoint loaded and the default
interactive session reached `ready`, resolving that weight-loading failure in
the observed run. An intermediate EMA-only FP32 file was insufficient once
the text encoder and model were resident.

### Interactive generation

- At 480x864 with three-latent blocks, the real BF16 EMA session began
  inference but emitted an XPU out-of-memory error before decoded frames.
  The message reported a failed 204 MiB allocation despite approximately
  8.14 GiB free; investigate allocator/driver limits rather than treating the
  free-memory figure as proof the allocation should succeed.
- A 256x448, one-latent diagnostic override also ended in a native access
  violation. Its failure location was not isolated. This was not a successful
  low-resolution run, and the normal saved resolution/block size was preserved.
- No successful generated video or full browser interaction is claimed.
- Temporary diagnostic loading monkeypatches were removed; they are not part
  of the installed runtime.

### Validation and reproducibility gaps

- Processing and ComfyUI tests: 11 passed after EMA processing changes.
- Startup-helper and ComfyUI tests: 13 passed at the SDK setup stage.
- Broader Stage2 tests: 33 passed, 2 failed on Windows directory `os.open/fsync`
  in `CheckpointTransaction.commit`; these are separate checkpoint-publication
  portability failures.
- The installer does not yet automatically fetch SYCL TLA or install
  KernelFoundry/oneDNN for `--build-kernels`. These were provisioned explicitly.
- Existing package versions are kept even if different from pinned requested
  versions. This host has diffusers 0.41.0 and transformers 5.20.1 rather than
  the release-tested 0.38.0 and 5.12.1. Test a controlled compatible environment
  before attributing remaining failures solely to the GPU driver.

## Next experiments

1. Reproduce generation with faulthandler and phase-level memory measurements:
   checkpoint load, anchor encoding, KV allocation, each denoise forward,
   cache commit, and VAE decode. Capture the reduced-resolution crash location.
2. Construct the inference transformer directly in BF16 through a supported
   initialization path, rather than relying on a process-global default dtype.
   Preserve required FP32 numerical state and avoid unnecessary host allocations.
3. Investigate XPU allocation limits/fragmentation and available commit under
   the real workload. Compare one- versus three-latent blocks without changing
   the model's trained block-size default.
4. Test the release-tested diffusers/transformers combination in a separate
   environment, preserving AI Playground's shared dependencies.
5. Evaluate SVDQuant after obtaining or generating its separate quantized
   weights; merely compiling its extension does not enable the quantized path.
6. Add processor tests for malformed source/EMA data, interrupted publication,
   source identity changes, and repeated installation. Validate derived
   manifests and runtime contract identity consistently.
7. Fix Windows checkpoint-directory durability handling and rerun the broader
   checkpoint tests. Do not silently claim POSIX directory fsync on Windows.
8. Finish prerequisite provisioning and strict build/load checks in the
   installer, then test on a clean machine without the development venv.
9. Acceptance target: a real interactive session produces finite decoded
   frames and a playable MP4, followed by authenticated ComfyUI viewer testing.

## Local commands

Run with ComfyUI's venv Python:

```powershell
$python = "C:\Users\gta\AppData\Local\Programs\AI Playground\resources\ComfyUI\.venv\Scripts\python.exe"
& $python .\scripts\install_solarwm.py --comfyui "C:\Users\gta\AppData\Local\Programs\AI Playground\resources\ComfyUI"
& $python -m pytest tests\integrations\test_prepare_stage2_ema.py tests\integrations\test_comfyui_solarwm.py -q
& $python -m pytest tests\kernels\test_dit_fused_ops.py -k "triton_layernorm_adaln_parity or triton_rmsnorm_parity" -q
```

Hugging Face access must be configured locally for gated downloads. Do not
commit tokens, weights, SDK downloads, local configs, or compiled binaries.
