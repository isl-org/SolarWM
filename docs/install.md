# Install SolarWM

`scripts/install_solarwm.py` installs the Wan2.2 runtime into the active
virtual environment, detects the already-installed PyTorch device, downloads
the released Wan2.2-5B Stage2 inference assets from Hugging Face, and can
configure an existing ComfyUI installation. It works on Windows, Linux, and
macOS; `torch` must be installed before running it.

## Install

Activate the environment containing the desired PyTorch build, then run:

```text
python scripts/install_solarwm.py
```

The script uses the upstream model repository `junchaoh-cs/SolarWM` and stores
weights in `SolarWM/models` by default. It never installs or changes PyTorch.
Run `python -m pip install torch` or the appropriate vendor command separately
first.

Use `--local-dir` to override the model directory:

```text
python scripts/install_solarwm.py --local-dir /path/to/SolarWM-models
```

The detected device selects the dependency set:

- CUDA installs the Wan extra and FlashAttention using the installed torch.
- Intel XPU installs `[wan-xpu]`, which excludes CUDA-only FlashAttention.
- CPU installs the portable Wan dependency set.

Use `--skip-weights` when the model directory is already populated. Weight
downloads are resumable and skipped entirely when the required files are
already present. Set `HF_TOKEN` in the environment or authenticate with `hf
auth login` before downloading a gated or private repository. The installer
also writes a local interactive YAML config with `model.base_path` and
`checkpoint.path` pointing at the selected model directory.

On an Intel XPU machine, pass `--build-kernels` to build the optional
SVDQuant SYCL extension and load/build the oneDNN qconv extension. The
SVDQuant build uses `--gpu-arch bmg-g31` by default; override it when needed.
The step is idempotent for an existing SVDQuant native extension:

```text
python scripts/install_solarwm.py --build-kernels --gpu-arch bmg-g31
```

The installer keeps any package already present in the active environment,
regardless of its version. It has no upgrade, sync, or uninstall step; install
or resolve incompatible existing versions separately if needed.

## Configure ComfyUI

Pass the ComfyUI installation directory to add SolarWM's integration and
model search paths to its `extra_model_paths.yaml`:

```text
python scripts/install_solarwm.py \
  --comfyui /path/to/ComfyUI
```

The default `SolarWM/models` directory is used for both the ComfyUI search paths
and the generated SolarWM config. The update is idempotent and keeps other
top-level ComfyUI path entries. Restart ComfyUI after the installer finishes.
ComfyUI must run with the environment
created by this installer, or with SolarWM installed into its own Python
environment, because the custom node imports the `solarwm` package.

## SVDQuant INT4 weights

SVDQuant quantized weights are hosted as GitHub release assets on `isl-org/SolarWM` under the `ptc` release tag. Each noise-level checkpoint corresponds to a single `.safetensors` file (~1.83 GiB / 1.96 GB), strictly complying with GitHub's 2 GiB per-asset limit without arbitrary sharding:

- **k1** (1 checkpoint, `set0`): single file `svdquant_weights.safetensors` (~1.83 GiB)
- **k2** (2 checkpoints, `set0` & `set1`): two files `svdquant_weights_set0.safetensors` and `svdquant_weights_set1.safetensors` (~1.83 GiB each)
- **k4** (4 checkpoints, `set0` .. `set3`): four files `svdquant_weights_set0.safetensors` through `svdquant_weights_set3.safetensors` (~1.83 GiB each)

By default, running `scripts/install_solarwm.py` downloads the `sdpa_k1` variant automatically:

```bash
# Downloads base weights, stage2 checkpoint, and sdpa_k1 SVDQuant weights
python scripts/install_solarwm.py

# Download a specific variant (e.g. sdpa_k2 or sdpa_k4)
python scripts/install_solarwm.py --svdquant-variant sdpa_k2

# Skip SVDQuant weight download
python scripts/install_solarwm.py --svdquant-variant none
```

The installer downloads the variant's `manifest.json` and all shards into:
`models/SolarWM-5B-sgf-stage2-81f/svdquant_int4/{variant}/`

### Uploading SVDQuant weights to the GitHub release

To upload a newly calibrated or exported SVDQuant variant to the release:

```bash
VARIANT=sdpa_k2
SRC_DIR=models/SolarWM-5B-sgf-stage2-81f/svdquant_int4/$VARIANT
STAGE_DIR=/tmp/svdquant_upload

mkdir -p "$STAGE_DIR"
cp "$SRC_DIR/manifest.json" "$STAGE_DIR/${VARIANT}_manifest.json"
for f in "$SRC_DIR"/*.safetensors; do
  ln -f "$f" "$STAGE_DIR/${VARIANT}_$(basename "$f")"
done

gh release upload ptc "$STAGE_DIR"/* --clobber -R isl-org/SolarWM
```

After upload, verify the private repository without downloading it:

```bash
hf tree "$HF_REPO" --repo-type model
```
# Windows XPU Triton prerequisites

The weights download step also runs `scripts/prepare_stage2_ema.py` to create
`SolarWM-5B-sgf-stage2-81f-ema-bf16`, a separate EMA-only BF16 inference
transaction. The original live+EMA FP32 release is retained. Local interactive
configs select the derived checkpoint when present. This reduces the Stage2
file from approximately 37.25 GiB to 9.31 GiB; BF16 conversion matches the
existing inference model dtype. It does not reduce runtime activation or KV-cache
requirements. The processor uses a read-only file mapping to avoid Windows'
whole-file private-mapping commit reservation and writes updated manifests.

On Windows XPU installations, `scripts/install_solarwm.py` also downloads the
official Level Zero SDK 1.34.0, checks its pinned SHA-256, and extracts it under
the active virtual environment's `level-zero-sdk` directory. A venv-local
`solarwm_triton_environment.pth` startup hook sets `LEVEL_ZERO_V1_SDK_PATH`
(preserving an existing setting) and initializes an installed MSVC x64 toolset
when no compiler is already configured. Restart ComfyUI after installation.
This does not install MSVC: install Visual Studio C++ Build Tools and the
Windows SDK first. Explicit `CXX` and existing compiler environments are preserved.

The PyTorch XPU dependency `intel-sycl-rt` supplies SYCL development files under
`.venv\Library\include\sycl\sycl.hpp` and `.venv\Library\lib\sycl.lib` on Windows,
outside `site-packages`. Triton discovers these through wheel metadata. If the
files are missing, restore the matching `intel-sycl-rt` wheel; alternatively,
install the matching oneAPI DPC++ component and initialize its environment.
Do not set `ONEAPI_ROOT` to a nonexistent toolkit, since that overrides Triton's
wheel-based discovery.
# SVDQuant checkpoint layout

Quantized weights are an overlay on the Stage2 `.pt` checkpoint, not a
standalone model. Set `model.svdquant_weights_root` to the directory containing
`sdpa_k1`, `sdpa_k2`, `sdpa_k4`, or the matching `sageattnv1` variants.
Each variant accepts either `manifest.json` and manifest-declared shard names,
or GitHub release filenames prefixed with the variant (for example
`sdpa_k1_manifest.json` and `sdpa_k1_svdquant_weights.safetensors`).
The installer and runtime validate every declared shard, tensor-to-shard mapping,
and set inventory. Keep all k4 shards, including set2. Unquantized parameters
still come from the release-selected EMA state in the original or processed
EMA-only BF16 `.pt` transaction.
