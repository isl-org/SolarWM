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

## Optional private SVDQuant weights

The installer accepts a private repository whose root contains
`svdquant_int4/`:

```bash
python scripts/install_solarwm.py \
  --svdquant-repo your-org/SolarWM-5B-svdquant-int4
```

The repository must be accessible to the Hugging Face account used by `hf`.
The weights are not uploaded by the installer.

## Upload SVDQuant weights to a private Hugging Face repository

These commands create and populate a private model repository. They are
provided for the owner of the weights and are not run by SolarWM:

```bash
export HF_REPO=your-org/SolarWM-5B-svdquant-int4
export MODEL_ROOT=/path/to/SolarWM/models
export SVDQUANT_ROOT="$MODEL_ROOT/SolarWM-5B-sgf-stage2-81f/svdquant_int4"

python -m pip install --upgrade huggingface_hub
hf auth login
hf repo create "$HF_REPO" --repo-type model --private
hf upload \
  "$HF_REPO" \
  "$SVDQUANT_ROOT" \
  svdquant_int4 \
  --repo-type model
```

For a large upload that should be resumable, stage the directory so the
repository has the same layout, then use:

```bash
mkdir -p /tmp/solarwm-svdquant-upload
cp -a "$SVDQUANT_ROOT" /tmp/solarwm-svdquant-upload/
hf upload-large-folder \
  "$HF_REPO" \
  /tmp/solarwm-svdquant-upload \
  --repo-type model
```

After upload, verify the private repository without downloading it:

```bash
hf tree "$HF_REPO" --repo-type model
```
