#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-wan22-xpu}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
# Do not bump without re-validating: 2.13.0+xpu and newer ship an XPU softmax kernel that
# returns wrong values (fp32) or NaNs (bf16) for several last-dimension sizes, which corrupts
# Wan2.2 VAE decode output.
PYTORCH_VERSION="${PYTORCH_VERSION:-2.12.1+xpu}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.27.1+xpu}"
XPU_INDEX_URL="${XPU_INDEX_URL:-https://download.pytorch.org/whl/xpu}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${REPO_ROOT}/environments/wan22-xpu-requirements.txt}"

# Use the Intel proxy by default, while allowing callers to override it.
export https_proxy="${https_proxy:-http://proxy-dmz.intel.com:912}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"

command -v uv >/dev/null 2>&1 || {
    echo "error: uv is required; install it from https://docs.astral.sh/uv/" >&2
    exit 1
}

cd "${REPO_ROOT}"
echo "Creating Python ${PYTHON_VERSION} environment at ${VENV_DIR}"
UV_VENV_CLEAR=1 uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
source ${VENV_DIR}/bin/activate
PYTHON="${VENV_DIR}/bin/python"

uv pip install --upgrade pip setuptools wheel

echo "Installing PyTorch ${PYTORCH_VERSION} from ${XPU_INDEX_URL}"
uv pip install --python "${PYTHON}" \
    "torch==${PYTORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${XPU_INDEX_URL}"

echo "Installing SolarWM and Wan2.2 inference dependencies"
# Do not use -e ".[wan]": that extra installs CUDA FlashAttention.
uv pip install --python "${PYTHON}" -e .
uv pip install --python "${PYTHON}" -r "${REQUIREMENTS_FILE}"

"${PYTHON}" - <<'PY'
import torch

print(f"torch: {torch.__version__}")
print(f"torch.xpu available: {hasattr(torch, 'xpu') and torch.xpu.is_available()}")
if hasattr(torch, "xpu"):
    print(f"torch.xpu device count: {torch.xpu.device_count()}")
print(f"torch.accelerator available: {hasattr(torch, 'accelerator')}")
if hasattr(torch, "accelerator"):
    print(f"current accelerator: {torch.accelerator.current_accelerator()}")
PY

"${PYTHON}" -m solarwm environment probe

echo
echo "Environment ready: ${PYTHON}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
