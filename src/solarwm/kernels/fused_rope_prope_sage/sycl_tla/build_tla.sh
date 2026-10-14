#!/usr/bin/env bash
# Build kernel_tla.sycl into a torch extension (kernel_tla.so) with the same
# flags sycl-tla's cmake/FindDPCPP.cmake uses.
#
#   ./build_tla.sh [out_dir]
#
# Env overrides: CUTLASS_DIR, CUTLASS_API_VERSION (v0.8 or v0.9.3), PY,
# SYCL_DEVICE (default: bmg-g31, the 0xe223 card; a bmg-g21 image does not load
# on it), EXTRA_FLAGS (appended to the compile).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/tmp/kernel_tla_build}"
if [[ -z "${CUTLASS_API_VERSION:-}" ]]; then
  case "${CUTLASS_DIR:-}" in
    *v0.9.3*) CUTLASS_API_VERSION=v0.9.3 ;;
    *v0.8*) CUTLASS_API_VERSION=v0.8 ;;
    "") CUTLASS_API_VERSION=v0.8 ;;
    *)
      echo "set CUTLASS_API_VERSION for custom CUTLASS_DIR: $CUTLASS_DIR" >&2
      exit 1
      ;;
  esac
fi
case "$CUTLASS_API_VERSION" in
  v0.8)
    CUTLASS_DIR="${CUTLASS_DIR:-$HOME/.cache/sycl-tla-v0.8-pin}"
    CUTLASS_API_FLAGS=(-DCUTLASS_TLA_V08)
    ;;
  v0.9.3)
    CUTLASS_DIR="${CUTLASS_DIR:-$HOME/.cache/sycl-tla-v0.9.3}"
    CUTLASS_API_FLAGS=()
    ;;
  *)
    echo "unsupported CUTLASS_API_VERSION: $CUTLASS_API_VERSION (expected v0.8 or v0.9.3)" >&2
    exit 1
    ;;
esac
PY="${PY:-$HOME/ternacy_opt_env/bin/python}"
SYCL_DEVICE="${SYCL_DEVICE:-bmg-g31}"
SPIRV_EXT="+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"

set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true; set -u
mkdir -p "$OUT"

# The non-causal FMHA path needs the k_blocks round-up from patches/.
grep -q 'k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}))' \
  "$CUTLASS_DIR/applications/flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp" ||
  { echo "apply patches/0001-fmha-round-up-noncausal-kv-tiles.patch to $CUTLASS_DIR first" >&2; exit 1; }

read -r TORCH_DIR PY_INC < <("$PY" -c "import torch, os, sysconfig; print(os.path.dirname(torch.__file__), sysconfig.get_paths()['include'])")

SYCL_FLAGS=(-fsycl -fsycl-targets=spir64_gen -fno-sycl-instrument-device-code)

icpx "${SYCL_FLAGS[@]}" -std=c++20 -O3 -fPIC -c -x c++ "$HERE/kernel_tla.sycl" -o "$OUT/kernel_tla.sycl.o" \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
  -DTORCH_EXTENSION_NAME=kernel_tla -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
  "${CUTLASS_API_FLAGS[@]}" \
  -I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" -I"$CUTLASS_DIR/applications" \
  -isystem "$TORCH_DIR/include" -isystem "$TORCH_DIR/include/torch/csrc/api/include" -isystem "$PY_INC" \
  -Wno-deprecated-declarations ${EXTRA_FLAGS:-}

icpx "${SYCL_FLAGS[@]}" -shared "$OUT/kernel_tla.sycl.o" -o "$OUT/kernel_tla.so" \
  -Xsycl-target-backend=spir64_gen "-device $SYCL_DEVICE" \
  -Xspirv-translator "-spirv-ext=$SPIRV_EXT" \
  -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" -lc10 -lc10_xpu -ltorch -ltorch_cpu -ltorch_xpu -ltorch_python

echo "built $OUT/kernel_tla.so for $SYCL_DEVICE with CUTLASS $CUTLASS_API_VERSION"
