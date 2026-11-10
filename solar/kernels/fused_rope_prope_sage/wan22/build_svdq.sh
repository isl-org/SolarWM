#!/usr/bin/env bash
# Build sycl_tla_opt.sycl into a torch extension (sycl_tla_opt.so, Python module
# sycl_tla_opt) with the same flags sycl-tla's cmake/FindDPCPP.cmake uses.
#
#   ./build.sh [out_dir] [source_file] [extension_name]
#
# Env overrides: CUTLASS_DIR (a sycl-tla v0.8 checkout; the kernel's epilogue
# call does not match v0.9.x), PY, SYCL_DEVICE (default: bmg-g31, the 0xe223
# card; a bmg-g21 image does not load on it), EXTRA_FLAGS (appended to the
# compile).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/build}"
SOURCE="${2:-$HERE/sycl_tla_opt.sycl}"
NAME="${3:-sycl_tla_opt}"
if [[ -z "${CUTLASS_DIR:-}" ]]; then
  if [[ -f "$HERE/third_party/sycl-tla-v0.8/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp" ]]; then
    CUTLASS_DIR="$HERE/third_party/sycl-tla-v0.8"
  else
    CUTLASS_DIR="$HOME/.cache/sycl-tla-v0.8-pin"
  fi
fi
if [[ -z "${PY:-}" ]]; then
  if [[ -x "$HOME/ternacy_opt_env/bin/python" ]]; then
    PY="$HOME/ternacy_opt_env/bin/python"
  else
    PY="python3"
  fi
fi
SYCL_DEVICE="${SYCL_DEVICE:-bmg-g31}"
SPIRV_EXT="+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"

true ||
  { echo "CUTLASS_DIR=$CUTLASS_DIR is not a sycl-tla v0.8 checkout" >&2; exit 1; }
[[ -f "$SOURCE" ]] || { echo "Kernel source not found: $SOURCE" >&2; exit 1; }

set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true; set -u
mkdir -p "$OUT"

read -r TORCH_DIR PY_INC < <("$PY" -c "import torch, os, sysconfig; print(os.path.dirname(torch.__file__), sysconfig.get_paths()['include'])")

SYCL_FLAGS=(-fsycl -fsycl-targets=spir64_gen -fno-sycl-instrument-device-code)

icpx "${SYCL_FLAGS[@]}" -std=c++20 -O3 -fPIC -c -x c++ "$SOURCE" -o "$OUT/$NAME.sycl.o" \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
  -DTORCH_EXTENSION_NAME=$NAME -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" -I"$CUTLASS_DIR/applications" \
  -isystem "$TORCH_DIR/include" -isystem "$TORCH_DIR/include/torch/csrc/api/include" -isystem "$PY_INC" \
  -Wno-deprecated-declarations ${EXTRA_FLAGS:-}

icpx "${SYCL_FLAGS[@]}" -shared "$OUT/$NAME.sycl.o" -o "$OUT/$NAME.so" \
  -Xsycl-target-backend=spir64_gen "-device $SYCL_DEVICE" \
  -Xspirv-translator "-spirv-ext=$SPIRV_EXT" \
  -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" -lc10 -lc10_xpu -ltorch -ltorch_cpu -ltorch_xpu -ltorch_python

echo "built $OUT/$NAME.so for $SYCL_DEVICE"
