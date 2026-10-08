#!/usr/bin/env bash
set -euo pipefail

# VTune GPU-Hotspots ROI profiling for the Wan2.2 Stage2 standalone camera
# rollout on Intel XPU, adapted from .agents/skills/vtune/SKILL.md (ignore the
# vLLM-specific parts of that skill -- this is a single-process batch job,
# not a multi-process server, so there is no outer `vtune -command
# resume/pause` bracket and no HTTP health-check loop).
#
# ROI strategy: `scripts/debug/run_stage2_vtune_roi.py` monkeypatches
# `_Stage2XpuProfiler.begin`/`.end` (normally a torch.profiler capture gated
# on runtime.stage2_xpu_profiler_start_chunk/end_chunk) to call
# `vtune_itt.itt_resume()`/`itt_pause()` at the same chunk boundaries instead.
# VTune itself is launched with `-start-paused`, so everything before the
# requested chunk window (model load, warmup chunks 0-3) is excluded from the
# GPU trace, and only the requested chunk window is collected.
#
# Usage:
#   scripts/debug/run_wan22_stage2_vtune.sh
#   START_CHUNK=4 END_CHUNK=5 scripts/debug/run_wan22_stage2_vtune.sh

VTUNE_BIN_DIR="${VTUNE_BIN_DIR:-/opt/intel/oneapi/vtune/latest/bin64}"
[[ -d "$VTUNE_BIN_DIR" ]] || VTUNE_BIN_DIR="/opt/intel/oneapi/vtune/2025.10/bin64"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# This VTune install ships only a static libittnotify.a (no unversioned
# libittnotify.so), and __itt_resume/__itt_pause are C macros with no
# linkable symbol of that name in the archive anyway (see vtune_itt.py's
# comment). scripts/debug/run_stage2_vtune_roi.py needs a real .so exporting
# callable wrappers; build one on first use from the SDK's static archive.
VTUNE_ITTNOTIFY_SHIM_DIR="${VTUNE_ITTNOTIFY_SHIM_DIR:-$REPO_ROOT/outputs/.ittnotify_shim}"
export VTUNE_ITTNOTIFY_SHIM="$VTUNE_ITTNOTIFY_SHIM_DIR/libittnotify_shim.so"
METRICS_SHIM_DIR="${METRICS_SHIM_DIR:-$REPO_ROOT/outputs/.metrics_discovery_shim}"
METRICS_SHIM="$METRICS_SHIM_DIR/libigdmd.so"
build_ittnotify_shim() {
  local vtune_sdk_include vtune_lib64
  vtune_sdk_include=$(dirname "$VTUNE_BIN_DIR")/sdk/include
  vtune_lib64=$(dirname "$VTUNE_BIN_DIR")/lib64/libittnotify.a
  [[ -f "$vtune_sdk_include/ittnotify.h" && -f "$vtune_lib64" ]] || return 1
  mkdir -p "$VTUNE_ITTNOTIFY_SHIM_DIR"
  cat > "$VTUNE_ITTNOTIFY_SHIM_DIR/shim.c" <<'SHIM_EOF'
#include <ittnotify.h>
void solarwm_itt_resume(void) { __itt_resume(); }
void solarwm_itt_pause(void)  { __itt_pause(); }
__itt_domain* solarwm_itt_domain_create(const char* name) {
    return __itt_domain_create(name);
}
__itt_string_handle* solarwm_itt_string_handle_create(const char* name) {
    return __itt_string_handle_create(name);
}
void solarwm_itt_task_begin(__itt_domain* d, __itt_string_handle* h) {
    __itt_task_begin(d, __itt_null, __itt_null, h);
}
void solarwm_itt_task_end(__itt_domain* d) {
    __itt_task_end(d);
}
SHIM_EOF
  gcc -shared -fPIC -I"$vtune_sdk_include" \
      -o "$VTUNE_ITTNOTIFY_SHIM" "$VTUNE_ITTNOTIFY_SHIM_DIR/shim.c" \
      "$vtune_lib64" -lpthread -ldl
}

build_metrics_discovery_shim() {
  local installed installed_md
  installed=$(ldconfig -p 2>/dev/null | awk '/libigdmd\.so\.[0-9]/{print $NF; exit}')
  [[ -n "$installed" && -f "$installed" ]] || return 1
  mkdir -p "$METRICS_SHIM_DIR"
  ln -sfn "$installed" "$METRICS_SHIM"
  installed_md=$(ldconfig -p 2>/dev/null | awk '/libmd\.so\.[0-9]/{print $NF; exit}')
  if [[ -n "$installed_md" && -f "$installed_md" ]]; then
    ln -sfn "$installed_md" "$METRICS_SHIM_DIR/libmd.so"
  fi
}

VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-wan22-xpu}"
SOLAR_MODEL_ROOT="${SOLAR_MODEL_ROOT:-/home/ssheorey/models/SolarWM}"
SOLAR_DATA_ROOT="${SOLAR_DATA_ROOT:-/home/ssheorey/data/SolarWM-Data/releases-v1}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml}"
BASE_PATH="$SOLAR_MODEL_ROOT/SolarWM-5B-base"
CHECKPOINT_PATH="$SOLAR_MODEL_ROOT/SolarWM-5B-sgf-stage2-81f"
DATA_INDEX_ROOT="$SOLAR_DATA_ROOT/example"
START_CHUNK="${START_CHUNK:-4}"
END_CHUNK="${END_CHUNK:-4}"
STAGE2_FUSED_KERNEL="${STAGE2_FUSED_KERNEL:-fused_rope_prope_sage}"
STAGE2_VAE_INT8_QUAROT="${STAGE2_VAE_INT8_QUAROT:-true}"
STAGE2_COMPILE_BLOCKS="${STAGE2_COMPILE_BLOCKS:-0}"
STAGE2_COMPILE_MODE="${STAGE2_COMPILE_MODE:-max-autotune}"
STAGE2_VAE_COMPILE="${STAGE2_VAE_COMPILE:-0}"
STAGE2_VAE_COMPILE_MODE="${STAGE2_VAE_COMPILE_MODE:-max-autotune}"
EXIT_AFTER_ROI="${EXIT_AFTER_ROI:-0}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-5}"
ANALYSIS="${ANALYSIS:-gpu-hotspots}"
FUSE_ROPE_PROPE="${FUSE_ROPE_PROPE:-0}"
START_PAUSED="${START_PAUSED:-1}"

RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/outputs/vtune_results}"
STAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR=$(realpath -m "$RESULT_ROOT")/stage2_chunk${START_CHUNK}-${END_CHUNK}_${STAMP}
OUTPUT_DIR="$REPO_ROOT/outputs/vtune-stage2-${STAMP}"

# ---- Pre-flight (SKILL.md section 9 + the vLLM script's non-vLLM checks) ----
preflight() {
  local fail=0

  if [[ -x "$VTUNE_BIN_DIR/vtune" ]]; then
    echo "  [ OK ] vtune: $VTUNE_BIN_DIR/vtune ($($VTUNE_BIN_DIR/vtune --version 2>&1 | head -1))"
  else
    echo "  [FAIL] vtune not found at $VTUNE_BIN_DIR/vtune (set VTUNE_BIN_DIR)"
    fail=1
  fi

  if [[ -f "$VTUNE_ITTNOTIFY_SHIM" ]] || build_ittnotify_shim; then
    echo "  [ OK ] ITT shim: $VTUNE_ITTNOTIFY_SHIM"
  else
    echo "  [FAIL] could not build ITT shim from the VTune SDK static archive"
    fail=1
  fi

  # VTune manages ZE tracing itself; a manually-set tracing layer causes
  # empty GPU timelines (SKILL.md section 9.1).
  local bad
  bad=$(env | grep -E '^(ZE_ENABLE_TRACING|ZE_LOADER_LAYERS|PTI_ENABLE)=' || true)
  if [[ -n "$bad" ]]; then
    echo "  [FAIL] Conflicting env vars set: $bad"
    fail=1
  else
    echo "  [ OK ] No ZE/PTI tracing conflicts"
  fi

  if command -v sycl-ls >/dev/null 2>&1 && sycl-ls 2>/dev/null | grep -qi '\[level_zero:gpu\]'; then
    echo "  [ OK ] sycl-ls reports a Level Zero GPU"
  else
    echo "  [WARN] sycl-ls did not report a level_zero:gpu device (continuing)"
  fi

  if build_metrics_discovery_shim; then
    echo "  [ OK ] libigdmd.so shim: $METRICS_SHIM"
  else
    echo "  [FAIL] libigdmd.so not on the loader path -- apt install intel-metrics-discovery"
    fail=1
  fi

  local free_gb
  free_gb=$(df -PB1G "$RESULT_ROOT" 2>/dev/null | awk 'NR==2 {print $4+0}')
  if [[ -n "$free_gb" && "$free_gb" -ge "$MIN_FREE_DISK_GB" ]]; then
    echo "  [ OK ] Free disk: ${free_gb} GB"
  else
    echo "  [FAIL] Free disk ${free_gb:-?} GB < ${MIN_FREE_DISK_GB} GB"
    fail=1
  fi

  for required_path in "$VENV_DIR/bin/python" "$CONFIG_PATH" \
      "$BASE_PATH/text_encoder/models_t5_umt5-xxl-enc-bf16.pth" "$BASE_PATH/tokenizer" \
      "$BASE_PATH/vae/Wan2.2_VAE.pth" "$CHECKPOINT_PATH/model.pt" \
      "$DATA_INDEX_ROOT/smoke-index.jsonl.gz" \
      "$REPO_ROOT/scripts/debug/run_stage2_vtune_roi.py" \
      "$REPO_ROOT/scripts/debug/vtune_itt.py"; do
    if [[ -e "$required_path" ]]; then
      echo "  [ OK ] $required_path"
    else
      echo "  [FAIL] missing: $required_path"
      fail=1
    fi
  done

  return $fail
}

mkdir -p "$RESULT_ROOT" "$OUTPUT_DIR"
echo "===== Pre-flight checks ====="
if ! preflight; then
  echo "Pre-flight checks FAILED. Refusing to start."
  exit 1
fi
echo
echo "Result dir: $RESULT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Chunk window: [$START_CHUNK, $END_CHUNK]"
echo "VTune analysis: $ANALYSIS"
mkdir -p "$RESULT_DIR"

cat > "$RESULT_DIR/metadata.json" <<META
{
  "start_chunk": $START_CHUNK,
  "end_chunk": $END_CHUNK,
  "config": "$CONFIG_PATH",
  "started_at": "$(date -Iseconds)"
}
META

# ---- Launch ----
# IMPORTANT: do NOT source oneAPI's setvars.sh into this shell -- it prepends
# oneAPI's own libsycl/libur onto LD_LIBRARY_PATH, which is ABI-incompatible
# with the wheels bundled in $VENV_DIR and breaks `import torch` with
# "undefined symbol ...LIBUR_LOADER_0.12". Only vtune's own bin dir is added
# to PATH; vtune finds its runtime libraries via its own rpath, not
# LD_LIBRARY_PATH, and the smoke test in this task's write-up confirmed GPU
# hardware metrics collect fine this way on this host (BMG dGPU, VTune
# 2025.10 -- contrary to the vLLM skill's warning about that combination,
# which is evidently specific to some other configuration).
echo
VTUNE_KNOBS=(-knob collect-programming-api=true)
if [[ "$ANALYSIS" == "gpu-offload" ]]; then
  # gpu-hotspots has no host-stack collection facility. gpu-offload records
  # the Python/native host stack that submitted each GPU task, so its report
  # can connect a kernel family back to echorope_apply/PRoPE call sites.
  VTUNE_KNOBS+=(
    -knob enable-stack-collection=true
    -knob enable-tasks-stack-collection=true
  )
elif [[ "$ANALYSIS" == "hotspots" ]]; then
  # The installed Metrics Discovery package lacks the unversioned libmd.so
  # required by VTune's gpu-offload collector. CPU Hotspots remains the
  # portable stack collector for this Python process; pair it with the
  # gpu-hotspots run for GPU metrics/kernel attribution.
  VTUNE_KNOBS=(
    -knob sampling-mode=sw
    -knob enable-stack-collection=true
  )
elif [[ "$ANALYSIS" != "gpu-hotspots" ]]; then
  echo "Unsupported ANALYSIS=$ANALYSIS (expected gpu-hotspots, gpu-offload, or hotspots)" >&2
  exit 2
fi

FUSE_SET=()
if [[ "$FUSE_ROPE_PROPE" == "1" || "$FUSE_ROPE_PROPE" == "true" ]]; then
  FUSE_SET=(--set "runtime.stage2_fuse_rope_prope=true")
fi

EXTRA_SET=()
if [[ -n "$STAGE2_FUSED_KERNEL" && "$STAGE2_FUSED_KERNEL" != "none" ]]; then
  EXTRA_SET+=(--set "runtime.stage2_fused_kernel=$STAGE2_FUSED_KERNEL")
fi
if [[ "$STAGE2_VAE_INT8_QUAROT" == "1" || "$STAGE2_VAE_INT8_QUAROT" == "true" ]]; then
  EXTRA_SET+=(--set "runtime.stage2_vae_int8_quarot=true")
fi
if [[ "$STAGE2_COMPILE_BLOCKS" == "1" || "$STAGE2_COMPILE_BLOCKS" == "true" ]]; then
  EXTRA_SET+=(--set "inference.stage2_xpu_compile_blocks=true" --set "inference.stage2_xpu_compile_mode=$STAGE2_COMPILE_MODE")
fi
if [[ "$STAGE2_VAE_COMPILE" == "1" || "$STAGE2_VAE_COMPILE" == "true" ]]; then
  EXTRA_SET+=(--set "runtime.stage2_vae_compile=true" --set "runtime.stage2_vae_compile_mode=$STAGE2_VAE_COMPILE_MODE")
fi

VTUNE_START_PAUSED=()
if [[ "$START_PAUSED" == "1" || "$START_PAUSED" == "true" ]]; then
  VTUNE_START_PAUSED=(-start-paused)
fi

echo "===== Launching vtune -collect $ANALYSIS ${VTUNE_START_PAUSED[*]} ====="
env -i HOME="$HOME" USER="${USER:-}" \
    PATH="$VTUNE_BIN_DIR:/usr/bin:/bin" \
    VTUNE_ITTNOTIFY_SHIM="$VTUNE_ITTNOTIFY_SHIM" \
    LD_LIBRARY_PATH="$METRICS_SHIM_DIR" \
    EXIT_AFTER_ROI="$EXIT_AFTER_ROI" \
    "$VTUNE_BIN_DIR/vtune" \
      -collect "$ANALYSIS" \
      "${VTUNE_START_PAUSED[@]}" \
      "${VTUNE_KNOBS[@]}" \
      -result-dir "$RESULT_DIR" \
      -- "$VENV_DIR/bin/python" "$REPO_ROOT/scripts/debug/run_stage2_vtune_roi.py" infer \
            --config "$CONFIG_PATH" \
            --set "model.base_path=$BASE_PATH" \
            --set "checkpoint.path=$CHECKPOINT_PATH" \
            --set "data.index_root=$DATA_INDEX_ROOT" \
            --set "data.transport.root=$DATA_INDEX_ROOT" \
            --set "data.test_index=smoke-index.jsonl.gz" \
            --set "inference.device=xpu" \
            --set "validation.sample_count=1" \
            --set "inference.run_id=vtune-stage2-$STAMP" \
            --set "runtime.stage2_inference_measurements=true" \
            --set "runtime.stage2_xpu_profiler=true" \
            --set "runtime.stage2_xpu_profiler_start_chunk=$START_CHUNK" \
            --set "runtime.stage2_xpu_profiler_end_chunk=$END_CHUNK" \
            "${FUSE_SET[@]}" \
            "${EXTRA_SET[@]}" \
            --set "runtime.output_dir=$OUTPUT_DIR"
VTUNE_EXIT=$?
echo "vtune exit code: $VTUNE_EXIT"

# ---- Headless reports (SKILL.md section 11.1) ----
echo
echo "===== Headless reports ====="
_write_report() {
  local name="$1" out="$2"; shift 2
  if "$VTUNE_BIN_DIR/vtune" "$@" -r "$RESULT_DIR" -format csv > "$out" 2>&1; then
    if [[ -s "$out" ]]; then
      echo "  wrote $name"
      return
    fi
    rm -f "$out"
    echo "  $name empty -- removed"
    return
  fi
  # A concurrently-open vtune-gui/vtune-server session on this result holds
  # its sqlite-db open and every CLI report call fails with
  # "Error: 0x40000006 (Insufficient permissions) -- .../sqlite-db" while it
  # does -- not a real filesystem permission problem. Fall back to querying a
  # throwaway copy, which gets its own independent (unlocked) sqlite-db.
  if grep -q "Insufficient permissions" "$out" 2>/dev/null; then
    echo "  $name: result is locked by a concurrent vtune-gui/vtune-server session; retrying against a copy"
    local tmp_copy
    tmp_copy=$(mktemp -d)/result_copy
    if cp -r "$RESULT_DIR" "$tmp_copy" 2>/dev/null \
        && "$VTUNE_BIN_DIR/vtune" "$@" -r "$tmp_copy" -format csv > "$out" 2>&1 \
        && [[ -s "$out" ]]; then
      echo "  wrote $name (via copy)"
    else
      echo "  $name FAILED even against a copy:"
      sed 's/^/    /' "$out" | head -5
      rm -f "$out"
    fi
    rm -rf "$(dirname "$tmp_copy")"
    return
  fi
  echo "  $name FAILED:"
  sed 's/^/    /' "$out" | head -5
  rm -f "$out"
}

_write_report "summary.csv"  "$RESULT_DIR/summary.csv"  -report summary
# On this VTune version (2025.10) the GPU-hotspots-by-kernel report is
# `-report hotspots -group-by computing-task` (the `gpu-hotspots` report
# *name* used on some other versions doesn't exist here -- names vary by
# release, per SKILL.md section 3.2's warning). Try both spellings.
# `-group-by computing-task` does NOT collapse to one row per kernel *name*:
# it further splits by exact Work Size (Global/Local), so a name like
# `gemm_kernel` or `gen_conv` that runs across many different tensor shapes
# becomes dozens to hundreds of rows. A row-count limit here truncates the
# row *list*, not a name-level aggregate, so a low -limit silently drops most
# instances of exactly the highest-frequency kernels and their true totals
# come out far too low if you then sum by name yourself (measured: -limit 30
# gave gemm_kernel=0.26s/322 instances; the true total at -limit 5000 was
# gemm_kernel=1.28s/3732 instances -- see the 2026-09-22 status log
# correction). Use a limit comfortably above the true row count (a few
# hundred kernel names x shape variants); 5000 costs nothing extra since this
# is CSV export, not the GUI.
_write_report "hotspots.csv" "$RESULT_DIR/hotspots.csv" \
  -report hotspots -group-by computing-task -limit 5000
if [[ ! -s "$RESULT_DIR/hotspots.csv" ]]; then
  _write_report "hotspots.csv" "$RESULT_DIR/hotspots.csv" \
    -report gpu-hotspots -group-by computing-task -limit 5000
fi
if [[ "$ANALYSIS" == "gpu-offload" ]]; then
  _write_report "python_stacks.csv" "$RESULT_DIR/python_stacks.csv" \
    -report hotspots -group-by computing-task,host-callstack -limit 5000
elif [[ "$ANALYSIS" == "hotspots" ]]; then
  _write_report "python_stacks.csv" "$RESULT_DIR/python_stacks.csv" \
    -report hotspots -group-by function,host-callstack -limit 5000
fi
# No tasks.csv: this driver only uses ITT resume/pause (SKILL.md's "batched
# window" ROI scheme), not per-step ITT *tasks*, so there is nothing for a
# tasks/top-tasks report to show. Add ITT task_begin/task_end calls in
# run_stage2_vtune_roi.py first if per-chunk phase lanes are needed later.

echo
echo "===== Post-run verification ====="
if [[ -s "$RESULT_DIR/summary.csv" ]]; then
  GPU_TIME=$(awk -F',' 'tolower($0) ~ /gpu time/{for(i=1;i<=NF;i++){gsub(/[" ]/,"",$i); if($i+0>0){print $i; exit}}}' "$RESULT_DIR/summary.csv")
  if [[ -n "${GPU_TIME:-}" ]]; then
    echo "  [ OK ] GPU Time = ${GPU_TIME}"
  else
    echo "  [FAIL] GPU Time = 0/unparsable -- check itt_resume()/itt_pause() fired (see stderr above)"
  fi
else
  echo "  [WARN] summary.csv missing"
fi

# Shrink the result dir (SKILL.md section 11.1).
if [[ "${VTUNE_KEEP_RAW:-0}" != "1" ]]; then
  BEFORE=$(du -sm "$RESULT_DIR" 2>/dev/null | awk '{print $1}')
  "$VTUNE_BIN_DIR/vtune" -finalize -r "$RESULT_DIR" -discard-raw-data >/dev/null 2>&1 || true
  AFTER=$(du -sm "$RESULT_DIR" 2>/dev/null | awk '{print $1}')
  echo "Result dir shrunk from ${BEFORE:-?} MB to ${AFTER:-?} MB (VTUNE_KEEP_RAW=1 to keep raw data)."
fi

echo
echo "Done. Result: $RESULT_DIR"
echo "Open with: vtune-gui $RESULT_DIR"
exit "$VTUNE_EXIT"
