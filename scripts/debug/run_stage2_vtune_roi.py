#!/usr/bin/env python3
"""Run `solarwm infer` with VTune GPU-Hotspots ROI control via the ITT API.

This launches the *same process* as `python -m solarwm infer ...`, but
monkeypatches `_Stage2XpuProfiler.begin`/`.end` (normally a torch.profiler
Chrome-trace capture gated on `runtime.stage2_xpu_profiler_start_chunk` /
`runtime.stage2_xpu_profiler_end_chunk`) so that, at the *same* chunk
boundaries, it resumes/pauses VTune collection instead of starting/stopping
the PyTorch profiler. `torch.profiler` is never touched by this script.

Launch under `vtune -collect gpu-hotspots -start-paused` (see
`scripts/debug/run_wan22_stage2_vtune.sh` for a full driver) so that model
load, warmup, and every chunk outside [start_chunk, end_chunk] are excluded
from the GPU trace, per `.agents/skills/vtune/SKILL.md` section 2
(Conditional / ROI-Based Profiling).

Usage (mirrors `python -m solarwm infer ...` exactly, plus the two profiler
chunk-window flags which now gate the VTune ROI instead of a torch trace):

    vtune -collect gpu-hotspots -start-paused -result-dir <dir> -- \\
        python scripts/debug/run_stage2_vtune_roi.py infer \\
            --config configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml \\
            --set "runtime.stage2_xpu_profiler=true" \\
            --set "runtime.stage2_xpu_profiler_start_chunk=4" \\
            --set "runtime.stage2_xpu_profiler_end_chunk=5" \\
            ... (remaining --set overrides as usual)

Note: `runtime.stage2_xpu_profiler=true` is required to arm the ROI gate at
all (same flag the torch-profiler path uses) even though no torch trace is
produced by this script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# scripts/debug/vtune_itt.py — a generic ITT ROI-control wrapper, copied
# verbatim from .agents/skills/vtune/scripts/vtune_itt.py. See that file's
# docstring; nothing in it is vLLM-specific.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vtune_itt  # noqa: E402

vtune_itt.init_paused()

from solarwm.backends.wan22.runtime import stage2 as _stage2_mod  # noqa: E402


def _roi_begin(self, *, case_slot: int, chunk_index: int) -> None:  # noqa: ANN001
    del case_slot
    if getattr(self, "_itt_active", False) or not self.enabled:
        return
    if chunk_index != self.start_chunk or getattr(self.device, "type", None) != "xpu":
        return
    self._itt_active = True
    self._captured = True
    sys.stderr.write(
        f"[vtune-roi] chunk_index={chunk_index}: itt_resume() "
        f"(backend={'available' if vtune_itt.itt_available() else 'NONE - no-op'})\n"
    )
    sys.stderr.flush()
    vtune_itt.itt_resume()


def _roi_end(self, *, chunk_index: int, force: bool = False) -> None:  # noqa: ANN001
    if not getattr(self, "_itt_active", False):
        return
    if not force and chunk_index != self.end_chunk:
        return
    self._itt_active = False
    import torch

    # Mandatory per SKILL.md section 6: without a synchronize before pause,
    # in-flight GPU kernels are missed and VTune reports GPU Time = 0.
    if torch.xpu.is_available():
        torch.xpu.synchronize()
    sys.stderr.write(f"[vtune-roi] chunk_index={chunk_index}: xpu sync + itt_pause()\n")
    sys.stderr.flush()
    vtune_itt.itt_pause()
    if os.environ.get("EXIT_AFTER_ROI") in ("1", "true", "True"):
        sys.stderr.write(
            f"[vtune-roi] chunk_index={chunk_index}: EXIT_AFTER_ROI active, exiting\n"
        )
        sys.stderr.flush()
        os._exit(0)


_stage2_mod._Stage2XpuProfiler.begin = _roi_begin  # type: ignore[method-assign]
_stage2_mod._Stage2XpuProfiler.end = _roi_end  # type: ignore[method-assign]

if __name__ == "__main__":
    from solarwm.cli import main

    raise SystemExit(main())
