"""Chunk-4 DiT step time with and without the fused DiT ops (LN+AdaLN, GELU GEMM, gated residual).

Usage: python ablate_dit_fused_ops.py LINEAR_QUANT:ATTENTION_BACKEND[:radial][:nofused] ...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_chunk4_timing import measure_pipeline  # noqa: E402

from solarwm.backends.wan22.runtime import loader  # noqa: E402


def main() -> None:
    original = loader.is_stage2_camera_inference
    for spec in sys.argv[1:]:
        parts = spec.split(":")
        lq, ab, flags = parts[0], parts[1], set(parts[2:])
        loader.is_stage2_camera_inference = (lambda _c: False) if "nofused" in flags else original
        res = measure_pipeline(linear_quant=lq, attn_backend=ab, radial="radial" in flags)
        steps = res["chunk4_steps_ms"]
        steady = sum(steps[k] for k in (1, 2, 3)) / 3
        print(f"{spec:40s} | steady step {steady:6.1f} ms | DiT {res['chunk4_dit_total_ms']:7.1f} ms", flush=True)
    loader.is_stage2_camera_inference = original


if __name__ == "__main__":
    main()
