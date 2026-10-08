#!/usr/bin/env python3
"""Minimal bf16 attention MRE: Wan22 `attention()` differs by device backend.

CUDA (default): FlashAttention varlen when installed.
XPU / CPU: ``torch.nn.functional.scaled_dot_product_attention``.

Inputs are **fixed** (no RNG): deterministic sin pattern from ``torch.arange``.

Usage
-----
On XPU (save reference tensors for the CUDA machine):

  python scripts/debug/wan22_bf16_attention_device_mre.py --device xpu --save-dir /tmp/wan22-attn-mre

On CUDA (compare to XPU reference and run local flash vs sdpa):

  python scripts/debug/wan22_bf16_attention_device_mre.py --device cuda --save-dir /tmp/wan22-attn-mre

Optional parity mode on CUDA (should match XPU ``wan_sdpa`` output closely):

  SOLARWM_WAN22_ATTENTION=sdpa python scripts/debug/wan22_bf16_attention_device_mre.py --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solarwm.backends.wan22.runtime.modeling.attention import (  # noqa: E402
    attention as wan_attention,
    wan22_attention_backend,
)

# Wan Stage2-ish self-attention token geometry (3 latent frames × 405 tokens/frame).
BATCH = 1
SEQ_LEN = 405
NUM_HEADS = 24
HEAD_DIM = 128
DTYPE = torch.bfloat16


def _pick_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda", 0)
    if name == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise SystemExit("xpu requested but torch.xpu is not available")
        return torch.device("xpu", 0)
    if name == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"unknown device: {name}")


def fixed_qkv(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build [B, L, H, D] bf16 tensors from a closed-form pattern (no randomness)."""

    elements = BATCH * SEQ_LEN * NUM_HEADS * HEAD_DIM
    base = torch.arange(elements, device=device, dtype=torch.float32)
    pattern = torch.sin(base * 0.013) + 0.25 * torch.cos(base * 0.007)
    flat = pattern.to(DTYPE)
    shape = (BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM)
    q = flat.view(shape).contiguous()
    k = flat.roll(17).view(shape).contiguous()
    v = flat.roll(31).view(shape).contiguous()
    return q, k, v


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Same math as the non-Flash branch in ``modeling/attention.py``."""

    q_t = q.transpose(1, 2).to(DTYPE)
    k_t = k.transpose(1, 2).to(DTYPE)
    v_t = v.transpose(1, 2).to(DTYPE)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=None, is_causal=False)
    return out.transpose(1, 2).contiguous()


def _tensor_digest(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().float().cpu()
    raw = cpu.numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "min": float(cpu.min().item()),
        "max": float(cpu.max().item()),
        "mean": float(cpu.mean().item()),
        "std": float(cpu.std().item()) if cpu.numel() > 1 else 0.0,
    }


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    ref = a.detach().float().cpu()
    cand = b.detach().float().cpu()
    if ref.shape != cand.shape:
        return {"ok": False, "reason": f"shape {list(cand.shape)} != {list(ref.shape)}"}
    diff = (ref - cand).abs()
    return {
        "ok": True,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "xpu", "cpu"), required=True)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Write reference tensors + report.json (share this dir with the other device)",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Load peer device report/tensors from this directory for cross-device diff",
    )
    args = parser.parse_args()

    device = _pick_device(args.device)
    q, k, v = fixed_qkv(device)

    with torch.no_grad():
        out_sdpa = sdpa_attention(q, k, v)
        out_wan = wan_attention(q, k, v, dtype=DTYPE)

    backend = wan22_attention_backend()
    report: dict[str, Any] = {
        "schema": "solarwm.wan22-bf16-attention-mre.v1",
        "device": device.type,
        "wan22_attention_backend": backend,
        "geometry": {
            "batch": BATCH,
            "seq_len": SEQ_LEN,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "dtype": "bfloat16",
        },
        "sdpa_explicit": _tensor_digest(out_sdpa),
        "wan_attention": _tensor_digest(out_wan),
        "wan_vs_explicit_sdpa_on_this_device": _compare(out_sdpa, out_wan),
    }

    ref_dir = args.reference_dir
    if ref_dir is not None:
        ref_dir = ref_dir.expanduser().resolve()
        peer_report_path = ref_dir / "report.json"
        if peer_report_path.is_file():
            peer = json.loads(peer_report_path.read_text())
            peer_wan = torch.load(ref_dir / "wan_attention.pt", map_location="cpu", weights_only=True)
            report["vs_peer_device"] = {
                "peer_device": peer.get("device"),
                "peer_backend": peer.get("wan22_attention_backend"),
                "wan_attention_vs_peer_wan": _compare(peer_wan, out_wan.cpu()),
            }
            peer_sdpa = ref_dir / "sdpa_explicit.pt"
            if peer_sdpa.is_file():
                report["vs_peer_device"]["sdpa_explicit_vs_peer_sdpa"] = _compare(
                    torch.load(peer_sdpa, map_location="cpu", weights_only=True),
                    out_sdpa.cpu(),
                )

    save_dir = args.save_dir
    if save_dir is not None:
        save_dir = save_dir.expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(q.cpu(), save_dir / "q.pt")
        torch.save(k.cpu(), save_dir / "k.pt")
        torch.save(v.cpu(), save_dir / "v.pt")
        torch.save(out_sdpa.cpu(), save_dir / "sdpa_explicit.pt")
        torch.save(out_wan.cpu(), save_dir / "wan_attention.pt")
        (save_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
