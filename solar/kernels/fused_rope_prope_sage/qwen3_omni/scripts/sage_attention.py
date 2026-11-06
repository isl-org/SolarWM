"""SageAttention (sage_causal SYCL kernel) as a transformers attention backend for Qwen3-Omni.

    import sage_attention; sage_attention.register()
    model = ...from_pretrained(..., attn_implementation="sage_xpu")

Where Sage runs (everything else falls back to torch SDPA, unchanged):
  * prefill only (q_len >= MIN_Q_LEN); decode steps (q_len == 1) use SDPA
  * head_dim 64 / 128, fp16 / bf16, scaling == head_dim ** -0.5 (the kernel's fixed scale)
  * no explicit mask: thinker prefill with batch 1 gets attention_mask=None + module.is_causal
    from the SDPA mask builder; the audio encoder (non-causal, per-chunk) also passes None.
    Any explicit mask (padding, sliding window, ...) falls back to SDPA.
  => thinker text model (D=128, GQA 32/4, causal) and audio encoder (D=64, non-causal) run Sage;
     the vision encoder (D=72) stays on SDPA.

Q/K/V arrive as [B,H,S,D] views of physically [B,S,H,D] tensors in prefill, so the
transpose(1,2).contiguous() below is free; GQA is native in the kernel (no repeat_kv).

COMPARE mode (set_compare(True)) additionally runs SDPA on every Sage call, returns the Sage
result, and records per-call error statistics in STATS (for the real-activation accuracy check).
"""
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("SAGE_BUILD_DIR", "/home/gta/omni_bench/kernel/build"))
import sage_causal  # noqa: E402

NAME = "sage_xpu"
MIN_Q_LEN = 2
_COMPARE = False
_FORCE_SDPA = False             # route every call to SDPA (A/B on one loaded model)
SDPA_LAYERS = set()             # thinker layer_idx values kept on SDPA (accuracy-sensitive layers)
_TIMING = False
TIMED = []                      # (start_event, end_event, info) for thinker prefill attention calls


def set_timing(on: bool):
    """Record XPU events around thinker-text prefill attention calls (both SDPA and Sage paths).
    Events are only resolved by attention_time_ms(), after the caller has synchronised."""
    global _TIMING
    _TIMING = on
    TIMED.clear()


def set_timing_active(on: bool):
    """Pause / resume recording without dropping events already recorded."""
    global _TIMING
    _TIMING = on


def attention_time_ms():
    return sum(a.elapsed_time(b) for a, b, _ in TIMED)
STATS = []                      # list of dicts, filled in compare mode
COUNTS = defaultdict(int)       # (path, reason) -> calls


def set_compare(on: bool):
    global _COMPARE
    _COMPARE = on


def set_sdpa_layers(layers):
    """e.g. set_sdpa_layers({3}): layer 3's error grows with context (cos 0.94 @8K -> 0.81 @32K)."""
    SDPA_LAYERS.clear()
    SDPA_LAYERS.update(layers)


def set_force_sdpa(on: bool):
    global _FORCE_SDPA
    _FORCE_SDPA = on


def _sdpa(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                  scaling=scaling, is_causal=is_causal, **kwargs)


def _why_not_sage(module, query, key, attention_mask, scaling):
    b, h, sq, d = query.shape
    if sq < MIN_Q_LEN:
        return "decode"
    if d not in (64, 128):
        return f"head_dim{d}"
    if query.dtype not in (torch.bfloat16, torch.float16) or key.dtype != query.dtype:
        return "dtype"
    if attention_mask is not None:
        return "explicit_mask"
    if SDPA_LAYERS and getattr(module, "layer_idx", None) in SDPA_LAYERS and getattr(module, "is_causal", False):
        return "layer_override"
    if scaling is not None and abs(scaling - d ** -0.5) > 1e-6:
        return "scaling"
    if h % key.shape[1] != 0:
        return "gqa_ratio"
    if getattr(module, "is_causal", False) and key.shape[2] < sq:
        return "kv_shorter_than_q"
    return None


def sage_attention_forward(module, query, key, value, attention_mask=None, dropout=0.0,
                           scaling=None, is_causal=None, **kwargs):
    if _TIMING and query.shape[2] > 1 and type(module).__name__.endswith("ThinkerTextAttention"):
        start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        start.record()
        out = _attention(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)
        end.record()
        TIMED.append((start, end, dict(layer=getattr(module, "layer_idx", None), sq=query.shape[2],
                                       skv=key.shape[2], heads=query.shape[1], kv_heads=key.shape[1],
                                       head_dim=query.shape[3])))
        return out
    return _attention(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)


def _attention(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None,
               is_causal=None, **kwargs):
    why = "forced" if _FORCE_SDPA else _why_not_sage(module, query, key, attention_mask, scaling)
    if why is not None:
        COUNTS[("sdpa", why)] += 1
        return _sdpa(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)
    causal = bool(is_causal) if is_causal is not None else bool(getattr(module, "is_causal", False))
    COUNTS[("sage", "causal" if causal else "noncausal")] += 1
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    out = sage_causal.forward(q, k, v, is_causal=causal)          # [B, Sq, H, D]
    if _COMPARE:
        ref, _ = _sdpa(module, query, key, value, None, 0.0, scaling, causal, **kwargs)
        o, r = out.float().flatten(), ref.float().flatten()
        STATS.append(dict(module=type(module).__name__, layer=getattr(module, "layer_idx", None),
                          sq=query.shape[2], skv=key.shape[2], heads=query.shape[1], kv_heads=key.shape[1],
                          head_dim=query.shape[3], causal=causal,
                          cos=F.cosine_similarity(o, r, dim=0).item(),
                          rel_l2=((o - r).norm() / r.norm()).item(),
                          max_abs=(o - r).abs().max().item()))
    return out, None


def register():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionInterface.register(NAME, sage_attention_forward)
    # Same mask construction as SDPA: batch-1 causal prefill gets mask=None (is_causal path).
    AttentionMaskInterface.register(NAME, sdpa_mask)
