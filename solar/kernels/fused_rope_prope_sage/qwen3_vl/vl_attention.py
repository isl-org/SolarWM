"""Attention backends for Qwen3-VL-8B long-context runs (chunked prefill + INT8 KV cache).

    import vl_attention; vl_attention.register()
    model = ...from_pretrained(..., attn_implementation="vl_xpu"); vl_attention.set_arm("sage" | "flash")

Prefill (q_len >= 2): one kernel call per (layer, chunk); the chunk's queries attend to every key in
the cache, with a bottom-right aligned causal mask (query i sees keys <= i + Skv - Sq).
  * "sage":  sage_causal  (smoothed INT4 Q*K^T, bf16 P*V)
  * "flash": flash_causal (bf16 flash attention, same sycl-tla tiling) -- the baseline
Decode (q_len == 1): torch SDPA with native GQA for both arms (no mask; the token sees all keys).

No attention mask is ever materialised: the registered mask builder returns None (an explicit
[chunk x 256K] mask would be 1 GiB per call, and torch SDPA with a mask falls back to a path that
needs ~192 GiB at that shape). Batch size 1, text-only causal attention is assumed; the vision
encoder keeps its own attention implementation.
"""
import os
import sys

import torch
import torch.nn.functional as F

KDIR = os.environ.get("SAGE_KERNEL_DIR", "/home/gta/omni_bench/kernel")
sys.path.insert(0, os.environ.get("SAGE_BUILD_DIR", f"{KDIR}/build"))
sys.path.insert(0, os.environ.get("FLASH_BUILD_DIR", f"{KDIR}/build_flash"))
import sage_causal   # noqa: E402
import flash_causal  # noqa: E402

NAME = "vl_xpu"
_ARM = "sage"
_TIMING = False
TIMED = []          # (start_event, end_event, sq, skv) for prefill attention calls


def set_arm(arm: str):
    global _ARM
    assert arm in ("sage", "flash"), arm
    _ARM = arm


def set_timing(on: bool, clear: bool = True):
    global _TIMING
    _TIMING = on
    if clear:
        TIMED.clear()


def attention_time_ms():
    """Total prefill attention time; call after torch.xpu.synchronize()."""
    return sum(a.elapsed_time(b) for a, b, *_ in TIMED)


def attention_forward(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None,
                      is_causal=None, **kwargs):
    """query [B,Hq,Sq,D], key/value [B,Hkv,Skv,D] (views of [B,S,H,D] storage). Returns [B,Sq,Hq,D]."""
    b, hq, sq, d = query.shape
    assert b == 1 and attention_mask is None, "batch-1, mask-free attention only"
    if scaling is not None:
        assert abs(scaling - d ** -0.5) < 1e-6, "kernels use the fixed 1/sqrt(D) scale"
    if sq == 1:  # decode: one query sees every key
        out = F.scaled_dot_product_attention(query, key, value, enable_gqa=True)
        return out.transpose(1, 2), None
    q = query.transpose(1, 2).contiguous()       # free: already [B,S,H,D] in memory
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    kernel = sage_causal if _ARM == "sage" else flash_causal
    if _TIMING:
        start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        start.record()
        out = kernel.forward(q, k, v, True)
        end.record()
        TIMED.append((start, end, sq, k.shape[1]))
        return out, None
    return kernel.forward(q, k, v, True), None


def _no_mask(*args, **kwargs):
    return None


def register():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    AttentionInterface.register(NAME, attention_forward)
    AttentionMaskInterface.register(NAME, _no_mask)
