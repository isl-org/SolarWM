"""Correctness tests for the causal SageAttention kernel (sage_causal) against SDPA.

Reference: FP32 SDPA with an explicit bottom-right aligned causal mask (query i sees keys
<= i + Skv - Sq; torch's is_causal is top-left aligned when Sq != Skv). The bf16 SDPA error
against the same reference is reported as the noise floor.

    /home/gta/omni_bench/env.sh python -m pytest -q test_sage_causal.py
"""
import sys
import pytest
import torch
import torch.nn.functional as F

import os
sys.path.insert(0, os.environ.get("SAGE_BUILD_DIR", "/home/gta/omni_bench/kernel/build"))
sys.path.insert(1, "/home/gta/omni_bench/kernel/build")  # sage_orig
import sage_causal  # noqa: E402
import sage_orig  # noqa: E402

DEV = "xpu"


def make_qkv(sq, skv, h, d, dtype, outliers=False, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(1, sq, h, d, device=DEV, dtype=torch.float32, generator=g)
    k = torch.randn(1, skv, h, d, device=DEV, dtype=torch.float32, generator=g)
    v = torch.randn(1, skv, h, d, device=DEV, dtype=torch.float32, generator=g)
    if outliers:
        # LLM-like keys: a few channels with large per-head bias + scaled queries -> peaky softmax
        bias = torch.zeros(h, d, device=DEV)
        bias[:, :4] = 12.0
        k = k + bias
        q = q * 2.0
    return [x.to(dtype).contiguous() for x in (q, k, v)]


def reference(q, k, v, causal):
    qt, kt, vt = (x.transpose(1, 2).float() for x in (q, k, v))
    mask = None
    if causal:
        sq, skv = q.shape[1], k.shape[1]
        i = torch.arange(sq, device=DEV)[:, None]
        j = torch.arange(skv, device=DEV)[None, :]
        mask = j <= i + (skv - sq)
    return F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask).transpose(1, 2)


def sdpa_lowp(q, k, v, causal):
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
    mask = None
    if causal:
        sq, skv = q.shape[1], k.shape[1]
        i = torch.arange(sq, device=DEV)[:, None]
        j = torch.arange(skv, device=DEV)[None, :]
        mask = j <= i + (skv - sq)
    return F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask).transpose(1, 2)


def errs(out, ref):
    o, r = out.float().flatten(), ref.float().flatten()
    cos = F.cosine_similarity(o, r, dim=0).item()
    rel = ((o - r).norm() / r.norm()).item()
    return cos, rel, (o - r).abs().max().item()


CASES = [
    # (sq, skv, heads, head_dim, causal)
    (2048, 2048, 32, 128, True),
    (4096, 4096, 32, 128, True),
    (1000, 1000, 32, 128, True),    # partial Q block and KV tile
    (17, 17, 32, 128, True),        # tiny
    (1, 1, 32, 128, True),
    (300, 2000, 32, 128, True),     # prefill on top of a cache (bottom-right aligned)
    (256, 257, 8, 128, True),       # diagonal offset of 1
    (2048, 2048, 32, 128, False),
    (1500, 1500, 20, 64, True),     # head_dim 64 config (QTile 128, KTile 64)
    (1500, 1500, 20, 64, False),    # audio-encoder-like, non-causal
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("outliers", [False, True], ids=["gauss", "outlier"])
@pytest.mark.parametrize("sq,skv,h,d,causal", CASES)
def test_accuracy(sq, skv, h, d, causal, dtype, outliers):
    q, k, v = make_qkv(sq, skv, h, d, dtype, outliers)
    ref = reference(q, k, v, causal)
    out = sage_causal.forward(q, k, v, is_causal=causal)
    assert out.shape == q.shape and torch.isfinite(out).all(), "non-finite output"
    cos, rel, mx = errs(out, ref)
    fcos, frel, _ = errs(sdpa_lowp(q, k, v, causal), ref)
    print(f"\n  sage cos={cos:.5f} rel={rel:.4f} max={mx:.3f} | sdpa-{str(dtype)[6:]} cos={fcos:.6f} rel={frel:.4f}")
    # Quantized Q*K^T is lossy (INT8 here; the original INT4 kernel is lossier on the same data;
    # see test_noncausal_more_accurate_than_original), so these bounds only catch masking bugs,
    # which show up as cos << 0.9. Real-activation accuracy is measured separately.
    assert cos > (0.95 if outliers else 0.98), f"cosine {cos:.4f} too low"


@pytest.mark.parametrize("sq", [1000, 4096])
def test_noncausal_more_accurate_than_original(sq):
    """INT8 Q*K^T: on the same data, the non-causal output must be at least as close to the fp32
    reference as the original INT4 kernel's (and clearly closer, given 127 vs 7 levels)."""
    q, k, v = make_qkv(sq, sq, 32, 128, torch.bfloat16)
    ref = reference(q, k, v, False)
    cos_new = errs(sage_causal.forward(q, k, v), ref)[0]
    cos_orig = errs(sage_orig.forward(q, k, v), ref)[0]
    assert cos_new >= cos_orig and cos_new > 0.999, f"cos {cos_new:.5f} vs original {cos_orig:.5f}"


def test_causal_mask_is_exact():
    """Future V never reaches past rows: with future values set to 100, rows < 512 are bit-identical.
    (K is left unchanged because the K centre is a mean over all keys, so changing future K
    legitimately changes the quantisation of past keys while staying mathematically exact.)"""
    q, k, v = make_qkv(1024, 1024, 32, 128, torch.bfloat16)
    v_future = v.clone()
    v_future[:, 512:] = 100.0
    out = sage_causal.forward(q, k, v, is_causal=True)
    out_future = sage_causal.forward(q, k, v_future, is_causal=True)
    assert torch.equal(out_future[:, :512], out[:, :512])
    assert not torch.equal(out_future[:, 512:], out[:, 512:])


def test_future_keys_only_change_quantisation():
    """Changing future K moves past rows only through the K centre, never to worse than the
    same-data accuracy against the reference."""
    q, k, v = make_qkv(1024, 1024, 32, 128, torch.bfloat16)
    k2, v2 = k.clone(), v.clone()
    k2[:, 512:] = torch.randn_like(k2[:, 512:]) * 3
    v2[:, 512:] = torch.randn_like(v2[:, 512:]) * 3
    out2 = sage_causal.forward(q, k2, v2, is_causal=True)
    ref2 = reference(q, k2, v2, True)
    assert errs(out2[:, :512], ref2[:, :512])[0] > 0.98


def test_rejects_causal_with_short_kv():
    q, k, v = make_qkv(64, 32, 8, 128, torch.bfloat16)
    with pytest.raises(RuntimeError):
        sage_causal.forward(q, k, v, is_causal=True)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("sq,skv,hq,hkv,d", [(2048, 2048, 32, 4, 128), (1000, 1000, 32, 4, 128),
                                            (300, 2000, 32, 4, 128), (1500, 1500, 20, 5, 64),
                                            (512, 512, 16, 16, 128)])
def test_gqa_matches_expanded(sq, skv, hq, hkv, d, causal):
    """Native GQA must match running the kernel on K/V repeated to the Q head count (up to the
    summation order of the dS GEMM, whose shape differs between the two calls)."""
    q, _, _ = make_qkv(sq, skv, hq, d, torch.bfloat16, seed=1)
    _, k, v = make_qkv(sq, skv, hkv, d, torch.bfloat16, seed=2)
    rep = hq // hkv
    k_rep = k.repeat_interleave(rep, dim=2).contiguous()
    v_rep = v.repeat_interleave(rep, dim=2).contiguous()
    out_gqa = sage_causal.forward(q, k, v, is_causal=causal)
    out_rep = sage_causal.forward(q, k_rep, v_rep, is_causal=causal)
    torch.testing.assert_close(out_gqa, out_rep, atol=4e-3, rtol=0)
    ref = reference(q, k_rep, v_rep, causal)
    assert errs(out_gqa, ref)[0] > 0.98


def test_rejects_bad_gqa_ratio():
    q, _, _ = make_qkv(64, 64, 32, 128, torch.bfloat16)
    _, k, v = make_qkv(64, 64, 5, 128, torch.bfloat16)
    with pytest.raises(RuntimeError):
        sage_causal.forward(q, k, v)
