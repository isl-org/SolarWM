"""Unit tests for fused DiT operators on Intel XPU with parity validation."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from solarwm.backends.wan22.runtime.modeling.causal_model import (
    CausalWanAttentionBlock,
    CausalWanModel,
)
from solarwm.backends.wan22.runtime.modeling.model import WanRMSNorm
from solarwm.kernels.dit_fused import (
    fused_gated_residual_add_,
    onednn_linear_gelu_tanh,
    triton_layernorm_adaln,
    triton_rmsnorm,
)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required for Triton/oneDNN kernels")
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("S,num_tokens", [(1215, 5), (405, 3)])
def test_triton_layernorm_adaln_parity(B: int, S: int, num_tokens: int):
    torch.manual_seed(42)
    dev = "xpu"
    C = 3072
    mod_seqlen = S // num_tokens

    x = torch.randn(B, S, C, device=dev, dtype=torch.bfloat16)
    scale = torch.randn(B, num_tokens, 1, C, device=dev, dtype=torch.bfloat16)
    shift = torch.randn(B, num_tokens, 1, C, device=dev, dtype=torch.bfloat16)

    # Reference eager
    norm = F.layer_norm(x.float(), (C,), eps=1e-6).to(torch.bfloat16)
    ref = (norm.unflatten(1, (num_tokens, mod_seqlen)) * (1.0 + scale) + shift).flatten(1, 2)

    # Fused Triton
    out = triton_layernorm_adaln(x, scale, shift, eps=1e-6)

    cos_sim = torch.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0).item()
    max_diff = (ref - out).abs().max().item()

    assert cos_sim > 0.9999, f"Cosine similarity {cos_sim} too low for B={B}, S={S}"
    assert max_diff < 0.25, f"Max diff {max_diff} too large for B={B}, S={S}"


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required for Triton/oneDNN kernels")
@pytest.mark.parametrize("C", [1024, 3072])
def test_triton_rmsnorm_parity(C: int):
    torch.manual_seed(42)
    dev = "xpu"
    B, S = 1, 1215

    x = torch.randn(B, S, C, device=dev, dtype=torch.bfloat16)
    weight = torch.randn(C, device=dev, dtype=torch.bfloat16)

    # Reference eager WanRMSNorm
    ref = (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)).type_as(x) * weight

    # Fused Triton
    out = triton_rmsnorm(x, weight, eps=1e-5)

    cos_sim = torch.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0).item()
    max_diff = (ref - out).abs().max().item()

    assert cos_sim > 0.9999, f"Cosine similarity {cos_sim} too low for C={C}"
    assert max_diff < 0.15, f"Max diff {max_diff} too large for C={C}"


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required for Triton/oneDNN kernels")
@pytest.mark.parametrize("gate_dim", [3, 4])
def test_fused_gated_residual_add_parity(gate_dim: int):
    torch.manual_seed(42)
    dev = "xpu"
    B, S, C = 1, 1215, 3072
    num_tokens = 5
    mod_seqlen = S // num_tokens

    x = torch.randn(B, S, C, device=dev, dtype=torch.bfloat16)
    y = torch.randn(B, S, C, device=dev, dtype=torch.bfloat16)

    if gate_dim == 4:
        gate = torch.randn(B, num_tokens, 1, C, device=dev, dtype=torch.bfloat16)
        ref = x + (y.unflatten(1, (num_tokens, mod_seqlen)) * gate).flatten(1, 2)
    else:
        gate = torch.randn(B, num_tokens, C, device=dev, dtype=torch.bfloat16)
        ref = x + (y.unflatten(1, (num_tokens, mod_seqlen)) * gate.unsqueeze(2)).flatten(1, 2)

    x_test = x.clone()
    fused_gated_residual_add_(x_test, y, gate, num_tokens=num_tokens, mod_seqlen=mod_seqlen)

    max_diff = (ref - x_test).abs().max().item()
    assert max_diff < 0.1, f"Max diff {max_diff} too large for gate_dim={gate_dim}"


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required for Triton/oneDNN kernels")
@pytest.mark.parametrize("with_bias", [True, False])
def test_onednn_linear_gelu_tanh_parity(with_bias: bool):
    torch.manual_seed(42)
    dev = "xpu"
    M, K, N = 1215, 3072, 13824

    x = torch.randn(1, M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, device=dev, dtype=torch.bfloat16) if with_bias else None

    # Reference eager
    ref = F.gelu(F.linear(x, w, b), approximate="tanh")

    # oneDNN post-op epilogue
    out = onednn_linear_gelu_tanh(x, w, b)

    cos_sim = torch.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0).item()
    max_diff = (ref - out).abs().max().item()

    assert cos_sim > 0.9999, f"Cosine similarity {cos_sim} too low for with_bias={with_bias}"
    assert max_diff < 0.05, f"Max diff {max_diff} too large for with_bias={with_bias}"


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required for Triton/oneDNN kernels")
def test_causal_block_fused_parity():
    torch.manual_seed(42)
    dev = "xpu"
    dim = 256
    ffn_dim = 512
    num_heads = 4
    num_tokens = 3
    mod_seqlen = 16
    S = num_tokens * mod_seqlen

    block_eager = CausalWanAttentionBlock(
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        local_attn_size=6,
        sink_size=1,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        dit_fused_ops=False,
    ).to(dev, dtype=torch.bfloat16)
    block_eager.eval()

    block_fused = CausalWanAttentionBlock(
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        local_attn_size=6,
        sink_size=1,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        dit_fused_ops=True,
    ).to(dev, dtype=torch.bfloat16)
    block_fused.eval()
    block_fused.load_state_dict(block_eager.state_dict())

    x = torch.randn(1, S, dim, device=dev, dtype=torch.bfloat16)
    e = torch.randn(1, num_tokens, 6, dim, device=dev, dtype=torch.bfloat16)
    grid_sizes = torch.tensor([[1, 4, 4]], device=dev)
    seq_lens = torch.tensor([S], device=dev)
    d = dim // num_heads
    freqs = torch.randn(1024, d // 2, device=dev, dtype=torch.float32)
    context = torch.randn(1, 16, dim, device=dev, dtype=torch.bfloat16)

    # Forward eager
    with torch.no_grad():
        out_eager, _ = block_eager(
            x.clone(),
            e.clone(),
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context,
            context_lens=None,
            kv_cache=None,
        )

        out_fused, _ = block_fused(
            x.clone(),
            e.clone(),
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context,
            context_lens=None,
            kv_cache=None,
        )

    cos_sim = torch.cosine_similarity(
        out_eager.flatten().float(), out_fused.flatten().float(), dim=0
    ).item()
    max_diff = (out_eager - out_fused).abs().max().item()

    assert not torch.isnan(out_fused).any(), "NaN detected in fused block output"
    assert cos_sim > 0.999, f"Block cosine similarity {cos_sim} too low"
    assert max_diff < 0.2, f"Block max diff {max_diff} too large"


def test_causal_model_dit_fused_ops_propagation():
    m = CausalWanModel(
        num_layers=2,
        dim=256,
        ffn_dim=512,
        num_heads=4,
        in_dim=16,
        out_dim=16,
        text_dim=256,
        dit_fused_ops=False,
    )

    assert not m.dit_fused_ops
    assert not m.blocks[0].dit_fused_ops
    assert not m.head.dit_fused_ops

    m.dit_fused_ops = True
    assert m.dit_fused_ops
    assert m.blocks[0].dit_fused_ops
    assert m.blocks[1].dit_fused_ops
    assert m.head.dit_fused_ops
    for mod in m.modules():
        if isinstance(mod, WanRMSNorm):
            assert getattr(mod, "fused_ops", False)

    m.dit_fused_ops = False
    assert not m.dit_fused_ops
    assert not m.blocks[0].dit_fused_ops
    assert not m.head.dit_fused_ops
    for mod in m.modules():
        if isinstance(mod, WanRMSNorm):
            assert not getattr(mod, "fused_ops", False)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required for Triton/oneDNN kernels")
def test_causal_block_multi_step_continuity():
    """Verify numeric stability and parity across 4 consecutive diffusion steps."""
    torch.manual_seed(123)
    dev = "xpu"
    dim = 256
    ffn_dim = 512
    num_heads = 4
    num_tokens = 3
    mod_seqlen = 16
    S = num_tokens * mod_seqlen

    block_eager = CausalWanAttentionBlock(
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        local_attn_size=6,
        sink_size=1,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        dit_fused_ops=False,
    ).to(dev, dtype=torch.bfloat16)
    block_eager.eval()

    block_fused = CausalWanAttentionBlock(
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        local_attn_size=6,
        sink_size=1,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        dit_fused_ops=True,
    ).to(dev, dtype=torch.bfloat16)
    block_fused.eval()
    block_fused.load_state_dict(block_eager.state_dict())

    x_e = torch.randn(1, S, dim, device=dev, dtype=torch.bfloat16)
    x_f = x_e.clone()

    grid_sizes = torch.tensor([[1, 4, 4]], device=dev)
    seq_lens = torch.tensor([S], device=dev)
    d = dim // num_heads
    freqs = torch.randn(1024, d // 2, device=dev, dtype=torch.float32)
    context = torch.randn(1, 16, dim, device=dev, dtype=torch.bfloat16)

    with torch.no_grad():
        for step in range(4):
            e = torch.randn(1, num_tokens, 6, dim, device=dev, dtype=torch.bfloat16)
            x_e, _ = block_eager(
                x_e, e, seq_lens=seq_lens, grid_sizes=grid_sizes,
                freqs=freqs, context=context, context_lens=None, kv_cache=None,
            )
            x_f, _ = block_fused(
                x_f, e, seq_lens=seq_lens, grid_sizes=grid_sizes,
                freqs=freqs, context=context, context_lens=None, kv_cache=None,
            )

            assert not torch.isnan(x_f).any(), f"NaN detected at step {step}"
            cos_sim = torch.cosine_similarity(x_e.flatten().float(), x_f.flatten().float(), dim=0).item()
            assert cos_sim > 0.998, f"Step {step} cosine similarity {cos_sim} too low"
