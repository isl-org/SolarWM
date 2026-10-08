from __future__ import annotations

import torch
import pytest

from solarwm.kernels.fused_rope_prope_sage.radial_attention import (
    RadialAttentionConfig,
    build_radial_block_mask,
    build_sage_block_indices,
    radial_density,
)


def test_radial_plan_is_rectangular_and_has_sink() -> None:
    config = RadialAttentionConfig(enabled=True, decay_factor=0.8, sink_frames=1)
    block_mask, q_pad, kv_pad, token_mask = build_radial_block_mask(
        query_tokens=1215,
        kv_tokens=7290,
        tokens_per_frame=405,
        query_start_frame=15,
        kv_start_frame=0,
        config=config,
        device="cpu",
    )

    assert q_pad == 65
    assert kv_pad == 6
    assert token_mask.shape == (3, 18, 405, 405)
    assert bool(token_mask[:, 0].all())
    assert not bool(token_mask[0, 10].any())  # distance 5 misses split-2 sampling
    assert bool(token_mask[0, 11].any())  # distance 4 is retained
    assert radial_density(token_mask) < 1.0
    assert block_mask.seq_lengths == (1280, 7296)


def test_radial_plan_changes_with_absolute_frame_position() -> None:
    config = RadialAttentionConfig(enabled=True, sink_frames=0)
    _, _, _, early = build_radial_block_mask(
        query_tokens=810,
        kv_tokens=7290,
        tokens_per_frame=405,
        query_start_frame=0,
        kv_start_frame=0,
        config=config,
        device="cpu",
    )
    _, _, _, late = build_radial_block_mask(
        query_tokens=810,
        kv_tokens=7290,
        tokens_per_frame=405,
        query_start_frame=16,
        kv_start_frame=0,
        config=config,
        device="cpu",
    )
    assert not torch.equal(early, late)


def test_sage_indices_are_padded_and_device_ready() -> None:
    config = RadialAttentionConfig(enabled=True, sink_frames=1)
    counts, indices = build_sage_block_indices(
        query_tokens=1215,
        kv_tokens=7290,
        tokens_per_frame=405,
        query_start_frame=15,
        kv_start_frame=0,
        config=config,
        device="cpu",
    )
    assert counts.shape == (10,)
    assert indices.shape[0] == 10
    assert indices.dtype == torch.int32
    assert torch.all(counts > 0)
    assert torch.all(indices < 228)
    cached_counts, cached_indices = build_sage_block_indices(
        query_tokens=1215,
        kv_tokens=7290,
        tokens_per_frame=405,
        query_start_frame=15,
        kv_start_frame=0,
        config=config,
        device="cpu",
    )
    assert cached_counts.data_ptr() == counts.data_ptr()
    assert cached_indices.data_ptr() == indices.data_ptr()


def test_flex_block_mask_matches_dense_masked_attention() -> None:
    from torch.nn.attention.flex_attention import flex_attention

    config = RadialAttentionConfig(enabled=True, sink_frames=1, block_size=4)
    block_mask, q_pad, kv_pad, token_mask = build_radial_block_mask(
        query_tokens=16,
        kv_tokens=64,
        tokens_per_frame=16,
        query_start_frame=7,
        kv_start_frame=0,
        config=config,
        device="cpu",
    )
    torch.manual_seed(0)
    q = torch.randn(1, 1, 16, 8)
    k = torch.randn(1, 1, 64, 8)
    v = torch.randn(1, 1, 64, 8)
    output = flex_attention(
        torch.nn.functional.pad(q, (0, 0, 0, q_pad)),
        torch.nn.functional.pad(k, (0, 0, 0, kv_pad)),
        torch.nn.functional.pad(v, (0, 0, 0, kv_pad)),
        block_mask=block_mask,
        scale=8**-0.5,
    )[:, :, :16]

    allowed = token_mask.permute(0, 2, 1, 3).reshape(16, 64)
    scores = q @ k.transpose(-2, -1) * 8**-0.5
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    expected = torch.softmax(scores, dim=-1) @ v
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Intel XPU is required",
)
def test_fused_radial_sdpa_and_sage_smoke() -> None:
    from solarwm.kernels.fused_rope_prope_sage.test_fused_rope_prope_sage import (
        _kwargs,
        create_test_inputs,
    )
    from solarwm.kernels.fused_rope_prope_sage import (
        fused_rope_prope_sage,
        fused_rope_prope_sdpa_split,
    )

    inputs = create_test_inputs(
        "xpu",
        rope_dtype="float32",
        q_frames=1,
        kv_frames=2,
        h_patches=2,
        w_patches=2,
        num_heads=2,
        head_dim=32,
    )
    config = RadialAttentionConfig(enabled=True, sink_frames=1)
    kwargs = _kwargs(inputs)
    sdpa = fused_rope_prope_sdpa_split(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        **kwargs,
        radial_config=config,
        radial_query_start_frame=1,
        radial_kv_start_frame=0,
    )
    sage = fused_rope_prope_sage(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        **kwargs,
        radial_config=config,
        radial_query_start_frame=1,
        radial_kv_start_frame=0,
    )
    torch.xpu.synchronize()
    assert sdpa.shape == sage.shape == inputs["q"].shape
    assert torch.isfinite(sdpa).all()
    assert torch.isfinite(sage).all()
