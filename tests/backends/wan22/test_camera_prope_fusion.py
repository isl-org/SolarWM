"""Numerical tests for folding EchoRoPE rotations into fused PRoPE matrices."""

from __future__ import annotations

import torch

from solarwm.backends.wan22.runtime.modeling.camera_prope import (
    _apply_tiled_projmat,
    _prepare_apply_fns_all_dim,
    prope_apply_fns_separate_cached,
)


def _rotate_pairs(feats: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Reference real-valued complex-pair rotation for [B, H, L, D] features."""
    pair_values = feats.reshape(*feats.shape[:-1], -1, 2)
    cos = freqs.real.unsqueeze(1)
    sin = freqs.imag.unsqueeze(1)
    real, imag = pair_values.unbind(dim=-1)
    return torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).flatten(-2)


def _camera_inputs(batch: int, tokens: int, *, dtype: torch.dtype = torch.float64):
    viewmats = torch.eye(4, dtype=dtype).repeat(batch, tokens, 1, 1)
    viewmats[..., 0, 3] = torch.linspace(-0.2, 0.2, tokens, dtype=dtype)
    viewmats[..., 1, 3] = torch.linspace(0.3, -0.1, tokens, dtype=dtype)
    intrinsics = torch.eye(3, dtype=dtype).repeat(batch, tokens, 1, 1)
    intrinsics[..., 0, 0] = 1.3
    intrinsics[..., 1, 1] = 0.8
    return viewmats, intrinsics


def test_fused_rope_prope_matches_sequential_per_ray_fp64() -> None:
    torch.manual_seed(7)
    batch, heads, tokens, head_dim = 1, 3, 5, 8
    feats = torch.randn(batch, heads, tokens, head_dim, dtype=torch.float64)
    freqs = torch.polar(
        torch.ones(batch, tokens, head_dim // 2, dtype=torch.float64),
        torch.randn(batch, tokens, head_dim // 2, dtype=torch.float64),
    )
    viewmats, intrinsics = _camera_inputs(batch, tokens)

    sequential_q, _, _ = _prepare_apply_fns_all_dim(
        head_dim, viewmats, intrinsics, None, None, None, None
    )
    fused_q, _, _ = _prepare_apply_fns_all_dim(
        head_dim,
        viewmats,
        intrinsics,
        None,
        None,
        None,
        None,
        rope_freqs_q=freqs,
        rope_freqs_kv=freqs,
    )

    expected = sequential_q(_rotate_pairs(feats, freqs))
    actual = fused_q(feats)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


def test_fused_rope_prope_matches_sequential_bf16() -> None:
    torch.manual_seed(8)
    batch, heads, tokens, head_dim = 1, 2, 6, 8
    feats = torch.randn(batch, heads, tokens, head_dim, dtype=torch.bfloat16)
    freqs = torch.polar(
        torch.ones(batch, tokens, head_dim // 2, dtype=torch.float32),
        torch.randn(batch, tokens, head_dim // 2, dtype=torch.float32),
    )
    viewmats, intrinsics = _camera_inputs(batch, tokens, dtype=torch.float32)
    sequential_q, _, _ = _prepare_apply_fns_all_dim(
        head_dim, viewmats, intrinsics, None, None, None, None
    )
    fused_q, _, _ = _prepare_apply_fns_all_dim(
        head_dim,
        viewmats,
        intrinsics,
        None,
        None,
        None,
        None,
        rope_freqs_q=freqs,
        rope_freqs_kv=freqs,
    )

    expected = sequential_q(_rotate_pairs(feats.float(), freqs).bfloat16())
    actual = fused_q(feats)
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.04)


def test_per_camera_projection_still_matches_per_ray_expansion() -> None:
    """Keep coverage of the existing per-camera PRoPE branch."""
    torch.manual_seed(9)
    feats = torch.randn(1, 2, 6, 8)
    camera_matrix = torch.eye(4).repeat(1, 2, 1, 1)
    camera_matrix[:, 1, 0, 3] = 0.25
    per_camera = _apply_tiled_projmat(feats, camera_matrix)
    expanded = _apply_tiled_projmat(feats, camera_matrix.repeat_interleave(3, dim=1))
    torch.testing.assert_close(per_camera, expanded)


def test_cached_fused_q_and_kv_transforms_are_reused_and_correct() -> None:
    torch.manual_seed(10)
    head_dim, q_tokens, kv_tokens = 8, 3, 5
    q_viewmats, q_Ks = _camera_inputs(1, q_tokens)
    kv_viewmats, kv_Ks = _camera_inputs(1, kv_tokens)
    q_freqs = torch.polar(torch.ones(1, q_tokens, 4), torch.randn(1, q_tokens, 4))
    kv_freqs = torch.polar(torch.ones(1, kv_tokens, 4), torch.randn(1, kv_tokens, 4))
    cache: dict = {}

    first = prope_apply_fns_separate_cached(
        cache,
        head_dim=head_dim,
        q_viewmats=q_viewmats,
        q_Ks=q_Ks,
        kv_viewmats=kv_viewmats,
        kv_Ks=kv_Ks,
        kv_window=(0, kv_tokens),
        rope_freqs_q=q_freqs,
        rope_freqs_kv=kv_freqs,
    )
    second = prope_apply_fns_separate_cached(
        cache,
        head_dim=head_dim,
        q_viewmats=q_viewmats,
        q_Ks=q_Ks,
        kv_viewmats=kv_viewmats,
        kv_Ks=kv_Ks,
        kv_window=(0, kv_tokens),
        rope_freqs_q=q_freqs,
        rope_freqs_kv=kv_freqs,
    )
    assert first[0] is second[0]
    assert first[1] is second[1]
    assert set(cache) == {"q_rope", ("kv_rope", (0, kv_tokens))}


def test_causal_self_attention_fused_kernels_execution() -> None:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return

    from solarwm.backends.wan22.runtime.modeling.causal_model import CausalWanSelfAttention, rope_params

    dev = "xpu"
    dim, num_heads = 256, 2
    head_dim = dim // num_heads
    frame_seqlen = 16
    local_attn_size = 4
    cache_tokens = local_attn_size * frame_seqlen
    capacity = cache_tokens
    physical_cache_tokens = cache_tokens * 2

    for kernel in ("reference", "fused_rope_prope_sdpa", "fused_rope_prope_sage"):
        attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            local_attn_size=local_attn_size,
            sink_size=0,
            camera_attention_mode="fused_prope",
            camera_translation_transform="linear",
            fused_kernel=kernel,
            rope_dtype="float64",
            frame_seq_length=frame_seqlen,
        ).to(device=dev, dtype=torch.bfloat16)

        q_tokens = 2 * frame_seqlen
        q = torch.randn(1, q_tokens, dim, dtype=torch.bfloat16, device=dev)
        grid_sizes = torch.tensor([[2, 4, 4]], device=dev)
        d = head_dim
        freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        ).to(device=dev, dtype=torch.complex128)

        cam_viewmats = torch.eye(4, device=dev, dtype=torch.float64).repeat(1, q_tokens, 1, 1)
        cam_K = torch.eye(3, device=dev, dtype=torch.float64).repeat(1, q_tokens, 1, 1)

        kv_cache = {
            "k": torch.zeros(1, physical_cache_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=dev),
            "v": torch.zeros(1, physical_cache_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=dev),
            "global_end_index": 0,
            "local_end_index": 0,
            "_circular_kv_cache": True,
            "_circular_capacity": capacity,
            "_circular_ring_start": 0,
        }
        kv_cam_vm = torch.eye(4, device=dev, dtype=torch.float64).repeat(1, physical_cache_tokens, 1, 1)
        kv_cam_K = torch.eye(3, device=dev, dtype=torch.float64).repeat(1, physical_cache_tokens, 1, 1)

        out, cache_update = attn(
            q,
            seq_lens=None,
            grid_sizes=grid_sizes,
            freqs=freqs,
            kv_cache=kv_cache,
            current_start=0,
            cam_viewmats=cam_viewmats,
            cam_K=cam_K,
            kv_cam_viewmats=kv_cam_vm,
            kv_cam_K=kv_cam_K,
            frame_seqlen=frame_seqlen,
            cache_update_policy="commit_detached",
        )
        assert out.shape == (1, q_tokens, dim)
        assert cache_update[0] == q_tokens

