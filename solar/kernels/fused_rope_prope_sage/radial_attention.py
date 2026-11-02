"""Optional Radial Attention planning for Wan2.2 camera-length inference.

The temporal band and diagonal policy follows the Apache-2.0 Radial Attention
implementation at commit ``72788d4``:
https://github.com/mit-han-lab/radial-attention

The upstream implementation assumes square CUDA attention.  This module keeps
only the policy and builds rectangular, logical-frame metadata for SolarWM's
``Q=[B, 1215, H, D]`` / ``KV=[B, 7290, H, D]`` window.  It deliberately does
not import FlashInfer or CUDA SageAttention.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch

_UPSTREAM_SPLIT_THRESHOLD = 128

try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:  # pragma: no cover - guarded by callers
    BlockMask = None


@dataclass(frozen=True)
class RadialAttentionConfig:
    """Static policy for one radial attention family."""

    enabled: bool = False
    decay_factor: float = 0.8
    sink_frames: int = 1
    block_size: int = 128
    model_type: str = "wan"

    def key(self) -> tuple[object, ...]:
        return (
            self.enabled,
            float(self.decay_factor),
            int(self.sink_frames),
            int(self.block_size),
            self.model_type,
        )


def _window_width(
    distance: int,
    tokens_per_frame: int,
    *,
    decay_factor: float,
    block_size: int,
    model_type: str,
) -> float:
    if model_type == "wan" and distance <= 1:
        return float(tokens_per_frame)
    if model_type != "wan":
        raise ValueError(f"unsupported radial model_type={model_type!r}")
    group = distance.bit_length()
    return max(
        float(block_size),
        2 ** tokens_per_frame.bit_length() / 2**group * float(decay_factor),
    )


def _split_factor(distance: int, tokens_per_frame: int) -> int:
    """Return upstream's temporal diagonal sampling factor.

    This intentionally uses the upstream hard-coded threshold and undecayed
    length. It is independent of the backend's physical sparse block size.
    """
    group = distance.bit_length()
    decay_length = 2 ** tokens_per_frame.bit_length() / 2**group
    if decay_length >= _UPSTREAM_SPLIT_THRESHOLD:
        return 1
    return max(1, int(_UPSTREAM_SPLIT_THRESHOLD / decay_length))


def _allow_token(
    q_index: int,
    kv_index: int,
    *,
    query_tokens: int,
    kv_tokens: int,
    tokens_per_frame: int,
    query_start_frame: int,
    kv_start_frame: int,
    sink_frames: int,
    decay_factor: float,
    block_size: int,
    model_type: str,
) -> bool:
    # FlexAttention evaluates padding too. Keep padded rows/columns harmless.
    if q_index >= query_tokens or kv_index >= kv_tokens:
        return q_index == kv_index
    q_frame, q_token = divmod(q_index, tokens_per_frame)
    kv_frame, kv_token = divmod(kv_index, tokens_per_frame)
    q_frame += query_start_frame
    kv_frame += kv_start_frame
    distance = abs(q_frame - kv_frame)
    if sink_frames > 0 and kv_frame < kv_start_frame + sink_frames:
        return True
    width = _window_width(
        distance,
        tokens_per_frame,
        decay_factor=decay_factor,
        block_size=block_size,
        model_type=model_type,
    )
    if width >= tokens_per_frame:
        return True
    split_factor = _split_factor(distance, tokens_per_frame)
    return abs(q_token - kv_token) <= width and distance % split_factor == 0


def _token_mask(
    *,
    query_tokens: int,
    kv_tokens: int,
    tokens_per_frame: int,
    query_start_frame: int,
    kv_start_frame: int,
    config: RadialAttentionConfig,
    device: torch.device,
) -> torch.Tensor:
    """Build a small per-frame mask, never a Q×KV production mask."""
    q_frames = (query_tokens + tokens_per_frame - 1) // tokens_per_frame
    kv_frames = (kv_tokens + tokens_per_frame - 1) // tokens_per_frame
    mask = torch.zeros((q_frames, kv_frames, tokens_per_frame, tokens_per_frame), dtype=torch.bool)
    q_tokens = torch.arange(tokens_per_frame, dtype=torch.int64)[:, None]
    kv_tokens_axis = torch.arange(tokens_per_frame, dtype=torch.int64)[None, :]
    for qf in range(q_frames):
        q_len = min(tokens_per_frame, query_tokens - qf * tokens_per_frame)
        for kf in range(kv_frames):
            k_len = min(tokens_per_frame, kv_tokens - kf * tokens_per_frame)
            if q_len <= 0 or k_len <= 0:
                continue
            distance = abs((qf + query_start_frame) - (kf + kv_start_frame))
            width = _window_width(
                distance,
                tokens_per_frame,
                decay_factor=config.decay_factor,
                block_size=config.block_size,
                model_type=config.model_type,
            )
            if config.sink_frames > 0 and kf + kv_start_frame < kv_start_frame + config.sink_frames:
                allowed = torch.ones((q_len, k_len), dtype=torch.bool)
            elif width >= tokens_per_frame:
                allowed = torch.ones((q_len, k_len), dtype=torch.bool)
            else:
                split_factor = _split_factor(distance, tokens_per_frame)
                allowed = (q_tokens[:q_len] - kv_tokens_axis[:, :k_len]).abs() <= width
                allowed &= distance % split_factor == 0
            mask[qf, kf, :q_len, :k_len] = allowed
    return mask


def _block_metadata(mask: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return partial/full row counts and ordered indices for a rectangular mask."""
    q_tokens, kv_tokens = mask.shape
    q_blocks = (q_tokens + block_size - 1) // block_size
    kv_blocks = (kv_tokens + block_size - 1) // block_size
    padded = torch.zeros((q_blocks * block_size, kv_blocks * block_size), dtype=torch.bool)
    padded[:q_tokens, :kv_tokens] = mask
    block_view = padded.view(q_blocks, block_size, kv_blocks, block_size)
    any_blocks = block_view.any(dim=3).any(dim=1)
    full_blocks = block_view.all(dim=3).all(dim=1)

    def ordered(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts = blocks.sum(dim=-1, dtype=torch.int32)
        indices = torch.argsort(blocks.to(torch.int32), dim=-1, descending=True, stable=True).to(torch.int32)
        return counts, indices

    partial = any_blocks & ~full_blocks
    partial_counts, partial_indices = ordered(partial)
    full_counts, full_indices = ordered(full_blocks)
    return partial_counts, partial_indices, full_counts, full_indices


@lru_cache(maxsize=128)
def _cached_token_mask(
    query_tokens: int,
    kv_tokens: int,
    tokens_per_frame: int,
    query_start_frame: int,
    kv_start_frame: int,
    config_key: tuple[object, ...],
) -> torch.Tensor:
    config = RadialAttentionConfig(
        enabled=bool(config_key[0]),
        decay_factor=float(config_key[1]),
        sink_frames=int(config_key[2]),
        block_size=int(config_key[3]),
        model_type=str(config_key[4]),
    )
    return _token_mask(
        query_tokens=query_tokens,
        kv_tokens=kv_tokens,
        tokens_per_frame=tokens_per_frame,
        query_start_frame=query_start_frame,
        kv_start_frame=kv_start_frame,
        config=config,
        device=torch.device("cpu"),
    )


def build_radial_block_mask(
    *,
    query_tokens: int,
    kv_tokens: int,
    tokens_per_frame: int,
    query_start_frame: int,
    kv_start_frame: int,
    config: RadialAttentionConfig,
    device: torch.device | str,
):
    """Build a rectangular FlexAttention ``BlockMask`` and its padding sizes."""
    if BlockMask is None:
        raise RuntimeError("Radial FlexAttention requires torch.nn.attention.flex_attention")
    if not config.enabled:
        raise ValueError("build_radial_block_mask requires config.enabled=True")
    if min(query_tokens, kv_tokens, tokens_per_frame) <= 0:
        raise ValueError("attention lengths and tokens_per_frame must be positive")
    if config.block_size <= 0:
        raise ValueError("radial block_size must be positive")
    return _cached_radial_block_mask(
        query_tokens,
        kv_tokens,
        tokens_per_frame,
        query_start_frame,
        kv_start_frame,
        config.key(),
        str(torch.device(device)),
    )


@lru_cache(maxsize=128)
def _cached_radial_block_mask(
    query_tokens: int,
    kv_tokens: int,
    tokens_per_frame: int,
    query_start_frame: int,
    kv_start_frame: int,
    config_key: tuple[object, ...],
    device_name: str,
):
    """Cache BlockMask metadata and its device copies across layers and steps."""
    config = RadialAttentionConfig(
        enabled=bool(config_key[0]),
        decay_factor=float(config_key[1]),
        sink_frames=int(config_key[2]),
        block_size=int(config_key[3]),
        model_type=str(config_key[4]),
    )
    device = torch.device(device_name)
    mask = _cached_token_mask(
        query_tokens,
        kv_tokens,
        tokens_per_frame,
        query_start_frame,
        kv_start_frame,
        config_key,
    )
    # ``_token_mask`` is grouped as [Q-frame, KV-frame, Q-token, KV-token].
    # Interleave the frame axes with their token axes before presenting it to
    # BlockMask, whose metadata is indexed by global Q/KV token positions.
    flat_mask = mask.permute(0, 2, 1, 3).reshape(
        mask.shape[0] * mask.shape[2],
        mask.shape[1] * mask.shape[3],
    )
    partial_counts, partial_indices, full_counts, full_indices = _block_metadata(
        flat_mask,
        config.block_size,
    )
    q_padded = (-query_tokens) % config.block_size
    kv_padded = (-kv_tokens) % config.block_size

    def mask_mod(b, h, q_idx, kv_idx):
        q_idx = q_idx.to(torch.int64)
        kv_idx = kv_idx.to(torch.int64)
        valid = (q_idx < query_tokens) & (kv_idx < kv_tokens)
        q_frame = q_idx // tokens_per_frame + query_start_frame
        kv_frame = kv_idx // tokens_per_frame + kv_start_frame
        q_token = q_idx % tokens_per_frame
        kv_token = kv_idx % tokens_per_frame
        distance = (q_frame - kv_frame).abs()
        # bit_length for positive tensor values: floor(log2(x)) + 1.
        group = torch.floor(torch.log2(distance.clamp_min(1).to(torch.float32))).to(torch.int64) + 1
        width = (
            2.0 ** tokens_per_frame.bit_length()
            / torch.pow(2.0, group.to(torch.float32))
            * float(config.decay_factor)
        ).clamp_min(float(config.block_size))
        adjacent = distance <= 1
        dense = adjacent | (width >= tokens_per_frame)
        decay_length = (
            2.0 ** tokens_per_frame.bit_length()
            / torch.pow(2.0, group.to(torch.float32))
        )
        split_factor = torch.where(
            decay_length >= float(_UPSTREAM_SPLIT_THRESHOLD),
            torch.ones_like(group),
            torch.floor(float(_UPSTREAM_SPLIT_THRESHOLD) / decay_length)
            .to(torch.int64)
            .clamp_min(1),
        )
        radial = ((q_token - kv_token).abs().to(torch.float32) <= width) & (
            distance.remainder(split_factor) == 0
        )
        sink = kv_frame < kv_start_frame + int(config.sink_frames)
        padding_self = (q_idx >= query_tokens) & (kv_idx == 0)
        return ((valid & (sink | dense | radial)) | padding_self).to(torch.bool)

    block_mask = BlockMask.from_kv_blocks(
        partial_counts[None, None].to(device=device, non_blocking=True),
        partial_indices[None, None].to(device=device, non_blocking=True),
        full_counts[None, None].to(device=device, non_blocking=True),
        full_indices[None, None].to(device=device, non_blocking=True),
        BLOCK_SIZE=config.block_size,
        mask_mod=mask_mod,
        seq_lengths=(query_tokens + q_padded, kv_tokens + kv_padded),
    )
    return block_mask, q_padded, kv_padded, mask


def build_sage_block_indices(
    *,
    query_tokens: int,
    kv_tokens: int,
    tokens_per_frame: int,
    query_start_frame: int,
    kv_start_frame: int,
    config: RadialAttentionConfig,
    device: torch.device | str,
    kv_block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sparse KV indices for 128-token query blocks."""
    if not config.enabled:
        raise ValueError("build_sage_block_indices requires config.enabled=True")
    if kv_block_size <= 0:
        raise ValueError("kv_block_size must be positive")
    return _cached_sage_block_indices(
        query_tokens,
        kv_tokens,
        tokens_per_frame,
        query_start_frame,
        kv_start_frame,
        config.key(),
        str(torch.device(device)),
        int(kv_block_size),
    )


@lru_cache(maxsize=128)
def _cached_sage_block_indices(
    query_tokens: int,
    kv_tokens: int,
    tokens_per_frame: int,
    query_start_frame: int,
    kv_start_frame: int,
    config_key: tuple[object, ...],
    device_name: str,
    kv_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache selected Sage blocks, including their persistent device copies."""
    config = RadialAttentionConfig(
        enabled=bool(config_key[0]),
        decay_factor=float(config_key[1]),
        sink_frames=int(config_key[2]),
        block_size=int(config_key[3]),
        model_type=str(config_key[4]),
    )
    device = torch.device(device_name)
    mask = _cached_token_mask(
        query_tokens,
        kv_tokens,
        tokens_per_frame,
        query_start_frame,
        kv_start_frame,
        config.key(),
    ).permute(0, 2, 1, 3).reshape(query_tokens, kv_tokens)
    q_blocks = (query_tokens + 127) // 128
    kv_blocks = (kv_tokens + kv_block_size - 1) // kv_block_size
    selected = torch.zeros((q_blocks, kv_blocks), dtype=torch.bool)
    padded_mask = torch.zeros((query_tokens, kv_blocks * kv_block_size), dtype=torch.bool)
    padded_mask[:, :kv_tokens] = mask
    for qb in range(q_blocks):
        q_slice = padded_mask[qb * 128 : min((qb + 1) * 128, query_tokens)]
        selected[qb] = (
            q_slice.any(dim=0).view(kv_blocks, kv_block_size).any(dim=1)
        )
    counts = selected.sum(dim=1, dtype=torch.int32)
    max_blocks = int(counts.max().item())
    indices = torch.argsort(selected.to(torch.int32), dim=1, descending=True, stable=True)[:, :max_blocks]
    return (
        counts.to(device=device, non_blocking=True),
        indices.to(device=device, dtype=torch.int32, non_blocking=True),
    )


def radial_density(mask: torch.Tensor) -> float:
    """Return selected token density for diagnostics."""
    return float(mask.float().mean().item())
