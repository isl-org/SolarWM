"""Activation capture hooks and Hessian collection per layer and timestep."""

from __future__ import annotations

import collections
from typing import Any, Mapping

import torch
import torch.nn as nn


class HessianCollector:
    """Accumulates activation covariance H = X^T X and per-channel absmax on CPU/GPU."""

    def __init__(self, in_features: int, max_subsample_tokens: int = 4096) -> None:
        self.in_features = in_features
        self.max_subsample_tokens = max_subsample_tokens
        self.hessian = torch.zeros((in_features, in_features), dtype=torch.float32)
        self.absmax = torch.zeros(in_features, dtype=torch.float32)
        self.total_tokens = 0
        self.subsample_tokens: list[torch.Tensor] = []
        self._current_subsample_count = 0

    def add_batch(self, x: torch.Tensor) -> None:
        """Add activation batch x [..., in_features]."""
        x_flat = x.detach().reshape(-1, self.in_features).float()
        tokens = x_flat.shape[0]
        if tokens == 0:
            return

        # Update running absmax
        batch_amax = x_flat.abs().amax(dim=0).cpu()
        self.absmax = torch.maximum(self.absmax, batch_amax)

        # Update Hessian on tensor device then accumulate to CPU
        h_batch = (x_flat.T @ x_flat).cpu()
        self.hessian += h_batch
        self.total_tokens += tokens

        # Retain subsample for activation clipping policy search
        if self._current_subsample_count < self.max_subsample_tokens:
            needed = self.max_subsample_tokens - self._current_subsample_count
            take = min(needed, tokens)
            self.subsample_tokens.append(x_flat[:take].cpu().clone())
            self._current_subsample_count += take

    def get_subsample(self) -> torch.Tensor:
        if not self.subsample_tokens:
            return torch.zeros((0, self.in_features), dtype=torch.float32)
        return torch.cat(self.subsample_tokens, dim=0)

    def normalize(self) -> None:
        if self.total_tokens > 0:
            self.hessian /= float(self.total_tokens)


class ActivationCaptureHook:
    """Hooks into Wan2.2 DiT blocks to capture activations per block and timestep bin."""

    def __init__(
        self,
        num_blocks: int = 30,
        timestep_bins: tuple[str, ...] = ("0", "1", "2", "3", "commit"),
        subsample_tokens_per_bin: int = 4096,
    ) -> None:
        self.num_blocks = num_blocks
        self.timestep_bins = timestep_bins
        self.current_bin: str = timestep_bins[0]
        self._handles: list[Any] = []

        # Collectors keyed by (block_idx, layer_key, bin)
        # layer_key is "x" (for q, k, v, ffn.0) or "ffn_down" (for ffn.2)
        self.stats: dict[tuple[int, str, str], HessianCollector] = collections.defaultdict(
            lambda: None  # initialized lazily with appropriate in_features
        )
        self._subsample_limit = subsample_tokens_per_bin

    def set_timestep_bin(self, bin_name: str) -> None:
        assert bin_name in self.timestep_bins, f"Unknown bin: {bin_name} (expected {self.timestep_bins})"
        self.current_bin = bin_name

    def _get_collector(self, block_idx: int, target: str, in_features: int) -> HessianCollector:
        key = (block_idx, target, self.current_bin)
        col = self.stats.get(key)
        if col is None:
            col = HessianCollector(in_features=in_features, max_subsample_tokens=self._subsample_limit)
            self.stats[key] = col
        return col

    def register(self, model: nn.Module) -> None:
        """Register forward pre-hooks on blocks and ffn.2 linears."""
        self.remove()

        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise ValueError("Model does not expose blocks attribute")

        for b_idx in range(len(blocks)):
            block = blocks[b_idx]

            # 1. Hook block input x (feeds q, k, v, ffn.0)
            def make_block_hook(b_idx=b_idx):
                def _hook(mod: nn.Module, inp: tuple[Any, ...]) -> None:
                    x = inp[0]
                    if isinstance(x, torch.Tensor):
                        col = self._get_collector(b_idx, "x", in_features=x.shape[-1])
                        col.add_batch(x)
                return _hook

            self._handles.append(block.register_forward_pre_hook(make_block_hook()))

            # 2. Hook ffn.2 input (h_ffn, after GELU)
            ffn = getattr(block, "ffn", None)
            if ffn is not None and hasattr(ffn, "2"):
                ffn_down = getattr(ffn, "2")

                def make_ffn_down_hook(b_idx=b_idx):
                    def _hook(mod: nn.Module, inp: tuple[Any, ...]) -> None:
                        h = inp[0]
                        if isinstance(h, torch.Tensor):
                            col = self._get_collector(b_idx, "ffn_down", in_features=h.shape[-1])
                            col.add_batch(h)
                    return _hook

                self._handles.append(ffn_down.register_forward_pre_hook(make_ffn_down_hook()))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
