"""Low-latency, one-latent-at-a-time Wan Stage2 inference.

This module deliberately does not call the batch rollout in ``stage2.py``.
The batch sampler owns its caches for the duration of a complete rollout,
whereas an interactive viewer must keep those caches alive between browser
control messages.  The numerical operations below mirror the inference half
of that sampler and use the provider's public-ish adapter methods.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from solarwm.errors import BackendContractError


@dataclass(frozen=True)
class InteractiveFrame:
    """A decoded frame tile and the camera state that produced it."""

    index: int
    pixels: Any
    pose: Any
    generated_at: float
    diffusion_seconds: float
    decode_seconds: float


class Stage2InteractiveSession:
    """Persistent NFE4 Stage2 sampler with one latent frame per step.

    The first step commits the encoded TI2V anchor to the KV cache.  Every
    subsequent step consumes one camera latent frame, runs the four denoise
    forwards and a detached commit forward, then decodes that latent through
    one continuous VAE cache.  ``camera`` is a one-frame token mapping with
    ``[1, frame_sequence_length, ...]`` tensors.
    """

    def __init__(
        self,
        provider: Any,
        *,
        first_latent: Any,
        condition: Mapping[str, Any],
        generator: Any,
        max_latent_frames: int,
        decode_tile: Any,
        first_pose: Any | None = None,
    ) -> None:
        import torch

        self.provider = provider
        self.device = provider.device
        self.first_latent = first_latent[:, :1].contiguous()
        self.condition = dict(condition)
        self.generator = generator
        self.max_latent_frames = int(max_latent_frames)
        self.frame_sequence_length = int(provider.config["model"]["frame_sequence_length"])
        self.dtype = self.first_latent.dtype
        self._decode_tile = decode_tile
        self._first_pose = first_pose
        self._index = 0
        self._closed = False
        self._kv_cache = provider.allocate_kv_cache(
            1, dtype=self.dtype, device=self.device
        )
        self._crossattn_cache = provider.allocate_crossattn_cache(
            1, dtype=self.dtype, device=self.device
        )
        self._steps = self._generation_steps()
        if len(self._steps) != 4:
            raise BackendContractError("interactive Stage2 requires self_forcing NFE4")
        channels = int(provider.config["model"]["latent_channels"])
        height = int(provider.config["data"]["latent_shape"][-2])
        width = int(provider.config["data"]["latent_shape"][-1])
        self._noise = torch.empty(
            (1, 1, channels, height, width),
            device=self.device,
            dtype=self.dtype,
        )
        self._denoise_noise = torch.empty_like(self._noise)
        self._frames = 0

    @classmethod
    def from_provider(
        cls,
        provider: Any,
        *,
        first_latent: Any,
        condition: Mapping[str, Any],
        seed: int,
        max_latent_frames: int,
        decode_tile: Any,
        first_pose: Any | None = None,
    ) -> "Stage2InteractiveSession":
        import torch

        generator = torch.Generator(device=provider.device)
        generator.manual_seed(int(seed))
        return cls(
            provider,
            first_latent=first_latent,
            condition=condition,
            generator=generator,
            max_latent_frames=max_latent_frames,
            decode_tile=decode_tile,
            first_pose=first_pose,
        )

    def _generation_steps(self) -> tuple[Any, ...]:
        from .stage2 import _generation_steps

        return _generation_steps(self.provider)

    @property
    def latent_index(self) -> int:
        return self._index

    @property
    def emitted_pixel_frames(self) -> int:
        return self._frames

    @property
    def done(self) -> bool:
        return self._closed or self._index >= self.max_latent_frames

    def step(self, camera: Mapping[str, Any], *, pose: Any | None = None) -> InteractiveFrame:
        """Generate and decode one latent frame.

        ``camera`` must already be relative-W2C Wan camera tokens for the
        current latent.  It is intentionally supplied at the step boundary so
        a caller can coalesce keyboard state while the previous frame runs.
        """

        import torch

        if self._closed:
            raise BackendContractError("interactive Stage2 session is closed")
        if self.done:
            raise BackendContractError("interactive Stage2 session reached its horizon")
        frame = self._index
        started = time.perf_counter()
        self._fill_noise(self._noise)
        latents = self._noise
        if frame == 0:
            latents[:, 0] = self.first_latent[:, 0]
        for step_index, step_value in enumerate(self._steps):
            timestep = torch.full(
                (1, 1),
                float(step_value),
                device=self.device,
                dtype=self.dtype,
            )
            if frame == 0:
                timestep[:, 0] = 0.0
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type in ("cuda", "xpu"),
            ):
                flow = self.provider.diffusion(
                    latents,
                    self.condition,
                    camera,
                    self._expand_timestep(timestep),
                    sequence_length=self.frame_sequence_length,
                    kv_cache=self._kv_cache,
                    crossattn_cache=self._crossattn_cache,
                    current_start=frame * self.frame_sequence_length,
                    cache_start=0,
                    cache_update_policy="inference_direct",
                )
                x0 = self.provider.diffusion.flow_to_x0(latents, flow, timestep)
            if frame == 0:
                x0 = self._restore_first(x0)
            if step_index + 1 < len(self._steps):
                self.provider._noise_into(self._denoise_noise, self.generator)
                next_timestep = torch.full(
                    (1, 1),
                    float(self._steps[step_index + 1]),
                    device=self.device,
                    dtype=torch.float32,
                )
                if frame == 0:
                    next_timestep[:, 0] = 0.0
                latents = (
                    self.provider.diffusion.scheduler.add_noise(
                        x0.flatten(0, 1).float(),
                        self._denoise_noise.flatten(0, 1).float(),
                        next_timestep.flatten(0, 1),
                    )
                    .unflatten(0, (1, 1))
                    .to(self.dtype)
                )
                if frame == 0:
                    latents = self._restore_first(latents)
            else:
                latents = x0
        diffusion_seconds = time.perf_counter() - started
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type in ("cuda", "xpu"),
        ):
            self.provider.diffusion(
                latents,
                self.condition,
                camera,
                self._expand_timestep(torch.zeros((1, 1), device=self.device, dtype=self.dtype)),
                sequence_length=self.frame_sequence_length,
                kv_cache=self._kv_cache,
                crossattn_cache=self._crossattn_cache,
                current_start=frame * self.frame_sequence_length,
                cache_start=0,
                cache_update_policy="commit_detached",
            )
        decode_started = time.perf_counter()
        pixels = self._decode_tile(latents)
        decode_seconds = time.perf_counter() - decode_started
        if not bool(torch.isfinite(latents).all().item()):
            raise BackendContractError("interactive Stage2 rollout produced non-finite latents")
        self._index += 1
        self._frames += int(pixels.shape[1])
        return InteractiveFrame(
            index=frame,
            pixels=pixels,
            pose=pose,
            generated_at=time.time(),
            diffusion_seconds=diffusion_seconds,
            decode_seconds=decode_seconds,
        )

    def _fill_noise(self, target: Any) -> None:
        self.provider._noise_into(target, self.generator)

    def _expand_timestep(self, timestep: Any) -> Any:
        from .stage0p5 import expand_timesteps_to_tokens

        return expand_timesteps_to_tokens(timestep, self.frame_sequence_length)

    def _restore_first(self, value: Any) -> Any:
        result = value.clone()
        result[:, :1] = self.first_latent
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._kv_cache = None
        self._crossattn_cache = None
        self._noise = None
        self._denoise_noise = None

    def __enter__(self) -> "Stage2InteractiveSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = ["InteractiveFrame", "Stage2InteractiveSession"]
