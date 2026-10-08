from __future__ import annotations

import torch

from solarwm.backends.wan22.runtime.interactive import Stage2InteractiveSession


class _Scheduler:
    def add_noise(self, x0, noise, timestep):
        return x0 + noise * timestep.reshape(-1, 1)


class _Diffusion:
    scheduler = _Scheduler()

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        latents,
        condition,
        camera,
        timestep,
        *,
        sequence_length,
        kv_cache,
        crossattn_cache,
        current_start,
        cache_start,
        cache_update_policy,
    ):
        self.calls.append((int(current_start), cache_update_policy, kv_cache, crossattn_cache))
        return torch.zeros_like(latents)

    @staticmethod
    def flow_to_x0(latents, flow, timestep):
        return latents


class _Provider:
    device = torch.device("cpu")
    dtype = torch.bfloat16
    denoising_steps = (1.0, 0.75, 0.5, 0.25)
    config = {
        "model": {"frame_sequence_length": 2, "latent_channels": 1},
        "data": {"latent_shape": [1, 1, 2, 2]},
    }

    def __init__(self):
        self.diffusion = _Diffusion()

    def allocate_kv_cache(self, batch_size, *, dtype, device):
        return [{"cache": "kv"}]

    def allocate_crossattn_cache(self, batch_size, *, dtype, device):
        return [{"cache": "cross"}]

    def _noise_into(self, value, generator):
        value.copy_(torch.randn(value.shape, generator=generator, dtype=value.dtype))


def _camera():
    return {
        "viewmats": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1),
        "K": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
    }


def test_interactive_session_keeps_caches_and_decodes_each_latent():
    provider = _Provider()
    decoded = []

    def decode(latents):
        decoded.append(latents.clone())
        return torch.zeros((1, 1 if len(decoded) == 1 else 4, 3, 2, 2))

    first = torch.ones((1, 1, 1, 2, 2), dtype=torch.bfloat16)
    session = Stage2InteractiveSession.from_provider(
        provider,
        first_latent=first,
        condition={"prompt_embeds": torch.zeros(1)},
        seed=7,
        max_latent_frames=3,
        decode_tile=decode,
    )
    first_frame = session.step(_camera())
    second_frame = session.step(_camera())

    assert first_frame.index == 0
    assert second_frame.index == 1
    assert len(decoded) == 2
    assert session.latent_index == 2
    assert session.emitted_pixel_frames == 5
    assert len(provider.diffusion.calls) == 10
    assert provider.diffusion.calls[0][2] is provider.diffusion.calls[4][2]
    assert provider.diffusion.calls[5][3] is provider.diffusion.calls[9][3]
    assert provider.diffusion.calls[5][0] == 2
    assert provider.diffusion.calls[9][0] == 2
    session.close()
    assert session.done
