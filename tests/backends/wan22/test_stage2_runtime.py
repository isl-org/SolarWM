from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from solarwm.backends.wan22.generation import GenerationPass
from solarwm.backends.wan22.runtime.stage2 import (
    RoleCheckpointReceipt,
    Wan5BStage2Runtime,
    _Stage2InferenceMeasurements,
    _allocate_stage2_inference_buffers,
    _compile_stage2_xpu_transformer_blocks,
    _published_default_weight_role,
    _stage2_xpu_vae_pipeline_enabled,
    _stage2_generated_sample,
    _stage2_initialization_receipt,
    _stage2_self_forcing_latents,
    _verified_stage2_inference_checkpoint,
    load_role_checkpoint,
    load_stage2_checkpoint,
    save_stage2_checkpoint,
    stage2_camera_rollout,
    verify_role_checkpoint,
)
from solarwm.backends.wan22.sgf import RoleInitialization
from solarwm.checkpoint import CheckpointTransaction
from solarwm.errors import BackendContractError


class _FakeScheduler:
    def add_noise(self, clean: object, noise: object, timestep: object) -> object:
        del noise, timestep
        return clean


def test_stage2_xpu_compile_wraps_only_transformer_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    compiled: list[object] = []
    blocks = torch.nn.ModuleList((torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)))
    diffusion = SimpleNamespace(module=SimpleNamespace(blocks=blocks))
    monkeypatch.setattr(torch, "compile", lambda block, **kwargs: compiled.append((block, kwargs)) or block)
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)

    _compile_stage2_xpu_transformer_blocks(diffusion, device=torch.device("xpu"))

    assert [kwargs for _, kwargs in compiled] == [{"dynamic": False}, {"dynamic": False}]
    assert diffusion.module.blocks is not blocks
    assert diffusion.module._solarwm_stage2_xpu_blocks_compiled is True
    _compile_stage2_xpu_transformer_blocks(diffusion, device=torch.device("xpu"))
    assert len(compiled) == 2

    # Test passing mode="max-autotune"
    compiled.clear()
    blocks2 = torch.nn.ModuleList((torch.nn.Linear(2, 2),))
    diffusion2 = SimpleNamespace(module=SimpleNamespace(blocks=blocks2))
    _compile_stage2_xpu_transformer_blocks(
        diffusion2, device=torch.device("xpu"), mode="max-autotune"
    )
    assert [kwargs for _, kwargs in compiled] == [{"dynamic": False, "mode": "max-autotune"}]


def test_stage2_xpu_compile_rejects_non_xpu_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    blocks = torch.nn.ModuleList((torch.nn.Linear(2, 2),))
    diffusion = SimpleNamespace(module=SimpleNamespace(blocks=blocks))
    monkeypatch.setattr(torch, "compile", lambda *_args, **_kwargs: pytest.fail("must not compile"))

    with pytest.raises(BackendContractError, match="requires inference.device=xpu"):
        _compile_stage2_xpu_transformer_blocks(diffusion, device=torch.device("cpu"))
    assert diffusion.module.blocks is blocks


def test_stage2_xpu_vae_pipeline_uses_serial_cpu_fallback() -> None:
    torch = pytest.importorskip("torch")
    provider = SimpleNamespace(
        device=torch.device("cpu"),
        config={
            "action": "infer",
            "inference": {"length": "camera", "stage2_xpu_vae_pipeline": True},
        },
        vae=SimpleNamespace(streaming_decode_session=lambda: pytest.fail("must not open on CPU")),
    )

    assert not _stage2_xpu_vae_pipeline_enabled(
        provider,
        output_latent_frames=60,
        rollout_latent_frames=60,
    )


def test_stage2_measurements_sync_and_reset_xpu_per_phase(tmp_path: Path) -> None:
    class FakeXpu:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def synchronize(self, device: object) -> None:
            self.calls.append(("synchronize", device))

        def reset_peak_memory_stats(self, device: object) -> None:
            self.calls.append(("reset", device))

        @staticmethod
        def max_memory_allocated(_device: object) -> int:
            return 123

        @staticmethod
        def max_memory_reserved(_device: object) -> int:
            return 456

    device = SimpleNamespace(type="xpu")
    xpu = FakeXpu()
    measurements = _Stage2InferenceMeasurements(
        {
            "action": "infer",
            "name": "measurement-test",
            "inference": {"length": "camera", "run_id": "measurement-test"},
            "runtime": {
                "output_dir": str(tmp_path),
                "stage2_inference_measurements": True,
            },
        },
        device=device,
        rank=3,
        torch_module=SimpleNamespace(xpu=xpu),
    )

    with measurements.phase("rollout_chunk", chunk_index=2):
        pass

    assert [name for name, _ in xpu.calls] == ["synchronize", "reset", "synchronize"]
    assert measurements.events[0]["phase"] == "rollout_chunk"
    assert measurements.events[0]["synchronized"] is True
    assert measurements.events[0]["memory"] == {
        "max_allocated_bytes": 123,
        "max_reserved_bytes": 456,
    }
    report = measurements.write_report()
    assert report == tmp_path / "runs" / "measurement-test" / "reports" / (
        "stage2-inference-measurements.rank-00003.jsonl"
    )
    assert '"phase":"rollout_chunk"' in report.read_text(encoding="utf-8")


def test_stage2_measurements_can_record_async_pipeline_telemetry(tmp_path: Path) -> None:
    class FakeXpu:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def synchronize(self, _device: object) -> None:
            self.calls.append("synchronize")

        def reset_peak_memory_stats(self, _device: object) -> None:
            self.calls.append("reset")

    xpu = FakeXpu()
    measurements = _Stage2InferenceMeasurements(
        {
            "action": "infer",
            "inference": {"length": "camera", "run_id": "async-measurement-test"},
            "runtime": {
                "output_dir": str(tmp_path),
                "stage2_inference_measurements": True,
            },
        },
        device=SimpleNamespace(type="xpu"),
        rank=0,
        torch_module=SimpleNamespace(xpu=xpu),
    )

    with measurements.phase("rollout_chunk", synchronize=False):
        pass
    measurements.record("vae_pipeline", tiles=14, wait_seconds=0.0)

    assert xpu.calls == ["reset"]
    assert measurements.events[0]["synchronized"] is False
    assert measurements.events[1]["phase"] == "vae_pipeline"
    assert measurements.events[1]["tiles"] == 14


def test_stage2_measurements_are_inert_when_disabled() -> None:
    xpu = SimpleNamespace(
        synchronize=lambda _device: pytest.fail("disabled measurements must not synchronize"),
        reset_peak_memory_stats=lambda _device: pytest.fail("disabled measurements must not reset"),
    )
    measurements = _Stage2InferenceMeasurements(
        {"runtime": {"stage2_inference_measurements": False}},
        device=SimpleNamespace(type="xpu"),
        rank=0,
        torch_module=SimpleNamespace(xpu=xpu),
    )

    with measurements.phase("rollout_chunk"):
        pass

    assert measurements.events == []
    assert measurements.write_report() is None


def _cache_alloc(batch_size: int, *, dtype: object, device: object) -> list[dict]:
    torch = pytest.importorskip("torch")
    return [
        {
            "k": torch.zeros((batch_size, 1), dtype=dtype, device=device),
            "v": torch.zeros((batch_size, 1), dtype=dtype, device=device),
            "global_end_index": 0,
            "local_end_index": 0,
            "_fused_prope_camera_metadata": {
                "viewmats": torch.eye(4, device=device).reshape(1, 1, 4, 4),
                "K": torch.eye(3, device=device).reshape(1, 1, 3, 3),
            },
        }
    ]


def test_kv_cache_indices_use_host_ints_and_accept_legacy_tensors() -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime.modeling.causal_model import (
        _cache_index,
        _set_cache_index,
    )

    host_cache = {"global_end_index": 3, "local_end_index": 5}
    assert _cache_index(host_cache, "global_end_index") == 3
    _set_cache_index(host_cache, "local_end_index", 7)
    assert host_cache["local_end_index"] == 7

    legacy_cache = {
        "global_end_index": torch.tensor([3], dtype=torch.long),
        "local_end_index": torch.tensor([5], dtype=torch.long),
    }
    assert _cache_index(legacy_cache, "global_end_index") == 3
    _set_cache_index(legacy_cache, "local_end_index", 7)
    assert legacy_cache["local_end_index"].tolist() == [7]


def test_stage2_kv_cache_allocator_uses_host_index_scalars() -> None:
    torch = pytest.importorskip("torch")

    class _Root:
        blocks = (object(),)
        num_heads = 2
        dim = 4

    runtime = Wan5BStage2Runtime.__new__(Wan5BStage2Runtime)
    runtime._kv_cache_allocator = None
    runtime.student = SimpleNamespace(module=_Root())
    runtime.config = {
        "model": {
            "local_attn_size": 2,
            "frame_sequence_length": 3,
        }
    }

    cache = runtime.allocate_kv_cache(1, dtype=torch.float32, device=torch.device("cpu"))[0]

    assert cache["global_end_index"] == 0
    assert cache["local_end_index"] == 0
    assert type(cache["global_end_index"]) is int
    assert type(cache["local_end_index"]) is int


def test_stage2_camera_circular_allocator_shapes() -> None:
    torch = pytest.importorskip("torch")

    class _Root:
        blocks = (object(),)
        num_heads = 2
        dim = 4

    runtime = Wan5BStage2Runtime.__new__(Wan5BStage2Runtime)
    runtime._kv_cache_allocator = None
    runtime.student = SimpleNamespace(module=_Root())
    runtime.config = {
        "action": "infer",
        "inference": {"length": "camera"},
        "model": {"local_attn_size": 2, "frame_sequence_length": 3},
    }

    # Default: WS13 unmirrored raw ring + persistent encoded ring
    cache_ws13 = runtime.allocate_kv_cache(1, dtype=torch.float32, device=torch.device("cpu"))[0]
    assert cache_ws13["k"].shape[1] == 6
    assert cache_ws13["v"].shape[1] == 6
    assert cache_ws13["k_encoded"].shape[1] == 6
    assert cache_ws13["v_encoded"].shape[1] == 6
    assert cache_ws13["_circular_kv_cache"] is True
    assert cache_ws13["_circular_capacity"] == 6
    assert cache_ws13["_circular_ring_start"] == 0
    assert type(cache_ws13["_circular_ring_start"]) is int

    # SageAttention: unmirrored raw ring without encoded ring (quantizes on the fly)
    runtime.config["runtime"] = {"stage2_fused_kernel": "fused_rope_prope_sage"}
    cache_sage = runtime.allocate_kv_cache(1, dtype=torch.float32, device=torch.device("cpu"))[0]
    assert cache_sage["k"].shape[1] == 6
    assert cache_sage["v"].shape[1] == 6
    assert "k_encoded" not in cache_sage
    assert "v_encoded" not in cache_sage
    assert cache_sage["_circular_kv_cache"] is True
    assert cache_sage["_circular_capacity"] == 6
    assert cache_sage["_circular_ring_start"] == 0

    # Explicit SDPA: unmirrored raw ring + encoded ring
    runtime.config["runtime"] = {"stage2_fused_kernel": "fused_rope_prope_sdpa"}
    cache_sdpa = runtime.allocate_kv_cache(1, dtype=torch.float32, device=torch.device("cpu"))[0]
    assert cache_sdpa["k"].shape[1] == 6
    assert cache_sdpa["v"].shape[1] == 6
    assert cache_sdpa["k_encoded"].shape[1] == 6
    assert cache_sdpa["v_encoded"].shape[1] == 6
    assert cache_sdpa["_circular_kv_cache"] is True
    assert cache_sdpa["_circular_capacity"] == 6

    # Opt-out: legacy mirrored ring without encoded ring
    runtime.config["runtime"] = {"stage2_encoded_kv_ring": False}
    cache_legacy = runtime.allocate_kv_cache(1, dtype=torch.float32, device=torch.device("cpu"))[0]
    assert cache_legacy["k"].shape[1] == 12
    assert cache_legacy["v"].shape[1] == 12
    assert "k_encoded" not in cache_legacy
    assert "v_encoded" not in cache_legacy
    assert cache_legacy["_circular_kv_cache"] is True
    assert cache_legacy["_circular_capacity"] == 6
    assert cache_legacy["_circular_ring_start"] == 0


def test_mirrored_circular_cache_keeps_wrapped_window_contiguous() -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime.modeling.causal_model import (
        CausalWanModel,
        _write_mirrored_ring,
    )

    cache = {
        "k": torch.zeros((1, 4, 1, 1)),
        "v": torch.zeros((1, 4, 1, 1)),
        "global_end_index": 2,
        "local_end_index": 2,
        "_circular_kv_cache": True,
        "_circular_capacity": 2,
        "_circular_ring_start": 0,
    }
    _write_mirrored_ring(cache["k"], start=0, values=torch.tensor([[[[1.0]], [[2.0]]]]))
    _write_mirrored_ring(cache["v"], start=0, values=torch.tensor([[[[1.0]], [[2.0]]]]))
    _write_mirrored_ring(cache["k"], start=0, values=torch.tensor([[[[3.0]]]]))
    _write_mirrored_ring(cache["v"], start=0, values=torch.tensor([[[[3.0]]]]))

    # A speculative denoise may overwrite ring slots, but it cannot advance
    # logical state until the detached commit has completed.
    assert cache["_circular_ring_start"] == 0
    assert cache["global_end_index"] == 2
    CausalWanModel._apply_cache_updates(
        object(),
        [cache],
        [(0, (3, 2, {"action": "circular_insert", "ring_start": 1}))],
    )
    assert cache["_circular_ring_start"] == 1
    assert cache["global_end_index"] == 3
    assert cache["local_end_index"] == 2
    assert cache["k"][0, 1:3, 0, 0].tolist() == [2.0, 3.0]
    assert cache["v"][0, 1:3, 0, 0].tolist() == [2.0, 3.0]


def test_circular_attention_uses_contiguous_wrapped_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime.modeling import causal_model

    attn = causal_model.CausalWanSelfAttention(
        dim=2,
        num_heads=1,
        local_attn_size=2,
        sink_size=0,
        qk_norm=False,
        frame_seq_length=1,
        use_echorope=False,
    )
    with torch.no_grad():
        for projection in (attn.q, attn.k, attn.v, attn.o):
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()
    monkeypatch.setattr(
        attn,
        "_rope_q_and_window_k",
        lambda *, q, k_window, **_kwargs: (q, k_window, None),
    )
    visible_windows: list[object] = []
    monkeypatch.setattr(
        causal_model,
        "attention",
        lambda q, k, _v: (visible_windows.append(k.detach().clone()) or q),
    )
    cache = {
        "k": torch.zeros((1, 4, 1, 2)),
        "v": torch.zeros((1, 4, 1, 2)),
        "global_end_index": 0,
        "local_end_index": 0,
        "_circular_kv_cache": True,
        "_circular_capacity": 2,
        "_circular_ring_start": 0,
    }
    grid = torch.tensor([[1, 1, 1]])
    for index, value in enumerate((1.0, 2.0, 3.0)):
        _, update = attn(
            torch.tensor([[[value, 0.0]]]),
            torch.tensor([1]),
            grid,
            torch.empty(0),
            cache,
            current_start=index,
            frame_seqlen=1,
            cache_update_policy="inference_direct",
        )
        causal_model.CausalWanModel._apply_cache_updates(
            object(), [cache], [(0, update)]
        )

    assert cache["_circular_ring_start"] == 1
    assert visible_windows[-1][0, :, 0, 0].tolist() == [2.0, 3.0]


def test_encoded_kv_ring_matches_mirrored_circular_cache_multi_chunk() -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime.modeling import causal_model

    torch.manual_seed(123)
    dim = 8
    num_heads = 2
    head_dim = dim // num_heads
    frame_seq_length = 2
    local_attn_size = 4  # capacity = 8 tokens
    chunk_tokens = 2    # 1 frame per chunk = 2 tokens
    capacity_tokens = local_attn_size * frame_seq_length
    dtype = torch.float32

    attn = causal_model.CausalWanSelfAttention(
        dim=dim,
        num_heads=num_heads,
        local_attn_size=local_attn_size,
        sink_size=0,
        qk_norm=False,
        frame_seq_length=frame_seq_length,
        use_echorope=True,
        camera_attention_mode="fused_prope",
    ).to(dtype=dtype)

    cache_old = {
        "k": torch.zeros((1, 2 * capacity_tokens, num_heads, head_dim), dtype=dtype),
        "v": torch.zeros((1, 2 * capacity_tokens, num_heads, head_dim), dtype=dtype),
        "global_end_index": 0,
        "local_end_index": 0,
        "_circular_kv_cache": True,
        "_circular_capacity": capacity_tokens,
        "_circular_ring_start": 0,
        "_fused_prope_camera_metadata": {
            "viewmats": torch.zeros((1, 2 * capacity_tokens, 4, 4), dtype=torch.float32),
            "K": torch.zeros((1, 2 * capacity_tokens, 3, 3), dtype=torch.float32),
        },
    }
    cache_new = {
        "k": torch.zeros((1, capacity_tokens, num_heads, head_dim), dtype=dtype),
        "v": torch.zeros((1, capacity_tokens, num_heads, head_dim), dtype=dtype),
        "k_encoded": torch.zeros((1, capacity_tokens, num_heads, head_dim), dtype=dtype),
        "v_encoded": torch.zeros((1, capacity_tokens, num_heads, head_dim), dtype=dtype),
        "global_end_index": 0,
        "local_end_index": 0,
        "_circular_kv_cache": True,
        "_circular_capacity": capacity_tokens,
        "_circular_ring_start": 0,
        "_fused_prope_camera_metadata": {
            "viewmats": torch.zeros((1, capacity_tokens, 4, 4), dtype=torch.float32),
            "K": torch.zeros((1, capacity_tokens, 3, 3), dtype=torch.float32),
        },
    }

    freqs = torch.randn((50, head_dim // 2), dtype=torch.complex128)
    grid = torch.tensor([[1, 1, 2]])

    for chunk_idx in range(5):
        vm = torch.eye(4).repeat(1, chunk_tokens, 1, 1)
        vm[..., 0, 3] = float(chunk_idx)
        K = torch.eye(3).repeat(1, chunk_tokens, 1, 1)

        prope_old = {}
        prope_new = {}

        for step in range(3):
            is_commit = (step == 2)
            policy = "commit_detached" if is_commit else "inference_direct"
            x = torch.randn(1, chunk_tokens, dim)

            # Apply camera metadata
            dummy_model = SimpleNamespace(
                sink_size=0,
                local_attn_size=local_attn_size,
                _slice_current_camera_tokens=lambda t, **kwargs: t,
            )
            _, _, vm_old, k_cam_old, _ = causal_model.CausalWanModel._stage_fused_camera_cache(
                dummy_model, [cache_old],
                cam_viewmats=vm, cam_K=K,
                current_start=chunk_idx * chunk_tokens,
                num_new_tokens=chunk_tokens,
                frame_seqlen=frame_seq_length,
                cache_update_policy=policy,
            )
            _, _, vm_new, k_cam_new, _ = causal_model.CausalWanModel._stage_fused_camera_cache(
                dummy_model, [cache_new],
                cam_viewmats=vm, cam_K=K,
                current_start=chunk_idx * chunk_tokens,
                num_new_tokens=chunk_tokens,
                frame_seqlen=frame_seq_length,
                cache_update_policy=policy,
            )

            out_old, upd_old = attn(
                x, torch.tensor([chunk_tokens]), grid, freqs,
                cache_old,
                current_start=chunk_idx * chunk_tokens,
                frame_seqlen=frame_seq_length,
                cam_viewmats=vm, cam_K=K,
                kv_cam_viewmats=vm_old, kv_cam_K=k_cam_old,
                prope_cache=prope_old,
                cache_update_policy=policy,
            )
            out_new, upd_new = attn(
                x, torch.tensor([chunk_tokens]), grid, freqs,
                cache_new,
                current_start=chunk_idx * chunk_tokens,
                frame_seqlen=frame_seq_length,
                cam_viewmats=vm, cam_K=K,
                kv_cam_viewmats=vm_new, kv_cam_K=k_cam_new,
                prope_cache=prope_new,
                cache_update_policy=policy,
            )

            torch.testing.assert_close(out_new, out_old, rtol=0, atol=0)

            if is_commit:
                causal_model.CausalWanModel._apply_cache_updates(
                    object(), [cache_old], [(0, upd_old)]
                )
                causal_model.CausalWanModel._apply_cache_updates(
                    object(), [cache_new], [(0, upd_new)]
                )


def test_inference_direct_cache_policy_writes_only_uncommitted_direct_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime.modeling import causal_model

    attn = causal_model.CausalWanSelfAttention(
        dim=2,
        num_heads=1,
        local_attn_size=2,
        sink_size=0,
        qk_norm=False,
        frame_seq_length=1,
        use_echorope=False,
    )
    with torch.no_grad():
        for projection in (attn.q, attn.k, attn.v, attn.o):
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()
    monkeypatch.setattr(
        attn,
        "_rope_q_and_window_k",
        lambda *, q, k_window, **_kwargs: (q, k_window, None),
    )
    monkeypatch.setattr(causal_model, "attention", lambda q, _k, _v: q)

    cache = {
        "k": torch.zeros((1, 2, 1, 2)),
        "v": torch.zeros((1, 2, 1, 2)),
        "global_end_index": 0,
        "local_end_index": 0,
    }
    value = torch.tensor([[[3.0, 5.0]]])
    _, (current_end, local_end, update) = attn(
        value,
        torch.tensor([1]),
        torch.tensor([[1, 1, 1]]),
        torch.empty(0),
        cache,
        current_start=0,
        frame_seqlen=1,
        cache_update_policy="inference_direct",
    )

    assert current_end == local_end == 1
    assert update["action"] == "direct_insert"
    assert cache["global_end_index"] == cache["local_end_index"] == 0
    assert torch.equal(cache["k"][:, :1, 0], value)
    assert torch.equal(cache["v"][:, :1, 0], value)


class _FakeStudent:
    def __init__(self) -> None:
        torch = pytest.importorskip("torch")
        self.parameter = torch.nn.Parameter(torch.tensor(0.5))
        self.scheduler = _FakeScheduler()
        self.calls: list[dict[str, object]] = []
        self.replays: list[dict[str, object]] = []

    def __call__(
        self,
        noisy: object,
        condition: object,
        camera: object,
        timestep: object,
        **kwargs: object,
    ) -> object:
        torch = pytest.importorskip("torch")
        del condition
        self.calls.append(
            {
                "grad": torch.is_grad_enabled(),
                "camera_tokens": camera["K"].shape[1],
                "timestep": timestep.clone(),
                **kwargs,
            }
        )
        return torch.full_like(noisy, 0.25)

    def forward_train_tf(self, noisy: object, *args: object, **kwargs: object) -> object:
        torch = pytest.importorskip("torch")
        del args
        self.replays.append({"grad": torch.is_grad_enabled(), **kwargs})
        return self.parameter.expand_as(noisy)

    @staticmethod
    def flow_to_x0(noisy: object, flow: object, timestep: object) -> object:
        sigma = timestep.float()
        while sigma.ndim < noisy.ndim:
            sigma = sigma.unsqueeze(-1)
        return noisy.float() - sigma * flow.float() / 1000.0


def test_stage2_rollout_is_no_grad_chunks_then_one_gradient_replay() -> None:
    torch = pytest.importorskip("torch")
    student = _FakeStudent()
    noise = torch.ones((1, 6, 1, 1, 1))
    first = torch.full((1, 1, 1, 1, 1), 9.0)
    camera = {
        "viewmats": torch.eye(4).repeat(1, 12, 1, 1),
        "K": torch.eye(3).repeat(1, 12, 1, 1),
    }
    rollout = stage2_camera_rollout(
        student=student,
        allocate_kv_cache=_cache_alloc,
        allocate_crossattn_cache=_cache_alloc,
        noise=noise,
        first_latent=first,
        condition={"prompt_embeds": torch.zeros((1, 1, 1))},
        camera=camera,
        denoising_steps=(
            torch.tensor(1000.0),
            torch.tensor(750.0),
            torch.tensor(500.0),
            torch.tensor(250.0),
        ),
        num_frame_per_block=3,
        frame_sequence_length=2,
        forced_exit_index=1,
        require_grad=True,
        validate_finite=True,
    )

    assert rollout.exit_index == 1
    assert rollout.denoised_timestep_from == 750.0
    assert rollout.denoised_timestep_to == 500.0
    assert len(student.calls) == 10
    assert all(call["grad"] is False for call in student.calls)
    assert [call["cache_update_policy"] for call in student.calls].count("commit_detached") == 2
    assert all(call["camera_tokens"] == 6 for call in student.calls)
    assert len(student.replays) == 1
    assert student.replays[0]["grad"] is True
    assert student.replays[0]["num_frame_per_block"] == 3
    assert rollout.loss_mask[:, 0].logical_not().all()
    assert rollout.loss_mask[:, 1:].all()
    assert torch.equal(rollout.output[:, :1], first)
    assert not rollout.cache_target.requires_grad
    rollout.output[:, 1:].sum().backward()
    assert student.parameter.grad is not None
    assert student.parameter.grad.abs().item() > 0


@pytest.mark.parametrize("latent_frames", (6, 42))
def test_stage2_generation_uses_one_persistent_cache_and_exact_nfe4(
    latent_frames: int,
) -> None:
    torch = pytest.importorskip("torch")
    student = _FakeStudent()
    allocations = {"kv": 0, "cross": 0}
    noise_shapes: list[tuple[int, ...]] = []
    final_tiles: list[tuple[int, int, object]] = []

    def kv_cache(batch_size: int, *, dtype: object, device: object) -> list[dict]:
        allocations["kv"] += 1
        return _cache_alloc(batch_size, dtype=dtype, device=device)

    def cross_cache(batch_size: int, *, dtype: object, device: object) -> list[dict]:
        allocations["cross"] += 1
        return _cache_alloc(batch_size, dtype=dtype, device=device)

    def noise(shape: object, generator: object) -> object:
        del generator
        normalized = tuple(int(value) for value in shape)
        noise_shapes.append(normalized)
        return torch.ones(normalized, dtype=torch.bfloat16)

    provider = SimpleNamespace(
        config={
            "model": {
                "num_frame_per_block": 3,
                "latent_channels": 1,
                "frame_sequence_length": 2,
            },
            "data": {"latent_shape": [latent_frames, 1, 1, 1]},
            "train": {"num_train_timesteps": 1000},
        },
        device=torch.device("cpu"),
        diffusion=student,
        denoising_steps=tuple(
            torch.tensor(value, dtype=torch.float32) for value in (1000, 750, 500, 250)
        ),
        allocate_kv_cache=kv_cache,
        allocate_crossattn_cache=cross_cache,
        _noise=noise,
        _inference_direct_cache_writes=True,
        _stage2_final_latent_tile_callback=lambda value, *, start, end: final_tiles.append(
            (start, end, value.clone())
        ),
    )
    generation_pass = GenerationPass(
        name="live_self_forcing_nfe4",
        weights="live",
        mode="autoregressive",
        solver="self_forcing",
        num_inference_steps=4,
        rollout_latent_frames=latent_frames,
        min_rollout_latent_frames=latent_frames,
        fixed_plan_pixel_frames=1 + 4 * (latent_frames - 1),
        variable_rollout_by_source=False,
    )
    first = torch.full((1, 1, 1, 1, 1), 9.0, dtype=torch.bfloat16)
    camera = {
        "viewmats": torch.eye(4).repeat(1, latent_frames * 2, 1, 1),
        "K": torch.eye(3).repeat(1, latent_frames * 2, 1, 1),
    }
    latents, schedule = _stage2_self_forcing_latents(
        provider,
        generation_pass,
        first,
        {"prompt_embeds": torch.zeros((1, 1, 1))},
        camera,
        torch.Generator(),
    )

    chunks = latent_frames // 3
    assert latents.shape == (1, latent_frames, 1, 1, 1)
    assert torch.equal(latents[:, :1], first)
    assert allocations == {"kv": 1, "cross": 1}
    assert len(student.calls) == 5 * chunks
    assert [call["cache_update_policy"] for call in student.calls].count("inference_direct") == 4 * chunks
    assert [call["cache_update_policy"] for call in student.calls].count("commit_detached") == chunks
    assert [call["current_start"] for call in student.calls] == [
        start * 2 for start in range(0, latent_frames, 3) for _ in range(5)
    ]
    assert all(call["timestep"].dtype == torch.bfloat16 for call in student.calls)
    assert torch.unique(student.calls[1]["timestep"]).tolist() == [0.0, 752.0]
    # Calls 0–4 complete chunk 0. Call 5 begins the next chunk at the
    # schedule's first denoise timestep; the re-noised 752 value remains
    # internal to chunk 0 and is not carried across chunk boundaries.
    assert torch.unique(student.calls[5]["timestep"]).tolist() == [1000.0]
    assert [(start, end) for start, end, _ in final_tiles] == [
        (start, start + 3) for start in range(0, latent_frames, 3)
    ]
    assert torch.equal(torch.cat([value for _, _, value in final_tiles], dim=1), latents)
    assert schedule["timesteps"] == [1000.0, 750.0, 500.0, 250.0]
    assert schedule["persistent_kv_cache"] is True
    assert noise_shapes == [(1, latent_frames, 1, 1, 1)] + [(1, 3, 1, 1, 1)] * (3 * chunks)


def test_stage2_inference_buffers_preserve_rng_draw_shapes_and_output() -> None:
    torch = pytest.importorskip("torch")

    def generate(*, preallocated: bool) -> tuple[object, list[object]]:
        draws: list[object] = []

        def noise(shape: object, generator: object) -> object:
            value = torch.randn(
                tuple(int(item) for item in shape),
                generator=generator,
                dtype=torch.bfloat16,
            )
            draws.append(value.float().clone())
            return value

        def noise_into(value: object, generator: object) -> None:
            torch.randn(tuple(value.shape), generator=generator, out=value)
            draws.append(value.float().clone())

        provider = SimpleNamespace(
            config={
                "model": {
                    "num_frame_per_block": 3,
                    "latent_channels": 1,
                    "frame_sequence_length": 2,
                },
                "data": {"latent_shape": [6, 1, 1, 1]},
                "train": {"num_train_timesteps": 1000},
            },
            device=torch.device("cpu"),
            diffusion=_FakeStudent(),
            denoising_steps=tuple(
                torch.tensor(value, dtype=torch.float32) for value in (1000, 750, 500, 250)
            ),
            allocate_kv_cache=lambda batch_size, *, dtype, device: _cache_alloc(
                batch_size, dtype=dtype, device=device
            ),
            allocate_crossattn_cache=lambda batch_size, *, dtype, device: _cache_alloc(
                batch_size, dtype=dtype, device=device
            ),
            _noise=noise,
            _noise_into=noise_into,
            _inference_direct_cache_writes=True,
            _stage2_inference_buffers=(
                _allocate_stage2_inference_buffers(
                    max_latent_frames=9,
                    chunk_latent_frames=3,
                    channels=1,
                    height=1,
                    width=1,
                    device=torch.device("cpu"),
                )
                if preallocated
                else None
            ),
        )
        generation_pass = GenerationPass(
            name="live_self_forcing_nfe4",
            weights="live",
            mode="autoregressive",
            solver="self_forcing",
            num_inference_steps=4,
            rollout_latent_frames=6,
            min_rollout_latent_frames=6,
            fixed_plan_pixel_frames=21,
            variable_rollout_by_source=False,
        )
        latents, _ = _stage2_self_forcing_latents(
            provider,
            generation_pass,
            torch.full((1, 1, 1, 1, 1), 9.0, dtype=torch.bfloat16),
            {"prompt_embeds": torch.zeros((1, 1, 1))},
            {
                "viewmats": torch.eye(4).repeat(1, 12, 1, 1),
                "K": torch.eye(3).repeat(1, 12, 1, 1),
            },
            torch.Generator().manual_seed(73),
        )
        return latents.clone(), draws

    allocated_latents, allocated_draws = generate(preallocated=False)
    reused_latents, reused_draws = generate(preallocated=True)

    assert torch.equal(reused_latents, allocated_latents)
    assert [value.shape for value in reused_draws] == [value.shape for value in allocated_draws]
    assert all(
        torch.equal(reused.float(), allocated.float())
        for reused, allocated in zip(reused_draws, allocated_draws, strict=True)
    )


def test_stage2_generated_sample_preserves_configured_denoising_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime import inference, stage2

    sentinel = [1000, 625, 250, 125]
    monkeypatch.setattr(
        stage2,
        "_stage2_self_forcing_latents",
        lambda *_: (torch.zeros((1, 1, 1, 1, 1)), {"timesteps": sentinel}),
    )
    monkeypatch.setattr(inference, "_encode_compare_mp4", lambda *_args, **_kwargs: b"compare")
    provider = SimpleNamespace(
        device=torch.device("cpu"),
        config={
            "data": {"fps": 16.0},
            "model": {"camera_translation_transform": "linear"},
            "train": {"denoising_step_list": sentinel},
        },
        _conditions=lambda *_args, **_kwargs: (
            torch.zeros((1, 1, 1, 1, 1)),
            {},
            {},
            None,
        ),
        vae=SimpleNamespace(
            decode=lambda *_args, **_kwargs: torch.zeros((1, 1, 1, 1, 1)),
            decode_streaming=lambda *_args, **_kwargs: torch.zeros((1, 1, 1, 1, 1)),
        ),
        video_encoder=lambda *_args, **_kwargs: b"video",
        _prepared={0: object()},
    )
    case = SimpleNamespace(
        slot=0,
        noise_seed=42,
        metadata={
            "generation_pass": {
                "name": "live_self_forcing_nfe4",
                "weights": "live",
                "mode": "autoregressive",
                "solver": "self_forcing",
                "num_inference_steps": 4,
                "rollout_latent_frames": 1,
            }
        },
    )

    generated = _stage2_generated_sample(provider, case, weights_id="step-100")

    assert generated.provenance["denoising_step_list"] == sentinel


def test_stage2_long_generation_uses_streaming_vae_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime import inference, stage2

    calls: list[int] = []

    class _VAE:
        @staticmethod
        def decode_streaming(value: object, *, chunk_latent_frames: int) -> object:
            calls.extend((int(value.shape[1]), chunk_latent_frames))
            return torch.zeros((1, 957, 3, 1, 1))

        @staticmethod
        def decode(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("long Stage2 generation must not use one-shot VAE decode")

    monkeypatch.setattr(
        stage2,
        "_stage2_self_forcing_latents",
        lambda *_: (
            torch.zeros((1, 240, 1, 1, 1)),
            {"timesteps": [1000, 750, 500, 250]},
        ),
    )
    monkeypatch.setattr(inference, "_encode_compare_mp4", lambda *_args, **_kwargs: b"compare")
    provider = SimpleNamespace(
        device=torch.device("cpu"),
        config={
            "data": {"fps": 16.0},
            "model": {"camera_translation_transform": "linear"},
            "train": {"denoising_step_list": [1000, 750, 500, 250]},
        },
        _conditions=lambda *_args, **_kwargs: (
            torch.zeros((1, 1, 1, 1, 1)),
            {},
            {},
            None,
        ),
        vae=_VAE(),
        video_encoder=lambda *_args, **_kwargs: b"video",
        _prepared={0: object()},
        _model_weight_role="ema",
    )
    case = SimpleNamespace(
        slot=0,
        noise_seed=42,
        metadata={
            "generation_pass": {
                "name": "model_self_forcing_nfe4",
                "weights": "model",
                "mode": "autoregressive",
                "solver": "self_forcing",
                "num_inference_steps": 4,
                "rollout_latent_frames": 240,
            }
        },
    )

    generated = _stage2_generated_sample(provider, case, weights_id="release#ema")

    assert calls == [240, 60]
    assert generated.shape == (1, 957, 3, 1, 1)
    assert generated.provenance["resolved_weights_role"] == "ema"
    assert generated.provenance["vae_decode"] == {
        "mode": "continuous_cached_tiles",
        "chunk_latent_frames": 60,
    }


@pytest.mark.parametrize(
    ("published_frames", "expected_trim", "expected_pad"),
    ((953, 4, 0), (960, 0, 3)),
)
def test_stage2_camera_publication_adjusts_to_source_and_serializes_original_c2w(
    monkeypatch: pytest.MonkeyPatch,
    published_frames: int,
    expected_trim: int,
    expected_pad: int,
) -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime import inference, stage2

    decoded = torch.arange(957, dtype=torch.float32).reshape(1, 957, 1, 1, 1)
    decoded = decoded.expand(-1, -1, 3, -1, -1).contiguous()
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], published_frames, axis=0)
    c2w[:, 0, 3] = np.arange(published_frames, dtype=np.float64) + 10.0
    encoded: list[object] = []
    compared: list[object] = []
    monkeypatch.setattr(
        stage2,
        "_stage2_self_forcing_latents",
        lambda *_: (
            torch.zeros((1, 240, 1, 1, 1)),
            {"timesteps": [1000, 750, 500, 250]},
        ),
    )

    def capture_compare(generated: object, prepared: object, **_kwargs: object) -> bytes:
        compared.extend((generated, prepared))
        return b"compare"

    def capture_video(value: object, **_kwargs: object) -> bytes:
        encoded.append(value)
        return b"video"

    monkeypatch.setattr(inference, "_encode_compare_mp4", capture_compare)
    prepared = SimpleNamespace(
        pixels=torch.zeros((published_frames, 3, 1, 1)),
        publication_pixel_frames=published_frames,
        publication_c2w=c2w,
    )
    provider = SimpleNamespace(
        device=torch.device("cpu"),
        config={
            "data": {"fps": 16.0},
            "model": {"camera_translation_transform": "linear"},
            "train": {"denoising_step_list": [1000, 750, 500, 250]},
        },
        _conditions=lambda *_args, **_kwargs: (
            torch.zeros((1, 1, 1, 1, 1)),
            {},
            {},
            None,
        ),
        vae=SimpleNamespace(decode_streaming=lambda *_args, **_kwargs: decoded),
        video_encoder=capture_video,
        _prepared={0: prepared},
        _model_weight_role="ema",
    )
    case = SimpleNamespace(
        slot=0,
        noise_seed=42,
        metadata={
            "generation_pass": {
                "name": "model_self_forcing_nfe4",
                "weights": "model",
                "mode": "autoregressive",
                "solver": "self_forcing",
                "num_inference_steps": 4,
                "rollout_latent_frames": 240,
            }
        },
    )

    generated = _stage2_generated_sample(provider, case, weights_id="release#ema")

    published = encoded[0]
    assert tuple(published.shape) == (1, published_frames, 3, 1, 1)
    if expected_pad:
        assert torch.equal(published[:, 956], published[:, 957])
        assert torch.equal(published[:, 956], published[:, 959])
    else:
        torch.testing.assert_close(published, decoded[:, :published_frames])
    assert compared[0] is published
    assert compared[1] is prepared
    assert generated.shape == (1, published_frames, 3, 1, 1)
    assert generated.provenance["model_output_pixel_frames"] == 957
    assert generated.provenance["published_pixel_frames"] == published_frames
    assert generated.provenance["trim_frames"] == expected_trim
    assert generated.provenance["tail_pad_frames"] == expected_pad
    assert generated.provenance["tail_padding"] == (
        "repeat_last_generated_frame" if expected_pad else "none"
    )
    serialized = np.load(io.BytesIO(generated.artifacts["camera.npy"]), allow_pickle=False)
    assert serialized.dtype == np.float64
    np.testing.assert_array_equal(serialized, c2w)


def test_stage2_hour_publication_selects_bounded_file_streaming(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime import inference, stage2

    latent_frames = 1026
    model_frames = 1 + 4 * (latent_frames - 1)
    published_frames = model_frames + 3
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        stage2,
        "_stage2_self_forcing_latents",
        lambda *_: (
            torch.zeros((1, latent_frames, 1, 1, 1)),
            {"timesteps": [1000, 750, 500, 250]},
        ),
    )

    def fake_stream(*args: object, **kwargs: object) -> object:
        calls.append({"args": args, **kwargs})
        return SimpleNamespace(
            video=b"streamed-video",
            compare=b"streamed-compare",
            shape=(1, published_frames, 3, 1, 1),
        )

    monkeypatch.setattr(inference, "_encode_stage2_streaming", fake_stream)
    prepared = SimpleNamespace(
        pixels=torch.zeros((1, 3, 1, 1)),
        publication_pixel_frames=published_frames,
        publication_c2w=np.repeat(np.eye(4, dtype=np.float64)[None], published_frames, axis=0),
        repeat_first_frame=True,
    )
    provider = SimpleNamespace(
        device=torch.device("cpu"),
        config={
            "data": {"fps": 16.0},
            "model": {"camera_translation_transform": "linear"},
            "train": {"denoising_step_list": [1000, 750, 500, 250]},
            "runtime": {"output_dir": str(tmp_path / "outputs" / "run")},
        },
        _conditions=lambda *_args, **_kwargs: (
            torch.zeros((1, 1, 1, 1, 1)),
            {},
            {},
            None,
        ),
        vae=SimpleNamespace(
            decode=lambda *_args, **_kwargs: pytest.fail("direct VAE decode was used"),
            decode_streaming=lambda *_args, **_kwargs: pytest.fail(
                "concatenating VAE decode was used"
            ),
        ),
        video_encoder=None,
        _prepared={0: prepared},
        _model_weight_role="ema",
    )
    case = SimpleNamespace(
        slot=0,
        noise_seed=42,
        metadata={
            "generation_pass": {
                "name": "model_self_forcing_nfe4",
                "weights": "model",
                "mode": "autoregressive",
                "solver": "self_forcing",
                "num_inference_steps": 4,
                "rollout_latent_frames": latent_frames,
            }
        },
    )

    generated = _stage2_generated_sample(provider, case, weights_id="release#ema")

    assert len(calls) == 1
    assert calls[0]["target_pixel_frames"] == published_frames
    assert calls[0]["chunk_latent_frames"] == 12
    assert generated.artifacts["video.mp4"] == b"streamed-video"
    assert generated.artifacts["compare.mp4"] == b"streamed-compare"
    assert generated.shape == (1, published_frames, 3, 1, 1)
    assert generated.provenance["vae_decode"] == {
        "mode": "continuous_cached_file_tiles",
        "chunk_latent_frames": 12,
    }


def test_stage2_camera_length_resolves_release_default_weights(tmp_path: Path) -> None:
    manifest = {
        "schema": "solarwm.public-weight-manifest.v1",
        "identity": {"model": {"weight_role": "live+ema"}},
        "load": {
            "default_weights": "ema",
            "entrypoint": ".",
            "format": "solarwm_wan_stage2_transaction_v1",
        },
    }
    (tmp_path / "release-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert _published_default_weight_role(tmp_path) == "ema"

    manifest["load"]["default_weights"] = "unknown"
    (tmp_path / "release-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BackendContractError, match="default weight role is invalid"):
        _published_default_weight_role(tmp_path)


def test_stage2_unconditional_matches_conditional_dtype() -> None:
    torch = pytest.importorskip("torch")

    class _TextEncoder:
        def __call__(self, captions: list[str]) -> dict[str, object]:
            return {
                "prompt_embeds": torch.ones(
                    (len(captions), 2, 3),
                    dtype=torch.float32,
                )
            }

    runtime = Wan5BStage2Runtime.__new__(Wan5BStage2Runtime)
    runtime._unconditional_cache = {}
    runtime.text_encoder = _TextEncoder()
    runtime.config = {"train": {"negative_prompt": ""}}
    runtime.device = torch.device("cpu")

    unconditional = runtime._unconditional(2, dtype=torch.bfloat16)

    assert unconditional["prompt_embeds"].dtype == torch.bfloat16
    assert unconditional["prompt_embeds"].shape == (2, 2, 3)


def test_stage2_role_checkpoint_verification_checks_real_file_size(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"role-checkpoint")
    initialization = RoleInitialization(
        role="student",
        path=str(checkpoint),
        weights="ema",
        expected_stage="stage1",
        expected_objective="anyflow_forward_map:v1_5",
        camera_translation_transform="linear",
        allow_anyflow_delta_drop=True,
    )

    receipt = verify_role_checkpoint(initialization)

    assert receipt.path == checkpoint
    assert receipt.object_bytes == checkpoint.stat().st_size
    assert receipt.step is None


def test_stage2_role_strictly_loads_member_step_and_semantics(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "model.pt"
    source = torch.nn.Linear(2, 2)
    initialization = RoleInitialization(
        role="teacher",
        path=str(checkpoint),
        weights="live",
        expected_stage="stage0p5",
        expected_objective="flow_matching",
        camera_translation_transform="linear",
    )

    def payload(*, objective: str = "flow_matching", include_live: bool = True) -> dict:
        value = {
            "global_step": 30000,
            "config": {
                "model": {
                    "family": "wan22_ti2v_5b",
                    "causal": False,
                    "camera_translation_transform": "linear",
                },
                "train": {"stage": "stage0p5", "objective": objective},
            },
        }
        if include_live:
            value["generator"] = source.state_dict()
        return value

    torch.save(payload(), checkpoint)
    receipt = verify_role_checkpoint(initialization)
    target = torch.nn.Linear(2, 2)
    ignored, verified = load_role_checkpoint(
        initialization=initialization,
        receipt=receipt,
        diffusion=SimpleNamespace(module=target),
    )
    assert ignored == ()
    assert verified.step == 30000
    assert verified.stage == "stage0p5"
    assert verified.objective == "flow_matching"
    assert verified.camera_translation_transform == "linear"
    assert all(
        torch.equal(target.state_dict()[key], value) for key, value in source.state_dict().items()
    )

    torch.save(payload(include_live=False), checkpoint)
    with pytest.raises(BackendContractError, match="checkpoint has no generator"):
        load_role_checkpoint(
            initialization=initialization,
            receipt=receipt,
            diffusion=SimpleNamespace(module=target),
        )

    torch.save(payload(objective="wrong"), checkpoint)
    with pytest.raises(BackendContractError, match="checkpoint metadata differs"):
        load_role_checkpoint(
            initialization=initialization,
            receipt=receipt,
            diffusion=SimpleNamespace(module=target),
        )


def _receipt(role: str) -> RoleCheckpointReceipt:
    return RoleCheckpointReceipt(
        role=role,
        path=Path(f"/{role}.pt"),
        object_bytes=1,
        step=3000 if role == "student" else 30000,
        weights="ema" if role == "student" else "live",
        stage="stage1" if role == "student" else "stage0p5",
        objective=("anyflow_forward_map:v1_5" if role == "student" else "flow_matching"),
        camera_translation_transform="linear",
    )


def test_stage2_initialization_receipt_is_readable_and_path_free() -> None:
    receipt = _stage2_initialization_receipt(
        {role: _receipt(role) for role in ("student", "teacher", "critic")}
    )
    assert receipt["initialization_id"].startswith("roles:")
    assert "digest:" not in receipt["initialization_id"]
    assert all("path" not in values for values in receipt["roles"].values())
    assert all("object_bytes" not in values for values in receipt["roles"].values())


def test_stage2_cuda_adapter_admits_multiple_logical_dp_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime import stage2

    class _Movable:
        def eval(self) -> _Movable:
            return self

        def requires_grad_(self, value: bool) -> _Movable:
            assert value is False
            return self

        def to(self, *args: object, **kwargs: object) -> _Movable:
            del args, kwargs
            return self

    topology = SimpleNamespace(
        raw_rank=3,
        raw_world_size=4,
        local_rank=0,
        dp_rank=3,
        dp_world_size=4,
        sp_rank=0,
        sp_size=1,
    )
    movable = _Movable()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(stage2, "initialize_torchrun", lambda _: topology)
    monkeypatch.setattr(
        stage2,
        "_verified_stage2_inference_checkpoint",
        lambda *_: (tmp_path / "model.pt", "b" * 64, 200),
    )
    monkeypatch.setattr(stage2, "_published_default_weight_role", lambda *_: "ema")
    monkeypatch.setattr(
        stage2.WanAssetLayout,
        "from_config",
        lambda _: SimpleNamespace(text_encoder="t5", tokenizer="tok", vae="vae"),
    )
    monkeypatch.setattr(
        stage2,
        "build_diffusion_architecture",
        lambda _: SimpleNamespace(module=movable),
    )
    monkeypatch.setattr(stage2, "WanTextEncoder", lambda *_, **__: movable)
    monkeypatch.setattr(stage2, "Wan5BVAE", lambda *_, **__: movable)
    config = {
        "model": {"family": "wan22_ti2v_5b"},
        "distributed": {"sequence_parallel_size": 1},
        "checkpoint": {"path": str(tmp_path)},
        "inference": {"length": "fixed"},
    }
    adapter = stage2.CudaWanStage2GenerationAdapter(config, SimpleNamespace())
    assert adapter.topology.dp_world_size == 4
    assert adapter.topology.dp_rank == 3

    generated = object()
    loaded: list[str] = []
    adapter._direct_model = True
    adapter._model_weight_role = "ema"
    adapter._load_role = loaded.append
    monkeypatch.setattr(stage2, "_stage2_generated_sample", lambda *_args, **_kwargs: generated)
    case = SimpleNamespace(slot=0, metadata={"generation_pass": {"weights": "model"}})

    assert adapter.generate(case, weights_id=adapter.weight_id("model")) is generated
    assert loaded == ["model"]

    config["inference"]["length"] = "camera"
    adapter._deferred_camera_inputs = object()
    adapter._materialize_deferred_camera_case = lambda value: adapter._prepared.__setitem__(
        value.slot, object()
    )
    assert adapter.generate(case, weights_id=adapter.weight_id("model")) is generated
    assert adapter._prepared == {}

    def fail_generation(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("generation failed")

    monkeypatch.setattr(stage2, "_stage2_generated_sample", fail_generation)
    with pytest.raises(RuntimeError, match="generation failed"):
        adapter.generate(case, weights_id=adapter.weight_id("model"))
    assert adapter._prepared == {}


def _checkpoint_config(tmp_path: Path) -> dict:
    return {
        "model": {
            "family": "wan22_ti2v_5b",
            "camera_translation_transform": "linear",
            "local_attn_size": 18,
            "score_local_attn_size": 21,
            "max_prior_clean_chunks": 5,
            "sink_size": 0,
            "rope_train_frames": 21,
            "use_echorope": False,
            "score_use_echorope": True,
        },
        "data": {
            "encoding": "online",
            "pixel_frames": 81,
            "height": 480,
            "width": 864,
            "train_index": "index.jsonl.gz",
        },
        "distributed": {"sequence_parallel_size": 1, "world_size": 1},
        "train": {
            "max_steps": 4000,
            "global_batch_size": 1,
            "denoising_step_list": [1000, 750, 500, 250],
            "critic_updates_per_student": 5,
            "score_min_timestep": 20.0,
            "score_max_timestep": 980.0,
            "context_timestep": 0.0,
            "per_rank_exit_step": True,
            "self_gradient_forcing_match_context": True,
            "self_gradient_forcing_cache_mode": "exit",
            "last_step_only": False,
            "real_guidance_scale": 3.0,
            "fake_guidance_scale": 0.0,
            "negative_prompt": "negative",
            "optimizer": {
                "lr": 2.0e-6,
                "betas": [0.0, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 0.0,
                "warmup_steps": 0,
                "min_lr_ratio": 1.0,
            },
            "critic_optimizer": {
                "lr": 4.0e-7,
                "betas": [0.0, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 0.0,
                "warmup_steps": 0,
                "min_lr_ratio": 1.0,
            },
            "ema": {"decay": 0.99, "start_step": 39, "update_every": 1},
        },
        "runtime": {"output_dir": str(tmp_path / "run")},
    }


class _CheckpointRuntime:
    def __init__(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")
        self.config = _checkpoint_config(tmp_path)
        self.student = SimpleNamespace(module=torch.nn.Linear(2, 2))
        self.critic = SimpleNamespace(module=torch.nn.Linear(2, 2))
        self.student_optimizer = torch.optim.AdamW(self.student.module.parameters())
        self.critic_optimizer = torch.optim.AdamW(self.critic.module.parameters())
        self.student_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.student_optimizer, lambda _: 1.0
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.critic_optimizer, lambda _: 1.0
        )
        self.role_receipts = {role: _receipt(role) for role in ("student", "teacher", "critic")}
        self.topology = SimpleNamespace(raw_rank=0)
        self.device = torch.device("cpu")
        self.critic_updates_per_student = 5
        self.ema_start_step = 39
        self.ema_update_every = 1
        self.ema = None
        self._global_step = 6
        self.student_step = 1
        self.data = _CheckpointReader()

    @property
    def global_step(self) -> int:
        return self._global_step

    def ensure_ema(self) -> object:
        raise AssertionError("EMA must not be requested before student step 39")


class _CheckpointReader:
    def __init__(self) -> None:
        self.position = 7

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = int(state["position"])


def test_stage2_checkpoint_rejects_scheduler_progress_drift(tmp_path: Path) -> None:
    runtime = _CheckpointRuntime(tmp_path)
    with pytest.raises(BackendContractError, match="scheduler progress"):
        save_stage2_checkpoint(runtime, 6)
    assert not (tmp_path / "run/checkpoint_model_000006").exists()


def test_stage2_checkpoint_state_quiesces_reader_before_rng_capture() -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime.stage2 import _local_rng_state

    class Reader:
        quiesced = False

        def checkpoint_state_dict(self) -> dict[str, int]:
            self.quiesced = True
            return {"position": 7}

        @staticmethod
        def state_dict() -> dict[str, int]:
            raise AssertionError("ordinary reader state must not bypass checkpoint quiescing")

    reader = Reader()
    state = _local_rng_state(0, torch.device("cpu"), reader)

    assert reader.quiesced is True
    assert state["reader"] == {"position": 7}


def test_stage2_resume_rejects_runtime_state_without_reader_schema() -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime.stage2 import _local_rng_state, _restore_rng_state

    reader = _CheckpointReader()
    state = _local_rng_state(0, torch.device("cpu"), reader)
    del state["schema"]

    with pytest.raises(BackendContractError, match="exact rank-local runtime state"):
        _restore_rng_state(
            [state],
            rank=0,
            device=torch.device("cpu"),
            reader=reader,
        )


def test_stage2_inference_verifies_the_complete_pair_before_allocation(
    tmp_path: Path,
) -> None:
    from solarwm.backends.wan22.runtime.stage2 import stage2_checkpoint_contract

    config = _checkpoint_config(tmp_path)
    target = tmp_path / "checkpoint-000196"
    with CheckpointTransaction(target) as transaction:
        (transaction.path / "model.pt").write_bytes(b"model")
        training = stage2_checkpoint_contract(
            config,
            {role: _receipt(role) for role in ("student", "teacher", "critic")},
        )
        contract = replace(
            training,
            parameterization="full-parameter-live-ema",
            sp_size=1,
            data_generation="inference-portable:v1",
            extras={
                "denoising_step_list": list(config["train"]["denoising_step_list"]),
                "attention": {
                    name: training.extras["attention"][name]
                    for name in (
                        "local_attn_size",
                        "max_prior_clean_chunks",
                        "sink_size",
                        "rope_train_frames",
                        "use_echorope",
                    )
                },
            },
        )
        committed = transaction.commit(
            step=196,
            contract=contract,
            required_components=("model.pt",),
            metadata={
                "schema": "solarwm.wan22-stage2-inference.v1",
                "available_weights": ["live", "ema"],
                "ema_present": True,
            },
        )

    model_path, identity, step = _verified_stage2_inference_checkpoint(
        config,
        target,
    )
    assert model_path == target / "model.pt"
    assert identity == committed.manifest_digest
    assert step == 196

    config["train"]["negative_prompt"] = "a user-selected inference prompt"
    config["train"]["critic_updates_per_student"] = 7
    config["train"]["score_min_timestep"] = 0.1
    config["train"]["fake_guidance_scale"] = 9.0
    config["train"]["ema"]["decay"] = 0.5
    _verified_stage2_inference_checkpoint(config, target / "model.pt")

    config["train"]["denoising_step_list"] = [1000, 500, 250, 125]
    with pytest.raises(BackendContractError, match="contract differs"):
        _verified_stage2_inference_checkpoint(config, target / "model.pt")


def test_stage2_checkpoint_is_atomic_pair_and_exactly_resumable(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    runtime = _CheckpointRuntime(tmp_path)
    for module, optimizer, scheduler, updates in (
        (
            runtime.student.module,
            runtime.student_optimizer,
            runtime.student_scheduler,
            1,
        ),
        (
            runtime.critic.module,
            runtime.critic_optimizer,
            runtime.critic_scheduler,
            6,
        ),
    ):
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            module(torch.ones((1, 2))).square().sum().backward()
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    expected_student = {
        key: value.detach().clone() for key, value in runtime.student.module.state_dict().items()
    }
    expected_critic = {
        key: value.detach().clone() for key, value in runtime.critic.module.state_dict().items()
    }
    torch.manual_seed(1234)
    identity = save_stage2_checkpoint(runtime, 6)
    checkpoint = tmp_path / "run/checkpoint_model_000006"
    assert len(identity) == 64
    assert (checkpoint / "critic.pt").is_file()
    assert (checkpoint / "model.pt").is_file()
    assert json.loads((checkpoint / "checkpoint-manifest.json").read_text())["metadata"] == {
        "commit_marker": "model.pt",
        "ema_present": False,
        "publication_order": ["critic.pt", "model.pt"],
        "schema": "solarwm.wan22-stage2-pair.v1",
        "student_step": 1,
    }
    with pytest.raises(BackendContractError, match=r"must contain exactly model\.pt"):
        _verified_stage2_inference_checkpoint(
            runtime.config,
            checkpoint / "model.pt",
        )
    expected_random = torch.rand(4)
    with torch.no_grad():
        for parameter in runtime.student.module.parameters():
            parameter.fill_(99.0)
        for parameter in runtime.critic.module.parameters():
            parameter.fill_(-99.0)
    torch.manual_seed(999)
    runtime.data.position = 99
    restored = load_stage2_checkpoint(runtime, checkpoint)
    assert restored.step == 6
    assert restored.student_step == 1
    assert runtime.global_step == 6
    assert runtime.data.position == 7
    assert torch.equal(torch.rand(4), expected_random)
    for key, value in runtime.student.module.state_dict().items():
        assert torch.equal(value, expected_student[key])
    for key, value in runtime.critic.module.state_dict().items():
        assert torch.equal(value, expected_critic[key])

    runtime.config["train"]["real_guidance_scale"] = 4.0
    with pytest.raises(BackendContractError, match="resume verification failed"):
        load_stage2_checkpoint(runtime, checkpoint)
    runtime.config["train"]["real_guidance_scale"] = 3.0
    (checkpoint / "critic.pt").unlink()
    with pytest.raises(BackendContractError, match="resume verification failed"):
        load_stage2_checkpoint(runtime, checkpoint)


def test_stage2_unified_runner_receives_the_runtime_provider(tmp_path: Path) -> None:
    from solarwm.backends.wan22.runtime.stage2 import Wan5BStage2Runtime

    seen: dict[str, object] = {}

    def runner(config: object, **kwargs: object) -> dict:
        seen["config"] = config
        seen.update(kwargs)
        return {"shared": True}

    runtime = object.__new__(Wan5BStage2Runtime)
    runtime.config = {
        "runtime": {"output_dir": str(tmp_path)},
    }
    runtime.generation_runner = runner
    assert runtime.run_generation(output_dir=tmp_path / "generation") == {"shared": True}
    assert seen["provider"] is runtime
    assert seen["output_dir"] == tmp_path / "generation"


def test_stage2_validation_declares_prepartitioned_cases() -> None:
    assert Wan5BStage2Runtime.build_cases_returns_partition is True
