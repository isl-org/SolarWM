"""Wan torchrun initialization and FSDP wrapping."""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from solarwm.errors import BackendContractError
from solarwm.runtime.topology import Topology


def apply_wan_activation_checkpointing(module: Any) -> int:
    """Wrap every Wan block with the configured non-reentrant policy."""

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    from .modeling.causal_model import CausalWanAttentionBlock

    blocks = tuple(
        candidate
        for candidate in module.modules()
        if isinstance(candidate, CausalWanAttentionBlock)
    )
    if not blocks:
        raise BackendContractError("Wan activation checkpointing found no transformer blocks")
    block_ids = {id(candidate) for candidate in blocks}
    non_reentrant = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        module,
        checkpoint_wrapper_fn=non_reentrant,
        check_fn=lambda candidate: id(candidate) in block_ids,
    )
    return len(blocks)


def synchronize_accelerator() -> None:
    """Synchronize the active PyTorch accelerator (CUDA, XPU, or torch.accelerator)."""

    import torch

    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        try:
            if accelerator.is_available():
                accelerator.synchronize()
                return
        except (AttributeError, RuntimeError):
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def empty_accelerator_cache() -> None:
    """Release cached blocks on every available device backend."""

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def resolve_device(config: Mapping[str, Any] | None = None, local_rank: int = 0) -> Any:
    """Resolve an explicit inference device or the current PyTorch accelerator."""

    import torch

    requested = None
    if isinstance(config, Mapping):
        for section_name in ("inference", "runtime"):
            section = config.get(section_name)
            if isinstance(section, Mapping) and section.get("device"):
                requested = str(section["device"])
                break
    if requested:
        return torch.device(requested)
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        try:
            if accelerator.is_available():
                current = accelerator.current_accelerator()
                if current is not None:
                    return torch.device(current)
        except (AttributeError, RuntimeError):
            pass
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return torch.device("xpu", local_rank)
    return torch.device("cpu")


def initialize_torchrun(sp_size: int) -> Topology:
    """Initialize the process layout without requiring CUDA for inference."""

    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise BackendContractError("Wan distributed runtime requires torch") from exc
    from .sequence_parallel import init_sequence_parallel

    if "WORLD_SIZE" in os.environ:
        topology = Topology.from_environ(sp_size)
    else:
        topology = Topology(1, 0, 1, 0, int(sp_size))
    if topology.raw_world_size > 1 and not dist.is_initialized():
        if not torch.cuda.is_available():
            raise BackendContractError("Wan distributed execution requires CUDA")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=1),
        )
    device = resolve_device(local_rank=topology.local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "xpu":
        accelerator = getattr(torch, "accelerator", None)
        if accelerator is not None and hasattr(accelerator, "set_device_index"):
            accelerator.set_device_index(device.index or topology.local_rank)
        else:
            torch.xpu.set_device(device)
    init_sequence_parallel(sp_size=int(sp_size))
    return topology


def wrap_transformer_fsdp(module: Any, config: Mapping[str, Any], topology: Topology) -> Any:
    """Wrap every Wan transformer block with the configured mixed-precision policy."""

    if topology.raw_world_size == 1:
        return module.to(topology.local_rank)
    import torch
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from .modeling.causal_model import CausalWanAttentionBlock
    from .sequence_parallel import get_fsdp_process_group

    train = config["train"]
    fsdp = train["fsdp"]
    strategy = {
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }[str(fsdp["strategy"])]
    policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={CausalWanAttentionBlock},
    )
    mixed = MixedPrecision(
        param_dtype=getattr(torch, str(fsdp["param_dtype"])),
        reduce_dtype=getattr(torch, str(fsdp["reduce_dtype"])),
        buffer_dtype=getattr(torch, str(fsdp.get("buffer_dtype", "bfloat16"))),
        cast_root_forward_inputs=bool(fsdp.get("cast_root_forward_inputs", True)),
    )
    wrapped = FullyShardedDataParallel(
        module,
        auto_wrap_policy=policy,
        process_group=get_fsdp_process_group(),
        sharding_strategy=strategy,
        mixed_precision=mixed,
        device_id=topology.local_rank,
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sync_module_states=False,
        limit_all_gathers=True,
    )
    if bool(fsdp.get("activation_checkpointing", False)):
        apply_wan_activation_checkpointing(wrapped)
    return wrapped


def cleanup_torchrun() -> None:
    import torch.distributed as dist

    synchronize_accelerator()
    empty_accelerator_cache()
    if dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "apply_wan_activation_checkpointing",
    "cleanup_torchrun",
    "empty_accelerator_cache",
    "initialize_torchrun",
    "resolve_device",
    "synchronize_accelerator",
    "wrap_transformer_fsdp",
]
