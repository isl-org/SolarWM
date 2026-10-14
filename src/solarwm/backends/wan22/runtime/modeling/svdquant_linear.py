"""SYCL SVDQuant INT4 Linear layer and model management for Intel XPU."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .model import WanRMSNorm


def convert_int4_weights_to_vnni8(packed_int4: torch.Tensor) -> torch.Tensor:
    """Reorder [N, K // 2] packed signed INT4 into VNNI-8 layout [N // 16, K // 8, 16] int32."""
    n, k_half = packed_int4.shape
    k = k_half * 2
    if n % 16 != 0 or k % 64 != 0:
        raise ValueError(f"VNNI-8 requires N % 16 == 0 and K % 64 == 0, got N={n}, K={k}")
    packed_u8 = packed_int4.contiguous().view(torch.uint8)
    dwords_per_row = k // 8
    row_dwords = packed_u8.view(n, dwords_per_row, 4).contiguous().view(torch.int32)
    return row_dwords.view(n // 16, 16, dwords_per_row).permute(0, 2, 1).contiguous()


def convert_int4_weights_to_dpas_b(packed_int4: torch.Tensor) -> torch.Tensor:
    """Prepack INT4 weights to DPAS B operand layout [N // 16, K // 64, 8, 16] int32."""
    n, k_half = packed_int4.shape
    return convert_int4_weights_to_vnni8(packed_int4).view(n // 16, (2 * k_half) // 64, 8, 16)


def convert_w_scales_to_dpas(w_scale: torch.Tensor) -> torch.Tensor:
    """Prepack weight scales [N, K // 64] -> [K // 64, N] float32."""
    return w_scale.t().contiguous().to(torch.float32)


def unpack_int4_signed(packed: torch.Tensor) -> torch.Tensor:
    """Unpacks [..., K/2] uint8 tensor into [..., K] signed int8 tensor in range [-8, 7]."""
    u8 = packed.to(torch.uint8)
    low = (u8 & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = ((u8 >> 4) & 0x0F).to(torch.int8)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack([low, high], dim=-1).flatten(-2)


_SYCL_KERNEL = None
_SYCL_KERNEL_TRIED = False


def get_sycl_svdquant_kernel() -> Any:
    """Dynamically loads the compiled SYCL SVDQuant kernel extension."""
    global _SYCL_KERNEL, _SYCL_KERNEL_TRIED
    if not _SYCL_KERNEL_TRIED:
        _SYCL_KERNEL_TRIED = True
        pkg_dir = None
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "packages" / "wan22_svdquant_dense_block"
            if candidate.exists():
                pkg_dir = candidate
                break
        if pkg_dir is not None and str(pkg_dir) not in sys.path:
            sys.path.insert(0, str(pkg_dir))
        try:
            _SYCL_KERNEL = importlib.import_module("svdquant_dense_kernel_grf128_n_tile64")
        except Exception as exc:
            import warnings
            warnings.warn(f"Failed to load compiled SYCL SVDQuant kernel: {exc}", stacklevel=2)
            _SYCL_KERNEL = None
    return _SYCL_KERNEL


def _as_2d_bf16(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1]).to(torch.bfloat16).contiguous()


def fake_quant_int4(x: torch.Tensor, smooth_inv: torch.Tensor, rho: float) -> torch.Tensor:
    """Reference group-64 INT4 fake quantization of x * smooth_inv with scale amax * rho / 7."""
    k = x.shape[-1]
    g = (x.float() * smooth_inv).view(-1, k // 64, 64)
    scale = (g.abs().amax(-1, keepdim=True) * (rho / 7.0)).clamp_min(1e-8)
    q = g / scale
    q = torch.where(q >= 0, torch.floor(q + 0.5), torch.ceil(q - 0.5)).clamp(-8, 7)
    return (q * scale).view(-1, k)


class SYCLSVDQuantLinear(nn.Module):
    """SVDQuant W4A4 linear: y = Q(x / s) @ (W s)^T + (x @ lora_a^T) @ lora_b^T + bias.

    Q is group-64 symmetric INT4 with scale amax * rho / 7, where rho is the per-bin activation
    clipping ratio from the checkpoint. With `norm` set, the output is RMS-normalized (fused into
    the GEMM epilogue on XPU).
    """

    TARGET_SUFFIXES = (
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "ffn.0",
        "ffn.2",
    )

    def __init__(
        self,
        sets_data: dict[str, dict[str, torch.Tensor]],
        set_mapping: dict[str, str],
        act_policy: dict[str, float] | None = None,
        device: torch.device | str = "xpu:0",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.set_mapping = set_mapping
        self.act_policy = act_policy or {}
        self.current_bin = "0"
        self.kernel_module = get_sycl_svdquant_kernel() if self.device.type == "xpu" else None
        self.norm: nn.Module | None = None
        self.shared_input: SharedInputQuant | None = None
        self.shared_index = 0

        self.sets = nn.ModuleDict()
        for set_name, tensors in sets_data.items():
            set_mod = nn.Module()
            qw = tensors["qweight"]
            ws = tensors["w_scale"].to(torch.bfloat16)
            if self.kernel_module is not None:
                set_mod.register_buffer("w_dpas_b", convert_int4_weights_to_dpas_b(qw).to(self.device))
                set_mod.register_buffer("w_scales_dpas", convert_w_scales_to_dpas(ws).to(self.device))
            else:
                w_unpacked = unpack_int4_signed(qw).to(torch.bfloat16)
                w_t = (w_unpacked * ws.repeat_interleave(64, dim=-1)).T.contiguous()
                set_mod.register_buffer("w_deq_t", w_t.to(self.device))
                del w_unpacked, w_t
            smooth = tensors["smooth_scale"].to(torch.float32)
            set_mod.register_buffer("smooth_inv", (1.0 / smooth).to(self.device).contiguous())
            set_mod.register_buffer("lora_a", tensors["lora_a"].to(self.device, torch.bfloat16).contiguous())
            set_mod.register_buffer("lora_b", tensors["lora_b"].to(self.device, torch.bfloat16).contiguous())
            bias = tensors.get("bias")
            if bias is not None:
                set_mod.register_buffer("bias", bias.to(self.device, torch.bfloat16).contiguous())
            else:
                set_mod.bias = None
            self.sets[set_name] = set_mod

    def set_bin(self, bin_name: str) -> None:
        self.current_bin = str(bin_name)

    @property
    def rho(self) -> float:
        return float(self.act_policy.get(self.current_bin, 1.0))

    def active_set(self) -> nn.Module:
        name = self.set_mapping.get(self.current_bin, "set0")
        if name not in self.sets:
            name = next(iter(self.sets.keys()))
        return self.sets[name]

    def use_kernel(self, x: torch.Tensor) -> bool:
        return self.kernel_module is not None and x.is_xpu

    def _norm_args(self) -> dict[str, Any]:
        if self.norm is None:
            return {}
        return {"norm_weight": self.norm.weight.to(torch.bfloat16).contiguous(), "norm_eps": float(self.norm.eps)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        s = self.active_set()
        x_2d = _as_2d_bf16(x)

        if self.use_kernel(x):
            if self.shared_input is not None:
                x_q, x_scales, x_lora = self.shared_input.get(x, x_2d, self)
                out = self.kernel_module.svdquant_w4a4_gemm(
                    x_q, s.w_dpas_b, s.w_scales_dpas, x_scales=x_scales,
                    lora_b=s.lora_b, bias=s.bias, x_lora=x_lora, **self._norm_args(),
                )
            else:
                out = self.kernel_module.svdquant_w4a4_gemm(
                    x_2d, s.w_dpas_b, s.w_scales_dpas, lora_a=s.lora_a, lora_b=s.lora_b, bias=s.bias,
                    x_smooth_inv=s.smooth_inv, x_rho=self.rho, **self._norm_args(),
                )
            return out.view(*orig_shape[:-1], -1)

        y = self.reference(x_2d)
        return y.view(*orig_shape[:-1], -1)

    def reference(self, x_2d: torch.Tensor) -> torch.Tensor:
        """PyTorch path with the kernel's quantization semantics (for CPU and verification)."""
        s = self.active_set()
        x_deq = fake_quant_int4(x_2d, s.smooth_inv, self.rho)
        if hasattr(s, "w_deq_t"):
            w_t = s.w_deq_t.float()
        else:
            w_t = dequant_dpas_weights(s.w_dpas_b, s.w_scales_dpas).t()
        y = x_deq @ w_t
        y = y + (x_2d.float() @ s.lora_a.float().t()) @ s.lora_b.float().t()
        if s.bias is not None:
            y = y + s.bias.float()
        y = y.to(torch.bfloat16)
        if self.norm is not None:
            y = self.norm(y)
        return y


def dequant_dpas_weights(w_dpas_b: torch.Tensor, w_scales_dpas: torch.Tensor) -> torch.Tensor:
    """Inverse of the DPAS prepack: returns the dequantized [N, K] float32 weight."""
    n16, kg = w_dpas_b.shape[:2]
    n, k = n16 * 16, kg * 64
    vnni = w_dpas_b.reshape(n16, k // 8, 16).permute(0, 2, 1).reshape(n, k // 8).contiguous()
    packed = vnni.view(torch.uint8).view(n, k // 2)
    w = unpack_int4_signed(packed).float()
    return w * w_scales_dpas.t().repeat_interleave(64, dim=-1)


class SharedInputQuant:
    """One prologue pass for linears that share an input and a smoothing vector (Q/K/V).

    Quantizes x once per distinct rho among the members and computes all members' LoRA-down
    projections with the concatenated lora_a in the same pass.
    """

    def __init__(self, members: list[SYCLSVDQuantLinear]) -> None:
        self.members = members
        self.rank = members[0].active_set().lora_a.shape[0]
        self.lora_a_cat = {
            name: torch.cat([m.sets[name].lora_a for m in members], dim=0).contiguous()
            for name in members[0].sets.keys()
        }
        self._key: tuple[Any, ...] | None = None
        self._x: torch.Tensor | None = None
        self._lora: torch.Tensor | None = None
        self._quant: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
        self._served: set[int] = set()

    def get(
        self, x: torch.Tensor, x_2d: torch.Tensor, member: SYCLSVDQuantLinear
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kern = member.kernel_module
        set_name = member.set_mapping.get(member.current_bin, "set0")
        if set_name not in member.sets:
            set_name = next(iter(member.sets.keys()))
        key = (x._version, member.current_bin, set_name)
        smooth_inv = member.sets[set_name].smooth_inv
        if self._x is not x or self._key != key or member.shared_index in self._served:
            self._x, self._key, self._quant, self._served = x, key, {}, set()
            x_q, x_scales, self._lora = kern.quantize_activation_int4(
                x_2d, self.lora_a_cat[set_name], smooth_inv=smooth_inv, rho=member.rho
            )
            self._quant[member.rho] = (x_q, x_scales)
        if member.rho not in self._quant:
            self._quant[member.rho] = kern.quantize_activation_int4(x_2d, smooth_inv=smooth_inv, rho=member.rho)
        x_q, x_scales = self._quant[member.rho]
        i, r = member.shared_index, self.rank
        lora = self._lora[:, i * r : (i + 1) * r]
        self._served.add(i)
        if len(self._served) == len(self.members):
            self._x, self._key, self._lora, self._quant, self._served = None, None, None, {}, set()
        return x_q, x_scales, lora


class SVDQuantFFN(nn.Module):
    """FFN (Linear -> GELU(tanh) -> Linear) on SVDQuant W4A4.

    On XPU the up projection fuses GELU and emits the INT4 (smoothed with the down projection's
    smoothing vector and rho) that the down projection consumes directly.
    """

    def __init__(self, up: SYCLSVDQuantLinear, down: SYCLSVDQuantLinear) -> None:
        super().__init__()
        self.up = up
        self.down = down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = _as_2d_bf16(x)
        up, down = self.up, self.down
        if up.use_kernel(x):
            su, sd = up.active_set(), down.active_set()
            kern = up.kernel_module
            h, h_q, h_scales = kern.svdquant_w4a4_gemm(
                x_2d, su.w_dpas_b, su.w_scales_dpas, lora_a=su.lora_a, lora_b=su.lora_b, bias=su.bias,
                activation="gelu_tanh", fuse_quant_out=True,
                x_smooth_inv=su.smooth_inv, x_rho=up.rho, out_smooth_inv=sd.smooth_inv, out_rho=down.rho,
            )
            out = kern.svdquant_w4a4_gemm(
                h_q, sd.w_dpas_b, sd.w_scales_dpas, x_scales=h_scales, x_bf16=h,
                lora_a=sd.lora_a, lora_b=sd.lora_b, bias=sd.bias,
            )
            return out.view(*orig_shape[:-1], -1)
        h = torch.nn.functional.gelu(up.reference(x_2d).float(), approximate="tanh").to(torch.bfloat16)
        return down.reference(h).view(*orig_shape[:-1], -1)


# Compatibility aliases
SVDQuantLinear = SYCLSVDQuantLinear
CUDASVDQuantLinear = SYCLSVDQuantLinear


class SVDQuantModelManager:
    """Manages replacing DiT linear layers with SVDQuant INT4 layers and tracking active timestep bins."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.original_linears: dict[str, nn.Linear | nn.Module] = {}
        self.original_modules: dict[str, nn.Module] = {}
        self.quant_linears: dict[str, SYCLSVDQuantLinear] = {}
        self._record_original_linears()

    def _record_original_linears(self) -> None:
        for b_idx, block in enumerate(self.model.blocks):
            targets = [
                ("self_attn.q", block.self_attn.q),
                ("self_attn.k", block.self_attn.k),
                ("self_attn.v", block.self_attn.v),
                ("ffn.0", block.ffn[0]),
                ("ffn.2", block.ffn[2]),
            ]
            for suffix, layer in targets:
                key = f"blocks.{b_idx}.{suffix}"
                self.original_linears[key] = layer
            self.original_modules[f"blocks.{b_idx}.ffn"] = block.ffn
            self.original_modules[f"blocks.{b_idx}.self_attn.norm_q"] = block.self_attn.norm_q
            self.original_modules[f"blocks.{b_idx}.self_attn.norm_k"] = block.self_attn.norm_k

    def restore_bf16(self) -> None:
        """Restores original full-precision BF16 linears."""
        for b_idx, block in enumerate(self.model.blocks):
            prefix = f"blocks.{b_idx}"
            block.self_attn.q = self.original_linears[f"{prefix}.self_attn.q"]
            block.self_attn.k = self.original_linears[f"{prefix}.self_attn.k"]
            block.self_attn.v = self.original_linears[f"{prefix}.self_attn.v"]
            block.self_attn.norm_q = self.original_modules[f"{prefix}.self_attn.norm_q"]
            block.self_attn.norm_k = self.original_modules[f"{prefix}.self_attn.norm_k"]
            block.ffn = self.original_modules[f"{prefix}.ffn"]
            block.ffn[0] = self.original_linears[f"{prefix}.ffn.0"]
            block.ffn[2] = self.original_linears[f"{prefix}.ffn.2"]
        self.quant_linears.clear()

    def unload_bf16_linears_to_cpu(self) -> None:
        """Offloads original linears to CPU to free VRAM."""
        for layer in self.original_linears.values():
            if hasattr(layer, "to"):
                layer.to("cpu")
            if hasattr(layer, "weight") and layer.weight is not None:
                layer.weight.data = layer.weight.data.to("cpu")
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.data = layer.bias.data.to("cpu")

    def restore_bf16_linears_to_device(self, device: torch.device | str) -> None:
        """Restores original linears to target device."""
        for layer in self.original_linears.values():
            if hasattr(layer, "to"):
                layer.to(device)

    def set_current_bin(self, bin_name: str) -> None:
        """Sets the active timestep bin across all 150 quantized linears."""
        for lin in self.quant_linears.values():
            lin.set_bin(bin_name)

    def load_svdquant_checkpoint(
        self,
        checkpoint_dir: Path | str,
        device: torch.device | str = "xpu:0",
        fused: bool = True,
    ) -> None:
        """Loads SVDQuant weights from checkpoint dir and replaces all 150 linears.

        With `fused`, Q/K/V share one input quantization + LoRA-down pass, the Q/K RMSNorms run
        in the GEMM epilogue, and the FFN becomes an `SVDQuantFFN` (GELU and INT4 output fused
        into the up projection).
        """
        ckpt = Path(checkpoint_dir)
        manifest_path = ckpt / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {checkpoint_dir}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        set_mapping = manifest.get(
            "set_mapping",
            {"0": "set0", "1": "set0", "2": "set0", "3": "set0", "commit": "set0"},
        )
        activation_policy = manifest.get("activation_policy", {})

        shard_files = sorted(ckpt.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}")

        parsed: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        for sf in shard_files:
            sf_tensors = load_file(str(sf), device="cpu")
            for full_key, tensor in sf_tensors.items():
                parts = full_key.split(".", 1)
                set_name, mod_key = parts[0], parts[1]
                last_dot = mod_key.rfind(".")
                module_name = mod_key[:last_dot]
                param_name = mod_key[last_dot + 1 :]
                parsed.setdefault(module_name, {}).setdefault(set_name, {})[param_name] = tensor
            del sf_tensors

        self.restore_bf16()
        for b_idx, block in enumerate(self.model.blocks):
            block_key = f"blocks.{b_idx}"
            lin: dict[str, SYCLSVDQuantLinear] = {}
            for suffix in SYCLSVDQuantLinear.TARGET_SUFFIXES:
                full_mod_name = f"{block_key}.{suffix}"
                if full_mod_name not in parsed:
                    raise KeyError(f"Missing weights for {full_mod_name} in {checkpoint_dir}")

                mod_act_policy: dict[str, float] = {}
                for bin_name, block_policies in activation_policy.items():
                    if block_key in block_policies and suffix in block_policies[block_key]:
                        mod_act_policy[str(bin_name)] = float(block_policies[block_key][suffix])

                lin[suffix] = SYCLSVDQuantLinear(
                    sets_data=parsed.pop(full_mod_name),
                    set_mapping=set_mapping,
                    act_policy=mod_act_policy,
                    device=device,
                )
                self.quant_linears[full_mod_name] = lin[suffix]

            attn = block.self_attn
            attn.q, attn.k, attn.v = lin["self_attn.q"], lin["self_attn.k"], lin["self_attn.v"]
            if not fused:
                block.ffn[0], block.ffn[2] = lin["ffn.0"], lin["ffn.2"]
                continue

            qkv = [lin["self_attn.q"], lin["self_attn.k"], lin["self_attn.v"]]
            if all(
                torch.equal(m.sets[name].smooth_inv, qkv[0].sets[name].smooth_inv)
                for m in qkv[1:]
                for name in qkv[0].sets.keys()
            ):
                shared = SharedInputQuant(qkv)
                for i, m in enumerate(qkv):
                    m.shared_input, m.shared_index = shared, i
            for m, norm_attr in ((attn.q, "norm_q"), (attn.k, "norm_k")):
                norm = getattr(attn, norm_attr)
                if isinstance(norm, WanRMSNorm):
                    m.norm = norm
                    setattr(attn, norm_attr, nn.Identity())
            block.ffn = SVDQuantFFN(lin["ffn.0"], lin["ffn.2"])

        del parsed


def stub_dit_linear_layers(model: nn.Module) -> dict[str, nn.Module]:
    """Replaces the 150 linear layers on CPU with empty nn.Identity modules.

    This ensures that subsequent `.to(device)` calls do NOT allocate the 10 GB
    of BF16 linear weights in device VRAM.
    """
    stubs: dict[str, nn.Module] = {}
    for b_idx, block in enumerate(model.blocks):
        targets = [
            ("self_attn.q", block.self_attn, "q"),
            ("self_attn.k", block.self_attn, "k"),
            ("self_attn.v", block.self_attn, "v"),
            ("ffn.0", block.ffn, 0),
            ("ffn.2", block.ffn, 2),
        ]
        for suffix, parent, attr in targets:
            key = f"blocks.{b_idx}.{suffix}"
            stub = nn.Identity()
            if isinstance(attr, int):
                parent[attr] = stub
            else:
                setattr(parent, attr, stub)
            stubs[key] = stub
    return stubs


def load_svdquant_into_model(
    model: nn.Module,
    checkpoint_dir: Path | str,
    device: torch.device | str = "xpu:0",
) -> SVDQuantModelManager:
    """Replaces all 150 linear layers in model with SYCLSVDQuantLinear layers."""
    manager = SVDQuantModelManager(model)
    manager.load_svdquant_checkpoint(checkpoint_dir, device=device)
    return manager


class TimestepTrackingDiffusionWrapper(nn.Module):
    """Wraps provider.diffusion to set the active timestep bin in SVDQuant linears per forward pass."""

    def __init__(self, inner_diffusion: Any, manager: SVDQuantModelManager) -> None:
        super().__init__()
        self.inner = inner_diffusion
        self.manager = manager

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        radial_step = kwargs.get("radial_step", None)
        if radial_step is not None:
            if radial_step < 4:
                bin_name = str(radial_step)
            else:
                bin_name = "commit"
            self.manager.set_current_bin(bin_name)
        return self.inner(*args, **kwargs)
