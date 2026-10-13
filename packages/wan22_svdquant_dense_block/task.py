import os
import sys
import json
import importlib
from pathlib import Path
import pytest
import torch

from kernelfoundry import TestBase

# Ensure package directory is on sys.path
_PKG_DIR = Path(__file__).parent.resolve()
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from qdq_utils import (
    convert_int4_weights_to_dpas_b,
    convert_w_scales_to_dpas,
    unpack_dpas_b_weights,
    pack_int4_signed,
    quantize_activation_int4,
    unpack_int4_signed,
)
from reference_kernels import bf16_linear, bf16_linear_activation
from triton_w4a4_gemm import triton_svdquant_w4a4_gemm

"""
[USER_INSTRUCTIONS_START]
Create sycl kernels using the sycl-tla library.
Workload: SolarWM Wan2.2 TI2V-5B Camera Transformer Block Projections & FFN
- Active sequence length: M = 1215 tokens (3 frames * 405 tokens/frame)
- BF16 LoRA rank: r = 32
- Group size: G = 64
1. Projection 0 (Q): M = 1215, K = 3072, N = 3072, LoRA rank 32, with optional RMSNorm (WanRMSNorm dim=3072, eps=1e-5)
2. Projection 1 (K): M = 1215, K = 3072, N = 3072, LoRA rank 32, with optional RMSNorm (WanRMSNorm dim=3072, eps=1e-5)
3. Projection 2 (V): M = 1215, K = 3072, N = 3072, LoRA rank 32, no normalization
4. FFN up-projection: M = 1215, K = 3072, N = 14336, LoRA rank 32, activation gelu_tanh, optional fused INT4 output quantization
5. FFN down-projection: M = 1215, K = 14336, N = 3072, LoRA rank 32, no activation

Optimization directives:
- Input handling: Quantization should be optional. If input x is already in INT4 (from a preceding layer with fused output quantization), do not quantize. If input x is BF16, dynamic INT4 group-64 quantization must be fused on the fly inside the SYCL kernel (similar to Nunchaku SVDQuant).
- Fused LoRA: LoRA (x @ lora_a^T @ lora_b^T) must be calculated on the fly inside the SYCL kernel without separate PyTorch passes.
- Dual GEMM: In the main loop, perform a dual GEMM (INT4 MMA, scale it and accumulate into the BF16 LoRA MMA). Alternately, use XMX / dpas for the INT4 GEMM and vector engines for the LoRA BF16 branch in the epilogue.
- Epilogue features:
  - Fused bias addition.
  - Optional RMSNorm normalization for query and key projections, following the Wan2.2-5B model.
  - Optional non-linearities (gelu_tanh, silu).
  - Optional fused output INT4 quantization (qout and oscales) in the epilogue to chain inter-layer execution without HBM round-trips.

Optimization Metric: Sum of runtimes across the 3 projections + 2 FFN layers (including gelu_tanh and RMSNorm).
Correctness: Verified against our Triton kernel (triton_svdquant_w4a4_gemm) in bfloat16 precision.
Runtime Reference: PyTorch native BF16 operations (F.linear / wan_rmsnorm).
[USER_INSTRUCTIONS_END]
"""

# [REFERENCE_START]
def reference_dense_block(
    x,
    q_w, q_b, q_la, q_lb, q_norm_w,
    k_w, k_b, k_la, k_lb, k_norm_w,
    v_w, v_b, v_la, v_lb,
    ffn_up_w, ffn_up_b, ffn_up_la, ffn_up_lb,
    ffn_down_w, ffn_down_b, ffn_down_la, ffn_down_lb,
):
    """PyTorch BF16 runtime baseline for the 3 projections (with Q/K RMSNorm) + 2 FFN layers."""
    q = bf16_linear(x, q_w, q_b, q_la, q_lb, norm_weight=q_norm_w)
    k = bf16_linear(x, k_w, k_b, k_la, k_lb, norm_weight=k_norm_w)
    v = bf16_linear(x, v_w, v_b, v_la, v_lb)
    ffn_up = bf16_linear_activation(x, ffn_up_w, ffn_up_b, ffn_up_la, ffn_up_lb, activation="gelu_tanh")
    ffn_down = bf16_linear(ffn_up, ffn_down_w, ffn_down_b, ffn_down_la, ffn_down_lb)
    return q, k, v, ffn_down
# [REFERENCE_END]


def execute_sycl_layer(
    kernel_module,
    x,
    w_dpas_b,
    w_scale,
    bias=None,
    lora_a=None,
    lora_b=None,
    norm_weight=None,
    norm_eps=1e-5,
    activation="none",
    x_scales=None,
    x_bf16=None,
    fuse_quant_out=False,
    x_lora=None,
):
    """Execute a single SVDQuant layer using the compiled SYCL kernel.

    - Quantization is optional: if x is already INT4 (torch.int8), it uses x directly with x_scales.
      If x is BF16, dynamic group-64 quantization is performed inside the fused SYCL kernel.
    - LoRA (x @ lora_a.T @ lora_b.T) is calculated inside the SYCL kernels: the down projection
      in the quantization prologue (or from a precomputed f32 ``x_lora``), the up projection in
      the GEMM epilogue.
    - Optional RMSNorm is applied in the SYCL kernel if norm_weight is provided.
    - Optional fused INT4 output quantization if fuse_quant_out is True.
    - ``w_dpas_b`` / ``w_scale`` are the model-load pre-packed DPAS B payload
      (``convert_int4_weights_to_dpas_b``) and f32 ``[K/64, N]`` scales
      (``convert_w_scales_to_dpas``), not the raw packed weights / scales.
    """
    return kernel_module.svdquant_w4a4_gemm(
        x,
        w_dpas_b,
        w_scale,
        x_scales,
        lora_a,
        lora_b,
        bias,
        norm_weight,
        norm_eps,
        activation,
        fuse_quant_out,
        x_bf16,
        x_lora,
    )


def make_sycl_dense_block_fn(kernel_module):
    """Build a callable for the full candidate SYCL dense block with fused FFN chaining."""
    # Model-load preparation: Q/K/V/FFN-up share x, so their lora_a are concatenated once and
    # one prologue pass over x yields the INT4 activations and all four LoRA down projections.
    lora_a_cat_cache = {}

    def sycl_dense_block(
        x,
        q_wq, q_ws, q_b, q_la, q_lb, q_norm_w,
        k_wq, k_ws, k_b, k_la, k_lb, k_norm_w,
        v_wq, v_ws, v_b, v_la, v_lb,
        ffn_up_wq, ffn_up_ws, ffn_up_b, ffn_up_la, ffn_up_lb,
        ffn_down_wq, ffn_down_ws, ffn_down_b, ffn_down_la, ffn_down_lb,
    ):
        key = tuple(t.data_ptr() for t in (q_la, k_la, v_la, ffn_up_la))
        if key not in lora_a_cat_cache:
            lora_a_cat_cache[key] = torch.cat((q_la, k_la, v_la, ffn_up_la)).contiguous()
        x_q, x_scales, x_lora = kernel_module.quantize_activation_int4(x, lora_a_cat_cache[key])
        r = q_la.shape[0]
        x_lora_q, x_lora_k, x_lora_v, x_lora_up = (x_lora[:, i * r:(i + 1) * r] for i in range(4))

        q = execute_sycl_layer(kernel_module, x_q, q_wq, q_ws, bias=q_b, lora_b=q_lb, norm_weight=q_norm_w, x_scales=x_scales, x_lora=x_lora_q)
        k = execute_sycl_layer(kernel_module, x_q, k_wq, k_ws, bias=k_b, lora_b=k_lb, norm_weight=k_norm_w, x_scales=x_scales, x_lora=x_lora_k)
        v = execute_sycl_layer(kernel_module, x_q, v_wq, v_ws, bias=v_b, lora_b=v_lb, x_scales=x_scales, x_lora=x_lora_v)
        ffn_up_res = execute_sycl_layer(
            kernel_module, x_q, ffn_up_wq, ffn_up_ws, bias=ffn_up_b, lora_b=ffn_up_lb,
            activation="gelu_tanh", x_scales=x_scales, x_lora=x_lora_up, fuse_quant_out=True
        )
        ffn_up_bf16, ffn_up_q, ffn_up_scales = ffn_up_res

        # FFN Down consumes INT4 directly (no separate quantization!) and ffn_up_bf16 for LoRA
        ffn_down = execute_sycl_layer(
            kernel_module, ffn_up_q, ffn_down_wq, ffn_down_ws, bias=ffn_down_b, lora_a=ffn_down_la, lora_b=ffn_down_lb,
            x_scales=ffn_up_scales, x_bf16=ffn_up_bf16, activation="none"
        )
        return q, k, v, ffn_down
    return sycl_dense_block


@pytest.fixture(scope="session")
def device():
    return torch.device("xpu")


@pytest.fixture(scope="session")
def candidate_kernel():
    """Import the compiled SYCL kernel."""
    return importlib.import_module(extension_name(load_hparams()))


@pytest.fixture(scope="session")
def kernel(use_reference):
    """Kernel fixture conforming to kernelfoundry interface:

    When --ref is passed, returns the PyTorch BF16 baseline.
    Otherwise, returns the SYCL candidate dense block callable.
    """
    if use_reference:
        return reference_dense_block
    else:
        return make_sycl_dense_block_fn(importlib.import_module(extension_name(load_hparams())))


@pytest.fixture(scope="session")
def block_data(device):
    """Generate deterministic benchmark and test data matching Wan2.2 TI2V-5B Stage 2 shapes.

    Active sequence length M = 1215, LoRA rank r = 32, group size 64.
    Projections: K = 3072, N = 3072, with Q/K RMSNorm.
    FFN Up: K = 3072, N = 14336.
    FFN Down: K = 14336, N = 3072.
    """
    torch.manual_seed(42)
    M = 1215
    K_hidden = 3072
    N_ffn = 14336
    R = 32

    # Input activation
    x = (torch.randn((M, K_hidden), dtype=torch.bfloat16, device=device) * 0.1).contiguous()

    def make_layer_params(k_dim, n_dim, has_norm=False):
        w_bf16 = (torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device) * 0.02).contiguous()
        bias = (torch.randn((n_dim,), dtype=torch.bfloat16, device=device) * 0.01).contiguous()
        lora_a = (torch.randn((R, k_dim), dtype=torch.bfloat16, device=device) * 0.01).contiguous()
        lora_b = (torch.randn((n_dim, R), dtype=torch.bfloat16, device=device) * 0.01).contiguous()
        norm_w = (torch.ones((n_dim,), dtype=torch.bfloat16, device=device) + torch.randn((n_dim,), dtype=torch.bfloat16, device=device) * 0.02).contiguous() if has_norm else None

        # INT4 quantization of weights for SVDQuant
        num_g = k_dim // 64
        w_grouped = w_bf16.float().view(n_dim, num_g, 64)
        g_max = torch.amax(torch.abs(w_grouped), dim=-1, keepdim=True)
        w_scales = torch.clamp(g_max / 7.0, min=1e-8)
        w_int4 = torch.clamp(torch.round(w_grouped / w_scales), -8, 7).to(torch.int8)
        w_q = pack_int4_signed(w_int4.view(n_dim, k_dim))
        w_scale = w_scales.view(n_dim, num_g).to(torch.bfloat16).contiguous()
        w_dpas_b = convert_int4_weights_to_dpas_b(w_q)

        return {
            "w_bf16": w_bf16,
            "bias": bias,
            "lora_a": lora_a,
            "lora_b": lora_b,
            "norm_w": norm_w,
            "w_q": w_q,
            "w_dpas_b": w_dpas_b,
            "w_scale_dpas": convert_w_scales_to_dpas(w_scale),
            "w_scale": w_scale,
        }

    q_p = make_layer_params(K_hidden, K_hidden, has_norm=True)
    k_p = make_layer_params(K_hidden, K_hidden, has_norm=True)
    v_p = make_layer_params(K_hidden, K_hidden, has_norm=False)
    ffn_up_p = make_layer_params(K_hidden, N_ffn, has_norm=False)
    ffn_down_p = make_layer_params(N_ffn, K_hidden, has_norm=False)

    ref_args = (
        x,
        q_p["w_bf16"], q_p["bias"], q_p["lora_a"], q_p["lora_b"], q_p["norm_w"],
        k_p["w_bf16"], k_p["bias"], k_p["lora_a"], k_p["lora_b"], k_p["norm_w"],
        v_p["w_bf16"], v_p["bias"], v_p["lora_a"], v_p["lora_b"],
        ffn_up_p["w_bf16"], ffn_up_p["bias"], ffn_up_p["lora_a"], ffn_up_p["lora_b"],
        ffn_down_p["w_bf16"], ffn_down_p["bias"], ffn_down_p["lora_a"], ffn_down_p["lora_b"],
    )

    sycl_args = (
        x,
        q_p["w_dpas_b"], q_p["w_scale_dpas"], q_p["bias"], q_p["lora_a"], q_p["lora_b"], q_p["norm_w"],
        k_p["w_dpas_b"], k_p["w_scale_dpas"], k_p["bias"], k_p["lora_a"], k_p["lora_b"], k_p["norm_w"],
        v_p["w_dpas_b"], v_p["w_scale_dpas"], v_p["bias"], v_p["lora_a"], v_p["lora_b"],
        ffn_up_p["w_dpas_b"], ffn_up_p["w_scale_dpas"], ffn_up_p["bias"], ffn_up_p["lora_a"], ffn_up_p["lora_b"],
        ffn_down_p["w_dpas_b"], ffn_down_p["w_scale_dpas"], ffn_down_p["bias"], ffn_down_p["lora_a"], ffn_down_p["lora_b"],
    )

    return {
        "x": x,
        "q_p": q_p,
        "k_p": k_p,
        "v_p": v_p,
        "ffn_up_p": ffn_up_p,
        "ffn_down_p": ffn_down_p,
        "ref_args": ref_args,
        "sycl_args": sycl_args,
    }


_HP_DEFAULTS = {
    "GRF": 128,
    "N_TILE": 64,
}
_REBUILD_KNOBS = tuple(_HP_DEFAULTS)

def load_hparams() -> dict:
    """Tuner-supplied overrides, falling back to the committed defaults."""
    return {**_HP_DEFAULTS, **json.loads(os.environ.get("KF_HP_JSON", "{}"))}

def rebuild_hparams() -> dict:
    """Compile-time knobs, kept aligned with the tuner's build cache key."""
    override = os.environ.get("KF_HP_BUILD_JSON")
    return json.loads(override) if override else {
        key: load_hparams()[key] for key in _REBUILD_KNOBS
    }

def extension_name(hp: dict) -> str:
    return "svdquant_dense_kernel_" + "_".join(
        f"{key.lower()}{value}" for key, value in sorted(hp.items())
    )

def write_tuning_header(pkg_dir: Path, hp: dict) -> None:
    lines = ["#pragma once"]
    for key, value in sorted(hp.items()):
        if isinstance(value, bool):
            lines.append(f"#define KF_{key} {int(value)}")
        elif isinstance(value, (int, float)):
            lines.append(f"#define KF_{key} {value}")
        elif isinstance(value, str):
            lines.append(f"#define KF_{key}_{value.upper()} 1")
        else:
            raise TypeError(f"hyperparameter {key} has untranslatable type {type(value)}")
    (pkg_dir / "tuning_params.h").write_text("\n".join(lines) + "\n")


class TestWan22SvdquantDenseBlock(TestBase):
    """KernelFoundry task for optimizing Wan2.2 TI2V-5B Stage 2 camera transformer dense block.

    Block encompasses:
    - 3 Projections (Q/K with RMSNorm, V: 1215 x 3072 x 3072, rank 32)
    - 2 FFN layers (FFN Up: 1215 x 3072 x 14336 + gelu_tanh, FFN Down: 1215 x 14336 x 3072, rank 32)
    """

    def build(self, gpu_arch) -> list[str]:
        if gpu_arch in ("b70", "bmg", "0xe223", "0xE223") or "bmg" in str(gpu_arch).lower():
            gpu_arch = "bmg-g31"
        pkg_dir = Path(__file__).parent.resolve()
        hp = rebuild_hparams()
        write_tuning_header(pkg_dir, hp)
        # Portable local paths within package
        sycl_tla_include = (pkg_dir / "sycl-tla" / "include").resolve()
        sycl_tla_apps = (pkg_dir / "sycl-tla" / "applications").resolve()
        sycl_tla_tools = (pkg_dir / "sycl-tla" / "tools" / "util" / "include").resolve()

        inc_paths = [str(sycl_tla_include), str(sycl_tla_apps), str(sycl_tla_tools)]
        for env_k in ["CPATH", "CPLUS_INCLUDE_PATH"]:
            curr = os.environ.get(env_k, "")
            to_add = [p for p in inc_paths if p not in curr]
            if to_add:
                os.environ[env_k] = ":".join(to_add) + (f":{curr}" if curr else "")

        artifacts = self.compile_torch_extension(
            extension_name=extension_name(hp),
            src=str(pkg_dir / "kernel.sycl"),
            output_dir=pkg_dir,
            gpu_arch=gpu_arch,
            backend="icpx",
        )
        return artifacts

    def build_reference(self, gpu_arch) -> list[str]:
        return []

    # =========================================================================
    # Correctness Tests (Direct bfloat16 comparison against Triton kernel oracle)
    #
    # Threshold Note: In bfloat16, 1 ULP around 2.0 is 0.0156, and around 4.0 is 0.03125.
    # Accumulating 3072..14336 dot-product terms across tile boundaries leads to ~1-2 ULPs
    # of difference due to IEEE 754 non-associativity. Hence atol=0.035..0.05 in bfloat16
    # is mathematically sound.
    # =========================================================================

    def test_correctness_reference_self_consistency(self, kernel, device, block_data, use_reference):
        """Verify that the reference execution is self-consistent and produces finite outputs."""
        if use_reference:
            q, k, v, ffn = kernel(*block_data["ref_args"])
            assert torch.isfinite(q).all(), "Reference Q projection contains NaN/inf"
            assert torch.isfinite(k).all(), "Reference K projection contains NaN/inf"
            assert torch.isfinite(v).all(), "Reference V projection contains NaN/inf"
            assert torch.isfinite(ffn).all(), "Reference FFN contains NaN/inf"

    def test_correctness_vs_triton_projection_q_rmsnorm(self, device, block_data, candidate_kernel, use_reference):
        """Verify SYCL Q projection with fused on-the-fly LoRA and fused RMSNorm in bfloat16."""
        if use_reference:
            pytest.skip("Skipping candidate kernel test during reference baseline evaluation")
        x = block_data["x"]
        p = block_data["q_p"]

        triton_out = triton_svdquant_w4a4_gemm(
            x, p["w_q"], p["w_scale"], bias=p["bias"], lora_a=p["lora_a"], lora_b=p["lora_b"],
            norm_weight=p["norm_w"], activation="none"
        )
        sycl_out = execute_sycl_layer(
            candidate_kernel, x, p["w_dpas_b"], p["w_scale_dpas"], bias=p["bias"], lora_a=p["lora_a"], lora_b=p["lora_b"],
            norm_weight=p["norm_w"], activation="none"
        )

        diff_bf16 = (triton_out - sycl_out).abs().max().item()
        assert diff_bf16 <= 0.05, f"Q (with RMSNorm) diff {diff_bf16:.4f} in bfloat16 exceeds tolerance"

    def test_correctness_vs_triton_projection_v(self, device, block_data, candidate_kernel, use_reference):
        """Verify SYCL V projection (no norm) in bfloat16."""
        if use_reference:
            pytest.skip("Skipping candidate kernel test during reference baseline evaluation")
        x = block_data["x"]
        p = block_data["v_p"]

        triton_out = triton_svdquant_w4a4_gemm(
            x, p["w_q"], p["w_scale"], bias=p["bias"], lora_a=p["lora_a"], lora_b=p["lora_b"], activation="none"
        )
        sycl_out = execute_sycl_layer(
            candidate_kernel, x, p["w_dpas_b"], p["w_scale_dpas"], bias=p["bias"], lora_a=p["lora_a"], lora_b=p["lora_b"], activation="none"
        )

        diff_bf16 = (triton_out - sycl_out).abs().max().item()
        assert diff_bf16 <= 0.035, f"V projection diff {diff_bf16:.4f} in bfloat16 exceeds tolerance"

    def test_dpas_b_weight_layout_roundtrip(self, block_data):
        """The pre-packed DPAS B ABI is [N/16, K/64, 8, 16] and preserves weights exactly."""
        for layer_name in ("q_p", "k_p", "v_p", "ffn_up_p", "ffn_down_p"):
            packed = block_data[layer_name]["w_q"]
            payload = block_data[layer_name]["w_dpas_b"]
            assert payload.dtype == torch.int32 and payload.is_contiguous()
            assert payload.shape == (packed.shape[0] // 16, packed.shape[1] // 32, 8, 16)
            assert torch.equal(unpack_dpas_b_weights(payload), packed)

    def test_correctness_vs_triton_ffn_up_gelu(self, device, block_data, candidate_kernel, use_reference):
        """Verify SYCL FFN Up (with gelu_tanh) and optional output quantization in bfloat16."""
        if use_reference:
            pytest.skip("Skipping candidate kernel test during reference baseline evaluation")
        x = block_data["x"]
        p = block_data["ffn_up_p"]

        triton_out = triton_svdquant_w4a4_gemm(
            x, p["w_q"], p["w_scale"], bias=p["bias"], lora_a=p["lora_a"], lora_b=p["lora_b"], activation="gelu_tanh"
        )
        sycl_res = execute_sycl_layer(
            candidate_kernel, x, p["w_dpas_b"], p["w_scale_dpas"], bias=p["bias"], lora_a=p["lora_a"], lora_b=p["lora_b"],
            activation="gelu_tanh", fuse_quant_out=True
        )
        sycl_out, sycl_qout, sycl_oscales = sycl_res

        diff_bf16 = (triton_out - sycl_out).abs().max().item()
        assert diff_bf16 <= 0.035, f"FFN Up gelu_tanh diff {diff_bf16:.4f} in bfloat16 exceeds tolerance"
        assert sycl_qout.shape == (1215, 14336 // 2), "Output INT4 tensor has unexpected shape"

    @staticmethod
    def _assert_int4_quantizes(q, scales, ref):
        """q/scales must be a round-to-nearest INT4 group-64 quantization of ref."""
        amax = ref.float().view(ref.shape[0], -1, 64).abs().amax(-1)
        torch.testing.assert_close(scales, torch.clamp(amax / 7.0, min=1e-8), rtol=1e-6, atol=0)
        s = scales.repeat_interleave(64, dim=1)
        err = (unpack_int4_signed(q).float() * s - ref.float()).abs()
        assert (err <= 0.5 * s * (1 + 1e-5)).all(), f"INT4 rounding error {(err / s).max().item():.3f} LSB"

    def test_correctness_prologue_quant_and_lora_down(self, block_data, candidate_kernel, use_reference):
        """One prologue pass: INT4 quantization of x and x @ lora_a^T for several layers at once."""
        if use_reference:
            pytest.skip("Skipping candidate kernel test during reference baseline evaluation")
        x = block_data["x"]
        lora_a = torch.cat([block_data[n]["lora_a"] for n in ("q_p", "k_p", "v_p", "ffn_up_p")])
        x_q, x_scales, x_lora = candidate_kernel.quantize_activation_int4(x, lora_a)
        self._assert_int4_quantizes(x_q, x_scales, x)
        torch.testing.assert_close(x_lora, x.float() @ lora_a.float().t(), rtol=1e-4, atol=1e-4)

    def test_correctness_fused_output_quant(self, block_data, candidate_kernel, use_reference):
        """FFN-up qout/oscales must quantize the kernel's own BF16 output."""
        if use_reference:
            pytest.skip("Skipping candidate kernel test during reference baseline evaluation")
        p = block_data["ffn_up_p"]
        out, qout, oscales = execute_sycl_layer(
            candidate_kernel, block_data["x"], p["w_dpas_b"], p["w_scale_dpas"], bias=p["bias"],
            lora_a=p["lora_a"], lora_b=p["lora_b"], activation="gelu_tanh", fuse_quant_out=True
        )
        self._assert_int4_quantizes(qout, oscales, out)

    def test_correctness_vs_triton_ffn_down(self, device, block_data, candidate_kernel, use_reference):
        """Verify SYCL FFN Down (K=14336) when consuming INT4 input directly."""
        if use_reference:
            pytest.skip("Skipping candidate kernel test during reference baseline evaluation")
        p = block_data["ffn_down_p"]
        # Simulate pre-quantized INT4 activation from preceding FFN up
        h_bf16 = (torch.randn((1215, 14336), dtype=torch.bfloat16, device=device) * 0.1).contiguous()
        h_q, h_scales = quantize_activation_int4(h_bf16, group_size=64)

        triton_out = triton_svdquant_w4a4_gemm(
            h_bf16, p["w_q"], p["w_scale"], bias=p["bias"], lora_a=p["lora_a"], lora_b=p["lora_b"], activation="none"
        )
        sycl_out = execute_sycl_layer(
            candidate_kernel, h_q, p["w_dpas_b"], p["w_scale_dpas"], bias=p["bias"], lora_a=p["lora_a"], lora_b=p["lora_b"],
            x_scales=h_scales, x_bf16=h_bf16, activation="none"
        )

        diff_bf16 = (triton_out - sycl_out).abs().max().item()
        assert diff_bf16 <= 0.05, f"FFN Down diff {diff_bf16:.4f} in bfloat16 exceeds tolerance"

    def test_correctness_vs_triton_dense_block(self, device, block_data, candidate_kernel, use_reference):
        """Verify full chained SYCL dense block vs sequential Triton execution."""
        if use_reference:
            pytest.skip("Skipping candidate kernel test during reference baseline evaluation")
        x = block_data["x"]
        qp = block_data["q_p"]
        kp = block_data["k_p"]
        vp = block_data["v_p"]
        up = block_data["ffn_up_p"]
        dp = block_data["ffn_down_p"]

        # Triton block reference
        t_q = triton_svdquant_w4a4_gemm(x, qp["w_q"], qp["w_scale"], bias=qp["bias"], lora_a=qp["lora_a"], lora_b=qp["lora_b"], norm_weight=qp["norm_w"])
        t_k = triton_svdquant_w4a4_gemm(x, kp["w_q"], kp["w_scale"], bias=kp["bias"], lora_a=kp["lora_a"], lora_b=kp["lora_b"], norm_weight=kp["norm_w"])
        t_v = triton_svdquant_w4a4_gemm(x, vp["w_q"], vp["w_scale"], bias=vp["bias"], lora_a=vp["lora_a"], lora_b=vp["lora_b"])
        t_up = triton_svdquant_w4a4_gemm(x, up["w_q"], up["w_scale"], bias=up["bias"], lora_a=up["lora_a"], lora_b=up["lora_b"], activation="gelu_tanh")
        t_down = triton_svdquant_w4a4_gemm(t_up, dp["w_q"], dp["w_scale"], bias=dp["bias"], lora_a=dp["lora_a"], lora_b=dp["lora_b"])

        # Candidate block
        sycl_dense_block_fn = make_sycl_dense_block_fn(candidate_kernel)
        s_q, s_k, s_v, s_down = sycl_dense_block_fn(*block_data["sycl_args"])

        assert (t_q - s_q).abs().max().item() <= 0.05, "Q diff exceeds bfloat16 tolerance"
        assert (t_k - s_k).abs().max().item() <= 0.05, "K diff exceeds bfloat16 tolerance"
        assert (t_v - s_v).abs().max().item() <= 0.05, "V diff exceeds bfloat16 tolerance"
        assert (t_down - s_down).abs().max().item() <= 0.05, "FFN down diff exceeds bfloat16 tolerance"

    # =========================================================================
    # Performance Benchmarks (Sum of 3 Projections + 2 FFN Layers)
    # =========================================================================

    @pytest.mark.performance
    def test_benchmark_dense_block_total(self, kernel, device, block_data, use_reference, measure_runtime_torch):
        """Target benchmark metric: Sum of 3 projections (Q/K RMSNorm, V) + 2 FFN layers (including gelu_tanh).

        Reference: PyTorch native BF16 execution.
        Kernel: Fused SVDQuant INT4 + BF16 LoRA execution with FFN chaining.
        """
        args = block_data["ref_args"] if use_reference else block_data["sycl_args"]
        measure_runtime_torch(kernel, device, args=args)

    @pytest.mark.performance
    def test_benchmark_projection_q(self, kernel, device, block_data, use_reference, candidate_kernel, measure_runtime_torch):
        """Benchmark Q projection with RMSNorm: M = 1215, K = 3072, N = 3072, rank 32."""
        x = block_data["x"]
        p = block_data["q_p"]
        if use_reference:
            measure_runtime_torch(
                bf16_linear, device,
                args=(x, p["w_bf16"], p["bias"], p["lora_a"], p["lora_b"]),
                kwargs={"norm_weight": p["norm_w"]}
            )
        else:
            q_fn = candidate_kernel.svdquant_w4a4_gemm
            measure_runtime_torch(
                q_fn, device,
                args=(x, p["w_dpas_b"], p["w_scale_dpas"]),
                kwargs={"bias": p["bias"], "lora_a": p["lora_a"], "lora_b": p["lora_b"], "norm_weight": p["norm_w"], "activation": "none"}
            )

    @pytest.mark.performance
    def test_benchmark_projection_v(self, kernel, device, block_data, use_reference, candidate_kernel, measure_runtime_torch):
        """Benchmark V projection: M = 1215, K = 3072, N = 3072, rank 32."""
        x = block_data["x"]
        p = block_data["v_p"]
        if use_reference:
            measure_runtime_torch(
                bf16_linear, device,
                args=(x, p["w_bf16"], p["bias"], p["lora_a"], p["lora_b"]),
            )
        else:
            measure_runtime_torch(
                candidate_kernel.svdquant_w4a4_gemm, device,
                args=(x, p["w_dpas_b"], p["w_scale_dpas"]),
                kwargs={"bias": p["bias"], "lora_a": p["lora_a"], "lora_b": p["lora_b"], "activation": "none"},
            )

    @pytest.mark.performance
    def test_benchmark_ffn_up(self, kernel, device, block_data, use_reference, candidate_kernel, measure_runtime_torch):
        """Benchmark FFN up-projection with gelu_tanh: M = 1215, K = 3072, N = 14336, rank 32."""
        x = block_data["x"]
        p = block_data["ffn_up_p"]
        if use_reference:
            measure_runtime_torch(
                bf16_linear_activation, device,
                args=(x, p["w_bf16"], p["bias"], p["lora_a"], p["lora_b"]),
                kwargs={"activation": "gelu_tanh"}
            )
        else:
            up_fn = candidate_kernel.svdquant_w4a4_gemm
            measure_runtime_torch(
                up_fn, device,
                args=(x, p["w_dpas_b"], p["w_scale_dpas"]),
                kwargs={"bias": p["bias"], "lora_a": p["lora_a"], "lora_b": p["lora_b"], "activation": "gelu_tanh", "fuse_quant_out": True}
            )

    @pytest.mark.performance
    def test_benchmark_ffn_down(self, kernel, device, block_data, use_reference, candidate_kernel, measure_runtime_torch):
        """Benchmark FFN down-projection: M = 1215, K = 14336, N = 3072, rank 32."""
        h_bf16 = (torch.zeros((1215, 14336), dtype=torch.bfloat16, device=device)).contiguous()
        h_q, h_scales = quantize_activation_int4(h_bf16, group_size=64)
        p = block_data["ffn_down_p"]
        if use_reference:
            measure_runtime_torch(
                bf16_linear, device,
                args=(h_bf16, p["w_bf16"], p["bias"], p["lora_a"], p["lora_b"])
            )
        else:
            down_fn = candidate_kernel.svdquant_w4a4_gemm
            measure_runtime_torch(
                down_fn, device,
                args=(h_q, p["w_dpas_b"], p["w_scale_dpas"]),
                kwargs={"bias": p["bias"], "lora_a": p["lora_a"], "lora_b": p["lora_b"], "x_scales": h_scales, "x_bf16": h_bf16, "activation": "none"}
            )


if __name__ == "__main__":
    task = TestWan22SvdquantDenseBlock()
    task.build(TestBase.get_machine_gpu_arch())
