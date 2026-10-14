import pytest
import torch

from solarwm.kernels.dit_svdquant import (
    bf16_attention_projections,
    bf16_ffn,
    bf16_linear,
    bf16_linear_activation,
    convert_int4_weights_to_dpas_b,
    convert_w_scales_to_dpas,
    e8m0_to_f32,
    f32_to_e8m0,
    pack_int4_signed,
    pseudo_qdq_linear,
    quantize_activation_int4,
    svdquant_fused_ffn_ref,
    svdquant_w4a4_linear_ref,
    triton_svdquant_w4a4_gemm,
    unpack_int4_signed,
)


def test_pack_unpack_int4_roundtrip():
    torch.manual_seed(42)
    original = torch.randint(-8, 8, (128, 256), dtype=torch.int8)

    packed = pack_int4_signed(original)
    assert packed.shape == (128, 128)
    assert packed.dtype == torch.int8

    unpacked = unpack_int4_signed(packed)
    assert unpacked.shape == original.shape
    assert unpacked.dtype == torch.int8
    assert torch.equal(original, unpacked)


def test_dpas_b_weight_layout_shape():
    """Model-load prepacking emits one raw DPAS B fragment per N/K tile."""
    packed = pack_int4_signed(torch.randint(-8, 8, (32, 128), dtype=torch.int8))
    payload = convert_int4_weights_to_dpas_b(packed)
    assert payload.shape == (2, 2, 8, 16)
    assert payload.dtype == torch.int32
    # Dword d of lane n holds K [8d, 8d + 8) of column n (row-wise VNNI-8).
    assert torch.equal(payload[1, 1, 3, 5:6].view(torch.int8), packed[16 + 5, 32 + 12:32 + 16])


def test_dpas_w_scale_layout():
    w_scale = torch.rand(32, 2, dtype=torch.bfloat16)
    scales = convert_w_scales_to_dpas(w_scale)
    assert scales.shape == (2, 32) and scales.dtype == torch.float32 and scales.is_contiguous()
    assert torch.equal(scales, w_scale.float().t())


def test_e8m0_bitshift_conversion():
    # Test specific exact power of 2 exponents
    # e8m0 = 127 corresponds to 2^(127-127) = 1.0
    # e8m0 = 128 corresponds to 2^(128-127) = 2.0
    # e8m0 = 126 corresponds to 2^(126-127) = 0.5
    e8m0 = torch.tensor([0, 126, 127, 128, 130], dtype=torch.uint8)
    f32 = e8m0_to_f32(e8m0)
    expected = torch.tensor([0.0, 0.5, 1.0, 2.0, 8.0], dtype=torch.float32)
    assert torch.equal(f32, expected)

    # Test round trip from float
    scales = torch.tensor([0.5, 1.0, 2.0, 8.0], dtype=torch.float32)
    e_encoded = f32_to_e8m0(scales)
    f_decoded = e8m0_to_f32(e_encoded)
    assert torch.equal(scales, f_decoded)


def test_bf16_reference_baseline():
    torch.manual_seed(42)
    m, k, n = 16, 64, 32
    x = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    b = torch.randn(n, dtype=torch.bfloat16)

    # 1. Linear
    out_lin = bf16_linear(x, w, b)
    assert out_lin.shape == (m, n)

    # 2. Linear + GELU
    out_gelu = bf16_linear_activation(x, w, b, activation="gelu_tanh")
    assert out_gelu.shape == (m, n)

    # 3. Two-layer FFN
    w_down = torch.randn(k, n, dtype=torch.bfloat16)
    b_down = torch.randn(k, dtype=torch.bfloat16)
    out_ffn = bf16_ffn(x, w, b, w_down, b_down, activation="gelu_tanh")
    assert out_ffn.shape == (m, k)

    # 4. Attention projections
    q, k_proj, v = bf16_attention_projections(x, w, w, w, b, b, b)
    assert q.shape == (m, n)
    assert k_proj.shape == (m, n)
    assert v.shape == (m, n)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Requires Intel XPU")
@pytest.mark.parametrize("activation", ["none", "gelu_tanh", "silu", "gelu"])
def test_triton_svdquant_parity_float_scales(activation: str):
    torch.manual_seed(42)
    device = "xpu"

    m, k, n = 32, 128, 64
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)

    w_raw = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=device)
    w_packed = pack_int4_signed(w_raw)
    w_scale = torch.rand(n, k // 64, device=device, dtype=torch.bfloat16) * 0.1 + 0.01

    s_in = torch.rand(k, device=device, dtype=torch.bfloat16) + 0.5
    bias = torch.randn(n, device=device, dtype=torch.bfloat16)

    lora_a = torch.randn(16, k, device=device, dtype=torch.bfloat16) * 0.01
    lora_b = torch.randn(n, 16, device=device, dtype=torch.bfloat16) * 0.01

    ref = svdquant_w4a4_linear_ref(
        x=x,
        w_packed=w_packed,
        w_scale=w_scale,
        s_in=s_in,
        bias=bias,
        lora_a=lora_a,
        lora_b=lora_b,
        group_size=64,
        activation=activation,
    )

    out = triton_svdquant_w4a4_gemm(
        x=x,
        w_packed=w_packed,
        w_scale=w_scale,
        s_in=s_in,
        bias=bias,
        lora_a=lora_a,
        lora_b=lora_b,
        group_size=64,
        activation=activation,
    )

    max_diff = torch.max(torch.abs(ref - out)).item()
    print(f"\n[Float scales | {activation}] Max diff: {max_diff:.4f}")
    assert max_diff < 0.1, f"Parity mismatch for {activation}: {max_diff}"


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Requires Intel XPU")
def test_triton_svdquant_parity_e8m0_scales():
    """Verify Triton kernel executes E8M0 power-of-2 scales via bit shifts with exact parity."""
    torch.manual_seed(42)
    device = "xpu"

    m, k, n = 32, 128, 64
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)

    w_raw = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=device)
    w_packed = pack_int4_signed(w_raw)

    # Generate E8M0 scale bytes
    w_scale_f32 = torch.rand(n, k // 64, device=device, dtype=torch.float32) * 0.1 + 0.01
    w_scale_e8m0 = f32_to_e8m0(w_scale_f32)
    assert w_scale_e8m0.dtype == torch.uint8

    s_in = torch.rand(k, device=device, dtype=torch.bfloat16) + 0.5
    bias = torch.randn(n, device=device, dtype=torch.bfloat16)
    lora_a = torch.randn(16, k, device=device, dtype=torch.bfloat16) * 0.01
    lora_b = torch.randn(n, 16, device=device, dtype=torch.bfloat16) * 0.01

    ref = svdquant_w4a4_linear_ref(
        x=x,
        w_packed=w_packed,
        w_scale=w_scale_e8m0,
        s_in=s_in,
        bias=bias,
        lora_a=lora_a,
        lora_b=lora_b,
        group_size=64,
        activation="gelu_tanh",
    )

    out = triton_svdquant_w4a4_gemm(
        x=x,
        w_packed=w_packed,
        w_scale=w_scale_e8m0,
        s_in=s_in,
        bias=bias,
        lora_a=lora_a,
        lora_b=lora_b,
        group_size=64,
        activation="gelu_tanh",
    )

    max_diff = torch.max(torch.abs(ref - out)).item()
    print(f"\n[E8M0 scales via bit shift] Max diff: {max_diff:.6f}")
    assert max_diff < 0.05, f"E8M0 parity mismatch: {max_diff}"


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Requires Intel XPU")
def test_triton_fused_output_quantization():
    """Verify Triton kernel can optionally fuse output INT4 quantization & packing."""
    torch.manual_seed(42)
    device = "xpu"

    m, k, n = 32, 128, 64
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)

    w_raw = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=device)
    w_packed = pack_int4_signed(w_raw)
    w_scale = torch.rand(n, k // 64, device=device, dtype=torch.bfloat16) * 0.1 + 0.01

    s_in = torch.rand(k, device=device, dtype=torch.bfloat16) + 0.5
    bias = torch.randn(n, device=device, dtype=torch.bfloat16)
    lora_a = torch.randn(16, k, device=device, dtype=torch.bfloat16) * 0.01
    lora_b = torch.randn(n, 16, device=device, dtype=torch.bfloat16) * 0.01

    ref_bf16, ref_qout, ref_oscale = svdquant_w4a4_linear_ref(
        x=x,
        w_packed=w_packed,
        w_scale=w_scale,
        s_in=s_in,
        bias=bias,
        lora_a=lora_a,
        lora_b=lora_b,
        group_size=64,
        activation="gelu_tanh",
        fuse_quant_out=True,
    )

    out_bf16, qout, oscale = triton_svdquant_w4a4_gemm(
        x=x,
        w_packed=w_packed,
        w_scale=w_scale,
        s_in=s_in,
        bias=bias,
        lora_a=lora_a,
        lora_b=lora_b,
        group_size=64,
        activation="gelu_tanh",
        fuse_quant_out=True,
    )

    assert qout.shape == (m, n // 2)
    assert qout.dtype == torch.int8
    assert oscale.shape == (m, n // 64)

    diff_bf16 = torch.max(torch.abs(ref_bf16 - out_bf16)).item()
    diff_scale = torch.max(torch.abs(ref_oscale - oscale)).item()
    print(f"\n[Fused QOut] BF16 diff: {diff_bf16:.6f}, Scale diff: {diff_scale:.6f}")
    assert diff_bf16 < 0.05
    assert diff_scale < 0.05
