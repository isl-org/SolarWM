"""Unit tests for Block-256 QuaRot W8A8 dynamic quantization on Wan2.2 VAE decoder."""

import pytest
import torch
import torch.nn.functional as F

from solarwm.backends.wan22.runtime.modeling.vae import (
    CausalConv3d,
    Decoder3d,
    WanVAE_,
)
from solarwm.backends.wan22.runtime.modeling.vae_quarot import (
    apply_vae_quarot,
    get_hadamard_matrix,
    install_quarot_conv,
    quantize_conv_weight_quarot,
    rotate_channels_last,
    uninstall_quarot_conv,
)


def test_hadamard_matrix_orthogonality_and_symmetry():
    # Verify exact algebraic properties in float64
    H = get_hadamard_matrix(256, device="cpu", dtype=torch.float64)
    assert H.shape == (256, 256)
    # Symmetry: H == H^T
    assert torch.equal(H, H.T)
    # Orthogonality: H @ H == I
    eye = torch.eye(256, dtype=torch.float64)
    diff = (H @ H - eye).abs().max().item()
    assert diff < 1e-12, f"Hadamard matrix not orthogonal: diff={diff}"


def test_quarot_exact_mathematical_invariance_conv3d():
    # Verify (X R) * (R^T W) == X * W down to machine precision
    torch.manual_seed(42)
    device = "cpu"
    N, IC, in_d, in_h, in_w = 1, 512, 3, 8, 8
    OC = 512
    kT, kH, kW = 3, 3, 3

    H = get_hadamard_matrix(256, device=device, dtype=torch.float64)
    x = torch.randn((N, IC, in_d, in_h, in_w), dtype=torch.float64, device=device)
    w = torch.randn((OC, IC, kT, kH, kW), dtype=torch.float64, device=device)
    bias = torch.randn(OC, dtype=torch.float64, device=device)

    # 1. Unrotated reference conv
    y_ref = F.conv3d(x, w, bias, stride=1, padding=1)

    # 2. Rotated activations
    x_rot = rotate_channels_last(x, H)

    # 3. Rotated weights along IC
    w_perm = w.permute(0, 2, 3, 4, 1)
    w_rot = (w_perm.reshape(-1, 256) @ H).view(OC, kT, kH, kW, IC).permute(0, 4, 1, 2, 3)

    # 4. Rotated conv
    y_rot = F.conv3d(x_rot, w_rot, bias, stride=1, padding=1)

    diff = (y_ref - y_rot).abs().max().item()
    assert diff < 1e-10, f"Max diff between unrotated and rotated conv too high: {diff}"


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required for oneDNN qconv")
@pytest.mark.parametrize("C", [256, 512, 1024])
def test_quarot_onednn_dynamic_quant_snr(C: int):
    torch.manual_seed(42)
    device = "xpu"
    N, T, H_dim, W_dim = 1, 3, 12, 12

    conv = CausalConv3d(C, C, 3, padding=1).to(device=device, dtype=torch.bfloat16)
    conv.to(memory_format=torch.channels_last_3d)

    x = torch.randn(N, C, T, H_dim, W_dim, device=device, dtype=torch.bfloat16).to(
        memory_format=torch.channels_last_3d
    )

    # Unquantized BF16 reference
    ref_out = conv(x)

    # Install QuaRot W8A8
    H = get_hadamard_matrix(256, device=device, dtype=torch.bfloat16)
    install_quarot_conv(conv, H, use_compile=False)

    q_out = conv(x)
    snr = 20 * torch.log10(ref_out.norm() / (ref_out - q_out).norm()).item()
    assert snr > 30.0, f"SNR {snr:.2f} dB is below 30 dB gate for C={C}"
    assert q_out.is_contiguous(memory_format=torch.channels_last_3d)

    # Uninstall
    uninstall_quarot_conv(conv)
    unpatched_out = conv(x)
    assert torch.equal(ref_out, unpatched_out)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required for oneDNN qconv")
def test_quarot_outlier_suppression_advantage():
    torch.manual_seed(42)
    device = "xpu"
    C = 512
    conv = CausalConv3d(C, C, 3, padding=1).to(device=device, dtype=torch.bfloat16)
    conv.to(memory_format=torch.channels_last_3d)

    # Activation with channel outlier (15x spike on channel 17)
    x = torch.randn(1, C, 3, 12, 12, device=device, dtype=torch.bfloat16).to(
        memory_format=torch.channels_last_3d
    )
    x[:, 17, :, :, :] *= 15.0

    ref_out = conv(x)

    # 1. Unrotated W8A8 dynamic quantization
    w = conv.weight.data
    w_scale = (w.abs().amax(dim=(1, 2, 3, 4)) / 127.0).clamp(min=1e-8)
    qw = (w / w_scale[:, None, None, None, None]).round().clamp(-127, 127).to(torch.int8).to(
        memory_format=torch.channels_last_3d
    )
    x_pad = F.pad(x, list(conv._padding))
    scale_unrot = (x_pad.abs().amax(dim=(0, 2, 3, 4)).amax() / 127.0).clamp(min=1e-8)
    qx_unrot = (x_pad / scale_unrot).round().clamp(-127, 127).to(torch.int8)

    zero_w = torch.zeros((C,), dtype=torch.int32, device=device)
    zero_x = torch.tensor(0, dtype=torch.int32, device=device)
    out_unrot = torch.ops.onednn.qconv_pointwise.tensor(
        qx_unrot, scale_unrot, zero_x,
        qw, w_scale, zero_w,
        conv.bias.data.float(), list(conv.stride), [0, 0, 0], list(conv.dilation), conv.groups,
        1.0, 0, torch.bfloat16, "none", [], ""
    )
    snr_unrot = 20 * torch.log10(ref_out.norm() / (ref_out - out_unrot).norm()).item()

    # 2. QuaRot W8A8
    H = get_hadamard_matrix(256, device=device, dtype=torch.bfloat16)
    install_quarot_conv(conv, H, use_compile=False)
    out_quarot = conv(x)
    snr_quarot = 20 * torch.log10(ref_out.norm() / (ref_out - out_quarot).norm()).item()

    # QuaRot should provide significant SNR improvement over unrotated on outliers
    gain = snr_quarot - snr_unrot
    assert gain > 10.0, f"QuaRot gain {gain:.2f} dB was expected to be > 10 dB on outliers"


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required for oneDNN qconv")
def test_quarot_streaming_cache_continuity():
    torch.manual_seed(42)
    device = "xpu"
    C = 512
    conv = CausalConv3d(C, C, 3, padding=1).to(device=device, dtype=torch.bfloat16)
    conv.to(memory_format=torch.channels_last_3d)

    H = get_hadamard_matrix(256, device=device, dtype=torch.bfloat16)

    # Simulate 3 chunks with causal temporal cache
    feat_cache_ref = [None]
    feat_cache_q = [None]

    for chunk in range(3):
        x = torch.randn(1, C, 3, 12, 12, device=device, dtype=torch.bfloat16).to(
            memory_format=torch.channels_last_3d
        )
        cache_x = x[:, :, -2:, :, :].clone()

        # BF16 Reference
        uninstall_quarot_conv(conv)
        ref_out = conv(x, feat_cache_ref[0])
        feat_cache_ref[0] = cache_x.clone()

        # QuaRot
        install_quarot_conv(conv, H, use_compile=False)
        q_out = conv(x, feat_cache_q[0])
        feat_cache_q[0] = cache_x.clone()

        snr = 20 * torch.log10(ref_out.norm() / (ref_out - q_out).norm()).item()
        assert snr > 32.0, f"Chunk {chunk} SNR {snr:.2f} dB below 32 dB"


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU device required for oneDNN qconv")
def test_apply_and_unpatch_vae_quarot():
    device = "xpu"
    dec = Decoder3d(
        dim=256,
        z_dim=48,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True][::-1],
        dropout=0.0,
    ).to(device, dtype=torch.bfloat16)

    # 1. Enable QuaRot
    patched = apply_vae_quarot(dec, enabled=True, use_compile=False)
    assert len(patched) == 28, f"Expected 28 primary convolutions, got {len(patched)}"

    # Verify all patched layers have _quarot_enabled == True
    for name, module in dec.named_modules():
        if name in patched:
            assert getattr(module, "_quarot_enabled", False) is True

    # 2. Disable QuaRot
    unpatched = apply_vae_quarot(dec, enabled=False)
    assert len(unpatched) == 28
    for name, module in dec.named_modules():
        if name in unpatched:
            assert getattr(module, "_quarot_enabled", False) is False
