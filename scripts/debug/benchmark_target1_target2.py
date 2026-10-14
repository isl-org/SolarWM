import time
import torch
import torch.nn.functional as F
from solarwm.backends.wan22.runtime.modeling.vae import CausalConv3d
from solarwm.backends.wan22.runtime.modeling.vae_quarot import (
    get_hadamard_matrix,
    quantize_conv_weight_quarot,
)
from solarwm.kernels.qconv_onednn import fast_qconv3d

def benchmark_target1_and_target2():
    dev = "xpu"
    torch.manual_seed(42)

    C = 256
    N, T, H_dim, W_dim = 1, 3, 30, 54
    block_size = 256

    conv = CausalConv3d(C, C, 3, padding=1).to(device=dev, dtype=torch.bfloat16)
    conv.to(memory_format=torch.channels_last_3d)

    H = get_hadamard_matrix(block_size, device=dev, dtype=torch.bfloat16)
    qw, w_scale, zero_w = quantize_conv_weight_quarot(conv, H)

    x = torch.randn(N, C, T, H_dim, W_dim, device=dev, dtype=torch.bfloat16).to(
        memory_format=torch.channels_last_3d
    )
    shortcut = torch.randn(N, C, T, H_dim, W_dim, device=dev, dtype=torch.bfloat16).to(
        memory_format=torch.channels_last_3d
    )

    spatial_pad = [0, 1, 1]

    # Staged input (temporal causal pad: pad_t=2)
    staged = F.pad(x, (0, 0, 0, 0, 2, 0)) # [1, 256, 5, 30, 54]
    N_s, C_s, T_s, H_s, W_s = staged.shape
    M_s = N_s * T_s * H_s * W_s * (C_s // block_size)

    rot_buf = torch.empty((M_s, block_size), dtype=torch.bfloat16, device=dev)

    # 1. Baseline: Eager QuaRot + separate aten::add
    accum_base = shortcut.clone()
    x_f = staged.permute(0, 2, 3, 4, 1).reshape(-1, block_size)
    x_r = x_f @ H
    sc = (x_r.abs().amax() / 127.0).clamp(min=1e-8).to(torch.float32)
    qx = (x_r / sc).round().clamp(-127, 127).to(torch.int8).view(N_s, T_s, H_s, W_s, C_s).permute(0, 4, 1, 2, 3).contiguous(memory_format=torch.channels_last_3d)
    conv_out = fast_qconv3d(qx, sc, qw, w_scale, conv.bias.float(), conv.stride, spatial_pad, conv.dilation, conv.groups)
    out_base = conv_out + accum_base

    # 2. Optimized: Target 1 (append_sum) + Target 2 (direct quantize)
    accum_opt = shortcut.clone()
    torch.mm(x_f, H, out=rot_buf)
    # Direct dynamic quant:
    x_absmax = rot_buf.abs().amax()
    sc_opt = (x_absmax / 127.0).clamp(min=1e-8).to(torch.float32)
    qx_opt = (rot_buf / sc_opt).round().clamp(-127, 127).to(torch.int8).view(N_s, T_s, H_s, W_s, C_s).permute(0, 4, 1, 2, 3)
    fast_qconv3d(qx_opt, sc_opt, qw, w_scale, conv.bias.float(), conv.stride, spatial_pad, conv.dilation, conv.groups, accumulate_into=accum_opt)

    diff_val = (out_base.float() - accum_opt.float()).abs().max().item()
    print(f"Numerical max diff: {diff_val}")

    # Microbenchmarks of components:
    N_ITERS = 200

    # Test standalone add:
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        res = conv_out + accum_base
    torch.xpu.synchronize()
    t_add = (time.perf_counter() - t0) * 1000 / N_ITERS

    # Test conv alone vs conv with append_sum:
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        out = fast_qconv3d(qx, sc, qw, w_scale, conv.bias.float(), conv.stride, spatial_pad, conv.dilation, conv.groups)
    torch.xpu.synchronize()
    t_conv_alone = (time.perf_counter() - t0) * 1000 / N_ITERS

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        out = fast_qconv3d(qx, sc, qw, w_scale, conv.bias.float(), conv.stride, spatial_pad, conv.dilation, conv.groups, accumulate_into=accum_opt)
    torch.xpu.synchronize()
    t_conv_accum = (time.perf_counter() - t0) * 1000 / N_ITERS

    print(f"Standalone aten::add latency:     {t_add:.3f} ms")
    print(f"fast_qconv3d alone:               {t_conv_alone:.3f} ms")
    print(f"fast_qconv3d with append_sum:     {t_conv_accum:.3f} ms")
    print(f"Hardware accumulate overhead:     {t_conv_accum - t_conv_alone:.3f} ms")
    print(f"Net savings of append_sum:        {t_add - (t_conv_accum - t_conv_alone):.3f} ms per block")
    print(f"Net savings across 14 blocks:     {(t_add - (t_conv_accum - t_conv_alone)) * 14:.2f} ms per tile")

if __name__ == "__main__":
    benchmark_target1_and_target2()
