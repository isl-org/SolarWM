"""Micro-benchmark Wan2.2 VAE decoder CausalConv3d input staging on XPU.

Compares, per representative decoder conv shape:
  cat_pad   torch.cat(cache, x) + F.pad (spatial + temporal) + conv(padding=0)   [legacy]
  buf_conv  persistent [cache | x] buffer + conv with spatial padding inside oneDNN
and reports which variants still launch oneDNN ``conv_reorder`` kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

CL = torch.channels_last_3d
SHAPES = [  # (C_in, C_out, T, H, W) decoder conv inputs before padding
    (1024, 1024, 1, 30, 54),
    (1024, 512, 2, 60, 108),
    (512, 512, 4, 120, 216),
    (256, 256, 4, 240, 432),
]


def timed(fn, iters=10):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / iters


def kernels(fn):
    with profile(activities=[ProfilerActivity.XPU]) as prof:
        fn()
        torch.xpu.synchronize()
    out = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.XPU:
            name = e.name.split("<")[0][:28]
            n, us = out.get(name, (0, 0.0))
            out[name] = (n + 1, round(us + e.device_time, 1))
    return out


def main():
    dev, dt = "xpu", torch.bfloat16
    for cin, cout, t, h, w in SHAPES:
        x = torch.randn(1, cin, t, h, w, device=dev, dtype=dt).contiguous(memory_format=CL)
        cache = torch.randn(1, cin, 2, h, w, device=dev, dtype=dt).contiguous(memory_format=CL)
        for wfmt in ("cl3d", "contig"):
            weight = torch.randn(cout, cin, 3, 3, 3, device=dev, dtype=dt) * 0.02
            weight = weight.contiguous(memory_format=CL) if wfmt == "cl3d" else weight.contiguous()
            bias = torch.zeros(cout, device=dev, dtype=dt)
            buf = torch.empty(1, cin, t + 2, h, w, device=dev, dtype=dt, memory_format=CL)

            def cat_pad():
                y = F.pad(torch.cat([cache, x], dim=2), (1, 1, 1, 1, 0, 0))
                return F.conv3d(y, weight, bias)

            def buf_conv():
                buf[:, :, :2].copy_(cache)
                buf[:, :, 2:].copy_(x)
                return F.conv3d(buf, weight, bias, padding=(0, 1, 1))

            pbuf = torch.zeros(1, cin, t + 2, h + 2, w + 2, device=dev, dtype=dt).contiguous(
                memory_format=CL
            )

            def pbuf_conv():
                pbuf[:, :, :2, 1:-1, 1:-1].copy_(cache)
                pbuf[:, :, 2:, 1:-1, 1:-1].copy_(x)
                return F.conv3d(pbuf, weight, bias)

            ref = cat_pad()
            diffs = [(ref.float() - f().float()).abs().max().item() for f in (buf_conv, pbuf_conv)]
            print(
                f"C{cin}->{cout} T{t} {h}x{w} w={wfmt:6s} | cat_pad {timed(cat_pad):7.3f} ms "
                f"| buf_conv {timed(buf_conv):7.3f} ms | pbuf_conv {timed(pbuf_conv):7.3f} ms "
                f"| max|diff| {diffs}"
            )
            for name, fn in (("cat_pad", cat_pad), ("buf_conv", buf_conv), ("pbuf_conv", pbuf_conv)):
                print(f"    {name:9s} {kernels(fn)}")


if __name__ == "__main__":
    main()
