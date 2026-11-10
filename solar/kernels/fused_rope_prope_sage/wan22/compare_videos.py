"""Compare two generated videos frame by frame (same prompt, image, seed; different attention).

    python3 wan/compare_videos.py wan/outputs/cat480_sdpa.npy wan/outputs/cat480_sage.npy --out wan/outputs/cat480
Writes <out>_compare.json (PSNR / SSIM / mean abs diff per frame, mean and worst) and <out>_side_by_side.mp4
(left: first video, middle: second, right: |difference| x 8).
"""
import argparse
import json
import subprocess

import numpy as np
import torch
import torch.nn.functional as F


def luma(x):  # [F, H, W, 3] uint8 -> [F, 1, H, W] float in [0, 1]
    x = torch.from_numpy(x).float() / 255.0
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).unsqueeze(1)


def ssim(a, b, win=11, sigma=1.5):
    g = torch.exp(-((torch.arange(win) - win // 2) ** 2) / (2 * sigma ** 2)); g = (g / g.sum())
    k = (g[:, None] @ g[None, :]).view(1, 1, win, win)
    mu_a, mu_b = F.conv2d(a, k), F.conv2d(b, k)
    s_aa = F.conv2d(a * a, k) - mu_a ** 2; s_bb = F.conv2d(b * b, k) - mu_b ** 2; s_ab = F.conv2d(a * b, k) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return m.mean(dim=(1, 2, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--out", required=True); ap.add_argument("--fps", type=int, default=16)
    args = ap.parse_args()
    va, vb = np.load(args.a), np.load(args.b)
    assert va.shape == vb.shape, (va.shape, vb.shape)
    fa, fb = torch.from_numpy(va).float() / 255.0, torch.from_numpy(vb).float() / 255.0
    mse = ((fa - fb) ** 2).mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-12))
    s = ssim(luma(va), luma(vb))
    mad = (fa - fb).abs().mean(dim=(1, 2, 3)) * 255
    rec = dict(a=args.a, b=args.b, frames=int(va.shape[0]), resolution=f"{va.shape[2]}x{va.shape[1]}",
               psnr_mean=round(float(psnr[torch.isfinite(psnr)].mean()), 2), psnr_min=round(float(psnr.min()), 2),
               ssim_mean=round(float(s.mean()), 4), ssim_min=round(float(s.min()), 4),
               mean_abs_diff_0_255=round(float(mad.mean()), 2),
               per_frame=dict(psnr=[round(float(x), 2) for x in psnr], ssim=[round(float(x), 4) for x in s]))
    json.dump(rec, open(f"{args.out}_compare.json", "w"), indent=1)
    diff = np.clip(np.abs(va.astype(np.int16) - vb.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
    side = np.concatenate([va, vb, diff], axis=2)
    f, h, w, _ = side.shape
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                          f"{args.out}_side_by_side.mp4"], stdin=subprocess.PIPE)
    p.stdin.write(side.tobytes()); p.stdin.close(); p.wait()
    print(json.dumps({k: v for k, v in rec.items() if k != "per_frame"}))


if __name__ == "__main__":
    main()
