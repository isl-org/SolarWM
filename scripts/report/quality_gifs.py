"""Animated first-chunk | last-chunk GIFs of each quality.py configuration, for the summary deck.

Reads the 153-frame videos saved by quality.py and writes one looping GIF per configuration to
outputs/report_slides/quality_gifs/<name>.gif at the model's 16 FPS. The first chunk has 9 frames
and the last chunk 12, so the first chunk holds its last frame for the remaining 3.

    python scripts/report/quality_gifs.py
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from quality import CHUNKS, CONFIGS, FRAMES_PER_CHUNK, OUT as VIDEOS

OUT = Path(__file__).resolve().parents[2] / "outputs/report_slides/quality_gifs"
SIZE = (180, 324)
GAP = 6
FRAME_MS = 60


def _frames(video: torch.Tensor) -> torch.Tensor:
    video = F.interpolate(video[0].float(), size=SIZE, mode="area")
    return ((video.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in [c[0] for c in CONFIGS if c[0] != "bf16_repeat"]:
        video = torch.load(VIDEOS / f"{name}.pt")
        first = video.shape[1] - FRAMES_PER_CHUNK * (CHUNKS - 1)
        a, b = _frames(video[:, :first]), _frames(video[:, -FRAMES_PER_CHUNK:])
        images = []
        for i in range(FRAMES_PER_CHUNK):
            pair = torch.zeros(SIZE[0], 2 * SIZE[1] + GAP, 3, dtype=torch.uint8)
            pair[:, :SIZE[1]] = a[min(i, first - 1)]
            pair[:, SIZE[1] + GAP:] = b[i]
            images.append(Image.fromarray(pair.numpy()).quantize(256, dither=Image.Dither.NONE))
        images[0].save(OUT / f"{name}.gif", save_all=True, append_images=images[1:], duration=FRAME_MS, loop=0)
        print(OUT / f"{name}.gif")
