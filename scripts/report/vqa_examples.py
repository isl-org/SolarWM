"""Tile the full 153-frame example videos of vqa_generate.py into one labelled grid per sample.

Writes outputs/report_chain/vqa_153f/examples/grid_<sample>.mp4: ground truth and the 12
configurations, 4 x 4 tiles at half resolution, 16 FPS.

    python scripts/report/vqa_examples.py
"""

import json
import subprocess

from vqa_generate import CONFIGS, OUT, _safe_key

LABELS = ["ground_truth"] + [name for name, _ in CONFIGS]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COLS, TILE_W, TILE_H = 4, 432, 240


if __name__ == "__main__":
    for sample in json.loads((OUT / "examples.json").read_text())["samples"]:
        safe = _safe_key(sample["key"])
        inputs, filters = [], []
        for i, label in enumerate(LABELS):
            inputs += ["-i", str(OUT / "examples" / f"{safe}_{label}.mp4")]
            filters.append(f"[{i}:v]scale={TILE_W}:{TILE_H},drawtext=fontfile={FONT}:text='{label}':x=8:y=8:fontsize=20:"
                           f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4[t{i}]")
        layout = "|".join(f"{(i % COLS) * TILE_W}_{(i // COLS) * TILE_H}" for i in range(len(LABELS)))
        rows = -(-len(LABELS) // COLS)
        filters.append("".join(f"[t{i}]" for i in range(len(LABELS)))
                       + f"xstack=inputs={len(LABELS)}:layout={layout}:fill=black,pad={COLS * TILE_W}:{rows * TILE_H}[out]")
        out = OUT / "examples" / f"grid_{sample['source']}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "16", str(out)], check=True)
        print(out)
