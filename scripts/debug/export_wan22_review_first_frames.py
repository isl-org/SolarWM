"""Export the first frame of each Wan2.2 review MP4 as a WebP asset."""

from __future__ import annotations

import argparse
from pathlib import Path

import av
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_DIR = REPO_ROOT / "outputs" / "wan22_5b_vae_review"
DEFAULT_ASSET_DIR = REPO_ROOT / "assets"


def export_first_frame(video_path: Path, asset_root: Path, overwrite: bool) -> Path:
    stem = video_path.stem
    dataset = stem.split("__", 1)[0]
    output_path = asset_root / dataset / f"{stem}.webp"
    if output_path.exists() and not overwrite:
        return output_path

    with av.open(str(video_path), mode="r") as container:
        frame = next(container.decode(video=0), None)
    if frame is None:
        raise RuntimeError(f"video has no frames: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = frame.to_image().convert("RGB")
    image.save(output_path, format="WEBP", quality=95, method=6)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    videos = sorted(args.video_dir.glob("*.mp4"))
    if not videos:
        parser.error(f"no MP4 files found in {args.video_dir}")

    for number, video_path in enumerate(videos, start=1):
        output_path = export_first_frame(video_path, args.asset_dir, args.overwrite)
        print(f"[{number}/{len(videos)}] {output_path}")


if __name__ == "__main__":
    main()
