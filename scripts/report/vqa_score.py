"""Score generated clips with the released DOVER and DOVER++ evaluators."""

from __future__ import annotations

import argparse
import csv
import json
import random
import zlib
from collections import defaultdict
from pathlib import Path

import decord
import numpy as np
import torch
from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from dover.models import DOVER


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/report_chain/vqa_153f"
DOVER_ROOT = Path("/home/ssheorey/code/DOVER")
MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
STD = torch.FloatTensor([58.395, 57.12, 57.375])

SAMPLE_TYPES = {
    "technical": {
        "fragments_h": 7,
        "fragments_w": 7,
        "fsize_h": 32,
        "fsize_w": 32,
        "aligned": 32,
        "clip_len": 32,
        "frame_interval": 2,
        "num_clips": 3,
    },
    "aesthetic": {
        "size_h": 224,
        "size_w": 224,
        "clip_len": 32,
        "frame_interval": 2,
        "t_frag": 32,
        "num_clips": 1,
    },
}

MODEL_CONFIG = {
    "backbone": {
        "technical": {"type": "swin_tiny_grpb"},
        "aesthetic": {"type": "conv_tiny"},
    },
    "backbone_preserve_keys": "technical,aesthetic",
    "divide_head": True,
    "vqa_head": {"in_channels": 768, "hidden_channels": 64},
}


def _rescale(technical: float, aesthetic: float) -> tuple[float, float, float]:
    t, a = (technical - 0.1107) / 0.07355, (aesthetic + 0.08285) / 0.03774
    return tuple(float(1 / (1 + np.exp(-x))) for x in (t * 0.6104 + a * 0.3896, a, t))


def _load_model(name: str, device: torch.device):
    model = DOVER(**MODEL_CONFIG)
    path = DOVER_ROOT / "pretrained_weights" / (
        "DOVER_plus_plus.pth" if name == "dover++" else "DOVER.pth"
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if name == "dover++" else checkpoint
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _views(path: Path, seed: int) -> dict[str, torch.Tensor]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    decord.bridge.set_bridge("torch")
    samplers = {}
    for name, options in SAMPLE_TYPES.items():
        if "t_frag" not in options:
            samplers[name] = UnifiedFrameSampler(
                options["clip_len"], options["num_clips"], options["frame_interval"]
            )
        else:
            samplers[name] = UnifiedFrameSampler(
                options["clip_len"] // options["t_frag"],
                options["t_frag"],
                options["frame_interval"],
                options["num_clips"],
            )
    views, _ = spatial_temporal_view_decomposition(str(path), SAMPLE_TYPES, samplers)
    for name, view in views.items():
        num_clips = SAMPLE_TYPES[name].get("num_clips", 1)
        views[name] = (
            ((view.permute(1, 2, 3, 0) - MEAN) / STD)
            .permute(3, 0, 1, 2)
            .reshape(view.shape[0], num_clips, -1, *view.shape[2:])
            .transpose(0, 1)
        )
    return views


def _score(model, path: Path, device: torch.device, seed: int) -> tuple[float, float, float]:
    views = {key: value.to(device) for key, value in _views(path, seed).items()}
    scores = [score.mean().item() for score in model(views)]
    technical, aesthetic = scores
    return _rescale(technical, aesthetic)


def _clips() -> list[dict]:
    samples_path = OUT / "samples.json"
    samples = {row["key"].replace("/", "__"): row for row in json.loads(samples_path.read_text())} if samples_path.exists() else {}
    rows = []
    for config_dir in sorted(OUT.iterdir()):
        if not config_dir.is_dir() or config_dir.name in {"examples", "ground_truth"}:
            continue
        for path in sorted(config_dir.glob("*_first.mp4")):
            key = path.name[: -len("_first.mp4")]
            last = config_dir / f"{key}_last.mp4"
            if not last.exists():
                continue
            rows.extend(
                [
                    {
                        "config": config_dir.name,
                        "key": key,
                        "source": samples.get(key, {}).get("source", "unknown"),
                        "split": "first",
                        "path": path,
                    },
                    {
                        "config": config_dir.name,
                        "key": key,
                        "source": samples.get(key, {}).get("source", "unknown"),
                        "split": "last",
                        "path": last,
                    },
                ]
            )
    for path in sorted((OUT / "ground_truth").glob("*_first.mp4")):
        key = path.name[: -len("_first.mp4")]
        last = OUT / "ground_truth" / f"{key}_last.mp4"
        if last.exists():
            rows.extend(
                [
                    {"config": "ground_truth", "key": key, "source": "ground_truth", "split": "first", "path": path},
                    {"config": "ground_truth", "key": key, "source": "ground_truth", "split": "last", "path": last},
                ]
            )
    return rows


def _bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci95": _bootstrap_ci(array, 42),
    }


def _aggregate(rows: list[dict]) -> dict:
    scores = defaultdict(dict)
    for row in rows:
        scores[(row["model"], row["config"], row["key"], row["split"])] = row

    aggregate = {}
    for model in sorted({row["model"] for row in rows}):
        configs = sorted({row["config"] for row in rows if row["model"] == model})
        aggregate[model] = {}
        for config in configs:
            first = {
                row["key"]: row
                for row in rows
                if row["model"] == model and row["config"] == config and row["split"] == "first"
            }
            last = {
                row["key"]: row
                for row in rows
                if row["model"] == model and row["config"] == config and row["split"] == "last"
            }
            generated = {key for (m, c, key, split) in scores if m == model and c == "bf16"}
            shared = sorted(set(first) & set(last) & generated)
            entry = {"n": len(shared)}
            for metric in ("overall", "aesthetic", "technical"):
                entry[metric] = {
                    "first": _summary([first[key][metric] for key in shared]),
                    "last": _summary([last[key][metric] for key in shared]),
                    "last_minus_first": _summary([last[key][metric] - first[key][metric] for key in shared]),
                }
                for split, values in (("first", first), ("last", last)):
                    differences = np.asarray([values[key][metric] - scores[(model, "bf16", key, split)][metric] for key in shared])
                    entry[metric][f"{split}_vs_bf16"] = {
                        "mean": float(differences.mean()),
                        "ci95": _bootstrap_ci(differences, 123),
                        "lower": int((differences < 0).sum()),
                        "lower_by_0.1": int((differences < -0.1).sum()),
                    }
            aggregate[model][config] = entry

    source_breakdown = {}
    for config in ("bf16", "svdq_w8a8"):
        matching = [row for row in rows if row["model"] == "dover++" and row["config"] == config]
        source_breakdown[config] = {}
        for source in sorted({row["source"] for row in matching}):
            source_rows = [row for row in matching if row["source"] == source]
            source_breakdown[config][source] = {"n": sum(row["split"] == "first" for row in source_rows)} | {
                metric: {split: float(np.mean([row[metric] for row in source_rows if row["split"] == split])) for split in ("first", "last")}
                for metric in ("overall", "aesthetic", "technical")
            }
    return {"aggregate": aggregate, "source_breakdown": source_breakdown}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--models", default="dover++,dover")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device or ("xpu" if torch.xpu.is_available() else "cpu"))
    clips = _clips()
    expected = 10 * 2 * len(list((OUT / "bf16").glob("*_first.mp4")))
    available = len([row for row in clips if row["config"] not in {"ground_truth"}])
    if not args.allow_incomplete and available != expected:
        raise RuntimeError(f"expected {expected} generated clips, found {available}")

    rows = []
    for model_name in args.models.split(","):
        model = _load_model(model_name, device)
        for number, clip in enumerate(clips, 1):
            overall, aesthetic, technical = _score(model, clip["path"], device, zlib.crc32(f"{clip['key']}_{clip['split']}".encode()))
            rows.append(
                {
                    "model": model_name,
                    "config": clip["config"],
                    "key": clip["key"],
                    "source": clip["source"],
                    "split": clip["split"],
                    "overall": overall,
                    "aesthetic": aesthetic,
                    "technical": technical,
                    "path": str(clip["path"]),
                }
            )
            print(f"[{model_name}] {number}/{len(clips)} {clip['config']} {clip['key']} {clip['split']}: {overall:.5f}", flush=True)
        del model
        if device.type == "xpu":
            torch.xpu.empty_cache()

    with (OUT / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "protocol": {
            "clip_frames": {"first": [0, 8], "last": [141, 152]},
            "short_clip_sampling": "official DOVER UnifiedFrameSampler; indices wrap modulo clip length, repeating frames for the 9/12-frame clips",
            "encoding": "ffmpeg libx264 -qp 0 -pix_fmt yuv444p at 16 FPS",
            "bootstrap_seed": 123,
        },
        "clips_scored": len(rows),
        "missing_generated_clips": expected - available,
        "rows": rows,
        **_aggregate(rows),
    }
    (OUT / "scores.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
