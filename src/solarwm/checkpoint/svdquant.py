"""Resolve and validate canonical or GitHub release-named SVDQuant assets."""

import json
from pathlib import Path


def resolve_manifest(directory: Path) -> Path:
    canonical = directory / "manifest.json"
    if canonical.is_file():
        return canonical
    prefixed = directory / f"{directory.name}_manifest.json"
    if prefixed.is_file():
        return prefixed
    raise FileNotFoundError(f"SVDQuant manifest missing in {directory}")


def resolve_shard(directory: Path, name: str) -> Path:
    if "/" in name or "\\" in name or Path(name).name != name:
        raise ValueError(f"Unsafe SVDQuant shard name: {name}")
    for candidate in (directory / name, directory / f"{directory.name}_{name}"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"SVDQuant shard missing: {directory / name}")


def validate_checkpoint(directory: Path) -> tuple[dict, list[Path]]:
    from safetensors import safe_open

    manifest = json.loads(resolve_manifest(directory).read_text(encoding="utf-8"))
    if manifest.get("format_version") != "solarwm-svdquant-v1":
        raise ValueError("Unsupported SVDQuant manifest format")
    shards = manifest.get("shards")
    weight_map = manifest.get("weight_map")
    if not isinstance(shards, list) or not shards or len(set(shards)) != len(shards):
        raise ValueError("SVDQuant manifest needs unique declared shards")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("SVDQuant manifest needs a nonempty weight_map")
    files = [resolve_shard(directory, name) for name in shards]
    actual = {}
    for name, path in zip(shards, files, strict=True):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in actual:
                    raise ValueError(f"Duplicate SVDQuant tensor: {key}")
                actual[key] = name
    if actual != weight_map:
        missing = sorted(set(weight_map) - set(actual))
        extra = sorted(set(actual) - set(weight_map))
        misplaced = sorted(key for key in actual.keys() & weight_map.keys()
                           if actual[key] != weight_map[key])
        raise ValueError(
            f"SVDQuant inventory differs: missing={missing[:5]}, "
            f"unexpected={extra[:5]}, wrong_shard={misplaced[:5]}"
        )
    sets = set(manifest.get("set_mapping", {}).values())
    if not sets or len(sets) != manifest.get("num_sets"):
        raise ValueError("SVDQuant set_mapping does not match num_sets")
    if {key.split(".", 1)[0] for key in actual} != sets:
        raise ValueError("SVDQuant tensor sets do not match set_mapping")
    return manifest, files
