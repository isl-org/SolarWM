"""Create a separate, verified EMA-only inference transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import warnings

import torch

from solarwm.checkpoint import verify_checkpoint
from solarwm.runtime.serialization import canonical_json_bytes


def prepare(source: Path, target: Path) -> Path:
    verified = verify_checkpoint(source)
    if target.exists():
        result = verify_checkpoint(target)
        if (
            result.metadata.get("source_manifest_digest") != verified.manifest_digest
            or result.metadata.get("inference_dtype") != "bfloat16"
        ):
            raise ValueError("Existing EMA checkpoint was derived from another source")
        return target
    release = json.loads((source / "release-manifest.json").read_text(encoding="utf-8"))
    if release["load"]["default_weights"] != "ema":
        raise ValueError("Source release does not select EMA by default")
    temporary = target.with_name(target.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Unfinished checkpoint processing exists: {temporary}")
    temporary.mkdir(parents=True)
    original = torch.UntypedStorage.from_file
    mapped = None
    # Windows copy-on-write mappings reserve commit for the entire checkpoint.
    # A read-only mapping needs no such reservation; never mutate its tensors.
    with (source / "model.pt").open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            backing = torch.frombuffer(mapped, dtype=torch.uint8).untyped_storage()
        try:
            torch.UntypedStorage.from_file = staticmethod(lambda *args, **kwargs: backing)
            payload = torch.load(
                source / "model.pt", map_location="cpu", weights_only=True, mmap=True,
            )
        finally:
            torch.UntypedStorage.from_file = original
        if not isinstance(payload.get("generator_ema"), dict) or not payload["generator_ema"]:
            raise ValueError("Checkpoint has no EMA tensor state")
        del payload["generator"]
        payload["generator_ema"] = {
            name: value.to(torch.bfloat16) if value.is_floating_point() else value.clone()
            for name, value in payload["generator_ema"].items()
        }
        print("Writing EMA-only checkpoint without loading both roles into RAM", flush=True)
        torch.save(payload, temporary / "model.pt")
        del payload, backing
        mapped.close()
    size = (temporary / "model.pt").stat().st_size
    manifest = json.loads((source / "checkpoint-manifest.json").read_bytes())
    manifest["files"] = [{"path": "model.pt", "size": size}]
    manifest["metadata"].update(
        available_weights=["ema"], source_manifest_digest=verified.manifest_digest,
        inference_dtype="bfloat16",
    )
    data = canonical_json_bytes(manifest)
    (temporary / "checkpoint-manifest.json").write_bytes(data)
    (temporary / "COMPLETE.json").write_bytes(canonical_json_bytes({
        "schema": "solarwm.checkpoint-complete.v2",
        "step": verified.step,
        "manifest_digest": hashlib.blake2s(data).hexdigest(),
    }))
    release["identity"]["model"]["weight_role"] = "ema"
    release["files"] = [
        {"path": name, "size": (temporary / name).stat().st_size}
        for name in ("model.pt", "checkpoint-manifest.json", "COMPLETE.json")
    ]
    (temporary / "release-manifest.json").write_bytes(canonical_json_bytes(release))
    for name in ("LICENSE.txt", "NOTICE.md"):
        if (source / name).is_file():
            shutil.copy2(source / name, temporary / name)
    verify_checkpoint(temporary)
    os.replace(temporary, target)
    print(f"EMA-only checkpoint: {target} ({size / 1024**3:.2f} GiB)", flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    arguments = parser.parse_args()
    prepare(arguments.source, arguments.target)
