"""SYCL build environment and sycl-tla tree discovery."""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path
from typing import NamedTuple

import torch

SYCL_TLA_SENTINEL_HEADERS = (
    "cute/tensor.hpp",
    "flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp",
    "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp",
)


class SyclEnv(NamedTuple):
    icpx: Path
    torch_dir: Path
    torch_inc: Path
    torch_lib: Path
    py_inc: Path
    sycl_tla_dir: Path
    sycl_tla_includes: list[Path]


def find_icpx() -> Path | None:
    """Locate Intel oneAPI icpx/icx compiler."""
    # Check PATH first
    compiler = shutil.which("icpx") or shutil.which("icx")
    if compiler:
        return Path(compiler).resolve()

    # Check ONEAPI_ROOT or standard /opt/intel/oneapi paths
    candidates = []
    if "ONEAPI_ROOT" in os.environ:
        candidates.append(Path(os.environ["ONEAPI_ROOT"]) / "compiler" / "latest" / "bin" / "icpx")
    candidates.extend([
        Path("/opt/intel/oneapi/compiler/latest/bin/icpx"),
        Path("/opt/intel/oneapi/compiler/2025.3/bin/icpx"),
        Path("/opt/intel/oneapi/compiler/2025.0/bin/icpx"),
        Path("/opt/intel/oneapi/compiler/2024.2/bin/icpx"),
    ])
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c.resolve()
    return None


def _check_sycl_tla_dir(base: Path) -> list[Path] | None:
    """Return [include, applications, tools/util/include] if sentinels exist."""
    include_dir = base / "include"
    apps_dir = base / "applications"
    tools_dir = base / "tools" / "util" / "include"

    # cute/tensor.hpp must be under include/
    if not (include_dir / "cute" / "tensor.hpp").is_file():
        return None
    # xe FMHA headers must be under applications/
    if not (apps_dir / "flash_attention_v2" / "collective" / "xe_fmha_fwd_mainloop.hpp").is_file():
        return None
    if not (apps_dir / "flash_attention_v2" / "collective" / "xe_fmha_fwd_epilogue.hpp").is_file():
        return None

    paths = [include_dir]
    if apps_dir.is_dir():
        paths.append(apps_dir)
    if tools_dir.is_dir():
        paths.append(tools_dir)
    return paths


def find_sycl_tla() -> tuple[Path, list[Path]]:
    """Discover a valid sycl-tla / CUTLASS directory tree.

    Search order:
    1. SOLARWM_SYCL_TLA_DIR or SYCL_TLA_DIR environment variables.
    2. Probing PyTorch include directories.
    3. User cache directory (~/.cache/solarwm/sycl-tla or ~/.cache/sycl-tla*).
    4. Bundled package directory (SolarWM/packages/wan22_svdquant_dense_block/sycl-tla).
    """
    env_dir = os.environ.get("SOLARWM_SYCL_TLA_DIR") or os.environ.get("SYCL_TLA_DIR")
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        incs = _check_sycl_tla_dir(p)
        if incs is not None:
            return p, incs

    # Probe PyTorch includes
    torch_dir = Path(torch.__file__).parent
    torch_inc = torch_dir / "include"
    incs = _check_sycl_tla_dir(torch_inc)
    if incs is not None:
        return torch_inc, incs

    # User cache candidates
    cache_dirs = [
        Path.home() / ".cache" / "solarwm" / "sycl-tla",
        Path.home() / ".cache" / "sycl-tla-v0.8-pin",
        Path.home() / ".cache" / "sycl-tla-v0.9.3",
    ]
    for c in cache_dirs:
        if c.is_dir():
            incs = _check_sycl_tla_dir(c)
            if incs is not None:
                return c.resolve(), incs

    # Bundled repo tree
    repo_root = Path(__file__).resolve().parents[3]
    bundled = repo_root / "packages" / "wan22_svdquant_dense_block" / "sycl-tla"
    if bundled.is_dir():
        incs = _check_sycl_tla_dir(bundled)
        if incs is not None:
            return bundled.resolve(), incs

    # If none found, return fallback path under cache
    fallback = Path.home() / ".cache" / "solarwm" / "sycl-tla"
    fallback_incs = [fallback / "include", fallback / "applications", fallback / "tools" / "util" / "include"]
    return fallback, fallback_incs


def get_sycl_env() -> SyclEnv:
    """Gather full environment needed to build SYCL extensions."""
    icpx = find_icpx()
    if icpx is None:
        raise RuntimeError("Intel oneAPI icpx/icx compiler not found. Please source setvars.sh.")

    torch_dir = Path(torch.__file__).parent
    torch_inc = torch_dir / "include"
    torch_lib = torch_dir / "lib"
    py_inc = Path(sysconfig.get_path("include"))

    sycl_tla_dir, sycl_tla_incs = find_sycl_tla()

    return SyclEnv(
        icpx=icpx,
        torch_dir=torch_dir,
        torch_inc=torch_inc,
        torch_lib=torch_lib,
        py_inc=py_inc,
        sycl_tla_dir=sycl_tla_dir,
        sycl_tla_includes=sycl_tla_incs,
    )
