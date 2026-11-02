"""Shared load-time JIT compiler for SYCL kernel extensions."""

from __future__ import annotations

try:
    import fcntl
except ImportError:
    fcntl = None
import hashlib
import importlib
import importlib.machinery
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import warnings
from pathlib import Path
from typing import Any, Sequence

import torch
from solarwm.kernels.sycl_env import get_sycl_env

_JIT_CACHE_DIR = Path.home() / ".cache" / "solarwm" / "jit"


def get_jit_cache_dir() -> Path:
    _JIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _JIT_CACHE_DIR


def compute_jit_key(
    src_paths: Sequence[Path],
    extra_flags: Sequence[str] = (),
    gpu_arch: str = "bmg-g31",
) -> str:
    """Compute a deterministic hash for the build artifacts."""
    hasher = hashlib.sha256()

    # PyTorch and Python versions
    hasher.update(torch.__version__.encode("utf-8"))
    hasher.update(sys.version.encode("utf-8"))
    hasher.update(gpu_arch.encode("utf-8"))

    # Toolchain info
    env = get_sycl_env()
    hasher.update(str(env.icpx).encode("utf-8"))
    try:
        ver_output = subprocess.check_output([str(env.icpx), "--version"], stderr=subprocess.STDOUT)
        hasher.update(ver_output)
    except Exception:
        pass

    # Source files contents
    for p in sorted(src_paths, key=lambda x: str(x)):
        p = Path(p).resolve()
        hasher.update(p.name.encode("utf-8"))
        if p.is_file():
            hasher.update(p.read_bytes())

    for f in extra_flags:
        hasher.update(f.encode("utf-8"))

    return hasher.hexdigest()[:24]


def compile_sycl_extension(
    extension_name: str,
    sources: Sequence[str | Path],
    extra_cflags: Sequence[str] = (),
    extra_ldflags: Sequence[str] = (),
    gpu_arch: str = "bmg-g31",
    include_sycl_tla: bool = True,
    timeout: int = 300,
) -> Path:
    """Compile SYCL source files into a shared library under cache directory.

    Guarded by a file lock. Returns Path to the compiled .so.
    """
    cache_dir = get_jit_cache_dir()
    src_paths = [Path(s).resolve() for s in sources]
    for s in src_paths:
        if not s.is_file():
            raise FileNotFoundError(f"Source file not found: {s}")

    key = compute_jit_key(src_paths, [*extra_cflags, *extra_ldflags], gpu_arch=gpu_arch)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if os.name == "nt" else ".so")
    module_filename = f"{extension_name}_{key}{ext_suffix}"
    final_so = cache_dir / module_filename
    lock_file = cache_dir / f"{extension_name}_{key}.lock"

    if final_so.is_file():
        return final_so

    env = get_sycl_env()

    with open(lock_file, "w") as lock_fd:
        if fcntl is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            if final_so.is_file():
                return final_so

            with tempfile.TemporaryDirectory(prefix=f"jit_{extension_name}_") as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                tmp_so = tmp_dir_path / module_filename

                cflags = [
                    "-fsycl",
                    "-fsycl-targets=spir64_gen",
                    "-fno-sycl-instrument-device-code",
                    "-std=c++20",
                    "-O3",
                    f"-DTORCH_EXTENSION_NAME={extension_name}",
                    "-DTORCH_API_INCLUDE_EXTENSION_H",
                    "-isystem", str(env.torch_inc),
                    "-isystem", str(env.torch_inc / "torch" / "csrc" / "api" / "include"),
                    "-isystem", str(env.py_inc),
                    "-Wno-deprecated-declarations",
                ]
                if os.name != "nt":
                    cflags.extend(["-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1"])

                if include_sycl_tla:
                    cflags.extend([
                        "-DCUTLASS_ENABLE_SYCL=1",
                        "-DSYCL_INTEL_TARGET=1",
                    ])
                    for inc in env.sycl_tla_includes:
                        cflags.extend(["-I", str(inc)])

                cflags.extend(extra_cflags)

                obj_files = []
                for i, src in enumerate(src_paths):
                    obj_path = tmp_dir_path / f"src_{i}.o"
                    cmd = [
                        str(env.icpx),
                        *cflags,
                        "-c", "-x", "c++", str(src),
                        "-o", str(obj_path),
                    ]
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                    if res.returncode != 0:
                        raise RuntimeError(
                            f"Compilation failed for {src}:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
                        )
                    obj_files.append(obj_path)

                spirv_ext = "+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"
                link_cmd = [
                    str(env.icpx),
                    "-fsycl",
                    "-fsycl-targets=spir64_gen",
                    "-fno-sycl-instrument-device-code",
                    "-shared",
                    *[str(o) for o in obj_files],
                    "-o", str(tmp_so),
                    "-Xsycl-target-backend=spir64_gen", f"-device {gpu_arch}",
                    "-Xspirv-translator", f"-spirv-ext={spirv_ext}",
                    f"-L{env.torch_lib}",
                    "-lc10", "-lc10_xpu", "-ltorch", "-ltorch_cpu", "-ltorch_xpu", "-ltorch_python",
                    *extra_ldflags,
                ]
                if os.name != "nt":
                    link_cmd.insert(10, f"-Wl,-rpath,{env.torch_lib}")
                else:
                    link_cmd.insert(1, f"-L{Path(sys.base_prefix) / 'libs'}")

                link_res = subprocess.run(link_cmd, capture_output=True, text=True, timeout=timeout)
                if link_res.returncode != 0:
                    raise RuntimeError(
                        f"Linking failed for {extension_name}:\nSTDOUT:\n{link_res.stdout}\nSTDERR:\n{link_res.stderr}"
                    )

                shutil.move(str(tmp_so), str(final_so))
                return final_so
        finally:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)


def load_sycl_extension(
    extension_name: str,
    sources: Sequence[str | Path],
    extra_cflags: Sequence[str] = (),
    extra_ldflags: Sequence[str] = (),
    gpu_arch: str = "bmg-g31",
    include_sycl_tla: bool = True,
) -> Any:
    """Compile and import a SYCL PyTorch extension module dynamically."""
    so_path = compile_sycl_extension(
        extension_name=extension_name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        gpu_arch=gpu_arch,
        include_sycl_tla=include_sycl_tla,
    )

    spec = importlib.util.spec_from_file_location(extension_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec to import {so_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[extension_name] = mod
    spec.loader.exec_module(mod)
    return mod
