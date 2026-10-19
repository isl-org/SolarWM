"""Fast QuaRot INT8 oneDNN 3D Convolution kernel on Intel XPU.

Provides `fast_qconv3d`, bypassing `aten::item()` host synchronizations
in PyTorch XPU's native `onednn::qconv_pointwise.tensor` binding.
"""

from __future__ import annotations

from importlib.machinery import EXTENSION_SUFFIXES
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Sequence

import torch

_MOD_NAME = "qconv_onednn_ops"
_MODULE = None
_DLL_DIRECTORY = None


def _module_suffix() -> str:
    suffix = next(
        (
            value
            for value in EXTENSION_SUFFIXES
            if value.endswith(".pyd" if os.name == "nt" else ".so")
        ),
        None,
    )
    return suffix or (".pyd" if os.name == "nt" else ".so")


def _module_path() -> Path:
    package_dir = Path(__file__).parent
    suffixes = [
        value
        for value in EXTENSION_SUFFIXES
        if value.endswith(".pyd" if os.name == "nt" else ".so")
    ]
    suffixes.append(_module_suffix())
    for suffix in dict.fromkeys(suffixes):
        candidate = package_dir / f"{_MOD_NAME}{suffix}"
        if candidate.exists():
            return candidate
    return package_dir / f"{_MOD_NAME}{_module_suffix()}"


def _dnnl_root() -> Path:
    configured = os.environ.get("DNNLROOT")
    if configured:
        return Path(configured)
    if os.name == "nt":
        roots = (
            os.environ.get("ProgramW6432"),
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
        )
        candidates = [
            Path(root) / "Intel" / "oneAPI" / "dnnl" / "latest"
            for root in roots
            if root
        ]
        return next((path for path in candidates if path.exists()), candidates[0])
    return Path("/opt/intel/oneapi/dnnl/latest")


def _build_extension(module_path: Path) -> bool:
    """Compile the SYCL oneDNN extension using icpx if available."""
    icpx = shutil.which("icpx") or shutil.which("icx")
    if not icpx:
        return False

    src_file = Path(__file__).parent / "qconv_onednn.cpp"
    if not src_file.exists():
        return False

    torch_dir = Path(torch.__file__).parent
    torch_inc = torch_dir / "include"
    torch_lib = torch_dir / "lib"
    py_inc = sysconfig.get_path("include")

    dnnl_root = _dnnl_root()
    dnnl_inc = dnnl_root / "include"
    dnnl_lib = dnnl_root / "lib"

    if not dnnl_lib.exists():
        # Fall back to system or torch-bundled paths if any
        dnnl_lib = torch_lib

    cmd = [
        icpx,
        "-shared",
        "-O3",
        "-std=c++17",
        "-fsycl",
        f"-DTORCH_EXTENSION_NAME={_MOD_NAME}",
        f"-I{py_inc}",
        f"-I{torch_inc}",
        f"-I{torch_inc / 'torch' / 'csrc' / 'api' / 'include'}",
        f"-I{dnnl_inc}",
        f"-L{torch_lib}",
        f"-L{dnnl_lib}",
        "-lc10",
        "-ltorch_cpu",
        "-ltorch",
        "-ltorch_python",
        "-lc10_xpu",
        "-ltorch_xpu",
        "-ldnnl",
        str(src_file),
        "-o",
        str(module_path),
    ]
    if os.name != "nt":
        cmd[2:2] = ["-fPIC"]
        cmd[-4:-4] = [f"-Wl,-rpath,{torch_lib}", f"-Wl,-rpath,{dnnl_lib}"]
    else:
        cmd.insert(1, f"-L{Path(sys.base_prefix) / 'libs'}")

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return res.returncode == 0
    except Exception:
        return False


def _get_module():
    global _MODULE, _DLL_DIRECTORY
    if _MODULE is not None:
        return _MODULE

    module_path = _module_path()
    if not module_path.exists():
        built = _build_extension(module_path)
        if not built or not module_path.exists():
            return None

    try:
        if os.name == "nt" and _DLL_DIRECTORY is None:
            dll_dir = _dnnl_root() / "bin"
            if dll_dir.is_dir():
                _DLL_DIRECTORY = os.add_dll_directory(str(dll_dir))
        spec = importlib.util.spec_from_file_location(_MOD_NAME, str(module_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULE = mod
        return _MODULE
    except Exception:
        return None


def is_available() -> bool:
    """Return True if fast_qconv3d SYCL extension is compiled and ready."""
    return _get_module() is not None


def fast_qconv3d(
    qx: torch.Tensor,
    x_scale: torch.Tensor,
    qw: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: Sequence[int] = (1, 1, 1),
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    groups: int = 1,
    output_dtype: torch.dtype = torch.bfloat16,
    post_op: str = "none",
    accumulate_into: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute 3D quantized convolution on Intel XPU without aten::item() synchronization.

    Parameters:
        qx: INT8 input tensor of shape [N, C_in, D_in, H_in, W_in], preferably channels_last_3d.
        x_scale: FP32 1-element scale tensor on XPU.
        qw: INT8 weight tensor of shape [C_out, C_in, kD, kH, kW].
        w_scale: FP32 scale tensor of shape [C_out] on XPU.
        bias: Optional FP32 or BF16 bias tensor of shape [C_out] on XPU.
        stride: Stride tuple [sD, sH, sW].
        padding: Padding tuple [pD, pH, pW] (or [pD_l, pH_l, pW_l, pD_r, pH_r, pW_r]).
        dilation: Dilation tuple [dD, dH, dW].
        groups: Number of convolution groups.
        output_dtype: Destination tensor dtype (torch.bfloat16 or torch.float32).
        post_op: Elementwise activation attribute ("none" or "relu").
        accumulate_into: Optional pre-existing destination tensor for in-place hardware accumulation (append_sum).
    """
    mod = _get_module()
    if mod is not None:
        return mod.fast_qconv3d(
            qx,
            x_scale,
            qw,
            w_scale,
            bias,
            list(stride),
            list(padding),
            list(dilation),
            groups,
            output_dtype,
            post_op,
            accumulate_into,
        )

    # Fallback to PyTorch native oneDNN (triggers aten::item())
    zero_x = torch.zeros(1, dtype=torch.int32, device=qx.device)
    zero_w = torch.zeros(qw.shape[0], dtype=torch.int32, device=qw.device)
    res = torch.ops.onednn.qconv_pointwise.tensor(
        qx,
        x_scale,
        zero_x,
        qw,
        w_scale,
        zero_w,
        bias,
        list(stride),
        list(padding[:3]),
        list(dilation),
        groups,
        1.0,
        0,
        output_dtype,
        post_op,
        [],
        "",
    )
    if accumulate_into is not None:
        accumulate_into.add_(res)
        return accumulate_into
    return res


def qconv_pointwise_tensor_fast(
    qx: torch.Tensor,
    x_scale: torch.Tensor,
    x_zero_point: torch.Tensor,
    qw: torch.Tensor,
    w_scale: torch.Tensor,
    w_zero_point: torch.Tensor | None,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    output_scale: float,
    output_zero_point: int,
    output_dtype: torch.dtype | None = None,
    attr: str = "none",
    scalars: Sequence[object] = (),
    algorithm: str | None = None,
    accumulate_into: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in fast replacement for `torch.ops.onednn.qconv_pointwise.tensor`."""
    return fast_qconv3d(
        qx=qx,
        x_scale=x_scale,
        qw=qw,
        w_scale=w_scale,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        output_dtype=output_dtype if output_dtype is not None else torch.bfloat16,
        post_op=attr,
        accumulate_into=accumulate_into,
    )


def clear_cache() -> None:
    """Clear primitive descriptor cache."""
    mod = _get_module()
    if mod is not None and hasattr(mod, "clear_cache"):
        mod.clear_cache()
