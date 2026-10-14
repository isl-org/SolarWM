"""Fast QuaRot INT8 oneDNN 3D Convolution kernel on Intel XPU.

Provides `fast_qconv3d`, bypassing `aten::item()` host synchronizations
in PyTorch XPU's native `onednn::qconv_pointwise.tensor` binding.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import Sequence

import torch

_SO_NAME = "qconv_onednn_ops.so"
_MOD_NAME = "qconv_onednn_ops"
_MODULE = None


def _build_extension(so_path: Path) -> bool:
    """Compile the SYCL oneDNN extension using icpx if available."""
    icpx = shutil.which("icpx")
    if not icpx:
        return False

    src_file = Path(__file__).parent / "qconv_onednn.cpp"
    if not src_file.exists():
        return False

    torch_dir = Path(torch.__file__).parent
    torch_inc = torch_dir / "include"
    torch_lib = torch_dir / "lib"
    py_inc = sysconfig.get_path("include")

    dnnl_root = Path(os.environ.get("DNNLROOT", "/opt/intel/oneapi/dnnl/latest"))
    dnnl_inc = dnnl_root / "include"
    dnnl_lib = dnnl_root / "lib"

    if not dnnl_lib.exists():
        # Fall back to system or torch-bundled paths if any
        dnnl_lib = torch_lib

    cmd = [
        icpx,
        "-shared",
        "-fPIC",
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
        f"-Wl,-rpath,{torch_lib}",
        f"-Wl,-rpath,{dnnl_lib}",
        str(src_file),
        "-o",
        str(so_path),
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return res.returncode == 0
    except Exception:
        return False


def _get_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    so_path = Path(__file__).parent / _SO_NAME
    if not so_path.exists():
        built = _build_extension(so_path)
        if not built or not so_path.exists():
            return None

    try:
        spec = importlib.util.spec_from_file_location(_MOD_NAME, str(so_path))
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
