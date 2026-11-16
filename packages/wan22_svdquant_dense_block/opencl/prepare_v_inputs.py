"""Save the shared inputs and SYCL reference outputs to an .npz, once.

Workloads (exact task shapes, the same operation: bf16 x -> INT4 quantization -> LoRA-A -> INT4 GEMM -> LoRA-B -> bias):
  v     : x (M=1215) with the v projection weights (N = 3072, K = 3072)
  wide  : x (M=1215) with the ffn_up projection weights (N = 14336, K = 3072), plain epilogue (no GELU) for timing
  v_m1100, v_m900 : the v layer on the first 1100 / 900 rows (bucket test; 1100 shares a bucket with 1215)

Torch is used only here. Usage: python opencl/prepare_v_inputs.py [out.npz]
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import task  # noqa: E402


def as_u16(t):
    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def sycl_ref(mod, x, p):
    out = mod.svdquant_w4a4_gemm(x, p["w_dpas_b"], p["w_scale_dpas"], bias=p["bias"], lora_a=p["lora_a"], lora_b=p["lora_b"])
    torch.xpu.synchronize()
    return out


def main(out):
    mod = importlib.import_module(task.extension_name(task.load_hparams()))
    bd = task.block_data.__wrapped__(torch.device("xpu"))
    x = bd["x"]
    v, wide = bd["v_p"], bd["ffn_up_p"]
    arrays = dict(x=as_u16(x))
    for name, p in (("v", v), ("wide", wide)):
        arrays[f"{name}_lora_a"] = as_u16(p["lora_a"])
        arrays[f"{name}_lora_b"] = as_u16(p["lora_b"])
        arrays[f"{name}_bias"] = as_u16(p["bias"])
        arrays[f"{name}_w_dpas_b"] = p["w_dpas_b"].cpu().contiguous().numpy().view(np.uint32)
        arrays[f"{name}_w_scale"] = p["w_scale_dpas"].cpu().contiguous().numpy().astype(np.float32)
        arrays[f"{name}_ref"] = as_u16(sycl_ref(mod, x, p))
    for m in (1100, 900):
        arrays[f"v_m{m}_ref"] = as_u16(sycl_ref(mod, x[:m].contiguous(), v))
    np.savez(out, **arrays)
    print("wrote", out, {k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ocl_probe/v_inputs2.npz")
