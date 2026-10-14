"""Dense block driver for VTune: warm up paused, then collect only ITERS block calls.

Run under `vtune -collect gpu-hotspots -start-paused ...`.
"""

import importlib
import os
import sys
from pathlib import Path

import ittapi
import torch

PKG = Path(__file__).resolve().parents[2] / "packages" / "wan22_svdquant_dense_block"
sys.path.insert(0, str(PKG))
import task  # noqa: E402

km = importlib.import_module(task.extension_name(task.load_hparams()))
data = task.block_data.__wrapped__(torch.device("xpu"))
block = task.make_sycl_dense_block_fn(km)
args = data["sycl_args"]
iters = int(os.environ.get("ITERS", "20"))

if os.environ.get("LAYER") == "down":
    # Profile only the FFN down GEMM (it shares a kernel name with Q/K/V), fed by FFN up.
    up, down = data["ffn_up_p"], data["ffn_down_p"]
    x_q, x_scales, x_lora = km.quantize_activation_int4(data["x"], up["lora_a"])
    h, h_q, h_scales = task.execute_sycl_layer(
        km, x_q, up["w_dpas_b"], up["w_scale_dpas"], bias=up["bias"], lora_b=up["lora_b"],
        activation="gelu_tanh", x_scales=x_scales, x_lora=x_lora, fuse_quant_out=True)

    def block(*_):
        task.execute_sycl_layer(
            km, h_q, down["w_dpas_b"], down["w_scale_dpas"], bias=down["bias"], lora_a=down["lora_a"],
            lora_b=down["lora_b"], x_scales=h_scales, x_bf16=h, activation="none")

for _ in range(20):
    block(*args)
torch.xpu.synchronize()

ittapi.resume()
for _ in range(iters):
    block(*args)
torch.xpu.synchronize()
ittapi.pause()
