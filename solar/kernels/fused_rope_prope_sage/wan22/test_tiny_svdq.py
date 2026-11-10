"""Self-test of the SVDQuant path on the tiny random Wan transformer from test_tiny.py (run that first):
calibration hooks -> conversion -> surgery (fused to_qkv, SVDQFFN) -> load -> forward vs the INT8 model."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wan_int8, wan_svdq, run_wan_i2v as R
root, dev = "/tmp/wan_tiny", torch.device("xpu")
m8 = wan_int8.load_expert_weights(wan_int8.build_expert(f"{root}/int8/transformer"), f"{root}/int8/transformer", dev)
calib = R.install_calibration([m8], ["transformer"])
torch.manual_seed(1)
x = torch.randn(1, 36, 9, 32, 64, device=dev, dtype=torch.bfloat16); t = torch.tensor([500.0], device=dev)
ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16)
with torch.no_grad():
    ref = m8(hidden_states=x, timestep=t, encoder_hidden_states=ctx, return_dict=False)[0].float()
print("calib keys", sorted(calib)[:3], len(calib))
wan_svdq.convert_expert(f"{root}/int8/transformer", calib, f"{root}/svdq/transformer", rank=32, alpha=0.5, prefix="transformer")
ms = wan_svdq.load_expert_weights(wan_svdq.build_expert(f"{root}/svdq/transformer"), f"{root}/svdq/transformer", dev)
print(type(ms.blocks[0].attn1.to_qkv).__name__, type(ms.blocks[0].ffn).__name__, ms.blocks[0].attn1.fused_projections)
with torch.no_grad():
    out = ms(hidden_states=x, timestep=t, encoder_hidden_states=ctx, return_dict=False)[0].float()
cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
print(f"SVDQ vs INT8 model output: cos={cos:.5f} finite={torch.isfinite(out).all().item()}")
