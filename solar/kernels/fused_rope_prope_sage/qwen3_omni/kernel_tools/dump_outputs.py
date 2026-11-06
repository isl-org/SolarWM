"""Save sage_causal outputs on fixed inputs (for bitwise comparison of two builds, one process each).
    ./env.sh python kernel/dump_outputs.py BUILD_DIR OUT.pt"""
import sys, torch
sys.path.insert(0, sys.argv[1]); import sage_causal as m
g = torch.Generator(device="xpu").manual_seed(3); outs = []
for sq, skv, hq, hkv, d, c in [(2048, 2048, 32, 4, 128, True), (1000, 1000, 32, 4, 128, True), (300, 2000, 32, 4, 128, True),
                               (777, 777, 16, 16, 128, True), (8192, 8192, 32, 4, 128, True), (2048, 2048, 32, 4, 128, False),
                               (1500, 1500, 20, 5, 64, True), (416, 416, 20, 20, 64, False)]:
    q = torch.randn(1, sq, hq, d, device="xpu", generator=g).bfloat16()
    k = torch.randn(1, skv, hkv, d, device="xpu", generator=g).bfloat16(); v = torch.randn(1, skv, hkv, d, device="xpu", generator=g).bfloat16()
    outs.append(m.forward(q, k, v, c).cpu())
torch.save(outs, sys.argv[2])
