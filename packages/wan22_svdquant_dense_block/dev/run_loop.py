import sys, os, importlib, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task, bench
mod = importlib.import_module(task.extension_name(task.load_hparams()))
bd = task.block_data.__wrapped__(torch.device("xpu"))
f = bench.layer_fns(mod, bd)[sys.argv[1] if len(sys.argv) > 1 else "block"]
for _ in range(5): f()
torch.xpu.synchronize()
for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 5): f()
torch.xpu.synchronize()
