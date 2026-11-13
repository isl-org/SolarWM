import sys, os, importlib, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task, bench
from torch.profiler import profile, ProfilerActivity
mod = importlib.import_module(task.extension_name(task.load_hparams()))
bd = task.block_data.__wrapped__(torch.device("xpu"))
fns = bench.layer_fns(mod, bd)
which = sys.argv[1] if len(sys.argv) > 1 else "block"
f = fns[which]
for _ in range(5): f()
torch.xpu.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as p:
    for _ in range(10): f()
    torch.xpu.synchronize()
rows = [e for e in p.key_averages() if e.device_time_total > 0]
for e in sorted(rows, key=lambda e: -e.device_time_total)[:12]:
    print(f"{e.key[:70]:70s} n={e.count:4d} avg={e.device_time_total/e.count/1000:8.3f} ms")
