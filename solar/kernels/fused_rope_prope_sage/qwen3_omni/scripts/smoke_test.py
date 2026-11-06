"""Phase 0 smoke test: load INT4 Qwen3-Omni thinker on XPU, answer OmniBench samples, record VRAM."""
import sys, time, torch
sys.path.insert(0, "/home/gta/omni_bench/scripts")
from model_loader import load
from omnibench_data import iter_samples, build_inputs
t = time.perf_counter()
model, proc = load("sdpa")
torch.xpu.synchronize()
print(f"load {time.perf_counter()-t:.1f}s  weights={torch.xpu.memory_allocated()/2**30:.2f} GiB  has_talker={model.has_talker}")
print("q_proj type:", type(model.thinker.model.language_model.layers[0].self_attn.q_proj).__name__)
for s in iter_samples(limit=int(sys.argv[1]) if len(sys.argv) > 1 else 3):
    inp = build_inputs(proc, s, "xpu")
    L = inp["input_ids"].shape[1]
    torch.xpu.reset_peak_memory_stats(); torch.xpu.synchronize(); t = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=8, do_sample=False, return_audio=False, thinker_max_new_tokens=8)
    torch.xpu.synchronize(); dt = time.perf_counter() - t
    seq = out[0] if isinstance(out, tuple) else out
    txt = proc.batch_decode(seq[:, L:], skip_special_tokens=True)[0]
    print(f"idx={s['index']} prompt_tokens={L} gen={dt:.2f}s peak={torch.xpu.max_memory_allocated()/2**30:.2f}GiB "
          f"pred={txt!r} gold={s['answer']}")
