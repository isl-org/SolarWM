"""Accuracy + speed of OpenCL Sage variants vs SYCL Sage on captured Qwen3-Omni activations.

For each captured layer: OpenCL variants run on the real Q/K/V via bench --load; SYCL sage_causal on the
same tensors; exact fp32 causal attention for R query rows (last 256 + evenly spread) over the whole
prefix. Reports cosine / relative L2 vs fp32 for each arm, OpenCL-vs-SYCL difference on all rows,
and median call time of each arm on this data.
usage: accuracy.py OUT.json variant=OPTS [variant=OPTS ...]
"""
import glob, json, os, subprocess, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/gta/omni_bench/kernel/build")
import sage_causal

CAP = "/home/gta/omni_bench/act_capture"
BENCH = os.environ.get("SAGE_OCL_BENCH", "/tmp/sage_ocl/bench")  # needs sage_causal.cl in its cwd
WORK = "/tmp/sage_ocl/acc/work"
R_LAST, R_SPREAD = 256, 768
out_json = sys.argv[1]
variants = [a.split("=", 1) for a in sys.argv[2:]]
os.makedirs(WORK, exist_ok=True)

def bits(t):  # bf16 tensor -> uint16 numpy
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)

def ref_rows(q, k, v, rows):  # q [S,H,D] k/v [S,Hkv,D] on xpu (bf16); exact fp32 causal, bottom-right (Sq=Skv)
    S, H, D = q.shape; Hkv = k.shape[1]; g = H // Hkv
    out = torch.empty(len(rows), H, D, device="xpu", dtype=torch.float32)
    ridx = torch.tensor(rows, device="xpu")
    qr = q[ridx].float()                                           # [R,H,D]
    keys = torch.arange(S, device="xpu")
    mask = keys[None, :] > ridx[:, None]                           # [R,S]
    for h in range(H):
        kh = k[:, h // g].float(); vh = v[:, h // g].float()
        s = (qr[:, h] @ kh.T) / D ** 0.5
        s.masked_fill_(mask, float("-inf"))
        out[:, h] = torch.softmax(s, dim=-1) @ vh
    return out

def metrics(a, b):  # a approx, b ref ; flattened per (row, head)
    a = a.float().reshape(-1, a.shape[-1]); b = b.float().reshape(-1, b.shape[-1])
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    rel = (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()
    return {"cos_mean": cos.mean().item(), "cos_min": cos.min().item(), "rel_l2": rel}

def time_sycl(q, k, v, n=5):
    for _ in range(2): sage_causal.forward(q, k, v, is_causal=True)
    ts = []
    for _ in range(n):
        s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        s.record(); o = sage_causal.forward(q, k, v, is_causal=True); e.record(); torch.xpu.synchronize()
        ts.append(s.elapsed_time(e)); del o
    return sorted(ts)[n // 2]

results = []
for d in sorted(glob.glob(f"{CAP}/*")):
    for qf in sorted(glob.glob(f"{d}/L*_q.pt"), key=lambda p: int(os.path.basename(p)[1:].split("_")[0])):
        tag = f"{os.path.basename(d)}/{os.path.basename(qf)[:-5]}"
        q = torch.load(qf); k = torch.load(qf.replace("_q.pt", "_k.pt")); v = torch.load(qf.replace("_q.pt", "_v.pt"))
        _, S, H, D = q.shape; Hkv = k.shape[2]
        pre = f"{WORK}/x"
        for n, t in (("q", q), ("k", k), ("v", v)): bits(t).tofile(f"{pre}.{n}.bits")
        rows = sorted(set(range(S - R_LAST, S)) | set(np.linspace(0, S - R_LAST - 1, R_SPREAD).astype(int).tolist()))
        qx, kx, vx = q.to("xpu"), k.to("xpu"), v.to("xpu")
        ref = ref_rows(qx[0], kx[0], vx[0], rows)
        sy = sage_causal.forward(qx, kx, vx, is_causal=True); torch.xpu.synchronize()
        row = {"layer": tag, "S": S, "sycl": metrics(sy[0][rows], ref), "sycl_ms": time_sycl(qx, kx, vx)}
        for name, opts in variants:
            env = dict(os.environ, SAGE_OCL_OPTIONS=opts)
            r = subprocess.run([BENCH, "--sq", str(S), "--sk", str(S), "--hq", str(H), "--hkv", str(Hkv),
                                "--large-grf", "--warmup", "2", "--iters", "5", "--load", pre, "--dump", pre, "--out-only"],
                               cwd="/tmp/sage_ocl", env=env, capture_output=True, text=True, check=True)
            ms = float([l for l in r.stdout.splitlines() if "total" in l][0].split()[2])
            o = torch.from_numpy(np.fromfile(f"{pre}.opencl.bits", dtype=np.uint16).view(np.int16).copy()) \
                     .view(torch.bfloat16).reshape(1, S, H, D).to("xpu")
            row[name] = metrics(o[0][rows], ref)
            row[name]["vs_sycl_rel_l2"] = (torch.linalg.vector_norm(o.float() - sy.float()) /
                                           torch.linalg.vector_norm(sy.float())).item()
            row[name]["bits_differ_frac"] = (o.view(torch.int16) != sy.view(torch.int16)).float().mean().item()
            row[name]["ms"] = ms
            del o
        print(json.dumps(row), flush=True)
        results.append(row)
        del qx, kx, vx, ref, sy; torch.xpu.empty_cache()
json.dump(results, open(out_json, "w"), indent=1)
