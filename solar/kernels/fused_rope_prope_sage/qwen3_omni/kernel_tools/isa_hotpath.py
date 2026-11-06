"""Count always-executed vs conditional instructions in the KV loop of the bf16 D=128 attention kernel."""
# Usage: build with IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=DIR bash build.sh ...; then python3 isa_hotpath.py DIR
import re, sys, glob, collections
d = sys.argv[1]
cands = []
for f in glob.glob(d + "/*simd16*.asm"):
    t = open(f).read()
    if "attentionIN7cutlass10bfloat16" in t: cands.append((t.count("dpas"), f))
f = min(cands)[1]; L = open(f).read().split("\n")  # D=128 has fewer DPAS per loop than D=64
# loop = from the target label of the last backward jmpi to that jmpi
back = None
for i, l in enumerate(L):
    m = re.search(r"jmpi\s+(_0_\d+)", l)
    if m:
        tgt = next((j for j, x in enumerate(L) if x.strip() == m.group(1) + ":"), None)
        if tgt is not None and tgt < i and "dpas" in "\n".join(L[tgt:i]): back = (tgt, i)
a, b = back; B = L[a:b]
labels = {m.group(1): i for i, l in enumerate(B) if (m := re.match(r"^\s*(_0_\d+|L\d+):\s*$", l))}
cond = [0] * len(B)
for i, l in enumerate(B):
    m = re.search(r"\((W&~?f[\d.]+|~?f[\d.]+)\)\s+(jmpi|goto)\s.*?\s(_0_\d+)", l)
    if m and m.group(3) in labels and labels[m.group(3)] > i:
        for k in range(i + 1, labels[m.group(3)]): cond[k] = 1
for sel, name in ((0, "always-executed"), (1, "conditional")):
    op = collections.Counter()
    for i, l in enumerate(B):
        s = l.strip()
        if cond[i] != sel or not s or s.startswith("//") or s.endswith(":"): continue
        base = re.sub(r"^\([^)]*\)\s*", "", s).split()[0].split(".")[0]
        if base not in ("jmpi", "goto", "join"): op[base] += 1
    print(f"{name}: {sum(op.values())}  {op.most_common(10)}")
