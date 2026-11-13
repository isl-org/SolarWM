"""Instruction-class counts of the GEMM main loop (region around the INT4 dpas) in IGC asm dumps."""
import re, sys, glob, collections
def stats(path):
    lines = open(path).read().split("\n")
    name = next((l for l in lines if l.startswith("//.kernel")), "")
    m = re.search(r"Cfg(.*?)E(Lb\d)(Lb\d)(L3Act\d)", name)
    idx = [i for i, l in enumerate(lines) if "dpas.8x8" in l and ":s4" in l]
    if not idx: return None
    # loop region: previous label before first s4 dpas .. next label/jmpi after last (back edge = jmpi to an earlier label)
    labels = {}
    for i, l in enumerate(lines):
        mm = re.match(r"^(_\d+_\d+):", l)
        if mm: labels[mm.group(1)] = i
    start = idx[0]
    while start > 0 and not re.match(r"^_\d+_\d+:", lines[start]): start -= 1
    end = idx[-1]
    while end < len(lines):
        mm = re.search(r"jmpi\s+(_\d+_\d+)", lines[end])
        if mm and labels.get(mm.group(1), 1e9) <= start: break
        end += 1
    cnt = collections.Counter()
    for l in lines[start:end + 1]:
        l = l.split("//")[0].strip()
        if not l or l.startswith("_") or l.startswith("sync"): continue
        l = re.sub(r"^\(.*?\)\s*", "", l)
        op = l.split()[0].split(".")[0]
        cnt[op] += 1
    tot = sum(cnt.values())
    return name[name.find("Cfg"):][:75], tot, cnt
for p in sorted(glob.glob(sys.argv[1] + "/*.asm")):
    r = stats(p)
    if r is None or "SVDQ_GEMM" not in open(p).read(2000): continue
    name, tot, cnt = r
    if "Lb0ELb0EL3Act0" not in name.replace("EL", "EL") and "ELb0ELb0EL3Act0E" not in name: continue
    c = cnt
    print(f"{name:75s} total={tot:4d} dpas={c['dpas']:2d} send={c['send']:3d} mov={c['mov']:3d} mul={c['mul']:3d} mad={c['mad']:3d} add/shl/etc={tot-c['dpas']-c['send']-c['mov']-c['mul']-c['mad']:3d}")
