import csv, sys
rows = list(csv.reader(open(sys.argv[1]), delimiter=';'))
hdr = rows[0]
cols = {"avg_ms":4,"n":5,"GRF":9,"spill":8,"Active":13,"Stall":14,"Idle":15,"Occ":20,"XMX%":27,"ALU0%":25,"SBID":34,"Send":36,"Pipe":37,"Dist":35,"L3busy":41,"L3stall":42,"L3rd":44,"memR":50,"memW":51,"L3miss":48}
print(f"{'kernel':44s}" + "".join(f"{k:>9s}" for k in cols))
for r in rows[1:]:
    if not r or "SVDQ" not in r[0] and "RmsNorm" not in r[0]: continue
    name = r[0].replace("(bool)","").replace("(int)","").replace("(Act)","")[:42]
    vals = []
    for k, i in cols.items():
        try: v = float(r[i]); vals.append(f"{v*1000 if k=='avg_ms' else v:9.1f}")
        except Exception: vals.append(f"{r[i][:8]:>9s}")
    print(f"{name:44s}" + "".join(vals))
print("(avg_ms column printed in microseconds)")
