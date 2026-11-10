"""Pick the VBench-2.0 subset: 2 dimensions per category, 3 prompts per dimension at 1/6, 1/2 and 5/6 of the
dimension's prompt list (1 seed each); Diversity: its middle prompt with 6 seeds. Writes subset.json with
the VBench-2.0 full-info entries (auxiliary_info included) and per-video seeds, shared by every arm. Rows are
ordered in 3 rounds of 11 videos (one prompt of every dimension, 2 Diversity seeds), so generating with
--limit 11 / 22 / 33 gives balanced partial results.

    python3 wan/vbench/select_subset.py <VBench-2.0 dir> wan/vbench/subset.json
"""
import json, sys
VB, OUT = sys.argv[1], sys.argv[2]
DIMS = {"Human Fidelity": ["Human_Anatomy", "Human_Identity"], "Creativity": ["Composition", "Diversity"],
        "Controllability": ["Camera_Motion", "Dynamic_Attribute"], "Physics": ["Mechanics", "Multi-View_Consistency"],
        "Commonsense": ["Motion_Rationality", "Instance_Preservation"]}
info = json.load(open(f"{VB}/vbench2/VBench2_full_info.json"))
rows, seed = [], 1000
for cat, dims in DIMS.items():
    for dim in dims:
        prompts = [p.strip() for p in open(f"{VB}/prompts/prompt/{dim}.txt") if p.strip()]
        n = len(prompts)
        picks = [prompts[n // 2]] if dim == "Diversity" else [prompts[n * (2 * k + 1) // 6] for k in range(3)]
        for k, p in enumerate(picks):
            entry = next(e for e in info if e["prompt_en"].strip() == p and dim in e["dimension"])
            for idx in range(6 if dim == "Diversity" else 1):
                rows.append(dict(round=idx // 2 if dim == "Diversity" else k, category=cat, dimension=dim, prompt=p, index=idx, seed=seed,
                                 file=f"{dim}/{p[:180]}-{idx}.mp4", auxiliary_info=entry.get("auxiliary_info")))
                seed += 1
rows.sort(key=lambda r: r["round"])   # stable: each third of the list covers every dimension
json.dump(rows, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"{len(rows)} videos per arm")
for r in rows:
    print(f"round {r['round']} {r['dimension']:24s} seed {r['seed']}  {r['prompt'][:100]}")
