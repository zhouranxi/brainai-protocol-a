# Independent recount straight from the JSON (no reuse of p37_judge's helpers).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent recount for the ADR-0036 sec 12 read-out. Shares no helper with code/p37_judge.py:/nreads the three raw JSON files directly and re-derives truncation, T0 text-level determinism,
the uncut-cell flip rates, the rank multisets of the top-3 flips, and the level values."""
import json
from collections import Counter, defaultdict
from pathlib import Path
D = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-21-15-17/github-repo/results/p37_stability")
ARMS = {"T0": "p37_c_t00_r3.json", "T7": "p37_c_t07_r3.json", "T7b": "p37_b_t07_r3.json"}
R = {a: json.loads((D / f).read_text(encoding="utf-8"))["rows"] for a, f in ARMS.items()}
print("== truncation (finish_reason) ==")
for a, rows in R.items():
    c = Counter(r["finish_reason"] for r in rows)
    by = {g: sum(1 for r in rows if r["grade"] == g and r["finish_reason"] == "length") for g in (1, 2, 3)}
    print(f"  {a}: {dict(c)}  truncated {c.get('length', 0)}/{len(rows)} = {c.get('length', 0)/len(rows):.1%}  by grade {by}")
# temp-0 arm: are the three texts of a cell literally identical?
print("== T0 determinism at the text level ==")
by = defaultdict(list)
for r in R["T0"]: by[(r["puzzle"], r["grade"])].append(r["text"])
ident = sum(1 for v in by.values() if len(set(v)) == 1)
print(f"  cells with 3 identical texts: {ident}/{len(by)}  distinct-text cells: {len(by)-ident}")
# truncation vs top-1 instability inside T7, per grade
print("== T7: per-puzzle top-1 flip(3) split by whether the cell was truncated ==")
def flipstat(rows, key):
    b = defaultdict(dict)
    for r in rows: b[(r["puzzle"], r["grade"])][r["rep"]] = (r[key], r["finish_reason"])
    for g in (1, 2, 3):
        clean = cut = 0; ncl = ncu = 0
        for (p, gg), reps in b.items():
            if gg != g: continue
            vals = {reps[k][0] == 1 for k in reps}; anycut = any(reps[k][1] == "length" for k in reps)
            if anycut: cut += len(vals) > 1; ncu += 1
            else: clean += len(vals) > 1; ncl += 1
        print(f"  G{g}: uncut cells flip {clean:>2}/{ncl:<3}={clean/max(1,ncl):.0%}   cells with >=1 cut flip {cut:>2}/{ncu:<3}")
flipstat(R["T7"], "rank")
# top-3: what does a flip look like?
print("== T7 top-3 recall: per-grade, the rank multiset of the flipping cells ==")
b = defaultdict(dict)
for r in R["T7"]: b[(r["puzzle"], r["grade"])][r["rep"]] = (r["rank"], r["top3"])
for g in (1, 2, 3):
    fl = [tuple(sorted(v[k][0] for k in v)) for v in b.values() if v[0][1] is not None and len({(x[1] == 1) for x in v.values()}) > 1 and list(v.values())[0][0] in range(4) and g == [gg for (p, gg) in b if b[(p, gg)] is v][0]]
    print(f"  G{g}: {dict(Counter(fl))}")
# top-3 recall level itself and T0 top-3
print("== top-3 recall level (share of rows with rank in 1..3) ==")
for a in ("T0", "T7"):
    for g in (1, 2, 3):
        rows = [r for r in R[a] if r["grade"] == g]
        print(f"  {a} G{g}: {sum(r['top3'] for r in rows)}/{len(rows)} = {sum(r['top3'] for r in rows)/len(rows):.0%}   top-1 {sum(r['rank']==1 for r in rows)/len(rows):.0%}   rank0(absent) {sum(r['rank']==0 for r in rows)}")
# truncated T7 rows: is rank coming from a partial list? compare top-1 and top-3 of truncated vs complete rows, per grade
print("== T7 rows: truncated vs complete, top-1 and top-3, per grade (descriptive, not stratified by difficulty) ==")
for g in (1, 2, 3):
    t = [r for r in R["T7"] if r["grade"] == g and r["finish_reason"] == "length"]; c = [r for r in R["T7"] if r["grade"] == g and r["finish_reason"] == "stop"]
    if t:
        print(f"  G{g}: cut n={len(t):>3} top-1 {sum(r['rank']==1 for r in t)/len(t):.0%} top-3 {sum(r['top3'] for r in t)/len(t):.0%} | complete n={len(c):>3} top-1 {sum(r['rank']==1 for r in c)/len(c):.0%} top-3 {sum(r['top3'] for r in c)/len(c):.0%}")
