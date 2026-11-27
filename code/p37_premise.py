import json
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P37 premise check (exploratory, no new model calls): is per-puzzle top-1 instability
(a) produced by the 3-candidate output format, and (b) a property of the puzzle or of the
generation? Uses the existing P32 2x2 (b3/b4) and P31 arms. Run from anywhere."""
from collections import defaultdict, Counter
from math import comb
from pathlib import Path
R = str(Path(__file__).resolve().parent.parent / "results") + "/"
def signp(w,l):
    n=w+l; k=min(w,l); return min(1.0, 2*sum(comb(n,i) for i in range(k+1))/2**n) if n else 1.0
# P32's 2x2 is the cleanest existing evidence on FORMAT vs HISTORY. Arms:
#   b3 = no history + full 3-candidate format   (format only)
#   b4 = no history + single-answer format       (neither)
#   M-c (P31) = history + format                 (both)
#   M-b (P31) = history + single answer          (history only)
srcs = {"b3 (format only)": ("p32_decompose/p32_decompose.json","arm","b3"), "b4 (neither)": ("p32_decompose/p32_decompose.json","arm","b4"),
        "M-c (both)": ("p31_confirm/p31_race.json","mech","c"), "M-b (history only)": ("p31_confirm/p31_race.json","mech","b")}
print("Per-puzzle top-1 instability (3 reps disagree), by grade -- the 2x2 of format x history")
print(f"{'arm':<22}{'G1':>10}{'G2':>10}{'G3':>10}{'all':>10}   unstable type (G3): order-flip / member-flip")
inst = {}
for name,(f,k,v) in srcs.items():
    rr = [r for r in json.load(open(R+f, encoding="utf-8"))["rows"] if r[k]==v]
    line = f"{name:<22}"; tot_u = tot_n = 0; of = mf = 0
    for g in (1,2,3):
        by = defaultdict(list)
        for r in rr:
            if r["grade"]==g: by[r["puzzle"]].append(r["rank"])
        u = [x for x in by.values() if len({(y==1) for y in x})>1]
        inst[(name,g)] = (len(u), len(by))
        tot_u += len(u); tot_n += len(by)
        line += f"{len(u):>4}/{len(by)}"
        if g==3:
            of = sum(1 for x in u if 0 not in x); mf = len(u)-of
    line += f"{tot_u:>5}/{tot_n}   {of} / {mf}"
    print(line)
print("\nFormat main effect on instability (format arms minus single-answer arms, pp), split by history:")
for h, (fm, sm) in (("history+", ("M-c (both)", "M-b (history only)")), ("history-", ("b3 (format only)", "b4 (neither)"))):
    for g in (1,2,3):
        (a,an),(b,bn) = inst[(fm,g)], inst[(sm,g)]
        print(f"  {h} G{g}: {fm.split(' ')[0]} {a}/{an} = {a/an:.0%}  vs  {sm.split(' ')[0]} {b}/{bn} = {b/bn:.0%}   diff {(a/an-b/bn)*100:+.0f}pp")
# Is the instability a property of the PUZZLE (shared across mechanisms) or of the MECHANISM?
print("\nIs instability shared across mechanisms on the same puzzle? (G3, pooled P31+R1200 for M-c/M-d; P32 for b3/b4)")
def unstable_set(f,k,v,g):
    rr = [r for r in json.load(open(R+f, encoding="utf-8"))["rows"] if r[k]==v and r["grade"]==g]
    by = defaultdict(list)
    for r in rr: by[r["puzzle"]].append(r["rank"]==1)
    return {p for p,x in by.items() if len(set(x))>1}
U = {"M-c": unstable_set("p31_confirm/p31_race.json","mech","c",3) | unstable_set("p35_fulltext/p35_race_1200.json","mech","c",3),
     "M-d": unstable_set("p31_confirm/p31_race.json","mech","d",3) | unstable_set("p35_fulltext/p35_race_1200.json","mech","d",3),
     "b3": unstable_set("p32_decompose/p32_decompose.json","arm","b3",3), "b4": unstable_set("p32_decompose/p32_decompose.json","arm","b4",3)}
import itertools
for a,b in itertools.combinations(U,2):
    A,B = U[a],U[b]; inter = len(A&B); exp = len(A)*len(B)/60
    print(f"  {a} ∩ {b}: {inter} puzzles (expected if independent {exp:.1f})  overlap ratio {inter/exp:.2f}")
