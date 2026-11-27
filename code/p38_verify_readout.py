#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P38 post-read-out verification. Written AFTER the frozen judge had run (2026-11-28).

Purpose: every supplementary number quoted in ADR-0037 sec 12 that is NOT in the
judge's printed output is re-derived here from the raw JSON, independently of
p38_judge.py (no shared helper). Read-only: touches no data file, no ADR, no tex.

All of it is EXPLORATORY. None of it changes the frozen verdicts (H1/H2/Q2).
"""
import json
from collections import defaultdict
from itertools import combinations
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))["rows"]


def t3(rank):
    return int(rank in (1, 2, 3))


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


A = load(RES / "p38_stateless" / "p38_a_t00_s0r3.json")
B = load(RES / "p38_stateless" / "p38_a_t07_s3r4.json")
P31 = load(RES / "p31_confirm" / "p31_race.json")
R12 = load(RES / "p35_fulltext" / "p35_race_1200.json")
MC0 = load(RES / "p37_stability" / "p37_c_t00_r3.json")

# M-a at temp 0: rep 0 of arm A (H1 showed all three identical)
a0 = {(r["puzzle"], r["grade"]): t3(r["rank"]) for r in A if r["rep"] == 0}
# pooled 10 draws at 0.7: P31 -> 0..2, R1200 -> 3..5, arm B -> 6..9
pool = defaultdict(dict)
for r in P31:
    if r["mech"] == "a":
        pool[(r["puzzle"], r["grade"])][r["rep"]] = t3(r["rank"])
for r in R12:
    if r["mech"] == "a":
        pool[(r["puzzle"], r["grade"])][3 + r["rep"]] = t3(r["rank"])
for r in B:
    pool[(r["puzzle"], r["grade"])][6 + (r["rep"] - 3)] = t3(r["rank"])
assert all(len(v) == 10 for v in pool.values()) and len(pool) == 180
cnt = {k: sum(v.values()) for k, v in pool.items()}           # 0..10
maj = {k: int(c >= 6) for k, c in cnt.items()}
mean = {k: c / 10 for k, c in cnt.items()}                     # expected recall per puzzle

c0 = {(r["puzzle"], r["grade"]): int(r["top3"]) for r in MC0 if r["rep"] == 0}

G = (1, 2, 3)
puzzles = sorted({p for p, _ in a0})

print("== 0. sanity: judge's Q2 reproduced independently ==")
for g in G:
    ks = [p for p in puzzles]
    w = sum(1 for p in ks if maj[(p, g)] and not a0[(p, g)])
    l = sum(1 for p in ks if a0[(p, g)] and not maj[(p, g)])
    print(f"  G{g}: maj {sum(maj[(p,g)] for p in ks)}/60  T0 {sum(a0[(p,g)] for p in ks)}/60  W/L {w}/{l}  p={sign_p(w,l):.3f}")

print("\n== 1. pooled-across-grades discordance (exploratory; NOT a pre-registered gate) ==")
W = sum(1 for p in puzzles for g in G if maj[(p, g)] and not a0[(p, g)])
L = sum(1 for p in puzzles for g in G if a0[(p, g)] and not maj[(p, g)])
print(f"  W(0.7-only)/L(0-only) = {W}/{L}   pooled sign p = {sign_p(W, L):.4f}")
print("  sign consistency by grade:", [("+1" if sum(maj[(p,g)] for p in puzzles) > sum(a0[(p,g)] for p in puzzles) else "-1") for g in G])

print("\n== 2. the discordant cells of Q2: draw-count of the 10 draws at 0.7 ==")
for g in G:
    disc = [(p, cnt[(p, g)], a0[(p, g)], maj[(p, g)]) for p in puzzles if a0[(p, g)] != maj[(p, g)]]
    lo = [d for d in disc if d[1] in (1, 2)]
    print(f"  G{g}: {len(disc)} discordant; of these 1-2/10: {len(lo)}, 3-5/10: {sum(1 for d in disc if 3 <= d[1] <= 5)}, 8-9/10: {sum(1 for d in disc if d[1] in (8,9))}")
    print("       " + ", ".join(f"{d[0]}={d[1]}/10(T0={d[2]})" for d in disc))

print("\n== 3. caliper sensitivity: mean-of-10 and single-draw (exploratory) ==")
for g in G:
    m0 = sum(a0[(p, g)] for p in puzzles)
    mmaj = sum(maj[(p, g)] for p in puzzles)
    mmean = sum(mean[(p, g)] for p in puzzles)
    singles = []
    for src, rows, off in (("P31", [r for r in P31 if r["mech"] == "a"], 0), ("R1200", [r for r in R12 if r["mech"] == "a"], 3), ("armB", B, 6)):
        for rep in range(3 if src != "armB" else 4):
            v = sum(t3(r["rank"]) for r in rows if r["grade"] == g and r["rep"] == (rep if src != "armB" else rep + 3))
            singles.append(v)
    print(f"  G{g}: T0 {m0}  majority-10 {mmaj} ({(mmaj-m0)/60*100:+.0f}pp)   mean-of-10 {mmean:.1f} ({(mmean-m0)/60*100:+.1f}pp)   single draws range {min(singles)}-{max(singles)} ({(min(singles)-m0)/60*100:+.0f}..{(max(singles)-m0)/60*100:+.0f}pp, n={len(singles)})")

print("\n== 4. equivalence / power sketch for Q2 (exploratory) ==")
# Under H0 'no temperature effect' each discordant cell is a fair coin; observed split W:L.
# Exact binomial two-sided p for pooled W=0 of n discordant:
print(f"  observed pooled W/L = {W}/{L}; two-sided exact binomial p (coin fair) = {sign_p(W, L):.4f}")
# Power to detect, at alpha=.05 two-sided, a per-grade shift of d puzzles when discordance rate is r of 60:
for r in (8, 13, 23):
    n = r  # expected discordant cells per grade under a 'rate r per 60' scenario
    # all discordant going one way is the best case; exact sign test needs k such that 2/2^k <= .05 -> k>=6
    print(f"    if {r} cells/grade were discordant and ALL fell one way: sign p = {sign_p(r, 0):.4f}  -> {'detectable' if sign_p(r,0) < .05 and r/60 >= 10/60*0 + 0 and r >= 6 else 'NOT detectable (need >=6 one-way cells for p<.05)'}, |diff| = {r/60*100:.0f}pp (gate needs >=10pp = 6 cells)")

print("\n== 5. Q4 per-grade: compare M-a 10-draw histogram with the 6-draw histogram in ADR sec 2 ==")
for g in G:
    h = defaultdict(int)
    for p in puzzles:
        h[cnt[(p, g)]] += 1
    print(f"  G{g}: {dict(sorted(h.items()))}")

print("\n== 6. Q3 cross-check: the confirmation batch's own +8pp (P31 M-c median-3 vs P31 M-a median-3) ==")
p31c = defaultdict(dict)
for r in P31:
    if r["mech"] == "c":
        p31c[(r["puzzle"], r["grade"])][r["rep"]] = t3(r["rank"])
c31 = {k: int(sorted(v.values())[1] == 1) for k, v in p31c.items()}
a31 = {k: int(sorted(v[i] for i in (0, 1, 2))[1] == 1) for k, v in pool.items()}
for g in G:
    ks = puzzles
    w = sum(1 for p in ks if c31[(p, g)] and not a31[(p, g)])
    l = sum(1 for p in ks if a31[(p, g)] and not c31[(p, g)])
    print(f"  G{g}: P31 M-c {sum(c31[(p,g)] for p in ks)}/60 vs P31 M-a {sum(a31[(p,g)] for p in ks)}/60  gain {(sum(c31[(p,g)] for p in ks)-sum(a31[(p,g)] for p in ks))/60*100:+.0f}pp  {w}/{l}  p={sign_p(w,l):.3f}")

print("\n== 7. which puzzles drive the G1 early gain at temp 0 (descriptive) ==")
for g in G:
    both = sum(1 for p in puzzles if c0[(p, g)] and a0[(p, g)])
    conly = [p for p in puzzles if c0[(p, g)] and not a0[(p, g)]]
    aonly = [p for p in puzzles if a0[(p, g)] and not c0[(p, g)]]
    none = sum(1 for p in puzzles if not c0[(p, g)] and not a0[(p, g)])
    print(f"  G{g}: both {both}, M-c only {len(conly)}, M-a only {len(aonly)}, neither {none}")
    if g == 1:
        print("       M-c-only at G1:", conly)
        print("       M-a-only at G1:", aonly)

print("\n== 8. M-a at temp 0: truncation (finish_reason=length) ==")
from collections import Counter
print("  arm A finish:", dict(Counter(r.get("finish_reason") for r in A)))
print("  arm B finish:", dict(Counter(r.get("finish_reason") for r in B)))
print("  arm A length rows by grade:", dict(Counter(r["grade"] for r in A if r.get("finish_reason") == "length")))
print("  arm A rank dist:", dict(Counter(r["rank"] for r in A)))
print("  arm B rank dist:", dict(Counter(r["rank"] for r in B)))

print("\n== 9. H1 independent: byte-identical text across the three arm-A reps ==")
txt = defaultdict(dict)
for r in A:
    txt[(r["puzzle"], r["grade"])][r["rep"]] = r.get("text")
ident = sum(1 for v in txt.values() if len(set(v.values())) == 1)
print(f"  identical-text cells {ident}/{len(txt)}")
rk = defaultdict(set)
for r in A:
    rk[(r["puzzle"], r["grade"])].add(r["rank"])
print(f"  identical-rank cells {sum(1 for v in rk.values() if len(v)==1)}/{len(rk)}")

print("\nDONE (verification only; verdicts unchanged)")
