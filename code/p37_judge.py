#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P37 (ADR-0036) frozen read-out, REVISED before launch (ADR-0036 sec 10, Lance 2026-11-27 12:33).
Written before any P37 data existed; revised before any P37 call was made.
Reads only rank / finish_reason / top3 — never the text.
Arms: T0  = p37_c_t00_r3   (mech c, temp 0.0, 3 reps)
      T7  = p37_c_t07_r3   (mech c, temp 0.7, 3 reps)
      T7b = p37_b_t07_r3   (mech b, temp 0.7, 3 reps)
Gates (as revised):
  S-1 noise floor: T0 flip(3) > 10%  -> batch INVALID, stop.
  S-2 temperature : T7 flip(3) - T0 flip(3) >= 30pp
  S-3 format     : T7 flip(3) - T7b flip(3) >= 20pp
  S-4' top-3     : T7 top-3-recall flip(3) <= 15% pooled AND <= 20% in every grade
  S-4 majority   : DESCRIPTIVE ONLY (was a >=15pp gate; demoted, see sec 10).
Stratification rule: every difference is reported per grade FIRST; a PASS requires the same
sign in all three grades. Pooled values are descriptive only.
"""
import json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
D = REPO / "results" / "p37_stability"
ARMS = {"T0": "p37_c_t00_r3.json", "T7": "p37_c_t07_r3.json", "T7b": "p37_b_t07_r3.json"}
REPS = 3


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d


def fisher(a, b, c, d):
    """two-sided Fisher exact on [[a,b],[c,d]]; returns p."""
    n = a + b + c + d; r1, r2, c1 = a + b, c + d, a + c
    def prob(x): return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    p0 = prob(a)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p0 + 1e-12))


def load(arm):
    p = D / ARMS[arm]
    if not p.exists(): return None
    return json.loads(p.read_text(encoding="utf-8"))["rows"]


def by_puzzle(rows, key):
    out = defaultdict(dict)
    for r in rows: out[(r["puzzle"], r["grade"])][r["rep"]] = r[key]
    return out


def flip(bp, k, grade):
    """share of puzzles whose first-k reps are not all equal on `key`; also n."""
    vals = [reps for (p, g), reps in bp.items() if g == grade and len(reps) >= k]
    f = sum(1 for reps in vals if len({reps[i] for i in range(k)}) > 1)
    return f, len(vals)


def mode_agreement(bp_t7, bp_t0, k, grade):
    """for each puzzle: mode of T7's first-k reps vs T0's 5-rep mode (the 'truth')."""
    agree = n = 0
    for key, reps in bp_t7.items():
        if key[1] != grade or len(reps) < k: continue
        truth = Counter(bp_t0[key].values()).most_common(1)[0][0]
        pred = Counter(reps[i] for i in range(k)).most_common(1)[0][0]
        agree += pred == truth; n += 1
    return agree, n


def main():
    arms = {a: load(a) for a in ARMS}
    missing = [a for a, r in arms.items() if r is None]
    if missing: print("not run yet:", missing); sys.exit(0)
    for a, r in arms.items():
        assert len(r) == 60 * 3 * REPS, f"{a}: expected {60*3*REPS} rows, got {len(r)}"
        assert {(x["puzzle"], x["grade"], x["rep"]) for x in r} == {(p, g, k) for p in {x["puzzle"] for x in r} for g in (1, 2, 3) for k in range(REPS)}, f"{a}: incomplete grid"
        err = sum(1 for x in r if x["finish_reason"] == "error")
        print(f"{a}: rows {len(r)}, errors {err}, truncated {sum(1 for x in r if x['finish_reason']=='length')} ({sum(1 for x in r if x['finish_reason']=='length')/len(r):.0f}%)")
    top1 = {a: by_puzzle(r, "rank") for a, r in arms.items()}
    for a in top1:  # convert rank -> top-1 indicator
        for k in top1[a]: top1[a][k] = {rep: int(rk == 1) for rep, rk in top1[a][k].items()}

    print("\n== flip(k): share of puzzles whose k draws disagree on top-1 ==")
    print(f"{'arm':<5}{'k':>3}" + "".join(f"{'G'+str(g):>12}" for g in (1, 2, 3)) + "{'pooled':>12}")
    F = {}
    for a in ("T0", "T7", "T7b"):
        for k in (2, 3):
            cells = [flip(top1[a], k, g) for g in (1, 2, 3)]
            F[(a, k)] = cells
            tot = (sum(c[0] for c in cells), sum(c[1] for c in cells))
            print(f"{a:<5}{k:>3}" + "".join(f"{c[0]:>5}/{c[1]:<3}" for c in cells) + f"{tot[0]:>6}/{tot[1]}")
    t3 = lambda a, g: F[(a, 3)][g - 1]

    print("\n== S-1 noise floor: T0 flip(3) ==")
    c5 = F[("T0", 3)]; pooled = sum(c[0] for c in c5) / sum(c[1] for c in c5)
    s1 = pooled > 0.10
    print(f"  T0 flip(3) pooled = {pooled:.1%}  (per grade {', '.join(f'G{g+1} {c[0]}/{c[1]}' for g, c in enumerate(c5))})  -> {'FAIL: batch INVALID' if s1 else 'PASS'}")
    if s1:
        print("\nVERDICT: S-1 FAIL - the engine is not deterministic at temp 0; stop here."); return

    print("\n== S-2 temperature effect: T7 flip(3) - T0 flip(3) ==")
    print(f"  {'grade':<6}{'T7':>12}{'T0':>12}{'diff pp':>10}")
    d2 = []
    for g in (1, 2, 3):
        (f7, n7), (f0, n0) = t3("T7", g), t3("T0", g); d = (f7 / n7 - f0 / n0) * 100; d2.append(d)
        print(f"  G{g:<5}{f7:>4}/{n7:<4}{f7/n7:>7.0%}{f0:>4}/{n0:<4}{f0/n0:>7.0%}{d:>+10.1f}")
    p7, p0 = (sum(t3('T7', g)[0] for g in (1,2,3)), sum(t3('T7', g)[1] for g in (1,2,3))), (sum(t3('T0', g)[0] for g in (1,2,3)), sum(t3('T0', g)[1] for g in (1,2,3)))
    pooled_d = (p7[0]/p7[1] - p0[0]/p0[1]) * 100
    same_sign = all(x > 0 for x in d2)
    s2 = same_sign and pooled_d >= 30
    print(f"  pooled diff {pooled_d:+.1f}pp (descriptive); Fisher p = {fisher(p7[0], p7[1]-p7[0], p0[0], p0[1]-p0[0]):.2e}; same sign in all grades: {same_sign}  -> {'PASS' if s2 else 'NOT DETECTED'}")

    print("\n== S-3 format effect: T7 flip(3) - T7b flip(3) ==")
    print(f"  {'grade':<6}{'T7 (c)':>12}{'T7b (b)':>12}{'diff pp':>10}")
    d3 = []
    for g in (1, 2, 3):
        (f7, n7), (fb, nb) = t3("T7", g), t3("T7b", g); d = (f7 / n7 - fb / nb) * 100; d3.append(d)
        print(f"  G{g:<5}{f7:>4}/{n7:<4}{f7/n7:>7.0%}{fb:>4}/{nb:<4}{fb/nb:>7.0%}{d:>+10.1f}")
    pb_ = (sum(t3('T7b', g)[0] for g in (1,2,3)), sum(t3('T7b', g)[1] for g in (1,2,3)))
    pooled_d3 = (p7[0]/p7[1] - pb_[0]/pb_[1]) * 100
    same_sign3 = all(x > 0 for x in d3)
    s3 = same_sign3 and pooled_d3 >= 20
    print(f"  pooled diff {pooled_d3:+.1f}pp (descriptive); Fisher p = {fisher(p7[0], p7[1]-p7[0], pb_[0], pb_[1]-pb_[0]):.2e}; same sign in all grades: {same_sign3}  -> {'PASS' if s3 else 'NOT DETECTED'}")

    print("\n== S-4 (DESCRIPTIVE, not a gate): T7 k-draw mode vs T0 3-draw mode, top-1 ==")
    print(f"  {'k':<3}" + "".join(f"{'G'+str(g):>14}" for g in (1, 2, 3)) + "{'pooled':>14}")
    for k in (1, 3):
        cells = [mode_agreement(top1["T7"], top1["T0"], k, g) for g in (1, 2, 3)]
        tot = (sum(c[0] for c in cells), sum(c[1] for c in cells))
        print(f"  {k:<3}" + "".join(f"{c[0]:>6}/{c[1]:<3}={c[0]/c[1]:>4.0%}" for c in cells) + f"{tot[0]/tot[1]:>13.0%}")

    print("\n== S-4' top-3 recall noise: T7 top-3-recall flip(3) ==")
    t3r = by_puzzle(arms["T7"], "top3")
    cells = [flip(t3r, 3, g) for g in (1, 2, 3)]
    pooled3 = sum(c[0] for c in cells) / sum(c[1] for c in cells)
    print(f"  {'grade':<6}{'top-3 flip':>14}{'top-1 flip':>14}")
    for g in (1, 2, 3):
        f, n = cells[g - 1]; f1, n1 = t3("T7", g)
        print(f"  G{g:<5}{f:>6}/{n:<4}={f/n:>5.0%}{f1:>6}/{n1:<4}={f1/n1:>5.0%}")
    s4p = (pooled3 <= 0.15) and all(c[0] / c[1] <= 0.20 for c in cells)
    print(f"  pooled {pooled3:.1%} (gate <=15%), max grade {max(c[0]/c[1] for c in cells):.1%} (gate <=20%)  -> {'PASS' if s4p else 'NOT DETECTED'}")

    v = {"S1": "PASS" if not s1 else "FAIL", "S2": "PASS" if s2 else "NOT DETECTED", "S3": "PASS" if s3 else "NOT DETECTED",
         "S4p_top3": "PASS" if s4p else "NOT DETECTED", "S4_majority": "DESCRIPTIVE"}
    (D / "readout.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
    print("\nVERDICTS:", v)


if __name__ == "__main__":
    main()
