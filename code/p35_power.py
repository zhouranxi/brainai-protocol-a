#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P35 power analysis: what would it take to turn the R-T0 verdict into a
real answer?

Pre-registered unit = per-puzzle paired sign test on the median-rank caliber.
Observed discordant pairs (ADR-0034 sec 6): G2 8:15 of 60, G3 6:14 of 60.
G1 9:10 (effect ~0) is included only to show the floor.

Q(a) more puzzles, same 3 resamples:  how many puzzles N for 80% power
    at the observed discordance rate and split?
Q(b) why adding puzzles is inefficient: how many puzzles have a stable
    per-puzzle top-1 across their 3 resamples at all?
Q(c) cost: GPU-hours per unit of power, using measured per-row latency.

Exact binomial for (a); instability census for (b). No simulation.
"""
import json
import statistics as st
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
R12 = REPO / "results" / "p35_fulltext" / "p35_race_1200.json"
R24 = REPO / "results" / "p35_fulltext" / "p35_race_2400.json"


def sign_p(w, l):
    n = w + l
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def load():
    a = json.loads(R12.read_text(encoding="utf-8"))["rows"]
    b = json.loads(R24.read_text(encoding="utf-8"))["rows"]
    d12 = {(r["puzzle"], r["grade"], r["rep"]): r["rank"] for r in a if r["mech"] == "d"}
    d24 = {(r["puzzle"], r["grade"], r["rep"]): r["rank"] for r in b}
    return d12, d24


def observed(d12, d24):
    """Per grade: per-puzzle ranks (3 reps each) for both arms."""
    out = {}
    for g in (1, 2, 3):
        pz = sorted({p for (p, gg, _) in d12 if gg == g})
        out[g] = {p: ([d12[(p, g, r)] for r in range(3)], [d24[(p, g, r)] for r in range(3)]) for p in pz}
    return out


def top1(ranks):
    return st.median(ranks) == 1


def discordance(obs_g):
    W = sum(1 for a, b in obs_g.values() if top1(b) and not top1(a))   # R2400 wins
    L = sum(1 for a, b in obs_g.values() if top1(a) and not top1(b))   # R1200 wins
    return W, L, len(obs_g)


def power_sign(N, pW, pL, alpha=0.05):
    """Exact power of two-sided sign test with N puzzles, discordance probs pW,pL."""
    pw = 0.0
    for w in range(N + 1):
        for l in range(N + 1 - w):
            n = w + l
            pr = comb(N, w) * comb(N - w, l) * pW ** w * pL ** l * (1 - pW - pL) ** (N - n)
            if pr < 1e-18:
                continue
            if sign_p(w, l) < alpha:
                pw += pr
    return pw


def N_for_power(pW, pL, target=0.80, lo=20, hi=2000):
    for N in range(lo, hi + 1, 20):
        if power_sign(N, pW, pL) >= target:
            return N
    return None


def main():
    d12, d24 = load()
    obs = observed(d12, d24)
    secs = {r["mech"]: [] for r in json.loads(R12.read_text(encoding="utf-8"))["rows"]}
    for r in json.loads(R12.read_text(encoding="utf-8"))["rows"]:
        secs[r["mech"]].append(r["secs"])
    secs24 = [r["secs"] for r in json.loads(R24.read_text(encoding="utf-8"))["rows"]]
    t_row12 = st.mean(secs["d"])   # M-d row at 1200 (base + critique)
    t_row24 = st.mean(secs24)      # M-d row at 2400

    print("=== observed (median-rank caliber, per puzzle) ===")
    print(f"{'grade':<6}{'R2400 wins':>11}{'R1200 wins':>11}{'ties':>6}{'pW':>7}{'pL':>7}{'sign p':>8}")
    rates = {}
    for g in (1, 2, 3):
        W, L, N = discordance(obs[g])
        pW, pL = W / N, L / N
        rates[g] = (pW, pL)
        print(f"G{g:<5}{W:>11}{L:>11}{N-W-L:>6}{pW:>7.3f}{pL:>7.3f}{sign_p(W, L):>8.3f}")

    print("\n=== Q(a) more puzzles, 3 reps each: N for 80% power (two-sided sign, alpha=.05) ===")
    print(f"{'grade':<6}{'effect (pp)':>12}{'N=60 power':>12}{'N for 80%':>11}{'new puzzles':>13}{'extra GPU-h':>13}")
    need = {}
    for g in (2, 3):
        pW, pL = rates[g]
        eff = (pW - pL) * 100
        p60 = power_sign(60, pW, pL)
        N = N_for_power(pW, pL)
        need[g] = N
        # each new puzzle costs: 3 reps x (row1200 + row2400) seconds, single-stream
        extra_h = (N - 60) * 3 * (t_row12 + t_row24) / 3600 if N else float("nan")
        print(f"G{g:<5}{eff:>+12.1f}{p60:>12.1%}{N if N else '>2000':>11}{(N-60) if N else '-':>13}{extra_h:>13.1f}")
    print(f"  (per-row latency measured: M-d@1200 {t_row12:.0f}s, M-d@2400 {t_row24:.0f}s)")

    print("\n=== Q(b) why adding puzzles is inefficient: per-puzzle instability ===")
    print("  A puzzle's 'win' is decided by the median of 3 draws at temp 0.7. Count how many")
    print("  puzzles have draws that DISAGREE on top-1 (not all three the same):")
    print(f"{'grade':<6}{'R1200 unstable':>16}{'R2400 unstable':>16}{'row-level discord.':>19}")
    for g in (2, 3):
        pz = sorted({p for (p, gg, _) in d12 if gg == g})
        u12 = sum(1 for p in pz if len({d12[(p, g, r)] == 1 for r in range(3)}) == 2)
        u24 = sum(1 for p in pz if len({d24[(p, g, r)] == 1 for r in range(3)}) == 2)
        ks = [k for k in d24 if k[1] == g]
        disc = sum(1 for k in ks if (d24[k] == 1) != (d12[k] == 1))
        print(f"G{g:<5}{u12:>10}/60{'':<4}{u24:>10}/60{'':<4}{disc:>10}/180 = {disc/len(ks):.0%}")
    print("  -> Roughly two thirds of puzzles are coin-flips at this temperature; only ~1/3 carry")
    print("     a stable per-puzzle signal, so most added puzzles add variance, not information.")
    print("  -> Raising reps per puzzle cannot be simulated from the same 3 draws (resampling them")
    print("     only adds noise); it needs fresh draws, which is the same cost as adding puzzles.")

    print("\n=== Q(c) cost per 10pp of power, G2 ===")
    pW, pL = rates[2]
    prev = None
    for N in (60, 120, 180, 240, 300, 400):
        pw = power_sign(N, pW, pL)
        h = N * 3 * (t_row12 + t_row24) / 3600
        print(f"  N={N:>4}: power {pw:>5.1%}   total GPU-h (both arms) {h:>6.1f}")
    return need


if __name__ == "__main__":
    main()
