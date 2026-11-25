#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P35 phase-B independent cross-check of code/p35_analyze.py (ADR-0034, Q2).

Written after the read-out, before the paper edit. Re-derives the Q2 gate
numbers from the raw JSON without importing p35_analyze, then adds what the
pre-registered read-out does not print:

  [1] replay        -- the pre-registered gate numbers, recomputed independently
  [2] row bootstrap -- CI for the R2400 - R1200 top-1 difference, paired per puzzle
  [3] power         -- what the 60-puzzle sign test could have detected
  [4] equivalence   -- can R-T0 be read as "no effect", or only "not established"?
  [5] within-arm    -- top-1 among truncated vs complete rows inside R1200 / R2400
  [6] length        -- output length by grade and arm
  [7] dup           -- is the dup doubling a scorer artifact of longer texts?
  [8] P31 anchor    -- the third draw of the same cell (original P31), same caliber
  [9] rep-to-rep    -- how far apart are two independent draws of one arm's M-d?

Everything is computed from results/p35_fulltext/*.json and results/p31_confirm.
Exploratory: adjudicates nothing, re-labels no gate. Prints only.
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"


def rows_of(mt):
    p = RES / "p35_fulltext" / f"p35_race_{mt}.json"
    return json.loads(p.read_text(encoding="utf-8"))["rows"]


def upper_median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def med_rank(rows):
    acc = defaultdict(list)
    for r in rows:
        acc[(r["grade"], r["puzzle"])].append(r["rank"])
    return {k: upper_median(v) for k, v in acc.items()}


def sign_p(w, l):
    from math import comb
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def top1_by_grade(med):
    out = {}
    for g in (1, 2, 3):
        v = [x for (gg, _), x in med.items() if gg == g]
        out[g] = sum(1 for x in v if x == 1) / len(v)
    return out


# ---------------- v2 scorer, verbatim (code/p34_scorer_audit.py:36-69) ----------------
ITEM_RE = re.compile(r"^\s*(?:\d+[\.、)]|[-•*]|\(\d+\)|（\d+）)")
NUM_RE = re.compile(r"^\s*(\d+)[\.、)]")
PCT_RE = re.compile(r"\d+(\.\d+)?\s*%")


def _parse_items(lines):
    items, last_no = [], 0
    for i, ln in enumerate(lines):
        if not ITEM_RE.match(ln):
            continue
        m = NUM_RE.match(ln)
        last_no = int(m.group(1)) if m else last_no + 1
        pct = None
        for j in (i, i + 1, i + 2):
            if j < len(lines):
                mm = PCT_RE.search(lines[j])
                if mm:
                    pct = float(mm.group(0).rstrip("%"))
                    break
        if pct is not None:
            items.append((i, last_no, pct))
    return items


def gold_rank_v2(text, gold):
    if text.startswith("[ERROR"):
        return -1
    lines = text.splitlines()
    items = _parse_items(lines)
    cand = [(pct, no) for (i, no, pct) in items if gold in lines[i]]
    if cand:
        return min(max(cand)[1], 3)
    return 0 if gold not in text else 1


def main():
    r12 = [r for r in rows_of(1200) if r["mech"] == "d"]
    r24 = rows_of(2400)
    p31 = [r for r in json.loads((RES / "p31_confirm" / "p31_race.json").read_text(encoding="utf-8"))["rows"]
           if r["mech"] == "d"]
    m12, m24 = med_rank(r12), med_rank(r24)
    pz = sorted({k[1] for k in m24})
    print(f"rows: R1200 M-d {len(r12)}, R2400 M-d {len(r24)}, P31 M-d {len(p31)}; puzzles {len(pz)}")
    print(f"same puzzle keys in both arms: {set(m12) == set(m24)}")

    # [1] replay of the gate
    print("\n[1] gate replay (independent re-computation)")
    a12, a24 = top1_by_grade(m12), top1_by_grade(m24)
    hits = []
    for g in (1, 2, 3):
        w = sum(1 for p in pz if m24[(g, p)] == 1 and m12[(g, p)] != 1)
        l = sum(1 for p in pz if m12[(g, p)] == 1 and m24[(g, p)] != 1)
        pp = sign_p(w, l)
        d = 100 * (a24[g] - a12[g])
        hit = abs(d) >= 10 and pp < 0.05
        hits.append(hit)
        print(f"    G{g}: R1200 {a12[g]:.1%} ({sum(1 for p in pz if m12[(g, p)] == 1)}/60)  "
              f"R2400 {a24[g]:.1%} ({sum(1 for p in pz if m24[(g, p)] == 1)}/60)  diff {d:+.1f} pp; "
              f"R2400 wins {w} / R1200 wins {l}; p = {pp:.3f}; hit = {hit}")
    print(f"    -> any hit: {any(hits)}  =>  {'R-T1' if any(hits) else 'R-T0'}")

    # [2] paired bootstrap over puzzles
    print("\n[2] paired bootstrap over the 60 puzzles (20,000 resamples, seed 0): R2400 - R1200 top-1")
    for g in (1, 2, 3):
        rng = random.Random(0)  # same stream for every grade, so [2] and [4] agree
        ds = []
        for _ in range(20000):
            s = [rng.choice(pz) for _ in pz]
            ds.append(sum(1 for p in s if m24[(g, p)] == 1) / 60 - sum(1 for p in s if m12[(g, p)] == 1) / 60)
        ds.sort()
        inside = ds[500] > -0.10 and ds[19500] < 0.10
        print(f"    G{g}: {100 * (a24[g] - a12[g]):+.1f} pp, 95% CI [{100 * ds[500]:+.1f}, {100 * ds[19500]:+.1f}] pp "
              f"-> {'inside' if inside else 'NOT inside'} +/-10 pp (equivalence "
              f"{'supported' if inside else 'NOT supported'})")

    # [3] power of the 60-puzzle sign test at the observed discordance rate
    print("\n[3] power of the pre-registered test (exact, two-sided alpha .05, per grade)")
    nd = {g: sum(1 for p in pz if (m24[(g, p)] == 1) != (m12[(g, p)] == 1)) for g in (1, 2, 3)}
    for g in (1, 2, 3):
        n = nd[g]
        # smallest majority that reaches p < .05 with n discordant pairs (take the larger side of the split)
        crit = next(k for k in range(n + 1) if sign_p(k, n - k) < 0.05 and k >= n - k)
        # power = P(split is significant) when each discordant pair favours R2400 with prob q
        line = []
        for q in (0.5, 0.8, 0.9, 1.0):
            pw = sum(_pmf(k, n, q) for k in range(n + 1) if sign_p(k, n - k) < 0.05)
            line.append(f"q={q:.0%}: {pw:.0%}")
        print(f"    G{g}: discordant pairs {n}/60; significance needs a split of at least {crit}:{n - crit}; "
              "power when each discordant pair favours R2400 with prob " + ", ".join(line))
    print("    reading: a 10 pp shift in a 60-puzzle cell is 6 puzzles; with ~20 discordant puzzles per grade")
    print("    that is a 13:7 split, which the two-sided sign test does not reach (it needs about 15:4).")

    # [5] within-arm truncation contrast
    print("\n[5] within-arm top-1, truncated vs complete rows (descriptive; truncation is not randomised)")
    for label, rw in (("R1200", r12), ("R2400", r24)):
        for g in (1, 2, 3):
            s = [r for r in rw if r["grade"] == g]
            t = [r for r in s if r["finish_reason"] == "length"]
            c = [r for r in s if r["finish_reason"] != "length"]
            print(f"    {label} G{g}: cut {sum(1 for r in t if r['rank'] == 1)}/{len(t)} "
                  f"({(sum(1 for r in t if r['rank'] == 1) / len(t)) if t else float('nan'):.1%})  |  "
                  f"complete {sum(1 for r in c if r['rank'] == 1)}/{len(c)} "
                  f"({sum(1 for r in c if r['rank'] == 1) / len(c):.1%})")

    # [6] output length by grade and arm
    print("\n[6] M-d critique-pass output length (characters), median [p10, p90]")
    for label, rw in (("R1200", r12), ("R2400", r24)):
        for g in (1, 2, 3):
            L = sorted(len(r["text"]) for r in rw if r["grade"] == g)
            print(f"    {label} G{g}: {L[len(L) // 2]} [{L[len(L) // 10]}, {L[9 * len(L) // 10]}]")

    # [7] dup vs length
    print("\n[7] duplicate-gold count vs text length")
    for label, rw in (("R1200", r12), ("R2400", r24)):
        L = sorted(len(r["text"]) for r in rw)
        med = L[len(L) // 2]
        lo = [r for r in rw if len(r["text"]) <= med]
        hi = [r for r in rw if len(r["text"]) > med]
        def m(x):
            return sum(r["dup"] for r in x) / len(x) if x else float("nan")
        print(f"    {label}: dup mean, shorter half {m(lo):.2f} (n={len(lo)}, median len {med})  |  "
              f"longer half {m(hi):.2f} (n={len(hi)})")
        for g in (1, 2, 3):
            s = [r for r in rw if r["grade"] == g]
            t = [r for r in s if r["finish_reason"] == "length"]
            print(f"      G{g}: dup mean all {m(s):.2f}; among the {len(t)} cut rows {m(t):.2f}")

    # [8] the third draw of the same cell: original P31
    print("\n[8] three draws of the same 60-puzzle M-d cell, median-rank caliber")
    m31 = med_rank(p31)
    a31 = top1_by_grade(m31)
    for g in (1, 2, 3):
        print(f"    G{g}: P31 {a31[g]:.1%}  R1200 {a12[g]:.1%}  R2400 {a24[g]:.1%}")
    print("    per-pair paired sign test (median-rank caliber):")
    for name, ma, mb in (("P31 vs R1200 (same cap, independent draws)", m31, m12),
                         ("P31 vs R2400", m31, m24),
                         ("R1200 vs R2400", m12, m24)):
        parts = []
        for g in (1, 2, 3):
            w = sum(1 for p in pz if mb[(g, p)] == 1 and ma[(g, p)] != 1)
            l = sum(1 for p in pz if ma[(g, p)] == 1 and mb[(g, p)] != 1)
            parts.append(f"G{g} {w}/{l} p={sign_p(w, l):.2f}")
        print(f"      {name}: " + ";  ".join(parts))
    print("    rep-to-rep spread inside each arm (single-rep top-1, 60 rows per rep):")
    for label, rw in (("P31", p31), ("R1200", r12), ("R2400", r24)):
        for g in (1, 2, 3):
            v = [100 * sum(1 for r in rw if r["grade"] == g and r["rep"] == k and r["rank"] == 1) / 60 for k in range(3)]
            print(f"      {label} G{g}: " + " ".join(f"{x:5.1f}%" for x in v) + f"   max gap {max(v) - min(v):5.1f} pp")

    # [9] R2400: is the residual truncation concentrated where the behaviour moved?
    print("\n[9] R2400 residual truncation by grade")
    for g in (1, 2, 3):
        s = [r for r in r24 if r["grade"] == g]
        n = sum(1 for r in s if r["finish_reason"] == "length")
        print(f"    G{g}: {n}/{len(s)} = {n / len(s):.1%}")
    print("\n[10] rank distribution, M-d, row-level")
    for label, rw in (("R1200", r12), ("R2400", r24)):
        for g in (1, 2, 3):
            c = Counter(r["rank"] for r in rw if r["grade"] == g)
            print(f"    {label} G{g}: " + "  ".join(f"rank {k}: {c[k]}" for k in sorted(c)))


def _pmf(k, n, q):
    from math import comb
    return comb(n, k) * q ** k * (1 - q) ** (n - k)


if __name__ == "__main__":
    main()
