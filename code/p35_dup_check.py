#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P35 phase B, exploratory: why does the duplicate-gold count rise at 2400 tokens?

The pre-registered secondary read-out reports mean `dup` per row (R1200 -> R2400:
G2 0.73 -> 1.38, G3 0.96 -> 1.89). `dup` (code/p25_race_v2.py:88, gold_dup_count) is the
number of confidence-bearing list-item lines that mention the gold answer. Two candidate
explanations:

  (i)  behavioral -- with the cap lifted, critique restates the gold answer more often;
  (ii) mechanical -- the counter scales with how much enumerated text survives. A cut text
       loses the lines after the cut point, so a longer text mechanically carries more
       item lines, and more of them mention gold.

Checks (all descriptive; truncation is not randomised, so nothing here is a gate):
  0. replay: recompute `dup` from the archived text, must equal the recorded value
  1. dup vs total number of confidence-bearing item lines (pooled over arms)
  2. dup stratified on text length, COMPLETE rows only, both arms
  3. per-puzzle paired comparison of dup (median over 3 reps), R2400 vs R1200
"""
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"

# v2 scorer item parser, verbatim (code/p34_scorer_audit.py:36-57)
ITEM_RE = re.compile(r"^\s*(?:\d+[\.、)]|[-•*]|\(\d+\)|（\d+）)")
NUM_RE = re.compile(r"^\s*(\d+)[\.、)]")
PCT_RE = re.compile(r"\d+(\.\d+)?\s*%")


def parse_items(lines):
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


def counts(text, gold):
    """(n_item_lines, n_gold_item_lines)"""
    lines = text.splitlines()
    items = parse_items(lines)
    return len(items), sum(1 for (i, _, _) in items if gold in lines[i])


def upper_median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def sign_p(w, l):
    from math import comb
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def rows_of(mt):
    return [r for r in json.loads((RES / "p35_fulltext" / f"p35_race_{mt}.json").read_text(encoding="utf-8"))["rows"]
            if r["mech"] == "d"]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    gold_of = {p["id"]: p["gold"] for p in json.loads((REPO / "code" / "p31_puzzles.json").read_text(encoding="utf-8"))}
    arms = {"R1200": rows_of(1200), "R2400": rows_of(2400)}
    for label, rw in arms.items():
        for r in rw:
            r["_n_items"], r["_n_gold"] = counts(r["text"], gold_of[r["puzzle"]])

    print("[0] replay: recomputed gold-item-line count vs recorded `dup`")
    for label, rw in arms.items():
        mism = sum(1 for r in rw if r["_n_gold"] != r["dup"])
        print(f"    {label}: mismatches {mism}/{len(rw)}")

    print("\n[1] dup vs number of confidence-bearing item lines (both arms pooled)")
    pooled = defaultdict(list)
    for rw in arms.values():
        for r in rw:
            pooled[min(r["_n_items"], 12)].append(r["dup"])
    for k in sorted(pooled):
        v = pooled[k]
        print(f"    {k:>2} item lines{'+' if k == 12 else ' '}: mean dup {mean(v):.2f}   n={len(v)}")

    print("\n    item lines per row, mean:")
    for label, rw in arms.items():
        print(f"    {label}: " + "   ".join(
            f"G{g}: items {mean([r['_n_items'] for r in rw if r['grade'] == g]):.2f}, "
            f"gold-item {mean([r['_n_gold'] for r in rw if r['grade'] == g]):.2f}" for g in (1, 2, 3)))

    print("\n[2] dup stratified on text length, COMPLETE rows only (200-character bins)")
    for g in (2, 3):
        print(f"    G{g}")
        b12, b24 = defaultdict(list), defaultdict(list)
        for r in arms["R1200"]:
            if r["grade"] == g and r["finish_reason"] != "length":
                b12[len(r["text"]) // 200].append(r["dup"])
        for r in arms["R2400"]:
            if r["grade"] == g and r["finish_reason"] != "length":
                b24[len(r["text"]) // 200].append(r["dup"])
        for b in sorted(set(b12) | set(b24)):
            a, c = b12.get(b, []), b24.get(b, [])
            fa = f"{mean(a):.2f} (n={len(a):>3})" if a else "-            "
            fc = f"{mean(c):.2f} (n={len(c):>3})" if c else "-            "
            print(f"      {b * 200:>5}-{b * 200 + 199:<5}: R1200 {fa:>16}   R2400 {fc:>16}")

    # common support: bins present in both arms
    print("\n    common length support (bins with >= 10 complete rows in BOTH arms), weighted by pooled n:")
    for g in (2, 3):
        b12, b24 = defaultdict(list), defaultdict(list)
        for r in arms["R1200"]:
            if r["grade"] == g and r["finish_reason"] != "length":
                b12[len(r["text"]) // 200].append(r["dup"])
        for r in arms["R2400"]:
            if r["grade"] == g and r["finish_reason"] != "length":
                b24[len(r["text"]) // 200].append(r["dup"])
        common = [b for b in sorted(set(b12) & set(b24)) if len(b12[b]) >= 10 and len(b24[b]) >= 10]
        if not common:
            print(f"      G{g}: no common support")
            continue
        wt = {b: len(b12[b]) + len(b24[b]) for b in common}
        tot = sum(wt.values())
        e12 = sum(wt[b] * mean(b12[b]) for b in common) / tot
        e24 = sum(wt[b] * mean(b24[b]) for b in common) / tot
        print(f"      G{g}: bins {common}; length-adjusted dup  R1200 {e12:.2f}  R2400 {e24:.2f}  "
              f"(difference {e24 - e12:+.2f}; raw difference {mean([r['dup'] for r in arms['R2400'] if r['grade'] == g]) - mean([r['dup'] for r in arms['R1200'] if r['grade'] == g]):+.2f})")

    print("\n[3] per-puzzle paired comparison on dup (upper median over 3 reps), R2400 vs R1200")
    for g in (1, 2, 3):
        m12, m24 = {}, {}
        for r in arms["R1200"]:
            m12.setdefault((r["grade"], r["puzzle"]), []).append(r["dup"])
        for r in arms["R2400"]:
            m24.setdefault((r["grade"], r["puzzle"]), []).append(r["dup"])
        ks = [k for k in m12 if k[0] == g]
        w = sum(1 for k in ks if upper_median(m24[k]) > upper_median(m12[k]))
        l = sum(1 for k in ks if upper_median(m24[k]) < upper_median(m12[k]))
        print(f"    G{g}: R2400 higher {w} / R1200 higher {l} / equal {len(ks) - w - l}; sign p = {sign_p(w, l):.3g}")

    print("\n[4] does the extra dup sit on the gold answer's rank? (rank unchanged => the commitment is the same)")
    for label, rw in arms.items():
        for g in (1, 2, 3):
            s = [r for r in rw if r["grade"] == g]
            hi = [r for r in s if r["dup"] >= 2]
            print(f"    {label} G{g}: rows with dup>=2: {len(hi)}/{len(s)}; top-1 among them "
                  f"{sum(1 for r in hi if r['rank'] == 1)}/{len(hi)}; top-1 among dup<2 "
                  f"{sum(1 for r in s if r['dup'] < 2 and r['rank'] == 1)}/{sum(1 for r in s if r['dup'] < 2)}")


if __name__ == "__main__":
    main()
