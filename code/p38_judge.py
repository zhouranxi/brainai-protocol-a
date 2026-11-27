#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P38 (ADR-0037) frozen read-out. Written before any P38 data existed.

Inputs (all under results/):
  arm A      p38_stateless/p38_a_t00_s0r3.json        M-a at temp 0, reps 0..2   (health gate + temp-0 baseline)
  arm B      p38_stateless/p38_a_t07_s3r4.json        M-a at temp 0.7, reps 3..6
  P31        p31_confirm/p31_race.json  (mech a)      M-a at temp 0.7, reps 0..2   -> pooled index 0..2
  R1200      p35_fulltext/p35_race_1200.json (mech a) M-a at temp 0.7, reps 0..2   -> pooled index 3..5
  arm B      (see above)                                            -> pooled index 6..9
Pool order is therefore P31, R1200, arm B: 10 draws per cell.
  M-c @ T0   p37_stability/p37_c_t00_r3.json  (rep 0; the three are byte-identical)
Reads only rank / top3 / finish_reason / text (text only for the determinism gate H1).

Frozen gates (ADR-0037 sec 5):
  H1  >= 95% of arm-A cells (108/180) have three byte-identical outputs, else arm A is VOID and we stop.
  H2  zero API errors / tracebacks.
  Q1  descriptive: R1 at temp 0 per grade (M-c@T0 vs M-a@T0), paired sign test. NO gate on stage 1.
  Q2  T-A if ANY grade has paired sign p < 0.05 AND |diff| >= 10 pp (M-a@0.7 majority-of-10 vs M-a@T0);
      T-0 otherwise. Per grade first; a single significant grade is enough (that is the question).
  Q3  descriptive: 2x2 (temperature x caliper) table of the early gain.
  Q4  descriptive: per-cell noise structure of M-a over 10 draws.
"""
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
A_ARM = RES / "p38_stateless" / "p38_a_t00_s0r3.json"
B_ARM = RES / "p38_stateless" / "p38_a_t07_s3r4.json"
P31 = RES / "p31_confirm" / "p31_race.json"
R1200 = RES / "p35_fulltext" / "p35_race_1200.json"
MC_T0 = RES / "p37_stability" / "p37_c_t00_r3.json"
GRADES = (1, 2, 3)


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * z and z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return (c - h) / d, (c + h) / d


def cells(rows, key):
    """(puzzle, grade) -> {rep: value}"""
    o = defaultdict(dict)
    for r in rows:
        o[(r["puzzle"], r["grade"])][r["rep"]] = r[key]
    return o


def top3_of(rank):
    return int(rank in (1, 2, 3))


def paired(c, a, g):
    """c, a: (puzzle, grade) -> bool. Returns (n, sum_c, sum_a, W, L, p)."""
    ks = [p for (p, gg) in c if gg == g and (p, gg) in a]
    w = sum(1 for p in ks if c[(p, g)] and not a[(p, g)])
    l = sum(1 for p in ks if a[(p, g)] and not c[(p, g)])
    return len(ks), sum(c[(p, g)] for p in ks), sum(a[(p, g)] for p in ks), w, l, sign_p(w, l)


def main():
    missing = [str(p) for p in (A_ARM, B_ARM, P31, R1200, MC_T0) if not p.exists()]
    if missing:
        print("missing inputs (not run yet):", missing)
        sys.exit(0)
    A = json.loads(A_ARM.read_text(encoding="utf-8"))["rows"]
    B = json.loads(B_ARM.read_text(encoding="utf-8"))["rows"]
    p31 = [r for r in json.loads(P31.read_text(encoding="utf-8"))["rows"] if r["mech"] == "a"]
    r12 = [r for r in json.loads(R1200.read_text(encoding="utf-8"))["rows"] if r["mech"] == "a"]
    mc0 = json.loads(MC_T0.read_text(encoding="utf-8"))["rows"]

    # ---------------- H2 / completeness ----------------
    print("== completeness ==")
    ok = True
    for name, rows, n_exp, reps in (("arm A (M-a, temp 0)", A, 540, set(range(0, 3))),
                                    ("arm B (M-a, temp 0.7, reps 3-6)", B, 720, set(range(3, 7))),
                                    ("P31 M-a", p31, 540, None), ("R1200 M-a", r12, 540, None)):
        keys = {(r["puzzle"], r["grade"], r["rep"]) for r in rows}
        reps_ok = reps is None or {r["rep"] for r in rows} == reps
        errs = sum(1 for r in rows if r.get("finish_reason") == "error")
        good = len(rows) == n_exp and len(keys) == n_exp and reps_ok and errs == 0
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {name}: {len(rows)} rows (expect {n_exp}), unique {len(keys)}, errors {errs}")
    h2 = ok
    print(f"H2 (completeness / no API errors): {'PASS' if h2 else 'FAIL'}")
    if not h2:
        print("\nVERDICT: H2 FAIL - incomplete or errored data; stop."); return

    # ---------------- H1 determinism ----------------
    print("\n== H1: arm A (temp 0) - are the three outputs of each cell byte-identical? ==")
    ca = cells(A, "text")
    ident = sum(1 for v in ca.values() if len({x for x in v.values()}) == 1)
    h1 = ident >= 0.95 * len(ca)
    print(f"  identical-text cells: {ident}/{len(ca)} = {ident/len(ca):.1%}   (gate >= 95% = 108/180)  -> {'PASS' if h1 else 'FAIL'}")
    if not h1:
        print("\nVERDICT: H1 FAIL - engine not deterministic for the stateless prompt; arm A VOID, stop."); return

    # ---------------- Q1: R1 at temp 0, descriptive ----------------
    # M-a at temp 0 = rep 0 of arm A (the three are identical by H1). M-c at temp 0 = rep 0 of P37 T0.
    a0 = {k: top3_of(v[0]) for k, v in cells(A, "rank").items()}
    c0 = {k: v[0] for k, v in cells(mc0, "top3").items()}
    print("\n== Q1 (descriptive, NO gate): R1 at temperature 0 - M-c top-3 recall vs M-a top-3 recall ==")
    print(f"  {'grade':<6}{'n':>4}{'M-c':>8}{'M-a':>8}{'gain':>9}{'W/L':>9}{'sign p':>9}")
    Q1 = {}
    for g in GRADES:
        n, sc, sa, w, l, p = paired(c0, a0, g)
        Q1[g] = (sc, sa, w, l, p)
        print(f"  G{g:<5}{n:>4}{sc/n:>8.0%}{sa/n:>8.0%}{(sc-sa)/n*100:>+8.0f}pp{f'{w}/{l}':>9}{p:>9.3f}")
    print("  (pre-registered: no gate at any grade; stage 1 has 10-38% power with 60 puzzles and is descriptive only)")

    # ---------------- pooled 10 draws of M-a at 0.7 ----------------
    # re-index: P31 -> 0..2, R1200 -> 3..5, arm B -> 6..9
    pooled = defaultdict(dict)
    for r in p31:
        pooled[(r["puzzle"], r["grade"])][r["rep"]] = top3_of(r["rank"])
    for r in r12:
        pooled[(r["puzzle"], r["grade"])][3 + r["rep"]] = top3_of(r["rank"])
    for r in B:
        pooled[(r["puzzle"], r["grade"])][6 + (r["rep"] - 3)] = top3_of(r["rank"])
    assert all(len(v) == 10 for v in pooled.values()), "pooled cells must have 10 draws"
    maj = {k: int(sum(v.values()) >= 6) for k, v in pooled.items()}       # majority-of-10 (>= 6/10)
    med = {k: int(sorted(v.values())[5] == 1) for k, v in pooled.items()}  # median-of-10 (upper median)

    # ---------------- Q2: did temperature move M-a's set-level reading? ----------------
    print("\n== Q2: M-a at temp 0.7 (majority of 10 draws) vs M-a at temp 0 - paired per grade ==")
    print(f"  {'grade':<6}{'M-a@0':>8}{'M-a@0.7':>9}{'diff':>9}{'W/L':>9}{'sign p':>9}   gate (p<.05 and |diff|>=10pp)")
    q2_hits = []
    for g in GRADES:
        n, s7, s0, w, l, p = paired(maj, a0, g)       # W = 0.7-only, L = 0-only
        diff = (s7 - s0) / n * 100
        hit = p < 0.05 and abs(diff) >= 10
        q2_hits.append((g, hit))
        print(f"  G{g:<5}{s0/n:>8.0%}{s7/n:>9.0%}{diff:>+8.0f}pp{f'{w}/{l}':>9}{p:>9.3f}   {'FIRES' if hit else '-'}")
    t_a = any(h for _, h in q2_hits)
    print(f"  VERDICT Q2: {'T-A (temperature moved the stateless reading at grade(s) ' + ', '.join(f'G{g}' for g, h in q2_hits if h) + ')' if t_a else 'T-0 (no shift detected at any grade)'}")

    # ---------------- Q3: 2x2 (temperature x caliper) ----------------
    print("\n== Q3 (descriptive): early gain M-c minus M-a, 2x2 temperature x caliper, per grade ==")
    t7 = json.loads((RES / "p37_stability" / "p37_c_t07_r3.json").read_text(encoding="utf-8"))["rows"]
    c7m = {k: int(sorted(v.values())[1] == 1) for k, v in cells(t7, "top3").items()}          # median-of-3
    p31c = defaultdict(dict)
    for r in json.loads(P31.read_text(encoding="utf-8"))["rows"]:
        if r["mech"] == "c":
            p31c[(r["puzzle"], r["grade"])][r["rep"]] = top3_of(r["rank"])
    c731 = {k: int(sorted(v.values())[1] == 1) for k, v in p31c.items()}                     # the confirmation batch's own M-c
    a7m = {k: int(sorted(v[i] for i in (0, 1, 2))[1] == 1) for k, v in pooled.items()}        # P31's M-a (median-of-3)
    a_r12 = {k: int(sorted(v[i] for i in (3, 4, 5))[1] == 1) for k, v in pooled.items()}      # R1200's M-a, median-of-3
    CALIPERS = (("T0 / single draw", c0, a0),
                ("T0.7 / P31 M-c vs P31 M-a, median-3", c731, a7m),
                ("T0.7 / P37-T7 M-c vs R1200 M-a, median-3", c7m, a_r12),
                ("T0.7 / P37-T7 M-c vs pooled M-a, majority-10", c7m, maj))
    for g in GRADES:
        print(f"  G{g}:")
        for lab, c, a in CALIPERS:
            n, sc, sa, w, l, p = paired(c, a, g)
            print(f"       {lab:<46} {(sc-sa)/n*100:+5.0f}pp   ({w}/{l}, p={p:.2f})")
    print("  (each cell uses a different draw of M-a / M-c; differences across cells of a row are sampling spread, not effects)")

    # ---------------- Q4: noise structure of M-a over 10 draws ----------------
    print("\n== Q4 (descriptive): M-a at temp 0.7 - gold-in-top-3 count per cell out of 10 draws ==")
    for g in GRADES:
        h = Counter(sum(v.values()) for (p, gg), v in pooled.items() if gg == g)
        unstable = sum(c for k, c in h.items() if 0 < k < 10)
        biased = sum(c for k, c in h.items() if k in (1, 2, 8, 9))
        fair = sum(c for k, c in h.items() if 3 <= k <= 7)
        print(f"  G{g}: {dict(sorted(h.items()))}")
        print(f"       unstable cells {unstable}/60 = {unstable/60:.0%}  (biased 1-2/8-9: {biased}, fair 3-7: {fair})")
    # flip(3) for arm B alone, to compare with P37's T7-b/T7 numbers on the same footing
    print("\n  flip(3) of M-a using P31's three draws (the same caliper as P37):")
    for g in GRADES:
        p31c_cells = {k: v for k, v in cells(p31, "rank").items() if k[1] == g}
        f = sum(1 for v in p31c_cells.values() if len({top3_of(x) for x in v.values()}) > 1)
        print(f"    G{g}: top-3 flip(3) {f}/60 = {f/60:.0%}")

    out = {"H1": "PASS" if h1 else "FAIL", "H2": "PASS" if h2 else "FAIL", "Q2": "T-A" if t_a else "T-0",
           "Q1": {f"G{g}": {"c": Q1[g][0], "a": Q1[g][1], "W": Q1[g][2], "L": Q1[g][3], "p": round(Q1[g][4], 4)} for g in GRADES}}
    (RES / "p38_stateless" / "readout.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nVERDICTS:", json.dumps(out))


if __name__ == "__main__":
    main()
