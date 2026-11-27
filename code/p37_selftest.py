#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic-data self-test of code/p37_judge.py (REVISED gates: 3 reps, S-4 descriptive, S-4' top-3).
Each scenario plants a known truth and checks the frozen read-out returns it.
Run BEFORE launch. Exit code 0 = all scenarios behave as expected."""
import json, random, sys, importlib.util, math, tempfile, shutil, io, contextlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pj", str(HERE / "p37_judge.py"))
pj = importlib.util.module_from_spec(spec); spec.loader.exec_module(pj)
REPS = 3
PZ = [f"P{i:02d}" for i in range(60)]


def fisher_ref(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    pr = lambda x: math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1); p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 + 1e-12))


def make(seed, p_flip, truth, p_bias=0.5, top3_flip=0.0):
    """Per (puzzle, grade): with prob p_flip the REPS draws are mixed (each a p_bias coin around
    the truth), else all equal the truth. top3 is stable unless top3_flip > 0."""
    rnd = random.Random(seed); rows = []
    for (p, g), t in truth.items():
        mixed = rnd.random() < p_flip[g]
        reps = [int(rnd.random() < p_bias) if t else int(rnd.random() > p_bias) for _ in range(REPS)] if mixed else [int(t)] * REPS
        if mixed and len(set(reps)) == 1: reps[rnd.randrange(REPS)] = 1 - reps[0]
        t3mix = rnd.random() < top3_flip
        for k, v in enumerate(reps):
            rows.append({"puzzle": p, "mech": "c", "grade": g, "rep": k, "temp": 0, "rank": 1 if v else 3,
                         "top3": int(t3mix and k == 0 and v == 0) ^ 1 if t3mix else 1,
                         "dup": 0, "difficulty": "medium", "secs": 1.0, "finish_reason": "stop", "text": ""})
    return rows


def run_scenario(name, t0, t7, t7b, expect):
    d = Path(tempfile.mkdtemp())
    for arm, rows in (("T0", t0), ("T7", t7), ("T7b", t7b)):
        (d / pj.ARMS[arm]).write_text(json.dumps({"rows": rows}), encoding="utf-8")
    pj.D = d
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pj.main()
    got = json.loads((d / "readout.json").read_text()) if (d / "readout.json").exists() else {"S1": "FAIL"}
    shutil.rmtree(d)
    ok = got == expect
    print(f"  [{'OK' if ok else 'MISMATCH'}] {name}: expected {expect}\n        got {got}")
    if not ok: print(buf.getvalue())
    return ok


if __name__ == "__main__":
    for t in [(30, 30, 2, 58), (50, 10, 20, 40), (5, 55, 5, 55), (1, 59, 0, 60)]:
        assert abs(pj.fisher(*t) - fisher_ref(*t)) < 1e-12, t
    print("fisher() matches brute-force reference on 4 tables")
    rnd = random.Random(1); TRUTH = {(p, g): rnd.random() < 0.6 for p in PZ for g in (1, 2, 3)}
    Z = {1: 0, 2: 0, 3: 0}; HI = {1: 0.5, 2: 0.6, 3: 0.6}; LO = {1: 0.1, 2: 0.1, 3: 0.1}
    P = "PASS"; N = "NOT DETECTED"
    ok = True
    ok &= run_scenario("A clean floor + big temp & format effects + stable top-3",
                       make(1, Z, TRUTH), make(2, HI, TRUTH), make(3, LO, TRUTH),
                       {"S1": P, "S2": P, "S3": P, "S4p_top3": P, "S4_majority": "DESCRIPTIVE"})
    ok &= run_scenario("B noisy engine at temp 0 -> batch INVALID, later gates not read",
                       make(4, {1: .2, 2: .2, 3: .2}, TRUTH), make(5, HI, TRUTH), make(6, LO, TRUTH),
                       {"S1": "FAIL"})
    ok &= run_scenario("C no temperature effect", make(7, Z, TRUTH), make(8, {1: .05, 2: .1, 3: .1}, TRUTH), make(9, LO, TRUTH),
                       {"S1": P, "S2": N, "S3": N, "S4p_top3": P, "S4_majority": "DESCRIPTIVE"})
    t7d = make(10, HI, TRUTH)
    for r in t7d:
        if r["grade"] == 2: r["rank"] = 1     # G2 perfectly stable at temp 0.7
    ok &= run_scenario("D Simpson trap: pooled S2 large, one grade flat -> NOT DETECTED",
                       make(11, Z, TRUTH), t7d, make(12, LO, TRUTH),
                       {"S1": P, "S2": N, "S3": N, "S4p_top3": P, "S4_majority": "DESCRIPTIVE"})
    ok &= run_scenario("E top-3 recall is noisy too -> S-4' NOT DETECTED",
                       make(13, Z, TRUTH), make(14, HI, TRUTH, top3_flip=0.5), make(15, LO, TRUTH),
                       {"S1": P, "S2": P, "S3": P, "S4p_top3": N, "S4_majority": "DESCRIPTIVE"})
    t7f = make(17, HI, TRUTH); rf = random.Random(170)
    byp = {}
    for r in t7f:
        if r["grade"] == 3: byp.setdefault(r["puzzle"], []).append(r)
    for pz_, rs in byp.items():          # 30% of G3 puzzles: top-3 drops in rep 0 only
        if rf.random() < 0.30: rs[0]["top3"] = 0
    ok &= run_scenario("F top-3 noisy in ONE grade only (30%) -> S-4' NOT DETECTED (per-grade cap)",
                       make(16, Z, TRUTH), t7f, make(18, LO, TRUTH),
                       {"S1": P, "S2": P, "S3": P, "S4p_top3": N, "S4_majority": "DESCRIPTIVE"})
    print("\nALL SELF-TESTS PASS" if ok else "\nSELF-TEST FAILURES")
    sys.exit(0 if ok else 1)
