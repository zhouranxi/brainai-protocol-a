#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic self-test of code/p38_judge.py (ADR-0037 sec 10). Run BEFORE launch.

Strategy: keep the REAL P31 / R1200 / P37 M-c files (so the pooling and the M-c baselines are exercised
exactly as they will be at read-out time) and generate only the two P38 arms synthetically:
  arm A (M-a, temp 0): built from a per-cell 'truth' with a controllable fraction of non-identical cells
  arm B (M-a, temp 0.7, reps 3..6): Bernoulli draws around a per-cell probability
Each scenario plants a known truth and checks the frozen read-out returns it."""
import contextlib
import importlib.util
import io
import json
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pj", str(HERE / "p38_judge.py"))
pj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pj)

# use the REAL puzzle ids so the synthetic arms align with the real R1200 / P37 files that the judge pools in
_real = json.loads((pj.RES / "p37_stability" / "p37_c_t00_r3.json").read_text(encoding="utf-8"))["rows"]
PZ = sorted({r["puzzle"] for r in _real})
assert len(PZ) == 60, PZ
KEYS = [(p, g) for p in PZ for g in (1, 2, 3)]


def rows_from(cells_map, mech="a", temp=0.7, rep_ids=(0, 1, 2), text_fn=None):
    rows = []
    for (p, g), vals in cells_map.items():
        for rep, top3 in zip(rep_ids, vals):
            # the text must NOT depend on the rep index: at temp 0 identical outputs are what H1 looks for.
            # A cell whose draws disagree gets a distinct text per distinct VALUE (a real engine would too).
            t = text_fn(p, g, rep, top3) if text_fn else f"{p}-G{g}-{'hit' if top3 else 'miss'}"
            rows.append({"puzzle": p, "mech": mech, "grade": g, "rep": rep, "temp": temp, "rank": 1 if top3 else 0,
                         "top3": int(top3), "dup": 0, "difficulty": "medium", "secs": 1.0, "finish_reason": "stop", "text": t})
    return rows


def make_p31_like(truth, seed):
    """three Bernoulli(0.9)-around-truth draws per cell (the real P31 has 3 draws at 0.7)."""
    rnd = random.Random(seed)
    return {k: [int(rnd.random() < (0.9 if truth[k] else 0.1)) for _ in range(3)] for k in KEYS}


def run(name, armA_cells, armB_cells, p31_cells, expect, forbid=()):
    d = Path(tempfile.mkdtemp())
    real = pj.RES
    # copy the real reference files, swap only P38's two arms and P31's M-a (so pooling is realistic)
    (d / "p31_confirm").mkdir(); (d / "p35_fulltext").mkdir(); (d / "p37_stability").mkdir(); (d / "p38_stateless").mkdir()
    p31 = json.loads((real / "p31_confirm" / "p31_race.json").read_text(encoding="utf-8"))
    keep = [r for r in p31["rows"] if r["mech"] != "a"]
    keep += rows_from(p31_cells)
    (d / "p31_confirm" / "p31_race.json").write_text(json.dumps({"rows": keep}, ensure_ascii=False), encoding="utf-8")
    for f in ("p35_fulltext/p35_race_1200.json", "p37_stability/p37_c_t00_r3.json", "p37_stability/p37_c_t07_r3.json"):
        (d / f).write_bytes((real / f).read_bytes())
    (d / "p38_stateless" / "p38_a_t00_s0r3.json").write_text(json.dumps({"rows": rows_from(armA_cells, temp=0.0)}, ensure_ascii=False), encoding="utf-8")
    (d / "p38_stateless" / "p38_a_t07_s3r4.json").write_text(json.dumps({"rows": rows_from(armB_cells, rep_ids=(3, 4, 5, 6))}, ensure_ascii=False), encoding="utf-8")
    pj.A_ARM, pj.B_ARM, pj.P31 = d / "p38_stateless" / "p38_a_t00_s0r3.json", d / "p38_stateless" / "p38_a_t07_s3r4.json", d / "p31_confirm" / "p31_race.json"
    pj.R1200, pj.MC_T0 = d / "p35_fulltext" / "p35_race_1200.json", d / "p37_stability" / "p37_c_t00_r3.json"
    pj.RES = d
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pj.main()
    out = buf.getvalue()
    ok = all(s in out for s in expect) and not any(s in out for s in forbid)
    print(f"  [{'OK' if ok else 'MISMATCH'}] {name}")
    if not ok:
        print(out)
    return ok


if __name__ == "__main__":
    rnd = random.Random(1)
    # a 'truth' per cell: hit-rate 0.5 / 0.45 / 0.75 by grade, like the real M-a
    TRUTH = {k: rnd.random() < {1: 0.5, 2: 0.45, 3: 0.75}[k[1]] for k in KEYS}
    ok = True
    # ---- H1: deterministic engine (identical texts) -> PASS ; noisy engine -> FAIL and stop
    det = {k: [int(TRUTH[k])] * 3 for k in KEYS}
    noisy = {k: ([int(TRUTH[k])] * 3 if rnd.random() < 0.85 else [int(TRUTH[k]), 1 - int(TRUTH[k]), int(TRUTH[k])]) for k in KEYS}
    ok &= run("H1 clean: identical texts -> H1 PASS, Q2 readable",
              det, {k: [int(rnd.random() < (0.9 if TRUTH[k] else 0.1)) for _ in range(4)] for k in KEYS}, make_p31_like(TRUTH, 7),
              ["H1 (determinism): PASS"] if False else ["identical-text cells: 180/180 = 100.0%", "PASS", "VERDICT Q2"])
    ok &= run("H1 noisy: 15% non-identical cells -> H1 FAIL and stop",
              noisy, {k: [int(TRUTH[k])] * 4 for k in KEYS}, make_p31_like(TRUTH, 8),
              ["H1 FAIL", "arm A VOID"], forbid=["VERDICT Q2"])
    # ---- Q2: temperature moves ONLY grade 1 by >=10 pp with p<.05 -> T-A
    moved = {k: [int(TRUTH[k])] * 3 for k in KEYS}
    hits = [k for k in KEYS if k[1] == 1 and not TRUTH[k]][:12]           # 12 stage-1 cells that temp 0 misses
    for k in hits:
        moved[k] = [1, 1, 1]                                               # ... but temp 0.7 hits them
    armB_moved = {k: [int(moved[k][0])] * 4 if k in hits else [int(TRUTH[k])] * 4 for k in KEYS}
    p31_moved = {k: [int(moved[k][0])] * 3 if k in hits else [int(TRUTH[k])] * 3 for k in KEYS}
    ok &= run("Q2 T-A: 12 stage-1 cells flip under 0.7 only (0/12 at temp 0) -> T-A at G1",
              {k: [int(TRUTH[k])] * 3 for k in KEYS}, armB_moved, p31_moved,
              ["FIRES", "VERDICT Q2: T-A (temperature moved the stateless reading at grade(s) G1)"])
    # ---- Q2: no shift, only sampling noise -> T-0
    ok &= run("Q2 T-0: temp 0.7 draws scatter around the same truth -> T-0",
              {k: [int(TRUTH[k])] * 3 for k in KEYS}, {k: [int(rnd.random() < (0.85 if TRUTH[k] else 0.15)) for _ in range(4)] for k in KEYS},
              make_p31_like(TRUTH, 9), ["VERDICT Q2: T-0"], forbid=["FIRES"])
    # ---- Q2 opposite-sign case: G1 +8 cells, G3 -8 cells. Pooled ~0 but EACH grade is 13 pp with p=.008.
    # The frozen rule is "any single grade firing is T-A", so this MUST be T-A (not a trap under this pre-registration).
    opp = dict(TRUTH)
    up = [k for k in KEYS if k[1] == 1 and not TRUTH[k]][:8]; dn = [k for k in KEYS if k[1] == 3 and TRUTH[k]][:8]
    for k in up: opp[k] = True
    for k in dn: opp[k] = False
    ok &= run("Q2 opposite signs: +8 cells at G1, -8 at G3 (pooled ~0; each grade 13pp, p=.008) -> T-A by the frozen single-grade rule",
              {k: [int(TRUTH[k])] * 3 for k in KEYS}, {k: [int(opp[k])] * 4 for k in KEYS}, {k: [int(opp[k])] * 3 for k in KEYS},
              ["VERDICT Q2: T-A"], forbid=["VERDICT Q2: T-0"])
    # ---- the real Simpson trap: every grade moves the SAME way but by <10 pp each (pooled ~+10 pp) -> T-0
    simp = dict(TRUTH)
    for g in (1, 2, 3):
        for k in [k for k in KEYS if k[1] == g and not TRUTH[k]][:4]: simp[k] = True
    ok &= run("Q2 Simpson trap: +4 cells at every grade (pooled +7pp, no grade reaches 10pp) -> T-0",
              {k: [int(TRUTH[k])] * 3 for k in KEYS}, {k: [int(simp[k])] * 4 for k in KEYS}, {k: [int(simp[k])] * 3 for k in KEYS},
              ["VERDICT Q2: T-0"], forbid=["FIRES"])
    print("\nALL SELF-TESTS PASS" if ok else "\nSELF-TEST FAILURES")
    sys.exit(0 if ok else 1)
