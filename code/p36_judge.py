#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P36 (ADR-0035) gate read-out. Merges the blinded judgements with the key ONLY here.
Verdict codes: C = commitment before the cut, N = no commitment.
'amb' rows are reported as a sensitivity band (counted as C for the lower bound of 'no commitment',
as N for the upper bound); the point estimate uses the recorded verdict.
Gate (frozen): R-C1 if Wilson 95% CI lower bound of no-commitment share (R1200) >= 25%;
               R-C0 if CI upper bound < 25%; otherwise report as-is, no top-up."""
import json, math, sys
from collections import Counter
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
D = REPO / "results" / "p36_commit"

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

def main():
    J = [json.loads(l) for l in (D / "judgments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    key = json.loads((D / "key.json").read_text(encoding="utf-8"))
    ids = [j["id"] for j in J]
    assert len(ids) == len(set(ids)) == 90, f"expected 90 unique judgements, got {len(ids)}"
    assert set(ids) == set(key), "judgement ids and key ids differ"
    for arm, n_exp in (("R1200", 60), ("R2400", 30)):
        rows = [j for j in J if j["id"].startswith(arm)]
        assert len(rows) == n_exp, f"{arm}: expected {n_exp}, got {len(rows)}"
        for j in rows: j.update(key[j["id"]])
    out = {}
    for arm in ("R1200", "R2400"):
        rows = [j for j in J if j["id"].startswith(arm)]
        n = len(rows); nc = sum(1 for j in rows if j["verdict"] == "N"); amb = sum(1 for j in rows if j.get("amb"))
        lo, hi = wilson(nc, n)
        ambN = sum(1 for j in rows if j.get("amb") and j["verdict"] == "N"); ambC = amb - ambN
        print(f"=== {arm}: truncated M-d rows, n={n} ===")
        print(f"  no commitment before the cut: {nc}/{n} = {nc/n:.1%}   Wilson 95% CI [{lo:.1%}, {hi:.1%}]")
        print(f"  ambiguous rows: {amb} (recorded N: {ambN}, recorded C: {ambC}) -> sensitivity band [{(nc-ambN)/n:.1%}, {(nc+ambC)/n:.1%}]")
        by = Counter(("U" + __import__('re').match(r"U(\d)", j["rule"]).group(1)) for j in rows if j["verdict"] == "N")
        print(f"  no-commitment by rule: {dict(sorted(by.items()))}")
        print(f"  commitment layouts: {dict(Counter(j['layout'] for j in rows if j['verdict']=='C'))}")
        rk = Counter(j["rank"] for j in rows if j["verdict"] == "N"); rc = Counter(j["rank"] for j in rows if j["verdict"] == "C")
        t1n = sum(1 for j in rows if j["verdict"] == "N" and j["rank"] == 1); t1c = sum(1 for j in rows if j["verdict"] == "C" and j["rank"] == 1)
        print(f"  recorded rank among NO-commitment rows: {dict(sorted(rk.items()))}  -> rank==1: {t1n}/{nc} = {t1n/nc:.0%}" if nc else "  (none)")
        print(f"  recorded rank among commitment rows:    {dict(sorted(rc.items()))}  -> rank==1: {t1c}/{n-nc} = {t1c/(n-nc):.0%}")
        print(f"  share of all rank==1 rows in the sample that had NO commitment: {t1n}/{t1n+t1c} = {t1n/(t1n+t1c):.0%}")
        print(f"  grade mix (N rows): {dict(sorted(Counter(j['grade'] for j in rows if j['verdict']=='N').items()))} | (C rows): {dict(sorted(Counter(j['grade'] for j in rows if j['verdict']=='C').items()))}")
        out[arm] = {"n": n, "no_commit": nc, "lo": lo, "hi": hi, "amb": amb, "t1_no_commit": t1n, "t1_commit": t1c}
    print("\n=== frozen gate (R1200) ===")
    r = out["R1200"]
    if r["lo"] >= 0.25: v = "R-C1"
    elif r["hi"] < 0.25: v = "R-C0"
    else: v = "INDETERMINATE (no top-up allowed by ADR-0035 sec 3)"
    print(f"  no-commitment {r['no_commit']}/{r['n']} = {r['no_commit']/r['n']:.1%}, Wilson CI [{r['lo']:.1%}, {r['hi']:.1%}]")
    print(f"  VERDICT: {v}")
    (D / "readout.json").write_text(json.dumps({"verdict": v, **{k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v2.items()} for k, v2 in out.items()},
                                                "judgements": J}, ensure_ascii=False, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
