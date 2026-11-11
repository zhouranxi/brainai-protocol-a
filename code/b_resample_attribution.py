#!/usr/bin/env python3
"""P6 Arm A -- b-resample attribution (ADR-0016 Appendix 34).

Question: how much of s0's b=0.515 shape outlier (vs the tight s1-s4
band 0.264-0.347) was leveraged by the single collapsed lambda=0.3
point that P5 (Appendix 33) proved to be run-level noise?

Method (frozen before any computation, Appendix 34):
  A1  Replace the archived s0 hi lambda=0.3 point with each of the 8
      P5 rerun readings (run-seed 1..8); refit the 5-point power law
      b (same fitting convention as Appendix 29 M3: OLS on
      (log d_delta, log |c_delta|), hi recipe, full 5 points).
      Frozen verdict: >=6/8 replacement b inside [0.264, 0.347]
      -> single-run sampling artifact; <=2/8 -> base-level real
      difference; 3-5/8 -> mixed, keep A1 wording with a note.
  A2  n=9 sensitivity fit (collapsed point + 8 reruns kept together).
      Directional check only: b < 0.40 supports an artifact verdict;
      b >= 0.40 together with an A1 artifact verdict -> report the
      conflict verbatim.

Read-only over runs/; results land in
results/p4_dose/b_resample_attribution.json.

Judgement rules FROZEN in Appendix 34 before this script ran; this
file implements them verbatim and adds nothing.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "p4_dose" / "dose_scan_seeds_v2.json"
P5_DIR = ROOT / "results" / "p5_collapse"
OUT = ROOT / "results" / "p4_dose" / "b_resample_attribution.json"

B_ORIG = 0.515          # Appendix 30 A1 table (5-pt full fit)
BAND_LO, BAND_HI = 0.264, 0.347   # s1-s4 tight band (frozen)
BAND_MEDIAN = 0.325
SENS_LIMIT = 0.40       # A2 sensitivity limit (frozen)


def fit_power(points):
    """OLS slope of log(d_delta) vs log(|c_delta|); same as App. 29 M3."""
    pts = [(math.log(p["d_delta"]), math.log(abs(p["c_delta"])))
           for p in points if p["c_delta"] != 0 and p["d_delta"] > 0]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / sxx


def r2_of(points, b):
    pts = [(math.log(p["d_delta"]), math.log(abs(p["c_delta"])))
           for p in points if p["c_delta"] != 0 and p["d_delta"] > 0]
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    ss_res = sum((y - (my + b * (x - mx))) ** 2 for x, y in pts)
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main() -> int:
    data = json.loads(V2.read_text(encoding="utf-8"))
    s0_hi = sorted([p for p in data["points"]
                    if p["recipe"] == "hi" and p["seed"] == 0],
                   key=lambda q: q["lambda"])
    assert len(s0_hi) == 5, f"expected 5 s0 hi points, got {len(s0_hi)}"
    base_pt = next(p for p in s0_hi if p["lambda"] == 0.3)

    reruns = []
    for r in range(1, 9):
        row = json.loads((P5_DIR / f"collapse_c1_seed{r}.json")
                         .read_text(encoding="utf-8"))["rows"][0]
        reruns.append({
            "run_seed": r,
            "d_delta": round(row["d_ent_id"] - base_pt["base"]["d_ent_id"], 3),
            "c_delta": round(row["c100_probe"] - base_pt["base"]["c100_probe"], 4),
        })

    # ---- A1: 8 replacement fits ----------------------------------------
    others = [p for p in s0_hi if p["lambda"] != 0.3]
    rows, bs = [], []
    for rr in reruns:
        curve = others + [{"d_delta": rr["d_delta"], "c_delta": rr["c_delta"]}]
        b = fit_power(curve)
        bs.append(b)
        rows.append({"run_seed": rr["run_seed"],
                     "lam03_d_delta": rr["d_delta"],
                     "lam03_c_delta": rr["c_delta"],
                     "b_replaced": round(b, 3),
                     "r2": round(r2_of(curve, b), 3),
                     "in_band": BAND_LO <= b <= BAND_HI})
    n_in = sum(r["in_band"] for r in rows)
    if n_in >= 6:
        a1_verdict = ("SAMPLING ARTIFACT: >=6/8 replacement b fall inside "
                      f"[{BAND_LO}, {BAND_HI}] -- s0's outlier was driven "
                      "by the collapsed lambda=0.3 point; M3/A1 wording "
                      "downgrades to 'b stable across seeds incl. s0'.")
    elif n_in <= 2:
        a1_verdict = ("BASE-LEVEL DIFFERENCE: <=2/8 replacement b inside "
                      f"[{BAND_LO}, {BAND_HI}] -- s0 shape specificity "
                      "survives beyond one bad point; keep 's0 outlier "
                      "seed' wording.")
    else:
        a1_verdict = (f"MIXED: {n_in}/8 in band; keep A1 wording, add "
                      "'partially driven by the bad point' note.")

    # ---- A2: n=9 sensitivity fit ---------------------------------------
    curve9 = s0_hi + [{"d_delta": rr["d_delta"], "c_delta": rr["c_delta"]}
                      for rr in reruns]
    b9 = fit_power(curve9)
    a2 = {"b_9pt": round(b9, 3),
          "n": len(curve9),
          "below_sens_limit": b9 < SENS_LIMIT,
          "sens_limit": SENS_LIMIT,
          "note": "b<0.40 directionally supports an artifact verdict"}

    # ---- assemble --------------------------------------------------------
    artifact = a1_verdict.startswith("SAMPLING ARTIFACT")
    conflict = artifact and not a2["below_sens_limit"]
    result = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "script": "code/b_resample_attribution.py",
            "inputs": ["results/p4_dose/dose_scan_seeds_v2.json",
                       "results/p5_collapse/collapse_c1_seed{1..8}.json"],
            "frozen_rules": "ADR-0016 Appendix 34 (2026-11-11 17:12, "
                            "frozen before any computation)",
            "constants": {"b_orig": B_ORIG, "band": [BAND_LO, BAND_HI],
                          "band_median": BAND_MEDIAN,
                          "sens_limit": SENS_LIMIT},
        },
        "A1_replacement_fits": {
            "rows": rows,
            "n_in_band": n_in, "n_runs": len(rows),
            "b_values": [round(b, 3) for b in bs],
            "b_mean": round(sum(bs) / len(bs), 3),
            "verdict": a1_verdict,
        },
        "A2_nine_point_sensitivity": a2,
        "conflict_flag": conflict,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("A1 -- replacement b per P5 run-seed (band "
          f"[{BAND_LO}, {BAND_HI}]):")
    for r in rows:
        mark = "IN " if r["in_band"] else "out"
        print(f"  seed{r['run_seed']}: lam0.3 d={r['lam03_d_delta']:>7.2f} "
              f"c={r['lam03_c_delta']:+.4f} -> b={r['b_replaced']:.3f} "
              f"(r2={r['r2']:.3f}) [{mark}]")
    print(f"  -> {n_in}/8 in band | b_mean={result['A1_replacement_fits']['b_mean']}")
    print(f"  -> {a1_verdict}\n")
    print(f"A2 -- 9-pt sensitivity: b={a2['b_9pt']} "
          f"({'< ' if a2['below_sens_limit'] else '>='}0.40)")
    if conflict:
        print("  !! CONFLICT: A1 says artifact but 9-pt b >= 0.40 -- report verbatim.")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
