#!/usr/bin/env python3
"""P6 Arm D readout (ADR-0016 Appendix 34, D1 frozen rule).

Assemble s0's second full 5-point curve (run-seed 1): 4 fresh runs
(lambda 0.03/0.1/1.0/3.0) + P5's collapse_c1_seed1 (lambda 0.3).
Fit b with the frozen App.29 M3 convention, apply the frozen D1 rule:
  D-b in [0.26, 0.35] -> base-level difference refuted (Arm A reverses)
  D-b >= 0.40         -> base-level difference confirmed, close
  D-b in (0.35, 0.40) -> edge band, report as-is
Also: id_keep gate (>= -0.02, all 5 runs), and a side-by-side of the
lambda=0.03 response (the Arm A suspect point) vs s1-s4 band.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_TBL = ROOT / "results" / "p4_full" / "full_base_lr0.125.json"
P5_DIR = ROOT / "results" / "p5_collapse"
D_DIR = ROOT / "results" / "p6_d_curve"
OUT = ROOT / "results" / "p6_d_curve" / "p6_d_readout.json"

S_OBS = 0.0161
BAND_LO, BAND_HI = 0.264, 0.347   # frozen D1 band
MODEL = "resnet18_lr0.125_wd0.0001_s0"


def fit_power(points):
    pts = [(math.log(p["d_delta"]), math.log(abs(p["c_delta"])))
           for p in points if p["c_delta"] != 0 and p["d_delta"] > 0]
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / sxx


def main() -> int:
    base = next(r for r in json.loads(BASE_TBL.read_text(encoding="utf-8"))["rows"]
                if r["model"] == MODEL)
    rows = []
    for lam in (0.03, 0.1, 0.3, 1.0, 3.0):
        if lam == 0.3:
            run = json.loads((P5_DIR / "collapse_c1_seed1.json")
                             .read_text(encoding="utf-8"))["rows"][0]
            src = "p5_collapse_c1_seed1"
        else:
            run = json.loads((D_DIR / f"p6_d_s0_seed1_lam{lam}.json")
                             .read_text(encoding="utf-8"))["rows"][0]
            src = f"p6_d_lam{lam}"
        dd = round(run["d_ent_id"] - base["d_ent_id"], 3)
        dc = round(run["c100_probe"] - base["c100_probe"], 4)
        ik = round(run["id_probe_acc"] - base["id_probe_acc"], 4)
        rows.append({"lambda": lam, "src": src, "d_delta": dd, "c_delta": dc,
                     "id_keep": ik, "pass_gate": ik >= -0.02,
                     "secant_pct": round(100 * (dc / dd) / S_OBS, 1)})
    gate_ok = all(r["pass_gate"] for r in rows)
    fit_rows = [r for r in rows if r["pass_gate"]]
    b = fit_power(fit_rows)
    r2 = None
    n = len(fit_rows)
    ys = [math.log(abs(r["c_delta"])) for r in fit_rows]
    xs = [math.log(r["d_delta"]) for r in fit_rows]
    my = sum(ys) / n
    mx = sum(xs) / n
    ss_res = sum((y - (my + b * (x - mx))) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if BAND_LO <= b <= BAND_HI:
        d1 = ("REFUTED: D-b in [0.26,0.35] -- s0 base-level shape "
              "difference is refuted; the original curve was a bad "
              "sample; Arm A verdict reverses, Appendix 24's b=0.502 "
              "downgrades to a leveraged artifact of that sample.")
    elif b >= 0.40:
        d1 = ("CONFIRMED: D-b >= 0.40 -- s0 base-level shape difference "
              "stands; close the loop (Arm A verdict kept).")
    else:
        d1 = (f"EDGE: D-b={b:.3f} in (0.35,0.40) -- report as-is.")

    lam03 = next(r for r in rows if r["lambda"] == 0.03)
    result = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "script": "code/p6_d_readout.py",
            "frozen_rules": "ADR-0016 Appendix 34 D1 (frozen before launch)",
            "constants": {"band": [BAND_LO, BAND_HI], "s_obs": S_OBS,
                          "id_floor": -0.02},
        },
        "curve_rows": rows,
        "D1": {"b_runseed1_5pt": round(b, 3), "r2": round(r2, 3),
               "n_fitted": len(fit_rows), "gate_all_pass": gate_ok,
               "verdict": d1},
        "lambda03_response": {k: lam03[k] for k in
                              ("d_delta", "c_delta", "id_keep", "secant_pct")},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"{'lam':>5} {'dd':>8} {'dc':>7} {'id_keep':>8} {'sec%':>5} gate")
    for r in rows:
        print(f"{r['lambda']:>5} {r['d_delta']:>8.3f} {r['c_delta']:>+7.4f} "
              f"{r['id_keep']:>8.4f} {r['secant_pct']:>5.1f} "
              f"{'ok' if r['pass_gate'] else 'FAIL'}")
    print(f"\nD1: b(run-seed1, 5pt) = {b:.3f} (r2={r2:.3f}) "
          f"vs band [{BAND_LO},{BAND_HI}]")
    print(f"-> {d1}")
    print(f"\nlambda=0.03 response: dd={lam03['d_delta']} "
          f"dc={lam03['c_delta']:+.4f} secant={lam03['secant_pct']}% "
          f"(placebo band 0.022; s1-s4 same point dc=0.037..0.073)")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
