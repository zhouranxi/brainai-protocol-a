#!/usr/bin/env python3
"""P7 Arm E readout (ADR-0016 Appendix 35, E1/E2/E3 frozen rules).

Assemble run-seed-1 full 5-point curves for s1 and s2, fit b with the
frozen App.29 M3 convention, and adjudicate band stability:

  E1 (main): both b_seed1 inside B=[0.264,0.347] -> band is run-stable,
             Appendix 34 verdict STANDS;
             both outside -> band is pseudo-tight, verdict REVERSES;
             one in / one out -> mixed, wording downgrades.
  E2 (descriptive): lambda=0.03 responses vs placebo band and the
             archived s1/s2 values (0.0679 / 0.0623).
  E3: id_keep >= -0.02 gate; failing points dropped to a 4-pt fit.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "p4_dose" / "dose_scan_seeds_v2.json"
BASE_TBL = ROOT / "results" / "p4_full" / "full_base_lr0.125.json"
E_DIR = ROOT / "results" / "p7_band"
OUT = E_DIR / "p7_band_readout.json"

S_OBS = 0.0161
BAND_LO, BAND_HI = 0.264, 0.347     # frozen four-seed band
B_ORIG = {1: 0.325, 2: 0.264}       # archived b (App.30 A1 table)
LAM03_ORIG = {1: 0.0679, 2: 0.0623} # archived lambda=0.03 dc
PLACEBO = 0.022
ID_FLOOR = -0.02
LAMS = (0.03, 0.1, 0.3, 1.0, 3.0)


def fit_power(points):
    pts = [(math.log(p["d_delta"]), math.log(abs(p["c_delta"])))
           for p in points if p["c_delta"] != 0 and p["d_delta"] > 0]
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
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
    bases = {r["model"]: r for r in
             json.loads(BASE_TBL.read_text(encoding="utf-8"))["rows"]}
    seeds, e2_rows = {}, []
    for s in (1, 2):
        model = f"resnet18_lr0.125_wd0.0001_s{s}"
        base = bases[model]
        rows = []
        for lam in LAMS:
            run = json.loads((E_DIR / f"p7_e_s{s}_seed1_lam{lam}.json")
                             .read_text(encoding="utf-8"))["rows"][0]
            dd = round(run["d_ent_id"] - base["d_ent_id"], 3)
            dc = round(run["c100_probe"] - base["c100_probe"], 4)
            ik = round(run["id_probe_acc"] - base["id_probe_acc"], 4)
            rows.append({"lambda": lam, "d_delta": dd, "c_delta": dc,
                         "id_keep": ik, "pass_gate": ik >= ID_FLOOR,
                         "secant_pct": round(100 * (dc / dd) / S_OBS, 1)})
        fit_rows = [r for r in rows if r["pass_gate"]]
        dropped = [r["lambda"] for r in rows if not r["pass_gate"]]
        b = fit_power(fit_rows)
        in_band = BAND_LO <= b <= BAND_HI
        seeds[s] = {"rows": rows, "b_seed1": round(b, 3),
                    "r2": round(r2_of(fit_rows, b), 3),
                    "n_fitted": len(fit_rows), "dropped": dropped,
                    "in_band": in_band,
                    "b_orig": B_ORIG[s],
                    "drift": round(b - B_ORIG[s], 3)}
        lam03 = next(r for r in rows if r["lambda"] == 0.03)
        e2_rows.append({"seed": s, "dc_new": lam03["c_delta"],
                        "dc_orig": LAM03_ORIG[s],
                        "in_placebo": abs(lam03["c_delta"]) <= PLACEBO,
                        "secant_pct": lam03["secant_pct"]})

    n_in = sum(seeds[s]["in_band"] for s in (1, 2))
    if n_in == 2:
        e1 = ("BAND RUN-STABLE: both rerun b inside [0.264,0.347] -- "
              "Appendix 34 verdict STANDS (s0 variance larger, outlier "
              "not pure noise).")
    elif n_in == 0:
        e1 = ("BAND PSEUDO-TIGHT: both rerun b outside -- Appendix 34 "
              "verdict REVERSES (run-level b variance is seed-generic; "
              "single-sample b is not a reliable cross-seed statistic).")
    else:
        e1 = ("MIXED: one rerun b in band, one out -- wording downgrades "
              "to 's0 samples higher (both draws outside) but variance "
              "scales partially overlap'.")

    all_rows = [r for s in (1, 2) for r in seeds[s]["rows"]]
    gate_min = min(r["id_keep"] for r in all_rows)
    result = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "script": "code/p7_band_readout.py",
            "frozen_rules": "ADR-0016 Appendix 35 (2026-11-11 20:02, "
                            "frozen before launch)",
            "constants": {"band": [BAND_LO, BAND_HI], "s_obs": S_OBS,
                          "placebo": PLACEBO, "id_floor": ID_FLOOR},
        },
        "curves": {f"s{s}": seeds[s] for s in (1, 2)},
        "E1_band_stability": {"n_in_band": n_in, "verdict": e1},
        "E2_lambda03": e2_rows,
        "E3_gate": {"id_keep_min": gate_min, "floor": ID_FLOOR,
                    "all_pass": gate_min >= ID_FLOOR},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    for s in (1, 2):
        d = seeds[s]
        print(f"s{s} curve (run-seed 1): b={d['b_seed1']} "
              f"(orig {d['b_orig']}, drift {d['drift']:+.3f}, "
              f"r2={d['r2']}, n={d['n_fitted']}"
              f"{', dropped ' + str(d['dropped']) if d['dropped'] else ''})")
        for r in d["rows"]:
            print(f"  lam {r['lambda']:>4}: dd={r['d_delta']:>8.3f} "
                  f"dc={r['c_delta']:+.4f} ik={r['id_keep']:+.4f} "
                  f"sec={r['secant_pct']:>5.1f}% "
                  f"{'ok' if r['pass_gate'] else 'GATE-FAIL'}")
    print(f"\nE1 ({n_in}/2 in band): {e1}")
    for e in e2_rows:
        mark = "in-placebo" if e["in_placebo"] else "out-of-band"
        print(f"E2 s{e['seed']} lam0.03: dc {e['dc_orig']} -> "
              f"{e['dc_new']:+.4f} [{mark}] sec {e['secant_pct']}%")
    print(f"E3: id_keep min {gate_min:+.4f} "
          f"({'all pass' if gate_min >= ID_FLOOR else 'FAILURES'})")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
