#!/usr/bin/env python3
"""P8 readout (ADR-0017, F1/F2/F3 frozen rules).

Per-seed third sampling (run-seed 2) + the two archived/prior
samplings (run-seed 0 from v2, run-seed 1 from P6-D / P7-E) give
three independent b draws per seed for s0/s1/s2:

  s0: 0.515 (v2) / 0.394 (P6-D, 5pt run-seed1) / b_new
  s1: 0.325 (v2) / 0.323 (P7-E)                / b_new
  s2: 0.264 (v2) / 0.377 (P7-E)                / b_new

s0's run-seed-2 curve: 4 fresh runs (lam 0.03/0.1/1.0/3.0) +
P5 collapse_c1_seed2 at lam 0.3 (selection frozen in ADR-0017).

  F1 (main): per-seed range W_s and sd; any W_s >= 0.10 ->
             non-trivial run-level variance confirmed;
             all W_s <= 0.05 -> P7 stratification may be luck;
             else report as-is.
  F2: pairwise Cohen's d on 3 draws per seed; d>=2 resolvable,
             d<1 not; no formal testing (n=3).
  F3: id_keep >= -0.02 gate on all 14 runs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "p4_dose" / "dose_scan_seeds_v2.json"
BASE_TBL = ROOT / "results" / "p4_full" / "full_base_lr0.125.json"
P5_DIR = ROOT / "results" / "p5_collapse"
P6_DIR = ROOT / "results" / "p6_d_curve"
P7_DIR = ROOT / "results" / "p7_band"
E_DIR = ROOT / "results" / "p8_bdist"
OUT = E_DIR / "p8_bdist_readout.json"

S_OBS = 0.0161
ID_FLOOR = -0.02
LAMS = (0.03, 0.1, 0.3, 1.0, 3.0)
B_PRIOR = {0: [0.515, 0.394], 1: [0.325, 0.323], 2: [0.264, 0.377]}


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


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) \
        if len(xs) > 1 else 0.0


def cohens_d(a, b):
    sp = math.sqrt(((len(a) - 1) * sd(a) ** 2 + (len(b) - 1) * sd(b) ** 2)
                   / (len(a) + len(b) - 2))
    return abs(mean(a) - mean(b)) / sp if sp > 0 else float("inf")


def main() -> int:
    bases = {r["model"]: r for r in
             json.loads(BASE_TBL.read_text(encoding="utf-8"))["rows"]}
    curves, gate_min, drops = {}, 1.0, []
    for s in (0, 1, 2):
        model = f"resnet18_lr0.125_wd0.0001_s{s}"
        base = bases[model]
        rows = []
        for lam in LAMS:
            if s == 0 and lam == 0.3:
                run = json.loads((P5_DIR / "collapse_c1_seed2.json")
                                 .read_text(encoding="utf-8"))["rows"][0]
                src = "p5_c1_seed2 (frozen pick)"
            else:
                run = json.loads((E_DIR / f"p8_s{s}_seed2_lam{lam}.json")
                                 .read_text(encoding="utf-8"))["rows"][0]
                src = f"p8_seed2_lam{lam}"
            dd = round(run["d_ent_id"] - base["d_ent_id"], 3)
            dc = round(run["c100_probe"] - base["c100_probe"], 4)
            ik = round(run["id_probe_acc"] - base["id_probe_acc"], 4)
            gate_min = min(gate_min, ik)
            rows.append({"lambda": lam, "src": src, "d_delta": dd,
                         "c_delta": dc, "id_keep": ik,
                         "pass_gate": ik >= ID_FLOOR,
                         "secant_pct": round(100 * (dc / dd) / S_OBS, 1)})
            if ik < ID_FLOOR:
                drops.append((f"s{s}", lam))
        fit_rows = [r for r in rows if r["pass_gate"]]
        b = fit_power(fit_rows)
        curves[s] = {"rows": rows, "b_seed2": round(b, 3),
                     "r2": round(r2_of(fit_rows, b), 3),
                     "n_fitted": len(fit_rows)}

    # ---- F1 ---------------------------------------------------------------
    f1 = {}
    for s in (0, 1, 2):
        bs = B_PRIOR[s] + [curves[s]["b_seed2"]]
        f1[f"s{s}"] = {"b_draws": bs, "mean": round(mean(bs), 3),
                       "sd": round(sd(bs), 3),
                       "W_s": round(max(bs) - min(bs), 3)}
    any_ge10 = any(v["W_s"] >= 0.10 for v in f1.values())
    all_le05 = all(v["W_s"] <= 0.05 for v in f1.values())
    if any_ge10:
        f1v = ("NON-TRIVIAL VARIANCE CONFIRMED: >=1 seed with W_s>=0.10 "
               "-- single-sample b is not a reliable cross-seed "
               "statistic; Appendix 35 wording stands and strengthens.")
    elif all_le05:
        f1v = ("SMALL UNIFORM VARIANCE: all W_s<=0.05 -- the P7 "
               "stratification (s1 stable / s2 drift) may have been "
               "second-draw luck.")
    else:
        f1v = "INTERMEDIATE: report per-seed W_s as-is; no dichotomy."

    # ---- F2 ---------------------------------------------------------------
    f2 = {}
    for a, b_ in ((0, 1), (0, 2), (1, 2)):
        da = B_PRIOR[a] + [curves[a]["b_seed2"]]
        db = B_PRIOR[b_] + [curves[b_]["b_seed2"]]
        d = cohens_d(da, db)
        f2[f"s{a}-s{b_}"] = {
            "mean_a": round(mean(da), 3), "mean_b": round(mean(db), 3),
            "sd_a": round(sd(da), 3), "sd_b": round(sd(db), 3),
            "cohens_d": round(d, 2) if math.isfinite(d) else None,
            "resolvable_at_3": d >= 2, "resolvable": None
            if not math.isfinite(d) else d >= 2}

    result = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "script": "code/p8_b_distribution_readout.py",
            "frozen_rules": "ADR-0017 Section 2 (2026-11-12 01:48, "
                            "frozen before launch)",
            "constants": {"s_obs": S_OBS, "id_floor": ID_FLOOR,
                          "b_prior": {str(k): v for k, v in B_PRIOR.items()}},
        },
        "curves": {f"s{s}": curves[s] for s in (0, 1, 2)},
        "F1_distribution": {"per_seed": f1, "verdict": f1v},
        "F2_resolvability": f2,
        "F3_gate": {"id_keep_min": round(gate_min, 4), "floor": ID_FLOOR,
                    "all_pass": gate_min >= ID_FLOOR, "dropped": drops},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    for s in (0, 1, 2):
        print(f"s{s} seed2 curve: b={curves[s]['b_seed2']} "
              f"(r2={curves[s]['r2']}, n={curves[s]['n_fitted']})")
        for r in curves[s]["rows"]:
            print(f"  lam {r['lambda']:>4}: dd={r['d_delta']:>8.3f} "
                  f"dc={r['c_delta']:+.4f} ik={r['id_keep']:+.4f} "
                  f"sec={r['secant_pct']:>5.1f}% "
                  f"{'ok' if r['pass_gate'] else 'GATE-FAIL'}")
    print()
    for s in (0, 1, 2):
        v = f1[f"s{s}"]
        print(f"F1 s{s}: draws={v['b_draws']} mean={v['mean']} "
              f"sd={v['sd']} W_s={v['W_s']}")
    print(f"-> {f1v}\n")
    for k, v in f2.items():
        print(f"F2 {k}: d={v['cohens_d']} "
              f"(means {v['mean_a']} vs {v['mean_b']})")
    print(f"\nF3: id_keep min {gate_min:+.4f} "
          f"({'all pass' if gate_min >= ID_FLOOR else 'FAILURES: ' + str(drops)})")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
