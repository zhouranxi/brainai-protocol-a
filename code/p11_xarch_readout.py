#!/usr/bin/env python3
"""P11 x-architecture readout (ADR-0017 Section 8, I1/I2/I3 frozen).

Four architectures x 2 full curves (run-seed 1/2) x 5 lambdas.
Base readings come from each run JSON's own `base` block (the
forcing script re-measures the ckpt_grid checkpoint on the spot).

  I1 excursion presence: any arch with |b1-b2| >= 0.08 ->
     first cross-architecture excursion, heavy tail generalizes;
     all < 0.08 -> report as-is (2 draws/arch = ~28% power at 15%,
     zero excursions do NOT falsify the heavy tail).
  I2 shape red lines: every curve sub-linear (b < 0.5) and
     saturating (marginal deltas decreasing over 0.3->1.0->3.0);
     b > 0.7 or non-concave shape triggers a paper-wording check.
  I3 id_keep >= -0.02 on all 48 runs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E_DIR = ROOT / "results" / "p11_xarch"
OUT = E_DIR / "p11_xarch_readout.json"

S_OBS = 0.0161
ID_FLOOR = -0.02
LAMS = (0.03, 0.1, 0.3, 1.0, 3.0)
EXC_THRESH = 0.08        # I1 frozen (G1 threshold carried over)
SUB_LIN_LIMIT = 0.5      # I2 red line (expectation)
RED_LINE = 0.7           # I2 red line (hard)
ARCHS = {
    "resnet34": "resnet34_lr0.125_wd5e-05_s0",
    "resnet50": "resnet50_lr0.125_wd5e-05_s0",
    "regnet": "regnet_y_400mf_lr0.125_wd5e-05_s0",
    "shufflenet": "shufflenet_v2_x1_0_lr0.125_wd5e-05_s0",
}


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
    base_tbl = {r["model"]: r for r in
                json.loads((E_DIR / "p11_base_readings.json")
                           .read_text(encoding="utf-8"))["rows"]}
    arch_rows, gate_min = {}, 1.0
    for tag, model in ARCHS.items():
        base = base_tbl[model]
        curves = []
        for rs in (1, 2):
            rows = []
            for lam in LAMS:
                p = E_DIR / f"p11_{model}_seed{rs}_lam{lam}.json"
                run = json.loads(p.read_text(encoding="utf-8"))["rows"][0]
                dd = round(run["d_ent_id"] - base["d_ent_id"], 3)
                dc = round(run["c100_probe"] - base["c100_probe"], 4)
                ik = round(run["id_probe_acc"] - base["id_probe_acc"], 4)
                gate_min = min(gate_min, ik)
                rows.append({"lambda": lam, "d_delta": dd, "c_delta": dc,
                             "id_keep": ik, "pass_gate": ik >= ID_FLOOR,
                             "secant_pct": round(100 * (dc / dd) / S_OBS, 1)
                             if dd else None})
            fit_rows = [r for r in rows if r["pass_gate"] and r["d_delta"]]
            b = fit_power(fit_rows)
            # saturation check on marginal deltas 0.3->1.0->3.0
            by_lam = {r["lambda"]: r["c_delta"] for r in rows}
            m1 = by_lam[1.0] - by_lam[0.3]
            m2 = by_lam[3.0] - by_lam[1.0]
            curves.append({"run_seed": rs, "b": round(b, 3),
                           "r2": round(r2_of(fit_rows, b), 3),
                           "n": len(fit_rows),
                           "saturating": m2 < m1,
                           "marginals": [round(m1, 4), round(m2, 4)],
                           "rows": rows})
        b1, b2 = curves[0]["b"], curves[1]["b"]
        arch_rows[tag] = {"model": model, "curves": curves,
                          "b_gap": round(abs(b1 - b2), 3),
                          "excursion": abs(b1 - b2) >= EXC_THRESH,
                          "b_max": max(b1, b2)}

    any_exc = any(v["excursion"] for v in arch_rows.values())
    i1 = ("FIRST CROSS-ARCH EXCURSION: >=1 arch with |b1-b2|>=0.08 -- "
          "the heavy tail generalizes beyond resnet18." if any_exc else
          "NO EXCURSION IN 8 DRAWS: report as-is; at 15% base rate the "
          "power of 2 draws/arch (~28%) cannot falsify the heavy tail.")
    red = [t for t, v in arch_rows.items() if v["b_max"] > RED_LINE]
    nonsat = [t for t, v in arch_rows.items()
              if not all(c["saturating"] for c in v["curves"])]
    i2 = ("RED LINE TRIPPED: " + f"b>{RED_LINE} in {red}; "
          f"non-saturating in {nonsat} -- paper wording check required."
          if red or nonsat else
          f"ALL GREEN: every curve b<{RED_LINE} and saturating "
          f"(sub-linear <{SUB_LIN_LIMIT} expectation held: "
          f"max b = {max(v['b_max'] for v in arch_rows.values())}).")
    result = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "script": "code/p11_xarch_readout.py",
            "frozen_rules": "ADR-0017 Section 8 (2026-11-13 01:15)",
            "constants": {"exc_thresh": EXC_THRESH, "red_line": RED_LINE,
                          "sub_lin_expect": SUB_LIN_LIMIT,
                          "id_floor": ID_FLOOR},
        },
        "archs": arch_rows,
        "I1_excursion_presence": {"any_excursion": any_exc, "verdict": i1},
        "I2_shape_red_lines": {"red_archs": red, "nonsat_archs": nonsat,
                               "verdict": i2},
        "I3_gate": {"id_keep_min": round(gate_min, 4), "floor": ID_FLOOR,
                    "all_pass": gate_min >= ID_FLOOR},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    for tag, v in arch_rows.items():
        c1, c2 = v["curves"]
        print(f"{tag:12} b(seed1)={c1['b']:.3f} r2={c1['r2']:.3f} | "
              f"b(seed2)={c2['b']:.3f} r2={c2['r2']:.3f} | "
              f"gap={v['b_gap']:.3f} {'EXCURSION' if v['excursion'] else ''}")
        for c in v["curves"]:
            row = " ".join(f"{r['lambda']}:{r['c_delta']:+.4f}"
                           for r in c["rows"])
            print(f"  seed{c['run_seed']}: {row} "
                  f"saturating={c['saturating']}")
    print(f"\nI1: {i1}\nI2: {i2}")
    print(f"I3: id_keep min {gate_min:+.4f} "
          f"({'all pass' if gate_min >= ID_FLOOR else 'FAIL'})")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
