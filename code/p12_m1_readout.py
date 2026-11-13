#!/usr/bin/env python3
"""P12 M1a readout (ADR-0018 Section 3, K1/K2 frozen).

16 instrumented runs (s0/s3 x run-seeds 7..14, lambda=0.03) with
per-epoch forcing-loss history and ep40 snapshots.

  K1 (main): failure = |dc| <= 0.022 (ADR-0017 definition).
     Layered readout: if every failing run's mean forcing loss over
     ep1-40 sits below the minimum of the healthy runs -> EARLY
     NON-ENGAGEMENT (the loss never bit); if the ranges overlap ->
     failure differentiates after ep40, defer to M1b fork.
     Also report the new failure rate k/16.
  K2: id_keep >= -0.02.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_TBL = ROOT / "results" / "p4_full" / "full_base_lr0.125.json"
E_DIR = ROOT / "results" / "p12_mechanism"
OUT = E_DIR / "p12_m1_readout.json"

PLACEBO = 0.022
ID_FLOOR = -0.02
SEEDS = range(7, 15)


def mean(xs):
    return sum(xs) / len(xs)


def main() -> int:
    bases = {r["model"]: r for r in
             json.loads(BASE_TBL.read_text(encoding="utf-8"))["rows"]}
    rows = []
    for s in (0, 3):
        model = f"resnet18_lr0.125_wd0.0001_s{s}"
        base = bases[model]
        for rs in SEEDS:
            p = E_DIR / f"p12_m1a_s{s}_seed{rs}_lam0.03.json"
            run = json.loads(p.read_text(encoding="utf-8"))["rows"][0]
            dc = round(run["c100_probe"] - base["c100_probe"], 4)
            dd = round(run["d_ent_id"] - base["d_ent_id"], 3)
            ik = round(run["id_probe_acc"] - base["id_probe_acc"], 4)
            hist = run.get("history", [])
            fl_1_40 = round(mean([h["forcing_loss"] for h in hist[:40]]), 5) \
                if hist else None
            rows.append({"seed_tag": f"s{s}", "run_seed": rs,
                         "d_delta": dd, "c_delta": dc, "id_keep": ik,
                         "fail": abs(dc) <= PLACEBO,
                         "fl_mean_1_40": fl_1_40,
                         "fl_mean_1_10": round(mean(
                             [h["forcing_loss"] for h in hist[:10]]), 5)
                         if hist else None,
                         "d_at_ep100": next(
                             (c["d_ent_batch"] for c in run["d_curve"]
                              if c["epoch"] == 100), None)})

    fails = [r for r in rows if r["fail"]]
    healthy = [r for r in rows if not r["fail"]]
    fl_fail = [r["fl_mean_1_40"] for r in fails if r["fl_mean_1_40"]]
    fl_health_min = min((r["fl_mean_1_40"] for r in healthy
                         if r["fl_mean_1_40"]), default=None)
    if fl_fail and fl_health_min is not None and max(fl_fail) < fl_health_min:
        k1 = ("EARLY NON-ENGAGEMENT: every failing run's ep1-40 mean "
              "forcing loss is below the healthy minimum -- the loss "
              "never engaged; mechanism (a) supported at the loss level.")
    elif not fails:
        k1 = (f"ZERO FAILURES in 16 draws (base rate 15% prior) -- K1 "
              "layered readout degenerate; M1b not triggered; update "
              "the rate CI with k=0.")
    elif fl_fail and fl_health_min is not None and min(fl_fail) >= fl_health_min:
        k1 = ("OVERLAP: failing runs' early forcing loss is inside the "
              "healthy range -- failure differentiates after ep40; "
              "defer to M1b fork adjudication (J1).")
    else:
        k1 = ("PARTIAL OVERLAP: report per-run values as-is; the "
              "layered rule does not separate; defer to M1b.")

    ik_min = min(r["id_keep"] for r in rows)
    result = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "script": "code/p12_m1_readout.py",
            "frozen_rules": "ADR-0018 Section 3 (2026-11-14 00:38)",
            "constants": {"placebo": PLACEBO, "id_floor": ID_FLOOR},
        },
        "rows": rows,
        "K1": {"n_fail": len(fails), "n_runs": len(rows),
               "fail_rate": round(len(fails) / len(rows), 3),
               "fail_runs": [f"{r['seed_tag']}_seed{r['run_seed']}" for r in fails],
               "fl_fail_range": [round(min(fl_fail), 5), round(max(fl_fail), 5)]
               if fl_fail else None,
               "fl_healthy_min": fl_health_min,
               "verdict": k1},
        "K2": {"id_keep_min": round(ik_min, 4), "floor": ID_FLOOR,
               "all_pass": ik_min >= ID_FLOOR},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"{'run':16} {'dd':>6} {'dc':>7} {'ik':>7} {'fl@1-40':>8} "
          f"{'fl@1-10':>8} {'d@100':>6} fail")
    for r in rows:
        print(f"{r['seed_tag']}_seed{r['run_seed']:<2}    "
              f"{r['d_delta']:>6.2f} {r['c_delta']:>+7.4f} "
              f"{r['id_keep']:>+7.4f} {r['fl_mean_1_40']:>8.5f} "
              f"{r['fl_mean_1_10']:>8.5f} "
              f"{(r['d_at_ep100'] or 0):>6.1f} "
              f"{'FAIL' if r['fail'] else ''}")
    print(f"\nK1 ({len(fails)}/16 fail): {k1}")
    print(f"K2: id_keep min {ik_min:+.4f} "
          f"({'all pass' if ik_min >= ID_FLOOR else 'FAIL'})")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
