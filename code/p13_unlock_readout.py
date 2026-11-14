#!/usr/bin/env python3
"""P13 unlock readout (ADR-0018 Section 7, L1/L2/L3 frozen).

12 fork runs: two locked (failed) ep-40 snapshots x three higher
doses (0.3/1.0/3.0) x two fork seeds. Unlock = final dc vs the
s0 base inside the historical healthy band (>= +0.05).

  L1: any dose with >=3/4 unlocks -> dose-specific lock;
      <=2/12 overall -> near-absolute lock; else report as-is.
  L2: unlocked runs' dd should recover (>= +20).
  L3: id_keep >= -0.02.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_TBL = ROOT / "results" / "p4_full" / "full_base_lr0.125.json"
E_DIR = ROOT / "results" / "p13_unlock"
OUT = E_DIR / "p13_unlock_readout.json"

UNLOCK_DC = 0.05     # L1 frozen
RECOVER_DD = 20      # L2 frozen
ID_FLOOR = -0.02
SRCS = ("seed7", "seed13")
LAMS = (0.3, 1.0, 3.0)
SEEDS = (201, 202)


def main() -> int:
    base = next(r for r in json.loads(BASE_TBL.read_text(encoding="utf-8"))["rows"]
                if r["model"] == "resnet18_lr0.125_wd0.0001_s0")
    rows = []
    for src in SRCS:
        for lam in LAMS:
            for fs in SEEDS:
                p = E_DIR / f"p13_unlock_{src}_lam{lam}_seed{fs}.json"
                run = json.loads(p.read_text(encoding="utf-8"))["rows"][0]
                dd = round(run["d_ent_id"] - base["d_ent_id"], 3)
                dc = round(run["c100_probe"] - base["c100_probe"], 4)
                ik = round(run["id_probe_acc"] - base["id_probe_acc"], 4)
                rows.append({"src": src, "lambda": lam, "fork_seed": fs,
                             "d_delta": dd, "c_delta": dc, "id_keep": ik,
                             "unlocked": dc >= UNLOCK_DC,
                             "dd_recovered": dd >= RECOVER_DD})

    per_dose = {lam: sum(r["unlocked"] for r in rows if r["lambda"] == lam)
                for lam in LAMS}
    n_unlock = sum(r["unlocked"] for r in rows)
    if any(v >= 3 for v in per_dose.values()):
        l1 = ("DOSE-SPECIFIC LOCK: >=1 dose with >=3/4 unlocks -- the "
              "low-d basin can be escaped by sufficient forcing dose.")
    elif n_unlock <= 2:
        l1 = ("NEAR-ABSOLUTE LOCK: <=2/12 unlocks across all doses -- the "
              "locked low-d basin is not escaped by any higher whiten "
              "dose; strongest form of geometric-intervention "
              "irreversibility.")
    else:
        l1 = (f"INTERMEDIATE: {n_unlock}/12 unlocks "
              f"(per-dose {per_dose}) -- report as-is.")
    anom = [f"{r['src']}@{r['lambda']}/s{r['fork_seed']}" for r in rows
            if r["unlocked"] and not r["dd_recovered"]]
    ik_min = min(r["id_keep"] for r in rows)

    result = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "script": "code/p13_unlock_readout.py",
            "frozen_rules": "ADR-0018 Section 7 (2026-11-14 20:18)",
            "constants": {"unlock_dc": UNLOCK_DC, "recover_dd": RECOVER_DD,
                          "id_floor": ID_FLOOR},
        },
        "rows": rows,
        "L1": {"n_unlock": n_unlock, "n_runs": len(rows),
               "per_dose": per_dose, "verdict": l1},
        "L2": {"decoupled_anomalies": anom},
        "L3": {"id_keep_min": round(ik_min, 4),
               "all_pass": ik_min >= ID_FLOOR},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"{'src':8} {'lam':>4} {'seed':>5} {'dd':>7} {'dc':>7} unlock")
    for r in rows:
        print(f"{r['src']:8} {r['lambda']:>4} {r['fork_seed']:>5} "
              f"{r['d_delta']:>7.2f} {r['c_delta']:>+7.4f} "
              f"{'YES' if r['unlocked'] else ''}")
    print(f"\nL1: {n_unlock}/12 unlocked | per-dose {per_dose}")
    print(f"-> {l1}")
    print(f"L2 decoupled: {anom if anom else 'none'}")
    print(f"L3: id_keep min {ik_min:+.4f}")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
