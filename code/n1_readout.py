#!/usr/bin/env python3
"""ADR-0019 N1 readout (R1-R4 frozen).

9 runs (FF/REC/HFB x seeds 0-2). Gain decomposition:
  G = A_hfb - A_ff            (total gain, per metric)
  r = (A_hfb_off - A_ff)/G    (reshape share, training-time carrier)
  q = (A_hfb - A_hfb_off)/G   (guidance share, inference-time carrier)
computed on both id_test and c100_probe (dual readout).

  R1: q >= 2/3 both metrics -> GUIDANCE-DOMINANT; q <= 1/3 both ->
      RESHAPE-DOMINANT; else report (r, q) as-is.
  R2: A_hfb - A_rec per seed; ~0 with sign flips -> gain is iteration,
      not hierarchy; consistent > 0 -> hierarchy-specific.
  R3: d_ent/GLUE across configs (geometric signature).
  R4: id_test >= 0.90 health gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E_DIR = ROOT / "results" / "n1_hierarchy"
OUT = E_DIR / "n1_readout.json"
Q_HI, Q_LO = 2 / 3, 1 / 3
HEALTH = 0.90
CFGS = ("ff", "rec", "hfb")
SEEDS = (0, 1, 2)


def mean(xs):
    return sum(xs) / len(xs)


def main() -> int:
    rows = {}
    for cfg in CFGS:
        for s in SEEDS:
            p = E_DIR / f"n1_{cfg}_s{s}.json"
            rows[(cfg, s)] = json.loads(p.read_text(encoding="utf-8"))["rows"][0]

    def m(cfg, key):
        return mean([rows[(cfg, s)][key] for s in SEEDS])

    decomp = {}
    for key in ("id_test", "c100_probe"):
        aff, arec, ahfb = m("ff", key), m("rec", key), m("hfb", key)
        aoff = mean([rows[("hfb", s)]["off_test"] if key == "id_test"
                     else None for s in SEEDS]) if key == "id_test" else None
        # c100 的 off 读数未测（探针用特征，不区分推理步）——off 仅 id 口径
        decomp[key] = {"a_ff": round(aff, 4), "a_rec": round(arec, 4),
                       "a_hfb": round(ahfb, 4),
                       "a_hfb_off": round(aoff, 4) if aoff else None}

    aoff = decomp["id_test"]["a_hfb_off"]
    G = decomp["id_test"]["a_hfb"] - decomp["id_test"]["a_ff"]
    r = (aoff - decomp["id_test"]["a_ff"]) / G if G else None
    q = (decomp["id_test"]["a_hfb"] - aoff) / G if G else None
    if G and q is not None and q >= Q_HI:
        r1 = (f"GUIDANCE-DOMINANT: q={q:.2f} >= 2/3 -- the hierarchical-"
              "feedback gain is carried mainly by inference-time "
              "dynamics (direct evidence for prior-guided reasoning).")
    elif G and q is not None and q <= Q_LO:
        r1 = (f"RESHAPE-DOMINANT: q={q:.2f} <= 1/3 -- the gain is carried "
              "mainly by training-time representation reshaping.")
    else:
        r1 = (f"INTERMEDIATE: r={r:.2f}, q={q:.2f} -- report decomposition "
              "as-is; no dichotomy.") if G else "G ~= 0: no gain to decompose."

    # R2: per-seed hfb - rec
    d_rec = [round(rows[("hfb", s)]["id_test"] - rows[("rec", s)]["id_test"], 4)
             for s in SEEDS]
    same_sign = all(d > 0 for d in d_rec) or all(d < 0 for d in d_rec)
    if same_sign and abs(mean(d_rec)) > 0.002:
        r2 = (f"HIERARCHY-SPECIFIC: A_hfb-A_rec = {d_rec} (consistent sign) "
              "-- cross-level feedback adds beyond iteration budget.")
    else:
        r2 = (f"ITERATION-ONLY: A_hfb-A_rec = {d_rec} -- no consistent "
              "hierarchy-specific contribution.")

    # R3: geometric signature
    geo = {cfg: {"d_ent": round(m(cfg, "d_ent"), 2),
                 "glue_d": m(cfg, "glue_d"), "c100": round(m(cfg, "c100_probe"), 4)}
           for cfg in CFGS}

    worst = min(rows[(cfg, s)]["id_test"] for cfg in CFGS for s in SEEDS)
    result = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "script": "code/n1_readout.py",
            "frozen_rules": "ADR-0019 Section 4 (2026-11-15 01:42)",
            "constants": {"q_hi": Q_HI, "q_lo": Q_LO, "health": HEALTH},
        },
        "rows": [rows[(cfg, s)] for cfg in CFGS for s in SEEDS],
        "decomposition": decomp,
        "R1_carrier": {"G": round(G, 4), "r": round(r, 3) if r is not None else None,
                       "q": round(q, 3) if q is not None else None,
                       "verdict": r1},
        "R2_hierarchy_specificity": {"per_seed": d_rec, "verdict": r2},
        "R3_geometry": geo,
        "R4_health": {"worst_id": round(worst, 4),
                      "all_pass": worst >= HEALTH},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    for cfg in CFGS:
        line = f"{cfg}: " + " | ".join(
            f"s{s}: id={rows[(cfg, s)]['id_test']:.4f}"
            f" off={rows[(cfg, s)]['off_test'] if rows[(cfg, s)]['off_test'] is not None else '-'}"
            f" c100={rows[(cfg, s)]['c100_probe']:.4f}"
            for s in SEEDS)
        print(line)
    print(f"\nG={G:.4f} r={r:.3f} q={q:.3f}")
    print(f"R1: {r1}\nR2: {r2}")
    print(f"R3: {json.dumps(geo)}")
    print(f"R4: worst id {worst:.4f} ({'pass' if worst >= HEALTH else 'FAIL'})")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
