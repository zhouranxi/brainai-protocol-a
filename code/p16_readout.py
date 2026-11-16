#!/usr/bin/env python3
"""P16 readout (ADR-0022 Section 2, G1-G4 frozen).

27 runs (3 bottlenecks x 3 configs x 3 seeds).
  G1 Go: any bottleneck with HFB-FF >= +1pp, consistent sign ->
     gain window exists, P17 starts on that bottleneck.
  G2 all-zero: every bottleneck |HFB-FF| < 1pp or inconsistent ->
     bottleneck-invariant negative, architecture line closes.
  G3 REC control: HFB-REC gap (hierarchy specificity).
  G4 health: FF id >= 0.60 per bottleneck (hard-but-learnable).
Probe numbers reported, never used for gating (ADR-0021 rule).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E_DIR = ROOT / "results" / "p16_bottleneck"
OUT = E_DIR / "p16_readout.json"
GO_MIN = 0.01
HEALTH = 0.60
BNS = ("mask", "width", "data")
CFGS = ("ff", "rec", "hfb")


def mean(xs):
    return sum(xs) / len(xs)


def main() -> int:
    rows = {}
    for b in BNS:
        for cfg in CFGS:
            for s in range(3):
                p = E_DIR / f"p16_{b}_{cfg}_s{s}.json"
                rows[(b, cfg, s)] = json.loads(
                    p.read_text(encoding="utf-8"))["rows"][0]

    per_bn = {}
    for b in BNS:
        ff = mean([rows[(b, "ff", s)]["id_test"] for s in range(3)])
        rec = mean([rows[(b, "rec", s)]["id_test"] for s in range(3)])
        hfb = mean([rows[(b, "hfb", s)]["id_test"] for s in range(3)])
        gaps = [round(rows[(b, "hfb", s)]["id_test"]
                      - rows[(b, "ff", s)]["id_test"], 4) for s in range(3)]
        consistent = all(g > 0 for g in gaps) or all(g < 0 for g in gaps)
        per_bn[b] = {"ff": round(ff, 4), "rec": round(rec, 4),
                     "hfb": round(hfb, 4),
                     "hfb_minus_ff": round(hfb - ff, 4),
                     "per_seed_gaps": gaps, "consistent": consistent,
                     "hfb_minus_rec": round(hfb - rec, 4),
                     "c100": {c: round(mean(
                         rows[(b, c, s)]["c100_probe"] for s in range(3)), 4)
                         for c in CFGS}}

    go = [b for b in BNS
          if per_bn[b]["hfb_minus_ff"] >= GO_MIN and per_bn[b]["consistent"]]
    ff_health = {b: per_bn[b]["ff"] >= HEALTH for b in BNS}
    if go:
        g1 = (f"GATE OPEN: bottleneck(s) {go} show HFB-FF >= +1pp with "
              "consistent sign -- gain window exists; P17 (carrier "
              "adjudication) starts on the strongest bottleneck.")
    else:
        g1 = ("ALL-ZERO GATE: no bottleneck reaches +1pp consistently -- "
              "the no-gain result is bottleneck-invariant; architecture "
              "line closes, next step is the LLM side (P18).")
    g3 = {b: ("hierarchy-specific" if per_bn[b]["hfb_minus_rec"] >= GO_MIN
              else "iteration-only") for b in BNS}
    bad = [b for b in BNS if not ff_health[b]]

    result = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "script": "code/p16_readout.py",
            "frozen_rules": "ADR-0022 Section 2 (2026-11-17 00:27)",
            "constants": {"go_min": GO_MIN, "health": HEALTH},
        },
        "per_bottleneck": per_bn,
        "G1_gate": {"go_bottlenecks": go, "verdict": g1},
        "G3_rec_control": g3,
        "G4_health": {"ff_id": {b: per_bn[b]["ff"] for b in BNS},
                      "failed": bad},
        "rows": [rows[(b, c, s)] for b in BNS for c in CFGS for s in range(3)],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    for b in BNS:
        v = per_bn[b]
        print(f"{b:6}: FF {v['ff']:.4f} REC {v['rec']:.4f} "
              f"HFB {v['hfb']:.4f} | HFB-FF {v['hfb_minus_ff']:+.4f} "
              f"({v['per_seed_gaps']}) HFB-REC {v['hfb_minus_rec']:+.4f} "
              f"| c100 FF/H/R {v['c100']['ff']}/{v['c100']['rec']}/{v['c100']['hfb']}")
    print(f"\nG1: {g1}")
    print(f"G3: {g3}")
    print(f"G4: FF health {ff_health} (failed: {bad if bad else 'none'})")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
