#!/usr/bin/env python3
"""N2 readout (ADR-0019 Section 8, S1/S2/B1/B2 frozen).

N2a: hfb x fb-scale {0.1,0.3,1.0} x seeds {0,1,2} -> S1 window
     test (>=+0.5pp vs FF, consistent sign) + S2 modulation
     survival (|s_final|, off vs native).
N2b: ff/rec seeds {3,4,5} pooled with N1pp {0,1,2} -> B1 c100
     stability (>=+3pp, >=5/6 same sign) + B2 d_ent~c100
     Spearman across 6 rec seeds (descriptive).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N1P = ROOT / "results" / "n1p_hierarchy"
N2 = ROOT / "results" / "n2"
OUT = N2 / "n2_readout.json"
GAIN_MIN = 0.005       # S1 frozen (+0.5pp)
OOD_MIN = 0.03         # B1 frozen (+3pp)
ALPHAS = ("0.1", "0.3", "1.0")


def spearman(xs, ys):
    rx = sorted(range(len(xs)), key=lambda i: xs[i])
    ry = sorted(range(len(ys)), key=lambda i: ys[i])
    n = len(xs)
    dx = [0] * n; dy = [0] * n
    for r, i in enumerate(rx): dx[i] = r
    for r, i in enumerate(ry): dy[i] = r
    mx = sum(dx) / n; my = sum(dy) / n
    num = sum((a - mx) * (b - my) for a, b in zip(dx, dy))
    den = (sum((a - mx) ** 2 for a in dx) * sum((b - my) ** 2 for b in dy)) ** 0.5
    return num / den if den else 0.0


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))["rows"][0]


def main() -> int:
    ff_n1p = {s: load(N1P / f"n1p_ff_s{s}.json") for s in range(3)}
    ff_mean = sum(r["id_test"] for r in ff_n1p.values()) / 3

    # ---- N2a ----
    a_rows = {}
    for a in ALPHAS:
        for s in range(3):
            p = N2 / f"n2a_hfb_a{a}_s{s}.json"
            a_rows[(a, s)] = load(p)
    per_alpha = {}
    for a in ALPHAS:
        gains = [round(a_rows[(a, s)]["id_test"] - ff_mean, 4) for s in range(3)]
        s_final = [a_rows[(a, s)].get("fb_scale_final") for s in range(3)]
        off_diff = [round(a_rows[(a, s)]["id_test"] -
                          (a_rows[(a, s)]["off_test"] or 0), 4) for s in range(3)]
        per_alpha[a] = {"gains_vs_ff": gains,
                        "mean_gain": round(sum(gains) / 3, 4),
                        "consistent": all(g > 0 for g in gains) or all(g < 0 for g in gains),
                        "fb_scale_final": s_final,
                        "off_minus_native": off_diff}
    best = max(per_alpha.items(), key=lambda kv: kv[1]["mean_gain"])
    if best[1]["mean_gain"] >= GAIN_MIN and best[1]["consistent"]:
        s1 = (f"WINDOW EXISTS: alpha={best[0]} mean gain "
              f"{best[1]['mean_gain']:+.4f} >= +0.5pp with consistent "
              "sign -- R1 decomposition restarts at this dose (N3).")
    elif all(v["mean_gain"] < GAIN_MIN for v in per_alpha.values()):
        s1 = ("WINDOW REJECTED: no alpha reaches +0.5pp -- the "
              "hierarchical-feedback family shows no ID gain; "
              "ADR-0019 main question closes as a negative result.")
    else:
        s1 = (f"NON-MONOTONE: per-alpha mean gains "
              f"{ {a: v['mean_gain'] for a, v in per_alpha.items()} } "
              "-- Goldilocks evidence; report curve as-is.")
    s2_alive = {a: [abs(v) > 0.05 for v in
                    (r.get("fb_scale_final") or (0, 0)) if v is not None]
                for a in ALPHAS for r in [a_rows[(a, 0)]]}

    # ---- N2b ----
    rec = {s: load(N1P / f"n1p_rec_s{s}.json") for s in range(3)}
    ff6 = dict(ff_n1p)
    for s in (3, 4, 5):
        ff6[s] = load(N2 / f"n1p_ff_s{s}.json")
        rec[s] = load(N2 / f"n1p_rec_s{s}.json")
    diffs = [round(rec[s]["c100_probe"] - ff6[s]["c100_probe"], 4) for s in range(6)]
    same = sum(1 for d in diffs if d > 0)
    mean_d = sum(diffs) / 6
    if mean_d >= OOD_MIN and same >= 5:
        b1 = (f"REC-OOD STABLE: mean c100 diff {mean_d:+.4f} >= +3pp, "
              f"{same}/6 seeds positive -- real effect, proceed to B2.")
    else:
        b1 = (f"REC-OOD UNSTABLE: mean {mean_d:+.4f}, {same}/6 positive "
              "-- downgrade to noise; N2b closes negative.")
    xs = [rec[s]["d_ent"] for s in range(6)]
    ys = [rec[s]["c100_probe"] for s in range(6)]
    rho = round(spearman(xs, ys), 3)

    result = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "script": "code/n2_readout.py",
            "frozen_rules": "ADR-0019 Section 8 (2026-11-15 19:09)",
            "constants": {"gain_min": GAIN_MIN, "ood_min": OOD_MIN},
        },
        "N2a_scale_sweep": {"per_alpha": per_alpha, "verdict": s1,
                            "modulation_alive": s2_alive},
        "N2b_rec_ood": {"per_seed_c100_diff": diffs, "mean_diff": round(mean_d, 4),
                        "n_positive": same, "verdict": b1,
                        "B2_dent_c100_spearman_rec6": rho},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    for a in ALPHAS:
        v = per_alpha[a]
        print(f"alpha {a}: gains={v['gains_vs_ff']} mean={v['mean_gain']:+.4f} "
              f"consistent={v['consistent']} s_final={v['fb_scale_final'][0]} "
              f"off-native={v['off_minus_native'][0]}")
    print(f"\nS1: {s1}")
    print(f"B1 diffs: {diffs} -> {b1}")
    print(f"B2 Spearman(d_ent, c100) over 6 rec seeds = {rho}")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
