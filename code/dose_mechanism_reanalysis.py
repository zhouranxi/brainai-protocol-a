#!/usr/bin/env python3
"""Dose-scan mechanism re-analysis (ADR-0016 Appendix 29).

Answers two open questions left by Appendix 28's adjudication, using ONLY
the archived 42-point v2 scan (results/p4_dose/dose_scan_seeds_v2.json):

  Q1  Is the hi-recipe low-dose secant plateau (17-38% of s_obs at
      lambda=0.03, 4/5 seeds >= 20%) a real functional response, or a
      small-denominator artifact (tiny d_delta inflates the ratio)?
  Q2  Is the across-seed drift of the power-law exponent
      (b = 0.502/0.325/0.264 for s0/s1/s2) driven by the same leverage
      effect on the log-log fit?

Read-only: never writes into runs/. Results land in
results/p4_dose/dose_mechanism_reanalysis.json for the Appendix 29
"results" section.

Judgement rules were FROZEN in ADR-0016 Appendix 29 before this script
was run; this file implements them verbatim and adds nothing.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "results" / "p4_dose" / "dose_scan_seeds_v2.json"
OUT = ROOT / "results" / "p4_dose" / "dose_mechanism_reanalysis.json"

S_OBS = 0.0161          # frozen in Appendix 26
PLACEBO_BAND = 0.022    # frozen in Appendix 25/26
SECANT_LABEL_THRESHOLD = 0.20  # 20% of s_obs, the 96-run label threshold


def secant_pct(c_delta: float, d_delta: float) -> float:
    """Secant slope as a percentage of s_obs (same convention as App. 26)."""
    return (c_delta / d_delta) / S_OBS * 100.0


def fit_power(points):
    """Least-squares slope of log(d_delta) vs log(|c_delta|); needs >=3 pts.

    Returns (b, r2) or (None, None) if any c_delta == 0 or < 3 points.
    """
    pts = [(math.log(p["d_delta"]), math.log(abs(p["c_delta"])))
           for p in points if p["c_delta"] != 0 and p["d_delta"] > 0]
    if len(pts) < 3:
        return None, None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return None, None
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    b = sxy / sxx
    ss_res = sum((y - (my + b * (x - mx))) ** 2 for x, y in pts)
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return b, r2


def fit_power_weighted(points):
    """Delta-d weighted least squares on the same log-log scale.

    Weight = d_delta (raw, not logged): large-denominator points dominate,
    which is exactly what deflates the leverage of small-delta points.
    """
    pts = [(math.log(p["d_delta"]), math.log(abs(p["c_delta"])), p["d_delta"])
           for p in points if p["c_delta"] != 0 and p["d_delta"] > 0]
    if len(pts) < 3:
        return None, None
    w = [d for _, _, d in pts]
    sw = sum(w)
    mx = sum(w * x for x, _, w in pts) / sw
    my = sum(w * y for _, y, w in pts) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
    if sxx == 0:
        return None, None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in pts)
    return sxy / sxx, None


def main() -> int:
    data = json.loads(ARCHIVE.read_text(encoding="utf-8"))
    pts = data["points"]

    for p in pts:
        p["secant_pct"] = secant_pct(p["c_delta"], p["d_delta"])
        p["in_band"] = abs(p["c_delta"]) <= PLACEBO_BAND

    # ---- M1: absolute magnitude of the low-dose response ---------------
    # hi recipe, lambda=0.03, every seed. Artifact iff the numerator
    # (c_delta) is inside the placebo band for >=3/5 seeds while the
    # secant reads >=20%.
    hi_low = [p for p in pts if p["recipe"] == "hi" and p["lambda"] == 0.03]
    m1_rows = [{
        "seed": p["seed"], "d_delta": round(p["d_delta"], 3),
        "c_delta": round(p["c_delta"], 4),
        "in_band": p["in_band"],
        "secant_pct": round(p["secant_pct"], 1),
    } for p in sorted(hi_low, key=lambda q: q["seed"])]
    n_in = sum(1 for r in m1_rows if r["in_band"])
    n_ge20 = sum(1 for r in m1_rows if abs(r["secant_pct"]) >= 20)
    if n_in >= 3:
        m1_verdict = "ARTIFACT: >=3/5 seeds have |c_delta| within the " \
                     "placebo band; the plateau is small-denominator " \
                     "inflation (numerator is noise-level)."
    elif (len(m1_rows) - n_in) >= 3:
        m1_verdict = "REAL RESPONSE: >=3/5 seeds exceed the placebo band " \
                     "in absolute c_delta; low-dose response is functional."
    else:
        m1_verdict = "MIXED: band membership splits; report as-is."

    # ---- M2: secant magnitude vs denominator size, all 42 points -------
    over = [p for p in pts if abs(p["secant_pct"]) >= SECANT_LABEL_THRESHOLD]
    under = [p for p in pts if abs(p["secant_pct"]) < SECANT_LABEL_THRESHOLD]
    if not over or not under:
        m2 = {"verdict": "degenerate: one class empty", "threshold_d": None}
    else:
        max_under_d = max(p["d_delta"] for p in under)
        min_over_d = min(p["d_delta"] for p in over)
        # Frozen rule: threshold = midpoint of the gap between the largest
        # below-threshold denominator and the smallest above-threshold one.
        if max_under_d < min_over_d:
            thr = (max_under_d + min_over_d) / 2
            m2 = {
                "max_denominator_of_below_threshold_points": round(max_under_d, 3),
                "min_denominator_of_above_threshold_points": round(min_over_d, 3),
                "threshold_d": round(thr, 3),
                "verdict": "CLEAN SEPARATION: every >=20% secant sits at "
                           "d_delta < %.2f and every point above it reads "
                           "< 20%% -- secant magnitude is set by the "
                           "denominator (artifact support)." % thr,
            }
        else:
            big_over = [p for p in over if p["d_delta"] > max_under_d]
            m2 = {
                "max_denominator_of_below_threshold_points": round(max_under_d, 3),
                "min_denominator_of_above_threshold_points": round(min_over_d, 3),
                "n_above_threshold_with_large_d_delta": len(big_over),
                "verdict": "NO SEPARATION: %d point(s) with large d_delta "
                           "still read >=20%% secant -- real high-slope "
                           "responses exist; artifact does not explain "
                           "everything." % len(big_over),
                "large_d_over_points": [{
                    "recipe": p["recipe"], "seed": p["seed"],
                    "lambda": p["lambda"], "d_delta": round(p["d_delta"], 3),
                    "c_delta": round(p["c_delta"], 4),
                    "secant_pct": round(p["secant_pct"], 1),
                } for p in big_over],
            }

    # ---- M3: does the exponent converge when low-leverage points drop? -
    # Full 5-point seeds only (s0/s1/s2), hi recipe, three fits each.
    m3 = {"fits": {}, "spans": {}}
    for seed in (0, 1, 2):
        hi_pts = sorted([p for p in pts
                         if p["recipe"] == "hi" and p["seed"] == seed],
                        key=lambda q: q["d_delta"])
        b_full, r2 = fit_power(hi_pts)
        drop_smallest = hi_pts[1:]
        b_drop, _ = fit_power(drop_smallest)
        b_w, _ = fit_power_weighted(hi_pts)
        m3["fits"][f"s{seed}"] = {
            "full_5pt": {"b": round(b_full, 3), "r2": round(r2, 3)},
            "drop_smallest_d": {"b": round(b_drop, 3),
                                "dropped_lambda": hi_pts[0]["lambda"],
                                "n": len(drop_smallest)},
            "weighted_by_d": {"b": round(b_w, 3)},
        }
    for mode in ("full_5pt", "drop_smallest_d", "weighted_by_d"):
        bs = [m3["fits"][f"s{s}"][mode]["b"] for s in (0, 1, 2)]
        m3["spans"][mode] = {
            "b_values": bs,
            "range": round(max(bs) - min(bs), 3),
        }
    span_a = m3["spans"]["full_5pt"]["range"]
    m3["narrowing"] = {}
    for mode in ("drop_smallest_d", "weighted_by_d"):
        span_x = m3["spans"][mode]["range"]
        m3["narrowing"][mode] = {
            "span": span_x,
            "narrowing_pct": round(100 * (span_a - span_x) / span_a, 1),
        }
    narrowing_ok = any(v["narrowing_pct"] >= 50
                       for v in m3["narrowing"].values())
    m3["verdict"] = (
        "CONVERGES: removing/deflating the smallest-denominator point "
        "narrows the across-seed b span by >=50% -- the drift is fitting "
        "leverage from small-d_delta points, not seed-level science."
        if narrowing_ok else
        "DOES NOT CONVERGE: the b drift survives low-leverage correction "
        "-- treat as genuine across-seed variation (upgrade to a finding).")

    # ---- assemble --------------------------------------------------------
    result = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "script": "code/dose_mechanism_reanalysis.py",
            "input": "results/p4_dose/dose_scan_seeds_v2.json (42 pts)",
            "frozen_rules": "ADR-0016 Appendix 29 (2026-11-10 14:45, "
                            "frozen before any computation)",
            "constants": {"s_obs": S_OBS, "placebo_band": PLACEBO_BAND,
                          "label_threshold_pct": 20},
        },
        "M1_low_dose_magnitude": {
            "question": "Is the hi lambda=0.03 secant plateau a real "
                        "response or a small-denominator artifact?",
            "rows": m1_rows,
            "n_in_band": n_in, "n_seeds": len(m1_rows),
            "n_secant_ge20": n_ge20,
            "verdict": m1_verdict,
        },
        "M2_secant_vs_denominator": m2,
        "M3_exponent_convergence": m3,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    # ---- human-readable summary ----------------------------------------
    print("M1 -- hi lambda=0.03, per seed (d_delta, c_delta, secant):")
    for r in m1_rows:
        band = "in-band" if r["in_band"] else "OUT"
        print(f"  s{r['seed']}: d={r['d_delta']:>7.2f}  "
              f"c={r['c_delta']:+.4f} [{band:>7}]  "
              f"secant={r['secant_pct']:>5.1f}%")
    print(f"  -> {m1_verdict}\n")

    print("M2 -- secant threshold vs denominator size:")
    print(f"  {m2['verdict']}\n")

    print("M3 -- power-law exponent across s0/s1/s2 (hi):")
    for s in (0, 1, 2):
        f = m3["fits"][f"s{s}"]
        print(f"  s{s}: full b={f['full_5pt']['b']}  "
              f"drop-smallest b={f['drop_smallest_d']['b']}  "
              f"weighted b={f['weighted_by_d']['b']}")
    for mode in ("full_5pt", "drop_smallest_d", "weighted_by_d"):
        sp = m3["spans"][mode]
        print(f"  span[{mode}] = {sp['range']}  (b = {sp['b_values']})")
    print(f"  -> {m3['verdict']}\n")

    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
