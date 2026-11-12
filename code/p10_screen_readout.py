#!/usr/bin/env python3
"""P10 screening-round readout (ADR-0017 Section 6, G1-G4 frozen).

Assemble the 20-curve adjudication pool (5 seeds x 4 draws each):

  s0: v2(seed0), P6D(seed1), P8(seed2), P10(seed3, lam0.3 = P5 c1_seed3)
  s1: v2(seed0), P7E(seed1), P8(seed2), P10(seed3)
  s2: v2(seed0), P7E(seed1), P8(seed2), P10(seed3)
  s3: v2(seed0), P10(seed3/4/5)
  s4: v2(seed0), P10(seed3/4/5)

  G1 excursion flag: |b - median(b of same seed)| >= 0.08
  G2 excursion rate: k/20 with Wilson 95% CI; >=15% -> routine
     property; <15% and zero new -> seed-specific downgrade.
  G3 stratification: s3/s4 rates vs s0/s1/s2 rates.
  G4 id_keep >= -0.02 on all 44 runs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_TBL = ROOT / "results" / "p4_full" / "full_base_lr0.125.json"
V2 = ROOT / "results" / "p4_dose" / "dose_scan_seeds_v2.json"
P5_DIR = ROOT / "results" / "p5_collapse"
P6_DIR = ROOT / "results" / "p6_d_curve"
P7_DIR = ROOT / "results" / "p7_band"
P8_DIR = ROOT / "results" / "p8_bdist"
E_DIR = ROOT / "results" / "p10_screen"
OUT = E_DIR / "p10_screen_readout.json"

S_OBS = 0.0161
ID_FLOOR = -0.02
LAMS = (0.03, 0.1, 0.3, 1.0, 3.0)
EXC_THRESH = 0.08     # G1 frozen
RATE_THRESH = 0.15    # G2 frozen


def fit_power(points):
    pts = [(math.log(p["d_delta"]), math.log(abs(p["c_delta"])))
           for p in points if p["c_delta"] != 0 and p["d_delta"] > 0]
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / sxx


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main() -> int:
    bases = {r["model"]: r for r in
             json.loads(BASE_TBL.read_text(encoding="utf-8"))["rows"]}
    v2 = json.loads(V2.read_text(encoding="utf-8"))["points"]

    def dd_dc(path, seed):
        row = json.loads(Path(path).read_text(encoding="utf-8"))["rows"][0]
        b = bases[f"resnet18_lr0.125_wd0.0001_s{seed}"]
        return (round(row["d_ent_id"] - b["d_ent_id"], 3),
                round(row["c100_probe"] - b["c100_probe"], 4),
                round(row["id_probe_acc"] - b["id_probe_acc"], 4))

    def v2_pt(seed, lam):
        # s3/s4 seed-0 curves: lam 0.03/0.3/3.0 live in the v2 archive,
        # lam 0.1/1.0 in the anchor batch (ADR-0016 App. 30)
        try:
            p = next(q for q in v2 if q["recipe"] == "hi"
                     and q["seed"] == seed and q["lambda"] == lam)
            return (p["d_delta"], p["c_delta"], p["id_keep"])
        except StopIteration:
            row = json.loads((ROOT / "results" / "p4s34_anchor" /
                              f"dose_seed_s{seed}_hi_lam{lam}.json")
                             .read_text(encoding="utf-8"))["rows"][0]
            b = bases[f"resnet18_lr0.125_wd0.0001_s{seed}"]
            return (round(row["d_ent_id"] - b["d_ent_id"], 3),
                    round(row["c100_probe"] - b["c100_probe"], 4),
                    round(row["id_probe_acc"] - b["id_probe_acc"], 4))

    pool = []      # dicts: seed, tag, rows
    gate_min = 1.0
    for s in (0, 1, 2, 3, 4):
        for rs in (0, 1, 2, 3):
            if s in (3, 4) and rs not in (0, 3, 4, 5):
                continue
            if s in (0, 1, 2) and rs not in (0, 1, 2, 3):
                continue
            rows, tag = [], f"s{s}_seed{rs}"
            for lam in LAMS:
                if rs == 0:
                    d, c, ik = v2_pt(s, lam)
                    src = "v2"
                elif s == 0 and lam == 0.3 and rs == 3:
                    d, c, ik = dd_dc(P5_DIR / "collapse_c1_seed3.json", 0)
                    src = "p5_c1_seed3 (frozen reuse)"
                else:
                    base = {1: P6_DIR, 2: P7_DIR}.get(rs, E_DIR)
                    if s == 0 and rs == 1:
                        base = P6_DIR
                    if (s, rs) in ((1, 1), (2, 1)):
                        base = P7_DIR
                    if (s, rs) in ((0, 2), (1, 2), (2, 2)):
                        base = P8_DIR
                    if rs == 3 or (s in (3, 4) and rs in (4, 5)):
                        base = E_DIR
                    name = (P5_DIR / f"collapse_c1_seed{rs}.json"
                            if False else None)
                    fname = {1: f"p6_d_s0_seed1_lam{lam}.json",
                             }.get(rs) if s == 0 else None
                    # explicit path per (s, rs); reused lam-0.3 points
                    # (s0 seed1/seed2/seed3 -> P5 c1 reads) resolved first
                    if (s, rs) == (0, 1) and lam == 0.3:
                        p = P5_DIR / "collapse_c1_seed1.json"
                    elif (s, rs) == (0, 2) and lam == 0.3:
                        p = P5_DIR / "collapse_c1_seed2.json"
                    elif (s, rs) == (0, 1):
                        p = P6_DIR / f"p6_d_s0_seed1_lam{lam}.json"
                    elif (s, rs) == (1, 1) or (s, rs) == (2, 1):
                        p = P7_DIR / f"p7_e_s{s}_seed1_lam{lam}.json"
                    elif (s, rs) in ((0, 2), (1, 2), (2, 2)):
                        p = P8_DIR / f"p8_s{s}_seed2_lam{lam}.json"
                    else:  # rs >= 3 new P10 runs
                        p = E_DIR / f"p10_s{s}_seed{rs}_lam{lam}.json"
                    d, c, ik = dd_dc(p, s)
                    src = str(p.relative_to(ROOT))
                gate_min = min(gate_min, ik)
                rows.append({"lambda": lam, "d_delta": d, "c_delta": c,
                             "id_keep": ik, "pass_gate": ik >= ID_FLOOR,
                             "src": src})
            fit_rows = [r for r in rows if r["pass_gate"]]
            b = fit_power(fit_rows) if len(fit_rows) >= 3 else None
            pool.append({"seed": s, "tag": tag, "b": b,
                         "n_fitted": len(fit_rows), "rows": rows})

    # G1 flags per seed
    per_seed = {}
    for s in (0, 1, 2, 3, 4):
        bs = [e["b"] for e in pool if e["seed"] == s]
        med = median(bs)
        per_seed[s] = {"b_draws": [round(x, 3) for x in bs],
                       "core_median": round(med, 3)}
        for e in pool:
            if e["seed"] == s:
                e["excursion"] = abs(e["b"] - med) >= EXC_THRESH
                e["dev"] = round(e["b"] - med, 3)

    n_exc = sum(1 for e in pool if e["excursion"])
    lo, hi = wilson_ci(n_exc, len(pool))
    new_exc = [e["tag"] for e in pool if e["excursion"]
               and not (e["tag"] in ("s0_seed0", "s2_seed1"))]
    old_exc = [e["tag"] for e in pool if e["excursion"]
               and e["tag"] in ("s0_seed0", "s2_seed1")]

    if n_exc / len(pool) >= RATE_THRESH:
        g2 = (f"ROUTINE PROPERTY: rate {n_exc}/{len(pool)} "
              f">= 15% -- cross-seed b designs must model excursions "
              f"(report per-seed medians).")
    elif not new_exc:
        g2 = (f"SEED-SPECIFIC DOWNGRADE: rate {n_exc}/{len(pool)} < 15% "
              f"with zero new excursions -- excursions concentrate in "
              f"s0/s2; wording downgrades to 'occasional, seed-specific'.")
    else:
        g2 = (f"INTERMEDIATE: rate {n_exc}/{len(pool)} "
              f"(CI [{lo:.2f}, {hi:.2f}]) -- report as-is.")

    s012 = [e for e in pool if e["seed"] in (0, 1, 2)]
    s34 = [e for e in pool if e["seed"] in (3, 4)]
    r012 = sum(e["excursion"] for e in s012)
    r34 = sum(e["excursion"] for e in s34)
    g3 = (f"stratification: s0/s1/s2 {r012}/{len(s012)} vs "
          f"s3/s4 {r34}/{len(s34)}"
          + (" -- excursions are seed-generic" if r34 > 0
             else " -- no excursions in s3/s4; stratification hypothesis "
                  "strengthens (concentrated in specific seeds)"))

    result = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "script": "code/p10_screen_readout.py",
            "frozen_rules": "ADR-0017 Section 6 (2026-11-12 08:05)",
            "constants": {"excursion_thresh": EXC_THRESH,
                          "rate_thresh": RATE_THRESH,
                          "id_floor": ID_FLOOR},
        },
        "pool": [{k: e[k] for k in ("seed", "tag", "b", "dev",
                                    "excursion", "n_fitted")}
                 for e in pool],
        "G1_per_seed": {f"s{s}": per_seed[s] for s in (0, 1, 2, 3, 4)},
        "G2_rate": {"n_excursion": n_exc, "n_pool": len(pool),
                    "wilson_ci": [round(lo, 3), round(hi, 3)],
                    "new_excursions": new_exc, "known_excursions": old_exc,
                    "verdict": g2},
        "G3_stratification": {"s012": f"{r012}/{len(s012)}",
                              "s34": f"{r34}/{len(s34)}", "note": g3},
        "G4_gate": {"id_keep_min": round(gate_min, 4),
                    "floor": ID_FLOOR, "all_pass": gate_min >= ID_FLOOR},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    for s in (0, 1, 2, 3, 4):
        entries = [e for e in pool if e["seed"] == s]
        line = f"s{s}: " + " ".join(
            f"{e['tag'].split('_')[1]}={e['b']:.3f}"
            + ("*" if e["excursion"] else "") for e in entries)
        line += f"  | core {per_seed[s]['core_median']}"
        print(line)
    print(f"\nG2: {g2}")
    print(f"    CI [{lo:.3f}, {hi:.3f}] | new: {new_exc} | known: {old_exc}")
    print(f"G3: {g3}")
    print(f"G4: id_keep min {gate_min:+.4f} "
          f"({'all pass' if gate_min >= ID_FLOOR else 'FAIL'})")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
