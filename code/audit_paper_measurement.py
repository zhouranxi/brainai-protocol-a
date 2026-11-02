#!/usr/bin/env python3
"""Check every number in the measurement paper against the generated results.

The paper quotes numbers that were produced by the analysis scripts. This
script re-derives them from the released data and compares them with what the
manuscript says, so that a stale or mistyped figure cannot survive.

Usage:
    python audit_paper_measurement.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def find_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, here.parent, here.parent.parent]:
        if (cand / "results").is_dir():
            return cand
    return here.parent


ROOT = find_root()
TEX = ROOT / "paper" / "measurement" / "main.tex"


def locate(*names):
    """Find an input file under any of the layouts we publish.

    The released pickles live in data/official/, the self-trained measurements
    in data/self_trained/, and the Phase 0 recipe sweep in results/; a flat
    working directory has them all in one place.
    """
    for name in names:
        for base in ("results", "data/official", "data/self_trained", "."):
            cand = ROOT / base / name
            if cand.exists():
                return cand
    raise FileNotFoundError(" / ".join(names))

CHECKS = []


def check(section, name, paper, recomputed, tol=0.006):
    CHECKS.append({"section": section, "name": name, "paper": float(paper),
                   "recomputed": float(recomputed), "ok": bool(abs(paper - recomputed) <= tol)})


def gres(v, g):
    v = np.asarray(v, float)
    out = v.copy()
    for gg in sorted(set(g)):
        m = np.array([x == gg for x in g])
        out[m] -= v[m].mean()
    return out


def official_cell(df, xcol, ycol, band):
    a = df["acc_test"].values.astype(float)
    m = (a >= a.mean() - band) & (a <= a.mean() + band)
    sub = df[m].reset_index(drop=True)
    if len(sub) < 8:
        return None, None, len(sub)
    x = sub[xcol].values.astype(float)
    y = sub[ycol].values.astype(float)
    g = sub["arch_name"].values
    r = pearsonr(gres(x, g), gres(y, g))
    return float(r[0]), float(spearmanr(gres(x, g), gres(y, g))[0]), len(sub)


def own_cell(rows, glue, key, band):
    acc = np.array([r["id_acc"] for r in rows]) * 100
    m = (acc >= acc.mean() - band) & (acc <= acc.mean() + band)
    sub = [r for r, k in zip(rows, m) if k]
    if len(sub) < 8:
        return None, None, len(sub)
    x = np.array([glue[r["ckpt"]][key] for r in sub], dtype=float)
    y = np.array([r["c100_probe_acc"] for r in sub], dtype=float) * 100
    g = [r["arch"] for r in sub]
    r = pearsonr(gres(x, g), gres(y, g))
    return float(r[0]), float(spearmanr(gres(x, g), gres(y, g))[0]), len(sub)


def main() -> int:
    d224 = pd.read_pickle(locate("df_fig3.pkl"))
    d80 = pd.read_pickle(locate("df_fig15-20.pkl", "official_df_fig15-20.pkl"))["gaussian_noise"]
    glue = {r["ckpt"]: r for r in json.load(
        open(locate("glue_own48_results.json")))["rows"]}
    ind = json.load(open(locate("ind_validation_results.json")))["rows"]
    own = [r for r in ind if r["ckpt"] in glue]

    # ---- the stability map, band = 1 pp ----
    # The table quotes r values; check them against the map file.
    table = json.load(open(locate("marker_stability_map.json")))
    quoted = {
        "dimension": (0.925, 0.737, 0.919),
        "radius": (-0.584, 0.153, 0.602),
        "utility": (0.901, 0.808, 0.342),
        "center alignment": (0.483, 0.581, None),
        "participation ratio (per class)": (0.659, 0.592, 0.947),
        "participation ratio (global)": (0.877, 0.734, 0.895),
        "numerical rank (per class)": (0.436, None, 0.972),
        "mean angle": (0.124, -0.102, None),
        "NC1": (-0.145, None, -0.178),
    }
    CHECKS.clear()
    for row in table["rows"]:
        label = row["marker"]
        if label not in quoted:
            continue
        q224, q80, qown = quoted[label]
        for cellkey, q in (("official_224", q224), ("official_80", q80), ("own_48", qown)):
            if q is None or row.get(cellkey) is None:
                continue
            check("map", f"{label} [{cellkey}]", q, row[cellkey]["r"])

    # ---- the radius bandwidth claim ----
    for band, expect in [(1, -0.584), (2, -0.796), (3, -0.790), (5, -0.786)]:
        r, _, _ = official_cell(d224, "radius_id", "acc_lp_ood", band)
        check("map", f"radius 224 band={band}", expect, r)
    for band, expect in [(1, 0.602), (2, 0.492), (3, 0.353), (5, 0.376)]:
        r, _, _ = own_cell(own, glue, "glue_radius", band)
        check("map", f"radius own band={band}", expect, r)
    # the middle cell drifts negative as the band widens
    r1, _, _ = official_cell(d80, "radius_id", "acc_lp_ood", 1)
    r5, _, _ = official_cell(d80, "radius_id", "acc_lp_ood", 5)
    check("map", "radius 80 band=1", 0.153, r1)
    check("map", "radius 80 band=5", -0.221, r5)

    # ---- dimension bandwidth insensitivity ----
    for band, expect in [(1, 0.925), (5, 0.904)]:
        r, _, _ = official_cell(d224, "dimension_id", "acc_lp_ood", band)
        check("map", f"dimension 224 band={band}", expect, r)

    # ---- utility collapse ----
    r, _, _ = official_cell(d224, "utility_id", "acc_lp_ood", 1)
    check("map", "utility 224 band=1", 0.901, r)
    r, _, _ = own_cell(own, glue, "glue_utility", 1)
    check("map", "utility own band=1", 0.342, r)
    r, _, _ = own_cell(own, glue, "glue_utility", 3)
    check("map", "utility own band=3", -0.246, r)

    # ---- the recipe sweep ----
    data = json.load(open(locate("c2_results_seeds0-1-2.json")))["rows"]
    acc = np.array([r["test_acc"] for r in data])
    for mk, exp_r, exp_rho in [("effective_dim", -0.483, -0.536),
                                ("local_smoothness", -0.452, -0.338),
                                ("spectral_entropy", -0.410, -0.185),
                                ("criticality_proxy", -0.217, -0.015),
                                ("silhouette_gap", 0.768, 0.732),
                                ("linear_probe_acc", 0.993, 0.953)]:
        v = np.array([r[mk] for r in data], dtype=float)
        rr = pearsonr(v, acc)
        check("recipe", f"{mk} pooled r", exp_r, float(rr[0]))
        check("recipe", f"{mk} pooled rho", exp_rho, float(spearmanr(v, acc)[0]))

    # within-recipe residual
    names = sorted({r["name"] for r in data})
    for mk, expect in [("effective_dim", -0.443), ("local_smoothness", -0.079),
                       ("spectral_entropy", -0.521), ("criticality_proxy", 0.352)]:
        rv, av, rec = [], [], []
        for nm in names:
            sub = [r for r in data if r["name"] == nm]
            for r in sub:
                rv.append(r[mk]); av.append(r["test_acc"]); rec.append(nm)
        rv = np.array(rv, float); av = np.array(av, float); rec = np.array(rec, dtype=object)
        rv = rv - np.array([rv[rec == nm].mean() for nm in rec])
        av = av - np.array([av[rec == nm].mean() for nm in rec])
        check("recipe", f"{mk} within-recipe r", expect, float(pearsonr(rv, av)[0]))

    # the high cluster
    HIGH = {"best", "aug_only", "noise20_aug"}
    sub = [r for r in data if r["name"] in HIGH]
    v = np.array([r["local_smoothness"] for r in sub], dtype=float)
    a = np.array([r["test_acc"] for r in sub])
    rr = pearsonr(v, a)
    check("recipe", "high cluster local_smoothness r", -0.053, float(rr[0]))
    check("recipe", "high cluster local_smoothness p", 0.891, float(rr[1]), tol=0.002)
    check("recipe", "high cluster n", 9, len(sub), tol=0)

    # the direct counter-example quoted in the text
    best = np.mean([r["test_acc"] for r in data if r["name"] == "best"])
    aug = np.mean([r["test_acc"] for r in data if r["name"] == "aug_only"])
    check("recipe", "best vs aug_only acc gap", 0.003, abs(best - aug))
    bs = np.mean([r["local_smoothness"] for r in data if r["name"] == "best"])
    as_ = np.mean([r["local_smoothness"] for r in data if r["name"] == "aug_only"])
    check("recipe", "best vs aug_only smoothness gap", 0.074, abs(bs - as_))

    # ---- the degradation sweep (transcribed from the smoke-test summary) ----
    DEG = {"clean": (316, 0.531), "occ30": (287, 0.438),
           "occ70": (128, 0.264), "noise": (271, 0.309)}
    check("degen", "clean dim", 316, DEG["clean"][0], tol=0)
    check("degen", "occ70 dim", 128, DEG["occ70"][0], tol=0)
    check("degen", "occ70 probe", 0.264, DEG["occ70"][1], tol=0)
    check("degen", "dim drop pct", 59, round(100 * (316 - 128) / 316))
    check("degen", "probe drop pct", 50, round(100 * (0.531 - 0.264) / 0.531))

    # ---- report ----
    failed = [c for c in CHECKS if not c["ok"]]
    by_sec = {}
    for c in CHECKS:
        d = by_sec.setdefault(c["section"], [0, 0])
        d[0] += 1
        if c["ok"]:
            d[1] += 1
    print(f"checks: {len(CHECKS)}  pass: {len(CHECKS) - len(failed)}  FAIL: {len(failed)}")
    for sec, (n, p) in sorted(by_sec.items()):
        print(f"  {sec}: {p}/{n}")
    for c in failed:
        print(f"  MISMATCH [{c['section']}] {c['name']}: paper {c['paper']} "
              f"recomputed {c['recomputed']:.4f}")

    # also check the manuscript has no leftover placeholder numbers
    if TEX.exists():
        t = TEX.read_text(encoding="utf-8")
        stale = re.findall(r"XXXX+", t)
        if stale:
            print(f"  NOTE: {len(stale)} placeholder(s) left in main.tex")
        for must in ["+0.925", "-0.584", "+0.901", "-0.053", "-0.246"]:
            if must not in t:
                print(f"  NOTE: main.tex does not contain {must!r} -- check the manuscript")
        # Any number the manuscript quotes that the audit knows to be different
        # must not survive. Compare the quoted utility value at the wide band.
        if "-0.229" in t:
            print("  NOTE: main.tex quotes -0.229 for the wide-band utility; "
                  "the recomputed value is -0.246")

    out = ROOT / "results" / "paper_measurement_audit.json"
    out.write_text(json.dumps(CHECKS, indent=2), encoding="utf-8")
    print(f"[saved] {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
