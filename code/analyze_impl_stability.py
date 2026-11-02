#!/usr/bin/env python3
"""Cross-implementation stability of the GLUE markers as OOD predictors.

The same marker, measured three ways:

    official implementation + official 224-model release
    official implementation + official 80-model release
    our own implementation + our own 48 models

Under one protocol -- ID-matched around the mean of `acc_test`, then
architecture-residualised -- a marker that is a genuine property of the
representation should behave the same in all three cells. A marker whose sign
depends on which implementation produced it is not a stable predictor, and
cannot be recommended as one.

This script writes the matrix and the band-sensitivity table that support the
central claim of the measurement-accuracy paper.

Usage:
    python analyze_impl_stability.py --outdir results
"""
from __future__ import annotations

import argparse
import json
import os
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


def group_residual(v, groups):
    """Remove the group main effects (C2 of Protocol A)."""
    v = np.asarray(v, float)
    out = v.copy()
    for g in sorted(set(groups)):
        m = np.array([x == g for x in groups])
        out[m] -= v[m].mean()
    return out


def id_matched(df, band, id_col="acc_test"):
    """C1: keep models whose ID accuracy is within `band` of the mean."""
    a = df[id_col].values.astype(float)
    m = (a >= a.mean() - band) & (a <= a.mean() + band)
    return df[m].reset_index(drop=True)


def protocol_corr(df, xcol, ycol, band, group_col="arch_name"):
    """Pearson/Spearman after C1 (ID matching) and C2 (group residualisation)."""
    sub = id_matched(df, band)
    if len(sub) < 8:
        return None, None, len(sub)
    x = sub[xcol].values.astype(float)
    y = sub[ycol].values.astype(float)
    g = sub[group_col].values
    rx, ry = group_residual(x, g), group_residual(y, g)
    r = pearsonr(rx, ry)[0]
    rho = spearmanr(rx, ry)[0]
    return float(r), float(rho), len(sub)


def own_protocol(rows, glue, key, band, xcol):
    """Same protocol on our own models, with our own GLUE implementation.

    NOTE: `id_acc` here is a fraction in [0, 1], so it is scaled to percentage
    points; the official releases already store `acc_test` in points.
    """
    acc = np.array([r["id_acc"] for r in rows]) * 100
    m = (acc >= acc.mean() - band) & (acc <= acc.mean() + band)
    sub = [r for r, k in zip(rows, m) if k]
    if len(sub) < 8:
        return None, None, len(sub)
    x = np.array([glue[r["ckpt"]][xcol] for r in sub], dtype=float)
    y = np.array([r["c100_probe_acc"] for r in sub], dtype=float) * 100
    g = [r["arch"] for r in sub]
    rx, ry = group_residual(x, g), group_residual(y, g)
    return float(pearsonr(rx, ry)[0]), float(spearmanr(rx, ry)[0]), len(sub)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--bands", type=float, nargs="+", default=[1, 2, 3, 5])
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d224 = pd.read_pickle(ROOT / "results" / "df_fig3.pkl")
    d80 = pd.read_pickle(ROOT / "results" / "official_df_fig15-20.pkl")["gaussian_noise"]
    glue_own = {r["ckpt"]: r for r in json.load(
        open(ROOT / "results" / "glue_own48_results.json"))["rows"]}
    ind = json.load(open(ROOT / "results" / "ind_validation_results.json"))["rows"]
    own = [r for r in ind if r["ckpt"] in glue_own]

    keys = [("dimension", "glue_dimension"), ("radius", "glue_radius"), ("utility", "glue_utility")]
    matrix, bands = {}, {}
    for key, owncol in keys:
        for band in args.bands:
            r1, rho1, n1 = protocol_corr(d224, f"{key}_id", "acc_lp_ood", band)
            r2, rho2, n2 = protocol_corr(d80, f"{key}_id", "acc_lp_ood", band)
            r3, rho3, n3 = own_protocol(own, glue_own, key, band, owncol)
            bands.setdefault(key, []).append(
                {"band": band, "r_224": r1, "rho_224": rho1, "n_224": n1,
                 "r_80": r2, "rho_80": rho2, "n_80": n2,
                 "r_own": r3, "rho_own": rho3, "n_own": n3})
        b = bands[key][0]
        matrix[key] = {"r_224": b["r_224"], "rho_224": b["rho_224"], "n_224": b["n_224"],
                       "r_80": b["r_80"], "rho_80": b["rho_80"], "n_80": b["n_80"],
                       "r_own": b["r_own"], "rho_own": b["rho_own"], "n_own": b["n_own"]}
        signs = [np.sign(v) for v in (b["r_224"], b["r_80"], b["r_own"])
                 if v is not None]
        matrix[key]["sign_consistent"] = len(set(signs)) == 1 if len(signs) == 3 else None

    # ---- report ----
    lines = ["# Cross-implementation stability of the GLUE markers", "",
             "Protocol: ID-matched around the mean of `acc_test` (band in percentage points),",
             "then architecture-residualised. Three cells: the official implementation on the",
             "official 224-model release, the official implementation on the official 80-model",
             "release, and our own implementation on our own 48 models.", "",
             "## The matrix (band = 1 pp)", "",
             "| Marker | official 224 | official 80 | our own 48 | sign consistent |",
             "|---|---|---|---|---|"]
    for key in matrix:
        m = matrix[key]
        def cell(r, rho, n):
            return f"n={n} (too few)" if r is None else f"r={r:+.3f} rho={rho:+.3f} (n={n})"
        lines.append(
            f"| `{key}` | {cell(m['r_224'], m['rho_224'], m['n_224'])} | "
            f"{cell(m['r_80'], m['rho_80'], m['n_80'])} | "
            f"{cell(m['r_own'], m['rho_own'], m['n_own'])} | "
            f"{'yes' if m['sign_consistent'] else 'NO'} |")
    lines += ["", "## Band sensitivity", ""]
    for key, rows_ in bands.items():
        lines += [f"### `{key}`", "",
                  "| band | official 224 | official 80 | our own 48 |", "|---|---|---|---|"]
        for b in rows_:
            def c(r, n):
                return f"n={n} (too few)" if r is None else f"r={r:+.3f} (n={n})"
            lines.append(f"| ±{b['band']:.0f} pp | {c(b['r_224'], b['n_224'])} | "
                         f"{c(b['r_80'], b['n_80'])} | {c(b['r_own'], b['n_own'])} |")
        lines.append("")

    out_md = outdir / "impl_stability_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json = outdir / "impl_stability_results.json"
    out_json.write_text(json.dumps({"matrix": matrix, "bands": bands}, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"[saved] {out_md}")
    print(f"[saved] {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
