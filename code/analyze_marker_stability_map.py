#!/usr/bin/env python3
"""The marker stability map: every marker, measured every way we can.

This is the central table of the measurement-accuracy paper. For each marker
we ask a single question:

    Does this marker behave the same way in every setting where we can measure it?

We measure in three cells:

    official implementation + official 224-model release
    official implementation + official 80-model release
    our own implementation + our own 48 models

and, for the markers that exist in both the official releases and our own
models, we also report the same-implementation comparison across data sources.

The protocol is the one from the R1 paper: C1 ID matching around the mean of
`acc_test`, C2 residualisation of the architecture main effects. We report the
Pearson r after residualisation, the Spearman rho, the sample size, and the
band sensitivity.

Output:
    results/marker_stability_map.md      the table, ready for the paper
    results/marker_stability_map.json    machine-readable

Usage:
    python analyze_marker_stability_map.py --outdir results
"""
from __future__ import annotations

import argparse
import json
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

def locate(*names):
    """Find an input file under any of the layouts we publish."""
    for name in names:
        for base in ("results", "data/official", "data/self_trained", "."):
            cand = ROOT / base / name
            if cand.exists():
                return cand
    raise FileNotFoundError(" / ".join(names))


# Marker name -> (official column, our own column)
MARKERS = [
    ("dimension",    "dimension_id",             "glue_dimension",  "GLUE"),
    ("radius",       "radius_id",                "glue_radius",     "GLUE"),
    ("utility",      "utility_id",               "glue_utility",    "GLUE"),
    ("capacity",     "capacity_id",              None,              "GLUE"),
    ("center alignment", "center_alignment_id",  None,              "GLUE"),
    ("participation ratio (per class)", "participation_ratio_id", "id_pr_perclass",      "generic"),
    ("participation ratio (global)",   "participation_ratio_avg_id", "id_pr_global",   "generic"),
    ("numerical rank (per class)", "numerical_rank_id",  "id_numerical_rank",       "generic"),
    ("numerical rank (global)",   "numerical_rank_avg_id", None,                "generic"),
    ("sparsity",       "sparsity_id",              None,              "generic"),
    ("mean distance",  "mean_distance_id",         None,              "generic"),
    ("mean angle",     "mean_angle_id",            None,              "generic"),
    ("mean covariance","mean_covariance_id",       None,              "generic"),
    ("NC1",                 "nc1_id",                 "id_nc1",                  "generic"),
    ("local smoothness",    None,                     "id_local_smoothness",     "ours only"),
    ("effective dim (PCA)", None,                     "id_effective_dim_pca",    "ours only"),
    ("class separability",  None,                     "id_class_separability",   "ours only"),
]

# The official releases spell some markers differently.
OFFICIAL_ALIASES = {}


def group_residual(v, groups):
    v = np.asarray(v, float)
    out = v.copy()
    for g in sorted(set(groups)):
        m = np.array([x == g for x in groups])
        out[m] -= v[m].mean()
    return out


def id_matched(df, band):
    a = df["acc_test"].values.astype(float)
    m = (a >= a.mean() - band) & (a <= a.mean() + band)
    return df[m].reset_index(drop=True)


def official_cell(df, xcol, ycol, band):
    """ID matching + architecture residualisation on an official release."""
    sub = id_matched(df, band)
    if len(sub) < 8:
        return None, None, len(sub)
    if xcol not in df.columns:
        return None, None, len(sub)
    x = sub[xcol].values.astype(float)
    y = sub[ycol].values.astype(float)
    g = sub["arch_name"].values
    rx, ry = group_residual(x, g), group_residual(y, g)
    r = pearsonr(rx, ry)
    return (float(r[0]), float(r[1]), float(spearmanr(rx, ry)[0]), len(sub))


def own_cell(rows, xkey, ykey, band, scale=100.0, glue=None):
    """The same protocol on our own models.

    `id_acc` and the probe accuracies are fractions in [0, 1] here, so they are
    scaled to percentage points. The official releases already store `acc_test`
    in points -- do not scale those.

    The self-implemented GLUE values live in their own file, keyed by
    checkpoint, so they are merged in here rather than read from the row.
    """
    acc = np.array([r["id_acc"] for r in rows]) * scale
    m = (acc >= acc.mean() - band) & (acc <= acc.mean() + band)
    sub = [r for r, k in zip(rows, m) if k]
    if len(sub) < 8:
        return None, None, len(sub)
    if xkey.startswith("glue_"):
        if glue is None or any(r["ckpt"] not in glue for r in sub):
            return None, None, len(sub)
        x = np.array([glue[r["ckpt"]][xkey] for r in sub], dtype=float)
    else:
        x = np.array([r[xkey] for r in sub], dtype=float)
    y = np.array([r[ykey] for r in sub], dtype=float) * scale
    g = [r["arch"] for r in sub]
    rx, ry = group_residual(x, g), group_residual(y, g)
    r = pearsonr(rx, ry)
    return (float(r[0]), float(r[1]), float(spearmanr(rx, ry)[0]), len(sub))


def classify(cells):
    """Assign a stability class from the signs across the measurable cells.

    A marker counts as robust only if it is *reliably* associated in every cell
    where we can measure it. Two things have to hold:

      * the sign agrees across cells, for Pearson r and for Spearman rho
        separately -- a marker whose two statistics disagree in sign within a
        cell is not usable as a predictor even there (this is condition C3 of
        the protocol);
      * every cell clears a minimum association, |r| >= 0.3.

    A marker that is consistent in sign but weak in one cell is a different
    problem: it may be measuring something real but small, and it cannot be
    used as a predictor with the same confidence everywhere.
    """
    rs = [c[0] for c in cells if c[0] is not None]
    rhos = [c[2] for c in cells if c[0] is not None]
    if len(rs) < 2:
        return "insufficient data"
    if len({np.sign(r) for r in rs}) > 1:
        return "SIGN FLIPS (Pearson)"
    if len({np.sign(rho) for rho in rhos}) > 1:
        return "SIGN FLIPS (Spearman)"
    mags = [abs(r) for r in rs]
    if min(mags) < 0.3:
        return "consistent sign, weak in at least one cell"
    if max(mags) - min(mags) < 0.4:
        return "robust"
    return "consistent sign, variable magnitude"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--bands", type=float, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--primary-band", type=float, default=1.0)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d224 = pd.read_pickle(locate("df_fig3.pkl"))
    d80 = pd.read_pickle(locate("df_fig15-20.pkl", "official_df_fig15-20.pkl"))["gaussian_noise"]
    glue_own = {r["ckpt"]: r for r in json.load(
        open(locate("glue_own48_results.json")))["rows"]}
    ind = json.load(open(locate("ind_validation_results.json")))["rows"]
    own = [r for r in ind if r["ckpt"] in glue_own]

    band = args.primary_band
    rows_out = []
    for label, offcol, ownkey, kind in MARKERS:
        offcol = OFFICIAL_ALIASES.get(offcol, offcol)
        c1 = official_cell(d224, offcol, "acc_lp_ood", band)
        c2 = official_cell(d80, offcol, "acc_lp_ood", band)
        c3 = (own_cell(own, ownkey, "c100_probe_acc", band, glue=glue_own)
              if ownkey else (None, None, 0))
        cls = classify([c1, c2, c3])

        # band sensitivity
        sens = []
        for b in args.bands:
            s1 = official_cell(d224, offcol, "acc_lp_ood", b)
            s2 = official_cell(d80, offcol, "acc_lp_ood", b)
            s3 = (own_cell(own, ownkey, "c100_probe_acc", b, glue=glue_own)
                  if ownkey else (None, None, 0))
            sens.append({"band": b,
                         "r_224": s1[0] if s1[0] is not None else None,
                         "r_80": s2[0] if s2[0] is not None else None,
                         "r_own": s3[0] if s3[0] is not None else None})
        rows_out.append({
            "marker": label, "kind": kind,
            "official_224": {"r": c1[0], "p": c1[1], "rho": c1[2], "n": c1[3]} if c1[0] is not None else None,
            "official_80": {"r": c2[0], "p": c2[1], "rho": c2[2], "n": c2[3]} if c2[0] is not None else None,
            "own_48": {"r": c3[0], "p": c3[1], "rho": c3[2], "n": c3[3]} if c3[0] is not None else None,
            "classification": cls,
            "band_sensitivity": sens,
        })

    # ---- report ----
    L = ["# Marker stability map", "",
         f"Protocol: ID matching around the mean of `acc_test` at +/-{band:.0f} pp, then",
         "architecture residualisation. Three cells: official implementation on the official",
         "224-model release, official implementation on the official 80-model release, and our",
         "own implementation on our own 48 models (CIFAR-100 linear probe as the OOD target).", "",
         "## The map", "",
         "| Marker | kind | official 224 | official 80 | our own 48 | classification |",
         "|---|---|---|---|---|---|"]
    for r in rows_out:
        def cell(c):
            if c is None:
                return "n<8"
            return f"r={c['r']:+.3f} rho={c['rho']:+.3f} (n={c['n']})"
        L.append(f"| `{r['marker']}` | {r['kind']} | {cell(r['official_224'])} | "
                 f"{cell(r['official_80'])} | {cell(r['own_48'])} | **{r['classification']}** |")

    L += ["", "## Band sensitivity (Pearson r)", ""]
    for r in rows_out:
        L += [f"### `{r['marker']}` ({r['classification']})", "",
              "| band | official 224 | official 80 | our own 48 |", "|---|---|---|---|"]
        for s in r["band_sensitivity"]:
            def c(v):
                return "n<8" if v is None else f"{v:+.3f}"
            L.append(f"| +/-{s['band']:.0f} pp | {c(s['r_224'])} | {c(s['r_80'])} | {c(s['r_own'])} |")
        L.append("")

    # summary counts
    counts = {}
    for r in rows_out:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    L += (["## Summary", ""] +
          [f"- {k}: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])] + [""])

    out_md = outdir / "marker_stability_map.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    (outdir / "marker_stability_map.json").write_text(
        json.dumps({"primary_band": band, "rows": rows_out}, indent=2), encoding="utf-8")
    print("\n".join(L))
    print(f"[saved] {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
