#!/usr/bin/env python3
"""Figure 5: three-seed (Δd, Δc) dose-response (ADR-0016 App. 24/25).

Panel (a) log-log for the high-D recipe: per-seed power-law fits with the
Appendix-24 anomaly rule imported verbatim from dose_shape_fit_seeds.py,
plus the observational slope (0.0161) and the placebo band (0.022).
Panel (b) linear-y for the low-D recipe: signed responses (including the
negative ones that a log axis cannot show) inside/near the placebo band.

Input : results/p4_dose/dose_scan_seeds.json, or
        runs/p4_dose/dose_scan_seeds.json in the released archive layout
        (the two layouts ship byte-identical locked JSON)
Output: paper/causality/figs/fig5_dose.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from dose_shape_fit_seeds import (  # noqa: E402  -- shared, frozen criteria
    BAND, ID_FLOOR, LAMS, SEEDS, find_anomaly, fit_power,
)

DATA_CANDIDATES = (
    ROOT / "results" / "p4_dose" / "dose_scan_seeds.json",
    ROOT / "runs" / "p4_dose" / "dose_scan_seeds.json",
)
DATA = next((p for p in DATA_CANDIDATES if p.is_file()), DATA_CANDIDATES[0])
OUT = ROOT / "paper" / "causality" / "figs"
S_OBS = 0.0161          # observational slope, d_ent scale (verify.py E4)
COLORS = {0: "#0072B2", 1: "#D55E00", 2: "#009E73"}  # Okabe-Ito
MARKERS = {0: "o", 1: "s", 2: "^"}
SIZES = {0.03: 22, 0.1: 32, 0.3: 44, 1.0: 58, 3.0: 74}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.6,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
})


def load_curves():
    doc = json.loads(DATA.read_text(encoding="utf-8"))
    curves = {(r, s): {} for r in ("lo", "hi") for s in SEEDS}
    for p in doc["points"]:
        curves[(p["recipe"], p["seed"])][p["lambda"]] = (
            p["d_delta"], p["c_delta"], p["id_keep"])
    if doc["meta"]["missing"]:
        raise RuntimeError(f"archive has missing points: {doc['meta']['missing']}")
    return curves


def clean_points(pts):
    """Appendix-24 rule, identical to the verdict script."""
    kept = {l: p for l, p in pts.items() if p[2] >= ID_FLOOR}
    return find_anomaly(kept)


def per_seed_fit(pts):
    """返回 (a, b) 或 None；拟合口径与判定脚本完全一致。"""
    pts = clean_points(pts)
    ds = [pts[l][0] for l in sorted(pts)]
    cs = [pts[l][1] for l in sorted(pts)]
    fit, _ = fit_power(ds, cs)
    return fit


def draw_panel(ax, curves, recipe, loglog):
    for seed in SEEDS:
        pts = curves[(recipe, seed)]
        lams = sorted(pts)
        xs = [pts[l][0] for l in lams]
        ys = [pts[l][1] for l in lams]
        ax.plot(xs, ys, "-", color=COLORS[seed], lw=0.9, alpha=0.40, zorder=2)
        ax.scatter(xs, ys, s=[SIZES[l] for l in lams],
                   marker=MARKERS[seed], color=COLORS[seed],
                   edgecolors="white", linewidths=0.5, zorder=3)
        if recipe == "hi":
            fit = per_seed_fit(pts)
            if fit:
                a, b = fit
                d0 = min(pts[l][0] for l in pts)
                d1 = max(pts[l][0] for l in pts)
                xx = np.logspace(np.log10(d0 * 0.85), np.log10(d1 * 1.15), 40)
                ax.plot(xx, a * xx ** b, "--", color=COLORS[seed],
                        lw=1.3, alpha=0.95, zorder=2.5)

    ax.set_xscale("log")
    if loglog:
        ax.set_yscale("log")
        ax.set_xlim(3, 320)
        ax.set_ylim(8e-4, 0.45)
        # placebo band: below the horizontal line
        ax.axhspan(8e-4, BAND, color="#BFBFBF", alpha=0.16, zorder=0)
        ax.axhline(BAND, color="#7F7F7F", ls=":", lw=0.9, zorder=1)
        ax.text(3.4, BAND * 1.12, "placebo band", fontsize=7, color="#666666")
        # observational prediction (slope 1 in d_ent units)
        xx = np.logspace(np.log10(3), np.log10(60), 20)
        ax.plot(xx, S_OBS * xx, color="#4D4D4D", ls=(0, (2, 2)),
                lw=1.0, zorder=1)
        ax.set_ylabel(r"transfer gain $\Delta c_{100}$")
        ax.set_title("(a) High-D recipe: sub-linear power laws")
    else:
        ax.set_ylim(-0.008, 0.036)
        ax.set_xlim(1.4, 32)
        ax.axhspan(-BAND, BAND, color="#BFBFBF", alpha=0.16, zorder=0)
        ax.axhline(BAND, color="#7F7F7F", ls=":", lw=0.9, zorder=1)
        ax.axhline(-BAND, color="#7F7F7F", ls=":", lw=0.9, zorder=1)
        ax.axhline(0, color="#999999", lw=0.6, zorder=1)
        ax.set_ylabel(r"transfer gain $\Delta c_{100}$")
        ax.set_title("(b) Low-D recipe: transfer-inert")
        ax.text(1.6, 0.0325, r"band $\pm0.022$", fontsize=7, color="#666666")
    ax.set_xlabel(r"geometry change $\Delta d_{\rm ent}$")
    ax.grid(alpha=0.18, lw=0.5, which="both" if loglog else "major")
    ax.tick_params(labelsize=8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves = load_curves()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    draw_panel(axes[0], curves, "hi", loglog=True)
    draw_panel(axes[1], curves, "lo", loglog=False)

    handles = []
    for seed in SEEDS:
        fit = per_seed_fit(curves[("hi", seed)])
        label = f"s{seed}" + (f"  (b={fit[1]:.3f})" if fit else "")
        handles.append(Line2D([], [], color=COLORS[seed], marker=MARKERS[seed],
                              linestyle="-", markersize=5, label=label))
    handles.append(Line2D([], [], color="#4D4D4D", ls=(0, (2, 2)),
                          label="observational slope"))
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.03), frameon=False,
               handlelength=2.4, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5_dose.{ext}", dpi=300)
    print(f"wrote {OUT}/fig5_dose.{{pdf,png}}")
    for seed in SEEDS:
        fit = per_seed_fit(curves[("hi", seed)])
        print(f"  hi s{seed}: b={fit[1]:.3f}" if fit else f"  hi s{seed}: no fit")


if __name__ == "__main__":
    main()
