#!/usr/bin/env python3
"""Figures for the measurement-accuracy paper.

  fig1_map    the stability map: every marker, every cell, sign and magnitude
  fig2_degeneration  the two explanations side by side

Usage:
    python make_figures_measurement.py --outdir figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, here.parent, here.parent.parent]:
        if (cand / "results").is_dir():
            return cand
    return here.parent


ROOT = find_root()

C_BASE = "#D55E00"     # the accuracy baseline, vermillion
C_MARK = "#0072B2"     # geometric markers, blue
C_WARN = "#CC79A7"     # markers whose sign is not stable, reddish purple
C_GRID = "#D8D8D8"
C_TEXT = "#222222"


def style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": C_TEXT,
        "axes.labelcolor": C_TEXT,
        "xtick.color": C_TEXT,
        "ytick.color": C_TEXT,
        "pdf.fonttype": 42,
    })


def fig_map(outdir: Path):
    """The stability map: three cells per marker, sign and magnitude together."""
    d = json.load(open(ROOT / "results" / "marker_stability_map.json"))
    rows = [r for r in d["rows"] if r["official_224"] or r["own_48"]]

    labels, cells = [], []
    for r in rows:
        vals = []
        for key in ("official_224", "official_80", "own_48"):
            c = r.get(key)
            vals.append(c["r"] if c else None)
        labels.append(r["marker"])
        cells.append(vals)

    fig, ax = plt.subplots(figsize=(7.2, 0.33 * len(rows) + 1.6))
    y = np.arange(len(rows))
    h = 0.26
    styles = [("official_224", -h, "o", C_MARK, "official implementation, 224-model release"),
              ("official_80", 0.0, "s", "#009E73", "official implementation, 80-model release"),
              ("own_48", h, "^", C_WARN, "our implementation, our 48 models")]

    for (key, dy, marker, color, label), idx in zip(styles, range(3)):
        xs = [row[idx] if row[idx] is not None else np.nan for row in cells]
        ax.plot(xs, y + dy, marker, color=color, ms=5.5, label=label,
                ls="none", mec="white", mew=0.6, zorder=3)

    ax.axvline(0, color=C_TEXT, lw=0.9, zorder=1)
    for v in (-0.5, 0.5):
        ax.axvline(v, color=C_GRID, lw=0.8, ls=(0, (3, 3)), zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson $r$ with OOD accuracy, after ID matching and architecture residualisation")
    ax.set_xlim(-1.05, 1.18)
    ax.grid(axis="x", color=C_GRID, lw=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # annotate the markers whose sign is not stable across cells
    for i, r in enumerate(rows):
        if "SIGN FLIPS" not in r["classification"]:
            continue
        cs = cells[i]
        present = [v for v in cs if v is not None]
        if not present:
            continue
        lo, hi = min(present), max(present)
        # The bracket goes below the row so that it never crosses a data point.
        yb = i + h + 0.10
        ax.plot([lo - 0.05, hi + 0.05], [yb, yb],
                color=C_WARN, lw=1.2, zorder=2, solid_capstyle="butt")
        ax.text(hi + 0.08, yb, "sign flips", color=C_WARN, fontsize=7,
                va="center", ha="left", style="italic")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22 / len(rows) - 0.13),
              ncol=3, frameon=False, borderpad=0.2)
    ax.set_title("Marker stability across implementations and data sources",
                 loc="left", pad=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_map.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_map")


GLUE_FAMILIES = {"dimension", "radius", "utility"}
SHIFT_COLS = ["task", "gaussian_noise", "brightness", "contrast",
              "fog", "jpeg_compression", "pixelate"]
SHIFT_LAB = {"task": "task\nshift", "gaussian_noise": "gaussian\nnoise",
             "brightness": "bright-\nness", "contrast": "contrast",
             "fog": "fog", "jpeg_compression": "jpeg\ncompr.",
             "pixelate": "pixelate"}
VERDICT_MARK = {"flip": "F", "shift-dependent": "S", "unstable-n": "N",
                "weak-same": "w", "stable-same": "s"}


def fig_shiftmap(outdir: Path):
    """The full sign-stability map: every family, every shift, both implementations.

    Colour = mean Pearson r within a shift group (averaged over the
    implementations that provide the cell); a dot marks a group where the two
    implementations disagree in sign; grey = not measurable (n<8 or |r|<0.05).
    Rows ordered by verdict class.  Adapted from results/stability_map.json.
    """
    from matplotlib.colors import TwoSlopeNorm
    d = json.load(open(ROOT / "results" / "stability_map.json"))
    V = d["verdicts"]
    order = (["flip"], sorted(k for k, v in V.items()
                               if v["verdict"] == "flip"),
             ["shift-dependent"], sorted(k for k, v in V.items()
                                         if v["verdict"] == "shift-dependent"),
             ["unstable-n"], sorted(k for k, v in V.items()
                                    if v["verdict"] == "unstable-n"),
             ["weak-same", "stable-same"],
             [k for k, v in V.items() if v["verdict"] in ("weak-same", "stable-same")])
    fams = [f for klass in order[0::2] for f in
            (order[order.index(klass) + 1] if order.index(klass) + 1 < len(order) else [])]
    fams = []
    i = 0
    while i < len(order):
        klass, keys = order[i], order[i + 1]
        fams.extend(keys)
        i += 2

    M = np.full((len(fams), len(SHIFT_COLS)), np.nan)
    DIS = np.zeros_like(M, dtype=bool)
    for fi, fam in enumerate(fams):
        for grp, cells in V[fam]["groups"].items():
            gname = grp.replace("corruption:", "")
            ci = SHIFT_COLS.index(gname)
            rs = [s["r"] for s in cells.values() if s["r"] is not None
                  and abs(s["r"]) >= 0.05]
            if not rs:
                continue
            M[fi, ci] = float(np.mean(rs))
            DIS[fi, ci] = len(rs) >= 2 and (max(rs) > 0) != (min(rs) > 0)

    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    for fi in range(len(fams)):
        for ci in range(len(SHIFT_COLS)):
            if np.isnan(M[fi, ci]):
                ax.add_patch(plt.Rectangle((ci - .5, fi - .5), 1, 1,
                                           facecolor="#ECECEC", edgecolor="white", lw=1.2))
            elif DIS[fi, ci]:
                ax.plot(ci, fi, "o", ms=3.2, color="#111111", zorder=4)
    ax.set_xticks(range(len(SHIFT_COLS)))
    ax.set_xticklabels([SHIFT_LAB[c] for c in SHIFT_COLS], fontsize=7.5)
    labels = [("\\textbf{%s}" % f if f in GLUE_FAMILIES else f) for f in fams]
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(fams, fontsize=7.5)
    for tick, fam in zip(ax.get_yticklabels(), fams):
        if fam in GLUE_FAMILIES:
            tick.set_fontweight("bold")
    # verdict letters in the right margin
    for fi, fam in enumerate(fams):
        v = V[fam]["verdict"]
        ax.text(len(SHIFT_COLS) - 0.38, fi, VERDICT_MARK[v], fontsize=7.5,
                ha="center", va="center",
                fontweight="bold" if v in ("flip", "shift-dependent") else "normal",
                color="#B2182B" if v == "flip" else ("#2166AC" if v == "shift-dependent" else "#666666"))
    ax.text(len(SHIFT_COLS) - 0.38, -1.05, "verdict", fontsize=7.5, ha="center",
            style="italic", color="#444444")
    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.09)
    cb.set_label("mean Pearson $r$ within group", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    ax.set_title("Sign of the marker-accuracy association by shift group\n"
                 "(band $\\pm$1pp, architecture-residualised; dot = implementations disagree)",
                 loc="left", fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_shiftmap.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_shiftmap")


def fig_degeneration(outdir: Path):
    """Two panels: the degradation sweep and the recipe sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # left: degradation sweep
    ax = axes[0]
    conds = ["clean", "occ30", "occ70", "noise"]
    dim = [316, 287, 128, 271]
    probe = [0.531, 0.438, 0.264, 0.309]
    x = np.arange(len(conds))
    ax.plot(x, dim, "o-", color=C_MARK, ms=5, label="effective dim (PCA)", zorder=3)
    ax.set_ylabel("effective dimension", color=C_MARK)
    ax.tick_params(axis="y", labelcolor=C_MARK)
    ax.set_ylim(0, 380)
    ax2 = ax.twinx()
    ax2.plot(x, probe, "s--", color=C_BASE, ms=4.5, label="linear probe acc.", zorder=3)
    ax2.set_ylabel("probe accuracy", color=C_BASE)
    ax2.tick_params(axis="y", labelcolor=C_BASE)
    ax2.spines["right"].set_visible(True)
    ax.set_xticks(x)
    ax.set_xticklabels(["clean", "30%\nocc", "70%\nocc", "noise"], fontsize=8)
    ax.set_title("(a) input information varies", loc="left", fontsize=9)
    ax.grid(axis="y", color=C_GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.text(0.02, 0.06, "marker and probe fall together\n(cannot separate the two explanations)",
            transform=ax.transAxes, fontsize=7, color=C_TEXT, style="italic")

    # right: recipe sweep, within-cluster
    ax = axes[1]
    data = json.load(open(ROOT / "results" / "c2_results_seeds0-1-2.json"))["rows"]
    LOW = {"plain", "plain_ls", "noise20", "noise50", "wd_high", "lr_high"}
    HIGH = {"best", "aug_only", "noise20_aug"}
    groups = [("low cluster", C_MARK, LOW), ("high cluster", C_BASE, HIGH)]
    for gi, (glabel, gcolor, recipes) in enumerate(groups):
        xs, ys = [], []
        for r in data:
            if r["name"] in recipes:
                xs.append(r["test_acc"] * 100)
                ys.append(r["local_smoothness"])
        ax.scatter(xs, ys, color=gcolor, s=22, alpha=0.85, label=f"{glabel} (n={len(xs)})",
                   zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("test accuracy (%)")
    ax.set_ylabel("local smoothness")
    ax.set_title("(b) input information fixed", loc="left", fontsize=9)
    ax.grid(color=C_GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.98, 0.04, "no ordering within either group", transform=ax.transAxes,
            fontsize=7, color=C_TEXT, style="italic", ha="right")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_degeneration.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_degeneration")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(ROOT / "figures"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    style()
    fig_map(outdir)
    fig_shiftmap(outdir)
    fig_degeneration(outdir)
    print(f"figures written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
