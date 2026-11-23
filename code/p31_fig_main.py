#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mechanism paper main figure: stage-resolved convergence curves.

Panel (a): endgame-metric (top-1) trajectories across evidence stages --
           self-critique's early lead vanishing at the endgame.
Panel (b): candidate-set recall (gold in top-3) trajectories --
           hypothesis maintenance's early-recall advantage.

Data: merged 90 puzzles (ADR-0030 v2 batch-1 30 @ temp0 + ADR-0031
confirmation 60 @ temp0.7x3 median rank). Numbers match paper Table 2
and the analysis notebook; regenerated from raw JSON below.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "mechanism" / "figs" / "fig_main.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def load_merged():
    """Return med[(mech, stage)][puzzle] = rank for merged 90 puzzles."""
    med = defaultdict(dict)

    rows31 = json.loads(
        (ROOT / "runs" / "v2_p31" / "p31_race.json").read_text(encoding="utf-8")
    )["rows"]
    raw = defaultdict(dict)
    for r in rows31:
        raw[(r["mech"], r["grade"], r["rep"])][r["puzzle"]] = r["rank"]
    for (m, g, _rep), pm in raw.items():
        if g in (1, 2, 3) and _rep == 0:
            pass
    # median over reps per (mech, grade, puzzle)
    acc = defaultdict(list)
    for (m, g, rep), pm in raw.items():
        for p, rk in pm.items():
            acc[(m, g, p)].append(rk)
    for (m, g, p), rks in acc.items():
        med[(m, g)][p] = median(rks)

    rows30 = json.loads(
        (ROOT / "runs" / "v2_p25" / "p25_race_v2.json").read_text(encoding="utf-8")
    )["rows"]
    for r in rows30:
        med[(r["mech"], r["grade"])][r["puzzle"]] = r["rank"]
    return med


def curves(med, pids, metric):
    """metric in {'top1','recall'} -> {mech: [rate@G1, rate@G2, rate@G3]}"""
    out = {}
    for m in "abcd":
        ys = []
        for g in (1, 2, 3):
            rs = [med[(m, g)][p] for p in pids]
            if metric == "top1":
                ys.append(sum(1 for x in rs if x == 1) / len(rs))
            else:
                ys.append(sum(1 for x in rs if x in (1, 2, 3)) / len(rs))
        out[m] = ys
    return out


def main():
    med = load_merged()
    pids = sorted({p for (m, g), pm in med.items() if g == 3 for p in pm})
    assert len(pids) == 90, f"expected 90 puzzles, got {len(pids)}"

    top1 = curves(med, pids, "top1")
    recall = curves(med, pids, "recall")

    # paper Table 2 cross-check
    assert [round(100 * top1[m][2]) for m in "abcd"] == [76, 76, 73, 77], \
        [round(100 * top1[m][2]) for m in "abcd"]

    style = {
        "a": dict(color="#8c8c8c", ls=":", lw=1.8, marker="o", ms=6,
                  label="Stateless (M-a)"),
        "b": dict(color="#e0a13a", ls="--", lw=1.8, marker="s", ms=6,
                  label="Lock-in (M-b)"),
        "c": dict(color="#2f6db4", ls="-", lw=2.4, marker="D", ms=6.5,
                  label="Maintenance (M-c)"),
        "d": dict(color="#c0392b", ls="-", lw=2.4, marker="o", ms=6.5,
                  label="Self-critique (M-d)"),
    }
    stages = [1, 2, 3]
    xlabels = ["Stage 1\n(category)", "Stage 2\n(strong cue)",
               "Stage 3\n(decisive cue)"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=300)
    panels = [("a", top1, "Top-1 accuracy (gold ranked #1)",
               "endgame metric"),
              ("b", recall, "Top-3 recall (gold in candidate set)",
               "candidate-set metric")]

    for ax, (key, data, title, note) in zip(axes, panels):
        for m in "abcd":
            ax.plot(stages, [100 * y for y in data[m]], **style[m])
        ax.set_xticks(stages)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylim(15, 100)
        ax.set_ylabel("puzzles (%)", fontsize=9)
        ax.set_title(f"{title}\n({note})", fontsize=9.5)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    ax = axes[0]
    # stage-resolved p-values M-d vs M-c (merged paired sign tests)
    for g, txt, dy in [(1, "p=.019", 2.2), (2, r"p<$10^{-4}$", 2.2),
                       (3, "p=.71", -4.6)]:
        yc = 100 * top1["c"][g - 1]
        yd = 100 * top1["d"][g - 1]
        y0 = max(yc, yd) + dy
        ax.annotate(txt, xy=(g, y0), ha="center", fontsize=7.5,
                    color="#444444")
    ax.annotate("early lead\ngone", xy=(3, 100 * top1["d"][2] + 6.5),
                ha="center", fontsize=8, color="#c0392b", style="italic")

    ax = axes[1]
    mid = (100 * recall["c"][1] + 100 * recall["a"][1]) / 2
    ax.annotate("+33 pp", xy=(2.06, mid), fontsize=8.5, color="#2f6db4",
                fontweight="bold")
    ax.annotate("", xy=(2.02, 100 * recall["c"][1]),
                xytext=(2.02, 100 * recall["a"][1]),
                arrowprops=dict(arrowstyle="<->", color="#2f6db4", lw=1.2))
    ax.text(1.0, 22, "stateless/lock-in list a single answer at early\n"
                     "stages, so their recall coincides with top-1;\n"
                     "maintenance keeps 3 live candidates",
            fontsize=7, color="#555555", va="bottom")

    handles = [plt.Line2D([], [], **style[m]) for m in "abcd"]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, bbox_inches="tight")
    print("written:", OUT)
    print("top1  :", {m: [round(100 * y) for y in v] for m, v in top1.items()})
    print("recall:", {m: [round(100 * y) for y in v] for m, v in recall.items()})


if __name__ == "__main__":
    sys.exit(main())
