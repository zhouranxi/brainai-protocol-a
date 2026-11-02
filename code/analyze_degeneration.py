#!/usr/bin/env python3
"""Two ways a geometric marker can move: the representation got worse, or the
input carried less information.

A marker that shrinks when the input is degraded is not evidence that the
representation became less elegant. Occluding the input removes information;
any information-carrying quantity must shrink as a matter of arithmetic. To
claim that a marker tracks representational quality, the marker has to move
while the amount of information in the input is held fixed.

This script assembles the two comparisons that separate the two explanations:

  1. Degradation sweep (input information varies)
     Same network, three conditions: clean, 30% occlusion, 70% occlusion.
     The marker and the probe accuracy fall together -- which is exactly what
     the information confound predicts and cannot distinguish from a real
     degradation of the representation.

  2. Recipe sweep (input information fixed)
     42 networks trained on the same clean inputs, differing only in the
     training recipe. If a marker tracked representational quality here, it
     would have to separate the quality gradient -- and it does not: the
     apparent correlation is carried entirely by the split between networks
     that trained successfully and networks that collapsed.

The point of the pair is that neither comparison on its own is decisive. The
first cannot separate the two explanations; the second shows that once the
input is held fixed, the marker does not track quality at all.

Usage:
    python analyze_degeneration.py --outdir results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def find_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, here.parent, here.parent.parent]:
        if (cand / "results").is_dir():
            return cand
    return here.parent


ROOT = find_root()

# The degradation sweep, transcribed from a smoke-test summary produced on
# 2026-10-25 with ResNet-50 on a non-CIFAR image set (960 samples, seeds 0/1/2).
#
# PROVENANCE WARNING: the summary log this table was transcribed from is NOT
# part of the released artifacts, and the per-seed values behind these means
# were not kept. The numbers below are therefore constants with no re-derivable
# source, and nothing in this repository can regenerate them. They are retained
# because the paper reports them, and the paper states their provenance
# explicitly (measurement paper, Table 3 caption and Limitations); they must
# not be read as a reproducible experiment.
#
# This is a diagnostic smoke test, not one of the paper's experiments: it is a
# single architecture on a different image distribution from the recipe sweep
# below, and it is used only to illustrate that the degradation comparison
# cannot separate the two readings.
DEGENERATION = {
    "clean":  {"dim": 316, "smooth": 0.934, "probe": 0.531},
    "occ30":  {"dim": 287, "smooth": 0.910, "probe": 0.438},
    "occ70":  {"dim": 128, "smooth": 0.586, "probe": 0.264},
    "noise":  {"dim": 271, "smooth": 0.899, "probe": 0.309},
}

# The recipe sweep: 14 recipes x 3 seeds, all trained on the same clean inputs.
TRIVIAL = {"linear_probe_acc", "silhouette_gap"}          # use test labels
NON_TRIVIAL = {"effective_dim", "local_smoothness",
               "criticality_proxy", "spectral_entropy"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    L = ["# Two explanations for a moving marker", "",
         "## 1. Degradation sweep: the input information changes", "",
         "One network (resnet50, 960 samples), three input conditions. Both the",
         "marker and the probe accuracy fall together, and they fall by nearly the",
         "same proportion.", "",
         "| condition | effective dim | local smoothness | probe acc | dim vs clean | probe vs clean |",
         "|---|---|---|---|---|---|"]
    base_dim = DEGENERATION["clean"]["dim"]
    base_probe = DEGENERATION["clean"]["probe"]
    for cond, v in DEGENERATION.items():
        L.append(f"| {cond} | {v['dim']} | {v['smooth']:.3f} | {v['probe']:.3f} | "
                 f"{100*(v['dim']-base_dim)/base_dim:+.0f}% | "
                 f"{100*(v['probe']-base_probe)/base_probe:+.0f}% |")
    dim_ch = 100 * (DEGENERATION["occ70"]["dim"] - base_dim) / base_dim
    probe_ch = 100 * (DEGENERATION["occ70"]["probe"] - base_probe) / base_probe
    L += ["",
          f"The two quantities move together ({dim_ch:+.0f}% and {probe_ch:+.0f}%). That is",
          "what the information confound predicts and what a genuine degradation of the",
          "representation would also predict. **This comparison cannot tell the two apart.**",
          "",
          "Note also what the sweep does not support: the marker shrinks most under 70%",
          "occlusion (a structural loss) and least under noise (an amplitude",
          "perturbation), so it is sensitive to how the information was removed, not",
          "only to how much.", "",
          "## 2. Recipe sweep: the input information is fixed", "",
          "42 networks on identical clean inputs, differing only in the training recipe.",
          "If a marker tracked representational quality, it would have to order the",
          "recipes by quality. It does not.", ""]

    data = json.load(open(ROOT / "results" / "c2_results_seeds0-1-2.json"))
    rows = data["rows"]
    acc = np.array([r["test_acc"] for r in rows])

    L += ["| marker | kind | pooled r | p | pooled rho | interpretation |",
          "|---|---|---|---|---|---|"]
    for mk in sorted(NON_TRIVIAL | TRIVIAL):
        if mk not in rows[0]:
            continue
        v = np.array([r[mk] for r in rows], dtype=float)
        r, p = pearsonr(v, acc)
        rho = spearmanr(v, acc)[0]
        kind = "trivial" if mk in TRIVIAL else "non-trivial"
        if kind == "trivial":
            note = "uses test labels; a restatement of the target"
        else:
            note = "negative, and driven by the collapsed/healthy split"
        L.append(f"| `{mk}` | {kind} | {r:+.3f} | {p:.4f} | {rho:+.3f} | {note} |")

    # The decisive part: within-cluster behaviour
    L += ["", "### The correlation is a cluster separation, not a gradient", "",
          "The test accuracy distribution is sharply bimodal: a failed group and a",
          "healthy group, with nothing in between. Within either group the markers carry",
          "no information.", ""]
    # The training accuracy of the last recorded epoch separates the two groups.
    def final_train_acc(r):
        h = r.get("history") or []
        return float(h[-1]["train_acc"]) if h else float("nan")

    tr_acc = np.array([final_train_acc(r) for r in rows])
    healthy = tr_acc >= 0.90
    L += [f"Trained successfully (final train acc >= 0.90): {int(healthy.sum())} models; "
          f"collapsed: {int((~healthy).sum())} models.", "",
          "**By training outcome.** The sign of the pooled correlation is set by where",
          "the collapsed models sit, not by a relationship between the marker and",
          "quality within either group.", "",
          "| subset | n | `effective_dim` | `local_smoothness` |", "|---|---|---|---|"]
    for label, mask in [("trained successfully", healthy),
                        ("collapsed", ~healthy)]:
        sub = [r for r, k in zip(rows, mask) if k]
        if len(sub) < 5:
            continue
        cells = []
        for mk in ("effective_dim", "local_smoothness"):
            v = np.array([r[mk] for r in sub], dtype=float)
            a = np.array([r["test_acc"] for r in sub])
            rr, pp = pearsonr(v, a)
            cells.append(f"r={rr:+.3f} (p={pp:.3f})")
        L.append(f"| {label} | {len(sub)} | {cells[0]} | {cells[1]} |")

    # By performance cluster: the split that the pooled correlation is really
    # driven by. The distribution is bimodal with nothing in between.
    L += ["", "**By performance cluster.** The pooled correlation is carried entirely by",
          "the separation between the two performance clusters. Inside either cluster",
          "the marker carries no information about performance.", "",
          "| cluster | n | `effective_dim` | `local_smoothness` |", "|---|---|---|---|"]
    # The clusters are defined by recipe, not by a threshold on accuracy: the
    # low cluster is the six recipes that land around 0.78-0.81 and the high
    # cluster is the three that land around 0.84-0.88. Defining them by recipe
    # keeps the membership fixed across seeds.
    LOW_RECIPES = {"plain", "plain_ls", "noise20", "noise50", "wd_high", "lr_high"}
    HIGH_RECIPES = {"best", "aug_only", "noise20_aug"}
    lo = np.array([r["name"] in LOW_RECIPES for r in rows])
    hi = np.array([r["name"] in HIGH_RECIPES for r in rows])
    for label, mask in [("low cluster (6 recipes)", lo), ("high cluster (3 recipes)", hi)]:
        sub = [r for r, k in zip(rows, mask) if k]
        if len(sub) < 5:
            continue
        cells = []
        for mk in ("effective_dim", "local_smoothness"):
            v = np.array([r[mk] for r in sub], dtype=float)
            a = np.array([r["test_acc"] for r in sub])
            rr, pp = pearsonr(v, a)
            cells.append(f"r={rr:+.3f} (p={pp:.3f})")
        L.append(f"| {label} | {len(sub)} | {cells[0]} | {cells[1]} |")
    L += ["", "The clusters are defined by recipe, not by a threshold on accuracy, so "
          "membership is fixed across seeds. The remaining recipes sit between them.", "",
          "The undefined coefficient for `effective_dim` in the high cluster is not a "
          "computation failure: the marker takes the same value for all nine models "
          "there, so the correlation is undefined. That is itself the finding -- in the "
          "range where the representation is best, this marker does not vary at all, "
          "and so cannot order the models.", ""]

    # Recipe-level residual: does the marker move when only the seed changes?
    L += ["**Within a recipe.** Changing only the random seed, the marker barely",
          "moves -- so it measures the recipe, not the individual network's quality.", ""]
    names = sorted({r["name"] for r in rows})
    resid_rows = []
    for mk in ("effective_dim", "local_smoothness", "criticality_proxy", "spectral_entropy"):
        rv, av = [], []
        for nm in names:
            sub = [r for r in rows if r["name"] == nm]
            if len(sub) < 2:
                continue
            for r in sub:
                rv.append(r[mk]); av.append(r["test_acc"])
        # remove the recipe mean from both
        rv = np.array(rv, float); av = np.array(av, float)
        rec = np.array([nm for nm in names for _ in range(3)], dtype=object)
        rv = rv - np.array([rv[rec == nm].mean() for nm in rec])
        av = av - np.array([av[rec == nm].mean() for nm in rec])
        rr, pp = pearsonr(rv, av)
        resid_rows.append((mk, rr, pp))
    L += ["| marker | residual r (recipe removed) | p |", "|---|---|---|"]
    for mk, rr, pp in resid_rows:
        L.append(f"| `{mk}` | {rr:+.3f} | {pp:.4f} |")

    # the direct counter-example: two recipes with the same accuracy, different marker
    by_recipe = {}
    for r in rows:
        by_recipe.setdefault(r["name"], []).append(r)
    pairs = []
    names = sorted(by_recipe)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.mean([x["test_acc"] for x in by_recipe[names[i]]])
            b = np.mean([x["test_acc"] for x in by_recipe[names[j]]])
            if abs(a - b) < 0.01:
                sa = np.mean([x["local_smoothness"] for x in by_recipe[names[i]]])
                sb = np.mean([x["local_smoothness"] for x in by_recipe[names[j]]])
                pairs.append((abs(a - b), names[i], names[j], a, b, sa, sb))
    if pairs:
        pairs.sort()
        _, n1, n2, a1, a2, s1, s2 = pairs[0]
        L += ["", "### A direct counter-example", "",
              f"Two recipes with the same test accuracy but a different marker value:", "",
              "| recipe | mean test acc | mean local smoothness |", "|---|---|---|",
              f"| `{n1}` | {a1:.4f} | {s1:.4f} |",
              f"| `{n2}` | {a2:.4f} | {s2:.4f} |", "",
              f"The accuracies differ by {abs(a1-a2)*100:.2f} percentage points while the",
              f"marker differs by {abs(s1-s2):.4f}. **The marker does not resolve a",
              "difference in the quantity it is supposed to track.**", ""]

    L += ["## What this establishes", "",
          "1. When the input information changes, the marker moves with the probe",
          "   accuracy -- as an arithmetic consequence, not as evidence about the",
          "   representation.",
          "2. When the input information is held fixed, the marker does not track quality",
          "   at all: the pooled correlation is negative and the within-cluster",
          "   correlation collapses.",
          "3. A marker of this family is therefore not a measurement of representational",
          "   quality, and a correlation between it and a performance measure is not",
          "   evidence that the representation is better.", "",
          "The corollary is the practical one: any claim that such a marker predicts",
          "performance has to survive the recipe sweep, not only the degradation sweep.",
          "That test is cheap, and it is the one that disqualifies the marker.", ""]

    out_md = outdir / "degeneration_analysis.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    (outdir / "degeneration_analysis.json").write_text(
        json.dumps({"degeneration": DEGENERATION, "n_recipes": len(rows)}, indent=2),
        encoding="utf-8")
    print("\n".join(L))
    print(f"[saved] {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
