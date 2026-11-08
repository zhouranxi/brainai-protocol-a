"""Paper 3 (causality) unified analysis: stats + figures from raw v2 data.

Single source of truth for every number in paper/causality/main.tex.
Run on locked data any time; E2 sections auto-extend as architectures land.

Outputs:
  results/paper3_stats.json      -- every number used in the paper
  results/paper3_report.md       -- human-readable summary
  paper/causality/figs/*.pdf     -- figures (matplotlib, no seaborn)

Usage: python code/paper3_analysis.py [--repo REPO_ROOT]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMS = ["gauss_iso", "gauss_aniso", "recolor", "tot_recolor"]
FAM_LABEL = {
    "gauss_iso": "iso noise",
    "gauss_aniso": "aniso noise (spectral-shape matched)",
    "recolor": "recolor (zero-noise whitening)",
    "tot_recolor": "total-spectrum recolor",
}


def family_slope(rows, model, fam):
    pts = [r for r in rows if r["model"] == model]
    base = next(r for r in pts if r["kind"] == "base")
    sel = [r for r in pts if r["kind"] == fam and r["glue_d"] is not None] + [base]
    sel = sorted(sel, key=lambda r: -(r["param"] if r["param"] is not None else 9))
    if len(sel) < 3:
        return None
    D = np.array([r["glue_d"] for r in sel], float)
    Y = np.array([r["c100_probe"] for r in sel], float)
    return float(np.polyfit(D, Y, 1)[0])


def base_anchor(rows):
    """Observational anchor over base rows."""
    bs = [r for r in rows if r["kind"] == "base"]
    D = np.array([r["glue_d"] for r in bs], float)
    Y = np.array([r["c100_probe"] for r in bs], float)
    return {
        "n": len(bs),
        "pearson_r": round(float(pearsonr(D, Y)[0]), 4),
        "spearman_rho": round(float(spearmanr(D, Y)[0]), 4),
        "slope": round(float(np.polyfit(D, Y, 1)[0]), 4),
        "d_range": [round(float(D.min()), 3), round(float(D.max()), 3)],
    }


def _load_rows(repo, *candidates):
    """Load first existing JSON with 'rows' from flat runs/ filenames."""
    for c in candidates:
        p = repo / "runs" / c
        if p.exists():
            try:
                return json.load(open(p))["rows"]
            except Exception:
                continue
    return None


def analyze_e1(repo):
    rows = _load_rows(repo, "v2_e1_full.json", "v2/e1/intervene_v2.json")
    models = sorted({r["model"] for r in rows})
    out = {"n_models": len(models), "n_rows": len(rows), "base": base_anchor(rows)}

    out["family_slopes"] = {}
    for f in FAMS:
        sl = [family_slope(rows, m, f) for m in models]
        sl = [s for s in sl if s is not None]
        out["family_slopes"][f] = {
            "median": round(float(np.median(sl)), 5),
            "neg": int(sum(s < 0 for s in sl)),
            "n": len(sl),
            "range": [round(min(sl), 4), round(max(sl), 4)],
            "per_model": [round(s, 5) for s in sl],
        }

    # Endpoint cells
    a2c, a2i, a2d, a2rd = [], [], [], []
    rc, rd, ic, idd = [], [], [], []
    for m in models:
        pts = [r for r in rows if r["model"] == m]
        b = next(r for r in pts if r["kind"] == "base")
        a = next(r for r in pts if r["kind"] == "gauss_aniso" and r["param"] == 2.0)
        t = next(r for r in pts if r["kind"] == "rotate")
        g = next(r for r in pts if r["kind"] == "gauss_iso" and r["param"] == 2.0)
        r0 = next(r for r in pts if r["kind"] == "recolor" and r["param"] == 0.0)
        a2c.append(a["c100_probe"] - b["c100_probe"])
        a2i.append(a["id_probe_acc"] - b["id_probe_acc"])
        a2d.append(a["glue_d"] - b["glue_d"])
        a2rd.append((a["glue_d"] - b["glue_d"]) / b["glue_d"])
        rc.append(t["c100_probe"] - b["c100_probe"])
        rd.append(t["glue_d"] - b["glue_d"])
        ic.append(g["c100_probe"] - b["c100_probe"])
        idd.append(g["glue_d"] - b["glue_d"])

    out["aniso2"] = {
        "dc100_median": round(float(np.median(a2c)), 4),
        "dc100_all_neg": bool(all(v < 0 for v in a2c)),
        "didp_median": round(float(np.median(a2i)), 4),
        "didp_all_neg": bool(all(v < 0 for v in a2i)),
        "dD_median": round(float(np.median(a2d)), 4),
        "relD_max": round(float(max(abs(v) for v in a2rd)), 4),
        "per_model_dc100": [round(v, 4) for v in a2c],
    }
    out["rotate"] = {
        "dc100_max_abs": round(float(max(abs(v) for v in rc)), 4),
        "dD_max_abs": round(float(max(abs(v) for v in rd)), 4),
    }
    out["iso2"] = {
        "dc100_median": round(float(np.median(ic)), 4),
        "dc100_all_neg": bool(all(v < 0 for v in ic)),
        "dD_median": round(float(np.median(idd)), 3),
    }
    # recolor0 endpoint
    r0c = []
    for m in models:
        pts = [r for r in rows if r["model"] == m]
        b = next(r for r in pts if r["kind"] == "base")
        r0 = next(r for r in pts if r["kind"] == "recolor" and r["param"] == 0.0)
        r0c.append(r0["c100_probe"] - b["c100_probe"])
    out["recolor0"] = {
        "dc100_median": round(float(np.median(r0c)), 4),
        "all_pos": bool(all(v > 0 for v in r0c)),
        "per_model": [round(v, 4) for v in r0c],
    }
    # geometric share: canonical = recolor family alone (matches paper3_ci.py,
    # the permutation floor, and the cross-architecture Table 2 definition);
    # pooled variant = median over models of per-model (rec+tot)/2 slopes,
    # matching the bootstrap/pooled definition in paper3_ci.py exactly.
    geo = out["family_slopes"]["recolor"]["median"]
    tot_sl = [s for s in (family_slope(rows, m, "tot_recolor") for m in models)
              if s is not None]
    rec_sl = [s for s in (family_slope(rows, m, "recolor") for m in models)
              if s is not None]
    geo_pooled = float(np.median([(a + c) / 2
                                  for a, c in zip(rec_sl, tot_sl)]))
    obs = out["base"]["slope"]
    out["geometric_share"] = {
        "geo_slope_median": round(float(geo), 5),
        "obs_slope": obs,
        "share_pct": round(float(geo) / obs * 100, 1),
        "pooled_geo_slope_median": round(geo_pooled, 5),
        "pooled_share_pct": round(geo_pooled / obs * 100, 1),
    }
    return out, rows, models


def analyze_e3(repo):
    rows = _load_rows(repo, "v2_e3.json", "v2/e3/intervene_v2.json")
    e3 = {r["model"]: r for r in rows if r["arm"] == "e3"}
    e3c = {r["model"]: r for r in rows if r["arm"] == "e3c"}
    models = sorted(e3)
    dc = [e3[m]["c100_probe"] - e3c[m]["c100_probe_ctrl"] for m in models]
    inf_rec = [e3[m]["id_fc_acc_inference_recolor"] - e3c[m]["id_fc_acc_orig"]
               for m in models]
    head_cross = [e3[m]["id_fc_acc_orig"] - e3c[m]["id_fc_acc_orig"]
                  for m in models]
    didp = [e3[m]["id_probe_acc"] - e3c[m]["id_probe_acc"] for m in models]

    out = {
        "n_models": len(models),
        "train_whiten_dc100": {
            "median": round(float(np.median(dc)), 4),
            "pos": int(sum(v > 0 for v in dc)),
            "wilcoxon_p": round(float(wilcoxon(dc).pvalue), 4),
            "per_model": [round(v, 4) for v in dc],
        },
        "inference_recolor_dfc": {
            "median": round(float(np.median(inf_rec)), 4),
            "neg": int(sum(v < 0 for v in inf_rec)),
            "wilcoxon_p": round(float(wilcoxon(inf_rec).pvalue), 4),
        },
        "head_original_space": {
            "median": round(float(np.median(head_cross)), 4),
            "neg": int(sum(v < 0 for v in head_cross)),
            "wilcoxon_p": round(float(wilcoxon(head_cross).pvalue), 4),
        },
        "id_probe_diff_median": round(float(np.median(didp)), 4),
    }
    # lr stratification
    strat = {}
    for lr in ("0.125", "0.5", "1.0"):
        ms = [m for m in models if f"lr{lr}_" in m]
        if ms:
            d2 = [e3[m]["c100_probe"] - e3c[m]["c100_probe_ctrl"] for m in ms]
            strat[lr] = {"n": len(ms),
                         "median": round(float(np.median(d2)), 4),
                         "per_model": [round(v, 4) for v in d2]}
    out["lr_strat"] = strat
    return out, e3, e3c


def analyze_e2(repo):
    """Incremental: analyze whatever E2 architectures have complete data."""
    out = {}
    for arch, tag in (("resnet34", "v2_e2_r34.json"),
                      ("resnet50", "v2_e2_r50.json"),
                      ("shufflenet_v2_x1_0", "v2_e2_sf.json"),
                      ("regnet_y_400mf", "v2_e2_reg.json")):
        rows = _load_rows(repo, tag, f"v2/{tag.replace('v2_', '')}/intervene_v2.json")
        if rows is None:
            continue
        models = sorted({r["model"] for r in rows})
        n_cond = min(sum(1 for r in rows if r["model"] == m) for m in models)
        if len(models) < 2 or n_cond < 15:
            out[arch] = {"status": "partial", "models": len(models),
                         "min_conditions": n_cond}
            continue
        b = base_anchor(rows)
        rec = [family_slope(rows, m, "recolor") for m in models]
        rec = [s for s in rec if s is not None]
        r0 = []
        a2c = []
        for m in models:
            pts = [r for r in rows if r["model"] == m]
            bb = next(r for r in pts if r["kind"] == "base")
            rr = next((r for r in pts if r["kind"] == "recolor" and r["param"] == 0.0), None)
            if rr:
                r0.append(rr["c100_probe"] - bb["c100_probe"])
            aa = next((r for r in pts if r["kind"] == "gauss_aniso" and r["param"] == 2.0), None)
            if aa:
                a2c.append(aa["c100_probe"] - bb["c100_probe"])
        out[arch] = {
            "status": "complete", "n_models": len(models), "n_cond": n_cond,
            "base": b,
            "recolor_slope_median": round(float(np.median(rec)), 5),
            "recolor_neg": int(sum(s < 0 for s in rec)),
            "recolor_share_pct": round(float(np.median(rec)) / b["slope"] * 100, 1)
            if b["slope"] else None,
            "recolor0_all_pos": bool(all(v > 0 for v in r0)),
            "recolor0_median": round(float(np.median(r0)), 4),
            "aniso2_dc100_median": round(float(np.median(a2c)), 4) if a2c else None,
            "aniso2_all_neg": bool(all(v < 0 for v in a2c)) if a2c else None,
        }
    return out


def make_figures(repo, e1, rows, models, e3, e3d, e3c, e2):
    figs = repo / "paper/causality/figs"
    figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "figure.dpi": 150})

    # ---- Fig 1: the 3x2 dissociation (endpoint bars) ----
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.6))
    cells = [
        ("Noise, $D_{eff}$ pinned", e1["aniso2"]["per_model_dc100"],
         "aniso=2.0 (D pinned)"),
        ("Noise, $D_{eff}$ up", None, None),  # iso2 below
        ("Zero noise, $D_{eff}$ up 1-8x", e1["recolor0"]["per_model"],
         "recolor(0)"),
    ]
    # panel 0: aniso2
    ax = axes[0]
    ax.bar(range(8), e1["aniso2"]["per_model_dc100"],
           color="#c0392b", alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("aniso noise=2.0\n($D_{eff}$ pinned, noise in)")
    ax.set_ylabel("$\\Delta$ CIFAR-100 acc")
    # panel 1: iso2 (D up via noise)
    pts_iso = []
    for m in models:
        pts = [r for r in rows if r["model"] == m]
        b = next(r for r in pts if r["kind"] == "base")
        g = next(r for r in pts if r["kind"] == "gauss_iso" and r["param"] == 2.0)
        pts_iso.append(g["c100_probe"] - b["c100_probe"])
    ax = axes[1]
    ax.bar(range(8), pts_iso, color="#e67e22", alpha=0.9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("iso noise=2.0\n($D_{eff}$ $\\times$6 median, noise in)")
    # panel 2: recolor0
    ax = axes[2]
    ax.bar(range(8), e1["recolor0"]["per_model"], color="#27ae60", alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("whitening recolor(0)\n($D_{eff}$ $\\times$1--8, zero noise)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_ylim(min(-0.14, ax.get_ylim()[0]), 0.06)
    fig.suptitle("Transfer change under matched-$D$ vs moved-$D$ interventions "
                 "(8 models)", y=1.04)
    fig.tight_layout()
    fig.savefig(figs / "fig1_dissociation.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: E1 slopes per family ----
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    fams = [f for f in FAMS]
    positions = np.arange(len(fams))
    for i, f in enumerate(fams):
        sl = e1["family_slopes"][f]["per_model"]
        ax.scatter(np.full(len(sl), i) + np.random.default_rng(0).uniform(
            -0.08, 0.08, len(sl)), sl, s=22, color="#2c3e50", alpha=0.75,
            zorder=3)
        med = np.median(sl)
        ax.hlines(med, i - 0.22, i + 0.22, color="#c0392b", lw=2.2, zorder=4)
    ax.axhline(e1["base"]["slope"], color="#2980b9", ls="--", lw=1.4,
               label=f"observational slope ({e1['base']['slope']:+.3f})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(["iso", "aniso", "recolor", "tot-rc"], fontsize=8)
    ax.set_ylabel("$d\\,c_{100}/d\\,D_{eff}$")
    ax.set_title("Causal vs observational slopes, 8 models")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(figs / "fig2_slopes.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: E3 phase separation ----
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    dc = e3["train_whiten_dc100"]["per_model"]
    axes[0].bar(range(len(dc)), dc, color="#27ae60", alpha=0.85)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("Formation phase: whitened training\n"
                      f"$\\Delta c_{{100}}$ median "
                      f"{e3['train_whiten_dc100']['median']:+.4f} "
                      f"(p={e3['train_whiten_dc100']['wilcoxon_p']})")
    axes[0].set_ylabel("$\\Delta$ CIFAR-100 acc")
    # inference panel: recolor-at-inference on original head
    inf = [e3d[m]["id_fc_acc_inference_recolor"] - e3c[m]["id_fc_acc_orig"]
           for m in sorted(e3d)]
    axes[1].bar(range(len(inf)), inf, color="#c0392b", alpha=0.85)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("Inference phase: recolor on trained head\n"
                      f"$\\Delta$ head acc median {np.median(inf):+.4f}")
    for ax in axes:
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(figs / "fig3_phases.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 4: cross-architecture (auto-extends) ----
    archs = [(a, d) for a, d in e2.items() if d.get("status") == "complete"]
    if not archs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    # prepend the E1 ResNet-18 anchor as the reference family
    names, obs_s, geo_s = ["r18"], [e1["base"]["slope"]], \
        e1["geometric_share"]["geo_slope_median"]
    obs_s = [obs_s[0]]
    geo_s = [geo_s]
    for a, d in archs:
        names.append(a.replace("_y_400mf", "").replace("_v2_x1_0", "")
                     .replace("shufflenet", "sfn"))
        obs_s.append(d["base"]["slope"])
        geo_s.append(d["recolor_slope_median"])
    x = np.arange(len(names))
    axes[0].bar(x - 0.18, obs_s, width=0.36, label="observational", color="#2980b9")
    axes[0].bar(x + 0.18, geo_s, width=0.36, label="causal (recolor)", color="#c0392b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_title("Slopes by architecture")
    axes[0].legend(fontsize=7.5)
    # observational r per arch (r18 reference + E2 families)
    rs = [e1["base"]["pearson_r"]] + [d["base"]["pearson_r"] for _, d in archs]
    axes[1].bar(x, rs, color="#8e44ad", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=8)
    axes[1].set_ylim(0.80, 1.02)
    axes[1].axhline(1.0, color="k", lw=0.5, ls=":")
    axes[1].set_title("Observational $r(D_{eff}, c_{100})$")
    for ax, vals in ((axes[1], rs),):
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.004, f"{v:+.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "fig4_arch.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(
        Path(__file__).resolve().parent.parent))
    repo = Path(ap.repo) if False else Path(ap.parse_args().repo)
    e1, rows, models = analyze_e1(repo)
    e3, e3d, e3c = analyze_e3(repo)
    e2 = analyze_e2(repo)
    stats = {"e1": e1, "e3": e3, "e2": e2}
    (repo / "results").mkdir(exist_ok=True)
    (repo / "results/paper3_stats.json").write_text(
        json.dumps(stats, indent=1), encoding="utf-8")

    # markdown report
    lines = ["# Paper 3 stats (auto-generated)\n"]
    lines.append(f"## E1: {e1['n_models']} models, {e1['n_rows']} rows\n")
    lines.append(f"base: {e1['base']}\n")
    for f, d in e1["family_slopes"].items():
        lines.append(f"- {f}: median {d['median']:+.5f}, neg {d['neg']}/{d['n']}, "
                     f"range {d['range']}")
    lines.append(f"\naniso2: {e1['aniso2']}")
    lines.append(f"rotate: {e1['rotate']}")
    lines.append(f"iso2: {e1['iso2']}")
    lines.append(f"recolor0: {e1['recolor0']}")
    lines.append(f"geometric share: {e1['geometric_share']}\n")
    lines.append(f"## E3: {e3['n_models']} paired models\n{json.dumps(e3, indent=1)}\n")
    lines.append(f"## E2:\n{json.dumps(e2, indent=1)}\n")
    (repo / "results/paper3_report.md").write_text("\n".join(lines), encoding="utf-8")

    make_figures(repo, e1, rows, models, e3, e3d, e3c, e2)
    print(json.dumps(stats["e1"]["geometric_share"], indent=1))
    print("E2:", {a: d.get("status") for a, d in e2.items()})
    print("figs + stats written")


if __name__ == "__main__":
    main()
