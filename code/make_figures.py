"""Generate paper figures 2-6 from final results.

Outputs: figures/fig{2..6}.{png,pdf} (300 dpi, colorblind-safe).

Data sources (all final, 2026-10-27):
  - ind_validation_results.json : 48 self-built resnet18 (c100 + tinyimg probes, 19 corruptions)
  - mod_arch_results.json       : 32 new-arch models (shufflenet/regnet)
  - r1_final_results.json       : T4 band robustness (official 224 models)
  - protocol_a_fig3_results.json: F6 three-way consistency (official 80 vs 224)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_AP = argparse.ArgumentParser(description="Generate paper figures 2-6.")
_AP.add_argument("--outdir", default=None, help="figure output directory")
_AP.add_argument("--data", default=None, help="results directory holding the JSON inputs")
_ARGS = _AP.parse_args()
def _find_root():
    here = Path(__file__).resolve().parent
    for cand in [here, here.parent, here.parent.parent]:
        if (cand / "results").is_dir() and (cand / "code").is_dir():
            return cand
    return here.parent if (here.parent / "results").is_dir() else here


ROOT = _find_root()
BASE = ROOT
RES = Path(_ARGS.data) if _ARGS.data else BASE / "results"
OUT = Path(_ARGS.outdir) if _ARGS.outdir else BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_BASE = "#D55E00"   # baseline (ID acc) - vermillion
C_GEO1 = "#0072B2"   # PR - blue
C_GEO2 = "#009E73"   # smoothness - green
C_GEO3 = "#CC79A7"   # numerical rank - pink
C_GRAY = "#7F7F7F"

MARKERS = [
    ("id_pr_perclass", "PR (per-class)", C_GEO1),
    ("id_local_smoothness", "Local smoothness", C_GEO2),
    ("id_numerical_rank", "Numerical rank", C_GEO3),
    ("id_acc", "ID accuracy (baseline)", C_BASE),
]


def load(name):
    return json.load(open(RES / name))


def group_residual(y, groups):
    """Subtract group means (family residualization, same as analysis)."""
    y = np.asarray(y, float)
    out = y.copy()
    for g in set(groups):
        m = np.array([gi == g for gi in groups])
        out[m] -= y[m].mean()
    return out


IND = load("ind_validation_results.json")["rows"]
MOD = load("mod_arch_results.json")["rows"]
ALL = [r for r in IND + MOD if r["id_acc"] > 0.3]  # drop divergent
print(f"models: ind={len(IND)} mod={len(MOD)} healthy={len(ALL)}")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- Fig 2
# Scatter: OOD acc vs predictor, raw (top) / lr-family residualized (bottom).
# Data: 79 healthy three-arch models, OOD = CIFAR-100 probe.
ood = np.array([r["c100_probe_acc"] for r in ALL]) * 100
acc = np.array([r["id_acc"] for r in ALL]) * 100
lr = [r["lr"] for r in ALL]
arch = [r["arch"] for r in ALL]
arch_c = {"resnet18": "#BBBBBB", "shufflenet_v2_x1_0": "#56B4E9", "regnet_y_400mf": "#E69F00"}
arch_n = {"resnet18": "ResNet-18", "shufflenet_v2_x1_0": "ShuffleNetV2", "regnet_y_400mf": "RegNetY"}

fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharey="row")
ood_res = group_residual(ood, lr)
for j, (key, label, c) in enumerate(MARKERS):
    x = np.array([r[key] for r in ALL]) * (100 if key == "id_acc" else 1)
    x_res = group_residual(x, lr)
    for row, (xx, xl) in enumerate([(x, "raw"), (x_res, "lr-residualized")]):
        ax = axes[row, j]
        for a in arch_c:
            m = np.array([ai == a for ai in arch])
            ax.scatter(xx[m], (ood if row == 0 else ood_res)[m],
                       s=14, c=arch_c[a], label=arch_n[a], alpha=0.85, lw=0)
        r = np.corrcoef(xx, ood if row == 0 else ood_res)[0, 1]
        ax.set_title(f"{label}\n$r={r:+.2f}$", fontsize=8.5)
        if row == 1:
            ax.set_xlabel(label)
        if j == 0:
            ax.set_ylabel(f"OOD acc ({'raw' if row == 0 else 'resid.'}, %)")
axes[0, 0].legend(loc="lower right", frameon=False)
fig.subplots_adjust(hspace=0.55, top=0.80)
fig.suptitle("Figure 2. Geometric markers vs CIFAR-100 transfer, raw (top) and lr-family residualized (bottom).\n"
             "79 healthy models across three architectures; baseline = ID accuracy.", fontsize=9, y=0.99)
save(fig, "fig2_scatter")

# ---------------------------------------------------------------- Fig 3
# Simpson's paradox on OFFICIAL 80-model data (gaussian_noise): pooled acc_test
# vs OOD probe is positive; within every architecture it flips negative.
import pandas as pd
def _locate_pkl(*names):
    """Locate a released pickle under any of the layouts we ship.

    The file is called df_fig15-20.pkl in the packaged data/ layout and
    official_df_fig15-20.pkl in the flat working-directory layout, so both
    spellings are accepted.
    """
    for name in names:
        for cand in (RES / name, ROOT / "data" / "official" / name,
                     ROOT / "results" / name, ROOT / name):
            if cand.exists():
                return cand
    raise FileNotFoundError(" / ".join(names))


odf = pd.read_pickle(_locate_pkl("df_fig15-20.pkl", "official_df_fig15-20.pkl"))["gaussian_noise"]
ox = odf["acc_test"].to_numpy()
oy = odf["acc_lp_ood"].to_numpy()
ofam = odf["arch_name"].to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
fam_c = plt.cm.viridis(np.linspace(0.05, 0.85, len(set(ofam))))
# left: pooled
ax = axes[0]
ax.scatter(ox, oy, s=16, c="#666666", alpha=0.8, lw=0)
b = np.polyfit(ox, oy, 1)
xs = np.linspace(ox.min(), ox.max(), 10)
ax.plot(xs, np.polyval(b, xs), c=C_BASE, lw=2)
ax.set_title(f"Pooled: $r={np.corrcoef(ox, oy)[0,1]:+.2f}$")
ax.set_xlabel("ID accuracy, acc_test (%)")
ax.set_ylabel("OOD probe accuracy (%)")
# right: per architecture
ax = axes[1]
for a, cc in zip(sorted(set(ofam)), fam_c):
    m = ofam == a
    ax.scatter(ox[m], oy[m], s=16, color=cc, lw=0, label=a)
    bb = np.polyfit(ox[m], oy[m], 1)
    xs = np.linspace(ox[m].min(), ox[m].max(), 5)
    ax.plot(xs, np.polyval(bb, xs), color=cc, lw=1.4)
    r = np.corrcoef(ox[m], oy[m])[0, 1]
    ax.annotate(f"$r={r:+.2f}$", (ox[m].mean(), oy[m].mean()),
                fontsize=7, color=cc, ha="center", va="bottom", xytext=(0, 4),
                textcoords="offset points")
ax.set_title("Within architecture: sign flips")
ax.set_xlabel("ID accuracy, acc_test (%)")
ax.legend(frameon=False, fontsize=6.5, loc="lower right")
fig.suptitle("Figure 3. Simpson's paradox (official 80 models, gaussian noise): ID accuracy predicts OOD transfer pooled,\n"
             "but negatively within every architecture.", fontsize=9, y=1.05)
save(fig, "fig3_simpson")

# ---------------------------------------------------------------- Fig 4
# Band robustness (official 224 models, T4): resid r vs band.
T4 = load("r1_final_results.json")["T4_robustness"]
bands = [1, 1.5, 2, 2.5, 3]  # by_band order per analysis (b=1..3 step .5)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
t4_keys = [
    ("utility_id", "Utility ($\\Psi$)", C_GEO2),
    ("dimension_id", "Effective dim ($D_{eff}$)", "#56B4E9"),
    ("radius_id", "Radius ($R_{eff}$)", C_GEO3),
    ("center_alignment_id", "Center alignment", C_GRAY),
]
for key, lab, c in t4_keys:
    ys = [float(s) for s in T4[key]["by_band"]]
    ax.plot(bands[: len(ys)], ys, "o-", c=c, label=lab, ms=4)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("ID-matching band $b$ (±pp)")
ax.set_ylabel("Residualized $r$ with OOD acc")
ax.set_title("Figure 4. Band robustness (official 224-model data): effect stable across matching bands")
ax.legend(frameon=False)
save(fig, "fig4_band")

# ---------------------------------------------------------------- Fig 5
# Heatmap: |rho| between markers and 19 corruption types (48 self-built models).
corrs = sorted(k[5:-4] for k in IND[0] if k.startswith("corr_") and k.endswith("_acc"))
mk_keys = ["id_pr_perclass", "id_local_smoothness", "id_numerical_rank",
           "id_class_separability", "id_acc"]
mk_labs = ["PR (per-class)", "Local smooth.", "Numerical rank", "Class sep.", "ID acc (base)"]
M = np.zeros((len(mk_keys), len(corrs)))
for i, mk in enumerate(mk_keys):
    x = np.array([r[mk] for r in IND])
    for j, c in enumerate(corrs):
        y = np.array([r[f"corr_{c}_acc"] for r in IND])
        M[i, j] = np.corrcoef(x, y)[0, 1]
order = np.argsort(-np.abs(M[:-1]).mean(axis=0))  # sort corruptions by mean |rho| of geo markers
corrs_s = [corrs[j].replace("_", " ") for j in order]
M = M[:, order]
fig, ax = plt.subplots(figsize=(9.2, 2.9))
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corrs_s)), corrs_s, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(mk_labs)), mk_labs, fontsize=8)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i,j]:+.1f}"[1::][:2] if False else f"{M[i,j]:.2f}".replace("0.", "."),
                ha="center", va="center", fontsize=5.6,
                color="white" if abs(M[i, j]) > 0.55 else "black")
fig.colorbar(im, ax=ax, label=r"$\rho$ (Spearman)", shrink=0.9)
ax.set_title("Figure 5. Marker-vs-corruption association (48 self-built models); baseline on bottom row")
save(fig, "fig5_corruption")

# ---------------------------------------------------------------- Fig 6
# Three-way consistency: r for each marker on official-80, official-224, self-built-48.
F6 = load("protocol_a_fig3_results.json")["F6"]
f6_keys = [
    ("participation_ratio_id", "PR", "id_pr_perclass"),
    ("utility_id", "Utility $\\Psi$", None),
    ("dimension_id", "Eff. dim $D_{eff}$", None),
    ("radius_id", "Radius $R_{eff}$", None),
    ("center_alignment_id", "Center align.", None),
    ("sparsity_id", "Sparsity", None),
]
# self-built 48 (c100), lr-family residualized for comparability;
# own cohort implements only PR / numerical rank / NC1 from this list (rest = no bar).
c100 = np.array([r["c100_probe_acc"] for r in IND]) * 100
lri = [r["lr"] for r in IND]
own = {}
for key, _, mk in f6_keys:
    if mk and mk in IND[0]:
        x = np.array([r[mk] for r in IND])
        own[key] = np.corrcoef(group_residual(x, lri), group_residual(c100, lri))[0, 1]
    else:
        own[key] = np.nan
x = np.arange(len(f6_keys))
w = 0.27
fig, ax = plt.subplots(figsize=(7.6, 3.4))
ax.bar(x - w, [F6[k]["r_80"] for k, _, _ in f6_keys], w, label="official 80 (ID-matched)", color="#0072B2")
ax.bar(x, [F6[k]["r_224"] for k, _, _ in f6_keys], w, label="official 224", color="#56B4E9")
ax.bar(x + w, [own.get(k, np.nan) for k, _, _ in f6_keys], w, label="self-built 48 (resid.)", color="#E69F00")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x, [lab for _, lab, _ in f6_keys], fontsize=8)
ax.set_ylabel("$r$ with OOD accuracy")
ax.set_title("Figure 6. Three-way consistency: same protocol on official-80 / official-224 / self-built cohorts")
ax.legend(frameon=False, fontsize=7.5)
save(fig, "fig6_threeway")

print("all figures done ->", OUT)
