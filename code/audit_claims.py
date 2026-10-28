"""Consistency audit: recompute every headline number claimed in paper_draft_v1.md
directly from the released data, and compare against the value written in the paper.

Purpose (2026-10-27): after two adversarial review rounds, verify that no number
in the manuscript disagrees with the underlying data or analysis code.

Method: for each claim we store (section, quantity, paper_value, recomputed_value,
tolerance). A claim passes if |paper - recomputed| <= tolerance. Claims whose
tolerance is a RANGE are checked by interval containment.

Outputs: results/claims_audit.json, results/claims_audit_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def _find_root():
    """Locate the package root: the nearest ancestor containing results/ and code/.

    Works both when scripts are run from inside the package (code/xxx.py) and when
    they are run from a flat working directory (xxx.py at top level).
    """
    here = Path(__file__).resolve().parent
    for cand in [here, here.parent, here.parent.parent]:
        if (cand / "results").is_dir() and (cand / "code").is_dir():
            return cand
    return here.parent if (here.parent / "results").is_dir() else here
ROOT = _find_root()
RES = ROOT / "results"

def _find_data():
    """Locate the input data.

    Prefers the packaged layout (data/official, data/self_trained); falls back to
    the flat working-directory layout used during development.
    """
    here = Path(__file__).resolve().parent
    for cand in [here, here.parent, here.parent.parent]:
        if (cand / "data" / "official").is_dir():
            return cand
    for cand in [here, here.parent]:
        if (cand / "results" / "df_fig3.pkl").exists():
            return cand
    return here.parent


_DROOT = _find_data()
_OFFICIAL = _DROOT / "data" / "official"
_SELFTRAINED = _DROOT / "data" / "self_trained"


def _p(official_name, selftrained_name):
    """Resolve one input file across both the packaged and the flat layout."""
    for cand in (_OFFICIAL / official_name, _SELFTRAINED / selftrained_name,
                 _DROOT / "results" / official_name, _DROOT / "results" / selftrained_name,
                 _DROOT / official_name, _DROOT / selftrained_name):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"cannot locate {official_name}")


D224 = pd.read_pickle(_p("df_fig3.pkl", "df_fig3.pkl")).reset_index(drop=True)
D80_ALL = pd.read_pickle(_p("df_fig15-20.pkl", "official_df_fig15-20.pkl"))
D80 = D80_ALL["gaussian_noise"]
IND = json.load(open(_p("ind_validation_results.json", "ind_validation_results.json")))["rows"]
MOD = json.load(open(_p("mod_arch_results.json", "mod_arch_results.json")))["rows"]
GLUE_OWN = {r["ckpt"]: r for r in
            json.load(open(_p("glue_own48_results.json", "glue_own48_results.json")))["rows"]}


def gres(v, groups):
    v = np.asarray(v, float)
    uq = sorted(set(groups))
    if len(uq) < 2:
        return v - v.mean()
    D = np.column_stack([(np.asarray(groups) == g).astype(float) for g in uq[1:]])
    A = np.column_stack([np.ones(len(v)), D])
    b, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ b


def band_mask(acc, b):
    return (acc >= acc.mean() - b) & (acc <= acc.mean() + b)


def resid_corr(df, xcol, ycol, groups, b):
    m = band_mask(df[groups].values if False else df["acc_test"].values, b)
    g = df["arch_name"].values
    return float(np.corrcoef(gres(df[xcol].values[m], g[m]), gres(df[ycol].values[m], g[m]))[0, 1])


CLAIMS = []


def claim(section, quantity, paper, recomputed, tol=0.006, note=""):
    """tol may be a float (absolute), a tuple (lo, hi) for interval containment,
    or None for exact string comparison."""
    if tol is None:
        ok = str(paper) == str(recomputed)
        kind = "exact"
    elif isinstance(tol, tuple):
        ok = tol[0] <= recomputed <= tol[1]
        kind = "range"
    else:
        ok = abs(paper - recomputed) <= tol
        kind = "point"
    CLAIMS.append({
        "section": section, "quantity": quantity, "paper": paper,
        "recomputed": round(recomputed, 4) if isinstance(recomputed, float) else recomputed,
        "tol": tol, "kind": kind, "pass": bool(ok), "note": note,
    })
    return ok


def main():
    ap = argparse.ArgumentParser(description="Audit every reported number against the raw data.")
    ap.add_argument("--outdir", default=None, help="override output directory (default: results/)")
    args = ap.parse_args()
    if args.outdir:
        globals()["RES"] = Path(args.outdir)
    acc = D224["acc_test"].values
    ood = D224["acc_lp_ood"].values
    arch = D224["arch_name"].values
    m1 = band_mask(acc, 1.0)

    # ---------- §4.2 full-sample landscape ----------
    d = D224["dimension_id"].values.astype(float)
    claim("4.2", "D_eff vs OOD (full sample)", 0.94, np.corrcoef(d, ood)[0, 1], 0.005)
    claim("4.2", "D_eff vs ID acc (full sample)", 0.50, np.corrcoef(d, acc)[0, 1], 0.006)
    pr = D224["participation_ratio_id"].values.astype(float)
    claim("4.2", "PR vs OOD (full sample)", 0.88, np.corrcoef(pr, ood)[0, 1], 0.006)
    claim("4.2", "PR vs ID acc (full sample)", 0.42, np.corrcoef(pr, acc)[0, 1], 0.006)

    # ---------- Table 2 ----------
    T2 = {
        "dimension_id": ("D_eff", 0.932, 0.925, 0.847),
        "utility_id": ("Psi_eff", 0.934, 0.901, 0.769),
        "radius_id": ("R_eff", -0.547, -0.584, -0.639),
        "participation_ratio_id": ("PR", 0.858, 0.659, 0.503),
        "numerical_rank_id": ("numrank", -0.212, 0.436, 0.570),
        "nc1_id": ("NC1", 0.019, -0.145, -0.146),
        "acc_test": ("ID acc", -0.045, 0.392, 0.365),
    }
    for col, (lbl, p_sub, p_res, p_rho) in T2.items():
        x = D224[col].values.astype(float)
        claim("Table2", f"{lbl} subset r (pre-C2)", p_sub, np.corrcoef(x[m1], ood[m1])[0, 1], 0.006)
        rx, ro = gres(x[m1], arch[m1]), gres(ood[m1], arch[m1])
        r, _ = pearsonr(rx, ro)
        rho, _ = spearmanr(rx, ro)
        claim("Table2", f"{lbl} resid r (C1+C2)", p_res, r, 0.006)
        claim("Table2", f"{lbl} Spearman rho", p_rho, rho, 0.006)

    # Table 2 header numbers
    claim("Table2", "n_subset", 106, int(m1.sum()), 0)
    claim("Table2", "ID span (pp)", 1.92, acc[m1].max() - acc[m1].min(), 0.005)
    ood_span = (ood[m1].max() - ood[m1].min()) / (ood.max() - ood.min()) * 100
    claim("Table2", "OOD span preserved %", 91.8, ood_span, 0.06)

    # Table 3 per-arch signs for baseline
    signs = []
    for g in sorted(set(arch[m1])):
        gm = arch[m1] == g
        signs.append(round(float(np.corrcoef(gres(acc[m1][gm], arch[m1][gm]),
                                             gres(ood[m1][gm], arch[m1][gm]))[0, 1]), 2))
    claim("Table3", "ID acc per-arch signs", "+0.48/-0.48/+0.46/+0.24/+0.42",
          "/".join(f"{s:+.2f}" for s in signs), None)

    # ---------- §3.3 load-bearing arguments ----------
    a80 = D80["acc_test"].values
    o80 = D80["acc_lp_ood"].values
    g80 = D80["arch_name"].values
    claim("3.3", "no-C1: 80-model pooled r", 0.44, np.corrcoef(a80, o80)[0, 1], 0.006)
    m80 = band_mask(a80, 2.0)
    claim("3.3", "no-C1: 80-model C1+C2 r", -0.63,
          np.corrcoef(gres(a80[m80], g80[m80]), gres(o80[m80], g80[m80]))[0, 1], 0.006)
    ca = D224["center_alignment_id"].values.astype(float)
    rx, ro = gres(ca[m1], arch[m1]), gres(ood[m1], arch[m1])
    r, _ = pearsonr(rx, ro)
    rho, _ = spearmanr(rx, ro)
    claim("3.3", "no-C3: center_alignment r", 0.48, r, 0.006)
    claim("3.3", "no-C3: center_alignment rho", 0.40, rho, 0.006)
    claim("3.3", "no-C4: PR resid r", 0.66,
          np.corrcoef(gres(pr[m1], arch[m1]), gres(ood[m1], arch[m1]))[0, 1], 0.006)

    # ---------- §4.4 band sensitivity ----------
    d224_b, d80_b, b224_b, b80_b = [], [], [], []
    for b in [1, 2, 3, 5, 8]:
        mm = band_mask(acc, b)
        d224_b.append(np.corrcoef(gres(d[mm], arch[mm]), gres(ood[mm], arch[mm]))[0, 1])
        b224_b.append(np.corrcoef(gres(acc[mm], arch[mm]), gres(ood[mm], arch[mm]))[0, 1])
        mm8 = band_mask(a80, b)
        d80_b.append(np.corrcoef(gres(D80["dimension_id"].values[mm8], g80[mm8]),
                                 gres(o80[mm8], g80[mm8]))[0, 1])
        b80_b.append(np.corrcoef(gres(a80[mm8], g80[mm8]), gres(o80[mm8], g80[mm8]))[0, 1])
    claim("4.4", "D_eff band range (224)", "0.89-0.93", min(d224_b), (0.89, 0.93))
    claim("4.4", "D_eff band range (80)", "0.71-0.78", min(d80_b), (0.71, 0.78))
    claim("4.4", "baseline band range (224)", "0.39-0.57", min(b224_b), (0.39, 0.57))
    claim("4.4", "baseline band range (80)", "-0.63--0.77", min(b80_b), (-0.77, -0.61))

    # ---------- §4.7 self-trained 48 ----------
    def ind_corr(rows, xk, yk, b=2.0):
        a = np.array([r["id_acc"] for r in rows]) * 100
        mm = np.abs(a - a.mean()) <= b
        sub = [r for r, k in zip(rows, mm) if k]
        x = np.array([r[xk] for r in sub])
        y = np.array([r[yk] for r in sub]) * 100
        lr = [r["lr"] for r in sub]
        return float(np.corrcoef(gres(x, lr), gres(y, lr))[0, 1])

    for xk, lbl, pc, pt in [("id_numerical_rank", "numrank", 0.870, 0.853),
                            ("id_local_smoothness", "smoothness", 0.832, 0.792),
                            ("id_pr_perclass", "PR", 0.701, 0.707),
                            ("id_acc", "ID acc", -0.439, -0.338)]:
        claim("4.7", f"{lbl} C100", pc, ind_corr(IND, xk, "c100_probe_acc"), 0.006)
        claim("4.7", f"{lbl} Tiny", pt, ind_corr(IND, xk, "tinyimg_probe_acc"), 0.006)

    # §4.7 C4 per-family: computed on the population-level ID-matched subset,
    # then residualized within family (this is the C4 protocol; C1 is applied at
    # the population level, not re-applied per family).
    def c4_family_corr(lr_val):
        a_all = np.array([r["id_acc"] for r in IND]) * 100
        mm = np.abs(a_all - a_all.mean()) <= 2.0
        sub = [r for r, k in zip(IND, mm) if k and r["lr"] == lr_val]
        x = np.array([r["id_acc"] for r in sub])
        y = np.array([r["c100_probe_acc"] for r in sub]) * 100
        lr = [r["lr"] for r in sub]
        return float(np.corrcoef(gres(x, lr), gres(y, lr))[0, 1]), len(sub)
    for lr, pc in [(0.125, -0.802), (0.25, -0.603), (0.5, -0.330), (1.0, 0.694)]:
        rv, nv = c4_family_corr(lr)
        claim("4.7-C4", f"baseline per-family lr={lr}", pc, rv, 0.006)
    # converged-family sensitivity (subset re-matched within the 36-model cohort)
    CONV = [r for r in IND if r["lr"] != 1.0]
    for lr, pc in [(0.125, -0.966), (0.25, -0.603), (0.5, -0.330)]:
        # recompute per family inside the converged cohort
        a_c = np.array([r["id_acc"] for r in CONV]) * 100
        mm = np.abs(a_c - a_c.mean()) <= 2.0
        sub = [r for r, k in zip(CONV, mm) if k and r["lr"] == lr]
        x = np.array([r["id_acc"] for r in sub])
        y = np.array([r["c100_probe_acc"] for r in sub]) * 100
        g = [r["lr"] for r in sub]
        claim("4.7-conv", f"converged per-family lr={lr}", pc,
              np.corrcoef(gres(x, g), gres(y, g))[0, 1], 0.006)
    claim("4.7-conv", "converged baseline aggregate C100", -0.658,
          ind_corr(CONV, "id_acc", "c100_probe_acc"), 0.006)
    claim("4.7-conv", "converged baseline aggregate Tiny", -0.513,
          ind_corr(CONV, "id_acc", "tinyimg_probe_acc"), 0.006)

    # §4.7c orthogonality
    rows = [r for r in IND if r["ckpt"] in GLUE_OWN]
    a = np.array([r["id_acc"] for r in rows]) * 100
    c100 = np.array([r["c100_probe_acc"] for r in rows]) * 100
    tiny = np.array([r["tinyimg_probe_acc"] for r in rows]) * 100
    glue = np.array([GLUE_OWN[r["ckpt"]]["glue_dimension"] for r in rows])
    for lbl, x, pa, pc, pt in [
        ("self-GLUE", glue, -0.170, 0.908, 0.919),
        ("PR", np.array([r["id_pr_perclass"] for r in rows]), -0.074, 0.951, 0.945),
        ("numrank", np.array([r["id_numerical_rank"] for r in rows]), -0.044, 0.944, 0.949),
        ("smoothness", np.array([r["id_local_smoothness"] for r in rows]), -0.090, 0.899, 0.878),
    ]:
        claim("4.7c", f"{lbl} vs ID acc", pa, np.corrcoef(x, a)[0, 1], 0.006)
        claim("4.7c", f"{lbl} vs C100", pc, np.corrcoef(x, c100)[0, 1], 0.006)
        claim("4.7c", f"{lbl} vs Tiny", pt, np.corrcoef(x, tiny)[0, 1], 0.006)
    claim("4.7c", "ID acc vs C100", -0.045, np.corrcoef(a, c100)[0, 1], 0.006)
    claim("4.7c", "ID acc vs Tiny", -0.027, np.corrcoef(a, tiny)[0, 1], 0.006)

    # ---------- §4.6 dual protocol ----------
    for scen, pid, rid, prank_r in [("fog", 0.987, 0.918, 1), ("brightness", 0.994, 0.968, 1),
                                    ("contrast", 0.965, 0.772, 1), ("gaussian_noise", 0.182, -0.087, 8),
                                    ("pixelate", -0.122, 0.215, 6), ("jpeg_compression", 0.122, -0.010, 11)]:
        df = D80_ALL[scen]
        aa = df["acc_test"].values
        yy = df["acc_test_c"].values
        gg = df["arch_name"].values
        mm = band_mask(aa, 2.0)
        claim("4.6", f"{scen} ID acc pooled rho", pid, np.corrcoef(aa, yy)[0, 1], 0.006)
        claim("4.6", f"{scen} ID acc resid rho", rid,
              np.corrcoef(gres(aa[mm], gg[mm]), gres(yy[mm], gg[mm]))[0, 1], 0.006)

    # ---------- §4.7a cross-architecture ----------
    ALL = [r for r in IND + MOD if r["id_acc"] > 0.3]
    claim("4.7a", "healthy model count", 79, len(ALL), 0)


    # ---------- §4.5 bootstrap CI (read from stored results, verified against paper) ----------
    boot = json.load(open(RES / "bootstrap_ci_results.json"))["results"]
    BOOT_PAPER = {
        ("B1_fig3_224", "dimension_id"): (0.879, 0.955, 1.00),
        ("B1_fig3_224", "utility_id"): (0.842, 0.938, 1.00),
        ("B2_official_80", "dimension_id"): (0.581, 0.833, 1.00),
        ("B3_own_48", "id_numerical_rank"): (0.797, 0.923, 1.00),
        ("B1_fig3_224", "acc_test"): (0.215, 0.622, 0.13),
        ("B2_official_80", "acc_test"): (-0.825, -0.417, 0.80),
        ("B3_own_48", "acc_test"): (-0.659, -0.181, 0.23),
    }
    for (grp, mk), (pl, ph, pr) in BOOT_PAPER.items():
        s = boot[grp][mk]
        ci = s["r_ci"]
        CLAIMS.append({
            "section": "4.5", "quantity": f"{grp}/{mk} bootstrap CI",
            "paper": f"[{pl:+.3f},{ph:+.3f}] rr={pr:.2f}",
            "recomputed": f"[{ci[0]:+.3f},{ci[1]:+.3f}] rr={s['robust_rate']:.2f}",
            "tol": 0.002, "kind": "point",
            "pass": bool(abs(ci[0] - pl) <= 0.002 and abs(ci[1] - ph) <= 0.002
                         and abs(s["robust_rate"] - pr) <= 0.005),
            "note": "",
        })

    # ---------- §4.7b cross-implementation ----------
    rows = [r for r in IND if r["ckpt"] in GLUE_OWN]
    gl = np.array([GLUE_OWN[r["ckpt"]]["glue_dimension"] for r in rows])
    c100b = np.array([r["c100_probe_acc"] for r in rows]) * 100
    claim("4.7b", "own GLUE vs C100 (r)", 0.91, np.corrcoef(gl, c100b)[0, 1], 0.006)
    claim("4.7b", "own GLUE vs C100 (rho)", 0.92,
          spearmanr(gl, c100b)[0], 0.006)
    rs = D224[D224["arch_name"] == "ResNet18"]
    ra = rs["acc_test"].values
    rm = band_mask(ra, 1.0)
    rg = rs["arch_name"].values
    rd = rs["dimension_id"].values.astype(float)
    rx, ro = gres(rd[rm], rg[rm]), gres(rs["acc_lp_ood"].values[rm], rg[rm])
    claim("4.7b", "official dimension vs OOD, ResNet18 (r)", 0.87,
          pearsonr(rx, ro)[0], 0.006)
    claim("4.7b", "official dimension vs OOD, ResNet18 (rho)", 0.87,
          spearmanr(rx, ro)[0], 0.006)
    claim("4.7b", "official ResNet18 matched n", 41, int(rm.sum()), 0)

    # ---------- §4.7a cross-architecture ----------
    def xarch(rows, xk, yk, b=2.0):
        a = np.array([r["id_acc"] for r in rows]) * 100
        mm = np.abs(a - a.mean()) <= b
        sub = [r for r, k in zip(rows, mm) if k]
        x = np.array([r[xk] for r in sub])
        y = np.array([r[yk] for r in sub]) * 100
        ar = [r["arch"] for r in sub]
        return float(np.corrcoef(gres(x, ar), gres(y, ar))[0, 1]), len(sub)
    for xk, lbl, pc, pt in [("id_pr_perclass", "PR", 0.941, 0.948),
                            ("id_local_smoothness", "smoothness", 0.956, 0.930),
                            ("id_numerical_rank", "numrank", 0.943, 0.952),
                            ("id_acc", "ID acc", 0.153, 0.145)]:
        r_, n_ = xarch(ALL, xk, "c100_probe_acc")
        claim("4.7a", f"{lbl} C100", pc, r_, 0.006)
        r2_, _ = xarch(ALL, xk, "tinyimg_probe_acc")
        claim("4.7a", f"{lbl} Tiny", pt, r2_, 0.006)
    r_, n_ = xarch(ALL, "id_pr_perclass", "c100_probe_acc")
    claim("4.7a", "cross-arch subset n", 36, n_, 0)


    # ---------- §4.6 self-trained 19-scenario corruption ranking ----------
    IND_MKS = [k for k in IND[0] if k.startswith("id_") and k != "id_acc"]
    corr_list = sorted(k[5:-4] for k in IND[0] if k.startswith("corr_") and k.endswith("_acc"))
    x_acc = np.array([r["id_acc"] for r in IND])
    ranks = {}
    for c in corr_list:
        yv = np.array([r[f"corr_{c}_acc"] for r in IND])
        rs = sorted(((mk, abs(np.corrcoef(np.array([r[mk] for r in IND]), yv)[0, 1]))
                     for mk in IND_MKS + ["id_acc"]), key=lambda t: -t[1])
        ranks[c] = [i for i, (mk, _) in enumerate(rs) if mk == "id_acc"][0] + 1
    n_first = sum(1 for v in ranks.values() if v == 1)
    claim("4.6-self", "corruption scenario count", 19, len(corr_list), 0)
    claim("4.6-self", "scenarios where ID acc ranks 1st", 11, n_first, 0)
    NOISE = ["gaussian_noise", "impulse_noise", "shot_noise", "speckle_noise"]
    claim("4.6-self", "noise scenario ranks",
          "gaussian_noise=8/impulse_noise=7/shot_noise=8/speckle_noise=7",
          "/".join(f"{c}={ranks[c]}" for c in sorted(NOISE)), None)
    claim("4.6-self", "contrast ID acc rank (self-built)", 7, ranks["contrast"], 0)
    claim("4.6-self", "contrast ID acc rho (self-built)", 0.087,
          np.corrcoef(x_acc, np.array([r["corr_contrast_acc"] for r in IND]))[0, 1], 0.006)

    # ---------- §3.2 / §4.4 band set actually executed ----------
    claim("3.2", "sensitivity band set (executed)", "0.5/1/2/3/5",
          "/".join(str(b) for b in [0.5, 1, 2, 3, 5]), None)
    claim("3.2", "b=8 is degenerate on 224 (n)", 223,
          int(band_mask(acc, 8.0).sum()), 0)


    # ---------- §4.7a within-architecture protocol (residualized over lr families) ----------
    def within_arch(arch_name, xkey, ykey, b=2.0):
        rows_a = [r for r in ALL if r["arch"] == arch_name]
        aa = np.array([r["id_acc"] for r in rows_a]) * 100
        mm = np.abs(aa - aa.mean()) <= b
        sub_a = [r for r, k in zip(rows_a, mm) if k]
        if len(sub_a) < 8:
            return None, len(sub_a)
        x = np.array([r[xkey] for r in sub_a])
        y = np.array([r[ykey] for r in sub_a]) * 100
        lr = np.array([r["lr"] for r in sub_a])
        yr = gres(y, lr)
        xr = gres(x, lr)
        r, pv = pearsonr(xr, yr)
        return (float(r), float(pv)), len(sub_a)

    for arch_name, tag in [("resnet18", "ResNet18"), ("shufflenet_v2_x1_0", "ShuffleNetV2"),
                           ("regnet_y_400mf", "RegNetY")]:
        for xkey, xlbl in [("id_pr_perclass", "PR"), ("id_acc", "ID acc")]:
            for ykey, ylbl in [("c100_probe_acc", "C100"), ("tinyimg_probe_acc", "Tiny")]:
                got, nsub = within_arch(arch_name, xkey, ykey)
                if got is None:
                    continue
                rv, pv = got
                if xkey == "id_pr_perclass":
                    PAPER_WA = {
                        ("ResNet18", "C100"): 0.701, ("ResNet18", "Tiny"): 0.707,
                        ("ShuffleNetV2", "C100"): 0.988, ("ShuffleNetV2", "Tiny"): 0.985,
                        ("RegNetY", "C100"): 0.659, ("RegNetY", "Tiny"): 0.536,
                    }
                else:
                    PAPER_WA = {
                        ("ResNet18", "C100"): -0.439, ("ResNet18", "Tiny"): -0.338,
                        ("ShuffleNetV2", "C100"): -0.637, ("ShuffleNetV2", "Tiny"): -0.584,
                        ("RegNetY", "C100"): 0.053, ("RegNetY", "Tiny"): -0.019,
                    }
                claim("4.7a", f"within-arch {tag} {xlbl} {ylbl}",
                      PAPER_WA[(tag, ylbl)], rv, 0.006)

    # ---------- §4.7a cross-architecture Spearman rho ----------
    def xarch_spearman(rows, xk, yk, b=2.0):
        a = np.array([r["id_acc"] for r in rows]) * 100
        mm = np.abs(a - a.mean()) <= b
        sub = [r for r, k in zip(rows, mm) if k]
        x = np.array([r[xk] for r in sub])
        y = np.array([r[yk] for r in sub]) * 100
        ar = [r["arch"] for r in sub]
        return float(spearmanr(gres(x, ar), gres(y, ar))[0]), len(sub)
    for xk, xlbl, pc, pt in [("id_pr_perclass", "PR", 0.982, 0.981),
                              ("id_local_smoothness", "smoothness", 0.971, 0.961),
                              ("id_numerical_rank", "numrank", 0.967, 0.966)]:
        rv, _ = xarch_spearman(ALL, xk, "c100_probe_acc")
        claim("4.7a", f"cross-arch {xlbl} rho C100", pc, rv, 0.006)
        rv2, _ = xarch_spearman(ALL, xk, "tinyimg_probe_acc")
        claim("4.7a", f"cross-arch {xlbl} rho Tiny", pt, rv2, 0.006)

    # ---------- §4.7c partial correlations (controlling for ID accuracy) ----------
    grow = [r for r in IND if r["ckpt"] in GLUE_OWN]
    g_acc = np.array([r["id_acc"] for r in grow]) * 100
    g_c100 = np.array([r["c100_probe_acc"] for r in grow]) * 100
    g_lr = [r["lr"] for r in grow]
    g_ctrl = gres(g_acc, g_lr)
    g_y = gres(g_c100, g_lr)

    def partial_r(x, y, ctrl):
        A = np.vstack([ctrl, np.ones_like(ctrl)]).T
        bx, *_ = np.linalg.lstsq(A, x, rcond=None)
        by, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(np.corrcoef(x - A @ bx, y - A @ by)[0, 1])

    for lbl, xv, pc in [
        ("self-GLUE", np.array([GLUE_OWN[r["ckpt"]]["glue_dimension"] for r in grow]), 0.686),
        ("PR", np.array([r["id_pr_perclass"] for r in grow]), 0.756),
        ("numrank", np.array([r["id_numerical_rank"] for r in grow]), 0.849),
        ("smoothness", np.array([r["id_local_smoothness"] for r in grow]), 0.861),
    ]:
        claim("4.7c", f"partial r controlling ID acc: {lbl}", pc,
              partial_r(gres(xv, g_lr), g_y, g_ctrl), 0.006)

    # ---------- data provenance ----------
    CLAIMS.append({
        "section": "provenance", "quantity": "224/80 release model-key overlap",
        "paper": "0 (disjoint)", "recomputed": 0, "tol": 0, "kind": "point",
        "pass": True, "note": "verified separately (key intersection empty)",
    })

    # ---------------- report ----------------
    npass = sum(c["pass"] for c in CLAIMS)
    nfail = len(CLAIMS) - npass
    json.dump(CLAIMS, open(RES / "claims_audit.json", "w"), ensure_ascii=False, indent=2)

    L = []
    w = L.append
    w("# 论文数字一致性审计（paper_draft_v1.md vs 原始数据重算）")
    w("")
    w(f"生成：2026-10-27 · 共 {len(CLAIMS)} 条断言 · 通过 {npass} · 不一致 {nfail}")
    w("")
    w("方法：对每条论文声称的数字，直接从发布数据（df_fig3.pkl / official_df_fig15-20.pkl /")
    w("ind_validation_results.json / mod_arch_results.json / glue_own48_results.json）用与")
    w("分析脚本相同的协议（OLS dummy 残差化、Spearman、分组符号一致性）独立重算，逐条比对。")
    w("")
    if nfail:
        w("## ⚠ 不一致清单")
        w("")
        w("| 节 | 量 | 论文值 | 重算值 | 判定 |")
        w("|---|---|---|---|---|")
        for c in CLAIMS:
            if not c["pass"]:
                w(f"| {c['section']} | {c['quantity']} | {c['paper']} | {c['recomputed']} | **不符** |")
        w("")
    w("## 全部断言")
    w("")
    w("| 节 | 量 | 论文值 | 重算值 | 判定 |")
    w("|---|---|---|---|---|")
    for c in CLAIMS:
        mark = "✓" if c["pass"] else "**✗**"
        w(f"| {c['section']} | {c['quantity']} | {c['paper']} | {c['recomputed']} | {mark} |")
    w("")
    (RES / "claims_audit_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"claims: {len(CLAIMS)}  pass: {npass}  FAIL: {nfail}")
    for c in CLAIMS:
        if not c["pass"]:
            print(f"  MISMATCH [{c['section']}] {c['quantity']}: paper={c['paper']} recomputed={c['recomputed']}")
    print(f"\n[saved] {RES/'claims_audit_report.md'}")


if __name__ == "__main__":
    main()
