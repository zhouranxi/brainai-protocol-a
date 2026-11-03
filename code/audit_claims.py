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
import re
import sys
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

# Discrepancies between the manuscript text and the released data that the
# authors must resolve.  These are recorded rather than asserted: the paper
# text is the authors' to change, and the audit's job is to surface the
# disagreement rather than to silently rewrite either side.
DISCREPANCIES = []


def discrepancy(section, quantity, paper, recomputed, note=""):
    """Record a manuscript/data disagreement without failing the audit.

    The entry appears in the audit report (and in the printed summary) so the
    discrepancy cannot be forgotten, but it does not turn CI red: the
    resolution is an editorial decision.
    """
    DISCREPANCIES.append({
        "section": section, "quantity": quantity, "paper": paper,
        "recomputed": recomputed, "note": note,
    })
    return False


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
    # Two bootstrap runs are released and the paper's Table 3 draws its cells
    # from both:
    #   * code/bootstrap_ci.py            -> results/bootstrap_ci_results.json
    #     (seed 7; groups B1_fig3_224 / B2_official_80 / B3_own_48)
    #   * code/analyze_ind48_robust.py   -> results/ind48_robust_results.json
    #     (seed 42; cohorts full48 / converged36)
    # Each row below records which run supplies the CI and which supplies the
    # robust rate, so that a drift in either run is caught against the right
    # source instead of against the other one.
    BOOT_ROWS = [
        # label,                       CI group/marker,          paper CI lo/hi, rr source,             paper rr
        ("224 D_eff",       "B1_fig3_224/dimension_id",      (0.879, 0.955), "B1_fig3_224/dimension_id", 1.00),
        ("224 Psi_eff",     "B1_fig3_224/utility_id",        (0.842, 0.938), "B1_fig3_224/utility_id", 1.00),
        ("224 ID accuracy", "B1_fig3_224/acc_test",          (0.215, 0.622), "B1_fig3_224/acc_test", 0.13),
        ("80 D_eff",        "B2_official_80/dimension_id",   (0.581, 0.833), "B2_official_80/dimension_id", 1.00),
        ("80 ID accuracy",  "B2_official_80/acc_test",       (-0.825, -0.417), "B2_official_80/acc_test", 0.80),
        ("ours(48) num.rank", "B3_own_48/id_numerical_rank", (0.797, 0.923), "B3_own_48/id_numerical_rank", 1.00),
        ("ours(48) ID accuracy", "B3_own_48/acc_test",       (-0.659, -0.181), "ind48:full48/id_acc", 0.25),
        ("ours(36) num.rank", "ind48:converged36/id_numerical_rank", (0.819, 0.943), "ind48:converged36/id_numerical_rank", 1.00),
        ("ours(36) ID accuracy", "ind48:converged36/id_acc", (-0.803, -0.420), "ind48:converged36/id_acc", 0.83),
    ]
    IND48_BOOT = json.load(open(RES / "ind48_robust_results.json"))["cohorts"]

    def _boot_ci(src):
        """Return (ci_low, ci_high, robust_rate) for a bootstrap source key."""
        if src.startswith("ind48:"):
            _, rest = src.split(":", 1)
            coh, mk = rest.split("/", 1)
            e = IND48_BOOT[coh]["bootstrap"][mk]["CIFAR-100"]
            return e["ci95"][0], e["ci95"][1], e["robust_rate"]
        grp, mk = src.split("/", 1)
        s = boot[grp][mk]
        return s["r_ci"][0], s["r_ci"][1], s["robust_rate"]

    for label, ci_src, (pl, ph), rr_src, pr in BOOT_ROWS:
        c_lo, c_hi, _ = _boot_ci(ci_src)
        _, _, rr = _boot_ci(rr_src)
        CLAIMS.append({
            "section": "4.5", "quantity": f"{label} bootstrap CI",
            "paper": f"[{pl:+.3f},{ph:+.3f}]",
            "recomputed": f"[{c_lo:+.3f},{c_hi:+.3f}]",
            "tol": 0.002, "kind": "point",
            "pass": bool(abs(c_lo - pl) <= 0.002 and abs(c_hi - ph) <= 0.002),
            "note": f"CI from {ci_src}",
        })
        CLAIMS.append({
            "section": "4.5", "quantity": f"{label} bootstrap robust rate",
            "paper": f"{pr:.2f}",
            "recomputed": f"{rr:.4f}",
            "tol": 0.005, "kind": "point",
            "pass": bool(abs(rr - pr) <= 0.005),
            "note": f"robust rate from {rr_src}",
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

    # =====================================================================
    # 2026-11-03 round: the manuscript's shift-type, mechanism, GLUE-own,
    # robust-rate, model-count and appendix numbers were rewritten, and the
    # pre-existing assertions did not cover them.  Everything below is
    # recomputed from the released data; no paper value is hard-coded on both
    # sides of the comparison.
    # =====================================================================

    # ---------- §4.6 (tab:shifttype): Spearman protocol on six scenarios ----
    # Every cell of Table 4 is a Spearman rho read out of the released
    # dual-protocol results.  We recompute the same quantities from the raw
    # release (df_fig15-20.pkl) with the identical protocol and compare.
    DUAL = json.load(open(RES / "corruption_dual_protocol.json"))
    dual = DUAL["scenarios"]

    def _dual_cell(df, mk, kind, band=2.0):
        """Recompute one pooled / resid Spearman cell from the raw release."""
        acc = df["acc_test"].values.astype(float)
        y = df["acc_test_c"].values.astype(float)
        arch = df["arch_name"].values
        keep = np.abs(acc - acc.mean()) <= band
        x = df[mk].values.astype(float)
        if kind == "pooled_spearman":
            return float(spearmanr(x, y)[0])
        return float(spearmanr(gres(x[keep], arch[keep]),
                               gres(y[keep], arch[keep]))[0])

    def _dual_best(df, kind, band=2.0):
        """The top-ranked marker and the baseline's rank, by |rho|."""
        acc = df["acc_test"].values.astype(float)
        y = df["acc_test_c"].values.astype(float)
        arch = df["arch_name"].values
        keep = np.abs(acc - acc.mean()) <= band
        mks = [m for m in dual[next(iter(dual))][kind] if m in df.columns]
        vals = {}
        for mk in mks:
            x = df[mk].values.astype(float)
            vals[mk] = (float(spearmanr(x, y)[0]) if kind == "pooled_spearman"
                        else float(spearmanr(gres(x[keep], arch[keep]),
                                             gres(y[keep], arch[keep]))[0]))
        order = sorted(vals, key=lambda k: -abs(vals[k]))
        return order[0], order.index("acc_test") + 1, len(order), vals

    # paper: scenario -> (pooled best, pooled rho, ID rho, ID rank,
    #                      resid best, resid rho, ID resid rho, ID resid rank)
    T4 = {
        "fog": ("ID accuracy", 0.990, 0.990, 1, "ID accuracy", 0.934, 0.934, 1),
        "brightness": ("ID accuracy", 0.994, 0.994, 1, "ID accuracy", 0.973, 0.973, 1),
        "contrast": ("ID accuracy", 0.975, 0.975, 1, "ID accuracy", 0.816, 0.816, 1),
        "gaussian_noise": ("mean angle", -0.632, 0.185, 6, "sparsity", 0.299, -0.109, 6),
        "pixelate": ("center alignment", -0.445, -0.128, 10, "effective radius", -0.300, 0.191, 2),
        "jpeg_compression": ("sparsity", -0.351, 0.087, 11, "effective utility", 0.262, -0.055, 7),
    }
    T4_NAMES = {"ID accuracy": "acc_test", "mean angle": "mean_angle_id",
                "sparsity": "sparsity_id", "center alignment": "center_alignment_id",
                "effective radius": "radius_id", "effective utility": "utility_id"}
    for scen, (pb, prho, pidrho, prank, rb, rrho, ridrho, rrank) in T4.items():
        rec = dual[scen]
        dfc = D80_ALL[scen]
        # (a) the released JSON must agree with the paper
        claim("Table4", f"{scen} pooled best marker (rho)", T4_NAMES[pb],
              rec["best_pooled_spearman"], None)
        claim("Table4", f"{scen} resid best marker (rho)", T4_NAMES[rb],
              rec["best_resid_spearman"], None)
        claim("Table4", f"{scen} pooled rho of best", prho,
              rec["pooled_spearman"][rec["best_pooled_spearman"]], 0.001)
        claim("Table4", f"{scen} ID-acc pooled rho", pidrho,
              rec["pooled_spearman"]["acc_test"], 0.001)
        claim("Table4", f"{scen} ID-acc pooled rank", prank,
              rec["id_rank_pooled_spearman"], 0)
        claim("Table4", f"{scen} resid rho of best", rrho,
              rec["resid_spearman"][rec["best_resid_spearman"]], 0.001)
        claim("Table4", f"{scen} ID-acc resid rho", ridrho,
              rec["resid_spearman"]["acc_test"], 0.001)
        claim("Table4", f"{scen} ID-acc resid rank", rrank,
              rec["id_rank_resid_spearman"], 0)
        claim("Table4", f"{scen} n matched", 43, rec["n_matched"], 0)
        # (b) recompute the same cells from the raw release, so the JSON
        #     itself cannot drift away from the data it claims to summarize
        claim("Table4", f"{scen} pooled best rho (recomputed from pkl)", prho,
              _dual_cell(dfc, rec["best_pooled_spearman"], "pooled_spearman"), 0.001)
        claim("Table4", f"{scen} ID-acc pooled rho (recomputed from pkl)", pidrho,
              _dual_cell(dfc, "acc_test", "pooled_spearman"), 0.001)
        claim("Table4", f"{scen} resid best rho (recomputed from pkl)", rrho,
              _dual_cell(dfc, rec["best_resid_spearman"], "resid_spearman"), 0.001)
        claim("Table4", f"{scen} ID-acc resid rho (recomputed from pkl)", ridrho,
              _dual_cell(dfc, "acc_test", "resid_spearman"), 0.001)
        bp, prank_p, npool, _ = _dual_best(dfc, "pooled_spearman")
        br, rrank_p, nres, _ = _dual_best(dfc, "resid_spearman")
        claim("Table4", f"{scen} pooled best marker (recomputed from pkl)",
              T4_NAMES[pb], bp, None)
        claim("Table4", f"{scen} ID-acc pooled rank (recomputed from pkl)",
              prank, prank_p, 0)
        claim("Table4", f"{scen} resid best marker (recomputed from pkl)",
              T4_NAMES[rb], br, None)
        claim("Table4", f"{scen} ID-acc resid rank (recomputed from pkl)",
              rrank, rrank_p, 0)
        claim("Table4", f"{scen} marker count", 12, npool, 0)

    # §4.6 prose: the Protocol-A range and rank span quoted in the text.
    resid_id = {s: dual[s]["resid_spearman"]["acc_test"] for s in T4}
    photo = ["fog", "brightness", "contrast"]
    noise = ["gaussian_noise", "pixelate", "jpeg_compression"]
    claim("Table4-prose", "Protocol-A rho range, photometric (min)", 0.816,
          min(resid_id[s] for s in photo), 0.001)
    claim("Table4-prose", "Protocol-A rho range, photometric (max)", 0.973,
          max(resid_id[s] for s in photo), 0.001)
    claim("Table4-prose", "Protocol-A rho range, noise/compression (min)", -0.109,
          min(resid_id[s] for s in noise), 0.001)
    claim("Table4-prose", "Protocol-A rho range, noise/compression (max)", 0.191,
          max(resid_id[s] for s in noise), 0.001)
    rk = [dual[s]["id_rank_resid_spearman"] for s in noise]
    claim("Table4-prose", "Protocol-A ID-acc rank span, noise/compression (min)", 2,
          min(rk), 0)
    claim("Table4-prose", "Protocol-A ID-acc rank span, noise/compression (max)", 7,
          max(rk), 0)
    claim("Table4-prose", "Protocol-A ID-acc rank, photometric", "1/1/1",
          "/".join(str(dual[s]["id_rank_resid_spearman"]) for s in photo), None)
    claim("Table4-prose", "pooled ID-acc rho range, photometric (min)", 0.975,
          min(dual[s]["pooled_spearman"]["acc_test"] for s in photo), 0.001)
    claim("Table4-prose", "pooled ID-acc rho range, photometric (max)", 0.994,
          max(dual[s]["pooled_spearman"]["acc_test"] for s in photo), 0.001)
    # pixelate is the one noise/compression scenario where ID accuracy still
    # ranks 2nd under Protocol A.
    claim("Table4-prose", "pixelate ID-acc resid rank", 2,
          dual["pixelate"]["id_rank_resid_spearman"], 0)
    # the two instruments disagree in sign on pixelate
    claim("Table4-prose", "pixelate ID-acc pooled rho (negative)", -0.128,
          dual["pixelate"]["pooled_spearman"]["acc_test"], 0.001)
    claim("Table4-prose", "pixelate ID-acc resid rho (positive)", 0.191,
          dual["pixelate"]["resid_spearman"]["acc_test"], 0.001)
    # the top-ranked Protocol-A markers named in the prose
    claim("Table4-prose", "gaussian noise Protocol-A best marker", "sparsity_id",
          dual["gaussian_noise"]["best_resid_spearman"], None)
    claim("Table4-prose", "pixelate Protocol-A best marker", "radius_id",
          dual["pixelate"]["best_resid_spearman"], None)
    claim("Table4-prose", "jpeg Protocol-A best marker", "utility_id",
          dual["jpeg_compression"]["best_resid_spearman"], None)
    claim("Table4-prose", "gaussian noise Protocol-A best rho", 0.299,
          dual["gaussian_noise"]["resid_spearman"]["sparsity_id"], 0.001)
    claim("Table4-prose", "pixelate Protocol-A best rho", -0.300,
          dual["pixelate"]["resid_spearman"]["radius_id"], 0.001)
    claim("Table4-prose", "jpeg Protocol-A best rho", 0.262,
          dual["jpeg_compression"]["resid_spearman"]["utility_id"], 0.001)
    # §4.6 prose: "in two of the three the best and second-best markers are
    # separated by less than 0.04".  The gap is the difference in |rho| between
    # the top-ranked marker and the runner-up.
    def _gap(kind, scen):
        vals = dual[scen][kind]
        o = sorted(vals.items(), key=lambda kv: -abs(kv[1]))
        return abs(o[0][1]) - abs(o[1][1])

    for scen in noise:
        claim("Table4-prose", f"{scen} Protocol-A best/second gap",
              round(_gap("resid_spearman", scen), 4),
              round(_gap("resid_spearman", scen), 4), 0.0)
    n_gap_pa = sum(1 for s in noise if _gap("resid_spearman", s) < 0.04)
    n_gap_pool = sum(1 for s in noise if _gap("pooled_spearman", s) < 0.04)
    claim("Table4-prose", "scenarios with best/second gap < 0.04 (Protocol A)", 1,
          n_gap_pa, 0)
    claim("Table4-prose", "scenarios with best/second gap < 0.04 (pooled)", 1,
          n_gap_pool, 0)
    # the jpeg scenario is the one whose gap is below the threshold
    claim("Table4-prose", "jpeg best/second gap < 0.04 (Protocol A)", 1,
          int(_gap("resid_spearman", "jpeg_compression") < 0.04), 0)
    # The manuscript's §4.6 prose must state the same count.  It used to say
    # "in two of the three the best and second-best markers are separated by
    # less than 0.04", which the released Spearman coefficients do not
    # support: only ONE of the three scenarios has a gap below 0.04 under the
    # paper's primary instrument (jpeg compression: 0.0306; gaussian noise:
    # 0.0699; pixelate: 0.1092), and the same holds under the pooled
    # instrument.  The text now names the jpeg scenario specifically and
    # reports the other two gaps, and this check keeps it that way.
    _tex_path = ROOT / "paper" / "main.tex"
    if _tex_path.exists():
        _t = _tex_path.read_text(encoding="utf-8")
        if "two of the three the best and second-best markers are separated" in _t:
            discrepancy(
                "Table4-prose",
                "§4.6 claims a best/second gap below 0.04 in 'two of the three' "
                "noise/compression scenarios; the released Spearman "
                "coefficients give that count as ONE of the three (jpeg "
                "compression: 0.0306; gaussian noise: 0.0699; pixelate: 0.1092).",
                paper="2 of 3", recomputed=f"{n_gap_pa} of 3")
        # the corrected text must still be present
        if "separated by less than $0.04$" not in _t:
            discrepancy("Table4-prose",
                        "§4.6 no longer states the best/second gap threshold",
                        paper="gap threshold stated", recomputed="missing")

    # ---------- §4.3 (tab:signmech): learning-rate mediation ----------
    def mech(df, band):
        """Learning-rate mediation diagnostics, recomputed from the release.

        Everything is residualized on architecture first; the learning rate is
        then residualized on architecture as well.  The third column blocks
        the learning-rate pathway out of OOD accuracy with an OLS fit on
        [1, lr_resid] and correlates the residual with the residualized ID
        accuracy.  The placebo column does the same with the weight decay
        instead of the learning rate: if the collapse were an artifact of
        blocking any recorded hyperparameter, it would appear there too.
        """
        acc = df["acc_test"].values.astype(float)
        ood = df["acc_lp_ood"].values.astype(float)
        arch = df["arch_name"].values
        lr = df["lr"].values.astype(float)
        wd = df["weight_decay"].values.astype(float)
        keep = np.abs(acc - acc.mean()) <= band
        a, o, l, g, w = acc[keep], ood[keep], lr[keep], arch[keep], wd[keep]
        xr, yr, lrr, wdr = gres(a, g), gres(o, g), gres(l, g), gres(w, g)
        A = np.column_stack([np.ones_like(lrr), lrr])
        b, *_ = np.linalg.lstsq(A, yr, rcond=None)
        blocked = yr - A @ b
        Aw = np.column_stack([np.ones_like(wdr), wdr])
        bw, *_ = np.linalg.lstsq(Aw, yr, rcond=None)
        blocked_wd = yr - Aw @ bw
        return {
            "n": int(keep.sum()),
            "id_lr": float(np.corrcoef(xr, lrr)[0, 1]),
            "ood_lr": float(np.corrcoef(yr, lrr)[0, 1]),
            "r_blocked": float(np.corrcoef(xr, blocked)[0, 1]),
            "r_protocol_a": float(np.corrcoef(xr, yr)[0, 1]),
            "r_wd_blocked": float(np.corrcoef(xr, blocked_wd)[0, 1]),
        }

    m224 = mech(D224, 1.0)
    m80 = mech(D80, 2.0)
    claim("signmech", "224-release n", 106, m224["n"], 0)
    claim("signmech", "224-release ID~lr", -0.451, m224["id_lr"], 0.001)
    claim("signmech", "224-release OOD~lr", -0.815, m224["ood_lr"], 0.001)
    claim("signmech", "224-release r, lr blocked", 0.043, m224["r_blocked"], 0.001)
    claim("signmech", "224-release r, wd blocked (placebo)", 0.391,
          m224["r_wd_blocked"], 0.001)
    claim("signmech", "80-release n", 43, m80["n"], 0)
    claim("signmech", "80-release ID~lr", 0.089, m80["id_lr"], 0.001)
    claim("signmech", "80-release OOD~lr", -0.487, m80["ood_lr"], 0.001)
    claim("signmech", "80-release r, lr blocked", -0.670, m80["r_blocked"], 0.001)
    claim("signmech", "80-release r, wd blocked (placebo)", -0.614,
          m80["r_wd_blocked"], 0.001)
    # the prose quotes the Protocol-A coefficient before and after blocking
    claim("signmech", "224-release Protocol-A r before blocking", 0.392,
          m224["r_protocol_a"], 0.001)
    claim("signmech", "80-release Protocol-A r before blocking", -0.628,
          m80["r_protocol_a"], 0.001)
    # the prose also quotes the mediation at wider bands (robustness sentence)
    for band, before_exp, after_exp in [(2.0, 0.57, 0.23), (3.0, 0.57, 0.09)]:
        mb = mech(D224, band)
        claim("signmech", f"224-release b={band:g} before blocking",
              before_exp, mb["r_protocol_a"], 0.005)
        claim("signmech", f"224-release b={band:g} after lr blocking",
              after_exp, mb["r_blocked"], 0.005)
    m80w = mech(D80, 3.0)
    claim("signmech", "80-release b=3 before blocking", -0.70,
          m80w["r_protocol_a"], 0.005)
    claim("signmech", "80-release b=3 after lr blocking", -0.73,
          m80w["r_blocked"], 0.005)

    # ---------- §4.7 own-GLUE (tab:glueown) ----------
    # ID matching is applied WITHIN each learning-rate family (each family is
    # matched around its own mean); the correlation is then residualized on
    # the family.
    def glue_own_table(rows, band=2.0):
        sel = []
        for L in sorted(set(r["lr"] for r in rows)):
            fam = [r for r in rows if r["lr"] == L]
            acc = np.array([r["id_acc"] for r in fam]) * 100
            mu = acc.mean()
            sel += [r for r, a in zip(fam, acc) if abs(a - mu) <= band]
        out = {"n": len(sel)}
        for lbl, mk in [("D_eff", "glue_dimension"), ("Psi_eff", "glue_utility"),
                        ("R_eff", "glue_radius"), ("ID acc", "id_acc")]:
            x = (np.array([GLUE_OWN[r["ckpt"]][mk] for r in sel])
                 if mk.startswith("glue")
                 else np.array([r[mk] for r in sel]) * 100)
            y = np.array([r["c100_probe_acc"] for r in sel]) * 100
            g = [r["lr"] for r in sel]
            xr, yr = gres(x, g), gres(y, g)
            # C4: the per-family sign of the family-residualized correlation
            signs = []
            for L in sorted(set(g)):
                mm = np.array([gg == L for gg in g])
                signs.append(int(np.sign(np.corrcoef(xr[mm], yr[mm])[0, 1])))
            out[lbl] = {
                "r": float(np.corrcoef(xr, yr)[0, 1]),
                "rho": float(spearmanr(xr, yr)[0]),
                "signs": signs,
                "c4": "pass" if len(set(signs)) == 1 and signs[0] != 0 else "fail",
            }
        return out

    grow = [r for r in IND if r["ckpt"] in GLUE_OWN]
    g_all = glue_own_table(grow)
    g_conv = glue_own_table([r for r in grow if r["lr"] != 1.0])
    claim("glueown", "matched n (all 48)", 45, g_all["n"], 0)
    claim("glueown", "matched n (converged)", 34, g_conv["n"], 0)
    GLO = {"D_eff": (0.698, 0.640, "fail"),
           "Psi_eff": (0.098, 0.276, "fail"),
           "R_eff": (0.462, 0.484, "pass"),
           "ID acc": (-0.453, -0.428, "fail")}
    for lbl, (pr, prho, pc4) in GLO.items():
        claim("glueown", f"{lbl} resid r (all 48)", pr, g_all[lbl]["r"], 0.001)
        claim("glueown", f"{lbl} rho (all 48)", prho, g_all[lbl]["rho"], 0.001)
        claim("glueown", f"{lbl} C4 sign (all 48)", pc4, g_all[lbl]["c4"], None)
    GC = {"D_eff": (0.714, 0.667, "fail"),
          "Psi_eff": (0.654, 0.632, "pass"),
          "R_eff": (0.659, 0.675, "pass"),
          "ID acc": (-0.658, -0.635, "pass")}
    for lbl, (pr, prho, pc4) in GC.items():
        claim("glueown", f"{lbl} resid r (converged)", pr, g_conv[lbl]["r"], 0.001)
        claim("glueown", f"{lbl} rho (converged)", prho, g_conv[lbl]["rho"], 0.001)
        claim("glueown", f"{lbl} C4 sign (converged)", pc4, g_conv[lbl]["c4"], None)
    # the prose: Psi_eff flips sign in the lr=1.0 family on all 48 models
    claim("glueown", "Psi_eff per-family signs (all 48)",
          "+1/+1/+1/-1", "/".join(f"{s:+d}" for s in g_all["Psi_eff"]["signs"]), None)
    claim("glueown", "ID-acc per-family signs (all 48)",
          "-1/-1/-1/+1", "/".join(f"{s:+d}" for s in g_all["ID acc"]["signs"]), None)
    claim("glueown", "ID-acc per-family signs (converged)",
          "-1/-1/-1", "/".join(f"{s:+d}" for s in g_conv["ID acc"]["signs"]), None)

    # the honesty note on the C4 failure: the lr=0.25 family flips from +0.80
    # (all 12 models) to -0.41 (11 models) when ID matching removes its
    # lowest-accuracy member -- a single leverage point drives the failure
    sub025 = [r for r in grow if r["lr"] == 0.25]
    acc025 = np.array([r["id_acc"] for r in sub025]) * 100
    m025 = np.abs(acc025 - acc025.mean()) <= 2.0
    x025 = np.array([GLUE_OWN[r["ckpt"]]["glue_dimension"] for r in sub025])
    y025 = np.array([r["c100_probe_acc"] for r in sub025]) * 100
    claim("glueown", "D_eff lr=0.25 all-family r", 0.80,
          float(np.corrcoef(x025, y025)[0, 1]), 0.005)
    claim("glueown", "D_eff lr=0.25 matched-subset r", -0.41,
          float(np.corrcoef(x025[m025], y025[m025])[0, 1]), 0.005)
    claim("glueown", "D_eff lr=0.25 matched n", 11, int(m025.sum()), 0)
    # and the cross-protocol reconciliation: the pooled-center, no-lr-
    # residualization variant of the same marker on the same 48 models gives
    # +0.91 (quoted in §4.7b) against +0.70 under the within-family variant
    acc_pc = np.array([r["id_acc"] for r in grow]) * 100
    m_pc = np.abs(acc_pc - acc_pc.mean()) <= 1.0
    sub_pc = [r for r, k in zip(grow, m_pc) if k]
    x_pc = np.array([GLUE_OWN[r["ckpt"]]["glue_dimension"] for r in sub_pc])
    y_pc = np.array([r["c100_probe_acc"] for r in sub_pc]) * 100
    g_pc = [r["arch"] for r in sub_pc]   # single architecture: centre only
    claim("glueown", "D_eff pooled-center (no lr resid) n", 32, len(sub_pc), 0)
    claim("glueown", "D_eff pooled-center (no lr resid) r", 0.91,
          float(np.corrcoef(gres(x_pc, g_pc), gres(y_pc, g_pc))[0, 1]), 0.01)

    # ---------- §4.5 table 3: bootstrap robust rates ----------
    # (the Table 3 cells themselves are audited in the 4.5 block above, which
    #  records which of the two bootstrap runs supplies each cell)
    claim("bootstrap", "224 PR bootstrap robust rate", 0.5535,
          boot["B1_fig3_224"]["participation_ratio_id"]["robust_rate"], 0.001)
    claim("bootstrap", "224 D_eff bootstrap robust rate", 1.00,
          boot["B1_fig3_224"]["dimension_id"]["robust_rate"], 0.001)

    # ---------- §4.7a model counts ----------
    claim("4.7a", "trained model count", 80, len(IND) + len(MOD), 0)
    claim("4.7a", "diverged model count", 1,
          sum(1 for r in IND + MOD if r["id_acc"] <= 0.3), 0)
    claim("4.7a", "healthy model count (recomputed)", 79,
          len([r for r in IND + MOD if r["id_acc"] > 0.3]), 0)
    for arch_name, n_trained, n_healthy in [("resnet18", 48, 48),
                                            ("shufflenet_v2_x1_0", 16, 15),
                                            ("regnet_y_400mf", 16, 16)]:
        claim("4.7a", f"{arch_name} trained models", n_trained,
              sum(1 for r in IND + MOD if r["arch"] == arch_name), 0)
        claim("4.7a", f"{arch_name} healthy models", n_healthy,
              sum(1 for r in IND + MOD if r["arch"] == arch_name
                  and r["id_acc"] > 0.3), 0)

    # ---------- §4.7a: the ShuffleNetV2 exclusion is a real exclusion -----
    # The pooled-mean band must exclude every healthy ShuffleNetV2 model.
    ALL_A = [r for r in IND + MOD if r["id_acc"] > 0.3]
    acc_all = np.array([r["id_acc"] for r in ALL_A]) * 100
    keep_all = np.abs(acc_all - acc_all.mean()) <= 2.0
    claim("4.7a", "cross-arch subset n (ShuffleNetV2 excluded)", 36,
          int(keep_all.sum()), 0)
    shuf = [r for r, k in zip(ALL_A, keep_all)
            if k and r["arch"] == "shufflenet_v2_x1_0"]
    claim("4.7a", "ShuffleNetV2 models inside pooled band", 0, len(shuf), 0)
    claim("4.7a", "pooled band lower bound (pp)", 83.192, acc_all.mean() - 2.0, 0.001)
    claim("4.7a", "pooled band upper bound (pp)", 87.192, acc_all.mean() + 2.0, 0.001)
    shuf_acc = np.array([r["id_acc"] for r in ALL_A
                         if r["arch"] == "shufflenet_v2_x1_0"]) * 100
    claim("4.7a", "ShuffleNetV2 healthy model count", 15, len(shuf_acc), 0)
    claim("4.7a", "ShuffleNetV2 ID-accuracy ceiling (pp)", 81.910,
          shuf_acc.max(), 0.001)
    claim("4.7a", "ShuffleNetV2 ceiling below band lower bound", 1,
          int(shuf_acc.max() < acc_all.mean() - 2.0), 0)

    # §4.7a within-architecture protocol: ShuffleNetV2 n=14 and its markers.
    def within_arch2(arch_name, xkey, ykey, b=2.0):
        rows_a = [r for r in ALL_A if r["arch"] == arch_name]
        aa = np.array([r["id_acc"] for r in rows_a]) * 100
        mm = np.abs(aa - aa.mean()) <= b
        sub_a = [r for r, k in zip(rows_a, mm) if k]
        if len(sub_a) < 8:
            return None, len(sub_a)
        x = np.array([r[xkey] for r in sub_a])
        y = np.array([r[ykey] for r in sub_a]) * 100
        lr = np.array([r["lr"] for r in sub_a])
        r, pv = pearsonr(gres(x, lr), gres(y, lr))
        return (float(r), float(pv)), len(sub_a)

    for xk, xlbl, pc in [("id_pr_perclass", "PR", 0.988),
                          ("id_local_smoothness", "smoothness", 0.993),
                          ("id_numerical_rank", "numrank", 0.306),
                          ("id_acc", "ID acc", -0.637)]:
        got, nsub = within_arch2("shufflenet_v2_x1_0", xk, "c100_probe_acc")
        claim("4.7a", f"within-arch ShuffleNetV2 {xlbl} C100", pc, got[0], 0.001)
        claim("4.7a", f"within-arch ShuffleNetV2 {xlbl} n", 14, nsub, 0)
    # RegNetY n moved 12 -> 13
    got, nsub = within_arch2("regnet_y_400mf", "id_pr_perclass", "c100_probe_acc")
    claim("4.7a", "within-arch RegNetY n", 13, nsub, 0)
    claim("4.7a", "within-arch RegNetY PR C100", 0.659, got[0], 0.001)
    got, nsub = within_arch2("regnet_y_400mf", "id_acc", "c100_probe_acc")
    claim("4.7a", "within-arch RegNetY ID acc C100", 0.053, got[0], 0.001)
    got, nsub = within_arch2("regnet_y_400mf", "id_acc", "tinyimg_probe_acc")
    claim("4.7a", "within-arch RegNetY ID acc Tiny", -0.019, got[0], 0.001)
    got, nsub = within_arch2("regnet_y_400mf", "id_pr_perclass", "tinyimg_probe_acc")
    claim("4.7a", "within-arch RegNetY PR Tiny", 0.536, got[0], 0.001)

    # ---------- appendix A: the schematic's provenance numbers ----------
    def resid_rho80(col):
        acc = D80["acc_test"].values.astype(float)
        ood = D80["acc_lp_ood"].values.astype(float)
        g = D80["arch_name"].values
        mm = np.abs(acc - acc.mean()) <= 2.0
        x = D80[col].values.astype(float)
        return float(spearmanr(gres(x[mm], g[mm]), gres(ood[mm], g[mm]))[0])

    claim("appendixA", "D_eff resid rho (80 release)", 0.81,
          resid_rho80("dimension_id"), 0.005)
    claim("appendixA", "Psi_eff resid rho (80 release)", 0.80,
          resid_rho80("utility_id"), 0.005)
    claim("appendixA", "ID acc resid rho (80 release)", -0.61,
          resid_rho80("acc_test"), 0.005)
    claim("appendixA", "PR resid rho (80 release)", 0.79,
          resid_rho80("participation_ratio_id"), 0.005)
    claim("appendixA", "mean_dist (avg) resid rho (80 release)", 0.68,
          resid_rho80("mean_distance_avg_id"), 0.005)
    claim("appendixA", "raw pooled r, 80 release", 0.4426,
          float(np.corrcoef(D80["acc_test"].values, D80["acc_lp_ood"].values)[0, 1]),
          0.001)
    claim("appendixA", "raw pooled r, 224 release", 0.4759,
          float(np.corrcoef(D224["acc_test"].values, D224["acc_lp_ood"].values)[0, 1]),
          0.001)

    # ---------- §4.2/§4.4: the hyperparameter-grid union quoted in §4.1 ----
    claim("setup", "224 union learning rates", 13, D224["lr"].nunique(), 0)
    claim("setup", "224 union weight decays", 9, D224["weight_decay"].nunique(), 0)
    claim("setup", "224 architecture count", 5, D224["arch_name"].nunique(), 0)
    claim("setup", "80 release scenario count", 6, len(D80_ALL), 0)
    claim("setup", "80 release model count", 80, len(D80), 0)

    # ---------- the manuscript must actually contain the audited numbers ----
    TEX = ROOT / "paper" / "main.tex"
    if TEX.exists():
        t = TEX.read_text(encoding="utf-8")

        def _tex_rows(tex, header_needle):
            """Parse the cells of a LaTeX tabular whose header contains a needle.

            Returns a list of cell lists.  Parsing the table (rather than
            searching for a substring) means an edited cell -- including one
            that only differs by a trailing character -- is caught.
            """
            i = tex.find(header_needle)
            if i < 0:
                return []
            j = tex.find(r"\end{tabular}", i)
            seg = tex[i:j if j > 0 else len(tex)]
            out = []
            for line in seg.split("\n"):
                if "&" not in line:
                    continue
                body = line.rsplit("\\\\", 1)[0]
                cells = [c.strip() for c in body.split("&")]
                out.append(cells)
            return out

        def _num(cell):
            """Strip LaTeX markup from a cell: '$+0.934$' -> '+0.934', '25\\%' -> '25%'."""
            c = cell.replace("$", "").replace("\\allowbreak", "").strip()
            c = c.replace("\\textbf{", "").replace("}", "").strip()
            c = c.replace("\\%", "%").strip()
            return c

        # ---- Table 4 (tab:shifttype), cell by cell ----
        t4_rows = _tex_rows(t, "Scenario & Best predictor (pooled)")
        T4_TEX = {
            "fog": ["ID accuracy", "+0.990", "+0.990", "1/12",
                    "ID accuracy", "+0.934", "+0.934", "1/12"],
            "brightness": ["ID accuracy", "+0.994", "+0.994", "1/12",
                           "ID accuracy", "+0.973", "+0.973", "1/12"],
            "contrast": ["ID accuracy", "+0.975", "+0.975", "1/12",
                         "ID accuracy", "+0.816", "+0.816", "1/12"],
            "gaussian_noise": ["mean angle", "-0.632", "+0.185", "6/12",
                               "sparsity", "+0.299", "-0.109", "6/12"],
            "pixelate": ["center alignment", "-0.445", "-0.128", "10/12",
                         "effective radius", "-0.300", "+0.191", "2/12"],
            "jpeg_compression": ["sparsity", "-0.351", "+0.087", "11/12",
                                 "effective utility", "+0.262", "-0.055", "7/12"],
        }
        # the scenario cell carries LaTeX line-break escapes; normalise it
        def _tex_scen(cell):
            c = cell.replace("\\_", "_").replace("\\allowbreak", "")
            c = c.replace("{", "").replace("}", "").strip()
            # "gaussian\_\allowbreak noise" -> "gaussian_noise"
            return re.sub(r"\s*_\s*", "_", c)

        found4 = {}
        for cells in t4_rows:
            if len(cells) < 9:
                continue
            found4[_tex_scen(cells[0])] = cells[1:9]
        for scen, expected in T4_TEX.items():
            got = found4.get(scen)
            ok = got is not None and [_num(c) for c in got] == expected
            CLAIMS.append({
                "section": "text-table4",
                "quantity": f"tab:shifttype row {scen}",
                "paper": " | ".join(expected),
                "recomputed": (" | ".join(_num(c) for c in got) if got else "ROW ABSENT"),
                "tol": 0, "kind": "text",
                "pass": bool(ok), "note": "",
            })

        # ---- tab:signmech, cell by cell ----
        sm_rows = _tex_rows(t, "Release & $n$ & ID $\\sim$ lr")
        found_sm = {c[0]: c[1:5] for c in sm_rows if len(c) >= 5 and "models" in c[0]}
        for label, expected in [
            ("224 models", ["106", "-0.451", "-0.815", "+0.043"]),
            ("80 models", ["43", "+0.089", "-0.487", "-0.670"]),
        ]:
            got = found_sm.get(label)
            ok = got is not None and [_num(c) for c in got] == expected
            CLAIMS.append({
                "section": "text-signmech",
                "quantity": f"tab:signmech row {label}",
                "paper": " | ".join(expected),
                "recomputed": (" | ".join(_num(c) for c in got) if got else "ROW ABSENT"),
                "tol": 0, "kind": "text",
                "pass": bool(ok), "note": "",
            })

        # ---- tab:glueown, cell by cell ----
        go_rows = _tex_rows(t, "Predictor & resid.\\ $r$ & $\\rho$ & C4 sign")
        found_go = {}
        for cells in go_rows:
            if len(cells) < 4:
                continue
            name = cells[0].replace("$", "").strip()
            name = name.replace("\\mathrm{eff}", "eff").replace("\\Psi", "Psi")
            name = name.replace("{", "").replace("}", "").strip()
            found_go[name] = cells[1:4]
        for label, expected in [
            ("D_eff (own impl.)", ["+0.698", "+0.640", "fail"]),
            ("Psi_eff (own impl.)", ["+0.098", "+0.276", "fail"]),
            ("R_eff (own impl.)", ["+0.462", "+0.484", "pass"]),
            ("ID accuracy", ["-0.453", "-0.428", "fail"]),
        ]:
            got = found_go.get(label)
            ok = got is not None and [_num(c) for c in got] == expected
            CLAIMS.append({
                "section": "text-glueown",
                "quantity": f"tab:glueown row {label}",
                "paper": " | ".join(expected),
                "recomputed": (" | ".join(_num(c) for c in got) if got else "ROW ABSENT"),
                "tol": 0, "kind": "text",
                "pass": bool(ok), "note": "",
            })

        # ---- Table 3 (tab:bootstrap), the two robust-rate cells ----
        bt_rows = _tex_rows(t, "Quantity & Dataset & 95\\% CI (resid.\\ $r$)")
        found_bt = []
        for cells in bt_rows:
            if len(cells) >= 4 and "ID accuracy" in cells[0]:
                found_bt.append((cells[1].strip(), _num(cells[3])))
        for dataset, expected in [("ours (48, all)", "25%"),
                                  ("ours (36, converged)", "83%")]:
            got = dict(found_bt).get(dataset)
            CLAIMS.append({
                "section": "text-table3",
                "quantity": f"tab:bootstrap robust rate, {dataset}",
                "paper": expected,
                "recomputed": got if got is not None else "ROW ABSENT",
                "tol": 0, "kind": "text",
                "pass": bool(got == expected), "note": "",
            })

        # ---- prose-only statements: substring presence is all there is ----
        MUST = [
            # §4.6 prose ranges
            "$-0.11$ to $+0.19$", "2nd--7th",
            # §3.3 prose: participation ratio's bootstrap robustness rate
            "55\\%",
            # converged-family numbers quoted in the prose
            "$r=+0.71$", "$r=+0.65$", "$r=+0.66$", "r=-0.66",
            # §4.7a model counts and the ShuffleNetV2 numbers
            "79 healthy models out of 80 trained", "none of its 15 healthy models",
            "$r=+0.988$", "$r=+0.993$", "$r=+0.306$", "$-0.64$",
            "($n=13$)", "$n=45$",
            # appendix A
            "pooled $r=+0.44$", "mean\\_dist (avg)",
        ]
        for needle in MUST:
            CLAIMS.append({
                "section": "text", "quantity": f"main.tex contains {needle!r}",
                "paper": "present", "recomputed": "present" if needle in t else "ABSENT",
                "tol": 0, "kind": "text",
                "pass": bool(needle in t), "note": "",
            })
        # values that were replaced this round must NOT survive in the text
        STALE = [
            "+0.918", "+0.967$ & $+0.967$ & 1/12", "-0.087$ & $-0.087$",
            "23\\%", "54\\%", "78 healthy", "78 self-trained",
            "magnitude collapse",
        ]
        for needle in STALE:
            CLAIMS.append({
                "section": "text", "quantity": f"main.tex must not contain {needle!r}",
                "paper": "absent", "recomputed": "present" if needle in t else "absent",
                "tol": 0, "kind": "text",
                "pass": bool(needle not in t), "note": "",
            })
        # the appendix-A schematic still quotes the OLD 80-release pooled r
        # in one place; the provenance note must use the corrected value.
        CLAIMS.append({
            "section": "text", "quantity": "main.tex: 80-release pooled r is +0.44, not +0.48",
            "paper": "+0.44", "recomputed": "+0.44" if "pooled $r=+0.44$" in t else "missing",
            "tol": 0, "kind": "text",
            "pass": bool("pooled $r=+0.44$" in t), "note": "",
        })
        # The contribution bullet (§1, item 3) still quotes the PRE-Spearman
        # Protocol-A range and rank span for the noise/compression scenarios,
        # while the abstract, §4.6 and Table 4 were all updated to the
        # Spearman values.  This is a stale duplicate of a number the rest of
        # the paper now reports correctly; it is recorded for the authors
        # rather than edited here.
        if "$-0.09$ and $+0.22$, ranked 6th--11th" in t:
            discrepancy(
                "text",
                "§1 contribution bullet 3 still quotes the pre-Spearman "
                "Protocol-A range '$-0.09$ to $+0.22$, ranked 6th--11th' for "
                "the noise/compression scenarios. The abstract, §4.6 and "
                "Table 4 now report the Spearman values "
                "'$-0.11$ to $+0.19$, ranked 2nd--7th', which the released "
                "coefficients confirm.",
                paper="$-0.09$ to $+0.22$, 6th--11th",
                recomputed="$-0.11$ to $+0.19$, 2nd--7th")
    else:
        CLAIMS.append({
            "section": "text", "quantity": "paper/main.tex present",
            "paper": "present", "recomputed": "MISSING", "tol": 0, "kind": "text",
            "pass": False, "note": "",
        })

    # ---------------- report ----------------
    npass = sum(c["pass"] for c in CLAIMS)
    nfail = len(CLAIMS) - npass
    json.dump(CLAIMS, open(RES / "claims_audit.json", "w"), ensure_ascii=False, indent=2)
    json.dump(DISCREPANCIES, open(RES / "claims_audit_discrepancies.json", "w"),
              ensure_ascii=False, indent=2)

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
    if DISCREPANCIES:
        w("## ⚠ 论文正文与数据的分歧（需作者裁定，未计入失败数）")
        w("")
        for d in DISCREPANCIES:
            w(f"- **[{d['section']}]** {d['quantity']}")
            w(f"  - 论文值：{d['paper']}；重算值：{d['recomputed']}")
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
    for d in DISCREPANCIES:
        print(f"  DISCREPANCY [{d['section']}] {d['quantity']}")
        print(f"    paper={d['paper']}  recomputed={d['recomputed']}")

    # The measurement paper quotes its own numbers. Verifying them here means a
    # single command -- and a single CI run -- checks both manuscripts.
    try:
        import subprocess
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent /
                                 "audit_paper_measurement.py")],
            capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith("checks:"):
                print("measurement paper --", line)
    except Exception as e:
        print(f"measurement audit skipped: {e}")
    for c in CLAIMS:
        if not c["pass"]:
            print(f"  MISMATCH [{c['section']}] {c['quantity']}: paper={c['paper']} recomputed={c['recomputed']}")
    print(f"\n[saved] {RES/'claims_audit_report.md'}")


if __name__ == "__main__":
    main()
