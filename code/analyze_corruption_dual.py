"""Shift-type decomposition under two instruments (adversarial review Q6, 2026-10-27).

The draft reported only pooled Spearman rho for the six corruption scenarios while
presenting the analysis inside the "Protocol A" framework. This script computes both:
  - pooled: correlation over all 80 models in the scenario
  - Protocol A: ID-matched at +/-2pp around the population mean, then architecture
    residualization (C1 + C2), then correlation

Statistics (2026-11-03 fix): the draft computed **Pearson** via ``np.corrcoef`` but
labelled the numbers as Spearman rho.  Both statistics are now computed and reported
side by side, since the paper's C3 condition requires reporting Pearson r and
Spearman rho separately.

JSON layout (per scenario):
  - ``pooled_pearson`` / ``resid_pearson``   : Pearson r  (the statistic actually
    computed in the draft)
  - ``pooled_spearman`` / ``resid_spearman`` : Spearman rho (the statistic the paper
    claims to report)
  - ``pooled`` / ``resid``                   : LEGACY aliases of the **Pearson**
    values, kept for backward compatibility with existing consumers.
  - ``best_pooled`` / ``best_resid`` / ``id_rank_pooled`` / ``id_rank_resid``
    : LEGACY fields, still computed on **Pearson** (unchanged from the draft).
  - ``best_pooled_spearman`` / ``best_resid_spearman`` /
    ``id_rank_pooled_spearman`` / ``id_rank_resid_spearman`` : the same quantities
    computed on **Spearman**, which is the paper's primary instrument.

Outputs: results/corruption_dual_protocol.json, results/corruption_dual_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
BAND = 2.0

MARKERS = [
    "acc_test", "participation_ratio_id", "dimension_id", "utility_id", "radius_id",
    "center_alignment_id", "axis_alignment_id", "capacity_id", "numerical_rank_id",
    "nc1_id", "mean_distance_id", "mean_angle_id", "mean_covariance_id", "sparsity_id",
]
NICE = {
    "acc_test": "ID accuracy",
    "participation_ratio_id": "participation ratio",
    "dimension_id": "effective dimension",
    "utility_id": "effective utility",
    "radius_id": "effective radius",
    "center_alignment_id": "center alignment",
    "axis_alignment_id": "axis alignment",
    "capacity_id": "capacity",
    "numerical_rank_id": "numerical rank",
    "nc1_id": "NC1",
    "mean_distance_id": "mean distance",
    "mean_angle_id": "mean angle",
    "mean_covariance_id": "mean covariance",
    "sparsity_id": "sparsity",
}


def group_residual(y, groups):
    y = np.asarray(y, float)
    out = y.copy()
    for g in set(groups):
        m = np.array([gi == g for gi in groups])
        out[m] -= y[m].mean()
    return out


def pearson(x, y):
    """Pearson r (the statistic the draft actually computed)."""
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    """Spearman rho (rank correlation), the statistic the paper claims to report."""
    return float(spearmanr(x, y)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    if getattr(args, "outdir", None):
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
    if args.outdir:
        globals()["RES"] = Path(args.outdir)
    if args.data:
        src = Path(args.data)
    else:
        src = None
        for cand in (ROOT / "data" / "official", ROOT / "results", ROOT):
            for name in ("df_fig15-20.pkl", "official_df_fig15-20.pkl"):
                if (cand / name).exists():
                    src = cand
                    break
            if src is not None:
                break
        if src is None:
            raise FileNotFoundError("df_fig15-20.pkl not found")
    _pkl = None
    for _name in ("df_fig15-20.pkl", "official_df_fig15-20.pkl"):
        if (src / _name).exists():
            _pkl = src / _name
            break
    if _pkl is None:
        raise FileNotFoundError(f"df_fig15-20.pkl not found under {src}")
    d = pd.read_pickle(_pkl)
    out = {"band": BAND, "scenarios": {}}

    L = []
    w = L.append
    w("# 偏移类型分解：双口径对照（pooled vs Protocol A）× 双统计量（Pearson / Spearman）")
    w("")
    w(f"生成：2026-11-03 · band=±{BAND:.0f}pp · 数据：official_df_fig15-20.pkl（80 模型 × 6 场景）")
    w("")
    w("背景：原稿只报 pooled Spearman ρ，但整节放在 Protocol A 框架下叙述。")
    w("本表并列两个口径：pooled = 全 80 模型相关；Protocol A = ID 匹配（±2pp）+ 架构残差化后相关。")
    w("")
    w("统计量说明：2026-11-03 修正 —— 原实现用 `np.corrcoef`（**Pearson**）却把数值标成 ρ（Spearman）。")
    w("下表同时给出 **Pearson r** 与 **Spearman ρ**（`r_pooled`/`r_Protocol A` 为 Pearson，"
      "`ρ_pooled`/`ρ_Protocol A` 为 Spearman）。论文主口径为 Spearman。")
    w("")

    for scen, df in d.items():
        acc = df["acc_test"].to_numpy()
        y = df["acc_test_c"].to_numpy()
        arch = df["arch_name"].to_numpy()
        keep = np.abs(acc - acc.mean()) <= BAND
        rec = {"pooled_pearson": {}, "resid_pearson": {},
               "pooled_spearman": {}, "resid_spearman": {},
               "n_matched": int(keep.sum())}
        for mk in MARKERS:
            if mk not in df.columns:
                continue
            x = df[mk].to_numpy()
            xr = group_residual(x[keep], arch[keep])
            yr = group_residual(y[keep], arch[keep])
            rec["pooled_pearson"][mk] = pearson(x, y)
            rec["resid_pearson"][mk] = pearson(xr, yr)
            rec["pooled_spearman"][mk] = spearman(x, y)
            rec["resid_spearman"][mk] = spearman(xr, yr)
        # --- backward-compatible aliases (LEGACY = Pearson, as in the original draft) ---
        rec["pooled"] = dict(rec["pooled_pearson"])
        rec["resid"] = dict(rec["resid_pearson"])
        for tag, src in (("pooled", "pooled_pearson"), ("resid", "resid_pearson")):
            order = sorted(rec[src], key=lambda k: -abs(rec[src][k]))
            rec[f"best_{tag}"] = order[0]
            rec[f"id_rank_{tag}"] = order.index("acc_test") + 1
        # --- primary (Spearman) versions, used by the paper ---
        for tag, src in (("pooled", "pooled_spearman"), ("resid", "resid_spearman")):
            order = sorted(rec[src], key=lambda k: -abs(rec[src][k]))
            rec[f"best_{tag}_spearman"] = order[0]
            rec[f"id_rank_{tag}_spearman"] = order.index("acc_test") + 1
        out["scenarios"][scen] = rec

        w(f"### {scen}（n={rec['n_matched']} after matching）")
        w("")
        w("| 指标 | r_pooled (Pearson) | r_Protocol A (Pearson) | "
          "ρ_pooled (Spearman) | ρ_Protocol A (Spearman) |")
        w("|---|---|---|---|---|")
        for mk in MARKERS:
            if mk not in rec["pooled_pearson"]:
                continue
            w(f"| {NICE.get(mk, mk)} | {rec['pooled_pearson'][mk]:+.3f} | "
              f"{rec['resid_pearson'][mk]:+.3f} | {rec['pooled_spearman'][mk]:+.3f} | "
              f"{rec['resid_spearman'][mk]:+.3f} |")
        w("")
        w(f"最佳（pooled, Spearman）= {NICE.get(rec['best_pooled_spearman'])}；"
          f"最佳（Protocol A, Spearman）= {NICE.get(rec['best_resid_spearman'])}；"
          f"ID acc 秩（Spearman）：pooled {rec['id_rank_pooled_spearman']}/{len(rec['pooled_spearman'])}，"
          f"Protocol A {rec['id_rank_resid_spearman']}/{len(rec['resid_spearman'])}")
        w("")
        w(f"（参考，Pearson 口径：最佳 pooled = {NICE.get(rec['best_pooled'])}，"
          f"最佳 Protocol A = {NICE.get(rec['best_resid'])}；"
          f"ID acc 秩 pooled {rec['id_rank_pooled']}/{len(rec['pooled_pearson'])}，"
          f"Protocol A {rec['id_rank_resid']}/{len(rec['resid_pearson'])}）")
        w("")

    json.dump(out, open(RES / "corruption_dual_protocol.json", "w"),
              ensure_ascii=False, indent=2)

    w("## 结论（Spearman 口径为主）")
    w("")
    agree, disagree = [], []
    for scen, rec in out["scenarios"].items():
        same_best = (rec["best_pooled_spearman"] == rec["best_resid_spearman"])
        agree.append(f"{scen}：最佳标记 pooled={NICE.get(rec['best_pooled_spearman'])}，"
                     f"Protocol A={NICE.get(rec['best_resid_spearman'])}；"
                     f"ID acc 秩 pooled {rec['id_rank_pooled_spearman']}/"
                     f"{len(rec['pooled_spearman'])}，"
                     f"Protocol A {rec['id_rank_resid_spearman']}/"
                     f"{len(rec['resid_spearman'])}")
        if not same_best:
            disagree.append(scen)
    for line in agree:
        w(f"- {line}")
    w("")
    if disagree:
        w(f"- Spearman 下 pooled 与 Protocol A 的最佳标记不一致的场景："
          f"{', '.join(disagree)}。")
    else:
        w("- Spearman 下两口径的最佳标记在全部六个场景一致。")
    w("")
    w("### Pearson vs Spearman 差异（统计量口径本身的影响）")
    w("")
    w("| 场景 | 最佳标记 pooled（Pearson→Spearman） | 最佳标记 Protocol A（Pearson→Spearman） | "
      "ID acc 秩 pooled | ID acc 秩 Protocol A |")
    w("|---|---|---|---|---|")
    for scen, rec in out["scenarios"].items():
        npool = len(rec["pooled_pearson"])
        w(f"| {scen} | {NICE.get(rec['best_pooled'])} → {NICE.get(rec['best_pooled_spearman'])} | "
          f"{NICE.get(rec['best_resid'])} → {NICE.get(rec['best_resid_spearman'])} | "
          f"{rec['id_rank_pooled']}/{npool} → {rec['id_rank_pooled_spearman']}/{npool} | "
          f"{rec['id_rank_resid']}/{npool} → {rec['id_rank_resid_spearman']}/{npool} |")
    w("")
    w("- → 偏移类型结论必须在两个口径下分别陈述；只报 pooled 是在测混杂，不是测现象。")
    w("- → 同时必须标明统计量：Pearson 与 Spearman 在 noise/compression 场景可给出不同的最佳标记。")
    w("")

    (RES / "corruption_dual_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[-12:]))
    print(f"\n[saved] {RES/'corruption_dual_report.md'}")
    print(f"[saved] {RES/'corruption_dual_protocol.json'}")


if __name__ == "__main__":
    main()
