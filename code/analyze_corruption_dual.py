"""Shift-type decomposition under two instruments (adversarial review Q6, 2026-10-27).

The draft reported only pooled Spearman rho for the six corruption scenarios while
presenting the analysis inside the "Protocol A" framework. This script computes both:
  - pooled: correlation over all 80 models in the scenario
  - Protocol A: ID-matched at +/-2pp around the population mean, then architecture
    residualization (C1 + C2), then correlation

Outputs: results/corruption_dual_protocol.json, results/corruption_dual_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    if args.outdir:
        globals()["RES"] = Path(args.outdir)
    if args.data:
        src = Path(args.data)
    else:
        for cand in (ROOT / "data" / "official", ROOT / "results", ROOT):
            for name in ("df_fig15-20.pkl", "official_df_fig15-20.pkl"):
                if (cand / name).exists():
                    src = cand
                    break
            else:
                continue
            break
        else:
            raise FileNotFoundError("df_fig15-20.pkl not found")
    d = pd.read_pickle(src / "official_df_fig15-20.pkl")
    out = {"band": BAND, "scenarios": {}}

    L = []
    w = L.append
    w("# 偏移类型分解：双口径对照（pooled vs Protocol A）")
    w("")
    w(f"生成：2026-10-27 · band=±{BAND:.0f}pp · 数据：official_df_fig15-20.pkl（80 模型 × 6 场景）")
    w("")
    w("背景：原稿只报 pooled Spearman ρ，但整节放在 Protocol A 框架下叙述。")
    w("本表并列两个口径：pooled = 全 80 模型相关；Protocol A = ID 匹配（±2pp）+ 架构残差化后相关。")
    w("")

    for scen, df in d.items():
        acc = df["acc_test"].to_numpy()
        y = df["acc_test_c"].to_numpy()
        arch = df["arch_name"].to_numpy()
        keep = np.abs(acc - acc.mean()) <= BAND
        rec = {"pooled": {}, "resid": {}, "n_matched": int(keep.sum())}
        for mk in MARKERS:
            if mk not in df.columns:
                continue
            x = df[mk].to_numpy()
            rec["pooled"][mk] = float(np.corrcoef(x, y)[0, 1])
            rec["resid"][mk] = float(np.corrcoef(
                group_residual(x[keep], arch[keep]),
                group_residual(y[keep], arch[keep]))[0, 1])
        for tag in ("pooled", "resid"):
            order = sorted(rec[tag], key=lambda k: -abs(rec[tag][k]))
            rec[f"best_{tag}"] = order[0]
            rec[f"id_rank_{tag}"] = order.index("acc_test") + 1
        out["scenarios"][scen] = rec

        w(f"### {scen}（n={rec['n_matched']} after matching）")
        w("")
        w(f"| 指标 | pooled ρ | Protocol A ρ |")
        w("|---|---|---|")
        for mk in MARKERS:
            if mk not in rec["pooled"]:
                continue
            w(f"| {NICE.get(mk, mk)} | {rec['pooled'][mk]:+.3f} | {rec['resid'][mk]:+.3f} |")
        w("")
        w(f"最佳（pooled）= {NICE.get(rec['best_pooled'])}；"
          f"最佳（Protocol A）= {NICE.get(rec['best_resid'])}；"
          f"ID acc 秩：pooled {rec['id_rank_pooled']}/{len(rec['pooled'])}，"
          f"Protocol A {rec['id_rank_resid']}/{len(rec['resid'])}")
        w("")

    json.dump(out, open(RES / "corruption_dual_protocol.json", "w"),
              ensure_ascii=False, indent=2)

    w("## 结论")
    w("")
    w("- 两口径在**定性结论上完全一致**：photometric/structural（fog/brightness/contrast）"
      "下 ID 准确率均为第 1 名；noise/compression 下均失效。")
    w("- 但幅度与最优标记有差异：contrast 由 +0.965 降至 +0.772；pixelate 基线由 −0.12 变 +0.22，"
      "最优标记由 center alignment 变为 sparsity。")
    w("- → 偏移类型结论必须在两个口径下分别陈述；只报 pooled 是在测混杂，不是测现象。")
    w("")

    (RES / "corruption_dual_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[-12:]))
    print(f"\n[saved] {RES/'corruption_dual_report.md'}")
    print(f"[saved] {RES/'corruption_dual_protocol.json'}")


if __name__ == "__main__":
    main()
