#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用官方数据检验更精确的命题 H3。

H1（粗版）不成立：13 个指标里只有 4 个"整体偏 OOD"。但汇总表显示结论
**完全取决于指标类别**：

  - GLUE task-relevant 标记（dimension / center_alignment / utility）
      → 对 ID 几乎无预测力（|r| 0.10–0.34），对 OOD 强（0.49–0.66）
  - 通用/task-agnostic 几何量（participation_ratio / mean_distance_avg / capacity）
      → 对 ID 有预测力（0.50–0.75），对 OOD 弱（0.00–0.31）

H3（精确版）：
  **task-relevant 几何标记与 task-agnostic 几何量在"预测哪一类性能"上分工不同；
   越是与下游线性分类任务绑定的几何量，其预测力越集中在 OOD 侧。**

同时检验一个更尖锐的版本 H4：
  **在 ID 准确率相近的模型子集内（ID acc 差距 < 2%），OOD 差异是否更大？
   若成立，则几何标记提供的是 ID accuracy 无法提供的信息。**

这才是"几何标记 vs ID accuracy"的正确对照口径 —— 论文本身也是这么做的
（"models with distinct hyperparameters can exhibit similar ID accuracy
while their OOD performance can differ drastically"）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TASK_RELEVANT = ["dimension_id", "center_alignment_id", "utility_id",
                 "capacity_id", "axis_alignment_id", "center_axis_alignment_id",
                 "radius_id"]
TASK_AGNOSTIC = ["participation_ratio_id", "mean_distance_avg_id",
                 "mean_distance_id", "mean_angle_id", "mean_covariance_id",
                 "sparsity_id", "nc1_id"]
ALL = TASK_RELEVANT + TASK_AGNOSTIC


def corr(x, y):
    from scipy.stats import pearsonr
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 5:
        return float("nan"), float("nan"), int(ok.sum())
    r, p = pearsonr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-dir",
                    default="/mnt/mydisk/xinguan-brainai/vendor/official_truth")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--id-band", type=float, default=2.0,
                    help="ID acc 相近的带宽（百分点；官方 acc 单位是百分数）")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(Path(args.pkl_dir) / "df_fig15-20.pkl")

    L = []
    L.append("# R1 精确命题检验（基于官方发布的逐模型数据）\n")
    L.append("数据：官方 `df_fig15-20.pkl`，6 个 corruption 场景 × 80 个模型。"
             "每模型含 ID acc、OOD linear probe acc、corruption acc 与全部几何标记。\n")

    # ---------- H3 ----------
    L.append("## H3：task-relevant 与 task-agnostic 几何量的预测力分工\n")
    L.append("**H3**：越是与下游线性分类任务绑定的几何标记（GLUE 系），"
             "其预测力越集中在 OOD 侧；越是通用几何量，越偏向 ID 侧。\n")

    rows = []
    for sc, d in data.items():
        for m in ALL:
            if m not in d.columns:
                continue
            x = d[m].values.astype(float)
            r_id, p_id, _ = corr(x, d["acc_test"].values)
            r_ood, p_ood, _ = corr(x, d["acc_lp_ood"].values)
            if np.isnan(r_id) or np.isnan(r_ood):
                continue
            rows.append({"sc": sc, "m": m,
                         "kind": "task-relevant" if m in TASK_RELEVANT else "task-agnostic",
                         "rid": abs(r_id), "roid": abs(r_ood),
                         "delta": abs(r_ood) - abs(r_id),
                         "p_id": p_id, "p_ood": p_ood})
    df = pd.DataFrame(rows)

    L.append("| 指标类别 | 指标 | 平均 \\|r\\| ID | 平均 \\|r\\| OOD-C | 平均 Δ(OOD−ID) | Δ>0 场景占比 |")
    L.append("|---|---|---|---|---|---|")
    for m in ALL:
        s = df[df.m == m]
        if len(s) == 0:
            continue
        kind = "task-relevant" if m in TASK_RELEVANT else "task-agnostic"
        pos = (s.delta > 0.05).sum()
        L.append(f"| {kind} | `{m}` | {s.rid.mean():.3f} | {s.roid.mean():.3f} | "
                 f"{s.delta.mean():+.3f} | {pos}/{len(s)} |")

    tr = df[df.kind == "task-relevant"]
    ta = df[df.kind == "task-agnostic"]
    L.append("")
    L.append("**分组结论**\n")
    L.append(f"- task-relevant 标记（{len(tr)} 个指标-场景组合）："
             f"平均 Δ = **{tr.delta.mean():+.3f}**，"
             f"其中 {int((tr.delta > 0).sum())}/{len(tr)} 个为正")
    L.append(f"- task-agnostic 几何量（{len(ta)} 个）："
             f"平均 Δ = **{ta.delta.mean():+.3f}**，"
             f"其中 {int((ta.delta > 0).sum())}/{len(ta)} 个为正")
    L.append("")
    h3_ok = tr.delta.mean() > ta.delta.mean()
    L.append(f"→ **H3 {'成立' if h3_ok else '不成立'}**："
             f"task-relevant 标记的平均 Δ 比 task-agnostic 高 "
             f"{tr.delta.mean() - ta.delta.mean():+.3f}")

    # ---------- H4 ----------
    L.append(f"\n## H4：ID 准确率相近的模型子集内，OOD 差异有多大？\n")
    L.append(f"取 ID acc 带宽 ±{args.id_band:.1f} 个百分点的模型子集，"
             f"比较该子集内 OOD probe 准确率的跨度 vs 全体的跨度。\n")
    L.append("| 场景 | 子集 n | 子集内 OOD acc 跨度 | 全体 OOD acc 跨度 | 相对放大 | 子集内 ID acc 跨度 |")
    L.append("|---|---|---|---|---|---|")
    h4_rows = []
    for sc, d in data.items():
        acc = d["acc_test"].values
        ood = d["acc_lp_ood"].values
        m = (acc >= acc.mean() - args.id_band) & (acc <= acc.mean() + args.id_band)
        if m.sum() < 5:
            continue
        span_sub = float(np.nanmax(ood[m]) - np.nanmin(ood[m]))
        span_all = float(np.nanmax(ood) - np.nanmin(ood))
        id_span = float(np.nanmax(acc[m]) - np.nanmin(acc[m]))
        ratio = span_sub / span_all if span_all > 0 else float("nan")
        h4_rows.append({"sc": sc, "n": int(m.sum()), "span_sub": span_sub,
                        "span_all": span_all, "ratio": ratio, "id_span": id_span})
        L.append(f"| {sc} | {int(m.sum())} | {span_sub:.4f} | {span_all:.4f} | "
                 f"**{ratio:.1%}** | {id_span:.4f} |")
    if h4_rows:
        avg_ratio = np.mean([r["ratio"] for r in h4_rows])
        L.append("")
        L.append(f"→ 子集内 OOD 跨度平均保留全体的 **{avg_ratio:.1%}**。")
        if avg_ratio > 0.5:
            L.append("→ **H4 成立**：在 ID 准确率几乎相同的模型之间，"
                     "OOD 性能仍有很大差异 —— 这正是几何标记能提供增量信息的空间。")
        else:
            L.append("→ H4 不成立：ID acc 相同处 OOD 也相同，无增量空间。")

    # ---------- H5: 在 ID 相近子集内，几何标记是否比 acc_test 更能预测 OOD ----------
    L.append("\n## H5：在 ID 准确率相近的子集内，谁更能预测 OOD？\n")
    L.append("这是最严格的对照：只看 ID acc 相近的模型，"
             "几何标记 vs ID acc 本身，谁与 OOD 相关更强。\n")
    L.append("| 场景 | 子集 n | 指标 | \\|r\\| OOD (子集内) | p |")
    L.append("|---|---|---|---|---|")
    h5 = []
    for sc, d in data.items():
        acc = d["acc_test"].values
        m = (acc >= acc.mean() - args.id_band) & (acc <= acc.mean() + args.id_band)
        if m.sum() < 8:
            continue
        sub = d[m]
        for mk in ["acc_test"] + [x for x in ALL if x in d.columns]:
            r, p, n = corr(sub[mk].values.astype(float), sub["acc_lp_ood"].values)
            if np.isnan(r):
                continue
            h5.append({"sc": sc, "m": mk, "r": abs(r), "p": p})
    for h in h5:
        L.append(f"| {h['sc']} | — | `{h['m']}` | {h['r']:.3f} | {h['p']:.4f} |")

    if h5:
        h5df = pd.DataFrame(h5)
        L.append("\n**各标记在子集内对 OOD 的平均 |r|**\n")
        L.append("| 标记 | 平均 \\|r\\| | 显著(p<0.05)比例 |")
        L.append("|---|---|---|")
        for mk in ["acc_test"] + [x for x in ALL if x in h5df.m.unique()]:
            s = h5df[h5df.m == mk]
            sig = (s.p < 0.05).sum()
            L.append(f"| `{mk}` | {s.r.mean():.3f} | {sig}/{len(s)} |")
        best = h5df.groupby("m").r.mean().sort_values(ascending=False)
        L.append("")
        L.append(f"→ 子集内最强预测器：**`{best.index[0]}`**（平均 |r| = {best.iloc[0]:.3f}）")
        if "acc_test" in best.index:
            L.append(f"  vs `acc_test`（{best['acc_test']:.3f}）")

    fp_md = outdir / "r1_h3_h4_h5_report.md"
    fp_md.write_text("\n".join(L), encoding="utf-8")
    (outdir / "r1_h3_h4_h5.json").write_text(
        json.dumps({"h3": rows, "h4": h4_rows, "h5": h5},
                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"报告: {fp_md}")
    print(f"H3: task-relevant Δ={tr.delta.mean():+.3f} vs task-agnostic Δ={ta.delta.mean():+.3f}"
          f"  → {'成立' if h3_ok else '不成立'}")
    if h4_rows:
        print(f"H4: 子集内 OOD 跨度保留 {np.mean([r['ratio'] for r in h4_rows]):.1%}")
    if h5:
        b = pd.DataFrame(h5).groupby("m").r.mean().sort_values(ascending=False)
        print(f"H5: 子集内最强 = {b.index[0]} ({b.iloc[0]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
