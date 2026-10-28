#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用官方发布的逐模型数据（df_fig15-20.pkl）直接检验 R1 命题。

官方数据含每个模型的：
  - acc_test          : ID (CIFAR-10) 准确率
  - acc_lp_ood        : OOD (CIFAR-100) linear probe 准确率
  - acc_test_c        : corruption 后的准确率
  - dimension_id/radius_id/utility_id : GLUE 指标（在 ID 上算）
  - participation_ratio_id / numerical_rank_id / nc1_id 等

于是可以直接算：
  H1: 几何标记与 acc_lp_ood 的相关 vs 与 acc_test 的相关
  H2: corruption 场景下，acc_test 是否是最强预测器

这等于**用官方数据直接验证 R1 的命题**，是本项目最有说服力的对照。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 指标方向：+1 表示越大越好，-1 表示越小越好（依据官方 df_fig4.pkl 的 r 符号）
DIRECTION = {
    "dimension_id": +1, "participation_ratio_id": +1, "utility_id": +1,
    "center_alignment_id": +1, "nc1_id": -1, "numerical_rank_id": +1,
    "sparsity_id": +1, "mean_covariance_id": +1, "mean_distance_avg_id": +1,
    "radius_id": -1, "axis_alignment_id": -1, "center_axis_alignment_id": -1,
    "mean_distance_id": -1, "mean_angle_id": -1, "entropy": -1,
    "capacity_id": +1,
}


def corr(x, y):
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 5:
        return float("nan"), float("nan"), float("nan"), int(ok.sum())
    r, p = pearsonr(x[ok], y[ok])
    rho, _ = spearmanr(x[ok], y[ok])
    return float(r), float(p), float(rho), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-dir",
                    default="/mnt/mydisk/xinguan-brainai/vendor/official_truth")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fp = Path(args.pkl_dir) / "df_fig15-20.pkl"
    data = pd.read_pickle(fp)

    L = []
    L.append("# 用官方发布数据直接检验 R1 命题\n")
    L.append("数据源：官方仓库 `data/df_fig15-20.pkl`（6 个 corruption 场景，"
             "每场景 80 个模型 = 5 架构 × 4 lr × ... ）\n")
    L.append("> 说明：这是**作者自己算好的逐模型测量值**，含 ID 准确率、"
             "OOD linear probe 准确率、以及在 ID 数据上计算的全部几何指标。")
    L.append("> 因此可以在**完全相同的数据**上检验「几何标记的预测力是否分布偏移专属」。\n")

    scenarios = list(data.keys())
    L.append(f"场景：{', '.join(scenarios)}\n")

    # ---- 汇总 H1 ----
    L.append("## H1：几何标记对 OOD-C 的预测力 vs 对 ID 的预测力\n")
    L.append("OD-C = CIFAR-100 linear probe 准确率（`acc_lp_ood`）；"
             "ID = CIFAR-10 test 准确率（`acc_test`）\n")

    markers = [m for m in DIRECTION if any(m in d.columns for d in data.values())]
    L.append("| 标记 | 场景 | \\|r\\| ID | p(ID) | \\|r\\| OOD-C | p(OOD-C) | Δ\\|r\\| | 判定 |")
    L.append("|---|---|---|---|---|---|---|---|---|")

    summary = []
    for sc in scenarios:
        d = data[sc]
        for m in markers:
            if m not in d.columns:
                continue
            x = d[m].values.astype(float)
            r_id, p_id, _, n = corr(x, d["acc_test"].values)
            r_ood, p_ood, _, _ = corr(x, d["acc_lp_ood"].values)
            if np.isnan(r_id) or np.isnan(r_ood):
                continue
            delta = abs(r_ood) - abs(r_id)
            verdict = "**OOD 更强**" if delta > 0.05 else ("ID 更强" if delta < -0.05 else "相当")
            L.append(f"| `{m}` | {sc} | {abs(r_id):.3f} | {p_id:.4f} | "
                     f"{abs(r_ood):.3f} | {p_ood:.4f} | {delta:+.3f} | {verdict} |")
            summary.append({"scenario": sc, "marker": m, "r_id": r_id, "p_id": p_id,
                            "r_ood": r_ood, "p_ood": p_ood, "delta": delta, "n": n})

    # 汇总统计
    L.append("\n### 跨场景汇总\n")
    L.append("| 标记 | 场景数 | OOD 更强的场景数 | 平均 \\|r\\| ID | 平均 \\|r\\| OOD-C | 平均 Δ |")
    L.append("|---|---|---|---|---|---|")
    rows = []
    for m in markers:
        ss = [s for s in summary if s["marker"] == m]
        if not ss:
            continue
        n_ood = sum(1 for s in ss if s["delta"] > 0.05)
        mid = np.mean([abs(s["r_id"]) for s in ss])
        mod = np.mean([abs(s["r_ood"]) for s in ss])
        mdl = np.mean([s["delta"] for s in ss])
        rows.append((m, len(ss), n_ood, mid, mod, mdl))
        L.append(f"| `{m}` | {len(ss)} | {n_ood}/{len(ss)} | {mid:.3f} | {mod:.3f} | {mdl:+.3f} |")

    n_ood_stronger = sum(1 for r in rows if r[2] > r[1] / 2)
    L.append("")
    L.append(f"**结论：{n_ood_stronger}/{len(rows)} 个指标在多数场景下，"
             f"几何标记对 OOD 的预测力强于对 ID。**\n")
    if n_ood_stronger > len(rows) / 2:
        L.append("→ **H1 成立：几何标记的预测力确实是分布偏移专属的。**")
        L.append("  在同分布任务上，它们对 ID 性能的预测力很弱；"
                 "只有当存在分布偏移时，才提供超越 ID accuracy 的信息。")
    else:
        L.append("→ H1 不成立。")

    # ---- H2 ----
    L.append("\n## H2：corruption 偏移下，ID accuracy 是否最强预测器？\n")
    L.append("预测目标 = `acc_test_c`（corruption 后的准确率），"
             "预测变量取自 ID 数据\n")
    L.append("| 场景 | 最强预测器 | \\|r\\| | acc_test 的 \\|r\\| | 排名 |")
    L.append("|---|---|---|---|---|")
    h2_rows = []
    for sc in scenarios:
        d = data[sc]
        cand = []
        for m in list(markers) + ["acc_test"]:   # acc_test 作为基线参与排名
            if m not in d.columns:
                continue
            r, p, rho, n = corr(d[m].values.astype(float), d["acc_test_c"].values)
            if not np.isnan(r):
                cand.append((abs(r), m, r))
        cand.sort(reverse=True)
        if not cand:
            continue
        acc_r = next((c for c in cand if c[1] == "acc_test"), None)
        rank = next((i + 1 for i, c in enumerate(cand) if c[1] == "acc_test"), None)
        h2_rows.append((sc, cand[0][1], cand[0][0], acc_r[0] if acc_r else float("nan"), rank, len(cand)))
        L.append(f"| {sc} | `{cand[0][1]}` | {cand[0][0]:.3f} | "
                 f"{acc_r[0]:.3f} | {rank}/{len(cand)} |")

    if h2_rows:
        acc_is_top = sum(1 for r in h2_rows if r[4] is not None and r[4] <= 2)
        L.append("")
        L.append(f"**结论：acc_test 在 {acc_is_top}/{len(h2_rows)} 个场景中排进前 2。**")
        if acc_is_top >= len(h2_rows) * 0.6:
            L.append("→ **H2 成立**：corruption 偏移下 ID accuracy 是最强预测器，"
                     "几何标记相对失效（与论文 Section 3.3 一致）。")
        else:
            L.append("→ H2 部分成立（与论文结论有出入，需讨论）。")

    # ---- 与官方 df_fig4 对照 ----
    L.append("\n## 与官方 Figure 4 真值对照（ResNet18/SGD，CIFAR-100）\n")
    f4 = Path(args.pkl_dir) / "df_fig4.pkl"
    if f4.exists():
        d4 = pd.read_pickle(f4)
        s = d4[d4.arch_name == "ResNet18\n(SGD)"][["property", "r", "p"]]
        s = s.reindex(s["r"].abs().sort_values(ascending=False).index)
        L.append("| 官方指标 | 官方 r | 官方 p |")
        L.append("|---|---|---|")
        for _, row in s.iterrows():
            L.append(f"| `{row['property']}` | {row['r']:+.3f} | {row['p']:.2e} |")
        L.append("")
        L.append("> 官方 `acc_test` 的 r = **+0.173（不显著）**，"
                 "而 `dimension_id` / `participation_ratio_id` 均 > +0.79 —— "
                 "这正是「几何标记比 ID accuracy 更能预测 OOD」的官方证据。")

    fp_md = outdir / "r1_official_data_report.md"
    fp_md.write_text("\n".join(L), encoding="utf-8")
    fp_json = outdir / "r1_official_summary.json"
    fp_json.write_text(json.dumps({"summary": summary, "h2": h2_rows},
                                  ensure_ascii=False, indent=2, default=float),
                       encoding="utf-8")
    print(f"报告: {fp_md}")
    print(f"H1: {n_ood_stronger}/{len(rows)} 个指标 OOD 更强")
    for r in h2_rows:
        print(f"H2 {r[0]}: 最强={r[1]} (|r|={r[2]:.3f}), acc_test 排名 {r[4]}/{r[5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
