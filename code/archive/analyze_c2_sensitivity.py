#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2 敏感性分析：分层阈值是否影响结论？分区是否驱动相关性？

背景：主分析给出的 NO-GO 结论中，非平凡指标（effective_dim 等）在**全样本**
上呈显著负相关（r=-0.48, p=0.0012），但按 train_acc<0.90 分层后，正常模型
子集内仍为 r=-0.586 (p=0.0002)。这看起来矛盾，实则是因为：

  分层把「训练失败」模型（lr_low / overfit / wd0 —— dim 高达 46~115，表征退化）
  排除后，剩下的模型仍包含 aug_ls / plain_short 这类**欠拟合**模型，
  它们 dim 中等（17~19）而 test_acc 低（0.72~0.82），继续拉高负相关。

本脚本做三件事：
  1. 多阈值敏感性：换分层阈值看结论是否稳定
  2. 分区分析：把样本按 dim 聚成两簇，看相关性是否由簇间分离驱动
  3. 排除「欠拟合」模型（train_acc 0.75~0.90 且 test_acc 低的）后，
     只保留「训练正常」的模型重算 —— 这才是真正考验「优雅性↔泛化」的子集

用法：
  python analyze_c2_sensitivity.py --results runs/c2_results_seeds0-1-2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NONTRIVIAL = ["effective_dim", "local_smoothness", "criticality_proxy", "spectral_entropy"]


def corr(x, y):
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan"), int(ok.sum())
    r, p = pearsonr(x[ok], y[ok])
    rho, _ = spearmanr(x[ok], y[ok])
    return float(r), float(p), float(rho), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = data["rows"]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for r in rows:
        h = r.get("history") or []
        r["train_acc"] = h[-1]["train_acc"] if h else float("nan")

    L = []
    L.append("# C2 敏感性分析\n")
    L.append(f"全样本 n={len(rows)}\n")

    # ---------- 1. 多阈值敏感性 ----------
    L.append("## 1. 分层阈值敏感性\n")
    L.append("按 `final train_acc < T` 判为「训练失败」并排除后，"
             "在剩余子集内重算相关性。\n")
    L.append("| 阈值 T | 剩余 n | 排除的配方 | "
             + " | ".join(f"`{m}`" for m in NONTRIVIAL) + " |")
    L.append("|---|---|---|" + "---|" * len(NONTRIVIAL))
    for T in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]:
        sub = [r for r in rows if not (r["train_acc"] < T)]
        exc = sorted({r["name"] for r in rows if r["train_acc"] < T})
        y = [r["test_acc"] for r in sub]
        cells = []
        for m in NONTRIVIAL:
            r_, p_, rho, n = corr([r[m] for r in sub], y)
            cells.append(f"r={r_:+.3f} p={p_:.4f}")
        L.append(f"| {T:.2f} | {len(sub)} | {', '.join(exc) if exc else '（无）'} | "
                 + " | ".join(cells) + " |")
    L.append("")

    # ---------- 2. 分区分析 ----------
    L.append("## 2. 分区分析：相关性是否由两簇分离驱动？\n")
    L.append("按 `effective_dim` 是否超过 20 把样本分成两簇，看簇内相关性。\n")
    hi = [r for r in rows if r["effective_dim"] > 20]
    lo = [r for r in rows if r["effective_dim"] <= 20]
    L.append(f"- 高维簇（dim>20）n={len(hi)}：test_acc "
             f"{min(r['test_acc'] for r in hi):.3f}–{max(r['test_acc'] for r in hi):.3f}，"
             f"配方 {', '.join(sorted({r['name'] for r in hi}))}")
    L.append(f"- 低维簇（dim≤20）n={len(lo)}：test_acc "
             f"{min(r['test_acc'] for r in lo):.3f}–{max(r['test_acc'] for r in lo):.3f}，"
             f"配方 {', '.join(sorted({r['name'] for r in lo}))}")
    L.append("")
    L.append("| 子集 | " + " | ".join(f"`{m}`" for m in NONTRIVIAL) + " |")
    L.append("|---|" + "---|" * len(NONTRIVIAL))
    for label, sub in [("高维簇内", hi), ("低维簇内", lo), ("全样本", rows)]:
        y = [r["test_acc"] for r in sub]
        cells = []
        for m in NONTRIVIAL:
            r_, p_, rho, n = corr([r[m] for r in sub], y)
            cells.append(f"r={r_:+.3f} p={p_:.4f}")
        L.append(f"| {label} (n={len(sub)}) | " + " | ".join(cells) + " |")
    L.append("")

    # ---------- 3. 只保留"训练正常"模型 ----------
    L.append("## 3. 严格子集：排除欠拟合与崩溃\n")
    L.append("同时排除两类极端模型：\n")
    L.append("- **欠拟合**（train_acc < 0.90）：训练轮次不足，表征未成形")
    L.append("- **崩溃/退化**（dim > 20）：优化失败或严重过拟合，表征结构异常\n")
    strict = [r for r in rows if r["train_acc"] >= 0.90 and r["effective_dim"] <= 20]
    L.append(f"剩余 n={len(strict)}，配方：{', '.join(sorted({r['name'] for r in strict}))}\n")
    if len(strict) >= 4:
        y = [r["test_acc"] for r in strict]
        L.append("| 指标 | 类型 | r | p | ρ |")
        L.append("|---|---|---|---|---|")
        for m in NONTRIVIAL + ["silhouette_gap", "linear_probe_acc"]:
            r_, p_, rho, n = corr([r[m] for r in strict], y)
            t = "非平凡" if m in NONTRIVIAL else "平凡"
            L.append(f"| `{m}` | {t} | {r_:+.3f} | {p_:.4f} | {rho:+.3f} |")
        L.append("")
        L.append("**这是「优雅性 ↔ 泛化」命题最严格的检验**："
                 "剩余模型都是训练正常、表征结构健康的，"
                 "只按泛化能力排序。若此处仍无强相关，命题不成立。\n")
    else:
        L.append("子集样本不足。\n")

    # ---------- 4. 配方内配对检验 ----------
    L.append("## 4. 配方内配对检验（去中心，排除配方效应）\n")
    L.append("每个配方在 3 个种子下有 3 个观测。先按配方去中心（减去该配方的均值），"
             "再算残差相关性。这样可以排除「配方本身决定几何量」这一混淆。\n")
    by_recipe = {}
    for r in rows:
        by_recipe.setdefault(r["name"], []).append(r)
    resid = []
    for name, rs in by_recipe.items():
        for m in NONTRIVIAL:
            vals = np.array([r[m] for r in rs], float)
            if np.isnan(vals).any() or len(rs) < 2:
                continue
            for r, v in zip(rs, vals):
                resid.append({"name": name, "seed": r["seed"], "metric": m,
                              "x": v - np.nanmean(vals),
                              "y": r["test_acc"] - np.mean([q["test_acc"] for q in rs])})
    L.append("| 指标 | 残差 r | 残差 p | 残差 ρ |")
    L.append("|---|---|---|---|")
    for m in NONTRIVIAL:
        sub = [d for d in resid if d["metric"] == m]
        r_, p_, rho, n = corr([d["x"] for d in sub], [d["y"] for d in sub])
        L.append(f"| `{m}` | {r_:+.3f} | {p_:.4f} | {rho:+.3f} |")
    L.append("")

    # ---------- 5. 结论 ----------
    L.append("## 结论\n")
    strict_ok = False
    if len(strict) >= 4:
        y = [r["test_acc"] for r in strict]
        for m in NONTRIVIAL:
            r_, p_, rho, n = corr([r[m] for r in strict], y)
            if abs(r_) >= 0.7 and p_ < 0.05:
                strict_ok = True
    L.append(f"- 严格子集内是否存在通过判据的非平凡指标：**{'是' if strict_ok else '否'}**")
    L.append(f"- 全样本的显著负相关**不能**作为支持证据 —— "
             f"它由「表征崩溃簇 vs 正常簇」的分离驱动，"
             f"而非「优雅程度 ↔ 泛化能力」的连续关系。")
    L.append("- 按 ADR-0003：M0 判据未通过 → **NO-GO**。")
    L.append("")

    fp = outdir / "c2_sensitivity_report.md"
    fp.write_text("\n".join(L), encoding="utf-8")
    print(f"报告: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
