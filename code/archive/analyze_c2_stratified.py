#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2 分层分析：区分「训练失败」与「正常训练但泛化差」两类模型。

动机（见 results/c2_report.md 判读）：
  seed0 的初步结果显示，非平凡指标（effective_dim / local_smoothness）与性能
  呈**负相关**，且呈现双峰分离。追查后发现峰值来自「训练失败」模型
  （lr_low / overfit / wd0 —— 这些模型 train_acc 正常但表征退化），
  与「正常训练但泛化较差」是**两类不同机制**。

  如果不区分，两者混在一起会：
    - 低估正常区间内的指标有效性
    - 也会高估指标对「训练是否崩溃」的判别力（这本身是另一回事）

本脚本做两件事：
  1. 用 train_acc 判定「训练是否失败」，分层重算相关性
  2. 报告「训练失败检测」与「正常模型排序」两个独立结论

用法：
  python analyze_c2_stratified.py --results runs/c2_results_seeds0-1-2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NONTRIVIAL = ["effective_dim", "local_smoothness", "criticality_proxy", "spectral_entropy"]
TRIVIAL = ["silhouette_gap", "linear_probe_acc"]


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

    # ---- 分层：训练失败 vs 正常训练 ----
    # 判据：最终 train_acc < 0.90 视为训练失败（欠拟合/优化失败）
    THRESH = 0.90
    for r in rows:
        hist = r.get("history") or []
        r["final_train_acc"] = hist[-1]["train_acc"] if hist else float("nan")
        r["failed"] = bool(
            (not np.isnan(r["final_train_acc"])) and r["final_train_acc"] < THRESH
        )

    failed = [r for r in rows if r["failed"]]
    ok_models = [r for r in rows if not r["failed"]]

    lines = []
    lines.append("# C2 分层分析：训练失败 vs 正常训练\n")
    lines.append(f"- 全部模型：{len(rows)}；训练失败（final train_acc < {THRESH}）："
                 f"{len(failed)}；正常训练：{len(ok_models)}")
    names_failed = sorted({r["name"] for r in failed})
    lines.append(f"- 失败配方：{', '.join(names_failed) if names_failed else '（无）'}\n")

    def block(subset, title):
        lines.append(f"## {title}（n={len(subset)}）\n")
        if len(subset) < 4:
            lines.append("样本不足，跳过相关性计算。\n")
            return
        y = [r["test_acc"] for r in subset]
        lines.append("| 指标 | 类型 | r | p | ρ | n |")
        lines.append("|---|---|---|---|---|---|")
        for m in NONTRIVIAL + TRIVIAL:
            x = [r[m] for r in subset]
            r_, p_, rho, n = corr(x, y)
            t = "非平凡" if m in NONTRIVIAL else "平凡"
            lines.append(f"| `{m}` | {t} | {r_:+.3f} | {p_:.4f} | {rho:+.3f} | {n} |")
        lines.append("")

    block(ok_models, "A. 正常训练模型（排除训练失败）")
    block(failed, "B. 训练失败模型（表征退化）")

    # ---- 独立结论 1：能否检测训练失败 ----
    lines.append("## 独立结论 1：指标能否检测「训练失败」\n")
    lines.append("这是一个**独立的、且更有实用价值**的问题："
                 "如果几何指标能识别出训练崩溃的模型，它就是一个可靠的"
                 "**训练健康度监控**工具（早停/调参辅助）。\n")
    lines.append("| 指标 | AUC（区分 失败 vs 正常） |")
    lines.append("|---|---|")
    for m in NONTRIVIAL:
        x = [r[m] for r in rows]
        y = [1 if r["failed"] else 0 for r in rows]
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, x)
        except Exception:
            auc = float("nan")
        lines.append(f"| `{m}` | {auc:.3f} |")
    lines.append("")

    # ---- 独立结论 2：正常模型内的排序能力 ----
    lines.append("## 独立结论 2：正常模型区间内的排序能力\n")
    lines.append("这是原 M0 判据的**收紧版**：只在训练正常的模型中比较，"
                 "看几何指标能否区分「泛化好」与「泛化差」。"
                 "这才是「优雅性 ↔ 泛化」命题的真正战场。\n")
    if len(ok_models) >= 4:
        y = [r["test_acc"] for r in ok_models]
        lines.append("| 指标 | 类型 | r | p | ρ |")
        lines.append("|---|---|---|---|---|")
        for m in NONTRIVIAL + TRIVIAL:
            x = [r[m] for r in ok_models]
            r_, p_, rho, n = corr(x, y)
            t = "非平凡" if m in NONTRIVIAL else "平凡"
            lines.append(f"| `{m}` | {t} | {r_:+.3f} | {p_:.4f} | {rho:+.3f} |")
        lines.append("")

    # ---- 明细 ----
    lines.append("## 逐模型明细\n")
    lines.append("| 配方 | 种子 | final_train_acc | 状态 | test_acc | "
                 + " | ".join(f"`{m}`" for m in NONTRIVIAL) + " |")
    lines.append("|---|---|---|---|---|" + "---|" * len(NONTRIVIAL))
    for r in rows:
        cells = [f"{r[m]:.4f}" if not np.isnan(r[m]) else "nan" for m in NONTRIVIAL]
        lines.append(f"| `{r['name']}` | {r['seed']} | {r['final_train_acc']:.4f} | "
                     f"{'失败' if r['failed'] else '正常'} | {r['test_acc']:.4f} | "
                     + " | ".join(cells) + " |")
    lines.append("")

    fp = outdir / "c2_stratified_report.md"
    fp.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告: {fp}")
    print(f"失败 {len(failed)} / 正常 {len(ok_models)} / 共 {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
