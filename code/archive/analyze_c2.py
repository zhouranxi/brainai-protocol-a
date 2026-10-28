#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2 结果分析：按 M0 判据出结论 + 生成图表。

判据来源：
  docs/adr/0003-M0-止损线.md      —— 相关性阈值与止损条件
  docs/adr/0002-几何量须排除信息量混淆.md —— 非平凡判据

输出：
  results/c2_report.md      结论报告
  results/c2_scatter.png    相关性散点图
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NONTRIVIAL = ["effective_dim", "local_smoothness", "criticality_proxy", "spectral_entropy"]
TRIVIAL = ["silhouette_gap", "linear_probe_acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="c2_results_*.json")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    from scipy.stats import pearsonr, spearmanr

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = data["rows"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 按种子分组算相关（判据要求 3 个种子下结论一致）----
    seeds = sorted({r["seed"] for r in rows})
    metrics = NONTRIVIAL + TRIVIAL

    per_seed = {}
    for s in seeds:
        rs = [r for r in rows if r["seed"] == s]
        y = np.array([r["test_acc"] for r in rs], dtype=float)
        per_seed[s] = {}
        for m in metrics:
            x = np.array([r[m] for r in rs], dtype=float)
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 3:
                per_seed[s][m] = {"r": float("nan"), "p": float("nan"),
                                  "rho": float("nan"), "n": int(ok.sum())}
                continue
            r, p = pearsonr(x[ok], y[ok])
            rho, _ = spearmanr(x[ok], y[ok])
            per_seed[s][m] = {"r": float(r), "p": float(p), "rho": float(rho),
                              "n": int(ok.sum())}

    # ---- 合并全部（pooled）----
    y_all = np.array([r["test_acc"] for r in rows], dtype=float)
    pooled = {}
    for m in metrics:
        x = np.array([r[m] for r in rows], dtype=float)
        ok = ~(np.isnan(x) | np.isnan(y_all))
        if ok.sum() < 3:
            pooled[m] = {"r": float("nan"), "p": float("nan"), "rho": float("nan")}
            continue
        r, p = pearsonr(x[ok], y_all[ok])
        rho, _ = spearmanr(x[ok], y_all[ok])
        pooled[m] = {"r": float(r), "p": float(p), "rho": float(rho)}

    # ---- M0 判据 ----
    def passes(m):
        """该指标在**所有种子**下都满足：|r|>=0.7, p<0.05, |rho|<1.0"""
        for s in seeds:
            st = per_seed[s].get(m)
            if not st or np.isnan(st["r"]) or np.isnan(st["p"]):
                return False
            if abs(st["r"]) < 0.7 or st["p"] >= 0.05:
                return False
            if not np.isnan(st["rho"]) and abs(st["rho"]) >= 1.0:
                return False
        return True

    go = [m for m in NONTRIVIAL if passes(m)]

    decision = "GO" if go else "NO-GO"

    # ---- 写报告 ----
    lines = []
    lines.append("# C2 实验报告：表征几何指标能否预测模型质量\n")
    lines.append(f"- 数据集：CIFAR-10（50000 train / 10000 test）")
    lines.append(f"- 架构：resnet18（torchvision，从头训练）")
    lines.append(f"- 模型组数：{len(rows)}（{len(seeds)} 种子 × {len(rows)//len(seeds)} 配方）")
    lines.append(f"- 非平凡指标（不使用标签）：{', '.join(NONTRIVIAL)}")
    lines.append(f"- 平凡指标（使用测试标签，与 test_acc 同源）：{', '.join(TRIVIAL)}\n")
    lines.append("## M0 结论\n")
    lines.append(f"### **{decision}**\n")
    if go:
        lines.append(f"通过判据的非平凡指标：**{', '.join(go)}**\n")
    else:
        lines.append("**无指标通过 M0 判据。**\n")
    lines.append("判据（ADR-0003 + ADR-0002）：\n")
    lines.append("1. 属非平凡指标（不使用测试标签）")
    lines.append("2. |Pearson r| ≥ 0.7 且 p < 0.05")
    lines.append("3. |Spearman ρ| < 1.0（不能只是性能的单调重标度）")
    lines.append("4. 全部种子下结论一致\n")

    lines.append("## 逐种子相关性\n")
    lines.append("| 指标 | 类型 | " +
                 " | ".join(f"seed {s}" for s in seeds) + " | pooled r | pooled ρ |")
    lines.append("|---|---|" + "---|" * (len(seeds) + 2))
    for m in metrics:
        t = "非平凡" if m in NONTRIVIAL else "平凡"
        cells = []
        for s in seeds:
            st = per_seed[s][m]
            cells.append(f"r={st['r']:+.3f} p={st['p']:.3f}")
        lines.append(f"| `{m}` | {t} | " + " | ".join(cells) +
                     f" | {pooled[m]['r']:+.3f} | {pooled[m]['rho']:+.3f} |")
    lines.append("")

    lines.append("## 逐模型明细（seed %d）\n" % seeds[0])
    lines.append("| 配方 | test_acc | " + " | ".join(f"`{m}`" for m in metrics) + " |")
    lines.append("|---|---|" + "---|" * len(metrics))
    for r in [x for x in rows if x["seed"] == seeds[0]]:
        cells = [f"{r[m]:.4f}" if not np.isnan(r[m]) else "nan" for m in metrics]
        lines.append(f"| `{r['name']}` | {r['test_acc']:.4f} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        colors = {s: c for s, c in zip(seeds, ["#378ADD", "#1D9E75", "#D4537E"])}
        for ax, m in zip(axes.flat, metrics):
            for s in seeds:
                rs = [r for r in rows if r["seed"] == s]
                x = [r[m] for r in rs]
                yy = [r["test_acc"] for r in rs]
                ax.scatter(x, yy, s=42, alpha=0.75, label=f"seed {s}", color=colors[s])
            st = pooled[m]
            if not np.isnan(st["r"]):
                ax.set_title(f"{m}\nr={st['r']:+.3f}, ρ={st['rho']:+.3f}"
                             + ("  [平凡]" if m in TRIVIAL else ""), fontsize=10)
            ax.set_xlabel(m, fontsize=8)
            ax.set_ylabel("test accuracy", fontsize=8)
            ax.grid(alpha=0.25, linestyle=":")
            ax.legend(fontsize=7)
        fig.suptitle("C2: 几何指标 vs 泛化性能（非平凡指标不使用标签）", fontsize=13)
        fig.tight_layout()
        fp_png = outdir / "c2_scatter.png"
        fig.savefig(fp_png, dpi=140)
        lines.append(f"![散点图]({fp_png.name})\n")
    except Exception as e:
        lines.append(f"（图表生成失败：{e}）\n")

    lines.append("## 判读\n")
    best_nontrivial = max(NONTRIVIAL,
                          key=lambda m: abs(pooled[m]["r"])
                          if not np.isnan(pooled[m]["r"]) else -1)
    br = pooled[best_nontrivial]["r"]
    lines.append(f"- 最强非平凡指标：`{best_nontrivial}`（pooled r = {br:+.3f}）")
    for m in TRIVIAL:
        lines.append(f"- 平凡指标 `{m}`：r = {pooled[m]['r']:+.3f}"
                     + ("（与 test_acc 同源，同义反复，不计入判据）" if abs(pooled[m]["r"]) > 0.7 else ""))
    lines.append("")

    fp_md = outdir / "c2_report.md"
    fp_md.write_text("\n".join(lines), encoding="utf-8")

    # 保存结构化结论
    fp_json = outdir / "c2_verdict.json"
    fp_json.write_text(json.dumps({
        "decision": decision, "go_candidates": go,
        "per_seed": per_seed, "pooled": pooled,
        "nontrivial": NONTRIVIAL, "trivial": TRIVIAL,
        "n_models": len(rows), "seeds": seeds,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"M0 决策: {decision}")
    print(f"通过的非平凡指标: {go}")
    print(f"报告: {fp_md}")
    print(f"结论: {fp_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
