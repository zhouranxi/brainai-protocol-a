#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R1 分析：几何标记的预测力是分布偏移专属的吗？

对每个几何标记 m，分别计算它与三类性能的相关性：
  - ID   : CIFAR-10 test accuracy（同分布）
  - OOD-C: CIFAR-100 linear probe accuracy（类别级偏移，论文主场景）
  - OOD-K: CIFAR-10C accuracy（corruption 偏移，论文对照组）

命题（论文 Section 3.3 + Discussion）：
  H1 (OOD-C): 几何标记对 OOD-C 的预测力 **显著强于** 对 ID 的预测力
  H2 (OOD-K): 在 corruption 偏移下，几何标记失效，而 ID accuracy 最强

输出：results/r1_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MARKERS = [
    "id_pr_perclass", "id_pr_global", "id_nc1", "id_numerical_rank",
    "id_effective_dim_pca", "id_local_smoothness", "id_class_separability",
]
TARGETS = [("id_acc", "ID (CIFAR-10)"),
           ("ood_c100_probe_acc", "OOD-C (CIFAR-100 probe)"),
           ("ood_c10c_acc", "OOD-K (CIFAR-10C)")]
# 平凡标记（用了标签）单独标注
TRIVIAL = {"id_class_separability"}


def corr(x, y):
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 4:
        return float("nan"), float("nan"), float("nan"), int(ok.sum())
    r, p = pearsonr(x[ok], y[ok])
    rho, _ = spearmanr(x[ok], y[ok])
    return float(r), float(p), float(rho), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--min-n", type=int, default=6)
    args = ap.parse_args()

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    L = []
    L.append("# R1 分析：几何标记的预测力是否分布偏移专属\n")
    L.append(f"- 模型数 n = {len(rows)}")
    L.append(f"- 评估协议（对齐 ICLR 2026 论文 Section 3.1）：")
    L.append("  - ID：模型原分类头在 CIFAR-10 test 上的准确率")
    L.append("  - OOD-C：**linear probe**（冻结特征，在 CIFAR-100 train 上训新头，test 评估）")
    L.append("  - OOD-K：CIFAR-10C（severity 5），标签空间不变 → 直接用原分类头")
    L.append("- 几何标记**只从 ID 数据**计算（预后性 prognostic 的定义）\n")

    # 逐模型明细
    L.append("## 逐模型明细\n")
    head = "| model | ID acc | OOD-C probe | OOD-K | PR(per-class) | PR(global) | NC1 | num.rank |"
    L.append(head)
    L.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda z: -z.get("id_acc", 0)):
        L.append(f"| `{r['ckpt']}` | {r['id_acc']:.4f} | "
                 f"{r.get('ood_c100_probe_acc', float('nan')):.4f} | "
                 f"{r.get('ood_c10c_acc', float('nan')):.4f} | "
                 f"{r['id_pr_perclass']:.2f} | {r['id_pr_global']:.2f} | "
                 f"{r['id_nc1']:.4f} | {r['id_numerical_rank']:.0f} |")
    L.append("")

    # 相关性矩阵
    L.append("## 相关性矩阵（几何标记 × 预测目标）\n")
    L.append("| 标记 | 类型 | " + " | ".join(f"{t[1]}" for t in TARGETS) + " | |Δr| OOD-C − ID |")
    L.append("|---|---|" + "---|" * (len(TARGETS) + 1))
    table = {}
    for m in MARKERS:
        if m not in rows[0]:
            continue
        x = [r[m] for r in rows]
        cells = []
        rs = {}
        for tkey, tname in TARGETS:
            if tkey not in rows[0]:
                cells.append("—")
                continue
            r_, p_, rho, n = corr(x, [r[tkey] for r in rows])
            rs[tkey] = (r_, p_, rho, n)
            star = "***" if p_ < 0.01 else ("*" if p_ < 0.05 else "")
            cells.append(f"{r_:+.3f}{star}")
        table[m] = rs
        t = "平凡" if m in TRIVIAL else "非平凡"
        if "ood_c100_probe_acc" in rs and "id_acc" in rs:
            d = abs(rs["ood_c100_probe_acc"][0]) - abs(rs["id_acc"][0])
            dcell = f"{d:+.3f}"
        else:
            dcell = "—"
        L.append(f"| `{m}` | {t} | " + " | ".join(cells) + f" | {dcell} |")
    L.append("")
    L.append("显著性：`*` p<0.05，`***` p<0.01\n")

    # H1 检验
    L.append("## H1 检验：几何标记对 OOD-C 的预测力是否强于对 ID？\n")
    h1 = []
    for m in MARKERS:
        if m not in table or m in TRIVIAL:
            continue
        rs = table[m]
        if "ood_c100_probe_acc" not in rs or "id_acc" not in rs:
            continue
        r_ood, p_ood, _, _ = rs["ood_c100_probe_acc"]
        r_id, p_id, _, _ = rs["id_acc"]
        if np.isnan(r_ood) or np.isnan(r_id):
            continue
        better = abs(r_ood) > abs(r_id)
        h1.append((m, r_id, p_id, r_ood, p_ood, better))
        L.append(f"- `{m}`: |r|_ID = {abs(r_id):.3f} (p={p_id:.4f})  vs  "
                 f"|r|_OOD-C = {abs(r_ood):.3f} (p={p_ood:.4f})  → "
                 f"{'**OOD-C 更强**' if better else 'ID 更强'}")
    L.append("")
    if h1:
        n_ood_stronger = sum(1 for x in h1 if x[5])
        L.append(f"**结论：{n_ood_stronger}/{len(h1)} 个非平凡标记对 OOD-C 的预测力强于 ID。**")
        L.append("")
        if n_ood_stronger > len(h1) / 2:
            L.append("→ 支持 H1：**几何标记的预测力是分布偏移专属的**。")
        else:
            L.append("→ **不支持 H1**：几何标记对 ID 的预测力并未系统性弱于 OOD-C，"
                     "说明 Phase 0 的「几何量不能预测 ID 性能」结论需要重新审视。")
    L.append("")

    # H2 检验
    L.append("## H2 检验：corruption 偏移下是否 ID accuracy 最强？\n")
    if "ood_c10c_acc" in rows[0]:
        cand = []
        for m in MARKERS:
            if m not in table:
                continue
            rs = table[m]
            if "ood_c10c_acc" not in rs or "id_acc" not in rs:
                continue
            cand.append((abs(rs["ood_c10c_acc"][0]), m))
        cand.sort(reverse=True)
        if cand:
            L.append("按 |r| 排序（对 OOD-K 的预测力）：\n")
            for v, m in cand:
                tag = "平凡" if m in TRIVIAL else ""
                L.append(f"- `{m}`: |r| = {v:.3f} {tag}")
            L.append("")
            L.append(f"最强：{cand[0][1]}。")
            if cand[0][1] == "id_acc" or "id_acc" in [m for _, m in cand[:2]]:
                L.append("→ **支持 H2**：corruption 偏移下 ID accuracy 是最强预测器，"
                         "几何标记失效（与论文一致）。")
            else:
                L.append("→ 未完全支持 H2。")
    L.append("")

    # 预测准确率对比（论文用「选中 vs 未选中」口径）
    L.append("## 预测准确率（论文 §4 口径：几何标记 vs ID accuracy 选模型）\n")
    if "ood_c100_probe_acc" in rows[0] and len(rows) >= 6:
        y_ood = np.array([r["ood_c100_probe_acc"] for r in rows])
        y_id = np.array([r["id_acc"] for r in rows])
        L.append(f"- 按 ID accuracy 选最好的模型 → 其 OOD-C 排名中位数：")
        L.append("")
        L.append("| 预测器 | 选中模型的 OOD-C acc | OOD-C 排名 (1=最好) | 命中率 |")
        L.append("|---|---|---|---|")
        # ID accuracy 基准
        best_by_id = int(np.argmax(y_id))
        L.append(f"| ID accuracy | {y_ood[best_by_id]:.4f} | "
                 f"{int((y_ood > y_ood[best_by_id]).sum())+1}/{len(rows)} | — |")
        for m in MARKERS:
            if m not in rows[0] or m in TRIVIAL:
                continue
            x = np.array([r[m] for r in rows])
            if np.isnan(x).any():
                continue
            # 论文规则：Deff / Ψeff 越大越好；这里按方向自适应
            best = int(np.argmax(x)) if "pr" in m or "rank" in m or "dim" in m else int(np.argmin(x))
            rank = int((y_ood > y_ood[best]).sum()) + 1
            hit = "✓ 选中 OOD 最好" if rank == 1 else ""
            L.append(f"| `{m}` | {y_ood[best]:.4f} | {rank}/{len(rows)} | {hit} |")
        L.append("")

    fp = outdir / "r1_report.md"
    fp.write_text("\n".join(L), encoding="utf-8")
    fp2 = outdir / "r1_analysis.json"
    fp2.write_text(json.dumps({"n": len(rows), "table": table}, ensure_ascii=False,
                              indent=2, default=float), encoding="utf-8")
    print(f"报告: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
