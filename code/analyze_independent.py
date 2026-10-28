#!/usr/bin/env python3
"""独立验证：在自建 48 组模型上复现"ID 受控 + 架构去混淆"协议。

对应 ADR-0011 第一层——不依赖 GLUE 私有代码，用已验证与官方数值一致的公开指标。

关键设计：
  - 自建模型：resnet18 + CIFAR-10，4 lr × 4 wd × 3 seed = 48 组，200 epoch
  - ID 目标：CIFAR-10 测试准确率
  - OOD 目标：CIFAR-100 linear probe（与官方协议一致：冻结特征，OOD 上训新头）
  - 偏移类型：CIFAR-10C corruption（分类别，与官方 T5 对齐）
  - 指标：participation_ratio（官方数值已对齐 ✓）、numerical_rank、NC1、
          mean_distance/angle/covariance/sparsity、local_smoothness、effective_dim_pca

输出：
  V1  全样本相关性（ID vs OOD）
  V2  ID 受控子集内：残差化 Pearson + Spearman
  V3  逐超参族分解 + 逐架构（本实验单架构，故改为"逐 lr 组"与"留一 lr 组"）
  V4  带宽敏感性
  V5  corruption 场景细分
  V6  与官方数据的一致性对照（同一协议、同一指标族，两套独立数据）
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

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
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def safe_corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r, p = pearsonr(x, y)
    rho, prho = spearmanr(x, y)
    return float(r), float(p), float(rho), float(prho)


def group_residual(v, groups):
    """去除分组主效应：对分组哑变量 OLS 回归取残差。"""
    v = np.asarray(v, float)
    uniq = sorted(set(groups))
    if len(uniq) < 2:
        return v - v.mean()
    D = np.column_stack([(np.asarray(groups) == g).astype(float) for g in uniq[1:]])
    A = np.column_stack([np.ones(len(v)), D])
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "data" / "self_trained" / "ind_validation_results.json"))
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--band", type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.results) as f:
        data = json.load(f)
    rows = data["rows"]

    L = []
    def w(s=""):
        L.append(s)
        print(s, flush=True)

    w("# 独立验证报告：自建 48 组模型上的受控协议复现")
    w("")
    w(f"模型：{len(rows)} 组 resnet18 / CIFAR-10，"
      f"{len({r['lr'] for r in rows})} lr × {len({r['wd'] for r in rows})} wd × "
      f"{len({r['seed'] for r in rows})} seed")
    w("指标：公开可算几何量（participation_ratio / numerical_rank / NC1 / "
      "mean_* / sparsity / local_smoothness / effective_dim_pca）")
    w("OOD：CIFAR-100 linear probe（冻结特征，OOD 上训新头，与官方协议一致）")
    w("")
    w("> **本层的目的**：验证「ID 受控 + 架构/超参去混淆」这个**协议**在"
      "完全独立训练的数据上是否依然成立，而不依赖于官方发布的测量值。")
    w("")

    id_acc = np.array([r["id_acc"] for r in rows], float)
    ood_acc = np.array([r.get("c100_probe_acc", np.nan) for r in rows], float)
    lr_groups = np.array([r["lr"] for r in rows])
    wd_groups = np.array([r["wd"] for r in rows])

    MARKERS = [k for k in rows[0].keys()
               if k.startswith("id_") and k not in ("id_acc",)]
    MARKERS.sort()

    # ---------- V1 全样本 ----------
    w("## V1 全样本：各公开几何量的预测力")
    w("")
    w("| 指标 | r(ID) | rho(ID) | r(OOD) | rho(OOD) |")
    w("|---|---|---|---|---|")
    v1 = {}
    for mk in MARKERS:
        x = np.array([r[mk] for r in rows], float)
        rid, pid, rhoi, _ = safe_corr(x, id_acc)
        rod, pod, rhoo, _ = safe_corr(x, ood_acc)
        v1[mk] = {"r_id": rid, "rho_id": rhoi, "r_ood": rod, "rho_ood": rhoo}
        w(f"| `{mk}` | {rid:+.3f} | {rhoi:+.3f} | {rod:+.3f} | {rhoo:+.3f} |")
    w("")

    # ---------- V2 ID 受控 ----------
    w(f"## V2 ID 受控子集（|acc − mean| ≤ {args.band:.0f} 个百分点）")
    w("")
    # 带宽单位 = 个百分点；id_acc 为 0-1 比例 → 换算为百分数比较
    id_pct = id_acc * 100.0
    m = (id_pct >= id_pct.mean() - args.band) & (id_pct <= id_pct.mean() + args.band)
    idx = np.where(m)[0]
    sub = [rows[i] for i in idx]
    w(f"子集 n = {len(sub)} / {len(rows)}")
    if len(sub) < 8:
        w("\n> ⚠️ 子集样本不足，无法做受控检验。请增大 `--band`。")
        with open(os.path.join(args.outdir, "ind_validation_report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(L))
        return
    s_id = np.array([r["id_acc"] for r in sub], float)
    s_ood = np.array([r["c100_probe_acc"] for r in sub], float)
    s_lr = np.array([r["lr"] for r in sub])
    s_wd = np.array([r["wd"] for r in sub])
    w(f"子集内 ID 跨度 = {(s_id.max()-s_id.min())*100:.2f} 个百分点；"
      f"OOD 跨度 = {(s_ood.max()-s_ood.min())*100:.2f} 个百分点"
      f"（全体 {(np.nanmax(ood_acc)-np.nanmin(ood_acc))*100:.2f}）")
    w("")
    ood_res_lr = group_residual(s_ood, s_lr)
    ood_res_both = group_residual(s_ood, [f"{a}_{b}" for a, b in zip(s_lr, s_wd)])
    w("残差化分别对 **lr 族**与 **lr×wd 组合**做（自建实验只有单一架构，"
      "故以超参族替代架构作为混杂源）。")
    w("")
    w("| 指标 | 原始 r | 残差化(lr) r | p | 残差化(lr×wd) r | Spearman rho | 稳健? |")
    w("|---|---|---|---|---|---|---|")
    v2 = {}
    for mk in MARKERS:
        x = np.array([r[mk] for r in sub], float)
        raw_r, _, _, _ = safe_corr(x, s_ood)
        r1, p1, _, _ = safe_corr(group_residual(x, s_lr), ood_res_lr)
        r2, p2, _, _ = safe_corr(group_residual(x, [f"{a}_{b}" for a, b in zip(s_lr, s_wd)]), ood_res_both)
        rho, prho = safe_corr(x, s_ood)[2:]
        robust = (np.sign(r1) == np.sign(rho)) and abs(rho) >= 0.5
        v2[mk] = {"raw_r": raw_r, "resid_lr_r": r1, "resid_lr_p": p1,
                  "resid_lrwdr": r2, "rho": rho, "robust": bool(robust)}
        w(f"| `{mk}` | {raw_r:+.3f} | {r1:+.3f} | {p1:.4f} | {r2:+.3f} | {rho:+.3f} | "
          f"{'✓' if robust else '✗'} |")
    w("")
    # 基线
    raw_r, _, rho_b, _ = safe_corr(s_id, s_ood)
    r1, p1, _, _ = safe_corr(group_residual(s_id, s_lr), ood_res_lr)
    w(f"**基线 `id_acc`（ID 准确率）**：原始 r={raw_r:+.3f} → 残差化 r={r1:+.3f} (p={p1:.4f})，"
      f"Spearman rho={rho_b:+.3f}")
    v2["id_acc"] = {"raw_r": raw_r, "resid_lr_r": r1, "resid_lr_p": p1, "rho": rho_b}
    w("")

    # ---------- V3 逐 lr 族 + 留一 ----------
    w("## V3 逐超参族分解（符号一致性检验）")
    w("")
    w("| 指标 | " + " | ".join(str(g) for g in sorted(set(s_lr))) + " | 一致? |")
    w("|---" * (len(set(s_lr)) + 2) + "|")
    for mk in MARKERS + ["id_acc"]:
        cells, signs = [], []
        for g in sorted(set(s_lr)):
            sel = s_lr == g
            if sel.sum() < 4:
                cells.append("n<4")
                continue
            xx = np.array([r[mk] for r in sub])[sel]
            yy = s_ood[sel]
            rr, _, _, _ = safe_corr(xx, yy)
            cells.append(f"{rr:+.2f}")
            if not np.isnan(rr):
                signs.append(np.sign(rr))
        w(f"| `{mk}` | " + " | ".join(cells) + f" | {'✓' if len(set(signs))==1 else '✗'} |")
    w("")
    w("**留一 lr 族（用其余族拟合效应，套到被留出的族上）**")
    w("")
    w("| 指标 | " + " | ".join(str(g) for g in sorted(set(s_lr))) + " |")
    w("|---" * len(set(s_lr)) + "|")
    for mk in ["participation_ratio_id", "numerical_rank_id", "nc1_id",
               "local_smoothness_id", "id_acc"]:
        if mk not in MARKERS + ["id_acc"]:
            continue
        cells = []
        for held in sorted(set(s_lr)):
            tr = s_lr != held
            te = s_lr == held
            if tr.sum() < 6 or te.sum() < 3:
                cells.append("n/a")
                continue
            # 单架构下，被留出的 lr 族在训练集中没有出现，哑变量无法外推
            # → 该族只能做「组内相关」，不能做留一外推
            try:
                X = np.column_stack([(s_lr[tr] == g).astype(float) for g in sorted(set(s_lr))[1:]])
                A = np.column_stack([np.ones(tr.sum()), X])
                xx = np.array([r[mk] for r in sub])[tr]
                yy = s_ood[tr]
                B = np.linalg.lstsq(A, np.column_stack([xx, yy]), rcond=None)[0]
                Xt = np.column_stack([(s_lr[te] == g).astype(float) for g in sorted(set(s_lr))[1:]])
                At = np.column_stack([np.ones(te.sum()), Xt])
                pred = At @ B[:, 0]
                rr, _, _, _ = safe_corr(pred, s_ood[te])
                cells.append(f"{rr:+.2f}")
            except Exception:
                cells.append("n/a")
        w(f"| `{mk}` | " + " | ".join(cells) + " |")
    w("")

    # ---------- V4 带宽 ----------
    w("## V4 带宽敏感性")
    w("")
    bands = [1.0, 2.0, 3.0, 5.0, 8.0]
    w("| 指标 | " + " | ".join(f"b={b:.0f}pp" for b in bands) + " |")
    w("|---" * (len(bands) + 1) + "|")
    for mk in MARKERS + ["id_acc"]:
        cells = []
        for b in bands:
            mm = (id_pct >= id_pct.mean() - b) & (id_pct <= id_pct.mean() + b)
            if mm.sum() < 8:
                cells.append("n<8")
                continue
            ss = [rows[i] for i in np.where(mm)[0]]
            gg = np.array([r["lr"] for r in ss])
            xx = np.array([r[mk] for r in ss], float)
            yy = np.array([r["c100_probe_acc"] for r in ss], float)
            try:
                rr, _, _, _ = safe_corr(group_residual(xx, gg), group_residual(yy, gg))
                cells.append(f"{rr:+.3f}" if not np.isnan(rr) else "nan")
            except Exception:
                cells.append("err")
        w(f"| `{mk}` | " + " | ".join(cells) + " |")
    w("")

    # ---------- V5 corruption ----------
    w("## V5 corruption 场景细分")
    w("")
    corr_keys = [k for k in rows[0] if k.startswith("corr_") and k.endswith("_acc")]
    if corr_keys:
        w("| 场景 | 最强预测器 | |rho| | `acc_test` 排名 | `acc_test` rho |")
        w("|---|---|---|---|---|")
        for ck in corr_keys:
            target = np.array([r[ck] for r in rows], float)
            ranked = []
            for mk in MARKERS + ["id_acc"]:
                if mk == "id_acc":
                    xx = id_acc
                else:
                    xx = np.array([r[mk] for r in rows], float)
                _, _, rho, _ = safe_corr(xx, target)
                ranked.append((mk, rho))
            ranked.sort(key=lambda t: -abs(t[1]) if not np.isnan(t[1]) else 0)
            best, brho = ranked[0]
            acc_rho = dict(ranked)["id_acc"]
            rank = [m for m, _ in ranked].index("id_acc") + 1
            w(f"| {ck[5:-4]} | `{best}` | {abs(brho):.3f} | {rank}/{len(ranked)} | {acc_rho:+.3f} |")
    else:
        w("_（本次结果未包含 corruption 准确率）_")
    w("")

    # ---------- V6 与官方对照 ----------
    w("## V6 与官方数据的一致性对照")
    w("")
    w("同一协议、同一指标族，两套完全独立的数据：")
    w("")
    w("| 指标 | 官方 80 模型（架构残差化 r） | 自建 48 组（lr 族残差化 r） | 一致? |")
    w("|---|---|---|---|")
    # 官方数据（80 模型，架构残差化）vs 自建数据（48 组，lr 族残差化）
    OFFICIAL = {
        "id_pr_perclass": ("participation_ratio_id", 0.659),
        "id_numerical_rank": ("numerical_rank_id", None),
        "id_nc1": ("nc1_id", None),
        "id_local_smoothness": ("local_smoothness_id", None),
        "id_effective_dim_pca": ("effective_dim_pca_id", None),
        "id_class_separability": ("class_separability_id", None),
    }
    for mk in MARKERS:
        if mk not in OFFICIAL:
            continue
        official_name, o = OFFICIAL[mk]
        s = v2.get(mk, {}).get("resid_lr_r")
        if o is None:
            w(f"| `{official_name}` | — | {s:+.3f} | (官方无对应列) |")
            continue
        same = "✓ 同向" if (not np.isnan(s)) and np.sign(o) == np.sign(s) else "✗ 异向"
        w(f"| `{official_name}` | {o:+.3f} | {s:+.3f} | {same} |")
    w("")
    w("> 官方数据中的 `dimension_id`/`utility_id` 来自 GLUE 私有实现，"
      "本层不含；可比的是 participation_ratio / mean_* 系列。")
    w("")

    out_md = os.path.join(args.outdir, "ind_validation_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    out_json = os.path.join(args.outdir, "ind_validation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"V1": v1, "V2": v2, "n_models": len(rows), "n_subset": len(sub)},
                  f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_md}\n[saved] {out_json}")


if __name__ == "__main__":
    main()
